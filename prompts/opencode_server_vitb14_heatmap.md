# OpenCode 任务：实现 DINOv2 ViT-B/14-reg 八角点热图网络并完成分阶段验证

你正在一台配有 NVIDIA RTX 4090 24GB 的 Linux 服务器上修改现有 Action 4 BB8 pose estimation 项目。项目目前已有 ResNet-FPN、BOP 数据读取、8 通道热图标签、损失、解码和评估代码；服务器版本可能已经包含 ResNet50 改造。你的任务是以最小、可审计、向后兼容的方式加入 `DINOv2 ViT-B/14 with registers` 热图模型，并完成代码验证、32 样本过拟合和 pilot 准备。

架构和训练要求以仓库内以下文档为准：

```text
docs/ViT架构设计方案.md
docs/ViT建议训练方案.md
```

不要假设服务器目录、数据集名称、Git 状态、DINOv2 安装方式或 ResNet50 改造状态。必须先检查实际环境，再行动。

## 一、不可违反的边界

1. 最终测试视频及其任何帧禁止用于训练、validation、伪标签、自训练、超参数选择、模型选择、阈值调整、crop/ROI 参数选择、PnP 参数选择或 early stopping。任何名为 `head_left_rgb_raw...mp4` 的文件均视为最终测试集；本任务不得读取、抽帧或分析它。
2. 不得使用已知曾有标签尺度错误的旧数据，除非完整几何审计证明当前文件已修正。不能根据目录名或用户口头描述跳过审计。
3. 修正数据路径应由用户以 `[CORRECTED_DATASET_DIR]` 提供。若占位符未替换，先列出候选及证据并要求确认，不要自行选择。
4. 不要自动执行 `git commit` 或 `git push`。
5. 不要覆盖已有 checkpoint、metrics、审计图或评估输出；创建新的输出目录。
6. 不要删除文件、重置用户改动、清理未跟踪文件或格式化整个项目。
7. 不要在预训练权重加载失败时静默退回随机初始化。
8. 不要直接开始 60 epoch 正式训练。必须按“单元测试 -> 32 样本过拟合 -> pilot -> 正式训练”的门槛推进。
9. 不要让单元测试依赖联网下载 DINOv2 代码或权重。
10. 如果数据路径、模型权重来源、scene split 或标签几何存在歧义，停止相应阶段并报告，不要猜测。

## 二、先执行只读检查并汇报

先检查：

1. `pwd`、`git status --short`、当前分支和远端。
2. 仓库中的 `AGENTS.md`（若存在）和所有适用项目约束。
3. GPU、driver、Python、PyTorch、torchvision、CUDA、cuDNN 版本。
4. 阅读至少以下文件；若服务器已有新增模型工厂或 ResNet50 代码，也一并阅读：

```text
README.md
docs/ViT架构设计方案.md
docs/ViT建议训练方案.md
docs/SERVER_TRAINING.md
docs/REPRODUCE_BB8_V4.md
bop_bb8/net/__init__.py
bop_bb8/net/resnet_fpn.py
bop_bb8/net/loss.py
bop_bb8/net/decode.py
bop_bb8/torch_data.py
bop_bb8/dataset.py
scripts/train_heatmap.py
scripts/evaluate_heatmap_checkpoint.py
tests/test_net.py
tests/test_bop_bb8.py
requirements.txt
```

5. 枚举候选修正 BOP 数据集，报告绝对路径、scene 数、RGB 数、`models_info.json`、修改时间和磁盘占用。
6. 检查服务器是否已有官方 DINOv2 repo、torch.hub cache 和 ViT-B/14-reg 权重。报告来源、revision/文件路径/hash（能获得时）和许可信息；不要在此阶段下载或修改环境。
7. 检查当前训练脚本是否已经支持 ResNet50、scene-level train/validation/synthetic-test split、resume、scheduler、gradient clipping 和模型工厂。不要把已有功能重复实现成第二套。

先给出简短现状和拟修改文件列表，再开始修改。

## 三、训练前数据几何审计

对用户确认的修正数据执行：

1. 检查 RGB、depth、mask、`mask_visib` 和三个 scene JSON 的数量、索引与实例数。
2. 核对模型尺寸、单位、`cam_t_m2c` 和角点定义。
3. 用数据中的 `K/R/t` 重投影固定顺序的 8 个三维角点。
4. 统计投影角点框与 `bbox_obj`、mask 框的宽高和面积比，排除角点集中在物体中心的旧尺度错误。
5. 检查 `mask_visib` 是 mask 子集且无异常空 mask。
6. 随机至少 32 张生成带编号角点和三维线框的叠加图；额外覆盖小目标、遮挡、边缘和极端视角。
7. 将统计、样本列表和图片保存到全新目录，例如：

```text
outputs/vit_data_audit/
```

几何审计失败则停止，不得进行过拟合或正式训练。

## 四、目标网络实现

### 4.1 主干

使用官方预训练模型：

```text
model_name          = dinov2_vitb14_reg
patch_size          = 14
embed_dim           = 768
blocks              = 12
heads               = 12
register_tokens     = 4
intermediate_layers = [2, 5, 8, 11]  # zero-based
```

生产代码优先使用官方模型真实提供的：

```python
backbone.get_intermediate_layers(
    images,
    n=[2, 5, 8, 11],
    reshape=True,
    return_class_token=False,
    norm=True,
)
```

但必须先在服务器实际版本上验证方法签名和返回值，不得仅照抄。预期四层均为 `[B,768,24,24]`。官方接口应移除 CLS 和 4 个 register tokens；必须用测试确认，不能手写 `tokens[:,1:]` 后 reshape。

输入必须是 `[B,3,336,336]`、RGB、现有 ImageNet normalization。高宽不能被 14 整除时主动报错，不依赖主干静默裁剪。

### 4.2 主干加载与可测试性

把“加载官方 DINOv2”和“任务网络”解耦：

- `DINOv2Heatmap` 能接收一个满足 `get_intermediate_layers` 的 backbone；
- 单元测试注入 `FakeDINOBackbone`，不得联网；
- 生产 loader 支持明确的本地 repo + 权重，或明确的官方 torch.hub cache；
- 记录 DINOv2 source、repo revision、model name、权重路径/hash；
- 预训练失败时抛出清晰错误，不得静默改用随机初始化；
- 不要把整份 DINOv2 源码复制进本项目。

不要为了方便随意新增 `timm` 等大型依赖。如果服务器已明确使用且项目接受，可以报告后再复用；否则优先官方 DINOv2 loader。

### 4.3 多层投影和融合

四个中间层分别：

```text
Conv2d(768, 256, 1, bias=False)
GroupNorm(32, 256)
GELU
```

得到 4 个 `[B,256,24,24]`，沿通道 concat 为 `[B,1024,24,24]`，然后：

```text
Conv2d(1024, 256, 1, bias=False)
GroupNorm(32, 256)
GELU
Conv2d(256, 256, 3, padding=1, bias=False)
GroupNorm(32, 256)
GELU
```

不要把四层称为不同 stride 的 FPN；它们空间尺寸相同，只是 Transformer 深度不同。

### 4.4 热图解码器

严格使用目标尺寸：

```text
[B,256,24,24]
 -> bilinear interpolate to 42x42, align_corners=False
 -> Conv2d(256,256,3,padding=1,bias=False) + GN(32) + GELU
 -> bilinear interpolate to 84x84, align_corners=False
 -> Conv2d(256,128,3,padding=1,bias=False) + GN(32) + GELU
 -> Conv2d(128,128,3,padding=1) + GELU + Dropout2d(0.1)
 -> Conv2d(128,8,1)
```

最终输出 `[B,8,84,84]` raw logits。禁止在 `forward()` 中调用 sigmoid。最后卷积 bias 初始化为 `-2.19`。

### 4.5 冻结和部分解冻

实现并测试：

```text
last_n=0:    全主干冻结并保持 eval
last_n=4:    只训练 blocks[8:12] 和 final norm；其余主干冻结并保持 eval
last_n=None: 全主干训练
```

部分解冻时不能因为外部 `model.train()` 就让冻结 block 的 DropPath 重新随机。decoder 正常跟随 train/eval。

### 4.6 模型工厂和向后兼容

建议新增：

```text
bop_bb8/net/vit_heatmap.py
bop_bb8/net/model_factory.py  # 仅在现有项目没有等价工厂时
```

修改：

```text
bop_bb8/net/__init__.py
scripts/train_heatmap.py
scripts/evaluate_heatmap_checkpoint.py
tests/test_net.py
```

评估脚本必须根据 checkpoint 配置重建 ResNet 或 ViT，不能继续硬编码 `ResNetFPN`。保留旧 ResNet CLI 默认行为和旧 checkpoint 兼容性。不要复制训练循环或损失/解码代码形成 ViT 专用的第二套不可维护流程。

## 五、训练入口要求

在不破坏旧命令的前提下加入 ViT 配置。CLI 具体名字遵循项目现有风格，但至少能表达：

```text
model family/name
DINOv2 source/local repo/weights
intermediate layers
decoder dim and dropout
freeze mode / last trainable blocks
physical batch and accumulation steps
bf16/fp16 selection
gradient clip
resume checkpoint
warmup + cosine schedule
fixed split.json
synthetic test scene exclusion
```

checkpoint 保存完整模型配置、DINOv2 身份、输入/热图/crop/损失/解码配置、scene split、optimizer、scheduler、scaler、global update step、best metric 和 RNG state。resume 时校验关键配置一致。

训练日志至少增加：focal 和 coordinate 分量、mean/median/P90/P95、PCK@2/5/10、每角点误差、peak probability、各参数组 LR、可训练参数量、吞吐、耗时和峰值显存。

## 六、固定第一版配置

```text
model_name                  = dinov2_vitb14_reg
pretrained                  = true
input_size                  = 336 336
intermediate_layers         = 2 5 8 11
decoder_dim                 = 256
heatmap_size                = 84 84
dropout                     = 0.1

crop_mode                   = bbox
crop_padding                = 1.5
heatmap_kernel              = gaussian
heatmap_sigma               = 2.0
loss                        = composite
focal_alpha                 = 2.0
focal_beta                  = 4.0
coordinate_loss_weight      = 10.0
decode_topk                 = 9

optimizer                   = AdamW
backbone_weight_decay       = 0.05
decoder_weight_decay        = 1e-4
gradient_clip_norm          = 1.0
physical_batch_size         = 8
gradient_accumulation       = 4
effective_batch_size        = 32
precision                   = bf16 preferred, fp16 fallback
max_epochs                  = 60
seed                        = 460731
```

若 batch 8 OOM，报告实际峰值显存后改为 batch 4 + accumulation 8；仍 OOM 才启用 block gradient checkpointing。不得静默把输入改成 224。

## 七、优化器和分阶段训练

正确建立 decoder/backbone decay/no-decay 参数组。bias、LayerNorm、GroupNorm、`cls_token`、register tokens 和 position embedding 不做 weight decay。打印并验证每组参数数量，确保所有可训练参数恰好出现一次。

阶段配置：

```text
Stage A, epochs 1-5:
  backbone fully frozen/eval
  decoder lr = 1e-3

Stage B, epochs 6-40:
  train blocks[8:12] + final norm
  decoder peak lr = 3e-4
  backbone peak lr = 2e-5
  2-epoch warmup + cosine

Stage C, epochs 41-60, conditional only:
  full fine-tune only when train and validation both plateau high,
  their gap is small, and a short trial improves validation
  decoder peak lr = 1e-4
  last block lr = 1e-5
  layer-wise lr decay = 0.8 preferred
  1-epoch warmup + cosine
```

scheduler 按 optimizer update step 计数，正确处理 gradient accumulation。Stage C 不是强制；若未执行，清楚报告依据。

## 八、增强约束

第一版可加入轻度、可配置的光度增强，且只用于 train：ColorJitter、gamma、轻微 noise、blur、JPEG degradation。增强必须在 ImageNet normalization 之前。

第一版禁止直接加入水平/垂直翻转、任意 rotation、perspective 或未同步更新二维角点和 `K_input` 的 crop。未来 detector-box jitter 只有在图像、角点和内参同步变换且测试覆盖后才能启用。

## 九、测试要求

至少新增：

1. fake backbone 离线构造，不触发 torch.hub 网络访问。
2. `[2,3,336,336] -> [2,8,84,84]`。
3. 非 14 整除输入明确报错。
4. 中间层数量、通道或空间 shape 错误明确报错。
5. CLS/register token 不进入 patch 网格；若 fake 返回 token 形式，也必须明确验证 adapter 行为或拒绝不合接口输出。
6. raw logits 有限，最后 bias 为 `-2.19`。
7. decoder forward/backward 梯度非零且有限。
8. 全冻结、最后四层和全解冻的 `requires_grad`、train/eval 与梯度传播正确。
9. optimizer 参数不遗漏、不重复，decay/no-decay 正确。
10. checkpoint 严格保存/重载，并由模型工厂恢复 ViT。
11. 旧 ResNet checkpoint/CLI/测试保持兼容。
12. train/validation/synthetic-test scene 无交集。
13. gradient accumulation 的 update 次数和 scheduler step 正确。

## 十、执行门槛

### Gate 0：代码验证

1. 语法检查。
2. 完整单元测试。
3. fake backbone CPU forward/backward。
4. 官方权重 GPU forward/backward。
5. 三种冻结状态测试。
6. 报告总参数量、可训练参数量、输出 shape、峰值显存和吞吐。

失败则修复，不进入 Gate 1。

### Gate 1：固定 32 样本过拟合

只从 train scenes 固定选择 32 个样本，关闭随机增强，保存样本索引。先测试 frozen backbone；必要时再解冻最后四层。生成全部 32 张预测/真值角点编号和线框叠加图。

通过条件：loss 和角点误差明显下降、PCK@5 提升、无全背景塌缩、通道编号正确、无统一尺度/偏移问题。失败必须先分析标签、valid mask、坐标尺度、raw logits、参数组和梯度，不得继续 pilot。

### Gate 2：pilot

从 train scenes 固定 500～1000 样本，训练约 10～20 epoch。验证冻结切换、metrics、可视化、checkpoint resume、显存和耗时。输出随机、最差、小目标、遮挡、边缘和极端视角样本。

### Gate 3：正式训练

只有 Gate 0～2 全部通过并向用户报告后，才准备或执行正式训练。若本次时间不足以完成正式训练，停在可复现的 pilot 结果，不要把未验证的长任务放到后台后声称完成。

建议输出目录：

```text
outputs/action4_corrected_dinov2_vitb14_reg_336/
```

### Gate 4：synthetic test

只有 best checkpoint 和所有参数冻结后，才在独立 synthetic test scenes 上运行一次。test 结果不再用于回头选择 checkpoint。

## 十一、ResNet50 公平对照

ViT 与 ResNet50 必须共享：修正数据、`split.json`、`336x336` 输入、`84x84` 热图、crop padding 1.5、损失、top-k 解码和 synthetic test。报告：

```text
parameter count / trainable count
peak VRAM / epoch time / inference throughput
mean / median / P90 / P95 corner error
PCK@5 / PCK@10
per-corner and difficult-case breakdown
PnP success, rotation, translation, ADD(-S), reprojection metrics
```

禁止通过查看最终真实测试视频反复挑选 ViT 或 ResNet。

## 十二、最终汇报格式

完成当前可执行阶段后，必须汇报：

1. 实际环境、Git 状态和适用约束。
2. 修正数据集绝对路径、scene 数和固定 split。
3. 数据几何审计统计与叠加图路径。
4. DINOv2 代码/权重来源、revision/path/hash 和加载方式。
5. 修改文件列表及每项修改原因；说明是否影响旧 ResNet。
6. 网络各阶段实际 shape、总参数量和各冻结阶段可训练参数量。
7. 测试命令和完整结果。
8. batch/precision/冻结状态下的峰值显存和吞吐。
9. 32 样本过拟合曲线、指标和 32 张叠加图位置。
10. pilot 的配置、train/validation 指标、最差样本、resume 验证、耗时和显存。
11. 是否满足正式训练条件；若未满足，明确阻塞点和证据。
12. 若执行正式训练，给出 best/last checkpoint、metrics、配置、日志和评估报告路径。
13. 与 ResNet50 的公平对照（若已完成）。
14. 所有尚未解决的风险及与文档方案的偏差。

不要只报告“代码写完”“训练成功”或最终 loss。没有标签审计、角点叠加图、validation 指标、严格重载和冻结状态证据，不能视为完成。
