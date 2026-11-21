# Action 4 八角点热图任务：ViT 建议训练方案

## 1. 训练目标

训练 `DINOv2 ViT-B/14-reg + 多层特征融合热图解码器`，从 `336x336` 的单物体 ROI 预测 `[8,84,84]` raw logits，并在修正后的 BOP 合成数据上完成可审计的训练、验证和独立 synthetic test。

训练是否成功不能只看 loss。至少要同时证明：

- 标签几何和角点编号正确；
- 32 样本能够过拟合；
- validation 二维角点误差持续改善；
- 没有全背景塌缩或通道混淆；
- synthetic test 上 PnP 位姿指标合理；
- 与 ResNet50 的比较使用完全相同的数据划分和输入输出条件。

最终测试视频及其帧不属于训练数据，也不属于 validation。模型、检测器、crop padding、解码参数、PnP 阈值和时序参数全部冻结后，才允许对它运行一次最终测试。

## 2. 训练前硬性门槛

### 2.1 环境记录

训练报告必须记录：

```text
git commit / dirty status
Python / PyTorch / torchvision / CUDA / cuDNN versions
GPU model and driver
DINOv2 source, repo revision and weight identity
dataset absolute path and modification time
command line and random seed
```

不得自动 `git commit` 或 `git push`。预训练权重和大 checkpoint 不应提交到普通 Git 仓库。

### 2.2 修正数据的几何审计

不要根据目录名假设数据已经修好。对选定数据集至少检查：

1. RGB、depth、mask、`mask_visib`、`scene_gt.json`、`scene_camera.json`、`scene_gt_info.json` 的数量和索引一致。
2. `models_info.json` 的尺寸、单位与 `cam_t_m2c` 一致。
3. 用 `K/R/t` 重投影 8 个三维角点，确认投影框与 `bbox_obj`、mask 的位置和尺度一致。
4. 检查固定角点顺序，尤其是前后面和左右边没有交换。
5. 随机抽取至少 32 张叠加编号角点和三维线框；额外查看小目标、强遮挡、边缘截断、背面和极端视角。
6. 保存审计统计和图片，不通过则停止训练。

### 2.3 数据划分

必须按 scene 划分，禁止把同一 scene 的相邻或同分布样本随机拆到不同集合。建议：

```text
train            about 70%
validation       about 20%
synthetic test   about 10%
```

具体比例可按 scene 数调整，但必须满足：

- 三个集合无 scene 交集；
- `split.json` 一旦生成就固定；
- validation 用于 checkpoint 和超参数选择；
- synthetic test 只在模型冻结后评估；
- ResNet18、ResNet50、ViT-S 消融和 ViT-B 使用同一个 `split.json`；
- 最终真实测试视频不出现在任何列表中。

## 3. 第一版固定配置

```text
model_name                  = dinov2_vitb14_reg
pretrained                  = true
input_size                  = 336 336
patch_size                  = 14
intermediate_layers         = 2 5 8 11
decoder_dim                 = 256
heatmap_size                = 84 84
num_corners                 = 8
dropout                     = 0.1

crop_mode                   = bbox
crop_padding                = 1.5
heatmap_kernel              = gaussian
heatmap_sigma               = 2.0

loss                        = composite
focal_alpha                 = 2.0
focal_beta                  = 4.0
coordinate_loss_weight      = 10.0
decode                      = top-k weighted
decode_topk                 = 9

optimizer                   = AdamW
decoder_weight_decay        = 1e-4
backbone_weight_decay       = 0.05
gradient_clip_norm          = 1.0
precision                   = bf16 preferred, fp16 fallback
effective_batch_size        = 32
seed                        = 460731
max_epochs                  = 60
best_metric                 = validation mean corner error in input pixels
```

`heatmap_sigma=2.0` 对应热图坐标，约等于输入坐标 8 px。它首先与 ResNet 公平对齐，不应在第一次训练中一边换主干一边重新搜索 sigma。

## 4. RTX 4090 的 batch 与显存策略

推荐起点：

```text
physical batch size     = 8
gradient accumulation   = 4
effective batch size    = 32
AMP                     = bf16 if stable
gradient checkpointing  = off initially
```

实际显存取决于 DINOv2 版本、是否保存四层中间激活、数据增强和 PyTorch 实现，不能只按参数量推断。必须先做真实 forward/backward 显存探测，并报告 `torch.cuda.max_memory_allocated()` 和 `max_memory_reserved()`。

调节顺序：

1. batch 8 正常：保持，使用 accumulation 4；
2. OOM：先降到 batch 4、accumulation 8；
3. 仍 OOM：再启用 Transformer block gradient checkpointing；
4. 不要降低输入到 224 来掩盖显存问题，因为这会把 patch 网格从 `24x24` 降到 `16x16`，改变定位能力和公平比较条件；
5. 也不要为了占满显存盲目扩大 batch。小数据下过大的 batch 可能削弱泛化。

若使用 fp16，启用 `GradScaler`；若使用 bf16，通常不需要 scaler，但应按当前 PyTorch 实现验证。出现 NaN/Inf 时先定位具体 loss、logit 和梯度，不要直接静默跳过 batch。

## 5. 优化器参数组

至少分为 decoder 和 backbone 两大组，并对 backbone 做 no-decay 排除：

```text
decoder conv weights:         configured decoder LR, weight_decay=1e-4
decoder norm/bias:            configured decoder LR, weight_decay=0
backbone matrix/conv weights: configured backbone LR, weight_decay=0.05
backbone norm/bias/tokens/
position embeddings:          configured backbone LR, weight_decay=0
```

下列参数通常不做 weight decay：

```text
all bias
LayerNorm / GroupNorm scale and bias
cls_token
register_tokens
pos_embed
```

实现后打印每个参数组的参数数量、学习率和 weight decay，并测试所有 `requires_grad=True` 参数恰好出现一次，避免遗漏或重复。

## 6. 三阶段训练

### 阶段 A：只训练解码器，epoch 1～5

```text
trainable:        projection, fusion, upsampling decoder, heatmap head
frozen:           entire DINOv2 backbone
backbone mode:    eval
decoder LR:       1e-3
backbone LR:      0
epochs:           5
```

目的：先让随机初始化的热图解码器适应预训练特征，避免第一批大梯度破坏主干。冻结主干时要保持其 eval，避免 DropPath 让固定特征随机变化。

### 阶段 B：解冻最后 4 个 block，epoch 6～40

```text
trainable backbone: blocks 8, 9, 10, 11 and final norm
frozen backbone:    patch embed, tokens, pos embed, blocks 0..7
decoder LR peak:    3e-4
backbone LR peak:   2e-5
epochs:             35
warmup:             first 2 epochs of this stage
scheduler:          cosine decay
```

这里的 `blocks 8..11` 是 Python 零基索引。冻结部分保持 eval，可训练 block 使用 train。阶段 B 是默认主训练阶段；如果 validation 已稳定且没有欠拟合，不必强行进入全量微调。

### 阶段 C：条件式全量微调，epoch 41～60

只有在以下证据同时存在时才进入：

- Stage B 的 train 与 validation 误差都停在较高水平，二者差距不大，表现更像欠拟合而非过拟合；
- 后四层已收敛，降低 decoder 学习率也不能继续改善 validation；
- 标签、数据划分、解码和优化器均已排除错误；
- 最近若干 epoch 没有明显过拟合恶化；
- 一个短程全量解冻试验确实改善 validation，而不只是降低 train error。

“validation 误差高于 ResNet50”本身不能证明需要全量解冻；如果 train 已明显优于 validation，那更像过拟合，应保持部分冻结并改善数据多样性。

建议配置：

```text
trainable:                full backbone + decoder
decoder LR peak:          1e-4
last backbone block LR:   1e-5
layer-wise LR decay:      0.8 from deep to shallow
epochs:                   at most 20
warmup:                   1 epoch
scheduler:                cosine decay
```

如果进入阶段 C 后 validation mean/P90 error 连续恶化，应保留阶段 B 的 best checkpoint，而不是以最后一轮覆盖。若实现 layer-wise decay 会显著复杂化第一版，可先对全主干使用统一 `5e-6`，但必须记录这一偏差，不能假装已经做了 LLRD。

## 7. 学习率调度与恢复训练

推荐每个阶段拥有明确的 warmup + cosine 区间，不要在解冻时突然把新参数从 0 跳到峰值。scheduler 应按 optimizer update step 而不是原始 mini-batch step 计数，以正确处理梯度累积。

checkpoint 至少保存：

```text
model state
optimizer state
scheduler state
AMP scaler state when applicable
epoch and global update step
best metric and best epoch
model architecture/config
input/heatmap/crop/loss/decode config
train/validation/synthetic-test scene lists
DINOv2 model/source/weight identity
random states for Python/NumPy/PyTorch/CUDA
```

resume 时先比较关键配置。模型名、层索引、输入尺寸、热图尺寸、角点数或 split 不一致时应拒绝恢复，除非用户明确启动迁移实验。

## 8. 数据增强

### 8.1 第一版允许的增强

优先加入不改变角点几何的光度增强，并保持强度适中：

```text
ColorJitter          p=0.8, brightness/contrast/saturation around 0.2, hue around 0.05
gamma jitter         p=0.3, gamma around [0.8, 1.2]
Gaussian noise       p=0.3, small sigma
Gaussian/motion blur p=0.2, mild kernel
JPEG degradation     p=0.2, moderate quality
```

具体实现应在 ImageNet normalization 之前，并确保像素范围正确。validation 和 test 不使用随机增强。

训练图像由 Blender 生成时，更重要的是渲染阶段的背景、灯光、曝光、材质、距离、视角、运动模糊和遮挡随机化。不要用最终测试视频提取背景或颜色统计。

### 8.2 暂不启用的几何增强

第一版不使用：

- 水平/垂直翻转；
- 任意旋转；
- perspective/affine；
- 没有同步更新角点、热图和 `K_input` 的随机 crop。

水平翻转不仅改变二维坐标，还会改变固定 3D 角点的语义对应，不能只翻图和热图。未来可加入 detector-box jitter 来模拟真实 ROI 误差，但必须通过同一 crop 仿射同步更新 `keypoints_2d` 和 `K_input`，并用单元测试验证。

## 9. 分阶段验证流程

### Gate 0：语法、单元测试和显存探测

执行：

1. Python 语法检查；
2. 完整单元测试；
3. fake backbone 的 CPU forward/backward；
4. 官方预训练 backbone 的 GPU forward/backward；
5. 分别测试全冻结、最后四层、全量解冻；
6. 输出参数量、可训练参数量、shape、吞吐量和峰值显存。

任何 token/shape、NaN、梯度或重载问题未解决前不得训练。

### Gate 1：固定 32 样本过拟合

使用来自 train scenes 的固定 32 个样本：

```text
random augmentation = off
shuffle             = fixed/reproducible
validation          = these same 32 samples for diagnosis only
training length     = enough to clearly overfit, not tied to formal 60 epochs
```

建议先冻结 backbone；若 decoder 无法过拟合，再解冻最后四层进行诊断。通过条件：

- loss 显著且稳定下降；
- mean corner error 接近几个输入像素，或至少持续趋近真值；
- PCK@5 明显升高；
- 8 个通道没有全背景塌缩；
- 预测峰与对应编号真值一致；
- 32 张预测/真值叠加图无统一尺度、偏移或角点顺序错误。

这里的数值门槛不能代替可视化。若失败，应先检查标签、valid mask、raw logits、坐标尺度和参数组，而不是直接正式训练。

### Gate 2：500～1000 样本 pilot

从 train scenes 固定选样，运行约 10～20 epoch：

- 验证 train/validation loss 和角点指标能正常改善；
- 验证冻结到部分解冻的状态切换；
- 验证 checkpoint 保存、resume 和严格加载；
- 输出随机样本、最差样本、小目标、遮挡和边缘样本可视化；
- 记录每轮耗时、吞吐和显存；
- 检查 peak probability，识别全背景或过度自信。

### Gate 3：正式训练

只有 Gate 0～2 全部通过后，才运行最多 60 epoch。建议新输出目录：

```text
outputs/action4_corrected_dinov2_vitb14_reg_336/
```

不得覆盖 ResNet 或旧错误标签训练所得 checkpoint。

### Gate 4：独立 synthetic test

选定 best checkpoint、冻结所有超参数后，只运行一次 synthetic test。该结果用于报告，不再回头选择 checkpoint。若根据 test 结果修改模型，必须将其视为新的实验，并重新建立未使用的 test 集。

## 10. 每轮指标与模型选择

训练和 validation 至少记录：

```text
total loss
focal loss component
coordinate loss component
mean corner error in 336x336 input pixels
median / P90 / P95 corner error
PCK@2 / PCK@5 / PCK@10
per-corner mean error for channels 0..7
peak probability mean and low-percentile
learning rate for every parameter group
trainable parameter count
epoch time / samples per second
peak allocated / reserved VRAM
```

best checkpoint 的主指标为 validation mean corner error，P90 和 PCK@5 作为重要辅助判断。若 mean 稍好但 P90 显著恶化，应检查失败样本，不要只按一个小数机械选模型。

建议按下列属性分组报告：

- visible fraction；
- 目标在 crop 中的像素尺寸；
- 遮挡强度；
- 是否靠近 crop 边缘或被截断；
- 正面、侧面、背面和极端俯仰视角；
- 每个角点通道。

## 11. PnP 级评估

二维角点误差是首要网络指标，但最终目标是位姿。validation 和 synthetic test 有真实 `K/R/t` 时，还应使用相同 PnP 配置报告：

```text
PnP success rate
rotation error in degrees
translation error in dataset units and relative percentage
ADD / ADD-S where symmetry definition requires
ADD-0.1d success rate
2D reprojection error
Proj2D@5px
RANSAC inlier count when using RANSAC
```

不能把 PnP 自身的重投影残差当作角点正确性的充分证据，因为一组系统性错误角点也可能被某个位姿拟合。应同时保留对真实二维角点和真实位姿的误差。

## 12. 与 ResNet50 的公平对照

公平比较固定：

```text
same corrected dataset
same split.json
same 336x336 ROI
same 84x84 Gaussian heatmaps
same crop padding 1.5
same corner order and valid mask
same composite loss
same top-k decoder
same best-checkpoint metric
same synthetic test protocol
```

分别报告：

- 参数量和可训练参数量；
- 峰值显存、训练耗时、推理吞吐；
- mean/median/P90/P95 corner error；
- PCK@5/PCK@10；
- PnP 位姿指标；
- 小目标、遮挡、边缘截断和极端视角的分组指标。

不能用最终视频的视觉效果反复挑选 ViT 或 ResNet；这会把测试集变成调参集。

## 13. 过拟合判断和处理顺序

典型过拟合信号：train error 继续下降，而 validation mean/P90 连续恶化；热图在合成图上很尖锐，在 validation 困难样本上错误且过度自信。

处理顺序：

1. 回到 best checkpoint，不使用 last checkpoint；
2. 不进入全量微调，保持只解冻最后四层；
3. 增强合成渲染的背景、光照、材质、模糊和遮挡多样性；
4. 增加合规光度增强；
5. 降低 backbone LR 或缩短训练；
6. 适当提高 decoder dropout（例如 0.1 到 0.2）；
7. 最后才考虑退回 ViT-S/14-reg。

不要优先通过增大 weight decay 或无限堆增强掩盖标签/划分错误。

## 14. 推荐执行顺序

```text
1. 环境与数据只读审计
2. 固定 split.json
3. 实现 ViT adapter + decoder + model factory
4. 离线单元测试
5. 官方权重加载与 shape/梯度/显存测试
6. 固定 32 样本过拟合
7. 500～1000 样本 pilot
8. 正式训练阶段 A
9. 正式训练阶段 B
10. 根据 validation 证据决定是否执行阶段 C
11. 冻结配置并评估 synthetic test
12. 与同条件 ResNet50 汇总比较
13. 冻结完整推理管线后，最终视频只运行一次
```

所有阶段都应使用新的输出目录并保留配置、命令、指标和叠加图，使任何结论都能追溯。
