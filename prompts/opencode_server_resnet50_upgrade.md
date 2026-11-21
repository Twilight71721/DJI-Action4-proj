# OpenCode 任务：将现有 BB8 热图网络升级为 ResNet50-FPN 并在修正数据上验证

你正在一台配有 NVIDIA RTX 4090 24GB 的 Linux 服务器上工作。现有项目此前已经部署并训练过 ResNet18-FPN，但旧的 10000 张数据曾存在标签尺度错误；现在用户已经修正该问题。本任务是以最小、可审计的改动加入 ResNet50-FPN，并在确认无误的修正数据上完成验证和训练准备。

不要假设服务器目录、数据集名称或当前 Git 状态。先检查实际环境，再行动。

## 一、必须遵守的边界

1. 最终测试视频及其任何帧都禁止用于训练、验证、伪标签、自训练、阈值选择、ROI 参数选择或 early stopping。任何名为 `head_left_rgb_raw...mp4` 的文件均视为最终测试集，不得读取、抽帧或分析。
2. 不得继续使用已知标签尺度错误的旧 `dataset/train_pbr`，除非审计证明它已经被原地修正；不能仅凭目录名判断。
3. 不要自动执行 `git commit` 或 `git push`。
4. 不要覆盖旧 checkpoint、旧 metrics 或旧评估输出。新输出必须使用独立目录。
5. 不要删除文件，不要重置用户已有改动，不要格式化整个项目。
6. 不要引入新的大型依赖；优先使用现有 PyTorch、torchvision、NumPy、OpenCV 和项目代码。
7. 如果修正后的数据集路径存在歧义、缺少 `models_info.json`、缺少 scene，或几何审计失败，停止训练并报告，不得自行选择“看起来像”的数据。

## 二、先读取和检查

先执行只读检查，并报告结果：

1. `pwd`、`git status --short`、当前分支和远端。
2. `nvidia-smi`、Python、PyTorch、torchvision、CUDA版本。
3. 阅读仓库中的 `AGENTS.md`（如果存在），然后阅读：
   - `README.md`
   - `docs/SERVER_TRAINING.md`
   - `docs/REPRODUCE_BB8_V4.md`
   - `bop_bb8/net/resnet_fpn.py`
   - `bop_bb8/net/loss.py`
   - `bop_bb8/net/decode.py`
   - `bop_bb8/torch_data.py`
   - `bop_bb8/dataset.py`
   - `scripts/train_heatmap.py`
   - `scripts/evaluate_heatmap_checkpoint.py`
   - `tests/test_net.py`
   - `tests/test_bop_bb8.py`
4. 枚举服务器上的候选BOP数据集，报告每个候选的scene数、RGB数、`models_info.json`位置、修改时间和磁盘占用。
5. 用户应将修正数据路径提供为 `[CORRECTED_DATASET_DIR]`。如果该占位符未替换，不要猜测；先根据实际目录列出候选并要求用户确认。

## 三、训练前几何审计（硬性门槛）

对确认的修正数据集执行完整审计：

1. RGB、depth、mask、`mask_visib`、`scene_gt.json`、`scene_camera.json`、`scene_gt_info.json`数量和索引一致。
2. `models_info.json`中的模型尺寸、单位和八角点定义明确。
3. `cam_t_m2c`单位与模型单位一致。
4. 使用数据中的 `K/R/t`重新投影8个三维角点。
5. 检查投影框与`bbox_obj`、mask范围的宽高比例；不得再次出现角点集中在物体中心的问题。
6. 检查`mask_visib`是`mask`子集，且没有空mask。
7. 随机抽取至少32张，生成带角点编号和三维线框的叠加图；另外单独选择目标较小、遮挡较强、靠近边缘和极端视角样本。
8. 把审计统计和可视化保存到新的输出目录，例如：

```text
outputs/resnet50_data_audit/
```

如果任何几何关系明显错误，立即停止，不得开始训练。报告错误样本、证据和可能原因。

## 四、ResNet50-FPN的最小代码改造

### 1. 网络定义

在现有`ResNetFPN`实现中增加ResNet50，不另写一套不兼容网络。

在`bop_bb8/net/resnet_fpn.py`中：

- 导入`resnet50`与当前torchvision真实存在的ResNet50预训练权重枚举；
- 优先使用`ResNet50_Weights.IMAGENET1K_V2`，但必须先确认服务器版本支持；如果不支持，明确回退到该版本实际提供的官方权重，不得编造名称；
- ResNet50的stage通道必须是：

```python
(256, 512, 1024, 2048)
```

- 保留`layer1..layer4`的stride `4/8/16/32`多尺度融合；
- 保留输出stride 4；
- 输出必须是raw logits，不能在`forward()`里调用`sigmoid`；
- 保持`freeze_backbone()`和冻结时BatchNorm保持eval的行为；
- 不改变ResNet18/34的默认行为和checkpoint兼容性。

目标接口：

```python
model = ResNetFPN(
    backbone="resnet50",
    fpn_dim=256,
    num_corners=8,
    pretrained=True,
)

images: [B, 3, 336, 336]
logits: [B, 8, 84, 84]
```

第一版不要同时加入新的注意力模块、可见性头或复杂decoder。先建立可解释的ResNet50-FPN基线。

### 2. 训练入口

在`scripts/train_heatmap.py`中：

- 将`resnet50`加入`--backbone`合法选项；
- 保留默认backbone为ResNet18，避免破坏旧命令；
- 确保`--input-size 336 336`自动得到`84×84`热图；
- 新增或确认支持独立的test scene排除机制。test scene不得参与梯度和best checkpoint选择；
- checkpoint必须保存完整的backbone、FPN维度、输入尺寸、热图尺寸、crop参数、scene划分和损失参数；
- 支持从`last.pth`安全恢复训练；恢复时验证模型配置一致；
- 增加可选的gradient clipping，默认本任务使用`1.0`；
- 增加warmup + cosine decay，或在不破坏旧行为的情况下提供明确开关；
- 每轮记录train loss、val loss、mean error、median/P90（如果实现成本合理）、PCK@5、PCK@10、学习率、耗时和峰值显存；
- best checkpoint主要按validation mean corner error选择，同时记录PCK。

### 3. 数据划分

必须按scene划分，禁止把同一scene中的图片随机分到train/validation/test。

如果用户没有提供固定划分：

- 先列出scene；
- 创建并保存一个`split.json`；
- 数据足够时建议约70% train、20% validation、10% synthetic test；
- 划分一旦生成便固定，ResNet18、ResNet50和未来ViT必须共享；
- 不要把synthetic test用于选择checkpoint。

### 4. 测试

扩展`tests/test_net.py`，至少覆盖：

1. ResNet50在`336×336`输入下输出`[B,8,84,84]`。
2. 输出是有限的raw logits。
3. 冻结backbone后所有backbone参数无梯度且BatchNorm保持eval。
4. 解冻后backbone参数恢复训练。
5. FPN和head参数能够收到非零梯度。
6. ResNet18/34现有测试仍然通过。
7. ResNet50 checkpoint保存后能够严格重新加载。
8. train/validation/test scene不重叠。

## 五、固定训练配置

第一版ResNet50使用以下配置，不要未经验证自行扩大模型：

```text
backbone                 = resnet50
pretrained               = true
input_size               = 336 336
heatmap_size             = 84 84
fpn_dim                  = 256
crop_mode                = bbox
crop_padding             = 1.5
heatmap_kernel           = gaussian
heatmap_sigma            = 2.0
loss                     = composite
focal_alpha              = 2.0
focal_beta               = 4.0
coordinate_loss_weight   = 10.0
optimizer                = AdamW
decoder_lr               = 1e-3
backbone_lr              = 1e-4
weight_decay             = 1e-4
batch_size               = 64
AMP                      = true
freeze_epochs            = 3
gradient_clip            = 1.0
epochs                   = 60
```

如果batch 64在实际完整管线上OOM，先报告峰值显存，再降到32；不要静默改变配置。不要为了占满4090显存盲目把batch扩大到128。

第一版不添加会改变角点编号的水平翻转。任何几何增强都必须同步更新角点、热图和裁剪内参；如果当前代码没有完整支持，就先不加入几何增强。可以在基线稳定后再加入不改变几何标签的光度增强。

## 六、分阶段验证，禁止直接跑60轮正式训练

### 阶段A：代码和单元测试

1. 运行Python语法检查。
2. 运行完整测试集。
3. 用随机`[2,3,336,336]`输入执行forward/backward。
4. 报告参数量、输出形状、峰值显存。

### 阶段B：32样本过拟合诊断

使用固定32个训练样本，关闭随机增强，验证网络能够明显过拟合正确标签。必须生成预测角点与真值角点叠加图。

如果现有训练入口不能真正实现“同一固定32样本的过拟合诊断”，可以增加一个最小的诊断选项或脚本，但不要复制整套训练代码。

通过条件：

- loss稳定下降；
- 角点预测趋近真值；
- 没有全背景塌缩；
- 热图峰值与角点编号一致；
- 可视化没有系统性尺度或顺序错误。

失败则停止并分析，不得继续pilot。

### 阶段C：pilot训练

从训练scene中使用约500～1000个样本运行10～20 epoch，验证：

- train/validation指标正常下降；
- checkpoint保存和恢复有效；
- 遮挡、边缘和极端视角样本没有系统性失败；
- 评估脚本可以输出随机样本和最差样本叠加图；
- 显存和每轮耗时可接受。

### 阶段D：正式训练

只有A/B/C全部通过后，才运行完整60 epoch。新输出目录建议：

```text
outputs/action4_corrected_resnet50_336_fpn256/
```

不得覆盖旧的`outputs/action4_v5_resnet18`或其他历史结果。

## 七、公平对照

为了判断收益是否来自ResNet50，而不是分辨率或FPN宽度，使用相同修正数据和相同scene划分补跑公平基线：

```text
ResNet18-FPN:
  input=336×336
  fpn_dim=256

ResNet50-FPN:
  input=336×336
  fpn_dim=256
```

其他训练参数、损失、解码、crop和epoch保持一致。报告：

- mean/median/P90 corner error；
- PCK@5、PCK@10；
- 按可见比例、目标尺寸、视角和边缘截断分组的结果；
- 参数量、峰值显存、训练耗时和推理吞吐量。

不要使用最终测试视频比较模型。

## 八、最终汇报格式

完成后必须汇报：

1. 实际使用的修正数据集绝对路径与scene划分。
2. 数据几何审计结果和32张叠加图位置。
3. 修改的文件列表及每项修改原因。
4. ResNet50总参数量和可训练参数量。
5. 测试命令与完整测试结果。
6. 32样本过拟合结果。
7. pilot训练指标、峰值显存、耗时和最差样本。
8. 是否满足正式训练条件。
9. 若已运行正式训练，给出best/last checkpoint、metrics.csv和评估报告路径。
10. 所有尚未解决的风险或不确定点。

不要只报告“训练成功”或最终loss；必须提供角点叠加图、验证指标和数据几何证据。
