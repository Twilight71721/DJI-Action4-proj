# Action 4 八角点热图任务：ViT 架构设计方案

## 1. 设计目标与边界

本方案用于从单个 DJI Osmo Action 4 的 RGB ROI 中预测固定顺序的 8 个三维包围盒角点热图。它只替换现有 ResNet-FPN 的特征提取与热图头，不改变标签定义和下游几何流程：

```text
完整 RGB 帧
  -> 检测/跟踪得到单个 Action 4 ROI
  -> 带上下文的方形 crop，并同步更新二维角点和 K_input
  -> DINOv2 ViT-B/14-reg
  -> 四层 Transformer patch 特征融合
  -> 卷积热图解码器
  -> 8 通道 84x84 raw logits
  -> sigmoid + top-k 加权解码
  -> crop 内 8 个二维角点
  -> 映射回完整视频帧
  -> PnP / RANSAC / 时序平滑
```

第一版的目标是得到一个可复现、可与 ResNet50-FPN 公平比较的 ViT 基线，而不是一次加入所有可能模块。第一版不加入 Transformer decoder、cross-attention、可见性分支、时序 Transformer、可微 PnP 或 ViT-L。

最终测试视频 `head_left_rgb_raw...mp4` 及其任何帧不得用于训练、验证、伪标签、超参数选择、阈值调整、ROI 参数选择或 early stopping。它只允许在模型和参数完全冻结后进行一次最终测试。

## 2. 为什么选择 DINOv2 ViT-B/14 with registers

主模型选择：

```text
official model name = dinov2_vitb14_reg
patch size          = 14
embedding dimension = 768
Transformer blocks  = 12
attention heads     = 12
register tokens     = 4
backbone parameters = about 86M
```

选择理由：

1. 当前任务是合成数据训练、真实视频测试，域差距明显。DINOv2 的自监督预训练特征比随机初始化 ViT 更适合这种小规模下游任务。
2. ViT-B 的容量明显高于 ResNet50，但 24GB RTX 4090 仍可通过小 batch、梯度累积和分阶段解冻训练。
3. `/14` 在 `336x336` 输入下得到 `24x24` patch 网格；既能保持全局关系，又能通过轻量解码器恢复到 `84x84` 热图。
4. register 版本含 4 个额外 token。它们不参与二维网格重排，但可以作为预训练时的内部信息槽；实现时必须显式确认它们已被官方接口移除。
5. ViT-S/14 可作为显存或训练速度受限时的备选消融模型，但不作为本方案主模型。ViT-L/14 暂不采用，因为当前更大的风险是合成到真实的域差距、标签质量和局部定位精度，而不是模型容量不足。

官方模型卡确认 ViT-B/14 的特征维度为 768、patch size 为 14，并提供 `dinov2_vitb14_reg`；官方实现的 `get_intermediate_layers(..., reshape=True)` 会在重排前移除 CLS 和 register tokens。实现仍必须针对服务器实际使用的 DINOv2 版本做形状测试：

- <https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md>
- <https://github.com/facebookresearch/dinov2/blob/main/dinov2/models/vision_transformer.py>

## 3. 输入、输出和不变量

### 3.1 输入

```text
tensor shape = [B, 3, 336, 336]
color order  = RGB
dtype        = float32 before autocast
normalization:
  mean = [0.485, 0.456, 0.406]
  std  = [0.229, 0.224, 0.225]
```

`336` 能被 patch size `14` 整除：

```text
336 / 14 = 24
```

因此 patch token 的空间网格是 `24x24`，每张图有 `576` 个 patch token。另有 1 个 CLS token 和 4 个 register tokens，但它们不能被误当作图像 patch。

第一版仅支持高宽都能被 14 整除的输入。构造模型或 forward 时应主动检查，而不是依赖 DINOv2 静默裁掉边缘像素。

### 3.2 输出

```text
logits shape = [B, 8, 84, 84]
logits type  = raw logits
stride       = 4 relative to 336x336 input
```

网络 `forward()` 不能调用 `sigmoid()`。现有 `focal_loss()` 和 `coordinate_loss()` 接收 raw logits；只有解码和可视化时才调用 `sigmoid()`。

8 个通道的角点顺序必须保持：

```text
0: (xmin, ymin, zmin)    1: (xmax, ymin, zmin)
2: (xmax, ymax, zmin)    3: (xmin, ymax, zmin)
4: (xmin, ymin, zmax)    5: (xmax, ymin, zmax)
6: (xmax, ymax, zmax)    7: (xmin, ymax, zmax)
```

## 4. 主干网络和中间层特征

### 4.1 层选择

从 12 个 Transformer block 中抽取第 3、6、9、12 个 block 的输出。以 Python 零基索引表示为：

```python
intermediate_layers = (2, 5, 8, 11)
```

建议通过官方接口获取：

```python
features = backbone.get_intermediate_layers(
    images,
    n=[2, 5, 8, 11],
    reshape=True,
    return_class_token=False,
    norm=True,
)
```

预期返回 4 个张量，每个形状都是：

```text
[B, 768, 24, 24]
```

这里的四层不是 ResNet 那种不同空间分辨率的金字塔，而是相同 `24x24` 网格上的不同语义深度：浅层偏局部纹理和边缘，深层偏全局形状和语义。不要把它错误描述成 stride 4/8/16/32 的 FPN。

### 4.2 DINOv2 适配器

建议将预训练模型加载和任务网络分开：

```text
DINOv2BackboneLoader
  - 负责官方模型名称、来源、缓存和预训练权重
  - 记录 repo revision / 权重文件 / hash（能获取时）
  - 失败时明确报错，不得静默改成随机初始化

DINOv2Heatmap
  - 接收一个满足 get_intermediate_layers 接口的 backbone
  - 负责特征投影、融合、上采样和热图输出
  - 单元测试可注入 FakeDINOBackbone，不依赖网络下载
```

这样做能避免单元测试每次访问 GitHub，同时防止模型代码和权重来源不可追踪。

加载优先级建议为：

1. 用户明确提供的本地 DINOv2 repo 和本地权重；
2. 服务器已缓存且来源明确的官方 `torch.hub` 模型；
3. 经用户允许后下载官方模型，并固定缓存；
4. 不允许在预训练加载失败时静默使用随机权重。

## 5. 特征投影与融合

每个中间层先独立投影到 256 通道：

```text
four times:
  Conv2d(768, 256, kernel_size=1, bias=False)
  GroupNorm(32, 256)
  GELU

each output: [B, 256, 24, 24]
```

使用 GroupNorm 而不是 BatchNorm，因为 ViT 全量微调时 batch 通常只有 8 左右。随后沿通道维拼接：

```text
concat -> [B, 1024, 24, 24]
```

融合模块：

```text
Conv2d(1024, 256, kernel_size=1, bias=False)
GroupNorm(32, 256)
GELU
Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
GroupNorm(32, 256)
GELU
```

融合后得到：

```text
[B, 256, 24, 24]
```

第一版使用 concat 而不是直接相加，因为不同深度特征的分布和语义不同；`1x1` 卷积让网络学习每层的组合权重。后续可把简单相加作为轻量消融，但不应在第一版同时维护两条复杂路径。

## 6. 热图解码器

### 6.1 空间恢复

由于 `24 -> 84` 不是整数倍，采用显式目标尺寸，而不是固定 `scale_factor`：

```text
[B, 256, 24, 24]
  -> bilinear interpolate(size=(42, 42), align_corners=False)
  -> Conv2d(256, 256, 3, padding=1, bias=False)
  -> GroupNorm(32, 256)
  -> GELU

[B, 256, 42, 42]
  -> bilinear interpolate(size=(84, 84), align_corners=False)
  -> Conv2d(256, 128, 3, padding=1, bias=False)
  -> GroupNorm(32, 128)
  -> GELU

[B, 128, 84, 84]
```

显式指定 `(42, 42)` 和 `(84, 84)` 可以避免浮点 scale factor 或奇数尺寸造成 off-by-one。第一版不采用反卷积，以减少棋盘伪影和额外的输出尺寸推导。

### 6.2 热图头

```text
Conv2d(128, 128, kernel_size=3, padding=1)
GELU
Dropout2d(p=0.1)
Conv2d(128, 8, kernel_size=1)
```

最后一层 bias 初始化为 `-2.19`：

```text
sigmoid(-2.19) approximately 0.10
```

稀疏角点热图应从偏背景的概率开始，而不是每个像素都从 0.5 开始。其余卷积采用 PyTorch 默认初始化或统一的 Kaiming 初始化；不要改动预训练主干参数的初始化。

### 6.3 端到端张量形状

| 阶段 | 张量形状 |
|---|---:|
| 输入 ROI | `[B, 3, 336, 336]` |
| patch embedding | `[B, 576, 768]`（不含特殊 token） |
| block 3/6/9/12 空间特征 | 4 × `[B, 768, 24, 24]` |
| 各层投影 | 4 × `[B, 256, 24, 24]` |
| concat | `[B, 1024, 24, 24]` |
| 融合 | `[B, 256, 24, 24]` |
| 第一次上采样 | `[B, 256, 42, 42]` |
| 第二次上采样 | `[B, 128, 84, 84]` |
| 8 角点 logits | `[B, 8, 84, 84]` |

解码器预计约 2.5M～3M 参数，总模型预计约 89M 参数。实际参数量必须由实现后脚本统计，不能把估算值当成验收结果。

## 7. 建议类接口

接口名称可根据服务器现有代码风格微调，但应保持以下能力：

```python
class DINOv2Heatmap(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        model_name: str = "dinov2_vitb14_reg",
        intermediate_layers: tuple[int, ...] = (2, 5, 8, 11),
        embed_dim: int = 768,
        decoder_dim: int = 256,
        num_corners: int = 8,
        input_size: tuple[int, int] = (336, 336),
        heatmap_size: tuple[int, int] = (84, 84),
        dropout: float = 0.1,
    ) -> None: ...

    def forward(self, images: torch.Tensor) -> torch.Tensor: ...

    def freeze_backbone(self, freeze: bool = True) -> None: ...

    def set_trainable_backbone_blocks(self, last_n: int | None) -> None: ...
```

行为约束：

- `last_n=0`：冻结全部主干，主干保持 eval，避免 DropPath 引入随机特征；
- `last_n=4`：只训练最后 4 个 block 和最终 norm，patch embedding、特殊 token、位置编码和前 8 个 block 冻结；
- `last_n=None`：全主干可训练；
- 部分解冻时，冻结的 block 应保持 eval，可训练 block 使用 train；
- decoder 始终跟随整体模型的 train/eval 状态；
- 参数分组不能依赖脆弱的字符串包含判断，至少要有测试确认每组参数数量和 `requires_grad` 状态。

## 8. 与现有项目的集成

建议新增和修改：

```text
new:
  bop_bb8/net/vit_heatmap.py

possibly new:
  bop_bb8/net/model_factory.py

modify:
  bop_bb8/net/__init__.py
  scripts/train_heatmap.py
  scripts/evaluate_heatmap_checkpoint.py
  tests/test_net.py
```

模型工厂应能从 checkpoint 配置重建 ResNet 或 ViT，避免评估脚本继续硬编码 `ResNetFPN`。旧 ResNet checkpoint 必须仍能加载；旧 CLI 默认值不得被悄悄改变。

共享部分保持不变：

- `HeatmapDataset` 和 ImageNet normalization；
- 8 角点顺序和 valid mask；
- Gaussian target，第一版 `sigma=2.0`；
- `focal_loss + 10 * coordinate_loss`；
- `sigmoid + decode_topk(k=9)`；
- crop 内坐标与完整帧坐标映射；
- PnP 和时序模块。

## 9. 必须通过的模型级测试

1. 注入 fake backbone，在无网络条件下构造模型。
2. 输入 `[2, 3, 336, 336]`，输出严格为 `[2, 8, 84, 84]`。
3. 输入高或宽不能被 14 整除时明确报错。
4. `get_intermediate_layers` 返回数量、通道和网格不符时明确报错。
5. 确认 CLS 和 4 个 register tokens 没有进入 `24x24` 网格。
6. 输出全部有限，且是 raw logits。
7. 最后一层 bias 初值为 `-2.19`。
8. decoder forward/backward 后收到非零且有限的梯度。
9. 全冻结时主干无梯度且保持 eval；最后 4 block 解冻时只有目标 block 有梯度；全解冻时主干能收到梯度。
10. 保存 checkpoint 后可用模型工厂严格重载，输出形状一致。
11. 现有 ResNet18/34/50、损失和解码测试继续通过。

## 10. 第一版明确不做的内容

- 不用最终测试视频做任何训练或模型选择；
- 不从随机初始化训练 ViT-B；
- 不将 CLS/register token reshape 成二维特征；
- 不把四个中间层误当成不同分辨率的 FPN；
- 不在 `forward()` 中 sigmoid；
- 不直接用最后一层 `24x24` 特征插值后接一个 `1x1` 头；
- 不一开始就全量解冻 86M 参数；
- 不加入会破坏固定角点编号的水平翻转；
- 不引入第二套标签、损失、解码或 PnP 实现；
- 不在第一版同时实现 ViT-S、ViT-L、时序模型和多任务头。

## 11. 主要风险与应对

### 风险 A：局部定位不如 ResNet-FPN

ViT 的原始网格只有 `24x24`，卷积上采样不能凭空恢复所有细节。通过四层特征融合、`84x84` 热图、坐标损失和较大的 ROI 输入缓解；最终必须以公平验证结果判断，而不能因模型更大就预设它一定更好。

### 风险 B：合成到真实的域差距

更大的 ViT 可能记住 Blender 风格。优先使用 DINOv2 预训练、冻结或只解冻后四层、使用光度和背景域随机化，并以独立合成 validation 选择 checkpoint。最终视频仍不可用于调参。

### 风险 C：特殊 token 或位置编码处理错误

只通过经验证的 `get_intermediate_layers(..., reshape=True)` 获取空间特征，并对 token 数与 `24x24` 网格做单元测试。不得手写 `tokens[:, 1:]` 后直接 reshape，因为 register 版本还有 4 个 register tokens。

### 风险 D：预训练来源不可复现

checkpoint 中保存模型名、DINOv2 repo 来源/revision、权重路径或 hash、输入归一化和层索引。下载失败时停止，禁止静默随机初始化。

### 风险 E：模型容量导致过拟合

采用分阶段解冻、较小 backbone LR、backbone weight decay、固定 scene split 和 validation 指标监控。训练方案详见 `docs/ViT建议训练方案.md`。
