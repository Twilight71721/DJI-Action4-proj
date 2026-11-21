可以。结合本地代码来看，BoxDreamer 对我们最有价值的不是完整的多视图网络，而是这条技术链：

> 固定顺序的 3D 包围盒角点 → 投影生成 8 通道热图 → ResNet/ViT 提取图像特征 → 热图预测头 → 解码 8 个二维角点 → PnP 恢复位姿

这与我们的目标高度一致。但 BoxDreamer 本身是“参考图条件下的位姿估计”，不能直接等同于我们计划的“单张 RGB 直接预测 8 个角点”。

## 1. BoxDreamer 实际在做什么

它的完整输入包含：

- 多张参考图像；
- 参考图像的已知相机位姿；
- 从参考图像重建出来的物体三维包围盒；
- 一张待估计位姿的 Query 图像。

整体流程是：

```text
参考图像 + 参考位姿
        ↓
稀疏三维重建
        ↓
确定物体 3D 包围盒的 8 个角点
        ↓
将 8 个 3D 角点投影到每张参考图
        ↓
生成参考图的 8 通道角点热图
        ↓
参考图：(RGB 特征 + 已知角点热图特征)
查询图：(RGB 特征 + 可学习的未知角点 token)
        ↓
BETR Transformer 跨视图融合
        ↓
预测 Query 图像的 8 通道热图
        ↓
提取 8 个二维角点
        ↓
PnP
        ↓
查询图像的 6D 位姿
```

对应的主入口是 [BoxDreamerModel.py](https://github.com/zju3dv/BoxDreamer/blob/e7200e06bbb5e6aa11b8d7015b4ea7783bdd6930/src/models/BoxDreamerModel.py#L112)。

模型在这里依次完成：

1. 读取图像、内参、位姿和 Query 标记；
2. 准备参考图的包围盒表示；
3. 调用 ResNet 或 DINOv2；
4. 调用 BETR；
5. 解码热图；
6. 使用 PnP 求姿态。

## 2. 8 个角点到底是怎样定义的

BoxDreamer 并不是随意编号角点，而是按照三维坐标最小值、最大值组合，固定为：

| 通道 | 三维坐标 |
|---|---|
| 0 | `(xmin, ymin, zmin)` |
| 1 | `(xmin, ymax, zmin)` |
| 2 | `(xmax, ymax, zmin)` |
| 3 | `(xmax, ymin, zmin)` |
| 4 | `(xmin, ymin, zmax)` |
| 5 | `(xmin, ymax, zmax)` |
| 6 | `(xmax, ymax, zmax)` |
| 7 | `(xmax, ymin, zmax)` |

代码在 [vis_utils.py](https://github.com/zju3dv/BoxDreamer/blob/e7200e06bbb5e6aa11b8d7015b4ea7783bdd6930/src/lightning/utils/vis/vis_utils.py#L1153)。

这个顺序极其重要。后续必须保证：

```text
热图第 i 个通道
    ↕
二维角点 keypoints_2d[i]
    ↕
三维角点 bbox_3d[i]
```

三者永远对应同一个角点。否则即使热图看起来正确，PnP 得到的姿态也会完全错误。

对 DJI Action 4，还需要先固定 GLB 的物体坐标系。例如：

- `+X`：相机右侧；
- `+Y`：相机顶部；
- `+Z`：相机正面或背面方向。

具体选哪套约定不是最重要的，重要的是渲染、标签生成、训练和 PnP 全部使用同一套坐标系。

## 3. BoxDreamer 如何生成监督标签

首先把 8 个三维角点通过相机位姿和内参投影到图像：

$$
\tilde p_i = K[R|t]\tilde P_i
$$

实现位置是 [camera_utils.py](https://github.com/zju3dv/BoxDreamer/blob/e7200e06bbb5e6aa11b8d7015b4ea7783bdd6930/src/datasets/utils/base/camera_utils.py#L62)。

然后，以每个二维角点为中心生成一个热图，最终得到：

```text
bbox_feat.shape = [B, 8, H, W]
```

实现位于 [bbox_utils.py](https://github.com/zju3dv/BoxDreamer/blob/e7200e06bbb5e6aa11b8d7015b4ea7783bdd6930/src/datasets/utils/base/bbox_utils.py#L215)。

其中：

- 第 0 通道只表示角点 0；
- 第 1 通道只表示角点 1；
- …
- 第 7 通道只表示角点 7。

数据集中的调用位置在 [base.py](https://github.com/zju3dv/BoxDreamer/blob/e7200e06bbb5e6aa11b8d7015b4ea7783bdd6930/src/datasets/base.py#L683)。

### 需要注意的实现细节

BoxDreamer 当前 `heatmap` 实现使用的是：

```python
torch.exp(-distance / scale_factor)
```

见 [bbox_utils.py](https://github.com/zju3dv/BoxDreamer/blob/e7200e06bbb5e6aa11b8d7015b4ea7783bdd6930/src/datasets/utils/base/bbox_utils.py#L292)。

它并不是最常见的标准二维高斯：

$$
\exp\left(-\frac{d^2}{2\sigma^2}\right)
$$

因此我们可以先复现它作为基线，但后续建议比较：

- BoxDreamer 当前热图；
- 标准 Gaussian heatmap；
- 固定 `sigma`；
- 根据物体投影大小动态调整 `sigma`。

## 4. BoxDreamer 的 ViT/DINOv2 部分

默认配置实际上使用的是 DINOv2 ViT-B/14：

```yaml
encoder:
  name: dino
  dino:
    cfg:
      model_type: dinov2_vitb14_reg
      freeze: True
```

见 [transformer.yaml](https://github.com/zju3dv/BoxDreamer/blob/e7200e06bbb5e6aa11b8d7015b4ea7783bdd6930/configs/model/transformer.yaml#L55)。

输入为 `224×224`，patch 大小为 14，所以：

```text
224 / 14 = 16
P = 16 × 16 = 256 patches
```

DINOv2 返回：

```text
[B, N, 256, 768]
```

其中：

- `B`：batch size；
- `N`：参考图加 Query 图数量；
- `256`：patch token 数；
- `768`：ViT-B 特征维度。

特征提取位于 [dinov2.py](https://github.com/zju3dv/BoxDreamer/blob/e7200e06bbb5e6aa11b8d7015b4ea7783bdd6930/src/models/modules/encoder/dinov2.py#L48)。

BETR 首先用一个 MLP adapter 和 LayerNorm 处理 DINO 特征：

```python
self.input_transform = Mlp(
    in_features=self.d_model,
    out_features=self.d_model,
    drop=0.1,
)
self.norm = nn.LayerNorm(self.d_model)
```

对应 [betr.py](https://github.com/zju3dv/BoxDreamer/blob/e7200e06bbb5e6aa11b8d7015b4ea7783bdd6930/src/models/modules/backbone/betr.py#L156)。

这部分很适合 ViT 方案复用：

```text
DINOv2 patch tokens
        ↓
MLP adapter + LayerNorm
        ↓
2D position encoding
        ↓
Transformer blocks
        ↓
每个 patch 输出 14×14×8
        ↓
Unpatchify
        ↓
[B, 8, 224, 224]
```

### BoxDreamer 很巧妙的输出头

每个 patch token 的维度为 768，经过线性层输出：

```python
Linear(768, 14 * 14 * 8)
```

代码在 [betr.py](https://github.com/zju3dv/BoxDreamer/blob/e7200e06bbb5e6aa11b8d7015b4ea7783bdd6930/src/models/modules/backbone/betr.py#L151)。

随后 `unpatchify` 将 256 个 patch 的输出重新拼成完整分辨率热图：

```text
[B, 256, 14×14×8]
        ↓
[B, 8, 224, 224]
```

实现位于 [betr.py](https://github.com/zju3dv/BoxDreamer/blob/e7200e06bbb5e6aa11b8d7015b4ea7783bdd6930/src/models/modules/backbone/betr.py#L230)。

这是最适合我们的 ViT 设计：结构清晰，而且不需要复杂的上采样卷积。

## 5. BoxDreamer 的 ResNet 部分存在问题

代码确实提供了 ResNet：

```python
if encoder_name == "resnet":
    self.rgb_encoder = ResNetWrapper(...)
```

见 [BoxDreamerModel.py](https://github.com/zju3dv/BoxDreamer/blob/e7200e06bbb5e6aa11b8d7015b4ea7783bdd6930/src/models/BoxDreamerModel.py#L95)。

但当前实现只获取 `layer4`：

```python
self.model.layer4.register_forward_hook(...)
```

见 [resnet.py](https://github.com/zju3dv/BoxDreamer/blob/e7200e06bbb5e6aa11b8d7015b4ea7783bdd6930/src/models/modules/encoder/resnet.py#L50)。

对于 `224×224` 输入：

```text
ResNet18 layer4 → [B, 512, 7, 7]
展开后 → [B, 49, 512]
```

而默认 BETR 期待：

```text
256 个 token
每个 token 768 维
```

因此直接把配置中的：

```yaml
encoder.name: dino
```

改成：

```yaml
encoder.name: resnet
```

很可能会同时出现两个不匹配：

1. `49` 个 ResNet token 对不上 `256` 个包围盒 patch token；
2. ResNet18 的 `512` 维对不上 BETR 的 `768` 维。

所以 ResNetWrapper 更像是一个尚未完整接通的备选实现，而不是可以直接运行的成熟分支。

## 6. 我们的 ResNet 应该怎样设计

对于单张 DJI Action 4 图像，我不建议只使用 `layer4`。原因是 `7×7` 的空间分辨率太低，不利于精确定位角点。

更合适的是：

```text
RGB crop
  ↓
ResNet18/34
  ├─ layer1: 56×56
  ├─ layer2: 28×28
  ├─ layer3: 14×14
  └─ layer4: 7×7
  ↓
FPN 或逐级反卷积/上采样
  ↓
融合成 56×56 特征
  ↓
Conv 3×3
  ↓
Conv 1×1，输出 8 通道
  ↓
8×56×56 heatmaps
```

然后再把热图坐标映射回原始输入：

$$
x_{\text{input}} = x_{\text{heatmap}}\frac{W_{\text{input}}}{W_{\text{heatmap}}}
$$

$$
y_{\text{input}} = y_{\text{heatmap}}\frac{H_{\text{input}}}{H_{\text{heatmap}}}
$$

这种设计对 ResNet 更自然，也比强行套用 ViT 的 `unpatchify` 更合适。

建议第一版使用：

- `ResNet18`；
- ImageNet 预训练；
- `layer2 + layer3 + layer4` 多尺度融合；
- 输出 `8×56×56`；
- 先冻结 backbone 训练 heatmap head；
- 收敛后解冻 `layer3/layer4` 微调。

## 7. ViT 与 ResNet 应该共享哪些部分

为了后续公平比较，两个模型应该只更换 backbone 和 decoder，其他部分完全一致：

| 模块 | ResNet | ViT |
|---|---|---|
| 输入 | 同一张裁剪 RGB | 同一张裁剪 RGB |
| 角点顺序 | 相同 | 相同 |
| 标签 | 8 通道热图 | 8 通道热图 |
| 损失函数 | 相同 | 相同 |
| 解码方法 | 相同 | 相同 |
| PnP | 相同 | 相同 |
| 数据划分 | 相同 | 相同 |
| 评价指标 | 相同 | 相同 |

模型接口可以统一为：

```python
heatmaps = model(images)
# images:   [B, 3, H, W]
# heatmaps: [B, 8, Hh, Wh]
```

这样训练、验证、角点解码和 PnP 都不需要为两个模型各写一套。

## 8. 热图解码部分是否应该直接复用

BoxDreamer 对每个热图取最高的 20 个位置，然后直接平均：

```python
max_vals, idxs = torch.topk(heatmaps, k=20, dim=2)
kpt_x_mean = xs.float().mean(dim=2)
kpt_y_mean = ys.float().mean(dim=2)
```

见 [box_utils.py](https://github.com/zju3dv/BoxDreamer/blob/e7200e06bbb5e6aa11b8d7015b4ea7783bdd6930/src/models/utils/box_utils.py#L75)。

这可以作为第一个 baseline，但它有一个明显问题：没有利用 `max_vals` 对坐标加权。如果热图有两个峰，直接平均可能落在两个峰中间。

更建议后续比较：

1. `argmax`；
2. BoxDreamer 的 `top-20 mean`；
3. `top-k weighted mean`；
4. 局部 soft-argmax。

例如加权版本：

$$
x_i =
\frac{\sum_{j \in TopK}p_jx_j}
{\sum_{j \in TopK}p_j}
$$

这通常会比无权重平均更稳定。

## 9. PnP 部分需要谨慎复用

PnP 实现位于 [box_utils.py](https://github.com/zju3dv/BoxDreamer/blob/e7200e06bbb5e6aa11b8d7015b4ea7783bdd6930/src/models/utils/box_utils.py#L113)。

它先调用：

```python
cv2.solvePnPRansac(...)
```

但紧接着又写了：

```python
success = False
```

见 [box_utils.py](https://github.com/zju3dv/BoxDreamer/blob/e7200e06bbb5e6aa11b8d7015b4ea7783bdd6930/src/models/utils/box_utils.py#L168)。

这会导致 RANSAC 结果被强制丢弃，最终始终退化为普通的：

```python
cv2.solvePnP(..., flags=cv2.SOLVEPNP_ITERATIVE)
```

因此我们不应该原样复制这一行。我们的版本应当：

- 正常保留 `solvePnPRansac` 结果；
- 检查内点数量；
- 失败时再回退到 `solvePnP`；
- 计算重投影误差；
- 角点置信度太低时拒绝输出姿态。

## 10. 损失函数怎么选

BoxDreamer 默认实际上使用 `SmoothL1Loss`，而不是 focal loss，配置在 [default.yaml](https://github.com/zju3dv/BoxDreamer/blob/e7200e06bbb5e6aa11b8d7015b4ea7783bdd6930/configs/model/loss/default.yaml#L2)。

预测输出经过：

```python
sigmoid → [0,1]
2*x-1  → [-1,1]
```

见 [betr.py](https://github.com/zju3dv/BoxDreamer/blob/e7200e06bbb5e6aa11b8d7015b4ea7783bdd6930/src/models/modules/backbone/betr.py#L405)。

因此它比较的是：

```text
predicted heatmap ∈ [-1,1]
ground-truth heatmap ∈ [-1,1]
```

建议我们的第一版也保持一致，先验证流程正确：

```text
8-channel Gaussian heatmap
+ sigmoid
+ SmoothL1Loss
```

之后再实验 CornerNet focal loss。仓库虽然实现了 focal loss，但默认配置并没有启用，代码在 [focal_loss.py](https://github.com/zju3dv/BoxDreamer/blob/e7200e06bbb5e6aa11b8d7015b4ea7783bdd6930/src/loss/utils/focal_loss.py#L6)。

## 11. 对我们项目的最终建议

我们不需要照搬 BoxDreamer 的这些部分：

- 多张参考图像输入；
- 稀疏三维重建；
- Query heatmap token masking；
- 跨视图 `N×P` Transformer；
- LoFTR、CoTracker 等匹配模块。

因为我们已经有 DJI Action 4 的 GLB，并且 HCCEPose 可以直接渲染大量带真实位姿的训练图像。

我们应该重点复用：

- 3D 包围盒角点顺序；
- 3D 到 2D 的投影；
- 8 通道角点热图标签；
- ViT 的 patch token 输出方式；
- ResNet/ViT 统一输出接口；
- 热图解码；
- PnP 位姿恢复；
- 重投影误差评价。

推荐开发顺序是：

```text
第一阶段：验证数据和几何
GLB角点 → Blender位姿/K → 投影二维角点 → 在渲染图上画出来

第二阶段：ResNet baseline
RGB → ResNet18+FPN → 8热图 → 角点 → PnP

第三阶段：ViT baseline
RGB → DINOv2 patch tokens → Transformer/MLP head → 8热图 → PnP

第四阶段：比较
角点误差、重投影误差、ADD/ADD-S、旋转误差、平移误差
```

当前最优先的不是马上写网络，而是先确认“DJI Action 4 的 8 个三维角点及其投影”完全正确。只要这个标签链条出现角点顺序、坐标轴或裁剪内参错误，ResNet 和 ViT 都不可能学出正确姿态。