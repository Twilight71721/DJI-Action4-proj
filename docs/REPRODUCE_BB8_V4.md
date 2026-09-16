# BB8 v4 ResNet-FPN 复现指南

本文复现当前 DJI Action 4 v4 数据读取、八角点投影、Gaussian 热图生成和
ResNet18-FPN smoke training。

## 1. 进入项目目录

以下命令均从项目根目录执行：

```bash
cd /home/twilight/University/njukym/cs/cv/exam-project
```

项目使用的 Python 解释器为：

```text
.venv-hccepose/bin/python
```

检查 Python、PyTorch 和 CUDA：

```bash
.venv-hccepose/bin/python --version
.venv-hccepose/bin/python -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('cuda:', torch.version.cuda); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 2. 检查数据集

当前默认数据集：

```text
dataset/action4-v4-random-validation-100
```

BOP 图像及标签位于：

```text
dataset/action4-v4-random-validation-100/train_pbr
```

模型信息位于：

```text
dataset/action4-v4-random-validation-100/models/models_info.json
dataset/action4-v4-random-validation-100/models/corners_3d.json
```

检查关键文件是否存在：

```bash
test -f dataset/action4-v4-random-validation-100/models/models_info.json
test -f dataset/action4-v4-random-validation-100/models/corners_3d.json
test -f dataset/action4-v4-random-validation-100/corner-projections.json
test -f dataset/action4-v4-random-validation-100/train_pbr/000000/scene_gt.json
test -f dataset/action4-v4-random-validation-100/train_pbr/000000/scene_camera.json
test -f dataset/action4-v4-random-validation-100/train_pbr/000000/scene_gt_info.json
```

统计 RGB、depth 和 mask 文件数量：

```bash
find dataset/action4-v4-random-validation-100/train_pbr/000000/rgb -type f | wc -l
find dataset/action4-v4-random-validation-100/train_pbr/000000/depth -type f | wc -l
find dataset/action4-v4-random-validation-100/train_pbr/000000/mask -type f | wc -l
find dataset/action4-v4-random-validation-100/train_pbr/000000/mask_visib -type f | wc -l
```

四项预期均为 `100`。

## 3. 运行完整测试

```bash
.venv-hccepose/bin/python -m unittest discover -s tests -v
```

当前预期结果：

```text
Ran 37 tests
OK
```

测试包含：

- v4 八角点编号；
- `P_c = R @ P_m + t` 投影；
- bbox 实例裁剪及相机内参更新；
- PNG 图像读取；
- Gaussian 热图峰值；
- 与 `corner-projections.json` 的逐点投影对照；
- ResNet18-FPN 输出尺寸；
- focal loss、坐标损失和热图解码。

## 4. 生成标签预览

预览第 0 帧：

```bash
.venv-hccepose/bin/python scripts/preview_bb8_labels.py \
  --image-id 0 \
  --out-dir outputs/bb8_v4_preview
```

输出文件：

```text
outputs/bb8_v4_preview/000000_000000_00_overlay.png
outputs/bb8_v4_preview/000000_000000_00_heatmaps.png
```

预览其他帧时只需要更改 `--image-id`：

```bash
.venv-hccepose/bin/python scripts/preview_bb8_labels.py \
  --image-id 59 \
  --out-dir outputs/bb8_v4_preview
```

默认标签配置为：

```text
input size:       256x256
instance crop:    bbox_obj square crop
crop padding:     1.3
heatmap size:     64x64
heatmap kernel:   Gaussian
sigma:            2.0 heatmap pixels
pose scale:       1.0
```

## 5. 检查训练参数

```bash
.venv-hccepose/bin/python scripts/train_heatmap.py --help
```

当前默认网络和损失：

```text
backbone:                  ResNet18
feature decoder:           FPN
output stride:             4
output channels:           8
loss:                      focal + 10 * coordinate loss
final heatmap head bias:   -2.19
```

## 6. 运行 3 epoch 快速 smoke test

该命令只验证训练程序能否端到端运行：

```bash
.venv-hccepose/bin/python scripts/train_heatmap.py \
  --epochs 3 \
  --batch-size 8 \
  --freeze-epochs 0 \
  --num-workers 2 \
  --out-dir outputs/heatmap_v4_smoke_reproduce \
  --seed 460731
```

## 7. 复现 10 epoch 实验

```bash
.venv-hccepose/bin/python scripts/train_heatmap.py \
  --epochs 10 \
  --batch-size 8 \
  --freeze-epochs 0 \
  --num-workers 2 \
  --out-dir outputs/heatmap_v4_composite_reproduce \
  --seed 460731
```

该实验使用固定随机种子将 100 张数据划分为：

```text
train: 80
validation: 20
```

当前基准实验的最佳结果出现在第 8 epoch 左右：

```text
validation corner error: approximately 41.38 px
PCK@5:                  approximately 0.169
```

100 张数据只用于验证训练链路，不能将该结果解释为最终精度。

## 8. 查看训练结果

训练目录包含：

```text
best.pth
last.pth
metrics.csv
```

查看指标：

```bash
column -s, -t outputs/heatmap_v4_composite_reproduce/metrics.csv
```

如果系统没有 `column`：

```bash
sed -n '1,20p' outputs/heatmap_v4_composite_reproduce/metrics.csv
```

检查 checkpoint：

```bash
.venv-hccepose/bin/python - <<'PY'
import torch

path = "outputs/heatmap_v4_composite_reproduce/best.pth"
checkpoint = torch.load(path, map_location="cpu", weights_only=False)
print("epoch:", checkpoint["epoch"] + 1)
print("metrics:", checkpoint["metrics"])
print("arguments:", checkpoint["args"])
PY
```

## 9. 显式指定全部关键参数

为了避免默认值以后发生变化，可以使用以下完整命令：

```bash
.venv-hccepose/bin/python scripts/train_heatmap.py \
  --dataset-root dataset/action4-v4-random-validation-100/train_pbr \
  --models-info dataset/action4-v4-random-validation-100/models/models_info.json \
  --obj-id 1 \
  --pose-scale 1.0 \
  --input-size 256 256 \
  --crop-mode bbox \
  --crop-padding 1.3 \
  --heatmap-kernel gaussian \
  --heatmap-sigma 2.0 \
  --backbone resnet18 \
  --fpn-dim 128 \
  --loss composite \
  --coordinate-loss-weight 10 \
  --batch-size 8 \
  --epochs 10 \
  --freeze-epochs 0 \
  --lr 0.001 \
  --backbone-lr-scale 0.1 \
  --weight-decay 0.0001 \
  --val-fraction 0.2 \
  --num-workers 2 \
  --device cuda \
  --amp \
  --seed 460731 \
  --out-dir outputs/heatmap_v4_composite_reproduce
```

## 10. CPU 回退

没有可用 CUDA 时，可以执行：

```bash
.venv-hccepose/bin/python scripts/train_heatmap.py \
  --epochs 3 \
  --batch-size 2 \
  --freeze-epochs 0 \
  --num-workers 0 \
  --device cpu \
  --no-amp \
  --out-dir outputs/heatmap_v4_cpu_smoke \
  --seed 460731
```

## 11. 结果解释限制

- 当前数据只有一个渲染场景，80/20 是实例级划分，不是独立场景划分；
- 当前数据没有手、支架、线缆和真实遮挡；
- 当前网络输入是实例 crop，真实视频推理时需要检测框、跟踪框或人工 ROI；
- 当前相机内参来自合成数据，不能直接替代真实视频相机标定；
- 不应根据这 100 张数据开始长时间正式训练。
