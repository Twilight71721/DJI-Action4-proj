# DJI Action 4 单目姿态估计

给定一段拍到 DJI Osmo Action 4 的 RGB 视频，YOLO 先定位相机机身，ResNet-FPN 再预测机身三维包围盒的八个角点在画面中的位置。视频工具会导出带框视频和逐帧二维坐标。要从角点进一步求旋转和平移，还需要拍摄相机的标定参数。

**从这里开始**：[在 CPU 上投影八个角点](#quickstart)只需仓库自带的几何参数。有 BOP 标注数据时可以[训练角点模型](#training)；已有两套训练权重时可以[处理视频](#video)。仓库提供源码和配置，完整训练集与训练好的 Action 4 权重需另行准备。

<a id="quickstart"></a>

## 在 CPU 上投影八个角点

下面的例子读取 仓库记录的 Action 4 包围盒尺寸，假设它位于相机正前方 500 毫米，计算八个角点在一张 640 × 480 图像上的像素坐标。这个例子只需 Python 3.10、NumPy、OpenCV 和 Pillow，无需 GPU、图片或预训练权重。

```bash
git clone https://github.com/Twilight71721/DJI-Action4-proj.git
cd DJI-Action4-proj
conda create -n action4-bb8 python=3.10 -y
conda activate action4-bb8
python -m pip install numpy==1.26.4 opencv-python-headless==4.9.0.80 Pillow==11.3.0
```

在仓库根目录运行：

```bash
python - <<'PY'
import numpy as np
from bop_bb8 import BOPModel, project_corners

model = BOPModel.from_models_info("configs/action4_v4/models_info.json", obj_id=1)
K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]])
pixels, visible, _ = project_corners(
    model.corners_mm(), K, np.eye(3), [0, 0, 500], image_size=(640, 480)
)
print(f"{visible.sum()}/8 corners visible")
print(pixels.round(1))
PY
```

第一行应为 `8/8 corners visible`，随后是八行 `[u, v]` 像素坐标，第一个角点约为 `[283.6, 217.7]`。这里的位姿是已知输入；下面的训练流程让网络从 RGB 图像预测这些坐标。

<a id="training"></a>

## 用自己的 BOP 数据训练

沿用上面的环境。项目的训练依赖文件固定了 Linux 下的 PyTorch CUDA 12.8 构建。安装后先检查网络能否完成一次前向计算：

```bash
python -m pip install -r requirements.txt
python scripts/check_server_setup.py --device cpu
```

成功时会看到 `network_smoke=OK output_shape=(1, 8, 64, 64)`。输入是一张 256 × 256 的 RGB 图，输出是八张 64 × 64 的角点热图。要检查 GPU，把 `cpu` 改为 `cuda`。

将数据整理成下面的结构。示例使用对象编号 `1`、场景 `000000`。模型尺寸和位姿平移量均用毫米。RGB 帧与三份场景标注的图像编号须对应。

```text
dataset/my-action4/
├── models/models_info.json          # 每个对象的三维包围盒尺寸与原点偏移
└── train_pbr/000000/
    ├── rgb/000000.png               # 后续帧按六位数字编号
    ├── scene_camera.json            # 每帧的相机内参 cam_K
    ├── scene_gt.json                # 每个实例的 obj_id、cam_R_m2c、cam_t_m2c
    └── scene_gt_info.json           # 同一实例的二维框 bbox_obj，格式为 [x, y, w, h]
```

读取器要求 `models_info.json` 中每个对象都有 `min_x/y/z` 和 `max_x/y/z` 六个字段。如果你的 BOP 文件提供的是各轴的最小值和尺寸，可按 `max_x = min_x + size_x` 补齐三轴最大值。[Action 4 文件示例](configs/action4_v4/models_info.json)

先检查一张标签，再跑一轮训练。将第一行换成你的数据目录；如果第一张图的场景号或帧号不同，同时修改预览参数。训练和验证划分至少需要两个有效实例。

```bash
DATA_DIR="dataset/my-action4"
python scripts/preview_bb8_labels.py \
  --dataset-root "$DATA_DIR/train_pbr" --models-info "$DATA_DIR/models/models_info.json" \
  --scene-id 000000 --image-id 0 --out-dir outputs/preview

python scripts/train_heatmap.py \
  --dataset-root "$DATA_DIR/train_pbr" --models-info "$DATA_DIR/models/models_info.json" \
  --device cpu --epochs 1 --batch-size 2 --num-workers 0 --no-pretrained \
  --out-dir outputs/first-train
```

打开 `outputs/preview/000000_000000_00_overlay.png`，检查线框是否贴合机身。训练完成后，`outputs/first-train/` 中有 `best.pth`、`last.pth` 和记录验证误差的 `metrics.csv`。一轮训练用于检查流程。正式训练时可改用 `--device cuda`、增加轮数，并用 `--val-scenes` 指定独立的验证场景。重复运行请使用新的输出目录。

<a id="video"></a>

## 用已有权重处理视频

这个入口需要 NVIDIA CUDA GPU、支持 `libx264` 编码的 FFmpeg，以及两套针对 Action 4 训练的权重：

| 输入 | 要求 |
| --- | --- |
| `--yolo` | 识别 Action 4 的 Ultralytics 检测器权重（`.pt`）。[检测器训练说明](docs/YOLO_ACTION4_BASELINE.md) |
| `--resnet` | 本项目生成的八角点 checkpoint（`.pth`）。默认机身框方案使用 256 × 256 输入、`crop_mode=bbox`、`crop_padding=1.3`。 |
| `--video` | 一段能被 OpenCV 读取的 RGB 视频。 |

安装上节的训练依赖，再安装 YOLO 依赖。可用 `ffmpeg -hide_banner -encoders | grep -w libx264` 检查编码器。如果当前环境的 FFmpeg 缺少它，可给推理命令传入 `--ffmpeg /path/to/ffmpeg`，选用支持该编码器的版本。将三个输入路径换成自己的文件：

```bash
python -m pip install -r requirements-yolo.txt
python scripts/infer_camera_video.py \
  --video videos/input.mp4 --yolo weights/action4-detector.pt \
  --resnet weights/action4-corners.pth --output outputs/video-preview --limit 30
```

先打开 `outputs/video-preview/index.html` 检查前 30 帧的效果。`camera_pose_full_video.mp4` 是叠加检测框和角点的画面，`predictions.jsonl` 每帧一条预测记录。处理整段视频时去掉 `--limit 30`，并指定新的输出目录。

## 测试与更多资料

安装两份依赖后，在仓库根目录运行 `python -m unittest discover -s tests -v`。本地没有对应 BOP 数据或 Blender 的 `bpy` 模块时，相关集成测试会跳过。

- [算法与评估设计](docs/ALGORITHM.md) 介绍角点投影、热图监督和位姿误差指标。
- [单张 RGB 矩形裁剪实验](docs/SINGLE_RGB_RECT_DESIGN_20260926.md) 说明另一种输入尺寸和训练配置。
- [PnP 后处理](docs/PNP_CUBOID_POSTPROCESS_20260926.md) 说明如何用相机内参和角点估计位姿。

本项目的原创代码采用 [MIT 许可证](LICENSE)。
