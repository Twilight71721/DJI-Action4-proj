# DJI Action 4 单目姿态估计

从 DJI Action 4 的单张 RGB 图像预测三维包围盒的八个投影角点。项目包含 BOP 数据读取、ResNet-FPN 热图训练、YOLO 检测、PnP 位姿估计和视频评估。

## 环境与验证

在仓库根目录执行。训练依赖按 Python 3.10 和 PyTorch CUDA 12.8 配置，完整测试还会导入 YOLO 模块。

```bash
conda create -n action4-bb8 python=3.10
conda activate action4-bb8
python -m pip install -r requirements.txt -r requirements-yolo.txt
python scripts/check_server_setup.py --device cpu
python -m unittest discover -s tests -v
```

## 数据与代码入口

- `bop_bb8/` 包含相机几何、裁剪、热图模型和解码逻辑。`configs/` 保存实验参数和角点定义。
- `scripts/train_heatmap.py` 训练八角点模型，`scripts/infer_camera_video.py` 处理视频。两者均可通过 `--help` 查看输入要求。
- 数据归档 `exam-project-dataset.tar.zst` 包含 `dataset/` 和旧版 `dataset.zip`。要恢复训练数据，在仓库根目录执行下方命令。部分 YOLO 划分使用旧项目路径的绝对符号链接，解包后使用这些划分前需将链接指向当前 `dataset/` 内的对应目标。
- 历史运行的权重和报告由各自的实验流程生成。渲染还需要本地模型、参考图和独立的 Blender 工程。将该工程放在 `third_party/Blender/`，当前配套版本为 `3d6776b`。[Blender 工程](https://github.com/Twilight71721/Blender) · [三维建模说明](docs/3DM-README.md) · [算法与评估设计](docs/ALGORITHM.md)

```bash
tar --zstd -xf "/media/twilight/My Passport/exam-project-dataset.tar.zst" dataset/
```

## 服务器训练

服务器端部署与训练请先阅读：

- `docs/SERVER_TRAINING.md`
- `skills/server-bb8-training/SKILL.md`
- `prompts/deepseek_v4_1_flash_server_training.md`

生成数据集、checkpoint、虚拟环境和预训练权重不存放在普通 Git 仓库中。

2026-09-26 新增可选的[单张 RGB 紧长方形输入方案](docs/SINGLE_RGB_RECT_DESIGN_20260926.md)：
无 RGB 灰边，按长宽比组成 batch，默认 ResNet18-FPN，保留固定 YOLO；已通过短流程验证，尚未正式训练。
通过 `scripts/train_single_rgb.py` 显式启动，以下历史基线的默认行为保持不变。

### 当前可复现的 ResNet18-FPN 基线

现已支持 `--backbone resnet50`（ImageNet V2 权重、Bottleneck 的 256/512/1024/2048 通道接入 FPN）。
新增 `--overfit` 明确使用全部相同实例训练与评估；可结合 `--cache-data --cosine-lr`。
历史过拟合实验使用了 200 张新版腕部场景图像。
本次为 400 个标注框裁剪、3200 个角点的训练集拟合检查，不代表检测或真实视频泛化精度。

当前训练默认使用：

```text
dataset/action4-v4-random-validation-100/train_pbr
```

这是 100 帧的管线验证集，不是最终规模的正式训练集。数据读取器根据每个实例的
`bbox_obj` 加 1.3 倍上下文，裁成方形后缩放到 `256x256`；相机内参同步更新。
网络输出 stride-4 的 `8x64x64` Gaussian 热图。角点编号与数据中的
`models/corners_3d.json` 完全一致：

```text
0 (xmin,ymin,zmin)  1 (xmax,ymin,zmin)
2 (xmax,ymax,zmin)  3 (xmin,ymax,zmin)
4 (xmin,ymin,zmax)  5 (xmax,ymin,zmax)
6 (xmax,ymax,zmax)  7 (xmin,ymax,zmax)
```

恢复上述验证集后，可运行标签预览和短程训练。

```bash
python scripts/preview_bb8_labels.py --image-id 0
python scripts/train_heatmap.py \
  --epochs 10 --batch-size 8 --freeze-epochs 0 \
  --out-dir outputs/heatmap_v4_composite_smoke --seed 460731
```

训练损失默认为热图 focal loss 加可微坐标 Smooth-L1。单独的 focal loss 在仅 100 张
数据时容易先学成全背景；最终热图层使用 `bias=-2.19`，令初始前景概率约为 0.1。
详细审计和实测结果见 `docs/bop_bb8_v4_pipeline.md`。

## 许可证

本项目的原创代码采用 [MIT 许可证](LICENSE)。
