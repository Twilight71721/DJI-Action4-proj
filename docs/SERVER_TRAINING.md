# Ubuntu 22.04 + RTX 4090 训练指南

本文用于在服务器上复现 DJI Action 4 BB8 八角点热图训练。服务器已知配置：

- Ubuntu 22.04；
- NVIDIA GeForce RTX 4090 24GB；
- NVIDIA driver `590.48.01`；
- `nvidia-smi` 报告 CUDA capability `13.1`。

`nvidia-smi` 中的 CUDA 13.1 表示驱动最高支持能力，不要求安装 CUDA 13.1 Toolkit。
本项目使用 PyTorch 2.8.0 的 CUDA 12.8 wheel，由 wheel 携带所需用户态 CUDA 库。

## 1. 克隆私有仓库

```text
git@github.com:Twilight71721/DJI-Action4-proj.git
```

如果服务器还没有 GitHub SSH key：

```bash
test -f ~/.ssh/id_ed25519.pub || ssh-keygen -t ed25519 -C "aitest-server"
cat ~/.ssh/id_ed25519.pub
```

只把打印出的 `.pub` 公钥添加到 GitHub，不要发送或复制私钥。添加公钥后执行：

```bash
ssh -T git@github.com
cd /home/aitest/exam-project
git clone git@github.com:Twilight71721/DJI-Action4-proj.git
cd DJI-Action4-proj
```

## 2. 创建 Conda 环境

以下步骤假设服务器已有 Conda。不要安装系统 CUDA Toolkit，也不要修改 NVIDIA 驱动。

```bash
conda create -n action4-bb8 python=3.10 -y
conda activate action4-bb8
python -m pip install --upgrade pip setuptools wheel
python -m pip install --timeout 120 --retries 10 -r requirements.txt
```

验证 GPU 和网络：

```bash
nvidia-smi
python scripts/check_server_setup.py
```

预期至少看到 `cuda_available=True`、RTX 4090 和
`network_smoke=OK output_shape=(1, 8, 64, 64)`。

## 3. 数据集不在 Git 仓库中

Git 仓库只保存代码和小型几何元数据。正式 BOP 数据集必须单独传到服务器，例如：

```text
dataset/action4-v5-train-5000/
├── models/
│   ├── models_info.json
│   └── corners_3d.json
└── train_pbr/
    ├── 000000/
    ├── 000001/
    └── ...
```

从 Windows PowerShell 传输的示例（将占位符换成真实服务器地址）：

```powershell
scp -r C:\exam-project\Blender-repo\dataset\action4-v5-train-5000 `
  aitest@SERVER_HOST:/home/aitest/exam-project/DJI-Action4-proj/dataset/
```

也可以先压缩再通过 ToDesk 文件传输。传输完成后必须比较压缩包 SHA-256。

## 4. 验证数据与测试

```bash
DATASET_DIR="dataset/action4-v5-train-5000"
python scripts/check_server_setup.py \
  --dataset-root "$DATASET_DIR/train_pbr" \
  --models-info "$DATASET_DIR/models/models_info.json"
python -m unittest discover -s tests -v
find "$DATASET_DIR/train_pbr" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort
```

## 5. 先运行 smoke training

```bash
python scripts/train_heatmap.py \
  --dataset-root "$DATASET_DIR/train_pbr" \
  --models-info "$DATASET_DIR/models/models_info.json" \
  --obj-id 1 --pose-scale 1.0 \
  --input-size 256 256 --crop-mode bbox --crop-padding 1.3 \
  --heatmap-kernel gaussian --heatmap-sigma 2.0 \
  --backbone resnet18 --loss composite --coordinate-loss-weight 10 \
  --batch-size 8 --epochs 1 --freeze-epochs 0 --num-workers 2 --limit 32 \
  --device cuda --amp --seed 460731 \
  --out-dir outputs/server_smoke
```

确认 `metrics.csv` 中的 loss 为有限值，并生成 `best.pth`、`last.pth`。

## 6. 正式训练

优先按 scene 划分训练集和验证集。下面假设 `000008`、`000009` 是预留验证 scene；
运行前必须以实际目录为准。

```bash
tmux new -s action4-train
conda activate action4-bb8
cd /home/aitest/exam-project/DJI-Action4-proj
DATASET_DIR="dataset/action4-v5-train-5000"
python scripts/train_heatmap.py \
  --dataset-root "$DATASET_DIR/train_pbr" \
  --models-info "$DATASET_DIR/models/models_info.json" \
  --obj-id 1 --pose-scale 1.0 \
  --input-size 256 256 --crop-mode bbox --crop-padding 1.3 \
  --heatmap-kernel gaussian --heatmap-sigma 2.0 \
  --backbone resnet18 --fpn-dim 128 \
  --loss composite --coordinate-loss-weight 10 \
  --batch-size 32 --epochs 50 --freeze-epochs 5 \
  --lr 0.001 --backbone-lr-scale 0.1 --weight-decay 0.0001 \
  --val-scenes 000008 000009 --num-workers 8 \
  --device cuda --amp --seed 460731 \
  --out-dir outputs/action4_v5_resnet18 2>&1 | tee outputs/action4_v5_resnet18.log
```

按 `Ctrl+B`、再按 `D` 退出 tmux 而不中断训练。重新进入：

```bash
tmux attach -t action4-train
```

如果出现 CUDA OOM，先把 batch size 依次降为 `16`、`8`，不要修改输入和标签定义。

## 7. ImageNet 预训练权重

默认 ResNet 第一次运行会下载约 47MB 的 `resnet18-f37072fd.pth`。网络受限时，
从另一台机器复制到：

```text
~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth
```

`--no-pretrained` 会改变训练条件，不应被静默使用。

## 8. 训练产物

- `best.pth`：验证角点误差最低的 checkpoint；
- `last.pth`：最后一轮 checkpoint；
- `metrics.csv`：逐 epoch 指标；
- 外层 `.log`：完整终端日志。

这些产物与数据集均被 `.gitignore` 排除。
