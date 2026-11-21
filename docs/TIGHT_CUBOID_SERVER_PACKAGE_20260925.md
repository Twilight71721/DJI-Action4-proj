# 紧八角框训练增量包：OpenCode / RTX 4090 操作说明

## 要完成的任务

在新目录使用包内代码，复用服务器上一轮 continuous_pose_background_30ep 的数据和前景/背景资产。只训练 30 轮紧框 ResNet，保留现有 YOLO 权重与检测设置不变，再进行真实帧端到端预测并回传结果。取消上一版的 YOLO 重训、重新生成检测标签和修改检测框定义。

本轮只调整 ResNet 输入裁剪，不修改 GLB、logo、渲染材质、模型架构和损失。ResNet 从未用真实角点微调的 continuous-pose epoch27 best 初始化。背景随机化、清晰/模糊配对、三维姿态、原划分继续保留。紧框参数固定为 `--crop-mode cuboid --crop-padding 1.0 --crop-border 4`。四周 4 像素为纯灰边，不取额外背景。详见 `docs/TIGHT_CUBOID_INPUT_20260925.md`。

包内含独立运行所需的代码、配置、测试、真实帧、固定使用的 YOLO 权重和少量审校材料。没有复制服务器已有的 5000 清晰/5000 模糊图、4352 个 RGBA 前景、混合背景库和 ResNet 起点权重。无需重新下载背景、渲染或上传大型原始数据。

## 1. 解压并校验

在压缩包所在目录执行，解压目录必须不存在，不要覆盖旧实验：

```bash
sha256sum -c resnet_tight_cuboid_fixed_yolo_20260925.tar.gz.sha256
tar -xzf resnet_tight_cuboid_fixed_yolo_20260925.tar.gz
cd resnet_tight_cuboid_fixed_yolo_20260925
sha256sum -c SHA256SUMS
```

使用之前成功训练的 Python/PyTorch CUDA 环境。`requirements.txt` 与 `requirements-yolo.txt` 用于核对版本；不要重新安装 CUDA 或驱动。YOLO 入口需要 ultralytics，缺少时在当前环境安装 `requirements-yolo.txt` 中指定的版本，保留现有可用 PyTorch。

```bash
python -c "import torch, ultralytics; print(torch.__version__, ultralytics.__version__); print(torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported())"
python -m unittest discover -s tests -p 'test_tight_crops.py' -v
python -m unittest discover -s tests -p 'test_online_background.py' -v
```

## 2. 找到服务器已有资产

下列 previous 路径来自上一轮服务器返回结果；先确认实际目录。如果路径变了，在服务器定位对应文件，不要随意换成同名的旧权重。

```bash
TIGHT_PREVIOUS=/home/aitest/exam-project/resnet_continuous_pose_background_20260924
TIGHT_PREPARED="$TIGHT_PREVIOUS/work/continuous_data"
TIGHT_BASE="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["base_dataset"])' "$TIGHT_PREPARED/online_config.json")"
TIGHT_MASKS="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["old_background_assets"])' "$TIGHT_PREPARED/online_config.json")"
TIGHT_CHECKPOINT="$TIGHT_PREVIOUS/outputs/continuous_pose_background_30ep/training/best.pth"
sha256sum "$TIGHT_CHECKPOINT"
```

起点 SHA256 只允许：

- 服务器原始权重：`96dac397a32491dddbacece40d3989a8975bf744fccd364689c23a3819f65fe1`
- 去优化器回传版：`a864512c1f1c90562469e0216ae30ccabd8eff7cfa2a71720ebd48a38c615ef9`

检查 `pose_assets/manifest.json`：4352 个前景，4096 train / 256 val。检查 `online_config.json`：pairs=12000、seed=20260927、expected_train_instances=40000、expected_val_instances=3048。复用原工作目录中固定的验证背景组合，不能重新随机划分。

如果原始工作目录或链接失效，先修复对应实际资产路径并保留旧配置副本。只有确认工作目录丢失时才另建新工作目录并执行 `prepare`；使用相同资产、seed 与 pairs，记录重建事实。不要静默减少样本或跳过校验。

## 3. 训练紧框 ResNet 并评估

```bash
TIGHT_ARGS=(
  --config configs/tight_cuboid_20260925.json
  --dataset "$TIGHT_BASE" --checkpoint "$TIGHT_CHECKPOINT"
  --assets "$TIGHT_PREVIOUS/pose_assets"
  --backgrounds "$TIGHT_PREVIOUS/work/mixed_backgrounds"
  --old-background-assets "$TIGHT_MASKS"
  --workspace "$TIGHT_PREPARED"
  --real-manifest "$PWD/real/manifest.json"
  --output "$PWD/outputs/tight_cuboid_fixed_yolo_30ep"
  --batch-size 32 --workers 8
)
python scripts/run_online_pose_training.py "${TIGHT_ARGS[@]}" --action check
python scripts/run_online_pose_training.py "${TIGHT_ARGS[@]}" --action smoke
python scripts/run_online_pose_training.py "${TIGHT_ARGS[@]}" --action train
python scripts/run_online_pose_training.py "${TIGHT_ARGS[@]}" --action evaluate
```

不要对已有工作目录执行 `--action all`，因为它包含重新准备。旧结果不覆盖，新输出目录为本包下的 `outputs/tight_cuboid_fixed_yolo_30ep`。训练是 30 轮、40000 实例/轮，BF16、batch32、lr1e-4、骨干lr1e-5、cosine。显存不足时统一调低 batch，再在新的输出目录完成对应 smoke 和训练。不要自动改输入分辨率或训练轮数。

`training/metrics.csv`、`training/status.json`、`training/train.log` 记录训练进度。默认无自动断点恢复；若中断，保留结果并明确报告中断轮数，不把不完整实验当成功。

评估分别覆盖旧 clean/blur val/test、新连续姿态验证。模型各自用自身裁剪，但误差统一换算到同一个旧参考坐标系；样本与角点分母必须一致。新姿态集参与选权重，不能称独立测试。ResNet 合成评估使用真值定位框，真实预测在第 4 步单独报告。

## 4. 固定现有 YOLO，运行真实帧预测

不训练 YOLO，不修改它的权重、类别名或检测标签。包内 `weights/yolo11s_frozen.pt` 就是已经训练好的现有检测器，SHA256 应为 `78c522b39c2f40dd7fb83447f7987e2cc2cae5f712bf7a6e6620233d66655dd6`。它曾接受真实机身 bbox 监督；本轮保持固定，ResNet 仍不接受真实八角点微调。

```bash
sha256sum weights/yolo11s_frozen.pt
python scripts/predict_tight_cuboid_real.py \
  --real-manifest real/manifest.json \
  --yolo weights/yolo11s_frozen.pt \
  --resnet outputs/tight_cuboid_fixed_yolo_30ep/training/best.pth \
  --box-definition body \
  --output outputs/tight_cuboid_fixed_yolo_30ep/real_detector
```

脚本只读取真实 RGB，忽略 manifest 中已有的人工框和旧保存框，重新运行固定 YOLO。conf=.25、imgsz=960、NMS IoU=.7 保持不变，保留全部候选与无检测帧。检测框坐标直接送入 ResNet 的紧裁剪/灰色补边预处理，不额外扩大真实背景范围，不以人工角点修正框。

**需要如实解释的局限：**训练根据八个 GT 投影点获得精确外接矩形，预测使用现有 YOLO 机身框作为近似定位。两者并非严格同一定义，部分投影角点可能在检测框或最终裁剪外。此时是端到端裁剪差异，不能把使用 GT 框的合成指标等同于真实效果，也不能声称已经保证八角全部包含。保留现有检测器便于先检查 ResNet 调整的效果；本轮不据此强行重训 YOLO。

入口内部显式使用 `--allow-body-box-approximation`，结果 JSON 中记录 `body_box_approximation=true`，保持 `detector_box_definition=body`，不能改名冒充 cuboid 检测器。查看 `real_detector/index.html` 及 `network/index.html`。真实视频曾用于背景和姿态设计，不是独立盲测。

完整视频留在本地，回传 ResNet 权重后可继续使用固定 YOLO 逐帧处理。本地视频入口在紧框模型配合机身框时需显式加 `--allow-body-box-approximation`。

## 5. 打包回传

确认 ResNet 30 轮、合成评估和固定 YOLO 的真实帧预测均完成，再执行：

```bash
python scripts/run_online_pose_training.py "${TIGHT_ARGS[@]}" --action pack
```

回传这两个文件：

- `outputs/tight_cuboid_fixed_yolo_30ep.tar.gz`
- `outputs/tight_cuboid_fixed_yolo_30ep.tar.gz.sha256`

结果包包含 ResNet best/last（去优化器）、训练配置/日志/指标、统一坐标的旧新对照、新姿态验证、固定 YOLO 的权重哈希、真实帧检测/角点可视化。不要回传原数据、背景库或临时缓存。若某阶段失败，保留具体日志并报告，不擅自用旧模型替代本轮结果。
