# ResNet18 长方形紧裁剪训练包：OpenCode 操作指南

本轮将刚讨论的输入、骨干、损失、解码与可选 PnP 集成成一次新实验。
目标是改善真实帧八角点预测，是否改善需要实际训练后比较，不能提前保证。
这是代码、权重和真实诊断帧的增量包；复用服务器已经生成的完整数据，不重复传输大批图片。
不覆盖旧实验，不使用本地小批检查数据代替正式数据。

## 1. 本轮究竟训练什么

| 项目 | 本轮设置 |
|---|---|
| 任务 | 单张 RGB 实例裁剪 → ResNet18-FPN → 8 张有序角点热图 → 8 个二维投影点 |
| 训练初始化 | 包内 ImageNet ResNet18 预训练；FPN/预测头重新初始化 |
| YOLO | 固定已有 `yolo11s_frozen.pt`，只用于真实帧定位，不重训 |
| RGB 输入 | 按机身紧裁剪的长方形；不扩到原来 1.3 倍正方形；无灰色 RGB 补边 |
| 训练框 / 推理框 | 训练用渲染 `bbox_obj`，真实推理用固定 YOLO 的机身框；都不使用真实八角真值生成输入 |
| 长宽 | 9 个近似长宽比例桶，从 384×128 到 128×384；同桶成批，按桶尺寸缩放 |
| 输出范围 | 特征和热图每边增加 32 格，允许投影角点落在 RGB 裁剪之外；这是输出范围，不是 RGB 灰边 |
| 损失/解码 | `spatial_kl` / `local`，局部半径 3；高斯目标 sigma=2 热图格 |
| 优化 | 30 epochs，batch=32，workers=8，BF16，AdamW，lr=3e-4，backbone lr×0.1，cosine |
| BatchNorm | 冻结预训练统计；前 2 轮冻结骨干，然后训练骨干参数 |
| 训练可见度 | 仅训练集过滤 `visib_fract < 0.05`，逐项留痕；验证/测试不做此筛选 |
| 背景/模糊 | 延续在线随机混合背景、清晰/模糊成对数据；不新增遮挡和裁剪扰动 |
| 真实角点 | 不参与这轮 ResNet 训练；已有 YOLO 本身曾用真实 bbox 训练，这一点保留 |
| PnP | 合成评估使用已知 K，对预测八点做透视拟合；同时保留原始八点结果 |

长宽比例桶会产生少量非等比缩放，不声称完全保持比例。热图监督保留所有八个点，
不会把裁剪外角点截到边界，也不会通过删除难点使验证指标好看。
本轮同时改变输入、骨干、损失及解码，因此不能把改善单独归因于 ResNet18。
ResNet50 旧权重和真实微调权重均不能加载到这个新实验中。

## 2. 复用服务器数据

已知上一轮工程：

```text
/home/aitest/exam-project/resnet_continuous_pose_background_20260924/
  work/continuous_data/online_config.json
  work/continuous_data/trainval/
  work/continuous_data/models/models_info.json
  pose_assets/manifest.json
  work/mixed_backgrounds/manifest.json
```

`online_config.json` 还引用这些既有目录及其实际图片：

```text
/home/aitest/exam-project/resnet5000_cleanblur_4090/dataset
/home/aitest/exam-project/resnet5000_background_mixed_20260923/augmentation
/home/aitest/exam-project/resnet_pose_mix_no_occlusion_20260924/work/pose_mix_data/backgrounds
```

部分背景是跨旧目录的链接，旧目录不能删除。程序会检查实际引用文件。
若目录迁移或缺失，请记录缺失路径并反馈，不得自动改用其他数据、重新随机划分、
临时生成少量姿态来凑数，或删除检查条件。只有本包不能重建服务器全部历史资产。

本轮固定的数据规模是：

- 原始 5000 张合成图的分组划分：4000 train / 500 val / 500 test，每张 2 个实例。
- 旧训练清晰与模糊共 16,000 实例；新姿态 12,000 对共 24,000 实例。
- 正式训练共 **40,000 原始实例**，执行训练专用可见度过滤后的实际数量由服务器审计输出。
- 固定验证 **3,048 实例**：旧清晰验证 1,000＋新姿态验证 2,048。
- 旧合成测试：清晰 1,000、模糊 1,000，使用 manifest 原始 test 场景，训练不读取它们。
- 前景姿态库：训练 4,096 个，验证 256 个；混合背景库 13,824 条记录。
- 姿态来自 SO(3) 抽样和视频姿态附近连续扰动；它仍是有限采样，不等于穷尽全部姿态。

服务器检查会核对上一轮返回记录中的 manifest 哈希、旧数据 RGB/几何、背景与 mask 哈希、
RGBA 有效性、场景分组、实例数及八点投影范围。新生成场景的 RGB 检查文件存在性，
RGBA 检查可读性/alpha；没有声称所有新增资产都具备逐文件历史哈希。
`input_preview/index.html` 保存服务器实际输入样本，验证换背景不会改变八点或内参变换。

## 3. 解压、环境与启动

把 `resnet18_rect_fixed_yolo_20260926.tar.gz` 和相邻 `.sha256` 放到服务器。
在压缩包所在目录核对，再解压至 `/home/aitest/exam-project` 的新目录：

```bash
sha256sum -c resnet18_rect_fixed_yolo_20260926.tar.gz.sha256
tar -xzf resnet18_rect_fixed_yolo_20260926.tar.gz -C /home/aitest/exam-project
cd /home/aitest/exam-project/resnet18_rect_fixed_yolo_20260926
sha256sum -c SHA256SUMS
conda activate action4-bb8
nvidia-smi
python -c "import torch, torchvision, ultralytics; print(torch.__version__, torchvision.__version__, ultralytics.__version__); print(torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported())"
```

使用此前已验证的 Python 3.10、torch 2.8.0+cu128、torchvision 0.23.0+cu128，
Ultralytics 8.3.233。优先复用已有环境，不要重装驱动、内核或系统 CUDA。
`requirements.txt`、`requirements-yolo.txt` 是依赖记录，只有缺失环境才按它们建立环境。
若环境版本不符，先说明现状，不能为了跑通而悄悄升级或退回 CPU。
本包自带 ImageNet 权重；入口仅为训练子进程设置包内 `TORCH_HOME`，无需联网下载或修改用户配置。

先运行包内测试：

```bash
PYTHONPATH=tests python -m unittest test_pnp test_rectangular_protocol test_net test_online_background test_background_augmentation test_bop_bb8 test_crop_jitter test_blur_augmentation test_rectangular_server
```

部分早期测试需要未打包的历史小数据集，会明确显示 `skipped`；这不能替代下面针对完整服务器数据的检查。

推荐在 tmux 中执行：

```bash
tmux new -s rect-r18
conda activate action4-bb8
cd /home/aitest/exam-project/resnet18_rect_fixed_yolo_20260926
python -u scripts/run_rectangular_server.py --stage all
```

默认顺序：完整检查 → 64 train / 64 val 的 1 轮短训 → 全量 30 轮 → 合成评估/PnP →
固定 YOLO 的真实 RGB 推理 → HTML 汇总 → 回传压缩包。短训不续接到正式训练。
任一步失败即停止。正式训练进度在：

```bash
tail -f outputs/rect_r18_fixed_yolo_30ep/training/train.log
tail -n 4 outputs/rect_r18_fixed_yolo_30ep/training/metrics.csv
```

入口不是断点恢复工具。需要逐阶段执行或在阶段间继续时：

```bash
python scripts/run_rectangular_server.py --stage check
python scripts/run_rectangular_server.py --stage smoke
python scripts/run_rectangular_server.py --stage train
python scripts/run_rectangular_server.py --stage evaluate
python scripts/run_rectangular_server.py --stage pack
```

已成功 `check` 后，不要再用 `all` 覆盖同一目录，应从下一阶段继续。
模型训练中断后保留现场，使用新的 `--output` 重跑完整流程；当前不自动续训。
如果评估中途失败，保留或重命名该次 `evaluation` 和 `real` 部分结果后再执行 `evaluate`。
不要删除旧实验。每个阶段使用相同 `--output`、`--previous`、batch 和 workers。

若 4090 OOM，保留日志，在新目录用 batch 16 重新检查和短训：

```bash
python -u scripts/run_rectangular_server.py --stage all --batch-size 16 --output outputs/rect_r18_fixed_yolo_30ep_b16
```

不要缩小输入、输出范围、训练轮数或训练数据来绕过 OOM。
GPU 速度受在线数据合成、磁盘和长宽桶影响；以正式训练前几轮实测时间估算剩余时间。

## 4. 怎么解释评估

正式权重根据完整 3,048 实例验证集的原始八点均误差选择 `best.pth`。
清晰/模糊测试用于最终报告，不用于选权重或调阈值。
合成集用真值机身框定位，反映角点网络；真实帧走固定 YOLO，反映实际两阶段流程。
它们不是同一种检测精度指标。真实帧保存全部检测和漏检数，绝不改用人工框补漏检。

指标报告原图像素误差及 `body_long_side_256` 误差：原图误差除以机身框长边，再乘 256。
PCK5 使用后一种单位；**不能直接与以前 256×256 大正方形裁剪的 PCK5 比高低**。
有需要时应将旧权重在同一批样本上换算到同一坐标尺度再比较。

PnP 只拟合预测点，不使用真实角点修正答案。报告原始点指标、接受比例、
拒绝计失败的全量 PCK5、仅接受部分指标以及拒绝时退回原始点的指标。
几何上接受不代表姿态正确，不能只挑接受样本展示。

真实帧没有拍摄相机的可信 K 和畸变参数，默认不做 PnP。不能把合成 K、被拍摄 Action4 的内参、
或未经验证的假定焦距当作拍摄相机标定。真实 PnP 不是训练的前置条件。
将来提供标定后，可单独运行 `infer_resnet_yolo_real.py --help` 中的
`--pnp-camera` 和 `--models-info`，按该图像分辨率提供 K、畸变、`image_size` 与 `source`。
本包 11 张真实 RGB 帧包括第 0、473、500 帧及历史 8 张诊断帧；分辨率不同，不能共用一个未变换的 K。
这些视频已用于背景和姿态设计，属于开发诊断，不能称为独立真实盲测。

## 5. 需要传回什么

顺利完成后入口生成：

```text
outputs/rect_r18_fixed_yolo_30ep.tar.gz
outputs/rect_r18_fixed_yolo_30ep.tar.gz.sha256
```

回传包含完整每轮日志/指标、配置和来源记录、数据与投影审计、实际输入预览、合成集原始预测
及 PnP 数值/叠图、11 张真实源帧、真实检测/输入/预测/热图、HTML 汇总，以及 best/last 推理权重。
导出的权重省略 optimizer 以减小传输，服务器原始完整检查点保留，不能用精简回传权重恢复优化器。
压缩包包含逐文件 SHA256SUMS 并在服务器回读核验。

训练启动后请向用户报告 GPU、实际过滤后 train 数、val 数、输出路径和首轮指标。
完成后报告 best epoch、清晰/模糊 test 数值和回传包大小/哈希；不要把 smoke 指标当正式结果。
若完整服务器姿态超出既定热图范围，保存 `geometry_audit.json` 并反馈，不得直接忽略边界检查。

## 6. 本地交付检查边界

本包附 `LOCAL_PACKAGE_QA.json` 与 `local_qa/`，记录解压包测试及本地小批流水线检查。
这些仅验证代码、坐标、推理及打包接口，没有代替服务器全量训练。
新 ResNet18 的真实泛化是否更好，要等本轮正式权重与真实预测回来再判断。
