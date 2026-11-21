# Action4 YOLO11s 检测基线

本实验输入完整场景 RGB，输出单类别 `action4` 的二维机身框。左右设备为同类不同实例。检测器不预测八角点；后续沿用 ResNet50 的 ROI 裁剪、热图与坐标变换流程。

## 本次结果（2026-09-21）

完成 34 轮后触发 patience=20 早停，最佳权重来自第 14 轮。训练及合成评估总耗时 218.5 秒，PyTorch 峰值分配显存约 3.08 GiB。

| 集合 | mAP50 | mAP50–95 | Precision@0.25 | Recall@0.25 | TP/FP/FN |
|---|---|---|---|---|---|
| validation | 0.9950 | 0.8153 | 1.000 | 1.000 | 60/0/0 |
| test | 0.9796 | 0.6896 | 1.000 | 0.800 | 32/0/8 |

24 张均匀采样的真实帧中，13 张没有输出框、9 张输出一个框、2 张输出两个框。输出框数不等于正确检测数；预览还可见画面边缘误检。因此当前模型真实域表现明显不足，不能因合成 AP 较高就认为检测器可用。这是冻结阈值下的定性诊断，没有人工真值，未计算真实域准确率。

优先后续方向：扩大真实视角、暗色机身和遮挡覆盖；加入独立真实图片的检测标注以及无设备负样本；按固定验证方案比较。不要仅降低阈值掩盖漏检，也不要把更多相似合成图等同于必然改善。

验证记录：6 项检测转换/匹配测试通过；全部 200 张、400 框的归一化往返检查最大误差约 0.000001176 像素；检查了 40 张标注预览和真实/合成预测联系表。没有改动已有 ResNet 代码、渲染队列或 BOP 原数据，没有 commit/push。

## 数据与标签

- 来源：`dataset/action4-current-approved-200-20260921`，200 张 RGB、400 个相机实例。
- 输出：`dataset/action4-yolo-approved200-20260921`；复制原始 RGB，不更改 BOP 数据。
- `scene_gt_info.json` 中的 `bbox_obj` 为 `[x,y,w,h]`，裁到图像范围后转换为 YOLO 的归一化中心/宽高格式。只框相机，不框附件。
- 像素可见数为零或完全出框的实例不写为目标；当前 400 个实例都可见，没有因此跳过的目标。
- 同一 scene 的所有变体保持在同一 split。拒绝重复 RGB 和重复分配的 scene。完整记录在输出的 `manifest.json`。

当前显式分组（`configs/yolo_approved200_split.json`）：

| 集合 | 场景 | 图像/实例 | 动作 |
|---|---|---|---|
| train | 000000–000014 | 150/300 | 初始抬腕、展开、扣合、折袖 |
| val | 000015–000017 | 30/60 | 折衣摆、压平 |
| test | 000018–000019 | 20/40 | 结尾抬腕、手势 |

这是小规模、非同分布的动作段留出实验。验证/测试仍共享模型、材质与场景生成器，不能代表跨场景或真实域泛化。此划分与 5000 张方案不同，不能直接混合比较成绩。后续正式数据必须重新统一分组，不能将本次训练中使用过的样本作为独立测试。

## 环境与训练

复用 `.venv-hccepose` 中现有 `ultralytics==8.3.233`、`torch==2.8.0+cu128`、`torchvision==0.23.0+cu128`。本次没有升级安装包或更改驱动。独立检测依赖记录在 `requirements-yolo.txt`。

COCO 预训练 YOLO11s，输入 960，batch 2；AdamW，初始学习率 0.001，cosine 调度，最多 100 轮，patience 20，固定种子 20260921。小 batch 为同机渲染保留显存，但共享 GPU 仍会影响双方速度。

训练增强使用轻微亮度/颜色/角度/尺度变化及 0.3 概率的 mosaic。关闭左右和上下翻转，以避免镜像文字和设备外观。Mosaic 属于二维检测增强，不是新生成的物理场景。最后 10 轮关闭 mosaic（若早停提前发生，则不一定进入这一阶段）。

最佳模型只根据验证集选择。训练结束后评估 test，输出所有留出图的预测。AP 评估使用框架的置信度扫描；报告另外统计固定 `conf=0.25`、NMS IoU 0.7、匹配 IoU 0.5 下的 Precision、Recall、误检和漏检，不能混淆两种阈值口径。

## 复现

在项目根目录执行；输出目录必须尚不存在，防止覆盖已有结果：

```bash
.venv-hccepose/bin/python scripts/prepare_yolo_bop.py \
  --source dataset/action4-current-approved-200-20260921 \
  --output dataset/action4-yolo-approved200-20260921 \
  --split configs/yolo_approved200_split.json

.venv-hccepose/bin/python scripts/train_yolo_action4.py \
  --data dataset/action4-yolo-approved200-20260921/data.yaml \
  --weights outputs/yolo11s_approved200_20260921/yolo11s.pt \
  --output outputs/yolo11s_approved200_20260921 \
  --epochs 100 --batch 2 --imgsz 960

.venv-hccepose/bin/python scripts/report_yolo_action4.py \
  --run outputs/yolo11s_approved200_20260921 \
  --dataset dataset/action4-yolo-approved200-20260921 \
  --real-frames outputs/video_reference_audit/head_left_rgb_raw/frames_full

.venv-hccepose/bin/python -m unittest discover -s tests -p 'test_yolo*.py' -v
```

预训练文件来自 https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt 。训练配置、版本、权重和数据清单 SHA-256 保存于 `run_config.json`。`status.json` 保存当前轮数或失败信息，`train/results.csv` 保存逐轮指标。

## 阅读结果

- 数据框检查：数据输出中的 `review/index.html`。
- 训练后报告：运行输出中的 `review/index.html`。
- 最佳权重：`train/weights/best.pt`。
- AP 指标：`metrics.json`。
- 固定阈值逐帧漏检/误检：`review/fixed_threshold_metrics.json`。
- 真实诊断：均匀选取 24 张已有目标视频帧；不用于训练、选权重或调整阈值。由于没有人工检测真值，只提供可视化与原始预测，不计算真实准确率。该视频已经指导合成场景设计，因此不是独立盲测。

只有两个设备的合成正样本不足以验证无目标场景的误报，后续应补充无相机背景、单设备、更多姿态和独立真实图片。真实预测不好时，应先分析域差异，不把高合成 AP 当成任务完成。

参考：[Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/)、[检测数据格式](https://docs.ultralytics.com/datasets/detect/)、[训练参数](https://docs.ultralytics.com/modes/train/)。实际运行参数以记录的已安装版本为准。
