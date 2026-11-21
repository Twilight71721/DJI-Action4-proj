# YOLO11s：清晰/模糊合成图与真实帧联合重训

本次任务是相机二维检测框，不是八角点预测。训练在本机 GPU 运行，历史模型和结果保留。

## 数据

- 源合成数据：`dataset/action4-cleanblur5000-20260923`，5,000 张原图和对应模糊版本。
- 沿用渲染批次划分：4,000 个训练原图、500 个验证原图、500 个测试原图。
- 本次实际合成训练输入：4,000 张清晰图 + 4,000 张模糊图。配对图共享标注和划分，不允许跨集合。
- 真实帧沿用此前复查的 48 张标注：32 张训练、8 张验证、8 张测试，每帧两个相机。
- 原有真实视频时间块及至少 50 帧的间隔不变；不把新八角点试标数据混入检测测试划分。
- 每轮真实训练帧重复采样 80 次，共 2,560 条真实样本条目；加 8,000 条合成样本条目，每轮共 10,560 条。真实帧约占 24.24%，但独立真实训练图仍只有 32 张。
- 共 8,032 张独立训练图、16,064 个检测框。重复采样用于保持真实域权重，不代表获取了新的真实数据。
- 图像用本地符号链接引用，避免复制完整数据；符号链接数据目录不能单独打包迁移到服务器。
- 本次使用本地完整的清晰/模糊 5,000 原图数据，不包含只在服务器生成、未回传图像的连续姿态随机背景数据。

合成图标签来自 BOP 的 `bbox_obj`，裁到图像边界；真实图沿用相机机身及镜头框，不包含线缆、腕带和安装支架。模糊操作不改变坐标。

## 训练配置

- 初始化：`outputs/yolo11s_real48_mixed_20260922/train/weights/best.pt`，从之前的真实帧混合模型继续微调。
- YOLO11s，输入 960，batch 4，最多 30 轮，patience 10。
- AdamW，初始学习率 0.0001，余弦衰减，3 轮 warmup，AMP。
- 保留旧检测器的增强设置；没有修改网络架构、损失或八角点解码。
- 数据：`dataset/action4-yolo-cleanblur5000-real48-20260925/data.yaml`。
- 结果：`outputs/yolo11s_cleanblur5000_real48_20260925`。

## 复现

在项目根目录执行：

```bash
.venv-hccepose/bin/python scripts/prepare_yolo_cleanblur.py \
  --source dataset/action4-cleanblur5000-20260923 \
  --output dataset/action4-yolo-cleanblur5000-20260925

.venv-hccepose/bin/python scripts/prepare_yolo_real_mix.py \
  --annotations outputs/action4_real48_annotation_20260922 \
  --synthetic dataset/action4-yolo-cleanblur5000-20260925 \
  --output dataset/action4-yolo-cleanblur5000-real48-20260925 \
  --real-repeat 80

bash outputs/yolo11s_cleanblur5000_real48_20260925/run.sh
```

上述转换命令拒绝覆盖已有数据集；训练拒绝覆盖已有 `train` 目录。复现实验需更换输出路径并同步修改运行脚本。

## 评估和状态

仅使用 8 张真实验证帧选择最佳权重，训练结束后自动评估 8 张真实测试帧，生成 `review/index.html` 和 `comparison/index.html`。与旧真实帧混合模型比较 mAP50、mAP50–95、固定 conf=0.25 / IoU=0.5 下的误检和漏检。

该视频此前已用于合成场景设计及多次诊断，不是独立拍摄的盲测；8 张测试图也不足以代表所有真实场景。沿用的框为助手视觉标注及复查，并非独立人工真值。由于初始化来自历史模型，合成留出场景也不能被当作新的盲测，本轮主报告使用冻结的真实检测划分。

`status.json` 记录训练轮数及训练/评估状态；`workflow_status.json` 记录包含生成报告在内的整个流程；`training.log` 保存完整输出；`run_config.json` 保存版本、数据清单哈希和初始化权重哈希。
