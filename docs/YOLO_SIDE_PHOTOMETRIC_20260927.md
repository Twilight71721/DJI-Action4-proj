# YOLO侧面真实帧与成像增强继续训练

## 模型保护

原权重保留：`outputs/yolo11s_cleanblur5000_real48_20260925/train/weights/best.pt`。

独立备份：`outputs/yolo11s_side_photo_20260927/baseline_backup/best.pt`。

二者SHA256均为 `78c522b39c2f40dd7fb83447f7987e2cc2cae5f712bf7a6e6620233d66655dd6`。备份附原训练配置与指标。`rollback.json` 记录原路径、备份、候选路径；新候选保存至独立 `train/weights/best.pt`，不会覆盖或自动替换任何现有推理模型。训练前后均校验两份旧权重。

回退只需将推理的 `--yolo` 或 `--weights` 指向上述备份文件，不需要重新训练。当前推理路径本来就没有变。

## 数据与标注

- 保留原8000张合成训练图（4000清晰＋4000模糊）、32张真实训练帧。
- 新增22张真实训练帧、44个相机框，覆盖侧面、斜侧面、不同倾斜和暗部。均来自既有训练时间段0–499、800–1299、1600–2099、2400–2849，与原留出帧至少间隔51帧。
- 标注由旧YOLO提议后逐框视觉审核，部分边界作人工坐标修订；不是未经检查的伪标签，也没有声称是独立人工确认的精确真值。移除2625帧墙面图案假阳性；110帧重复/漏检候选与2815帧图像边缘截断候选未加入。
- 框仍定义为相机机身及镜头，不把线缆、腕带和安装支架整体包入；机身前方已有线缆遮挡不擦除。
- 固定原8张真实验证、8张真实测试，图像和标签逐文件哈希保持不变。新的侧面训练帧不加入评估集。
- 独立训练图8054张；真实训练帧共54张。旧真实重复40次、新真实重复80次，总计每轮11040条，真实占27.54%。重复采样不等于增加独立真实数据。
- 视频曾用于背景/姿态设计，固定测试也已经被多轮查看；仅作同视频开发对照，不是独立盲测。

数据：`dataset/action4-yolo-side-photo-20260927`。预览在其 `review/index.html`。合成图片使用软链接复用旧数据；不要单独搬走链接目录。

## 继续训练

使用备份的当前YOLO11s已学习参数初始化，重新建立低学习率优化器，不恢复旧优化器；不从短训权重开始。

本地GPU，最多20轮，patience7，imgsz960，batch4，workers2，AdamW lr5e-5，余弦衰减、AMP。关闭mosaic/mixup，保留小幅几何增强（degrees3、translate0.04、scale0.15）并由YOLO同步更新框。水平/垂直翻转关闭。HSV关闭以避免与自定义颜色增强叠加。

`YOLOPhotometric` 插入训练数据的几何变换之后、Format之前，对BGR转RGB应用已存在balanced配方，然后恢复BGR：约20%不额外增强、68%轻度、12%中等。全部训练域都参与在线随机抽样，不新增磁盘图片；验证测试无此变换。模型仍使用YOLO自身的LetterBox预处理，这不同于ResNet的矩形输入。

增强只改RGB，不改bbox/classes/instance对象。训练启动回调核对实际数据加载器确实包含成像增强，以及每轮11040条采样。关闭close_mosaic，避免Ultralytics重建变换时移除自定义增强。已有Ultralytics默认的其他变换仍按安装版本执行，因此不是仅成像增强的严格单因素实验。

```bash
.venv-hccepose/bin/python -u scripts/run_yolo_side_photo.py \
  --run outputs/yolo11s_side_photo_20260927 \
  --dataset dataset/action4-yolo-side-photo-20260927 --epochs 20
```

此命令拒绝覆盖已有train目录。进度：`training.log`、`status.json`（训练阶段）、`workflow_status.json`（完整流程）。短训只验证执行路径，位于独立smoke目录，不续接到正式训练。

## 自动评估与回退判断

完成后自动：真实验证/测试 → HTML逐帧报告 → 与旧YOLO在同一8帧测试上比较 → 两模型在原合成测试图上作退化检查。best由验证集选取，测试不用于选轮数。

主要检查mAP50–95、固定conf0.25/IoU0.5的误检与漏检，以及框是否误包支架或裁掉镜头。合成测试是历史回归检查，不声称新盲测。报告在`review/index.html`和`comparison/index.html`；数值决策记录`candidate_decision.json`。无论数值如何均不自动激活新模型。

YOLO框改善不保证ResNet角点直接改善：两者是不同任务。采用候选前还应复跑同一批ResNet真实诊断，避免检测框分布变化使角点模型退化。
