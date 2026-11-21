# ResNet 输入改为紧贴八角框：数据、训练和预测

## 本次调整的原理

旧输入以机身 `bbox_obj` 为依据，外扩 1.3 倍并取原图中的正方形区域；外扩带入手臂、衣服和地面等上下文。新输入使用八个三维包围盒角点在原图上的投影，以全部八点的 `min(x), min(y), max(x), max(y)` 确定最小轴对齐矩形。只采样该矩形里的原图像素，保持长宽比缩放到 256×256，长宽比造成的空白填 RGB(114,114,114)。四周保留约 4 个输入像素的纯色边缘，让最外侧点进入 64×64 热图的有效范围；边缘不采样原图背景。

这属于紧框裁剪，不是相机轮廓分割，矩形内仍可能有背景。框和角点只作为几何标签，不画进神经网络的输入 RGB。原始三维角点顺序、K、R、t 不改，裁剪变换同步作用于输入 K、2D 角点、热图和预测的原图坐标还原。

ResNet50-FPN、256 输入、64 热图、损失、归一化和当前稳定 Top-9 解码保留。清晰/模糊配对、连续三维姿态、独立外观和灯光、在线随机背景保留；不加入人工遮挡或裁剪扰动。旧 checkpoint 和旧实验结果保留。

## 已生成的数据与检查

本地目录：`dataset/action4-tight-cuboid-20260925/`。

| 内容 | 数量/结果 |
|---|---:|
| 基础合成图 | 5,000 张，每张 2 台相机 |
| 清晰＋模糊图 | 10,000 张，共 20,000 实例 |
| 训练 | 4,000 基础图，清晰/模糊合计 16,000 实例 |
| 验证 | 500 基础图，清晰/模糊合计 2,000 实例 |
| 测试 | 500 基础图，清晰/模糊合计 2,000 实例 |
| 原图采样面积/旧输入面积中位数 | 55.76% |
| 独立 OpenCV 投影与标签最大差异 | 0.000030 输入像素以内 |
| 原有按场景分组 | 保持，不重新打乱 |

面积比例不是背景像素比例，也不是准确率。原数据所有 10,000 个基础实例的八个几何投影点在新虚拟输入窗口中均保留。其中 6 个实例的部分投影点本来就在原始图像边界外：该位置只能填灰并保留几何监督，不能恢复被拍摄范围截掉的图像；对应 YOLO 标签按标准图像范围截断，`source_valid` 和 `all_corners_inside_source` 保留这一标志。这类截断目标不保证通过检测框包含图外全部角点。

主要文件：

- `crop_index.jsonl`：每个基础实例的八点范围、裁剪变换、原图/输入角点、K 和原图可见范围标志。
- `review/index.html`：40 个训练实例与 9 个真实参考实例的旧输入/新输入对照。真实人工点仅供审校，没有加入本次训练。
- `yolo/data.yaml`、`yolo/labels/`：以同一定义投影八角外接框为目标的检测标签，类别为 `action4_cuboid`。
- YOLO 图片使用指向原始数据的符号链接；ResNet 训练时动态裁剪原 RGB，不需要再次压缩上传全部图片。

当前已通过训练/预测像素一致性、正反坐标变换、OpenCV 独立重投影、负原点裁剪、八通道热图、在线背景与旧流程兼容性检查。另在本地完成 8 个训练实例、8 个验证实例、1 轮的 BF16 GPU 短流程检查。该短流程 checkpoint **不用于正式精度或视频报告**。完整服务器训练尚未开始。

## 预测端保持现有 YOLO

按用户最新决定，取消 YOLO 重训。本轮训练使用八角点精确外接矩形，真实预测使用原 YOLO 机身框作为近似定位，沿用相同紧裁剪/灰色补边变换。YOLO 权重、置信度阈值、NMS 和原始框坐标不改。

两种框仍有定义差异：9 个现有人工参考实例中，旧 YOLO 框并不完全包含所有八角投影点。因此不能承诺真实裁剪保留全部点，也不能把真实失败全部归因于 ResNet。此差异单独标记为 `body_box_approximation=true`，不会通过真实 GT 修正检测框或改名伪装为八角框检测器。

`predict_tight_cuboid_real.py` 使用 `--box-definition body` 并显式传递 `--allow-body-box-approximation`；视频入口也支持这一显式选项。此前生成的 cuboid 检测标签只作为历史审计保留，本轮不用于训练 YOLO。

## 4090 服务器使用

更新本次改动的 Python 文件及 `configs/tight_cuboid_20260925.json`，复用上一轮服务器上已有的数据、4352 个 RGBA 前景和混合背景池，不必重新渲染或上传原图。路径应以服务器现有文件为准。新配置初始化自 **continuous_pose_background_30ep 的 epoch27 best.pth**，未使用真实角点微调。

允许的 checkpoint SHA256：

- 服务器原版：`96dac397a32491dddbacece40d3989a8975bf744fccd364689c23a3819f65fe1`
- 本地回传版：`a864512c1f1c90562469e0216ae30ccabd8eff7cfa2a71720ebd48a38c615ef9`

先核对原版 SHA，然后在已安装依赖的训练环境中执行。以下复用上一轮完整的 `work/continuous_data`（12000 pairs，seed=20260927）；输出采用新目录。

```bash
TIGHT_PREVIOUS=/home/aitest/exam-project/resnet_continuous_pose_background_20260924
TIGHT_BASE=/home/aitest/exam-project/resnet5000_cleanblur_4090/dataset
TIGHT_MASKS=/home/aitest/exam-project/resnet5000_background_mixed_20260923/augmentation
TIGHT_CHECKPOINT="$TIGHT_PREVIOUS/outputs/continuous_pose_background_30ep/training/best.pth"
sha256sum "$TIGHT_CHECKPOINT"

python -m unittest discover -s tests -p 'test_tight_crops.py' -v
python -m unittest discover -s tests -p 'test_online_background.py' -v

TIGHT_ARGS=(
  --config configs/tight_cuboid_20260925.json
  --dataset "$TIGHT_BASE" --checkpoint "$TIGHT_CHECKPOINT"
  --assets "$TIGHT_PREVIOUS/pose_assets"
  --backgrounds "$TIGHT_PREVIOUS/work/mixed_backgrounds"
  --old-background-assets "$TIGHT_MASKS"
  --workspace "$TIGHT_PREVIOUS/work/continuous_data"
  --real-manifest "$TIGHT_PREVIOUS/real/manifest.json"
  --output "$PWD/outputs/tight_cuboid_fixed_yolo_30ep"
  --batch-size 32 --workers 8
)
python scripts/run_online_pose_training.py "${TIGHT_ARGS[@]}" --action check
python scripts/run_online_pose_training.py "${TIGHT_ARGS[@]}" --action smoke
python scripts/run_online_pose_training.py "${TIGHT_ARGS[@]}" --action train
python scripts/run_online_pose_training.py "${TIGHT_ARGS[@]}" --action evaluate
python scripts/run_online_pose_training.py "${TIGHT_ARGS[@]}" --action pack
```

复用已有工作目录时不要用 `--action all`，它会尝试重新 prepare。若旧工作目录已丢失，改为新的 `--workspace`，执行 check 后再执行 prepare。新流程仍是 30 轮，每轮 40,000 训练实例；验证为旧清晰 1000＋新固定姿态 2048，共 3048。BF16、batch32、学习率1e-4/骨干1e-5、cosine，初始化模型权重但重新建立优化器。显存不足时两阶段都使用同一较小 batch，再完成匹配的 smoke；不要修改系统驱动。

关键训练参数为 `--crop-mode cuboid --crop-padding 1.0 --crop-border 4`，已接入服务器入口。旧配置不含这些字段时继续走旧裁剪流程。

新评估脚本 `scripts/evaluate_tight_cuboid.py` 使用各自模型训练时的裁剪，将预测还原到同一旧 1.3 倍框坐标计算误差，核对样本 ID、真值和有效角点分母一致；避免目标放大导致两个不同输入坐标系的 px 数被错误横比。新姿态集属于选权重验证集，不称独立测试。合成评估只计算真值定位下的角点误差；真实帧另用固定 YOLO 测试，并显式记录机身框近似裁剪的差异。继续训练也增加了优化步数，所以结果不属于严格的等步数单因素消融。

本轮不重新生成 YOLO 标签，也不重训检测器。完整命令以本次包内 `OPENCODE_SERVER_GUIDE.md` 为准：ResNet 30 轮 → 固定 YOLO 真实帧预测 → 回传结果。

## 代码改动范围

`bop_bb8/cropping.py` 集中定义裁剪；`dataset.py`、`background.py`、`online_background.py` 接入训练数据；`train_heatmap.py` 与 `run_online_pose_training.py` 保存/传递新配置；`infer_resnet_yolo_real.py`、`infer_camera_video.py` 接入相同预测变换；`evaluate_cleanblur5000.py` 默认拒绝误用旧评估裁剪，新的 `evaluate_tight_cuboid.py` 明确进行跨裁剪评估；`prepare_tight_cuboid.py` 生成索引、检测标签、审校页；`tests/test_tight_crops.py` 验证几何和像素一致性。

没有修改网络结构和损失，也没有覆盖旧数据或旧结果。
