# 单张 RGB 八角点网络：紧长方形输入修订

日期：2026-09-26。状态：代码、几何审计、单元测试和短流程验证已完成；尚未进行正式 30 轮训练，不能宣称真实帧精度提高。

后续已新增[可选透视 PnP 后处理及验证](PNP_CUBOID_POSTPROCESS_20260926.md)。它不改变训练；合成评估可使用已有渲染内参，真实帧必须显式提供拍摄相机参数。默认仍保留原始八点结果。

## 1. 对老师反馈的落实

老师说长方形输入不必补成灰色正方形，这是正确的。当前 ResNet-FPN 没有固定尺寸的分类全连接输出，卷积和特征融合支持长方形。

本轮采用以下输入链路：

```text
训练：合成 RGB + BOP bbox_obj → 紧机身矩形 → 接近原长宽比的分组尺寸
预测：真实 RGB + 既有 YOLO 框 → 同样的紧机身矩形 → 同样的分组尺寸
                                    ↓
                           ImageNet RGB 归一化
                                    ↓
                       ResNet18 → 多尺度 FPN
                                    ↓
                         八个有固定编号的热图
                                    ↓
                      最强峰附近解码 → 原图八点
```

- RGB 不添加灰边、不扩大为 1.3 倍上下文方框，不画标签线再送入网络。
- 训练保持随机背景和清晰/模糊配对；不增加人工遮挡，不使用真实八点标签微调，不引入参考图。
- 既有 YOLO 权重和检测框语义保持不变。
- 训练使用合成真值机身框，预测使用 YOLO 机身框。二者采用同样的框定义和处理规则，但检测误差仍会造成分布差异，不能说完全相同。
- 上一轮使用“真值八角框范围裁剪”训练、“YOLO 机身框裁剪”预测，存在输入定义差异；本轮明确修订这一点。

预览：`outputs/single_rgb_rect_20260926/preview/index.html`。第 500 帧实际 ROI 为原帧中的 `300×161`，本轮网络输入为 `320×192`，无 RGB 灰边。预览使用保存的同一 YOLO 框，不是新检测结果。

## 2. 为什么不能只把灰边删掉

需要预测的是模型三维包围盒的投影角点。它们是几何上的虚拟点，未必都位于可见机身的二维矩形内。直接将热图限制在紧机身 ROI 中，就可能失去部分角点监督，或迫使网络把角点贴在边缘。

因此本轮区分两个范围：

1. **输入范围**：真实紧矩形 RGB，无额外背景或填充。
2. **输出坐标范围**：允许热图预测 ROI 外的虚拟角点。

具体采用 stride 4 的核心特征图，四周增加固定 `heatmap_margin=32` 个热图格。以 `W×H=320×192` 输入为例，核心网格为 `80×48`，输出为 `8×112×144`（C,H,W）。坐标映射：

```text
heatmap_x = input_x / 4 + 32
heatmap_y = input_y / 4 + 32
input_xy  = (heatmap_xy - 32) * 4
original_xy = input_xy * [crop_width/input_width, crop_height/input_height] + crop_origin
```

实现上在 FPN 特征层复制延伸边缘特征，并加入可学习的全局上下文和坐标投影，再生成热图。**延伸的是特征/输出，不是 RGB 图片。**这是本项目的实验性输出头，不能称为 BoxDreamer 原版方法，也没有证据保证它比其他输出头更好。它增加了计算量；后续应根据框外点误差评估其必要性。标准卷积自身的 padding 仍存在，也不等于给输入照片补灰边。

不裁掉、截断或强行拉回框外角点；训练启用 `require_all_corners`，任何不能监督完整八点的实例都会报错。

## 3. 长方形如何组成 batch

直接将不同 H、W 的张量堆叠成普通 batch 不成立。因此根据机身矩形的长宽比，选择最近的尺寸组，组内组成 batch：

```text
384×128, 352×160, 320×192, 288×224, 256×256,
224×288, 192×320, 160×352, 128×384
```

尺寸均可被 32 整除，不添加 RGB 补边，不丢弃末尾不足一个 batch 的样本。近似正方形目标仍可能落入 `256×256` 组。

这是小幅非等比缩放，不是严格保持长宽比：本地审计训练数据的长宽比畸变因子中位数约 1.064，最大约 1.145；几何坐标和内参按两个轴分别同步缩放。因此不能将这种操作描述为“完全无形变”。

分组顺序与背景采样使用可复现的种子。为了避免小组尾部 batch 影响 BatchNorm，默认固定 ImageNet 预训练的 BN running statistics，骨干解冻后 BN 的 affine 参数仍参与学习。这是实验选择，可能限制域适应，需通过验证集评估。

## 4. ResNet50 是否过大

ResNet50 不是明显错误的选择，真实帧失败也不能直接归因于参数过多。但本任务先建立较小、可诊断的基线更合适。本轮默认 ResNet18-FPN，并保留 ResNet50 的同协议入口。

本项目实测（包含 FPN128、八点输出头及本轮输出延伸）：

| 网络 | 参数量 | Conv/Linear MACs / 图 | 本机 FP32 前向中位耗时 |
|---|---:|---:|---:|
| ResNet18-FPN | 11,612,872 | 5.245 G | 1.54 ms |
| ResNet50-FPN | 24,313,032 | 8.206 G | 2.84 ms |

条件：RTX PRO 2000 Blackwell Laptop GPU、batch 1、`320×192` 输入、相同输出头；20 次预热、50 次前向计时。MACs 只统计卷积/线性层，不是完整 FLOPs。耗时不包括读图、反向、优化器，**不能用于声称 RTX4090 上的整轮训练时间**。

ResNet18 参数约少 52%，但“小”不保证泛化更好。模型容量、数据/标签偏差、预训练特征、损失与解码都可能影响结果。若比较 18/50，保持同一数据划分、预处理、背景种子、更新预算与指标。两者调用的 torchvision 预训练权重分别为 R18 ImageNet V1、R50 ImageNet V2，因此这是实用骨干方案对比，也并非严格只改变参数数量的因果实验。

官方结构资料：[ResNet18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)、[ResNet50](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html)。表中数值来自本地完整 FPN 网络，并非官方分类模型参数量。

## 5. 损失和解码修改

旧默认保留 `focal + coordinate loss` 和全局 top-9。新配置显式选用：

- `spatial_kl`：每个角点的 Gaussian 标签归一化为空间概率分布，与网络 logits 的空间 log-softmax 计算 KL。计算使用 FP32。整体降低某个通道所有 logits 不能降低这项损失，避免把“全背景数值很小”当作定位成功。
- `local` 解码：先找到该通道最强峰，只在半径 3 的邻域内计算加权亚像素位置，避免全局 top-k 把两个相距很远的峰平均到无意义位置。选择了错误的峰时仍会预测错误；没有几何美化或真实性保证。
- 新配置不叠加全局 soft-argmax 坐标损失，避免训练优化全局均值、推理使用局部峰的目标差异。KL 与局部解码也不完全等价，需要实测。
- heatmap 导出使用相同的空间 softmax 语义，峰值不能解释为定位正确率。

这是一套联合修订的候选方案。若新方案改善，不能直接宣称某一个改动是唯一原因；需要在同协议上逐项对照。

## 6. 数据审计和评估规则

本地用于流程核验的是 `outputs/continuous_pose_local_check_20260924/prepared`，包含旧合成数据和小批新姿态，**不是服务器完整扩增后的数据规模**。

| 项目 | 本地检查结果 |
|---|---:|
| 原训练实例（机身裁剪，非原图数） | 16,096 |
| 过滤后训练实例 | 16,092 |
| 验证实例（不做本轮过滤） | 1,128 |
| 过滤后训练所需最大热图边距 | 28 格 |
| 训练所需边距第 99 百分位 | 7 格 |
| 验证所需最大边距（仅报告） | 9 格 |
| 配置边距 32 下八点几何覆盖率 | 训练/验证均为 100% |

最后一行只是标签是否可表示，不是预测准确率。

训练过滤条件明确为 `visib_fract < 0.05`。排除的 4 个实例为两组清晰/模糊配对：

- `000133/18/1`、`100133/18/1`：可见比例约 0.000468，紧裁剪后只剩约一列像素。
- `000185/16/1`、`100185/16/1`：可见比例约 0.033686。

来源文件未删除，排除列表写入审计 JSON。热图边距由训练集最大需求 28 加 2 格余量并向 4 的倍数取整得到 32；验证集不能用来调边距。

评估使用原始帧像素误差，以及将机身框长边归一到 256 后的误差/PCK。归一化公式：`original_error / max(crop_width,crop_height) * 256`。这样不会因长宽比分组不同而改变物理误差口径。训练日志的新 `val_err_px` / `val_pck5` 使用这个归一化尺度，并在配置/checkpoint 标记 `body_long_side_256`。**不能直接与历史“256 方形上下文裁剪”的 PCK 混比。**

新离线评估把每个实例八个点都计入分母，框外点不被静默排除；投影非有限或在相机后方则明确拒绝生成成功指标。完整验证要求场景清单和标注指纹匹配；测试拒绝复用训练/验证场景标注指纹。但指纹不能证明所有背景、相邻视频帧和渲染父样本独立，仍须检查来源组。

已有真实视频参与过姿态和背景设计，只能作为开发诊断；后续汇报泛化仍需另外独立的视频/拍摄段。

## 7. 已完成的验证与限制

1. 93 项相关单元测试通过，覆盖旧模型、矩形裁剪、原图坐标还原、框外八点、背景合成、分组 batch、BN、局部峰、KL 数值/梯度，以及旧微调代码的评估接口。
2. GPU BF16 短训练：16 个训练实例、16 个验证实例、1 轮，完成梯度更新和 checkpoint 写入。训练循环报告约 14.2 秒；这不是正式速度估计。
3. checkpoint 重新加载后，对新姿态场景 16 个实例进行独立脚本评估，128 个点均保留、数值有限。
4. 使用已保存的固定 YOLO 框，运行矩形真实帧推理流程，输出实际 RGB 输入、热图和原图坐标。短训权重未收敛，不能拿它的预测当成改进效果。

正式训练未启动；旧权重和旧结果未替换。本轮没有解决所有外观、暗光、模糊、几何标签不确定性；也没有因为画出了八点而声称获得正确 6D 位姿。没有可靠真实内参时不强行执行 PnP。

## 8. 代码入口和运行方法

新增配置：`configs/single_rgb_rect_20260926.json`。

新增入口：

- `scripts/audit_rectangular_protocol.py`：检查几何可表示范围、记录排除项及划分。
- `scripts/train_single_rgb.py`：封装 `check/smoke/train`，复用已有 prepared 数据，记录代码哈希及命令。
- `scripts/evaluate_single_rgb.py`：按新坐标/指标协议评估合成数据。
- `tests/test_rectangular_protocol.py`：新协议测试。

修改的共用模块：`cropping.py`、`dataset.py`、`torch_data.py`、`background.py`、`online_background.py`、`net/resnet_fpn.py`、`net/loss.py`、`net/decode.py`、`net/__init__.py`；训练和真实帧推理入口为 `train_heatmap.py`、`infer_resnet_yolo_real.py`。旧参数默认值不变，`heatmap_margin=0` 的旧模型 state dict 不增加新层。

本地重做数据检查：

```bash
.venv-hccepose/bin/python scripts/train_single_rgb.py \
  --prepared outputs/continuous_pose_local_check_20260924/prepared \
  --output outputs/single_rgb_rect_20260926 \
  --action check
```

服务器应使用服务器上**完整的新姿态 prepared 目录**，不要误用上面的本地小批检查集。`online_config.json` 的背景、RGBA、基础数据路径必须指向服务器现有文件。操作示例（先在项目根目录设置实际路径，使用已有 PyTorch 环境）：

```bash
BB8_PREPARED=/absolute/path/to/full/prepared
python scripts/train_single_rgb.py --prepared "$BB8_PREPARED" \
  --output outputs/single_rgb_rect_r18 --action check
python scripts/train_single_rgb.py --prepared "$BB8_PREPARED" \
  --output outputs/single_rgb_rect_r18 --action smoke
python scripts/train_single_rgb.py --prepared "$BB8_PREPARED" \
  --output outputs/single_rgb_rect_r18 --action train
```

默认正式配置为 ImageNet 预训练 R18、FPN128、30 轮、batch32、BF16、AdamW 学习率 3e-4、骨干学习率系数 0.1、前 2 轮仅训练输出部分、余弦学习率。`--workers` 和 `--batch-size` 可根据 4090 实测调整；若比较骨干，保持两组相同设置。不要自动在失败后缩小数据或跳过报错样本。

ResNet50 同协议比较仅替换骨干和输出目录：

```bash
python scripts/train_single_rgb.py --prepared "$BB8_PREPARED" \
  --output outputs/single_rgb_rect_r50 --backbone resnet50 --action train
```

两个模型都从各自 ImageNet 权重初始化，不加载旧灰边模型，也不加载真实八点微调权重。目录中已有训练产物时脚本拒绝覆盖。

服务器完整数据如检查失败，先看 `geometry_audit.json` 的训练异常项。不能用验证/测试难例调参数，也不能盲目把输出热图边距扩大到数千格。

真实帧推理使用：

```bash
python scripts/infer_resnet_yolo_real.py \
  --checkpoint outputs/single_rgb_rect_r18/training/best.pth \
  --detections /absolute/path/to/fixed_yolo_detections.json \
  --output outputs/single_rgb_rect_r18/real_review
```

检测 JSON 使用已有格式：`predictions` 列表，每项包含 `source` 原帧路径、`xyxy` 框列表和 `confidence` 列表。推理从 checkpoint 读取尺寸组、边距、解码和损失语义。

旧 `evaluate_cleanblur5000.py`、旧视频脚本等入口仅支持历史裁剪协议，会拒绝本轮矩形 checkpoint；不要绕过其检查或手动把新模型改标为旧模式。新模型当前通过以上帧推理入口验证，整段视频入口尚未迁移。

测试命令：

```bash
PYTHONPATH=tests .venv-hccepose/bin/python -m unittest \
  test_rectangular_protocol test_net test_tight_crops test_online_background \
  test_background_augmentation test_real_adaptation test_bop_bb8 \
  test_crop_jitter test_blur_augmentation
```

下一步先完成同协议训练和独立评估。若仍失败，应分别检查合成验证、真实人工机身框、固定 YOLO 框下的八点误差，以及框内/框外角点的误差，才能区分特征泛化、检测框偏差和输出头的问题。
