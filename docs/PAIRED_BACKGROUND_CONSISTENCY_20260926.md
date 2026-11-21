# 配对背景一致性训练

本实验保留 ResNet18-FPN、紧矩形 RGB 输入、八点编号、输出热图边缘、局部峰值解码和固定 YOLO。仅修改训练期的样本组织与损失。

## 数据与损失

每个基础实例产生两张图片：同一几何标签、裁剪、尺寸、前景和模糊参数，背景来源不同。旧的 mask 合成与新的 RGBA 合成都支持配对。第一张与原在线采样行为一致；第二张使用独立的、可重现的背景随机种子。变化只发生在背景及其合成边缘；模糊后边缘像素自然会受背景颜色影响。

每对图片在同一个模型中前向传播。对每个有效的编号角点，在空间维度归一化得到分布 P 和 Q；M=(P+Q)/2。

    supervised = (KL(GT || P) + KL(GT || Q)) / 2
    consistency = (KL(P || M) + KL(Q || M)) / 2
    total = supervised + consistency_weight * consistency

KL 与 JS 均在 FP32 对数概率中计算，按有效角点平均。JS 使用自然对数，理论范围为 [0, ln(2)]，两侧都反向传播。无效角点不参与 JS。两个输出完全相同但都错误时，JS 为零，真值监督仍会惩罚错误；不能取消真值监督。

## 训练入口

`scripts/train_heatmap.py` 新增以下参数，默认关闭配对，不改变原单图训练：

- `--paired-background`：需要已有 `--online-background-config` 且损失为 `spatial_kl`。
- `--consistency-weight 0`：两张图均受真值监督的公平对照。
- `--consistency-weight 0.1`：本轮的初始一致性权重，尚未经大规模验证。

`--batch-size` 表示基础实例数，实际前向输入图片数是其两倍。例如 batch-size=8 为16张图片。配对模式关闭 cuDNN 自动算法选择，使用确定性 cuDNN 算法，并关闭 TF32。未承诺跨设备或跨版本逐位相同。

`paired_terms.jsonl` 记录每轮监督损失、JS、实例数、RGB 数量和优化步数；原 `metrics.csv` 的验证 loss 仍是单图真值监督，以便比较。验证与推理不需要第二张图。

`scripts/train_single_rgb.py` 已有配置透传能力；在矩形训练配置中增加 `paired_background: true` 和 `consistency_weight: 0.1` 即可传到训练入口。旧服务器整套校验和归档流程尚未打包为本实验专用版本，本轮不将小实验权重替换为正式模型。

## 本地小实验

```bash
.venv-hccepose/bin/python scripts/run_paired_background_pilot.py \
  --output outputs/paired_background_pilot_20260926
.venv-hccepose/bin/python scripts/report_paired_background_pilot.py \
  --output outputs/paired_background_pilot_20260926
```

输出目录必须不存在。两组从现有第28轮最佳 ResNet18 权重初始化，使用本地可用合成数据中的512个训练实例和512个验证实例，3轮，学习率3e-5，batch-size=8。唯一配置差别为 JS 权重0或0.1。两组均使用相同的配对图像和相同优化预算。比较最后一轮，不能按真实帧挑选权重。

这不是从头训练，也不是服务器完整连续姿态集的实验。背景敏感性探针使用32个已见训练几何，重新放到与训练来源路径隔离的背景上，报告换背景的角点位移、JS，以及两张图相对真值的误差。它不等于独立姿态泛化评估。真实21个检测只做定性对照，没有可信全八点真值，不报告真实准确率。

## 检查

```bash
PYTHONPATH=tests .venv-hccepose/bin/python -m unittest \
  test_paired_background test_online_background test_rectangular_protocol test_net
```

测试包含两类前景来源的配对标签、相同模糊参数、可重现采样、原始第一视图不变、JS对称性、极端数值、双侧梯度、无效角点，以及“错误但一致”的结果仍有监督损失。
