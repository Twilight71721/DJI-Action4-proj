# BB8 v4 数据与 ResNet-FPN 管线审计

## 基础约定变更

本管线以 `dataset/action4-v4-random-validation-100` 为当前数据源。角点顺序已改为该
数据的 `models/corners_3d.json` 顺序：0→1 沿 +x，1→2 沿 +y，4–7 对应 z=max。
旧代码的 0→1 沿 +y 顺序不再用于新模型 checkpoint。

## 数据结论

- BOP 根目录：`dataset/action4-v4-random-validation-100/train_pbr`；
- 模型信息：同级外层目录的 `models/models_info.json`；
- 100 张 `640x480` PNG，单场景、每帧一个实例；
- `cam_t_m2c` 与 v4 模型均已是 mm，不需要额外尺度修正；
- 直接读取 4 个样本（0、17、59、98）并重新投影，裁剪前二维角点与渲染器保存的
  `corner-projections.json` 对齐到 `1e-3 px`；
- 深度、mask 与 mask_visib 均非空，但这批数据没有真实遮挡，`visib_fract` 约为 1。

旧 `dataset/train_pbr` 的审计仍有参考价值：它使用了居中后的原始 GLB，并按
1000 mm/Blender unit 写出位姿，而非 70.5 mm PLY。网格顶点投影与 bbox 的平均 IoU
为 0.989、RMSE 为 0.83 px。它不再是默认训练数据，因此没有对原始 JSON 做原地修改。

## 训练设计

全图 stride-4 会把尺寸变化和位置搜索同时交给仅 100 张数据。当前采用实例级任务：

1. 从 `bbox_obj` 生成 1.3 倍方形 crop；
2. crop/resize 到 `256x256`，同步左乘仿射矩阵更新 K；
3. 输出 `8x64x64`、sigma=2 的标准 Gaussian 热图；
4. ResNet18 的 layer1..4 通过 FPN 融合到 stride 4；
5. 最终 heatmap head bias 初始化为 -2.19；
6. 默认损失为 CornerNet focal 加 10 倍归一化可微坐标 Smooth-L1；
7. 解码后乘以 4 回到 crop 输入坐标，PnP 使用同步更新后的 `K_input`。

实例裁剪意味着真实视频推理时需要检测框、跟踪框或人工 ROI。若最终系统必须直接处理
全图，应另加检测器，不能把当前姿态网络误称为同时完成检测。

## 验证结果

```bash
.venv-hccepose/bin/python -m unittest discover -s tests -v
```

37 项通过。测试覆盖 PNG、实例裁剪、K 更新、v4 角点顺序、渲染器参考投影、Gaussian
峰值、网络张量、loss 梯度和解码。

10 epoch、80/20 固定随机划分的 smoke 结果：

- 旧 focal-only：20 epoch 最佳验证误差约 94.73 px，出现全背景倾向；
- composite：第 1 epoch 105.36 px，第 3 epoch 48.77 px，最佳第 8 epoch 41.38 px；
- 第 8 epoch PCK@5 为 0.169。

结果证明端到端梯度和角点位置监督有效，但 80 张训练图不足以得到可部署精度，也不能
用于声称真实视频泛化。正式训练前仍需更多渲染数据、真实视频分布统计、遮挡/支架/手臂
建模以及独立场景划分。
