# 成像增强与视角采样

## 本轮范围

保留固定 YOLO、ResNet18-FPN、矩形输入、空间 KL 损失与原始角点编号。新增纯 RGB 成像增强、六方向审计和有上限的视角加权。没有新增遮挡、裁剪扰动、真实角点监督或背景一致性损失；没有修改已有正式权重。

## 成像增强

实现位于 `bop_bb8/photometric.py`，仅使用已有 NumPy、OpenCV 和 PyTorch，无新增依赖。

先以20%概率直接返回原图；其余样本依次独立抽取：

| 操作 | 条件概率 | 参数 |
|---|---|---|
| 颜色/曝光组 | 80% | 增益0.7–1.3，对比度0.75–1.25，gamma0.8–1.25，饱和度0.7–1.25，逐通道增益0.92–1.08 |
| 运动模糊 | 15% | 居中的3或5像素线核，随机方向 |
| 高斯噪声 | 30% | 8bit标准差0.5–3.0 |
| JPEG压缩 | 30% | 质量65–95 |

参数是首轮实验设置，不是从真实拍摄标定出来的最优参数。运动模糊施加在当前网络输入尺度，不能直接视为原视频像素。已有清晰/模糊配比保留。没有旋转、缩放、平移或透视扭曲，几何标签不变。训练数据归一化被反变换为RGB，增强后重新归一化；验证和普通推理不启用增强。若与配对背景同时使用，配对图共用成像随机数；本轮不启用配对训练。

## 视角定义与采样

观察方向使用物体坐标中的 `-R.T @ t - center`，最大绝对分量分为X±、Y±、Z±。根据已有渲染器坐标约定，X对应左右表面，Y对应上下表面，Z对应正背面。不能把X正负直接解释成真实视频的图像左右，更不能把六个粗分箱当作覆盖所有姿态的证明。

服务器回传的4096个独立训练姿态前景分布：X+1034、X-993、Y+515、Y-446、Z+481、Z-627。两侧X合计2027，约49.5%。这是**前景姿态数**，不是39996条最终训练实例数。本地只有一小部分连续姿态资产，统计必须独立报告，不能当作服务器完整训练集分布。

加权规则：`weight = min(sqrt(max_bin_count / bin_count), 3)`。在每个矩形尺寸分组内部，按该权重有放回抽取与原来相同数量的实例；每轮图片数、批次数不变。它增加稀缺方向的曝光，但不保证六方向全局等量，也不能生成缺失角度。在某个尺寸分组中只有一个方向时，无法改变该分组的方向分布。每轮实际抽样数量存入 `view_sampling_epochs.jsonl`。

## 训练配置与复现

新增命令行开关：

- `--photometric-augmentation`
- `--view-balance --view-weight-cap 3`
- `--deterministic-cudnn`：关闭cuDNN自动选算法和TF32，固定cuDNN确定性设置。

三个开关默认关闭，历史训练默认行为保留。

本地对照：

```bash
.venv-hccepose/bin/python scripts/run_photo_view_pilot.py \
  --output outputs/photo_view_pilot_20260926
.venv-hccepose/bin/python scripts/report_photo_view_pilot.py \
  --output outputs/photo_view_pilot_20260926
```

输出目录必须不存在。三组同初始最佳权重、512个训练实例、512个验证实例、3轮、学习率3e-5、batch-size8。每轮67次更新，固定最后一轮比较：control、photo、photo_view。加权组保持预算一致，但允许重复采样。这是小规模继续训练，不是从头完整训练。

完整服务器实验配置已准备，尚未运行：

- `configs/single_rgb_rect_photo_20260926.json`：仅成像增强。
- `configs/single_rgb_rect_photo_view_20260926.json`：成像增强加视角加权。

在服务器现有完整数据上，使用项目已有 `scripts/train_single_rgb.py`，传入 `--prepared` 完整数据目录、`--config` 上述配置、一个新的 `--output`，以及 `--action train`。配置延续30轮、ImageNet初始化的完整协议，未自动改成从本地小实验权重继续训练。

后续交付：`resnet18_photometric_20260926.tar.gz` 使用单独的 `single_rgb_rect_photo_balanced_20260926.json`，不是上述pilot配方。新增balanced配置保留20%原图、约68%轻度/12%中等成像增强，视角加权关闭；原pilot默认参数保持不变以便复现。服务器入口已支持显式`--config`、增强预览和配置校验。详细操作见 `docs/PHOTOMETRIC_SERVER_20260926.md`。完整服务器训练收益尚待验证。

## 验证

```bash
PYTHONPATH=tests .venv-hccepose/bin/python -m unittest \
  test_photometric_views test_paired_background test_online_background \
  test_rectangular_protocol test_net
```

报告包含普通验证、固定随机种子的增强验证、六方向误差、21个真实检测定性对照。增强验证仍是模拟成像变化，不能作为真实准确率。真实图上的点更分散也不等于编号和位置正确。旧正式权重继续保留。
