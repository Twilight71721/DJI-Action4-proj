# 成像增强服务器实验：OpenCode 操作指南

## 目的与决策

此次只增加在线成像增强，沿用固定 YOLO、ResNet18-FPN128、紧矩形 RGB 输入（无灰边）、输出热图 margin=32、空间 KL 损失及 local 解码。无新增遮挡、裁剪扰动、几何扭曲、视角加权、配对背景一致性或真实角点监督。

上一轮本地512实例/3轮实验，成像增强使模拟退化验证误差6.13→4.93、PCK5 80.93%→84.33%，普通验证误差4.16→4.44。它提示增强有价值，也提示过强增强的取舍；不能作为真实视频已改善的证据。因此本包采用**高覆盖、主要轻度**的新 `balanced` 配方。此新配方没有被那次小实验验证，最终收益须由本次完整训练检验。

每个训练实例每轮独立抽取，理论比例：20%不额外增强，68%轻度，12%中等。不固定挑出一部分图片；同一图片跨轮可获得不同参数。不产生几万张磁盘副本，训练集长度不变。“不增强”是保留原有在线背景/清晰模糊数据结果，不保证每张都是清晰图。

增强在紧裁剪并缩放到实际输入尺寸后、ImageNet归一化前等价施加。数据包装器先精确恢复uint8 RGB，完成处理后重新归一化；保持标签、内参坐标变换和热图原样。

| 操作 | 轻度 | 中等 | 概率 |
|---|---|---|---|
| 亮度增益 | 0.85–1.15 | 0.7–1.3 | 被增强样本全部 |
| 对比度 | 0.9–1.1 | 0.75–1.25 | 同上 |
| gamma | 0.9–1.1 | 0.8–1.25 | 同上 |
| 饱和度 | 0.85–1.15 | 0.7–1.25 | 同上 |
| 逐通道增益 | 0.96–1.04 | 0.92–1.08 | 同上 |
| 居中运动模糊 | 3像素线核 | 3或5像素线核 | 增强样本中10%，全体约8% |
| 高斯噪声 | std 0.5–1.5 | std 0.5–3.0 | 增强样本中25%，全体约20% |
| JPEG | quality85–95 | quality65–95 | 增强样本中20%，全体约16% |

噪声单位为8bit像素值，模糊核单位为网络输入像素；后面三项独立抽取，可能叠加。原有清晰/模糊配比保留，因此额外模糊采用较低概率。八个角点即使被模糊影响也仍保留监督。不往输入图上画标签；框和角点仅用于预览叠图。增强不是锐化或修图，也不能保证消除合成到真实的差异。

验证、清晰/模糊测试和真实推理均**不新增成像增强**，防止改变基准和挑选有利样本。包内本地预览只检查实现，不用其评价模型。

## 包内与外部依赖

本包是增量训练工程，含全部所需Python模块/脚本/测试、配置、ImageNet ResNet18权重、冻结YOLO权重、11张真实诊断RGB帧、旧基线指标和操作说明。没有复制服务器已有的大批图片；不能当作完整数据备份。

复用：

```text
/home/aitest/exam-project/resnet_continuous_pose_background_20260924/
  work/continuous_data/online_config.json
  work/continuous_data/trainval/
  work/continuous_data/models/models_info.json
  pose_assets/manifest.json
  work/mixed_backgrounds/manifest.json
```

其 online_config 继续引用 resnet5000_cleanblur_4090/dataset、resnet5000_background_mixed_20260923/augmentation 和 resnet_pose_mix_no_occlusion_20260924/work/pose_mix_data/backgrounds 等旧目录。保留这些目录/链接。程序核对历史哈希和资源存在性，不得通过删检查、改划分、重建小数据绕过缺失。

原始train40000实例（原5000图中的训练场景构成16000实例，加新连续姿态24000实例），可见度过滤后上一轮39996。固定val3048；原清晰test1000、模糊test1000。4,096训练前景姿态、256验证姿态、13,824背景记录；记录数不等于独立场景数。保持原场景分组，不把测试素材移入训练。真实视频曾参与背景与姿态设计，11张真实帧只是开发诊断，没有可信八角真值指标。

## 启动

在压缩包所在目录：

```bash
sha256sum -c resnet18_photometric_20260926.tar.gz.sha256
tar -xzf resnet18_photometric_20260926.tar.gz -C /home/aitest/exam-project
cd /home/aitest/exam-project/resnet18_photometric_20260926
sha256sum -c SHA256SUMS
conda activate action4-bb8
nvidia-smi
PYTHONPATH=tests python -m unittest test_photometric_views test_paired_background test_online_background test_rectangular_protocol test_net test_rectangular_server test_pnp
```

使用已有torch2.8.0+cu128、torchvision0.23.0+cu128、Ultralytics8.3.233环境，无需imgaug或新依赖。不要改系统驱动。要求CUDA及BF16，不能退回CPU。缺失依赖按requirements记录处理；包内ImageNet权重避免联网下载。

建议在tmux里运行下面**完整命令**，不要省略 `--config`（默认配置是旧的不增强训练）：

```bash
python -u scripts/run_rectangular_server.py --stage all \
  --config configs/single_rgb_rect_photo_balanced_20260926.json \
  --output outputs/rect_r18_photometric_30ep
```

流程：完整数据/几何审计 → 实际增强输入预览 → 64train/64val短训1轮 → 全量30轮 → 完整合成评估及固定YOLO真实帧推理 → HTML报告 → 回传包。正式训练从包内ImageNet初始化，预测头重新初始化，**不从小实验/短训/真实微调权重继续训练**。这与旧30轮正式基线的初始化协议一致。

4090默认batch32、workers8、BF16、AdamW lr3e-4、backbone lr倍率0.1、cosine、前2轮冻结骨干，BN统计冻结。增强在线执行会增加CPU工作量，以前两轮实测估时，不能承诺固定训练时长。

```bash
tail -f outputs/rect_r18_photometric_30ep/training/train.log
tail -n 4 outputs/rect_r18_photometric_30ep/training/metrics.csv
```

程序记录配置、源码哈希、训练来源、所有轮次指标；检查增强确实开启且profile为balanced。输入预览固定同一RGB背景展示三轮成像参数，实际训练照常逐轮随机背景。原RGB与增强图的几何标签逐项比较。

逐阶段运行时，将 `--stage all` 替换为 `check`、`smoke`、`train`、`evaluate`、`pack`，其余参数保持相同。成功check后不要再次all覆盖现场，从下一阶段继续。训练中断不支持优化器自动恢复；保留现场，在新output重新跑。评估失败先保留/重命名部分evaluation和real目录再重试evaluate。

若OOM，保留日志，在新output增加 `--batch-size 16` 从check开始；所有阶段参数一致，并说明batch变化对可比性的影响。不要减分辨率、轮数或数据来跑通。数据迁移可以指定 `--previous`，但外部绝对路径及哈希必须仍然有效；缺失请反馈。

## 评估和回传

best依据未新增增强的完整验证集原始八点误差选择。报告保留原始编号连线，真实数据不做假内参PnP或重新连线美化。合成PnP仅作已有辅助指标，必须优先比较原始八点。

对比 reference_baseline/summary.json 的同一划分和 `body_long_side_256` 尺度：完整val3048、清晰test1000、模糊test1000。PCK5阈值是在目标框长边归一到256后的5像素。不要混用历史正方形大裁剪指标，不用真实帧挑权重。参考baseline文件只是之前结果，没有伪装成本轮重跑对照。

重点检查：普通验证/清晰测试是否下降，原模糊测试是否改善，全部真实检测的点位及编号是否改善、是否还聚集。真实帧没有可信八角真值，不报告虚构PCK。“更分散/更像框”不能独立证明精度提升。若收益与损失冲突如实报告，不自动替换旧正式模型。

完成后传回：

```text
outputs/rect_r18_photometric_30ep.tar.gz
outputs/rect_r18_photometric_30ep.tar.gz.sha256
```

包含best/last推理权重、每轮日志、数据/增强预览、配置与来源、合成预测指标、真实帧/YOLO框/实际输入/八点/热图、HTML和SHA256SUMS。原始完整检查点保留在服务器；回传权重去除optimizer减小体积。训练结束需报告best epoch、三类合成指标、全部真实帧观察、包大小与哈希。

LOCAL_PACKAGE_QA.json记录本地包检查范围；本地短训只验证执行路径，不代表本次30轮已完成或balanced配方已有收益。
