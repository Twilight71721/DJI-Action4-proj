# 多角度相机＋在线随机背景：服务器执行说明

目标：把姿态与背景独立采样，让同一相机姿态持续接触不同背景。不是继续上传大量离线合成图片。正式训练仍在4090服务器进行，本地只渲染前景、检查数据及执行短流程测试。

## 本轮数据

- 新增272张原生三维渲染：248个训练方向／旋转组合，24个留出组合。训练网格覆盖方位0–330°（30°步长）、俯仰−60/−30/0/30/60°和接近正上／正下、画面内旋转0/90/180/270°。留出角度在网格之间。采样是离散覆盖，不宣称覆盖连续空间每个姿态或提供真实视频逐帧GT。
- 保留此前246张视频导向前景。合计494张训练前景＋24张留出前景。原生mask、K/R/t和八角点来自同一次渲染，没有对渲染图片作假三维旋转。
- 旧5000数据保留原分组。原训练4000基础帧×2清晰模糊×2实例=16000实例；新增8000清晰模糊对=16000实例，每轮共32000实例。
- **所有训练实例都换背景**。旧样本从原清晰RGB和既有机身mask重建，新样本用RGBA alpha合成。每轮重新采样背景图片、背景裁片及翻转；清晰和模糊伙伴使用同一背景。不会只给新姿态换背景，也不会每轮重复固定组合。
- 背景直接复用服务器上一轮大池（通常5632张），不重新上传。随机留32张图片给新增验证集，其余混合使用；背景来源不分A/B。每轮有限抽样不保证所有背景都被访问，来源元信息仍保存。
- 保留清晰／模糊1:1。新模糊sigma在256裁剪下为0.3–1.4，旧模糊按原标定参数；先合成再模糊，几何标签不模糊。
- 不新增人工遮挡。旧图本来存在的自然遮挡会保留。裁剪扰动仍关闭，padding=1.3；不改变网络结构或损失。
- 新增固定验证：24个未训练姿态×4个未训练背景×清晰模糊=192实例。与旧清晰验证1000实例合计1192，用合并平均角点误差选best。另分别报告旧val/test和新姿态跨背景误差及波动。此新集合用于验证选权重，不能称为最终独立测试集。

## 既有解码限制

本包以数据方案改动为主，保留原Top9解码及模型接口；已发现的极低响应分母截断问题仍存在。真实图出现角点飞向左上角时，应检查热图响应，不能直接归因于背景。不要把本包称为解码修复版。固定姿态跨背景的稳定性也不能替代GT误差。

## 解压与环境

使用上次已能成功训练的Python环境，不重装CUDA／驱动。将增量包解压到新目录，保留以前数据和结果。

```bash
sha256sum -c resnet_multiview_online_background_20260924.tar.gz.sha256
tar -xzf resnet_multiview_online_background_20260924.tar.gz
cd resnet_multiview_online_background_20260924
sha256sum -c SHA256SUMS
python -m unittest discover -s tests -p 'test_online_background.py' -v
python -m unittest discover -s tests -p 'test_background_augmentation.py' -v
```

## 查找服务器已有文件

按实际路径设置变量，下列为此前服务器目录示例，路径不存在就定位对应文件，不能换成不相关数据。

```bash
ONLINE_BASE=/home/aitest/exam-project/resnet5000_cleanblur_4090/dataset
ONLINE_OLD_ASSETS=/home/aitest/exam-project/resnet5000_background_mixed_20260923/augmentation
ONLINE_PREVIOUS=/home/aitest/exam-project/resnet_pose_mix_no_occlusion_20260924
ONLINE_BACKGROUNDS="$ONLINE_PREVIOUS/work/pose_mix_data/backgrounds"
ONLINE_CHECKPOINT="$ONLINE_PREVIOUS/outputs/pose_mix_no_occlusion_30ep/training/best.pth"
ONLINE_WORK="$PWD/work/online_data"
ONLINE_RESULTS="$PWD/outputs/multiview_online_background_30ep"
```

`ONLINE_BACKGROUNDS/manifest.json`需包含至少5000张已下载背景；`ONLINE_OLD_ASSETS/camera_masks/manifest.json`为旧实例mask。起点为本次回传pose_mix的第30轮best，允许原始和去优化器回传版本的两个哈希，见config。不从头随机初始化。

## 执行

```bash
ONLINE_ARGS=(
  --dataset "$ONLINE_BASE"
  --checkpoint "$ONLINE_CHECKPOINT"
  --backgrounds "$ONLINE_BACKGROUNDS"
  --old-background-assets "$ONLINE_OLD_ASSETS"
  --workspace "$ONLINE_WORK"
  --output "$ONLINE_RESULTS"
  --workers 8 --batch-size 32
)
python scripts/run_online_pose_training.py "${ONLINE_ARGS[@]}" --action all
```

依次校验、准备链接与固定验证、短训练检查、正式训练30轮、旧集／真实帧评估、新姿态跨背景评估、打包。BF16，lr1e-4，骨干lr1e-5，cosine，全部解冻。不要额外启用全量缓存或持久化DataLoader worker，否则可能破坏随epoch变化的背景采样。当前DataLoader每轮重建worker，seed可复现。

也可分别使用 `check / prepare / smoke / train / evaluate / pack`。已有阶段成功就执行后续阶段，不要重新用`all`覆盖工作目录。失败查看对应日志，保留错误信息；修复后使用新结果目录，不把smoke权重当正式best。

旧数据及RGBA通过绝对符号链接复用，准备后不要移动它们。每轮32000实例不代表32000独立三维姿态。在线读取和合成会增加CPU负载，实际耗时以第一轮记录为准；不能直接保证比上一轮44分钟更快。

## 必须回传

完成后回传`outputs/multiview_online_background_30ep.tar.gz`及其校验文件。包内包括best/last权重、训练日志、配置、起点哈希、真实帧对照、旧集指标、pose_holdout指标、合成配置和背景清单。报告新旧数据是否都换背景，以及各项误差，不能只报loss或挑选最好看的框。

真实视频已参与背景和姿态设计，真实帧无可靠八角点GT，仍作定性诊断。网络是否减少了背景依赖，要结合固定姿态跨背景的误差和预测波动判断，不能在训练前保证。
