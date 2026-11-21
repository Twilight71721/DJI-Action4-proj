# 真实视频背景与多样照片混合：单方案服务器训练

本包取代尚未执行的裁剪 A/B 方案。本次只训练一个新模型：随机背景 + 已验证的裁剪扰动。复用服务器已有5000清晰/5000模糊数据和5000模型best.pth，不重新渲染、不重新传输整套数据，不启动A/B任务，不删除历史文件。

## 已准备的数据

- 统一背景池1536张：Open Images/CVDF公共照片1024张，用户视频256帧中抽取的512个背景区域。
- 视频背景尽量绕开YOLO检测到的相机区域，包含墙面、桌面、衣物、手臂、地面等；这是自动排除策略，并非每个背景都有人工作绝对无相机保证。
- 用户要求不按来源或视频片段隔离，全部背景统一随机采样。文件保留哈希和来源信息仅供复现、外部照片署名，不用于分组限制。Open Images作者/原始链接/CC BY许可在augmentation/manifest.json；SOURCES.md说明下载和缩放处理。
- 合成目标仍使用原训练划分4000基础帧、每帧2实例、清晰/模糊配对，共16000实例/轮；合成验证500帧、测试500帧不并入训练目标。
- 8000个相机mask来自原渲染的完整机身投影mask：保留轮廓内原RGB，包括线缆、手指等遮挡留下的像素；只替换轮廓外的背景，不凭空恢复隐藏纹理。直接用mask_visib会把遮挡区挖成背景缺口，预览后已改用完整投影轮廓。采用原分辨率ROI存储以减少文件大小；mask不是bbox矩形。
- 模型输入不包含画出的框或绿色标签线。

## 固定训练方案

| 设置 | 值 |
|---|---|
| 正式训练次数 | 1次，不跑A/B |
| 初始权重 | 上一轮5000配对训练的best.pth，严格核对SHA256 |
| 网络/损失/解码 | 原ResNet50-FPN / focal+10×coordinate / sigmoid weighted Top-9 |
| 输入/热图 | 256×256 / 8×64×64 |
| 训练 | 30轮，batch32，每轮500次更新，BF16 AMP，全部解冻 |
| 学习率 | 新建AdamW，head1e-4、backbone1e-5，cosine，weight_decay1e-4 |
| 背景 | 80%随机替换，20%原图；从全部1536张均匀选择 |
| 背景额外变化 | 随机截取短边50%–100%的方形区域、随机水平翻转、亮度×0.8–1.2 |
| 裁剪扰动 | 中心±5% bbox宽高，padding[1.2,1.4]；图像/标签/K同步 |
| 模糊 | 清晰与模糊训练实例仍1:1，背景各自随机采样；不增加新的在线随机模糊 |
| 权重选择 | 只按原清晰验证集平均角点误差保存best.pth，另存last.pth |

模糊分支需要特别处理：当替换背景时，从服务器已有清晰原图取出相机，先合成新背景，然后按该图已有native_sigma/kernel在原裁剪尺度模糊，最后缩放为256。这样避免直接把已经混入旧背景颜色的模糊边缘粘到新背景上。未替换背景的20%继续使用原固定清晰/模糊图。

仅改变背景不会改变角点、热图、K、R、t；随机裁剪则由同一裁剪统一重算标签。背景采样由seed/epoch/scene/frame/instance确定，每轮变化且可复现。不要启用最终训练裁剪缓存或persistent_workers，以免冻结增强。

## 上传、解压、环境

只上传 `resnet5000_background_mixed_20260923.tar.gz` 和同名 `.sha256`。不要上传旧A/B包，不需要重新上传5000图片。

```bash
sha256sum -c resnet5000_background_mixed_20260923.tar.gz.sha256
tar -xzf resnet5000_background_mixed_20260923.tar.gz
cd resnet5000_background_mixed_20260923
sha256sum -c SHA256SUMS
```

使用上次成功的Python/Conda环境，例如原来的resnet3305环境，先核实实际名称。无新增训练依赖；本地准备背景时用过ultralytics，但服务器不需要它，也不需要下载外部照片。原依赖见requirements.txt。不要改驱动、系统CUDA、内核或代理。

历史服务器资产路径来自上次回传结果，请先检查是否还在：

```bash
BG_BASE=/home/aitest/exam-project/resnet5000_cleanblur_4090
BG_DATA="$BG_BASE/dataset"
BG_INIT="$BG_BASE/outputs/resnet5000_paired/best.pth"
BG_REAL="$BG_BASE/real/manifest.json"
BG_OUT="$PWD/outputs/background_mixed_30ep"
```

起点必须匹配：

```text
4872f6fd9253fa63ca7df2a5123246ca964f3d1b8a8026213061f370345cedab
```

如果资产移动了，只替换路径，不改数据或起点。增量包有augmentation/背景与mask，还有最近上传的frame0供结果展示，原来的真实诊断帧复用旧real/manifest.json。

## 检查与预览

```bash
nvidia-smi
python -c 'import torch; print(torch.__version__); print(torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported())'
python -m unittest discover -s tests -p 'test_background_augmentation.py' -v
python -m unittest discover -s tests -p 'test_crop_jitter.py' -v
python -m unittest discover -s tests -p 'test_net.py' -v
python scripts/run_background_training.py --dataset "$BG_DATA" --checkpoint "$BG_INIT" --real-manifest "$BG_REAL" --output "$BG_OUT" --action check
```

check校验起点、原10000图片及标注哈希、trainval映射、1536个背景和8000个mask。耗时包括磁盘读盘，不能跳过哈希失败。包内local_verification/preview/index.html已包含实际合成预览。可在服务器重新生成：

```bash
python scripts/preview_background_training.py --dataset "$BG_DATA" --background-assets augmentation --output outputs/server_background_preview
```

预览是数据增强示例，不是训练效果。完整投影轮廓会保留轮廓内原有遮挡物片段及少量边缘纹理，不等于纯可见相机语义分割。验证图像和标签一致性检查必须通过。

## 一次启动

在tmux中激活已有环境，进入新包目录，重新设置上述路径变量，然后：

```bash
python -u scripts/run_background_training.py \
  --dataset "$BG_DATA" --checkpoint "$BG_INIT" \
  --real-manifest "$BG_REAL" --output "$BG_OUT" \
  --workers 4 --action all
```

all执行：完整校验 → 1轮限量smoke（默认64训练实例）→ 单个正式30轮任务 → 原模型/新模型统一评估 → 自动打包回传。正式模型重新从原best.pth初始化，不从smoke权重继续。优化器重建，因此不是精确断点恢复。

日志位于 `$BG_OUT/training/train.log`，逐轮指标在metrics.csv，状态在status.json。用服务器每轮实际seconds估算剩余时间，不将本地32样本检查的耗时当成全量训练耗时。

若batch32 OOM，保留失败目录，换新BG_OUT并在所有命令加 `--batch-size 16` 重新运行；不要缩小输入或默默切换精度。如果smoke失败、非有限损失、缺mask或hash错误，报告具体原因，不跳过检查。

可分阶段使用 `--action check/smoke/train/evaluate/pack`。train要求同配置同增强资产的smoke通过。已有输出目录不覆盖。训练中断可保留失败目录并换新目录重新从原起点运行；当前不提供精确optimizer/scheduler/RNG断点恢复。评估中断则保留并重命名旧review，再用同参数--action evaluate，然后--action pack，无需重新训练。

## 结果和解释

评估只比较已存在的原模型与这次新模型，不另训控制组。报告包括清晰/模糊合成验证和测试、固定裁剪压力测试、原真实诊断帧和frame0输入/旧预测/新预测三列图。合成误差统一映射回padding1.3的256像素裁剪；保留所有原有效角点，报告P95与PCK@5、每family与逐实例NPZ。

按用户要求混合背景来源，真实视频源可能同时参与背景训练和展示，所以真实结果是效果展示，不是独立盲测。没有可信真实八角点真值，不能报告真实PCK或宣称“准确率提升X%”。框更规整也不自动等于角点正确。本次背景和裁剪同时改变且无A/B，不做单因素因果结论。合成目标的固定划分仍保留。

all结束在输出目录旁生成 `background_mixed_30ep.tar.gz` 和 `.sha256`。回传这两个文件即可；包含best/last、日志、配置、哈希、review和预测数组，不包含训练dataset，也不重复打包背景池。回传权重省略optimizer以减小体积；服务器原完整checkpoint保留。结果包可推理，不能作为精确断点恢复包。
