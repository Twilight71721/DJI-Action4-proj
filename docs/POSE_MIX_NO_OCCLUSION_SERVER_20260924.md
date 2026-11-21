# 新姿态＋多样背景：无额外遮挡的服务器训练包

这是增量包，复用服务器已有的 5000 张清晰／模糊数据和上一轮 `background_mixed_30ep` 的最佳权重。正式训练在服务器 RTX 4090 上进行。本地仅验证了少量样本和启动流程，不把本地 smoke 当成正式结果。

## 明确方案

| 项目 | 配置 |
|---|---|
| 额外遮挡 | **不添加**；保留旧训练图片本来存在的自然遮挡 |
| 新姿态素材 | 82 个模板、246 张 RGBA 前景及 mask、K/R/t、BB8 |
| 新背景 | 服务器下载／复用 5120 张 Open Images 照片，混合旧池 512 个真实视频背景裁片，正常共 5632 张 |
| 新数据 | 8000 组，每组同背景同姿态的清晰／模糊各一张，共 16000 张单相机图 |
| 旧数据 | 原 train 划分 4000 基础帧 × 清晰／模糊 × 两台相机 = 16000 个实例 |
| 正式每轮 | 合计 32000 个实例，新旧各半，整体清晰／模糊严格 1:1 |
| 网络／解码 | 保持 ResNet50-FPN、256 输入、64 热图、原损失及 Top-9 解码 |
| 裁剪 | 固定 bbox padding=1.3；**关闭裁剪扰动** |
| 新图变化 | 前景均匀缩放 0.75–1.05、平移 ±32 px，同步变换 K 和角点；不裁掉相机 |
| 模糊 | 合成后施加 Gaussian，256 输入等效 sigma 0.3–1.4；标签不模糊 |
| 起点 | 上一轮背景训练第 28 轮的 best.pth；重建优化器，不是精确断点续训 |
| 训练 | 单方案 30 轮，默认 batch32、workers8、BF16 AMP，lr1e-4，backbone lr1e-5，cosine，全解冻 |
| 权重选择 | 只用旧清晰验证集的平均角点误差；原测试集只用于最终评估 |

旧数据保留原 RGB，本轮只为新增姿态预合成随机背景，不启用旧版在线背景替换器。背景不按来源分 A/B，统一采样，来源信息仅用于追溯和许可署名。新背景组合固定 seed 生成一次，训练时打乱顺序；本轮不是每个 epoch 重新换背景。

## 包内和服务器已有文件

包内包含 `pose_assets/`、训练／合成／评估脚本、`bop_bb8/`、固定配置、数据元信息哈希、真实诊断帧及 bbox、回归测试、局部验证记录和说明。包内的绿色点预览不是训练 RGB。

**不重复上传**旧 5000 张清晰／模糊数据、模型权重和旧背景池。下面是上次服务器回传记录中的路径，OpenCode 必须先核对实际位置；移动了就改路径，不能重新划分数据、替换起点或悄悄减少背景数量。

```bash
POSE_BASE=/home/aitest/exam-project/resnet5000_cleanblur_4090
POSE_DATA="$POSE_BASE/dataset"
POSE_PREVIOUS=/home/aitest/exam-project/resnet5000_background_mixed_20260923
POSE_INIT="$POSE_PREVIOUS/outputs/background_mixed_30ep/training/best.pth"
POSE_OLD_BG="$POSE_PREVIOUS/augmentation"
```

服务器原始起点 SHA256：

```text
91251527dcc1c1d5bb51c20f0054e2264a39b00e096956db1b92820139a61ded
```

回传的去优化器版本也允许使用，SHA256 在配置的 `allowed_checkpoint_sha256` 中。不要用 smoke 权重或历史 3305 权重代替。

## 解压与环境

```bash
sha256sum -c resnet_pose_mix_no_occlusion_20260924.tar.gz.sha256
tar -xzf resnet_pose_mix_no_occlusion_20260924.tar.gz
cd resnet_pose_mix_no_occlusion_20260924
sha256sum -c SHA256SUMS
```

复用之前成功训练的 Conda/Python 环境。本轮无新增训练依赖，也不需要 Blender、Ultralytics、驱动或系统 CUDA 修改。`requirements.txt` 供核对，不要直接重装一个已经能运行的环境。

```bash
nvidia-smi
python -c 'import torch,cv2,numpy; print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported())'
python -m unittest discover -s tests -p 'test_pose_mix_data.py' -v
```

建议额外预留至少 20 GB 空间，实际取决于 PNG 大小。新数据目录内使用绝对符号链接引用旧数据；不要在准备后移动或删除旧数据和工作目录。

## 检查并启动

在新包目录中设置上面的 `POSE_*` 路径，再设置两个**互不嵌套的新目录**：

```bash
POSE_WORK="$PWD/work/pose_mix_data"
POSE_OUT="$PWD/outputs/pose_mix_no_occlusion_30ep"

python scripts/run_pose_mix_training.py \
  --dataset "$POSE_DATA" --checkpoint "$POSE_INIT" \
  --existing-backgrounds "$POSE_OLD_BG" \
  --workspace "$POSE_WORK" --output "$POSE_OUT" --action check
```

`check` 校验起点、旧数据划分/图片/几何元信息、246 张前景和真实诊断帧。任何哈希失败先报告，不能跳过。

在 tmux 中激活同一环境后运行：

```bash
python -u scripts/run_pose_mix_training.py \
  --dataset "$POSE_DATA" --checkpoint "$POSE_INIT" \
  --existing-backgrounds "$POSE_OLD_BG" \
  --workspace "$POSE_WORK" --output "$POSE_OUT" \
  --workers 8 --action all
```

`all` 依次执行：校验 → 下载/复用背景 → 合成新清晰/模糊配对数据 → 建立只读旧数据链接与训练合并目录 → 用新姿态样本做限量 smoke → 从原始起点正式训练 → 同一规则评估前后两个模型 → 压缩结果回传。

可以分阶段运行 `--action prepare`、`smoke`、`train`、`evaluate`、`pack`。完成 prepare 后检查 `$POSE_WORK/mixed/preview/index.html`；前景比例、方向、清晰/模糊效果及绿色角点都应合理。绿色点只写入 preview，不能出现在训练输入。若发现问题，保留数据和日志，报告具体例子后再修复，不能静默继续。

## 中断、显存与日志

- 背景下载脚本可复用已下载 JPEG。若网络失败，查看 `download_progress.json`，恢复后重跑 prepare；若 `mixed/` 已部分生成，则保留它并换一个新的工作目录进行合成。禁止覆盖已有正式训练目录。
- smoke 使用新增合成样本与旧验证样本；正式训练重新加载原始 best.pth，不继承 smoke 权重。
- 按项目服务器技能默认从 batch32 开始。确认 4090 显存有余量时，可以在 smoke 和后续所有阶段统一加 `--batch-size 64`；必须先通过 batch64 的 smoke，不能沿用 batch32 的结果。批量大小改变会改变每轮优化器步数，实际值写入配置。
- 若 batch64 OOM，保留原输出，改用新输出目录，并保留相同工作目录复用已合成数据；依次使用 `--action smoke/train/evaluate/pack`，各阶段统一加 `--batch-size 32`。不要再次使用 `all` 重复 prepare，不要删失败记录、缩小输入或改精度。配置不同必须重新通过 smoke。
- 默认不提供完整 optimizer/scheduler/RNG 断点恢复。训练被打断时保留原输出，按明确的新目录从原起点重新开始；不要声称是精确续训。
- 逐轮结果在 `$POSE_OUT/training/metrics.csv`，日志在 `train.log`，状态在 `status.json`。用服务器实际完成的前几轮 seconds 估算时长，不使用本地限量 smoke 时间推断正式耗时。

## 评估与回传

最终比较旧清晰/模糊 val、test 的平均误差、P95、PCK@5，以及固定框的压力测试和真实帧预测对照。压力测试中的裁剪变化仅用于评估，不表示训练启用了裁剪扰动。Before 是 background_mixed_30ep 的 best，trained 是本轮按清晰验证集选择的 best。

真实视频已参与姿态和背景设计，没有可靠八角点 GT，因此只能展示真实效果，不能报告真实 PCK 或宣称独立盲测提升。新姿态大部分为粗角度候选，背景多样化不证明所有真实姿态已精确覆盖。清晰/模糊和不同背景派生样本全部留在训练组，不把这些衍生图拆到旧验证／测试。

结束后回传：

```text
outputs/pose_mix_no_occlusion_30ep.tar.gz
outputs/pose_mix_no_occlusion_30ep.tar.gz.sha256
```

结果包含 best/last 推理权重、配置、日志、评估 HTML、JSON、预测数组和训练数据来源哈希；不包含大训练集和背景池，权重导出会去掉 optimizer 以减小传输。服务器原始权重和新合成数据保留。
