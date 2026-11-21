# 连续三维姿态与独立外观、灯光：4090 服务器执行说明

## 目的与数据定义

上轮新增网格把 left 外观绑定到 roll=0/180，把 right 外观绑定到 roll=90/270，验证姿态集中于 roll=45。本轮修正这一数据设计：姿态、外观、光照分别使用独立、可复现的随机数流。随机背景保留，并增加大量真实视频背景裁片。

- 新前景：4,096 个训练姿态、256 个固定验证姿态。大约一半从 SO(3) 均匀抽样，另一半在既有视频参考旋转附近抽样连续轴角扰动（各轴旋转向量分量标准差 18°）。使用真实三维旋转和透视投影，不是旋转二维图片。视频参考姿态是估计值，不能当真实位姿 GT。
- 两种已有相机外观（标签、标记与屏幕版本）独立抽样；机身 roughness=0.45–0.80、specular=0.12–0.30 也独立于姿态。这里不声称已生成无限种纹理。
- 灯的位置、面积、颜色、能量与曝光独立于姿态和外观抽样，参数保存在每条渲染记录中。背景在每个 epoch 在线独立抽样。
- 本地已经完成 64 个样本的渲染与标签检查。**4,352 个正式前景需要在服务器渲染**。包内 pose_assets 是已确定的渲染计划和模型元数据，尚非完整前景库；渲染完成才会生成 manifest.json 和 SHA256SUMS.json。
- 同一批新验证姿态到训练姿态的最近旋转角差中位数：原248网格约21.12°，新4096采样约7.27°；新方案最大仍约24.18°。这项比较只针对原248网格与新4096前景库，不含旧5000数据和旧视频前景。这是采样覆盖的几何统计，不是模型效果，也不表示覆盖了所有可能姿态。
- 真实背景增量：1,024 个视频帧，每帧 8 张，共 8,192 张 384×384 JPEG90 裁片。保留墙面、衣服、手臂、地砖等背景；避免检测框及邻近帧检测框，并扩大25%+24像素。自动排除不能保证零漏检，预览和溯源保留。视频背景已参与适配，真实视频不再是独立盲测。
- 与服务器原5632背景（5120网上照片、512真实裁片）合并后，去重前13824张，约63%来自真实视频。统一随机抽取，不做来源A/B两组训练；原网页作者和许可元数据保留。若服务器旧池数目不同，以合并manifest为准。
- 背景划分按原始帧分组、按图片哈希去重，确保同帧裁片不横跨背景训练/验证；相邻视频帧仍可能相似，不声称独立视频测试。
- 保留旧5000数据原划分。旧训练16000实例＋新增12000清晰/模糊对=每轮40000实例。新增4096姿态全部被采样；同姿态不同对与不同epoch会换背景。姿态/外观/光照在这4352次渲染时采样，此后每epoch重新抽样的是背景和模糊程度，**不是每epoch重新渲染**。
- 清晰/模糊1:1。新增模糊按256输入尺度 sigma 0.3–1.4；先合成RGB后模糊，八角点位置及监督热图保持原定义。旧样本继续原标定模糊参数。
- 不加人工遮挡，旧图已有自然遮挡保留；不加裁剪扰动，bbox padding=1.3。保持ResNet50-FPN、256输入、64热图、原损失和Top9解码。极低响应解码分母截断问题未在本包修改。

## 解压与环境

解压到新目录，不覆盖以前结果：

```bash
sha256sum -c resnet_continuous_pose_background_20260924.tar.gz.sha256
tar -xzf resnet_continuous_pose_background_20260924.tar.gz
cd resnet_continuous_pose_background_20260924
sha256sum -c SHA256SUMS
```

训练复用服务器上次成功运行的 PyTorch CUDA 环境。渲染需要 Python 3.10、bpy 3.6.0、包内定制 BlenderProc；独立环境可避免影响原训练环境。不需要修改CUDA、驱动或系统配置。

```bash
# 使用已安装的 conda；若没有，使用现有 Python3.10 的 venv。
conda create -n action4-render310 python=3.10 -y
conda run -n action4-render310 python -m pip install -r requirements-continuous-render.txt
```

本地验证使用已有Python3.10/bpy3.6环境；没有远程验证服务器驱动、EGL/OptiX和新建环境。若依赖或渲染失败，保留日志并解决具体错误，不可跳过mask和投影审计、不改用二维旋转冒充三维。

## 渲染

先跑第一个24张分组核查环境、显存和用时：

```bash
conda run --no-capture-output -n action4-render310 python scripts/render_pose_replacement_pilot.py render --output pose_assets --group continuous_left_000 --samples 16
```

成功后执行全部；已完成且存在metadata的分组会跳过。中断分组会保留为interrupted目录，然后重新渲染，既有结果不删除。

```bash
conda run --no-capture-output -n action4-render310 python scripts/build_continuous_pose_bank.py --action render --output pose_assets
```

日志在pose_assets/continuous_*.log。最终应得到4352条manifest，4096 train、256 val；角点投影误差<0.001px、256裁剪标签全有效，RGBA和原生mask均存在。审计失败不要训练。预览在pose_assets/review/index.html。每组不超过24张，避免一次保存4352张深度图占用内存。

渲染耗时应由4090首组实测外推。此次本地64张约233秒，仅供流程参考，不可直接当4090预计耗时。服务器这轮还需渲染，不能只估训练时间。

## 混合背景

下面路径是此前服务器目录示例，需要按真实位置确认：

```bash
CONT_BASE=/home/aitest/exam-project/resnet5000_cleanblur_4090/dataset
CONT_MASKS=/home/aitest/exam-project/resnet5000_background_mixed_20260923/augmentation
CONT_OLD_BG=/home/aitest/exam-project/resnet_pose_mix_no_occlusion_20260924/work/pose_mix_data/backgrounds
CONT_CHECKPOINT=/home/aitest/exam-project/resnet_multiview_online_background_20260924/outputs/multiview_online_background_30ep/training/best.pth

# 在原训练Python环境运行。这里只建立链接，不上传、复制或下载已有网上背景。
python scripts/prepare_real_background_extension.py --action merge --existing "$CONT_OLD_BG" --extra real_backgrounds --output work/mixed_backgrounds
```

起点必须是最近完成的multiview_online_background_30ep/best.pth；允许服务器原始checkpoint和去优化器回传版两种SHA，见config。不要因旧文件同名而用错权重。原始SHA应为`b5d44e64e4b7305bb39bd233912dfcdfd43dbc9547d24af13b070bcccbef69ce`。

## 检查与训练

```bash
python -m unittest discover -s tests -p 'test_continuous_pose_data.py' -v
python -m unittest discover -s tests -p 'test_online_background.py' -v

CONT_ARGS=(
  --config configs/continuous_pose_background_20260924.json
  --dataset "$CONT_BASE" --checkpoint "$CONT_CHECKPOINT"
  --assets "$PWD/pose_assets" --backgrounds "$PWD/work/mixed_backgrounds"
  --old-background-assets "$CONT_MASKS"
  --workspace "$PWD/work/continuous_data"
  --output "$PWD/outputs/continuous_pose_background_30ep"
  --batch-size 32 --workers 8
)
python scripts/run_online_pose_training.py "${CONT_ARGS[@]}" --action all
```

依次完整校验→准备固定验证集和在线数据索引→短训练检查→30轮训练→评估→打包。BF16、学习率1e-4、骨干1e-5、cosine、全部解冻。不启用缓存或persistent workers，保证epoch背景更新。已有工作目录不可再次用all覆盖，可分别执行check/prepare/smoke/train/evaluate/pack阶段。训练不完整、样本数不对、非有限loss或smoke契约不匹配会停止。

验证：旧清晰1000实例＋新256姿态×4背景×清晰模糊=2048实例，合计3048。仍按合并平均角点误差选best，因此新姿态实例权重约67%，比上一轮更高；报告时须同时看两部分，不能直接横比混合val总分。新集合被用于选权重，**不能称作独立最终测试**。另评估原固定clean/blur test和原真实帧，并对每个新验证姿态报告跨背景误差及预测波动。

## 返回结果

回传 `outputs/continuous_pose_background_30ep.tar.gz` 及 `.sha256`。包含best/last（去优化器）、完整30轮指标、配置和SHA契约、旧测试和真实帧结果、新姿态验证图与指标、前景及背景manifest。不要回传整个渲染临时目录，不要只发loss截图。真实帧没有可靠八角点GT，只作定性诊断。

本地核查在local_verification目录，GPU短训练只是流程检查，不是正式训练。正式效果必须看服务器本轮与起点的同集对照。
