# 任务：修复 v6 渲染的 3 个已知问题，并生成正式 5000 张训练集

仓库：`C:\exam-project\Blender-repo`（先运行 `git status`，保护现有未提交修改）
参考数据：`dataset\action4-v6-domain-pilot-200`（不要覆盖、不要删除；v4/v5 数据同样不要动）

## 背景（Linux 端对 v6 pilot 200 张的复核结论，已确认）

标签几何已通过：投影框/bbox 比 1.04–1.18、角点与 `corner-projections.json` 误差 0.0 px、depth 有效、`mask_visib` 是 `mask` 子集、BOP 结构完整。但还有 3 个问题需要修：

1. 实际像素遮挡偏轻：目标 55/30/15（≤5% / 5–20% / 20–35%），实际 none 59%、light 29.5%、moderate 11.5%，mean 2.65%、max 34.7%。原因是 occluder 是沿相机射线放置的小椭球，经常只擦到物体边缘。
2. 尺寸过滤用的是投影 3D AABB（4–25%、128–300px），导致 79/200 帧的真实可见 mask 面积低于 4%。最终过滤应改为基于实际渲染出的 mask。
3. `corner-projections.json` 的 `rgb` 字段和 `validation-summary.json` 的 `dataset_root`/`contact_sheet` 仍是 `C:\` 绝对路径，应改为相对路径。

先阅读代码再修改，不要猜测结构：
`render_action4.py`、`run_render.py`、`action4_domain_v6.py`、`validate_action4_random.py`、`package_action4_regression.py`、`action4_geometry.py`。

## 修改一：遮挡按实际像素命中 55/30/15

1. 每帧先按确定性调度分配目标遮挡等级（避免随机漂移），例如每 20 帧中 11 帧 none（≤5%）、6 帧 light（5–20%）、3 帧 moderate（20–35%）。
2. 使用渲染后的实际像素计算：`occlusion = 1 - visible_mask_pixels / full_mask_pixels`；截断帧使用图像内 mask（沿用现有 in-image 语义）。
3. 只有落入该等级区间（边界允许 ±3 个百分点容差）的帧才接受；否则调整 occluder 的尺寸/位置（按投影物体大小缩放椭球、加大覆盖使遮挡落在物体内部而不是只擦边）并重试；设最大尝试次数，超限则报告失败而不是静默放宽门槛。
4. 硬性底线：任何接受帧 `occlusion <= 0.35 + 容差`，validator 的 `--min-bop-visibility 0.60` 必须保留。
5. 保持现有遮挡物类型（线缆/腕带/支架/前臂代理）语义不变；任何遮挡物都不能写入 Action 4 的 mask。

## 修改二：尺寸过滤改为实际 mask

1. 最终接受条件改为基于渲染后的图像内 mask：
   - full mask bbox 宽度 ≥ 100 px（主要分布在 100–260 px）；
   - visible mask 面积比例 ≥ 2%；
   - 可见比例 ≥ 0.60（与修改一一致）；
   - 可选上限：full mask 宽度 ≤ 350 px。
2. 原有投影 AABB 约束可保留为采样先验，但不能作为唯一接受条件。
3. 分辨率保持 640×480；禁止渲染后裁剪/缩放（K 必须与图像一致）。

## 修改三：相对路径

1. `corner-projections.json` 中所有路径字段改为相对数据集根目录，例如 `train_pbr/000000/rgb/000000.png`。
2. `validation-summary.json`、`render_metadata`、package/manifest 中的 `dataset_root`/`contact_sheet` 等也改为相对或可移植路径。
3. 生成后用简单扫描确认所有 JSON 中不再出现 `C:\` 或反斜杠绝对路径。

## 校准（必须先做）

用修改后的代码生成 `dataset\action4-v6-calibration-40`（40 帧，至少 2 个 scene，固定新种子），运行 `validate_action4_random.py` 并检查：

- 实际遮挡等级分布是否落在 55/30/15 ±3% 内；
- 接受帧是否全部满足 100 px / 2% / 0.60 条件；
- BOP 文件结构、JSON schema、K/R/t、角点定义与 v6 pilot 完全一致；
- 无绝对路径。

不满足就继续调 occluder，不要直接开始 5000 张。校准结果与目标/实际对照统计先报告。

## 生成正式集

校准通过后生成 `dataset\action4-v6-train-5000`：

- 10 个 scene（`000000`–`000009`），每个 500 帧；
- 每个 scene 独立固定种子并记录（建议 `665600 + scene 序号`）；
- 渲染参数与 v6 pilot 保持一致（glb、soft-area、domain-v6、random-intrinsics、fill-light、PNG、cycles 50、Linear、`--mask-workers 0`），参考命令见 pilot 报告 Run command，以修改后的参数为准；
- 逐 scene 渲染并校验；任一 scene 的硬性检查失败立即停止并报告，不要静默继续；
- 参考速度约 3.9 秒/帧，5000 张预计 5–6 小时，保持机器不休眠。

## 全量验证与打包

1. 对 5000 张运行 `validate_action4_random.py --allow-truncation --min-bop-visibility 0.60`，硬性检查必须全部通过。
2. 生成 `validation-summary.json`、`validation-report.md`、按遮挡等级分类的 contact sheet、按尺寸分层的 contact sheet、每 scene 统计。
3. 打包（训练不需要 92MB 的 GLB，不要打进包）：
   - 每个 scene 一个 zip：`action4-v6-train-5000-scene-000000.zip` … `000009.zip`（方便 ToDesk 分批传输）；
   - 一个小的 `action4-v6-train-5000-regression-32.zip`（从正式集抽 32 张代表帧，含 BOP JSON 以及 `models_info.json`/`corners_3d.json`），供服务器先做快速回归；
   - 一个 `action4-v6-train-5000-manifest.json`，包含每个 zip 及关键文件的 SHA-256、帧数、随机种子、校验结果。
4. 不要 commit、不要 push；数据集不进 Git。

## 完成后报告

- 修改的文件和原因；
- 校准结果：目标 vs 实际遮挡分布、尺寸分布；
- 5000 张每 scene 数量、失败重采样次数、总耗时；
- 打包路径和 SHA-256；
- 仍然存在的风险（前臂仍是几何代理、屏幕内容为程序生成、相机内参为占位等）。
