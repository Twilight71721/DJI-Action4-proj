# 动作、朝向、光照和 AprilTag 修正

## 旧 200 张为何漏掉折衣动作

旧版 `folded_denim()` 每批只生成静态衣物，帧间仅改变手腕和观察相机，没有抓袖、折袖、衣摆翻折对应的衣物状态。`orient_action4_screen_to_viewer()` 还主动让后屏朝向观察镜头，随后只叠加有限扰动。灯位、面积和颜色固定，仅强度变化约 ±15%。因此 200 个不同文件并不代表 200 种有意义的动作或观察条件。

用户截图给出了抓取门襟/袖口、衣物铺展、抓袖、折袖并压平的证据。本次增加的衣摆翻折和翻腕极端情况属于诊断扩展，不能冒称已经从这四张截图统计出了真实动作频率。

## 已实现

- 衣物动作：`button`、`sleeve_grasp`、`right_sleeve_fold`、`left_sleeve_fold`、`hem_fold`、`press`、`turnover`、`spread`。
- 每个动作四个阶段；袖子与衣摆做连续几何变形，按钮、口袋和缝线随同变化。手腕由衣物接触锚点驱动，左右手可使用不同手势。
- 相机固定在腕部装配坐标系，取消动态模式的朝观察镜头约束；通过腕部旋转产生侧面、前面、后屏、顶部和底部的不同可见程度。实际覆盖以 `review/dynamic_coverage.json` 为准，不以预设名称代替测量。
- 四组光照：中性柔光、暖色侧光、暗冷光、逆光，改变灯位、面积、强度、颜色、主辅光比例和环境光。动作之间循环错开光照顺序，避免所有动作都把同一阶段固定对应同一灯光。
- 延长前臂近端，减少上一批前臂裁切边缘入镜问题。
- 旧 5×5 手工黑白图案替换为标准 AprilTag `36h11`，左手 ID 18，右手 ID 19。纸张为独立表面，保留白边，使用漫反射和轻微高光。

## 标记证据

原始 `frame_000000.jpg` 同时检测到 ID 18、19；用户截图 3 检测到 19、截图 4 检测到 18。对检测四角做透视展开后，将 8×8 单元与 OpenCV 标准输出比较，上述四次检测均为 0 位差异。

因此应称作 AprilTag 定位标记；没有生成普通网页 QR 码，也没有把参考帧裁剪后贴回合成画面。新纹理由 `cv2.aruco.generateImageMarker` 生成，并使用检测器回读检查 ID。生成纹理可解码不意味着每张远距离、倾斜或受遮挡的渲染图都可解码。

## 输出与复现

输出目录：`dataset/action4-hccepose-actions-pilot32-v2-20260921`。

32 张、1218×924、每张两个目标实例，属于动作覆盖诊断样本。原始 200 张保留不变；没有把这批直接作为正式训练集。

```bash
.venv-hccepose/bin/python scripts/run_action4_dynamic_pilot.py \
  --output dataset/action4-dynamic-new
.venv-hccepose/bin/python scripts/review_action4_wrist_dataset.py \
  dataset/action4-dynamic-new --expected 32 --allow-invisible
.venv-hccepose/bin/python -m unittest discover -s tests -p test_bop_bb8.py -v
```

首次试渲染暴露了贝塞尔缝线控制柄拉伸问题；修正方法是先将缝线曲面转换成网格，再使用与衣物相同的变形。失败试样保留在不含 `v2` 的目录中。

实测主导可见面：底部 24、左右侧面共 36、镜头面 3、后屏 1；没有顶部占主导的实例。本批是困难视角补充，朝向仍不均衡。四组光照各 8 张，16 张码图纹理全部通过回读；姿态独立投影最大差异约 `0.000192 px`，20 项现有回归测试通过。

其中 6/64 个实例完全不可见：3 个完全出框，3 个在画面内被遮挡（包含部分出框后剩余区域被遮挡的情况）。它们不能作正常识别正样本；`--allow-invisible` 会检查原生 BOP 的空可见 mask、零可见率、`[-1,-1,-1,-1]` bbox 约定，逐项报告，并以 `min_visib_fract=1e-6` 从正样本加载检查中排除。正样本清单有 58 个实例。严格检查的原始失败报告保留为 `review/validation_strict.json`。允许不可见样本通过结构审计，不代表通过视觉或训练集验收。

修改涉及 `scripts/render_action4_wrist_hccepose.py`、`scripts/review_action4_wrist_dataset.py`，新增 `scripts/action4_dynamic_scene.py` 和 `scripts/run_action4_dynamic_pilot.py`。动态行为仅通过 `--action` 启用，旧入口默认行为保留。

## 泛化的验证方法和限制

覆盖不足意味着遇到未覆盖朝向、遮挡和交互时存在泛化风险，但不能据此量化真实识别性能，更不能把这 32 张宣称为泛化提升的证明。应在训练预算相同的条件下，对比旧数据和改进数据，分别统计不同朝向、遮挡、尺寸、动作、光照下的角点/姿态误差及失败率。测试视频不能参与本次场景调参；同一视频必须按连续动作片段隔离，不能随机拆相邻帧。

衣物当前仍为运动学变形，没有布料自碰撞、真实褶皱求解或手指接触约束。手指姿态、支架尺寸、屏幕实时取景和皮肤/牛仔纹理仍不够真实；需要逐类视觉检查，不能只靠 BOP 数值检查验收。当前仅依据少量参考做均衡诊断采样，并非全视频事件分布。

屏幕与贴纸通常可提供很强线索，但测试必须覆盖它们侧向、反光、被挡或不可读的情况，避免只验证标记清晰可见的样本。

参考：

- [Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World](https://arxiv.org/abs/1703.06907)：随机化是减少仿真到真实域差异的一种方法，不保证本项目效果。
- [OpenCV ArUco/AprilTag 字典和生成接口](https://docs.opencv.org/4.8.0/de/d67/group__objdetect__aruco.html)。
