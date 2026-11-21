# Action 4 真实视频域引导的合成数据制作指南

## 1. 目标与边界

本任务不是把 2850 帧真实视频直接变成训练集，也不是逐帧临摹。目标是从真实视频中提取**可复用的场景类型、构图分布、零件关系、遮挡方式和成像特征**，再用 Blender 生成有完整几何真值的训练数据。

正式结果应称为“针对该视频域适配的合成数据”。因为资产设计和采样分布已经参考了这段视频，不能再把它表述为完全未见过的盲测集。如果后续需要无偏最终评估，应另留一个未参与设计的视频或真实图像集。

最高优先级依次为：

1. 保留真实 Action 4 GLB 的几何、材质、UV 和对象坐标系；
2. 让屏幕朝向、顶部方向、支架连接、腕带接触和线缆端点符合物理关系；
3. 让合成样本的类别比例接近真实视频，而不是只做“漂亮的正面图”；
4. 保持 BOP 位姿、深度、mask、`mask_visib`、bbox 和 BB8 投影严格一致；
5. 最后才增加轻度噪声、模糊、曝光和颜色扰动。

## 2. 已核实的输入

| 内容 | 本地位置 | 说明 |
|---|---|---|
| 源视频 | `/home/twilight/Downloads/head_left_rgb_raw (2).mp4(1)/head_left_rgb_raw.mp4` | 3248×2464，约 29.97 FPS，2850 帧，95.10 s |
| 全部真实帧 | `outputs/video_reference_audit/head_left_rgb_raw/frames_full/` | `frame_000000.jpg` 至 `frame_002849.jpg` |
| 帧时间表 | `outputs/video_reference_audit/head_left_rgb_raw/frames.csv` | 零基帧号、时间戳、文件名 |
| 全帧联系表 | `outputs/video_reference_audit/head_left_rgb_raw/contact_sheets_all_frames/` | 29 张，每张最多 100 帧 |
| 代表性裁剪 | `outputs/video_reference_audit/head_left_rgb_raw/detail_crops/` | 屏幕、支架、线缆与双腕局部 |
| 真实场景报告 | `outputs/video_reference_audit/head_left_rgb_raw/scene_analysis.md` | 已对全部联系表和代表帧复核 |
| 目标模型 | `dataset/action4-v6-domain-pilot-200/models/obj_000001.glb` | 915032 顶点、1499206 面、单材质；不得用代理盒替换 |
| 视觉预览脚本 | `scripts/blender_real_scene_preview.py` | standalone `bpy 3.6.0` + EEVEE，只做快速外观迭代 |
| 正式渲染入口 | `dataset/code/run_render.py`、`render_action4.py` | HCCEPose 派生的 BlenderProc/BOP 流水线 |

压缩包内采用相同的逻辑目录，但以包根目录为基准。服务器上的绝对路径可能不同，OpenCode 必须先定位文件，不能照抄本机绝对路径。

## 3. 真实视频中应复现的内容

视频大致包含以下阶段。边界是用于初始化分类器的观察值，不是不可修改的真值：

| 初始帧段 | 场景阶段 | 关键特征 |
|---|---|---|
| `0–100` | `raised_display` | 双腕抬起，后屏、贴纸、顶部标签、支架最清楚 |
| `101–2779` | `tabletop_manipulation` | 在网格桌面和蓝色衣物附近抓握、捏取、翻折，遮挡和出框频繁 |
| `2780–2849` | `raised_gesture_end` | 再次抬腕，张手、握拳、竖拇指，线缆和底座清楚 |

不要把每一帧定义成一个类别。相邻视频帧高度相关，正确做法是建立多标签分类：

- `scene_stage`：`raised_display`、`lowering_transition`、`tabletop_manipulation`、`raised_gesture_end`；
- `hand_pose_left/right`：`open`、`grasp`、`pinch`、`fist`、`thumbs_up`、`partial_or_unknown`；
- `camera_view_left/right`：`rear_screen`、`top_rear_oblique`、`side`、`front_lens`、`bottom_mount`、`unknown`；
- `screen_state_left/right`：`on_readable`、`on_glare`、`off_or_dark`、`occluded`；
- `background`：`pale_wall`、`grid_mat`、`denim`、`mixed`；
- `occlusion_bin`：`0_05`、`05_20`、`20_35`、`gt_35`；
- `truncation`：`none`、`mild`、`strong`；
- `motion_blur`、`exposure`、`instance_count`、`notes`。

详尽字段和判定规则见 Skill 的 `references/frame-taxonomy.md`。

## 4. 从纠错过程得到的硬约束

这些迭代不是审美偏好，而是服务器实现必须遵守的工程约束。

### 4.1 初版：轮廓和连接关系过于“道具化”

本地配图（早期结果：手、支架、相机关系不可信） `assets/action4_realism/user_feedback/01_initial_result_tabletop.png`

问题包括：简化手臂、刚性腕带、相机像悬浮盒子、支架层级和插头不完整、贴纸和屏幕内容缺少真实位置约束。处理方式是把手/腕带/底座/转轴/相机/插头/线缆拆成明确父子层级，并优先修轮廓和接触。

### 4.2 不能用代理相机代替用户 GLB

本地配图（代理外观版本，仅可作为失败样例） `assets/action4_realism/generated/real_scene_open_v2.png`

早期预览复制导入 GLB 时继承了 `hide_render=True`，导致真正网格不可见，画面主要由代理结构承担。这会让外观、角点、边框和材质全部失真。修复后：

- 原始导入对象仅作为不可见模板；
- 每个实例复制同一个 mesh datablock；
- 实例显式设置 `hide_render=False`、`hide_viewport=False`；
- `body.data is base_mesh.data`，不允许额外 modifier 改几何；
- 缩放统一为 `(0.001, 0.001, 0.001)`，不得三个轴分别缩放。

### 4.3 不能为了匹配一个画面而换错背景

本地配图（用户要求保留的桌面构图） `assets/action4_realism/user_feedback/04_keep_tabletop_but_fix_camera_direction.png`

视频里既有浅色墙面的抬腕阶段，也有网格垫和蓝色衣物的桌面阶段。用户指出：当时其余部分已经接近要求，应保留桌面背景，只修 Action 4 的方向。由此得到两条规则：

1. 背景必须由 `scene_stage` 驱动，不能把所有帧统一换成墙面；
2. 一次迭代只改被指出的变量，使用固定 seed/相机/手势做 A/B，避免多个变化互相掩盖。

### 4.4 “上下翻正”不等于“屏幕正对观察相机”

本地配图（顶部方向已修正，但后屏仍没有正确立起） `assets/action4_realism/generated/real_scene_tabletop_open_camera_flipped_v1.png`

真实参考中，屏幕平面近似朝向观察者，顶部白色编号标签和红圈按键位于屏幕上方：

本地配图（真实参考：后屏立起并朝向观察相机） `assets/action4_realism/user_feedback/05_real_screen_should_face_observer.png`

仅绕桌面法线旋转只能改变左右或上下，不能改变屏幕法向。当前正确做法以观察相机位置直接构造旋转基：

- GLB 后屏朝外法向为局部 `-Z`；
- 屏幕视觉上边为局部 `+Y`；
- 局部 `-Z` 指向渲染相机；
- 局部 `+Y` 沿手掌方向投影到屏幕平面；
- 额外加入约 `12°` 顶部可见倾角；
- 旋转后重新把线缆第一个控制点接回插头。

本地配图（目前正确的桌面朝向基线） `assets/action4_realism/generated/real_scene_tabletop_open_screen_upright_v1.png`

该版本左右屏幕法向与观察方向夹角约 `12°`。这只是当前桌面类别的基线，正式生成时应在真实分布范围内小幅随机，而不是每张固定为 12°。

## 5. 推荐实施流程

### 阶段 A：只读审计与可复现基线

1. 阅读 `AGENTS.md`、README、现有渲染器、BOP 写入器和验证器。
2. 运行 `git status --short`，不得覆盖已有数据集或用户未提交改动。
3. 核对 GLB 的顶点数、面数、材质、包围盒、缩放和模型坐标系。
4. 用固定 seed 保存当前正式渲染基线、参数 JSON 和联系表。
5. 不先生成数千张；先建立可重复的 8 视角或少量场景 A/B。

### 阶段 B：逐帧多标签分类

1. 以 `frames.csv` 为主键读取全部帧。
2. 先按时间连续性自动分段，再每 5–10 帧取一张候选关键帧。
3. 使用图像相似度、光流或简单特征提出候选标签，但必须人工复核阶段切换、少数类和低置信度样本。
4. 输出 `frame_taxonomy.csv` 和 `segment_summary.json`，每一帧保留 `segment_id`。
5. 统计的是时长分布和事件分布，避免相邻静止帧把某类权重无限放大。

推荐训练/验证划分按连续 `segment_id` 进行；禁止随机打散逐帧后再切分，否则相邻帧几乎相同，会造成数据泄漏和虚高指标。

### 阶段 C：资产与姿态重建

建议对象层级：

```text
WristRig_L / WristRig_R
├── ArmHandMesh
├── CuffRoot
│   ├── CuffFabric
│   ├── CuffGreenTrim
│   ├── CuffBlackStrap
│   └── CuffOuterPad
├── MountRoot
│   ├── WristBasePlate
│   ├── PivotJoint
│   ├── HingeBlocks
│   └── ShortArm
├── Action4Root
│   ├── ExactAction4GLB       # 唯一训练目标
│   ├── ScreenMarkerDecal
│   └── TopIdDecal
└── CableRoot
    ├── Connector
    ├── StrainRelief
    └── CableCurve
```

其中手、腕带、支架、线缆和衣物都是上下文/遮挡物，不能混入 Action 4 的完整对象 mask。屏幕贴纸和顶部编号贴不改变 3D bbox；如果它们作为 Action 4 外观的一部分参与对象 mask，必须明确且在所有标注路径保持一致。

### 阶段 D：用 `bpy` 快速视觉迭代

`scripts/blender_real_scene_preview.py` 的用途是快速验证：背景、两腕构图、GLB 朝向、贴纸位置、支架连接和线缆端点。它不输出 BOP 标注，不能直接替代正式生成器。

```bash
cd /path/to/exam-project

.venv-hccepose/bin/python scripts/blender_real_scene_preview.py \
  --scene tabletop \
  --pose open \
  --screen-tilt-deg 12 \
  --width 1280 \
  --height 960 \
  --samples 96 \
  --version-tag pilot01
```

输出到 `outputs/blender_real_scene_preview/real_scene_tabletop_open_pilot01.{png,blend}`。更完整的 API、坐标约定和故障排查见 [BPY_ACTION4_USAGE.md](BPY_ACTION4_USAGE.md)。

### 阶段 E：合并到 HCCEPose/BlenderProc 正式流水线

当前项目正式渲染器基于 HCCEPose 派生的 BlenderProc；需把 `bpy` 原型中已经确认的结构迁移到：

- `dataset/code/action4_domain_v6.py`：资产创建、材质、附件父子关系、类别采样；
- `dataset/code/render_action4.py`：相机、场景、对象姿态、渲染与 metadata；
- `dataset/code/run_render.py`：批次参数与安全的输出目录；
- 相应验证器：BOP、mask、投影、类别配额和视觉审计。

迁移时禁止直接复制预览脚本的假设：standalone 脚本只有 EEVEE，正式流水线的渲染引擎、坐标换算、对象属性和 writer 必须以现有 HCCEPose/BlenderProc 实现为准。

### 阶段 F：分层 pilot，而不是直接全量生产

建议先生成 100–200 张 pilot，并覆盖：

- 两个主要背景阶段；
- 后屏、顶部斜视、侧面、正面镜头和底座视角；
- 张手、抓握、捏取、握拳、竖拇指；
- 单/双实例；
- 0–5%、5–20%、20–35% 三档遮挡；
- 完整、轻微截断、明显但仍可识别三档；
- 亮屏清晰、反光和部分遮挡。

每类至少输出一张 contact sheet、代表性全图和对象 crop。确认后再按类别配额批量生成，拒绝“先产 5000 张再看”的流程。

## 6. 真实性验收标准

### 6.1 几何和物理关系

- GLB mesh identity、顶点/面数、材质槽和统一缩放未改变；
- 后屏法向符合类别目标，顶部方向正确，不出现镜像或倒置；
- 支架与腕带有接触，Action 4 不悬浮；
- 插头对齐接口，线缆首端跟随相机，曲线不穿过机身/手臂；
- 腕带贴合手腕，绿色包边、织物主体和黑色结构可区分；
- 附件没有被误计入目标 3D bbox。

### 6.2 标签和 BOP 一致性

- `scene_camera.json`、`scene_gt.json`、`scene_gt_info.json` 索引一致；
- RGB/depth/mask/mask_visib 尺寸一致，`mask_visib ⊆ mask`；
- 可见比例由像素 mask 计算，不由场景参数估算；
- BB8 角点使用固定模型坐标和既有顺序，越界点不 clamp；
- 贴纸、屏幕内容和外观随机化不改变目标几何；
- 遮挡物真实影响 depth 和 `mask_visib`，但不污染完整对象 mask。

### 6.3 分布真实性

- 每个样本写入 `scene_stage`、手势、视角、背景、遮挡、截断、屏幕状态和随机 seed；
- 合成类别直方图与真实视频的“事件/时间段统计”对照，而不是只对原始帧数对照；
- 少数困难类可过采样，但报告真实比例和训练比例两套数字；
- 保留固定姿态 A/B 和每类 contact sheet，肉眼检查仍是必要门槛；
- 可选地比较真实 ROI 与合成 ROI 的亮度、色温、对象尺寸、边缘清晰度和视觉特征分布，但这些统计不能替代几何检查。

### 6.4 数据泄漏约束

- 真实帧作为参考证据和最终域内评估素材；默认不直接复制到合成 RGB，不直接作为屏幕纹理；
- 可以用真实帧估计类别比例、色温、曝光、对象占比和模糊范围；
- 如果未来决定把真实帧加入训练，必须单独标记为 real-data experiment，并按连续时间段重新划分，不能再用同一视频报告无偏测试结果。

## 7. OpenCode 的预期交付

服务器端完成后至少应交付：

1. `frame_taxonomy.csv`、`segment_summary.json` 和分类审计报告；
2. 资产层级/坐标系说明，GLB 不变量检查结果；
3. 固定 seed 的修改前/后 A/B 图和每类 contact sheet；
4. 100–200 张完整 BOP pilot；
5. 类别、姿态、遮挡、截断、亮屏状态分布统计；
6. BOP/BB8/mask/深度验证报告和失败样本列表；
7. 改动文件清单、执行命令、耗时/显存、已知限制；
8. 不自动 `git commit`，不自动 `git push`，不覆盖已有数据集。

只有 pilot 同时通过几何、标签、分布和视觉四类验收，才进入全量生成。
