# 给服务器 OpenCode 的完整 Prompt

将下面代码块完整复制给服务器上的 OpenCode。先把本交付 ZIP 解压到仓库外或一个明确的新目录，再把实际路径替换进 Prompt；不要假设服务器与本机绝对路径相同。

```text
你现在负责把“真实视频域引导的 DJI Action 4 合成场景”集成进现有项目。你没有此前对话记忆。真实性、几何真值和可复现性优先于生成数量；不要直接批量生成正式训练集。

一、必须先阅读和核实

1. 完整阅读仓库 AGENTS.md（如有）、README、docs/ACTION4_REALISTIC_DATASET_GUIDE.md、docs/BPY_ACTION4_USAGE.md，以及 skills/action4-realistic-synthesis/SKILL.md 和它引用的两个 references。
2. 运行 git status --short，记录并保护现有未提交改动。不要自动 git commit/push，不覆盖任何已有数据集。
3. 定位而不是猜测以下内容：
   - references/real_frames/frames_full 下 2850 张真实帧；
   - frames.csv、contact_sheets_all_frames、detail_crops、scene_analysis.md；
   - 用户精确 Action 4 GLB；
   - scripts/blender_real_scene_preview.py；
   - 现有 HCCEPose 派生 BlenderProc 的入口、BOP writer、mask 和 BB8 验证器。
4. 报告真实帧数量/尺寸/时间范围、GLB 顶点/面数/材质/包围盒、bpy 和 BlenderProc 环境。路径或依赖缺失时明确报告，不要伪造接口。

二、任务目标

从真实帧提取多标签场景分布，并以这些分布驱动 Blender 合成。不是 2850 帧对应 2850 类，也不是逐帧像素临摹。正式结果称为 video-domain-adapted；参考视频默认不直接粘贴为合成背景或屏幕纹理。

需要复现的主要因素：
- 抬腕浅墙、下降过渡、桌面/网格垫/蓝色衣物操作、结尾抬腕手势；
- 张手、抓握、捏取、握拳、竖拇指；
- 后屏、顶部斜视、侧面、前镜头和底座视角；
- 双实例的不同尺度/方向/遮挡，轻度到中度出框；
- 亮屏实时画面感、UI、反光、左侧编码贴、顶部编号贴和红圈按键；
- 灰色织物腕带、荧光绿包边、黑色底座/枢轴/铰链、插头和有重力弧度的线缆；
- 运动模糊、曝光和白平衡只能做轻量且有记录的变化。

三、逐帧分类

1. 按 skill/reference 中的 schema 为全部帧生成 frame_taxonomy.csv，并生成 segment_summary.json。
2. 先按连续时间段/动作事件分组；自动分类只能给候选。人工复核阶段边界、少数类、强遮挡和低置信度帧。
3. 输出原始时长分布与去相邻冗余后的事件分布。训练/验证若使用真实数据，必须按连续 segment 划分，不能随机逐帧划分。
4. 初始观察边界可用 0–100、101–2779、2780–2849 作为检查起点，但必须根据帧证据修正，不能把它当硬编码真值。

四、Blender 资产和朝向硬约束

1. 必须使用用户精确 GLB，不得以代理盒、PLY 或自制相机替代，不得改变 mesh/material/UV，不得非均匀缩放。
2. 当前模型已核实：后屏外法向为局部 -Z，屏幕上方为局部 +Y。仍要用可视化复核。
3. “上下翻正”和“屏幕正对观察相机”是两个约束：用观察相机方向和手掌方向构造正交旋转基，使局部 -Z 朝观察相机、局部 +Y 保持顶部朝上。桌面基线可从 12 度顶部可见倾角开始，但正式分布要合理随机化。
4. 手/腕带/支架/线缆/衣物是上下文和遮挡物，不属于 Action 4 的完整 mask 或 3D bbox。贴纸和屏幕内容不改变 BB8 几何。
5. 固定层级：wrist→cuff/mount→Action4；connector 跟随 Action4；cable 起点重新连接到 connector，避免旋转后断开。拒绝悬浮、穿模、漂移和无接触阴影。
6. 保留历次失败经验：复制 GLB 后必须显式取消实例 hide_render/hide_viewport，并断言实例共享正确 mesh、无 modifier；不要再次让隐藏的 GLB 被代理模型取代。

五、实施方法

1. 先用 scripts/blender_real_scene_preview.py 和 standalone bpy 做固定 seed A/B：一次只改一个变量，保存新的 version-tag，不覆盖旧图。
2. 对 raised/tabletop、open/thumbs_up、关键视角分别保存全图和 crop。和 package 中 assets/action4_realism/generated、user_feedback 及真实 detail_crops 对照。
3. 视觉基线通过后，只迁移已验证的资产创建、父子关系、坐标和采样逻辑到现有 HCCEPose/BlenderProc，不要用预览脚本取代正式 writer。
4. 在现有代码风格下做最小改动。优先预计修改 action4_domain_v6.py、render_action4.py、run_render.py 和相关验证器；实际文件以仓库检查为准。
5. 每张正式图记录 scene_stage、左右手势、左右视角、屏幕状态、背景、遮挡、截断、seed、连续相机/光照参数和资产版本。

六、pilot 与验证

1. 先生成 100–200 张新的分层 pilot，禁止直接生成 5000 张。目录已存在就使用新的、语义明确的目录名。
2. pilot 覆盖主要场景、至少五种视角、五种手势、单双实例、三档 0–35% 遮挡、三档截断、亮屏可读/反光/遮挡。
3. 输出每类 contact sheet 和代表 crop，与真实帧对照。
4. 必须验证：
   - GLB identity、顶点/面数、材质槽、统一 scale；
   - 屏幕法向、顶部方向、支架接触、插头和线缆连接、穿模；
   - K/R/t 与 BB8 投影一致，角点顺序固定，越界不 clamp；
   - RGB/depth/mask/mask_visib 尺寸与实例索引一致；
   - mask_visib 是 mask 子集，可见比例来自像素；
   - 上下文附件不进入目标完整 mask；
   - 无空 mask、全零 depth、NaN/Inf、全白屏幕或严重过曝；
   - 类别配额、对象尺寸、遮挡、截断、亮屏、背景和手势都有统计。
5. 任一验证失败就保留失败图、seed 和日志，修复后只重跑最小相关集合。pilot 未通过前不要扩大生成规模。

七、最终交付

- frame_taxonomy.csv、segment_summary.json 和分类审计报告；
- 修改前/后固定 seed A/B、每类 contact sheet、失败样本；
- 100–200 张完整 BOP pilot；
- 几何/BOP/BB8/mask/分布验证报告；
- 所有改动文件、运行命令、测试结果、耗时/显存和已知限制；
- 明确说明哪些部分是程序化近似（例如当前简化手模型），不能把它们宣称为照片级真实资产。

先完成只读审计和实施计划，再开始修改。修改后运行已有相关测试；若测试成本高，先跑最小固定 seed 回归和 pilot 验证。不要做无关重构。
```
