---
name: action4-realistic-synthesis
description: 使用逐帧真实视频证据设计、预览、验证并集成 DJI Action 4 的视频域适配 Blender/BOP 合成数据；适用于帧分类、场景/手势/遮挡配额、精确 GLB 朝向、腕带支架线缆建模、bpy 预览和 HCCEPose/BlenderProc 正式生成。不要用于直接训练模型或把参考视频像素复制进合成 RGB。
---

# Action 4 真实感合成

先读仓库 `AGENTS.md`、主指导文档和现有渲染代码。不要假设服务器路径与本地相同。修改前运行 `git status --short`，保留未提交内容，不覆盖现有数据集，不自动 commit/push。

## 工作顺序

1. 核实真实帧、`frames.csv`、联系表、目标 GLB、现有 HCCEPose/BlenderProc 入口和验证器。
2. 分类视频时读 [references/frame-taxonomy.md](references/frame-taxonomy.md)。不要把每帧当独立类别；按连续 segment 分组，防止相邻帧泄漏。
3. 修改 Blender 或渲染器时读 [references/blender-hccepose-workflow.md](references/blender-hccepose-workflow.md)。
4. 用 standalone `bpy` 做固定 seed 视觉 A/B；它不是正式 BOP 生成器。
5. 把验收通过的资产/朝向/采样逻辑迁移到现有 HCCEPose 派生 BlenderProc 流水线。
6. 先生成 100–200 张分层 pilot，运行几何、BOP、mask、BB8、类别分布和视觉检查；通过后再扩大规模。

## 不变量

- 使用用户提供的精确 GLB；不以代理盒、PLY 或重建网格替代，不进行非均匀缩放。
- 当前 GLB 后屏法向为局部 `-Z`，屏幕上方向为局部 `+Y`；先用可视化复核，再依据观察相机和上方向构造旋转基。
- 手、腕带、底座、线缆和衣物属于上下文/遮挡物，不进入 Action 4 的完整目标 mask 或 3D bbox。
- 贴纸和屏幕内容不得改变三维角点；所有二维投影来自真实 K、R、t，不 clamp 越界角点。
- `mask_visib` 必须由真实遮挡后的像素计算，且是完整 mask 的子集。
- 背景、视角、手势、屏幕、遮挡、截断和成像变化必须由分类分布驱动并写入 metadata。
- 真实视频只用于证据、统计和评估；默认不把帧直接作为合成背景或屏幕纹理。
- 因为设计参考了该视频，结果应称为 video-domain-adapted；不要声称它仍是盲测。

## 停止条件

出现以下任一情况时，不得进入全量生成：GLB 几何/材质被替换、屏幕方向或顶部方向错误、附件漂浮/断开/穿模、目标 mask 被附件污染、BOP/BB8 投影不一致、相邻真实帧跨训练/验证泄漏、类别 pilot 缺失、contact sheet 暴露系统性失真。

完成时报告：改动文件、命令、pilot 路径、测试结果、类别统计、失败样本、渲染速度/显存和剩余限制。
