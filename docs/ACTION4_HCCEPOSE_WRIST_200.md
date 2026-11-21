# HCCEPose 双腕场景 200 张查看样本

## 目的与当前质量

使用用户已有 Action 4 GLB，在本地 HCCEPose 派生 BlenderProc 中重建双腕场景，输出 200 张有完整 BOP 标注的 RGB 图片，供查看与比较。它们是参考视频引导的合成重建，**目前尚未达到与真实帧几乎相同的照片级效果**。

当前重点复现了双实例、灰色腕带与绿色包边、黑色支架与枢轴、线缆、屏幕与 UI、黑白标记、顶部编号、网格桌面和牛仔衬衫。没有把真实视频帧作为背景或屏幕纹理，没有使用 ImageGen 替换 HCCEPose 的输出。

## 输出

本次目录：`dataset/action4-hccepose-realistic-200-20260921`。

- `train_pbr/000000` 至 `000019`：每个 scene 10 张，RGB 为 1624×1232 PNG，每张两台相机。
- `rgb`、`depth`、`mask`、`mask_visib` 和三个 BOP JSON 均来自正式 BlenderProc/Cycles/BOP 流程。
- `models`：原 GLB、原 `models_info.json` 和固定顺序的 `corners_3d.json`。
- `render_metadata/batch_XXX`：每批种子、物体身份检查、独立投影、相机距离和合成屏幕纹理。
- `review/index.html`：逐张图库。
- `review/overview.jpg`：每批一张代表图。
- `review/contact_01.jpg` 等：全部 200 张的联系表。
- `review/overlays`：每批第一帧的双实例 BB8 叠加图。
- `review/validation.json`：自动检查的实测结果；以这个文件为准，不以本说明替代验收。

## 实际使用的渲染器

入口：`scripts/render_action4_wrist_hccepose.py`。

脚本明确优先导入 `third_party/Blender/vendor/blenderproc`，使用 HCCEPose 修改版的 `bproc.init()`、`bproc.loader.load_obj()`、`bproc.renderer.render()` 和 `bproc.writer.write_bop()`。最终 metadata 记录了实际模块路径。

本地 `.venv-hccepose` 的 `bpy 3.6.0` **包含 Cycles**。此前“只能使用 EEVEE”的判断不正确：需要先调用 `bpy.ops.preferences.addon_enable(module="cycles")`。本次已实测 OptiX GPU 渲染成功，无需修改系统驱动或 CUDA。

## 模型与标签约定

- 目标网格仍为原 GLB 的 915032 顶点、1499206 面，左右实例共享原网格数据，不加几何修改器、不做非均匀缩放。
- GLB 本地顶点为毫米，场景对象使用统一 `0.001` 缩放。
- 原 UV、颜色、法线等纹理保留；为了接近参考中的深色机身，Base Color 增加 `0.20` 乘数，Specular 设置为 `0.20`。这属于材质调整，不应描述成材质完全未变。
- 屏幕、标签、红色按键圈为独立外观覆盖层；手、腕带、支架、线缆和衣物不属于 BOP 目标对象。
- 使用固定八角点顺序，越界投影不裁到图像边缘。
- 深度单位写出为毫米，`depth_scale=0.1`。
- 以 `delta=1 mm` 计算可见掩码；紧贴表面的显示层/纸贴与机身表面在该容差下视为同一可见表面。
- 当前 HCCEPose 内置 BOP 工具箱的 bbox 宽高定义为 `max-min`，检查器遵守该约定。
- `bbox_obj` 在更大的画布上计算，保留出框范围；`bbox_visib` 对应画面内可见像素。对于出框实例，检查器验证完整框包含画面内 mask，不能将完整框直接与被裁剪的 mask 框判等。可见比例使用原生 `px_count_visib / px_count_all`。
- 原生 mask 与 GT info 分别在原尺寸和三倍画布渲染，边缘光栅化存在细小数值差异。本批 124/400 个实例的可见像素计数相差 1–3 像素；检查器设 5 像素容差并报告实测差异，没有修改原生标注来掩盖差异。

## 新增资产

手部使用 MakeHuman CC0 基础网格的手掌与前臂区域，保留 UV，进行局部指节弯曲与镜像。皮肤采用 Mindfront 的 Aksel CC0 贴图并做颜色调整。来源和许可记录在 `assets/action4_realism/makehuman/SOURCES.md`，没有安装 MakeHuman 应用。

来源：

- [MakeHuman 基础网格](https://github.com/makehumancommunity/makehuman/blob/master/makehuman/data/3dobjs/base.obj)
- [MakeHuman 资产许可](https://github.com/makehumancommunity/makehuman/blob/master/LICENSE.md)
- [Skins 02 资产包](https://static.makehumancommunity.org/assets/assetpacks/skins02.html)

腕带、支架、贴纸、线缆、衣物为程序化几何。屏幕内容来自一个独立的合成场景视角，加程序化 UI；这是近似显示内容，尚未严格模拟两台 Action 4 各自真实的实时镜头画面。

## 复现

在项目根目录执行，指定不存在的新输出目录：

```bash
.venv-hccepose/bin/python scripts/run_action4_wrist_200.py \
  --output dataset/action4-hccepose-realistic-200-new

.venv-hccepose/bin/python scripts/review_action4_wrist_dataset.py \
  dataset/action4-hccepose-realistic-200-new --expected 200
```

只运行第一批 10 张：

```bash
.venv-hccepose/bin/python scripts/run_action4_wrist_200.py \
  --output dataset/action4-hccepose-first-batch-new --batch-limit 1
```

已有完整批次可用 `--resume` 跳过。若中断发生在批次写入途中，应先检查日志与该批目录，不要盲目重复运行，也不要删除旧结果；单批脚本遇到已存在的工作目录会报错。

数据加载回归测试沿用仓库实际命令：

```bash
.venv-hccepose/bin/python -m unittest discover -s tests -p test_bop_bb8.py -v
```

## 自动验证的含义

本次实测：200 张不同 RGB、400 个目标实例，60 张抬腕、140 张桌面；八角点独立投影最大差异 `0.000484 px`，32 个实例通过现有训练数据加载器检查。4 个实例部分出框，mask/info 最大像素计数差异为 3。按照上述原生约定与容差完成检查，`errors=[]`。仓库 `test_bop_bb8.py` 的 20 项回归测试通过。

`action4_hccepose_200_rgb.zip` 包含全部 200 张原始 PNG（约 361 MB），附原始路径清单；完整深度、遮罩、姿态与模型仍在数据目录中。

检查 JSON 索引、RGB/depth/mask 形状、旋转矩阵、正深度、可见掩码子集、mask bbox、图像重复、原始场景矩阵与 BOP 投影差异，并通过现有 `BOPInstanceDataset` 抽查 32 个实例的 336×336 ROI / 8×84×84 热图。

这些检查能证明数据链路一致，**不能证明照片级真实，也不能证明训练收益**。没有用该数据重新训练模型或对真实视频报告准确率。

## 尚未解决的真实感差距

- 手部不是视频中本人的扫描资产，肤色、关节、腕纹与动作只能近似。
- 手势采用局部几何弯曲，尚未构成完整骨骼、碰撞和接触求解；捏取衣物等交互不精确。
- 部分较远或偏转视角会露出裁剪前臂的截断边缘；部分弯曲手指形状不自然。这些是已观察到的缺陷，几何标签检查通过不代表画面没有这些问题。
- 支架和腕带按观察重建，没有实物 CAD 或实测尺寸。
- 衣物使用程序化褶皱与细节，尚未经过真实布料仿真，领口与袖口有明显建模感。
- 屏幕、贴纸、光照和曝光近似参考，未逐帧反求相机内外参与光场。
- 200 张覆盖不同种子、手势、姿态和镜头距离，但不是 200 个真实帧的精确一一重建，也没有覆盖视频中所有重遮挡和出框分布。

因此，本批适合用于检查当前重建能达到什么程度，不能直接宣称已满足“与真实帧几乎相同”的要求或作为最终正式训练集验收。
