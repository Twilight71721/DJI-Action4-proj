# Blender 预览与 HCCEPose/BlenderProc 集成

## 两条流水线

- standalone `bpy`: 快速检查外观、层级、坐标、构图，输出 PNG/BLEND；不产正式 BOP。
- HCCEPose 派生 BlenderProc: 正式产 RGB/depth/mask/mask_visib、K/R/t、BOP 与 BB8。

不得因为预览图看起来正确就跳过正式标注验证。

## GLB 与朝向

保持导入 GLB 的 mesh datablock、材质和统一 `0.001` 缩放。当前已核实局部 `-Z` 是后屏外法向、局部 `+Y` 是屏幕上方向。使用观察相机方向和手掌方向构造正交旋转基，不仅调整单个 Euler 角。

每次实例化验证：可见状态、mesh identity、modifier 为空、材质槽、scale、bbox 和顶点/面数。贴纸、标签和插头使用独立子对象；不要重建相机主体。

## 附件层级

腕部驱动腕带和底座，底座驱动 Action 4，插头固定在 Action 4，线缆第一控制点由插头位置更新。所有上下文资产都必须设置成非 BOP 目标，允许其遮挡目标并影响 depth/`mask_visib`。

## 场景采样

采样器接收逐帧分类得到的类别配额，先选 `scene_stage`，再条件采样背景、手势、相机相对腕部姿态、单双实例、遮挡/截断和成像参数。每张图把类别、连续参数、seed 和资产版本写入 metadata。

## 验证

固定 seed 做修改前/后 A/B。pilot 同时检查：

- GLB 不变量、屏幕法向、顶部方向、连接和穿模；
- K/R/t 投影、固定八角点顺序、正深度、越界不 clamp；
- RGB/depth/mask 尺寸和实例索引，`mask_visib ⊆ mask`；
- 遮挡像素比例、附件不污染目标 mask；
- 类别直方图、对象尺度、截断、亮屏、曝光和模糊；
- 每类 contact sheet 与真实代表帧的人工对照。

目录存在时不要覆盖；使用新 pilot 名称。未通过任一项时保留失败图和 seed，修复后只重跑相关最小集合。
