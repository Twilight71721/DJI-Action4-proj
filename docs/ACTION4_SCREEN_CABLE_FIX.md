# 双屏与线缆穿臂修正

## 修正的实际问题

原始 GLB 已有前屏、镜头和后屏。此前只增加后屏显示内容，且将画面连到材质的漫反射颜色，强光下容易呈现白纸般的效果；前屏没有对应的显示内容。线缆使用固定贝塞尔控制点，没有检查控制点之间或对侧前臂的碰撞。

本次保留原 GLB 的几何、UV 和目标坐标约定，改动如下：

- 前置 LCD 和后置 LCD 各有独立显示表面。前屏边界经过原模型近景和表面射线检查，只覆盖前置小屏，镜头保留。
- 画面接发光通道，玻璃使用近黑色漫反射底色和适度反射，避免把 LCD 当成印有图片的白纸。
- AprilTag 只出现在后屏。纸贴由 22 mm 缩小到 16 mm，保留白边，不替代整个后屏。
- 每台虚拟相机分别从其镜头位置渲染一张合成取景图；前后屏使用对应设备的取景内容，前屏取中心裁剪。**目前使用每批第 0 帧，批内没有逐帧更新，不应称为完整实时屏幕仿真。**
- 动态场景的右侧装配调整滚转方向，使插头朝外；线缆从插头实际出口沿外侧绕过前臂，并延伸出画面，避免之前在皮肤中间结束的效果。
- 新线缆为半径 1.8 mm 的实际管状几何，沿密集折线生成，没有贝塞尔控制点间的隐式过冲。
- 使用细分后的双手/前臂网格建立碰撞代理。仅在代理中封闭裁剪边界，以便区分内外；渲染网格不因此被改动。每帧同时检查两根线对两只手臂的距离，扣除线径和采样间距余量，要求至少 1 mm 安全距离。

## 输出

修正版目录：`dataset/action4-screen-cable-corrected32-20260921`。

- `review/index.html`：32 张动作图片。
- `review/device_displays.jpg`：原始 GLB 的前后屏近景检查。这是独立资产视图，不冒称真实视频帧。
- `review/cable_before_after.jpg`：同一折袖场景的旧版与修正版对照。
- `review/validation.json`：BOP、深度、遮罩、角点和数据加载检查。
- `review/screen_cable_audit.json`：逐帧线缆间距的汇总。
- `render_metadata/batch_XXX/metadata.json` 中的 `cable_clearance`：线径、连接点误差、采样间距、双臂间距和保守余量。

旧数据和失败试样保留，没有覆盖原先 200 张或上一批 32 张。名称包含 `fixed32` 的目录是修正过程中的试样，应使用上面明确指定的 `corrected32` 目录。

## 代码与验证

修改：`scripts/action4_wrist_assets.py`、`scripts/action4_dynamic_scene.py`、`scripts/render_action4_wrist_hccepose.py`。

新增：`scripts/action4_cable_geometry.py`、`tests/test_action4_cable_geometry.py`。

```bash
.venv-hccepose/bin/python scripts/run_action4_dynamic_pilot.py \
  --output dataset/action4-screen-cable-new
.venv-hccepose/bin/python scripts/review_action4_wrist_dataset.py \
  dataset/action4-screen-cable-new --expected 32 --allow-invisible
.venv-hccepose/bin/python -m unittest discover -s tests -p test_action4_cable_geometry.py -v
.venv-hccepose/bin/python -m unittest discover -s tests -p test_bop_bb8.py -v
```

新增的 3 项解析几何检查和原有 20 项 BOP 回归测试通过。完全不可见的目标仍须单列，并从正常正样本加载中排除。

本次 32 张全部生成，64 根线缆状态、128 项线缆与手臂组合检查全部通过。扣除线径与最大采样步长的一半后，最小保守表面间距为 `15.912 mm`，插头连接点误差为 0。

**BOP 全量验收仍未通过**：`000000/000001` 的右侧实例，原生 GT info 统计可见像素 1602，而保存的可见 mask 有 1595 个像素，相差 7，超过检查器原定的 5 像素容差。本次没有修改原生标注或放宽容差来消除这条错误。59 个实例具有可见像素，另 5 个完全不可见。角点、深度等检查结果见原始 `validation.json`。因此这里交付的是显示与线缆修正的验证样张，不宣称正式训练数据验收通过。

这些检查针对线缆与双臂的几何关系，不代表完成了线缆/衣物动力学、自碰撞或全部附件接触仿真；衣物和手指的真实感仍有上一批已说明的限制。
