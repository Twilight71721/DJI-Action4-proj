# `bpy` 在 Action 4 真实场景预览中的使用方法

## 1. 原理

`bpy` 是 Blender 的 Python API。它可以创建对象、导入 GLB、设置父子层级、材质、相机和灯光，并保存 `.blend`、渲染 PNG。当前预览脚本使用项目虚拟环境内嵌的 `bpy 3.6.0`，只提供 `BLENDER_EEVEE`；它适合快速几何和构图迭代，不负责正式 BOP 数据写入。

本项目的正式数据生成走 HCCEPose 派生的 BlenderProc。两者关系是：

```text
真实帧观察 → standalone bpy 快速预览 → 人工 A/B 验收
                                      ↓
                      迁移到 HCCEPose/BlenderProc
                                      ↓
                   RGB/depth/mask/BOP/BB8 正式输出
```

## 2. 环境检查

在项目根目录运行：

```bash
.venv-hccepose/bin/python -c "import bpy; print(bpy.app.version_string)"
```

本机已核实为 `3.6.0`。不要用系统 `python`；本机甚至可能没有名为 `python` 的命令。服务器若路径不同，应先找到能成功 `import bpy` 的解释器。

可检查可用渲染引擎：

```bash
.venv-hccepose/bin/python - <<'PY'
import bpy
items = bpy.context.scene.bl_rna.properties['render'].fixed_type.properties['engine'].enum_items
print([item.identifier for item in items])
PY
```

当前环境只有 `BLENDER_EEVEE`。不要在此解释器里强行设置 `CYCLES`。

## 3. 运行预览

默认运行：

```bash
.venv-hccepose/bin/python scripts/blender_real_scene_preview.py
```

推荐桌面基线：

```bash
.venv-hccepose/bin/python scripts/blender_real_scene_preview.py \
  --model dataset/action4-v6-domain-pilot-200/models/obj_000001.glb \
  --output-dir outputs/blender_real_scene_preview \
  --scene tabletop \
  --pose open \
  --screen-tilt-deg 12 \
  --width 1280 \
  --height 960 \
  --samples 96 \
  --version-tag experiment_name
```

抬腕/墙面类别：

```bash
.venv-hccepose/bin/python scripts/blender_real_scene_preview.py \
  --scene raised \
  --pose open \
  --width 1280 \
  --height 960 \
  --samples 96 \
  --version-tag raised_pilot01
```

参数含义：

| 参数 | 作用 |
|---|---|
| `--model` | 精确 Action 4 GLB，不能换成代理模型 |
| `--output-dir` | PNG 和 `.blend` 的输出目录 |
| `--scene` | `raised` 使用浅墙背景；`tabletop` 使用网格桌垫和蓝色衣物 |
| `--pose` | 当前提供 `open`、`thumbs_up` 两种程序化预览手势 |
| `--camera-pitch-deg` | 调试相机相对腕部的 pitch；未设置时按场景取默认值 |
| `--screen-tilt-deg` | 桌面场景中屏幕朝向观察相机后，再露出顶部的倾角 |
| `--samples` | EEVEE TAA 渲染采样数 |
| `--version-tag` | 输出文件名后缀；每轮使用新名字，禁止覆盖比较基线 |

## 4. 坐标和朝向约定

导入模型后必须先以可视化或网格特征核实坐标。本项目已确认的当前 GLB 约定是：

- 后屏朝外法向：局部 `-Z`；
- 屏幕视觉上边/顶部方向：局部 `+Y`；
- 屏幕水平方向：局部 `+X`；
- GLB 以毫米建模，场景实例统一缩放为 `0.001`；
- 不允许通过非均匀 scale “修外形”。

让屏幕朝向观察相机时，不应凭经验堆叠 Euler 角，而应构造正交基：

```python
view_direction = (viewer_location - world_position).normalized()
screen_up = hand_direction - view_direction * hand_direction.dot(view_direction)
screen_up.normalize()

tilt = math.radians(top_tilt_degrees)
tilted_up = (screen_up * math.cos(tilt) + view_direction * math.sin(tilt)).normalized()
screen_normal = (view_direction * math.cos(tilt) - screen_up * math.sin(tilt)).normalized()

local_z_world = -screen_normal
local_x_world = tilted_up.cross(local_z_world).normalized()
local_y_world = local_z_world.cross(local_x_world).normalized()
world_rotation = Matrix((local_x_world, local_y_world, local_z_world)).transposed().to_quaternion()
camera_root.rotation_quaternion = wrist_root.matrix_world.to_quaternion().inverted() @ world_rotation
```

这种做法同时约束“屏幕朝谁”和“顶部朝哪”，比只调一个 Euler pitch 更稳定。

## 5. 安全地复制 GLB 实例

预览脚本把导入对象当成隐藏模板，再让左右相机共享同一 mesh datablock：

```python
base_mesh.hide_render = True
base_mesh.hide_viewport = True

body = base_mesh.copy()
body.data = base_mesh.data
bpy.context.collection.objects.link(body)
body.parent = camera_root
body.scale = (0.001, 0.001, 0.001)
body.hide_render = False
body.hide_viewport = False
```

需要保留断言：

```python
if body.data is not base_mesh.data or body.modifiers:
    raise RuntimeError("Action 4 instance must preserve the imported GLB mesh without modifiers")
```

早期失败版本正是因为复制对象仍继承隐藏状态，真正 GLB 没有进入渲染。每次实例化后都应检查 `hide_render`、`hide_viewport`、mesh identity、modifier、材质槽和 scale。

## 6. 父子层级和局部坐标

固定附件必须使用局部坐标挂在其刚体父节点上：

```python
child.parent = parent
child.location = local_location
child.rotation_euler = local_rotation
```

建议规则：

- 腕带、底座和相机根节点挂在 `wrist_root`；
- GLB、屏幕贴纸、顶部标签和插头挂在 `camera_root`；
- 线缆曲线整体挂在 `wrist_root`，第一个控制点由插头世界坐标反算到腕部局部坐标；
- 调整相机旋转后，必须更新线缆起点，不能保留旧的世界位置。

对于线缆：

```python
connector_world = connector.matrix_world @ Vector((0.007, 0.0, 0.0))
connector_local = wrist_root.matrix_world.inverted() @ connector_world
points[0].co = connector_local
```

## 7. 材质与贴纸

- 腕带使用高粗糙度材质并以 Noise→Bump 提供细密织纹；
- 绿色包边、黑色橡胶、底座塑料、白色标签分别使用独立材质；
- 标签/编码贴是极薄的平面几何，避免与机身表面共面导致 z-fighting；
- 不要在预览阶段重写 GLB 原材质来“看起来更像”；新增外观部件应放在独立对象；
- 正式流水线要明确贴纸是否属于目标完整 mask，并保持 writer 一致。

## 8. 保存、打开和比较

脚本会先保存 `.blend`，再渲染 PNG：

```python
bpy.ops.wm.save_as_mainfile(filepath=os.fspath(blend_path))
bpy.ops.render.render(write_still=True)
```

如果服务器安装了完整 Blender，可用：

```bash
blender outputs/blender_real_scene_preview/real_scene_tabletop_open_experiment_name.blend
```

若只装了 pip `bpy`，仍可无界面生成和修改 `.blend`，但不能依赖桌面 UI。每次改一个因素并换 `--version-tag`，把旧/新 PNG 并排比较。

## 9. 常见故障

| 现象 | 原因 | 修复 |
|---|---|---|
| 只看到代理盒，不见真实相机 | GLB 模板或实例仍被隐藏 | 检查实例 `hide_render=False`，确认共享正确 mesh |
| 顶部标签在屏幕下方 | 局部 `+Y` 方向被倒置 | 同时约束屏幕法向和 up，不只修 roll |
| 屏幕还是平躺 | 只绕桌面法线旋转 | 令局部 `-Z` 对准观察相机位置 |
| 线缆与插头断开 | 相机旋转后曲线控制点未更新 | 用矩阵把插头世界坐标换到腕部局部坐标 |
| 背景与参考阶段不符 | 场景类型未由分类驱动 | `raised`/`tabletop` 分开建模和采样 |
| 画面像塑料玩具 | 轮廓、接触和材质层级都过度简化 | 先修 P0 结构，再调光照/噪声 |
| 设置 Cycles 报错 | 当前 pip `bpy` 只编译了 EEVEE | 预览继续用 EEVEE，正式渲染走现有 BlenderProc 环境 |
| 图像看起来正确但标签错 | standalone 预览没有 BOP writer | 迁移到正式流水线后运行全部投影/mask验证 |

## 10. 迁移到正式渲染器的检查表

迁移不是复制整份脚本，而是复制已验证的资产逻辑和坐标约束：

- [ ] 保留正式渲染器现有的 GLB 加载和 BOP 属性；
- [ ] 将场景类别作为显式配置和 metadata；
- [ ] 把附件设置成非 BOP 目标对象；
- [ ] 保证遮挡物进入 depth 和 `mask_visib` 计算；
- [ ] 使用正式相机 K/R/t 计算 BB8；
- [ ] 记录所有随机种子和类别字段；
- [ ] 固定姿态 A/B 与 `bpy` 预览方向一致；
- [ ] 先 pilot，后全量。
