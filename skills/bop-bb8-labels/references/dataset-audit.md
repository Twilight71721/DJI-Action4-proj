# 已核实的数据事实与异常

此记录用于指导**再次核验**，不替代运行时检查。用户已确认后续使用 `/home/twilight/University/njukym/cs/cv/exam-project/dataset/train_pbr`。

## 已见到的格式

- `dataset/train_pbr` 有 `000000` 至 `000009` 十个场景；每个场景的 `scene_gt.json` 和 `scene_camera.json` 有 1000 个图像 ID，RGB 是 `640×480` JPEG。
- 各场景含 `rgb/`, `depth/`, `mask/`, `mask_visib/`, `scene_gt.json`, `scene_camera.json`, `scene_gt_info.json`, `scene_gt_coco.json`。`mask` 命名例如 `000000_000000.png`，后一个六位数是 `scene_gt` 数组的实例索引，不是 `obj_id`。
- 在已检查的 `dataset/train_pbr` 十个场景中，每帧当前只有一个 `obj_id=1` 实例；实现仍要按数组索引取实例，不要把“只有一个”写成不可变假设。
- `scene_gt[image_id][instance_idx]` 包含 `cam_R_m2c`、`cam_t_m2c`、`obj_id`。`scene_camera[image_id]` 包含 `cam_K`、`depth_scale`，以及可选的世界到相机位姿；本任务直接使用 **m2c**，不需要世界位姿。`scene_gt_info[image_id][instance_idx].bbox_obj` 是 `(x,y,width,height)`，并有 `bbox_visib`、`visib_fract`。
- 模型候选位于 `dataset/dji-action4/models/obj_000001.ply`，元数据位于 `dataset/dji-action4/models/models_info.json`。`obj_id=1` 的 AABB 尺寸约为 `(70.5,45.48,38.73)` **mm**。项目转换脚本 `scripts/glb_to_bop.py` 默认把 GLB 最大边缩放至 `70.5 mm`。BOP 文档在 `third_party/HCCEPose/bop_toolkit/docs/bop_datasets_format.md` 中声明模型点和 `cam_t_m2c` 都使用 mm。

## 必须先解决的模型—数据一致性问题

用上述 `70.5 mm` 候选模型元数据、`scene_gt` 与逐帧 `cam_K` 投影了 `scene 000000` 的前三帧。数值为角点投影外接框约 `(x,y,w,h)` 与 `scene_gt_info.bbox_obj` 的对照：

| 路径 / 帧 | 角点投影外接框 | `bbox_obj` | `t_z` |
|---|---|---|---:|
| `dataset/train_pbr/000000`, ID `0` | `(315.1,229.2,20.0,25.4)` | `(180,53,277,384)` | `1798.7 mm` |
| `dataset/train_pbr/000000`, ID `1` | `(315.3,234.4,20.0,15.1)` | `(196,135,263,201)` | `2487.9 mm` |
| `dataset/train_pbr/000000`, ID `2` | `(315.3,232.1,19.8,19.9)` | `(198,119,249,245)` | `2642.9 mm` |

这些差距远超普通 AABB 与物体轮廓的差异，**不能直接拿当前 PLY 为 `dataset/train_pbr` 生成正确角点标签**。作为对照，同一候选模型用于 `dataset/dji-action4/train_pbr/000000` 的样本时，投影与其 `bbox_obj` 接近：帧 `0` 投影 `(262.7,314.4,46.5,46.7)`，标注框 `(265,317,39,40)`。这说明当前 PLY 与嵌套路径样本至少局部相容，但用户已经选择根目录数据，不能据此换路径。

诊断时，仅把当前 AABB 角点在模型坐标系中临时乘一个比例，用不同帧的 `bbox_obj` 对照；`000000` 的 ID `0..5`、`000005` 的 ID `0..2` 的最佳比例约为 **12.4–13.0**，大部分外接框的均方根差为数个像素。这提示根目录图像和 `cam_R_m2c` 可能大体一致，但渲染时模型的尺寸明显不是当前 `70.5 mm` PLY 的尺寸。**这些比例只是诊断结果，不是模型来源或公制尺度的证明**。先找该批数据对应的原始模型或生成参数，再验证多场景投影；不能把 12.7 当成常数直接写进训练代码。若未来要把标签接入 PnP，还需明确最终三维模型与位移向量使用同一公制单位。

## 检查用本地文件

- `dataset/train_pbr/000000/scene_gt.json`、`scene_camera.json`、`scene_gt_info.json`、`rgb/000000.jpg`
- `dataset/dji-action4/models/models_info.json`、`obj_000001.ply`
- `dataset/dji-action4/train_pbr/000000/scene_gt.json`、`scene_camera.json`、`scene_gt_info.json`
- `scripts/glb_to_bop.py`（模型 mm 尺度和中心化）
- `third_party/BoxDreamer/src/datasets/utils/base/bbox_utils.py`（角点热图函数）
- `third_party/BoxDreamer/src/datasets/utils/base/camera_utils.py`（投影）
