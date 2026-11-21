# 逐帧分类与采样规范

## 输出

以 `frame_index_zero_based` 为主键输出 `frame_taxonomy.csv`，至少包含：

```text
frame_index_zero_based,timestamp_seconds,filename,segment_id,scene_stage,
hand_pose_left,hand_pose_right,camera_view_left,camera_view_right,
screen_state_left,screen_state_right,background,occlusion_left,occlusion_right,
truncation_left,truncation_right,motion_blur,exposure,instance_count,
review_status,confidence,notes
```

另输出 `segment_summary.json`，记录每个连续片段的起止帧、主标签、事件持续时间、代表帧和人工复核结果。

## 标签

- `scene_stage`: `raised_display`, `lowering_transition`, `tabletop_manipulation`, `raised_gesture_end`。
- `hand_pose_*`: `open`, `grasp`, `pinch`, `fist`, `thumbs_up`, `partial_or_unknown`。
- `camera_view_*`: `rear_screen`, `top_rear_oblique`, `side`, `front_lens`, `bottom_mount`, `unknown`。
- `screen_state_*`: `on_readable`, `on_glare`, `off_or_dark`, `occluded`, `not_visible`。
- `background`: `pale_wall`, `grid_mat`, `denim`, `mixed`。
- `occlusion_*`: `0_05`, `05_20`, `20_35`, `gt_35`, `unknown`。
- `truncation_*`: `none`, `mild`, `strong`, `not_visible`。
- `motion_blur`: `none`, `mild`, `strong`。
- `exposure`: `normal`, `bright`, `dark`, `mixed`。
- `review_status`: `auto`, `human_verified`, `needs_review`。

不要把 `visib_fract` 当成每个 BB8 角点的可见性。遮挡比例若无实例分割只能标为估计区间，并在 `notes` 说明。

## 分类流程

1. 用时间邻接建立初始 segment；检测明显场景/动作变化作为候选边界。
2. 每个 segment 至少检查首、中、尾帧；低置信度、少数类、强遮挡和边界帧全部人工复核。
3. 自动模型只能提出候选，不允许把臆测的精确标签当真值。
4. 统计两套分布：原始帧时长分布，以及去除相邻冗余后的事件分布。
5. 合成配额以事件分布为主，对困难少数类做有记录的过采样。

## 数据划分

真实数据若用于训练/验证，必须按连续 `segment_id` 划分；禁止随机逐帧划分。相同动作事件、近邻帧和几乎相同的取景不能跨 split。

参考视频已经用于合成域设计，因此它只适合作为域内诊断/适配评估。需要无偏测试时，使用另一段没有参与分类和调参的视频。

## 合成配额最低覆盖

pilot 应覆盖所有主要 `scene_stage`、至少五种 `camera_view`、五种手势、三档 0–35% 遮挡、三档截断、单双实例、亮屏可读/反光/遮挡。对真实视频没有证据的组合不要无依据大量生成。
