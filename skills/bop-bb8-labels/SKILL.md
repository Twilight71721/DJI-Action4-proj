---
name: bop-bb8-labels
description: 为本项目的 BOP 格式 DJI Action 4 数据生成有序的三维包围盒八角点、二维投影和八通道热图监督标签；在实现前核对模型、位姿、内参、尺度及实例索引。不用于训练网络或修改渲染器。
---

# BOP BB8 监督标签

你正在 `/home/twilight/University/njukym/cs/cv/exam-project` 中为单目 RGB → 八角点热图网络完成**标签与数据读取部分**。用户已指定后续使用 `dataset/train_pbr`，保留这个图像来源；不可自动换成另一批训练图。先读 [references/dataset-audit.md](references/dataset-audit.md)；其中的异常是已观察到的证据，不是允许你自动修正数据的指令。保持中文解释，代码和字段名保留英文。

## 开工门槛

1. 若当前项目提供 `AGENTS.md`，先读它；同时读取 `README.md`、相关代码和数据样本。确认实际仓库结构，不假设已有训练框架。
2. 核实 `dataset/train_pbr/<scene_id>/rgb/<image_id>.jpg`、同场景 `scene_gt.json`、`scene_camera.json`、`scene_gt_info.json` 的关联；按图像 ID 查记录，按 `scene_gt` 数组索引查物体实例。`scene_gt_info` 和 `mask`/`mask_visib` 使用同一实例索引。
3. 核实 3D 模型及 `models_info.json` 是否属于这批图像。选至少数张不同场景与可见度的样本，用给定 `K`、`cam_R_m2c`、`cam_t_m2c` 投影 3D 角点，将投影外接框与 `bbox_obj`、RGB 机身位置对照，并保存编号 `0..7` 的可视化。若出现系统性的尺度或位置不一致，**停止标签实施**，给出样本、数值和可能原因，请用户提供或确认 `dataset/train_pbr` 对应的模型/生成时的尺度；不可静默切换到 `dataset/dji-action4/train_pbr`，不可把从标注框反推的倍数当作已证实的缩放参数。

## 几何与标签不变量

- 使用与 `obj_id` 对应、单位为 **mm** 的模型角点；`models_info.json` 的 `min_*`/`max_*` 或 PLY 顶点范围可提供 AABB。固定角点顺序：`(xmin,ymin,zmin)`, `(xmin,ymax,zmin)`, `(xmax,ymax,zmin)`, `(xmax,ymin,zmin)`, `(xmin,ymin,zmax)`, `(xmin,ymax,zmax)`, `(xmax,ymax,zmax)`, `(xmax,ymin,zmax)`。保持热图通道、二维角点和 PnP 三维角点的索引一致。
- BOP 的 `cam_R_m2c` 是 row-major 的 9 个值，`cam_t_m2c` 是 3 个值；均为**模型坐标 → OpenCV 相机坐标**。直接算 `P_c = R_m2c @ P_m + t_m2c`。不要再组合 `cam_R_w2c`/`cam_t_w2c`，不要反转该物体位姿，也不要只缩放 `P_m` 或 `t` 中一个。
- 优先用每帧 `scene_camera.json[image_id].cam_K`，row-major 重排为 `3×3`。对 `Z_c > epsilon` 的角点执行 `q = K @ P_c`、`(u,v)=(q_x/q_z,q_y/q_z)`。校验数值有限性；对相机后方或图像范围外的角点返回 `valid=False`，不要在图像边缘伪造峰值。
- 若输入经过裁剪与缩放，以裁剪框 `(x0,y0,wc,hc)`、目标 `(W,H)` 更新像素内参：`K_crop = [[sx,0,-sx*x0],[0,sy,-sy*y0],[0,0,1]] @ K`，其中 `sx=W/wc`、`sy=H/hc`。若热图比输入图更小，再按热图宽高缩放投影坐标或内参；禁止混用原图、输入图和热图坐标。
- 基线热图严格依据本地 `third_party/BoxDreamer/src/datasets/utils/base/bbox_utils.py::make_bbox_features(type="heatmap")`：对每个像素算 `d_i=sqrt((x-u_i)^2+(y-v_i)^2)`，`center=mean_i(p_i)`，`scale_i=(||p_i-center||/10)^2`，`h_i=exp(-d_i/scale_i)`。输出按角点顺序为 `[8,Hh,Wh]`。加上除零保护和逐实例归一化；无效角点的通道只保留背景值，不做边缘归一化。明确说明这不是标准二维 Gaussian。若完全复现 BoxDreamer，把 `[0,1]` 映射到 `[-1,1]`；若选择其他范围，必须与未来预测头/损失一致，并写进配置或测试。
- `visib_fract` 表示机身像素可见比例，不等同于每个虚拟包围盒角点可见。可把低可见度样本筛选或分层评估，但不要把遮挡角点随意改为图像边界，也不要假定物体轮廓角点就是 3D AABB 投影角点。

## 交付与验证

通过数据审计后，再以最小范围实现可复用的数据读取与标签生成函数；不要实施 ViT、训练、PnP 或无关重构。返回每实例至少 `image`, `corners_3d[8,3]`, `keypoints_2d[8,2]`, `valid[8]`, `heatmaps[8,Hh,Wh]`, `K_input[3,3]`, `R_m2c[3,3]`, `t_m2c[3]`, `scene_id`, `image_id`, `instance_idx`，并说明坐标系、单位、数值范围。按已有项目风格命名和组织文件。

测试必须覆盖：8 通道/顺序与形状、投影公式和单位、裁剪内参、`Z<=0`/图外点、热图峰值位置、无 NaN、图像 ID 与实例索引、可视化线框与 `bbox_obj` 的对照。先运行项目已有的相关测试；若没有测试框架，增加针对这部分的小型测试。最后列出改动文件、运行命令、通过结果与尚未解决的问题。不要自动 `git commit` 或 `git push`。
