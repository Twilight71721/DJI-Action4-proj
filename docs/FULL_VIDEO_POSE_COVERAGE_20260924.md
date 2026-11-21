# 全视频姿态素材与覆盖记录（2026-09-24）

本轮保留第 500 帧用户最后选择的累计左转 20°、上抬 2°，将工作从两个局部帧扩展到整个视频。输出目录为 `outputs/full_video_pose_coverage_20260924`，核查入口为其中的 `review/index.html`。

## 这次覆盖的含义

视频共 2850 帧、约 95 秒。每隔 25 帧设置参考，并加入末帧、标记支持的特殊视角和放下手臂时的加密参考，共 144 个参考时刻、两台相机 288 个参考视图。所有 5700 条“帧 × 相机”记录都有时间索引，没有把同一画面半边里的两台相机混成一个身份。

姿态素材包括背屏、接口侧、DJI 侧，以及侧面朝向前面/背屏的不同倾斜程度和画面内旋转。初版及过渡修正一共渲染 84 个模板、252 张 RGBA；其中 2 个初版过渡模板的 6 张图片已被替换，服务器素材清单保留 **82 个模板、246 张有效前景**。每个模板有基础角度和两个邻近角度，光照也有变化。不同背景下重复合成不会增加独立三维姿态数量。

**时间范围完整、有对应角度候选，不等于每一帧真实姿态都精确匹配。** 大部分无标记视图采用可见面粗分类，姿态角度是估计。快速转动段使用相邻参考的旋转插值，不把插值当测量结果。标记支持的姿态也受估计内参和标记安装位置影响。本轮没有生成真实位姿 GT 或宣称真实对齐误差小于某个像素。

## 未确定项

- 第 1175、1425、1450、2150 帧的左侧相机被手臂、袖口或暗部遮蔽，无法可靠确定其面朝向。核查页不为这四个参考强行展示匹配姿态；它们对应的时间邻域同样在覆盖表中标记未解决。
- 全时间轴有 138 条相机漏检记录。为了建立裁剪索引，使用邻帧框插值，但它们明确标为 `interpolated_missing_detection_not_GT`。插值框不是人工标注，且不能据此证明物体可见。
- 末帧的边界截断、手/线缆遮挡属于真实图像现象。新前景库是孤立完整机身，不代表相应遮挡与截断增强已经做好。
- 这次没有完成整段视频的逐帧相机替换。之前 16 张已接受的真实背景替换小样仍在旧包中；第 500 帧的局部合成边缘问题没有通过“保留角度”自动解决。

## 标签与验证

使用既有精确 GLB 和 HCCEPose 所用的 BlenderProc/Cycles 原生流水线，不改三维几何或角点顺序。每张图独立保存 RGBA、原生机身 mask、K、R、t，以及由同一三维模型投影的 BB8。

`audit.json` 检查原生 BOP 与清单角点投影一致性（阈值 0.001 px）、旋转矩阵、mask 包含关系、机身 mask 与三维角点投影范围，以及现有训练读取器生成的 8×64×64 热图是否有限。这个误差度量只检查合成数据内部一致性。

本次实测：252 张全部通过，最大原生 BOP 投影差异为 **0.0001954 px**。另用 3 个真实背景裁片试合成 12 张图（9 清晰、3 随机模糊），全部通过训练读取与标签一致性检查；相关 7 项单元测试通过。12 张只用于流程验证，不计入正式扩充训练集。

相关回归验证：

```bash
.venv-hccepose/bin/python -m unittest discover -s tests -p 'test_pose_replacement.py'
.venv-hccepose/bin/python -m unittest discover -s tests -p 'test_video_pose_coverage.py'
.venv-hccepose/bin/python scripts/review_full_video_pose_coverage.py \
  --output outputs/full_video_pose_coverage_20260924
```

## 服务器使用方式

压缩包是前景与标签增量，不包含大背景库、原始视频、训练权重或 Blender。新素材全部属于 `video_adaptation_all` 同一训练来源组，及其背景/模糊派生图片均限定为训练用。原视频已参与姿态设计和背景选择，不能再称作独立盲测视频。

解压到新目录，阅读包内 `README.md`。附带的 `compose_pose_backgrounds.py` 只需要 NumPy/OpenCV，在服务器已有背景目录上运行。它在 RGB 上混合清晰和多档 Gaussian 模糊，原生 mask 与三维投影保持不变；输出 BOP 的 `train_pbr`，不生成伪造的背景深度。

先生成少量合成样本检查，再扩大规模。背景应来自服务器准备的多样图片池；真实帧作背景时需使用不含残留目标相机的区域，避免粘贴出第二台无标注相机。

这次没有更改网络、解码器、损失函数或启动新训练。上一轮 5000 张清晰/模糊配对训练集和服务器结果仍保留；本轮素材尚未用于它们的训练。

## 可追溯文件

- `plan.json`：生成旋转、光照和参考来源。
- `manifest.json`：可用前景与已替代候选的不同 `usage`。
- `coverage_summary.json`：当前计数与限制。
- `frame_coverage.csv` / `frame_coverage.jsonl`：每帧每相机的时间映射和未解决状态。
- `reference_review.json`：288 个参考视图的候选来源与确认状态。
- `segment_summary.json`：连续时间分段与参考分布，分段不意味着可拆到不同测试集合。
- `audit.json`：252 张原生渲染的标签验证。
- `review/index.html`：按时间查看真实参考、候选及不确定项。

复现新输出目录：先执行 `build_full_video_pose_coverage.py --output NEW_DIRECTORY`，再加 `--refine-transition` 执行一次。依次读取 `plan.json` 的 group id 调用既有 `render_pose_replacement_pilot.py render --output NEW_DIRECTORY --group GROUP_ID`，最后运行本轮 review 脚本。不要向已渲染的 BOP group 重复追加渲染。

## 本轮文件改动

- 新增 `scripts/build_full_video_pose_coverage.py`：按实体相机关联检测框、建立参考视角和原生渲染计划、加密快速转动段。
- 新增 `scripts/review_full_video_pose_coverage.py`：复用原生标签审计、输出时间覆盖表和不确定项核查页面。
- 新增 `tests/test_video_pose_coverage.py`：验证检测顺序改变/漏检时的身份记录，以及候选旋转的合法性。
- 新增本文档，更新 `docs/DATASET_AND_POSE_STATUS_20260924.md` 和输出目录内的工作状态。既有网络、解码和训练代码没有在本轮修改。
