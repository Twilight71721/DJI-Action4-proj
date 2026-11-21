# 衬衫与前臂外观修订

原先衣物由薄片拼接，袖子整体拉宽、领片生硬，大尺度高反差噪声造成云斑外观。前臂被近似为椭圆管，拉伸后的皮肤法线贴图又产生长条凹凸感。

## 本次修改

- `scripts/action4_tailored_denim.py`：新增分片衣身、前襟和腰边、收窄且有厚度的袖管、袖口、连续翻领、贴合布面的口袋与袋盖；肩袖边界重新匹配。使用静止布料坐标生成细斜纹，并降低大块噪声反差。缝线采用短针脚。
- `scripts/action4_wrist_assets.py`：前臂轮廓采用平滑的粗细变化与轻微中心线弯曲，腕带附近和手部顶点不变。为前臂增加材质权重，降低被拉伸的法线贴图影响，保留手部原有法线细节。
- `scripts/render_action4_wrist_hccepose.py`：参考帧场景接入新版衣物、前臂，继续使用现有衣物动作变形、HCCEPose/BlenderProc 和原始 Action 4 GLB。旧场景入口保持原衣物建模方式。
- `tests/test_action4_hand_anatomy.py`：增加回归检查，确认前臂修改不移动手部与腕带连接区域。
- `scripts/review_action4_reference_dataset.py`：汇总报告纳入上一轮底部支架与机身包围体的检查。

相机贴图、左右手镜像修正、腕带曲面和底部安装逻辑沿用上一轮结果。

## 输出与验证

最终局部样张：`outputs/action4_denim_arm_checked`，扣衣扣与折袖各 2 张，共 4 张，1624×1232，Cycles 48 samples；三种光照。真实参考帧为 500 和 2490。

- 对照页：`review/appearance_comparison.html`。
- `review/validation.json`：BOP、mask、深度、独立角点投影与加载检查。
- `review/attachment_validation.json`：腕带、支架、插头、线缆间距以及底部连接检查。
- 渲染参数：`generation_config.json`；完整每帧参数及检查在 `render_metadata`。
- 回归命令：`.venv-hccepose/bin/python -m unittest discover -s tests -p 'test_action4_*.py' -v`。

前后主对照使用相同 seed（20260922）和第 500 帧参考配置。图像分辨率及采样数有差异；衣物近景仅为浏览页裁切，合成 RGB 与标注没有进行后期图像修改。

## 范围与剩余问题

这是局部外观修订，不是照片级复原。衣物褶皱仍由几何构造和动作形变产生，没有布料—手部接触求解；扣衣扣时的指尖接触、皮肤颜色和真实衣物洗水痕迹仍有差异。当前无真实头部相机标定，姿态也没有逐帧二维关键点误差验收。200 张全量批次继续保持暂停；这些小样不能当作已完成的 200 张交付。

验证结果：9 项回归测试通过；4 张 RGB、8 个相机实例全部可见，BOP 与附件检查均无错误。独立八角点投影最大误差约 0.000134 px。64 项支架—机身包围体检查通过。腕带最小采样间距约 2.162 mm，附件约 2.448 mm，线缆保守表面间距约 27.483 mm。采样时有 2 个姿态提案被拒绝并重新采样，原因保留在逐帧 metadata 中；未通过的提案未被渲染成最终样张。
