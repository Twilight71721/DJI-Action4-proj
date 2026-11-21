# Action 4 底部安装修正（2026-09-21）

旧抬腕图的连接杆终点在相机局部 `-Y` 底部，但杆身从转轴直接斜穿机身侧壁。之前仅检查附件与皮肤，不能发现这种机身相交。此外，真实第 0 帧的转轴在朝向观察相机的腕带表面，旧采样器却把抬腕也套成双腕内侧安装。

## 修正

- `scripts/action4_reference_profiles.py`：抬腕安装面按真实第 0 帧改为朝观察相机的腕带表面；相机底部置于短连接件上方。桌面安装仍随腕部姿态估计。新增相机包围体与最终支架网格的三角形相交和内部顶点检查，底部安装板同时检查贴合间隙。
- `scripts/action4_wrist_assets.py`：增加独立底部安装板和连接颈。连接路径先位于机身底面以外，再到安装板；不再用终点位于底部作为安装正确的证据。
- `scripts/render_action4_wrist_hccepose.py`：渲染前执行该检查，写入每帧 `bottom_mount_clearance`。
- `scripts/run_action4_reference_200.py`：拒绝继续使用已标记 `INCOMPLETE_REJECTED.md` 的输出目录，避免修正前后混在同一批次。
- `tests/test_action4_bottom_mount.py`：正常底部连接通过；旧的“底部终点但杆身斜穿侧壁”必须被拒绝。

## 样张与验证

- 抬腕：`outputs/action4_bottom_reference_mount/review/index.html`，参考真实第 0 帧。
- 桌面：`outputs/action4_bottom_mount_table_check/train_pbr/000000/rgb/000000.png`。
- 检查汇总：`outputs/action4_bottom_reference_mount/review/attachment_checks.json`。
- 两类样张的支架与机身保守包围体相交数均为 0，底部安装板与包围体底面间距约 0.125 mm（小于 0.5 mm 贴合容差）。这不是测量得到的磁吸底座精确 CAD。
- `./.venv-hccepose/bin/python -m unittest discover -s tests -p 'test_action4_*.py' -v`。

相机 GLB 不重建、不变更拓扑，后屏标签及前屏/镜头划分保持原有已确认设置。腕带、附件对自身皮肤、线缆对双臂的检查仍在执行。

## 批次状态及限制

`dataset/action4-reference-fitted-200-20260921` 已停止并标记不合格，保留用于排查，不能作为最终 200 张交付。本次仅验证抬腕与桌面安装结构样张。此前试做的长折弯支架 `outputs/action4_bottom_mount_check` 仅是排查中间结果，不符合真实抬腕安装关系，未被选为最终方案。

当前没有真实观察相机内参，手部、相机倾角及衣物形状仍是参考帧引导的估计，未达到逐帧二维误差验收；衣物也没有物理接触求解。不得把局部安装检查通过表述为完整场景已与真实视频一致，或已经证明训练泛化能力。
