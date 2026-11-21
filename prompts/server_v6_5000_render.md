# 任务：在服务器上跑 DJI Action 4 v6 正式 5000 张 Blender 渲染

服务器：Ubuntu 22.04，RTX 4090 24GB，驱动 590.48.01，用户 `aitest`。
背景：Windows 上的 Blender 渲染代码已搬到服务器。目标是在服务器完成
`dataset/action4-v6-train-5000`（10 个 scene × 500 帧）的渲染、验证和打包。
所有操作不使用 sudo，不安装系统 CUDA/驱动，不改动系统 Python。

## 第 0 步：定位代码与确认修复已包含

1. 先找到搬过来的仓库（含 `run_render.py` / `render_action4.py` / `action4_domain_v6.py` /
   `validate_action4_random.py` / `package_action4_regression.py`），在常见位置查找，例如：
   `find /home/aitest/exam-project -maxdepth 4 -name run_render.py`。
2. 检查代码是否已包含三个修复（缺失就停止并报告，不要在旧代码上渲染）：
   - occluder 按渲染后实际像素命中 55/30/15 目标等级（确定性调度 + 实际 mask 像素判定）；
   - 尺寸接受条件基于实际 mask（full mask bbox 宽 ≥ 100px、visible mask 面积 ≥ 2%、可见比例 ≥ 0.60）；
   - 输出 JSON 中无 `C:\` 绝对路径，路径为相对数据集根目录。
3. 确认资产目录存在：`assets/experiments/dji-action4-glb-aligned-v4`（GLB 约 92MB 和相关 metadata）。

## 第 1 步：准备渲染环境（上次卡在 bpy 下载）

- 项目内已有 Miniconda：`/home/aitest/exam-project/miniconda3`，虚拟环境 `action4-render`
  （Python 3.10）。环境存在就直接用：
  ```bash
  source /home/aitest/exam-project/miniconda3/etc/profile.d/conda.sh
  conda activate action4-render
  python --version   # 应为 3.10.x
  ```
- 安装/补齐依赖，bpy 必须使用 Blender 官方 wheel（不要从清华镜像找，PyPI 没有 3.6.0）：
  ```bash
  cd <渲染仓库目录>
  curl -L --retry 20 --retry-all-errors -C - \
    -o bpy-3.6.0-cp310-cp310-manylinux_2_28_x86_64.whl \
    https://download.blender.org/pypi/bpy/bpy-3.6.0-cp310-cp310-manylinux_2_28_x86_64.whl
  ls -l bpy-3.6.0-cp310-cp310-manylinux_2_28_x86_64.whl   # 必须为 371,432,584 字节
  echo "ede2c95ace7848f4f5a075a7d8cc3a9e643c335f4596c3c15087d93c7ae5f56a  bpy-3.6.0-cp310-cp310-manylinux_2_28_x86_64.whl" | sha256sum -c -
  ```
  校验不通过严禁安装；删除坏文件后重试。如果多次下载仍失败，停止并报告，
  由我从本地（已确认该 URL 可用）下载后用 ToDesk 传给你。
- 其余依赖用清华 PyPI 镜像安装，然后验证导入：
  ```bash
  python -m pip install --timeout 120 --retries 10 \
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple -r requirements-render.txt
  PYTHONPATH="$PWD/vendor" python -c 'import bpy, blenderproc; print(bpy.app.version_string, blenderproc.__version__)'
  ```
  预期输出 `3.6.0 2.5.0`。vendor 目录若不存在：
  `unzip -qn HCCEPose/blenderproc.zip -d vendor`。

## 第 2 步：1 帧 smoke（验证 headless + GPU + mask）

- 确认 EGL 可用：`ldconfig -p | grep libEGL`；`nvidia-smi` 正常。
- 用 1 帧试渲染（新目录，不要覆盖已有数据），确认：RGB 非黑、depth 有非零值、
  mask 和 mask_visib 非空且 mask_visib 是 mask 子集、BOP JSON 正常生成。
- 如报 EGL/pyrender 错误，先报告完整报错，不要盲目改系统库。

## 第 3 步：校准集必须先通过

- 若仓库里已有 `dataset/action4-v6-calibration-40`，直接运行校验；
  若没有，先生成 40 帧（至少 2 个 scene，固定种子）。
- 运行 `validate_action4_random.py --allow-truncation --min-bop-visibility 0.60`，检查：
  实际遮挡等级分布落在 55/30/15 ±3%；接受帧满足 100px / 2% / 0.60；
  BOP schema、K/R/t、角点定义与 v6 pilot 一致；JSON 无绝对路径。
- 不满足就停止并报告（不要放宽门槛继续渲染 5000 张）。

## 第 4 步：启动 5000 张渲染

```bash
source /home/aitest/exam-project/miniconda3/etc/profile.d/conda.sh
conda activate action4-render
tmux new -s action4-v6-render
cd <渲染仓库目录>
python run_render.py \
  --dataset-root dataset/action4-v6-train-5000 \
  --model-asset glb --background gray --lighting-mode soft-area \
  --validation-randomization --domain-v6 --random-intrinsics --fill-light \
  --color-file-format PNG --texture-interpolation Linear --cycles-samples 50 \
  --max-occluders 1 \
  --batches 10 --frames-per-batch 500 --frames-per-scene 500 --seed 665600 \
  --camera-min-m 0.14 --camera-max-m 0.36 \
  --min-projected-box-area 0.04 --max-projected-box-area 0.25 \
  --min-projected-width-px 128 --max-projected-width-px 300 \
  --mask-workers 0 --gpu-id 0 2>&1 | tee dataset/action4-v6-train-5000/render.log
```
说明：参数以修改后的实际参数为准（若新增了遮挡等级调度的开关，按代码默认值使用）。
逐 scene 渲染：每个 scene 完成后立即做硬性检查；任一 scene 失败就停止并报告，
不要静默继续。参考速度约 3.9 秒/帧（4060），4090 会更快，总耗时以日志为准。

## 第 5 步：全量验证与报告

1. 对 5000 张运行完整校验（`--allow-truncation --min-bop-visibility 0.60`），
   硬性检查必须全部通过；输出 `validation-summary.json`、`validation-report.md`、
   按遮挡等级/尺寸分层的 contact sheet。
2. 报告：每 scene 帧数与失败重采样次数、实际遮挡分布、尺寸分布、截断比例、
   总耗时与峰值显存、输出路径。
3. 验证通过后，把数据集放到训练仓库可读取的位置（例如
   `/home/aitest/exam-project/DJI-Action4-proj/dataset/action4-v6-train-5000`，
   复制或软链接均可），并用训练仓库的 `scripts/check_server_setup.py` 做一次读取自检。
4. 不要 commit、不要 push；数据集不进 Git。
