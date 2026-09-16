# 给 DeepSeek V4.1 Flash 的服务器训练 Prompt

将下面完整内容发送给服务器上的 DeepSeek V4.1 Flash。方括号占位符需要先替换；如果
暂时不知道数据集路径，可以保留，要求它检查后向你确认。

```text
你现在负责在一台 Ubuntu 22.04、NVIDIA RTX 4090 24GB 的服务器上部署并训练
DJI Action 4 的 BB8 八角点热图网络。服务器 NVIDIA driver 是 590.48.01，
nvidia-smi 显示 CUDA Version 13.1。

你没有之前对话的记忆。必须先取得并完整阅读仓库内：

- AGENTS.md（如果存在）
- README.md
- requirements.txt
- docs/SERVER_TRAINING.md
- skills/server-bb8-training/SKILL.md
- skills/server-bb8-training/references/server-runbook.md
- scripts/train_heatmap.py
- scripts/check_server_setup.py

GitHub 私有仓库：

git@github.com:Twilight71721/DJI-Action4-proj.git

预期工作目录：

/home/aitest/exam-project/DJI-Action4-proj

请依次完成以下任务，并在每一步给出实际命令和结果，不要凭空假设成功。

1. 检查 `nvidia-smi`、`df -h`、Conda、Python、当前目录和已有 Git 仓库。

2. 如果仓库不存在，从上述 SSH URL clone。若服务器没有 GitHub SSH key：
   - 使用 Ed25519 创建 key；
   - 只打印 `.pub` 公钥；
   - 绝不打印私钥、token 或密码；
   - 提醒我把公钥加入有权访问该私有仓库的 GitHub 账户；
   - 在我完成之前暂停 clone，不要改用包含明文 token 的 URL。

3. 进入仓库，读取 `skills/server-bb8-training/SKILL.md` 并严格执行。创建 Conda
   环境 `action4-bb8`，Python 3.10。按照 `requirements.txt` 安装依赖。

4. 不要安装或修改 NVIDIA 驱动、Linux 内核或系统 CUDA Toolkit。项目使用
   PyTorch 2.8.0 + CUDA 12.8 wheel；驱动 590.48.01 能够运行它。不要因为
   nvidia-smi 显示 CUDA 13.1 就寻找 CUDA 13.1 版 PyTorch。

5. 运行 `python scripts/check_server_setup.py`，必须确认：
   - torch 为 2.8.0+cu128；
   - torchvision 为 0.23.0+cu128；
   - torch.cuda.is_available() 为 True；
   - GPU 是 RTX 4090；
   - 网络输出为 1×8×64×64；
   - 没有静默回退到 CPU。

6. 数据集不会从 GitHub clone 下来。请检查下面的候选位置以及项目 `dataset/`
   目录，不要假设它已经存在：

   [DATASET_DIRECTORY_OR_UNKNOWN]

   有效数据集必须包含：
   - models/models_info.json
   - train_pbr/数字场景目录/rgb
   - depth
   - mask
   - mask_visib
   - scene_camera.json
   - scene_gt.json
   - scene_gt_info.json

   如果数据集不存在，停止训练并告诉我：服务器应放置的准确目标路径、从 Windows
   使用 scp 传输的命令模板，以及如何计算和核对 SHA-256。不要创建空数据集，也不要
   下载不相关的公开数据代替。

7. 找到数据后，设置只在当前 shell 使用的 `DATASET_DIR`，运行：

   python scripts/check_server_setup.py \
     --dataset-root "$DATASET_DIR/train_pbr" \
     --models-info "$DATASET_DIR/models/models_info.json"

   然后运行 `python -m unittest discover -s tests -v`。明确区分“因为旧本地数据不存在
   而 skip”的测试和真正失败。新数据必须通过显式 server setup checker。

8. 枚举真实存在的 scene。不要发明 000008、000009。如果只有一个 scene，先使用
   固定 seed 的 instance split；如果有多个独立 scene，优先保留完整 scene 作为验证集。
   汇报最终训练/验证划分。

9. 先运行 1 epoch、最多 32 个样本的 smoke training。使用 CUDA、AMP、ResNet18、
   256×256 bbox crop、64×64 Gaussian heatmap、sigma=2、pose-scale=1.0 和 composite
   loss。确认 loss 有限，且生成 metrics.csv、best.pth、last.pth。

10. smoke 完全通过后才允许开始正式训练。正式训练使用 tmux，初始参数：
    - batch size 32；
    - epochs 50；
    - freeze epochs 5；
    - num workers 8；
    - learning rate 0.001；
    - backbone learning-rate scale 0.1；
    - weight decay 0.0001；
    - seed 460731；
    - CUDA + AMP；
    - 输出到 outputs/action4_v5_resnet18。

    所有数据路径和关键参数必须显式写入命令。若 CUDA OOM，只依次降低 batch size
    32→16→8，不要改变图像、热图、角点顺序或 pose scale。

11. 默认使用 ImageNet 预训练 ResNet18。首次运行可能下载约 47MB 的
    `resnet18-f37072fd.pth`。如果下载失败，告诉我把该文件传到：

    ~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth

    不得擅自使用 `--no-pretrained` 改变实验条件。

12. 最终汇报：仓库 commit；Python、PyTorch、torchvision、CUDA runtime、GPU；
    数据集和 models_info 的绝对路径；scene 数和样本数；训练/验证划分；smoke 命令与
    结果；正式训练完整命令；tmux session；输出和日志路径；首个 epoch 指标；显存占用；
    尚未解决的问题。

重要约束：

- 不要 commit 或 push；
- 不要把数据集、checkpoint 或日志上传 GitHub；
- 不要删除服务器现有数据；
- 不要修改驱动、内核、系统 CUDA 或全局 shell 配置；
- 不要泄露 SSH 私钥或凭据；
- 不要在检查失败时强行启动长时间训练；
- 不要把 100 张 v4 管线验证集误称为正式训练集；
- Action 4 v4 的单位是 mm，必须使用 `--pose-scale 1.0`；
- 八角点编号不能改变。
```
