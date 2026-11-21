# ResNet5000 裁剪扰动 A/B 服务器实验

## 目的与实验边界

验证当前网络对裁剪位置和尺度敏感的问题，能否通过训练时的裁剪扰动缓解。复用已有 5000 张清晰和 5000 张固定模糊图片，不重新渲染、不重新分组、不上传完整数据集。旧模型和旧结果保留。仅准备了本地短程检查，正式实验需要在服务器执行；不能把短程结果当作泛化结论。

| 项目 | A | B |
|---|---|---|
| 起点 | 原5000配对训练 best.pth | 完全相同的文件 |
| 网络／损失／解码 | ResNet50-FPN / focal+10×coordinate / sigmoid weighted Top-9 | 相同 |
| 训练图 | 4000清晰+对应4000模糊，共16000实例 | 相同 |
| 裁剪 | padding=1.3固定 | 中心独立偏移±5% bbox宽高，padding均匀采样[1.2,1.4] |
| 优化器 | 新建AdamW，不恢复旧动量 | 相同 |
| 学习率 | head=1e-4，backbone=1e-5，cosine，全部解冻 | 相同 |
| 训练预算 | 30轮，batch32，每轮500次更新 | 相同 |
| 精度 | BF16 AMP | 相同 |
| 主比较权重 | 第30轮 last.pth | 相同 |

这是从同一起点出发的配对继续训练实验，不是从头训练，也不是完整断点续训。两组共同改变了旧100轮任务的优化器状态、学习率、batch和精度；这些在A/B之间一致，所以比较的是此次继续训练条件下裁剪扰动的增量效果。单个随机种子用于筛选方向，不能宣称统计显著。

BF16用于避免本地FP16短程试验中出现的GradScaler溢出跳步。两组均使用BF16，最终评估统一FP32；原训练入口默认FP16未改。4090支持情况由PyTorch运行时检查，不静默降级到CPU。

## 数据和标签如何保持一致

直接从原始RGB取新裁剪，不对已经缩小的256图反复放缩。由新裁剪同步更新K_input，重新投影八角点并生成64×64高斯热图。三维角点顺序、物体R/t、单位不变，传给网络的图没有画框。

扰动使用seed、epoch、scene、frame、instance组成的局部随机源，不消耗采样器或模型的随机数。每轮变化且可复现；DataLoader不开persistent_workers。不缓存最终训练裁剪，否则会把随机扰动冻结。验证图没有随机扰动。

为保留监督，候选裁剪若丢失原裁剪有效角点，则最多重采样8次，仍不满足则退回原裁剪。因此最终被接受的扰动不再是严格均匀分布，这是有意的边界保护。原本无效的角点仍由有效性掩码处理。

## 解压和定位已有资产

将 resnet5000_crop_ab_delta_20260923.tar.gz 与同名 .sha256 放在同一目录，校验并解压到新目录，不覆盖旧项目：

```bash
sha256sum -c resnet5000_crop_ab_delta_20260923.tar.gz.sha256
tar -xzf resnet5000_crop_ab_delta_20260923.tar.gz
cd resnet5000_crop_ab_delta_20260923
sha256sum -c SHA256SUMS
```

复用上次成功训练的Python/Conda环境，历史包使用过 `resnet3305`，项目通用环境也可能叫 `action4-bb8`。先查询实际环境，不要为了名字一致重装。依赖见 requirements.txt，原验证组合torch2.8.0、torchvision0.23.0、NumPy1.26.4、OpenCV4.9.0.80；本次没有新增依赖。不要安装驱动、系统CUDA、修改系统代理。

当前服务器路径来自上次回传checkpoint，请检查存在性再使用：

```bash
CROP_AB_BASE=/home/aitest/exam-project/resnet5000_cleanblur_4090
CROP_AB_DATA="$CROP_AB_BASE/dataset"
CROP_AB_INIT="$CROP_AB_BASE/outputs/resnet5000_paired/best.pth"
CROP_AB_REAL="$CROP_AB_BASE/real/manifest.json"
CROP_AB_OUT="$PWD/outputs/crop_ab_30ep"
```

若旧项目已移动，只替换这些路径。必须使用5000配对模型，而非3305模型或last.pth；起点SHA256必须为：

```text
4872f6fd9253fa63ca7df2a5123246ca964f3d1b8a8026213061f370345cedab
```

本包不含大型训练图片或起始权重。包含用户刚提供的frame0原图和两个人工bbox，评估时自动追加到旧真实诊断集；这些图不会参与训练。

## 校验和预览

```bash
nvidia-smi
python -c 'import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported())'
python -m unittest discover -s tests -p 'test_crop_jitter.py' -v
python -m unittest discover -s tests -p 'test_net.py' -v
python -m unittest discover -s tests -p 'test_bop_bb8.py' -v
python -m unittest discover -s tests -p 'test_blur_augmentation.py' -v
python scripts/run_crop_ab.py --dataset "$CROP_AB_DATA" --checkpoint "$CROP_AB_INIT" --real-manifest "$CROP_AB_REAL" --output "$CROP_AB_OUT" --action check
python scripts/preview_crop_jitter.py --dataset "$CROP_AB_DATA" --output outputs/server_crop_preview
```

旧测试若缺历史数据可能skip；本次check必须通过，不能用skip替代它。check校验10000图片哈希、1759个模型/标注元数据哈希、原始分组和trainval映射，防止训练目录误含test或模糊val。完整数据校验有磁盘读取时间，不是训练卡死。预览必须零新增角点丢失，坐标误差<0.001px。

## 正式启动

在tmux中运行。先在shell里确认环境，再创建/进入会话，并在会话内重新设置上面的四个路径变量。

```bash
tmux new -s crop-ab
```

在包目录、正确环境、变量已定义后执行：

```bash
python -u scripts/run_crop_ab.py \
  --dataset "$CROP_AB_DATA" --checkpoint "$CROP_AB_INIT" \
  --real-manifest "$CROP_AB_REAL" --output "$CROP_AB_OUT" \
  --workers 4 --action all
```

all按顺序：完整校验 → A/B各1轮短程检查（默认各64个训练实例）→ A正式30轮 → B正式30轮 → 统一评估 → 结果压缩。正式A/B始终重新加载原起点，不从smoke权重开始，也不让B加载A的权重。只使用一块GPU，两个实验串行执行。

训练时查看 `outputs/crop_ab_30ep/A/train.log`、`B/train.log` 和各自 `metrics.csv`。启动器终端只报告阶段；逐轮进度在日志里。每轮记录seconds，可用服务器实际耗时估算剩余时间，不用本地小样本耗时外推4090全量训练。

如batch32 OOM，保留失败输出，新建实验输出目录，在**所有后续命令**增加 `--batch-size 16`，两组都重新运行smoke及正式实验。不要只改B、不要缩小输入尺寸；batch16变成每轮1000次更新，两组预算仍必须一致。不要使用 --cache-data，不额外开启在线模糊、不添加旋转或其他增强。

如果A完成而B失败，修复环境原因后可保留B失败目录并将其重命名归档，然后执行 `--action train --arms B`，从共同初始权重重跑B。不要伪造status。若更改batch、学习率、数据、精度或训练步数，则需新目录重跑两组。当前没有精确断点恢复功能；只加载model不能冒充恢复optimizer/scheduler/RNG后的续训。

分阶段命令同样支持：`--action smoke`、`--action train`、`--action evaluate`、`--action pack`。train要求相同配置的smoke成功。evaluate仅接受配置一致、完成的正式A/B。输出目录不覆盖；评估重跑前保留并重命名旧review。

## 怎么判断是否改善

主比较固定使用两个last.pth，避免比较不等训练步数。每轮保存的best.pth只由未扰动清晰验证集误差选出，作为次要参考，不在真实帧或测试集上挑模型。

评估包含：

- 原模型 / A / B 的完整清晰验证、模糊验证、清晰测试、模糊测试；
- 验证图padding=1.45及左右偏移5%的压力测试；
- 旧真实诊断帧及新frame0的输入/原预测/A/B四列对照；
- 平均误差、P95、PCK@5、每family指标、全部逐实例预测和误差NPZ。

压力测试误差统一映射回基准padding1.3的256像素坐标。原有效角点即使被扰动裁掉也保留在分母，并单独报告裁掉数量，不能通过删除困难角点让指标虚高。仅使用val压力测试诊断裁剪鲁棒性，test用于方案固定后的最终比较。

支持本方向的结果是：B较A在扰动验证集稳定变好，常规测试无明显退化，多个真实帧也呈现一致改善。只在原合成测试上变好，不等于解决真实泛化。真实图无可信八角点真值，不能编造真实PCK或把框更规整当作正确。没有改善也有价值：说明仅补裁剪鲁棒性不足，应继续独立检查姿态、背景/屏幕依赖或真实监督。

## 回传

all完成后自动生成输出目录旁的 `crop_ab_30ep.tar.gz` 和 `.sha256`。只回传这两个文件，不要再打包dataset。内容包括两组best/last模型、配置、命令、起点和源文件哈希、指标、日志、完整review和预测数组。

为节省传输，压缩包内四个模型省略optimizer状态，参数、BN缓冲和配置完整保留，适合本地推理；服务器原始完整checkpoint保持不变。包内记录各原始checkpoint哈希。回传包不是精确训练断点恢复包。

把输出路径、实际GPU/PyTorch版本、两组完成轮数和更新数、结果包路径及SHA256告诉用户。
