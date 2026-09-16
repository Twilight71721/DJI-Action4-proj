# Project server runbook

## Repository and environment

- SSH repository: `git@github.com:Twilight71721/DJI-Action4-proj.git`
- Expected checkout: `/home/aitest/exam-project/DJI-Action4-proj`
- Conda environment: `action4-bb8`
- Python: 3.10
- Verified packages: torch 2.8.0+cu128, torchvision 0.23.0+cu128,
  numpy 1.26.4, OpenCV 4.9.0.80, tqdm 4.70.1.

The driver-reported CUDA version is not the version that must be installed as a
Toolkit. Driver 590.48.01 can execute the CUDA 12.8 runtime packaged with the
PyTorch wheel.

## Data contract

The repository intentionally excludes generated data. A dataset directory must contain:

```text
<dataset>/models/models_info.json
<dataset>/train_pbr/<scene>/rgb/
<dataset>/train_pbr/<scene>/depth/
<dataset>/train_pbr/<scene>/mask/
<dataset>/train_pbr/<scene>/mask_visib/
<dataset>/train_pbr/<scene>/scene_camera.json
<dataset>/train_pbr/<scene>/scene_gt.json
<dataset>/train_pbr/<scene>/scene_gt_info.json
```

For Action 4 v4, object id is 1, dimensions and pose translations are in mm,
and `--pose-scale 1.0` is mandatory. The eight corner order is defined by
`configs/action4_v4/corners_3d.json`.

## Pretrained weights

Torchvision downloads `resnet18-f37072fd.pth` on the first pretrained run. If
the server cannot reach the host, copy the file to
`~/.cache/torch/hub/checkpoints/`. Do not put it in Git. Running with
`--no-pretrained` is a different experiment and requires explicit user choice.

## Failures

- `Repository not found`: confirm the exact `.git` URL and that the registered GitHub account has access to the private repository.
- `Permission denied (publickey)`: show `~/.ssh/id_ed25519.pub`, never the private key, and wait for GitHub registration.
- `torch.cuda.is_available() == False`: verify the active Conda environment and installed `+cu128` wheel before considering system changes.
- CUDA OOM: reduce batch size 32 → 16 → 8.
- Missing pretrained download: transfer the cached 47MB file, or obtain explicit approval to use `--no-pretrained`.
- Missing scene: enumerate actual numeric scene directories and adjust the split; never reuse validation scenes for training merely to make the command run.
