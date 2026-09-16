---
name: server-bb8-training
description: Configure and validate an Ubuntu RTX 4090 server, obtain the private DJI-Action4-proj repository and external BOP dataset, and run the BB8 ResNet-FPN heatmap training safely. Use for first-time server setup, smoke training, formal training, or diagnosing CUDA/data/startup failures for this project; do not use for Blender rendering.
---

# Server BB8 Training

Use the repository as the source of truth. Before acting, read
[`references/server-runbook.md`](references/server-runbook.md) and the checked-out
`docs/SERVER_TRAINING.md`. Do not assume the dataset was committed to Git.

## Required invariants

- Target host: Ubuntu 22.04, RTX 4090 24GB, NVIDIA driver 590.48.01.
- Use Conda environment `action4-bb8` with Python 3.10.
- Use the versions in `requirements.txt`; the verified pair is PyTorch 2.8.0 + torchvision 0.23.0, CUDA 12.8 wheels.
- Do not install or replace NVIDIA drivers, the kernel, or a system CUDA Toolkit. The PyTorch wheel supplies its CUDA runtime.
- Never print or request a private SSH key, GitHub token, password, or cookie. It is safe to show an SSH `.pub` public key.
- Dataset geometry is millimetres; use `--pose-scale 1.0`.
- Preserve the v4 eight-corner order and use explicit `--dataset-root` and `--models-info` paths.
- Do not silently fall back to CPU or `--no-pretrained` for formal training.
- Do not commit datasets, checkpoints, logs, environments, model weights, or secrets.

## Workflow

1. Inspect `nvidia-smi`, disk space, Conda availability, repository state and actual dataset paths. Report missing prerequisites before changing anything.
2. Clone `git@github.com:Twilight71721/DJI-Action4-proj.git`. If SSH authentication is absent, generate an Ed25519 key only when needed, display the public key, and pause for the user to add it to GitHub.
3. Create/activate `action4-bb8`, install `requirements.txt`, then run `python scripts/check_server_setup.py`.
4. Locate the external BOP dataset. Require `train_pbr/<scene>/scene_gt.json` and `models/models_info.json`. If absent, explain how to transfer it and stop before training.
5. Run the environment/data checker and unit tests. Treat a skipped local legacy-dataset test differently from a failure; the explicit checker must pass against the supplied dataset.
6. Run a one-epoch, 32-sample smoke training. Confirm finite loss, `metrics.csv`, `best.pth`, `last.pth`, GPU use and no OOM.
7. Inspect scene IDs. Prefer holding out entire scenes for validation. Never invent `--val-scenes`; choose only directories that exist and keep at least one training scene.
8. Start formal training inside tmux with AMP and explicit arguments. Begin with batch size 32 on the 4090; reduce it on OOM rather than changing image or heatmap geometry.
9. Report the exact command, paths, package/CUDA versions, split, output directory and first observed metrics.

## Stop conditions

Stop and ask for the missing external state when:

- the private repository cannot be accessed after the public key is registered;
- the dataset or `models_info.json` is absent;
- `torch.cuda.is_available()` is false;
- data validation, projection tests or smoke training fails;
- the requested validation scenes do not exist;
- disk space is insufficient for the dataset plus checkpoints.

Do not start a long run by weakening a failed check.
