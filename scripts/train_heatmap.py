#!/usr/bin/env python3
"""Train the ResNet+FPN 8-corner heatmap baseline on the local BOP dataset.

Example:
    .venv-hccepose/bin/python scripts/train_heatmap.py \
        --val-scenes 000008 000009 --batch-size 8 --epochs 30 \
        --out-dir outputs/heatmap_resnet18

The default pipeline crops each instance to 256x256 and outputs 64x64 Gaussian
corner heatmaps. Validation uses the
same top-k weighted decoding planned for PnP and reports the mean corner error
in input pixels plus PCK@5px.  ``best.pth``/``last.pth`` and ``metrics.csv``
are written to ``--out-dir``.
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_DIR = PROJECT_DIR / "dataset" / "action4-v4-random-validation-100"
sys.path.insert(0, str(PROJECT_DIR))

from bop_bb8 import BOPInstanceDataset, BOPModel, CORNER_SIGNS  # noqa: E402
from bop_bb8.net import (  # noqa: E402
    ResNetFPN,
    coordinate_loss,
    decode_topk,
    focal_loss,
    heatmap_loss,
    heatmap_to_input_points,
)
from bop_bb8.torch_data import HeatmapDataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_DIR / "train_pbr")
    parser.add_argument(
        "--models-info",
        type=Path,
        default=DEFAULT_DATASET_DIR / "models" / "models_info.json",
    )
    parser.add_argument("--obj-id", type=int, default=1)
    parser.add_argument("--model-scale", type=float, default=1.0)
    parser.add_argument(
        "--pose-scale", type=float, default=1.0,
        help="multiply BOP cam_t_m2c by this factor (v4 data is already in mm)",
    )
    parser.add_argument("--min-visib-fract", type=float, default=0.0)
    parser.add_argument("--val-scenes", nargs="+", default=None)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument(
        "--input-size",
        type=int,
        nargs=2,
        default=[256, 256],
        help="network input as W H (must be divisible by 4 for the stride-4 FPN output)",
    )
    parser.add_argument("--crop-mode", choices=["bbox", "full"], default="bbox")
    parser.add_argument("--crop-padding", type=float, default=1.3)
    parser.add_argument("--heatmap-kernel", choices=["gaussian", "boxdreamer"], default="gaussian")
    parser.add_argument("--heatmap-sigma", type=float, default=2.0)
    parser.add_argument("--backbone", choices=["resnet18", "resnet34"], default="resnet18")
    parser.add_argument("--fpn-dim", type=int, default=128)
    parser.add_argument(
        "--loss",
        choices=["composite", "focal", "smoothl1"],
        default="composite",
        help="focal = CornerNet focal loss (handles the peaked targets); "
        "smoothl1 = BoxDreamer baseline, collapses to an all-zero map on this data",
    )
    parser.add_argument("--focal-alpha", type=float, default=2.0)
    parser.add_argument("--focal-beta", type=float, default=4.0)
    parser.add_argument("--coordinate-loss-weight", type=float, default=10.0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr-scale", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--freeze-epochs",
        type=int,
        default=5,
        help="epochs with the backbone frozen (head-only warmup)",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="cap samples per split (smoke tests)")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--out-dir", type=Path, default=PROJECT_DIR / "outputs" / "heatmap_resnet18"
    )
    return parser.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA is not available, falling back to CPU")
        return torch.device("cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    return device


def save_checkpoint(
    path: Path,
    model: ResNetFPN,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: argparse.Namespace,
    metrics: dict,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "metrics": metrics,
        },
        path,
    )


@torch.no_grad()
def evaluate(
    model: ResNetFPN,
    loader: DataLoader,
    device: torch.device,
    criterion,
    heatmap_size: tuple[int, int],
    input_size: tuple[int, int],
    use_amp: bool,
) -> tuple[float, float, float]:
    """Return ``(loss, mean_corner_error_px, pck@5px)`` over valid corners."""
    model.eval()
    loss_sum, sample_count = 0.0, 0
    err_sum, hit_count, valid_count = 0.0, 0, 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["heatmap"].to(device, non_blocking=True)
        valid = batch["valid"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets, valid, batch["keypoints_2d"].to(device))
        loss_sum += float(loss) * images.size(0)
        sample_count += images.size(0)

        points = heatmap_to_input_points(
            decode_topk(torch.sigmoid(logits.float()), k=9), heatmap_size, input_size
        )
        distance = (points - batch["keypoints_2d"].to(device)).norm(dim=-1)
        err_sum += distance[valid].sum().item()
        hit_count += (distance[valid] <= 5.0).sum().item()
        valid_count += int(valid.sum().item())
    return (
        loss_sum / max(sample_count, 1),
        err_sum / max(valid_count, 1),
        hit_count / max(valid_count, 1),
    )


def main() -> None:
    args = parse_args()
    seed_all(args.seed)
    device = resolve_device(args.device)
    in_w, in_h = args.input_size
    if in_w % 4 or in_h % 4:
        raise SystemExit(f"--input-size must be divisible by 4 (FPN stride 4), got {in_w}x{in_h}")
    heatmap_size = (in_h // 4, in_w // 4)

    model_info = BOPModel.from_models_info(args.models_info, args.obj_id, args.model_scale)
    available = sorted(
        p.name for p in args.dataset_root.iterdir() if (p / "scene_gt.json").exists()
    )
    if not 0.0 < args.val_fraction < 1.0:
        raise SystemExit(f"--val-fraction must be in (0, 1), got {args.val_fraction}")

    dataset_kwargs = dict(
        input_size=(in_w, in_h),
        heatmap_size=heatmap_size,
        min_visib_fract=args.min_visib_fract,
        pose_scale=args.pose_scale,
        crop_mode=args.crop_mode,
        crop_padding=args.crop_padding,
        heatmap_kernel=args.heatmap_kernel,
        heatmap_sigma=args.heatmap_sigma,
    )
    if args.val_scenes:
        val_scenes = list(args.val_scenes)
        missing = [s for s in val_scenes if s not in available]
        if missing:
            raise SystemExit(f"val scenes {missing} not found under {args.dataset_root}; available: {available}")
        train_scenes = [s for s in available if s not in set(val_scenes)]
        if not train_scenes:
            raise SystemExit("no training scenes left after holding out validation scenes")
        train_torch = HeatmapDataset(
            BOPInstanceDataset(args.dataset_root, model_info, scenes=train_scenes, **dataset_kwargs)
        )
        val_torch = HeatmapDataset(
            BOPInstanceDataset(args.dataset_root, model_info, scenes=val_scenes, **dataset_kwargs)
        )
        split_description = f"train scenes={train_scenes} val scenes={val_scenes}"
    else:
        full_torch = HeatmapDataset(
            BOPInstanceDataset(args.dataset_root, model_info, scenes=available, **dataset_kwargs)
        )
        generator = torch.Generator().manual_seed(args.seed)
        order = torch.randperm(len(full_torch), generator=generator).tolist()
        n_val = max(1, int(round(len(order) * args.val_fraction)))
        val_indices = order[:n_val]
        train_indices = order[n_val:]
        if not train_indices:
            raise SystemExit("validation split leaves no training samples")
        train_torch = Subset(full_torch, train_indices)
        val_torch = Subset(full_torch, val_indices)
        split_description = f"deterministic instance split seed={args.seed} val_fraction={args.val_fraction}"
    if args.limit:
        train_torch = Subset(train_torch, range(min(args.limit, len(train_torch))))
        val_torch = Subset(val_torch, range(min(args.limit, len(val_torch))))

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    train_loader = DataLoader(train_torch, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_torch, shuffle=False, **loader_kwargs)

    print(
        f"[data] {split_description} | train samples={len(train_torch)} | "
        f"val samples={len(val_torch)}"
    )
    print(f"[data] input={in_w}x{in_h} heatmap={heatmap_size[1]}x{heatmap_size[0]} (stride 4)")
    print(
        f"[data] crop={args.crop_mode} padding={args.crop_padding} pose_scale={args.pose_scale:.12f} "
        f"kernel={args.heatmap_kernel} sigma={args.heatmap_sigma}"
    )
    print(f"[device] {device}  amp={args.amp and device.type == 'cuda'}")

    model = ResNetFPN(
        backbone=args.backbone,
        fpn_dim=args.fpn_dim,
        num_corners=len(CORNER_SIGNS),
        pretrained=not args.no_pretrained,
    ).to(device)
    backbone_params = list(model.backbone.parameters())
    head_params = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    optimizer = torch.optim.AdamW(
        [
            {"params": head_params, "lr": args.lr},
            {"params": backbone_params, "lr": args.lr * args.backbone_lr_scale},
        ],
        weight_decay=args.weight_decay,
    )
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[model] {args.backbone}+FPN parameters={n_total / 1e6:.2f}M")

    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    if args.loss == "composite":

        def criterion(prediction, targets, valid, keypoints):
            return focal_loss(
                prediction, targets, valid, alpha=args.focal_alpha, beta=args.focal_beta
            ) + args.coordinate_loss_weight * coordinate_loss(
                prediction, keypoints, valid, (in_h, in_w)
            )

    elif args.loss == "focal":

        def criterion(prediction, targets, valid, keypoints):
            return focal_loss(
                prediction, targets, valid, alpha=args.focal_alpha, beta=args.focal_beta
            )

    else:

        def criterion(prediction, targets, valid, keypoints):
            return heatmap_loss(prediction, targets, valid)

    print(f"[loss] {args.loss}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    best_err = float("inf")
    with open(args.out_dir / "metrics.csv", "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_err_px", "val_pck5", "lr", "seconds"])
        for epoch in range(args.epochs):
            freeze = epoch < args.freeze_epochs
            model.freeze_backbone(freeze)
            model.train()
            start = time.time()
            running, steps = 0.0, 0
            progress = tqdm(
                train_loader, desc=f"epoch {epoch + 1}/{args.epochs}", leave=False
            )
            for batch in progress:
                images = batch["image"].to(device, non_blocking=True)
                targets = batch["heatmap"].to(device, non_blocking=True)
                valid = batch["valid"].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    logits = model(images)
                    loss = criterion(
                        logits, targets, valid, batch["keypoints_2d"].to(device, non_blocking=True)
                    )
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running += loss.item()
                steps += 1
                progress.set_postfix(loss=f"{running / steps:.4f}")
            train_loss = running / max(steps, 1)

            val_loss, val_err, val_pck = evaluate(
                model, val_loader, device, criterion, heatmap_size, (in_h, in_w), use_amp
            )
            elapsed = time.time() - start
            writer.writerow(
                [epoch + 1, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{val_err:.4f}",
                 f"{val_pck:.6f}", optimizer.param_groups[0]["lr"], f"{elapsed:.1f}"]
            )
            csv_file.flush()

            metrics = {
                "val_loss": val_loss,
                "val_err_px": val_err,
                "val_pck5": val_pck,
                "input_size": (in_w, in_h),
                "heatmap_size": heatmap_size,
            }
            if val_err < best_err:
                best_err = val_err
                save_checkpoint(args.out_dir / "best.pth", model, optimizer, epoch, args, metrics)
            save_checkpoint(args.out_dir / "last.pth", model, optimizer, epoch, args, metrics)
            print(
                f"[epoch {epoch + 1:03d}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"val_err_px={val_err:.2f} val_pck@5={val_pck:.3f} "
                f"backbone_frozen={freeze} time={elapsed:.1f}s"
            )
    print(f"[done] best val_err_px={best_err:.2f}; checkpoints in {args.out_dir}")


if __name__ == "__main__":
    main()
