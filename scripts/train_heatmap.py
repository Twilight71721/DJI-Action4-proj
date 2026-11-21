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
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import cv2
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
    decode_heatmaps,
    focal_loss,
    heatmap_loss,
    spatial_kl_loss,
    heatmap_to_input_points,
)
from bop_bb8.torch_data import HeatmapDataset, PairedHeatmapDataset, ShapeBucketBatchSampler  # noqa: E402
from bop_bb8.net.loss import spatial_js_loss  # noqa: E402
from bop_bb8.augment import RandomBlurDataset  # noqa: E402
from bop_bb8.background import RandomBackgroundBOPDataset  # noqa: E402
from bop_bb8.online_background import OnlinePoseBackgroundDataset  # noqa: E402
from bop_bb8.photometric import PhotometricDataset  # noqa: E402
from bop_bb8.view_sampling import dataset_view_bins, view_weights, FACE_NAMES  # noqa: E402


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
    parser.add_argument('--train-min-visib-fract', type=float, default=None,
                        help='Training-only visibility filter; validation keeps min-visib-fract')
    parser.add_argument("--val-scenes", nargs="+", default=None)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument(
        "--input-size",
        type=int,
        nargs=2,
        default=[256, 256],
        help="network input as W H (must be divisible by 4 for the stride-4 FPN output)",
    )
    parser.add_argument("--crop-mode", choices=["bbox", "full", "cuboid", "rect"], default="bbox")
    parser.add_argument('--rect-sizes', nargs='+', type=lambda value: tuple(map(int, value.split('x'))),
                        help='Aspect buckets as WxH, e.g. 320x192 256x256 192x320; no RGB padding')
    parser.add_argument('--heatmap-margin', type=int, default=0,
                        help='Output-only margin in stride-4 cells; does not pad RGB inputs')
    parser.add_argument('--require-all-corners', action='store_true')
    parser.add_argument('--decoder', choices=['topk', 'local'], default='topk')
    parser.add_argument('--decode-radius', type=int, default=3)
    parser.add_argument('--freeze-bn', action='store_true',
                        help='Keep pretrained backbone BatchNorm statistics; train affine weights')
    parser.add_argument("--crop-padding", type=float, default=1.3)
    parser.add_argument("--crop-border", type=int, default=4,
                        help="Solid-color input-pixel border for tight cuboid crops")
    parser.add_argument("--crop-jitter-shift", type=float, default=0.0)
    parser.add_argument("--crop-jitter-padding", type=float, nargs=2, default=None)
    parser.add_argument("--background-assets", type=Path, default=None)
    parser.add_argument("--online-background-config", type=Path, default=None)
    parser.add_argument("--background-probability", type=float, default=0.8)
    parser.add_argument("--init-checkpoint", type=Path, default=None,
                        help="Load model weights strictly; restart optimizer and schedule for BOTH arms")
    parser.add_argument("--strict-updates", action="store_true",
                        help="Fail on AMP skipped updates so paired runs have equal optimizer steps")
    parser.add_argument("--heatmap-kernel", choices=["gaussian", "boxdreamer"], default="gaussian")
    parser.add_argument("--heatmap-sigma", type=float, default=2.0)
    parser.add_argument("--backbone", choices=["resnet18", "resnet34", "resnet50"], default="resnet18")
    parser.add_argument("--overfit", action="store_true", help="train/evaluate on all the same instances; NOT a validation split")
    parser.add_argument("--cache-data", action="store_true", help="cache deterministic tensor crops in CPU RAM")
    parser.add_argument("--blur-probability", type=float, default=0.0,
                        help="training-only random blur probability; 0 preserves the baseline")
    parser.add_argument("--blur-sigma", type=float, nargs=2, default=[0.4, 1.6],
                        help="Gaussian blur sigma range in input-crop pixels")
    parser.add_argument("--blur-kernel-size", type=int, default=9)
    parser.add_argument("--cosine-lr", action="store_true")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--fpn-dim", type=int, default=128)
    parser.add_argument(
        "--loss",
        choices=["composite", "focal", "smoothl1", "spatial_kl"],
        default="composite",
        help="focal = CornerNet focal loss (handles the peaked targets); "
        "smoothl1 = BoxDreamer baseline, collapses to an all-zero map on this data",
    )
    parser.add_argument("--focal-alpha", type=float, default=2.0)
    parser.add_argument("--focal-beta", type=float, default=4.0)
    parser.add_argument("--coordinate-loss-weight", type=float, default=10.0)
    parser.add_argument('--paired-background', action='store_true', help='Two RGB views per base instance')
    parser.add_argument('--consistency-weight', type=float, default=0., help='Spatial JS weight; zero is the paired control')
    parser.add_argument('--photometric-augmentation',action='store_true')
    parser.add_argument('--photometric-profile',choices=['pilot','balanced'],default='pilot')
    parser.add_argument('--view-balance',action='store_true',help='Capped inverse-sqrt view frequency sampling within shape buckets')
    parser.add_argument('--view-weight-cap',type=float,default=3.)
    parser.add_argument('--deterministic-cudnn',action='store_true')
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
    parser.add_argument("--amp-dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument(
        "--out-dir", type=Path, default=PROJECT_DIR / "outputs" / "heatmap_resnet18"
    )
    args = parser.parse_args()
    if args.view_balance and args.rect_sizes is None:
        parser.error('view-balance requires rectangular shape buckets')
    if not np.isfinite(args.view_weight_cap) or args.view_weight_cap<1:
        parser.error('view-weight-cap must be finite and >= 1')
    if (args.view_balance or args.photometric_augmentation) and (args.overfit or args.cache_data):
        parser.error('View/photometric experiments require uncached, held-out validation')
    if not np.isfinite(args.consistency_weight) or args.consistency_weight < 0:
        parser.error('consistency-weight must be finite and nonnegative')
    if args.consistency_weight and not args.paired_background:
        parser.error('consistency-weight requires paired-background')
    if args.paired_background and (args.online_background_config is None or args.loss != 'spatial_kl'):
        parser.error('paired-background requires online-background-config and spatial_kl')
    if args.heatmap_margin < 0 or args.decode_radius < 0:
        parser.error('heatmap-margin and decode-radius must be >= 0')
    if args.crop_mode == 'rect':
        if args.crop_padding != 1.0 or args.crop_border != 0:
            parser.error('rect crops require --crop-padding 1 --crop-border 0')
        if not args.val_scenes and not args.overfit:
            parser.error('rect protocol requires grouped --val-scenes; instance random split is disabled')
    elif args.rect_sizes is not None:
        parser.error('rect-sizes requires --crop-mode rect')
    if args.crop_jitter_shift or args.crop_jitter_padding is not None or args.background_assets is not None or args.online_background_config is not None:
        if args.cache_data or args.overfit or not args.val_scenes or args.crop_mode not in {"bbox", "cuboid", "rect"}:
            parser.error("crop/background augmentation requires bbox, explicit val scenes, no cache-data and no overfit")
    if args.online_background_config is not None and (args.background_assets is not None or args.blur_probability):
        parser.error("online-background-config provides its own background pool and paired blur")
    if not 0 <= args.background_probability <= 1:
        parser.error("background-probability must be in [0,1]")
    try:
        RandomBlurDataset([], args.blur_probability, args.blur_sigma, args.blur_kernel_size)
    except ValueError as error:
        parser.error(str(error))
    return args


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
    amp_dtype: torch.dtype = torch.float16,
    decoder: str = 'topk',
    decode_radius: int = 3,
    heatmap_margin: int = 0,
    normalize_body_size: bool = False,
    criterion_uses_input_hw: bool = False,
) -> tuple[float, float, float]:
    """Return ``(loss, mean_corner_error_px, pck@5px)`` over valid corners."""
    model.eval()
    loss_sum, sample_count = 0.0, 0
    err_sum, hit_count, valid_count = 0.0, 0, 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["heatmap"].to(device, non_blocking=True)
        valid = batch["valid"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
            logits = model(images)
            arguments = (logits, targets, valid, batch["keypoints_2d"].to(device))
            loss = criterion(*arguments, tuple(images.shape[-2:])) if criterion_uses_input_hw else criterion(*arguments)
        loss_sum += float(loss) * images.size(0)
        sample_count += images.size(0)

        points = heatmap_to_input_points(
            decode_heatmaps(logits.float(), decoder, decode_radius), tuple(logits.shape[-2:]),
            tuple(images.shape[-2:]), margin=heatmap_margin
        )
        delta = points - batch["keypoints_2d"].to(device)
        if normalize_body_size:
            crop_wh = batch['crop_xywh'][:, 2:].to(device)
            image_wh = delta.new_tensor([images.shape[-1], images.shape[-2]])
            delta = delta * (crop_wh / image_wh)[:, None, :]
            delta = delta * (256 / crop_wh.amax(1))[:, None, None]
        distance = delta.norm(dim=-1)
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
    if any((args.out_dir/name).exists() for name in ['best.pth', 'last.pth', 'metrics.csv']):
        raise FileExistsError(f'Training artifacts already exist in {args.out_dir}; use a new output directory')
    args.input_contract_version = 2 if args.crop_mode == 'rect' else 1
    args.error_units = 'body_long_side_256' if args.crop_mode == 'rect' else 'input_pixels'
    cv2.setNumThreads(1)
    torch.set_num_threads(args.torch_threads)
    seed_all(args.seed)
    device = resolve_device(args.device)
    if args.paired_background or args.deterministic_cudnn:
        # Keep the small coefficient comparison free of cuDNN autotuning differences.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
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
        crop_border=args.crop_border,
        heatmap_kernel=args.heatmap_kernel,
        heatmap_sigma=args.heatmap_sigma,
        heatmap_margin=args.heatmap_margin,
        rect_sizes=args.rect_sizes,
        require_all_corners=args.require_all_corners,
    )
    train_dataset_kwargs = dict(dataset_kwargs)
    if args.train_min_visib_fract is not None:
        if not 0 <= args.train_min_visib_fract <= 1:
            raise ValueError('train-min-visib-fract must be in [0,1]')
        if not args.val_scenes and not args.overfit:
            raise ValueError('Training-only visibility filtering requires grouped val scenes')
        train_dataset_kwargs['min_visib_fract'] = args.train_min_visib_fract
    if args.overfit:
        if args.val_scenes:
            raise SystemExit("--overfit cannot be combined with --val-scenes")
        train_torch = HeatmapDataset(
            BOPInstanceDataset(args.dataset_root, model_info, scenes=available, **train_dataset_kwargs)
        )
        val_torch = train_torch
        split_description = "OVERFIT: identical training/evaluation instances, no held-out validation"
    elif args.val_scenes:
        val_scenes = list(args.val_scenes)
        missing = [s for s in val_scenes if s not in available]
        if missing:
            raise SystemExit(f"val scenes {missing} not found under {args.dataset_root}; available: {available}")
        train_scenes = [s for s in available if s not in set(val_scenes)]
        args.train_scenes = train_scenes
        if args.crop_mode == 'rect':
            args.scene_gt_sha256 = {s: hashlib.sha256((args.dataset_root/s/'scene_gt.json').read_bytes()).hexdigest()
                                   for s in available}
        if not train_scenes:
            raise SystemExit("no training scenes left after holding out validation scenes")
        train_torch = HeatmapDataset(
            BOPInstanceDataset(args.dataset_root, model_info, scenes=train_scenes, **train_dataset_kwargs)
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
    jitter_base = None
    if args.crop_jitter_shift or args.crop_jitter_padding is not None or args.background_assets is not None or args.online_background_config is not None:
        augmented_class = RandomBackgroundBOPDataset if args.background_assets is not None else BOPInstanceDataset
        background_kwargs = dict(background_assets=args.background_assets,
                                 background_probability=args.background_probability) if args.background_assets is not None else {}
        if args.online_background_config is not None:
            augmented_class = OnlinePoseBackgroundDataset
            background_kwargs = dict(online_background_config=args.online_background_config)
        jitter_base = augmented_class(
            args.dataset_root, model_info, scenes=train_scenes, **train_dataset_kwargs,
            crop_jitter_shift=args.crop_jitter_shift,
            crop_jitter_padding=args.crop_jitter_padding, crop_jitter_seed=args.seed, **background_kwargs)
        wrapper = PairedHeatmapDataset if args.paired_background else HeatmapDataset
        train_torch = wrapper(jitter_base)
    if args.limit:
        if args.online_background_config is not None:
            # Include legacy and new sources in the bounded pipeline check.
            train_torch = Subset(train_torch, np.linspace(0, len(train_torch)-1, min(args.limit, len(train_torch)), dtype=int).tolist())
            val_torch = Subset(val_torch, np.linspace(0, len(val_torch)-1, min(args.limit, len(val_torch)), dtype=int).tolist())
        else:
            train_torch = Subset(train_torch, range(min(args.limit, len(train_torch))))
            val_torch = Subset(val_torch, range(min(args.limit, len(val_torch))))

    if args.cache_data:
        train_torch = [train_torch[i] for i in tqdm(range(len(train_torch)), desc="cache train")]
        val_torch = train_torch if args.overfit else [
            val_torch[i] for i in tqdm(range(len(val_torch)), desc="cache validation")
        ]

    if args.blur_probability > 0:
        train_torch = RandomBlurDataset(
            train_torch, args.blur_probability, args.blur_sigma, args.blur_kernel_size
        )
        print(f"[augmentation] training only: blur probability={args.blur_probability} "
              f"sigma={args.blur_sigma} kernel={args.blur_kernel_size}; labels unchanged")

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    shape_sampler = None
    photometric_base = None
    sample_weights = None
    if args.view_balance:
        labels = dataset_view_bins(train_torch)
        sample_weights, counts = view_weights(labels,args.view_weight_cap)
        (args.out_dir/'view_sampling.json').write_text(json.dumps(dict(
            bins=dict(zip(FACE_NAMES,counts.tolist())),weight_cap=args.view_weight_cap,
            scope='Training only; replacement within shape bucket; same epoch length and update count'),indent=2))
    if args.photometric_augmentation:
        photometric_base = PhotometricDataset(train_torch,args.seed,args.photometric_profile)
        train_torch = photometric_base
    if args.rect_sizes is not None:
        loader_kwargs.pop('batch_size')
        shape_sampler = ShapeBucketBatchSampler(train_torch, args.batch_size, True, args.seed, sample_weights)
        train_loader = DataLoader(train_torch, batch_sampler=shape_sampler, **loader_kwargs)
        val_loader = DataLoader(val_torch, batch_sampler=ShapeBucketBatchSampler(
            val_torch, args.batch_size, False, args.seed), **loader_kwargs)
    else:
        train_loader = DataLoader(train_torch, shuffle=True, **loader_kwargs)
        val_loader = DataLoader(val_torch, shuffle=False, **loader_kwargs)

    print(
        f"[data] {split_description} | train samples={len(train_torch)} | "
        f"val samples={len(val_torch)}"
    )
    print(f"[data] input buckets={args.rect_sizes or [(in_w, in_h)]} "
          f"heatmap stride=4 output margin={args.heatmap_margin}; error units={args.error_units}")
    print(
        f"[data] crop={args.crop_mode} padding={args.crop_padding} pose_scale={args.pose_scale:.12f} "
        f"kernel={args.heatmap_kernel} sigma={args.heatmap_sigma}"
    )
    print(f"[device] {device}  amp={args.amp and device.type == 'cuda'}")

    model = ResNetFPN(
        backbone=args.backbone,
        fpn_dim=args.fpn_dim,
        num_corners=len(CORNER_SIGNS),
        pretrained=not args.no_pretrained and args.init_checkpoint is None,
        freeze_bn=args.freeze_bn,
        heatmap_margin=args.heatmap_margin,
    ).to(device)
    if args.init_checkpoint is not None:
        checkpoint = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=True)
        args.initial_checkpoint_sha256 = hashlib.sha256(args.init_checkpoint.read_bytes()).hexdigest()
        print(f"[init] model weights from {args.init_checkpoint}; fresh optimizer and schedule")
        del checkpoint
    backbone_params = list(model.backbone.parameters())
    head_params = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    optimizer = torch.optim.AdamW(
        [
            {"params": head_params, "lr": args.lr},
            {"params": backbone_params, "lr": args.lr * args.backbone_lr_scale},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs) if args.cosine_lr else None
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[model] {args.backbone}+FPN parameters={n_total / 1e6:.2f}M")

    use_amp = args.amp and device.type == "cuda"
    amp_dtype = getattr(torch, args.amp_dtype)
    if use_amp and amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("bfloat16 AMP is not supported by this GPU")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    if args.loss == "composite":

        def criterion(prediction, targets, valid, keypoints, input_hw):
            return focal_loss(
                prediction, targets, valid, alpha=args.focal_alpha, beta=args.focal_beta
            ) + args.coordinate_loss_weight * coordinate_loss(
                prediction, keypoints, valid, input_hw, margin=args.heatmap_margin
            )

    elif args.loss == "focal":

        def criterion(prediction, targets, valid, keypoints, input_hw):
            return focal_loss(
                prediction, targets, valid, alpha=args.focal_alpha, beta=args.focal_beta
            )

    elif args.loss == 'spatial_kl':

        def criterion(prediction, targets, valid, keypoints, input_hw):
            return spatial_kl_loss(prediction, targets, valid)

    else:

        def criterion(prediction, targets, valid, keypoints, input_hw):
            return heatmap_loss(prediction, targets, valid)

    print(f"[loss] {args.loss}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(train_samples=len(train_torch), eval_samples=len(val_torch), split_description=split_description)
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=2))
    evaluation_options = dict(decoder=args.decoder, decode_radius=args.decode_radius,
                              heatmap_margin=args.heatmap_margin,
                              normalize_body_size=args.crop_mode == 'rect', criterion_uses_input_hw=True)
    baseline = evaluate(model, val_loader, device, criterion, heatmap_size, (in_h, in_w), use_amp, amp_dtype, **evaluation_options)
    (args.out_dir / "baseline.json").write_text(json.dumps(dict(zip(("loss", "err_px", "pck5"), baseline)), indent=2))
    print(f"[baseline] loss={baseline[0]:.4f} err_px={baseline[1]:.2f} pck5={baseline[2]:.4f}")
    best_err = float("inf")
    with open(args.out_dir / "metrics.csv", "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_err_px", "val_pck5", "lr", "seconds"])
        for epoch in range(args.epochs):
            if shape_sampler is not None:
                shape_sampler.set_epoch(epoch)
            if jitter_base is not None:
                jitter_base.set_epoch(epoch)
            if photometric_base is not None:
                photometric_base.set_epoch(epoch)
            if args.view_balance:
                sampled=[i for batch in shape_sampler for i in batch]
                with (args.out_dir/'view_sampling_epochs.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(epoch=epoch+1,counts=dict(zip(FACE_NAMES,
                        np.bincount(labels[sampled],minlength=6).tolist())),samples=len(sampled)))+'\n')
            freeze = epoch < args.freeze_epochs
            model.freeze_backbone(freeze)
            model.train()
            start = time.time()
            running, steps = 0.0, 0
            supervised_sum, consistency_sum, paired_count = 0., 0., 0
            progress = tqdm(
                train_loader, desc=f"epoch {epoch + 1}/{args.epochs}", leave=False, disable=not sys.stderr.isatty()
            )
            for batch in progress:
                images = batch["image"].to(device, non_blocking=True)
                targets = batch["heatmap"].to(device, non_blocking=True)
                valid = batch["valid"].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
                    if args.paired_background:
                        other = batch['image_pair'].to(device, non_blocking=True)
                        first_logits, second_logits = model(torch.cat([images, other])).chunk(2)
                        supervised = .5 * (spatial_kl_loss(first_logits, targets, valid) +
                                           spatial_kl_loss(second_logits, targets, valid))
                        consistency = spatial_js_loss(first_logits, second_logits, valid)
                        loss = supervised + args.consistency_weight * consistency
                        supervised_sum += supervised.detach().item() * images.size(0)
                        consistency_sum += consistency.detach().item() * images.size(0)
                        paired_count += images.size(0)
                    else:
                        logits = model(images)
                        loss = criterion(
                            logits, targets, valid, batch["keypoints_2d"].to(device, non_blocking=True),
                            tuple(images.shape[-2:])
                        )
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite training loss at epoch {epoch + 1}, step {steps}")
                previous_scale = scaler.get_scale()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                if args.strict_updates and scaler.get_scale() < previous_scale:
                    raise RuntimeError("AMP skipped an optimizer update; paired update budget no longer holds")
                running += loss.item()
                steps += 1
                progress.set_postfix(loss=f"{running / steps:.4f}")
            train_loss = running / max(steps, 1)
            if args.paired_background:
                with (args.out_dir/'paired_terms.jsonl').open('a') as pair_log:
                    pair_log.write(json.dumps(dict(epoch=epoch+1, instances=paired_count,
                        rgb_views=paired_count*2, optimizer_steps=steps,
                        supervised=supervised_sum/max(paired_count, 1),
                        consistency_js=consistency_sum/max(paired_count, 1),
                        consistency_weight=args.consistency_weight))+'\n')

            val_loss, val_err, val_pck = evaluate(
                model, val_loader, device, criterion, heatmap_size, (in_h, in_w), use_amp, amp_dtype,
                **evaluation_options
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
                "heatmap_margin": args.heatmap_margin,
                "rect_sizes": args.rect_sizes,
                "error_units": ('body_long_side_256' if args.crop_mode == 'rect' else 'input_pixels'),
            }
            if val_err < best_err:
                best_err = val_err
                save_checkpoint(args.out_dir / "best.pth", model, optimizer, epoch, args, metrics)
            save_checkpoint(args.out_dir / "last.pth", model, optimizer, epoch, args, metrics)
            if scheduler is not None:
                scheduler.step()
            print(
                f"[epoch {epoch + 1:03d}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"val_err_px={val_err:.2f} val_pck@5={val_pck:.3f} "
                f"backbone_frozen={freeze} time={elapsed:.1f}s"
            )
    print(f"[done] best val_err_px={best_err:.2f}; checkpoints in {args.out_dir}")


if __name__ == "__main__":
    main()
