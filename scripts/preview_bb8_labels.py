#!/usr/bin/env python3
"""Preview ``bop_bb8`` label samples: wireframe overlay + 8-channel heatmaps.

Example:
    .venv-hccepose/bin/python scripts/preview_bb8_labels.py \
        --dataset-root dataset/action4-v4-random-validation-100/train_pbr \
        --scene-id 000000 --image-id 0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from bop_bb8 import BOPInstanceDataset, BOPModel  # noqa: E402


def heatmap_montage(heatmaps: np.ndarray, scale: int = 4) -> np.ndarray:
    panels = []
    for channel in heatmaps:
        img = np.clip(channel * 255.0, 0, 255).astype(np.uint8)
        img = cv2.applyColorMap(img, cv2.COLORMAP_TURBO)
        if scale != 1:
            img = cv2.resize(img, (img.shape[1] * scale, img.shape[0] * scale),
                             interpolation=cv2.INTER_NEAREST)
        panels.append(img)
    top = np.hstack(panels[:4])
    bottom = np.hstack(panels[4:])
    return np.vstack([top, bottom])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    v4_root = PROJECT_DIR / "dataset" / "action4-v4-random-validation-100"
    parser.add_argument("--dataset-root", type=Path, default=v4_root / "train_pbr")
    parser.add_argument("--models-info", type=Path,
                        default=v4_root / "models" / "models_info.json")
    parser.add_argument("--obj-id", type=int, default=1)
    parser.add_argument("--model-scale", type=float, default=1.0)
    parser.add_argument("--scene-id", default="000000")
    parser.add_argument("--image-id", type=int, default=0)
    parser.add_argument("--instance-idx", type=int, default=0)
    parser.add_argument("--input-size", type=int, nargs=2, default=[256, 256], help="W H")
    parser.add_argument("--heatmap-size", type=int, nargs=2, default=[64, 64], help="H W")
    parser.add_argument("--crop-padding", type=float, default=1.3)
    parser.add_argument("--out-dir", type=Path, default=PROJECT_DIR / "outputs" / "bb8_preview")
    args = parser.parse_args()

    model = BOPModel.from_models_info(args.models_info, args.obj_id, args.model_scale)
    dataset = BOPInstanceDataset(
        args.dataset_root, model, scenes=[args.scene_id],
        input_size=tuple(args.input_size), crop_mode="bbox", crop_padding=args.crop_padding,
        heatmap_size=tuple(args.heatmap_size),
        heatmap_kernel="gaussian", heatmap_sigma=2.0,
    )
    sample = dataset.load_instance(args.scene_id, args.image_id, args.instance_idx)

    with open(args.dataset_root / args.scene_id / "scene_gt_info.json") as f:
        import json
        info = json.load(f)
    bbox_obj = tuple(info[str(args.image_id)][args.instance_idx]["bbox_obj"])

    from bop_bb8.visualize import draw_sample
    stem = f"{args.scene_id}_{args.image_id:06d}_{args.instance_idx:02d}"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = args.out_dir / f"{stem}_overlay.png"
    heatmap_path = args.out_dir / f"{stem}_heatmaps.png"
    draw_sample(sample, bbox_obj=bbox_obj, out_path=overlay_path)
    montage = heatmap_montage(sample.heatmaps)
    cv2.putText(montage, f"{stem} s={args.model_scale} valid={sample.valid.sum()}/8", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(heatmap_path), montage)

    print(f"[sample] {stem}  t_z={sample.t_m2c[2]:.1f}mm  visib={sample.visib_fract:.2f}")
    print(f"[sample] corners_3d[0]={sample.corners_3d[0]} mm, model scale={args.model_scale}")
    print(f"[sample] keypoints_2d={np.round(sample.keypoints_2d, 1).tolist()}")
    print(f"[sample] valid={sample.valid.tolist()}")
    print(f"[sample] heatmaps {sample.heatmaps.shape} range=[{sample.heatmaps.min():.3f}, {sample.heatmaps.max():.3f}]")
    print(f"[out] {overlay_path}")
    print(f"[out] {heatmap_path}")


if __name__ == "__main__":
    main()
