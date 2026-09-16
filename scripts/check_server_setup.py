#!/usr/bin/env python3
"""Validate the server CUDA runtime and optionally load one BOP sample."""
from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from bop_bb8 import BOPInstanceDataset, BOPModel  # noqa: E402
from bop_bb8.net import ResNetFPN  # noqa: E402
from bop_bb8.torch_data import HeatmapDataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, help="BOP train_pbr directory")
    parser.add_argument("--models-info", type=Path, help="BOP models_info.json")
    parser.add_argument("--obj-id", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"python={sys.version.split()[0]} platform={platform.platform()}")
    print(f"numpy={np.__version__} opencv={cv2.__version__}")
    print(f"torch={torch.__version__} torchvision={torchvision.__version__}")
    print(f"torch_cuda_runtime={torch.version.cuda} cuda_available={torch.cuda.is_available()}")

    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise SystemExit("CUDA was requested but torch.cuda.is_available() is False")
        properties = torch.cuda.get_device_properties(0)
        print(f"gpu={torch.cuda.get_device_name(0)} vram_gib={properties.total_memory / 2**30:.2f}")

    device = torch.device(args.device)
    model = ResNetFPN(pretrained=False).to(device).eval()
    with torch.no_grad():
        output = model(torch.zeros(1, 3, 256, 256, device=device))
    if tuple(output.shape) != (1, 8, 64, 64) or not torch.isfinite(output).all():
        raise SystemExit(f"unexpected network output: shape={tuple(output.shape)}")
    print("network_smoke=OK output_shape=(1, 8, 64, 64)")

    if bool(args.dataset_root) != bool(args.models_info):
        raise SystemExit("--dataset-root and --models-info must be provided together")
    if args.dataset_root:
        if not args.dataset_root.is_dir():
            raise SystemExit(f"dataset root not found: {args.dataset_root}")
        if not args.models_info.is_file():
            raise SystemExit(f"models_info not found: {args.models_info}")
        bop_model = BOPModel.from_models_info(args.models_info, args.obj_id)
        dataset = BOPInstanceDataset(
            args.dataset_root,
            bop_model,
            input_size=(256, 256),
            heatmap_size=(64, 64),
            crop_mode="bbox",
            crop_padding=1.3,
            heatmap_kernel="gaussian",
            heatmap_sigma=2.0,
        )
        if len(dataset) == 0:
            raise SystemExit("dataset contains no matching object instances")
        sample = HeatmapDataset(dataset)[0]
        expected = {"image": (3, 256, 256), "heatmap": (8, 64, 64), "valid": (8,)}
        for key, shape in expected.items():
            if tuple(sample[key].shape) != shape:
                raise SystemExit(f"{key} has shape {tuple(sample[key].shape)}, expected {shape}")
        if not sample["valid"].any():
            raise SystemExit("first sample has no valid projected corners")
        first_scene, first_image, _ = dataset.instances[0]
        print(
            f"dataset_smoke=OK samples={len(dataset)} first_scene={first_scene} "
            f"first_image={first_image}"
        )


if __name__ == "__main__":
    main()
