#!/usr/bin/env python3
"""Evaluate a saved BB8 heatmap checkpoint and render prediction overlays."""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from bop_bb8 import BOPInstanceDataset, BOPModel, CORNER_SIGNS  # noqa: E402
from bop_bb8.net import ResNetFPN, decode_topk, heatmap_to_input_points  # noqa: E402
from bop_bb8.torch_data import HeatmapDataset  # noqa: E402
from bop_bb8.visualize import EDGES  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--models-info", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=460731)
    parser.add_argument("--random-per-split", type=int, default=4)
    parser.add_argument("--worst", type=int, default=4)
    parser.add_argument("--all-overlays", action="store_true")
    return parser.parse_args()


def metric_summary(errors: np.ndarray) -> dict[str, float | int]:
    finite = errors[np.isfinite(errors)]
    if not finite.size:
        return {"corner_count": 0}
    return {
        "corner_count": int(finite.size),
        "mean_err_px": float(finite.mean()),
        "median_err_px": float(np.median(finite)),
        "p90_err_px": float(np.percentile(finite, 90)),
        "p95_err_px": float(np.percentile(finite, 95)),
        "max_err_px": float(finite.max()),
        "pck2": float(np.mean(finite <= 2.0)),
        "pck5": float(np.mean(finite <= 5.0)),
        "pck10": float(np.mean(finite <= 10.0)),
    }


def draw_overlay(
    image_rgb: np.ndarray,
    truth: np.ndarray,
    prediction: np.ndarray,
    valid: np.ndarray,
    title: str,
) -> np.ndarray:
    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    truth_color = (40, 220, 40)
    prediction_color = (220, 40, 220)
    for a, b in EDGES:
        if valid[a] and valid[b]:
            gt_a = tuple(np.rint(truth[a]).astype(int))
            gt_b = tuple(np.rint(truth[b]).astype(int))
            pr_a = tuple(np.rint(prediction[a]).astype(int))
            pr_b = tuple(np.rint(prediction[b]).astype(int))
            cv2.line(image, gt_a, gt_b, truth_color, 2, cv2.LINE_AA)
            cv2.line(image, pr_a, pr_b, prediction_color, 1, cv2.LINE_AA)
    for index in range(len(CORNER_SIGNS)):
        if not valid[index]:
            continue
        gt = tuple(np.rint(truth[index]).astype(int))
        pred = tuple(np.rint(prediction[index]).astype(int))
        cv2.circle(image, gt, 4, truth_color, -1, cv2.LINE_AA)
        cv2.drawMarker(
            image, pred, prediction_color, cv2.MARKER_CROSS, 10, 2, cv2.LINE_AA
        )
        cv2.putText(
            image,
            str(index),
            (pred[0] + 4, pred[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            prediction_color,
            1,
            cv2.LINE_AA,
        )
    image = cv2.copyMakeBorder(image, 44, 0, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))
    cv2.putText(
        image, title, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255),
        1, cv2.LINE_AA,
    )
    cv2.putText(
        image, "GT", (5, 35), cv2.FONT_HERSHEY_SIMPLEX,
        0.43, truth_color, 1, cv2.LINE_AA,
    )
    cv2.putText(
        image, "PRED", (40, 35), cv2.FONT_HERSHEY_SIMPLEX,
        0.43, prediction_color, 1, cv2.LINE_AA,
    )
    return image


def make_contact_sheet(panels: list[np.ndarray], columns: int = 4) -> np.ndarray:
    if not panels:
        raise ValueError("no panels")
    height, width = panels[0].shape[:2]
    rows = (len(panels) + columns - 1) // columns
    blank = np.full_like(panels[0], 32)
    padded = panels + [blank] * (rows * columns - len(panels))
    return np.vstack(
        [np.hstack(padded[row * columns:(row + 1) * columns]) for row in range(rows)]
    )


def main() -> None:
    args = parse_args()
    torch.set_num_threads(4)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved = checkpoint.get("args", {})
    input_size = tuple(int(value) for value in saved.get("input_size", [256, 256]))
    input_w, input_h = input_size
    heatmap_size = tuple(int(value) for value in checkpoint["metrics"]["heatmap_size"])
    if heatmap_size != (input_h // 4, input_w // 4):
        raise ValueError(f"unexpected heatmap size {heatmap_size} for input {input_size}")

    available = sorted(
        path.name for path in args.dataset_root.iterdir()
        if path.is_dir() and path.name.isdigit() and (path / "scene_gt.json").exists()
    )
    val_scenes = set(saved.get("val_scenes") or [])
    missing = val_scenes.difference(available)
    if missing:
        raise ValueError(f"checkpoint validation scenes are missing: {sorted(missing)}")

    bop_model = BOPModel.from_models_info(
        args.models_info,
        int(saved.get("obj_id", 1)),
        float(saved.get("model_scale", 1.0)),
    )
    base_dataset = BOPInstanceDataset(
        args.dataset_root,
        bop_model,
        scenes=available,
        input_size=input_size,
        heatmap_size=heatmap_size,
        min_visib_fract=float(saved.get("min_visib_fract", 0.0)),
        pose_scale=float(saved.get("pose_scale", 1.0)),
        crop_mode=saved.get("crop_mode", "bbox"),
        crop_padding=float(saved.get("crop_padding", 1.3)),
        heatmap_kernel=saved.get("heatmap_kernel", "gaussian"),
        heatmap_sigma=float(saved.get("heatmap_sigma", 2.0)),
    )
    loader = DataLoader(
        HeatmapDataset(base_dataset),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model = ResNetFPN(
        backbone=saved.get("backbone", "resnet18"),
        fpn_dim=int(saved.get("fpn_dim", 128)),
        num_corners=len(CORNER_SIGNS),
        pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    sample_count = len(base_dataset)
    predictions = np.full((sample_count, 8, 2), np.nan, dtype=np.float32)
    truths = np.full((sample_count, 8, 2), np.nan, dtype=np.float32)
    valid_all = np.zeros((sample_count, 8), dtype=bool)
    max_probabilities = np.full((sample_count, 8), np.nan, dtype=np.float32)
    with torch.inference_mode():
        for batch in tqdm(loader, desc="inference"):
            indices = batch["index"].numpy()
            images = batch["image"].to(device, non_blocking=True)
            logits = model(images)
            probabilities = torch.sigmoid(logits.float())
            points = heatmap_to_input_points(
                decode_topk(logits, k=9, from_logits=True), heatmap_size, (input_h, input_w)
            )
            predictions[indices] = points.cpu().numpy()
            truths[indices] = batch["keypoints_2d"].numpy()
            valid_all[indices] = batch["valid"].numpy()
            max_probabilities[indices] = probabilities.amax(dim=(-2, -1)).cpu().numpy()

    corner_errors = np.linalg.norm(predictions - truths, axis=-1)
    corner_errors[~valid_all] = np.nan
    sample_mean = np.nanmean(corner_errors, axis=1)
    sample_max = np.nanmax(corner_errors, axis=1)
    sample_pck5 = np.nanmean(np.where(valid_all, corner_errors <= 5.0, np.nan), axis=1)
    scene_ids = np.asarray([item[0] for item in base_dataset.instances])
    split_names = np.asarray(["val" if scene in val_scenes else "train" for scene in scene_ids])
    if not val_scenes and not saved.get("overfit", False):
        order = torch.randperm(sample_count, generator=torch.Generator().manual_seed(int(saved.get("seed", 0)))).numpy()
        n_val = max(1, int(round(sample_count * float(saved.get("val_fraction", 0.2)))))
        split_names[order[:n_val]] = "val"
    if saved.get("limit", 0):
        raise ValueError("Evaluation of limited training splits requires the explicit saved instance list")

    projected_spans = np.nanmax(truths, axis=1) - np.nanmin(truths, axis=1)
    object_bbox_spans = np.empty((sample_count, 2), dtype=np.float64)
    crop_scales = np.empty((sample_count, 2), dtype=np.float64)
    source_size = (int(base_dataset.crop[2]), int(base_dataset.crop[3]))
    for index, (scene_id, image_id, instance_idx) in enumerate(base_dataset.instances):
        _, _, scene_info = base_dataset._load_scene(scene_id)
        bbox = scene_info[str(image_id)][instance_idx]["bbox_obj"]
        crop = base_dataset._bbox_crop(bbox, source_size, base_dataset.crop_padding)
        if base_dataset.crop_mode == "full":
            crop = base_dataset.crop
        crop_scales[index] = (crop[2] / input_w, crop[3] / input_h)
        object_bbox_spans[index] = (
            float(bbox[2]) * input_w / crop[2],
            float(bbox[3]) * input_h / crop[3],
        )
    span_ratios = projected_spans / np.maximum(object_bbox_spans, 1e-6)
    median_span_ratio = np.median(span_ratios, axis=0)
    severe_label_geometry_mismatch = bool(
        (median_span_ratio < 0.8).any() or (median_span_ratio > 1.5).any()
    )

    summary = {
        "evaluation_protocol": "training-set overfit; no held-out validation" if saved.get("overfit") else "train/validation",
        "backbone": saved.get("backbone", "resnet18"),
        "input_size": input_size,
        "uses_ground_truth_bbox": base_dataset.crop_mode == "bbox",
        "excluded_corners": int((~valid_all).sum()),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch_one_based": int(checkpoint["epoch"]) + 1,
        "checkpoint_metrics": checkpoint.get("metrics", {}),
        "dataset_root": str(args.dataset_root.resolve()),
        "sample_count": sample_count,
        "train_scenes": [scene for scene in available if scene not in val_scenes],
        "val_scenes": sorted(val_scenes),
        "all": metric_summary(corner_errors),
        "original_image_pixels": metric_summary(np.where(valid_all, np.linalg.norm((predictions - truths) * crop_scales[:, None, :], axis=-1), np.nan)),
        "per_scene": {scene: metric_summary(corner_errors[scene_ids == scene]) for scene in available},
        "train": metric_summary(corner_errors[split_names == "train"]),
        "val": metric_summary(corner_errors[split_names == "val"]),
        "label_projection_vs_object_bbox": {
            "median_width_ratio": float(median_span_ratio[0]),
            "median_height_ratio": float(median_span_ratio[1]),
            "median_area_ratio": float(np.median(np.prod(span_ratios, axis=1))),
            "severe_mismatch": severe_label_geometry_mismatch,
            "meaning": "projected 3D-corner bbox span divided by bbox_obj span in the network crop",
        },
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_dir / "predictions.npz", predictions=predictions, truths=truths, valid=valid_all, crop_scales=crop_scales)
    rows = []
    for index, (scene_id, image_id, instance_idx) in enumerate(base_dataset.instances):
        rows.append(
            {
                "dataset_index": index,
                "split": split_names[index],
                "scene_id": scene_id,
                "image_id": image_id,
                "instance_idx": instance_idx,
                "mean_err_px": float(sample_mean[index]),
                "max_err_px": float(sample_max[index]),
                "pck5": float(sample_pck5[index]),
                "mean_peak_probability": float(np.nanmean(max_probabilities[index])),
            }
        )
    with open(args.out_dir / "per_sample_metrics.csv", "w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(args.out_dir / "summary.json", "w") as output:
        json.dump(summary, output, indent=2)

    rng = np.random.default_rng(args.seed)
    train_indices = np.flatnonzero(split_names == "train")
    val_indices = np.flatnonzero(split_names == "val")
    random_train = rng.choice(
        train_indices, size=min(args.random_per_split, len(train_indices)), replace=False
    ).tolist()
    random_val = rng.choice(
        val_indices, size=min(args.random_per_split, len(val_indices)), replace=False
    ).tolist()
    worst_val = val_indices[np.argsort(sample_mean[val_indices])[-args.worst:][::-1]].tolist()
    worst_train = train_indices[np.argsort(sample_mean[train_indices])[-args.worst:][::-1]].tolist()
    selections = [
        ("random_train", index) for index in random_train
    ] + [
        ("random_val", index) for index in random_val
    ] + [
        ("worst_val", index) for index in worst_val
    ] + [
        ("worst_train", index) for index in worst_train
    ]

    overlay_dir = args.out_dir / "overlays"
    overlay_dir.mkdir(exist_ok=True)
    panels = []
    selected_rows = []
    for group, index in selections:
        sample = base_dataset[index]
        row = rows[index]
        title = (
            f"{group} #{index} {row['mean_err_px']:.2f}px"
        )
        panel = draw_overlay(
            sample.image, truths[index], predictions[index], valid_all[index], title
        )
        output_path = overlay_dir / f"{group}_{row['scene_id']}_{row['image_id']:06d}_{row['instance_idx']}.png"
        cv2.imwrite(str(output_path), panel)
        panels.append(panel)
        selected_rows.append({**row, "group": group, "overlay": str(output_path)})
    contact_sheet = make_contact_sheet(panels)
    cv2.imwrite(str(args.out_dir / "prediction_contact_sheet.png"), contact_sheet)
    with open(args.out_dir / "selected_samples.json", "w") as output:
        json.dump(selected_rows, output, indent=2)

    if args.all_overlays:
        all_dir = args.out_dir / "all_overlays"
        all_dir.mkdir(exist_ok=True)
        for index, row in enumerate(rows):
            sample = base_dataset[index]
            title = f"{row['scene_id']}/{row['image_id']:06d}/{row['instance_idx']} {row['mean_err_px']:.2f}px"
            panel = draw_overlay(sample.image, truths[index], predictions[index], valid_all[index], title)
            cv2.imwrite(str(all_dir / f"{index:04d}.jpg"), panel)

    report_lines = [
        f"# Action 4 {summary['backbone']} checkpoint evaluation",
        "",
        f"- Protocol: {summary['evaluation_protocol']}",
        f"- Errors are measured in {input_w}x{input_h} crop pixels; crops use ground-truth boxes.",
        f"- Checkpoint epoch: {summary['checkpoint_epoch_one_based']}",
        f"- Dataset samples: {sample_count}",
        f"- Train scenes: {', '.join(summary['train_scenes'])}",
        f"- Validation scenes: {', '.join(summary['val_scenes'])}",
        "",
        "| Split | Mean error | Median | P95 | PCK@2 | PCK@5 | PCK@10 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ("train", "val", "all"):
        metrics = summary[split]
        if not metrics["corner_count"]:
            continue
        report_lines.append(
            f"| {split} | {metrics['mean_err_px']:.3f}px | "
            f"{metrics['median_err_px']:.3f}px | {metrics['p95_err_px']:.3f}px | "
            f"{metrics['pck2']:.4f} | {metrics['pck5']:.4f} | {metrics['pck10']:.4f} |"
        )
    report_lines.extend(
        [
            "",
            "Green wireframe/circles are ground truth. Magenta wireframe/crosses are predictions.",
            "These metrics measure fit to the same synthetic rendering domain; they do not establish",
            "generalisation to the real head-camera video.",
            "",
        ]
    )
    geometry = summary["label_projection_vs_object_bbox"]
    report_lines.extend(
        [
            "## Label geometry sanity check",
            "",
            f"- Median projected-box / object-bbox width ratio: {geometry['median_width_ratio']:.4f}",
            f"- Median projected-box / object-bbox height ratio: {geometry['median_height_ratio']:.4f}",
            f"- Median projected-box / object-bbox area ratio: {geometry['median_area_ratio']:.6f}",
            f"- Severe mismatch: {geometry['severe_mismatch']}",
            "",
        ]
    )
    if severe_label_geometry_mismatch:
        report_lines.extend(
            [
                "The projected 3D box does not enclose the rendered object. Pixel-error and PCK",
                "against these labels can therefore look excellent while measuring the wrong target.",
                "Check model units/scale against cam_t_m2c before using this checkpoint.",
                "",
            ]
        )
    (args.out_dir / "report.md").write_text("\n".join(report_lines))

    print(json.dumps(summary, indent=2))
    print(f"[out] {args.out_dir / 'prediction_contact_sheet.png'}")
    print(f"[out] {args.out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
