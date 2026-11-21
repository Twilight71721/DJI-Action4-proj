#!/usr/bin/env python3
"""Audit whether a BOP dataset's poses/intrinsics agree with a candidate 3D model.

Project the model AABB corners with the per-frame cam_K / cam_R_m2c / cam_t_m2c
and compare the projected bounding box against scene_gt_info bbox_obj.  Also fit
a per-sample and a global diagnostic scale factor for the model (hypothesis:
the rendered object is the same shape at a different metric scale).

This is a data-audit tool, not the label generator.  The fitted scale is a
diagnostic result only and must not be treated as a confirmed model parameter.

Usage:
    .venv-hccepose/bin/python scripts/audit_bb8_projection.py \
        --dataset-root dataset/train_pbr \
        --model dataset/dji-action4/models/obj_000001.ply \
        --out outputs/bb8_audit
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy.optimize import minimize_scalar

PROJECT_DIR = Path(__file__).resolve().parent.parent

# Fixed ordering required by the label spec (model frame, mm).
CORNER_SIGNS = np.array(
    [
        [0, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
        [1, 0, 1],
    ],
    dtype=np.float64,
)

CORNER_COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 128, 0),
    (128, 0, 255),
]

EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


@dataclass
class Sample:
    scene_id: str
    image_id: int
    instance_idx: int
    obj_id: int
    R: np.ndarray
    t: np.ndarray
    K: np.ndarray
    bbox_obj: np.ndarray
    visib_fract: float

    @property
    def name(self) -> str:
        return f"{self.scene_id}_{self.image_id:06d}_{self.instance_idx:02d}"


def aabb_corners(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return lo[None, :] + CORNER_SIGNS * (hi - lo)[None, :]


def project(corners: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray, eps: float = 1e-6):
    P_c = corners @ R.T + t[None, :]
    z = P_c[:, 2]
    valid = z > eps
    uv = np.full((len(corners), 2), np.nan)
    uv[valid] = (K @ P_c[valid].T).T[:, :2] / z[valid, None]
    return uv, valid, P_c


def bbox_of(uv: np.ndarray, valid: np.ndarray | None = None):
    if valid is not None:
        uv = uv[valid]
    lo = np.nanmin(uv, axis=0)
    hi = np.nanmax(uv, axis=0)
    return np.array([lo[0], lo[1], hi[0] - lo[0], hi[1] - lo[1]])


def projected_bbox_in_front(corners: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray):
    """bbox of all corners with ``Z > 0`` (image bounds are not enforced)."""
    uv, _, points_cam = project(corners, K, R, t)
    if (points_cam[:, 2] <= 1e-6).any():
        return None
    return bbox_of(uv)


def iou_xywh(a: np.ndarray, b: np.ndarray) -> float:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    x0, y0 = max(ax0, bx0), max(ay0, by0)
    x1, y1 = min(ax0 + aw, bx0 + bw), min(ay0 + ah, by0 + bh)
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


def load_scene(dataset_root: Path, scene_id: str):
    with open(dataset_root / scene_id / "scene_gt.json") as f:
        gt = json.load(f)
    with open(dataset_root / scene_id / "scene_camera.json") as f:
        cam = json.load(f)
    with open(dataset_root / scene_id / "scene_gt_info.json") as f:
        info = json.load(f)
    return gt, cam, info


def check_associations(dataset_root: Path, scenes: list[str]) -> dict:
    report = {}
    for scene_id in scenes:
        gt, cam, info = load_scene(dataset_root, scene_id)
        n_gt, n_cam, n_info = len(gt), len(cam), len(info)
        ids_ok = set(gt) == set(cam) == set(info)
        counts, obj_ids = set(), set()
        mask_issues = []
        for image_id, insts in gt.items():
            counts.add(len(insts))
            obj_ids.update(int(i["obj_id"]) for i in insts)
            for iidx in range(len(insts)):
                mask = dataset_root / scene_id / "mask" / f"{int(image_id):06d}_{iidx:06d}.png"
                if not mask.exists():
                    mask_issues.append(str(mask))
        report[scene_id] = {
            "n_scene_gt": n_gt,
            "n_scene_camera": n_cam,
            "n_scene_gt_info": n_info,
            "ids_match": ids_ok,
            "instances_per_image": sorted(counts),
            "obj_ids": sorted(obj_ids),
            "missing_masks": mask_issues[:5],
        }
    return report


def is_valid_bbox(bbox: np.ndarray) -> bool:
    bbox = np.asarray(bbox)
    return bool((bbox >= 0).all() and bbox[2] > 0 and bbox[3] > 0)


def pick_samples(dataset_root: Path, scenes: list[str], per_scene: int) -> list[Sample]:
    samples = []
    for scene_id in scenes:
        gt, cam, info = load_scene(dataset_root, scene_id)
        candidates = sorted(
            (int(k) for k in gt if is_valid_bbox(info[str(k)][0]["bbox_obj"])),
            key=lambda i: -info[str(i)][0]["visib_fract"],
        )
        hi = candidates[: max(1, per_scene // 2)]
        lo = candidates[-max(1, per_scene // 2):][::-1]
        chosen = sorted(set(hi + lo))
        for image_id in chosen:
            inst = gt[str(image_id)][0]
            K = np.array(cam[str(image_id)]["cam_K"], dtype=np.float64).reshape(3, 3)
            R = np.array(inst["cam_R_m2c"], dtype=np.float64).reshape(3, 3)
            t = np.array(inst["cam_t_m2c"], dtype=np.float64)
            bbox = np.array(info[str(image_id)][0]["bbox_obj"], dtype=np.float64)
            samples.append(Sample(
                scene_id, image_id, 0, int(inst["obj_id"]), R, t, K, bbox,
                float(info[str(image_id)][0]["visib_fract"]),
            ))
    return samples


def scale_upper_bound(sample: Sample, corners: np.ndarray, margin: float = 0.999) -> float | None:
    """Largest uniform scale that keeps every corner in front of the camera."""
    rc = corners @ sample.R.T
    behind = rc[:, 2] < 0
    if not behind.any():
        return None
    return float(margin * sample.t[2] / (-rc[behind, 2]).max())


def bounded_search(fn, lo: float, hi: float, n_grid: int = 200) -> float:
    """Log-grid scan followed by Brent refinement (robust to plateau regions)."""
    grid = np.geomspace(lo, hi, n_grid)
    values = np.array([fn(float(s)) for s in grid])
    k = int(np.argmin(values))
    a = float(grid[max(0, k - 1)])
    b = float(grid[min(n_grid - 1, k + 1)])
    res = minimize_scalar(fn, bounds=(a, b), method="bounded",
                          options={"xatol": 1e-4})
    return float(res.x)


def fit_scale(sample: Sample, corners: np.ndarray, bounds=(0.05, 60.0)) -> float:
    valid_hi = scale_upper_bound(sample, corners)
    if valid_hi is not None:
        bounds = (bounds[0], min(bounds[1], valid_hi))
    if bounds[1] <= bounds[0] * 1.01:
        return float("nan")

    def objective(s: float) -> float:
        bbox = projected_bbox_in_front(s * corners, sample.K, sample.R, sample.t)
        if bbox is None:
            return 1e9
        err = bbox - sample.bbox_obj
        return float(np.sum(err ** 2))

    return bounded_search(objective, bounds[0], bounds[1])


def draw_overlay(image_path: Path, sample: Sample, corners: np.ndarray,
                 s_fit: float, out_path: Path) -> None:
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(image_path)

    uv1, valid1, _ = project(corners, sample.K, sample.R, sample.t)
    uv2, valid2, _ = project(s_fit * corners, sample.K, sample.R, sample.t)
    finite1 = np.isfinite(uv1).all(axis=1)
    finite2 = np.isfinite(uv2).all(axis=1)
    colors1 = [(200, 200, 200) for _ in EDGES]
    colors2 = [CORNER_COLORS[a] for a, _ in EDGES]
    for (a, b), c1, c2 in zip(EDGES, colors1, colors2):
        if finite1[a] and finite1[b]:
            cv2.line(img, tuple(np.round(uv1[a]).astype(int)), tuple(np.round(uv1[b]).astype(int)), c1, 1, cv2.LINE_AA)
        if finite2[a] and finite2[b]:
            cv2.line(img, tuple(np.round(uv2[a]).astype(int)), tuple(np.round(uv2[b]).astype(int)), c2, 2, cv2.LINE_AA)
    for i in range(8):
        if finite2[i]:
            p = tuple(np.round(uv2[i]).astype(int))
            cv2.circle(img, p, 3, CORNER_COLORS[i], -1, cv2.LINE_AA)
            cv2.putText(img, str(i), (p[0] + 4, p[1] - 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, CORNER_COLORS[i], 2, cv2.LINE_AA)
    x, y, w, h = sample.bbox_obj.astype(int)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    text = (f"{sample.name}  t_z={sample.t[2]:.0f}mm  visib={sample.visib_fract:.2f}  "
            f"s_fit={s_fit:.2f}  bbox_obj={tuple(sample.bbox_obj.astype(int))}")
    cv2.putText(img, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_DIR / "dataset" / "train_pbr")
    parser.add_argument("--model", type=Path, default=PROJECT_DIR / "dataset" / "dji-action4" / "models" / "obj_000001.ply")
    parser.add_argument("--models-info", type=Path, default=PROJECT_DIR / "dataset" / "dji-action4" / "models" / "models_info.json")
    parser.add_argument("--obj-id", type=int, default=1)
    parser.add_argument("--out", type=Path, default=PROJECT_DIR / "outputs" / "bb8_audit")
    parser.add_argument("--scenes", nargs="+", default=["000000", "000002", "000005", "000007", "000009"])
    parser.add_argument("--per-scene", type=int, default=6, help="number of samples per scene (high+low visibility)")
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    scenes = args.scenes
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.models_info) as f:
        models_info = json.load(f)
    mi = models_info[str(args.obj_id)]
    lo = np.array([mi["min_x"], mi["min_y"], mi["min_z"]], dtype=np.float64)
    hi = np.array([mi["max_x"], mi["max_y"], mi["max_z"]], dtype=np.float64)
    mesh = trimesh.load(args.model)
    v = np.asarray(mesh.vertices)
    lo_v, hi_v = v.min(axis=0), v.max(axis=0)
    corners = aabb_corners(lo, hi)

    print(f"[model] {args.model.name}")
    print(f"[model] models_info AABB size={hi - lo} mm, diameter={mi['diameter']:.2f} mm")
    print(f"[model] PLY vertex  AABB size={hi_v - lo_v} mm, center={(lo_v + hi_v) / 2}")

    assoc = check_associations(dataset_root, sorted(d.name for d in dataset_root.iterdir() if d.is_dir() and d.name.isdigit()))
    print("[assoc] per-scene checks:")
    for scene_id, rep in assoc.items():
        flag = "OK" if rep["ids_match"] and not rep["missing_masks"] else "FAIL"
        print(f"  {scene_id}: {flag} ids_match={rep['ids_match']} instances/image={rep['instances_per_image']} "
              f"obj_ids={rep['obj_ids']} missing_masks={len(rep['missing_masks'])}")

    samples = pick_samples(dataset_root, scenes, args.per_scene)

    rows = []
    for s in samples:
        bbox1 = projected_bbox_in_front(corners, s.K, s.R, s.t)
        if bbox1 is None:
            rows.append({"name": s.name, "error": "some corners behind camera"})
            continue
        s_fit = fit_scale(s, corners)
        bbox2 = projected_bbox_in_front(s_fit * corners, s.K, s.R, s.t)
        if bbox2 is None:
            rows.append({"name": s.name, "error": "fit scale puts corners behind camera"})
            continue
        mask_path = dataset_root / s.scene_id / "mask" / f"{s.image_id:06d}_{s.instance_idx:06d}.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        ys, xs = np.nonzero(mask > 0)
        mask_bbox = np.array([xs.min(), ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1])
        rows.append({
            "name": s.name,
            "t_z_mm": float(s.t[2]),
            "visib_fract": s.visib_fract,
            "bbox_obj": s.bbox_obj.tolist(),
            "mask_bbox": mask_bbox.tolist(),
            "bbox_proj_s1": bbox1.tolist(),
            "iou_s1": iou_xywh(bbox1, s.bbox_obj),
            "s_fit": float(s_fit),
            "bbox_proj_sfit": bbox2.tolist(),
            "iou_sfit": iou_xywh(bbox2, s.bbox_obj),
            "s_from_bbox_wh": float(np.sqrt(
                (s.bbox_obj[2] * s.bbox_obj[3]) / max(bbox1[2] * bbox1[3], 1e-9))),
        })
        draw_overlay(dataset_root / s.scene_id / "rgb" / f"{s.image_id:06d}.jpg",
                     s, corners, s_fit, out_dir / "overlays" / f"{s.name}.png")

    def global_objective(s: float) -> float:
        total = 0.0
        for sample in samples:
            bbox = projected_bbox_in_front(s * corners, sample.K, sample.R, sample.t)
            if bbox is None:
                return 1e9
            err = bbox - sample.bbox_obj
            total += float(np.sum(err ** 2))
        return total

    hi_bounds = [scale_upper_bound(sample, corners) for sample in samples]
    hi_global = min((h for h in hi_bounds if h is not None), default=60.0)
    s_global = bounded_search(global_objective, 0.05, min(60.0, hi_global))
    rmse_global = float(np.sqrt(global_objective(s_global) / (4 * len(samples))))

    print("\n[samples]")
    header = f"{'sample':<24} {'t_z':>8} {'visib':>6} {'bbox_obj':>18} {'proj s=1':>18} {'IoU1':>5} {'s_fit':>6} {'IoUfit':>6}"
    print(header)
    for r in rows:
        if "error" in r:
            print(f"{r['name']:<24} {r['error']}")
            continue
        print(f"{r['name']:<24} {r['t_z_mm']:>8.0f} {r['visib_fract']:>6.2f} "
              f"{str([int(x) for x in r['bbox_obj']]):>18} "
              f"{str([round(x, 1) for x in r['bbox_proj_s1']]):>18} "
              f"{r['iou_s1']:>5.2f} {r['s_fit']:>6.2f} {r['iou_sfit']:>6.2f}")
    print(f"\n[global fit] scale={s_global:.3f}, bbox RMSE={rmse_global:.2f} px")
    sfits = [r["s_fit"] for r in rows if "error" not in r]
    print(f"[per-sample s_fit] min={min(sfits):.2f} max={max(sfits):.2f} "
          f"median={np.median(sfits):.2f}")

    with open(out_dir / "audit.json", "w") as f:
        json.dump({
            "dataset_root": str(dataset_root),
            "model": str(args.model),
            "model_aabb_size_mm": (hi - lo).tolist(),
            "associations": assoc,
            "samples": rows,
            "scale_fit": {"global": s_global, "bbox_rmse_px": rmse_global,
                          "per_sample_median": float(np.median(sfits))},
        }, f, indent=2)
    print(f"[out] {out_dir / 'audit.json'}")
    print(f"[out] {out_dir / 'overlays'} ({len(samples)} images)")


if __name__ == "__main__":
    main()
