"""Validate native BOP outputs and build image galleries for wrist renders."""

import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bop_bb8.dataset import BOPInstanceDataset
from bop_bb8.geometry import BOPModel


def box(mask):
    yy, xx = np.where(mask)
    # Match this HCCEPose revision's bop_toolkit_lib.misc.calc_2d_bbox.
    return [int(xx.min()), int(yy.min()), int(xx.max() - xx.min()), int(yy.max() - yy.min())] if len(xx) else [-1, -1, -1, -1]


def contact_sheet(items, destination, columns=4, width=360):
    height = round(width * 1232 / 1624)
    sheet = Image.new("RGB", (columns * width, ((len(items) + columns - 1) // columns) * (height + 28)), "#182127")
    draw = ImageDraw.Draw(sheet)
    for i, (path, label) in enumerate(items):
        image = Image.open(path).convert("RGB")
        image.thumbnail((width, height))
        x, y = i % columns * width, i // columns * (height + 28)
        sheet.paste(image, (x + (width - image.width) // 2, y))
        draw.text((x + 7, y + height + 6), label, fill="white")
    sheet.save(destination, quality=92)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--expected", type=int, default=200)
    parser.add_argument("--allow-invisible", action="store_true",
                        help="Audit invisible BOP targets explicitly and exclude them from positive loader samples")
    args = parser.parse_args()
    root = args.root.resolve()
    review = root / "review"
    review.mkdir(exist_ok=True)
    (review / "overlays").mkdir(exist_ok=True)
    (review / "thumbs").mkdir(exist_ok=True)
    corners = np.asarray([x["xyz"] for x in json.loads((root / "models/corners_3d.json").read_text())["corners"]])
    errors, records, hashes, projection_errors, visibilities, widths, distances = [], [], [], [], [], [], []
    counts = Counter()
    truncated_instances = 0
    mask_count_deltas = []
    invisible_instances = []
    total_instances = 0
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    scenes = sorted((root / "train_pbr").glob("[0-9]*"))
    for scene in scenes:
        gt = json.loads((scene / "scene_gt.json").read_text())
        camera = json.loads((scene / "scene_camera.json").read_text())
        info = json.loads((scene / "scene_gt_info.json").read_text())
        meta = json.loads((root / "render_metadata" / f"batch_{int(scene.name):03d}" / "metadata.json").read_text())
        if set(gt) != set(camera) or set(gt) != set(info):
            errors.append(f"{scene.name}: JSON indices differ")
        for fid in sorted(gt, key=int):
            frame = int(fid)
            name = f"{scene.name}_{frame:06d}"
            path = scene / "rgb" / f"{frame:06d}.png"
            rgb = cv2.imread(str(path))
            depth = cv2.imread(str(scene / "depth" / f"{frame:06d}.png"), cv2.IMREAD_UNCHANGED)
            if rgb is None or depth is None:
                errors.append(f"{name}: missing RGB/depth")
                continue
            if rgb.shape[:2] != depth.shape or depth.dtype != np.uint16:
                errors.append(f"{name}: invalid depth shape/type")
            if tuple(rgb.shape[1::-1]) != tuple(meta["resolution"]):
                errors.append(f"{name}: resolution mismatch")
            hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
            overlay = rgb.copy()
            frame_meta = meta["frames"][frame]
            counts[frame_meta["scene"]] += 1
            counts["pose_" + frame_meta["hand_pose"]] += 1
            action = frame_meta.get("action") or {}
            lighting = frame_meta.get("lighting") or {}
            if action:
                counts["action_" + action["name"]] += 1
            if lighting:
                counts["lighting_" + lighting["profile"]] += 1
            keypoint_records = []
            for index, inst in enumerate(gt[fid]):
                total_instances += 1
                R = np.asarray(inst["cam_R_m2c"]).reshape(3, 3)
                t = np.asarray(inst["cam_t_m2c"])
                K = np.asarray(camera[fid]["cam_K"]).reshape(3, 3)
                xyz = (R @ corners.T).T + t
                uv = (K @ xyz.T).T
                uv = uv[:, :2] / uv[:, 2:]
                expected = np.asarray(frame_meta["instances"][index]["corners_2d_independent"])
                delta = float(np.max(np.abs(expected - uv)))
                projection_errors.append(delta)
                distances.append(float(np.linalg.norm(t) / 1000))
                if not np.isfinite(uv).all() or (xyz[:, 2] <= 0).any() or delta > 0.005:
                    errors.append(f"{name}/{index}: invalid projection {delta}")
                if not np.allclose(R.T @ R, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(R), 1, atol=1e-5):
                    errors.append(f"{name}/{index}: invalid rotation")
                m = cv2.imread(str(scene / "mask" / f"{frame:06d}_{index:06d}.png"), 0)
                v = cv2.imread(str(scene / "mask_visib" / f"{frame:06d}_{index:06d}.png"), 0)
                if m is None or v is None:
                    errors.append(f"{name}/{index}: missing mask")
                    continue
                m, v = m > 0, v > 0
                if m.shape != depth.shape or v.shape != depth.shape or np.any(v & ~m):
                    errors.append(f"{name}/{index}: invalid mask shape/subset")
                invisible = not v.any()
                if invisible:
                    invisible_instances.append({"scene": scene.name, "frame": frame, "instance": index,
                                                "reason": "outside_image" if not m.any() else "fully_occluded_in_image"})
                if (not m.any() or invisible) and not args.allow_invisible:
                    errors.append(f"{name}/{index}: empty mask")
                if v.any() and np.any(depth[v] == 0):
                    errors.append(f"{name}/{index}: missing visible depth")
                actual_box = box(m)
                instance_info = info[fid][index]
                bx, by, bw, bh = instance_info["bbox_obj"]
                ax, ay, aw, ah = actual_box
                truncated = not invisible and (bx < 0 or by < 0 or bx + bw >= rgb.shape[1] or by + bh >= rgb.shape[0])
                # Native BOP computes bbox_obj on a larger canvas, retaining offscreen extents.
                if invisible and args.allow_invisible:
                    # This native writer sets both boxes to -1 when no pixel is visible.
                    object_box_valid = instance_info["bbox_obj"] == [-1, -1, -1, -1]
                elif truncated:
                    truncated_instances += 1
                    object_box_valid = bx <= ax and by <= ay and bx + bw >= ax + aw and by + bh >= ay + ah
                else:
                    object_box_valid = actual_box == instance_info["bbox_obj"]
                if not object_box_valid or box(v) != instance_info["bbox_visib"]:
                    errors.append(f"{name}/{index}: bbox differs from mask")
                # Masks and info use separate native renders (normal vs 3x canvas).
                # Their floating-point rasterization can differ by a few edge pixels.
                count_delta = int(v.sum()) - instance_info["px_count_visib"]
                mask_count_deltas.append(count_delta)
                if abs(count_delta) > 5 or instance_info["px_count_all"] + 5 < int(m.sum()):
                    errors.append(f"{name}/{index}: pixel count differs from mask")
                expected_fraction = float(instance_info["px_count_visib"] / max(instance_info["px_count_all"], 1))
                if not np.isclose(expected_fraction, instance_info["visib_fract"]):
                    errors.append(f"{name}/{index}: visibility fraction differs")
                visibilities.append(expected_fraction)
                if bw >= 0:
                    widths.append(bw)
                for a, b in edges:
                    cv2.line(overlay, tuple(np.round(uv[a]).astype(int)), tuple(np.round(uv[b]).astype(int)), (70, 235, 70), 2, cv2.LINE_AA)
                for j, point in enumerate(uv):
                    cv2.putText(overlay, str(j), tuple(np.round(point).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, .5, (30, 100, 255), 1, cv2.LINE_AA)
                keypoint_records.append({"instance_index": index, "uv": uv.tolist(),
                                         "in_frame": ((uv[:, 0] >= 0) & (uv[:, 0] < rgb.shape[1]) &
                                                      (uv[:, 1] >= 0) & (uv[:, 1] < rgb.shape[0])).tolist()})
            if frame == 0:
                cv2.imwrite(str(review / "overlays" / f"{name}.jpg"), overlay)
            thumb = Image.open(path)
            thumb.thumbnail((406, 308))
            thumb.save(review / "thumbs" / f"{name}.jpg", quality=87)
            records.append({"name": name, "rgb": str(path.relative_to(root)), "scene": frame_meta["scene"],
                            "pose": frame_meta["hand_pose"], "keypoints": keypoint_records,
                            "frame": frame, "action": action.get("name", ""),
                            "progress": action.get("progress"), "lighting": lighting.get("profile", ""),
                            "views": [i.get("dominant_view", "unknown") for i in frame_meta["instances"]]})
    if len(records) != args.expected:
        errors.append(f"Expected {args.expected} RGB frames, got {len(records)}")
    if len(hashes) != len(set(hashes)):
        errors.append("Duplicate RGB files")
    model = BOPModel.from_models_info(root / "models/models_info.json")
    dataset = BOPInstanceDataset(root / "train_pbr", model, scenes=[s.name for s in scenes], input_size=(336, 336), heatmap_size=(84, 84),
                                 crop_mode="bbox", heatmap_kernel="gaussian",
                                 min_visib_fract=1e-6 if args.allow_invisible else 0.0)
    loaded = 0
    for index in np.unique(np.linspace(0, len(dataset) - 1, min(32, len(dataset)), dtype=int)):
        sample = dataset[int(index)]
        if sample.image.shape != (336, 336, 3) or sample.heatmaps.shape != (8, 84, 84) or not np.isfinite(sample.heatmaps).all():
            errors.append(f"Loader sample {index} invalid")
        loaded += 1
    summary = {"frames": len(records), "instances": total_instances, "unique_rgb": len(set(hashes)),
               "max_independent_projection_error_px": max(projection_errors, default=0),
               "visibility_range": [min(visibilities), max(visibilities)],
               "bbox_width_range_px": [min(widths), max(widths)],
               "camera_distance_range_m": [min(distances), max(distances)],
               "class_counts": dict(counts), "loader_samples_checked": loaded,
               "truncated_instances": truncated_instances,
               "mask_info_count_tolerance_px": 5,
               "max_mask_info_count_delta_px": max(map(abs, mask_count_deltas), default=0),
               "instances_with_mask_info_count_delta": sum(d != 0 for d in mask_count_deltas),
               "allow_invisible": args.allow_invisible,
               "invisible_instances": invisible_instances,
               "positive_loader_instances": len(dataset),
               "errors": errors, "geometry_validation_passed": not errors,
               "visual_status": "Synthetic reconstruction; not photorealistically identical to reference video"}
    (review / "validation.json").write_text(json.dumps(summary, indent=2))
    (review / "frames.json").write_text(json.dumps(records, indent=2))
    overview = [(root / r["rgb"], f"{r['name']} {r['action'] or r['scene']} / {r['lighting'] or r['pose']}") for r in records if r["frame"] == 0 or (r["action"] and r["frame"] == 3)]
    contact_sheet(overview, review / "overview.jpg")
    for page, start in enumerate(range(0, len(records), 25)):
        items = [(root / r["rgb"], f"{r['name']} {r['action'] or r['pose']} {r['lighting']}") for r in records[start:start + 25]]
        contact_sheet(items, review / f"contact_{page + 1:02d}.jpg", columns=5, width=324)
    cards = "\n".join(f'<a class="card" href="../{html.escape(r["rgb"])}" target="_blank" data-kind="{r["scene"]}"><img loading="lazy" src="thumbs/{r["name"]}.jpg"><span>{r["name"]} · {r["action"] or r["scene"]} · {r["lighting"] or r["pose"]}</span></a>' for r in records)
    page = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>Action 4 · HCCEPose __COUNT__</title>
<style>body{margin:32px;background:#151c22;color:#e7edf0;font:16px system-ui}h1{font-size:27px}p{max-width:1000px;color:#abbcc7;line-height:1.8}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:16px}.card{color:#cadce8;text-decoration:none;background:#25313b;border-radius:8px;overflow:hidden}.card img{width:100%;display:block}.card span{display:block;padding:12px;font-size:13px}button{background:#344754;color:white;border:0;padding:10px 20px;margin:0 8px 20px 0;border-radius:4px}a{color:#8dccff}</style>
<h1>Action 4 · HCCEPose / Cycles · __COUNT__ 张场景样本</h1>
<p>使用原始相机 GLB，加入手部、腕带、支架、屏幕、顶部标签、线缆与桌面衣物。点击查看 __RESOLUTION__ 原图。<br>这是参考视频引导的合成重建，仍有明显合成痕迹，不能称为与真实帧几乎相同；未将真实帧像素贴入合成图。动态衣物采用几何变形，尚未完成布料物理和手部接触求解。</p>
<p><a href="overview.jpg">代表图</a> · <a href="validation.json">几何验证报告</a> __COMPARISON__</p>
<button onclick="filter('all')">全部</button><button onclick="filter('raised')">抬腕</button><button onclick="filter('tabletop')">桌面</button>
<div class="grid">''' + cards + '''</div><script>function filter(kind){document.querySelectorAll('.card').forEach(c=>c.style.display=kind==='all'||c.dataset.kind===kind?'block':'none')}</script></html>'''
    page = page.replace("__COUNT__", str(len(records))).replace("__RESOLUTION__", "×".join(map(str, meta["resolution"])))
    page = page.replace("__COMPARISON__", ' · <a href="reference_comparison.jpg">真实帧对照</a>' if (review / "reference_comparison.jpg").exists() else "")
    (review / "index.html").write_text(page)
    print(json.dumps(summary, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
