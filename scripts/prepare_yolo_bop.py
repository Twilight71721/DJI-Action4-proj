"""Convert BOP images to one-class YOLO labels using explicit scene groups."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

from PIL import Image, ImageDraw


def clip_box(box, width, height):
    import math

    x, y, w, h = map(float, box)
    if not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0:
        raise ValueError(f"Invalid bbox: {box}")
    x1, y1 = max(0.0, x), max(0.0, y)
    x2, y2 = min(float(width), x + w), min(float(height), y + h)
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def normalize_box(box, width, height):
    x1, y1, x2, y2 = box
    return [(x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
            (x2 - x1) / width, (y2 - y1) / height]


def scene_assignments(config, available):
    assignments = {}
    names = set()
    for group in config["groups"]:
        if group["split"] not in {"train", "val", "test"} or group["name"] in names:
            raise ValueError("Invalid split or duplicated group name")
        names.add(group["name"])
        for scene in group["scenes"]:
            if scene in assignments:
                raise ValueError(f"Scene assigned more than once: {scene}")
            assignments[scene] = (group["split"], group["name"])
    if set(assignments) != set(available):
        raise ValueError("Split manifest must assign every available scene exactly once")
    if not {"train", "val"}.issubset({v[0] for v in assignments.values()}):
        raise ValueError("Training and validation scenes are required")
    return assignments


def prepare(source, output, split_path):
    source, output = source.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(output)
    config = json.loads(split_path.read_text())
    scenes = sorted(p for p in (source / "train_pbr").iterdir() if p.name.isdigit())
    assignments = scene_assignments(config, [p.name for p in scenes])
    records, hashes, preview = [], {}, []
    for scene in scenes:
        split, group = assignments[scene.name]
        gt = json.loads((scene / "scene_gt.json").read_text())
        info = json.loads((scene / "scene_gt_info.json").read_text())
        if set(gt) != set(info):
            raise ValueError(f"GT/info frame IDs differ: {scene}")
        for frame_id in sorted(gt, key=int):
            image_id = f"{int(frame_id):06d}"
            candidates = [p for p in (scene / "rgb").glob(f"{image_id}.*")
                          if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
            if len(candidates) != 1:
                raise ValueError(f"Expected one RGB: {scene}/{image_id}")
            rgb = candidates[0]
            with Image.open(rgb) as im:
                im.load()
                width, height = im.size
            digest = hashlib.sha256(rgb.read_bytes()).hexdigest()
            if digest in hashes:
                raise ValueError(f"Duplicate RGB: {rgb} and {hashes[digest]}")
            hashes[digest] = str(rgb)
            if len(gt[frame_id]) != len(info[frame_id]):
                raise ValueError(f"Instance count mismatch: {rgb}")
            boxes, invisible = [], []
            for index, (obj, ann) in enumerate(zip(gt[frame_id], info[frame_id])):
                if obj["obj_id"] != 1:
                    raise ValueError("Only Action4 obj_id=1 is supported")
                box = clip_box(ann["bbox_obj"], width, height)
                if ann["px_count_visib"] == 0 or box is None:
                    invisible.append(index)
                    continue
                boxes.append({"instance": index, "xyxy": box,
                              "xywhn": normalize_box(box, width, height),
                              "visib_fract": ann["visib_fract"]})
            records.append(dict(scene=scene.name, frame=image_id, split=split, group=group,
                                source=str(rgb), sha256=digest, width=width, height=height,
                                boxes=boxes, invisible_instances=invisible))
    output.mkdir(parents=True)
    for record in records:
        split = record["split"]
        stem = f"{record['scene']}_{record['frame']}"
        rgb = Path(record["source"])
        destination = output / "images" / split / (stem + rgb.suffix)
        labels = output / "labels" / split / (stem + ".txt")
        destination.parent.mkdir(parents=True, exist_ok=True)
        labels.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(rgb, destination)
        labels.write_text("".join("0 " + " ".join(f"{v:.9f}" for v in b["xywhn"]) + "\n"
                                  for b in record["boxes"]))
        record["image"] = str(destination.relative_to(output))
        record["label"] = str(labels.relative_to(output))
        if record["frame"] in {"000000", "000005"}:
            with Image.open(rgb) as im:
                tile = im.convert("RGB")
            draw = ImageDraw.Draw(tile)
            for box in record["boxes"]:
                draw.rectangle(box["xyxy"], outline="#00ff70", width=4)
            tile.thumbnail((406, 308))
            panel = Image.new("RGB", (406, 336), "#18212c")
            panel.paste(tile, (0, 28))
            ImageDraw.Draw(panel).text((8, 6), f"{split} | {stem} | {record['group']}", fill="white")
            preview.append(panel)
    summary = dict(images=dict(Counter(r["split"] for r in records)),
                   instances={s: sum(len(r["boxes"]) for r in records if r["split"] == s)
                              for s in ("train", "val", "test")},
                   invisible_instances=sum(len(r["invisible_instances"]) for r in records),
                   bbox_policy="bbox_obj, image-clipped, normalized; accessories excluded",
                   split_policy=config, source=str(source), unique_rgb=len(hashes))
    (output / "manifest.json").write_text(json.dumps(dict(summary=summary, frames=records), indent=2))
    # JSON-quoted strings are also valid YAML scalars.
    splits = "".join(f"{s}: images/{s}\n" for s in ("train", "val", "test") if s in summary["images"])
    (output / "data.yaml").write_text(f"path: {json.dumps(str(output))}\n{splits}names:\n  0: action4\n")
    (output / "review").mkdir()
    sheet = Image.new("RGB", (406 * 4, 336 * ((len(preview) + 3) // 4)), "#18212c")
    for i, tile in enumerate(preview):
        sheet.paste(tile, ((i % 4) * 406, (i // 4) * 336))
    sheet.save(output / "review/labels.jpg", quality=90)
    (output / "review/index.html").write_text(
        '<meta charset="utf-8"><title>Action4 YOLO labels</title>'
        '<h1>Action4 YOLO label review</h1><p>Green: image-clipped full object bbox. '
        f'{len(preview)} representatives; held-out action segments.</p><img width="100%" src="labels.jpg">')
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.output, args.split)
