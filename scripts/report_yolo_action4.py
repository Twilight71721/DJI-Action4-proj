"""Build a detector review page; optional unlabeled real-frame diagnostics."""
from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[2] - b[0]) * (b[3] - b[1])
    return intersection / max(aa + bb - intersection, 1e-12)


def prediction_boxes(path, width, height):
    if not path.exists():
        return []
    result = []
    for line in path.read_text().splitlines():
        cls, cx, cy, w, h, confidence = map(float, line.split())
        if int(cls) != 0:
            raise ValueError("Unexpected predicted class")
        result.append(dict(xyxy=[(cx - w / 2) * width, (cy - h / 2) * height,
                                 (cx + w / 2) * width, (cy + h / 2) * height],
                           confidence=confidence))
    return sorted(result, key=lambda p: -p["confidence"])


def score_frame(predictions, targets):
    remaining = set(range(len(targets)))
    matches = []
    for pred in predictions:
        candidates = [(iou(pred["xyxy"], targets[j]["xyxy"]), j) for j in remaining]
        value, index = max(candidates, default=(0.0, -1))
        if value >= 0.5:
            remaining.remove(index)
            matches.append(value)
    return dict(tp=len(matches), fp=len(predictions) - len(matches), fn=len(remaining),
                matched_ious=matches)


def make_sheet(paths, destination, columns=4):
    sheet = Image.new("RGB", (columns * 406, ((len(paths) + columns - 1) // columns) * 338), "#18212c")
    for i, path in enumerate(paths):
        with Image.open(path) as im:
            tile = im.convert("RGB")
        tile.thumbnail((406, 308))
        x, y = (i % columns) * 406, (i // columns) * 338
        sheet.paste(tile, (x, y + 25))
        ImageDraw.Draw(sheet).text((x + 5, y + 5), path.stem, fill="white")
    sheet.save(destination, quality=90)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--real-frames", type=Path)
    args = parser.parse_args()
    run, dataset = args.run.resolve(), args.dataset.resolve()
    review = run / "review"
    review.mkdir(exist_ok=True)
    manifest = json.loads((dataset / "manifest.json").read_text())
    metrics = json.loads((run / "metrics.json").read_text())
    results, panels = {}, []
    for split in ("val", "test"):
        frames = []
        for record in manifest["frames"]:
            if record["split"] != split:
                continue
            stem = Path(record["image"]).stem
            predictions = prediction_boxes(run / f"{split}_predictions/labels/{stem}.txt",
                                           record["width"], record["height"])
            score = score_frame(predictions, record["boxes"])
            score.update(image=record["image"], group=record["group"], predictions=predictions)
            frames.append(score)
        tp, fp, fn = (sum(f[k] for f in frames) for k in ("tp", "fp", "fn"))
        results[split] = dict(tp=tp, fp=fp, fn=fn, precision=tp / max(tp + fp, 1),
                              recall=tp / max(tp + fn, 1), frames=frames)
        # Rank failures first, then worse matching localization.
        worst = sorted(frames, key=lambda f: (-(f["fp"] + f["fn"]),
                                              np.mean(f["matched_ious"]) if f["matched_ious"] else 0))[:8]
        by_image = {r["image"]: r for r in manifest["frames"]}
        for f in worst:
            record = by_image[f["image"]]
            with Image.open(dataset / f["image"]) as im:
                panel = im.convert("RGB")
            draw = ImageDraw.Draw(panel)
            for b in record["boxes"]:
                draw.rectangle(b["xyxy"], outline="#00ff70", width=4)
            for b in f["predictions"]:
                draw.rectangle(b["xyxy"], outline="#ff8030", width=3)
            path = review / (split + "_" + Path(f["image"]).name)
            panel.save(path)
            panels.append(path)
    make_sheet(panels, review / "heldout.jpg")
    report = dict(confidence=0.25, nms_iou=0.7, matching_iou=0.5, splits=results)
    (review / "fixed_threshold_metrics.json").write_text(json.dumps(report, indent=2))
    real_section = ""
    if args.real_frames:
        from ultralytics import YOLO

        paths = sorted(args.real_frames.glob("*.jpg"))
        if not paths:
            raise FileNotFoundError(args.real_frames)
        selected = [paths[i] for i in np.linspace(0, len(paths) - 1, min(24, len(paths)), dtype=int)]
        model = YOLO(str(run / "train/weights/best.pt"))
        predictions, real_panels = [], []
        for path in selected:
            result = model.predict(str(path), imgsz=960, conf=0.25, iou=0.7,
                                   device=0, verbose=False)[0]
            destination = review / f"real_{path.stem}.jpg"
            Image.fromarray(result.plot()[:, :, ::-1]).save(destination, quality=90)
            real_panels.append(destination)
            predictions.append(dict(source=str(path), xyxy=result.boxes.xyxy.cpu().tolist(),
                                    confidence=result.boxes.conf.cpu().tolist()))
        make_sheet(real_panels, review / "real_diagnostics.jpg")
        (review / "real_predictions.json").write_text(json.dumps(dict(
            selection="24 uniformly spaced frames; fixed before inspecting predictions",
            usage="Unlabeled target-domain diagnostic only, not independent test accuracy",
            predictions=predictions), indent=2))
        empty = sum(not p["xyxy"] for p in predictions)
        real_section = (f'<h2>真实视频抽帧诊断</h2><p>{len(predictions)} 张均匀抽帧中，'
                        f'{empty} 张没有输出框。输出框数量不代表正确检测数量。</p>'
                        '<p>未提供人工真值，不能计算准确率。该视频已用于合成域设计，不是独立盲测。'
                        '未用这些帧训练或选择权重。</p>'
                        '<a href="real_diagnostics.jpg"><img src="real_diagnostics.jpg"></a>')
    rows = []
    for split in ("val", "test"):
        m, fixed = metrics[split], results[split]
        rows.append(f'<tr><td>{split}</td><td>{m["metrics/mAP50(B)"]:.3f}</td>'
                    f'<td>{m["metrics/mAP50-95(B)"]:.3f}</td>'
                    f'<td>{fixed["precision"]:.3f}</td><td>{fixed["recall"]:.3f}</td>'
                    f'<td>{fixed["fp"]}</td><td>{fixed["fn"]}</td></tr>')
    with (run / "train/results.csv").open() as f:
        history = list(csv.DictReader(f))
    (review / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Action4 YOLO11s baseline</title>'
        '<style>body{font:16px sans-serif;max-width:1200px;margin:30px auto;background:#f4f6f8;color:#172331}'
        'img{width:100%}td,th{padding:10px;border:1px solid #ccc}table{border-collapse:collapse}</style>'
        '<h1>Action4 YOLO11s 检测基线</h1><p>'
        f'{manifest["summary"]["images"]["train"]} 张训练 / '
        f'{manifest["summary"]["images"]["val"]} 张验证 / '
        f'{manifest["summary"]["images"]["test"]} 张测试；按分组划分，'
        '非随机同分布划分。最佳模型只按验证集选择。测试集仅在训练结束后评估。</p>'
        f'<p>{html.escape(manifest["summary"].get("evaluation_note", ""))}</p>'
        f'<p>完成 {len(history)} 轮训练。固定预测阈值 conf=0.25，匹配 IoU≥0.5。</p>'
        '<table><tr><th>集合</th><th>mAP50</th><th>mAP50–95</th><th>Precision@0.25</th>'
        '<th>Recall@0.25</th><th>误检数</th><th>漏检数</th></tr>' + ''.join(rows) + '</table>'
        '<h2>训练曲线</h2><img src="../train/results.png">'
        '<h2>验证与测试代表性失败 / 定位较差样本</h2><p>绿色：标注；橙色：预测。</p>'
        '<a href="heldout.jpg"><img src="heldout.jpg"></a>' + real_section +
        '<p><a href="../train/weights/best.pt">最佳权重</a> · '
        '<a href="fixed_threshold_metrics.json">逐图诊断</a> · '
        '<a href="../run_config.json">训练配置与版本</a></p>')
    print(json.dumps({s: {k: v for k, v in r.items() if k != "frames"}
                      for s, r in results.items()}, indent=2))


if __name__ == "__main__":
    main()
