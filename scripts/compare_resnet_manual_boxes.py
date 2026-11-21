"""Compare frozen YOLO and manually estimated boxes with the same BB8 model."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from infer_resnet_yolo_real import prepare_crop, to_original, draw_points
from bop_bb8.net import ResNetFPN, decode_topk, heatmap_to_input_points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boxes", type=Path, required=True)
    parser.add_argument("--yolo-predictions", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "comparison.json").exists():
        raise FileExistsError("Comparison already exists")
    annotations = json.loads(args.boxes.read_text())
    detections = json.loads(args.yolo_predictions.read_text())["predictions"]
    lookup = {Path(e["source"]).name: e for e in detections}
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved = checkpoint["args"]
    size, padding = tuple(saved["input_size"]), saved["crop_padding"]
    torch.set_num_threads(4)
    model = ResNetFPN(backbone=saved["backbone"], fpn_dim=saved["fpn_dim"], pretrained=False).cuda().eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    records, panels, cards, arrays = [], [], [], {}
    with torch.inference_mode():
        for ann in annotations["annotations"]:
            source = Path(ann["source"])
            detection = lookup[source.name]
            index = ann["yolo_detection_index"]
            image = cv2.imread(str(source))
            if image is None:
                raise FileNotFoundError(source)
            boxes = {"YOLO": detection["xyxy"][index], "MANUAL": ann["xyxy"]}
            record = dict(source=str(source), detection_index=index, results={})
            stem = f"{source.stem}_det{index}"
            images = []
            for method, box in boxes.items():
                roi, tensor, crop = prepare_crop(image, box, size, padding)
                logits = model(tensor[None].cuda()).float()
                probs = torch.sigmoid(logits)
                points = heatmap_to_input_points(decode_topk(logits, k=9, from_logits=True),
                                                 tuple(probs.shape[-2:]), (size[1], size[0]))[0].cpu().numpy()
                original = to_original(points, crop, size)
                arrays[f"{stem}_{method}"] = probs[0].cpu().numpy()
                if not np.isfinite(original).all():
                    raise ValueError("Non-finite prediction")
                prediction = draw_points(roi, points)
                images.extend([roi, prediction])
                record["results"][method] = dict(box_xyxy=box, crop_xywh=crop,
                    corners_roi=points.tolist(), corners_original=original.tolist(),
                    heatmap_peaks=probs[0].amax(dim=(-2, -1)).cpu().tolist())
            panel = np.hstack(images)
            panel = cv2.copyMakeBorder(panel, 55, 0, 0, 0, cv2.BORDER_CONSTANT, value=(25, 25, 25))
            cv2.putText(panel, stem, (8, 20), 0, .5, (255, 255, 255), 1)
            for j, label in enumerate(("YOLO crop", "YOLO -> corners", "MANUAL crop", "MANUAL -> corners")):
                cv2.putText(panel, label, (j*size[0]+8, 43), 0, .48, (255, 255, 255), 1)
            cv2.imwrite(str(args.output / f"{stem}.jpg"), panel)
            panels.append(panel)
            # Show both predictions in exactly the same original-image viewport.
            b = np.array(list(boxes.values()))
            x1, y1 = np.maximum(0, b[:, :2].min(axis=0)-80).astype(int)
            x2, y2 = np.minimum([image.shape[1], image.shape[0]], b[:, 2:].max(axis=0)+80).astype(int)
            contexts = []
            for method, box in boxes.items():
                vis = draw_points(image, np.array(record["results"][method]["corners_original"]), thickness=2)
                color = (0, 200, 255) if method == "YOLO" else (0, 240, 0)
                cv2.rectangle(vis, tuple(np.rint(box[:2]).astype(int)), tuple(np.rint(box[2:]).astype(int)), color, 2)
                vis = vis[y1:y2, x1:x2]
                scale = min(500/vis.shape[1], 500/vis.shape[0])
                vis = cv2.resize(vis, None, fx=scale, fy=scale)
                tile = np.full((535, 500, 3), 25, np.uint8)
                tile[35:35+vis.shape[0], :vis.shape[1]] = vis
                cv2.putText(tile, method, (8, 23), 0, .6, color, 1)
                contexts.append(tile)
            cv2.imwrite(str(args.output / f"{stem}_context.jpg"), np.hstack(contexts))
            records.append(record)
            cards.append(f'<article><h2>{html.escape(stem)}</h2><img src="{stem}.jpg">'
                         f'<details><summary>相同原图区域对照（左 YOLO / 右人工框）</summary>'
                         f'<img src="{stem}_context.jpg"></details></article>')
    cv2.imwrite(str(args.output / "overview.jpg"), np.vstack(panels))
    np.savez_compressed(args.output / "heatmaps.npz", **arrays)
    output = dict(checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                  manual_boxes_sha256=hashlib.sha256(args.boxes.read_bytes()).hexdigest(),
                  preprocessing=dict(input_size=size, padding=padding, decoder="sigmoid + weighted top9"),
                  no_training=True, no_corner_ground_truth=True, records=records)
    (args.output / "comparison.json").write_text(json.dumps(output, indent=2))
    (args.output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>人工机身框与 YOLO 框对照</title>'
        '<style>body{font:17px sans-serif;background:#14212a;color:#eee;max-width:1100px;margin:35px auto}'
        'img{max-width:100%}a{color:#8dd2ff}article{background:#263641;padding:12px;margin:20px 0}p{line-height:1.7}</style>'
        '<h1>固定 ResNet50：人工机身框 vs YOLO 框</h1>'
        '<p>3 张真实帧、4 个相机实例。先根据原图人工估计紧包机身框，排除支架、腕带和线缆，再运行模型。'
        '人工框不是经过独立复核的精确真值，没有按模型输出反复调框。</p>'
        '<p>每行依次为：YOLO 裁剪、YOLO 框下角点、人工框裁剪、人工框下角点。'
        '权重、1.3 倍裁剪、256×256 输入、归一化和解码完全相同。未训练模型，未使用真实角点标签，不能计算角点误差或 PCK。</p>'
        '<p><a href="manual_boxes.json">人工框坐标</a> · <a href="comparison.json">全部预测坐标</a></p>'
        '<details><summary>查看人工框标注预览</summary><img src="manual_box_review.jpg"></details>'
        + ''.join(cards))
    print(f"Completed {len(records)} paired comparisons: {args.output / 'index.html'}")


if __name__ == "__main__":
    main()
