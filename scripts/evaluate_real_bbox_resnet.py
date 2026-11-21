"""Freeze bbox inputs and compare real eight-corner predictions in common coordinates."""
import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bop_bb8.net import decode_topk, heatmap_to_input_points
from bop_bb8.real_adaptation import match_boxes, original_to_reference, summarize_real
from bop_bb8.visualize import EDGES
from evaluate_cleanblur5000 import load_model
from infer_resnet_yolo_real import prepare_crop, to_original


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False))


def freeze_cases(real_root, yolo_path, destination):
    from ultralytics import YOLO
    annotation_path = real_root / 'annotations.json'
    records = json.loads(annotation_path.read_text())['records']
    detector = YOLO(str(yolo_path))
    cases, detections = [], []
    for frame in sorted({r['frame'] for r in records}):
        targets = [r for r in records if r['frame'] == frame]
        source = (real_root / targets[0]['image']).resolve()
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if any(r['image_sha256'] != digest for r in targets):
            raise ValueError(f'Image content changed: {source}')
        p = detector.predict(str(source), imgsz=960, conf=.25, iou=.7,
                             device=0, verbose=False)[0]
        boxes, confidences = p.boxes.xyxy.cpu().tolist(), p.boxes.conf.cpu().tolist()
        matches = match_boxes([r['box_xyxy'] for r in targets], boxes)
        detections.append(dict(frame=frame, source=str(source), boxes=boxes, confidences=confidences))
        for i, r in enumerate(targets):
            match = matches.get(i)
            cases.append(dict(r, source=str(source),
                              yolo_box=boxes[match[0]] if match else None,
                              yolo_confidence=confidences[match[0]] if match else None,
                              bbox_iou=match[1] if match else None))
    document = dict(annotation_sha256=hashlib.sha256(annotation_path.read_bytes()).hexdigest(),
                    yolo_weights=str(yolo_path), yolo_sha256=hashlib.sha256(yolo_path.read_bytes()).hexdigest(),
                    yolo_confidence=.25, yolo_nms_iou=.7, matching_iou=.5,
                    reference_box_note='Frozen visually reviewed reference boxes from the real corner annotation set; not calibrated ground truth.',
                    detections=detections, records=cases)
    write_json(destination, document)
    del detector
    torch.cuda.empty_cache()
    return document


def draw_wire(image, points, color, basis=None):
    result = image.copy()
    points = np.rint(points).astype(int)
    for a, b in EDGES:
        cv2.line(result, tuple(points[a]), tuple(points[b]), color, 1, cv2.LINE_AA)
    for i, p in enumerate(points):
        inferred = basis is not None and basis[i] != 'visible_edge_reference'
        cv2.circle(result, tuple(p), 3, color, 1 if inferred else -1, cv2.LINE_AA)
        cv2.putText(result, str(i), tuple(p + [3, -3]), cv2.FONT_HERSHEY_SIMPLEX, .32, color, 1, cv2.LINE_AA)
    return result


def panel(image, title):
    p = cv2.copyMakeBorder(image, 30, 0, 0, 0, cv2.BORDER_CONSTANT, value=(30, 35, 40))
    cv2.putText(p, title, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, .43, (245, 245, 245), 1, cv2.LINE_AA)
    return p


def run_evaluation(cases, checkpoints, names, output):
    records = [dict(r, predictions={}) for r in cases['records']]
    images = {r['source']: cv2.imread(r['source']) for r in records}
    tensors, slots, crops = [], [], {}
    for i, r in enumerate(records):
        original = images[r['source']]
        if original is None:
            raise FileNotFoundError(r['source'])
        for mode, box in [('reference', r['box_xyxy']), ('yolo', r['yolo_box'])]:
            if box is None:
                continue
            roi, tensor, crop = prepare_crop(original, box, (256, 256), 1.3)
            if mode == 'reference' and list(crop) != r['crop_xywh']:
                raise ValueError(f'Reference crop changed: {r["id"]}')
            crops[i, mode] = (roi, crop)
            tensors.append(tensor)
            slots.append((i, mode))
    metadata, heatmaps = {}, {}
    for name, checkpoint in zip(names, checkpoints):
        model, metadata[name] = load_model(checkpoint, torch.device('cuda'))
        for r in records:
            r['predictions'][name] = dict(reference=None, yolo=None)
        with torch.inference_mode():
            for start in range(0, len(tensors), 4):
                logits = model(torch.stack(tensors[start:start+4]).cuda()).float()
                points = heatmap_to_input_points(decode_topk(logits, from_logits=True),
                                                tuple(logits.shape[-2:]), (256, 256)).cpu().numpy()
                probs = logits.sigmoid().cpu().numpy()
                for offset, (predicted, hm) in enumerate(zip(points, probs), start):
                    i, mode = slots[offset]
                    r = records[i]
                    roi, crop = crops[i, mode]
                    original = to_original(predicted, crop, (256, 256))
                    common = original_to_reference(original, r['crop_xywh'])
                    target = np.array(r['keypoints_input'])
                    r['predictions'][name][mode] = dict(
                        points_original=original.tolist(), points_reference=common.tolist(),
                        points_crop=predicted.tolist(), crop_xywh=list(crop),
                        errors_reference_px=np.linalg.norm(common-target, axis=1).tolist(),
                        errors_original_px=np.linalg.norm(original-np.array(r['keypoints_original']), axis=1).tolist(),
                        peaks=hm.max(axis=(1, 2)).tolist(),
                        target_corners_outside_crop=int(np.any((original_to_reference(r['keypoints_original'], crop) < 0) |
                                                             (original_to_reference(r['keypoints_original'], crop) >= 256), axis=1).sum()))
                    heatmaps[f'{name}_{i}_{mode}'] = hm
        del model
        torch.cuda.empty_cache()
    metrics = {name: {split: {mode: summarize_real(records, name, mode, split)
                             for mode in ('reference', 'yolo')}
                      for split in ('adapt', 'validation', 'test')} for name in names}
    paired = {name: {split: {mode: summarize_real(records, name, mode, split, matched_only=True)
                            for mode in ('reference', 'yolo')}
                     for split in ('adapt', 'validation', 'test')} for name in names}
    report = dict(models=metadata, metrics=metrics, matched_instances_only=paired,
                  coordinate_unit='256px fixed reference crop; predictions from both bbox modes first map back to original pixels',
                  note='Small same-video pilot with approximate manual full-eight labels, including inferred hidden/virtual corners. All eight supervised. Missing detector outputs count as failed PCK; mean error is conditional on outputs. Unannotated cameras are not counted as false positives.',
                  split_instances=dict(Counter(r['split'] for r in records)),
                  yolo_matched=sum(r['yolo_box'] is not None for r in records), records=records)
    write_json(output / 'results.json', report)
    np.savez_compressed(output / 'heatmaps.npz', **heatmaps)
    cards, table = [], []
    for name in names:
        for split in ('adapt', 'validation', 'test'):
            for mode in ('reference', 'yolo'):
                m = metrics[name][split][mode]
                mean = m['all8']['mean_error_px']
                table.append(f'<tr><td>{html.escape(name)}</td><td>{split}</td><td>{mode}</td>'
                             f'<td>{m["predicted_instances"]}/{m["instances"]}</td>'
                             f'<td>{mean:.2f}</td><td>{m["all8"]["pck5"]:.1%}</td></tr>' if mean is not None else
                             f'<tr><td>{html.escape(name)}</td><td>{split}</td><td>{mode}</td><td>0/{m["instances"]}</td><td>无输出</td><td>0%</td></tr>')
    for i, r in enumerate(records):
        full = images[r['source']].copy()
        for mode, color in [('reference', (255, 220, 0)), ('yolo', (0, 150, 255))]:
            box = r['box_xyxy'] if mode == 'reference' else r['yolo_box']
            if box is None:
                continue
            x1, y1, x2, y2 = np.rint(box).astype(int)
            x, y, w, h = crops[i, mode][1]
            cv2.rectangle(full, (x1, y1), (x2, y2), color, 5)
            cv2.rectangle(full, (x, y), (x+w, y+h), color, 2)
            cv2.putText(full, mode.upper() + ' BBOX / CROP', (x, max(25, y-12)),
                        cv2.FONT_HERSHEY_SIMPLEX, .8, color, 2, cv2.LINE_AA)
        cv2.imwrite(str(output / f'{r["id"]}_full.jpg'), full, [cv2.IMWRITE_JPEG_QUALITY, 93])
        row_images = []
        for mode in ('reference', 'yolo'):
            if (i, mode) not in crops:
                blank = np.zeros((256, 256*(len(names)+1), 3), np.uint8)
                row_images.append(panel(blank, 'YOLO MISS: no GT-box fallback'))
                continue
            roi, crop = crops[i, mode]
            raw = roi.copy()
            box = r['box_xyxy'] if mode == 'reference' else r['yolo_box']
            corners = original_to_reference(np.array(box).reshape(2, 2), crop)
            cv2.rectangle(raw, tuple(np.rint(corners[0]).astype(int)), tuple(np.rint(corners[1]).astype(int)), (255, 220, 0), 1)
            panes = [panel(raw, mode.upper() + ' INPUT + BBOX')]
            target = original_to_reference(r['keypoints_original'], crop)
            for name in names:
                p = r['predictions'][name][mode]
                overlay = draw_wire(roi, target, (0, 255, 100), r['point_basis'])
                overlay = draw_wire(overlay, p['points_crop'], (230, 30, 230))
                panes.append(panel(overlay, name + f' {np.mean(p["errors_reference_px"]):.1f}px'))
            row_images.append(np.hstack(panes))
        cv2.imwrite(str(output / f'{r["id"]}.jpg'), np.vstack(row_images), [cv2.IMWRITE_JPEG_QUALITY, 95])
        overlap = '漏检' if r['bbox_iou'] is None else f'{r["bbox_iou"]:.3f}'
        cards.append(f'<article><h2>{r["id"]} · {r["split"]} · bbox IoU {overlap}</h2>'
                     f'<a href="{r["id"]}_full.jpg">查看整帧：参考框、YOLO 框及两者裁剪范围</a>'
                     f'<p>上排：参考框裁剪；下排：新版 YOLO 框裁剪。</p><img loading="lazy" src="{r["id"]}.jpg"></article>')
    (output / 'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        '<title>ResNet 真实框配对评估</title><style>body{max-width:1450px;margin:30px auto;padding:0 20px;background:#14212a;color:#eef3f6;font:17px/1.65 sans-serif}img{max-width:100%;image-rendering:auto}td,th{border:1px solid #506776;padding:8px}table{border-collapse:collapse}a{color:#85d7ff}article{border-top:1px solid #456;padding:24px 0}</style>'
        '<h1>真实相机：参考 bbox 与新版 YOLO bbox 配对评估</h1>'
        '<p>相同 ResNet 权重、相同真实原图、padding=1.3、256×256 输入、稳定 Top-9 解码。框线只在展示图绘制，不进入网络。</p>'
        '<p>绿色为人工八点参考（空心点为推定点），紫色为预测。每个实例八点全部评估，YOLO 漏检计入 PCK 分母；平均误差只统计有输出实例。两类框的误差都换算到固定参考裁剪的 256 像素坐标。</p>'
        '<p>仅 7 帧 / 9 实例：5 训练、3 验证、1 测试。标签包含人工推定隐藏点，同视频已用于适配与诊断，不是独立视频盲测。未标注的其他相机不计作误检。</p>'
        '<table><tr><th>模型</th><th>分组</th><th>bbox</th><th>有输出/全部</th><th>8点平均误差 px</th><th>PCK@5</th></tr>'
        + ''.join(table) + '</table><p><a href="results.json">逐角点数据及可见/推定点分组指标</a> · <a href="cases.json">固定 bbox 输入清单</a></p>' + ''.join(cards) + '</html>')
    print(json.dumps(dict(yolo_matched=report['yolo_matched'], metrics=metrics), ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--real', type=Path, required=True)
    parser.add_argument('--yolo', type=Path)
    parser.add_argument('--cases', type=Path)
    parser.add_argument('--checkpoints', type=Path, nargs='+', required=True)
    parser.add_argument('--names', nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if len(args.names) != len(args.checkpoints) or len(set(args.names)) != len(args.names):
        parser.error('Provide unique names matching checkpoints')
    if not args.cases and not args.yolo:
        parser.error('Provide --cases or --yolo')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.cases:
        cases = json.loads(args.cases.read_text())
        expected = hashlib.sha256((args.real / 'annotations.json').read_bytes()).hexdigest()
        if cases['annotation_sha256'] != expected:
            raise ValueError('Frozen annotation metadata changed')
        for r in cases['records']:
            if hashlib.sha256(Path(r['source']).read_bytes()).hexdigest() != r['image_sha256']:
                raise ValueError('Frozen image content changed')
        write_json(args.output / 'cases.json', cases)
    else:
        cases = freeze_cases(args.real.resolve(), args.yolo.resolve(), args.output / 'cases.json')
    run_evaluation(cases, args.checkpoints, args.names, args.output)


if __name__ == '__main__':
    main()
