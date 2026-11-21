"""Compare a frozen baseline with fine-tuning on the same labeled real test frames."""
import argparse
import html
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from ultralytics import YOLO

from report_yolo_action4 import prediction_boxes, score_frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--baseline-label', default='原 1000 张合成数据模型')
    parser.add_argument('--trained-label', default='加入 32 张真实训练帧后微调')
    args = parser.parse_args()
    run, dataset = args.run.resolve(), args.dataset.resolve()
    output = run / 'comparison'
    output.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    manifest = json.loads((dataset / 'manifest.json').read_text())
    model = YOLO(str(args.baseline.resolve()))
    result = model.val(data=str(dataset / 'data.yaml'), split='test', imgsz=960,
                       batch=2, device=0, workers=2, project=str(output),
                       name='baseline_test', plots=True, verbose=False)
    baseline_ap = {k: float(v) for k, v in result.results_dict.items()}
    fine_ap = json.loads((run / 'metrics.json').read_text())['test']
    rows, cards = [], []
    for r in manifest['frames']:
        if r['split'] != 'test':
            continue
        source = dataset / r['image']
        p = model.predict(str(source), imgsz=960, conf=0.25, iou=0.7,
                          device=0, verbose=False)[0]
        baseline = [dict(xyxy=b, confidence=c) for b, c in zip(p.boxes.xyxy.cpu().tolist(), p.boxes.conf.cpu().tolist())]
        baseline.sort(key=lambda b: -b['confidence'])
        fine = prediction_boxes(run / 'test_predictions/labels' / (source.stem + '.txt'), r['width'], r['height'])
        panels = []
        scores = dict(image=r['image'], targets=r['boxes'])
        for name, predictions in [('baseline', baseline), ('mixed', fine)]:
            scores[name] = dict(score_frame(predictions, r['boxes']), predictions=predictions)
            im = Image.open(source).convert('RGB')
            draw = ImageDraw.Draw(im)
            for box in r['boxes']:
                draw.rectangle(box['xyxy'], outline='#00ff70', width=7)
            for box in predictions:
                draw.rectangle(box['xyxy'], outline='#ff8030', width=5)
            im.thumbnail((812, 616))
            panels.append(im)
        sheet = Image.new('RGB', (1624, 644), '#18212c')
        for i, panel in enumerate(panels):
            sheet.paste(panel, (i * 812, 28))
        ImageDraw.Draw(sheet).text((10, 7), source.stem + ' | BEFORE', fill='white')
        ImageDraw.Draw(sheet).text((822, 7), 'AFTER | Green=annotation Orange=prediction', fill='white')
        sheet.save(output / (source.stem + '.jpg'), quality=94)
        cards.append(f'<h2>{source.stem}</h2><a href="{source.stem}.jpg"><img src="{source.stem}.jpg"></a>')
        rows.append(scores)
    totals = {}
    for name, ap in [('baseline', baseline_ap), ('mixed', fine_ap)]:
        tp, fp, fn = (sum(r[name][k] for r in rows) for k in ('tp', 'fp', 'fn'))
        totals[name] = dict(tp=tp, fp=fp, fn=fn, precision=tp / max(tp + fp, 1),
                            recall=tp / max(tp + fn, 1), ap=ap)
    labels = dict(baseline=args.baseline_label, mixed=args.trained_label)
    report = dict(confidence=0.25, matching_iou=0.5, nms_iou=0.7,
                  model_labels=labels, baseline_weights=str(args.baseline.resolve()),
                  note=manifest['summary']['evaluation_note'], totals=totals, frames=rows)
    (output / 'comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    table = []
    for name, r in totals.items():
        table.append(f'<tr><td>{html.escape(labels[name])}</td><td>{r["ap"]["metrics/mAP50(B)"]:.3f}</td><td>{r["ap"]["metrics/mAP50-95(B)"]:.3f}</td><td>{r["precision"]:.3f}</td><td>{r["recall"]:.3f}</td><td>{r["fp"]}</td><td>{r["fn"]}</td></tr>')
    target_count = sum(len(r['targets']) for r in rows)
    (output / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>YOLO 重训前后对比</title><style>body{max-width:1650px;margin:30px auto;font:17px sans-serif;background:#14212a;color:#eee}img{width:100%}td,th{padding:10px;border:1px solid #aaa}a{color:#8dd2ff}</style>'
        f'<h1>同一批 {len(rows)} 张真实测试帧：YOLO 重训前后对比</h1><p>{target_count} 个相机框。'
        f'左侧 baseline：{html.escape(args.baseline_label)}；右侧 mixed：{html.escape(args.trained_label)}。'
        '绿色为标注，橙色为预测。固定 conf=0.25、匹配 IoU≥0.5。</p><p>' + html.escape(report['note']) + '</p><table><tr><th>模型</th><th>mAP50</th><th>mAP50–95</th><th>Precision</th><th>Recall</th><th>误检</th><th>漏检</th></tr>' + ''.join(table) + '</table>' + ''.join(cards))
    print(json.dumps(totals, indent=2))


if __name__ == '__main__':
    main()
