"""Build a traceable real/synthetic detector dataset from visually reviewed boxes."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

from PIL import Image, ImageDraw

from prepare_yolo_bop import normalize_box


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--synthetic', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--real-repeat', type=int, default=10)
    args = parser.parse_args()
    if args.real_repeat < 1:
        parser.error('--real-repeat must be positive')
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(out)
    selection = json.loads((args.annotations / 'selection.json').read_text())
    boxes = json.loads((args.annotations / 'boxes_display.json').read_text())
    real = selection['frames']
    assert {str(r['frame']) for r in real} == set(boxes)
    assert len(real) == len(boxes) == 48
    for a in real:
        for b in real:
            if a['split'] != b['split']:
                assert abs(a['frame'] - b['frame']) >= 51
                assert a['segment'][1] < b['segment'][0] or b['segment'][1] < a['segment'][0]
    records, seen, cards = [], set(), []
    for r in real:
        split = r['split']
        stem = f"real_{r['frame']:06d}"
        source = Path(r['source'])
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        assert digest not in seen
        seen.add(digest)
        destination = out / 'images' / split / (stem + '.jpg')
        label = out / 'labels' / split / (stem + '.txt')
        destination.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        annotations = []
        for i, box in enumerate(boxes[str(r['frame'])]):
            x1, y1, x2, y2 = [v * 4 for v in box]
            assert 0 <= x1 < x2 <= r['width'] and 0 <= y1 < y2 <= r['height']
            annotations.append(dict(instance=i, xyxy=[x1, y1, x2, y2],
                                    xywhn=normalize_box([x1, y1, x2, y2], r['width'], r['height'])))
        assert len(annotations) == 2
        label.write_text(''.join('0 ' + ' '.join(f'{v:.9f}' for v in b['xywhn']) + '\n' for b in annotations))
        record = dict(r, sha256=digest, domain='real', scene='real_video',
                      group=f"real_{r['segment'][0]}_{r['segment'][1]}",
                      boxes=annotations, image=str(destination.relative_to(out)),
                      label=str(label.relative_to(out)), invisible_instances=[])
        records.append(record)
        im = Image.open(source).convert('RGB')
        draw = ImageDraw.Draw(im)
        for b in annotations:
            draw.rectangle(b['xyxy'], outline='#00ff70', width=6)
        im.thumbnail((1218, 924))
        review = out / 'review'
        review.mkdir(exist_ok=True)
        im.save(review / (stem + '.jpg'), quality=93)
        cards.append(f'<article><h2>{stem} · {split}</h2><a href="{stem}.jpg"><img loading="lazy" src="{stem}.jpg"></a></article>')
    synthetic = json.loads((args.synthetic / 'manifest.json').read_text())
    for original in synthetic['frames']:
        if original['split'] != 'train':
            continue
        r = dict(original, domain='synthetic')
        assert r['sha256'] not in seen
        seen.add(r['sha256'])
        for field in ('image', 'label'):
            source = (args.synthetic / r[field]).resolve()
            destination = out / r[field]
            destination.symlink_to(source)
        records.append(r)
    summary = dict(images=dict(Counter(r['split'] for r in records)),
                   domains=dict(Counter(r['domain'] for r in records)),
                   real_images=dict(Counter(r['split'] for r in real)),
                   instances={s: sum(len(r['boxes']) for r in records if r['split'] == s) for s in ('train', 'val', 'test')},
                   real_train_repeat=args.real_repeat,
                   synthetic_source=str(args.synthetic.resolve()),
                   annotation_method='Agent visual estimates from originals; all 96 boxes inspected in enlarged crops; not independently human-verified',
                   bbox_policy='Tight camera body including lens; exclude wristband, mount, cable; image-clipped',
                   evaluation_note='验证与测试均为真实帧。按视频时间块划分，块间留空至少 50 帧；同一视频且曾用于合成场景设计，不是独立拍摄盲测。标注为助手逐图视觉估计并复查，未经独立人工复核。',
                   selection_policy=selection['policy'])
    (out / 'manifest.json').write_text(json.dumps(dict(summary=summary, frames=records), indent=2, ensure_ascii=False))
    training = []
    for r in records:
        if r['split'] == 'train':
            training.extend([str(out / r['image'])] * (args.real_repeat if r['domain'] == 'real' else 1))
    (out / 'train.txt').write_text('\n'.join(training) + '\n')
    (out / 'data.yaml').write_text(f'path: {json.dumps(str(out))}\ntrain: train.txt\nval: images/val\ntest: images/test\nnames:\n  0: action4\n')
    (out / 'review/index.html').write_text('<!doctype html><meta charset="utf-8"><title>真实帧相机框标注</title><style>body{max-width:1250px;margin:30px auto;font:17px sans-serif;background:#14212a;color:#eee}img{max-width:100%}article{margin:30px 0}</style><h1>48 张真实帧 · 96 个相机框</h1><p>32 张训练 / 8 张验证 / 8 张测试。绿色框仅标相机机身及镜头，排除支架、腕带、线缆。</p><p>' + summary['evaluation_note'] + '</p>' + ''.join(cards))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
