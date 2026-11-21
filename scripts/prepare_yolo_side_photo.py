"""Add reviewed real training frames while freezing prior detector holdouts."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

import cv2
import numpy as np

from prepare_yolo_bop import normalize_box


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous',type=Path,required=True)
    parser.add_argument('--annotations',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();old=args.previous.resolve();out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    for split in ['train','val','test']:
        for kind in ['images','labels']:(out/kind/split).mkdir(parents=True)
    previous=json.loads((old/'manifest.json').read_text())
    annotations=json.loads(args.annotations.read_text())
    records=[];training=[];seen=set();frozen=[]
    for original in previous['frames']:
        record=dict(original)
        for field in ['image','label']:
            path=(old/record[field]).resolve()
            if not path.is_file():raise FileNotFoundError(path)
            (out/record[field]).symlink_to(path)
        seen.add(record['sha256'])
        repeat=40 if record['domain']=='real' else 1
        record['sampling_repeat']=repeat if record['split']=='train' else 1
        record['new_side_annotation']=False
        records.append(record)
        if record['split']=='train':training.extend([str(out/record['image'])]*repeat)
        else:
            for field in ['image','label']:
                frozen.append(dict(path=record[field],sha256=hashlib.sha256((out/record[field]).read_bytes()).hexdigest()))
    train_blocks={tuple(r['segment']) for r in records if r['domain']=='real' and r['split']=='train'}
    holdout=[r for r in records if r['split']!='train']
    (out/'review').mkdir()
    cards=[]
    for r in annotations['frames']:
        if not r.get('reviewed') or r['split']!='train':raise ValueError('Only reviewed training frames may be added')
        frame=r['frame'];segments=[s for s in train_blocks if s[0]<=frame<=s[1]]
        assert len(segments)==1
        assert all(abs(frame-v['frame'])>=51 for v in holdout)
        source=Path(r['source']);digest=hashlib.sha256(source.read_bytes()).hexdigest();assert digest not in seen;seen.add(digest)
        stem=f'real_side_{frame:06d}';image=out/f'images/train/{stem}.jpg';label=out/f'labels/train/{stem}.txt'
        shutil.copy2(source,image)
        rgb=cv2.imread(str(image));height,width=rgb.shape[:2];assert (width,height)==(r['width'],r['height'])
        boxes=[]
        for i,b in enumerate(r['boxes_xyxy']):
            x1,y1,x2,y2=b;assert 0<=x1<x2<=width and 0<=y1<y2<=height
            boxes.append(dict(instance=i,xyxy=b,xywhn=normalize_box(b,width,height)))
            cv2.rectangle(rgb,(x1,y1),(x2,y2),(0,255,80),5)
        assert len(boxes)==2
        label.write_text(''.join('0 '+' '.join(f'{v:.9f}' for v in b['xywhn'])+'\n' for b in boxes))
        cv2.imwrite(str(out/f'review/{stem}.jpg'),cv2.resize(rgb,(1218,924)))
        records.append(dict(frame=frame,split='train',segment=list(segments[0]),source=str(source),
            width=width,height=height,sha256=digest,domain='real',scene='real_video',
            group=f'real_{segments[0][0]}_{segments[0][1]}',boxes=boxes,image=str(image.relative_to(out)),
            label=str(label.relative_to(out)),invisible_instances=[],sampling_repeat=80,
            new_side_annotation=True,annotation_method=r['annotation_method']))
        training.extend([str(image)]*80)
        cards.append(f'<h2>新增训练帧 {frame}</h2><img loading="lazy" src="{stem}.jpg">')
    (out/'train.txt').write_text('\n'.join(training)+'\n')
    (out/'data.yaml').write_text(f'path: {json.dumps(str(out))}\ntrain: train.txt\nval: images/val\ntest: images/test\nnames:\n  0: action4\n')
    summary=dict(previous=previous['summary'],images=dict(Counter(r['split'] for r in records)),
        real_images=dict(Counter(r['split'] for r in records if r['domain']=='real')),
        training_entries=len(training),new_real_frames=len(annotations['frames']),
        new_real_boxes=sum(len(r['boxes_xyxy']) for r in annotations['frames']),
        evaluation_note=previous['summary']['evaluation_note'],
        annotation_method='Agent-reviewed proposals with manual corrections; not independently human verified.',
        photometric='Online balanced RGB augmentation on all training domains; no changes to validation/test RGB or boxes.',
        original_real_repeat=40,new_side_repeat=80,
        real_entry_fraction=sum(r['sampling_repeat'] for r in records if r['domain']=='real' and r['split']=='train')/len(training))
    (out/'manifest.json').write_text(json.dumps(dict(summary=summary,frames=records),ensure_ascii=False,indent=2))
    (out/'frozen_holdouts.json').write_text(json.dumps(frozen,indent=2))
    shutil.copy2(args.annotations,out/'new_annotations.json')
    sheets=[]
    for p in sorted(args.annotations.parent.glob('reviewed_*.jpg')):
        shutil.copy2(p,out/'review'/p.name);sheets.append(f'<img src="{p.name}">')
    (out/'review/index.html').write_text('<meta charset="utf-8"><style>body{background:#15232c;color:#eee;font:18px system-ui;max-width:1250px;margin:30px auto}img{max-width:100%}</style><h1>YOLO侧面真实帧补充标注</h1><p>22帧44框，来自既有训练时间块。助手逐框视觉复查，不是独立人工确认真值；未修改固定验证和测试集。绿色仅用于展示，不写入训练图。</p>'+''.join(sheets)+''.join(cards))
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
