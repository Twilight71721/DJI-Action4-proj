"""Check paired crop geometry on actual training images and render label overlays."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bop_bb8 import BOPInstanceDataset, BOPModel
from bop_bb8.visualize import EDGES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.dataset/'manifest.json').read_text())
    model = BOPModel.from_models_info(args.dataset/'models/models_info.json')
    kwargs = dict(scenes=manifest['splits']['train'], input_size=(256,256), heatmap_size=(64,64),
                  crop_mode='bbox', crop_padding=1.3, heatmap_kernel='gaussian')
    baseline = BOPInstanceDataset(args.dataset/'clean', model, **kwargs)
    jitter = BOPInstanceDataset(args.dataset/'clean', model, **kwargs,
             crop_jitter_shift=.05, crop_jitter_padding=(1.2,1.4), crop_jitter_seed=20260923)
    rows, changes, max_error, lost = [], 0, 0., 0
    selected = np.linspace(0, len(baseline)-1, 120, dtype=int)
    for position, idx in enumerate(selected):
        old = baseline[int(idx)]
        originals = old.keypoints_2d * (np.array(old.crop_xywh[2:])/256) + old.crop_xywh[:2]
        samples = [old]
        for epoch in [0,1,2]:
            jitter.set_epoch(epoch)
            new = jitter[int(idx)]
            expected = (originals-new.crop_xywh[:2]) * (256/np.array(new.crop_xywh[2:]))
            max_error = max(max_error, float(np.max(np.abs(expected-new.keypoints_2d))))
            lost += int((old.valid & ~new.valid).sum())
            changes += int(old.crop_xywh != new.crop_xywh)
            samples.append(new)
        if position < 12:
            images = []
            for label, sample in zip(['Original', 'Epoch 1', 'Epoch 2', 'Epoch 3'], samples):
                image = cv2.cvtColor(sample.image, cv2.COLOR_RGB2BGR)
                pts = np.rint(sample.keypoints_2d).astype(int)
                for a,b in EDGES:
                    if sample.valid[a] and sample.valid[b]:
                        cv2.line(image, tuple(pts[a]), tuple(pts[b]), (50,240,50), 1)
                for i, pt in enumerate(pts):
                    if sample.valid[i]:
                        cv2.circle(image, tuple(pt), 2, (0,255,255), -1)
                        cv2.putText(image, str(i), tuple(pt+[2,-2]), cv2.FONT_HERSHEY_SIMPLEX, .3, (0,255,255), 1)
                image = np.vstack([np.zeros((24,256,3),np.uint8), image])
                cv2.putText(image,label,(5,17),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
                images.append(image)
            name=f'preview_{position:02d}.jpg'
            cv2.imwrite(str(args.output/name), np.hstack(images))
            rows.append(f'<p>{old.scene_id}/{old.image_id}/{old.instance_idx}</p><img src="{name}">')
    result=dict(instances=120,epochs_checked=3,changed_crops=changes,comparison_count=360,
                max_coordinate_error_px=max_error,lost_previously_valid_corners=lost)
    if lost or max_error > .001 or changes < 300:
        raise RuntimeError(result)
    (args.output/'verification.json').write_text(json.dumps(result,indent=2))
    (args.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>裁剪扰动标签检查</title>'
        '<style>body{background:#14212a;color:#eee;max-width:1100px;margin:30px auto;font:17px system-ui}img{max-width:100%}</style>'
        '<h1>原始裁剪 / 三个训练轮次的扰动</h1><p>绿色为同步变换后的三维包围盒标签；不是网络预测。只改变裁剪，不向网络输入画线。</p>'
        + '<pre>'+json.dumps(result,indent=2)+'</pre>'+''.join(rows))
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
