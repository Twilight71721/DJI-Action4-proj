"""Fixed-budget A/B evaluation: unchanged decoder, common-coordinate crop stress, real overlays."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bop_bb8 import BOPInstanceDataset, BOPModel
from bop_bb8.torch_data import HeatmapDataset
from evaluate_cleanblur5000 import load_model, predict
from evaluate_heatmap_checkpoint import metric_summary
from infer_resnet_yolo_real import prepare_crop, draw_points


class ShiftedDataset(BOPInstanceDataset):
    def __init__(self, *args, shift=0, **kwargs):
        self.shift = shift
        super().__init__(*args, **kwargs)

    def _bbox_crop(self, bbox, image_size, padding):
        x, y, w, h = bbox
        return BOPInstanceDataset._bbox_crop((x+self.shift*w, y, w, h), image_size, padding)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ['dataset', 'baseline', 'a', 'b', 'real-manifest', 'output']:
        p.add_argument('--'+flag, type=Path, required=True)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--limit', type=int, default=0, help='Pipeline check only, never a formal metric')
    args = p.parse_args()
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    data = args.dataset.resolve()
    manifest = json.loads((data/'manifest.json').read_text())
    info = BOPModel.from_models_info(data/'models/models_info.json')
    family_by_scene = {j['scene']: j['family'] for j in manifest['jobs']}
    cases = [('val_clean', 'val', 'clean', 1.3, 0), ('val_blur', 'val', 'blur', 1.3, 0),
             ('val_padding145', 'val', 'clean', 1.45, 0),
             ('val_shift_left5', 'val', 'clean', 1.3, -.05),
             ('val_shift_right5', 'val', 'clean', 1.3, .05),
             ('test_clean', 'test', 'clean', 1.3, 0), ('test_blur', 'test', 'blur', 1.3, 0)]
    records = json.loads(args.real_manifest.read_text())['records']
    records = [dict(r, image=str((args.real_manifest.parent/r['image']).resolve())) for r in records]
    extra = ROOT/'real/frame0/manifest.json'
    if extra.exists():
        records += [dict(r, image=str((extra.parent/r['image']).resolve()))
                    for r in json.loads(extra.read_text())['records']]
    real_crops = []
    for r in records:
        original = cv2.imread(r['image'])
        if original is None:
            raise FileNotFoundError(r['image'])
        roi, tensor, crop = prepare_crop(original, r['box'], (256, 256), 1.3)
        real_crops.append((roi, tensor))
        r.update(crop=list(crop), predictions={}, peaks={})
    summary = dict(incomplete_diagnostic=bool(args.limit), models={}, metrics={},
        protocol='Before / A last / B last; fixed 30-epoch continuation, no model selection on test or real frames.',
        coordinates='All synthetic errors mapped to the corresponding unperturbed padding1.3 256px crop.',
        real_protocol='Previously inspected diagnostic frames, no reliable eight-corner GT. Qualitative only, not blind tests.',
        caveat='Single paired seed; no statistical significance claim. best.pth is saved as secondary, not used here.')
    for name, path in [('before', args.baseline), ('A', args.a), ('B', args.b)]:
        model, metadata = load_model(path, torch.device('cuda'))
        summary['models'][name] = metadata
        summary['metrics'][name] = {}
        for case, split, variant, padding, shift in cases:
            ds = ShiftedDataset(data/variant, info, scenes=manifest['splits'][split],
                input_size=(256, 256), heatmap_size=(64, 64), crop_mode='bbox',
                crop_padding=padding, heatmap_kernel='gaussian', heatmap_sigma=2., shift=shift)
            if args.limit:
                ds.instances = ds.instances[:args.limit]
            loader = DataLoader(HeatmapDataset(ds), batch_size=16, shuffle=False, num_workers=args.workers)
            preds, targets, valid, peaks = [], [], [], []
            for batch in loader:
                pred, peak = predict(model, batch['image'], torch.device('cuda'))
                preds.append(pred); peaks.append(peak)
                targets.append(batch['keypoints_2d'].numpy()); valid.append(batch['valid'].numpy())
            preds, targets, valid, peaks = map(np.concatenate, (preds, targets, valid, peaks))
            common_preds, common_targets, common_valid = [], [], []
            for i, (scene, frame, instance) in enumerate(ds.instances):
                gt, cameras, gt_info = ds._load_scene(scene)
                box = gt_info[str(frame)][instance]['bbox_obj']
                cam = cameras[str(frame)]
                if 'width' in cam and 'height' in cam:
                    size = (cam['width'], cam['height'])
                else:
                    size = ds.crop[2:]
                ref = BOPInstanceDataset._bbox_crop(box, size, 1.3)
                changed = ds._bbox_crop(box, size, padding)
                scale = np.array(changed[2:])/256
                mapped_pred = (preds[i]*scale + changed[:2] - np.array(ref[:2])) * (256/np.array(ref[2:]))
                mapped_gt = (targets[i]*scale + changed[:2] - np.array(ref[:2])) * (256/np.array(ref[2:]))
                # The common valid set is invariant to perturbation; lost corners still count as failures.
                z = (info.corners_mm() @ np.array(gt[str(frame)][instance]['cam_R_m2c']).reshape(3,3).T
                     + np.array(gt[str(frame)][instance]['cam_t_m2c']))[:,2]
                keep = (z > 1e-6) & np.isfinite(mapped_gt).all(1) & (mapped_gt >= 0).all(1) & (mapped_gt < 256).all(1)
                common_preds.append(mapped_pred); common_targets.append(mapped_gt); common_valid.append(keep)
            cp, ct, cv = map(np.asarray, (common_preds, common_targets, common_valid))
            if not np.isfinite(cp).all():
                raise RuntimeError(f'Nonfinite predictions: {name}/{case}')
            errors = np.linalg.norm(cp-ct, axis=-1)
            errors[~cv] = np.nan
            families = np.array([family_by_scene[s] for s, _, _ in ds.instances])
            stats = dict(metric_summary(errors), instances=len(ds),
                lost_previously_valid_corners=int((cv & ~valid).sum()),
                per_family={str(f): metric_summary(errors[families == f]) for f in sorted(set(families))})
            summary['metrics'][name][case] = stats
            np.savez_compressed(out/f'{name}_{case}.npz', predictions=cp, targets=ct, valid=cv,
                errors=errors, raw_crop_predictions=preds, peaks=peaks, index=np.array(ds.instances), family=families)
            print(name, case, stats['mean_err_px'], stats['pck5'], flush=True)
        for start in range(0, len(real_crops), 16):
            pred, peak = predict(model, torch.stack([r[1] for r in real_crops[start:start+16]]), torch.device('cuda'))
            for offset, (points, scores) in enumerate(zip(pred, peak), start):
                records[offset]['predictions'][name] = points.tolist()
                records[offset]['peaks'][name] = scores.tolist()
        del model
        torch.cuda.empty_cache()
    rows, cards = [], []
    for name, sets in summary['metrics'].items():
        for case, s in sets.items():
            rows.append(f'<tr><td>{name}</td><td>{case}</td><td>{s["mean_err_px"]:.3f}</td>'
                        f'<td>{s["p95_err_px"]:.3f}</td><td>{s["pck5"]:.2%}</td>'
                        f'<td>{s["lost_previously_valid_corners"]}</td></tr>')
    for i, r in enumerate(records):
        roi = real_crops[i][0]
        panes = [roi] + [draw_points(roi, np.array(r['predictions'][name])) for name in ['before', 'A', 'B']]
        panel = np.vstack([np.zeros((26, 1024, 3), np.uint8), np.hstack(panes)])
        for j, label in enumerate(['Input', 'Before', 'A: continue', 'B: crop jitter']):
            cv2.putText(panel, label, (j*256+5, 18), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255), 1)
        filename = f'real_{i:03d}.jpg'
        cv2.imwrite(str(out/filename), panel)
        cards.append(f'<h3>frame {r["frame"]} / {r["method"]} / instance {r["instance"]}</h3><img src="{filename}">')
    (out/'summary.json').write_text(json.dumps(summary, indent=2))
    (out/'real_predictions.json').write_text(json.dumps(dict(records=records), indent=2))
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>裁剪扰动 A/B 对照</title>'
        '<style>body{background:#14212a;color:#eee;font:17px/1.6 system-ui;max-width:1250px;margin:30px auto}'
        'td,th{padding:8px;border:1px solid #888}img{max-width:100%}</style><h1>裁剪扰动 A/B 对照</h1>'
        + ('<p>限量管线检查，不是正式实验结果。</p>' if args.limit else '')
        + '<p>Before 为5000张训练后的起点；A为继续训练；B仅增加裁剪扰动。两组主比较使用同训练步数的 last.pth。'
        '误差统一到原始 padding=1.3 的256像素裁剪；超出扰动裁剪的原有效角点也保留在评估分母中。</p>'
        '<p>真实帧没有可靠角点真值，仅作定性诊断；预测变动或框看起来规整不等于更准确。</p>'
        '<table><tr><th>模型</th><th>集合</th><th>均值px</th><th>P95px</th><th>PCK@5</th><th>裁掉原有效角点</th></tr>'
        + ''.join(rows) + '</table>' + ''.join(cards))
    print(out/'index.html')


if __name__ == '__main__':
    main()
