"""Compare crop policies in one fixed coordinate system, using synthetic holdouts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bop_bb8 import BOPInstanceDataset, BOPModel
from bop_bb8.geometry import crop_and_scale_intrinsics, parse_intrinsics
from bop_bb8.torch_data import HeatmapDataset
from evaluate_cleanblur5000 import load_model, predict
from evaluate_heatmap_checkpoint import metric_summary, draw_overlay


def map_points(points, source_K, reference_K):
    homogeneous = np.c_[points, np.ones(len(points))]
    mapped = homogeneous @ (reference_K @ np.linalg.inv(source_K)).T
    return mapped[:, :2] / mapped[:, 2:]


class ReferenceDataset(HeatmapDataset):
    def __getitem__(self, index):
        batch = super().__getitem__(index)
        ds = self.dataset
        scene, frame, instance = ds.instances[index]
        _, cameras, info = ds._load_scene(scene)
        camera = cameras[str(frame)]
        if 'width' in camera and 'height' in camera:
            size = (camera['width'], camera['height'])
        else:
            candidates = [ds.root/scene/'rgb'/f'{frame:06d}{suffix}' for suffix in ('.png', '.jpg', '.jpeg')]
            with Image.open(next(p for p in candidates if p.is_file())) as source:
                size = source.size
        crop = ds._bbox_crop(info[str(frame)][instance]['bbox_obj'], size, 1.3)
        batch['reference_K'] = crop_and_scale_intrinsics(parse_intrinsics(camera['cam_K']), crop, (256, 256))
        return batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['dataset', 'baseline', 'checkpoint', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--prepared', type=Path)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--limit', type=int, default=0, help='Pipeline check only')
    args = parser.parse_args()
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    device = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifest = json.loads((args.dataset/'manifest.json').read_text())
    info = BOPModel.from_models_info(args.dataset/'models/models_info.json')
    specs = [(f'{split}_{variant}', args.dataset/variant, manifest['splits'][split], info)
             for split in ['val', 'test'] for variant in ['clean', 'blur']]
    if args.prepared:
        config = json.loads((args.prepared/'online_config.json').read_text())
        specs.append(('pose_holdout', args.prepared/'trainval', [config['new_val_scene']],
                      BOPModel.from_models_info(args.prepared/'models/models_info.json')))
    folders = {'review': args.output/'review'}
    if args.prepared:
        folders['pose_holdout'] = args.output/'pose_holdout'
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=False)
    protocol = dict(incomplete_diagnostic=bool(args.limit), models={}, metrics={},
                    coordinates='Predictions and targets mapped to the SAME old body-bbox x1.3 256px reference crop. Fixed valid-corner denominator.',
                    input_protocol='Each checkpoint uses its saved crop policy. Synthetic crops use GT boxes (oracle localization).',
                    caveat='New crop continuation changes crop policy and optimizer update count; not a matched-step crop-only ablation.',
                    real_evaluation='Not run here. Real inference requires an independently predicted cuboid bbox; no ground-truth box is presented as a detection.')
    reports = {key: dict(protocol, models={}, metrics={}) for key in folders}
    cards = {key: [] for key in folders}
    baseline_indices = {}
    for name, path in [('before', args.baseline), ('tight', args.checkpoint)]:
        model, metadata = load_model(path, device, allow_tight=True)
        saved = metadata['args']
        if name == 'tight' and saved['crop_mode'] != 'cuboid':
            raise ValueError('Expected a newly trained cuboid-crop checkpoint')
        for key in reports:
            reports[key]['models'][name] = metadata
            reports[key]['metrics'][name] = {}
        for case, root, scenes, model_info in specs:
            key = 'pose_holdout' if case == 'pose_holdout' else 'review'
            folder = folders[key]
            ds = BOPInstanceDataset(root, model_info, scenes=scenes, input_size=(256, 256),
                heatmap_size=(64, 64), crop_mode=saved['crop_mode'], crop_padding=saved['crop_padding'],
                crop_border=saved.get('crop_border', 4), heatmap_kernel='gaussian', heatmap_sigma=2.)
            if args.limit:
                ds.instances = ds.instances[:args.limit]
            if not len(ds):
                raise ValueError('Empty evaluation split: '+case)
            if name == 'before':
                baseline_indices[case] = ds.instances.copy()
            elif ds.instances != baseline_indices[case]:
                raise ValueError('Evaluation sample identities differ between models')
            if case == 'pose_holdout':
                expected = config['expected_new_val_instances']
                if not args.limit and len(ds) != expected:
                    raise ValueError('Incomplete held-out pose set')
                reports[key].update(expected_instances=expected, actual_instances=len(ds),
                                    holdout_role='Validation used for checkpoint selection; not an independent test')
            pred_all, target_all, valid_all, native_all = [], [], [], []
            for batch in DataLoader(ReferenceDataset(ds), batch_size=16, shuffle=False, num_workers=args.workers):
                pred, _ = predict(model, batch['image'], device)
                for j in range(len(pred)):
                    source_K = batch['K_input'][j].numpy()
                    reference_K = batch['reference_K'][j].numpy()
                    cp = map_points(pred[j], source_K, reference_K)
                    ct = map_points(batch['keypoints_2d'][j].numpy(), source_K, reference_K)
                    z = (model_info.corners_mm() @ batch['R_m2c'][j].numpy().T + batch['t_m2c'][j].numpy())[:, 2]
                    valid = (z > 1e-6) & np.isfinite(ct).all(1) & (ct >= 0).all(1) & (ct < 256).all(1)
                    pred_all.append(cp); target_all.append(ct); valid_all.append(valid); native_all.append(pred[j])
            pred, target, valid, native = map(np.asarray, (pred_all, target_all, valid_all, native_all))
            if not np.isfinite(pred).all():
                raise ValueError('Nonfinite predictions')
            if name == 'tight':
                with np.load(folder/f'before_{case}.npz') as old:
                    if not np.array_equal(valid, old['valid']) or not np.allclose(target, old['targets'], atol=1e-3):
                        raise ValueError('Reference targets or evaluation denominator changed')
            err = np.linalg.norm(pred-target, axis=-1)
            err[~valid] = np.nan
            stats = dict(metric_summary(err), instances=len(ds), excluded_corners=int((~valid).sum()))
            if case == 'pose_holdout' and not args.limit:
                if len(ds) % 8:
                    raise ValueError('Expected four paired clean/blur backgrounds per pose')
                for offset, variant in [(0, 'clean'), (1, 'blur')]:
                    points = pred[offset::2].reshape(-1, 4, 8, 2)
                    stats[variant] = dict(metric_summary(err[offset::2]),
                        mean_background_displacement_px=float(np.linalg.norm(points-points.mean(1, keepdims=True), axis=-1).mean()))
            reports[key]['metrics'][name][case] = stats
            np.savez_compressed(folder/f'{name}_{case}.npz', predictions=pred, targets=target,
                                valid=valid, errors=err, native_predictions=native, index=np.array(ds.instances))
            for i in np.linspace(0, len(ds)-1, min(6, len(ds)), dtype=int):
                sample = ds[int(i)]
                filename = f'{name}_{case}_{i:04d}.jpg'
                cv2.imwrite(str(folder/filename), draw_overlay(sample.image, sample.keypoints_2d,
                              native[i], sample.valid, f'{name} {case} {i}'))
                cards[key].append(f'<p>{name} / {case} / {i}</p><img src="{filename}">')
            print(name, case, stats['mean_err_px'], stats['pck5'], flush=True)
        del model
        torch.cuda.empty_cache()
    for key, folder in folders.items():
        report = reports[key]
        (folder/'summary.json').write_text(json.dumps(report, indent=2))
        rows = [f'<tr><td>{name}</td><td>{case}</td><td>{s["mean_err_px"]:.3f}</td><td>{s["pck5"]:.2%}</td></tr>'
                for name, cases in report['metrics'].items() for case, s in cases.items()]
        (folder/'index.html').write_text('<meta charset="utf-8"><title>紧八角框对比</title>'
            '<style>body{background:#14212a;color:#eee;font:17px/1.7 sans-serif;max-width:1100px;margin:30px auto}td{padding:10px}img{max-width:100%}</style>'
            '<h1>紧八角框训练评估</h1>' + ('<p>限量流程检查，不能作为实验结论。</p>' if args.limit else '') +
            '<p>各模型使用对应训练裁剪。误差统一换算到旧 1.3 倍框的 256px 坐标，评估样本与角点分母严格一致。'
            '可视化仍显示各自实际输入。合成图采用真值框，仅评价定位正确时的角点网络，不能代替真实 YOLO 端到端效果。'
            '新姿态集合为选权重使用的验证集。真实帧尚未在本评估中测试。</p>'
            '<table><tr><th>模型</th><th>集合</th><th>统一坐标均值px</th><th>PCK@5</th></tr>'+''.join(rows)+'</table>'+''.join(cards[key]))


if __name__ == '__main__':
    main()
