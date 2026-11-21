"""Evaluate the rectangular model on an explicit held-out BOP scene list."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bop_bb8 import BOPInstanceDataset, BOPModel
from bop_bb8.net import ResNetFPN, decode_heatmaps, heatmap_to_input_points
from bop_bb8.torch_data import HeatmapDataset, ShapeBucketBatchSampler
from bop_bb8.pnp import fit_perspective_cuboid
from evaluate_heatmap_checkpoint import metric_summary
from bop_bb8.visualize import EDGES


def write_preview(dataset, predictions, targets, normalized_errors, pnp_arrays, output):
    """Show fixed random cases and worst errors, in original-image coordinates."""
    import cv2
    rng = np.random.default_rng(20260926)
    selected = set(rng.choice(len(dataset), min(8, len(dataset)), replace=False).tolist())
    selected.update(np.argsort(normalized_errors.mean(1))[-8:].tolist())
    cards = []
    for i in sorted(selected):
        sample = dataset[i]
        image = cv2.imread(sample.image_path)
        x, y, w, h = sample.crop_xywh
        # Display a context crop only; the separately saved input is exactly the network RGB.
        pad = int(max(w,h)*.35)+2
        left, top = max(0,x-pad), max(0,y-pad)
        right, bottom = min(image.shape[1],x+w+pad), min(image.shape[0],y+h+pad)
        panels = []
        points_list = [predictions[i]]
        if pnp_arrays and pnp_arrays['pnp_accepted'][i]:
            points_list.append(pnp_arrays['pnp_points'][i])
        for points in points_list:
            panel = image[top:bottom,left:right].copy()
            for pts,color in [(targets[i],(40,220,40)), (points,(220,40,220))]:
                pts = np.rint(pts-[left,top]).astype(int)
                for a,b in EDGES:
                    cv2.line(panel,tuple(pts[a]),tuple(pts[b]),color,2,cv2.LINE_AA)
                for number,xy in enumerate(pts):
                    cv2.circle(panel,tuple(xy),3,color,-1)
                    cv2.putText(panel,str(number),tuple(xy+[3,-3]),cv2.FONT_HERSHEY_SIMPLEX,.4,color,1)
            panels.append(panel)
        filename = f'case_{i:06d}.jpg'
        inputname = f'input_{i:06d}.png'
        cv2.imwrite(str(output/filename),np.hstack(panels))
        cv2.imwrite(str(output/inputname),cv2.cvtColor(sample.image,cv2.COLOR_RGB2BGR))
        status = ('接受' if pnp_arrays['pnp_accepted'][i] else '拒绝') if pnp_arrays else '未运行'
        cards.append(f'<h3>{html.escape(str(dataset.instances[i]))} · 原始误差 '
            f'{normalized_errors[i].mean():.2f}px · PnP {status}</h3>'
            f'<p><a href="{inputname}">查看实际无补边输入</a></p><img src="{filename}">')
    (output/'index.html').write_text('<meta charset="utf-8"><style>body{font:18px system-ui}img{max-width:100%}</style>'
        '<h1>合成集预测检查</h1><p>固定随机 8 项＋误差最大 8 项，重复项合并。绿色：真值；紫色：预测。'
        '左：原始八点；右：PnP（仅在接受时展示）。此图额外显示上下文，实际网络输入见链接。'
        '误差按原始机身框长边归一到 256 像素。</p>'+''.join(cards))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ['checkpoint', 'dataset-root', 'models-info', 'output']:
        parser.add_argument('--'+flag, type=Path, required=True)
    parser.add_argument('--scenes', nargs='+', required=True)
    parser.add_argument('--role', choices=['validation', 'test', 'diagnostic'], required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--pnp', action='store_true', help='Compare optional PnP using known synthetic intrinsics')
    parser.add_argument('--pnp-threshold', type=float, default=5., help='RANSAC tolerance at body long side 256 px')
    args = parser.parse_args()
    if args.limit and args.role != 'diagnostic':
        raise ValueError('A limited evaluation must be labeled diagnostic')
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    saved = checkpoint['args']
    if saved.get('input_contract_version') != 2 or saved['crop_mode'] != 'rect':
        raise ValueError('Expected a rectangular protocol-v2 checkpoint')
    if args.role == 'validation' and set(args.scenes) != set(saved['val_scenes']):
        raise ValueError('Use all saved validation scenes, or label this as a diagnostic')
    if args.role == 'test' and set(args.scenes) & set(saved['val_scenes']):
        raise ValueError('Validation scenes cannot be reported as independent test scenes')
    hashes = {s:hashlib.sha256((args.dataset_root/s/'scene_gt.json').read_bytes()).hexdigest()
              for s in args.scenes}
    source_hashes = saved.get('scene_gt_sha256', {})
    if args.role == 'test':
        if not source_hashes:
            raise ValueError('Checkpoint lacks split fingerprints; use diagnostic role')
        if set(hashes.values()) & set(source_hashes.values()):
            raise ValueError('Test scene annotations duplicate training or validation scenes')
    if args.role == 'validation' and any(source_hashes.get(s) != hashes[s] for s in args.scenes):
        raise ValueError('Validation annotation fingerprints differ from training-time validation')
    margin = int(saved['heatmap_margin'])
    size = tuple(saved['input_size'])
    model_info = BOPModel.from_models_info(args.models_info, saved['obj_id'], saved['model_scale'])
    dataset = BOPInstanceDataset(args.dataset_root, model_info,
        scenes=args.scenes, crop_mode='rect', crop_padding=1., crop_border=0,
        input_size=size, heatmap_size=(size[1]//4, size[0]//4), rect_sizes=saved['rect_sizes'],
        heatmap_margin=margin, heatmap_kernel=saved['heatmap_kernel'], heatmap_sigma=saved['heatmap_sigma'],
        pose_scale=saved['pose_scale'])
    if args.limit:
        dataset.instances = dataset.instances[:args.limit]
    if not len(dataset):
        raise ValueError('Empty evaluation split')
    wrapped = HeatmapDataset(dataset)
    loader = DataLoader(wrapped, batch_sampler=ShapeBucketBatchSampler(wrapped, args.batch_size, False),
                        num_workers=args.workers)
    model = ResNetFPN(saved['backbone'], saved['fpn_dim'], pretrained=False,
                      heatmap_margin=margin).to(args.device).eval()
    model.load_state_dict(checkpoint['model'], strict=True)
    predictions, truths = np.empty((len(dataset), 8, 2)), np.empty((len(dataset), 8, 2))
    canvas_valid = np.empty((len(dataset), 8), dtype=bool)
    scale_lengths = np.empty(len(dataset))
    intrinsics = np.empty((len(dataset), 3, 3))
    with torch.inference_mode():
        for batch in loader:
            corners = torch.as_tensor(model_info.corners_mm(), dtype=batch['R_m2c'].dtype)
            camera_points = torch.einsum('bij,kj->bki', batch['R_m2c'], corners) + batch['t_m2c'][:, None, :]
            if not torch.isfinite(camera_points).all() or (camera_points[..., 2] <= 0).any():
                raise ValueError('Non-finite or behind-camera target; refusing invalid geometry')
            images = batch['image'].to(args.device)
            logits = model(images).float()
            points = heatmap_to_input_points(decode_heatmaps(logits, saved['decoder'], saved['decode_radius']),
                tuple(logits.shape[-2:]), tuple(images.shape[-2:]), margin=margin).cpu().numpy()
            crop = batch['crop_xywh'].numpy()
            scale = crop[:, 2:] / np.array([images.shape[-1], images.shape[-2]])
            index = batch['index'].numpy()
            predictions[index] = points*scale[:, None, :] + crop[:, None, :2]
            truths[index] = batch['keypoints_2d'].numpy()*scale[:, None, :] + crop[:, None, :2]
            canvas_valid[index] = batch['valid'].numpy()
            scale_lengths[index] = crop[:, 2:].max(1)
            K_original = batch['K_input'].numpy().astype(float)
            K_original[:, :2, :] *= scale[:, :, None]
            K_original[:, :2, 2] += crop[:, :2]
            intrinsics[index] = K_original
    errors = np.linalg.norm(predictions-truths, axis=-1)
    normalized = errors * (256/scale_lengths[:, None])
    report = dict(role=args.role, limited_diagnostic=bool(args.limit), instances=len(dataset),
        localization='oracle synthetic body boxes; not detector end-to-end accuracy',
        input='RGB rectangle only; no RGB padding; no references',
        error_units='Original-image px and body-long-side normalized to 256 px; not interchangeable with prior crop-pixel PCK',
        original_pixels=metric_summary(errors), body_long_side_256=metric_summary(normalized),
        denominator_corners=int(errors.size), outside_prediction_canvas=int((~canvas_valid).sum()),
        all_finite=bool(np.isfinite(errors).all()),
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        scene_gt_sha256=hashes,
        generalization_caveat='Scene names alone do not prove independent sources; audit render/video parent groups.')
    # Retain every held-out point in the denominator, including out-of-canvas failures.
    if not np.isfinite(errors).all():
        raise ValueError('Non-finite evaluation geometry; refusing a finite-only success metric')
    pnp_records, pnp_arrays = [], {}
    if args.pnp:
        cv2_seed = 20260926
        import cv2
        cv2.setRNGSeed(cv2_seed)
        projected = np.full_like(predictions, np.nan)
        accepted = np.zeros(len(dataset), dtype=bool)
        for i in range(len(dataset)):
            fit = fit_perspective_cuboid(model_info.corners_mm(), predictions[i], intrinsics[i],
                reprojection_threshold=args.pnp_threshold * scale_lengths[i]/256)
            pnp_records.append(dict(identity=dataset.instances[i], **fit))
            if fit['accepted']:
                accepted[i] = True
                projected[i] = fit['points']
        pnp_errors = np.full_like(errors, np.inf)
        pnp_errors[accepted] = np.linalg.norm(projected[accepted]-truths[accepted], axis=-1)
        pnp_normalized = pnp_errors * (256/scale_lengths[:, None])
        fallback_errors = np.where(accepted[:, None], pnp_errors, errors)
        report['pnp'] = dict(camera_model='Known synthetic K, zero rendering distortion; no GT corner input to fitting',
            rng_seed=cv2_seed, threshold_body_long_side_256=args.pnp_threshold,
            accepted_instances=int(accepted.sum()), total_instances=len(dataset),
            statuses=dict(Counter(r['status'] for r in pnp_records)),
            accepted_only_body_long_side_256=metric_summary(pnp_normalized[accepted]),
            pck5_all_corners_rejection_counts_as_failure=float((pnp_normalized <= 5).mean()),
            raw_fallback_body_long_side_256=metric_summary(fallback_errors * (256/scale_lengths[:, None])),
            note='Acceptance is geometric consistency, not correctness. Raw predictions remain unchanged.')
        pnp_arrays = dict(pnp_points=projected, pnp_accepted=accepted, pnp_original_errors=pnp_errors)
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output/'predictions.npz', predictions=predictions, targets=truths,
        original_errors=errors, normalized_errors=normalized, canvas_valid=canvas_valid,
        index=np.asarray(dataset.instances), K_original=intrinsics, **pnp_arrays)
    if args.pnp:
        (args.output/'pnp_fits.json').write_text(json.dumps(pnp_records, indent=2)+'\n')
    (args.output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    write_preview(dataset, predictions, truths, normalized, pnp_arrays, args.output)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
