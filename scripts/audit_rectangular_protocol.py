"""Audit rectangular inputs and select an output margin from training geometry only."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bop_bb8 import BOPInstanceDataset, BOPModel
from bop_bb8.cropping import rectangular_crop_window
from bop_bb8.geometry import crop_and_scale_intrinsics, parse_intrinsics, parse_rotation, project_corners


def scan_geometry(root, model, scenes, sizes, configured_margin, min_visibility=0.):
    dataset = BOPInstanceDataset(root, model, scenes=scenes, crop_mode='rect',
        crop_padding=1., crop_border=0, input_size=(256, 256), rect_sizes=sizes)
    margins, distortions, cases = [], [], []
    buckets = Counter()
    invalid, excluded = [], []
    for i, (scene, frame, instance) in enumerate(dataset.instances):
        gt, cameras, info = dataset._load_scene(scene)
        visibility = float(info[str(frame)][instance].get('visib_fract', 1.))
        if visibility < min_visibility:
            excluded.append(dict(identity=[scene, frame, instance], visib_fract=visibility,
                                 reason='training-only visibility threshold'))
            continue
        size = dataset.get_input_size(i)
        x, y, w, h = info[str(frame)][instance]['bbox_obj']
        crop = rectangular_crop_window((x, y, x+w, y+h), dataset._image_sizes[(scene, frame)])
        K = crop_and_scale_intrinsics(parse_intrinsics(cameras[str(frame)]['cam_K']), crop, size)
        entry = gt[str(frame)][instance]
        points, _, _ = project_corners(model.corners_mm(), K, parse_rotation(entry['cam_R_m2c']),
                                      entry['cam_t_m2c'], size)
        if not np.isfinite(points).all():
            invalid.append([scene, frame, instance])
            continue
        grid = points/4
        core = np.array(size)/4
        required = max(0, math.ceil(float(-grid.min())), math.floor(float((grid-core).max()))+1)
        margins.append(required)
        distortion = math.exp(abs(math.log((size[0]/size[1])/(crop[2]/crop[3]))))
        distortions.append(distortion)
        buckets[f'{size[0]}x{size[1]}'] += 1
        cases.append(dict(identity=[scene, frame, instance], required_margin=required,
                          crop_xywh=list(crop), input_size=list(size)))
    if not margins:
        raise ValueError('No finite projected instances in this split')
    return dict(instances=len(dataset), eligible_instances=len(margins), excluded_instances=excluded,
        behind_camera_or_nonfinite_instances=invalid,
        required_margin_max=max(margins), required_margin_p99=float(np.percentile(margins, 99)),
        configured_margin=configured_margin,
        all_eight_inside_fraction=float(np.mean(np.asarray(margins) <= configured_margin)),
        aspect_buckets=dict(buckets), max_aspect_distortion_factor=max(distortions),
        median_aspect_distortion_factor=float(np.median(distortions)),
        worst_margin_cases=sorted(cases, key=lambda r:r['required_margin'], reverse=True)[:8],
        instance_index_sha256=hashlib.sha256(json.dumps(dataset.instances).encode()).hexdigest())


def audit(prepared, recipe):
    prepared = Path(prepared)
    online = json.loads((prepared/'online_config.json').read_text())
    available = sorted(p.name for p in (prepared/'trainval').iterdir() if (p/'scene_gt.json').is_file())
    val = online['val_scenes']
    if not set(val).issubset(available):
        raise ValueError('Missing held-out scenes')
    train = [s for s in available if s not in set(val)]
    old_train = set(online['old_train_scenes'])
    if old_train & set(val):
        raise ValueError('Legacy training scenes overlap validation')
    # Pair IDs are generated together: both appearance variants must share a split.
    split_by_pair = {}
    for key, record in online['new_records'].items():
        split = 'val' if key.split('/')[0] in val else 'train'
        group = record.get('parent_id', record['pair_id'])
        if group in split_by_pair and split_by_pair[group] != split:
            raise ValueError('A generated foreground group crosses train/validation')
        split_by_pair[group] = split
    config = recipe['training']
    model = BOPModel.from_models_info(prepared/'models/models_info.json')
    stats = {name: scan_geometry(prepared/'trainval', model, scenes, config['rect_sizes'], config['heatmap_margin'],
                                config.get('train_min_visib_fract', 0.) if name == 'train' else 0.)
             for name, scenes in [('train', train), ('val', val)]}
    recommended = int(math.ceil((stats['train']['required_margin_max']+2)/4)*4)
    return dict(scope='Metadata projection check only; no inference accuracy claim',
        rgb_padding=False, source_box='BOP bbox_obj (oracle body localization); deployment uses frozen YOLO body boxes',
        configured_heatmap_margin=config['heatmap_margin'], recommended_heatmap_margin=recommended,
        margin_selection='training maximum + 2 cells; validation never chooses margin',
        stats=stats, train_scenes=train, val_scenes=val,
        prepared_config_sha256=hashlib.sha256((prepared/'online_config.json').read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/single_rgb_rect_20260926.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.prepared, json.loads(args.config.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['train_scenes', 'val_scenes']}, indent=2))


if __name__ == '__main__':
    main()
