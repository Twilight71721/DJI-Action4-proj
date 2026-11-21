"""Continuous SO(3) sampling with independent appearance and lighting streams."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bop_bb8 import BOPModel
from bop_bb8.geometry import project_corners
from build_multiview_background_bank import audit
from server_background_assets import sha, dump


def sample_records(anchors, center, counts=(4096, 256), seed=20260927,
                   appearance_seed=1, light_seed=2):
    records = []
    for split_index, (split, count) in enumerate(zip(['train', 'val'], counts)):
        pose_rng = np.random.default_rng([seed, split_index, 0])
        appearance_rng = np.random.default_rng([seed, split_index, appearance_seed])
        light_rng = np.random.default_rng([seed, split_index, light_seed])
        for i in range(count):
            if pose_rng.random() < .5:
                R = Rotation.random(random_state=pose_rng).as_matrix()
                family, anchor = 'uniform_so3', None
            else:
                anchor = int(pose_rng.integers(len(anchors)))
                delta = Rotation.from_rotvec(np.deg2rad(pose_rng.normal(0, 18, 3)))
                R = delta.as_matrix() @ np.asarray(anchors[anchor]['R'])
                family = 'video_neighborhood'
            side = ['left', 'right'][appearance_rng.integers(2)]
            records.append(dict(
                id=f'continuous_{split}_{i:05d}', split=split, kind='foreground',
                family=family, anchor_index=anchor, source_group=f'continuous_{split}_{i:05d}',
                appearance=side, R=R.tolist(), t_mm=(np.array([0, 0, 205.])-R@center).tolist(),
                body_roughness=float(appearance_rng.uniform(.45, .8)),
                body_specular=float(appearance_rng.uniform(.12, .30)),
                light_location=light_rng.uniform([-.35, -.35, -.15], [.35, .35, .02]).tolist(),
                light_size=float(light_rng.uniform(.25, .65)),
                light_color=light_rng.uniform(.85, 1., 3).tolist(),
                light_energy=float(light_rng.uniform(6, 18)),
                exposure=float(light_rng.uniform(-.65, .0)),
                real_pose_ground_truth=False))
    return records


def prepare(out, anchor_path, models, glb, train_count, val_count, seed):
    model = BOPModel.from_models_info(models/'models_info.json')
    # These are orientation anchors only. Their original appearance/background is discarded.
    anchors = json.loads(anchor_path.read_text())['samples']
    anchors = [r for r in anchors if r.get('usage') == 'foreground_candidate_for_server_compositing']
    center = (model.min_xyz_mm+model.max_xyz_mm)/2
    rows = sample_records(anchors, center, (train_count, val_count), seed)
    K = [[700, 0, 256], [0, 700, 256], [0, 0, 1]]
    for r in rows:
        _, valid, _ = project_corners(model.corners_mm(), K, r['R'], r['t_mm'], image_size=(512, 512))
        if not valid.all():
            raise ValueError('Projected corner outside image: '+r['id'])
    out.mkdir(parents=True, exist_ok=False)
    shutil.copytree(models, out/'models')
    groups = []
    for side in ['left', 'right']:
        jobs = [r for r in rows if r['appearance'] == side]
        for start in range(0, len(jobs), 24):
            groups.append(dict(id=f'continuous_{side}_{start//24:03d}', side=side, K=K, records=jobs[start:start+24]))
    dump(out/'plan.json', dict(width=512, height=512, glb=str(glb), groups=groups, anchors=[], seed=seed,
         train_count=train_count, val_count=val_count, anchor_sha256=sha(anchor_path),
         sampling='50% uniform SO3, 50% independent continuous perturbations around video orientation anchors; appearance/light independent RNG streams',
         status='Finite samples from continuous distributions; not every possible orientation; real anchors are uncalibrated estimates'))
    # Statistical audit describes realized sampling; stream independence is tested separately.
    stats = {}
    for split in ['train', 'val']:
        rs = [r for r in rows if r['split'] == split]
        stats[split] = dict(count=len(rs), appearances={s:sum(r['appearance']==s for r in rs) for s in ['left','right']},
            families={f:sum(r['family']==f for r in rs) for f in ['uniform_so3','video_neighborhood']})
    dump(out/'sampling_audit.json', stats)
    print('PREPARED', stats, flush=True)


def render(out):
    plan = json.loads((out/'plan.json').read_text())
    for g in plan['groups']:
        target = out/'renders'/g['id']
        if (target/'metadata.json').exists():
            continue
        if target.exists():
            # Keep interrupted work for inspection; never overwrite it as a completed group.
            i = 0
            while target.with_name(target.name+f'.interrupted_{i}').exists():
                i += 1
            target.rename(target.with_name(target.name+f'.interrupted_{i}'))
        with (out/(g['id']+'.log')).open('w') as log:
            subprocess.run([sys.executable, str(ROOT/'scripts/render_pose_replacement_pilot.py'),
                'render', '--output', str(out), '--group', g['id'], '--samples', '16'],
                stdout=log, stderr=subprocess.STDOUT, check=True)
        print('RENDERED', g['id'], flush=True)
    audit(out)
    dump(out/'SHA256SUMS.json', {str(p.relative_to(out)):sha(p) for p in
         [out/'manifest.json', *(out/'models').glob('*.json'), *[out/r[k] for r in json.loads((out/'manifest.json').read_text())['samples'] for k in ['rgba','mask']]]})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--action', choices=['prepare','render','audit'], required=True)
    p.add_argument('--anchors', type=Path, default=ROOT/'exports/full_video_pose_assets_20260924/manifest.json')
    p.add_argument('--models', type=Path, default=ROOT/'dataset/action4-cleanblur5000-20260923/models')
    p.add_argument('--glb', type=Path, default=ROOT/'dataset/action4-v6-domain-pilot-200/models/obj_000001.glb')
    p.add_argument('--train-count', type=int, default=4096)
    p.add_argument('--val-count', type=int, default=256)
    p.add_argument('--seed', type=int, default=20260927)
    a = p.parse_args()
    if a.action == 'prepare':
        if min(a.train_count, a.val_count) < 1:
            p.error('positive train and val counts required')
        prepare(a.output.resolve(), a.anchors, a.models, a.glb, a.train_count, a.val_count, a.seed)
    elif a.action == 'render':
        render(a.output.resolve())
    else:
        audit(a.output.resolve())
