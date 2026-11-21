"""Build fixed clean/blur pairs, preserving previous split membership."""
import argparse
import concurrent.futures
import collections
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import sys

import cv2
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bop_bb8 import BOPInstanceDataset


def generate(task):
    cv2.setNumThreads(1)
    source, clean, blurred, record, boxes = task
    image = cv2.imread(str(source))
    if image is None:
        raise ValueError(f'Unreadable image: {source}')
    sides = [BOPInstanceDataset._bbox_crop(b, (image.shape[1], image.shape[0]), 1.3)[2]
             for b in boxes]
    # Convert the proposed 256-input sigma to native pixels. Use the smaller
    # crop to avoid over-blurring a smaller instance sharing the same frame.
    sigma = max(0.3, record['sigma_input_target'] * min(sides) / 256)
    kernel = 2 * math.ceil(3 * sigma) + 1
    degraded = cv2.GaussianBlur(image, (kernel, kernel), sigma, borderType=cv2.BORDER_REFLECT_101)
    if np.array_equal(image, degraded):
        raise ValueError(f'Blur did not change any pixels: {source}')
    shutil.copy2(source, clean)
    if not cv2.imwrite(str(blurred), degraded, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        raise IOError(blurred)
    record.update(native_sigma=sigma, kernel=kernel, crop_sides=sides,
                  effective_input_sigma=[sigma * 256 / s for s in sides],
                  clean_sha256=hashlib.sha256(clean.read_bytes()).hexdigest(),
                  blur_sha256=hashlib.sha256(blurred.read_bytes()).hexdigest())
    return record


def select_groups(jobs, val_target, test_target, rng):
    order=list(jobs);rng.shuffle(order)
    states={(0,0):([],[])}
    for j in order:
        for (v,t),(vs,ts) in list(states.items()):
            w=j['frames']
            if v+w<=val_target:
                states.setdefault((v+w,t),(vs+[j['scene']],ts))
            if t+w<=test_target:
                states.setdefault((v,t+w),(vs,ts+[j['scene']]))
    if (val_target,test_target) not in states:
        raise ValueError(f'Cannot allocate whole jobs: {val_target}, {test_target}')
    return states[val_target,test_target]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'dataset/action4-reference-5000-20260921')
    parser.add_argument('--previous',type=Path,default=ROOT/'dataset/action4-resnet3305-stratified-20260922/manifest.json')
    parser.add_argument('--output',type=Path,default=ROOT/'dataset/action4-cleanblur5000-20260923')
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args();src=args.source;out=args.output
    if out.exists():raise FileExistsError(out)
    old=json.loads(args.previous.read_text());previous={j['scene']:j['split'] for j in old['jobs']}
    old_hashes={(f['scene'],f['frame']):f['sha256'] for f in old['frames']}
    jobs=[];rng=random.Random(20260923)
    for path in sorted((src/'completed').glob('*.json')):
        c=json.loads(path.read_text());j=json.loads((src/f"jobs/{c['id']:04d}.json").read_text())
        audit=json.loads((src/c['audit_directory']/'validation.json').read_text())
        assert not audit['errors'] and audit['geometry_validation_passed']
        jobs.append(dict(scene=f"{c['id']:06d}",family=j['index'],frames=c['frames'],
                         action=j['action'],reference_frame=j['reference_frame'],
                         split=previous.get(f"{c['id']:06d}")))
    assert sum(j['frames'] for j in jobs)==5000
    # Existing rare-family allocations and whole-batch grouping give unequal
    # small-family proportions. Retain these and balance the global 4000/500/500.
    targets={15:(50,40),16:(50,50),17:(50,50),18:(25,20),19:(25,25)}
    for family in sorted({j['family'] for j in jobs}):
        group=[j for j in jobs if j['family']==family];pending=[j for j in group if j['split'] is None]
        if not pending:continue
        v,t=targets[family]
        v-=sum(j['frames'] for j in group if j['split']=='val')
        t-=sum(j['frames'] for j in group if j['split']=='test')
        val,test=select_groups(pending,v,t,rng)
        for j in pending:j['split']='val' if j['scene'] in val else 'test' if j['scene'] in test else 'train'
    counts=collections.Counter()
    for j in jobs:counts[j['split']]+=j['frames']
    assert dict(counts)==dict(train=4000,val=500,test=500),counts
    out.mkdir(parents=True)
    shutil.copytree(src/'models',out/'models',ignore=shutil.ignore_patterns('*.ply','*.obj','*.glb','*.png','*.jpg'))
    splits={s:[j['scene'] for j in jobs if j['split']==s] for s in ['train','val','test']}
    family_stats={};views={s:collections.Counter() for s in splits};tasks=[]
    for j in jobs:
        scene=j['scene'];source=src/'train_pbr'/scene
        gt=json.loads((source/'scene_gt_info.json').read_text())
        images=sorted((source/'rgb').glob('*.png'));assert len(images)==j['frames']
        for variant in ['clean','blur']:
            dest=out/variant/scene;(dest/'rgb').mkdir(parents=True)
            for name in ['scene_gt.json','scene_gt_info.json','scene_camera.json']:
                shutil.copy2(source/name,dest/name)
        meta=json.loads((src/f"render_metadata/batch_{int(scene):03d}/metadata.json").read_text())
        for f in meta['frames']:
            for inst in f['instances']:views[j['split']][inst['dominant_view']]+=1
        family_stats.setdefault(str(j['family']),collections.Counter())[j['split']]+=j['frames']
        for p in images:
            rec=dict(scene=scene,frame=str(int(p.stem)),family=j['family'],split=j['split'],
                     clean=f'clean/{scene}/rgb/{p.name}',blur=f'blur/{scene}/rgb/{p.name}')
            tasks.append((p,out/rec['clean'],out/rec['blur'],rec,[x['bbox_obj'] for x in gt[rec['frame']]]))
    # Exactly 80% mild / 20% moderate within each family and split; the clean
    # vs blur ratio is independently fixed at 1:1 through two saved variants.
    buckets=collections.defaultdict(list)
    for task in tasks:buckets[(task[3]['family'],task[3]['split'])].append(task[3])
    for key,records in sorted(buckets.items()):
        rng.shuffle(records);n=round(len(records)*0.2)
        for k,r in enumerate(records):
            r['blur_level']='moderate' if k<n else 'mild'
            r['sigma_input_target']=rng.uniform(1.2,1.6) if k<n else rng.uniform(0.4,1.0)
    frames=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for n,rec in enumerate(pool.map(generate,tasks,chunksize=5),1):
            key=(rec['scene'],rec['frame'])
            if key in old_hashes:assert rec['clean_sha256']==old_hashes[key],key
            frames.append(rec)
            if n%250==0:print(f'Generated {n}/5000 pairs',flush=True)
    assert len({f['clean_sha256'] for f in frames})==5000
    assert len({f['blur_sha256'] for f in frames})==5000
    assert not({f['clean_sha256'] for f in frames}&{f['blur_sha256'] for f in frames})
    # Training sees both variants; validation used for checkpoint selection is
    # clean only. Fixed blurred val/test are evaluated separately after training.
    (out/'trainval').mkdir()
    for j in jobs:
        scene=j['scene']
        if j['split'] in ['train','val']:(out/'trainval'/scene).symlink_to('../clean/'+scene)
        if j['split']=='train':(out/'trainval'/f'{100000+int(scene):06d}').symlink_to('../blur/'+scene)
    manifest=dict(version=1,seed=20260923,counts=dict(counts),base_images=5000,saved_images=10000,
                  instances_per_image=2,train_images_with_variants=8000,train_instances=16000,
                  splits=splits,old_test_scenes=old['splits']['test'],jobs=jobs,frames=frames,
                  family_counts=family_stats,view_counts=views,previous_membership_preserved=True,
                  blur_policy=dict(mild_fraction=.8,mild_sigma=[.4,1.0],moderate_sigma=[1.2,1.6],
                                   native_sigma_floor=.3,conversion='sigma_native=max(0.3,target_sigma*min(padded_crop_sides)/256)',
                                   padding=1.3,border='BORDER_REFLECT_101',native_kernel='2*ceil(3*sigma_native)+1',
                                   limitation='Equivalent input sigma is approximate: crop, resizing and native sampling affect blur. No exact camera PSF match is claimed.'),
                  selection='Clean validation only; blurred validation and both test variants evaluated separately',
                  protocol='Whole render jobs grouped; previous 3305 membership frozen. Shared reference pose families across splits, not unseen-action generalization.')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(dict(counts=counts,views=views,families=family_stats),indent=2))


if __name__=='__main__':main()
