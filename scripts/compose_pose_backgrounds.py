"""Server-side RGB-only BOP composition from the local pilot's compact assets.

This is a data composer, not a training launcher. All generated scenes are train-only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import cv2
import numpy as np


def bbox(mask):
    y,x=np.where(mask>0)
    return [int(x.min()),int(y.min()),int(x.max()-x.min()+1),int(y.max()-y.min()+1)] if len(x) else [-1,-1,-1,-1]


def background_patch(image,size,rng):
    h,w=image.shape[:2]
    side=max(2,int(min(h,w)*rng.uniform(.6,1.)))
    x=int(rng.integers(w-side+1));y=int(rng.integers(h-side+1))
    patch=image[y:y+side,x:x+side]
    if rng.random()<.5:patch=patch[:,::-1]
    return cv2.resize(patch,(size,size),interpolation=cv2.INTER_AREA)


def compose(assets,backgrounds,out,count,seed):
    if count<1:raise ValueError('count must be positive')
    manifest=json.loads((assets/'manifest.json').read_text())
    candidates=[r for r in manifest['samples'] if r['usage']=='replacement_pilot_visual_checked']
    foregrounds=[r for r in manifest['samples'] if r['usage']=='foreground_candidate_for_server_compositing']
    if not foregrounds:raise ValueError('No foregrounds in manifest')
    files=sorted(p for p in backgrounds.rglob('*') if p.suffix.lower() in {'.jpg','.jpeg','.png'} and p.is_file())
    if not files:raise ValueError('No server background images found')
    out.mkdir(parents=True,exist_ok=False)
    shutil.copytree(assets/'models',out/'models')
    rng=np.random.default_rng(seed);scenes={};records=[]
    for i in range(count):
        # Real-frame replacements preserve their real context; they are not cut
        # out again and pasted on another image.
        use_real=bool(candidates) and rng.random()<.2
        pool=candidates if use_real else foregrounds
        r=pool[int(rng.integers(len(pool)))];mask=cv2.imread(str(assets/r['mask']),0)
        source_bg=None
        if use_real:
            image=cv2.imread(str(assets/r['composite']))
            visible=cv2.imread(str((assets/r['composite']).parent/'mask_visib.png'),0)
        else:
            rgba=cv2.imread(str(assets/r['rgba']),cv2.IMREAD_UNCHANGED)
            source_bg=files[int(rng.integers(len(files)))];raw=cv2.imread(str(source_bg))
            if raw is None:raise ValueError(f'Unreadable background: {source_bg}')
            bg=background_patch(raw,512,rng)
            alpha=rgba[:,:,3:].astype(np.float32)/255
            image=np.rint(rgba[:,:,:3]*alpha+bg*(1-alpha)).clip(0,255).astype(np.uint8)
            visible=mask.copy()
        sigma256=0. if rng.random()<.5 else float(rng.uniform(.3,1.4))
        if sigma256:
            sigma=sigma256*2;kernel=2*int(np.ceil(3*sigma))+1
            image=cv2.GaussianBlur(image,(kernel,kernel),sigma)
        sid=f'{i//1000:06d}';iid=i%1000;scene=out/'train_pbr'/sid
        if sid not in scenes:
            for folder in ['rgb','mask','mask_visib']:(scene/folder).mkdir(parents=True,exist_ok=True)
            scenes[sid]=dict(gt={},camera={},info={})
        cv2.imwrite(str(scene/'rgb'/f'{iid:06d}.png'),image)
        cv2.imwrite(str(scene/'mask'/f'{iid:06d}_000000.png'),mask)
        cv2.imwrite(str(scene/'mask_visib'/f'{iid:06d}_000000.png'),visible)
        g=scenes[sid];key=str(iid)
        g['gt'][key]=[dict(obj_id=1,cam_R_m2c=np.array(r['R']).reshape(-1).tolist(),cam_t_m2c=r['t_mm'])]
        g['camera'][key]=dict(cam_K=np.array(r['K']).reshape(-1).tolist(),width=512,height=512)
        g['info'][key]=[dict(bbox_obj=bbox(mask),bbox_visib=bbox(visible),px_count_all=int((mask>0).sum()),
                            px_count_valid=int((mask>0).sum()),px_count_visib=int((visible>0).sum()),
                            visib_fract=float((visible>0).sum()/max(1,(mask>0).sum())))]
        records.append(dict(scene=sid,image_id=iid,parent_id=r['id'],source_group=r['source_group'],
            real_context=use_real,background=None if source_bg is None else str(source_bg.relative_to(backgrounds)),
            gaussian_sigma_at_256=sigma256,split='train'))
    for sid,g in scenes.items():
        for name,key in [('scene_gt','gt'),('scene_camera','camera'),('scene_gt_info','info')]:
            (out/'train_pbr'/sid/f'{name}.json').write_text(json.dumps(g[key],indent=2))
    (out/'composition_manifest.json').write_text(json.dumps(dict(seed=seed,count=count,records=records,
        status='train_only_pilot_composites_not_an_independent_evaluation_set',
        background_count=len(files),full_scene_depth_available=False,
        note='Repeated composites are not new independent 3D poses. This pilot does not cover all video poses.'),indent=2))
    print(json.dumps(dict(count=count,backgrounds=len(files),output=str(out))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--backgrounds',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--count',type=int,required=True)
    p.add_argument('--seed',type=int,default=20260924)
    a=p.parse_args()
    compose(a.assets.resolve(),a.backgrounds.resolve(),a.output.resolve(),a.count,a.seed)
