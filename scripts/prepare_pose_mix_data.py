"""Create paired clean/blur foreground composites and a leakage-checked BOP view."""
import argparse
import json
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bop_bb8 import BOPInstanceDataset, BOPModel
from bop_bb8.geometry import project_corners
from compose_pose_backgrounds import background_patch, bbox
from server_background_assets import dump, sha


def transform_foreground(rgba, mask, K, scale, dx, dy):
    A = np.array([[scale, 0, 256 * (1-scale) + dx],
                  [0, scale, 256 * (1-scale) + dy], [0, 0, 1]], float)
    alpha = rgba[:, :, 3:].astype(np.float32) / 255
    color = cv2.warpAffine(rgba[:, :, :3].astype(np.float32) * alpha, A[:2], (512, 512))
    alpha = cv2.warpAffine(alpha, A[:2], (512, 512))[:, :, None]
    mask = cv2.warpAffine(mask, A[:2], (512, 512), flags=cv2.INTER_NEAREST)
    return color, alpha, mask, A @ np.asarray(K), A


def prepare(assets, backgrounds, base, out, pairs=8000, seed=20260925):
    if pairs < 1:
        raise ValueError('pairs must be positive')
    source = json.loads((assets/'manifest.json').read_text())
    pool = [r for r in source['samples'] if r['usage']=='foreground_candidate_for_server_compositing']
    if not pool:
        raise ValueError('No foreground assets')
    bg_manifest = json.loads((backgrounds/'manifest.json').read_text())
    files = [backgrounds/r['path'] for r in bg_manifest['records']]
    if not files or any(not p.is_file() for p in files):
        raise ValueError('Incomplete background pool')
    manifest = json.loads((base/'manifest.json').read_text())
    model = BOPModel.from_models_info(base/'models/models_info.json')
    np.testing.assert_allclose(model.corners_mm(), BOPModel.from_models_info(assets/'models/models_info.json').corners_mm(), atol=1e-8)
    out.mkdir(parents=True, exist_ok=False)
    (out/'preview').mkdir()
    shutil.copytree(base/'models', out/'models')
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(pool))
    scenes, records, preview, max_error = {}, [], [], 0.
    cv2.setNumThreads(1)
    for pair in range(pairs):
        r = pool[int(order[pair % len(order)])]
        rgba = cv2.imread(str(assets/r['rgba']), -1)
        mask = cv2.imread(str(assets/r['mask']), 0)
        if rgba is None or mask is None or rgba.shape != (512,512,4):
            raise ValueError('Invalid foreground '+r['id'])
        scale, dx, dy = rng.uniform(.75,1.05), rng.uniform(-32,32), rng.uniform(-32,32)
        color, alpha, mask, K, A = transform_foreground(rgba, mask, r['K'], scale, dx, dy)
        uv, valid, _ = project_corners(model.corners_mm(), K, r['R'], r['t_mm'], image_size=(512,512))
        if not valid.all():
            raise ValueError('Unexpected truncation; no cropping/occlusion requested')
        expected = np.c_[r['keypoints_2d'],np.ones(8)] @ A.T
        error = float(np.abs(uv-expected[:,:2]).max());max_error=max(max_error,error)
        if error > .001:
            raise ValueError('Transformed camera/label mismatch')
        bg_id = int(rng.integers(len(files)))
        raw = cv2.imread(str(files[bg_id]))
        if raw is None:
            raise ValueError('Unreadable background '+str(files[bg_id]))
        bg = background_patch(raw, 512, rng)
        clean = np.clip(np.rint(color + bg*(1-alpha)),0,255).astype(np.uint8)
        box = bbox(mask)
        crop = BOPInstanceDataset._bbox_crop(box, (512,512), 1.3)
        sigma256 = float(rng.uniform(.3,1.4))
        native_sigma = sigma256 * crop[2]/256
        kernel = 2*int(np.ceil(3*native_sigma))+1
        blurred = cv2.GaussianBlur(clean, (kernel,kernel), native_sigma)
        for variant, image in enumerate([clean, blurred]):
            number=pair*2+variant;sid=f'{200000+number//1000:06d}';iid=number%1000
            scene=out/'new_train'/sid
            if sid not in scenes:
                for folder in ['rgb','mask','mask_visib']:(scene/folder).mkdir(parents=True)
                scenes[sid]=dict(gt={},camera={},info={})
            cv2.imwrite(str(scene/'rgb'/f'{iid:06d}.png'),image)
            for folder in ['mask','mask_visib']:cv2.imwrite(str(scene/folder/f'{iid:06d}_000000.png'),mask)
            g=scenes[sid];key=str(iid);pixels=int((mask>0).sum())
            g['gt'][key]=[dict(obj_id=1,cam_R_m2c=np.array(r['R']).reshape(-1).tolist(),cam_t_m2c=r['t_mm'])]
            g['camera'][key]=dict(cam_K=K.reshape(-1).tolist(),width=512,height=512)
            g['info'][key]=[dict(bbox_obj=box,bbox_visib=box,px_count_all=pixels,px_count_valid=pixels,px_count_visib=pixels,visib_fract=1.)]
            records.append(dict(scene=sid,image_id=iid,pair_id=pair,variant='clean' if variant==0 else 'blur',
                parent_id=r['id'],source_group='video_adaptation_all',split='train',
                background=bg_manifest['records'][bg_id]['path'],sigma256=0. if variant==0 else sigma256,
                affine=A.tolist(),extra_occlusion=False))
        if pair < 12:
            panes=[]
            for image in [clean,blurred]:
                im=image.copy()
                for point in uv:cv2.circle(im,tuple(np.rint(point).astype(int)),3,(0,255,0),-1)
                panes.append(cv2.resize(im,(256,256)))
            path=f'pair_{pair:03d}.jpg';cv2.imwrite(str(out/'preview'/path),np.hstack(panes));preview.append(path)
        if (pair+1)%500==0:print('[compose]',pair+1,'/',pairs,flush=True)
    for sid,g in scenes.items():
        for name,key in [('scene_gt','gt'),('scene_camera','camera'),('scene_gt_info','info')]:
            dump(out/'new_train'/sid/f'{name}.json',g[key])
    split=manifest['splits']
    old_scenes=set(split['train']+split['val']) | {f'{100000+int(s):06d}' for s in split['train']}
    actual={p.name for p in (base/'trainval').iterdir() if (p/'scene_gt.json').exists()}
    if actual!=old_scenes or old_scenes.intersection(scenes):
        raise ValueError('Unexpected trainval scenes or scene collision')
    merged=out/'trainval';merged.mkdir()
    for sid in sorted(old_scenes):(merged/sid).symlink_to((base/'trainval'/sid).resolve(),target_is_directory=True)
    for sid in sorted(scenes):(merged/sid).symlink_to((out/'new_train'/sid).resolve(),target_is_directory=True)
    dump(out/'composition_manifest.json',dict(pairs=pairs,images=pairs*2,seed=seed,records=records,
        foreground_manifest_sha256=sha(assets/'manifest.json'),background_manifest_sha256=sha(backgrounds/'manifest.json'),
        base_manifest_sha256=sha(base/'manifest.json'),old_scene_ids=sorted(old_scenes),new_scene_ids=sorted(scenes),
        val_scenes=split['val'],extra_occlusion=False,crop_jitter=False,max_projection_error_px=max_error))
    (out/'preview/index.html').write_text('<!doctype html><meta charset="utf-8"><h1>新姿态：左清晰／右模糊</h1><p>绿色点仅用于核查，不在训练 RGB 中；没有额外遮挡。</p>'+''.join(f'<p><img src="{p}"></p>' for p in preview))
    print('[prepared]',pairs*2,'new train images',out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for flag in ['assets','backgrounds','base-dataset','output']:p.add_argument('--'+flag,type=Path,required=True)
    p.add_argument('--pairs',type=int,default=8000);p.add_argument('--seed',type=int,default=20260925)
    a=p.parse_args();prepare(a.assets.resolve(),a.backgrounds.resolve(),a.base_dataset.resolve(),a.output.resolve(),a.pairs,a.seed)
