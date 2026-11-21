"""Link multi-angle foregrounds to a BOP view; random backgrounds are sampled online."""
import argparse,json,shutil,sys
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from bop_bb8 import BOPModel,BOPInstanceDataset
from bop_bb8.online_background import composite_rgba_crop,sample_background
from bop_bb8.geometry import project_corners


def dump(p,data):p.write_text(json.dumps(data,indent=2))


def split_backgrounds(pool,seed):
    rng=np.random.default_rng(seed);order=rng.permutation(len(pool));n_hold=min(32,max(1,len(pool)//10))
    def group(r):return r.get('background_group',r['sha256'])
    held_groups={group(pool[i]) for i in order[:n_hold]}
    held_hashes=set()
    while True:
        old=(len(held_groups),len(held_hashes))
        held_hashes.update(r['sha256'] for r in pool if group(r) in held_groups)
        held_groups.update(group(r) for r in pool if r['sha256'] in held_hashes)
        if old==(len(held_groups),len(held_hashes)):break
    hold=[int(i) for i in order if group(pool[i]) in held_groups or pool[i]['sha256'] in held_hashes]
    train=[int(i) for i in order if group(pool[i]) not in held_groups and pool[i]['sha256'] not in held_hashes]
    if not train:raise ValueError('No training backgrounds after group holdout')
    return train,hold


def prepare(base,assets,backgrounds,old_assets,out,pairs=8000,seed=20260926,minimum_backgrounds=5000):
    if pairs<1:raise ValueError('pairs must be positive')
    pool=json.loads((backgrounds/'manifest.json').read_text())['records']
    if len(pool)<minimum_backgrounds:raise ValueError('Insufficient background pool')
    files=[str((backgrounds/r['path']).resolve()) for r in pool]
    if not all(Path(f).is_file() for f in files):raise FileNotFoundError('Incomplete background pool')
    rng=np.random.default_rng(seed);train_indices,hold_indices=split_backgrounds(pool,seed)
    hold=[files[i] for i in hold_indices];train_bg=[files[i] for i in train_indices]
    manifest=json.loads((base/'manifest.json').read_text());bank=json.loads((assets/'manifest.json').read_text())['samples']
    train=[r for r in bank if r.get('split','train')=='train'];val=[r for r in bank if r.get('split')=='val']
    if not train or not val:raise ValueError('Need train and held-out pose assets')
    if pairs<len(train) and minimum_backgrounds>=5000:raise ValueError('Every training pose must be sampled')
    out.mkdir(parents=True,exist_ok=False);(out/'trainval').mkdir();shutil.copytree(base/'models',out/'models')
    model=BOPModel.from_models_info(base/'models/models_info.json')
    np.testing.assert_allclose(model.corners_mm(),BOPModel.from_models_info(assets/'models/models_info.json').corners_mm(),atol=1e-8)
    old_train=manifest['splits']['train']+[f'{100000+int(s):06}' for s in manifest['splits']['train']]
    for s in old_train+manifest['splits']['val']:(out/'trainval'/s).symlink_to((base/'trainval'/s).resolve(),target_is_directory=True)
    groups={};records={};pose_order=rng.permutation(len(train));max_error=0.
    def add(r,number,split,pair_id,variant,bg=None):
        nonlocal max_error
        scene_id=f'{300000+number//1000:06}' if split=='train' else '400000'
        iid=number%1000 if split=='train' else number
        scene=out/'trainval'/scene_id
        if scene_id not in groups:
            (scene/'rgb').mkdir(parents=True);groups[scene_id]=dict(gt={},info={},camera={})
        rgba_path=(assets/r['rgba']).resolve();rgba=cv2.imread(str(rgba_path),-1)
        if rgba is None:raise FileNotFoundError(rgba_path)
        uv,valid,_=project_corners(model.corners_mm(),r['K'],r['R'],r['t_mm'],image_size=(512,512))
        error=float(np.max(abs(uv-np.array(r['keypoints_2d']))));max_error=max(max_error,error)
        if error>.001 or not valid.all():raise ValueError('Foreground projection mismatch')
        box=r['bbox_xywh'];target=scene/'rgb'/f'{iid:06}.png'
        if split=='train':
            target.symlink_to(rgba_path)
            records[f'{scene_id}/{iid}/0']=dict(rgba=str(rgba_path),pair_id=pair_id,blurred=bool(variant),parent_id=r['id'])
        else:
            local_rng=np.random.default_rng(seed+pair_id);patch,_=sample_background([bg],local_rng)
            rgb=composite_rgba_crop(rgba,patch,(0,0,512,512),(512,512),0.)
            if variant:
                crop=BOPInstanceDataset._bbox_crop(box,(512,512),1.3);sigma=.9*crop[2]/256
                kernel=2*int(np.ceil(3*sigma))+1;rgb=cv2.GaussianBlur(rgb,(kernel,kernel),sigma)
            cv2.imwrite(str(target),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
        g=groups[scene_id];key=str(iid)
        g['gt'][key]=[dict(obj_id=1,cam_R_m2c=np.array(r['R']).reshape(-1).tolist(),cam_t_m2c=r['t_mm'])]
        g['camera'][key]=dict(cam_K=np.array(r['K']).reshape(-1).tolist(),width=512,height=512)
        g['info'][key]=[dict(bbox_obj=box,bbox_visib=box,visib_fract=1.)]
    for pair in range(pairs):
        r=train[int(pose_order[pair%len(train)])]
        for variant in [0,1]:add(r,pair*2+variant,'train',pair,variant)
    for j,r in enumerate(val):
        for b in range(4):
            pair=j*4+b
            for variant in [0,1]:add(r,pair*2+variant,'val',pair,variant,hold[pair%len(hold)])
    for sid,g in groups.items():
        for name,key in [('scene_gt','gt'),('scene_gt_info','info'),('scene_camera','camera')]:dump(out/'trainval'/sid/(name+'.json'),g[key])
    config=dict(base_dataset=str(base.resolve()),old_background_assets=str(old_assets.resolve()),old_train_scenes=old_train,
        train_backgrounds=train_bg,heldout_backgrounds=hold,new_records=records,
        val_scenes=manifest['splits']['val']+['400000'],old_val_scenes=manifest['splits']['val'],
        new_val_scene='400000',seed=seed,pairs=pairs,train_pose_assets=len(train),heldout_pose_assets=len(val),
        policy='all training instances get an independently sampled background each epoch; no extra occlusion; paired clean/blur',
        max_projection_error_px=max_error,expected_new_val_instances=len(val)*8,
        background_holdout_policy='source-frame groups and exact image hashes disjoint; not an independent-video split',
        expected_train_instances=16000+2*pairs,expected_val_instances=1000+len(val)*8)
    dump(out/'online_config.json',config)
    print('PREPARED',len(train),'train poses',len(val),'heldout poses',2*pairs,'new samples',len(train_bg),'training backgrounds',flush=True)
    return config


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['base','assets','backgrounds','old-assets','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--pairs',type=int,default=8000);a=p.parse_args();prepare(a.base.resolve(),a.assets.resolve(),a.backgrounds.resolve(),a.old_assets.resolve(),a.output.resolve(),a.pairs)
