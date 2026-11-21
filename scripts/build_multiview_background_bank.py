"""Systematic object-view rendering; backgrounds are composed later on the server."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from render_pose_replacement_pilot import view_rotation,rotation
from bop_bb8 import BOPModel,BOPInstanceDataset
from bop_bb8.geometry import project_corners


def prepare(out):
    out.mkdir(parents=True,exist_ok=False)
    source=ROOT/'outputs/full_video_pose_coverage_20260924'
    shutil.copytree(source/'models',out/'models')
    model=BOPModel.from_models_info(out/'models/models_info.json')
    center=(model.min_xyz_mm+model.max_xyz_mm)/2
    groups=[];jobs={'left':[],'right':[]}
    orientations=[(a,e,r,'train') for e in [-60,-30,0,30,60] for a in range(0,360,30) for r in [0,90,180,270]]
    orientations += [(0,e,r,'train') for e in [-89.9,89.9] for r in [0,90,180,270]]
    orientations += [(a,e,45,'val') for e in [-45,-15,15,45] for a in range(15,360,60)]
    for i,(az,el,roll,split) in enumerate(orientations):
        a,e=np.deg2rad([az,el]);d=[np.cos(e)*np.sin(a),np.sin(e),np.cos(e)*np.cos(a)]
        R=rotation([0,0,roll])@view_rotation(d);t=np.array([0,0,205.])-R@center
        uv,valid,_=project_corners(model.corners_mm(),[[700,0,256],[0,700,256],[0,0,1]],R,t,image_size=(512,512))
        assert valid.all() and np.linalg.det(R)>.999
        side='left' if i%2==0 else 'right'
        jobs[side].append(dict(id=f'multiview_{i:04}',kind='foreground',family=f'az{az}_el{el}',
            source_group=f'grid_{az}_{el}_{roll}',split=split,azimuth_deg=az,elevation_deg=el,roll_deg=roll,
            R=R.tolist(),t_mm=t.tolist(),delta_degrees=[0,0,roll],light_energy=[7.,10.,13.][i%3],
            exposure=[-.65,-.45,-.2][(i//3)%3],real_pose_ground_truth=False))
    for side,records in jobs.items():
        for start in range(0,len(records),24):
            groups.append(dict(id=f'grid_{side}_{start//24:02}',side=side,K=[[700,0,256],[0,700,256],[0,0,1]],records=records[start:start+24]))
    plan=dict(width=512,height=512,groups=groups,anchors=[],glb=json.loads((source/'plan.json').read_text())['glb'],
        status='248 systematic train orientations plus 24 held-out interleaved orientations; not continuous full SO3 coverage')
    (out/'plan.json').write_text(json.dumps(plan,indent=2))


def audit(out):
    import cv2
    from review_pose_replacement_pilot import draw
    model=BOPModel.from_models_info(out/'models/models_info.json');records=[];tiles=[];errors=[]
    for g in json.loads((out/'plan.json').read_text())['groups']:
        render=out/'renders'/g['id'];meta=json.loads((render/'metadata.json').read_text());scene=render/'bop/train_pbr/000000'
        gt=json.loads((scene/'scene_gt.json').read_text());cams=json.loads((scene/'scene_camera.json').read_text())
        ds=BOPInstanceDataset(render/'bop/train_pbr',model,input_size=(256,256),heatmap_size=(64,64),crop_mode='bbox',crop_padding=1.3,heatmap_kernel='gaussian',heatmap_sigma=2.)
        for i,r in enumerate(meta['group']['records']):
            pose=gt[str(i)][0];K=np.array(cams[str(i)]['cam_K']).reshape(3,3);R=np.array(pose['cam_R_m2c']).reshape(3,3);t=pose['cam_t_m2c']
            uv,valid,_=project_corners(model.corners_mm(),K,R,t,image_size=(512,512));error=float(np.max(np.abs(uv-r['keypoints_2d'])))
            assert error<.001 and valid.all()
            sample=ds.load_instance('000000',i,0)
            assert sample.valid.all() and np.isfinite(sample.heatmaps).all(),r['id']
            rgba=render/'rgba'/f'{i:06d}.png';mask=scene/'mask'/f'{i:06d}_000000.png'
            im=cv2.imread(str(rgba),-1);m=cv2.imread(str(mask),0);assert im.shape==(512,512,4) and m.any()
            y,x=np.where(m>0);box=[int(x.min()),int(y.min()),int(np.ptp(x)+1),int(np.ptp(y)+1)]
            records.append(dict(r,K=K.tolist(),R=R.tolist(),t_mm=t,keypoints_2d=uv.tolist(),bbox_xywh=box,
                rgba=str(rgba.relative_to(out)),mask=str(mask.relative_to(out)),usage='foreground_candidate_for_server_compositing'))
            errors.append(error)
            alpha=im[:,:,3:]/255;neutral=np.rint(im[:,:,:3]*alpha+100*(1-alpha)).astype(np.uint8)
            tile=cv2.resize(draw(neutral,uv),(192,192));cv2.putText(tile,r['id'],(4,185),cv2.FONT_HERSHEY_SIMPLEX,.35,(255,255,255),1);tiles.append(tile)
    review=out/'review';review.mkdir(exist_ok=True)
    for start in range(0,len(tiles),24):
        group=tiles[start:start+24];group += [np.zeros_like(group[0])]*(24-len(group))
        cv2.imwrite(str(review/f'views_{start//24:02}.jpg'),np.vstack([np.hstack(group[i:i+6]) for i in range(0,24,6)]))
    (out/'manifest.json').write_text(json.dumps(dict(samples=records,max_projection_error_px=max(errors)),indent=2))
    counts={s:sum(r['split']==s for r in records) for s in ['train','val']}
    (review/'index.html').write_text('<meta charset="utf-8"><h1>多角度前景：绿色线仅为标签预览</h1><p>'+str(counts)+'。不是逐帧真实位姿真值。</p>'+''.join(f'<img style="max-width:100%" src="views_{i:02}.jpg">' for i in range((len(tiles)+23)//24)))
    print('AUDIT',len(records),max(errors),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--action',choices=['prepare','render','audit'],required=True);a=p.parse_args();out=a.output.resolve()
    if a.action=='prepare':prepare(out)
    elif a.action=='audit':audit(out)
    else:
        for g in json.loads((out/'plan.json').read_text())['groups']:
            if (out/'renders'/g['id']/'metadata.json').exists():continue
            log=out/(g['id']+'.log')
            with log.open('w') as f:subprocess.run([sys.executable,str(ROOT/'scripts/render_pose_replacement_pilot.py'),'render','--output',str(out),'--group',g['id'],'--samples','16'],stdout=f,stderr=subprocess.STDOUT,check=True)
            print('RENDERED',g['id'],flush=True)
        audit(out)
