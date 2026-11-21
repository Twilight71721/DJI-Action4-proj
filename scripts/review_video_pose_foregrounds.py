"""Audit native BOP projections and render an evidence-linked pose gallery."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from bop_bb8 import BOPModel, BOPInstanceDataset
from bop_bb8.geometry import project_corners
from review_pose_replacement_pilot import draw, panel


def main(out):
    plan=json.loads((out/'plan.json').read_text())
    review=out/'review';review.mkdir(exist_ok=True)
    model=BOPModel.from_models_info(out/'models/models_info.json')
    samples=[];checks=[];cards=[]
    for group in plan['groups']:
        render=out/'renders'/group['id']
        meta=json.loads((render/'metadata.json').read_text())
        assert meta['geometry_unchanged']
        scene=render/'bop/train_pbr/000000'
        gt=json.loads((scene/'scene_gt.json').read_text())
        cams=json.loads((scene/'scene_camera.json').read_text())
        ds=BOPInstanceDataset(render/'bop/train_pbr',model,input_size=(256,256),heatmap_size=(64,64),
                              crop_mode='bbox',crop_padding=1.3,heatmap_kernel='gaussian',heatmap_sigma=2.)
        tiles=[]
        for i,r in enumerate(meta['group']['records']):
            K=np.array(cams[str(i)]['cam_K']).reshape(3,3)
            R=np.array(gt[str(i)][0]['cam_R_m2c']).reshape(3,3)
            t=np.array(gt[str(i)][0]['cam_t_m2c'])
            uv,valid,_=project_corners(model.corners_mm(),K,R,t,image_size=(512,512))
            err=float(np.max(np.linalg.norm(uv-np.array(r['keypoints_2d']),axis=1)))
            mask=cv2.imread(str(scene/'mask'/f'{i:06d}_000000.png'),0)
            vis=cv2.imread(str(scene/'mask_visib'/f'{i:06d}_000000.png'),0)
            assert err<.001 and valid.all() and np.linalg.det(R)>.999
            assert not np.any((vis>0)&(mask==0))
            sample=ds.load_instance('000000',i,0)
            assert sample.heatmaps.shape==(8,64,64) and np.isfinite(sample.heatmaps).all()
            ys,xs=np.where(mask>0)
            bbox=[int(xs.min()),int(ys.min()),int(np.ptp(xs)+1),int(np.ptp(ys)+1)]
            assert xs.min()>=uv[:,0].min()-2 and xs.max()<=uv[:,0].max()+2
            assert ys.min()>=uv[:,1].min()-2 and ys.max()<=uv[:,1].max()+2
            record=dict(r,K=K.tolist(),R=R.tolist(),t_mm=t.tolist(),keypoints_2d=uv.tolist(),
                        bbox_xywh=bbox,rgba=str((render/'rgba'/f'{i:06d}.png').relative_to(out)),
                        mask=str((scene/'mask'/f'{i:06d}_000000.png').relative_to(out)),
                        usage='foreground_candidate_for_server_compositing')
            samples.append(record);checks.append(dict(id=r['id'],max_projection_error_px=err))
            rgba=cv2.imread(str(out/record['rgba']),-1);alpha=rgba[:,:,3:]/255
            neutral=np.rint(rgba[:,:,:3]*alpha+105*(1-alpha)).astype(np.uint8)
            if r['delta_degrees']==[0,0,0] and r.get('source_preview'):
                original=cv2.imread(str(out/r['source_preview']))
                sheet=panel([original,neutral,draw(neutral,uv)],['REAL REFERENCE','RECENTERED 3D VIEW','SYNTHETIC BB8 GT'],320)
                name=f'{r["id"]}.jpg';cv2.imwrite(str(review/name),sheet)
                cards.append(f'<section><h3>{r["id"]}</h3><img src="{name}"></section>')
            if i%3==0:tiles.append(panel([draw(neutral,uv)],[r['id']],224))
        for n in range(0,len(tiles),12):
            part=tiles[n:n+12]
            while len(part)%3:part.append(np.zeros_like(part[0]))
            name=f'{group["id"]}_sheet_{n//12}.jpg'
            cv2.imwrite(str(review/name),np.vstack([np.hstack(part[j:j+3]) for j in range(0,len(part),3)]))
    audit=dict(rendered=len(samples),max_native_bop_projection_error_px=max(c['max_projection_error_px'] for c in checks),
               training_reader_heatmaps_checked=len(samples),geometry_and_mask_checks_passed=True,checks=checks)
    (out/'audit.json').write_text(json.dumps(audit,indent=2))
    (out/'manifest.json').write_text(json.dumps(dict(samples=samples,counts=dict(Counter(r['usage'] for r in samples)),
        status='video_adaptation_foregrounds_not_real_pose_ground_truth',
        caveats=['Real intrinsics and marker mounting geometry are estimated.',
                 'View is recentered, not a replacement in the original image.',
                 'Full-video scan is complete; full pose coverage is not certified.',
                 'All descendants are train-only and share video_adaptation_all.']),indent=2))
    page='''<!doctype html><meta charset="utf-8"><title>全视频姿态扩展检查</title>
    <style>body{max-width:1100px;margin:30px auto;background:#14212a;color:#eee;font:17px/1.7 system-ui}img{max-width:100%}section{background:#223743;padding:16px;margin:20px 0}a{color:#7ddff5}</style>
    <h1>全视频扫描后的标记辅助姿态候选</h1><p>这是相机前景姿态检查，未直接替换原图。真实参考内参和标记安装位置为估计值，绿色框是合成相机的投影标签，不是真实帧测量真值。</p>'''
    page+=f'<p>{len(samples)} 张渲染图，{len(cards)} 个参考视角。另有侧视候选需要单独检查；本页不能证明完整视频的所有姿态均已覆盖。</p>'
    page+=''.join(cards)+'<p><a href="../audit.json">几何检查</a> · <a href="../manifest.json">样本清单</a></p>'
    (review/'index.html').write_text(page)
    print(json.dumps({k:v for k,v in audit.items() if k!='checks'}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    main(p.parse_args().output.resolve())
