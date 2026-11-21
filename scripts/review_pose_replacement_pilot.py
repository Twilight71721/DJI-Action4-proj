"""Compose a bounded real-frame replacement pilot and audit native BOP labels."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from bop_bb8 import BOPModel, BOPInstanceDataset
from bop_bb8.geometry import project_corners

EDGES = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]


def compose_replacement(real, rgba, removal, occluder=None):
    """Remove only the annotated old body; preserve explicitly known occluders."""
    if real.shape != rgba[:,:,:3].shape or removal.shape != real.shape[:2]:
        raise ValueError('Replacement geometry mismatch')
    if occluder is None:
        occluder=np.zeros(removal.shape,np.uint8)
    removal=cv2.dilate(removal,np.ones((7,7),np.uint8))
    removal[occluder>0]=0
    clean=cv2.inpaint(real,removal,5,cv2.INPAINT_TELEA)
    alpha=rgba[:,:,3:].astype(np.float32)/255
    image=np.rint(rgba[:,:,:3]*alpha+clean*(1-alpha)).clip(0,255).astype(np.uint8)
    image[occluder>0]=real[occluder>0]
    exposed=(removal>0)&(rgba[:,:,3]<250)&(occluder==0)
    ratio=float(exposed.sum()/max(1,np.count_nonzero(removal)))
    return image,clean,removal,exposed,ratio


def draw(image, points):
    result=image.copy();points=np.rint(points).astype(int)
    for a,b in EDGES:cv2.line(result,points[a],points[b],(50,240,80),1,cv2.LINE_AA)
    for j,q in enumerate(points):
        cv2.circle(result,q,3,(0,240,255),-1)
        cv2.putText(result,str(j),q+[4,-4],cv2.FONT_HERSHEY_SIMPLEX,.45,(0,240,255),1,cv2.LINE_AA)
    return result


def panel(images, captions, size=384):
    panes=[]
    for image,caption in zip(images,captions):
        canvas=np.zeros((size+32,size,3),np.uint8)
        canvas[32:]=cv2.resize(image,(size,size),interpolation=cv2.INTER_AREA)
        cv2.putText(canvas,caption,(8,22),cv2.FONT_HERSHEY_SIMPLEX,.48,(240,240,240),1,cv2.LINE_AA)
        panes.append(canvas)
    return np.hstack(panes)


def main(out):
    plan=json.loads((out/'plan.json').read_text())
    decisions=json.loads((out/'visual_review.json').read_text()) if (out/'visual_review.json').exists() else {}
    dest=out/'review';dest.mkdir(exist_ok=False)
    composites=out/'replacements';composites.mkdir()
    model=BOPModel.from_models_info(out/'models/models_info.json')
    checks=[];samples=[];cards=[];pose_tiles=[]
    for group in plan['groups']:
        render=out/'renders'/group['id']
        metadata=json.loads((render/'metadata.json').read_text())
        scene=render/'bop/train_pbr/000000'
        gt=json.loads((scene/'scene_gt.json').read_text())
        cameras=json.loads((scene/'scene_camera.json').read_text())
        info=json.loads((scene/'scene_gt_info.json').read_text())
        ds=BOPInstanceDataset(render/'bop/train_pbr',model,input_size=(256,256),heatmap_size=(64,64),
                              crop_mode='bbox',crop_padding=1.3,heatmap_kernel='gaussian',heatmap_sigma=2.)
        for i,r in enumerate(metadata['group']['records']):
            cam=np.array(cameras[str(i)]['cam_K']).reshape(3,3)
            pose=gt[str(i)][0];R=np.array(pose['cam_R_m2c']).reshape(3,3);t=np.array(pose['cam_t_m2c'])
            expected=np.array(r['keypoints_2d'])
            uv,valid,_=project_corners(model.corners_mm(),cam,R,t,image_size=(512,512))
            error=float(np.max(np.linalg.norm(uv-expected,axis=1)))
            assert error<.001 and valid.all(),(r['id'],error)
            mask=cv2.imread(str(scene/'mask'/f'{i:06d}_000000.png'),0)
            vis=cv2.imread(str(scene/'mask_visib'/f'{i:06d}_000000.png'),0)
            assert mask.shape==(512,512) and not np.any((vis>0)&(mask==0))
            ys,xs=np.where(mask>0);bbox=[int(xs.min()),int(ys.min()),int(xs.max()-xs.min()+1),int(ys.max()-ys.min()+1)]
            assert np.isfinite(uv).all() and np.linalg.det(R)>.999
            lo,hi=uv.min(0),uv.max(0)
            assert xs.min()>=lo[0]-2 and xs.max()<=hi[0]+2 and ys.min()>=lo[1]-2 and ys.max()<=hi[1]+2
            loaded=ds.load_instance('000000',i,0)
            assert loaded.heatmaps.shape==(8,64,64) and np.isfinite(loaded.heatmaps).all()
            # Alpha includes thin decals; native BOP body masks remain the unchanged GLB.
            rgba=cv2.imread(str(render/'rgba'/f'{i:06d}.png'),cv2.IMREAD_UNCHANGED)
            assert rgba.shape==(512,512,4) and rgba[:,:,3].max()==255
            alpha=rgba[:,:,3:]/255
            neutral=np.rint(rgba[:,:,:3]*alpha+np.full((512,512,3),105)*(1-alpha)).astype(np.uint8)
            check=dict(id=r['id'],native_bop_projection_error_px=error,mask_pixels=int((mask>0).sum()),
                mask_visible_pixels=int((vis>0).sum()),bbox_from_mask=bbox,bbox_native=info[str(i)][0]['bbox_obj'],
                heatmaps_finite=True,all_corners_valid=True)
            checks.append(check)
            record=dict(r,K=cam.tolist(),R=R.tolist(),t_mm=t.tolist(),keypoints_2d=uv.tolist(),
                rgba=str((render/'rgba'/f'{i:06d}.png').relative_to(out)),
                mask=str((scene/'mask'/f'{i:06d}_000000.png').relative_to(out)),bbox_xywh=bbox)
            if r['kind']=='foreground':
                if i%4 in [0,3]:
                    tile=panel([draw(neutral,uv)],[r['id'].replace('pose_','')],224)
                    pose_tiles.append(tile)
                record['usage']='foreground_candidate_for_server_compositing'
            else:
                anchor=plan['anchors'][r['anchor']]
                original=cv2.imread(str(out/'sources'/f'anchor_{r["anchor"]}.png'))
                removal=cv2.imread(str(out/'sources'/f'anchor_{r["anchor"]}_removal.png'),0)
                occluder=cv2.imread(str(out/anchor['occluder_mask']),0) if anchor.get('occluder_mask') else None
                mixed,clean,region,exposed,ratio=compose_replacement(original,rgba,removal,occluder)
                reasons=[]
                if anchor['known_occlusion_unresolved']:reasons.append('unresolved_real_foreground_occlusion')
                if ratio>.20:reasons.append('too_much_exposed_inpainting')
                if r['id'] in decisions.get('rejected',[]):reasons.append('user_rejected_pose_and_occlusion')
                accepted=not reasons and r['id'] in decisions.get('accepted_replacements',[])
                record.update(exposed_inpaint_fraction=ratio,real_reference_group=anchor['source_group'],
                    replacement_rejection_reasons=reasons,usage='rejected_replacement' if reasons else (
                        'replacement_pilot_visual_checked' if accepted else 'replacement_candidate_pending_visual_review'),
                    reference_pose_estimated=True,real_scene_depth_available=False)
                item=composites/r['id'];item.mkdir()
                cv2.imwrite(str(item/'clean.png'),mixed)
                cv2.imwrite(str(item/'removal_mask.png'),region)
                cv2.imwrite(str(item/'exposed_inpaint_mask.png'),exposed.astype(np.uint8)*255)
                composite_visible=mask.copy()
                if occluder is not None:composite_visible[occluder>0]=0
                cv2.imwrite(str(item/'mask_visib.png'),composite_visible)
                record['composite_visible_pixels']=int((composite_visible>0).sum())
                record['occluder_method']=anchor.get('occluder_method','no foreground occluder annotated for this candidate')
                # These are severity previews at fixed geometry, not additional poses.
                mild=cv2.GaussianBlur(mixed,(9,9),1.2)
                medium=cv2.GaussianBlur(mixed,(15,15),2.4)
                cv2.imwrite(str(item/'mild.png'),mild);cv2.imwrite(str(item/'medium.png'),medium)
                record['composite']=str((item/'clean.png').relative_to(out))
                annotation=dict(K=cam.tolist(),cam_R_m2c=R.reshape(-1).tolist(),cam_t_m2c=t.tolist(),
                    keypoints_2d=uv.tolist(),bbox_obj=bbox,corner_order=plan['corner_order'],
                    mask_visib='mask_visib.png',occluder_method=record['occluder_method'],
                    note='Synthetic camera GT. Reference real pose is estimated; full real-scene depth is unknown.',
                    candidate=not reasons)
                (item/'labels.json').write_text(json.dumps(annotation,indent=2))
                overview=panel([original,mixed,draw(mixed,uv)],['REAL INPUT','REPLACED RGB','SYNTHETIC GT 0-7'])
                file=f'{r["id"]}.jpg';cv2.imwrite(str(dest/file),overview)
                blurfile=f'{r["id"]}_blur.jpg'
                cv2.imwrite(str(dest/blurfile),panel([mixed,mild,medium],['CLEAR','MILD: sigma=0.6 at 256','MEDIUM: sigma=1.2 at 256']))
                status='排除：遮挡或修补比例未通过' if reasons else ('用户已确认保留' if accepted else '待视觉复核的候选')
                cards.append(f'<section><h3>{r["id"]} · frame {anchor["frame"]} / {anchor["side"]} · {status}</h3>'
                    f'<p>姿态变化 {r["delta_degrees"]} 度；暴露修补区域占移除区域 {ratio:.1%}。角点来自新渲染相机。</p>'
                    f'<img src="{file}"><details><summary>查看清晰 / 轻度 / 中度模糊</summary><img src="{blurfile}"></details></section>')
            samples.append(record)
    sheets=[]
    for start in range(0,len(pose_tiles),16):
        chunk=pose_tiles[start:start+16]
        while len(chunk)%4:chunk.append(np.zeros_like(chunk[0]))
        sheet=np.vstack([np.hstack(chunk[i:i+4]) for i in range(0,len(chunk),4)])
        name=f'pose_sheet_{start//16}.jpg';cv2.imwrite(str(dest/name),sheet);sheets.append(name)
    audit=dict(rendered=len(checks),groups=len(plan['groups']),
        max_native_bop_projection_error_px=max(r['native_bop_projection_error_px'] for r in checks),
        body_mask_subset_checks_passed=True,current_training_reader_checked=True,checks=checks,
        no_bulk_background_download=True,no_training_started=True)
    (out/'audit.json').write_text(json.dumps(audit,indent=2))
    manifest=dict(status='pilot_pending_visual_review',samples=samples,counts=dict(Counter(s['usage'] for s in samples)),
        caveats=['Only three real anchor frames; not full real-video pose coverage.',
                 'Generic view bins are augmentation proposals, not real pose measurements.',
                 'Unresolved occlusion is rejected, not silently pasted over.',
                 'All descendants of the same source neighborhood must stay in the same dataset split.',
                 'Real composites have no trustworthy full-scene depth map. Native depth describes foreground only.'])
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
    <title>本地姿态与真实帧替换试样</title><style>body{background:#14212a;color:#edf1f4;font:17px/1.7 system-ui;max-width:1200px;margin:30px auto;padding:0 20px}a{color:#83d7f2}img{max-width:100%;height:auto}section{background:#20343f;padding:18px;margin:24px 0}h1{font-size:30px}.notice{border-left:4px solid #d9b268;padding:14px;background:#263b46}summary{cursor:pointer}</style>
    <h1>本地姿态与真实帧替换试样</h1><p>大量背景下载与动态组合留在服务器；本页检查新相机姿态、真实场景替换和标签。</p>
    <p class="notice">本轮是小批管线验证：132 个渲染实例（112 个扩展姿态前景 + 20 个真实帧替换尝试）。真实参考目前为第 0、500、1350 帧，不代表已覆盖整段视频。真实内参和参考姿态为估计值；绿色框是新渲染相机的投影真值，不是网络预测。</p>
    <p>原 GLB 几何不变，沿用当前 0～7 通道顺序。HCCEPose 派生 BlenderProc 的原生 BOP writer 生成几何与 mask；真实图合成另存，不伪造真实背景深度。</p>
    <h2>真实帧相机替换</h2><p>修补只用于原机身移除后的暴露边缘。第 500 帧左机身遮挡未解决，明确排除；其他样本仍需检查附件连接、边缘残留和外观。</p>'''
    page+=''.join(cards)+'<h2>扩展姿态前景及投影标签</h2><p>展示部分左右机身与 14 个视角/参考姿态类别；背景为检查用中性底色。底面和侧面等是覆盖检查，不表示已经匹配真实视频。</p>'
    page+=''.join(f'<img src="{s}">' for s in sheets)
    page+='<p><a href="../audit.json">标签检查</a> · <a href="../manifest.json">全部样本及排除记录</a></p></html>'
    (dest/'index.html').write_text(page)
    print(json.dumps(dict(audit={k:v for k,v in audit.items() if k!='checks'},counts=manifest['counts']),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args().output.resolve())
