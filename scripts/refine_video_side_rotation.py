"""Local 3D yaw/roll refinement, with the projected lens center held fixed."""
import argparse
import json
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from bop_bb8 import BOPModel, BOPInstanceDataset
from bop_bb8.geometry import project_corners
from render_pose_replacement_pilot import rotation
from review_pose_replacement_pilot import compose_replacement, panel, draw


def prepare(source,out):
    original=json.loads((source/'plan.json').read_text())
    out.mkdir(exist_ok=False)
    shutil.copytree(source/'models',out/'models')
    shutil.copytree(source/'sources',out/'sources')
    groups=[]
    for i,a in enumerate(original['anchors'][:2]):
        R=np.array(a['R']);t=np.array(a['t_mm'])
        pivot=np.array([15.,3.,15.8])
        pivot_camera=R@pivot+t
        records=[]
        for yaw,roll in [(0,0),(-6,0),(-12,0),(-18,0),(0,-5),(-6,-5),(-12,-5),(-18,-5)]:
            adjusted=rotation([0,0,roll])@rotation([0,yaw,0])@R
            translation=pivot_camera-adjusted@pivot
            top_plane=float((-adjusted.T@translation)[1]-21.65)
            if top_plane>=0:raise ValueError('Adjustment exposes top label face')
            records.append(dict(id=f'frame_{a["frame"]:06d}_yaw{yaw}_roll{roll}',kind='replacement',
                anchor=i,source_group='video_adaptation_all',family='port_side_rotation_refinement',
                R=adjusted.tolist(),t_mm=translation.tolist(),delta_degrees=[0,yaw,roll],
                yaw_left_degrees=-yaw,roll_ccw_degrees=-roll,top_label_plane_camera_side_mm=top_plane,
                light_energy=7.,exposure=-.65,real_pose_ground_truth=False,
                pivot_model_mm=pivot.tolist(),pivot_camera_mm=pivot_camera.tolist()))
        groups.append(dict(id=f'refine_{i}',side='left',K=a['K'],records=records))
    plan=dict(original,groups=groups,anchors=original['anchors'][:2],
              reference_plan=str(source/'plan.json'),rotation_convention='camera Y left yaw; camera Z negative image-plane roll; lens-center pivot')
    (out/'plan.json').write_text(json.dumps(plan,indent=2))


def prepare_pitch(source,out,left_degrees=None,up_degrees=None):
    original=json.loads((source/'plan.json').read_text())
    out.mkdir(exist_ok=False)
    shutil.copytree(source/'models',out/'models')
    shutil.copytree(source/'sources',out/'sources')
    i=next(i for i,a in enumerate(original['anchors']) if a['frame']==500)
    a=original['anchors'][i];R=np.array(a['R']);t=np.array(a['t_mm'])
    pivot=np.array([15.,3.,15.8]);pivot_camera=R@pivot+t
    records=[]
    if (left_degrees is None)!=(up_degrees is None):
        raise ValueError('Specify both --left-degrees and --up-degrees')
    left_values=[left_degrees] if left_degrees is not None else [18,24,30]
    up_values=[up_degrees] if up_degrees is not None else [0,5,10]
    for up in up_values:
        for left in left_values:
            # Raise the already upward-pointing lens face toward image-up,
            # away from the observer. Negative camera-X reduces front visibility.
            adjusted=rotation([-up,0,0])@rotation([0,-left,0])@R
            translation=pivot_camera-adjusted@pivot
            top_plane=float((-adjusted.T@translation)[1]-21.65)
            assert top_plane<0
            records.append(dict(id=f'frame_000500_left{left}_up{up}',kind='replacement',anchor=i,
                source_group='video_adaptation_all',family='port_side_left_up_refinement',
                R=adjusted.tolist(),t_mm=translation.tolist(),delta_degrees=[-up,-left,0],
                yaw_left_degrees=left,additional_left_degrees=left-18,pitch_raise_degrees=up,
                roll_ccw_degrees=0,top_label_plane_camera_side_mm=top_plane,
                light_energy=7.,exposure=-.65,real_pose_ground_truth=False,
                pivot_model_mm=pivot.tolist(),pivot_camera_mm=pivot_camera.tolist()))
    group=dict(id='left_up_500',side='left',K=a['K'],records=records)
    selected=records[0]['id'] if left_degrees is not None else 'frame_000500_left24_up5'
    plan=dict(original,groups=[group],reference_plan=str(source/'plan.json'),
              default_records={'500':selected},grid_columns=min(3,len(records)),
              user_requested_parameters=left_degrees is not None,
              rotation_convention='R = camera_X(-raise) @ camera_Y(-left) @ previous_hidden_top_R; lens pivot fixed',
              prior_review_pose='left18 / up0; additional yaw reported relative to this pose')
    (out/'plan.json').write_text(json.dumps(plan,indent=2))


def caption(record):
    if 'pitch_raise_degrees' in record:
        return f'左转 {record["yaw_left_degrees"]}°（再转 {record["additional_left_degrees"]}°），上抬 {record["pitch_raise_degrees"]}°'
    return f'左转 {record["yaw_left_degrees"]}° / 逆时针 {record["roll_ccw_degrees"]}°'


def review(out):
    plan=json.loads((out/'plan.json').read_text())
    dest=out/'review';dest.mkdir(exist_ok=True)
    model=BOPModel.from_models_info(out/'models/models_info.json')
    checks=[];cards=[]
    for group in plan['groups']:
        render=out/'renders'/group['id'];scene=render/'bop/train_pbr/000000'
        meta=json.loads((render/'metadata.json').read_text())
        gt=json.loads((scene/'scene_gt.json').read_text());cams=json.loads((scene/'scene_camera.json').read_text())
        ds=BOPInstanceDataset(render/'bop/train_pbr',model,input_size=(256,256),heatmap_size=(64,64),
            crop_mode='bbox',crop_padding=1.3,heatmap_kernel='gaussian',heatmap_sigma=2.)
        anchor=group['records'][0]['anchor'];a=plan['anchors'][anchor]
        real=cv2.imread(str(out/'sources'/f'anchor_{anchor}.png'))
        removal=cv2.imread(str(out/'sources'/f'anchor_{anchor}_removal.png'),0)
        occluder=cv2.imread(str(out/'sources'/f'anchor_{anchor}_occluder.png'),0)
        tiles=[]
        cards.append(f'<h2>第 {a["frame"]} 帧</h2>')
        for j,r in enumerate(meta['group']['records']):
            K=np.array(cams[str(j)]['cam_K']).reshape(3,3)
            R=np.array(gt[str(j)][0]['cam_R_m2c']).reshape(3,3);t=np.array(gt[str(j)][0]['cam_t_m2c'])
            uv,valid,_=project_corners(model.corners_mm(),K,R,t,image_size=(512,512))
            err=float(np.max(np.linalg.norm(uv-np.array(r['keypoints_2d']),axis=1)))
            assert err<.001 and valid.all() and np.linalg.det(R)>.999
            assert (-R.T@t)[1]-21.65<0
            np.testing.assert_allclose(R@np.array(r['pivot_model_mm'])+t,r['pivot_camera_mm'],atol=.0002)
            assert np.isfinite(ds.load_instance('000000',j,0).heatmaps).all()
            rgba=cv2.imread(str(render/'rgba'/f'{j:06d}.png'),-1)
            mixed,_,_,exposed,ratio=compose_replacement(real,rgba,removal,occluder)
            mask=cv2.imread(str(scene/'mask'/f'{j:06d}_000000.png'),0)
            vis=cv2.imread(str(scene/'mask_visib'/f'{j:06d}_000000.png'),0)
            assert not np.any((vis>0)&(mask==0))
            alpha=rgba[:,:,3:].astype(float)/255
            overlay=np.rint(rgba[:,:,:3]*(alpha*.5)+real*(1-alpha*.5)).astype(np.uint8)
            contours=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
            cv2.drawContours(overlay,contours,-1,(70,255,80),1)
            # Show the raw overlap too; repair/inpainting must not hide misalignment.
            label=f'LEFT {r["yaw_left_degrees"]} / UP {r["pitch_raise_degrees"]}' if 'pitch_raise_degrees' in r else f'YAW LEFT {r["yaw_left_degrees"]} / CCW {r["roll_ccw_degrees"]}'
            tile=panel([overlay],[label],256);tiles.append(tile)
            name=f'{r["id"]}.jpg'
            cv2.imwrite(str(dest/name),panel([real,overlay,mixed,draw(mixed,uv)],
                ['REAL','50% OVERLAP / BODY OUTLINE','COMPOSITE DRAFT','NEW SYNTHETIC GT'],384))
            checks.append(dict(id=r['id'],native_projection_error_px=err,
                top_face_hidden=True,lens_pivot_preserved=True,heatmaps_finite=True,
                exposed_repair_ratio=ratio,training_eligible=False))
            cards.append(f'<section><h3>{caption(r)}</h3><img src="{name}"></section>')
        columns=plan.get('grid_columns',4)
        while len(tiles)%columns:tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(str(dest/f'frame_{a["frame"]}_grid.jpg'),np.vstack([np.hstack(tiles[k:k+columns]) for k in range(0,len(tiles),columns)]))
    (out/'audit.json').write_text(json.dumps(dict(checks=checks,training_released=0),indent=2))
    page='''<!doctype html><meta charset="utf-8"><title>左转姿态细调</title>
    <style>body{background:#152630;color:#eee;max-width:1500px;margin:30px auto;padding:0 20px;font:17px/1.7 system-ui}img{max-width:100%}section{background:#233b47;margin:20px 0;padding:15px}a{color:#88d9f4}</style>
    <h1>左转姿态细调</h1><p>对原三维相机做局部旋转，固定镜头中心投影。顶部标签面仍背向观察相机。每档重新渲染与生成 K/R/t、mask 和角点；没有二维旋转图片冒充三维姿态。</p>
    <p>半透明叠加保留原始机身，绿色线是合成机身轮廓，用于检查重合程度。替换草稿仍有未解决的接口遮挡与修补边缘，暂不加入训练。</p>'''
    if plan.get('default_records'):
        page=page.replace('左转姿态细调','左转与上抬姿态细调')
        rs=plan['groups'][0]['records']
        lefts='/'.join(str(v) for v in sorted({r['additional_left_degrees'] for r in rs}))
        ups='/'.join(str(v) for v in sorted({r['pitch_raise_degrees'] for r in rs}))
        page+=f'<p>本轮针对第 500 帧：以此前左转 18°的候选为参照，再左转 {lefts}°，上抬 {ups}°。上抬指让镜头面朝上转、减小朝向观察者的角度，镜头中心位置保持不变。</p>'
    defaults={125:(12,0),500:(18,0)}
    selections=[]
    for group in plan['groups']:
        a=plan['anchors'][group['records'][0]['anchor']]
        frame=a['frame'];yaw,roll=defaults[frame]
        default_id=plan.get('default_records',{}).get(str(frame),f'frame_{frame:06d}_yaw{-yaw}_roll{-roll}')
        chosen=next(r for r in group['records'] if r['id']==default_id)
        current=f'{default_id}.jpg'
        options=''
        for r in group['records']:
            selected=' selected' if r['id']==default_id else ''
            options+=f'<option value="{r["id"]}.jpg"{selected}>{caption(r)}</option>'
        page+=f'<section><h2>第 {frame} 帧：本轮检查候选</h2><p><span id="caption-{frame}">{caption(chosen)}</span>；这些是局部调整量，不是经过标定的真实相机角度。</p>'
        page+=f'<label>切换相邻角度：<select onchange="document.getElementById(\'selected-{frame}\').src=this.value;document.getElementById(\'caption-{frame}\').textContent=this.options[this.selectedIndex].text">{options}</select></label>'
        page+=f'<img id="selected-{frame}" src="{current}"><details><summary>展开 {len(group["records"])} 档叠加图</summary><img src="frame_{frame}_grid.jpg"></details></section>'
        selections.append(dict(frame=frame,id=chosen['id'],yaw_left_degrees=chosen['yaw_left_degrees'],
            roll_ccw_degrees=chosen['roll_ccw_degrees'],pitch_raise_degrees=chosen.get('pitch_raise_degrees',0),
            status='user_requested_pose_parameters_composite_not_accepted' if plan.get('user_requested_parameters') else 'assistant_visual_candidate_not_user_confirmed',training_eligible=False))
    (out/'selection.json').write_text(json.dumps(dict(selections=selections,
        note='The original lens-center estimate is held fixed; this step refines rotation only and does not certify exact registration.'),indent=2))
    page+='<details><summary>展开全部候选的完整对照</summary>'+''.join(cards)+'</details>'
    (dest/'index.html').write_text(page)
    print(json.dumps(dict(rendered=len(checks),max_projection_error=max(r['native_projection_error_px'] for r in checks),training_released=0)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','prepare-pitch','review'])
    p.add_argument('--source',type=Path,default=ROOT/'outputs/video_side_pose_hidden_top_20260924')
    p.add_argument('--left-degrees',type=int)
    p.add_argument('--up-degrees',type=int)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.mode=='prepare':prepare(a.source.resolve(),a.output.resolve())
    elif a.mode=='prepare-pitch':prepare_pitch(a.source.resolve(),a.output.resolve(),a.left_degrees,a.up_degrees)
    else:review(a.output.resolve())
