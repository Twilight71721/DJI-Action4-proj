"""Inspect constrained side poses without releasing uncertain replacements."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from bop_bb8 import BOPModel, BOPInstanceDataset
from bop_bb8.geometry import project_corners
from review_pose_replacement_pilot import compose_replacement, panel, draw


def main(out):
    plan=json.loads((out/'plan.json').read_text())
    review=out/'review';review.mkdir(exist_ok=True)
    model=BOPModel.from_models_info(out/'models/models_info.json')
    checks=[];samples=[];cards=[]
    for group in plan['groups']:
        render=out/'renders'/group['id']
        if not (render/'metadata.json').exists():continue
        meta=json.loads((render/'metadata.json').read_text())
        r=meta['group']['records'][0];i=r['anchor'];a=plan['anchors'][i]
        scene=render/'bop/train_pbr/000000'
        native=json.loads((scene/'scene_gt.json').read_text())['0'][0]
        K=np.array(json.loads((scene/'scene_camera.json').read_text())['0']['cam_K']).reshape(3,3)
        R=np.array(native['cam_R_m2c']).reshape(3,3);t=np.array(native['cam_t_m2c'])
        uv,valid,_=project_corners(model.corners_mm(),K,R,t,image_size=(512,512))
        error=float(np.linalg.norm(uv-np.array(r['keypoints_2d']),axis=1).max())
        assert valid.all() and error<.001
        ds=BOPInstanceDataset(render/'bop/train_pbr',model,input_size=(256,256),heatmap_size=(64,64),
                              crop_mode='bbox',crop_padding=1.3,heatmap_kernel='gaussian',heatmap_sigma=2.)
        assert np.isfinite(ds.load_instance('000000',0,0).heatmaps).all()
        # The label is on +Y. Signed distance of the camera from that plane is
        # negative when the face points away; this is not an image repaint.
        top_side_mm=float((-R.T@t)[1]-21.65)
        if 'visibility_constraint' in a:assert top_side_mm<0
        rgba=cv2.imread(str(render/'rgba/000000.png'),-1)
        real=cv2.imread(str(out/'sources'/f'anchor_{i}.png'))
        removal=cv2.imread(str(out/'sources'/f'anchor_{i}_removal.png'),0)
        occluder=cv2.imread(str(out/'sources'/f'anchor_{i}_occluder.png'),0)
        mixed,_,region,exposed,ratio=compose_replacement(real,rgba,removal,occluder)
        np.testing.assert_array_equal(mixed[occluder>0],real[occluder>0])
        alpha=rgba[:,:,3:]/255
        neutral=np.rint(rgba[:,:,:3]*alpha+100*(1-alpha)).astype(np.uint8)
        name=f'side_{i}.jpg'
        cv2.imwrite(str(review/name),panel([real,neutral,mixed,draw(mixed,uv)],
                      ['REAL','CONSTRAINED POSE','COMPOSITE DRAFT','SYNTHETIC GT'],384))
        cv2.imwrite(str(review/f'repair_{i}.png'),exposed.astype(np.uint8)*255)
        record=dict(id=r['id'],frame=a['frame'],R=R.tolist(),t_mm=t.tolist(),K=K.tolist(),
                    keypoints_2d=uv.tolist(),top_label_plane_camera_side_mm=top_side_mm,
                    top_label_face_back_facing=top_side_mm<0,
                    native_bop_projection_error_px=error,exposed_repair_ratio=ratio,
                    real_pose_ground_truth=False,training_eligible=False,
                    usage='replacement_draft_not_released',
                    unresolved=['Port cover identity pending user clarification',
                                'Removal/occlusion edges need refinement before scene replacement'])
        samples.append(record);checks.append(dict(id=r['id'],projection_error=error,heatmaps_finite=True,
                                                 occluder_pixels_preserved=True))
        cards.append(f'<section><h2>第 {a["frame"]} 帧 · 图像左相机</h2>'
          f'<p>顶部标签面背向观察相机：{top_side_mm<0}。这是改变三维旋转后的结果，未删除或涂掉标签。</p>'
          f'<img src="{name}"><p>四列：真实帧、修正姿态、合成草稿、新三维角点投影。绿色框为合成标签。</p>'
          f'<p>暴露修补区域比例 {ratio:.1%}，仍未放行训练。线缆及银白标签的已标注区域保持原像素。</p></section>')
    (out/'audit.json').write_text(json.dumps(dict(checks=checks,samples=samples),indent=2))
    (out/'manifest.json').write_text(json.dumps(dict(status='pose_correction_draft_not_training_release',samples=samples),indent=2))
    page='''<!doctype html><meta charset="utf-8"><title>侧视姿态修正：顶部标签面不可见</title>
    <style>body{max-width:1500px;margin:30px auto;background:#14212a;color:#eee;font:17px/1.7 system-ui;padding:0 20px}img{max-width:100%}section{background:#243640;padding:20px;margin:20px 0}a{color:#8bdef5}</style>
    <h1>侧视姿态修正：顶部标签面不可见</h1>
    <p>旧候选露出了真实帧中不可见的顶部标签面，因此排除。新拟合加入该面不可见的约束；参考内参与手工对应点仍为估计值，尚不能称为精确真实姿态。</p>
    <p>本页用于核对角度。替换边缘及接口附近的遮挡尚待处理，合成草稿不会送入训练。</p>'''
    (review/'index.html').write_text(page+''.join(cards)+'<p><a href="../audit.json">投影与遮挡检查</a></p>')
    print(json.dumps(dict(rendered=len(checks),max_projection_error=max(x['projection_error'] for x in checks),training_released=0)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    main(p.parse_args().output.resolve())
