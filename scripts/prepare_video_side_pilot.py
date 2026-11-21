"""Fit user-identified side faces; all real poses remain estimates, not GT."""
import argparse
import json
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bop_bb8.geometry import crop_and_scale_intrinsics


def fit_hidden_top(xyz,uv,K,initial_R,initial_t):
    """Fit a near-lens side while keeping the +Y label face back-facing."""
    def centered_basis(t):
        forward=t/np.linalg.norm(t)
        right=np.cross(forward,[0,-1,0]);right/=np.linalg.norm(right)
        return np.array([right,np.cross(forward,right),forward])
    def orientation(dy,dz,roll,t):
        direction=np.array([1.,dy,dz]);direction/=np.linalg.norm(direction)
        right=np.cross([0,0,1],direction);right/=np.linalg.norm(right)
        B=np.array([right,np.cross(-direction,right),-direction])
        c,s=np.cos(roll),np.sin(roll)
        return centered_basis(t).T@np.array([[c,-s,0],[s,c,0],[0,0,1]])@B
    def residual(p):
        R=orientation(*p[:3],p[3:])
        projected=cv2.projectPoints(xyz,cv2.Rodrigues(R)[0],p[3:],K,None)[0][:,0]
        return (projected-uv).ravel()
    best=None
    for dy in [-.05,-.2,-.5]:
        t=np.array(initial_t,float)
        base=orientation(dy,.2,0,t)
        twist=centered_basis(t)@initial_R@base.T@centered_basis(t).T
        roll=np.arctan2(twist[1,0],twist[0,0])
        start=np.r_[dy,.2,roll,t]
        fit=least_squares(residual,start,bounds=([-1.5,0,-np.pi,-2000,-2000,80],
                                               [-.025,1.5,np.pi,2000,2000,2000]),max_nfev=300)
        if best is None or fit.cost<best.cost:best=fit
    return orientation(*best.x[:3],best.x[3:]),best.x[3:]


def main(out,hide_top=False):
    out.mkdir(exist_ok=False)
    (out / 'sources').mkdir()
    shutil.copytree(ROOT / 'outputs/pose_replacement_pilot_20260924/models', out / 'models')
    annotations = [
        dict(frame=125, side='left', face_x=35.0, crop_xywh=[1248,1649,534,534],
             quad=[[164,218],[403,180],[421,340],[175,387]], lens=[313,154],
             removal=[[124,169],[246,142],[352,134],[392,144],[403,180],[422,338],
                      [405,355],[177,391],[137,373],[126,273],[95,251],[99,207]],
             occluder=[[0,267],[67,260],[108,247],[160,267],[181,296],[285,310],
                       [311,316],[315,343],[288,357],[175,378],[140,377],[96,389],[0,444]]),
        dict(frame=500, side='left', face_x=35.0, crop_xywh=[1379,1641,412,412],
             quad=[[192,220],[438,185],[449,349],[196,369]], lens=[335,157],
             removal=[[99,169],[241,165],[269,142],[403,142],[430,162],[447,188],
                      [450,343],[433,354],[193,374],[143,352],[100,328]],
             occluder=[[0,214],[104,205],[166,226],[231,243],[335,243],[363,282],
                       [371,319],[238,335],[182,337],[145,346],[0,366]]),
        dict(frame=475, side='right', face_x=-35.0, crop_xywh=[1885,1633,449,449],
             quad=[[126,177],[372,186],[394,371],[126,350]], lens=None,
             removal=[[101,146],[218,142],[264,156],[360,176],[381,186],[399,343],
                      [393,372],[365,378],[126,355],[106,339],[81,246],[90,186]],
             occluder=[]),
    ]
    K = np.array([[2000.,0,1624],[0,2000.,1232],[0,0,1]])
    groups = []
    for i,a in enumerate(annotations):
        x,y,w,h = a['crop_xywh']
        image = cv2.imread(str(ROOT / f'outputs/video_reference_audit/head_left_rgb_raw/frames_full/frame_{a["frame"]:06d}.jpg'))
        crop = cv2.resize(image[y:y+h,x:x+w], (512,512), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out/'sources'/f'anchor_{i}.png'),crop)
        cam = crop_and_scale_intrinsics(K,a['crop_xywh'],(512,512))
        ys = [-21,21,21,-21] if a['face_x']>0 else [21,-21,-21,21]
        xyz = np.array([[a['face_x'],yy,zz] for yy,zz in zip(ys,[11.84,11.84,-15.8,-15.8])],float)
        uv = np.array(a['quad'],float)
        if a['lens'] is not None:
            xyz = np.vstack([xyz,[15,3,15.8]])
            uv = np.vstack([uv,a['lens']])
        ok,r,t = cv2.solvePnP(xyz,uv,cam,None,flags=cv2.SOLVEPNP_SQPNP)
        if not ok: raise RuntimeError('Side-face PnP failed')
        r,t = cv2.solvePnPRefineLM(xyz,uv,cam,None,r,t)
        R = cv2.Rodrigues(r)[0]
        if hide_top and a['side']=='left':
            R,translation=fit_hidden_top(xyz,uv,cam,R,t[:,0])
            t=translation[:,None];r=cv2.Rodrigues(R)[0]
        direction = -R.T@t[:,0]; direction /= np.linalg.norm(direction)
        assert direction[0]*a['face_x']>0 and np.linalg.det(R)>.999
        projected = cv2.projectPoints(xyz,r,t,cam,None)[0][:,0]
        a.update(K=cam.tolist(),R=R.tolist(),t_mm=t[:,0].tolist(),
                 landmarks_xyz_mm=xyz.tolist(),landmarks_uv=uv.tolist(),
                 landmark_rmse_px=float(np.sqrt(np.mean(np.sum((uv-projected)**2,axis=1)))),
                 view_direction_model=direction.tolist(),real_pose_ground_truth=False,
                 status='candidate_pending_visual_inspection')
        if hide_top and a['side']=='left':
            assert direction[1]<0
            a['visibility_constraint']='Top +Y label face must be back-facing, as identified by user'
        for name,poly in [('removal',a['removal']),('occluder',a['occluder'])]:
            mask=np.zeros((512,512),np.uint8)
            if poly: cv2.fillPoly(mask,[np.array(poly,np.int32)],255)
            cv2.imwrite(str(out/'sources'/f'anchor_{i}_{name}.png'),mask)
        record=dict(id=f'side_{a["frame"]:06d}_{a["side"]}',kind='replacement',anchor=i,
                    family='port_side' if a['side']=='left' else 'dji_side',
                    source_group='video_adaptation_all',R=R.tolist(),t_mm=t[:,0].tolist(),
                    delta_degrees=[0,0,0],light_energy=7.,exposure=-.65)
        groups.append(dict(id=f'side_anchor_{i}',side=a['side'],K=cam.tolist(),records=[record]))
    plan=dict(glb=str(ROOT/'dataset/action4-v6-domain-pilot-200/models/obj_000001.glb'),
              groups=groups,anchors=annotations,corner_order='current bop_bb8.geometry.CORNER_SIGNS',
              width=512,height=512,reference_calibrated=False)
    (out/'plan.json').write_text(json.dumps(plan,indent=2))
    print([(a['frame'],a['landmark_rmse_px']) for a in annotations])


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=ROOT/'outputs/video_side_pose_pilot_20260924')
    p.add_argument('--hide-top',action='store_true')
    a=p.parse_args();main(a.output.resolve(),a.hide_top)
