"""Select diverse marker-supported pose proposals across the complete video."""
import argparse
import json
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np
from scipy.optimize import brentq

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from bop_bb8 import BOPModel
from render_pose_replacement_pilot import rotation


def recenter(R,t):
    forward=np.asarray(t)/np.linalg.norm(t)
    right=np.cross(forward,[0,-1,0]);right/=np.linalg.norm(right)
    C=np.array([right,np.cross(forward,right),forward])
    return C@np.asarray(R)


def angular_distance(A,B):
    return np.degrees(np.arccos(np.clip((np.trace(A@B.T)-1)/2,-1,1)))


def select_marker_candidate(marker,frame):
    """Use inspected side visibility to reject the planar mirror solution.

    The lowered-arm interval was inspected in the video contact sheets. The
    user identified the near-lens port side and the opposite DJI-marked side.
    This prior is specific to this video, not a general marker-pose rule.
    """
    candidates=[]
    for index,fit in enumerate(marker['candidates']):
        if fit['rmse_px']>3:continue
        direction=-np.asarray(fit['R']).T@np.asarray(fit['t_mm'])
        direction/=np.linalg.norm(direction)
        if frame>=113:
            sign=1 if marker['id']==18 else -1
            if direction[0]*sign<=.05:continue
        candidates.append((index,fit))
    return min(candidates,key=lambda pair:pair[1]['rmse_px']) if candidates else None


def main(inventory,out):
    out.mkdir(parents=True,exist_ok=False)
    shutil.copytree(ROOT/'outputs/pose_replacement_pilot_20260924/models',out/'models')
    (out/'source_previews').mkdir()
    model=BOPModel.from_models_info(out/'models/models_info.json')
    rows=[json.loads(s) for s in (inventory/'frames.jsonl').read_text().splitlines()]
    groups=[];selected_all=[]
    for tag in [18,19]:
        candidates=[]
        for row in rows:
            for m in row['markers']:
                if m['id']!=tag or not m['candidates']:continue
                selected_fit=select_marker_candidate(m,row['frame'])
                if selected_fit is None:continue
                index,fit=selected_fit
                candidates.append(dict(frame=row['frame'],tag=tag,fit=fit,tag_uv=m['uv'],candidate_index=index,
                                       selection='side_visibility_prior_after_frame_112_then_reprojection',
                                       R_centered=recenter(fit['R'],fit['t_mm']).tolist()))
        selected=[min(candidates,key=lambda c:c['frame'])]
        while len(selected)<24:
            distances=[min(angular_distance(np.array(c['R_centered']),np.array(s['R_centered'])) for s in selected) for c in candidates]
            if max(distances)<8:break
            selected.append(candidates[int(np.argmax(distances))])
        side='left' if tag==18 else 'right';records=[]
        for j,c in enumerate(sorted(selected,key=lambda c:c['frame'])):
            frame=c['frame'];row=rows[frame]
            image=cv2.imread(str(ROOT/f'outputs/video_reference_audit/head_left_rgb_raw/frames_full/frame_{frame:06d}.jpg'))
            candidates_box=[o for o in row['objects'] if tag in o['tag_ids']]
            if candidates_box:
                a,b,d,e=max(candidates_box,key=lambda x:x['confidence'])['crop_xyxy']
            else:
                uv=np.array(c['tag_uv']);lo=uv.min(0);hi=uv.max(0);center=(lo+hi)/2
                size=max(hi-lo)*4;a,b=np.maximum(0,center-size/2).astype(int);d,e=np.minimum([image.shape[1],image.shape[0]],center+size/2).astype(int)
            preview=f'source_previews/{side}_{frame:06d}.jpg'
            cv2.imwrite(str(out/preview),cv2.resize(image[b:e,a:d],(512,512),interpolation=cv2.INTER_AREA))
            c['source_preview']=preview;selected_all.append(c)
            for v,delta in enumerate([[0,0,0],[4,-4,5],[-4,4,-5]]):
                R=rotation(delta)@np.array(c['R_centered'])
                xyz=model.corners_mm()@R.T
                def extent(z):
                    p=xyz+[0,0,z];uv=700*p[:,:2]/p[:,2:]
                    return max(np.ptp(uv,axis=0))-325
                z=float(brentq(extent,80,500))
                records.append(dict(id=f'video_{tag}_{frame:06d}_{v}',kind='foreground',
                    family=f'video_{frame:06d}_tag{tag}',source_group='video_adaptation_all',
                    R=R.tolist(),t_mm=[0,0,z],delta_degrees=delta,light_energy=[8.,10.,12.][v],exposure=[-.4,-.6,-.2][v],
                    reference_frame=frame,reference_tag=tag,source_preview=preview,
                    pose_proposal_method='planar marker, estimated K and tag mounting; view recentered for isolated rendering',
                    real_pose_ground_truth=False,marker_candidate_index=c['candidate_index'],
                    candidate_selection=c['selection']))
        groups.append(dict(id=f'video_{side}',side=side,K=[[700,0,256],[0,700,256],[0,0,1]],records=records))
    plan=dict(width=512,height=512,seed=20260924,glb=str(ROOT/'dataset/action4-v6-domain-pilot-200/models/obj_000001.glb'),
        groups=groups,anchors=[],status='marker_supported_foregrounds_only_unresolved_views_not_covered',
        selected_sources=selected_all,source_frames_scanned=len(rows),
        marker_ambiguity_filter='User-confirmed port side +X for tag18 and DJI side -X for tag19; applies to inspected lowered-arm interval 113..2849.',
        caveat='Marker fit accuracy is not real pose accuracy; no scene replacement is performed here.')
    (out/'plan.json').write_text(json.dumps(plan,indent=2))
    print(json.dumps(dict(selected_references=len(selected_all),renders=sum(len(g['records']) for g in groups))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inventory',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.inventory.resolve(),a.output.resolve())
