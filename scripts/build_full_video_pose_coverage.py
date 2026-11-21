"""Video-wide view catalogue and bounded synthetic orientation neighborhoods.

Time coverage is tracked separately from uncertain real pose estimates. No
interpolated box or visually assigned view bin is exported as real pose GT.
"""
import argparse
from collections import Counter
import csv
import itertools
import json
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np
from scipy.optimize import brentq
from scipy.spatial.transform import Rotation, Slerp

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from bop_bb8 import BOPModel
from prepare_video_pose_candidates import recenter, select_marker_candidate, angular_distance
from render_pose_replacement_pilot import rotation

# Ordered physical devices: port side/tag18, DJI side/tag19. Labels assigned
# by inspecting all ten time contact sheets; image-half labels are not used.
# R rear; B/b rear-side with large/small rear-screen exposure;
# S edge-on side; f/F side-front with small/large front-plane exposure.
VIEW_ROWS='''
0 RR 25 RR 50 RR 75 RR 100 RR
125 fF 150 SS 175 bb 200 bB 225 bb 250 Sb 275 bB
300 SS 325 Bf 350 SS 375 fS 400 Ff 425 Sf 450 Sf 475 SS 500 SF 525 bS 550 Bb 575 SS
600 SF 625 fF 650 SS 675 SS 700 SS 725 SS 750 SS 775 SF 800 FF 825 SF 850 Sf 875 fF
900 SF 925 fF 950 fF 975 fF 1000 FF 1025 fF 1050 fF 1075 bF 1100 Sf 1125 fF 1150 Bb 1175 Bf
1200 Rf 1225 Bb 1250 bf 1275 SS 1300 BF 1325 Bf 1350 BB 1375 bb 1400 bb 1425 Sb 1450 SS 1475 SS
1500 SS 1525 bS 1550 bS 1575 BF 1600 bf 1625 bS 1650 bb 1675 bS 1700 SS 1725 BS 1750 bf 1775 bf
1800 fF 1825 Sf 1850 SS 1875 SS 1900 bS 1925 SF 1950 BS 1975 bS 2000 BS 2025 BS 2050 Bf 2075 bS
2100 bb 2125 SF 2150 Sf 2175 bF 2200 SF 2225 Sf 2250 SS 2275 SS 2300 Bb 2325 Bb 2350 bB 2375 bF
2400 BS 2425 Rf 2450 SF 2475 BB 2500 BB 2525 bS 2550 Bb 2575 BB 2600 BB 2625 Bf 2650 Bf 2675 BB
2700 BB 2725 BB 2750 bB 2775 fF 2800 fF 2825 fF 2849 fF
'''
ROLL_LEFT={175:-10,200:-10,225:-10,275:-10,300:-10,325:-25,375:-15,400:15,
           800:15,1000:15,1025:10,1300:-15,1350:-10,2300:-25,2325:-25,
           2400:-25,2425:-25,2475:20,2500:15,2550:-25,2575:-20,2600:-15,
           2625:-25,2650:-15,2675:-30,2700:-30,2725:-30,2750:15,2825:10,2849:15}
ROLL_RIGHT={200:10,225:-10,325:-20,400:10,450:5,525:10,750:15,1000:10,
            1100:5,1200:-30,1250:-15,1300:-15,1325:10,1350:20,1400:-10,
            1450:5,1925:-30,2175:-20,2200:-15,2300:20,2350:25,2500:15,
            2550:25,2575:15,2600:25,2675:35,2750:35,2800:25,2825:35,2849:35}
NAMES={'R':'rear_screen','B':'side_rear_large','b':'side_rear_small',
       'S':'side_edge_on','f':'side_front_small','F':'side_front_large'}


def box_cost(a,b):
    if a is None or b is None:return .6 if a is not b else 0.
    a=np.array(a['box_xyxy']);b=np.array(b['box_xyxy'])
    sa=np.maximum(a[2:]-a[:2],1);sb=np.maximum(b[2:]-b[:2],1)
    return float(5*np.linalg.norm((a[:2]+a[2:]-b[:2]-b[2:])/2)/max(sa.max(),sb.max(),120)
                 +.5*np.abs(np.log(sa/sb)).sum())


def associate(rows):
    """Global continuity plus tag identity; missing detections stay explicit."""
    history=[];previous=[];scores=None
    for row in rows:
        objects=row['objects'];states=[];unary=[]
        for a,b in itertools.product([None]+list(range(len(objects))),repeat=2):
            if a is not None and a==b:continue
            pair=[None if k is None else objects[k] for k in [a,b]]
            if all(o is not None for o in pair):
                centers=[sum(o['box_xyxy'][::2])/2 for o in pair]
                if centers[0]>=centers[1]:continue
            cost=0.
            for tag,o in zip([18,19],pair):
                if o is None:cost+=4.;continue
                cost+=-np.log(max(o['confidence'],.001))
                if (37-tag) in o['tag_ids']:cost+=100
                if tag in o['tag_ids']:cost-=1
            states.append((a,b));unary.append(cost)
        if scores is None:
            new=np.array(unary);parents=np.full(len(states),-1)
        else:
            costs=np.array([[sum(box_cost(previous_objects[k] if k is not None else None,
                                          objects[l] if l is not None else None)
                                for k,l in zip(old,current)) for old in previous] for current in states])
            weighted=costs+scores[None,:];parents=np.argmin(weighted,axis=1)
            new=np.array(unary)+weighted[np.arange(len(states)),parents]
        history.append((states,parents));previous=states;scores=new;previous_objects=objects
    index=int(np.argmin(scores));assignments=[]
    for states,parents in reversed(history):
        assignments.append(states[index]);index=int(parents[index])
    assignments.reverse()
    tracks=[]
    for side in range(2):
        observed={f:rows[f]['objects'][pair[side]] for f,pair in enumerate(assignments) if pair[side] is not None}
        fs=np.array(sorted(observed));boxes=np.array([observed[f]['box_xyxy'] for f in fs])
        interpolated=np.stack([np.interp(np.arange(len(rows)),fs,boxes[:,j]) for j in range(4)],axis=1)
        tracks.append([dict(box_xyxy=interpolated[f].tolist(),detection=observed.get(f),
                           box_status='detected_proposal' if f in observed else 'interpolated_missing_detection_not_GT') for f in range(len(rows))])
    return tracks


def canonical_view(side,code,roll):
    z={'B':-.95,'b':-.4,'S':.06,'f':.35,'F':.95,'R':-2.5}[code]
    d=np.array([1 if side==0 else -1,-.38 if side==0 else -.2,z],float);d/=np.linalg.norm(d)
    right=np.cross([0,0,1],d);right/=np.linalg.norm(right)
    return rotation([0,0,roll])@np.array([right,np.cross(-d,right),-d])


def padded_crop(image,box):
    box=np.array(box);center=(box[:2]+box[2:])/2;size=int(max(box[2:]-box[:2])*1.5)
    x,y=np.floor(center-size/2).astype(int);h,w=image.shape[:2]
    dst=np.full((size,size,3),80,np.uint8)
    xa,ya,xb,yb=max(0,x),max(0,y),min(w,x+size),min(h,y+size)
    if xb>xa and yb>ya:dst[ya-y:yb-y,xa-x:xb-x]=image[ya:yb,xa:xb]
    return cv2.resize(dst,(512,512),interpolation=cv2.INTER_AREA)


def prepare(out):
    out.mkdir(exist_ok=False);(out/'source_previews').mkdir()
    shutil.copytree(ROOT/'outputs/pose_replacement_pilot_20260924/models',out/'models')
    rows=[json.loads(s) for s in (ROOT/'outputs/video_pose_coverage_20260924/frames.jsonl').read_text().splitlines()]
    tracks=associate(rows)
    tokens=VIEW_ROWS.split();views={int(tokens[i]):tokens[i+1] for i in range(0,len(tokens),2)}
    assert sorted(views)==list(range(0,2850,25))+[2849]
    marker_plan=json.loads((ROOT/'outputs/video_pose_foregrounds_sidechecked_20260924/plan.json').read_text())
    keyframes=sorted(set(views)|{s['frame'] for s in marker_plan['selected_sources']}|{108,110,112,114,116,118,120,122})
    model=BOPModel.from_models_info(out/'models/models_info.json')
    accepted=json.loads((ROOT/'outputs/video_side_pose_user20up2_20260924/plan.json').read_text())['groups'][0]['records'][0]
    templates=[[],[]];references=[]
    for f in keyframes:
        image=cv2.imread(str(ROOT/f'outputs/video_reference_audit/head_left_rgb_raw/frames_full/frame_{f:06d}.jpg'))
        for side,tag in enumerate([18,19]):
            nearest=min(views,key=lambda k:abs(k-f));code=views[nearest][side]
            roll=(ROLL_LEFT if side==0 else ROLL_RIGHT).get(nearest,0)
            selected=None
            for marker in rows[f]['markers']:
                if marker['id']==tag:selected=select_marker_candidate(marker,f)
            method='contact_sheet_view_bin_with_orientation_neighborhood'
            if selected:
                _,fit=selected;R=recenter(fit['R'],fit['t_mm']);method='marker_side_checked_estimate'
            elif f==500 and side==0:
                R=recenter(accepted['R'],accepted['t_mm']);method='user_retained_20deg_left_2deg_up'
            else:R=canonical_view(side,code,roll)
            distances=[angular_distance(R,np.array(t['R'])) for t in templates[side]]
            # Deduplicate only same nominal view family. The bin estimates are
            # not measurements; metadata retains their provenance.
            eligible=[i for i,t in enumerate(templates[side]) if t['code']==code and distances[i]<5]
            if eligible:
                k=min(eligible,key=lambda i:distances[i]);error=distances[k]
            else:
                k=len(templates[side]);error=0
                templates[side].append(dict(id=f'view_{tag}_{k:03d}',R=R.tolist(),code=code,method=method,
                    source_frame=f,reference_frames=[]))
            template=templates[side][k];template['reference_frames'].append(f)
            preview=f'source_previews/frame_{f:06d}_tag{tag}.jpg'
            cv2.imwrite(str(out/preview),padded_crop(image,tracks[side][f]['box_xyxy']))
            references.append(dict(frame=f,device=f'tag{tag}',side='left' if side==0 else 'right',
                family=NAMES[code],view_code=code,template=template['id'],source_preview=preview,
                pose_method=method,nearest_template_angle_deg=error,real_pose_ground_truth=False,
                box_status=tracks[side][f]['box_status'],
                visual_review='assistant_contact_sheet_family_review_not_precise_pose_confirmation'))
    groups=[]
    for side,ts in enumerate(templates):
        records=[]
        for template in ts:
            reference=next(r for r in references if r['template']==template['id'])
            # Wider than marker-fit perturbations because unmarked side views
            # only identify a neighborhood, not a unique calibrated rotation.
            deltas=[[0,0,0],[8,-8,-12],[-8,8,12]]
            for v,delta in enumerate(deltas):
                R=rotation(delta)@np.array(template['R']);xyz=model.corners_mm()@R.T
                z=float(brentq(lambda z:np.ptp(700*xyz[:,:2]/(xyz[:,2:]+z),axis=0).max()-325,80,600))
                records.append(dict(id=f'{template["id"]}_v{v}',kind='foreground',family=NAMES[template['code']],
                    source_group='video_adaptation_all',R=R.tolist(),t_mm=[0,0,z],delta_degrees=delta,
                    light_energy=[7.,10.,13.][v],exposure=[-.65,-.45,-.2][v],source_preview=reference['source_preview'],
                    reference_frame=reference['frame'],reference_frames=template['reference_frames'],
                    template=template['id'],pose_proposal_method=template['method'],real_pose_ground_truth=False))
        for start in range(0,len(records),60):
            groups.append(dict(id=f'full_{side}_{start//60:02d}',side='left' if side==0 else 'right',
                K=[[700,0,256],[0,700,256],[0,0,1]],records=records[start:start+60]))
    plan=dict(width=512,height=512,glb=str(ROOT/'dataset/action4-v6-domain-pilot-200/models/obj_000001.glb'),
        groups=groups,anchors=[],templates=sum(templates,[]),references=references,
        status='full_time_reference_catalogue_render_pending',source_frames=2850,
        full_pose_accuracy_verified=False,corner_order='current bop_bb8.geometry.CORNER_SIGNS',
        definitions=dict(time_coverage='Every frame assigned to nearest reviewed keyframe; not per-frame pose fitting.',
                         pose_family_coverage='Reviewed visible-face families represented by synthetic orientation neighborhoods.',
                         true_pose_coverage='Cannot certify without calibrated/annotated real poses.'))
    (out/'plan.json').write_text(json.dumps(plan,indent=2))
    assignments=[]
    for f,row in enumerate(rows):
        nearest=min(keyframes,key=lambda k:abs(k-f))
        refs=[r for r in references if r['frame']==nearest]
        for side,r in enumerate(refs):
            assignments.append(dict(frame=f,timestamp_seconds=row['timestamp_seconds'],device=r['device'],
                reference_frame=nearest,reference_distance_frames=abs(f-nearest),family=r['family'],
                template=r['template'],box_status=tracks[side][f]['box_status'],
                status='temporal_family_assignment_not_per_frame_pose_GT'))
    with (out/'frame_coverage.jsonl').open('w') as stream:
        for r in assignments:stream.write(json.dumps(r)+'\n')
    (out/'tracked_boxes.json').write_text(json.dumps(tracks))
    summary=dict(frames=len(rows),keyframes=len(keyframes),reference_views=len(references),
        templates=sum(map(len,templates)),renders=sum(len(g['records']) for g in groups),
        families=dict(Counter(r['family'] for r in references)),
        missing_detection_assignments=sum(r['box_status'].startswith('interpolated') for r in assignments),
        max_reference_distance_frames=max(r['reference_distance_frames'] for r in assignments),
        exact_real_pose_coverage_verified=False)
    (out/'preparation.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))


def refine_transition(out):
    """Bridge the fast lowering motion, where nearest coarse bins switch faces."""
    plan=json.loads((out/'plan.json').read_text())
    if any(g['id'].startswith('transition_') for g in plan['groups']):
        raise ValueError('Transition refinement already present')
    rows=[json.loads(s) for s in (ROOT/'outputs/video_pose_coverage_20260924/frames.jsonl').read_text().splitlines()]
    model=BOPModel.from_models_info(out/'models/models_info.json')
    old=json.loads((ROOT/'outputs/video_side_pose_leftturn_20260924/plan.json').read_text())
    left125=next(r for r in old['groups'][0]['records'] if r['yaw_left_degrees']==12 and r['roll_ccw_degrees']==0)
    for side,tag in enumerate([18,19]):
        knots={}
        for f in [105,112,113]:
            m=next((m for m in rows[f]['markers'] if m['id']==tag),None)
            fit=select_marker_candidate(m,f) if m else None
            if fit:knots[f]=recenter(fit[1]['R'],fit[1]['t_mm'])
        if side==0:knots[125]=recenter(left125['R'],left125['t_mm'])
        else:knots[125]=canonical_view(1,'F',0)
        times=sorted(knots);interpolate=Slerp(times,Rotation.from_matrix([knots[f] for f in times]))
        records=[]
        for ref in plan['references']:
            f=ref['frame']
            if ref['device']!=f'tag{tag}' or not 105<f<=125 or f in knots and f!=125:continue
            if f==125 and side==1:continue
            R=interpolate(f).as_matrix();z=-R[2,2]
            code='B' if z<-.65 else 'b' if z<-.15 else 'S' if z<.15 else 'f' if z<.65 else 'F'
            tid=f'transition_{tag}_{f:06d}'
            method='manual_refined_reference_not_user_confirmed' if f==125 else 'transition_slerp_estimate_not_measured'
            ref.update(template=tid,family=NAMES[code],view_code=code,pose_method=method,nearest_template_angle_deg=0)
            plan['templates'].append(dict(id=tid,R=R.tolist(),code=code,method=method,source_frame=f,reference_frames=[f]))
            for v,delta in enumerate([[0,0,0],[8,-8,-12],[-8,8,12]]):
                rv=rotation(delta)@R;xyz=model.corners_mm()@rv.T
                distance=float(brentq(lambda z:np.ptp(700*xyz[:,:2]/(xyz[:,2:]+z),axis=0).max()-325,80,600))
                records.append(dict(id=f'{tid}_v{v}',kind='foreground',family=NAMES[code],
                    source_group='video_adaptation_all',R=rv.tolist(),t_mm=[0,0,distance],delta_degrees=delta,
                    light_energy=[7.,10.,13.][v],exposure=[-.65,-.45,-.2][v],source_preview=ref['source_preview'],
                    reference_frame=f,reference_frames=[f],template=tid,pose_proposal_method=method,real_pose_ground_truth=False))
        plan['groups'].append(dict(id=f'transition_{side}',side='left' if side==0 else 'right',
            K=[[700,0,256],[0,700,256],[0,0,1]],records=records))
    plan['transition_note']='Slerp bridges marker-supported 105/112/113 and side-view 125; interpolation is a proposal, not measured real GT.'
    (out/'plan.json').write_text(json.dumps(plan,indent=2))
    refs={(r['frame'],r['device']):r for r in plan['references']}
    coverage=[json.loads(s) for s in (out/'frame_coverage.jsonl').read_text().splitlines()]
    for row in coverage:
        ref=refs[row['reference_frame'],row['device']];row.update(template=ref['template'],family=ref['family'])
    (out/'frame_coverage.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in coverage))
    summary=json.loads((out/'preparation.json').read_text())
    summary.update(templates=len(plan['templates']),renders=sum(len(g['records']) for g in plan['groups']),
                   families=dict(Counter(r['family'] for r in plan['references'])),transition_refinement_applied=True)
    (out/'preparation.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(dict(templates=len(plan['templates']),renders=sum(len(g['records']) for g in plan['groups']))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--refine-transition',action='store_true');args=p.parse_args()
    (refine_transition if args.refine_transition else prepare)(args.output.resolve())
