"""Build reviewed 2D supports plus explicitly uncertain eight-corner pose drafts.

No network prediction is used for annotation. Estimated K is used only to display
unobserved corners; those coordinates are excluded from training/evaluation.
"""
from pathlib import Path
import sys,json,hashlib
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from bop_bb8.heatmap import make_bbox_heatmaps
from bop_bb8.visualize import EDGES
from infer_resnet_yolo_real import prepare_crop


def project(xyz,R,t,K):
    return cv2.projectPoints(xyz,cv2.Rodrigues(R)[0],t,K,None)[0][:,0]


def fit_pose(xyz,uv,K,face):
    ok,r,t=cv2.solvePnP(xyz,uv,K,None,flags=cv2.SOLVEPNP_SQPNP)
    if not ok:raise ValueError('Landmark fit failed')
    r,t=cv2.solvePnPRefineLM(xyz,uv,K,None,r,t)
    candidates=[(r,t)]
    if face=='dji_side':
        # A single planar face has two plausible poses. The label side is
        # visible to the left in these three selected frames (+Y facing us).
        result=cv2.solvePnPGeneric(xyz,uv,K,None,flags=cv2.SOLVEPNP_IPPE)
        candidates.extend(zip(result[1],result[2]))
    accepted=[]
    for r,t in candidates:
        R=cv2.Rodrigues(r)[0];t=t[:,0]
        direction=-R.T@t;direction/=np.linalg.norm(direction)
        if (R@xyz.T+t[:,None])[2].min()<=0:continue
        if 'rear' in face and direction[2]>=0:continue
        if face=='rear_side' and direction[0]>=0:continue
        if face=='dji_side' and (direction[0]>=0 or direction[1]<=0):continue
        error=np.square(project(xyz,R,t,K)-uv).sum()
        accepted.append((error,R,t))
    if not accepted:raise ValueError('No candidate satisfies visible-face assumptions')
    _,R,t=min(accepted,key=lambda c:c[0])
    return R,t


def dashed(im,a,b,color):
    length=np.linalg.norm(b-a)
    for x in np.arange(0,length,12):
        start=a+(b-a)*x/max(length,1);end=a+(b-a)*min(x+6,length)/max(length,1)
        cv2.line(im,tuple(np.rint(start).astype(int)),tuple(np.rint(end).astype(int)),color,1,cv2.LINE_AA)


def draw_labels(image,points,valid):
    result=image.copy()
    for a,b in EDGES:
        if valid[a] and valid[b]:cv2.line(result,tuple(np.rint(points[a]).astype(int)),tuple(np.rint(points[b]).astype(int)),(90,230,80),2,cv2.LINE_AA)
        else:dashed(result,points[a],points[b],(20,165,255))
    for i,p in enumerate(points):
        color=(90,255,80) if valid[i] else (20,185,255)
        p=np.rint(p).astype(int);cv2.circle(result,tuple(p),5,color,-1 if valid[i] else 2,cv2.LINE_AA)
        cv2.putText(result,str(i),tuple(p+[7,-7]),cv2.FONT_HERSHEY_SIMPLEX,.65,color,2,cv2.LINE_AA)
    return result


def build(output):
    document=json.loads((output/'manual_supports.json').read_text())
    corners=np.array([c['xyz'] for c in document['corner_definition']['corners']],np.float64)
    (output/'review').mkdir(exist_ok=True);(output/'crops').mkdir(exist_ok=True)
    rows=[];rng=np.random.default_rng(20260925);tensors=[];heatmaps=[];valids=[];targets=[]
    for rec in document['records']:
        image=cv2.imread(str(output/rec['image']));assert image.shape[:2]==(2464,3248)
        assert hashlib.sha256((output/rec['image']).read_bytes()).hexdigest()==rec['image_sha256']
        xyz=np.array([s['xyz_mm'] for s in rec['supports']],np.float64)
        uv=np.array([s['uv_original'] for s in rec['supports']],np.float64)
        K=np.array([[2000.,0,1624],[0,2000.,1232],[0,0,1]])
        R,t=fit_pose(xyz,uv,K,rec['face']);fitted=project(corners,R,t,K)
        residual=project(xyz,R,t,K)-uv
        samples=[];review_crop=np.array(rec['annotation_crop_xywh'])
        # Sensitivity to unknown focal length and manually placed edge extensions.
        # This is NOT a confidence interval and excludes model-shape/distortion errors.
        for focal in [1200,1600,2000,2400,2800]:
            for trial in range(12):
                Kj=K.copy();Kj[0,0]=Kj[1,1]=focal
                uvj=uv+rng.normal(0,4*review_crop[2]/600,uv.shape)
                try:Rj,tj=fit_pose(xyz,uvj,Kj,rec['face'])
                except (ValueError,cv2.error):continue
                samples.append(project(corners,Rj,tj,Kj))
        if not samples:raise ValueError('No finite sensitivity samples')
        sensitivity=np.max(np.linalg.norm(np.array(samples)-fitted,axis=-1),axis=0)
        points=fitted.copy();valid=np.zeros(8,bool);states=['geometry_inferred_not_ground_truth']*8
        for s in rec['supports']:
            if s['kind']=='rear_aabb_edge_extension':
                i=s['reference_corner'];points[i]=s['uv_original'];valid[i]=s['supervise_corner']
                states[i]='manual_edge_extension' if valid[i] else 'manual_uncertain_excluded'
        roi,tensor,crop=prepare_crop(image,rec['box_xyxy'],(256,256),1.3)
        inp=(points-np.array(crop[:2]))*256/np.array(crop[2:])
        assert ((inp[valid]>=0)&(inp[valid]<256)).all()
        # Invalid targets have zero heatmaps AND false loss masks, never negatives.
        hm=make_bbox_heatmaps(inp/4,valid,(64,64),kernel='gaussian',sigma=2.)
        assert not hm[~valid].any()
        tensors.append(tensor.numpy());heatmaps.append(hm);targets.append(np.where(valid[:,None],inp,0).astype(np.float32));valids.append(valid)
        review=(points-review_crop[:2])*600/review_crop[2:]
        x,y,w,h=review_crop;raw=cv2.resize(image[y:y+h,x:x+w],(600,600),interpolation=cv2.INTER_LINEAR)
        drawn=draw_labels(raw,review,valid)
        supports=raw.copy()
        for s in rec['supports']:
            q=np.array(s['uv_review'],int);cv2.circle(supports,tuple(q),4,(255,220,30),-1)
            name=str(s['reference_corner'])+(' rim' if s['kind']=='front_body_rim_not_aabb' else '')
            cv2.putText(supports,name,tuple(q+[6,-6]),0,.48,(255,220,30),1,cv2.LINE_AA)
        panel=np.hstack([raw,supports,drawn]);panel=cv2.copyMakeBorder(panel,36,0,0,0,cv2.BORDER_CONSTANT,value=(20,25,30))
        cv2.putText(panel,f'{rec["id"]}: REAL | MANUAL SURFACE SUPPORTS | CORNERS: GREEN = SUPERVISED, ORANGE = EXCLUDED',(10,25),0,.60,(255,255,255),1,cv2.LINE_AA)
        cv2.imwrite(str(output/'review'/f'{rec["id"]}.jpg'),panel,[cv2.IMWRITE_JPEG_QUALITY,95])
        cv2.imwrite(str(output/'crops'/f'{rec["id"]}.png'),roi)
        rec.update(crop_xywh=list(crop),keypoints_original=points.tolist(),keypoints_input=inp.tolist(),valid=valid.tolist(),point_status=states,geometry_draft=dict(K=K.tolist(),R=R.tolist(),t_mm=t.tolist(),corners_original=fitted.tolist(),landmark_rmse_review_px=float(np.sqrt(np.mean(np.sum(residual**2,axis=-1)))*600/review_crop[2]),sensitivity_max_input_px=(sensitivity*256/crop[2]).tolist(),sensitivity_trials=len(samples),calibrated=False,visibility_prior='DJI side -X and label side +Y face observer' if rec['face']=='dji_side' else 'Manually identified rear/side faces face observer'),status='assistant_reviewed_partial_2d_labels_with_geometry_draft',annotation_noise_note='Rounded rims are extended to ideal box edges; manual 2D labels are approximate, not calibrated pose GT.')
        rows.append(rec)
        print(rec['id'],'valid',np.flatnonzero(valid).tolist(),'fit RMSE in review pixels',rec['geometry_draft']['landmark_rmse_review_px'],flush=True)
    result=dict(schema_version=1,corner_definition=document['corner_definition'],protocol='Partial manual real annotations. Eight coordinates are stored per object; only valid=true may enter loss/metrics. All geometry-only hidden corners excluded. Same video has been used for background adaptation; test labels are held out but this is not a blind external-video test.',split_note='0 and 100 share one time group in adapt; 475 is also adapt. The entire 1350-1549 block is validation; 2349 is test. Pilot only: five adapt objects, three validation, one test; inadequate for a final generalization claim.',input_size=[256,256],heatmap_size=[64,64],crop_padding=1.3,records=rows)
    (output/'annotations.json').write_text(json.dumps(result,indent=2))
    np.savez_compressed(output/'partial_supervision.npz',image=np.array(tensors),heatmap=np.array(heatmaps),valid=np.array(valids),keypoints_2d=np.array(targets),id=np.array([r['id'] for r in rows]),split=np.array([r['split'] for r in rows]))
    assert len({(r['time_group'],r['split']) for r in rows})==len({r['time_group'] for r in rows})
    assert len({r['image_sha256'] for r in rows})==len({r['frame'] for r in rows})
    return result


if __name__=='__main__':
    build(ROOT/'dataset/action4-real-corners-pilot-20260925')
