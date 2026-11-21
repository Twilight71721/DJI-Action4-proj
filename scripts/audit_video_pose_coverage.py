"""Inventory every real frame; detections/marker fits remain proposals, never pose GT."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO

ROOT=Path(__file__).resolve().parents[1]


def fit_tag(tag_id,uv,width,height):
    cx,cy,size=(17.1,4.1,29.6) if tag_id==18 else (20.3,4.6,27.7)
    d=.4*size;z=-16.24
    xyz=np.array([[cx+d,cy+d,z],[cx-d,cy+d,z],[cx-d,cy-d,z],[cx+d,cy-d,z]],float)
    K=np.array([[2000.,0,width/2],[0,2000.,height/2],[0,0,1.]])
    result=cv2.solvePnPGeneric(xyz,np.asarray(uv,float),K,None,flags=cv2.SOLVEPNP_IPPE)
    candidates=[]
    for r,t in zip(result[1],result[2]):
        r,t=cv2.solvePnPRefineLM(xyz,np.asarray(uv,float),K,None,r,t)
        R=cv2.Rodrigues(r)[0];cam=xyz@R.T+t.ravel()
        projected=cv2.projectPoints(xyz,r,t,K,None)[0][:,0]
        error=float(np.sqrt(np.mean(np.sum((projected-uv)**2,axis=1))))
        direction=-R.T@t.ravel();direction/=np.linalg.norm(direction)
        if cam[:,2].min()>0 and direction[2]<0:
            candidates.append(dict(K=K.tolist(),R=R.tolist(),t_mm=t.ravel().tolist(),
                rmse_px=error,view_direction_model=direction.tolist()))
    candidates.sort(key=lambda r:r['rmse_px'])
    return candidates


def main(out,limit):
    out.mkdir(parents=True,exist_ok=False);(out/'representatives').mkdir()
    frames=sorted((ROOT/'outputs/video_reference_audit/head_left_rgb_raw/frames_full').glob('frame_*.jpg'))
    if limit:frames=frames[:limit]
    cv2.setNumThreads(2);torch.set_num_threads(4)
    detector=YOLO(str(ROOT/'outputs/yolo11s_real48_mixed_20260922/train/weights/best.pt'))
    params=cv2.aruco.DetectorParameters();params.cornerRefinementMethod=cv2.aruco.CORNER_REFINE_SUBPIX
    tags=cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11),params)
    records=[];started=time.monotonic()
    with (out/'frames.jsonl').open('w') as stream:
        for start in range(0,len(frames),8):
            paths=frames[start:start+8]
            images=[cv2.imread(str(p)) for p in paths]
            if any(im is None for im in images):raise RuntimeError('Unreadable reference frame')
            reduced=[cv2.resize(im,(1624,1232),interpolation=cv2.INTER_AREA) for im in images]
            predictions=detector.predict(reduced,imgsz=960,conf=.15,iou=.5,max_det=4,device=0,verbose=False)
            for path,image,pred in zip(paths,images,predictions):
                frame=int(path.stem.split('_')[-1]);h,w=image.shape[:2]
                boxes=pred.boxes.xyxy.cpu().numpy()*2;scores=pred.boxes.conf.cpu().numpy()
                corners,ids,_=tags.detectMarkers(image)
                markers=[]
                if ids is not None:
                    for c,ident in zip(corners,ids.ravel()):
                        if int(ident) not in [18,19]:continue
                        uv=c[0];poses=fit_tag(int(ident),uv,w,h)
                        markers.append(dict(id=int(ident),uv=uv.tolist(),candidates=poses,
                            status='estimated_K_and_tag_geometry_not_real_GT'))
                objects=[]
                for j,(box,score) in enumerate(zip(boxes,scores)):
                    x0,y0,x1,y1=box;cx,cy=(x0+x1)/2,(y0+y1)/2
                    size=max(x1-x0,y1-y0)*1.5
                    a,b=max(0,int(cx-size/2)),max(0,int(cy-size/2))
                    c,d=min(w,int(cx+size/2)),min(h,int(cy+size/2))
                    crop=image[b:d,a:c]
                    g=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
                    attached=[m['id'] for m in markers if x0<=np.mean(m['uv'],0)[0]<=x1 and y0<=np.mean(m['uv'],0)[1]<=y1]
                    item=dict(detection_index=j,box_xyxy=box.tolist(),confidence=float(score),
                        image_side='image_left' if cx<w/2 else 'image_right',tag_ids=attached,
                        crop_xyxy=[a,b,c,d],gray_mean=float(g.mean()),laplacian_variance=float(cv2.Laplacian(g,cv2.CV_64F).var()),
                        truncated=bool(x0<4 or y0<4 or x1>w-4 or y1>h-4),
                        pose_status='marker_proposal_needs_alignment_review' if attached else 'unresolved_no_marker')
                    if frame%25==0 or frame==len(frames)-1:
                        file=f'frame_{frame:06d}_d{j}.jpg'
                        cv2.imwrite(str(out/'representatives'/file),cv2.resize(crop,(256,256),interpolation=cv2.INTER_AREA))
                        item['preview']='representatives/'+file
                    objects.append(item)
                record=dict(frame=frame,timestamp_seconds=frame/29.97002997002997,objects=objects,markers=markers,
                    count_status='two_candidates' if len(objects)==2 else 'needs_count_review',
                    anatomical_identity_known=False)
                records.append(record);stream.write(json.dumps(record)+'\n');stream.flush()
            (out/'status.json').write_text(json.dumps(dict(state='scanning',completed=len(records),total=len(frames),elapsed_seconds=time.monotonic()-started)))
            if start%80==0:print('SCANNED',len(records),'/',len(frames),flush=True)
    summary=dict(state='inventory_complete_pose_coverage_not_complete',frames=len(records),
        detected_instances=sum(len(r['objects']) for r in records),frames_with_marker=sum(bool(r['markers']) for r in records),
        marker_observations=sum(len(r['markers']) for r in records),
        frames_needing_count_review=sum(r['count_status']!='two_candidates' for r in records),
        elapsed_seconds=time.monotonic()-started)
    (out/'status.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--limit',type=int,default=0)
    a=p.parse_args();main(a.output.resolve(),a.limit)
