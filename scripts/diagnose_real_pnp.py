"""Real-frame PnP sensitivity study with explicit, uncalibrated camera hypotheses."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from bop_bb8 import BOPModel
from bop_bb8.pnp import fit_perspective_cuboid, validate_camera
from bop_bb8.visualize import EDGES


def all_point_fit(corners,points,K):
    """Illustrate an all-point least-squares solution, never label it accepted."""
    corners=np.ascontiguousarray(corners,dtype=np.float64)
    points=np.ascontiguousarray(points,dtype=np.float64)
    K,_=validate_camera(K,())
    base=dict(diagnostic_only=True,accepted=False,points=None)
    if corners.shape!=(8,3) or points.shape!=(8,2) or not np.isfinite(points).all():
        raise ValueError('Expected 8 finite correspondences')
    if np.linalg.matrix_rank(points-points.mean(0))<2:
        return dict(base,status='collapsed_observations')
    candidates=[]
    for flag in [cv2.SOLVEPNP_SQPNP,cv2.SOLVEPNP_EPNP,cv2.SOLVEPNP_ITERATIVE]:
        try:
            ok,rvec,tvec=cv2.solvePnP(corners,points,K,None,flags=flag)
            if not ok:continue
            rvec,tvec=cv2.solvePnPRefineLM(corners,points,K,None,rvec,tvec)
            rotation=cv2.Rodrigues(rvec)[0]
            if ((corners@rotation.T+tvec.reshape(1,3))[:,2]<=0).any():continue
            projected=cv2.projectPoints(corners,rvec,tvec,K,None)[0].reshape(8,2)
            if not np.isfinite(projected).all():continue
            errors=np.linalg.norm(projected-points,axis=1)
            candidates.append(dict(status='unfiltered_all8_fit_not_validated',points=projected.tolist(),
                rvec=rvec.ravel().tolist(),tvec=tvec.ravel().tolist(),initial_solver=flag,
                mean_residual_px=float(errors.mean()),rmse_px=float(np.sqrt(np.mean(errors**2)))))
        except cv2.error:
            continue
    if not candidates:return dict(base,status='no_positive_depth_solution')
    return dict(base,**min(candidates,key=lambda c:c['rmse_px']))


def panel(image,box,raw,fit,viewport,title,color):
    left,top,right,bottom=viewport
    scale=min(380/(right-left),380/(bottom-top))
    w,h=max(1,round((right-left)*scale)),max(1,round((bottom-top)*scale))
    matrix=np.array([[scale,0,-left*scale],[0,scale,-top*scale]],float)
    canvas=cv2.warpAffine(image,matrix,(w,h),borderValue=(30,30,30))
    bbox=np.rint((np.array(box).reshape(2,2)-[left,top])*scale).astype(int)
    cv2.rectangle(canvas,tuple(bbox[0]),tuple(bbox[1]),(230,180,40),1)
    rp=np.rint((raw-[left,top])*scale).astype(int)
    for p in rp:cv2.circle(canvas,tuple(p),2,(130,130,130),-1)
    if fit is not None:
        fp=np.rint((fit-[left,top])*scale).astype(int)
        for a,b in EDGES:cv2.line(canvas,tuple(fp[a]),tuple(fp[b]),color,1,cv2.LINE_AA)
        for i,p in enumerate(fp):
            cv2.circle(canvas,tuple(p),3,color,-1)
            cv2.putText(canvas,str(i),tuple(p+[3,-3]),cv2.FONT_HERSHEY_SIMPLEX,.37,color,1,cv2.LINE_AA)
    else:cv2.putText(canvas,'NO ACCEPTED POSE',(10,35),cv2.FONT_HERSHEY_SIMPLEX,.48,(80,80,255),1)
    header=np.full((35,w,3),25,np.uint8)
    cv2.putText(header,title,(7,23),cv2.FONT_HERSHEY_SIMPLEX,.46,color,1,cv2.LINE_AA)
    return np.vstack([header,canvas])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'panels').mkdir()
    path=args.result/'real/network/predictions.json'
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    records=json.loads(path.read_text())['instances']
    files={p.name:p for p in (args.result/'real/source_frames').rglob('*') if p.suffix in ['.jpg','.png']}
    corners=BOPModel.from_models_info(args.result/'models_info.json').corners_mm()
    factors=[.6,1.,1.6];results=[];cards=[]
    for record in records:
        image=cv2.imread(str(files[Path(record['source']).name]))
        h,w=image.shape[:2]
        points=np.asarray(record['corners_original'],float)
        box=record['yolo_box_xyxy'];long=max(box[2]-box[0],box[3]-box[1])
        hypotheses=[]
        for factor in factors:
            f=factor*max(w,h)
            K=np.array([[f,0,(w-1)/2],[0,f,(h-1)/2],[0,0,1.]])
            cv2.setRNGSeed(20260926)
            strict=fit_perspective_cuboid(corners,points,K,reprojection_threshold=5*long/256)
            unfiltered=all_point_fit(corners,points,K)
            for fit in [strict,unfiltered]:
                if fit['points'] is not None:
                    pp=np.array(fit['points'])
                    fit['mean_point_change_body256']=float(np.linalg.norm(pp-points,axis=1).mean()*256/long)
                    fit['projected_span_over_yolo_span']=float(np.ptp(pp,axis=0).max()/long)
            hypotheses.append(dict(factor=factor,K=K.tolist(),dist_coeffs=[],strict=strict,all8=unfiltered))
        shapes=[points,np.array(box).reshape(2,2)]
        for hyp in hypotheses:
            for key in ['strict','all8']:
                if hyp[key]['points'] is not None:shapes.append(np.array(hyp[key]['points']))
        bounds=np.vstack(shapes);lo,hi=bounds.min(0),bounds.max(0);padding=max(hi-lo)*.12+8
        viewport=(*np.floor(lo-padding).astype(int),*np.ceil(hi+padding).astype(int))
        row=dict(id=record['id'],source=record['source'],image_size=[w,h],input_points=points.tolist(),hypotheses=hypotheses)
        results.append(row)
        parts=[]
        for hyp in hypotheses:
            strict,all8=hyp['strict'],hyp['all8']
            fit=np.array(strict['points']) if strict['points'] is not None else None
            extra=np.array(all8['points']) if all8['points'] is not None else None
            panels=[panel(image,box,points,points,viewport,'Original 8 points',(230,60,230)),
                panel(image,box,points,fit,viewport,'PnP RANSAC: '+('PASS*' if strict['accepted'] else 'REJECT'),(80,230,80)),
                panel(image,box,points,extra,viewport,'All-8 fit: UNVERIFIED',(0,175,255))]
            filename=f'panels/{record["id"]}_f{hyp["factor"]:g}.jpg'
            cv2.imwrite(str(args.output/filename),np.hstack(panels))
            stricttext=f'几何通过，{strict["inlier_count"]}/8 内点' if strict['accepted'] else f'拒绝：{strict["status"]}'
            if strict['accepted']:
                stricttext+=f'；盒子投影跨度/YOLO 框跨度 {strict["projected_span_over_yolo_span"]:.1%}'
            residual=all8.get('mean_point_change_body256')
            extra_text=f'强制拟合平均移动 {residual:.1f} px（框长边归一到 256）' if residual is not None else '强制拟合也未得到有效解'
            hidden='' if hyp['factor']==1. else ' hidden'
            parts.append(f'<div class="hyp" data-f="{hyp["factor"]:g}"{hidden}><p>{stricttext}。{extra_text}。</p><img src="{filename}"></div>')
        cards.append(f'<article><h2>{record["id"]}</h2>'+''.join(parts)+'</article>')
    counts={str(f):dict(accepted=sum(r['hypotheses'][i]['strict']['accepted'] for r in results),total=len(results),
                       statuses=dict(Counter(r['hypotheses'][i]['strict']['status'] for r in results))) for i,f in enumerate(factors)}
    output=dict(intrinsics_calibrated=False,source_sha256=digest,source_predictions_unchanged=True,
        correspondence='Original model corner IDs 0..7; rewired visualization is NOT used as new physical correspondence',
        camera_assumptions='fx=fy=factor*max(original W,H), principal point at original image center, zero distortion. '
                           'No focal estimate, calibration or metric pose accuracy is claimed.',
        focal_factors=factors,strict_rule='RANSAC+LM, at least 6 inliers at 5px when body long side=256; geometry only',
        counts=counts,results=results)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
    (args.output/'results.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    table=''.join(f'<tr><td>{f} × 原图长边</td><td>{v["accepted"]}/{v["total"]}</td></tr>' for f,v in counts.items())
    page='''<!doctype html><meta charset="utf-8"><title>真实帧 PnP：假设内参诊断</title>
<style>body{font:18px/1.65 system-ui;max-width:1280px;margin:32px auto;padding:0 24px;background:#172630;color:#e9f1f8}img{max-width:100%}article{margin:35px 0;border-top:1px solid #536775}strong{color:#ffcc88}td,th{border:1px solid #567;padding:8px}table{border-collapse:collapse}select{font-size:18px}label{position:sticky;top:0;display:block;background:#233e4e;padding:12px}</style>
<h1>真实帧加 PnP：规则的框是否真的有帮助？</h1>
<p><strong>没有拍摄相机的标定内参。</strong>本页是三组假设焦距的敏感性实验，默认 f=原图长边，主点为原图中心、畸变为零。不是标定结果，不可当作可信真实 6D 位姿。</p>
<p>左：网络原始八点。中：RANSAC PnP，至少 6 点一致才画绿色框，否则明确拒绝。右：不剔除错误点的全八点 PnP＋LM，仅用橙色展示拟合结果，<strong>不作为接受结果</strong>。灰点保留原始预测。PnP 用原来的三维 0–7 编号，不使用重新连线后的连接关系。</p>
<p>规则盒子来自模型的刚性几何，不代表八点定位正确。几何通过也可能得到错误位置或过小的盒子。21 个检测全部保留，没有用真值修正。该实验不改模型或原预测。</p>
<table><tr><th>假设焦距 fx=fy</th><th>几何通过数（不是准确率）</th></tr>'''+table+'''</table>
<label>切换全页假设焦距：<select id="focal"><option value="0.6">0.6 × 原图长边</option><option value="1" selected>1.0 × 原图长边</option><option value="1.6">1.6 × 原图长边</option></select></label>
'''+''.join(cards)+'''<script>document.querySelector('#focal').addEventListener('change',e=>{document.querySelectorAll('.hyp').forEach(x=>{x.hidden=x.dataset.f!==e.target.value})});</script>'''
    (args.output/'index.html').write_text(page)
    print(json.dumps(counts,indent=2))


if __name__=='__main__':main()
