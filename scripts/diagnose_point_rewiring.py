"""Reconnect fixed predicted points for display; do not modify corner predictions."""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import itertools
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bop_bb8.geometry import CORNER_SIGNS
from bop_bb8.visualize import EDGES


def edge_key(permutation):
    return tuple(sorted(tuple(sorted((int(permutation[a]), int(permutation[b])))) for a,b in EDGES))


@lru_cache(maxsize=1)
def cube_topologies():
    # 8! assignments / 48 cube graph symmetries = 840 distinct edge sets.
    graphs = {}
    for permutation in itertools.permutations(range(8)):
        graphs.setdefault(edge_key(permutation), permutation)
    assert len(graphs) == 840
    return np.asarray(list(graphs.values()), dtype=int), list(graphs)


def reconnect(points):
    points = np.asarray(points, dtype=np.float64)
    if points.shape != (8,2) or not np.isfinite(points).all():
        raise ValueError('Expected eight finite fixed image points')
    permutations, graphs = cube_topologies()
    design = np.c_[np.ones(8), CORNER_SIGNS*2-1]
    projection = design @ np.linalg.pinv(design)
    ordered = points[permutations]
    approximations = np.einsum('ij,pjk->pik',projection,ordered)
    mse = np.square(approximations-ordered).sum((1,2))/8
    # Prefer original connectivity if tied, so an arbitrary symmetry cannot change it.
    winner = int(np.argmin(mse))
    original = edge_key(range(8))
    original_index = graphs.index(original)
    if np.isclose(mse[winner],mse[original_index],atol=1e-10,rtol=1e-10):
        winner = original_index
    edges = graphs[winner]
    span = float(np.linalg.norm(np.ptp(points,axis=0)))
    distances = np.linalg.norm(points[:,None]-points[None,:],axis=-1)
    close = [(a,b) for a in range(8) for b in range(a+1,8) if distances[a,b]<.025*max(span,1e-9)]
    for node in range(8):
        assert sum(node in pair for pair in edges)==3
    return dict(edges=[list(e) for e in edges],original_edges=[list(e) for e in original],
        topology_count=len(graphs),points_original=points.tolist(),points_moved=False,
        original_affine_rmse_px=float(np.sqrt(mse[original_index])),
        rewired_affine_rmse_px=float(np.sqrt(mse[winner])),
        normalized_affine_rmse=float(np.sqrt(mse[winner])/max(span,1e-9)),
        changed_edges=len(set(edges)-set(original)),near_coincident_pairs=close,
        semantic_corner_correspondence_preserved=False,
        method='Exhaustive cube topology search, least-squares affine projection consistency. '
               'Fit is used only for ranking: displayed endpoints are the exact original predictions. '
               'No intrinsic calibration, physical aspect constraint, pose recovery or GT input.')


def draw_panel(source, points, box, edges, viewport, title):
    left,top,right,bottom = viewport
    width,height=right-left,bottom-top
    scale=min(420/width,420/height)
    image=np.full((max(1,round(height*scale)),max(1,round(width*scale)),3),32,np.uint8)
    # This canvas is a review viewport, never a model input or an edited training image.
    transform=np.array([[scale,0,-left*scale],[0,scale,-top*scale]],float)
    image=cv2.warpAffine(source,transform,(image.shape[1],image.shape[0]),borderValue=(32,32,32))
    xy=np.rint((points-[left,top])*scale).astype(int)
    bb=np.rint((np.array(box).reshape(2,2)-[left,top])*scale).astype(int)
    cv2.rectangle(image,tuple(bb[0]),tuple(bb[1]),(240,170,30),1)
    for a,b in edges:
        cv2.line(image,tuple(xy[a]),tuple(xy[b]),(230,50,230),1,cv2.LINE_AA)
    for i,p in enumerate(xy):
        cv2.circle(image,tuple(p),3,(0,245,255),-1,cv2.LINE_AA)
        cv2.putText(image,str(i),tuple(p+[4,-4]),cv2.FONT_HERSHEY_SIMPLEX,.42,(0,245,255),1,cv2.LINE_AA)
    head=np.full((34,image.shape[1],3),25,np.uint8)
    cv2.putText(head,title,(8,23),cv2.FONT_HERSHEY_SIMPLEX,.5,(245,245,245),1,cv2.LINE_AA)
    return np.vstack([head,image])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'panels').mkdir()
    predictions=args.result/'real/network/predictions.json'
    before=hashlib.sha256(predictions.read_bytes()).hexdigest()
    source= json.loads(predictions.read_text())
    files={p.name:p for p in (args.result/'real/source_frames').rglob('*') if p.suffix in ['.jpg','.png']}
    rows=[];cards=[]
    for record in source['instances']:
        points=np.asarray(record['corners_original'],float)
        result=reconnect(points)
        np.testing.assert_array_equal(np.asarray(result['points_original']),points)
        image=cv2.imread(str(files[Path(record['source']).name]))
        if image is None:raise ValueError(record['source'])
        box=record['yolo_box_xyxy']
        bounds=np.vstack([points,np.array(box).reshape(2,2)])
        lo,hi=bounds.min(0),bounds.max(0)
        padding=max(hi-lo)*.18+8
        viewport=(*np.floor(lo-padding).astype(int),*np.ceil(hi+padding).astype(int))
        panels=[draw_panel(image,points,box,result[key],viewport,title) for key,title in
                [('original_edges','Original numbered edges'),('edges','Rewired - SAME 8 points')]]
        filename=f'panels/{record["id"]}.jpg'
        cv2.imwrite(str(args.output/filename),np.hstack(panels))
        rows.append(dict(id=record['id'],**result))
        cards.append(f'<article><h2>{record["id"]}</h2><p>换掉 {result["changed_edges"]}/12 条边；'
                     f'近重合点对 {len(result["near_coincident_pairs"])}；点坐标完全不变。</p>'
                     f'<img src="{filename}"><details><summary>查看原始编号的新连接关系</summary>'
                     f'<pre>{json.dumps(result["edges"])}</pre></details></article>')
    after=hashlib.sha256(predictions.read_bytes()).hexdigest()
    assert before==after
    output=dict(source_sha256=before,source_unchanged=True,instances=len(rows),
        prediction_coordinates_unchanged=True,training_or_inference_modified=False,results=rows)
    (args.output/'rewiring.json').write_text(json.dumps(output,indent=2)+'\n')
    page='''<!doctype html><meta charset="utf-8"><title>只改连线，不改预测点</title>
<style>body{font:18px/1.65 system-ui;max-width:1100px;margin:32px auto;padding:0 24px;background:#16252f;color:#e9f1f8}img{max-width:100%}article{margin:34px 0;border-top:1px solid #4e626e}summary{cursor:pointer}strong{color:#ffcb83}</style>
<h1>只重新连线：21 个检测逐项对照</h1>
<p>左：模型原来的编号连线。右：八点不移动，遍历 840 种盒子连接关系，选择最符合仿射投影盒子结构的一种。两边的点号都仍是网络原始点号，蓝框为相同 YOLO 框。</p>
<p><strong>这是连线展示实验，不是新模型，也不是姿态修复。</strong>仿射结构只用于候选排序，没有把拟合后的点画出来，没有补点、拉开重合点或使用真值。近透视等情况下该排序也可能选错。</p>
<p>改变连线会改变点的物理连接语义；更像盒子不代表每个角点正确。图中采用统一放大的展示取景，不能当作实际网络输入。所有 21 个检测均保留。</p>
'''
    (args.output/'index.html').write_text(page+''.join(cards))
    print(json.dumps(dict(instances=len(rows),all_points_unchanged=True,source_unchanged=True,
        output=str(args.output),changed=sum(r['changed_edges']>0 for r in rows)),indent=2))


if __name__=='__main__':
    main()
