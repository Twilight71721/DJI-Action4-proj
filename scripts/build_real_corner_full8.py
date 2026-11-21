"""Export nine fully annotated real cuboids; visibility never removes a target.

Manual visible references and inferred hidden vertices are kept separately in
provenance. A projective cuboid fit enforces the corner/edge correspondence; it
is an uncalibrated 2D annotation aid, not a calibrated metric camera pose.
"""
from pathlib import Path
import hashlib
import html
import json
import shutil
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bop_bb8.geometry import CORNER_SIGNS
from bop_bb8.heatmap import make_bbox_heatmaps
from bop_bb8.visualize import EDGES
from infer_resnet_yolo_real import prepare_crop

FACES = [[0,1,2,3], [4,5,6,7], [0,3,7,4], [1,2,6,5], [0,1,5,4], [2,3,7,6]]


def fit_projective_cuboid(points, basis):
    xyz = np.c_[CORNER_SIGNS * 2 - 1, np.ones(8)]
    uv = (np.asarray(points, np.float64) - 300) / 300
    rows = []
    for q, (x, y), kind in zip(xyz, uv, basis):
        weight = 2.0 if kind == 'visible_edge_reference' else 1.0
        rows.append(weight * np.r_[q, np.zeros(4), -x*q])
        rows.append(weight * np.r_[np.zeros(4), q, -y*q])
    _, _, vh = np.linalg.svd(np.array(rows))
    projection = vh[-1].reshape(3, 4)
    if np.median(xyz @ projection[2]) < 0:
        projection = -projection
    projected = xyz @ projection.T
    if np.any(projected[:, 2] <= 1e-6):
        raise ValueError('Cuboid crosses the camera plane')
    result = projected[:, :2] / projected[:, 2:] * 300 + 300
    for face in FACES:
        polygon = result[face].astype(np.float32)
        if not cv2.isContourConvex(polygon):
            raise ValueError(f'Self-intersecting/degenerate face: {face}')
    return result, projection


def draw_full8(image, points, basis):
    result = image.copy()
    for a, b in EDGES:
        cv2.line(result, tuple(np.rint(points[a]).astype(int)),
                 tuple(np.rint(points[b]).astype(int)), (80, 225, 100), 2, cv2.LINE_AA)
    for i, point in enumerate(points):
        p = np.rint(point).astype(int)
        direct = basis[i] == 'visible_edge_reference'
        color = (70, 255, 100) if direct else (40, 210, 255)
        cv2.circle(result, tuple(p), 5, color, -1 if direct else 2, cv2.LINE_AA)
        anchor = p + [8, -8]
        cv2.putText(result, str(i), tuple(anchor), cv2.FONT_HERSHEY_SIMPLEX,
                    .7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(result, str(i), tuple(anchor), cv2.FONT_HERSHEY_SIMPLEX,
                    .7, color, 2, cv2.LINE_AA)
    return result


def build(output):
    manual = json.loads((output / 'manual_corners.json').read_text())
    for folder in ['images', 'crops', 'review']:
        (output / folder).mkdir(exist_ok=True)
    records, tensor_rows = [], []
    for record in manual['records']:
        source = ROOT / record['source_partial_dataset'] / record['image']
        destination = output / record['image']
        if not destination.exists():
            shutil.copy2(source, destination)
        if hashlib.sha256(destination.read_bytes()).hexdigest() != record['image_sha256']:
            raise ValueError('Source image hash mismatch')
        image = cv2.imread(str(destination))
        points, projection = fit_projective_cuboid(record['manual_corners_review'], record['point_basis'])
        c = np.array(record['annotation_crop_xywh'])
        original = points * c[2:] / 600 + c[:2]
        roi, tensor, crop = prepare_crop(image, record['box_xyxy'], (256, 256), 1.3)
        inp = (original - np.array(crop[:2])) * 256 / np.array(crop[2:])
        if not np.isfinite(inp).all() or (inp < 0).any() or (inp >= 256).any():
            raise ValueError(f'Full cuboid outside crop: {record["id"]}')
        valid = np.ones(8, dtype=bool)
        targets = make_bbox_heatmaps(inp / 4, valid, (64, 64), kernel='gaussian', sigma=2.)
        assert np.all(targets.max(axis=(1,2)) > .999)
        tensor_rows.append(dict(image=tensor.numpy(), heatmap=targets, valid=valid,
                                keypoints_2d=inp.astype(np.float32)))
        x,y,w,h = c
        raw = cv2.resize(image[y:y+h, x:x+w], (600,600), interpolation=cv2.INTER_LINEAR)
        overlay = draw_full8(raw, points, record['point_basis'])
        cv2.imwrite(str(output/'review'/f'{record["id"]}_overlay.png'), overlay)
        panel = np.hstack([raw, overlay])
        panel = cv2.copyMakeBorder(panel, 40, 0, 0, 0, cv2.BORDER_CONSTANT, value=(20,25,30))
        cv2.putText(panel, f'{record["id"]} | REAL / ANNOTATION: 8 OF 8 CORNERS', (12,27),
                    cv2.FONT_HERSHEY_SIMPLEX, .65, (245,245,245), 1, cv2.LINE_AA)
        cv2.imwrite(str(output/'review'/f'{record["id"]}.jpg'), panel, [cv2.IMWRITE_JPEG_QUALITY,95])
        cv2.imwrite(str(output/'crops'/f'{record["id"]}.png'), roi)
        delta = np.linalg.norm(points-np.array(record['manual_corners_review']), axis=-1)
        result = dict(record, keypoints_review=points.tolist(), keypoints_original=original.tolist(),
                      keypoints_input=inp.tolist(), crop_xywh=list(crop), valid=valid.tolist(),
                      annotated=[True]*8, annotation_type='manual_full8_with_inferred_vertices',
                      geometry_check=dict(projective_camera_normalized_cube=projection.tolist(),
                                          max_adjustment_review_px=float(delta.max()),
                                          calibrated=False, metric_pose_ground_truth=False))
        records.append(result)
        print(record['id'], '8/8 targets, max geometric adjustment at 600px', round(delta.max(),2), flush=True)
    document = dict(schema_version=2, corner_definition=manual['corner_definition'],
                    protocol='Every instance has eight annotated targets, including hidden corners. All eight enter training. Hidden/inferred provenance is metadata, not a loss mask. Manual estimates are not calibrated 6D ground truth.',
                    input_size=[256,256], heatmap_size=[64,64], crop_padding=1.3, records=records)
    (output/'annotations.json').write_text(json.dumps(document, indent=2))
    np.savez_compressed(output/'full8_supervision.npz',
                        **{k:np.stack([row[k] for row in tensor_rows]) for k in tensor_rows[0]},
                        id=np.array([r['id'] for r in records]), split=np.array([r['split'] for r in records]))
    sections = []
    for r in records:
        points_table=''.join(f'<tr><td>{i}</td><td>{p[0]:.1f}</td><td>{p[1]:.1f}</td><td>{"边缘参考" if r["point_basis"][i]=="visible_edge_reference" else "推定补全"}</td><td>是</td></tr>' for i,p in enumerate(r['keypoints_original']))
        sections.append(f'<article><h2>第 {r["frame"]} 帧 · {"左" if r["instance"]==0 else "右"}相机 <span class="badge">8 / 8</span></h2><p>{html.escape(r["split"])} · 八点均参与监督</p><a href="{r["id"]}_overlay.png"><img loading="lazy" src="{r["id"]}.jpg" alt="原图与全部八个标注点"></a><details><summary>查看 0–7 号原图坐标</summary><table><tr><th>编号</th><th>x</th><th>y</th><th>依据</th><th>参与监督</th></tr>{points_table}</table></details></article>')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>真实帧完整八角点标注</title><style>body{max-width:1280px;margin:36px auto;padding:0 24px;background:#14212a;color:#edf3f6;font:17px/1.7 sans-serif}h1,h2{line-height:1.4}img{width:100%;height:auto;border-radius:8px}a{color:#82d6fa}article{border-top:1px solid #445c6b;padding:25px 0}table{border-collapse:collapse;margin:16px 0}td,th{padding:8px 18px;border:1px solid #567}.badge{background:#245f43;font-size:18px;padding:4px 12px;border-radius:15px;margin-left:12px}.note{background:#253b49;padding:18px;border-radius:8px}summary{cursor:pointer;color:#b4e0fa}</style><h1>真实帧完整八角点标注</h1><div class="note"><b>7 张真实帧 · 9 个相机实例 · 72 / 72 个角点</b><br>每个相机均标齐 0–7 号八个点，并生成八张监督热图。遮挡或隐藏不再等于“没有标注”。</div><p>绿色实点依据可见边缘，黄色空心点为补全点；两者都参与训练。所有点使用同一套三维包围盒编号，推定点结合可见面、模型几何和框体一致性校核。它们属于人工近似标注，尚无标定后的精确三维真值。</p><p>左侧为原图，右侧为完整标注；这里展示的是标注，不是模型预测。</p><p><a href="../annotations.json">完整坐标 JSON</a> · <a href="../full8_supervision.npz">八通道训练数据</a> · <a href="../README.md">数据说明</a></p>'''+''.join(sections)+'</html>'
    (output/'review/index.html').write_text(page)
    return document


if __name__ == '__main__':
    build(ROOT/'dataset/action4-real-corners-full8-20260925')
