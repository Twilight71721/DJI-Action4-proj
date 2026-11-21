"""Compare existing predictions with uncalibrated rigid cuboid proposals."""
from __future__ import annotations

import argparse
from collections import Counter
import html
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bop_bb8.geometry import BOPModel
from bop_bb8.rigid import fit_rigid_cuboid
from infer_resnet_yolo_real import prepare_crop, draw_points
from evaluate_heatmap_checkpoint import metric_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostic', type=Path, default=ROOT/'outputs/local_cause_diagnostic_20260923')
    parser.add_argument('--model-info', type=Path, default=ROOT/'dataset/action4-cleanblur5000-20260923/models/models_info.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    (args.output/'images').mkdir()
    cases = json.loads((args.diagnostic/'manifest.json').read_text())['cases']
    data = np.load(args.diagnostic/'predictions.npz')
    raw = data['points'][list(data['variants']).index('baseline')]
    assert list(data['ids']) == [c['id'] for c in cases]
    corners = BOPModel.from_models_info(args.model_info).corners_mm()
    errors = {k:[] for k in ['raw','rigid','rigid_box','oracle_rigid']}
    records, cards = [], {'real':[], 'synthetic':[]}
    started = time.time()
    seen_families = set()
    for i,c in enumerate(cases):
        crop = np.array(c['crop'])
        box = (np.array(c['box']).reshape(2,2)-crop[:2])*256/crop[2:]
        fit = fit_rigid_cuboid(corners,raw[i],starts=4)
        fit_box = fit_rigid_cuboid(corners,raw[i],box_xyxy=box.ravel(),box_weight=2.,starts=12)
        result = dict(id=c['id'],domain=c['domain'],box_crop_xyxy=box.ravel().tolist(),raw=raw[i].tolist(),rigid=fit,rigid_box=fit_box)
        if c['domain']=='synthetic':
            target = np.array(c['target'])
            valid = np.array(c['valid'])
            oracle = fit_rigid_cuboid(corners,target,starts=4)
            for k,points in [('raw',raw[i]),('rigid',fit['points']),('rigid_box',fit_box['points']),('oracle_rigid',oracle['points'])]:
                e = np.full(8,np.nan) if points is None else np.linalg.norm(np.array(points)-target,axis=1)
                e[~valid] = np.nan
                errors[k].append(e)
            show = c['family'] not in seen_families
            seen_families.add(c['family'])
        else:
            show = True
        if show:
            image = cv2.imread(c['source'])
            roi,_,actual_crop = prepare_crop(image,c['box'],(256,256),1.3)
            assert np.array_equal(crop,actual_crop)
            panes=[]
            for key,points,label in [('input',None,'输入与二维框'),('raw',raw[i],'原始八点'),('rigid',fit['points'],'仅刚体拟合'),('rigid_box',fit_box['points'],'刚体 + 二维框先验')]:
                view = roi.copy() if points is None else draw_points(roi,np.array(points))
                if key=='input':
                    cv2.rectangle(view,tuple(np.rint(box[0]).astype(int)),tuple(np.rint(box[1]).astype(int)),(255,180,0),1)
                if c['domain']=='synthetic' and key!='input':
                    for p in target[valid]:
                        cv2.circle(view,tuple(np.rint(p).astype(int)),5,(40,240,40),1)
                filename=f"images/{c['id']}_{key}.jpg"
                cv2.imwrite(str(args.output/filename),view)
                panes.append(f'<figure><figcaption>{label}</figcaption><img loading="lazy" src="{filename}" alt="{c["id"]} {label}"></figure>')
            statuses={'prior_dominated_collapsed_input':'原始点塌缩，结果依赖框先验','large_correction_unverified':'角点修正较大，未经真值验证','geometric_fit_unverified':'几何拟合结果，未经真值验证','collapsed_observations':'原始点完全重合，拒绝无先验拟合'}
            status=statuses.get(fit_box['status'],fit_box['status'])
            correction=fit_box.get('mean_corner_correction_px',float('nan'))
            cards[c['domain']].append(f'<article><h3>{html.escape(c["id"])} · {html.escape(c.get("view",""))}</h3><p>{status}；加入框先验后平均移动 {correction:.1f}px。真实图无角点真值。</p><div class="grid">'+''.join(panes)+'</div></article>')
        records.append(result)
        if (i+1)%12==0:
            print(f'{i+1}/{len(cases)} elapsed={time.time()-started:.1f}s',flush=True)
    metrics={k:metric_summary(np.array(v)) for k,v in errors.items()}
    counts={k:dict(Counter(r[k]['status'] for r in records if r['domain']=='real')) for k in ['rigid','rigid_box']}
    summary=dict(projection='scaled_orthographic',camera_intrinsics_used=False,model_dimensions_mm=np.ptp(corners,axis=0).tolist(),
                 source=str(args.diagnostic),elapsed_seconds=time.time()-started,synthetic=metrics,real_status_counts=counts,
                 real_accuracy_available=False,box_prior_weight=2.,weights_changed=False,training_performed=False,
                 note='Shape-constrained proposals, not calibrated 6D poses. Rectangle priors may fabricate plausible shape from incorrect points.')
    (args.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    (args.output/'predictions.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
    names={'raw':'原始 Top-9','rigid':'仅刚体约束','rigid_box':'刚体 + 框先验','oracle_rigid':'真值拟合：仅检查近似误差'}
    rows=''.join(f'<tr><td>{names[k]}</td><td>{v["mean_err_px"]:.2f}</td><td>{v["pck5"]*100:.1f}%</td><td>{v["p95_err_px"]:.2f}</td></tr>' for k,v in metrics.items())
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>无需内参的刚体约束对照</title><style>
body{font:16px/1.65 system-ui;max-width:1400px;margin:30px auto;padding:0 22px;background:#14212a;color:#eee}a{color:#8dd2ff}.note{padding:18px;background:#3a3026;border-left:4px solid #f5bf64}table{border-collapse:collapse}td,th{padding:9px 18px;border:1px solid #53616b}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}figure{margin:0}img{width:100%}article{background:#263641;padding:16px;margin:22px 0}summary{cursor:pointer;font-size:22px}@media(max-width:850px){.grid{grid-template-columns:repeat(2,1fr)}} </style>
<h1>无需相机内参的刚体约束对照</h1><p>固定三维长宽高比例，拟合整体旋转、统一缩放与二维平移。保留旧模型和原始预测。</p>
<div class="note">这是弱透视近似，不是已标定的真实 6D 位姿。尤其右侧加入二维框先验后，规整的形状可能来自先验，不能据此认定定位正确。各角点编号不交换，也没有把二维边强行设为直角。</div>
<p><a href="ANALYSIS.md">方法、结果及复现说明</a> · <a href="summary.json">机器可读指标</a></p><h2>120 个合成验证实例：是否真的更准</h2><p>全部使用同一基准 256px 坐标；真值拟合仅测弱透视近似误差，不参与网络预测或真实帧拟合。</p><table><tr><th>方法</th><th>平均误差 px</th><th>PCK@5</th><th>P95 px</th></tr>ROWS</table>
<h2>24 个真实实例</h2><p>二维框来自已有手工估计标注。没有可信八角点真值，因此只展示形态变化，不报告真实准确率。</p>REAL
<details><summary>20 个合成示例（每 family 首个实例）</summary><p>绿色空心圈为真值。黄色点、紫色线为预测。</p>SYN</details></html>'''
    (args.output/'index.html').write_text(page.replace('ROWS',rows).replace('REAL',''.join(cards['real'])).replace('SYN',''.join(cards['synthetic'])))
    table='\n'.join(f'| {names[k]} | {v["mean_err_px"]:.3f} | {v["pck5"]*100:.2f}% | {v["p95_err_px"]:.3f} |' for k,v in metrics.items())
    report=f'''# 无内参刚体约束对照

## 方法

固定已知三维框尺寸比例，拟合 `p_i = s (R (X_i - mean(X)))_xy + t`，其中 R 属于 SO(3)、s>0、t 为二维平移。未知焦距与距离合并为尺度，不声称恢复真实距离或标定姿态。相对深度变化较小时弱透视是近似，强透视和畸变不能用此模型精确解释。

仅角点拟合：对现有 Top-9 八点做鲁棒最小二乘。框先验版本另惩罚投影外接矩形与现有二维框的差，权重固定为 2。使用确定性的多初值求解，不交换角点编号；有界数值搜索也不保证找到全局最优。框先验不是额外观测到的角点，目标可见轮廓框与三维包围盒投影范围不一定一致，可能引入偏差。

核心模块 `bop_bb8/rigid.py`，报告脚本 `scripts/report_rigid_cuboid.py`。没有修改网络、训练数据和默认解码；没有重新训练。只新增可复用拟合模块、测试及本报告。

## 120 个合成验证实例 / 960 个角点

| 方法 | 平均误差 px | PCK@5 | P95 px |
|---|---:|---:|---:|
{table}

真值拟合行不使用网络预测，仅检查弱透视表示在这些透视图上的近似能力。其他行拟合只读模型预测和既有二维框，没有使用角点真值。

本次结果：仅刚体约束的平均误差从 {metrics['raw']['mean_err_px']:.2f}px 变为 {metrics['rigid']['mean_err_px']:.2f}px；框先验版本为 {metrics['rigid_box']['mean_err_px']:.2f}px。约束降低了本批最坏单点误差，但平均精度下降，因此不能直接替换默认输出。真值拟合仍有 {metrics['oracle_rigid']['mean_err_px']:.2f}px 平均偏差，说明弱透视近似本身也有代价。这不是八个角点预测正确性的替代品；真实图检查仍可看到原始塌缩点群经约束后停留在错误局部。

## 24 个真实实例

状态计数：`{json.dumps(counts,ensure_ascii=False)}`。

原始点跨度不足框长边 25% 时，带框拟合标记 `prior_dominated_collapsed_input`；平均修正超过框长边 10% 时标记 `large_correction_unverified`。这些仅为说明结果依赖程度的启发式标记，**不是标定置信度，也不是有效姿态认证**。所有其他拟合也标记 unverified。

图上变规整不能替代真实定位验证。如果合成误差增加，不能将这一版本直接替换默认解码。若真实输入八个通道本来都预测同一点，几何约束与框只提供尺寸、位置和形状的限制，不能凭空确定正确的朝向。

## 复现

从项目根目录执行：
```bash
.venv-hccepose/bin/python -m unittest discover -s tests -p test_rigid.py -v
.venv-hccepose/bin/python scripts/report_rigid_cuboid.py --output outputs/rigid_cuboid_diagnostic_repeat
```

输出目录必须不存在，以避免覆盖已有结果。使用上一轮固定样本及预测；依赖当前项目已有 scipy、numpy、OpenCV 和 torch 环境。

弱透视参考：[The Weak-Perspective Camera](https://www.cse.iitd.ac.in/~suban/vision/geometry/node29.html)。本实现是独立诊断拟合，不是该教材或 DOPE 的代码复现。
'''
    (args.output/'ANALYSIS.md').write_text(report)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()
