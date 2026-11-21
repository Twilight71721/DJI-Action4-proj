"""Fixed-weight local comparison of independent and joint cuboid decoders."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from bop_bb8.geometry import BOPModel
from bop_bb8.joint_decode import decode_joint_cuboid
from bop_bb8.net import ResNetFPN,decode_topk,heatmap_to_input_points
from bop_bb8.torch_data import IMAGENET_MEAN,IMAGENET_STD
from infer_resnet_yolo_real import prepare_crop,draw_points
from evaluate_heatmap_checkpoint import metric_summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostic',type=Path,default=ROOT/'outputs/local_cause_diagnostic_20260923')
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'outputs/server_return_5000_20260923/best.pth')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True);(args.output/'images').mkdir()
    torch.set_num_threads(4);cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    a=checkpoint['args'];model=ResNetFPN(backbone=a['backbone'],fpn_dim=a['fpn_dim'],pretrained=False).to(device).eval()
    model.load_state_dict(checkpoint['model'],strict=True)
    cases=json.loads((args.diagnostic/'manifest.json').read_text())['cases']
    corners=BOPModel.from_models_info(ROOT/'dataset/action4-cleanblur5000-20260923/models/models_info.json').corners_mm()
    saved=np.load(args.diagnostic/'predictions.npz')
    assert list(saved['ids'])==[c['id'] for c in cases]
    summary,records,heatmaps={},[],[]
    cards={'real':[],'synthetic':[]};start=time.time()
    for variant in ['baseline','background_blur']:
        tensors,rois,boxes=[],[],[]
        for c in cases:
            image=cv2.imread(c['source']);roi,_,crop=prepare_crop(image,c['box'],(256,256),1.3)
            assert np.array_equal(crop,c['crop'])
            box=(np.array(c['box']).reshape(2,2)-np.array(crop[:2]))*256/np.array(crop[2:])
            f=roi.astype(np.float32)/255
            if variant=='background_blur':
                (x1,y1),(x2,y2)=np.rint(box).astype(int)
                mask=np.zeros((256,256),bool);mask[max(0,y1-2):min(256,y2+3),max(0,x1-2):min(256,x2+3)]=True
                blurred=cv2.GaussianBlur(f,(31,31),8);blurred[mask]=f[mask];f=blurred
            tensor=torch.from_numpy(cv2.cvtColor(f,cv2.COLOR_BGR2RGB).copy()).permute(2,0,1)
            tensor=(tensor-torch.tensor(IMAGENET_MEAN)[:,None,None])/torch.tensor(IMAGENET_STD)[:,None,None]
            tensors.append(tensor);rois.append(np.uint8(np.clip(np.rint(f*255),0,255)));boxes.append(box.ravel())
        probs=[]
        with torch.inference_mode():
            for begin in range(0,len(cases),16):
                chunk=tensors[begin:begin+16];n=len(chunk)
                batch=torch.stack(chunk+[chunk[-1]]*(16-n)).to(device)
                probs.append(model(batch).float().sigmoid()[:n].cpu())
        probabilities=torch.cat(probs);heatmaps.append(probabilities.numpy())
        raw=heatmap_to_input_points(decode_topk(probabilities,k=9),(64,64),(256,256)).numpy()
        reference=saved['points'][list(saved['variants']).index(variant)]
        repeat_delta=float(np.abs(raw-reference).max())
        errors={k:[] for k in ['top9','local_peak','joint_selected','joint_rigid','candidate_oracle']}
        accepted=[];states=[];realstates=[];families=set();valids=[]
        for i,c in enumerate(cases):
            result=decode_joint_cuboid(probabilities[i].numpy(),corners,box_xyxy=boxes[i])
            local=np.array([p['points'][0] if p['points'] else [np.nan,np.nan] for p in result['candidates']])
            predictions=dict(top9=raw[i],local_peak=local,joint_selected=np.array(result['selected_points']) if result['selected_points'] is not None else np.full((8,2),np.nan),
                             joint_rigid=np.array(result['rigid_points']) if result['rigid_points'] is not None else np.full((8,2),np.nan))
            entry=dict(id=c['id'],domain=c['domain'],variant=variant,joint=result,top9=raw[i].tolist(),local_peak=local.tolist())
            records.append(entry)
            if c['domain']=='synthetic':
                gt=np.array(c['target']);valid=np.array(c['valid']);valids.append(valid);accepted.append(result['accepted']);states.append(result['status'])
                for k,p in predictions.items():
                    e=np.linalg.norm(p-gt,axis=1);e[~valid]=np.nan;errors[k].append(e)
                oracle=np.array([min(np.linalg.norm(np.array(p['points'])-gt[j],axis=1)) if p['points'] else np.nan for j,p in enumerate(result['candidates'])])
                oracle[~valid]=np.nan;errors['candidate_oracle'].append(oracle)
                show=variant=='baseline' and c['family'] not in families
                families.add(c['family'])
            else:
                realstates.append(result['status']);show=variant=='baseline'
            if show:
                panes=[]
                for key,points,label in [('input',None,'输入'),('top9',raw[i],'原始 Top-9'),('local_peak',local,'独立最高峰'),('joint_selected',predictions['joint_selected'],'联合选峰'),('joint_rigid',predictions['joint_rigid'],'对应刚体投影')]:
                    view=rois[i].copy()
                    if points is not None and np.isfinite(points).all():
                        view=draw_points(view,points)
                    if c['domain']=='synthetic' and key!='input':
                        for p in gt[valid]:
                            cv2.circle(view,tuple(np.rint(p).astype(int)),5,(40,240,40),1)
                    if key in ['joint_selected','joint_rigid'] and not result['accepted']:
                        cv2.putText(view,'REJECTED / UNVERIFIED',(3,15),0,.38,(0,80,255),1)
                    filename=f"images/{c['id']}_{key}.jpg";cv2.imwrite(str(args.output/filename),view)
                    panes.append(f'<figure><figcaption>{label}</figcaption><img loading="lazy" src="{filename}" alt="{c["id"]} {label}"></figure>')
                status='通过启发式一致性检查，仍未验证定位正确' if result['accepted'] else '候选证据不足／几何不一致：拒绝作为检测结果'
                cards[c['domain']].append(f'<article><h3>{html.escape(c["id"])}</h3><p>{status}；改变 {result.get("changed_channels",0)} 个通道的候选。相对响应不是准确率。</p><div class="grid">'+''.join(panes)+'</div></article>')
            if (i+1)%24==0:
                print(f'{variant} {i+1}/{len(cases)} elapsed={time.time()-start:.1f}s',flush=True)
        valid=np.array(valids);accept=np.array(accepted)
        metrics={}
        for k,e in errors.items():
            e=np.array(e)
            metrics[k]=metric_summary(e)
            metrics[k]['pck5_all_valid']=float(((e<=5)&valid).sum()/valid.sum())
            metrics[k]['missing_corner_count']=int((valid&~np.isfinite(e)).sum())
            if k.startswith('joint'):
                metrics[k]['pck5_rejections_count_as_failure']=float(((e<=5)&valid&accept[:,None]).sum()/valid.sum())
        summary[variant]=dict(synthetic=metrics,synthetic_accepted_instances=int(accept.sum()),synthetic_instance_count=len(accept),
                              synthetic_statuses=dict(Counter(states)),real_statuses=dict(Counter(realstates)),repeat_max_difference_px=repeat_delta)
        np.savez_compressed(args.output/f'{variant}_errors.npz',**{k:np.array(v) for k,v in errors.items()},accepted=accept,valid=valid)
    np.savez_compressed(args.output/'heatmaps.npz',probabilities=np.array(heatmaps),variants=np.array(['baseline','background_blur']),ids=np.array([c['id'] for c in cases]))
    (args.output/'predictions.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
    protocol=dict(checkpoint=str(args.checkpoint),checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                  input_manifest=str(args.diagnostic/'manifest.json'),camera_intrinsics_used=False,training_performed=False,device=str(device),
                  parameters=dict(max_peaks=6,min_ratio=.05,beam_width=256,geometric_sigma_px=8,refine_count=12,min_span_box_ratio=.35),
                  elapsed_seconds=time.time()-start,results=summary)
    (args.output/'summary.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2))
    names=dict(top9='原始 Top-9',local_peak='独立局部最高峰',joint_selected='刚体引导联合选峰',joint_rigid='联合选峰后的刚体投影',candidate_oracle='候选上限：借助真值选峰，仅诊断')
    rows=[];md=[]
    for variant,v in summary.items():
        for key,m in v['synthetic'].items():
            row=[variant,names[key],f'{m["mean_err_px"]:.2f}',f'{m["pck5_all_valid"]*100:.1f}%',f'{m["pck5_rejections_count_as_failure"]*100:.1f}%' if key.startswith('joint') else '—']
            rows.append('<tr>'+''.join('<td>'+x+'</td>' for x in row)+'</tr>');md.append('| '+' | '.join(row)+' |')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>热图候选联合刚体解码</title><style>body{font:16px/1.65 system-ui;max-width:1550px;margin:30px auto;padding:0 20px;background:#14212a;color:#eee}a{color:#8dd2ff}.note{background:#403427;padding:16px;border-left:4px solid #efbd65}.grid{display:grid;grid-template-columns:repeat(5,1fr);gap:10px}figure{margin:0}img{width:100%}article{padding:18px;background:#263641;margin:22px 0}td,th{border:1px solid #596773;padding:7px 12px}table{border-collapse:collapse}summary{cursor:pointer;font-size:22px}@media(max-width:900px){.grid{grid-template-columns:repeat(2,1fr)}} </style>
<h1>保留多个热图候选，共同选择八个角点</h1><p>同一权重、同一输入，比较独立解码与刚体引导的候选组合搜索。</p><p class="note">真实帧无八角点真值。通过一致性检查不代表姿态正确；被拒绝的候选仍展示供诊断，并标红。最后一列刚体投影有弱透视近似误差；倒数第二列是实际选中的峰，不能保证构成严格刚体。</p><p><a href="ANALYSIS.md">方法与完整结果说明</a> · <a href="summary.json">指标</a></p>
<table><tr><th>输入</th><th>方法</th><th>有输出点平均误差 px</th><th>全部有效真值点 PCK@5</th><th>拒绝计失败 PCK@5</th></tr>ROWS</table><p>候选上限使用真值，仅衡量候选中有没有正确点，不是可部署方法。其他方法不读真值。表中未隐藏被拒绝但仍有输出的候选。</p><h2>24 个真实实例</h2>REAL<details><summary>20 个合成实例：绿色空心圈为真值</summary>SYN</details></html>'''
    (args.output/'index.html').write_text(page.replace('ROWS',''.join(rows)).replace('REAL',''.join(cards['real'])).replace('SYN',''.join(cards['synthetic'])))
    report='''# 热图联合刚体解码对照

每个热图提取最多 6 个相互分离的局部峰，最低相对响应 0.05，3×3 局部加权求坐标。八个通道的语义编号固定，不进行无约束匹配。以 beam=256 搜索组合，代价包含相对峰强度和弱透视刚体拟合残差，再细化 12 组候选。几何残差尺度固定 8px；这是有限搜索，不保证全局最优。

与上一版不同，本次搜索发生在峰被丢弃之前。输出同时保留选中峰与刚体投影，避免把刚体近似偏差误归因于选峰。二维框仅提供宽松的最小投影跨度（框长边的 35%），没有要求三维框投影四边贴合二维框。

没有改权重或训练数据，也没有修改默认推理解码。所有参数在本次对照前固定。使用之前的 120 个合成验证实例和 24 个真实实例；不是独立最终测试集。真实框为已有手工估计框，非真值姿态。背景模糊对照严格保护目标矩形内像素。

## 结果

| 输入 | 方法 | 有输出点平均误差 px | PCK@5 全部真值点 | 拒绝计失败 PCK@5 |
|---|---|---:|---:|---:|
'''+ '\n'.join(md)+'''

拒绝规则：拟合跨度不足最小跨度，或少于 6 个选中峰在相应刚体投影点 12px 内。该规则只是启发式一致性检查，不是已标定置信度；错误但自洽的角点仍可能通过。没有强行把拒绝结果算作成功。缺失输出的真值点在 PCK 分母中保留、计失败，平均误差仅覆盖有输出点。

候选上限借助真值为每个角点选择最近候选，只在评估中计算。若候选中不存在正确位置，联合选择不能创造正确峰；若上限高但实际低，候选选择、近似模型或有限搜索仍有改善空间。

## 复现

```bash
.venv-hccepose/bin/python -m unittest discover -s tests -p test_joint_decode.py -v
.venv-hccepose/bin/python scripts/report_joint_cuboid.py --output outputs/joint_cuboid_repeat
```

新目录必须不存在，以保留旧结果。原始热图、逐例候选、选择编号、拒绝状态以及原始预测复算差异均保存在本目录的 npz/json 文件中。本次只用验证样本做探索，不应反复根据真实浏览结果调参再宣称独立泛化。
'''
    (args.output/'ANALYSIS.md').write_text(report)
    print(json.dumps(protocol,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()
