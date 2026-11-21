"""Compare the preserved 3305 baseline and a clean/blur-trained checkpoint."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from bop_bb8 import BOPInstanceDataset,BOPModel
from bop_bb8.net import ResNetFPN,decode_topk,heatmap_to_input_points
from bop_bb8.torch_data import HeatmapDataset
from evaluate_heatmap_checkpoint import metric_summary,draw_overlay,make_contact_sheet
from infer_resnet_yolo_real import prepare_crop,draw_points


def load_model(path,device,allow_tight=False):
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    args=checkpoint['args']
    legacy=args['crop_mode']=='bbox' and args['crop_padding']==1.3
    tight=allow_tight and args['crop_mode']=='cuboid' and args['crop_padding']==1.0
    assert args['input_size']==[256,256] and (legacy or tight), 'Use a crop-aware evaluator for tight checkpoints'
    model=ResNetFPN(backbone=args['backbone'],fpn_dim=args['fpn_dim'],pretrained=False).to(device).eval()
    model.load_state_dict(checkpoint['model'],strict=True)
    metadata=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  epoch_one_based=checkpoint['epoch']+1,args=args)
    return model,metadata


@torch.inference_mode()
def predict(model,images,device):
    logits=model(images.to(device)).float();p=torch.sigmoid(logits)
    points=heatmap_to_input_points(decode_topk(logits,k=9,from_logits=True),tuple(p.shape[-2:]),(256,256))
    return points.cpu().numpy(),p.amax(dim=(-2,-1)).cpu().numpy()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--real-manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--limit',type=int,default=0,help='Diagnostic only; marked incomplete in report')
    args=parser.parse_args();torch.set_num_threads(4);cv2.setNumThreads(1)
    device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    data=args.dataset.resolve();manifest=json.loads((data/'manifest.json').read_text())
    info=BOPModel.from_models_info(data/'models/models_info.json',1,1.)
    records=json.loads(args.real_manifest.read_text())['records']
    crops=[]
    for r in records:
        image=cv2.imread(str(args.real_manifest.parent/r['image']))
        if image is None:raise FileNotFoundError(r['image'])
        roi,tensor,crop=prepare_crop(image,r['box'],(256,256),1.3)
        crops.append((roi,tensor));r['crop']=crop;r['predictions']={};r['peaks']={}
    summary=dict(protocol=manifest['protocol'],incomplete_diagnostic=bool(args.limit),
                 counts=manifest['counts'],synthetic={},checkpoints={},
                 real_protocol='8 original diagnostic frames, manual and fixed YOLO boxes, no reliable corner GT; not a blind real test.',
                 comparison_caveat='Old3305 vs new5000 paired: data, augmentation and optimizer-step count differ. Both use the same preserved holdouts; not a blur-only ablation.')
    panels={};families={j['scene']:j['family'] for j in manifest['jobs']}
    for name,path in [('baseline3305',args.baseline),('new5000paired',args.checkpoint)]:
        model,meta=load_model(path,device);summary['checkpoints'][name]=meta;summary['synthetic'][name]={}
        for split in ['val','test']:
            for variant in ['clean','blur']:
                key=f'{split}_{variant}'
                base=BOPInstanceDataset(data/variant,info,scenes=manifest['splits'][split],input_size=(256,256),
                        heatmap_size=(64,64),crop_mode='bbox',crop_padding=1.3,heatmap_kernel='gaussian',heatmap_sigma=2.)
                if args.limit:base.instances=base.instances[:args.limit]
                loader=DataLoader(HeatmapDataset(base),batch_size=16,shuffle=False,num_workers=args.workers)
                preds=[];targets=[];valid=[];peaks=[]
                for batch in loader:
                    pred,peak=predict(model,batch['image'],device)
                    preds.append(pred);peaks.append(peak);targets.append(batch['keypoints_2d'].numpy());valid.append(batch['valid'].numpy())
                preds,targets,valid,peaks=map(np.concatenate,[preds,targets,valid,peaks])
                errors=np.linalg.norm(preds-targets,axis=-1);errors[~valid]=np.nan
                index=np.array(base.instances);family=np.array([families[x[0]] for x in base.instances])
                metrics=dict(metric_summary(errors),instances=len(base),excluded_corners=int((~valid).sum()),
                             per_family={str(f):metric_summary(errors[family==f]) for f in sorted(set(family))})
                if split=='test':
                    old_mask=np.array([x[0] in manifest['old_test_scenes'] for x in base.instances])
                    metrics['preserved_335_image_test']=dict(metric_summary(errors[old_mask]),instances=int(old_mask.sum()))
                summary['synthetic'][name][key]=metrics
                np.savez_compressed(out/f'{name}_{key}.npz',predictions=preds,targets=targets,valid=valid,
                                    errors=errors,peaks=peaks,index=index,family=family)
                cards=[]
                for i in np.linspace(0,len(base)-1,min(8,len(base)),dtype=int):
                    s=base[int(i)];cards.append(draw_overlay(s.image,s.keypoints_2d,preds[i],s.valid,
                                                    f'{name} {key} {s.scene_id}/{s.image_id}/{s.instance_idx}'))
                panels.setdefault(key,[]).append(cards)
                print(name,key,metrics['mean_err_px'],flush=True)
        for start in range(0,len(crops),16):
            predictions,peaks=predict(model,torch.stack([x[1] for x in crops[start:start+16]]),device)
            for k,(pred,peak) in enumerate(zip(predictions,peaks),start):
                records[k]['predictions'][name]=pred.tolist();records[k]['peaks'][name]=peak.tolist()
        del model
        if device.type=='cuda':torch.cuda.empty_cache()
    parts=[]
    for key,groups in panels.items():
        filename=f'{key}.jpg';cv2.imwrite(str(out/filename),make_contact_sheet([np.hstack(pair) for pair in zip(*groups)],columns=2))
        parts.append(f'<h2>{key}: 左3305基线 / 右新模型</h2><img loading="lazy" src="{filename}">')
    for i,r in enumerate(records):
        roi=crops[i][0];panel=np.hstack([roi,*[draw_points(roi,np.array(r['predictions'][n])) for n in summary['checkpoints']]])
        filename=f"real_{r['frame']:06d}_{r['method']}_{r['instance']}.jpg";cv2.imwrite(str(out/filename),panel)
        parts.append(f"<h3>真实 {r['frame']} {r['method']} {r['instance']}：原图 / 3305基线 / 新模型</h3><img loading='lazy' src='{filename}'>")
    rows=[]
    for name,sets in summary['synthetic'].items():
        for key,m in sets.items():
            rows.append(f'<tr><td>{name}</td><td>{key}</td><td>{m["instances"]}</td><td>{m["mean_err_px"]:.3f}</td><td>{m["pck5"]:.2%}</td><td>{m["p95_err_px"]:.3f}</td></tr>')
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    (out/'real_predictions.json').write_text(json.dumps(dict(records=records,protocol=summary['real_protocol']),indent=2))
    title='5000 清晰+模糊配对训练评估'
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>'+title+'</title><style>body{max-width:1300px;margin:30px auto;background:#14212a;color:#eee;font:17px sans-serif}img{max-width:100%}td,th{padding:8px;border:1px solid #888}</style><h1>'+title+'</h1>'+
        ('<p>注意：这是限量管线检查，不是完整评估。</p>' if args.limit else '')+
        '<p>4000基础训练帧×清晰/模糊=8000训练图（16000实例）。清晰验证500帧选权重；模糊验证及清晰/模糊测试分别报告。误差为256输入像素。旧模型为3305模型，不是200张模型。</p><p>数据量、模糊及每轮更新次数同时变化，不是模糊单因素消融。真实帧无可靠角点真值，只作定性诊断。</p><table><tr><th>模型</th><th>集合</th><th>实例</th><th>均值px</th><th>PCK@5</th><th>P95px</th></tr>'+''.join(rows)+'</table>'+''.join(parts))
    print(out/'index.html')


if __name__=='__main__':main()
