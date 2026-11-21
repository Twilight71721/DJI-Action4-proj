"""Audit actual view directions and compare augmentation pilots on frozen validation inputs."""
import argparse
import csv
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from bop_bb8 import BOPInstanceDataset,BOPModel
from bop_bb8.torch_data import HeatmapDataset,IMAGENET_MEAN,IMAGENET_STD
from bop_bb8.photometric import PhotometricDataset,augment_rgb
from bop_bb8.view_sampling import dataset_view_bins,view_bin,FACE_NAMES
from report_paired_background_pilot import load,decode,tensor
from infer_resnet_yolo_real import to_original,draw_points


def stats(errors):
    a=np.asarray(errors,float)
    return dict(corners=int(a.size),mean_error=float(a.mean()) if a.size else None,
                pck5=float((a<=5).mean()) if a.size else None)


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output
    cv2.setNumThreads(1);torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    config=json.loads((out/'control/config.json').read_text())
    model_info=BOPModel.from_models_info(Path(config['models_info']),1,1.)
    kwargs=dict(input_size=tuple(config['input_size']),heatmap_size=(64,64),crop_mode='rect',
        crop_padding=1.,crop_border=0,rect_sizes=config['rect_sizes'],heatmap_margin=config['heatmap_margin'],
        heatmap_kernel='gaussian',heatmap_sigma=config['heatmap_sigma'],require_all_corners=True)
    train=BOPInstanceDataset(Path(config['dataset_root']),model_info,scenes=config['train_scenes'],
        min_visib_fract=config['train_min_visib_fract'],**kwargs)
    val=BOPInstanceDataset(Path(config['dataset_root']),model_info,scenes=config['val_scenes'],**kwargs)
    train_labels=dataset_view_bins(HeatmapDataset(train));val_labels=dataset_view_bins(HeatmapDataset(val))
    center=(model_info.min_xyz_mm+model_info.max_xyz_mm)/2
    bank_path=ROOT/'outputs/server_return_continuous_pose_20260925/continuous_pose_background_30ep/foreground_manifest.json'
    assets=json.loads(bank_path.read_text())['samples']
    bank_labels=[view_bin(r['R'],r['t_mm'],center) for r in assets if r['split']=='train']
    counts=lambda labels:dict(zip(FACE_NAMES,np.bincount(labels,minlength=6).tolist()))
    audit=dict(local_train_instances=counts(train_labels),local_val_instances=counts(val_labels),
        server_distinct_train_pose_assets=counts(bank_labels),server_asset_manifest=str(bank_path),
        caveat='4096 pose assets are not the 39996 training instances. Local data is a smaller subset.',
        definition='Dominant component of observer direction -R^T t - object center. X lateral, Y top/bottom, Z front/rear according to renderer.')
    (out/'view_audit.json').write_text(json.dumps(audit,indent=2))
    selected=np.linspace(0,len(val)-1,min(config['limit'],len(val)),dtype=int)
    base=HeatmapDataset(val);cached=[base[int(i)] for i in selected]
    augmented=PhotometricDataset(cached,seed=20260928)
    degraded=[augmented[i] for i in range(len(cached))]
    assert all(torch.equal(a['keypoints_2d'],b['keypoints_2d']) for a,b in zip(cached,degraded))
    preview_ids=[]
    for label in range(6):
        indices=np.flatnonzero(train_labels==label)
        if len(indices):preview_ids.append(int(indices[len(indices)//2]))
    preview_cards=[];preview_metadata=[]
    for index in preview_ids:
        sample=train[index];rgb=sample.image
        views=[rgb];settings=[]
        for seed in [11,19,31]:
            image,meta=augment_rgb(rgb,np.random.default_rng(seed));views.append(image);settings.append(meta)
        panels=[]
        for title,image in zip(['original','variation 1','variation 2','variation 3'],views):
            bgr=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
            # Show RGB without drawn labels; labels are verified equal in the training data path.
            panel=cv2.resize(bgr,(300,round(300*bgr.shape[0]/bgr.shape[1])))
            panel=cv2.copyMakeBorder(panel,28,0,0,0,cv2.BORDER_CONSTANT)
            cv2.putText(panel,title,(8,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
            panels.append(panel)
        filename=f'augmentation_{FACE_NAMES[train_labels[index]]}.jpg'
        cv2.imwrite(str(out/filename),np.hstack(panels))
        preview_cards.append(f'<h3>{FACE_NAMES[train_labels[index]]}</h3><img src="{filename}">')
        preview_metadata.append(dict(instance=train.instances[index],settings=settings))
    (out/'augmentation_preview.json').write_text(json.dumps(preview_metadata,indent=2))
    real_root=ROOT/'outputs/server_return_rect_r18_20260926/rect_r18_fixed_yolo_30ep/real'
    real=json.loads((real_root/'network/predictions.json').read_text())['instances']
    real_points={r['id']:{} for r in real}
    results={}
    for name,path in [('initial',Path(config['init_checkpoint'])),('control',out/'control/last.pth'),
                      ('photo',out/'photo/last.pth'),('photo_view',out/'photo_view/last.pth')]:
        model,saved=load(path);results[name]={}
        for variant,items in [('clean',cached),('degraded',degraded)]:
            all_errors=[];per_view={k:[] for k in FACE_NAMES}
            for pos,row in enumerate(items):
                image=row['image'];size=(image.shape[-1],image.shape[-2])
                logits=model(image[None].cuda()).float()
                points=decode(logits,size,saved)[0].cpu()
                wh=row['crop_xywh'][2:].float()
                delta=(points-row['keypoints_2d'])*wh/torch.tensor(size)*256/wh.max()
                errors=torch.linalg.vector_norm(delta,dim=-1)[row['valid']].tolist()
                all_errors.extend(errors);per_view[FACE_NAMES[val_labels[selected[pos]]]].extend(errors)
            results[name][variant]=dict(overall=stats(all_errors),views={k:stats(v) for k,v in per_view.items()})
        for r in real:
            image=cv2.imread(str(real_root/'network/inputs'/f'{r["id"]}.png'))
            rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB);size=(image.shape[1],image.shape[0])
            points=decode(model(tensor(rgb)[None].cuda()).float(),size,saved)[0].cpu().numpy()
            real_points[r['id']][name]=to_original(points,r['crop_xywh'],size).tolist()
        del model;torch.cuda.empty_cache()
        print(name,json.dumps(results[name]['clean']['overall']),json.dumps(results[name]['degraded']['overall']),flush=True)
    (out/'evaluation.json').write_text(json.dumps(dict(protocol='Fixed 512 validation instances; FP32; all valid corners; body long side256. Photometric probe is simulated, not real accuracy.',
        instances=[val.instances[int(i)] for i in selected],models=results),indent=2))
    (out/'real_predictions.json').write_text(json.dumps(real_points,indent=2))
    real_cards=[]
    for r in real:
        image=cv2.imread(str(real_root/'source_frames/frames'/Path(r['source']).name))
        box=np.array(r['yolo_box_xyxy']);side=max(box[2:]-box[:2])*1.5;origin=(box[:2]+box[2:])/2-side/2;scale=320/side
        transform=np.array([[scale,0,-origin[0]*scale],[0,scale,-origin[1]*scale]])
        panels=[]
        for name in results:
            crop=cv2.warpAffine(image,transform,(320,320))
            crop=draw_points(crop,(np.array(real_points[r['id']][name])-origin)*scale)
            crop=cv2.copyMakeBorder(crop,28,0,0,0,cv2.BORDER_CONSTANT)
            cv2.putText(crop,name,(8,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1);panels.append(crop)
        filename=f'real_{r["id"]}.jpg';cv2.imwrite(str(out/filename),np.hstack(panels))
        real_cards.append(f'<h3>{r["id"]}</h3><img src="{filename}">')
    rows=[]
    for name,r in results.items():
        a,b=r['clean']['overall'],r['degraded']['overall']
        rows.append(f'<tr><td>{name}</td><td>{a["mean_error"]:.3f}</td><td>{a["pck5"]:.2%}</td><td>{b["mean_error"]:.3f}</td><td>{b["pck5"]:.2%}</td></tr>')
    audit_rows=''.join(f'<tr><td>{k}</td><td>{audit["local_train_instances"][k]}</td><td>{audit["server_distinct_train_pose_assets"][k]}</td></tr>' for k in FACE_NAMES)
    view_rows=''.join('<tr><td>'+k+'</td>'+''.join(f'<td>{results[name]["clean"]["views"][k]["mean_error"]:.2f}</td>'
        if results[name]['clean']['views'][k]['mean_error'] is not None else '<td>无样本</td>' for name in results)+'</tr>' for k in FACE_NAMES)
    (out/'index.html').write_text('''<!doctype html><meta charset="utf-8"><title>成像增强与视角采样：本地实验</title>
<style>body{background:#172630;color:#eee;font:18px/1.7 sans-serif;max-width:1350px;margin:30px auto;padding:20px}img{max-width:100%}td,th{border:1px solid #789;padding:8px}table{border-collapse:collapse}a{color:#9df}</style>
<h1>成像增强与视角采样：本地实验</h1>
<p>保留ResNet18与空间KL损失；固定YOLO。三组从同一最佳权重继续训练512个实例、3轮，最后一轮统一比较。
control：原训练；photo：增加颜色、运动模糊、噪声、JPEG增强；photo_view：相同增强＋有上限的稀缺视角加权。没有裁剪扰动、额外遮挡或真实角点微调。</p>
<h2>视角统计</h2><p>X为模型左右方向，Y为上下方向，Z为正背面方向；按观察方向最大分量归类。这是粗分箱，不能证明具体困难角度覆盖充分。
本地训练实例和服务器4096个独立前景姿态不是同一计数单位，不能直接比较数量。服务器前景X两侧已有2027个，不能笼统归因于侧面总数不足。</p>
<table><tr><th>方向</th><th>本地训练实例</th><th>服务器独立训练姿态</th></tr>'''+audit_rows+'''</table>
<h2>固定验证集结果</h2><p>512个合成验证实例，FP32，误差按机身框长边256像素归一。增强验证是固定随机种子的模拟退化，不是真实准确率。</p>
<table><tr><th>模型</th><th>原图误差↓</th><th>原图PCK5↑</th><th>增强图误差↓</th><th>增强图PCK5↑</th></tr>'''+''.join(rows)+'''</table>
<h2>原图分视角误差</h2><table><tr><th>方向</th><th>initial</th><th>control</th><th>photo</th><th>photo_view</th></tr>'''+view_rows+'''</table>
<p><a href="evaluation.json">完整指标及各视角样本数</a> · <a href="view_audit.json">视角审计</a> · <a href="protocol.json">训练配置</a></p>
<h2>成像增强预览</h2><p>以下为网络RGB输入；所有变换均不改变几何标签。20%概率保留原图，不做几何扭曲。</p>'''+''.join(preview_cards)+
'''<h2>真实帧定性对照</h2><p>原权重、对照、成像增强、增强＋加权。这里使用原始编号连线，不使用重连或PnP；无可信真实八点真值，不报告真实准确率。</p>'''+''.join(real_cards))


if __name__=='__main__':main()
