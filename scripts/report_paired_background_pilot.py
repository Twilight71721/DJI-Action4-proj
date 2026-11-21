"""Report paired training and a fixed held-background sensitivity probe without real GT claims."""
import argparse
import csv
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bop_bb8 import BOPModel
from bop_bb8.online_background import OnlinePoseBackgroundDataset
from bop_bb8.torch_data import IMAGENET_MEAN, IMAGENET_STD
from bop_bb8.net import ResNetFPN, decode_heatmaps, heatmap_to_input_points
from bop_bb8.net.loss import spatial_js_loss, spatial_kl_loss
from infer_resnet_yolo_real import draw_points, to_original


def tensor(rgb):
    x = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float()/255
    return (x-torch.tensor(IMAGENET_MEAN)[:,None,None])/torch.tensor(IMAGENET_STD)[:,None,None]


def decode(logits, size, saved):
    return heatmap_to_input_points(decode_heatmaps(logits, saved['decoder'], saved['decode_radius']),
        logits.shape[-2:], size[::-1], margin=saved['heatmap_margin'])


def load(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    saved = checkpoint['args']
    model = ResNetFPN(backbone=saved['backbone'], fpn_dim=saved['fpn_dim'], pretrained=False,
                      heatmap_margin=saved['heatmap_margin']).cuda().eval()
    model.load_state_dict(checkpoint['model'], strict=True)
    return model, saved


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = json.loads((out/'control/config.json').read_text())
    model_info = BOPModel.from_models_info(Path(config['models_info']), 1, 1.)
    data = OnlinePoseBackgroundDataset(Path(config['dataset_root']), model_info,
        scenes=config['train_scenes'], online_background_config=config['online_background_config'],
        input_size=tuple(config['input_size']), heatmap_size=(64,64), crop_mode='rect',
        crop_padding=1., crop_border=0, rect_sizes=config['rect_sizes'], heatmap_margin=config['heatmap_margin'],
        heatmap_kernel='gaussian', heatmap_sigma=config['heatmap_sigma'], require_all_corners=True,
        min_visib_fract=config['train_min_visib_fract'], crop_jitter_seed=config['seed'])
    selected = np.linspace(0, len(data)-1, min(config['limit'],len(data)), dtype=int)
    indices = selected[np.linspace(0, len(selected)-1, 32, dtype=int)]
    # Seen geometry, but unseen background sources: a sensitivity diagnostic, not a pose holdout.
    data.background_files = [Path(p) for p in data.online_config['heldout_backgrounds']]
    assert set(data.background_files).isdisjoint(map(Path,data.online_config['train_backgrounds']))
    data.legacy.backgrounds = [dict(path=str(p)) for p in data.background_files]
    data.set_epoch(10000)
    pairs = [data.load_pair(int(i)) for i in indices]
    previews = []
    for i,(a,b) in enumerate(pairs[:8]):
        views=[]
        for sample in [a,b]:
            rgb=draw_points(cv2.cvtColor(sample.image,cv2.COLOR_RGB2BGR),sample.keypoints_2d)
            views.append(cv2.resize(rgb,(384,round(384*rgb.shape[0]/rgb.shape[1]))))
        name=f'pair_{i:02d}.jpg'
        cv2.imwrite(str(out/name),np.hstack(views))
        previews.append(name)
    result = dict(probe_protocol='32 seen training geometries, 2 fixed held-out backgrounds each; FP32; no real GT.',
                  probe_instances=[data.instances[int(i)] for i in indices],
                  changed_rgb_pixel_fraction=float(np.mean([
                      (np.abs(a.image.astype(float)-b.image.astype(float)).max(-1)>2).mean() for a,b in pairs])),
                  models={})
    source = ROOT/'outputs/server_return_rect_r18_20260926/rect_r18_fixed_yolo_30ep/real'
    real = json.loads((source/'network/predictions.json').read_text())['instances']
    real_points = {r['id']:{} for r in real}
    for name,path in [('initial',Path(config['init_checkpoint'])),('control',out/'control/last.pth'),
                      ('consistency',out/'consistency/last.pth')]:
        model,saved=load(path)
        js_values, kl_values, shifts, errors, hits = [], [], [], [], []
        for a,b in pairs:
            images=torch.stack([tensor(a.image),tensor(b.image)]).cuda()
            logits=model(images).float()
            target=torch.from_numpy(a.heatmaps)[None].cuda()
            valid=torch.from_numpy(a.valid)[None].cuda()
            js_values.append(spatial_js_loss(logits[:1],logits[1:],valid).item())
            kl_values.append(spatial_kl_loss(logits,target.repeat(2,1,1,1),valid.repeat(2,1)).item())
            size=(a.image.shape[1],a.image.shape[0])
            points=decode(logits,size,saved).cpu().numpy()
            scale=np.array(a.crop_xywh[2:])/size*256/max(a.crop_xywh[2:])
            shifts.extend(np.linalg.norm((points[0]-points[1])*scale,axis=-1)[a.valid].tolist())
            err=np.linalg.norm((points-a.keypoints_2d)*scale,axis=-1)[:,a.valid].ravel()
            errors.extend(err.tolist());hits.extend((err<=5).tolist())
        result['models'][name]=dict(mean_js=float(np.mean(js_values)),mean_supervised_kl=float(np.mean(kl_values)),
            background_shift_px256=float(np.mean(shifts)),probe_error_px256=float(np.mean(errors)),
            probe_pck5=float(np.mean(hits)))
        for r in real:
            image=cv2.imread(str(source/'network/inputs'/f'{r["id"]}.png'))
            rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            logits=model(tensor(rgb)[None].cuda()).float()
            size=(image.shape[1],image.shape[0])
            points=decode(logits,size,saved)[0].cpu().numpy()
            real_points[r['id']][name]=to_original(points,r['crop_xywh'],size).tolist()
        del model
        torch.cuda.empty_cache()
    for name in ['control','consistency']:
        metrics=list(csv.DictReader((out/name/'metrics.csv').open()))
        result['models'][name]['validation_history']=metrics
        result['models'][name]['training_terms']=[json.loads(line) for line in (out/name/'paired_terms.jsonl').read_text().splitlines()]
    control=result['models']['control']['training_terms']
    treated=result['models']['consistency']['training_terms']
    assert [(r['instances'],r['rgb_views'],r['optimizer_steps']) for r in control]==[(r['instances'],r['rgb_views'],r['optimizer_steps']) for r in treated]
    baselines=[json.loads((out/name/'baseline.json').read_text()) for name in ['control','consistency']]
    assert baselines[0] == baselines[1], 'Initial validation differs between experimental arms'
    result['initial_validation']=baselines[0]
    cards=[]
    for r in real:
        image=cv2.imread(str(source/'source_frames/frames'/Path(r['source']).name))
        if image is None: raise FileNotFoundError(r['source'])
        box=np.array(r['yolo_box_xyxy']);center=(box[:2]+box[2:])/2
        side=max(box[2:]-box[:2])*1.5;origin=center-side/2;scale=400/side
        matrix=np.array([[scale,0,-origin[0]*scale],[0,scale,-origin[1]*scale]])
        views=[]
        for name in ['initial','control','consistency']:
            crop=cv2.warpAffine(image,matrix,(400,400))
            points=(np.array(real_points[r['id']][name])-origin)*scale
            crop=draw_points(crop,points)
            crop=cv2.copyMakeBorder(crop,32,0,0,0,cv2.BORDER_CONSTANT)
            cv2.putText(crop,name,(10,23),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
            views.append(crop)
        filename=f'real_{r["id"]}.jpg';cv2.imwrite(str(out/filename),np.hstack(views))
        cards.append(f'<h3>{r["id"]}</h3><img src="{filename}">')
    (out/'diagnostic.json').write_text(json.dumps(result,indent=2))
    (out/'real_predictions.json').write_text(json.dumps(real_points,indent=2))
    initial_val=result['initial_validation']
    control_val=result['models']['control']['validation_history'][-1]
    consistency_val=result['models']['consistency']['validation_history'][-1]
    cprobe=result['models']['control'];tprobe=result['models']['consistency']
    (out/'ANALYSIS.md').write_text(f'''# 本地配对背景实验记录

两组同初始化、相同512个训练实例、每轮1024张RGB、3轮、每轮67次更新，共201次更新。训练配置除输出路径和一致性权重以外完全相同。两组训练前验证结果逐值一致。比较固定最后一轮，不使用真实帧选择模型。

512个合成验证实例：初始误差 {initial_val['err_px']:.4f}、PCK5 {initial_val['pck5']:.4%}；配对监督对照误差 {float(control_val['val_err_px']):.4f}、PCK5 {float(control_val['val_pck5']):.4%}；加0.1×JS后误差 {float(consistency_val['val_err_px']):.4f}、PCK5 {float(consistency_val['val_pck5']):.4%}。归一尺度为机身框长边256像素。

32个已见训练几何换到未训练背景的探针：对照换背景点位移 {cprobe['background_shift_px256']:.4f}，加一致性 {tprobe['background_shift_px256']:.4f}；对应角点真值误差 {cprobe['probe_error_px256']:.4f} 和 {tprobe['probe_error_px256']:.4f}。两组均为100% PCK5，该探针较容易，不能当作独立姿态泛化证据。两张图中平均 {result['changed_rgb_pixel_fraction']:.1%} 的RGB像素差值超过2灰度级。

在本次小范围、单随机种子、短时间继续训练中，一致性损失没有显示出明确的额外收益。合成验证误差略低于配对对照，但没有低于原权重；背景点位移也没有优于对照。不能据此断言更长训练或其他权重一定无效，也不能把微小差异解释为统计显著改善。

真实21个检测仅用于同框定性展示，未参与训练或权重选择，没有可靠全八点真值。请在index.html检查原模型、对照、一致性模型三列。历史失败点团仍需单独检查，不能用合成PCK声称真实问题解决。
''')
    rows=[]
    for name,m in result['models'].items():
        rows.append(f'<tr><td>{name}</td><td>{m["mean_js"]:.4f}</td><td>{m["background_shift_px256"]:.2f}</td><td>{m["probe_error_px256"]:.2f}</td><td>{m["probe_pck5"]:.2%}</td></tr>')
    initial=baselines[0]
    val_rows=f'<tr><td>initial</td><td>{initial["err_px"]:.3f}</td><td>{initial["pck5"]:.2%}</td></tr>'
    for name in ['control','consistency']:
        row=result['models'][name]['validation_history'][-1]
        val_rows+=f'<tr><td>{name}</td><td>{float(row["val_err_px"]):.3f}</td><td>{float(row["val_pck5"]):.2%}</td></tr>'
    (out/'index.html').write_text('''<!doctype html><meta charset="utf-8"><title>配对背景一致性：本地小规模对照</title>
<style>body{background:#172630;color:#edf3f7;font:18px/1.7 sans-serif;max-width:1300px;margin:30px auto;padding:20px}img{max-width:100%}td,th{border:1px solid #789;padding:9px}table{border-collapse:collapse}a{color:#9df}</style>
<h1>配对背景一致性：本地小规模对照</h1>
<p>两组从相同 ResNet18 权重继续训练，每组 512 个基础样本、每轮 1024 张图片、3 轮、相同随机种子和优化步数。
control：双份真值监督；consistency：相同监督＋0.1×JS。比较固定第 3 轮，不按真实帧挑选权重。没有真实角点微调。</p>
<h2>固定合成验证子集：512 个实例</h2><table><tr><th>模型</th><th>误差↓</th><th>PCK5↑</th></tr>'''+val_rows+'''</table>
<h2>换背景敏感性探针</h2>
<p>下表是 32 个已见训练几何换成未训练背景后的诊断，非独立姿态测试。位移与误差统一为机身长边256像素。稳定不等于正确，必须同时看角点误差。</p>
<table><tr><th>模型</th><th>背景对 JS↓</th><th>换背景点位移↓</th><th>真值误差↓</th><th>PCK5↑</th></tr>'''+''.join(rows)+'''</table>
<p><a href="diagnostic.json">完整指标与逐轮验证结果</a> · <a href="protocol.json">实验配置</a> · <a href="ANALYSIS.md">实验解读</a></p>
<h2>配对样本：同样的八点标签，不同背景</h2>'''+''.join(f'<img src="{p}">' for p in previews)+
'''<h2>真实帧对照</h2><p>左：原权重；中：配对监督；右：配对监督＋一致性。无可靠真实八点真值，不报告真实准确率。未使用 PnP。</p>'''+''.join(cards))
    print(json.dumps({k:{a:b for a,b in v.items() if a not in ['validation_history','training_terms']} for k,v in result['models'].items()},indent=2))


if __name__=='__main__':
    main()
