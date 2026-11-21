"""Evaluate held-out poses across four fixed unseen backgrounds, with GT and stability."""
import argparse,json,sys
from pathlib import Path
import cv2,numpy as np,torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from bop_bb8 import BOPInstanceDataset,BOPModel
from bop_bb8.torch_data import HeatmapDataset
from evaluate_cleanblur5000 import load_model,predict
from evaluate_heatmap_checkpoint import metric_summary,draw_overlay


def main():
    p=argparse.ArgumentParser()
    for name in ['prepared','baseline','checkpoint','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    conf=json.loads((a.prepared/'online_config.json').read_text());modelinfo=BOPModel.from_models_info(a.prepared/'models/models_info.json')
    ds=BOPInstanceDataset(a.prepared/'trainval',modelinfo,scenes=[conf['new_val_scene']],input_size=(256,256),heatmap_size=(64,64),crop_mode='bbox',crop_padding=1.3,heatmap_kernel='gaussian',heatmap_sigma=2.)
    expected=conf.get('expected_new_val_instances',192)
    summary=dict(incomplete_diagnostic=len(ds)!=expected,expected_instances=expected,actual_instances=len(ds),protocol='Held-out rendered poses and background source images. Four backgrounds per pose; clean and blur paired. This is validation used for checkpoint selection, not an independent test. No real GT.',models={});cards=[]
    for name,path in [('before',a.baseline),('trained',a.checkpoint)]:
        model,meta=load_model(path,torch.device('cuda'));preds=[];targets=[];valid=[]
        for b in DataLoader(HeatmapDataset(ds),batch_size=16,shuffle=False,num_workers=0):
            pred,_=predict(model,b['image'],torch.device('cuda'));preds.append(pred);targets.append(b['keypoints_2d'].numpy());valid.append(b['valid'].numpy())
        pred,target,v=map(np.concatenate,[preds,targets,valid]);err=np.linalg.norm(pred-target,axis=-1);err[~v]=np.nan
        entries={}
        for variant,label in [(0,'clean'),(1,'blur')]:
            points=pred[variant::2].reshape(-1,4,8,2)
            displacement=np.linalg.norm(points-points.mean(axis=1,keepdims=True),axis=-1)
            entries[label]=dict(metric_summary(err[variant::2]),mean_background_displacement_px=float(displacement.mean()))
        summary['models'][name]=dict(metadata=meta,metrics=entries)
        np.savez_compressed(a.output/f'{name}.npz',predictions=pred,targets=target,valid=v,errors=err)
        for pose in range(len(ds)//8):
            pane=np.hstack([draw_overlay(ds[pose*8+i*2].image,target[pose*8+i*2],pred[pose*8+i*2],v[pose*8+i*2],f'{name} background {i}') for i in range(4)])
            fn=f'{name}_{pose:02}.jpg';cv2.imwrite(str(a.output/fn),pane);cards.append(f'<h3>{name} pose {pose}: four backgrounds</h3><img src="{fn}">')
        del model;torch.cuda.empty_cache()
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2))
    (a.output/'index.html').write_text('<meta charset="utf-8"><h1>固定姿态跨背景评估</h1><p>每个留出姿态4种固定留出背景，分别评估清晰和模糊。绿色GT，紫色预测。稳定不等于准确；incomplete_diagnostic=true表示仅流程检查。</p><pre>'+json.dumps({k:v['metrics'] for k,v in summary['models'].items()},indent=2)+'</pre>'+''.join(cards))


if __name__=='__main__':main()
