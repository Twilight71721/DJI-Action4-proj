"""Evaluate locked joint-decoder settings on the remaining clean validation instances."""
from pathlib import Path
import argparse
import json
import sys
import time

import cv2
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from bop_bb8 import BOPModel,BOPInstanceDataset
from bop_bb8.joint_decode import decode_joint_cuboid
from bop_bb8.net import ResNetFPN,decode_topk,heatmap_to_input_points
from bop_bb8.torch_data import IMAGENET_MEAN,IMAGENET_STD
from evaluate_heatmap_checkpoint import metric_summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    torch.set_num_threads(4);cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    root=ROOT/'dataset/action4-cleanblur5000-20260923'
    manifest=json.loads((root/'manifest.json').read_text())
    excluded=json.loads((ROOT/'outputs/local_cause_diagnostic_20260923/manifest.json').read_text())['cases']
    excluded={(c['scene'],c['frame'],c['instance']) for c in excluded if c['domain']=='synthetic'}
    info=BOPModel.from_models_info(root/'models/models_info.json')
    data=BOPInstanceDataset(root/'clean',info,scenes=sorted({j['scene'] for j in manifest['jobs'] if j['split']=='val'}),input_size=(256,256),heatmap_size=(64,64),crop_mode='bbox',crop_padding=1.3)
    indices=[i for i,key in enumerate(data.instances) if tuple(key) not in excluded]
    assert len(data)==1000 and len(indices)==880,(len(data),len(indices))
    ckpt=torch.load(ROOT/'outputs/server_return_5000_20260923/best.pth',map_location='cpu',weights_only=False)
    a=ckpt['args'];device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=ResNetFPN(backbone=a['backbone'],fpn_dim=a['fpn_dim'],pretrained=False).to(device).eval();model.load_state_dict(ckpt['model'],strict=True)
    errors={k:[] for k in ['top9','local_peak','joint_selected','joint_rigid','candidate_oracle']}
    records=[];valids=[];accepted=[];start=time.time();mean=torch.tensor(IMAGENET_MEAN)[:,None,None];std=torch.tensor(IMAGENET_STD)[:,None,None]
    for begin in range(0,len(indices),16):
        samples=[data[i] for i in indices[begin:begin+16]]
        tensors=[(torch.from_numpy(s.image.copy()).permute(2,0,1).float()/255-mean)/std for s in samples]
        n=len(samples);batch=torch.stack(tensors+[tensors[-1]]*(16-n)).to(device)
        with torch.inference_mode():
            prob=model(batch).float().sigmoid()[:n].cpu()
            top9=heatmap_to_input_points(decode_topk(prob,k=9),(64,64),(256,256)).numpy()
        for j,s in enumerate(samples):
            _,_,boxes=data._load_scene(s.scene_id)
            x,y,w,h=boxes[str(s.image_id)][s.instance_idx]['bbox_obj'];crop=np.array(s.crop_xywh)
            box=(np.array([[x,y],[x+w,y+h]])-crop[:2])*256/crop[2:]
            joint=decode_joint_cuboid(prob[j].numpy(),info.corners_mm(),box_xyxy=box.ravel())
            local=np.array([p['points'][0] if p['points'] else [np.nan,np.nan] for p in joint['candidates']])
            pred={'top9':top9[j],'local_peak':local}
            for k,v in [('joint_selected','selected_points'),('joint_rigid','rigid_points')]:
                pred[k]=np.array(joint[v]) if joint[v] is not None else np.full((8,2),np.nan)
            for k,p in pred.items():
                e=np.linalg.norm(p-s.keypoints_2d,axis=1);e[~s.valid]=np.nan;errors[k].append(e)
            oracle=np.array([min(np.linalg.norm(np.array(p['points'])-s.keypoints_2d[k],axis=1)) if p['points'] else np.nan for k,p in enumerate(joint['candidates'])])
            oracle[~s.valid]=np.nan;errors['candidate_oracle'].append(oracle)
            valids.append(s.valid);accepted.append(joint['accepted'])
            records.append(dict(scene=s.scene_id,frame=s.image_id,instance=s.instance_idx,status=joint['status'],changed_channels=joint.get('changed_channels',0)))
        if (begin//16+1)%10==0:
            print(f'{begin+n}/{len(indices)} elapsed={time.time()-start:.1f}s',flush=True)
    valid=np.array(valids);accept=np.array(accepted);metrics={}
    for k,e in errors.items():
        e=np.array(e);m=metric_summary(e);m['pck5_all_valid']=float(((e<=5)&valid).sum()/valid.sum())
        m['missing_corner_count']=int((valid&~np.isfinite(e)).sum())
        if k.startswith('joint'):
            m['pck5_rejections_count_as_failure']=float(((e<=5)&valid&accept[:,None]).sum()/valid.sum())
        metrics[k]=m
    summary=dict(instance_count=len(indices),excluded_diagnostic_instances=len(excluded),accepted_instances=int(accept.sum()),elapsed_seconds=time.time()-start,metrics=metrics,
                 note='Remaining validation subset, not an untouched test set. Fixed joint decoder settings; no retraining or parameter selection here.')
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    (args.output/'instances.json').write_text(json.dumps(records))
    np.savez_compressed(args.output/'errors.npz',**{k:np.array(v) for k,v in errors.items()},accepted=accept,valid=valid)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
