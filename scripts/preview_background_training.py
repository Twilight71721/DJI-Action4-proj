"""Verify online composition on real dataset entries and show clean/blur examples."""
import argparse
import json
from pathlib import Path
import sys
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from bop_bb8 import BOPInstanceDataset,BOPModel
from bop_bb8.background import RandomBackgroundBOPDataset,mask_in_crop
from bop_bb8.visualize import EDGES


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--background-assets',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    cv2.setNumThreads(1)
    manifest=json.loads((args.dataset/'manifest.json').read_text())
    scenes=manifest['splits']['train']+[f'{100000+int(s):06d}' for s in manifest['splits']['train']]
    kwargs=dict(scenes=scenes,input_size=(256,256),heatmap_size=(64,64),crop_mode='bbox',crop_padding=1.3,
         heatmap_kernel='gaussian',crop_jitter_shift=.05,crop_jitter_padding=(1.2,1.4),crop_jitter_seed=20260924)
    model=BOPModel.from_models_info(args.dataset/'models/models_info.json')
    base=BOPInstanceDataset(args.dataset/'trainval',model,**kwargs)
    augmented=RandomBackgroundBOPDataset(args.dataset/'trainval',model,**kwargs,
         background_assets=args.background_assets,background_probability=1.)
    max_label_error=0.;changed=0;valid_changes=0
    for idx in np.linspace(0,len(base)-1,120,dtype=int):
        for epoch in range(3):
            base.set_epoch(epoch);augmented.set_epoch(epoch)
            before=base[int(idx)];after=augmented[int(idx)]
            max_label_error=max(max_label_error,float(np.nanmax(np.abs(before.keypoints_2d-after.keypoints_2d))))
            valid_changes+=int((before.valid!=after.valid).sum())
            for field in ['heatmaps','K_input','R_m2c','t_m2c']:
                np.testing.assert_array_equal(getattr(before,field),getattr(after,field))
            changed+=int(not np.array_equal(before.image,after.image))
    if max_label_error!=0 or valid_changes or changed!=360:raise RuntimeError('Composition label/image verification failed')
    scenes_show=[s for s in ['000034','000001','000020','000064','000106','000158'] if s in scenes]
    rows=[];preview_records=[]
    for scene in scenes_show:
        for variant in ['clean','blur']:
            key=scene if variant=='clean' else f'{100000+int(scene):06d}'
            index=next(i for i,x in enumerate(base.instances) if x[0]==key and x[2]==0)
            base.set_epoch(0);original=base[index]
            record=augmented.masks[f'{scene}/{original.image_id:06d}/0']
            native=cv2.imread(str(args.background_assets/'camera_masks'/record['path']),0)
            mask=mask_in_crop(native,record['roi_xywh'],original.crop_xywh)
            mask=cv2.resize(mask,(256,256),interpolation=cv2.INTER_NEAREST)
            outputs=[];metas=[]
            # Choose the first random examples of each provenance solely for readable preview;
            # actual training samples uniformly from the combined pool.
            for prefix in ['images/real_','images/web_']:
                for seed in range(100):
                    image,meta=augmented.replacement(original,np.random.default_rng(seed))
                    if meta['background'].startswith(prefix):break
                outputs.append(image);metas.append(meta)
            labels=outputs[-1].copy();pts=np.rint(original.keypoints_2d).astype(int)
            for a,b in EDGES:
                if original.valid[a] and original.valid[b]:cv2.line(labels,tuple(pts[a]),tuple(pts[b]),(40,255,40),1)
            images=[original.image,cv2.cvtColor(mask,cv2.COLOR_GRAY2RGB),*outputs,labels]
            panel=np.vstack([np.zeros((26,1280,3),np.uint8),np.hstack(images)])
            for i,name in enumerate(['Original '+variant,'Camera mask','Real video background','Diverse photo background','Same corner labels']):
                cv2.putText(panel,name,(i*256+4,18),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1)
            filename=f'{scene}_{variant}.jpg';cv2.imwrite(str(out/filename),cv2.cvtColor(panel,cv2.COLOR_RGB2BGR))
            rows.append(f'<h3>{scene} / {variant}</h3><img src="{filename}">')
            preview_records.append(dict(scene=scene,variant=variant,frame=original.image_id,backgrounds=metas))
    verification=dict(instances=120,epochs=3,compositions_checked=360,changed_images=changed,
          max_label_error_px=max_label_error,validity_changes=valid_changes,
          train_instances=len(base),formal_training_started=False)
    (out/'verification.json').write_text(json.dumps(verification,indent=2))
    (out/'preview_records.json').write_text(json.dumps(preview_records,indent=2))
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>随机背景训练样本</title>'
         '<style>body{background:#14212a;color:#eee;font:17px/1.6 system-ui;max-width:1320px;margin:30px auto}img{max-width:100%}</style>'
         '<h1>统一混合背景 · 单一训练方案</h1><p>1024张外部照片 + 来自256个真实帧的512个背景区域。'
         '训练时80%概率换背景，20%保留原图；清晰／模糊配对及裁剪扰动保留。背景混用，不按来源划分。</p>'
         '<p>右侧绿色线是用于检查的原有标签，模型输入不带线。这里展示数据增强，不是训练后的预测。</p>'
         '<p>模糊分支从清晰前景先合成，再按原模糊参数整体处理，减少旧背景边缘残留。使用完整机身投影轮廓，轮廓内原有遮挡像素保留，不凭空恢复隐藏纹理。</p>'
         '<pre>'+json.dumps(verification,indent=2)+'</pre>'+''.join(rows))
    print(json.dumps(verification,indent=2))


if __name__=='__main__':main()
