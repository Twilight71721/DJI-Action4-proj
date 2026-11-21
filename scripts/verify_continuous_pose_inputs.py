"""Inspect the actual network inputs and verify labels survive online augmentation."""
import argparse
import json
from pathlib import Path
import sys
import cv2
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from bop_bb8 import BOPModel,BOPInstanceDataset
from bop_bb8.online_background import OnlinePoseBackgroundDataset
from review_pose_replacement_pilot import draw


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepared',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();conf=json.loads((a.prepared/'online_config.json').read_text());a.output.mkdir(parents=True,exist_ok=False)
    scenes=sorted(p.name for p in (a.prepared/'trainval').iterdir() if p.name not in conf['val_scenes'])
    kwargs=dict(scenes=scenes,input_size=(256,256),heatmap_size=(64,64),crop_mode='bbox',crop_padding=1.3,heatmap_kernel='gaussian',heatmap_sigma=2.,crop_jitter_seed=conf['seed'])
    model=BOPModel.from_models_info(a.prepared/'models/models_info.json')
    base=BOPInstanceDataset(a.prepared/'trainval',model,**kwargs)
    data=OnlinePoseBackgroundDataset(a.prepared/'trainval',model,online_background_config=a.prepared/'online_config.json',**kwargs)
    old=[i for i,v in enumerate(data.instances) if v[0] in conf['old_train_scenes']]
    new=[i for i,v in enumerate(data.instances) if v[0] not in conf['old_train_scenes']]
    indices=[old[0],old[1],old[len(old)//2],old[len(old)//2+1]]+new[::2][:48]
    cards=[];checks=[]
    for number,i in enumerate(indices):
        reference=base[i];images=[]
        for epoch in [0,1,2]:
            data.set_epoch(epoch);s=data[i]
            for key in ['keypoints_2d','valid','heatmaps','K_input','R_m2c','t_m2c']:
                np.testing.assert_array_equal(getattr(reference,key),getattr(s,key))
            images.append(cv2.cvtColor(s.image,cv2.COLOR_RGB2BGR))
        assert not np.array_equal(images[0],images[1])
        data.set_epoch(0);pair=data[i+1] if i in new else data[i]
        if i in new:
            np.testing.assert_array_equal(pair.keypoints_2d,reference.keypoints_2d)
            assert not np.array_equal(pair.image,data[i].image)
        panel=np.hstack([*images,cv2.cvtColor(pair.image,cv2.COLOR_RGB2BGR),draw(images[0],reference.keypoints_2d)])
        fn=f'sample_{number:03d}.jpg';cv2.imwrite(str(a.output/fn),panel)
        cards.append(f'<h3>{data.instances[i]}</h3><img src="{fn}">')
        checks.append(dict(instance=data.instances[i],unchanged_geometry=True,background_changes=True))
    (a.output/'verification.json').write_text(json.dumps(dict(checks=checks,train_instances=len(data),formal_training=False),indent=2))
    (a.output/'index.html').write_text('<meta charset="utf-8"><style>body{background:#16242d;color:white;font:18px system-ui;margin:30px}img{max-width:100%}</style><h1>连续三维姿态＋混合背景：实际网络输入</h1><p>每行：epoch0清晰 / epoch1 / epoch2 / 新样本的epoch0模糊伙伴（旧样本此列仍为自身） / epoch0标签叠加。绿色框只用于检查，不进入网络。</p><p>本地64个前景用于流程验证；服务器计划4096训练＋256验证，尚未正式训练。</p>'+''.join(cards))
    print('VERIFIED',len(checks),'actual network inputs')


if __name__=='__main__':main()
