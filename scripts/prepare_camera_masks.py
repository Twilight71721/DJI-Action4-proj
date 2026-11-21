"""Pack full camera silhouettes, preserving existing occluder pixels inside the silhouette."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def repair_visible_mask(visible, full):
    if visible.shape != full.shape:
        raise ValueError('Visible and full mask shapes differ')
    # Keep photographed/rendered occluder pixels inside the camera silhouette.
    # Replacing these with background creates artificial holes in the body.
    return ((full>0)*255).astype('uint8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=ROOT/'dataset/action4-reference-5000-20260921/train_pbr')
    p.add_argument('--manifest',type=Path,default=ROOT/'dataset/action4-cleanblur5000-20260923/manifest.json')
    p.add_argument('--output',type=Path,default=ROOT/'assets/background_randomization_20260923/camera_masks')
    args=p.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False);cv2.setNumThreads(1)
    frames=[f for f in json.loads(args.manifest.read_text())['frames'] if f['split']=='train']
    def process(item):
        f,i=item;scene=f['scene'];frame=int(f['frame']);name=f'{frame:06d}_{i:06d}.png'
        visible=cv2.imread(str(args.source/scene/'mask_visib'/name),0)
        full=cv2.imread(str(args.source/scene/'mask'/name),0)
        if visible is None or full is None:raise FileNotFoundError((scene,name))
        ys,xs=np.nonzero(full)
        if not len(xs):raise ValueError('Empty object mask')
        x0=max(0,int(xs.min())-3);y0=max(0,int(ys.min())-3)
        x1=min(full.shape[1],int(xs.max())+4);y1=min(full.shape[0],int(ys.max())+4)
        v=visible[y0:y1,x0:x1];roi=repair_visible_mask(v,full[y0:y1,x0:x1])
        path=out/scene/name;path.parent.mkdir(exist_ok=True)
        cv2.imwrite(str(path),roi,[cv2.IMWRITE_PNG_COMPRESSION,9])
        return f'{scene}/{frame:06d}/{i}',dict(path=str(path.relative_to(out)),roi_xywh=[x0,y0,x1-x0,y1-y0],
            source_hw=list(full.shape),visible_pixels=int((v>0).sum()),mask_pixels=int((roi>0).sum()),
            repaired_pixels=int(((roi>0)&(v==0)).sum()),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    tasks=[(f,i) for f in frames for i in range(2)]
    with ThreadPoolExecutor(max_workers=6) as pool:records=dict(pool.map(process,tasks))
    result=dict(version=2,count=len(records),mask_policy='Full rendered body silhouette; preserve original RGB including any occluder pixels inside silhouette. No hidden texture is reconstructed.',
                empty_visible_masks=sum(v['visible_pixels']==0 for v in records.values()),
                total_bytes=sum((out/v['path']).stat().st_size for v in records.values()),records=records)
    (out/'manifest.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)


if __name__=='__main__':main()
