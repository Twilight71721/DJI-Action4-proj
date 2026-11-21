"""Reconcile BOP pixel statistics with delivered masks, retaining native JSON."""
import argparse
import json
import shutil
from pathlib import Path
import cv2
import numpy as np


def bbox(mask):
    y,x=np.where(mask)
    return [int(x.min()),int(y.min()),int(x.max()-x.min()),int(y.max()-y.min())] if len(x) else [-1]*4


def reconcile(root):
    backup=root/'native_annotations'
    if backup.exists():raise RuntimeError('Native backups already exist; do not repeat reconciliation')
    backup.mkdir();changes=[]
    for scene in sorted((root/'train_pbr').glob('[0-9]*')):
        native=backup/scene.name;native.mkdir()
        for name in ('scene_gt_info.json','scene_gt_coco.json'):shutil.copy2(scene/name,native/name)
        info=json.loads((scene/'scene_gt_info.json').read_text())
        coco=json.loads((scene/'scene_gt_coco.json').read_text())
        for fid,instances in info.items():
            depth=cv2.imread(str(scene/'depth'/f'{int(fid):06d}.png'),cv2.IMREAD_UNCHANGED)
            eligible=[]
            for i,item in enumerate(instances):
                mask=cv2.imread(str(scene/'mask'/f'{int(fid):06d}_{i:06d}.png'),0)>0
                visible=cv2.imread(str(scene/'mask_visib'/f'{int(fid):06d}_{i:06d}.png'),0)>0
                if np.any(visible & ~mask):raise RuntimeError('Visible mask not subset')
                old=dict(item);h,w=mask.shape
                bx,by,bw,bh=item['bbox_obj']
                cropped=(bx<0 or by<0 or bx+bw>=w or by+bh>=h or mask[0].any() or mask[-1].any() or mask[:,0].any() or mask[:,-1].any())
                if not cropped:
                    item['px_count_all']=int(mask.sum());item['bbox_obj']=bbox(mask) if visible.any() else [-1]*4
                item['px_count_valid']=int((mask & (depth>0)).sum())
                item['px_count_visib']=int(visible.sum());item['bbox_visib']=bbox(visible)
                item['visib_fract']=item['px_count_visib']/max(item['px_count_all'],1)
                if item!=old:changes.append(dict(scene=scene.name,frame=int(fid),instance=i,before=old,after=dict(item)))
                if visible.any() and mask.any():eligible.append(item)
            annotations=sorted([a for a in coco['annotations'] if int(a['image_id'])==int(fid)],key=lambda a:a['id'])
            if len(annotations)!=len(eligible):raise RuntimeError('COCO instance order/count mismatch')
            for annotation,item in zip(annotations,eligible):
                if annotation['area']!=item['px_count_visib']:raise RuntimeError('COCO area differs from mask')
                annotation['ignore']=bool(item['visib_fract']<.1)
        for image in coco['images']:
            path=Path(image['file_name']).with_suffix('.png')
            if not (scene/path).is_file():raise RuntimeError('COCO RGB missing')
            image['file_name']=str(path)
        for name,data in [('scene_gt_info.json',info),('scene_gt_coco.json',coco)]:
            (scene/name).write_text(json.dumps(data,indent=2))
    (root/'annotation_reconciliation.json').write_text(json.dumps(dict(changes=changes,rgb_depth_masks_poses_modified=False),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    reconcile(parser.parse_args().root.resolve())
