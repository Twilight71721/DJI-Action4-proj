"""Online background replacement using native camera masks; pose labels stay unchanged."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .dataset import BOPInstanceDataset
from .cropping import extract_window, fill_outside_box


def mask_in_crop(mask_roi, roi_xywh, crop_xywh):
    rx,ry,rw,rh=map(int,roi_xywh);x,y,w,h=map(int,crop_xywh)
    if mask_roi.shape != (rh,rw):
        raise ValueError('Mask ROI shape does not match metadata')
    canvas=np.zeros((h,w),np.uint8)
    left,top=max(rx,x),max(ry,y);right,bottom=min(rx+rw,x+w),min(ry+rh,y+h)
    if right>left and bottom>top:
        canvas[top-y:bottom-y,left-x:right-x]=mask_roi[top-ry:bottom-ry,left-rx:right-rx]
    return canvas


def compose_native(foreground, background, mask, blur_sigma=0., blur_kernel=1):
    if foreground.shape != background.shape or mask.shape != foreground.shape[:2]:
        raise ValueError('Foreground, background and mask sizes must match')
    # Feather only the boundary at half an input pixel; the interior stays opaque.
    edge_sigma=max(.3,.5*foreground.shape[1]/256)
    edge_kernel=2*int(np.ceil(3*edge_sigma))+1
    alpha=cv2.GaussianBlur(mask.astype(np.float32)/255,(edge_kernel,edge_kernel),edge_sigma)[...,None]
    mixed=np.clip(np.rint(alpha*foreground.astype(np.float32)+(1-alpha)*background),0,255).astype(np.uint8)
    if blur_sigma>0:
        mixed=cv2.GaussianBlur(mixed,(int(blur_kernel),int(blur_kernel)),float(blur_sigma))
    return mixed


class RandomBackgroundBOPDataset(BOPInstanceDataset):
    def __init__(self,*args,background_assets,background_probability=.8,**kwargs):
        super().__init__(*args,**kwargs)
        if not 0<=background_probability<=1:
            raise ValueError('background_probability must be in [0,1]')
        self.background_probability=float(background_probability)
        self.assets=Path(background_assets)
        self.backgrounds=json.loads((self.assets/'manifest.json').read_text())['records']
        self.masks=json.loads((self.assets/'camera_masks/manifest.json').read_text())['records']
        if not self.backgrounds:
            raise ValueError('Empty background pool')
        self.data_root=self.root.parent
        manifest=json.loads((self.data_root/'manifest.json').read_text())
        self.frames={(f['scene'],int(f['frame'])):f for f in manifest['frames']}
        if any(f'{int(s)%100000:06d}/{f:06d}/{i}' not in self.masks for s,f,i in self.instances):
            raise ValueError('Missing training instance mask')

    def replacement(self,sample,rng,exclude_background=None):
        original_scene=f'{int(sample.scene_id)%100000:06d}'
        key=f'{original_scene}/{sample.image_id:06d}/{sample.instance_idx}'
        record=self.masks[key]
        mask_roi=cv2.imread(str(self.assets/'camera_masks'/record['path']),cv2.IMREAD_GRAYSCALE)
        if mask_roi is None:
            raise FileNotFoundError(record['path'])
        mask=mask_in_crop(mask_roi,record['roi_xywh'],sample.crop_xywh)
        if not mask.any():
            # An entirely invisible object cannot be used as a cutout.
            return sample.image,dict(applied=False,reason='empty_visible_mask')
        source=self.frames[(original_scene,sample.image_id)]
        clean=cv2.imread(str(self.data_root/source['clean']))
        if clean is None:
            raise FileNotFoundError(source['clean'])
        if list(clean.shape[:2])!=record['source_hw']:
            raise ValueError('Mask and source image dimensions differ')
        x,y,w,h=sample.crop_xywh
        foreground=(extract_window(clean,sample.crop_xywh) if sample.content_bbox_xyxy is not None
                    else clean[y:y+h,x:x+w])
        choices = self.backgrounds if exclude_background is None else [
            item for item in self.backgrounds if item['path'] != exclude_background]
        if not choices:
            raise ValueError('Paired backgrounds require at least two different source paths')
        selected=choices[int(rng.integers(len(choices)))]
        background=cv2.imread(str(self.assets/selected['path']))
        if background is None:
            raise FileNotFoundError(selected['path'])
        bh,bw=background.shape[:2]
        side=max(2,int(min(bw,bh)*rng.uniform(.5,1.)))
        bx=int(rng.integers(0,bw-side+1));by=int(rng.integers(0,bh-side+1))
        background=background[by:by+side,bx:bx+side]
        flip=bool(rng.integers(2))
        if flip:background=background[:,::-1]
        background=cv2.resize(background,(w,h),interpolation=cv2.INTER_AREA if w<side else cv2.INTER_LINEAR)
        gain=float(rng.uniform(.8,1.2));background=np.clip(background.astype(np.float32)*gain,0,255)
        blurred=int(sample.scene_id)>=100000
        # Rebuild the blurred variant AFTER compositing to avoid retaining the old background halo.
        image=compose_native(foreground,background,mask,
            source['native_sigma'] if blurred else 0.,source['kernel'] if blurred else 1)
        if sample.content_bbox_xyxy is not None:
            image=fill_outside_box(image,sample.crop_xywh,sample.content_bbox_xyxy)
        size=(sample.image.shape[1],sample.image.shape[0])
        if (w,h)!=size:
            image=cv2.resize(image,size,
                interpolation=cv2.INTER_AREA if size[0]*size[1]<w*h else cv2.INTER_LINEAR)
        return cv2.cvtColor(image,cv2.COLOR_BGR2RGB),dict(applied=True,background=selected['path'],
            background_crop_xywh=[bx,by,side,side],flip=flip,gain=gain,blurred=blurred)

    def load_instance(self,scene_id,image_id,instance_idx):
        sample=super().load_instance(scene_id,image_id,instance_idx)
        rng=np.random.default_rng(np.random.SeedSequence(
            [self.crop_jitter_seed,self.epoch,int(scene_id),int(image_id),int(instance_idx),20260924]))
        if rng.random()>=self.background_probability:
            return sample
        sample.image,_=self.replacement(sample,rng)
        return sample
