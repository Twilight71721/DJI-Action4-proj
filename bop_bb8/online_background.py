"""Epoch-varying backgrounds for both legacy cameras and new RGBA pose assets."""
import json
import copy
from pathlib import Path
import cv2
import numpy as np
from .background import RandomBackgroundBOPDataset
from .dataset import BOPInstanceDataset
from .cropping import extract_window, fill_outside_box


def composite_rgba_crop(rgba, background, crop, input_size, sigma256=0., content_box=None):
    x,y,w,h=map(int,crop)
    foreground=(extract_window(rgba,crop,fill=0) if content_box is not None else rgba[y:y+h,x:x+w])
    if foreground.shape[:2]!=(h,w) or foreground.shape[2]!=4:
        raise ValueError('RGBA/crop mismatch')
    alpha=foreground[:,:,3:].astype(np.float32)/255
    bg=cv2.resize(background,(w,h),interpolation=cv2.INTER_AREA)
    image=np.rint(foreground[:,:,:3]*alpha+bg*(1-alpha)).clip(0,255).astype(np.uint8)
    if sigma256:
        sigma=sigma256*w/input_size[0];kernel=2*int(np.ceil(3*sigma))+1
        image=cv2.GaussianBlur(image,(kernel,kernel),sigma)
    if content_box is not None:
        image=fill_outside_box(image,crop,content_box)
    return cv2.cvtColor(cv2.resize(image,input_size,interpolation=cv2.INTER_AREA if w>input_size[0] else cv2.INTER_LINEAR),cv2.COLOR_BGR2RGB)


def sample_background(files,rng):
    index=int(rng.integers(len(files)));image=cv2.imread(str(files[index]))
    if image is None:raise FileNotFoundError(files[index])
    h,w=image.shape[:2];side=max(2,int(min(h,w)*rng.uniform(.5,1.)))
    x=int(rng.integers(w-side+1));y=int(rng.integers(h-side+1));patch=image[y:y+side,x:x+side]
    if rng.random()<.5:patch=patch[:,::-1]
    return patch,index


class OnlinePoseBackgroundDataset(BOPInstanceDataset):
    def __init__(self,*args,online_background_config,**kwargs):
        super().__init__(*args,**kwargs)
        config=json.loads(Path(online_background_config).read_text())
        self.online_config=config
        self.new_records=config['new_records'];self.background_files=[Path(p) for p in config['train_backgrounds']]
        if not self.background_files:raise ValueError('Empty background pool')
        self.legacy=RandomBackgroundBOPDataset(Path(config['base_dataset'])/'trainval',self.model,
            scenes=config['old_train_scenes'],background_assets=Path(config['old_background_assets']),
            input_size=self.input_size,heatmap_size=self.heatmap_size,crop_mode=self.crop_mode,
            crop_padding=self.crop_padding,crop_border=self.crop_border,
            rect_sizes=self.rect_sizes,heatmap_margin=self.heatmap_margin)
        self.legacy.backgrounds=[dict(path=str(p)) for p in self.background_files]
        for scene,frame,instance in self.instances:
            if scene not in config['old_train_scenes'] and f'{scene}/{frame}/{instance}' not in self.new_records:
                raise ValueError('Missing online composition metadata')

    def load_instance(self,scene_id,image_id,instance_idx):
        sample=super().load_instance(scene_id,image_id,instance_idx)
        key=f'{scene_id}/{image_id}/{instance_idx}'
        if key in self.new_records:
            record=self.new_records[key]
            rng=np.random.default_rng(np.random.SeedSequence([self.crop_jitter_seed,self.epoch,record['pair_id'],71]))
            rgba=cv2.imread(record['rgba'],-1)
            if rgba is None:raise FileNotFoundError(record['rgba'])
            bg,_=sample_background(self.background_files,rng)
            sigma=float(rng.uniform(.3,1.4)) if record['blurred'] else 0.
            size=(sample.image.shape[1],sample.image.shape[0])
            sample.image=composite_rgba_crop(rgba,bg,sample.crop_xywh,size,sigma,
                                            content_box=sample.content_bbox_xyxy)
        else:
            # Clean/blur partners use the same sampled background; only RGB blur differs.
            rng=np.random.default_rng(np.random.SeedSequence([self.crop_jitter_seed,self.epoch,int(scene_id)%100000,int(image_id),instance_idx,70]))
            sample.image,meta=self.legacy.replacement(sample,rng)
            if not meta['applied']:raise ValueError('Cannot randomize an empty object mask')
        return sample

    def load_pair(self, index):
        """Same geometry and blur, independent backgrounds; retain the original first view."""
        first = self[index]
        second = copy.deepcopy(first)
        scene, frame, instance = self.instances[index]
        key = f'{scene}/{frame}/{instance}'
        if key in self.new_records:
            record = self.new_records[key]
            original_rng = np.random.default_rng(np.random.SeedSequence(
                [self.crop_jitter_seed, self.epoch, record['pair_id'], 71]))
            _, first_index = sample_background(self.background_files, original_rng)
            sigma = float(original_rng.uniform(.3, 1.4)) if record['blurred'] else 0.
            choices = [p for p in self.background_files if p != self.background_files[first_index]]
            if not choices:
                raise ValueError('Paired backgrounds require different source paths')
            rng = np.random.default_rng(np.random.SeedSequence(
                [self.crop_jitter_seed, self.epoch, record['pair_id'], 71, 1]))
            bg, _ = sample_background(choices, rng)
            rgba = cv2.imread(record['rgba'], -1)
            if rgba is None:
                raise FileNotFoundError(record['rgba'])
            second.image = composite_rgba_crop(rgba, bg, first.crop_xywh,
                (first.image.shape[1], first.image.shape[0]), sigma, content_box=first.content_bbox_xyxy)
        else:
            seed = [self.crop_jitter_seed, self.epoch, int(scene)%100000, int(frame), instance, 70]
            _, meta = self.legacy.replacement(first, np.random.default_rng(np.random.SeedSequence(seed)))
            if not meta['applied']:
                raise ValueError('Cannot pair an empty mask')
            second.image, _ = self.legacy.replacement(first,
                np.random.default_rng(np.random.SeedSequence(seed+[1])), exclude_background=meta['background'])
        return first, second
