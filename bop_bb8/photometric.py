"""Moderate RGB-only training augmentation; never transform geometry or labels."""
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .torch_data import IMAGENET_MEAN, IMAGENET_STD


def augment_rgb(rgb, rng, profile='pilot'):
    if profile not in ['pilot', 'balanced']:
        raise ValueError(f'Unknown photometric profile: {profile}')
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('Expected uint8 RGB')
    if rng.random() < .2:
        return rgb.copy(), {'unchanged': True}
    x = rgb.astype(np.float32)/255
    meta = {'unchanged': False}
    gentle = profile == 'balanced' and rng.random() < .85
    if profile == 'balanced':
        meta['strength'] = 'gentle' if gentle else 'moderate'
    if profile == 'balanced' or rng.random() < .8:
        gain, contrast, gamma, saturation = [float(rng.uniform(*bounds)) for bounds in
            ([(.85,1.15),(.9,1.1),(.9,1.1),(.85,1.15)] if gentle else
             [(.7,1.3),(.75,1.25),(.8,1.25),(.7,1.25)])]
        white_balance = rng.uniform(.96 if gentle else .92,1.04 if gentle else 1.08,3).astype(np.float32)
        x = np.clip((x-.5)*contrast+.5,0,1)
        gray = x @ np.array([.299,.587,.114],np.float32)
        x = gray[...,None] + saturation*(x-gray[...,None])
        x = np.clip(x*gain*white_balance,0,1)**gamma
        meta['color'] = dict(gain=gain,contrast=contrast,gamma=gamma,saturation=saturation,
                             white_balance=white_balance.tolist())
    if rng.random() < (.1 if profile == 'balanced' else .15):
        size = int(rng.choice([3] if gentle else [3,5])); kernel = np.zeros((size,size),np.float32)
        angle = float(rng.uniform(0,np.pi));radius=(size-1)/2;center=np.array([radius,radius])
        delta=radius*np.array([np.cos(angle),np.sin(angle)])
        cv2.line(kernel,tuple(np.rint(center-delta).astype(int)),tuple(np.rint(center+delta).astype(int)),1.,1)
        kernel /= kernel.sum()
        x=cv2.filter2D(x,-1,kernel,borderType=cv2.BORDER_REFLECT_101)
        meta['motion_blur']=dict(kernel=size,angle_radians=angle)
    if rng.random() < (.25 if profile == 'balanced' else .3):
        sigma=float(rng.uniform(.5,1.5 if gentle else 3.))
        x=x+rng.normal(0,sigma/255,x.shape).astype(np.float32)
        meta['noise_sigma_255']=sigma
    result=np.rint(np.clip(x,0,1)*255).astype(np.uint8)
    if rng.random() < (.2 if profile == 'balanced' else .3):
        quality=int(rng.integers(85 if gentle else 65,96))
        ok,encoded=cv2.imencode('.jpg',cv2.cvtColor(result,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,quality])
        if not ok:raise RuntimeError('JPEG augmentation encoding failed')
        result=cv2.cvtColor(cv2.imdecode(encoded,cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
        meta['jpeg_quality']=quality
    return result,meta


class PhotometricDataset(Dataset):
    """Apply after ImageNet normalization is inverted; preserve the sample's other fields."""
    def __init__(self,dataset,seed=0,profile='pilot'):
        if profile not in ['pilot', 'balanced']:
            raise ValueError(f'Unknown photometric profile: {profile}')
        self.dataset=dataset;self.seed=seed;self.epoch=0;self.profile=profile

    def __len__(self):return len(self.dataset)

    def set_epoch(self,epoch):self.epoch=int(epoch)

    def __getitem__(self,index):
        source=self.dataset[index]
        result=dict(source)
        mean=torch.tensor(IMAGENET_MEAN)[:,None,None];std=torch.tensor(IMAGENET_STD)[:,None,None]
        for name in ['image','image_pair']:
            if name not in source:continue
            rgb=(source[name]*std+mean).permute(1,2,0).mul(255).round().clamp(0,255).byte().numpy()
            # Paired backgrounds share all photometric draws, including the noise realization.
            rng=np.random.default_rng(np.random.SeedSequence([self.seed,self.epoch,int(source['index']),930]))
            image,_=augment_rgb(rgb,rng,self.profile)
            value=torch.from_numpy(np.ascontiguousarray(image)).permute(2,0,1).float()/255
            result[name]=(value-mean)/std
        return result
