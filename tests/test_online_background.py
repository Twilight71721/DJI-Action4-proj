import json
import sys
from pathlib import Path
import unittest
import cv2
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from test_background_augmentation import TestBackgroundDataset
from bop_bb8.dataset import BOPInstanceDataset
from bop_bb8.online_background import OnlinePoseBackgroundDataset,composite_rgba_crop


class TestOnlineBackground(TestBackgroundDataset):
    def setUp(self):
        super().setUp()
        scene=self.data/'clean/000001'
        for sid in ['000002','000003']:(self.data/'trainval'/sid).symlink_to(scene)
        color=cv2.imread(str(scene/'rgb/000000.png'));alpha=np.zeros(color.shape[:2],np.uint8);alpha[60:100,76:124]=255
        rgba=np.dstack([color,alpha]);cv2.imwrite(str(self.data/'rgba.png'),rgba)
        pool=json.loads((self.assets/'manifest.json').read_text())['records']
        self.config=self.data/'online.json';self.config.write_text(json.dumps(dict(
            base_dataset=str(self.data),old_background_assets=str(self.assets),old_train_scenes=['000001','100001'],
            train_backgrounds=[str(self.assets/r['path']) for r in pool],new_records={
                f'{sid}/0/0':dict(rgba=str(self.data/'rgba.png'),pair_id=5,blurred=bool(i),parent_id='pose')
                for i,sid in enumerate(['000002','000003'])})))
        self.online_kwargs=dict(self.kwargs,scenes=['000001','100001','000002','000003'],crop_jitter_shift=0.,crop_jitter_padding=None)

    def test_epoch_randomization_and_unchanged_geometry_all_sources(self):
        base=BOPInstanceDataset(self.data/'trainval',self.model,**self.online_kwargs)
        online=OnlinePoseBackgroundDataset(self.data/'trainval',self.model,online_background_config=self.config,**self.online_kwargs)
        epoch0=[online[i].image.copy() for i in range(len(online))]
        for i in range(len(online)):
            a,b=base[i],online[i]
            for key in ['keypoints_2d','valid','heatmaps','K_input','R_m2c','t_m2c']:
                np.testing.assert_array_equal(getattr(a,key),getattr(b,key))
            np.testing.assert_array_equal(b.image,epoch0[i])
            self.assertFalse(np.array_equal(a.image,b.image))
        online.set_epoch(1)
        for i in range(len(online)):self.assertFalse(np.array_equal(epoch0[i],online[i].image))
        online.set_epoch(0)
        for i in range(len(online)):np.testing.assert_array_equal(epoch0[i],online[i].image)

    def test_foreground_interior_and_background_pixels(self):
        rgba=np.full((64,64,4),0,np.uint8);rgba[16:48,16:48]=[20,40,60,255]
        result=composite_rgba_crop(rgba,np.full((64,64,3),[100,150,200],np.uint8),(0,0,64,64),(64,64))
        np.testing.assert_array_equal(result[20:44,20:44],np.tile([60,40,20],(24,24,1)))
        np.testing.assert_array_equal(result[:10,:10],np.tile([200,150,100],(10,10,1)))

class TestOnlineWorkers(TestOnlineBackground):
    def test_epoch_reaches_fresh_workers(self):
        import torch
        from torch.utils.data import DataLoader
        from bop_bb8.torch_data import HeatmapDataset
        ds=OnlinePoseBackgroundDataset(self.data/'trainval',self.model,online_background_config=self.config,**self.online_kwargs)
        loader=DataLoader(HeatmapDataset(ds),batch_size=4,num_workers=2,shuffle=False)
        a=next(iter(loader))
        ds.set_epoch(1);b=next(iter(loader))
        self.assertFalse(torch.equal(a['image'],b['image']))
        self.assertTrue(torch.equal(a['keypoints_2d'],b['keypoints_2d']))


if __name__=='__main__':unittest.main()
