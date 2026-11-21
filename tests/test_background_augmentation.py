import json
from pathlib import Path
import tempfile
import unittest
import sys

import cv2
import numpy as np

from bop_bb8.background import RandomBackgroundBOPDataset, compose_native, mask_in_crop
from bop_bb8.dataset import BOPInstanceDataset
from bop_bb8.geometry import BOPModel
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_camera_masks import repair_visible_mask


class TestBackgroundMath(unittest.TestCase):
    def test_mask_crop_offset_and_padding(self):
        roi=np.arange(24,dtype=np.uint8).reshape(4,6)
        full=np.zeros((15,20),np.uint8);full[5:9,8:14]=roi
        for crop in [(6,4,10,8),(10,7,6,6),(0,0,3,3)]:
            x,y,w,h=crop
            np.testing.assert_array_equal(mask_in_crop(roi,(8,5,6,4),crop),full[y:y+h,x:x+w])

    def test_composite_preserves_interior_and_changes_background(self):
        foreground=np.full((64,64,3),[40,90,180],np.uint8)
        background=np.full_like(foreground,[180,30,20])
        mask=np.zeros((64,64),np.uint8);mask[16:48,16:48]=255
        result=compose_native(foreground,background,mask)
        np.testing.assert_array_equal(result[24:40,24:40],foreground[24:40,24:40])
        np.testing.assert_array_equal(result[:8,:8],background[:8,:8])
        expected=cv2.GaussianBlur(result,(7,7),1.2)
        np.testing.assert_array_equal(compose_native(foreground,background,mask,1.2,7),expected)

    def test_full_silhouette_keeps_occluded_pixels_in_foreground(self):
        full=np.zeros((40,40),np.uint8);full[4:36,4:36]=255
        visible=full.copy();visible[12:15,12:15]=0;visible[4:23,23:30]=0
        repaired=repair_visible_mask(visible,full)
        self.assertTrue((repaired[12:15,12:15]==255).all())
        self.assertTrue((repaired[5:20,24:29]==255).all())
        self.assertTrue((repaired[full==0]==0).all())
        self.assertTrue((repaired[visible>0]==255).all())
        np.testing.assert_array_equal(repaired,full)


class TestBackgroundDataset(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        root=Path(tmp.name);self.data=root/'data';self.assets=root/'assets'
        scene=self.data/'clean/000001';(scene/'rgb').mkdir(parents=True)
        (self.data/'trainval').mkdir();(self.data/'trainval/000001').symlink_to(scene)
        (self.data/'trainval/100001').symlink_to(scene)
        self.model=BOPModel(1,np.array([-10.,-8.,-5.]),np.array([10.,8.,5.]))
        image=np.full((160,200,3),[140,160,170],np.uint8);image[60:100,76:124]=[20,60,120]
        cv2.imwrite(str(scene/'rgb/000000.png'),image)
        for name,value in dict(scene_gt={'0':[dict(obj_id=1,cam_R_m2c=np.eye(3).ravel().tolist(),cam_t_m2c=[0,0,100])]},
            scene_gt_info={'0':[dict(bbox_obj=[76,60,48,40])]},
            scene_camera={'0':dict(cam_K=[200,0,100,0,200,80,0,0,1],width=200,height=160)}).items():
            (scene/(name+'.json')).write_text(json.dumps(value))
        (self.data/'manifest.json').write_text(json.dumps(dict(frames=[dict(scene='000001',frame=0,
             clean='clean/000001/rgb/000000.png',native_sigma=1.,kernel=7)])))
        (self.assets/'camera_masks').mkdir(parents=True);(self.assets/'images').mkdir()
        cv2.imwrite(str(self.assets/'camera_masks/mask.png'),np.full((40,48),255,np.uint8))
        (self.assets/'camera_masks/manifest.json').write_text(json.dumps(dict(records={'000001/000000/0':
             dict(path='mask.png',roi_xywh=[76,60,48,40],source_hw=[160,200])})))
        records=[]
        for i in range(4):
            rng=np.random.default_rng(i);image=rng.integers(0,256,(90,100,3),dtype=np.uint8)
            cv2.imwrite(str(self.assets/f'images/{i}.png'),image);records.append(dict(path=f'images/{i}.png'))
        (self.assets/'manifest.json').write_text(json.dumps(dict(records=records)))
        self.kwargs=dict(scenes=['000001','100001'],input_size=(64,64),heatmap_size=(16,16),crop_mode='bbox',
             heatmap_kernel='gaussian',crop_jitter_shift=.05,crop_jitter_padding=(1.2,1.4),crop_jitter_seed=14)

    def test_labels_seed_and_disabled_behavior(self):
        baseline=BOPInstanceDataset(self.data/'trainval',self.model,**self.kwargs)
        augmented=RandomBackgroundBOPDataset(self.data/'trainval',self.model,**self.kwargs,
             background_assets=self.assets,background_probability=1.)
        disabled=RandomBackgroundBOPDataset(self.data/'trainval',self.model,**self.kwargs,
             background_assets=self.assets,background_probability=0.)
        previous=None
        for epoch in range(3):
            for ds in [baseline,augmented,disabled]:ds.set_epoch(epoch)
            for i in [0,1]:
                original,new,off=baseline[i],augmented[i],disabled[i]
                for field in ['keypoints_2d','valid','heatmaps','K_input','R_m2c','t_m2c']:
                    np.testing.assert_array_equal(getattr(original,field),getattr(new,field))
                self.assertEqual(original.crop_xywh,new.crop_xywh)
                np.testing.assert_array_equal(off.image,original.image)
                np.testing.assert_array_equal(new.image,augmented[i].image)
                self.assertGreater(np.mean(np.abs(original.image.astype(float)-new.image)),1)
            if previous is not None:self.assertFalse(np.array_equal(previous,new.image))
            previous=new.image


if __name__=='__main__':unittest.main()
