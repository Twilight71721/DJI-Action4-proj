import unittest
import numpy as np
import torch

from bop_bb8.photometric import augment_rgb,PhotometricDataset
from bop_bb8.torch_data import ShapeBucketBatchSampler,IMAGENET_MEAN,IMAGENET_STD
from bop_bb8.view_sampling import view_bin,view_weights


class TestPhotometric(unittest.TestCase):
    def test_balanced_coverage_and_severity(self):
        image=np.random.default_rng(0).integers(0,256,(17,29,3),dtype=np.uint8)
        counts={'unchanged':0,'gentle':0,'moderate':0}
        for seed in range(1000):
            result,meta=augment_rgb(image,np.random.default_rng(seed),'balanced')
            counts['unchanged' if meta['unchanged'] else meta['strength']]+=1
            if meta['unchanged']:
                np.testing.assert_array_equal(result,image)
            else:
                self.assertIn('color',meta)
                if meta['strength']=='gentle':
                    self.assertTrue(.85<=meta['color']['gain']<=1.15)
                    self.assertLessEqual(meta.get('motion_blur',{}).get('kernel',3),3)
                    self.assertGreaterEqual(meta.get('jpeg_quality',95),85)
                    self.assertLessEqual(meta.get('noise_sigma_255',0),1.5)
            self.assertEqual(result.shape,image.shape)
        self.assertTrue(150<counts['unchanged']<250,counts)
        self.assertTrue(620<counts['gentle']<740,counts)
        self.assertTrue(80<counts['moderate']<160,counts)
        with self.assertRaises(ValueError):
            augment_rgb(image,np.random.default_rng(0),'unknown')

    def test_shape_range_reproducibility_and_source_preserved(self):
        image=np.random.default_rng(0).integers(0,256,(47,91,3),dtype=np.uint8)
        original=image.copy();changed=0
        for seed in range(25):
            a,meta=augment_rgb(image,np.random.default_rng(seed))
            b,other=augment_rgb(image,np.random.default_rng(seed))
            np.testing.assert_array_equal(a,b);self.assertEqual(meta,other)
            self.assertEqual(a.shape,image.shape);self.assertEqual(a.dtype,np.uint8)
            changed+=not np.array_equal(a,image)
        np.testing.assert_array_equal(image,original);self.assertGreater(changed,15)

    def test_labels_unchanged_and_paired_photometry_shared(self):
        torch.manual_seed(2)
        mean=torch.tensor(IMAGENET_MEAN)[:,None,None];std=torch.tensor(IMAGENET_STD)[:,None,None]
        image=(torch.randint(0,256,(3,32,64)).float()/255-mean)/std
        row=dict(image=image,image_pair=image.clone(),index=7,heatmap=torch.randn(8,8,16),
                 valid=torch.ones(8,dtype=torch.bool),keypoints_2d=torch.randn(8,2))
        source=image.clone();data=PhotometricDataset([row],seed=4,profile='balanced')
        sample=data[0]
        torch.testing.assert_close(sample['image'],sample['image_pair'],rtol=0,atol=0)
        for name in ['heatmap','valid','keypoints_2d']:self.assertIs(sample[name],row[name])
        torch.testing.assert_close(row['image'],source,rtol=0,atol=0)
        seen=[sample['image']]
        for epoch in range(1,5):data.set_epoch(epoch);seen.append(data[0]['image'])
        self.assertTrue(any(not torch.equal(seen[0],x) for x in seen[1:]))


class Shapes:
    def __len__(self):return 120
    def get_input_size(self,index):return (96,64) if index<60 else (64,96)


class TestViews(unittest.TestCase):
    def test_directions_include_translation_and_center(self):
        for axis in range(3):
            for sign in [1,-1]:
                d=np.zeros(3);d[axis]=sign*100
                self.assertEqual(view_bin(np.eye(3),-d,np.zeros(3)),axis*2+int(sign<0))
        self.assertEqual(view_bin(np.eye(3),[0,0,1],[100,0,0]),1)

    def test_bounded_weights_and_reproducible_same_budget_sampler(self):
        labels=np.array([0]*50+[1]*10+[0]*50+[1]*10)
        weights,counts=view_weights(labels,cap=2.)
        self.assertEqual(weights.max(),2.);self.assertEqual(counts[:2].tolist(),[100,20])
        sampler=ShapeBucketBatchSampler(Shapes(),8,True,5,weights)
        original=ShapeBucketBatchSampler(Shapes(),8,True,5)
        a=list(sampler);self.assertEqual(a,list(sampler));self.assertEqual(len(a),len(original))
        self.assertEqual(sum(map(len,a)),120)
        rare=[]
        for epoch in range(20):
            sampler.set_epoch(epoch);batches=list(sampler)
            self.assertTrue(all(len({Shapes().get_input_size(i) for i in b})==1 for b in batches))
            rare.append(sum(labels[i]==1 for b in batches for i in b))
        self.assertGreater(np.mean(rare),25)


if __name__=='__main__':unittest.main()
