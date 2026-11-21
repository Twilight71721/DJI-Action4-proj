import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_pose_mix_data import transform_foreground


class TestPoseMixData(unittest.TestCase):
    def test_scale_translation_updates_projection_and_mask_together(self):
        rgba=np.zeros((512,512,4),np.uint8);rgba[180:300,200:320]=[50,100,150,255]
        mask=rgba[:,:,3].copy();K=np.array([[700,0,256],[0,700,256],[0,0,1.]])
        color,alpha,moved,newK,A=transform_foreground(rgba,mask,K,.8,20,-10)
        xyz=np.array([[0,0,300],[15,-10,300],[-20,15,300]],float)
        before=xyz@K.T;before=before[:,:2]/before[:,2:]
        after=xyz@newK.T;after=after[:,:2]/after[:,2:]
        np.testing.assert_allclose(after,(np.c_[before,np.ones(3)]@A.T)[:,:2])
        center=A@np.array([260,240,1]);x,y=np.rint(center[:2]).astype(int)
        self.assertEqual(moved[y,x],255)
        np.testing.assert_allclose(color[y,x],[50,100,150]);self.assertEqual(alpha[y,x,0],1)
        self.assertFalse(moved[0].any())

    def test_transparent_rgb_does_not_leave_dark_or_colored_fringe(self):
        rgba=np.full((512,512,4),255,np.uint8);rgba[:,:,3]=0
        color,alpha,mask,_,_=transform_foreground(rgba,rgba[:,:,3],np.eye(3),.83,3.2,-8.1)
        self.assertFalse(color.any());self.assertFalse(alpha.any());self.assertFalse(mask.any())


if __name__=='__main__':unittest.main()
