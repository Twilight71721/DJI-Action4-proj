"""Protect foreground occluders and label-independent RGB replacement."""
import sys
from pathlib import Path
import unittest

import numpy as np
import cv2

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from review_pose_replacement_pilot import compose_replacement
from render_pose_replacement_pilot import view_rotation
from prepare_video_pose_candidates import select_marker_candidate
from prepare_video_side_pilot import fit_hidden_top


class TestPoseReplacement(unittest.TestCase):
    def test_occluder_pixels_are_preserved(self):
        rng=np.random.default_rng(42)
        real=rng.integers(0,256,(64,64,3),dtype=np.uint8)
        rgba=np.zeros((64,64,4),np.uint8)
        rgba[16:48,16:48]=[50,80,110,255]
        removal=np.zeros((64,64),np.uint8);removal[16:48,16:48]=255
        occluder=np.zeros_like(removal);occluder[20:30,20:40]=255
        out,_,expanded,_,_=compose_replacement(real,rgba,removal,occluder)
        np.testing.assert_array_equal(out[occluder>0],real[occluder>0])
        self.assertFalse(expanded[occluder>0].any())
        np.testing.assert_array_equal(out[35:40,20:40],rgba[35:40,20:40,:3])
        np.testing.assert_array_equal(out[:8],real[:8])

    def test_exposed_repair_is_detected_and_inputs_unchanged(self):
        real=np.full((64,64,3),80,np.uint8)
        rgba=np.zeros((64,64,4),np.uint8);rgba[24:40,24:40]=[50,80,110,255]
        removal=np.zeros((64,64),np.uint8);removal[16:48,16:48]=255
        inputs=[x.copy() for x in (real,rgba,removal)]
        _,_,_,exposed,ratio=compose_replacement(real,rgba,removal)
        self.assertGreater(ratio,.5)
        self.assertTrue(exposed.any())
        for before,after in zip(inputs,(real,rgba,removal)):
            np.testing.assert_array_equal(before,after)

    def test_view_rotations_are_right_handed(self):
        for d in [[0,.2,-1],[0,.2,1],[1,.2,.1],[-1,.2,-.1],[0,-1,.6]]:
            R=view_rotation(d)
            np.testing.assert_allclose(R@R.T,np.eye(3),atol=1e-12)
            self.assertAlmostEqual(np.linalg.det(R),1.)
            unit=np.array(d)/np.linalg.norm(d)
            np.testing.assert_allclose(R@unit,[0,0,-1],atol=1e-12)

    def test_marker_mirror_with_lower_error_is_rejected_by_known_side(self):
        candidates=[]
        for x,error in [(-1,.2),(1,1.1)]:
            candidates.append(dict(R=view_rotation([x,.1,-.4]).tolist(),t_mm=[0,0,400],rmse_px=error))
        selected=select_marker_candidate(dict(id=18,candidates=candidates),1197)
        self.assertEqual(selected[0],1)
        self.assertIsNone(select_marker_candidate(dict(id=18,candidates=candidates[:1]),1197))

    def test_constrained_pose_recovers_hidden_top_with_off_axis_camera(self):
        xyz=np.array([[35,-21,11.84],[35,21,11.84],[35,21,-15.8],[35,-21,-15.8],[15,3,15.8]],float)
        K=np.array([[1900,0,300],[0,1900,-200],[0,0,1]],float)
        t=np.array([15.,90.,400.]);forward=t/np.linalg.norm(t)
        right=np.cross(forward,[0,-1,0]);right/=np.linalg.norm(right)
        C=np.array([right,np.cross(forward,right),forward])
        R=C.T@view_rotation([1,-.3,.25])
        uv=cv2.projectPoints(xyz,cv2.Rodrigues(R)[0],t,K,None)[0][:,0]
        fitted,translation=fit_hidden_top(xyz,uv,K,R,t)
        projected=cv2.projectPoints(xyz,cv2.Rodrigues(fitted)[0],translation,K,None)[0][:,0]
        np.testing.assert_allclose(projected,uv,atol=.001)
        self.assertLess((-fitted.T@translation)[1],0)
        self.assertAlmostEqual(np.linalg.det(fitted),1.)


if __name__=='__main__':
    unittest.main()
