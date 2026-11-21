import unittest
import numpy as np
from scipy.spatial.transform import Rotation

from bop_bb8.geometry import corners_from_aabb
from bop_bb8.rigid import project_weak_perspective
from bop_bb8.joint_decode import peak_candidates,decode_joint_cuboid


class TestJointDecode(unittest.TestCase):
    def setUp(self):
        self.corners=corners_from_aabb(np.array([-35.,-22.,-16.]),np.array([35.,22.,16.]))
        rotation=Rotation.from_euler('xyz',[25,-32,18],degrees=True).as_matrix()
        self.truth=project_weak_perspective(self.corners,rotation,2.,np.array([128.,128.]))
        self.yy,self.xx=np.mgrid[:64,:64]
        self.heat=np.array([np.exp(-((self.xx-p[0]/4)**2+(self.yy-p[1]/4)**2)/2) for p in self.truth])

    def test_secondary_peak_recovers_from_inconsistent_stronger_peak(self):
        self.heat[0]*=.65
        self.heat[0]+=np.exp(-((self.xx-50)**2+(self.yy-50)**2)/2)
        result=decode_joint_cuboid(self.heat,self.corners,box_xyxy=[25,25,230,230])
        self.assertTrue(result['accepted'])
        self.assertGreater(result['selected_indices'][0],0)
        self.assertLess(np.linalg.norm(np.array(result['selected_points'])-self.truth,axis=1).max(),2.)

    def test_flat_and_collapsed_maps_do_not_become_supported_pose(self):
        result=decode_joint_cuboid(np.ones((8,64,64)),self.corners)
        self.assertFalse(result['accepted'])
        maps=np.repeat(self.heat[:1],8,axis=0)
        result=decode_joint_cuboid(maps,self.corners,box_xyxy=[25,25,230,230])
        self.assertFalse(result['accepted'])

    def test_rectangular_coordinate_scaling_and_invalid_values(self):
        maps=np.zeros((8,20,30));maps[:,10,15]=1
        candidates=peak_candidates(maps,input_size=(100,240))
        np.testing.assert_allclose(candidates[0]['points'][0],[120,50])
        maps[0,0,0]=np.nan
        with self.assertRaises(ValueError):
            peak_candidates(maps)


if __name__=='__main__':
    unittest.main()
