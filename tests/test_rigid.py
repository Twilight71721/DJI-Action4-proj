import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from bop_bb8.geometry import corners_from_aabb
from bop_bb8.rigid import fit_rigid_cuboid, project_weak_perspective


class TestRigidCuboid(unittest.TestCase):
    def setUp(self):
        self.corners = corners_from_aabb(np.array([-35., -22., -16.]), np.array([35., 22., 16.]))
        self.rotation = Rotation.from_euler('xyz', [25., -32., 18.], degrees=True).as_matrix()
        self.points = project_weak_perspective(self.corners, self.rotation, 2., np.array([123., 117.]))

    def test_exact_projection_recovery_and_rigidity(self):
        result = fit_rigid_cuboid(self.corners, self.points, starts=3)
        np.testing.assert_allclose(result['points'], self.points, atol=1e-5)
        rotation = np.array(result['rotation'])
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-8)
        self.assertAlmostEqual(np.linalg.det(rotation), 1., places=8)
        restored = project_weak_perspective(self.corners,rotation,result['scale_px_per_model_unit'],result['center'])
        np.testing.assert_allclose(restored,result['points'],atol=1e-7)

    def test_collapsed_points_are_not_reported_as_valid_pose(self):
        result = fit_rigid_cuboid(self.corners,np.full((8,2),100.))
        self.assertEqual(result['status'],'collapsed_observations')
        self.assertIsNone(result['points'])

    def test_rectangle_prior_is_explicitly_marked(self):
        result = fit_rigid_cuboid(self.corners,np.full((8,2),100.),box_xyxy=[30,40,210,200],starts=3)
        self.assertEqual(result['status'],'prior_dominated_collapsed_input')
        self.assertTrue(result['uses_box_prior'])
        self.assertFalse(result['calibrated_pose'])
        self.assertTrue(np.isfinite(result['points']).all())

    def test_zero_weight_disables_rectangle_prior(self):
        result = fit_rigid_cuboid(self.corners,np.full((8,2),100.),box_xyxy=[30,40,210,200],box_weight=0)
        self.assertEqual(result['status'],'collapsed_observations')

    def test_missing_observations_and_invalid_inputs(self):
        points = self.points.copy()
        points[:2] = np.nan
        result = fit_rigid_cuboid(self.corners,points,starts=3)
        np.testing.assert_allclose(result['points'],self.points,atol=1e-4)
        points[:6] = np.nan
        self.assertEqual(fit_rigid_cuboid(self.corners,points)['status'],'insufficient_correspondences')
        with self.assertRaises(ValueError):
            fit_rigid_cuboid(self.corners,self.points,box_xyxy=[1,2,0,4])


if __name__ == '__main__':
    unittest.main()
