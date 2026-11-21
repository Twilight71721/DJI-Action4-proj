import unittest

import cv2
import numpy as np

from bop_bb8.geometry import corners_from_aabb, crop_and_scale_intrinsics
from bop_bb8.pnp import fit_perspective_cuboid


class TestPerspectiveCuboid(unittest.TestCase):
    def setUp(self):
        cv2.setRNGSeed(26)
        self.corners = corners_from_aabb(np.array([-35., -22., -16.]), np.array([35., 22., 16.]))
        self.K = np.array([[900., 0, 640.], [0, 920., 360.], [0, 0, 1.]])
        self.rvec = np.array([.35, -.5, .18])
        self.tvec = np.array([10., -8., 330.])
        self.points = cv2.projectPoints(self.corners, self.rvec, self.tvec, self.K, None)[0].reshape(8, 2)

    def test_exact_perspective_recovery(self):
        result = fit_perspective_cuboid(self.corners, self.points, self.K)
        self.assertTrue(result['accepted'])
        np.testing.assert_allclose(result['points'], self.points, atol=1e-5)
        np.testing.assert_allclose(result['tvec'], self.tvec, atol=1e-4)

    def test_ransac_actually_rejects_two_wrong_points(self):
        noisy = self.points.copy()
        noisy[[1, 6]] += [[85, -60], [-95, 70]]
        result = fit_perspective_cuboid(self.corners, noisy, self.K, reprojection_threshold=2.)
        self.assertTrue(result['accepted'])
        self.assertEqual(result['inlier_count'], 6)
        self.assertFalse({1, 6} & set(result['inlier_indices']))
        np.testing.assert_allclose(result['points'], self.points, atol=1e-4)

    def test_tight_anisotropic_crop_with_updated_intrinsics(self):
        crop, size = (530, 260, 220, 170), (320, 192)
        points = (self.points-np.array(crop[:2])) * (np.array(size)/crop[2:])
        K_crop = crop_and_scale_intrinsics(self.K, crop, size)
        result = fit_perspective_cuboid(self.corners, points, K_crop)
        self.assertTrue(result['accepted'])
        restored = np.array(result['points']) * (np.array(crop[2:])/size) + crop[:2]
        np.testing.assert_allclose(restored, self.points, atol=1e-5)
        np.testing.assert_allclose(result['tvec'], self.tvec, atol=1e-4)

    def test_distortion_uses_the_supplied_camera_model(self):
        distortion = np.array([-.3, .1, .002, -.001, 0.])
        points = cv2.projectPoints(self.corners, self.rvec, self.tvec, self.K, distortion)[0].reshape(8, 2)
        result = fit_perspective_cuboid(self.corners, points, self.K, distortion=distortion)
        self.assertTrue(result['accepted'])
        np.testing.assert_allclose(result['tvec'], self.tvec, atol=1e-4)

    def test_collapsed_and_missing_evidence_do_not_produce_a_box(self):
        collapsed = fit_perspective_cuboid(self.corners, np.full((8, 2), 100.), self.K)
        self.assertFalse(collapsed['accepted'])
        self.assertIsNone(collapsed['points'])
        missing = self.points.copy()
        missing[:3] = np.nan
        self.assertFalse(fit_perspective_cuboid(self.corners, missing, self.K)['accepted'])

    def test_invalid_intrinsics_cannot_be_silently_replaced(self):
        for K in [None, np.zeros((3, 3)), self.K * np.nan]:
            with self.assertRaises(ValueError):
                fit_perspective_cuboid(self.corners, self.points, K)


if __name__ == '__main__':
    unittest.main()
