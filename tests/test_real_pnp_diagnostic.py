import sys
from pathlib import Path
import unittest

import cv2
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from diagnose_real_pnp import all_point_fit
from bop_bb8.geometry import corners_from_aabb


class TestDiagnosticFit(unittest.TestCase):
    def test_exact_case_reprojects_but_is_still_only_a_diagnostic(self):
        corners=corners_from_aabb([-35,-22,-16],[35,22,16])
        K=np.array([[900.,0,640],[0,900,360],[0,0,1]])
        points=cv2.projectPoints(corners,np.array([.3,-.5,.2]),np.array([10.,-8.,330.]),K,None)[0].reshape(8,2)
        saved=points.copy();fit=all_point_fit(corners,points,K)
        self.assertTrue(fit['diagnostic_only'])
        self.assertFalse(fit['accepted'])
        np.testing.assert_allclose(fit['points'],points,atol=1e-4)
        np.testing.assert_array_equal(points,saved)

    def test_fully_collapsed_points_do_not_invent_a_solution(self):
        fit=all_point_fit(corners_from_aabb([-1,-2,-3],[1,2,3]),np.zeros((8,2)),np.eye(3))
        self.assertIsNone(fit['points'])


if __name__=='__main__':unittest.main()
