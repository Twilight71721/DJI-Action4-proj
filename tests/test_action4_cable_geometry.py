"""Analytic checks for the signed clearance used by the cable collision audit."""

from pathlib import Path
import sys
import unittest

import numpy as np

try:
    import bpy
except ImportError:
    bpy = None

if bpy is not None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from action4_cable_geometry import clearance
    from mathutils.bvhtree import BVHTree


@unittest.skipIf(bpy is None, "requires the bpy rendering environment")
class TestCableClearance(unittest.TestCase):
    def setUp(self):
        points = np.array([[-1,-1,-1], [1,-1,-1], [1,1,-1], [-1,1,-1],
                           [-1,-1,1], [1,-1,1], [1,1,1], [-1,1,1]], dtype=float)*.02
        faces = [(0,3,2,1), (4,5,6,7), (0,1,5,4), (1,2,6,5), (2,3,7,6), (3,0,4,7)]
        self.tree = BVHTree.FromPolygons(points, faces)
        self.bounds = (points.min(axis=0), points.max(axis=0))

    def test_inside_is_negative(self):
        self.assertAlmostEqual(clearance(self.tree, (0,0,0), self.bounds), -.02, places=6)

    def test_exterior_clearance_includes_near_surface(self):
        self.assertAlmostEqual(clearance(self.tree, (.025,0,0), self.bounds), .005, places=6)

    def test_far_point_is_not_misclassified_by_a_face_normal(self):
        self.assertAlmostEqual(clearance(self.tree, (-1,0,0), self.bounds), .98, places=6)


if __name__ == "__main__":
    unittest.main()
