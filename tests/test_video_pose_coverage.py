"""Temporal association must not turn missing detections into real annotations."""
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from build_full_video_pose_coverage import associate, canonical_view


class TestVideoPoseCoverage(unittest.TestCase):
    def test_device_identity_survives_detection_reordering_and_missing_box(self):
        def obj(x, tag):
            return dict(box_xyxy=[x, 100, x + 100, 200], confidence=.9, tag_ids=[tag])

        rows = [dict(objects=[obj(10, 18), obj(210, 19)]),
                dict(objects=[obj(211, 19), obj(11, 18)]),
                dict(objects=[obj(212, 19)]),
                dict(objects=[obj(13, 18), obj(213, 19)])]
        left, right = associate(rows)
        self.assertEqual(left[1]['detection']['tag_ids'], [18])
        self.assertEqual(right[1]['detection']['tag_ids'], [19])
        self.assertIsNone(left[2]['detection'])
        self.assertEqual(left[2]['box_status'], 'interpolated_missing_detection_not_GT')
        self.assertAlmostEqual(left[2]['box_xyxy'][0], 12.)

    def test_view_families_use_valid_rotations_and_expected_visible_side(self):
        for side in [0, 1]:
            for code in ['R', 'B', 'b', 'S', 'f', 'F']:
                for roll in [-30, 0, 35]:
                    R = canonical_view(side, code, roll)
                    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
                    self.assertAlmostEqual(np.linalg.det(R), 1.)
                    direction = -R.T @ np.array([0., 0., 1.])
                    self.assertGreater(direction[0] * (1 if side == 0 else -1), 0)
                    if code in ['R', 'B', 'b']:
                        self.assertLess(direction[2], 0)
                    elif code in ['f', 'F']:
                        self.assertGreater(direction[2], 0)


if __name__ == '__main__':
    unittest.main()
