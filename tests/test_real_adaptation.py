import unittest
from pathlib import Path

import numpy as np

from bop_bb8.real_adaptation import (
    RealCornerDataset, error_statistics, match_boxes, original_to_reference,
)

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / 'dataset/action4-real-corners-full8-20260925'


class TestRealEvaluation(unittest.TestCase):
    def test_detector_miss_is_a_pck_failure_not_zero_error(self):
        m = error_statistics([1., 3., np.inf, np.inf])
        self.assertEqual(m['mean_error_px'], 2.)
        self.assertEqual(m['pck5'], .5)
        self.assertEqual(m['predicted_corners'], 2)

    def test_no_outputs_has_no_conditional_mean(self):
        m = error_statistics(np.full(8, np.inf))
        self.assertIsNone(m['mean_error_px'])
        self.assertEqual(m['pck5'], 0.)

    def test_detections_cannot_match_two_cameras(self):
        result = match_boxes([[0, 0, 100, 100], [5, 5, 105, 105]], [[0, 0, 100, 100]])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], (0, 1.))

    def test_wrong_detection_never_falls_back_to_reference(self):
        self.assertEqual(match_boxes([[0, 0, 100, 100]], [[150, 150, 250, 250]]), {})

    def test_common_coordinates_do_not_depend_on_detection_crop(self):
        original = np.array([[180., 220.], [330., 350.]])
        reference = [100, 150, 300, 300]
        for crop in [[80, 120, 400, 400], [120, 170, 240, 240]]:
            local = (original-np.array(crop[:2]))*256/np.array(crop[2:])
            restored = local*np.array(crop[2:])/256 + np.array(crop[:2])
            np.testing.assert_allclose(original_to_reference(restored, reference),
                                       original_to_reference(original, reference))


@unittest.skipUnless((REAL/'full8_supervision.npz').exists(), 'Local real dataset unavailable')
class TestRealSplit(unittest.TestCase):
    def test_real_train_cannot_contain_heldout_groups(self):
        sets = {s: RealCornerDataset(REAL, s) for s in ('adapt', 'validation', 'test')}
        self.assertEqual([len(x) for x in sets.values()], [5, 3, 1])
        for a, b in [('adapt', 'validation'), ('adapt', 'test'), ('validation', 'test')]:
            self.assertFalse({r['image_sha256'] for r in sets[a].records} &
                             {r['image_sha256'] for r in sets[b].records})
            self.assertFalse({r['time_group'] for r in sets[a].records} &
                             {r['time_group'] for r in sets[b].records})
        self.assertTrue(all(sets['adapt'][i]['valid'].all() for i in range(5)))
        inferred = sum(x != 'visible_edge_reference' for r in sets['adapt'].records for x in r['point_basis'])
        self.assertGreater(inferred, 0)
        self.assertEqual(sum(int(sets['adapt'][i]['valid'].sum()) for i in range(5)), 40)


if __name__ == '__main__':
    unittest.main()
