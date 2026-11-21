import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from bop_bb8 import BOPInstanceDataset, BOPModel
from bop_bb8.torch_data import HeatmapDataset


class TestCropJitter(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        scene = self.root / '000001'
        (scene / 'rgb').mkdir(parents=True)
        yy, xx = np.mgrid[:240, :320]
        image = np.stack([xx % 256, yy, (xx + yy) % 256], axis=-1).astype('uint8')
        cv2.imwrite(str(scene / 'rgb/000000.png'), image)
        self.image = image
        self.K = np.array([[250., 0, 160], [0, 250., 120], [0, 0, 1]])
        self.model = BOPModel(1, np.array([-20., -15., -10.]), np.array([20., 15., 10.]))
        for name, data in {
            'scene_gt': {'0': [dict(obj_id=1, cam_R_m2c=np.eye(3).ravel().tolist(), cam_t_m2c=[0, 0, 120])]},
            'scene_camera': {'0': dict(cam_K=self.K.ravel().tolist(), width=320, height=240)},
            'scene_gt_info': {'0': [dict(bbox_obj=[114, 86, 92, 68], visib_fract=1)]},
        }.items():
            (scene / (name + '.json')).write_text(json.dumps(data))

    def data(self, **kwargs):
        return BOPInstanceDataset(self.root, self.model, scenes=['000001'], crop_mode='bbox',
                                  input_size=(256, 192), heatmap_size=(48, 64),
                                  heatmap_kernel='gaussian', **kwargs)

    def test_disabled_is_exact_and_validation_does_not_change(self):
        base = self.data()
        off = self.data(crop_jitter_shift=0, crop_jitter_padding=None, crop_jitter_seed=99)
        a = base[0]
        off.set_epoch(9)
        b = off[0]
        for key in ('image', 'keypoints_2d', 'valid', 'heatmaps', 'K_input'):
            np.testing.assert_array_equal(getattr(a, key), getattr(b, key))

    def test_image_geometry_heatmap_and_protected_corners(self):
        base = self.data()[0]
        data = self.data(crop_jitter_shift=.05, crop_jitter_padding=(1.2, 1.4), crop_jitter_seed=7)
        crops = set()
        # Independent full-frame projection, not the dataset crop projection code.
        xyz = self.model.corners_mm() + [0, 0, 120]
        homogeneous = xyz @ self.K.T
        full = homogeneous[:, :2] / homogeneous[:, 2:]
        for epoch in range(20):
            data.set_epoch(epoch)
            s = data[0]
            crops.add(s.crop_xywh)
            x, y, w, h = s.crop_xywh
            expected = (full - [x, y]) * [256 / w, 192 / h]
            np.testing.assert_allclose(s.keypoints_2d, expected, atol=2e-5)
            self.assertTrue(s.valid[base.valid].all())
            image = self.image[y:y+h, x:x+w]
            interpolation = cv2.INTER_AREA if 256 * 192 < w * h else cv2.INTER_LINEAR
            expected_image = cv2.cvtColor(cv2.resize(image, (256, 192), interpolation=interpolation), cv2.COLOR_BGR2RGB)
            np.testing.assert_array_equal(s.image, expected_image)
            for i in np.flatnonzero(s.valid):
                row, col = np.unravel_index(s.heatmaps[i].argmax(), s.heatmaps[i].shape)
                self.assertLessEqual(np.linalg.norm(np.array([col, row]) - expected[i] / 4), .708)
        self.assertGreater(len(crops), 10)

    def test_local_rng_repeatability_and_worker_independence(self):
        data = self.data(crop_jitter_shift=.05, crop_jitter_padding=(1.2, 1.4), crop_jitter_seed=42)
        np.random.seed(10)
        expected = np.random.rand(3)
        np.random.seed(10)
        first = data[0]
        np.testing.assert_array_equal(expected, np.random.rand(3))
        np.testing.assert_array_equal(first.image, data[0].image)
        data.set_epoch(3)
        direct = next(iter(DataLoader(HeatmapDataset(data), num_workers=0)))
        worker = next(iter(DataLoader(HeatmapDataset(data), num_workers=2)))
        for key in ('image', 'heatmap', 'keypoints_2d', 'valid', 'K_input'):
            self.assertTrue(torch.equal(direct[key], worker[key]))

    def test_boundary_and_fallback(self):
        data = self.data(crop_jitter_shift=.5, crop_jitter_padding=(1., 1.), crop_jitter_seed=2)
        base = self.data()[0]
        for epoch in range(10):
            data.set_epoch(epoch)
            sample = data[0]
            self.assertTrue(sample.valid[base.valid].all())
            x, y, w, h = sample.crop_xywh
            self.assertGreaterEqual(min(x, y), 0)
            self.assertLessEqual(x + w, 320)
            self.assertLessEqual(y + h, 240)

    def test_invalid_parameters(self):
        for kwargs in [dict(crop_jitter_shift=-1), dict(crop_jitter_shift=float('nan')),
                       dict(crop_jitter_padding=(1.4, 1.2)), dict(crop_jitter_seed=-1)]:
            with self.assertRaises(ValueError):
                self.data(**kwargs)


if __name__ == '__main__':
    unittest.main()
