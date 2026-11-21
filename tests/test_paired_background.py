import unittest
from unittest.mock import patch

import numpy as np
import torch

from bop_bb8.net.loss import spatial_js_loss, spatial_kl_loss
from bop_bb8.online_background import OnlinePoseBackgroundDataset, composite_rgba_crop
from bop_bb8.torch_data import PairedHeatmapDataset
from test_online_background import TestOnlineBackground


class TestSpatialJS(unittest.TestCase):
    def test_identical_and_symmetric(self):
        torch.manual_seed(3)
        a, b = torch.randn(2, 8, 9, 13), torch.randn(2, 8, 9, 13)
        valid = torch.ones(2, 8, dtype=torch.bool)
        self.assertAlmostEqual(spatial_js_loss(a, a, valid).item(), 0., places=6)
        ab = spatial_js_loss(a, b, valid)
        self.assertGreater(ab.item(), 0.)
        self.assertLessEqual(ab.item(), np.log(2))
        torch.testing.assert_close(ab, spatial_js_loss(b, a, valid))

    def test_extreme_logits_have_finite_loss_and_gradients(self):
        a = torch.tensor([[[[1000., -1000.]]]], requires_grad=True)
        b = torch.tensor([[[[-1000., 1000.]]]], requires_grad=True)
        loss = spatial_js_loss(a, b, torch.ones(1, 1, dtype=torch.bool))
        loss.backward()
        self.assertAlmostEqual(loss.item(), np.log(2), places=6)
        self.assertTrue(torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all())

    def test_both_views_receive_gradients_and_invalid_channels_are_excluded(self):
        a = torch.randn(1, 8, 4, 5, requires_grad=True)
        b = torch.randn(1, 8, 4, 5, requires_grad=True)
        valid = torch.tensor([[True, False, False, False, False, False, False, False]])
        loss = spatial_js_loss(a, b, valid)
        loss.backward()
        for grad in [a.grad, b.grad]:
            self.assertGreater(grad[:, 0].abs().sum().item(), 0.)
            self.assertEqual(grad[:, 1:].abs().sum().item(), 0.)
        empty = spatial_js_loss(a, b, torch.zeros_like(valid))
        self.assertEqual(empty.item(), 0.)
        empty.backward()

    def test_agreeing_wrong_predictions_still_receive_supervision(self):
        logits = torch.tensor([[[[10., -10.]]]])
        target = torch.tensor([[[[0., 1.]]]])
        valid = torch.ones(1, 1, dtype=torch.bool)
        self.assertAlmostEqual(spatial_js_loss(logits, logits, valid).item(), 0., places=6)
        self.assertGreater(spatial_kl_loss(logits, target, valid).item(), 10.)


class TestPairData(TestOnlineBackground):
    def test_pair_geometry_reproducibility_blur_and_original_view(self):
        data = OnlinePoseBackgroundDataset(self.data/'trainval', self.model,
            online_background_config=self.config, **self.online_kwargs)
        with patch('bop_bb8.online_background.composite_rgba_crop', wraps=composite_rgba_crop) as compose:
            for i in range(len(data)):
                a, b = data.load_pair(i)
                np.testing.assert_array_equal(a.image, data[i].image)
                np.testing.assert_array_equal(b.image, data.load_pair(i)[1].image)
                self.assertFalse(np.array_equal(a.image, b.image))
                for key in ['keypoints_2d', 'heatmaps', 'valid', 'K_input', 'R_m2c', 't_m2c', 'crop_xywh']:
                    np.testing.assert_array_equal(getattr(a, key), getattr(b, key))
                if a.scene_id in ['000002', '000003']:
                    # Most recent load_pair calls made two composites with identical blur sigma.
                    calls = compose.call_args_list
                    self.assertEqual(calls[-1].args[4], calls[-2].args[4])
        wrapped = PairedHeatmapDataset(data)
        sample = wrapped[0]
        self.assertEqual(sample['image'].shape, sample['image_pair'].shape)
        previous = sample['image_pair'].clone()
        data.set_epoch(2)
        self.assertFalse(torch.equal(previous, wrapped[0]['image_pair']))


if __name__ == '__main__':
    unittest.main()
