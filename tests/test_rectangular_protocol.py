import sys
from pathlib import Path
import unittest

import cv2
import numpy as np
import torch
from torch.utils.data import Subset

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import test_online_background as fixtures
from bop_bb8.dataset import BOPInstanceDataset
from bop_bb8.net import ResNetFPN, decode_local_peak, heatmap_to_input_points, spatial_kl_loss
from bop_bb8.online_background import OnlinePoseBackgroundDataset
from bop_bb8.torch_data import HeatmapDataset, ShapeBucketBatchSampler
from infer_resnet_yolo_real import prepare_crop, to_original

torch.set_num_threads(2)


class TestRectangularData(fixtures.TestOnlineBackground):
    def setUp(self):
        fixtures.TestOnlineBackground.setUp(self)
        self.kwargs_rect = dict(self.online_kwargs, crop_mode='rect', crop_padding=1., crop_border=0,
            input_size=(64, 64), rect_sizes=[(96, 64), (64, 96)], heatmap_margin=16,
            require_all_corners=True)

    def test_exact_pixels_geometry_and_inference_agree_without_rgb_padding(self):
        ds = BOPInstanceDataset(self.data/'trainval', self.model, **self.kwargs_rect)
        sample = ds[0]
        original = cv2.imread(sample.image_path)
        roi, tensor, crop = prepare_crop(original, (76, 60, 124, 100), (64, 64), 1., 'rect', 0,
                                          self.kwargs_rect['rect_sizes'])
        self.assertEqual(crop, (76, 60, 48, 40))
        self.assertEqual(sample.image.shape, (64, 96, 3))
        expected = cv2.cvtColor(cv2.resize(original[60:100, 76:124], (96, 64)), cv2.COLOR_BGR2RGB)
        np.testing.assert_array_equal(sample.image, expected)
        np.testing.assert_array_equal(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB), expected)
        torch.testing.assert_close(tensor, HeatmapDataset(ds)[0]['image'], atol=0, rtol=0)
        self.assertTrue(sample.valid.all())
        self.assertEqual(sample.heatmaps.shape, (8, 48, 56))
        native, _ = cv2.projectPoints(self.model.corners_mm(), np.zeros(3), np.array([0., 0, 100.]),
                                      np.array([[200., 0, 100], [0, 200, 80], [0, 0, 1]]), None)
        np.testing.assert_allclose(to_original(sample.keypoints_2d, crop, (96, 64)),
                                   native.reshape(8, 2), atol=1e-5)

    def test_rgb_crop_does_not_depend_on_ground_truth_pose(self):
        ds = BOPInstanceDataset(self.data/'trainval', self.model, **self.kwargs_rect)
        before = ds[0]
        gt, _, _ = ds._load_scene('000001')
        gt['0'][0]['cam_t_m2c'] = [3, 1, 100]
        after = ds[0]
        np.testing.assert_array_equal(before.image, after.image)
        self.assertEqual(before.crop_xywh, after.crop_xywh)
        self.assertFalse(np.array_equal(before.keypoints_2d, after.keypoints_2d))

    def test_off_roi_corners_are_not_clipped_or_removed(self):
        ds = BOPInstanceDataset(self.data/'trainval', self.model, **self.kwargs_rect)
        _, _, info = ds._load_scene('000001')
        info['0'][0]['bbox_obj'] = [85, 68, 30, 24]
        sample = ds[0]
        self.assertTrue((sample.keypoints_2d < 0).any())
        self.assertTrue(sample.valid.all())
        points = torch.tensor(sample.keypoints_2d/4+16)
        recovered = heatmap_to_input_points(points, sample.heatmaps.shape[-2:], sample.image.shape[:2], 16)
        np.testing.assert_allclose(recovered, sample.keypoints_2d, atol=1e-5)
        ds.heatmap_margin = 0
        with self.assertRaisesRegex(ValueError, 'outside the prediction canvas'):
            ds[0]

    def test_background_pipeline_preserves_rectangle_and_labels(self):
        baseline = BOPInstanceDataset(self.data/'trainval', self.model, **self.kwargs_rect)
        ds = OnlinePoseBackgroundDataset(self.data/'trainval', self.model,
                                          online_background_config=self.config, **self.kwargs_rect)
        for i in range(len(ds)):
            a, b = baseline[i], ds[i]
            self.assertEqual(a.image.shape, b.image.shape)
            self.assertFalse((b.image[0] == 114).all())
            for field in ['keypoints_2d', 'heatmaps', 'valid', 'K_input']:
                np.testing.assert_array_equal(getattr(a, field), getattr(b, field))


class TestRectangularNetwork(unittest.TestCase):
    def test_legacy_evaluator_still_accepts_four_argument_criterion(self):
        from train_heatmap import evaluate
        model = ResNetFPN('resnet18', 16, pretrained=False)
        batch = dict(image=torch.zeros(1, 3, 64, 64), heatmap=torch.zeros(1, 8, 16, 16),
                     valid=torch.ones(1, 8, dtype=torch.bool), keypoints_2d=torch.zeros(1, 8, 2))
        values = evaluate(model, [batch], torch.device('cpu'),
            lambda logits, targets, valid, points: logits.square().mean(), (16, 16), (64, 64), False)
        self.assertTrue(np.isfinite(values).all())

    def test_outside_head_has_gradients_and_batchnorm_stats_stay_fixed(self):
        model = ResNetFPN('resnet18', 16, pretrained=False, heatmap_margin=4, freeze_bn=True)
        model.train()
        mean = model.backbone.bn1.running_mean.clone()
        logits = model(torch.randn(2, 3, 64, 96))
        self.assertEqual(tuple(logits.shape), (2, 8, 24, 32))
        target = torch.zeros_like(logits)
        target[:, :, 1, 2] = 1.  # outside the RGB's core grid
        spatial_kl_loss(logits, target, torch.ones(2, 8, dtype=torch.bool)).backward()
        for parameter in [model.context_proj.weight, model.coordinate_proj.weight,
                          model.backbone.conv1.weight, model.backbone.bn1.weight]:
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0.)
        torch.testing.assert_close(mean, model.backbone.bn1.running_mean, rtol=0, atol=0)

    def test_loss_is_stable_and_cannot_be_reduced_by_suppressing_whole_channel(self):
        target = torch.zeros(1, 2, 16, 24)
        target[0, 0, 4, 9] = 1.
        target[0, 1] = torch.nan
        valid = torch.tensor([[True, False]])
        logits = torch.zeros_like(target, requires_grad=True)
        loss = spatial_kl_loss(logits, target, valid)
        torch.testing.assert_close(loss, spatial_kl_loss(logits-10000, target, valid))
        correct = logits.detach().clone()
        correct[0, 0, 4, 9] = 8.
        self.assertLess(float(spatial_kl_loss(correct, target, valid)), float(loss.detach()))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertLess(float(logits.grad[0, 0, 4, 9]), 0.)
        self.assertEqual(float(logits.grad[0, 1].abs().sum()), 0.)
        with self.assertRaisesRegex(ValueError, 'nonempty'):
            spatial_kl_loss(logits, torch.zeros_like(target), valid)

    def test_local_decoding_does_not_average_separate_modes_or_underflow(self):
        logits = torch.full((1, 1, 16, 24), -10020.)
        logits[0, 0, 4, 3] = -10000.
        logits[0, 0, 12, 20] = -10000.5
        points = decode_local_peak(logits, 2)
        torch.testing.assert_close(points, torch.tensor([[[3., 4.]]]), atol=1e-5, rtol=0)
        torch.testing.assert_close(points, decode_local_peak(logits+10000, 2), atol=1e-5, rtol=0)
        with self.assertRaisesRegex(ValueError, 'non-finite'):
            decode_local_peak(torch.full_like(logits, torch.nan))

    def test_shape_batches_are_reproducible_complete_and_unpadded(self):
        class MetadataOnly:
            def __len__(self):
                return 11
            def get_input_size(self, index):
                return (96, 64) if index % 2 else (64, 96)
            def __getitem__(self, index):
                raise AssertionError('Batch grouping should only need metadata')
        base = MetadataOnly()
        selected = Subset(HeatmapDataset(base), [9, 6, 2, 5, 4, 7, 0])
        sampler = ShapeBucketBatchSampler(selected, 2, True, 42)
        a = list(sampler)
        np.random.seed(99)
        self.assertEqual(a, list(ShapeBucketBatchSampler(selected, 2, True, 42)))
        self.assertEqual(sorted(i for group in a for i in group), list(range(len(selected))))
        for group in a:
            self.assertEqual(len({base.get_input_size(selected.indices[i]) for i in group}), 1)
        sampler.set_epoch(1)
        self.assertNotEqual(a, list(sampler))


if __name__ == '__main__':
    unittest.main()
