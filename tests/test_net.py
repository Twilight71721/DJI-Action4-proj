import unittest
from pathlib import Path

import torch

from bop_bb8 import BOPInstanceDataset, BOPModel
from bop_bb8.net import (
    ResNetFPN,
    coordinate_loss,
    decode_argmax,
    decode_soft_argmax,
    decode_topk,
    focal_loss,
    heatmap_loss,
    heatmap_to_input_points,
)
from bop_bb8.torch_data import HeatmapDataset

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_DIR / "dataset" / "train_pbr"
MODELS_INFO = PROJECT_DIR / "dataset" / "dji-action4" / "models" / "models_info.json"
V4_DIR = PROJECT_DIR / "dataset" / "action4-v4-random-validation-100"


def make_peak_heatmaps() -> torch.Tensor:
    """Two channels with Gaussian peaks at (10.4, 6.2) and (25.0, 20.0)."""
    heatmaps = torch.zeros(1, 2, 32, 32)
    for channel, (x, y) in enumerate(((10.4, 6.2), (25.0, 20.0))):
        xs = torch.arange(32)[None, :]
        ys = torch.arange(32)[:, None]
        heatmaps[0, channel] = torch.exp(-((xs - x) ** 2 + (ys - y) ** 2) / (2 * 2.0**2))
    return heatmaps


class TestResNetFPN(unittest.TestCase):
    def test_output_shape_as_logits(self):
        model = ResNetFPN(pretrained=False)
        model.eval()
        with torch.no_grad():
            out = model(torch.rand(2, 3, 64, 64))
        self.assertEqual(tuple(out.shape), (2, 8, 16, 16))
        self.assertTrue(torch.isfinite(out).all())
        probs = torch.sigmoid(out)
        self.assertGreaterEqual(float(probs.min()), 0.0)
        self.assertLessEqual(float(probs.max()), 1.0)

    def test_freeze_backbone_keeps_bn_in_eval(self):
        model = ResNetFPN(pretrained=False)
        model.freeze_backbone(True)
        model.train()
        self.assertFalse(model.backbone.training)
        self.assertFalse(any(p.requires_grad for p in model.backbone.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.head.parameters()))
        model.freeze_backbone(False)
        model.train()
        self.assertTrue(model.backbone.training)
        self.assertTrue(all(p.requires_grad for p in model.backbone.parameters()))


class TestDecode(unittest.TestCase):
    def test_argmax_hits_nearest_pixel(self):
        points = decode_argmax(make_peak_heatmaps())
        self.assertEqual(tuple(points[0, 0].tolist()), (10.0, 6.0))
        self.assertEqual(tuple(points[0, 1].tolist()), (25.0, 20.0))

    def test_topk_weighted_is_subpixel(self):
        points = decode_topk(make_peak_heatmaps(), k=9)
        torch.testing.assert_close(points[0, 0], torch.tensor([10.4, 6.2]), atol=0.5, rtol=0)

    def test_topk_shapes(self):
        points = decode_topk(make_peak_heatmaps(), k=4, weighted=False)
        self.assertEqual(tuple(points.shape), (1, 2, 2))

    def test_soft_argmax_near_peak(self):
        points = decode_soft_argmax(make_peak_heatmaps(), temperature=50.0)
        torch.testing.assert_close(points[0, 0], torch.tensor([10.4, 6.2]), atol=0.5, rtol=0)

    def test_to_input_points_scaling(self):
        points = torch.tensor([[[80.0, 60.0]]])
        out = heatmap_to_input_points(points, (120, 160), (480, 640))
        torch.testing.assert_close(out, torch.tensor([[[320.0, 240.0]]]))


class TestLoss(unittest.TestCase):
    def test_zero_for_perfect_prediction(self):
        target = torch.rand(2, 8, 16, 16)
        valid = torch.ones(2, 8, dtype=torch.bool)
        self.assertAlmostEqual(float(heatmap_loss(torch.logit(target), target, valid)), 0.0, places=6)

    def test_invalid_channels_are_ignored(self):
        target = torch.zeros(1, 8, 8, 8)
        valid = torch.zeros(1, 8, dtype=torch.bool)
        valid[0, 0] = True
        logits = torch.zeros(1, 8, 8, 8)
        baseline = float(heatmap_loss(logits, target, valid))
        logits[0, 3] += 5.0
        self.assertAlmostEqual(float(heatmap_loss(logits, target, valid)), baseline, places=7)

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            heatmap_loss(
                torch.zeros(1, 8, 8, 8),
                torch.zeros(1, 8, 4, 4),
                torch.ones(1, 8, dtype=torch.bool),
            )

    def test_coordinate_loss_prefers_correct_peak(self):
        logits = torch.full((1, 1, 16, 16), -8.0)
        logits[0, 0, 6, 10] = 8.0
        valid = torch.ones(1, 1, dtype=torch.bool)
        correct = torch.tensor([[[40.0, 24.0]]])
        wrong = torch.tensor([[[8.0, 56.0]]])
        self.assertLess(
            float(coordinate_loss(logits, correct, valid, (64, 64))),
            float(coordinate_loss(logits, wrong, valid, (64, 64))),
        )


class TestFocalLoss(unittest.TestCase):
    def test_zero_for_perfect_prediction(self):
        target = torch.zeros(2, 8, 16, 16)
        target[:, :, 8, 8] = 1.0
        logits = torch.where(target > 0.5, 10.0, -10.0)
        valid = torch.ones(2, 8, dtype=torch.bool)
        self.assertLess(float(focal_loss(logits, target, valid)), 1e-3)

    def test_positive_peak_is_penalised(self):
        target = torch.zeros(1, 8, 16, 16)
        target[0, :, 8, 8] = 1.0
        logits = torch.full((1, 8, 16, 16), -10.0)
        valid = torch.ones(1, 8, dtype=torch.bool)
        self.assertGreater(float(focal_loss(logits, target, valid)), 1.0)

    def test_missed_peak_gradient_is_alive(self):
        # regression guard: probability clamping used to zero this gradient
        target = torch.zeros(1, 1, 8, 8)
        target[0, 0, 4, 4] = 1.0
        logits = torch.full((1, 1, 8, 8), -20.0, requires_grad=True)
        valid = torch.ones(1, 1, dtype=torch.bool)
        focal_loss(logits, target, valid).backward()
        self.assertGreater(abs(float(logits.grad[0, 0, 4, 4])), 0.1)

    def test_invalid_channels_are_ignored(self):
        target = torch.zeros(1, 8, 8, 8)
        target[0, 0, 4, 4] = 1.0
        valid = torch.zeros(1, 8, dtype=torch.bool)
        valid[0, 0] = True
        logits = torch.full((1, 8, 8, 8), -2.0)
        baseline = float(focal_loss(logits, target, valid))
        logits[0, 3] += 5.0
        self.assertAlmostEqual(float(focal_loss(logits, target, valid)), baseline, places=7)


@unittest.skipUnless(DATASET_ROOT.exists() and MODELS_INFO.exists(), "local BOP dataset not available")
class TestHeatmapTorchDataset(unittest.TestCase):
    def test_tensor_shapes_and_normalization(self):
        model = BOPModel.from_models_info(MODELS_INFO, obj_id=1, scale=1.0)
        dataset = BOPInstanceDataset(
            DATASET_ROOT, model, scenes=["000000"], heatmap_size=(120, 160)
        )
        item = HeatmapDataset(dataset)[0]
        self.assertEqual(tuple(item["image"].shape), (3, 480, 640))
        self.assertEqual(item["image"].dtype, torch.float32)
        self.assertEqual(tuple(item["heatmap"].shape), (8, 120, 160))
        self.assertEqual(item["valid"].dtype, torch.bool)
        self.assertEqual(tuple(item["keypoints_2d"].shape), (8, 2))
        self.assertLessEqual(float(item["image"].max()), 3.0)
        self.assertGreaterEqual(float(item["image"].min()), -3.0)

    @unittest.skipUnless((V4_DIR / "train_pbr").exists(), "v4 validation dataset not available")
    def test_v4_crop_tensor_and_pose_metadata(self):
        model = BOPModel.from_models_info(V4_DIR / "models" / "models_info.json", obj_id=1)
        dataset = BOPInstanceDataset(
            V4_DIR / "train_pbr", model, scenes=["000000"], input_size=(256, 256),
            heatmap_size=(64, 64), crop_mode="bbox", heatmap_kernel="gaussian",
        )
        item = HeatmapDataset(dataset)[0]
        self.assertEqual(tuple(item["image"].shape), (3, 256, 256))
        self.assertEqual(tuple(item["heatmap"].shape), (8, 64, 64))
        self.assertEqual(tuple(item["K_input"].shape), (3, 3))
        self.assertEqual(tuple(item["R_m2c"].shape), (3, 3))
        self.assertEqual(tuple(item["t_m2c"].shape), (3,))


if __name__ == "__main__":
    unittest.main()
