import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from bop_bb8 import (
    BOPInstanceDataset,
    BOPModel,
    corners_from_aabb,
    crop_and_scale_intrinsics,
    make_bbox_heatmaps,
    project_corners,
    scale_keypoints_to_heatmap,
)
from bop_bb8.visualize import draw_sample

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_DIR / "dataset" / "train_pbr"
MODELS_INFO = PROJECT_DIR / "dataset" / "dji-action4" / "models" / "models_info.json"
V4_DIR = PROJECT_DIR / "dataset" / "action4-v4-random-validation-100"
V4_ROOT = V4_DIR / "train_pbr"
V4_MODELS_INFO = V4_DIR / "models" / "models_info.json"


def synthetic_scene():
    corners = corners_from_aabb([-10.0, -20.0, -30.0], [10.0, 20.0, 30.0])
    K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    R = np.eye(3)
    t = np.array([0.0, 0.0, 1000.0])
    return corners, K, R, t


class TestGeometry(unittest.TestCase):
    def test_corner_order_and_shape(self):
        corners = corners_from_aabb([1.0, 2.0, 3.0], [11.0, 22.0, 33.0])
        self.assertEqual(corners.shape, (8, 3))
        expected = [
            [1, 2, 3], [11, 2, 3], [11, 22, 3], [1, 22, 3],
            [1, 2, 33], [11, 2, 33], [11, 22, 33], [1, 22, 33],
        ]
        np.testing.assert_allclose(corners, np.array(expected, dtype=float))

    def test_projection_formula_and_units(self):
        corners, K, R, t = synthetic_scene()
        kps, valid, points_cam = project_corners(corners, K, R, t, image_size=(640, 480))
        self.assertTrue(valid.all())
        np.testing.assert_allclose(points_cam[:, 2], t[2] + np.array([-30, -30, -30, -30, 30, 30, 30, 30]))
        np.testing.assert_allclose(kps[0], [320.0 - 10 / 970 * 500, 240.0 - 20 / 970 * 500])

    def test_projection_behind_camera_and_out_of_image(self):
        K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
        corners = np.array([
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 10.0],
            [1000.0, 0.0, 10.0],
            [-1000.0, 0.0, 10.0],
            [0.0, 1000.0, 10.0],
            [0.0, -1000.0, 10.0],
            [0.0, 0.0, 10.0],
            [0.0, 0.0, 10.0],
        ])
        kps, valid, _ = project_corners(corners, K, np.eye(3), np.zeros(3), image_size=(640, 480))
        self.assertFalse(valid[0])
        self.assertTrue(np.isnan(kps[0]).all())
        self.assertTrue(valid[1])
        for i in (2, 3, 4, 5):
            self.assertFalse(valid[i], f"corner {i} should be outside the image")
        self.assertTrue(np.isfinite(kps[2]).all())
        self.assertTrue(valid[6])

    def test_crop_and_scale_intrinsics(self):
        corners, K, R, t = synthetic_scene()
        crop = (100, 50, 400, 300)
        out_size = (200, 150)
        K_crop = crop_and_scale_intrinsics(K, crop, out_size)
        kps_full, valid, _ = project_corners(corners, K, R, t, image_size=(640, 480))
        kps_crop, valid_crop, _ = project_corners(corners, K_crop, R, t, image_size=out_size)
        expected = np.stack([(kps_full[:, 0] - crop[0]) * out_size[0] / crop[2],
                             (kps_full[:, 1] - crop[1]) * out_size[1] / crop[3]], axis=1)
        np.testing.assert_allclose(kps_crop[valid_crop], expected[valid_crop])
        np.testing.assert_allclose(kps_crop[~valid_crop], expected[~valid_crop], equal_nan=True)


class TestHeatmap(unittest.TestCase):
    def test_shape_range_and_peak_position(self):
        kps = np.array([
            [10.0, 10.0], [20.0, 10.0], [30.0, 10.0], [40.0, 10.0],
            [10.0, 50.0], [20.0, 50.0], [30.0, 50.0], [40.0, 50.0],
        ])
        heatmaps = make_bbox_heatmaps(kps, np.ones(8, dtype=bool), (64, 64))
        self.assertEqual(heatmaps.shape, (8, 64, 64))
        self.assertEqual(heatmaps.dtype, np.float32)
        self.assertTrue(np.isfinite(heatmaps).all())
        self.assertGreaterEqual(heatmaps.min(), 0.0)
        self.assertLessEqual(heatmaps.max(), 1.0)
        for i in range(8):
            row, col = np.unravel_index(np.argmax(heatmaps[i]), heatmaps[i].shape)
            self.assertEqual((row, col), (int(kps[i][1]), int(kps[i][0])))

    def test_reference_formula(self):
        kps = np.array([
            [10.0, 12.0], [25.0, 12.0], [40.0, 12.0], [55.0, 12.0],
            [10.0, 47.0], [25.0, 47.0], [40.0, 47.0], [55.0, 47.0],
        ])
        valid = np.ones(8, dtype=bool)
        heatmaps = make_bbox_heatmaps(kps, valid, (64, 64))
        center = kps.mean(axis=0)
        yy, xx = np.mgrid[0:64, 0:64].astype(float)
        i = 5
        d = np.sqrt((xx - kps[i, 0]) ** 2 + (yy - kps[i, 1]) ** 2)
        scale = (np.linalg.norm(kps[i] - center) / 10) ** 2
        ref = np.exp(-d / scale)
        ref /= ref.max()
        np.testing.assert_allclose(heatmaps[i], ref.astype(np.float32), rtol=1e-6, atol=1e-6)

    def test_gaussian_kernel(self):
        kps = np.tile(np.array([[10.0, 12.0]]), (8, 1))
        heatmaps = make_bbox_heatmaps(
            kps, np.ones(8, dtype=bool), (32, 32), kernel="gaussian", sigma=2.0
        )
        self.assertAlmostEqual(float(heatmaps[0, 12, 10]), 1.0, places=6)
        self.assertAlmostEqual(float(heatmaps[0, 12, 12]), np.exp(-0.5), places=6)

    def test_invalid_channel_is_background(self):
        kps = np.array([
            [10.0, 10.0], [20.0, 10.0], [30.0, 10.0], [40.0, 10.0],
            [10.0, 50.0], [20.0, 50.0], [30.0, 50.0], [40.0, 50.0],
        ])
        valid = np.ones(8, dtype=bool)
        valid[3] = False
        valid[6] = False
        heatmaps = make_bbox_heatmaps(kps, valid, (64, 64))
        self.assertTrue(np.all(heatmaps[3] == 0.0))
        self.assertTrue(np.all(heatmaps[6] == 0.0))
        self.assertGreater(heatmaps[2].max(), 0.0)

    def test_zero_scale_guard(self):
        kps = np.tile(np.array([[16.0, 16.0]]), (8, 1))
        heatmaps = make_bbox_heatmaps(kps, np.ones(8, dtype=bool), (32, 32))
        self.assertTrue(np.isfinite(heatmaps).all())
        self.assertEqual(np.argmax(heatmaps[0]), 16 * 32 + 16)

    def test_custom_value_range(self):
        kps = np.tile(np.array([[8.0, 8.0]]), (8, 1))
        heatmaps = make_bbox_heatmaps(kps, np.ones(8, dtype=bool), (16, 16), value_range=(-1.0, 1.0))
        self.assertAlmostEqual(float(heatmaps.min()), -1.0, places=6)
        self.assertAlmostEqual(float(heatmaps.max()), 1.0, places=6)

    def test_scale_keypoints_to_heatmap(self):
        kps = np.array([[320.0, 240.0]])
        out = scale_keypoints_to_heatmap(kps, (480, 640), (120, 160))
        np.testing.assert_allclose(out, [[80.0, 60.0]])


@unittest.skipUnless(DATASET_ROOT.exists() and MODELS_INFO.exists(), "local BOP dataset not available")
class TestRealDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = BOPModel.from_models_info(MODELS_INFO, obj_id=1, scale=1.0)
        cls.dataset = BOPInstanceDataset(DATASET_ROOT, cls.model, scenes=["000000"], heatmap_size=(120, 160))
        with open(DATASET_ROOT / "000000" / "scene_camera.json") as f:
            cls.camera = json.load(f)
        with open(DATASET_ROOT / "000000" / "scene_gt_info.json") as f:
            cls.info = json.load(f)

    def test_index_and_fields(self):
        self.assertEqual(len(self.dataset), 1000)
        sample = self.dataset[0]
        self.assertEqual((sample.scene_id, sample.image_id, sample.instance_idx), ("000000", 0, 0))
        self.assertEqual(sample.image.shape, (480, 640, 3))
        self.assertEqual(sample.image.dtype, np.uint8)
        self.assertEqual(sample.corners_3d.shape, (8, 3))
        self.assertEqual(sample.keypoints_2d.shape, (8, 2))
        self.assertEqual(sample.valid.shape, (8,))
        self.assertEqual(sample.heatmaps.shape, (8, 120, 160))
        self.assertEqual(sample.K_input.shape, (3, 3))
        self.assertTrue(np.isfinite(sample.heatmaps).all())

    def test_K_and_mask_association(self):
        sample = self.dataset[0]
        K_gt = np.array(self.camera["0"]["cam_K"], dtype=float).reshape(3, 3)
        np.testing.assert_allclose(sample.K_input, K_gt)
        mask_path = DATASET_ROOT / "000000" / "mask" / f"{sample.image_id:06d}_{sample.instance_idx:06d}.png"
        self.assertTrue(mask_path.exists())
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        ys, xs = np.nonzero(mask > 0)
        mask_bbox = (xs.min(), ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)
        bbox_obj = tuple(self.info["0"][0]["bbox_obj"])
        for got, want in zip(mask_bbox, bbox_obj):
            self.assertLessEqual(abs(got - want), 1)

    def test_keypoints_match_manual_projection(self):
        sample = self.dataset.load_instance("000000", 3, 0)
        with open(DATASET_ROOT / "000000" / "scene_gt.json") as f:
            gt = json.load(f)
        inst = gt["3"][0]
        R = np.array(inst["cam_R_m2c"], dtype=float).reshape(3, 3)
        t = np.array(inst["cam_t_m2c"], dtype=float)
        kps, valid, _ = project_corners(sample.corners_3d, sample.K_input, R, t, image_size=(640, 480))
        np.testing.assert_allclose(sample.keypoints_2d[valid], kps[valid], rtol=1e-5)
        np.testing.assert_array_equal(sample.valid, valid)
        np.testing.assert_allclose(sample.t_m2c, t)

    def test_crop_resize_heatmap_consistency(self):
        dataset = BOPInstanceDataset(
            DATASET_ROOT, self.model, scenes=["000000"],
            crop=(100, 50, 400, 300), input_size=(200, 150), heatmap_size=(75, 100),
        )
        sample = dataset[0]
        self.assertEqual(sample.image.shape, (150, 200, 3))
        self.assertEqual(sample.heatmaps.shape, (8, 75, 100))
        K_gt = np.array(self.camera["0"]["cam_K"], dtype=float).reshape(3, 3)
        np.testing.assert_allclose(sample.K_input, crop_and_scale_intrinsics(K_gt, (100, 50, 400, 300), (200, 150)))
        for i in range(8):
            if sample.valid[i]:
                expected = sample.keypoints_2d[i] * np.array([100 / 200, 75 / 150])
                row, col = np.unravel_index(np.argmax(sample.heatmaps[i]), sample.heatmaps[i].shape)
                self.assertLessEqual(abs(col - expected[0]), 1.0)
                self.assertLessEqual(abs(row - expected[1]), 1.0)

    def test_visib_fract_filter(self):
        dataset = BOPInstanceDataset(DATASET_ROOT, self.model, scenes=["000000"], min_visib_fract=0.99)
        self.assertGreater(len(dataset), 0)
        self.assertLess(len(dataset), 1000)
        for _, image_id, instance_idx in dataset.instances:
            self.assertGreaterEqual(float(self.info[str(image_id)][instance_idx]["visib_fract"]), 0.99)

    def test_visualization(self):
        sample = self.dataset[0]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "sample.png"
            drawn = draw_sample(sample, bbox_obj=tuple(self.info["0"][0]["bbox_obj"]), out_path=out)
            self.assertTrue(out.exists())
            self.assertEqual(drawn.shape, sample.image.shape)


@unittest.skipUnless(V4_ROOT.exists() and V4_MODELS_INFO.exists(), "v4 validation dataset not available")
class TestV4Dataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model = BOPModel.from_models_info(V4_MODELS_INFO, obj_id=1)
        cls.dataset = BOPInstanceDataset(
            V4_ROOT,
            model,
            scenes=["000000"],
            input_size=(256, 256),
            heatmap_size=(64, 64),
            crop_mode="bbox",
            crop_padding=1.3,
            heatmap_kernel="gaussian",
            heatmap_sigma=2.0,
        )
        with open(V4_DIR / "corner-projections.json") as f:
            cls.reference = {int(x["frame"]): x for x in json.load(f)["samples"]}

    def test_png_and_instance_crop(self):
        sample = self.dataset[0]
        self.assertTrue(sample.image_path.endswith(".png"))
        self.assertEqual(sample.image.shape, (256, 256, 3))
        self.assertEqual(sample.heatmaps.shape, (8, 64, 64))
        self.assertEqual(sample.crop_xywh[2], sample.crop_xywh[3])
        self.assertTrue(sample.valid.all())

    def test_projection_matches_renderer_reference(self):
        for frame in (0, 17, 59, 98):
            sample = self.dataset.load_instance("000000", frame, 0)
            ref = np.asarray(self.reference[frame]["corners_2d"], dtype=np.float64)
            x0, y0, wc, hc = sample.crop_xywh
            expected = np.column_stack(
                ((ref[:, 0] - x0) * 256.0 / wc, (ref[:, 1] - y0) * 256.0 / hc)
            )
            np.testing.assert_allclose(sample.keypoints_2d, expected, atol=1e-3)

    def test_heatmap_peak_decodes_to_corner(self):
        sample = self.dataset[0]
        expected = sample.keypoints_2d / 4.0
        for channel in range(8):
            row, col = np.unravel_index(
                np.argmax(sample.heatmaps[channel]), sample.heatmaps[channel].shape
            )
            self.assertLessEqual(abs(col - expected[channel, 0]), 1.0)
            self.assertLessEqual(abs(row - expected[channel, 1]), 1.0)


if __name__ == "__main__":
    unittest.main()
