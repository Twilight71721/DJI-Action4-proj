import sys
from pathlib import Path
import unittest

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import test_online_background as fixtures
from bop_bb8.cropping import corner_bounds, extract_window, raster_bounds, tight_crop_window
from bop_bb8.dataset import BOPInstanceDataset
from bop_bb8.geometry import crop_and_scale_intrinsics
from bop_bb8.online_background import OnlinePoseBackgroundDataset
from bop_bb8.torch_data import HeatmapDataset
from infer_resnet_yolo_real import prepare_crop, to_original
from evaluate_tight_cuboid import ReferenceDataset, map_points


class TestTightCropGeometry(unittest.TestCase):
    def test_negative_origin_does_not_wrap_source_pixels(self):
        image = np.arange(36, dtype=np.uint8).reshape(6, 6)
        result = extract_window(image, (-2, -1, 6, 5), (-1, -1, 3, 3))
        np.testing.assert_array_equal(result[1:5, 2:6], image[:4, :4])
        self.assertTrue((result[0] == 114).all())
        self.assertTrue((result[:, :2] == 114).all())

    def test_extremal_corners_stay_away_from_heatmap_border(self):
        points = np.array([[20, 30], [60, 30], [60, 50], [20, 50],
                           [25, 35], [65, 35], [65, 55], [25, 55]], float)
        crop = tight_crop_window(corner_bounds(points))
        transformed = (points-np.array(crop[:2])) * 256/np.array(crop[2:])
        self.assertTrue((transformed >= 3).all())
        self.assertTrue((transformed < 253).all())
        np.testing.assert_allclose(to_original(transformed, crop, (256, 256)), points)
        self.assertEqual(raster_bounds((20, 30, 65, 55)), (20, 30, 66, 56))

    def test_reject_invalid_corner_geometry(self):
        with self.assertRaises(ValueError):
            corner_bounds(np.full((8, 2), np.nan))
        with self.assertRaises(ValueError):
            tight_crop_window((0, 0, 0, 0))
        with self.assertRaises(ValueError):
            tight_crop_window((0, 0, 10, 10), border=128)


class TestTightCropDataset(fixtures.TestOnlineBackground):
    def tight_kwargs(self):
        return dict(self.online_kwargs, crop_mode='cuboid', crop_padding=1., crop_border=4,
                    input_size=(256, 256), heatmap_size=(64, 64))

    def test_training_and_inference_pixels_and_geometry_match(self):
        ds = BOPInstanceDataset(self.data/'trainval', self.model, **self.tight_kwargs())
        sample = ds[0]
        original = cv2.imread(sample.image_path)
        roi, tensor, crop = prepare_crop(original, sample.content_bbox_xyxy,
                                         (256, 256), 1., 'cuboid', 4)
        self.assertEqual(crop, sample.crop_xywh)
        np.testing.assert_array_equal(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB), sample.image)
        torch.testing.assert_close(tensor, HeatmapDataset(ds)[0]['image'], rtol=0, atol=0)
        self.assertTrue(sample.valid.all())
        np.testing.assert_allclose(sample.heatmaps.max(axis=(1, 2)), 1.)
        K = np.array([[200., 0, 100], [0, 200, 80], [0, 0, 1]])
        native, _ = cv2.projectPoints(self.model.corners_mm(), np.zeros(3), np.array([0., 0, 100.]), K, None)
        np.testing.assert_allclose(to_original(sample.keypoints_2d, crop, (256, 256)),
                                   native.reshape(8, 2), atol=1e-5)
        np.testing.assert_allclose(sample.K_input, crop_and_scale_intrinsics(K, crop, (256, 256)))

    def test_scene_pixels_outside_rectangle_have_no_effect(self):
        ds = BOPInstanceDataset(self.data/'trainval', self.model, **self.tight_kwargs())
        before = ds[0]
        original = cv2.imread(before.image_path)
        x1, y1, x2, y2 = raster_bounds(before.content_bbox_xyxy)
        altered = 255-original
        altered[y1:y2, x1:x2] = original[y1:y2, x1:x2]
        cv2.imwrite(before.image_path, altered)
        np.testing.assert_array_equal(before.image, ds[0].image)
        self.assertTrue((before.image[0] == 114).all())
        self.assertTrue((before.image[-1] == 114).all())

    def test_online_sources_keep_tight_geometry_and_neutral_padding(self):
        kwargs = self.tight_kwargs()
        ds = OnlinePoseBackgroundDataset(self.data/'trainval', self.model,
                                          online_background_config=self.config, **kwargs)
        baseline = BOPInstanceDataset(self.data/'trainval', self.model, **kwargs)
        for i in range(len(ds)):
            a, b = ds[i], baseline[i]
            for key in ['keypoints_2d', 'heatmaps', 'valid', 'K_input']:
                np.testing.assert_array_equal(getattr(a, key), getattr(b, key))
            self.assertTrue(a.valid.all())
            self.assertTrue((a.image[0] == 114).all())
            self.assertTrue((a.image[-1] == 114).all())

    def test_padding_and_jitter_cannot_silently_expand_tight_crops(self):
        with self.assertRaises(ValueError):
            BOPInstanceDataset(self.data/'trainval', self.model,
                               **dict(self.tight_kwargs(), crop_padding=1.3))
        with self.assertRaises(ValueError):
            BOPInstanceDataset(self.data/'trainval', self.model,
                               **dict(self.tight_kwargs(), crop_jitter_shift=.05))

    def test_reference_evaluation_agrees_across_crops_without_camera_size_fields(self):
        tight = BOPInstanceDataset(self.data/'trainval', self.model, **self.tight_kwargs())
        old = BOPInstanceDataset(self.data/'trainval', self.model,
                                **dict(self.tight_kwargs(), crop_mode='bbox', crop_padding=1.3))
        for ds in [old, tight]:
            _, cameras, _ = ds._load_scene(ds.instances[0][0])
            cameras['0'].pop('width', None)
            cameras['0'].pop('height', None)
        a, b = ReferenceDataset(old)[0], ReferenceDataset(tight)[0]
        reference_a = map_points(a['keypoints_2d'].numpy(), a['K_input'].numpy(), a['reference_K'])
        reference_b = map_points(b['keypoints_2d'].numpy(), b['K_input'].numpy(), b['reference_K'])
        np.testing.assert_allclose(reference_a, reference_b, atol=1e-4)
        np.testing.assert_allclose(reference_a, old[0].keypoints_2d, atol=1e-4)


if __name__ == '__main__':
    unittest.main()
