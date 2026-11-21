import unittest

import torch
from torchvision.transforms.functional import gaussian_blur

from bop_bb8.augment import RandomBlurDataset


class TestRandomBlurDataset(unittest.TestCase):
    def setUp(self):
        self.sample = {
            "image": torch.arange(3 * 32 * 32).reshape(3, 32, 32).float().remainder(7) / 7,
            "heatmap": torch.rand(8, 8, 8),
            "keypoints_2d": torch.rand(8, 2),
            "valid": torch.ones(8, dtype=torch.bool),
            "K_input": torch.eye(3),
        }

    def test_disabled_is_exact_and_does_not_consume_rng(self):
        before = torch.random.get_rng_state()
        result = RandomBlurDataset([self.sample], probability=0)[0]
        self.assertIs(result, self.sample)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

    def test_cache_and_all_labels_remain_unchanged(self):
        originals = {k: v.clone() for k, v in self.sample.items()}
        result = RandomBlurDataset([self.sample], probability=1)[0]
        self.assertFalse(torch.equal(result["image"], self.sample["image"]))
        for key, original in originals.items():
            self.assertTrue(torch.equal(self.sample[key], original))
            if key != "image":
                self.assertTrue(torch.equal(result[key], original))

    def test_blur_is_resampled_after_cache_and_seed_reproducible(self):
        dataset = RandomBlurDataset([self.sample], probability=1)
        torch.manual_seed(123)
        first, second = dataset[0]["image"], dataset[0]["image"]
        self.assertFalse(torch.equal(first, second))
        torch.manual_seed(123)
        self.assertTrue(torch.equal(first, dataset[0]["image"]))

    def test_normalized_blur_matches_rgb_blur(self):
        rgb = self.sample["image"]
        mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        sample = dict(self.sample, image=(rgb - mean) / std)
        result = RandomBlurDataset([sample], probability=1, sigma=(1, 1))[0]
        expected = (gaussian_blur(rgb, 9, 1.0) - mean) / std
        torch.testing.assert_close(result["image"], expected, atol=1e-6, rtol=1e-5)

    def test_invalid_parameters(self):
        for kwargs in [dict(probability=1.1), dict(sigma=(0, 1)),
                       dict(sigma=(2, 1)), dict(sigma=(1, float("nan"))),
                       dict(kernel_size=4)]:
            with self.assertRaises(ValueError):
                RandomBlurDataset([], **kwargs)


if __name__ == "__main__":
    unittest.main()
