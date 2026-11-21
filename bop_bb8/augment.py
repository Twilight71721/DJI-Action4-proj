"""Training-only image degradation applied after deterministic data caching."""
from __future__ import annotations

import math

import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import gaussian_blur


class RandomBlurDataset(Dataset):
    """Mix clean and blurred crops without modifying cached tensors or labels."""

    def __init__(self, dataset, probability=0.5, sigma=(0.4, 1.6), kernel_size=9):
        self.dataset = dataset
        self.probability = float(probability)
        self.sigma = tuple(float(v) for v in sigma)
        self.kernel_size = int(kernel_size)
        if not 0 <= self.probability <= 1:
            raise ValueError("blur probability must be between 0 and 1")
        if len(self.sigma) != 2 or not all(math.isfinite(v) for v in self.sigma):
            raise ValueError("blur sigma must contain two finite values")
        if not 0 < self.sigma[0] <= self.sigma[1]:
            raise ValueError("blur sigma must satisfy 0 < min <= max")
        if self.kernel_size < 3 or self.kernel_size % 2 == 0:
            raise ValueError("blur kernel size must be odd and at least 3")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        if self.probability == 0 or torch.rand(()).item() >= self.probability:
            return sample
        sigma = torch.empty(()).uniform_(*self.sigma).item()
        result = dict(sample)
        # A normalized Gaussian filter with reflection padding commutes with
        # per-channel ImageNet normalization, so cached normalized crops are safe.
        result["image"] = gaussian_blur(sample["image"], self.kernel_size, sigma)
        return result
