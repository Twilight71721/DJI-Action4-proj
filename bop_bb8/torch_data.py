"""Torch ``Dataset`` wrapper around :class:`bop_bb8.dataset.BOPInstanceDataset`.

Kept out of ``bop_bb8/__init__.py`` so the label-generation tools stay
numpy-only and do not require torch.  Each item is a dict of tensors:

- ``image``:        ``[3, H, W]`` float32, ImageNet-normalised RGB;
- ``heatmap``:      ``[8, Hh, Wh]`` float32 probabilities in ``[0, 1]``;
- ``valid``:        ``[8]`` bool, corners with a meaningful target;
- ``keypoints_2d``: ``[8, 2]`` float32 input-frame pixels, NaN -> 0;
- ``index``:        int, position in the wrapped dataset.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .dataset import BOPInstanceDataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
_MEAN = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
_STD = torch.tensor(IMAGENET_STD).view(3, 1, 1)


class HeatmapDataset(Dataset):
    """Expose :class:`BOPInstanceSample` as training tensors."""

    def __init__(self, dataset: BOPInstanceDataset, normalize: bool = True) -> None:
        self.dataset = dataset
        self.normalize = bool(normalize)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict:
        sample = self.dataset[index]
        image = torch.from_numpy(np.ascontiguousarray(sample.image)).permute(2, 0, 1)
        image = image.float().div_(255.0)
        if self.normalize:
            image = (image - _MEAN) / _STD
        keypoints = torch.from_numpy(np.nan_to_num(sample.keypoints_2d, nan=0.0))
        return {
            "image": image,
            "heatmap": torch.from_numpy(np.ascontiguousarray(sample.heatmaps)).float(),
            "valid": torch.from_numpy(np.ascontiguousarray(sample.valid)).bool(),
            "keypoints_2d": keypoints.float(),
            "index": index,
            "K_input": torch.from_numpy(np.ascontiguousarray(sample.K_input)).float(),
            "R_m2c": torch.from_numpy(np.ascontiguousarray(sample.R_m2c)).float(),
            "t_m2c": torch.from_numpy(np.ascontiguousarray(sample.t_m2c)).float(),
        }
