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
from torch.utils.data import Sampler, Subset

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
        return self.as_tensors(sample, index)

    def as_tensors(self, sample, index):
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
            "crop_xywh": torch.tensor(sample.crop_xywh, dtype=torch.int64),
        }


class PairedHeatmapDataset(HeatmapDataset):
    """Two background views sharing one geometric supervision target."""

    def __getitem__(self, index):
        first, second = self.dataset.load_pair(index)
        result = self.as_tensors(first, index)
        for key in ['keypoints_2d', 'heatmaps', 'valid', 'K_input', 'R_m2c', 't_m2c', 'crop_xywh']:
            np.testing.assert_array_equal(getattr(first, key), getattr(second, key))
        result['image_pair'] = self.as_tensors(second, index)['image']
        return result


class ShapeBucketBatchSampler(Sampler):
    """Batch equal-sized rectangles; optional weighted replacement preserves epoch size."""

    def __init__(self, dataset, batch_size: int, shuffle: bool, seed: int = 0, sample_weights=None):
        if batch_size < 1:
            raise ValueError('batch_size must be positive')
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.sample_weights = None if sample_weights is None else np.asarray(sample_weights,float)
        if self.sample_weights is not None:
            if (not shuffle or self.sample_weights.shape != (len(dataset),) or
                    not np.isfinite(self.sample_weights).all() or (self.sample_weights<=0).any()):
                raise ValueError('Positive per-instance weights require a shuffled training sampler')
        self.groups = {}
        for index in range(len(dataset)):
            base, source_index = dataset, index
            while hasattr(base, 'dataset'):
                if isinstance(base, Subset):
                    source_index = base.indices[source_index]
                base = base.dataset
            if hasattr(base, 'get_input_size'):
                size = tuple(base.get_input_size(source_index))
            else:
                size = tuple(base[source_index]['image'].shape[-2:][::-1])
            self.groups.setdefault(size, []).append(index)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        return sum((len(indices)+self.batch_size-1)//self.batch_size for indices in self.groups.values())

    def __iter__(self):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch]))
        batches = []
        for group in self.groups.values():
            indices = list(group)
            if self.sample_weights is not None:
                weights=self.sample_weights[indices]
                indices=rng.choice(indices,size=len(indices),replace=True,p=weights/weights.sum()).tolist()
            elif self.shuffle:
                rng.shuffle(indices)
            batches.extend(indices[start:start+self.batch_size]
                           for start in range(0, len(indices), self.batch_size))
        if self.shuffle:
            rng.shuffle(batches)
        yield from batches
