"""Network building blocks for the 8-corner heatmap baseline (torch required).

This subpackage is intentionally not imported from ``bop_bb8/__init__.py`` so
the label-generation tools stay numpy-only.
"""
from .decode import (
    decode_argmax,
    decode_soft_argmax,
    decode_topk,
    heatmap_to_input_points,
)
from .loss import coordinate_loss, focal_loss, heatmap_loss
from .resnet_fpn import ResNetFPN

__all__ = [
    "ResNetFPN",
    "coordinate_loss",
    "focal_loss",
    "heatmap_loss",
    "decode_argmax",
    "decode_topk",
    "decode_soft_argmax",
    "heatmap_to_input_points",
]
