"""BOP -> ordered 8-corner heatmap labels (BoxDreamer-style baseline)."""
from .dataset import BOPInstanceDataset, BOPInstanceSample
from .geometry import (
    CORNER_SIGNS,
    BOPModel,
    corners_from_aabb,
    crop_and_scale_intrinsics,
    parse_intrinsics,
    parse_rotation,
    project_corners,
    scale_keypoints_to_heatmap,
)
from .heatmap import DEFAULT_VALUE_RANGE, make_bbox_heatmaps

__all__ = [
    "BOPInstanceDataset",
    "BOPInstanceSample",
    "CORNER_SIGNS",
    "BOPModel",
    "corners_from_aabb",
    "crop_and_scale_intrinsics",
    "parse_intrinsics",
    "parse_rotation",
    "project_corners",
    "scale_keypoints_to_heatmap",
    "DEFAULT_VALUE_RANGE",
    "make_bbox_heatmaps",
]
