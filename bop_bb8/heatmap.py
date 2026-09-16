"""8-channel corner heatmaps for BB8 corner supervision.

Baseline (``third_party/BoxDreamer/src/datasets/utils/base/bbox_utils.py``,
``type="heatmap"``), per channel ``i``:

    d_i(x, y) = sqrt((x - u_i)^2 + (y - v_i)^2)
    center    = mean_i(p_i)
    scale_i   = (||p_i - center|| / 10)^2
    h_i(x, y) = exp(-d_i(x, y) / scale_i)

followed by per-channel max normalization.  This is *not* a standard 2D
Gaussian kernel (no 2D quadratic form, no 2*sigma^2 denominator).

``kernel="boxdreamer"`` preserves that baseline. ``kernel="gaussian"`` uses
the standard keypoint target ``exp(-d^2/(2*sigma^2))`` and is the training
default for the v4 instance-crop pipeline.

Shared adaptations required by the label spec:

- ``center`` is the mean of the *valid* corners only.
- invalid corners (``Z <= 0`` or outside the input image) keep the background
  value; their channel is never normalized at the image border.
- normalization is clamped against division by zero.
- values are mapped into ``value_range``; the default ``(0.0, 1.0)`` means
  background ``0.0``.  BoxDreamer maps to ``[-1, 1]``; the choice here must
  match the future prediction head / loss and is therefore part of the config.
"""
from __future__ import annotations

import numpy as np

BACKGROUND = 0.0
DEFAULT_VALUE_RANGE = (0.0, 1.0)
SCALE_EPS = 1e-6


def make_bbox_heatmaps(
    keypoints_2d: np.ndarray,
    valid: np.ndarray,
    out_size: tuple[int, int],
    value_range: tuple[float, float] = DEFAULT_VALUE_RANGE,
    kernel: str = "boxdreamer",
    sigma: float = 2.0,
) -> np.ndarray:
    """Build ``[8, H, W]`` corner heatmaps.

    ``keypoints_2d`` must already be expressed on the heatmap grid (see
    :func:`bop_bb8.geometry.scale_keypoints_to_heatmap`); ``out_size`` is
    ``(H, W)``.
    """
    kps = np.asarray(keypoints_2d, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if kps.shape != (8, 2) or valid.shape != (8,):
        raise ValueError(f"expected keypoints [8, 2] and valid [8], got {kps.shape} {valid.shape}")
    if kernel not in {"boxdreamer", "gaussian"}:
        raise ValueError(f"unsupported heatmap kernel: {kernel}")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    height, width = out_size
    out_lo, out_hi = float(value_range[0]), float(value_range[1])
    heatmaps = np.full((8, height, width), out_lo, dtype=np.float32)
    if not valid.any():
        return heatmaps

    # Unit spacing: pixel indices act as pixel coordinates, as in BoxDreamer.
    grid_y, grid_x = np.mgrid[0:height, 0:width].astype(np.float64)
    center = kps[valid].mean(axis=0)

    for i in range(8):
        if not valid[i]:
            continue
        u, v = kps[i]
        dx2_dy2 = (grid_x - u) ** 2 + (grid_y - v) ** 2
        if kernel == "gaussian":
            channel = np.exp(-dx2_dy2 / (2.0 * sigma ** 2))
        else:
            scale = (np.linalg.norm(kps[i] - center) / 10.0) ** 2
            scale = max(scale, SCALE_EPS)
            channel = np.exp(-np.sqrt(dx2_dy2) / scale)
        peak = channel.max()
        if peak <= 0.0:
            continue
        channel /= peak
        heatmaps[i] = (out_lo + (out_hi - out_lo) * channel).astype(np.float32)
    return heatmaps
