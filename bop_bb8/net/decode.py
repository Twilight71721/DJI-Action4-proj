"""Heatmap decoding: corner coordinates from probability maps.

All functions accept ``[B, C, H, W]`` probabilities (the networks' output) and
return ``[B, C, 2]`` float coordinates ``(x, y)`` on the *heatmap grid*.  Use
:func:`heatmap_to_input_points` to map them back to input-frame pixels before
PnP or pixel-error metrics.  BoxDreamer's ``top-20 mean`` baseline is covered
by ``decode_topk(..., k=20, weighted=False)``; the weighted variant is the
planned default because an unweighted mean splits between two peaks.
"""
from __future__ import annotations

import torch


def _indices_to_xy(flat_index: torch.Tensor, width: int) -> torch.Tensor:
    x = (flat_index % width).to(torch.float32)
    y = (flat_index // width).to(torch.float32)
    return torch.stack((x, y), dim=-1)


def decode_argmax(heatmaps: torch.Tensor) -> torch.Tensor:
    """Integer grid coordinates of the per-channel maximum."""
    if heatmaps.dim() != 4:
        raise ValueError(f"expected [B, C, H, W], got {tuple(heatmaps.shape)}")
    width = heatmaps.shape[-1]
    flat = heatmaps.reshape(heatmaps.shape[0], heatmaps.shape[1], -1)
    return _indices_to_xy(flat.argmax(dim=-1), width)


def decode_topk(heatmaps: torch.Tensor, k: int = 9, weighted: bool = True) -> torch.Tensor:
    """Mean of the ``k`` highest pixels, optionally weighted by their values."""
    if heatmaps.dim() != 4:
        raise ValueError(f"expected [B, C, H, W], got {tuple(heatmaps.shape)}")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    height, width = heatmaps.shape[-2:]
    k = min(int(k), height * width)
    flat = heatmaps.reshape(heatmaps.shape[0], heatmaps.shape[1], -1)
    values, indices = torch.topk(flat, k=k, dim=-1)
    points = _indices_to_xy(indices, width)
    if not weighted:
        return points.mean(dim=-2)
    weights = values.clamp_min(0.0)
    denom = weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return (points * weights.unsqueeze(-1)).sum(dim=-2) / denom


def decode_soft_argmax(heatmaps: torch.Tensor, temperature: float = 50.0) -> torch.Tensor:
    """Expected coordinate under a spatial softmax of the heatmap."""
    if heatmaps.dim() != 4:
        raise ValueError(f"expected [B, C, H, W], got {tuple(heatmaps.shape)}")
    height, width = heatmaps.shape[-2:]
    flat = heatmaps.reshape(heatmaps.shape[0], heatmaps.shape[1], -1)
    prob = torch.softmax(flat * temperature, dim=-1)
    xs = torch.arange(width, device=flat.device, dtype=flat.dtype).repeat(height)
    ys = torch.arange(height, device=flat.device, dtype=flat.dtype).repeat_interleave(width)
    x = (prob * xs).sum(dim=-1)
    y = (prob * ys).sum(dim=-1)
    return torch.stack((x, y), dim=-1)


def heatmap_to_input_points(
    points: torch.Tensor,
    heatmap_size: tuple[int, int],
    input_size: tuple[int, int],
) -> torch.Tensor:
    """Scale ``[..., 2]`` ``(x, y)`` from the heatmap grid to input pixels.

    ``heatmap_size`` and ``input_size`` are both ``(H, W)``; the mapping is a
    pure per-axis scaling, mirroring
    :func:`bop_bb8.geometry.scale_keypoints_to_heatmap`.
    """
    scale = torch.tensor(
        [
            input_size[1] / heatmap_size[1],
            input_size[0] / heatmap_size[0],
        ],
        device=points.device,
        dtype=points.dtype,
    )
    return points * scale
