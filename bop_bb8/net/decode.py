"""Heatmap decoding: corner coordinates from probability maps.

Functions accept ``[B, C, H, W]`` maps; ``decode_topk(from_logits=True)``
accepts raw network logits instead of probabilities. They return ``[B, C, 2]``
float coordinates ``(x, y)`` on the *heatmap grid*. Use
:func:`heatmap_to_input_points` to map them back to input-frame pixels before
PnP or pixel-error metrics.  BoxDreamer's ``top-20 mean`` baseline is covered
by ``decode_topk(..., k=20, weighted=False)``. Both global top-k variants can
average between separated peaks; weighting does not guarantee a local maximum.
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


def decode_topk(heatmaps: torch.Tensor, k: int = 9, weighted: bool = True,
                *, from_logits: bool = False) -> torch.Tensor:
    """Top-k mean without an absolute probability floor.

    Prefer ``from_logits=True`` at inference: log-sigmoid normalization also
    handles values whose sigmoid underflows. Probability-only all-zero maps
    return NaN, since they contain no localization evidence.
    """
    if heatmaps.dim() != 4:
        raise ValueError(f"expected [B, C, H, W], got {tuple(heatmaps.shape)}")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    height, width = heatmaps.shape[-2:]
    k = min(int(k), height * width)
    flat = heatmaps.float().reshape(heatmaps.shape[0], heatmaps.shape[1], -1)
    if from_logits:
        flat = torch.nn.functional.logsigmoid(flat)
    values, indices = torch.topk(flat, k=k, dim=-1)
    points = _indices_to_xy(indices, width)
    if not weighted:
        return points.mean(dim=-2)
    if from_logits:
        weights = torch.softmax(values, dim=-1)
        return (points * weights.unsqueeze(-1)).sum(dim=-2)
    weights = values.clamp_min(0.0)
    maximum = weights.amax(dim=-1, keepdim=True)
    weights = weights / torch.where(maximum > 0, maximum, torch.ones_like(maximum))
    denom = weights.sum(dim=-1, keepdim=True)
    result = (points * weights.unsqueeze(-1)).sum(dim=-2) / torch.where(
        denom > 0, denom, torch.ones_like(denom))
    return result.masked_fill(maximum <= 0, float("nan"))


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


def decode_local_peak(logits: torch.Tensor, radius: int = 3) -> torch.Tensor:
    """Subpixel mean around the strongest peak, never across distant modes.

    Inputs are raw logits of a spatial heatmap distribution. No confidence
    threshold or geometric fitting is applied; a wrong peak remains wrong.
    """
    if logits.dim() != 4 or not isinstance(radius, int) or radius < 0:
        raise ValueError('Expected [B,C,H,W] logits and an integer radius >= 0')
    if not torch.isfinite(logits).all():
        raise ValueError('Cannot decode non-finite logits')
    height, width = logits.shape[-2:]
    flat = logits.float().flatten(2)
    peak = _indices_to_xy(flat.argmax(-1), width)
    xs = torch.arange(width, device=logits.device).repeat(height)
    ys = torch.arange(height, device=logits.device).repeat_interleave(width)
    local = ((xs - peak[..., 0, None]).abs() <= radius) & (
        (ys - peak[..., 1, None]).abs() <= radius)
    weights = torch.softmax(flat.masked_fill(~local, -torch.inf), dim=-1)
    return torch.stack(((weights * xs).sum(-1), (weights * ys).sum(-1)), dim=-1)


def decode_heatmaps(logits: torch.Tensor, method: str = 'topk', radius: int = 3) -> torch.Tensor:
    """Shared train-validation/inference dispatch; old checkpoints keep top-9."""
    if method == 'topk':
        return decode_topk(logits, k=9, from_logits=True)
    if method == 'local':
        return decode_local_peak(logits, radius=radius)
    raise ValueError(f'Unknown heatmap decoder: {method}')


def heatmap_probabilities(logits: torch.Tensor, loss: str = 'composite') -> torch.Tensor:
    """Export the probability convention used by the checkpoint's objective."""
    if loss == 'spatial_kl':
        return torch.softmax(logits.float().flatten(2), dim=-1).reshape_as(logits)
    return logits.float().sigmoid()


def heatmap_to_input_points(
    points: torch.Tensor,
    heatmap_size: tuple[int, int],
    input_size: tuple[int, int],
    margin: int = 0,
) -> torch.Tensor:
    """Scale ``[..., 2]`` ``(x, y)`` from the heatmap grid to input pixels.

    ``heatmap_size`` and ``input_size`` are both ``(H, W)``; the mapping is a
    per-axis scaling after removing the output-only margin, mirroring
    :func:`bop_bb8.geometry.scale_keypoints_to_heatmap`.
    """
    if not isinstance(margin, int) or margin < 0 or min(heatmap_size) <= 2*margin:
        raise ValueError('Invalid output heatmap margin')
    scale = torch.tensor(
        [
            input_size[1] / (heatmap_size[1]-2*margin),
            input_size[0] / (heatmap_size[0]-2*margin),
        ],
        device=points.device,
        dtype=points.dtype,
    )
    return (points - margin) * scale
