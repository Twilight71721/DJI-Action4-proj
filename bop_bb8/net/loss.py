"""Heatmap regression losses shared by all backbones.

Both losses take the networks' raw **logits** and the ``[0, 1]`` targets from
``bop_bb8.heatmap``.  ``focal_loss`` (CornerNet-style, adapted from
BoxDreamer's ``src/loss/utils/focal_loss.py``) is the default: the targets are
extremely peaked, so a plain Smooth-L1/MSE on probabilities can be driven to
~0 by predicting an all-background map.

Never clamp probabilities before a log-based loss: BoxDreamer clamps to
``[1e-4, 1 - 1e-4]``, and once every output saturates below the clamp the loss
gradient is exactly zero, so training freezes at ``-log(1e-4)`` per positive
pixel.  ``logsigmoid`` on the logits keeps missed peaks recoverable.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

POS_EPS = 1e-4


def _valid_mask(valid: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if valid.shape != reference.shape[:2]:
        raise ValueError(
            f"valid must be {tuple(reference.shape[:2])}, got {tuple(valid.shape)}"
        )
    return valid.to(reference.dtype).unsqueeze(-1).unsqueeze(-1)


def heatmap_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    beta: float = 1.0,
) -> torch.Tensor:
    """Smooth-L1 between ``sigmoid(logits)`` and the ``[0, 1]`` targets.

    ``valid`` is ``[B, C]`` and excludes corners that are behind the camera or
    project outside the input frame, whose target channel is pure background.
    The result is averaged over valid channels and pixels so samples with
    fewer visible corners do not dominate.
    """
    if logits.shape != targets.shape:
        raise ValueError(f"shape mismatch: {logits.shape} vs {targets.shape}")
    weight = _valid_mask(valid, logits)
    per_pixel = F.smooth_l1_loss(
        torch.sigmoid(logits), targets, reduction="none", beta=beta
    )
    denom = weight.sum() * per_pixel.shape[-2] * per_pixel.shape[-1]
    return (per_pixel * weight).sum() / denom.clamp_min(1.0)


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    alpha: float = 2.0,
    beta: float = 4.0,
) -> torch.Tensor:
    """CornerNet focal loss on raw logits.

    Positive pixels are targets at (or numerically indistinguishable from)
    ``1.0``; every other pixel is a negative weighted by ``(1 - target)^beta``,
    which softens the penalty in the ring around each peak.  The total is
    normalised by the number of positive pixels in the batch, so background
    pixels cannot dominate.  ``valid`` masks channels whose corner is behind
    the camera or outside the image (all-background targets).
    """
    if logits.shape != targets.shape:
        raise ValueError(f"shape mismatch: {logits.shape} vs {targets.shape}")
    mask = _valid_mask(valid, logits)
    target = targets.clamp(0.0, 1.0)
    prob = torch.sigmoid(logits)

    positive = ((target >= 1.0 - POS_EPS) * mask).to(logits.dtype)
    negative = (1.0 - positive) * mask
    pos_loss = -F.logsigmoid(logits) * (1.0 - prob).pow(alpha) * positive
    neg_loss = (
        -F.logsigmoid(-logits) * prob.pow(alpha) * (1.0 - target).pow(beta) * negative
    )
    num_pos = positive.sum()
    total = pos_loss.sum() + neg_loss.sum()
    if float(num_pos) > 0.0:
        return total / num_pos
    return neg_loss.sum()


def coordinate_loss(
    logits: torch.Tensor,
    keypoints_input: torch.Tensor,
    valid: torch.Tensor,
    input_size: tuple[int, int],
) -> torch.Tensor:
    """Differentiable coordinate loss from a spatial softmax over logits.

    Coordinates are normalized before Smooth-L1 so x/y and different input
    resolutions have comparable scale. This complements the sparse focal
    target and prevents the all-background solution.
    """
    if logits.dim() != 4 or keypoints_input.shape != (*logits.shape[:2], 2):
        raise ValueError(
            f"expected logits [B,C,H,W] and points [B,C,2], got "
            f"{tuple(logits.shape)} and {tuple(keypoints_input.shape)}"
        )
    batch, channels, height, width = logits.shape
    spatial = torch.softmax(logits.reshape(batch, channels, -1), dim=-1)
    xs = torch.arange(width, device=logits.device, dtype=logits.dtype).repeat(height)
    ys = torch.arange(height, device=logits.device, dtype=logits.dtype).repeat_interleave(width)
    predicted = torch.stack(((spatial * xs).sum(-1), (spatial * ys).sum(-1)), dim=-1)

    input_h, input_w = input_size
    target = keypoints_input.to(logits.dtype).clone()
    target[..., 0] *= width / input_w
    target[..., 1] *= height / input_h
    norm = logits.new_tensor([width, height])
    per_corner = F.smooth_l1_loss(predicted / norm, target / norm, reduction="none").sum(-1)
    weight = valid.to(logits.dtype)
    return (per_corner * weight).sum() / weight.sum().clamp_min(1.0)
