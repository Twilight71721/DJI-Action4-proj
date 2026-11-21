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
    margin: int = 0,
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
    if margin < 0 or min(height, width) <= 2*margin:
        raise ValueError('Invalid output heatmap margin')
    target[..., 0] *= (width-2*margin) / input_w
    target[..., 1] *= (height-2*margin) / input_h
    target += margin
    norm = logits.new_tensor([width, height])
    per_corner = F.smooth_l1_loss(predicted / norm, target / norm, reduction="none").sum(-1)
    weight = valid.to(logits.dtype)
    return (per_corner * weight).sum() / weight.sum().clamp_min(1.0)


def spatial_kl_loss(logits: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Match one normalized spatial distribution for each ordered corner.

    Unlike independent pixel classification, adding a constant to a channel's
    logits cannot reduce this loss. Empty/invalid targets never become fake
    border labels. FP32 log-softmax keeps the calculation stable under AMP.
    """
    if logits.shape != targets.shape or logits.dim() != 4:
        raise ValueError('Expected matching [B,C,H,W] logits and targets')
    _valid_mask(valid, logits)
    target = targets.float().flatten(2)
    selected = valid.bool()
    target = torch.where(selected.unsqueeze(-1), target, torch.zeros_like(target))
    if not torch.isfinite(target).all() or (target < 0).any():
        raise ValueError('Valid spatial targets must be finite and nonnegative')
    mass = target.sum(-1, keepdim=True)
    if (selected & (mass.squeeze(-1) <= 0)).any():
        raise ValueError('A supervised corner must have nonempty heatmap support')
    target = target / mass.clamp_min(torch.finfo(torch.float32).tiny)
    log_prob = F.log_softmax(logits.float().flatten(2), dim=-1)
    terms = F.kl_div(log_prob, target, reduction='none').sum(-1)
    return torch.where(selected, terms, torch.zeros_like(terms)).sum() / selected.sum().clamp_min(1)


def spatial_js_loss(first: torch.Tensor, second: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Symmetric JS between corresponding corner distributions; both views receive gradients."""
    if first.shape != second.shape or first.dim() != 4:
        raise ValueError('Expected matching [B,C,H,W] logits')
    _valid_mask(valid, first)
    selected = valid.bool()
    a = F.log_softmax(first.float().flatten(2)[selected], dim=-1)
    b = F.log_softmax(second.float().flatten(2)[selected], dim=-1)
    if a.shape[0] == 0:
        return a.sum() + b.sum()
    log_middle = torch.logaddexp(a, b) - 0.6931471805599453
    return .5 * ((a.exp() * (a-log_middle)).sum(-1) +
                 (b.exp() * (b-log_middle)).sum(-1)).mean()
