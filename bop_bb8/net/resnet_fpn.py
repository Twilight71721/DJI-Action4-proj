"""ResNet + FPN heatmap network for the 8 ordered bounding-box corners.

Design notes (adapted from BoxDreamer's BETR heatmap head, not its ResNet
wrapper):

- The backbone is a torchvision ImageNet-pretrained ResNet-18/34/50.  Following
  the FPN recipe, ``layer1..layer4`` are tapped at strides 4/8/16/32 and fused
  top-down, so the output stays at **stride 4** (e.g. 120x160 for 480x640).
  BoxDreamer's ``ResNetWrapper`` only hooks ``layer4`` (7x7 tokens for a 224
  input), which is too coarse for corner localisation and was never wired to
  its decoder end-to-end.
- ``forward`` returns raw **logits**.  The losses in ``bop_bb8.net.loss`` and
  the CornerNet focal loss operate on logits (``logsigmoid``), because
  clamping probabilities before a log-based loss creates a dead zone: once
  every prediction saturates below the clamp, gradients vanish and training
  stalls at ``-log(eps)`` per positive pixel. Legacy objectives use sigmoid;
  the spatial-KL objective instead uses a per-corner spatial softmax.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    resnet18,
    resnet34,
    resnet50,
)

_BUILDERS = {
    "resnet18": (resnet18, ResNet18_Weights.IMAGENET1K_V1, (64, 128, 256, 512)),
    "resnet34": (resnet34, ResNet34_Weights.IMAGENET1K_V1, (64, 128, 256, 512)),
    "resnet50": (resnet50, ResNet50_Weights.IMAGENET1K_V2, (256, 512, 1024, 2048)),
}


class ResNetFPN(nn.Module):
    """ResNet backbone + top-down FPN + 1x1 heatmap head.

    Input:  ``[B, 3, H, W]`` RGB, ImageNet-normalised.
    Output: ``[B, num_corners, H // 4 + 2m, W // 4 + 2m]`` raw logits for
    input dimensions divisible by 4, with optional output-only margin ``m``.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        fpn_dim: int = 128,
        num_corners: int = 8,
        pretrained: bool = True,
        freeze_bn: bool = False,
        heatmap_margin: int = 0,
    ) -> None:
        super().__init__()
        if backbone not in _BUILDERS:
            raise ValueError(
                f"unsupported backbone {backbone!r}; expected one of {sorted(_BUILDERS)}"
            )
        builder, weights, channels = _BUILDERS[backbone]
        net = builder(weights=weights if pretrained else None)
        net.fc = nn.Identity()
        net.avgpool = nn.Identity()
        self.backbone_name = backbone
        self.backbone = net
        self.num_corners = int(num_corners)
        self.fpn_dim = int(fpn_dim)
        if not isinstance(heatmap_margin, int) or heatmap_margin < 0:
            raise ValueError('heatmap_margin must be an integer >= 0')
        self.heatmap_margin = heatmap_margin

        # Lateral 1x1 taps, indexed like layer1..layer4 (strides 4/8/16/32).
        self.lateral1 = nn.Conv2d(channels[0], fpn_dim, kernel_size=1)
        self.lateral2 = nn.Conv2d(channels[1], fpn_dim, kernel_size=1)
        self.lateral3 = nn.Conv2d(channels[2], fpn_dim, kernel_size=1)
        self.lateral4 = nn.Conv2d(channels[3], fpn_dim, kernel_size=1)
        self.smooth = nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, padding=1)
        self.head = nn.Sequential(
            nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(fpn_dim, num_corners, kernel_size=1),
        )
        # A sparse keypoint map should start near background, not at p=0.5.
        # This is the standard CornerNet/CenterNet prior (sigmoid(-2.19) ~= 0.1).
        nn.init.constant_(self.head[-1].bias, -2.19)
        if heatmap_margin:
            # Output-only extension: the RGB image is never padded. Global
            # context and coordinates allow virtual corners beyond the ROI.
            self.context_proj = nn.Conv2d(fpn_dim, fpn_dim, kernel_size=1)
            self.coordinate_proj = nn.Conv2d(2, fpn_dim, kernel_size=1, bias=False)
        self._backbone_frozen = False
        self._freeze_bn = bool(freeze_bn)
        if self._freeze_bn:
            self.train(self.training)

    def extract_features(self, images: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Return ``(c1, c2, c3, c4)`` at strides 4/8/16/32."""
        x = self.backbone.conv1(images)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        c1 = self.backbone.layer1(x)
        c2 = self.backbone.layer2(c1)
        c3 = self.backbone.layer3(c2)
        c4 = self.backbone.layer4(c3)
        return c1, c2, c3, c4

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        c1, c2, c3, c4 = self.extract_features(images)
        p = self.lateral4(c4)
        p = self.lateral3(c3) + F.interpolate(p, size=c3.shape[-2:], mode="nearest")
        p = self.lateral2(c2) + F.interpolate(p, size=c2.shape[-2:], mode="nearest")
        p = self.lateral1(c1) + F.interpolate(p, size=c1.shape[-2:], mode="nearest")
        p = F.relu(self.smooth(p), inplace=True)
        if self.heatmap_margin:
            height, width = p.shape[-2:]
            margin = self.heatmap_margin
            context = self.context_proj(F.adaptive_avg_pool2d(p, 1))
            p = F.pad(p, (margin, margin, margin, margin), mode='replicate')
            ys = torch.arange(-margin, height+margin, device=p.device, dtype=p.dtype)
            xs = torch.arange(-margin, width+margin, device=p.device, dtype=p.dtype)
            yy, xx = torch.meshgrid(ys / max(height-1, 1), xs / max(width-1, 1), indexing='ij')
            coordinates = torch.stack((xx*2-1, yy*2-1))[None]
            p = p + context + self.coordinate_proj(coordinates)
        return self.head(p)

    def freeze_backbone(self, freeze: bool = True) -> None:
        """Toggle backbone gradients; a frozen backbone also stays in eval mode.

        ``nn.Module.train`` is overridden so BatchNorm running statistics are
        not updated while the backbone is frozen, even after ``model.train()``.
        """
        self._backbone_frozen = bool(freeze)
        for param in self.backbone.parameters():
            param.requires_grad = not freeze
        if freeze:
            self.backbone.eval()

    def train(self, mode: bool = True) -> "ResNetFPN":
        super().train(mode)
        if self._backbone_frozen:
            self.backbone.eval()
        elif self._freeze_bn:
            # Keep pretrained running statistics; affine weights still learn.
            for module in self.backbone.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self
