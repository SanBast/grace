"""Fusion of the degraded mask, the exemplar and the image.

A fusion module combines the three inputs into one tensor for the U-Net. Two
variants are available:

    concat    stack the channels, then mix them with a 1x1x1 convolution
    film      pool the exemplar, then scale and shift the degraded mask

Both variants start as an identity on the degraded mask. The convolution of
`concat` starts at the identity matrix. The last layer of the FiLM network
starts at zero. Training therefore starts from a pass-through, and the exemplar
path grows from there.

`concat` is the default.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["ConcatFusion", "FiLMFusion", "build_fusion", "FUSIONS"]


class ConcatFusion(nn.Module):
    """Stack the channels and mix them with a 1x1x1 convolution."""

    def __init__(self, use_exemplar: bool = True, use_image: bool = True):
        super().__init__()
        self.use_exemplar = use_exemplar
        self.use_image = use_image
        self.out_channels = 1 + int(use_exemplar) + int(use_image)
        self.mix = nn.Conv3d(self.out_channels, self.out_channels, kernel_size=1)
        with torch.no_grad():
            self.mix.weight.zero_()
            self.mix.weight[:, :, 0, 0, 0] = torch.eye(self.out_channels)
            self.mix.bias.zero_()

    def forward(self, corrupted: torch.Tensor,
                exemplar: torch.Tensor | None = None,
                image: torch.Tensor | None = None) -> torch.Tensor:
        parts = [corrupted]
        if self.use_exemplar:
            parts.append(exemplar if exemplar is not None else torch.zeros_like(corrupted))
        if self.use_image:
            parts.append(image if image is not None else torch.zeros_like(corrupted))
        return self.mix(torch.cat(parts, dim=1))


class FiLMFusion(nn.Module):
    """Scale and shift the degraded mask with statistics of the exemplar.

    The module pools the exemplar and the image to one value each. A small
    network maps these values to a gain and an offset. The output also holds
    the unmodified exemplar and image channels.
    """

    def __init__(self, use_exemplar: bool = True, use_image: bool = True,
                 hidden: int = 32):
        super().__init__()
        self.use_exemplar = use_exemplar
        self.use_image = use_image
        self.out_channels = 1 + int(use_exemplar) + int(use_image)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.net = nn.Sequential(
            nn.Linear(1 + int(use_image), hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2),
        )
        with torch.no_grad():
            self.net[-1].weight.zero_()
            self.net[-1].bias.zero_()

    def forward(self, corrupted: torch.Tensor,
                exemplar: torch.Tensor | None = None,
                image: torch.Tensor | None = None) -> torch.Tensor:
        if exemplar is None:
            exemplar = torch.zeros_like(corrupted)
        if image is None:
            image = torch.zeros_like(corrupted)

        summary = [self.pool(exemplar).flatten(1)]
        if self.use_image:
            summary.append(self.pool(image).flatten(1))
        gain, offset = self.net(torch.cat(summary, dim=1)).chunk(2, dim=1)
        gain = (1.0 + torch.tanh(gain)).view(-1, 1, 1, 1, 1)
        offset = offset.view(-1, 1, 1, 1, 1)

        parts = [gain * corrupted + offset]
        if self.use_exemplar:
            parts.append(exemplar)
        if self.use_image:
            parts.append(image)
        return torch.cat(parts, dim=1)


FUSIONS = {"concat": ConcatFusion, "film": FiLMFusion}


def build_fusion(name: str, use_exemplar: bool = True,
                 use_image: bool = True) -> nn.Module:
    """Make a fusion module by name."""
    if name not in FUSIONS:
        raise ValueError(f"unknown fusion {name!r}, expected one of {sorted(FUSIONS)}")
    return FUSIONS[name](use_exemplar=use_exemplar, use_image=use_image)
