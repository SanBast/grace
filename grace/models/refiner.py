"""The refiner: a fusion module and a 3D U-Net.

The model takes a degraded mask, a retrieved exemplar mask and an image. It
gives two logits for each voxel, for background and foreground.

Exemplar dropout sets the exemplar to zero for a fraction of the samples during
training. The model therefore also works when retrieval returns no case.

A checkpoint holds the configuration and the weights in one file. Use
`GraceRefiner.load` to restore both.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from monai.networks.nets import UNet

from grace.models.fusion import build_fusion

__all__ = ["RefinerConfig", "GraceRefiner"]


@dataclass
class RefinerConfig:
    """The architecture of a refiner."""

    use_exemplar: bool = True
    use_image: bool = True
    fusion: str = "concat"
    channels: tuple[int, ...] = (32, 64, 128, 256, 320)
    strides: tuple[int, ...] = (2, 2, 2, 2)
    num_res_units: int = 2
    norm: str = "INSTANCE"
    dropout: float = 0.0
    exemplar_dropout: float = 0.10

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "RefinerConfig":
        """Make a config from a dictionary. Unknown keys are ignored."""
        fields = {f for f in cls.__dataclass_fields__}
        kept = {k: v for k, v in payload.items() if k in fields}
        for key in ("channels", "strides"):
            if key in kept:
                kept[key] = tuple(kept[key])
        return cls(**kept)


class GraceRefiner(nn.Module):
    """A retrieval-conditioned mask refiner."""

    def __init__(self, config: RefinerConfig | None = None):
        super().__init__()
        self.config = config or RefinerConfig()
        self.fusion = build_fusion(self.config.fusion,
                                   use_exemplar=self.config.use_exemplar,
                                   use_image=self.config.use_image)
        self.unet = UNet(
            spatial_dims=3,
            in_channels=self.fusion.out_channels,
            out_channels=2,
            channels=tuple(self.config.channels),
            strides=tuple(self.config.strides),
            num_res_units=self.config.num_res_units,
            norm=self.config.norm,
            dropout=self.config.dropout,
        )

    def forward(self, corrupted: torch.Tensor,
                exemplar: torch.Tensor | None = None,
                image: torch.Tensor | None = None) -> torch.Tensor:
        """Give the voxel logits, shaped (B, 2, D, H, W).

        During training the module applies exemplar dropout.
        """
        if exemplar is not None and self.training and self.config.exemplar_dropout > 0:
            keep = (torch.rand(exemplar.shape[0], 1, 1, 1, 1, device=exemplar.device)
                    > self.config.exemplar_dropout).float()
            exemplar = exemplar * keep
        return self.unet(self.fusion(corrupted, exemplar=exemplar, image=image))

    @torch.no_grad()
    def refine(self, corrupted: torch.Tensor,
               exemplar: torch.Tensor | None = None,
               image: torch.Tensor | None = None) -> torch.Tensor:
        """Give a binary mask, shaped (B, 1, D, H, W)."""
        self.eval()
        return self.forward(corrupted, exemplar, image).argmax(dim=1, keepdim=True).float()

    def save(self, path: str | Path, extra: dict | None = None) -> Path:
        """Write the configuration and the weights to one file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"config": self.config.to_dict(), "state_dict": self.state_dict()}
        if extra:
            payload.update(extra)
        torch.save(payload, path)
        return path

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cpu") -> "GraceRefiner":
        """Read a checkpoint that `save` wrote."""
        payload = torch.load(path, map_location=device, weights_only=False)
        model = cls(RefinerConfig.from_dict(payload["config"]))
        model.load_state_dict(payload["state_dict"], strict=True)
        return model.to(device)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
