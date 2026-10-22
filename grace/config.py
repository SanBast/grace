"""Configurations for training and evaluation.

Both dataclasses convert to and from a dictionary. The trainer writes its
configuration next to the checkpoint. An evaluation run therefore records the
exact settings that produced a number.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

__all__ = ["TrainConfig", "EvalConfig"]


@dataclass
class TrainConfig:
    """Settings for one training run."""

    train_manifest: str
    val_manifest: str
    output_dir: str

    # Data
    size: tuple[int, int, int] = (96, 96, 96)
    severity: str = "mixed"
    val_severity: str = "strong"
    exemplar_mode: str = "random"
    nn_table: str | None = None
    use_image: bool = True
    exclude_datasets: list[str] = field(default_factory=list)
    max_val_cases: int = 200

    # Optimisation
    epochs: int = 100
    batch_size: int = 4
    num_workers: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    amp: bool = True
    seed: int = 42

    # Schedule
    eval_every: int = 2
    log_every: int = 50

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "TrainConfig":
        fields = set(cls.__dataclass_fields__)
        kept = {k: v for k, v in payload.items() if k in fields}
        if "size" in kept:
            kept["size"] = tuple(kept["size"])
        return cls(**kept)


@dataclass
class EvalConfig:
    """Settings for one evaluation run."""

    checkpoint: str
    manifest: str
    pool_manifest: str
    output: str

    size: tuple[int, int, int] = (96, 96, 96)
    severities: list[str] = field(default_factory=lambda: ["default", "mid", "strong"])
    exemplar_mode: str = "random"
    nn_table: str | None = None
    exclude_datasets: list[str] = field(default_factory=list)
    datasets: list[str] = field(default_factory=list)
    anatomies: list[str] = field(default_factory=list)
    max_cases: int = 200
    batch_size: int = 4
    num_workers: int = 2
    boundary_metrics: bool = True
    seed: int = 42

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "EvalConfig":
        fields = set(cls.__dataclass_fields__)
        kept = {k: v for k, v in payload.items() if k in fields}
        if "size" in kept:
            kept["size"] = tuple(kept["size"])
        return cls(**kept)
