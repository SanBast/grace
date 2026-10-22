"""The training and evaluation dataset.

Each item holds a ground truth mask, a degraded copy of that mask, the image,
and one retrieved exemplar mask. The refiner reads the degraded mask, the
exemplar and the image. It predicts the ground truth.

Determinism depends on `seed`:

    seed=None     a new degradation for each epoch. Use this to train.
    seed=<int>    the same degradation for a given index. Use this to evaluate.

An evaluation with a fixed seed gives the same degraded mask on every run. Two
checkpoints then compare on identical inputs.

Every item holds an `exemplar` tensor. The tensor holds zeros if retrieval
finds no candidate, and `has_exemplar` is then 0.0. The shapes stay constant,
so the default PyTorch collate function works.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from grace.corruption import PRESETS, TRAINING_PRESETS, corrupt_mask, sample_severity
from grace.data.manifest import Manifest
from grace.data.volumes import load_case, load_case_cached
from grace.retrieval.index import ExemplarIndex

__all__ = ["RefinementDataset"]

# A large prime. It separates the streams of two adjacent sample indices.
_SEED_STRIDE = 1_000_003


class RefinementDataset(Dataset):
    """Pairs of degraded and ground truth masks, with a retrieved exemplar."""

    def __init__(self, manifest: Manifest,
                 index: ExemplarIndex | None = None,
                 size: tuple[int, int, int] = (96, 96, 96),
                 severity: str = "mixed",
                 exemplar_mode: str = "random",
                 use_image: bool = True,
                 seed: int | None = None,
                 cache_exemplars: bool = True):
        """
        Args:
            manifest: the cases to serve.
            index: the exemplar pool. Pass None to serve zero exemplars.
            size: the target volume shape.
            severity: a key of `corruption.PRESETS`, a key of
                `corruption.TRAINING_PRESETS`, or "clean".
            exemplar_mode: one of `retrieval.index.MODES`.
            use_image: load the image. Pass False to train on masks only.
            seed: None for random degradation, or an integer for a
                reproducible degradation.
            cache_exemplars: hold decoded exemplar masks in memory.
        """
        if severity not in PRESETS and severity not in TRAINING_PRESETS and severity != "clean":
            raise KeyError(
                f"unknown severity {severity!r}. Expected a preset "
                f"{sorted(PRESETS)}, a schedule {sorted(TRAINING_PRESETS)}, "
                "or 'clean'."
            )
        self.manifest = manifest
        self.index = index
        self.size = tuple(size)
        self.severity = severity
        self.exemplar_mode = exemplar_mode if index is not None else "none"
        self.use_image = use_image
        self.seed = seed
        self.cache_exemplars = cache_exemplars

    def __len__(self) -> int:
        return len(self.manifest)

    def _rng(self, position: int) -> np.random.Generator:
        """Give the generator for one item."""
        if self.seed is None:
            return np.random.default_rng()
        return np.random.default_rng(self.seed * _SEED_STRIDE + position)

    def _severity_for(self, rng) -> str:
        """Resolve the severity of one item. A schedule draws a new severity."""
        if self.severity in TRAINING_PRESETS:
            return sample_severity(self.severity, rng)
        return self.severity

    def _load_exemplar(self, position: int, rng) -> tuple[np.ndarray, str]:
        """Give the exemplar mask and the uid of its source."""
        empty = np.zeros((1,) + self.size, dtype=np.float32)
        if self.index is None:
            return empty, ""
        choice = self.index.pick(self.manifest[position], self.exemplar_mode, rng)
        if choice is None:
            return empty, ""
        loader = load_case_cached if self.cache_exemplars else load_case
        _, mask = loader(self.index.pool.image_path(choice),
                         self.index.pool.mask_path(choice), self.size)
        return np.array(mask, dtype=np.float32), self.index.pool[choice].uid

    def __getitem__(self, position: int) -> dict:
        entry = self.manifest[position]
        rng = self._rng(position)
        image, mask = load_case(self.manifest.image_path(position),
                                self.manifest.mask_path(position), self.size)

        severity = self._severity_for(rng)
        corrupted = corrupt_mask(mask[0], severity=severity, rng=rng)[None]
        exemplar, exemplar_uid = self._load_exemplar(position, rng)

        if not self.use_image:
            image = np.zeros_like(image)

        return {
            "image": torch.from_numpy(image),
            "mask": torch.from_numpy(mask),
            "corrupted": torch.from_numpy(corrupted),
            "exemplar": torch.from_numpy(exemplar),
            "has_exemplar": torch.tensor(1.0 if exemplar_uid else 0.0),
            "case_id": entry.case_id,
            "uid": entry.uid,
            "dataset": entry.dataset,
            "anatomy": entry.anatomy,
            "modality": entry.modality,
            "class_key": entry.class_key,
            "severity": severity,
            "exemplar_uid": exemplar_uid,
        }
