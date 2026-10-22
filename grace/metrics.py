"""Metrics for binary 3D masks.

The package reports three metrics:

    dice            volumetric overlap
    hausdorff_95    95th percentile surface distance, in voxels
    boundary_f1     surface agreement within a tolerance

Report hausdorff_95 together with Dice. A refiner can add a false positive blob
far from the organ. Dice then decreases by a small amount, and the surface
distance increases by a large amount.

Empty masks:

    both masks empty      Dice 1.0, distances 0.0
    one mask empty        Dice 0.0, distances NaN
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt

__all__ = ["dice", "hausdorff_95", "boundary_f1", "dice_batch", "surface_of"]


def _as_bool(mask) -> np.ndarray:
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    return np.asarray(mask) > 0.5


def dice(pred, target) -> float:
    """Give the Dice similarity coefficient of two binary masks."""
    a, b = _as_bool(pred), _as_bool(target)
    total = a.sum() + b.sum()
    if total == 0:
        return 1.0
    return float(2.0 * np.logical_and(a, b).sum() / total)


def surface_of(mask) -> np.ndarray:
    """Give the boundary voxels of a binary mask."""
    mask = _as_bool(mask)
    if not mask.any():
        return np.zeros_like(mask)
    return mask & ~binary_erosion(mask, iterations=1)


def _surface_distances(pred, target):
    """Give the distances between the two surfaces, in both directions.

    Returns (None, None) if one mask is empty.
    """
    a, b = _as_bool(pred), _as_bool(target)
    if not a.any() or not b.any():
        return None, None
    surface_a, surface_b = surface_of(a), surface_of(b)
    # distance_transform_edt measures the distance to the nearest zero.
    # Invert the surface masks to get the distance to the surface.
    to_b = distance_transform_edt(~surface_b)
    to_a = distance_transform_edt(~surface_a)
    return to_b[surface_a], to_a[surface_b]


def hausdorff_95(pred, target) -> float:
    """Give the 95th percentile symmetric surface distance, in voxels.

    The 95th percentile limits the effect of single stray voxels. The maximum
    distance would follow those voxels for every case.
    """
    forward, backward = _surface_distances(pred, target)
    if forward is None:
        a, b = _as_bool(pred), _as_bool(target)
        return 0.0 if (not a.any() and not b.any()) else float("nan")
    return float(max(np.percentile(forward, 95), np.percentile(backward, 95)))


def boundary_f1(pred, target, tolerance: float = 2.0) -> float:
    """Give the F1 score of the surface agreement within `tolerance` voxels."""
    forward, backward = _surface_distances(pred, target)
    if forward is None:
        a, b = _as_bool(pred), _as_bool(target)
        return 1.0 if (not a.any() and not b.any()) else 0.0
    precision = float((forward <= tolerance).mean())
    recall = float((backward <= tolerance).mean())
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def dice_batch(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Give the Dice of each sample in a batch of binary tensors, shaped (B, ...).

    This function keeps the data on the device. The training loop therefore
    avoids one host transfer for each sample.
    """
    flat_pred = pred.reshape(pred.shape[0], -1).float()
    flat_target = target.reshape(target.shape[0], -1).float()
    overlap = (flat_pred * flat_target).sum(dim=-1)
    total = flat_pred.sum(dim=-1) + flat_target.sum(dim=-1) + eps
    return 2.0 * overlap / total
