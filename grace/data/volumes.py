"""NIfTI input and resampling.

All volumes go to a cube of fixed size, 96 voxels on each side by default. One
refiner can then process a hippocampus and a liver in the same batch.

Intensity normalisation uses a window from the 1st to the 99th percentile. The
same window applies to CT, MRI and ultrasound.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy.ndimage import zoom

__all__ = ["read_nifti", "resize_volume", "normalise_intensity", "load_case",
           "load_case_cached"]


def read_nifti(path: str | Path) -> np.ndarray:
    """Load a NIfTI file as a float32 array. Remove any axis after the third."""
    volume = np.asarray(nib.load(str(path)).dataobj, dtype=np.float32)
    while volume.ndim > 3:
        volume = volume[..., 0]
    return volume


def resize_volume(volume: np.ndarray, size: tuple[int, int, int],
                  order: int) -> np.ndarray:
    """Resample a (D, H, W) volume to `size`.

    Args:
        volume: the input volume.
        size: the target shape.
        order: the spline order. Use 0 for a mask and 1 for an image.
    """
    if volume.shape == tuple(size):
        return volume.astype(np.float32)
    factors = [target / current for target, current in zip(size, volume.shape)]
    return zoom(volume, factors, order=order).astype(np.float32)


def normalise_intensity(volume: np.ndarray) -> np.ndarray:
    """Map a volume to [0, 1] with its 1st and 99th intensity percentiles.

    The output is float32. `numpy.percentile` gives float64 values, and the
    arithmetic would otherwise promote the volume to float64.
    """
    if volume.size == 0:
        return volume.astype(np.float32)
    low, high = np.percentile(volume, [1, 99])
    if high <= low:
        return np.zeros(volume.shape, dtype=np.float32)
    scaled = (volume - low) / (high - low)
    return np.clip(scaled, 0.0, 1.0).astype(np.float32)


def load_case(image_path: str | Path, mask_path: str | Path,
              size: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Load one case as (image, mask). Both arrays are (1, D, H, W) float32.

    The function binarises the mask after the resample. A nearest neighbour
    resample of a binary label therefore stays binary.
    """
    image = normalise_intensity(resize_volume(read_nifti(image_path), size, order=1))
    mask = (resize_volume(read_nifti(mask_path), size, order=0) > 0.5).astype(np.float32)
    return image[None], mask[None]


@lru_cache(maxsize=256)
def _load_case_cached(image_path: str, mask_path: str,
                      size: tuple[int, int, int]):
    return load_case(image_path, mask_path, size)


def load_case_cached(image_path: str | Path, mask_path: str | Path,
                     size: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Give the result of `load_case` from a memory cache.

    Exemplar retrieval reads the same few hundred masks many times. The cache
    holds 256 cases, approximately 1 GB at 96 cubed.
    """
    return _load_case_cached(str(image_path), str(mask_path), tuple(size))
