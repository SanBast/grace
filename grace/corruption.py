"""Synthetic degradation of binary masks.

The refiner learns to correct a damaged mask. Real pairs of bad prediction and
ground truth are rare, so this module makes the pairs synthetically.

Each operation works on the signed distance transform. The damage therefore has
smooth boundaries, and it resembles the output of an undertrained network.

A preset holds the probabilities and magnitude ranges for one severity. Five
presets are available, in order of increasing damage:

    default, mid, strong, very_strong, extreme

Use extreme for evaluation only. No training schedule includes it. At that
severity the mask keeps little of the original shape, so it probes the
behaviour of the refiner at the low Dice end.

All functions take a numpy Generator. No function reads global random state, so
the same sample index and seed always give the same mask.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, label

__all__ = [
    "PRESETS",
    "TRAINING_PRESETS",
    "signed_distance",
    "corrupt_mask",
    "sample_severity",
]


# ---------------------------------------------------------------------------
# Primitives. Each one takes a binary (D, H, W) array and returns the same
# shape.
# ---------------------------------------------------------------------------


def signed_distance(mask: np.ndarray) -> np.ndarray:
    """Give the distance to the boundary. The values inside the mask are positive."""
    inside = mask.astype(bool)
    if not inside.any():
        return -distance_transform_edt(~inside)
    if inside.all():
        return distance_transform_edt(inside)
    return distance_transform_edt(inside) - distance_transform_edt(~inside)


def elastic_boundary(mask: np.ndarray, rng, sigma: float, magnitude: float) -> np.ndarray:
    """Move the boundary in and out with a smooth random field."""
    noise = gaussian_filter(rng.standard_normal(mask.shape) * magnitude, sigma=sigma)
    return ((signed_distance(mask) + noise) > 0).astype(mask.dtype)


def boundary_jitter(mask: np.ndarray, rng, noise_std: float, smooth_sigma: float) -> np.ndarray:
    """Move the boundary with a longer correlation length than elastic_boundary.

    Both operations together give error at two spatial scales. A weak network
    makes this type of error: a wrong outline plus local roughness.
    """
    noise = gaussian_filter(rng.standard_normal(mask.shape) * noise_std, sigma=smooth_sigma)
    return ((signed_distance(mask) + noise) > 0).astype(mask.dtype)


def shrink(mask: np.ndarray, fraction: float) -> np.ndarray:
    """Erode the mask by a fraction of its own thickness.

    The distance transform scales the severity with the object. A hippocampus
    and a liver then receive comparable damage.
    """
    if not mask.any():
        return mask.copy()
    distance = distance_transform_edt(mask.astype(bool))
    peak = distance.max()
    if peak == 0:
        return mask.copy()
    return (distance > peak * fraction).astype(mask.dtype)


def grow(mask: np.ndarray, rng, voxels: float, smooth_sigma: float = 2.0) -> np.ndarray:
    """Dilate the mask by `voxels`, then roughen the new boundary."""
    if not mask.any():
        return mask.copy()
    background = distance_transform_edt(~mask.astype(bool))
    grown = background <= voxels
    if smooth_sigma <= 0:
        return grown.astype(mask.dtype)
    noise = gaussian_filter(rng.standard_normal(mask.shape), sigma=smooth_sigma)
    shell = (background > 0) & (background < voxels * 2)
    sdf = signed_distance(grown)
    sdf[shell] += noise[shell] * 1.5
    return (sdf > 0).astype(mask.dtype)


def false_positive_blob(mask: np.ndarray, rng, count: int, size_fraction: float) -> np.ndarray:
    """Add smooth blobs near the object.

    A network can give a neighbouring structure the label of the target. The
    blobs stay within a few voxels of the object. Their size scales with the
    bounding box of the object.
    """
    out = mask.copy()
    if not mask.any():
        return out
    coords = np.nonzero(mask)
    low = np.array([c.min() for c in coords])
    high = np.array([c.max() for c in coords])
    extent = high - low + 1
    background = distance_transform_edt(~mask.astype(bool))

    for _ in range(count):
        near = (background > 3) & (background < max(5, int(extent.mean() * 0.4)))
        for axis in range(3):
            margin = int(extent[axis] * 0.6)
            window = np.zeros(mask.shape, dtype=bool)
            index = [slice(None)] * 3
            index[axis] = slice(max(0, low[axis] - margin),
                                min(mask.shape[axis], high[axis] + margin))
            window[tuple(index)] = True
            near &= window
        if not near.any():
            continue

        candidates = np.nonzero(near)
        pick = int(rng.integers(len(candidates[0])))
        centre = np.array([candidates[axis][pick] for axis in range(3)])
        radius = np.array([
            max(2, int(side * size_fraction * rng.uniform(0.4, 1.2)))
            for side in extent
        ])

        patch_shape = radius * 3
        field = gaussian_filter(rng.standard_normal(tuple(patch_shape)),
                                sigma=radius * 0.4)
        blob = field > np.percentile(field, 65)
        # Keep the largest connected component. The blob is then one object.
        components, n_components = label(blob)
        if n_components > 1:
            sizes = [int((components == i).sum()) for i in range(1, n_components + 1)]
            blob = components == (int(np.argmax(sizes)) + 1)

        target, source = [], []
        for axis in range(3):
            start = int(centre[axis] - patch_shape[axis] // 2)
            stop = start + int(patch_shape[axis])
            clipped_start = max(0, start)
            clipped_stop = min(mask.shape[axis], stop)
            target.append(slice(clipped_start, clipped_stop))
            offset = clipped_start - start
            source.append(slice(offset, offset + clipped_stop - clipped_start))
        out[tuple(target)] = np.maximum(out[tuple(target)],
                                        blob[tuple(source)].astype(out.dtype))
    return out


def split_object(mask: np.ndarray, rng, smooth_sigma: float = 6.0) -> np.ndarray:
    """Remove approximately half of the object along a smooth random surface.

    A network can find one lobe of an organ and miss the remainder.
    """
    if not mask.any():
        return mask.copy()
    field = gaussian_filter(rng.standard_normal(mask.shape), sigma=smooth_sigma)
    threshold = float(np.median(field[mask.astype(bool)]))
    keep = field > threshold if rng.random() < 0.5 else field < threshold
    out = mask.copy()
    out[~keep] = 0
    return out


def drop_slices(mask: np.ndarray, rng, fraction: float) -> np.ndarray:
    """Set a continuous run of axial slices to zero.

    A 2D segmentor fails this way at the first and last slices of an organ. The
    cross section there is small and ambiguous.
    """
    if not mask.any():
        return mask.copy()
    out = mask.copy()
    occupied = np.nonzero(mask.any(axis=(0, 1)))[0]
    if len(occupied) < 3:
        return out
    first, last = int(occupied.min()), int(occupied.max())
    width = max(1, int((last - first) * fraction))
    if rng.random() < 0.5:
        out[:, :, last - width + 1:last + 1] = 0
    else:
        out[:, :, first:first + width] = 0
    return out


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

DEFAULT = {
    "elastic_prob": 0.50, "elastic_sigma": (5.0, 12.0), "elastic_magnitude": (2.0, 6.0),
    "jitter_prob": 0.40, "jitter_std": (1.0, 3.0), "jitter_sigma": (6.0, 14.0),
    "shrink_prob": 0.25, "shrink_fraction": (0.05, 0.25),
    "grow_prob": 0.25, "grow_voxels": (1.5, 5.0),
    "blob_prob": 0.30, "blob_count": (1, 2), "blob_size": (0.10, 0.35),
    "split_prob": 0.10,
    "drop_slices_prob": 0.20, "drop_slices_fraction": (0.10, 0.35),
    "max_operations": 3,
}

STRONG = {
    "elastic_prob": 0.75, "elastic_sigma": (5.0, 11.0), "elastic_magnitude": (3.5, 9.0),
    "jitter_prob": 0.75, "jitter_std": (1.8, 4.0), "jitter_sigma": (6.0, 13.0),
    "shrink_prob": 0.40, "shrink_fraction": (0.10, 0.32),
    "grow_prob": 0.40, "grow_voxels": (2.0, 6.5),
    "blob_prob": 0.50, "blob_count": (1, 2), "blob_size": (0.12, 0.40),
    "split_prob": 0.20,
    "drop_slices_prob": 0.30, "drop_slices_fraction": (0.12, 0.35),
    "max_operations": 4,
}

EXTREME = {
    "elastic_prob": 0.95, "elastic_sigma": (4.0, 10.0), "elastic_magnitude": (6.0, 14.0),
    "jitter_prob": 0.95, "jitter_std": (3.5, 7.0), "jitter_sigma": (5.0, 12.0),
    "shrink_prob": 0.60, "shrink_fraction": (0.25, 0.60),
    "grow_prob": 0.60, "grow_voxels": (5.0, 11.0),
    "blob_prob": 0.80, "blob_count": (2, 5), "blob_size": (0.25, 0.60),
    "split_prob": 0.45,
    "drop_slices_prob": 0.60, "drop_slices_fraction": (0.25, 0.55),
    "max_operations": 6,
}


def _blend(low: dict, high: dict, alpha: float) -> dict:
    """Interpolate two presets. Scalars and range tuples blend element by element."""
    out = {}
    for key, left in low.items():
        right = high[key]
        if isinstance(left, tuple):
            out[key] = tuple(type(left[i])(left[i] + alpha * (right[i] - left[i]))
                             for i in range(len(left)))
        else:
            out[key] = type(left)(left + alpha * (right - left))
    return out


MID = _blend(DEFAULT, STRONG, 0.5)

VERY_STRONG = _blend(STRONG, {
    **STRONG,
    "shrink_fraction": (0.20, 0.55),
    "grow_voxels": (4.0, 9.0),
    "jitter_std": (2.5, 5.5),
    "blob_count": (2, 4),
    "blob_size": (0.20, 0.55),
    "drop_slices_fraction": (0.20, 0.50),
    "max_operations": 5,
}, 0.5)


PRESETS = {
    "default": DEFAULT,
    "mid": MID,
    "strong": STRONG,
    "very_strong": VERY_STRONG,
    "extreme": EXTREME,
}

# Severity mixtures for training. Each sample draws its own severity, so the
# model sees the full range. The severity "clean" passes the mask through. The
# model then also learns to keep a correct input.
TRAINING_PRESETS = {
    "mixed": [("clean", 0.05), ("default", 0.25), ("mid", 0.30), ("strong", 0.40)],
    "mixed_wide": [("clean", 0.08), ("default", 0.22), ("mid", 0.25),
                   ("strong", 0.30), ("very_strong", 0.15)],
}


def sample_severity(schedule: str, rng) -> str:
    """Draw one severity name from a mixture in TRAINING_PRESETS."""
    if schedule not in TRAINING_PRESETS:
        raise KeyError(
            f"unknown schedule {schedule!r}, expected one of {sorted(TRAINING_PRESETS)}"
        )
    draw = rng.random()
    total = 0.0
    for name, weight in TRAINING_PRESETS[schedule]:
        total += weight
        if draw < total:
            return name
    return TRAINING_PRESETS[schedule][-1][0]


# ---------------------------------------------------------------------------
# Composite
# ---------------------------------------------------------------------------


def corrupt_mask(mask: np.ndarray, severity: str = "default", rng=None) -> np.ndarray:
    """Degrade a binary (D, H, W) mask at the given severity.

    The function draws each operation independently. It applies at most
    `max_operations` of them in sequence. If it draws no operation, it applies
    boundary jitter.

    Args:
        mask: binary array. Values above 0.5 are foreground.
        severity: a key of PRESETS, or "clean" to pass the mask through.
        rng: numpy Generator. The function makes a new Generator if you omit
            this argument. Supply one for every result that you report.

    Returns:
        A float32 array of 0.0 and 1.0 with the shape of the input.
    """
    rng = np.random.default_rng() if rng is None else rng
    binary = (mask > 0.5).astype(np.float32)
    if severity == "clean" or not binary.any():
        return binary
    if severity not in PRESETS:
        raise KeyError(f"unknown severity {severity!r}, expected one of "
                       f"{sorted(PRESETS)} or 'clean'")
    cfg = PRESETS[severity]

    operations = []
    if rng.random() < cfg["elastic_prob"]:
        sigma = rng.uniform(*cfg["elastic_sigma"])
        magnitude = rng.uniform(*cfg["elastic_magnitude"])
        operations.append(lambda m: elastic_boundary(m, rng, sigma, magnitude))
    if rng.random() < cfg["jitter_prob"]:
        std = rng.uniform(*cfg["jitter_std"])
        sigma = rng.uniform(*cfg["jitter_sigma"])
        operations.append(lambda m: boundary_jitter(m, rng, std, sigma))
    if rng.random() < cfg["shrink_prob"]:
        fraction = rng.uniform(*cfg["shrink_fraction"])
        operations.append(lambda m: shrink(m, fraction))
    if rng.random() < cfg["grow_prob"]:
        voxels = rng.uniform(*cfg["grow_voxels"])
        operations.append(lambda m: grow(m, rng, voxels))
    if rng.random() < cfg["blob_prob"]:
        count = int(rng.integers(cfg["blob_count"][0], cfg["blob_count"][1] + 1))
        size = rng.uniform(*cfg["blob_size"])
        operations.append(lambda m: false_positive_blob(m, rng, count, size))
    if rng.random() < cfg["split_prob"]:
        operations.append(lambda m: split_object(m, rng))
    if rng.random() < cfg["drop_slices_prob"]:
        fraction = rng.uniform(*cfg["drop_slices_fraction"])
        operations.append(lambda m: drop_slices(m, rng, fraction))

    if not operations:
        std = rng.uniform(*cfg["jitter_std"])
        sigma = rng.uniform(*cfg["jitter_sigma"])
        operations.append(lambda m: boundary_jitter(m, rng, std, sigma))
    if len(operations) > cfg["max_operations"]:
        keep = rng.choice(len(operations), size=cfg["max_operations"], replace=False)
        operations = [operations[i] for i in sorted(keep)]

    out = binary
    for operation in operations:
        out = operation(out)
    return (out > 0.5).astype(np.float32)
