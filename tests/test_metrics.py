"""Tests for the metrics."""

from __future__ import annotations

import numpy as np
import torch

from grace.metrics import boundary_f1, dice, dice_batch, hausdorff_95


def test_identical_masks(small_mask):
    assert dice(small_mask, small_mask) == 1.0
    assert hausdorff_95(small_mask, small_mask) == 0.0
    assert boundary_f1(small_mask, small_mask) == 1.0


def test_disjoint_masks():
    a = np.zeros((16, 16, 16), dtype=np.float32)
    b = np.zeros((16, 16, 16), dtype=np.float32)
    a[2:5, 2:5, 2:5] = 1
    b[10:13, 10:13, 10:13] = 1
    assert dice(a, b) == 0.0
    assert hausdorff_95(a, b) > 5.0


def test_two_empty_masks_match():
    empty = np.zeros((8, 8, 8), dtype=np.float32)
    assert dice(empty, empty) == 1.0
    assert hausdorff_95(empty, empty) == 0.0


def test_one_empty_mask_is_a_total_miss(small_mask):
    empty = np.zeros_like(small_mask)
    assert dice(empty, small_mask) == 0.0
    assert np.isnan(hausdorff_95(empty, small_mask))


def test_dice_falls_as_the_overlap_falls(small_mask):
    shifted = np.roll(small_mask, 3, axis=0)
    far = np.roll(small_mask, 7, axis=0)
    assert dice(small_mask, small_mask) > dice(shifted, small_mask)
    assert dice(shifted, small_mask) > dice(far, small_mask)


def test_batch_dice_matches_the_single_version(small_mask):
    batch = torch.from_numpy(np.stack([small_mask, np.roll(small_mask, 3, axis=0)]))
    truth = torch.from_numpy(np.stack([small_mask, small_mask]))
    scores = dice_batch(batch, truth).numpy()
    assert np.isclose(scores[0], 1.0, atol=1e-5)
    assert np.isclose(scores[1], dice(np.roll(small_mask, 3, axis=0), small_mask),
                      atol=1e-4)
