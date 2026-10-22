"""Tests for the synthetic degradation."""

from __future__ import annotations

import numpy as np

from grace.corruption import PRESETS, corrupt_mask, sample_severity
from grace.metrics import dice


def test_clean_returns_the_input(small_mask):
    out = corrupt_mask(small_mask, severity="clean",
                       rng=np.random.default_rng(0))
    assert dice(out, small_mask) == 1.0


def test_same_seed_gives_the_same_mask(small_mask):
    first = corrupt_mask(small_mask, "strong", np.random.default_rng(7))
    second = corrupt_mask(small_mask, "strong", np.random.default_rng(7))
    assert np.array_equal(first, second)


def test_different_seeds_give_different_masks(small_mask):
    first = corrupt_mask(small_mask, "strong", np.random.default_rng(1))
    second = corrupt_mask(small_mask, "strong", np.random.default_rng(2))
    assert not np.array_equal(first, second)


def test_output_is_binary_and_keeps_the_shape(small_mask):
    out = corrupt_mask(small_mask, "mid", np.random.default_rng(0))
    assert out.shape == small_mask.shape
    assert out.dtype == np.float32
    assert set(np.unique(out)).issubset({0.0, 1.0})


def test_severity_order_follows_the_presets(small_mask):
    """A higher severity must give a lower mean Dice."""
    means = []
    for severity in ["default", "mid", "strong", "very_strong", "extreme"]:
        scores = [
            dice(corrupt_mask(small_mask, severity, np.random.default_rng(seed)),
                 small_mask)
            for seed in range(12)
        ]
        means.append(float(np.mean(scores)))
    assert means == sorted(means, reverse=True), means


def test_empty_mask_stays_empty():
    empty = np.zeros((16, 16, 16), dtype=np.float32)
    out = corrupt_mask(empty, "extreme", np.random.default_rng(0))
    assert out.sum() == 0


def test_every_preset_runs(small_mask):
    for severity in PRESETS:
        out = corrupt_mask(small_mask, severity, np.random.default_rng(3))
        assert out.shape == small_mask.shape


def test_schedule_draws_a_known_severity():
    rng = np.random.default_rng(0)
    names = {sample_severity("mixed", rng) for _ in range(200)}
    assert names.issubset({"clean", "default", "mid", "strong"})
