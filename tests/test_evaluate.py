"""Tests for the evaluation, the summary and the gate."""

from __future__ import annotations

import numpy as np
import pytest

from grace.data.dataset import RefinementDataset
from grace.evaluate import evaluate, harm_by_quality, summarise
from grace.gate import apply_gate, gate_curve
from grace.models.refiner import GraceRefiner, RefinerConfig
from grace.retrieval.index import ExemplarIndex

SIZE = (16, 16, 16)
SMALL = RefinerConfig(channels=(8, 16, 32), strides=(2, 2), num_res_units=1)


@pytest.fixture
def records(manifests):
    """Evaluate an untrained refiner. The scores are poor, the format is right."""
    train, test = manifests
    dataset = RefinementDataset(test, index=ExemplarIndex(train), size=SIZE,
                                severity="strong", seed=42)
    model = GraceRefiner(SMALL)
    return evaluate(model, dataset, batch_size=2, num_workers=0, device="cpu")


def test_one_record_for_each_case(records, manifests):
    _, test = manifests
    assert len(records) == len(test)


def test_a_record_holds_the_scores_and_the_metadata(records):
    record = records[0]
    for key in ("uid", "class_key", "severity", "raw_dice", "refined_dice",
                "delta_dice", "raw_hd95", "refined_hd95"):
        assert key in record, key
    assert np.isclose(record["delta_dice"],
                      record["refined_dice"] - record["raw_dice"])


def test_boundary_metrics_are_optional(manifests):
    train, test = manifests
    dataset = RefinementDataset(test, index=ExemplarIndex(train), size=SIZE,
                                severity="mid", seed=1)
    records = evaluate(GraceRefiner(SMALL), dataset, batch_size=2,
                       num_workers=0, device="cpu", boundary_metrics=False)
    assert "raw_hd95" not in records[0]


def test_the_summary_groups_the_classes(records):
    summary = summarise(records)
    assert summary["n"] == len(records)
    assert set(summary["per_class"]) == {r["class_key"] for r in records}
    assert all(block["n"] >= 1 for block in summary["per_class"].values())


def test_the_summary_reports_the_tail(records):
    summary = summarise(records)
    assert summary["delta_dice_worst"] <= summary["delta_dice_p05"]
    assert summary["delta_dice_p05"] <= summary["delta_dice_mean"] + 1e-9


def test_an_empty_record_list_gives_an_empty_summary():
    assert summarise([])["n"] == 0


def _synthetic_records():
    """Make records with a known shape: harm increases with input quality."""
    rows = []
    for raw, delta in [(0.10, 0.30), (0.15, 0.25), (0.35, 0.15), (0.45, 0.10),
                       (0.60, 0.04), (0.70, -0.02), (0.85, -0.03), (0.95, -0.08)]:
        rows.append({"raw_dice": raw, "refined_dice": raw + delta,
                     "delta_dice": delta, "class_key": "D/organ"})
    return rows


def test_harm_bands_cover_every_case():
    rows = harm_by_quality(_synthetic_records())
    assert sum(row["n"] for row in rows) == 8


def test_the_harm_rate_rises_with_the_input_quality():
    rows = [row for row in harm_by_quality(_synthetic_records()) if row["n"]]
    rates = [row["percent_harmed_01"] for row in rows]
    assert rates == sorted(rates), rates


def test_the_gate_keeps_a_good_input_unchanged():
    gated = apply_gate(_synthetic_records(), threshold=0.5)
    for record in gated:
        if record["raw_dice"] >= 0.5:
            assert record["gated"] is True
            assert record["delta_dice"] == 0.0
            assert record["refined_dice"] == record["raw_dice"]
        else:
            assert record["gated"] is False


def test_the_gate_curve_refines_more_as_the_threshold_rises():
    rows = gate_curve(_synthetic_records(), thresholds=[0.2, 0.5, 0.8, 1.01])
    fractions = [row["percent_refined"] for row in rows]
    assert fractions == sorted(fractions)
    assert rows[-1]["percent_refined"] == 100.0


def test_the_gate_lowers_the_harm_rate():
    rows = gate_curve(_synthetic_records(), thresholds=[0.5, 1.01])
    assert rows[0]["percent_harmed_01"] < rows[1]["percent_harmed_01"]
