"""The input quality gate.

A gate refines a case only when the quality of its input is below a threshold.
It leaves the other cases unchanged.

`gate_curve` gives the trade-off at several thresholds: how many cases the
gate refines, the mean delta over all cases, and how many cases it harms.
Select a threshold from that curve, then set it with `apply_gate`.

A deployment needs an estimate of the input quality, because the ground truth
is absent. Use the confidence of the upstream segmentor, or a quality control
model. The records of an evaluation hold the true Dice, so `gate_curve` gives
an upper bound on the performance of a real gate.
"""

from __future__ import annotations

import numpy as np

__all__ = ["apply_gate", "gate_curve"]


def apply_gate(records: list[dict], threshold: float,
               quality_key: str = "raw_dice") -> list[dict]:
    """Apply a gate to a list of evaluation records.

    The function copies each record. For a case above the threshold it sets the
    refined score to the raw score and the delta to zero.

    Args:
        records: the output of `grace.evaluate.evaluate`.
        threshold: refine a case only when its quality is below this value.
        quality_key: the field that holds the quality estimate.

    Returns:
        A new list of records with a `gated` field.
    """
    gated = []
    for record in records:
        copy = dict(record)
        refine = record[quality_key] < threshold
        copy["gated"] = not refine
        if not refine:
            copy["refined_dice"] = record["raw_dice"]
            copy["delta_dice"] = 0.0
            if "raw_hd95" in record:
                copy["refined_hd95"] = record["raw_hd95"]
        gated.append(copy)
    return gated


def gate_curve(records: list[dict], thresholds=None,
               quality_key: str = "raw_dice") -> list[dict]:
    """Give the trade-off of the gate at several thresholds.

    Each row holds the fraction of refined cases, the mean delta over all
    cases, and the harm rate. The threshold 1.01 refines every case.

    Args:
        records: the output of `grace.evaluate.evaluate`.
        thresholds: the values to test. The default is 0.3 to 1.01.
        quality_key: the field that holds the quality estimate.
    """
    thresholds = thresholds if thresholds is not None else [0.3, 0.5, 0.65, 0.8, 1.01]
    rows = []
    for threshold in thresholds:
        gated = apply_gate(records, threshold, quality_key=quality_key)
        delta = [r["delta_dice"] for r in gated]
        refined = [r for r in gated if not r["gated"]]
        rows.append({
            "threshold": threshold,
            "percent_refined": 100.0 * len(refined) / len(gated),
            "delta_dice_mean": float(np.mean(delta)),
            "percent_harmed_01": 100.0 * float(np.mean([d < -0.01 for d in delta])),
            "percent_harmed_05": 100.0 * float(np.mean([d < -0.05 for d in delta])),
            "delta_dice_worst": float(np.min(delta)),
        })
    return rows
