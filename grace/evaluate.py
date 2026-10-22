"""Evaluation of a refiner.

`evaluate` gives one record for each case. `summarise` aggregates the records.
Each record holds the score before and after refinement, so you can group the
results again later without a second inference pass.

The delta of a case depends on the quality of its input. Report a delta
together with the input score that produced it.

Set a fixed seed. Each case then receives the same degraded mask on every run,
and two checkpoints compare on identical inputs.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from grace.config import EvalConfig
from grace.data.dataset import RefinementDataset
from grace.data.manifest import load_manifest
from grace.metrics import boundary_f1, dice, hausdorff_95
from grace.models.refiner import GraceRefiner
from grace.retrieval.index import ExemplarIndex, load_nn_table

__all__ = ["evaluate", "summarise", "run_evaluation"]

logger = logging.getLogger("grace.evaluate")

# Bands of input quality for the harm analysis.
QUALITY_BANDS = [(0.0, 0.2), (0.2, 0.5), (0.5, 0.8), (0.8, 1.01)]


@torch.no_grad()
def evaluate(model: GraceRefiner, dataset, batch_size: int = 4,
             num_workers: int = 2, device: str | torch.device | None = None,
             boundary_metrics: bool = True) -> list[dict]:
    """Refine every case of a dataset and give one record for each case.

    Args:
        model: the refiner.
        dataset: a RefinementDataset, or a Subset of one.
        batch_size: cases for each forward pass.
        num_workers: data loader workers.
        device: the device. The function selects CUDA if it is available.
        boundary_metrics: compute hausdorff_95 and boundary_f1. These metrics
            run on the CPU and add approximately 0.3 s for each case.

    Returns:
        A list of records. Each record holds the metadata of the case and its
        scores before and after refinement.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    model = model.to(device).eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers)
    records: list[dict] = []

    for batch in loader:
        corrupted = batch["corrupted"].to(device)
        exemplar = batch["exemplar"].to(device)
        image = batch["image"].to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(corrupted, exemplar, image)
        refined = logits.argmax(dim=1, keepdim=True).float().cpu().numpy()

        truth = batch["mask"].numpy()
        degraded = batch["corrupted"].numpy()

        for i in range(len(truth)):
            record = {
                "uid": batch["uid"][i],
                "case_id": batch["case_id"][i],
                "dataset": batch["dataset"][i],
                "anatomy": batch["anatomy"][i],
                "modality": batch["modality"][i],
                "class_key": batch["class_key"][i],
                "severity": batch["severity"][i],
                "has_exemplar": float(batch["has_exemplar"][i]),
                "exemplar_uid": batch["exemplar_uid"][i],
                "raw_dice": dice(degraded[i], truth[i]),
                "refined_dice": dice(refined[i], truth[i]),
            }
            record["delta_dice"] = record["refined_dice"] - record["raw_dice"]
            if boundary_metrics:
                record["raw_hd95"] = hausdorff_95(degraded[i], truth[i])
                record["refined_hd95"] = hausdorff_95(refined[i], truth[i])
                record["raw_boundary_f1"] = boundary_f1(degraded[i], truth[i])
                record["refined_boundary_f1"] = boundary_f1(refined[i], truth[i])
            records.append(record)

    return records


def _mean(values: list[float]) -> float:
    """Give the mean, and ignore the NaN values."""
    clean = [v for v in values if not np.isnan(v)]
    return float(np.mean(clean)) if clean else float("nan")


def summarise(records: list[dict]) -> dict:
    """Aggregate the records of one evaluation.

    The summary holds a global block, a block for each class, and the harm
    analysis. Keep `n` for each class. A run of 200 cases over 37 classes gives
    approximately 5 cases for each class, and a per-class delta then carries
    large uncertainty.
    """
    if not records:
        return {"n": 0}

    raw = [r["raw_dice"] for r in records]
    refined = [r["refined_dice"] for r in records]
    delta = [r["delta_dice"] for r in records]

    summary = {
        "n": len(records),
        "floor_dice_mean": _mean(raw),
        "refined_dice_mean": _mean(refined),
        "delta_dice_mean": _mean(delta),
        "delta_dice_std": float(np.std(delta)),
        "delta_dice_p05": float(np.percentile(delta, 5)),
        "delta_dice_worst": float(np.min(delta)),
        "n_with_exemplar": int(sum(r["has_exemplar"] for r in records)),
    }

    if "raw_hd95" in records[0]:
        summary["floor_hd95_mean"] = _mean([r["raw_hd95"] for r in records])
        summary["refined_hd95_mean"] = _mean([r["refined_hd95"] for r in records])
        summary["delta_hd95_mean"] = _mean(
            [r["refined_hd95"] - r["raw_hd95"] for r in records])

    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[record["class_key"]].append(record)
    summary["per_class"] = {
        key: {
            "n": len(group),
            "floor_dice": _mean([r["raw_dice"] for r in group]),
            "refined_dice": _mean([r["refined_dice"] for r in group]),
            "delta_dice": _mean([r["delta_dice"] for r in group]),
        }
        for key, group in sorted(grouped.items())
    }

    summary["harm_by_quality"] = harm_by_quality(records)
    return summary


def harm_by_quality(records: list[dict], bands=None) -> list[dict]:
    """Count the harmed cases in each band of input quality.

    A case is harmed when its delta falls below the threshold. The function
    reports two thresholds, 0.01 and 0.05 Dice, for each band.
    """
    bands = bands or QUALITY_BANDS
    rows = []
    for low, high in bands:
        group = [r for r in records if low <= r["raw_dice"] < high]
        if not group:
            rows.append({"band": f"{low:.1f}-{high:.1f}", "n": 0})
            continue
        delta = [r["delta_dice"] for r in group]
        rows.append({
            "band": f"{low:.1f}-{high:.1f}",
            "n": len(group),
            "delta_dice_mean": _mean(delta),
            "percent_harmed_01": 100.0 * np.mean([d < -0.01 for d in delta]),
            "percent_harmed_05": 100.0 * np.mean([d < -0.05 for d in delta]),
            "delta_dice_p05": float(np.percentile(delta, 5)),
        })
    return rows


def run_evaluation(config: EvalConfig) -> dict:
    """Evaluate one checkpoint at every severity of a configuration.

    The function writes the summary and all per-case records to
    `config.output`. Keep the records. The harm analysis and the gate curve
    read them without a second inference pass.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GraceRefiner.load(config.checkpoint, device=device)

    queries = load_manifest(config.manifest)
    if config.datasets or config.anatomies:
        queries = queries.filter(datasets=config.datasets or None,
                                 anatomies=config.anatomies or None)
    pool = load_manifest(config.pool_manifest)
    nn_table = load_nn_table(config.nn_table) if config.nn_table else None
    index = ExemplarIndex(pool, nn_table=nn_table,
                          exclude_datasets=config.exclude_datasets)
    logger.info("exemplar pool: %s", index.report())

    results = {"checkpoint": str(config.checkpoint),
               "config": config.to_dict(),
               "pool": index.report(),
               "severities": {}}

    for severity in config.severities:
        dataset = RefinementDataset(
            queries, index=index, size=config.size, severity=severity,
            exemplar_mode=config.exemplar_mode, seed=config.seed,
        )
        if 0 < config.max_cases < len(dataset):
            rng = np.random.default_rng(config.seed)
            picks = rng.choice(len(dataset), size=config.max_cases, replace=False)
            dataset = Subset(dataset, sorted(int(p) for p in picks))

        records = evaluate(model, dataset, batch_size=config.batch_size,
                           num_workers=config.num_workers, device=device,
                           boundary_metrics=config.boundary_metrics)
        summary = summarise(records)
        results["severities"][severity] = {"summary": summary, "records": records}
        logger.info("severity=%s floor=%.4f refined=%.4f delta=%+.4f n=%d",
                    severity, summary["floor_dice_mean"],
                    summary["refined_dice_mean"], summary["delta_dice_mean"],
                    summary["n"])

    output = Path(config.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))
    logger.info("wrote %s", output)
    return results
