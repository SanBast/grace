"""Command line interface.

Five commands:

    grace manifest    build and audit a sample manifest
    grace nn-table    build the nearest neighbour table for retrieval
    grace train       train a refiner
    grace evaluate    score a checkpoint and write the per-case records
    grace gate        print the gate curve from an evaluation file

Each command parses its arguments and calls one library function. Import the
library directly if you need more control.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from grace.config import EvalConfig, TrainConfig

__all__ = ["main"]


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")


def cmd_manifest(args) -> None:
    """Build a manifest, then report the class sizes."""
    from grace.data.manifest import build_manifest, check_manifest

    manifest, skipped = build_manifest(args.root, args.split, strict=not args.allow_unknown)
    output = Path(args.output or f"manifests/{args.split}.json")
    manifest.save(output)

    report = check_manifest(manifest)
    print(json.dumps(report, indent=2))
    print(f"wrote {output}")
    if skipped:
        print(f"skipped {len(skipped)} files, first few: {skipped[:5]}")
    if report["n_singleton_classes"]:
        print(f"WARNING: {report['n_singleton_classes']} classes hold one case. "
              "Those queries receive no exemplar.")


def cmd_nn_table(args) -> None:
    """Build the nearest neighbour table of the queries over the pool."""
    from grace.data.manifest import load_manifest
    from grace.retrieval.embeddings import build_nn_table

    queries = load_manifest(args.manifest)
    pool = load_manifest(args.pool)
    table = build_nn_table(queries, pool, grid=args.grid, output=args.output)
    print(f"wrote {args.output} with {len(table)} pairs")


def cmd_train(args) -> None:
    """Train a refiner."""
    from grace.models.refiner import RefinerConfig
    from grace.train import train

    config = TrainConfig(
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
        output_dir=args.output_dir,
        size=(args.size,) * 3,
        severity=args.severity,
        val_severity=args.val_severity,
        exemplar_mode=args.exemplar_mode,
        nn_table=args.nn_table,
        use_image=not args.no_image,
        exclude_datasets=args.exclude_datasets or [],
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    refiner = RefinerConfig(
        use_image=not args.no_image,
        fusion=args.fusion,
        exemplar_dropout=args.exemplar_dropout,
    )
    train(config, refiner)


def cmd_evaluate(args) -> None:
    """Score a checkpoint at each severity."""
    from grace.evaluate import run_evaluation

    config = EvalConfig(
        checkpoint=args.checkpoint,
        manifest=args.manifest,
        pool_manifest=args.pool,
        output=args.output,
        size=(args.size,) * 3,
        severities=args.severities,
        exemplar_mode=args.exemplar_mode,
        nn_table=args.nn_table,
        exclude_datasets=args.exclude_datasets or [],
        datasets=args.datasets or [],
        anatomies=args.anatomies or [],
        max_cases=args.max_cases,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        boundary_metrics=not args.no_boundary_metrics,
        seed=args.seed,
    )
    results = run_evaluation(config)
    for severity, block in results["severities"].items():
        summary = block["summary"]
        print(f"{severity:12s} floor={summary['floor_dice_mean']:.4f} "
              f"refined={summary['refined_dice_mean']:.4f} "
              f"delta={summary['delta_dice_mean']:+.4f} n={summary['n']}")


def cmd_gate(args) -> None:
    """Print the gate curve from an evaluation file."""
    from grace.gate import gate_curve

    results = json.loads(Path(args.results).read_text())
    block = results["severities"][args.severity]
    rows = gate_curve(block["records"], thresholds=args.thresholds)
    print(f"{'threshold':>10} {'refined %':>10} {'mean delta':>11} "
          f"{'harmed %':>9} {'worst':>8}")
    for row in rows:
        print(f"{row['threshold']:>10.2f} {row['percent_refined']:>10.1f} "
              f"{row['delta_dice_mean']:>+11.4f} {row['percent_harmed_01']:>9.1f} "
              f"{row['delta_dice_worst']:>+8.3f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="grace", description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="command", required=True)

    p = subparsers.add_parser("manifest", help="build and audit a manifest")
    p.add_argument("--root", required=True, help="corpus root")
    p.add_argument("--split", default="train")
    p.add_argument("--output", default=None, help="output JSON path")
    p.add_argument("--allow-unknown", action="store_true",
                   help="skip files with no declared anatomy")
    p.set_defaults(func=cmd_manifest)

    p = subparsers.add_parser("nn-table", help="build the retrieval table")
    p.add_argument("--manifest", required=True, help="manifest of the queries")
    p.add_argument("--pool", required=True, help="manifest of the exemplar pool")
    p.add_argument("--output", default="manifests/nn_table.json")
    p.add_argument("--grid", type=int, default=16)
    p.set_defaults(func=cmd_nn_table)

    p = subparsers.add_parser("train", help="train a refiner")
    p.add_argument("--train-manifest", required=True)
    p.add_argument("--val-manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--size", type=int, default=96)
    p.add_argument("--severity", default="mixed")
    p.add_argument("--val-severity", default="strong")
    p.add_argument("--exemplar-mode", default="random")
    p.add_argument("--nn-table", default=None)
    p.add_argument("--fusion", default="concat", choices=["concat", "film"])
    p.add_argument("--exemplar-dropout", type=float, default=0.10)
    p.add_argument("--no-image", action="store_true")
    p.add_argument("--exclude-datasets", nargs="+", default=None)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_train)

    p = subparsers.add_parser("evaluate", help="score a checkpoint")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--manifest", required=True, help="manifest of the queries")
    p.add_argument("--pool", required=True, help="manifest of the exemplar pool")
    p.add_argument("--output", default="results/evaluation.json")
    p.add_argument("--size", type=int, default=96)
    p.add_argument("--severities", nargs="+",
                   default=["default", "mid", "strong"])
    p.add_argument("--exemplar-mode", default="random",
                   choices=["nearest", "random", "wrong-class", "none"])
    p.add_argument("--nn-table", default=None)
    p.add_argument("--exclude-datasets", nargs="+", default=None,
                   help="datasets to remove from the exemplar pool")
    p.add_argument("--datasets", nargs="+", default=None,
                   help="restrict the queries to these datasets")
    p.add_argument("--anatomies", nargs="+", default=None,
                   help="restrict the queries to these anatomies")
    p.add_argument("--max-cases", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--no-boundary-metrics", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_evaluate)

    p = subparsers.add_parser("gate", help="print the gate curve")
    p.add_argument("--results", required=True, help="evaluation JSON file")
    p.add_argument("--severity", default="strong")
    p.add_argument("--thresholds", nargs="+", type=float, default=None)
    p.set_defaults(func=cmd_gate)

    return parser


def main(argv=None) -> None:
    _configure_logging()
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
