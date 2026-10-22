"""End to end example on a synthetic corpus.

The script writes a small corpus of balls, builds the manifests, trains a small
refiner for a few epochs, evaluates it and prints the gate curve. It runs on a
CPU in approximately three minutes.

Use it to verify an installation. The corpus holds balls, so the refiner it
trains has no use. Replace the first step with your own data to start a real
run.

    pip install -e .
    python examples/quickstart.py --output-dir /tmp/grace-quickstart
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from grace.config import TrainConfig
from grace.data.dataset import RefinementDataset
from grace.data.manifest import build_manifest, check_manifest
from grace.data.synthetic import make_demo_corpus
from grace.evaluate import evaluate, summarise
from grace.gate import gate_curve
from grace.models.refiner import GraceRefiner, RefinerConfig
from grace.retrieval.index import ExemplarIndex
from grace.train import train

SIZE = (16, 16, 16)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="/tmp/grace-quickstart")
    parser.add_argument("--epochs", type=int, default=4)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("1. write the corpus")
    corpus = make_demo_corpus(output_dir / "corpus", cases_per_class=6,
                              test_cases=2)

    print("2. build and audit the manifests")
    train_manifest, _ = build_manifest(corpus, "train")
    test_manifest, _ = build_manifest(corpus, "test")
    train_path = train_manifest.save(output_dir / "manifests" / "train.json")
    test_path = test_manifest.save(output_dir / "manifests" / "test.json")
    print(json.dumps(check_manifest(train_manifest), indent=2))

    print("3. train")
    config = TrainConfig(
        train_manifest=str(train_path),
        val_manifest=str(test_path),
        output_dir=str(output_dir / "run"),
        size=SIZE,
        epochs=args.epochs,
        batch_size=2,
        num_workers=0,
        eval_every=1,
        log_every=1000,
        max_val_cases=6,
    )
    small = RefinerConfig(channels=(8, 16, 32), strides=(2, 2), num_res_units=1)
    train(config, small)

    print("4. evaluate")
    model = GraceRefiner.load(output_dir / "run" / "best.pt")
    dataset = RefinementDataset(test_manifest, index=ExemplarIndex(train_manifest),
                                size=SIZE, severity="strong", seed=42)
    records = evaluate(model, dataset, batch_size=2, num_workers=0, device="cpu")
    summary = summarise(records)
    print(f"   input   {summary['floor_dice_mean']:.4f}")
    print(f"   refined {summary['refined_dice_mean']:.4f}")
    print(f"   delta   {summary['delta_dice_mean']:+.4f}  over n={summary['n']}")

    print("5. gate curve")
    for row in gate_curve(records):
        print(f"   threshold {row['threshold']:.2f}  "
              f"refined {row['percent_refined']:5.1f}%  "
              f"delta {row['delta_dice_mean']:+.4f}  "
              f"harmed {row['percent_harmed_01']:5.1f}%")

    print(f"\nartefacts are in {output_dir}")


if __name__ == "__main__":
    main()
