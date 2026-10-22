"""A smoke test for the training loop.

The test runs two epochs of a small architecture on the synthetic corpus. It
checks the output files and the format of the history. It does not check the
quality of the result.
"""

from __future__ import annotations

import json

from grace.config import TrainConfig
from grace.models.refiner import GraceRefiner, RefinerConfig
from grace.train import build_datasets, train


def _config(manifests, tmp_path, corpus) -> TrainConfig:
    train_manifest, test_manifest = manifests
    train_path = train_manifest.save(tmp_path / "train.json")
    test_path = test_manifest.save(tmp_path / "test.json")
    return TrainConfig(
        train_manifest=str(train_path),
        val_manifest=str(test_path),
        output_dir=str(tmp_path / "run"),
        size=(16, 16, 16),
        severity="mixed",
        val_severity="strong",
        epochs=2,
        batch_size=2,
        num_workers=0,
        eval_every=1,
        log_every=1000,
        max_val_cases=4,
    )


def test_build_datasets_shares_one_pool(manifests, tmp_path, corpus):
    config = _config(manifests, tmp_path, corpus)
    train_dataset, val_dataset, index = build_datasets(config)
    assert len(train_dataset) == len(manifests[0])
    assert len(val_dataset) == len(manifests[1])
    assert index.report()["n_singleton_classes"] == 0
    # The validation queries retrieve from the training corpus.
    assert val_dataset.index is index


def test_training_writes_the_expected_files(manifests, tmp_path, corpus):
    config = _config(manifests, tmp_path, corpus)
    small = RefinerConfig(channels=(8, 16, 32), strides=(2, 2), num_res_units=1)
    history = train(config, small)

    output_dir = tmp_path / "run"
    for name in ("best.pt", "last.pt", "config.json", "history.json"):
        assert (output_dir / name).exists(), name

    assert len(history["train_loss"]) == 2
    assert len(history["val_dice"]) == 2
    assert all(0.0 <= value <= 1.0 for _, value in history["val_dice"])

    saved = json.loads((output_dir / "config.json").read_text())
    assert saved["train"]["epochs"] == 2
    assert saved["refiner"]["channels"] == [8, 16, 32]


def test_the_checkpoint_reloads(manifests, tmp_path, corpus):
    config = _config(manifests, tmp_path, corpus)
    config.epochs = 1
    small = RefinerConfig(channels=(8, 16), strides=(2,), num_res_units=1)
    train(config, small)

    model = GraceRefiner.load(tmp_path / "run" / "best.pt")
    assert model.config.channels == (8, 16)
