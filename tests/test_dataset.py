"""Tests for the dataset."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from grace.data.dataset import RefinementDataset
from grace.metrics import dice
from grace.retrieval.index import ExemplarIndex

SIZE = (24, 24, 24)


@pytest.fixture
def dataset(manifests):
    train, _ = manifests
    return RefinementDataset(train, index=ExemplarIndex(train), size=SIZE,
                             severity="strong", seed=42)


def test_item_holds_the_expected_shapes(dataset):
    item = dataset[0]
    for key in ("image", "mask", "corrupted", "exemplar"):
        assert item[key].shape == (1,) + SIZE, key
        assert item[key].dtype == torch.float32
    assert item["uid"] == dataset.manifest[0].uid
    assert item["severity"] == "strong"


def test_a_fixed_seed_repeats_the_degradation(dataset):
    first = dataset[0]["corrupted"]
    second = dataset[0]["corrupted"]
    assert torch.equal(first, second)


def test_no_seed_varies_the_degradation(manifests):
    train, _ = manifests
    dataset = RefinementDataset(train, index=None, size=SIZE,
                                severity="strong", seed=None)
    assert not torch.equal(dataset[0]["corrupted"], dataset[0]["corrupted"])


def test_the_degradation_lowers_the_dice(dataset):
    item = dataset[0]
    assert dice(item["corrupted"], item["mask"]) < 0.99


def test_the_exemplar_comes_from_the_same_class(dataset):
    item = dataset[0]
    assert item["has_exemplar"] == 1.0
    assert item["exemplar_uid"].rsplit("/", 1)[0] == item["class_key"]
    assert item["exemplar_uid"] != item["uid"]


def test_no_index_gives_a_zero_exemplar(manifests):
    train, _ = manifests
    dataset = RefinementDataset(train, index=None, size=SIZE, seed=0)
    item = dataset[0]
    assert item["exemplar"].sum() == 0
    assert item["has_exemplar"] == 0.0


def test_clean_severity_keeps_the_mask(manifests):
    train, _ = manifests
    dataset = RefinementDataset(train, index=None, size=SIZE,
                                severity="clean", seed=0)
    item = dataset[0]
    assert torch.equal(item["corrupted"], item["mask"])


def test_a_schedule_draws_several_severities(manifests):
    train, _ = manifests
    dataset = RefinementDataset(train, index=None, size=SIZE,
                                severity="mixed", seed=1)
    severities = {dataset[i]["severity"] for i in range(len(dataset))}
    assert severities.issubset({"clean", "default", "mid", "strong"})


def test_no_image_gives_a_zero_image(manifests):
    train, _ = manifests
    dataset = RefinementDataset(train, index=None, size=SIZE, use_image=False,
                                seed=0)
    assert dataset[0]["image"].sum() == 0


def test_the_default_collate_works(dataset):
    """Every item holds the same keys and shapes, so no custom collate is needed."""
    batch = next(iter(DataLoader(dataset, batch_size=2)))
    assert batch["corrupted"].shape == (2, 1) + SIZE
    assert len(batch["uid"]) == 2
    assert batch["has_exemplar"].shape == (2,)


def test_an_unknown_severity_raises(manifests):
    train, _ = manifests
    with pytest.raises(KeyError, match="unknown severity"):
        RefinementDataset(train, size=SIZE, severity="catastrophic")


def test_the_mask_stays_binary_after_the_resample(dataset):
    values = np.unique(dataset[0]["mask"].numpy())
    assert set(values).issubset({0.0, 1.0})
