"""Tests for exemplar retrieval.

The query must never receive its own mask. These tests check that rule in each
mode.
"""

from __future__ import annotations

import numpy as np
import pytest

from grace.data.manifest import Entry
from grace.retrieval.embeddings import build_nn_table, shape_embedding
from grace.retrieval.index import ExemplarIndex


@pytest.fixture
def index(manifests):
    train, _ = manifests
    return ExemplarIndex(train)


def test_pool_report(index):
    report = index.report()
    assert report["n_classes"] == 4
    assert report["n_singleton_classes"] == 0


def test_uid_separates_two_classes_of_one_case(manifests):
    """One patient gives one case for each organ. The uid must separate them."""
    train, _ = manifests
    uids = [e.uid for e in train.entries]
    case_ids = [e.case_id for e in train.entries]
    assert len(set(uids)) == len(train)
    assert len(set(case_ids)) < len(train)


def test_random_mode_keeps_the_class(index, manifests):
    train, _ = manifests
    rng = np.random.default_rng(0)
    for entry in train.entries:
        choice = index.pick(entry, "random", rng)
        assert choice is not None
        assert index.pool[choice].class_key == entry.class_key


def test_random_mode_excludes_the_query(index, manifests):
    train, _ = manifests
    rng = np.random.default_rng(0)
    for entry in train.entries:
        for _ in range(10):
            choice = index.pick(entry, "random", rng)
            assert index.pool[choice].uid != entry.uid


def test_wrong_class_mode_changes_the_class(index, manifests):
    train, _ = manifests
    entry = train.entries[0]
    rng = np.random.default_rng(0)
    for _ in range(10):
        choice = index.pick(entry, "wrong-class", rng)
        assert index.pool[choice].class_key != entry.class_key


def test_none_mode_gives_no_exemplar(index, manifests):
    train, _ = manifests
    assert index.pick(train.entries[0], "none", np.random.default_rng(0)) is None


def test_unknown_class_gives_no_exemplar(index):
    stranger = Entry(case_id="case00", dataset="Nowhere", anatomy="organ",
                     modality="CT", image="a.nii.gz", mask="a.nii.gz")
    assert index.pick(stranger, "random", np.random.default_rng(0)) is None


def test_exclusion_removes_a_dataset_from_the_pool(manifests):
    train, _ = manifests
    index = ExemplarIndex(train, exclude_datasets=["ACDC"])
    assert all(e.dataset != "ACDC" for e in index.pool.entries)
    acdc = next(e for e in train.entries if e.dataset == "ACDC")
    assert index.pick(acdc, "random", np.random.default_rng(0)) is None


def test_the_same_seed_gives_the_same_choice(index, manifests):
    train, _ = manifests
    entry = train.entries[0]
    first = index.pick(entry, "random", np.random.default_rng(5))
    second = index.pick(entry, "random", np.random.default_rng(5))
    assert first == second


def test_unknown_mode_raises(index, manifests):
    train, _ = manifests
    with pytest.raises(ValueError, match="unknown mode"):
        index.pick(train.entries[0], "magic", np.random.default_rng(0))


def test_shape_embedding_has_unit_length(small_mask):
    vector = shape_embedding(small_mask, grid=8)
    assert vector.shape == (512,)
    assert np.isclose(np.linalg.norm(vector), 1.0, atol=1e-5)


def test_nearest_mode_follows_the_table(manifests, tmp_path):
    train, test = manifests
    table = build_nn_table(test, train, grid=8, output=tmp_path / "nn_table.json")
    assert table
    index = ExemplarIndex(train, nn_table=table)
    entry = test.entries[0]
    choice = index.pick(entry, "nearest", np.random.default_rng(0))
    assert index.pool[choice].uid == table[entry.uid]


def test_nearest_mode_keeps_the_class(manifests):
    train, test = manifests
    table = build_nn_table(test, train, grid=8)
    index = ExemplarIndex(train, nn_table=table)
    for entry in test.entries:
        choice = index.pick(entry, "nearest", np.random.default_rng(0))
        assert index.pool[choice].class_key == entry.class_key
