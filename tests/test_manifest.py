"""Tests for the manifest.

The grouping by class key is the part that matters. A mistake there inflates
every score, so these tests check the parse and the audit.
"""

from __future__ import annotations

import pytest

from grace.data.manifest import (
    build_manifest,
    check_manifest,
    class_counts,
    load_manifest,
)


def test_build_resolves_dataset_anatomy_and_case(manifests):
    train, _ = manifests
    entry = next(e for e in train.entries if e.dataset == "Task03_Liver")
    assert entry.anatomy in {"liver", "liver_tumour"}
    assert entry.case_id.startswith("case")
    assert entry.modality == "CT"


def test_multi_token_anatomy_stays_whole(manifests):
    train, _ = manifests
    anatomies = {e.anatomy for e in train.entries}
    assert "liver_tumour" in anatomies
    assert "right_ventricle" in anatomies


def test_class_key_groups_the_cases(manifests):
    train, _ = manifests
    counts = class_counts(train)
    assert len(counts) == 4
    assert all(n == 3 for n in counts.values()), counts


def test_audit_reports_no_singleton_class(manifests):
    train, _ = manifests
    report = check_manifest(train)
    assert report["n_singleton_classes"] == 0
    assert report["n_classes"] == 4
    assert report["median_class_size"] == 3


def test_audit_finds_a_singleton_class(manifests):
    """A class of one case cannot supply an exemplar. The audit must say so."""
    train, _ = manifests
    reduced = train.filter(anatomies=["liver"])
    reduced.entries = reduced.entries[:1]
    report = check_manifest(reduced)
    assert report["n_singleton_classes"] == 1


def test_filter_by_dataset_and_exclusion(manifests):
    train, _ = manifests
    only_acdc = train.filter(datasets=["ACDC"])
    assert {e.dataset for e in only_acdc.entries} == {"ACDC"}

    without_acdc = train.filter(exclude_datasets=["ACDC"])
    assert "ACDC" not in {e.dataset for e in without_acdc.entries}
    assert len(only_acdc) + len(without_acdc) == len(train)


def test_save_and_load_round_trip(manifests, tmp_path):
    train, _ = manifests
    path = train.save(tmp_path / "train.json")
    restored = load_manifest(path)
    assert len(restored) == len(train)
    assert restored.entries[0] == train.entries[0]


def test_strict_mode_rejects_an_undeclared_anatomy(corpus):
    """A file that holds no declared anatomy must raise an error."""
    import nibabel as nib
    import numpy as np

    extra = corpus / "labels" / "train" / "Task03_Liver_case99_kidney.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((8, 8, 8), dtype=np.float32), np.eye(4)), extra)
    try:
        with pytest.raises(ValueError, match="no anatomy"):
            build_manifest(corpus, "train", strict=True)
        manifest, skipped = build_manifest(corpus, "train", strict=False)
        assert extra.name in skipped
        assert all("kidney" != e.anatomy for e in manifest.entries)
    finally:
        extra.unlink()
