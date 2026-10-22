"""Shared fixtures. Every test runs on a synthetic corpus, with no download."""

from __future__ import annotations

import pytest

from grace.data.synthetic import ball, make_demo_corpus


@pytest.fixture(scope="session")
def corpus(tmp_path_factory):
    """Write a small corpus to disk and give its root path.

    Four classes, three train cases and one test case for each class.
    """
    return make_demo_corpus(tmp_path_factory.mktemp("corpus"))


@pytest.fixture(scope="session")
def manifests(corpus):
    """Give the train and test manifests of the synthetic corpus."""
    from grace.data.manifest import build_manifest

    train, _ = build_manifest(corpus, "train")
    test, _ = build_manifest(corpus, "test")
    return train, test


@pytest.fixture
def small_mask():
    """Give a binary ball of 32 voxels on each side."""
    return ball((32, 32, 32), (16, 16, 16), radius=8.0)
