"""Exemplar selection from a pool of verified cases.

The refiner reads one retrieved mask on a separate input channel. This module
chooses that mask. It groups a pool manifest by class key, then selects one
case for each query.

Build the pool from the training split. The pool and the queries must hold
different cases. `pick` excludes the query by its unique id.

Four modes are available:

    nearest       the top-1 neighbour from a precomputed table
    random        a random case of the same class
    wrong-class   a random case of a different class
    none          no exemplar

Use `random` as the default. Use `wrong-class` and `none` as controls. The
`none` mode measures the contribution of the exemplar channel.

For a holdout experiment, pass `exclude_datasets`. The pool then drops those
datasets, and no query can retrieve a case from them.
"""

from __future__ import annotations

import json
from pathlib import Path

from grace.data.manifest import Entry, Manifest

__all__ = ["ExemplarIndex", "MODES", "load_nn_table"]

MODES = ("nearest", "random", "wrong-class", "none")


class ExemplarIndex:
    """A pool of candidate exemplars, grouped by class key."""

    def __init__(self, pool: Manifest,
                 nn_table: dict[str, str] | None = None,
                 exclude_datasets: list[str] | None = None):
        """
        Args:
            pool: the manifest that supplies the exemplars.
            nn_table: a map from query uid to pool uid. The `nearest` mode
                reads it. Build it with `grace.retrieval.embeddings`.
            exclude_datasets: datasets to remove from the pool.
        """
        self.pool = pool.filter(exclude_datasets=exclude_datasets)
        self.nn_table = nn_table or {}

        self.by_class: dict[str, list[int]] = {}
        self.by_uid: dict[str, int] = {}
        for position, entry in enumerate(self.pool.entries):
            self.by_class.setdefault(entry.class_key, []).append(position)
            self.by_uid[entry.uid] = position
        self.class_keys = sorted(self.by_class)

    def __len__(self) -> int:
        return len(self.pool)

    def pick(self, entry: Entry, mode: str, rng) -> int | None:
        """Choose one pool position for a query.

        Args:
            entry: the query entry.
            mode: one of MODES.
            rng: numpy Generator. The same seed gives the same choice.

        Returns:
            A position in `self.pool`, or None if the pool holds no candidate.
        """
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}, expected one of {MODES}")
        if mode == "none":
            return None

        if mode == "wrong-class":
            others = [key for key in self.class_keys if key != entry.class_key]
            if not others:
                return None
            wrong_key = others[int(rng.integers(len(others)))]
            candidates = self.by_class[wrong_key]
            return candidates[int(rng.integers(len(candidates)))]

        if mode == "nearest":
            neighbour = self.nn_table.get(entry.uid)
            if neighbour is not None and neighbour != entry.uid:
                position = self.by_uid.get(neighbour)
                if position is not None:
                    return position
            # The table holds no usable neighbour. Fall back to `random`.

        candidates = [p for p in self.by_class.get(entry.class_key, [])
                      if self.pool.entries[p].uid != entry.uid]
        if not candidates:
            return None
        return candidates[int(rng.integers(len(candidates)))]

    def report(self) -> dict:
        """Summarise the pool. Check this before you train or evaluate."""
        sizes = sorted(len(v) for v in self.by_class.values())
        return {
            "n_pool_cases": len(self.pool),
            "n_classes": len(self.by_class),
            "median_class_size": sizes[len(sizes) // 2] if sizes else 0,
            "n_singleton_classes": sum(1 for s in sizes if s < 2),
            "n_nn_pairs": len(self.nn_table),
        }


def load_nn_table(path: str | Path) -> dict[str, str]:
    """Read a nearest neighbour table from JSON."""
    return {str(k): str(v) for k, v in json.loads(Path(path).read_text()).items()}
