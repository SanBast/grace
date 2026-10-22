"""Shape embeddings and the nearest neighbour table.

The `nearest` mode of ExemplarIndex needs a map from query case id to pool case
id. This module builds that map.

The embedding of a mask is the mask at a coarse resolution, 16 voxels on each
side, flattened and normalised to unit length. The cosine similarity of two
such vectors measures the agreement of two shapes.

To use another encoder, pass your own vectors through the `query_embeddings`
and `pool_embeddings` arguments of `build_nn_table`. The vectors must have unit
length, because the function ranks the candidates by a dot product.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from grace.data.manifest import Manifest
from grace.data.volumes import read_nifti, resize_volume

__all__ = ["shape_embedding", "build_embeddings", "build_nn_table"]


def shape_embedding(mask: np.ndarray, grid: int = 16) -> np.ndarray:
    """Give a unit length vector that describes the shape of a mask."""
    coarse = resize_volume((mask > 0.5).astype(np.float32), (grid,) * 3, order=1)
    vector = coarse.reshape(-1)
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def build_embeddings(manifest: Manifest, grid: int = 16,
                     verbose: bool = True) -> np.ndarray:
    """Give one shape embedding for each entry of a manifest.

    Returns:
        An array of shape (len(manifest), grid ** 3).
    """
    vectors = np.zeros((len(manifest), grid ** 3), dtype=np.float32)
    for index in range(len(manifest)):
        mask = read_nifti(manifest.mask_path(index))
        vectors[index] = shape_embedding(mask, grid=grid)
        if verbose and (index + 1) % 200 == 0:
            print(f"  embedded {index + 1}/{len(manifest)}")
    return vectors


def build_nn_table(query: Manifest, pool: Manifest,
                   query_embeddings: np.ndarray | None = None,
                   pool_embeddings: np.ndarray | None = None,
                   grid: int = 16,
                   same_class_only: bool = True,
                   output: str | Path | None = None) -> dict[str, str]:
    """Map the uid of each query to the uid of its nearest pool case.

    The function excludes the query from its own candidates. With
    `same_class_only`, it restricts the candidates to the class of the query.

    Args:
        query: the manifest of the queries.
        pool: the manifest that supplies the exemplars.
        query_embeddings: optional vectors for `query`. The function computes
            shape embeddings if you omit them.
        pool_embeddings: optional vectors for `pool`.
        grid: the resolution of the shape embedding.
        same_class_only: restrict the candidates to the class of the query.
        output: optional path for a JSON copy of the table.

    Returns:
        A map from query uid to pool uid.
    """
    if query_embeddings is None:
        query_embeddings = build_embeddings(query, grid=grid)
    if pool_embeddings is None:
        pool_embeddings = build_embeddings(pool, grid=grid)

    positions_by_class: dict[str, list[int]] = {}
    for position, entry in enumerate(pool.entries):
        positions_by_class.setdefault(entry.class_key, []).append(position)

    table: dict[str, str] = {}
    for index, entry in enumerate(query.entries):
        if same_class_only:
            candidates = positions_by_class.get(entry.class_key, [])
        else:
            candidates = list(range(len(pool)))
        candidates = [p for p in candidates if pool.entries[p].uid != entry.uid]
        if not candidates:
            continue
        similarity = pool_embeddings[candidates] @ query_embeddings[index]
        table[entry.uid] = pool.entries[candidates[int(np.argmax(similarity))]].uid

    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(table, indent=2))
    return table
