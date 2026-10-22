"""A synthetic corpus for verification.

`make_demo_corpus` writes a small corpus of balls in the layout that
`build_manifest` expects. The example script and the test suite both use it, so
neither needs a download.

The masks are balls with a random centre. They exercise the code paths. They
hold no anatomy.
"""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np

__all__ = ["ball", "make_demo_corpus", "DEMO_DATASETS"]

# Two datasets, two modalities, four classes. One anatomy name holds two
# tokens, which exercises the suffix match of `build_manifest`.
DEMO_DATASETS = {
    "Task03_Liver": ("CT", ["liver", "liver_tumour"]),
    "ACDC": ("MRI", ["left_ventricle", "right_ventricle"]),
}


def ball(shape: tuple[int, int, int], centre, radius: float) -> np.ndarray:
    """Give a binary ball of `radius` voxels at `centre`."""
    grid = np.indices(shape).astype(np.float32)
    squared = sum((grid[axis] - centre[axis]) ** 2 for axis in range(3))
    return (squared <= radius ** 2).astype(np.float32)


def make_demo_corpus(root: str | Path,
                     datasets: dict | None = None,
                     cases_per_class: int = 4,
                     test_cases: int = 1,
                     grid: int = 24,
                     seed: int = 0) -> Path:
    """Write a synthetic corpus to `root` and give the path.

    Args:
        root: the target directory.
        datasets: a map from dataset name to (modality, anatomy list). The
            default is DEMO_DATASETS.
        cases_per_class: cases to write for each anatomy.
        test_cases: cases of each anatomy that go to the test split. The
            remainder go to the train split.
        grid: voxels on each side of a volume.
        seed: the seed of the random centre and the image noise.

    Layout:
        root/dataset_info.json
        root/images/{train,test}/<dataset>_<case>_<anatomy>.nii.gz
        root/labels/{train,test}/<same name>
    """
    root = Path(root)
    datasets = datasets or DEMO_DATASETS
    rng = np.random.default_rng(seed)
    info = {"datasets": []}
    affine = np.eye(4)

    for dataset, (modality, anatomies) in datasets.items():
        info["datasets"].append({"name": dataset, "modality": modality,
                                 "classes": list(anatomies)})
        for anatomy in anatomies:
            for index in range(cases_per_class):
                split = "test" if index >= cases_per_class - test_cases else "train"
                for folder in ("images", "labels"):
                    (root / folder / split).mkdir(parents=True, exist_ok=True)

                centre = np.array([grid / 2] * 3) + rng.normal(0, 1.5, size=3)
                mask = ball((grid,) * 3, centre, radius=6.0 + rng.uniform(-1, 1))
                image = mask * 0.8 + rng.normal(0, 0.05, size=mask.shape)

                name = f"{dataset}_case{index:02d}_{anatomy}.nii.gz"
                nib.save(nib.Nifti1Image(image.astype(np.float32), affine),
                         root / "images" / split / name)
                nib.save(nib.Nifti1Image(mask, affine),
                         root / "labels" / split / name)

    (root / "dataset_info.json").write_text(json.dumps(info, indent=2))
    return root
