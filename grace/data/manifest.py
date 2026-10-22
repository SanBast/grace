"""The sample manifest.

A manifest is a JSON file. It holds one entry for each case and anatomy pair:

    {
      "root": "data/unified_perclass",
      "split": "train",
      "entries": [
        {
          "case_id": "liver_42",
          "dataset": "Task03_Liver",
          "anatomy": "liver",
          "modality": "CT",
          "image": "images/train/Task03_Liver_liver_42_liver.nii.gz",
          "mask":  "labels/train/Task03_Liver_liver_42_liver.nii.gz"
        }
      ]
    }

Retrieval groups the cases by the class key "<dataset>/<anatomy>". An error in
the anatomy field changes this grouping. A class of 200 cases can become 200
classes of one case, and a class of one case supplies no exemplar. A training
run shows no symptom of this condition.

The manifest holds the grouping in a file. You can read it, compare versions of
it, and keep it with your results. Run `check_manifest` to find the classes
that hold one case.

`build_manifest` reads the anatomy names from dataset_info.json. It matches the
longest suffix of each filename against the declared class list. It reports
every filename that it cannot resolve.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

__all__ = [
    "Entry",
    "Manifest",
    "build_manifest",
    "load_manifest",
    "class_counts",
    "check_manifest",
]

MODALITY_SUFFIXES = {"CT", "MR", "MRI", "US"}


@dataclass
class Entry:
    """One binary segmentation case."""

    case_id: str
    dataset: str
    anatomy: str
    modality: str
    image: str
    mask: str

    @property
    def class_key(self) -> str:
        """Give the key that retrieval groups by."""
        return f"{self.dataset}/{self.anatomy}"

    @property
    def uid(self) -> str:
        """Give the unique id of the entry.

        A case id repeats across the classes of one dataset, because one
        patient gives one case for each organ. The uid adds the class key, so
        it identifies one entry in the corpus.
        """
        return f"{self.dataset}/{self.anatomy}/{self.case_id}"


@dataclass
class Manifest:
    """One split of a corpus, as a list of entries and the root directory."""

    root: Path
    split: str
    entries: list[Entry]

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> Entry:
        return self.entries[index]

    def image_path(self, index: int) -> Path:
        return self.root / self.entries[index].image

    def mask_path(self, index: int) -> Path:
        return self.root / self.entries[index].mask

    def save(self, path: str | Path) -> Path:
        """Write the manifest to `path` as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "root": str(self.root),
            "split": self.split,
            "entries": [asdict(entry) for entry in self.entries],
        }
        path.write_text(json.dumps(payload, indent=2))
        return path

    def filter(self, datasets: list[str] | None = None,
               anatomies: list[str] | None = None,
               exclude_datasets: list[str] | None = None) -> "Manifest":
        """Give a new manifest that holds only the entries that match.

        A `datasets` pattern matches the dataset field exactly or as a prefix.
        The pattern "AMOS22_MR" selects the MR half of a corpus. The pattern
        "AMOS22" selects both halves.
        """
        kept = []
        for entry in self.entries:
            if datasets and not _matches_any(entry.dataset, datasets):
                continue
            if exclude_datasets and _matches_any(entry.dataset, exclude_datasets):
                continue
            if anatomies and entry.anatomy not in anatomies:
                continue
            kept.append(entry)
        return Manifest(root=self.root, split=self.split, entries=kept)


def _matches_any(dataset: str, patterns: list[str]) -> bool:
    return any(dataset == p or dataset.startswith(p) for p in patterns)


def load_manifest(path: str | Path) -> Manifest:
    """Read a manifest that `Manifest.save` wrote."""
    payload = json.loads(Path(path).read_text())
    return Manifest(
        root=Path(payload["root"]),
        split=payload["split"],
        entries=[Entry(**entry) for entry in payload["entries"]],
    )


# ---------------------------------------------------------------------------
# Build a manifest from a directory tree
# ---------------------------------------------------------------------------


def _load_declared_classes(root: Path) -> dict[str, set[str]]:
    """Read the anatomy names of each dataset from dataset_info.json.

    The file holds {"datasets": [{"name": ..., "classes": [...], "modality":
    ...}]}. A corpus can split one dataset by modality into "AMOS22_CT" and
    "AMOS22_MR". The classes then also register under the on-disk prefix
    "AMOS22", because the filenames hold no modality suffix.
    """
    info_path = root / "dataset_info.json"
    if not info_path.exists():
        return {}
    payload = json.loads(info_path.read_text())
    declared: dict[str, set[str]] = {}
    for dataset in payload.get("datasets", []):
        name = dataset["name"]
        classes = set(dataset.get("classes", []))
        declared.setdefault(name, set()).update(classes)
        head, _, tail = name.rpartition("_")
        if head and tail in MODALITY_SUFFIXES:
            declared.setdefault(head, set()).update(classes)
    return declared


def _load_modalities(root: Path) -> dict[str, str]:
    """Read the modality of each dataset from dataset_info.json."""
    info_path = root / "dataset_info.json"
    if not info_path.exists():
        return {}
    payload = json.loads(info_path.read_text())
    return {d["name"]: d.get("modality", "unknown")
            for d in payload.get("datasets", [])}


def _split_stem(stem: str, declared: dict[str, set[str]]) -> tuple[str, str, str] | None:
    """Split "Task03_Liver_liver_42_liver" into (dataset, anatomy, case_id).

    The dataset prefix is the first two tokens for a Medical Segmentation
    Decathlon name, such as "Task03_Liver". For other names it is the first
    token. The anatomy is the longest suffix of up to three tokens that the
    class list declares. The function returns None if no suffix matches.
    """
    tokens = stem.split("_")
    if len(tokens) < 3:
        return None

    candidate_datasets = []
    if tokens[0].startswith("Task") and len(tokens) >= 4:
        candidate_datasets.append(("_".join(tokens[:2]), tokens[2:]))
    candidate_datasets.append((tokens[0], tokens[1:]))

    for dataset, body in candidate_datasets:
        classes = declared.get(dataset)
        if not classes:
            continue
        for width in range(min(3, len(body) - 1), 0, -1):
            anatomy = "_".join(body[-width:])
            if anatomy in classes:
                return dataset, anatomy, "_".join(body[:-width])
    return None


def build_manifest(root: str | Path, split: str,
                   strict: bool = True) -> tuple[Manifest, list[str]]:
    """Scan `root/labels/<split>` and pair each mask with its image.

    Args:
        root: corpus root. It holds `images/`, `labels/` and dataset_info.json.
        split: subdirectory name, such as "train" or "test".
        strict: raise an error if a filename holds no declared anatomy. Set it
            to False to build the manifest and inspect the skipped files.

    Returns:
        (manifest, skipped). The list `skipped` holds the filenames with no
        declared anatomy. The manifest excludes those files.
    """
    root = Path(root)
    labels_dir = root / "labels" / split
    images_dir = root / "images" / split
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"no label directory at {labels_dir}")

    declared = _load_declared_classes(root)
    if not declared:
        raise FileNotFoundError(
            f"{root / 'dataset_info.json'} is absent or declares no classes. "
            "The anatomy names come from that file."
        )
    modalities = _load_modalities(root)

    entries: list[Entry] = []
    skipped: list[str] = []
    for mask_path in sorted(labels_dir.glob("*.nii*")):
        stem = mask_path.name.replace(".nii.gz", "").replace(".nii", "")
        parsed = _split_stem(stem, declared)
        if parsed is None:
            skipped.append(mask_path.name)
            continue
        dataset, anatomy, case_id = parsed
        image_path = images_dir / mask_path.name
        if not image_path.exists():
            skipped.append(mask_path.name)
            continue
        entries.append(Entry(
            case_id=case_id,
            dataset=dataset,
            anatomy=anatomy,
            modality=modalities.get(dataset, "unknown"),
            image=str(image_path.relative_to(root)),
            mask=str(mask_path.relative_to(root)),
        ))

    if strict and skipped:
        raise ValueError(
            f"{len(skipped)} files under {labels_dir} hold no anatomy that "
            f"dataset_info.json declares. The first few are: {skipped[:5]}. "
            "Add the missing class names to dataset_info.json, or pass "
            "strict=False to skip these files."
        )
    return Manifest(root=root, split=split, entries=entries), skipped


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------


def class_counts(manifest: Manifest) -> dict[str, int]:
    """Count the cases of each class key. The largest class comes first."""
    counts = Counter(entry.class_key for entry in manifest.entries)
    return dict(counts.most_common())


def check_manifest(manifest: Manifest) -> dict:
    """Summarise a manifest and count the conditions that break retrieval.

    A class of one case cannot supply an exemplar to its own query. The query
    then receives no exemplar. Check `n_singleton_classes` before you train.
    """
    counts = class_counts(manifest)
    singletons = [key for key, n in counts.items() if n < 2]
    sizes = sorted(counts.values())
    median = sizes[len(sizes) // 2] if sizes else 0
    return {
        "n_cases": len(manifest),
        "n_classes": len(counts),
        "median_class_size": median,
        "smallest_class_size": sizes[0] if sizes else 0,
        "n_singleton_classes": len(singletons),
        "singleton_classes": singletons[:20],
        "n_datasets": len({entry.dataset for entry in manifest.entries}),
        "modalities": sorted({entry.modality for entry in manifest.entries}),
    }
