"""GRACE: retrieval-augmented refinement of 3D medical segmentation masks.

The refiner takes a segmentation mask, an image and one retrieved mask of the
same anatomy. It gives a corrected mask. One set of weights serves every
anatomy and modality in the corpus.

Typical use:

    from grace.data.manifest import load_manifest
    from grace.data.dataset import RefinementDataset
    from grace.models.refiner import GraceRefiner
    from grace.retrieval.index import ExemplarIndex
    from grace.evaluate import evaluate, summarise

    pool = load_manifest("manifests/train.json")
    queries = load_manifest("manifests/test.json")
    index = ExemplarIndex(pool)
    dataset = RefinementDataset(queries, index=index, severity="strong", seed=42)
    model = GraceRefiner.load("checkpoints/grace_u.pt")
    print(summarise(evaluate(model, dataset)))

See README.md for the command line interface.
"""

__version__ = "0.1.0"

from grace.config import EvalConfig, TrainConfig
from grace.models.refiner import GraceRefiner, RefinerConfig

__all__ = [
    "__version__",
    "EvalConfig",
    "GraceRefiner",
    "RefinerConfig",
    "TrainConfig",
]
