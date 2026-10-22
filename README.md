# GRACE

Retrieval-augmented refinement of 3D medical segmentation masks.

GRACE takes a segmentation mask, its image, and one retrieved mask of the same
anatomy from a pool of verified cases. It gives a corrected mask. One set of
weights serves every anatomy and modality in the corpus.

The package holds the refiner, the degradation schedule that trains it, the
retrieval index, the evaluation protocol and an input quality gate.

## Install

```bash
git clone https://github.com/robustml-eurecom/grace.git
cd grace
pip install -e .
```

Requirements: Python 3.10 or later, PyTorch 2.1 or later, MONAI, nibabel,
scipy. A CPU is sufficient for inference on one case. Training needs a GPU with
16 GB or more.

Verify the installation on a synthetic corpus:

```bash
python examples/quickstart.py --output-dir /tmp/grace-quickstart
```

## How it works

The refiner is a 3D U-Net. It reads three channels: the mask to correct, a
retrieved mask of the same class, and the image. A fusion module combines the
three before the first encoder block. The loss is Dice and cross entropy
against the ground truth.

Training needs pairs of a damaged mask and a correct mask. Such pairs are rare,
so `grace.corruption` makes them. It degrades a ground truth mask with
operations on its signed distance transform: boundary deformation at two
spatial scales, erosion, dilation, false positive blobs, a split along a smooth
surface, and a run of dropped slices. Five severity presets set the
probabilities and the magnitudes. Each training sample draws its own severity.

Retrieval groups the pool by the class key `<dataset>/<anatomy>`. For a query
it returns one case of the same class, either a random one or the nearest
neighbour from a precomputed table. Exemplar dropout zeroes the retrieved
channel for a fraction of the training samples, so the refiner also works when
retrieval returns nothing.

## Data layout

Put the images and the labels in two parallel trees. Each label file holds one
binary mask. Declare the anatomy names of each dataset in `dataset_info.json`.

```
corpus/
    dataset_info.json
    images/train/Task03_Liver_case00_liver.nii.gz
    labels/train/Task03_Liver_case00_liver.nii.gz
    images/test/...
    labels/test/...
```

```json
{
  "datasets": [
    {"name": "Task03_Liver", "modality": "CT", "classes": ["liver", "liver_tumour"]},
    {"name": "ACDC", "modality": "MRI", "classes": ["left_ventricle", "right_ventricle"]}
  ]
}
```

A filename holds `<dataset>_<case>_<anatomy>`. The anatomy can hold several
tokens, such as `liver_tumour`.

## Use

### 1. Build the manifests

```bash
grace manifest --root corpus --split train --output manifests/train.json
grace manifest --root corpus --split test  --output manifests/test.json
```

A manifest lists each case with its dataset, anatomy, modality and file paths.
The command resolves the anatomy against `dataset_info.json` and prints the
class sizes.

Read that report. Retrieval groups the cases by the key `<dataset>/<anatomy>`,
and a class of one case supplies no exemplar. The command warns when it finds
such a class.

### 2. Build the retrieval table (optional)

```bash
grace nn-table --manifest manifests/train.json --pool manifests/train.json \
               --output manifests/nn_table.json
```

The table holds the nearest same-class neighbour of each case, by the cosine
similarity of a coarse shape embedding. Pass it to the `nearest` exemplar mode.
Without a table, the `random` mode takes a random same-class case.

### 3. Train

```bash
grace train --train-manifest manifests/train.json \
            --val-manifest manifests/test.json \
            --output-dir runs/grace_u \
            --severity mixed --exemplar-mode random \
            --epochs 100 --batch-size 4
```

The run writes `best.pt`, `last.pt`, `config.json` and `history.json`.

The validation score of one epoch carries noise, because each epoch draws new
degradations. Use `grace evaluate` for a number that you report.

### 4. Evaluate

```bash
grace evaluate --checkpoint runs/grace_u/best.pt \
               --manifest manifests/test.json \
               --pool manifests/train.json \
               --severities default mid strong \
               --output results/grace_u.json
```

The file holds a summary, a block for each class, the harm analysis, and one
record for each case. Each record holds the Dice, HD95 and boundary F1 before
and after refinement.

Keep the records. The gate curve and any later grouping read them without a
second inference pass.

The evaluation seeds the degradation per case, so two checkpoints compare on
identical inputs.

### 5. Select a gate

```bash
grace gate --results results/grace_u.json --severity strong
```

A gate refines a case only when the estimated quality of its input is below a
threshold. The curve shows how many cases the gate refines, the mean delta over
all cases, and how many cases it harms.

## Python API

```python
from grace.data.dataset import RefinementDataset
from grace.data.manifest import load_manifest
from grace.evaluate import evaluate, summarise
from grace.models.refiner import GraceRefiner
from grace.retrieval.index import ExemplarIndex

pool = load_manifest("manifests/train.json")
queries = load_manifest("manifests/test.json")

index = ExemplarIndex(pool)
dataset = RefinementDataset(queries, index=index, severity="strong", seed=42)
model = GraceRefiner.load("runs/grace_u/best.pt")

records = evaluate(model, dataset)
print(summarise(records)["delta_dice_mean"])
```

Refine one mask:

```python
import torch

refined = model.refine(
    corrupted=torch.rand(1, 1, 96, 96, 96).round(),
    exemplar=torch.rand(1, 1, 96, 96, 96).round(),
    image=torch.rand(1, 1, 96, 96, 96),
)
```

## Experiments

| Experiment | Flags |
|---|---|
| Severity sweep | `--severities default mid strong very_strong extreme` |
| No exemplar, the control for retrieval | `--exemplar-mode none` |
| Wrong class query | `--exemplar-mode wrong-class` |
| Nearest neighbour retrieval | `--exemplar-mode nearest --nn-table manifests/nn_table.json` |
| Hold a dataset out of the pool | `--exclude-datasets ACDC` |
| One organ only | `--anatomies liver` |

The `extreme` severity is for evaluation only. No training schedule includes
it.

For a holdout experiment, exclude the dataset in both commands. Pass
`--exclude-datasets` to `grace train` to remove it from the training data and
from the pool. Pass the same flag to `grace evaluate` to keep it out of the
pool at test time.

## Package layout

```
grace/
    corruption.py        severity presets and the degradation operations
    metrics.py           Dice, HD95, boundary F1
    config.py            TrainConfig and EvalConfig
    train.py             the training loop
    evaluate.py          per-case records, the summary, the harm analysis
    gate.py              the input quality gate
    cli.py               the command line interface
    data/
        manifest.py      the sample manifest and its audit
        volumes.py       NIfTI input and resampling
        dataset.py       the training and evaluation dataset
        synthetic.py     a synthetic corpus for verification
    retrieval/
        index.py         exemplar selection
        embeddings.py    shape embeddings and the neighbour table
    models/
        fusion.py        the concat and FiLM fusion modules
        refiner.py       the refiner and its checkpoint format
```

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The tests build a synthetic corpus of balls. They need no download and no GPU.

## License

MIT. See LICENSE.
