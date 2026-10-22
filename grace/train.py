"""Training loop for the refiner.

One run writes four files to the output directory:

    best.pt        the weights with the highest validation Dice
    last.pt        the weights after the final epoch
    config.json    the configuration of the run
    history.json   the loss and the validation scores of each epoch

The validation score of one epoch carries noise, because each epoch draws new
degradations. Use `grace.evaluate` for any number that you report.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from monai.losses import DiceCELoss
from torch.utils.data import DataLoader, Subset

from grace.config import TrainConfig
from grace.data.dataset import RefinementDataset
from grace.data.manifest import load_manifest
from grace.metrics import dice_batch
from grace.models.refiner import GraceRefiner, RefinerConfig
from grace.retrieval.index import ExemplarIndex, load_nn_table

__all__ = ["train", "build_datasets"]

logger = logging.getLogger("grace.train")


def build_datasets(config: TrainConfig) -> tuple[RefinementDataset, RefinementDataset, ExemplarIndex]:
    """Make the training and validation datasets, and the exemplar pool.

    The pool comes from the training manifest. Validation queries therefore
    retrieve from the training corpus, as at deployment.
    """
    train_manifest = load_manifest(config.train_manifest)
    val_manifest = load_manifest(config.val_manifest)
    if config.exclude_datasets:
        train_manifest = train_manifest.filter(exclude_datasets=config.exclude_datasets)

    nn_table = load_nn_table(config.nn_table) if config.nn_table else None
    index = ExemplarIndex(train_manifest, nn_table=nn_table,
                          exclude_datasets=config.exclude_datasets)
    logger.info("exemplar pool: %s", index.report())

    train_dataset = RefinementDataset(
        train_manifest, index=index, size=config.size,
        severity=config.severity, exemplar_mode=config.exemplar_mode,
        use_image=config.use_image, seed=None,
    )
    val_dataset = RefinementDataset(
        val_manifest, index=index, size=config.size,
        severity=config.val_severity, exemplar_mode=config.exemplar_mode,
        use_image=config.use_image, seed=config.seed,
    )
    return train_dataset, val_dataset, index


def _subsample(dataset, count: int, seed: int):
    """Take `count` random items from a dataset."""
    if count <= 0 or count >= len(dataset):
        return dataset
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(dataset), size=count, replace=False)
    return Subset(dataset, sorted(int(p) for p in picks))


def _to_device(batch: dict, device: torch.device) -> tuple:
    """Move the tensors of one batch to the device."""
    return (
        batch["corrupted"].to(device, non_blocking=True),
        batch["exemplar"].to(device, non_blocking=True),
        batch["image"].to(device, non_blocking=True),
        batch["mask"].to(device, non_blocking=True),
    )


@torch.no_grad()
def validate(model: GraceRefiner, loader: DataLoader,
             device: torch.device) -> tuple[float, float]:
    """Give the mean refined Dice and the mean Dice of the degraded input."""
    model.eval()
    refined_scores, floor_scores = [], []
    for batch in loader:
        corrupted, exemplar, image, mask = _to_device(batch, device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(corrupted, exemplar, image)
        prediction = logits.argmax(dim=1, keepdim=True).float()
        refined_scores.append(dice_batch(prediction, mask).cpu())
        floor_scores.append(dice_batch(corrupted, mask).cpu())
    model.train()
    if not refined_scores:
        return 0.0, 0.0
    return (float(torch.cat(refined_scores).mean()),
            float(torch.cat(floor_scores).mean()))


def train(config: TrainConfig,
          refiner_config: RefinerConfig | None = None) -> dict:
    """Run one training job.

    Args:
        config: the settings of the run.
        refiner_config: the architecture. The default is `RefinerConfig()`.

    Returns:
        The history dictionary. The function also writes it to the output
        directory.
    """
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, val_dataset, _ = build_datasets(config)
    val_dataset = _subsample(val_dataset, config.max_val_cases, config.seed + 1)
    logger.info("train cases=%d  val cases=%d", len(train_dataset), len(val_dataset))

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, drop_last=True, pin_memory=pin_memory,
        persistent_workers=config.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=pin_memory,
    )

    model = GraceRefiner(refiner_config).to(device)
    logger.info("refiner parameters: %.2f M", model.n_parameters() / 1e6)

    optimiser = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=config.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp and device.type == "cuda")
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True, include_background=False)

    payload = {"train": config.to_dict(), "refiner": model.config.to_dict()}
    (output_dir / "config.json").write_text(json.dumps(payload, indent=2))

    history = {"train_loss": [], "val_dice": [], "val_floor": [], "epoch_time": []}
    best_dice = -1.0

    for epoch in range(config.epochs):
        model.train()
        started = time.time()
        total_loss, n_steps = 0.0, 0

        for step, batch in enumerate(train_loader):
            corrupted, exemplar, image, mask = _to_device(batch, device)
            optimiser.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16,
                                enabled=config.amp and device.type == "cuda"):
                logits = model(corrupted, exemplar, image)
                loss = loss_fn(logits, mask.long())
            scaler.scale(loss).backward()
            if config.grad_clip > 0:
                scaler.unscale_(optimiser)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimiser)
            scaler.update()

            total_loss += float(loss.item())
            n_steps += 1
            if (step + 1) % config.log_every == 0:
                logger.info("epoch %d step %d/%d loss=%.4f",
                            epoch, step + 1, len(train_loader), loss.item())

        scheduler.step()
        epoch_loss = total_loss / max(n_steps, 1)
        history["train_loss"].append(epoch_loss)
        history["epoch_time"].append(round(time.time() - started, 1))
        logger.info("epoch %d loss=%.4f (%.0fs)", epoch, epoch_loss,
                    history["epoch_time"][-1])

        if (epoch + 1) % config.eval_every == 0:
            refined, floor = validate(model, val_loader, device)
            history["val_dice"].append([epoch, refined])
            history["val_floor"].append([epoch, floor])
            logger.info("epoch %d val refined=%.4f floor=%.4f delta=%+.4f",
                        epoch, refined, floor, refined - floor)
            if refined > best_dice:
                best_dice = refined
                model.save(output_dir / "best.pt",
                           extra={"epoch": epoch, "val_dice": refined})

        model.save(output_dir / "last.pt", extra={"epoch": epoch})
        (output_dir / "history.json").write_text(json.dumps(history, indent=2))

    logger.info("done. best validation Dice=%.4f", best_dice)
    return history
