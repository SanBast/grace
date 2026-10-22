"""Tests for the fusion modules and the refiner."""

from __future__ import annotations

import torch

from grace.models.fusion import ConcatFusion, FiLMFusion, build_fusion
from grace.models.refiner import GraceRefiner, RefinerConfig

SIZE = (16, 16, 16)
# A small architecture. The default configuration needs a GPU.
SMALL = RefinerConfig(channels=(8, 16, 32), strides=(2, 2), num_res_units=1)


def _inputs(batch: int = 2):
    corrupted = torch.rand(batch, 1, *SIZE).round()
    exemplar = torch.rand(batch, 1, *SIZE).round()
    image = torch.rand(batch, 1, *SIZE)
    return corrupted, exemplar, image


def test_concat_fusion_starts_as_a_pass_through():
    """The convolution starts at the identity, so the output copies the input."""
    fusion = ConcatFusion()
    corrupted, exemplar, image = _inputs()
    out = fusion(corrupted, exemplar, image)
    assert out.shape == (2, 3, *SIZE)
    assert torch.allclose(out[:, 0:1], corrupted, atol=1e-6)
    assert torch.allclose(out[:, 1:2], exemplar, atol=1e-6)


def test_film_fusion_starts_as_a_pass_through():
    """The last layer starts at zero, so the gain is 1 and the offset is 0."""
    fusion = FiLMFusion()
    corrupted, exemplar, image = _inputs()
    out = fusion(corrupted, exemplar, image)
    assert out.shape == (2, 3, *SIZE)
    assert torch.allclose(out[:, 0:1], corrupted, atol=1e-6)


def test_fusion_channel_count_follows_the_flags():
    assert build_fusion("concat").out_channels == 3
    assert build_fusion("concat", use_image=False).out_channels == 2
    assert build_fusion("concat", use_exemplar=False, use_image=False).out_channels == 1


def test_forward_gives_two_logits_for_each_voxel():
    model = GraceRefiner(SMALL)
    corrupted, exemplar, image = _inputs()
    logits = model(corrupted, exemplar, image)
    assert logits.shape == (2, 2, *SIZE)


def test_refine_gives_a_binary_mask():
    model = GraceRefiner(SMALL)
    corrupted, exemplar, image = _inputs()
    out = model.refine(corrupted, exemplar, image)
    assert out.shape == (2, 1, *SIZE)
    assert set(out.unique().tolist()).issubset({0.0, 1.0})


def test_exemplar_dropout_runs_in_training_mode_only():
    model = GraceRefiner(RefinerConfig(channels=(8, 16), strides=(2,),
                                       num_res_units=1, exemplar_dropout=1.0))
    corrupted, exemplar, image = _inputs()
    model.train()
    with_dropout = model(corrupted, exemplar, image)
    model.eval()
    without_dropout = model(corrupted, exemplar, image)
    assert not torch.allclose(with_dropout, without_dropout)


def test_save_and_load_round_trip(tmp_path):
    model = GraceRefiner(SMALL)
    path = model.save(tmp_path / "refiner.pt", extra={"epoch": 3})
    restored = GraceRefiner.load(path)

    assert restored.config == model.config
    corrupted, exemplar, image = _inputs()
    model.eval()
    restored.eval()
    assert torch.allclose(model(corrupted, exemplar, image),
                          restored(corrupted, exemplar, image), atol=1e-6)


def test_the_config_survives_a_dictionary_round_trip():
    config = RefinerConfig(fusion="film", exemplar_dropout=0.25)
    restored = RefinerConfig.from_dict(config.to_dict())
    assert restored == config


def test_the_config_ignores_unknown_keys():
    config = RefinerConfig.from_dict({"fusion": "film", "learning_rate": 0.1})
    assert config.fusion == "film"
