"""Figure 5's capture, and the schedule `D62c` overrides.

The load-bearing property is negative: turning capture on must change **nothing**
about what the model produces. The attention arm re-runs the same seeds as the
main grid, so if capture perturbed the numerics the maps would describe a second
model that merely resembles the one whose forecasts the paper reports.
"""

from __future__ import annotations

from dataclasses import asdict, fields

import numpy as np
import polars as pl
import pytest

torch = pytest.importorskip("torch")

from itransformer_btc.attention import (  # noqa: E402
    TERCILES,
    lookback_volatility,
    tercile_edges,
    tercile_maps,
)
from itransformer_btc.config import ORIGINS  # noqa: E402
from itransformer_btc.features import build_features  # noqa: E402
from itransformer_btc.model import (  # noqa: E402
    ITransformerConfig,
    LongScheduleConfig,
)
from itransformer_btc.runner import RunCell  # noqa: E402
from itransformer_btc.segments import load_bars, usable_mask  # noqa: E402
from itransformer_btc.splits import build_origin_tensors  # noqa: E402
from itransformer_btc.train import TrainSchedule  # noqa: E402


@pytest.fixture(scope="module")
def tensors():
    feats = build_features(usable_mask(load_bars()))
    return build_origin_tensors(feats, ORIGINS[0], 8)


# -- capture must be inert ---------------------------------------------------


def test_capture_is_not_a_config_field() -> None:
    """``write_artifacts`` records ``asdict(cfg)``, so a field here would appear in
    every iTransformer ``meta/*.json`` and change bytes the 684-run grid has
    already produced. ``model.py`` states the rule; this enforces it."""
    names = {f.name for f in fields(ITransformerConfig())}
    assert "capture" not in names
    assert "last_weights" not in names


def test_capture_is_off_by_default() -> None:
    layer = ITransformerConfig().build().layers[0].attention
    assert layer.capture is False
    assert layer.last_weights is None


def test_capture_does_not_change_a_single_prediction_bit() -> None:
    torch.manual_seed(42)
    model = ITransformerConfig(pred_len=24).build().eval()
    x = torch.randn(16, 96, 8)
    with torch.no_grad():
        before = model(x).clone()
    for layer in model.layers:
        layer.attention.capture = True
    with torch.no_grad():
        after = model(x)
    assert torch.equal(before, after)
    assert model.layers[0].attention.last_weights.shape == (16, 8, 8)


def test_captured_weights_are_row_stochastic() -> None:
    model = ITransformerConfig(pred_len=24).build().eval()
    for layer in model.layers:
        layer.attention.capture = True
    with torch.no_grad():
        model(torch.randn(4, 96, 8))
    weights = model.layers[0].attention.last_weights
    assert torch.allclose(weights.sum(dim=-1), torch.ones(4, 8), atol=1e-5)


def test_the_uniform_arm_captures_nothing_because_it_computes_nothing() -> None:
    """`D50`'s arm replaces the softmax with a mean, so there is no weight matrix
    to record --- and a figure built from one would be inventing it."""
    model = ITransformerConfig(pred_len=24, uniform_attention=True).build().eval()
    for layer in model.layers:
        layer.attention.capture = True
    with torch.no_grad():
        model(torch.randn(4, 96, 8))
    assert model.layers[0].attention.last_weights is None


# -- the regimes -------------------------------------------------------------


def test_tercile_edges_split_into_three_roughly_equal_bins() -> None:
    rng = np.random.default_rng(0)
    vol = rng.random(9_000)
    low, high = tercile_edges(vol)
    bins = np.digitize(vol, [low, high])
    counts = [int((bins == b).sum()) for b in range(3)]
    assert all(abs(c - 3_000) < 100 for c in counts), counts


def test_lookback_volatility_reads_the_target_channel(tensors) -> None:
    split = tensors.test_blocks[0]
    vol = lookback_volatility(split)
    assert vol.shape == (len(split),)
    assert (vol > 0).all()
    assert np.allclose(vol, split.x[:, :, 0].std(axis=1), atol=1e-6)


def test_tercile_maps_emit_one_row_per_tercile_layer_and_variate_pair(tensors) -> None:
    model = ITransformerConfig(pred_len=24).build()
    frame = tercile_maps(model, tensors, torch.device("cpu"))
    assert set(frame.get_column("tercile").unique().to_list()) == set(TERCILES)
    assert frame.height == len(TERCILES) * 2 * 8 * 8, "3 terciles x 2 layers x 8 x 8"
    assert set(frame.columns) == {
        "tercile", "layer", "i", "j", "weight", "n_windows", "vol_low", "vol_high",
    }
    # Rows of a softmax, averaged: still a distribution.
    per_row = frame.group_by(["tercile", "layer", "i"]).agg(pl.col("weight").sum())
    assert np.allclose(per_row.get_column("weight").to_numpy(), 1.0, atol=1e-5)
    assert (frame.get_column("n_windows").to_numpy() > 0).all()


def test_tercile_maps_leave_capture_off_afterwards(tensors) -> None:
    """A sweep must not leave a model quietly accumulating detached weight tensors
    for the rest of a twelve-hour session."""
    model = ITransformerConfig(pred_len=24).build()
    tercile_maps(model, tensors, torch.device("cpu"))
    for layer in model.layers:
        assert layer.attention.capture is False
        assert layer.attention.last_weights is None


# -- `D62c`'s schedule, and `D62b`'s capacity --------------------------------


def test_default_schedule_reproduces_the_grid_exactly() -> None:
    """``train_one``'s former defaults and ``TrainSchedule``'s must agree, or the
    refactor silently re-specifies all 684 runs already on disk."""
    schedule = TrainSchedule()
    assert (
        schedule.max_epochs,
        schedule.patience,
        schedule.lr,
        schedule.lr_halve_every,
    ) == (30, 5, 1e-4, 4)
    assert ITransformerConfig().schedule() == schedule


def test_schedule_is_a_method_so_config_bytes_do_not_move() -> None:
    """``asdict`` sees fields, not methods. A schedule field would rewrite
    ``meta['config']`` for every future run and make the grid's own output look
    like a different vintage --- the false positive ``code_sha256`` exists to
    prevent."""
    assert "schedule" not in {f.name for f in fields(ITransformerConfig())}
    assert asdict(ITransformerConfig()) == asdict(LongScheduleConfig())


def test_long_schedule_widens_the_schedule_and_nothing_else() -> None:
    schedule = LongScheduleConfig().schedule()
    assert (schedule.max_epochs, schedule.patience, schedule.lr_halve_every) == (
        60, 10, 8,
    )
    assert schedule.lr == TrainSchedule().lr
    assert LongScheduleConfig().d_ff == ITransformerConfig().d_ff


def test_capacity_arm_widens_d_ff_and_only_d_ff() -> None:
    cell = RunCell("capacity", 1, 12, 24, 42)
    cfg = cell.model_config()
    assert cfg.d_ff == 512
    assert (cfg.d_model, cfg.e_layers, cfg.n_heads, cfg.dropout) == (128, 2, 8, 0.1)
    assert cfg.schedule() == TrainSchedule()
    assert cell.run_id == "itrc_o01_K12_H024_s42"


def test_capacity_arm_really_is_larger() -> None:
    """Root §6.2 pre-registers this run so a flat 8->12 rung cannot be read as an
    under-tuning artefact. An arm with the same parameter count would answer
    nothing."""
    base = ITransformerConfig(pred_len=24).build().n_parameters()
    wide = ITransformerConfig(pred_len=24, d_ff=512).build().n_parameters()
    assert wide > base
    # Two layers, each gaining (d_model x d_ff + d_ff) + (d_ff x d_model) weights.
    assert wide - base == 2 * ((128 * 256 + 256) + (256 * 128))


def test_attention_arm_uses_the_main_grid_configuration() -> None:
    """The arm exists to record what the main grid's model attends to, so its
    configuration must be the main grid's --- otherwise the maps belong to a
    different model."""
    attention_cfg = RunCell("attention", 1, 8, 24, 42).model_config()
    main_cfg = RunCell("main", 1, 8, 24, 42).model_config()
    assert asdict(attention_cfg) == asdict(main_cfg)
    assert attention_cfg.schedule() == main_cfg.schedule()
