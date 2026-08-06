"""Feature, split and model tests — the claims root §5, §6 and §8 make about maths.

Reads ``data/raw/BTCUSDT_1h.parquet``. Writes nothing.

Several of these exist because the claim they check turned out to be false the
first time it was run: `D52a` (Rogers–Satchell is not strictly positive), `D52b`
(the Naive-RW tilt was ~2x overstated), `D52d` (the overfit check cannot pass
with dropout on). A test that only ever passed would not have found them.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import torch
from torch import nn

from itransformer_btc.config import ORIGINS
from itransformer_btc.features import (
    TARGET,
    TARGET_INDEX,
    VARIATE_ORDER,
    build_features,
    ladder_columns,
)
from itransformer_btc.model import ITransformer, ITransformerConfig
from itransformer_btc.segments import build_segments, load_bars, usable_mask
from itransformer_btc.splits import build_origin_tensors
from itransformer_btc.train import scale_invariance_check, set_seed


@pytest.fixture(scope="session")
def raw() -> pl.DataFrame:
    return usable_mask(load_bars())


@pytest.fixture(scope="session")
def feats(raw: pl.DataFrame) -> pl.DataFrame:
    return build_features(raw)


# -- the twelve variates — root §5.1, §5.2 -----------------------------------


def test_twelve_variates_in_ladder_order(feats: pl.DataFrame) -> None:
    """Rung K is the first K columns, so ``r`` is channel 0 at every rung."""
    assert len(VARIATE_ORDER) == 12
    assert VARIATE_ORDER[TARGET_INDEX] == TARGET
    assert list(feats.columns[2:]) == list(VARIATE_ORDER)
    for k in (1, 4, 8, 12):
        assert ladder_columns(k) == list(VARIATE_ORDER[:k])
    with pytest.raises(ValueError):
        ladder_columns(16)


def test_every_variate_is_finite(feats: pl.DataFrame) -> None:
    """`D14` — the segment law is what makes each variate a total function."""
    for name in VARIATE_ORDER:
        col = feats.get_column(name)
        assert col.is_finite().all(), name
        assert col.null_count() == 0, name


def test_f3_has_two_degrees_of_freedom(feats: pl.DataFrame) -> None:
    """Root §5.1 — the third intensity member is the difference of the first two.

    Exact to the last bit, not merely correlated: ``log(q/n) = log q - log n``.
    That is why F3 contributes 2 dof to K_eff and not 3, and why
    ``log_mean_trade_size`` sits in the deliberately redundant K=12 rung.
    """
    residual = feats.get_column("log_mean_trade_size") - (
        feats.get_column("log_quote_volume") - feats.get_column("log_trade_count")
    )
    assert float(residual.abs().max()) < 1e-9


def test_rogers_satchell_is_not_strictly_positive(raw: pl.DataFrame) -> None:
    """`D52a`. Root §5.1's claim about all three F2 estimators is false for one.

    RS vanishes on a shadowless bar — H equal to one of O/C and L equal to the
    other. Such a bar has H > L and passes the segment law; it is a marubozu,
    not a degenerate bar. Parkinson and Garman-Klass really are strictly
    positive, which is why the stabiliser applies to RS alone.
    """
    bars = raw.filter(pl.col("usable"))
    rs = (
        (pl.col("high") / pl.col("close")).log() * (pl.col("high") / pl.col("open")).log()
        + (pl.col("low") / pl.col("close")).log() * (pl.col("low") / pl.col("open")).log()
    )
    zeros = bars.select(rs.alias("rs")).filter(pl.col("rs") <= 0)
    assert zeros.height == 33, f"expected 33 shadowless bars, found {zeros.height}"


def test_the_stabiliser_lands_inside_the_measured_support(feats: pl.DataFrame) -> None:
    """`D52a` — kappa = 1e-9 chosen so log kappa is not an out-of-support spike.

    A hard floor far below support (1e-12 gives -27.6, about -11 sigma) would
    distort the instance normalisation of every window containing one and would
    smuggle a categorical marubozu flag into a continuous variate.
    """
    col = feats.get_column("log_rogers_satchell")
    floor = float(np.log(1e-9))
    assert float(col.min()) == pytest.approx(floor, abs=1e-6)
    assert float(col.quantile(0.001)) > floor, "the floor should sit in the low tail"
    assert int((col <= floor + 1e-9).sum()) <= 40


def test_features_drop_exactly_one_bar_per_segment(
    raw: pl.DataFrame, feats: pl.DataFrame
) -> None:
    """`D52c`. ``r`` is per segment, so each segment's first bar has no predecessor.

    Computing ``r`` on a concatenated series instead would inject cross-gap
    returns into mu_g and sigma_g before any window is excluded — the 33-hour
    2018-02-08 outage booked as a one-hour return.
    """
    n_segments = len(build_segments(raw))
    assert int(raw.get_column("usable").sum()) - feats.height == n_segments


# -- splits — root §8.1, §8.2 ------------------------------------------------


def test_purge_holds_and_scaler_sees_training_only(feats: pl.DataFrame) -> None:
    """FATAL, `D24`. Targets may not cross a boundary; inputs may."""
    for origin in ORIGINS[:3]:
        t = build_origin_tensors(feats, origin, 8)
        assert t.train.ts.max() < int(origin.val_start.timestamp() * 1000)
        assert len(t.val) > 0
        assert t.scaler.columns == tuple(ladder_columns(8))


def test_test_blocks_use_origin_semantics(feats: pl.DataFrame) -> None:
    """`D51b` — 720 forecast origins per clean block, never 601."""
    counts = [
        len(split)
        for origin in ORIGINS
        for split in build_origin_tensors(feats, origin, 1).test_blocks
    ]
    assert max(counts) == 720
    assert sum(1 for n in counts if n == 720) >= 70


def test_naive_rw_is_the_drift_free_baseline(feats: pl.DataFrame) -> None:
    """`D31`/`D52b`. ``y_z = -mu_g/sigma_g``, and the tilt changes sign.

    ``y_z = 0`` would silently mean ``r_hat = mu_g``, a constant-drift model
    wearing the EMH baseline's name. The measured magnitude is ~2x smaller than
    §7 first claimed, but it flips sign across origins, so it is not a constant
    a reader could subtract.
    """
    ratios = [
        build_origin_tensors(feats, origin, 1).scaler.target_mu_over_sigma
        for origin in ORIGINS
    ]
    assert max(abs(r) for r in ratios) < 0.02, "if this grows, revisit D52b's numbers"
    assert min(ratios) < 0 < max(ratios), "the tilt must change sign across origins"


# -- the model — root §6.1, §6.2, §6.3 ---------------------------------------


def test_parameter_count_is_identical_at_every_rung() -> None:
    """Root §6.2 — K changes the token count, not a single weight shape."""
    cfg = ITransformerConfig()
    counts = {k: ITransformer(cfg).n_parameters() for k in (1, 4, 8, 12)}
    assert len(set(counts.values())) == 1
    assert next(iter(counts.values())) == 280_472


def test_k1_runs_and_is_not_a_bare_identity() -> None:
    """`D50`. At N=1 softmax returns weight 1, but W_V, W_O and the residual remain."""
    set_seed(42)
    out = ITransformer(ITransformerConfig())(torch.randn(4, 96, 1))
    assert out.shape == (4, 24)
    assert torch.isfinite(out).all()


def test_uniform_attention_arm_differs_without_changing_capacity() -> None:
    """`D50` — the control must isolate what attention selects, not capacity."""
    base = ITransformerConfig()
    uniform = ITransformerConfig(uniform_attention=True)
    assert ITransformer(base).n_parameters() == ITransformer(uniform).n_parameters()

    x = torch.randn(4, 96, 8)
    set_seed(42)
    a = ITransformer(base).eval()(x)
    set_seed(42)
    b = ITransformer(uniform).eval()(x)
    assert not torch.allclose(a, b)


def test_use_norm_scale_invariance() -> None:
    """FATAL, `D03`. ``MSE(c x)/c^2 == MSE(x)``, not ``MSE(c x) == MSE(x)``.

    The source specification's version cannot pass: the target is a channel of
    the same array, so it scales too and the loss scales by c squared.
    """
    set_seed(42)
    model = ITransformer(ITransformerConfig()).eval()
    x, y = torch.randn(32, 96, 8), torch.randn(32, 24)
    base, scaled = scale_invariance_check(model, x, y, c=100.0)
    assert abs(base - scaled) / base < 1e-3


def test_use_norm_off_breaks_the_invariant() -> None:
    """The invariant must have teeth: it has to fail when ``use_norm`` is off.

    Root §6.3 — if the flag is ever disabled the whole scaler argument collapses
    and the scaler again affects learning. A check that passed either way would
    not detect that.
    """
    set_seed(42)
    model = ITransformer(ITransformerConfig(use_norm=False)).eval()
    x, y = torch.randn(32, 96, 8), torch.randn(32, 24)
    base, scaled = scale_invariance_check(model, x, y, c=100.0)
    assert abs(base - scaled) / base > 1e-2


def test_single_batch_overfits_with_dropout_off() -> None:
    """`D52d`. Root §16's plumbing check — but only with ``dropout=0.0``.

    With the configured 0.1 still active the loss floors near 7e-2 and a reader
    following the instruction literally concludes the plumbing is broken.
    """
    set_seed(42)
    model = ITransformer(ITransformerConfig(dropout=0.0)).train()
    x, y = torch.randn(8, 96, 8), torch.randn(8, 24)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(200):
        opt.zero_grad(set_to_none=True)
        loss = nn.functional.mse_loss(model(x), y)
        loss.backward()
        opt.step()
    assert loss.item() < 1e-3, f"plumbing broken: {loss.item():.3e}"
