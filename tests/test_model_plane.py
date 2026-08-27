"""Feature, split and model tests — the claims root §5, §6 and §8 make about maths.

Reads ``data/raw/BTCUSDT_1h.parquet``. Writes nothing.

Several of these exist because the claim they check turned out to be false the
first time it was run: `D52a` (Rogers–Satchell is not strictly positive), `D52b`
(the Naive-RW tilt was ~2x overstated), `D52d` (the overfit check cannot pass
with dropout on). A test that only ever passed would not have found them.
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest
import torch
from torch import nn

from itransformer_btc.baselines import (
    RIDGE_ALPHAS,
    DLinearConfig,
    LSTMConfig,
    NaiveConfig,
    PatchTSTConfig,
    RidgeConfig,
    assert_baseline_alignment,
)
from itransformer_btc.config import ORIGINS, PRED_LEN, SEQ_LEN
from itransformer_btc.features import (
    TARGET,
    TARGET_INDEX,
    VARIATE_ORDER,
    build_features,
    ladder_columns,
)
from itransformer_btc.model import ITransformer, ITransformerConfig
from itransformer_btc.segments import build_segments, load_bars, usable_mask
from itransformer_btc.splits import (
    OriginTensors,
    Scaler,
    SplitTensors,
    build_origin_tensors,
)
from itransformer_btc.train import (
    RunSpec,
    scale_invariance_check,
    set_seed,
    write_artifacts,
)


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


# -- the §7 baselines — `D56` ------------------------------------------------


def _synthetic_tensors(
    n_train: int = 200, n_val: int = 64, k: int = 8, *, signal: bool = False
) -> OriginTensors:
    """An ``OriginTensors`` with no data behind it, for the plumbing tests.

    Real windows cost seconds to build and prove nothing extra here: every test
    below is about a model's shapes, its objective or its selection rule, none of
    which reads a timestamp. ``signal=True`` makes the target a fixed linear
    function of the window plus noise, so ridge's validation curve has an
    interior minimum; with ``signal=False`` the best predictor is the training
    mean and alpha runs to the top of the grid, which is its own test.
    """
    rng = np.random.default_rng(0)
    beta = rng.standard_normal((SEQ_LEN * k, PRED_LEN)) / (SEQ_LEN * k) ** 0.5

    def split(n: int, t0: int) -> SplitTensors:
        x = rng.standard_normal((n, SEQ_LEN, k)).astype(np.float32)
        y_all = rng.standard_normal((n, PRED_LEN, k)).astype(np.float32)
        if signal:
            y_all[:, :, TARGET_INDEX] = (
                x.reshape(n, -1) @ beta + 0.5 * rng.standard_normal((n, PRED_LEN))
            ).astype(np.float32)
        return SplitTensors(
            x=x,
            y=np.ascontiguousarray(y_all[:, :, TARGET_INDEX]),
            y_all=y_all,
            ts=np.arange(t0, t0 + n, dtype=np.int64) * 3_600_000,
        )

    columns = list(VARIATE_ORDER[:k])
    return OriginTensors(
        origin=ORIGINS[0],
        k=k,
        scaler=Scaler(np.zeros(k), np.ones(k), tuple(columns)),
        train=split(n_train, 0),
        val=split(n_val, n_train),
        test_blocks=(split(32, n_train + n_val),),
        block_labels=(1,),
    )


def test_baselines_forecast_the_target_channel() -> None:
    """Every model writes ``(B, H)`` to ``preds/``, whatever its ``forward`` returns.

    `D56`: the channel-independent baselines carry their published all-channel
    objective, so ``forward`` is ``(B, H, N)`` — but root §10.4's prediction file
    holds one channel for every model, and ``forecast_target`` is where each one
    says which.
    """
    x = torch.randn(4, SEQ_LEN, 8)
    for cfg in (RidgeConfig(k=8), DLinearConfig(), PatchTSTConfig()):
        model = cfg.build().eval()
        assert model.forecast_target(x).shape == (4, PRED_LEN)
    for cfg in (DLinearConfig(), PatchTSTConfig()):
        model = cfg.build().eval()
        assert model(x).shape == (4, PRED_LEN, 8)
        assert torch.equal(model.forecast_target(x), model(x)[:, :, TARGET_INDEX])


def test_channel_independent_baselines_are_channel_independent() -> None:
    """DLinear and PatchTST must ignore the other channels **at prediction time**.

    That is the architecture's claim, and it is what makes their K label mean
    something different from the ladder's: the other seven variates reach the
    target's forecast only through weights shared across channels and supervised
    on all of them. Hence ``loss_target() == "all"`` — trained on the target
    channel alone these would be K=1 wearing a K=8 label, which is `D40`'s
    collapse and would quietly return the paper's central architectural
    comparison to univariate-versus-multivariate.
    """
    set_seed(42)
    x = torch.randn(4, SEQ_LEN, 8)
    disturbed = x.clone()
    disturbed[:, :, 1:] = torch.randn(4, SEQ_LEN, 7)

    for cfg in (DLinearConfig(), PatchTSTConfig()):
        assert cfg.loss_target() == "all"
        assert cfg.channel_independent is True
        model = cfg.build().eval()
        with torch.no_grad():
            assert torch.allclose(
                model.forecast_target(x), model.forecast_target(disturbed), atol=1e-6
            )


def test_ridge_is_multivariate_at_prediction_time() -> None:
    """`D17`'s whole point: ridge reads every one of the ``L x K`` inputs.

    The contrast with the test above is the reason both baselines exist. Ridge
    answers "does the *information* help"; a channel-independent transformer
    cannot, because it never sees two variates at once.
    """
    set_seed(42)
    cfg = RidgeConfig(k=8)
    model = cfg.build().eval()
    with torch.no_grad():
        model.weight.normal_()
    assert model.weight.shape == (SEQ_LEN * 8, PRED_LEN)
    assert model.n_parameters() == SEQ_LEN * 8 * PRED_LEN + PRED_LEN

    x = torch.randn(4, SEQ_LEN, 8)
    disturbed = x.clone()
    disturbed[:, :, 1:] = torch.randn(4, SEQ_LEN, 7)
    with torch.no_grad():
        assert not torch.allclose(
            model.forecast_target(x), model.forecast_target(disturbed)
        )


def test_ridge_selects_alpha_on_validation() -> None:
    """Root §11 — the one hyperparameter this study selects, and where.

    Re-fitting with the chosen alpha as the *only* candidate has to reproduce the
    reported validation MSE exactly. That checks the selection without
    re-implementing it, which a hand-rolled argmin in the test would do and would
    then agree with a broken implementation.
    """
    tensors = _synthetic_tensors(signal=True)
    spec = RunSpec("rdg", 1, 8, PRED_LEN, 42)

    model, resolved, outcome = RidgeConfig(k=8).fit(
        tensors, spec, device=torch.device("cpu")
    )
    assert resolved.alpha in RIDGE_ALPHAS
    assert RidgeConfig(k=8).alpha is None, "the input config must not be mutated"
    # A solve, not a loop — and that is what tells Table 3 why this row has no
    # epochs-to-stop rather than a missing one.
    assert outcome.epochs_run == 0
    assert model.n_parameters() == SEQ_LEN * 8 * PRED_LEN + PRED_LEN

    _, again, single = RidgeConfig(k=8, alphas=(resolved.alpha,)).fit(
        tensors, spec, device=torch.device("cpu")
    )
    assert again.alpha == resolved.alpha
    assert single.best_val_mse == pytest.approx(outcome.best_val_mse, rel=1e-12)


def test_ridge_warns_when_alpha_pins_to_the_grid_edge() -> None:
    """With no signal the best linear predictor is the training mean.

    Alpha then runs to the top of the grid — a finding, not a failure. It is
    warned about because a boundary selection is also what an unbracketed grid
    looks like, and the number alone cannot tell the two apart.
    """
    tensors = _synthetic_tensors(signal=False)
    with pytest.warns(UserWarning, match="edge of"):
        _, resolved, _ = RidgeConfig(k=8).fit(
            tensors, RunSpec("rdg", 1, 8, PRED_LEN, 42), device=torch.device("cpu")
        )
    assert resolved.alpha == RIDGE_ALPHAS[-1]


def test_write_artifacts_records_the_selected_alpha(tmp_path) -> None:
    """One writer, one schema — and it carries what ridge selected (`D56`, root §12).

    ``write_artifacts`` is the only definition of the ``meta/*.json`` contract.
    Duplicating it for the baselines would have created a second definition of
    §12, which is the drift surface `D54d` exists to prevent; instead it takes
    the protocol, so a ridge run and an iTransformer run are described by the
    same keys and the alpha lands in ``config`` for free.
    """
    tensors = _synthetic_tensors(signal=True)
    spec = RunSpec("rdg", 1, 8, PRED_LEN, 42)
    model, resolved, outcome = RidgeConfig(k=8).fit(
        tensors, spec, device=torch.device("cpu")
    )
    _, meta_path = write_artifacts(
        model, tensors, spec, resolved, outcome, torch.device("cpu"), root=tmp_path
    )

    meta = json.loads(meta_path.read_text())
    assert meta["status"] == "complete"
    assert meta["run_id"] == "rdg_o01_K08_H024_s42"
    assert meta["config"]["alpha"] == resolved.alpha
    assert meta["config"]["k"] == 8
    assert meta["n_parameters"] == model.n_parameters()


def _preds(path, timestamps: list[int], block: int = 1) -> None:
    pl.DataFrame(
        {
            "block": np.full(len(timestamps), block, dtype=np.int8),
            "step": np.ones(len(timestamps), dtype=np.int16),
            "timestamp": np.array(timestamps, dtype=np.int64),
            "y_true": np.zeros(len(timestamps), dtype=np.float32),
            "y_pred": np.zeros(len(timestamps), dtype=np.float32),
        }
    ).write_parquet(path)


def test_baseline_alignment_holds_and_has_teeth(tmp_path) -> None:
    """`D45` — a baseline is only comparable on its comparator's exact windows.

    Equal by construction, both window sets coming from ``window_starts`` with
    the same origin and semantics, which is why this assertion is cheap. It must
    still fail on a mismatch: RelMSE across two samples is not a ratio, and the
    two would differ systematically — test-window survival is conditioned on
    *future* gaps, and outages cluster on stress.
    """
    preds = tmp_path / "preds"
    preds.mkdir()
    stamps = [t * 3_600_000 for t in range(10)]
    _preds(preds / "rdg_o01_K08_H024_s42.parquet", stamps)
    _preds(preds / "itr_o01_K08_H024_s42.parquet", stamps)
    assert_baseline_alignment(
        "rdg_o01_K08_H024_s42", "itr_o01_K08_H024_s42", [tmp_path]
    )

    _preds(preds / "dlin_o01_K08_H024_s42.parquet", stamps[:-1])
    with pytest.raises(ValueError, match="evaluated window sets differ"):
        assert_baseline_alignment(
            "dlin_o01_K08_H024_s42", "itr_o01_K08_H024_s42", [tmp_path]
        )

    # A missing comparator must be distinguishable from a passing check, which is
    # why the runner catches this specific exception and says so out loud.
    with pytest.raises(FileNotFoundError):
        assert_baseline_alignment(
            "ptst_o01_K08_H024_s42", "itr_o01_K08_H024_s42", [tmp_path]
        )


@pytest.mark.parametrize(
    ("cfg", "lr", "ceiling"),
    [
        (PatchTSTConfig(dropout=0.0), 1e-3, 1e-3),
        (DLinearConfig(), 1e-2, 1e-6),
    ],
)
def test_single_batch_overfits_for_each_baseline(cfg, lr: float, ceiling: float) -> None:
    """Root §16's plumbing check, per model — and `D52d` again, one door along.

    `D52d` recorded that §16's instruction cannot be followed literally with
    dropout on. The same trap has a second door: **the learning rate in it is
    iTransformer's, not universal**, and the two baselines need corrections in
    *opposite* directions, so no single constant serves all three models.
    Measured, 8 samples, 200 steps, seeds 42-44:

    ========  ========  =====================  ==========================
    Model     lr 1e-3   lr 1e-2                Verdict
    ========  ========  =====================  ==========================
    iTransf.  1.3e-10   —                      §16 as written
    PatchTST  3.3e-06 … 5.2e-04                lr 1e-2 **diverges** to ~1.0
    DLinear   8.5e-02 … 9.7e-02 (stalls)       1.6e-09 … 3.9e-08
    ========  ========  =====================  ==========================

    DLinear's floor at lr 1e-3 is convergence, not plumbing: with ``B = A`` the
    decomposition telescopes to ``A(I-P) + AP = A``, so the model class contains
    every linear map from the lookback to the horizon and an exact fit exists.
    Half of ``(A, B)`` is gauge, which is what makes the descent slow. A reader
    following §16 literally would conclude DLinear is broken when it is not —
    which is why this is parameterised and measured rather than asserted at one
    constant.
    """
    set_seed(42)
    model = cfg.build().train()
    x = torch.randn(8, SEQ_LEN, 8)
    y = torch.randn(8, PRED_LEN, 8)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(200):
        opt.zero_grad(set_to_none=True)
        loss = nn.functional.mse_loss(model(x), y)
        loss.backward()
        opt.step()
    assert loss.item() < ceiling, f"plumbing broken: {loss.item():.3e}"


def test_ridge_overfits_by_construction() -> None:
    """Ridge's §16 check is a solve, so "overfit" means exact interpolation.

    Eight samples in a 768-dimensional design at a negligible alpha: the training
    residual is at floating-point zero, and anything else means the design matrix
    or the intercept is wired wrong.
    """
    tensors = _synthetic_tensors(n_train=8, n_val=8, signal=True)
    _, _, outcome = RidgeConfig(k=8, alphas=(1e-8,)).fit(
        tensors, RunSpec("rdg", 1, 8, PRED_LEN, 42), device=torch.device("cpu")
    )
    assert outcome.train_loss < 1e-12, f"plumbing broken: {outcome.train_loss:.3e}"


def test_lstm_is_multivariate_and_target_channel() -> None:
    """`D64` — LSTM's K=8 means what ridge's does, not what DLinear's does.

    The two channel-independent baselines wear their K label through an
    all-channel objective with shared weights: *trained on* eight channels,
    predicting the target from its own history alone (`D56`). An LSTM reads all
    eight channels of every timestep and emits the target, so perturbing a
    non-target channel **must** move its forecast. Without this the arm would be
    K=1 wearing a K=8 label, which is the collapse `D40` exists to prevent.
    """
    torch.manual_seed(0)
    model = LSTMConfig(k=8).build().eval()
    x = torch.randn(4, SEQ_LEN, 8)
    other = x.clone()
    other[:, :, TARGET_INDEX + 1] += 5.0

    with torch.no_grad():
        base = model.forecast_target(x)
        moved = model.forecast_target(other)

    assert base.shape == (4, PRED_LEN)
    assert not torch.allclose(base, moved), (
        "the LSTM ignored a non-target channel, so its K=8 label is not true"
    )
    assert model.cfg.loss_target() == "target"
    assert model.cfg.channel_independent is False


@pytest.mark.parametrize("mode", ["persist", "seasonal"])
def test_naive_comparators_copy_the_right_past_value(mode: str) -> None:
    """`D64` — the two secondary baselines root §7 listed and `D56` recorded unbuilt.

    Both are closed forms with no parameter, so the test is the definition: a
    persistence forecast is the last observed return repeated, and a seasonal one
    is the return a daily cycle back, step for step. Neither is Naive-RW, which
    forecasts ``y_hat_raw = 0`` and needs no run at all (`D31`).
    """
    cfg = NaiveConfig(mode=mode, k=1)
    model = cfg.build().eval()
    x = torch.randn(4, SEQ_LEN, 1)

    with torch.no_grad():
        out = model.forecast_target(x)

    assert out.shape == (4, PRED_LEN)
    assert model.n_parameters() == 0
    target = x[:, :, TARGET_INDEX]
    if mode == "persist":
        assert torch.equal(out, target[:, -1:].expand(-1, PRED_LEN))
    else:
        assert torch.equal(out, target[:, -PRED_LEN:])


def test_seasonal_naive_stays_inside_the_lookback_at_a_long_horizon() -> None:
    """H=168 asks for seven cycles the 96-bar lookback does not contain.

    The modulo repeats the last daily cycle instead of indexing past the window,
    which is the only behaviour available: a lookback cannot supply a value it
    never saw. Only the H=24 arms run in the manifest, so this guards a path the
    horizon sweep would otherwise reach as an IndexError rather than a number.
    """
    model = NaiveConfig(mode="seasonal", k=1, pred_len=168).build().eval()
    x = torch.randn(2, SEQ_LEN, 1)
    with torch.no_grad():
        out = model.forecast_target(x)
    assert out.shape == (2, 168)
    assert torch.equal(out[:, :24], out[:, 24:48])


def test_single_batch_overfits_for_the_lstm() -> None:
    """Root §16's plumbing check for the LSTM (`D64`) — its own test, and why.

    It cannot join the parameterised baseline check above: that one supervises
    ``model(x)`` against an ``(8, H, N)`` target, because DLinear and PatchTST
    carry their published all-channel objective. The LSTM is multivariate and
    target-channel, so it returns ``(8, H)`` and needs the matching target.

    Measured here, 8 samples, 200 steps, dropout off, seeds 42-44:

    ========  =====================================  ====================
    lr        loss after 200 steps                   Verdict
    ========  =====================================  ====================
    1e-3      6.8e-09 … 1.5e-08                      §16 as written
    3e-3      6.5e-10 … 1.6e-09                      best
    1e-2      2.7e-10 … 3.5e-08                      one seed an order worse
    3e-2      2.3e-09 … 4.0e-05                      destabilising
    ========  =====================================  ====================

    So the LSTM is the one baseline that needs **no** learning-rate correction —
    iTransformer's own 1e-3 drives it to ~1e-8 on every seed. That is worth
    recording rather than discovering: the test above exists precisely because
    PatchTST and DLinear need corrections in opposite directions, and a reader
    would reasonably assume a third correction was hiding here.

    `D52d` holds at this door too: the same run with the configured
    ``dropout=0.1`` floors at **5.5e-04**, and a reader following §16 literally
    would call the plumbing broken when it is not.
    """
    set_seed(42)
    model = LSTMConfig(k=8, dropout=0.0).build().train()
    x = torch.randn(8, SEQ_LEN, 8)
    y = torch.randn(8, PRED_LEN)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(200):
        opt.zero_grad(set_to_none=True)
        loss = nn.functional.mse_loss(model(x), y)
        loss.backward()
        opt.step()
    assert loss.item() < 1e-7, f"plumbing broken: {loss.item():.3e}"


@pytest.mark.parametrize(
    ("arm", "cfg"),
    [
        ("lstm", LSTMConfig(k=8)),
        ("npst", NaiveConfig(mode="persist", k=8)),
        ("nsea", NaiveConfig(mode="seasonal", k=8)),
    ],
)
def test_new_baselines_reach_write_artifacts(arm: str, cfg, tmp_path) -> None:
    """`D64`'s three arms go all the way to a complete run on disk.

    The shape tests above prove each model computes the right thing; this proves
    the path around it exists — ``fit`` returns the protocol's triple, and
    ``write_artifacts`` accepts it and marks the run complete. Root §10.5 counts
    a run finished only when both files exist **and** ``meta.status`` says so, so
    anything less is silently re-run, and a resume that re-runs everything is how
    a session's budget disappears.

    The naive arms are the interesting case: ``n_parameters`` is genuinely 0 and
    ``epochs_run`` is genuinely 0, and the writer must record both rather than
    treat either as a missing value.
    """
    tensors = _synthetic_tensors(signal=True)
    spec = RunSpec(arm, 1, 8, PRED_LEN, 42)
    model, resolved, outcome = cfg.fit(tensors, spec, device=torch.device("cpu"))
    preds_path, meta_path = write_artifacts(
        model, tensors, spec, resolved, outcome, torch.device("cpu"), root=tmp_path
    )

    meta = json.loads(meta_path.read_text())
    assert meta["status"] == "complete"
    assert meta["run_id"] == f"{arm}_o01_K08_H024_s42"
    assert meta["n_parameters"] == model.n_parameters()
    assert preds_path.exists()

    frame = pl.read_parquet(preds_path)
    assert set(frame.columns) >= {"block", "step", "timestamp", "y_true", "y_pred"}
    assert frame.height > 0
    assert frame["y_pred"].is_finite().all(), "a forecast came back non-finite"

    if arm != "lstm":
        assert model.n_parameters() == 0
        assert outcome.epochs_run == 0
