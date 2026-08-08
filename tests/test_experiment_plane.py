"""Manifest, K_eff and metric tests — what root §9 and §10 claim about the grid.

Reads ``data/raw/BTCUSDT_1h.parquet``. Writes nothing and trains nothing: every
statistic here is checked against a closed form or against a synthetic panel
whose answer is known in advance, so the suite stays runnable on CPU in seconds.

Several exist because the quantity they check was wrong the first time it was
measured: the lookback stable rank returned 1.0 at every rung until the channels
were standardised per window, and the cross-lag PR was non-monotone in K until it
moved off the raw covariance.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from itransformer_btc import keff, metrics, runner
from itransformer_btc.config import ORIGINS, FalsificationOrigin
from itransformer_btc.features import build_features
from itransformer_btc.segments import load_bars, usable_mask
from itransformer_btc.splits import build_origin_tensors


@pytest.fixture(scope="session")
def feats() -> pl.DataFrame:
    return build_features(usable_mask(load_bars()))


# -- the manifest — root §10.2, §10.4 ----------------------------------------


def test_manifest_is_deduplicated_by_run_id() -> None:
    """The sweep's H=24 slice shares 48 ``run_id``s with the main grid.

    Root §10.4 makes ``run_id`` the identity of a run, so executing a shared cell
    twice would mean two files racing for one path. 582 nominal cells are 534
    real runs.
    """
    cells = runner.manifest()
    ids = [c.run_id for c in cells]
    assert len(ids) == len(set(ids))
    assert len(cells) == 534

    counts: dict[str, int] = {}
    for cell in cells:
        counts[cell.arm] = counts.get(cell.arm, 0) + 1
    assert counts == {"main": 300, "uniform": 75, "fresh": 15, "horizon": 144}
    assert 300 + 75 + 15 + 192 - len(cells) == 48


def test_shards_partition_the_manifest_exactly() -> None:
    """Disjoint and exhaustive, and a group is never split across workers.

    Splitting a group would make both workers build the same tensors; leaving a
    gap would silently drop cells from a table.
    """
    cells = runner.manifest()
    for n in (1, 2, 3):
        parts = [runner.shard(cells, i, n) for i in range(n)]
        ids = [{c.run_id for c in part} for part in parts]
        assert sum(len(s) for s in ids) == len(cells)
        assert set().union(*ids) == {c.run_id for c in cells}
        owners: dict[tuple, int] = {}
        for i, part in enumerate(parts):
            for cell in part:
                owners.setdefault(cell.group, i)
                assert owners[cell.group] == i


def test_arm_tags_keep_the_arms_apart() -> None:
    """`D50`'s uniform arm and the falsification arm must not collide with main."""
    cells = {c.arm: c for c in runner.manifest()}
    assert cells["main"].run_id.startswith("itr_")
    assert cells["uniform"].run_id.startswith("itru_")
    assert cells["fresh"].run_id.startswith("itrf_")
    assert cells["uniform"].model_config().uniform_attention is True
    assert cells["main"].model_config().uniform_attention is False


# -- the falsification arm — root §8.1 ---------------------------------------


def test_fresh_origin_lands_on_block_four(feats: pl.DataFrame) -> None:
    """``o_i + 90 days`` is exactly where base block 4 opens.

    90 days, not 3 calendar months: blocks are 30 **days**, so only the day
    offset makes "the *same* calendar blocks 4-6" true rather than approximately
    true. Its training window is the aged one shifted whole, so both models see
    the same span length and the comparison is not confounded by volume.
    """
    base = ORIGINS[0]
    fresh = FalsificationOrigin(base)
    assert fresh.origin == base.block(4)[0]
    assert [b for b, _, _ in fresh.blocks()] == [4, 5, 6]
    assert [(lo, hi) for _, lo, hi in fresh.blocks()] == [
        base.block(b) for b in (4, 5, 6)
    ]
    assert (fresh.train_sub_end - fresh.train_start) == (
        base.train_sub_end - base.train_start
    )

    tensors = build_origin_tensors(feats, fresh, 8)
    assert tensors.block_labels == (4, 5, 6)
    assert tensors.train.ts.max() < int(fresh.val_start.timestamp() * 1000)


# -- K_eff — root §5.4 -------------------------------------------------------


def test_keff_reads_training_spans_only(feats: pl.DataFrame) -> None:
    """FATAL, `D44`. RQ1's regressor may not see a bar its outcome is measured on.

    The gate is additionally confined to the pre-first-origin span (`D02`), which
    contains no origin's test block at all.
    """
    origin = ORIGINS[0]
    rows, windows = keff._training_windows(feats, origin, 8)
    ts = feats.get_column("ts_ms").to_numpy()
    lo = int(origin.train_start.timestamp() * 1000)
    hi = int(origin.train_sub_end.timestamp() * 1000)
    assert len(rows) == int(((ts >= lo) & (ts < hi)).sum())
    assert len(windows) > 0
    assert hi <= int(origin.test_start.timestamp() * 1000)


def test_participation_ratio_hits_both_bounds() -> None:
    """PR is 1 on a rank-one spectrum and K on a flat one — the interval it claims."""
    assert keff.participation_ratio(np.array([5.0, 0.0, 0.0])) == pytest.approx(1.0)
    assert keff.participation_ratio(np.ones(8)) == pytest.approx(8.0)
    with pytest.raises(ValueError):
        keff.participation_ratio(np.zeros(4))


def test_lookback_stable_rank_is_scale_free() -> None:
    """It must not move when one channel is rescaled — the bug that made it useless.

    Before the per-window standardisation, the merely-centred version returned
    1.00 at every rung on real data: ``log_quote_volume`` deviations are orders
    of magnitude larger than ``r`` deviations, so one row dominated both the
    Frobenius and the spectral norm and the statistic reported "one effective
    direction" everywhere. That is a units artefact, not a property of the data.
    """
    rng = np.random.default_rng(0)
    windows = rng.normal(size=(64, 96, 4))
    plain = keff.lookback_stable_rank(windows)
    rescaled = windows.copy()
    rescaled[:, :, 1] *= 1_000.0
    assert keff.lookback_stable_rank(rescaled) == pytest.approx(plain, rel=1e-9)
    assert 1.0 <= plain <= 4.0


def test_keff_row_is_bounded_and_reports_its_divergence(feats: pl.DataFrame) -> None:
    """Every measure that claims ``[1, K]`` must actually stay inside it."""
    row = keff.keff_row(feats, ORIGINS[0], 8)
    assert 1.0 <= row.pr_raw <= 8.0
    assert 1.0 <= row.pr_window_norm <= 8.0
    assert 1.0 <= row.stable_rank_lookback <= 8.0
    assert 0.0 < row.pr_lookback_ratio <= 1.0
    assert row.divergence == pytest.approx(row.stable_rank_lookback - row.pr_raw)


def test_gate_action_is_disclosure_not_a_recut() -> None:
    """`D48` — "re-cut the ladder" named no reachable alternative, so it is gone."""
    assert "PASS" in keff.gate_verdict(6.0)
    verdict = keff.gate_verdict(4.0)
    assert "DISCLOSE" in verdict and "Do not re-cut" in verdict


# -- metrics — root §9 -------------------------------------------------------


def test_non_overlapping_phase_is_midnight_utc() -> None:
    """`D46` — 24 alignments exist and each gives a different Sharpe; pick one first."""
    hours = np.arange(0, 48) * metrics.HOUR_MS
    keep = metrics.non_overlapping_mask(hours)
    assert keep.sum() == 2
    assert list(np.flatnonzero(keep)) == [0, 24]


def test_decay_is_on_the_skill_scale_and_guards_its_denominator() -> None:
    """`D23` and `D05`. On RelMSE every pre-registered tau is unreachable.

    The origin with non-positive mean skill must be **named**, not silently
    dropped: D is a proportion of an edge, and an origin with no edge has no
    proportion of one.
    """
    rows = []
    for origin, skills in (
        ("good", [0.10, 0.08, 0.06, 0.04, 0.02, 0.00]),
        ("dead", [-0.01] * 6),
    ):
        for block, r2 in enumerate(skills, start=1):
            rows.append({
                "model": "itr",
                "origin_index": 1 if origin == "good" else 2,
                "origin": origin, "k": 8, "pred_len": 24, "block": block,
                "mse": 1.0 - r2, "mse_naive": 1.0, "n_windows": 720, "r2_oos": r2,
            })
    result = metrics.decay(pl.DataFrame(rows))
    assert result.excluded_origins == ("dead",)
    assert set(result.table.get_column("origin").to_list()) == {"good"}

    # Mean skill is 0.05, so D(1) = (0.05 - 0.10)/0.05 = -1 and D(6) = +1.
    d = result.table.sort("block").get_column("D").to_list()
    assert d[0] == pytest.approx(-1.0)
    assert d[-1] == pytest.approx(1.0)

    censored = result.b_star(tau=5.0)
    assert censored.get_column("b_star").to_list() == [6]
    assert censored.get_column("event").to_list() == [False]


def test_b_star_keeps_its_columns_when_every_origin_is_excluded() -> None:
    """`D55`. The measured grid excluded all fifteen origins, not an edge case.

    ``decay``'s ``R2_oos > 0`` guard is correct and stays. What was wrong is that
    ``b_star`` then inferred its schema from zero rows and returned a frame with
    **no columns**, so the notebook's ``bs["b_star"]`` raised
    ``ColumnNotFoundError`` and marked the twelve-hour Kaggle version failed at
    the moment its grid output was the only thing worth keeping. `D54e` gates the
    estimators on grid *completeness*, which is a different failure.

    The distinction the columns have to survive for: an empty frame means the
    estimand is **undefined**, whereas a frame of sixes with ``event=False``
    means every origin is **censored** — an edge that never decays past tau. The
    two must not be reported in one wording.
    """
    rows = [
        {
            "model": "itr", "origin_index": i, "origin": f"dead-{i}", "k": 8,
            "pred_len": 24, "block": block, "mse": 1.02, "mse_naive": 1.0,
            "n_windows": 720, "r2_oos": -0.018,
        }
        for i in (1, 2, 3)
        for block in range(1, 7)
    ]
    result = metrics.decay(pl.DataFrame(rows))
    assert result.excluded_origins == ("dead-1", "dead-2", "dead-3")
    assert result.table.height == 0

    bs = result.b_star(tau=metrics.TAU_HEADLINE)
    assert bs.height == 0
    assert bs.columns == ["origin", "tau", "b_star", "event"]

    # The exact line that took the notebook down.
    times, events = bs["b_star"].to_numpy(), bs["event"].to_numpy()
    assert len(times) == 0

    # Kaplan-Meier over no observations must degrade, not raise.
    km = metrics.kaplan_meier(times, events)
    assert km.n_events == 0 and km.n_censored == 0
    assert km.median == float("inf")
    assert km.median_interval == (float("inf"), float("inf"))


def test_hln_factor_guard_fires_where_it_must() -> None:
    """Root §9.2 refuses to report where ``T + 1 - 2h + h(h-1)/T <= 0``.

    At h=24 the factor is exactly 0 at T=24, so a silent negative would produce a
    complex statistic reported as a real one.
    """
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="HLN factor"):
        metrics.dm_test(rng.normal(size=24), rng.normal(size=24), h=24)
    assert metrics.dm_test(rng.normal(size=720), rng.normal(size=720), h=24).T == 720


def test_clark_west_lifts_the_statistic_on_a_nested_pair() -> None:
    """`D29` — standard DM is undersized against exactly this alternative.

    The large model genuinely helps here; the unadjusted differential is dragged
    towards the small model by the large one's estimation noise, and Clark-West's
    ``+(y_small - y_large)^2`` term is what recovers it.
    """
    rng = np.random.default_rng(7)
    y = rng.normal(size=720)
    small = np.zeros(720)
    large = 0.05 * y + rng.normal(scale=0.25, size=720)
    cw = metrics.clark_west_test(y, small, large, h=24)
    dm = metrics.dm_test((y - small) ** 2, (y - large) ** 2, h=24)
    assert cw.one_sided and not dm.one_sided
    assert cw.statistic > dm.statistic


def test_bootstrap_p_never_returns_zero() -> None:
    """A finite bootstrap cannot support p = 0; the floor is ``1/(B+1)``."""
    blocks = list(range(1, 7))
    panel = pl.DataFrame({
        "origin": [f"o{i:02d}" for i in range(15) for _ in blocks],
        "block": blocks * 15,
        "A": [1.0 - 0.5 * b for _ in range(15) for b in blocks],
    })
    result = metrics.panel_beta1(panel, B=999, seed=1)
    assert result.beta1 == pytest.approx(-0.5)
    assert result.p_rademacher >= 1.0 / 1000.0
    assert result.headline_p == max(result.p_rademacher, result.p_webb)
    assert result.n_clusters == 15 and result.n_observations == 90


def test_beta1_is_the_mean_of_within_origin_slopes() -> None:
    """`D42`'s algebraic identity — inference is a one-sample test on G numbers.

    Citing "15 x 6 = 90 observations" invites the reader to infer power that does
    not exist. If this identity ever breaks, that claim breaks with it.
    """
    rng = np.random.default_rng(3)
    blocks = list(range(1, 7))
    panel = pl.DataFrame({
        "origin": [f"o{i:02d}" for i in range(15) for _ in blocks],
        "block": blocks * 15,
        "A": [float(rng.normal()) for _ in range(15) for _ in blocks],
    })
    result = metrics.panel_beta1(panel, B=99, seed=1)
    assert result.beta1 == pytest.approx(float(result.within_slopes.mean()))
    assert len(result.within_slopes) == 15


def test_unbalanced_panel_is_refused() -> None:
    """The identity above holds only on a balanced panel, so say so loudly."""
    panel = pl.DataFrame({
        "origin": ["a", "a", "a", "b", "b"],
        "block": [1, 2, 3, 1, 2],
        "A": [0.1, 0.2, 0.3, 0.1, 0.2],
    })
    with pytest.raises(ValueError, match="unbalanced"):
        metrics.panel_beta1(panel, B=99)


def test_tost_needs_the_margin_to_conclude_equivalence() -> None:
    """`D49` — a non-significant delta is a failure to reject, not equivalence."""
    tight = np.full(15, 1e-6) + np.linspace(-1e-7, 1e-7, 15)
    assert metrics.tost_equivalence(tight, margin=1e-3).equivalent
    assert not metrics.tost_equivalence(tight, margin=1e-9).equivalent


def test_j_test_identifies_the_true_regressor() -> None:
    """`D32` — RQ1 is a non-nested comparison, and the test must discriminate."""
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(20), 4)
    k = np.tile(np.array([1.0, 4.0, 8.0, 12.0]), 20)
    k_eff = np.tile(np.array([1.0, 3.3, 4.3, 4.0]), 20)
    y = (
        -0.1 * k_eff
        + rng.normal(scale=0.05, size=80)
        + np.repeat(rng.normal(size=20), 4)
    )
    _, p_k_needs_keff = metrics.j_test(y, k, k_eff, groups)
    _, p_keff_needs_k = metrics.j_test(y, k_eff, k, groups)
    assert p_k_needs_keff < 0.01      # K alone is inadequate
    assert p_keff_needs_k > 0.05      # K_eff alone suffices


def test_kaplan_meier_reports_censoring_rather_than_hiding_it() -> None:
    """`D41` — an origin that never crosses tau is censored at 6, not missing.

    A bare mean over the crossers would condition on the event and bias the
    recommended cadence downward, which is the number the abstract carries.
    """
    curve = metrics.kaplan_meier(np.array([6.0] * 5), np.array([False] * 5))
    assert curve.median == math.inf
    assert curve.n_events == 0 and curve.n_censored == 5

    crossed = metrics.kaplan_meier(np.array([2.0] * 5), np.array([True] * 5))
    assert crossed.median == 2.0
