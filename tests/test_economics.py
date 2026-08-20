"""Root §13.5's economic evaluation, with `D46`'s three specifications pinned.

Each of the three moves every number in Table 8, so each gets a test that fails
if the specification is quietly re-decided: the 00:00 UTC phase, the flat-day
accounting that bounds how optimistic the drawdown is, and the slippage band.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from itransformer_btc.economics import (
    SLIPPAGE_BAND,
    TAKER_FEE_PER_SIDE,
    buy_and_hold,
    deflated_sharpe,
    economics_table,
    jobson_korkie_memmel,
    max_drawdown,
    moments,
    net_returns,
    positions,
    run_strategy,
)

HOUR_MS = 3_600_000
ORIGIN_MS = 1_577_836_800_000  # 2020-01-01T00:00Z, the first origin

ARTIFACTS = (
    Path(__file__).resolve().parent.parent / "notebooks" / "outputs" / "artifacts"
)
needs_grid = pytest.mark.skipif(
    not (ARTIFACTS / "preds" / "itr_o01_K08_H024_s42.parquet").exists(),
    reason="grid output not present in this checkout",
)

META = {"sigma_g": 0.009, "mu_g": 1e-05, "origin": "2020-01"}


def _preds(n_days: int, seed: int = 0, drop_hours: tuple[int, ...] = ()) -> pl.DataFrame:
    """24-step forecasts issued every hour, so exactly one per day survives phase.

    ``timestamp`` is the **window start**, as ``write_artifacts`` writes it, and
    ``L = 96`` is a multiple of 24, so selecting starts at hour 0 selects targets
    opening at hour 0.
    """
    rng = np.random.default_rng(seed)
    starts = [
        ORIGIN_MS + h * HOUR_MS for h in range(n_days * 24) if h not in drop_hours
    ]
    block, step, ts, y_true, y_pred = [], [], [], [], []
    for t in starts:
        day = (t - ORIGIN_MS) // (24 * HOUR_MS)
        for s in range(1, 25):
            block.append(int(day // 30) + 1)
            step.append(s)
            ts.append(int(t))
            y_true.append(float(rng.standard_normal()))
            y_pred.append(float(rng.standard_normal()) * 0.05)
    return pl.DataFrame(
        {
            "block": block,
            "step": step,
            "timestamp": ts,
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )


# -- `D46` specification 1: the phase ----------------------------------------


def test_positions_open_only_at_midnight_utc() -> None:
    """Twenty-four admissible alignments of a non-overlapping daily partition
    exist and each gives a different Sharpe, MDD and turnover, so the phase is
    fixed in advance rather than chosen after seeing the equity curve."""
    frame = positions(_preds(10), META["sigma_g"], META["mu_g"])
    assert frame.height == 10
    assert ((frame.get_column("timestamp").to_numpy() // HOUR_MS) % 24 == 0).all()


def test_position_is_the_sign_of_the_cumulative_forecast() -> None:
    frame = positions(_preds(30, seed=3), META["sigma_g"], META["mu_g"])
    forecast = frame.get_column("forecast_raw").to_numpy()
    position = frame.get_column("position").to_numpy()
    assert np.array_equal(np.sign(forecast), position)
    assert set(np.unique(position)) <= {-1.0, 0.0, 1.0}


def test_realised_return_carries_the_drift_the_forecast_does_not() -> None:
    """`D31`. The position comes from the drift-free forecast; the realised
    return is the actual market move and therefore includes ``H * mu_g``."""
    preds = _preds(5, seed=9)
    with_drift = positions(preds, META["sigma_g"], mu_g=0.01)
    without = positions(preds, META["sigma_g"], mu_g=0.0)
    assert np.allclose(
        with_drift.get_column("realised_raw").to_numpy()
        - without.get_column("realised_raw").to_numpy(),
        24 * 0.01,
    )
    assert np.array_equal(
        with_drift.get_column("position").to_numpy(),
        without.get_column("position").to_numpy(),
    )


# -- `D46` specification 2: flat days, which bound the drawdown ---------------


def test_missing_daily_slots_are_counted_not_silently_skipped() -> None:
    """Positions exist only where a valid window exists, and outages cluster on
    stress, so the strategy is flat precisely across the large-drawdown periods.
    Reported MDD is optimistic by an amount only this count lets a reader bound.
    """
    intact = run_strategy(_preds(30, seed=2), META, 0.0005, mdd_interval=False)
    assert intact.n_flat_days == 0

    holed = run_strategy(
        _preds(30, seed=2, drop_hours=(48, 72, 96)), META, 0.0005, mdd_interval=False
    )
    assert holed.n_flat_days == 3
    assert holed.n_periods == intact.n_periods - 3


# -- `D46` specification 3: the slippage band --------------------------------


def test_costs_reduce_net_return_monotonically() -> None:
    """Fixing the fee exactly while leaving slippage blank fixes the lever that
    costs nothing and leaves open the one that decides whether the strategy makes
    money."""
    preds = _preds(120, seed=1)
    nets = [
        run_strategy(preds, META, s, mdd_interval=False).net_log_return
        for s in SLIPPAGE_BAND
    ]
    assert nets[0] > nets[1] > nets[2]


def test_a_held_position_is_charged_once_not_every_period() -> None:
    position = np.array([1.0, 1.0, 1.0, -1.0])
    net = net_returns(position, np.zeros(4), 0.0005)
    unit = TAKER_FEE_PER_SIDE + 0.0005
    assert net == pytest.approx([-unit, 0.0, 0.0, -2 * unit])


# -- the summary statistics --------------------------------------------------


def test_max_drawdown_of_a_monotone_gain_is_zero() -> None:
    assert max_drawdown(np.full(10, 0.01)) == pytest.approx(0.0, abs=1e-12)


def test_max_drawdown_recovers_a_known_fall() -> None:
    """Up 0.2 in logs, then down 0.4: the trough sits ``exp(-0.4)`` below the peak."""
    assert max_drawdown(np.array([0.2, -0.4])) == pytest.approx(1 - math.exp(-0.4))


def test_annualisation_uses_365_because_crypto_trades_every_day() -> None:
    result = run_strategy(_preds(200, seed=5), META, 0.0005, mdd_interval=False)
    assert result.sharpe_annualised == pytest.approx(
        result.sharpe_per_period * math.sqrt(365)
    )


def test_buy_and_hold_is_the_comparator_and_trades_once() -> None:
    """Naive-RW holds a constant zero position, so its return series has zero
    variance and its Sharpe is undefined; comparing against it is meaningless."""
    hold = buy_and_hold(_preds(60, seed=7), META, 0.0005, mdd_interval=False)
    assert hold.turnover_per_period == pytest.approx(0.5 / hold.n_periods)


def test_jobson_korkie_memmel_is_zero_against_itself() -> None:
    rng = np.random.default_rng(8)
    series = rng.standard_normal(200) * 0.01
    z, p = jobson_korkie_memmel(series, series)
    assert z == pytest.approx(0.0, abs=1e-9)
    assert p == pytest.approx(1.0)


# -- the Deflated Sharpe Ratio (`D46`) ---------------------------------------


def test_deflated_sharpe_is_a_probability() -> None:
    dsr = deflated_sharpe(
        0.05, T=180, skew=-0.3, kurtosis=6.0, n_trials=45, var_sharpe=0.002
    )
    assert 0.0 <= dsr <= 1.0


def test_deflated_sharpe_falls_as_the_trial_count_rises() -> None:
    """The whole point of the statistic: a Sharpe that survives 5 candidates need
    not survive 500."""
    few = deflated_sharpe(0.10, 180, -0.2, 5.0, n_trials=5, var_sharpe=0.002)
    many = deflated_sharpe(0.10, 180, -0.2, 5.0, n_trials=500, var_sharpe=0.002)
    assert few > many


def test_deflated_sharpe_refuses_an_undefined_trial_set() -> None:
    assert math.isnan(deflated_sharpe(0.1, 180, 0.0, 3.0, n_trials=1, var_sharpe=0.002))
    assert math.isnan(deflated_sharpe(0.1, 180, 0.0, 3.0, n_trials=45, var_sharpe=0.0))


def test_moments_of_a_normal_sample_are_near_zero_and_three() -> None:
    rng = np.random.default_rng(10)
    skew, kurtosis = moments(rng.standard_normal(100_000))
    assert abs(skew) < 0.05
    assert 2.9 < kurtosis < 3.1


# -- against the real artifact -----------------------------------------------


@needs_grid
def test_economics_table_covers_every_slippage_level_and_states_its_dsr_inputs() -> None:
    table = economics_table(
        [ARTIFACTS], keys=[("itr", 8), ("rdg", 8)], origin_indices=(1,), seed=3
    )
    assert table.height == 2 * len(SLIPPAGE_BAND)
    assert sorted(set(table.get_column("slippage_per_side").to_list())) == list(
        SLIPPAGE_BAND
    )
    # Root §12: a number in the manuscript must be redoable from what is reported.
    assert (table.get_column("dsr_n_trials").to_numpy() > 1).all()
    assert (table.get_column("n_periods").to_numpy() > 100).all()
    assert set(table.columns) >= {
        "max_drawdown",
        "mdd_ci_low",
        "mdd_ci_high",
        "turnover_per_period",
        "sortino_annualised",
        "hold_sharpe_annualised",
        "jk_memmel_p",
        "dsr",
    }
