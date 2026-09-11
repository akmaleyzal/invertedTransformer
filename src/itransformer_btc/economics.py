"""Root §13.5's economic evaluation -- Table 8 and Figure 7.

`D60g` found this never ran at all: no positions, no equity curve, no Deflated
Sharpe Ratio anywhere in the session log. `D46` had already fixed three
specifications in advance, each of which moves every number in Table 8, and each
is honoured here rather than re-decided:

1. **Phase.** Positions open at **00:00 UTC**. There are 24 admissible
   alignments of a non-overlapping daily partition, each with a different Sharpe,
   MDD and turnover; choosing the phase after seeing the equity curve is a free
   parameter on the paper's economic claim. :func:`metrics.non_overlapping_mask`
   already implements exactly this and is reused rather than reimplemented.
2. **Gap-spanning returns are forbidden.** A position held across a downtime
   block has no defined realised return, and the obvious ``log(C_{t+24}/C_t)`` is
   the cross-gap return §4.3 forbids everywhere else -- at the 2018-02-08 block a
   nominal 24-hour trade would book a 57-hour move. The pipeline already makes
   this unreachable: ``windows.enumerate_windows`` validates every retained
   window by ``t[s + L + H - 1] - t[s] == (L + H - 1)`` hours, so a surviving
   forecast's target bars are contiguous by construction. The check is therefore
   *vacuous*, and saying so is the point -- what is **not** vacuous is
   :attr:`StrategyResult.n_flat_days`, the calendar days inside a test block with
   no surviving 00:00 window because the window was rejected upstream. Outages
   cluster on stress, so the strategy is flat precisely across the
   large-drawdown periods and the reported MDD is optimistic by an amount only
   that count lets a reader bound.
3. **Costs.** A 0.04% taker fee per side, plus slippage at a documented
   sensitivity band of 0.02% / 0.05% / 0.10% per side, with Table 8 reported at
   all three. Fixing the fee exactly while leaving slippage blank fixes the lever
   that costs nothing and leaves open the one that decides whether the strategy
   makes money.

**The comparator is buy-and-hold, and the reason belongs in the caption.** Root
§13.5 asks for a Sharpe test "against the naive strategy", but Naive-RW forecasts
``y_raw = 0`` and therefore holds a constant zero position: its return series has
zero variance and its Sharpe is undefined. Comparing against it is not
conservative, it is meaningless.

**Everything is computed on raw, drift-free log-returns, never scaler-space**
(`D31`). ``y_z = 0`` in scaler space means ``r_hat = mu_g``, the training-window
mean hourly return, so a sign rule read there is a sign rule on a constant-drift
model -- and ``mu_g/sigma_g`` **changes sign across origins**, so it is not a
tilt a reader could mentally subtract.

Upstream
--------
**Written here on numpy; no backtesting package is a dependency.**

``deflated_sharpe`` -- D. H. Bailey and M. Lopez de Prado, "The deflated Sharpe
ratio: Correcting for selection bias, backtest overfitting, and non-normality,"
*J. Portfolio Manage.*, vol. 40, no. 5, pp. 94-107, 2014. The published
definition is followed, but the *arguments* are this study's and `D46` explains
why: DSR counts candidates whose Sharpe was computed on the **same** return
series, so it is computed **per origin** from that origin's non-overlapping
24-hour strategy returns and their **per-period** Sharpe -- never the annualised
one, which would inflate it by ``sqrt(periods per year)``. ``N`` is the number
of configurations evaluated on that origin's own test span, not the 1,620-run
total; the total is reported separately as the development trial count.
:data:`itransformer_btc.config.SOURCE_PROVENANCE` carries this row in full.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl

from itransformer_btc.config import BLOCK_DAYS, PRED_LEN
from itransformer_btc.metrics import (
    load_meta,
    load_predictions,
    non_overlapping_mask,
    normal_cdf,
    normal_quantile,
)

#: Binance spot taker fee, per side. Fixed; the band below is the sensitivity,
#: because slippage is the lever that decides whether the strategy makes money
#: and the project's own reference library anchors BTC effective spreads near
#: 0.30%.
TAKER_FEE_PER_SIDE: Final = 0.0004

#: Root §13.5's documented slippage band, per side. Table 8 is reported at
#: all three, never at one.
SLIPPAGE_BAND: Final[tuple[float, ...]] = (0.0002, 0.0005, 0.0010)

#: Crypto trades every day, so annualising a daily Sharpe uses 365, not 252.
PERIODS_PER_YEAR: Final = 365

#: Euler-Mascheroni, for the Deflated Sharpe Ratio's expected-maximum threshold.
EULER_MASCHERONI: Final = 0.5772156649015329

#: Expected block length of the stationary bootstrap behind the MDD interval, in
#: daily periods. From ~180 observations a point maximum drawdown is
#: uninterpretable, and a block short enough to break the drawdown's own
#: persistence would understate the interval.
MDD_BLOCK_DAYS: Final = 5

#: Draws for the drawdown interval.
MDD_BOOTSTRAP_B: Final = 1_999


@dataclass(frozen=True, slots=True)
class StrategyResult:
    """One (model, origin, slippage) cell of Table 8.

    ``sharpe_per_period`` is the figure the Deflated Sharpe Ratio consumes;
    feeding it the annualised one inflates it by ``sqrt(PERIODS_PER_YEAR)`` and
    is the easiest single way to report a strategy as significant when it is not.
    """

    n_periods: int
    n_flat_days: int
    mean_net: float
    sharpe_per_period: float
    sharpe_annualised: float
    sortino_annualised: float
    max_drawdown: float
    mdd_ci_low: float
    mdd_ci_high: float
    turnover_per_period: float
    net_log_return: float
    net_total_return: float


def positions(preds: pl.DataFrame, sigma_g: float, mu_g: float) -> pl.DataFrame:
    """Exploratory long/cash daily decisions from inverse-scaled forecasts.

    Observed complete targets are selected retrospectively. This conditional
    simulation cannot establish executable performance across missing outcomes.
    """
    if not np.isfinite(sigma_g) or sigma_g <= 0 or not np.isfinite(mu_g):
        raise ValueError("finite mean and positive scale are required")
    per_window = preds.group_by("timestamp").agg(
        pl.col("y_pred").sum().alias("_f_z"), pl.col("y_true").sum().alias("_a_z"),
        pl.len().alias("_n_steps"), pl.col("step").n_unique().alias("_unique"),
        pl.col("step").min().alias("_first"), pl.col("step").max().alias("_last"),
        pl.col("block").first().alias("block"),
    ).sort("timestamp")
    if per_window.filter((pl.col("_n_steps") != 24) | (pl.col("_unique") != 24) |
                         (pl.col("_first") != 1) | (pl.col("_last") != 24)).height:
        raise ValueError("economics requires complete 24-step daily forecasts")
    if preds.select(pl.any_horizontal(pl.col("y_true", "y_pred").is_null() |
                                     ~pl.col("y_true", "y_pred").is_finite()).any()).item():
        raise ValueError("non-finite economic inputs")
    keep = non_overlapping_mask(per_window["timestamp"].to_numpy())
    return per_window.filter(pl.Series(keep)).with_columns(
        (pl.col("_f_z") * sigma_g + 24 * mu_g).alias("forecast_raw"),
        (pl.col("_a_z") * sigma_g + 24 * mu_g).alias("realised_raw"),
    ).with_columns((pl.col("forecast_raw") > 0).cast(pl.Float64).alias("position")).select(
        "timestamp", "block", "position", "realised_raw", "forecast_raw"
    )


def net_returns(position: np.ndarray, realised: np.ndarray, slippage_per_side: float) -> np.ndarray:
    """Self-financing log equity increments of separate daily long/cash trades.

    Every long trade opens from cash and closes after 24h, including the final
    trade and trades beside missing days. Buy at P*(1+c), sell at P'*(1-c):
    wealth multiplier = exp(realised)*(1-c)/(1+c). Cash earns zero.
    This explicit round-trip policy replaces the old unfinanced short ledger.
    """
    position, realised = np.asarray(position), np.asarray(realised)
    if position.ndim != 1 or position.shape != realised.shape or not len(position):
        raise ValueError("position and realised must be nonempty equal 1D arrays")
    if not np.isfinite(position).all() or not np.isfinite(realised).all():
        raise ValueError("non-finite ledger input")
    if not np.isin(position, [0., 1.]).all():
        raise ValueError("this spot simulation permits only long/cash positions")
    if not np.isfinite(slippage_per_side) or slippage_per_side < 0:
        raise ValueError("slippage must be finite and nonnegative")
    cost = TAKER_FEE_PER_SIDE + slippage_per_side
    if cost >= 1:
        raise ValueError("per-side cost must be below one")
    return position * (realised + math.log1p(-cost) - math.log1p(cost))


def max_drawdown(net: np.ndarray) -> float:
    """Largest peak-to-trough fall of the cumulative curve, as a fraction."""
    if len(net) == 0:
        return float("nan")
    equity = np.exp(np.cumsum(net))
    peak = np.maximum.accumulate(np.maximum(equity, 1.0))
    return float((1.0 - equity / peak).max())


def _stationary_bootstrap_mdd(
    net: np.ndarray, B: int = MDD_BOOTSTRAP_B, seed: int = 42
) -> tuple[float, float]:
    """Percentile interval for the maximum drawdown (Politis & Romano 1994).

    Geometric block lengths with mean :data:`MDD_BLOCK_DAYS`, so the resample
    preserves the short-run persistence a drawdown is made of. An i.i.d.
    bootstrap would shatter exactly the runs the statistic measures and return an
    interval that is both too narrow and too low.
    """
    n = len(net)
    if n < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    p = 1.0 / MDD_BLOCK_DAYS
    starts = rng.integers(0, n, size=(B, n))
    jumps = rng.random((B, n)) < p
    draws = np.empty(B)
    for b in range(B):
        idx = np.empty(n, dtype=np.int64)
        i = int(starts[b, 0])
        for t in range(n):
            idx[t] = i
            i = int(starts[b, t]) if jumps[b, t] else (i + 1) % n
        draws[b] = max_drawdown(net[idx])
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def summarise(
    position: np.ndarray,
    realised: np.ndarray,
    slippage_per_side: float,
    n_flat_days: int,
    mdd_interval: bool = True,
    seed: int = 42,
) -> StrategyResult:
    """Summarise conditional daily round trips. Sharpe/Sortino use simple returns; equity and MDD use log increments. MAR=0 uses all periods. Annualisation does not correct missing calendar outcomes."""
    net = net_returns(position, realised, slippage_per_side)
    n = len(net)
    if n < 2:
        raise ValueError(f"a strategy needs at least two periods, got {n}")

    simple = np.expm1(net)
    mean = float(simple.mean())
    sd = float(simple.std(ddof=1))
    downside_sd = float(np.sqrt(np.mean(np.minimum(simple, 0.0) ** 2)))
    sharpe = mean / sd if sd > 0 else float("nan")
    sortino = (
        mean / downside_sd * math.sqrt(PERIODS_PER_YEAR)
        if downside_sd > 0
        else float("nan")
    )
    low, high = (
        _stationary_bootstrap_mdd(net, seed=seed)
        if mdd_interval
        else (float("nan"), float("nan"))
    )
    total_log = float(net.sum())
    return StrategyResult(
        n_periods=n,
        n_flat_days=n_flat_days,
        mean_net=mean,
        sharpe_per_period=sharpe,
        sharpe_annualised=sharpe * math.sqrt(PERIODS_PER_YEAR),
        sortino_annualised=sortino,
        max_drawdown=max_drawdown(net),
        mdd_ci_low=low,
        mdd_ci_high=high,
        # Half the mean absolute position change: expected round trips per period.
        turnover_per_period=float(np.mean(position)),
        net_log_return=total_log,
        net_total_return=float(math.expm1(total_log)),
    )


def _flat_days(frame: pl.DataFrame) -> int:
    """Unavailable daily slots, NOT a bound on unobserved loss or drawdown.

    The legacy result field n_flat_days is retained for file compatibility;
    it counts excluded slots, not verified live decisions to remain in cash.
    """
    return int(frame["block"].n_unique() * BLOCK_DAYS - frame.height)


def run_strategy(
    preds: pl.DataFrame,
    meta: dict,
    slippage_per_side: float,
    mdd_interval: bool = True,
    seed: int = 42,
) -> StrategyResult:
    """Conditional long/cash daily round-trip simulation for one saved forecast run."""
    frame = positions(preds, float(meta["sigma_g"]), float(meta["mu_g"]))
    return summarise(
        frame.get_column("position").to_numpy().astype(np.float64),
        frame.get_column("realised_raw").to_numpy().astype(np.float64),
        slippage_per_side,
        _flat_days(frame),
        mdd_interval=mdd_interval,
        seed=seed,
    )


def buy_and_hold(
    preds: pl.DataFrame,
    meta: dict,
    slippage_per_side: float,
    mdd_interval: bool = True,
    seed: int = 42,
) -> StrategyResult:
    """Always-long DAILY ROUND TRIPS on observed slots; retained API name, not uninterrupted buy-and-hold."""
    frame = positions(preds, float(meta["sigma_g"]), float(meta["mu_g"]))
    return summarise(
        np.ones(frame.height),
        frame.get_column("realised_raw").to_numpy().astype(np.float64),
        slippage_per_side,
        _flat_days(frame),
        mdd_interval=mdd_interval,
        seed=seed,
    )


def jobson_korkie_memmel(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Test of equal Sharpe ratios with Memmel's (2003) correction.

    Root §13.5 asks for a test or an interval on the Sharpe difference rather
    than a bare point. Returns ``(z, two_sided_p)`` for ``SR_a - SR_b``.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) != len(b):
        raise ValueError(f"paired test needs equal lengths, got {len(a)} and {len(b)}")
    t = len(a)
    mu_a, mu_b = float(a.mean()), float(b.mean())
    sd_a, sd_b = float(a.std(ddof=1)), float(b.std(ddof=1))
    if sd_a <= 0 or sd_b <= 0:
        return float("nan"), float("nan")
    cov = float(np.cov(a, b, ddof=1)[0, 1])
    theta = (
        2.0 * sd_a**2 * sd_b**2
        - 2.0 * sd_a * sd_b * cov
        + 0.5 * mu_a**2 * sd_b**2
        + 0.5 * mu_b**2 * sd_a**2
        - (mu_a * mu_b / (sd_a * sd_b)) * cov**2
    ) / t
    numerator = sd_b * mu_a - sd_a * mu_b
    if theta <= 0:
        # theta vanishes **exactly** when the two series coincide: with
        # ``a == b`` the expression collapses to ``m^2 s^2 - m^2 s^2``. The
        # difference is then identically zero with zero variance, so the answer
        # is z = 0 rather than undefined -- two identical strategies have
        # identical Sharpe ratios. A vanishing theta beside a non-zero numerator
        # is a genuine breakdown and stays undefined.
        if abs(numerator) <= 1e-15:
            return 0.0, 1.0
        return float("nan"), float("nan")
    z = numerator / math.sqrt(theta)
    return z, 2.0 * (1.0 - normal_cdf(abs(z)))


def deflated_sharpe(
    sharpe_per_period: float,
    T: int,
    skew: float,
    kurtosis: float,
    n_trials: int,
    var_sharpe: float,
) -> float:
    """Bailey & Lopez de Prado's Deflated Sharpe Ratio, computed **per origin**.

    `D46` made this computable. The earlier prescription -- ``N`` = the whole
    development trial count, about 837 -- could not be executed, because it named
    neither ``V[SR]`` nor the skewness and kurtosis the statistic needs; and if it
    could, it would return about 0 by construction: at ``N = 837`` and ``T = 180``
    the threshold sits at ``SR0 + 1.645/sqrt(T-1) ~ SR0 + 0.123``, essentially
    unmeetable. That is a second guaranteed null beside `D23`'s, reading to a
    referee as either a failed strategy or a misapplied statistic with no way to
    tell which.

    ``N`` is therefore the number of distinct strategy configurations evaluated on
    **this origin's** test span, and ``var_sharpe`` the observed variance of their
    Sharpe ratios. The development total is reported separately in Limitations and
    is *not* ``N``: those runs span largely disjoint test periods, seeds, horizons
    and baselines that never competed for one backtest, whereas the DSR counts
    candidates selected from one return series.

    Args:
        sharpe_per_period: **Per period**, never annualised.
        T: Number of periods.
        skew: Sample skewness of the per-period returns.
        kurtosis: Sample kurtosis, not excess.
        n_trials: Configurations evaluated on this origin's span.
        var_sharpe: Variance of their per-period Sharpe ratios.

    Returns:
        The probability that the observed Sharpe exceeds what selection alone
        would have produced.
    """
    if n_trials < 2 or not var_sharpe > 0 or T < 2:
        return float("nan")
    threshold = math.sqrt(var_sharpe) * (
        (1.0 - EULER_MASCHERONI) * normal_quantile(1.0 - 1.0 / n_trials)
        + EULER_MASCHERONI * normal_quantile(1.0 - 1.0 / (n_trials * math.e))
    )
    denominator = (
        1.0
        - skew * sharpe_per_period
        + 0.25 * (kurtosis - 1.0) * sharpe_per_period**2
    )
    if not denominator > 0:
        return float("nan")
    return float(
        normal_cdf(
            (sharpe_per_period - threshold) * math.sqrt(T - 1) / math.sqrt(denominator)
        )
    )


def moments(net: np.ndarray) -> tuple[float, float]:
    """Sample skewness and non-excess kurtosis of a return series."""
    centred = np.asarray(net, dtype=np.float64) - np.mean(net)
    sd = centred.std(ddof=0)
    if sd <= 0:
        return float("nan"), float("nan")
    return float((centred**3).mean() / sd**3), float((centred**4).mean() / sd**4)


def _origin_run_ids(roots: list[Path], origin_index: int, pred_len: int) -> list[str]:
    """Every run at one origin and horizon covering the full six-block span.

    The falsification arm is excluded: it is scored on blocks 4-6 only, so its
    strategy runs on half the span and its Sharpe is not a candidate the others
    competed against.
    """
    found: set[str] = set()
    for root in roots:
        pattern = f"*_o{origin_index:02d}_K*_H{pred_len:03d}_s*.parquet"
        for path in (root / "preds").glob(pattern):
            if not path.stem.startswith("itrf_"):
                found.add(path.stem)
    return sorted(found)


def economics_table(
    roots: list[Path], keys: list[tuple[str, int]], origin_indices: tuple[int, ...],
    slippages: tuple[float, ...] = SLIPPAGE_BAND, pred_len: int = PRED_LEN, seed: int = 42,
) -> pl.DataFrame:
    """Conditional long/cash simulation; inferential trading claims withheld."""
    if pred_len != 24:
        raise ValueError("the economic protocol requires H=24")
    rows = []
    for origin_index in origin_indices:
        n_trials = len(_origin_run_ids(roots, origin_index, pred_len))
        for model, k in keys:
            run_id = f"{model}_o{origin_index:02d}_K{k:02d}_H{pred_len:03d}_s42"
            meta = load_meta(run_id, roots)
            frame = positions(load_predictions(run_id, roots), float(meta["sigma_g"]), float(meta["mu_g"]))
            pos, realised = [frame[n].to_numpy().astype(np.float64) for n in ("position", "realised_raw")]
            for slippage in slippages:
                result = summarise(pos, realised, slippage, _flat_days(frame), mdd_interval=False, seed=seed)
                hold = summarise(np.ones(len(pos)), realised, slippage, _flat_days(frame), mdd_interval=False)
                rows.append({
                    "model": f"{model}-K{k}", "origin_index": origin_index, "origin": str(meta["origin"]),
                    "slippage_per_side": slippage, **asdict(result),
                    "hold_sharpe_annualised": hold.sharpe_annualised, "hold_net_total_return": hold.net_total_return,
                    "jk_memmel_z": float("nan"), "jk_memmel_p": float("nan"), "dsr": float("nan"),
                    "dsr_n_trials": n_trials, "dsr_var_sharpe": float("nan"),
                    "evaluation_status": "conditional on future target availability; not an executable backtest",
                    "policy": "long/cash; each daily trade opens and closes; terminal and both-side costs included",
                    "risk_return_scale": "simple daily return; 365-period annualisation is a conditional diagnostic",
                    "inference_status": "JK/DSR and MDD confidence intervals withheld",
                })
    return pl.DataFrame(rows)


#: Buy-and-hold's name in :func:`equity_curves`. Not a model tag: it runs no
#: model. Root §13.2 requires the economic result be reported beside it, and
#: :func:`economics_table` already carries it as the ``hold_*`` columns.
HOLD_LABEL: Final = "Always-long daily trades"


def equity_curves(
    roots: list[Path],
    keys: list[tuple[str, int]],
    origin_indices: tuple[int, ...],
    slippages: tuple[float, ...] = SLIPPAGE_BAND,
    pred_len: int = PRED_LEN,
) -> pl.DataFrame:
    """Figure 7's input -- cumulative net equity per (model, origin, slippage).

    A zero-cost curve is emitted alongside the three priced ones, because §13.5's
    figure is "before and after costs, at three slippage levels" and the
    before-costs line is what makes the cost band legible.
    """
    rows: list[pl.DataFrame] = []
    for origin_index in origin_indices:
        for model, k in keys:
            run_id = f"{model}_o{origin_index:02d}_K{k:02d}_H{pred_len:03d}_s42"
            meta = load_meta(run_id, roots)
            frame = positions(
                load_predictions(run_id, roots),
                float(meta["sigma_g"]),
                float(meta["mu_g"]),
            )
            position = frame.get_column("position").to_numpy().astype(np.float64)
            realised = frame.get_column("realised_raw").to_numpy().astype(np.float64)
            for slippage in (0.0, *slippages):
                net = net_returns(position, realised, slippage)
                if (model, k) == keys[0]:
                    # Once per (origin, slippage), not once per model. Root §13.2
                    # states the economic result AGAINST buy-and-hold -- +20.6%
                    # net against +29.0% -- so a figure without it lets the
                    # strategy's own curve read as skill. Same `hold_position`
                    # :func:`economics_table` uses for its ``hold_*`` columns, so
                    # the figure and the table cannot disagree.
                    hold = net_returns(np.ones(len(position)), realised, slippage)
                    rows.append(
                        pl.DataFrame(
                            {
                                "model": [HOLD_LABEL] * len(hold),
                                "origin": [str(meta["origin"])] * len(hold),
                                "origin_index": [origin_index] * len(hold),
                                "slippage_per_side": [slippage] * len(hold),
                                "period": np.arange(1, len(hold) + 1, dtype=np.int32),
                                "timestamp": frame.get_column("timestamp").to_numpy(),
                                "equity": np.exp(np.cumsum(hold)),
                            }
                        )
                    )
                rows.append(
                    pl.DataFrame(
                        {
                            "model": [f"{model}-K{k}"] * len(net),
                            "origin": [str(meta["origin"])] * len(net),
                            "origin_index": [origin_index] * len(net),
                            "slippage_per_side": [slippage] * len(net),
                            "period": np.arange(1, len(net) + 1, dtype=np.int32),
                            "timestamp": frame.get_column("timestamp").to_numpy(),
                            "equity": np.exp(np.cumsum(net)),
                        }
                    )
                )
    return pl.concat(rows)
