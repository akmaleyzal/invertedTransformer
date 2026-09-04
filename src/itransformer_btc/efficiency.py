"""Root §4.5's preliminary market-efficiency tests.

Run once, reported in the Data section. The point is to convert "efficient
market" from an assumption into a finding: §4.5 forbids *claiming* the market is
efficient and requires stating that the evidence is mixed and time-varying
(Urquhart 2016; Nadarajah & Chu 2017; Bariviera 2017; Sensoy 2019), then
reporting our own numbers beside it.

Reported over two spans, because "time-varying" is a claim about variation and
one full-sample row cannot exhibit it: the whole sample, and each origin's
**21-month training sub-block** -- the same span the scaler is fitted on. Those
rows are descriptive and gate nothing, so the span rule `D44` imposes on K_eff
does not bind, but reading a test block merely to describe the data would still
be indefensible when avoiding it costs one filter.

``arch`` and ``statsmodels`` accept numpy arrays, so this module crosses root
§16's stats boundary without pandas ever entering the process.

Upstream
--------
**This is the only module in the package that calls a third-party statistics
implementation, and it does so at root §16's named boundary -- imported inside
the function that needs it, never at module level.**

- Variance ratio -- ``arch.unitroot.VarianceRatio``
  (https://github.com/bashtage/arch, NCSA; accessed 2026-09-03), implementing
  A. W. Lo and A. C. MacKinlay, "Stock market prices do not follow random
  walks: Evidence from a simple specification test," *Rev. Financial Stud.*,
  vol. 1, no. 1, pp. 41-66, 1988.
- ADF -- ``statsmodels.tsa.stattools.adfuller`` (BSD-3-Clause), implementing
  D. A. Dickey and W. A. Fuller, *J. Amer. Statist. Assoc.*, vol. 74, no. 366,
  pp. 427-431, 1979.
- ``hurst_rs`` -- **written here.** H. E. Hurst, "Long-term storage capacity of
  reservoirs," *Trans. Amer. Soc. Civil Eng.*, vol. 116, no. 1, pp. 770-799,
  1951. No package provides a rescaled-range estimator under a licence and an
  API this project already depends on, so taking a dependency to import a
  single function was the worse trade.

:data:`itransformer_btc.config.SOURCE_PROVENANCE` carries these rows in full.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import polars as pl

from itransformer_btc.config import ORIGINS, OriginLike

#: Lags for the Lo-MacKinlay variance ratio. Powers of two spanning two hours to
#: two thirds of a day, which brackets the horizons this study forecasts.
VR_LAGS: Final[tuple[int, ...]] = (2, 4, 8, 16)

#: Smallest R/S block. Below ~16 points the rescaled range is dominated by its
#: own small-sample bias and the log-log slope bends upward on white noise.
HURST_MIN_N: Final = 16

#: Blocks needed for the log-log regression to mean anything. Two points define a
#: line exactly and would report a slope with no residual to doubt it.
HURST_MIN_BLOCK_SIZES: Final = 3

#: "c" admits a non-zero drift in the random walk, which BTC plainly has. Passed
#: explicitly rather than left to the library default: root §16 forbids a magic
#: number, and a silent default is worse than one -- it is a magic number nobody
#: can see.
VR_TREND: Final = "c"


@dataclass(frozen=True, slots=True)
class VarianceRatioRow:
    """One Lo-MacKinlay variance ratio. ``vr`` near 1 is consistent with a random walk."""

    lag: int
    vr: float
    statistic: float
    p_value: float


@dataclass(frozen=True, slots=True)
class ADFRow:
    """Augmented Dickey-Fuller on log-returns, which should reject a unit root."""

    statistic: float
    p_value: float
    used_lag: int
    n_obs: int


def hurst_rs(x: np.ndarray, min_n: int = HURST_MIN_N, max_n: int | None = None) -> float:
    """Hurst exponent by rescaled range, read off the log-log plot as an OLS slope.

    ``H ~ 0.5`` on log-returns is the no-long-memory reading §4.5 pins. Block
    sizes are dyadic; at each size the series is cut into non-overlapping blocks
    and the mean R/S over them is the point that enters the regression.

    Args:
        x: The series to measure. Table 2 passes **log-returns**; passing a level
            series is the control that shows the estimator responds to memory at
            all, and returns ``H`` near 1.
        min_n: Smallest block. See :data:`HURST_MIN_N`.
        max_n: Largest block, defaulting to ``len(x) // 4`` so the largest size
            still averages over four blocks rather than reporting one range.

    Returns:
        The estimated Hurst exponent.

    Raises:
        ValueError: If fewer than :data:`HURST_MIN_BLOCK_SIZES` usable block
            sizes fit inside the series.
    """
    x = np.asarray(x, dtype=np.float64)
    n_total = len(x)
    if max_n is None:
        max_n = n_total // 4
    sizes = [n for n in (min_n * 2**i for i in range(64)) if n <= max_n]
    if len(sizes) < HURST_MIN_BLOCK_SIZES:
        raise ValueError(
            f"R/S needs at least {HURST_MIN_BLOCK_SIZES} block sizes between "
            f"{min_n} and {max_n}; got {len(sizes)} at n={n_total}"
        )

    logs_n: list[float] = []
    logs_rs: list[float] = []
    for n in sizes:
        blocks = x[: (n_total // n) * n].reshape(-1, n)
        deviate = np.cumsum(blocks - blocks.mean(axis=1, keepdims=True), axis=1)
        spread = deviate.max(axis=1) - deviate.min(axis=1)
        sd = blocks.std(axis=1, ddof=1)
        # A constant block has no scale to rescale by. Dropping it is not
        # imputation -- nothing is invented, the block simply carries no R/S.
        keep = sd > 0
        if not keep.any():
            continue
        logs_n.append(float(np.log(n)))
        logs_rs.append(float(np.log(float((spread[keep] / sd[keep]).mean()))))

    if len(logs_n) < HURST_MIN_BLOCK_SIZES:
        raise ValueError(
            f"only {len(logs_n)} block sizes carried a non-zero standard deviation"
        )
    slope, _ = np.polyfit(np.asarray(logs_n), np.asarray(logs_rs), 1)
    return float(slope)


def variance_ratios(
    r: np.ndarray, lags: tuple[int, ...] = VR_LAGS
) -> list[VarianceRatioRow]:
    """Lo-MacKinlay variance ratio at each lag, from a **return** series.

    ``arch.unitroot.VarianceRatio`` consumes a **level** series and differences
    it itself, so the returns are cumulated back to a level here rather than
    handed over raw. Fed the returns directly it reports ``VR = 1/lag`` -- 0.49,
    0.25, 0.12, 0.06 at lags 2, 4, 8, 16 on white noise -- the signature of
    over-differencing, and a p-value of 0.0000 that would read as decisive
    evidence against a random walk while being evidence of nothing at all. The
    unit tests pin both directions.

    The cumulation is safe across gaps and does not smuggle one in. ``r`` is
    computed **per segment** with each segment's first bar dropped (root §4.3),
    so the series contains no cross-gap return; the cumulative sum merely glues
    the segments into a pseudo-level whose first differences are exactly the
    returns the estimator then recovers. No value is fabricated at a boundary,
    which is what §2's no-imputation rule is about.

    Args:
        r: Log-returns.
        lags: Multi-period horizons for the numerator variance.

    Returns:
        One row per lag, in the order given.
    """
    from arch.unitroot import VarianceRatio  # root §16's named stats boundary

    level = np.cumsum(np.asarray(r, dtype=np.float64))
    rows: list[VarianceRatioRow] = []
    for lag in lags:
        ratio = VarianceRatio(level, lags=lag, trend=VR_TREND, overlap=True)
        rows.append(
            VarianceRatioRow(
                lag=lag,
                vr=float(ratio.vr),
                statistic=float(ratio.stat),
                p_value=float(ratio.pvalue),
            )
        )
    return rows


def adf(r: np.ndarray) -> ADFRow:
    """Augmented Dickey-Fuller with AIC lag selection, on log-returns."""
    from statsmodels.tsa.stattools import adfuller  # root §16's named stats boundary

    stat, p_value, used_lag, n_obs, *_ = adfuller(
        np.asarray(r, dtype=np.float64), autolag="AIC"
    )
    return ADFRow(
        statistic=float(stat),
        p_value=float(p_value),
        used_lag=int(used_lag),
        n_obs=int(n_obs),
    )


def _row(span: str, r: np.ndarray) -> dict[str, float | str | int]:
    """One Table 2 row: ADF, Hurst, and the variance ratio at every lag."""
    unit_root = adf(r)
    row: dict[str, float | str | int] = {
        "span": span,
        "n": int(len(r)),
        "adf_stat": unit_root.statistic,
        "adf_p": unit_root.p_value,
        "hurst": hurst_rs(r),
    }
    for ratio in variance_ratios(r):
        row[f"vr_{ratio.lag}"] = ratio.vr
        row[f"vr_p_{ratio.lag}"] = ratio.p_value
    return row


def _training_returns(features: pl.DataFrame, origin: OriginLike) -> np.ndarray:
    """Log-returns of one origin's 21-month training sub-block.

    ``[train_start, train_sub_end)``, half-open, matching
    :func:`itransformer_btc.keff._training_windows` exactly -- the two must cut
    the same span or Table 2 and Table 2b describe different data.
    """
    lo = int(origin.train_start.timestamp() * 1000)
    hi = int(origin.train_sub_end.timestamp() * 1000)
    return (
        features.filter((pl.col("ts_ms") >= lo) & (pl.col("ts_ms") < hi))
        .get_column("r")
        .to_numpy()
    )


#: Shortest sub-block the R/S regression can describe: enough rows for
#: :data:`HURST_MIN_BLOCK_SIZES` dyadic sizes, each averaged over four blocks.
MIN_SPAN_ROWS: Final = HURST_MIN_N * 2 ** (HURST_MIN_BLOCK_SIZES - 1) * 4


def efficiency_table(
    features: pl.DataFrame, origins: list[OriginLike] | None = None
) -> pl.DataFrame:
    """Table 2 -- one row for the whole sample, one per origin's training sub-block.

    Args:
        features: The frame :func:`itransformer_btc.features.build_features`
            returns, carrying ``ts_ms`` and ``r``.
        origins: Defaults to the full walk-forward grid.

    Returns:
        ``span, n, adf_stat, adf_p, hurst`` plus a ``vr_{lag}`` / ``vr_p_{lag}``
        pair per lag. The ``"full"`` row comes first; origin rows follow in
        walk-forward order.

    An origin whose sub-block is shorter than :data:`MIN_SPAN_ROWS` is skipped.
    On the real artifact that never happens -- the shortest sub-block holds
    roughly 15,000 bars -- so the branch exists for synthetic and truncated
    frames, and a caller can tell it fired by comparing the row count against
    ``len(origins) + 1``.
    """
    grid = list(origins if origins is not None else ORIGINS)
    rows = [_row("full", features.get_column("r").to_numpy())]
    for origin in grid:
        returns = _training_returns(features, origin)
        if len(returns) < MIN_SPAN_ROWS:
            continue
        rows.append(_row(origin.label, returns))
    return pl.DataFrame(rows)
