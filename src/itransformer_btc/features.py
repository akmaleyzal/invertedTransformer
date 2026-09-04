"""The twelve variates, and the K ladder cut over them.

All twelve are **engineered**; none is a raw kline column. What is excluded is a
class, not a list: technical indicators, multi-bar rolling statistics, calendar
dummies, cross-asset, on-chain, macro. Root §5.3 carries the argument — K is
RQ1's independent variable, and anything outside families F1–F5 breaks the
taxonomy that makes K_eff interpretable.

**No variate uses a rolling window.** Every one is a pure per-bar function of
the current bar, except ``r``, which uses the current and previous close. That
is a structural safety property rather than a style choice: with no rolling
window anywhere, the ``center=True`` leak class is unrepresentable (root §5.3).

**The ladder is cumulative and its order is load-bearing.** Column order is
ladder order, so rung K is exactly the first K columns and ``r`` is channel 0 at
every rung. Root §6.2 requires the loss be MSE on the **target channel only** at
every rung; with ``r`` pinned at index 0 that is one constant, not a lookup.

Upstream
--------
**All twelve variates are written here in polars from their published closed
forms. Nothing is taken from a technical-analysis library, and that exclusion is
a design decision rather than an oversight** -- RSI, MACD and Bollinger belong
to no F1-F5 family, so admitting them would break the taxonomy that makes
``K_eff`` interpretable and render the K=8 versus K=12 contrast meaningless
(`D37`).

The F2 volatility estimators, each per-bar:

- M. Parkinson, "The extreme value method for estimating the variance of the
  rate of return," *J. Business*, vol. 53, no. 1, pp. 61-65, 1980.
- M. B. Garman and M. J. Klass, "On the estimation of security price
  volatilities from historical data," *J. Business*, vol. 53, no. 1,
  pp. 67-78, 1980.
- L. C. G. Rogers and S. E. Satchell, "Estimating variance from high, low and
  closing prices," *Ann. Appl. Probab.*, vol. 1, no. 4, pp. 504-512, 1991.

Two departures worth stating where a reader meets the code. **No estimator is
trailing-averaged** (`D13`): every variate is a pure per-bar function, which is
what makes the ``center=True`` leakage class structurally unrepresentable and
licenses root §8.3's no-embargo argument (`D15`). And Rogers-Satchell **is not
strictly positive** -- it vanishes on the 33 shadowless bars in this sample --
so it is taken as ``log(RS + 1e-9)``, the floor chosen to land inside the
measured support rather than as 33 out-of-support spikes; Parkinson and
Garman-Klass need no floor (`D52a`).
:data:`itransformer_btc.config.SOURCE_PROVENANCE` carries this row in full.
"""

from __future__ import annotations

import math
from typing import Final

import polars as pl

from itransformer_btc.segments import HOUR_MS, usable_mask

#: Ladder order. Rung K is ``VARIATE_ORDER[:K]``. Root §5.2's unique consistent
#: cut (`D01` — the source specification's K=8 rung summed to nine and
#: double-assigned ``log_mean_trade_size``).
VARIATE_ORDER: Final[tuple[str, ...]] = (
    # F1 price trajectory — K=1 is `r` alone
    "r",
    "upper_shadow",
    "lower_shadow",
    # F3 intensity, first member — completes K=4
    "log_quote_volume",
    # K=8: intensity, order flow, intrabar location
    "log_trade_count",
    "taker_buy_ratio",
    "signed_flow",
    "vwap_location",
    # K=12: the F2 volatility estimators + the dependent intensity member
    "log_parkinson",
    "log_garman_klass",
    "log_rogers_satchell",
    "log_mean_trade_size",
)

TARGET: Final = "r"
TARGET_INDEX: Final = 0

#: Parkinson's normaliser, ``1 / (4 ln 2)``.
_PARKINSON_C: Final = 1.0 / (4.0 * math.log(2.0))
#: Garman–Klass's second-term coefficient, ``2 ln 2 - 1`` ≈ 0.386. Strictly
#: below 0.5, which is what keeps the estimator positive: ``|ln(C/O)| <=
#: ln(H/L)`` because C and O both lie in ``[L, H]``, so GK >= 0.114 (ln H/L)^2.
_GK_C: Final = 2.0 * math.log(2.0) - 1.0

#: Stabiliser for Rogers–Satchell only (`D52`).
#:
#: Root §5.1 claims all three F2 estimators are "strictly positive once H == L
#: bars are excluded". That holds for Parkinson (proportional to ``(ln H/L)^2``)
#: and for Garman–Klass (bounded below by ``0.114 (ln H/L)^2``), but **not** for
#: Rogers–Satchell, which is
#:
#:     ln(H/C) ln(H/O) + ln(L/C) ln(L/O)
#:
#: and vanishes on any bar with no shadows at all — H equal to one of O/C and L
#: equal to the other. Such a bar has H > L, carries real trade information, and
#: passes the segment law; it is a marubozu, not a degenerate bar. Measured: 33
#: of 75,091 usable bars, 0.044%.
#:
#: ``1e-9`` is chosen so ``log(kappa) = -20.7`` lands **inside the measured
#: support** of ``log RS`` — median -10.91, 0.1st percentile -17.57, minimum
#: -23.5 — in the low tail where a shadowless bar belongs. A hard floor far
#: below support (1e-12 gives -27.6, about -11 sigma) would instead create 33
#: spikes that distort the instance normalisation of every window containing
#: one, and would smuggle a categorical marubozu flag into a continuous
#: variate — the convenience-variate failure root §5.2 forbids. The shift it
#: applies to a typical bar is negligible: at the median RS of 1.8e-5, adding
#: 1e-9 moves ``log RS`` by 5e-5.
#:
#: Deliberately **not** applied to Parkinson or Garman–Klass: both are provably
#: positive, their measured minima are 1.16e-8 and 1.48e-8, and adding kappa
#: there would shift the smallest values by roughly 8% for no reason.
_RS_STABILISER: Final = 1e-9


#: Two eight-variate subsets, built to differ in **effective** rank while holding
#: nominal K fixed at 8 (`D70`).
#:
#: RQ1 asks whether the marginal benefit of added variates is governed by nominal
#: count or by effective dimensionality, and the ladder answers it only through a
#: panel: K and K_eff move together there, ``corr(K, K_eff) = 0.828``, so the two
#: explanations are separated by a non-nested test rather than by contrast. These
#: two rungs separate them **directly** — same K, same target, same everything
#: else, and PR is the only thing that moves.
#:
#: - ``redundant`` loads F2 whole. All three volatility estimators carry about one
#:   independent degree of freedom between them (root §5.1), and all three of F3
#:   are present where the third is the difference of the first two. Low PR by
#:   construction.
#: - ``orthogonal`` takes one or two from each of F1-F5 and never doubles up
#:   inside a family. High PR by construction.
#:
#: ``r`` leads both, because :data:`TARGET_INDEX` is 0 and every consumer reads
#: the target there.
MATCHED_K_SUBSETS: dict[str, tuple[str, ...]] = {
    "redundant": (
        "r",
        "log_parkinson",
        "log_garman_klass",
        "log_rogers_satchell",
        "log_quote_volume",
        "log_trade_count",
        "log_mean_trade_size",
        "taker_buy_ratio",
    ),
    "orthogonal": (
        "r",
        "upper_shadow",
        "lower_shadow",
        "log_parkinson",
        "log_quote_volume",
        "log_trade_count",
        "taker_buy_ratio",
        "vwap_location",
    ),
}


def ladder_columns(k: int) -> list[str]:
    """The variate names at rung ``k``.

    Raises:
        ValueError: If ``k`` is not one of the pre-registered rungs. Rungs are
            fixed before any model runs (root §3); an ad-hoc K is a new
            experiment and must be declared as one.
    """
    if k not in (1, 4, 8, 12):
        raise ValueError(f"K must be a pre-registered rung 1/4/8/12, got {k}")
    return list(VARIATE_ORDER[:k])


def build_features(frame: pl.DataFrame) -> pl.DataFrame:
    """Compute all twelve variates, per segment, dropping what is undefined.

    ``r`` is computed **within** each segment. Computing it on a concatenated
    series would inject giant cross-gap returns into ``mu_g`` and ``sigma_g``
    before any window is excluded (root §4.3): the 2018-02-08 outage would book
    a 33-hour move as a one-hour return, and the scaler would be fitted on it.

    The first bar of each segment therefore yields a null ``r`` and is dropped.
    That shortens every segment by one bar, which the window enumerator picks up
    for free — the dropped bar leaves a two-hour jump at the segment head, and
    :func:`itransformer_btc.segments.build_segments` splits on any jump.

    Args:
        frame: Bars carrying ``ts_ms`` and the raw kline columns. ``usable`` is
            derived if absent.

    Returns:
        ``ts_ms``, ``usable`` (all True), and the twelve variates in ladder
        order as ``Float64``, with no nulls.

    Raises:
        ValueError: If any variate is null or non-finite. Both are impossible
            once the segment law has excluded zero-volume and ``H == L`` bars,
            so either means the exclusion did not run.
    """
    if "usable" not in frame.columns:
        frame = usable_mask(frame)

    rows = frame.filter(pl.col("usable")).sort("ts_ms")

    # Segment identity from the timestamp alone. Excluded bars are already gone,
    # so they show up here as jumps, exactly as downtime does.
    rows = rows.with_columns(
        (pl.col("ts_ms").diff().fill_null(HOUR_MS) != HOUR_MS).cum_sum().alias("_seg")
    )

    log_h_l = (pl.col("high") / pl.col("low")).log()
    log_c_o = (pl.col("close") / pl.col("open")).log()
    vwap = pl.col("quote_volume") / pl.col("volume")

    out = rows.with_columns(
        # -- F1 price trajectory, 3 dof -------------------------------------
        (pl.col("close").log() - pl.col("close").log().shift(1).over("_seg")).alias("r"),
        (pl.col("high") / pl.max_horizontal("open", "close")).log().alias("upper_shadow"),
        (pl.min_horizontal("open", "close") / pl.col("low")).log().alias("lower_shadow"),

        # -- F3 intensity, 2 dof — the third is the difference of the first two
        pl.col("quote_volume").log().alias("log_quote_volume"),
        pl.col("trades").log().alias("log_trade_count"),
        (pl.col("quote_volume") / pl.col("trades")).log().alias("log_mean_trade_size"),

        # -- F4 order flow, 1–2 dof -----------------------------------------
        # Base-denominated: the canonical buyer-initiated volume share (`D12`).
        # The quote-denominated variant is a robustness check, not the default.
        (pl.col("taker_buy_base") / pl.col("volume")).alias("taker_buy_ratio"),

        # -- F5 intrabar location, 1 dof ------------------------------------
        # A total function, not a partial one, because H == L bars are segment
        # breaks (`D14`). Without that exclusion this divides by zero.
        ((vwap - pl.col("close")) / (pl.col("high") - pl.col("low"))).alias("vwap_location"),

        # -- F2 volatility estimators, ~1 dof, redundant by construction -----
        # Per-bar with no trailing average (`D13`). Pre-smoothing over 24 bars
        # is strictly less informative: the model can compute that average
        # itself and cannot recover what smoothing destroyed (root §5.3).
        # Parkinson and Garman-Klass are provably positive once H > L, so their
        # logs are total. Rogers-Satchell is NOT — it vanishes on shadowless
        # bars — hence the stabiliser, and only there (`D52`).
        (_PARKINSON_C * log_h_l.pow(2)).log().alias("log_parkinson"),
        (0.5 * log_h_l.pow(2) - _GK_C * log_c_o.pow(2)).log().alias("log_garman_klass"),
        (
            (pl.col("high") / pl.col("close")).log() * (pl.col("high") / pl.col("open")).log()
            + (pl.col("low") / pl.col("close")).log() * (pl.col("low") / pl.col("open")).log()
            + _RS_STABILISER
        ).log().alias("log_rogers_satchell"),
    ).with_columns(
        # A deterministic product of two other K=8 members. Kept, with the
        # dependence disclosed (`D12`): it weakens the claim that K=8 is the
        # rung of maximum effective rank, and the measured participation ratio
        # settles that question rather than the argument doing so.
        (
            (2.0 * pl.col("taker_buy_ratio") - 1.0) * pl.col("log_quote_volume")
        ).alias("signed_flow"),
    )

    # The first bar of each segment has no predecessor inside its segment.
    out = out.filter(pl.col("r").is_not_null())

    out = out.select(["ts_ms", "usable", *VARIATE_ORDER]).with_columns(
        [pl.col(c).cast(pl.Float64) for c in VARIATE_ORDER]
    )

    offenders = {
        name: n
        for name in VARIATE_ORDER
        if (
            n := int(
                out.select(
                    (~pl.col(name).is_finite() | pl.col(name).is_null()).sum()
                ).item()
            )
        )
    }
    if offenders:
        raise ValueError(
            f"non-finite variate values: {offenders}. Every variate is total "
            f"once zero-volume and H == L bars are excluded by the segment law "
            f"(root §4.3 / `D14`), so this means the exclusion did not run."
        )
    return out
