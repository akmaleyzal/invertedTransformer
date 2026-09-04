"""Effective dimensionality — RQ1's independent variable, measured before training.

Root §5.4. These statistics run **before any model does**, and two of the study's
three claimed contributions rest on them: the separation of nominal K from
effective dimensionality, and the K-versus-K_eff horse race in §9.1.

Three constraints shape every function here, and each closes a leak the earlier
design left open:

* **`D44` — span.** Every reported K_eff declares its span, and the one that
  feeds RQ1's regression is computed **per origin on that origin's own 21-month
  training sub-block**. Nothing previously forbade computing it over 2018-2026,
  a span containing every origin's test blocks; the regressor would then be
  estimated on the same data as the outcome and RQ1's claim would be partly
  circular, while §11's fatal checklist item still passed because it audits only
  the *gate*. That was the one leakage path surviving every checklist item by
  construction.
* **`D44` — construct validity.** PR on the K x K *contemporaneous* correlation
  matrix is blind to cross-lag structure, while the model consumes a K x 96
  block and embeds each variate's entire lookback. Two variates can be
  near-uncorrelated contemporaneously yet near-redundant to a model with a
  96-hour lookback. A lookback-aware measure is therefore reported on the same
  rungs, and the divergence between them is reported whatever it turns out to
  be. If the construct does not correspond to what the architecture consumes,
  the second contribution is a measurement-validity failure rather than a
  finding — which is what a methods referee will spend the review on.
* **`D04` — the instance-normalisation confound.** ``use_norm=True`` divides each
  window by its own per-variate sigma over L, so the F2 estimators contribute
  *shape*, not *level*. The 8->12 rung can flatten for a reason that has nothing
  to do with redundancy. PR is therefore measured on window-normalised features
  as well as raw, both reported, and the confound disclosed in Limitations
  whatever RQ1 returns.

Upstream
--------
**Written here on numpy; the participation ratio has no reference
implementation to vendor.** It is a random-matrix statistic, not a library
function, and it enters this study as RQ1's independent variable rather than as
a diagnostic -- which is why its span and its matrix are pinned rather than
left to a default.

- L. Laloux, P. Cizeau, J.-P. Bouchaud, and M. Potters, "Noise dressing of
  financial correlation matrices," *Phys. Rev. Lett.*, vol. 83, no. 7,
  pp. 1467-1470, 1999.
- V. Plerou, P. Gopikrishnan, B. Rosenow, L. A. N. Amaral, T. Guhr, and
  H. E. Stanley, "Random matrix approach to cross correlations in financial
  data," *Phys. Rev. E*, vol. 65, no. 6, 066126, 2002.

Two departures from the naive reading, both load-bearing. PR is taken on the
**correlation** matrix and never the covariance one: the covariance spectrum is
not monotone in K, and its ordering is a statement about units, since
``log_quote_volume``'s deviations sit two orders of magnitude above ``r``'s
(`D53a`, `D53b`, `D81`). And every reported ``K_eff`` -- the RQ1 regressor
included -- is measured on a **training-only** span, per origin, so the
regressor never reads the test period (`D02`, `D44`).
:data:`itransformer_btc.config.SOURCE_PROVENANCE` carries this row in full.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import polars as pl

from itransformer_btc.config import (
    DATA_START,
    FIRST_ORIGIN,
    K_LADDER,
    ORIGINS,
    SEQ_LEN,
    WINDOW_SPAN,
    OriginLike,
)
from itransformer_btc.features import ladder_columns
from itransformer_btc.segments import HOUR_MS
from itransformer_btc.splits import window_starts

#: Root §8.5's Stage 3b trigger, pre-registered numerically **before** measuring:
#: a gate without a number stated in advance is not a gate (`D02`).
GATE_PR_FLOOR: float = 5.0

#: Windows sampled for the lookback-aware measures. The stable rank is a per
#: window SVD, so the cost is linear in this, and the sample is drawn by a fixed
#: stride — deterministic, not random, so the number is regenerable under root
#: §12 without carrying a seed.
LOOKBACK_SAMPLE: int = 2_000


def participation_ratio(eigenvalues: np.ndarray) -> float:
    """``PR = (sum lambda)^2 / sum lambda^2``, bounded in ``[1, K]``.

    PR is 1 when one eigenvalue carries everything and K when the spectrum is
    flat, so it reads directly as "how many independent directions are actually
    here". Negative eigenvalues from floating-point noise on a
    positive-semidefinite matrix are clipped to zero rather than dropped, since
    dropping them would change the trace and so the numerator.

    Raises:
        ValueError: If the spectrum sums to zero — a degenerate block that no
            downstream number could be computed from.
    """
    lam = np.clip(np.asarray(eigenvalues, dtype=np.float64), 0.0, None)
    total = lam.sum()
    if total <= 0:
        raise ValueError("degenerate spectrum: eigenvalues sum to zero")
    return float(total * total / np.square(lam).sum())


def contemporaneous_pr(values: np.ndarray) -> float:
    """PR of the ``K x K`` correlation matrix — Table 2b's first column.

    Args:
        values: ``(n, K)`` observations, one row per bar.

    The correlation matrix rather than the covariance, so the statistic is
    invariant to the arbitrary units the twelve variates carry: log-returns,
    log-volumes and a bounded ratio do not share a scale, and on the covariance
    the log-volume channel would dominate the spectrum for that reason alone.
    """
    corr = np.atleast_2d(
        np.corrcoef(np.asarray(values, dtype=np.float64), rowvar=False)
    )
    return participation_ratio(np.linalg.eigvalsh(corr))


def window_normalised_pr(windows: np.ndarray) -> float:
    """PR after per-window standardisation over L — `D04`'s required companion.

    Args:
        windows: ``(n, L, K)``.

    This reproduces exactly what ``use_norm=True`` hands the embedding: each
    channel of each window centred and divided by its own sigma over the
    lookback. Level information is gone by construction, so if the raw and
    normalised PR disagree at the K=12 rung, the F2 estimators' apparent
    redundancy is an artefact of the normalisation rather than a property of the
    data — and RQ1's axis is confounded in a way no post-hoc analysis removes.
    """
    x = np.asarray(windows, dtype=np.float64)
    mean = x.mean(axis=1, keepdims=True)
    std = np.sqrt(x.var(axis=1, keepdims=True) + 1e-12)
    return contemporaneous_pr(((x - mean) / std).reshape(-1, x.shape[2]))


def stable_rank(matrix: np.ndarray) -> float:
    """``||M||_F^2 / ||M||_2^2`` — bounded in ``[1, min(rows, cols)]``.

    The lookback-aware measure that is *commensurable* with the contemporaneous
    PR: applied to a ``K x L`` block with ``K <= 12 < 96`` it lives in the same
    ``[1, K]`` interval, so the two sit in one table and can be compared rung by
    rung. The ``K*L x K*L`` covariance PR below cannot — its ceiling is ``K*L``,
    which is 1,152 at K=12.
    """
    singular = np.linalg.svd(np.asarray(matrix, dtype=np.float64), compute_uv=False)
    if singular[0] <= 0:
        raise ValueError("degenerate window block: largest singular value is 0")
    return float(np.square(singular).sum() / (singular[0] ** 2))


def lookback_stable_rank(windows: np.ndarray, sample: int = LOOKBACK_SAMPLE) -> float:
    """Mean stable rank of each window's ``K x L`` block, **as the model sees it**.

    Args:
        windows: ``(n, L, K)``.
        sample: Windows to evaluate, taken by a fixed stride across the whole
            span so the sample spreads over the sub-block rather than
            concentrating at its head.

    Each channel is standardised **within its window** first — exactly what
    ``use_norm=True`` does before the embedding. Centring alone is not enough and
    the difference is not cosmetic: measured on origin 1, the merely-centred
    version returns 1.00 / 1.00 / 1.16 / 1.65 across the four rungs, because
    ``log_quote_volume`` deviations are two orders of magnitude larger than
    ``r`` deviations in absolute terms, so one row dominates both the Frobenius
    and the spectral norm and the statistic reports "one effective direction" at
    every rung. That is a units artefact, not a finding about the data.

    With standardisation the quantity has a closed form worth stating: the block
    is ``K x L`` with unit-variance rows, so ``||M||_F^2 = K L`` and
    ``sigma_1^2 = L lambda_1`` where ``lambda_1`` is the leading eigenvalue of
    the **within-window** correlation matrix of the K channels. The stable rank
    is therefore ``K / lambda_1`` — the reciprocal of the dominant direction's
    share, bounded in ``[1, K]`` and directly comparable to the contemporaneous
    PR beside it.
    """
    x = np.asarray(windows, dtype=np.float64)
    if len(x) == 0:
        raise ValueError("no windows to measure")
    stride = max(1, len(x) // sample)
    blocks = np.transpose(x[::stride][:sample], (0, 2, 1))   # (m, K, L)
    blocks = blocks - blocks.mean(axis=2, keepdims=True)
    blocks = blocks / np.sqrt(np.square(blocks).mean(axis=2, keepdims=True) + 1e-12)
    return float(np.mean([stable_rank(b) for b in blocks]))


def lookback_correlation_pr(windows: np.ndarray) -> float:
    """PR of the ``K*L x K*L`` **correlation** spectrum — §5.4's first alternative.

    The correlation matrix, not the covariance §5.4 names literally. On the raw
    covariance the statistic is dominated by whichever channel happens to carry
    the largest variance, and it stops being monotone in K: measured on origin 1
    it returned 92.1 / 3.0 / 44.0 / 8.8 across the four rungs, where the drop
    from K=1 to K=4 is entirely the arrival of ``log_quote_volume`` and says
    nothing about dimensionality. Standardising the ``K*L`` columns first makes
    it scale-free, exactly as :func:`contemporaneous_pr` uses the correlation
    matrix for the same reason.

    This is the only measure here that sees genuine **cross-lag** structure — the
    stable rank above sees cross-*variate* structure inside a window. Its ceiling
    is ``K*L``, so it is **not** on the contemporaneous PR's scale; report it as
    a fraction of that ceiling (:attr:`KeffRow.pr_lookback_ratio`) when comparing
    rungs.

    **The name was wrong until `D81`, and the correction is not only cosmetic.**
    This function has computed the correlation spectrum since `D53b`, but it,
    its field and its parquet column were all called ``...covariance...``, so a
    reader checking whether `D53b`'s fix had landed found the word it replaced.
    Worse, `D53b` justified the fix by monotonicity in K and the correlation
    spectrum **is not monotone in K either** --- measured 92.1 / 21.9 / 37.3 /
    15.5 across the rungs, against the covariance's 92.1 / 3.0 / 44.0 / 8.8. What
    the change actually buys is scale-freeness, which is a real reason and the
    only one that survives; the ratio to the ``K*L`` ceiling then falls
    0.980 / 0.078 / 0.047 / 0.020, i.e. **anti**-monotone in K while the
    contemporaneous PR rises. The two K_eff constructs are ordinally opposed and
    root §4.1b has to say so (`D44`).
    """
    x = np.asarray(windows, dtype=np.float64)
    flat = x.reshape(len(x), -1)
    flat = flat - flat.mean(axis=0, keepdims=True)
    scale = np.sqrt(np.square(flat).mean(axis=0, keepdims=True) + 1e-24)
    flat = flat / scale
    gram = (flat.T @ flat) / max(1, len(flat) - 1)
    return participation_ratio(np.linalg.eigvalsh(gram))


@dataclass(frozen=True, slots=True)
class KeffRow:
    """One (origin, rung) cell of Table 2b."""

    origin: str
    origin_index: int
    k: int
    n_rows: int
    n_windows: int
    pr_raw: float
    pr_window_norm: float
    stable_rank_lookback: float
    pr_lookback_corr: float

    @property
    def pr_lookback_ratio(self) -> float:
        """``pr_lookback_corr / (K * L)`` — the cross-lag PR as a share of its ceiling.

        The raw value lives in ``[1, K*L]`` and so cannot be compared rung to
        rung; this can.
        """
        return self.pr_lookback_corr / (self.k * SEQ_LEN)

    @property
    def divergence(self) -> float:
        """``stable_rank - pr_raw`` — §5.4 requires this be reported as such.

        Positive means the lookback carries structure the contemporaneous
        correlation cannot see; negative means variates that look independent
        bar-to-bar turn redundant once 96 hours of each are in view. Either way
        it goes in §4.1b: if the effective-dimensionality construct does not
        correspond to what the architecture consumes, the study's second claimed
        contribution is a measurement-validity failure rather than a finding.
        """
        return self.stable_rank_lookback - self.pr_raw


def _training_windows(
    features: pl.DataFrame, origin: OriginLike, k: int, seq_len: int = SEQ_LEN
) -> tuple[np.ndarray, np.ndarray]:
    """Rows and windows of one origin's 21-month training sub-block.

    Both are cut from ``[train_start, train_sub_end)`` and nothing else — the
    span the scaler is fitted on. This is `D44`'s closure: RQ1's regressor may
    not see a single bar its outcome is measured on.
    """
    columns = ladder_columns(k)
    ts = features.get_column("ts_ms").to_numpy()
    values = features.select(columns).to_numpy()

    lo = int(origin.train_start.timestamp() * 1000)
    hi = int(origin.train_sub_end.timestamp() * 1000)
    rows = values[(ts >= lo) & (ts < hi)]

    starts = window_starts(
        ts, origin.train_start, origin.train_sub_end, "contained", WINDOW_SPAN
    )
    if len(starts) == 0:
        raise ValueError(f"origin {origin.label}: no training window to measure")
    idx = starts[:, None] + np.arange(seq_len)[None, :]
    return rows, values[idx]


def keff_row(features: pl.DataFrame, origin: OriginLike, k: int) -> KeffRow:
    """Measure every K_eff variant for one (origin, rung) cell."""
    rows, windows = _training_windows(features, origin, k)
    return KeffRow(
        origin=origin.label,
        origin_index=origin.index,
        k=k,
        n_rows=len(rows),
        n_windows=len(windows),
        pr_raw=contemporaneous_pr(rows),
        pr_window_norm=window_normalised_pr(windows),
        stable_rank_lookback=lookback_stable_rank(windows),
        pr_lookback_corr=lookback_correlation_pr(windows),
    )


def keff_table(
    features: pl.DataFrame,
    origins: list[OriginLike] | None = None,
    rungs: tuple[int, ...] = K_LADDER,
) -> pl.DataFrame:
    """Table 2b — every rung at every origin, on training spans only.

    Returns:
        One row per (origin, rung): the raw, window-normalised and two
        lookback-aware measures side by side, plus their divergence.

    K=1 is included even though its PR is identically 1. §9.1's horse race
    regresses on the rung's K_eff, and dropping the rung whose value is known in
    advance would unbalance the panel for no gain.
    """
    grid = list(origins if origins is not None else ORIGINS)
    return pl.DataFrame(
        [
            {
                "origin": row.origin,
                "origin_index": row.origin_index,
                "k": row.k,
                "n_rows": row.n_rows,
                "n_windows": row.n_windows,
                "pr_raw": row.pr_raw,
                "pr_window_norm": row.pr_window_norm,
                "stable_rank_lookback": row.stable_rank_lookback,
                "pr_lookback_corr": row.pr_lookback_corr,
                "pr_lookback_ratio": row.pr_lookback_ratio,
                "divergence": row.divergence,
            }
            for origin in grid
            for k in rungs
            for row in (keff_row(features, origin, k),)
        ]
    )


def corr_k_keff(table: pl.DataFrame, column: str = "pr_raw") -> float:
    """``corr(K, K_eff)`` across the rungs — §9.1 requires it in Table 2b.

    A reader is entitled to this before reading the non-nested comparison: if it
    sits near 1, the two theories are close to collinear and the horse race has
    little to separate them, whatever the reported p-value says.
    """
    means = table.group_by("k").agg(pl.col(column).mean().alias("keff")).sort("k")
    k = means.get_column("k").to_numpy().astype(np.float64)
    keff = means.get_column("keff").to_numpy()
    return float(np.corrcoef(k, keff)[0, 1])


def gate_pr(features: pl.DataFrame, k: int = 8) -> float:
    """Stage 3b's gate value — **pre-first-origin span only** (`D02`).

    Computed on ``[2018-01, 2020-01)``, which contains no origin's test block.
    The full-sample rolling PR is descriptive and may inform no design decision:
    every origin's test block lies inside it, so a ladder re-cut driven by it
    would be a design choice made with the answers already in hand.
    """
    columns = ladder_columns(k)
    ts = features.get_column("ts_ms").to_numpy()
    lo = int(DATA_START.timestamp() * 1000)
    hi = int(FIRST_ORIGIN.timestamp() * 1000)
    rows = features.select(columns).to_numpy()[(ts >= lo) & (ts < hi)]
    if len(rows) == 0:
        raise ValueError("the pre-first-origin span holds no usable bar")
    return contemporaneous_pr(rows)


def gate_verdict(measured: float, floor: float = GATE_PR_FLOOR) -> str:
    """Stage 3b's action, which is **disclosure, not a re-cut** (`D48`).

    §8.5 originally said a PR below the floor should re-cut the ladder, but
    `D01` establishes that exactly one consistent cut exists over F1-F5, so
    "re-cut" named no reachable alternative. The gate therefore reports, the
    grid proceeds unchanged, and the divergence from §5.2's expected values is
    disclosed in §4.1b. A gate whose only action is unreachable is not a gate.
    """
    if measured >= floor:
        return (
            f"PASS: measured PR at K=8 is {measured:.3f} >= {floor:.1f}. "
            f"Proceed; report the value in Table 2b."
        )
    return (
        f"DISCLOSE: measured PR at K=8 is {measured:.3f} < {floor:.1f}. "
        f"Root §8.5 / `D48` — proceed unchanged and disclose the divergence "
        f"from §5.2's expected K_eff in §4.1b. Do not re-cut the ladder: `D01` "
        f"leaves no second consistent cut over F1-F5."
    )


# -- Figure 2b: the descriptive rolling statistics (root §5.4) ---------------
#
# Both functions below are **descriptive only**, and the constraint is not
# decorative. Every origin's test block lies inside the full sample, so a
# full-sample rolling statistic **may inform no design decision** (`D02`): the
# statistic that *gates* the ladder is :func:`gate_pr` on the pre-first-origin
# span, and substituting one for the other would be the leak §5.4 exists to
# forbid. What these are for is establishing H2's premise --- that the
# microstructure-to-return mapping is regime-specific --- **before a single
# epoch runs**, which is why Figure 2b sits in the paper ahead of every result.

#: Root §5.4's window. Ninety days is long enough for an 8 x 8 correlation to be
#: estimated from ~2,160 bars and short enough to resolve a regime change.
ROLLING_WINDOW_DAYS: Final = 90

#: Step between consecutive windows. One day, so the curve is readable without
#: emitting one point per bar. Deterministic, so root §12 can regenerate it.
ROLLING_STEP_DAYS: Final = 1


def _rolling_spans(
    ts: np.ndarray, window_days: int, step_days: int
) -> list[tuple[int, int, int]]:
    """``(window_end_ms, lo, hi)`` for each window, sliced **by time, not position**.

    Position slicing would silently shorten a window wherever the series has a
    gap, and gaps are not uniform: 26 of 27 downtime blocks fall in 2018-2021 and
    none after 2023-03 (`D45`). A position-sliced "90-day" window would therefore
    span more calendar time early in the sample than late --- a trend in the
    estimator that a reader would read as a trend in the market.
    """
    window_ms = window_days * 24 * HOUR_MS
    step_ms = step_days * 24 * HOUR_MS
    spans: list[tuple[int, int, int]] = []
    for end in range(int(ts[0]) + window_ms, int(ts[-1]) + 1, step_ms):
        lo = int(np.searchsorted(ts, end - window_ms, side="left"))
        hi = int(np.searchsorted(ts, end, side="left"))
        spans.append((end, lo, hi))
    return spans


def rolling_pr(
    features: pl.DataFrame,
    k: int = 8,
    window_days: int = ROLLING_WINDOW_DAYS,
    step_days: int = ROLLING_STEP_DAYS,
) -> pl.DataFrame:
    """Rolling participation ratio over the full sample --- **descriptive only**.

    Args:
        features: The frame :func:`itransformer_btc.features.build_features`
            returns, carrying ``ts_ms`` and the twelve variates.
        k: Rung to measure. Eight is the rung RQ2 contrasts against K=1.

    Returns:
        ``window_end_ms, n_rows, pr`` --- one row per window.

    Never pass this to a regression and never compare it against
    :data:`GATE_PR_FLOOR`. It reads the test period by construction.
    """
    columns = ladder_columns(k)
    ts = features.get_column("ts_ms").to_numpy()
    values = features.select(columns).to_numpy()

    rows = []
    for end, lo, hi in _rolling_spans(ts, window_days, step_days):
        block = values[lo:hi]
        if len(block) <= len(columns):
            continue        # a correlation needs more rows than columns
        rows.append(
            {"window_end_ms": end, "n_rows": len(block), "pr": contemporaneous_pr(block)}
        )
    return pl.DataFrame(rows)


def rolling_ols_r2(
    features: pl.DataFrame,
    k: int = 8,
    window_days: int = ROLLING_WINDOW_DAYS,
    step_days: int = ROLLING_STEP_DAYS,
) -> pl.DataFrame:
    """In-window R^2 of ``r_{t+1} ~ (K features at t)`` --- **descriptive only**.

    Root §5.4: if this is unstable, H2's premise --- that the
    microstructure-to-return mapping is regime-specific --- is established before
    a single epoch runs. In-window rather than out-of-sample, deliberately: the
    question is whether the *relationship* moves, not whether it forecasts, and
    the second question is what the entire rest of the study answers.

    The target is formed **within a segment**: a row whose successor is not
    exactly one hour later is dropped, because ``r`` at the next row would then
    be the first return after an outage and the pair would straddle a gap the
    segment law (§4.3) exists to keep apart.

    Returns:
        ``window_end_ms, n_pairs, r2`` --- one row per window.
    """
    columns = ladder_columns(k)
    ts = features.get_column("ts_ms").to_numpy()
    values = features.select(columns).to_numpy()
    target = features.get_column("r").to_numpy()

    contiguous = np.zeros(len(ts), dtype=bool)
    contiguous[:-1] = np.diff(ts) == HOUR_MS

    rows = []
    for end, lo, hi in _rolling_spans(ts, window_days, step_days):
        # Clamped so the last window cannot ask for a successor that does not
        # exist; the mask and both slices then have one length by construction
        # rather than by a length check that fires after an IndexError would.
        n = min(hi, len(ts) - 1) - lo
        if n <= 0:
            continue
        keep = contiguous[lo : lo + n]
        x = values[lo : lo + n][keep]
        y = target[lo + 1 : lo + 1 + n][keep]
        if len(x) <= len(columns) + 1:
            continue
        design = np.column_stack([np.ones(len(x)), x])
        coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
        residual = y - design @ coefficients
        total = float(((y - y.mean()) ** 2).sum())
        rows.append({
            "window_end_ms": end,
            "n_pairs": len(y),
            "r2": float(1.0 - (residual ** 2).sum() / total) if total > 0 else 0.0,
        })
    return pl.DataFrame(rows)
