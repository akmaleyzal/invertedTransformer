"""Metrics, tests, and the three RQ estimators.

Root §9. Everything here consumes ``preds/{run_id}.parquet`` and
``meta/{run_id}.json`` and produces numbers that go straight into
``artifacts/paper_numbers.json`` — which is why nothing in this module reads a
model or a tensor. Root §12: a number that cannot be regenerated from a
persisted prediction file plus a config hash is a documented failure, not a
footnote.

Six decisions here are load-bearing, and each reverses something the source
design said:

* **`D31`** — Naive-RW is ``y_z = -mu_g/sigma_g``, never ``y_z = 0``. In scaler
  space zero means ``r_hat = mu_g``, the training-window mean hourly return: a
  constant-drift model wearing the EMH baseline's name. Measured, ``mu_g/sigma_g``
  spans -0.00818 … +0.01733 and **changes sign across origins**, so it is not a
  constant tilt a reader could mentally subtract.
* **`D23`** — ``D(i,b)`` lives on the **skill** scale, not on RelMSE. On RelMSE
  every pre-registered tau is arithmetically unreachable and RQ3 returns "no
  decay detected" by construction, before a single epoch runs.
* **`D05` follow-on** — the decay denominator is the **within-origin mean**, not
  block 1. One 30-day block under heavy tails would sit in the denominator of
  five quantities, making their errors perfectly correlated, and ``b*`` reads a
  threshold crossing straight off the series, so an unlucky block 1 moves the
  crossing by whole blocks.
* **`D42`** — every ratio metric is formed from **seed-averaged MSEs**, never
  from an average of per-seed ratios. The two differ by Jensen, and the second
  additionally requires pairing seed 42 at K=1 with seed 42 at K=8, which are
  independent training runs of different models: any of 5! orderings gives a
  different answer.
* **`D29`** — nested pairs get **Clark-West**, not Diebold-Mariano. Under the
  null with nested models and estimated parameters the loss differential has a
  mean shifted away from zero, so standard DM is systematically undersized
  against the alternative this study exists to establish.
* **`D34`** — the long-run variance estimator is **rectangular**, not Bartlett.
  Under the DM null, h-step optimal forecast errors are MA(h-1), so every
  autocovariance to lag 23 is genuinely nonzero and equally real; Bartlett
  weights shrink the lag-22 term by ~92%, understating the variance and
  producing exactly the over-optimistic p-values this module exists to prevent.

Upstream
--------
**Every estimator here is written on numpy from its published definition. No
statistical package is imported at module level, and none is vendored.** The
algorithms are cited to their papers; the implementations are this study's.

- ``dm_test`` — F. X. Diebold and R. S. Mariano, "Comparing predictive
  accuracy," *J. Bus. Econ. Statist.*, vol. 13, no. 3, pp. 253-263, 1995, with
  the small-sample correction of D. Harvey, S. Leybourne, and P. Newbold,
  "Testing the equality of prediction mean squared errors," *Int. J.
  Forecast.*, vol. 13, no. 2, pp. 281-291, 1997. Validated against R's
  ``forecast::dm.test``
  (https://pkg.robjhyndman.com/forecast/reference/dm.test.html, accessed
  2026-09-03) — a validation target, not a dependency.
- ``clark_west_test`` — T. E. Clark and K. D. West, "Approximately normal tests
  for equal predictive accuracy in nested models," *J. Econometrics*, vol. 138,
  no. 1, pp. 291-311, 2007. Used for every nested pair, where standard DM is
  undersized against the alternative this study exists to establish (`D29`).
- Wild cluster restricted bootstrap — A. C. Cameron, J. B. Gelbach, and
  D. L. Miller, *Rev. Econ. Statist.*, vol. 90, no. 3, pp. 414-427, 2008;
  J. G. MacKinnon, M. O. Nielsen, and M. D. Webb, *J. Econometrics*, vol. 232,
  no. 2, pp. 272-299, 2023; the ``(1 + count)/(1 + B)`` p-value from
  A. C. Davison and D. V. Hinkley, *Bootstrap Methods and their Application*,
  CUP, 1997. **Not the** ``wildboottest`` **package**, which root §9.2 names as
  the reference implementation but which this package does not import
  (`D42`, `D53d`).

The rectangular long-run variance is deliberate, and it is the reason
``statsmodels``' ``cov_hac`` is absent here: that estimator is Bartlett by
default (`D34`). :data:`itransformer_btc.config.SOURCE_PROVENANCE` carries
these rows in full.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from itransformer_btc.config import BLOCK_HOURS, PRED_LEN

#: ``{model}_o{origin:02d}_K{K:02d}_H{H:03d}_s{seed}`` (root §10.4).
RUN_ID_PATTERN = re.compile(
    r"^(?P<model>[a-z0-9]+)_o(?P<origin>\d{2})_K(?P<k>\d{2})"
    r"_H(?P<h>\d{3})_s(?P<seed>\d+)$"
)

HOUR_MS = 3_600_000

#: Root §3's pre-registered thresholds. The headline is 5%; the rest are
#: sensitivities. Choosing tau after seeing the decay curve is p-hacking.
TAU_HEADLINE: float = 0.05
TAU_SENSITIVITY: tuple[float, ...] = (0.025, 0.05, 0.10, 0.50)

#: Declared so `DecayResult.b_star` keeps its columns when every origin is
#: excluded (`D55`). An inferred schema over zero rows yields a frame with no
#: columns at all, and the caller's ``bs["b_star"]`` then raises rather than
#: reporting the pre-registered null.
B_STAR_SCHEMA: dict[str, pl.DataType] = {
    "origin": pl.Utf8,
    "tau": pl.Float64,
    "b_star": pl.Int64,
    "event": pl.Boolean,
}


# -- artifact I/O ------------------------------------------------------------


def parse_run_id(run_id: str) -> dict[str, int | str]:
    """Decompose a ``run_id`` into its five components.

    Raises:
        ValueError: If it does not match root §10.4's pattern. A run whose id
            cannot be parsed cannot be placed in the grid, and skipping it
            silently would drop a cell from a table without saying so.
    """
    match = RUN_ID_PATTERN.match(run_id)
    if match is None:
        raise ValueError(f"{run_id!r} is not a root §10.4 run_id")
    g = match.groupdict()
    return {
        "model": g["model"],
        "origin_index": int(g["origin"]),
        "k": int(g["k"]),
        "pred_len": int(g["h"]),
        "seed": int(g["seed"]),
    }


def _locate(run_id: str, roots: list[Path], kind: str, suffix: str) -> Path:
    for root in roots:
        candidate = Path(root) / kind / f"{run_id}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"{kind}/{run_id}{suffix} in none of {[str(r) for r in roots]}"
    )


def load_predictions(run_id: str, roots: list[Path]) -> pl.DataFrame:
    """Read one run's raw predictions.

    Roots are searched in order, so a working directory listed first shadows an
    older attached dataset — which is what lets a single re-run cell take effect
    without deleting the previous session's output.
    """
    return pl.read_parquet(_locate(run_id, roots, "preds", ".parquet"))


def load_meta(run_id: str, roots: list[Path]) -> dict:
    return json.loads(_locate(run_id, roots, "meta", ".json").read_text())


# -- point metrics -----------------------------------------------------------


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.square(np.asarray(y_true) - np.asarray(y_pred))))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rel_mse(model: float, naive: float) -> float:
    """``MSE_model / MSE_naive`` — controls for period difficulty (root §9.1)."""
    return model / naive


def r2_oos(model: float, naive: float) -> float:
    """``1 - RelMSE`` (`D20`) — the readable form of the same quantity.

    RelMSE near 1.00 is hard to read; ``R2_oos`` reads directly as skill against
    a random walk, and its sign is the whole question.
    """
    return 1.0 - rel_mse(model, naive)


def raw_rmse(mse_z: float, sigma_g: float) -> float:
    """RMSE back in raw log-return units — root §9.1's second reporting scale.

    "RMSE 0.0043 on hourly log-returns" tells a reader far more than "MSE 0.187
    on normalized data", and stating ``sigma_g`` is what lets the two reconcile.
    """
    return math.sqrt(mse_z) * sigma_g


def non_overlapping_mask(timestamps: np.ndarray) -> np.ndarray:
    """Window starts whose forecast period opens at **00:00 UTC** (`D46`).

    There are 24 admissible alignments of a non-overlapping daily partition and
    each gives a different Sharpe, MDD and turnover, so the phase is fixed in
    advance rather than chosen after seeing the equity curve.

    ``timestamp`` in the prediction file is the **window start**; the first
    target hour is ``start + L``. With ``L = 96`` a multiple of 24 the phase is
    preserved, so selecting starts at hour 0 selects targets opening at hour 0.
    """
    return (np.asarray(timestamps) // HOUR_MS) % 24 == 0


# -- directional accuracy and its testing regime (`D21`) ---------------------


def pesaran_timmermann(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    """Pesaran-Timmermann (1992) test of directional predictability.

    Returns:
        ``(statistic, one-sided p)`` against ``N(0,1)``. Without a null
        hypothesis, directional accuracy is a descriptive number; this supplies
        the null.

    Zero targets are excluded rather than assigned a direction: a zero
    log-return has no sign to predict, and assigning one would inflate the hit
    rate by whatever the model happened to output there.
    """
    a = np.sign(np.asarray(actual, dtype=np.float64))
    f = np.sign(np.asarray(predicted, dtype=np.float64))
    keep = a != 0
    a, f = a[keep], f[keep]
    n = len(a)
    if n < 2:
        return float("nan"), float("nan")

    hit = float(np.mean(a == f))
    py = float(np.mean(a > 0))
    px = float(np.mean(f > 0))
    p_star = py * px + (1 - py) * (1 - px)

    var_hit = p_star * (1 - p_star) / n
    var_star = (
        (2 * py - 1) ** 2 * px * (1 - px)
        + (2 * px - 1) ** 2 * py * (1 - py)
        + 4 * py * px * (1 - py) * (1 - px) / n
    ) / n
    denom = var_hit - var_star
    if denom <= 0:
        return float("nan"), float("nan")

    stat = (hit - p_star) / math.sqrt(denom)
    return stat, 0.5 * math.erfc(stat / math.sqrt(2.0))


def _hit_rate(actual: np.ndarray, predicted: np.ndarray) -> float:
    keep = np.sign(actual) != 0
    if not keep.any():
        return float("nan")
    return float(np.mean(np.sign(actual[keep]) == np.sign(predicted[keep])))


@dataclass(frozen=True, slots=True)
class DirectionalAccuracy:
    """DA at the three horizons §9.1 requires, with their testing regimes.

    ``da_h24`` and ``da_cum`` carry p-values **only** on the non-overlapping
    sample. On hourly spacing their targets overlap by 23 of 24 hours, giving
    lag-1 autocorrelation of about 23/24; Pesaran-Timmermann's variance is then
    far too small and the test over-rejects badly. The overlapping figures are
    reported descriptively, without p-values, and the resulting power loss
    (T = 30 per block) is stated rather than recovered by using the invalid
    sample.
    """

    da_h1: float
    p_h1: float
    da_hH_overlapping: float
    da_hH: float
    p_hH: float
    da_cum_overlapping: float
    da_cum: float
    p_cum: float
    n_h1: int
    n_non_overlapping: int


def directional_accuracy(frame: pl.DataFrame) -> DirectionalAccuracy:
    """Compute all three DA variants for one run's predictions."""
    last_step = int(frame.get_column("step").max())

    step1 = frame.filter(pl.col("step") == 1)
    a1 = step1.get_column("y_true").to_numpy()
    f1 = step1.get_column("y_pred").to_numpy()
    _, p1 = pesaran_timmermann(a1, f1)

    step_h = frame.filter(pl.col("step") == last_step)
    ts_h = step_h.get_column("timestamp").to_numpy()
    a_h = step_h.get_column("y_true").to_numpy()
    f_h = step_h.get_column("y_pred").to_numpy()
    keep_h = non_overlapping_mask(ts_h)
    _, p_h = pesaran_timmermann(a_h[keep_h], f_h[keep_h])

    cum = (
        frame.group_by("timestamp")
        .agg(pl.col("y_true").sum(), pl.col("y_pred").sum())
        .sort("timestamp")
    )
    ts_c = cum.get_column("timestamp").to_numpy()
    a_c = cum.get_column("y_true").to_numpy()
    f_c = cum.get_column("y_pred").to_numpy()
    keep_c = non_overlapping_mask(ts_c)
    _, p_c = pesaran_timmermann(a_c[keep_c], f_c[keep_c])

    return DirectionalAccuracy(
        da_h1=_hit_rate(a1, f1),
        p_h1=p1,
        da_hH_overlapping=_hit_rate(a_h, f_h),
        da_hH=_hit_rate(a_h[keep_h], f_h[keep_h]),
        p_hH=p_h,
        da_cum_overlapping=_hit_rate(a_c, f_c),
        da_cum=_hit_rate(a_c[keep_c], f_c[keep_c]),
        p_cum=p_c,
        n_h1=len(a1),
        n_non_overlapping=int(keep_c.sum()),
    )


# -- per-block tables --------------------------------------------------------


def assert_same_windows(left: pl.DataFrame, right: pl.DataFrame, what: str) -> None:
    """`D45` — two models may only be compared on identical evaluated windows.

    Naive-RW needs no 96-bar lookback, so unless it is restricted to the window
    set its comparator actually evaluated, RelMSE is a ratio across two different
    samples. Test-window survival is conditioned on *future* gaps and outages
    cluster on stress, so the two samples would differ precisely in their
    high-volatility content.

    Raises:
        ValueError: If the evaluated ``(block, timestamp)`` sets differ.
    """
    def _key(frame: pl.DataFrame) -> np.ndarray:
        pairs = frame.select(["block", "timestamp"]).unique().sort(["block", "timestamp"])
        return (
            pairs.get_column("block").to_numpy().astype(np.int64) * (1 << 44)
            + pairs.get_column("timestamp").to_numpy().astype(np.int64) // HOUR_MS
        )

    a, b = _key(left), _key(right)
    if len(a) != len(b) or not np.array_equal(a, b):
        raise ValueError(
            f"{what}: evaluated window sets differ ({len(a)} vs {len(b)} "
            f"windows). RelMSE across two samples is not a ratio."
        )


def block_metrics(frame: pl.DataFrame, naive_z: float) -> pl.DataFrame:
    """Per-block MSE, MAE, RelMSE and ``R2_oos`` for one run.

    Args:
        frame: One run's predictions.
        naive_z: ``-mu_g/sigma_g`` from ``meta.json`` (`D31`). Passing 0 here
            silently substitutes a constant-drift model for the EMH baseline.

    The Naive-RW error is computed on exactly the rows the model was scored on,
    so :func:`assert_same_windows` has nothing to check for this pair — the
    sample is shared by construction. It still applies across *models*.
    """
    return (
        frame.with_columns(
            (pl.col("y_true") - pl.col("y_pred")).pow(2).alias("_se"),
            (pl.col("y_true") - pl.col("y_pred")).abs().alias("_ae"),
            (pl.col("y_true") - naive_z).pow(2).alias("_se_naive"),
        )
        .group_by("block")
        .agg(
            pl.col("timestamp").n_unique().alias("n_windows"),
            pl.col("_se").count().alias("n_points"),
            pl.col("_se").mean().alias("mse"),
            pl.col("_ae").mean().alias("mae"),
            pl.col("_se_naive").mean().alias("mse_naive"),
        )
        .with_columns(
            (pl.col("mse") / pl.col("mse_naive")).alias("rel_mse"),
            (1.0 - pl.col("mse") / pl.col("mse_naive")).alias("r2_oos"),
        )
        .sort("block")
    )


def gather_grid(run_ids: list[str], roots: list[Path]) -> pl.DataFrame:
    """Per (run, block) metrics for many runs — the input to every RQ estimator.

    Returns:
        Long frame with ``run_id, model, origin_index, origin, k, pred_len,
        seed, block, n_windows, mse, mae, mse_naive, rel_mse, r2_oos, sigma_g``.

    Raises:
        FileNotFoundError: If any run is absent. A quietly short grid produces a
            table whose cells came from different arms, which is worse than no
            table.
    """
    rows: list[pl.DataFrame] = []
    missing: list[str] = []
    for run_id in run_ids:
        try:
            preds = load_predictions(run_id, roots)
            meta = load_meta(run_id, roots)
        except FileNotFoundError:
            missing.append(run_id)
            continue
        parts = parse_run_id(run_id)
        rows.append(
            block_metrics(preds, float(meta["naive_rw_z"])).with_columns(
                pl.lit(run_id).alias("run_id"),
                pl.lit(str(parts["model"])).alias("model"),
                pl.lit(int(parts["origin_index"])).cast(pl.Int32).alias("origin_index"),
                pl.lit(str(meta["origin"])).alias("origin"),
                pl.lit(int(parts["k"])).cast(pl.Int32).alias("k"),
                pl.lit(int(parts["pred_len"])).cast(pl.Int32).alias("pred_len"),
                pl.lit(int(parts["seed"])).cast(pl.Int32).alias("seed"),
                pl.lit(float(meta["sigma_g"])).alias("sigma_g"),
            )
        )
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} of {len(run_ids)} runs are absent, first few: "
            f"{missing[:5]}. Complete the grid or pass the subset explicitly — "
            f"a silently short grid mixes arms inside one table."
        )
    if not rows:
        raise ValueError("no runs gathered")
    return pl.concat(rows)


def seed_average(grid: pl.DataFrame) -> pl.DataFrame:
    """Average MSE across seeds **before** any ratio is formed (`D42`).

    Seeds are computational noise, not population draws. Averaging ratios
    instead would differ by Jensen and would additionally require pairing seed
    42 at K=1 with seed 42 at K=8 — independent training runs of different
    models, where any of 5! orderings gives a different answer.

    The cell mean still carries Monte-Carlo error, which enters as measurement
    error in the dependent variable: unbiased for beta1, but inflating residual
    variance. ``n_seeds`` and ``mse_seed_std`` are carried so §9.2's dispersion
    rule can bind the error bar to the aggregation level (`D30`) — seed std is a
    Monte-Carlo diagnostic, never the uncertainty on an origin-aggregated row.
    """
    return (
        grid.group_by(["model", "origin_index", "origin", "k", "pred_len", "block"])
        .agg(
            pl.col("mse").mean().alias("mse"),
            pl.col("mae").mean().alias("mae"),
            pl.col("mse_naive").mean().alias("mse_naive"),
            pl.col("mse").std().alias("mse_seed_std"),
            pl.col("n_windows").first().alias("n_windows"),
            pl.col("sigma_g").first().alias("sigma_g"),
            pl.col("mse").count().alias("n_seeds"),
        )
        .with_columns(
            (pl.col("mse") / pl.col("mse_naive")).alias("rel_mse"),
            (1.0 - pl.col("mse") / pl.col("mse_naive")).alias("r2_oos"),
        )
        .sort(["model", "origin_index", "k", "pred_len", "block"])
    )


# -- RQ2: the amplification gap and its decay --------------------------------


def amplification(
    seed_avg: pl.DataFrame,
    k_small: int = 1,
    k_large: int = 8,
    model: str = "itr",
    pred_len: int = 24,
) -> pl.DataFrame:
    """``A(i,b) = [MSE_K1 - MSE_K8] / MSE_K1`` — RQ2's dependent variable.

    **K=8, never K=12.** K=12 carries deliberate redundancy (root §5.2), so
    using it would confound decay with that redundancy — which is why the pair
    is a parameter with a pre-registered default rather than a free choice.

    Both models are evaluated on the same block, so period difficulty cancels in
    the ratio, and it cancels *well*: ``MSE_model`` and ``MSE_naive`` on one
    block correlate near 1. That is the argument the whole ratio-metric design
    rests on.
    """
    base = seed_avg.filter(
        (pl.col("model") == model) & (pl.col("pred_len") == pred_len)
    )
    small = (
        base.filter(pl.col("k") == k_small)
        .select(["origin_index", "origin", "block", "mse", "n_windows"])
        .rename({"mse": "mse_small", "n_windows": "n_small"})
    )
    large = (
        base.filter(pl.col("k") == k_large)
        .select(["origin_index", "block", "mse", "n_windows"])
        .rename({"mse": "mse_large", "n_windows": "n_large"})
    )

    joined = small.join(large, on=["origin_index", "block"], how="inner")
    if joined.height != small.height:
        raise ValueError(
            f"K={k_small} has {small.height} cells but only {joined.height} "
            f"matched K={k_large}; the panel must be balanced before beta1"
        )
    mismatched = joined.filter(pl.col("n_small") != pl.col("n_large"))
    if mismatched.height:
        raise ValueError(
            f"`D45`: {mismatched.height} cells evaluate K={k_small} and "
            f"K={k_large} on different window counts; A would be a ratio across "
            f"two samples"
        )
    return joined.with_columns(
        ((pl.col("mse_small") - pl.col("mse_large")) / pl.col("mse_small")).alias("A")
    ).sort(["origin_index", "block"])


def attention_amplification(
    seed_avg: pl.DataFrame, k: int = 8, pred_len: int = 24
) -> pl.DataFrame:
    """``A_attn(i,b) = [MSE_uniformK8 - MSE_K8] / MSE_uniformK8`` (`D50`).

    K=1 versus K=8 does **not** isolate attention: the two arms differ in
    *information* and in *whether attention is active* simultaneously, so a
    decaying ``A(b)`` is equally consistent with "cross-variate attention
    overfits regime-specific structure" — a capacity story — as with the
    information story RQ2 claims. This contrast holds information fixed and
    varies only what attention selects, at runs Figure 5 needs anyway.
    """
    sel = seed_avg.filter((pl.col("k") == k) & (pl.col("pred_len") == pred_len))
    uniform = (
        sel.filter(pl.col("model") == "itru")
        .select(["origin_index", "origin", "block", "mse"])
        .rename({"mse": "mse_uniform"})
    )
    attended = (
        sel.filter(pl.col("model") == "itr")
        .select(["origin_index", "block", "mse"])
        .rename({"mse": "mse_attended"})
    )
    return (
        uniform.join(attended, on=["origin_index", "block"], how="inner")
        .with_columns(
            (
                (pl.col("mse_uniform") - pl.col("mse_attended"))
                / pl.col("mse_uniform")
            ).alias("A_attn")
        )
        .sort(["origin_index", "block"])
    )


# -- RQ3: skill decay and the retraining cadence -----------------------------


@dataclass(frozen=True, slots=True)
class DecayResult:
    """Per-origin ``D(i,b)`` and the censored ``b*`` it implies."""

    table: pl.DataFrame
    excluded_origins: tuple[str, ...]

    def b_star(self, tau: float) -> pl.DataFrame:
        """``b*(i) = min{b : D(i,b) > tau}``, **right-censored at 6** (`D41`).

        ``min{.}`` does not commute with averaging, so pooling MSEs across
        origins and *then* taking the minimum is a different estimand and is
        forbidden. Each origin contributes one observation, censored or not.

        The schema is declared rather than inferred (`D55`). When `decay`'s
        non-positive-skill guard excludes *every* origin, ``self.table`` is empty
        and an inferred schema yields a frame with no columns, so a caller's
        ``bs["b_star"]`` raises ``ColumnNotFoundError`` — which is what took the
        Kaggle notebook down at the exact moment its grid output was the only
        thing worth keeping. That guard firing is the **expected** outcome under
        non-positive skill, not an edge case: root §10.3's first measured run
        returned ``R2_oos = -0.0183`` and the completed grid returned it at all
        fifteen origins. `D54e` gates the estimators on grid *completeness*,
        which is a different failure and does not cover this path.

        An empty return means the estimand is **undefined** — there is no edge to
        lose a proportion of — which is not the same as every origin being
        censored at 6, where an edge exists and simply never decays past tau.
        Callers must report the two differently; ``excluded_origins`` is what
        tells them apart.
        """
        rows = []
        for key, part in self.table.group_by("origin", maintain_order=True):
            label = key[0] if isinstance(key, tuple) else key
            crossed = part.filter(pl.col("D") > tau).sort("block")
            rows.append(
                {
                    "origin": str(label),
                    "tau": tau,
                    "b_star": int(crossed.get_column("block")[0]) if crossed.height else 6,
                    "event": bool(crossed.height),
                }
            )
        return pl.DataFrame(rows, schema=B_STAR_SCHEMA)


def decay(
    seed_avg: pl.DataFrame,
    k: int = 8,
    model: str = "itr",
    pred_len: int = 24,
) -> DecayResult:
    """``D(i,b)`` on the **skill** scale (`D23`), normalised within origin (`D05`).

    ``D(i,b) = [mean_b' R2_oos(i,b') - R2_oos(i,b)] / mean_b' R2_oos(i,b')``

    On the RelMSE scale the pre-registered thresholds are unreachable: with
    ``RelMSE(1) = 0.996``, even *total* destruction of the model's edge gives
    ``D = 1/0.996 - 1 = 0.402%``, so tau = 2.5% would require the model to become
    2% worse than forecasting zero. RQ3 would return "no decay detected"
    regardless of the data — a result fixed by a units mismatch rather than by
    the market. On the skill scale ``D`` runs from 0 (no decay) through 1 (edge
    fully gone) and the taus are commensurate with it.

    The denominator is the within-origin **mean** rather than block 1's value
    (`D05` follow-on): block 1 is one 30-day block under heavy tails and
    volatility clustering, and it would otherwise sit in the denominator of five
    quantities and make their errors perfectly correlated.

    **Origins with non-positive mean skill are excluded and named**, never
    silently dropped: ``D`` is a proportion of an edge, and an origin with no
    edge has no proportion of one. Root §10.3's first measured run returned
    ``R2_oos = -0.0183``, so this guard may well be the common case rather than
    the edge case §9.1 assumed.
    """
    sel = seed_avg.filter(
        (pl.col("model") == model)
        & (pl.col("k") == k)
        & (pl.col("pred_len") == pred_len)
    ).sort(["origin_index", "block"])

    reference = sel.group_by("origin").agg(pl.col("r2_oos").mean().alias("r2_ref"))
    joined = sel.join(reference, on="origin", how="left")

    excluded = tuple(
        sorted(joined.filter(pl.col("r2_ref") <= 0).get_column("origin").unique().to_list())
    )
    kept = joined.filter(pl.col("r2_ref") > 0).with_columns(
        ((pl.col("r2_ref") - pl.col("r2_oos")) / pl.col("r2_ref")).alias("D")
    )
    return DecayResult(
        table=kept.select(
            ["origin", "origin_index", "block", "n_windows", "r2_oos", "r2_ref", "D"]
        ).sort(["origin_index", "block"]),
        excluded_origins=excluded,
    )


# -- survival analysis for b* (`D41`) ----------------------------------------


@dataclass(frozen=True, slots=True)
class SurvivalCurve:
    """Kaplan-Meier estimate on the 30-day block grid."""

    times: np.ndarray
    survival: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    n_events: int
    n_censored: int

    @property
    def median(self) -> float:
        """Smallest block with ``S(t) <= 0.5``; ``inf`` when never reached.

        ``inf`` is the honest answer, and root §3 fixes its wording: *"no decay
        detected within 180 days"* — a right-censored result, not a missing one.
        """
        below = self.times[self.survival <= 0.5]
        return float(below[0]) if len(below) else float("inf")

    @property
    def median_interval(self) -> tuple[float, float]:
        """Confidence set for the median: blocks whose CI band straddles 0.5.

        Table 5 carries this interval, never a bare integer, and the abstract's
        recommended cadence is this interval.
        """
        straddle = self.times[(self.lower <= 0.5) & (self.upper >= 0.5)]
        if not len(straddle):
            return (self.median, self.median)
        return (float(straddle[0]), float(straddle[-1]))


def _normal_quantile(p: float) -> float:
    """Acklam's inverse normal CDF — avoids a scipy import for one number."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q, r = p - 0.5, (p - 0.5) ** 2
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def normal_quantile(p: float) -> float:
    """Standard normal inverse CDF -- the public name of :func:`_normal_quantile`.

    Exists because :mod:`itransformer_btc.economics` needs it for the Deflated
    Sharpe Ratio threshold, and root §15's flattening rule says a module reaches a
    sibling **by name**. Reaching for a private one across modules would work in
    the package and read as an accident in the notebook.
    """
    return _normal_quantile(p)


def normal_cdf(x: float) -> float:
    """Standard normal CDF, via ``erfc`` so no optional dependency is implied."""
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def _loglog_band(surv: float, var_sum: float, z: float) -> tuple[float, float]:
    """Log-log transformed Greenwood band — stays inside ``[0, 1]`` at small G.

    The plain Greenwood band routinely leaves the unit interval at G = 15, which
    would make the median interval unreadable at exactly the sample size this
    study has.
    """
    if surv <= 0.0 or surv >= 1.0 or var_sum <= 0.0:
        return surv, surv
    se = math.sqrt(var_sum) / abs(math.log(surv))
    lo = surv ** math.exp(z * se)
    hi = surv ** math.exp(-z * se)
    return float(min(lo, hi)), float(max(lo, hi))


def kaplan_meier(
    times: np.ndarray, events: np.ndarray, alpha: float = 0.05
) -> SurvivalCurve:
    """Kaplan-Meier with Greenwood log-log confidence bands.

    ``b*`` is right-censored survival data on a six-point grid: an origin that
    never crosses tau is censored at 6, not missing. Reporting a bare mean over
    the crossers would condition on the event and bias the recommended cadence
    downward — which is the number the abstract carries.
    """
    t = np.asarray(times, dtype=np.float64)
    e = np.asarray(events, dtype=bool)
    z = _normal_quantile(1 - alpha / 2)

    surv, var_sum = 1.0, 0.0
    out_t, out_s, out_lo, out_hi = [], [], [], []
    for time in np.unique(t):
        at_risk = int(np.sum(t >= time))
        died = int(np.sum((t == time) & e))
        if at_risk > 0 and died > 0:
            surv *= 1.0 - died / at_risk
            if at_risk > died:
                var_sum += died / (at_risk * (at_risk - died))
        lo, hi = _loglog_band(surv, var_sum, z)
        out_t.append(time)
        out_s.append(surv)
        out_lo.append(lo)
        out_hi.append(hi)

    return SurvivalCurve(
        times=np.array(out_t),
        survival=np.array(out_s),
        lower=np.array(out_lo),
        upper=np.array(out_hi),
        n_events=int(e.sum()),
        n_censored=int(len(e) - e.sum()),
    )


def logrank(
    times_a: np.ndarray, events_a: np.ndarray,
    times_b: np.ndarray, events_b: np.ndarray,
) -> tuple[float, float]:
    """Two-sample log-rank test — H3's "larger K decays faster" (`D41`).

    Returns:
        ``(chi-square statistic on 1 df, two-sided p)``.
    """
    t = np.concatenate([times_a, times_b]).astype(np.float64)
    e = np.concatenate([events_a, events_b]).astype(bool)
    g = np.concatenate([np.zeros(len(times_a)), np.ones(len(times_b))]).astype(bool)

    observed = expected = variance = 0.0
    for time in np.unique(t[e]):
        n_risk = int(np.sum(t >= time))
        n_risk_b = int(np.sum((t >= time) & g))
        d = int(np.sum((t == time) & e))
        d_b = int(np.sum((t == time) & e & g))
        if n_risk < 2 or d == 0:
            continue
        share = n_risk_b / n_risk
        observed += d_b
        expected += d * share
        variance += d * share * (1 - share) * (n_risk - d) / (n_risk - 1)
    if variance <= 0:
        return float("nan"), float("nan")
    chi2 = (observed - expected) ** 2 / variance
    return float(chi2), float(math.erfc(math.sqrt(chi2 / 2.0)))


# -- Diebold-Mariano and Clark-West (`D29`, `D34`) ---------------------------


def _rectangular_lrv(d: np.ndarray, h: int) -> float:
    """``[gamma_0 + 2 sum_{k=1}^{h-1} gamma_k] / T`` — the variance of ``d_bar``.

    Rectangular, not Bartlett (`D34`). Under the DM null, h-step optimal
    forecast errors are MA(h-1), so all autocovariances to lag ``h-1`` are
    genuinely nonzero and equally real. Bartlett weights shrink the lag-22 term
    by about 92%, understating the long-run variance and producing exactly the
    over-optimistic p-values this estimator exists to prevent. ``statsmodels``'
    ``cov_hac`` is Bartlett by default, so a literal reading of "Newey-West"
    fails validation against R's ``forecast::dm.test``.
    """
    t = len(d)
    dm = d - d.mean()
    total = float(dm @ dm) / t
    for k in range(1, min(h, t)):
        total += 2.0 * float(dm[k:] @ dm[:-k]) / t
    return total / t


def _bartlett_lrv(d: np.ndarray, h: int) -> float:
    """Only ever the fallback, and its use is reported (root §9.2)."""
    t = len(d)
    dm = d - d.mean()
    total = float(dm @ dm) / t
    for k in range(1, min(h, t)):
        total += 2.0 * (1.0 - k / h) * float(dm[k:] @ dm[:-k]) / t
    return total / t


@dataclass(frozen=True, slots=True)
class TestResult:
    """One forecast-comparison test, carrying everything needed to redo it."""

    name: str
    statistic: float
    p_value: float
    T: int
    h: int
    one_sided: bool
    fallback_fired: bool

    def __str__(self) -> str:
        tail = "  [Bartlett fallback fired]" if self.fallback_fired else ""
        side = "one-sided" if self.one_sided else "two-sided"
        return (
            f"{self.name}: S*={self.statistic:+.4f}  p={self.p_value:.4g} "
            f"({side})  T={self.T}  h={self.h}{tail}"
        )


def _upper_tail(stat: float, df: int) -> float:
    """``P(T_df > stat)`` — Student-t, falling back to the normal without scipy."""
    try:
        from scipy import stats as _stats  # root §16's named stats boundary

        return float(_stats.t.sf(stat, df=df))
    except ImportError:  # pragma: no cover - scipy ships with the Kaggle image
        return 0.5 * math.erfc(stat / math.sqrt(2.0))


def _hln_and_p(d: np.ndarray, h: int, name: str, one_sided: bool) -> TestResult:
    """Harvey-Leybourne-Newbold correction, referred to ``t(T-1)``.

    ``S* = S sqrt[(T + 1 - 2h + h(h-1)/T) / T]``, compared against Student-t
    with ``T-1`` degrees of freedom — **not** the standard normal. The factor is
    asserted positive before use: at ``h = 24`` it is exactly 0 at ``T = 24`` and
    0.047 at ``T = 30``, precisely the T a non-overlapping 30-day block would
    produce, so a silent negative would yield a complex statistic reported as a
    real one.
    """
    d = np.asarray(d, dtype=np.float64)
    t = len(d)
    if t < 2:
        raise ValueError(f"{name}: T={t} is too small for a loss differential")
    factor = (t + 1 - 2 * h + h * (h - 1) / t) / t
    if factor <= 0:
        raise ValueError(
            f"{name}: the HLN factor is {factor:.4f} <= 0 at T={t}, h={h}. "
            f"Root §9.2 refuses to report where it fails; state T instead."
        )

    variance = _rectangular_lrv(d, h)
    fallback = False
    if variance <= 0:
        # The rectangular estimator is not guaranteed positive in finite
        # samples. Root §9.2: fall back to Bartlett and *report that it fired*.
        variance = _bartlett_lrv(d, h)
        fallback = True
        if variance <= 0:
            raise ValueError(f"{name}: no positive long-run variance at T={t}")

    stat = float(d.mean() / math.sqrt(variance) * math.sqrt(factor))
    upper = _upper_tail(abs(stat), t - 1)
    if one_sided:
        p = upper if stat >= 0 else 1.0 - upper
    else:
        p = 2.0 * upper
    return TestResult(name, stat, float(min(p, 1.0)), t, h, one_sided, fallback)


def hln_test(
    d: np.ndarray, h: int, name: str = "HLN", one_sided: bool = False
) -> TestResult:
    """Harvey-Leybourne-Newbold on a **pre-assembled** loss differential.

    :func:`dm_test` and :func:`clark_west_test` build their own differential from
    forecasts. Table 6 assembles one itself --- the Clark-West adjusted series
    aggregated over the 24 forecast steps of each origin, so that ``T`` counts
    window starts and ``h`` stays 24, which is the sample root §9.2 pins. Handing
    that series back through the forecasts would require inventing a pair of
    pseudo-forecasts whose differential happens to equal it.

    Args:
        d: The loss differential, one value per forecast origin.
        h: Forecast horizon, setting the truncation lag at ``h - 1``.
        name: Label carried into the result.
        one_sided: True for a nested pair, where the alternative is directional.
    """
    return _hln_and_p(np.asarray(d, dtype=np.float64), h, name, one_sided)


def dm_test(
    loss_a: np.ndarray, loss_b: np.ndarray, h: int, name: str = "DM"
) -> TestResult:
    """Diebold-Mariano for a **non-nested** pair — iTransformer vs DLinear etc.

    Do not use this on K=1 vs K=8, on anything vs Naive-RW, or on Ridge-K1 vs
    Ridge-K8: those pairs are nested, and there the statistic is not
    asymptotically ``N(0,1)`` (Clark & McCracken 2001; McCracken 2007). Use
    :func:`clark_west_test`.
    """
    return _hln_and_p(
        np.asarray(loss_a) - np.asarray(loss_b), h, name, one_sided=False
    )


def clark_west_test(
    y: np.ndarray,
    pred_small: np.ndarray,
    pred_large: np.ndarray,
    h: int,
    name: str = "Clark-West",
) -> TestResult:
    """Clark-West (2007) for a **nested** pair — the comparisons that carry the paper.

    ``f_t = (y - y_small)^2 - (y - y_large)^2 + (y_small - y_large)^2``

    The third term is the adjustment. Under the null of equal population
    predictive ability the larger model's extra estimation noise makes it look
    worse, so the unadjusted differential has a mean shifted away from zero and
    standard DM is systematically undersized **against the alternative this
    study exists to establish**. The Stage 5 gate turns a title decision on this
    statistic, which is why the choice is not left to the caller.

    One-sided by construction: the alternative is that the larger model helps.
    """
    y = np.asarray(y, dtype=np.float64)
    s = np.asarray(pred_small, dtype=np.float64)
    lg = np.asarray(pred_large, dtype=np.float64)
    f = np.square(y - s) - np.square(y - lg) + np.square(s - lg)
    return _hln_and_p(f, h, name, one_sided=True)


def per_origin_loss(frame: pl.DataFrame) -> pl.DataFrame:
    """Mean squared error per forecast origin — the ``d_t`` series DM consumes.

    Root §9.2 pins the DM sample: **per (origin, block)**, on the overlapping
    hourly loss differential, ``T ~ 720``, ``h = 24``, truncation lag 23. Block
    level statistics are combined across cells by a stated method and **never**
    by concatenating ``d_t`` across origins: the model changes at each origin, so
    the DM null has no interpretation across that boundary.
    """
    return (
        frame.with_columns((pl.col("y_true") - pl.col("y_pred")).pow(2).alias("_se"))
        .group_by(["block", "timestamp"])
        .agg(pl.col("_se").mean().alias("loss"))
        .sort(["block", "timestamp"])
    )


# -- RQ2's core regression: beta1 with origin FE and a wild cluster bootstrap -


@dataclass(frozen=True, slots=True)
class Beta1Result:
    """``A(i,b) = alpha_i + beta1 b + eps`` with clustered inference (`D06`, `D42`)."""

    beta1: float
    t_statistic: float
    cluster_se: float
    p_rademacher: float
    p_webb: float
    n_clusters: int
    n_observations: int
    within_slopes: np.ndarray
    B: int

    @property
    def headline_p(self) -> float:
        """The more conservative of the two weight schemes, as root §9.2 requires."""
        return max(self.p_rademacher, self.p_webb)

    def __str__(self) -> str:
        return (
            f"beta1 = {self.beta1:+.6f}   t = {self.t_statistic:+.3f}   "
            f"G = {self.n_clusters}   N = {self.n_observations}\n"
            f"WCR one-sided p (H1: beta1 < 0): Rademacher {self.p_rademacher:.4f}, "
            f"Webb {self.p_webb:.4f}  ->  headline {self.headline_p:.4f}\n"
            f"Effective independence is bounded near 4 by the training-window "
            f"overlap (root §8.1), well below G = {self.n_clusters}."
        )


def _weights(kind: str, shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    if kind == "rademacher":
        return rng.choice(np.array([-1.0, 1.0]), size=shape)
    if kind == "webb":
        # Webb's 6-point distribution. At G = 15 Rademacher already admits
        # 2^15 = 32,768 distinct draws, a minimum two-sided p of about 6e-5, so
        # the original small-G justification for preferring Webb no longer
        # binds — both are reported and the more conservative is the headline.
        atoms = np.array([
            -math.sqrt(1.5), -1.0, -math.sqrt(0.5),
            math.sqrt(0.5), 1.0, math.sqrt(1.5),
        ])
        return rng.choice(atoms, size=shape)
    raise ValueError(f"unknown weight scheme {kind!r}")


def _balanced_matrix(panel: pl.DataFrame, value: str) -> tuple[np.ndarray, np.ndarray]:
    """``(G x B)`` outcome matrix and the block axis, or a loud failure.

    Built by hand rather than with ``pivot`` so the code does not depend on which
    polars major version the Kaggle image happens to ship.
    """
    origins = sorted(set(panel.get_column("origin").to_list()))
    blocks = sorted(set(int(b) for b in panel.get_column("block").to_list()))
    index = {(o, b): i for i, (o, b) in enumerate([(o, b) for o in origins for b in blocks])}

    out = np.full(len(index), np.nan)
    for origin, block, val in zip(
        panel.get_column("origin").to_list(),
        panel.get_column("block").to_list(),
        panel.get_column(value).to_list(),
    ):
        out[index[(str(origin), int(block))]] = float(val)

    matrix = out.reshape(len(origins), len(blocks))
    if np.isnan(matrix).any():
        raise ValueError(
            "unbalanced panel: beta1's reduction to the mean of within-slopes "
            "holds only when every origin carries every block"
        )
    return matrix, np.array(blocks, dtype=np.float64)


def panel_beta1(
    panel: pl.DataFrame,
    value: str = "A",
    B: int = 99_999,
    seed: int = 42,
) -> Beta1Result:
    """Fit ``A(i,b) = alpha_i + beta1 b + eps`` and test ``H1: beta1 < 0``.

    Args:
        panel: Long frame with ``origin``, ``block`` and ``value``. Must be
            balanced — every origin carries the same block set.
        value: Dependent variable column.
        B: Bootstrap replications. 99,999 as pre-registered.
        seed: Bootstrap seed, recorded so the p-value is regenerable (root §12).

    **Without ``alpha_i``, beta1 absorbs origin-level difficulty** (`D06`). With
    origin fixed effects and a balanced panel, ``beta1`` reduces algebraically to
    the simple mean of the origin-specific within-slopes, so inference on the
    paper's core claim is a one-sample test on **G** numbers. Citing "15 x 6 = 90
    observations" invites the reader to infer power that does not exist; both
    counts are reported, and effective independence is bounded near 4 by the
    training-window overlap (root §8.1).

    The bootstrap is **restricted** (WCR — the null imposed when generating
    samples), bootstraps the **cluster-robust t-statistic** rather than beta-hat,
    and is **one-sided at alpha = 0.05 declared in advance**. WCU is severely
    size-distorted at small G (MacKinnon, Nielsen & Webb 2023), and the
    asymptotic refinement comes from bootstrapping *t* (Cameron, Gelbach &
    Miller 2008). A side chosen after seeing the sign is not pre-registered.
    """
    a, x = _balanced_matrix(panel, value)
    g, n_blocks = a.shape
    xd = x - x.mean()
    sxx = float(xd @ xd)

    within = a - a.mean(axis=1, keepdims=True)
    beta = float((within * xd).sum() / (g * sxx))
    resid = within - beta * xd
    score = resid @ xd
    variance = float((score @ score) / (g * sxx) ** 2)
    se = math.sqrt(variance) if variance > 0 else float("nan")
    t_obs = beta / se if se == se and se > 0 else float("nan")

    # Restricted residuals: with beta1 = 0 imposed the fitted value is the origin
    # mean, so u_tilde is exactly the within-origin demeaned outcome. Because
    # each row of u_tilde already sums to zero, the bootstrap origin means are
    # unchanged and the whole replication collapses to s = u_tilde @ xd.
    s = within @ xd

    def _p(kind: str) -> float:
        rng = np.random.default_rng(seed)
        weights = _weights(kind, (B, g), rng)
        beta_star = (weights @ s) / (g * sxx)
        score_star = weights * s[None, :] - beta_star[:, None] * sxx
        var_star = np.square(score_star).sum(axis=1) / (g * sxx) ** 2
        ok = var_star > 0
        t_star = beta_star[ok] / np.sqrt(var_star[ok])
        # (1 + count) / (1 + B), not count / B (Davison & Hinkley 1997): the
        # observed statistic is one of its own reference distribution, and the
        # naive form returns a literal p = 0, which is not a probability any
        # finite bootstrap can support. At B = 99,999 the floor it reports is
        # 1e-5, and at G = 15 Rademacher's own granularity bounds it at ~3e-5
        # anyway — so the floor is honest rather than conservative padding.
        below = int(np.sum(t_star <= t_obs))  # H1: beta1 < 0, left tail
        return (1.0 + below) / (1.0 + int(ok.sum()))

    return Beta1Result(
        beta1=beta,
        t_statistic=t_obs,
        cluster_se=se,
        p_rademacher=_p("rademacher"),
        p_webb=_p("webb"),
        n_clusters=g,
        n_observations=g * n_blocks,
        within_slopes=(within * xd).sum(axis=1) / sxx,
        B=B,
    )


@dataclass(frozen=True, slots=True)
class EquivalenceResult:
    """TOST verdict on a rung `D49` pre-registers as flat."""

    mean_delta: float
    margin: float
    p_lower: float
    p_upper: float
    n: int

    @property
    def equivalent(self) -> bool:
        return max(self.p_lower, self.p_upper) < 0.05

    def __str__(self) -> str:
        verdict = "EQUIVALENT (flat)" if self.equivalent else "NOT shown equivalent"
        return (
            f"TOST: mean delta = {self.mean_delta:+.6f}, margin = +/-{self.margin:.6f}, "
            f"p = ({self.p_lower:.4f}, {self.p_upper:.4f}), G = {self.n}  ->  {verdict}"
        )


def tost_equivalence(
    deltas: np.ndarray, margin: float, alpha: float = 0.05
) -> EquivalenceResult:
    """Two one-sided tests — RQ1's pre-registered equivalence check (`D49`).

    RQ1's claim that the 8->12 rung is flat is an assertion of **no effect**, and
    a non-significant ΔMSE is a failure to reject, not evidence of equivalence.
    The margin is fixed in advance at ``0.25 x ΔMSE(4->8)``: choosing it after
    seeing the rung is the same p-hacking root §3 forbids for tau.

    Args:
        deltas: One within-origin ΔMSE per cluster. The inferential unit is the
            origin, never the (origin, block) cell.
        margin: ``Δ_eq``, positive.
    """
    d = np.asarray(deltas, dtype=np.float64)
    n = len(d)
    if n < 2:
        raise ValueError("TOST needs at least two clusters")
    se = float(np.std(d, ddof=1) / math.sqrt(n))
    if se <= 0:
        raise ValueError("zero dispersion across clusters; TOST is undefined")
    mean = float(d.mean())
    return EquivalenceResult(
        mean_delta=mean,
        margin=abs(margin),
        p_lower=_upper_tail((mean + abs(margin)) / se, n - 1),   # H0: mu <= -margin
        p_upper=_upper_tail(-(mean - abs(margin)) / se, n - 1),  # H0: mu >= +margin
        n=n,
    )


def j_test(
    y: np.ndarray, x_a: np.ndarray, x_b: np.ndarray, groups: np.ndarray
) -> tuple[float, float]:
    """Davidson-MacKinnon J-test of model A against model B (`D32`).

    RQ1 is a **non-nested** comparison: "benefit tracks K" and "benefit tracks
    K_eff" are two different regressors for the same outcome, and neither nests
    the other. Fitting both and comparing R-squared answers nothing; the J-test
    augments A with B's fitted values and asks whether they still carry
    information.

    All three inputs are within-transformed by ``groups`` first — the (origin x
    block) fixed effects of §9.1's specification — so the comparison is
    identified from within-cell variation across rungs, which is the only
    variation that distinguishes the two theories.

    Returns:
        ``(t statistic on B's fitted values, two-sided p)``. A significant t
        means A alone is inadequate. Run it both ways: if both reject, neither
        explanation is sufficient; if neither does, the data cannot separate
        them, which at ``corr(K, K_eff) ~ 0.97`` is the outcome to expect and to
        report plainly.
    """
    def _demean(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.float64)
        out = v.astype(np.float64).copy()
        for g in np.unique(groups):
            mask = groups == g
            out[mask] = v[mask] - v[mask].mean()
        return out

    yd, ad, bd = _demean(y), _demean(x_a), _demean(x_b)
    fitted_b = bd * (float(bd @ yd) / float(bd @ bd))

    design = np.column_stack([ad, fitted_b])
    coef, *_ = np.linalg.lstsq(design, yd, rcond=None)
    resid = yd - design @ coef
    dof = len(yd) - design.shape[1] - len(np.unique(groups))
    if dof <= 0:
        return float("nan"), float("nan")
    sigma2 = float(resid @ resid) / dof
    cov = sigma2 * np.linalg.pinv(design.T @ design)
    se = math.sqrt(max(cov[1, 1], 0.0))
    if se <= 0:
        return float("nan"), float("nan")
    t = float(coef[1] / se)
    return t, 2.0 * _upper_tail(abs(t), dof)


def minimum_detectable_beta1(
    within_slopes: np.ndarray, alpha: float = 0.05, power: float = 0.80
) -> float:
    """The MDE root §13.2 calls the most damaging omission on its list.

    Every design choice in this study implies someone reasoned about precision,
    and no number was written down. If the MDE exceeds the plausible magnitude
    of ``A``, RQ2 must be repositioned as descriptive **before** the grid runs —
    otherwise a non-significant beta1 is indistinguishable from a design that
    could never have detected decay.

    Computed from the between-origin dispersion of the within-slope, which is
    exactly what the Stage 5 pilot estimates. Returned negative, since the
    alternative is one-sided and downward.
    """
    g = len(within_slopes)
    if g < 2:
        return float("nan")
    se = float(np.std(within_slopes, ddof=1) / math.sqrt(g))
    return -(_normal_quantile(1 - alpha) + _normal_quantile(power)) * se


# -- drivers for the paper's tables (`D62a`) ---------------------------------
#
# Everything below reads what the grid already wrote. None of it trains and none
# of it needs a GPU. Each existed as a function nobody called, or as a number
# that lived only in `CLAUDE.md` prose and therefore fell outside §12's
# regenerability contract.


def directional_accuracy_table(run_ids: list[str], roots: list[Path]) -> pl.DataFrame:
    """DA at all three horizons for many runs (`D21`) -- an input to Table 4.

    :func:`directional_accuracy` has existed and been tested since the model
    plane was built and **was never called**: no DA figure appears in
    ``paper_numbers.json`` or anywhere in the session log. This is the driver.

    The three variants do not share a testing regime and the table keeps that
    visible rather than tidying it away. ``da_h1`` carries a Pesaran-Timmermann
    p-value on hourly spacing. ``da_hH`` and ``da_cum`` carry one **only** on the
    non-overlapping sample; their ``*_overlapping`` twins are descriptive and have
    no p-value at all, because on hourly spacing those targets overlap by 23 of
    24 hours, giving lag-1 autocorrelation near 23/24 -- PT's variance is then far
    too small and the test over-rejects badly. The resulting power loss is
    **stated** as ``n_non_overlapping``, never recovered by using the invalid
    sample.

    Args:
        run_ids: Runs to measure.
        roots: Artifact roots, working directory first.

    Returns:
        One row per run: its identity, all eight DA figures, and both sample sizes.
    """
    rows: list[dict[str, float | str | int]] = []
    for run_id in run_ids:
        parts = parse_run_id(run_id)
        meta = load_meta(run_id, roots)
        da = directional_accuracy(load_predictions(run_id, roots))
        rows.append(
            {
                "run_id": run_id,
                "model": str(parts["model"]),
                "origin": str(meta["origin"]),
                "origin_index": int(parts["origin_index"]),
                "k": int(parts["k"]),
                "pred_len": int(parts["pred_len"]),
                "seed": int(parts["seed"]),
                "da_h1": da.da_h1,
                "p_h1": da.p_h1,
                "da_hH": da.da_hH,
                "p_hH": da.p_hH,
                "da_hH_overlapping": da.da_hH_overlapping,
                "da_cum": da.da_cum,
                "p_cum": da.p_cum,
                "da_cum_overlapping": da.da_cum_overlapping,
                "n_h1": da.n_h1,
                "n_non_overlapping": da.n_non_overlapping,
            }
        )
    return pl.DataFrame(rows)


def raw_scale_table(seed_avg: pl.DataFrame) -> pl.DataFrame:
    """Add RMSE in raw log-return units -- root §9.1's second metric scale.

    "RMSE 0.0043 on hourly log-returns" tells a reader far more than "MSE 0.187
    on normalized data". Both scales are reported and ``sigma_g`` is stated, which
    is what lets the two reconcile. :func:`raw_rmse` has existed all along and,
    like :func:`directional_accuracy`, was never called.
    """
    return seed_avg.with_columns(
        (pl.col("mse").sqrt() * pl.col("sigma_g")).alias("rmse_raw")
    )


def falsification_relmse(seed_avg: pl.DataFrame) -> pl.DataFrame:
    """``aged - fresh`` on **RelMSE**, per (origin, block) -- `D60i`.

    Root §8.1's falsification arm is the only design in the study that identifies
    decay directly, and the number reported for it was a units artefact. The
    notebook printed ``mean(aged - fresh) = -0.053341`` as a raw scaler-space MSE
    difference. The two arms are fitted at origins 90 days apart and therefore
    carry **different sigma_g** -- 0.009151 against 0.007297 at origin 1 -- so
    that difference compares numbers in different units. The matching naive
    baselines differ by -0.053196, i.e. about **99.7% of it is scaler drift**, and
    the sign reads backwards, appearing to say the aged model beat the fresh one.

    Root §9.1 already forbade the comparison by requiring RelMSE "to control for
    period difficulty". The arm was simply never brought under the rule, and the
    corrected figure lived only in prose. **The raw-MSE figure must not appear in
    the manuscript**, and the general rule this defect bought is that any
    cross-origin model comparison is on RelMSE or ``R2_oos``, never on
    scaler-space MSE.

    Returns:
        ``origin_index, origin, block, rel_aged, rel_fresh, gap_rel_mse`` over the
        cells the arm covers -- blocks 4-6 at each origin, which are the same
        calendar hours the aged model was scored on.
    """
    sel = seed_avg.filter(pl.col("pred_len") == 24)
    aged = (
        sel.filter((pl.col("model") == "itr") & (pl.col("k") == 8))
        .select(["origin_index", "origin", "block", "rel_mse"])
        .rename({"rel_mse": "rel_aged"})
    )
    fresh = (
        sel.filter(pl.col("model") == "itrf")
        .select(["origin_index", "block", "rel_mse"])
        .rename({"rel_mse": "rel_fresh"})
    )
    return (
        aged.join(fresh, on=["origin_index", "block"], how="inner")
        .with_columns((pl.col("rel_aged") - pl.col("rel_fresh")).alias("gap_rel_mse"))
        .sort(["origin_index", "block"])
    )


def per_origin_relmse(
    seed_avg: pl.DataFrame, model: str, k: int | None = None
) -> pl.DataFrame:
    """One RelMSE per origin for one arm: window-weighted over its test blocks.

    Seed-averaged MSEs first, ratio second, exactly as `D42` requires --- the two
    orders differ by Jensen, and the per-seed form would additionally have to pair
    seed 42 of one arm with seed 42 of another, which are independent training
    runs of different models.

    Args:
        seed_avg: :func:`seed_average`'s output.
        model: Model tag.
        k: Rung, when the tag carries more than one.

    Returns:
        ``origin, rel_mse, n_windows``, sorted by origin.
    """
    part = seed_avg.filter(
        (pl.col("model") == model) & (pl.col("pred_len") == PRED_LEN)
    )
    if k is not None:
        part = part.filter(pl.col("k") == k)
    return (
        part.group_by("origin")
        .agg(
            (pl.col("mse") * pl.col("n_windows")).sum().alias("_num"),
            (pl.col("mse_naive") * pl.col("n_windows")).sum().alias("_den"),
            pl.col("n_windows").sum().alias("n_windows"),
        )
        .with_columns((pl.col("_num") / pl.col("_den")).alias("rel_mse"))
        .drop(["_num", "_den"])
        .sort("origin")
    )


def paired_contrast(
    seed_avg: pl.DataFrame,
    left: tuple[str, int | None],
    right: tuple[str, int | None],
) -> dict:
    """Paired difference in RelMSE between two arms, across the origins they share.

    **The contrast the marginal columns cannot give you** (`D82`). Table 4 and
    Table 9 print each arm's mean RelMSE with its standard error *across* origins,
    and a reader differencing two such rows is comparing marginal spreads when the
    arms are evaluated on the same fifteen origins with the same naive baselines.
    The paired standard error is roughly half the marginal one here, so overlapping
    error bars in those tables say nothing about whether the arms differ.

    It is what RQ1's matched-K pair needs in particular. ``itro`` and ``itrr`` hold
    K = 8, the target and the seeds fixed and move only the participation ratio, so
    their difference is the direct K-versus-K_eff contrast the ladder can only
    infer through a panel at ``corr(K, K_eff) = 0.828``. Reporting them as two
    separate rows against the ladder leaves that difference uncomputed.

    Sign convention: **positive means the left arm is worse**, since a higher
    RelMSE is a worse forecast.

    The origin is the inferential unit (`D30`) and RelMSE is scale-free, so this
    never compares scaler-space MSEs fitted under different ``sigma_g`` (`D60i`).
    It is a **post-hoc** statistic with no multiplicity control of its own: it
    guides what belongs in the paper, and root §9.2's pre-registered machinery is
    what a confirmatory claim goes through.

    Args:
        seed_avg: :func:`seed_average`'s output.
        left: ``(model_tag, k)``; ``k`` may be None when the tag has one rung.
        right: The arm it is measured against.

    Returns:
        ``left, right, mean_diff, se, t, p_two_sided, ci_low, ci_high,
        n_origins, left_better``. ``p_two_sided`` is referred to ``t(G-1)``.
    """
    a = per_origin_relmse(seed_avg, left[0], left[1])
    b = per_origin_relmse(seed_avg, right[0], right[1])
    joined = a.join(b, on="origin", how="inner", suffix="_right").sort("origin")
    diff = (
        joined.get_column("rel_mse").to_numpy()
        - joined.get_column("rel_mse_right").to_numpy()
    )
    g = int(diff.size)
    label = lambda arm: arm[0] if arm[1] is None else f"{arm[0]}-K{arm[1]}"
    if g < 2:
        return {
            "left": label(left), "right": label(right), "n_origins": g,
            "mean_diff": None, "se": None, "t": None, "p_two_sided": None,
            "ci_low": None, "ci_high": None, "left_better": None,
        }

    mean = float(diff.mean())
    se = float(diff.std(ddof=1) / math.sqrt(g))
    if se > 0:
        t_stat = mean / se
        p = 2.0 * _upper_tail(abs(t_stat), g - 1)
        half = _t_critical(g - 1) * se
    else:
        # Identical at every origin. The attention arm is exactly this: it
        # reproduces the main grid bit for bit, so the contrast is a zero with no
        # spread and a t-statistic would be 0/0 (`D62d`).
        t_stat, p, half = (0.0, 1.0, 0.0) if mean == 0.0 else (math.inf, 0.0, 0.0)

    return {
        "left": label(left),
        "right": label(right),
        "mean_diff": mean,
        "se": se,
        "t": t_stat,
        "p_two_sided": float(p),
        "ci_low": mean - half,
        "ci_high": mean + half,
        "n_origins": g,
        "left_better": int((diff < 0).sum()),
    }


def _t_critical(df: int) -> float:
    """Two-sided 5% Student-t critical value, normal fallback without scipy."""
    try:
        from scipy import stats as _stats  # root §16's named stats boundary

        return float(_stats.t.ppf(0.975, df=df))
    except ImportError:  # pragma: no cover - scipy ships with the Kaggle image
        return 1.959963984540054


def beta1_with_coverage(
    panel: pl.DataFrame,
    min_coverage: float = 0.9,
    B: int = 99_999,
    seed: int = 42,
) -> tuple[Beta1Result, Beta1Result | None]:
    """beta1 on the full panel, and on well-covered blocks only (`D45`).

    Test-window survival is conditioned on **future** gaps -- whether a forecast
    issued at *s* is evaluated depends on whether the next 120 hours contain an
    outage, information unavailable at *s* -- and Binance outages cluster on
    stress. So within an origin the surviving sample composition trends, the
    dropped targets are systematically the high-volatility ones, and beta1 would
    absorb that trend as though it were decay. Root §9.2 requires either a
    coverage covariate or a re-estimate on well-covered blocks; this is the
    second.

    Restricting usually leaves an **unbalanced** panel, and
    :func:`_balanced_matrix` refuses one by design: beta1's reduction to the mean
    of within-slopes holds only when every origin carries every block. ``None``
    comes back in that case, and that is the honest report -- the check could not
    be run, not that it passed. Only a restriction that removes whole origins
    leaves something estimable.

    Args:
        panel: :func:`amplification`'s output, carrying ``n_large``.
        min_coverage: Surviving windows as a fraction of :data:`BLOCK_HOURS`.
        B: Bootstrap draws, passed through to :func:`panel_beta1`.
        seed: Bootstrap seed, passed through.

    Returns:
        ``(full, restricted_or_None)``.
    """
    full = panel_beta1(panel, B=B, seed=seed)
    restricted = panel.filter((pl.col("n_large") / float(BLOCK_HOURS)) >= min_coverage)
    try:
        return full, panel_beta1(restricted, B=B, seed=seed)
    except ValueError:
        # Unbalanced after the restriction. Loosening the estimator to produce a
        # number here would answer a different question than the one asked.
        return full, None


def panel_beta1_covariate(
    panel: pl.DataFrame,
    value: str = "A",
    covariate: str = "coverage",
    B: int = 99_999,
    seed: int = 42,
) -> Beta1Result:
    """``A(i,b) = alpha_i + beta1 b + beta2 c(i,b) + eps``, clustered on origin.

    Root §9.2 requires **either** block coverage as a covariate **or** beta1
    re-estimated on well-covered blocks. Until `D80` neither had been produced:
    :func:`beta1_with_coverage` attempts the second and honestly returns ``None``
    because restricting unbalances the panel, so the requirement was reported as
    unmet by one route and never attempted by the other. This is the other route,
    and unlike the restriction it always runs --- the panel stays balanced because
    nothing is dropped.

    Why it matters, in the terms `D45` puts it: test-window survival is
    conditioned on **future** gaps, outages cluster on stress, so within an origin
    the surviving sample composition trends and beta1 would absorb that trend as
    though it were decay. Adding coverage as a regressor asks what is left of
    beta1 once the trend it could be absorbing is accounted for.

    Estimated by Frisch-Waugh inside the origin fixed effects: both the outcome
    and the block index are within-demeaned and then residualised on the
    within-demeaned coverage, after which beta1 is the simple slope. That is
    algebraically the two-regressor fit and it lets the same cluster-robust
    sandwich and the same restricted wild bootstrap carry over unchanged --- WCR,
    bootstrapping the cluster-robust *t*, one-sided at ``H1: beta1 < 0``, with
    Rademacher and Webb weights and the ``(1 + count) / (1 + B)`` floor (`D42`,
    `D53d`).

    Args:
        panel: Balanced long frame with ``origin``, ``block``, ``value`` and
            ``covariate``.
        value: Dependent variable column.
        covariate: The control. Coverage as a fraction, not a count, so
            ``beta2`` reads per unit of surviving-window share.
        B: Bootstrap replications.
        seed: Bootstrap seed, recorded so the p-value is regenerable (root §12).

    Returns:
        A :class:`Beta1Result` whose ``beta1`` is the coverage-adjusted slope.
        ``within_slopes`` are the per-origin adjusted slopes, so they still
        average to ``beta1`` on a balanced panel.

    Raises:
        ValueError: If the panel is unbalanced, or if coverage has no
            within-origin variation at all --- in which case there is nothing to
            adjust for and :func:`panel_beta1` is the estimator that applies.
    """
    a, blocks = _balanced_matrix(panel, value)
    c, _ = _balanced_matrix(panel, covariate)
    g, n_blocks = a.shape

    within = a - a.mean(axis=1, keepdims=True)
    cw = c - c.mean(axis=1, keepdims=True)
    xw = np.tile(blocks - blocks.mean(), (g, 1))

    scc = float((cw * cw).sum())
    if scc <= 0.0:
        raise ValueError(
            "coverage has no within-origin variation, so it cannot be a "
            "covariate here; panel_beta1 is the estimator that applies"
        )

    # Frisch-Waugh: residualise the regressor of interest and the outcome on the
    # control, both already swept of origin means. beta1 and the residuals of the
    # two-regressor fit are then exactly those of the simple fit on the residuals.
    xr = xw - (float((xw * cw).sum()) / scc) * cw
    ar = within - (float((within * cw).sum()) / scc) * cw
    sxx = float((xr * xr).sum())
    if sxx <= 0.0:
        raise ValueError("the block index is collinear with coverage within origin")

    beta = float((ar * xr).sum() / sxx)
    resid = ar - beta * xr
    score = (resid * xr).sum(axis=1)
    variance = float((score * score).sum()) / sxx**2
    se = math.sqrt(variance) if variance > 0 else float("nan")
    t_obs = beta / se if se == se and se > 0 else float("nan")

    # Restricted residuals: imposing beta1 = 0 leaves alpha_i and beta2, and ar is
    # already swept of both, so u_tilde is ar itself. Per-cluster inner products
    # against the fixed regressors are all a bootstrap draw needs.
    ux = (ar * xr).sum(axis=1)
    uc = (ar * cw).sum(axis=1)
    xx = (xr * xr).sum(axis=1)
    cx = (cw * xr).sum(axis=1)

    def _p(kind: str) -> float:
        rng = np.random.default_rng(seed)
        weights = _weights(kind, (B, g), rng)
        beta_star = (weights @ ux) / sxx
        delta_star = (weights @ uc) / scc
        score_star = (
            weights * ux[None, :]
            - delta_star[:, None] * cx[None, :]
            - beta_star[:, None] * xx[None, :]
        )
        var_star = np.square(score_star).sum(axis=1) / sxx**2
        ok = var_star > 0
        t_star = beta_star[ok] / np.sqrt(var_star[ok])
        below = int(np.sum(t_star <= t_obs))  # H1: beta1 < 0, left tail
        return (1.0 + below) / (1.0 + int(ok.sum()))

    return Beta1Result(
        beta1=beta,
        t_statistic=t_obs,
        cluster_se=se,
        p_rademacher=_p("rademacher"),
        p_webb=_p("webb"),
        n_clusters=g,
        n_observations=g * n_blocks,
        within_slopes=(ar * xr).sum(axis=1) / (xr * xr).sum(axis=1),
        B=B,
    )
