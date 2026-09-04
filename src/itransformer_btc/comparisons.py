"""Table 6: every pair, the right statistic for each, and honest multiplicity.

Root §7 calls this the comparison an LTSF-literate reviewer wants, and `D40`
gave every baseline an explicit K precisely so it could be built. `D56` supplied
the missing models; `D60g` found the missing *call* -- the session log contains
zero Diebold-Mariano, Romano-Wolf or Model Confidence Set lines. This module is
that call.

Four decisions here are load-bearing, and each is pinned by root §9.2 rather
than chosen locally.

**Which statistic.** The comparisons that carry the paper are **nested**: the
ladder is cumulative, so K=1's feature set is a strict subset of K=8's under one
architecture; Naive-RW is nested inside every model in §7; Ridge-K1 inside
Ridge-K8. Under the null of equal population predictive ability with nested
models and estimated parameters, the loss differential has a mean shifted away
from zero -- the larger model's extra estimation noise makes it look worse --
and the statistic is not asymptotically ``N(0,1)`` (Clark & McCracken 2001;
McCracken 2007). Standard DM is therefore systematically **undersized against
the alternative this study exists to establish**. Nested pairs take Clark-West;
non-nested pairs take DM. Which one ran is a **column of the output**, never an
inference the reader has to make.

**Which multiplicity correction.** White's Reality Check (2000) and Hansen's SPA
(2005) test a *one-against-many* null and return one p-value for that composite;
they say nothing about the all-pairs matrix this table is (`D35`). With ten
models the matrix holds 45 tests and at alpha = 0.05 expects ~2.3 spurious
rejections under a complete null. Romano-Wolf (2005) stepdown controls FWER
across all pairs and is bootstrap-based like the machinery already here. The
Model Confidence Set answers the question a reader actually has -- *which models
are indistinguishable from the best* -- which the pairwise matrix does not.

**What the bootstrap resamples.** Origins, not observations. Blocks within an
origin come from one trained model, and root §9.2 fixes the inferential unit at
the origin: G = 15, with effective independence bounded near 4 by the 79.2%
training-window overlap, which is stated wherever these p-values are. **One
resample of origins is applied to every pair at once** -- that shared draw is
what preserves the cross-pair dependence Romano-Wolf needs, and what makes this
a stepdown rather than 45 separate tests.

**How cells are combined.** Root §9.2 pins the DM sample per (origin, block) on
the overlapping hourly loss differential, requires the combining method be
*stated*, and forbids concatenating ``d_t`` across origins, because the model
changes at each origin and the DM null has no interpretation across that
boundary. Stated, then: the per-origin mean differential is the unit, the
cluster bootstrap over the 15 of them is the inference, and the per-(origin,
block) HLN statistic travels beside it as a diagnostic with its ``T`` and ``h``.

Upstream
--------
**Both procedures are written here on numpy. Neither existed in this package
until `D62a`, and no package supplying them is a dependency.**

- ``romano_wolf`` — J. P. Romano and M. Wolf, "Stepwise multiple testing as
  formalized data snooping," *Econometrica*, vol. 73, no. 4, pp. 1237-1282,
  2005. It controls FWER across the all-pairs matrix, which is what White's
  Reality Check (2000) and Hansen's SPA (2005) cannot do: those test a
  one-against-many null and say nothing about a pairwise family (`D35`).
- ``model_confidence_set``, ``mcs_table`` — P. R. Hansen, A. Lunde, and
  J. M. Nason, "The model confidence set," *Econometrica*, vol. 79, no. 2,
  pp. 453-497, 2011. Reported at 90% and 75% as a membership column in Table 6.

Romano-Wolf is reported in two columns after `D79` — the all-pairs family and
the declared claim family — because widening the model set from 12 to 15 took
it from 31 of 66 rejections to 0 of 105 while no effect moved.
:data:`itransformer_btc.config.SOURCE_PROVENANCE` carries these rows in full.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl

from itransformer_btc.config import ORIGINS, PRED_LEN, TEST_BLOCKS
from itransformer_btc.metrics import (
    hln_test,
    load_meta,
    load_predictions,
    parse_run_id,
)

#: ``(model_tag, k)``. ``("naive", 0)`` denotes Naive-RW, which needs no run:
#: root §7 defines it as ``y_raw = 0``, mapped into scaler space as
#: ``y_z = -mu_g/sigma_g`` (`D31`), evaluated on exactly the rows its comparator
#: was scored on.
ModelKey = tuple[str, int]

#: Naive-RW's sentinel key.
NAIVE: Final[ModelKey] = ("naive", 0)

#: Models whose feature sets nest along K under one architecture. ``itru`` is
#: absent deliberately: it is the uniform-attention arm of ``itr`` at K=8 and
#: sees exactly the same eight variates, so it nests against nothing along K --
#: the arm varies *what attention selects*, not what the model can see (`D50`).
NESTS_ALONG_K: Final[frozenset[str]] = frozenset({"itr", "rdg"})

#: Bootstrap draws. The floor on any p-value is ``1/(1+B)`` (`D53d`): a literal
#: ``p = 0`` is not a probability, and the observed statistic belongs to its own
#: reference distribution (Davison & Hinkley 1997).
DEFAULT_B: Final = 9_999

#: Model Confidence Set levels root §9.2 asks for, as a membership column.
MCS_LEVELS: Final[tuple[float, ...]] = (0.10, 0.25)

#: The claim families the stepdown is *also* reported within (`D79`), in the
#: order Table 6 groups them.
#:
#: **Why a second column exists at all.** Romano-Wolf controls FWER over the
#: family it is given, and this table gives it the Cartesian product. Widening
#: the model list from twelve to fifteen took the matrix from 66 pairs to 105 and
#: the surviving rejections from 31 to **zero** -- not because any effect moved,
#: but because the two naive comparators generate the largest ``|t|`` in the
#: table (up to 8.5) and the shared bootstrap draw puts them into the max-``|t|``
#: null that every other hypothesis is judged against. Two closed-form baselines
#: that no claim in the paper rests on cost the paper every adjusted rejection it
#: had. That is a property of the declared family, not of the data.
#:
#: **What this column is and is not.** ``p_romano_wolf`` over all pairs stays the
#: headline: it is what root §9.2 says in as many words, and it is the
#: conservative number. ``p_romano_wolf_family`` steps down within one claim
#: family instead, which is the standard construction when the families are
#: declared in advance -- and this one was **not**, so it is reported as
#: *post-hoc* and labelled that way wherever it appears. Reporting both is the
#: only honest option left once the all-pairs column has been seen: suppressing
#: the narrower one hides how much of the collapse is the family's doing, and
#: promoting it to headline would be choosing the correction after seeing which
#: correction rejects.
FAMILY_ORDER: Final[tuple[str, ...]] = (
    "vs-naive", "ladder", "architecture", "other",
)


def pair_family(left: ModelKey, right: ModelKey) -> str:
    """Which claim of the paper a pair speaks to (`D79`).

    Four families, mutually exclusive and exhaustive by construction, each
    matching a sentence the manuscript actually makes:

    * ``vs-naive`` --- every pair containing Naive-RW. Root §13.2's headline
      disclosure, *no model beats Naive-RW*, is exactly this family.
    * ``ladder`` --- both sides the same model tag, so only K moves. RQ1's rungs
      and ridge's own rungs.
    * ``architecture`` --- both sides the same K, different tags. The
      channel-independence comparison root §13.1 makes a Related Work pillar.
    * ``other`` --- the remainder, which no claim rests on: a rung of one model
      against a different rung of another.

    Declared as a function of the *keys* alone, so no p-value can enter the
    definition.
    """
    if NAIVE in (left, right):
        return "vs-naive"
    if left[0] == right[0]:
        return "ladder"
    if left[1] == right[1]:
        return "architecture"
    return "other"


def nesting_order(left: ModelKey, right: ModelKey) -> tuple[ModelKey, ModelKey] | None:
    """``(small, large)`` if one information set nests inside the other, else None.

    **The orientation is not cosmetic and getting it wrong inverts the answer.**
    Clark-West is
    ``f = (y - y_small)^2 - (y - y_large)^2 + (y_small - y_large)^2``, and the
    adjustment term is symmetric while the first two are not. Swapping the roles
    therefore reports ``-(first two) + adjustment``, which is not a Clark-West
    statistic at all. Against Naive-RW the error is loud in exactly the wrong
    way: every model returns a large positive statistic and appears to beat the
    baseline, while its own sample MSE is worse -- a contradiction a reader would
    have to resolve by disbelieving one number or the other.

    Root §9.2 names the nested pairs: K=1 vs K=8 under one architecture, any
    model against Naive-RW, and Ridge-K1 against Ridge-K8. Cross-arm pairs are
    **not** on that list and are treated as non-nested here even where the
    information sets happen to nest, because the model classes differ and §9.2's
    enumeration is the pre-registered one.
    """
    model_left, k_left = left
    model_right, k_right = right
    if model_left == NAIVE[0]:
        return (left, right)
    if model_right == NAIVE[0]:
        return (right, left)
    if model_left != model_right:
        return None
    if model_left not in NESTS_ALONG_K or k_left == k_right:
        return None
    return (left, right) if k_left < k_right else (right, left)


def is_nested(left: ModelKey, right: ModelKey) -> bool:
    """Whether one model's information set is a strict subset of the other's."""
    return nesting_order(left, right) is not None


def label(key: ModelKey) -> str:
    """``itr-K8``, or ``Naive-RW`` for the sentinel."""
    return "Naive-RW" if key == NAIVE else f"{key[0]}-K{key[1]}"


# -- the aligned prediction panel --------------------------------------------


@dataclass(frozen=True, slots=True)
class PredictionPanel:
    """Seed-averaged forecasts for every model at every origin, aligned row-wise.

    Every model in the study is scored on **identical** evaluated windows --
    verified on the artifact: at origin 1, ``itr``, ``ptst`` and ``rdg`` each hold
    88,992 rows over 3,708 timestamps, their timestamp sets are equal, and
    ``max |y_true_itr - y_true_ptst|`` is exactly 0. `D45` makes that an
    assertion rather than an assumption, because a differential across two
    samples is not a differential, and :func:`build_panel` refuses a mismatch.

    Forecasts are averaged **across seeds before** any differential is formed,
    which is `D42`'s order of operations one level down from the ratio metrics it
    was written for: average the primitive, then form the derived quantity. The
    alternative -- differencing per seed and averaging statistics -- requires
    pairing seed 42 at K=1 with seed 42 at K=8, which are independent training
    runs of different models.
    """

    keys: tuple[ModelKey, ...]
    origin_indices: tuple[int, ...]
    origins: tuple[str, ...]
    #: origin index -> ``(n_rows,)`` block labels, sorted with the arrays below.
    block: dict[int, np.ndarray]
    #: origin index -> ``(n_rows,)`` realised target, shared by every model.
    y_true: dict[int, np.ndarray]
    #: ``(key, origin index)`` -> ``(n_rows,)`` seed-averaged forecast.
    y_pred: dict[tuple[ModelKey, int], np.ndarray]
    #: Forecast steps per window, so a per-origin reduction can recover ``T``.
    pred_len: int


def _run_ids(
    key: ModelKey, origin_index: int, roots: list[Path], pred_len: int
) -> list[str]:
    """Every seed of one cell that is actually on disk, in seed order."""
    model, k = key
    stem = f"{model}_o{origin_index:02d}_K{k:02d}_H{pred_len:03d}_s"
    found: set[str] = set()
    for root in roots:
        for path in (root / "preds").glob(f"{stem}*.parquet"):
            found.add(path.stem)
    return sorted(found, key=lambda run_id: int(parse_run_id(run_id)["seed"]))


def available_keys(
    keys: list[ModelKey],
    roots: list[Path],
    pred_len: int = PRED_LEN,
    origin_indices: tuple[int, ...] | None = None,
) -> tuple[list[ModelKey], list[ModelKey]]:
    """Split ``keys`` into those with a run at **every** origin, and the rest.

    Returns:
        ``(present, absent)``, both in the order given.

    :func:`build_panel` raises on a key missing at any origin, and that raise is
    right: a matrix short one origin for one model compares models over
    different origin sets, which is `D45` one level up. But an arm that has not
    run *anywhere* is a different situation --- the report should name it, the
    way an absent robustness arm is named, rather than refuse to generate until
    a GPU session finishes.

    Partial coverage is deliberately **not** rescued here. A key present at some
    origins and absent at others still reaches :func:`build_panel` and still
    raises: that is the defect, and quietly dropping it would hide it.
    """
    indices = origin_indices or tuple(o.index for o in ORIGINS)
    present: list[ModelKey] = []
    absent: list[ModelKey] = []
    for key in keys:
        if key == NAIVE or any(
            _run_ids(key, index, roots, pred_len) for index in indices
        ):
            present.append(key)
        else:
            absent.append(key)
    return present, absent


def build_panel(
    keys: list[ModelKey],
    roots: list[Path],
    pred_len: int = PRED_LEN,
    origin_indices: tuple[int, ...] | None = None,
) -> PredictionPanel:
    """Load, seed-average and align every model's forecasts.

    Args:
        keys: Models to compare. :data:`NAIVE` may appear and needs no run.
        roots: Artifact roots, working directory first.
        pred_len: Horizon. Table 6 is the headline H=24.
        origin_indices: Defaults to every origin in the walk-forward grid.

    Raises:
        FileNotFoundError: If a key has no run at some origin. A silently short
            matrix would compare models over different origin sets, which is the
            defect `D45` names one level up.
        ValueError: If two models were evaluated on different windows.
    """
    indices = origin_indices or tuple(o.index for o in ORIGINS)
    labels = tuple(o.label for o in ORIGINS if o.index in indices)

    block: dict[int, np.ndarray] = {}
    y_true: dict[int, np.ndarray] = {}
    y_pred: dict[tuple[ModelKey, int], np.ndarray] = {}
    signature: dict[int, tuple] = {}

    for origin_index in indices:
        for key in keys:
            if key == NAIVE:
                continue
            runs = _run_ids(key, origin_index, roots, pred_len)
            if not runs:
                raise FileNotFoundError(
                    f"{key} has no run at origin {origin_index} (H={pred_len}) in "
                    f"{[str(r) for r in roots]}"
                )
            stacked: list[np.ndarray] = []
            for run_id in runs:
                frame = load_predictions(run_id, roots).sort(
                    ["block", "timestamp", "step"]
                )
                sig = (
                    tuple(frame.get_column("block").to_list()),
                    tuple(frame.get_column("timestamp").to_list()),
                )
                if origin_index not in signature:
                    signature[origin_index] = sig
                    block[origin_index] = frame.get_column("block").to_numpy()
                    y_true[origin_index] = (
                        frame.get_column("y_true").to_numpy().astype(np.float64)
                    )
                elif sig != signature[origin_index]:
                    raise ValueError(
                        f"`D45`: {run_id} was evaluated on different windows than "
                        f"the first model at origin {origin_index}; a differential "
                        f"across two samples is not a differential"
                    )
                stacked.append(
                    frame.get_column("y_pred").to_numpy().astype(np.float64)
                )
            y_pred[(key, origin_index)] = np.mean(stacked, axis=0)

        if NAIVE in keys:
            # Root §7 / `D31`. Constant in scaler space, and the constant is
            # -mu_g/sigma_g rather than 0: zero there means r_hat = mu_g, the
            # training-window mean hourly return, so the "EMH baseline" would
            # silently be a constant-drift model. Read from a comparator's meta,
            # which is where the grid logged it per origin.
            donor = next(k for k in keys if k != NAIVE)
            meta = load_meta(_run_ids(donor, origin_index, roots, pred_len)[0], roots)
            y_pred[(NAIVE, origin_index)] = np.full(
                len(y_true[origin_index]), float(meta["naive_rw_z"])
            )

    return PredictionPanel(
        keys=tuple(keys),
        origin_indices=tuple(indices),
        origins=labels,
        block=block,
        y_true=y_true,
        y_pred=y_pred,
        pred_len=pred_len,
    )


def _per_window(values: np.ndarray, pred_len: int) -> np.ndarray:
    """Mean over the ``pred_len`` forecast steps of each window.

    The reduction is what keeps ``T`` counting **window starts** and ``h`` equal
    to the horizon, which is the sample root §9.2 pins: ``T ~ 720`` per block,
    truncation lag 23 at H=24. Left unreduced, ``T`` would be 17,280 and the HLN
    factor would be computed for a horizon the series does not have.
    """
    return values.reshape(-1, pred_len).mean(axis=1)


def differential(
    panel: PredictionPanel, left: ModelKey, right: ModelKey, origin_index: int
) -> np.ndarray:
    """Per-window loss differential for one pair at one origin, ``left - right``.

    Positive means ``right`` forecasts better. For a **nested** pair the
    Clark-West adjustment ``+(y_left - y_right)^2`` is added back, which is the
    whole content of `D29`: without it the larger model's extra estimation noise
    biases the differential against the alternative the study exists to test.

    Raises:
        ValueError: If the pair nests and ``left`` is not the restricted model.
            :func:`nesting_order` explains why silently accepting either order
            would report a quantity that is not a Clark-West statistic; refusing
            is cheaper than a table nobody can reconcile.
    """
    order = nesting_order(left, right)
    if order is not None and order[0] != left:
        raise ValueError(
            f"{label(left)} vs {label(right)} nests the other way: Clark-West "
            f"needs the restricted model first, so pass "
            f"({label(order[0])}, {label(order[1])}). See nesting_order."
        )
    y = panel.y_true[origin_index]
    a = panel.y_pred[(left, origin_index)]
    b = panel.y_pred[(right, origin_index)]
    d = np.square(y - a) - np.square(y - b)
    if order is not None:
        d = d + np.square(a - b)
    return _per_window(d, panel.pred_len)


def per_origin_differential(
    panel: PredictionPanel, left: ModelKey, right: ModelKey
) -> np.ndarray:
    """``(G,)`` mean differential per origin -- the unit inference runs on."""
    return np.array(
        [
            float(differential(panel, left, right, index).mean())
            for index in panel.origin_indices
        ]
    )


def per_origin_loss(panel: PredictionPanel, key: ModelKey) -> np.ndarray:
    """``(G,)`` mean squared error per origin, for the Model Confidence Set."""
    return np.array(
        [
            float(
                np.square(panel.y_true[index] - panel.y_pred[(key, index)]).mean()
            )
            for index in panel.origin_indices
        ]
    )


# -- clustered inference over origins ----------------------------------------


def _studentised(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Column means and their cluster standard errors, ``G = matrix.shape[0]``."""
    g = matrix.shape[0]
    return matrix.mean(axis=0), matrix.std(axis=0, ddof=1) / math.sqrt(g)


def cluster_bootstrap_t(
    per_origin: np.ndarray, B: int = DEFAULT_B, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Observed and bootstrap studentised statistics, resampling **origins**.

    Args:
        per_origin: ``(G, P)`` -- one mean differential per origin, per pair.
        B: Bootstrap draws.
        seed: Generator seed.

    Returns:
        ``(t_obs, t_boot)`` of shapes ``(P,)`` and ``(B, P)``. The bootstrap
        statistics are centred on the observed mean, so they are draws from the
        null. A resample that happens to pick one origin ``G`` times has no
        dispersion; it contributes 0 rather than an infinity.
    """
    g = per_origin.shape[0]
    theta, se = _studentised(per_origin)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_obs = np.where(se > 0, theta / se, 0.0)

    rng = np.random.default_rng(seed)
    draws = per_origin[rng.integers(0, g, size=(B, g))]
    theta_b = draws.mean(axis=1)
    se_b = draws.std(axis=1, ddof=1) / math.sqrt(g)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_boot = np.where(se_b > 0, (theta_b - theta) / se_b, 0.0)
    return t_obs, t_boot


def romano_wolf(
    per_origin: np.ndarray, B: int = DEFAULT_B, seed: int = 42
) -> np.ndarray:
    """Stepdown FWER-controlled p-values across every pair (Romano & Wolf 2005).

    Two-sided, deliberately: the family is "is there **any** difference in
    predictive ability between these two models", and mixing one- and two-sided
    alternatives inside one stepdown would make the controlled family
    ill-defined. The directional Clark-West reading travels in the raw column
    beside it.

    Args:
        per_origin: ``(G, P)`` mean differential per origin, per pair.
        B: Bootstrap draws.
        seed: Generator seed.

    Returns:
        ``(P,)`` adjusted p-values, monotone in ``|t|``.
    """
    t_obs, t_boot = cluster_bootstrap_t(per_origin, B=B, seed=seed)
    order = list(np.argsort(-np.abs(t_obs)))
    adjusted = np.empty(per_origin.shape[1])
    remaining = list(order)
    running = 0.0
    for position in order:
        block_max = np.abs(t_boot[:, remaining]).max(axis=1)
        raw = (1 + int((block_max >= abs(t_obs[position])).sum())) / (1 + B)
        running = max(running, raw)  # stepdown monotonicity
        adjusted[position] = min(running, 1.0)
        remaining.remove(position)
    return adjusted


def model_confidence_set(
    losses: np.ndarray, alpha: float, B: int = DEFAULT_B, seed: int = 42
) -> list[int]:
    """Hansen, Lunde & Nason (2011) MCS by the ``T_max`` statistic.

    Answers what a reader actually wants from Table 6 and the pairwise matrix
    does not: *which models are indistinguishable from the best*. Reported at
    90% and 75% as a membership column (root §9.2).

    Args:
        losses: ``(G, M)`` mean loss per origin per model.
        alpha: 0.10 for the 90% set, 0.25 for the 75% set.
        B: Bootstrap draws.
        seed: Generator seed.

    Returns:
        Column indices of the surviving models.
    """
    g, m = losses.shape
    alive = list(range(m))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, g, size=(B, g))

    while len(alive) > 1:
        sub = losses[:, alive]
        deviation = sub - sub.mean(axis=1, keepdims=True)
        theta, se = _studentised(deviation)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(se > 0, theta / se, 0.0)
        t_max = float(t.max())

        draws = deviation[idx]
        theta_b = draws.mean(axis=1)
        se_b = draws.std(axis=1, ddof=1) / math.sqrt(g)
        with np.errstate(divide="ignore", invalid="ignore"):
            t_b = np.where(se_b > 0, (theta_b - theta) / se_b, 0.0)
        p = (1 + int((t_b.max(axis=1) >= t_max).sum())) / (1 + B)

        if p >= alpha:
            break
        alive.pop(int(np.argmax(t)))  # eliminate the worst, then re-test
    return alive


# -- Table 6 -----------------------------------------------------------------


def _cell_diagnostics(
    panel: PredictionPanel, left: ModelKey, right: ModelKey, nested: bool
) -> dict[str, float | int | bool]:
    """Median HLN statistic over the (origin, block) cells, and how many reject.

    Root §9.2 computes the statistic **per (origin, block)** and states ``T``
    alongside every p-value. Those cell statistics do not carry the headline ---
    combining them would mean concatenating across a boundary the DM null does
    not survive --- but a table reporting only the origin-level number would hide
    how uniform the pair's ordering actually is across 90 cells.
    """
    stats: list[float] = []
    rejects = 0
    t_min = -1
    fallback = False
    name = f"{label(left)} vs {label(right)}"
    for index in panel.origin_indices:
        d = differential(panel, left, right, index)
        blocks = _per_window(
            panel.block[index].astype(np.float64), panel.pred_len
        ).round()
        for b in range(1, TEST_BLOCKS + 1):
            cell = d[blocks == float(b)]
            if len(cell) < 2:
                continue
            result = hln_test(cell, panel.pred_len, name=name, one_sided=nested)
            stats.append(result.statistic)
            rejects += int(result.p_value < 0.05)
            t_min = result.T if t_min < 0 else min(t_min, result.T)
            fallback = fallback or result.fallback_fired
    return {
        "s_star_median": float(np.median(stats)) if stats else float("nan"),
        "n_cells": len(stats),
        "n_cells_reject": rejects,
        "T_min": t_min,
        "fallback_fired": fallback,
    }


def pair_matrix(
    panel: PredictionPanel, B: int = DEFAULT_B, seed: int = 42
) -> pl.DataFrame:
    """Table 6 --- every unordered pair, with its statistic, p-values and MCS flags.

    Returns:
        One row per pair: ``left, right, nested, statistic_name, t_cluster,
        p_raw, p_romano_wolf, family, p_romano_wolf_family, s_star_median,
        n_cells, n_cells_reject, T_min, h, fallback_fired, G``, plus MCS
        membership for both models of the row.

    ``p_raw`` is one-sided for a nested pair, where root §9.2's alternative is
    directional, and two-sided otherwise. ``p_romano_wolf`` is always two-sided
    --- see :func:`romano_wolf`.

    **Two adjusted columns, and which is the headline is not a free choice**
    (`D79`). ``p_romano_wolf`` controls FWER across all pairs, which is what root
    §9.2 pre-registers, and it stays the headline. ``p_romano_wolf_family`` steps
    down inside the claim family :func:`pair_family` assigns; the families were
    declared after the all-pairs column had been read, so that column is
    **post-hoc** and every rendering of it says so.
    """
    keys = list(panel.keys)
    # Oriented, not merely enumerated: a nested pair is emitted restricted-model
    # first, so ``differential`` computes the Clark-West statistic rather than its
    # sign-inverted lookalike. See :func:`nesting_order`.
    pairs = [
        nesting_order(a, b) or (a, b)
        for i, a in enumerate(keys)
        for b in keys[i + 1 :]
    ]

    per_origin = np.column_stack(
        [per_origin_differential(panel, a, b) for a, b in pairs]
    )
    t_obs, t_boot = cluster_bootstrap_t(per_origin, B=B, seed=seed)
    p_adjusted = romano_wolf(per_origin, B=B, seed=seed)

    # `D79`. The same stepdown, run again inside each claim family. Same draw
    # seed, so the two columns differ in the family and in nothing else --- which
    # is what makes the difference between them readable as the cost of the
    # Cartesian product rather than as bootstrap noise.
    families = [pair_family(a, b) for a, b in pairs]
    p_family = np.ones(len(pairs))
    for name in FAMILY_ORDER:
        members = [i for i, f in enumerate(families) if f == name]
        if not members:
            continue
        p_family[members] = romano_wolf(per_origin[:, members], B=B, seed=seed)

    losses = np.column_stack([per_origin_loss(panel, k) for k in keys])
    members = {
        alpha: {keys[i] for i in model_confidence_set(losses, alpha, B=B, seed=seed)}
        for alpha in MCS_LEVELS
    }

    rows = []
    for position, (left, right) in enumerate(pairs):
        nested = is_nested(left, right)
        t = float(t_obs[position])
        if nested:
            # Directional: the alternative is that the larger model helps, and a
            # positive differential is what "helps" means here.
            count = int((t_boot[:, position] >= t).sum())
        else:
            count = int((np.abs(t_boot[:, position]) >= abs(t)).sum())
        rows.append(
            {
                "left": label(left),
                "right": label(right),
                "nested": nested,
                "statistic_name": "Clark-West" if nested else "DM-HLN",
                "t_cluster": t,
                "p_raw": (1 + count) / (1 + B),
                "p_romano_wolf": float(p_adjusted[position]),
                "family": families[position],
                "p_romano_wolf_family": float(p_family[position]),
                **_cell_diagnostics(panel, left, right, nested),
                "h": panel.pred_len,
                "G": int(per_origin.shape[0]),
                "left_in_mcs_90": left in members[0.10],
                "right_in_mcs_90": right in members[0.10],
                "left_in_mcs_75": left in members[0.25],
                "right_in_mcs_75": right in members[0.25],
            }
        )
    return pl.DataFrame(rows)


def mcs_table(
    panel: PredictionPanel, B: int = DEFAULT_B, seed: int = 42
) -> pl.DataFrame:
    """Model Confidence Set membership per model, with its mean loss and rank.

    The ``se_across_origins`` column is `D30`'s rule made structural: this row is
    aggregated across origins, so its dispersion is the standard error across
    origins and never the seed standard deviation, which measures
    re-initialisation noise on one fixed dataset and is roughly an order of
    magnitude smaller.
    """
    keys = list(panel.keys)
    losses = np.column_stack([per_origin_loss(panel, k) for k in keys])
    members = {
        alpha: {keys[i] for i in model_confidence_set(losses, alpha, B=B, seed=seed)}
        for alpha in MCS_LEVELS
    }
    mean_loss = losses.mean(axis=0)
    rank = {int(position): r + 1 for r, position in enumerate(np.argsort(mean_loss))}
    return pl.DataFrame(
        [
            {
                "model": label(key),
                "mean_loss": float(mean_loss[i]),
                "se_across_origins": float(
                    losses[:, i].std(ddof=1) / math.sqrt(losses.shape[0])
                ),
                "rank": rank[i],
                "in_mcs_90": key in members[0.10],
                "in_mcs_75": key in members[0.25],
            }
            for i, key in enumerate(keys)
        ]
    ).sort("rank")
