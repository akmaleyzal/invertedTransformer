"""Every table and figure the manuscript needs, generated rather than transcribed.

Root §12: *"Aggregation writes ``paper_numbers.json``, and every table and figure
is generated **from that file** rather than transcribed. A number that cannot be
regenerated is a documented failure, not a footnote."*

`D60g` recorded what the grid actually left behind: eight tables and seven
figures promised by §13.4, **none generated**, and four of them --- Table 6,
Table 8, Figure 5, Figure 7 --- with no inputs at all. `D56` had already fixed
the missing *models*; what stayed missing was the missing *call*. This module is
that call.

Two files, and the distinction is load-bearing. ``notebooks/outputs/artifacts/
paper_numbers.json`` is the **grid's** output and stays immutable evidence.
:func:`build_report` reads it, computes every analysis pass the grid never ran,
and returns the **manuscript's** single source --- which names the grid file by
digest, so the two cannot silently diverge. Tables and figures read only the
second. Note that repo-root ``artifacts/`` holds one stale 2026-08-06 CPU smoke
run and is not the results directory (`D60f`); passing it here is a caller error
and is rejected on the run count.

Three reporting rules from ``CLAUDE.md`` are enforced here rather than left to
the person writing the paper, because each has already been broken once:

* **Dispersion is bound to the aggregation level** (`D30`). Anything aggregated
  across origins carries the SE **across origins**; seed dispersion appears only
  as a Monte-Carlo diagnostic column. Measured on this grid the two differ by
  roughly seventy-fold --- seed std 0.001334 against an origin SE of 0.0959 ---
  so reporting the first as the second would understate the headline uncertainty
  by about that factor, reintroducing through a reporting convention exactly the
  overstated precision the wild cluster bootstrap was added to prevent.
* **Cross-origin comparisons are on RelMSE or R2_oos, never scaler-space MSE**
  (`D60i`). Two arms fitted at different origins carry different ``sigma_g``, so
  a raw MSE difference compares numbers in different units --- which is how the
  falsification arm shipped a figure that was 99.7% scaler drift and whose sign
  read backwards.
* **RQ3's wording is fixed and is not interchangeable** (`D60b`). *"The decay
  estimand is undefined under non-positive out-of-sample skill"*, **never** *"no
  decay detected within 180 days"*: the second is §3's right-censored phrasing
  and it asserts an edge the data does not contain.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl

from itransformer_btc.budget import budget_table
from itransformer_btc.comparisons import (
    NAIVE,
    ModelKey,
    build_panel,
    label,
    mcs_table,
    pair_matrix,
)
from itransformer_btc.config import (
    BARS_ACTUAL,
    BARS_EXPECTED,
    DATA_END,
    DATA_START,
    GAP_BLOCKS,
    MISSING_BARS,
    ORIGINS,
    PRED_LEN,
    SWEEP_ORIGIN_INDICES,
)
from itransformer_btc.economics import SLIPPAGE_BAND, economics_table, equity_curves
from itransformer_btc.efficiency import efficiency_table
from itransformer_btc.keff import rolling_ols_r2, rolling_pr
from itransformer_btc.metrics import (
    amplification,
    attention_amplification,
    beta1_with_coverage,
    directional_accuracy_table,
    falsification_relmse,
    gather_grid,
    load_meta,
    parse_run_id,
    raw_scale_table,
    seed_average,
)
from itransformer_btc.segments import break_summary
from itransformer_btc.runner import completed_run_ids

#: Every model Table 6 compares, in the order the table prints them. Twelve
#: models is 66 unordered pairs --- which is precisely why `D35` replaced SPA and
#: Reality Check here: those test a one-against-many null and say nothing about
#: an all-pairs matrix, where a complete null expects ~3 spurious rejections at
#: alpha = 0.05.
COMPARISON_KEYS: Final[tuple[ModelKey, ...]] = (
    ("itr", 1), ("itr", 4), ("itr", 8), ("itr", 12),
    ("itru", 8),
    ("rdg", 1), ("rdg", 4), ("rdg", 8), ("rdg", 12),
    ("dlin", 8), ("ptst", 8),
    NAIVE,
)

#: Models carried into the economic evaluation. A subset, because §13.5 asks for
#: an interval on every figure and a twelve-model Table 8 would be a wall of
#: numbers rather than a comparison. Ridge is in it because `D60c` made it the
#: finding: at its selected alpha it shrinks close enough to the training mean
#: that it nearly *is* the baseline, and it loses to Naive-RW by ~30x less than
#: any deep model.
ECONOMIC_KEYS: Final[tuple[ModelKey, ...]] = (
    ("itr", 1), ("itr", 8), ("rdg", 8), ("dlin", 8), ("ptst", 8),
)

#: Arms whose runs may legitimately be absent: they are the `D62` robustness
#: arms, executed after the 684-run grid. A generator that crashed on their
#: absence would make the report un-runnable until a GPU session finished.
ROBUSTNESS_TAGS: Final[dict[str, str]] = {
    "itrl": "longsched",
    "itrc": "capacity",
    "itra": "attention",
}

#: Figure 5's regimes, fixed before any map was seen (`D48`): calm is the bottom
#: tercile of realised volatility across all test blocks, stress the top.
TERCILE_SHOWN: Final[tuple[str, str]] = ("calm", "stress")

#: The 684-run grid is the floor for a report. Below it the panel is unbalanced
#: and §9.1's estimators refuse it by design (`D54e`).
GRID_FLOOR: Final = 684

#: Human-readable model names, for tables and figure legends.
MODEL_NAMES: Final[dict[str, str]] = {
    "itr": "iTransformer",
    "itru": "iTransformer (uniform attn.)",
    "itrf": "iTransformer (fresh)",
    "itrl": "iTransformer (long schedule)",
    "itrc": "iTransformer (d_ff 512)",
    "itra": "iTransformer (attn. captured)",
    "rdg": "Ridge",
    "dlin": "DLinear",
    "ptst": "PatchTST",
}

_MISSING: Final = "---"


# -- formatting --------------------------------------------------------------


def fmt(value: float | int | None, digits: int = 4) -> str:
    """A number for LaTeX, or an em-dash. **Never** the string ``nan``.

    A NaN printed into a table is a defect wearing a value's clothes: it reads as
    a measurement and it is the absence of one. Anything non-finite comes back as
    an em-dash, so the gap is visible to a reader and to the test that forbids it.
    """
    if value is None:
        return _MISSING
    number = float(value)
    if not math.isfinite(number):
        return _MISSING
    if digits == 0:
        return f"{int(round(number)):,}"
    return f"{number:.{digits}f}"


def tex_escape(text: str) -> str:
    """Escape the characters that appear in this study's labels."""
    for old, new in (("_", r"\_"), ("%", r"\%"), ("&", r"\&"), ("#", r"\#")):
        text = text.replace(old, new)
    return text


def tabular(
    caption: str,
    tag: str,
    header: list[str],
    rows: list[list[str]],
    align: str,
    note: str = "",
) -> str:
    """One ``booktabs`` table float, ready to be included by the manuscript.

    ``booktabs`` and no vertical rules: the IEEE house style §1 targets, and the
    one that survives a two-column layout without a reader's help.
    """
    # The provenance line names the *command*, not the module. It is a string
    # literal in flattened notebook source, and the generator refuses to emit a
    # cell whose executable source still mentions the package by name (`D59`) ---
    # correctly, since it cannot tell a comment about the package from a
    # reference to it. Naming the command is more useful to a reader anyway.
    lines = [
        "% GENERATED --- do not hand-edit.",
        "% Regenerate: python tools/build_report.py",
        r"\begin{table}[!t]",
        r"\centering",
        r"\caption{" + caption + "}",
        r"\label{" + tag + "}",
        r"\begin{tabular}{" + align + "}",
        r"\toprule",
        " & ".join(header) + r" \\",
        r"\midrule",
    ]
    lines += [" & ".join(row) + r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    if note:
        lines.append(r"\vspace{2pt}\par\footnotesize " + note)
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def se_across(values: np.ndarray) -> float:
    """Standard error across origins --- `D30`'s only admissible ``+/-``."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return float("nan")
    return float(values.std(ddof=1) / math.sqrt(len(values)))


def _star(p: float | None, threshold: float = 0.05) -> str:
    """A significance mark, so a reader does not have to scan a p-value column."""
    if p is None or not math.isfinite(float(p)):
        return ""
    return r"$^{*}$" if float(p) < threshold else ""


# -- what a report is built from ---------------------------------------------


@dataclass(frozen=True, slots=True)
class ReportInputs:
    """Everything the tables and figures read, computed exactly once.

    The numbers dict is the manuscript's single source (root §12). The frames
    beside it are what the *figures* need and the JSON must not carry: Figure 2b
    alone is 3,044 points, and a JSON file that a human is expected to read
    should not be padded with a series only matplotlib consumes.
    """

    numbers: dict
    seed_avg: pl.DataFrame
    amplification: pl.DataFrame
    rolling_pr: pl.DataFrame
    rolling_r2: pl.DataFrame
    equity: pl.DataFrame
    attention: pl.DataFrame | None


# -- sections ----------------------------------------------------------------


def _provenance(artifacts: Path, grid: dict) -> dict:
    grid_path = artifacts / "paper_numbers.json"
    return {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "grid_paper_numbers": str(grid_path),
        "grid_paper_numbers_sha256": _sha256(grid_path),
        "grid_generated_utc": grid.get("generated_utc"),
        "artifacts_root": str(artifacts),
    }


def _dataset_section(bars: pl.DataFrame) -> dict:
    """Table 1 --- provenance, the measured gap profile, and per-origin budgets.

    Everything here is **measured from the artifact**, never transcribed from
    §4.1: `D33` exists because a report and its own gaps file disagreed by one
    bar, in the direction that flatters the data, and nothing downstream noticed.
    The ``H == L`` count in particular was assumed additive to the zero-volume
    count until `D51c` measured it: there are **3** unusable bars, and they are
    the *same* 3 bars, because no volume implies no trades implies a high and a
    low that never separate.
    """
    summary = break_summary(bars, DATA_START, DATA_END)
    budgets = budget_table(bars)
    return {
        "window": [DATA_START.isoformat(), DATA_END.isoformat()],
        "bars_expected": BARS_EXPECTED,
        "bars_actual": BARS_ACTUAL,
        "missing_bars": MISSING_BARS,
        "gap_blocks": GAP_BLOCKS,
        "measured": {
            "calendar_hours": summary.calendar_hours,
            "bars_present": summary.bars_present,
            "bars_usable": summary.bars_usable,
            "missing_bars": summary.missing_bars,
            "zero_volume_bars": summary.zero_volume_bars,
            "flat_bars": summary.flat_bars,
            "zero_trade_bars": summary.zero_trade_bars,
            "excluded_positions": summary.excluded_positions,
            "break_runs": summary.break_runs,
            "segments": summary.segments,
        },
        "per_origin": [
            {
                "origin": budget.label,
                "train_windows": budget.windows_measured,
                "closed_form": budget.windows_closed_form,
                "closed_form_agrees": budget.closed_form_agrees,
                "loss_pct": budget.loss_pct,
                "test_block_starts": list(budget.test_block_starts),
                "worst_block_starts": int(min(budget.test_block_starts)),
            }
            for budget in budgets
        ],
    }


def _architecture_section(run_ids: list[str], roots: list[Path]) -> dict:
    """Table 3 --- K, capacity and epochs-to-early-stop for every model.

    §6.2 requires epochs-to-stop logged per rung, and the reason is exact: it is
    how a reader tells a flat 8->12 rung from an under-trained one. Measured over
    the whole grid it is also what `D62c` rests on --- **0 of 444 iTransformer
    runs reached the 30-epoch cap**, so the cap was never the binding constraint
    and widening it alone would have been a no-op.

    Every baseline carries an explicit K (`D40`). For the channel-independent
    pair that K means *trained on eight channels*, not *predicts the target from
    eight channels*, and their ``best_val_mse`` is an all-channel figure that is
    **not** comparable to the ladder's target-channel one (`D56`).
    """
    rows: dict[tuple[str, int, int], dict] = {}
    for run_id in run_ids:
        parts = parse_run_id(run_id)
        key = (str(parts["model"]), int(parts["k"]), int(parts["pred_len"]))
        meta = load_meta(run_id, roots)
        row = rows.setdefault(key, {
            "model": key[0], "k": key[1], "pred_len": key[2],
            "n_parameters": meta.get("n_parameters"),
            "epochs": [], "n_runs": 0,
            "config": meta.get("config", {}),
            "schedule": meta.get("schedule"),
        })
        row["epochs"].append(int(meta.get("epochs_run", 0)))
        row["n_runs"] += 1

    out = []
    for row in rows.values():
        epochs = np.asarray(row.pop("epochs"), dtype=np.float64)
        row["epochs_mean"] = float(epochs.mean())
        row["epochs_max"] = int(epochs.max())
        row["epochs_at_cap"] = int((epochs >= 30).sum())
        out.append(row)
    out.sort(key=lambda row: (row["model"], row["pred_len"], row["k"]))
    return {"cells": out}


def _horizon_section(seed_avg: pl.DataFrame) -> dict:
    """Table 7 --- the horizon sweep, restricted to the four sweep origins.

    Restricted deliberately. The sweep ran at origins 1, 5, 10 and 15 (`D48`,
    named in advance so the choice could not follow the result), while H=24 also
    ran at all fifteen. Aggregating H=24 over fifteen origins and every other
    horizon over four would put a different sample in every column of one table,
    and the horizons would then differ in origin composition as well as in
    horizon --- the confound this study is built to avoid everywhere else.
    """
    sweep = seed_avg.filter(
        (pl.col("model") == "itr")
        & pl.col("origin_index").is_in(list(SWEEP_ORIGIN_INDICES))
    )
    per_origin = sweep.group_by(["origin", "k", "pred_len"]).agg(
        pl.col("rel_mse").mean().alias("rel_mse"),
        pl.col("r2_oos").mean().alias("r2_oos"),
        pl.col("mse_seed_std").mean().alias("seed_std"),
    )
    cells = []
    for (k, h), part in per_origin.group_by(["k", "pred_len"], maintain_order=True):
        r2 = part.get_column("r2_oos").to_numpy()
        cells.append({
            "k": int(k),
            "pred_len": int(h),
            "r2_oos": float(r2.mean()),
            "se_across_origins": se_across(r2),
            "rel_mse": float(part.get_column("rel_mse").to_numpy().mean()),
            "seed_std": float(part.get_column("seed_std").to_numpy().mean()),
            "n_origins": int(len(r2)),
        })
    cells.sort(key=lambda row: (row["pred_len"], row["k"]))
    return {"origins": list(SWEEP_ORIGIN_INDICES), "cells": cells}


def _robustness_section(seed_avg: pl.DataFrame, grid_r2: dict[int, float]) -> dict:
    """`D62b`, `D62c`, `D62d` --- the three arms, reported apart from RQ1-RQ3.

    An absent arm returns a ``status`` rather than raising: these run after the
    684-run grid, and a report that cannot be generated until a GPU session
    finishes is a report nobody regenerates.

    Whatever they return goes in the paper. A robustness arm reported only when
    it agrees with the headline is not a robustness arm, and §13.2 carries that
    as a disclosure.
    """
    out: dict = {}
    present = set(seed_avg.get_column("model").unique().to_list())
    for tag, arm in ROBUSTNESS_TAGS.items():
        if tag not in present:
            out[arm] = {
                "status": "not run",
                "model_tag": tag,
                "note": "exploratory `D62` arm; execute the manifest to populate it",
            }
            continue
        part = seed_avg.filter(
            (pl.col("model") == tag) & (pl.col("pred_len") == PRED_LEN)
        )
        per_origin = part.group_by(["origin", "k"]).agg(
            pl.col("r2_oos").mean().alias("r2_oos"),
            pl.col("rel_mse").mean().alias("rel_mse"),
        )
        cells = []
        for (k,), sub in per_origin.group_by(["k"], maintain_order=True):
            r2 = sub.get_column("r2_oos").to_numpy()
            cells.append({
                "k": int(k),
                "r2_oos": float(r2.mean()),
                "se_across_origins": se_across(r2),
                "n_origins": int(len(r2)),
                "grid_r2_oos_same_rung": grid_r2.get(int(k)),
            })
        cells.sort(key=lambda row: row["k"])
        out[arm] = {"status": "run", "model_tag": tag, "cells": cells}
    return out


def _load_attention(artifacts: Path) -> pl.DataFrame | None:
    """Figure 5's maps, or ``None`` --- they arrive only with the `D62d` arm."""
    folder = artifacts / "attn"
    files = sorted(folder.glob("*.parquet")) if folder.exists() else []
    if not files:
        return None
    frames = []
    for path in files:
        parts = parse_run_id(path.stem)
        frames.append(
            pl.read_parquet(path).with_columns(
                pl.lit(path.stem).alias("run_id"),
                pl.lit(int(parts["origin_index"])).cast(pl.Int32).alias("origin_index"),
                pl.lit(int(parts["seed"])).cast(pl.Int32).alias("seed"),
            )
        )
    return pl.concat(frames)


# -- the enriched paper_numbers.json ----------------------------------------


def build_report(
    artifacts: Path,
    bars: pl.DataFrame,
    features: pl.DataFrame,
    *,
    roots: list[Path] | None = None,
    bootstrap_b: int = 9_999,
    seed: int = 42,
    log=print,
) -> ReportInputs:
    """The manuscript's single source, plus the frames its figures read.

    Args:
        artifacts: Directory holding ``preds/``, ``meta/`` and the grid's own
            ``paper_numbers.json`` --- ``notebooks/outputs/artifacts/`` (`D60f`).
        bars: ``usable_mask(load_bars(...))`` --- Table 1's measured gap profile.
        features: The frame ``build_features`` returns. §4.5's efficiency tests
            and Figure 2b's rolling statistics read it; neither reads a
            prediction file.
        bootstrap_b: Draws for Romano-Wolf and the Model Confidence Set. The floor
            on any bootstrap p-value is ``1/(1+B)`` (`D53d`).

    Returns:
        A :class:`ReportInputs`. Every section of ``numbers`` names the span, the
        n and the dispersion measure it used; nothing in it is a bare point
        estimate a reader would have to trust.

    Raises:
        FileNotFoundError: If the grid's ``paper_numbers.json`` is absent.
        ValueError: If fewer than 684 runs are present --- a partial panel is a
            different estimand, not a noisier one (`D54e`).
    """
    grid_path = artifacts / "paper_numbers.json"
    if not grid_path.exists():
        raise FileNotFoundError(
            f"{grid_path} is absent. The grid's own aggregation writes it; this "
            f"function enriches it and never replaces it. Repo-root artifacts/ "
            f"holds one stale CPU smoke run and is not the results directory "
            f"(`D60f`)."
        )
    grid = json.loads(grid_path.read_text(encoding="utf-8"))

    # A resumed Kaggle session holds earlier runs under /kaggle/input/<slug>/ and
    # this session's under /kaggle/working/, so the caller passes what
    # ``discover_roots`` found. Locally the two coincide (root §10.5).
    roots = list(roots) if roots else [artifacts]
    run_ids = sorted(completed_run_ids(roots))
    log(f"report: {len(run_ids)} completed runs under {artifacts}")
    if len(run_ids) < GRID_FLOOR:
        raise ValueError(
            f"only {len(run_ids)} runs found, below the {GRID_FLOOR}-run grid. A "
            f"partial panel is a different estimand, not a noisier one (`D54e`). "
            f"Repo-root artifacts/ is a stale smoke run (`D60f`)."
        )

    raw = gather_grid(run_ids, roots)
    seed_avg = seed_average(raw)
    log(f"report: {raw.height} run-block rows -> {seed_avg.height} seed-averaged cells")

    dataset = _dataset_section(bars)
    architecture = _architecture_section(run_ids, roots)
    log(f"report: {len(architecture['cells'])} architecture cells")

    # §4.5 and Figure 2b read the feature frame; neither touches a prediction.
    efficiency = efficiency_table(features)
    roll_pr = rolling_pr(features, k=8)
    roll_r2 = rolling_ols_r2(features, k=8)
    log(f"report: efficiency {efficiency.height} spans, rolling {roll_pr.height} windows")

    # Table 6.
    panel = build_panel(list(COMPARISON_KEYS), roots)
    pairs = pair_matrix(panel, B=bootstrap_b, seed=seed)
    mcs = mcs_table(panel, B=bootstrap_b, seed=seed)
    log(f"report: {pairs.height} pairs, MCS over {mcs.height} models")

    # Table 8 and Figure 7.
    origin_indices = tuple(origin.index for origin in ORIGINS)
    economics = economics_table(
        roots, list(ECONOMIC_KEYS), origin_indices, SLIPPAGE_BAND, seed=seed
    )
    equity = equity_curves(roots, list(ECONOMIC_KEYS), origin_indices, SLIPPAGE_BAND)
    log(f"report: {economics.height} economic cells, {equity.height} equity points")

    # DA, raw scale, the falsification arm's real number, and `D45`'s coverage check.
    ladder_ids = [
        run_id for run_id in run_ids
        if parse_run_id(run_id)["model"] == "itr"
        and int(parse_run_id(run_id)["pred_len"]) == PRED_LEN
    ]
    da = directional_accuracy_table(ladder_ids, roots)
    log(f"report: DA over {da.height} runs")

    raw_scale = raw_scale_table(seed_avg)
    falsification = falsification_relmse(seed_avg)
    amp = amplification(seed_avg)
    attn_amp = attention_amplification(seed_avg)
    beta_full, beta_covered = beta1_with_coverage(amp, seed=seed)

    grid_r2 = {int(row["k"]): float(row["R2_oos"]) for row in grid["rq1"]["rung_effects"]}

    per_rung_raw = (
        raw_scale.filter((pl.col("model") == "itr") & (pl.col("pred_len") == PRED_LEN))
        .group_by(["origin", "k"]).agg(pl.col("rmse_raw").mean().alias("rmse_raw"))
        .group_by("k").agg(
            pl.col("rmse_raw").mean().alias("rmse_raw"),
            pl.col("rmse_raw").std().alias("sd_across_origins"),
            pl.len().alias("n_origins"),
        ).sort("k")
    )

    da_by_rung = (
        da.group_by("k").agg(
            pl.col("da_h1").mean().alias("da_h1"),
            pl.col("p_h1").median().alias("p_h1_median"),
            pl.col("da_hH").mean().alias("da_hH"),
            pl.col("p_hH").median().alias("p_hH_median"),
            pl.col("da_cum").mean().alias("da_cum"),
            pl.col("p_cum").median().alias("p_cum_median"),
            pl.col("da_hH_overlapping").mean().alias("da_hH_overlapping"),
            pl.col("da_cum_overlapping").mean().alias("da_cum_overlapping"),
            pl.len().alias("n_runs"),
        ).sort("k")
    )

    # Table 4's per-model summary, on the scale-free metrics only (`D60i`).
    main = seed_avg.filter(pl.col("pred_len") == PRED_LEN)
    per_model = []
    for tag, k in COMPARISON_KEYS:
        if tag == "naive":
            continue
        cell = main.filter((pl.col("model") == tag) & (pl.col("k") == k))
        by_origin = cell.group_by("origin").agg(
            pl.col("rel_mse").mean().alias("rel_mse"),
            pl.col("r2_oos").mean().alias("r2_oos"),
            pl.col("mse").mean().alias("mse"),
            pl.col("mse_seed_std").mean().alias("seed_std"),
        )
        r2 = by_origin.get_column("r2_oos").to_numpy()
        membership = {row["model"]: row for row in mcs.to_dicts()}
        name = label((tag, k))
        per_model.append({
            "model": name,
            "model_tag": tag,
            "k": k,
            "rel_mse": float(by_origin.get_column("rel_mse").to_numpy().mean()),
            "r2_oos": float(r2.mean()),
            "se_across_origins": se_across(r2),
            "seed_std": float(by_origin.get_column("seed_std").to_numpy().mean()),
            "n_origins": int(by_origin.height),
            "n_seeds": int(cell.get_column("n_seeds").max()),
            "in_mcs_90": bool(membership.get(name, {}).get("in_mcs_90", False)),
            "in_mcs_75": bool(membership.get(name, {}).get("in_mcs_75", False)),
        })

    gap = falsification.get_column("gap_rel_mse").to_numpy()
    by_origin_gap = (
        falsification.group_by("origin")
        .agg(pl.col("gap_rel_mse").mean().alias("gap"))
        .sort("origin")
    )
    origin_gap = by_origin_gap.get_column("gap").to_numpy()

    numbers = {
        "derived_from": _provenance(artifacts, grid),
        "input_parquet": grid.get("input_parquet"),
        "input_sha256": grid.get("input_sha256"),
        "input_sha256_source": grid.get("input_sha256_source"),
        "code_sha256": grid.get("code_sha256"),
        "runs_complete": len(run_ids),
        "runs_in_grid_file": grid.get("runs_complete"),
        # Carried through verbatim and unrecomputed: these are the confirmatory
        # answers, and re-deriving them here would create a second definition of
        # a number root §12 wants to have exactly one of.
        "keff": grid["keff"],
        "rq1": grid["rq1"],
        "rq2": grid["rq2"],
        "rq3": grid["rq3"],
        # Everything below is new; the grid computed none of it.
        "dataset": dataset,
        "architecture": architecture,
        "keff_rolling": {
            "window_days": 90,
            "descriptive_only": True,
            "note": (
                "Full-sample span: every origin's test block lies inside it, so "
                "root section 5.4 forbids this informing any design decision. "
                "The gate is gate_pr on the pre-first-origin span (`D02`)."
            ),
            "pr": {
                "n": roll_pr.height,
                "min": float(roll_pr.get_column("pr").min()),
                "max": float(roll_pr.get_column("pr").max()),
                "mean": float(roll_pr.get_column("pr").mean()),
                "sd": float(roll_pr.get_column("pr").std()),
            },
            "ols_r2": {
                "n": roll_r2.height,
                "min": float(roll_r2.get_column("r2").min()),
                "max": float(roll_r2.get_column("r2").max()),
                "mean": float(roll_r2.get_column("r2").mean()),
                "sd": float(roll_r2.get_column("r2").std()),
            },
        },
        "efficiency": efficiency.to_dicts(),
        "comparisons": {
            "models": [label(key) for key in COMPARISON_KEYS],
            "B": bootstrap_b,
            "p_floor": 1.0 / (1 + bootstrap_b),
            "pairs": pairs.to_dicts(),
            "mcs": mcs.to_dicts(),
        },
        "main_results": {
            "note": (
                "Aggregated across origins, so the dispersion is the SE ACROSS "
                "ORIGINS and the seed std is a Monte-Carlo diagnostic beside it, "
                "never the error bar (`D30`)."
            ),
            "by_model": per_model,
        },
        "economics": {
            "taker_fee_per_side": 0.0004,
            "slippage_band": list(SLIPPAGE_BAND),
            "phase_utc_hour": 0,
            "cells": economics.to_dicts(),
        },
        "directional_accuracy": {
            "note": (
                "h=1 is tested on hourly spacing; h=H and the cumulative return "
                "are tested on NON-OVERLAPPING windows only. The overlapping "
                "figures are descriptive and carry no p-value: their targets "
                "overlap by 23 of 24 hours, so Pesaran-Timmermann's variance is "
                "far too small and the test over-rejects badly (`D21`)."
            ),
            "by_rung": da_by_rung.to_dicts(),
            "n_runs": da.height,
        },
        "horizons": _horizon_section(seed_avg),
        "falsification": {
            "metric": "RelMSE",
            "note": (
                "Reported on RelMSE, never on scaler-space MSE (`D60i`): the two "
                "arms are fitted 90 days apart and carry different sigma_g, so a "
                "raw MSE difference compares numbers in different units. The "
                "shipped -0.053341 was ~99.7% scaler drift with its sign reversed."
            ),
            "mean_gap_rel_mse": float(gap.mean()),
            "se_across_origins": se_across(origin_gap),
            "n_cells": int(len(gap)),
            "n_origins": int(by_origin_gap.height),
            "origins_favouring_aged": int((origin_gap < 0).sum()),
            "origins_within_5e5_of_zero": int((np.abs(origin_gap) < 5e-5).sum()),
            "by_origin": by_origin_gap.to_dicts(),
        },
        "attention_amplification": {
            "note": (
                "A_attn holds information fixed and varies only what attention "
                "selects, which K=1 versus K=8 cannot do (`D50`)."
            ),
            "mean": float(attn_amp.get_column("A_attn").mean()),
            "n_cells": int(attn_amp.height),
        },
        "raw_scale": {
            "note": "RMSE in raw log-return units beside MSE in scaler space (§9.1).",
            "by_rung": per_rung_raw.to_dicts(),
        },
        "coverage": {
            "note": (
                "`D45`: test-window survival is conditioned on FUTURE gaps, and "
                "outages cluster on stress, so within an origin the surviving "
                "sample composition trends and beta1 would absorb it."
            ),
            "min_coverage": 0.9,
            "full": {
                "beta1": beta_full.beta1,
                "t": beta_full.t_statistic,
                "headline_p": beta_full.headline_p,
                "G": beta_full.n_clusters,
                "N": beta_full.n_observations,
            },
            "restricted": (
                None if beta_covered is None else {
                    "beta1": beta_covered.beta1,
                    "t": beta_covered.t_statistic,
                    "headline_p": beta_covered.headline_p,
                    "G": beta_covered.n_clusters,
                    "N": beta_covered.n_observations,
                }
            ),
            "restricted_unavailable_reason": (
                None if beta_covered is not None else
                "restricting to well-covered blocks unbalances the panel, and "
                "beta1's reduction to the mean of within-slopes holds only on a "
                "balanced one. That the check cannot run IS the `D45` finding."
            ),
        },
        "robustness": _robustness_section(seed_avg, grid_r2),
    }

    return ReportInputs(
        numbers=numbers,
        seed_avg=seed_avg,
        amplification=amp,
        rolling_pr=roll_pr,
        rolling_r2=roll_r2,
        equity=equity,
        attention=_load_attention(artifacts),
    )


def build_paper_numbers(
    artifacts: Path, bars: pl.DataFrame, features: pl.DataFrame, **kwargs
) -> dict:
    """:func:`build_report`'s numbers alone --- the manuscript's single source."""
    return build_report(artifacts, bars, features, **kwargs).numbers


# -- tables ------------------------------------------------------------------


def _table1(numbers: dict) -> str:
    data = numbers["dataset"]
    measured = data["measured"]
    rows = [
        [
            row["origin"],
            fmt(row["train_windows"], 0),
            fmt(row["loss_pct"], 2),
            fmt(row["worst_block_starts"], 0),
            "yes" if row["closed_form_agrees"] else "no",
        ]
        for row in data["per_origin"]
    ]
    note = (
        f"BTCUSDT spot 1\\,h, Binance REST. Window {data['window'][0][:10]} to "
        f"{data['window'][1][:10]}, end exclusive. "
        f"{data['bars_expected']:,} expected, {data['bars_actual']:,} actual, "
        f"{data['missing_bars']} missing across {data['gap_blocks']} downtime "
        f"blocks. Unusable bars: {measured['zero_volume_bars']} zero-volume, "
        f"{measured['flat_bars']} with $H=L$, {measured['zero_trade_bars']} "
        f"zero-trade --- and they are the \\emph{{same}} bars (`D51c'), so the "
        f"total is {measured['excluded_positions']}, not their sum. "
        f"Windows are counted segment-wise; the closed form of section 4.3 is "
        f"kept only as an upper bound (`D51a')."
    )
    return tabular(
        "Data provenance and the per-origin window budget.",
        "tab:dataset",
        ["Origin", "Train windows", "Loss (\\%)", "Worst test block", "Closed form agrees"],
        rows,
        "lrrrc",
        note,
    )


def _table2(numbers: dict) -> str:
    rows = []
    for row in numbers["efficiency"]:
        rows.append([
            tex_escape(str(row["span"])),
            fmt(row["n"], 0),
            fmt(row["adf_stat"], 2) + _star(row["adf_p"]),
            fmt(row["hurst"], 3),
            fmt(row.get("vr_2"), 3) + _star(row.get("vr_p_2")),
            fmt(row.get("vr_4"), 3) + _star(row.get("vr_p_4")),
            fmt(row.get("vr_8"), 3) + _star(row.get("vr_p_8")),
            fmt(row.get("vr_16"), 3) + _star(row.get("vr_p_16")),
        ])
    note = (
        "Log-returns. ADF: stationarity. Hurst by rescaled range: $H\\approx0.5$ "
        "reads as no long memory. Variance ratio (Lo--MacKinlay): $VR\\approx1$ is "
        "consistent with a random walk, $VR<1$ is mean reversion. "
        "$^{*}$ marks $p<0.05$. We do \\emph{not} claim the market is efficient: "
        "the evidence here is mixed --- the variance ratio rejects the random walk "
        "at every lag while the Hurst exponent sits slightly above one half --- and "
        "that is the finding section 4.5 asks for."
    )
    return tabular(
        "Preliminary market-efficiency tests, full sample and per training sub-block.",
        "tab:efficiency",
        ["Span", "$n$", "ADF", "Hurst", "$VR_2$", "$VR_4$", "$VR_8$", "$VR_{16}$"],
        rows,
        "lrrrrrrr",
        note,
    )


def _table2b(numbers: dict) -> str:
    keff = numbers["keff"]
    rows = [
        [
            fmt(row["k"], 0),
            fmt(row["PR_raw"], 3) + " $\\pm$ " + fmt(row.get("PR_raw_sd"), 3),
            fmt(row.get("PR_windownorm"), 3),
            fmt(row.get("stable_rank"), 3),
            fmt(row.get("crosslag_share"), 3),
        ]
        for row in keff["per_rung"]
    ]
    note = (
        r"Participation ratio per rung, measured \textbf{per origin on that "
        r"origin's own 21-month training sub-block} (`D44'), which is what keeps "
        r"RQ1's regressor from reading a single bar its outcome is measured on. "
        r"$\pm$ is the standard deviation across origins. "
        f"$corr(K, K_{{eff}}) = {fmt(keff['corr_k_keff'], 3)}$, against the "
        r"$\approx 0.97$ section 9.1 anticipated --- so the $K$-versus-$K_{eff}$ "
        r"horse race is \emph{more} identifiable than that section feared. "
        f"The Stage 3b gate measured {fmt(keff['gate_pr_k8_pre_first_origin'], 3)} "
        f"at $K=8$ on the pre-first-origin span, below the pre-registered floor "
        f"of {fmt(keff['gate_floor'], 1)}; `D48''s prescribed action is "
        r"disclosure, not a re-cut, and the grid proceeded unchanged. $K=12$ "
        r"carries a \emph{lower} PR than $K=8$, so the redundancy that rung was "
        r"designed to contain is stronger than section 5.2 expected, not weaker."
    )
    return tabular(
        "Effective dimensionality per rung.",
        "tab:keff",
        ["$K$", "PR (raw)", "PR (window-norm.)", "Stable rank", "Cross-lag share"],
        rows,
        "rrrrr",
        note,
    )


def _table3(numbers: dict) -> str:
    rows = []
    for cell in numbers["architecture"]["cells"]:
        if cell["pred_len"] != PRED_LEN:
            continue
        name = MODEL_NAMES.get(cell["model"], cell["model"])
        rows.append([
            tex_escape(name),
            fmt(cell["k"], 0),
            fmt(cell["n_parameters"], 0),
            fmt(cell["epochs_mean"], 2),
            fmt(cell["epochs_max"], 0),
            fmt(cell["epochs_at_cap"], 0),
            fmt(cell["n_runs"], 0),
        ])
    note = (
        "Every hyperparameter is adopted unchanged from Liu et al. (2024) except "
        "$d_{model}$, reduced from 512 to 128 against $\\sim$14{,}000 training "
        "windows; \\textbf{nothing is tuned per rung} (`D38'), which is what makes "
        "the rungs comparable. Ridge's $\\alpha$ is the only hyperparameter "
        "selected anywhere in this study, on the validation sub-block. "
        "Parameter count is identical across rungs \\emph{at a fixed horizon} "
        "(`D60h'). \\textbf{Epochs at cap} counts runs reaching the 30-epoch "
        "budget: no iTransformer run ever does, so the binding constraint is the "
        "learning-rate schedule, not the budget. For DLinear and PatchTST, $K=8$ "
        "means \\emph{trained on eight channels} through their published "
        "all-channel objective, not \\emph{predicts the target from eight "
        "channels}, and their validation losses are all-channel figures not "
        "comparable to the ladder's (`D56')."
    )
    return tabular(
        "Architectures, capacity, and epochs to early stop at $H=24$.",
        "tab:hyperparameters",
        ["Model", "$K$", "Params", "Epochs (mean)", "Max", "At cap", "Runs"],
        rows,
        "lrrrrrr",
        note,
    )


def _table4(numbers: dict) -> str:
    rows = []
    for row in numbers["main_results"]["by_model"]:
        membership = "90\\%, 75\\%" if row["in_mcs_75"] else (
            "90\\%" if row["in_mcs_90"] else _MISSING
        )
        rows.append([
            tex_escape(row["model"]),
            fmt(row["rel_mse"], 4),
            fmt(row["r2_oos"], 4) + " $\\pm$ " + fmt(row["se_across_origins"], 4),
            fmt(row["seed_std"], 6),
            membership,
            fmt(row["n_origins"], 0),
            fmt(row["n_seeds"], 0),
        ])
    note = (
        "$R^2_{oos} = 1 - \\text{RelMSE}$ against Naive-RW, which forecasts a zero "
        "raw log-return and is mapped into scaler space as $\\hat{y}_z = "
        "-\\mu_g/\\sigma_g$ (`D31'). \\textbf{Every model is worse than Naive-RW at "
        "every rung}; ridge is worse by roughly thirty times less than any deep "
        "model. The $\\pm$ is the standard error \\emph{across origins}; the seed "
        "column is Monte-Carlo noise on one fixed dataset and is a diagnostic, "
        "never the error bar (`D30'). MCS is the Model Confidence Set (Hansen, "
        "Lunde \\& Nason 2011) at the stated levels."
    )
    return tabular(
        "Main results at $H=24$, aggregated across all fifteen origins.",
        "tab:main",
        ["Model", "RelMSE", "$R^2_{oos}$", "Seed std", "In MCS", "Origins", "Seeds"],
        rows,
        "lrrrcrr",
        note,
    )


def _table5(numbers: dict) -> str:
    rq3 = numbers["rq3"]
    rows = []
    for row in rq3["b_star"]:
        headline = " (headline)" if abs(row["tau"] - rq3["tau_headline"]) < 1e-12 else ""
        rows.append([
            fmt(100 * row["tau"], 1) + "\\%" + headline,
            tex_escape(str(row["status"]).upper()),
            fmt(row.get("median_b_star"), 1),
            fmt(row.get("ci_low"), 1) + "--" + fmt(row.get("ci_high"), 1),
            fmt(row.get("events"), 0),
            fmt(row.get("censored"), 0),
            fmt(row.get("n_origins"), 0),
        ])
    excluded = ", ".join(rq3["excluded_origins"])
    note = (
        "$b^{*}(i) = \\min\\{b : D(i,b) > \\tau\\}$, right-censored at six blocks. "
        "\\textbf{The decay estimand is undefined under non-positive out-of-sample "
        "skill}: $D(i,b)$ is a proportion of skill lost and there is no skill to "
        "lose a proportion of. This is \\emph{not} the right-censored result "
        "``no decay detected within 180 days'', which would assert an edge the "
        "data does not contain (`D60b'). All fifteen origins are excluded on mean "
        f"$R^2_{{oos}} \\leq 0$ and are named rather than dropped: {excluded}. "
        "The log-rank test of H3 is unavailable because neither arm has a "
        "surviving origin, so H3 is \\textbf{untestable}, not rejected."
    )
    return tabular(
        "RQ3: retraining cadence at each pre-registered threshold.",
        "tab:decay",
        ["$\\tau$", "Status", "Median $b^{*}$", "CI", "Events", "Censored", "Origins"],
        rows,
        "llrrrrr",
        note,
    )


def _table6(numbers: dict) -> str:
    comparisons = numbers["comparisons"]
    rows = []
    for row in comparisons["pairs"]:
        rows.append([
            tex_escape(row["left"]),
            tex_escape(row["right"]),
            "CW" if row["statistic_name"].startswith("Clark") else "DM",
            fmt(row["t_cluster"], 3),
            fmt(row["p_raw"], 4),
            fmt(row["p_romano_wolf"], 4),
            fmt(row["T_min"], 0),
        ])
    note = (
        "\\textbf{CW} is Clark--West (2007), used on every \\emph{nested} pair --- "
        "the ladder is cumulative and Naive-RW is nested inside every model, and "
        "there standard DM is not asymptotically $N(0,1)$ and is systematically "
        "undersized against the alternative this study exists to establish "
        "(`D29'). \\textbf{DM} is Diebold--Mariano with the "
        "Harvey--Leybourne--Newbold correction, referred to $t(T-1)$, on a "
        "rectangular long-run variance with lag $h-1 = 23$ --- never Bartlett, "
        "which would shrink $\\hat\\gamma_{23}$ by about 92\\% and understate the "
        "variance (`D34'). $t$ is clustered on the origin, $G = 15$. "
        f"$p_{{RW}}$ is the Romano--Wolf (2005) stepdown across all "
        f"{len(comparisons['pairs'])} pairs, $B = {comparisons['B']:,}$, floor "
        f"${fmt(comparisons['p_floor'], 6)}$ (`D53d'). White's Reality Check and "
        "Hansen's SPA are \\emph{not} used here: they test a one-against-many null "
        "and say nothing about an all-pairs matrix (`D35'). $T$ is the smallest "
        "per-cell sample; $h = 24$."
    )
    return tabular(
        "Pairwise forecast comparison with family-wise error control.",
        "tab:dm",
        ["Left", "Right", "Stat.", "$t$", "$p_{raw}$", "$p_{RW}$", "$T_{min}$"],
        rows,
        "llcrrrr",
        note,
    )


def _table7(numbers: dict) -> str:
    horizons = numbers["horizons"]
    rows = [
        [
            fmt(cell["pred_len"], 0),
            fmt(cell["k"], 0),
            fmt(cell["rel_mse"], 4),
            fmt(cell["r2_oos"], 4) + " $\\pm$ " + fmt(cell["se_across_origins"], 4),
            fmt(cell["seed_std"], 6),
            fmt(cell["n_origins"], 0),
        ]
        for cell in horizons["cells"]
    ]
    note = (
        "Origins " + ", ".join(str(i) for i in horizons["origins"]) + ", named in "
        "advance (`D48') so the choice could not follow the result. $H=24$ is "
        "restricted to the same four origins here even though it ran at all "
        "fifteen: aggregating one column over fifteen origins and the rest over "
        "four would make the horizons differ in origin composition as well as in "
        "horizon. $\\pm$ is the standard error across origins."
    )
    return tabular(
        "Horizon sweep.",
        "tab:horizons",
        ["$H$", "$K$", "RelMSE", "$R^2_{oos}$", "Seed std", "Origins"],
        rows,
        "rrrrrr",
        note,
    )


def _table8(numbers: dict) -> str:
    cells = pl.DataFrame(numbers["economics"]["cells"])
    rows = []
    for (model, slippage), part in cells.group_by(
        ["model", "slippage_per_side"], maintain_order=True
    ):
        sharpe = part.get_column("sharpe_annualised").to_numpy()
        mdd = part.get_column("max_drawdown").to_numpy()
        rows.append([
            tex_escape(str(model)),
            fmt(100 * float(slippage), 2) + "\\%",
            fmt(sharpe.mean(), 3) + " $\\pm$ " + fmt(se_across(sharpe), 3),
            fmt(part.get_column("sortino_annualised").to_numpy().mean(), 3),
            fmt(mdd.mean(), 3),
            fmt(part.get_column("mdd_ci_low").to_numpy().mean(), 3) + "--"
            + fmt(part.get_column("mdd_ci_high").to_numpy().mean(), 3),
            fmt(part.get_column("turnover_per_period").to_numpy().mean(), 3),
            fmt(part.get_column("net_total_return").to_numpy().mean(), 4),
            fmt(part.get_column("dsr").to_numpy().mean(), 4),
        ])
    rows.sort(key=lambda row: (row[0], row[1]))
    note = (
        "Position from the sign of the cumulative 24-step forecast on \\emph{raw, "
        "drift-free} log-returns, opened at \\textbf{00:00 UTC} and held 24 hours, "
        "non-overlapping --- the phase is fixed in advance because there are 24 "
        "admissible alignments and each gives a different Sharpe (`D46'). Taker "
        "fee 0.04\\% per side, plus the slippage shown. Sharpe and Sortino are "
        "annualised at 365 periods; the DSR is computed per origin from the "
        "\\emph{per-period} Sharpe, with $N$ the configurations evaluated on that "
        "origin's own test span --- not the 837-run development total, which is "
        "reported separately in Limitations and is a different quantity (`D46'). "
        "$\\pm$ is the standard error across origins; the MDD interval is a "
        "stationary bootstrap. The strategy is flat wherever no valid window "
        "survives, and outages cluster on stress, so the reported drawdown is "
        "optimistic by an amount the flat-day count bounds (`D45')."
    )
    return tabular(
        "Economic evaluation across the pre-registered slippage band.",
        "tab:economics",
        ["Model", "Slip.", "Sharpe (ann.)", "Sortino", "MDD", "MDD CI",
         "Turnover", "Net return", "DSR"],
        rows,
        "llrrrrrrr",
        note,
    )


def render_tables(numbers: dict, out_dir: Path) -> list[Path]:
    """Write every table as a standalone ``.tex`` float.

    Returns:
        The paths written, in table order.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    builders = {
        "table1_dataset.tex": _table1,
        "table2_efficiency.tex": _table2,
        "table2b_keff.tex": _table2b,
        "table3_architecture.tex": _table3,
        "table4_main.tex": _table4,
        "table5_decay.tex": _table5,
        "table6_dm.tex": _table6,
        "table7_horizons.tex": _table7,
        "table8_economics.tex": _table8,
    }
    written = []
    for name, builder in builders.items():
        path = out_dir / name
        path.write_text(builder(numbers), encoding="utf-8")
        written.append(path)
    return written


# -- figures -----------------------------------------------------------------


def _pyplot():
    """Matplotlib on a headless backend, imported late.

    Late, because ``report`` is imported by the notebook's definition cells and a
    Kaggle session that never renders a figure should not pay for the import.
    ``Agg`` because no display exists in either place this runs.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save(fig, out_dir: Path, stem: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in ("pdf", "png"):
        path = out_dir / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight", dpi=200)
        paths.append(path)
    return paths


def _as_datetime(ms: np.ndarray) -> np.ndarray:
    return np.asarray(ms, dtype="int64").astype("datetime64[ms]")


def _figure2b(inputs: ReportInputs, out_dir: Path) -> list[Path]:
    """Rolling PR and rolling OLS $R^2$ --- H2's premise, before any model runs."""
    plt = _pyplot()
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 4.6), sharex=True)

    pr = inputs.rolling_pr
    axes[0].plot(_as_datetime(pr.get_column("window_end_ms").to_numpy()),
                 pr.get_column("pr").to_numpy(), lw=1.0, color="#22577a")
    axes[0].set_ylabel("PR at $K=8$")
    axes[0].grid(alpha=0.25, lw=0.5)

    r2 = inputs.rolling_r2
    axes[1].plot(_as_datetime(r2.get_column("window_end_ms").to_numpy()),
                 r2.get_column("r2").to_numpy(), lw=1.0, color="#c1121f")
    axes[1].set_ylabel("in-window $R^2$")
    axes[1].set_xlabel("window end (UTC)")
    axes[1].grid(alpha=0.25, lw=0.5)

    fig.suptitle("90-day rolling participation ratio and OLS fit (descriptive only)",
                 fontsize=10)
    fig.tight_layout()
    paths = _save(fig, out_dir, "figure2b_rolling")
    plt.close(fig)
    return paths


def _figure3(inputs: ReportInputs, out_dir: Path) -> list[Path]:
    """Figure 3 --- ``A(i,b)``: fifteen per-origin lines plus the fitted slope.

    **One series, not four** (`D36`). ``A(i,b)`` is defined only for the K=1
    versus K=8 pair and §3 fixes RQ2 on that pair, ``never K=12``; plotting an
    ``A`` for four rungs would require silently inventing a different estimand
    from the one $\\beta_1$ is regressed on. The per-origin form is also the
    better figure: it displays the actual identification, which is within-origin
    slopes. **This figure carries the entire paper.**
    """
    plt = _pyplot()
    amp = inputs.amplification
    numbers = inputs.numbers
    fig, ax = plt.subplots(figsize=(7.0, 4.2))

    for (origin,), part in amp.group_by(["origin"], maintain_order=True):
        part = part.sort("block")
        ax.plot(part.get_column("block").to_numpy(), part.get_column("A").to_numpy(),
                lw=0.8, alpha=0.55, marker="o", ms=2.5, color="#4a6fa5")

    beta1 = float(numbers["rq2"]["beta1"])
    blocks = np.arange(1, 7, dtype=float)
    intercept = float(amp.get_column("A").mean()) - beta1 * blocks.mean()
    ax.plot(blocks, intercept + beta1 * blocks, lw=2.4, color="#c1121f",
            label=f"fitted $\\beta_1$ = {beta1:+.6f}")

    mde = float(numbers["rq2"]["minimum_detectable_beta1"])
    ax.plot(blocks, intercept + mde * blocks, lw=1.6, ls="--", color="#333333",
            label=f"MDE at 80% power = {mde:+.6f}")

    ax.axhline(0.0, lw=0.8, color="#888888")
    ax.set_xlabel("test block $b$ (30 days each)")
    ax.set_ylabel("$A(i,b) = [MSE_{K1} - MSE_{K8}] / MSE_{K1}$")
    ax.set_title("Multivariate gap against model age, one line per origin", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    paths = _save(fig, out_dir, "figure3_decay")
    plt.close(fig)
    return paths


def _figure4(inputs: ReportInputs, out_dir: Path) -> list[Path]:
    """RelMSE per test block, every model, averaged over origins."""
    plt = _pyplot()
    main = inputs.seed_avg.filter(pl.col("pred_len") == PRED_LEN)
    fig, ax = plt.subplots(figsize=(7.0, 4.2))

    for tag, k in COMPARISON_KEYS:
        if tag == "naive":
            continue
        cell = main.filter((pl.col("model") == tag) & (pl.col("k") == k))
        if cell.height == 0:
            continue
        by_block = cell.group_by("block").agg(
            pl.col("rel_mse").mean().alias("rel_mse")
        ).sort("block")
        ax.plot(by_block.get_column("block").to_numpy(),
                by_block.get_column("rel_mse").to_numpy(),
                marker="o", ms=3.5, lw=1.2, label=label((tag, k)))

    ax.axhline(1.0, lw=1.2, color="#000000", ls="--", label="Naive-RW")
    ax.set_xlabel("test block $b$ (30 days each)")
    ax.set_ylabel("RelMSE (lower is better)")
    ax.set_title("RelMSE per block; everything above the dashed line loses to Naive-RW",
                 fontsize=10)
    ax.legend(fontsize=7, ncol=2, frameon=False)
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    paths = _save(fig, out_dir, "figure4_relmse")
    plt.close(fig)
    return paths


def _figure5(inputs: ReportInputs, out_dir: Path) -> list[Path]:
    """Attention maps, calm against stress --- terciles fixed in advance (`D48`).

    Returns an empty list when the `D62d` arm has not run: an empty axes labelled
    as a figure is worse than a named absence, because only one of the two tells
    a reader that nothing was measured.
    """
    if inputs.attention is None:
        return []
    plt = _pyplot()
    maps = inputs.attention
    layers = sorted(set(maps.get_column("layer").to_list()))
    fig, axes = plt.subplots(
        len(layers), 2, figsize=(6.4, 3.0 * len(layers)), squeeze=False
    )
    for row, layer in enumerate(layers):
        for col, tercile in enumerate(TERCILE_SHOWN):
            part = maps.filter(
                (pl.col("layer") == layer) & (pl.col("tercile") == tercile)
            ).group_by(["i", "j"]).agg(pl.col("weight").mean()).sort(["i", "j"])
            size = int(part.get_column("i").max()) + 1
            grid = part.get_column("weight").to_numpy().reshape(size, size)
            image = axes[row][col].imshow(grid, cmap="magma", vmin=0.0)
            axes[row][col].set_title(f"layer {layer}, {tercile}", fontsize=9)
            axes[row][col].set_xlabel("attended variate")
            axes[row][col].set_ylabel("query variate")
            fig.colorbar(image, ax=axes[row][col], fraction=0.046)
    fig.suptitle(
        "Variate attention, calm versus stress terciles of realised volatility",
        fontsize=10,
    )
    fig.tight_layout()
    paths = _save(fig, out_dir, "figure5_attention")
    plt.close(fig)
    return paths


def _figure6(inputs: ReportInputs, out_dir: Path) -> list[Path]:
    """Horizon sensitivity: $R^2_{oos}$ against $H$, one line per rung."""
    plt = _pyplot()
    cells = pl.DataFrame(inputs.numbers["horizons"]["cells"])
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for (k,), part in cells.group_by(["k"], maintain_order=True):
        part = part.sort("pred_len")
        ax.errorbar(
            part.get_column("pred_len").to_numpy(),
            part.get_column("r2_oos").to_numpy(),
            yerr=part.get_column("se_across_origins").to_numpy(),
            marker="o", ms=4, lw=1.2, capsize=3, label=f"$K={int(k)}$",
        )
    ax.axhline(0.0, lw=1.2, color="#000000", ls="--", label="Naive-RW")
    ax.set_xscale("log")
    ax.set_xticks([1, 3, 24, 168])
    ax.set_xticklabels(["1", "3", "24", "168"])
    ax.set_xlabel("forecast horizon $H$ (hours)")
    ax.set_ylabel("$R^2_{oos}$")
    ax.set_title("Horizon sweep at four named origins", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    paths = _save(fig, out_dir, "figure6_horizons")
    plt.close(fig)
    return paths


def _figure7(inputs: ReportInputs, out_dir: Path) -> list[Path]:
    """Equity curves at all three pre-registered slippage levels."""
    plt = _pyplot()
    equity = inputs.equity
    slippages = sorted(set(equity.get_column("slippage_per_side").to_list()))
    fig, axes = plt.subplots(1, len(slippages), figsize=(4.0 * len(slippages), 3.6),
                             sharey=True, squeeze=False)
    for col, slippage in enumerate(slippages):
        axis = axes[0][col]
        part = equity.filter(pl.col("slippage_per_side") == slippage)
        for (model,), sub in part.group_by(["model"], maintain_order=True):
            by_period = sub.group_by("period").agg(
                pl.col("equity").mean().alias("equity")
            ).sort("period")
            axis.plot(by_period.get_column("period").to_numpy(),
                      by_period.get_column("equity").to_numpy(),
                      lw=1.2, label=str(model))
        axis.axhline(0.0, lw=0.8, color="#888888")
        axis.set_title(f"slippage {100 * float(slippage):.2f}% per side", fontsize=9)
        axis.set_xlabel("trading day since origin")
        axis.grid(alpha=0.25, lw=0.5)
    axes[0][0].set_ylabel("cumulative net log return")
    axes[0][-1].legend(fontsize=7, frameon=False)
    fig.suptitle("Equity curves, averaged over the fifteen origins", fontsize=10)
    fig.tight_layout()
    paths = _save(fig, out_dir, "figure7_equity")
    plt.close(fig)
    return paths


def render_figures(inputs: ReportInputs, out_dir: Path, log=print) -> list[Path]:
    """Write every figure as ``.pdf`` and ``.png``.

    Figure 5 is skipped, **by name**, when the `D62d` attention arm has not run:
    an empty axes labelled as a figure reads as a measurement of nothing, and a
    named absence reads as what it is.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, builder in (
        ("figure2b", _figure2b),
        ("figure3", _figure3),
        ("figure4", _figure4),
        ("figure5", _figure5),
        ("figure6", _figure6),
        ("figure7", _figure7),
    ):
        paths = builder(inputs, out_dir)
        if not paths:
            log(f"report: {name} SKIPPED --- its input has not been produced yet")
            continue
        written.extend(paths)
    return written
