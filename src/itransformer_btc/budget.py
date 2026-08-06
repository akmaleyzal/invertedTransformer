"""Per-origin window accounting — the assertion target of root §11.

Importers: ``tests/test_data_plane.py``, and the Stage 2 launcher in
``notebooks/``. Reads ``data/raw/BTCUSDT_1h.parquet`` only through
:func:`itransformer_btc.segments.load_bars`; writes nothing.

``docs/ORIGIN_WINDOW_BUDGET.md`` was committed *before* any run so the pipeline
has something to be checked **against** rather than something to be tuned
**to**. This module measures the same quantities from the artifact and compares.

A divergence is a finding, not a nuisance. The committed table is derived from
the 27 downtime blocks alone, while the segment law (root §4.3) also breaks at
zero-volume and ``high == low`` bars, whose count has never been measured. If
those bars fall inside training spans, measured windows are **lower** than the
table and the table needs regenerating.

Root §11 requires an **exact equality per origin**, never a comparison against
the pooled 4.9% figure — asserted pooled, it fires spuriously at fourteen of
fifteen origins, gets loosened until it passes, and then can no longer
distinguish positional drift from ordinary between-origin variation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import polars as pl

from itransformer_btc.config import (
    ORIGINS,
    PRED_LEN,
    SEQ_LEN,
    STARTS_LOST_PER_BREAK,
    TEST_BLOCKS,
    WINDOW_SPAN,
    Origin,
)
from itransformer_btc.segments import (
    HOUR_MS,
    BreakSummary,
    break_summary,
    build_segments,
)
from itransformer_btc.windows import count_windows

#: Origin label to ``(break_runs, excluded_positions, windows_kept)`` for the
#: 21-month training sub-block. **Measured from the artifact on 2026-08-06**,
#: superseding the derived table `D26` shipped with (`D51`). It is pinned here
#: rather than recomputed inside the test so the test can catch *drift*: with
#: both sides computed the same way, a regression would agree with itself and
#: pass. Regenerate with :func:`format_markdown` and update both this dict and
#: ``docs/ORIGIN_WINDOW_BUDGET.md`` together, never one alone.
#:
#: The derived table it replaces read, for the first four origins,
#: ``(10, 86, 13_917) (12, 62, 13_727) (11, 47, 13_861) (12, 40, 13_797)``.
#: It diverged at twelve of fifteen origins for two reasons, both structural:
#: it never counted the three unusable bars, and its closed form charges a
#: short segment a negative window count. See `D51`.
COMMITTED_TRAIN_BUDGET: Final[dict[str, tuple[int, int, int]]] = {
    "2020-01": (11, 87, 13_934),
    "2020-06": (13, 63, 13_701),
    "2020-11": (12, 48, 13_741),
    "2021-04": (13, 41, 13_716),
    "2021-09": (14, 30, 13_560),
    "2022-02": (14, 32, 13_558),
    "2022-07": (9, 20, 14_165),
    "2022-12": (8, 19, 14_285),
    "2023-05": (2, 6, 15_021),
    "2023-10": (1, 2, 15_072),
    "2024-03": (1, 2, 15_120),
    "2024-08": (1, 2, 15_096),
    "2025-01": (1, 2, 15_096),
    "2025-06": (0, 0, 15_217),
    "2025-11": (0, 0, 15_217),
}


@dataclass(frozen=True, slots=True)
class OriginBudget:
    """Measured window accounting for one origin's training sub-block."""

    origin: Origin
    summary: BreakSummary
    windows_measured: int
    windows_closed_form: int
    test_block_starts: tuple[int, ...]

    @property
    def label(self) -> str:
        return self.origin.label

    @property
    def loss_pct(self) -> float:
        """Window starts lost to breaks, as a percentage of a gap-free span."""
        ceiling = self.summary.calendar_hours - STARTS_LOST_PER_BREAK
        return 100.0 * (1.0 - self.windows_measured / ceiling) if ceiling else 0.0

    @property
    def closed_form_agrees(self) -> bool:
        """Whether root §4.3's arithmetic matches the segment-wise truth.

        Disagreement means some segment is shorter than one window, so the
        closed form has gone negative somewhere and been silently absorbed.
        """
        return self.windows_measured == self.windows_closed_form


def surviving_block_starts(frame: pl.DataFrame, origin: Origin, b: int) -> int:
    """Window starts surviving inside test block ``b`` — out of 720.

    **Test blocks do not use the training semantics, and the difference is 119
    windows per block.** A training window must lie wholly inside its span,
    because its target may not cross into validation (root §8.2). A *test*
    window may not: root §8.3 states explicitly that a window's 96-bar input
    reaching back across the boundary is past information legitimately available
    to a forecaster at that moment, and blocking it "would make the evaluation
    unrealistically pessimistic". So every one of the block's 720 hours is an
    admissible forecast origin; what disqualifies one is a break inside the 120
    bars it spans, wherever those bars fall.

    Counting test blocks the training way yields 601 out of 720 even for a
    perfectly clean block — a 16.5% phantom loss that would be read as outage
    damage and would enter §9.2's block-coverage covariate as pure noise.
    """
    lo, hi = origin.block(b)
    lo_ms = int(lo.timestamp() * 1000)
    hi_ms = int(hi.timestamp() * 1000)
    span_ms = (WINDOW_SPAN - 1) * HOUR_MS

    usable = set(
        frame.filter(pl.col("usable")).get_column("ts_ms").to_list()
    )
    survivors = 0
    for start in range(lo_ms, hi_ms, HOUR_MS):
        # The window is contiguous exactly when every hour it spans is usable.
        if all((start + k * HOUR_MS) in usable for k in range(WINDOW_SPAN)):
            survivors += 1
    return survivors


def origin_budget(frame: pl.DataFrame, origin: Origin) -> OriginBudget:
    """Measure one origin's training sub-block and its six test blocks.

    The sub-block is ``[train_start, train_sub_end)`` — 21 months, not 24
    (`D25`). Test-block figures are surviving window *starts* inside each block.
    """
    summary = break_summary(frame, origin.train_start, origin.train_sub_end)
    segments = build_segments(frame, origin.train_start, origin.train_sub_end)

    ceiling = summary.calendar_hours - STARTS_LOST_PER_BREAK
    blocks = [surviving_block_starts(frame, origin, b) for b in range(1, TEST_BLOCKS + 1)]

    return OriginBudget(
        origin=origin,
        summary=summary,
        windows_measured=count_windows(segments, SEQ_LEN, PRED_LEN),
        windows_closed_form=ceiling - summary.window_starts_lost,
        test_block_starts=tuple(blocks),
    )


def budget_table(frame: pl.DataFrame) -> list[OriginBudget]:
    """Measure every origin in the committed grid."""
    return [origin_budget(frame, origin) for origin in ORIGINS]


def format_markdown(budgets: list[OriginBudget]) -> str:
    """Render the measured table in the shape of ``ORIGIN_WINDOW_BUDGET.md``.

    Used to regenerate the document when measurement supersedes derivation —
    never to silently overwrite it. Root §12: a number that cannot be
    regenerated is a documented failure, not a footnote.
    """
    head = (
        "| # | Origin | Training sub-block | Breaks | Excluded | Windows kept "
        "| Loss | Test-block starts B1…B6 |\n"
        "|---:|---|---|---:|---:|---:|---:|---|\n"
    )
    rows = [
        f"| {b.origin.index:>2} | {b.origin.origin:%Y-%m-%d} "
        f"| {b.origin.train_start:%Y-%m-%d} → {b.origin.train_sub_end:%Y-%m-%d} "
        f"| {b.summary.break_runs} | {b.summary.excluded_positions} "
        f"| {b.windows_measured:,} | {b.loss_pct:.1f}% "
        f"| {' / '.join(str(n) for n in b.test_block_starts)} |"
        for b in budgets
    ]
    return head + "\n".join(rows) + "\n"
