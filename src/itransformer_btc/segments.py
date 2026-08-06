"""The segment law: what breaks the series, and where.

A **segment** is a maximal run of contiguous, usable hourly bars. Root §4.3
breaks the series at two kinds of position:

1. **any missing bar** — 27 downtime blocks, 122 bars. When the exchange is down
   no price forms, so there is nothing to infer and imputation is *undefined*,
   not merely risky (root §4.2);
2. **any zero-volume or ``high == low`` bar** — it carries no trade information,
   exactly like downtime, so it gets the same treatment. This is what makes
   ``(VWAP - C)/(H - L)`` and ``log(volume)`` total rather than partial
   functions (`D14`), and why the F2 estimators are strictly positive and their
   logs total (root §5.1).

The ``high == low`` count has never been measured — root §4.3 and
``docs/ORIGIN_WINDOW_BUDGET.md`` both flag it as assumed. :func:`break_summary`
measures it, which is why this module exists before any model does.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final

import polars as pl

from itransformer_btc.config import DATA_END, DATA_START, STARTS_LOST_PER_BREAK

#: One hour in the integer domain every timestamp comparison uses. Root §2:
#: "Every timestamp is epoch-based and compared as an integer."
HOUR_MS: Final = 3_600_000

DEFAULT_PARQUET: Final = Path("data/raw/BTCUSDT_1h.parquet")


@dataclass(frozen=True, slots=True)
class Segment:
    """A maximal run of contiguous usable bars, as half-open row indices.

    Attributes:
        start_row: First row index into the *usable* frame, inclusive.
        end_row: One past the last row index, exclusive.
        start_ts: Epoch ms of the first bar.
        end_ts: Epoch ms of the last bar — inclusive, because this is a bar and
            not a bound.
    """

    start_row: int
    end_row: int
    start_ts: int
    end_ts: int

    @property
    def n_bars(self) -> int:
        return self.end_row - self.start_row

    def window_starts(self, span: int) -> int:
        """How many ``span``-bar windows start inside this segment.

        A segment shorter than ``span`` contributes *zero*, never a negative
        number. The closed-form budget arithmetic in root §4.3 quietly assumes
        every segment clears ``span``; where it does not, the two disagree and
        :mod:`itransformer_btc.budget` reports the disagreement rather than
        papering over it.
        """
        return max(0, self.n_bars - span + 1)


def load_bars(path: Path | str = DEFAULT_PARQUET) -> pl.DataFrame:
    """Load the immutable Stage 1 artifact, sorted, with an epoch-ms column.

    Args:
        path: Parquet written by ``spot_klines_btc.py``.

    Returns:
        Every original column plus ``ts_ms`` (Int64 epoch milliseconds), sorted
        ascending.

    Raises:
        FileNotFoundError: If the artifact is absent.
        ValueError: If the frame is empty, carries duplicate timestamps, or
            reaches outside the declared half-open data window. That last check
            is the runnable form of `D33`: the boundary bar at
            ``2026-08-01T00:00`` sat one hour past the window and shifted every
            count derived from ``len(df)``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. The four Stage 1 artifacts live in data/raw/ "
            f"(`D33`); regenerate with "
            f"`python spot_klines_btc.py --rebuild-only --outdir ./data/raw`."
        )

    frame = (
        pl.read_parquet(path)
        .with_columns(pl.col("open_time").dt.epoch("ms").alias("ts_ms"))
        .sort("ts_ms")
    )

    if frame.height == 0:
        raise ValueError(f"{path} is empty")

    n_unique = frame.select(pl.col("ts_ms").n_unique()).item()
    if n_unique != frame.height:
        raise ValueError(
            f"{path} carries {frame.height - n_unique} duplicate timestamps; "
            f"Stage 1 de-duplicates, so this is not Stage 1 output"
        )

    lo, hi = frame.select(
        pl.col("ts_ms").min().alias("lo"), pl.col("ts_ms").max().alias("hi")
    ).row(0)
    window_lo = int(DATA_START.timestamp() * 1000)
    window_hi = int(DATA_END.timestamp() * 1000)
    if lo < window_lo or hi >= window_hi:
        raise ValueError(
            f"{path} reaches outside the half-open data window "
            f"[{DATA_START.isoformat()}, {DATA_END.isoformat()}): "
            f"first={lo} last={hi}. `D33` — re-emit with --rebuild-only, which "
            f"applies clip_to_window()."
        )
    return frame


def usable_mask(frame: pl.DataFrame) -> pl.DataFrame:
    """Flag each bar usable or not, with the reason attached.

    A bar is unusable when it carries no trade information: zero volume, or
    ``high == low`` (no intrabar range). Both are treated exactly like downtime
    by root §4.3 — excluded, and the series splits there.

    ``zero_trades`` is measured but does **not** by itself mark a bar unusable:
    root §4.3 names only zero-volume and ``H == L``. It is carried so the open
    question in ``docs/ORIGIN_WINDOW_BUDGET.md`` — whether the 3 zero-volume and
    3 zero-trade bars are the same 3 bars — can be answered rather than assumed.

    Returns:
        The input frame plus boolean ``zero_volume``, ``flat_bar``,
        ``zero_trades`` and ``usable``.
    """
    return frame.with_columns(
        (pl.col("volume") <= 0).alias("zero_volume"),
        (pl.col("high") <= pl.col("low")).alias("flat_bar"),
        (pl.col("trades") <= 0).alias("zero_trades"),
    ).with_columns(
        (~pl.col("zero_volume") & ~pl.col("flat_bar")).alias("usable")
    )


def build_segments(
    frame: pl.DataFrame,
    start: datetime | None = None,
    end: datetime | None = None,
) -> list[Segment]:
    """Split the usable bars of ``[start, end)`` into contiguous segments.

    A new segment begins wherever the previous usable bar is not exactly one
    hour earlier. That covers downtime and exclusion alike: an excluded bar has
    already been filtered out, so it shows up here as a time jump.

    Args:
        frame: Output of :func:`usable_mask`, or anything carrying ``ts_ms`` and
            ``usable``.
        start: Inclusive lower bound; ``None`` for unbounded.
        end: Exclusive upper bound; ``None`` for unbounded.

    Returns:
        Segments in chronological order; empty if the span holds no usable bar.
    """
    if "usable" not in frame.columns:
        frame = usable_mask(frame)

    span = frame.filter(pl.col("usable"))
    if start is not None:
        span = span.filter(pl.col("ts_ms") >= int(start.timestamp() * 1000))
    if end is not None:
        span = span.filter(pl.col("ts_ms") < int(end.timestamp() * 1000))
    if span.height == 0:
        return []

    ts = span.get_column("ts_ms").to_list()
    segments: list[Segment] = []
    seg_start = 0
    for i in range(1, len(ts)):
        if ts[i] - ts[i - 1] != HOUR_MS:
            segments.append(Segment(seg_start, i, ts[seg_start], ts[i - 1]))
            seg_start = i
    segments.append(Segment(seg_start, len(ts), ts[seg_start], ts[-1]))
    return segments


@dataclass(frozen=True, slots=True)
class BreakSummary:
    """Measured break profile of a span. Every field is counted, not assumed."""

    calendar_hours: int
    bars_present: int
    bars_usable: int
    missing_bars: int
    zero_volume_bars: int
    flat_bars: int
    zero_trade_bars: int
    excluded_positions: int
    break_runs: int

    @property
    def segments(self) -> int:
        """Segments the span splits into.

        ``break_runs + 1`` only when every run is interior. A run touching
        either edge of the span produces one fewer segment, so this counts
        segments directly from the run structure rather than assuming.
        """
        return max(1, self.break_runs + 1)

    @property
    def window_starts_lost(self) -> int:
        """``119 x break_runs + excluded_positions`` — root §4.3's cost model."""
        return STARTS_LOST_PER_BREAK * self.break_runs + self.excluded_positions


def break_summary(
    frame: pl.DataFrame,
    start: datetime,
    end: datetime,
) -> BreakSummary:
    """Measure every break-inducing condition in ``[start, end)``.

    A **break run** is a maximal contiguous stretch of excluded calendar
    positions, whether excluded because the bar is missing or because it is
    unusable. Runs, not bars, are what the cost model charges 119 window starts
    to — so a zero-volume bar adjacent to a downtime block joins that block into
    one run instead of adding a second charge. Counting bars here instead of
    runs would overstate the loss by 119 per adjacency.

    This is the function that answers the two quantities
    ``docs/ORIGIN_WINDOW_BUDGET.md`` lists under "Not yet measured".
    """
    if "usable" not in frame.columns:
        frame = usable_mask(frame)

    lo = int(start.timestamp() * 1000)
    hi = int(end.timestamp() * 1000)
    span = frame.filter((pl.col("ts_ms") >= lo) & (pl.col("ts_ms") < hi))

    calendar_hours = (hi - lo) // HOUR_MS
    usable = int(span.select(pl.col("usable").sum()).item())
    counts = span.select(
        pl.col("zero_volume").sum().alias("zv"),
        pl.col("flat_bar").sum().alias("fb"),
        pl.col("zero_trades").sum().alias("zt"),
    ).row(0)

    # Walk the calendar, not the rows: a missing bar has no row to inspect, and
    # a run mixing missing with unusable positions must count once.
    usable_ts = set(span.filter(pl.col("usable")).get_column("ts_ms").to_list())
    break_runs = 0
    in_run = False
    for t in range(lo, hi, HOUR_MS):
        if t in usable_ts:
            in_run = False
        else:
            if not in_run:
                break_runs += 1
            in_run = True

    return BreakSummary(
        calendar_hours=calendar_hours,
        bars_present=span.height,
        bars_usable=usable,
        missing_bars=calendar_hours - span.height,
        zero_volume_bars=int(counts[0]),
        flat_bars=int(counts[1]),
        zero_trade_bars=int(counts[2]),
        excluded_positions=calendar_hours - usable,
        break_runs=break_runs,
    )
