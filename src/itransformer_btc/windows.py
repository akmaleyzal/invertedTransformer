"""Window enumeration, validated by timestamp rather than by position.

Importers: ``itransformer_btc.budget`` and ``tests/test_data_plane.py``.
Reads no file directly — it consumes a frame produced by
:mod:`itransformer_btc.segments` and writes nothing.

Root §4.3 names this the highest-probability silent bug in the pipeline: after
any row drop, positional sliding closes gaps invisibly, and a window that spans
a two-day outage looks identical to one that does not. The rule is therefore

    window [s, s+L+H) is valid  <=>  t[s+L+H-1] - t[s] == (L+H-1) hours

and it is checked on every emitted window, not sampled. The check is cheap; the
failure it prevents is a leak no metric would reveal, because a model trained
across a gap looks *better*, not worse.

**The purge is structural here, not a separate step.** A window occupies
``[s, s+L+H)`` and its target is the final ``H`` bars. Enumerating only windows
that lie wholly inside a span therefore guarantees the last target ends exactly
at the span boundary — which is what root §8.2 asks for at *both* boundaries,
train→validation and train→test (`D24`). There is no separate purge to forget.
"""

from __future__ import annotations

from datetime import datetime

import polars as pl

from itransformer_btc.config import PRED_LEN, SEQ_LEN
from itransformer_btc.segments import HOUR_MS, Segment, build_segments, usable_mask


def enumerate_windows(
    frame: pl.DataFrame,
    start: datetime | None = None,
    end: datetime | None = None,
    seq_len: int = SEQ_LEN,
    pred_len: int = PRED_LEN,
) -> list[int]:
    """Every valid window start inside ``[start, end)``, as epoch ms.

    Windows are built *inside* segments and never across them, so no window can
    span a break. Enumerating inside ``[start, end)`` also applies the purge:
    the last window's target ends at ``end``, never past it.

    Args:
        frame: Bars carrying ``ts_ms``; ``usable`` is derived if absent.
        start: Inclusive lower bound of the span.
        end: Exclusive upper bound of the span.
        seq_len: Lookback ``L``.
        pred_len: Horizon ``H``.

    Returns:
        Window-start timestamps in ascending order.

    Raises:
        ValueError: If any emitted window fails the timestamp identity. That is
            an assertion about the segment builder, not about the data, so a
            failure means the pipeline is broken rather than the market.
    """
    span = seq_len + pred_len
    segments = build_segments(frame, start, end)
    if not segments:
        return []

    rows = (frame if "usable" in frame.columns else usable_mask(frame)).filter(
        pl.col("usable")
    )
    if start is not None:
        rows = rows.filter(pl.col("ts_ms") >= int(start.timestamp() * 1000))
    if end is not None:
        rows = rows.filter(pl.col("ts_ms") < int(end.timestamp() * 1000))
    ts = rows.get_column("ts_ms").to_list()

    starts: list[int] = []
    for seg in segments:
        for s in range(seg.start_row, seg.end_row - span + 1):
            last = s + span - 1
            if ts[last] - ts[s] != (span - 1) * HOUR_MS:
                raise ValueError(
                    f"window at ts={ts[s]} spans a break: "
                    f"t[{last}] - t[{s}] = {ts[last] - ts[s]} ms, expected "
                    f"{(span - 1) * HOUR_MS} ms. The segment builder is wrong; "
                    f"do not relax this check."
                )
            starts.append(ts[s])
    return starts


def count_windows(
    segments: list[Segment],
    seq_len: int = SEQ_LEN,
    pred_len: int = PRED_LEN,
) -> int:
    """Total window starts across segments — the measured truth.

    Uses ``max(0, n - span + 1)`` per segment, so a segment shorter than one
    window contributes nothing rather than a negative count. Root §4.3's closed
    form ``(bars - 119) - [119 x breaks + missing]`` is algebraically identical
    to this **only while every segment clears the span**; where one does not,
    this is right and the closed form is not.
    """
    span = seq_len + pred_len
    return sum(seg.window_starts(span) for seg in segments)
