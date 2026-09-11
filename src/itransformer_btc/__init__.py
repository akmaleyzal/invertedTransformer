"""Generated projection of notebooks/iTransformer.ipynb.

Edit the notebook and export only through its final cell. CLAUDE.md governs the
study. Polars is the data-plane implementation choice; it supports centered
rolling windows. Safety comes from per-bar features, chronology tests and purged
splits, not from an alleged limitation of the library."""

from __future__ import annotations

from itransformer_btc.config import (
    ORIGINS,
    PRED_LEN,
    SEQ_LEN,
    STARTS_LOST_PER_BREAK,
    WINDOW_SPAN,
    Origin,
    origin_grid,
)

__all__ = [
    "ORIGINS",
    "Origin",
    "PRED_LEN",
    "SEQ_LEN",
    "STARTS_LOST_PER_BREAK",
    "WINDOW_SPAN",
    "origin_grid",
]
