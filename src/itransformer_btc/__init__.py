"""Data plane and experiment scaffolding for the spot-only iTransformer study.

Root ``CLAUDE.md`` is the project law, and it is the *only* one: the
subdirectory files were deleted on 2026-08-06 because a rule that loads solely
when a file in its subtree is opened is absent exactly when an agent reasons
about the area without opening one. Two of those rules shape every module here:

* **polars only.** Its rolling API is backward-closed by construction, so the
  ``center=True`` leak class is unrepresentable. Stage 1 ingest
  (``spot_klines_btc.py``, at the repository root) is the one documented
  exemption and lives outside this package.
* **Fail loudly.** A schema mismatch, a window-count mismatch or a hash mismatch
  raises. The anti-leakage checklist is ``assert``s wherever it can be, because
  a checklist that lives only in prose is a checklist nobody runs.
"""

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
