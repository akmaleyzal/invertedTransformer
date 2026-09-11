"""Per-origin splits, the scaler, and the tensors the training loop slices.

Three things live here because they are one decision: which windows exist, what
standardises them, and how they reach the device.

**Window semantics differ by split, and the difference is 119 windows (`D51`).**
A *training* or *validation* window must lie wholly inside its span — its H-step
target may not cross the boundary, which is the purge at both boundaries (root
§8.2 / `D24`). A *test* window may not: root §8.3 states that its 96-bar
lookback reaching back across the boundary is past information legitimately
available to a forecaster, and that blocking it would make the evaluation
unrealistically pessimistic. Every hour of a test block is an admissible
forecast origin.

**The scaler is fitted on the 21-month sub-block and nothing else**, at every
origin. Moving ``train_end`` is a leak, not a mismatch (root §8.2).

Upstream
--------
**Both the evaluation protocol and the scaler are written here. Neither is a
library call, and the scaler in particular is not scikit-learn's.**

- Rolling-origin evaluation -- L. J. Tashman, "Out-of-sample tests of
  forecasting accuracy: An analysis and review," *Int. J. Forecast.*, vol. 16,
  no. 4, pp. 437-450, 2000; C. Bergmeir and J. M. Benitez, "On the use of
  cross-validation for time series predictor evaluation," *Information
  Sciences*, vol. 191, pp. 192-213, 2012 -- the primary justification for
  rolling-origin over a fixed origin.
- Purging -- M. Lopez de Prado, *Advances in Financial Machine Learning*.
  Hoboken, NJ: Wiley, 2018, ch. 7. Adopted; **embargo and CPCV deliberately are
  not**, and root §8.3 and §8.4 carry the written arguments rather than leaving
  a protocol element silently absent (`D15`). The purge runs at **both**
  boundaries -- train/validation as well as train/test -- which the source is
  not read as requiring and which `D24` shows is the one that governs model
  selection.
- ``Scaler`` -- **not** ``sklearn.preprocessing.StandardScaler``, though it
  computes the same thing. Under ``use_norm=True`` the outer affine scaler
  cancels algebraically (root §6.3), so what it actually controls is the
  reporting scale and the baselines that have no internal normalisation;
  writing it here keeps that fitted object inside the per-origin tensor build
  rather than taking a dependency for two lines of arithmetic.

:data:`itransformer_btc.config.SOURCE_PROVENANCE` carries these rows in full.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import numpy as np
import polars as pl

from itransformer_btc.config import (
    PRED_LEN,
    SEQ_LEN,
    WINDOW_SPAN,
    OriginLike,
)
from itransformer_btc.features import TARGET_INDEX, ladder_columns
from itransformer_btc.segments import HOUR_MS
import hashlib

Semantics = Literal["contained", "origin"]


def window_starts(
    ts: np.ndarray, start: datetime, end: datetime, semantics: Semantics,
    span: int = WINDOW_SPAN, *, seq_len: int = SEQ_LEN,
) -> np.ndarray:
    """Input-start indices; test membership uses the FIRST TARGET timestamp (A01).

    A forecast at t consumes [t-L, t) and predicts [t, t+H). Training and
    validation keep their complete windows inside their respective spans.
    """
    if semantics not in ("contained", "origin") or span < 2:
        raise ValueError("invalid window semantics or span")
    if semantics == "origin" and not 0 < seq_len < span:
        raise ValueError("origin semantics require 0 < seq_len < span")
    ts = np.asarray(ts)
    if ts.ndim != 1 or np.any(np.diff(ts) <= 0) or np.any(ts % HOUR_MS != 0):
        raise ValueError("timestamps must be unique, increasing UTC hour boundaries")
    lo, hi = int(start.timestamp() * 1000), int(end.timestamp() * 1000)
    if hi <= lo:
        raise ValueError("window end must follow start")
    first = np.arange(max(0, len(ts) - span + 1), dtype=np.int64)
    contiguous = (ts[first + span - 1] - ts[first]) == (span - 1) * HOUR_MS
    if semantics == "contained":
        inside = (ts[first] >= lo) & (ts[first + span - 1] < hi)
    else:
        issued = ts[first + seq_len]
        inside = (issued >= lo) & (issued < hi)
    return first[contiguous & inside]


@dataclass(frozen=True, slots=True)
class Scaler:
    """Per-channel standardiser fitted on the training sub-block only.

    Root §6.3: the outer affine scaler **cancels algebraically** under instance
    normalisation, because ``(z - m)/s`` recovers ``(x - mean_t)/std_t`` with
    ``mu_g`` and ``sigma_g`` dropping out. What it still controls is the
    *reporting scale* of every metric, and learning for the baselines that have
    no internal normalisation. StandardScaler is chosen for literature
    comparability, inertness under ``use_norm=True``, and cross-model
    consistency — not because it changes what the transformer learns.
    """

    mean: np.ndarray
    std: np.ndarray
    columns: tuple[str, ...]

    @classmethod
    def fit(cls, values: np.ndarray, columns: list[str]) -> "Scaler":
        std = values.std(axis=0, ddof=0)
        if not np.all(np.isfinite(std)) or np.any(std <= 0):
            raise ValueError(
                f"degenerate channel std in the training sub-block: "
                f"{dict(zip(columns, std))}"
            )
        return cls(values.mean(axis=0), std, tuple(columns))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std

    @property
    def target_mu_over_sigma(self) -> float:
        """``mu_g / sigma_g`` on the target channel — the Naive-RW offset.

        Root §7 / `D31`: a random walk in price implies ``y_raw = 0``, but the
        metrics live on standardised returns, so ``y_z = 0`` would silently mean
        ``r_hat = mu_g``, the training-window mean hourly return — a constant
        drift model wearing the EMH baseline's name. The baseline is mapped as
        ``y_z = -mu_g / sigma_g`` instead, and this value is logged per origin
        so the size of the tilt is auditable.
        """
        return float(self.mean[TARGET_INDEX] / self.std[TARGET_INDEX])


@dataclass(frozen=True, slots=True)
class SplitTensors:
    """Inputs, target returns, all-channel targets and first-target timestamps. The default objective consumes y; all-channel sensitivity consumes y_all. Every output artifact predicts the return channel."""

    x: np.ndarray      # (n, L, K) float32, standardised
    y: np.ndarray      # (n, H)    float32, standardised target channel
    y_all: np.ndarray  # (n, H, K) float32, every channel's H-step target
    ts: np.ndarray     # (n,) int64, forecast origin = first target bar open (UTC)

    def __len__(self) -> int:
        return len(self.ts)


@dataclass(frozen=True, slots=True)
class OriginTensors:
    """Everything one training run consumes, already standardised."""

    origin: OriginLike
    k: int
    scaler: Scaler
    train: SplitTensors
    val: SplitTensors
    test_blocks: tuple[SplitTensors, ...]
    #: One-indexed block label per entry of ``test_blocks``. ``(1,…,6)`` for a
    #: normal origin, ``(4, 5, 6)`` for the falsification arm — which is why the
    #: label is stored rather than recovered from position.
    block_labels: tuple[int, ...]
    training_selection: dict | None = None
    representation: dict | None = None

    @property
    def naive_rw_z(self) -> float:
        """The Naive-RW prediction in scaler space (`D31`)."""
        return -self.scaler.target_mu_over_sigma


def _gather(
    values: np.ndarray, starts: np.ndarray, ts: np.ndarray, seq_len: int, pred_len: int
) -> SplitTensors:
    """Slice windows out of a standardised array by index arithmetic.

    No ``Dataset``, no ``DataLoader``, no per-item Python. Root §10.3: at ~280k
    parameters the run is dominated by data movement and interpreter overhead,
    which a per-item loader maximises — the naive path costs roughly 10x and
    puts the grid outside the weekly GPU quota outright.
    """
    if len(starts) == 0:
        return SplitTensors(
            x=np.empty((0, seq_len, values.shape[1]), np.float32),
            y=np.empty((0, pred_len), np.float32),
            y_all=np.empty((0, pred_len, values.shape[1]), np.float32),
            ts=np.empty(0, np.int64),
        )
    rows = starts[:, None] + np.arange(seq_len)[None, :]
    tgt = starts[:, None] + seq_len + np.arange(pred_len)[None, :]
    targets = values[tgt].astype(np.float32, copy=False)
    return SplitTensors(
        x=values[rows].astype(np.float32, copy=False),
        # Copied out rather than left as a strided view of `targets`: `y` is what
        # the whole pipeline reads, and a non-contiguous array of it would make
        # every downstream `from_numpy` and `reshape` behave differently for a
        # reason nobody would think to look for.
        y=np.ascontiguousarray(targets[:, :, TARGET_INDEX]),
        y_all=targets,
        ts=ts[starts + seq_len],
    )


def build_origin_tensors(
    features: pl.DataFrame,
    origin: OriginLike,
    k: int,
    seq_len: int = SEQ_LEN,
    pred_len: int = PRED_LEN,
    columns: tuple[str, ...] | None = None,
    train_window_limit: int | None = None,
    selection_seed: int = 1729,
    representation: str = "identity",
) -> OriginTensors:
    """Build every split for one (origin, K) cell.

    The scaler is fitted on the **rows** of the 21-month sub-block, before any
    window is cut, and then applied to every split. Fitting it on validation or
    test rows is the leak root §11 calls fatal.

    Raises:
        ValueError: If the training split is empty, or if the last training
            window's target reaches at or past ``val_start`` — the purge
            assertion (`D24`), checked here rather than trusted.
    """
    # Named columns override the rung; the historical matched-K arm changes
    # feature identity as well as PR. ``k`` still has to agree with the set:
    # it is what the model is built with, and a mismatch would be silent.
    columns = tuple(columns) if columns else ladder_columns(k)
    if len(columns) != k:
        raise ValueError(f"{len(columns)} columns for K={k}")
    ts = features.get_column("ts_ms").to_numpy()
    values = features.select(columns).to_numpy()
    span = seq_len + pred_len

    train_idx = window_starts(ts, origin.train_start, origin.train_sub_end,
                              "contained", span)
    val_idx = window_starts(ts, origin.val_start, origin.val_end, "contained", span)
    if len(train_idx) == 0:
        raise ValueError(f"origin {origin.label}: empty training split")

    last_train_target = ts[train_idx[-1] + span - 1]
    if last_train_target >= int(origin.val_start.timestamp() * 1000):
        raise ValueError(
            f"origin {origin.label}: a training target reaches into validation "
            f"({last_train_target}); the purge did not hold"
        )

    scaler = Scaler.fit(values[train_idx[0] : train_idx[-1] + span], columns)
    scaled = scaler.transform(values)
    n_available = len(train_idx)
    fit_rows = scaled[train_idx[0]:train_idx[-1] + span].astype(np.float64)
    representation_meta = None
    if representation != "identity":
        if k != 8 or representation not in ("repr_identity", "whiten", "correlate"):
            raise ValueError("representation controls require the same K=8 base columns")
        covariance = np.cov(fit_rows[:, 1:], rowvar=False, ddof=0)
        eig, vectors = np.linalg.eigh(covariance)
        floor = max(float(eig.max()) * 1e-6, 1e-8)
        transform = np.eye(k)
        if representation != "repr_identity":
            transform[1:, 1:] = (vectors * (1. / np.sqrt(np.maximum(eig, floor)))) @ vectors.T
        if representation == "correlate":
            corr = .95 * np.ones((k-1, k-1)) + .05 * np.eye(k-1)
            ev, q = np.linalg.eigh(corr)
            transform[1:, 1:] = transform[1:, 1:] @ ((q * np.sqrt(ev)) @ q.T)
        if np.linalg.matrix_rank(transform) != k or np.linalg.cond(transform) > 1e6:
            raise ValueError("representation is not safely invertible")
        represented = fit_rows @ transform
        eigenvalues = np.linalg.eigvalsh(np.corrcoef(represented, rowvar=False))
        representation_meta = {
            "name": representation, "matrix": transform.tolist(),
            "inverse": np.linalg.inv(transform).tolist(),
            "condition_number": float(np.linalg.cond(transform)),
            "training_pr": float(eigenvalues.sum()**2 / (eigenvalues**2).sum()),
            "fit_scope": "purged training rows only", "target_preserved": True,
            "use_norm": False, "eigenvalue_floor": floor,
        }
        scaled = (scaled @ transform).astype(np.float32)
    if train_window_limit is not None:
        if train_window_limit < 1 or n_available < train_window_limit:
            raise ValueError(f"{origin.label}: {n_available} training windows < required {train_window_limit}")
        selected = np.random.default_rng(selection_seed).choice(n_available, train_window_limit, replace=False)
        train_idx = train_idx[np.sort(selected)]
    selection = {"available": n_available, "selected": len(train_idx),
                 "limit": train_window_limit, "seed": selection_seed,
                 "forecast_times_sha256": hashlib.sha256(ts[train_idx + seq_len].tobytes()).hexdigest(),
                 "scaler_fit": "all purged training rows before subsampling"}

    blocks = origin.blocks()
    return OriginTensors(
        origin=origin,
        k=k,
        scaler=scaler,
        train=_gather(scaled, train_idx, ts, seq_len, pred_len),
        val=_gather(scaled, val_idx, ts, seq_len, pred_len),
        test_blocks=tuple(
            _gather(scaled, window_starts(ts, lo, hi, "origin", span, seq_len=seq_len),
                    ts, seq_len, pred_len)
            for _, lo, hi in blocks
        ),
        block_labels=tuple(label for label, _, _ in blocks),
        training_selection=selection, representation=representation_meta,
    )
