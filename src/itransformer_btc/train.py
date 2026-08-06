"""Training loop, run identity, and the two files every run must leave behind.

Root §10.3's regime is the load-bearing part: **load the whole split to the
device once, then batch by index-slicing that tensor.** No ``Dataset``, no
``DataLoader``, no workers. At ~280k parameters the compute is trivial and the
run is dominated entirely by data movement and Python overhead, which a per-item
loader maximises — the naive path costs roughly 10x and puts the 837-run grid
outside the 30 h weekly quota outright.

Root §10.4: **persist raw predictions, always.** They are required for the
Diebold-Mariano test, the per-regime analysis and the economic evaluation.
Re-running the grid because only metrics were saved is an expensive, avoidable
mistake.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch import Tensor, nn

from itransformer_btc.model import ITransformer, ITransformerConfig
from itransformer_btc.splits import OriginTensors, SplitTensors

ARTIFACTS: Path = Path("artifacts")


def set_seed(seed: int) -> None:
    """Seed every source of nondeterminism root §16 names.

    ``cudnn.deterministic`` costs throughput and is set anyway: a run that
    cannot be reproduced cannot enter the manuscript (root §12).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pick_device() -> torch.device:
    """Prefer CUDA; fall back to CPU."""
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def supports_native_bf16(device: torch.device) -> bool:
    """True only on sm_80+.

    Never gate on ``torch.cuda.is_bf16_supported()``: it defaults to
    ``including_emulation=True`` and returns **True** on a T4 (sm_75), selecting
    an emulated bf16 path slower than fp32 (root §10.3).
    """
    if device.type != "cuda":
        return False
    return torch.cuda.get_device_capability(device.index or 0)[0] >= 8


@dataclass(frozen=True, slots=True)
class RunSpec:
    """Deterministic, human-readable run identity (root §10.4).

    Changing any component deliberately **orphans** prior outputs rather than
    silently reusing a mismatched result.
    """

    model: str
    origin_index: int
    k: int
    pred_len: int
    seed: int

    @property
    def run_id(self) -> str:
        return (
            f"{self.model}_o{self.origin_index:02d}_K{self.k:02d}"
            f"_H{self.pred_len:03d}_s{self.seed}"
        )


@dataclass(frozen=True, slots=True)
class TrainOutcome:
    """What one completed run produced, beyond its two artifacts."""

    run_id: str
    epochs_run: int
    best_val_mse: float
    train_loss: float
    wall_time_s: float
    n_parameters: int
    device: str


def _to_device(split: SplitTensors, device: torch.device) -> tuple[Tensor, Tensor]:
    return (
        torch.from_numpy(split.x).to(device, non_blocking=True),
        torch.from_numpy(split.y).to(device, non_blocking=True),
    )


@torch.no_grad()
def _mean_loss(model: nn.Module, x: Tensor, y: Tensor, batch: int = 512) -> float:
    """Mean MSE over a split, batched to bound peak memory rather than for speed."""
    if len(x) == 0:
        return float("nan")
    model.eval()
    total = 0.0
    for i in range(0, len(x), batch):
        total += nn.functional.mse_loss(
            model(x[i : i + batch]), y[i : i + batch], reduction="sum"
        ).item()
    return total / (len(x) * y.shape[1])


@torch.no_grad()
def predict(model: ITransformer, x: Tensor, batch: int = 512) -> np.ndarray:
    model.eval()
    if len(x) == 0:
        return np.empty((0, model.cfg.pred_len), np.float32)
    return np.concatenate(
        [model(x[i : i + batch]).cpu().numpy() for i in range(0, len(x), batch)]
    )


def train_one(
    tensors: OriginTensors,
    spec: RunSpec,
    cfg: ITransformerConfig,
    *,
    device: torch.device | None = None,
    batch_size: int = 32,
    max_epochs: int = 30,
    patience: int = 5,
    lr: float = 1e-4,
    lr_halve_every: int = 4,
) -> tuple[ITransformer, TrainOutcome]:
    """Train one (origin, K, seed) cell and return the best-validation model.

    The learning rate halves every **four** epochs, not every epoch (`D47`):
    per-epoch halving reaches ~4e-7 by epoch 9, which makes the 30-epoch budget
    and the patience-5 early stop decorative — the model stops moving long
    before either can bind.

    ``epochs_run`` and the final training loss come back because root §6.2
    requires them logged per rung: they are how a reader tells a flat 8->12 rung
    from an under-trained one.
    """
    device = device or pick_device()
    set_seed(spec.seed)

    model = ITransformer(cfg).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    schedule = torch.optim.lr_scheduler.StepLR(
        optimiser, step_size=lr_halve_every, gamma=0.5
    )

    # The whole split, resident. Index-slicing it is the entire batching
    # strategy; shuffling permutes an index tensor on device, never the data.
    x_tr, y_tr = _to_device(tensors.train, device)
    x_va, y_va = _to_device(tensors.val, device)

    best_val = float("inf")
    best_state: dict[str, Tensor] | None = None
    epochs_run = 0
    train_loss = float("nan")
    stale = 0
    started = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        model.train()
        order = torch.randperm(len(x_tr), device=device)
        running = 0.0
        for i in range(0, len(order), batch_size):
            idx = order[i : i + batch_size]
            optimiser.zero_grad(set_to_none=True)
            # MSE on the target channel only, at every rung (`D39`). The model
            # returns that channel alone, so this cannot silently drift to the
            # all-channel loss the reference implementation defaults to.
            loss = nn.functional.mse_loss(model(x_tr[idx]), y_tr[idx])
            loss.backward()
            optimiser.step()
            running += loss.item() * len(idx)
        schedule.step()

        epochs_run = epoch
        train_loss = running / len(x_tr)
        val = _mean_loss(model, x_va, y_va)

        if val < best_val - 1e-9:
            best_val, stale = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, TrainOutcome(
        run_id=spec.run_id,
        epochs_run=epochs_run,
        best_val_mse=best_val,
        train_loss=train_loss,
        wall_time_s=time.perf_counter() - started,
        n_parameters=model.n_parameters(),
        device=str(device),
    )


def scale_invariance_check(
    model: ITransformer, x: Tensor, y: Tensor, c: float = 100.0
) -> tuple[float, float]:
    """Root §6.3's corrected ``use_norm`` invariant (`D03`).

    The source specification said to multiply the input by 100 and assert
    identical losses. That **cannot pass**: the target is a channel of the same
    array, so it scales too and the loss scales by ``c^2``. The invariant that
    does hold is ``MSE(c x) / c^2 == MSE(x)``.

    Returns:
        ``(MSE(x), MSE(c x) / c^2)`` — equal to floating-point tolerance while
        ``use_norm`` is active, and visibly unequal the moment it is not.
    """
    return _mean_loss(model, x, y), _mean_loss(model, x * c, y * c) / (c * c)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def _input_sha256(report: Path = Path("data/raw/BTCUSDT_1h_report.json")) -> str:
    try:
        return json.loads(report.read_text())["artifact_sha256"]["BTCUSDT_1h.parquet"]
    except Exception:
        return "unknown"


def write_artifacts(
    model: ITransformer,
    tensors: OriginTensors,
    spec: RunSpec,
    cfg: ITransformerConfig,
    outcome: TrainOutcome,
    device: torch.device,
    root: Path = ARTIFACTS,
) -> tuple[Path, Path]:
    """Write ``preds/{run_id}.parquet`` and ``meta/{run_id}.json``.

    A run is complete **only when both files exist and ``status == "complete"``**
    (root §10.5). Anything else is re-run from scratch; intra-run checkpointing
    is deliberately omitted, since at ~90 s per run it costs more complexity than
    it saves.
    """
    (root / "preds").mkdir(parents=True, exist_ok=True)
    (root / "meta").mkdir(parents=True, exist_ok=True)

    frames = []
    for b, split in zip(tensors.block_labels, tensors.test_blocks):
        # The label, not the position: the falsification arm's first tensor is
        # block 4, and re-indexing it to 1 would silently compare the fresh
        # model against the aged model's wrong blocks.
        if len(split) == 0:
            continue
        x, _ = _to_device(split, device)
        pred = predict(model, x)
        n, h = pred.shape
        frames.append(
            pl.DataFrame(
                {
                    "block": np.full(n * h, b, dtype=np.int8),
                    "step": np.tile(np.arange(1, h + 1, dtype=np.int16), n),
                    "timestamp": np.repeat(split.ts, h),
                    "y_true": split.y.reshape(-1),
                    "y_pred": pred.reshape(-1),
                }
            )
        )
    preds = (
        pl.concat(frames)
        if frames
        else pl.DataFrame(
            schema={
                "block": pl.Int8,
                "step": pl.Int16,
                "timestamp": pl.Int64,
                "y_true": pl.Float32,
                "y_pred": pl.Float32,
            }
        )
    )

    preds_path = root / "preds" / f"{spec.run_id}.parquet"
    meta_path = root / "meta" / f"{spec.run_id}.json"
    preds.write_parquet(preds_path)

    meta = {
        "run_id": spec.run_id,
        "spec": asdict(spec),
        "config": asdict(cfg),
        "origin": tensors.origin.label,
        "origin_index": tensors.origin.index,
        "block_labels": list(tensors.block_labels),
        "k": tensors.k,
        "variates": list(tensors.scaler.columns),
        "git_sha": _git_sha(),
        "input_sha256": _input_sha256(),
        "n_train": len(tensors.train),
        "n_val": len(tensors.val),
        "n_test_per_block": [len(s) for s in tensors.test_blocks],
        # Root §7 / `D31`: logged per origin so the drift tilt the Naive-RW
        # baseline carries in scaler space is auditable rather than assumed away.
        "mu_g": float(tensors.scaler.mean[0]),
        "sigma_g": float(tensors.scaler.std[0]),
        "mu_over_sigma": tensors.scaler.target_mu_over_sigma,
        "naive_rw_z": tensors.naive_rw_z,
        "epochs_run": outcome.epochs_run,
        "best_val_mse": outcome.best_val_mse,
        "train_loss": outcome.train_loss,
        "wall_time_s": outcome.wall_time_s,
        "n_parameters": outcome.n_parameters,
        "device": outcome.device,
        "status": "complete",
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    return preds_path, meta_path


def is_complete(run_id: str, root: Path = ARTIFACTS) -> bool:
    """Root §10.5's idempotence rule, as one function."""
    preds = root / "preds" / f"{run_id}.parquet"
    meta = root / "meta" / f"{run_id}.json"
    if not (preds.exists() and meta.exists()):
        return False
    try:
        return json.loads(meta.read_text()).get("status") == "complete"
    except Exception:
        return False
