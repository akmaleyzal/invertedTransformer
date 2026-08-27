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

import hashlib
import json
import os
import random
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import polars as pl
import torch
from torch import Tensor, nn

from itransformer_btc.splits import OriginTensors, SplitTensors

ARTIFACTS: Path = Path("artifacts")

DEFAULT_PARQUET: Path = Path("data/raw/BTCUSDT_1h.parquet")

#: Environment variable naming the input artifact actually consumed.
#:
#: Root §12 forbids comparing numbers produced under different input-artifact
#: hashes, so every ``meta/*.json`` must name the vintage it read. The
#: repository-relative default is right locally and **wrong everywhere the grid
#: actually runs**: on Kaggle the artifact arrives as an attached Dataset under
#: ``/kaggle/input/<slug>/``, and root §10.5 forbids hard-coding that slug. With
#: the path unresolvable the digest logged as ``"unknown"`` on every Kaggle run,
#: which is §12 unenforceable at precisely the place the grid executes. The
#: launcher and the worker CLI both set this from the path they were handed.
INPUT_PARQUET_ENV: str = "ITBTC_PARQUET"

#: Digest supplied by a launcher that has no package files to hash.
#:
#: :func:`code_sha256` normally hashes ``*.py`` beside this module, which needs
#: a ``__file__`` — and a notebook that carries the package as **plain
#: definition cells** rather than materialised files has none. The generator
#: computes the identical digest from ``src/itransformer_btc/`` and sets this,
#: so the number in ``meta/*.json`` still names the code that ran (root §12,
#: `D54b`) and still matches the digest a local checkout of the same source
#: produces. Left ``None`` in every file-based context, where the real hash is
#: strictly better because nothing has to be told to keep it honest.
CODE_SHA256_OVERRIDE: str | None = None


#: Serialises the seeded prologue of a run — seeding, then building the module.
#:
#: Only the prologue. Everything after it draws from the **device's own** CUDA
#: generator, which :func:`set_seed` scopes per device, so two workers pinned to
#: two GPUs never touch each other's stream. The prologue is milliseconds against
#: a ~32 s run, so holding a lock across it costs no measurable parallelism and
#: buys back the one thing run-level parallelism would otherwise destroy: the
#: bit-exact reproducibility `D62d` demonstrated and root §12 requires (`D68`).
SEED_LOCK = threading.Lock()


def set_seed(seed: int, device: torch.device | None = None) -> None:
    """Seed every source of nondeterminism root §16 names, scoped to one device.

    ``cudnn.deterministic`` costs throughput and is set anyway: a run that
    cannot be reproduced cannot enter the manuscript (root §12).

    **``device`` is what makes two GPUs safe (`D68`).** ``torch.manual_seed``
    reseeds the CPU generator *and every CUDA device*, so with two workers running
    concurrently one worker's seeding would reset the other's stream mid-training
    and neither run would reproduce. Given a device, this seeds the CPU generator
    and **only that device's** generator, leaving the other worker's untouched.

    Single-device behaviour is unchanged, which is what keeps the 894 completed
    runs reproducible: with one device in use, seeding it alone and seeding all of
    them set the same generator to the same value.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    # ``torch.manual_seed`` would fan out to every CUDA device; the default
    # generator is the CPU one and nothing else.
    torch.default_generator.manual_seed(seed)
    if torch.cuda.is_available():
        if device is not None and torch.device(device).type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.manual_seed(seed)
        else:
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


@dataclass(frozen=True, slots=True)
class TrainSchedule:
    """Root §6.2's optimisation budget, as data rather than as call-site defaults.

    The defaults reproduce the 684-run grid exactly, and that is the point.
    `D47` fixed ``lr_halve_every`` at 4 because per-epoch halving reaches ~4e-7
    by epoch 9, which makes both the 30-epoch budget and the patience-5 stop
    decorative.

    Measured afterwards, the grid early-stopped at a **mean of 10.49 epochs and
    never once reached the cap** across 444 iTransformer runs, maximum 26. So the
    binding constraint was the schedule, not the budget: by epoch 26 the learning
    rate is ~1.6e-6 and by epoch 30 ~7.8e-7, and raising ``max_epochs`` alone
    would have been a no-op. That is why `D62c`'s robustness arm widens the
    schedule instead, and why this exists as an overridable object rather than as
    four numbers frozen into a signature.
    """

    max_epochs: int = 30
    patience: int = 5
    lr: float = 1e-4
    lr_halve_every: int = 4


class Architecture(Protocol):
    """What the trainer, the runner and the artifact writer need of a config.

    :func:`write_artifacts` is the **only** definition of the ``meta/*.json``
    schema — root §12's traceability contract expressed as code — and it was
    bound to ``ITransformer``/``ITransformerConfig`` until the §7 baselines
    arrived (`D56`). A second writer in ``baselines.py`` would have made two
    definitions of one contract, which is the drift surface `D54d` exists to
    prevent, so the writer was widened to this protocol instead of copied.

    **Everything here is a method, never a field, and that is load-bearing.**
    ``write_artifacts`` records ``asdict(cfg)``, so a field added merely to steer
    dispatch would appear in every iTransformer ``meta/*.json`` and change bytes
    the study has already produced. Methods are invisible to
    :func:`dataclasses.asdict`; fields are not.
    """

    pred_len: int

    def build(self) -> nn.Module:
        """A fresh, untrained module for this configuration."""

    def loss_target(self) -> str:
        """``"target"`` or ``"all"`` — which target tensor the loss reads.

        See :class:`itransformer_btc.splits.SplitTensors`: the ladder is
        single-channel by `D39`, the channel-independent baselines are
        all-channel by their own published objective, and that difference is what
        makes their K label mean anything.
        """

    def fit(
        self,
        tensors: OriginTensors,
        spec: RunSpec,
        *,
        device: torch.device | None = None,
    ) -> tuple[nn.Module, "Architecture", TrainOutcome]:
        """Fit one cell: the model, the **resolved** config, and the outcome.

        The config comes back because selection happens for one model and not the
        others. Every iTransformer hyperparameter is fixed a priori and identical
        at every rung (`D38`), so its ``fit`` returns what it was given; ridge's
        alpha is chosen on the validation sub-block, and with ARIMA outside the
        minimal set that is the **only** hyperparameter selected anywhere in this
        study (root §11). A chosen value that never reached ``meta['config']``
        would be a number the manuscript could not regenerate.
        """


class Forecaster(Protocol):
    """What the artifact writer needs of a fitted model."""

    cfg: Architecture

    def eval(self) -> "Forecaster":
        """Inference mode — dropout off."""

    def forecast_target(self, x: Tensor) -> Tensor:
        """``(B, L, K) -> (B, H)`` on the target channel: what ``preds/`` holds."""


def _to_device(
    split: SplitTensors, device: torch.device, *, target: str = "target"
) -> tuple[Tensor, Tensor]:
    """Move one split's inputs and its loss target to the device.

    ``target`` selects the width, not the content: ``"target"`` is the ``r``
    channel the ladder is scored on, ``"all"`` is every channel, which is what a
    channel-independent baseline's published objective supervises.
    """
    y = split.y if target == "target" else split.y_all
    return (
        torch.from_numpy(split.x).to(device, non_blocking=True),
        torch.from_numpy(y).to(device, non_blocking=True),
    )


@torch.no_grad()
def _mean_loss(model: nn.Module, x: Tensor, y: Tensor, batch: int = 512) -> float:
    """Mean MSE over a split, batched to bound peak memory rather than for speed.

    The divisor is every element of one sample's target, so this is the mean over
    ``(H,)`` for a single-channel target and over ``(H, N)`` for an all-channel
    one — the same quantity ``mse_loss`` returns, computed in pieces.
    """
    if len(x) == 0:
        return float("nan")
    model.eval()
    total = 0.0
    for i in range(0, len(x), batch):
        total += nn.functional.mse_loss(
            model(x[i : i + batch]), y[i : i + batch], reduction="sum"
        ).item()
    return total / (len(x) * int(np.prod(y.shape[1:])))


@torch.no_grad()
def predict(model: Forecaster, x: Tensor, batch: int = 512) -> np.ndarray:
    """The target channel's H-step forecasts for every window, batched.

    Routed through ``forecast_target`` rather than ``__call__`` because a
    channel-independent baseline trained on its published all-channel objective
    returns ``(B, H, N)`` from ``forward`` while ``preds/`` holds one channel
    (root §10.4). Sniffing the rank of the output instead would leave the
    prediction file's meaning resting on a shape no model ever declared.
    """
    model.eval()
    if len(x) == 0:
        return np.empty((0, model.cfg.pred_len), np.float32)
    return np.concatenate(
        [
            model.forecast_target(x[i : i + batch]).cpu().numpy()
            for i in range(0, len(x), batch)
        ]
    )


def train_one(
    tensors: OriginTensors,
    spec: RunSpec,
    cfg: Architecture,
    *,
    device: torch.device | None = None,
    batch_size: int = 32,
    max_epochs: int | None = None,
    patience: int | None = None,
    lr: float | None = None,
    lr_halve_every: int | None = None,
) -> tuple[nn.Module, TrainOutcome]:
    """Train one (origin, K, seed) cell and return the best-validation model.

    The learning rate halves every **four** epochs, not every epoch (`D47`):
    per-epoch halving reaches ~4e-7 by epoch 9, which makes the 30-epoch budget
    and the patience-5 early stop decorative — the model stops moving long
    before either can bind.

    ``epochs_run`` and the final training loss come back because root §6.2
    requires them logged per rung: they are how a reader tells a flat 8->12 rung
    from an under-trained one.

    **One trainer serves every gradient model in the study**, iTransformer and
    the §7 baselines alike (`D56`). The schedule, the patience and the early-stop
    rule are root §6.2's, and duplicating them per model would be a second
    definition of the training protocol — the same drift `D54d` names for the
    artifact schema. The config supplies only what genuinely differs: the module
    to build, and the width of the target its loss reads.
    """
    device = device or pick_device()

    # The schedule comes from the config when it declares one, so an arm can widen
    # it without a new trainer (`D62c`). ``hasattr`` rather than a Protocol member:
    # ridge is a solve with no epochs at all, and DLinear and PatchTST take root
    # §6.2's schedule unchanged, so requiring the method would add a stub to three
    # classes to say "nothing special". Explicit arguments still win, which is what
    # keeps ``stage5_pilot`` and the tests able to shorten a run.
    training_schedule = cfg.schedule() if hasattr(cfg, "schedule") else TrainSchedule()
    max_epochs = training_schedule.max_epochs if max_epochs is None else max_epochs
    patience = training_schedule.patience if patience is None else patience
    lr = training_schedule.lr if lr is None else lr
    lr_halve_every = (
        training_schedule.lr_halve_every if lr_halve_every is None else lr_halve_every
    )

    # Seeding and construction together, under one lock: both draw from the CPU
    # generator, which every worker shares whatever device it owns (`D68`).
    with SEED_LOCK:
        set_seed(spec.seed, device)
        model = cfg.build().to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    schedule = torch.optim.lr_scheduler.StepLR(
        optimiser, step_size=lr_halve_every, gamma=0.5
    )

    # The whole split, resident. Index-slicing it is the entire batching
    # strategy; shuffling permutes an index tensor on device, never the data.
    loss_target = cfg.loss_target()
    x_tr, y_tr = _to_device(tensors.train, device, target=loss_target)
    x_va, y_va = _to_device(tensors.val, device, target=loss_target)

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
            # For the ladder this is MSE on the target channel only, at every
            # rung (`D39`): `ITransformerConfig.loss_target` is the constant
            # "target" and `ITransformer.forward` returns that channel alone, so
            # it cannot drift to the all-channel loss the reference
            # implementation defaults to — which would make K=12 a 12-task
            # problem and K=1 a 1-task one, varying supervision with the study's
            # own independent variable. A channel-independent baseline says
            # "all" and gets the all-channel objective it is published with.
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
    model: Forecaster, x: Tensor, y: Tensor, c: float = 100.0
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


def code_sha256() -> str:
    """Digest of this package's own source — the git sha's stand-in off-repo.

    Root §12 asks a run to name the code that produced it, and names the git sha
    as the way. **There is no git repository on Kaggle**, so the sha logs as
    ``"unknown"`` there and the traceability contract loses its code half exactly
    where the grid runs. Hashing the package source answers the same question and
    answers it better: it identifies the code that ran, not the commit someone
    happened to be standing on with a dirty tree.

    Line endings are normalised, so a CRLF checkout on Windows and the LF copy a
    notebook materialises give the **same** digest for identical logic. Without
    that, every Kaggle run would appear to be a different code vintage from the
    local run of the same commit — a false positive on the one check §12 exists
    to make possible.

    ``CODE_SHA256_OVERRIDE`` short-circuits this where there are no files to
    hash — a notebook carrying the package as plain definition cells. The
    ``__file__`` lookup below sits *after* that check on purpose: in such a
    launcher there is no module file at all, so reaching it would raise rather
    than return the digest the traceability contract asks for.
    """
    if CODE_SHA256_OVERRIDE is not None:
        return CODE_SHA256_OVERRIDE
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def resolve_input_parquet(parquet: Path | str | None = None) -> Path:
    """The input artifact this process consumed: argument, then env, then default."""
    if parquet is not None:
        return Path(parquet)
    from_env = os.environ.get(INPUT_PARQUET_ENV)
    return Path(from_env) if from_env else DEFAULT_PARQUET


def _input_sha256(parquet: Path | str | None = None) -> tuple[str, str]:
    """``(digest, provenance)`` for the parquet actually read.

    The Stage 1 report sitting beside the artifact is preferred, because that
    digest was written by the ingest script over the bytes it emitted and is the
    figure root §4.1 pins. Where the report did not travel — an attached Kaggle
    Dataset holding the parquet alone — the file is hashed directly, which yields
    the same number by construction.

    Returns ``("unknown", "unresolved")`` rather than raising: a run that cannot
    name its input is a documented failure under §12, and failing the run instead
    would lose 90 s of GPU time to a bookkeeping problem.
    """
    path = resolve_input_parquet(parquet)
    report = path.with_name(f"{path.stem}_report.json")
    try:
        return json.loads(report.read_text())["artifact_sha256"][path.name], "report"
    except Exception:
        pass
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest(), "file-digest"
    except Exception:
        return "unknown", "unresolved"


def write_artifacts(
    model: Forecaster,
    tensors: OriginTensors,
    spec: RunSpec,
    cfg: Architecture,
    outcome: TrainOutcome,
    device: torch.device,
    root: Path = ARTIFACTS,
    attention: "pl.DataFrame | None" = None,
) -> tuple[Path, Path]:
    """Write ``preds/{run_id}.parquet`` and ``meta/{run_id}.json``.

    A run is complete **only when both files exist and ``status == "complete"``**
    (root §10.5). Anything else is re-run from scratch; intra-run checkpointing
    is deliberately omitted, since at ~30 s per run measured (`D57`) it costs far
    more complexity than it saves.

    **This function is the schema.** Root §12 admits no number into the
    manuscript that does not resolve to a prediction file, a config hash and a
    documented decision, and these two files are where all three live. Every
    model in the study — the ladder, ridge, DLinear, PatchTST — writes through
    here, typed against :class:`Forecaster` and :class:`Architecture` rather than
    against one model, so there is exactly one place the contract can be read and
    exactly one place it can be broken.
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

    if attention is not None:
        # Figure 5's input (`D62d`). A third directory rather than a column on
        # ``preds``: the maps are one row per (tercile, layer, variate pair) and
        # the forecasts are one row per (block, timestamp, step), so joining them
        # into one file would mean padding one of the two with nulls. Completeness
        # still keys off ``preds`` and ``meta`` alone, so an arm that writes no map
        # is complete without one.
        (root / "attn").mkdir(parents=True, exist_ok=True)
        attention.write_parquet(root / "attn" / f"{spec.run_id}.parquet")

    input_parquet = resolve_input_parquet()
    input_digest, input_provenance = _input_sha256(input_parquet)
    meta = {
        "run_id": spec.run_id,
        "spec": asdict(spec),
        "config": asdict(cfg),
        # Recorded separately because a schedule override is a **method**, and
        # ``asdict`` sees fields only (`D62c`). ``LongScheduleConfig`` adds no
        # field, so its ``config`` block is byte-identical to the main arm's and
        # this key is the only place the difference is visible --- besides the
        # ``run_id`` tag, which is what keeps the two from colliding on disk.
        "schedule": (
            asdict(cfg.schedule()) if hasattr(cfg, "schedule") else None
        ),
        "origin": tensors.origin.label,
        "origin_index": tensors.origin.index,
        "block_labels": list(tensors.block_labels),
        "k": tensors.k,
        "variates": list(tensors.scaler.columns),
        "git_sha": _git_sha(),
        # Root §12's code half. `git_sha` is "unknown" off-repo, which is every
        # Kaggle session; `code_sha256` answers the same question there.
        "code_sha256": code_sha256(),
        "input_parquet": str(input_parquet),
        "input_sha256": input_digest,
        "input_sha256_source": input_provenance,
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
