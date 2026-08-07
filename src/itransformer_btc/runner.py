"""The run manifest, resume, the budget guard, and the two-GPU launcher.

Root §10. This module is what a Kaggle notebook calls; the notebook itself
installs, discovers inputs, calls :func:`launch_workers`, and saves. Logic in a
notebook is a defect (``notebooks/CLAUDE.md``), and a run queue is logic.

**Two GPUs are two independent run *processes*, not two threads and not
``nn.DataParallel``.** Root §10.3 rejects DataParallel on cost grounds — at
batch 32 the scatter/gather transfer costs more than the split saves, and
parallelism belongs at the *run* level because the grid is many small runs
rather than one large one. Threads are rejected for a second, sharper reason:
``torch.manual_seed`` seeds **every** CUDA device, so two threads seeding
concurrently would clobber each other's generator mid-run and root §12's
reproducibility contract would be unenforceable. One process per GPU with
``CUDA_VISIBLE_DEVICES`` pinned gives each worker its own global RNG, its own
interpreter lock and crash isolation, at the cost of rebuilding the feature
frame once per worker — seconds against hours.

**Work is split statically, by group.** A *group* is one ``(arm, origin, K, H)``
cell, whose seeds share a tensor build; shards take groups round-robin. Root
§10.5's idempotence rule makes this safe with no coordination: a run is complete
only when both artifacts exist and ``status == "complete"``, so a worker that
finishes early drains whatever is still pending regardless of which shard owned
it.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from itransformer_btc.config import (
    HORIZONS,
    K_LADDER,
    ORIGINS,
    PRED_LEN,
    SEEDS,
    SWEEP_ORIGIN_INDICES,
    FalsificationOrigin,
    OriginLike,
)
from itransformer_btc.model import ITransformerConfig
from itransformer_btc.splits import OriginTensors, build_origin_tensors
from itransformer_btc.train import (
    ARTIFACTS,
    DEFAULT_PARQUET,
    INPUT_PARQUET_ENV,
    RunSpec,
    is_complete,
    pick_device,
    train_one,
    write_artifacts,
)

#: Arm to the ``model`` component of ``run_id``. Distinct tags mean a changed
#: arm **orphans** prior outputs rather than silently reusing a mismatched
#: result (root §10.4).
ARM_MODEL_TAG: dict[str, str] = {
    "main": "itr",       # 15 origins x 4 K x 5 seeds, H=24
    "uniform": "itru",   # `D50` — attention forced uniform, K=8
    "fresh": "itrf",     # root §8.1 falsification arm, trained at o_i + 90 d
    "horizon": "itr",    # `D08`/`D48` — 4 named origins x 4 K x 4 H x 3 seeds
}

#: Seeds for the horizon sweep. Three, not five: root §10.2 budgets 192 runs for
#: it, and root §10.3 says to cut the sweep before cutting seed counts if the
#: grid ever stops fitting, because `D30` and `D49` depend on the seed counts.
SWEEP_SEEDS: tuple[int, ...] = SEEDS[:3]

#: Root §10.5. Checked at **run boundaries**, not epoch boundaries: runs are
#: short, epochs are shorter, and the checkpoint granularity is the run.
SESSION_BUDGET_H: float = 11.0
RESERVE_H: float = 0.5


@dataclass(frozen=True, slots=True)
class RunCell:
    """One (arm, origin, K, H, seed) cell of the grid."""

    arm: str
    origin_index: int
    k: int
    pred_len: int
    seed: int

    @property
    def model_tag(self) -> str:
        return ARM_MODEL_TAG[self.arm]

    @property
    def spec(self) -> RunSpec:
        return RunSpec(
            model=self.model_tag,
            origin_index=self.origin_index,
            k=self.k,
            pred_len=self.pred_len,
            seed=self.seed,
        )

    @property
    def run_id(self) -> str:
        return self.spec.run_id

    @property
    def group(self) -> tuple[str, int, int, int]:
        """The tensor-build key. Seeds inside a group share one build."""
        return (self.arm, self.origin_index, self.k, self.pred_len)

    def origin(self) -> OriginLike:
        base = ORIGINS[self.origin_index - 1]
        return FalsificationOrigin(base) if self.arm == "fresh" else base

    def model_config(self) -> ITransformerConfig:
        """Hyperparameters, identical at every rung except where an arm says otherwise.

        **No per-rung tuning** (`D38`): holding capacity fixed is what makes the
        rungs comparable, and tuning per rung would confound the ladder with
        model selection. The only field an arm may move is
        ``uniform_attention``, which *is* the arm.
        """
        return ITransformerConfig(
            pred_len=self.pred_len,
            uniform_attention=(self.arm == "uniform"),
        )


def manifest(
    arms: tuple[str, ...] = ("main", "uniform", "fresh", "horizon"),
) -> list[RunCell]:
    """Every iTransformer run in the study, deduplicated and ordered.

    Root §10.2's accounting, arm by arm:

    ==========  =====  =========================================================
    Arm         Runs   Composition
    ==========  =====  =========================================================
    main          300  15 origins x 4 K x 5 seeds (`D49` — 5 at *every* rung,
                       because the 8->12 rung is RQ1's designed contrast and
                       cannot carry the fewest)
    uniform        75  15 x K=8 x 5 seeds (`D50`)
    fresh          15  one fresh model per origin at ``o_i + 90 d``
    horizon       192  4 named origins x 4 K x 4 H x 3 seeds (`D08`, `D48`)
    ==========  =====  =========================================================

    **48 of those cells are literally the same run.** The sweep's ``H=24`` slice
    at seeds 42-44 carries the same ``run_id`` as the corresponding main-grid
    cells, so the union is **534 unique runs**, not 582. Deduplicating is not a
    saving quietly banked: root §10.4 makes ``run_id`` the identity of a run, so
    executing one twice would mean two files racing for one path.

    Ordering is by group, so the seeds of a cell reuse one tensor build, and
    groups are emitted arm by arm so a shard split stays balanced across the
    heavy ``H=168`` cells.
    """
    cells: list[RunCell] = []

    if "main" in arms:
        cells += [
            RunCell("main", o.index, k, PRED_LEN, s)
            for o in ORIGINS for k in K_LADDER for s in SEEDS
        ]
    if "uniform" in arms:
        cells += [
            RunCell("uniform", o.index, 8, PRED_LEN, s) for o in ORIGINS for s in SEEDS
        ]
    if "fresh" in arms:
        # One seed. The arm asks whether the aged-minus-fresh gap is zero, and
        # that contrast is between two models, not between five initialisations.
        cells += [RunCell("fresh", o.index, 8, PRED_LEN, SEEDS[0]) for o in ORIGINS]
    if "horizon" in arms:
        cells += [
            RunCell("horizon", i, k, h, s)
            for i in SWEEP_ORIGIN_INDICES
            for k in K_LADDER
            for h in HORIZONS
            for s in SWEEP_SEEDS
        ]

    seen: set[str] = set()
    unique: list[RunCell] = []
    for cell in cells:
        if cell.run_id not in seen:
            seen.add(cell.run_id)
            unique.append(cell)
    return unique


def discover_roots(working: Path = ARTIFACTS) -> list[Path]:
    """Artifact roots to search, working directory first.

    Root §10.5: discover by **globbing** ``/kaggle/input/*/``, never a hard-coded
    dataset slug, so the Kaggle Dataset can be renamed without editing code. Any
    input directory holding a ``preds`` folder counts, whatever it is called, and
    one nesting level is searched because Kaggle wraps some dataset uploads in an
    extra folder.
    """
    roots = [Path(working)]
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        roots += sorted(p for p in kaggle_input.iterdir() if (p / "preds").is_dir())
        roots += sorted(p.parent for p in kaggle_input.glob("*/*/preds") if p.is_dir())
    seen: set[str] = set()
    return [r for r in roots if not (str(r) in seen or seen.add(str(r)))]


def completed_run_ids(roots: list[Path]) -> set[str]:
    """Run ids complete under root §10.5: both artifacts present, status complete.

    A prediction file without its meta, or a meta whose status is anything else,
    is **not** complete and the run is redone from scratch. Intra-run
    checkpointing is deliberately omitted: at roughly 90 s per run it costs more
    complexity than it saves.
    """
    done: set[str] = set()
    for root in roots:
        meta_dir = Path(root) / "meta"
        if not meta_dir.is_dir():
            continue
        for meta_path in meta_dir.glob("*.json"):
            run_id = meta_path.stem
            if not (Path(root) / "preds" / f"{run_id}.parquet").exists():
                continue
            try:
                if json.loads(meta_path.read_text()).get("status") == "complete":
                    done.add(run_id)
            except (json.JSONDecodeError, OSError):
                continue
    return done


def pending(cells: list[RunCell], roots: list[Path]) -> list[RunCell]:
    """Manifest minus what is already complete, order preserved."""
    done = completed_run_ids(roots)
    return [c for c in cells if c.run_id not in done]


def shard(cells: list[RunCell], index: int, count: int) -> list[RunCell]:
    """Round-robin by **group**, so a cell's seeds share one tensor build.

    Sharding by cell instead would send consecutive seeds to different workers
    and make both build the same tensors — correct, but paying the build cost
    twice for nothing.
    """
    groups: list[tuple[str, int, int, int]] = []
    for cell in cells:
        if cell.group not in groups:
            groups.append(cell.group)
    owned = {g for i, g in enumerate(groups) if i % count == index}
    return [c for c in cells if c.group in owned]


class BudgetGuard:
    """Root §10.5's session budget, checked at run boundaries.

    Hitting Kaggle's own 12 h wall interactively loses ``/kaggle/working``
    entirely, so the guard stops early enough that Save Version still runs. It
    also refuses to *start* a run it does not expect to finish, using the
    observed mean wall time: stopping at 10.9 h and then beginning a 98 s run is
    the failure mode a naive elapsed-only check has.
    """

    def __init__(
        self, budget_h: float = SESSION_BUDGET_H, reserve_h: float = RESERVE_H
    ) -> None:
        self.deadline = time.perf_counter() + (budget_h - reserve_h) * 3600.0
        self.durations: list[float] = []

    def record(self, seconds: float) -> None:
        self.durations.append(seconds)

    @property
    def mean_run_s(self) -> float:
        return sum(self.durations) / len(self.durations) if self.durations else 120.0

    @property
    def remaining_s(self) -> float:
        return self.deadline - time.perf_counter()

    def may_start(self) -> bool:
        return self.remaining_s > self.mean_run_s


class _TensorCache:
    """Small LRU over ``(arm, origin, K, H)`` builds.

    Bounded because one build is up to 70.12 MB of training tensor plus its
    validation and test blocks (root §10.3 / `D25`); a few is comfortable in
    Kaggle's RAM, an unbounded cache across 154 groups is not.
    """

    def __init__(self, features: pl.DataFrame, size: int = 3) -> None:
        self.features = features
        self.size = size
        self._store: OrderedDict[tuple, OriginTensors] = OrderedDict()

    def get(self, cell: RunCell) -> OriginTensors:
        key = cell.group
        if key in self._store:
            self._store.move_to_end(key)
            return self._store[key]
        tensors = build_origin_tensors(
            self.features, cell.origin(), cell.k, pred_len=cell.pred_len
        )
        self._store[key] = tensors
        while len(self._store) > self.size:
            self._store.popitem(last=False)
        return tensors


@dataclass(frozen=True, slots=True)
class ExecutionSummary:
    """What one worker did, and what is left.

    ``remaining`` and ``estimated_sessions`` are printed on exit because a
    session that ends without saying how much is left forces the next one to
    re-derive it (``notebooks/CLAUDE.md``).
    """

    completed: int
    skipped: int
    failed: int
    #: Pending **in this shard**, not in the whole manifest. The notebook prints
    #: the global figure; a worker only knows its own queue.
    remaining: int
    wall_time_s: float
    mean_run_s: float

    @property
    def estimated_sessions(self) -> float:
        if self.remaining == 0:
            return 0.0
        usable = (SESSION_BUDGET_H - RESERVE_H) * 3600.0
        return self.remaining * self.mean_run_s / usable

    def __str__(self) -> str:
        return (
            f"completed {self.completed}  skipped {self.skipped}  "
            f"failed {self.failed}  remaining {self.remaining}\n"
            f"wall {self.wall_time_s / 3600:.2f} h  mean run {self.mean_run_s:.1f} s  "
            f"estimated sessions left {self.estimated_sessions:.2f}"
        )


def execute(
    cells: list[RunCell],
    features: pl.DataFrame,
    *,
    out_root: Path = ARTIFACTS,
    roots: list[Path] | None = None,
    guard: BudgetGuard | None = None,
    device=None,
    log=print,
) -> ExecutionSummary:
    """Run a shard to completion or to the budget, whichever comes first.

    A failing run is logged and skipped rather than aborting the shard: with 534
    runs, losing the rest of a session to one bad cell is worse than finishing
    the others and letting root §10.5's resume pick that cell up next session.
    """
    guard = guard or BudgetGuard()
    roots = roots or discover_roots(out_root)
    device = device or pick_device()
    cache = _TensorCache(features)

    started = time.perf_counter()
    done = completed_run_ids(roots)
    completed = skipped = failed = 0
    queue = list(cells)

    for position, cell in enumerate(queue, start=1):
        if cell.run_id in done or is_complete(cell.run_id, out_root):
            skipped += 1
            continue
        if not guard.may_start():
            log(
                f"budget guard: {guard.remaining_s / 60:.1f} min left, mean run "
                f"{guard.mean_run_s:.0f} s — stopping cleanly so the version saves"
            )
            break

        began = time.perf_counter()
        try:
            tensors = cache.get(cell)
            model, outcome = train_one(
                tensors, cell.spec, cell.model_config(), device=device
            )
            write_artifacts(
                model, tensors, cell.spec, cell.model_config(), outcome, device,
                root=out_root,
            )
        except Exception as exc:  # noqa: BLE001 - one bad cell must not end the shard
            failed += 1
            log(f"[{position}/{len(queue)}] {cell.run_id} FAILED: {exc!r}")
            continue

        elapsed = time.perf_counter() - began
        guard.record(elapsed)
        completed += 1
        log(
            f"[{position}/{len(queue)}] {cell.run_id}  "
            f"epochs={outcome.epochs_run}  val={outcome.best_val_mse:.6f}  "
            f"{elapsed:.1f}s  n_train={len(tensors.train)}"
        )

    return ExecutionSummary(
        completed=completed,
        skipped=skipped,
        failed=failed,
        remaining=len(pending(queue, discover_roots(out_root))),
        wall_time_s=time.perf_counter() - started,
        mean_run_s=guard.mean_run_s,
    )


@dataclass(frozen=True, slots=True)
class PilotResult:
    """Root §8.5's Stage 5 gate. Note what it does **not** touch: the test blocks."""

    val_mse: dict[int, float]
    clark_west: object
    n_val: int
    passed: bool

    def __str__(self) -> str:
        rungs = "  ".join(f"K={k}: {v:.6f}" for k, v in sorted(self.val_mse.items()))
        verdict = (
            "PASS — K=8 beats K=1 on validation; keep the framing as written"
            if self.passed
            else "FAIL — reposition the title to the descriptive variant NOW, "
                 "not in week nine (root §8.5)"
        )
        return f"validation MSE  {rungs}\n{self.clark_west}\n{verdict}"


def stage5_pilot(
    features: pl.DataFrame,
    *,
    origin_index: int = 1,
    rungs: tuple[int, ...] = K_LADDER,
    seeds: tuple[int, ...] = SEEDS[:3],
    out_root: Path = ARTIFACTS,
    device=None,
    log=print,
) -> PilotResult:
    """Origin 1, 4 K x 3 seeds, scored on the **validation** sub-block (`D27`).

    §11's final item requires the test blocks be opened once, after the design is
    frozen; a gate that repositions the title on a test-block result cannot
    coexist with it. The validation sub-block is the leak-free instrument for a
    go/no-go on architecture.

    The twelve cells are ordinary main-grid ``run_id``s and their artifacts are
    written, so the pilot costs the grid nothing: §10.5's resume finds them
    complete and the main run skips them. That is deliberate and is *why* the
    gate must run on validation — with a test-block gate, the origin that decided
    the paper's framing would end up back inside the evidence for it.

    The gate statistic is **Clark-West, not DM** (`D29`): K=1's feature set is a
    strict subset of K=8's under the same architecture and sample, so the pair is
    nested and standard DM is undersized against exactly the alternative being
    tested. Predictions are averaged across seeds before the test, matching §9.1's
    order of operations.
    """
    from itransformer_btc import metrics
    from itransformer_btc.train import predict

    device = device or pick_device()
    origin = ORIGINS[origin_index - 1]
    cache = _TensorCache(features, size=1)

    val_mse: dict[int, float] = {}
    val_pred: dict[int, "object"] = {}
    y_val = None

    import numpy as _np
    import torch as _torch

    for k in rungs:
        cell = RunCell("main", origin_index, k, PRED_LEN, seeds[0])
        tensors = cache.get(cell)
        y_val = tensors.val.y
        stacked = []
        for seed in seeds:
            spec = RunCell("main", origin_index, k, PRED_LEN, seed).spec
            model, outcome = train_one(tensors, spec, cell.model_config(), device=device)
            write_artifacts(model, tensors, spec, cell.model_config(), outcome,
                            device, root=out_root)
            stacked.append(
                predict(model, _torch.from_numpy(tensors.val.x).to(device))
            )
            log(f"pilot {spec.run_id}  val={outcome.best_val_mse:.6f}  "
                f"{outcome.wall_time_s:.1f}s")
        mean_pred = _np.mean(_np.stack(stacked), axis=0)
        val_pred[k] = mean_pred
        val_mse[k] = float(_np.mean((y_val - mean_pred) ** 2))

    small, large = min(rungs), 8
    # One loss value per forecast origin: the DM/CW series is indexed by the
    # moment the forecast was issued, not by the (origin, step) pair.
    cw = metrics.clark_west_test(
        y_val.mean(axis=1),
        val_pred[small].mean(axis=1),
        val_pred[large].mean(axis=1),
        h=PRED_LEN,
        name=f"Clark-West K={small} vs K={large} (validation)",
    )
    return PilotResult(
        val_mse=val_mse,
        clark_west=cw,
        n_val=len(y_val),
        passed=bool(cw.p_value < 0.05 and val_mse[large] < val_mse[small]),
    )


def build_feature_frame(parquet: Path = DEFAULT_PARQUET) -> pl.DataFrame:
    """Load the immutable artifact and compute the twelve variates."""
    from itransformer_btc.features import build_features
    from itransformer_btc.segments import load_bars, usable_mask

    return build_features(usable_mask(load_bars(parquet)))


def launch_workers(
    n_workers: int = 2,
    *,
    parquet: Path = DEFAULT_PARQUET,
    out_root: Path = ARTIFACTS,
    arms: tuple[str, ...] = ("main", "uniform", "fresh", "horizon"),
    budget_h: float = SESSION_BUDGET_H,
    reserve_h: float = RESERVE_H,
    package_root: Path = Path("src"),
    poll_s: float = 20.0,
) -> int:
    """Spawn one process per GPU and stream their logs. Returns the summed exit code.

    Each child gets ``CUDA_VISIBLE_DEVICES`` set to exactly one physical GPU, so
    it sees that GPU as ``cuda:0`` and its global RNG belongs to it alone. That
    is what makes ``set_seed`` mean what root §16 says it means with two GPUs in
    play.

    ``package_root`` is the directory holding the importable ``itransformer_btc``
    package, and it is a parameter rather than a constant because it is not the
    same directory in the two places this runs: ``src/`` from a checkout, and the
    working directory from a notebook that materialised the package itself
    (`D54`). The child is a fresh interpreter, so it inherits nothing from the
    parent's ``sys.path`` and must be told.
    """
    log_dir = Path(out_root) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    procs: list[subprocess.Popen] = []
    handles: list[tuple[int, Path, object]] = []
    offsets: list[int] = []

    for index in range(n_workers):
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(index)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(package_root), env.get("PYTHONPATH", "")]
        ).strip(os.pathsep)
        env["PYTHONUNBUFFERED"] = "1"
        # The child records the input-artifact digest in every meta it writes,
        # and can only find the artifact if it is told where it is (root §12).
        env[INPUT_PARQUET_ENV] = str(parquet)

        path = log_dir / f"worker{index}.log"
        handle = path.open("w", encoding="utf-8")
        procs.append(
            subprocess.Popen(
                [
                    sys.executable, "-m", "itransformer_btc.runner",
                    "--shard", str(index),
                    "--shards", str(n_workers),
                    "--parquet", str(parquet),
                    "--out", str(out_root),
                    "--arms", ",".join(arms),
                    "--budget-h", str(budget_h),
                    "--reserve-h", str(reserve_h),
                ],
                env=env, stdout=handle, stderr=subprocess.STDOUT,
            )
        )
        handles.append((index, path, handle))
        offsets.append(0)

    def _drain() -> None:
        for slot, (index, path, _) in enumerate(handles):
            text = path.read_text(encoding="utf-8", errors="replace")
            if len(text) > offsets[slot]:
                for line in text[offsets[slot]:].splitlines():
                    if line.strip():
                        print(f"[gpu{index}] {line}")
                offsets[slot] = len(text)

    try:
        while any(p.poll() is None for p in procs):
            time.sleep(poll_s)
            _drain()
    finally:
        for proc in procs:
            proc.wait()
        _drain()
        for _, _, handle in handles:
            handle.close()

    return sum(p.returncode or 0 for p in procs)


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="One GPU worker for the grid.")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--out", type=Path, default=ARTIFACTS)
    parser.add_argument("--arms", type=str, default="main,uniform,fresh,horizon")
    parser.add_argument("--budget-h", type=float, default=SESSION_BUDGET_H)
    parser.add_argument("--reserve-h", type=float, default=RESERVE_H)
    args = parser.parse_args(argv)

    # Set, not defaulted: a worker invoked directly with --parquet must record
    # that artifact's digest, not whatever the launcher happened to export.
    os.environ[INPUT_PARQUET_ENV] = str(args.parquet)

    arms = tuple(a.strip() for a in args.arms.split(",") if a.strip())
    features = build_feature_frame(args.parquet)
    roots = discover_roots(args.out)
    # Shard the FULL manifest, then subtract what is done — never the reverse.
    # Sharding the pending list makes the partition a function of how many runs
    # happen to be complete at that instant, so two workers that start seconds
    # apart (one finishes a run while the other is still building features) get
    # partitions that are not complementary: some groups owned by both, some by
    # neither. `execute` skips completed cells anyway, so filtering afterwards
    # costs nothing and the partition stays deterministic.
    todo = pending(shard(manifest(arms), args.shard, args.shards), roots)

    device = pick_device()
    print(
        f"shard {args.shard}/{args.shards}  device={device}  "
        f"pending in shard={len(todo)}  features={features.height} rows",
        flush=True,
    )
    summary = execute(
        todo, features, out_root=args.out, roots=roots,
        guard=BudgetGuard(args.budget_h, args.reserve_h), device=device,
        log=lambda msg: print(msg, flush=True),
    )
    print(summary, flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
