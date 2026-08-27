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
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import torch

from itransformer_btc.config import (
    HORIZONS,
    K_LADDER,
    ORIGINS,
    PRED_LEN,
    SEQ_LEN,
    SEEDS,
    SWEEP_ORIGIN_INDICES,
    FalsificationOrigin,
    OriginLike,
)
# Names, not the module. In the flattened notebook there is no ``baselines``
# module object to attribute off — every definition lands in one namespace — so
# ``baselines.RidgeConfig`` would be a NameError hours into a Kaggle session
# while passing every parse-level check here (root §15, `D58`).
from itransformer_btc.baselines import (
    DLinearConfig,
    LSTMConfig,
    NaiveConfig,
    PatchTSTConfig,
    RidgeConfig,
    assert_baseline_alignment,
)
from itransformer_btc.features import MATCHED_K_SUBSETS
from itransformer_btc.attention import tercile_maps
from itransformer_btc.model import ITransformerConfig, LongScheduleConfig
from itransformer_btc.splits import OriginTensors, build_origin_tensors
from itransformer_btc.train import (
    ARTIFACTS,
    DEFAULT_PARQUET,
    INPUT_PARQUET_ENV,
    Architecture,
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
    "ridge": "rdg",      # `D17` — is a transformer needed at all? K = 1,4,8,12
    "dlinear": "dlin",   # root §7 — "not optional", K=8
    "patchtst": "ptst",  # root §7 — SOTA channel-independent, K=8
    "lstm": "lstm",      # root §7 — the RNN the crypto literature reaches for, K=8
    "persist": "npst",   # root §7 — y_hat = last observed return, K=1
    "seasonal": "nsea",  # root §7 — y_hat = return at t-24, K=1
    # `D70`'s four arms. All exploratory, all declared before running, all
    # reported whatever they show — root §13.2's commitment, which an arm
    # reported only when it agrees with the headline does not meet.
    "orthogonal": "itro",  # K=8, one or two per family — high effective rank
    "redundant": "itrr",   # K=8, F2 and F3 loaded whole — low effective rank
    "look048": "l048",     # L=48 at K=8 — half the pre-registered lookback
    "look192": "l192",     # L=192 at K=8 — double it
    "tuned": "itrt",       # K=8 at the config origin 1's validation preferred
    # `D62`'s three exploratory arms. Distinct tags, so none can collide with a
    # completed run_id and all 684 stay complete under resume.
    "attention": "itra",  # `D62d` — Figure 5's attention maps, K=8
    "longsched": "itrl",  # `D62c` — LR halved every 8, 60 epochs, patience 10
    "capacity": "itrc",   # `D62b` — root §6.2's own larger-d_ff run at K=12
}

#: The §7 comparators (`D56`). Two things key off this set, and both follow from
#: these arms being a *different model* rather than a different configuration of
#: the same one: the `D45` window-alignment assertion runs for them, and their
#: ``run_id`` prefix keeps them apart in every table.
BASELINE_ARMS: tuple[str, ...] = (
    "ridge", "dlinear", "patchtst", "lstm", "persist", "seasonal",
)

#: Every arm, in execution order. iTransformer first, so that a session cut short
#: leaves the ladder — which RQ1, RQ2 and RQ3 all read — complete before the
#: comparators, and so a baseline's alignment assertion finds its reference on
#: disk rather than reporting itself unchecked.
#: `D62`'s exploratory arms. Ordered **last** so a session cut short loses
#: robustness rather than anything RQ1-RQ3 reads, on the same reasoning that put
#: the baselines after the ladder.
ROBUSTNESS_ARMS: tuple[str, ...] = (
    "attention", "longsched", "capacity",
    # `D70`. Ordered with the rest of the robustness block, after the baselines,
    # so a session cut short loses a robustness arm rather than an RQ input.
    "orthogonal", "redundant", "look048", "look192", "tuned",
)

ALL_ARMS: tuple[str, ...] = (
    "main", "uniform", "fresh", "horizon", *BASELINE_ARMS, *ROBUSTNESS_ARMS,
)

#: Seeds for the horizon sweep. Three, not five: root §10.2 budgets 192 runs for
#: it, and root §10.3 says to cut the sweep before cutting seed counts if the
#: grid ever stops fitting, because `D30` and `D49` depend on the seed counts.
SWEEP_SEEDS: tuple[int, ...] = SEEDS
#: Was ``SEEDS[:3]``, and the reason was budget rather than design (`D70`): root
#: §10.3 sized the sweep against a single-GPU session. With both devices working
#: the third and fourth seed cost wall-clock the session has, and `D30`'s rule
#: cuts the other way once they are affordable — a number aggregated across
#: origins carries an SE across origins, and seed dispersion is the Monte-Carlo
#: diagnostic beside it. Three seeds is a thin diagnostic.

#: Seeds for the stochastic baselines. Three, matching root §10.2's baseline
#: budget of "3 stochastic x 3 seeds". `D49`'s five-seed rule is about the
#: **rungs of the ladder**, where the 8->12 contrast cannot be the one carrying
#: the fewest; a baseline is a single cell rather than a rung, and a fourth and
#: fifth seed there would buy precision on a number no hypothesis is stated about.
BASELINE_SEEDS: tuple[int, ...] = SEEDS
#: Also raised from three (`D70`), and the attention arm is why it matters most:
#: root §13.2 admits attention maps only when they are "validated for stability
#: across seeds", and the measured calm-to-stress shift (**+0.00056**) is smaller
#: than the between-seed standard deviation of a single weight (**0.00064**). A
#: claim that rests on that comparison should not rest on three draws.

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
    #: Lookback. Only the `D70` sweep moves it; every other arm takes root §6.2's
    #: 96. It is **not** in ``run_id`` — root §10.4 fixes that format — so the two
    #: sweep arms carry their own tags and cannot collide with the ladder.
    seq_len: int = SEQ_LEN
    #: Name in :data:`MATCHED_K_SUBSETS`, or "" for the rung's own columns.
    subset: str = ""

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
        """The **shard** key. Seeds inside a group stay with one worker."""
        return (self.arm, self.origin_index, self.k, self.pred_len)

    @property
    def tensor_key(self) -> tuple[bool, int, int, int]:
        """The **tensor-build** key, which is coarser than :attr:`group`.

        :func:`build_origin_tensors` reads the origin, K and H and nothing else,
        and only the falsification arm changes the origin object. So ridge at
        (origin 7, K=8, H=24) consumes byte-for-byte the tensors the main arm
        already built there, and keying the cache by arm would rebuild them —
        150 redundant builds across the baseline arms. Sharding still keys on
        :attr:`group`, because that partition must be a function of the arm.
        """
        return (
            self.arm == "fresh",
            self.origin_index,
            self.k,
            self.pred_len,
            # `D70`: a lookback and a column set change the tensors, so two cells
            # that differ in either must not share a cached build. Left out, the
            # L=192 arm would silently train on L=96 windows.
            self.seq_len,
            self.subset,
        )

    def columns(self) -> tuple[str, ...] | None:
        """The named column set this cell trains on, or ``None`` for its rung."""
        return MATCHED_K_SUBSETS[self.subset] if self.subset else None

    def origin(self) -> OriginLike:
        base = ORIGINS[self.origin_index - 1]
        return FalsificationOrigin(base) if self.arm == "fresh" else base

    def model_config(self, overrides: dict[str, Architecture] | None = None) -> Architecture:
        """The arm's configuration — hyperparameters fixed a priori in every case.

        **No per-rung tuning** (`D38`): holding capacity fixed is what makes the
        rungs comparable, and tuning per rung would confound the ladder with
        model selection. The only field an iTransformer arm may move is
        ``uniform_attention``, which *is* the arm.

        The rule extends to the §7 baselines rather than exempting them. Ridge's
        alpha is the single exception root §11 names, and it is not chosen here:
        :meth:`RidgeConfig.fit` selects it on the validation sub-block and
        returns the resolved config, which is what reaches ``meta['config']``.
        """
        if self.arm == "ridge":
            return RidgeConfig(pred_len=self.pred_len, k=self.k)
        if self.arm == "dlinear":
            return DLinearConfig(pred_len=self.pred_len)
        if self.arm == "patchtst":
            return PatchTSTConfig(pred_len=self.pred_len)
        if self.arm == "lstm":
            return LSTMConfig(pred_len=self.pred_len, k=self.k)
        if self.arm in ("persist", "seasonal"):
            return NaiveConfig(mode=self.arm, pred_len=self.pred_len, k=self.k)
        if self.arm == "tuned":
            # Selected once, on origin 1's validation, exactly where `D27` put the
            # Stage 5 gate and for the same reason: a selection that reads a test
            # block cannot coexist with root §11's "test blocks are opened once".
            # The notebook computes it in the prelude and hands it in; there is no
            # default, because a tuned arm silently running the untuned config
            # would answer the referee's question with the wrong number.
            if not overrides or "tuned" not in overrides:
                raise ValueError(
                    "the tuned arm needs its selected config passed in "
                    "(`configs={'tuned': ...}`); run tune_on_validation first"
                )
            return overrides["tuned"]
        if self.arm in ("look048", "look192", "orthogonal", "redundant"):
            return ITransformerConfig(seq_len=self.seq_len, pred_len=self.pred_len)
        if self.arm == "longsched":
            # `D62c`. The only thing that moves is the schedule, which is a
            # method, so meta['config'] is byte-identical to the main arm's and
            # meta['schedule'] is where the difference shows.
            return LongScheduleConfig(pred_len=self.pred_len)
        if self.arm == "capacity":
            # `D62b`. Root §6.2 pre-registers exactly this --- "one robustness run
            # at K=12 with larger d_ff, so a flat 8->12 rung cannot be read as an
            # under-tuning artefact" --- and it was never built. ``d_ff`` IS a
            # config field, so this run's meta records the widening, correctly:
            # it is the only thing differing from the rung it answers for.
            return ITransformerConfig(pred_len=self.pred_len, d_ff=512)
        return ITransformerConfig(
            pred_len=self.pred_len,
            uniform_attention=(self.arm == "uniform"),
        )

    def reference_run_id(self) -> str:
        """The iTransformer run this cell is compared against (`D45`).

        Same origin, same K, same horizon, first seed — the main-grid cell whose
        evaluated window set this run's must equal exactly before any RelMSE or
        DM statistic is formed across the two.
        """
        return RunSpec(
            ARM_MODEL_TAG["main"], self.origin_index, self.k, self.pred_len, SEEDS[0]
        ).run_id


def manifest(arms: tuple[str, ...] = ALL_ARMS) -> list[RunCell]:
    """Every run in the study, deduplicated and ordered.

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
    ridge          60  15 x 4 K, deterministic (`D17`)
    dlinear        45  15 x K=8 x 3 seeds (root §7)
    patchtst       45  15 x K=8 x 3 seeds (root §7)
    lstm           45  15 x K=8 x 3 seeds (root §7, `D64`)
    persist        15  15 origins, deterministic (root §7, `D64`)
    seasonal       15  15 origins, deterministic (root §7, `D64`)
    ==========  =====  =========================================================

    **48 of those cells are literally the same run.** The sweep's ``H=24`` slice
    at seeds 42-44 carries the same ``run_id`` as the corresponding main-grid
    cells, so the iTransformer union is **534 unique runs**, not 582, and the
    whole manifest is **684**. Deduplicating is not a saving quietly banked: root
    §10.4 makes ``run_id`` the identity of a run, so executing one twice would
    mean two files racing for one path.

    **The three baseline arms are new, and their absence was `D56`.** Root §7
    calls DLinear and PatchTST "not optional" and §10.2 budgets 255 baseline
    runs, but no baseline class existed and this manifest never contained one —
    so §10.2's 789 was never executable, and the study's central architectural
    comparison had no data. 150 of that 255 are built: the deferred remainder is
    ARIMA, LSTM, naive-persist and seasonal-naive, listed in ``baselines.py``
    rather than left silently unbuilt. Naive-RW needs no run at all, being
    computed inside :func:`itransformer_btc.metrics.block_metrics` on exactly the
    rows its comparator was scored on.

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
    if "ridge" in arms:
        # One seed. Ridge is a solve, not an optimisation: a second seed would
        # reproduce the first to the last bit. The seed component of ``run_id``
        # is carried only because root §10.4 fixes the format.
        cells += [
            RunCell("ridge", o.index, k, PRED_LEN, SEEDS[0])
            for o in ORIGINS for k in K_LADDER
        ]
    if "dlinear" in arms:
        cells += [
            RunCell("dlinear", o.index, 8, PRED_LEN, s)
            for o in ORIGINS for s in BASELINE_SEEDS
        ]
    if "patchtst" in arms:
        cells += [
            RunCell("patchtst", o.index, 8, PRED_LEN, s)
            for o in ORIGINS for s in BASELINE_SEEDS
        ]
    if "lstm" in arms:
        cells += [
            RunCell("lstm", o.index, 8, PRED_LEN, s)
            for o in ORIGINS for s in BASELINE_SEEDS
        ]
    for naive in ("persist", "seasonal"):
        # One seed each. Neither has a parameter and neither consumes RNG, so a
        # second seed would reproduce the first to the last bit — the same
        # reasoning that gives ridge one seed.
        if naive in arms:
            cells += [
                RunCell(naive, o.index, 1, PRED_LEN, SEEDS[0]) for o in ORIGINS
            ]
    for arm, subset in (("orthogonal", "orthogonal"), ("redundant", "redundant")):
        # Same K, same seeds, same everything but the column set (`D70`). Five
        # seeds because this is RQ1's direct contrast and `D49`'s reasoning about
        # the ladder applies to it: the rung carrying the comparison cannot be the
        # one carrying the fewest draws.
        if arm in arms:
            cells += [
                RunCell(arm, o.index, 8, PRED_LEN, s, subset=subset)
                for o in ORIGINS for s in SEEDS
            ]
    for arm, seq_len in (("look048", 48), ("look192", 192)):
        # L is the one first-order hyperparameter root §6.2 never varied. 192 is
        # the ceiling: window cost per break is ``L + H - 1``, so at 336 the
        # pooled loss approaches root §4.3's tolerance and the early origins,
        # where every outage lives, would carry it worst.
        if arm in arms:
            cells += [
                RunCell(arm, o.index, 8, PRED_LEN, s, seq_len=seq_len)
                for o in ORIGINS for s in SWEEP_SEEDS
            ]
    if "tuned" in arms:
        # The config is selected once on origin 1's validation and handed to
        # ``execute``; these cells only carry it across the panel.
        cells += [
            RunCell("tuned", o.index, 8, PRED_LEN, s)
            for o in ORIGINS for s in SEEDS
        ]
    if "attention" in arms:
        # Three seeds, because root §13.2 admits attention maps only when they are
        # "validated for stability across seeds" (Jain & Wallace 2019; Wiegreffe &
        # Pinter 2019). One seed makes that claim unfalsifiable; five would buy
        # precision on a descriptive figure. Same seeds as the main arm, so each
        # cell must reproduce its twin bit for bit --- which is the arm's second
        # product and the study's reproducibility statement.
        cells += [
            RunCell("attention", o.index, 8, PRED_LEN, s)
            for o in ORIGINS for s in BASELINE_SEEDS
        ]
    if "longsched" in arms:
        # K in {1, 8} only: the arm asks whether the null survives a longer
        # schedule, and that needs the control and the treatment, not the whole
        # ladder. Three seeds --- `D49`'s five-seed rule protects the 8->12
        # contrast, which this arm does not contain.
        cells += [
            RunCell("longsched", o.index, k, PRED_LEN, s)
            for o in ORIGINS for k in (1, 8) for s in BASELINE_SEEDS
        ]
    if "capacity" in arms:
        # Five seeds, matching the K=12 rung it is compared against.
        cells += [
            RunCell("capacity", o.index, 12, PRED_LEN, s) for o in ORIGINS for s in SEEDS
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
    checkpointing is deliberately omitted: at ~30 s per run measured (`D57`) it
    costs far more complexity than it saves.
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
        key = cell.tensor_key
        if key in self._store:
            self._store.move_to_end(key)
            return self._store[key]
        tensors = build_origin_tensors(
            self.features,
            cell.origin(),
            cell.k,
            seq_len=cell.seq_len,
            pred_len=cell.pred_len,
            columns=cell.columns(),
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


def _assert_alignment(cell: RunCell, roots: list[Path], log) -> None:
    """`D45`, enforced when the file is written rather than when the table is built.

    Root §7 requires every baseline to be scored on **exactly** the surviving
    window set of the run it is compared against. The two sets are equal by
    construction — both come from :func:`window_starts` with the same origin,
    span and semantics — which is why this costs microseconds and why it is the
    only thing that would notice if that ever stopped holding.

    A missing comparator is **reported, never swallowed**. The check is then
    unrun, and an unrun check that prints nothing is indistinguishable from a
    passing one; :data:`ALL_ARMS` orders the ladder first precisely so this stays
    the rare case rather than the normal one.
    """
    reference = cell.reference_run_id()
    try:
        assert_baseline_alignment(cell.run_id, reference, roots)
    except FileNotFoundError:
        log(
            f"  {cell.run_id}: `D45` alignment UNCHECKED — comparator "
            f"{reference} is not on disk in {[str(r) for r in roots]}"
        )


def execute(
    cells: list[RunCell],
    features: pl.DataFrame,
    *,
    out_root: Path = ARTIFACTS,
    roots: list[Path] | None = None,
    guard: BudgetGuard | None = None,
    device=None,
    configs: dict[str, Architecture] | None = None,
    log=print,
) -> ExecutionSummary:
    """Run a shard to completion or to the budget, whichever comes first.

    A failing run is logged and skipped rather than aborting the shard: with 684
    runs, losing the rest of a session to one bad cell is worse than finishing
    the others and letting root §10.5's resume pick that cell up next session.
    A failing *invariant* is the opposite case and does end the shard — see
    :func:`_assert_alignment`.
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
            # The config comes back **resolved**: identical for every
            # iTransformer arm (`D38` — nothing is tuned), and carrying the
            # chosen alpha for ridge, which is the one selection root §11 admits.
            # Writing the config that went in would lose it.
            model, cfg, outcome = cell.model_config(configs).fit(
                tensors, cell.spec, device=device
            )
            # Figure 5's maps, and only for the arm that exists to produce them
            # (`D62d`). Captured after training rather than during it, so nothing
            # about the optimisation changes and the arm reproduces its main-grid
            # twin bit for bit.
            maps = (
                tercile_maps(model, tensors, device) if cell.arm == "attention" else None
            )
            write_artifacts(
                model, tensors, cell.spec, cfg, outcome, device,
                root=out_root, attention=maps,
            )
        except Exception as exc:  # noqa: BLE001 - one bad cell must not end the shard
            failed += 1
            log(f"[{position}/{len(queue)}] {cell.run_id} FAILED: {exc!r}")
            continue

        # Outside the try, and deliberately fatal. A window-set mismatch between
        # a baseline and its comparator is the defect class root §11 calls
        # fatal — RelMSE across two samples is not a ratio — and continuing would
        # fill Table 6 with statistics that mean nothing. One bad *cell* must not
        # end a shard; one broken *invariant* must.
        if cell.arm in BASELINE_ARMS:
            _assert_alignment(cell, roots, log)

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



def visible_devices() -> list[torch.device]:
    """Every CUDA device the session can see, or ``[cpu]`` when there is none.

    Root §10.1 lists 2 x T4 and root §10.3 says one is enough — which was true of
    a 969-run manifest that fits an 11 h session. It stops being true the moment
    the grid grows, and the 894-run session left **half the hardware idle for 7.8
    hours** because nothing asked this question (`D68`).
    """
    if not torch.cuda.is_available():
        return [torch.device("cpu")]
    return [torch.device("cuda", i) for i in range(torch.cuda.device_count())]


def execute_parallel(
    cells: list[RunCell],
    features: pl.DataFrame,
    *,
    devices: list[torch.device] | None = None,
    out_root: Path = ARTIFACTS,
    roots: list[Path] | None = None,
    guard: BudgetGuard | None = None,
    configs: dict[str, Architecture] | None = None,
    log=print,
) -> ExecutionSummary:
    """:func:`execute`, one worker thread per device, off a shared queue.

    **Run level, never batch level** (root §10.3). ``nn.DataParallel`` is rejected
    there with a measured reason that has not changed: at batch 32 and ~280k
    parameters the scatter/gather costs more than splitting saves, and DDP is
    worse again for 969 runs of ~32 s each, where the process group is paid per
    run. What parallelises cleanly is the *grid*, because a run is already the
    unit of work and nothing crosses between two of them.

    Threads rather than processes, because §15's notebook carries the package as
    definition cells in one kernel namespace: a subprocess inherits none of it and
    ``python -m itransformer_btc.runner`` has no files to import. ``launch_workers``
    remains the path from a checkout. Threads are the path from the notebook, and
    the GIL is not the constraint here — every run spends its time inside CUDA
    kernels and tensor ops that release it.

    **Determinism is the property this must not cost**, and the whole design is
    that one guarantee (`D68`):

    - each worker owns one device and never touches another's;
    - :func:`set_seed` is device-scoped, so seeding ``cuda:0`` leaves ``cuda:1``'s
      generator alone;
    - the CPU generator is shared, so seeding and module construction happen under
      :data:`itransformer_btc.train.SEED_LOCK` — milliseconds against a ~32 s run;
    - each worker keeps its **own** tensor cache, so no build races another.

    Everything after the prologue draws from the device's own CUDA generator. A
    run therefore produces the same bytes whether it ran alone or beside another,
    which is what `D62d` demonstrated for the attention arm and what root §12
    requires of every number in the manuscript.

    The budget guard is shared and its methods are called under a lock, so two
    workers cannot both slip past a deadline that only one of them had room for.
    """
    devices = devices or visible_devices()
    if len(devices) < 2:
        return execute(
            cells, features,
            out_root=out_root, roots=roots, guard=guard,
            device=devices[0] if devices else None, configs=configs, log=log,
        )

    guard = guard or BudgetGuard()
    roots = roots or discover_roots(out_root)
    done = completed_run_ids(roots)

    queue = list(cells)
    cursor = 0
    completed = skipped = failed = 0
    state = threading.Lock()
    started = time.perf_counter()
    log(f"run-level parallelism across {[str(d) for d in devices]} (`D68`)")

    def take() -> tuple[int, RunCell] | None:
        nonlocal cursor
        with state:
            if cursor >= len(queue) or not guard.may_start():
                return None
            cursor += 1
            return cursor, queue[cursor - 1]

    def worker(device: torch.device) -> None:
        nonlocal completed, skipped, failed
        cache = _TensorCache(features, size=2)
        while (item := take()) is not None:
            position, cell = item
            if cell.run_id in done or is_complete(cell.run_id, out_root):
                with state:
                    skipped += 1
                continue

            began = time.perf_counter()
            try:
                tensors = cache.get(cell)
                model, cfg, outcome = cell.model_config(configs).fit(
                    tensors, cell.spec, device=device
                )
                maps = (
                    tercile_maps(model, tensors, device)
                    if cell.arm == "attention"
                    else None
                )
                write_artifacts(
                    model, tensors, cell.spec, cfg, outcome, device,
                    root=out_root, attention=maps,
                )
            except Exception as exc:  # noqa: BLE001 - one bad cell must not end the shard
                with state:
                    failed += 1
                log(f"[{position}/{len(queue)}] {device} {cell.run_id} FAILED: {exc!r}")
                continue

            if cell.arm in BASELINE_ARMS:
                _assert_alignment(cell, roots, log)

            elapsed = time.perf_counter() - began
            with state:
                guard.record(elapsed)
                completed += 1
            log(
                f"[{position}/{len(queue)}] {device} {cell.run_id}  "
                f"epochs={outcome.epochs_run}  val={outcome.best_val_mse:.6f}  "
                f"{elapsed:.1f}s  n_train={len(tensors.train)}"
            )

    threads = [
        threading.Thread(target=worker, args=(d,), name=f"grid-{d}", daemon=True)
        for d in devices
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    if cursor < len(queue):
        log(
            f"budget guard: stopped with {len(queue) - cursor} cells unstarted — "
            f"resume picks them up next session (root §10.5)"
        )

    return ExecutionSummary(
        completed=completed,
        skipped=skipped,
        failed=failed,
        remaining=len(pending(queue, discover_roots(out_root))),
        wall_time_s=time.perf_counter() - started,
        mean_run_s=guard.mean_run_s,
    )



#: The grid the tuned arm searches. Declared here, before it runs, because a
#: search space chosen after seeing which config won is not a search (`D70`).
#:
#: Three knobs root §6.2 adopted from Liu et al. (2024) and never varied, each at
#: the published value and one step either side. ``d_ff`` is absent on purpose:
#: `D62b`'s capacity arm already swept it and made things worse at 14 of 15
#: origins, and repeating it here would spend the budget re-answering a question
#: that has an answer.
TUNING_GRID: tuple[dict[str, object], ...] = tuple(
    {"d_model": d, "e_layers": e, "lr": lr}
    for d in (64, 128, 256)
    for e in (2, 3)
    for lr in (1e-4, 3e-4, 1e-3)
)

#: Epochs each probe is given. Short on purpose: this ranks configurations, it
#: does not train them. The winner is then trained under root §6.2's full
#: schedule like every other arm.
TUNING_EPOCHS: int = 6


def tune_on_validation(
    features: pl.DataFrame,
    *,
    origin_index: int = 1,
    k: int = 8,
    device=None,
    log=print,
) -> tuple[ITransformerConfig, list[dict]]:
    """Pick one iTransformer config on one origin's **validation** sub-block.

    This exists to answer the one attack on the null that root §6.2 leaves open.
    Nothing in this study is tuned — every hyperparameter is adopted from
    Liu et al. (2024) and held identical at every rung, which is what makes the
    rungs comparable (`D38`). A referee reads that as *"you did not try"*, and the
    honest reply is a number rather than a paragraph: **if the configuration the
    validation set prefers is still worse than a random walk, the null is not an
    artefact of the defaults.**

    Three properties keep it inside the pre-registration:

    - **Validation only, one origin.** Exactly where `D27` put the Stage 5 gate,
      and for its reason: root §11 opens the test blocks once, after the design is
      frozen, so a selection that reads one cannot coexist with it.
    - **The grid is declared before it runs** (:data:`TUNING_GRID`). A space
      chosen after seeing the winner is not a search.
    - **The arm is exploratory and reported whatever it shows** (root §13.2). It
      does not enter RQ1's ladder comparison, and it gets its own row.

    The probes are deterministic given the data, so a resumed session recomputes
    the same winner rather than needing it persisted. Its trial count enters
    root §13.5's development trial total, which is stated rather than concealed.

    Returns:
        The winning config under root §6.2's full schedule, and the ranked table.
    """
    device = device or pick_device()
    origin = ORIGINS[origin_index - 1]
    tensors = build_origin_tensors(features, origin, k, pred_len=PRED_LEN)

    rows: list[dict] = []
    for index, point in enumerate(TUNING_GRID):
        cfg = ITransformerConfig(
            pred_len=PRED_LEN,
            d_model=int(point["d_model"]), e_layers=int(point["e_layers"]),
        )
        spec = RunSpec("tuned", origin_index, k, PRED_LEN, SEEDS[0])
        _, outcome = train_one(
            tensors, spec, cfg, device=device,
            max_epochs=TUNING_EPOCHS, patience=TUNING_EPOCHS,
            lr=float(point["lr"]),
        )
        rows.append({**point, "val_mse": outcome.best_val_mse})
        log(f"  probe {index + 1}/{len(TUNING_GRID)} {point} -> {outcome.best_val_mse:.6f}")

    rows.sort(key=lambda r: r["val_mse"])
    best = rows[0]
    log(f"tuned config selected on origin {origin_index} validation: {best}")
    return (
        ITransformerConfig(
            pred_len=PRED_LEN,
            d_model=int(best["d_model"]), e_layers=int(best["e_layers"]),
        ),
        rows,
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
    # The *name*, not the module, for the reason given at the top of this file.
    # ``from itransformer_btc import metrics`` binds a module **object**, and the
    # flattened notebook has no such object — so ``metrics.clark_west_test`` was a
    # NameError six minutes into a Kaggle session while satisfying every check the
    # repository had (`D59`).
    from itransformer_btc.metrics import clark_west_test
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
    cw = clark_west_test(
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
    arms: tuple[str, ...] = ALL_ARMS,
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
    parser.add_argument("--arms", type=str, default=",".join(ALL_ARMS))
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
