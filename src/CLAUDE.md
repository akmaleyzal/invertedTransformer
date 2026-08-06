# src/ — code-local rules

Root `CLAUDE.md` is the project law. This file adds only what is specific to writing code here.

## Frameworks

**PyTorch is the only deep-learning framework.** Do not add, import, or suggest **TensorFlow,
Keras, or JAX**. A reference implementation that exists only in TF gets ported to torch idioms, not
wrapped.

## The polars boundary

**polars is the data plane.** Segmentation, window enumeration, split generation, and feature
construction use polars lazy scans (`pl.scan_parquet`, pushed-down filters). pandas does not appear
in any of them — **and this rule is absolute inside `src/`.**

**pandas is permitted at exactly one boundary here:** converting to numpy or pandas for a statistics
library that accepts nothing else — `statsmodels` (ADF), `arch` (VarianceRatio), `wildboottest`.
That boundary is a **named function** (e.g. `to_stats_frame`), not scattered `.to_pandas()` calls,
so it can be found by search and reviewed as a unit.

**Stage 1 ingest is exempt, and it lives outside `src/`.** `spot_klines_btc.py` at the repository
root is pandas by design (root §2, §16): it computes no rolling window, so the correctness argument
below has nothing to bite on. That exemption is scoped to that one file. It is **not** a precedent
for `src/` — anything reading `data/raw/*.parquet` from here on is polars.

**This is a correctness argument, not only a speed one.** polars' rolling API is backward-closed by
construction, so the `center=True` leak that root §2 forbids is *unrepresentable*; in pandas it is
one keyword away. The source specification's §6.2 purge snippet is written in pandas and must be
**re-expressed in polars/numpy**, not copied.

Note also that root §5.3 means **no feature uses a rolling window at all** — every variate is a
per-bar function. If you find yourself reaching for any rolling API, stop: either you are
implementing something outside the twelve variates, or you have misread §5.1.

## Training loop

The training tensor for one origin is **70.12 MB** — `15,217 × 96 × 12 × 4 B`, sized at the largest
origin's 21-month sub-block and sliced per origin (`D25`; the count varies 13,558 … 15,217, see
`docs/ORIGIN_WINDOW_BUDGET.md`). **~80 MB is the 24-month figure and is wrong — it silently assumes
training runs on the validation months too, which is `D24`'s leak.** Load it to GPU once and batch by
index-slicing it.

- **No `Dataset`. No `DataLoader`. No workers.** At ~280k parameters the run is dominated by data
  movement and Python overhead, which a per-item loader maximises. Root §10.3 gives the numbers.
- Shuffle by permuting an index tensor on device, not by moving data.
- Training touches no DataFrame. Features arrive as a pre-built `float32` tensor.
- **Two GPUs = two independent run workers**, one pinned per `cuda:N`, pulling from a shared run
  queue. Not `nn.DataParallel` — at batch 32 the scatter/gather costs more than it saves.
- Device-agnostic code. Never hard-code `.cuda()`. Gate precision on
  `torch.cuda.get_device_capability(0)[0] >= 8`, **never** on `torch.cuda.is_bf16_supported()`,
  which returns True on the T4 via emulation and selects a path slower than fp32.

## Structure

Logic lives here as an **importable package**, unit-testable on CPU. A notebook imports it and calls
it; see `notebooks/CLAUDE.md`.

- Python ≥ 3.11 syntax, type hints on every public function, Google-style docstrings.
- Config in YAML loaded into dataclasses. **No magic numbers in code.**
- Comments explain *why*. Match the density of surrounding code.
- Fail loudly. A schema mismatch, a window-count mismatch, or a hash mismatch raises — it does not
  warn and continue. Root §11's checklist items are `assert`s wherever they can be.

## Every run writes two files

`preds/{run_id}.parquet` and `meta/{run_id}.json`, per root §10.4. **Raw predictions always** — they
are required for the DM test, per-regime analysis, and the economic evaluation, and re-running the
grid because only metrics were saved is an expensive, avoidable mistake. A run is complete only when
both exist and `meta.status == "complete"`; anything else is re-run from scratch.

## What exists here now

```
itransformer_btc/config.py     design constants, the derived 15-origin grid, FalsificationOrigin
itransformer_btc/segments.py   segment law, break measurement, artifact loading
itransformer_btc/windows.py    timestamp-validated window enumeration
itransformer_btc/budget.py     per-origin accounting — root §11's assertion target
itransformer_btc/features.py   the twelve variates, per-bar, in ladder order
itransformer_btc/splits.py     window semantics per split, the scaler, the tensors
itransformer_btc/model.py      encoder-only iTransformer + the uniform-attention arm
itransformer_btc/train.py      training loop, run identity, the two artifacts
itransformer_btc/keff.py       §5.4's pre-model measurement — RQ1's regressor and the Stage 3b gate
itransformer_btc/metrics.py    §9 — RelMSE, R²_oos, DA/PT, A, A_attn, D, b*, DM/CW, β₁ WCR, TOST, J
itransformer_btc/runner.py     the 534-run manifest, resume, budget guard, two-GPU launcher
```

`tests/test_data_plane.py` runs the assertable half of root §11 against the real artifact;
`tests/test_model_plane.py` checks what root §5, §6 and §8 claim about the mathematics;
`tests/test_experiment_plane.py` checks what §9 and §10 claim about the grid. **53 tests**, all on
CPU in ~18 s. Run them before anything else — between them they found `D51`, `D52` and `D53`, and
several assert a claim that was false the first time it ran.

**Two GPUs are two independent run *processes*.** Threads are wrong here for a reason worth stating
once: `torch.manual_seed` seeds **every** CUDA device, so two threads seeding concurrently clobber
each other's generator mid-run and root §12's reproducibility contract becomes unenforceable. One
process per GPU with `CUDA_VISIBLE_DEVICES` pinned gives each worker its own global RNG, its own
interpreter lock and crash isolation, and costs one feature-frame rebuild per worker.

**Shard the full manifest, then subtract what is done — never the reverse** (`D53c`). Sharding the
*pending* list makes the partition a function of how many runs happen to be complete at that instant,
so two workers starting seconds apart get partitions that are not complementary.

Still missing: the **comparison baselines** — ARIMA, LSTM, DLinear, PatchTST and ridge. Naive-RW is
already computable from the scaler and is what every RelMSE in `metrics.py` divides by, so RQ1, RQ2
and RQ3 are all answerable without them; the baselines are for Table 4's positioning and Table 6's DM
matrix, a separate ~255 runs. Also missing: the economic evaluation of §13.5.

**One run has been executed end to end**: `itr_o01_K08_H024_s42`, 97.8 s on **CPU** — no CUDA device
is available locally, so §10.3's 60–100 s per-run figure is still a T4 estimate and is **not**
confirmed. The transferable number is **9.8 s/epoch**. Take the T4 measurement on the first Kaggle
session.
