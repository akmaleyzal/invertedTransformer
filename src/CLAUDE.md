# src/ — code-local rules

Root `CLAUDE.md` is the project law. This file adds only what is specific to writing code here.

## Frameworks

**PyTorch is the only deep-learning framework.** Do not add, import, or suggest **TensorFlow,
Keras, or JAX**. A reference implementation that exists only in TF gets ported to torch idioms, not
wrapped.

## The polars boundary

**polars is the data plane.** Ingest, validation, segmentation, and feature construction use polars
lazy scans (`pl.scan_parquet`, pushed-down filters). pandas does not appear in any of them.

**pandas is permitted at exactly one boundary:** converting to numpy or pandas for a statistics
library that accepts nothing else — `statsmodels` (ADF), `arch` (VarianceRatio), `wildboottest`.
That boundary is a **named function** (e.g. `to_stats_frame`), not scattered `.to_pandas()` calls,
so it can be found by search and reviewed as a unit.

**This is a correctness argument, not only a speed one.** polars' rolling API is backward-closed by
construction, so the `center=True` leak that root §2 forbids is *unrepresentable*; in pandas it is
one keyword away. The source specification's §6.2 purge snippet is written in pandas and must be
**re-expressed in polars/numpy**, not copied.

Note also that root §5.3 means **no feature uses a rolling window at all** — every variate is a
per-bar function. If you find yourself reaching for any rolling API, stop: either you are
implementing something outside the twelve variates, or you have misread §5.1.

## Training loop

The training tensor for one origin is ~80 MB. **Load it to GPU once and batch by index-slicing it.**

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

## Before writing pipeline code

The environment is unresolved (root §16): `pyproject.toml` declares `requires-python >= 3.14` with
three dependencies, none of them torch, and torch wheel availability on 3.14 is unverified.
`requirements.txt` is a UTF-16 dump of the superseded project. Resolve this first.
