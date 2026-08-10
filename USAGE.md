# USAGE — how to run this study, and what happens at each step

Operational companion to `README.md` (what the study is) and `CLAUDE.md` (**the law** — what may and
may not be done). This file answers a narrower question: *what do I type, in what order, and what
should come back?*

Where this file and `CLAUDE.md` disagree, **`CLAUDE.md` wins** and this file is the defect.

---

## 0. Two-minute orientation

The deliverable is a **manuscript**, not a model. Everything below exists to produce numbers that can
be regenerated on demand, because §12 forbids any figure entering the paper unless it resolves to a
persisted prediction file plus a config hash.

| | |
|---|---|
| One command to check the repo is sound | `python -m pytest tests/ -q` → **59 passed**, ~13 s, CPU only |
| One command to run one experiment | `python -m itransformer_btc.runner --arms fresh --shard 0 --shards 15` |
| One command after editing `src/` | `python tools/build_notebook.py` — the notebook carries the package (`D54`) |
| One notebook to run the whole study | `notebooks/iTransformer.ipynb` on Kaggle 2×T4. **Self-contained**: attach the data, nothing else |
| Where results land | `artifacts/preds/`, `artifacts/meta/`, `artifacts/paper_numbers.json` |

**Nothing in `data/raw/` is ever written to.** Four Stage 1 artifacts live there and are immutable;
every derived thing goes to `data/processed/` or `artifacts/`.

---

## 1. Install

Python **≥ 3.11**. The core dependency set is the data plane only — everything else sits behind a
named extra, so each dependency's reason is visible rather than ambient.

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate elsewhere

pip install -e ".[dev,train]"   # polars, pyarrow, numpy + torch + pytest
```

| Extra | Contents | When you need it |
|---|---|---|
| *(core)* | polars, pyarrow, numpy | Always — the data plane |
| `train` | torch | Local training. **Not installed on Kaggle** — the image ships its own |
| `ingest` | requests, pandas | Only to re-download klines (Stage 1) |
| `stats` | pandas, scipy, statsmodels, arch, wildboottest | The named stats boundary (§16). `metrics.py` needs scipy only for the Student-t tail, and falls back to a normal approximation without it |
| `dev` | pytest | Running the suite |

**Do not pin torch or polars against this venv when running on Kaggle.** Kaggle ships its own image;
the notebook installs only what is genuinely missing.

---

## 2. Repository map

```
invertedTransformer/
├── CLAUDE.md                       project law — read before anything
├── README.md                       what the study is
├── USAGE.md                        this file
├── spot_klines_btc.py              Stage 1 ingest (pandas — the one documented exemption)
├── data/raw/                       IMMUTABLE. four Stage 1 artifacts
├── data/processed/                 the writable half
├── docs/DIVERGENCE_REGISTER.md     every correction, with its evidence
├── docs/ORIGIN_WINDOW_BUDGET.md    per-origin window accounting — the assertion target
├── src/itransformer_btc/           the package. all logic lives here
├── tools/build_notebook.py         regenerates the notebook FROM src/ (D54)
├── tests/                          59 tests, CPU, ~13 s
├── notebooks/iTransformer.ipynb    the launcher — self-contained, GENERATED, never hand-edited
└── artifacts/{preds,meta,logs}/    + paper_numbers.json
```

**The notebook carries the package** (`D54`, `D58`). Twelve cells **define** each module directly in
the kernel namespace — no files, no package, nothing to import — so a Kaggle session needs the
notebook and `BTCUSDT_1h.parquet` and nothing else. (It used to write those files with
`%%writefile`; that existed only so two GPU *subprocesses* could import them, and `D57`'s measured
~30 s per run made the sequential path fit the budget, so the files bought nothing.)

Those cells are **generated from `src/`** and `tests/test_notebook_sync.py` asserts each equals its
module under the two declared removals — intra-package imports and the `__main__` guard — so the
workflow after any change under `src/` is:

```bash
python tools/build_notebook.py        # regenerate
python -m pytest tests/test_notebook_sync.py -q
git add src/ notebooks/iTransformer.ipynb
```

Hand-editing a package cell in the notebook is a defect; the next generator run reverts it silently.

### The package, module by module

| Module | Responsibility | Key guarantee it enforces |
|---|---|---|
| `config.py` | Design constants; the **derived** 15-origin grid; `FalsificationOrigin` | The grid is derived, never transcribed — a stale integer cannot survive a spacing change |
| `segments.py` | Loads the artifact; the segment law; break measurement | Raises if the frame reaches outside the half-open data window (`D33`) |
| `windows.py` | Timestamp-validated window enumeration | Raises on any window failing `t[s+L+H−1] − t[s] == (L+H−1) h` |
| `budget.py` | Per-origin/per-block window accounting | Root §11's **exact-equality** assertion target |
| `features.py` | The twelve variates, per bar, in ladder order | Raises on any non-finite variate — impossible unless the segment law did not run |
| `splits.py` | Window semantics per split; the scaler; the tensors | Raises if a training target reaches into validation (`D24`) |
| `model.py` | Encoder-only iTransformer + the uniform-attention arm | Returns the target channel alone, so the loss is single-channel by construction (`D39`) |
| `train.py` | Training loop; run identity; the two artifacts | GPU-resident, no `DataLoader`; LR halves every 4 epochs (`D47`) |
| `keff.py` | §5.4's pre-model measurement; the Stage 3b gate | Every K_eff is computed on a **training-only** span (`D44`) |
| `metrics.py` | §9 — every metric, test and RQ estimator | Ratio metrics come from seed-averaged MSEs, never averaged ratios (`D42`) |
| `runner.py` | The 534-run manifest; resume; budget guard; the 2-GPU launcher | Shards the **full** manifest, then subtracts what is done (`D53c`) |

`tools/build_notebook.py` sits outside the package on purpose: it is build tooling, not study logic,
and nothing under `src/` may import it.

---

## 3. The pipeline, stage by stage

Nine stages. Each names what it reads, what it writes, and what it will refuse to do.

```
  Stage 1  ingest ─────────► data/raw/*.parquet|csv|jsonl|json    [run once, already done]
     │
  Stage 2  segments + budget ──► assertions only
     │
  Stage 3  features ───────► 12 variates in memory
     │
  Stage 3b K_eff ──────────► artifacts/keff_table.parquet + the gate
     │
  Stage 4  invariants ─────► assertions only
     │
  Stage 5  pilot gate ─────► 12 main-grid runs, judged on VALIDATION
     │
  Stage 6  the grid ───────► artifacts/{preds,meta}/*  (534 runs)
     │
  Stage 7  evaluation ─────► RQ1, RQ2, RQ3
     │
  Stage 8  save ───────────► artifacts/paper_numbers.json
```

### Stage 1 — ingest (already done; do not re-run casually)

Produces the four immutable artifacts. **Re-running changes the vintage**, and §12 forbids comparing
numbers across vintages — every `meta/*.json` records the parquet sha256 it consumed.

```bash
python spot_klines_btc.py --self-test                          # 24 checks, no network
python spot_klines_btc.py --rebuild-only --outdir ./data/raw   # re-emit from retained JSONL
python spot_klines_btc.py --start 2018-01-01 --end 2026-08-01  # full download (hits the API)
```

`--end` is **exclusive**. `clip_to_window()` enforces that; the boundary bar at `2026-08-01T00:00`
once survived and shifted every count derived from `len(df)` (`D33`).

Expected artifact state:

| Field | Value |
|---|---|
| `bars_expected` / `bars_actual` | 75,216 / 75,094 |
| `coverage_pct` | 99.8378 |
| `missing_bars` / `gap_blocks` / `largest_gap_bars` | 122 / 27 / 33 |
| Last bar | `2026-07-31T23:00` |
| `BTCUSDT_1h.parquet` sha256 (first 16) | `8270a84b07c2923b` |

### Stage 2 — segments and the window budget

A **segment** is a maximal run of contiguous *usable* bars. The series breaks at (a) any missing bar
and (b) any zero-volume or `high == low` bar. Measured: exactly **3** unusable bars, and they are the
*same* 3 bars in all three senses — `2019-06-07T21:00`, `2021-02-11T03:00`, `2023-03-24T12:00`.

```python
from itransformer_btc.segments import load_bars, usable_mask
from itransformer_btc.budget import budget_table, COMMITTED_TRAIN_BUDGET

bars = usable_mask(load_bars())          # 75,094 rows; 75,091 usable
budgets = budget_table(bars)             # per origin, ~20 s
```

**The assertion is exact equality per origin**, never against the pooled 4.9%. Asserted pooled it
fires spuriously at fourteen of fifteen origins, gets loosened until it passes, and then can no
longer distinguish positional drift from ordinary between-origin variation — which is the one defence
against the highest-probability silent bug in the pipeline.

Regenerate `docs/ORIGIN_WINDOW_BUDGET.md` (and update `budget.py::COMMITTED_TRAIN_BUDGET` **in the
same commit**, never one alone):

```bash
python -c "import sys; sys.path.insert(0,'src'); \
  from itransformer_btc.segments import load_bars, usable_mask; \
  from itransformer_btc.budget import budget_table, format_markdown; \
  print(format_markdown(budget_table(usable_mask(load_bars()))))"
```

Two window semantics, differing by exactly 119 windows:

| Split | Semantics | Why |
|---|---|---|
| Train, validation | `"contained"` — whole window inside the span | The H-step target may not cross the boundary (§8.2, `D24`) |
| Test blocks | `"origin"` — only the *start* inside | §8.3 licenses the 96-bar lookback reaching back; blocking it makes evaluation unrealistically pessimistic. Every one of a block's **720** hours is an admissible forecast origin (`D51b`) |

### Stage 3 — features

```python
from itransformer_btc.features import build_features, ladder_columns
features = build_features(bars)          # 75,062 rows x 12 variates
```

75,091 usable bars → 75,062 feature rows. The 29 dropped rows are **one per segment**: `r` is computed
per segment, so each segment's first bar has no predecessor (`D52c`).

Column order **is** ladder order, so rung K is exactly the first K columns and `r` is channel 0 at
every rung:

| K | Columns |
|---|---|
| 1 | `r` |
| 4 | + `upper_shadow`, `lower_shadow`, `log_quote_volume` |
| 8 | + `log_trade_count`, `taker_buy_ratio`, `signed_flow`, `vwap_location` |
| 12 | + `log_parkinson`, `log_garman_klass`, `log_rogers_satchell`, `log_mean_trade_size` |

**No variate uses a rolling window.** That is a structural safety property, not a style choice: with
no rolling window anywhere, the `center=True` leak class is unrepresentable.

### Stage 3b — effective dimensionality, and the gate

```python
from itransformer_btc import keff
print(keff.gate_verdict(keff.gate_pr(features)))   # pre-first-origin span only
table = keff.keff_table(features)                  # ~50 s, 15 origins x 4 rungs
```

**Measured, and it does not match the design's expectation** (`D53f`):

| K | PR (raw) | PR (window-normalised) | Stable rank | Expected in §5.2 |
|---:|---:|---:|---:|---:|
| 1 | 1.00 | 1.00 | 1.00 | 1 |
| 4 | 3.33 | 3.32 | 2.36 | ~3.5 |
| 8 | **4.27** | 4.01 | 2.70 | ~6.5 |
| 12 | **3.98** | 3.65 | 2.17 | ~7 |

Gate PR at K=8 is **4.393 < 5.0**, so the gate does **not** pass. `D48`'s prescribed action is
**disclosure, not a ladder re-cut** — `D01` establishes there is no second consistent cut over F1–F5,
so "re-cut" named no reachable alternative. The grid proceeds unchanged and §4.1b reports it.

Two consequences are substantive rather than procedural: K=12's PR is *lower* than K=8's, so §5.2's
redundancy control is **stronger** evidence than designed; and `corr(K, K_eff) = 0.828` rather than
the ≈0.97 §9.1 feared, so RQ1's horse race is **more** identifiable than assumed.

### Stage 4 — pre-flight invariants

Three checks, each of which failed the first time it ran.

| Check | Correct form | The trap |
|---|---|---|
| `use_norm` scale invariance | `MSE(c·x)/c² == MSE(x)` | The source spec says `MSE(c·x) == MSE(x)`, which **cannot** pass: the target is a channel of the same array, so it scales too |
| Single-batch overfit | Run with **`dropout=0.0`** → ~1e-10 | With the configured 0.1 the loss floors near 7e-2, and a reader concludes the plumbing is broken when it is not (`D52d`) |
| Naive-RW first | `ŷ_z = −μ_g/σ_g` | `ŷ_z = 0` silently means `r̂ = μ_g`, a constant-drift model wearing the EMH baseline's name (`D31`) |

Measured `μ_g/σ_g` spans **−0.00818 … +0.01733** and **changes sign** across origins, so it is not a
constant a reader could mentally subtract — and it tracks the same bull/bear cycle H2 invokes as its
own mechanism.

### Stage 5 — the pilot gate, on validation

```python
from itransformer_btc import runner
pilot = runner.stage5_pilot(features)    # 12 runs: origin 1, 4 K x 3 seeds
print(pilot)
```

**Judged on the validation sub-block, never on test** (`D27`). §11 requires the test blocks be opened
once, after the design is frozen; a gate that repositions the title on a test-block result cannot
coexist with that. The twelve cells are ordinary main-grid `run_id`s, so Stage 6 skips them.

The statistic is **Clark–West, not Diebold–Mariano** (`D29`): K=1's feature set is a strict subset of
K=8's under the same architecture and sample, so the pair is nested and standard DM is systematically
undersized against exactly the alternative being tested.

**If the gate fails, reposition the title to the descriptive variant now — not in week nine.** First
evidence, one seed at origin 1: validation MSE 0.469075 (K=1) vs 0.467904 (K=8), Clark–West
`S* = +0.728, p = 0.233`.

### Stage 6 — the grid

**534 unique runs.** 582 nominal cells minus 48 that are literally the same run: the horizon sweep's
H=24 slice at seeds 42–44 shares `run_id`s with the main grid, and `run_id` **is** the identity of a
run (`D53e`).

| Arm | Runs | Composition | Tag |
|---|---:|---|---|
| main | 300 | 15 origins × 4 K × 5 seeds | `itr_` |
| uniform | 75 | 15 × K=8 × 5 seeds — attention forced uniform (`D50`) | `itru_` |
| fresh | 15 | one fresh model per origin at `o_i + 90 d` | `itrf_` |
| horizon | 144 | 4 named origins × 4 K × 4 H × 3 seeds, minus the 48 shared | `itr_` |

```bash
# one worker, one GPU (or CPU)
python -m itransformer_btc.runner --shard 0 --shards 1

# two workers, one per GPU — what the notebook does
CUDA_VISIBLE_DEVICES=0 python -m itransformer_btc.runner --shard 0 --shards 2
CUDA_VISIBLE_DEVICES=1 python -m itransformer_btc.runner --shard 1 --shards 2
```

| Flag | Default | Meaning |
|---|---|---|
| `--shard` / `--shards` | 0 / 1 | Round-robin **by group**, so a cell's seeds share one tensor build |
| `--arms` | `main,uniform,fresh,horizon` | Comma-separated subset |
| `--parquet` | `data/raw/BTCUSDT_1h.parquet` | Input artifact |
| `--out` | `artifacts` | Where the two files per run go |
| `--budget-h` / `--reserve-h` | 11.0 / 0.5 | Session budget, checked at **run boundaries** |

**Two GPUs are two independent run *processes*, not two threads.** `torch.manual_seed` seeds *every*
CUDA device, so two threads seeding concurrently would clobber each other's generator mid-run and
§12's reproducibility contract would be unenforceable. One process per GPU with
`CUDA_VISIBLE_DEVICES` pinned gives each worker its own global RNG, its own interpreter lock and
crash isolation.

**This CLI path is for a checkout, not for the notebook (`D58`).** Subprocesses import
`itransformer_btc`, and the notebook has no package to import — its modules are definitions in one
kernel namespace, so it runs the grid **in-kernel and sequentially** via `execute()`. That costs
roughly 4.5 h against 2.31 h and was accepted because `D57` showed both fit an 11 h session.
`launch_workers` remains here, tested, and is the path to use from a checkout — and the one a
1-minute grid will need, where sequential does not fit.

**`nn.DataParallel` is rejected**: at batch 32 the scatter/gather costs more than the split saves.
Parallelism belongs at the *run* level — the grid is many small runs, not one large one.

**No `Dataset`, no `DataLoader`, no workers.** At ~280k parameters the run is dominated by data
movement and Python overhead, which a per-item loader maximises — roughly 10× worse, which puts the
grid outside the 30 h weekly quota outright.

### Stage 7 — evaluation

```python
from itransformer_btc import metrics
done     = sorted(runner.completed_run_ids(runner.discover_roots()))
grid     = metrics.gather_grid(done, runner.discover_roots())
seed_avg = metrics.seed_average(grid)      # seed-averaged MSEs FIRST, ratios second
```

| RQ | Estimator | Functions |
|---|---|---|
| **RQ1** | Free rung effects + TOST equivalence on 8→12 + non-nested K vs K_eff | `tost_equivalence`, `j_test` |
| **RQ2** | `A(i,b) = [MSE_K1 − MSE_K8]/MSE_K1`, then `A = αᵢ + β₁b + ε` | `amplification`, `panel_beta1` |
| **RQ3** | `D(i,b)` on the skill scale, censored `b*`, Kaplan–Meier | `decay`, `DecayResult.b_star`, `kaplan_meier` |

Four things that are easy to get wrong and are handled for you:

1. **Order of operations.** Every ratio is formed from seed-averaged MSEs, never from an average of
   per-seed ratios (`D42`) — they differ by Jensen, and the second would require pairing seed 42 at
   K=1 with seed 42 at K=8, independent runs of different models where any of 5! orderings gives a
   different answer.
2. **The error bar is bound to the aggregation level** (`D30`). Per-cell → mean ± std across seeds.
   Anything aggregated across origins → **SE across origins**, with seed std reported separately as a
   Monte-Carlo diagnostic. Reporting seed std on an aggregated row understates headline uncertainty
   by roughly an order of magnitude.
3. **`β₁`'s effective n is the cluster count.** On a balanced panel with origin fixed effects, `β̂₁`
   reduces algebraically to the mean of the origin-specific within-slopes, so inference is a
   one-sample test on **G = 15** numbers — and effective independence is bounded near **4** by the
   79.2% training-window overlap between consecutive origins (`D28`).
4. **The bootstrap is restricted, studentized, one-sided, B = 99,999**, both Rademacher and Webb
   weights, with the more conservative as the headline. Its p-value floors at `1/(B+1)`, because no
   finite bootstrap supports a literal zero (`D53d`).

### Stage 8 — save

`artifacts/paper_numbers.json` plus five parquet tables. **Every table and figure in the manuscript
is generated from that file, never transcribed.**

---

## 4. Running on Kaggle

`notebooks/iTransformer.ipynb`, 36 cells (25 code), load through evaluation. **Self-contained**: its
twelve module cells define the package in the kernel namespace, so nothing is imported and the
repository is not uploaded (`D54`, `D58`).

Setup:

1. Upload `data/raw/` as a Kaggle Dataset (must contain `BTCUSDT_1h.parquet`; include
   `BTCUSDT_1h_report.json` so the input digest is read from Stage 1 rather than recomputed).
2. New Notebook → attach that one Dataset → **Accelerator: GPU T4 ×2**.
3. Upload `iTransformer.ipynb` (File → Import Notebook), or paste it. Discovery is by **globbing**,
   never by dataset slug, so the Dataset can be renamed freely.
4. **Save Version → Save & Run All.** Never the interactive editor.

**Do not attach the repository**, and do not add a `src/` to `sys.path`. The notebook asserts that
the `itransformer_btc` it imported lives in its own working directory and fails loudly otherwise:
two copies of the code, one of them stale, is the failure the assertion exists to catch.

**Why never the editor:** the 20-minute idle timeout kills long interactive sessions, and hitting
Kaggle's own 12 h wall interactively loses `/kaggle/working` **entirely**.

**Session chaining.** Session *N* writes to `/kaggle/working`; Save Version publishes it as a
Dataset; session *N+1* attaches that Dataset as an input. Resume is automatic and needs no manual
bookkeeping: a run is complete only when **both** artifacts exist and `meta.status == "complete"`.

### If the session ends mid-grid — the expected case

Written when the grid was estimated at ~10–20 wall hours against an 11 h budget, which made two
sessions the plan rather than the accident. **Measured, the 534-run grid took 2.31 h** on two T4s and
~4.5 h in one kernel (`D57`), so one session is now the expected case and this section is the
contingency it was built to be. It still holds verbatim, and a 1-minute grid will need it. Nothing
restarts from zero:

| | |
|---|---|
| What is lost | At most **one run** — the one in flight, ~30 s measured. An interrupted run leaves no `meta`, so it is redone |
| What is kept | Every completed `preds/{run_id}.parquet` + `meta/{run_id}.json`, plus `keff_table.parquet` and `naive_rw_by_origin.parquet` |
| How the next session knows | `pending()` = manifest − completed, matched by `run_id`. Roots are discovered by **glob** over `/kaggle/input/*/preds` and `/kaggle/input/*/*/preds`, so the Dataset's name and nesting do not matter |
| What you do | Save Version → attach that output as the next session's input → run again |

Two guards make that hold, and both were added on 2026-08-07 (`D54e`, `D54f`):

- **The evaluation cells are gated on `GRID_COMPLETE`.** A partial grid is an unbalanced panel and
  §9.1's estimators refuse one *by design* — `amplification` raises rather than compare K=1 at eleven
  origins against K=8 at ten. That is correct and stays; what changed is that the notebook no longer
  *calls* them until the panel exists, so a partial session prints its resume instructions and ends
  cleanly instead of erroring in its last cells. **Partial evaluation is never offered**: a
  half-panel `β₁` is a different estimand, not a noisier one.
- **The budget guard bounds the session, not the worker.** Kaggle's 12 h wall starts at cell 0, so
  the prelude — data, `K_eff`, invariants, and the twelve pilot runs, ~20–25 min — is subtracted
  before the grid gets its budget. Unchanged by `D58`: the arithmetic never depended on where the
  grid ran, only on when the session started.

**Do not use the interactive editor.** The 20-minute idle timeout kills the session, and hitting the
12 h wall interactively loses `/kaggle/working` **entirely** — that is the one way to actually lose
completed runs.

| Limit | Value | Consequence |
|---|---|---|
| Session runtime | 12 h | Budget guard stops at 10.5 h so the version saves |
| GPU quota | 30 h / week | The whole grid must fit in roughly one week |
| `/kaggle/working` | 20 GB | Predictions total ≈ 0.5–2 GB — fits with room |
| `/kaggle/input` | read-only | Everything is *written* to `/kaggle/working` |

**Timing is not yet confirmed on a T4.** Measured locally: **97.8 s per run, 9.8 s/epoch, on CPU with
6 threads** — no CUDA device was available. §10.3's 60–100 s per-run figure remains an estimate.
Extrapolating, 534 runs ÷ 2 workers ≈ 7 wall-hours, inside one weekly quota. **Take the real T4
number on the first session and replace §10.3's block.**

---

## 5. Artifacts and their schemas

### `artifacts/preds/{run_id}.parquet` — one row per (block, window, step)

| Column | Type | Meaning |
|---|---|---|
| `block` | `Int8` | Test block, 1–6. The falsification arm carries **4, 5, 6** — labels, not positions |
| `step` | `Int16` | Forecast step, 1…H |
| `timestamp` | `Int64` | Epoch ms UTC of the **window start**. The first target hour is `start + L` |
| `y_true` | `Float32` | Standardised log-return |
| `y_pred` | `Float32` | Standardised log-return |

`timestamp` is the window start, not the forecast origin. Since `L = 96` is a multiple of 24,
hour-of-day is preserved, which is what makes the 00:00 UTC non-overlapping phase (`D46`) selectable
by `timestamp // 3.6e6 % 24 == 0`.

### `artifacts/meta/{run_id}.json`

Resolved config, `git_sha`, **`code_sha256`**, **`input_parquet`**, `input_sha256`,
**`input_sha256_source`**, `origin`, `origin_index`, `block_labels`, `k`, `variates`, `n_train`,
`n_val`, `n_test_per_block`, `mu_g`, `sigma_g`, `mu_over_sigma`, `naive_rw_z`, `epochs_run`,
`best_val_mse`, `train_loss`, `wall_time_s`, `n_parameters`, `device`, `status`.

The three bold fields are `D54`. `git_sha` is `"unknown"` on Kaggle — there is no git repository
there — so `code_sha256`, the digest of the package source with line endings normalised, is what
identifies the code §12 requires every number to resolve to. `input_sha256_source` is `"report"` when
the digest came from the Stage 1 report beside the artifact and `"file-digest"` when the parquet was
hashed directly; both are correct, and stating which is what makes the number checkable.

### Run identity

```
run_id = {model}_o{origin:02d}_K{K:02d}_H{H:03d}_s{seed}
         e.g.  itr_o07_K08_H024_s42     itru_o01_K08_H024_s43     itrf_o01_K08_H024_s42
```

Changing any component deliberately **orphans** prior outputs rather than silently reusing a
mismatched result.

---

## 6. Expected numbers — use these to tell a break from a change

| Quantity | Value |
|---|---|
| Tests | **53 passed**, ~18 s |
| Bars / usable / unusable | 75,094 / 75,091 / **3** (the same 3 bars in all three senses) |
| Feature rows | 75,062 (29 dropped, one per segment) |
| Origins | 15, 2020-01 … 2025-11, 5-month spacing |
| Training windows per origin | 13,558 … 15,217 (raw frame) · **13,545 … 15,217** (feature frame — the assertion target) |
| Largest training tensor | **70.12 MB** |
| Parameters | **280,472**, identical at every rung |
| Test-block survival | 0% loss at 74 of 90 cells; worst 439/720 |
| Gate PR at K=8 | **4.393** (< 5.0 — disclose, do not re-cut) |
| First measured run | `itr_o01_K08_H024_s42`: RelMSE **1.0183**, `R²_oos` **−0.0183** |

**That last row is worth watching.** `D20` anticipates `R²_oos ≈ +0.004`; the first measurement is
negative and four times larger in magnitude. If it survives the grid, RQ2's `A(i,b)` becomes a ratio
of two *negative* skills, and §9.1's guard on `R²_oos(i,1) ≤ 0` stops being an edge case.

---

## 7. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `FileNotFoundError: data/raw/BTCUSDT_1h.parquet` | Artifacts absent, or still in `data/` | `python spot_klines_btc.py --rebuild-only --outdir ./data/raw` |
| `ValueError: … reaches outside the half-open data window` | The boundary bar is back (`D33`) | Re-emit with `--rebuild-only`, which applies `clip_to_window()` |
| `AssertionError: budget drift against the committed table` | The artifact changed, or the origin grid moved | Regenerate §2's table **and** `COMMITTED_TRAIN_BUDGET` in one commit |
| `ValueError: non-finite variate values` | The segment law did not run | Pass `usable_mask(...)` output into `build_features` |
| `ValueError: … the purge did not hold` | A training target reaches into validation (`D24`) | A real leak. Do not loosen the assertion |
| `ValueError: unbalanced panel` | Grid incomplete — some (origin, block) cells missing | Finish the grid; `β₁`'s within-slope identity needs balance |
| `ValueError: the HLN factor is … <= 0` | T too small for h (exactly 0 at T=24, h=24) | Correct behaviour. §9.2 refuses to report there — state T instead |
| `FileNotFoundError: N of M runs are absent` | `gather_grid` was given ids without artifacts | Pass the completed subset, or finish the grid. A silently short grid mixes arms in one table |
| Workers overlap or leave gaps | Sharding the *pending* list (`D53c`) | Shard the full manifest, then subtract what is done |
| A result looks too good | **Leakage, until proven otherwise** | Hunt for it. §11's checklist, then the boundary × channel grid in §5.3 |

---

## 8. What is not built yet

Stated plainly, because a reader should not have to discover it by running:

- **Comparison baselines** — ARIMA, LSTM, DLinear, PatchTST and ridge (~255 runs). Naive-RW *is*
  available, computed from the scaler, and every RelMSE divides by it, so **RQ1, RQ2 and RQ3 are all
  answerable without them**. The baselines are for Table 4's positioning and Table 6's DM matrix.
- **The economic evaluation** of §13.5 — the sign rule, per-segment holding returns, the cost band
  and the per-origin DSR.
- **The optional K=16 rung** (`D22`), whose run condition is origin 1's Stage 5 gate passing at
  α = 0.05 **and** ≥ 5 GPU-hours of weekly quota remaining. Running it reopens the no-embargo
  argument in §8.3, which must be re-derived first.
- **A confirmed T4 timing.** Everything measured so far is CPU.

---

## 9. Before you change anything

1. Run the tests. They found `D51`, `D52` and `D53`, and several assert a claim that was false the
   first time it ran.
2. Read `CLAUDE.md` §2's hard constraints. No TensorFlow/Keras/JAX. No pandas in the data plane. No
   imputation, ever. No winsorizing. No centred windows. No scaler fitted outside the 21-month
   sub-block.
3. If you find a contradiction between documents, it becomes **`D54`** and goes in
   `docs/DIVERGENCE_REGISTER.md` with its evidence. Absorbing one silently is the exact failure that
   register exists to prevent.
