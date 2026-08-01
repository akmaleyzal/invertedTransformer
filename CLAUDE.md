# CLAUDE.md

Guidance for Claude Code (and any human contributor) working in this repository.

---

## 1. Project Definition

**Deliverable:** a production-grade **deep learning model implemented in PyTorch** using the
**inverted Transformer (iTransformer)** architecture, which forecasts **Bitcoin (BTC/USDT) at
1-minute granularity**, conditioned on three exogenous information sources of *different sampling
frequencies*:

| Role | Series | Native granularity | Source file |
| --- | --- | --- | --- |
| **Endogenous target** | BTC/USDT OHLCV (Binance) | 1 minute | `data/raw/btc_usdt_binance_{2018..2026}.parquet` |
| Exogenous (fast) | XAU/USD Gold OHLCV | 1 minute | `data/raw/xauusd_2018_2026.parquet` |
| Exogenous (slow) | FED Broad trade-weighted US Dollar Index | 1 day | `data/raw/US_Dollar_Index_2018_2026.parquet` |
| Exogenous (very slow) | US Macroeconomics (31 indicators) | 1 month | `data/raw/fed_economic_data_2018_2026.parquet` |

**Hard constraints (non-negotiable):**

1. **PyTorch only.** `torch` is the sole deep learning framework. **Do not add, import, or suggest
   TensorFlow, Keras, JAX, or any Google-authored DL framework.** If a reference implementation
   exists only in TF, port it to PyTorch idioms rather than adding the dependency.
2. **The core model must be the inverted Transformer (iTransformer).** Comparative baselines are
   allowed and encouraged, but they are baselines — iTransformer is the shipped artifact.
3. **Zero look-ahead leakage.** Every transformation must be causal. This is the single most common
   way financial forecasting projects produce fraudulent-looking results. See §16.
4. **Multi-granularity fusion is the central engineering problem**, not an afterthought. See §6.

---

## 2. Environment & Stack

Virtual environment lives at `.venv/` in the repo root. On Windows, the bare `python` command
resolves to the Microsoft Store stub — **always invoke the venv interpreter explicitly**:

```powershell
& "D:\pythonProject\invertedTransformer\.venv\Scripts\python.exe" -m <module>
```

Pinned versions already present in `requirements.txt` (do not silently upgrade):

| Package | Version | Use |
| --- | --- | --- |
| `torch` | 2.13.0 | **the only** DL framework |
| `polars` | 1.43.1 | primary dataframe engine (lazy scans over 4.4M-row parquet) |
| `pandas` | 3.0.5 | interop / `merge_asof` where polars `join_asof` is awkward |
| `duckdb` | 1.5.5 | ad-hoc SQL over parquet without materialising |
| `numpy` | 2.5.1 | memmap-backed feature matrices |
| `scikit-learn` | 1.9.0 | scalers, metrics, `TimeSeriesSplit` scaffolding |
| `optuna` | 4.9.0 | hyperparameter search (TPE + pruning) |
| `scipy`, `matplotlib`, `seaborn`, `tqdm` | — | statistics, diagnostics, plots, progress |

Missing but expected to be added when needed (ask before installing):
`pyarrow` (parquet engine for pandas), `onnx` + `onnxruntime` (export verification),
`statsmodels` (Diebold–Mariano, ADF/KPSS tests), `numba` (fast rolling estimators).

**Device policy:** write device-agnostic code (`device = torch.device("cuda" if
torch.cuda.is_available() else "cpu")`). Never hard-code `.cuda()`. Use `torch.autocast` with
`bfloat16` **only when compute capability ≥ 8.0** (`torch.cuda.get_device_capability(0)[0] >= 8`),
otherwise `float16` + `GradScaler`, otherwise plain `float32`. Do **not** gate this on
`torch.cuda.is_bf16_supported()` — see §2.1 for why it lies on the T4.

The pinned versions above describe the local venv. **Kaggle ships its own image**, so the
notebook must run against whatever `torch` and `polars` are already there and only pip-install
what is genuinely missing.

---

### 2.1 Execution environment — Kaggle T4 ×2

Training happens on Kaggle's free tier, and its limits are not a footnote — they shape the
architecture of the run. Full operating manual: **`docs/KAGGLE_GUIDE.md`**.

| Limit | Value | Why it matters |
| --- | --- | --- |
| Session runtime | **12 h** (CPU & GPU), 9 h TPU | The `full` profile does **not** fit in one session |
| GPU quota | **30 h / week / account** | The complete programme is ~25–40 GPU-hours ⇒ ~2 calendar weeks |
| Idle timeout (interactive) | **20 min** | Long stages must run via *Save Version → Save & Run All*, not the editor |
| `/kaggle/working` | **20 GB**, saved as version output | ~2–4 GB of checkpoints across all stages — fits, but housekeep |
| RAM / CPU | **~29 GB**, 4 cores | Feature hygiene peaks at 8–12 GB; DataLoader is CPU-bound at 4 cores |
| `/kaggle/input` | read-only | Resume reads from here; checkpoints are always *written* to `/kaggle/working` |

**Design consequences already implemented in the notebooks:**

- **Preprocessing left Kaggle entirely.** It needs no GPU but used to run in every GPU session,
  and it was the session's RAM ceiling. It now runs locally in `01_preprocess.ipynb`, so the
  30 h/week quota buys only training. See §3.2.
- **Staged execution.** `run_baselines`, `run_ablation`, `run_walkforward` are stage switches.
  Run one stage per session; per-tag checkpoints carry over.
- **Self-imposed budget.** `session_budget_hours` (default 11) + `reserve_hours` (0.5) stop
  training at an epoch boundary so the checkpoint is written and the version can be saved.
  Hitting Kaggle's own 12 h wall in interactive mode loses `/kaggle/working` entirely.
- **Cross-session resume.** `KAGGLE_RESUME_DIR` points at the previous session's output
  dataset. Checkpoints are read from `RESUME_DIR`, written to `CKPT_DIR`. `run_id` is
  `{profile}_L{seq_len}_H{pred_len}_d{d_model}_s{seed}` — changing any of those five values
  deliberately orphans the old checkpoints rather than silently loading a mismatched model.
- **Dataset auto-discovery.** `/kaggle/input` is searched for the folder holding the frozen
  artifact (`02_train`) or the 12 raw files (`01_preprocess`, only in the CPU-session fallback),
  so the Kaggle Dataset slug does not have to match a hard-coded path.
- **Session RAM is no longer the constraint.** A training session holds ~1 GB of `float32`
  features instead of 8–12 GB of polars frames, so the ~29 GB host RAM is now slack rather than
  a limit.

**Hardware traps confirmed on this platform:**

- `torch.cuda.is_bf16_supported()` defaults to `including_emulation=True` and returns **True on
  the T4 (sm_75)**, because a bf16 *tensor* can be allocated there even without a bf16
  tensor-core path. Trusting it selects **emulated bf16, slower than fp32**. Detect with
  `torch.cuda.get_device_capability(0)[0] >= 8` instead, and fall back to fp16 + `GradScaler`.
- Per-epoch cost at the `full` profile is dominated by **data movement (~91 MB/batch)**, not
  model size, so a `DLinear` baseline costs nearly what the iTransformer costs. Budget for it.
- `nn.DataParallel` (not DDP) is the right call inside a Kaggle notebook — DDP needs process
  spawning. Enter `torch.autocast` in the main thread; it propagates into DataParallel's
  worker threads. Always checkpoint `model.module.state_dict()`.

---

## 3. Target Repository Layout

Build toward this structure. Create directories as work reaches them; do not scaffold empty stubs
en masse.

```
invertedTransformer/
├── CLAUDE.md                      # this file
├── README.md
├── requirements.txt
├── configs/
│   ├── data.yaml                  # paths, date ranges, alignment policy
│   ├── features.yaml              # feature block toggles + windows
│   ├── model_itransformer.yaml    # architecture hyperparameters
│   └── train.yaml                 # optimiser, schedule, early stopping
├── data/
│   ├── raw/                       # IMMUTABLE. never write here.
│   ├── interim/                   # aligned master minute grid (parquet)
│   └── processed/                 # FROZEN FEATURE ARTIFACTS — see §3.2
│       └── features_{profile}/    # features.npy, close.npy, timestamps.npy,
│                                  # scaler.json, feature_manifest.json, prep_metadata.json
├── src/
│   ├── data/
│   │   ├── loaders.py             # per-source typed loaders + schema validation
│   │   ├── calendars.py           # UTC master grid, session masks, holiday logic
│   │   ├── align.py               # as-of joins, staleness, release-lag application
│   │   └── dataset.py             # torch Dataset / Sampler over the master grid
│   ├── features/
│   │   ├── price.py               # returns, frac-diff, OHLC volatility estimators
│   │   ├── volume.py              # liquidity, Amihud, VWAP deviation
│   │   ├── cross_asset.py         # BTC↔Gold correlation/beta/lead-lag
│   │   ├── macro.py               # release-lagged macro, surprises, regimes
│   │   └── temporal.py            # cyclical time encodings, session dummies
│   ├── models/
│   │   ├── layers.py              # DataEmbedding_inverted, Attention, EncoderLayer
│   │   ├── itransformer.py        # the model
│   │   └── baselines.py           # RandomWalk, DLinear, vanilla Transformer
│   ├── training/
│   │   ├── losses.py              # Huber, pinball, directional-aware
│   │   ├── trainer.py             # loop, AMP, clipping, checkpointing
│   │   └── splits.py              # purged + embargoed walk-forward splits
│   ├── evaluation/
│   │   ├── metrics.py             # MSE/MAE/MASE/DA/DM-test
│   │   ├── backtest.py            # cost-aware economic evaluation
│   │   └── report.py              # tables + figures
│   ├── export/
│   │   ├── to_torchscript.py
│   │   ├── to_onnx.py
│   │   └── bundle.py              # weights + scaler + config + metadata
│   └── utils/
│       ├── seed.py, logging.py, io.py
├── scripts/
│   ├── 01_build_master.py
│   ├── 02_build_features.py
│   ├── 03_train.py
│   ├── 04_tune_optuna.py
│   ├── 05_evaluate.py
│   └── 06_export.py
├── docs/
│   └── KAGGLE_GUIDE.md            # how to run the two-notebook flow inside Kaggle's limits
├── tools/
│   ├── build_split_notebooks.py   # partitions iTransformer.ipynb into 01 + 02, byte-identically
│   └── split_cells/               # the injected cells: artifact contract, freeze, discover, load
├── notebooks/
│   ├── iTransformer.ipynb         # REFERENCE + equivalence yardstick — see §3.1
│   ├── 01_preprocess.ipynb        # SHIPPED, runs LOCALLY — raw → frozen artifact
│   └── 02_train.ipynb             # SHIPPED, runs on KAGGLE GPU — artifact → model
└── artifacts/
    ├── checkpoints/  runs/  reports/  models/
```

### 3.1 Where the pipeline actually lives right now

`src/` and `scripts/` do not exist yet. The whole pipeline is implemented in notebooks, because
training runs on Kaggle's free T4 ×2 and a notebook is what Kaggle executes. Since 2026-07-30 it
lives in **two** notebooks, split on the line between what needs a GPU and what does not:

| Notebook | Runs on | Does |
| --- | --- | --- |
| `01_preprocess.ipynb` (42 cells) | **local machine** | load → validate → gold tz → align → features → hygiene → **freeze + verify artifact** |
| `02_train.ipynb` (53 cells) | **Kaggle GPU** | load + verify artifact → split → window → train → gate → evaluate → backtest → ablate → export |
| `iTransformer.ipynb` (70 cells) | either | **reference implementation and equivalence yardstick.** Not the shipped path. |

**Neither notebook is edited by hand.** Both are generated:

```powershell
& $PY tools/build_split_notebooks.py
```

The script copies source cells **byte-identically**, splits exactly two of them at text anchors,
and injects the new cells from `tools/split_cells/`. It asserts that every one of the 70 source
cells is routed to a notebook, split, or explicitly replaced — so a source cell cannot go
missing silently. Cells 3–8 (imports/device/theme), cell 11 (`Config`), the `UTC`/`ts` fragment,
and the artifact-contract cell are duplicated **on purpose**; see §3.2.

Consequences for anyone working here:

- **Edit `iTransformer.ipynb`, then rebuild.** Editing `01_preprocess.ipynb` or
  `02_train.ipynb` directly puts them out of sync with their source and the next rebuild
  silently discards the change.
- **The notebooks are production code, not exploration.** The "notebooks are exploration only"
  convention applies to *other* notebooks. Changes get the same scrutiny `src/` would: causal
  features, train-only statistics, gates before results.
- **Verify by executing, not by reading.** `PROFILE = "tiny"` runs `01` then `02` on CPU in a
  few minutes over three months of data. Any edit should be followed by both runs ending with
  `ALL CELLS OK` and `ALL GATES PASS`. An AST parse is not verification: most defects here are
  shape, unit, or ordering errors that only appear at runtime.
- **Equivalence is measured, not assumed.** `features.npy` from `01` is **bit-identical** to
  the single notebook's `X` at the `tiny` profile (`sha256 c3ad8cd3…`), with timestamps and
  scaler equal. Re-check this after changing anything in `01`.
- When `src/` is eventually extracted, `iTransformer.ipynb` is the reference to port *from*,
  and the port must reproduce its `tiny`-profile numbers exactly.

### 3.2 The frozen-artifact contract

`01` and `02` communicate through one directory and nothing else. `02` **never opens
`data/raw/`**.

| File | Content | Why this form |
| --- | --- | --- |
| `features.npy` | `float32 (T, N)` | what the model consumes; standardised, variate order fixed |
| `close.npy` | `float64 (T,)` | raw close for price reconstruction — must never be derived from a standardised column |
| `timestamps.npy` | `int64` epoch-µs UTC | every split boundary is an integer comparison; an integer cannot carry a timezone it forgot to declare |
| `scaler.json` | mean/std/winsor bounds + `fitted_on` | records *which split* the statistics came from, alongside the statistics |
| `feature_manifest.json` | variate order, groups, `target_index`, `fracdiff_d`, gold offset, release-lag table, dropped columns, PCA rank | the data-dependent choices `02` cannot recompute without the raw data |
| `prep_metadata.json` | raw-file hashes, `features_sha256`, `manifest_sha256`, `scaler_sha256`, **frozen fields**, library versions | the chain of evidence `02` verifies |

**Frozen fields** — `CFG` fields that shape the matrix. A training session that disagrees on any
of them is stopped:

```
profile, grid_start, grid_end, train_end, val_end, test_end,
seq_len, pred_len, blocks, macro_n_pca, fracdiff_grid, fracdiff_width,
winsor_q, collinear_thresh, gold_utc_offset_h
```

`train_end` is frozen because **the scaler is fitted on rows `t <= train_end`** — moving it on
the training side is a leak, not a mismatch. `seq_len` is frozen because warm-up truncation is
`1440 + seq_len + 60`, so it decides which rows exist at all.

**Free fields** — anything a session may vary without rebuilding: `d_model`, `n_heads`,
`e_layers`, `d_ff`, `dropout`, `lr`, `weight_decay`, `batch_size`, `epochs`, `loss`, `seed`,
and every stage/budget switch.

**Six rejection rules**, all hard `assert`, printed as a PASS/FAIL table before anything else
runs: (1) `features.npy` hash, (2) manifest hash, (3) frozen fields, (4) shape agreement,
(5) `feature_order[target_index] == "btc_logret_1"`, (6) `scaler.json` hash. Rule 5 matters most
— a reordered matrix does not error, it produces plausible-looking garbage.

Two implementation details that are easy to get wrong:

- **`02` loads the matrix fully, not with `mmap_mode='r'`.** The leakage gate overwrites
  `X[:, TARGET_IDX]` in place and restores it in a `finally`; a read-only mapping raises. The
  full read is ~1 GB at the `full` profile, well inside budget. `mmap=True` is used only for
  `01`'s round-trip verification, where streaming is what is wanted.
- **`train_row` and `n_tr` are recomputed from `CFG` in `02`, never read from the artifact.**
  A scaler gate that compared the artifact's own row count against the artifact's own claim
  would pass unconditionally.

---

## 4. Data Inventory — Verified Facts

Everything below was **measured from the actual files**, not assumed. Re-verify after any data
refresh.

### 4.1 Critical global caveat

`data/csv_to_parquet.py` wrote every CSV with `dtype=str`. **All columns in every parquet file are
`String`.** Missing values are stored as **empty strings `""`**, not nulls. Any loader must:

```python
pl.col(c).replace("", None).cast(pl.Float64, strict=True)
```

Use `strict=True` so a malformed cell raises instead of silently becoming null.

### 4.2 BTC/USDT — Binance 1-minute

- Files: `btc_usdt_binance_2018.parquet` … `btc_usdt_binance_2026.parquet` (9 files).
- Columns: `timestamp` (epoch **milliseconds**), `open`, `high`, `low`, `close`, `volume`,
  `datetime` (ISO-8601 with `+00:00`, i.e. **UTC**).
- Rows: **4,460,232** total.
- Range: **2018-01-01 00:00:00 UTC → 2026-06-30 23:59:00 UTC**.
- Duplicates: **0**.
- Continuity: 4,460,200 consecutive 1-minute steps; **31 discontinuities** (Binance maintenance
  windows). Largest observed gap ≈ **10h01m**; others 1h–3h30m.
- `volume` is base-asset (BTC) volume. Quote volume, trade count, and taker-buy volume are **not
  present** — do not invent order-flow features that require them.

**Handling:** the BTC minute index defines the **master grid**. Gaps must be explicitly represented
(`btc_is_synthetic` flag), never silently interpolated. Windows spanning a gap larger than a
configured tolerance (default: 15 minutes) should be **dropped from training**, not filled.

### 4.3 XAU/USD Gold — 1-minute

- File: `xauusd_2018_2026.parquet`.
- Columns: `Date` (`YYYYMMDD`), `Timestamp` (`HH:MM:SS`), `Open`, `High`, `Low`, `Close`, `Volume`.
- Rows: **2,979,344**. Range: **2018-01-01 23:00 → 2026-05-31 23:59**. Duplicates: **0**.
- `Volume` is **tick volume**, not notional; it is never exactly zero.
- Continuity: 2,975,809 one-minute steps, plus
  - **~1,664 breaks of ≈1h01m** — daily maintenance/rollover break,
  - **~391 breaks of ≈2d01h** — **weekend market closure**,
  - ~140 small 2–4 minute gaps — thin liquidity.

> **✅ RESOLVED — timezone: the file is already UTC, offset = 0.** The timestamps carry no
> timezone label, so this was measured rather than assumed, two ways, and both agree.
> *Structural*: the last bar before the weekly break is Fri 21:59 in winter and Fri 20:59 in
> summer, reopening Sun 23:00 / Sun 22:00 — exactly a New York 17:00 close and 18:00 reopen in
> both DST regimes. *Statistical*: scanning candidate offsets −12h…+12h, the contemporaneous
> BTC↔gold 1-minute return correlation peaks at **+0h with ρ = +0.0626** while every other
> offset sits at ±0.002 — a ~30× separation.
> Pinned as `CFG.gold_utc_offset_h = 0` in the notebook and recorded in
> `feature_manifest.json` as `gold_utc_offset_hours`. Set it to `"auto"` to re-derive the scan
> from the data. **Re-run the detection after any data refresh** — an unresolved or changed
> offset silently injects look-ahead or look-behind bias into every cross-asset feature.

### 4.4 FED Broad Trade-Weighted US Dollar Index — daily

- File: `US_Dollar_Index_2018_2026.parquet`.
- Columns: `Date` (`YYYY-MM-DD`), `US_Dollar_Index`.
- Rows: **2,195**. Range: **2018-01-01 → 2026-05-29**.
- **97 empty values** (`""`) — US market holidays and the 2018-01-01 opening row.
- Business-day frequency: weekends are **absent rows**, holidays are **present rows with empty
  values**. Both must be handled.

> **⚠️ VERIFY — series identity and base.** Values start near `109.64` in Jan-2018. The FRED
> *nominal* broad index (`DTWEXBGS`, base Jan-2006 = 100) was ≈114–115 then; the *real* broad index
> (`RTWEXBGS`, monthly) was ≈100.9. So this file is **either rebased or a different vintage**.
> Measured range: 109.64 on 2018-01-02 → 118.88 on 2026-05-29, min/max 106.5–130.0.
> Confirm the exact FRED series ID and base period, and document it in `configs/data.yaml`.
>
> **✅ RESOLVED — redundancy with the monthly `Real_Broad_Dollar_Index`.** Measured against
> this daily file: level correlation **0.973**, month-over-month log-change correlation
> **0.989**, ratio drifting **1.062 ± 0.013** — a nominal/real pair rather than a constant
> rebasing. The monthly column is therefore **dropped** (`MACRO_DROP` in the notebook). Keeping
> both would split cross-variate attention across two copies of one signal, and the daily file
> is strictly more informative.

### 4.5 US Macroeconomics — monthly

- File: `fed_economic_data_2018_2026.parquet`.
- Rows: **101**. `Date` is **month-end**: `2018-01-31 → 2026-05-31`.
- 32 columns: `Date`, `CPI`, `CPI_Core`, `PPI`, `PCE`, `PCE_Core`, `Fed_Funds_Rate`,
  `Fed_Funds_Target`, `Fed_Funds_Target_Lower`, `Treasury_10Y`, `Treasury_2Y`, `Treasury_5Y`,
  `Real_Interest_Rate`, `M1`, `M2`, `Fed_Balance_Sheet`, `Reserves`, `GDP`, `GDP_Real`,
  `Unemployment_Rate`, `Labor_Force_Participation`, `Non_Farm_Payroll`, `Industrial_Production`,
  `Capacity_Utilization`, `Retail_Sales`, `Real_Broad_Dollar_Index`, `Consumer_Sentiment`,
  `Inflation_Expectations_5Y`, `Inflation_Expectations_10Y`, `Bank_Credit`, `Consumer_Credit`,
  `Mortgage_Rate_30Y`.
- Empty values: `GDP` (4), `GDP_Real` (4), `Consumer_Credit` (1).
- **`GDP` / `GDP_Real` are quarterly series already forward-filled to monthly** (2018-01 and 2018-02
  carry the identical value `20328.553`). Treat them as quarterly with a step function; do not
  compute month-over-month deltas naively.

### 4.6 Usable common range

| Series | Ends |
| --- | --- |
| BTC | 2026-06-30 |
| Gold | 2026-05-31 |
| Macro | 2026-05-31 (month-end row) |
| USD Index | 2026-05-29 |

**Master grid: `2018-01-02 00:00 UTC` → `2026-05-31 23:59 UTC`** (start chosen after gold's first
bar and the USD index's first non-empty value). The extra BTC month (June 2026) is reserved as a
**strict hold-out** only if exogenous inputs can be legitimately carried forward with a documented
staleness flag; otherwise discard it.

---

## 5. Stage 1 — Loading & Validation

Implement in `src/data/loaders.py`. One function per source, each returning a `polars.LazyFrame`
with a canonical schema and a UTC `datetime` column.

**Requirements:**

1. **Explicit schema contract.** Declare expected column names and dtypes; assert on load. Fail
   loudly on drift.
2. **Cast from string** with `strict=True` after mapping `""` → `null`.
3. **Timestamp normalisation** to `Datetime(time_unit="us", time_zone="UTC")`:
   - BTC: parse `datetime`, cross-check against `timestamp` (ms) — they must agree exactly.
   - Gold: `Date + " " + Timestamp` → naive datetime → **apply the resolved offset** → UTC.
   - USD Index / Macro: date-only → set to `00:00:00 UTC` (a daily/monthly observation is
     *knowable* only after its release; see §6.3).
4. **Validation battery** (`scripts/01_build_master.py` prints a report):
   - monotonic non-decreasing timestamps, zero duplicates;
   - OHLC sanity: `low ≤ min(open, close) ≤ max(open, close) ≤ high`, all `> 0`;
   - `volume ≥ 0`;
   - gap census: count and duration histogram, plus a table of every gap `> 5 min`;
   - null/empty census per column;
   - extreme-return census: `|log return| > 10%` in one minute → list them, verify against known
     events (e.g. 2020-03-12, 2021-05-19, FTX 2022-11) rather than deleting.
5. **Never mutate `data/raw/`.** It is the immutable source of truth.

**Performance:** use `pl.scan_parquet("data/raw/btc_usdt_binance_*.parquet")` and push filters into
the scan. The full BTC set is ~4.5M rows and fits comfortably in memory as float32, but lazy scans
keep the pipeline composable.

---

## 6. Stage 2 — Multi-Granularity Temporal Alignment (the core problem)

Four series sampled at 1 min / 1 min (with closures) / 1 day / 1 month must become **one aligned
matrix on the BTC minute grid** without leaking future information. This section is the part that
most determines whether the project succeeds.

### 6.1 Governing principle

> **At master timestamp `t`, a feature may only use information that a real observer would have
> possessed at or before `t`.**

Every join is therefore a **backward as-of join**, never a nearest or forward join.

### 6.2 Gold → minute grid

Gold is *asynchronous*: closed on weekends and during a daily break, while BTC trades 24/7.

- Use `polars.join_asof(strategy="backward")` on UTC timestamps.
- **Emit a staleness feature**: `gold_staleness_min = (t - last_gold_obs_time).minutes`, then
  transform as `log1p(staleness)` and additionally a binary `gold_market_open`.
- **Do not forward-fill gold *returns*.** Forward-filling the *price* is correct; deriving returns
  from a forward-filled price manufactures long runs of exactly-zero returns that corrupt
  volatility and correlation estimates. Instead:
  - compute gold returns **on gold's own clock**, then as-of join the return series;
  - mask cross-asset statistics to the gold-open subset, and expose the mask to the model.
- **Weekend gap handling:** the Friday-close → Sunday-open gold return is a genuine 2-day return.
  Do not distribute it across the weekend minutes. Represent it as a separate
  `gold_weekend_gap_return` feature that becomes non-zero only at the reopening bar and decays.
- Cap staleness: if `gold_staleness_min > 3 days` (data outage, not a normal weekend), mark the
  window invalid.

### 6.3 USD Index (daily) and Macro (monthly) → minute grid — publication lag

**This is where leakage usually enters.** A month-end-dated macro observation was **not knowable at
month end**. Applying it from its nominal date is look-ahead bias.

Implement a **release-lag table** in `configs/data.yaml`, applied *before* any join:

| Indicator group | Nominal date | Applied-from rule (conservative) |
| --- | --- | --- |
| CPI, CPI_Core | month `M` | ~13th of `M+1`, 13:30 UTC → lag **+1 month, +13 days** |
| PPI | month `M` | ~14th of `M+1` |
| PCE, PCE_Core | month `M` | ~last business day of `M+1` → lag **+2 months** (safe) |
| Non_Farm_Payroll, Unemployment_Rate, Labor_Force_Participation | month `M` | 1st Friday of `M+1`, 13:30 UTC |
| Retail_Sales | month `M` | ~mid `M+1` |
| Industrial_Production, Capacity_Utilization | month `M` | ~15th of `M+1` |
| GDP, GDP_Real | quarter `Q` | advance ≈ `Q_end + 30d`; use **+1 quarter** (safe) |
| M1, M2, Bank_Credit, Consumer_Credit, Reserves, Fed_Balance_Sheet | month `M` | +1 month |
| Fed_Funds_Rate/Target, Treasury_*, Mortgage_Rate_30Y | market-observed | monthly aggregate → +1 month |
| Consumer_Sentiment, Inflation_Expectations_* | month `M` | +1 month |
| Real_Broad_Dollar_Index (monthly) | month `M` | +1 month |
| US_Dollar_Index (daily file) | date `d` | available **after** `d`'s close → apply from `d+1 00:00 UTC` |

Rules:

- **Default to over-lagging.** If a release date is uncertain, add a full extra period. Losing a
  little signal is vastly cheaper than publishing a leaked result.
- Market-observed rates (Treasury yields, effective fed funds) inside a *monthly* file are still
  monthly *aggregates* — lag them by one month like everything else. Do not treat them as daily.
- After lagging, join with `strategy="backward"` and emit **`macro_age_days`** and
  **`dxy_age_days`** features so the model knows how stale each block is.
- **Vintage caveat:** this dataset contains *revised* values, not real-time vintages (ALFRED). Even
  with correct release lags, revisions leak a small amount of future information. State this
  limitation explicitly in any report. If precision matters, obtain ALFRED vintage data.

### 6.4 Master builder output

`scripts/01_build_master.py` writes `data/interim/master_1min.parquet`:

- index: BTC minute grid, UTC, monotonic;
- BTC OHLCV (raw, unscaled) + `btc_is_synthetic`;
- gold OHLCV as-of joined + `gold_staleness_min`, `gold_market_open`;
- USD index as-of joined (lagged) + `dxy_age_days`;
- macro block as-of joined (lagged) + `macro_age_days`;
- **no derived features yet** — this file is the alignment contract, and it should be regenerable
  and diffable.

Assert on write: row count equals master grid length; no unexpected nulls after the warm-up prefix;
the first valid row is at or after the max of all sources' first available (lagged) observations.

---

## 7. Stage 3 — Feature Engineering

Implement in `src/features/`, toggled by `configs/features.yaml`. Every feature must be **causal**
(computed from a trailing window ending at `t`, inclusive) and **documented with its lookback**.

### 7.1 Target definition

Forecast **log returns, not price levels.**

```
y_t^{(h)} = log(close_{t+h}) - log(close_t),   h ∈ {1, 5, 15, 30, 60} minutes
```

Rationale: 1-minute BTC price is a near-unit-root process. A model predicting the *level* achieves
a spectacular-looking R² by echoing `close_t`, while carrying zero information. **Any report that
shows MSE on price levels is meaningless — always report on returns.**

Secondary target heads (optional, multi-task): realised volatility over `[t, t+h]`, and the sign of
`y_t^{(h)}` for directional accuracy.

### 7.2 BTC feature block (`src/features/price.py`, `volume.py`)

| Group | Features |
| --- | --- |
| Returns | `log_return` at 1, 5, 15, 30, 60, 240, 1440 min |
| Fractional differentiation | fixed-width frac-diff of `log(close)` with `d ∈ [0.2, 0.6]`; choose the smallest `d` passing ADF at 95% — preserves memory that integer differencing destroys (López de Prado, ch. 5) |
| Range volatility (from OHLC) | Parkinson, Garman–Klass, Rogers–Satchell, Yang–Zhang over 15/60/1440 min |
| Realised volatility | sum of squared 1-min returns over 15/60/1440 min; **bipower variation** and the `RV − BV` jump component |
| Distributional | rolling skew, kurtosis, `max(|r|)` over 60 min |
| Volume / liquidity | log-volume, volume z-score vs. same-minute-of-week baseline, Amihud illiquidity `|r| / volume`, rolling volume share |
| Microstructure proxies | VWAP deviation `(close − VWAP_w) / VWAP_w`, Corwin–Schultz high-low spread estimator, close-position-in-range `(C−L)/(H−L)` |
| Momentum (limited set) | RSI(14), MACD histogram, ATR(14) normalised by price, Bollinger %B — **cap the count**; hundreds of collinear indicators degrade cross-variate attention |

### 7.3 Gold feature block (`src/features/cross_asset.py`)

- Gold log returns on gold's own clock (1, 5, 60, 1440 min), as-of joined.
- Gold realised volatility (same estimators, gold clock).
- `gold_staleness_min`, `gold_market_open`, `gold_weekend_gap_return`.
- **Cross-asset**, computed only over gold-open minutes and then as-of joined:
  - rolling Pearson/Spearman correlation of BTC↔gold returns over 60/240/1440 min;
  - rolling OLS beta of BTC on gold + residual;
  - **lead–lag**: cross-correlation at lags 1–15 min in both directions (BTC leads gold / gold
    leads BTC), summarised as argmax-lag and peak value. Use only past data on both sides.
  - normalised spread z-score of `log(BTC) − β·log(Gold)`.

### 7.4 USD Index & macro blocks (`src/features/macro.py`)

- Level, log-level, and Δ over 1/5/21 business days for the daily USD index; percentile rank over a
  trailing 252-day window.
- Macro: for each indicator, **year-over-year and month-over-month change**, plus a
  **trailing z-score over 36 months** — raw levels of `CPI`, `M2`, `Fed_Balance_Sheet` etc. are
  strongly trending and will dominate the scaler.
- Derived regime features:
  - yield-curve slope `Treasury_10Y − Treasury_2Y` (recession proxy);
  - real rate `Fed_Funds_Rate − CPI_YoY`;
  - liquidity impulse `Δ log(M2)` and `Δ log(Fed_Balance_Sheet)`;
  - "surprise" proxy: `actual − 12-month-ahead random-walk forecast` (a cheap stand-in for
    consensus-vs-actual when survey data is unavailable — label it clearly as a proxy).
- `macro_age_days`, `dxy_age_days` staleness features.
- **Dimensionality control:** 31 macro indicators × several transforms explodes the variate count
  and dilutes attention. Either (a) select ≤10 indicators by economic reasoning + mutual information
  on the *training split only*, or (b) compress the macro block with PCA fitted on the training
  split, keeping 3–5 components, and treat each component as a variate.

### 7.5 Temporal block (`src/features/temporal.py`)

1-minute crypto has strong, exploitable intraday seasonality.

- Cyclical encodings (`sin`/`cos` pairs) for minute-of-hour, hour-of-day, day-of-week,
  day-of-month, month-of-year.
- Session dummies in UTC: Asia (00:00–08:00), Europe (07:00–16:00), US (13:00–21:00), overlap flags.
- Flags for the **CME futures gap window**, weekend, month-end/quarter-end, and scheduled release
  minutes (13:30 UTC CPI/NFP, 19:00 UTC FOMC).
- Minutes-to-next / minutes-since-last scheduled macro release.

### 7.6 Feature hygiene rules

1. **All rolling windows are trailing and closed on the right.** Verify with a unit test that
   shifting the input forward by `k` shifts every feature forward by exactly `k`.
2. **Warm-up truncation:** drop the first `max_lookback` rows (the longest window, e.g. 1440 min or
   36 months of macro) after feature construction.
3. **No global statistics.** Means, stds, quantiles, PCA loadings, frac-diff `d`, and feature
   selections are fitted **on the training split only** and applied unchanged to val/test.
4. **Outliers:** winsorise features at training-split 0.1%/99.9% quantiles. **Do not winsorise the
   target** — tail events are the phenomenon of interest. Use a robust loss instead (§12.3).
5. **Collinearity:** drop features with |ρ| > 0.98 against an already-selected feature (keep the
   simpler one). Report a VIF table.
6. **Persist a feature manifest** (`artifacts/feature_manifest.json`): name, module, formula,
   lookback, dtype, train-split scaler parameters. Inference must reconstruct features from this
   manifest, not from ad-hoc code.
7. **Storage:** write `data/processed/features.npy` as `float32` memmap plus a
   `feature_names.json`. ~4.4M rows × ~60 features × 4 B ≈ 1.1 GB — memmap it, don't hold copies.

---

## 8. Stage 4 — Windowing & Dataset

Implement `src/data/dataset.py`.

- A sample is `(X, y)` with `X ∈ R^{L × N}` (`L` = lookback minutes, `N` = number of variates) and
  `y ∈ R^{H}` (or `R^{H × 1}` for the BTC return head).
- **Sliding windows with stride.** At 1-minute granularity, stride 1 yields ~4.4M near-duplicate
  windows; consecutive windows share `L−1` of `L` rows. Use `stride ∈ {1, 5, 15}` for training
  (5 is a good default: keeps ~880k windows, cuts redundancy and epoch time) and **stride 1 for
  validation/test** so evaluation covers every timestamp.
- **Reject windows** that (a) span a BTC data gap > tolerance, (b) contain `gold_staleness_min`
  above the cap, or (c) fall inside a purge/embargo zone (§9.1). Precompute a boolean
  `valid_window_start` mask once and index into it — do not filter inside `__getitem__`.
- Return `float32` tensors. Do the `float32` cast once at build time.
- `DataLoader`: `num_workers=4–8`, `persistent_workers=True`, `pin_memory=True`,
  `prefetch_factor=4`, `drop_last=True` for training. On Windows, guard the entry point with
  `if __name__ == "__main__":` — worker processes re-import the module.
- **Never shuffle across the split boundary.** Shuffling *within* the training split is correct and
  necessary; shuffling across train/val/test is leakage.

---

## 9. Stage 5 — Splitting Strategy

### 9.1 Purged & embargoed chronological splits

Implement `src/training/splits.py`. Standard k-fold and random splits are **invalid** here — they
train on the future. Overlapping windows also leak across an ordinary chronological boundary:
a training window ending at `t_train_end` has a label reaching to `t_train_end + H`, which overlaps
the first validation window.

Apply **purging + embargo** (López de Prado, *Advances in Financial Machine Learning*, ch. 7):

```
purge   = H_max                       # forecast horizon of the label
embargo = L_max + H_max + safety      # longest feature lookback + horizon + margin
gap_between_splits = purge + embargo  # in minutes
```

Drop every window whose `[start, end + H]` interval intersects the gap.

### 9.2 Default split

| Split | Range | Approx. share |
| --- | --- | --- |
| Train | 2018-01-02 → 2023-12-31 | ~70% |
| Validation | 2024-01-01 → 2024-12-31 | ~12% |
| Test (hold-out) | 2025-01-01 → 2026-05-31 | ~18% |

With a gap of `embargo` minutes inserted at each boundary.

### 9.3 Walk-forward evaluation (required for the final report)

A single split over a regime-shifting asset is not enough evidence. Run an **expanding-window
walk-forward**: train on `[start, T_k]`, validate on the next 3 months, test on the following
3 months, roll forward by 3 months. Report mean ± std of every metric across folds. This is the
headline result; the single split is for development speed only.

### 9.4 Regime stratification in reporting

Bucket test results by volatility tercile, by bull/bear/sideways regime, and by year. A model that
is excellent in low-volatility 2019 and catastrophic in March 2020 is not a good model, and an
aggregate metric hides that.

---

## 10. Stage 6 — Model Architecture: the Inverted Transformer

### 10.1 Why inverted, specifically for this project

A vanilla Transformer for time series embeds **each timestamp** as a token: `X ∈ R^{L×N} → L`
tokens of width `N`. iTransformer **inverts the axes**: it embeds **each variate's entire lookback
series** as one token, giving `N` tokens of width `d_model`.

Consequences that matter here:

1. **Attention cost is `O(N²·d)`, independent of `L`.** With `N ≈ 40` variates and `L = 1440`
   minutes, attention is trivially cheap while the model still sees a full day of 1-minute history.
   A time-token Transformer at `L = 1440` would need a 1440×1440 attention matrix per head.
   **This is the decisive architectural argument for this project.**
2. **Attention now models cross-variate correlation** — exactly the BTC ↔ gold ↔ USD ↔ macro
   structure we want to learn — while the feed-forward network models each variate's temporal
   dynamics.
3. **Heterogeneous variates are no longer forced into a shared timestamp token.** Multivariate
   time-token embedding jams a 1-minute BTC return and a monthly CPI z-score into the same token,
   which is physically meaningless. Inversion keeps them in separate tokens.
4. **Attention weights are directly interpretable** as a learned variate-correlation map — useful
   for showing *which* exogenous inputs actually contribute.

### 10.2 Reference implementation skeleton (PyTorch)

`src/models/layers.py`:

```python
import torch
import torch.nn as nn


class DataEmbedding_inverted(nn.Module):
    """Embed each variate's whole lookback series into one token.

    x: (B, L, N) -> (B, N, d_model)
    No positional encoding: variate order is arbitrary, and temporal order is
    already carried inside each token by the linear projection over L.
    """

    def __init__(self, seq_len: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.value_embedding = nn.Linear(seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)              # (B, N, L)
        return self.dropout(self.value_embedding(x))


class FullAttention(nn.Module):
    """Scaled dot-product attention over the VARIATE axis."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1,
                 output_attention: bool = False):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout_p = dropout
        self.output_attention = output_attention

    def forward(self, x: torch.Tensor):
        B, N, _ = x.shape
        q = self.q_proj(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)

        attn = None
        if self.output_attention:
            scale = self.d_head ** -0.5
            scores = (q @ k.transpose(-2, -1)) * scale
            attn = scores.softmax(dim=-1)
            out = attn @ v
        else:
            # NOTE: no causal mask here. Masking applies to the TIME axis, and
            # this attention runs over the VARIATE axis, where all tokens are
            # contemporaneous by construction. Causality is enforced upstream,
            # in feature construction and windowing.
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout_p if self.training else 0.0
            )

        out = out.transpose(1, 2).reshape(B, N, -1)
        return self.out_proj(out), attn


class EncoderLayer(nn.Module):
    """Pre-norm block: attention across variates, FFN per variate."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 dropout: float = 0.1, activation: str = "gelu",
                 output_attention: bool = False):
        super().__init__()
        self.attn = FullAttention(d_model, n_heads, dropout, output_attention)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        h, attn = self.attn(self.norm1(x))
        x = x + self.dropout(h)
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x, attn
```

`src/models/itransformer.py`:

```python
class iTransformer(nn.Module):
    """Inverted Transformer for BTC 1-minute forecasting.

    Input  : (B, L, N) — L lookback minutes, N variates (variate 0 = BTC target)
    Output : (B, H, N), or (B, H, 1) when only the BTC head is projected.
    """

    def __init__(self, seq_len, pred_len, n_variates, d_model=512, n_heads=8,
                 e_layers=3, d_ff=2048, dropout=0.1, activation="gelu",
                 use_norm=True, target_index=0, project_target_only=True,
                 output_attention=False):
        super().__init__()
        self.pred_len = pred_len
        self.use_norm = use_norm
        self.target_index = target_index
        self.project_target_only = project_target_only

        self.enc_embedding = DataEmbedding_inverted(seq_len, d_model, dropout)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout, activation, output_attention)
            for _ in range(e_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.projector = nn.Linear(d_model, pred_len)

    def forward(self, x, return_attention: bool = False):
        # ---- RevIN-style per-instance normalisation (non-stationarity) ----
        if self.use_norm:
            means = x.mean(dim=1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
            x = x / stdev

        h = self.enc_embedding(x)                      # (B, N, d_model)
        attns = []
        for layer in self.layers:
            h, a = layer(h)
            attns.append(a)
        h = self.norm(h)

        out = self.projector(h).permute(0, 2, 1)       # (B, pred_len, N)

        if self.use_norm:
            out = out * stdev[:, 0, :].unsqueeze(1) + means[:, 0, :].unsqueeze(1)

        if self.project_target_only:
            out = out[:, :, self.target_index : self.target_index + 1]

        return (out, attns) if return_attention else out
```

**Notes on the skeleton:**

- `use_norm` implements the RevIN / series-stationarisation trick (denormalise with the *same*
  statistics after prediction). **Keep it on.** Crypto is aggressively non-stationary and this is
  worth more than most architectural tuning.
- The `.detach()` on `means`/`stdev` is deliberate: normalisation statistics must not receive
  gradients.
- `project_target_only=True` avoids wasting capacity predicting exogenous variates. Setting it to
  `False` turns exogenous prediction into an **auxiliary multi-task objective**, which sometimes
  regularises usefully — make it an ablation, not an assumption.
- Prefer `scaled_dot_product_attention` (fused/Flash kernels) and only fall back to the explicit
  path when attention maps are needed for interpretation.
- Consider a **linear residual skip** (DLinear-style): `y = iTransformer(x) + Linear(x[:, :, target])`.
  Linear models are startlingly strong baselines on long-horizon forecasting, and the skip gives the
  Transformer a head start it only has to improve on.

### 10.3 Exogenous-aware variant (recommended experiment)

Our setting is precisely *endogenous target + exogenous covariates*, which is what **TimeXer**
(NeurIPS 2024) targets: patch-level tokens for the endogenous series, variate-level tokens for
exogenous series, and a global endogenous token bridging them via cross-attention. Implement it in
`src/models/` as an alternative head and compare head-to-head with plain iTransformer under
identical splits. Report both.

---

## 11. Stage 7 — Hyperparameters

### 11.1 Defaults (start here)

| Parameter | Default | Notes |
| --- | --- | --- |
| `seq_len` (L) | **1440** | one full day of minutes; cheap because attention is over variates |
| `pred_len` (H) | **60** | also train separate models/heads for 1, 5, 15, 30 |
| `n_variates` (N) | ~40 | after selection/compression (§7.4) |
| `d_model` | **512** | 256 if N < 20 or overfitting |
| `n_heads` | **8** | `d_model % n_heads == 0` |
| `e_layers` | **3** | paper range 2–4; more rarely helps at this N |
| `d_ff` | **2048** | official iTransformer often uses 512; try both |
| `dropout` | **0.1** | raise to 0.2–0.3 if val loss diverges early |
| `activation` | `gelu` | |
| `use_norm` (RevIN) | **True** | effectively mandatory here |
| `output_attention` | `False` | `True` only for interpretation runs |
| Optimiser | **AdamW** | `betas=(0.9, 0.98)`, `eps=1e-8` |
| `lr` | **3e-4** | paper uses 1e-4 … 5e-4 |
| `weight_decay` | **1e-4** | exclude bias and LayerNorm params |
| `batch_size` | **64** | 256 if VRAM allows; scale `lr` ~√batch |
| Scheduler | **cosine + 5% linear warmup** | `OneCycleLR` is a fine alternative |
| `epochs` | **30** | with early stopping |
| Early stopping | **patience 5** on val loss | restore best weights |
| `grad_clip` | **1.0** | global norm |
| Loss | **Huber (`delta=1.0` on standardised returns)** | see §12.3 |
| AMP | **bf16** on CUDA | fall back to fp16+GradScaler, then fp32 |
| Seed | **42** (and 1, 7, 13, 2024 for seed-variance runs) | |

**Parameter budget sanity check:** embedding `L·d_model = 1440·512 ≈ 737k`; each encoder layer
`≈ 4·d_model² + 2·d_model·d_ff ≈ 3.1M`; projector `d_model·H ≈ 31k`. Three layers ≈ **10M
parameters** — comfortably trainable on one consumer GPU.

### 11.2 Optuna search space (`scripts/04_tune_optuna.py`)

`optuna==4.9.0` is already installed. Use `TPESampler(multivariate=True, group=True)` +
`MedianPruner(n_warmup_steps=3)`.

```python
space = {
    "seq_len":      trial.suggest_categorical("seq_len", [480, 960, 1440, 2880]),
    "d_model":      trial.suggest_categorical("d_model", [128, 256, 512]),
    "n_heads":      trial.suggest_categorical("n_heads", [4, 8, 16]),
    "e_layers":     trial.suggest_int("e_layers", 2, 4),
    "d_ff":         trial.suggest_categorical("d_ff", [512, 1024, 2048]),
    "dropout":      trial.suggest_float("dropout", 0.0, 0.3, step=0.05),
    "lr":           trial.suggest_float("lr", 1e-5, 1e-3, log=True),
    "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
    "batch_size":   trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
    "loss":         trial.suggest_categorical("loss", ["mse", "huber", "pinball"]),
}
```

**Rules for tuning:**

- Optimise on the **validation split only**. The test split is opened once, at the end.
- Enforce `d_model % n_heads == 0` — prune invalid trials with `optuna.TrialPruned()`.
- Budget 50–100 trials with pruning; log every trial to `artifacts/runs/optuna.db` (SQLite storage
  so runs are resumable).
- Report the **top-5 configurations**, not just the argmax — a single best trial on a noisy
  financial validation set is largely luck.

---

## 12. Stage 8 — Training

`src/training/trainer.py` + `scripts/03_train.py`.

### 12.1 Loop requirements

1. **Reproducibility:** seed `random`, `numpy`, `torch`, `torch.cuda`; set
   `torch.backends.cudnn.deterministic = True` and `benchmark = False` for final runs (turn
   `benchmark=True` back on for exploratory speed). Set `PYTHONHASHSEED`. Seed DataLoader workers
   with a `worker_init_fn`.
2. **AMP:** `torch.autocast(device_type="cuda", dtype=torch.bfloat16)`. Keep loss computation in
   fp32.
3. **Gradient clipping** before `optimizer.step()`.
4. **Checkpointing:** save `{model_state, optimizer_state, scheduler_state, epoch, best_metric,
   config, feature_manifest_hash, git_commit, torch_version}`. A checkpoint that cannot be tied back
   to its exact feature pipeline is worthless.
5. **Logging:** per-epoch train/val loss, LR, grad-norm, epoch wall-time, peak VRAM. Write JSONL to
   `artifacts/runs/<run_id>/metrics.jsonl` — no external tracking service required.
6. **Early stopping** on validation loss with best-weight restore.
7. **Overfit-a-single-batch smoke test before every long run.** If the model cannot drive loss to
   ~0 on 8 samples in 200 steps, the architecture or data plumbing is broken. Do not launch a
   multi-hour run without this passing.

### 12.2 Sanity gates that must pass before training is considered valid

- [ ] Shift test: shifting inputs by `k` shifts every feature by exactly `k`.
- [ ] Leakage test: replace the target with pure noise → validation metrics must collapse to
      baseline. If the model still "predicts" well, features contain the target.
- [ ] Split test: assert `max(train_time) + embargo ≤ min(val_time)`, likewise val→test.
- [ ] Scaler test: assert scaler parameters were fitted on the training split only (hash-check).
- [ ] Baseline test: the random-walk baseline is computed and logged **first**, before any model
      training.

### 12.3 Loss functions (`src/training/losses.py`)

- **MSE** — comparability with the literature; report it always.
- **Huber / Smooth L1** — recommended default. 1-minute crypto returns are heavy-tailed; MSE lets a
  handful of flash-crash minutes dominate the gradient.
- **Pinball (quantile) loss** at `q ∈ {0.1, 0.5, 0.9}` — gives prediction intervals, which are far
  more actionable than a point forecast in this domain.
- **Directional-aware** — e.g. `L = Huber(y, ŷ) + λ · softplus(−y · ŷ)`. Useful because at a
  1-minute horizon, *sign* accuracy is what a strategy monetises. Keep `λ` small (0.05–0.2) and
  treat it as an ablation, not a default.

---

## 13. Stage 9 — Validation, Testing, Evaluation

### 13.1 Baselines that must be beaten (non-optional)

At a 1-minute horizon the naive baseline is **brutally strong**. Reporting a model result without
these numbers side by side is not an acceptable output.

1. **Random walk / naive persistence** — `ŷ = 0` for returns (i.e. `price_{t+h} = price_t`).
2. **Historical mean return** over a trailing window.
3. **ARIMA / AR(p)** on returns.
4. **DLinear** — one linear layer over the lookback; frequently competitive with Transformers.
5. **Vanilla (time-token) Transformer** — isolates the contribution of *inversion*.
6. **iTransformer, BTC-only** — isolates the contribution of the *exogenous variables*.

### 13.2 Metrics (`src/evaluation/metrics.py`)

**Statistical**

- MSE, RMSE, MAE **on log returns** (never on price levels).
- **MASE** relative to the naive baseline. `MASE ≥ 1` means the model is useless.
- R² / explained variance — expect small positive values; at 1-minute horizons anything above ~0.05
  is genuinely notable and should raise a leakage suspicion until it is re-verified.
- **Directional accuracy** and Matthews correlation coefficient on `sign(y)`, excluding near-zero
  returns (`|y| < ε`) which are noise.
- **Diebold–Mariano test** vs. each baseline. A metric difference without a significance test is not
  evidence.
- Pinball loss + empirical coverage of prediction intervals if quantile heads are trained.

**Economic (`src/evaluation/backtest.py`)** — statistical improvement that does not survive costs is
not a result.

- Long/flat/short strategy driven by the forecast sign and a confidence threshold.
- **Realistic costs:** Binance spot taker fee ~0.04–0.10% per side, plus slippage (at least
  0.5× the half-spread, worse in high volatility). At 1-minute frequency, costs dominate — a
  strategy trading every minute pays ~100%+ annualised in fees.
- Report: cumulative return, annualised Sharpe, Sortino, max drawdown, Calmar, hit rate, average
  win/loss, **turnover**, and net-of-cost P&L.
- **Deflated Sharpe ratio** (López de Prado) to correct for the multiple-testing inflation caused by
  the hyperparameter search.

### 13.3 Diagnostics

- Residual autocorrelation (Ljung–Box) — remaining structure means the model missed signal.
- Error distribution by regime, hour-of-day, and day-of-week.
- **Attention-map analysis**: average the variate-attention matrix across the test set to show which
  exogenous variates the model actually uses. This is the direct scientific payoff of choosing an
  inverted architecture, and it belongs in the report.
- **Ablation table** (mandatory in the final report):

  | Configuration | MSE | MASE | Dir. Acc. | Net Sharpe |
  | --- | --- | --- | --- | --- |
  | BTC only | | | | |
  | BTC + Gold | | | | |
  | BTC + USD Index | | | | |
  | BTC + Macro | | | | |
  | BTC + all exogenous | | | | |

- **Seed variance:** repeat the final configuration over ≥5 seeds and report mean ± std. A single
  seed is an anecdote.

### 13.4 Test-set discipline

The test split is touched **once**, after the model, features, and hyperparameters are frozen. If
the test result prompts a change, everything after that change must be re-validated and the test
re-run counts as a new experiment — **say so explicitly in the report**. Silent test-set iteration
is the fastest route to a result that evaporates in production.

---

## 14. Stage 10 — Export & Inference Contract

`src/export/` + `scripts/06_export.py`.

**Artifact bundle** (`artifacts/models/<run_id>/`):

```
model.pt                 # torch.save(state_dict) — always
model_scripted.pt        # torch.jit.script or trace
model.onnx               # opset >= 17, dynamic batch axis
scaler.json              # per-feature mean/std from the TRAIN split
feature_manifest.json    # exact feature definitions + order + lookbacks
config.yaml              # full resolved config
metadata.json            # git commit, torch version, data hash, metrics, UTC timestamp
inference_example.py     # end-to-end runnable example
```

**Requirements:**

1. `model.eval()` and `torch.no_grad()` before tracing.
2. **Verify parity:** PyTorch vs. TorchScript vs. ONNX Runtime outputs must agree within `1e-4`
   (fp32) on a fixed random batch. Fail the export if they do not.
3. **Dynamic batch axis**, fixed `seq_len` and `n_variates`.
4. **Feature order is part of the contract.** Serialise it and assert it at inference. A silently
   reordered feature matrix produces plausible-looking garbage.
5. **Document the inference contract explicitly** in `inference_example.py`: input shape
   `(B, L, N)`, dtype `float32`, features already scaled with `scaler.json`, variate order per
   `feature_manifest.json`, output shape `(B, H, 1)` in **standardised log-return space**, plus the
   exact inverse transform back to price.
6. **State the staleness policy at inference time:** which exogenous inputs may be forward-filled
   and for how long before the prediction must be refused.
7. Do **not** export a `pickle` of the full model object — `state_dict` + code is the portable and
   reviewable form.

---

## 15. Improvisations & Advanced Roadmap

Implement after the core pipeline is verified end-to-end. Each item should be a measured
experiment against the frozen baseline, not a speculative addition.

1. **TimeXer-style exogenous handling** (§10.3) — the most directly relevant upgrade for this data.
2. **Hybrid linear + iTransformer** — DLinear residual skip; often a free accuracy gain.
3. **Non-stationary attention** — de-stationary attention factors recover the non-stationarity that
   instance normalisation removes (Non-stationary Transformers, NeurIPS 2022).
4. **Multi-horizon multi-task head** — shared encoder, separate projectors for `H ∈ {1,5,15,30,60}`.
5. **Quantile heads** for calibrated uncertainty; pair with the interval-coverage diagnostic.
6. **Volatility-scaled targets** — divide the target by trailing realised volatility. Homogenises
   the target across regimes and usually improves optimisation stability substantially.
7. **Sample weighting by uniqueness / return attribution** — overlapping labels at 1-minute
   granularity make effective sample size far smaller than row count (López de Prado, ch. 4).
8. **Frequency-domain features** — FFT/wavelet energy in trailing windows as extra variates.
9. **Variate selection via attention scores** — prune low-attention variates, retrain, verify no
   degradation. Yields a smaller, faster, more defensible model.
10. **Knowledge distillation** into a small student for latency-critical inference.
11. **Conformal prediction** for distribution-free intervals with finite-sample coverage guarantees.
12. **Regime-conditional ensembling** — separate experts for high/low-volatility regimes with a
    gating network.
13. **Gradient checkpointing + `torch.compile`** if the model grows; `torch.compile(model,
    mode="max-autotune")` typically gives 1.3–2× on modern GPUs.
14. **DuckDB-backed feature store** — `duckdb` is already a dependency; use it to query the master
    parquet without materialising, which keeps feature iteration fast.

---

## 16. Hard Rules — Anti-Leakage Checklist

Re-read before **every** commit that touches data or features.

| ❌ Never | ✅ Instead |
| --- | --- |
| `fillna(method="bfill")` on anything | forward-fill only, with a staleness feature |
| `df.interpolate()` across a gap | leave the gap; flag and drop the window |
| Fit scaler on the full dataset | fit on training split only, persist parameters |
| `train_test_split(shuffle=True)` | chronological split + purge + embargo |
| Standard k-fold CV | purged, embargoed walk-forward |
| Use month-end macro dates directly | apply the publication-lag table (§6.3) |
| Compute correlation with a centred window | trailing windows, closed on the right |
| Report MSE on price levels | report on log returns, plus MASE vs. naive |
| Tune on the test set | tune on validation; open test once |
| Drop outlier returns | keep them; use a robust loss |
| Delete BTC gap rows silently | flag them and exclude affected windows |
| Backward-fill gold across the weekend | as-of backward join + staleness feature |
| Trust a Sharpe > 3 at 1-minute frequency | assume leakage until proven otherwise |

**If a result looks too good, it is leakage until proven otherwise.** The correct response to an
unexpectedly high validation score is to hunt for the leak, not to celebrate.

---

## 17. Verification Tasks — status

Resolve these before any result is reported as final. Answers live in the notebook's config
cell and in `feature_manifest.json`; move them to `configs/data.yaml` when `src/` is extracted.

1. ✅ **Gold timezone offset** (§4.3) — **RESOLVED: the file is already UTC, offset = 0.** Two
   independent lines of evidence agree. *Structural*: the last bar before the weekly break is
   Fri 21:59 in winter and Fri 20:59 in summer, reopening Sun 23:00 / Sun 22:00 — exactly a
   New York 17:00 close and 18:00 reopen in both DST regimes. *Statistical*: scanning offsets
   −12 h…+12 h, the contemporaneous BTC↔gold 1-minute return correlation peaks at +0 h with
   ρ = +0.0626 while every other offset sits at ±0.002 — a ~30× separation. Pinned as
   `CFG.gold_utc_offset_h = 0`; set it to `"auto"` to re-derive from the data.
2. ⏳ **USD Index series identity and base period** (§4.4) — still open. Measured: starts 109.64
   on 2018-01-02, ends 118.88 on 2026-05-29, range 106.5–130.0. That does not match FRED
   `DTWEXBGS` (≈114–115 in Jan-2018) at its published base, so the file is rebased or a
   different vintage. Confirm the exact series ID against the source.
3. ⏳ **Macro release-lag table** — still approximate. Every lag is rounded *up*, so the error is
   conservative, but it is not scraped from the actual 2018–2026 BLS/BEA/Fed calendars.
4. ⏳ **Revised vs. vintage data** — documented as a known limitation in the notebook's §17. Not
   fixable without ALFRED vintages; quantification still outstanding.
5. ✅ **Redundancy between the daily USD index and monthly `Real_Broad_Dollar_Index`** —
   **RESOLVED: measured level correlation 0.973 and month-over-month log-change correlation
   0.989, with a ratio drifting 1.062 ± 0.013** (consistent with a nominal/real pair rather
   than a constant rebasing). The monthly column is **dropped** (`MACRO_DROP`); the daily file
   is strictly more informative and keeping both would split cross-variate attention across
   two copies of one signal.
6. ⏳ **Binance gap provenance** — 31 discontinuities confirmed present and flagged
   (`btc_is_synthetic`, windows rejected), but not yet cross-checked against Binance's
   published maintenance announcements.

---

## 18. References

### Core architecture (read these first)

1. **Liu, Y., Hu, T., Zhang, H., Wu, H., Wang, S., Ma, L., & Long, M. (2024).**
   *iTransformer: Inverted Transformers Are Effective for Time Series Forecasting.*
   **ICLR 2024 (Spotlight).** arXiv:2310.06625.
   [arXiv](https://arxiv.org/abs/2310.06625) ·
   [OpenReview](https://openreview.net/forum?id=JePfAI8fah) ·
   [Official PyTorch code](https://github.com/thuml/iTransformer)
   *The primary reference. Variate tokens, attention over variates, FFN for temporal
   representation, and the finding that inversion enables effective use of arbitrarily long
   lookbacks.*

2. **Wang, Y., Wu, H., Dong, J., Liu, Y., Qiu, Y., Zhang, H., Wang, J., & Long, M. (2024).**
   *TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables.*
   **NeurIPS 2024.** arXiv:2402.19072.
   [arXiv](https://arxiv.org/abs/2402.19072) ·
   [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/0113ef4642264adc2e6924a3cbbdf532-Abstract-Conference.html) ·
   [Official code](https://github.com/thuml/TimeXer)
   *Directly matches this project's endogenous-target-plus-exogenous-covariates setting.*

### Normalisation and non-stationarity

3. **Kim, T., Kim, J., Tae, Y., Park, C., Choi, J.-H., & Choo, J. (2022).**
   *Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution
   Shift.* **ICLR 2022.**
   [OpenReview](https://openreview.net/forum?id=cGDAkQo1C0p)
   *RevIN — the `use_norm` block in §10.2.*

4. **Liu, Y., Wu, H., Wang, J., & Long, M. (2022).**
   *Non-stationary Transformers: Exploring the Stationarity in Time Series Forecasting.*
   **NeurIPS 2022.** arXiv:2205.14415.
   [arXiv](https://arxiv.org/abs/2205.14415) ·
   [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2022/hash/4054556fcaa934b0bf76da52cf4f92cb-Abstract-Conference.html)

### Comparative architectures / required baselines

5. **Nie, Y., Nguyen, N. H., Sinthong, P., & Kalagnanam, J. (2023).**
   *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers (PatchTST).*
   **ICLR 2023.** arXiv:2211.14730.
   [arXiv](https://arxiv.org/abs/2211.14730) · [Code](https://github.com/yuqinie98/PatchTST)

6. **Zeng, A., Chen, M., Zhang, L., & Xu, Q. (2023).**
   *Are Transformers Effective for Time Series Forecasting? (DLinear / LTSF-Linear).*
   **AAAI 2023.** arXiv:2205.13504.
   [arXiv](https://arxiv.org/abs/2205.13504)
   *The linear baseline that must be beaten. Take it seriously.*

7. **Zhou, H., et al. (2021).** *Informer: Beyond Efficient Transformer for Long Sequence
   Time-Series Forecasting.* **AAAI 2021 (Best Paper).** arXiv:2012.07436.

8. **Wu, H., Xu, J., Wang, J., & Long, M. (2021).** *Autoformer: Decomposition Transformers with
   Auto-Correlation for Long-Term Series Forecasting.* **NeurIPS 2021.** arXiv:2106.13008.

9. **Wu, H., Hu, T., Liu, Y., Zhou, H., Wang, J., & Long, M. (2023).**
   *TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis.* **ICLR 2023.**
   arXiv:2210.02186.

### Financial machine learning methodology

10. **López de Prado, M. (2018).** *Advances in Financial Machine Learning.* Wiley.
    ISBN 978-1-119-48208-6.
    *Chapters 4 (sample uniqueness), 5 (fractional differentiation), 7 (purged & embargoed CV),
    and the deflated Sharpe ratio. §7, §9, and §13 of this document follow it directly.*

11. **Bergmeir, C., & Benítez, J. M. (2012).** *On the use of cross-validation for time series
    predictor evaluation.* **Information Sciences, 191, 192–213.**

12. **Diebold, F. X., & Mariano, R. S. (1995).** *Comparing Predictive Accuracy.*
    **Journal of Business & Economic Statistics, 13(3), 253–263.**

### Implementation resources

13. **Time-Series Library (TSLib), THUML, Tsinghua University.**
    [github.com/thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library)
    *Unified PyTorch implementations of iTransformer, TimeXer, PatchTST, TimesNet, and more.
    Useful as a correctness reference for baselines — port what is needed rather than adding it
    as a dependency.*

14. **PyTorch documentation** — `torch.nn.functional.scaled_dot_product_attention`,
    `torch.autocast`, `torch.compile`, `torch.jit`, `torch.onnx`.

---

## 19. Working Conventions for Claude Code

**Style**

- Python ≥ 3.11 syntax. Type hints on every public function. Google-style docstrings on modules and
  public functions.
- Format with `black` (line length 100) and lint with `ruff` if the user adds them; otherwise match
  the surrounding file's existing style exactly.
- Config lives in YAML under `configs/` and is loaded into dataclasses — **no magic numbers buried
  in code**.
- Comments explain *why*, not *what*. Match the comment density of surrounding code.

**Process**

- **Never write to `data/raw/`.**
- Prefer `polars` lazy scans for anything touching the 4.4M-row BTC table.
- Do not add dependencies without asking. **Never add TensorFlow, Keras, or JAX.**
- Run the sanity gates (§12.2) before any training run longer than a few minutes.
- When reporting results, always include: the baseline, the metric, the split, the seed count, and
  the cost assumptions. A bare number is not a result.
- Long-running training belongs in a background process, with checkpoints written every epoch so a
  crash does not lose the run.
- **Edit `notebooks/iTransformer.ipynb`, never the two generated notebooks.** Then run
  `python tools/build_split_notebooks.py` and execute **both** `01_preprocess.ipynb` and
  `02_train.ipynb` at `PROFILE = "tiny"`, confirming each ends with `ALL CELLS OK` and
  `ALL GATES PASS`. An AST parse is not verification: most defects here are shape, unit, or
  ordering errors that only appear at runtime.
- **After any change to `01`, re-check equivalence.** `features.npy` must stay bit-identical to
  the single notebook's `X` at `tiny`. If it changes, either the change was intended (say so
  explicitly, and note that published artifacts are now stale) or it is a bug.
- **A frozen field never changes on the training side.** If §3.2's frozen list needs a new
  value, rebuild the artifact in `01`. Forcing it in `02` defeats the only mechanism proving
  that multi-session results share an input.

**Units and spacing — the two ways evaluation code lies quietly**

Both of these were live bugs found in review; they produce plausible numbers rather than errors,
which is exactly what makes them dangerous.

1. **Basis points are raw-return units.** `X` is standardised, so a threshold of `1 bp = 1e-4`
   compared against standardised values is ~`1/σ` times too small — at σ ≈ 1.3e-3 it filters
   0.1% of samples instead of the intended ~10%. Every bp-denominated quantity must be divided
   by `SIGMA_TARGET = sd[TARGET_IDX]` before touching a standardised array, or the array must
   be multiplied back into raw units first.
2. **Capping evaluation windows changes their spacing.** `eval_max_windows` thins a split by
   keeping every k-th window, so predictions land k minutes apart, not 1. Any annualisation
   (`periods_per_year`), turnover, or CAGR computed as if k = 1 inflates Sharpe by `sqrt(k)`.
   Derive the spacing from the loader (`eval_row_step`) and pass it through.
3. **The deflated Sharpe ratio takes the per-period Sharpe**, not the annualised one. Feeding
   it an annualised value multiplies the statistic by `sqrt(periods_per_year)` (~750× at
   h = 60) and returns ≈ 1.000 for anything at all — silently disabling the multiple-testing
   correction it exists to apply.

**Command cheatsheet**

```powershell
$PY = "D:\pythonProject\invertedTransformer\.venv\Scripts\python.exe"

# Today: two notebooks are the pipeline, both generated from iTransformer.ipynb.
& $PY tools/build_split_notebooks.py        # rebuild after ANY edit to the source notebook

# The real verification: full CPU execution of both, at PROFILE='tiny' (~5 min total).
# 01 must end with ALL GATES PASS; 02 must end with ALL CELLS OK and parity torchscript OK.
#   01_preprocess.ipynb  ->  data/processed/features_tiny/   (6 files, ~32 MB)
#   02_train.ipynb       ->  artifacts/{checkpoints,runs,models}/tiny_L120_H15_d64_s42/

# Equivalence check against the single notebook (must be bit-identical):
#   np.array_equal(X_from_iTransformer_ipynb, np.load('data/processed/features_tiny/features.npy'))

# Once src/ is extracted, these become the entry points:
& $PY scripts/01_build_master.py   --config configs/data.yaml
& $PY scripts/02_build_features.py --config configs/features.yaml
& $PY scripts/03_train.py          --config configs/train.yaml --model configs/model_itransformer.yaml
& $PY scripts/04_tune_optuna.py    --n-trials 50 --storage sqlite:///artifacts/runs/optuna.db
& $PY scripts/05_evaluate.py       --run-id <run_id> --split test
& $PY scripts/06_export.py         --run-id <run_id> --formats torchscript,onnx
```

**Definition of done for the shipped model**

- [ ] All §17 verification tasks resolved and documented.
- [ ] All §12.2 sanity gates passing.
- [ ] **Every reported result traced to one artifact hash.** The `features_full` artifact's
      `features_sha256` is recorded in the report, and every session that contributed a number
      logged the same hash. Numbers from different hashes are not comparable and must not share
      a table.
- [ ] Walk-forward evaluation (§9.3) complete, mean ± std reported per fold.
- [ ] Ablation table (§13.3) complete.
- [ ] Beats naive, DLinear, and BTC-only baselines with a Diebold–Mariano p-value < 0.05.
- [ ] Economic backtest net of realistic costs reported, whatever the sign of the result.
- [ ] ≥5 seeds run; mean ± std reported.
- [ ] Export bundle complete with verified PyTorch/TorchScript/ONNX parity.
- [ ] Every known limitation stated plainly in the report.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
