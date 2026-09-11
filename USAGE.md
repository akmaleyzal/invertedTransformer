# Running the iTransformer notebook

[notebooks/iTransformer.ipynb](notebooks/iTransformer.ipynb) is the primary implementation. Full experiments run on Kaggle. Local Python is used for small checks, source export and inspection of saved evidence.

## First Kaggle session

1. Upload the saved notebook. Attach `BTCUSDT_1h.parquet` as a Dataset; do not attach the repository or a `src/` package.
2. Select **GPU T4 x2**. The training preflight refuses another GPU setup. Enable Internet if a reporting dependency is missing; setup installs only missing packages and keeps Kaggle's supplied PyTorch.
3. Read Kaggle's accelerator quota meter. Replace `WEEKLY_GPU_HOURS_REMAINING = None` in the setup cell with the actual remaining hours. Enter 30 only if 30 hours remain.
4. Leave `ANALYSIS_ONLY = False`. For a fresh session leave `SESSION_ALREADY_USED_H = 0.0`; otherwise enter time used before the first cell. Rerunning setup in the same kernel preserves its initial clock.
5. Save and Run All in order. Prefer **Save Version → Save & Run All** when practical. If running interactively, save the output before stopping the accelerator session.

The notebook cannot read your account quota or account for simultaneous spending by other notebooks. Recheck the meter each session. A finished cell does not release an interactive GPU allocation; stop the session after saving.

## Budget and continuation

The default session ceiling is **11.5 hours**, below the requested 12-hour limit, with **0.75 hours reserved for saving**. Training therefore stops after at most 10.75 hours from setup, sooner if weekly quota is smaller or the session was already used. Pilot, tuning and grid share this deadline.

The grid uses one independent experiment per T4. A model stays on one device, so its batch size and objective do not change with GPU count. Validation selection is sequential and cached. The 2,130-run grid may require several sessions or quota weeks; the earlier grid's timings do not predict the new baseline schedules. See [Kaggle's GPU guidance](https://www.kaggle.com/docs/efficient-gpu-usage) and [T4 x2 announcement](https://www.kaggle.com/discussions/product-feedback/361104).

To resume:

1. Save the complete output containing `artifacts/`; verify the saved version actually contains those files.
2. Stop the old accelerator/session.
3. Attach that output to the next session alongside the same data.
4. Select T4 x2, recheck quota and Run All.

Accepted prior predictions, weights and metadata are consolidated into the new output. Checkpoints and validation caches are carried forward and identity-checked when consumed. Thus the newest saved output is the bundle to attach next time, including sessions that ended during validation before producing predictions.

Training checkpoints contain current/best weights, optimizer, scheduler, epoch, early-stopping state, RNG state and fitting time. Every completed epoch is saved atomically. An interrupted partial epoch is redone from its preceding saved boundary. Changing input, source, configuration, training selection, PyTorch version or device type invalidates an unfinished-fit resume. Completed predictions remain usable for CPU analysis.

`session_status.json` lists pending IDs, partial checkpoints, elapsed time and next action. Partial grids never enter complete-study reporting. A paused validation stage causes later stages to skip cleanly so output can save. Platform shutdowns and unsaved interactive output cannot be recovered by a Python checkpoint alone.

## Analysis after training

If time remains, the notebook renders the report. Otherwise attach the complete output in a **Kaggle CPU** session and set `ANALYSIS_ONLY = True`; quota entry is then unnecessary. The saved tuning selection is loaded, and pilot/training fits are skipped.

The revised report requires all 2,130 declared IDs, one code/input vintage, complete horizons, common forecast calendars and matching raw outcomes. It records achieved training counts, seeds, representation diagnostics and capped fits. Reported p-values and intervals remain exploratory diagnostics because independent origins are not established. The report makes no causal PR-only, prospective-power, optimal-cadence or executable-backtest claim.

## Revised grid

| Arms | Runs |
|---|---:|
| Main iTransformer: 15 origins × 4 K × 5 seeds | 300 |
| Uniform attention / fresh | 75 / 75 |
| Horizon sweep after shared-ID deduplication | 240 |
| Ridge | 60 |
| DLinear / PatchTST / LSTM | 75 each |
| Persist / seasonal | 15 each |
| Orthogonal / redundant feature subsets | 75 each |
| Lookback 48 / 192 | 75 each |
| Tuned / attention capture / capacity | 75 each |
| Long schedule at K1 and K8 | 150 |
| Validation refresh | 75 |
| Identity / whitened / correlated invertible representations | 75 each |
| DLinear / PatchTST all-channel sensitivity | 75 each |
| **Unique total** | **2,130** |

Each runner selects **11,500 training windows**, without replacement, using seed 1729 and training timestamps alone. The scaler is fitted on full purged training rows before sampling. Insufficient surviving windows raise an error instead of silently lowering the count. Sampling never depends on future test availability.

Fresh and validation-refresh arms both use five seeds and original blocks 4–6. Validation refresh holds the old training span fixed; fresh also updates training. The three representation arms preserve the same K8 information and target channel, fit invertible transforms on training only, and all disable instance normalization. They identify representation effects, not an isolated causal effect of participation ratio.

DLinear/PatchTST use a 120-epoch cap, patience 12 and origin-1 validation LR selection over 1e-4, 1e-3 and 1e-2, separately for target/all-channel objectives. All candidates and caps are reported. Target-only channel-independent models have **effective input K=1**, despite the supplied K8 tensor. Larger budgets and early stopping do not prove convergence.

## Artifacts and timing

| Path under `artifacts/` | Meaning |
|---|---|
| `preds/<run_id>.parquet` | Block, horizon step, forecast origin, input start, target timestamp and scaled true/predicted return |
| `meta/<run_id>.json` | Requested/resolved config, schedule, input/code/file hashes, scaler, counts, training selection, objectives and status |
| `weights/<run_id>.pt` | Best selected model state dictionary |
| `checkpoints/<run_id>.pt` | Incomplete epoch-boundary training state |
| `validation/<digest>.json` | Completed validation-only fit keyed by provenance and configuration |
| `meta/tuning_selection.json` | Every candidate and selected configuration |
| `attn/*.parquet` | Attention-capture outputs |
| `session_status.json` | Session progress and continuation instructions |
| `paper_numbers.json`, analysis Parquets | Complete-grid summaries and panels |

`timestamp == forecast_origin` is the opening time of the first target bar. Inputs end one hour earlier. `input_start = forecast_origin − L hours`; `target_timestamp = forecast_origin + (step − 1) hours`. Blocks are selected by issuance time. Historical input-start timestamps can be relabelled/reblocked in memory, but forecasts never saved cannot be recovered without inference or training.

Execution resume requires matching input/code/requested config/schedule, readable schema and prediction/weight byte hashes. Missing provenance is incomplete. Checkpoints load with `weights_only=True`. Hashes detect mismatches, not authorship: attach your own saved outputs.

## Editing and local checks

1. Edit notebook definitions or workflow cells and save.
2. Add external imports in Library, with package projection imports in the relevant module cell's `metadata.itbtc.projection_imports`. Use `projection_remove_imports` for obsolete imports.
3. Follow the notebook's final local sync cell. Keep it commented in uploaded/committed notebooks. It invokes `tools/notebook_to_src.py`, verifies module-body round trips and refreshes digests.
4. Run `tools/build_notebook.py --check`, relevant tests and `tools/notebook_map.py` when presentation changes. Keep notebook and generated projection together. Never author fixes in `src/` or rebuild from the old template.

An ordered-program digest invalidates all outputs after dependency changes. Archived executed evidence is `notebooks/outputs/iTransformer_before_A01_A15.ipynb`.

Use D: for temporary work in this Windows workspace:

```powershell
$env:TEMP='D:\pythonProject\invertedTransformer\.research\temp'
$env:TMP=$env:TEMP
New-Item -ItemType Directory -Force -Path $env:TEMP | Out-Null
.venv/Scripts/python.exe -X utf8 tools/build_notebook.py --check
.venv/Scripts/python.exe -m pytest tests/test_audit_repairs.py -q
```

Run test files in separate fresh processes on a memory-constrained machine. No full model grid is needed locally.

Original predictions and `paper/paper_numbers.json` remain historical. Corrected analysis lives separately in `paper/reanalysis_2026-09-09/`. The optional `tools/build_report.py --out paper/reanalysis_2026-09-09` command reanalyses saved evidence without training; add `--check` to verify that report. Never overwrite historical predictions to make them pass the new resume gate.

For the detailed A01–A15 disposition, evidence and limitations, read [the remediation report](docs/AUDIT_REMEDIATION_2026-09-09.md).
