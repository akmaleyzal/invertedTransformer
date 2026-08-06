# notebooks/ — launcher rules

Root `CLAUDE.md` is the project law. This file adds only what is specific to notebooks.

## A notebook is a launcher, not a program

Its whole job on Kaggle is five steps:

1. install the package from the attached Kaggle Dataset (or the repo), plus only what the Kaggle
   image genuinely lacks;
2. discover inputs by **globbing `/kaggle/input/*/`** — never a hard-coded dataset slug, so the
   dataset can be renamed without editing code;
3. build the run manifest and subtract the `run_id`s already complete;
4. run the queue across both GPUs;
5. save.

**Logic in a notebook is a defect.** If a cell contains a feature definition, a window builder, a
loss, or a metric, it belongs in `src/` where it can be unit-tested on CPU. The rule exists because
notebook-as-source-of-truth is precisely what made the superseded pipeline unverifiable — see root
§17.

## Kaggle session discipline

- **Grid runs use *Save Version → Save & Run All*, never the interactive editor.** The 20-minute
  idle timeout kills long interactive sessions, and hitting the 12 h wall interactively loses
  `/kaggle/working` entirely.
- The budget guard (`SESSION_BUDGET_H = 11.0`, `RESERVE_H = 0.5`) trips at **run boundaries** and
  exits cleanly so the version saves. Root §10.5.
- Session *N*'s output becomes session *N+1*'s input dataset. Resume is automatic and needs no
  manual bookkeeping: completed runs are those with both artifacts and `status == "complete"`.
- Everything is **written** to `/kaggle/working`; `/kaggle/input` is read-only.
- Print the remaining run count and the estimated sessions left on exit. A session that ends without
  saying how much is left forces the next one to re-derive it.

## Do not

- Re-introduce the generate-notebooks-from-a-source-notebook workflow. It is dead.
- Pin torch or polars versions against the local venv. Kaggle ships its own image; run against what
  is there and install only what is missing.
- Leave a notebook whose outputs are stale relative to `src/`. Outputs are evidence; stale evidence
  is worse than none.

## What exists here now

`itransformer_kaggle.ipynb` — the launcher, 21 cells, load through evaluation. Nine stages: setup ·
data and integrity (Stage 2) · the twelve variates · K_eff (Stage 3b) · pre-flight invariants
(Stage 4) · the Stage 5 gate on **validation** · the 534-run grid across both T4s · RQ1/RQ2/RQ3 ·
save. Every cell calls into `src/`; the only things it defines itself are the glob helpers that
locate the package and the parquet, and those exist precisely so no dataset slug is hard-coded.

**Its markdown carries the *why*, not just the what.** Each banner names the divergence-register
entry the cell enforces, because a reader who does not know why the Stage 5 gate runs on validation
will eventually move it to test, and the notebook is where that decision is visible.

Cells 0–4 (setup through pre-flight) run on CPU in about a minute and were executed locally against
the real artifact before commit. The grid and evaluation cells are compile-checked only — they need
a GPU and a completed grid.

`markdown-example.ipynb` is unrelated scratch and is exempt.
