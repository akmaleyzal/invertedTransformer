# Audit remediation checkpoint

Updated 2026-09-12. **Implementation and documentation complete. No GPU run has occurred.**

## Operating constraints — still live

Notebook is primary. `src/` changes ONLY through its final commented sync cell. Never run the stale
template writer. Training only on Kaggle T4 x2, 12h/session and 30h/week user limits. TEMP/TMP point
to `D:/pythonProject/invertedTransformer/.research/temp`. Preserve the original 1620 predictions,
`paper/paper_numbers.json`, the executed notebook archive and the user reference notes. No commits
and no subagents. Do not rerun the applied migration scripts in `.research/audit-repair-work/`; they
assert the old source and will fail.

## What A01-A15 delivered

First-target timing and common evaluation calendars; mean seed loss then block and origin weighting;
exploratory clustered statistics with no confirmatory rejection; raw-return direction and conditional
long/cash economics; block-1 decay guard; strict provenance; epoch resume and one shared Kaggle
deadline; target-only baselines separated from all-channel, on validation-selected longer schedules;
fixed 11500 training windows; five-seed fresh plus a validation-refresh control; three invertible
representations. Current manifest **2130**, historical **1620**. Notebook is 357 cells with outputs
cleared. All changes exported through the final cell; notebook map regenerated.

## Verification — all green, 2026-09-12

| Check | Command | Result |
|---|---|---|
| Full per-file suite | `.research/audit-repair-work/run_tests.py` | **14 files, 249 tests, exit 0 each.** Artifact-map staleness is gone — it was the one failure in the previous record and the rerun it asked for has now happened |
| Upstream forward parity | `check_upstream_parity.py` | 8 pinned checks pass; max abs err 7.16e-7 against 2e-5 tolerance. CPU forward only |
| Training window budget | `window_budget.json` | 72 spans, available 11689-15265, all >= 11500 |
| Report drift | `tools/build_report.py --check --out paper/reanalysis_2026-09-09` | `report is current`, exit 0 |
| LaTeX | tectonic, `.research/temp/latex-manuscript/` | Compiles. **6 pages**, font-shape warnings only |
| Two-GPU path | `tests/test_audit_repairs.py`, 3 new cases | `visible_devices()` returns both devices when CUDA reports 2; one worker thread per device; both devices receive work off the shared cursor; every cell runs exactly once with none dropped or doubled; one device falls back to the serial path; the notebook grid cell passes `visible_devices()` and shares `SESSION_GUARD`. A separate probe on 60 cells split them 30/30 |

**What the two-GPU checks do not cover.** Local torch is `2.13.0+cpu` and
`torch.cuda.is_available()` is False, so the CUDA half is exercised by faking the device report.
That proves the *routing and queue discipline* — the part that would idle half the hardware, as the
894-run session did — and proves nothing about real-device behaviour: the CUDA branch of `set_seed`,
generator isolation between two live streams, throughput, 16 GB occupancy with two concurrent
workers, and the sm_75 precision gate are all unverified until a T4 session runs.

**Operational note before that session.** Select the **T4 x2** accelerator. On a single-GPU
accelerator `visible_devices()` returns one device and the grid runs serially — correct behaviour,
no error, roughly twice the wall time. The grid cell logs
`run-level parallelism across ['cuda:0', 'cuda:1']` when both are in use; its absence is the signal
that only one was attached.

Machine evidence: `.research/audit-repair-work/pytest_results.json`, `upstream_parity.json`,
`window_budget.json`.

## Documentation — the "LAST" item is closed

`CLAUDE.md` and `paper/CLAUDE.md` were held back until implementation and checks finished, as the
user required. Both are done.

- **`CLAUDE.md`** carries an audit supersession header, the repositioned title, and corrections
  through sections 2, 3, 5.4, 6.1, 7, 8.1, 8.5, 9.1, 9.2, 10.2, 10.5, 12, 13.1, 13.2, 13.5, 14, 15
  and 16. Section 14 gains the **D90** index entry, one row per finding. Historical results are kept
  and labelled historical rather than deleted, because section 12 forbids losing provenance.
- **`paper/CLAUDE.md`** replaces four posture rules the study can no longer defend — binding
  pre-registration, the "no decay detected within 180 days" phrasing, a recommended cadence, and a
  hedged priority claim — and adds two: name which `paper_numbers.json` a number came from, and never
  blur observed results with the specified-but-untrained protocol.
- Earlier in the remediation: `README.md`, `USAGE.md`, `docs/DIVERGENCE_REGISTER.md` (D90),
  `docs/ORIGIN_WINDOW_BUDGET.md`, `docs/NOTEBOOK_MAP.md`, the research specification,
  `docs/AUDIT_REMEDIATION_2026-09-09.md` and `docs/LITERATURE_SCOPE_2026-09-10.md`.

## Still open — and none of it is a documentation task

1. **No GPU run.** The 2130-run manifest needs Kaggle T4 x2. Code, manifest and tests exist; no
   training observation of it does. Enter remaining quota, save every session output, resume until
   all 2130 ids pass validation. Analysis can finish in a CPU session afterwards.
2. **The manuscript is 6 pages against a 10-14 page target** (root section 13.1). It compiles, it has
   no TODO markers, and its numbers come from `paper/reanalysis_2026-09-09/`. It is a complete draft
   at short length, not a finished submission.
3. **Six claims stay unavailable after any rerun**, because each needs a different design rather than
   more seeds: a causal effect of participation ratio alone, the absence of predictability in Bitcoin
   generally, an optimal retraining cadence, inference from independent origins, the cause of every
   gap, and an executable profitable strategy across the whole calendar.

**Do not claim zero vulnerabilities, and do not claim the new experiments have run.** Passing tests
are not observations. No independent dated preregistration evidence was ever supplied, so the study
is described as declared-before-the-grid, never as pre-registered.
