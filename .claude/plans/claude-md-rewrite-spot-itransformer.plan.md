# Plan: CLAUDE.md Rewrite — Spot-Only iTransformer Research Programme

**Source PRD**: `.claude/prds/claude-md-rewrite-spot-itransformer.prd.md`
**Selected Milestones**: 1–7 (all). See *Milestone bundling* below.
**Complexity**: Large (documentation-only; no runtime code produced)
**Date**: 2026-08-05

## Summary

Replace the obsolete 74 KB root `CLAUDE.md` with a governing document set for the spot-only,
1-hour, walk-forward iTransformer study. The root file carries the complete programme law —
research questions, corrected variate ladder, data contract, statistical protocol, Kaggle 2×T4
execution and session-continuation protocol, anti-leakage checklist, traceability and paper
contracts. Scoped `CLAUDE.md` files carry directory-local rules only. A long-form divergence
register records all 21 corrections to the two source specifications with evidence.

No experiment runs and no pipeline code is written by this plan.

### Milestone bundling

The PRD lists 7 milestones. They describe one artifact set, not seven; splitting produces seven
plans that all edit the same files and cannot be validated independently (milestone 7 *is* the
validation of 1–6). This plan therefore selects all seven and marks all seven rows `in-progress`.
Deviation from `/plan`'s single-milestone default, stated deliberately.

---

## Patterns to Mirror

The code plane is **greenfield**: no `src/`, `tests/`, or `paper/` exists; no `.py` file remains in
the tree. No naming, error-handling, logging, data-access, or test pattern can be cited from
existing code, and none is invented here. The patterns below are **documentation and artifact**
patterns from surviving files.

| Category | Source | Pattern |
|---|---|---|
| Doc structure | `CLAUDE.md` §1–§19 (superseded) | Numbered top-level sections, hard-constraint block up front, `❌ Never / ✅ Instead` two-column rule table, command cheatsheet last. Mirror the **form**, discard every rule's content |
| Frozen-contract idiom | `CLAUDE.md` §3.2 (superseded) | Explicit *frozen fields* vs *free fields* split, with hard-assert rejection rules printed as a PASS/FAIL table before anything runs. Re-target at experiment configs instead of feature artifacts |
| Run identity | `CLAUDE.md` §2.1 (superseded) | `run_id = {profile}_L{seq_len}_H{pred_len}_d{d_model}_s{seed}` — deterministic, human-readable, changing any component deliberately orphans prior outputs. Re-target at grid runs |
| Report schema | `data/raw/BTCUSDT_1h_report.json` | Flat JSON: `bars_expected`, `bars_actual`, `coverage_pct`, `gap_blocks`, `missing_bars`, `largest_gap_bars`, `duplicate_timestamps`, `monotonic_index`, `ohlc_violations`, `per_year_coverage`. These field names are Table 1's source of truth — reference them, do not rename |
| Gap listing | `data/raw/BTCUSDT_1h_gaps.csv` | `gap_start,gap_end,missing_bars`, 27 rows. Defines segment boundaries |
| PRD/plan artifacts | `.claude/prds/*.prd.md` | Front-matter block (Created / Branch / Supersedes / Source inputs), then severity-tagged tables |

---

## Files to Change

| File | Action | Why |
|---|---|---|
| `CLAUDE.md` | **UPDATE** (full rewrite) | Root project law. Currently declares the superseded design |
| `docs/DIVERGENCE_REGISTER.md` | CREATE | Long-form evidence for the 21 corrections. Too large and too rarely read to sit in root |
| `src/CLAUDE.md` | CREATE | Code-local rules: torch-only, polars data plane, no-DataLoader training regime |
| `notebooks/CLAUDE.md` | CREATE | Thin-notebook rule; notebooks are launchers, never source of truth |
| `paper/CLAUDE.md` | CREATE | Manuscript-local rules: citation discipline, table/figure source binding |
| `README.md` | UPDATE | Currently 21 bytes. Must state what the project now is |
| `notebooks/iTransformer.ipynb` | DELETE | Superseded 1-minute reference implementation. Tracked in git — recoverable |
| `notebooks/01_preprocess.ipynb` | DELETE | Superseded. Tracked |
| `notebooks/02_train.ipynb` | DELETE | Superseded. Tracked |
| `notebooks/markdown-example.ipynb` | KEEP | Unrelated scratch file, harmless |
| `.claude/prds/claude-md-rewrite-spot-itransformer.prd.md` | UPDATE | Milestone rows 1–7 → `in-progress`, `Plan` cell → this file |
| `pyproject.toml` | **NOT TOUCHED** | Environment rebuild is out of scope per PRD. Requirement recorded in doc only |

Directories `docs/`, `src/`, `paper/` exist or are created empty; `src/` and `paper/` receive only
their scoped `CLAUDE.md` in this plan.

---

## Tasks

### Task 1: Tombstone and purge (PRD milestone 1)
- **Action**: Delete the three superseded notebooks via `git rm` (recoverable from commit `ee55c9d`
  and earlier). Rewrite `README.md` to state the current project in ~10 lines. Draft the single
  tombstone paragraph that will close the new root `CLAUDE.md`: what the project was, that it was
  superseded on 2026-08-05, and that rationale lives in git history — nothing more.
- **Mirror**: none applicable (deletion + prose).
- **Validate**: `git status` shows three staged deletions; searching the tree for `Gold`, `XAU`,
  `US_Dollar_Index`, `fracdiff`, `TimeXer` returns hits only inside the two source `.md` files and
  the tombstone.

### Task 2: Divergence register (PRD milestone 2)
- **Action**: Create `docs/DIVERGENCE_REGISTER.md`. One row per correction, IDs `D01`–`D21`,
  matching the PRD's numbering exactly. Each row carries six fields: **Severity** (F/C/U/I),
  **Source location** (which file and section), **Defect** (what the source says and why it is
  wrong), **Resolution** (what this project does instead), **Evidence** (the measurement, the
  arithmetic, or the citation that settles it), **Paper disclosure** (which manuscript section must
  state the deviation). Reserve `D22+` for corrections found later.
- **Mirror**: PRD severity tagging; `data/raw/BTCUSDT_1h_report.json` field names when citing
  measured evidence.
- **Validate**: 21 rows present, all six fields populated on every row, no row says "TBD" in
  Resolution.

### Task 3: Root CLAUDE.md — §1 to §9, the research programme (PRD milestone 3)
- **Action**: Write the scientific half of the root document.
  - **§1 Project definition** — the deliverable is a manuscript, not a model. No production
    inference path exists.
  - **§2 Hard constraints** — non-negotiable block: **PyTorch is the only deep-learning framework;
    TensorFlow, Keras, and JAX are forbidden and must not be added, imported, or suggested**;
    spot-only BTCUSDT 1 h, no futures variates; **no imputation of any kind** (no ffill, no bfill,
    no interpolation, no reindexing to a full grid); segmentation instead; **polars is the data
    plane, pandas is not**; UTC throughout.
  - **§3 Research questions** — RQ1/RQ2/RQ3 with pre-registered dependent variables and the
    τ ∈ {2.5%, 5%, 10%} threshold set, headline τ = 5%. RQ2 compares K=1 against K=8, never K=12.
  - **§4 Data contract** — measured facts as a table (75,216 expected bars, 75,094 actual,
    27 gap blocks, 122 missing, largest 33, coverage 99.838%, 0 OHLC violations, 0 duplicates,
    monotonic, 3 zero-volume bars). Gaps are exchange downtime, not missing values — Rubin's
    taxonomy stated and explicitly ruled inapplicable. Window-loss estimate `119 × 27 + 122 = 3,335
    ≈ 4.4%`, reconciled per origin as a hard assertion. **D10**: the parquet on disk was written
    `fill_policy: "ffill"` with 122 synthetic bars and must not be consumed as-is.
  - **§5 Variate ladder and feature-engineering policy** — the corrected cut (**D01**):
    K=1 `{r}`; K=4 `+ {upper_shadow, lower_shadow, log_quote_volume}`;
    K=8 `+ {log_trade_count, taker_buy_ratio, signed_flow, vwap_location}`;
    K=12 `+ {Parkinson, Garman–Klass, Rogers–Satchell, log_mean_trade_size}`. Families F1–F5 total
    12. Plus the **feature-engineering policy** subsection: all 12 variates are engineered
    transforms of the 11 raw kline columns; technical indicators, multi-bar rolling statistics,
    calendar dummies, cross-asset, on-chain, sentiment and macro are excluded *by design*, on the
    linear-span argument — any linear function of the L=96 lookback already lies inside the span of
    `Linear(96 → d_model)`, so it raises nominal K while adding ≈0 effective rank, which is the
    exact phenomenon RQ1 measures. Pin the underspecified items: F2 per-bar vs trailing-averaged
    (**D13**), `taker_buy_ratio` denominator (**D14**), the `H == L` degenerate-bar rule which may
    not be an imputation (**D14**), and the `signed_flow` dependence disclosure (**D12**).
  - **§6 Model specification** — architecture, `d_model=128` with its N-not-L justification, the
    K=1 attention-degeneracy note as a *designed control*, the K=12 rung as a *designed redundancy
    control*, `use_norm=True` as mandatory, and the corrected scale-invariance test
    `MSE(c·x)/c² == MSE(x)` (**D03**). State the instance-norm/volatility-level confound (**D04**)
    and the requirement to measure participation ratio on window-normalised features as well as raw.
  - **§7 Baselines** — Naive-RW (`ŷ = 0`, never last-return), Naive-persist, Seasonal-naive, ARIMA,
    LSTM, DLinear, PatchTST, **plus multivariate ridge on the same K features** (**D17**).
  - **§8 Walk-forward protocol** — 13 origins (2020-01 … 2026-01) derived from the 2018-01 start,
    with the false 2017-08 attribution deleted (**D07**); rolling not expanding, with reason;
    24-month train / 21-month scaler sub-block / 3-month validation / H-step purge; the written
    embargo justification (**D15**); the CPCV rejection paragraph; timestamp-validated windows.
  - **§9 Metrics and statistical tests** — MSE/MAE/RelMSE/DA, `R²_oos` alongside RelMSE (**D20**),
    `A(b)` and the **RelMSE-based** `D(b)` (**D05**), the decay regression with origin fixed effects
    `A(i,b) = αᵢ + β₁·b + ε`, clustered by origin, wild cluster bootstrap with **Webb 6-point
    weights** at 13 clusters, and the cross-origin calendar-adjacency disclosure (**D06**).
    DM with Newey–West lag h−1 and the HLN small-sample correction against Student-t(T−1).
    Seeds: 5 for the K∈{1,8} pair, 3 elsewhere (**D18**). MAPE forbidden on log-returns.
    Deflated Sharpe with N = trials actually run, and the H=24 trading-rule/DA-step definitions
    pinned (**D21**).
- **Mirror**: superseded `CLAUDE.md` §1–§2 block layout; §16's two-column rule table for §2.
- **Validate**: every `D01`–`D21` ID that belongs to the science half appears in the text; the K
  ladder membership lists sum to 1/4/8/12.

### Task 4: Root CLAUDE.md — §10, Kaggle execution and continuation (PRD milestone 4)
- **Action**: Write the execution protocol against the stated hardware envelope.
  - **Envelope**: 2×T4 (sm_75, 16 GB each), **12 h per session**, **30 GPU-h per week**, 20-minute
    interactive idle timeout, `/kaggle/working` 20 GB and saved as version output,
    `/kaggle/input` read-only.
  - **Run accounting**: main grid 13 origins × 4 K × 3 seeds = 156, plus 52 extra seed runs for the
    K∈{1,8} pair at 5 seeds; horizon sweep 4 origins × 4 K × 4 H × 3 seeds = 192 (**D08** resolves
    the 4-vs-3 conflict in favour of 4); baselines 169; ridge 52. **≈ 621 runs.**
  - **Cost model with its assumptions stated** (**D19**): per-origin training tensor is
    ~17,400 × 96 × 12 × 4 B ≈ **80 MB** and must be **resident in VRAM**; the per-item
    `Dataset`/`DataLoader` path is bypassed entirely and batches are index-slices of a GPU tensor.
    Under that regime ~60–100 s per run ⇒ **≈ 6–15 wall-hours across 2 GPUs**, inside one session
    and one week's quota with slack. Under a naive 4-worker DataLoader the same grid is roughly an
    order of magnitude worse and **exceeds the weekly quota** — state both numbers so the regime is
    understood as load-bearing, not stylistic.
  - **Both GPUs**: two workers, each pinned to one `cuda:N`, pulling from a shared run queue.
    `nn.DataParallel` is explicitly rejected at batch 32 — it costs more in transfer than it saves.
  - **Run identity**: `run_id = {model}_o{origin:02d}_K{K:02d}_H{H:03d}_s{seed}`, e.g.
    `itr_o07_K08_H024_s42`. Mirrors the superseded `run_id` idiom: deterministic, human-readable,
    and any change deliberately orphans prior outputs rather than silently reusing them.
  - **Per-run outputs**: `preds/{run_id}.parquet` (block, step, y_true, y_pred) and
    `meta/{run_id}.json` (resolved config, git sha, input-artifact hash, epochs, best val, wall
    time, `status`). Estimated total ≈ 0.5–2 GB — comfortably inside 20 GB.
  - **Idempotence and resume**: a run counts as complete only when both files exist *and*
    `status == "complete"`. Anything else is re-run from scratch — intra-run checkpointing is not
    worth the complexity at ~90 s per run. Resume scans `/kaggle/input/*/preds/` ∪
    `/kaggle/working/preds/` by glob discovery, never a hard-coded dataset slug, subtracts completed
    `run_id`s from the manifest, and executes the remainder.
  - **Budget guard**: `SESSION_BUDGET_H = 11.0`, `RESERVE_H = 0.5`, checked **at run boundaries**
    (not epoch boundaries — runs are short). On trip: stop, flush, print the remaining count, exit
    cleanly so the version saves. Hitting Kaggle's own 12 h wall interactively loses
    `/kaggle/working` entirely.
  - **Session chaining**: grid execution uses *Save Version → Save & Run All* (batch), never the
    interactive editor, because the 20-minute idle timeout kills long interactive runs. Session N's
    output dataset becomes session N+1's input.
  - **Precision**: T4 is sm_75. `torch.cuda.is_bf16_supported()` returns True there via emulation
    and is **slower than fp32** — gate on `torch.cuda.get_device_capability(0)[0] >= 8` instead and
    fall back to fp16 + `GradScaler`, then fp32. At this model size fp32 is likely fastest; measure.
- **Mirror**: superseded `CLAUDE.md` §2.1 run_id and budget-guard idioms.
- **Validate**: the section states the run count, the GPU-hour estimate, the assumptions behind it,
  the counterfactual cost, and the resume rule. All five present.

### Task 5: Root CLAUDE.md — §11 to §17, discipline and contracts (PRD milestone 6)
- **Action**:
  - **§11 Anti-leakage checklist** — the corrected list. The superseded `ffill`-permitting rule is
    gone; no rule may permit an action and its negation. Mark each item as either a hard assertion
    in the pipeline or explicit manual review. The five fatal items called out as fatal.
  - **§12 Traceability contract** — no number enters the manuscript unless regenerable from a named
    persisted artifact plus a config hash. Naming that makes this checkable.
  - **§13 Paper production** — IMRaD skeleton mapped to the RQs; Tables 1–8 and Figures 1–7 each
    bound to the artifact and stage that generates them; the mandatory-disclosure list (K=1
    degeneracy, K=12 designed redundancy, CPCV rejection, `use_norm`/scaler relationship,
    no-imputation defence, attention-is-not-explanation caveat); **citation discipline: no citation
    enters the manuscript without a verified DOI and the source read** (**D16**), with the known
    bad entries flagged.
  - **§14 Divergence summary** — compact table, `D01`–`D21`, ID + severity + one-line defect +
    one-line resolution, pointing at `docs/DIVERGENCE_REGISTER.md` for evidence.
  - **§15 Repository layout and commands** — target tree; the rule that logic lives in an importable
    package and notebooks are launchers.
  - **§16 Working conventions** — Python ≥ 3.11 syntax, type hints, no magic numbers, and the
    standing rule that the two source `.md` files are **inputs, not authority**: where they
    disagree with this document, this document wins and the register says why.
  - **§17 Tombstone** — the paragraph from Task 1.
- **Mirror**: superseded `CLAUDE.md` §16 rule-table form and §19 conventions/cheatsheet layout.
- **Validate**: root byte size ≤ ~30,000; no forbidden-framework mention outside the prohibition
  itself; every `D01`–`D21` ID appears in §14.

### Task 6: Scoped CLAUDE.md set (PRD milestone 5)
- **Action**:
  - `src/CLAUDE.md` — **torch only, never TensorFlow/Keras/JAX**. **polars is the data plane**:
    ingest, validation, segmentation, and feature construction use polars lazy scans; pandas is
    permitted *only* at the single conversion boundary where a statistics library demands it
    (`statsmodels`, `arch`, `wildboottest`), and that boundary must be named in code. Record the
    correctness argument, not just the speed one: polars' rolling API is backward-closed by
    construction, so the `rolling(center=True)` leak the spec calls fatal is **unrepresentable** —
    unlike pandas, where it is one keyword away. The spec's §6.2 reference snippet is pandas and
    must be re-expressed in polars/numpy. Training touches no DataFrame at all: pre-built
    GPU-resident tensors, index-slice batching, no `DataLoader`.
  - `notebooks/CLAUDE.md` — a Kaggle notebook is a thin launcher: install the package, discover
    inputs, run the queue, save. Logic in a notebook is a defect. The superseded
    generate-notebooks-from-a-source-notebook workflow is dead.
  - `paper/CLAUDE.md` — every number cites its artifact; every citation cites a verified DOI;
    deviations from the pre-registered design are disclosed, not smoothed over.
- **Mirror**: root document's rule-table form, condensed.
- **Validate**: each scoped file is under ~4 KB and contains no rule already stated in root.

### Task 7: PRD synchronisation
- **Action**: In `.claude/prds/claude-md-rewrite-spot-itransformer.prd.md`, set milestone rows 1–7
  `Status` → `in-progress` and `Plan` → `.claude/plans/claude-md-rewrite-spot-itransformer.plan.md`.
  Append the F6-rung question (below) to Open Questions.
- **Validate**: seven rows updated; no other PRD content altered.

### Task 8: Consistency validation pass (PRD milestone 7)
- **Action**: Read the finished set end to end as a cold session would. Check every success metric
  in the PRD. Produce a short pass/fail report against the seven metrics and the 21 divergence IDs.
  Any contradiction found becomes `D22+` and is fixed, not noted.
- **Validate**: the commands below all return the expected result.

---

## Validation

No test suite exists; validation is documentary and mechanical.

```bash
# 1. Root size ceiling (target <= ~30000 bytes; superseded file was 74715)
wc -c CLAUDE.md

# 2. No superseded vocabulary survives outside the tombstone and the source .md files
grep -niE 'gold|xau|dollar.index|fracdiff|frozen artifact|multi-granularity|1-minute' CLAUDE.md

# 3. Forbidden frameworks appear only inside the prohibition
grep -rniE 'tensorflow|keras|jax' CLAUDE.md src/CLAUDE.md notebooks/CLAUDE.md paper/CLAUDE.md

# 4. All 21 divergence IDs present in both the root summary and the long-form register
grep -oE 'D[0-2][0-9]' CLAUDE.md | sort -u | wc -l          # expect 21
grep -cE '^\| *D[0-9]{2}' docs/DIVERGENCE_REGISTER.md        # expect 21

# 5. pandas confined to the named stats boundary
grep -n 'pandas' src/CLAUDE.md

# 6. Legacy notebooks removed, recoverable
git status --short notebooks/
git log --oneline -1 -- notebooks/iTransformer.ipynb

# 7. Ladder arithmetic stated correctly somewhere in root
grep -nE 'K=8|K=12' CLAUDE.md | head
```

Manual gate (cannot be automated): read §2, §5, §11 consecutively and confirm no rule permits both
an action and its negation.

---

## Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Root file overshoots the size ceiling and rebuilds the 74 KB monolith | **high** | Ceiling checked as validation step 1. Overflow is pushed into `docs/DIVERGENCE_REGISTER.md` or a scoped file, never absorbed |
| "Everything from the PRD in CLAUDE.md" conflicts with "root ≤ 25 KB" | **high** | Resolved by splitting *evidence* from *rules*: root carries every rule and a compact 21-row divergence summary; long-form evidence lives in `docs/`. Root stays self-sufficient for execution |
| Deleting three tracked notebooks loses reference material | low | All three are tracked (`git ls-files` confirms); recoverable from `ee55c9d` and earlier. Deletion staged, not forced |
| A correction (D02, D04, D05, D06) changes headline results vs the pre-registered source design | medium | Each is a register row with a mandatory paper-disclosure field. Deviation is published, not hidden |
| polars mandate collides with the spec's pandas reference implementation and with stats libraries | **high** | `src/CLAUDE.md` names the single conversion boundary explicitly. The spec's §6.2 snippet is marked for re-expression rather than copied |
| `requires-python >= 3.14` has no torch wheel, so the local CPU verification path does not exist | medium | Already an open question in the PRD; this plan records the requirement without touching `pyproject.toml`. Must be resolved before any code plan |
| Kaggle cost estimate (6–15 h) proves optimistic | medium | The estimate ships with its assumptions and its counterfactual. First real timing measurement replaces it, and the resume protocol makes an overrun survivable rather than fatal |
| New contradictions discovered during writing are absorbed silently | medium | Task 8 assigns them IDs `D22+`. Silent absorption is the exact failure being fixed |

---

## Open Question raised by this plan

**Does the ladder gain an optional F6 rung at K≈16?** Candidate variates are nonlinear in the
lookback and therefore genuinely outside the span of `Linear(96 → d_model)`: trailing realized
variance, signed-flow autocorrelation, VPIN-style toxicity. Framed as an explicit **fifth rung** it
strengthens RQ1 (a second, larger high-nominal-K / low-K_eff control); folded into the existing four
it destroys the instrument. Recommendation: pre-register it as optional, run only if the Stage 5
pilot passes and quota allows. **Needs a decision before Task 3 writes §5.**

---

## Acceptance

- [ ] All 8 tasks complete
- [ ] All 7 validation commands return the expected result
- [ ] Manual contradiction gate passes (§2, §5, §11 read consecutively)
- [ ] All 7 PRD success metrics met
- [ ] `D01`–`D21` each resolved in the register with all six fields populated
- [ ] Root `CLAUDE.md` is self-sufficient: a cold session needs neither source `.md` to execute
- [ ] Patterns mirrored from the superseded document's **form**, with none of its rules surviving

---

**WAITING FOR CONFIRMATION**: proceed with this plan? (yes / no / modify)
