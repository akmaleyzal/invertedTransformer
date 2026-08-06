# CLAUDE.md Rewrite — Spot-Only iTransformer Research Programme

**Created:** 2026-08-05
**Branch:** `rewrite/new-topic`
**Supersedes:** the root `CLAUDE.md` dated pre-2026-08-04 (1-minute BTC + Gold/USD/Macro fusion, production-model deliverable)
**Source inputs:** `research_specification_itransformer_btc.md`, `reference_library_itransformer_btc.md`

---

## Problem

The project pivoted from *shipping a production forecasting model* to *publishing a walk-forward
research paper*, and the repository's governing document did not move with it. The root
`CLAUDE.md` (74 KB) still declares a 1-minute, four-source, multi-granularity fusion pipeline whose
deliverable is a TorchScript/ONNX export bundle. The actual project is a 1-hour, single-asset,
spot-only study whose deliverable is a manuscript answering three research questions about
effective dimensionality and temporal decay.

Every rule in the current document is therefore either irrelevant or **actively wrong** for the new
design. The most dangerous class is the second: the current anti-leakage table explicitly permits
`fillna(method="ffill")` "with a staleness feature", while the new specification proves forward-fill
is fatal here — it fabricates a zero-then-jump return and zeroes the Parkinson and Garman–Klass
estimators at exactly the bar before a volatility event. A future session that follows the
document as written will corrupt the study and produce numbers that look plausible and cannot be
defended.

Cost of leaving it unsolved: the paper's methodology section is generated from a document that
contradicts its own protocol, and no reviewer challenge can be answered from the repository.

---

## Evidence

Concrete, measured or verified in this repository on 2026-08-05:

- **The data on disk already violates the new specification.**
  `data/raw/BTCUSDT_1h_report.json` records `"fill_policy": "ffill"`, `"rows_written": 75216`,
  `"synthetic_bars": 122`. The specification's §2.3 forbids filling of any kind. The artifact the
  pipeline would consume today is non-compliant.
- **The specification's own variate ladder does not add up.** K=8 is defined as K=4 plus
  "remainder of F3, F4, F5" = 2 + 2 + 1 = 5 variates on a base of 4, giving **9**.
  `log_mean_trade_size` is simultaneously assigned to the K=8 rung and to the K=12 rung.
- **A design gate reads test-period data.** Stage 3b computes the participation ratio over the full
  2018–2026 sample and uses it to re-cut the K ladder. Every origin's test block is inside that
  sample.
- **The `use_norm` verification test as written cannot pass.** Multiplying the scaled input by 100
  multiplies the target by 100 as well (same array), so the loss scales by 10⁴. The specification
  asserts the two losses "must be identical to numerical precision".
- **Internal count disagreements.** §6.4 states a 4-value K sweep for the horizon experiment;
  Stage 8 of the same document states 3. The reference library says "your 12 origins"; the
  specification says 13.
- **A pipeline entry point is missing.** `binance_spot_klines.py` is referenced as Stage 1 in the
  specification and is no longer present in the working tree. Its raw JSONL cache
  (`BTCUSDT_1h_raw.jsonl`) survives, so the data is re-derivable offline, but the tool is not.
- **The environment does not match either document.** `pyproject.toml` declares three dependencies
  (pandas, pyarrow, requests) and `requires-python >= 3.14`; `requirements.txt` is a UTF-16 dump of
  the previous project's stack (polars, duckdb, optuna). Neither torch, statsmodels, nor `arch` is
  declared, and all three are load-bearing in the new protocol.
- **Data quality is measured, not assumed.** 27 gap blocks, 122 missing bars, largest gap 33 bars,
  coverage 99.838%, zero OHLC violations, zero duplicates, monotonic index. Estimated window loss
  `119 × 27 + 122 = 3,335`, about **4.4%** — comfortably inside the specification's 16% tolerance,
  so no narrowing of the analysis window is required.
- **Origin count verified independently.** With the window 2018-01-01 → 2026-08-01, a fixed 24-month
  training span and 180-day test span at 6-month spacing yields origins 2020-01 … 2026-01 = **13**,
  final test ending 2026-06-30. The specification's stated cause ("spot history starting 2017-08 is
  what makes thirteen possible") is false; 13 follows from the 2018-01 start it already declares.

*Assumption — needs validation:* that the local venv can be rebuilt for the new stack on Python
3.14 (torch wheel availability for 3.14 is unconfirmed at time of writing).

---

## Users

- **Primary — a future Claude Code session in this repository.** Arrives with no memory of this
  conversation, loads `CLAUDE.md` before doing anything, and executes whatever it says. It cannot
  distinguish a stale rule from a current one. This is the user the document is actually written
  for, and the reason contradictions are fatal rather than untidy.
- **Secondary — the author, when a reviewer or examiner asks "why did you do it that way?"**
  Needs every methodological choice to have a written, cited, dated justification that can be read
  aloud without improvisation.
- **Tertiary — a replicator** who has the repository and the paper and wants to reproduce Table 5.

- **Not for:** anyone consuming a deployed forecasting service. There is no production inference
  path in this project; the model is an experimental instrument, not a shipped artifact.

---

## Hypothesis

We believe **a rewritten, internally consistent, contradiction-free CLAUDE.md set — with every
departure from the two source specifications recorded and justified — will make every number in the
resulting manuscript defensible** for **the author under review and for future sessions executing
the pipeline**.

We'll know we're right when **every claim in the manuscript can be traced, without improvisation, to
(a) a persisted prediction file, (b) a configuration hash, and (c) a documented decision with a
rationale — and when an independent read of the document set surfaces zero unresolved
contradictions against the measured data and against itself.**

---

## Success Metrics

| Metric | Target | How measured |
|---|---|---|
| **Unresolved contradictions** in the doc set vs. the two source `.md` files and vs. the measured data | **0** | Each of the 21 defects below appears in the divergence register with a stated resolution; a full read-through finds no rule stated two incompatible ways |
| **Divergence register completeness** | 100% of departures from source `.md` carry: what changed, why, evidence, and the paper section that must state it | Register row count ≥ 21; every row has all four fields populated |
| **Leakage rules unambiguous** | Every rule in the anti-leakage checklist is either a hard assertion in the pipeline or explicitly marked "manual review" | No rule permits both an action and its negation; the old `ffill`-permitting table is gone |
| **Traceability contract** | Every table and figure in the paper skeleton names the artifact it is generated from | Each of Tables 1–8 and Figures 1–7 has a named source artifact and a generating stage |
| **Kaggle budget realism** | Documented GPU-hour estimate for the full 517-run grid, with the resume protocol that survives a 12 h session cut and a 30 h/week quota | Estimate present with its assumptions stated; resume protocol specifies what makes a run idempotent |
| **Doc size discipline** | Root file stays navigable | ~~Root ≤ ~25 KB~~ — **MISSED. Actual: 50,160 bytes** (was 74,715; −33%). Cause: a direct conflict with the instruction *"claude.md yang berisi semua yang ada di prd"*, resolved toward the explicit instruction. Recorded rather than silently rewritten. Clean cut lines if slimming is later preferred: §13 paper production (~5 KB) and §9.2 statistical detail (~4 KB), both of which already have scoped homes |
| **Zero orphaned rules** | No rule in the new set references the 1-minute / Gold / USD / macro design | Searching for the superseded vocabulary returns only the tombstone paragraph |

---

## Scope

### MVP — the minimum to test the hypothesis

A **CLAUDE.md set** that a cold session can follow end-to-end without consulting either source
`.md`, comprising:

1. **A root `CLAUDE.md`** — project law. Definition of the deliverable (a manuscript, not a model),
   the three research questions and their pre-registered dependent variables and thresholds, the
   hard constraints, the anti-leakage checklist in its corrected form, the execution environment and
   session-continuation protocol, and pointers to the scoped files.
2. **Scoped `CLAUDE.md` files** in the directories where local rules apply — code, notebooks,
   and manuscript — each carrying only what is specific to that directory.
3. **A divergence register** — every place the new document departs from
   `research_specification_itransformer_btc.md` or `reference_library_itransformer_btc.md`, with the
   defect, the resolution, the evidence, and the paper section that must disclose it. This is the
   artifact that makes "dipertanggungjawabkan" enforceable rather than aspirational.
4. **A tombstone** — one paragraph recording that the 1-minute multi-source design was superseded on
   2026-08-05 and pointing at git history. No other trace of it survives in documentation.
5. **A traceability contract** — the rule that a number may not enter the manuscript unless it is
   reproducible from a persisted artifact, plus the naming that makes that checkable.
6. **A paper-production section** — IMRaD skeleton mapped to the RQs, the table and figure
   inventory with their source artifacts, citation discipline (no citation without a verified DOI
   and the source read), venue conventions, and the standing instruction that the methodology must
   state the K=1 degeneracy, the deliberate redundancy of the K=12 rung, the CPCV rejection, the
   `use_norm`/scaler relationship, the no-imputation defence, and the attention-is-not-explanation
   caveat.

### Contradictions the doc set must resolve

Acceptance criteria. Each row must appear in the divergence register with the stated resolution.
Severity **F** = fatal to the paper's validity if unresolved; **C** = internal contradiction;
**U** = underspecified; **I** = improvement over the source.

| # | Sev | Where | Defect | Required resolution |
|---|---|---|---|---|
| 1 | **F** | spec §2.2 | K=8 rung sums to 9; `log_mean_trade_size` double-assigned | Re-cut so nominal counts match membership. Exactly one consistent cut exists: **K=8 = K=4 + {`log_trade_count`, `taker_buy_ratio`, `signed_flow`, `vwap_location`}**; **K=12 = K=8 + {Parkinson, Garman–Klass, Rogers–Satchell, `log_mean_trade_size`}**. Families total 12 |
| 2 | **F** | spec §8 Stage 3b | K_eff gate computed on the full 2018–2026 sample, then used to re-cut the ladder — a design decision informed by test-period data | Gating participation ratio must be computed on the **pre-first-origin span only** (2018-01 → 2020-01). Full-sample rolling PR remains, labelled descriptive-only, and may inform no design choice |
| 3 | **F** | spec §4.4 | `use_norm` verification test asserts identical losses after ×100 input scaling; the target scales too, so loss scales ×10⁴ | Correct invariant: `MSE(c·x)/c² == MSE(x)`, equivalently RelMSE unchanged. State the corrected test |
| 4 | **F** | spec §4.4 vs §2.2 | Instance normalisation divides each window by its own per-variate σ over L, so the F2 estimators contribute *shape*, not *level*. The 8→12 rung can flatten for a reason unrelated to redundancy — confounding RQ1's independent variable | Measure participation ratio on **window-normalised** features in addition to raw. Report both. Disclose the confound in Limitations regardless of outcome |
| 5 | **F** | spec §7.1 | `D(b) = [MSE(b) − MSE(1)] / MSE(1)` compares different calendar periods, so it conflates model decay with the market getting harder | Define `D(b)` on **RelMSE** (model vs. Naive-RW within the same block). `A(b)` needs no such control — same block, ratio cancels |
| 6 | **F** | spec §7.2 | Decay regression `A(b) = β₀ + β₁b + ε` omits origin fixed effects; β₁ absorbs origin-level difficulty | Specify `A(i,b) = αᵢ + β₁·b + ε` with origin fixed effects, clustered by origin, wild cluster bootstrap. At 13 clusters prefer **Webb 6-point weights** over Rademacher. Note that origin *i* block 6 is calendar-adjacent to origin *i+1* block 1, so cluster-by-origin is necessary but not sufficient; disclose |
| 7 | **C** | spec §6.1 | "Spot history starting 2017-08 is what makes thirteen possible" | False. 13 follows from the declared 2018-01-01 start. Either delete the claim, or extend ingestion to 2017-08 and re-derive the origin count — but not both silently |
| 8 | **C** | spec §6.4 vs §8 Stage 8 | Horizon sweep is 4 K values in one place, 3 in the other | Pin one. If 4 K × 4 H × 4 origins × 3 seeds = 192 is kept, correct Stage 8 |
| 9 | **C** | ref library §D | "your 12 origins" | 13. Correct |
| 10 | **F** | repo state | `BTCUSDT_1h.parquet` written with `fill_policy: "ffill"`, 122 synthetic bars — the exact operation §2.3 forbids | The pipeline must consume **unfilled** bars. Either regenerate with no fill, or drop rows flagged synthetic before any feature is computed. The gap listing defines segment boundaries and is retained as a diagnostic |
| 11 | **C** | repo state | `binance_spot_klines.py` referenced as Stage 1, absent from the tree | Restore or re-specify Stage 1. The raw JSONL cache makes offline re-derivation possible without re-hitting the API |
| 12 | **U** | spec §2.2 | `signed_flow = (2·ratio − 1)·log_volume` is a deterministic product of two other K=8 members, weakening "K=8 is the rung of maximum effective rank" | Keep it, but disclose the dependence and let the measured PR at K=8 settle it. If PR is materially below the ~6.5 estimate, the ladder re-cut of item 2 applies |
| 13 | **U** | spec §2.2 | F2 estimators: per-bar or trailing-window average? Per-bar Parkinson at 1 h is extremely noisy | Pin the choice and its window explicitly; a rolling variant must be backward-looking and closed on the right |
| 14 | **U** | spec §2.1 | `taker_buy_ratio` denominator unspecified (base vs. quote); `(VWAP − C)/(H − L)` divides by zero on a flat bar, and 3 zero-volume / zero-trade bars exist in the data | Pin the denominator. Specify the degenerate-bar rule explicitly — and it may not be an imputation |
| 15 | **U** | spec §6.1 | Embargo marked "not applied (justified)" with no justification written | Write the justification, or apply an embargo. Silence here is exactly what a reviewer flags |
| 16 | **C** | ref library §P, §H | arXiv 2509.23494 dated 2026 (the identifier is 2025); *Symmetry* vol. 18 dated 2025 (vol. 17 is 2025); several entries self-declared "assembled from memory" | Standing rule: **no citation enters the manuscript without a verified DOI and the source read.** Mark every unverified entry as such until cleared |
| 17 | **I** | spec §5 | Baselines control for architecture (K=1 iTransformer) but not for "is a transformer needed at all" | Add a **multivariate ridge/linear baseline on the same K features**. Separates "does the extra information help" from "does cross-variate attention help". Cost is seconds per run |
| 18 | **I** | spec §4.2 | 3 seeds is too few to report `mean ± std` as a headline | 5 seeds minimum for the RQ2 pair (K∈{1,8}); 3 acceptable elsewhere. At this model size the marginal cost is minutes |
| 19 | **I** | spec §6.4 | "the added cost is hours, not days" is true only under one implementation regime | Document the regime: the full training tensor is ~80 MB and must be **GPU-resident**, with the per-item DataLoader bypassed. Under a naive loader the grid is roughly an order of magnitude more expensive. On 2×T4, run **two independent runs concurrently, one per GPU** — data parallelism at batch 32 is a net loss |
| 20 | **I** | spec §7.1 | RelMSE against `ŷ = 0` is exactly `1 − R²_oos` and will sit near 0.99–1.00 | Report `R²_oos` alongside; it is the readable form of the same quantity |
| 21 | **U** | spec §8 Stage 11 | Trading rule undefined at H=24 — which step drives the position? Directional accuracy on which step? | Pin both. Also require the Deflated Sharpe Ratio with N = trials actually run, since the grid is ~517 runs |

### Out of scope

- **Writing the manuscript.** The doc set governs how it gets written; it is not the manuscript.
- **Running any experiment.** No origin, no grid, no pilot. This PRD produces documentation only.
- **Re-deriving the data.** The existing raw cache and gap report are treated as given evidence;
  regenerating the unfilled parquet is downstream work, not part of the doc rewrite.
- **Implementing the code layout.** The doc declares that logic lives in an importable package with
  a thin Kaggle notebook; building it is a separate plan.
- **Rebuilding the environment.** The doc records the dependency requirement; the install is
  downstream.
- **Resurrecting the 1-minute design.** Superseded. One tombstone paragraph, no appendix — two
  incompatible versions of the same leakage rule is precisely the failure mode being fixed.
- **Multi-asset or futures variates.** Rejected upstream by user decision and by the spot-only
  constraint; the doc records the rejection and its cost, and does not reopen it.

---

## Decisions already fixed (2026-08-05)

Recorded so a later session does not relitigate them.

| Decision | Choice | Consequence |
|---|---|---|
| Document topology | Root `CLAUDE.md` + scoped per-directory files | Root stays navigable; local rules live next to the code they govern |
| Old project material | Purged, one-paragraph tombstone | No surviving contradictory rule; rationale recoverable from git |
| Code home | Importable package + thin Kaggle notebook | Pipeline becomes unit-testable on CPU locally; the notebook stops being production code |
| Doc authority | Full programme, including manuscript production | Traceability is enforceable from day one rather than retrofitted |
| Execution target | Kaggle 2×T4, retained | Session-continuation and quota protocol are first-class content, not a footnote |

---

## Delivery Milestones

<!-- Status: pending | in-progress | complete -->

| # | Milestone | Outcome | Status | Plan |
|---|---|---|---|---|
| 1 | Tombstone and purge | The repository no longer states a project definition that contradicts the current one. A cold session reading it cannot arrive at the 1-minute fusion design | complete | [plan](../plans/claude-md-rewrite-spot-itransformer.plan.md) |
| 2 | Divergence register | All 21 rows above exist with defect, resolution, evidence, and the paper section that must disclose it. The author can answer any "why did you deviate?" from one file | complete | [plan](../plans/claude-md-rewrite-spot-itransformer.plan.md) |
| 3 | Root CLAUDE.md | A cold session can state the deliverable, the three RQs with pre-registered thresholds, the hard constraints, and the corrected anti-leakage checklist without opening any other file | complete | [plan](../plans/claude-md-rewrite-spot-itransformer.plan.md) |
| 4 | Kaggle execution and continuation protocol | The documented path survives a 12 h session cut and a 30 h/week quota: work is resumable, partially-completed grids are recoverable, and the GPU-hour estimate states its assumptions | complete | [plan](../plans/claude-md-rewrite-spot-itransformer.plan.md) |
| 5 | Scoped CLAUDE.md set | Rules governing code, notebooks, and manuscript live beside what they govern; the root does not grow back into a monolith | complete | [plan](../plans/claude-md-rewrite-spot-itransformer.plan.md) |
| 6 | Traceability and paper contract | Every table and figure names its source artifact; no number may enter the manuscript without a persisted prediction file and a configuration hash; citation discipline is stated as a hard rule | complete | [plan](../plans/claude-md-rewrite-spot-itransformer.plan.md) |
| 7 | Consistency validation pass | An independent read of the finished set against the measured data and against itself surfaces zero unresolved contradictions | complete — **with a stated gap** | [plan](../plans/claude-md-rewrite-spot-itransformer.plan.md) |

**Milestone 7, honestly reported.** The five-lens adversarial audit (`wf_becd8b67-d95`, 2026-08-05)
completed **three** lenses — defensibility, leakage, statistics — producing 43 findings. Its verifier
stage and the remaining two lenses (**consistency**, **Kaggle/execution**) died on a session limit,
so every finding was adjudicated instead by direct re-derivation against the document text and the
artifacts on disk. Twenty-eight survived as **D23–D50**; ten are fatal; all are fixed in the
documents, not merely noted. The milestone's success criterion — "zero unresolved contradictions" —
is met **for the three lenses that ran**. The Kaggle lens is the only one that would have audited §10
end to end, and §10 is precisely where `D25` found a 14% error in every sample-count and tensor-size
figure, so its absence is a real gap and is carried as an open item in the register. Claiming a clean
pass here would be the failure the register exists to prevent.

---

## Open Questions

- [ ] **Does the analysis window start 2018-01 or 2017-08?** The specification declares 2018-01 and
      then credits 2017-08 for the origin count. 2017-08 would add roughly one further origin. Data
      in hand covers 2018-01 onward only, so extending means a re-fetch. *Impact:* origin count,
      Table 1, and every per-origin aggregate.
- [ ] **What is the pinned Python and torch version?** `pyproject.toml` requires ≥ 3.14; torch wheel
      availability for 3.14 is unverified, and Kaggle ships its own image regardless. *Impact:*
      whether the local CPU verification path is viable at all.
- [ ] **Is `binance_spot_klines.py` restored, or is Stage 1 re-specified?** The raw cache survives;
      the tool does not. *Impact:* whether Table 1's provenance is reproducible from the repository.
- [ ] **RQ3's answer space is 5 values plus "censored".** With 6 blocks of 30 days, `b*` resolves
      only to 30-day granularity and only out to 180 days. Is that an acceptable answer to "optimal
      retraining cadence", or does the test span need to lengthen? *Impact:* RQ3's framing, and
      whether the finding is a cadence or a lower bound on one.
- [ ] **Which manuscript language is authoritative** — the Indonesian title suggests an Indonesian
      venue, the reference library assumes IEEE style. *Impact:* the paper-production section.
- [ ] **Does the K=12 rung survive the measured PR?** If measured effective rank at K=8 lands far
      below the reasoned ~6.5, the ladder is re-cut before any training. The document must say what
      "far below" means numerically, before the measurement, or the gate is not a gate.
- [ ] **Is the specification's `no embargo` decision correct?** It is defensible — the only
      contamination path at the boundary is label overlap, which the H-purge closes — but the
      argument is currently unwritten and therefore untested.
- [ ] **User instruction truncated.** The originating request ended mid-sentence at
      *"…hal teknis tersebut masih masa"*. Any requirement in the cut portion is unrepresented here.
- [ ] **Does the ladder gain an optional F6 rung at K≈16?** All 12 variates are already engineered
      transforms; what is excluded is technical indicators, multi-bar rolling statistics, and
      calendar dummies. The exclusion rests on the **linear-span argument**: any linear function of
      the L=96 lookback already lies inside the span of `Linear(96 → d_model)`, so it raises nominal
      K while adding ≈0 effective rank — the exact phenomenon RQ1 measures. Candidates that are
      genuinely *outside* that span (trailing realized variance, signed-flow autocorrelation,
      VPIN-style toxicity) could form an explicit **fifth rung** that strengthens RQ1 as a second
      high-nominal-K / low-K_eff control. Folded into the existing four rungs instead, it destroys
      the instrument. *Impact:* §5 of the new root document; the main grid's run count.

---

## Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| The new document repeats the old one's failure mode and grows into an unread monolith | **high** | high | Root size ceiling as an explicit success metric; deep material pushed into scoped files. The 74 KB predecessor is the proof this happens by default |
| A defect in the source specification is copied forward unexamined | medium | **fatal** | The divergence register is a milestone, not a nicety. 21 defects were found in one read; more likely exist |
| Documentation drifts from the pipeline once code exists | high | high | Traceability contract makes the drift detectable: a number that cannot be regenerated from a named artifact is a documented failure, not a silent one |
| Kaggle quota exhausted mid-grid, partial results unrecoverable | medium | high | Continuation protocol is milestone 4. Runs must be individually idempotent and individually resumable, so a session cut costs one run, not one grid |
| The doc set fixes contradictions the author disagrees with | medium | medium | Every resolution is a register row with its evidence, reversible by editing one row rather than by re-reading two specifications |
| Corrections change the study's results relative to the source design | medium | medium | Items 2, 4, 5, and 6 can move the headline numbers. Each must be disclosed in the manuscript as a deviation from the pre-registered design, with its reason |
| Citations carried forward unverified into the manuscript | **high** | high | Both source files self-declare unverified entries. Hard rule: no citation without a verified DOI and the source read; unverified entries stay marked until cleared |
| The 4.4% window-loss estimate is wrong because the reconciliation is never run | low | medium | The specification already requires rejected-window counts reconciled against `119 × gap_blocks` per origin; the doc must make it an assertion, not a suggestion |

---

*Status: DRAFT — requirements only. Implementation planning pending via `/plan`.*
