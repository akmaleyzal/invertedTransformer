# Divergence Register

Every departure of `CLAUDE.md` from the two source specifications
(`research_specification_itransformer_btc.md`, `reference_library_itransformer_btc.md`), with the
evidence that settles it and the manuscript section that must disclose it.

**Purpose.** This file is what makes *"dipertanggungjawabkan"* enforceable rather than aspirational.
If a reviewer, examiner, or replicator asks "why did you deviate from your own specification?", the
answer is a row here — not improvisation.

**Severity.** **F** fatal to validity if unresolved · **C** internal contradiction · **U**
underspecified · **I** improvement over the source.

**Rule.** A contradiction discovered later gets the next free ID and a full row. Absorbing one
silently is the exact failure this file exists to prevent.

**Status legend.** `resolved` — decision made and encoded in `CLAUDE.md`. `open` — decision made,
but an external action is still required before the pipeline can run.

---

## D01 — The K ladder does not add up

| Field | Content |
|---|---|
| **Severity** | **F** |
| **Source** | `research_specification_itransformer_btc.md` §2.2, "The ladder" table |
| **Defect** | K=8 is defined as K=4 plus "remainder of F3, F4, F5". Remainder of F3 = {`log_trade_count`, `log_mean_trade_size`} = 2; F4 = 2; F5 = 1. On a base of 4 that gives **9, not 8**. Separately, `log_mean_trade_size` is assigned to the K=8 rung *and* again to K=12 ("+ F2, `log_mean_trade_size`"), so one variate is counted twice |
| **Resolution** | Re-cut so nominal counts match membership. **K=8 = K=4 + {`log_trade_count`, `taker_buy_ratio`, `signed_flow`, `vwap_location`}**; **K=12 = K=8 + {Parkinson, Garman–Klass, Rogers–Satchell, `log_mean_trade_size`}**. `CLAUDE.md` §5.2 |
| **Evidence** | Family cardinalities F1=3, F2=3, F3=3, F4=2, F5=1 sum to 12, matching the top rung. Given K=1={`r`} and K=4=F1∪{`log_quote_volume`}, the two remaining rungs must each add exactly 4, and only one assignment of the eight remaining variates satisfies both counts *and* keeps F2 intact as the redundancy control. The cut is therefore forced, not chosen |
| **Paper disclosure** | §3.2. State the ladder as corrected; no disclosure of the source error is required, since the source is unpublished |
| **Status** | resolved |

## D02 — The K_eff gate reads test-period data

| Field | Content |
|---|---|
| **Severity** | **F** |
| **Source** | `research_specification_itransformer_btc.md` §2.4 and §8 Stage 3b |
| **Defect** | Stage 3b computes the participation ratio over the full 2018–2026 sample and uses it as a **gate that re-cuts the K ladder**. Every origin's test block lies inside that sample, so a design decision is informed by data the design is later evaluated on. The specification also states the trigger only qualitatively — "if K_eff at K=8 is far below ~6.5" — so the gate is unfalsifiable as written |
| **Resolution** | The **gating** PR is computed on the **pre-first-origin span only, 2018-01-01 → 2020-01-01** — data that is training data at every origin. The full-sample rolling PR is retained but labelled *descriptive only* and may inform no design decision. The trigger is pre-registered numerically: **re-cut the ladder if measured PR at K=8 < 5.0**. `CLAUDE.md` §5.4, §8.5 |
| **Evidence** | The first origin is 2020-01 (§8.1), so 2018-01 → 2020-01 is inside every origin's training window and outside every test block. "Far below" admits any post-hoc reading; 5.0 does not |
| **Paper disclosure** | §3.3 — state that the gate used pre-origin data only, and give the pre-registered trigger. §4.1b may report the full-sample rolling PR as descriptive |
| **Status** | resolved |

## D03 — The `use_norm` verification test cannot pass as written

| Field | Content |
|---|---|
| **Severity** | **F** |
| **Source** | `research_specification_itransformer_btc.md` §4.4, "Verification test" |
| **Defect** | "Train one model on StandardScaler output and one on the same input multiplied by 100. Under `use_norm=True` both losses must be identical to numerical precision." The target is a channel of the same scaled array, so multiplying the input by `c` multiplies the target by `c`; instance denormalisation multiplies predictions by `c`; MSE therefore scales by `c²`. At c=100 the loss is 10⁴× larger. A team running this test would see it "fail" and either disable `use_norm` or chase a non-existent bug |
| **Resolution** | Corrected invariant: **`MSE(c·x)/c² == MSE(x)`**, equivalently RelMSE unchanged, since RelMSE is a ratio in which the scale factor cancels. `CLAUDE.md` §6.3 |
| **Evidence** | The specification's own §4.4 algebra shows `(z − m)/s = (x − mean_t(x_W))/std_t(x_W)`: the *network input* is scale-invariant. It does not follow that the *loss* is, because the loss is computed in scaler space against a target that scales with the array |
| **Paper disclosure** | §3.4 — report the invariance test and the invariant actually used |
| **Status** | resolved |

## D04 — Instance normalisation strips volatility level, confounding RQ1

| Field | Content |
|---|---|
| **Severity** | **F** |
| **Source** | interaction between §4.4 (`use_norm=True`) and §2.2 (family F2), unaddressed in either |
| **Defect** | `use_norm=True` divides each input window by its own per-variate standard deviation over L=96. The F2 estimators (Parkinson, Garman–Klass, Rogers–Satchell) are *volatility level* measurements; after per-window standardisation only their *shape* survives. The 8→12 rung can therefore flatten because its information was normalised away — a mechanism entirely distinct from the redundancy the rung was designed to demonstrate. RQ1's independent variable is confounded by the architecture's own normalisation |
| **Resolution** | Measure the participation ratio on **window-normalised features in addition to raw**, report both, and disclose the confound in Limitations regardless of which way the rung goes. `CLAUDE.md` §5.4 |
| **Evidence** | Structural: RevIN-style instance normalisation subtracts the per-window mean and divides by the per-window standard deviation of each channel. Any channel whose information content is its *scale* is thereby reduced to its *pattern*. This applies to F2 and to `log_quote_volume`, not to `r` or `taker_buy_ratio`, which are already scale-free |
| **Paper disclosure** | §3.3 (both PR measurements), §4.1b (both reported), §4.8 (the confound stated plainly) |
| **Status** | resolved |

## D05 — `D(b)` conflates model decay with market difficulty

| Field | Content |
|---|---|
| **Severity** | **F** |
| **Source** | `research_specification_itransformer_btc.md` §7.1, RQ3 dependent variable |
| **Defect** | `D(b) = [MSE(b) − MSE(1)] / MSE(1)` compares a model's error in block *b* against its error in block 1 — **different calendar months, with different volatility**. A rising `D(b)` is consistent with model decay *and* with the market simply becoming harder, and the two are not separable. RQ3's answer ("optimal retraining cadence") would then be partly a statement about market conditions |
| **Resolution** | Define `D(b) = [RelMSE(b) − RelMSE(1)] / RelMSE(1)`, where RelMSE is measured against the Naive-RW baseline **within the same block**. `CLAUDE.md` §9.1 |
| **Evidence** | The specification already defines RelMSE and states its purpose as "controls for period difficulty" (§7.1) — it simply does not apply it to `D(b)`. `A(b)` needs no equivalent correction: both models are evaluated on the same block, so difficulty cancels in the ratio |
| **Paper disclosure** | §3.7 (definition), §4.4 (results). State that `D(b)` is naive-normalised and why |
| **Status** | resolved |

## D06 — The decay regression is underspecified

| Field | Content |
|---|---|
| **Severity** | **F** |
| **Source** | `research_specification_itransformer_btc.md` §7.2, "Decay regression" |
| **Defect** | The specification writes `A(b) = β₀ + β₁·b + ε` and then correctly requires clustering by origin, wild cluster bootstrap, and seed averaging. It omits **origin fixed effects**. Without `αᵢ`, between-origin variation in baseline difficulty enters the residual, and β₁ is identified partly from cross-origin differences rather than purely from time-since-training. The specification also defaults to Rademacher weights implicitly by naming the standard wild bootstrap, and does not address dependence *across* origins |
| **Resolution** | Specify **`A(i,b) = αᵢ + β₁·b + ε`** with origin fixed effects, clustered by origin, wild cluster bootstrap with **Webb 6-point weights**, seeds averaged within each (origin, block) cell first. Disclose the cross-origin adjacency. `CLAUDE.md` §9.2 |
| **Evidence** | Three independent points. (1) Fixed effects: block index *b* is nested within origin *i*; without `αᵢ`, origin-level difficulty contaminates β₁. (2) Weights: at G=13 clusters, Rademacher weights admit only 2¹³ = 8,192 distinct bootstrap draws, which bounds attainable p-value resolution; Webb's 6-point distribution is the standard remedy below ~12–15 clusters. (3) Adjacency: origin *i*'s test span is `[o_i, o_i + 180d)` and origin *i+1* begins at `o_i + 6 months`, so block 6 of one origin is calendar-adjacent to block 1 of the next; volatility clustering makes those residuals correlated across the cluster boundary. **Superseded in part**: `D26` re-cut the spacing to 5 months (G = 15), so the weights argument in (2) no longer binds and the adjacency in (3) becomes an overlap — `D28` supplies the stronger statement |
| **Paper disclosure** | §3.7 (full specification, cluster count stated explicitly), §4.3 (results), §4.8 (cross-origin dependence as a stated limitation) |
| **Status** | resolved |

## D07 — The origin count is credited to the wrong cause

| Field | Content |
|---|---|
| **Severity** | **C** |
| **Source** | `research_specification_itransformer_btc.md` §6.1, "Origin derivation" |
| **Defect** | "Spot history starting 2017-08 is what makes thirteen possible; the futures-based design reached only ten." The same section declares the analysis window as 2018-01-01 → 2026-08-01. The two statements cannot both be load-bearing, and a reader following the 2017-08 claim would fetch five extra months of data for no gain |
| **Resolution** | Delete the claim. Thirteen follows from the declared **2018-01-01** start. `CLAUDE.md` §8.1 |
| **Evidence** | Arithmetic: earliest origin = 2018-01-01 + 24 months = 2020-01-01. Latest origin satisfying `o + 180d ≤ 2026-08-01` is 2026-02-02, so with 6-month spacing from 2020-01-01 the origins are 2020-01, 2020-07, 2021-01, 2021-07, 2022-01, 2022-07, 2023-01, 2023-07, 2024-01, 2024-07, 2025-01, 2025-07, 2026-01 — **thirteen**, the last testing through 2026-06-30. Data in hand (`actual_first_bar_utc`) begins 2018-01-01T00:00Z |
| **Paper disclosure** | §3.6 — derive 13 from the stated window. No mention of 2017-08 |
| **Status** | resolved |

## D08 — Horizon sweep K count stated two ways

| Field | Content |
|---|---|
| **Severity** | **C** |
| **Source** | §6.4 ("4 origins × 4 K × 4 H × 3 seeds = 192") vs §8 Stage 8 ("4 origins × 3 K × 4 H × 3 seeds") |
| **Resolution** | **4 K values.** `CLAUDE.md` §10.2 |
| **Evidence** | 4 × 4 × 4 × 3 = 192 matches the stated total; 4 × 3 × 4 × 3 = 144 does not. Stage 8 is the typo |
| **Paper disclosure** | §3.6 / Table 7 — state the sweep dimensions once, consistently |
| **Status** | resolved |

## D09 — Reference library says twelve origins

| Field | Content |
|---|---|
| **Severity** | **C** |
| **Source** | `reference_library_itransformer_btc.md` §D, closing note: "Direct justification for your 12 origins" |
| **Resolution** | Thirteen. `CLAUDE.md` §8.1 |
| **Evidence** | Stale text from the superseded futures-based design, which the specification itself says "reached only ten". Superseded twice over; see D07 for the derivation |
| **Paper disclosure** | none — internal inconsistency in an unpublished working file |
| **Status** | resolved |

## D10 — The artifact on disk violates the specification

| Field | Content |
|---|---|
| **Severity** | **F** |
| **Source** | repository state vs `research_specification_itransformer_btc.md` §2.3 |
| **Defect** | `data/raw/BTCUSDT_1h_report.json` records `"fill_policy": "ffill"`, `"rows_written": 75216`, `"synthetic_bars": 122`. The parquet on disk has been **reindexed to a full hourly grid and forward-filled** — the two operations §2.3 forbids by name. Consuming it as-is fabricates 122 zero-return bars, forces `H = L = O = C` on each, and drives all three F2 estimators to zero immediately before whatever event caused the outage |
| **Resolution** | The pipeline must consume **unfilled** bars: regenerate with no fill, or drop every row flagged synthetic *before any feature is computed*. `BTCUSDT_1h_gaps.csv` is retained as a diagnostic and as the source of segment boundaries. `CLAUDE.md` §4.4 |
| **Evidence** | The report's own `fill_policy` and `synthetic_bars` fields; `bars_actual` 75,094 vs `rows_written` 75,216, a difference of exactly the 122 missing bars |
| **Paper disclosure** | §3.1 — state that no filling is applied and that the segmentation rule replaces it. The intermediate filled artifact is not part of the pipeline and needs no mention |
| **Status** | **open** — regeneration or filtering must happen before Stage 3 |

## D11 — The Stage 1 entry point is missing

| Field | Content |
|---|---|
| **Severity** | **C** |
| **Source** | `research_specification_itransformer_btc.md` §8 Stage 1, §12 week 1 |
| **Defect** | `binance_spot_klines.py` is named as the ingest tool in the execution pipeline and in the timeline. It is not present in the working tree, and it was never committed, so it is not recoverable from history either |
| **Resolution** | Restore or re-specify Stage 1 before Table 1 can claim reproducible provenance. `data/raw/BTCUSDT_1h_raw.jsonl` preserves the raw API responses, so the parquet is re-derivable offline without re-hitting the API. `CLAUDE.md` §4.4 |
| **Evidence** | `git ls-files` and a filesystem search return no `.py` at the repository root; the JSONL cache is present |
| **Paper disclosure** | §3.1 — provenance must cite a tool that exists and a procedure that can be re-run |
| **Status** | **open** — blocks reproducibility, not correctness |

## D12 — `signed_flow` is a deterministic product of two other K=8 members

| Field | Content |
|---|---|
| **Severity** | **U** |
| **Source** | `research_specification_itransformer_btc.md` §2.2, family F4 |
| **Defect** | `signed_flow = (2·taker_buy_ratio − 1) · log_volume`. Both factors are themselves variates at the K=8 rung (`taker_buy_ratio` in F4, `log_quote_volume` in F3). The rung the specification calls "of maximum effective rank" therefore contains a variate that is a pointwise product of two of its own members. The product is nonlinear, so it is not perfectly redundant, but the claim is weaker than stated |
| **Resolution** | Keep the variate — the interaction it encodes (signed flow scaled by activity) is the standard order-flow-imbalance construction and is theoretically motivated. **Disclose the dependence**, and let the measured participation ratio at K=8 settle how much rank it actually contributes. `CLAUDE.md` §5.1 |
| **Evidence** | Definitional. Note the practical consequence: if measured PR at K=8 lands below the pre-registered 5.0 trigger, D02's ladder re-cut applies, and this dependence is the most likely reason |
| **Paper disclosure** | §3.2 — define `signed_flow` and note that it is an interaction of two other variates in the same rung |
| **Status** | resolved |

## D13 — F2 estimators: per-bar or trailing-averaged

| Field | Content |
|---|---|
| **Severity** | **U** |
| **Source** | `research_specification_itransformer_btc.md` §2.2 (family F2) vs §2.2's own mandate that "every rolling indicator must use a backward-looking window" |
| **Defect** | Parkinson, Garman–Klass, and Rogers–Satchell are single-bar estimators, but the specification's rolling-window mandate implies some of them roll, and it never says which. At 1-hour granularity a single-bar range estimator is extremely noisy, which is the usual argument for averaging over a trailing window — leaving a genuine and unstated choice |
| **Resolution** | **Per-bar, with no trailing average.** `CLAUDE.md` §5.1, §5.3 |
| **Evidence** | Follows from the linear-span argument that governs the whole feature policy: iTransformer embeds each variate's entire L=96 lookback through `Linear(96 → d_model)`, whose span contains every linear function of that lookback — including any trailing mean. Feeding a pre-smoothed estimator is therefore *strictly less informative* than feeding the per-bar series: the model can compute the average itself, but cannot recover what smoothing destroyed. The noise objection is answered by the same fact. **Consequence:** with F2 per-bar, no variate in the study uses a rolling window at all, which makes the `center=True` leak class structurally unrepresentable and is what licenses the no-embargo argument in D15 |
| **Paper disclosure** | §3.2 — state that all estimators are per-bar and give the linear-span reason; it is a defensible methodological position, not an omission |
| **Status** | resolved |

## D14 — Undefined denominators and a division by zero

| Field | Content |
|---|---|
| **Severity** | **U** |
| **Source** | `research_specification_itransformer_btc.md` §2.1 (column table), §2.2 (families F4, F5) |
| **Defect** | Two gaps. (1) `taker_buy_ratio` is not defined as base- or quote-denominated; the kline provides both `taker_buy_base_volume` and `taker_buy_quote_volume`, and the two differ whenever price moves within the bar. (2) `vwap_location = (VWAP − C)/(H − L)` divides by zero on any bar where `H == L`, and `VWAP = quote_volume/volume` and `log_quote_volume` are both undefined when `volume = 0`. The data contains **3 zero-volume and 3 zero-trade bars**, so this is a live case, not a hypothetical |
| **Resolution** | (1) **Base-denominated**: `taker_buy_ratio = taker_buy_base_volume / volume`, the canonical buyer-initiated volume share; the quote-denominated variant is a robustness check. (2) **A zero-volume or `H == L` bar is a segment break**, treated exactly like exchange downtime: excluded, with the series splitting there. `CLAUDE.md` §4.3, §5.1 |
| **Evidence** | `data/raw/BTCUSDT_1h_report.json` reports `zero_volume_bars: 3`, `zero_trade_bars: 3`. The segment-break treatment is the only resolution consistent with the no-imputation law: a bar in which nothing traded carries no trade information, which is the same condition as downtime, so it takes the same policy. Substituting any value would be imputation under another name. Cost: 3 extra breaks, raising estimated window loss from ~4.4% to ~4.9% — see the cost accounting in `CLAUDE.md` §4.3. Side effect: it makes `log(volume)`, `VWAP`, and all three F2 logarithms **total functions**, since `H == L` is the only case driving an F2 estimator to zero |
| **Paper disclosure** | §3.1 (the segment rule covers both downtime and zero-activity bars, with counts in Table 1), §3.2 (`taker_buy_ratio` denominator) |
| **Status** | resolved |

## D15 — Embargo declared "justified" with no justification

| Field | Content |
|---|---|
| **Severity** | **U** |
| **Source** | `research_specification_itransformer_btc.md` §6.1, evaluation-protocol table: "Embargo — not applied (justified)" |
| **Defect** | No justification appears anywhere in the document. Omitting an embargo is defensible in this design, but silence on a López de Prado protocol element is exactly what a reviewer familiar with that literature will flag, and the author would have to improvise the argument under questioning |
| **Resolution** | Write the argument. An embargo guards against test-period information reaching the training set. Two paths exist, both closed: **label overlap**, closed by the H-step purge; and **feature lookback**, which cannot run test→train because no feature uses a rolling window (D13), so no test-period bar can influence any training-set feature value. The reverse direction — a test-time feature looking back into the training period — is *past* information legitimately available to a real forecaster and is not leakage. `CLAUDE.md` §8.3 |
| **Evidence** | Depends on D13. The argument is only airtight while every variate is a per-bar function; introducing any rolling feature reopens the question, and `CLAUDE.md` §8.3 says so explicitly |
| **Paper disclosure** | §3.6 — state the purge, state that no embargo is applied, and give this two-path argument in two sentences |
| **Status** | resolved |

## D16 — Unverified and mis-dated citations

| Field | Content |
|---|---|
| **Severity** | **C** |
| **Source** | `reference_library_itransformer_btc.md` §§E, H, P, Q and its own header warning |
| **Defect** | The library states of itself that "volume numbers, page ranges, and DOIs below were assembled from search results, not from access to published versions", and marks several sections "assembled from memory in this revision". Two concrete errors are visible without leaving the file: **Yang et al.**, arXiv:2509.23494, is dated **2026** although the `2509` identifier places it in September **2025**; and **Symmetry 18(1), 32** is dated **2025** although volume 17 is the 2025 volume. Several DOIs (e.g. `10.1016/j.irfa.2026.100029`) follow patterns that warrant checking before use |
| **Resolution** | Standing rule: **no citation enters the manuscript without a verified DOI and the source read.** Every entry stays marked unverified until cleared. `CLAUDE.md` §13.3 |
| **Evidence** | arXiv identifiers encode YYMM; `2509` is 2025-09. Symmetry publishes one volume per year, with volume 17 in 2025. The library's own header and per-section "verification status" notes |
| **Paper disclosure** | none directly, but the rule governs every citation in every section. An examiner asking what a cited paper says must get an answer |
| **Status** | **open** — verification is ongoing work, not a one-time fix |

## D17 — No control for "is a transformer needed at all?"

| Field | Content |
|---|---|
| **Severity** | **I** |
| **Source** | `research_specification_itransformer_btc.md` §5, baseline table |
| **Gap** | The baseline set controls for architecture in one direction only. K=1 iTransformer isolates *cross-variate attention* while holding everything else fixed — an excellent control for "does attention help?". Nothing in the set answers "does a transformer help, given the same information?" DLinear and PatchTST are channel-independent, so neither consumes the K-variate feature set the way iTransformer does |
| **Resolution** | Add a **multivariate ridge regression on the same K features**, α selected on the validation sub-block. `CLAUDE.md` §7 |
| **Evidence** | Cost is negligible — 15 origins × 4 rungs = 60 closed-form fits, seconds each. Benefit is a clean two-way decomposition: ridge vs Naive-RW answers *does the information help*; iTransformer vs ridge answers *does the architecture help*; K=8 vs K=1 answers *does cross-variate attention help*. Without ridge, the first two questions are entangled |
| **Paper disclosure** | §3.5 (baseline), Table 4 (results), §4.2 (the decomposition is part of RQ1's answer) |
| **Status** | resolved |

## D18 — Three seeds is too few for the headline claim

| Field | Content |
|---|---|
| **Severity** | **I** |
| **Source** | `research_specification_itransformer_btc.md` §4.2 (seeds 42, 43, 44), §7.2 ("all metrics as mean ± std across 3 seeds") |
| **Gap** | A standard deviation estimated from n=3 is itself extremely noisy, and the specification's own risk table anticipates the problem ("seed variance exceeds the effect → increase to 5 seeds") — as a contingency rather than a design choice. RQ2, the core claim, rests entirely on the K=1 vs K=8 contrast |
| **Resolution** | **5 seeds (42–46) for K ∈ {1, 8}**; 3 seeds for K ∈ {4, 12}. `CLAUDE.md` §6.2 |
| **Evidence** | Marginal cost is 15 origins × 2 rungs × 2 extra seeds = 60 runs ≈ 1.5 GPU-hours under the regime in D19 — a rounding error against a 30 h weekly quota. Reacting to seed noise *after* seeing results would be a design change made in response to the outcome |
| **Paper disclosure** | §3.4 (seed counts, stated per rung), Table 4 (n reported alongside mean ± std) |
| **Status** | resolved |

## D19 — "Hours, not days" holds only in one implementation regime

| Field | Content |
|---|---|
| **Severity** | **I** |
| **Source** | `research_specification_itransformer_btc.md` §6.4: "a single iTransformer run completes in minutes on free-tier GPU, so the added cost is hours, not days" |
| **Gap** | True, but only under an implementation regime the specification never states. The claim is not wrong; it is unconditioned, and the unstated condition is the difference between fitting the weekly quota and blowing through it |
| **Resolution** | Document the regime and its counterfactual. The per-origin training tensor is `17,400 × 96 × 12 × 4 B ≈ 80 MB` and **must be resident in GPU memory**, with batching by index-slice and **no `Dataset`/`DataLoader`**. Use the two T4s as **two independent run workers**, one pinned per device — not `nn.DataParallel`. `CLAUDE.md` §10.3 |
| **Evidence** | The model is ~280k parameters at ~420–475 steps/epoch (D25); compute per step is negligible, so wall time is dominated by data movement and Python overhead, which is exactly what a per-item `DataLoader` maximises. Estimated ~60–100 s per run GPU-resident ⇒ ≈ 10–20 wall-hours for ~837 runs across 2 GPUs; roughly an order of magnitude worse with a naive 4-worker loader, which **exceeds 30 h/week**. On `DataParallel`: at batch 32 the per-step scatter/gather cost exceeds the saving from splitting 16 samples per device; parallelism belongs at the run level, because the grid is many small runs rather than one large one |
| **Paper disclosure** | §3.4 or an implementation note — reporting hardware and wall time is standard, and the regime explains the number |
| **Status** | resolved. Replace the estimate with the first real measurement |

## D20 — RelMSE near 1.00 is hard to read

| Field | Content |
|---|---|
| **Severity** | **I** |
| **Source** | `research_specification_itransformer_btc.md` §7.1 |
| **Gap** | Against `ŷ = 0`, the naive MSE on log-returns *is* the realised variance, so RelMSE equals `1 − R²_oos` and will sit at roughly 0.99–1.00 throughout. A table of numbers differing in the third decimal place obscures the size of the effect |
| **Resolution** | Report **`R²_oos = 1 − RelMSE`** alongside RelMSE. `CLAUDE.md` §9.1 |
| **Evidence** | Identity, not approximation: `MSE_naive = E[r²] = Var(r)` when `ŷ = 0` and returns are approximately mean-zero. `R²_oos = 0.004` reads immediately; `RelMSE = 0.996` does not |
| **Paper disclosure** | §3.7, Table 4. Note in the text that anything above roughly 0.05 at this horizon should raise a leakage suspicion until re-verified |
| **Status** | resolved |

## D21 — Trading rule and directional accuracy undefined at H=24

| Field | Content |
|---|---|
| **Severity** | **U** |
| **Source** | `research_specification_itransformer_btc.md` §7.1 (DA), §8 Stage 11 ("simple long/short rule from directional predictions") |
| **Defect** | At H=24 the model emits 24 predictions per window and windows are spaced 1 hour apart, so "the directional prediction" is ambiguous across two axes: which forecast step drives the position, and whether overlapping windows each generate a trade. The choice changes turnover by more than an order of magnitude and therefore dominates every cost-adjusted number in Table 8. Left unspecified, the economic evaluation is unreproducible |
| **Resolution** | **Position from the sign of the cumulative H-step forecast, held H hours, non-overlapping** — at H=24 that is at most one trade per day. **DA reported at step h=1, at step h=24, and on the cumulative 24-hour return**, tested with Pesaran–Timmermann. `CLAUDE.md` §9.1, §13.5 |
| **Evidence** | The cumulative forecast is the quantity the position actually earns over the holding period, so it is the coherent signal for a position held that long. Non-overlapping execution keeps turnover interpretable and prevents the same forecast being traded 24 times. Also required: the Deflated Sharpe Ratio takes the **per-period** Sharpe, never the annualised one — feeding it an annualised value inflates the statistic by `sqrt(periods_per_year)` and silently disables the multiple-testing correction it exists to apply |
| **Paper disclosure** | §3.7 (DA definitions), §4.7 (trading rule stated before any return figure), Table 8 (turnover reported so the rule is checkable) |
| **Status** | resolved |

## D22 — No rung tests genuinely nonlinear features

| Field | Content |
|---|---|
| **Severity** | **I** |
| **Source** | new — arises from the feature-engineering policy in `CLAUDE.md` §5.3, absent from both sources |
| **Gap** | The ladder's four rungs are all built from per-bar functions, so every one of them lies inside the span of the inverted embedding's linear projection over the lookback. The design therefore never tests what happens when *genuinely new* information — nonlinear in the lookback — is added at high nominal K. A reviewer can reasonably ask whether the flat 8→12 rung reflects redundancy or simply the absence of anything new |
| **Resolution** | Pre-register an **optional fifth rung at K=16**, adding features that are provably outside the linear span: trailing realised variance (quadratic in returns), signed-flow autocorrelation, VPIN-style order-flow toxicity. Run **only** if the Stage 5 pilot passes and quota allows. It must be an explicit fifth rung; folded into the existing four it destroys the instrument. `CLAUDE.md` §5.3 |
| **Evidence** | The linear-span argument cuts both ways: it is why MA/EMA/momentum indicators are excluded, and it is precisely why a quadratic functional such as realised variance is *not* excluded by the same reasoning. Adding the rung supplies a second high-nominal-K / low-K_eff control and strengthens RQ1's central contrast rather than diluting it. Cost if run: 15 origins × 1 rung × 5 seeds = 75 additional runs |
| **Paper disclosure** | §3.2 — state the rung as pre-registered and optional, and state whether it was run. If it was not run, say why (pilot outcome or quota), because an unrun pre-registered arm must be accounted for |
| **Status** | resolved — pre-registered as optional; execution decision deferred to Stage 5 |

---

# Second pass — D23 onwards

**Provenance.** D01–D22 record defects in the two source specifications. D23 onwards record defects
found in **this document set itself**, by a five-lens adversarial audit run on 2026-08-05
(`wf_becd8b67-d95`). Three lenses completed — defensibility, leakage, statistics; two (consistency,
Kaggle) died on a session limit and have **not** been run. The audit's independent verifier stage
also died, so **every entry below was adjudicated by direct re-derivation against the document text
or the artifact on disk**, not accepted on an agent's word. Where the audit's proposed fix was wrong,
the entry says so and records the correct one (see D33). The two unrun lenses are logged as an open
item: this pass is **incomplete by construction**, and saying otherwise would be the exact failure
this register exists to prevent.

---

## D23 — The pre-registered τ is arithmetically unreachable; RQ3 is a guaranteed null

| Field | Content |
|---|---|
| **Severity** | **F** |
| **Source** | new — internal contradiction between `CLAUDE.md` §9.1 (`D(b)` definition), §3 (τ = 5%) and D20 |
| **Defect** | `D(b) = [RelMSE(b) − RelMSE(1)] / RelMSE(1)`, with τ ∈ {2.5%, 5%, 10%}. D20 establishes that against `ŷ = 0` the RelMSE "will sit at roughly 0.99–1.00 throughout" and `R²_oos ≈ 0.004`. Since `RelMSE = 1 − R²_oos`, `RelMSE(1) = 0.996`, so even **total** destruction of the model's edge (`R²_oos(b) = 0`) gives `D(b) = 1/0.996 − 1 = 0.402%`. τ = 2.5% requires `R²_oos(b) = −0.021`; τ = 5% requires `−0.046`. All three thresholds demand the model become 2–10% *worse than forecasting zero* |
| **Resolution** | Re-scale the metric to the quantity that moves. `D(i,b) = [R²_oos(i,1) − R²_oos(i,b)] / R²_oos(i,1)` — proportional loss of skill, so τ = 5% means "5% of the edge is gone" and τ = 100% means "the edge is gone". Guard the denominator: origins with `R²_oos(i,1) ≤ 0` contribute no `b*` and are excluded, stated as such. `CLAUDE.md` §9.1, §3 |
| **Evidence** | Pure arithmetic on D20's own stated magnitudes, requiring no data. `1/0.996 − 1 = 0.00402 < 0.025`. The τ values were carried over from the source specification without checking commensurability with the metric D05 replaced them against |
| **Paper disclosure** | §3.7 — print the derivation next to the threshold so the reader sees τ is commensurate with the metric. §4.4 — report `b*` on the skill-loss scale |
| **Status** | resolved — metric re-scaled before any run, so the pre-registration is not violated |

---

## D24 — No purge at the train/validation boundary; early stopping selects on contaminated data

| Field | Content |
|---|---|
| **Severity** | **F** |
| **Source** | new — `CLAUDE.md` §8.1 purge row, §8.2, §11 item 11 |
| **Defect** | The purge is specified **only** at the train–test boundary. The 24-month window splits 21 months training / final 3 months validation, and nothing bounds a training window whose H-step target extends past the 21-month mark into the validation sub-block. §8.2's derivation applies the `−H` term at `T_end` (the origin), not at the train/validation split. §11's phrasing — "every train–test boundary" — silently excludes the boundary that actually governs model selection |
| **Resolution** | Enumerate training windows to `val_start − L − H`, so the last training target ends at `val_start`. Purge at **both** boundaries. Rewrite §11 item 11 as "H-step purge active at every boundary between disjoint splits: train→validation and train→test", and add the assertion `max(target_index over training windows) < val_start`. `CLAUDE.md` §8.1, §8.2, §11 |
| **Evidence** | Contamination is small (~24 windows) but it is the defect class §11 declares fatal, and it lands on the split that decides early stopping (`patience 5 on validation MSE`, §6.2) and ridge α (§7). Validation-window *inputs* reaching back into the training period remain legitimate — that is past information, §8.3 — so the asymmetry is deliberate and must be stated as such |
| **Paper disclosure** | §3.6 — state the purge at both boundaries and the deliberate input/target asymmetry |
| **Status** | resolved |

---

## D25 — ~17,400 windows is the 24-month count; the training set is 13,558–15,217

| Field | Content |
|---|---|
| **Severity** | **F** |
| **Source** | new — `CLAUDE.md` §5.3 reason 3, §6.2, §8.2, §10.3 |
| **Defect** | 24 months = 17,520 h; minus `L + H` = 17,400 — exactly the figure quoted in four places, and `17,400 × 96 × 12 × 4 B = 80.18 MB` matches §10.3's "≈ 80 MB" to the digit, as does `545 × 32 = 17,440` steps/epoch. But §8.1 reserves the final 3 months for validation, so the training set is 21 months ≈ 15,336 h ⇒ **~15,217 windows** before gap losses, and as few as **13,558** once measured gap losses are applied (see docs/ORIGIN_WINDOW_BUDGET.md). Every figure downstream inherits a 14% error — or else the intended implementation trains on all 24 months and validates in-sample, which is D24's leak |
| **Resolution** | State that training windows are enumerated over the 21-month sub-block only. Correct 17,400 → the **measured per-origin range 13,558–15,217** in §5.3, §6.2, §8.2 and §10.3 (ceiling 15,217 where a span is gap-free); 80 MB → **≤70 MB**, sized at the ceiling and sliced per origin; 545 → **~420–475** steps/epoch. Add a split-generation assertion that the count matches the 21-month arithmetic minus the logged gap losses, checked against `docs/ORIGIN_WINDOW_BUDGET.md` |
| **Evidence** | The arithmetic is exact and the coincidence with the tensor size and steps/epoch is decisive: the numbers were derived from 24 months, not from the split |
| **Paper disclosure** | Table 3 sample counts, and §3.6 where the split is described |
| **Status** | resolved |

---

## D26 — Block index b is collinear with calendar month; β₁ does not identify model age

| Field | Content |
|---|---|
| **Severity** | **F** |
| **Source** | new — `CLAUDE.md` §8.1 origin grid against §9.2 decay regression |
| **Defect** | Origins sit on a fixed 1 Jan / 1 Jul grid with 6 × 30-day blocks, so block `b` maps to the same calendar window in **every** origin: `b=1` is always January or July, `b=6` always June or December. With origin fixed effects `αᵢ` absorbing the origin level, β₁ is identified purely from within-origin variation in `b` — which is exactly month-of-half-year variation, identically aligned across all 13 origins. Month dummies are a deterministic function of `b` and cannot be added post hoc. §9.2 discloses only cross-origin *adjacency*, a variance issue; this is a **bias** issue |
| **Resolution** | **Origins spaced 5 months, 15 of them**, 2020-01 … 2025-11. Plus a pre-registered falsification arm: for every origin train a **fresh** model at `o_i + 90 days` and evaluate it on the same calendar blocks 4–6 as the aged model; if the aged-minus-fresh gap is zero while β₁ < 0, β₁ is calendar, not age (15 extra runs). `CLAUDE.md` §8.1, §9.2, §10.2 |
| **Evidence** | Deterministic consequence of 6-month spacing against a 180-day test span. β₁ < 0 is otherwise observationally equivalent to "February/August are harder for microstructure features than January/July". **The spacing is settled by one expression**: with origins spaced *s* months and 30-day blocks, block `b` at origin *i* lands on calendar month `m₀ + s·i + (b−1) (mod 12)`, so for fixed `b` the months visited form a coset of ⟨*s*⟩ ≤ ℤ₁₂, of size **12/gcd(s,12)**. `s`=6 → 2 months; `s`=3 → 4; `s`=5 → **12, full decoupling**. Only *s* coprime to 12 works, and 5 maximises origins inside the span. Measured against `data/raw/BTCUSDT_1h_gaps.csv`, `s`=5 also beats `s`=3 on consecutive training overlap (79.2% vs 87.5%) and on worst test-block loss (33.9% vs 50.4%) |
| **Paper disclosure** | §3.6 (design), §3.7 (identification), §4.3 (the claim itself), §4.8 |
| **Status** | resolved by design change |
| **Correction, 2026-08-06** | This entry first prescribed **3-month interleaved** spacing, and `CLAUDE.md` was edited to match. On re-derivation that was wrong: `gcd(3,12) = 3` leaves four calendar phases, so 3-month spacing *reduces* the confound instead of removing it, while worsening training overlap to 87.5%. Changed to 5 months. Recorded rather than silently rewritten — the register's own rule (§14) applies to the register |

---

## D27 — The Stage 5 gate opens test blocks, contradicting §11's final item

| Field | Content |
|---|---|
| **Severity** | **F** |
| **Source** | new — `CLAUDE.md` §8.5 Stage 5 against §11 final item and §10.5 |
| **Defect** | §11: "The test blocks are opened once, after the design is frozen." §8.5 Stage 5: "1 origin × 4 K × 3 seeds … If K=8 does not significantly beat K=1, reposition the title". A significance judgement requires out-of-sample errors, so the gate is evaluated on the pilot origin's test blocks — and §10.5's idempotence plus §10.4's deterministic `run_id` mean the resume logic finds those runs complete and contributes them verbatim to Table 4, `A(i,b)`, and the β₁ regression. The pilot origin is unnamed, the test unnamed, α unstated. §13.5 treats this as a trial count; it is not — it is a selection event over the paper's conclusion, which the DSR does not correct |
| **Resolution** | Run the Stage 5 gate on the pilot origin's **validation sub-block**, the leak-free instrument for a go/no-go on architecture. If a test-block gate is genuinely required, designate one origin a burnt hold-out, exclude its runs from every table and from the regression, and state the reduced cluster count. Either way, disclose the pilot as a selection event distinct from the trial count. `CLAUDE.md` §8.5, §11, §13.5 |
| **Evidence** | The two sentences cannot both be obeyed. This is D02's defect — a design decision made on data the design is later evaluated on — recurring one stage later in the pipeline |
| **Paper disclosure** | §3.6 and §13.2 — state what the pilot decided and on which data |
| **Status** | resolved |

---

## D28 — Consecutive origins share 75% of their training data; the 13 clusters are not independent

| Field | Content |
|---|---|
| **Severity** | **F** |
| **Source** | new — `CLAUDE.md` §8.1 against §9.2 "Disclose the residual dependence" |
| **Defect** | A 24-month window advanced 6 months means origin *i* and *i+1* train on 18 of the same 24 months — **75% overlap**; *i* and *i+2* share 50%. Moreover, since the test span (180 days) equals the spacing, origin *i+1*'s training window **fully contains** origin *i*'s entire test period. §9.2 discloses only that "origin *i*'s block 6 is calendar-adjacent to origin *i+1*'s block 1" — the weaker mechanism. Wild cluster bootstrap assumes independence *between* clusters; two consecutive origins' models are fitted on 75% identical data, so the effective number of independent units is far below 13 and the bootstrapped p-value on β₁ is anticonservative by an unquantified amount |
| **Resolution** | State the overlap fractions and the test-inside-training containment explicitly, replacing the adjacency wording. Under `D26`'s 5-month grid the fractions are **79.2% / 58.3% / 37.5% / 16.7%** at strides 1–4, and windows are disjoint only at **stride 5**, so the training-disjoint subset is **3 origins** — {2020-01, 2022-02, 2024-03} or any of the four parallel triples. Re-estimate β₁ on all five triples and report their spread; at G = 3 it will very likely be inconclusive, and **that is the finding** — it bounds what the full-panel p-value can honestly claim. Consider a moving-block bootstrap over calendar time as the alternative to an i.i.d.-over-clusters bootstrap, and state which was chosen. `CLAUDE.md` §8.1, §9.2 |
| **Effective independence** | Bounded by `total span / training window ≈ 96 / 24 ≈ 4` **regardless of spacing** — packing origins closer raises nominal G without adding information. That is why `D26` chose 15 well-separated origins over 25 tightly-packed ones, and it is the number to quote when a referee asks for the effective cluster count |
| **Evidence** | The sophisticated correction (Webb weights for small G) addresses the smaller of the two problems while the larger one is described as calendar adjacency. A referee from the econometrics side will ask for the effective cluster count and "13" will not survive |
| **Paper disclosure** | §3.7 and §4.8, with the overlap fractions stated numerically |
| **Status** | resolved |

---

## D29 — Every headline Diebold–Mariano comparison is nested; standard DM is invalid there

| Field | Content |
|---|---|
| **Severity** | **F** |
| **Source** | new — `CLAUDE.md` §9.2 "DM for every comparative claim" |
| **Defect** | The comparisons that matter are all **nested**: the ladder is cumulative, so K=1's feature set is a strict subset of K=8's under the same architecture and sample; Naive-RW (`ŷ = 0`) is nested inside every model in §7; Ridge-K1 inside Ridge-K8. Under the null of equal population predictive ability with nested models and estimated parameters, the DM loss differential has a mean shifted away from zero and the statistic is not asymptotically N(0,1) (Clark & McCracken 2001; McCracken 2007). The test is systematically **undersized against the alternative the paper wants to establish** |
| **Resolution** | Split the DM prescription. **Nested** pairs (K=1 vs K=8, any model vs Naive-RW, Ridge-K1 vs Ridge-K8): Clark–West (2007) adjusted statistic, or Clark–McCracken ENC-NEW/MSE-F with their non-standard critical values — name which. **Non-nested** pairs (iTransformer vs DLinear vs PatchTST vs LSTM): standard DM with the HLN correction. Acknowledge Diebold (2015), which argues DM remains valid when *forecasts* rather than models are the object, and take a position. `CLAUDE.md` §9.2 |
| **Evidence** | The Stage 5 gate (D27) turns a title decision on a nested comparison decided by a test biased against finding the effect — a viable paper can be killed by the wrong critical value. Every RelMSE-vs-Naive claim in Tables 4 and 6 rests on a nested comparison |
| **Paper disclosure** | §3.7 — state the nested/non-nested split and the statistic used for each |
| **Status** | resolved |

---

## D30 — Seed standard deviation is the wrong error bar for origin-aggregated results

| Field | Content |
|---|---|
| **Severity** | **F** |
| **Source** | new — `CLAUDE.md` §9.2 final line against §13.4 Table 4; echoed in `paper/CLAUDE.md` |
| **Defect** | §9.2 makes seed-std the universal uncertainty display, and Table 4 aggregates over 13 origins and 6 blocks. But §9.2 itself argues, correctly, that "seeds are computational noise, not population draws" and must be averaged away *before* inference. Seed dispersion measures re-initialisation noise on one fixed dataset; origin dispersion measures the sampling variability of the estimand, and in walk-forward crypto evaluation the second is typically an order of magnitude larger. Reporting seed-std as "±" on an origin-aggregated row understates the headline uncertainty by roughly that factor |
| **Resolution** | Bind the dispersion measure to the aggregation level. Per-cell (origin, block) numbers: mean ± std across seeds, with n stated. Any number aggregated across origins: mean ± standard error across origins, or a cluster-bootstrap CI, with seed-std reported separately as a Monte-Carlo-noise diagnostic. State in §9.2 and `paper/CLAUDE.md`: "the inferential unit is the origin; seed dispersion is a diagnostic, never the uncertainty on an aggregated estimate" |
| **Evidence** | The two rules in §9.2 contradict each other — seeds declared non-inferential for the regression, then made the sole uncertainty display for the results table. This reintroduces exactly the overstated precision the wild cluster bootstrap was added to prevent |
| **Paper disclosure** | Table 4 caption and §3.7 |
| **Status** | resolved |

---

## D31 — `ŷ = 0` in scaler space is a drift forecast, not a zero-return random walk

| Field | Content |
|---|---|
| **Severity** | **F** |
| **Source** | new — `CLAUDE.md` §7 against §6.3 and §9.1 |
| **Defect** | §9.1 fixes all metrics on standardised log-returns and §6.3 confirms the target is a channel of the scaled array, `z = (r − μ_g)/σ_g` with `μ_g` fitted on the 21-month sub-block. Setting `ŷ_z = 0` therefore means `r̂ = μ_g` — the training-window mean hourly return — **not** `r̂ = 0`. The mandatory EMH baseline is a constant-drift model. Over a bull training window, `μ_g/σ_g ≈ 0.037`, whose square is ~35% of the `R²_oos ≈ 0.004` D20 anticipates; over 24 steps the tilt is `24μ_g` against `σ_g√24`, i.e. ≈ 0.18σ of systematic long bias in exactly the cumulative signal §13.5 trades on |
| **Resolution** | Define Naive-RW in **raw log-return space**, `ŷ_raw = 0`, and map it into scaler space as `ŷ_z = −μ_g/σ_g` for metric computation. Log `μ_g` and `μ_g/σ_g` per origin in `meta/*.json` so the size is auditable, and use raw-space (drift-free) returns for the §13.5 sign rule and the §9.1 DA definitions. `CLAUDE.md` §7, §9.1, §13.5 |
| **Evidence** | `μ_g` varies by origin with the bull/bear cycle — the *same* cycle H2 invokes as its mechanism — so the nuisance is confounded with the effect of interest and does not wash out across origins |
| **Paper disclosure** | §3.5 — state the space in which the baseline is defined, and §4.8 |
| **Status** | resolved |

---

## D32 — RQ1's "regressed on K and on K_eff separately" is not identified at three rung deltas

| Field | Content |
|---|---|
| **Severity** | **F** |
| **Source** | new — `CLAUDE.md` §9.1 ΔMSE row and §3 RQ1 |
| **Defect** | Four rungs give three ΔMSE values, and K_eff is measured once per rung on a fixed span (§5.4/D02) so it does not vary by origin, block or seed. Stacking 234 rows creates no information about the K_eff slope — it is identified only from between-rung variation, so the effective G is 3 and any standard error off 234 rows is fictitious. Compounding: consecutive deltas share an MSE with opposite sign, inducing mechanical correlation ≈ −0.5 that no clustering scheme addresses; and with 3 points plus an intercept there is 1 residual df, so the two specifications cannot be distinguished. Across the four rungs K and K_eff correlate at r ≈ 0.97, and essentially all discriminating leverage sits in the single 8→12 rung |
| **Resolution** | Two changes. (1) Make K_eff genuinely varying: measure PR **per origin on that origin's own 21-month training sub-block** — leak-free, since it uses training data only — turning K_eff into a 15 × 4 panel regressor identifiable with origin fixed effects. (2) Re-specify RQ1 as a within-panel comparison: fit `MSE(i,b,K) = γ_ib + f(K) + ε` with (origin × block) fixed effects clustered by origin, estimate `f(·)` as free rung effects, and compare the two theories as a **non-nested model comparison** (Vuong, or Davidson–MacKinnon J). Report `corr(K, K_eff)` in Table 2b. State plainly that with four rungs this is a comparison of effect sizes supported by a panel, not an OLS on three points. `CLAUDE.md` §3, §5.4, §9.1 |
| **Evidence** | The paper is titled after this comparison. An OLS on three points answers it with an F-test carrying 1 residual df — an unidentified model comparison that any reviewer with econometrics training will name |
| **Paper disclosure** | §3.7 and §4.2 |
| **Status** | resolved |

---

## D33 — §4.1's paths point at an empty directory; the artifact carries an out-of-window boundary bar

| Field | Content |
|---|---|
| **Severity** | **C** |
| **Source** | new — `CLAUDE.md` §4.1, §4.4, §15 against the tree; supersedes the status of D10 and D11 |
| **Defect** | Four mismatches, verified against the files on disk on 2026-08-06. (1) `data/raw/` is **empty**; all four artifacts sit in `data/`. §4.1, §4.4 and §15 all name `data/raw/`. (2) `data/raw/BTCUSDT_1h_report.json` contains **no** `fill_policy`, `rows_written` or `synthetic_bars` key — the three fields D10 quotes as its entire evidence. (3) The report reads `bars_actual` 75,095 / `missing_bars` 121 / `coverage_pct` 99.8391 against §4.1's 75,094 / 122 / 99.8378. (4) `actual_last_bar_utc` is `2026-08-01T00:00:00` although `requested_end_utc_exclusive` is the same instant, so one bar lies **past** the declared boundary — which is why 2026 coverage reads 100.02% and why the report's 121 disagrees with the 122 that `BTCUSDT_1h_gaps.csv` sums to |
| **Resolution** | **The audit's proposed fix — "re-derive §4.1 from the report (75,095 / 121 / 99.8391)" — is wrong and is rejected**: it would bake the out-of-window bar into the measured facts. §4.1's figures are the correct *in-window* values. The defect is in the artifact, not the table. Therefore: drop the `2026-08-01T00:00` bar at ingest, after which the report reproduces 75,094 / 122 / 99.8378 exactly and 2026 coverage falls to 100.0%; correct every path from `data/raw/` to `data/` (or move the artifacts, which §2's "`data/raw/` is IMMUTABLE" makes the cleaner option); and replace §11's no-imputation verification — which is unrunnable, there being no synthetic-flag column to filter — with `parquet_rows == bars_actual` plus an assertion that the timestamp diff set contains the 27 gap blocks |
| **Evidence** | The gaps CSV sums to 122 missing bars across 27 blocks; the report's 121 differs by exactly the boundary bar. Verified directly **at audit time (2026-08-05)**: the four artifacts sat in `data/`, `ls data/raw/` was empty, and `spot_klines_btc.py` (651 lines) sits at the repository root. **Closed 2026-08-06**: the artifacts were moved to `data/raw/` with `git mv` and every path in the document set updated, so the paths in this row describe the pre-move tree and are retained as the audit record only |
| **Paper disclosure** | Table 1 provenance, §3.1 — and the end-exclusive convention stated once so no coverage figure exceeds 100% |
| **Status** | resolved as a rule; **D10 and D11 are closed by this entry** — the parquet holds 75,095 rows, equal to `bars_actual`, i.e. unfilled, so the ffill defect is regenerated away; and the Stage 1 script exists as `spot_klines_btc.py`. Both closures must be re-recorded with the regeneration date and the artifact sha256 under §12 |

---

## D34 — Newey–West Bartlett weights contradict the prescribed `dm.test` validation target

| Field | Content |
|---|---|
| **Severity** | **C** |
| **Source** | new — `CLAUDE.md` §9.2, DM paragraph |
| **Defect** | Two sentences in one paragraph prescribe different estimators. "Newey–West HAC with lag h−1" means Bartlett-weighted autocovariances; R's `forecast::dm.test`, named in the next sentence as the validation target, uses the **rectangular truncated** estimator `V̂ = γ̂₀ + 2Σ_{k=1}^{h−1} γ̂_k`, which is what the HLN correction in the same paragraph is derived for. Under the DM null, h-step optimal forecast errors are MA(h−1), so all autocovariances to lag 23 are genuinely nonzero and equally real; Bartlett weights shrink γ̂₂₂ by ~92%, understating the long-run variance and producing **over-rejection** — the failure the paragraph opens by warning against |
| **Resolution** | Replace with "the rectangular (truncated) long-run variance estimator, as in Diebold–Mariano (1995) and as implemented in `forecast::dm.test`". Add the guard the rectangular estimator needs: it is not guaranteed positive in finite samples, so if `V̂ ≤ 0` fall back to Bartlett with automatic bandwidth and report that the fallback fired for that pair. `CLAUDE.md` §9.2 |
| **Evidence** | `statsmodels` `cov_hac` is Bartlett by default, so a literal implementation of the first sentence fails the validation prescribed by the second. Independently re-derived and confirmed correct in the same paragraph: the HLN formula, lag 23 at H=24, and referring `S*` to Student-t(T−1) |
| **Paper disclosure** | §3.7 and the Table 6 caption |
| **Status** | resolved |

---

## D35 — SPA and Reality Check do not control family-wise error for a pairwise matrix

| Field | Content |
|---|---|
| **Severity** | **C** |
| **Source** | new — `CLAUDE.md` §9.2 multiplicity sentence against §13.4 Table 6 |
| **Defect** | The diagnosis is right and the remedy is the wrong tool. White's Reality Check (2000) and Hansen's SPA (2005) test a **one-against-many** null — "no model in the candidate set beats a single designated benchmark" — and return one p-value for that composite. They say nothing about the all-pairs comparisons Table 6 is defined to contain. With 8+ models the matrix holds 28+ tests and remains uncorrected; at α = 0.05 roughly 1.4 spurious rejections are expected under a complete null |
| **Resolution** | Match the tool to the object. Pairwise matrix: **Romano–Wolf (2005) stepdown**, which controls FWER across all pairs and is bootstrap-based like the machinery already in the pipeline. For "which models are indistinguishable from the best" — what a reader actually wants from Table 6 — report the **Model Confidence Set** (Hansen, Lunde & Nason 2011) at 90% and 75% as a membership column. Retain SPA only where the paper genuinely poses a one-against-many null ("does any iTransformer configuration beat Naive-RW"), labelled as answering that question. `CLAUDE.md` §9.2 |
| **Evidence** | Definitional: SPA's null is `max_k E[d_k] ≤ 0` against one benchmark, not a family of pairwise nulls |
| **Paper disclosure** | §3.7 and the Table 6 caption |
| **Status** | resolved |

---

## D36 — Figure 3 plots `A(b)` for four K values; `A` is defined only as the K=1-vs-K=8 gap

| Field | Content |
|---|---|
| **Severity** | **C** |
| **Source** | new — `CLAUDE.md` §13.4 Figure 3 against §9.1 and §3 |
| **Defect** | §9.1 defines `A(i,b) = [MSE_K1 − MSE_K8]/MSE_K1` — one series — and §3 reinforces "RQ2 compares K=1 against K=8, never K=12". Figure 3 asks for four curves indexed by K ∈ {1,4,8,12}. Under the §9.1 definition `A` at K=1 is identically zero, at K=4 and K=12 undefined, at K=8 the one real series. The figure the paper is said to rest on cannot be drawn from the defined quantity |
| **Resolution** | Figure 3 shows the **single** `A(b)` series for K=1 vs K=8, plotted as 13 thin per-origin lines with the fitted `αᵢ + β₁b` overlay and its bootstrap band — which displays the actual identification (within-origin slopes) and is the more informative key figure. If a family is wanted instead, define `A_j(i,b) = [MSE_K1 − MSE_Kj]/MSE_K1` explicitly in §9.1, state that only *j* = 8 feeds the regression, and caption the *j* = 12 curve descriptive-only because of the designed redundancy. `CLAUDE.md` §9.1, §13.4 |
| **Evidence** | Left unfixed, whoever generates the figure invents a definition — and the figure and the regression then report different quantities under the same symbol, in the paper's graphical abstract |
| **Paper disclosure** | Figure 3 caption |
| **Status** | resolved — Option A adopted |

---

## D37 — The linear-span argument is not a theorem under `use_norm=True`

| Field | Content |
|---|---|
| **Severity** | **C** |
| **Source** | new — `CLAUDE.md` §5.3 reason 2 against §5.4 / D04, and against §6.2 (`d_model` 128 > L 96) |
| **Defect** | Three problems with one load-bearing argument. (1) Instance normalisation divides each channel by its **own** per-window σ; the map `x ↦ (a·x)/std_t(a·x_W)` is not linear in `x`, so a trailing average introduced as a *separate variate* is normalised by its own scale and does not lie in the span of the normalised original channel. D04 states this mechanism precisely and draws the opposite conclusion. (2) With `d_model = 128 > L = 96` the projection is generically injective, so the full lookback survives and the linear/nonlinear line is **not** the recoverable/unrecoverable line — which undercuts D22's stated rationale for the K=16 rung. (3) The exclusion list is internally inconsistent with its own justification: MA/EMA/MACD are linear in the lookback, but RSI (a ratio of sums of positive and negative parts) and Bollinger bands (a rolling standard deviation) are **not**, so the span argument excludes half its own list and would admit the other half |
| **Resolution** | Demote reason 2 from theorem to parsimony argument, and move the load to reason 1. Restate: features linear in the lookback are recoverable from the embedding in principle; adding them as separate tokens changes inductive bias and per-channel normalisation, not information content, while inflating nominal K for reasons unrelated to the families RQ1 partitions. Exclude technical indicators on the **taxonomy** ground alone (they belong to no F1–F5 family and so break RQ1's independent variable). Note in one sentence that per-channel instance normalisation is itself a nonlinearity, reconciling with D04. Re-justify D22's K=16 rung on its real distinguishing ground — it introduces multi-bar functionals that reopen the rolling-window leakage surface §5.3 closed, and therefore reopens D15's no-embargo argument. `CLAUDE.md` §5.3, and D13/D15/D22 re-verified against the restatement |
| **Evidence** | D13 and D15 both cite this argument as their evidence, so the restatement must be checked against both. The **structural** consequence survives intact and is what actually matters: no variate uses a rolling window, so the `center=True` leak class remains unrepresentable — that claim rests on the twelve definitions in §5.1, not on the span argument |
| **Paper disclosure** | §3.2 — give the parsimony and taxonomy arguments, not a span theorem |
| **Status** | resolved |

---

## D38 — §11 claims validation-based hyperparameter selection that never occurs

| Field | Content |
|---|---|
| **Severity** | **C** |
| **Source** | new — `CLAUDE.md` §6.2 against §11 |
| **Defect** | §6.2 states a single fixed configuration with no provenance: `d_model` is argued for, but lr, batch, `e_layers`, `d_ff`, dropout and patience are asserted. Nothing is selected on anything. Yet §11 asserts "Hyperparameters selected on the validation sub-block, never on B1–B6"; only ARIMA order and ridge α actually are. The document also never states that one configuration serves all four rungs, nor whether a configuration adequate at K=1 is adequate at K=12 |
| **Resolution** | State provenance ("adopted unchanged from Liu et al. 2024 except `d_model`") and state plainly that **no per-rung tuning is performed**, with the reason: holding capacity fixed is what makes the rungs comparable. Correct the §11 item to name only the models where selection occurs. Add one pre-registered robustness run at K=12 with larger `d_ff` or lr, so the flat rung is not an under-tuning artefact. `CLAUDE.md` §6.2, §11 |
| **Evidence** | Bears directly on §13.5's DSR trial count ("≈621 plus development trials") and on RQ1's linchpin: as written, the flat 8→12 rung has an untested alternative explanation |
| **Paper disclosure** | §3.4 and Table 3 |
| **Status** | resolved |

---

## D39 — Whether the loss is computed on the target channel or on all N channels is unspecified

| Field | Content |
|---|---|
| **Severity** | **U** |
| **Source** | new — `CLAUDE.md` §6.1 architecture block against §6.2 loss row |
| **Defect** | The architecture selects the target channel at the output; the loss row says only "MSE". Standard iTransformer implementations compute the loss over **all N channels**. The two are materially different objectives and the difference scales with K: an all-channel loss makes K=12 a 12-task problem and K=1 a 1-task problem, so the amount of auxiliary supervision varies with the study's independent variable |
| **Resolution** | Pin it: the loss is **MSE on the target channel only**, at every rung, so the training objective is identical across the ladder. Add it to §11 as a verifiable assertion. `CLAUDE.md` §6.1, §6.2, §11 |
| **Evidence** | If the loss were all-channel, RQ1's axis is confounded by the number of auxiliary tasks and K=1 is not the stated control but a different learning problem. The reference implementation defaults to the option that breaks the design |
| **Paper disclosure** | §3.4 |
| **Status** | resolved |

---

## D40 — Baselines carry no K assignment, so the channel-independence comparison is undefined

| Field | Content |
|---|---|
| **Severity** | **U** |
| **Source** | new — `CLAUDE.md` §7 against §10.2 run accounting |
| **Defect** | Ridge is explicitly "on the same K features" and budgeted at 13 × 4 K. LSTM, DLinear and PatchTST have no K assignment, and §10.2's `13 × 13 = 169` gives each exactly one run per origin — a single, unstated feature set. Yet §13.1 makes the channel-independence debate a Related Work pillar and PatchTST is described as "SOTA, channel-independent". A channel-independent baseline evaluated at an unknown K cannot speak to that debate |
| **Resolution** | Assign K explicitly for every baseline and revise §10.2: run DLinear, PatchTST and LSTM at **K=8** minimum (matching the RQ2 rung), ideally across all four rungs as ridge already is. Report the K of every model in Tables 3 and 4. `CLAUDE.md` §7, §10.2 |
| **Evidence** | The comparison an LTSF-literate reviewer cares about is iTransformer at K=8 versus PatchTST at K=8 on identical information. If PatchTST silently ran at K=1, the paper's central architectural comparison collapses into univariate-versus-multivariate |
| **Paper disclosure** | Tables 3 and 4, and §3.5 |
| **Status** | resolved |

---

## D41 — `b*` has no estimator, no origin index, and no method for censored data

| Field | Content |
|---|---|
| **Severity** | **U** |
| **Source** | new — `CLAUDE.md` §3 RQ3, §9.1 `D(b)` row, §13.4 Table 5 |
| **Defect** | D06 gave `A` an origin index; `D(b)` and `b*` never received one. It is undefined whether they are computed per origin then aggregated, or on pooled MSEs — and `min{·}` does not commute with averaging. Per origin, the study yields 13 values of `b*`, many right-censored at 6: interval-censored survival data, for which no estimator is prescribed, so no median cadence and no interval can be produced. H3 ("larger K decays faster") has no test specified anywhere |
| **Resolution** | Index it: `D(i,b)`, `b*(i) = min{b : D(i,b) > τ}`, censored at 6. Analyse the 13 values with an interval-censored survival estimator (Turnbull/Kaplan–Meier on the 30-day grid) and report the **median `b*` with a confidence interval** per τ. Test H3 with a log-rank test across K, or an interval-censored AFT model with K as covariate. Table 5 carries the interval, not a point. `CLAUDE.md` §9.1, §9.2, §13.4 |
| **Evidence** | RQ3's answer is claimed contribution #3 and §13.1 requires the cadence in the abstract; a bare integer there violates `paper/CLAUDE.md`'s own "never a bare number". Downstream of D23: fix the metric scale first, or every `b*(i)` is censored and the survival curve is degenerate |
| **Paper disclosure** | §3.7, §4.4, Table 5 |
| **Status** | resolved |

---

## D42 — The wild cluster bootstrap recipe is incomplete at every choice that moves the p-value

| Field | Content |
|---|---|
| **Severity** | **U** |
| **Source** | new — `CLAUDE.md` §9.2 decay-regression requirement 3 |
| **Defect** | The paragraph names the weight distribution and the software and omits everything else. (1) **Restricted (WCR, null imposed) versus unrestricted (WCU)** is unstated — the single most consequential omission at small G (MacKinnon, Nielsen & Webb 2023). (2) It is not stated that the bootstrap applies to the **studentized** statistic; the asymptotic refinement comes from bootstrapping *t*, not β̂ (Cameron, Gelbach & Miller 2008). (3) B is unset. (4) α and sidedness are unset, though §3 states the directional claim β₁ < 0 — and a one-sided test chosen after seeing the sign is not pre-registered. Separately, the stated reason for preferring Webb does not bind: 2¹³ = 8,192 Rademacher draws give a minimum two-sided p ≈ 0.00024, far below α = 0.05. Also unstated: the reference distribution is **t(G−1) = t(12)**, not the software default t(N−K) = t(64) — a 12% difference in critical value — and with 13 origin dummies plus a slope the CRV meat matrix has rank ≤ 13 against 14 parameters, so only single-coefficient tests are available |
| **Resolution** | State the complete recipe: "Wild cluster **restricted** bootstrap (impose H₀ when generating samples), bootstrapping the cluster-robust *t*-statistic, B = 99,999, cluster = origin, one-sided test of H₁: β₁ < 0 at α = 0.05 declared in advance, `wildboottest` with `impose_null=True`. Cluster-robust *t* is referred to **t(G−1) = t(14)**, never t(N−K) = t(74); with 15 origin dummies plus a slope the CRV matrix has rank ≤ 15 against 16 parameters, so only single-coefficient tests are reported." Replace the Webb justification with: "report both Rademacher and Webb weights; if they disagree the more conservative is the headline." Note that with a balanced panel β̂₁ equals the mean of the 15 origin-specific within-slopes, so **the effective n for β₁ is 15, not 90** — and effective *independence* is nearer 4 (`D28`). State both so the observation count does not imply power that does not exist. `CLAUDE.md` §9.2 |
| **Interaction with `D26`** | The defect above was diagnosed at G = 13, where the small-G regime genuinely bites. `D26`'s re-cut raises G to 15, above the literature's G ≲ 12 threshold, and Rademacher then admits 2¹⁵ = 32,768 draws (minimum two-sided p ≈ 6·10⁻⁵). The recipe is unchanged — WCR, studentized, both weight schemes reported — but the *reason* for reporting Webb is now robustness, not necessity |
| **Evidence** | β₁'s p-value is the abstract's headline number. As written, two implementers get different p-values from identical prediction files — the exact failure §12's traceability contract exists to prevent |
| **Paper disclosure** | §3.7, with the cluster count and effective n both stated |
| **Status** | resolved |

---

## D43 — The leakage surface is declared closed on an incomplete enumeration

| Field | Content |
|---|---|
| **Severity** | **U** |
| **Source** | new — `CLAUDE.md` §5.3 closing corollary and §8.3, feeding D15 |
| **Defect** | The no-rolling-window claim itself checks out: all twelve variates in §5.1 are per-bar functions and `r` uses only the current and previous close. The **closure claim built on it** does not. "The leakage surface collapses to exactly two paths — segment crossing and scaler fitting" is scoped to *feature construction* at the *train–test* boundary. It omits, and therefore leaves unassertable: the train–validation boundary (D24); the model-selection channel (D27); the evaluated-sample-composition channel (D45); and the between-origin training-window overlap channel (D28). Because §5.3 declares the surface closed, §11 contains no checklist item for any of the four |
| **Resolution** | Narrow the claim to what the argument establishes: "no *rolling-feature* leak path exists, so the `center=True` class is unrepresentable." Then re-enumerate the surface as a grid of **boundary** (train/val, train/test, cross-origin) × **channel** (features, labels, scaler, model selection, evaluated-sample composition), with one §11 item per non-empty cell. Keep §8.3's feature-lookback argument verbatim — it is correct — but state that it covers one cell of that grid, not the whole surface. `CLAUDE.md` §5.3, §8.3, §11; D15 re-verified |
| **Evidence** | D15's paper disclosure sends the two-path argument into the manuscript. A reviewer who knows López de Prado tests an enumeration for completeness, and it fails at the first boundary that is not train–test. A closed-surface claim is also what stops the hunt §2 mandates ("assume leakage until proven otherwise") |
| **Paper disclosure** | §3.6 — present the grid, not a two-path list |
| **Status** | resolved |

---

## D44 — K_eff's span is undeclared for the RQ1 regressor, and the statistic is blind to cross-lag structure

| Field | Content |
|---|---|
| **Severity** | **U** |
| **Source** | new — `CLAUDE.md` §5.4 measurement table against §3, §11 item 5, §13.4 Table 2b |
| **Defect** | Two problems. (1) **Span.** Every span constraint in the document names either *the gate* or *the rolling* statistic. §5.4's first table row — the static per-rung PR that "supplies K_eff" and that §13.4 routes into RQ1 — declares no span, so nothing forbids computing RQ1's regressor over 2018–2026, which contains all thirteen origins' test blocks. §11 item 5 audits only the gate, so this path survives every existing checklist item by construction. (2) **Construct.** PR is computed on the K × K correlation matrix of *contemporaneous* variate values, while the model consumes a K × 96 block and embeds each variate's entire lookback. Two variates can be near-uncorrelated contemporaneously yet near-redundant to a model with a 96-hour lookback, and conversely. The statistic is blind to exactly the cross-lag structure the inverted embedding exists to exploit |
| **Resolution** | (1) Compute K_eff **per origin on that origin's own 21-month training sub-block** — which also makes it vary, and is what D32 needs. Reserve the pre-first-origin PR for the gate alone; label any full-sample PR descriptive and never use it as a regressor. Amend §11 item 5 to "every reported K_eff, including the RQ1 regressor, is computed on training-only spans; the gate additionally uses the pre-first-origin span." (2) Report the contemporaneous PR **and** at least one lookback-aware measure on the same rungs — the PR of the K·L × K·L covariance spectrum, or the stable rank of the K × 96 window block averaged over windows. Pre-register which is RQ1's regressor before Stage 3b; report the divergence between measures in §4.1b whatever it is. `CLAUDE.md` §5.4, §11, §13.4 |
| **Evidence** | If K_eff is a full-sample statistic, the regressor is estimated on the same data as the outcome and RQ1's claim is partly circular. If the construct does not correspond to what the model sees, contribution #2 is a measurement-validity failure rather than a finding |
| **Paper disclosure** | §3.3, §4.1b, and §4.8 |
| **Status** | resolved |

---

## D45 — Window loss is reported globally; per-origin and per-block it ranges from 0% to 50%

| Field | Content |
|---|---|
| **Severity** | **U** |
| **Source** | new — `CLAUDE.md` §4.3 cost accounting, §8.2, §8.5 Stage 2 gate, §11 item 12 |
| **Defect** | §4.3 gives one pooled figure (≈ 4.9%) and then requires the rejected-window count to be "asserted per origin against this estimate", with `breaks` never defined per origin and no per-origin break table anywhere. Measured against `data/raw/BTCUSDT_1h_gaps.csv`, 26 of 27 blocks fall in 2018–2021 and none after 2023-03, so per-origin **training**-window loss runs from ~11% down to 0% and is monotone in calendar time, while per-(origin, block) **test**-window loss runs 0%–50.4% — origin 2020-07 loses 363 of 720 window starts in B6 (breaks at 2020-11-30, 12-21, 12-25 ⇒ 119 × 3 + 6). Origins 12 and 13 contain zero breaks, so an assertion against a 4.9% global estimate fires spuriously at eleven of thirteen origins and would predictably be loosened until it passes — disarming the single defence against the pipeline's most likely silent failure. Two consequences follow: (a) the gap distribution reintroduces, monotonically in time, the training-volume variation the fixed rolling window was chosen to eliminate; (b) test-window survival is conditioned on *future* gaps, and since Binance outages cluster on stress the dropped targets are systematically the high-volatility ones, so β₁ absorbs a within-origin coverage trend |
| **Resolution** | **Done: `docs/ORIGIN_WINDOW_BUDGET.md`** carries the per-origin and per-(origin, block) table, derived from `gaps.csv` and committed so a Stage 2 run has something to be checked *against* rather than tuned *to*. Under `D26`'s 15-origin grid the measured spread is **13,558 … 15,217** training windows (0.0%–11.2% loss) and test-block loss reaches **33.9%** (origin 1, B2), with every origin from 9 onward clean. Rewrite §11 item 12 to assert **exact equality per origin** — the quantity is computable, not estimated. Move §8.5's Stage-2 window-loss gate from the global series to the per-(origin, block) level. Report surviving-window counts per cell in Tables 1 and 5, add block coverage as a covariate in the decay regression or re-run β₁ restricted to cells with ≥ 95% coverage, and either subsample every origin's training set to the smallest origin's count or report a sensitivity holding training size constant. Add a §11 item: every baseline is evaluated on **exactly** the surviving window set of the run it is compared against — assert equality of evaluated timestamps before computing RelMSE, since Naive-RW needs no 96-bar lookback and would otherwise be scored on a different sample. `CLAUDE.md` §4.3, §8.2, §8.5, §9.2, §11 |
| **Evidence** | The 50.4% figure was re-derived independently from `gaps.csv` and matches to the window. §4.3 names the bug the assertion exists to catch ("positional sliding closes gaps invisibly"); specified at the wrong granularity it cannot catch it |
| **Paper disclosure** | Table 1 (per-origin), Table 5 (per-block), §4.8 for the future-conditioned exclusion |
| **Status** | resolved |

---

## D46 — The economic evaluation has an unfixed trade phase, gap-spanning returns, and an unimplementable DSR

| Field | Content |
|---|---|
| **Severity** | **U** |
| **Source** | new — `CLAUDE.md` §13.5 against §4.3 and §9.1 |
| **Defect** | Four problems. (1) The **phase** of the non-overlapping partition is unspecified — 24 admissible alignments per block, each with a different Sharpe and MDD, and nothing forbids choosing it after seeing the equity curve. (2) A position held across a downtime block has no defined realised return, and the obvious `log(C_{t+24}/C_t)` is exactly the cross-gap return §4.3 forbids everywhere else; at the 2018-02-08 block a nominal 24-hour trade would book a 57-hour move. (3) Positions exist only where a valid window exists, so per D45 the strategy is flat precisely across outages — and outages cluster on stress, so the omitted periods are disproportionately the large-drawdown ones and reported MDD is optimistic by an unbounded amount. (4) The **DSR is not computable from the specification**: `SR₀` requires `V[SR]`, the variance of Sharpe ratios across the N trials, plus the skewness and kurtosis of per-period returns — none mentioned. And N ≈ 621 is the wrong quantity: DSR's N counts candidates whose Sharpe was computed on the *same* return series, whereas the 621 runs span 13 largely disjoint test periods, seeds, horizons and baselines that never competed for one backtest |
| **Resolution** | Fix the phase in advance (positions open at 00:00 UTC) and state it. Define the realised holding return **per segment**, using the same rule as `r`, and skip any holding period containing a break rather than computing across it; report the count of skipped periods per block. Rewrite the DSR sentence as a recipe: computed per origin on the T non-overlapping 24-hour strategy returns, using the per-period Sharpe, the sample skewness and kurtosis of those returns, and `SR₀` from **N = the number of distinct configurations evaluated on that origin's test span**, with `V[SR]` the observed variance of their Sharpe ratios; the 621-run total is reported separately as the development trial count and discussed in Limitations, and is **not** N. Attach inference to the rest of Table 8: a Ledoit–Wolf or Jobson–Korkie/Memmel test for Sharpe differences against the naive strategy, and bootstrap intervals for MDD. `CLAUDE.md` §13.5, §11 |
| **Evidence** | With N = 621 and T = 180, the DSR threshold is `SR₀ + 1.645/√(T−1) ≈ SR₀ + 0.123`, essentially unmeetable — a second guaranteed null alongside D23, and one that reads to a referee as either a failed strategy or a misapplied statistic with no way to tell which. §13.5 is also the one place returns are computed outside the segment-aware feature code, which is why the cross-gap return would appear there and nowhere else |
| **Paper disclosure** | §3.7, §4.7, Table 8, Figure 7, and §4.8 |
| **Status** | resolved |

---

## D47 — The learning-rate schedule makes the 30-epoch budget and patience-5 decorative

| Field | Content |
|---|---|
| **Severity** | **U** |
| **Source** | new — `CLAUDE.md` §6.2 |
| **Defect** | Under type-1, lr = 1e-4 · 2^(1−epoch): by epoch 9 it is ~4e-7 and by epoch 15 ~6e-9. Effective learning stops after roughly 7–8 epochs, so "max epochs 30" and "patience 5 on validation MSE" describe a budget that can never bind. The only convergence check in the document is §16's single-batch overfit, which passes trivially and says nothing about the full-data run |
| **Resolution** | Either extend the schedule (halve every 3–5 epochs, or cosine) so the budget and patience are real, or state plainly that the effective budget is ~8 epochs and justify it. Log and report epochs-to-early-stop and final training loss **per rung**, so under-training can be ruled out from the reported numbers rather than assumed away. `CLAUDE.md` §6.2, §10.4 |
| **Evidence** | Training adequacy is otherwise an alternative explanation for every K contrast: a K=12 model has 12 tokens of attention structure to fit in the same ~7 effective epochs as a K=1 model with none, so a flat 8→12 rung is equally consistent with redundancy and with insufficient optimisation |
| **Paper disclosure** | Table 3 or the supplement |
| **Status** | resolved |

---

## D48 — Four outcome-determining choices are left to be made after results are seen

| Field | Content |
|---|---|
| **Severity** | **U** |
| **Source** | new — `CLAUDE.md` §10.2, §13.4 Figure 5, §13.5, §5.4/§8.5 |
| **Defect** | (1) The horizon sweep uses "4 origins" and never says which — choosing them after the main grid is origin selection. (2) Figure 5 contrasts "calm vs stress regime" with neither defined, so the windows can be picked after seeing the attention maps. (3) The fee is fixed exactly (0.04% per side) and slippage not at all, although the project's own reference library anchors BTC effective spreads near 0.30% — an assumption that would reverse the sign of every number in Table 8. (4) The Stage 3b gate says "re-cut the ladder" if PR < 5.0 but never says re-cut to *what*, while D01 argues exactly one consistent cut exists — so the gate has no defined action. Related: D22's K=16 arm has an unverifiable run condition ("the Stage 5 pilot passes", "quota allows") and no pre-stated predicted outcome |
| **Resolution** | Fix all four before Stage 5. Name the sweep origins in §10.2. Define calm/stress numerically (bottom and top tercile of realised volatility across all blocks) so Figure 5's windows are data-determined. State a slippage figure with a pre-registered sensitivity band (0.02% / 0.05% / 0.10% per side) and report Table 8 at all three. Replace the PR gate's action with either a named fallback cut or "report the measured PR and proceed unchanged, disclosing the divergence from expected values". For D22: name the pilot origin, the test and α; state the GPU-hour figure below which quota counts as insufficient; pre-register the predicted direction; fix seeds, horizon and origin set now; and if the arm is not run, report the exact clause that failed with its number. `CLAUDE.md` §5.4, §8.5, §10.2, §13.4, §13.5 |
| **Evidence** | A document otherwise scrupulous about pre-registration leaves four levers unfixed, three of which directly determine a reported figure — and the pre-registered thresholds are precisely the ones that cost nothing, while slippage, which decides whether the strategy makes money, is blank |
| **Paper disclosure** | §13.2 as a named item, plus each section listed above |
| **Status** | resolved |

---

## D49 — The flat 8→12 rung has the fewest seeds and no equivalence test

| Field | Content |
|---|---|
| **Severity** | **I** |
| **Source** | new — `CLAUDE.md` §6.2 seeds row (D18) against §5.2 and §3 |
| **Defect** | Two coupled problems. (1) D18 raised seeds on the RQ2 rungs and left K ∈ {4,12} at n=3 — but the 8→12 rung **is** RQ1's designed contrast and the demonstration that nominal K is the wrong axis. D18's own reasoning applies to it with equal force, and the unbalanced counts make ΔMSE per rung heteroskedastic (each delta pairs an n=3 cell with an n=5 cell, and adjacent deltas share a cell, inducing correlation ≈ −0.5) with no weighting specified. (2) "Flat 8→12" is an assertion of **no effect**, and nothing specifies how a null is established. A non-significant ΔMSE is a failure to reject, not evidence of equivalence — and at n=3 it is a near-guaranteed failure to reject regardless of the truth |
| **Resolution** | Raise K ∈ {4,12} to **5 seeds** (15 × 2 rungs × 2 seeds = 60 runs, ≈1.5 GPU-hours against a 30 h quota). Pre-register an **equivalence test** for the flat rung: two one-sided tests on ΔMSE₈→₁₂ against a margin `Δ_eq` fixed in advance — the natural choice being a stated fraction of the 4→8 gain — stated in §3 alongside τ, because choosing it after seeing the rung is the p-hacking §3 already forbids. Specify inverse-variance weighting for ΔMSE contrasts while seed counts remain unequal. Report n in every row of Table 4. `CLAUDE.md` §3, §6.2, §9.1, §9.2, §10.2 |
| **Evidence** | §5.2 and `paper/CLAUDE.md` both insist a flat rung is the designed contrast and not a null, while the machinery to demonstrate that distinction is absent — so the paper would assert it without being able to show it |
| **Status** | resolved |

---

## D50 — The uniform-attention control is budgeted for Figure 5 but never used as an arm

| Field | Content |
|---|---|
| **Severity** | **I** |
| **Source** | new — `CLAUDE.md` §6.2 K=1 degeneracy note against §13.2 |
| **Defect** | Two problems. (1) The stated K=1 reduction is wrong in detail: at N=1 the softmax weight is 1, but the attention block still applies the value and output projections and the residual, so K=1 is `Linear(L→d_model) → [W_O W_V x + x] → LayerNorm → FFN → Linear(d_model→H)`, not the stated form. (2) More seriously, K=1 versus K=8 varies **information and architecture simultaneously**, so it does not "isolate exactly the contribution of cross-variate attention". A decaying `A(b)` is equally consistent with "cross-variate attention overfits regime-specific structure" — a capacity story — as with the information story RQ2 claims. The clean control, an iTransformer at K=8 with attention forced uniform, is already required by §13.2 for the interpretability section and is never used as an arm |
| **Resolution** | Correct the K=1 reduced form in §6.2 and state that parameter count is identical across rungs. Promote **uniform-attention K=8** to a third arm in the main grid (15 origins × 5 seeds = 75 runs, inside quota) and define a second contrast `A_attn(i,b) = [MSE_uniformK8 − MSE_K8]/MSE_uniformK8` alongside `A(i,b)`; reporting both decompositions answers information-versus-attention directly, and reuses runs Figure 5 already needs. `CLAUDE.md` §6.2, §9.1, §10.2, §13.2 |
| **Evidence** | Ridge (D17) separates information from attention only under a linear model. The uniform-attention arm is the only control that does it within the architecture under study |
| **Status** | resolved |

---

# Sixth pass — the Kaggle deployment surface, 2026-08-07

`D51` came from asserting the data accounting, `D52` from building the features and the network,
`D53` from building the experiment plane. **`D54` came from asking what the notebook actually needs
in order to run on Kaggle** — the deployment surface, which is exactly what the unrun
Kaggle/execution audit lens would have examined.

## D54 — The launcher cannot run without a second Dataset, and loses both halves of §12's contract there

| Field | Content |
|---|---|
| **Severity** | **C** |
| **Source** | new — `notebooks/itransformer_kaggle.ipynb` cell 2 against `CLAUDE.md` §10.5, §12, §15 |
| **Defect** | Three defects with one root cause: the launcher assumed the repository would be present on the machine that runs it. (a) **Cell 2 globbed for `src/itransformer_btc/__init__.py` and pushed the hit onto `sys.path`**, so the notebook could not run unless the repository was *also* uploaded as a Kaggle Dataset and kept in step with the notebook by hand — two artifacts that must agree, with nothing checking that they do. (b) **`_git_sha()` returns `"unknown"` on Kaggle**, there being no git repository there; §12 names the git sha as one of the three things every number must resolve to, so the traceability contract lost its code half at precisely the place the grid executes. (c) **`_input_sha256()` read the hard-coded path `data/raw/BTCUSDT_1h_report.json`**, which does not exist on Kaggle — the artifact arrives under `/kaggle/input/<slug>/` and §10.5 forbids hard-coding that slug — so the input digest also logged as `"unknown"`, and §12's rule that numbers from different vintages may not share a table became unenforceable |
| **Resolution** | (a) The notebook **carries the package** in twelve `%%writefile` cells, materialises `itransformer_btc/` into the working directory before importing it, and then asserts at runtime that the imported `__file__` lives there — so a stray `src/` on `sys.path` fails loudly instead of silently supplying different code. Files rather than in-cell definitions, because the grid runs as two **subprocesses** pinned one per GPU and a subprocess inherits none of the kernel's namespace. (b) `train.code_sha256()` hashes the package's own source, line endings normalised so a CRLF checkout and an LF materialisation of the same logic give the same digest; it is recorded in every `meta/*.json` beside `git_sha`. (c) `train.resolve_input_parquet()` reads the `ITBTC_PARQUET` environment variable — set by the notebook, by `launch_workers` for each child, and by the worker CLI from its own `--parquet`; `_input_sha256()` prefers the Stage 1 report sitting beside the artifact and falls back to hashing the parquet, recording which in `input_sha256_source`. `CLAUDE.md` §10.5, §12, §15, §16 |
| **Evidence** | Measured 2026-08-07 by materialising the package into a scratch directory with the repository absent from `sys.path`, then running cells 0–24 plus one worker subprocess: `code_sha256` identical in-process and in the child, `input_sha256 = 8270a84b07c2923b…` matching §4.1's pinned digest at `input_sha256_source = "report"`, `git_sha = "unknown"` exactly as on Kaggle, and every §4.1/§5.4/§6.2 figure reproduced — 75,094 bars, 3 unusable, budget exact at all 15 origins, gate PR 4.393, `corr(K, K_eff)` 0.828, 280,472 parameters, `μ_g/σ_g` spanning −0.00818…+0.01733 |
| **Cost, and how it is paid** | A second copy of ~4,000 lines, which is the failure mode this register exists to prevent. It is paid down rather than accepted: the copy is **generated** by `tools/build_notebook.py`, and `tests/test_notebook_sync.py` asserts it byte-identical to `src/`, so editing `src/` without regenerating fails the suite. Hand-editing the notebook's package cells is a defect, and the generator silently reverts it on its next run |
| **`D54e` — partial sessions** | **F.** The grid is ~10–20 wall hours against an 11 h budget, so a session that ends mid-grid is the *expected* case. It crashed. A partial grid is an unbalanced panel, and §9.1's estimators refuse one by design — `amplification` raises rather than compare K=1 at eleven origins against K=8 at ten, and RQ1's `wide[4] - wide[8]` broadcast-errors first. Simulated at the real two-shard stop shape (200 of 534, round-robin by group): K=1 complete at 11 origins, K=4/8/12 at 10. **The estimators are right; where the exception landed was not** — the last cells of a twelve-hour session, marking the Kaggle version failed at the moment its output was the only thing worth keeping. Resolution: RQ1/RQ2/RQ3 and the `paper_numbers.json` write are gated on `GRID_COMPLETE`; they print what remains and how to resume, and exit cleanly. Partial evaluation is never offered as a fallback, because a half-panel β₁ is a different estimand rather than a noisier one |
| **`D54f` — the budget clock** | **C.** `BudgetGuard.deadline` is set from `time.perf_counter()` *inside each worker*, while Kaggle's 12 h wall runs from the notebook's first cell. The prelude — Stage 2, Stage 3b, Stage 4 and the twelve pilot training runs of Stage 5, ~20–25 min — therefore sat outside the budget, and the two clocks drifted apart by however long it took. Resolution: the notebook stamps `SESSION_T0` in cell 0 and passes `budget_h = 11.0 − elapsed`, so the guard bounds what §10.1 actually limits. Hitting the wall interactively loses `/kaggle/working` entirely, so the margin is not somewhere to be approximate |
| **Resume, verified** | Granularity is one `run_id` (~90 s), so a session cut at run 200 of 534 loses at most the run in flight per GPU. Demonstrated 2026-08-07: a worker re-invoked on a shard holding one completed run reported `pending in shard=0` and did no work. `discover_roots`' two glob expressions were checked against both Kaggle layouts — `<slug>/preds` and `<slug>/artifacts/preds` — with a data-only Dataset correctly ignored |
| **Disclose in** | §3.1 (provenance) — the manuscript states that runs were produced by a notebook-materialised copy of the package identified by `code_sha256`, not by a git checkout |
| **Status** | resolved |

---

## Smaller corrections absorbed without a numbered entry

Each was verified and each changes one sentence, not a rule:

- **D20's evidence row splits in two.** `RelMSE = 1 − R²_oos` is an **exact** identity for any `ŷ_bench = 0` and any mean of *y*, so the "approximately mean-zero" caveat is not load-bearing there and is deleted. Separately, `MSE_naive = (1/n)Σr² = V̂ar(r) + r̄²` equals the variance **only** when `r̄ ≈ 0` — and on scaler-space returns `r̄ ≠ 0` in general because `μ_g` is fitted on a different window (D31). The current row labels the pair "identity, not approximation", which is half right and half wrong.
- **D05 gains the reason it works, and a caveat.** RelMSE is well behaved because `MSE_model` and `MSE_naive` on the *same* block correlate near 1, so their errors largely cancel — the argument that makes the whole ratio-metric design defensible, currently unstated. But `RelMSE(b)` and `RelMSE(1)` come from different blocks and are essentially uncorrelated, so nothing cancels in `D(b)`: one noisy 30-day reference block sits in the denominator of `D(i,2)`…`D(i,6)`, and `b*` reads a threshold crossing straight off it. Replace the single-block reference with `D(i,b) = RelMSE(i,b)/mean_{b'}[RelMSE(i,b')] − 1`, or fit a within-origin trend and read the crossing off the fitted line; attach a block-bootstrap band.
- **D02's evidence row is factually wrong** and is replaced. "2018-01 → 2020-01 is inside every origin's training window" is false under a 24-month **rolling** window: origins 5–13 contain none of the gate span. The conclusion survives — the gate span ends where the first test block begins — but the published reason must be the correct one. Add the substantive caveat the register missed: the gate span coincides with origin 1's training window only, so under a rolling window the K=8 correlation structure may drift across origins, which is a further argument for the per-origin K_eff of D44.
- **The DM unit of observation is pinned.** DM is computed per (origin, block) on the overlapping hourly loss differential, T ≈ 720, h = 24, truncation lag 23; block-level statistics are combined across cells by a stated method, **never** by concatenating `d_t` across origins, because the model changes at each origin and the DM null has no interpretation across that boundary. Assert `(T + 1 − 2h + h(h−1)/T) > 0` before applying the HLN factor and refuse to report where it fails — at h = 24 the factor is exactly 0 at T = 24 and 0.047 at T = 30, precisely the T a non-overlapping 30-day block would produce. State T alongside every reported p-value.
- **Pesaran–Timmermann is aligned with the trading rule.** DA at h=1 on hourly-spaced forecasts, tested with PT (1992). DA at h=24 and DA on the cumulative 24-hour return computed on **non-overlapping** windows matching §13.5, tested on that sample; the overlapping versions are descriptive only, without p-values. On hourly spacing those two variants' targets overlap by 23 of 24 hours, giving lag-1 autocorrelation ≈ 23/24, so PT's variance is far too small and the test over-rejects badly. Note the resulting power loss explicitly.
- **The order of operations for seed averaging is fixed.** All ratio metrics (`A`, RelMSE, `R²_oos`, `D`, ΔMSE) are formed from **seed-averaged MSEs**, never from averages of per-seed ratios — the two differ by Jensen, and the second additionally requires pairing seed 42 at K=1 with seed 42 at K=8, which are independent training runs of different models, so any of 5! orderings gives a different answer. Note that the cell mean still carries Monte-Carlo error, which enters as measurement error in the dependent variable: unbiased for β₁, but inflating residual variance, which is why n is reported per cell.
- **The log-ratio column is demoted to an appendix.** With `A` bounded in roughly [0, 0.005] given `R²_oos ≈ 0.004`, `log(MSE_K1/MSE_K8) = A + A²/2` differs from `A` by under 0.1% relative — two columns identical to five decimal places. Report the transform once as a robustness check, state the trigger for revisiting if `A` empirically spans wider, and use linear `A` throughout.
- **The `H == L` bar count is measured before any estimate depends on it.** §4.3's thirty breaks account for 27 downtime blocks and 3 zero-volume bars; the number of `H == L` bars is in neither the report nor anywhere else, yet §11 asserts a reconciliation against it. Measure it as a Stage 2 output, confirm whether the 3 zero-volume and 3 zero-trade bars are the same bars, and add both counts to the report schema and Table 1 before §4.3's arithmetic and D14's cost figure are restated.
- **§11 item 13 is restated in window terms.** "Split indices persisted as JSON and verified non-overlapping" fails in both readings: within an origin, row-index disjointness is true by construction and does not imply *window* disjointness, which is exactly D24's leak; across origins, splits must overlap by design (D28), so the item is unsatisfiable and would be quietly dropped. Replace with `max(target_index over training windows) < val_start`, `max(target_index over validation windows) < test_start`, plus row-level disjointness per origin — and an explicit note that cross-origin row overlap is by design, so no one later "fixes" it.
- **The priority claim is hedged and documented.** "First walk-forward evaluation of iTransformer on a crypto asset with explicit decay measurement" becomes "to the best of our knowledge, the first …", accompanied by a one-paragraph search protocol in §2 (databases, exact query strings, search date, inclusion criteria, hits screened). §13.3 already declares every reference-library entry unverified, so the evidence base for a bare priority claim is acknowledged as unverified in the same document set.
- **§13.2 gains four disclosures**: statistical power and the minimum detectable β₁; the single-venue, single-pair scope, naming which microstructure variates are venue-specific; that hyperparameters were adopted rather than tuned and are identical at every rung (D38); and a study-level multiplicity plan declaring which tests are confirmatory (β₁ at τ = 5%) and which exploratory (τ sensitivities, horizon sweep, per-rung DM cells).
- **A power statement is added before the grid runs.** No minimum detectable β₁ appears anywhere, although every design choice (5 seeds, 15 origins, wild bootstrap) implies someone reasoned about precision. Estimate the between-origin dispersion of the within-slope from the Stage 5 pilot, compute the MDE at G = 15 by simulation, publish it in the methodology, and pre-register the interpretation of a null. If the MDE exceeds the plausible magnitude of `A`, reposition RQ2 as descriptive **before** running the grid.
- **The technical-indicator critique is either tested or dropped as positioning.** Related Work builds the paper's stance on a critique of a literature that uses RSI, MACD and Bollinger bands, while the design never tests whether they help. Either add one pre-registered K=8+TI arm or an indicator-set ridge/DLinear baseline at all origins (~50 runs), reported once in Table 4 — turning "we excluded them" into "we tested and excluded them" — or drop the critique from the positioning.

---

## Open items requiring external action

| ID | What must happen | Blocks |
|---|---|---|
| ~~**D10**~~ | ~~Regenerate the parquet with no fill~~ — **closed by D33**: the parquet holds 75,095 rows = `bars_actual`, i.e. unfilled. Record the regeneration date and sha256 under §12 | — |
| ~~**D11**~~ | ~~Restore `binance_spot_klines.py`~~ — **closed by D33**: the Stage 1 script exists at the repository root as `spot_klines_btc.py` (651 lines) | — |
| **D16** | Verify every citation against its source | manuscript submission |
| **D33** | Drop the `2026-08-01T00:00` boundary bar at ingest, then re-emit the report. **Path fork closed 2026-08-06 by moving** the four artifacts into `data/raw/`, not by rewriting the paths to `data/` | Stage 2 onwards; Table 1 |
| **D26** | Re-cut the origin grid to break the block/calendar-month collinearity, then re-derive the origin count and §10.2's run accounting | Stage 4 onwards |
| **audit** | Run the two adversarial lenses that died on a session limit — **consistency** and **Kaggle/execution**. Neither has examined this document set. The Kaggle lens is the only one that would have audited §10 end to end, and §10's numbers are among those D25 corrected | Stage 4 sign-off |

## Related unresolved questions

Tracked in `.claude/prds/claude-md-rewrite-spot-itransformer.prd.md` under *Open Questions*: the
Python/torch version pin, RQ3's 30-day answer resolution, and the manuscript's authoritative
language. None of them changes a rule in `CLAUDE.md`; each changes work that follows from one.

---

# Long-form passes moved from `CLAUDE.md` §14, 2026-08-24

Passes three through eleven (`D51`–`D62`) were written into `CLAUDE.md` §14 and lived there
alone (`D60f`). They are reproduced below **verbatim**, unedited, so §14 can carry a one-line index
per ID instead of the full text. §15 already calls this file the long-form evidence; this is that
claim made true. The sixth pass expands the single `D54` entry above rather than replacing it.

# Third pass — the first defects found by *running* the code

**`D51` is what Stage 2 was for.** It is a single register entry covering four defects that no amount
of re-reading would have produced, because each is a disagreement between the document set and the
artifact, found on 2026-08-06 by building `src/itransformer_btc/` and asserting
`docs/ORIGIN_WINDOW_BUDGET.md` against `BTCUSDT_1h.parquet`. The derived table diverged at **twelve of
fifteen origins**. This is the pattern §14's preamble predicted: the two unrun audit lenses were
consistency and execution, and every one of these sits in exactly that gap.

| ID | Sev | Defect | Resolution |
|---|---|---|---|
| D51a | F | §4.3's closed form `(bars − 119) − [119×breaks + excluded]` is not an identity. A segment of `n < 120` bars contributes zero windows but is charged `n − 119`, and the negative is absorbed silently. Origin 2022-02 holds an 80-bar segment; seven origins are understated by 39 … 137 windows | Count windows **segment-wise**, `Σ max(0, nᵢ − 119)`. Keep the closed form only as an upper bound, asserted as `closed_form ≤ measured` — the other direction would mean counting windows that do not exist |
| D51b | F | Test-block survival was accounted with **training** semantics, requiring the whole 120-bar window inside the block. That returns 601 of 720 on a *clean* block — a 16.5% phantom loss that §9.2 would absorb into the block-coverage covariate as noise | Test blocks hold **720** forecast origins. §8.3 already licenses the 96-bar lookback crossing backwards; only a break inside the spanned 120 bars disqualifies a start. Measured: 74 of 90 cells clean, worst 439/720 |
| D51c | C | The `H == L` count was never measured and §4.3 assumed it additive to the zero-volume count | Measured: **3 bars**, and they are the **same 3 bars** as the zero-volume and zero-trade ones — `2019-06-07T21:00`, `2021-02-11T03:00`, `2023-03-24T12:00`. No volume ⇒ no trades ⇒ high and low never separate. Total unusable is 3, not 9 |
| D51d | I | §8.1's table claims 5-month spacing visits "12 — all of them" calendar months per block. True for `b=1` only: blocks are 30 **days**, not calendar months, so `b=2…6` visit 7 … 11 | State 12/7/11/11/11/11 against 6-month spacing's 2/2/2/3/3/2. `D26`'s conclusion is unaffected and in fact survives measurement; the "12" was an idealisation of its own algebra |

Also corrected: the largest training tensor is **70.12 MB**, not "≤ 70 MB" (`D25` rounded the wrong
way), and the training-window floor is **13,558**, not 13,558.

# Fourth pass — defects found by building the model, 2026-08-06

`D51` came from asserting the *data* accounting. **`D52` came from building the features and the
network**, which is a different surface and produced a different class of defect: two claims in this
document that are provably false about the mathematics, and two magnitudes that were written before
anything was measured.

| ID | Sev | Defect | Resolution |
|---|---|---|---|
| D52a | F | §5.1 asserts all three F2 estimators are "strictly positive once `H == L` bars are excluded", so their logs are total. **False for Rogers–Satchell.** `ln(H/C)·ln(H/O) + ln(L/C)·ln(L/O)` vanishes on any bar with no shadows at all — H equal to one of O/C, L equal to the other. Such a bar has `H > L`, carries real information, and passes the segment law: it is a marubozu. Measured: **33 of 75,091 usable bars**, and `log 0 = −∞` propagates into the K=12 rung | The claim holds for Parkinson (∝ `(ln H/L)²`) and Garman–Klass (≥ `0.114 (ln H/L)²`, since `2ln2−1 ≈ 0.386 < 0.5`); restate it for those two only. Rogers–Satchell uses `log(RS + κ)` with **κ = 1e-9 fixed**, chosen so `log κ = −20.7` lands inside the measured support (median −10.91, 0.1st pct −17.57, min −23.5) rather than as 33 out-of-support spikes that would distort the instance normalisation of every window containing one and smuggle a categorical marubozu flag into a continuous variate. Not applied to the other two: their minima are 1.16e-8 and 1.48e-8, where κ would shift the smallest values ~8% for nothing. Disclose in §4.1b |
| D52b | C | §7 states `μ_g/σ_g ≈ 0.037` on a bull window, "~35% of `R²_oos`", "≈ 0.18σ over 24 steps". Written before measurement | Measured across all 15 origins: **−0.00818 … +0.01733**, so ~2× smaller; square is **7.5%** not 35%; tilt **0.085σ** not 0.18σ. The argument survives and in one respect strengthens — `μ_g/σ_g` **changes sign** across origins, so it is not a constant a reader can subtract. §7 corrected |
| D52c | U | The training-window count in `ORIGIN_WINDOW_BUDGET.md` is measured on the **raw** usable bars, but windows are cut from the **feature** frame, and §4.3 drops the first bar of each segment because `r` is per-segment. The tensors therefore hold 0 … 13 fewer windows than the table asserts, one per segment | Both numbers are right about different frames. State the assertion target as the **feature frame** — `13,545 … 15,217` — and keep the raw-frame table as the gap-accounting document. The delta is exactly the segment count and is asserted as such |
| D52d | I | The single-batch overfit check in §16 cannot pass as written: with `dropout=0.1` active the loss floors well above zero, and a reader following it literally concludes the plumbing is broken | Run it with `dropout=0.0`. Measured: **8.26e-10** after 200 steps on 8 samples — pass. With dropout left on, 6.8e-2 |

Confirmed correct by measurement, and worth recording because each was an assumption: parameter
count is **280,472** and **identical at every rung** (§6.2's claim, exact); `MSE(c·x)/c² == MSE(x)`
holds to 5e-5 relative (`D03`); `log_mean_trade_size` equals `log_quote_volume − log_trade_count` to
the last bit, confirming F3's 2-dof claim.

# Fifth pass — defects found by building the experiment plane, 2026-08-06

`D51` came from asserting the data accounting and `D52` from building the features and the network.
**`D53` came from building the grid, the K_eff measurement and the metrics** — the three modules that
turn runs into answers — and from running the Stage 3b gate for the first time.

| ID | Sev | Defect | Resolution |
|---|---|---|---|
| D53a | F | §5.4's "stable rank of the `K × 96` window block" is a **units artefact** as literally specified. Centring alone leaves `log_quote_volume` deviations two orders of magnitude above `r` deviations, so one row dominates both the Frobenius and the spectral norm. Measured at origin 1: **1.00 / 1.00 / 1.16 / 1.65** across the four rungs — "one effective direction" everywhere, which is a statement about units and not about data | Standardise each channel **within its window** first, which is exactly what `use_norm=True` does before the embedding. The statistic then has a closed form: `K / λ₁` of the within-window correlation matrix, bounded in `[1, K]` and commensurable with the contemporaneous PR. Measured: 1.00 / 2.36 / 2.70 / 2.17 |
| D53b | F | §5.4's "PR of the `K·L × K·L` **covariance** spectrum" is not monotone in K and is uninterpretable. Measured at origin 1: **92.1 / 21.9 / 37.3 / 15.5** — the collapse from K=1 to K=4 is entirely the arrival of `log_quote_volume` and says nothing about dimensionality | Use the **correlation** spectrum, for the same reason `contemporaneous_pr` does. Its ceiling is `K·L`, so it is reported as a fraction of that ceiling and is **not** on the contemporaneous PR's scale. It is nevertheless the only measure here that sees genuine cross-lag structure; the stable rank sees cross-*variate* structure inside a window. Say which is which in §4.1b |
| D53c | F | Sharding the **pending** list races. Two workers launched together compute their partition at slightly different moments — one finishes a run while the other is still building features — and the partitions stop being complementary: some groups owned by both, some by neither | Shard the **full manifest**, then subtract what is complete. `execute` skips completed cells anyway, so the filter costs nothing and the partition is deterministic |
| D53d | C | The wild cluster bootstrap returned a literal **p = 0**. `mean(t* ≤ t_obs)` has no floor, and no finite bootstrap can support that as a probability | `(1 + count)/(1 + B)` (Davison & Hinkley 1997) — the observed statistic belongs to its own reference distribution. At B = 99,999 the floor is 1e-5, and at G = 15 Rademacher's own granularity bounds it near 3e-5 regardless |
| D53e | C | §10.2's total **double-counts 48 runs**. The horizon sweep's H=24 slice at seeds 42–44 carries the *same* `run_id` as the corresponding main-grid cells, and `run_id` is the identity of a run (§10.4), so 582 nominal iTransformer cells are **534 real runs**. Executing a shared cell twice would mean two files racing for one path | Deduplicate in the manifest. The grid total is **789**, not 837: 534 iTransformer + 195 baselines + 60 ridge. The sweep is still "4 × 4 × 4 × 3 = 192 cells" in the paper; 144 of them are new work |
| D53f | **U** | **The Stage 3b gate does not pass.** Measured PR at K=8 on the pre-first-origin span is **4.393 < 5.0**, and per-origin PR at **K=12 (3.98) is *lower* than at K=8 (4.27)** — §5.2 expected ~6.5 and ~7. `corr(K, K_eff) = 0.828`, not the ≈0.97 §9.1 anticipated | `D48`'s action is **disclosure, not a re-cut**, and it is taken: the grid proceeds unchanged and §4.1b reports the divergence. Two consequences are substantive rather than procedural. First, the K=12 rung is **more** redundant than designed — §5.2's deliberate-redundancy control is stronger evidence than expected, not weaker. Second, at 0.828 the K-versus-K_eff horse race is **more** identifiable than §9.1 feared, so `D32`'s non-nested comparison is worth running rather than a formality. Fix the hypothesis to the measurement, never the reverse (§5.4) |

Also measured, and recorded because it bears on the Stage 5 gate: at origin 1 with a **single** seed,
validation MSE is 0.469075 at K=1 against 0.467904 at K=8 — K=8 ahead by 0.25%, Clark–West
`S* = +0.728, p = 0.233`. That is not the gate, which averages three seeds, but it is the first
evidence about it and it points the same way as §10.3's `R²_oos = −0.0183`. If the real gate fails,
§8.5's instruction is to reposition the title to the descriptive variant **now, not in week nine**.

# Sixth pass — the Kaggle deployment surface, 2026-08-07

`D51` came from asserting the data accounting, `D52` from building the features and the network,
`D53` from building the experiment plane. **`D54` came from asking what the notebook actually needs
in order to run on Kaggle** — the deployment surface, which is precisely what the unrun
Kaggle/execution lens would have examined, and which no amount of re-reading `src/` would surface,
because every defect in it is invisible on a machine that happens to have the repository.

| ID | Sev | Defect | Resolution |
|---|---|---|---|
| D54a | C | The launcher globbed for `src/itransformer_btc/__init__.py` and pushed the hit onto `sys.path`, so it could not run unless the **repository was also uploaded as a second Kaggle Dataset** and kept in step with the notebook by hand — two artifacts that must agree, with nothing checking that they do | The notebook carries the package in twelve `%%writefile` cells and materialises it before importing, then asserts `itransformer_btc.__file__` lives under its own working directory. Files rather than in-cell definitions because the two GPU workers are **subprocesses** and inherit no namespace. §15 |
| D54b | C | `_git_sha()` is `"unknown"` on Kaggle — no git repository exists there — so §12's three-part contract lost its **code** half at exactly the place the grid executes, while §12 read as though it had not | `code_sha256`: the hash of the package source, line endings normalised so a CRLF checkout and an LF materialisation of one logic give one digest. Recorded beside `git_sha` in every `meta/*.json` and in `paper_numbers.json`. §12 |
| D54c | C | `_input_sha256()` read the hard-coded `data/raw/BTCUSDT_1h_report.json`, absent on Kaggle: the artifact arrives under `/kaggle/input/<slug>/` and §10.5 forbids hard-coding that slug. The **input** digest therefore also logged `"unknown"`, and §12's rule that two vintages may not share a table became unenforceable | `ITBTC_PARQUET`, set by the notebook, by `launch_workers` per child, and by the worker CLI from its own `--parquet`. The Stage 1 report beside the artifact is preferred; hashing the parquet is the fallback; `input_sha256_source` records which. §12 |
| D54d | I | Nothing would police the second copy of ~4,000 lines the fix creates — the drift failure this register exists to prevent | The notebook is **generated** by `tools/build_notebook.py`; `tests/test_notebook_sync.py` asserts it byte-identical to `src/` and `--check` fails the suite on drift. §15, §16 |
| D54e | F | **The evaluation cells crash on a partial session, which is the normal session.** A grid stopped at run 200 of 534 leaves an unbalanced panel, and §9.1's estimators refuse one *by design* — `amplification` raises rather than compare K=1 at eleven origins against K=8 at ten, and RQ1's `wide[4] - wide[8]` broadcast-errors first. Simulated at the real two-shard stop shape: K=1 complete at 11 origins, K=4/8/12 at 10. The estimators are right; **where the exception lands is not** — it marks the Kaggle version failed at the exact moment its output is the only thing worth keeping | The estimators stay strict and the notebook does not call them until the panel exists: cells RQ1/RQ2/RQ3 and the `paper_numbers.json` write are gated on `GRID_COMPLETE`, print what remains and how to resume, and exit cleanly. A half-panel β₁ is a **different estimand**, not a noisier one, so partial evaluation is never the fallback |
| D54f | C | **The budget guard bounds the worker, not the session.** `BudgetGuard.deadline` is set from `time.perf_counter()` inside each worker, but Kaggle's 12 h wall runs from cell 0 — so the prelude (data, K_eff, invariants, and the *twelve pilot training runs*, ~20–25 min) sat outside the budget entirely, and the two clocks drifted apart by however long it took | The notebook stamps `SESSION_T0` in cell 0 and passes `budget_h = 11.0 − elapsed` to `launch_workers`, so the guard bounds what §10.1 actually limits. Hitting the wall interactively loses `/kaggle/working` entirely, so this margin is not somewhere to be approximate |

Measured 2026-08-07 with the repository absent from `sys.path`: every §4.1/§5.4/§6.2 figure
reproduced from the materialised copy — 75,094 bars, 3 unusable, window budget exact at all fifteen
origins, gate PR **4.393**, `corr(K, K_eff)` **0.828**, **280,472** parameters, `μ_g/σ_g` spanning
−0.00818 … +0.01733 — and one worker subprocess wrote a `meta` carrying
`input_sha256 = 8270a84b07c2923b…` from source `"report"`, matching §4.1's pinned digest.

# Seventh pass — defects found by *running the grid to completion*, 2026-08-08

`D51`–`D53` came from building the pipeline and `D54` from asking what Kaggle needs. **`D55`–`D58`
came from the first full 534-run session** — the only lens that reads the code *after* the answers
exist rather than before, and it found two defects that are invisible until the results have a
particular shape.

| ID | Sev | Defect | Resolution |
|---|---|---|---|
| **D55** | F | `DecayResult.b_star()` inferred its schema from its rows, so when `decay`'s `R2_oos > 0` guard excluded **every** origin it returned a frame with no columns and the RQ3 cell's `bs["b_star"]` raised `ColumnNotFoundError` — marking the twelve-hour version failed at the moment its grid output was the only thing worth keeping. The guard firing is the **expected** outcome under non-positive skill, not an edge case: §10.3's first run already returned `R2_oos = -0.0183` and the grid returned it at all fifteen origins. `D54e` gates on grid *completeness*, a different failure | `B_STAR_SCHEMA` declares the four columns. The RQ3 cell branches: an empty table means the estimand is **undefined** — there is no edge to lose a proportion of — which is *not* the right-censored "no decay detected within 180 days" §3 pre-registers, and reporting both in one wording would claim skill the grid never found. Log-rank likewise refuses to print `chi2=nan` where the statistic is 0/0. **Closed 2026-08-08** |
| **D56** | F | §7 calls DLinear and PatchTST "not optional" and §10.2 budgets 255 baseline runs, but **no baseline model exists in `src/`** and the manifest (534 = `main 300 + uniform 75 + fresh 15 + horizon 144`) never contained one. §10.2's 789 was never executable. `metrics.dm_nonnested()` sits waiting for input that has never existed, so Table 6 has no inputs and the paper's central architectural comparison has no data | `src/itransformer_btc/baselines.py` + manifest keys: **ridge (K=1,4,8,12) 60, DLinear (K=8) 45, PatchTST (K=8) 45**, manifest **534 → 684**, arms ordered so the ladder completes first. `write_artifacts` was **generalised to a protocol, never copied** — §12's schema keeps exactly one definition, and the iTransformer `meta/*.json` was verified byte-identical across the change but for `code_sha256` and `wall_time_s`. `D45` is asserted per baseline run against its main-grid comparator and is **fatal**, not skipped. Three consequences are new and written into §7: the channel-independent baselines' all-channel objective and what their K label therefore means, DLinear's internal centred moving average against §5.3, and PatchTST at ~16× iTransformer's wall time. ARIMA, LSTM, naive-persist and seasonal-naive are **deferred with a written reason**, not silently unbuilt. **Closed 2026-08-10** |
| **D57** | U | §10.3 estimated 60–100 s per run on a T4 and 10–20 h for the grid. Measured: **~30 s** and **2.31 h** for 534 runs on two T4s. The regime was right; the arithmetic on it was 2–3× pessimistic per run and 4–8× overall, so the weekly quota was never the binding constraint every §10.2 trade-off was made against | §10.3 carries the measured numbers, §10.5's resume argument reads ~30 s. The slack is what makes a second granularity affordable and what made `D58` possible. **Closed 2026-08-08** |
| **D58** | C | §15 and §10.3 described twelve `%%writefile` cells materialising a package, imported by **two GPU subprocesses**. That form existed for one reason — a subprocess reaches code only from disk — and `D57` dissolved it. Keeping the description would have left the governing document contradicting the artifact, which is worse than the defects it catches | Notebook flattened to **definition cells** in one kernel namespace; grid runs in-kernel and sequential. `D54a`'s *conclusion* (no repository Dataset) stands; its *mechanism* is superseded. `launch_workers` is retained and tested for the checkout path and for a 1-minute grid, where sequential will not fit. `code_sha256` is pinned by the generator, with the honesty cost stated in §12. **Closed 2026-08-08** |

# Eighth pass — the first defect found by *running the flattened notebook*, 2026-08-11

`D54` came from asking what Kaggle needs and `D55`–`D58` from running the grid. **`D59` came from
running the notebook `D58` produced**, and it is the flattening's own failure mode: a defect that
every check in this repository answers correctly and that only the interpreter can see.

| ID | Sev | Defect | Resolution |
|---|---|---|---|
| **D59** | F | `stage5_pilot` reached the gate statistic through `from itransformer_btc import metrics` — an import binding a **module object**. Flattening drops intra-package imports (§15) and **no cell defines a module object**, so `metrics.clark_west_test` was a `NameError`. Every existing check passed, because each asks a question this defect answers correctly: the cell parses, it compiles, it equals `src/` byte for byte, it names no surviving `itransformer_btc`, and `test_definition_cells_execute_in_one_namespace` executes the definitions without ever entering that function body. It surfaced at **365 s on Kaggle** — past Stage 2, Stage 3b, Stage 4 and all twelve Stage 5 pilot runs — and marked the version failed. `runner.py` already carried the rule in a comment, for the baseline configs (`D56`); `metrics` was simply missed | Import the **name**: `from itransformer_btc.metrics import clark_west_test`. Three defences, because a comment is not a check. (1) `flatten_module_source` computes the names each dropped import would have bound to a module object and **refuses to emit a cell that still reads one**, carrying the fix in the message. (2) `_intra_package_import` now also matches the **relative** form `from . import x`, which it did not — that form survived flattening and would raise `ImportError` on the first run rather than being quietly equivalent. (3) `tests/test_notebook_sync.py` walks **every** cell with `symtable` against the executed namespace and fails on any global that is read and never bound — a symbol-table question rather than a spelling one, so a local named `metrics` does not cry wolf. It finds exactly one permitted hole, `__file__` inside `code_sha256`, unreachable behind the pinned `CODE_SHA256_OVERRIDE` and listed by name rather than allowed everywhere. **Closed 2026-08-11** |

**What this says about the format, stated because it is the cost side of `D58`'s trade.** Flattening
is subtractive over two declared categories, and *that is still true*: the defect was not a rewrite
but a deletion whose consequence lived elsewhere in the file. A module-object import is the one
construct whose meaning the deletion changes rather than preserves, so `src/` may reach a sibling
**only by name**. The rule is now enforced at generation instead of remembered.

**The twelve pilot runs may or may not survive, and the difference is Kaggle's, not the code's.**
They are ordinary main-grid `run_id`s and their artifacts were written before the exception, so
§10.5's resume finds them complete **if** the failed version published `/kaggle/working` as its
output. Whether a version that ends in a papermill error publishes anything is not established here
and must not be assumed; if it did not, the loss is those twelve runs, about six minutes.

# Ninth pass — the defects found by *having the answers*, 2026-08-20

`D51`–`D53` came from building the pipeline, `D54` from asking what Kaggle needs, `D55`–`D58` from
running the grid and `D59` from running the notebook `D58` produced. **`D60` came from reading the
grid's output against the document that specified it** — the one lens that cannot run until the
answers exist. It found no coding defect. Every entry below is the document being wrong about the
world, or silent where the world had spoken.

**The governing document was seven hours stale and the gap was not visible from inside it.**
`CLAUDE.md` was last written 2026-08-11 09:43 local (`77cbb5b`, closing `D59`);
`paper_numbers.json` was written 2026-08-11T09:51:37Z, i.e. 16:51 local. Between those two stamps
the grid ran to completion and answered all three research questions, and nothing in the repository
required the document to notice. That is the failure mode this section exists for, and it is why
`D60a` is fatal rather than clerical: §8.5 pre-registered an action on the Stage 5 result, the
result arrived, and the action did not happen for nine days.

| ID | Sev | Defect | Resolution |
|---|---|---|---|
| **D60a** | F | **The Stage 5 gate failed and the document did not say so.** Clark–West K=1 vs K=8 on origin 1's validation sub-block returned `S* = +0.8759, p = 0.1906` one-sided, `T = 1845, h = 24` — a failure to reject at α = 0.05. §8.5 pre-registers exactly one action on that outcome: *"reposition the title to the descriptive variant now, not in week nine."* The session log printed that sentence back verbatim. §1 still carried the comparative title, §5.3's K=16 arm still read as live, and §13.2 still listed the pilot as a selection event that might or might not have fired | §1 title repositioned, old title retained with its reason. §8.5 carries a gate-outcome table for all three stages. §5.3 records the K=16 arm as **not run, clause 1 failed**. §13.2 adds the disclosure. **Closed 2026-08-20** |
| **D60b** | F | **RQ1, RQ2 and RQ3 all have answers and none was in the document.** RQ1: `R²_oos` = −0.0205 / −0.0187 / −0.0180 / −0.0186 at K = 1/4/8/12, ΔMSE 4→8 = +0.000636, 8→12 = **−0.000437**, TOST vs ±0.000159 → **not equivalent** (the 8→12 rung is worse, not flat); the J-test rejects the K explanation (`t = +3.293, p = 0.0011`) and does not reject K_eff (`t = −0.348, p = 0.7281`). RQ2: β₁ = **+0.000256**, wrong sign, WCR p = 0.7381, and **inside** the MDE of −0.000920, which is §9.2 requirement 6's pre-registered trigger. RQ3: `decay_panel` **empty**, all 15 origins excluded on `R²_oos ≤ 0`, `b*` **undefined** at every τ, log-rank unavailable in both arms | §3 carries a measured-answers table beside the hypotheses. §9.1 states that the `R²_oos ≤ 0` guard is the only case, not an edge case, and fixes the RQ3 wording — *"the estimand is undefined under non-positive out-of-sample skill"*, never *"no decay detected within 180 days"*. §9.2 requirement 6 records the MDE beside the estimate. §10.3 records that the first run's −0.0183 was the whole distribution. **Closed 2026-08-20** |
| **D60c** | C | **The baseline ordering inverts the paper's premise and appeared nowhere.** Mean `R²_oos` over every test block: ridge **−0.000568**, PatchTST −0.016312, iTransformer-K8 −0.017993, DLinear −0.026248. Every model loses to Naive-RW, and the *linear* model loses by ~30× less than any deep one — at ridge's selected α it shrinks close enough to the training mean that it is nearly the baseline itself. `D17` added ridge to ask "is a transformer needed at all?"; the measured answer is **no**, and §7 recorded only a single-cell validation hint pointing that way | §7 states the ordering and the reading. §13.2 makes it a mandatory disclosure. It is the frame every other result must be read inside, so it belongs in Results before RQ1, not in a baseline table at the back |
| **D60d** | U | **§7's PatchTST projection was falsified and §10.3's per-run figure moved again.** §7 scaled a CPU measurement (1810 s, ~16× iTransformer per run) onto a T4 and predicted ~6 h for the PatchTST arm alone, a 684-run manifest near the 11 h budget, and "two sessions is now the expected case". Measured: PatchTST **95.6 s** mean, **~2.6×** iTransformer, arm **1.19 h**; whole manifest **6.52 h, single `cuda:0`, one session**, mean 35.0 s per run, 684 complete / 0 skipped / 0 failed | §7 and §10.3 carry the measured per-arm means. The transferable rule, which is what was actually wrong: **a throughput ratio measured on one device does not scale to another** — PatchTST's B×N = 256 folding is a penalty a 6-thread CPU pays in full and a T4 largely absorbs |
| **D60e** | C | **`D59` left open whether a papermill-failed Kaggle version publishes its output, and the answer was already on disk.** §14 hedged that the twelve pilot runs "may or may not survive" | It publishes. The grid session opened `already complete: 12  pending: 672` and skipped them. §10.5 states it and drops the hedge; the loss from `D59` was **zero runs**, not twelve |
| **D60f** | U | **§12 and §15 point at the wrong directory.** Repo-root `artifacts/` holds **one** run from a 2026-08-06 CPU smoke test. The grid output is at `notebooks/outputs/artifacts/`, committed at `29c0646`. Six panel parquets exist there and are documented nowhere; `logs/`, `tables/` and `figures/` do not exist in either location, though §15's layout implies all three | §15 carries the real inventory and marks repo-root `artifacts/` as stale. §12 names the real `paper_numbers.json`. §15's one-line description of the register is corrected: it holds D01–D50 and D54 only, and **D51–D53 and D55–D60 live in §14 alone** |
| **D60g** | U | **Table 6, Table 8, Figure 5 and Figure 7 have no inputs and were never run**, while §13.4 promises eight tables and seven figures. The session log contains zero Diebold–Mariano, Romano–Wolf or Model Confidence Set lines; no economic evaluation ran; attention weights were never persisted. `metrics.dm_nonnested()` sits waiting for input a second time — `D56` fixed the missing *models*, not the missing *call* | §13.4 carries a state table naming each deliverable, its state and what is missing. Two are load-bearing: **Table 6** is one aggregation pass from existing, since `preds/` holds the target channel for all 684 runs; **Figure 5** is not, because attention maps were not saved and need a re-run |
| **D60h** | I | Three unscoped or superseded numbers. (1) §6.2's "parameter count identical across rungs" is true only *within* a horizon — measured 277,505 / 277,763 / **280,472** / 299,048 at H = 1/3/24/168, because the projection is `Linear(d_model → H)`. (2) `D52a` gives Rogers–Satchell "min −23.5, 0.1st pct −17.57"; measured post-κ the min is **−20.723**, which is `log 1e-9`, the floor itself, and q0.1% is **−19.805**. Both are true, of the pre-κ and post-κ frames respectively, and the document says neither. (3) `input_sha256_source` is **`file-digest`** on all 684 runs, not `"report"`, because the Kaggle Dataset carries only the parquet | §6.2 scopes the claim to a fixed horizon. This row records the RS frames. §12 states that the file-digest path is the operative one on Kaggle and that the digest is nonetheless §4.1's pinned vintage, `8270a84b07c2923b…`, under a single `code_sha256 ee63120991695c6c…` across all 684 metas |
| **D60i** | F | **The falsification arm's headline number is a units artefact.** The notebook reports `mean(aged − fresh) = −0.053341` over 45 (origin, block) cells as raw scaler-space MSE. The two arms are fitted at origins 90 days apart and therefore carry **different `σ_g`** — 0.009151 against 0.007297 at origin 1 — so the comparison is between numbers in different units. The matching naive baselines differ by **−0.053196**, i.e. **~99.7% of the reported gap is scaler drift**, and the sign reads backwards: taken at face value it says the aged model beat the fresh one, which is the opposite of what the scale-free metric says. §9.1 already forbids exactly this by requiring RelMSE "to control for period difficulty"; the falsification arm was simply never brought under that rule | Report the gap on the **scale-free** metric: `mean(aged − fresh) RelMSE = **+0.000828**`, the fresh model better by 0.083% of naive MSE — H2's predicted direction, at a magnitude that **flips sign at 7 of 15 origins**. The honest verdict is that the arm is **uninformative at this effect size**, matching what the MDE says about β₁, which is expected since the arm identifies the same quantity. §9.2 corrected. **The raw-MSE figure must not appear in the manuscript.** Any cross-origin model comparison is on RelMSE or `R²_oos`, never on scaler-space MSE — that is the general rule this defect buys |

**What this pass says about the document, stated because it is the cost side of pre-registration.**
Nothing here is a bug in `src/`; the code did what §9.1 and §8.5 told it to, printed the gate
failure in the words §8.5 uses, and named every excluded origin rather than dropping it. Seven of
the nine entries are the document failing to *absorb* a result it had already commissioned. The one
substantive analysis error, `D60i`, is a metric computed outside the rule §9.1 states — which is
the same shape: a rule that exists and was not applied at one site.

# Tenth pass — the defect found by *running the test suite*, 2026-08-20

Found while verifying `D60`, and recorded separately because it has nothing to do with the results
and everything to do with the artifact that produced them.

| ID | Sev | Defect | Resolution |
|---|---|---|---|
| **D61** | C | **The committed notebook is the Kaggle *export*, not the generator's output, and the suite has been red on `main` since `9926acd` ("Kaggle Notebook \| iTransformer.ipynb \| Version 4").** It carries `papermill` metadata, 26 execution counts and 12 cells of committed output. `tests/test_notebook_sync.py` fails twice: `test_notebook_is_not_stale` (`tools/build_notebook.py --check` reports stale) and `test_notebook_is_valid_nbformat_with_gpu_metadata`, whose assertion reads `cell["outputs"] == [], "committed outputs go stale; strip them"`. §15 says the notebook is **"GENERATED, never hand-edited"** and §16 makes regeneration part of editing `src/`; re-uploading Kaggle's copy over it is the same class of drift, arriving from the other direction. **The code is not affected and this must not be read as a code defect:** `code_sha256()` over the current `src/` returns `ee63120991695c6c…`, byte-equal to the notebook's pinned `CODE_SHA256_OVERRIDE` and to the digest stamped on all 684 metas, and every per-cell byte-equality test still passes. `src/` has not moved since the grid ran | **Regenerate**: `python tools/build_notebook.py`, then commit. The outputs being discarded are not evidence being lost — `notebooks/logs-iTransformer.txt` (committed separately at `8bc0112`) is the same console stream, complete, and is what `D60` was derived from. **Not done in the `D60` pass**, because it deletes committed content and that is the repository owner's call, not a documentation edit's. Until it is done, `python -m pytest tests/ -q` reads **2 failed, 84 passed**, and a suite that is expected to be red is a suite nobody reads |

**Why this is worth an ID rather than a fix-and-forget.** The two artifacts that must agree — `src/`
and the notebook — *do* agree on every line of code; the generator and its tests proved that. What
diverged is the notebook's *shape*, and the check that caught it is the same one `D54d` added to
police exactly this. It worked. What failed is that its failure was left standing for nine days, in
the same window `D60` documents, for the same reason.

# Eleventh pass — the deliverables the grid never produced, 2026-08-21

`D60` came from reading the grid's output against the document. **`D62` came from generating the
deliverables that reading said were missing** — and it is the first pass whose entries are mostly
*absences* rather than contradictions. Nothing here is a defect in the 684 runs; every number they
produced stands. What was missing was the code that turns them into a paper, and the arms that would
let the paper's null survive a referee.

| ID | Sev | Defect | Resolution |
|---|---|---|---|
| **D62a** | U | **Tables 2, 6, 7, 8 and Figures 2b, 3, 4, 5, 6, 7 were promised by §13.4 and never generated**, four of them with no inputs at all (`D60g`). `metrics.dm_test()` and `directional_accuracy()` existed, were tested, and were **never called on the 684-run panel**; Romano–Wolf and the Model Confidence Set did not exist in code at all; §4.5's efficiency tests had no implementation; `keff` had no rolling variant for Figure 2b | `src/itransformer_btc/report.py` + `tools/build_report.py` assemble **`paper/paper_numbers.json`** — the manuscript's single source — from the grid's immutable output plus every analysis pass it never ran, naming the grid file by sha256 so the two cannot silently diverge. Nine tables and six figures render from that file and nothing else. `keff.rolling_pr` / `rolling_ols_r2` added for Figure 2b, both **descriptive only** (§5.4). **Closed 2026-08-21** |
| **D62b** | I | §6.2 pre-registers *"one robustness run at K=12 with larger `d_ff`, so a flat 8→12 rung cannot be read as an under-tuning artefact"* and **it was never built** — `ARM_MODEL_TAG` carried seven arms and none of them this one | `capacity` arm, tag `itrc`, `d_ff = 512` at K=12, 15 origins × 5 seeds = **75 runs**. `d_ff` *is* a config field, so the meta records the widening — correctly, since it is the only thing that differs from the rung it is compared against |
| **D62c** | I | The null's most obvious attack — *"you under-trained"* — had no answer, and the document implied the epoch cap was the constraint. **Measured over all 684 metas it is not: 0 of 444 iTransformer runs reach the 30-epoch cap** (mean 10.49, max 26). The binding constraint is the **LR schedule** — `lr_halve_every = 4` puts the rate at ≈1.6e-6 by epoch 26 — so raising `max_epochs` alone is a provable no-op | `longsched` arm, tag `itrl`, `lr_halve_every = 8`, `max_epochs = 60`, `patience = 10`, K ∈ {1, 8}, 15 origins × 3 seeds = **90 runs**. `TrainSchedule` carries the defaults and reproduces the grid exactly; `LongScheduleConfig` is a plain subclass adding **no field**, so `asdict(cfg)` is unchanged and the schedule is recorded under its own `meta` key. **Exploratory, declared under §13.2's confirmatory/exploratory rule, never mixed into RQ1–RQ3** |
| **D62d** | U | Attention weights were never persisted — `VariateAttention.forward` computed `softmax(scores)` and discarded it — so Figure 5 had no input and §13.2's interpretability claim rested on `A_attn` alone | Runtime `capture` attribute, a plain `nn.Module` attribute and **never** an `ITransformerConfig` field: `write_artifacts` records `asdict(cfg)`, so a field there would change bytes all 684 metas already carry. The branch consumes no RNG, so a captured run is bit-identical to an uncaptured one — which makes the `attention` arm (tag `itra`, 45 runs) a **reproducibility check of the whole grid** as well as Figure 5's input |
| **D62e** | F | **`D60i`'s corrected falsification figure existed only in `CLAUDE.md` prose.** No cell computed it, so §12's regenerability contract did not cover the paper's own correction — the one number the document had already caught being wrong | `metrics.falsification_relmse`, carried in `paper_numbers.json`. It reproduces **+0.000828** independently, confirming the correction |
| **D62f** | C | **`D60i` states the falsification gap "flips sign at 7 of 15 origins". Measured, it is 6.** The mean matches exactly, so the computation is not in dispute; the count is. Two origins sit within 5e-5 of zero — `2022-07` at **+0.000046** and `2024-08` at **−0.000012** — so the count is sensitive to whether RelMSE is built from seed-averaged MSEs or averaged per cell, and either answer is defensible | Report **6 of 15**, and report beside it that two origins are within 5e-5 of zero. The honest statement is that **the sign is not stable**, not that it flips at exactly *k* origins. `D60i`'s substantive verdict — the arm is uninformative at this effect size — is unchanged and slightly strengthened |
| **D62g** | U | **The grid and the reporting code are now two `code_sha256` vintages** — the 684 runs at `ee63120991695c6c…`, everything written after `D62` at `fec3e8b4af4e453a…`. §12 forbids numbers from different vintages sharing a table, and nothing said what to do when the *analysis* moves while the *runs* do not | State it. The vintage that matters for a number is the vintage of the **runs that produced it**, which is unchanged; the reporting code is a reader of those runs, not a producer of them. The `D62` robustness arms are the exception — they *are* new runs at the new vintage — and that is precisely why they get their own table rather than a column in Table 4 |
| **D62h** | I | The obvious drift guard for the second generated artifact — exact string equality on `paper_numbers.json` — is **measurably the wrong instrument**. polars aggregates `group_by` in parallel, so summation order varies between runs and a mean over float32 cells lands on a different eighth significant digit. Two consecutive builds in one process differ on **28 of ~8,000 numbers** at a relative ~1e-7, below the precision the underlying float32 columns carry at all | `tools/build_report.py --check` compares **structure exactly and floats within 1e-6**, and reports the *path* of the first real difference rather than a bare "stale". Two NaNs compare equal, because ridge is a solve and its one seed gives an undefined seed std — correctly |

**What generating the deliverables actually surfaced.** Three results that no amount of re-reading
would have produced, and each sharpens the paper rather than changing it:

- **The Model Confidence Set at both 90% and 75% contains Naive-RW and all four ridge rungs, and no
  deep model.** Rank order by mean loss: Naive-RW, `rdg-K4`, `rdg-K8`, `rdg-K12`, `rdg-K1`, then
  every iTransformer arm, PatchTST and DLinear. This is `D60c`'s ordering promoted from a table
  footnote to a formal statement about which models are indistinguishable from the best.
- **`D35`'s multiplicity argument is vindicated by its own numbers.** Against Naive-RW, raw
  Clark–West rejects at α = 0.05 for **8 of 11** models; after Romano–Wolf stepdown across all 66
  pairs, **none** does — every adjusted p is ≥ 0.336. Across the whole matrix the correction removes
  26 of 57 rejections. A paper reporting the raw column would have claimed eight results it does not
  have.
- **Clark–West is positive where `R²_oos` is negative, and both are true.** `t` is +2.198 for
  Naive-RW versus `itr-K8` while that arm's `R²_oos` is −0.0180. That is the statistic behaving as
  designed: it credits the larger model for the estimation noise the null imposes. The honest joint
  reading is that **any population-level edge the added variates carry is smaller than the estimation
  error required to exploit it** — a sharper sentence than either number alone.

**And one number that needs its context stated before a reader mis-reads it.** The §13.5 sign
strategy on `itr-K8` returns **+20.6%** net of the 0.04% fee and 0.02% slippage, mean over fifteen
origins, at an annualised Sharpe of **+0.377** — and buy-and-hold over the same spans returns
**+29.0%**. The strategy underperforms holding; its DSR is **0.173**, so the Sharpe is not
distinguishable from the best of the configurations evaluated on that span; and at the top of the
pre-registered slippage band it falls to +12.5% and a Sharpe of +0.104. Ridge is worse still, at
−1.02. **A positive P&L under a negative `R²_oos` is not a contradiction**: MSE and directional P&L
are different objectives, and a sample dominated by BTC's 2020–2026 rise pays a mostly-long position
for the drift rather than for the forecast. Report the three numbers together — strategy, hold, DSR —
or the first alone will be read as skill.

Absorbing a contradiction silently is the exact failure this register exists to prevent.

---

# Twelfth pass — the deliverable changed what the notebook is for (2026-08-27)

The lens: **reading the notebook as an examiner would**, rather than as a machine that executes it.
Neither earlier lens asks that question. `D58` optimised the notebook for a Kaggle session nobody
watches, and it was right to; §1's deliverable then made the notebook something a thesis examiner
opens alongside the manuscript, and the shape that served the first purpose actively defeats the
second.

| ID | Sev | Defect | Resolution | § |
|---|---|---|---|---|
| D63 | I | Eighteen module-sized dump cells — `metrics.py` alone ~1.400 lines in one cell — so the notebook reads as a `.py` file split at module boundaries. §15 optimised it for a launcher nobody opens, while §1's deliverable makes it an artefact an examiner reads | `SECTION_MAP`: 137 cells cut by logical group, each preceded by an HTML markdown heading naming what it does and which rule it enforces. Identity in `cell.metadata`, byte-equality moved from per-cell to per-module | 15, 16 |
| D64 | I | Three of §7's four deferred baselines cost minutes and were never built, leaving rows that read as unfinished work — and *no deep model beats Naive-RW* asserted without testing the deep model the crypto literature reaches for first | LSTM (`lstm`, 45), naive-persist (`npst`, 15), seasonal-naive (`nsea`, 15). Manifest 894 → **969**. ARIMA stays deferred with a written reason | 7, 10.2 |

## `D63` — what the segmentation contract actually guarantees

Three properties, and the third is the one that made the first two cheap.

**Cells are contiguous, exhaustive line ranges.** `split_module_cells` cuts the flattened module at
anchors named in `SECTION_MAP`, each anchor a module-level definition. Every line lands in exactly one
cell, in order. Nothing is dropped, nothing is duplicated, and the function asserts that itself before
returning.

**Identity lives in `cell.metadata`, not in a banner comment.** The old `# ═══ metrics.py ═══` header
was a line the cell carried that its module did not, which was harmless while a cell *was* a module
and would have forced a strip step the moment cells became slices. The metadata tag costs nothing,
renders nothing, and keeps every cell body a byte-exact slice — so
`tests/test_notebook_sync.py::test_module_cells_concatenate_to_flattened_source` compares raw
concatenation against `flatten_module_source` with no normalisation for a reader to distrust.

**Exactly one line is additive, and it is declared.** Every cell after a module's first opens with
`from __future__ import annotations`. This is not decoration: a `__future__` import is scoped to its
code unit, and Kaggle runs Python 3.11. Without it `RidgeConfig.build`, whose return annotation names
a `RidgeForecaster` defined one cell later, raises `NameError` **at class-definition time** — an error
that would have surfaced in a Kaggle session and nowhere in local testing on 3.14, where PEP 649 makes
annotations lazy by default. This is the same failure shape as `D59`: a name, unbound by the
flattening, that no cheap check would have reached. Finding it before shipping is what the property
above bought.

The guarantee moved from per-cell to per-module and weakened nothing. A line lost between two cells
fails the concatenation check exactly as a changed line would.

**One defect surfaced while wiring the tests, and it is worth its own paragraph.** `tests/` loads the
generator with `importlib.util.spec_from_file_location` and never registered it in `sys.modules`.
Harmless until the generator grew a frozen dataclass: `dataclasses` resolves a field annotation
through `sys.modules[cls.__module__]`, which was `None`, and the decorator raised
`AttributeError: 'NoneType' object has no attribute '__dict__'` — from *seven* tests at once, all of
them by way of a fixture, none of them about dataclasses. Registering the module before
`exec_module` is one line and is what a loader is supposed to do.

## `D64` — why three deferrals closed and one did not

`D56` recorded, in writing, that ARIMA, LSTM, naive-persist and seasonal-naive did not exist in
`src/`. That record was the right thing to keep: a deferral nobody states is a deferral nobody
notices. What it did not carry was a cost estimate, and once measured the estimate is what decides.

**The two naive comparators are closed forms with no parameters.** Persistence repeats the last
observed return; seasonal-naive repeats the return one daily cycle back, step for step. Both cost
microseconds and neither consumes RNG, so both take one seed for the same reason ridge does. Leaving
them deferred put two rows in the results table that read as unfinished work rather than as
measurements — the cheapest rows in the study, unbuilt.

**LSTM is the weightier one, and it is multivariate.** DLinear and PatchTST wear their K=8 label
through a published all-channel objective with shared weights: *trained on* eight channels, predicting
the target from its own history alone (`D56`). An LSTM reads all eight channels of every timestep and
emits the target, so its K=8 means what ridge's and iTransformer's mean, and `loss_channels` is
`target` rather than `all` — which makes its `best_val_mse` comparable to the ladder's where the other
two baselines' are not. `tests/test_model_plane.py` asserts the property directly, by perturbing a
non-target channel and requiring the forecast to move; without that the arm would be K=1 wearing a
K=8 label, the exact collapse `D40` exists to prevent.

It matters because the headline claim is *no model beats Naive-RW at any variate count*, and the
recurrent network is what the crypto forecasting literature this paper argues against reaches for
first. Leaving the most-cited deep model untested is a hole a reviewer finds in one pass.

**ARIMA stays deferred, with its reason now written down.** On hourly crypto log-returns, AIC order
selection lands at or near (0,0,0) — which *is* the naive baseline. The result is predictable from the
ADF and variance-ratio numbers §4.5 already reports, so the row would duplicate one the table has. That
is a prediction, not a measurement, and it is recorded as such: if it is ever to be relied on rather
than asserted, run it.

**The seasonal comparator carries one guard worth naming.** At `H = 168` the forecast asks for seven
daily cycles that a 96-bar lookback does not contain. The index is taken modulo 24, so the last cycle
repeats rather than the model indexing past the window — the only behaviour available, since a
lookback cannot supply a value it never saw. Only the H=24 arms are in the manifest, so this guards a
path the horizon sweep would otherwise reach as an `IndexError` rather than as a number.

---

## `D65` — the name an evaluation cell took from the package

`analysis_result.md` diagnosed this on 2026-08-24 and it was still on `main` three days later, so it
is recorded here rather than left in a working note.

The save cell opened with `_digest, _provenance = _input_sha256(PARQUET)`. In a flattened notebook
every module shares one kernel namespace, so that assignment replaced `report._provenance` — a
**function** `build_report` calls one cell later — with a string. The last cell of the 894-run session
died with `TypeError: 'str' object is not callable`, at 28,512 s, after the grid had run for 7.8
hours. Nothing was lost: `paper_numbers.json` and the six panel parquets were written at 28,421 s and
every `preds/`, `meta/` and `attn/` file was already safe. What was lost was the version's *status* —
a papermill error marks the run failed at the moment its output is the only thing worth keeping.

**This is `D59` from the opposite direction.** `D59` was a name the flattening *unbound*; this is a
name an evaluation cell *rebound*. Both are invisible to every parse check, both surface only when the
interpreter reaches the line, and both cost a session.

§15 already noted that `DEFAULT_PARQUET` and `HOUR_MS` collide across modules and called them harmless
**because the values are equal in both definitions**. That is a property of those two names, not a
property of the format, and nothing was checking which kind any new collision was. The generator's own
collision check compared module against module; it had no opinion about a scaffolding cell.

Resolution: the binding is `_digest_source`, and
`tests/test_notebook_sync.py::test_no_scaffolding_cell_shadows_a_package_name` refuses any scaffolding
cell that *assigns* a name a definition cell defines. Imports are excluded — re-importing `numpy as np`
binds the same module object, so it is a shadow by name and not by value. Four names are allowlisted,
each with its reason in the code: `ARTIFACTS` and `DEFAULT_PARQUET` (the notebook resolves the real
paths), `CODE_SHA256_OVERRIDE` (§12's pinned digest) and `SESSION_BUDGET_H` (`D54f`).

## `D66` — one Library cell, and imports out of the definition cells

With `D63`'s segmentation in place, every one of the 137 definition cells still opened with the
imports of the module it came from — the same twenty-odd lines, over and over, in the artefact a
reader examines.

Resolution: a single **Library** section at the top of the notebook, immediately after Setup, holding
every module-level import the package makes. It is **generated from the flattened modules**, not typed
out, so a dependency that appears in `src/` cannot go missing from it; `from X import a` and
`from X import b` are merged into one line per module rather than repeated.

Module-level imports then become the **third declared subtractive category**, beside intra-package
imports and `runner.py`'s `__main__` guard. Two consequences, both deliberate:

- **Function-local imports stay.** `report._pyplot` defers matplotlib on purpose, so the package
  imports cleanly with no plotting backend installed; hoisting that into a cell that runs at session
  start would undo the decision. The removal reads `ast.parse(...).body` and nothing deeper.
- **The reference for the equality check moved.** A module's cells now rejoin to
  `flatten_module_body(name)` rather than `flatten_module_source(name)`. Byte-exact, still — "not
  equivalent, not equal after formatting" — the reference moved, the guarantee did not.

Blank lines that *followed* a removed import block go with it; the ones that preceded it stay and
become the separator. Done positionally rather than by collapsing blank runs in the text afterwards,
because a docstring may legitimately hold three blank lines in a row and a text-level rule would eat
those too.

**One consequence worth stating rather than discovering.** `from __future__ import annotations` now
prefixes **every** definition cell, not only the continuation ones, because it too is a module-level
import and is removed with the rest. A future import above a module docstring is legal and compiles,
but it demotes the docstring to a plain expression — so `test_no_executable_package_refs`, which reads
prose out before looking for executable package references, compares against the rejoined body with
the prefix stripped rather than against the cells as they stand.

---

## `D67` — the future import belonged in one cell, and the reasoning that put it in 140 was checkable

`D63` prefixed every definition cell with `from __future__ import annotations`, reasoning that a
`__future__` import is scoped to its code unit and that splitting a module across cells therefore
loses it. That reasoning is **correct about `compile()` and wrong about the notebook.**

IPython accumulates `__future__` compiler flags across a session: `InteractiveShell.compile.flags` is
OR-ed with every future import a cell executes, and every later cell is compiled under the
accumulated set. Kaggle runs the notebook through papermill to ipykernel to `run_cell`, which is that
same path. One directive in the Library cell therefore reaches all 140 definition cells.

**Measured rather than assumed**, on IPython 9.13:

```
flags before: 16896
flags after : 16794112        # 0x1000000, CO_FUTURE_ANNOTATIONS, OR-ed in
forward-annotation cell succeeded: True
```

The second cell carried no future import of its own and still defined `def build(self) -> B` for a
`B` that did not exist. That is the whole claim, tested end to end.

**What this buys.** The last additive transformation is gone. A definition cell is now a contiguous
slice of `flatten_module_body` and **nothing else** — no prefix for a reader to wonder about, no strip
step in the equality check, no line in the notebook that is not in `src/`.

**What it costs, stated rather than discovered.** Correctness now depends on IPython's flag
accumulation instead of on each cell being self-contained. Three consequences a reader is entitled to:

- Running one definition cell *in isolation*, in a fresh kernel, compiles its annotations eagerly.
  `RidgeConfig.build` names a `RidgeForecaster` defined one cell later and would raise at
  class-definition time. The notebook's declared execution mode is **Save & Run All** (root §10.1), in
  which the Library cell always precedes it, so this is a property of misuse rather than of the
  artefact.
- Exporting the cells to a `.py` and running that file has the same exposure, and the same answer:
  `src/` is the file form, and it carries the directive per module.
- If a future IPython stopped accumulating, the notebook would break everywhere at once. The suite
  reads the flags off the Library cell and compiles every later cell under them, so that change fails
  in `test_definition_cells_execute_in_one_namespace` rather than on Kaggle.

**The transferable part is the lens, not the fix.** `D63`'s prefix was defensible reasoning about
Python semantics that was never checked against the runtime the code actually meets. The check cost
one `InteractiveShell` and four lines. Every claim in this register about *what the interpreter does*
is worth the same treatment.

---

## `D68` — half the hardware idled for 7.8 hours, and the note explaining why was out of date

The 894-run session printed, before starting:

```
NOTE: 2 GPUs visible, using cuda:0 only. Threads are not the fix —
torch.manual_seed seeds EVERY CUDA device, so two threads would clobber
each other's generator mid-run.
```

The observation was correct and the conclusion did not follow. `torch.manual_seed` does fan out to
every CUDA device — but that is a property of **how the code seeded**, not of threads. Nothing forced
the seeding to be global.

**What changed.** `set_seed` takes a device. Given one, it seeds the CPU generator and *only that
device's* CUDA generator, leaving a concurrent worker's stream untouched. The CPU generator is still
shared, and both seeding and module construction draw from it, so those two steps happen together
under a single `SEED_LOCK` — **milliseconds against a ~32 s run**, so the lock costs no measurable
parallelism. Everything after the prologue draws from the device's own generator.

The consequence is the one that matters: **a run produces the same bytes whether it ran alone or beside
another.** That is the property `D62d` demonstrated for the attention arm — 270/270 cells identical,
`max|ΔMSE| = 0` — and the property root §12 requires of every number in the manuscript. Run-level
parallelism that cost it would not be worth having.

**Single-device behaviour is unchanged by construction**, which is what keeps the 894 completed runs
reproducible: with one device in use, seeding it alone and seeding all of them set the same generator
to the same value.

### Why threads, and why not DataParallel or DDP

Root §10.3 already rejected `nn.DataParallel` with a measured reason — at batch 32 and ~280k
parameters the scatter/gather costs more than splitting saves — and that has not changed. **DDP is
worse here for a sharper reason**: it parallelises one large training job, and this grid is 969 small
ones. A process group set up and torn down per ~32 s run pays the overhead 969 times to parallelise
something that is already the unit of work. Neither is a fit for a workload whose natural grain is the
run.

Threads rather than the subprocesses `launch_workers` spawns, because §15's notebook carries the
package as definition cells in one kernel namespace: a subprocess inherits none of it and
`python -m itransformer_btc.runner` has no files to import. `launch_workers` stays for the checkout
path. The GIL is not the constraint — every run spends its time inside CUDA kernels and tensor ops that
release it.

Each worker keeps its **own** tensor cache, so two builds never race; the shared cursor and the budget
guard are touched only under one lock, so two workers cannot both slip past a deadline only one of them
had room for.

### What is not verified, stated rather than implied

**No machine this suite runs on has a CUDA device**; local torch is a CPU build. The two T4s exist only
on Kaggle. So there is no throughput measurement for this change, and `D58`'s **2.31 h** remains the
only run-level figure this project has actually taken. What *is* tested is the part that would corrupt
a grid on any hardware: that two workers sharing one cursor hand no cell to both and drop none — a
double-run means two writers on one `preds/` path (root §10.4), and a dropped one means a manifest that
never completes and a resume that never converges.

The device-scoped branch of `set_seed` is likewise dead code off a GPU. The test that can run checks
the half that the 894 completed runs depend on: that the same seed still produces the same CPU draws,
with or without a device argument.

---

## `D69` — two governing documents, and only one of them knew it

`AGENTS.md` was a full copy of `CLAUDE.md`: 2,298 lines, headed `# CLAUDE.md`, opening *"Governing
document for this repository. Read it before doing anything else"* and declaring itself authoritative
as of **2026-08-20**. It was taken before the 2026-08-24 compaction, so by 2026-08-27 the two files
disagreed on §10.3's parallelism rule, on the manifest count, and on the existence of `D63`–`D68` —
while both claimed to govern.

Nothing in the repository referenced it. An agent reading `AGENTS.md` by convention would have found
a document that looked authoritative, was internally consistent, and was a week stale — the worst of
the three possible states, because nothing about it announces itself as a copy.

This is `D54a`'s failure exactly, one level up: **two artifacts required to agree, with nothing
checking that they did.** There it was the notebook and a code Dataset; here it is two copies of
project law. The resolution is the same shape as §15's answer for the notebook — stop having two.

`AGENTS.md` is now a pointer: what governs, where the long-form evidence lives, and why the copy was
removed. Nothing is lost. `CLAUDE.md` is the compacted successor, this register holds the long-form
§14 text the compaction moved out, and the deleted lines are in git history at `2dc8706`.

**The transferable rule, and §15 already states it for directories:** *one governing document, and a
second copy of it is a defect rather than a convenience.* §15 deleted three of four subdirectory
`CLAUDE.md` files for the same reason — 55–65% overlap, with the non-overlapping part being precisely
what must not fail open. A root-level duplicate under another name is that failure with a wider blast
radius, because it does not even have a directory scope to limit when it applies.

---

## `D70` — what the second GPU bought, and the one arm that is not robustness

`D68` roughly halved the wall time of a manifest. Spent well, that is not "the same grid, sooner" — it
is arms the single-device budget could not hold beside the base manifest. Five were added. Four are
robustness; **one changes the evidence for an RQ**, and it is worth separating from the rest.

### The matched-K pair — RQ1 tested by contrast rather than inferred

RQ1 asks whether the marginal benefit of added variates is governed by nominal count **K** or by
effective dimensionality **K_eff**. On the ladder the two are confounded by construction: adding
variates raises both, and the measured `corr(K, K_eff) = 0.828` is what is left of the separation. So
`D32` answered it with a non-nested J-test on a panel — identified, and fragile, because the K_eff
axis has four points and its variation across origins is doing the work.

Two eight-variate subsets separate them **directly**. Same K, same target, same seeds, same
everything, and the participation ratio is the only thing that moves:

| Subset | Composition | PR (feature frame) |
|---|---|---|
| `redundant` | F2 whole (all three volatility estimators, ~1 dof between them) and F3 whole (the third is the difference of the first two) | **3.609** |
| `orthogonal` | one or two per family across F1–F5, never doubling inside a family | **5.011** |
| ladder K=8, for reference | the pre-registered cut | 4.668 |

The pair **brackets the ladder's own rung**, which is the useful shape: whatever the ladder's K=8
result is, there is a lower-rank and a higher-rank K=8 either side of it. If accuracy tracks K_eff
rather than K, these two must differ while K does not; if they do not differ, the K_eff explanation
loses the one test that could have distinguished it by contrast.

Both subsets lead with `r`, because `TARGET_INDEX` is 0 and every consumer reads the target there.
`build_origin_tensors` gained a `columns` argument and asserts `len(columns) == k`, so a subset that
disagreed with its own K would fail loudly rather than train a differently-shaped model.

### The lookback sweep — the hyperparameter §6.2 never varied

`L = 96` was adopted from Liu et al. (2024) and held fixed, like everything else. It is nonetheless a
**first-order** choice for a transformer, and the horizon sweep varied H four ways while L stayed
still. L ∈ {48, 192} closes it.

**192 is the ceiling, and the reason is §4.3 rather than time.** Window cost per break is `L + H − 1`,
so a break costs 359 start positions at L=336 against 119 at L=96. Pooled that approaches root §4.3's
16% tolerance, and per origin it lands worst on 2018–2021, where 26 of 27 downtime blocks live. The
sweep would then measure gap density as much as lookback.

L has **no slot in `run_id`** — root §10.4 fixes that format — so the two arms carry their own tags and
`tensor_key` gained `seq_len`. Without that second half, the cache would hand the L=192 arm the L=96
windows the ladder built one cell earlier: a wrong answer with no error.

### The tuned arm — answering "you did not try" with a number

§6.2 states that nothing is tuned, and that is a design property: holding capacity fixed is what makes
the rungs comparable (`D38`). It reads to a referee as an omission. The reply should be a measurement:
**if the configuration the validation set prefers is still worse than a random walk, the null is not
an artefact of the defaults.**

Three properties keep it inside the pre-registration:

- **The grid is a module constant** (`TUNING_GRID`, 18 points over `d_model`, `e_layers`, `lr`), so it
  is in git before the probes run. A space chosen after seeing the winner is not a search. It contains
  §6.2's adopted configuration, asserted in the suite — a search that cannot return the incumbent
  cannot show the incumbent was reasonable.
- **Validation only, one origin.** Exactly where `D27` put the Stage 5 gate and for its reason: §11
  opens the test blocks once, after the design is frozen.
- **It sits outside the ladder.** `model_config` **refuses** to build the tuned arm without the
  selected config rather than falling back to §6.2's — a silent fallback would complete the run, write
  a plausible meta, and claim a selection that never happened.

`d_ff` is absent from the grid on purpose: `D62b` already swept it and made things worse at 14 of 15
origins.

### Seeds on the exploratory arms, three to five

The three-seed counts were **budget, not design** — root §10.3 sized the sweep against a single-GPU
session. The attention arm is where it matters most: §13.2 admits attention maps only when they are
"validated for stability across seeds", and the measured calm-to-stress shift (**+0.00056**) is smaller
than the between-seed standard deviation of one weight (**0.00064**). A claim resting on that
comparison should not rest on three draws.

### What did not move, and why the reasons are not budget

| Fixed | Why |
|---|---|
| Ladder seeds at 5 | `D18`, `D49` — the 8→12 rung is the designed contrast and cannot carry the fewest draws |
| The K rungs | `D01` — exactly one consistent cut exists; a fifth rung is a different experiment |
| iTransformer hyperparameters **inside** the ladder | `D38` — fixed capacity is what makes rungs comparable. This is why the tuned arm is outside it |
| Origin spacing at 5 months | §8.1 — effective independence is bounded near 4 *whatever* the spacing, so denser origins inflate G without adding information and worsen the overlap §9.2 must disclose |
| Model size, batch size | Not a GPU-count question. Run-level parallelism doubles **throughput**, not per-run capacity: a model still has to fit one T4, and at 280,472 parameters against 16 GB the utilisation is under 1%. Memory was never the constraint |

### Accounting

Manifest **969 → 1,620**. **Nothing is orphaned** — five seeds is a superset of three, and the five new
arms carry tags of their own, so all 894 completed runs are still in the manifest. The suite's old
"pre-`D62` core totals 684" assertion was a statement about *composition at one moment*; it is replaced
by the property root §10.4 actually guarantees, that no completed `run_id` falls out of the manifest.

**726 pending ≈ 6.4 GPU-hours ≈ 3.2 h wall on two devices** — which is the point: on one device it
would be 6.4 h and would not fit beside the base manifest in an 11 h session.

New contradictions found later take IDs **D71+**.
