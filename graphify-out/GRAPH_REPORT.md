# Graph Report - D:\pythonProject\invertedTransformer  (2026-08-07)

## Corpus Check
- 31 files · ~98,757 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1016 nodes · 1689 edges · 76 communities (69 shown, 7 thin omitted)
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 106 edges (avg confidence: 0.74)
- Token cost: 568,345 input · 0 output

## Community Hubs (Navigation)
- Data-Plane Test Suite
- Experiment-Plane Test Suite
- Origin Grid And Design Constants
- Stage 1 Kline Ingest
- Segment Law Implementation
- Run Cells And Pilot Result
- Grid Runner And Budget Guard
- CLAUDE.md Rewrite Plan
- Training Artifacts And Provenance
- Model-Plane Test Suite
- iTransformer Model Definition
- Window Budget Divergences
- Per-Origin Window Accounting
- Notebook Sync Tests
- Leakage Checklist And Ladder Rules
- Channel-Independence Literature
- Split Tensors And Scaler
- Traceability And Notebook Provenance
- Clustered-Inference Divergences
- Prediction Gathering And Amplification
- Effective Dimensionality Measures
- Diebold-Mariano And Clark-West
- Encoder Layers And Embedding
- Paper Production And Attention Debate
- Forecast Comparison Statistics
- Pre-Registration And Reporting Rules
- Baselines And Efficiency Tests
- Notebook Generator
- Divergence Register Open Items
- Metric Definitions And Window Assertions
- Survival Analysis And Power
- Purging And Window Construction
- RQ2 Design And Origin Overlap
- Decay Regression And CPCV Rejection
- Grid Arms And Artifact Schemas
- Retraining Cadence And Deflated Sharpe
- Clustered Inference And Plan Tasks
- Nested Forecast Test Literature
- K Ladder And K_eff Divergences
- Seed, Horizon And Schedule Divergences
- Kaggle Deployment Surface (D54)
- Variate Families F1-F5
- K_eff Table Construction
- Equivalence And Non-Nested Tests
- Baseline And Metric Scales
- Prediction And Scale Invariance
- RQ1 - K Versus K_eff
- Twelve Variate Construction
- Panel Beta1 And Wild Bootstrap
- Directional Accuracy Testing
- Kaggle Execution Envelope
- Architecture And Hyperparameters
- No-Imputation And Scaler Rules
- Adaptive Markets Mechanism
- Writing Posture And Gates
- Data Contract And Segment Law
- Embargo And Feature-Span Divergences
- Multiplicity Control Literature
- Crypto Microstructure Literature
- Cross-Variate Attention Design
- Decay Reporting Conventions
- The Three Research Questions
- Metric-Scale Divergences
- K_eff Row Divergence Measures
- Decay And b-star Estimation
- Bitcoin Efficiency Literature
- Binance Data Source
- Random Matrix Theory Provenance
- RQ3 Degradation Curve (Source Spec)
- Crypto Deep-Learning Baselines
- Reality Check And SPA
- Foundational Transformer Papers
- DLinear And Linear Baselines
- Pipeline And Module Inventory
- Horizon Sweep Divergence
- Repository Root

## God Nodes (most connected - your core abstractions)
1. `ITransformerConfig` - 22 edges
2. `FalsificationOrigin` - 21 edges
3. `execute()` - 20 edges
4. `ITransformer` - 18 edges
5. `RunCell` - 18 edges
6. `write_artifacts()` - 18 edges
7. `Origin` - 17 edges
8. `OriginTensors` - 16 edges
9. `build_origin_tensors()` - 16 edges
10. `RunSpec` - 16 edges

## Surprising Connections (you probably didn't know these)
- `Never A Bare Number` --semantically_similar_to--> `Traceability Contract`  [INFERRED] [semantically similar]
  paper/CLAUDE.md → CLAUDE.md
- `data/raw is never written to` --semantically_similar_to--> `D33 — Empty data/raw and an out-of-window boundary bar`  [INFERRED] [semantically similar]
  USAGE.md → docs/DIVERGENCE_REGISTER.md
- `Two GPUs are two independent processes, not two threads` --semantically_similar_to--> `D19 — "Hours, not days" holds only in one implementation regime`  [INFERRED] [semantically similar]
  USAGE.md → docs/DIVERGENCE_REGISTER.md
- `D13 — F2 estimators: per-bar or trailing-averaged` --rationale_for--> `Stage 3 — the twelve per-bar variates`  [INFERRED]
  docs/DIVERGENCE_REGISTER.md → USAGE.md
- `Stage 6 — the 534-run grid` --conceptually_related_to--> `D48 — Four outcome-determining choices left open after results`  [INFERRED]
  USAGE.md → docs/DIVERGENCE_REGISTER.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **The Ladder's Designed Controls** — claude_k1_degeneracy_control, claude_k12_redundancy_control, claude_uniform_attention_arm, claude_falsification_arm [INFERRED 0.85]
- **What Every Reported Number Must Resolve To** — claude_persist_raw_predictions, claude_code_sha256, claude_input_parquet_env, claude_paper_numbers_json [EXTRACTED 1.00]
- **The Clustered Inference Stack For beta1** — claude_decay_regression, claude_wild_cluster_bootstrap, claude_cluster_dependence, claude_dispersion_by_aggregation, claude_minimum_detectable_beta1 [EXTRACTED 1.00]
- **Nine-stage pipeline from ingest to paper_numbers.json** — usage_stage1_ingest, usage_stage2_segments_budget, usage_stage3_features, usage_stage3b_keff_gate, usage_stage4_invariants, usage_stage5_pilot_gate, usage_stage6_grid, usage_stage7_evaluation, usage_stage8_save [EXTRACTED 1.00]
- **Leakage surface as boundary x channel grid, one checklist item per cell** — docs_divergence_register_d43, docs_divergence_register_d24, docs_divergence_register_d27, docs_divergence_register_d44, docs_divergence_register_d45, docs_divergence_register_d28 [EXTRACTED 1.00]
- **Metrics whose pre-registered thresholds return a null by construction** — docs_divergence_register_d23, docs_divergence_register_d41, docs_divergence_register_d46 [INFERRED 0.85]
- **Variate families, the K ladder, and the K_eff measurement that scores it** — research_specification_itransformer_btc_family_f1_price_trajectory, research_specification_itransformer_btc_family_f2_volatility_estimators, research_specification_itransformer_btc_family_f3_intensity, research_specification_itransformer_btc_family_f4_order_flow, research_specification_itransformer_btc_family_f5_intrabar_location, research_specification_itransformer_btc_k_ladder, research_specification_itransformer_btc_participation_ratio, research_specification_itransformer_btc_k_eff [EXTRACTED 1.00]
- **The five fatal anti-leakage controls** — research_specification_itransformer_btc_segmentation_policy, research_specification_itransformer_btc_timestamp_window_validation, research_specification_itransformer_btc_purge, research_specification_itransformer_btc_standardscaler, research_specification_itransformer_btc_backward_looking_rolling_rule, research_specification_itransformer_btc_leakage_checklist [EXTRACTED 1.00]
- **RQ2 inference chain: A(b) → clustered decay regression → bootstrap → Figure 3** — research_specification_itransformer_btc_a_b_metric, research_specification_itransformer_btc_decay_regression, research_specification_itransformer_btc_wild_cluster_bootstrap, research_specification_itransformer_btc_figure3_decay_curve, research_specification_itransformer_btc_rq2 [EXTRACTED 1.00]
- **Attention interpretability caveat and its test battery** — reference_library_itransformer_btc_jain_wallace_2019, reference_library_itransformer_btc_wiegreffe_pinter_2019, reference_library_itransformer_btc_serrano_smith_2019, reference_library_itransformer_btc_uniform_attention_ablation, reference_library_itransformer_btc_attention_not_explanation_caveat [EXTRACTED 1.00]
- **H2's dual grounding: capacity-robustness trade-off, adaptive markets, and order flow** — reference_library_itransformer_btc_han_ye_zhan_2024, reference_library_itransformer_btc_khuntia_pattanayak_2018, reference_library_itransformer_btc_lo_2004_amh, reference_library_itransformer_btc_makarov_schoar_2020, reference_library_itransformer_btc_h2_theoretical_grounding [EXTRACTED 1.00]
- **The CLAUDE.md doc-set rewrite deliverable** — _claude_prds_claude_md_rewrite_spot_itransformer_prd_mvp_claude_md_set, _claude_prds_claude_md_rewrite_spot_itransformer_prd_divergence_register, _claude_prds_claude_md_rewrite_spot_itransformer_prd_tombstone, _claude_plans_claude_md_rewrite_spot_itransformer_plan_task3_root_research_programme, _claude_plans_claude_md_rewrite_spot_itransformer_plan_task6_scoped_claude_md_set [EXTRACTED 1.00]

## Communities (76 total, 7 thin omitted)

### Community 0 - "Data-Plane Test Suite"
Cohesion: 0.06
Nodes (39): budgets(), frame(), DataFrame, fixture, Data-plane tests: the runnable half of root §11's anti-leakage checklist. Reads…, `D26`'s whole reason for existing, as an executable claim. For fixed ``b`` the…, Root §11 — train ∩ val = ∅ and (train ∪ val) ∩ test = ∅., `D25` — 24 months is the leak; 21 is the design. (+31 more)

### Community 1 - "Experiment-Plane Test Suite"
Cohesion: 0.06
Nodes (40): DataFrame, Manifest, K_eff and metric tests — what root §9 and §10 claim about the grid.…, FATAL, `D44`. RQ1's regressor may not see a bar its outcome is measured on. The…, PR is 1 on a rank-one spectrum and K on a flat one — the interval it claims., It must not move when one channel is rescaled — the bug that made it useless.…, Every measure that claims ``[1, K]`` must actually stay inside it., `D48` — "re-cut the ladder" named no reachable alternative, so it is gone., `D46` — 24 alignments exist and each gives a different Sharpe; pick one first. (+32 more)

### Community 2 - "Origin Grid And Design Constants"
Cohesion: 0.09
Nodes (18): add_months(), FalsificationOrigin, Origin, origin_grid(), datetime, Design constants and the walk-forward origin grid. Every number here is fixed…, One walk-forward origin and every boundary derived from it. All boundaries are…, Start of the 24-month rolling training window. (+10 more)

### Community 3 - "Stage 1 Kline Ingest"
Cohesion: 0.12
Nodes (29): DatetimeIndex, assemble(), clip_to_window(), fetch_range(), find_gaps(), integrity_report(), KlineFetcher, main() (+21 more)

### Community 4 - "Segment Law Implementation"
Cohesion: 0.09
Nodes (26): break_summary(), BreakSummary, build_segments(), load_bars(), DataFrame, datetime, Path, The segment law: what breaks the series, and where. A **segment** is a maximal… (+18 more)

### Community 5 - "Run Cells And Pilot Result"
Cohesion: 0.09
Nodes (18): ExecutionSummary, PilotResult, OriginLike, The tensor-build key. Seeds inside a group share one build., Hyperparameters, identical at every rung except where an arm says otherwise.…, Small LRU over ``(arm, origin, K, H)`` builds. Bounded because one build is up…, What one worker did, and what is left. ``remaining`` and ``estimated_sessions``…, Root §8.5's Stage 5 gate. Note what it does **not** touch: the test blocks. (+10 more)

### Community 6 - "Grid Runner And Budget Guard"
Cohesion: 0.12
Nodes (24): BudgetGuard, build_feature_frame(), completed_run_ids(), discover_roots(), execute(), launch_workers(), _main(), manifest() (+16 more)

### Community 7 - "CLAUDE.md Rewrite Plan"
Cohesion: 0.12
Nodes (28): GPU-resident cost model with its stated counterfactual (no DataLoader), Milestone bundling decision — all seven milestones in one plan, Patterns to Mirror — documentation and artifact patterns from surviving files, Plan: CLAUDE.md Rewrite — Spot-Only iTransformer Research Programme, polars data plane with a single named pandas conversion boundary, Risk: root file rebuilds the 74 KB monolith — size ceiling as validation step, Deterministic run_id idiom — changing a component orphans prior outputs, Task 1: Tombstone and purge (+20 more)

### Community 8 - "Training Artifacts And Provenance"
Cohesion: 0.14
Nodes (23): device, One split's inputs, targets, and the timestamps they were issued at., SplitTensors, code_sha256(), _git_sha(), _input_sha256(), is_complete(), Path (+15 more)

### Community 9 - "Model-Plane Test Suite"
Cohesion: 0.12
Nodes (23): feats(), DataFrame, fixture, Feature, split and model tests — the claims root §5, §6 and §8 make about…, `D52c`. ``r`` is per segment, so each segment's first bar has no predecessor.…, FATAL, `D24`. Targets may not cross a boundary; inputs may., `D51b` — 720 forecast origins per clean block, never 601., `D31`/`D52b`. ``y_z = -mu_g/sigma_g``, and the tilt changes sign. ``y_z = 0``… (+15 more)

### Community 10 - "iTransformer Model Definition"
Cohesion: 0.14
Nodes (19): ITransformer, ITransformerConfig, Encoder-only iTransformer: each variate is a token, attention runs across them.…, ``(B, L, N) -> (B, H)`` on the target channel. The output is the target channel…, Hyperparameters, adopted unchanged from Liu et al. (2024) bar ``d_model``.…, Seed every source of nondeterminism root §16 names. ``cudnn.deterministic``…, set_seed(), Root §6.2 — K changes the token count, not a single weight shape. (+11 more)

### Community 11 - "Window Budget Divergences"
Cohesion: 0.13
Nodes (20): D24 — No purge at the train/validation boundary, D25 — 17,400 is the 24-month count; training is 21 months, D54e — Partial sessions crashed the evaluation cells, The closed form is an upper bound, not an identity, budget.py::COMMITTED_TRAIN_BUDGET pinned values, Budget constants — L, H, 119, spacing, 720, Feature-frame training range 13,545 … 15,217, Gap blocks straddling a span boundary charged whole (+12 more)

### Community 12 - "Per-Origin Window Accounting"
Cohesion: 0.14
Nodes (16): budget_table(), format_markdown(), origin_budget(), OriginBudget, DataFrame, Per-origin window accounting — the assertion target of root §11. Importers:…, Whether root §4.3's arithmetic matches the segment-wise truth. Disagreement…, Window starts surviving inside test block ``b`` — out of 720. **Test blocks do… (+8 more)

### Community 13 - "Notebook Sync Tests"
Cohesion: 0.13
Nodes (18): _code_sources(), _materialised(), notebook(), fixture, The notebook carries the package; these tests stop the two copies drifting.…, Every ``%%writefile`` cell precedes the first ``import itransformer_btc``. Cell…, ``build_notebook.py --check`` agrees the committed file is current., Code cells only. The markdown cells name both ``itransformer_btc`` and… (+10 more)

### Community 14 - "Leakage Checklist And Ladder Rules"
Cohesion: 0.13
Nodes (19): Anti-Leakage Checklist, Clark–West For Nested Pairs, Dependency Extras — ingest / stats / train / dev, Harvey–Leybourne–Newbold Small-Sample Correction, Optional Fifth Rung K=16, The K Ladder — 1 / 4 / 8 / 12, Leakage Surface As Boundary × Channel Grid, A Result That Looks Too Good Is Leakage (+11 more)

### Community 15 - "Channel-Independence Literature"
Cohesion: 0.15
Nodes (19): Anatolyev & Gerko (2005) — A Trading Approach to Testing for Predictability, Bergmeir & Benítez (2012) — Cross-validation for Time Series Predictor Evaluation, Bergmeir, Hyndman & Koo (2018) — Validity of CV for Autoregressive Series, Channel independence versus dependence debate, Han, Ye & Zhan (2024) — Capacity and Robustness Trade-off, TKDE 36(11), Ilbert et al. (2024) — SAMformer, channel-wise attention with identity-attention ablation, Kim et al. (2022) — RevIN, Reversible Instance Normalization, Kim et al. (2024) — Are Self-Attentions Effective for Time Series Forecasting? (NeurIPS) (+11 more)

### Community 16 - "Split Tensors And Scaler"
Cohesion: 0.15
Nodes (15): Semantics, build_origin_tensors(), _gather(), DataFrame, datetime, ndarray, OriginLike, Per-origin splits, the scaler, and the tensors the training loop slices. Three… (+7 more)

### Community 17 - "Traceability And Notebook Provenance"
Cohesion: 0.14
Nodes (18): Artifact Vintage And sha256 Pinning, code_sha256 Names The Code Off-Repo, Divergence Register Rule — Next Free ID, The Notebook Is Generated, Never Hand-Edited, ITBTC_PARQUET Resolves The Input Digest, Logic Lives In The Package; A Notebook Is A Launcher, paper_numbers.json Generates Every Table, Persist Raw Predictions, Always (+10 more)

### Community 18 - "Clustered-Inference Divergences"
Cohesion: 0.16
Nodes (18): Cameron, Gelbach & Miller (2008) — bootstrap-based cluster improvements, D06 — The decay regression is underspecified, D07 — The origin count is credited to the wrong cause, D09 — Reference library says twelve origins, D26 — Block index b is collinear with calendar month, D28 — Consecutive origins share most of their training data, D30 — Seed std is the wrong error bar for origin-aggregated results, D36 — Figure 3 plots A(b) for four K; A is one series (+10 more)

### Community 19 - "Prediction Gathering And Amplification"
Cohesion: 0.15
Nodes (18): amplification(), attention_amplification(), block_metrics(), gather_grid(), load_meta(), load_predictions(), _locate(), parse_run_id() (+10 more)

### Community 20 - "Effective Dimensionality Measures"
Cohesion: 0.18
Nodes (16): contemporaneous_pr(), gate_verdict(), lookback_covariance_pr(), lookback_stable_rank(), participation_ratio(), ndarray, Effective dimensionality — RQ1's independent variable, measured before…, PR after per-window standardisation over L — `D04`'s required companion. Args:… (+8 more)

### Community 21 - "Diebold-Mariano And Clark-West"
Cohesion: 0.16
Nodes (16): _bartlett_lrv(), clark_west_test(), dm_test(), _hln_and_p(), mae(), mse(), ndarray, ``[gamma_0 + 2 sum_{k=1}^{h-1} gamma_k] / T`` — the variance of ``d_bar``.… (+8 more)

### Community 22 - "Encoder Layers And Embedding"
Cohesion: 0.15
Nodes (9): EncoderLayer, InvertedEmbedding, Tensor, Attention over variates, then a position-wise FFN. Post-norm, as in the paper., Args: ``x`` of shape ``(B, L, N)``. Returns ``(B, H)``., Embed each variate's entire lookback: ``Linear(L -> d_model)``. With ``d_model…, ``(B, L, N) -> (B, N, d_model)``., Multi-head attention over the variate axis, optionally forced uniform. (+1 more)

### Community 23 - "Paper Production And Attention Debate"
Cohesion: 0.17
Nodes (16): Task 5: Root CLAUDE.md §11–§17, discipline and contracts, Paper-production section — IMRaD skeleton, table/figure inventory, citation discipline, Attention-is-not-explanation caveat for Section 4.6, Brownlees & Gallo (2006) — Ultra-High Frequency Data Handling Concerns, Citation verification status — entries assembled from memory, verify before citing, Hansen & Lunde (2006) — Realized Variance and Market Microstructure Noise, Gap pattern in Indonesian crypto-forecasting literature (five recurring defects), Jain & Wallace (2019) — Attention is not Explanation (+8 more)

### Community 24 - "Forecast Comparison Statistics"
Cohesion: 0.14
Nodes (16): Cameron, Gelbach & Miller (2008) — Bootstrap-Based Improvements with Clustered Errors, Diebold & Mariano (1995) — Comparing Predictive Accuracy, Diebold–Mariano test for every comparative claim, DA — directional accuracy, Stage 11 — economic evaluation with transaction costs, Eleven-stage execution pipeline, HAC (Newey–West) variance estimator at lag h−1, Harvey, Leybourne & Newbold (1997) — Testing equality of prediction MSEs (+8 more)

### Community 25 - "Pre-Registration And Reporting Rules"
Cohesion: 0.13
Nodes (15): b* As Interval-Censored Survival Data, Session Budget Guard, Citation Discipline — Verified DOI, Source Read, Hedged Priority Claim With Search Protocol, Idempotent Resume Across Sessions, IMRaD Structure, 10–14 Pages, K=12 Deliberate Redundancy As Control, Evaluation Gated On Grid Completeness (+7 more)

### Community 26 - "Baselines And Efficiency Tests"
Cohesion: 0.14
Nodes (15): ARIMA baseline (order by AIC on the training window), Lo & MacKinlay (1988) — Stock Market Prices Do Not Follow Random Walks, Target: log_return, single- and multi-step, LSTM baseline (2 layers, hidden 128, dropout 0.1), MAPE prohibited on log-returns, Preliminary market-efficiency tests (Variance Ratio, Hurst, ADF), MinMaxScaler excluded on correctness grounds, Naive-persist baseline (ŷ = last observed return) (+7 more)

### Community 27 - "Notebook Generator"
Cohesion: 0.19
Nodes (14): build(), _code(), guarded(), _lines(), main(), _markdown(), module_cell_source(), Assemble ``notebooks/iTransformer.ipynb`` from ``src/itransformer_btc/``. Root… (+6 more)

### Community 28 - "Divergence Register Open Items"
Cohesion: 0.15
Nodes (14): Divergence Register, D10 — The artifact on disk violates the specification (ffill), D11 — The Stage 1 entry point is missing, D14 — Undefined denominators and a division by zero, D16 — Unverified and mis-dated citations, D33 — Empty data/raw and an out-of-window boundary bar, Open items requiring external action, Second-pass audit provenance — two lenses never ran (+6 more)

### Community 29 - "Metric Definitions And Window Assertions"
Cohesion: 0.15
Nodes (13): assert_same_windows(), logrank(), per_origin_loss(), r2_oos(), Metrics, tests, and the three RQ estimators. Root §9. Everything here consumes…, ``MSE_model / MSE_naive`` — controls for period difficulty (root §9.1)., ``1 - RelMSE`` (`D20`) — the readable form of the same quantity. RelMSE near…, RMSE back in raw log-return units — root §9.1's second reporting scale. "RMSE… (+5 more)

### Community 30 - "Survival Analysis And Power"
Cohesion: 0.14
Nodes (12): kaplan_meier(), _loglog_band(), minimum_detectable_beta1(), _normal_quantile(), The MDE root §13.2 calls the most damaging omission on its list. Every design…, Kaplan-Meier estimate on the 30-day block grid., Smallest block with ``S(t) <= 0.5``; ``inf`` when never reached. ``inf`` is the…, Confidence set for the median: blocks whose CI band straddles 0.5. Table 5… (+4 more)

### Community 31 - "Purging And Window Construction"
Cohesion: 0.23
Nodes (13): Backward-looking rolling windows only (center=False), Brownlees & Gallo (2006) — Ultra-High Frequency Data Handling Concerns, build_windows() — timestamp-validated window builder, Why forward-fill (and interpolation) is actively wrong here, Leakage checklist (first five items fatal), Rejection of the 'near-gap' indicator variate, H-step purge at every train–test boundary, Rubin (1976) — Inference and Missing Data (+5 more)

### Community 32 - "RQ2 Design And Origin Overlap"
Cohesion: 0.21
Nodes (12): Participant-Composition Mechanism For H2, Cross-Origin Training Overlap 79.2%, CPCV Rejection Paragraph, Five-Month Origin Spacing, Fifteen Origins, Test-Window Survival Conditioned On Future Gaps, Mandatory Disclosures, Manuscript Is The Deliverable, RQ2 — Does The Multivariate Gap Decay With Model Age (+4 more)

### Community 33 - "Decay Regression And CPCV Rejection"
Cohesion: 0.26
Nodes (12): A(b) = [MSE_K1(b) − MSE_K8(b)] / MSE_K1(b) — RQ2 dependent variable, Bergmeir & Benítez (2012) — Cross-validation for time series predictor evaluation, Rejection of Combinatorial Purged Cross-Validation, Decay regression A(b) = β₀ + β₁·b + ε — the paper's core claim, Experiment grid (~517 runs), Figure 3 — decay curve A(b) versus b, López de Prado (2018) — Advances in Financial Machine Learning, Origin derivation — 13 origins, 2020-01 … 2026-01 (+4 more)

### Community 34 - "Grid Arms And Artifact Schemas"
Cohesion: 0.20
Nodes (11): D17 — No control for "is a transformer needed at all?", D19 — "Hours, not days" holds only in one implementation regime, D40 — Baselines carry no K assignment, D50 — The uniform-attention control is never used as an arm, GPU-resident batching, no Dataset or DataLoader, Four grid arms — main, uniform, fresh, horizon, What is not built yet, artifacts/preds/{run_id}.parquet schema (+3 more)

### Community 35 - "Retraining Cadence And Deflated Sharpe"
Cohesion: 0.18
Nodes (11): arXiv:2505.00356 (2025) — Retraining frequency of global models in retail demand forecasting, Bailey, Borwein, López de Prado & Zhu (2014) — Probability of Backtest Overfitting, Bailey & López de Prado (2014) — The Deflated Sharpe Ratio, Citation mapping by paper section, Diebold & Mariano (1995) — Comparing Predictive Accuracy, Gama et al. (2014) — A Survey on Concept Drift Adaptation (blind vs informed retraining), Harvey, Leybourne & Newbold (1997) — HLN small-sample correction, HLN small-sample correction formula S* against Student-t(T-1) (+3 more)

### Community 36 - "Clustered Inference And Plan Tasks"
Cohesion: 0.20
Nodes (10): Linear-span argument for excluding technical indicators, Open question: optional F6 rung at K≈16, Task 3: Root CLAUDE.md §1–§9, the research programme, Open questions (window start, torch version, Stage 1 tool, RQ3 answer space, F6 rung), Cameron, Gelbach & Miller (2008) — Wild cluster bootstrap, Cameron & Miller (2015) — Practitioner's Guide to Cluster-Robust Inference, Clustered inference for the decay regression (A(b) ~ b, 13 clusters), CPCV rejection defence — strategy selection versus controlled architecture comparison (+2 more)

### Community 37 - "Nested Forecast Test Literature"
Cohesion: 0.24
Nodes (10): Clark & McCracken (2001) — nested model forecast tests, Clark & West (2007) — adjusted statistic for nested forecasts, D03 — The use_norm verification test cannot pass as written, D27 — The Stage 5 gate opens test blocks, D29 — Every headline DM comparison is nested, Diebold (2015) — DM valid when forecasts are the object, McCracken (2007) — non-standard limiting distributions, Clark–West, not Diebold–Mariano, at the Stage 5 gate (+2 more)

### Community 38 - "K Ladder And K_eff Divergences"
Cohesion: 0.33
Nodes (10): D01 — The K ladder does not add up, D02 — The K_eff gate reads test-period data, D12 — signed_flow is a deterministic product of two other K=8 members, D32 — RQ1's K vs K_eff regression is not identified at three rung deltas, D39 — Target-channel vs all-channel loss unspecified, D44 — K_eff's span is undeclared and the statistic is blind to cross-lag structure, D48 — Four outcome-determining choices left open after results, Column order is ladder order (K = first K columns) (+2 more)

### Community 39 - "Seed, Horizon And Schedule Divergences"
Cohesion: 0.20
Nodes (10): D18 — Three seeds is too few for the headline claim, D21 — Trading rule and directional accuracy undefined at H=24, D38 — Validation-based hyperparameter selection that never occurs, D45 — Window loss is reported globally; per-block it reaches 50%, D46 — Unfixed trade phase, gap-spanning returns, unimplementable DSR, D47 — The LR schedule makes the epoch budget decorative, D49 — The flat 8→12 rung has the fewest seeds and no equivalence test, Liu et al. (2024) — iTransformer (+2 more)

### Community 40 - "Kaggle Deployment Surface (D54)"
Cohesion: 0.27
Nodes (10): D54 — The launcher cannot run without a second Dataset, D54f — The budget clock started inside each worker, Session budget guard checked at run boundaries, tools/build_notebook.py — notebook generator, code_sha256 — code identity where git is absent, input_sha256_source — report vs file-digest provenance, artifacts/meta/{run_id}.json schema, The notebook carries the package (twelve %%writefile cells) (+2 more)

### Community 41 - "Variate Families F1-F5"
Cohesion: 0.31
Nodes (10): Retention of all eleven meaningful kline columns, F2 — Volatility estimators (Parkinson, Garman–Klass, Rogers–Satchell), F3 — Intensity (log_quote_volume, log_trade_count, log_mean_trade_size), F4 — Order flow (taker_buy_ratio, signed_flow), F5 — Intrabar price location ((VWAP − C)/(H − L)), Hansen & Lunde (2006) — Realized Variance and Market Microstructure Noise, K=12 rung as deliberate redundancy control, The K ladder (1 / 4 / 8 / 12) (+2 more)

### Community 42 - "K_eff Table Construction"
Cohesion: 0.29
Nodes (10): corr_k_keff(), keff_row(), keff_table(), DataFrame, OriginLike, Rows and windows of one origin's 21-month training sub-block. Both are cut from…, Measure every K_eff variant for one (origin, rung) cell., Table 2b — every rung at every origin, on training spans only. Returns: One row… (+2 more)

### Community 43 - "Equivalence And Non-Nested Tests"
Cohesion: 0.20
Nodes (8): EquivalenceResult, j_test(), TOST verdict on a rung `D49` pre-registers as flat., Two one-sided tests — RQ1's pre-registered equivalence check (`D49`). RQ1's…, Davidson-MacKinnon J-test of model A against model B (`D32`). RQ1 is a **non-…, ``P(T_df > stat)`` — Student-t, falling back to the normal without scipy., tost_equivalence(), _upper_tail()

### Community 44 - "Baseline And Metric Scales"
Cohesion: 0.22
Nodes (9): Baselines, Each With An Explicit K, Preliminary Market-Efficiency Tests, D(i,b) On The Skill Scale, R²_oos = 1 − RelMSE, Naive-RW Mapped As y_z = −mu_g/sigma_g, Multivariate Ridge Baseline, Romano–Wolf Stepdown And Model Confidence Set, Pre-Registered Threshold tau = 5% (+1 more)

### Community 45 - "Prediction And Scale Invariance"
Cohesion: 0.28
Nodes (9): Module, no_grad, _mean_loss(), predict(), ndarray, Tensor, Mean MSE over a split, batched to bound peak memory rather than for speed., Root §6.3's corrected ``use_norm`` invariant (`D03`). The source specification… (+1 more)

### Community 46 - "RQ1 - K Versus K_eff"
Cohesion: 0.28
Nodes (9): ΔMSE per rung — RQ1 dependent variable, F1 — Price trajectory (r, upper_shadow, lower_shadow), K_eff — effective dimensionality, Exclusion of log(C/O) as a separate variate, Nominal variate count K, Participation ratio PR = (sum lambda)^2 / sum lambda^2, Rolling participation ratio (90-day window), RQ1 — Nominal count K or effective dimensionality K_eff? (+1 more)

### Community 47 - "Twelve Variate Construction"
Cohesion: 0.22
Nodes (8): build_features(), ladder_columns(), DataFrame, The twelve variates, and the K ladder cut over them. All twelve are…, Compute all twelve variates, per segment, dropping what is undefined. ``r`` is…, The variate names at rung ``k``. Raises: ValueError: If ``k`` is not one of the…, gate_pr(), Stage 3b's gate value — **pre-first-origin span only** (`D02`). Computed on…

### Community 48 - "Panel Beta1 And Wild Bootstrap"
Cohesion: 0.22
Nodes (7): _balanced_matrix(), Beta1Result, panel_beta1(), ``A(i,b) = alpha_i + beta1 b + eps`` with clustered inference (`D06`, `D42`)., The more conservative of the two weight schemes, as root §9.2 requires., ``(G x B)`` outcome matrix and the block axis, or a loud failure. Built by hand…, Fit ``A(i,b) = alpha_i + beta1 b + eps`` and test ``H1: beta1 < 0``. Args:…

### Community 49 - "Directional Accuracy Testing"
Cohesion: 0.22
Nodes (9): directional_accuracy(), DirectionalAccuracy, _hit_rate(), non_overlapping_mask(), pesaran_timmermann(), Window starts whose forecast period opens at **00:00 UTC** (`D46`). There are…, Pesaran-Timmermann (1992) test of directional predictability. Returns:…, DA at the three horizons §9.1 requires, with their testing regimes. ``da_h24``… (+1 more)

### Community 50 - "Kaggle Execution Envelope"
Cohesion: 0.29
Nodes (8): Directional Accuracy And Its Testing Regime, Economic Evaluation And Deflated Sharpe Ratio, Falsification Arm — Fresh Model At o+90d, GPU-Resident Regime, No DataLoader, Kaggle 2×T4 Execution Envelope, Gate Precision On Compute Capability, Not is_bf16_supported, Run Accounting — 789 Runs, Uniform-Attention Arm

### Community 51 - "Architecture And Hyperparameters"
Cohesion: 0.25
Nodes (8): d_model = 128, Not 512, Hyperparameters — Adopted, Never Tuned, Encoder-Only iTransformer Architecture, K=1 Attention Degeneracy As Designed Control, Learning Rate Halves Every Four Epochs, No Causal Mask On The Variate Axis, PyTorch Is The Only Framework, Tombstone — The Superseded 1-Minute Project

### Community 52 - "No-Imputation And Scaler Rules"
Cohesion: 0.25
Nodes (8): Forward-Fill Fabricates Zero-Then-Jump, Instance-Normalisation Confound On RQ1, No Imputation Anywhere, No Winsorization Of Extreme Returns, Rubin's Taxonomy Does Not Apply, Corrected Scale-Invariance Test, StandardScaler Chosen; Robust And MinMax Rejected, use_norm Makes The Outer Scaler Cancel

### Community 53 - "Adaptive Markets Mechanism"
Cohesion: 0.25
Nodes (8): Gama et al. (2014) — A survey on concept drift adaptation, Han, Ye & Zhan (2024) — Capacity and Robustness Trade-off, IEEE TKDE, Khuntia & Pattanayak (2018) — Adaptive Market Hypothesis (crypto), Lo (2004) — The Adaptive Markets Hypothesis, Shifting participant-composition mechanism behind H2, reference_library_itransformer_btc.md, Rolling OLS R² of r_{t+1} on K=8 features, Stage 3b gate — re-cut the ladder if measured K_eff at K=8 is far below ~6.5

### Community 54 - "Writing Posture And Gates"
Cohesion: 0.29
Nodes (7): Attention Is Not Explanation, Stage 3b Gate — PR < 5.0, Minimum Detectable beta1 Published Before Test Blocks, Pre-Registration Is Binding, Pre-Registered TOST Equivalence Margin, Critique Practice, Not People, paper/CLAUDE.md — Writing Posture

### Community 55 - "Data Contract And Segment Law"
Cohesion: 0.33
Nodes (7): Data Contract — BTCUSDT 1h, All Eleven Kline Columns Retained, Per-Origin Window Budget Exact Equality, Rogers–Satchell Stabiliser kappa = 1e-9, The Segment Law, Windows Validated By Timestamp Not Index, Variate Families F1–F5

### Community 56 - "Embargo And Feature-Span Divergences"
Cohesion: 0.43
Nodes (7): D04 — Instance normalisation strips volatility level, confounding RQ1, D13 — F2 estimators: per-bar or trailing-averaged, D15 — Embargo declared justified with no justification, D22 — No rung tests genuinely nonlinear features, D37 — The linear-span argument is not a theorem under use_norm, D43 — The leakage surface is declared closed on an incomplete enumeration, López de Prado (2018) — purging, embargo, CPCV

### Community 57 - "Multiplicity Control Literature"
Cohesion: 0.29
Nodes (7): D34 — Newey–West Bartlett contradicts the dm.test validation target, D35 — SPA and Reality Check cannot correct a pairwise matrix, Diebold & Mariano (1995) — comparing predictive accuracy, Hansen (2005) — Superior Predictive Ability test, Hansen, Lunde & Nason (2011) — Model Confidence Set, Romano & Wolf (2005) — stepdown FWER control, White (2000) — Reality Check

### Community 58 - "Crypto Microstructure Literature"
Cohesion: 0.33
Nodes (7): Cont, Kukanov & Stoikov (2014) — The Price Impact of Order Book Events, H2's theoretical grounding — capacity-robustness trade-off plus adaptive markets, Khuntia & Pattanayak (2018) — Adaptive Market Hypothesis and Evolving Predictability of Bitcoin, Lo (2004) — The Adaptive Markets Hypothesis, Makarov & Schoar (2020) — Trading and Arbitrage in Cryptocurrency Markets, Marshall, Nguyen & Visaltanachoti (2019) — Bitcoin Liquidity (~0.30% effective spreads), Order-flow participant-composition mechanism for H2

### Community 59 - "Cross-Variate Attention Design"
Cohesion: 0.38
Nodes (7): Stage 10 — cross-variate attention extraction and regime heatmaps, Cross-variate attention (each variate is a token), Hyperparameter configuration (L=96, H=24, d_model=128, d_ff=256, e_layers=2, n_heads=8), iTransformer encoder-only architecture, K=1 attention degeneracy as the designed control, Liu et al. (2024) — iTransformer, ICLR 2024, use_norm=True — instance normalization, mandatory not a tuning knob

### Community 60 - "Decay Reporting Conventions"
Cohesion: 0.47
Nodes (6): Decay Regression With Origin Fixed Effects, Dispersion Bound To Aggregation Level, Figure 3 — Decay Curve, One Series, A(i,b) — RQ2 Dependent Variable, Seed-Averaged MSEs Before Any Ratio, Never A Bare Number

### Community 61 - "The Three Research Questions"
Cohesion: 0.40
Nodes (6): K_eff Via Participation Ratio, Every K_eff Declares A Training-Only Span, Non-Nested K Versus K_eff Comparison, RQ1 — Nominal K Or Effective Dimensionality, RQ3 — Optimal Retraining Cadence, Three Questions The Study Asks

### Community 62 - "Metric-Scale Divergences"
Cohesion: 0.60
Nodes (6): D05 — D(b) conflates model decay with market difficulty, D20 — RelMSE near 1.00 is hard to read, D23 — The pre-registered tau is arithmetically unreachable, D31 — y_z = 0 in scaler space is a drift forecast, Smaller corrections absorbed without a numbered entry, Expected numbers table — tell a break from a change

### Community 63 - "K_eff Row Divergence Measures"
Cohesion: 0.33
Nodes (4): KeffRow, One (origin, rung) cell of Table 2b., ``pr_lookback_cov / (K * L)`` — the cross-lag PR as a share of its ceiling. The…, ``stable_rank - pr_raw`` — §5.4 requires this be reported as such. Positive…

### Community 64 - "Decay And b-star Estimation"
Cohesion: 0.33
Nodes (5): decay(), DecayResult, Per-origin ``D(i,b)`` and the censored ``b*`` it implies., ``b*(i) = min{b : D(i,b) > tau}``, **right-censored at 6** (`D41`). ``min{.}``…, ``D(i,b)`` on the **skill** scale (`D23`), normalised within origin (`D05`).…

### Community 65 - "Bitcoin Efficiency Literature"
Cohesion: 0.40
Nodes (5): Bariviera (2017) — Inefficiency of Bitcoin Revisited (dynamic Hurst), Lo & MacKinlay (1988) — Variance Ratio test, Nadarajah & Chu (2017) — On the Inefficiency of Bitcoin (rebuttal), Sensoy (2019) — High-frequency analysis of Bitcoin efficiency, Urquhart (2016) — The Inefficiency of Bitcoin

### Community 66 - "Binance Data Source"
Cohesion: 0.50
Nodes (5): Binance REST /api/v3/klines, binance_spot_klines.py (Stage 1 ingest), BTCUSDT spot 1h dataset, 2018-01-01 → 2026-08-01, gap_blocks — the number to read from the fetcher report, Spot-only constraint (no futures data)

### Community 67 - "Random Matrix Theory Provenance"
Cohesion: 0.67
Nodes (4): iTransformer benchmark framing — few-features-one-entity versus many-entities, Laloux et al. (1999) — Noise Dressing of Financial Correlation Matrices, Participation ratio as effective dimensionality K_eff, Plerou et al. (2002) — Random Matrix Approach to Cross Correlations (participation ratio)

### Community 68 - "RQ3 Degradation Curve (Source Spec)"
Cohesion: 1.00
Nodes (3): D(b) = [MSE(b) − MSE(1)] / MSE(1) — RQ3 degradation curve, RQ3 — Optimal retraining cadence, and does it depend on K?, Pre-registered degradation threshold tau

## Ambiguous Edges - Review These
- `reference_library_itransformer_btc.md` → `Lo (2004) — The Adaptive Markets Hypothesis`  [AMBIGUOUS]
  research_specification_itransformer_btc.md · relation: references

## Knowledge Gaps
- **86 isolated node(s):** `invertedtransformer`, `Directional Accuracy And Its Testing Regime`, `Session Budget Guard`, `paper_numbers.json Generates Every Table`, `Dependency Extras — ingest / stats / train / dev` (+81 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **7 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `reference_library_itransformer_btc.md` and `Lo (2004) — The Adaptive Markets Hypothesis`?**
  _Edge tagged AMBIGUOUS (relation: references) - confidence is low._
- **Why does `FalsificationOrigin` connect `Origin Grid And Design Constants` to `Experiment-Plane Test Suite`, `Run Cells And Pilot Result`, `Grid Runner And Budget Guard`?**
  _High betweenness centrality (0.021) - this node is a cross-community bridge._
- **Why does `build_origin_tensors()` connect `Split Tensors And Scaler` to `Experiment-Plane Test Suite`, `Run Cells And Pilot Result`, `Grid Runner And Budget Guard`, `Model-Plane Test Suite`, `Twelve Variate Construction`?**
  _High betweenness centrality (0.016) - this node is a cross-community bridge._
- **Why does `ITransformerConfig` connect `iTransformer Model Definition` to `Training Artifacts And Provenance`, `Run Cells And Pilot Result`, `Encoder Layers And Embedding`, `Grid Runner And Budget Guard`?**
  _High betweenness centrality (0.010) - this node is a cross-community bridge._
- **Are the 7 inferred relationships involving `ITransformerConfig` (e.g. with `BudgetGuard` and `ExecutionSummary`) actually correct?**
  _`ITransformerConfig` has 7 INFERRED edges - model-reasoned connections that need verification._
- **Are the 5 inferred relationships involving `FalsificationOrigin` (e.g. with `BudgetGuard` and `ExecutionSummary`) actually correct?**
  _`FalsificationOrigin` has 5 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `ITransformer` (e.g. with `RunSpec` and `TrainOutcome`) actually correct?**
  _`ITransformer` has 2 INFERRED edges - model-reasoned connections that need verification._