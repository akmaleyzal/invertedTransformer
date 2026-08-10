# CLAUDE.md

Governing document for this repository. Read it before doing anything else.

**Status:** authoritative as of **2026-08-08**. It supersedes both source specifications
(`research_specification_itransformer_btc.md`, `reference_library_itransformer_btc.md`), which are
**inputs, not authority**. Where they disagree with this file, this file wins, and
`docs/DIVERGENCE_REGISTER.md` says why. It also supersedes the pre-2026-08-05 version of itself
entirely — see §17.

**Second pass, 2026-08-06.** A five-lens adversarial audit was run against this document set. Three
lenses completed and produced 43 findings; **the two remaining lenses — consistency and
Kaggle/execution — never ran**, and the audit's own verifier stage died, so every finding was
adjudicated here by direct re-derivation against the text or the artifact on disk. Twenty-eight
survived as **D23–D50** (§14), ten of them fatal, and the rules below already reflect them. `D10` and
`D11` are closed by `D33`. **This pass is incomplete by design and must not be read as clearance**:
the unrun Kaggle lens is the only one that would have audited §10 end to end, and §10 is where `D25`
found a 14% error in every sample-count and tensor-size figure.

---

## 1. Project definition

**The deliverable is a manuscript, not a model.** There is no production inference path, no export
bundle, no serving contract. The model is an experimental instrument.

**Working title:** *Nominal Variates or Effective Dimensionality? A Walk-Forward Evaluation of
iTransformer for Hourly Bitcoin Forecasting.*
**Target venue:** Indonesian informatics journal (Sinta), IMRaD, 10–14 pages, 35–45 references,
IEEE citation style. Lock the title only after experiments conclude.

The study is **spot-only, single-asset, feature-based**: BTCUSDT 1-hour klines from Binance, and
nothing else. No futures. No second asset. No macro, on-chain, or sentiment data.

---

## 2. Hard constraints — non-negotiable

| ❌ Never | ✅ Instead |
|---|---|
| Add, import, or suggest **TensorFlow, Keras, or JAX** | **PyTorch is the only deep-learning framework.** Port any TF-only reference to torch idioms |
| Use **pandas** in the data plane | **polars** for segmentation, features, windowing, splits. pandas only at the one named stats boundary (§16) and in Stage 1 ingest (below) |
| `ffill`, `bfill`, `interpolate`, or reindex to a full hourly grid | **Segmentation.** A gap splits the series; windows are built inside segments, never across them |
| Impute anything, anywhere | Nothing. Missing bars are exchange downtime — no price formed, so there is no value to infer (§4.2) |
| Winsorize, clip, or drop extreme returns | Keep them. Extreme regimes are the object of study |
| `rolling(center=True)`, or any centred window | **No feature in this study uses a rolling window at all** (§5.3) |
| Fit a scaler on anything but the training sub-block | `StandardScaler` refit at every origin on the 21-month sub-block only |
| Add futures, second assets, or exogenous data | Spot BTCUSDT 1 h. The scope was fixed by decision, not by convenience |
| Report MSE on price levels, or MAPE on log-returns | MSE/MAE on standardised log-returns; RelMSE and `R²_oos` against Naive-RW |
| Cite a paper you have not read, or a DOI you have not verified | §13.3. No exceptions |
| Trust a result that looks too good | Assume leakage until proven otherwise |

**Stage 1 ingest is exempt from the polars rule, explicitly (2026-08-06).** `spot_klines_btc.py`
is pandas, was committed that way, and stays that way. The ban's justification is a *correctness*
argument, not a taste one: polars' rolling API is backward-closed by construction, so the
`center=True` leak class is unrepresentable there and is one keyword away in pandas (§16). Stage 1
computes **no rolling window at all** — it paginates REST responses, coerces strings to numerics,
de-duplicates, clips to the half-open window, and counts gaps — so the argument has nothing to bite
on, and a rewrite would risk the one artifact the whole study rests on to buy nothing. The ban
applies in **full** from segmentation onward, i.e. everywhere in `src/`. Do not read this exemption
as licence to reach for pandas downstream; the named stats boundary in §16 is the only other place
it may appear.

Timezone is **UTC everywhere**. Every timestamp is epoch-based and compared as an integer.

---

## 3. Research questions — pre-registered

Fixed before any model runs. Changing any of them after seeing results is a new experiment and must
be declared as one.

| Code | Question | Hypothesis | Dependent variable |
|---|---|---|---|
| **RQ1** | Is the marginal benefit of added variates governed by nominal count **K** or by effective dimensionality **K_eff**? | H1: benefit tracks K_eff. Real gains at K=1→4→8, flat at 8→12 | `ΔMSE` per rung, regressed separately on K and on K_eff |
| **RQ2** | Does the multivariate-over-univariate gap narrow as time-since-training grows? | H2: it narrows. The microstructure-to-return mapping is regime-specific | `A(i,b) = [MSE_K1 − MSE_K8]/MSE_K1`; claim is **β₁ < 0** |
| **RQ3** | What retraining cadence is optimal, and does it depend on K? | H3: larger K decays faster | `b* = min{b : D(b) > τ}` |

**RQ2 compares K=1 against K=8, never K=12.** K=12 carries deliberate redundancy (§5.2); using it
would confound decay with that redundancy.

**Pre-registered threshold:** headline **τ = 5%**, sensitivity reported at τ ∈ {2.5%, 5%, 10%, 50%}.
Choosing τ after seeing the decay curve is p-hacking.

**τ is a fraction of skill lost, not a fraction of RelMSE (`D23`) — and the distinction decides
whether RQ3 can return an answer at all.** Under the original `D(b)` definition on RelMSE, D20's own
magnitudes make every τ unreachable: with `RelMSE(1) = 0.996`, even *total* destruction of the
model's edge gives `D(b) = 1/0.996 − 1 = 0.402%`, so τ = 2.5% would require the model to become
2% *worse than forecasting zero*. RQ3 would return "no decay detected" regardless of the data — a
result fixed before a single epoch runs, by a units mismatch rather than by the market. `D(i,b)` is
therefore defined on the skill scale (§9.1), where τ = 5% means "5% of the edge is gone" and
τ = 100% means "the edge is gone".

**Pre-registered equivalence margin (`D49`).** RQ1's claim that the 8→12 rung is flat is an assertion
of **no effect**, and a non-significant ΔMSE is a failure to reject, not evidence of equivalence.
Fixed in advance: the rung counts as flat if two one-sided tests reject at α = 0.05 against
`Δ_eq = 0.25 × ΔMSE₄→₈`, i.e. the 8→12 gain is at most a quarter of the 4→8 gain. Choosing `Δ_eq`
after seeing the rung is the same p-hacking this section forbids for τ.

**Pre-registered answer space for RQ3.** Six 30-day blocks means `b*` resolves only to 30-day
granularity and only out to 180 days. If no block crosses τ, the honest answer is *"no decay
detected within 180 days"* — a right-censored result, not a missing one. Say so in those words.

**Mechanism behind H2** (economic, not just statistical): shifting participant composition —
retail-dominated flow 2018–2020, the 2021 leverage cycle, institutional flow after spot-ETF
approval in 2024. Order-flow predictability should decay as market making tightens. Grounded in the
Adaptive Markets Hypothesis (Lo 2004; Khuntia & Pattanayak 2018) and in the capacity–robustness
trade-off (Han, Ye & Zhan 2024) independently.

**Claimed contributions:** (1) first walk-forward evaluation of iTransformer on a crypto asset with
explicit decay measurement; (2) separation of nominal variate count from effective dimensionality as
competing explanations for cross-variate gains; (3) evidence-based retraining cadence under a
pre-registered degradation threshold.

---

## 4. Data contract

### 4.1 Measured facts

Measured from `data/raw/BTCUSDT_1h_report.json`, not assumed. Re-verify after any refresh.
**The four artifacts live in `data/raw/` (`D33`, resolved 2026-08-06).** The register offered two
ways to close `D33` — correct every path in this document to `data/`, or move the files into
`data/raw/`. The move was taken, because §2's immutability rule reads more cleanly when the immutable
inputs sit under a directory whose name says so, and because `data/processed/` then sits beside it as
the only writable half of `data/`. Any path of the form `data/BTCUSDT_*` in an older document or
transcript is stale by exactly one directory level.

| Field | Value (in-window) |
|---|---|
| Symbol / interval / source | BTCUSDT spot, 1 h, Binance REST `/api/v3/klines` |
| Window | 2018-01-01T00:00Z → 2026-08-01T00:00Z, **end exclusive** |
| `bars_expected` / `bars_actual` | 75,216 / 75,094 |
| `coverage_pct` | 99.8378 |
| `missing_bars` / `gap_blocks` / `largest_gap_bars` | 122 / **27** / 33 |
| `duplicate_timestamps` / `monotonic_index` | 0 / true |
| `ohlc_violations` / `taker_buy_exceeds_volume` | 0 / 0 |
| `zero_volume_bars` / `zero_trade_bars` | 3 / 3 |
| Worst year | 2018, 99.292% |

**The report on disk now reproduces this table exactly (`D33` closed, 2026-08-06).** It previously
read `bars_actual` 75,095 / `missing_bars` 121 / `coverage_pct` 99.8391 with
`actual_last_bar_utc` = `2026-08-01T00:00:00` — one bar **past** the declared end-exclusive boundary,
which also produced the impossible 2026 coverage of 100.02%. `spot_klines_btc.py` now enforces the
half-open window in `clip_to_window()` and the artifacts were rebuilt from the retained JSONL with
`--rebuild-only`, touching no network. Verified after the rebuild: 75,216 / 75,094 / 99.8378,
122 missing across 27 blocks, largest 33, last bar `2026-07-31T23:00`, 2026 coverage 100.000%, and
`parquet_rows == bars_actual == 75,094` with `BTCUSDT_1h_gaps.csv` summing to 122.

**Why that one bar mattered more than its size suggests.** Gap *detection* was never wrong —
`find_gaps` already used `inclusive="left"`, so the CSV always said 122. Only counts derived from
`len(df)` were off. The artifact therefore carried a report whose `missing_bars` **contradicted its
own gaps file**, in the direction that flatters the data, and every per-year coverage figure would
have inherited it. A defect that leaves two artifacts disagreeing is the kind that survives review;
the regression test for it lives in `spot_klines_btc.py --self-test`.

**Artifact vintage** (§12 — numbers produced under different hashes are not comparable):

| Artifact | sha256 (first 16) |
|---|---|
| `BTCUSDT_1h.parquet` | `8270a84b07c2923b` |
| `BTCUSDT_1h_gaps.csv` | `cfab4cf4c20ec00d` |
| `BTCUSDT_1h_raw.jsonl` | `30721a663bd2ce58` |

Full digests live in `artifact_sha256` inside the report itself, written by the ingest script. Regenerated
2026-08-06T06:38Z. Any run whose `meta/*.json` names a different parquet digest is a different vintage.

**All eleven meaningful kline columns are retained.** Truncating to OHLCV silently destroys
families F3, F4, F5 and collapses the ladder from 12 to 6. Three columns carry information
underivable from OHLC: `quote_asset_volume` (→ VWAP), `number_of_trades` (→ intensity),
`taker_buy_base_volume` (→ signed flow). The twelfth field `ignore` is dropped. Numeric fields
arrive as **strings** and must be coerced explicitly — silent failure otherwise.

### 4.2 Gaps are not missing values

BTCUSDT trades continuously, so zero-trade hours are ruled out. What remains is **exchange downtime
and scheduled maintenance**, confirmed against `data/raw/BTCUSDT_1h_gaps.csv` (27 rows, `D33`).

Rubin's MCAR/MAR/MNAR taxonomy applies to values that exist but went unobserved. **When the exchange
is down no price forms** — no matching, no book, nothing to approximate. Imputation is not risky
here, it is *undefined*: there is no ground truth, so no metric can justify any imputation choice.
This also retires the MNAR diagnostic, since there is no missingness mechanism left to model.
Cite Rubin (1976) precisely in order to argue the taxonomy does **not** apply.

**Why forward-fill is actively wrong here.** With bar `t` missing and `P_t := P_{t−1}`:

```
r_t     = log(P_t / P_{t−1})   = 0      <- fabricated zero
r_{t+1} = log(P_{t+1} / P_t)            <- full 2-hour return compressed into one bar
```

A zero-then-jump pattern that never occurred, which a model will learn as signal. Worse here
specifically: ffill forces `H = L = O = C`, so Parkinson and Garman–Klass return **zero volatility
immediately before an extreme bar**, poisoning all of F2. `volume = 0` gives `log(0) = −inf` and
`taker_buy_ratio = 0/0`. Linear interpolation removes the zero-then-jump but understates realized
volatility at exactly the outage boundaries — opposite bias, equal contamination.

### 4.3 Segment law

A segment is a maximal run of contiguous, usable hourly bars. Segments are broken by:

1. any missing bar (27 downtime blocks, 122 bars); **and**
2. any **zero-volume or `H == L` bar** — it carries no trade information, exactly like downtime, so
   it receives the same treatment: excluded, and the series splits there. This is what makes the
   `(VWAP − C)/(H − L)` division and `log(volume)` total, not partial, functions (`D14`).

`r` is computed **per segment**; the first bar of each segment yields NaN and is dropped. Computing
returns on a concatenated series injects giant cross-gap returns into μ_g and σ_g before any window
is excluded.

**Windows are validated by timestamp, never by positional index.** This is the highest-probability
silent bug in the pipeline: after any row drop, positional sliding closes gaps invisibly.

```
window [s, s+L+H) is valid  ⟺  t[s + L + H − 1] − t[s] == (L + H − 1) hours
```

**Cost accounting.** A break costs `L + H − 1 = 119` window start positions plus its own bars, so
cost is dominated by the *number of breaks*, not their length:

```
windows lost ≈ 119 × breaks + missing bars
             ≈ 119 × 30 + 125  ≈ 3,695  ≈ 4.9% of ~74,975
```

Thirty breaks = 27 downtime blocks + 3 zero-volume bars — **plus the `H == L` bars, whose count is
not in the report and has never been measured**. Emit it at Stage 2 before this arithmetic or D14's
cost figure is treated as final, and confirm whether the 3 zero-volume and 3 zero-trade bars are the
same bars. Read **`gap_blocks`** from the report, not `missing_bars`. Well inside the 16% tolerance,
so **no narrowing of the analysis window is required**.

**But the pooled figure is the wrong granularity for anything except that feasibility argument
(`D45`).** Gaps are not uniformly distributed: 26 of 27 downtime blocks fall in 2018–2021 and none
after 2023-03. Consequently per-origin **training**-window loss runs from ~11% at the earliest origin
down to **0%** at the latest and is monotone in calendar time; per-(origin, block) **test**-window
loss runs from 0% to **50.4%** — the origin at 2020-07 loses 363 of 720 window starts in its sixth
block, from breaks at 2020-11-30, 2020-12-21 and 2020-12-25 (`119 × 3 + 6`). Two consequences:

- the rejected-window count must be asserted **per origin, by exact equality** against the committed
  break table in **`docs/ORIGIN_WINDOW_BUDGET.md`** —
  `rejected(origin) == 119 × breaks(origin) + missing(origin)`. Asserted
  against the pooled 4.9%, it fires spuriously at fourteen of fifteen origins on the first run, gets
  loosened until it passes, and then can no longer distinguish positional drift from ordinary
  between-origin variation. That is the *only* defence against the silent bug named above;
- **test-window survival is conditioned on future gaps.** Whether a forecast issued at *s* is
  evaluated depends on whether the next 120 hours contain an outage — information unavailable at *s*.
  Since Binance outages cluster on stress, the dropped targets are systematically the high-volatility
  ones. Report surviving-window counts per cell and state the exclusion in Limitations (§9.2, §11).

### 4.4 The artifact on disk — `D10` and `D11` closed, one defect remaining

**Superseded by `D33`.** The earlier text asserted that `data/raw/BTCUSDT_1h.parquet` was written
with `"fill_policy": "ffill"`, `"rows_written": 75216`, `"synthetic_bars": 122` and must not be
consumed as-is. **None of those three keys exists in the report on disk**, and the parquet held
75,095 rows — equal to the then-reported `bars_actual`, i.e. **unfilled** (75,094 after the boundary
bar was dropped; the point stands, no row was ever synthesised). The artifact was regenerated; the register
was not updated. `D10` is therefore closed, and its prescribed remedy ("drop every row flagged
synthetic") was in any case unrunnable, there being no flag column to filter on. The runnable
replacement is in §11: assert `parquet_rows == bars_actual` and assert the timestamp diff set
contains the 27 gap blocks. Record the regeneration date and the artifact sha256 under §12 — without
them, no run can establish which vintage it consumed, and §12 forbids comparing numbers across
vintages.

**That last defect is now closed too** (2026-08-06). The boundary bar at `2026-08-01T00:00` was
dropped by `clip_to_window()` and the artifacts re-emitted from the retained JSONL; the parquet holds
75,094 rows and the report carries `artifact_sha256`. `D33` is closed in full — both its path fork
(the artifacts moved to `data/raw/`) and its data defect.

`BTCUSDT_1h_gaps.csv` is retained as a diagnostic and as the source of segment boundaries.
`BTCUSDT_1h_raw.jsonl` makes offline re-derivation possible without re-hitting the API.

**Stage 1 exists.** `D11` claimed `binance_spot_klines.py` was "absent from the tree and never
committed". The ingest script sits at the repository root as **`spot_klines_btc.py`** (651 lines).
`D11` is closed; the name in Table 1 must match the file.

### 4.5 Preliminary market-efficiency tests

Run once, report in the Data section. Converts "efficient market" from assumption into finding.

| Test | Library | Reading |
|---|---|---|
| Variance Ratio (Lo–MacKinlay) | `arch.unitroot.VarianceRatio` | VR ≈ 1 → consistent with random walk |
| Hurst exponent | R/S implementation | H ≈ 0.5 → no long memory |
| ADF | `statsmodels.tsa.stattools.adfuller` | log-returns stationary |

Do **not** claim the market is efficient. State that evidence is mixed and time-varying
(Urquhart 2016; Nadarajah & Chu 2017; Bariviera 2017; Sensoy 2019), then report your own numbers.

---

## 5. Variates, the K ladder, and feature-engineering policy

### 5.1 Families

| Family | Variates | Independent dof |
|---|---|---|
| **F1** Price trajectory | `r = log(C/C₋₁)`, `upper_shadow = log(H/max(O,C))`, `lower_shadow = log(min(O,C)/L)` | 3 |
| **F2** Volatility estimators | `log` Parkinson, `log` Garman–Klass, `log` Rogers–Satchell | ~1, redundant by construction |
| **F3** Intensity | `log_quote_volume`, `log_trade_count`, `log_mean_trade_size` | 2 — the third is the difference of the first two |
| **F4** Order flow | `taker_buy_ratio`, `signed_flow` | 1–2 |
| **F5** Intrabar location | `vwap_location = (VWAP − C)/(H − L)`, `VWAP = quote_volume/volume` | 1 |

Twelve variates total. Pinned definitions (`D12`, `D13`, `D14`):

- `taker_buy_ratio = taker_buy_base_volume / volume` — **base-denominated**, the canonical
  buyer-initiated volume share. The quote-denominated variant is a robustness check, not the default.
- `signed_flow = (2·taker_buy_ratio − 1) · log_quote_volume`. **Disclose** that this is a
  deterministic product of two other K=8 members; it weakens the claim that K=8 is the rung of
  maximum effective rank, and the measured participation ratio settles it.
- **F2 estimators are per-bar, with no trailing average** — see §5.3. All three are provably ≥ 0
  given `H ≥ max(O,C)` and `L ≤ min(O,C)`. **Only two of the three are strictly positive once
  `H == L` bars are excluded** (`D52`): Parkinson is `∝ (ln H/L)²`, and Garman–Klass is bounded below
  by `0.114 (ln H/L)²` because `2ln2 − 1 ≈ 0.386 < 0.5` and `|ln(C/O)| ≤ ln(H/L)`. **Rogers–Satchell
  is not**: it vanishes on any shadowless bar — H equal to one of O/C and L equal to the other — and
  33 such bars exist. It therefore uses `log(RS + 1e-9)`; the constant, its justification and the
  measured distribution are in `D52a`. `log` is total for the other two as written.
- `log(C/O)` is **not** a variate. Crypto bars are contiguous, so `log(C/O) ≈ r` at ρ ≈ 0.99 —
  silent duplication that inflates K without inflating K_eff, corrupting RQ1's own axis.

### 5.2 The ladder — corrected (`D01`)

The source specification's K=8 rung sums to **nine** and double-assigns `log_mean_trade_size`.
Exactly one consistent cut exists, and this is it:

| K | Members added | Cumulative | Expected K_eff |
|---|---|---|---|
| **1** | `r` | 1 | 1 |
| **4** | `upper_shadow`, `lower_shadow`, `log_quote_volume` | 4 | ~3.5 |
| **8** | `log_trade_count`, `taker_buy_ratio`, `signed_flow`, `vwap_location` | 8 | ~6.5 |
| **12** | Parkinson, Garman–Klass, Rogers–Satchell, `log_mean_trade_size` | 12 | ~7 |

**The 8→12 rung is deliberately redundant and functions as a control, not an accident.** It adds
four nominal variates carrying almost no additional rank. If accuracy rises 4→8 then flattens 8→12,
that is not a failed experiment — it is the demonstration that nominal K is the wrong axis.
**State this explicitly in the methodology**, next to the K=1 degeneracy note (§6.2). Left unstated,
a reviewer reads the flat rung as a null result rather than as the designed contrast.

The K_eff values above are **reasoned from family structure, not measured**. They may be wrong.
Fix the hypothesis to the measurement, never the measurement to the hypothesis (§5.4).

**No variate may be added outside F1–F5.** A "near-gap indicator" flag, a calendar dummy, or any
other convenience variate breaks the family taxonomy and renders the K=8 vs K=12 contrast
uninterpretable. Losing a few percent of windows is far cheaper than contaminating the study's
primary independent variable.

### 5.3 Feature-engineering policy — why exactly twelve

**All twelve variates are engineered.** None is a raw kline column; the raw feed is eleven columns
and every variate is a transform of it. What is excluded is a specific *class*: technical indicators
(RSI, MACD, Bollinger, MA/EMA), multi-bar rolling statistics, calendar and session dummies,
cross-asset, on-chain, sentiment, and macro data.

Four reasons, and **the first is load-bearing** (`D37` — it was previously the second):

1. **K is RQ1's independent variable.** Anything outside F1–F5 breaks the taxonomy that makes K_eff
   interpretable. **This alone excludes technical indicators**, and it is the reason to state in the
   paper: RSI, MACD and Bollinger bands belong to no family in §5.1, so admitting them makes the
   K=8-versus-K=12 contrast uninterpretable. The exclusion does not depend on reason 2.
2. **The parsimony argument — not a span theorem (`D37`).** iTransformer embeds each variate's
   *entire* L=96 lookback through `Linear(96 → d_model)`. A **linear** function of that lookback —
   MA(n), EMA, momentum, n-step differences — is recoverable from the embedding *in principle*, so
   adding it as a separate variate raises nominal K while changing inductive bias and per-channel
   normalisation rather than information content. That is close to the phenomenon RQ1 exists to
   measure, and injecting it into the ladder contaminates the instrument.

   **Three caveats, stated because the earlier formulation overclaimed.** (a) Under `use_norm=True`
   each channel is divided by its **own** per-window σ, and `x ↦ (a·x)/std_t(a·x_W)` is not linear
   in `x` — so a trailing average added as a *separate variate* is normalised by its own scale and is
   **not** literally inside the span of the normalised original channel. Per-channel instance
   normalisation is itself a nonlinearity; D04 says so and must not be contradicted here. (b) With
   `d_model = 128 > L = 96` the projection is generically **injective**, so the full lookback
   survives it and the linear/nonlinear line is *not* the recoverable/unrecoverable line. (c)
   Consequently the exclusion list is not derived from this reason: MA/EMA/MACD are linear in the
   lookback, but RSI (a ratio of sums of positive and negative parts) and Bollinger bands (a rolling
   standard deviation) are not. Reason 1 carries the exclusions; reason 2 explains why adding linear
   transforms would inflate K without inflating information.
3. **Sample budget.** **13,558–15,217** training windows per origin against 12 × 96 = 1,152 input
   dimensions. That is the **21-month** sub-block, measured per origin (`D25`, `D45`,
   `docs/ORIGIN_WINDOW_BUDGET.md`); 17,400 is the 24-month count and must not appear.
4. **Benchmark positioning.** iTransformer's own suite splits into few-features-one-entity
   (ETTh1 = 7, Weather = 21) and many-entities (Electricity = 321, Traffic = 862), and the paper does
   not distinguish them. K ≤ 12 sits deliberately in the first regime; measuring K_eff makes the
   distinction quantitative. This is the strongest available framing for the RQ1 contribution.

**Corollary — no feature uses a rolling window.** Applying reason 2 consistently: pre-smoothing an
F2 estimator over 24 bars is *strictly less informative* than feeding the per-bar estimator, because
the model can compute that average itself and cannot recover what smoothing destroyed. Every one of
the twelve variates is therefore a pure per-bar function of the current bar, except `r`, which uses
the current and previous close.

This is a **structural safety property, not a style choice**: with no rolling window anywhere, the
`center=True` leak class is unrepresentable.

**But the surface is not therefore closed (`D43`).** The earlier claim — that the leakage surface
"collapses to exactly two paths, segment crossing and scaler fitting, both closed by assertion" — is
scoped to *feature construction* at the *train–test* boundary, and a two-path enumeration fails at
the first boundary that is not train–test. What it omits: the **train–validation** boundary (§8.2,
`D24`); the **model-selection** channel (§8.5, `D27`); the **evaluated-sample-composition** channel
(§4.3, `D45`); and the **cross-origin training-window overlap** channel (§8.1, `D28`). Because the
closure claim licensed their absence, §11 carried no item for any of them. Enumerate the surface
instead as a grid of **boundary** (train/val · train/test · cross-origin) × **channel** (features ·
labels · scaler · model selection · evaluated-sample composition), with one §11 item per non-empty
cell. §8.3's feature-lookback argument stays verbatim — it is correct — but it covers **one cell**,
not the whole grid. A closed-surface claim is also what stops the hunt §2 mandates.

**Optional fifth rung (`D22`, re-justified by `D37`).** Pre-registered as an **optional K=16 rung**
adding trailing realized variance, signed-flow autocorrelation and VPIN-style order-flow toxicity.
The old rationale — that nonlinear functionals "lie outside the embedding's span" — does not survive
`d_model > L` and is withdrawn. The real distinguishing ground is that these are **multi-bar
functionals**: they reintroduce the rolling-window leakage surface this section closes, and running
the rung therefore **reopens the no-embargo argument in §8.3 / `D15`**, which must be re-derived
before the rung executes. As an explicit fifth rung it still supplies a second high-nominal-K /
low-K_eff control. **Folded into the existing four rungs it destroys the instrument** — separate rung
or nothing. **Run condition, made verifiable (`D48`):** origin 1's Stage 5 gate passes at α = 0.05,
**and** at least 5 GPU-hours of weekly quota remain after the main grid completes. Seeds 42–46,
H = 24, all origins. Pre-registered prediction: if benefit tracks K_eff, K=16 improves on K=12 by
more than the 8→12 rung does. If the arm is not run, §13.2 reports **which clause failed, with its
number**.

### 5.4 Pre-model measurement — `K_eff`

Run **before training anything**. These produce RQ1's independent variable and test H2's premise at
zero extra data cost.

| Measurement | Span | Definition | Purpose |
|---|---|---|---|
| Participation ratio | **per origin, that origin's 21-month training sub-block** (`D44`) | `PR = (Σλᵢ)² / Σλᵢ²` on the correlation matrix of each rung | Supplies K_eff **as a varying panel regressor**. Bounded in [1, K] |
| Lookback-aware PR | same | PR of the `K·L × K·L` covariance spectrum, or stable rank of the `K × 96` window block averaged over windows | **`D44`** — the contemporaneous PR is blind to cross-lag structure |
| PR on window-normalised features | same | Same, after per-window standardisation over L | **`D04`** — see below |
| Gate PR | **pre-first-origin span only** (`D02`) | as row 1 | Gates the ladder; informs no other number |
| Rolling PR | full sample | 90-day rolling window, 2018–2026 | **descriptive only** — may inform no design decision |
| Rolling OLS R² | full sample | `r_{t+1} ~ (K=8 features)`, 90-day window | If unstable, H2's premise is established before a single epoch runs |

**`D44` — every reported K_eff declares its span, and RQ1's regressor is training-only.** Previously
only *the gate* and *the rolling* statistic carried a span constraint; the static per-rung PR that
feeds RQ1 declared none, so nothing forbade computing it over 2018–2026 — a span containing every
origin's test blocks. The regressor would then be estimated on the same data as the outcome, and
RQ1's claim partly circular, while §11's fatal item still passed because it audits only the gate.
That was the one leakage path surviving every checklist item by construction. Computing PR **per
origin on training data** closes it and, as a bonus, makes K_eff vary — which is what `D32` needs to
make the K-versus-K_eff comparison identifiable at all. Any full-sample PR is labelled descriptive in
Table 2b and is never a regressor.

**`D44` — and the construct is not what the model sees.** PR on the K × K *contemporaneous*
correlation matrix is blind to cross-lag structure, while the model consumes a K × 96 block and
embeds each variate's entire lookback. Two variates can be near-uncorrelated contemporaneously yet
near-redundant to a model with a 96-hour lookback, and conversely. Report the contemporaneous PR
**and** at least one lookback-aware measure on the same rungs; pre-register which is RQ1's regressor
before Stage 3b, and report the divergence between them in §4.1b whatever it turns out to be. If the
effective-dimensionality construct does not correspond to what the architecture consumes, the
study's second claimed contribution is a measurement-validity failure rather than a finding — which
is the question a methods referee will spend the review on.

**`D04` — the instance-normalisation confound.** `use_norm=True` divides each window by its own
per-variate σ over L, so the F2 estimators contribute *shape*, not *level*. The 8→12 rung can
therefore flatten for a reason that has nothing to do with redundancy, confounding RQ1's axis. PR
must be measured on **window-normalised features as well as raw**, both reported, and the confound
disclosed in Limitations whatever the outcome. Provenance for the statistic: Laloux et al. (1999),
Plerou et al. (2002).

**`D02` — the gate may not read the future.** The K_eff value that *gates the ladder* is computed on
the **pre-first-origin span only (2018-01 → 2020-01)**. The full-sample rolling PR is descriptive
only and may inform no design decision — every origin's test block lies inside it. Pre-register the
trigger numerically before measuring: **if measured PR at K=8 falls below 5.0, re-cut the ladder.**
A gate without a number stated in advance is not a gate.

---

## 6. Model specification

### 6.1 Architecture

Encoder-only iTransformer. Each variate is a token; attention runs across variates, not time.

```
Input  (B, L, N)  →  transpose (B, N, L)
InvertedEmbedding : Linear(L → d_model)          → (B, N, d_model)
Encoder × e_layers:  MHA over N tokens → LayerNorm → FFN(d_model→d_ff→d_model) → LayerNorm
Projection : Linear(d_model → H)                 → (B, N, H)
transpose → (B, H, N) → select target channel    → (B, H, 1)
```

No causal mask. Masking applies to the time axis; this attention runs over the variate axis, where
all tokens are contemporaneous. Causality is enforced upstream, in features and windowing.

### 6.2 Hyperparameters

| Parameter | Value | Note |
|---|---|---|
| `seq_len` L | 96 | 4 days |
| `pred_len` H | 24 | headline; sweep {1, 3, 24, 168} |
| `d_model` | 128 | **attention sequence length is N (≤12), not L.** `d_model=512` over-parameterises against ~14,000 samples (`D25`). State this in the paper |
| `d_ff` / `e_layers` / `n_heads` | 256 / 2 / 8 | ≈ 280k parameters |
| `dropout` / activation | 0.1 / GELU | |
| Optimiser / lr | Adam / 1e-4 | |
| Schedule | **halve every 4 epochs** (`D47`) | type-1 per-epoch halving reaches ~4e-7 by epoch 9, so the budget below could never bind |
| Batch / max epochs / early stop | 32 / 30 / patience 5 on validation MSE | epochs-to-early-stop and final training loss **logged per rung** |
| Loss | **MSE on the target channel only** (`D39`) | at every rung, so the objective is identical across the ladder |
| `use_norm` | **True** | mandatory, not a tuning knob |
| Seeds | **5 at every rung** (42,43,44,45,46) (`D18`, `D49`) | 3 seeds is too few for a `mean ± std` headline — and the 8→12 rung is RQ1's designed contrast, so it cannot carry the fewest |

**Provenance, and why nothing here is tuned (`D38`).** Every value except `d_model` is **adopted
unchanged from Liu et al. (2024)**; `d_model` was reduced from 512 for the sample-size reason in the
table. **No per-rung tuning is performed**, deliberately: holding capacity fixed is what makes the
rungs comparable. §11's checklist item on validation-based hyperparameter selection therefore applies
to **ARIMA order and ridge α only** — those are the two models where selection actually occurs.
State this provenance in §3.4 and Table 3; left unstated, it feeds §13.5's DSR trial count as an
unknown. Because one configuration serves all four rungs, add **one pre-registered robustness run at
K=12 with larger `d_ff`**, so a flat 8→12 rung cannot be read as an under-tuning artefact.

**The loss is single-channel, and this is not a detail (`D39`).** Standard iTransformer
implementations compute the loss over **all N channels**. Under an all-channel loss, K=12 becomes a
12-task problem and K=1 a 1-task problem, so the amount of auxiliary supervision would vary with the
study's own independent variable — and K=1 would not be the stated control but a different learning
problem. The reference implementation defaults to the option that breaks the design, so §11 carries
this as a verifiable assertion.

**K=1 degeneracy is the control, not a bug — with two corrections (`D50`).** At N=1 self-attention
over a single token gives softmax weight 1. iTransformer at K=1 therefore reduces to
`Linear(L→d_model) → [W_O W_V x + x] → LayerNorm → FFN → Linear(d_model→H)` — the value and output
projections and the residual **remain**; it is not a bare identity. Parameter count is identical
across rungs. **State this in the methodology**; unexplained, an examiner reads it as an
implementation error.

**But K=1 vs K=8 does not isolate attention.** The two arms differ in *information* and in *whether
attention is active*, simultaneously. A decaying `A(b)` is equally consistent with "cross-variate
attention overfits regime-specific structure" — a capacity story — as with the information story RQ2
claims, and ridge (D17) separates the two only under a linear model. The clean control is an
iTransformer at **K=8 with attention forced uniform**, which §13.2 already requires for the
interpretability section. It is promoted to a **third arm of the main grid** (§10.2), giving the
second contrast `A_attn(i,b)` in §9.1; reporting both decompositions answers
information-versus-attention directly, at runs Figure 5 needs anyway.

### 6.3 `use_norm` and the scaler

The outer per-channel affine scaler **cancels algebraically** under instance normalisation. With
`z = (x − μ_g)/σ_g`, the instance statistics are `m = (mean_t(x_W) − μ_g)/σ_g` and
`s = std_t(x_W)/σ_g`, so `(z − m)/s = (x − mean_t(x_W))/std_t(x_W)` — μ_g and σ_g vanish. True for
StandardScaler, RobustScaler, and MinMaxScaler alike.

What the scaler *does* control: the **reporting scale** of every metric (MSE is in units of σ_g²),
and **learning for baselines without internal normalisation** (LSTM, ARIMA, ridge). Cross-model
scale consistency matters more than the choice.

**StandardScaler**, for three reasons that belong in the paper: literature comparability (the entire
LTSF literature reports MSE on z-scored data); inertness under `use_norm=True`; consistency across
all models. **RobustScaler is rejected** — on fat tails `σ > IQR/1.349`, so dividing by the smaller
quantity makes outliers *larger* in scaled space and lets crash bars dominate MSE more, which is
exactly wrong for a study about regime behaviour. The fat-tail concern argues for Huber/MAE loss,
not for a different scaler. **MinMaxScaler is excluded on correctness**: bounded by training min/max,
it produces out-of-range values whenever a test regime exceeds the training range — a recurring
defect in the crypto LSTM literature, worth one sentence in Related Work.

**Corrected verification test (`D03`).** The source specification says to multiply the input by 100
and assert identical losses. That cannot pass: the target is a channel of the same array, so it
scales too and the loss scales by `c²`. The correct invariant is

```
MSE(c · x) / c²  ==  MSE(x)          equivalently: RelMSE unchanged
```

Run it before the main grid. If `use_norm` is ever disabled, the entire argument above collapses and
the scaler again affects learning — document the flag state either way.

**No winsorization.** Clipping at ±5σ removes exactly the events driving regime-dependent decay.
State the decision explicitly or a reviewer reads it as an oversight.

---

## 7. Baselines

**Every baseline carries an explicit K (`D40`).** A channel-independent baseline evaluated at an
unstated K cannot speak to the channel-independence debate that §13.1 makes a Related Work pillar.
The comparison an LTSF-literate reviewer wants is iTransformer at K=8 versus PatchTST at K=8 on
identical information; if PatchTST silently ran at K=1, the paper's central architectural comparison
collapses back into univariate-versus-multivariate. Report the K of every model in Tables 3 and 4.

| Baseline | K | Role | Configuration |
|---|---|---|---|
| **Naive-RW** | — | mandatory, EMH baseline | **`ŷ_raw = 0`** in raw log-return space (`D31`) — never the last return |
| Naive-persist | — | secondary | ŷ = last observed return |
| Seasonal-naive | — | daily pattern | ŷ = return at t−24 |
| ARIMA | 1 | classical | order by AIC on the training window |
| LSTM | **8** | RNN | 2 layers, hidden 128, dropout 0.1 |
| **DLinear** | **8** | mandatory | trend–seasonal decomposition + linear |
| **PatchTST** | **8** | SOTA, channel-independent | patch 16, stride 8 |
| **Ridge (multivariate)** | **1, 4, 8, 12** | **added, `D17`** | L2-regularised linear on the same K features, α by validation |

**Naive-RW uses ŷ = 0 — in raw log-return space, and the space is not optional (`D31`).** A random
walk in price implies zero expected return; using the last return gives a weaker baseline and
flatters your results. But §9.1 fixes all metrics on **standardised** log-returns and §6.3 confirms
the target is a channel of the scaled array, `z = (r − μ_g)/σ_g` with `μ_g` fitted on the 21-month
sub-block. Setting `ŷ_z = 0` therefore means `r̂ = μ_g` — the training-window **mean hourly return** —
so the "EMH baseline" would silently be a constant-drift model. This is material, not pedantic — **though smaller than
this section first claimed** (`D52`). Measured across all fifteen origins, `μ_g/σ_g` ranges
−0.00818 … **+0.01733**, the maximum at origin 2025-01; the figure `0.037` written here before
anything was measured is roughly **2× too large**. Its square is then 3.0e-4, about **7.5%** of the
`R²_oos ≈ 0.004` D20 anticipates rather than 35%, and the 24-step tilt is ≈ **0.085σ** of systematic
long bias rather than 0.18σ, in exactly the cumulative signal §13.5 trades on.

The correction does not weaken the argument, and one detail sharpens it: `μ_g/σ_g` **changes sign
across origins** — negative at 2020-01, 2022-12, 2023-05, 2023-10, 2024-03, positive elsewhere — so
the nuisance is not a constant tilt that a reader could mentally subtract. It tracks the bull/bear
cycle H2 invokes as its own mechanism, which is precisely why it is confounded with the effect of
interest and does not wash out. Worse, `μ_g` varies by origin with the bull/bear cycle — the same
cycle H2 invokes as its mechanism — so the nuisance is confounded with the effect of interest and
does not wash out.

Therefore: define the baseline as `ŷ_raw = 0` and map it into scaler space as **`ŷ_z = −μ_g/σ_g`**
for metric computation. Log `μ_g` and `μ_g/σ_g` per origin in `meta/*.json` so the size is auditable.
Use raw-space (drift-free) returns for the §13.5 sign rule and the §9.1 DA definitions.

**Baselines are scored on exactly the same surviving windows (`D45`).** Naive-RW needs no 96-bar
lookback, so unless it is restricted to the window set its comparator actually evaluated, RelMSE is a
ratio across two different samples. Assert equality of evaluated timestamps before computing it.

**Why ridge was added.** K=1 iTransformer controls for *architecture* — it answers "does
cross-variate attention help?" It does not answer "is a transformer needed at all?" Ridge on the
same K features separates *does the information help* from *does attention help*. It costs seconds
per run and closes a question a reviewer will otherwise ask.

**DLinear and PatchTST are not optional.** A missing DLinear is the first thing a reviewer familiar
with the LTSF literature will flag.

---

## 8. Walk-forward protocol

### 8.1 Scheme

Rolling-origin walk-forward with purging.

| Component | Value |
|---|---|
| Training window | 24 months, **fixed** (rolling, not expanding) |
| Training sub-block / validation sub-block | 21 months / final 3 months |
| Purge | H steps at **both** boundaries: train→validation **and** train→test (`D24`) |
| Embargo | not applied — justified in §8.3 |
| Test blocks | 6 × 30 days after the origin, **no retraining** |
| Origin spacing / count | **5 months / 15 origins** — not 6 months / 13 (`D26`) |

**Origin derivation (`D07`, superseded by `D26`).** With the window 2018-01-01 → 2026-08-01, 24
months of training plus 180 days of testing, the earliest origin is **2020-01-01** and the latest
satisfying `o + 180d ≤ 2026-08-01` is **2025-11-01**. At 5-month spacing that gives **fifteen**
origins: 2020-01, 2020-06, 2020-11, 2021-04, 2021-09, 2022-02, 2022-07, 2022-12, 2023-05, 2023-10,
2024-03, 2024-08, 2025-01, 2025-06, 2025-11. Fifteen follows from the **2018-01 start** and the
5-month spacing. The source specification's claim that "spot history starting 2017-08 is what makes
thirteen possible" is false and is deleted; the data in hand begins 2018-01-01. `D07` and `D09`
argued the count was thirteen, not twelve — correct against 6-month spacing, and now superseded:
the spacing itself was the defect.

**Why the spacing is 5 months (`D26`) — this is load-bearing for RQ2, and the arithmetic decides
it.** With origins spaced *s* months and 30-day blocks, the calendar month on which block `b` lands
at origin *i* is `m₀ + s·i + (b−1) (mod 12)`. For fixed `b`, as *i* ranges over the origins, the set
of months visited is a coset of the subgroup generated by *s* in ℤ₁₂, whose size is **12/gcd(s,12)**.
That single expression settles the design:

| *s* | months visited per `b` | origins | consecutive training overlap | worst test block loss |
|---|---|---|---|---|
| 6 (original) | **2** — Jan or Jul | 13 | 75.0% | 50.4% |
| 3 (interleaved) | **4** | 25 | 87.5% | 50.4% |
| **5** | **12 — all of them** | **15** | **79.2%** | **33.9%** |

At *s* = 6, `b` is a deterministic function of calendar month up to a two-phase alternation, month
dummies cannot be added post hoc, and a significant β₁ < 0 is observationally equivalent to
"February and August are harder for microstructure features than January and July" — a bias that no
analysis run after the grid can remove. **Interleaving at 3 months is not the fix**: `gcd(3,12) = 3`
leaves only four phases, so it *reduces* the confound rather than eliminating it. Only *s* coprime
to 12 gives full decoupling, and among those, 5 months maximises the origin count inside the
available span.

**The nominal cluster count is not what denser spacing buys.** Effective independence is bounded by
`total span / training window ≈ 96 / 24 ≈ 4` independent training sets *regardless of spacing* —
packing origins closer inflates G without adding information, while worsening the overlap that §9.2
must disclose. That is why 15 well-separated origins beat 25 tightly-packed ones here.

**Falsification arm, pre-registered.** For every origin, train a **fresh** model at `o_i + 90 days`
and evaluate it on the *same* calendar blocks 4–6 as the aged model. If the aged-minus-fresh gap is
zero while β₁ < 0, then β₁ is calendar, not age. This is the only design that identifies decay
directly; it costs one extra run per origin.

**Rolling, not expanding**, because with an expanding window the training set size changes at each
origin and the effect of model age cannot be separated from the effect of training data volume.
**Caveat (`D45`):** gap density is monotone in calendar time — 26 of 27 downtime blocks fall in
2018–2021 and none after 2023-03 — so per-origin training-window loss runs from **11.2% at origin 6
down to 0.0% at origins 14–15**, and the surviving training count ranges **13,558 … 15,217 windows**.
That partially reintroduces the volume variation the fixed window was chosen to eliminate. Control
for it by **subsampling every origin's training set to 13,558 windows**, the smallest origin's count,
and report the uncontrolled version as the sensitivity. Per-origin figures:
`docs/ORIGIN_WINDOW_BUDGET.md`.

**Cluster dependence (`D28`).** A 24-month window advanced 5 months means consecutive origins share
19 of 24 months — **79.2%** of their training data; origins two apart share 58.3%, three apart 37.5%,
four apart 16.7%. Windows become disjoint only at **stride 5** (25 months apart), so the
training-disjoint subset holds just **3 origins** — {2020-01, 2022-02, 2024-03} or any of the four
parallel triples. The clusters used for inference in §9.2 are therefore **not independent draws**.
State this numerically, never as "calendar adjacency" — see §9.2.

### 8.2 Purge

The `− H` term in window enumeration is the purge: the last retained training window has a target
ending exactly at the boundary, so **no observation is discarded** — only ~24 window configurations
out of 13,558–15,217 (`D25`). The purge and the segment law share their logic: neither discards
observations, both discard window *configurations*.

**Two boundaries, not one (`D24`).** Training windows are enumerated to `val_start − L − H`, so the
last training target ends at `val_start`. The purge applies at the **train→validation** boundary as
well as train→test. Without it, a training window whose H-step target reaches past the 21-month mark
carries validation observations into training — and validation is what decides early stopping
(`patience 5`, §6.2) and ridge α (§7), so the contaminated split is the one that governs *model
selection*. The contamination is small (~24 windows) and it is the defect class §11 calls fatal.

The asymmetry is deliberate: a validation window's 96-bar **input** may reach back into the training
period. That is past information, legitimately available to a forecaster at that moment (§8.3), and
blocking it would make the evaluation unrealistically pessimistic. Only **targets** are purged.

Log the rejection count per origin and assert it against the **per-origin** break table, not against
the pooled §4.3 estimate — see `D45`: origins differ in break count from ~12 down to zero, so a
pooled assertion fires spuriously at most origins and would be loosened until it passes, disarming
the one defence against positional-index drift.

The scaler is fitted on the **21-month sub-block only**, at every origin, never on validation or
test. Moving `train_end` is a leak, not a mismatch.

### 8.3 Why no embargo — the written justification (`D15`)

An embargo guards against test-period information reaching the training set. Two paths exist and
both are closed:

1. **Label overlap** — a training window whose target extends past `T_end` into the test period.
   Closed by the H-step purge.
2. **Feature lookback** — a feature at a test timestamp computed from a window reaching back into
   the training period. This is *past* information, legitimately available to a real forecaster at
   that moment; it is not leakage, and blocking it would make the evaluation unrealistically
   pessimistic. It cannot run the other direction because **no feature uses a rolling window**
   (§5.3): every variate is a per-bar function, so no test-period bar can influence any training-set
   feature value.

The remaining train→test channel is the model weights themselves, which is the object of study.
Hence no embargo. This argument depends on §5.3 and must be re-examined if any rolling feature is
ever introduced.

### 8.4 Rejecting CPCV — include this paragraph in the methodology

> Combinatorial Purged Cross-Validation (López de Prado, 2018) was considered but not adopted. CPCV
> generates backtest paths with non-chronological block ordering, under which *time-since-training*
> — the primary independent variable in RQ2 — is undefined. CPCV also assumes stability of the
> data-generating process across blocks, an assumption this study explicitly tests. Walk-forward was
> chosen because it preserves temporal ordering, remains consistent with evaluation protocols in the
> long-term time series forecasting literature, and still applies purging of H steps at every
> training boundary.

Answer the counter-argument rather than ignoring it: the Knowledge-Based Systems (2024) backtest-
overfitting comparison concludes CPCV beats walk-forward. Its target is *strategy selection* among
many candidates, where block shuffling is desirable; this is a *controlled architecture comparison*
where time-since-training is the independent variable.

### 8.5 Stage gates

| Stage | Gate |
|---|---|
| **2** Data validation | Set the analysis window from the coverage report, not from assumption. **Evaluate window loss per (origin, block), not on the pooled series** (`D45`) — the pooled 4.9% passes while individual blocks lose up to 50%. Loss > 20% in any test block → report the surviving-window count in Table 5 and add block coverage as a regression covariate; never relax the segment rule. Also emit the per-origin break table and the measured `H == L` count |
| **3b** Pre-model measurement | Measured PR at K=8 **< 5.0** → **report the measured PR and proceed unchanged, disclosing the divergence from the expected values in §5.2** (`D48`). The gate's action is disclosure, not a re-cut: D01 establishes that exactly one consistent cut exists, so "re-cut the ladder" named no reachable alternative |
| **5** RQ1 pilot | `use_norm` scale-invariance test first. Then **origin 1 only, 4 K × 3 seeds, evaluated on that origin's validation sub-block** (`D27`). **Gate is K=1 vs K=8, never K=1 vs K=12** — K=12 is built to be redundant, and gating on it would kill a viable paper for the wrong reason. Test: Clark–West at α = 0.05, one-sided (`D29` — the pair is nested, so standard DM is biased against finding the effect). If K=8 does not beat K=1, reposition the title to the descriptive variant now, not in week nine. Also estimate the between-origin dispersion of the within-slope here and publish the **minimum detectable β₁** before any test block is opened |

**The Stage 5 gate runs on validation, not on test (`D27`).** §11's final item requires that test
blocks be opened once, after the design is frozen; a gate that repositions the title on a test-block
result cannot coexist with it. The validation sub-block is the leak-free instrument for a go/no-go on
architecture. It also matters mechanically: §10.5's idempotence plus §10.4's deterministic `run_id`
mean a resumed session finds the pilot runs already complete and feeds them verbatim into Table 4 and
the β₁ regression — so a test-block pilot would put the origin that decided the framing back into the
evidence for it. If a test-block gate is ever genuinely required, designate one origin a burnt
hold-out, exclude it from every table and from the regression, and state the reduced cluster count.
Either way, disclose the pilot in §13.2 as a **selection event**, distinct from the DSR trial count —
the DSR does not correct for selection over a paper's conclusion.

---

## 9. Metrics and statistical tests

### 9.1 Metrics

| Metric | Definition | Role |
|---|---|---|
| MSE | on standardised log-returns | primary |
| MAE | | outlier-robust |
| RelMSE | `MSE_model(b) / MSE_naive(b)` | controls for period difficulty |
| **R²_oos** | `1 − RelMSE` (`D20`) | the readable form of the same quantity |
| DA | directional accuracy | practical relevance |
| **A(i,b)** | `[MSE_K1 − MSE_K8]/MSE_K1` at origin i, block b | **RQ2 dependent variable** |
| **A_attn(i,b)** | `[MSE_uniformK8 − MSE_K8]/MSE_uniformK8` (`D50`) | separates *attention* from *information* |
| **D(i,b)** | `[R²_oos(i,1) − R²_oos(i,b)] / R²_oos(i,1)` (`D05`, rescaled by `D23`) | **RQ3 dependent variable** |
| rung effect | free effect of K in `MSE(i,b,K) = γ_ib + f(K) + ε` (`D32`) | **RQ1 dependent variable** |

**`D05` — why decay is measured against a same-block baseline.** The source definition compares
`MSE(b)` against `MSE(1)` across *different calendar months*, conflating model decay with the market
getting harder. Normalising by the block's own naive baseline removes period difficulty. `A(b)` needs
no such control: both models are evaluated on the same block, so difficulty cancels in the ratio —
and it cancels *well* because `MSE_model` and `MSE_naive` on the same block correlate near 1, which
is the argument that makes the whole ratio-metric design defensible.

**`D23` — why `D(i,b)` is on the skill scale.** On the RelMSE scale the pre-registered τ values are
arithmetically unreachable (§3), so RQ3 would return "no decay detected" by construction. On the
skill scale `D` runs from 0 (no decay) through 1 (edge fully gone) and τ ∈ {2.5%, 5%, 10%, 50%} are
commensurate with it. **Guard the denominator:** an origin with `R²_oos(i,1) ≤ 0` contributes no
`b*` and is excluded, stated as such — never silently dropped.

**`D41` — `b*` carries an origin index and a confidence interval.** `b*(i) = min{b : D(i,b) > τ}`,
**right-censored at 6** when no block crosses. That yields one censored observation per origin, which
is interval-censored survival data: report the **median `b*` with its confidence interval** per τ
from a Turnbull/Kaplan–Meier estimator on the 30-day grid. Test H3 ("larger K decays faster") with a
log-rank test across K, or an interval-censored AFT model with K as covariate. Table 5 carries the
interval, never a bare integer — and the abstract's recommended cadence is that interval.
`min{·}` does not commute with averaging, so pooling MSEs across origins and *then* taking the
minimum is a different estimand and is forbidden.

**`D05` follow-on — do not divide by a single reference block.** `RelMSE(i,1)` is estimated from one
30-day block under heavy tails and volatility clustering, and would sit in the denominator of
`D(i,2)`…`D(i,6)`, making their errors perfectly correlated; because `b*` reads a threshold crossing
straight off the series, an unlucky block 1 moves the crossing by whole blocks. Normalise against the
within-origin mean, `mean_{b'}[·]`, or fit a within-origin trend and read the crossing off the fitted
line, and attach a block-bootstrap band (stationary bootstrap, block length ≥ 24).

**`D32` — RQ1 is a panel comparison, not an OLS on three points.** Four rungs give three ΔMSE values,
and stacking 360 (origin, block, rung) rows creates no information about a K_eff slope that varies
only between rungs — the effective G would be 3, adjacent deltas share an MSE (mechanical correlation
≈ −0.5), and with an intercept there is 1 residual degree of freedom, so the two theories cannot be
distinguished. Two changes make the horse race identifiable. **K_eff is measured per origin** on that
origin's own 21-month training sub-block (§5.4) — leak-free, and it makes the regressor vary. Then
fit `MSE(i,b,K) = γ_ib + f(K) + ε` with (origin × block) fixed effects clustered by origin, estimate
`f(·)` as free rung effects, and compare the K and K_eff explanations as a **non-nested model
comparison** (Vuong, or Davidson–MacKinnon J). Report `corr(K, K_eff)` in Table 2b — across the four
rungs it is ≈ 0.97, and a reader is entitled to know that before reading the comparison.

**Order of operations for seed averaging (`D42` follow-on).** Every ratio metric — `A`, `A_attn`,
RelMSE, `R²_oos`, `D`, ΔMSE — is formed from **seed-averaged MSEs**, never from an average of
per-seed ratios. The two differ by Jensen, and the second additionally requires pairing seed 42 at
K=1 with seed 42 at K=8, which are independent training runs of different models: any of 5!
orderings gives a different answer. State n per cell; the cell mean still carries Monte-Carlo error,
which enters as measurement error in the dependent variable — unbiased for β₁, but inflating residual
variance.

**The log ratio is an appendix robustness check, not a parallel column.** With `A` bounded in roughly
[0, 0.005] given the expected `R²_oos ≈ 0.004` (D20), `log(MSE_K1/MSE_K8) = A + A²/2` differs from
`A` by under 0.1% relative — two columns identical to five decimal places. Use linear `A` throughout;
revisit only if `A` empirically spans an order of magnitude, and state that trigger now.

**Report two metric scales.** MSE in scaler space for literature comparability, RMSE in raw
log-return units for interpretability, with the conversion factor σ_g stated so the two reconcile.
"RMSE 0.0043 on hourly log-returns" tells a reader far more than "MSE 0.187 on normalized data".

**MAPE is forbidden on log-returns** — the denominator explodes near zero. On reconstructed prices it
is dominated by random-walk behaviour and uninformative. The Lewis (1982) thresholds common in
Indonesian journals do not apply here; say so once in Related Work.

**Directional accuracy at H=24 (`D21`).** Report DA at step h=1, at step h=24, and on the
**cumulative 24-hour return** — without a null hypothesis, DA is a descriptive number. The three
variants do **not** share a testing regime. DA at h=1, on hourly-spaced forecasts, is tested with
Pesaran–Timmermann (1992). DA at h=24 and DA on the cumulative return are computed on
**non-overlapping** windows matching the §13.5 execution rule, and tested with PT on that sample; on
hourly spacing their targets overlap by 23 of 24 hours, giving lag-1 autocorrelation ≈ 23/24, so
PT's variance is far too small and the test over-rejects badly. The overlapping-window versions are
reported as **descriptive only, without p-values**. Note the resulting power loss (T = 30 per block)
explicitly rather than recovering it by using the invalid sample.

### 9.2 Mandatory tests

**Diebold–Mariano for every comparative claim — but not the same DM for every pair (`D29`).** The
comparisons that carry the paper are **nested**: the ladder is cumulative, so K=1's feature set is a
strict subset of K=8's under the same architecture and sample; Naive-RW (`ŷ = 0`) is nested inside
every model in §7; Ridge-K1 inside Ridge-K8. Under the null of equal population predictive ability
with nested models and estimated parameters, the loss differential has a mean shifted away from zero
— the larger model's extra estimation noise makes it look worse under the null — and the statistic is
not asymptotically N(0,1) (Clark & McCracken 2001; McCracken 2007). Standard DM is therefore
systematically **undersized against the alternative this study exists to establish**, and the Stage 5
gate would turn a title decision on it.

| Pair type | Statistic |
|---|---|
| **Nested** — K=1 vs K=8, any model vs Naive-RW, Ridge-K1 vs Ridge-K8 | **Clark–West (2007)** adjusted statistic (add `Σ(ŷ_small − ŷ_large)²` back to the differential), or Clark–McCracken ENC-NEW/MSE-F with their non-standard critical values. Name which |
| **Non-nested** — iTransformer vs DLinear vs PatchTST vs LSTM | standard DM with the HLN correction below |

Acknowledge Diebold (2015), which argues DM remains valid when *forecasts* rather than models are the
object, and take a position rather than leaving it silent — a reviewer will know both sides.

**The variance estimator is rectangular, not Bartlett (`D34`).** At H > 1 forecast errors overlap, so
use the **truncated (rectangular)** long-run variance estimator
`V̂(d̄) = [γ̂₀ + 2Σ_{k=1}^{h−1} γ̂_k]/T` with lag `h−1` (lag 23 at H=24), as in Diebold–Mariano (1995)
and as implemented in `forecast::dm.test`. **Do not use Newey–West Bartlett weights here**: under the
DM null, h-step optimal forecast errors are MA(h−1), so all autocovariances to lag 23 are genuinely
nonzero and equally real, and Bartlett weights shrink γ̂₂₂ by ~92% — understating the long-run
variance and producing exactly the over-optimistic p-values this paragraph exists to prevent.
`statsmodels` `cov_hac` is Bartlett by default, so a literal reading of "Newey–West" fails the
validation prescribed in the next sentence. The rectangular estimator is not guaranteed positive in
finite samples: if `V̂ ≤ 0`, fall back to Bartlett with automatic bandwidth and **report that the
fallback fired for that pair**.

Apply the Harvey–Leybourne–Newbold small-sample correction

```
S* = S · sqrt[ (T + 1 − 2h + h(h−1)/T) / T ]
```

and compare `S*` against Student-t with `T−1` degrees of freedom, **not** the standard normal.
Validate any custom implementation against R's `forecast::dm.test()`.

**Pin the DM sample, or Table 6 is not reproducible.** DM is computed **per (origin, block)** on the
overlapping hourly loss differential: `T ≈ 720`, `h = 24`, truncation lag 23. Block-level statistics
are combined across the (origin, block) cells by a stated method — **never** by concatenating `d_t`
across origins, because the model changes at each origin and the DM null has no interpretation across
that boundary. Assert `(T + 1 − 2h + h(h−1)/T) > 0` before applying the HLN factor and refuse to
report where it fails: at `h = 24` the factor is exactly 0 at `T = 24` and 0.047 at `T = 30`,
precisely the T a non-overlapping 30-day block would produce. State T alongside every reported
p-value.

**Multiplicity: SPA and Reality Check are the wrong tool for a pairwise matrix (`D35`).** White's
Reality Check (2000) and Hansen's SPA (2005) test a *one-against-many* null — "no model in the
candidate set beats a single designated benchmark" — and return one p-value for that composite. They
say nothing about the all-pairs comparisons Table 6 contains; with 8+ models that matrix holds 28+
tests and at α = 0.05 expects ~1.4 spurious rejections under a complete null. Therefore:

- **pairwise matrix** → **Romano–Wolf (2005) stepdown**, which controls FWER across all pairs and is
  bootstrap-based like the machinery already in the pipeline;
- **"which models are indistinguishable from the best"** — what a reader actually wants from Table 6
  → **Model Confidence Set** (Hansen, Lunde & Nason 2011) at 90% and 75%, as a membership column;
- **SPA retained only** where the paper genuinely poses a one-against-many null ("does any
  iTransformer configuration beat Naive-RW"), labelled as answering that specific question.

**Decay regression — the paper's core claim (`D06`).** The naive specification is wrong:

```
correct:   A(i,b) = αᵢ + β₁·b + ε          ← origin fixed effects
wrong:     A(b)   = β₀  + β₁·b + ε
```

Without `αᵢ`, β₁ absorbs origin-level difficulty. Four requirements:

1. **Origin fixed effects**, so β₁ is identified from within-origin variation across blocks.
2. **Cluster by origin.** Blocks within an origin come from one trained model, so ordinary standard
   errors badly overstate precision. **The effective n for β₁ is the cluster count, not the
   observation count (`D42`).** The panel is balanced (`b = 1…6` in every origin), so with origin
   fixed effects β̂₁ reduces algebraically to the simple mean of the origin-specific within-slopes:
   `β̂₁ = (1/G)·Σ_g β̂₁,g`. Inference on the paper's core claim is a one-sample test on **G = 15**
   numbers. Citing "15 × 6 = 90 observations" invites the reader to infer power that does not
   exist — state both, and state that effective independence is bounded near 4 by the training-window
   overlap (§8.1).
3. **Wild cluster bootstrap — the complete recipe (`D42`)**, because every unstated choice moves the
   p-value and §12 requires the number to be regenerable:
   **restricted (WCR — impose H₀ when generating samples**, not unrestricted; WCU is severely
   size-distorted at small G, MacKinnon, Nielsen & Webb 2023**)**, bootstrapping the **cluster-robust
   t-statistic** (not β̂ — the asymptotic refinement comes from bootstrapping *t*, Cameron, Gelbach &
   Miller 2008), **B = 99,999**, cluster = origin, **one-sided** test of H₁: β₁ < 0 at **α = 0.05
   declared in advance** (a side chosen after seeing the sign is not pre-registered).
   `wildboottest` with `impose_null=True` in Python, `fwildclusterboot` in R.
   **Report both Rademacher and Webb 6-point weights**; if they disagree, the more conservative is
   the headline. The original justification for preferring Webb never bound and binds even less at
   G = 15: Rademacher admits 2¹⁵ = 32,768 distinct draws, a minimum two-sided p ≈ 6·10⁻⁵, far below
   α = 0.05. At G = 15 the literature's small-G regime (G ≲ 12) is no longer the binding concern —
   which is a genuine side-benefit of `D26`'s re-cut, and one more reason it beat the 6-month grid.
4. **Reference distribution and rank.** Cluster-robust *t* is referred to **t(G−1) = t(14)**, never
   the software default t(N−K) — here t(14) versus t(74), a ~9% difference in critical value at 5%
   two-sided. With 15 origin dummies plus a slope the CRV meat matrix has rank ≤ 15 against 16
   parameters, so it is singular: **only single-coefficient tests are reported**, never joint ones.
5. **Average over seeds first.** Seeds are computational noise, not population draws. Treating them
   as independent observations inflates n and is not defensible. See §9.1 for the order of
   operations on ratio metrics — seed-averaged MSEs first, ratio second, never the reverse.
6. **Publish the minimum detectable β₁ before opening the test blocks.** Every design choice here
   (5 seeds, the origin count, the bootstrap) implies someone reasoned about precision; no number is
   written down. Estimate the between-origin dispersion of the within-slope from the Stage 5 pilot,
   compute the MDE at the actual G by simulation, and **pre-register the interpretation of a null**.
   If the MDE exceeds the plausible magnitude of `A`, reposition RQ2 as descriptive *before* running
   the grid, not after — otherwise a non-significant β₁ is indistinguishable from a design that could
   never have detected decay.

**Disclose the residual dependence — and state the larger mechanism, not the smaller one (`D28`).**
The wording "origin *i*'s block 6 is calendar-adjacent to origin *i+1*'s block 1" names the weaker
problem. The stronger one is structural: with a 24-month window advanced 5 months, consecutive
origins train on **79.2% identical data** (58.3% at stride 2, 37.5% at stride 3, 16.7% at stride 4),
and each origin's test period lies **entirely inside** the training window of later origins. Two
consecutive origins' models are therefore near-identical fits to near-identical data, so their
`A(i,b)` series are dependent by construction — not merely through shared volatility. Cluster-robust
inference assumes independence *between* clusters; here the effective number of independent training
sets is bounded near **4** (§8.1), far below G = 15, and the bootstrapped p-value on β₁ is
**anticonservative by an unquantified amount**. State the overlap fractions numerically in
Methodology and Limitations, and report a **robustness re-estimate of β₁ on the training-disjoint
subset** — stride 5, i.e. {2020-01, 2022-02, 2024-03} or any of the four parallel triples, **G = 3**.
Report all five triples and their spread; at G = 3 the estimate will very likely be inconclusive, and
**that is itself the finding** — it bounds what the full-panel p-value can honestly claim. A
moving-block bootstrap over calendar time is the acceptable alternative to an i.i.d.-over-clusters
bootstrap; report which was chosen and **state the cluster count explicitly**.

**Also add block coverage as a covariate, or re-run β₁ on well-covered blocks only (`D45`).**
Test-window survival is conditioned on *future* gaps and outages cluster on stress, so within an
origin the surviving sample composition trends — and β₁ would absorb it.

**Reporting: bind the dispersion measure to the aggregation level (`D30`).** Seed dispersion measures
re-initialisation noise on one fixed dataset; origin dispersion measures the sampling variability of
the estimand, and in walk-forward crypto evaluation the second is typically an order of magnitude
larger. Reporting seed-std as "±" on an origin-aggregated row understates the headline uncertainty by
roughly that factor — reintroducing, through the reporting convention, exactly the overstated
precision the wild cluster bootstrap was added to prevent. Therefore:

- **per-cell (origin, block)** numbers → mean ± std across seeds, **with n stated**;
- **any number aggregated across origins** (Table 4 included) → mean ± **standard error across
  origins**, or a cluster-bootstrap CI, with seed-std reported separately as a Monte-Carlo-noise
  diagnostic column.

**The inferential unit is the origin. Seed dispersion is a diagnostic, never the uncertainty on an
aggregated estimate. Never a single number.**

---

## 10. Execution — Kaggle 2×T4

### 10.1 Envelope

| Limit | Value | Consequence |
|---|---|---|
| Session runtime | **12 h** | Self-imposed budget stops earlier so the version saves |
| GPU quota | **30 h / week / account** | The whole grid must fit in roughly one week |
| Idle timeout (interactive) | 20 min | Grid execution uses *Save Version → Save & Run All*, never the editor |
| `/kaggle/working` | 20 GB, saved as version output | Predictions total ≈ 0.5–2 GB — fits with room |
| `/kaggle/input` | read-only | Resume reads from here; everything is *written* to `/kaggle/working` |
| GPUs | 2 × T4, sm_75, 16 GB each | See §10.3 |

### 10.2 Run accounting

Derived against **15 origins** (`D26`, §8.1).

| Block | Count |
|---|---|
| Main grid — 15 origins × 4 K × **5 seeds** (`D49`) | 300 |
| Uniform-attention arm — 15 × K=8 × 5 seeds (`D50`) | 75 |
| Falsification arm — fresh model at `o_i + 90 d`, every origin (`D26`) | 15 |
| Horizon sweep — **origins 1, 5, 10, 15** × 4 K × 4 H × 3 seeds (`D08`, `D48`) | 192 |
| Baselines — 15 × (4 deterministic + 3 stochastic × 3 seeds), **each at K=8** (`D40`) | 195 |
| Ridge — 15 × 4 K (`D17`) | 60 |
| **Total** | **≈ 837** |

**`D49` folds the old "extra seeds" row in**: 5 seeds now run at *every* rung, not only K∈{1,8}. The
8→12 rung is RQ1's designed contrast and cannot carry the fewest seeds. **`D48` names the four sweep
origins in advance** — choosing them after the main grid would be origin selection.

**Superseded twice, and the table above is the nominal count, not the run count.** `D53e`
deduplicates the sweep's H=24 slice against the main grid, giving **789** real runs — 534
iTransformer + 195 baselines + 60 ridge — and the 534 iTransformer half is what actually executed on
2026-08-08. `D57` then replaced the timing this paragraph rests on. Both are recorded below rather
than by editing the arithmetic away, because the trade-offs elsewhere in this document were made
against these figures and a reader needs to see which ones moved.

**Quota check, as written and now falsified.** At the §10.3 regime (~60–100 s per run, two GPUs as independent workers), 837 runs
land at **≈ 7–12 wall-hours**, plus the horizon sweep's H=168 cells and the ARIMA/LSTM baselines,
which are heavier — call it **10–20 h**. That fits inside one 30 h weekly quota with room for a
re-run, across **two sessions** at the 11 h self-imposed budget. The grid grew from ≈ 621 to ≈ 837
(+35%) because `D49` raised seeds at every rung, `D50` added the attention control, `D26` added the
falsification arm and two origins. It still fits; if a future addition does not, drop the horizon
sweep to 3 origins before touching seed counts, because the seed counts are what `D30` and `D49`
depend on.

`D08` resolves the source specification's 4-K-vs-3-K conflict in favour of **4**, matching its own
arithmetic (4 × 4 × 4 × 3 = 192).

### 10.3 Cost model — and the regime it depends on (`D19`)

Per origin the training tensor is at most `15,217 × 96 × 12 × 4 B ≈ 70 MB` (`D25` — the 21-month
sub-block, not the 24-month window); targets add ~1.5 MB. The count **varies by origin**, 13,558 to
15,217, because gap density is monotone in calendar time (`D45`, §8.1) — size the buffer at the
maximum and slice per origin. **It fits entirely in a T4's 16 GB many times over, and must be
resident there.**

**The required regime:** load the whole split to GPU once, then batch by index-slicing that tensor.
**Do not construct a per-item `Dataset` and `DataLoader`.** At this model size (~280k parameters,
**~420–475 steps/epoch** at batch 32) the compute is trivial and the run is dominated entirely by
data movement and Python overhead.

| Regime | Per run | Whole grid |
|---|---|---|
| **GPU-resident, no DataLoader** | **~30 s measured** (`D57`) | **2.31 h** at 534 runs on two T4s; **~4.5 h** in one kernel |
| Naive `DataLoader`, 4 workers | ~10× worse | ~45 h — **exceeds the weekly quota outright** |

Both numbers are stated so the regime is understood as load-bearing, not stylistic.

**The estimate was 2–3× pessimistic per run and 4–8× overall (`D57`).** The row above read
"~60–100 s" and "≈ 10–20 wall-hours" until the grid actually ran on 2026-08-08: 534 runs, two T4
workers, **2.31 h wall**, mean 31.6 s and 28.3 s per run on the two devices. Nothing about the
*regime* was wrong — the GPU-resident path is what made it fast — only the arithmetic on top of it.
Two consequences follow, and the second is larger than it looks:

- the weekly quota was never the binding constraint it was written to be, so a second granularity or
  an extra arm is affordable in a way §10.2 assumed it was not; and
- **two GPU workers stopped being necessary**, which is what let §15's notebook drop to a single
  kernel. At ~4.5 h sequential the grid fits the 11 h budget with room, so the second T4 now buys
  wall-clock rather than feasibility.

**First real measurement, 2026-08-06** — `itr_o01_K08_H024_s42`, the run this table's estimate was
written for:

| Quantity | Measured |
|---|---|
| Wall time | **97.8 s**, 10 epochs (early stop), 13,924 training windows, 436 steps/epoch |
| Per epoch | **9.8 s** — the transferable number; wall time depends on where early stopping lands |
| Device | **CPU, 6 threads.** No CUDA device was available locally |
| Parameters | 280,472, matching §6.2's "≈ 280k" exactly |

**The T4 measurement has now been taken, as that block instructed (2026-08-08, `D57`).** The full
grid ran on 2 × T4: **534 runs, 2.31 h wall**, mean **31.6 s** on `cuda:0` over 263 runs and
**28.3 s** on `cuda:1` over 259, zero failures, zero skips. So the CPU figure above was ~3× the T4
figure, and the T4 is ~3× faster than the estimate this table was built on.

The regime is confirmed exactly as specified — GPU-resident, no `DataLoader`, whole training split
42.8 MB at K=8, 280,472 parameters. What is **not** confirmed is the arithmetic layered on it: the
weekly quota was never close to binding, and that slack is what §15's single-kernel notebook spends
(`D58`) and what makes a second sampling granularity affordable at all.

**The first result, recorded because §12 requires every number be regenerable.** On origin 1's six
test blocks: `MSE_model = 1.3194`, `MSE_naive = 1.2956`, **`RelMSE = 1.0183`, `R²_oos = −0.0183`** —
the model is 1.8% *worse* than Naive-RW. One origin, one seed, one rung out of 837 runs, so it is
evidence about nothing in §3 yet. It is worth stating for one reason: `D20` anticipates
`R²_oos ≈ +0.004`, and the first measurement is negative and four times larger in magnitude. If that
survives the grid, RQ2's `A(i,b)` is a ratio of two negative skills and §9.1's guard on
`R²_oos(i,1) ≤ 0` stops being an edge case and becomes the common case. Watch it.

**Parallelism belongs at the *run* level, never the batch level** — the grid is many small runs, not
one large one. **`nn.DataParallel` is rejected**: at batch 32 the scatter/gather transfer costs more
than the split saves.

**Run-level parallelism is now optional, and the notebook does without it (`D58`).** Two independent
workers, one pinned per `cuda:N` off a shared queue, is what produced the 2.31 h measurement and
remains the fastest way to execute the grid. It is no longer *required*, because `D57`'s numbers put
the sequential path at ~4.5 h inside an 11 h budget. That mattered because workers are
**subprocesses**, a subprocess inherits none of the kernel's namespace, and §15's notebook now
carries the package as definitions in that namespace rather than as files on disk — so a subprocess
could not reach the code at all. The trade was made deliberately: roughly 2 h of wall-clock bought
the removal of the materialise-then-import step and everything that could go stale inside it.

`launch_workers` stays in `runner.py` and stays tested. It is the path to take from a checkout,
where the package *is* importable, and it is what a future granularity grid should use if the
sequential figure ever stops fitting the budget — at 1-minute bars it will not fit.

**Precision.** T4 is sm_75. `torch.cuda.is_bf16_supported()` defaults to
`including_emulation=True` and returns **True** there, selecting *emulated* bf16 that is slower than
fp32. Gate on `torch.cuda.get_device_capability(0)[0] >= 8` instead, falling back to fp16 +
`GradScaler`, then fp32. At this model size fp32 is likely fastest — measure before choosing.

### 10.4 Run identity and outputs

```
run_id = {model}_o{origin:02d}_K{K:02d}_H{H:03d}_s{seed}      e.g.  itr_o07_K08_H024_s42
```

Deterministic and human-readable. Changing any component deliberately **orphans** prior outputs
rather than silently reusing a mismatched result.

| Artifact | Content |
|---|---|
| `preds/{run_id}.parquet` | `block, step, timestamp, y_true, y_pred` — **raw predictions, always** |
| `meta/{run_id}.json` | resolved config, git sha, input-artifact sha256, epochs run, best val, wall time, `status` |

**Persist raw predictions, not just metrics.** They are required for the DM test, per-regime
analysis, and the economic evaluation. Re-running 837 experiments because predictions were not saved
is an expensive, avoidable mistake.

### 10.5 Continuation across sessions

**Idempotence.** A run is complete **only when both files exist and `meta.status == "complete"`.**
Anything else is re-run from scratch. Intra-run checkpointing is deliberately omitted: at **~30 s**
per run measured (`D57`) it costs far more complexity than it saves — the figure was written as
~90 s and the real one makes the argument three times stronger.

**Resume.** Discover completed `run_id`s by globbing `/kaggle/input/*/preds/` ∪
`/kaggle/working/preds/` — **never a hard-coded dataset slug**, so the Kaggle Dataset name is free to
change. Subtract them from the manifest and execute the remainder. A resumed session therefore
performs no wasted work and needs no manual bookkeeping.

**Budget guard.** `SESSION_BUDGET_H = 11.0`, `RESERVE_H = 0.5`, checked **at run boundaries**, not
epoch boundaries — runs are short, epochs are shorter, and the checkpoint granularity is the run. On
trip: stop, flush, print the remaining count and the estimated sessions left, exit cleanly so the
version saves. **Hitting Kaggle's own 12 h wall interactively loses `/kaggle/working` entirely.**

**The guard measures the session, not the worker (`D54f`).** `BudgetGuard` sets its deadline from
`time.perf_counter()` where it is constructed, but the 12 h wall runs from the notebook's first
cell — so the prelude (Stage 2, Stage 3b, Stage 4, and the *twelve pilot training runs* of Stage 5,
call it 20–25 minutes) would sit outside the budget and the two clocks would drift apart by exactly
that much. The notebook stamps `SESSION_T0` in cell 0 and hands the grid what is **left** of the
11 h, not a fresh 11 h. This survived the `D58` flattening unchanged: the arithmetic never depended
on where the grid ran, only on when the session started.

**A partial session is the expected case, and it ends cleanly (`D54e`).** The grid is ~10–20 wall
hours against an 11 h budget, so two sessions is the plan and not the exception. Three properties
make that safe, and the third had to be added:

1. **Resume granularity is one run**, ~30 s measured (`D57`). A run is complete only when both
   artifacts exist and `meta.status == "complete"`, so an interrupted run leaves no meta and is
   simply redone — the loss is at most one run.
2. **Discovery is by glob**, `/kaggle/input/*/preds` and `/kaggle/input/*/*/preds`, so the previous
   session's output Dataset is found under whatever name it was given and however Kaggle nested it.
   Verified against both layouts.
3. **Evaluation is gated on grid completeness.** A partial grid is an unbalanced panel and §9.1's
   estimators refuse one by design; `amplification` raises rather than compare K=1 at eleven origins
   against K=8 at ten. That is correct and stays. What was wrong is that the exception landed in the
   last cells of a twelve-hour session, marking the version failed at the moment its output was the
   only thing worth keeping. The estimators are therefore **not called** until the panel exists.
   Partial evaluation is never the fallback: a half-panel β₁ is a different estimand, not a noisier
   one.

**Session chaining.** Session *N* writes to `/kaggle/working`; Save Version publishes it as a
Dataset; session *N+1* attaches that Dataset as input. Quota arithmetic: at ~6–15 h for the full
grid, the 30 h weekly budget absorbs one complete pass plus a re-run, in one or two sessions.

**What is attached, and what is not (`D54`).** Exactly two kinds of Dataset: the **immutable data
artifact**, and the **previous session's output** when resuming. The repository is *not* attached —
`notebooks/iTransformer.ipynb` carries the package as definition cells and needs nothing on disk
(§15, `D58`). Uploading the
repository as a second Dataset was the old protocol and its failure mode was silent: the notebook and
the code Dataset were two artifacts required to agree, with nothing checking that they did, so a
notebook updated without re-uploading the code ran last week's package and said nothing. The parquet
is likewise never re-downloaded inside the notebook even though Stage 1 could: a fresh download is a
new vintage, and §12 forbids numbers from two vintages sharing a table.

---

## 11. Anti-leakage checklist

Verify before the main grid. **The first ten are fatal.** The list is organised by the
boundary × channel grid of §5.3 (`D43`), because a two-path enumeration missed four channels
entirely and the checklist that rested on it returned green ticks where the pipeline leaks.

**Features and labels**

- [ ] **F** — Log-returns computed **per segment**, before the scaler is fitted
- [ ] **F** — Windows validated by **timestamp**, not positional index; no window spans a break
- [ ] **F** — **No imputation anywhere**: no ffill, no bfill, no interpolation, no reindexing to a full grid.
      Runnable form (`D33`): assert `parquet_rows == bars_actual` and assert the timestamp diff set
      contains the 27 gap blocks. The old "drop rows flagged synthetic" test cannot run — there is no
      flag column on the artifact
- [ ] Zero-volume and `H == L` bars excluded and treated as segment breaks (`D14`), with the
      **measured** `H == L` count emitted at Stage 2 — it is currently assumed, not measured
- [ ] No rolling window on any feature — verify by inspection, since §5.3 makes this structural

**Scaler**

- [ ] **F** — `StandardScaler` refit at every origin, on the 21-month training sub-block only
- [ ] Identical scaler and scale space for iTransformer and every baseline
- [ ] **F** — Naive-RW mapped as `ŷ_z = −μ_g/σ_g`, not `ŷ_z = 0` (`D31`); `μ_g`, `μ_g/σ_g` logged per origin

**Boundaries — train/validation, train/test**

- [ ] **F** — H-step purge active at **every boundary between disjoint splits**: train→validation
      **and** train→test (`D24`)
- [ ] **F** — `max(target_index over training windows) < val_start` and
      `max(target_index over validation windows) < test_start` (`D24`) — window-span assertions, not
      row-index ones. Row-level split disjointness is true by construction and does not imply it
- [ ] Per origin, `train ∩ val == ∅` and `(train ∪ val) ∩ test == ∅` at row level. **Cross-origin row
      overlap is by design** (`D28`) and is handled inferentially in §9.2 — do not "fix" it
- [ ] **F** — The training-window count matches the 21-month arithmetic minus logged gap losses
      (`D25`) — 13,558–15,217 per `docs/ORIGIN_WINDOW_BUDGET.md`, not ~17,400

**Model selection**

- [ ] **F** — The test blocks are opened once, after the design is frozen — **and the Stage 5 gate
      runs on validation, not on test** (`D27`)
- [ ] Hyperparameters: **ARIMA order and ridge α only** are selected on the validation sub-block;
      every iTransformer hyperparameter is fixed a priori and identical at every rung (`D38`)
- [ ] **F** — Loss is MSE on the **target channel only**, at every rung (`D39`)
- [ ] **F** — `use_norm=True` confirmed active; `MSE(c·x)/c² == MSE(x)` passes (`D03`)

**Effective dimensionality**

- [ ] **F** — **Every reported K_eff, including the RQ1 regressor**, is computed on a training-only
      span; the gate additionally uses the pre-first-origin span (`D02`, `D44`). Auditing only the
      gate leaves RQ1's regressor free to read the test period

**Evaluated-sample composition**

- [ ] Rejected-window count logged per origin and reconciled by **exact equality** against the
      per-origin break table — `rejected(origin) == 119 × breaks(origin) + missing(origin)` — not
      against the pooled estimate (`D45`), which is wrong at fourteen of fifteen origins
- [ ] Surviving-window counts reported per (origin, block); block coverage entered as a regression
      covariate or β₁ re-run on well-covered blocks (`D45`)
- [ ] Every baseline evaluated on **exactly** the surviving window set of the run it is compared
      against — assert equality of evaluated timestamps before computing RelMSE (`D45`)
- [ ] No P&L return computed across a segment boundary (`D46`)

**Traceability**

- [ ] Split indices persisted as JSON
- [ ] Raw predictions saved for every run

**If a result looks too good, it is leakage until proven otherwise.** The correct response to an
unexpectedly high score is to hunt for the leak, not to celebrate.

---

## 12. Traceability contract

**No number enters the manuscript unless it is regenerable.** Concretely, every reported figure must
resolve to:

1. a **persisted prediction file** — `preds/{run_id}.parquet`;
2. a **config hash** — the `meta/{run_id}.json` entry naming the code and the sha256 of the input
   feature artifact; and
3. a **documented decision** — a divergence-register row, if the number depends on any departure
   from the source specifications.

**Item 2 names the code by digest, not only by commit (`D54`).** The git sha was the whole of it
until 2026-08-07, and it is `"unknown"` on Kaggle — there is no git repository there, which is to say
the contract lost its code half at exactly the place the grid executes. `meta/*.json` therefore
carries **`code_sha256`**, the hash of the package's own source with line endings normalised, beside
`git_sha`.

**Off-repo the digest is pinned, not computed (`D58`).** `code_sha256()` hashes the `*.py` beside
itself, and §15's notebook has no files at all — the modules are definition cells. So the generator
computes the digest from `src/itransformer_btc/` and pins it as `CODE_SHA256_OVERRIDE`, which
`code_sha256()` returns unchanged when set. This is weaker in one specific way and it must be stated
rather than glossed: a computed hash cannot lie about the code beside it, whereas a pinned one is
only as honest as the generator that wrote it. That is what `tests/test_notebook_sync.py` exists to
enforce, asserting every cell equals its module under the declared transformation — the digest and
the cells are checked by the same run, so a notebook carrying a stale digest carries stale cells too
and fails there first. It answers the same question and answers it better: it identifies the code that ran, not
the commit someone was standing on with a dirty tree. Likewise `input_sha256` resolves through the
**`ITBTC_PARQUET`** environment variable rather than a repository-relative path, and
`input_sha256_source` records whether the digest came from the Stage 1 report beside the artifact or
from hashing the parquet directly. Before `D54` both fields logged `"unknown"` on every Kaggle run,
so this section was unenforceable in practice while reading as though it were not.

Aggregation writes `artifacts/paper_numbers.json`, and every table and figure is generated *from that
file* rather than transcribed. **Numbers produced under different input-artifact hashes are not
comparable and must not share a table.** A number that cannot be regenerated is a documented failure,
not a footnote.

---

## 13. Paper production

### 13.1 Structure

IMRaD, 10–14 pages. Abstract 200–250 words and **must contain concrete numbers** — the β₁ value, the
percentage decay, the recommended cadence. An abstract without numbers reads as a proposal.

`1 Introduction` · `2 Related Work` (architectures; the channel-independence debate; crypto DL;
evaluation protocols; preprocessing practice; gap synthesis) · `3 Methodology` (3.1 provenance and
segmentation · 3.2 variates and effective dimensionality · 3.3 efficiency tests and pre-model
measurement · 3.4 architecture · 3.5 baselines · 3.6 walk-forward and purging · 3.7 metrics and
clustered inference) · `4 Results` (4.1 data and efficiency · 4.1b effective dimensionality ·
4.2 RQ1 · **4.3 RQ2 — core of the paper** · 4.4 RQ3 · 4.5 horizons · 4.6 attention · 4.7 economics ·
4.8 limitations) · `5 Conclusion`, answering all three RQs with numbers.

### 13.2 Mandatory disclosures

Each of these is a place a reviewer will otherwise find a hole:

- the **K=1 attention degeneracy** as a designed control (§6.2);
- the **K=12 rung's deliberate redundancy** as a designed contrast (§5.2);
- the **CPCV rejection** paragraph, verbatim (§8.4);
- the **`use_norm` / scaler** relationship and the corrected invariance test (§6.3);
- the **no-imputation** defence and why Rubin's taxonomy does not apply (§4.2);
- the **instance-normalisation confound** on F2 and RQ1 (§5.4, `D04`);
- **attention is not explanation** — present attention maps as *descriptive evidence of variate
  reliance*, validated for stability across seeds, never as causal explanation. Minimum bar: seed-
  stability reporting plus a uniform-attention ablation (Jain & Wallace 2019; Wiegreffe & Pinter
  2019). Note that the debate is scoped to RNN-era NLP and its transfer to variate-level attention
  in LTSF is genuinely open — that is itself a limitation sentence;
- **every departure from the source design**, with its reason, per §12;
- **revised-vintage data**: this dataset is not real-time vintage; state it as a known limitation;
- **statistical power** — the minimum detectable β₁ and the pre-registered interpretation of a null
  (§9.2 requirement 6). Omitting it is the most damaging gap on this list: every design choice
  implies someone reasoned about precision, and no number was written down;
- **single-venue, single-pair scope** — everything is Binance BTCUSDT, so the microstructure variates
  (`taker_buy_ratio`, `log_trade_count`, `vwap_location`) are **venue-specific** and results may not
  transfer to another exchange. This is an external-validity limit distinct from revised vintage;
- **hyperparameter provenance** — adopted from Liu et al. (2024), not tuned, identical at every rung
  (`D38`), and what that implies for the flat 8→12 rung;
- **study-level multiplicity** — declare which tests are *confirmatory* (β₁ at τ = 5%) and which are
  *exploratory* (τ sensitivities, horizon sweep, per-rung DM cells). §9.2's Romano–Wolf and MCS cover
  the model matrix only; nothing currently governs the family spanning three RQs, four τ values, four
  horizons and the DM matrix;
- **the Stage 5 pilot as a selection event** (`D27`), stated separately from the DSR trial count;
- **the training-window overlap between origins** and the effective cluster count (`D28`);
- **the future-conditioned exclusion of test windows** near outages (`D45`).

**Priority claims are hedged and documented.** §3's contribution (1) reads "first walk-forward
evaluation of iTransformer on a crypto asset with explicit decay measurement". Written flat, it is
refutable by a single search hit, and §13.3 already declares every entry in the reference library
unverified — so the evidence base for the claim is acknowledged as unverified in the same document
set. Write **"to the best of our knowledge, the first …"** and add a one-paragraph search protocol to
§2: databases queried, exact query strings, search date, inclusion criteria, hits screened. Then let
the weight sit on the substantive contribution — explicit decay measurement with clustered inference
under a pre-registered threshold — which stands whether or not someone else has run iTransformer
on BTC.

### 13.3 Citation discipline

**No citation enters the manuscript without a verified DOI and the source read.** Both source `.md`
files self-declare entries "assembled from memory"; treat every one as unverified until cleared.
Known-bad entries found so far (`D16`): arXiv 2509.23494 is dated 2026 but its identifier is 2025;
*Symmetry* vol. 18 is dated 2025 but vol. 17 is the 2025 volume. Target ≥ 60% of references from the
last five years; methodological classics are exempt. Use Zotero or Mendeley from the start.

### 13.4 Tables and figures

Each binds to an artifact and a stage. **Figure 3 carries the entire paper** — if only one figure
appears in a graphical abstract, it is Figure 3.

| # | Table | # | Figure |
|---|---|---|---|
| 1 | Dataset, gap profile, **per-origin** windows lost, `H == L` count | 1 | Walk-forward scheme with purging and segments |
| 2 | Descriptives + ADF + VR + Hurst | 2 | Architecture and inverted tokenization |
| 2b | Eigenspectrum, PR per rung **per origin**, and `corr(K, K_eff)` | 2b | Rolling PR and rolling OLS R² — **establishes H2's premise before any model runs** |
| 3 | Hyperparameters **and K, all models**; epochs-to-stop per rung | 3 | **Decay curve `A(b)` vs b — 15 per-origin lines + fitted `αᵢ + β₁b` with bootstrap band — key figure** |
| 4 | Main results, ± SE **across origins**, n per cell | 4 | RelMSE per block, all models |
| 5 | Per-block `D(i,b)`, surviving-window count, `b*` **with CI** at each τ | 5 | Attention heatmap: calm vs stress, **terciles of realised volatility** |
| 6 | DM matrix — statistic named per pair, T stated, Romano–Wolf adjusted, MCS column | 6 | Horizon sensitivity |
| 7 | Horizon sweep | 7 | Equity curve before and after costs, at three slippage levels |
| 8 | Economics: Sharpe, DSR, MDD, turnover, **each with an interval** | | |

**Figure 3 shows one series, not four (`D36`).** `A(i,b)` is defined only as the K=1-versus-K=8 gap
(§9.1), and §3 fixes RQ2 on that pair "never K=12". Plotting "A(b) for K=1,4,8,12" asks for a
quantity that is identically zero at K=1 and undefined at K=4 and K=12 — so whoever generated it
would silently invent `A_j = [MSE_K1 − MSE_Kj]/MSE_K1`, a *different* estimand from the one β₁ is
regressed on, and reintroduce the redundant K=12 rung into a decay comparison §3 forbids. The
per-origin-lines form above is also the better figure: it displays the actual identification, which
is within-origin slopes.

**Figure 5's regimes are data-determined (`D48`).** Calm = bottom tercile, stress = top tercile of
realised volatility across all test blocks. Picking the windows after seeing the attention maps
would make the paper's interpretability claim a free parameter.

### 13.5 Economic evaluation

Position from the **sign of the cumulative H-step forecast** — computed on **raw, drift-free**
log-returns, not scaler-space (`D31`) — held H hours, **non-overlapping** — at H=24 that bounds
turnover to one trade per day and keeps costs interpretable (`D21`).

**Three specifications the earlier text left open, each of which moves every number in Table 8
(`D46`).**

1. **Phase.** Positions open at **00:00 UTC**. There are 24 admissible alignments of a
   non-overlapping daily partition, each with a different Sharpe, MDD and turnover; choosing the
   phase after seeing the equity curve is a free parameter on the paper's economic claim.
2. **Gap-spanning returns are forbidden.** A position held across a downtime block has no defined
   realised return, and the obvious `log(C_{t+24}/C_t)` is exactly the cross-gap return §4.3 forbids
   everywhere else — at the 2018-02-08 block a nominal 24-hour trade would book a 57-hour move.
   Define the holding return **per segment**, by the same rule as `r`, and **skip** any holding
   period containing a break. Report the count of skipped periods per block, because positions exist
   only where a valid window exists and the strategy is therefore flat precisely across outages
   (`D45`) — which, since outages cluster on stress, are disproportionately the large-drawdown
   periods. Reported MDD is optimistic by an amount a reader cannot otherwise bound.
3. **Costs.** A **0.04% taker fee per side**, plus slippage at a **pre-registered sensitivity band of
   0.02% / 0.05% / 0.10% per side**, with Table 8 reported at all three. Fixing the fee exactly while
   leaving slippage blank fixes the lever that costs nothing and leaves open the one that decides
   whether the strategy makes money — and the project's own reference library anchors BTC effective
   spreads near 0.30%.

Report Sharpe, Sortino, max drawdown, turnover, and net P&L — **each with an interval**, not as bare
points: a Ledoit–Wolf or Jobson–Korkie/Memmel test for the Sharpe difference against the naive
strategy, and bootstrap intervals for MDD, which from ~180 observations is otherwise uninterpretable.

**The Deflated Sharpe Ratio, made computable (`D46`).** The earlier prescription — "DSR with
N = the number of configurations actually tried (≈837 under the `D26`/`D49`/`D50` grid; ≈621 when the
sentence was written)" — cannot be executed and would return ≈ 0 by
construction if it could. `SR₀` requires **`V[SR]`, the variance of the Sharpe ratios across the N
trials**, plus the skewness and kurtosis of the per-period returns; none was named, so N alone is
insufficient. And N is the wrong quantity: DSR counts candidates whose Sharpe was computed on the
**same return series** and from which the best was selected, whereas the 837 runs span largely
disjoint test periods, seeds, horizons and baselines that never competed for one backtest. At N = 837
and T = 180 the threshold is `SR₀ + 1.645/√(T−1) ≈ SR₀ + 0.123`, essentially unmeetable — a second
guaranteed null alongside `D23`, reading to a referee as either a failed strategy or a misapplied
statistic with no way to tell which.

Therefore: **DSR is computed per origin** on that origin's T non-overlapping 24-hour strategy
returns, from the **per-period** Sharpe (never the annualised one — feeding an annualised SR inflates
it by √(periods per year)), the sample skewness and kurtosis of those returns, and `SR₀` derived from
**N = the number of distinct strategy configurations evaluated on that origin's test span**, with
`V[SR]` the observed variance of their Sharpe ratios. The 837-run total is reported **separately** as
the development trial count and discussed in Limitations — concealing it is selection bias, but it
is not N.

---

## 14. Divergence summary

Twenty-two corrections to the source specifications. Evidence, and the paper section that must
disclose each, live in **`docs/DIVERGENCE_REGISTER.md`**. Severity: **F** fatal · **C** contradiction
· **U** underspecified · **I** improvement.

| ID | Sev | Defect in source | Resolution here |
|---|---|---|---|
| D01 | F | K=8 rung sums to 9; `log_mean_trade_size` double-assigned | Unique consistent cut, §5.2 |
| D02 | F | K_eff gate computed on the full sample, including test periods | Gate on pre-first-origin span; trigger PR < 5.0 pre-registered |
| D03 | F | `use_norm` test asserts identical loss after ×100 scaling | `MSE(c·x)/c² == MSE(x)` |
| D04 | F | Instance norm strips volatility *level*, confounding RQ1 | Measure PR on window-normalised features too; disclose |
| D05 | F | `D(b)` conflates decay with period difficulty | Define `D(b)` on RelMSE |
| D06 | F | Decay regression omits origin fixed effects | `A(i,b) = αᵢ + β₁b + ε`, Webb weights, cross-origin caveat |
| D07 | C | "2017-08 history makes 13 origins possible" | False; the count follows from the 2018-01 start. Claim deleted — count itself superseded by D26 |
| D08 | C | Horizon sweep 4 K in one place, 3 in another | 4, matching the 192-run arithmetic |
| D09 | C | Reference library says "12 origins" | 13 |
| D10 | F | Parquet on disk written with `ffill`, 122 synthetic bars | Must not be consumed as-is; regenerate or drop flagged rows |
| D11 | C | `binance_spot_klines.py` referenced, absent from tree | Restore or re-specify Stage 1 |
| D12 | U | `signed_flow` is a product of two other K=8 members | Kept, dependence disclosed, measured PR settles it |
| D13 | U | F2: per-bar or trailing-averaged, unspecified | **Per-bar, no rolling** — linear-span argument, §5.3 |
| D14 | U | `taker_buy_ratio` denominator; `H == L` division by zero | Base-denominated; zero-volume and `H == L` bars are segment breaks |
| D15 | U | Embargo "not applied (justified)" with no justification | Justification written, §8.3 |
| D16 | C | Unverified and mis-dated citations | No citation without a verified DOI and the source read |
| D17 | I | No control for "is a transformer needed at all?" | Multivariate ridge baseline added |
| D18 | I | 3 seeds too few for a `mean ± std` headline | 5 seeds for K∈{1,8} |
| D19 | I | "hours, not days" true only in one regime | GPU-resident regime documented with its counterfactual |
| D20 | I | RelMSE near 1.00 is hard to read | Report `R²_oos = 1 − RelMSE` alongside |
| D21 | U | Trading rule and DA step undefined at H=24 | Cumulative-forecast sign, non-overlapping, H-hour hold; DA at h=1, h=24, cumulative |
| D22 | I | No rung tests genuinely nonlinear features | Optional pre-registered K=16 fifth rung, §5.3 |

### Second pass — defects in *this* document set

D01–D22 correct the source specifications. **D23–D50 correct this document**, found by a five-lens
adversarial audit on 2026-08-05 and adjudicated by direct re-derivation against the text or the
artifact on disk. **Two of the five lenses — consistency and Kaggle/execution — never ran** (session
limit), so this pass is incomplete by construction; that is logged as an open item in the register.

| ID | Sev | Defect in **this** document | Resolution here |
|---|---|---|---|
| D23 | F | τ on `D(b)` is arithmetically unreachable — RQ3 is a guaranteed null | `D(i,b)` rescaled to proportional skill loss, §3, §9.1 |
| D24 | F | No purge at the train/validation boundary; early stopping selects on contaminated data | Purge at both boundaries, §8.1, §8.2, §11 |
| D25 | F | ~17,400 is the 24-month count; training is 21 months | 13,558–15,217 windows, ≤70 MB, §5.3, §8.2, §10.3 |
| D26 | F | Block index `b` is collinear with calendar month at 6-month spacing | **5-month spacing, 15 origins** (12/12 calendar phases) + falsification arm, §8.1 |
| D27 | F | Stage 5 gate opens test blocks, contradicting §11 | Gate runs on the validation sub-block, §8.5 |
| D28 | F | Consecutive origins share 75–87.5% of training data; clusters not independent | 79.2% stated numerically; effective independence ≈ 4; G=3 disjoint check, §9.2 |
| D29 | F | Every headline DM comparison is nested; standard DM is invalid there | Clark–West for nested, DM+HLN for non-nested, §9.2 |
| D30 | F | Seed std is the wrong error bar on origin-aggregated results | Dispersion bound to aggregation level, §9.2 |
| D31 | F | `ŷ_z = 0` in scaler space is a drift forecast, not a random walk | `ŷ_z = −μ_g/σ_g`, §7 |
| D32 | F | RQ1's regression is unidentified at three rung deltas | Per-origin K_eff panel + non-nested comparison, §9.1 |
| D33 | C | §4.1 paths point at an empty `data/raw/`; artifact has an out-of-window bar | Paths corrected, boundary bar dropped; **D10, D11 closed**, §4.1, §4.4 |
| D34 | C | Newey–West Bartlett contradicts the `dm.test` validation target | Rectangular truncated estimator, §9.2 |
| D35 | C | SPA/Reality Check cannot correct a pairwise matrix | Romano–Wolf stepdown + Model Confidence Set, §9.2 |
| D36 | C | Figure 3 plots `A` for four K; `A` is defined only for K1-vs-K8 | One series, 15 per-origin lines + fitted overlay, §13.4 |
| D37 | C | The linear-span argument is not a theorem under `use_norm`, and excludes half its own list | Demoted to parsimony; taxonomy carries the exclusions, §5.3 |
| D38 | C | §11 claims validation-based hyperparameter selection that never happens | Provenance stated, no per-rung tuning, §6.2, §11 |
| D39 | U | Target-channel vs all-channel loss unspecified — confounds K | Target channel only, at every rung, §6.2 |
| D40 | U | Baselines carry no K, so channel-independence is untestable | Every baseline given an explicit K, §7 |
| D41 | U | `b*` has no estimator, origin index, or censoring method | Interval-censored survival + log-rank for H3, §9.1 |
| D42 | U | Wild bootstrap recipe incomplete at every choice that moves the p-value | WCR, studentized, B, α, sidedness, t(G−1), §9.2 |
| D43 | U | Leakage surface declared closed on a two-path enumeration | Boundary × channel grid; one §11 item per cell, §5.3, §11 |
| D44 | U | K_eff's span undeclared for the RQ1 regressor; PR blind to cross-lag | Per-origin training-only span + lookback-aware measure, §5.4 |
| D45 | U | Window loss reported globally; per-block it reaches 50% | Per-origin/per-block accounting and exact assertion, §4.3, §11 |
| D46 | U | Economic evaluation: unfixed phase, gap-spanning returns, unimplementable DSR | Phase fixed, per-segment returns, DSR recipe, §13.5 |
| D47 | U | LR halved each epoch makes the 30-epoch budget decorative | Halve every 4 epochs; log epochs-to-stop per rung, §6.2 |
| D48 | U | Four outcome-determining choices left open after results | Sweep origins, regimes, slippage band, gate action fixed |
| D49 | I | Flat 8→12 rung has the fewest seeds and no equivalence test | 5 seeds at every rung + pre-registered TOST margin, §3, §6.2 |
| D50 | I | Uniform-attention control budgeted for Figure 5 but never an arm | Promoted to a main-grid arm; `A_attn` defined, §6.2, §9.1 |

### Third pass — the first defects found by *running* the code

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

### Fourth pass — defects found by building the model, 2026-08-06

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

### Fifth pass — defects found by building the experiment plane, 2026-08-06

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

### Sixth pass — the Kaggle deployment surface, 2026-08-07

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

### Seventh pass — defects found by *running the grid to completion*, 2026-08-08

`D51`–`D53` came from building the pipeline and `D54` from asking what Kaggle needs. **`D55`–`D58`
came from the first full 534-run session** — the only lens that reads the code *after* the answers
exist rather than before, and it found two defects that are invisible until the results have a
particular shape.

| ID | Sev | Defect | Resolution |
|---|---|---|---|
| **D55** | F | `DecayResult.b_star()` inferred its schema from its rows, so when `decay`'s `R2_oos > 0` guard excluded **every** origin it returned a frame with no columns and the RQ3 cell's `bs["b_star"]` raised `ColumnNotFoundError` — marking the twelve-hour version failed at the moment its grid output was the only thing worth keeping. The guard firing is the **expected** outcome under non-positive skill, not an edge case: §10.3's first run already returned `R2_oos = -0.0183` and the grid returned it at all fifteen origins. `D54e` gates on grid *completeness*, a different failure | `B_STAR_SCHEMA` declares the four columns. The RQ3 cell branches: an empty table means the estimand is **undefined** — there is no edge to lose a proportion of — which is *not* the right-censored "no decay detected within 180 days" §3 pre-registers, and reporting both in one wording would claim skill the grid never found. Log-rank likewise refuses to print `chi2=nan` where the statistic is 0/0. **Closed 2026-08-08** |
| **D56** | F | §7 calls DLinear and PatchTST "not optional" and §10.2 budgets 255 baseline runs, but **no baseline model exists in `src/`** and the manifest (534 = `main 300 + uniform 75 + fresh 15 + horizon 144`) never contained one. §10.2's 789 was never executable. `metrics.dm_nonnested()` sits waiting for input that has never existed, so Table 6 has no inputs and the paper's central architectural comparison has no data | `src/itransformer_btc/baselines.py` + manifest keys. Minimum defensible set first: **ridge (K=1,4,8,12), DLinear (K=8), PatchTST (K=8)** — enough to answer whether iTransformer failed or the whole LTSF class did, which is the question that decides what the negative result means. ARIMA, LSTM and the naive variants follow or are struck from §7 **with a written reason**, never left silently unbuilt. **Open** |
| **D57** | U | §10.3 estimated 60–100 s per run on a T4 and 10–20 h for the grid. Measured: **~30 s** and **2.31 h** for 534 runs on two T4s. The regime was right; the arithmetic on it was 2–3× pessimistic per run and 4–8× overall, so the weekly quota was never the binding constraint every §10.2 trade-off was made against | §10.3 carries the measured numbers, §10.5's resume argument reads ~30 s. The slack is what makes a second granularity affordable and what made `D58` possible. **Closed 2026-08-08** |
| **D58** | C | §15 and §10.3 described twelve `%%writefile` cells materialising a package, imported by **two GPU subprocesses**. That form existed for one reason — a subprocess reaches code only from disk — and `D57` dissolved it. Keeping the description would have left the governing document contradicting the artifact, which is worse than the defects it catches | Notebook flattened to **definition cells** in one kernel namespace; grid runs in-kernel and sequential. `D54a`'s *conclusion* (no repository Dataset) stands; its *mechanism* is superseded. `launch_workers` is retained and tested for the checkout path and for a 1-minute grid, where sequential will not fit. `code_sha256` is pinned by the generator, with the honesty cost stated in §12. **Closed 2026-08-08** |

New contradictions found later take IDs **D59+**. Absorbing one silently is the exact failure this
register exists to prevent.

---

## 15. Repository layout

```
invertedTransformer/
├── CLAUDE.md                       # this file — project law
├── README.md
├── USAGE.md                        # operational companion: commands, stages, schemas, expected numbers
├── docs/DIVERGENCE_REGISTER.md     # long-form evidence: D01–D50 and D54. D51–D53 are in §14 only
├── docs/ORIGIN_WINDOW_BUDGET.md    # per-origin/per-block window accounting — D45's assertion target
├── src/                            # importable package; module inventory in USAGE.md §2
├── tools/build_notebook.py         # generates notebooks/iTransformer.ipynb FROM src/ (D54)
├── notebooks/iTransformer.ipynb    # the launcher — self-contained, GENERATED, never hand-edited
├── paper/                          # manuscript; CLAUDE.md holds writing posture
├── spot_klines_btc.py              # Stage 1 ingest (was mis-named `binance_spot_klines.py`, D11/D33)
├── data/raw/                       # IMMUTABLE. the four Stage 1 artifacts live HERE (D33, resolved)
├── data/processed/                 # features_1h.parquet, splits.json — the writable half
└── artifacts/{preds,meta,logs,tables,figures}/  + paper_numbers.json
```

**Logic lives in the package. A notebook is a launcher.** The superseded
generate-notebooks-from-a-source-notebook workflow is dead, and logic in a notebook is a defect —
it is what made the previous pipeline unverifiable and un-unit-testable. If a cell contains a feature
definition, a window builder, a loss or a metric, it belongs in `src/` where it can be unit-tested on
CPU. **Never leave a notebook whose outputs are stale relative to `src/`**: outputs are evidence, and
stale evidence is worse than none.

**The launcher is self-contained, and that is not a weakening of the rule above (`D54`).**
`notebooks/iTransformer.ipynb` carries the whole package, so a Kaggle session needs the notebook and
`BTCUSDT_1h.parquet` and **nothing else** — no repository Dataset to upload, keep in step by hand,
and silently run stale. The rule is unchanged because no definition moved: the cells *transcribe*
`src/` and the generator is what writes them.

**It carries the package as definition cells, not as files (`D58`, superseding `D54a`'s form).**
Twelve cells define each module directly in the kernel namespace — plain `def`, `class` and constant
bodies that the cells below call by name. There is no `itransformer_btc` package on the running
machine, nothing on `sys.path`, and nothing to import. The earlier form wrote twelve files with
`%%writefile` and imported them back, and existed for exactly one reason: the grid ran as two
subprocesses and a subprocess can reach code only from disk. `D57` removed that reason — at ~30 s per
run the sequential grid is ~4.5 h inside an 11 h budget — so the files bought nothing and the
materialise-then-import step went with them.

Four consequences, each load-bearing:

- **The flattening is subtractive, over exactly two declared categories.** Intra-package imports are
  removed, by `ast` node span rather than line matching so parenthesised and function-local ones come
  out right; and `runner.py`'s `if __name__ == "__main__":` guard is removed, because in a cell
  `__name__` *is* `"__main__"` and the guard would launch the entire grid the moment its definition
  cell ran. Everything else is verbatim, and `tests/test_notebook_sync.py` asserts each cell equals
  the module under exactly that transformation — not "equivalent", not "equal after formatting".
- **Two module-level names collide** once the namespaces merge — `DEFAULT_PARQUET` (`segments`,
  `train`) and `HOUR_MS` (`segments`, `metrics`). Both are the same value in both definitions, so
  last-cell-wins is harmless. Anything else colliding would not be, which is why the generator
  compiles what it emits and the sync tests re-derive the set rather than trusting a list.
- **Generated, never hand-edited.** A second copy of ~4,000 lines is the drift this repository has a
  whole register to prevent: `python tools/build_notebook.py` writes it, the sync tests police it,
  and `--check` fails the suite the moment `src/` moves without it. **Edit `src/`, re-run the
  generator, commit both.** A hand-edit to a definition cell is a defect and the next generator run
  reverts it.
- **The digest replaces the commit, and is now pinned rather than computed.** `code_sha256()` hashes
  the `*.py` beside itself, which needs a `__file__` no definition cell has. The generator therefore
  computes the digest from `src/itransformer_btc/` and pins it into the notebook as
  `CODE_SHA256_OVERRIDE` (§12, `D54b`). It is the **same number** a local checkout of the same source
  reports, which is the whole point: a run from the notebook and a run from the repository must not
  look like different code vintages.

**The evaluation cells are code that exists only in the notebook, and they are tested as such.**
`tests/test_notebook_cells.py` executes the bytes the notebook actually contains against synthetic
panels. Testing the generator's constants instead would pass while a stale notebook still crashed —
which is precisely how `D55` reached Kaggle and cost a twelve-hour session.

**Two rules for `src/` that are not derivable from the sections above.** Shuffle by permuting an
index tensor *on device*, never by moving data — the point of §10.3's GPU-resident regime is that
data does not move after the initial load. And keep the code device-agnostic: never hard-code
`.cuda()`, and gate precision on `torch.cuda.get_device_capability(0)[0] >= 8`, never on
`torch.cuda.is_bf16_supported()`, which returns True on a T4 via emulation and selects a path slower
than fp32.

**One `CLAUDE.md` per directory is not a goal, and three of the four were deleted for cause
(2026-08-06).** `src/CLAUDE.md` and `notebooks/CLAUDE.md` restated this file at 55–65% overlap, and
their non-overlapping content was precisely the content that **must not fail open**: a subdirectory
`CLAUDE.md` loads only when a file in that subtree is touched, so a prohibition living there is
absent exactly when an agent reasons about the area without opening a file. Rules whose violation is
catastrophic — the polars boundary, PyTorch-only, no `DataLoader`, Save & Run All — therefore live
**here**, where they are loaded every turn. The test for adding a new one: *is this rule local to
that directory **and** harmless if it fails to load?* `paper/CLAUDE.md` is the only survivor, because
manuscript posture is the one area where a missed rule costs a weaker paragraph rather than a
corrupted result. The module inventory that `src/CLAUDE.md` carried lives in `USAGE.md` §2.

---

## 16. Working conventions

**Style.** Python ≥ 3.11 syntax, type hints on every public function, Google-style docstrings.
Config in YAML loaded into dataclasses — **no magic numbers buried in code**. Comments explain *why*.

**The polars boundary.** polars is the data plane: segmentation, features, windowing, splits, all
via lazy scans. **pandas is permitted at exactly two places, both named** — (1) converting to numpy
or pandas for `statsmodels`, `arch`, or `wildboottest`, which accept nothing else, via a named
function, not scattered `.to_pandas()` calls; and (2) **Stage 1 ingest**, `spot_klines_btc.py`, for
the reason given in §2 — it computes no rolling window, so the correctness argument below does not
apply to it. Nowhere else. Training touches no DataFrame at all: pre-built
GPU-resident tensors, index-slice batching, no `DataLoader`.

This is a correctness argument, not only a speed one: polars' rolling API is backward-closed by
construction, so the `center=True` leak is **unrepresentable** — in pandas it is one keyword away.
The source specification's §6.2 purge snippet is pandas and must be **re-expressed**, not copied.

**Reproducibility.** Seed `random`, `numpy`, `torch`, `torch.cuda`; set `PYTHONHASHSEED`;
`cudnn.deterministic = True` for final runs. Record git sha, **`code_sha256`** and input-artifact
sha256 in every `meta/*.json` — the middle one because the first is `"unknown"` off-repo, which is
every Kaggle session (§12, `D54`).

**The notebook is generated, and regenerating it is part of editing `src/`.** After any change under
`src/itransformer_btc/`, run `python tools/build_notebook.py` and commit the notebook with the same
change. `tests/test_notebook_sync.py` asserts each definition cell equals its module under the two
declared removals (§15, `D58`) — not "equivalent", not "equal after formatting" — so skipping this
fails the suite rather than shipping a launcher that runs last week's code. `tests/test_notebook_cells.py`
covers the evaluation cells, which exist nowhere else. Adding a module to the package means adding it
to `MODULE_ORDER` in the generator; the generator refuses to run otherwise, because a silently
omitted module would leave a name undefined somewhere deep in a twelve-hour session.

**Before any long run.** Overfit a single batch: if the model cannot drive loss to ~0 on 8 samples
in 200 steps, the plumbing is broken. **Run it with `dropout=0.0`** (`D52`) — with the configured
0.1 still active the loss floors around 7e-2 and a reader following the instruction literally
concludes the plumbing is broken when it is not. Measured on the real pipeline: **8.26e-10** with
dropout off, 6.8e-2 with it on. Compute and log the **Naive-RW baseline first**, before any model
trains.

**Environment — resolved 2026-08-06.** `pyproject.toml` now declares `requires-python >= 3.11`
(was `>= 3.14`, with torch undeclared and its wheel availability there unverified). Core dependencies
are the data plane only — **polars, pyarrow, numpy**. Everything else sits behind a named extra, so
each dependency's reason is visible rather than ambient:

| Extra | Contents | Why it is separate |
|---|---|---|
| `ingest` | requests, pandas | Stage 1 only. pandas in the core list would make the §2 exemption ambient |
| `stats` | pandas, scipy, statsmodels, arch, wildboottest | The one named boundary where data leaves polars |
| `train` | torch | Unpinned and **not installed on Kaggle** — the image ships its own |
| `dev` | pytest | |

`requirements.txt` — a UTF-16 dump of the superseded project — is deleted. Kaggle ships its own image
regardless: the notebook runs against whatever torch and polars are already there and pip-installs
only what is genuinely missing.

---

## 17. Tombstone

Before 2026-08-05 this repository pursued a different project: a production-grade 1-minute BTC/USDT
forecasting model fusing four sources at different sampling frequencies (BTC 1 min, XAU/USD 1 min,
FED Broad Dollar Index daily, 31 US macro indicators monthly), delivered as a TorchScript/ONNX export
bundle, built through a two-notebook Kaggle pipeline generated from a reference notebook and joined
by a frozen-feature-artifact contract. **That project is superseded in full.** Its rules — including
an anti-leakage table that permitted forward-fill, which §2 of this document now forbids as fatal —
are void. Rationale and implementation live in git history at `ee55c9d` and earlier; the notebooks
were removed at `cadbdf7`+. Nothing from it is authoritative, and no rule of it may be cited.
