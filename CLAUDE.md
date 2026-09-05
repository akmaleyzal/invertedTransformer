# CLAUDE.md

Governing document for this repository. Read it before doing anything else.

**Authoritative as of 2026-08-24.** It supersedes both source specifications
(`research_specification_itransformer_btc.md`, `reference_library_itransformer_btc.md`) — inputs, not
authority — and the pre-2026-08-05 project entirely (§17). `docs/DIVERGENCE_REGISTER.md` carries the
long-form evidence for every divergence; §14 is the index.

**The grid has run and the answers are in.** The 684-run manifest completed 2026-08-11, the 894-run
manifest on 2026-08-21, and **every pre-registered gate that could fail, failed**: Stage 5 did not
reject at α = 0.05, RQ2's β₁ came back with the wrong sign and inside its own MDE, and RQ3's estimand
is undefined at all fifteen origins because no arm has positive out-of-sample skill. Read §1's title
decision first: it is no longer the title this document opened with.

---

## 1. Project definition

**The deliverable is a manuscript, not a model.** No production inference path, no export bundle, no
serving contract. The model is an experimental instrument.

**Working title (`D60a`, 2026-08-20):** *A Pre-Registered Walk-Forward Evaluation of iTransformer and
Linear Baselines for Hourly Bitcoin Return Forecasting: No Out-of-Sample Skill at Any Variate Count.*

**Superseded title:** *Nominal Variates or Effective Dimensionality? A Walk-Forward Evaluation of
iTransformer for Hourly Bitcoin Forecasting.* Retained, not deleted (§12 forbids losing provenance).
§8.5 pre-registered the trigger — *"If K=8 does not beat K=1, reposition the title to the descriptive
variant now, not in week nine"* — and Stage 5 returned Clark–West `S* = +0.8759, p = 0.1906`. The old
title poses a comparative question whose premise is that added variates buy accuracy; at every rung
`R²_oos` is negative, so the question has no numerator. The K-versus-K_eff horse race moves to §4.2 of
the manuscript with `corr(K, K_eff) = 0.828` beside it.

**Target venue:** Indonesian informatics journal (Sinta), IMRaD, 10–14 pages, 35–45 references, IEEE
style. Scope is **spot-only, single-asset, feature-based**: BTCUSDT 1-hour klines from Binance and
nothing else. No futures, no second asset, no macro/on-chain/sentiment data.

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
| Add futures, second assets, or exogenous data | Spot BTCUSDT 1 h. Scope fixed by decision, not convenience |
| Report MSE on price levels, or MAPE on log-returns | MSE/MAE on standardised log-returns; RelMSE and `R²_oos` against Naive-RW |
| Cite a paper you have not read, or a DOI you have not verified | §13.3. No exceptions |
| Trust a result that looks too good | Assume leakage until proven otherwise |

**Stage 1 ingest is exempt from the polars rule.** `spot_klines_btc.py` is pandas and stays that way.
The ban is a *correctness* argument — polars' rolling API is backward-closed, so the `center=True`
leak class is unrepresentable there and one keyword away in pandas. Stage 1 computes **no rolling
window at all** (it paginates, coerces, de-duplicates, clips, counts gaps), so the argument has
nothing to bite on. The ban applies in **full** from segmentation onward, i.e. everywhere in `src/`.

Timezone is **UTC everywhere**. Every timestamp is epoch-based and compared as an integer.

---

## 3. Research questions — pre-registered

Fixed before any model ran. Changing any of them after seeing results is a new experiment and must be
declared as one.

| Code | Question | Hypothesis | Dependent variable |
|---|---|---|---|
| **RQ1** | Is the marginal benefit of added variates governed by nominal count **K** or by effective dimensionality **K_eff**? | H1: benefit tracks K_eff. Real gains at K=1→4→8, flat at 8→12 | `ΔMSE` per rung, regressed separately on K and on K_eff |
| **RQ2** | Does the multivariate-over-univariate gap narrow as time-since-training grows? | H2: it narrows. The microstructure-to-return mapping is regime-specific | `A(i,b) = [MSE_K1 − MSE_K8]/MSE_K1`; claim is **β₁ < 0** |
| **RQ3** | What retraining cadence is optimal, and does it depend on K? | H3: larger K decays faster | `b* = min{b : D(b) > τ}` |

**Measured answers (`D60b`, 2026-08-11).** The questions stand as pre-registered; these are what the
grid returned. Recorded here so a reader who stops at §3 does not leave with hypotheses and no
outcomes.

| Code | Pre-registered claim | Measured | Verdict |
|---|---|---|---|
| **RQ1** | benefit tracks K_eff; gains 1→4→8, flat 8→12 | ΔMSE 4→8 = **+0.000636**, 8→12 = **−0.000437**; TOST vs `Δ_eq = ±0.000159` gives p = (0.9734, 0.0002) | **Not shown equivalent.** The 8→12 rung is not flat — it is *worse*. J-test: K augmented by K_eff `t = +3.293, p = 0.0011`; K_eff augmented by K `t = −0.348, p = 0.7281`, so the K explanation is rejected and the K_eff explanation is not. H1 survives its own horse race **on a ladder where every rung has negative skill** |
| **RQ2** | β₁ < 0, the gap narrows with model age | β₁ = **+0.000256**, WCR one-sided p = **0.7381**, G = 15, N = 90 | **Not supported, and the sign is wrong.** MDE at 80% power is **−0.000920** and the estimate lies inside it, so §9.2 requirement 6 fires: RQ2 is **descriptive** |
| **RQ3** | `b*` at τ = 5%, larger K decays faster | `b*` **UNDEFINED** at all four τ; 15 of 15 origins excluded on mean `R²_oos ≤ 0`; log-rank unavailable in both arms | **No answer, and the reason is not censoring.** `D(i,b)` is a proportion of skill lost and there is no skill to lose a proportion of. H3 is **untestable**, not rejected |

**RQ3's wording is fixed and not interchangeable (`D60b`).** Report *"the decay estimand is undefined
under non-positive out-of-sample skill"*. **Never** *"no decay detected within 180 days"* — that is
the right-censored phrasing pre-registered for a different situation and it asserts an edge the data
does not contain.

**RQ2 compares K=1 against K=8, never K=12.** K=12 carries deliberate redundancy (§5.2); using it
would confound decay with that redundancy.

**Pre-registered thresholds, all fixed in advance — choosing any of them after seeing the curve is
p-hacking:**

- **τ**: headline **5%**, sensitivity at τ ∈ {2.5%, 5%, 10%, 50%}.
- **τ is a fraction of skill lost, not of RelMSE (`D23`).** On the RelMSE scale every τ is
  unreachable: with `RelMSE(1) = 0.996`, *total* destruction of the edge gives `D(b) = 0.402%`, so
  RQ3 would return "no decay" by units mismatch rather than by the market. `D(i,b)` is therefore
  defined on the skill scale (§9.1), where τ = 5% means "5% of the edge is gone".
- **Equivalence margin (`D49`).** RQ1's "flat 8→12" is an assertion of *no effect*, and a
  non-significant ΔMSE is a failure to reject, not equivalence. The rung counts as flat if two
  one-sided tests reject at α = 0.05 against `Δ_eq = 0.25 × ΔMSE₄→₈`.
- **Answer space for RQ3.** Six 30-day blocks means `b*` resolves only to 30-day granularity out to
  180 days. If no block crosses τ, the honest answer is *"no decay detected within 180 days"* — a
  right-censored result, in those words. (Not the situation that occurred; see above.)

**Mechanism behind H2** (economic, not just statistical): shifting participant composition —
retail-dominated flow 2018–2020, the 2021 leverage cycle, institutional flow after spot-ETF approval
in 2024. Order-flow predictability should decay as market making tightens. Grounded in the Adaptive
Markets Hypothesis (Lo 2004; Khuntia & Pattanayak 2018) and in the capacity–robustness trade-off
(Han, Ye & Zhan 2024) independently.

**Claimed contributions:** (1) first walk-forward evaluation of iTransformer on a crypto asset with
explicit decay measurement; (2) separation of nominal variate count from effective dimensionality as
competing explanations for cross-variate gains; (3) evidence-based retraining cadence under a
pre-registered degradation threshold. Hedge (1) as "to the best of our knowledge" — §13.2.

---

## 4. Data contract

### 4.1 Measured facts

Measured from `data/raw/BTCUSDT_1h_report.json`, not assumed. Re-verify after any refresh. The four
Stage 1 artifacts live in **`data/raw/`** (`D33`); any `data/BTCUSDT_*` path in an older document is
stale by one directory level.

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

**Artifact vintage** (§12 — numbers produced under different hashes are not comparable). Regenerated
2026-08-06T06:38Z; full digests in `artifact_sha256` inside the report.

| Artifact | sha256 (first 16) |
|---|---|
| `BTCUSDT_1h.parquet` | `8270a84b07c2923b` |
| `BTCUSDT_1h_gaps.csv` | `cfab4cf4c20ec00d` |
| `BTCUSDT_1h_raw.jsonl` | `30721a663bd2ce58` |

**All eleven meaningful kline columns are retained.** Truncating to OHLCV silently destroys families
F3, F4, F5 and collapses the ladder from 12 to 6. Three columns carry information underivable from
OHLC: `quote_asset_volume` (→ VWAP), `number_of_trades` (→ intensity), `taker_buy_base_volume`
(→ signed flow). The twelfth field `ignore` is dropped. Numeric fields arrive as **strings** and must
be coerced explicitly — silent failure otherwise.

**One defect class, now closed, worth remembering.** The report once carried `bars_actual` 75,095 /
`missing_bars` 121 with a last bar one hour **past** the end-exclusive boundary — a `missing_bars`
that **contradicted its own gaps file**, in the direction that flatters the data. Gap *detection* was
never wrong; only counts derived from `len(df)` were. `clip_to_window()` now enforces the half-open
window and the regression test lives in `spot_klines_btc.py --self-test`. A defect that leaves two
artifacts disagreeing is the kind that survives review.

### 4.2 Gaps are not missing values

BTCUSDT trades continuously, so zero-trade hours are ruled out. What remains is **exchange downtime
and scheduled maintenance**, confirmed against `data/raw/BTCUSDT_1h_gaps.csv` (27 rows).

Rubin's MCAR/MAR/MNAR taxonomy applies to values that exist but went unobserved. **When the exchange
is down no price forms** — no matching, no book, nothing to approximate. Imputation is not risky here,
it is *undefined*: there is no ground truth, so no metric can justify any imputation choice. This also
retires the MNAR diagnostic. Cite Rubin (1976) precisely in order to argue the taxonomy does **not**
apply.

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

A segment is a maximal run of contiguous, usable hourly bars. Segments are broken by (1) any missing
bar (27 blocks, 122 bars); **and** (2) any **zero-volume or `H == L` bar** — it carries no trade
information, exactly like downtime, so it is excluded and the series splits there. This is what makes
`(VWAP − C)/(H − L)` and `log(volume)` total, not partial, functions (`D14`).

`r` is computed **per segment**; the first bar of each segment yields NaN and is dropped. Computing
returns on a concatenated series injects giant cross-gap returns into μ_g and σ_g.

**Windows are validated by timestamp, never by positional index.** Highest-probability silent bug in
the pipeline: after any row drop, positional sliding closes gaps invisibly.

```
window [s, s+L+H) is valid  ⟺  t[s + L + H − 1] − t[s] == (L + H − 1) hours
```

**Unusable bars, measured (`D51c`): 3, and they are the *same* 3 bars** as the zero-volume and
zero-trade ones — `2019-06-07T21:00`, `2021-02-11T03:00`, `2023-03-24T12:00`. No volume ⇒ no trades ⇒
high and low never separate. Total unusable is 3, not 9.

**Cost accounting.** A break costs `L + H − 1 = 119` window start positions plus its own bars, so cost
is dominated by the *number* of breaks. Pooled: `119 × 30 + 125 ≈ 3,695 ≈ 4.9%` of ~74,975 — well
inside the 16% tolerance, so **no narrowing of the analysis window is required**. Read `gap_blocks`
from the report, not `missing_bars`.

**The closed form is an upper bound, not an identity (`D51a`).** A segment of `n < 120` bars
contributes zero windows but is charged `n − 119`, and the negative is absorbed silently — origin
2022-02 holds an 80-bar segment and seven origins were understated by 39…137 windows. Count windows
**segment-wise**, `Σ max(0, nᵢ − 119)`; keep the closed form only as an assertion
`closed_form ≤ measured`.

**Test blocks hold 720 forecast origins, not 601 (`D51b`).** Accounting them with *training* semantics
— whole 120-bar window inside the block — returns 601/720 on a *clean* block, a 16.5% phantom loss
§9.2 would absorb as noise. §8.3 licenses the 96-bar lookback crossing backwards; only a break inside
the spanned 120 bars disqualifies a start. Measured: 74 of 90 cells clean, worst 439/720.

**The pooled figure is the wrong granularity for anything except feasibility (`D45`).** 26 of 27
downtime blocks fall in 2018–2021 and none after 2023-03. Per-origin **training** loss runs ~11% down
to **0%**, monotone in calendar time; per-(origin, block) **test** loss runs 0% to **50.4%**. Two
consequences:

- rejected-window counts are asserted **per origin, by exact equality** against
  **`docs/ORIGIN_WINDOW_BUDGET.md`** — `rejected(origin) == 119 × breaks(origin) + missing(origin)`.
  Asserted against the pooled 4.9% it fires spuriously at fourteen of fifteen origins, gets loosened
  until it passes, and can no longer distinguish positional drift from ordinary variation. That is the
  *only* defence against the silent bug named above;
- **test-window survival is conditioned on future gaps.** Whether a forecast issued at *s* is
  evaluated depends on whether the next 120 hours contain an outage — information unavailable at *s*.
  Binance outages cluster on stress, so dropped targets are systematically the high-volatility ones.
  Report surviving counts per cell and state the exclusion in Limitations (§9.2, §11).

### 4.4 The artifact on disk

`D10` (ffill / synthetic bars) and `D11` (missing Stage 1 script) are **closed** — neither defect ever
existed on the artifact; the register was stale, not the parquet. `D33`'s boundary-bar defect is
closed too (§4.1). The runnable replacements are in §11: assert `parquet_rows == bars_actual` and
assert the timestamp diff set contains the 27 gap blocks. `BTCUSDT_1h_gaps.csv` is retained as the
source of segment boundaries; `BTCUSDT_1h_raw.jsonl` makes offline re-derivation possible.

### 4.5 Preliminary market-efficiency tests

Run once, report in the Data section. Converts "efficient market" from assumption into finding.

| Test | Library | Reading |
|---|---|---|
| Variance Ratio (Lo–MacKinlay) | `arch.unitroot.VarianceRatio` | VR ≈ 1 → consistent with random walk |
| Hurst exponent | R/S implementation | H ≈ 0.5 → no long memory |
| ADF | `statsmodels.tsa.stattools.adfuller` | log-returns stationary |

Do **not** claim the market is efficient. State that evidence is mixed and time-varying (Urquhart
2016; Nadarajah & Chu 2017; Bariviera 2017; Sensoy 2019), then report your own numbers. Reported full
sample **and** per origin's 21-month training sub-block, because the required claim is about
*variation* and one row cannot exhibit it.

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

Pinned definitions:

- `taker_buy_ratio = taker_buy_base_volume / volume` — **base-denominated** (`D12`/`D14`). The
  quote-denominated variant is a robustness check, not the default.
- `signed_flow = (2·taker_buy_ratio − 1) · log_quote_volume`. **Disclose** that this is a
  deterministic product of two other K=8 members; it weakens the claim that K=8 is the rung of maximum
  effective rank, and the measured PR settles it.
- **F2 estimators are per-bar, no trailing average** (`D13`, §5.3). All three are ≥ 0 given
  `H ≥ max(O,C)` and `L ≤ min(O,C)`. **Only two are strictly positive once `H == L` bars are excluded**
  (`D52a`): Parkinson is `∝ (ln H/L)²` and Garman–Klass is bounded below by `0.114 (ln H/L)²`.
  **Rogers–Satchell is not** — it vanishes on any shadowless bar (marubozu), and **33 of 75,091 usable
  bars** are such. It therefore uses `log(RS + κ)` with **κ = 1e-9 fixed**, chosen so `log κ = −20.7`
  lands inside the measured support rather than as 33 out-of-support spikes that would distort the
  instance normalisation of every window containing one and smuggle a categorical marubozu flag into a
  continuous variate. Not applied to the other two (minima 1.16e-8, 1.48e-8). Post-κ the RS support is
  min **−20.723** (the floor itself), q0.1% **−19.805**; the pre-κ figures were −23.5 / −17.57
  (`D60h`). Disclose in §4.1b.
- `log(C/O)` is **not** a variate. Crypto bars are contiguous, so `log(C/O) ≈ r` at ρ ≈ 0.99 — silent
  duplication that inflates K without inflating K_eff, corrupting RQ1's own axis.

### 5.2 The ladder — corrected (`D01`)

The source specification's K=8 rung sums to **nine** and double-assigns `log_mean_trade_size`. Exactly
one consistent cut exists:

| K | Members added | Cumulative | Expected K_eff | Measured PR |
|---|---|---|---|---|
| **1** | `r` | 1 | 1 | 1.000 |
| **4** | `upper_shadow`, `lower_shadow`, `log_quote_volume` | 4 | ~3.5 | — |
| **8** | `log_trade_count`, `taker_buy_ratio`, `signed_flow`, `vwap_location` | 8 | ~6.5 | **4.269** |
| **12** | Parkinson, Garman–Klass, Rogers–Satchell, `log_mean_trade_size` | 12 | ~7 | **3.984** |

**The 8→12 rung is deliberately redundant and functions as a control, not an accident.** It adds four
nominal variates carrying almost no additional rank. If accuracy rises 4→8 then flattens 8→12, that is
the demonstration that nominal K is the wrong axis. **State this explicitly in the methodology**, next
to the K=1 degeneracy note (§6.2); left unstated, a reviewer reads the flat rung as a null result
rather than as the designed contrast. Measured, K=12's PR is *below* K=8's — the rung is **more**
redundant than designed, so the control is stronger evidence than expected. Fix the hypothesis to the
measurement, never the reverse (§5.4).

**No variate may be added outside F1–F5.** A near-gap indicator, a calendar dummy, or any other
convenience variate breaks the taxonomy and renders the K=8 vs K=12 contrast uninterpretable. Losing a
few percent of windows is far cheaper than contaminating the primary independent variable.

### 5.3 Feature-engineering policy — why exactly twelve

**All twelve variates are engineered.** None is a raw kline column. What is excluded is a specific
*class*: technical indicators (RSI, MACD, Bollinger, MA/EMA), multi-bar rolling statistics, calendar
and session dummies, cross-asset, on-chain, sentiment, and macro data.

1. **K is RQ1's independent variable — this reason alone carries the exclusions** (`D37`). Anything
   outside F1–F5 breaks the taxonomy that makes K_eff interpretable. RSI, MACD and Bollinger belong to
   no family in §5.1, so admitting them makes the K=8-versus-K=12 contrast uninterpretable.
2. **Parsimony, not a span theorem** (`D37`). iTransformer embeds each variate's entire L=96 lookback
   through `Linear(96 → d_model)`, so a *linear* function of that lookback — MA, EMA, momentum, n-step
   differences — is recoverable in principle; adding it as a separate variate raises nominal K while
   changing inductive bias and per-channel normalisation rather than information content, which is
   close to the phenomenon RQ1 exists to measure. Three caveats, because the earlier formulation
   overclaimed: (a) under `use_norm=True` each channel is divided by its **own** per-window σ and
   `x ↦ (a·x)/std_t(a·x_W)` is not linear in `x`, so a separately-added trailing average is not
   literally inside the span; (b) with `d_model = 128 > L = 96` the projection is generically
   **injective**, so the linear/nonlinear line is *not* the recoverable/unrecoverable line; (c) RSI
   and Bollinger are not linear in the lookback anyway. Reason 1 carries the exclusions; reason 2
   explains why adding linear transforms would inflate K without inflating information.
3. **Sample budget.** **13,558–15,217** training windows per origin against 12 × 96 = 1,152 input
   dimensions — the **21-month** sub-block, per origin (`D25`, `D45`, `docs/ORIGIN_WINDOW_BUDGET.md`).
   17,400 is the 24-month count and must not appear. On the **feature** frame the range is
   **13,545–15,217**, exactly one window fewer per segment because `r` drops each segment's first bar
   (`D52c`); assert against the feature-frame numbers and keep the raw-frame table for gap accounting.
4. **Benchmark positioning.** iTransformer's own suite splits into few-features-one-entity
   (ETTh1 = 7, Weather = 21) and many-entities (Electricity = 321, Traffic = 862) without
   distinguishing them. K ≤ 12 sits deliberately in the first regime; measuring K_eff makes the
   distinction quantitative. Strongest available framing for the RQ1 contribution.

**Corollary — no feature uses a rolling window.** Pre-smoothing an F2 estimator over 24 bars is
strictly *less* informative than the per-bar estimator: the model can compute that average itself and
cannot recover what smoothing destroyed. Every variate is a pure per-bar function of the current bar,
except `r`, which uses the current and previous close. This is a **structural safety property**: with
no rolling window anywhere, the `center=True` leak class is unrepresentable.

**But the surface is not therefore closed (`D43`).** That closure claim is scoped to *feature
construction* at the *train–test* boundary and fails at every other boundary: **train–validation**
(§8.2, `D24`), **model selection** (§8.5, `D27`), **evaluated-sample composition** (§4.3, `D45`), and
**cross-origin training overlap** (§8.1, `D28`). Enumerate the surface as a grid of **boundary**
(train/val · train/test · cross-origin) × **channel** (features · labels · scaler · model selection ·
evaluated-sample composition), with one §11 item per non-empty cell. §8.3's feature-lookback argument
covers **one cell**, not the grid. A closed-surface claim is what stops the hunt §2 mandates.

**Optional fifth rung — K=16 (`D22`). NOT RUN; clause 1 failed (`D60a`).** Pre-registered as a rung
adding trailing realized variance, signed-flow autocorrelation and VPIN-style toxicity, as a second
high-nominal-K / low-K_eff control. Its old rationale (nonlinear functionals "outside the embedding's
span") does not survive `d_model > L` and is withdrawn; the real ground is that these are **multi-bar
functionals** that reintroduce the rolling-window leakage surface and therefore **reopen §8.3's
no-embargo argument**, which would have to be re-derived first. Run condition (`D48`): origin 1's
Stage 5 gate passes at α = 0.05, **and** ≥ 5 GPU-hours of weekly quota remain. **Clause 1 failed** —
`S* = +0.8759, p = 0.1906`; clause 2 was never reached and quota was never binding. §13.2 names clause
1. Running K=16 anyway would be exactly the post-hoc rung this section forbids, and its pre-registered
prediction ("K=16 improves on K=12 by more than the 8→12 rung does") is untestable now that the 8→12
rung is itself negative. **Separate rung or nothing** — folded into the existing four it destroys the
instrument.

### 5.4 Pre-model measurement — `K_eff`

Run **before training anything**. Produces RQ1's independent variable and tests H2's premise at zero
extra data cost.

| Measurement | Span | Definition | Purpose |
|---|---|---|---|
| Participation ratio | **per origin, that origin's 21-month training sub-block** (`D44`) | `PR = (Σλᵢ)² / Σλᵢ²` on the correlation matrix of each rung | RQ1's regressor, **varying**. Bounded in [1, K] |
| Lookback-aware PR | same | PR of the `K·L × K·L` **correlation** spectrum (`D53b`), reported as a fraction of its `K·L` ceiling | the contemporaneous PR is blind to cross-lag structure |
| Stable rank | same | `K / λ₁` of the **within-window correlation** matrix (`D53a`), bounded [1, K] | cross-*variate* structure inside a window |
| PR on window-normalised features | same | Same, after per-window standardisation over L | **`D04`** — see below |
| Gate PR | **pre-first-origin span only** (`D02`) | as row 1 | Gates the ladder; informs no other number |
| Rolling PR | full sample | 90-day rolling window, 2018–2026 | **descriptive only** — may inform no design decision |
| Rolling OLS R² | full sample | `r_{t+1} ~ (K=8 features)`, 90-day window | If unstable, H2's premise is established before a single epoch runs |

**Two of those definitions were units artefacts as first written (`D53a`, `D53b`), and the corrections
are load-bearing.** "Stable rank of the `K × 96` window block" with centring alone leaves
`log_quote_volume` deviations two orders of magnitude above `r`, so one row dominates both norms —
measured 1.00 / 1.00 / 1.16 / 1.65 across the rungs, a statement about units. Standardise each channel
**within its window** first (what `use_norm=True` does anyway); measured then 1.00 / 2.36 / 2.70 /
2.17. Likewise the `K·L × K·L` **covariance** spectrum is not monotone in K — 92.1 / 21.9 / 37.3 /
15.5, the K=1→4 collapse being entirely `log_quote_volume`'s arrival. Use the **correlation** spectrum.

**`D44` — every reported K_eff declares its span, and RQ1's regressor is training-only.** A
full-sample PR would be estimated on the same data as the outcome, making RQ1 partly circular while
§11's fatal item still passed (it audits only the gate). Per-origin training-only PR closes it and
makes the regressor vary, which is what `D32` needs for identifiability. Any full-sample PR is
labelled descriptive in Table 2b and is never a regressor.

**`D44` — and the construct is not what the model sees.** PR on the contemporaneous K × K correlation
matrix is blind to cross-lag structure while the model consumes a K × 96 block. Report the
contemporaneous PR **and** at least one lookback-aware measure on the same rungs; pre-register which
is RQ1's regressor before Stage 3b; report the divergence in §4.1b whatever it is. If the construct
does not correspond to what the architecture consumes, the second claimed contribution is a
measurement-validity failure rather than a finding — the question a methods referee will spend the
review on.

**`D04` — the instance-normalisation confound.** `use_norm=True` divides each window by its own
per-variate σ over L, so F2 contributes *shape*, not *level*. The 8→12 rung can flatten for a reason
unrelated to redundancy. PR must be measured on **window-normalised features as well as raw**, both
reported, confound disclosed in Limitations whatever the outcome. Provenance: Laloux et al. (1999),
Plerou et al. (2002).

**`D02` — the gate may not read the future.** The gating K_eff is computed on the **pre-first-origin
span only (2018-01 → 2020-01)**; the rolling PR is descriptive and may inform no design decision.
Pre-registered trigger: **if measured PR at K=8 falls below 5.0, disclose** (§8.5 — `D48` replaced
"re-cut" with disclosure, because D01 establishes exactly one consistent cut exists and "re-cut" names
no reachable alternative). **Measured: 4.393 — the gate did not pass.** `corr(K, K_eff) = 0.828`
against the ≈0.97 §9.1 anticipated, which makes `D32`'s horse race **more** identifiable, not less.

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

No causal mask. Masking applies to the time axis; this attention runs over the variate axis, where all
tokens are contemporaneous. Causality is enforced upstream, in features and windowing.

### 6.2 Hyperparameters

| Parameter | Value | Note |
|---|---|---|
| `seq_len` L | 96 | 4 days |
| `pred_len` H | 24 | headline; sweep {1, 3, 24, 168} |
| `d_model` | 128 | **attention sequence length is N (≤12), not L.** `d_model=512` over-parameterises against ~14,000 samples (`D25`). State this in the paper |
| `d_ff` / `e_layers` / `n_heads` | 256 / 2 / 8 | ≈ 280k parameters |
| `dropout` / activation | 0.1 / GELU | |
| Optimiser / lr | Adam / 1e-4 | |
| Schedule | **halve every 4 epochs** (`D47`) | per-epoch halving reaches ~4e-7 by epoch 9, so the budget could never bind |
| Batch / max epochs / early stop | 32 / 30 / patience 5 on validation MSE | epochs-to-early-stop and final training loss **logged per rung** |
| Loss | **MSE on the target channel only** (`D39`) | at every rung, so the objective is identical across the ladder |
| `use_norm` | **True** | mandatory, not a tuning knob |
| Seeds | **5 at every rung** (42–46) (`D18`, `D49`) | 3 seeds is too few for a `mean ± std` headline — and the 8→12 rung is RQ1's designed contrast, so it cannot carry the fewest |

**Nothing here is tuned (`D38`).** Every value except `d_model` is **adopted unchanged from Liu et al.
(2024)**; `d_model` was reduced from 512 for the sample-size reason above. **No per-rung tuning** —
holding capacity fixed is what makes the rungs comparable. §11's item on validation-based
hyperparameter selection therefore applies to **ARIMA order and ridge α only**. State this provenance
in §3.4 and Table 3; left unstated it feeds §13.5's DSR trial count as an unknown. Because one config
serves all four rungs, one pre-registered robustness run at K=12 with larger `d_ff` exists so a flat
8→12 rung cannot be read as under-tuning — built as the `capacity` arm (§10.2, `D62b`).

**The epoch cap is not the binding constraint at the headline horizon, and the claim carries its
scope (`D62c`, rescoped by `D78`).** On the 1,620-run grid, **0 of 300** iTransformer runs at H = 24
reach 30 epochs; at H = 168 the sweep trains longer and **5 of 80** do. The binding constraint is the
**LR schedule** (`lr_halve_every = 4` puts the rate at ≈1.6e-6 by epoch 26), so raising `max_epochs`
alone is a provable no-op — `itrl` runs to 60 and never exceeds 21. The `longsched` arm widens the
schedule instead.

**But the cap does bind for two baselines, and that was never disclosed (`D78`).** DLinear reaches it
in **56 of 75** runs and PatchTST in **39 of 75**. Where an arm sits at its cap its loss is a
truncated-training figure, so "DLinear is the worst model" is confounded with "DLinear is the most
truncated model" until the at-cap count is printed beside it — which Table 3 now does, reading each
arm's cap from its own `meta['schedule']` rather than assuming 30.

**The loss is single-channel, and this is not a detail (`D39`).** Standard iTransformer
implementations compute the loss over **all N channels**. Under an all-channel loss, K=12 becomes a
12-task problem and K=1 a 1-task problem, so auxiliary supervision would vary with the study's own
independent variable — and K=1 would not be the stated control but a different learning problem. The
reference implementation defaults to the option that breaks the design, so §11 carries this as a
verifiable assertion.

**K=1 degeneracy is the control, not a bug (`D50`).** At N=1 self-attention over a single token gives
softmax weight 1, so iTransformer reduces to
`Linear(L→d_model) → [W_O W_V x + x] → LayerNorm → FFN → Linear(d_model→H)` — the value and output
projections and the residual **remain**; it is not a bare identity. **State this in the methodology**;
unexplained, an examiner reads it as an implementation error.

**Parameter count is identical across rungs at a fixed horizon, and only there (`D60h`).** The
projection is `Linear(d_model → H)`, so the count moves with H and only with H: **277,505** at H=1,
**277,763** at H=3, **280,472** at H=24, **299,048** at H=168 — and within each, identical at
K = 1, 4, 8, 12, which is the claim that matters because the ladder is the comparison. Written flat it
is falsified by this project's own Table 7.

**But K=1 vs K=8 does not isolate attention.** The two arms differ in *information* and in *whether
attention is active*, simultaneously. A decaying `A(b)` is equally consistent with "cross-variate
attention overfits regime-specific structure" — a capacity story — as with RQ2's information story,
and ridge (`D17`) separates the two only under a linear model. The clean control is an iTransformer at
**K=8 with attention forced uniform**, promoted to a **third arm of the main grid** (§10.2), giving
`A_attn(i,b)` in §9.1.

### 6.3 `use_norm` and the scaler

The outer per-channel affine scaler **cancels algebraically** under instance normalisation. With
`z = (x − μ_g)/σ_g`, the instance statistics are `m = (mean_t(x_W) − μ_g)/σ_g` and
`s = std_t(x_W)/σ_g`, so `(z − m)/s = (x − mean_t(x_W))/std_t(x_W)` — μ_g and σ_g vanish. True for
StandardScaler, RobustScaler and MinMaxScaler alike.

What the scaler *does* control: the **reporting scale** of every metric (MSE is in units of σ_g²), and
**learning for baselines without internal normalisation** (LSTM, ARIMA, ridge). Cross-model scale
consistency matters more than the choice.

**StandardScaler**, for three reasons that belong in the paper: literature comparability (the LTSF
literature reports MSE on z-scored data); inertness under `use_norm=True`; consistency across models.
**RobustScaler is rejected** — on fat tails `σ > IQR/1.349`, so dividing by the smaller quantity makes
outliers *larger* in scaled space and lets crash bars dominate MSE more, exactly wrong for a study
about regime behaviour. The fat-tail concern argues for Huber/MAE loss, not a different scaler.
**MinMaxScaler is excluded on correctness**: bounded by training min/max, it produces out-of-range
values whenever a test regime exceeds the training range — a recurring defect in the crypto LSTM
literature, worth one sentence in Related Work.

**Corrected verification test (`D03`).** The source specification's "multiply input by 100, assert
identical loss" cannot pass — the target is a channel of the same array, so the loss scales by `c²`.
The invariant is

```
MSE(c · x) / c²  ==  MSE(x)          equivalently: RelMSE unchanged
```

Run before the main grid; measured rel **2.68e-06**. If `use_norm` is ever disabled the entire
argument above collapses and the scaler again affects learning — document the flag state either way.

**No winsorization.** Clipping at ±5σ removes exactly the events driving regime-dependent decay. State
the decision explicitly or a reviewer reads it as an oversight.

---

## 7. Baselines

**Every baseline carries an explicit K (`D40`).** A channel-independent baseline at an unstated K
cannot speak to the channel-independence debate §13.1 makes a Related Work pillar. The comparison an
LTSF-literate reviewer wants is iTransformer at K=8 versus PatchTST at K=8 on identical information;
if PatchTST silently ran at K=1, the central architectural comparison collapses back into
univariate-versus-multivariate. Report the K of every model in Tables 3 and 4.

| Baseline | K | Role | Configuration | Built? |
|---|---|---|---|---|
| **Naive-RW** | — | mandatory, EMH baseline | **`ŷ_raw = 0`** in raw log-return space (`D31`) — never the last return | needs no run |
| **Naive-persist** | **1** | secondary | ŷ = last observed return | ✅ 15 runs (`D64`) |
| **Seasonal-naive** | **1** | daily pattern | ŷ = return at t−24 | ✅ 15 runs (`D64`) |
| ARIMA | 1 | classical | order by AIC on the training window | deferred |
| **LSTM** | **8** | RNN, **multivariate** | 2 layers, hidden 128, dropout 0.1 | ✅ 45 runs (`D64`) |
| **DLinear** | **8** | mandatory | trend–seasonal decomposition + linear | ✅ 45 runs |
| **PatchTST** | **8** | SOTA, channel-independent | patch 16, stride 8 | ✅ 45 runs |
| **Ridge (multivariate)** | **1, 4, 8, 12** | `D17` | L2 on the same K features, α by validation | ✅ 60 runs |

**Three of the four deferrals are closed (`D64`); ARIMA alone remains, and with a reason.** `D56`
recorded in writing that nobody had built LSTM, naive-persist or seasonal-naive, and the record stood
for the honest reason that a deferral nobody states is a deferral nobody notices. They are built now:
the two naive comparators are closed forms costing microseconds, and leaving them *deferred* put two
rows in the results table that read as unfinished work rather than as measurements. LSTM is the
weightier of the three — it is the deep model the crypto forecasting literature this paper argues
against reaches for first, and claiming *no deep model beats Naive-RW* while leaving the most-cited
one untested is a hole a reviewer finds in a single pass.

**ARIMA stays deferred, and this is its written reason.** On hourly crypto log-returns AIC selection
lands at or near order (0,0,0), which *is* the naive baseline: the result is predictable from the
ADF and variance-ratio numbers §4.5 already reports, and running it would add a row that duplicates
one the table has. If that prediction is ever to be relied on rather than asserted, run it — but do
not leave the row unexplained.

**LSTM's K=8 does not mean what DLinear's and PatchTST's mean (`D64`).** Those two wear the label
through their published all-channel objective with shared weights — *trained on* eight channels,
predicting the target from its own history alone (`D56`). An LSTM reads all eight channels of every
timestep and emits the target, so its K=8 means what ridge's and iTransformer's mean, and it is
therefore the recurrent comparator on the ladder's own information set. `loss_channels` is
`target`, so unlike DLinear and PatchTST its `best_val_mse` **is** comparable to the ladder's. State
this wherever its numbers appear.

**The two naive comparators are K=1, and are not Naive-RW.** Both read the target channel and nothing
else, so 1 is their honest label — `D40` requires the label, not that the label be large. Naive-RW
remains separate and needs no run at all: it forecasts `ŷ_raw = 0`, not the last return (`D31`), and
naive-persist exists precisely to show that the weaker comparator is weaker rather than to assert it.

**DLinear and PatchTST are not optional**: a missing DLinear is the first thing an LTSF-literate
reviewer flags.

**Naive-RW uses `ŷ_raw = 0` — in raw log-return space, and the space is not optional (`D31`).** A
random walk in price implies zero expected return; using the last return gives a weaker baseline and
flatters your results. But §9.1 fixes all metrics on **standardised** log-returns, so `ŷ_z = 0` would
mean `r̂ = μ_g` — the training-window mean hourly return — making the "EMH baseline" a constant-drift
model. Measured across all fifteen origins, `μ_g/σ_g` spans **−0.00818 … +0.01733** (`D52b`; the
pre-measurement figure 0.037 was ~2× too large). Its square is 3.0e-4, about **7.5%** of the
`R²_oos ≈ 0.004` D20 anticipated, and the 24-step tilt is ≈ **0.085σ**. The correction does not weaken
the argument: `μ_g/σ_g` **changes sign across origins** — negative at 2020-01, 2022-12, 2023-05,
2023-10, 2024-03 — so it is not a constant tilt a reader could subtract, and it tracks the bull/bear
cycle H2 invokes as its own mechanism, which is why it is confounded with the effect of interest.

Therefore: define the baseline as `ŷ_raw = 0`, map it into scaler space as **`ŷ_z = −μ_g/σ_g`** for
metric computation, and log `μ_g` and `μ_g/σ_g` per origin in `meta/*.json`. Use raw-space
(drift-free) returns for the §13.5 sign rule and the §9.1 DA definitions.

**Baselines are scored on exactly the same surviving windows (`D45`).** Naive-RW needs no 96-bar
lookback, so unless restricted to the window set its comparator evaluated, RelMSE is a ratio across
two different samples. Assert equality of evaluated timestamps before computing it. (Vacuous for
Naive-RW, which `block_metrics` computes from `naive_rw_z` on exactly its comparator's rows; binding
and **fatal** for every other pair.)

**Why ridge was added.** K=1 iTransformer controls for *architecture* — "does cross-variate attention
help?" It does not answer "is a transformer needed at all?" Ridge on the same K features separates
*does the information help* from *does attention help*, costs seconds, and closes a question a
reviewer will otherwise ask.

**What "K = 8" means for a channel-independent model (`D56`) — four design decisions, each logged as a
`meta/*.json` field:**

- **The channel-independent baselines carry their published all-channel objective, weights shared
  across channels.** Trained on the target channel alone, DLinear-K8 and PatchTST-K8 would be
  numerically identical to their K=1 selves — K=1 wearing a K=8 label, the collapse `D40` exists to
  prevent. Their K label means **trained on eight channels**, not **predicts the target from eight
  channels**; `loss_channels` / `channel_independent` are logged per run. Consequence to state
  wherever their numbers appear: their `best_val_mse` is an **all-channel** figure and is *not*
  comparable to the ladder's target-channel figure. Only metrics computed from `preds/` are comparable
  across models.
- **`D39`'s single-channel loss is unchanged for the ladder.** The all-channel objective is the
  baselines' own; the rungs stay target-channel at every K.
- **DLinear's decomposition contains a centred moving average, and §5.3 survives it.** The ban is
  scoped to *features*, where a rolling statistic over the full series lets a later bar reach an
  earlier feature value. DLinear's average is computed at inference time from the 96 bars of the
  window itself, all of which precede the first forecast hour, and its padding replicates the window's
  own endpoints. No test-period bar can influence any training-set value, so §8.3 is untouched.
  Reproducing the published decomposition matters more than avoiding the word "centred".
- **PatchTST reuses iTransformer's encoder block and capacity verbatim** — `d_model` 128, `d_ff` 256,
  2 layers, 8 heads, dropout 0.1, same `EncoderLayer` — so the two differ in *what a token is* and
  nothing else. 302,360 parameters against 280,472. §6.2's no-tuning rule extends to the baselines,
  and **ridge's α remains the only hyperparameter selected anywhere in this study**.

**Measured per-run wall time on the T4, full 684-run manifest (`D60d`):** iTransformer **36.4 s**,
uniform-attention **24.3 s**, fresh **32.5 s**, ridge **0.2 s**, DLinear **21.9 s**, PatchTST
**95.6 s**. PatchTST is **~2.6×** iTransformer, not the ~16× a CPU measurement predicted; its 45-run
arm took **1.19 h**, not ~6 h. The transferable rule: **a throughput ratio measured on one device does
not scale to another** — PatchTST's B×N = 256 folding is a penalty a 6-thread CPU pays in full and a
T4 largely absorbs.

**The baseline ordering inverts the paper's premise (`D60c`).** Mean `R²_oos` over every test block:
**ridge −0.000568**, PatchTST −0.016312, iTransformer-K8 −0.017993, DLinear −0.026248. Every model is
worse than Naive-RW; ridge is worse by roughly **thirty times less** than any deep model, and its α
selection drives it close enough to the training mean that it nearly *is* the baseline. Read plainly:
at heavy shrinkage the linear model loses almost nothing and the three deep models spend their
capacity making things worse. This is the answer to the question `D17` added ridge to ask — *is a
transformer needed at all?* — and the answer is **no**. It belongs in Results **before** RQ1, and in
Related Work beside the channel-independence debate, not buried in a table at the back.

---

## 8. Walk-forward protocol

### 8.1 Scheme

Rolling-origin walk-forward with purging.

| Component | Value |
|---|---|
| Training window | 24 months, **fixed** (rolling, not expanding) |
| Training / validation sub-block | 21 months / final 3 months |
| Purge | H steps at **both** boundaries: train→validation **and** train→test (`D24`) |
| Embargo | not applied — justified in §8.3 |
| Test blocks | 6 × 30 days after the origin, **no retraining** |
| Origin spacing / count | **5 months / 15 origins** — not 6 months / 13 (`D26`) |

**Origins (`D26`, superseding `D07`/`D09`):** 2020-01, 2020-06, 2020-11, 2021-04, 2021-09, 2022-02,
2022-07, 2022-12, 2023-05, 2023-10, 2024-03, 2024-08, 2025-01, 2025-06, 2025-11. Earliest satisfies
24 months of training from the 2018-01 start; latest satisfies `o + 180d ≤ 2026-08-01`. The source
claim that "spot history starting 2017-08 is what makes thirteen possible" is false and deleted.

**Why the spacing is 5 months (`D26`) — load-bearing for RQ2.** With origins spaced *s* months and
30-day blocks, the calendar month block `b` lands on at origin *i* is `m₀ + s·i + (b−1) (mod 12)`, so
the months visited form a coset of size **12/gcd(s,12)**:

| *s* | months visited per `b` | origins | consecutive training overlap | worst test block loss |
|---|---|---|---|---|
| 6 (original) | **2** — Jan or Jul | 13 | 75.0% | 50.4% |
| 3 (interleaved) | **4** | 25 | 87.5% | 50.4% |
| **5** | **all** | **15** | **79.2%** | **33.9%** |

At *s* = 6, `b` is a deterministic function of calendar month up to a two-phase alternation, month
dummies cannot be added post hoc, and a significant β₁ < 0 is observationally equivalent to "February
and August are harder than January and July" — a bias no post-grid analysis can remove. Interleaving
at 3 months is not the fix (`gcd(3,12) = 3`, four phases). Only *s* coprime to 12 fully decouples, and
among those 5 maximises the origin count. **Measured (`D51d`), blocks are 30 *days*, not calendar
months**, so the visited-month counts are **12/7/11/11/11/11** against 6-month spacing's 2/2/2/3/3/2 —
the conclusion survives; the "12 for all b" was an idealisation of its own algebra.

**The nominal cluster count is not what denser spacing buys.** Effective independence is bounded by
`total span / training window ≈ 96/24 ≈ 4` independent training sets *regardless of spacing*; packing
origins closer inflates G without adding information while worsening the overlap §9.2 must disclose.

**Falsification arm, pre-registered.** For every origin, train a **fresh** model at `o_i + 90 days`
and evaluate it on the *same* calendar blocks 4–6 as the aged model. If the aged-minus-fresh gap is
zero while β₁ < 0, β₁ is calendar, not age. Only design that identifies decay directly; one extra run
per origin. **Report the gap on RelMSE, never on scaler-space MSE** — see `D60i` in §9.2.

**Rolling, not expanding**, because with an expanding window the training set size changes at each
origin and model age cannot be separated from training data volume. **Caveat (`D45`):** gap density is
monotone in calendar time, so per-origin training loss runs **11.2% at origin 6 down to 0.0% at
origins 14–15** and the surviving count ranges **13,558 … 15,217** — partially reintroducing the
volume variation the fixed window was chosen to eliminate. Control by **subsampling every origin's
training set to 13,558 windows** and report the uncontrolled version as sensitivity. Per-origin
figures: `docs/ORIGIN_WINDOW_BUDGET.md`.

**Cluster dependence (`D28`).** A 24-month window advanced 5 months means consecutive origins share 19
of 24 months — **79.2%** of their training data; two apart 58.3%, three apart 37.5%, four apart 16.7%.
Windows become disjoint only at **stride 5**, so the training-disjoint subset holds just **3 origins**
— {2020-01, 2022-02, 2024-03} or any of the four parallel triples. The clusters used for inference in
§9.2 are therefore **not independent draws**. State this numerically, never as "calendar adjacency".

### 8.2 Purge

The `− H` term in window enumeration is the purge: the last retained training window has a target
ending exactly at the boundary, so **no observation is discarded** — only ~24 window configurations
out of 13,558–15,217 (`D25`). The purge and the segment law share their logic: neither discards
observations, both discard window *configurations*.

**Two boundaries, not one (`D24`).** Training windows are enumerated to `val_start − L − H`, so the
last training target ends at `val_start`. Without the train→validation purge, a training window whose
H-step target reaches past the 21-month mark carries validation observations into training — and
validation is what decides early stopping and ridge α, so the contaminated split is the one governing
*model selection*. Small (~24 windows) and in the defect class §11 calls fatal.

The asymmetry is deliberate: a validation window's 96-bar **input** may reach back into the training
period. That is past information legitimately available to a forecaster at that moment (§8.3), and
blocking it would make the evaluation unrealistically pessimistic. Only **targets** are purged.

Log the rejection count per origin and assert it against the **per-origin** break table, never the
pooled §4.3 estimate (`D45`): origins differ from ~12 breaks down to zero, so a pooled assertion fires
spuriously at most origins, gets loosened until it passes, and disarms the one defence against
positional-index drift.

The scaler is fitted on the **21-month sub-block only**, at every origin, never on validation or test.
Moving `train_end` is a leak, not a mismatch.

### 8.3 Why no embargo — the written justification (`D15`)

An embargo guards against test-period information reaching the training set. Two paths exist, both
closed:

1. **Label overlap** — a training window whose target extends past `T_end` into the test period.
   Closed by the H-step purge.
2. **Feature lookback** — a feature at a test timestamp computed from a window reaching back into the
   training period. That is *past* information, legitimately available to a real forecaster; blocking
   it would make the evaluation unrealistically pessimistic. It cannot run the other direction because
   **no feature uses a rolling window** (§5.3): every variate is a per-bar function, so no test-period
   bar can influence any training-set feature value.

The remaining train→test channel is the model weights themselves, which is the object of study. Hence
no embargo. This argument depends on §5.3 and must be re-examined if any rolling feature is ever
introduced.

### 8.4 Rejecting CPCV — include this paragraph in the methodology

> Combinatorial Purged Cross-Validation (López de Prado, 2018) was considered but not adopted. CPCV
> generates backtest paths with non-chronological block ordering, under which *time-since-training* —
> the primary independent variable in RQ2 — is undefined. CPCV also assumes stability of the
> data-generating process across blocks, an assumption this study explicitly tests. Walk-forward was
> chosen because it preserves temporal ordering, remains consistent with evaluation protocols in the
> long-term time series forecasting literature, and still applies purging of H steps at every training
> boundary.

Answer the counter-argument rather than ignoring it: the Knowledge-Based Systems (2024)
backtest-overfitting comparison concludes CPCV beats walk-forward. Its target is *strategy selection*
among many candidates, where block shuffling is desirable; this is a *controlled architecture
comparison* where time-since-training is the independent variable.

### 8.5 Stage gates

| Stage | Gate |
|---|---|
| **2** Data validation | Set the analysis window from the coverage report, not from assumption. **Evaluate window loss per (origin, block), not on the pooled series** (`D45`). Loss > 20% in any test block → report the surviving-window count in Table 5 and add block coverage as a regression covariate; never relax the segment rule. Emit the per-origin break table and the measured `H == L` count |
| **3b** Pre-model measurement | Measured PR at K=8 **< 5.0** → **report it and proceed unchanged, disclosing the divergence from §5.2** (`D48`). The action is disclosure, not a re-cut: D01 establishes exactly one consistent cut exists |
| **5** RQ1 pilot | `use_norm` scale-invariance test first. Then **origin 1 only, 4 K × 3 seeds, on that origin's validation sub-block** (`D27`). **Gate is K=1 vs K=8, never K=1 vs K=12** — K=12 is built to be redundant. Test: Clark–West at α = 0.05, one-sided (`D29`; the pair is nested, so standard DM is biased against the effect). If K=8 does not beat K=1, reposition the title to the descriptive variant **now, not in week nine**. Also estimate between-origin dispersion of the within-slope and publish the **minimum detectable β₁** before any test block is opened |

**All three gates have run (`D60a`, 2026-08-11).** A gate whose outcome lives somewhere else is a gate
a future session can miss.

| Stage | Outcome | Action taken |
|---|---|---|
| **2** | Window budget matched `docs/ORIGIN_WINDOW_BUDGET.md` at **all 15 origins by exact equality**; 75,094 bars, 3 unusable and they are the *same* 3 bars (`D51c`); worst test block 439/720 | **Passed.** No relaxation; per-block surviving counts logged for §9.2's coverage covariate |
| **3b** | PR at K=8 = **4.393 < 5.0**; K=12 PR (3.984) *below* K=8's (4.269); `corr(K, K_eff) = 0.828` against the ≈0.97 expected | **Disclosed, not re-cut.** Grid proceeded unchanged; §4.1b reports the divergence from §5.2's reasoned 1/3.5/6.5/7 |
| **5** | `use_norm` invariance rel **2.68e-06**; single-batch overfit **1.055e-10** at `dropout=0.0`; 12 pilot runs; **Clark–West K=1 vs K=8 on validation `S* = +0.8759, p = 0.1906` one-sided, `T = 1845, h = 24`** | **FAILED.** Title repositioned 2026-08-20 (§1, `D60a`). K=16's clause 1 is thereby failed and the arm is not run (§5.3) |

**Stage 5 failing is not Stage 5 being uninformative.** It measured exactly what it was built to
measure: at origin 1, on leak-free validation data, eight variates do not beat one at α = 0.05. The
gate then did its job — it repositioned a claim before fifteen origins of test blocks were opened,
which is the entire reason `D27` moved it off the test set. Record it in §13.2 as a **selection
event**, separate from the DSR trial count. **MDE published as required: −0.000920** at 80% power,
α = 0.05, against an observed **+0.000256** — inside the MDE, which is §9.2 requirement 6's trigger.

**The Stage 5 gate runs on validation, not on test (`D27`).** §11's final item requires test blocks be
opened once, after the design is frozen; a gate that repositions the title on a test-block result
cannot coexist with it. It also matters mechanically: §10.5's idempotence plus §10.4's deterministic
`run_id` mean a resumed session finds the pilot runs complete and feeds them verbatim into Table 4 and
the β₁ regression, so a test-block pilot would put the origin that decided the framing back into the
evidence for it. If a test-block gate is ever genuinely required, designate one origin a burnt
hold-out, exclude it from every table and from the regression, and state the reduced cluster count.

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

**`D05` — decay is measured against a same-block baseline.** Comparing `MSE(b)` against `MSE(1)`
across *different calendar months* conflates model decay with the market getting harder; normalising
by the block's own naive baseline removes period difficulty. `A(b)` needs no such control — both
models are evaluated on the same block, so difficulty cancels in the ratio, and it cancels *well*
because `MSE_model` and `MSE_naive` on the same block correlate near 1.

**`D23` — `D(i,b)` is on the skill scale**, where it runs 0 (no decay) → 1 (edge gone) and τ ∈ {2.5%,
5%, 10%, 50%} are commensurate. **Guard the denominator:** an origin with `R²_oos(i,1) ≤ 0` contributes
no `b*` and is excluded, stated as such — never silently dropped.

**The guard is not an edge case. It is the only case (`D60b`).** All **15 of 15** origins have mean
`R²_oos ≤ 0` and are excluded by name; `decay_panel.parquet` has **zero rows**; `b*` is **UNDEFINED**
at every τ; the log-rank test is unavailable because neither arm has a surviving origin, so **H3 is
untestable rather than rejected**. Two consequences carry into the manuscript: RQ2's `A(i,b)` is a
ratio built from two negative skills, so its sign is not interpretable the way §3 assumed; and report
*"the decay estimand is undefined under non-positive out-of-sample skill"*, **never** *"no decay
detected within 180 days"*.

**`D41` — `b*` carries an origin index and a confidence interval.** `b*(i) = min{b : D(i,b) > τ}`,
**right-censored at 6**. That is interval-censored survival data: report the **median `b*` with its
CI** per τ from a Turnbull/Kaplan–Meier estimator on the 30-day grid, and test H3 with a log-rank test
across K or an interval-censored AFT model with K as covariate. Table 5 carries the interval, never a
bare integer; the abstract's recommended cadence *is* that interval. `min{·}` does not commute with
averaging, so pooling MSEs across origins and *then* taking the minimum is a different estimand and is
forbidden.

**`D05` follow-on — do not divide by a single reference block.** `RelMSE(i,1)` is estimated from one
30-day block under heavy tails and would sit in the denominator of `D(i,2)`…`D(i,6)`, making their
errors perfectly correlated; an unlucky block 1 then moves the threshold crossing by whole blocks.
Normalise against the within-origin mean, or fit a within-origin trend and read the crossing off the
fitted line, with a block-bootstrap band (stationary bootstrap, block length ≥ 24).

**`D32` — RQ1 is a panel comparison, not an OLS on three points.** Four rungs give three ΔMSE values;
stacking 360 rows creates no information about a K_eff slope varying only between rungs — effective
G = 3, adjacent deltas share an MSE (mechanical correlation ≈ −0.5), 1 residual dof. Two changes make
the horse race identifiable: **K_eff is measured per origin** on that origin's own 21-month training
sub-block (§5.4), leak-free and varying; then fit `MSE(i,b,K) = γ_ib + f(K) + ε` with (origin × block)
fixed effects clustered by origin, `f(·)` as free rung effects, and compare the K and K_eff
explanations as a **non-nested model comparison** (Vuong, or Davidson–MacKinnon J). Report
`corr(K, K_eff)` in Table 2b — **measured 0.828**, and a reader is entitled to know that first.

**Order of operations for seed averaging (`D42`).** Every ratio metric — `A`, `A_attn`, RelMSE,
`R²_oos`, `D`, ΔMSE — is formed from **seed-averaged MSEs**, never from an average of per-seed ratios.
The two differ by Jensen, and the second additionally requires pairing seed 42 at K=1 with seed 42 at
K=8, which are independent training runs of different models: any of 5! orderings gives a different
answer. State n per cell; the cell mean still carries Monte-Carlo error, which enters as measurement
error in the dependent variable — unbiased for β₁, inflating residual variance.

**The log ratio is an appendix robustness check, not a parallel column.** With `A` bounded in roughly
[0, 0.005], `log(MSE_K1/MSE_K8) = A + A²/2` differs from `A` by under 0.1% relative. Use linear `A`
throughout; revisit only if `A` empirically spans an order of magnitude — that trigger is stated now.

**Report two metric scales.** MSE in scaler space for literature comparability, RMSE in raw log-return
units for interpretability, with σ_g stated so the two reconcile. "RMSE 0.0043 on hourly log-returns"
tells a reader far more than "MSE 0.187 on normalized data".

**MAPE is forbidden on log-returns** — the denominator explodes near zero. On reconstructed prices it
is dominated by random-walk behaviour and uninformative. The Lewis (1982) thresholds common in
Indonesian journals do not apply here; say so once in Related Work.

**Directional accuracy at H=24 (`D21`).** Report DA at step h=1, at step h=24, and on the **cumulative
24-hour return**; without a null hypothesis DA is a descriptive number. The three variants do **not**
share a testing regime. DA at h=1 on hourly-spaced forecasts is tested with Pesaran–Timmermann (1992).
DA at h=24 and on the cumulative return are computed on **non-overlapping** windows matching §13.5's
execution rule and tested with PT on that sample; on hourly spacing their targets overlap by 23 of 24
hours (lag-1 autocorrelation ≈ 23/24), so PT's variance is far too small and the test over-rejects
badly. Overlapping-window versions are **descriptive only, no p-values**. State the resulting power
loss (T = 30 per block) rather than recovering it from the invalid sample.

### 9.2 Mandatory tests

**Diebold–Mariano for every comparative claim — but not the same DM for every pair (`D29`).** The
comparisons that carry the paper are **nested**: the ladder is cumulative, Naive-RW (`ŷ = 0`) is
nested inside every model in §7, Ridge-K1 inside Ridge-K8. Under the null of equal population
predictive ability with nested models and estimated parameters, the loss differential has a mean
shifted away from zero and the statistic is not asymptotically N(0,1) (Clark & McCracken 2001;
McCracken 2007). Standard DM is therefore systematically **undersized against the alternative this
study exists to establish**.

| Pair type | Statistic |
|---|---|
| **Nested** — K=1 vs K=8, any model vs Naive-RW, Ridge-K1 vs Ridge-K8 | **Clark–West (2007)** adjusted statistic (add `Σ(ŷ_small − ŷ_large)²` back to the differential), or Clark–McCracken ENC-NEW/MSE-F with their non-standard critical values. Name which |
| **Non-nested** — iTransformer vs DLinear vs PatchTST vs LSTM | standard DM with the HLN correction below |

Acknowledge Diebold (2015), which argues DM remains valid when *forecasts* rather than models are the
object, and take a position rather than leaving it silent.

**The variance estimator is rectangular, not Bartlett (`D34`).** At H > 1 forecast errors overlap, so
use the **truncated (rectangular)** long-run variance estimator
`V̂(d̄) = [γ̂₀ + 2Σ_{k=1}^{h−1} γ̂_k]/T` with lag `h−1` (23 at H=24), as in Diebold–Mariano (1995) and
`forecast::dm.test`. **Do not use Newey–West Bartlett weights**: under the DM null, h-step optimal
forecast errors are MA(h−1), so all autocovariances to lag 23 are genuinely nonzero and equally real,
and Bartlett weights shrink γ̂₂₂ by ~92%, understating the long-run variance and producing exactly the
over-optimistic p-values this paragraph prevents. `statsmodels` `cov_hac` is Bartlett by default. The
rectangular estimator is not guaranteed positive in finite samples: if `V̂ ≤ 0`, fall back to Bartlett
with automatic bandwidth and **report that the fallback fired for that pair**.

Apply the Harvey–Leybourne–Newbold small-sample correction

```
S* = S · sqrt[ (T + 1 − 2h + h(h−1)/T) / T ]
```

and compare `S*` against Student-t with `T−1` dof, **not** the standard normal. Validate any custom
implementation against R's `forecast::dm.test()`.

**Pin the DM sample, or Table 6 is not reproducible.** DM is computed **per (origin, block)** on the
overlapping hourly loss differential: `T ≈ 720`, `h = 24`, truncation lag 23. Block-level statistics
are combined across cells by a stated method — **never** by concatenating `d_t` across origins,
because the model changes at each origin and the DM null has no interpretation across that boundary.
Assert `(T + 1 − 2h + h(h−1)/T) > 0` before applying the HLN factor and refuse to report where it
fails: at `h = 24` the factor is exactly 0 at `T = 24` and 0.047 at `T = 30`, precisely the T a
non-overlapping 30-day block produces. State T alongside every reported p-value.

**Multiplicity: SPA and Reality Check are the wrong tool for a pairwise matrix (`D35`).** White's
Reality Check (2000) and Hansen's SPA (2005) test a *one-against-many* null and say nothing about the
all-pairs comparisons Table 6 contains; with 8+ models that matrix holds 28+ tests and expects ~1.4
spurious rejections at α = 0.05. Therefore:

- **pairwise matrix** → **Romano–Wolf (2005) stepdown**, controlling FWER across all pairs;
- **"which models are indistinguishable from the best"** → **Model Confidence Set** (Hansen, Lunde &
  Nason 2011) at 90% and 75%, as a membership column;
- **SPA retained only** where the paper genuinely poses a one-against-many null ("does any
  iTransformer configuration beat Naive-RW"), labelled as answering that specific question.

**Decay regression — the paper's core claim (`D06`).**

```
correct:   A(i,b) = αᵢ + β₁·b + ε          ← origin fixed effects
wrong:     A(b)   = β₀  + β₁·b + ε
```

Without `αᵢ`, β₁ absorbs origin-level difficulty. Six requirements:

1. **Origin fixed effects**, so β₁ is identified from within-origin variation across blocks.
2. **Cluster by origin. The effective n for β₁ is the cluster count, not the observation count**
   (`D42`). The panel is balanced, so with origin fixed effects β̂₁ reduces algebraically to the mean
   of the origin-specific within-slopes: inference on the core claim is a one-sample test on **G = 15**
   numbers. Citing "15 × 6 = 90 observations" invites the reader to infer power that does not exist —
   state both, and state that effective independence is bounded near 4 by training-window overlap.
3. **Wild cluster bootstrap — the complete recipe (`D42`)**, because every unstated choice moves the
   p-value and §12 requires the number be regenerable: **restricted (WCR — impose H₀ when generating
   samples**, not unrestricted; WCU is severely size-distorted at small G, MacKinnon, Nielsen & Webb
   2023**)**, bootstrapping the **cluster-robust t-statistic** (not β̂ — the asymptotic refinement
   comes from bootstrapping *t*, Cameron, Gelbach & Miller 2008), **B = 99,999**, cluster = origin,
   **one-sided** test of H₁: β₁ < 0 at **α = 0.05 declared in advance**. `wildboottest` with
   `impose_null=True` in Python, `fwildclusterboot` in R. **Report both Rademacher and Webb 6-point
   weights**; if they disagree the more conservative is the headline. At G = 15 Rademacher admits
   2¹⁵ = 32,768 draws (min two-sided p ≈ 6e-5), so the small-G concern no longer binds — a genuine
   side-benefit of `D26`'s re-cut. **The bootstrap p-value uses `(1 + count)/(1 + B)`** (Davison &
   Hinkley 1997) — the observed statistic belongs to its own reference distribution, and
   `mean(t* ≤ t_obs)` has no floor and returned a literal p = 0 (`D53d`).
4. **Reference distribution and rank.** Cluster-robust *t* is referred to **t(G−1) = t(14)**, never
   the software default t(N−K) — a ~9% difference in critical value at 5% two-sided. With 15 origin
   dummies plus a slope the CRV meat matrix has rank ≤ 15 against 16 parameters, so it is singular:
   **only single-coefficient tests are reported**, never joint ones.
5. **Average over seeds first.** Seeds are computational noise, not population draws. See §9.1's order
   of operations — seed-averaged MSEs first, ratio second, never the reverse.
6. **Publish the minimum detectable β₁ before opening the test blocks**, and **pre-register the
   interpretation of a null**. If the MDE exceeds the plausible magnitude of `A`, reposition RQ2 as
   descriptive *before* running the grid — otherwise a non-significant β₁ is indistinguishable from a
   design that could never have detected decay.

   **Done, and the trigger fired (`D60b`).** MDE at 80% power, α = 0.05, G = 15, from the pilot's
   between-origin dispersion: **β₁ = −0.000920**. Observed **+0.000256** — wrong sign and **inside**
   the MDE, so RQ2 is reported as **descriptive**. State the null's interpretation in exactly these
   terms: the design could not have detected a decay smaller than −0.00092, and the estimate points
   the other way; report the two together or a reader will read power into a null that has none.
   Supporting numbers: `t = +0.717`, cluster SE 0.000358, WCR one-sided p = 0.7381 (Rademacher) /
   0.7346 (Webb), B = 99,999, G = 15, N = 90. The five training-disjoint triples (`D28`, G = 3 each)
   return β₁ from **−0.000926 to +0.001625**, p from 0.0761 to 0.9322 — inconclusive in both
   directions, which bounds what the full-panel p-value can honestly claim. The uniform-attention
   decomposition `A_attn` gives β₁ = **−0.000006**, p = 0.3387: attention neither decays nor helps.

**The falsification arm is reported on RelMSE, never on scaler-space MSE (`D60i`).** The notebook's
`mean(aged − fresh) = −0.053341` over 45 cells **is not interpretable as written**: the two arms are
fitted 90 days apart and carry **different `σ_g`** (0.009151 vs 0.007297 at origin 1), so it compares
numbers in different units — the matching naive baselines differ by −0.053196, i.e. **~99.7% of it is
scaler drift**, and the sign reads backwards. On the scale-free metric the gap is
**mean(aged − fresh) RelMSE = +0.000828**: the fresh model better by 0.083% of naive MSE, H2's
predicted direction, at a magnitude whose **sign is not stable** — it flips at **6 of 15** origins,
with 2022-07 at **+0.000046** and 2024-08 at **−0.000012**, both within 5e-5 of zero, so the count
depends on whether RelMSE is built from seed-averaged MSEs or averaged per cell (`D62f`). Verdict: the
arm is **uninformative at this effect size**, matching what the MDE says about β₁ — expected, since it
identifies the same quantity. **The raw-MSE figure must not appear in the manuscript. Any cross-origin
model comparison is on RelMSE or `R²_oos`, never on scaler-space MSE** — the general rule this defect
buys.

**Disclose the residual dependence — the structural mechanism, not the weaker one (`D28`).** With a
24-month window advanced 5 months, consecutive origins train on **79.2% identical data** (58.3% /
37.5% / 16.7% at strides 2–4), and each origin's test period lies **entirely inside** the training
window of later origins. Two consecutive origins' models are near-identical fits to near-identical
data, so their `A(i,b)` series are dependent by construction — not merely through shared volatility.
Cluster-robust inference assumes independence *between* clusters; the effective number of independent
training sets is bounded near **4**, far below G = 15, so the bootstrapped p-value on β₁ is
**anticonservative by an unquantified amount**. State the overlap fractions numerically in Methodology
and Limitations, and report the **training-disjoint re-estimate** (stride 5, G = 3, all five triples
with their spread). At G = 3 the estimate will very likely be inconclusive, and **that is itself the
finding**. A moving-block bootstrap over calendar time is the acceptable alternative to an
i.i.d.-over-clusters bootstrap; report which was chosen and **state the cluster count explicitly**.

**Also add block coverage as a covariate, or re-run β₁ on well-covered blocks only (`D45`).**
Test-window survival is conditioned on *future* gaps and outages cluster on stress, so within an
origin the surviving sample composition trends — and β₁ would absorb it.

**Bind the dispersion measure to the aggregation level (`D30`).** Seed dispersion measures
re-initialisation noise on one fixed dataset; origin dispersion measures the sampling variability of
the estimand, and in walk-forward crypto evaluation the second is typically an order of magnitude
larger. Reporting seed-std as "±" on an origin-aggregated row understates headline uncertainty by
roughly that factor — reintroducing through the reporting convention exactly the overstated precision
the wild cluster bootstrap prevents. Therefore:

- **per-cell (origin, block)** numbers → mean ± std across seeds, **with n stated**;
- **any number aggregated across origins** (Table 4 included) → mean ± **standard error across
  origins**, or a cluster-bootstrap CI, with seed-std as a separate Monte-Carlo-noise diagnostic
  column.

**The inferential unit is the origin. Seed dispersion is a diagnostic, never the uncertainty on an
aggregated estimate. Never a single number.**

---

## 10. Execution — Kaggle T4

### 10.1 Envelope

| Limit | Value | Consequence |
|---|---|---|
| Session runtime | **12 h** | Self-imposed budget stops earlier so the version saves |
| GPU quota | **30 h / week / account** | Never the binding constraint it was written to be (`D57`) |
| Idle timeout (interactive) | 20 min | Grid execution uses *Save Version → Save & Run All*, never the editor |
| `/kaggle/working` | 20 GB, saved as version output | Predictions ≈ 0.5–2 GB — fits with room |
| `/kaggle/input` | read-only | Resume reads from here; everything is *written* to `/kaggle/working` |
| GPUs | 2 × T4, sm_75, 16 GB each | **Both are used** — one worker per device (`D68`, §10.3). 16 GB is per device and is nowhere near binding: the largest training tensor is 70.12 MB |

### 10.2 Run accounting

| Arm | Tag | Runs | Composition |
|---|---|---|---|
| Main grid | `itr` | 300 | 15 origins × 4 K × 5 seeds (`D49`) |
| Uniform attention | `itru` | 75 | 15 × K=8 × 5 seeds (`D50`) |
| Falsification (fresh at `o_i + 90 d`) | `itrf` | 15 | 15 origins (`D26`) |
| Horizon sweep | `itr` | 144 | origins 1, 5, 10, 15 × 4 K × 4 H × 3 seeds (`D08`, `D48`), **H=24 slice deduplicated** against the main grid (`D53e`) |
| Ridge | `rdg` | 60 | 15 × 4 K (`D17`) |
| DLinear | `dlin` | 45 | 15 × K=8 × 3 seeds |
| PatchTST | `ptst` | 45 | 15 × K=8 × 3 seeds |
| **Subtotal — the manifest that ran 2026-08-11** | | **684** | 534 iTransformer + 150 baselines |
| Attention capture | `itra` | 45 | 15 × K=8 × 3 seeds — Figure 5's maps, **and a bit-exact reproducibility check of the main grid** (`D62d`) |
| Long schedule | `itrl` | 90 | 15 × K ∈ {1, 8} × 3 seeds, `lr_halve_every = 8`, 60 epochs, patience 10 (`D62c`) |
| Capacity | `itrc` | 75 | 15 × K=12 × 5 seeds at `d_ff = 512` — §6.2's own pre-registered run (`D62b`) |
| **Subtotal — the manifest that ran 2026-08-21** | | **894** | |
| LSTM | `lstm` | 45 | 15 × K=8 × 3 seeds — multivariate, target-channel loss (`D64`) |
| Naive-persist | `npst` | 15 | 15 origins, deterministic, K=1 (`D64`) |
| Seasonal-naive | `nsea` | 15 | 15 origins, deterministic, K=1 (`D64`) |
| **Subtotal — after `D64`** | | **969** | |
| Orthogonal K=8 | `itro` | 75 | 15 × 5 seeds, one or two variates per family — **high** effective rank (`D70`) |
| Redundant K=8 | `itrr` | 75 | 15 × 5 seeds, F2 and F3 loaded whole — **low** effective rank (`D70`) |
| Lookback L=48 | `l048` | 75 | 15 × K=8 × 5 seeds (`D70`) |
| Lookback L=192 | `l192` | 75 | 15 × K=8 × 5 seeds (`D70`) |
| Tuned | `itrt` | 75 | 15 × K=8 × 5 seeds at the config origin 1's **validation** preferred (`D70`) |
| **Exploratory arms raised from 3 seeds to 5** | | +186 | horizon 144→240, DLinear/PatchTST/LSTM/attention 45→75 each, longsched 90→150 (`D70`) |
| **Total manifest** | | **1,620** | **complete on disk, 2026-09-03** — see the measured row below |

**Measured on disk, not planned (2026-09-04).** All **1,620** runs carry `status: complete` and a
**single** `code_sha256` `bfb43f21028da322…`, so `D62g`'s mixed vintage is gone: the previously
completed 894 were **re-run**, not resumed, which is also how `D76`'s corrected tuned configuration
reached the 75 `itrt` runs. `meta/` holds 1,621 files — the runs plus `tuning_selection.json`.
Per arm: `itr` 540 · `itrl` 150 · `dlin` `itra` `itrc` `itro` `itrr` `itrt` `itru` `l048` `l192`
`lstm` `ptst` 75 each · `rdg` 60 · `itrf` `npst` `nsea` 15 each. The counts above this row are the
manifest as it *grew*; these are what ran. **`notebooks/outputs/RUN_ANALYSIS.md` is the authority on
what the grid returned**, and its own provenance table names the superseded `36fa9c77…` vintage.

**`D70`'s five arms are exploratory, declared before running, and reported whatever they show** —
§13.2's commitment, which an arm reported only when it agrees with the headline does not meet. None
enters RQ1's ladder comparison; each gets its own row.

**The matched-K pair is the one that changes an RQ's evidence rather than its robustness.** RQ1 asks
whether benefit tracks nominal K or K_eff, and the ladder can only answer through a panel, because the
two move together there — `corr(K, K_eff) = 0.828`, separated by a non-nested J-test rather than by
contrast. These two rungs separate them **directly**: same K=8, same target, same seeds, and PR is the
only thing that moves. Measured on the feature frame, **3.609 against 5.011**, either side of the
ladder's own K=8 rung at 4.668.

**What did not move, and the reasons are not budget.** Seeds on the **ladder** stay at 5 (`D18`,
`D49` — the 8→12 rung is the designed contrast and cannot carry the fewest); the K rungs stay as cut
(`D01` — exactly one consistent cut exists); iTransformer's hyperparameters inside the ladder stay
fixed (`D38` — holding capacity fixed is what makes the rungs comparable, and the tuned arm sits
*outside* the ladder for exactly that reason); origin spacing stays at 5 months (§8.1 — effective
independence is bounded near 4 whatever the spacing, so denser origins inflate G without adding
information).

**The three `D64` arms are ordered last, after the `D62` robustness arms**, on the same reasoning: a
session cut short loses a comparator rather than an RQ input. Their tags are new, so the 894 completed
`run_id`s resume untouched — asserted in `tests/test_experiment_plane.py`, which still requires the
pre-`D62` core to total exactly 684.

**Cost: under an hour.** The two naive arms are closed forms and train nothing; LSTM is 45 runs at
roughly iTransformer's per-run cost. Nothing here threatens the 11 h session budget or the weekly
quota, which §10.3 already records was never binding.

The three `D62` arms are ordered **after** the baselines, so a truncated session loses robustness
rather than an RQ input, and are **exploratory** under §13.2's confirmatory/exploratory rule. Zero
`run_id` collisions with the completed 684, asserted in `tests/test_experiment_plane.py`: three new
tags mean three new namespaces, and §10.4 makes a changed component **orphan** prior outputs.

**Superseded counts, kept because trade-offs elsewhere were made against them.** 837 was the nominal
total before `D53e` deduplicated the sweep's H=24 slice; 789 was the ceiling this design implies
(534 + 195 nominal baselines + 60 ridge) but was never executable, because none of the seven baseline
models existed in `src/` (`D56`). Read 789 as the ceiling and 684/894 as what the manifest emits. If a
future addition does not fit, drop the horizon sweep to 3 origins before touching seed counts —
`D30` and `D49` depend on the seed counts.

### 10.3 Cost model — and the regime it depends on (`D19`)

Per origin the training tensor is at most **70.12 MB** (`15,217 × 96 × 12 × 4 B`, the 21-month
sub-block, not the 24-month window — `D25`); targets add ~1.5 MB. The count **varies by origin**,
13,558 to 15,217, so size the buffer at the maximum and slice per origin. **It fits in a T4's 16 GB
many times over, and must be resident there.**

**The required regime:** load the whole split to GPU once, then batch by index-slicing that tensor.
**Do not construct a per-item `Dataset` and `DataLoader`.** At ~280k parameters and ~420–475
steps/epoch the compute is trivial and the run is dominated entirely by data movement and Python
overhead.

| Regime | Per run | Whole grid |
|---|---|---|
| **GPU-resident, no DataLoader** | **~30 s** at 534 (`D57`); **35.0 s** over 684 (`D60d`) | **2.31 h** at 534 on two T4s; **6.52 h** at 684 on one, measured. The 894-run manifest took **7.79 h** on one — the figure `D68` exists to halve |
| Naive `DataLoader`, 4 workers | ~10× worse | ~45 h — **exceeds the weekly quota outright** |

Both are stated so the regime is understood as load-bearing, not stylistic. The original estimate
(60–100 s per run, 10–20 wall-hours) was 2–3× pessimistic per run and 4–8× overall (`D57`); the regime
was right, the arithmetic on it was not.

**Full-grid timing, measured (`D60d`) — and this is the single-device figure, superseded as the *plan* by `D68` but not as the *measurement*:** 684 runs **sequentially in one kernel on a single `cuda:0`**,
wall **6.52 h**, mean **35.0 s**, 684 complete / 0 skipped / 0 failed. Per arm: iTransformer 36.4 s
(4.49 h over 444 runs), uniform 24.3 s (0.51 h), fresh 32.5 s (0.14 h), ridge 0.2 s, DLinear 21.9 s
(0.27 h), PatchTST 95.6 s (1.19 h). The 894-run manifest adds ~3.6 h.

**First result, recorded because §12 requires every number be regenerable.** Origin 1, K=8, seed 42:
`MSE_model = 1.3194`, `MSE_naive = 1.2956`, **`RelMSE = 1.0183`, `R²_oos = −0.0183`**. **It survived
the grid (`D60b`)** — mean `R²_oos` across all fifteen origins and every test block is **−0.0205** at
K=1, **−0.0187** at K=4, **−0.0180** at K=8, **−0.0186** at K=12. Every origin, every rung, negative.
`D20`'s anticipated `+0.004` is off by a sign and a factor of four, and §9.1's guard excludes all
fifteen origins, which is what makes RQ3 undefined rather than large or small.

**Parallelism belongs at the *run* level, never the batch level** — the grid is many small runs, not
one large one. **`nn.DataParallel` is rejected**: at batch 32 the scatter/gather transfer costs more
than the split saves. **DDP is rejected for the same shape of reason and more sharply**: a process
group is set up and torn down per run, and a run is ~32 s, so the overhead is paid 969 times to
parallelise something that is already the unit of work.

**Both GPUs are used, and the mechanism is threads (`D68`).** `execute_parallel` runs one worker per
visible device off a shared queue; `visible_devices()` reports what the session has and the notebook
hands it straight to the grid cell. Threads rather than the **subprocesses** `launch_workers` spawns,
because a subprocess inherits none of the kernel's namespace and §15's notebook carries the package as
definitions in that namespace rather than as files on disk — `launch_workers` remains the path from a
checkout. The GIL is not the constraint: every run spends its time inside CUDA kernels and tensor ops
that release it.

**The 894-run session left half the hardware idle for 7.8 hours**, and the note in the grid cell said
why: *"threads are not the fix — `torch.manual_seed` seeds EVERY CUDA device."* That was true of the
seeding as it stood, and it is what changed. `set_seed` now takes a device and seeds the CPU generator
plus **only that device's**; the CPU generator is shared, so seeding and module construction happen
together under one `SEED_LOCK` — milliseconds against a ~32 s run. Everything after the prologue draws
from the device's own generator, so **a run produces the same bytes whether it ran alone or beside
another**, which is the property `D62d` demonstrated and §12 requires. Single-device behaviour is
unchanged by construction: seeding one device and seeding all of them set the same generator to the
same value when only one is in use, so the 894 completed runs stay reproducible.

**Throughput is unverified off Kaggle and stated as such.** The two T4s exist only there; no machine
this suite runs on has a CUDA device at all. What is tested here is the part that would corrupt a grid
on any hardware — that two workers sharing one cursor never hand the same cell to both, and never drop
one. `D58`'s 2.31 h measurement remains the only run-level figure this project has actually taken.

**Precision.** T4 is sm_75. `torch.cuda.is_bf16_supported()` defaults to `including_emulation=True`
and returns **True** there, selecting *emulated* bf16 slower than fp32. Gate on
`torch.cuda.get_device_capability(0)[0] >= 8` instead, falling back to fp16 + `GradScaler`, then fp32.
At this model size fp32 is likely fastest — measure before choosing.

### 10.4 Run identity and outputs

```
run_id = {model}_o{origin:02d}_K{K:02d}_H{H:03d}_s{seed}      e.g.  itr_o07_K08_H024_s42
```

Deterministic and human-readable. Changing any component deliberately **orphans** prior outputs rather
than silently reusing a mismatched result.

| Artifact | Content |
|---|---|
| `preds/{run_id}.parquet` | `block, step, timestamp, y_true, y_pred` — **raw predictions, always** |
| `meta/{run_id}.json` | resolved config, git sha, `code_sha256`, input-artifact sha256, epochs run, best val, wall time, `status` |

**Persist raw predictions, not just metrics.** They are required for the DM test, per-regime analysis
and the economic evaluation. Re-running the grid because predictions were not saved is an expensive,
avoidable mistake.

### 10.5 Continuation across sessions

**Idempotence.** A run is complete **only when both files exist and `meta.status == "complete"`.**
Anything else is re-run from scratch. Intra-run checkpointing is deliberately omitted: at ~35 s per run
it costs far more complexity than it saves.

**Resume.** Discover completed `run_id`s by globbing `/kaggle/input/*/preds/` ∪
`/kaggle/input/*/*/preds/` ∪ `/kaggle/working/preds/` — **never a hard-coded dataset slug**, so the
Kaggle Dataset name is free to change. Subtract from the manifest and execute the remainder.

**Budget guard.** `SESSION_BUDGET_H = 11.0`, `RESERVE_H = 0.5`, checked **at run boundaries**, not
epoch boundaries. On trip: stop, flush, print the remaining count and estimated sessions left, exit
cleanly so the version saves. **Hitting Kaggle's 12 h wall interactively loses `/kaggle/working`
entirely.**

**The guard measures the session, not the worker (`D54f`).** `BudgetGuard` sets its deadline where it
is constructed, but the 12 h wall runs from cell 0 — so the prelude (Stage 2, 3b, 4 and the twelve
Stage 5 pilot runs, ~20–25 min) would sit outside the budget. The notebook stamps `SESSION_T0` in
cell 0 and hands the grid what is **left** of the 11 h. Measured: prelude 7 min, grid received 10.88 h
of 11.0.

**A partial session is designed for but did not occur.** The 684-run manifest finished in **6.52 h**
inside **one** session. The machinery is verified but unexercised at the session boundary, with one
exception that matters: **resume across a *failed* version is established (`D60e`)** — a version ending
in a papermill error *does* publish `/kaggle/working`, and the grid session opened `already complete:
12  pending: 672`, skipping the twelve pilot runs written before `D59`'s `NameError`. The loss from
that failure was **zero runs**.

Three properties make a partial session safe:

1. **Resume granularity is one run** (~35 s). An interrupted run leaves no meta and is simply redone.
2. **Discovery is by glob**, so the previous session's output Dataset is found under whatever name it
   was given and however Kaggle nested it. Verified against both layouts.
3. **Evaluation is gated on grid completeness.** A partial grid is an unbalanced panel and §9.1's
   estimators refuse one by design — `amplification` raises rather than compare K=1 at eleven origins
   against K=8 at ten. That is correct and stays; what was wrong is that the exception landed in the
   last cells of a twelve-hour session, marking the version failed at the moment its output was the
   only thing worth keeping. The estimators are therefore **not called** until the panel exists.
   Partial evaluation is never the fallback: a half-panel β₁ is a different estimand, not a noisier
   one.

**Session chaining.** Session *N* writes to `/kaggle/working`; Save Version publishes it as a Dataset;
session *N+1* attaches that Dataset as input.

**What is attached, and what is not (`D54`).** Exactly two kinds of Dataset: the **immutable data
artifact**, and the **previous session's output** when resuming. The repository is *not* attached —
`notebooks/iTransformer.ipynb` carries the package as definition cells (§15, `D58`). Uploading the
repository as a second Dataset was the old protocol and its failure mode was silent: notebook and code
Dataset were two artifacts required to agree with nothing checking that they did. The parquet is
likewise never re-downloaded inside the notebook even though Stage 1 could: a fresh download is a new
vintage, and §12 forbids numbers from two vintages sharing a table.

---

## 11. Anti-leakage checklist

Verify before the main grid. **The first ten are fatal.** Organised by the boundary × channel grid of
§5.3 (`D43`), because a two-path enumeration missed four channels entirely.

**Features and labels**

- [ ] **F** — Log-returns computed **per segment**, before the scaler is fitted
- [ ] **F** — Windows validated by **timestamp**, not positional index; no window spans a break
- [ ] **F** — **No imputation anywhere**: no ffill, bfill, interpolation, or reindexing to a full grid.
      Runnable form (`D33`): assert `parquet_rows == bars_actual` and assert the timestamp diff set
      contains the 27 gap blocks
- [ ] Zero-volume and `H == L` bars excluded and treated as segment breaks (`D14`), with the measured
      `H == L` count emitted at Stage 2
- [ ] No rolling window on any feature — verify by inspection, since §5.3 makes this structural

**Scaler**

- [ ] **F** — `StandardScaler` refit at every origin, on the 21-month training sub-block only
- [ ] Identical scaler and scale space for iTransformer and every baseline
- [ ] **F** — Naive-RW mapped as `ŷ_z = −μ_g/σ_g`, not `ŷ_z = 0` (`D31`); `μ_g`, `μ_g/σ_g` logged per
      origin

**Boundaries — train/validation, train/test**

- [ ] **F** — H-step purge active at **every boundary between disjoint splits**: train→validation
      **and** train→test (`D24`)
- [ ] **F** — `max(target_index over training windows) < val_start` and
      `max(target_index over validation windows) < test_start` (`D24`) — window-span assertions, not
      row-index ones. Row-level split disjointness is true by construction and does not imply it
- [ ] Per origin, `train ∩ val == ∅` and `(train ∪ val) ∩ test == ∅` at row level. **Cross-origin row
      overlap is by design** (`D28`) and is handled inferentially in §9.2 — do not "fix" it
- [ ] **F** — Training-window count matches the 21-month arithmetic minus logged gap losses (`D25`) —
      13,545–15,217 on the feature frame (`D52c`), not ~17,400

**Model selection**

- [ ] **F** — Test blocks are opened once, after the design is frozen — **and the Stage 5 gate runs on
      validation, not on test** (`D27`)
- [ ] Hyperparameters: **ARIMA order and ridge α only** are selected on the validation sub-block;
      every iTransformer hyperparameter is fixed a priori and identical at every rung (`D38`)
- [ ] **F** — Loss is MSE on the **target channel only**, at every rung (`D39`)
- [ ] **F** — `use_norm=True` confirmed active; `MSE(c·x)/c² == MSE(x)` passes (`D03`)

**Effective dimensionality**

- [ ] **F** — **Every reported K_eff, including the RQ1 regressor**, is computed on a training-only
      span; the gate additionally uses the pre-first-origin span (`D02`, `D44`). Auditing only the gate
      leaves RQ1's regressor free to read the test period

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

**No number enters the manuscript unless it is regenerable.** Every reported figure must resolve to:

1. a **persisted prediction file** — `preds/{run_id}.parquet`;
2. a **config hash** — the `meta/{run_id}.json` entry naming the code and the sha256 of the input
   feature artifact; and
3. a **documented decision** — a divergence-register row, if the number depends on any departure from
   the source specifications.

**Item 2 names the code by digest, not only by commit (`D54b`).** `git_sha` is `"unknown"` on Kaggle —
there is no git repository there, which is to say the contract lost its code half at exactly the place
the grid executes. `meta/*.json` therefore carries **`code_sha256`**, the hash of the package source
with line endings normalised, beside `git_sha`. It identifies the code that ran, not the commit
someone was standing on with a dirty tree.

**Off-repo the digest is pinned, not computed (`D58`).** `code_sha256()` hashes the `*.py` beside
itself and §15's notebook has no files, so the generator computes the digest from
`src/itransformer_btc/` and pins it as `CODE_SHA256_OVERRIDE`. This is weaker in one specific way that
must be stated rather than glossed: a computed hash cannot lie about the code beside it, a pinned one
is only as honest as the generator. `tests/test_notebook_sync.py` enforces it, asserting every cell
equals its module under the declared transformation — digest and cells are checked by the same run, so
a notebook carrying a stale digest carries stale cells and fails there first.

Likewise `input_sha256` resolves through the **`ITBTC_PARQUET`** environment variable rather than a
repository-relative path, and `input_sha256_source` records whether the digest came from the Stage 1
report or from hashing the parquet. **On Kaggle the file-digest fallback is the operative path, not
the exception (`D60h`)**: all 684 runs logged `"file-digest"` because the attached Dataset carries the
parquet and not the report. The digest is nonetheless the right one — `8270a84b07c2923b…`, §4.1's
pinned vintage — under a single `code_sha256 ee63120991695c6c…` across all 684 metas. Either attach
the report or read the field as informational.

Aggregation writes `paper_numbers.json`, and every table and figure is generated *from that file*
rather than transcribed. **The grid's copy is at `notebooks/outputs/artifacts/paper_numbers.json`**
(immutable evidence); the manuscript's is at **`paper/paper_numbers.json`**, which names the grid file
by sha256 so the two cannot silently diverge (`D60f`, `D62a`). Repo-root `artifacts/` holds one stale
2026-08-06 CPU smoke run and is **not** the results directory. **Numbers produced under different
input-artifact hashes are not comparable and must not share a table.** A number that cannot be
regenerated is a documented failure, not a footnote.

**When the analysis moves while the runs do not (`D62g`).** The 684 runs are `code_sha256
ee63120991695c6c…`; everything written after `D62` is `fec3e8b4af4e453a…`. The vintage that matters
for a number is the vintage of the **runs that produced it**, which is unchanged; the reporting code
is a reader of those runs, not a producer of them. The `D62` robustness arms are the exception — they
*are* new runs at the new vintage — and that is why they get their own table rather than a column in
Table 4.

---

## 13. Paper production

### 13.1 Structure

IMRaD, 10–14 pages. Abstract 200–250 words and **must contain concrete numbers** — the β₁ value, the
percentage decay, the recommended cadence. An abstract without numbers reads as a proposal.

`1 Introduction` · `2 Related Work` (architectures; the channel-independence debate; crypto DL;
evaluation protocols; preprocessing practice; gap synthesis) · `3 Methodology` (3.1 provenance and
segmentation · 3.2 variates and effective dimensionality · 3.3 efficiency tests and pre-model
measurement · 3.4 architecture · 3.5 baselines · 3.6 walk-forward and purging · 3.7 metrics and
clustered inference) · `4 Results` (4.1 data and efficiency · 4.1b effective dimensionality · 4.2 RQ1
· **4.3 RQ2 — core of the paper** · 4.4 RQ3 · 4.5 horizons · 4.6 attention · 4.7 economics ·
4.8 limitations) · `5 Conclusion`, answering all three RQs with numbers.

### 13.2 Mandatory disclosures

Each is a place a reviewer will otherwise find a hole:

- the **K=1 attention degeneracy** as a designed control (§6.2);
- the **K=12 rung's deliberate redundancy** as a designed contrast (§5.2);
- the **CPCV rejection** paragraph, verbatim (§8.4);
- the **`use_norm` / scaler** relationship and the corrected invariance test (§6.3);
- the **no-imputation** defence and why Rubin's taxonomy does not apply (§4.2);
- the **instance-normalisation confound** on F2 and RQ1 (§5.4, `D04`);
- **attention is not explanation** — attention maps are *descriptive evidence of variate reliance*,
  validated for stability across seeds, never causal. Minimum bar: seed-stability reporting plus the
  uniform-attention ablation (Jain & Wallace 2019; Wiegreffe & Pinter 2019). Note that the debate is
  scoped to RNN-era NLP and its transfer to variate-level attention in LTSF is genuinely open — itself
  a limitation sentence;
- **every departure from the source design**, with its reason, per §12;
- **revised-vintage data**: not real-time vintage; state as a known limitation;
- **statistical power** — the MDE and the pre-registered interpretation of a null (§9.2 req. 6).
  Omitting it is the most damaging gap on this list;
- **single-venue, single-pair scope** — everything is Binance BTCUSDT, so the microstructure variates
  are **venue-specific**. An external-validity limit distinct from revised vintage;
- **hyperparameter provenance** — adopted from Liu et al. (2024), not tuned, identical at every rung
  (`D38`), and what that implies for the flat 8→12 rung;
- **study-level multiplicity** — declare which tests are *confirmatory* (β₁ at τ = 5%) and which
  *exploratory* (τ sensitivities, horizon sweep, per-rung DM cells, both `D62` robustness arms).
  §9.2's Romano–Wolf and MCS cover the model matrix only;
- **the Stage 5 pilot as a selection event** (`D27`), separate from the DSR trial count — and note it
  **fired**: the gate failed and the title was repositioned (`D60a`), so the paper's framing is
  downstream of it;
- **the K=16 arm's failed run condition, by clause number** — clause 1 (`D60a`);
- **that no model beats Naive-RW at any rung or any K** (`D60b`, `D60c`) — the headline finding and
  the frame every other result is read inside. Mean `R²_oos`: ridge −0.0006, PatchTST −0.0163,
  iTransformer-K8 −0.0180, iTransformer-K1 −0.0205, DLinear −0.0262;
- **that RQ3 returns no answer because its estimand is undefined**, not because decay was searched for
  and not found (`D60b`);
- **that RQ2 is reported as descriptive** under §9.2 req. 6, with the MDE (−0.000920) printed beside
  the estimate (+0.000256);
- **the training-window overlap between origins** and the effective cluster count (`D28`);
- **the future-conditioned exclusion of test windows** near outages (`D45`);
- **what "K = 8" means for the channel-independent baselines** (`D56`, §7) — *trained on* eight
  channels through shared weights, predicting the target from its own history alone. Left unstated, a
  reader takes DLinear-K8 and PatchTST-K8 to be multivariate predictors in the sense ridge and
  iTransformer are, and the channel-independence pillar loses the distinction it exists to draw. State
  also that their validation losses are all-channel and not comparable to the ladder's;
- **DLinear's internal centred moving average** (`D56`, §7) — rolling, centred, confined to the 96-bar
  lookback, so §8.3's no-embargo argument stands. A reviewer who knows DLinear will look for this;
- **the two `D62` robustness arms as *exploratory*, with their outcomes whatever those are** —
  long-schedule answers "you under-trained", capacity answers "you under-capacitised", both in their
  own table, never mixed into RQ1–RQ3. State also that neither was run to rescue the null: **an arm
  reported only when it agrees with the headline is not a robustness arm**, and a reader is entitled
  to that commitment in writing before the numbers appear;
- **that the epoch cap does not bind at H = 24** (`D62c`, `D78`) — 0 of 300 — so "train it longer"
  is answered by the LR schedule, not the budget; **and that it does bind for DLinear (56 of 75) and
  PatchTST (39 of 75)**, so their losses are truncated-training figures and the ordering must be read
  with that beside it;
- **that the attention arm reproduces the main grid bit-for-bit** (`D62d`), the reproducibility
  statement the paper would otherwise lack, and what licenses reading its maps as maps *of the model
  whose numbers are reported*;
- **that Romano–Wolf removes every rejection against Naive-RW** (`D62a`) — raw Clark–West rejects for
  8 of 11 models at α = 0.05, the stepdown leaves none, adjusted p ≥ 0.336. Reporting the raw column
  alone would claim eight results the study does not have;
- **that Clark–West is positive where `R²_oos` is negative**, and why both are true: the statistic
  credits the larger model for the estimation noise the null imposes (`t` = +2.198 for Naive-RW vs
  `itr-K8` while that arm's `R²_oos` is −0.0180), so the joint reading is that **any population-level
  edge is smaller than the estimation error required to exploit it**;
- **the economic result with its comparator and its DSR beside it** (`D62a`) — **+20.6%** net against
  buy-and-hold's **+29.0%**, annualised Sharpe **+0.377**, **DSR 0.173**, falling to +12.5% and
  Sharpe +0.104 at the top of the slippage band; ridge at −1.02. A positive P&L under a negative
  `R²_oos` is not a contradiction — MSE and directional P&L are different objectives, and a sample
  dominated by BTC's 2020–2026 rise pays a mostly-long position for the drift, not the forecast.
  Report the three numbers together or the first alone reads as skill.

**Priority claims are hedged and documented.** §3's contribution (1) reads "first walk-forward
evaluation of iTransformer on a crypto asset with explicit decay measurement". Written flat it is
refutable by a single search hit, and §13.3 declares every reference-library entry unverified. Write
**"to the best of our knowledge, the first …"** and add a one-paragraph search protocol to §2:
databases queried, exact query strings, search date, inclusion criteria, hits screened. Then let the
weight sit on the substantive contribution — explicit decay measurement with clustered inference under
a pre-registered threshold — which stands regardless.

### 13.3 Citation discipline

**No citation enters the manuscript without a verified DOI and the source read.** Both source `.md`
files self-declare entries "assembled from memory"; treat every one as unverified until cleared.
Known-bad entries (`D16`): arXiv 2509.23494 is dated 2026 but its identifier is 2025; *Symmetry*
vol. 18 is dated 2025 but vol. 17 is the 2025 volume. Target ≥ 60% of references from the last five
years; methodological classics exempt. Use Zotero or Mendeley from the start.

### 13.4 Tables and figures

Each binds to an artifact and a stage. **Figure 3 carries the entire paper.**

| # | Table | # | Figure |
|---|---|---|---|
| 1 | Dataset, gap profile, **per-origin** windows lost, `H == L` count | 1 | Walk-forward scheme with purging and segments |
| 2 | Descriptives + ADF + VR + Hurst | 2 | Architecture and inverted tokenization |
| 2b | Eigenspectrum, PR per rung **per origin**, `corr(K, K_eff)` | 2b | Rolling PR and rolling OLS R² — **establishes H2's premise before any model runs** |
| 3 | Hyperparameters **and K, all models**; epochs-to-stop per rung | 3 | **Decay curve `A(b)` vs b — 15 per-origin lines + fitted `αᵢ + β₁b` with bootstrap band — key figure** |
| 4 | Main results, ± SE **across origins**, n per cell, MCS membership | 4 | RelMSE per block, all models |
| 5 | Per-block `D(i,b)`, surviving-window count, `b*` **with CI** at each τ | 5 | Attention heatmap: calm vs stress, **terciles of realised volatility** |
| 6 | DM matrix — statistic named per pair, T stated, Romano–Wolf adjusted, MCS column | 6 | Horizon sensitivity |
| 7 | Horizon sweep | 7 | Equity curve before and after costs, at three slippage levels |
| 8 | Economics: Sharpe, Sortino, MDD, turnover, DSR, **each with an interval** | | |
| 9 | Exploratory arms (`D62`, `D70`) apart from RQ1-RQ3: `R²_oos` ± SE across origins, the main grid at the same rung, and the difference. An arm not yet run says **not run** by name (`D74`) | | |

**Figure 3 shows one series, not four (`D36`).** `A(i,b)` is defined only as the K=1-versus-K=8 gap and
§3 fixes RQ2 on that pair. Plotting "A(b) for K=1,4,8,12" asks for a quantity identically zero at K=1
and undefined at K=4 and K=12, so whoever generated it would silently invent a *different* estimand
from the one β₁ is regressed on and reintroduce the redundant K=12 rung into a decay comparison §3
forbids. The per-origin-lines form is also the better figure: it displays the actual identification.

**Figure 5's regimes are data-determined (`D48`).** Calm = bottom tercile, stress = top tercile of
realised volatility across all test blocks. Picking windows after seeing the maps would make the
interpretability claim a free parameter.

**State, 2026-08-21 (`D62a`).** `python tools/build_report.py` writes **`paper/paper_numbers.json`**
and renders everything from it; the notebook's cell 9 runs the same functions on Kaggle.

| Deliverable | State | Generator |
|---|---|---|
| Table 1 | **generated** | `report._table1`, measured via `segments.break_summary` and `budget.budget_table`, never transcribed from §4.1 |
| Table 2 | **generated** | `efficiency.efficiency_table`, full sample **and** each origin's 21-month sub-block |
| Table 2b | **generated** | `report._table2b` from the grid's `keff` section |
| Table 3 | **generated** | `report._architecture_section`, read from all metas — where `D62c`'s "0 of 444 at the cap" is visible |
| Table 4 | **generated** | `report._table4`; dispersion bound to aggregation level per `D30` |
| Table 5 | **generated** | `report._table5`, carrying `D60b`'s **undefined** wording and the fifteen excluded origins by name |
| Table 6 | **generated** | `comparisons.pair_matrix` + `comparisons.mcs_table`, 66 pairs over 12 models |
| Table 7 | **generated** | `report._horizon_section`, restricted to the four named sweep origins at **every** horizon so the columns share a sample |
| Table 8 | **generated** | `economics.economics_table`, all three slippage levels, each figure with an interval |
| Table 9 | **generated** | `report._table9` from `_robustness_section`; §13.2 requires the exploratory arms in their **own** table and nothing rendered one until `D74` |
| Figure 2b | **generated** | `keff.rolling_pr` / `rolling_ols_r2`, **descriptive only** (§5.4) |
| Figure 3 | **generated** | `report._figure3`; one series, not four (`D36`), with the MDE drawn beside the fit |
| Figure 4 | **generated** | `report._figure4` |
| Figure 5 | **generated** from the `D62d` arm | `attention.tercile_maps` — the only deliverable needing GPU time, because attention weights were not persisted by the original grid |
| Figure 6 | **generated** | `report._figure6` |
| Figure 7 | **generated** | `report._figure7` from `economics.equity_curves` |
| Figure 1 | **generated** | `report._figure1` from `config.ORIGINS` — every origin's train/validation/test blocks on a calendar axis, purge marked at **both** boundaries, and `D28`'s overlap visible in the staircase. Was *drawn by hand*, which meant it did not exist (`D75`) |
| Figure 2 | **generated** | `report._figure2`, every tensor shape read from a live `ITransformerConfig` so it cannot drift from the model the grid ran (`D75`) |

**Key results from the deliverables, on the 1,620-run grid.** The **Model Confidence Set at both 90%
and 75% contains Naive-RW, all four ridge rungs, and `lstm-K8`** — rank order by mean loss: Naive-RW,
`rdg-K4`, `rdg-K8`, `rdg-K12`, **`lstm-K8`**, `rdg-K1`, then every iTransformer arm, PatchTST,
DLinear. **One deep model is in the set**, which the 894-run report's "and no deep model" no longer
survives: the LSTM's `R²_oos` is −0.00156 against iTransformer-K8's −0.01799, it wins at 15 of 15
origins paired, and it is indistinguishable from ridge (paired p = 0.114). It still does not *beat*
Naive-RW, so §13.2's headline disclosure stands — but the sharper claim built on top of it does not,
and the honest framing is **not "deep learning fails here" but "the attention-based long-sequence
architectures fail here"**.

Across the whole matrix Romano–Wolf now removes **90 of 90** raw rejections, leaving zero of 105
pairs — an artefact of the family, not of the data (`D79`). Within the declared claim families the
picture is legible again: `vs-naive` rejects **nothing** under either correction, so the headline is
robust to the choice; the ladder keeps K=1 against K=4/8/12 and K=4 against K=8, and loses the 8→12
rung, which is the designed contrast behaving as designed.

### 13.5 Economic evaluation

Position from the **sign of the cumulative H-step forecast** — computed on **raw, drift-free**
log-returns, not scaler-space (`D31`) — held H hours, **non-overlapping**; at H=24 that bounds turnover
to one trade per day (`D21`).

Three specifications, each of which moves every number in Table 8 (`D46`):

1. **Phase.** Positions open at **00:00 UTC**. There are 24 admissible alignments of a non-overlapping
   daily partition, each with a different Sharpe, MDD and turnover; choosing the phase after seeing the
   equity curve is a free parameter on the paper's economic claim.
2. **Gap-spanning returns are forbidden.** A position held across a downtime block has no defined
   realised return, and `log(C_{t+24}/C_t)` is exactly the cross-gap return §4.3 forbids everywhere
   else — at the 2018-02-08 block a nominal 24-hour trade would book a 57-hour move. Define the
   holding return **per segment** and **skip** any holding period containing a break. Report skipped
   periods per block: the strategy is flat precisely across outages, which cluster on stress, so
   reported MDD is optimistic by an amount a reader cannot otherwise bound.
3. **Costs.** A **0.04% taker fee per side**, plus slippage at a **pre-registered band of 0.02% /
   0.05% / 0.10% per side**, with Table 8 at all three. Fixing the fee exactly while leaving slippage
   blank fixes the lever that costs nothing and leaves open the one that decides whether the strategy
   makes money; the project's reference library anchors BTC effective spreads near 0.30%.

Report Sharpe, Sortino, max drawdown, turnover and net P&L — **each with an interval**: a Ledoit–Wolf
or Jobson–Korkie/Memmel test for the Sharpe difference against the naive strategy, and bootstrap
intervals for MDD, which from ~180 observations is otherwise uninterpretable.

**The Deflated Sharpe Ratio, made computable (`D46`).** "DSR with N = the number of configurations
tried" cannot be executed and would return ≈ 0 by construction if it could: `SR₀` requires **`V[SR]`,
the variance of Sharpe ratios across the N trials**, plus the skewness and kurtosis of the per-period
returns, none of which N supplies. And N is the wrong quantity — DSR counts candidates whose Sharpe
was computed on the **same return series** and from which the best was selected, whereas the grid's
runs span largely disjoint test periods, seeds, horizons and baselines that never competed for one
backtest. At N = 837 and T = 180 the threshold is `SR₀ + 1.645/√(T−1) ≈ SR₀ + 0.123`, essentially
unmeetable — a second guaranteed null alongside `D23`.

Therefore: **DSR is computed per origin** on that origin's T non-overlapping 24-hour strategy returns,
from the **per-period** Sharpe (never the annualised one — feeding an annualised SR inflates it by
√(periods per year)), the sample skewness and kurtosis of those returns, and `SR₀` derived from
**N = the number of distinct strategy configurations evaluated on that origin's test span**, with
`V[SR]` the observed variance of their Sharpe ratios. The full run total is reported **separately** as
the development trial count and discussed in Limitations — concealing it is selection bias, but it is
not N.

---

## 14. Divergence index

**This section is an index. The long-form evidence for every ID lives in
`docs/DIVERGENCE_REGISTER.md`** — including the full text of passes three through eleven, which was
carried here until 2026-08-24. Severity: **F** fatal · **C** contradiction · **U** underspecified ·
**I** improvement. Every rule the register produced is already absorbed into §1–§13; the index exists
so a defect cannot be re-introduced without someone noticing it has an ID.

### D01–D22 — defects in the source specifications

| ID | Sev | Defect in source | Resolution | § |
|---|---|---|---|---|
| D01 | F | K=8 rung sums to 9; `log_mean_trade_size` double-assigned | Unique consistent cut | 5.2 |
| D02 | F | K_eff gate computed on the full sample, including test periods | Gate on pre-first-origin span; trigger PR < 5.0 pre-registered | 5.4 |
| D03 | F | `use_norm` test asserts identical loss after ×100 scaling | `MSE(c·x)/c² == MSE(x)` | 6.3 |
| D04 | F | Instance norm strips volatility *level*, confounding RQ1 | PR on window-normalised features too; disclose | 5.4 |
| D05 | F | `D(b)` conflates decay with period difficulty | Same-block naive baseline | 9.1 |
| D06 | F | Decay regression omits origin fixed effects | `A(i,b) = αᵢ + β₁b + ε`, clustered | 9.2 |
| D07 | C | "2017-08 history makes 13 origins possible" | False; deleted. Count superseded by D26 | 8.1 |
| D08 | C | Horizon sweep 4 K in one place, 3 in another | 4, matching the 192-cell arithmetic | 10.2 |
| D09 | C | Reference library says "12 origins" | 13, then 15 by D26 | 8.1 |
| D10 | F | Parquet claimed written with `ffill`, 122 synthetic bars | **Closed by D33** — no such keys, no synthetic row ever existed | 4.4 |
| D11 | C | `binance_spot_klines.py` referenced, absent from tree | **Closed by D33** — it is `spot_klines_btc.py` at the root | 4.4 |
| D12 | U | `signed_flow` is a product of two other K=8 members | Kept, dependence disclosed, measured PR settles it | 5.1 |
| D13 | U | F2: per-bar or trailing-averaged, unspecified | **Per-bar, no rolling** | 5.3 |
| D14 | U | `taker_buy_ratio` denominator; `H == L` division by zero | Base-denominated; zero-volume and `H == L` are segment breaks | 4.3, 5.1 |
| D15 | U | Embargo "not applied (justified)" with no justification | Justification written | 8.3 |
| D16 | C | Unverified and mis-dated citations | No citation without a verified DOI and the source read | 13.3 |
| D17 | I | No control for "is a transformer needed at all?" | Multivariate ridge baseline added | 7 |
| D18 | I | 3 seeds too few for a `mean ± std` headline | 5 seeds (extended to every rung by D49) | 6.2 |
| D19 | I | "hours, not days" true only in one regime | GPU-resident regime documented with its counterfactual | 10.3 |
| D20 | I | RelMSE near 1.00 is hard to read | Report `R²_oos = 1 − RelMSE` alongside | 9.1 |
| D21 | U | Trading rule and DA step undefined at H=24 | Cumulative-forecast sign, non-overlapping, H-hour hold; DA at h=1, h=24, cumulative | 9.1, 13.5 |
| D22 | I | No rung tests genuinely nonlinear features | Optional K=16 fifth rung — **not run**, see D60a | 5.3 |

### D23–D50 — defects in *this* document, found by adversarial audit 2026-08-05

Two of five lenses (consistency, Kaggle/execution) never ran; D51–D62 are what they would have found.

| ID | Sev | Defect | Resolution | § |
|---|---|---|---|---|
| D23 | F | τ on `D(b)` is arithmetically unreachable — RQ3 a guaranteed null | `D(i,b)` rescaled to proportional skill loss | 3, 9.1 |
| D24 | F | No purge at train/validation; early stopping selects on contaminated data | Purge at both boundaries | 8.1, 8.2, 11 |
| D25 | F | ~17,400 is the 24-month count; training is 21 months | 13,558–15,217 windows, 70.12 MB | 5.3, 8.2, 10.3 |
| D26 | F | Block index `b` collinear with calendar month at 6-month spacing | **5-month spacing, 15 origins** + falsification arm | 8.1 |
| D27 | F | Stage 5 gate opens test blocks, contradicting §11 | Gate runs on the validation sub-block | 8.5 |
| D28 | F | Consecutive origins share 75–87.5% of training data | 79.2% stated numerically; effective independence ≈ 4; G=3 check | 9.2 |
| D29 | F | Every headline DM comparison is nested; standard DM invalid there | Clark–West for nested, DM+HLN for non-nested | 9.2 |
| D30 | F | Seed std is the wrong error bar on origin-aggregated results | Dispersion bound to aggregation level | 9.2 |
| D31 | F | `ŷ_z = 0` in scaler space is a drift forecast, not a random walk | `ŷ_z = −μ_g/σ_g` | 7 |
| D32 | F | RQ1's regression unidentified at three rung deltas | Per-origin K_eff panel + non-nested comparison | 9.1 |
| D33 | C | §4.1 paths point at an empty `data/raw/`; artifact has out-of-window bar | Paths corrected, boundary bar dropped; **D10, D11 closed** | 4.1, 4.4 |
| D34 | C | Newey–West Bartlett contradicts the `dm.test` validation target | Rectangular truncated estimator | 9.2 |
| D35 | C | SPA/Reality Check cannot correct a pairwise matrix | Romano–Wolf stepdown + Model Confidence Set | 9.2 |
| D36 | C | Figure 3 plots `A` for four K; `A` is defined only for K1-vs-K8 | One series, 15 per-origin lines + fitted overlay | 13.4 |
| D37 | C | Linear-span argument is not a theorem under `use_norm` | Demoted to parsimony; taxonomy carries the exclusions | 5.3 |
| D38 | C | §11 claims validation-based hyperparameter selection that never happens | Provenance stated, no per-rung tuning | 6.2, 11 |
| D39 | U | Target-channel vs all-channel loss unspecified — confounds K | Target channel only, at every rung | 6.2 |
| D40 | U | Baselines carry no K, so channel-independence is untestable | Every baseline given an explicit K | 7 |
| D41 | U | `b*` has no estimator, origin index, or censoring method | Interval-censored survival + log-rank for H3 | 9.1 |
| D42 | U | Wild bootstrap recipe incomplete at every choice that moves the p-value | WCR, studentized, B, α, sidedness, t(G−1) | 9.2 |
| D43 | U | Leakage surface declared closed on a two-path enumeration | Boundary × channel grid; one §11 item per cell | 5.3, 11 |
| D44 | U | K_eff's span undeclared for the RQ1 regressor; PR blind to cross-lag | Per-origin training-only span + lookback-aware measure | 5.4 |
| D45 | U | Window loss reported globally; per-block it reaches 50% | Per-origin/per-block accounting and exact assertion | 4.3, 11 |
| D46 | U | Economics: unfixed phase, gap-spanning returns, unimplementable DSR | Phase fixed, per-segment returns, DSR recipe | 13.5 |
| D47 | U | LR halved each epoch makes the 30-epoch budget decorative | Halve every 4 epochs; log epochs-to-stop per rung | 6.2 |
| D48 | U | Four outcome-determining choices left open after results | Sweep origins, regimes, slippage band, gate action fixed | 5.3, 8.5, 13.4 |
| D49 | I | Flat 8→12 rung has the fewest seeds and no equivalence test | 5 seeds at every rung + pre-registered TOST margin | 3, 6.2 |
| D50 | I | Uniform-attention control budgeted for Figure 5 but never an arm | Promoted to a main-grid arm; `A_attn` defined | 6.2, 9.1 |

### D51–D62 — defects found by *running* the code, the grid, and the deliverables

Each pass names the lens that found it, because the lens is the transferable part: **D51** asserting
the data accounting (2026-08-06), **D52** building the features and the network (08-06), **D53**
building the experiment plane (08-06), **D54** asking what Kaggle needs (08-07), **D55–D58** running
the grid to completion (08-08…08-10), **D59** running the flattened notebook (08-11), **D60** reading
the grid's output against the document that specified it (08-20), **D61** running the test suite
(08-20), **D62** generating the deliverables (08-21).

| ID | Sev | Defect | Resolution | § |
|---|---|---|---|---|
| D51a | F | §4.3's closed form is not an identity — a segment of `n < 120` bars is charged `n − 119` and the negative absorbed silently; 7 origins understated by 39…137 windows | Count segment-wise `Σ max(0, nᵢ − 119)`; closed form kept only as `closed_form ≤ measured` | 4.3 |
| D51b | F | Test-block survival accounted with *training* semantics → 601/720 phantom loss on a clean block | Test blocks hold **720** forecast origins; 74 of 90 cells clean, worst 439/720 | 4.3 |
| D51c | C | `H == L` count never measured, assumed additive to zero-volume | Measured **3**, and the *same* 3 bars. Total unusable is 3, not 9 | 4.3 |
| D51d | I | "5-month spacing visits 12 calendar months per block" true for `b=1` only | 12/7/11/11/11/11 vs 6-month's 2/2/2/3/3/2; D26's conclusion survives | 8.1 |
| D52a | F | "All three F2 estimators strictly positive" is **false for Rogers–Satchell** — vanishes on 33 marubozu bars, `log 0 = −∞` into the K=12 rung | Claim restated for Parkinson and Garman–Klass only; RS uses `log(RS + 1e-9)` | 5.1 |
| D52b | C | `μ_g/σ_g ≈ 0.037` written before measurement | Measured −0.00818 … +0.01733; ~2× smaller, and it **changes sign** across origins | 7 |
| D52c | U | Window budget measured on the **raw** frame; windows are cut from the **feature** frame | Assert on the feature frame, 13,545–15,217; delta is exactly the segment count | 5.3, 11 |
| D52d | I | Single-batch overfit check cannot pass with `dropout=0.1` | Run at `dropout=0.0`; measured 8.26e-10 (6.8e-2 with dropout on) | 16 |
| D53a | F | "Stable rank of the `K × 96` block" is a units artefact — one row dominates both norms | Standardise within window first; `K / λ₁` of the within-window correlation matrix | 5.4 |
| D53b | F | PR of the `K·L × K·L` **covariance** spectrum is not monotone in K | Use the **correlation** spectrum, reported as a fraction of its `K·L` ceiling | 5.4 |
| D53c | F | Sharding the **pending** list races — two workers compute partitions at different moments | Shard the **full manifest**, then subtract what is complete | 10.5 |
| D53d | C | Wild cluster bootstrap returned a literal **p = 0** | `(1 + count)/(1 + B)` — the observed statistic belongs to its own reference distribution | 9.2 |
| D53e | C | §10.2's total double-counts 48 runs — the sweep's H=24 slice shares `run_id` with the main grid | Deduplicate in the manifest: 582 nominal cells are **534** real runs | 10.2 |
| D53f | U | **Stage 3b gate does not pass**: PR at K=8 = 4.393 < 5.0, K=12 PR *below* K=8's, `corr(K, K_eff) = 0.828` | Disclose, do not re-cut (`D48`). K=12 is **more** redundant than designed; the horse race is **more** identifiable | 5.4, 8.5 |
| D54a | C | Launcher needed the repository as a second Kaggle Dataset, kept in step by hand | Notebook carries the package itself. (Its *form* superseded by D58) | 15 |
| D54b | C | `git_sha` is `"unknown"` on Kaggle — §12 lost its code half where the grid runs | `code_sha256`, line endings normalised, in every `meta/*.json` | 12 |
| D54c | C | `_input_sha256()` read a hard-coded repo path, absent on Kaggle | `ITBTC_PARQUET`; report preferred, parquet-hash fallback; `input_sha256_source` records which | 12 |
| D54d | I | Nothing would police the second copy of ~4,000 lines | Notebook is **generated**; `tests/test_notebook_sync.py` + `--check` fail the suite on drift | 15, 16 |
| D54e | F | Evaluation cells crash on a partial session — the estimators are right, but the exception marks the version failed when its output is all that matters | Gate the estimators on `GRID_COMPLETE`; print what remains; exit cleanly | 10.5 |
| D54f | C | Budget guard bounds the worker, not the session — the prelude sat outside the 11 h | `SESSION_T0` in cell 0; grid gets what is **left** | 10.5 |
| D55 | F | `b_star()` inferred its schema from its rows, so the all-origins-excluded case returned a column-less frame and raised — and that case is the **expected** one | `B_STAR_SCHEMA` declares the columns; the RQ3 cell branches to **undefined**, never to "no decay detected" | 9.1 |
| D56 | F | **No baseline model existed in `src/`** — §10.2's 789 was never executable and Table 6 had no inputs | ridge 60 + DLinear 45 + PatchTST 45; manifest 534 → **684**. ARIMA/LSTM/naive-persist/seasonal-naive **deferred with a written reason** | 7, 10.2 |
| D57 | U | §10.3 estimated 60–100 s per run and 10–20 h for the grid | Measured ~30 s and **2.31 h** at 534. The weekly quota was never binding | 10.3 |
| D58 | C | §15 described `%%writefile` cells feeding two GPU subprocesses — a form D57 dissolved | Notebook flattened to **definition cells** in one kernel; `launch_workers` retained for the checkout path | 15 |
| D59 | F | `from itransformer_btc import metrics` binds a **module object** no definition cell defines → `NameError` **365 s into a Kaggle session**. Every existing check passed, because each asks a question the defect answers correctly | Import the **name**. Generator refuses to emit a cell reading a dropped module object; `symtable` walk over every cell against the executed namespace | 15 |
| D60a | F | **Stage 5 failed and the document did not say so** for nine days — §8.5 pre-registered exactly one action on that outcome | Title repositioned; gate-outcome table added; K=16 recorded as not run, clause 1 | 1, 5.3, 8.5 |
| D60b | F | **RQ1, RQ2 and RQ3 all had answers and none was in the document** | Measured-answers table in §3; the `R²_oos ≤ 0` guard is the only case; RQ3 wording fixed | 3, 9.1, 9.2 |
| D60c | C | The baseline ordering inverts the paper's premise and appeared nowhere — ridge loses to Naive-RW by ~30× less than any deep model | Stated in §7 and made a mandatory disclosure. Results **before** RQ1 | 7, 13.2 |
| D60d | U | §7 scaled a CPU ratio onto a T4 and predicted ~6 h for PatchTST alone and two sessions | Measured 95.6 s / 1.19 h / **6.52 h in one session**. Rule: **a throughput ratio does not transfer between devices** | 7, 10.3 |
| D60e | C | D59 left open whether a papermill-failed version publishes its output | It does — `already complete: 12  pending: 672`. Loss from D59 was **zero runs** | 10.5 |
| D60f | U | §12/§15 point at repo-root `artifacts/`, which holds one stale CPU smoke run | Real path is `notebooks/outputs/artifacts/`; six panel parquets inventoried | 12, 15 |
| D60g | U | Tables 6, 8 and Figures 5, 7 had no inputs and were never run | State table added; closed by `D62a` | 13.4 |
| D60h | I | Three unscoped numbers: parameter count across rungs, RS support pre/post κ, `input_sha256_source` | Parameter claim scoped to a fixed horizon; RS frames recorded; file-digest declared the operative Kaggle path | 5.1, 6.2, 12 |
| D60i | F | **The falsification arm's headline number is a units artefact** — two `σ_g`, ~99.7% scaler drift, sign backwards | Report on RelMSE: **+0.000828**. General rule: **no cross-origin comparison on scaler-space MSE** | 8.1, 9.2 |
| D61 | C | The committed notebook is the Kaggle **export**, not the generator's output — suite red on `main` since `9926acd` (2 failed, 84 passed). Code unaffected: `code_sha256` byte-equal | `python tools/build_notebook.py`, then commit. The console evidence lives in `notebooks/logs-iTransformer.txt`, so nothing is lost | 15, 16 |
| D62a | U | Nine tables and six figures promised by §13.4 and never generated; `dm_test()` and `directional_accuracy()` existed, were tested, and were **never called on the panel**; Romano–Wolf and MCS did not exist in code | `src/itransformer_btc/report.py` + `tools/build_report.py` → `paper/paper_numbers.json`, naming the grid file by sha256 | 13.4 |
| D62b | I | §6.2's pre-registered larger-`d_ff` robustness run was never built | `capacity` arm, `itrc`, `d_ff = 512` at K=12, 75 runs | 6.2, 10.2 |
| D62c | I | The null's most obvious attack — "you under-trained" — had no answer; the epoch cap was implied to bind and **does not** (0 of 444) | `longsched` arm, `itrl`, `lr_halve_every = 8`, 60 epochs, patience 10, 90 runs | 6.2, 10.2 |
| D62d | U | Attention weights were never persisted, so Figure 5 had no input | Runtime `capture` attribute, **never** a config field — the branch consumes no RNG, so a captured run is bit-identical and the arm doubles as a reproducibility check | 10.2, 13.4 |
| D62e | F | `D60i`'s corrected falsification figure existed only in prose — §12's contract did not cover the document's own correction | `metrics.falsification_relmse`, in `paper_numbers.json`, reproduces **+0.000828** independently | 9.2 |
| D62f | C | `D60i` says the falsification gap flips sign at 7 of 15 origins; measured, **6** | Report 6, and that two origins sit within 5e-5 of zero. The honest statement is that **the sign is not stable** | 9.2 |
| D62g | U | Grid and reporting code are now two `code_sha256` vintages, and nothing said what that means | The vintage that matters is the **runs'**, which is unchanged; the `D62` arms are new runs and get their own table | 12 |
| D62h | I | Exact string equality on `paper_numbers.json` is the wrong drift guard — polars' parallel `group_by` moves the 8th significant digit on 28 of ~8,000 numbers | `--check` compares structure exactly and floats within 1e-6, reporting the **path** of the first real difference | 15 |

### D63–D64 — defects found by *reading the notebook as an examiner would*

The twelfth pass (2026-08-27) has a lens neither earlier one carries: **what a reader sees**. `D58`
optimised the notebook for a Kaggle session nobody watches and was right to; §1's deliverable then
made it something an examiner opens beside the manuscript, and the shape that served the first
purpose defeats the second.

| ID | Sev | Defect | Resolution | § |
|---|---|---|---|---|
| D63 | I | Eighteen module-sized dump cells — `metrics.py` alone ~1,400 lines in one — so the notebook reads as a `.py` file split at module boundaries rather than as a notebook | `SECTION_MAP`: **137 cells cut by logical group**, each preceded by an HTML markdown heading naming what it does and which rule it enforces. Identity in `cell.metadata`; byte-equality moved from per-cell to per-module; one declared additive line (`from __future__ import annotations`) on every continuation cell, without which Kaggle's Python 3.11 raises `NameError` at class-definition time | 15, 16 |
| D64 | I | Three of §7's four deferred baselines cost minutes and were never built, leaving rows that read as unfinished work — and *no deep model beats Naive-RW* asserted without testing the deep model the crypto literature reaches for first | LSTM (`lstm`, 45 runs, **multivariate**, target-channel loss), naive-persist (`npst`, 15), seasonal-naive (`nsea`, 15). Manifest **894 → 969**. ARIMA stays deferred **with a written reason**: AIC lands at ~(0,0,0), which *is* the naive baseline | 7, 10.2 |

| D65 | **F** | An evaluation cell bound `_provenance` — the name of a `report.py` **function** `build_report` calls one cell later — to a string. The last cell of the 894-run session died `TypeError: 'str' object is not callable` at 28,512 s, 7.8 hours in. `D59` from the opposite direction: not a name the flattening unbound, but one a scaffolding cell rebound. Zero data lost; the version's *status* was | Binding renamed `_digest_source`; `test_no_scaffolding_cell_shadows_a_package_name` refuses any scaffolding cell that **assigns** a package name. Imports excluded — same object, not a value collision. Four allowlisted names, each with its reason in code | 15 |
| D66 | I | All 137 definition cells still opened with their module's imports — the same twenty-odd lines repeated through the artefact a reader examines | One **Library** cell after Setup, **generated from** the flattened modules so nothing can go missing, `from X import` merged per module. Module-level imports become the **third declared subtractive category**; function-local ones stay (`report._pyplot` defers matplotlib on purpose). Equality reference moves to `flatten_module_body` — byte-exact still | 15 |

| D67 | I | `D63` prefixed every definition cell with `from __future__ import annotations`, on the reasoning that a future import is scoped to its code unit. True of `compile()`, **false of the notebook**: IPython accumulates future flags across a session, so 140 cells repeated a line one cell already supplied | The directive lives in the Library cell alone. Definition cells are pure slices — the last additive transformation is gone. Verified empirically, not assumed: `compile.flags` `16896 → 16794112`, and a following cell defines a forward annotation without raising. The suite compiles cells under the accumulated flags, so a change in that behaviour fails here | 15 |

| D68 | I | The 894-run session used **one** of two T4s for 7.8 hours. The grid cell said why — *"threads are not the fix, `torch.manual_seed` seeds EVERY CUDA device"* — a correct observation with a conclusion that did not follow: the seeding was global because nothing had made it otherwise | `set_seed(seed, device)` seeds the CPU generator and **only that device's**; seeding plus module construction under one `SEED_LOCK`, milliseconds against a ~32 s run. `execute_parallel` runs one worker per visible device off a shared queue. Run level, never batch — DP rejected as before, DDP worse again at 969 short runs. **Throughput unverified off Kaggle**; what is tested is that no cell reaches both workers and none is dropped | 10.3 |

| D69 | **C** | `AGENTS.md` was a 2,298-line pre-compaction copy of `CLAUDE.md`, headed `# CLAUDE.md` and declaring itself authoritative — so two files claimed to govern and disagreed on §10.3, the manifest count and the existence of `D63`–`D68`. Nothing referenced it, and nothing announced it as a copy | Replaced with a pointer to `CLAUDE.md`, the register and `USAGE.md`. `D54a` one level up: two artifacts required to agree with nothing checking. **One governing document; a second copy is a defect, not a convenience** | 15 |

| D70 | I | Five arms that `D68`'s second GPU made affordable, and one of them is not robustness: RQ1's K-versus-K_eff contrast existed only inside a panel, because the ladder moves both together (`corr = 0.828`) | **Matched-K rungs** — two K=8 subsets, PR **3.609** vs **5.011**, either side of the ladder's 4.668, so RQ1 is tested by contrast rather than inferred. **Lookback sweep** L ∈ {48, 192}, the one first-order hyperparameter §6.2 never varied. **Tuned arm**, grid declared before running, selected on origin 1's *validation* where `D27` put the Stage 5 gate. **Exploratory seeds 3 → 5**. Manifest **969 → 1,620**, nothing orphaned | 5.1, 10.2 |

| D71 | **F** | `find_parquet` globbed five fixed patterns, deepest `*/*/BTCUSDT_1h.parquet`. The path Kaggle's web UI hands a user — `/kaggle/input/datasets/<owner>/<slug>/BTCUSDT_1h.parquet` — is **three** levels, one past the deepest. The file is attached, visible, and the session dies in the setup cell | `rglob` fallback after the ordered patterns; `data/raw/` still preferred, multiple copies reported rather than silently chosen (§12). Tested against four layouts, extracted **from the committed notebook** (`D55`). §10.5's slug-independence and depth-independence are one requirement | 10.5 |

| D72 | **F** | The 2026-08-27 session printed `parquet   /kaggle/working/__notebook__.ipynb` and died 48 s later inside polars: `File out of specification`. Discovery matched on **name** and never checked the file was a parquet, so a wrong answer travelled three cells before failing, and failed somewhere that could name neither the file nor why it was chosen. **Root cause of the binding is not established** — no committed `find_parquet` can produce that path, and replay raises correctly | `looks_like_parquet`: `PAR1` at both ends, checked at the point of selection. Failing candidates are skipped, so a corrupt copy no longer shadows a valid one; when all fail the error lists them. Rule: **matching on name has verified nothing** | 10.5 |

| D73 | I | The notebook outline was the *package*, not the study: eighteen `##` banners named after `.py` files, with all nine execution stages trailing behind them, so a reader met `config.py` through `report.py` and only then reached the work. Module and section banners were both `<div>` blocks, so the two levels looked alike; heading language was half English, half Indonesian | **Phases**, not files — 24 of them, each carrying its modules and then its execution: config, muat data, pra-proses, split, algoritma, latih, metrik, baseline, grid, evaluasi, laporan. Hierarchy is `##` phase, `###` module or step, `####` section. `MODULE_ORDER` re-cut to that order and asserted against `PHASES` in `build()`; it stays a valid topological order of the import graph, and `package_digest()` hashes `sorted(glob)` so **`code_sha256` is unchanged and no completed run is orphaned**. All prose Indonesian | 15 |

| D74 | **F** | The manifest grew to 1,620 runs and the reporting layer still knew 684 of them. `COMPARISON_KEYS` named 8 of `ARM_MODEL_TAG`'s 17 tags and `ROBUSTNESS_TAGS` three more, so **480 runs produced no table row, no figure line and no MCS membership** — `lstm`/`npst`/`nsea` (`D64`) and the five `D70` arms. Two were load-bearing: LSTM is the deep model §13.2's *no deep model beats Naive-RW* most needs tested, and `itro`/`itrr` is RQ1's only **direct** K-versus-K_eff contrast. And §13.2's *own table* for the exploratory arms did not exist — `render_tables` wrote nine, none of them it | Three `D64` baselines into `COMPARISON_KEYS`; five `D70` arms into `ROBUSTNESS_TAGS`, never the ladder (§10.2); **Table 9** renders every arm, "not run" **by name**; `available_keys` names an arm absent everywhere instead of aborting the report, while `build_panel` still raises on **partial** coverage, which is `D45` | 7, 10.2, 13.4 |

| D75 | **F** | **Two of §13.4's eight figures had never been produced**, and three of the six that had were wrong in the direction that looks like a result. Figures 1 and 2 were marked *drawn by hand* and therefore did not exist. Figure 5's `imshow(..., vmin=0.0)` with no `vmax` rendered every attention panel one flat colour, so a reader took **"attention is uniform"** from the colour bar rather than the data — `D50`'s null asserted by a plotting keyword — with axes labelled `0…7` so the one question the figure answers was unanswerable. Figure 7's axis said *cumulative net log return* over `exp(cumsum(net))`, a wealth multiple: an order of magnitude out in the figure the economic claim rests on, and **buy-and-hold** — the comparator §13.2 states the result against — was not drawn. Figure 4 put 11 series on a 10-colour cycle, painting `itr-K1` and `ptst-K8` identically | Figures **1 and 2 generated** from `ORIGINS` and a live `ITransformerConfig`, so neither can be silently wrong or drift from the model that ran; Figure 5 on a diverging map centred on `1/N` with shared limits and **variate names**; Figure 7 relabelled `net equity multiple (1.0 = break-even)` with buy-and-hold from the same `hold_position` Table 8 uses; Figure 4 hue-by-family, dash-by-rung. **A figure that renders is not a figure that is right** | 13.4 |

### D76–D84 — defects found by *re-reading the 1,620-run grid against the code that produced it*

The thirteenth pass (2026-09-02) ran after the full manifest completed. Its lens is the one no earlier
pass had: **a finished grid, read against the document that specified it and the code that emitted
it**. Four of the nine are consequences of `D64` and `D74` adding models that were right to add — the
cost landed somewhere else and nothing failed.

| ID | Sev | Defect | Resolution | § |
|---|---|---|---|---|
| D76 | **F** | The `tuned` arm ranked eighteen configurations on `d_model × e_layers × lr` and returned a bare `ITransformerConfig`, which has no `lr` — so the winner's `1e-3` was **selected and then discarded** and the arm ran the winner's architecture at the default `1e-4`, a point the same search had not ranked first. Record and run disagreed | `TunedConfig` carries `lr` as a field and returns `TrainSchedule(lr=self.lr)`; the notebook asserts `TUNED_CONFIG.lr == TUNING_TABLE[0]["lr"]` before the grid starts. **The 75 `itrt` runs on disk predate this and must be re-run** | 6.2, 10.2 |
| D77 | U | `lstm-K8` is the only deep model in the Model Confidence Set and the only model in the matrix with no economics row | `("lstm", 8)` added to `ECONOMIC_KEYS`; 225 cells → 270. It is the **worst** strategy in Figure 7 at every slippage level, which is MSE and directional P&L being different objectives, measured | 13.5 |
| D78 | C | `epochs_at_cap` hardcoded 30, so `itrl` (cap 60) was miscounted; and "0 of 444 reached the cap" had gone stale | Cap read from each run's own `meta['schedule']`. Measured: 0 of 300 at H=24, 5 of 80 at H=168 — and **DLinear 56 of 75, PatchTST 39 of 75**, never disclosed before | 6.2, 13.2, 13.4 |
| D79 | C | Widening `COMPARISON_KEYS` from 12 models to 15 took Romano–Wolf from **31 of 66** rejections to **0 of 105** while no effect moved: the two naive comparators carry `\|t\|` up to 8.5 and the shared draw puts them in the max-`\|t\|` null | Second column `p_romano_wolf_family`, stepped down inside a declared claim family, beside the unchanged all-pairs headline. `vs-naive` rejects nothing under either, so the headline is robust to the choice. Post-hoc and labelled so | 9.2, 13.4 |
| D80 | U | §9.2 asks for a coverage covariate **or** a restriction to well-covered blocks. The restriction returns `None` (it unbalances the panel) and the covariate was never built, so neither ran | `panel_beta1_covariate`, Frisch–Waugh inside the origin fixed effects with the same WCR bootstrap. β₁ = **+0.000250**, p = 0.7305, against the uncontrolled +0.000256 / 0.7381 | 9.2 |
| D81 | C | `lookback_covariance_pr` has computed the **correlation** spectrum since `D53b`, but function, field and parquet column all still said covariance — and the correlation spectrum is **not monotone in K** either, which is the reason `D53b` gave | Renamed to `lookback_correlation_pr` / `pr_lookback_corr`. The real justification is scale-freeness. §5.4 mislabels `92.1 / 21.9 / 37.3 / 15.5` as covariance; they are the correlation figures, and the two `K_eff` constructs are **ordinally opposed** | 5.4 |
| D82 | U | Every comparison was printed as two marginal means with marginal SEs, on arms scored on the same fifteen origins. The paired SE is **half** the marginal one, and `itrr` versus `itro` — RQ1's only direct contrast — was computed nowhere | `per_origin_relmse`, `paired_contrast`, a `contrasts` section, a paired column in Table 9. Matched-K measures **+0.001241, t = +6.27, 14 of 15 origins** — half the whole K=1→K=8 ladder gain, at fixed K | 9.1, 13.4 |
| D83 | **F** | Figure 7 averaged each day over whatever origins still had data. Origins carry 146 to 180 tradable days, so the denominator fell across the tail and the first drop-out rendered as a **near-vertical seven-point fall at day 146** — read as a crash, in the figure the economic claim rests on. The early drop-outs are the outage-heavy origins, so the tail was also optimistic | Every curve truncated at the shortest series, the day named in the subtitle, so each plotted day averages all fifteen | 13.4 |
| D85 | **F** | `run_id` encodes no configuration, so §10.4's orphaning rule covers only a renamed arm. `D76` changed the tuned arm's learning rate and none of its 75 ids, so a resumed session would have skipped all of them and reported the superseded configuration under the corrected caption | `pending` passes `code_sha256()` to `completed_run_ids`; within a vintage resume is unchanged, across vintages the grid re-runs. The filter is **off by default** because the report is a reader of runs, not a producer (`D62g`) | 10.4, 10.5, 12 |
| D84 | C | `npst` and `nsea` sit at RelMSE ≈ 2.0 while every model the paper argues about is inside `[1.000, 1.027]`, so Figure 4 was a flat line and `D60c`'s ordering was invisible | Two panels: full range above, data-derived zoom below. Ridge and LSTM on the line, DLinear worst, the iTransformer band between | 13.4 |

**The rule this pass bought, stated once because it fired four times: a change that is correct in one
place is not free in another.** `D64` and `D74` were right to add the LSTM and the two naive
comparators. Doing so cost Table 6 every adjusted rejection (`D79`), Figure 4 its entire dynamic
range (`D84`), and left the one model whose standing changed out of the economics (`D77`) — none of
which failed a test, because each produced an artifact that still rendered.

### D86–D88 — defects found by *reading the notebook as the deliverable it became*

The fourteenth pass (2026-09-04) has the lens §1 implies and no earlier pass applied: **the notebook
is what an examiner opens.** `D63` and `D73` already moved in that direction, cutting module dumps
into sectioned cells and the outline into phases. What they left untouched was everything about the
notebook that a *reader* needs and a *launcher* does not — evidence, and a way to find anything in
354 cells.

| ID | Sev | Defect | Resolution | § |
|---|---|---|---|---|
| D86 | **F** | `D61` forbade committing the Kaggle export over the generated notebook, correctly, and nobody stated the price: the committed notebook carried **zero outputs** for the life of the project, so the artefact §15 calls *evidence* had none — while the only executed copy was the one file that could not be committed. §15's own "stale evidence is worse than none" was read as licensing *no* evidence | `build_notebook.py --preserve-outputs`: outputs carried forward only onto **byte-identical** cells, dropped everywhere else, definition cells always cleared. 12 cells and 169 kB of the 1,620-run session's evidence now survive regeneration, and `--check` stays byte-stable | 15 |
| D87 | **F** | 144 definition cells carried `metadata.itbtc`; the **seventeen cells that produce every artefact carried none**, and seven of them had no heading either. *Which cell writes the figures* and *which writes the metric parquets* were unanswerable except by reading 354 cells in order — in the artefact a reader opens to answer exactly that | Every step cell gets a stable slug and a `reads`/`writes` manifest, rendered as a **MENULIS/MEMBACA** strip under its banner; an index cell near the top prints the map, built from the cells actually emitted rather than from `PHASES`. `tests/test_artifact_map.py` asserts each declared path against the cell body | 15 |
| D88 | I | The notebook became the thing people edit while `src/` stayed the only place a change could be *made*, so editing a cell meant retyping it into a module — two copies kept in step by hand, which is `D54a` and `D69` in a new costume | `tools/notebook_to_src.py`: cells regrouped by `itbtc`, removed lines restored at their aligned anchors, then **re-flattened and refused unless byte-identical**. Activation lives in a fully commented last cell, inert on Kaggle where neither `tools/` nor `src/` exists | 15, 16 |

**`D87` is filed **F**, not **I**, and the reason is §1.** A launcher nobody opens can be unnavigable
at no cost. A manuscript's companion artefact cannot: an examiner who cannot find where a figure was
produced has no way to check that it was produced the way the methodology says, and §12's whole
contract is that every number resolves to something regenerable. Unfindable is unverifiable.

**New contradictions found later take IDs D89+. Absorbing one silently is the exact failure this
register exists to prevent.**

---

## 15. Repository layout

**The notebook is the primary surface; `src/` is its tested projection (`D88`, 2026-09-04).** This
inverts the emphasis of everything below it, and the emphasis is the only thing it inverts. §1's
deliverable is a manuscript, and `notebooks/iTransformer.ipynb` is what an examiner opens beside it,
what runs on Kaggle, and now what you may edit directly. `src/itransformer_btc/` remains the
importable package the test suite loads and the thing `code_sha256` hashes — half of §12's contract —
so it is not a shadow and must not be described as one.

What changed is that the two are now kept in step **in both directions**, by the same byte-exact
comparison, and neither is retyped by hand:

| Direction | Command | What it guarantees |
|---|---|---|
| `src/` → notebook | `python tools/build_notebook.py` | Cells are byte-exact slices of the flattened modules; `--check` fails the suite on drift |
| notebook → `src/` | `python tools/notebook_to_src.py` | Re-flattens what it wrote and refuses unless it reproduces the cells byte for byte |

**Three rules follow, and none of them is optional.**

- **Edit either side; commit both.** They are one change. A commit carrying only one of them is the
  two-copies-nobody-checks failure `D54a` and `D69` are both instances of.
- **An import can only be added in `src/`.** Module-level imports are stripped by the flattening and
  re-emitted once in the Library cell, which is generated *from* the modules (`D66`), so a cell has
  no import line to edit. `notebook_to_src.py` refuses with a message naming `src/` rather than
  guessing where the text belongs.
- **The sync cell at the bottom of the notebook stays commented out.** On Kaggle there is no `tools/`
  and no `src/`; an active cell there fails in the last cell of a twelve-hour session or writes
  rubbish into the session's working directory. Activation is a deliberate act in a local checkout.

**Committed outputs are evidence, and are now kept (`D86`).** `D61` forbids overwriting the generated
notebook with the Kaggle export, and it was right — the export carries papermill metadata and turns
the suite red. Its cost went unstated for a month: the committed notebook carried **zero outputs**,
so the artefact this section calls evidence had none, while the only executed copy of it was the one
thing nobody was allowed to commit. `build_notebook.py` now carries outputs forward onto
**byte-identical** cells and drops them everywhere else, so an output that survives regeneration
provably describes the code above it. Definition cells are exempt and always cleared: they define and
return nothing, and the one thing they can emit is their own docstring echoed back.

**Every producing cell declares what it reads and writes (`D87`).** The 144 definition cells carried
`metadata.itbtc`; the seventeen cells that actually produce something carried nothing, so *which cell
makes the figures* and *which one writes the metric parquets* could not be answered except by reading
354 cells in order. Each step cell now carries a stable slug and a `reads`/`writes` manifest, its
banner renders that manifest, and an index cell near the top prints the map, built from the cells
actually emitted. `tests/test_artifact_map.py` asserts every declared path against the cell body, so
a manifest cannot outrun its code.

```
invertedTransformer/
├── CLAUDE.md                       # this file — project law
├── README.md
├── USAGE.md                        # operational companion: commands, stages, schemas, expected numbers
├── docs/DIVERGENCE_REGISTER.md     # long-form evidence for D01–D62; §14 is the index
├── docs/ORIGIN_WINDOW_BUDGET.md    # per-origin/per-block window accounting — D45's assertion target
├── src/                            # importable package; module inventory in USAGE.md §2
├── tools/build_notebook.py         # src/ -> notebook; carries outputs forward (D54, D86, D87)
├── tools/notebook_to_src.py        # notebook -> src/; the return leg, verified byte-exact (D88)
├── tools/build_report.py           # generates paper/ FROM the artifacts on disk (D62a). CPU only
├── notebooks/iTransformer.ipynb    # THE deliverable — self-contained, generated, editable both ways
├── notebooks/logs-iTransformer.txt # the Kaggle session console stream — D60's evidence base
├── notebooks/outputs/artifacts/    # THE grid output (below)
├── paper/                          # manuscript + GENERATED deliverables (below)
├── spot_klines_btc.py              # Stage 1 ingest (was mis-named `binance_spot_klines.py`, D11/D33)
├── data/raw/                       # IMMUTABLE. the four Stage 1 artifacts live HERE (D33)
├── data/processed/                 # features_1h.parquet, splits.json — the writable half
└── artifacts/                      # repo-root: ONE stale 2026-08-06 CPU smoke run. NOT the results (D60f)
```

```
notebooks/outputs/artifacts/        paper/
├── preds/{run_id}.parquet          ├── CLAUDE.md              # manuscript writing posture
├── meta/{run_id}.json              ├── paper_numbers.json     # THE manuscript source (§12)
├── paper_numbers.json              ├── tables/table{1,2,2b,3,4,5,6,7,8,9}.tex
├── run_block_metrics.parquet       ├── figures/figure{1,2,2b,3,4,5,6,7}.{pdf,png}
├── seed_averaged_cells.parquet     └── panels/*.parquet
├── amplification_panel.parquet
├── decay_panel.parquet             # 0 rows — empty by the D55 guard (D60b)
├── keff_table.parquet
└── naive_rw_by_origin.parquet
```

`logs/`, as a directory, does not exist in either location.

**Logic lives in the package, and the notebook is the artefact a reader examines (`D63`).** If a
cell contains a feature definition, a window builder, a loss or a metric, it belongs in `src/` where
it can be unit-tested on CPU — that has not changed, and it is what makes the notebook generable and
testable at all. What changed is the notebook's job: §1's deliverable is a manuscript, and the
notebook is examined alongside it, so its **shape answers to a reader**, not to the generator's
convenience. Eighteen module-sized dumps were the shape of a launcher nobody opens. **Never leave a
notebook whose outputs are stale relative to `src/`**: outputs are evidence, and stale evidence is
worse than none.

**The launcher is self-contained, and that is not a weakening of the rule (`D54`).**
`notebooks/iTransformer.ipynb` carries the whole package, so a Kaggle session needs the notebook and
`BTCUSDT_1h.parquet` and **nothing else** — no repository Dataset to upload, keep in step by hand, and
silently run stale. No definition moved: the cells *transcribe* `src/` and the generator writes them.

**The outline is the pipeline, not the package (`D73`).** Twenty-four **phases**, each opening a
`##` banner and named for what that step of the study *does* — Persiapan, Pustaka, Konfigurasi,
Muat data, Pra-proses, Efisiensi, Split, K_eff, Algoritma, Loop latih, Metrik, Baseline,
Perbandingan, Ekonomi, Attention, Runner, Pelaporan, Provenance, Invarian, Gerbang, Grid,
Evaluasi, Simpan, Tabel. A phase carries its modules first and its orchestration steps second, so
each stage of the study keeps its definitions and its execution together instead of eighteen modules
stacking up in one block with every Stage cell trailing behind them. `PHASES` in the generator is
that outline; `build()` refuses to run unless the concatenation of `phase.modules` equals
`MODULE_ORDER`, because a phase table that drops or repeats a module produces a `NameError` hours
into a Kaggle session rather than a failure here. **Headings and prose are Indonesian; identifiers,
code and error strings are not.**

**It carries the package as definition cells, not as files (`D58`), segmented by logical group
(`D63`).** Eighteen modules, each opening a `###` banner inside its phase and cut into `####`
subsections of functions that work together — 137 definition cells of 20–120 lines plus 21
orchestration cells, every one preceded by an HTML markdown heading naming what it does and which
rule of this document it enforces. `SECTION_MAP` in the generator is that cut; `main()` refuses to
run when a module is missing from it, for the same reason it refuses on a missing `MODULE_ORDER`
entry. Cells define plain `def`, `class` and constant bodies in the kernel namespace; nothing is on
`sys.path` and nothing is imported. Five consequences, each load-bearing:

- **The flattening is subtractive, over exactly three declared categories.** *Intra-package imports*
  are removed, by `ast` node span rather than line matching so parenthesised and function-local ones
  come out right. *`runner.py`'s `if __name__ == "__main__":` guard* is removed, because in a cell
  `__name__` *is* `"__main__"` and the guard would launch the entire grid. And *module-level imports*
  are removed (`D66`), because a single **Library** cell at the top of the notebook emits every import
  the package makes — generated from the modules themselves, so a dependency that appears in `src/`
  cannot go missing there. Function-local imports **stay**: `report._pyplot` defers matplotlib on
  purpose, so that the package imports cleanly without a plotting backend, and hoisting that into a
  cell that runs at session start would undo the decision. Everything else is verbatim.
- **Cells are contiguous, exhaustive line ranges, and their identity lives in `cell.metadata`.** A
  banner comment would be a line the cell carries that its module does not, and the check below would
  have to strip it; the metadata tag costs nothing and keeps every cell body a byte-exact slice.
  `tests/test_notebook_sync.py` therefore asserts that **a module's cells, rejoined in order, equal
  `flatten_module_body(name)`** — not "equivalent", not "equal after formatting". Segmentation moved
  that guarantee from per-cell to per-module and the Library cell moved its *reference* from
  `flatten_module_source` to `flatten_module_body`; neither weakened it. A line lost between two cells
  fails there exactly as a changed line would.
- **Nothing is additive. A cell is a slice and only a slice (`D67`).** `from __future__ import
  annotations` appears **once**, in the Library cell, and in no definition cell — because IPython
  accumulates `__future__` compiler flags across a session, and Kaggle runs the notebook through
  papermill to ipykernel to `run_cell`, the same path. Measured on IPython 9.13: `compile.flags` goes
  `16896 → 16794112` after that cell, and a following cell carrying no future import of its own
  defines a forward annotation without raising. Plain `compile()` does *not* inherit, so
  `tests/test_notebook_sync.py` reads the flags off the Library cell and compiles every later cell
  under them — modelling the interpreter the notebook actually runs on rather than the one the test
  process happens to be. **This is the property the whole contract rests on**: if a future release of
  IPython stopped accumulating, every definition cell would compile its annotations eagerly and
  `RidgeConfig.build` — whose annotation names a `RidgeForecaster` defined one cell later — would
  raise at class-definition time. The test is the detector.
- **The declared interpreter is one field, and the suite follows it.** `language_info.version` in the
  notebook is what `tests/test_notebook_sync.py` parses every cell against, so raising the floor is a
  one-field change rather than a code change. It bounds *syntax*, not runtime behaviour — it would not
  have caught the future-import hazard above — so it is a floor, not a guarantee.
- **A module therefore reaches a sibling by name, never by module object** (`D59`).
  `from itransformer_btc.metrics import clark_west_test` binds a function another cell defines;
  `from itransformer_btc import metrics` binds a *module* no cell defines, leaving every `metrics.x`
  dangling until execution reaches it — six minutes into a Kaggle session, in the case that happened.
  The generator refuses to emit such a cell and the sync tests check every cell's names against the
  executed namespace with `symtable`.
- **Two module-level names collide** once namespaces merge — `DEFAULT_PARQUET` (`segments`, `train`)
  and `HOUR_MS` (`segments`, `metrics`). Both are the same value in both definitions, so
  last-cell-wins is harmless. Anything else colliding would not be, which is why the generator
  compiles what it emits and the sync tests re-derive the set rather than trusting a list.
- **The digest is pinned, not computed.** `code_sha256()` needs a `__file__` no definition cell has,
  so the generator pins `CODE_SHA256_OVERRIDE` from `src/itransformer_btc/` (§12). It is the **same
  number** a local checkout of the same source reports — a run from the notebook and a run from the
  repository must not look like different code vintages.

**Both generated artifacts are never hand-edited.** `python tools/build_notebook.py` writes the
notebook; `python tools/build_report.py` writes `paper/`. `--check` on either fails the suite the
moment `src/` or the artifacts move without it. Editing a `.tex` or a definition cell by hand is a
defect and the next generator run reverts it. The report check compares structure exactly and floats
within 1e-6, because polars aggregates `group_by` in parallel and exact equality would report drift on
thread scheduling (`D62h`).

**The evaluation cells are code that exists only in the notebook, and they are tested as such.**
`tests/test_notebook_cells.py` executes the bytes the notebook actually contains against synthetic
panels. Testing the generator's constants instead would pass while a stale notebook still crashed —
precisely how `D55` reached Kaggle and cost a twelve-hour session.

**Two rules for `src/` not derivable from the sections above.** Shuffle by permuting an index tensor
*on device*, never by moving data — the point of §10.3's GPU-resident regime is that data does not
move after the initial load. And keep the code device-agnostic: never hard-code `.cuda()`, and gate
precision on `torch.cuda.get_device_capability(0)[0] >= 8`, never on
`torch.cuda.is_bf16_supported()`.

**One `CLAUDE.md` per directory is not a goal, and three of four were deleted for cause.**
`src/CLAUDE.md` and `notebooks/CLAUDE.md` restated this file at 55–65% overlap, and their
non-overlapping content was precisely what **must not fail open**: a subdirectory `CLAUDE.md` loads
only when a file in that subtree is touched, so a prohibition living there is absent exactly when an
agent reasons about the area without opening a file. Rules whose violation is catastrophic — the
polars boundary, PyTorch-only, no `DataLoader`, Save & Run All — therefore live **here**. Test for
adding a new one: *is this rule local to that directory **and** harmless if it fails to load?*
`paper/CLAUDE.md` is the only survivor, because manuscript posture costs a weaker paragraph rather
than a corrupted result. The module inventory `src/CLAUDE.md` carried lives in `USAGE.md` §2.

---

## 16. Working conventions

**Style.** Python ≥ 3.11 syntax, type hints on every public function, Google-style docstrings. Config
in YAML loaded into dataclasses — **no magic numbers buried in code**. Comments explain *why*.

**The polars boundary.** polars is the data plane: segmentation, features, windowing, splits, all via
lazy scans. **pandas is permitted at exactly two places, both named** — (1) converting to numpy or
pandas for `statsmodels`, `arch` or `wildboottest`, which accept nothing else, via a named function,
not scattered `.to_pandas()` calls; and (2) **Stage 1 ingest**, `spot_klines_btc.py` (§2). Nowhere
else. Training touches no DataFrame at all: pre-built GPU-resident tensors, index-slice batching, no
`DataLoader`. This is a correctness argument, not only a speed one — polars' rolling API is
backward-closed, so the `center=True` leak is **unrepresentable**; in pandas it is one keyword away.
The source specification's §6.2 purge snippet is pandas and must be **re-expressed**, not copied.

**Reproducibility.** Seed `random`, `numpy`, `torch`, `torch.cuda`; set `PYTHONHASHSEED`;
`cudnn.deterministic = True` for final runs. Record git sha, **`code_sha256`** and input-artifact
sha256 in every `meta/*.json` — the middle one because the first is `"unknown"` off-repo, which is
every Kaggle session (§12).

**The notebook and `src/` are one change, whichever side you edited (`D88`).** Editing `src/` means
running `python tools/build_notebook.py` before committing; editing a notebook cell means **saving
the notebook first**, then running `python tools/notebook_to_src.py`, which reads the file on disk
and not the live kernel. Either way both files go in the same commit. Skipping the step fails the
suite rather than shipping a notebook that runs last week's code. Adding a module means adding it to
`MODULE_ORDER` **and** giving it a `SECTION_MAP` entry — the generator refuses to run otherwise,
because a silently omitted module leaves a name undefined deep in a twelve-hour session. Adding a
*step* means giving it a slug and a `reads`/`writes` manifest (`D87`); a producing cell with neither
is invisible in the artefact map and fails `tests/test_artifact_map.py`.

**Never commit the Kaggle export over the generated notebook** (`D61`): it carries papermill
metadata and execution counts, and turns the suite red. **Its outputs are worth keeping, and there is
now a way that does not break the rule** (`D86`): save the export somewhere under
`notebooks/outputs/`, then
`python tools/build_notebook.py --preserve-outputs <that file>`. Outputs land only on byte-identical
cells; everything else is dropped and counted in the build log. `--no-preserve-outputs` builds clean.

**Before any long run.** Overfit a single batch: if the model cannot drive loss to ~0 on 8 samples in
200 steps, the plumbing is broken. **Run it with `dropout=0.0`** (`D52d`) — with the configured 0.1
active the loss floors around 7e-2 and a reader following the instruction literally concludes the
plumbing is broken when it is not. Measured: **8.26e-10** with dropout off, 6.8e-2 with it on. Compute
and log the **Naive-RW baseline first**, before any model trains.

**Environment.** `pyproject.toml` declares `requires-python >= 3.11`. Core dependencies are the data
plane only — **polars, pyarrow, numpy**; everything else sits behind a named extra so each
dependency's reason is visible rather than ambient:

| Extra | Contents | Why it is separate |
|---|---|---|
| `ingest` | requests, pandas | Stage 1 only. pandas in the core list would make the §2 exemption ambient |
| `stats` | pandas, scipy, statsmodels, arch, wildboottest | The one named boundary where data leaves polars |
| `train` | torch | Unpinned and **not installed on Kaggle** — the image ships its own |
| `dev` | pytest | |

Kaggle ships its own image regardless: the notebook runs against whatever torch and polars are already
there and pip-installs only what is genuinely missing.

---

## 17. Tombstone

Before 2026-08-05 this repository pursued a different project: a production-grade 1-minute BTC/USDT
forecasting model fusing four sources at different sampling frequencies (BTC 1 min, XAU/USD 1 min, FED
Broad Dollar Index daily, 31 US macro indicators monthly), delivered as a TorchScript/ONNX export
bundle, built through a two-notebook Kaggle pipeline generated from a reference notebook and joined by
a frozen-feature-artifact contract. **That project is superseded in full.** Its rules — including an
anti-leakage table that permitted forward-fill, which §2 now forbids as fatal — are void. Rationale
and implementation live in git history at `ee55c9d` and earlier; the notebooks were removed at
`cadbdf7`+. Nothing from it is authoritative, and no rule of it may be cited.
