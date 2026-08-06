# Research Specification: Nominal Variates versus Effective Dimensionality in iTransformer

**Working title:** *Nominal Variates or Effective Dimensionality? A Walk-Forward Evaluation of iTransformer for Hourly Bitcoin Forecasting*

**Indonesian title (target venue is an Indonesian journal):** *Variat Nominal atau Dimensionalitas Efektif? Evaluasi Walk-Forward iTransformer pada Peramalan Bitcoin Per Jam*

> Lock the title **after** experiments conclude. If RQ1 returns null, switch to the descriptive alternative (see Section 11).

### Revision note for this version

The study is now **spot-only, single-asset, feature-based** (formerly "option A"). Every futures-derived variate is removed, the variate ladder is rebuilt around microstructure columns present in the Binance spot kline, and RQ1 is reframed from *does K help* to *does K or K_eff govern the benefit*.

Materially changed: Sections 1, 2, 6.1, 6.2, 6.4, 7, 8, 10, 11, 12, 13. New: Section 2.4.
Unchanged and still correct: Sections 3, 4 (including the full `use_norm` derivation in 4.4), 5, 6.3, 9.

Two alternatives were considered and rejected. **Multi-asset spot** (12 crypto log-returns, structurally closer to the Electricity/Traffic benchmarks) was rejected by user decision. **Futures-based features** were rejected by the spot-only constraint.

---

## 1. Research questions and hypotheses

| Code | Question | Hypothesis | Dependent variable |
|---|---|---|---|
| RQ1 | Is the marginal benefit of added variates governed by the nominal count K or by effective dimensionality K_eff? | H1: Benefit tracks K_eff. Real gains at K=1→4→8, flat at 8→12. | ΔMSE per rung, regressed separately on K and on K_eff |
| RQ2 | Does the accuracy gap between multivariate and univariate models narrow as time-since-training increases? | H2: It narrows. The microstructure-to-return mapping is regime-specific. | `A(b) = [MSE_K1(b) − MSE_K8(b)] / MSE_K1(b)`; the claim is β₁ < 0 in `A(b) ~ b` |
| RQ3 | What is the optimal retraining cadence, and does it depend on K? | H3: Larger K decays faster. | `b* = min{b : D(b) > τ}`, where `D(b) = [MSE(b) − MSE(1)] / MSE(1)` |

**RQ2 compares K=1 against K=8, not K=12.** K=8 is the rung of maximum effective rank. Using K=12 would confound the decay effect with the redundancy deliberately built into that rung (Section 2.2).

**RQ3 requires a pre-registered threshold.** Headline τ = 5%, with sensitivity reported at τ ∈ {2.5%, 5%, 10%}. Without a fixed rule, "optimal cadence" is describable but not answerable, and choosing the threshold after seeing the decay curve is indistinguishable from p-hacking.

### Mechanism behind H2

The earlier multi-asset design grounded H2 in cross-asset correlation instability. That mechanism is unavailable here and is replaced by a stronger one: **shifting composition of market participants**. Retail-dominated flow in 2018–2020, the leverage cycle of 2021, institutional flow after spot ETF approval in 2024. The predictive content of order-flow imbalance should decay as market making tightens.

This maps directly onto Khuntia & Pattanayak (2018) on the Adaptive Market Hypothesis, already present in the reference library: if predictability itself evolves, a static model must decay. Paired with Han et al. (2024) on the capacity–robustness trade-off, H2 rests on one finance-theoretic and one ML-theoretic foundation independently.

**Claimed contributions:**
1. First walk-forward evaluation of iTransformer on a crypto asset with explicit decay measurement.
2. Separation of nominal variate count from effective dimensionality as competing explanations for cross-variate gains — a distinction the iTransformer paper does not draw, and which its own benchmark heterogeneity (7 features from one transformer versus 321 clients) invites.
3. Evidence-based retraining cadence under a pre-registered degradation threshold.

---

## 2. Data

### 2.1 Sources

| Item | Specification |
|---|---|
| Asset | BTCUSDT, **spot only** |
| Exchange | Binance |
| Granularity | 1 hour |
| Period | 2018-01-01 to 2026-08-01, end exclusive (~75,216 bars) |
| Download source | Binance REST `/api/v3/klines` via `binance_spot_klines.py` |
| Timezone | UTC throughout |

**No futures data anywhere in the paper.** `funding_rate`, `oi_change`, `ls_account_ratio`, and `basis` are removed. This costs four variates and buys three things: spot history reaches back to 2017-08 rather than 2019-09, so the window spans the 2018 bear market and the complete COVID crash; provenance needs one justification rather than three; and the multi-symbol availability probe that previously gated the design disappears entirely, since BTCUSDT spot predates 2018-01 by five months.

**The REST API is now sufficient, reversing the earlier decision.** Bulk archive was required only for `openInterestHist`, `topLongShortAccountRatio`, and `takerlongshortRatio`, which serve just 30 days over the API. With those variates gone, roughly 76 paginated requests cover eight years of hourly klines in under two minutes, at ~380 weight against a 6,000/minute ceiling.

Single endpoint:
```
GET /api/v3/klines?symbol=BTCUSDT&interval=1h&startTime=...&endTime=...&limit=1000
```

**All eleven meaningful kline columns must be retained.** A fetcher that truncates to OHLCV silently destroys variate families F3, F4, and F5, collapsing the ladder from 12 to 6. Three columns carry information that cannot be derived from OHLC at all:

| Column | Feeds |
|---|---|
| `quote_asset_volume` | F5 intrabar price location, via VWAP = quote_volume / volume |
| `number_of_trades` | F3 intensity, via mean order size = quote_volume / trades |
| `taker_buy_base_volume` | F4 signed order flow — direction, not magnitude |

The twelfth field (`ignore`) is always zero and is dropped. Numeric fields arrive as **strings**; without explicit coercion every downstream computation is silently wrong rather than raising an error.

### 2.2 Variate families and the K ladder

Variates are grouped into five families by information source. The point of the grouping is that **nominal count and effective rank diverge**, and that divergence is the object of RQ1.

| Family | Variates | Independent dof |
|---|---|---|
| F1 Price trajectory | `r` = log(C/C₋₁), `upper_shadow` = log(H / max(O,C)), `lower_shadow` = log(min(O,C) / L) | 3 |
| F2 Volatility estimators | Parkinson, Garman–Klass, Rogers–Satchell | ~1, redundant by construction |
| F3 Intensity | `log_quote_volume`, `log_trade_count`, `log_mean_trade_size` | 2, the third is the difference of the first two |
| F4 Order flow | `taker_buy_ratio`, `signed_flow` = (2·ratio − 1)·log_volume | 1–2 |
| F5 Intrabar price location | `(VWAP − C) / (H − L)`, where VWAP = quote_volume / volume | 1 |

> **Do not add `log(C/O)` as a separate variate.** Crypto bars are contiguous, so `log(C/O) ≈ r` at correlation ≈ 0.99. This appeared in the earlier K=5 layer as `oc_return` and is now removed. It is silent duplication that inflates K without inflating K_eff, corrupting the very axis RQ1 measures.

**The ladder:**

| K | Adds | Expected K_eff |
|---|---|---|
| 1 | `r` | 1 |
| 4 | + remainder of F1, `log_quote_volume` | ~3.5 |
| 8 | + remainder of F3, F4, F5 | ~6.5 |
| 12 | + F2 (three volatility estimators), `log_mean_trade_size` | ~7 |

**The 8→12 rung is deliberately redundant and functions as a control, not an accident.** It adds four nominal variates carrying almost no additional rank. If accuracy rises from 4→8 and then flattens from 8→12, that is not a failed experiment — it is the demonstration that nominal K is the wrong axis. **State this explicitly in the methodology**, alongside the K=1 degeneracy note in Section 4.3. Left unstated, a reviewer reads the flat rung as a null result rather than as the designed contrast.

> **Mandatory:** every rolling indicator must use a backward-looking window. Use `pandas.rolling()` with `center=False`. Setting `center=True` is a fatal leak.

> **Reject the "near-gap" indicator variate.** A tempting fix for the data gaps in Section 2.3 is a binary flag marking bars adjacent to an outage, so gap-spanning windows become usable. Do not do it. It adds a variate belonging to none of F1–F5, breaking the ladder and rendering the K=8 versus K=12 contrast uninterpretable. Losing a few percent of windows is far cheaper than contaminating the study's primary independent variable.

### 2.3 Data handling: gaps are not missing values

Missing hourly bars in BTCUSDT spot have exactly one cause. Binance emits no kline when no trade occurs in an interval, but BTCUSDT has traded every second for eight years, so zero-trade intervals are ruled out. What remains is **exchange downtime and scheduled maintenance**, confirmed empirically against the fetcher's gap report.

**This reclassifies the problem.** Rubin's taxonomy (MCAR / MAR / MNAR) applies to values that exist but went unobserved. When the exchange is down no price forms: no order matching, no book, nothing to approximate. Imputation here is not merely risky, it is **undefined** — there is no ground truth, therefore no metric can justify any imputation choice. This also retires the MNAR diagnostic that would otherwise be mandatory, since there is no missingness mechanism left to model.

**Policy: segmentation, never imputation.**

| Issue | Policy |
|---|---|
| Missing bars (exchange downtime) | Gaps split the series into contiguous **segments**. Build windows inside segments, never across them. No reindexing to a full grid, no filling of any kind. |
| Extreme outliers | **Do not remove.** Extreme regimes are the object of study, not noise. |
| Winsorization | None. See Section 4.4. |

**Why forward-fill is actively wrong here**, notwithstanding the leakage-checklist item that permits it. That item correctly prevents lookahead; it does not make ffill safe. With bar `t` missing and `P_t := P_{t−1}`:

```
r_t     = log(P_t / P_{t−1})   = 0                 <- fabricated zero
r_{t+1} = log(P_{t+1} / P_t)   = the full 2-hour return compressed into one bar
```

This manufactures a **zero-then-jump** pattern that never occurred in the market, and a model searching for signal will learn it — an artefact of cleaning, not of microstructure. Worse for this design specifically: ffill forces `H = L = O = C`, so Parkinson and Garman–Klass return **zero volatility** immediately before an extreme bar, poisoning all of F2. Meanwhile `volume = 0` gives `log(0) = −inf` and `taker_buy_ratio = 0/0` gives NaN; forward-filling volume instead fabricates trading activity that did not happen. Linear interpolation on log-price removes the zero-then-jump but produces artificially smooth returns, understating realized volatility at exactly the outage boundaries — opposite bias, equal contamination.

**Validate windows by timestamp, not by position.** This is the highest-probability silent bug in the whole pipeline:

```python
valid = (t[s + L + H - 1] - t[s]) == pd.Timedelta(hours=L + H - 1)
```

If windows are slid over a positional index after `dropna()`, gaps close silently and windows cross time discontinuities without raising anything.

**Order of operations matters.** Compute log-returns **per segment** — the first bar of each segment yields NaN and is dropped — and only then fit the scaler. Computing returns on a concatenated series injects giant cross-gap returns into μ_g and σ_g before any window is excluded. Per Section 4.4 this does not reach iTransformer's input while `use_norm=True`, but it does contaminate the reported metric scale and the baselines that lack internal normalization.

**Cost accounting.** A gap of `g` bars invalidates `L + H − 1 + g` window start positions. With L=96 and H=24 that is `119 + g`, so cost is dominated by the constant rather than by gap length:

```
windows lost ≈ 119 × (number of gap blocks) + (total missing bars)
```

Forty one-hour gaps cost about 4,800 windows; one forty-hour gap costs about 159 — a factor of thirty. The number to read from the fetcher's report is therefore **`gap_blocks`**, not total missing bars. Scheduled maintenance produces few long gaps, which is the cheapest profile. Below roughly 100 gap blocks, losses stay under 16% of ~75,000 windows and no further discussion is warranted.

Report gap count, durations, attribution to exchange downtime, the exclusion rule, and the percentage of windows lost. One paragraph is enough, and it corrects a defect most local-journal crypto forecasting papers do not acknowledge exists.

### 2.4 Pre-model measurements

Run these **before training anything**. They produce the K_eff values RQ1 regresses on, and they test H2's premise independently of any model.

| Measurement | Definition | Purpose |
|---|---|---|
| Participation ratio | `PR = (Σλᵢ)² / Σλᵢ²` on the correlation matrix of each K rung | Supplies K_eff for RQ1. Bounded between 1 and K. |
| Rolling PR | Same statistic over a 90-day rolling window across 2018–2026 | Tests whether dependency structure is stable. The swing matters more than the level. |
| Rolling OLS R² | `r_{t+1} ~ (K=8 features)`, 90-day window | If R²(t) is unstable, H2's premise is established before a single epoch runs. |

Both belong in the paper as a descriptive subsection. They convert RQ2's mechanism from an assertion into a measured premise at zero additional data cost, mirroring what Section 7.3's efficiency tests do for the random-walk assumption.

**Honest caveat on the K_eff figures in Section 2.2.** The values ~3.5 / ~6.5 / ~7 are reasoned from family structure, not measured. They may be wrong. If measured PR at K=8 lands near 4 rather than 6.5, H1's predicted saturation point moves and the ladder must be re-cut before the main grid runs. Fix the hypothesis to the measurement, never the measurement to the hypothesis.

---

## 3. Target and transformation

**Target:** `log_return`, single-step and multi-step — **not** raw price.

```python
log_return = np.log(close).diff()
```

**Rationale:** forecasting price levels produces artificially low MSE because the price series is close to a random walk. A persistence baseline will match or beat any model.

**Price reconstruction** (for supplementary reporting only, not the primary metric):
```python
price_hat = close[t] * np.exp(np.cumsum(return_hat))
```

**Compute returns per segment before fitting the scaler.** See Section 2.3. Differencing across a gap boundary produces a spurious multi-hour return that contaminates μ_g and σ_g.

**Normalization:** fit `StandardScaler` **only on the training window at each origin**. Not on the full series, not on validation, not on test blocks.

Note that iTransformer applies instance normalization internally (`use_norm=True`), which algebraically cancels any per-channel affine scaler. This does **not** exempt the preprocessing step: the outer scaler still fixes the reporting scale of all metrics and still governs learning for baselines that lack internal normalization. See Section 4.4 for the derivation and full justification.

---

## 4. Model specification

### 4.1 iTransformer architecture

Encoder-only. Each variate is treated as a token; attention operates across variates rather than across time steps.

```
Input  : (B, L, N)
Transpose → (B, N, L)
InvertedEmbedding : Linear(L → d_model)     → (B, N, d_model)
Encoder × e_layers:
    MultiHeadAttention over N tokens
    LayerNorm
    FeedForward: Linear(d_model → d_ff) → GELU → Linear(d_ff → d_model)
    LayerNorm
Projection : Linear(d_model → H)            → (B, N, H)
Transpose  → (B, H, N)
Select target channel → (B, H, 1)
```

### 4.2 Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| `seq_len` (L) | 96 | 4 days |
| `pred_len` (H) | 24 | headline; sweep {1, 3, 24, 168} |
| `d_model` | 128 | reduced from 512 because N is at most 12 |
| `d_ff` | 256 | |
| `e_layers` | 2 | |
| `n_heads` | 8 | |
| `dropout` | 0.1 | |
| `activation` | GELU | |
| Optimizer | Adam | |
| Learning rate | 1e-4 | type-1 schedule (halved each epoch) |
| Batch size | 32 | |
| Max epochs | 30 | |
| Early stopping | patience 5 on validation MSE | |
| Loss | MSE | |
| Seeds | 42, 43, 44 | |
| `use_norm` | **True** | instance normalization — see Section 4.4, this flag has architectural consequences |

> **Justification for `d_model=128`:** iTransformer's attention sequence length equals N (the variate count), not L. With N=12, `d_model=512` over-parameterizes relative to ~17,000 training samples. State this justification in the paper.

### 4.3 Critical technical note: degeneracy at K=1

At N=1, self-attention over a single token yields a softmax weight of 1, making the attention operation an identity. iTransformer with K=1 effectively reduces to:

```
Linear(L → d_model) → LayerNorm → FeedForward → Linear(d_model → H)
```

This is **not a design flaw — it is precisely the right control.** Comparing K=1 against K=12 isolates exactly the contribution of cross-variate attention, because every other component is identical.

State this explicitly in the methodology. Left unexplained, an examiner may read it as an implementation error.

### 4.4 `use_norm` and the scaler choice

**`use_norm=True` is mandatory and is not a tuning knob.** It controls instance normalization: each input window is normalized by its own statistics along the time dimension, then denormalized at the output. This has a consequence that determines how the preprocessing scaler must be justified.

#### The outer scaler cancels algebraically

Let the outer scaler produce `z = (x − μ_g)/σ_g` using global statistics fitted on the training window. For any window W, the instance statistics become:

```
m = mean_t(z_W) = (mean_t(x_W) − μ_g) / σ_g
s = std_t(z_W)  = std_t(x_W) / σ_g
```

Instance normalization then yields:

```
(z − m) / s = [ (x − μ_g)/σ_g − (mean_t(x_W) − μ_g)/σ_g ] / [ std_t(x_W)/σ_g ]
            = (x − mean_t(x_W)) / std_t(x_W)
```

`μ_g` and `σ_g` vanish entirely.

This holds for **any per-channel affine scaler** — StandardScaler, RobustScaler, MinMaxScaler alike. While `use_norm=True`, all of them deliver bit-identical input to the network. The scaler choice therefore does not affect what iTransformer learns.

#### What the scaler does control

**Metric scale.** Model output is denormalized back into scaler space, and targets live in scaler space, so MSE is expressed in units of `σ_g²`. Switching to RobustScaler rescales every reported MSE by `(σ_g / IQR_g)²` — a factor of 2–4× on fat-tailed data.

**Baseline models.** LSTM and ARIMA have no instance normalization. For them the scaler genuinely affects learning. Cross-model consistency matters more than the specific choice: training iTransformer and LSTM in different scale spaces would invalidate the comparison.

#### Justification for StandardScaler

Three reasons, all of which belong in the paper:

1. **Literature comparability.** The entire LTSF literature reports MSE on z-score normalized data. Any other scaler makes your numbers incomparable to published iTransformer, PatchTST, and DLinear results.
2. **Inertness under `use_norm=True`.** For iTransformer the affine scaler choice is eliminated by instance normalization; it purely fixes the reporting scale.
3. **Consistency** across all models, including baselines that lack internal normalization.

#### Why the fat-tail argument does not favour RobustScaler

The usual intuition — crypto log-returns have extreme kurtosis, so σ is contaminated and RobustScaler is safer — runs the wrong way.

On fat-tailed distributions `σ > IQR/1.349`. RobustScaler divides by the **smaller** quantity, making outliers **larger** in scaled space. Under MSE loss, training becomes more dominated by a handful of crash bars, not less.

This is worse for this study specifically: RQ2 concerns behaviour across regimes. Inflating the effective weight of extreme events distorts precisely the quantity being measured.

The fat-tail concern is a legitimate argument for **Huber or MAE loss**, not for changing the scaler.

#### MinMaxScaler is excluded

Not a preference — a correctness issue. MinMaxScaler is bounded by training-set min and max. During test periods, returns exceed the training range whenever a new regime appears, producing out-of-`[0,1]` values the model never saw in training.

This is a recurring defect in the LSTM literature on crypto forecasting, where MinMaxScaler is applied to **price levels** so that every new all-time high in the test period falls outside the fitted range. Worth one sentence in Related Work as part of the methodological critique.

#### What actually stabilizes the data

The log-return transform, not the scaler. That is what converts a non-stationary series with exponential trend into one approximately stationary in mean.

The residual problem is heteroskedasticity — variance shifting across regimes. No global scaler addresses this, because all of them apply a single statistic across the entire period. Instance normalization inside iTransformer is what handles it. This connection between preprocessing choice and architecture belongs in the methodology.

#### No winsorization

Do not clip returns at ±5σ. That removes exactly the events driving regime-dependent decay. State this decision explicitly; a reviewer will otherwise read it as an oversight.

#### Verification test

Confirm `use_norm` is actually active before running the main grid. Train one model on StandardScaler output and one on the same input multiplied by 100. Under `use_norm=True` both losses must be identical to numerical precision.

If `use_norm` is disabled for any reason, the entire analysis above collapses: the outer scaler again affects learning, and the fat-tail argument becomes relevant. Document the flag state in the paper either way.

#### Robustness check to run

One origin, three seeds, RobustScaler versus StandardScaler. Half a day of work, and it closes the question with evidence rather than argument. Expected result: the `RelMSE` difference falls within seed variance, because RelMSE is a ratio and the scale factor cancels.

---

## 5. Baselines

| Baseline | Role | Configuration |
|---|---|---|
| **Naive-RW** | mandatory, EMH baseline | ŷ = 0 for log-return |
| Naive-persist | secondary comparator | ŷ = last observed return |
| Seasonal-naive | daily pattern | ŷ = return at t−24 |
| ARIMA | classical econometrics | order selected by AIC on the training window |
| LSTM | RNN | 2 layers, hidden 128, dropout 0.1 |
| **DLinear** | mandatory | trend-seasonal decomposition + linear |
| PatchTST | SOTA transformer, channel-independent | patch 16, stride 8 |

> **Naive-RW uses ŷ = 0, not the last return.** A random walk in price implies zero expected return. Using the last return produces a weaker baseline and makes your results look better than they are.

> **DLinear and PatchTST are not optional.** A missing DLinear is the first thing a reviewer familiar with the LTSF literature will flag.

---

## 6. Evaluation protocol

### 6.1 Scheme

**Rolling-origin walk-forward validation with purging.**

| Component | Value |
|---|---|
| Training window length | 24 months, **fixed** |
| Training sub-block | 21 months |
| Validation sub-block | final 3 months of the training window |
| Purge | H steps at the train–test boundary |
| Embargo | not applied (justified) |
| Test blocks | 6 × 30 days after the origin, **no retraining** |
| Origin spacing | 6 months |
| Origin count | **13** |

**Origin derivation.** Each origin consumes 24 months of training plus 180 days of testing, so 30 months total. With the window 2018-01-01 → 2026-08-01 and 6-month spacing, origins fall at 2020-01, 2020-07, … 2026-01 inclusive — thirteen in all. The final origin tests through 2026-06-30, inside the data boundary. Spot history starting 2017-08 is what makes thirteen possible; the futures-based design reached only ten.

**Why rolling rather than expanding:** with an expanding window the training set size changes at each origin, so the effect of model age cannot be separated from the effect of training data volume.

### 6.2 Purge implementation

```python
def build_windows(data, ts, L, H, train_end_idx, freq="1h"):
    """ts: DatetimeIndex aligned with data. Windows are validated by
    timestamp so they never span an exchange-downtime gap (Section 2.3)."""
    span = pd.Timedelta(freq) * (L + H - 1)
    X, y = [], []
    for t in range(0, train_end_idx - L - H + 1):
        if ts[t + L + H - 1] - ts[t] != span:   # window melintasi celah
            continue
        X.append(data[t : t+L])
        y.append(data[t+L : t+L+H])
    return np.array(X), np.array(y)
```

The `- H` term is the purge. The final retained training window has a target ending exactly at `T_end`, so **no observation is discarded** — only 24 window configurations out of roughly 17,400.

**The timestamp check is the segmentation rule from Section 2.3**, and it shares its logic with purging: neither discards observations, both discard window *configurations*. Extending an already-justified principle costs nothing methodologically, which is why segmentation needs no separate defence in the paper beyond one sentence.

Note the `continue` branch is the only thing standing between the pipeline and a silent cross-gap window. Log the rejection count per origin and assert it matches the `119 × gap_blocks` estimate; a mismatch means the timestamps and the data array have drifted out of alignment.

### 6.3 Rejecting CPCV

Include this paragraph in the methodology:

> Combinatorial Purged Cross-Validation (López de Prado, 2018) was considered but not adopted. CPCV generates backtest paths with non-chronological block ordering, under which *time-since-training* — the primary independent variable in RQ2 — is undefined. CPCV also assumes stability of the data-generating process across blocks, an assumption this study explicitly tests. Walk-forward was chosen because it preserves temporal ordering, remains consistent with evaluation protocols in the long-term time series forecasting literature, and still applies purging of H steps at every training boundary.

### 6.4 Experiment grid

**Main grid** — 13 origins × **4 K values** × 3 seeds = **156 runs**, each producing 6 measurement points (B1–B6).

**Horizon sweep** — 4 origins × 4 K × 4 H values × 3 seeds = **192 runs**.

**Baselines** — 13 origins × (4 deterministic + 3 stochastic × 3 seeds) = **169 runs**, most of them very cheap.

Total ≈ 517 runs, up from 430. The increase comes from the ladder growing from three rungs to four; K=12 exists to be redundant (Section 2.2) and its runs cannot be trimmed without removing the control. With ~17,000 training rows a single iTransformer run completes in minutes on free-tier GPU, so the added cost is hours, not days.

---

## 7. Metrics and statistical tests

### 7.1 Metrics

| Metric | Formula | Purpose |
|---|---|---|
| MSE | mean squared error on normalized log-returns | primary metric |
| MAE | mean absolute error | outlier-robust |
| RelMSE | MSE_model(b) / MSE_naive(b) | controls for period difficulty |
| DA | proportion of correct directional predictions | practical relevance |
| A(b) | [MSE_K1(b) − MSE_K8(b)] / MSE_K1(b) | **dependent variable for RQ2**. K=8, not K=12 — see Section 1 |
| D(b) | [MSE(b) − MSE(1)] / MSE(1) | degradation curve, **dependent variable for RQ3** |
| ΔMSE per rung | MSE(K_j) − MSE(K_{j+1}) | **dependent variable for RQ1**, regressed on K and on K_eff separately |

**Report A(b) as a log ratio in the regression.** A(b) is a ratio of MSEs and is right-skewed; `log(MSE_K1 / MSE_K8)` is better behaved under OLS and reduces to A(b) for small differences. Report both, and note the choice.

**Report two metric scales.** MSE in scaler space for literature comparability, and RMSE in raw log-return units for interpretability. "RMSE 0.0043 on hourly log-returns" tells a reader far more than "MSE 0.187 on normalized data." State the conversion factor `σ_g` so the two are reconcilable.

> **Do not use MAPE on log-returns.** Log-returns frequently approach zero, so the MAPE denominator explodes and the metric becomes practically undefined. MAPE may only be reported on reconstructed prices, and even there it is dominated by random-walk behaviour and thus uninformative. The Lewis (1982) thresholds commonly cited in Indonesian journals do not apply in this context.

### 7.2 Mandatory statistical tests

**Diebold–Mariano test** for every comparative claim.

At H > 1, forecast errors across origins overlap, so a HAC (Newey–West) variance estimator with lag h−1 is **required**. At H=24 that means lag 23. Skipping this correction produces over-optimistic p-values.

Apply the Harvey–Leybourne–Newbold small-sample correction:

```
S* = S · sqrt[ (T + 1 − 2h + h(h−1)/T) / T ]
```

where T is the out-of-sample observation count and h is the forecast horizon. Compare S* against a Student-t distribution with (T−1) degrees of freedom, not the standard normal. Validate any custom Python implementation against R's `forecast::dm.test()`.

**Decay regression** — the paper's core claim:
```
A(b) = β₀ + β₁·b + ε
```
A significant β₁ < 0 is the affirmative answer to the title question.

**The naive OLS specification is wrong and must be corrected.** The design gives 13 origins × 6 blocks = **78 observations**, but the six blocks within an origin are produced by the *same trained model*, so their residuals are correlated within origin. Three requirements follow:

1. **Cluster by origin.** Ordinary standard errors overstate precision badly here.
2. **Use wild cluster bootstrap.** With only 13 clusters — well under the usual rule of thumb of roughly 30–40 — conventional cluster-robust standard errors are biased downward. The wild cluster bootstrap is the standard remedy at this cluster count. In Python, `wildboottest`; in R, `fwildclusterboot`.
3. **Average over seeds first.** Seeds are computational noise, not draws from a population. Averaging the three seeds within each (origin, block) cell before regression is correct; treating them as 234 independent observations is not.

A random-intercept mixed model with origin as the grouping factor is an acceptable alternative and may read more naturally to a reviewer from a statistics background. Report whichever is chosen, and state the cluster count explicitly — a reviewer who knows this literature will look for it.

**Reporting** — all metrics as mean ± std across 3 seeds. Never a single number.

### 7.3 Preliminary market efficiency tests

Run once at the outset, report in the Data section:

| Test | Library | Interpretation |
|---|---|---|
| Variance Ratio (Lo–MacKinlay) | `arch.unitroot.VarianceRatio` | VR ≈ 1 → consistent with random walk |
| Hurst exponent | `hurst` or an R/S implementation | H ≈ 0.5 → no long memory |
| ADF | `statsmodels.tsa.stattools.adfuller` | log-returns are stationary |

This is what converts "efficient market" from an assumption into a finding.

---

## 8. Execution pipeline

```
Stage 1   Ingest
          └─ binance_spot_klines.py: REST /api/v3/klines, BTCUSDT 1h
          └─ retain all 11 columns; coerce string numerics explicitly
          └─ output: BTCUSDT_1h.parquet + _report.json + _gaps.csv

Stage 2   Data validation
          └─ read gap_blocks and per-year coverage from _report.json
          └─ attribute gaps to exchange downtime; NO reindexing, NO filling
          └─ derive segment boundaries; estimate windows lost = 119×blocks + bars
          └─ set the analysis window FROM the coverage report, not from assumption
          └─ data quality report → Table 1

Stage 3   Feature construction
          └─ log-returns computed PER SEGMENT, then concatenated
          └─ build F1-F5, all backward-looking
          └─ stationarity, VR, Hurst tests → Table 2
          └─ output: features_1h.parquet

Stage 3b  Pre-model measurement  (Section 2.4)
          └─ participation ratio per K rung → K_eff values for RQ1
          └─ rolling PR and rolling OLS R^2, 90-day window → Table 2b, Figure 2b
          └─ GATE: if K_eff at K=8 is far below ~6.5, re-cut the ladder now

Stage 4   Split generator
          └─ produce 13 origins (train 24mo, val 3mo, purge H, test 6×30 days)
          └─ window validity checked by TIMESTAMP, not position
          └─ log rejected-window count per origin; assert against the estimate
          └─ persist split indices as JSON rather than recomputing
          └─ verify no overlap

Stage 5   RQ1 pilot  ← DECISION GATE
          └─ verify use_norm scale-invariance (Section 4.4) before anything else
          └─ 1 origin × 4 K × 3 seeds
          └─ optional: RobustScaler vs StandardScaler robustness check
          └─ GATE IS K=1 vs K=8, NOT K=1 vs K=12.
             K=12 is built to be redundant; gating on it would kill a viable
             paper for the wrong reason. If K=8 does not significantly beat
             K=1, reposition the title to the descriptive variant now, not in
             week nine.

Stage 6   Main grid
          └─ for each origin:
             fit scaler on the 21-month training block only
             build windows with purge
             train, early stop on validation
             predict B1..B6 with no retraining
             persist raw predictions, not just metrics

Stage 7   Baselines
          └─ run all 7 baselines on exactly the same splits

Stage 8   Horizon sweep
          └─ 4 origins × 3 K × 4 H × 3 seeds

Stage 9   Aggregation and testing
          └─ compute MSE, MAE, RelMSE, DA, A(b)
          └─ DM test with HAC
          └─ regress A(b) ~ b

Stage 10  Attention extraction
          └─ extract cross-variate attention weights per block
          └─ heatmaps per regime → Figure 5

Stage 11  Economic evaluation
          └─ simple long/short rule from directional predictions
          └─ apply 0.04% taker fee + slippage
          └─ Sharpe, max drawdown, turnover
```

**Persist raw predictions, not just metrics.** They are required for the DM test, per-regime analysis, and economic evaluation. Re-running 430 experiments because predictions were not saved is an expensive mistake.

---

## 9. Paper structure

IMRaD format, target 10–14 pages.

### Abstract (200–250 words)
Must contain concrete numbers: the β₁ value, percentage decay, and the recommended retraining cadence. An abstract without numbers reads like a proposal, not a result.

### 1. Introduction
- Context of crypto asset forecasting and limitations of existing approaches
- Gap: LTSF evaluation on crypto typically uses a fixed chronological split, ignoring non-stationarity
- Contributions (3 items, Section 1 of this document)
- Paper structure

### 2. Related Work
- Transformer architectures for time series forecasting: Informer, Autoformer, PatchTST, DLinear, iTransformer
- The channel-independence versus channel-dependence debate
- Deep learning for cryptocurrency forecasting
- Evaluation protocols: walk-forward, purging, CPCV
- Preprocessing practice: prevalence of MinMaxScaler on price levels and its out-of-range failure mode
- **Gap synthesis:** no prior work measures the temporal decay of the cross-variate advantage on crypto

### 3. Methodology
- 3.1 Data provenance, gap characterization, and the segmentation rule
- 3.2 Variate families and the K ladder, including effective dimensionality
- 3.3 Preliminary market efficiency tests and pre-model dependency measurement
- 3.4 iTransformer architecture, including the K=1 degeneracy note, the deliberate redundancy of the K=12 rung, and the `use_norm` scaler justification
- 3.5 Baselines
- 3.6 Walk-forward protocol with purging, including the CPCV rejection paragraph
- 3.7 Metrics and statistical tests, including the clustered-inference specification

### 4. Results and Discussion
- 4.1 Data characteristics, gap profile, and market efficiency
- 4.1b Effective dimensionality: static and rolling participation ratio
- 4.2 RQ1: nominal count versus effective dimensionality
- 4.3 RQ2: decay of the cross-variate advantage ← **core of the paper**
- 4.4 RQ3: optimal retraining cadence
- 4.5 Horizon sensitivity
- 4.6 Cross-variate attention interpretation
- 4.7 Economic evaluation with transaction costs
- 4.8 Limitations

### 5. Conclusion
Answer all three RQs explicitly with numbers. State practical implications. Note directions for future work.

---

## 10. Tables and figures

### Tables

| No | Content |
|---|---|
| 1 | Dataset description, period, bar count, gap profile (blocks, durations, windows lost) |
| 2 | Descriptive statistics + ADF + Variance Ratio + Hurst |
| 2b | Correlation matrix and eigenvalue spectrum per K rung, with participation ratio → **K_eff column feeds RQ1** |
| 3 | Hyperparameter configuration for all models |
| 4 | Main results: MSE, MAE, RelMSE, DA per model and per K (mean ± std) |
| 5 | Per-block decay B1–B6 for each K, with D(b) and the derived b* at τ ∈ {2.5%, 5%, 10%} |
| 6 | Diebold–Mariano p-values, pairwise comparison matrix |
| 7 | Horizon sweep H ∈ {1, 3, 24, 168} |
| 8 | Economic evaluation: Sharpe, MDD, turnover, before and after costs |

### Figures

| No | Content |
|---|---|
| 1 | Rolling-origin walk-forward scheme with purging and segment boundaries |
| 2 | iTransformer architecture and inverted tokenization flow |
| 2b | Rolling participation ratio and rolling OLS R², 2018–2026 ← **establishes H2's premise before any model runs** |
| 3 | **Decay curve A(b) versus b for K = 1, 4, 8, 12** ← key figure |
| 4 | RelMSE per block, all models |
| 5 | Cross-variate attention heatmap: calm regime versus stress regime |
| 6 | Horizon sensitivity |
| 7 | Strategy equity curve before and after costs |

Figure 3 carries the entire paper. If only one figure could appear in a graphical abstract, it is Figure 3.

Table 2b changed role. It previously argued that K=12 is *not* redundant. It now **measures how redundant each rung is**, and that measurement is RQ1's independent variable rather than a defensive footnote.

---

## 11. Risk management

| Risk | Likelihood | Mitigation |
|---|---|---|
| RQ1 null (K=8 not superior to K=1) | medium | Pilot at Stage 5, gated on K=8. If null, switch the title to *Effective Dimensionality and Temporal Decay in Cross-Variate Attention: A Walk-Forward Study of iTransformer on Hourly Bitcoin Data* and reposition as a finding supporting PatchTST's channel-independence claim |
| Measured K_eff far below the ~6.5 estimate | **medium–high** | Detected at Stage 3b, before any training. Re-cut the ladder to the measured saturation point. This is why Stage 3b precedes Stage 5 |
| Flat 4→8 rung, no multivariate benefit at all | medium | Full null with no cross-asset fallback, since option C was rejected. **Early warning is the rolling OLS R² at Stage 3b**: if microstructure R² is already near zero and stable, renegotiate the title then rather than after the main grid |
| No detectable decay | medium | Still a valid finding: the advantage is structural rather than regime-specific |
| Gap blocks far above ~100 | low | Fetcher report reveals this at Stage 2. If window loss exceeds ~20%, narrow the analysis window to the best-covered span rather than relaxing the segmentation rule |
| Seed variance exceeds the effect | medium | Increase to 5 seeds; report confidence intervals |
| Only 13 clusters for the decay regression | **certain** | Not a risk to mitigate but a specification to get right: wild cluster bootstrap, Section 7.2 |
| GPU quota exhausted | medium | Prioritize the main grid; the horizon sweep can be trimmed to 2 origins |

### Leakage checklist

Verify before running the main grid:

- [ ] `StandardScaler` refit at every origin, on the 21-month training sub-block only
- [ ] `use_norm=True` confirmed active; scale-invariance verification test passed (Section 4.4)
- [ ] Identical scaler and scale space used for iTransformer and all baselines
- [ ] **Windows validated by timestamp, not by positional index** — no window spans a gap
- [ ] **Log-returns computed per segment before the scaler is fitted**
- [ ] **No reindexing to a full hourly grid anywhere in the pipeline**
- [ ] No `rolling(center=True)` on any feature
- [ ] Hyperparameters selected from the validation block, never from B1–B6
- [ ] H-step purge active at every train–test boundary
- [ ] No imputation of any kind: no ffill, no bfill, no interpolation
- [ ] Rejected-window count logged per origin and reconciled against `119 × gap_blocks`
- [ ] Split indices persisted and verified non-overlapping
- [ ] Raw predictions saved for every run

The first five items are fatal. The purge item has small impact but remains mandatory.

**The three bolded items replace the former "forward-fill only, never bfill" rule.** That rule was written to prevent lookahead and it does so correctly, but it does not make forward-fill safe when the target is a log-return: ffill fabricates a zero followed by a compressed jump, and zeroes out the Parkinson and Garman–Klass estimators at exactly the wrong bar. Section 2.3 gives the derivation.

---

## 12. Timeline

| Week | Activity | Deliverable |
|---|---|---|
| 1 | Ingest via `binance_spot_klines.py`, read gap report, fix the analysis window | `BTCUSDT_1h.parquet`, Table 1 |
| 2 | Per-segment features, stationarity tests, **pre-model PR and rolling R²** | `features_1h.parquet`, Tables 2, 2b, Figure 2b, K_eff gate |
| 3 | Split generator, iTransformer implementation, RQ1 pilot at K=1 vs K=8 | Title decision gate |
| 4–5 | Main grid, 156 runs | Raw predictions persisted |
| 6 | Baselines, 7 models | Table 4 |
| 7 | Horizon sweep | Table 7, Figure 6 |
| 8 | Aggregation, DM tests, decay regression | Tables 5, 6; Figures 3, 4 |
| 9 | Attention extraction, economic evaluation | Figures 5, 7; Table 8 |
| 10–11 | Draft writing | Complete draft |
| 12 | Revision, format to journal template | Submission-ready manuscript |

---

## 13. Core references

1. Liu, Y., Hu, T., Zhang, H., Wu, H., Wang, S., Ma, L., & Long, M. (2024). iTransformer: Inverted Transformers Are Effective for Time Series Forecasting. *ICLR 2024*.
2. Zeng, A., Chen, M., Zhang, L., & Xu, Q. (2023). Are Transformers Effective for Time Series Forecasting? *AAAI 2023*. (DLinear)
3. Nie, Y., Nguyen, N. H., Sinthong, P., & Kalagnanam, J. (2023). A Time Series is Worth 64 Words: Long-term Forecasting with Transformers. *ICLR 2023*. (PatchTST)
4. Han, L., Ye, H.-J., & Zhan, D.-C. (2024). The Capacity and Robustness Trade-off: Revisiting the Channel Independent Strategy for Multivariate Time Series Forecasting. *IEEE TKDE*, 36(11), 7129–7142. (theoretical basis for H2)
5. Diebold, F. X., & Mariano, R. S. (1995). Comparing Predictive Accuracy. *Journal of Business & Economic Statistics*.
6. Harvey, D., Leybourne, S., & Newbold, P. (1997). Testing the equality of prediction mean squared errors. *International Journal of Forecasting*.
7. Bergmeir, C., & Benítez, J. M. (2012). On the use of cross-validation for time series predictor evaluation. *Information Sciences*, 191, 192–213.
8. López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
9. Lo, A. W., & MacKinlay, A. C. (1988). Stock Market Prices Do Not Follow Random Walks. *Review of Financial Studies*.
10. Gama, J., Žliobaitė, I., Bifet, A., Pechenizkiy, M., & Bouchachia, A. (2014). A survey on concept drift adaptation. *ACM Computing Surveys*, 46(4).

### Added for the spot-only, feature-based design

11. Brownlees, C. T., & Gallo, G. M. (2006). Financial Econometric Analysis at Ultra-High Frequency: Data Handling Concerns. *Computational Statistics & Data Analysis*, 51(4), 2232–2245. — canonical high-frequency data handling; already applied to Bitcoin price filtering in later work. **Verified.**
12. Yang, J., Hu, Y., Zhang, K., Niu, L., Yu, P. S., & Ding, K. (2026). Revisiting Multivariate Time Series Forecasting with Missing Values. arXiv:2509.23494. — shows empirically that unsupervised imputation corrupts the data distribution and degrades forecast accuracy. **Verified as a preprint;** check for a peer-reviewed version before submission and cite as supporting evidence rather than as the load-bearing citation.
13. Rubin, D. B. (1976). Inference and Missing Data. *Biometrika*, 63(3), 581–592. — origin of the MCAR / MAR / MNAR taxonomy. Cited in order to explain why it does **not** apply to exchange downtime.
14. Hansen, P. R., & Lunde, A. (2006). Realized Variance and Market Microstructure Noise. *JBES*, 24(2), 127–161. — discarding observations can improve volatility estimators.
15. Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2008). Bootstrap-Based Improvements for Inference with Clustered Errors. *Review of Economics and Statistics*, 90(3), 414–427. — wild cluster bootstrap, required at 13 clusters (Section 7.2).
16. Lo, A. W. (2004). The Adaptive Markets Hypothesis. *Journal of Portfolio Management*, 30(5), 15–29. — theoretical root of the participant-composition mechanism behind H2.

Entries 13–16 were assembled from memory rather than verified against the source in this revision. **Verify volume, issue, and page ranges before citing.**

A fuller, section-mapped reference list is maintained separately in `reference_library_itransformer_btc.md`.

Verify every citation against the original source before submission.
