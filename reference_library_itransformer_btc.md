# Reference Library: iTransformer Bitcoin Forecasting Paper

Companion to `research_specification_itransformer_btc.md`. This supersedes the earlier Indonesian reference file and expands it with four previously uncovered areas: non-stationarity handling, directional-accuracy testing, attention interpretability, and crypto market microstructure.

Entries marked **[NEW]** were not in the earlier list. Entries marked **[SPOT-A]** were added when the design moved to spot-only, feature-based variates.

> **What the design change did to this library.** Futures microstructure references lost their load-bearing role and Section I narrowed to spot order flow — where Makarov & Schoar and Cont et al. became *more* important, not less, since `taker_buy_ratio` is now a core variate rather than one of twelve. Two new sections were added: **P** on data provenance and missing values, and **Q** on effective dimensionality. Section E gained clustered-inference references. Nothing was deleted.

> **Verification required.** Volume numbers, page ranges, and DOIs below were assembled from search results, not from access to published versions. Verify every entry against the original source before submission. For arXiv entries, check whether a peer-reviewed version now exists and cite that instead.

---

## Priority reading

Four entries that change how you should write the paper. Read these first.

### 1. The theoretical basis for H2 already exists

**Han, L., Ye, H.-J., & Zhan, D.-C. (2024). The Capacity and Robustness Trade-off: Revisiting the Channel Independent Strategy for Multivariate Time Series Forecasting.** *IEEE Transactions on Knowledge and Data Engineering*, 36(11), 7129–7142. arXiv:2304.05206.

Channel-Dependent strategies have higher capacity but lack robustness on distributionally drifted series; Channel-Independent trades capacity for robust prediction. This is your H2 stated as a general proposition, already tested empirically and theoretically.

Your contribution sharpens as a result: you are not discovering the trade-off, you are **measuring the rate at which it manifests** on crypto data — which nobody has done.

### 2. There is a paper arguing self-attention may not help at all in TSF **[NEW]**

**Kim, D., Park, J., Lee, J., & Kim, H. (2024). Are Self-Attentions Effective for Time Series Forecasting?** *Advances in Neural Information Processing Systems*, 37, 114180–114209.

This questions the value of self-attention in forecasting generally — a more radical position than the channel-independence critique. It is directly adjacent to RQ1 and an examiner familiar with recent NeurIPS work may raise it. Cite it in Related Work as the strongest sceptical position, then note that your K=1 versus K=12 design tests exactly that question on crypto data.

### 3. There is a paper concluding CPCV beats walk-forward

**Backtest overfitting in the machine learning era: A comparison of out-of-sample testing methods in a synthetic controlled environment.** *Knowledge-Based Systems* (2024). DOI: 10.1016/j.knosys.2024.112477 (verify).

CPCV outperforms on Probability of Backtest Overfitting and Deflated Sharpe Ratio; walk-forward shows weaker false-discovery prevention with higher temporal variability.

**This is the paper a reviewer will use against your methodology.** Cite and answer it rather than ignoring it. Your defence: their evaluation targets *strategy selection* among many candidates, where block shuffling is desirable. Yours is a *controlled architecture comparison* where time-since-training is the independent variable.

### 4. Your attention analysis needs a caveat **[NEW]**

**Jain, S., & Wallace, B. C. (2019). Attention is not Explanation.** *NAACL-HLT 2019*, 3543–3556. arXiv:1902.10186.

**Wiegreffe, S., & Pinter, Y. (2019). Attention is not not Explanation.** *EMNLP-IJCNLP 2019*, 11–20. DOI: 10.18653/v1/D19-1002. arXiv:1908.04626.

Section 4.6 of your specification extracts attention maps as interpretation. There is an unresolved debate about whether attention weights constitute explanation at all. Writing that section without acknowledging it is a visible gap.

The practical fix is cheap: Wiegreffe and Pinter propose a test battery including a **uniform-weights baseline** and **variance calibration across random seeds**. You already train 3 seeds per configuration, so the seed-variance check costs nothing. Adding a uniform-attention ablation costs one extra run per configuration.

Framing to use: attention maps are presented as *descriptive evidence of variate reliance*, validated for stability across seeds, not as causal explanation.

---

## A. LTSF architectures

For Related Work and Methodology.

| Reference | Role |
|---|---|
| **Liu, Y., Hu, T., Zhang, H., Wu, H., Wang, S., Ma, L., & Long, M. (2024). iTransformer: Inverted Transformers Are Effective for Time Series Forecasting.** *ICLR 2024* (Spotlight). arXiv:2310.06625. OpenReview `JePfAI8fah` | Primary model. Mandatory. |
| **Zeng, A., Chen, M., Zhang, L., & Xu, Q. (2023). Are Transformers Effective for Time Series Forecasting?** *AAAI 2023*, 11121–11128. arXiv:2205.13504 | DLinear. Mandatory baseline. |
| **Nie, Y., Nguyen, N. H., Sinthong, P., & Kalagnanam, J. (2023). A Time Series is Worth 64 Words: Long-term Forecasting with Transformers.** *ICLR 2023*. arXiv:2211.14730 | PatchTST. Mandatory baseline and lead proponent of channel-independence. |
| **Vaswani, A., et al. (2017). Attention Is All You Need.** *NeurIPS 2017* | Foundational attention citation. |
| **Zhou, H., et al. (2021). Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting.** *AAAI 2021* | Origin of the LTSF-transformer literature. |
| **Wu, H., Xu, J., Wang, J., & Long, M. (2021). Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting.** *NeurIPS 2021* | Architectural context. Optional baseline. |
| **Zhang, Y., & Yan, J. (2023). Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting.** *ICLR 2023* | Alternative approach to cross-dimension dependency. |
| **Wu, H., Hu, T., Liu, Y., Zhou, H., Wang, J., & Long, M. (2023). TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis.** *ICLR 2023* **[NEW]** | Also the source of Time-Series-Library, the standard codebase for LTSF benchmarking. Cite if you use it. |
| **Wang, S., Wu, H., Shi, X., Hu, T., Luo, H., Ma, L., Zhang, J. Y., & Zhou, J. (2024). TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting.** *ICLR 2024* **[NEW]** | Recent non-transformer competitor. Useful for showing awareness of the current landscape. |
| **Wang, Y., Wu, H., Dong, J., Qin, G., Zhang, H., Liu, Y., Qiu, Y., Wang, J., & Long, M. (2024). TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables.** arXiv:2402.19072 **[NEW]** | From the same lab as iTransformer, focused on exogenous variables. Directly relevant if you frame your K=12 layer as exogenous inputs. |

Key claim to quote from the iTransformer paper: temporal-token embeddings that fuse multiple variates at a single timestamp may **fail to learn variate-centric representations and produce meaningless attention maps**. This is the direct justification for your Section 4.6 analysis.

---

## B. Channel independence versus dependence

The theoretical frame for RQ1 and RQ2.

| Reference | Role |
|---|---|
| **Han, Ye & Zhan (2024)**, TKDE 36(11):7129–7142 | Covered above. The single most important reference for H2. |
| **Kim, D., Park, J., Lee, J., & Kim, H. (2024). Are Self-Attentions Effective for Time Series Forecasting?** *NeurIPS* 37:114180–114209 **[NEW]** | Covered above. Strongest sceptical position. |
| **Montero-Manso, P., & Hyndman, R. J. (2021). Principles and Algorithms for Forecasting Groups of Time Series: Locality and Globality.** *International Journal of Forecasting*, 37(4), 1632–1653 | Local versus global framing. Conceptual basis for the CI/CD distinction. |
| **Ilbert, R., et al. (2024). SAMformer: Unlocking the Potential of Transformers in Time Series Forecasting with Sharpness-Aware Minimization and Channel-Wise Attention.** *ICML 2024*. arXiv:2402.10198 **[NEW]** | Channel-wise attention, closely related to iTransformer's mechanism. Notably includes an identity-attention ablation — a design pattern you can borrow for your K=1 control. |
| **Zhao, L., & Shen, Y. (2025). Proactive Model Adaptation Against Concept Drift for Online Time Series Forecasting.** arXiv:2412.08435 | Links concept drift to modern LTSF architectures. Relevant to RQ3 discussion. |

Summary for Related Work: PatchTST and DLinear favour channel-independence; iTransformer favours explicit cross-variate attention; Han et al. explain the trade-off as capacity versus robustness; Kim et al. question whether self-attention helps at all. **Nobody has tested the temporal dimension of this trade-off on crypto data.** That is your gap.

---

## C. Non-stationarity and normalization **[NEW SECTION]**

Needed for Methodology Section 3.3. iTransformer applies instance normalization internally; you should cite where that comes from and why it matters on non-stationary data.

| Reference | Role |
|---|---|
| **Kim, T., Kim, J., Tae, Y., Park, C., Choi, J.-H., & Choo, J. (2022). Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift.** *ICLR 2022*. OpenReview `cGDAkQo1C0p` | RevIN. The origin of instance normalization in LTSF. Explains why per-instance normalization mitigates train/test distribution discrepancy — highly relevant given your non-stationarity premise. |
| **Liu, Y., Wu, H., Wang, J., & Long, M. (2022). Non-stationary Transformers: Exploring the Stationarity in Time Series Forecasting.** *NeurIPS* 35:9881–9893 | Argues that over-stationarization weakens a model's ability to distinguish genuine non-stationary events. A useful nuance: normalizing away regime information may itself cost you signal. |
| **Fan, W., et al. (2023). Dish-TS: A General Paradigm for Alleviating Distribution Shift in Time Series Forecasting.** *AAAI 2023* | Points out that input and prediction windows may follow different distributions, so input-window statistics alone may be insufficient. |
| **Passalis, N., et al. (2019). Deep Adaptive Input Normalization for Time Series Forecasting.** *IEEE TNNLS* | Earlier learned-normalization approach. Optional. |
| **Ye, W., Deng, S., Zou, Q., & Gui, N. (2024). Frequency Adaptive Normalization for Non-stationary Time Series Forecasting.** *NeurIPS* 37:31350–31379 | Recent development. Optional, cite only if you discuss normalization alternatives. |

Practical note: because iTransformer normalizes per instance internally, this does **not** exempt you from fitting `StandardScaler` on the training window only during preprocessing. State both facts explicitly so a reviewer does not assume you conflated them.

---

## D. Evaluation methodology

For Methodology Section 3.5.

| Reference | Role |
|---|---|
| **Tashman, L. J. (2000). Out-of-sample Tests of Forecasting Accuracy: An Analysis and Review.** *International Journal of Forecasting*, 16(4), 437–450 | Classic reference for out-of-sample evaluation. Recommends testing across multiple periods. |
| **Bergmeir, C., & Benítez, J. M. (2012). On the Use of Cross-validation for Time Series Predictor Evaluation.** *Information Sciences*, 191, 192–213 | **Primary justification for rolling-origin.** Widely cited as the standard for time series evaluation. |
| **Bergmeir, C., Hyndman, R. J., & Koo, B. (2018). A Note on the Validity of Cross-validation for Evaluating Autoregressive Time Series Prediction.** *Computational Statistics & Data Analysis*, 120, 70–83 | Important nuance: under certain conditions K-fold CV is valid for autoregressive series. Citing it shows you understand the limits of your own argument. |
| **Cerqueira, V., Torgo, L., & Mozetič, I. (2020). Evaluating Time Series Forecasting Models: An Empirical Study on Performance Estimation Methods.** *Machine Learning*, 109, 1997–2028. arXiv:1905.11744 | Empirical comparison of estimation schemes. Supports choosing rolling-origin over alternatives. |
| **Bergmeir, C., Costantini, M., & Benítez, J. M. (2014). On the Usefulness of Cross-validation for Directional Forecast Evaluation.** *Computational Statistics & Data Analysis*, 76, 132–143 **[NEW]** | Specifically about **directional** forecast evaluation, which is one of your metrics. |
| **López de Prado, M. (2018). *Advances in Financial Machine Learning*.** Wiley | Source of purging, embargo, and CPCV. Mandatory both for adopting purging and for rejecting CPCV. |
| **Knowledge-Based Systems (2024), backtest overfitting comparison** | Counter-argument. Covered above. |

Key point from Bergmeir & Benítez: rolling-origin gives a more robust assessment than fixed-origin because it captures performance across differing conditions — seasonal shifts, level changes, trend evolution. Direct justification for your 12 origins.

---

## E. Statistical tests

For Methodology Section 3.6 and Results Table 6.

| Reference | Role |
|---|---|
| **Diebold, F. X., & Mariano, R. S. (1995). Comparing Predictive Accuracy.** *Journal of Business & Economic Statistics*, 13(3), 253–263 | Base test. Mandatory. |
| **Harvey, D., Leybourne, S., & Newbold, P. (1997). Testing the Equality of Prediction Mean Squared Errors.** *International Journal of Forecasting*, 13(2), 281–291 | Small-sample correction. Mandatory for your configuration. |
| **Newey, W. K., & West, K. D. (1987). A Simple, Positive Semi-definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix.** *Econometrica*, 55(3), 703–708 | HAC estimator. Mandatory because H > 1. |
| **Pesaran, M. H., & Timmermann, A. (1992). A Simple Nonparametric Test of Predictive Performance.** *Journal of Business & Economic Statistics*, 10(4), 461–465 **[NEW]** | **Makes Directional Accuracy statistically testable.** Without it, DA is a descriptive number with no null hypothesis. With it, you can state whether directional performance beats chance. |
| **Pesaran, M. H., & Timmermann, A. (2009). Testing Dependence Among Serially Correlated Multicategory Variables.** *Journal of the American Statistical Association*, 104(485), 325–337 **[NEW]** | Extension handling serial correlation, which your hourly data has. Prefer this over the 1992 version if your returns show autocorrelation. |
| **Anatolyev, S., & Gerko, A. (2005). A Trading Approach to Testing for Predictability.** *Journal of Business & Economic Statistics*, 23(4), 455–461 **[NEW]** | Excess predictability test tied to a trading rule rather than statistical loss. Bridges your Section 4.3 and Section 4.7. |
| **Giacomini, R., & White, H. (2006). Tests of Conditional Predictive Ability.** *Econometrica*, 74(6), 1545–1578 | Conditional predictive ability. Optional but strengthens methodology. |
| **Lo, A. W., & MacKinlay, A. C. (1988). Stock Market Prices Do Not Follow Random Walks: Evidence from a Simple Specification Test.** *Review of Financial Studies*, 1(1), 41–66 | Variance ratio test, for the market efficiency subsection. |
| **Hansen, P. R. (2005). A Test for Superior Predictive Ability.** *Journal of Business & Economic Statistics*, 23(4), 365–380 **[NEW]** | Multiple-comparison-aware test. Relevant because you compare 7+ models; pairwise DM tests alone do not control family-wise error. |

### Clustered inference for the decay regression **[SPOT-A]**

The `A(b) ~ b` regression has 13 origins × 6 blocks = 78 observations, but blocks within an origin share one trained model, so residuals cluster by origin. Thirteen clusters sits well below the usual rule of thumb.

| Reference | Role |
|---|---|
| **Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2008). Bootstrap-Based Improvements for Inference with Clustered Errors.** *Review of Economics and Statistics*, 90(3), 414–427 | **The wild cluster bootstrap.** Standard remedy when the cluster count is small enough that conventional cluster-robust standard errors are biased downward. Directly applicable. |
| **MacKinnon, J. G., & Webb, M. D. (2017). Wild Bootstrap Inference for Wildly Different Cluster Sizes.** *Journal of Applied Econometrics*, 32(2), 233–254 | Refinement covering unequal cluster sizes. Relevant if some origins lose more windows to gaps than others — which they will. |
| **Cameron, A. C., & Miller, D. L. (2015). A Practitioner's Guide to Cluster-Robust Inference.** *Journal of Human Resources*, 50(2), 317–372 | The practical guide. Cite this one if only a single citation fits. |

Implementation: `wildboottest` in Python, `fwildclusterboot` in R. State the cluster count explicitly in the methodology — a reviewer familiar with this literature will look for it, and 13 is low enough that silence reads as an oversight.

**Verification status:** these four were assembled from memory in this revision. Confirm volume, issue, and pages before citing.

---

**HLN correction formula** for the methodology section:

```
S* = S · sqrt[ (T + 1 − 2h + h(h−1)/T) / T ]
```

T = out-of-sample observation count, h = forecast horizon. Compare S* against Student-t with (T−1) degrees of freedom, not standard normal. At h = 1 loss differentials are serially uncorrelated under the null; at h > 1 the HAC estimator corrects overlap-induced autocorrelation. Misspecifying h invalidates the variance estimate.

Validate any custom Python implementation against R's `forecast::dm.test()`, which already implements the HLN modification.

---

## F. Bitcoin market efficiency

For Introduction and Results Section 4.1.

| Reference | Role |
|---|---|
| **Urquhart, A. (2016). The Inefficiency of Bitcoin.** *Economics Letters*, 148, 80–82 | First study of Bitcoin market efficiency. Concludes inefficient over the full sample but **becoming less inefficient over time**. |
| **Nadarajah, S., & Chu, J. (2017). On the Inefficiency of Bitcoin.** *Economics Letters*, 150, 6–9 | Rebuts Urquhart. Finds that a simple power transformation of Bitcoin returns **does** satisfy weak-form EMH. |
| **Bariviera, A. F. (2017). The Inefficiency of Bitcoin Revisited: A Dynamic Approach.** *Economics Letters*, 161, 1–4 | Dynamic Hurst-exponent approach. Finds efficiency improving over time while volatility exhibits long memory. |
| **Sensoy, A. (2019). The Inefficiency of Bitcoin Revisited: A High-Frequency Analysis with Alternative Currency Pairs.** *Finance Research Letters*, 28, 68–73 | High-frequency analysis. Relevant because your data is hourly. |
| **Khuntia, S., & Pattanayak, J. K. (2018). Adaptive Market Hypothesis and Evolving Predictability of Bitcoin.** *Economics Letters*, 167, 26–28 **[NEW]** | Adaptive Market Hypothesis framing: predictability **evolves** rather than being fixed. This is arguably the single best theoretical justification for H2 from the finance side — if predictability itself is time-varying, a static model must decay. |

| **Lo, A. W. (2004). The Adaptive Markets Hypothesis: Market Efficiency from an Evolutionary Perspective.** *Journal of Portfolio Management*, 30(5), 15–29 **[SPOT-A]** | The theoretical root Khuntia & Pattanayak apply to Bitcoin. Now load-bearing rather than optional: with the cross-asset correlation mechanism gone, AMH carries H2's economic argument. Verify pages before citing. |

**How to use these.** Do not claim the market is efficient. State that evidence is mixed and time-varying, then report your own VR and Hurst results for your period and granularity. This converts an assumption into a finding.

Strategic bonus: the fact that Bitcoin efficiency **evolves** reinforces H2 from an economic rather than machine-learning direction. Pair Khuntia & Pattanayak (2018) with Han et al. (2024) — you then have both a finance-theoretic and an ML-theoretic argument for the same hypothesis. That is a notably strong position for a paper at this level.

---

## G. Concept drift and retraining cadence

For RQ3, Section 4.4.

| Reference | Role |
|---|---|
| **Gama, J., Žliobaitė, I., Bifet, A., Pechenizkiy, M., & Bouchachia, A. (2014). A Survey on Concept Drift Adaptation.** *ACM Computing Surveys*, 46(4), Article 44, 1–37 | Canonical survey. Source of the **blind retraining** (periodic) versus **informed retraining** (drift-triggered) distinction, which maps directly onto RQ3. |
| **Lu, J., Liu, A., Dong, F., Gu, F., Gama, J., & Zhang, G. (2019). Learning under Concept Drift: A Review.** *IEEE TKDE*, 31(12), 2346–2363 | More recent review. |
| **On the retraining frequency of global models in retail demand forecasting.** arXiv:2505.00356 (2025) | **Highly relevant.** Empirically tests retraining frequency for global models. Methodologically parallel to RQ3 in a different domain. Verify publication status. |
| **Analyzing the retraining frequency of global forecasting models: towards more stable forecasting systems.** arXiv:2506.05776 (2025) | Same research direction. Useful for framing RQ3 as an actively studied question rather than an invented one. |
| **Cavalcante, R. C., & Oliveira, A. L. I. (2015). An Approach to Handle Concept Drift in Financial Time Series Based on Extreme Learning Machines and Explicit Drift Detection.** *IJCNN 2015* **[NEW]** | Concept drift specifically in financial time series. Older, but directly on-domain. |

The two retraining-frequency preprints matter strategically: their existence proves RQ3 is a recognised research question, while simultaneously showing that **nobody has answered it for crypto assets with channel-dependent architectures**. That is your position.

---

## H. Deep learning for cryptocurrency forecasting

For Related Work. Select 5–8, not all.

| Reference | Role |
|---|---|
| **Fischer, T., & Krauss, C. (2018). Deep Learning with Long Short-Term Memory Networks for Financial Market Predictions.** *European Journal of Operational Research*, 270(2), 654–669 | Classic LSTM-for-finance reference. Among the most cited in the field. |
| **Seabe, P. L., Moutsinga, C. R. B., & Pindza, E. (2023). Forecasting Cryptocurrency Prices Using LSTM, GRU, and Bi-directional LSTM: A Deep Learning Approach.** *Fractal and Fractional*, 7(2), 203 | Frequently cited RNN baseline study. |
| **Khaniki, M. A. L., & Manthouri, M. (2024). Enhancing Price Prediction in Cryptocurrency Using Transformer Neural Network and Technical Indicators.** arXiv:2403.03606 | Transformer with technical indicators on BTC, ETH, LTC. |
| **Izadi, M. A., & Hajizadeh, E. (2025). Time Series Prediction for Cryptocurrency Markets with Transformer and Parallel Convolutional Neural Networks.** *Applied Soft Computing*, 177, Art. 113229 | Hybrid transformer for crypto. Q1 journal. |
| **A Novel Hybrid Transformer-Based Deep Learning Approach for Multi-Step Bitcoin Price Forecasting.** *Engineering, Technology & Applied Science Research* (2026) | Notable because it operates on **log-differenced closing prices** rather than levels. Supports your log-return target choice. |
| **From LSTM to GPT-2: Recurrent and Transformer-Based Deep Learning Architectures for Multivariate High-Liquidity Cryptocurrency Price Forecasting.** *Symmetry*, 18(1), 32 (2025) | Multivariate comparison framework using Binance data. Methodological comparator. |
| **Quang, H. N., et al. (2025). Analysis and Forecasting of Bitcoin Price Volatility: A Deep Learning Approach Using DNN, LSTM, Transformers, and the ARMA-GARCH Model.** *Journal of Applied Mathematics*, 2025, Art. 9089827 **[NEW]** | Uses **hourly** BTC data and compares against GARCH. Closest match to your granularity and baseline structure. |

**Use these to build the gap.** The recurring pattern: fixed chronological splits, price-level targets, absent naive baselines. Note the pattern neutrally, then position your corrections.

---

## I. Crypto market microstructure and order flow **[NEW SECTION]**

Justifies variate families F3, F4, and F5. **These references became more important, not less, when futures data was dropped.** Under the old design `taker_buy_ratio` was one of twelve variates and funding rate carried much of the microstructure argument. It is now one of eight at the K=8 rung and the sole source of *signed* order-flow information, so the theoretical justification has to be correspondingly stronger. Without these, a reviewer will ask why taker-buy imbalance should carry predictive information at all.

Ignore the funding-rate framing in any note below; it no longer applies.

| Reference | Role |
|---|---|
| **Makarov, I., & Schoar, A. (2020). Trading and Arbitrage in Cryptocurrency Markets.** *Journal of Financial Economics*, 135(2), 293–319 | Foundational crypto microstructure paper. Documents that a **common component in order flow explains much of the common component in returns** across Bitcoin exchanges. Direct justification for including `taker_buy_ratio`. |
| **Cont, R., Kukanov, A., & Stoikov, S. (2014). The Price Impact of Order Book Events.** *Journal of Financial Econometrics*, 12(1), 47–88 | Canonical order-flow-imbalance reference. Finds trade flow imbalance carries more information than volume alone. |
| **Almeida, J., & Gonçalves, T. (2024). Cryptocurrency Market Microstructure: A Systematic Literature Review.** *Annals of Operations Research* (verify venue) | Survey. Efficient single citation covering the whole microstructure landscape. |
| **Marshall, B. R., Nguyen, H. T., & Visaltanachoti, N. (2019). Bitcoin Liquidity.** (verify venue) | High-frequency Bitcoin liquidity. Reports average effective spreads around 0.30%, which is a useful empirical anchor for your Section 4.7 cost assumptions. |
| **Order Flow and Cryptocurrency Returns.** *International Review of Financial Analysis* (2026), DOI 10.1016/j.irfa.2026.100029 (verify) **[NEW]** | Recent, directly on-topic. Shows the order-flow–return relation persists across a broad cross-section of cryptocurrencies. |
| **Bitcoin Wild Moves: Evidence from Order Flow Toxicity and Price Jumps.** *Research in International Business and Finance* (2025) **[NEW]** | VPIN and price jumps. Relevant if you discuss regime-conditional behaviour in Section 4.3. Also documents time-of-day and day-of-week effects in BTC order flow — supports adding calendar variates. |
| **Amihud, Y. (2002). Illiquidity and Stock Returns: Cross-section and Time-series Effects.** *Journal of Financial Markets*, 5(1), 31–56 | The Amihud illiquidity measure, widely applied to crypto. Cite if you add a liquidity variate. |

One paragraph in Methodology citing Makarov & Schoar plus Cont et al. converts family F4 from an arbitrary column into a theoretically motivated variate. Worth writing, and now closer to mandatory than optional.

**These also supply the mechanism for H2.** The participant-composition argument — retail-dominated flow 2018–2020, the leverage cycle of 2021, institutional flow after spot ETF approval in 2024 — predicts that order-flow predictability decays as market making tightens. Pair the microstructure references here with Khuntia & Pattanayak (2018) and Lo (2004) in Section F to give H2 an economic mechanism rather than a purely statistical one.

---

## J. Economic evaluation and backtest overfitting

For Section 4.7.

| Reference | Role |
|---|---|
| **Bailey, D. H., & López de Prado, M. (2014). The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality.** *The Journal of Portfolio Management*, 40(5), 94–107 | **Directly applicable.** You run hundreds of configurations; any reported Sharpe must be deflated by the number of trials. |
| **Bailey, D. H., Borwein, J., López de Prado, M., & Zhu, Q. J. (2014). Pseudo-Mathematics and Financial Charlatanism: The Effects of Backtest Overfitting on Out-of-Sample Performance.** *Notices of the AMS*, 61(5), 458–471 | Probability of Backtest Overfitting. Strong citation for the limitations section. |
| **Harvey, C. R., Liu, Y., & Zhu, H. (2016). ...and the Cross-Section of Expected Returns.** *Review of Financial Studies*, 29(1), 5–68 | Multiple testing in financial research. Broader context. |
| **Lo, A. W. (2002). The Statistics of Sharpe Ratios.** *Financial Analysts Journal*, 58(4), 36–52 | Statistical inference for Sharpe ratios under dependence. |
| **White, H. (2000). A Reality Check for Data Snooping.** *Econometrica*, 68(5), 1097–1126 **[NEW]** | The original data-snooping correction. Pairs with Hansen (2005) in Section E. |

If you report Sharpe in Table 8, also report the Deflated Sharpe Ratio with N = the number of configurations actually tried during development. Concealing the trial count is itself a form of selection bias, and a reviewer who knows this literature will ask.

---

## K. Attention interpretability **[NEW SECTION]**

For Section 4.6. Do not write that section without these.

| Reference | Role |
|---|---|
| **Jain, S., & Wallace, B. C. (2019). Attention is not Explanation.** *NAACL-HLT 2019*, 3543–3556. arXiv:1902.10186 | The sceptical position: attention weights frequently fail to correlate with gradient-based feature importance, and alternative attention distributions can yield equivalent predictions. |
| **Wiegreffe, S., & Pinter, Y. (2019). Attention is not not Explanation.** *EMNLP-IJCNLP 2019*, 11–20. arXiv:1908.04626 | The rebuttal, and the source of the practical test battery: uniform-weights baseline, seed-variance calibration, frozen-weights diagnostic, adversarial attention training. |
| **Serrano, S., & Smith, N. A. (2019). Is Attention Interpretable?** *ACL 2019*, 2931–2951 | Erasure-based analysis. Concludes attention noisily indicates importance but is not a reliable indicator. |
| **Bastings, J., & Filippova, K. (2020). The Elephant in the Interpretability Room: Why Use Attention as Explanation When We Have Saliency Methods?** *BlackboxNLP 2020* | Argues saliency methods are preferable. Useful if you want to add a gradient-based check alongside attention maps. |

Note that this debate is scoped to RNN-era NLP architectures. Whether it transfers to variate-level attention in LTSF is genuinely open — which is itself worth one sentence in your limitations. You are not required to resolve it; you are required to show you know it exists.

**Minimum bar for Section 4.6:** report attention-map stability across your 3 seeds, and add a uniform-attention ablation. Then describe attention as descriptive evidence of variate reliance, not causal explanation.

---

## L. Realized volatility

Reserve material, relevant only if you extend toward volatility forecasting.

| Reference | Role |
|---|---|
| **Corsi, F. (2009). A Simple Approximate Long-Memory Model of Realized Volatility.** *Journal of Financial Econometrics*, 7(2), 174–196 | HAR-RV. Standard volatility-forecasting baseline. |
| **Andersen, T. G., & Bollerslev, T. (1998). Answering the Skeptics: Yes, Standard Volatility Models Do Provide Accurate Forecasts.** *International Economic Review*, 39(4), 885–905 | Foundation of realized volatility. |
| **Liu, L. Y., Patton, A. J., & Sheppard, K. (2015). Does Anything Beat 5-Minute RV? A Comparison of Realized Measures Across Multiple Asset Classes.** *Journal of Econometrics*, 187(1), 293–311 | Why 5-minute sampling became the standard. |

---

## M. Indonesian-language literature

Citing Indonesian publications helps with a Sinta target: it signals local-context awareness and improves relevance for reviewers.

- Rizkilloh, M. F., & Widiyanesti, S. (2022). Prediksi Harga Cryptocurrency Menggunakan Algoritma Long Short Term Memory (LSTM). *Jurnal RESTI*, 6(1), 25–31. DOI: 10.29207/resti.v6i1.3630
- Nirraca, M., & Hartati, E. (2024). Prediksi Harga Bitcoin Menggunakan Metode Long Short Term Memory. *Jurnal Digital Teknologi Informasi*, 7(1), 55–65
- Comparative Analysis of LSTM, GRU, and Bi-LSTM Deep Learning Models for Time Series Cryptocurrency Price Forecasting. *Sinkron* (2025)
- Munir, M., et al. (2025). Prediksi Harga Bitcoin Menggunakan Algoritma Long Short-Term Memory. *JIPI*, (verify volume)

**The gap pattern in local literature.** Indonesian-language publications on crypto price forecasting show a recurring set of methodological choices:

1. Target is **price level**, not returns, producing artificially low MSE and RMSE
2. **Fixed chronological split**, evaluated once, with no repeated origins
3. **No naive baseline**, so it is unknown whether the model beats a random walk
4. Metrics reported as **MAPE on price**, which always looks favourable on near-random-walk series
5. **No acknowledgement that exchange data contains gaps at all** **[SPOT-A]** — neither reported nor handled, so whatever the library default did to them is undocumented

Your design corrects all five. State this **neutrally, without singling out authors** — a methodological critique of common practice, not of individuals:

> Most crypto price forecasting studies published in national journals model price levels directly and evaluate results on a single chronological split. This makes it difficult to assess whether reported accuracy improvements exceed a naive predictor, given that financial price series approximate a random walk. This study models log-returns and includes a persistence baseline in every results table.

---

## N. Citation mapping by paper section

| Section | Core references |
|---|---|
| 1. Introduction | Liu 2024; Han 2024; Khuntia & Pattanayak 2018; Urquhart 2016; Nadarajah & Chu 2017; Gama 2014; 2–3 Indonesian |
| 2. Related Work — architectures | Vaswani 2017; Zhou 2021; Wu 2021; Zeng 2023; Nie 2023; Zhang & Yan 2023; Wu 2023; Liu 2024; Wang 2024 |
| 2. Related Work — CI/CD debate | Han 2024; Kim 2024; Montero-Manso & Hyndman 2021; Nie 2023; Ilbert 2024 |
| 2. Related Work — crypto | Fischer & Krauss 2018; Seabe 2023; Khaniki 2024; Izadi & Hajizadeh 2025; Quang 2025; Indonesian refs |
| 2. Related Work — evaluation | Tashman 2000; Bergmeir & Benítez 2012; Bergmeir 2018; Cerqueira 2020; López de Prado 2018 |
| 3.1 Data provenance and gaps | Brownlees & Gallo 2006; Yang 2026; Rubin 1976; Hansen & Lunde 2006 |
| 3.2 Variates and effective dimensionality | Makarov & Schoar 2020; Cont 2014; Almeida & Gonçalves 2024; Laloux 1999; Plerou 2002 |
| 3.2 Market efficiency tests | Lo & MacKinlay 1988; Urquhart 2016; Bariviera 2017; Sensoy 2019; Khuntia & Pattanayak 2018 |
| 3.3 Architecture | Liu 2024; Vaswani 2017; Kim 2022 (RevIN); Liu 2022 (Non-stationary) |
| 3.4 Baselines | Zeng 2023; Nie 2023; Hochreiter & Schmidhuber 1997 |
| 3.5 Evaluation protocol | Bergmeir & Benítez 2012; Tashman 2000; López de Prado 2018; KBS 2024 |
| 3.6 Statistical tests | Diebold & Mariano 1995; Harvey 1997; Newey & West 1987; Pesaran & Timmermann 1992/2009; Hansen 2005; Cameron, Gelbach & Miller 2008 |
| 4.1b Effective dimensionality | Laloux 1999; Plerou 2002; Liu 2024 |
| 4.2 RQ1 results | Han 2024; Kim 2024; Nie 2023; Laloux 1999 |
| 4.3 RQ2 results | Han 2024; Gama 2014; Khuntia & Pattanayak 2018; Lo 2004; Makarov & Schoar 2020 |
| 4.4 RQ3 results | Gama 2014; arXiv:2505.00356; arXiv:2506.05776; Cavalcante & Oliveira 2015 |
| 4.6 Attention interpretation | Liu 2024; Jain & Wallace 2019; Wiegreffe & Pinter 2019; Serrano & Smith 2019 |
| 4.7 Economic evaluation | Bailey & López de Prado 2014; Bailey 2014; Lo 2002; White 2000; Anatolyev & Gerko 2005; Marshall 2019 |
| 4.8 Limitations | KBS 2024; Bergmeir 2018; Jain & Wallace 2019 |

---

## O. Management notes

**Target count.** For a 10–14 page paper in a Sinta 4 journal, aim for 35–45 references. This document lists roughly 70 candidates. Cut anything you have not actually read.

**Recency ratio.** Target at least 60 percent from the last five years. Methodological classics — Diebold & Mariano 1995, Lo & MacKinlay 1988, Newey & West 1987, Pesaran & Timmermann 1992 — are fine regardless of age.

**Access.** Most architecture references are open on arXiv. For paywalled items (TKDE, Economics Letters, Journal of Econometrics, JFE), try the UNESA library, Google Scholar, or author preprints. Never cite a paper you have not read — an examiner may ask what it says.

**Reference manager.** Use Zotero or Mendeley from the start. Citation style follows the target journal's template, typically IEEE for Indonesian informatics journals.

**Preprint status.** Several entries above are currently preprints. Re-check before submission and update to the published version where one exists. Preprints are citable, but published versions carry more weight.

**Minimum viable set.** If time is short, these eighteen are non-negotiable: Liu 2024, Zeng 2023, Nie 2023, Han 2024, Kim 2024, Kim 2022 (RevIN), Bergmeir & Benítez 2012, López de Prado 2018, Diebold & Mariano 1995, Harvey 1997, Pesaran & Timmermann 1992, Gama 2014, Urquhart 2016, Makarov & Schoar 2020, Bailey & López de Prado 2014, **Brownlees & Gallo 2006**, **Cameron, Gelbach & Miller 2008**, **Lo 2004**.

The three additions are structural rather than decorative. Brownlees & Gallo licenses the data-handling section, Cameron et al. is required for the decay regression to be correctly specified at 13 clusters, and Lo 2004 carries H2's economic mechanism now that the cross-asset correlation argument is gone.

---

## P. Data provenance and missing values **[NEW SECTION, SPOT-A]**

For Methodology Section 3.1. The gap-handling decision needs citation support because it departs from the default that most reviewers expect.

| Reference | Role |
|---|---|
| **Brownlees, C. T., & Gallo, G. M. (2006). Financial Econometric Analysis at Ultra-High Frequency: Data Handling Concerns.** *Computational Statistics & Data Analysis*, 51(4), 2232–2245 | **Canonical reference for high-frequency data handling.** Filters, outlier detection, aggregation into analysable series. Already applied to Bitcoin transaction-price filtering in later work, so the transfer to crypto has precedent. **Verified in search.** |
| **Yang, J., Hu, Y., Zhang, K., Niu, L., Yu, P. S., & Ding, K. (2026). Revisiting Multivariate Time Series Forecasting with Missing Values.** arXiv:2509.23494 | Shows empirically that imputation without supervision corrupts the underlying distribution and **actively degrades** forecast accuracy in MTSF. The most directly on-point defence of the no-imputation decision. **Verified as a preprint** — check for a peer-reviewed version and treat as supporting rather than load-bearing. |
| **Rubin, D. B. (1976). Inference and Missing Data.** *Biometrika*, 63(3), 581–592 | Origin of MCAR / MAR / MNAR. Cited in order to argue the taxonomy does **not** apply: when the exchange is down no price forms, so there is no unobserved value to infer. |
| **Little, R. J. A., & Rubin, D. B. (2019). *Statistical Analysis with Missing Data*, 3rd ed.** Wiley | Standard textbook treatment. Use for the one-sentence taxonomy statement rather than Rubin 1976 if a book citation reads better in your target venue. |
| **Hansen, P. R., & Lunde, A. (2006). Realized Variance and Market Microstructure Noise.** *JBES*, 24(2), 127–161 | Shows that discarding a large number of observations can **improve** volatility estimators — the precedent for preferring window exclusion over imputation. Corroborated through secondary citation, primary not read. |
| **Barndorff-Nielsen, O. E., Hansen, P. R., Lunde, A., & Shephard, N. (2009). Realized Kernels in Practice: Trades and Quotes.** *Econometrics Journal*, 12(3), C1–C32 | Practical high-frequency data cleaning. Routinely cited alongside Brownlees & Gallo. |

**Verification status.** Only the first two were verified against sources in this revision. The remaining four are from memory; confirm volume, issue, and pages.

**The literature gap here is real.** Missing-data practice specific to *crypto exchange* data is thin. A short subsection characterizing Binance gaps over 2018–2026 — count, duration distribution, attribution to downtime, windows lost — is a small original contribution that costs nothing, because the data is already in hand.

---

## Q. Effective dimensionality **[NEW SECTION, SPOT-A]**

For Section 2.4 and Results 4.1b. RQ1 now regresses on K_eff, so the participation ratio needs a citable provenance rather than appearing as an invented statistic.

| Reference | Role |
|---|---|
| **Laloux, L., Cizeau, P., Bouchaud, J.-P., & Potters, M. (1999). Noise Dressing of Financial Correlation Matrices.** *Physical Review Letters*, 83(7), 1467–1470 | Random matrix theory applied to financial correlation matrices. The standard entry point for arguing that a correlation matrix's eigenvalue spectrum, not its dimension, carries the information. |
| **Plerou, V., Gopikrishnan, P., Rosenow, B., Amaral, L. A. N., Guhr, T., & Stanley, H. E. (2002). Random Matrix Approach to Cross Correlations in Financial Data.** *Physical Review E*, 65(6), 066126 | Companion to Laloux et al. Source of the participation ratio as a measure of how many components genuinely contribute. |
| **Han, Ye & Zhan (2024)**, TKDE 36(11):7129–7142 | Already in Section B. Relevant again here: the capacity–robustness trade-off predicts that adding *redundant* channels should cost robustness without buying capacity — which is exactly what the K=12 rung tests. |

**Verification status:** all three from memory. Verify before citing.

**Framing.** The iTransformer paper's own benchmarks split into two structurally different types — few features from one entity (ETTh1 with 7, Weather with 21) versus the same metric across many entities (Electricity with 321 clients, Traffic with 862 sensors) — and the paper does not distinguish them. Measuring K_eff makes that distinction quantitative. This is the strongest available framing for the RQ1 contribution.

---
