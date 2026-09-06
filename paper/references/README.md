# `paper/references/` — reference library

**Generated. Do not hand-edit** — run `python tools/fetch_references.py --index`; the source is `references.bib`.

`CLAUDE.md` §13.3 forbids a citation without a **verified DOI and the source read**. The `verified` tier below records how far each entry got; only `read` clears §13.3 in full.

| Tier | Meaning | Count |
|---|---|---|
| `read` | read in full | 8 |
| `doi-resolved` | DOI resolved | 52 |
| `artifact` | identity from the PDF itself | 8 |
| `screened` | search result only | 2 |

**70 entries — 52 with a PDF on disk, 18 metadata-only.**

## Transformer based

*LTSF transformer architectures, the channel-independence debate, and the attention-as-explanation dispute*

`paper/references/transformer-based/` — 10 entries

| Key | Reference | Year | Identifier | Tier |
|---|---|---|---|---|
| `vaswani2017attention` | Attention Is All You Need — *Advances in Neural Information Processing Systems (NIPS)* | 2017 | arXiv:1706.03762 | `artifact` |
| `jain2019attention` | Attention is not Explanation — *Proc. NAACL-HLT* | 2019 | `10.48550/arXiv.1902.10186` | `doi-resolved` |
| `wiegreffe2019attention` | Attention is not not Explanation — *Proc. EMNLP-IJCNLP* | 2019 | `10.48550/arXiv.1908.04626` | `doi-resolved` |
| `nie2023patchtst` | A Time Series is Worth 64 Words: Long-term Forecasting with Transformers — *International Conference on Learning Representations (ICLR)* | 2023 | arXiv:2211.14730 | `read` |
| `wu2023timesnet` | TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis — *International Conference on Learning Representations (ICLR)* | 2023 | arXiv:2210.02186 | `artifact` |
| `han2024capacity` | The Capacity and Robustness Trade-off: Revisiting the Channel Independent Strategy for Multivariate Time Series Forecasting — *IEEE Transactions on Knowledge and Data Engineering* | 2024 | `10.1109/TKDE.2024.3400008` | `doi-resolved` |
| `kim2024selfattn` | Are Self-Attentions Effective for Time Series Forecasting? — *Advances in Neural Information Processing Systems (NeurIPS)* | 2024 | arXiv:2405.16877 | `artifact` |
| `liu2024itransformer` | iTransformer: Inverted Transformers Are Effective for Time Series Forecasting — *International Conference on Learning Representations (ICLR)* | 2024 | arXiv:2310.06625 | `read` |
| `lin2025tqnet` | Temporal Query Network for Efficient Multivariate Time Series Forecasting | 2025 | arXiv:2505.12917 | `doi-resolved` |
| `liu2025timebridge` | TimeBridge: Non-Stationarity Matters for Long-term Time Series Forecasting | 2025 | arXiv:2410.04442 | `doi-resolved` |

## Baselines and components

*Non-transformer comparators, plus the components the model is built from (instance normalisation, optimiser)*

`paper/references/baselines-and-components/` — 7 entries

| Key | Reference | Year | Identifier | Tier |
|---|---|---|---|---|
| `hoerl1970ridge` | Ridge Regression: Biased Estimation for Nonorthogonal Problems — *Technometrics* | 1970 | `10.1080/00401706.1970.10488634` | `doi-resolved` |
| `hochreiter1997long` | Long Short-Term Memory — *Neural Computation* | 1997 | `10.1162/neco.1997.9.8.1735` | `doi-resolved` |
| `kingma2015adam` | Adam: A Method for Stochastic Optimization — *International Conference on Learning Representations (ICLR)* | 2015 | `10.48550/arXiv.1412.6980` | `doi-resolved` |
| `kim2022revin` | Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift — *International Conference on Learning Representations (ICLR)* | 2022 | — | `read` |
| `zeng2023dlinear` | Are Transformers Effective for Time Series Forecasting? — *Proceedings of the AAAI Conference on Artificial Intelligence* | 2023 | `10.1609/aaai.v37i9.26317` | `read` |
| `nochumsohn2025mtlinear` | A Multi-Task Learning Approach to Linear Multivariate Forecasting | 2025 | arXiv:2502.03571 | `doi-resolved` |
| `shifts2026` | Tackling Time-Series Forecasting Generalization via Mitigating Concept Drift — *International Conference on Learning Representations (ICLR)* | 2026 | arXiv:2510.14814 | `doi-resolved` |

## Crypto market

*Bitcoin and cryptocurrency forecasting, market microstructure, and market efficiency*

`paper/references/crypto-market/` — 11 entries

| Key | Reference | Year | Identifier | Tier |
|---|---|---|---|---|
| `lo2004adaptive` | The Adaptive Markets Hypothesis: Market Efficiency from an Evolutionary Perspective — *The Journal of Portfolio Management* | 2004 | `10.3905/jpm.2004.442611` | `doi-resolved` |
| `urquhart2016inefficiency` | The inefficiency of Bitcoin — *Economics Letters* | 2016 | `10.1016/j.econlet.2016.09.019` | `doi-resolved` |
| `makarov2020trading` | Trading and arbitrage in cryptocurrency markets — *Journal of Financial Economics* | 2020 | `10.1016/j.jfineco.2019.07.001` | `doi-resolved` |
| `briola2022dependency` | Dependency Structures in Cryptocurrency Market from High to Low Frequency — *Entropy* | 2022 | `10.3390/e24111548` | `doi-resolved` |
| `cryptostress2022` | Forecasting cryptocurrencies' price with the financial stress index: a graph neural network prediction — *Applied Economics Letters* | 2022 | `10.1080/13504851.2022.2141436` | `artifact` |
| `anastasopoulos2024orderflow` | Order Flow and Cryptocurrency Returns | 2024 | `10.2139/ssrn.5020002` | `doi-resolved` |
| `cryptovolgnn2025` | Forecasting cryptocurrency volatility: a novel framework based on the evolving multiscale graph neural network — *Financial Innovation* | 2025 | `10.1186/s40854-025-00768-x` | `doi-resolved` |
| `fpca2025` | Intraday Functional PCA Forecasting of Cryptocurrency Returns | 2025 | arXiv:2505.20508 | `artifact` |
| `btchawkes2026` | Forecasting Bitcoin price movements using multivariate Hawkes processes and limit order book data — *Decisions in Economics and Finance* | 2026 | `10.1007/s10203-026-00570-z` | `artifact` |
| `btcmemory2026` | Forecasting Bitcoin Price Movements: Memory, Path Dependence and Persistence — *Finance a \'uv\v{e}r -- Czech Journal of Economics and Finance* | 2026 | `10.32065/CJEF.2026.01.03` | `artifact` |
| `btcwalkforward2026` | Machine Learning-Based Bitcoin Trading Under Transaction Costs: Evidence From Walk-Forward Forecasting | 2026 | arXiv:2606.00060 | `artifact` |

## Evaluation protocol

*How out-of-sample performance is measured: rolling-origin evaluation, concept drift, backtest overfitting*

`paper/references/evaluation-protocol/` — 7 entries

| Key | Reference | Year | Identifier | Tier |
|---|---|---|---|---|
| `tashman2000outofsample` | Out-of-sample tests of forecasting accuracy: an analysis and review — *International Journal of Forecasting* | 2000 | `10.1016/S0169-2070(00)00065-0` | `read` |
| `bergmeir2012use` | On the use of cross-validation for time series predictor evaluation — *Information Sciences* | 2012 | `10.1016/j.ins.2011.12.028` | `doi-resolved` |
| `bailey2014deflated` | The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality — *The Journal of Portfolio Management* | 2014 | `10.3905/jpm.2014.40.5.094` | `doi-resolved` |
| `gama2014survey` | A survey on concept drift adaptation — *ACM Computing Surveys* | 2014 | `10.1145/2523813` | `doi-resolved` |
| `bergmeir2018note` | A note on the validity of cross-validation for evaluating autoregressive time series prediction — *Computational Statistics & Data Analysis* | 2018 | `10.1016/j.csda.2017.11.003` | `read` |
| `cerqueira2020evaluating` | Evaluating time series forecasting models: an empirical study on performance estimation methods — *Machine Learning* | 2020 | `10.1007/s10994-020-05910-7` | `read` |
| `arian2024backtest` | Backtest overfitting in the machine learning era: A comparison of out-of-sample testing methods in a synthetic controlled environment — *Knowledge-Based Systems* | 2024 | `10.1016/j.knosys.2024.112477` | `read` |

## Statistical tests

*The tests the code actually runs on the panel: predictive accuracy under nesting, multiplicity control, clustered inference, directional accuracy*

`paper/references/statistical-tests/` — 13 entries

| Key | Reference | Year | Identifier | Tier |
|---|---|---|---|---|
| `newey1987simple` | A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix — *Econometrica* | 1987 | `10.2307/1913610` | `doi-resolved` |
| `lo1988stock` | Stock Market Prices Do Not Follow Random Walks: Evidence from a Simple Specification Test — *The Review of Financial Studies* | 1988 | `10.1093/rfs/1.1.41` | `doi-resolved` |
| `pesaran1992simple` | A Simple Nonparametric Test of Predictive Performance — *Journal of Business & Economic Statistics* | 1992 | `10.1080/07350015.1992.10509922` | `doi-resolved` |
| `diebold1995comparing` | Comparing Predictive Accuracy — *Journal of Business & Economic Statistics* | 1995 | `10.1080/07350015.1995.10524599` | `doi-resolved` |
| `harvey1997testing` | Testing the equality of prediction mean squared errors — *International Journal of Forecasting* | 1997 | `10.1016/S0169-2070(96)00719-4` | `doi-resolved` |
| `white2000reality` | A Reality Check for Data Snooping — *Econometrica* | 2000 | `10.1111/1468-0262.00152` | `doi-resolved` |
| `clark2001tests` | Tests of equal forecast accuracy and encompassing for nested models — *Journal of Econometrics* | 2001 | `10.1016/S0304-4076(01)00071-9` | `doi-resolved` |
| `romano2005stepwise` | Stepwise Multiple Testing as Formalized Data Snooping — *Econometrica* | 2005 | `10.1111/j.1468-0262.2005.00615.x` | `doi-resolved` |
| `clark2007approximately` | Approximately normal tests for equal predictive accuracy in nested models — *Journal of Econometrics* | 2007 | `10.1016/j.jeconom.2006.05.023` | `doi-resolved` |
| `mccracken2007asymptotics` | Asymptotics for out of sample tests of Granger causality — *Journal of Econometrics* | 2007 | `10.1016/j.jeconom.2006.07.020` | `doi-resolved` |
| `cameron2008bootstrap` | Bootstrap-Based Improvements for Inference with Clustered Errors — *The Review of Economics and Statistics* | 2008 | `10.1162/rest.90.3.414` | `doi-resolved` |
| `hansen2011model` | The Model Confidence Set — *Econometrica* | 2011 | `10.3982/ECTA5771` | `doi-resolved` |
| `mackinnon2023cluster` | Cluster-robust inference: A guide to empirical practice — *Journal of Econometrics* | 2023 | `10.1016/j.jeconom.2022.04.001` | `doi-resolved` |

## Feature construction

*Estimators the variates are built from (family F2) and the random matrix theory behind K_eff*

`paper/references/feature-construction/` — 4 entries

| Key | Reference | Year | Identifier | Tier |
|---|---|---|---|---|
| `parkinson1980extreme` | The Extreme Value Method for Estimating the Variance of the Rate of Return — *The Journal of Business* | 1980 | `10.1086/296071` | `doi-resolved` |
| `rogers1991estimating` | Estimating Variance From High, Low and Closing Prices — *The Annals of Applied Probability* | 1991 | `10.1214/aoap/1177005835` | `doi-resolved` |
| `laloux1999noise` | Noise Dressing of Financial Correlation Matrices — *Physical Review Letters* | 1999 | `10.1103/PhysRevLett.83.1467` | `doi-resolved` |
| `plerou2002random` | Random matrix approach to cross correlations in financial data — *Physical Review E* | 2002 | `10.1103/PhysRevE.65.066126` | `doi-resolved` |

## Not on disk

*Resolved DOI, no legal free full text found. Obtain through the UNESA library or the DOI — never cite one of these without reading it first (§13.3).*

18 entries

| Key | Reference | Year | Identifier | Tier |
|---|---|---|---|---|
| `rubin1976inference` | Inference and missing data — *Biometrika* | 1976 | `10.1093/biomet/63.3.581` | `doi-resolved` |
| `garman1980estimation` | On the Estimation of Security Price Volatilities from Historical Data — *The Journal of Business* | 1980 | `10.1086/296072` | `doi-resolved` |
| `brownlees2006financial` | Financial econometric analysis at ultra-high frequency: Data handling concerns — *Computational Statistics & Data Analysis* | 2006 | `10.1016/j.csda.2006.09.030` | `doi-resolved` |
| `giacomini2006conditional` | Tests of Conditional Predictive Ability — *Econometrica* | 2006 | `10.1111/j.1468-0262.2006.00718.x` | `doi-resolved` |
| `pesaran2007selection` | Selection of estimation window in the presence of breaks — *Journal of Econometrics* | 2007 | `10.1016/j.jeconom.2006.03.010` | `doi-resolved` |
| `kaufman2012leakage` | Leakage in data mining: Formulation, detection, and avoidance — *ACM Transactions on Knowledge Discovery from Data* | 2012 | `10.1145/2382577.2382579` | `doi-resolved` |
| `rossi2013instability` | Advances in Forecasting under Instability — *Handbook of Economic Forecasting* | 2013 | `10.1016/B978-0-444-62731-5.00021-X` | `doi-resolved` |
| `bergmeir2014directional` | On the usefulness of cross-validation for directional forecast evaluation — *Computational Statistics & Data Analysis* | 2014 | `10.1016/j.csda.2014.02.001` | `doi-resolved` |
| `zliobaite2015evaluation` | Evaluation methods and decision theory for classification of streaming data with temporal dependence — *Machine Learning* | 2014 | `10.1007/s10994-014-5441-4` | `doi-resolved` |
| `hansen2015equivalence` | Equivalence Between Out-of-Sample Forecast Comparisons and Wald Statistics — *Econometrica* | 2015 | `10.3982/ECTA10581` | `doi-resolved` |
| `inoue2017rolling` | Rolling window selection for out-of-sample forecasting with time-varying parameters — *Journal of Econometrics* | 2017 | `10.1016/j.jeconom.2016.03.006` | `doi-resolved` |
| `lopezdeprado2018advances` | Advances in Financial Machine Learning | 2018 | — | `screened` |
| `makridakis2018concerns` | Statistical and Machine Learning forecasting methods: Concerns and ways forward — *PLOS ONE* | 2018 | `10.1371/journal.pone.0194889` | `doi-resolved` |
| `arnott2019protocol` | A Backtesting Protocol in the Era of Machine Learning — *The Journal of Financial Data Science* | 2019 | `10.3905/jfds.2019.1.064` | `doi-resolved` |
| `baur2019calendar` | Bitcoin time-of-day, day-of-week and month-of-year effects in returns and trading volume — *Finance Research Letters* | 2019 | `10.1016/j.frl.2019.04.023` | `doi-resolved` |
| `lu2019conceptdrift` | Learning under Concept Drift: A Review — *IEEE Transactions on Knowledge and Data Engineering* | 2019 | `10.1109/TKDE.2018.2876857` | `doi-resolved` |
| `ma2019dayofweek` | On the day-of-the-week effects of Bitcoin markets: international evidence — *China Finance Review International* | 2019 | `10.1108/CFRI-12-2018-0158` | `doi-resolved` |
| `hyndman2021fpp` | Forecasting: Principles and Practice | 2021 | — | `screened` |

## Free to read, but not fetchable by script

No subscription needed — only a browser, because the host answers a script with 403 or a bot challenge. Save into the named category folder **keeping the filename**, so the duplicate guard recognises it.

- **baselines-and-components/**`RevIN, Reversible Instance Normalization against Distribution Shift (Kim, ICLR 2022).pdf` — already saved
  - https://openreview.net/pdf?id=cGDAkQo1C0p
  - CLAUDE.md 6.3 -- origin of use_norm. ICLR mints no DOI; OpenReview is the version of record. Blocked by bot challenge, not by paywall
- **crypto-market/**`The Inefficiency of Bitcoin. urquhart2016.pdf` — already saved
  - https://www.sciencedirect.com/science/article/pii/S0165176516303640
  - CLAUDE.md 4.5 -- market efficiency. Unpaywall reports this OA in the Elsevier open archive; ScienceDirect returns 403 to a script

## Rules this directory follows

- **Nothing is ever duplicated.** Files were regrouped into the category folders above on 2026-09-05 by same-volume rename — never by copy — and the migration refused to run until its table accounted for every PDF on disk with no colliding destination. `--audit` reports **0 duplicate pairs**.
- **Four files were renamed, each for a factual reason.** The largest: a file named *Financial econometric analysis at ultra-high frequency* was, on its own first page, **Clark & West (2007)** — the study's headline statistic filed under someone else's title.
- **No file is fabricated.** An entry with no legal free full text carries a resolved DOI and an acquisition route, never a placeholder.
- **The artifact outranks the search result.** Crossref returned a *different paper* for 9 of the 18 originally curated PDFs; where a filename, a search hit and page 1 of the PDF disagree, page 1 wins. See `D89`.
- **`verified` in this directory is not `verified` in `SOURCE_PROVENANCE`.** That flag means *read*, and flipping it requires reading the paper, not resolving its DOI.

