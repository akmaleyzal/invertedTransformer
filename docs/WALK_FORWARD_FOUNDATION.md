# Walk-forward foundation — what Figure 1 draws, and what published work licenses it

**Purpose.** CLAUDE.md §8 states the protocol. This document states *why each element of it is
defensible*, element by element, with a verified citation beside each one — and, where no citation
exists, says so in those words. It is the answer to the question a methods referee will actually ask:
*why this scheme and not another?*

Every key below resolves in `paper/references/references.bib`. Every DOI was confirmed against
Crossref by DOI lookup (not by relevance search — see `D89` for why that distinction matters).
`verified=doi-resolved` means the identifier and metadata are right and **the source has not been read
end to end**; §13.3 forbids citing above the tier you have actually reached, so anything that carries
argumentative weight in the manuscript must be promoted to `read` first.

---

## 1. What Figure 1 actually shows

`report._figure1` draws two panels, from `config.ORIGINS` and nothing else.

**Upper panel — the scheme.** Fifteen horizontal rows, one per origin `o_i`, on a calendar-year axis
2018–2026. Each row carries, left to right: a 21-month **training** sub-block, a 3-month
**validation** sub-block, then six 30-day **test blocks** in alternating shade. Purge markers sit at
both validation boundaries. What the shape itself states, with no caption needed:

- the window is **rolling and fixed-length** (24 months), not expanding — the staircase moves right at
  both ends rather than growing at one;
- origins are spaced **5 months**, not 6 — consecutive rows shift by less than half a test span;
- consecutive rows **overlap by 79.2 % of their training data** (`D28`), which is a mandatory
  disclosure the picture makes visible.

**Lower panel — one origin resolved in days.** At the eight-year scale of the upper panel `H = 24 h`
is 0.03 % of the training window and thinner than the line drawn for it, so a reader takes its absence
for its absence. The lower panel resolves origin 1 in days: train → purge → validation → purge → six
test blocks, with the out-of-sample span shaded as one region beginning exactly at `o`, and the
**falsification arm** (a model trained fresh at `o + 90 d`, scored on blocks 4–6) drawn beneath. The
purge bands are drawn **wider than 24 hours and the title says so** — the honest way to show a
quantity too small to see.

---

## 2. Element-by-element grounding

| # | Element of Figure 1 | Why it is there | Published grounding |
|---|---|---|---|
| 1 | **Multiple origins, not one train/test split** | A single split estimates accuracy at one point in one regime; its error is an estimate with a sample size of one origin. | `tashman2000outofsample` §3.1 names the three defects of a fixed origin: one error per lead time, susceptibility to "occurrences unique to that origin", and summary measures that blend lead times into a "mélange". §4.3 adds the calendar argument — a test period "marks a single calendar interval … likely to reflect a single phase of the business cycle", so **multiple test periods** are required; Pack (1990) measured one method's MAPE at lead 4 moving 3.1 % → 5.8 % across test periods. `bergmeir2012use` — rolling-origin captures performance across differing conditions. `hyndman2021fpp` §5.10 — the textbook statement. |
| 2 | **Chronological ordering preserved; no K-fold, no shuffling** | RQ2's independent variable is *time since training*. Any scheme that reorders blocks leaves it undefined. | `bergmeir2018note` is cited **against** us and is the stronger move: it *proves* K-fold CV valid for a purely autoregressive model with uncorrelated errors. Its three assumptions fail here, and §2.1 works through them one at a time. `cerqueira2020evaluating` is the empirical half: on real-world non-stationary series, out-of-sample estimators dominate CV variants, and CV's error is **optimistically** biased. |
| 3 | **Rolling 24-month window, *not* expanding** | With an expanding window, training-set size grows with each origin, so *model age* cannot be separated from *training data volume* — and RQ2 is a claim about age. | **`tashman2000outofsample` §4.4 states this study's exact argument, in 2000**: "the principal purpose of a rolling window is to level the playing field in a multiperiod comparison of forecasting accuracy. We might analyze whether a particular method's performance deteriorates between an earlier and later test period. **The comparison would be confounded if the second fit period were longer than the first.**" That is the evaluation-design reason. `giacomini2006conditional` is the econometric one — their framework treats a forecast as the output of a *method* estimated on a finite, non-growing window, so estimation error does not vanish; an expanding window sits outside it. `pesaran2007selection` — under structural breaks the optimal window is *finite*. `rossi2013instability` — out-of-sample performance is itself time-varying. |
| 4 | **Window length = 24 months** | Sample budget: 13,545–15,217 training windows on the feature frame against 12 × 96 input dimensions (`D25`, `docs/ORIGIN_WINDOW_BUDGET.md`). | `inoue2017rolling` — window length is a *free parameter with real consequences* and is estimable. **This study does not estimate it.** Cite the paper in Limitations; do not imply the 24 months were optimised. |
| 5 | **21 / 3 train–validation split inside the window** | Early stopping and ridge α are selected somewhere, and that somewhere must be inside the training window and before the origin. | `hansen2015equivalence` — the sample-split ratio is not innocuous: the power of an out-of-sample comparison depends on where the split falls, so a split chosen after seeing results is a researcher degree of freedom. `arnott2019protocol` — the split, like every protocol element, is declared before the test period is opened. |
| 6 | **H-step purge at *both* boundaries** (train→val and train→test) | A training window whose 24-step target reaches past the boundary carries post-boundary observations into training. At the train→val boundary the contaminated split is the one governing **model selection** (`D24`). | `lopezdeprado2018advances` ch. 7 — purging, the source of the technique in a financial-ML setting. `kaufman2012leakage` — the general taxonomy: leakage in *features* versus leakage in *training examples*; the purge closes the second class, and §5.3's no-rolling-feature property closes the first. |
| 7 | **No embargo** | An embargo guards the *reverse* channel: a test-period bar influencing a training-set feature value. Here no feature uses a rolling window (§5.3), so that channel does not exist. | `lopezdeprado2018advances` ch. 7 defines the embargo and what it protects against; §8.3's argument is that the protected-against path is structurally absent. `kaufman2012leakage` supplies the vocabulary that makes "structurally absent" checkable rather than asserted. **The argument is conditional on §5.3 and must be re-derived if any rolling feature is added.** |
| 8 | **Six 30-day test blocks, no retraining inside them** | This is *periodic (blind) retraining* evaluated at a fixed cadence. RQ3 asks what the cadence should be, which requires observing performance as a function of block index. | `gama2014survey` — source of the blind (periodic) vs informed (drift-triggered) retraining distinction that RQ3 maps onto directly. `lu2019conceptdrift` — the recent review beside it. `zliobaite2015evaluation` — evaluation under temporal dependence is misleading without a persistence-style baseline, which is why §7 makes Naive-RW mandatory. |
| 9 | **Out-of-sample = everything right of `o`** | There is no fourth split. A forecaster at `o` has seen everything before it and nothing after. | `tashman2000outofsample` — the definition of out-of-sample under rolling origins; the shaded span in the lower panel is that definition drawn. |
| 10 | **Errors clustered by origin, not pooled as 90 independent cells** | Consecutive origins share 79.2 % of their training data (58.3 / 37.5 / 16.7 % at strides 2–4), so `A(i,b)` series are dependent by construction. | The dependence argument is **this study's own**, from `D28`'s overlap arithmetic — Tashman does *not* state that rolling-origin errors are statistically dependent, and citing him for it would be citing what he did not write. What he *does* supply is §6.3: Fildes et al. (1998), over 263 series, found the relative **ranking** of methods changed appreciably as the origin varied, which "should discourage forecasters from using a single forecasting origin" — the ground for reporting per-origin and for `D30`'s origin-level dispersion. `cameron2008bootstrap`, `mackinnon2023cluster` supply the inference; §9.2 bounds effective independent training sets near 4. |
| 11 | **CPCV considered and rejected** | CPCV reorders blocks non-chronologically, which leaves time-since-training undefined; it also assumes DGP stability across blocks, which is the assumption this study tests. | `lopezdeprado2018advances` (source of CPCV) and `arian2024backtest` — the *Knowledge-Based Systems* 2024 paper that concludes **CPCV beats walk-forward**. **Cite it and answer it**; §4.2 gives the answer out of that paper's own methods section. Ignoring it is the reviewable failure. |
| 12 | **Test period opened once, after the design is frozen** | The Stage 5 gate that repositioned the title ran on **validation**, not on test (`D27`). | `arnott2019protocol` — the protocol-level authority for pre-registration, held-out test periods and reporting the full trial count. `bailey2014deflated` (already in the library) for the trial-count consequence. |
| 13 | **The result — no model beats Naive-RW** | Read in the frame the protocol establishes, this is a replication, not a bug. | `makridakis2018concerns` — ML methods underperforming simple statistical benchmarks out of sample, under a rolling-origin protocol, over a large series collection. |

---

## 2.1 Why the strongest paper *for* cross-validation does not license it here

Both sources were read end to end on 2026-09-06. This section exists because "K-fold CV is fine for
time series, Bergmeir proved it" is the single most likely challenge to §8, and the answer is not a
preference — it is an assumption check that either passes or fails.

**What `bergmeir2018note` actually proves.** Theorem 1: the cross-validated estimate converges in
probability to the true prediction error, `P̂E →p PE`, under three assumptions.

| | Assumption | Holds here? |
|---|---|---|
| **A1** | `{y_t}` is a **stationary and ergodic** nonlinear AR(p) in its **own lags** — `y_t = g(x_t, θ) + ε_t` with `x_t = (y_{t−1}, …, y_{t−p})` — and the model is **correctly specified** | **No, twice.** iTransformer at K = 8 or 12 consumes eight to twelve *distinct variates*, not p lags of the target, so the study is outside the model class the theorem is stated over. And stationarity is not a background condition here — RQ2 and RQ3 exist to *test* whether the microstructure-to-return mapping is stable in time. Assuming A1 would assume the answer. |
| **A2** | the leave-one-out estimator is consistent | Not assessed. Nothing in this study turns on it. |
| **A3** | the errors `{ε_t}` form a **martingale difference sequence**, hence are serially **uncorrelated** | **No, by construction at H = 24.** Every experiment in the paper is one-step-ahead. An optimal *h*-step forecast error is MA(*h*−1), so at H = 24 the errors are MA(23) and cannot be an MDS. This project already asserts exactly that elsewhere: §9.2 uses a **rectangular long-run variance estimator with truncation lag h−1 = 23** precisely because "all autocovariances to lag 23 are genuinely nonzero". Using K-fold CV would contradict the variance estimator the same manuscript defends. |

The authors state the consequence themselves: *"the violation of A3 follows naturally since ε_t is no
longer a MDS … Therefore, the CV does not work any longer."* Their Experiment 3 measures it — a
seasonal AR(12) process fitted with AR(1…5) models leaves CV **more** biased than out-of-sample
evaluation (MPAE ≈ −44 to −95 against OOS ≈ −26 to −42), and the bias is in the direction of
*under-estimating* error.

**Their own remedy is worth adopting as a robustness check.** They recommend a Ljung–Box test on the
pooled out-of-sample residuals: if the residuals show serial correlation, CV is invalid. Reporting
that test on this study's residuals would convert the argument above from an appeal to assumptions
into a measurement. It is not currently run.

**What `cerqueira2020evaluating` adds is the empirical half, on data of this frequency.** Eleven
estimators over 62 real-world series at half-hourly, hourly and daily granularity, plus Bergmeir's
three synthetic stationary DGPs. Three findings, in ascending order of importance to us:

1. On the **synthetic stationary** cases, cross-validation wins — Bergmeir reproduced, not disputed.
2. Splitting the 62 series by a wavelet-spectrum stationarity test gives **31 stationary / 31
   non-stationary**. On the stationary half plain CV ranks best; on the **non-stationary half it falls
   to among the worst**, with Holdout and repeated holdout on top.
3. **The direction of the bias is systematic**: cross-validation **under-estimates** the loss, while
   out-of-sample and prequential methods **over-estimate** it.

Finding 3 is the one to put in the manuscript. This study's headline is a null — *no model beats
Naive-RW* — and an optimistically biased estimator is precisely the one that manufactures skill that
is not there. Choosing the pessimistic estimator is the conservative choice for the claim being made.

They also fault the earlier CV-favourable comparisons for scoring out-of-sample at a **single origin**,
invoking Tashman's recommendation of multiple test periods — which is what §8.1's fifteen origins
supply, and which makes their criticism inapplicable to this design.

**State the scope honestly.** Both papers study **univariate, purely autoregressive, one-step-ahead**
forecasting. Neither is a multivariate transformer at H = 24. They are cited for the *assumption
boundary* and the *direction of the bias*, not as evidence about this architecture.

---

## 3. What the literature does **not** give you

Three elements of Figure 1 are this study's own design decisions. Presenting them as though a paper
prescribed them is the failure mode this section exists to prevent. Present them as *reasoned choices
with a stated consequence* — a stronger position than a borrowed one, because the reasoning is
checkable.

**3.1 The 5-month origin spacing (`D26`).** No published protocol prescribes it. The reasoning is
arithmetic and is stated in §8.1: with origins spaced *s* months and 30-day blocks, the calendar month
that block `b` lands on is `m₀ + s·i + (b−1) (mod 12)`, so the months visited form a coset of size
`12/gcd(s,12)`. At *s* = 6 each block index visits only two calendar months, so `b` is a deterministic
function of calendar month up to a two-phase alternation, and **a significant β₁ would be
observationally equivalent to a month-of-year effect**. Only *s* coprime to 12 decouples the two; among
those, 5 maximises the origin count subject to the data span.

What the literature supplies is the *premise*, and it must be cited or the argument reads as
numerology: `baur2019calendar` measures month-of-year, day-of-week and time-of-day effects **in
Bitcoin specifically**, and `ma2019dayofweek` measures the same class independently. The confound is
documented in this asset; the spacing is our response to it.

**3.2 The falsification arm (fresh model at `o + 90 d`, scored on blocks 4–6).** Standard rolling-origin
protocols do not carry one. It exists because aged-model decay and calendar difficulty are otherwise
observationally equivalent: if the aged-minus-fresh gap is zero while β₁ < 0, then β₁ is calendar, not
age. The nearest published relative is the blind-vs-informed retraining framing in `gama2014survey`,
but the design is ours. Report it on **RelMSE, never on scaler-space MSE** (`D60i`) — the two arms
carry different `σ_g`.

**3.3 The six-block, 180-day test span.** A convention, not an optimum. Its consequence is stated
rather than hidden: `b*` resolves only to 30-day granularity out to 180 days, so "no decay detected
within 180 days" is a *right-censored* result and must be worded as one. (Not the situation that
occurred — see `D60b`; the estimand turned out undefined for a different reason.)

---

## 4. The four objections a referee will raise, and where the answer lives

| Objection | Answer | Citation |
|---|---|---|
| "Why not K-fold cross-validation? It is more sample-efficient." | Valid only under conditions this study does not meet, and it destroys RQ2's independent variable outright. | `bergmeir2018note` (states the conditions), `cerqueira2020evaluating` (shows the empirical reversal on real data) |
| "CPCV is the modern standard and beats walk-forward." | The paper that shows this measured a **single-path, unpurged** walk-forward. §4.2 works through its methods section. | `arian2024backtest`, `lopezdeprado2018advances` |
| "Your fifteen origins are not fifteen independent observations." | Correct, and stated numerically rather than as "calendar adjacency": 79.2 % overlap at stride 1; effective independent training sets bounded near 4; training-disjoint re-estimate at G = 3 reported with its spread. | `tashman2000outofsample`, `cameron2008bootstrap`, `mackinnon2023cluster` |
| "An expanding window would use more data." | It would also make model age inseparable from training volume, and it leaves the estimation-error framework the comparison relies on. | `tashman2000outofsample` §4.4, `giacomini2006conditional`, `pesaran2007selection` |
| **"Tashman says you must recalibrate as the origin rolls. You did not."** | Correct, and deliberate — see §4.1 below. The handicap he warns about is RQ3's estimand. | `tashman2000outofsample` §4.2 |

### 4.1 The recalibration objection, in full

This is the sharpest objection a referee who has actually read Tashman will raise, and it was found
by reading him rather than by citing him. §4.2 of that paper says:

> Recalibration is the preferred procedure. Updating without recalibrating imposes an arbitrary
> handicap on the forecasting method. […] When it is a (causal) regression model under evaluation,
> failure to recalibrate transforms a rolling-origin evaluation into a fixed-origin evaluation at one
> step ahead and into meaningless figures at longer horizons.

Read carelessly, that condemns this design: weights are frozen for the whole 180-day test span.
Three things separate the two cases, and all three belong in the manuscript.

1. **The degeneracy he names does not occur here, and his own sentence says why.** His mechanism is
   that "the addition of a new data point changes neither the inputs to nor the coefficients of the
   forecasting equation" — a static regression on exogenous predictors. Here the **inputs move**: at
   every forecast origin inside a test block the 96-bar lookback rolls forward, so each of the 720
   forecasts per block is issued from a different input window. Only the *weights* are fixed.
2. **The frozen weights are the measurement, not an omission.** Tashman is optimising an estimate of a
   method's accuracy; RQ3 asks what accuracy *costs* when a model is left in place — the difference
   between the two is the whole point of the study. Reporting a recalibrated-at-every-step number
   would answer a question no one in this paper asked.
3. **Recalibration does happen, at a stated cadence.** Every one of the fifteen origins is a complete
   refit on its own 24-month window: this is recalibration every five months, with the decay in
   between measured rather than assumed. And the **falsification arm** — a model trained fresh at
   `o + 90 d` and scored on the same blocks 4–6 — is precisely the recalibrated comparator Tashman
   asks for, run at one interior point.

Say all three. Saying only the third invites the reader to ask why the interior of the test span was
left alone.

### 4.2 The CPCV objection, answered from the objecting paper's own methods

`arian2024backtest` is the strongest recent case against this protocol, and §8.4 promises an answer.
Reading it supplies one that a referee can check line by line, because the answer is in that paper's
**methods section**, not in a difference of opinion about aims.

**What it found.** Over 28 strategy trials in a synthetic environment (Heston stochastic volatility,
Merton jump-diffusion, drift-burst, regime-switching Markov) plus S&P 500 data, Combinatorial Purged
CV attains the lowest Probability of Backtest Overfitting, and Walk-Forward the lowest best-trial
Deflated Sharpe Ratio and the least stable PBO over time. Take that seriously: it is a real result
about a real deficiency.

**What its Walk-Forward actually is.** Three facts from the paper, in its own words and code.

1. **Single path.** §3.4.2 configures `CrossValidatorController('walkforward', n_splits=4)` — four
   sequential segments. §2.4.7 states the consequence plainly: WF "creates a single backtest path …
   it tests the strategy **only once**, providing limited insight into its robustness under different
   market conditions." That is Tashman's **fixed-origin** design, whose three defects he named in 2000
   (§2 row 1 above), and it is what `cerqueira2020evaluating` independently faults the earlier
   CV-favourable comparisons for. This study runs **fifteen origins × six blocks**.
2. **Unpurged.** In the same section, `purgedkfold` and `combinatorialpurged` each receive `times=`
   and `embargo=0.02`; `walkforward` receives **neither**, and §2.4.2 defines WFCV with no purge at
   all. Their measured gap therefore confounds *combinatorial* with *purged* — the very mechanism
   their §2.4.3 credits for preventing look-ahead bias. This study's walk-forward is purged at **both**
   boundaries (§8.2, `D24`).
3. **They endorse it for this study's purpose.** §2.4.2: WFCV "is particularly pertinent in financial
   machine learning due to its ability to mitigate overfitting and **model decay** risks", and should
   be employed "alongside other methods like CPCV" — not replaced by them. RQ2 and RQ3 *are* model
   decay.

**And read the effect sizes before repeating the abstract.** Kruskal–Wallis on PBO gives
η² = 0.0102; mean PBO is 0.4523 for Walk-Forward against 0.4005 for CPCV; and **Walk-Forward versus
K-Fold is p = 1.0** — indistinguishable. The deficits that are real in their data are the best-trial
DSR (0.189 against ≈ 0.44) and the temporal stability of PBO, not the level of PBO.

**The estimand argument still stands and is stated last, not first.** PBO and DSR measure
*strategy-selection* overfitting among candidates competing on one return series. This study is a
controlled architecture comparison in which time-since-training is the independent variable, and
CPCV's non-chronological block ordering leaves that variable undefined. Leading with this reads as
special pleading; leading with (1) and (2) does not, because they are facts about their code.

**This is Related Work material, not only a defensive paragraph.** The recent finance-ML case against
walk-forward rests on a single-path, unpurged implementation of it, twenty-four years after the
forecasting literature deprecated single-origin evaluation — and `cerqueira2020evaluating` made the
identical criticism, in the opposite direction, of the studies that favoured cross-validation. Stating
that once, with the section numbers, is a contribution the paper can make cheaply.

### 4.3 What Tashman also settles about the metrics

Not a Figure 1 element, but it grounds two rules the manuscript otherwise asserts on its own
authority. §6.2.1 tells forecasters to **avoid scale-dependent error measures** such as RMSE and MAD
when averaging over series that differ in scale or volatility, and to use a **ratio against a naive
method** instead — Collopy and Armstrong's relative absolute error, which "standardize[s] the
component series for degree of change and, hence, degree of forecasting difficulty."

`RelMSE = MSE_model / MSE_naive` is the squared-error analogue of that statistic, and the phrase
*degree of forecasting difficulty* is `D05`'s argument for normalising by the block's own naive
baseline, written in 2000. `D60i` — no cross-origin comparison on scaler-space MSE, because each
origin carries its own `σ_g` — is the same rule reached the hard way, by a units artefact that made a
falsification-arm number read backwards. Cite §6.2.1 at both places.

---

## 5. Draft paragraph for §3.6 of the manuscript

> Models are evaluated under a rolling-origin walk-forward protocol
> [`tashman2000outofsample`; `bergmeir2012use`; `hyndman2021fpp`]. Fifteen origins are placed at
> five-month intervals from 2020-01 to 2025-11. At each origin the model is fitted on a **fixed
> twenty-four-month rolling window** — the final three months of which are held out for early stopping
> and for selection of the ridge penalty — and then evaluated, without retraining, on six consecutive
> thirty-day blocks. The window is rolling rather than expanding so that model age is not confounded
> with training-sample size, and because the asymptotic framework for comparing forecasting *methods*
> presumes a finite estimation window [`giacomini2006conditional`]; under structural breaks the
> optimal window is in any case finite [`pesaran2007selection`; `rossi2013instability`]. Window length
> itself is a free parameter that this study fixes rather than estimates [`inoue2017rolling`]. Targets
> are purged by `H` steps at **both** internal boundaries, so that no training window's target overlaps
> either the validation or the test period [`lopezdeprado2018advances`; `kaufman2012leakage`]; no
> embargo is applied, because no feature in this study uses a rolling window and the reverse leakage
> channel an embargo protects against is therefore structurally absent. Combinatorial purged
> cross-validation was considered and rejected: it reorders blocks non-chronologically, under which
> *time-since-training* — the independent variable of RQ2 — is undefined, and it presumes the
> block-to-block stability this study exists to test [`arian2024backtest`]. Because consecutive origins
> share 79.2 % of their training data, forecast errors are dependent across origins
> [`tashman2000outofsample`]; inference therefore clusters on the origin, with the effective number of
> independent training sets bounded near four [`cameron2008bootstrap`; `mackinnon2023cluster`]. The
> five-month spacing is coprime to twelve, which decouples the test-block index from the calendar
> month; at six-month spacing each block index would visit only two calendar months, and a decay
> coefficient would be observationally equivalent to the month-of-year effect documented in this asset
> [`baur2019calendar`; `ma2019dayofweek`].

---

## 6. Status of these citations

Eighteen entries were added on 2026-09-06 at `verified=doi-resolved`. Per §13.3 that is **half** the
requirement: the identifier is confirmed, the source is not read. Four carry the most weight, and two
of them have since been read end to end.

| Entry | Tier | Why it matters | State |
|---|---|---|---|
| `bergmeir2018note` | **read** (18 pp) | The strongest published case *for* K-fold CV on time series. §2.1 checks its A1–A3 against this study. | Monash working-paper copy on disk. **Not** the CSDA version of record — the text is the July 2017 preprint, the page numbers in the entry are the journal's. |
| `cerqueira2020evaluating` | **read** (28 pp) | The empirical half, on half-hourly/hourly/daily data. Supplies the direction-of-bias finding. | arXiv v1 on disk. |
| `tashman2000outofsample` | **read** (14 pp) | THE canonical citation of rolling-origin evaluation. §3.1, §4.3, §4.4 and §6.2.1 all carry weight; §4.2 raises the objection §4.1 above answers. | Journal PDF on disk. Supplied by hand 2026-09-06 after Unpaywall returned `closed`. |
| `arian2024backtest` | **read** (27 pp) | Argues CPCV **beats** walk-forward; §8.4 promises an answer, and §4.2 gives one from its own methods section. | Journal PDF on disk. Supplied by hand 2026-09-06 through institutional access after Unpaywall returned `closed`. |

**All four are now read, and `tools/fetch_references.py`'s `PAYWALLED` list is empty.** The remaining
citation debt is elsewhere: of 70 bib entries, **8 are `read`**, 53 `doi-resolved`, 8 `artifact`,
2 `screened`. §13.3 needs `read` for every entry the manuscript cites, and §13.1 budgets 35–45
references — so this is the largest single piece of work left in the project, and no part of it is
computational.

**Reading these four changed the document rather than only its `verified` flags**, which is the whole
argument for §13.3's second half. Four things moved that a summary would not have surfaced: §2 row 3
gained Tashman's §4.4 sentence, which states this study's rolling-window rationale better than the
econometric citations do; §2 row 10 **lost** a claim that he establishes the dependence of
rolling-origin errors, which he does not; §4.1 exists because his §4.2 is a live objection to this
design; and §4.2 exists because `arian2024backtest`'s walk-forward turns out to be `n_splits=4`,
single-path and unpurged — a fact available only from its methods section.

`tools/fetch_references.py` downloads only what is legally free; entries whose `note` says `pdf=none`
have a resolved DOI standing in for a file, never a fabricated one.
