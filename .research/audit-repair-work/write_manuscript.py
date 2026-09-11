"""Write the bounded historical-results draft; figures/numbers remain generated."""
from pathlib import Path
root = Path(__file__).resolve().parents[2]
(root / 'paper/manuscript.tex').write_text(r'''% Historical reanalysis draft. New controlled GPU experiments have not run.
% Generate figures/tables/macros: python tools/build_report.py
% Original paper/paper_numbers.json remains historical evidence.
\documentclass[onecolumn,journal,a4paper,11pt]{IEEEtran}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{booktabs,graphicx,amsmath,amssymb,url}
\usepackage[hidelinks]{hyperref}
\graphicspath{{reanalysis_2026-09-09/figures/}}
\input{reanalysis_2026-09-09/tables/manuscript_numbers.tex}
\title{iTransformer for Hourly Bitcoin Returns: A Corrected Walk-Forward
Reanalysis and Controlled Rerun Protocol}
\author{Akmaley~Zal}
\begin{document}
\maketitle

\begin{abstract}
This study examines an iTransformer variate ladder for hourly BTCUSDT spot
log-return forecasting. We reanalyse \StudyRuns{} preserved runs over
\StudyOrigins{} rolling origins after correcting forecast timestamps, target
alignment, loss aggregation, directional accuracy and economic accounting.
Forecast quality is measured on common surviving target windows relative to a
zero-return random-walk forecast, averaging individual seed losses and then
weighting blocks and origins equally. At a 24-hour horizon, the historical
eight-variate iTransformer has mean out-of-sample $R^2=\KEightRtwo{}$
(descriptive across-origin SE \KEightSE{}), versus \RidgeRtwo{}
(SE \RidgeSE{}) for four-variate Ridge. Neural results average five seeds;
Ridge is deterministic. These negative averages describe the tested procedures
and population, not the absence of Bitcoin predictability in general.
Overlapping origins limit inferential interpretation, and a first-block
skill-loss crossing does not identify an optimal retraining cadence. The
historical models also contain architecture and objective differences that
reanalysis cannot remove. We therefore specify a separate 2,130-run exploratory
protocol with corrected iTransformer/PatchTST implementations, fixed training
counts, information-preserving representation controls, refresh controls and
baseline optimization sensitivities. That protocol is implemented for
resumable Kaggle T4 x2 execution but has not produced new training results.
The contribution is an auditable evaluation and explicit separation of
observed evidence from untested experimental changes.
\end{abstract}

\begin{IEEEkeywords}
iTransformer, Bitcoin returns, walk-forward evaluation, forecast alignment,
reproducibility, effective dimensionality
\end{IEEEkeywords}

\section{Introduction}
Adding variables to a forecast can change both the available information and
the difficulty of estimating a useful mapping. This distinction matters for
hourly financial returns, where small differences in forecast error can be
obscured by alignment, scaling or evaluation choices. The central instrument
in this study is iTransformer, with a ladder of one, four, eight and twelve
features derived from the same BTCUSDT spot bars. We examine the association
between this ladder, training-set participation ratio and out-of-sample error.

Three questions organize the investigation. RQ1 asks how forecast error varies
across the variate ladder and across representations of the same information.
RQ2 asks how the multivariate-versus-univariate error gap changes over six
blocks after model selection. RQ3 asks when skill falls below a fraction of
its positive first-block reference. The third question is descriptive: an
optimal retraining policy would additionally require an objective, retraining
costs and comparisons of policies on a shared calendar.

The present evidence is a corrected reanalysis of saved forecasts. An audit
identified differences between the intended estimands and their implementation.
In particular, historical timestamps denoted the beginning of the input
window, and seed ensembles and individual-run losses had been used for
different parts of the analysis. We correct the evaluation without modifying
the original prediction files. Changes to the architecture, training objective
or availability of forecasts are assigned to a new experiment rather than
attributed retrospectively to the old results.

\section{Related Work and Scope}
iTransformer embeds each variate's history as a token and applies attention
across variates \cite{liu2024itransformer}. This motivates a direct test of a
variate ladder, but neither its tokenization nor performance on other datasets
establishes skill on Bitcoin returns. PatchTST uses temporal patches and shared
channel-independent modelling \cite{nie2023patchtst}. Linear forecasting
comparators are also relevant given the LTSF-Linear study
\cite{zeng2023transformers}; its benchmark findings are not imported as a
ranking for the financial sample considered here.

Prior work by Bysik and {\'S}lepaczuk already evaluates iTransformer alongside
other models on hourly BTC-USDT with walk-forward splits and transaction
costs \cite{btcwalkforward2026}. Its abstract supports this overlap in scope.
We do not infer from abstract screening that the work lacks a particular
metric, control or registration. Accordingly, this draft makes no priority
claim for applying iTransformer to cryptocurrency or for using walk-forward
evaluation.

A targeted English-language source check on 10 September 2026 used queries
combining iTransformer with Bitcoin, walk-forward, effective dimensionality,
model decay and log returns. The repository records the exact queries,
opened sources and extent of reading in
\texttt{docs/LITERATURE\_SCOPE\_2026-09-10.md}. It is a bounded check, not a
systematic literature review. No independently dated external registration
has been supplied for this study; the revised analysis and new protocol are
explicitly exploratory.

\section{Data and Evaluation Methods}
\subsection{Data and population}
The source is Binance BTCUSDT spot hourly klines from 1 January 2018 to
1 August 2026, end exclusive. The retained raw fields support price,
volatility, trading-intensity, taker-flow and intrabar-price-location features.
The target is the next bar's close-to-close log return; the forecast horizon
contains 24 individual hourly returns for the headline analysis. UTC is used
throughout.

Missing or unusable bars split the data into continuous segments. No
imputation, clipping of extreme returns or bridging of gaps is performed.
The first return of a segment is unavailable and is dropped. A missing bar
in the downloaded artifact does not by itself identify exchange downtime,
absence of trading or the data-collection failure that caused it. All
forecasting results are conditional on complete continuous input and target
windows. This restriction can select on future availability, limiting both
generalization and economic interpretation.

\input{reanalysis_2026-09-09/tables/table1_dataset.tex}

\subsection{Chronology and common targets}
Origins are spaced five months apart. Each uses 21 months for gradient
training, the following three months for validation, and six 30-day test
blocks. Train and validation windows are wholly contained in their spans;
their targets cannot cross into the next split. The test input may use
history before the test boundary because that history is already observed.

Let $t$ denote the opening timestamp of the first target bar. A lookback of
$L$ hourly bars starts at $t-L$ hours and ends at $t-1$ hour; the target at
step $h$ begins at $t+h-1$ hours. Block assignment uses $t$. The prediction
schema records input start, forecast origin and each target timestamp
separately. Complete horizons, unique keys and chronological relationships
are checked before aggregation.

Historical prediction files are relabelled by their recorded lookback and
reblocked in memory. Comparisons use the intersection of available forecast
times within each origin, horizon and block. Raw actual returns must agree
across the compared runs after inverse scaling. This aligns L48/L96/L192
comparisons but cannot create forecasts never saved by the historical runs.
The common intersection defines the reported population and may be smaller
than any one model's standalone evaluation sample.

\subsection{Scaling, features and losses}
Scalers are fitted only on the purged training rows of each origin. A raw
return $r$ is standardized as $z=(r-\mu_i)/\sigma_i$. The price random walk
predicts raw return zero, corresponding to $-\mu_i/\sigma_i$ in standardized
space. Predicting standardized zero would instead predict the training mean.
Features are per-bar quantities computed from information available by the
forecast time. Their chronology, rather than the choice of dataframe library,
provides the protection against future information.

For model $m$, seed $s$, origin $i$ and block $b$, the primary loss is
\begin{equation}
 L_{msib}=\frac{1}{|\mathcal T_{ib}|H}
 \sum_{t\in\mathcal T_{ib}}\sum_{h=1}^{H}
 (\hat z_{msith}-z_{ith})^2.
\end{equation}
Seed losses are averaged, followed by
$\mathrm{RelMSE}_{mib}=\bar L_{mib}/L_{0ib}$ and
$R^2_{mib}=1-\mathrm{RelMSE}_{mib}$. Blocks and origins then receive equal
weight. This is performance of the evaluated individual-seed procedures,
not MSE of a seed-averaged ensemble prediction. Cumulative-return errors
and direction are different outcomes and do not replace this step-level
loss in forecast-comparison tests.

\subsection{Dimensionality, age and inference}
Participation ratio is measured from training feature correlations. Changing
the original K ladder changes feature identity and information as well as
nominal dimension. Consequently, associations between participation ratio
and loss do not identify a causal dimensionality effect. The new
information-preserving controls are described separately below.

RQ2 uses $A(i,b)=(L_{K1,ib}-L_{K8,ib})/L_{K1,ib}$ and describes its
within-origin slope over blocks. Time since selection/deployment differs
from time since the most recent gradient-training target, which precedes
selection by approximately the validation interval. Metadata records both
the planned cutoff and actual last training target.

The exploratory J-test uses origin-by-block fixed effects and CR1 covariance
clustered by origin, with a $t(G-1)$ reference. Origin overlap leaves
cross-origin dependence unresolved. Clustered-inference methods require
attention to the sampling structure; their practical interpretation is
discussed by MacKinnon et al. \cite{mackinnon2023cluster}. Here, the
independent-origin p-values, bootstrap results, multiple-comparison
adjustments and model-confidence-set memberships remain diagnostics.
They do not license confirmatory rejection. K differences in nonlinear
trained models are not automatically assigned Clark--West nesting.

The reported minimum detectable effect uses observed TEST slopes and is a
post-analysis diagnostic, not prospective power. Five stride-5 origin
triplets are a sensitivity display with only three origins each. Neither
these small triplets nor an invented effective cluster count resolves the
inferential limitation.

For RQ3, $D(i,b)=[R^2(i,1)-R^2(i,b)]/R^2(i,1)$ is defined only when
$R^2(i,1)>0$. Threshold crossing is evaluated at 5\%, with 2.5/10/50\%
sensitivities. A nonpositive reference is undefined. A positive reference
with no crossing is right-censored at six blocks. Independent-subject
confidence bands and log-rank comparisons are withheld, and no optimal
retraining cadence is estimated.

\section{Historical Reanalysis Results}
\subsection{Forecast error and interpretation}
Table~\ref{tab:main} reports the corrected historical comparisons. The K8
iTransformer mean $R^2_{oos}$ is \KEightRtwo{} with descriptive across-origin
SE \KEightSE{}; Ridge K4 is \RidgeRtwo{} with SE \RidgeSE{}. Neural
comparisons average five seeds per origin and Ridge is deterministic.
These summaries use the shared H24 forecast population, mean seed losses
and equal block/origin weighting, with a raw-zero-return baseline.

The iTransformer K1 mean is \KOneRtwo{}. A multivariate model can have a
smaller error than its univariate counterpart while both have negative
average skill against the random walk. This allows a descriptive variate
comparison, but does not establish profitable prediction or a dimensionality
mechanism. Negative overall averages also do not imply that every origin
or every alternative model specification has negative skill.

\input{reanalysis_2026-09-09/tables/table4_main.tex}

\subsection{Age and first-block reference}
The RQ2 slope is \BetaSlope{} per block, with diagnostic cluster SE
\BetaSE{}. Its positive direction is not descriptive evidence for a
narrowing gap. Failure to reject a negative slope is not proof that decay
does not occur; the limitations of overlap and post-test power assessment
apply. Figure~\ref{fig:age} displays the origin trajectories rather than
reducing them to a single apparent decay curve.

\begin{figure}[htbp]
\centering
\includegraphics[width=0.82\textwidth]{figure3_decay.pdf}
\caption{K1-versus-K8 loss advantage across six test blocks in the historical
reanalysis. Each thin line is an origin. The MDE line is a post-test
diagnostic, not prospective power or a new training result.}
\label{fig:age}
\end{figure}

Under the corrected positive-first-block reference, \DecayExcluded{}
origins are excluded and \DecayDefined{} contributes to the H24 K8 crossing
summary. The reported median block is \DecayMedian{}, conditional on that
very small contributing set. It must not be read as a recommended
retraining interval for the entire study. This differs from the old
future-dependent reference and illustrates why a code correction can change
the meaning of an apparently simple decay summary.

The historical aged-minus-fresh RelMSE gap over B4--B6 is \FreshGap{}
(descriptive across-origin SE \FreshGapSE{}). The historical fresh arm uses
one seed while the aged arm averages five, and both training and validation
move in the fresh intervention. This is a combined procedure comparison;
the new five-seed validation-refresh decomposition has not yet run.

\subsection{Architecture and optimization limitations}
Historical predictions came from an iTransformer port without the official
final encoder LayerNorm and a PatchTST port using the earlier local encoder.
DLinear/PatchTST used all-channel objectives while the principal iTransformer
used target-return loss. Several baseline runs reached their historical
epoch cap. Recomputing metrics cannot remove those differences, so the
historical ranking is a ranking of the recorded configurations and training
procedures. It is not a definitive ranking of the architecture families.

\subsection{Direction and conditional economics}
Directional accuracy now uses inverse-scaled returns. For a cumulative
forecast the drift term is $H\mu_i$, not zero. The economic simulator is
long/cash, with a daily round trip at each retained midnight issuance and
both entry and exit costs. Its always-long comparator also trades daily;
it is not continuous buy-and-hold. Missing future targets still determine
which observations are evaluable. Economic summaries therefore describe a
conditional simulation rather than an executable whole-calendar strategy.

Risk metrics use simple returns derived from log wealth. Downside deviation
is the root mean square shortfall below zero over all retained periods,
including zero shortfalls on positive-return periods. Initial capital enters
maximum drawdown. No return, Sharpe, significance or market-efficiency claim
is used to overcome the selection limitation or the forecast-error findings.

\section{Controlled Rerun Protocol}
The following is implemented but unrun. Its results must be reported
separately from the preceding section. The revised manifest has 2,130 unique
run IDs and requires a new code/input-consistent prediction vintage.

\subsection{iTransformer and model controls}
The main iTransformer now includes final encoder LayerNorm. The uniform
branch shares the attention-weight dropout path and freezes unused Q/K
parameters. Allocated parameter count is disclosed separately from trainable
count and does not establish equal effective capacity. PatchTST uses
BatchNorm, residual attention logits, corrected flatten order and the
projection/residual dropout paths, with explicit normalization and padding
adaptations. Small CPU forward checks against pinned official definitions
cover four iTransformer K values and PatchTST target/all objectives in both
evaluation and training modes. They do not establish GPU training parity.

Target-only DLinear and PatchTST receive only the target channel inside the
channel-independent model, so their effective input dimension is one even
when the supplied data tensor contains eight features. Separate all-channel
arms retain the alternative objective. Each objective uses validation-only
LR selection over $10^{-4},10^{-3},10^{-2}$, a cap of 120 epochs and patience
12. Achieved epochs and capped runs must accompany any new ranking; greater
training opportunity is not proof of convergence.

\subsection{Sample, information and refresh controls}
Every run selects 11,500 surviving training windows without replacement
using seed 1729 and training timestamps alone. Scalers use full purged
training rows before this selection. A timestamp-only check confirms that
all declared spans can supply the count. Achieved counts and timestamp
digests will also be checked from run metadata.

Three K8 representation arms retain identical information: identity,
training-fitted whitening of the non-target channels, and an invertible
correlated representation after whitening. The target remains unchanged and
instance normalization is disabled in all three. Matrices, inverses,
condition numbers and training PR are recorded. These comparisons can test
representation sensitivity but still change geometry and optimization;
they cannot identify a PR-only causal effect.

Fresh and validation-refresh arms use five seeds each and share the original
B4--B6 calendar. Validation refresh holds the original training fixed while
moving checkpoint selection to the updated validation window. Fresh also
updates training. The contrast separates these procedures more clearly than
the historical combined intervention, without equating it to a causal
explanation of market regime change.

\subsection{Execution and evidence preservation}
The primary artifact is \texttt{notebooks/iTransformer.ipynb}; package
sources are exported only through its final cell. Two Kaggle T4 GPUs execute
independent runs, one per device. Epoch-boundary checkpoints retain the
optimizer, scheduler, best state, patience and RNG state. Resume accepts
matching code, data, configuration and fit identity; prediction completion
also checks schema and byte hashes.

A shared deadline covers pilot, tuning and the grid. The default ceiling is
11.5 session hours with 45 minutes reserved for saving, and the remaining
weekly quota must be entered from the account meter. The user's limits are
12 hours per session and 30 hours weekly. Completed validation probes are
cached; interrupted epochs can be replayed from the previous saved boundary.
Each saved output consolidates accepted earlier runs and pending checkpoints.
Saving the output and stopping the platform session remain user actions.
The full grid may require multiple sessions or quota weeks; no runtime
guarantee is inferred from the shorter historical schedules.

\section{Limitations and Reproducibility}
The sample is one asset, exchange, granularity and historical period.
Continuous-window selection excludes unavailable outcomes and does not
identify why they are missing. Origin overlap limits inference, and multiple
testing corrections cannot repair incorrect dependence assumptions.
Validation selection, finite training budgets, seed variation and model
adaptations qualify architecture comparisons. Participation ratio is a
training-set representation diagnostic rather than a measure of predictive
information.

The original 1,620 predictions and executed notebook remain archived. The
corrected report is written to a separate directory with prediction-code,
analysis-code, input and grid hashes. Every numeric macro in this draft is
generated from the same JSON as its tables. A drift check recomputes that
report. Regression checks cover target chronology, aggregation, raw direction,
decay counterexamples, strict resume, checkpoint replay and representation
invariance. Small local checks do not certify the new T4 training grid,
which is explicitly pending.

\section{Conclusion}
The corrected historical evaluation gives negative average skill for the
tested headline iTransformer and baseline configurations relative to the
raw-zero-return forecast. It still permits descriptive comparisons across
variates, origins and model procedures. It does not establish absence of
predictability throughout Bitcoin markets, a causal role for effective
dimensionality, an optimal retraining cadence or executable trading profits.
The implemented controlled rerun addresses several design confounds, while
its experimental outcomes remain to be obtained and reported without
selective omission.

\bibliographystyle{IEEEtran}
\bibliography{references/references}
\end{document}
''', encoding='utf-8')
print('Wrote historical-reanalysis manuscript and separate unrun protocol')
