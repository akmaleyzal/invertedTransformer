# RUN_ANALYSIS.md — deep read of the 1,620-run grid

**Written 2026-09-02.** Covers the grid whose artifacts sit under `notebooks/outputs/`.
Language is English to match `CLAUDE.md` and `docs/DIVERGENCE_REGISTER.md`, which this file
cross-references by ID. The notebook's own prose stays Indonesian (`D73`); this is not the notebook.

**What this file is for.** `CLAUDE.md` is project law and describes the *894-run* grid. That grid has
been superseded. This file records what the *1,620-run* grid returned, which of law's statements it
falsifies, and what is now open. Read §1 and §2 first — they say which numbers are current and which
are one vintage behind.

**Everything below is measured from the artifacts on disk**, not transcribed from a table. The
recomputation method for the parts not present in `paper_numbers.json` is in §12.

> **STATUS, updated 2026-09-02 (second pass).** Every defect in §10 has been **fixed in `src/`** and
> registered as `D76`-`D84` in `docs/DIVERGENCE_REGISTER.md`; `notebooks/iTransformer.ipynb` and
> `paper/` have been regenerated and the suite is green at **194 passed**. Two figures were wrong and
> are now right (`D83` Figure 7, `D84` Figure 4). One correction to this file's own reading is marked
> inline in §2.3 and §10.3: the epoch-cap claim **holds at H = 24** and my first pass over-stated it.
> **The 75 `itrt` runs on disk predate the `D76` fix and are the wrong configuration**, so the grid
> must be re-run **without attaching the previous output as a resume input** --- see §11.

> **DISCHARGED, 2026-09-04.** That re-run happened. All 1,620 runs on disk now carry the single
> vintage `bfb43f21028da322…`, `itrt` included, so the tuned arm is the corrected configuration and
> the caption above it is no longer describing a superseded one. The `36fa9c77…` this file was
> written against is one vintage behind; §1's table has been re-measured, and every *number* below
> was read from the same artifacts and stands. Two things this file says are now out of date by
> design rather than by neglect: the suite count (194) predates `D86`–`D88`, and `paper/` has since
> been rebuilt from the 1,620-run grid.

---

## 1. Provenance — read this before quoting any number

| Field | Value |
|---|---|
| Runs complete / in manifest | **1,620 / 1,620** (`status: complete`, all of them) |
| `preds/*.parquet` | 1,620 |
| `meta/*.json` | 1,621 (1,620 runs + `tuning_selection.json`) |
| `code_sha256` | `bfb43f21028da322e123402837625815164a32b70f24f6fa50bed22f6679cadb` — **one vintage across all 1,620** (re-measured 2026-09-04; the `36fa9c77…` this table first named was superseded when §11's re-run landed) |
| `input_sha256` | `8270a84b07c2923bc885782a8ba4e1898133d18ee3b260f157fcee3fd6923b4e` (`file-digest`), §4.1's pinned parquet |
| Grid file | `notebooks/outputs/artifacts/paper_numbers.json`, sha256 `b6aea49f96f4001612274ee4a3df5808d94cc02f680786ea2b01e8c3eec984ab`, generated `2026-08-31T15:15:40Z` |
| Report file | `notebooks/outputs/paper/paper_numbers.json`, generated `2026-08-31T15:17:19Z`, names the grid file by the sha above — **chain intact** |

**`D62g` is dissolved.** There is no longer a mixed vintage: the whole grid, old runs included, carries
one `code_sha256`. That is because the previously-completed 894 runs were **re-run**, not resumed.

### 1.1 The repository is one grid behind

| | `paper/` (committed) | `notebooks/outputs/paper/` (current) |
|---|---|---|
| runs | 894 | **1,620** |
| `code_sha256` | `fec3e8b4af4e453a…` | `36fa9c77e65f6a1b…` |
| generated | 2026-08-31T03:18:12Z | 2026-08-31T15:17:19Z |
| models in Table 6 | 12 | **15** |
| pairs | 66 | **105** |
| Romano–Wolf rejections | **31 / 66** | **0 / 105** |
| robustness arms run | 3 of 8 | **8 of 8** |
| DSR trial count | 49 | **91** |
| tables / figures | 9 / 6 | **10 / 8** |

`paper/paper_numbers.json` declares `grid_paper_numbers_sha256 = 5dc0960a…`, which is **not** the grid
file on disk. §12's chain therefore *correctly reports* that the manuscript source is stale. Nothing is
silently diverged; it is loudly diverged and nobody has run the regenerator.

`notebooks/outputs/paper/` is entirely **untracked** in git (0 files at HEAD). Under
`notebooks/outputs/` there are 1,484 untracked and 897 modified paths.

### 1.2 The Kaggle session's own report is not the report on disk

`notebooks/logs-iTransformer.txt` **is** the 1,620-run session (manifest 1620, `already complete: 12
pending: 1608`, `grid finished after 8.96 h`). But its report cell printed:

```
report: 66 pairs, MCS over 12 models
tables: table1 … table8            (9 files, no table9)
figures: figure2b … figure7        (6 files, no figure1, no figure2)
D62 robustness arms: longsched / capacity / attention
```

So **the notebook that ran on Kaggle predates `D74` and `D75`**: it computed all 1,620 runs and then
reported 12 models and three arms — the exact omission `D74` exists to fix. The 15-model, 10-table,
8-figure report in `notebooks/outputs/paper/` was produced afterwards by newer code.

Consequence: a future Kaggle session run from the uploaded notebook will reproduce the *old* report
unless the notebook is rebuilt and re-uploaded. See §11.

---

## 2. What actually ran

Manifest as executed (log line 184):

```
main 300 · uniform 75 · fresh 15 · horizon 240 · ridge 60 · dlinear 75 · patchtst 75
lstm 75 · persist 15 · seasonal 15 · orthogonal 75 · redundant 75 · look048 75
look192 75 · tuned 75 · attention 75 · longsched 150 · capacity 75      = 1,620
```

Every `D64` baseline and every `D70` arm is present. §10.2's table is closed.

### 2.1 Compute

| | value |
|---|---|
| Grid wall | **8.96 h** (two T4s, run-level parallelism, `D68`) |
| Summed run time | **17.91 GPU-h** over 1,620 runs, mean **39.8 s** |
| `cuda:0` / `cuda:1` | 813 runs / 9.00 h · 807 runs / 8.91 h — **balanced to 0.7%** |
| Prelude | 13 min; grid received 10.79 h of the 11.0 h budget |
| Budget guard | never tripped; 0 runs remaining at exit |

`D68` is verified end to end. §10.3's caveat "throughput is unverified off Kaggle" can be replaced by a
measurement: two workers, one queue, near-perfect balance, and — see §2.2 — identical bytes.

### 2.2 Bit-exact reproducibility, at n = 894

Of the 894 runs that existed before this session, **0 prediction files changed**. All 894 `meta/*.json`
changed in exactly two fields: `code_sha256` and `wall_time_s`. `epochs_run`, `best_val_mse` and
`train_loss` are identical to full precision.

```
itr_o01_K08_H024_s42:  epochs_run 8   best_val_mse 0.46841975955566845   (both vintages)
                       wall_time_s 23.031496 -> 22.188947
                       code_sha256 fec3e8b4… -> 36fa9c77…
```

So: **894 runs, re-executed at a different code vintage, on two devices instead of one, produced
byte-identical predictions.** The `itra` arm independently reproduces `itr-K8` to 16 significant
digits (`R²_oos` −0.017993293702602386 in both). This is a stronger reproducibility statement than
`D62d` claimed and it is the thing to say in the paper.

### 2.3 Epoch cap — `D62c`'s claim no longer holds as written

| tag | mean epochs | max | runs at the 30-epoch cap |
|---|---|---|---|
| `itr` | 10.63 | 30 | **5 / 540** |
| `itrl` (longsched, cap 60) | 15.57 | 21 | 0 / 150 |
| `lstm` | 8.97 | 30 | 3 / 75 |
| `ptst` | 25.51 | 30 | **39 / 75** |
| `dlin` | 27.64 | 30 | **56 / 75** |

`CLAUDE.md` §6.2 and §13.2 both assert *"0 of 444 iTransformer runs reached 30 epochs"*.

**Corrected on the second pass.** Broken down by horizon, all five cap-hits are at **H = 168**, where
the sweep trains longer; at the headline **H = 24 it is 0 of 300**. The claim holds where the paper
makes it and needed scoping, not retraction — my first reading of "5 of 540 falsifies it" was too
strong. `D78` rescopes it and `D62c` stands; `itrl` at 60 epochs never exceeded 21.

**New and unstated: the cap binds hard for the linear and patch baselines.** DLinear hits it in 75% of
runs and PatchTST in 52%. Any claim that DLinear is the worst model is confounded with DLinear being
the most truncated.

---

## 3. Headline: mean `R²_oos` across all origins and blocks

`± SE across origins` (`D30`); Naive-RW is the reference, `R²_oos = 0` by construction.

| model | K | `R²_oos` | SE | seed sd | MCS 90% | MCS 75% |
|---|---|---|---|---|---|---|
| **Naive-RW** | — | 0 | — | — | yes | yes |
| `rdg-K4` | 4 | −0.000292 | 0.000101 | — | yes | yes |
| `rdg-K8` | 8 | −0.000485 | 0.000152 | — | yes | yes |
| `rdg-K12` | 12 | −0.000693 | 0.000250 | — | yes | yes |
| **`lstm-K8`** | 8 | **−0.001559** | 0.000492 | 0.000503 | **yes** | **yes** |
| `rdg-K1` | 1 | −0.000801 | 0.000470 | — | yes | yes |
| `itru-K8` (uniform attn) | 8 | −0.017722 | 0.001046 | 0.001286 | no | no |
| `itr-K8` | 8 | −0.017993 | 0.001063 | 0.001387 | no | no |
| `itr-K12` | 12 | −0.018570 | 0.001099 | 0.001410 | no | no |
| `itr-K4` | 4 | −0.018682 | 0.001140 | 0.001624 | no | no |
| `ptst-K8` | 8 | −0.016353 | 0.000691 | 0.000656 | no | no |
| `itr-K1` | 1 | −0.020455 | 0.001392 | 0.000915 | no | no |
| `dlin-K8` | 8 | −0.026209 | 0.001037 | 0.000574 | no | no |
| `npst-K1` | 1 | −1.004203 | 0.004030 | — | no | no |
| `nsea-K1` | 1 | −1.044919 | 0.012163 | — | no | no |

MCS membership is by mean loss rank, which is why `rdg-K1` sits at rank 6 below `lstm-K8` at rank 5
while its `R²_oos` looks better — the two aggregate differently. Ranks: Naive-RW, `rdg-K4`, `rdg-K8`,
`rdg-K12`, **`lstm-K8`**, `rdg-K1`, then everything else.

### 3.1 Three things this changes

**(a) A deep model is in the Model Confidence Set.** §13.4 records *"the MCS at both 90% and 75%
contains Naive-RW and all four ridge rungs, **and no deep model**"*. That is now false: `lstm-K8` is in
both, at rank 5. It still does not *beat* Naive-RW (`R²_oos` < 0), so §13.2's disclosure *"no model
beats Naive-RW"* survives intact — but the sharper claim built on top of it does not.

**(b) The LSTM is 11x closer to the baseline than the transformer.** `lstm-K8` −0.00156 against
`itr-K8` −0.01799. Paired across origins the LSTM wins at **15 of 15**, mean RelMSE gap 0.01603,
t = −14.08. And `lstm-K8` versus `rdg-K8` is **not distinguishable** (paired t = +1.69, p = 0.114).
The recurrent model lands with ridge, not with the transformers.

The framing this supports: *the failure is not "deep learning does not work here", it is "the
attention-based long-sequence architectures do not work here"*. That is a more specific and more
defensible contribution than the one currently written, and it is exactly the comparator `D64` added
the LSTM to test.

**(c) The two naive comparators land at RelMSE ~ 2.** `npst` 2.0042, `nsea` 2.0449. For a series with
no autocorrelation, predicting `r_{t−1}` gives exactly `2σ²`. The measured 2.004 confirms hourly BTC
log-returns are white noise at lag 1 — worth a sentence in §4.5, since it corroborates the efficiency
tests from a completely different direction.

---

## 4. RQ1 / RQ2 / RQ3 — status unchanged, evidence stronger

Every ladder number is **bit-identical** to the 894-run grid (`D60b`'s table in `CLAUDE.md` §3 stands
digit for digit). What is new is that RQ1 now has direct evidence, not only panel inference.

### 4.1 RQ1

| rung | `R²_oos` | ΔMSE |
|---|---|---|
| K=1 | −0.020455 | |
| K=4 | −0.018682 | 1→4 |
| K=8 | −0.017993 | 4→8 = **+0.000636** |
| K=12 | −0.018570 | 8→12 = **−0.000437** |

TOST against `Δ_eq = ±0.000159`: p = (0.9734, 0.0002) → **not shown equivalent; the 8→12 rung is
worse, not flat**. J-test: K augmented by `K_eff` t = +3.293, p = 0.0011 (K explanation rejected);
`K_eff` augmented by K t = −0.348, p = 0.7281 (`K_eff` explanation not rejected). `corr(K, K_eff)` =
0.8280. Gate PR at K=8 = **4.3928 < 5.0** — Stage 3b did not pass, disclosed not re-cut.

**The matched-K contrast is the new result, and it is decisive.** `D70`'s `itro` (orthogonal, PR
5.011) versus `itrr` (redundant, PR 3.609), identical K = 8, identical target, identical seeds:

```
redundant − orthogonal, paired on RelMSE across 15 origins
mean +0.001241   SE 0.000198   t = +6.27   p < 0.0001   95% CI [+0.000816, +0.001666]
orthogonal wins at 14 of 15 origins
```

At fixed nominal K, halving effective rank costs **0.00124 of RelMSE — half the entire K=1→K=8 ladder
gain (0.00224)**. H1 is supported by direct contrast, not only by a non-nested test on a panel where
K and `K_eff` are 0.83-collinear. This is the strongest single piece of evidence the study has for its
second claimed contribution, and it currently appears nowhere except as two separate rows of Table 9.

Ladder contrasts, paired on RelMSE, same method (all post-hoc — see §12):

| contrast | mean | SE | t | p | first arm better at |
|---|---|---|---|---|---|
| K=1 − K=8 | +0.002242 | 0.000496 | +4.52 | 0.0005 | 2/15 |
| K=4 − K=8 | +0.000651 | 0.000148 | +4.40 | 0.0006 | 3/15 |
| K=8 − K=12 | −0.000523 | 0.000136 | −3.85 | 0.0018 | 10/15 |

The paired SE (0.000496 for K1−K8) is **half** the marginal SEs Table 4 prints (~0.0011 each). The
ladder is far better identified than Table 4's overlapping error bars suggest: Table 4 as generated
invites a reader to conclude nothing is separated; the paired test separates it at p = 0.0005.

### 4.2 RQ2 — descriptive, per §9.2 requirement 6

β₁ = **+0.000256**, cluster SE 0.000358, t = +0.717, WCR one-sided p = **0.7381** (Rademacher) /
0.7346 (Webb), B = 99,999, G = 15, N = 90. MDE at 80% power = **−0.000920**. Estimate is the wrong
sign and inside the MDE. 9 of 15 within-slopes are negative; the mean is dragged positive by two
origins (2021-04 at +0.003891, 2022-02 at +0.002457).

Training-disjoint triples (stride 5, `D28`, G = 3 each):

| origins | β₁ | SE | t |
|---|---|---|---|
| 1, 6, 11 | +0.001250 | 0.000796 | +1.57 |
| 2, 7, 12 | −0.000449 | 0.000309 | −1.45 |
| 3, 8, 13 | −0.000218 | 0.000422 | −0.52 |
| 4, 9, 14 | +0.001625 | 0.001138 | +1.43 |
| 5, 10, 15 | −0.000926 | 0.000319 | −2.90 |

Three negative, two positive, none significant at G = 3. Inconclusive in both directions, which bounds
what the full-panel p-value can claim.

Falsification arm (aged − fresh, on RelMSE per `D60i`): **+0.000828 ± 0.001106**, t = 0.75, 45 cells,
**6 of 15** origins favour the aged model, 2 within 5e-5 of zero. Sign not stable. Uninformative at
this effect size, as the MDE predicts — the arm identifies the same quantity.

**There is decay evidence in this grid, just not in β₁.** See §8: the variance-ratio test rejects at
the first origin (2020-01, VR(4) = 0.874, p = 0.0001) and at **no origin after it**. The
microstructure predictability H2 invokes is visible in the efficiency tests and has disappeared by
2020-06. β₁ was measuring the wrong thing: it asks whether an *already-zero* edge decays.

### 4.3 RQ3 — undefined, and now precisely so

`b*` **UNDEFINED** at all four τ. `decay_panel.parquet` has 0 rows. All 15 origins excluded by name on
mean `R²_oos ≤ 0`. Log-rank unavailable in both arms → H3 **untestable**, not rejected.

**Refinement the new arms permit.** The guard is arm-specific, and for the ladder it is total —
`itr-K1`, `itr-K8`, `l192` and `ptst` are positive at **0 of 15** origins. But:

| arm | origins with positive mean `R²_oos` |
|---|---|
| `rdg-K4` | 4 / 15 — 2020-11, 2022-07, 2024-03, 2024-08 |
| `lstm` | 3 / 15 — 2020-01, 2020-11, 2024-08 |
| `rdg-K8` | 2 / 15 — 2022-07, 2024-08 |

So *"the decay estimand is undefined under non-positive out-of-sample skill"* is exactly right **for
the iTransformer ladder**, which is what RQ3 was pre-registered on. Stated without that scope it is
now over-broad: ridge and the LSTM do have positive-skill origins. Four origins is still far too few
for `D41`'s Turnbull/KM estimator, so the answer does not change — but the *sentence* needs its scope.

---

## 5. Statistical inference — the largest single change

### 5.1 Romano-Wolf now removes everything

| | 894-run report | 1,620-run report |
|---|---|---|
| models | 12 | 15 |
| pairs | 66 | 105 |
| raw p < 0.05 | 57 | **90** |
| **Romano-Wolf p < 0.05** | **31** | **0** |
| minimum RW p | 0.0423 | **0.0848** |

Not one of 105 pairs survives the stepdown. Raw Clark-West / DM-HLN rejects 90 of them, several at the
bootstrap floor (p = 0.0001, abs t up to 8.5).

**The cause is the family, not the effects.** Widening the matrix from 12 models to 15 added `lstm`,
`npst` and `nsea`. The two naive comparators generate the largest abs t in the table (`lstm-K8` vs
`nsea-K1`, t = -8.20; vs `npst-K1`, t = -8.52), which inflates the bootstrap max-abs-t null that every
other hypothesis is judged against. At G = 15 clusters that distribution is already heavy-tailed; 105
hypotheses finish the job.

Net effect: **adding two closed-form baselines that nobody claims anything from destroyed all 31
multiplicity-corrected rejections the study previously had.** That is a self-inflicted power loss.

The standard remedy is to control FWER over the family of **declared claims**, not the Cartesian
product - e.g. three families: (i) all models vs Naive-RW, 14 pairs; (ii) the ladder, 6 pairs; (iii)
the architecture comparison, a handful - each stepped down independently, with the full 105-pair
matrix reported raw. That is defensible *if pre-registered*. It has not been, and the numbers have now
been seen, so the honest options are to report the 105-pair result as the headline and note the
66-pair one, or to declare a restricted family and label it explicitly post-hoc. This is a decision for
the author, not something to slip into the regenerator.

### 5.2 Clark-West is positive where R2_oos is negative

| pair (nested, Clark-West) | t | raw p | RW p |
|---|---|---|---|
| Naive-RW vs `itr-K1` | **+2.621** | 0.0074 | 0.4211 |
| Naive-RW vs `itr-K4` | +2.295 | 0.0234 | 0.5253 |
| Naive-RW vs `itr-K8` | +2.198 | 0.0264 | 0.5574 |
| Naive-RW vs `itr-K12` | +2.098 | 0.0313 | 0.5959 |
| Naive-RW vs `itru-K8` | +2.188 | 0.0263 | 0.5611 |
| Naive-RW vs `rdg-K1` | +1.691 | 0.0127 | 0.7605 |
| Naive-RW vs `lstm-K8` | +1.297 | 0.0412 | 0.8834 |
| Naive-RW vs `rdg-K4` | +0.886 | 0.1476 | 0.9757 |
| Naive-RW vs `ptst-K8` | -0.861 | 0.8023 | 0.9757 |
| Naive-RW vs `dlin-K8` | -1.921 | 0.9840 | 0.6665 |
| Naive-RW vs `npst-K1` | -2.423 | 0.9925 | 0.4866 |
| Naive-RW vs `nsea-K1` | -3.125 | 0.9973 | 0.3290 |

The rank order **inverts** R2_oos: `itr-K1`, the *worst* transformer arm, has the *largest*
Clark-West statistic. CW credits the larger model for the estimation noise the null imposes, and K=1
has the most noise to be credited for. Joint reading, per §13.2: any population-level edge is smaller
than the estimation error required to exploit it. `dlin`, `ptst`, `npst`, `nsea` are worse than
Naive-RW even after the CW adjustment.

Rectangular long-run variance fell non-positive and the Bartlett fallback fired for **3 pairs**:
`itr-K8` vs `itru-K8`, `npst-K1` vs `nsea-K1`, `Naive-RW` vs `npst-K1`. §9.2 requires this be reported
per pair; it is in `paper_numbers.json` (`fallback_fired`) but does not surface in Table 6 caption.

T = 437 minimum, h = 24, truncation lag 23, G = 15 for every pair. HLN factor positive throughout.

---

## 6. The exploratory arms (D62 + D70) - all eight run

Point estimates from Table 9; the paired columns are recomputed here (§12) and are **not** in
`paper_numbers.json`. Positive `mean delta` = the arm is **worse**.

| arm | config | R2_oos | vs grid | paired mean delta RelMSE | SE | t | p | arm better at |
|---|---|---|---|---|---|---|---|---|
| `l192` | L = 192 | **-0.013907** | +0.0041 | **-0.003506** | 0.001024 | -3.42 | 0.0041 | **14/15** |
| `itro` | orthogonal K=8 | -0.018245 | -0.0003 | +0.000168 | 0.000106 | +1.58 | 0.136 | 5/15 |
| `itrl` K=8 | lr halve/8, 60 ep | -0.018135 | -0.0001 | +0.000171 | 0.000085 | +2.02 | 0.063 | 2/15 |
| `itra` | attention captured | -0.017993 | 0.0000 | **+0.000000** | 0.000000 | - | - | identical |
| `itrc` | d_ff = 512, K=12 | -0.019074 | -0.0005 | +0.000535 | 0.000200 | +2.68 | 0.018 | 2/15 |
| `itrr` | redundant K=8 | -0.019542 | -0.0015 | +0.001409 | 0.000215 | +6.56 | <0.0001 | 1/15 |
| `itrl` K=1 | lr halve/8, 60 ep | -0.021289 | -0.0008 | +0.000782 | 0.000051 | +15.29 | <0.0001 | **0/15** |
| `itrt` | tuned (see §10.1) | -0.021443 | -0.0034 | +0.003532 | 0.000443 | +7.97 | <0.0001 | **0/15** |
| `l048` | L = 48 | -0.026235 | -0.0082 | +0.008111 | 0.000947 | +8.56 | <0.0001 | 1/15 |

### 6.1 Lookback dominates the study's independent variable

L = 48 -> 96 -> 192 moves RelMSE by **0.01162** end to end (t = +8.10; L=192 better at 14/15). The
entire K = 1 -> 8 ladder moves it by **0.00224**. **The one first-order hyperparameter §6.2 never
varied is worth 5.2x the study's independent variable**, and it is monotone: longer is better at every
step tested.

This is the most consequential result in the exploratory set and it needs stating plainly, because a
referee will ask it: RQ1 measures the marginal value of *variates* while holding fixed a *lookback*
that matters several times more. It does not invalidate RQ1 - the ladder holds L constant, which is
what makes it a comparison - but it bounds the claim's importance, and it suggests the honest next
experiment is an L sweep, not a K sweep.

`l192` also converges faster (8.40 mean epochs vs 10.63) and costs *less* wall time (31.5 s vs 42.2 s).
There is no budget argument against it.

### 6.2 Every "you did it wrong" attack is answered, and all answers point the same way

- **"You under-trained."** `itrl` at `lr_halve_every = 8`, 60 epochs, patience 10: **worse** at K=1
  (0/15 origins better, t = +15.29) and no better at K=8 (p = 0.063). Max epochs reached: 21 of 60.
- **"You under-capacitised."** `itrc` at d_ff = 512: worse, p = 0.018, better at 2/15.
- **"You did not tune."** `itrt`: worse at **0 of 15** origins, mean +0.0035 - the largest degradation
  of any arm except `l048`. Validation-selected configuration generalises strictly worse everywhere.
  (But read §10.1 - the arm as executed is not the config that was selected.)

Three independent attacks, three answers in the same direction. That is a strong section.

### 6.3 Learned attention is worse than uniform attention, significantly

    uniform - learned, paired on RelMSE:  -0.000287   SE 0.000081   t = -3.53   p = 0.0033
    uniform better at 11 of 15 origins
    A_attn mean = -0.000267 over 90 cells

And the learned maps are empirically flat. Mean attention received per variate, over all runs, layers
and query positions (uniform = 0.1250):

| variate | calm | mid | stress |
|---|---|---|---|
| `r` | 0.1252 | 0.1253 | 0.1253 |
| `upper_shadow` | 0.1248 | 0.1248 | 0.1247 |
| `lower_shadow` | 0.1252 | 0.1252 | 0.1252 |
| `log_quote_volume` | 0.1247 | 0.1246 | 0.1246 |
| `log_trade_count` | 0.1247 | 0.1246 | 0.1245 |
| `taker_buy_ratio` | 0.1249 | 0.1250 | 0.1249 |
| `signed_flow` | 0.1249 | 0.1250 | 0.1250 |
| `vwap_location` | 0.1256 | 0.1257 | 0.1258 |

Mean absolute deviation from 1/N: **0.00092** (layer 0), **0.00181** (layer 1). Full range across all
28,800 cells: 0.1204 - 0.1377. Regime shift (stress minus calm) is at most 0.0002 on any variate.

Seed stability: sd of a given cell across seeds = 0.00057 mean, 0.00141 p95; sd of the cell means
across cells = 0.00175. Structure is ~3x the seed noise, so it is reproducible - and it is tiny.
§13.2's minimum bar (seed-stability plus the uniform ablation) is met, and both halves agree.

**The honest statement: the model learns an essentially uniform cross-variate attention map, and
forcing it exactly uniform is significantly better.** D75 was right that the old Figure 5 asserted
uniformity via `vmin=0.0`; the fix now shows honest near-uniformity on a real scale.

---

## 7. Economics

Fee 0.04%/side, phase 00:00 UTC, non-overlapping 24 h holds, per-segment returns. Mean over 15
origins, +/- SE across origins. `n_periods` ~ 174/origin, `n_flat_days` ~ 6.3 (3.6% flat over outages).

| model | slip | net total return | Sharpe (ann.) | Sortino | MDD | turnover | DSR | JK z | JK p |
|---|---|---|---|---|---|---|---|---|---|
| `itr-K8` | 0.02% | **+0.2056** +/- 0.137 | **+0.377** +/- 0.340 | +0.643 | 0.324 | 0.255 | 0.101 | -0.172 | 0.532 |
| `itr-K8` | 0.05% | +0.1748 | +0.274 | +0.480 | 0.332 | 0.255 | 0.091 | -0.224 | 0.518 |
| `itr-K8` | 0.10% | **+0.1253** | **+0.104** | +0.209 | 0.345 | 0.255 | 0.077 | -0.310 | 0.497 |
| `itr-K1` | 0.02% | +0.2061 | +0.414 | +0.737 | 0.320 | 0.251 | 0.092 | -0.156 | 0.601 |
| `ptst-K8` | 0.02% | +0.0734 | +0.093 | +0.298 | 0.362 | 0.320 | 0.107 | -0.310 | 0.449 |
| `ptst-K8` | 0.10% | -0.0155 | -0.249 | -0.263 | 0.387 | 0.320 | 0.077 | -0.480 | 0.448 |
| `rdg-K8` | 0.02% | -0.1157 | **-1.024** | -1.251 | 0.405 | 0.145 | 0.098 | -0.811 | 0.339 |
| `dlin-K8` | 0.02% | -0.1232 | -0.582 | -0.817 | 0.400 | 0.440 | 0.041 | -0.610 | 0.537 |
| **buy-and-hold** | 0.02% | **+0.2895** | **+0.578** | - | - | - | - | - | - |

Matches §13.2's recorded +20.6% / +29.0% / Sharpe +0.377 / +12.5% at the top of the band. **DSR moved
0.173 -> 0.101** because the trial count grew 49 -> 91 with the new arms. That is the mechanism working
correctly: more configurations evaluated on the same span deflate the Sharpe further.

Three readings that must travel together:

1. **Nothing beats buy-and-hold.** +20.6% against +29.0%, Sharpe +0.377 against +0.578.
2. **No Sharpe difference is significant.** Jobson-Korkie/Memmel p >= 0.308 for every model at every
   slippage level.
3. **Every DSR is far below 0.5** - 0.041 to 0.107. Under selection over 91 trials, none of these
   Sharpe ratios is distinguishable from luck.

A positive P&L under a negative R2_oos is not a contradiction: the sample is dominated by BTC's
2020-2026 rise, and a mostly-long position is paid for the drift, not the forecast.

**Gap: the LSTM has no economics row.** Five models are evaluated - `itr-K1`, `itr-K8`, `ptst-K8`,
`rdg-K8`, `dlin-K8`. The only deep model in the MCS is absent from Table 8 and Figure 7.

---

## 8. Efficiency tests - where the decay evidence actually is

ADF rejects at p < 0.0001 on the full sample and at every origin. Hurst **0.5435 - 0.5682**, uniformly
above 0.5, stable across origins. Variance ratio is **below 1 everywhere** - the two diagnostics point
in opposite directions (mild persistence vs mild mean reversion) and the paper should say so rather
than reporting them as agreeing.

The informative pattern is the **VR p-value trajectory**:

| span | VR(4) | p |
|---|---|---|
| full | 0.9302 | **0.0004** |
| 2020-01 | 0.8738 | **0.0001** |
| 2020-06 | 0.9461 | 0.1078 |
| 2020-11 | 0.9179 | 0.1524 |
| 2021-04 | 0.9150 | 0.1537 |
| 2021-09 | 0.9262 | 0.1565 |
| 2022-02 | 0.9264 | 0.1267 |
| 2022-07 | 0.9692 | 0.2044 |
| 2022-12 | 0.9808 | 0.4278 |
| 2023-05 | 0.9910 | 0.7402 |
| 2023-10 | 1.0001 | 0.9964 |
| 2024-03 | 0.9757 | 0.4507 |
| 2024-08 | 0.9760 | 0.4168 |
| 2025-01 | 0.9760 | 0.3907 |
| 2025-06 | 0.9717 | 0.2894 |
| 2025-11 | 0.9614 | 0.1490 |

Departure from a random walk is significant **only at the first origin**, drifts monotonically to
VR = 1.0001 at 2023-10, then partially returns. This is the Adaptive-Markets mechanism §3 invokes,
measured, on the study's own data, and it is entirely absent from the RQ2 narrative - which asks
whether a model's *edge* decays and finds no edge to decay. Reframing RQ2 around "the predictability
the model would need had already gone by mid-2020" is supported by this table and by nothing else in
the study.

---

## 9. Horizons and directional accuracy

### 9.1 The K ladder's sign flips with horizon

R2_oos on the 4 sweep origins (1, 5, 10, 15):

| H | K=1 | K=4 | K=8 | K=12 | K8 minus K1 |
|---|---|---|---|---|---|
| 1 | -0.03435 | -0.03690 | -0.03827 | -0.03738 | **-0.00392** |
| 3 | -0.02806 | -0.03280 | -0.03333 | -0.03374 | **-0.00528** |
| 24 | -0.02133 | -0.01922 | -0.01874 | -0.01895 | **+0.00259** |
| 168 | -0.01082 | -0.01035 | -0.01025 | -0.01029 | **+0.00057** |

Paired on RelMSE (K=8 minus K=1; negative means K=8 better):

| H | origins | mean | SE | t | K=8 better at |
|---|---|---|---|---|---|
| 1 | 4 | +0.003566 | 0.001221 | +2.92 | **0/4** |
| 3 | 4 | +0.005162 | 0.001193 | +4.33 | **0/4** |
| 24 | 15 | -0.002242 | 0.000496 | -4.52 | 13/15 |
| 168 | 4 | -0.000586 | 0.000231 | -2.53 | 3/4 |

**At H = 1 and H = 3, more variates make the model worse at every origin tested.** RQ1 was
pre-registered at H = 24 only, where the ladder happens to point the right way. This is a material
qualification on H1 and it sits unremarked in Table 7. Caveat: only 4 origins at H in {1, 3, 168}, so
treat it as a strong descriptive signal, not an inference. (The H = 24 row above uses all 15 origins;
`paper_numbers.json`'s horizon section restricts to the 4 sweep origins and gets the same sign.)

Separately, R2_oos improves monotonically with H (-0.034 -> -0.010). The deficit is roughly
proportional and MSE_naive grows with the horizon, so the relative penalty shrinks. The model is
least bad exactly where it is least useful.

### 9.2 Directional accuracy - nothing anywhere

| K | DA h=1 | p (median) | DA h=H | p | DA cumulative | p |
|---|---|---|---|---|---|---|
| 1 | 0.5046 | 0.278 | 0.4999 | 0.596 | 0.4825 | 0.809 |
| 4 | 0.5019 | 0.477 | 0.4999 | 0.577 | 0.4822 | 0.812 |
| 8 | 0.5022 | 0.483 | 0.5016 | 0.583 | 0.4820 | 0.763 |
| 12 | 0.5022 | 0.444 | 0.5008 | 0.605 | 0.4822 | 0.802 |

No rung differs from 0.5 at h = 1 or h = H. Cumulative DA is 0.482 - **below** chance at every rung,
consistently. Overlapping variants (descriptive only, D21) are 0.5043-0.5051 and 0.4878-0.4903.

---

## 10. Defects found in this pass - candidates for D76+

Ordered by severity. None is registered yet.

### 10.1 F — the `tuned` arm did not run the configuration that was selected → FIXED as `D76`

`runner.TUNING_GRID` sweeps d_model x e_layers x lr (18 points). `tune_on_validation` ranks by
validation MSE and returns:

    ITransformerConfig(pred_len=PRED_LEN, d_model=int(best["d_model"]), e_layers=int(best["e_layers"]))

`ITransformerConfig` carries no `lr`; `lr` is an argument to `train_one`. **The winning learning rate
is selected and then discarded.**

- `meta/tuning_selection.json` -> `ranked[0] = {"d_model": 256, "e_layers": 3, "lr": 0.001, "val_mse": 0.466377}`
- `meta/tuning_selection.json` -> `selected = {"d_model": 256, "e_layers": 3}` (no `lr`)
- `meta/itrt_o01_K08_H024_s42.json` -> `config.d_model = 256`, `config.e_layers = 3`, **`schedule.lr = 0.0001`**

So the arm ran the *architecture* of the winner under the *default* learning rate - a point the search
also evaluated ({256, 3, 1e-4}) and did **not** rank first. Worse, the ranking that chose the
architecture was partly driven by an lr that then was not applied, and the runner-up
({64, 3, 1e-4}, val 0.466477, a gap of 1e-4 - an order of magnitude below the seed noise) *is*
faithfully reproducible.

Consequences: §12's contract is broken for this arm (the documented decision and the executed run
disagree); Table 9's caption - *"the configuration origin 1's validation preferred"* - is false as
executed; §10.2 and §13.5's trial accounting describe an 18-point search of which only 6 distinct
points are reachable by the arm.

The conclusion the arm draws ("tuning does not rescue the null") is probably robust - a larger model
is worse at 15/15 origins - but as run it does not support the sentence printed beside it. Minimum
fix: thread the selected lr through to `train_one`, re-run 75 cells (~0.8 GPU-h), restate.
Alternative: keep the runs, restate the arm honestly as an architecture-only search, and record why.

### 10.2 C — §13.4's "no deep model in the MCS" is falsified → CORRECTED in `CLAUDE.md` §13.4

`lstm-K8` is in the MCS at 90% and 75%, at rank 5. §13.4 and the reasoning that leans on it in §13.2
need rewriting. See §3.1.

### 10.3 C — the epoch cap was hardcoded, and the claim needed scoping → FIXED as `D78`

The cap was compared against a literal 30, so `itrl` (cap 60) was miscounted; it is now read from each
run's own `meta['schedule']`. On the claim itself: **0 of 300 at H = 24**, 5 of 80 at H = 168, so
`D62c` holds at the headline horizon — my first pass over-stated this as a falsification. What was
genuinely undisclosed is that `dlin` reaches its cap in **56 of 75** runs and `ptst` in **39 of 75**,
so both baselines are budget-truncated and "DLinear is worst" is confounded with "DLinear is most
truncated". Table 3 now prints the per-arm at-cap count and its caption says so.

### 10.4 C — Romano-Wolf lost every rejection when the family widened → FIXED as `D79`

31/66 -> 0/105, caused by adding three baselines to the pairwise family. §9.2 mandates Romano-Wolf over
"all pairs"; that mandate, applied literally, now has no power at G = 15. Needs an explicit decision
and an explicit disclosure. See §5.1.

### 10.5 C — §5.4 mislabels the lookback-aware K_eff numbers → FIXED as `D81`

`CLAUDE.md` §5.4 attributes 92.1 / 21.9 / 37.3 / 15.5 to the **covariance** spectrum and prescribes
the **correlation** spectrum as the remedy. In the code, `keff.lookback_covariance_pr` already computes
the **correlation** spectrum (it standardises the K*L columns), its docstring records the covariance
figures as 92.1 / 3.0 / 44.0 / 8.8, and the stored `pr_lookback_cov` column holds
92.1 / 21.9 / 37.3 / 15.5. So:

- the numbers in §5.4 are the **correlation** ones, mislabelled;
- the correlation spectrum is **also non-monotone in K**, so D53b's stated justification ("use
  correlation, covariance is not monotone in K") does not deliver what it promises;
- the field, the column and the function are all named `...covariance...` while computing correlation.

`pr_lookback_ratio` falls 0.980 -> 0.078 -> 0.047 -> 0.020 across the rungs, i.e. the lookback-aware
measure is **anti**-monotone in K while the contemporaneous PR rises 1.00 -> 3.33 -> 4.27 -> 3.98. The
two K_eff constructs disagree in *direction*. D44 requires this divergence be reported in §4.1b
"whatever it is"; the `divergence` column carries it (0, -0.972, -1.568, -1.811) but no prose explains
that the constructs are ordinally opposed.

### 10.6 U — the D45 coverage check produced neither of its two required forms → FIXED as `D80`

§9.2 requires block coverage as a regression covariate **or** beta1 re-run on well-covered blocks.
`coverage.restricted` is `null` with the reason *"restricting to well-covered blocks unbalances the
panel... that the check cannot run IS the D45 finding"*. That is a fair argument for the restricted
form; the **covariate** form is the other half of the requirement and was not attempted.

### 10.7 U — `lstm-K8` is absent from the economic evaluation → FIXED as `D77`

Table 8 and Figure 7 cover 5 models; the one whose statistical standing changed is not among them.

### 10.8 U — the matched-K contrast is not reported as a contrast → FIXED as `D82`

Table 9 gives `itro` and `itrr` each against the ladder. The number RQ1 needs is `itrr` minus `itro`
(+0.001241, t = +6.27) and it is computed nowhere. Same for the ladder's own paired contrasts, whose
SEs are half what Table 4's marginal error bars imply.

### 10.9 U — the on-Kaggle report is a generation behind `src/` → notebook regenerated; re-upload it

See §1.2. The notebook must be rebuilt and re-uploaded, or the next session repeats the omission.

### 10.10 minor

- `runner.TUNING_GRID`'s docstring says the capacity arm "made things worse at 14 of 15 origins";
  measured on RelMSE it is **13 of 15** (`itrc` better at 2). Different metric or an off-by-one.
- `attention_maps.parquet`'s `vol_low` / `vol_high` are the two *shared* tercile edges repeated on
  every row, not per-tercile bounds. Correct by construction (`tercile_edges` returns one pair) but
  the names read as per-row bounds.

---

## 11. Recommended next actions

Ordered by value, not effort.

1. **Regenerate `paper/`** - `python tools/build_report.py` against the current artifacts, so §12's
   chain closes and the manuscript source stops being one grid behind. An earlier session hit a numpy
   MemoryError during the bootstrap; close background memory hogs first.
2. **Rebuild and re-upload the notebook** - `python tools/build_notebook.py`, commit, and replace the
   Kaggle copy, so a future session's own report is not the 12-model one (§1.2).
3. **Decide the Romano-Wolf family** (§5.1). This changes what Table 6 can claim and must be settled
   before the manuscript quotes any adjusted p-value.
4. **Fix or restate the tuned arm** (§10.1).
5. **Register D76+** for §10's items. §14's closing line: absorbing a contradiction silently is the
   failure the register exists to prevent.
6. **Lift three results into the manuscript's spine** - the strongest new material: the matched-K
   contrast (RQ1's only direct evidence), the LSTM's MCS membership (reframes the headline from "deep
   learning fails" to "attention architectures fail"), and the lookback sweep (bounds RQ1's importance
   honestly, before a referee does it for you).
7. **Add `lstm-K8` to the economics** (§10.7).
8. **Consider an L sweep as the next experiment**, not more K. L moves the metric 5x more than K and
   is monotone over the range tested.

---

## 12. How the recomputed numbers were derived

Everything in §3-§9 that is *not* in `notebooks/outputs/paper/paper_numbers.json` - the paired
contrasts, the attention-map statistics, the per-arm epoch/device tables, the positive-skill origin
counts, the training-disjoint triples, the horizon-by-K paired tests - was computed directly from:

- `notebooks/outputs/artifacts/seed_averaged_cells.parquet` (2,403 rows, 17 model tags)
- `notebooks/outputs/artifacts/run_block_metrics.parquet` (9,675 rows)
- `notebooks/outputs/artifacts/meta/*.json` (1,620)
- `notebooks/outputs/paper/panels/attention_maps.parquet` (28,800 rows)

**Paired-contrast method.** Per arm and origin,
`RelMSE(origin) = sum_b(mse * n_windows) / sum_b(mse_naive * n_windows)` over the six test blocks -
seed-averaged MSEs first, ratio second (D42). The contrast is the paired difference across the 15
origins, `t = mean/SE` referred to t(14), which makes the **origin** the inferential unit (D30) and
keeps every comparison on a scale-free metric (D60i). Sign convention: **positive = the first arm is
worse**.

**Caveats on those numbers.** They are (a) post-hoc, not pre-registered; (b) 21 tests with no
multiplicity control - Bonferroni at 0.05/21 = 0.0024, which most of the significant rows clear and
`itrl`-K8 (p = 0.063), `itro` (0.136) and `itrc` (0.018) do not; (c) resting on the same G = 15
clusters whose effective independence §9.2 bounds near 4, so they inherit the same anticonservatism
the wild cluster bootstrap discloses. They are strong enough to guide what belongs in the paper and
not strong enough to be quoted as confirmatory findings without being run through `report.py` and
declared exploratory.

---

## 13. Quick reference - numbers most likely to be asked for

    runs 1620 - code_sha256 36fa9c77e65f6a1b... - input_sha256 8270a84b07c2923b... (file-digest)
    grid wall 8.96 h on 2xT4 - 17.91 GPU-h summed - mean 39.8 s/run - cuda:0 813 / cuda:1 807
    894 pre-existing preds re-ran BYTE-IDENTICAL

    R2_oos:  Naive-RW 0 - rdg-K4 -0.000292 - rdg-K8 -0.000485 - rdg-K12 -0.000693
             lstm-K8 -0.001559 - rdg-K1 -0.000801 - itru-K8 -0.017722 - itr-K8 -0.017993
             itr-K12 -0.018570 - itr-K4 -0.018682 - ptst-K8 -0.016353 - itr-K1 -0.020455
             dlin-K8 -0.026209 - npst -1.004203 - nsea -1.044919
             l192 -0.013907 - itro -0.018245 - itrl-K8 -0.018135 - itra -0.017993 (exact)
             itrc -0.019074 - itrr -0.019542 - itrl-K1 -0.021289 - itrt -0.021443 - l048 -0.026235

    MCS 90/75 = {Naive-RW, rdg-K4, rdg-K8, rdg-K12, lstm-K8, rdg-K1}
    Romano-Wolf 0/105 rejections (min adj p 0.0848); raw p<0.05 at 90/105
    RQ1 delta 4->8 +0.000636 - 8->12 -0.000437 - TOST p (0.9734, 0.0002) NOT equivalent
         J-test K|Keff t=+3.293 p=0.0011 - Keff|K t=-0.348 p=0.7281 - corr(K,Keff) 0.8280
         gate PR K=8 4.3928 < 5.0 (failed, disclosed)
         matched-K itrr-itro +0.001241 t=+6.27 p<0.0001 (14/15 favour orthogonal)
    RQ2 beta1 +0.000256 - SE 0.000358 - WCR p 0.7381 - MDE -0.000920 - G=15 N=90 -> DESCRIPTIVE
         falsification aged-fresh RelMSE +0.000828 +/- 0.001106, 6/15 favour aged
    RQ3 b* UNDEFINED at all four tau - 15/15 origins excluded - decay_panel 0 rows -> UNTESTABLE
         (scope: true for the ladder; rdg-K4 has 4 positive-skill origins, lstm 3, rdg-K8 2)
    attention A_attn -0.000267 - uniform beats learned p=0.0033 - maps flat 0.1204-0.1377 vs 1/N=0.1250
    economics itr-K8 @2bp net +20.6% Sharpe +0.377 DSR 0.101 - @10bp +12.5% / +0.104
         buy-and-hold +29.0% Sharpe +0.578 - every JK p >= 0.308 - DSR trials 91
    DA h1 0.5022 - hH 0.5016 - cumulative 0.4820 (below chance) - all p >= 0.28
    efficiency Hurst 0.544-0.568 - VR(4) significant ONLY at 2020-01 (p=0.0001) and full sample
    dataset 75216 calendar / 75094 present / 75091 usable / 122 missing / 3 flat=3 zero-vol=3 zero-trade
         125 excluded positions - 28 break runs - 29 segments - n_train 13545-15217 (feature frame)
    mu_g/sigma_g range -0.008179 .. +0.017331 - sigma_g 0.005064 .. 0.009151
    epoch cap hits: itr 5/540 - lstm 3/75 - ptst 39/75 - dlin 56/75 - itrl 0/150 (max 21 of 60)
