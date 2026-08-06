# paper/ — manuscript rules

Root `CLAUDE.md` is the project law; §13 defines the structure, the mandatory disclosures, and the
table/figure inventory. This file adds only what is specific to writing.

## Three rules that override convenience

1. **Every number cites its artifact.** A figure that is not regenerable from
   `artifacts/paper_numbers.json` — which is itself built from `preds/{run_id}.parquet` and
   `meta/{run_id}.json` — does not go in. Tables and figures are **generated**, never transcribed.
   Numbers produced under different input-artifact hashes are not comparable and must not share a
   table. Root §12.
2. **Every citation cites a verified DOI, and you have read the source.** Both source `.md` files
   self-declare entries assembled from memory, and two are already known wrong (`D16`). An examiner
   may ask what a cited paper says.
3. **Every deviation is disclosed, not smoothed.** `docs/DIVERGENCE_REGISTER.md` names the section
   each of D01–D22 must appear in. A deviation that improves the design is still a deviation.

## Writing posture

- **Pre-registration is binding.** τ = 5% headline, RQ2 on K=1 vs K=8, the ladder gate at PR < 5.0.
  If a result prompts a change, the change is a *new experiment* and must be labelled one.
- **Report the null honestly.** "No decay detected within 180 days" is a right-censored finding, not
  a failure. A flat 8→12 rung is the designed contrast, not a null — say so where it appears, not
  only in the methodology.
- **Never a bare number.** Every metric carries: the baseline, the split, the seed count, and the
  cost assumptions. **The dispersion measure is bound to the aggregation level** (`D30`): a per-cell
  (origin, block) number reports `mean ± std` across seeds with n stated; **any number aggregated
  across origins — Table 4 included — reports ± standard error across origins**, with seed std shown
  separately as a Monte-Carlo diagnostic. The inferential unit is the origin. Seed dispersion is a
  diagnostic, never the uncertainty on an aggregated estimate — reporting it as one understates the
  headline uncertainty by roughly an order of magnitude.
- **`b*` and every economic statistic carry an interval, not a point** (`D41`, `D46`). "Retraining
  cadence is 90 days" with no confidence interval is a reading of a noisy curve, not a result — and
  the abstract is required to contain that number.
- **Priority claims are hedged and their search protocol is published** (§13.2). "To the best of our
  knowledge, the first …", with databases, query strings, search date and hits screened in §2. A bare
  priority claim is the easiest thing in the paper for a referee to refute with one citation.
- **Critique practice, not people.** The methodological gap in the local literature — price-level
  targets, single chronological splits, no naive baseline, MAPE on price, gaps unacknowledged — is
  stated as a pattern in common practice, never attributed to named authors.
- **Do not claim the market is efficient.** State that the evidence is mixed and time-varying, then
  report your own VR and Hurst results for this period and granularity.
- Attention maps are **descriptive evidence of variate reliance**, validated for seed stability,
  never causal explanation. Root §13.2.

## Length and reference discipline

10–14 pages, IMRaD, 35–45 references, IEEE style, Sinta target. Aim ≥ 60% of references from the
last five years; methodological classics (Diebold–Mariano 1995, Lo–MacKinlay 1988, Newey–West 1987,
Pesaran–Timmermann 1992, Rubin 1976) are exempt. **Cut anything you have not read.**

The abstract must contain concrete numbers — β₁, the percentage decay, the recommended cadence. An
abstract without numbers reads as a proposal, not a result.

## The one figure that matters

Figure 3, the decay curve `A(b)` vs `b` for K = 1, 4, 8, 12. It carries the paper. If only one
figure could appear in a graphical abstract, it is that one.
