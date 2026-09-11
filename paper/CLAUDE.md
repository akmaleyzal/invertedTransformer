# paper/ — writing posture

Root `CLAUDE.md` is the project law and this file does not restate it. §13 fixes the structure, the
mandatory disclosures and the table/figure inventory — including Figure 3's one-series form (`D36`).
§9 fixes every metric and which dispersion measure it carries (`D30`). §12 fixes what may enter the
manuscript at all. §13.3 fixes citation discipline. Read those there.

**This file is the only surviving directory-local `CLAUDE.md`, and it survives on one property**: a
subdirectory `CLAUDE.md` loads only when a file in that subtree is touched, so a rule living here is
absent whenever the agent is not writing prose. That makes it the wrong place for anything
catastrophic — and the right place for posture, where a missed rule costs a weaker paragraph rather
than a corrupted result. Root §15 states the test for adding another.

What follows is only what governs the *act of writing*, and is therefore nowhere in root.

**Revised after the September 2026 audit (`D90`).** Four posture rules below used to instruct prose
this study can no longer defend — a binding pre-registration, a right-censored decay phrase, a
recommended cadence, and a hedged priority claim. They are replaced, not softened. Root §1 lists
every withdrawn claim; `docs/AUDIT_REMEDIATION_2026-09-09.md` maps each to its finding.

## Three refusals

1. **Cut anything you have not read.** Both source `.md` files self-declare their reference lists as
   assembled from memory, and two entries are already known wrong (`D16`). An examiner may ask what a
   cited paper says. A citation you cannot summarise is a citation you delete.
2. **Critique practice, not people.** The methodological gap in the local literature — price-level
   targets, single chronological splits, no naive baseline, MAPE on price, gaps unacknowledged — is
   stated as a pattern in common practice, never attributed to named authors.
3. **Never a bare number.** Every metric in the prose carries its baseline, its split, its seed count
   and its cost assumptions. Root §9.2 fixes *which* dispersion measure; this fixes that one always
   appears.

## Posture

- **Say "declared before the grid ran", never "pre-registered" (`A06`).** τ = 5% headline, RQ2 on
  K=1 vs K=8, the ladder gate at PR < 5.0 were all fixed before any model trained, and the prose may
  say exactly that. It may not say *pre-registered*: that word claims a dated deposit a reader can
  check, and none exists. Everything changed after the audit is written as a **new exploratory
  protocol** — labelled one in the text, not folded into the design as though it had always been
  there. The MDE and the equivalence margin carry the word **post-analysis** wherever they appear,
  because both are computed from effects this study had already observed.
- **Report the null in the right words, and there are three of them (`A07`).** A decay result is
  **undefined** when the block-1 reference skill is not positive, **not crossed within six blocks**
  when the reference is positive and the threshold is never reached, and **crossed at block b**
  otherwise. The first two are different findings and the prose must not merge them. *"No decay
  detected within 180 days"* is **forbidden in this manuscript** — root §3 licenses that phrasing for
  the second case, and the second case did not occur: all fifteen origins are undefined, so writing
  it here asserts an edge the data does not contain. A flat 8→12 rung is the designed contrast, not a
  null — say so *where it appears* in Results, not only once in the methodology, because a reader
  meeting the flat rung in Table 4 will not remember §3.
- **Write the interval where an estimand exists, and no number where none does (`A07`, `A11`).**
  `b*` is a discrete description of six 30-day blocks and carries its interval; it is **not** a
  retraining recommendation, and "retraining cadence is 90 days" is not a sentence this paper
  contains in any form. Same rule in the economics: Sharpe, Sortino, MDD and net P&L are reported
  with an interval **only** where the interval's assumptions hold on this sample. An unsupported
  interval printed beside a number reads as inference and is worse than the bare number labelled
  honestly.
- **Claim no priority (`A15`).** Not "the first", and not "to the best of our knowledge, the first"
  either — the hedge went out with the claim. `docs/LITERATURE_SCOPE_2026-09-10.md` is a **bounded
  reading log**, names itself one, and is not a systematic search; primary work on iTransformer, on
  hourly BTC and on walk-forward evaluation already exists. The contribution sentences come from
  root §3, which no longer asserts a first.
- **Bound every negative result to what was tested (`A15`).** "No out-of-sample skill" written flat
  is a claim about Bitcoin. Written correctly it names the models, the target, the horizon, the
  preprocessing, the sample and the aggregation — and notes that individual origins can carry
  positive skill while the mean is negative.
- **Do not claim the market is efficient.** State that the evidence is mixed and time-varying, then
  report your own VR and Hurst numbers for this period and this granularity.
- **Attention is not explanation.** Attention maps are descriptive evidence of variate reliance,
  validated for seed stability. The Jain & Wallace / Wiegreffe & Pinter debate is scoped to RNN-era
  NLP and its transfer to variate-level attention in LTSF is genuinely open — that openness is itself
  a limitation sentence, not a gap to paper over.
- **Name which `paper_numbers.json` a number came from (`D90`, root §12).** Three now coexist: the
  historical grid's, the pre-audit `paper/paper_numbers.json`, and
  `paper/reanalysis_2026-09-09/paper_numbers.json`. The prose reads the third, through
  `reanalysis_2026-09-09/tables/manuscript_numbers.tex`. Which file a figure came from is part of the
  figure, and two of them must never share a table.
- **Never let a sentence blur what was observed with what was only specified (`D90`).** The 1,620
  historical runs, recomputed under corrected evaluation, are evidence. The 2,130-run protocol is
  declared, implemented and **untrained** — it has produced no ranking, no decay result and no
  economic figure. Any sentence that lets a reader think otherwise is the exact failure root §3's
  third contribution exists to prevent, and it is the easiest one in this paper to commit by accident:
  the two sit in adjacent paragraphs throughout Methodology.

## Length

10–14 pages, IMRaD, 35–45 references, IEEE style, Sinta target. Aim ≥ 60% of references from the last
five years; methodological classics (Diebold–Mariano 1995, Lo–MacKinlay 1988, Newey–West 1987,
Pesaran–Timmermann 1992, Rubin 1976) are exempt.

The abstract must contain concrete numbers, and they are the numbers that exist: mean `R²_oos` per
model against Naive-RW, the standard error **across origins**, the seed count, and the origin and
block counts the average is taken over. Not β₁ as a headline — root §9.2 reports it as descriptive
with its post-analysis MDE beside it — and **not a recommended cadence**, which is not a quantity
this study estimates. An abstract without numbers reads as a proposal; an abstract carrying a number
the paper cannot produce is worse.
