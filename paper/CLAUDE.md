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

- **Pre-registration is binding, and the prose must sound like it.** τ = 5% headline, RQ2 on K=1 vs
  K=8, the ladder gate at PR < 5.0. If a result prompts a change, the change is a *new experiment*
  and is labelled one in the text — not folded into the design as though it had always been there.
- **Report the null honestly, and in the right words.** "No decay detected within 180 days" is a
  right-censored finding, not a failure. A flat 8→12 rung is the designed contrast, not a null — say
  so *where it appears* in Results, not only once in the methodology, because a reader meeting the
  flat rung in Table 4 will not remember §3.
- **Write the interval, not the point.** "Retraining cadence is 90 days" with no confidence interval
  is a reading of a noisy curve, not a result — and the abstract is required to carry that number.
- **Hedge the priority claim and publish its search protocol.** "To the best of our knowledge, the
  first …", with databases, query strings, search date and hits screened. A bare priority claim is
  the easiest thing in the paper for a referee to refute with a single citation.
- **Do not claim the market is efficient.** State that the evidence is mixed and time-varying, then
  report your own VR and Hurst numbers for this period and this granularity.
- **Attention is not explanation.** Attention maps are descriptive evidence of variate reliance,
  validated for seed stability. The Jain & Wallace / Wiegreffe & Pinter debate is scoped to RNN-era
  NLP and its transfer to variate-level attention in LTSF is genuinely open — that openness is itself
  a limitation sentence, not a gap to paper over.

## Length

10–14 pages, IMRaD, 35–45 references, IEEE style, Sinta target. Aim ≥ 60% of references from the last
five years; methodological classics (Diebold–Mariano 1995, Lo–MacKinlay 1988, Newey–West 1987,
Pesaran–Timmermann 1992, Rubin 1976) are exempt.

The abstract must contain concrete numbers — β₁, the percentage decay, the recommended cadence with
its interval. An abstract without numbers reads as a proposal, not a result.
