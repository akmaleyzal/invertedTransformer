# Bounded literature and source check — 10 September 2026

Purpose: correct unsupported novelty and implementation claims after A01–A15. This is an English-language targeted web/source check, not a systematic review. Search results are discovery leads; no hit total, exhaustive coverage, or full-paper reading is inferred from a snippet.

Queries submitted through web search:

1. `"iTransformer" "Bitcoin" "walk-forward"`
2. `"iTransformer" "effective dimensionality" forecasting`
3. `"Bitcoin" "forecasting" "model decay"`
4. `"iTransformer" cryptocurrency "log returns"`

| Source opened | Extent used here | Supported statement |
|---|---|---|
| [Bysik and Ślepaczuk, arXiv:2606.00060](https://arxiv.org/abs/2606.00060) | Author/title/abstract page; not a full-paper methodological review | Hourly BTC-USDT, walk-forward evaluation and iTransformer already appear together in prior work. No broad priority claim is justified |
| [Liu et al., iTransformer](https://arxiv.org/abs/2310.06625) | Abstract and inspected official implementation definitions; pinned forward checks | Variate tokenization and cross-variate attention motivate the tested model; no inference about Bitcoin performance follows from that design |
| [Nie et al., PatchTST](https://arxiv.org/abs/2211.14730) | Abstract and inspected official backbone/layers/RevIN definitions; pinned forward checks | Patching and shared channel-independent modelling; the local target/all objectives and adaptations must be disclosed |
| [Zeng et al., LTSF-Linear](https://arxiv.org/abs/2205.13504) | Abstract page for the limited background statement | Simple linear forecasting models are a relevant comparator; their published benchmarks do not establish this study's ranking |

Other search leads included MSPCIFormer on the Wiley publisher site, financial-return forecasting studies on arXiv, and an institutional thesis. They were not relied on for substantive comparisons here. Absence from this small table is not evidence that a topic has no prior work.

Implementation revisions and downloaded-source SHA-256 values are in [upstream_parity.json](../.research/audit-repair-work/upstream_parity.json). The parity script loads only the named inspected definitions; cached reference files stay under ignored D: temp. GPU or training equivalence is not claimed.

The manuscript cites these sources only for the scoped statements above. Bibliographic metadata verification, abstract screening, relevant-section reading and full-paper reading are distinct. Existing bibliography annotations are not upgraded wholesale. No assertion that earlier work lacks a particular metric, design or preregistration is made without checking the relevant full text.

Conclusion for this draft: describe the actual evaluation and reproducibility contribution. Do not use “first,” “to the best of our knowledge, first,” or “pre-registered” as a substitute for evidence. No external dated registration was supplied for the study.
