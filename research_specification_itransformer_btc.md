# Research specification — current authority

The earlier source proposal is superseded by [CLAUDE.md](CLAUDE.md) and the primary [iTransformer notebook](notebooks/iTransformer.ipynb). Its complete pre-audit text remains in git at `f4cf9479873a0bf7c0e6fc4fe7afd026a315d106`. It must not override corrections to the experiment or be mistaken for preregistration evidence.

The study evaluates hourly BTCUSDT spot log returns with an iTransformer variate ladder. RQ1 describes differences associated with K and training PR; RQ2 describes the K1-versus-K8 gap over time since selection; RQ3 describes threshold crossings relative to positive first-block skill. None identifies a causal PR-only effect or an optimal retraining policy.

After the September audit, the protocol adds controlled training counts, information-preserving representations, validation-refresh and five-seed fresh arms, objective and optimization sensitivities, corrected architectures and target timing. These are new exploratory experiments, not results already observed. Existing predictions retain their original model/code vintage.

Use the maintained documents below instead of a second copy of the protocol:

| Need | Document |
|---|---|
| Governing constraints and interpretation | [CLAUDE.md](CLAUDE.md) |
| Notebook operations, 2,130-run manifest and Kaggle continuation | [USAGE.md](USAGE.md) |
| A01–A15 changes, evidence and remaining empirical requirements | [Remediation report](docs/AUDIT_REMEDIATION_2026-09-09.md) |
| Raw, candidate and selected window counts | [Window budget](docs/ORIGIN_WINDOW_BUDGET.md) |
| Completed historical reanalysis draft | [manuscript.tex](paper/manuscript.tex) |
| Scope of source reading and novelty search | [Literature log](docs/LITERATURE_SCOPE_2026-09-10.md) |

No-imputation is a population/design choice; it does not prove why bars are absent. Polars supports centered rolling operations; this study avoids future information through its per-bar features and tested chronology. Negative mean out-of-sample skill only describes the evaluated procedure/sample. ARIMA is out of scope, with no claim that an unperformed AIC search would select a particular order.
