"""Update non-governing documents; CLAUDE.md is deliberately left until last."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
budget = (ROOT / '.research/audit-repair-work/raw_window_budget.md').read_text(encoding='utf-8')
(ROOT / 'docs/ORIGIN_WINDOW_BUDGET.md').write_text('''# Origin window budget

Updated after A01/A12, 10 September 2026. The notebook's splitting/budget cells are primary; the corresponding generated modules are `splits.py` and `budget.py`. Input SHA-256 is `8270a84b07c2923bc885782a8ba4e1898133d18ee3b260f157fcee3fd6923b4e`.

## Three counts with different meanings

1. Raw-bar window counts describe the data artifact, before feature formation.
2. Feature-frame counts describe candidates actually available to training. Return construction removes the first row of each segment, so these can be smaller.
3. The new runner selects **11,500** candidates without replacement, training-only seed **1729**, at every origin and arm. Available counts are diagnostic; selected counts are the training exposure control. Scaler fitting uses all purged training rows before sampling.

For the entire new 2,130-run manifest, 72 unique training span/L/H combinations have **11,689–15,265** candidates. All support the fixed 11,500 count. This is a timestamp check, not a training result. [Machine evidence](../.research/audit-repair-work/window_budget.json) includes every span and the selected timestamp hash. Run metadata must still confirm the achieved count on Kaggle.

The historical main L96/H24 feature-frame counts were 13,545–15,217 and were **not subsampled**. Earlier instructions to use 13,558, 13,545 or 13,520 as the global floor are superseded: the new control also covers long-lookback and long-horizon arms.

## Raw-bar accounting, L=96 and H=24

Training inputs and targets lie wholly inside each training span. Counts are segment-wise, `sum(max(0, segment_length − L − H + 1))`; a short segment contributes zero, never a negative count. Breaks include missing bars and unusable bars. Their cause is not inferred from the REST artifact.

Test-block starts now mean the **first target bar**, with lookback permitted before the block boundary. A clean block has 720 possible hourly issuance times. A gap anywhere in its input or complete horizon excludes the window. These are raw-frame counts, not the smaller common-calendar intersections used for cross-model evaluation.

''' + budget + '''
The three zero-volume, zero-trade and flat bars are the same observed bars. Count their union, not their sum. The raw training-count regression target remains `COMMITTED_TRAIN_BUDGET`; the table's changed test counts follow the corrected first-target timestamp convention.

## Interpretation

Fixed calendar duration does not fix sample size. Fixed sample size does not equalize all optimizers' update counts or eliminate changes in market conditions. Missing outcomes are not imputed; the population is surviving continuous windows. Coverage is descriptive and may be a covariate, but it cannot recover unavailable outcomes or prove missingness is ignorable. No assertion that missing bars are exchange downtime, nontrading, or stress-related is made without separate evidence.

## Reproduce

```powershell
.venv/Scripts/python.exe -X utf8 .research/audit-repair-work/check_window_budget.py
.venv/Scripts/python.exe -m pytest tests/test_data_plane.py tests/test_audit_repairs.py -q
```

For session configuration and continuation, use [USAGE](../USAGE.md). For each audit disposition, use [the remediation report](AUDIT_REMEDIATION_2026-09-09.md).
''', encoding='utf-8')
(ROOT / 'research_specification_itransformer_btc.md').write_text('''# Research specification — current authority

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
''', encoding='utf-8')
path = ROOT / 'notebooks/outputs/RUN_ANALYSIS.md'
old = path.read_text(encoding='utf-8')
path.write_text('''> **Historical analysis only.** The content below describes earlier code/results and is preserved as evidence. It is superseded for current interpretation by [A01–A15 remediation](../../docs/AUDIT_REMEDIATION_2026-09-09.md) and the separate [corrected historical report](../../paper/reanalysis_2026-09-09/paper_numbers.json). Use [USAGE](../../USAGE.md) for the new Kaggle protocol. Do not treat old preregistration, nested-test, decay, gap-cause or architecture claims below as current project law.

''' + old, encoding='utf-8')
path = ROOT / 'reference_library_itransformer_btc.md'
old = path.read_text(encoding='utf-8')
path.write_text('''> **Historical source list, not citation authority.** Use [the maintained bibliography](paper/references/references.bib) and [the reading-scope log](docs/LITERATURE_SCOPE_2026-09-10.md). A remembered reference, resolved identifier, abstract screening and full-paper reading are different evidence levels. Current novelty or methodological claims must not be inferred from this historical list.

''' + old, encoding='utf-8')
path = ROOT / 'docs/DIVERGENCE_REGISTER.md'
old = path.read_text(encoding='utf-8').replace('New contradictions found later take IDs **D90+**.', '')
path.write_text(old + '''
## D90 — Research workflow A01–A15 remediation (2026-09-09/10)

The [September audit](RESEARCH_WORKFLOW_AUDIT_2026-09-09.md) found that passing implementation tests and a single prediction vintage did not validate target timing, aggregation, inference or interpretation. Historical statements above are evidence of prior decisions, not instructions to restore superseded behavior.

The primary implementation is now **notebooks/iTransformer.ipynb**. Direct notebook edits are preserved; `src/` is generated only by its last sync cell. The template writer is disabled. Current law is CLAUDE.md; source specifications point to it rather than maintaining a competing protocol.

[The detailed remediation ledger](AUDIT_REMEDIATION_2026-09-09.md) maps each A01–A15 to implementation, checks and empirical requirements. It covers first-target timing/common calendars, seed-loss/block aggregation, clustered exploratory diagnostics, rejection of automatic CW nesting, post-test MDE labeling, block-1 decay, refresh and representation controls, official-code forward parity, baseline objectives/budgets, raw direction and conditional long/cash accounting, fixed training counts, and strict resumability with Kaggle session limits.

The old 1,620 runs remain historical evidence. Corrected reanalysis lives separately under paper/reanalysis_2026-09-09. The 2,130-run revised protocol requires new Kaggle training and is not complete merely because its code and tests exist. No independent preregistration, causal PR-only identification, optimal cadence, exchange-gap cause, universal absence of predictability or executable-backtest result is asserted.

Machine evidence is kept in .research/audit-repair-work; migration scripts are already applied and must not be replayed. TEMP/TMP use D: on this Windows workspace. The user requested that CLAUDE.md be updated last, after implementation and verification.

New contradictions found later take IDs **D91+**.
''', encoding='utf-8')
print('Updated budget, specification pointers, historical notices and D90; CLAUDE untouched')
