# iTransformer: hourly Bitcoin walk-forward research

**Start with [notebooks/iTransformer.ipynb](notebooks/iTransformer.ipynb).** It contains the implementation and the complete Kaggle workflow. Edit that notebook; generate `src/` only through its final, locally enabled export cell. The old `build_notebook.py` template cannot overwrite notebook edits.

This study evaluates BTCUSDT spot hourly log returns, using 15 rolling origins, a K ∈ {1,4,8,12} feature ladder and six 30-day test blocks. It examines feature-set effects and changes with model selection age. Threshold crossings are descriptive; they do not identify an optimal retraining policy.

| Current state | Meaning |
|---|---|
| Primary implementation | Self-contained notebook; no repository package required on Kaggle |
| Revised grid | **2,130 runs**, including equal training counts, five fresh seeds, validation-refresh and invertible-representation controls, and baseline objective sensitivities |
| Training | Kaggle **2 × T4**, one independent run per GPU; the revised grid has **not been trained during remediation** |
| Session protection | Manual remaining-quota entry, deadline including tuning, 45-minute saving reserve, epoch checkpoints, cached validation and a consolidated output bundle |
| Historical evidence | Original **1,620 runs** in `notebooks/outputs/artifacts/`, preserved and incompatible with current-code execution resume |
| Corrected historical analysis | [paper/reanalysis_2026-09-09/](paper/reanalysis_2026-09-09/), using the same old predictions with corrected target calendars and estimands |
| Data | `data/raw/BTCUSDT_1h.parquet`; observed missing bars have no verified causal attribution |

Read [USAGE.md](USAGE.md) before running on Kaggle. Enter the current quota, attach the data, select T4 x2, and attach saved output in subsequent sessions. A complete grid can be analysed on Kaggle CPU to save GPU quota. Several sessions or quota weeks may be needed; earlier runtime measurements do not establish the revised grid's cost.

[The remediation report](docs/AUDIT_REMEDIATION_2026-09-09.md) maps A01–A15 to changes, checks and remaining experimental evidence. [The original audit](docs/RESEARCH_WORKFLOW_AUDIT_2026-09-09.md) remains historical evidence. This is research-validity remediation, not a guarantee of zero software or dependency vulnerabilities.

Project rules: [CLAUDE.md](CLAUDE.md). Navigation: [notebook map](docs/NOTEBOOK_MAP.md), [window accounting](docs/ORIGIN_WINDOW_BUDGET.md), [divergence register](docs/DIVERGENCE_REGISTER.md). Older research specifications are historical inputs, not current authority.
