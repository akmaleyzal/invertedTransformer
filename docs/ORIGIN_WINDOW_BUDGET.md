# Origin window budget

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

| # | Origin | Training sub-block | Breaks | Excluded | Windows kept | Loss | Test-block starts B1…B6 |
|---:|---|---|---:|---:|---:|---:|---|
|  1 | 2020-01-01 | 2018-01-01 → 2019-10-01 | 11 | 87 | 13,934 | 8.3% | 720 / 476 / 600 / 599 / 720 / 675 |
|  2 | 2020-06-01 | 2018-06-01 → 2020-03-01 | 13 | 63 | 13,701 | 10.0% | 627 / 691 / 720 / 720 / 720 / 720 |
|  3 | 2020-11-01 | 2018-11-01 → 2020-08-01 | 12 | 48 | 13,741 | 9.7% | 679 / 437 / 720 / 599 / 600 / 477 |
|  4 | 2021-04-01 | 2019-04-01 → 2021-01-01 | 13 | 41 | 13,716 | 10.1% | 477 / 720 / 720 / 720 / 597 / 720 |
|  5 | 2021-09-01 | 2019-09-01 → 2021-06-01 | 14 | 30 | 13,560 | 10.9% | 656 / 663 / 720 / 720 / 720 / 720 |
|  6 | 2022-02-01 | 2020-02-01 → 2021-11-01 | 14 | 32 | 13,558 | 10.9% | 720 / 720 / 720 / 720 / 720 / 720 |
|  7 | 2022-07-01 | 2020-07-01 → 2022-04-01 | 9 | 20 | 14,165 | 6.9% | 720 / 720 / 720 / 720 / 720 / 720 |
|  8 | 2022-12-01 | 2020-12-01 → 2022-09-01 | 8 | 19 | 14,285 | 6.1% | 720 / 720 / 720 / 599 / 720 / 720 |
|  9 | 2023-05-01 | 2021-05-01 → 2023-02-01 | 2 | 6 | 15,021 | 1.6% | 720 / 720 / 720 / 720 / 720 / 720 |
| 10 | 2023-10-01 | 2021-10-01 → 2023-07-01 | 1 | 2 | 15,072 | 0.8% | 720 / 720 / 720 / 720 / 720 / 720 |
| 11 | 2024-03-01 | 2022-03-01 → 2023-12-01 | 1 | 2 | 15,120 | 0.8% | 720 / 720 / 720 / 720 / 720 / 720 |
| 12 | 2024-08-01 | 2022-08-01 → 2024-05-01 | 1 | 2 | 15,096 | 0.8% | 720 / 720 / 720 / 720 / 720 / 720 |
| 13 | 2025-01-01 | 2023-01-01 → 2024-10-01 | 1 | 2 | 15,096 | 0.8% | 720 / 720 / 720 / 720 / 720 / 720 |
| 14 | 2025-06-01 | 2023-06-01 → 2025-03-01 | 0 | 0 | 15,217 | 0.0% | 720 / 720 / 720 / 720 / 720 / 720 |
| 15 | 2025-11-01 | 2023-11-01 → 2025-08-01 | 0 | 0 | 15,217 | 0.0% | 720 / 720 / 720 / 720 / 720 / 720 |

The three zero-volume, zero-trade and flat bars are the same observed bars. Count their union, not their sum. The raw training-count regression target remains `COMMITTED_TRAIN_BUDGET`; the table's changed test counts follow the corrected first-target timestamp convention.

## Interpretation

Fixed calendar duration does not fix sample size. Fixed sample size does not equalize all optimizers' update counts or eliminate changes in market conditions. Missing outcomes are not imputed; the population is surviving continuous windows. Coverage is descriptive and may be a covariate, but it cannot recover unavailable outcomes or prove missingness is ignorable. No assertion that missing bars are exchange downtime, nontrading, or stress-related is made without separate evidence.

## Reproduce

```powershell
.venv/Scripts/python.exe -X utf8 .research/audit-repair-work/check_window_budget.py
.venv/Scripts/python.exe -m pytest tests/test_data_plane.py tests/test_audit_repairs.py -q
```

For session configuration and continuation, use [USAGE](../USAGE.md). For each audit disposition, use [the remediation report](AUDIT_REMEDIATION_2026-09-09.md).
