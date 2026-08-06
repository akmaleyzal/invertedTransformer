# Origin window budget

Per-origin and per-block window accounting for the walk-forward grid. Required by `D45`: §11's
rejected-window assertion is an **exact equality per origin**, and an exact assertion needs an exact
table. The pooled 4.9%/5.2% figure survives only as §4.3's feasibility argument for not narrowing the
analysis window — it is wrong at almost every individual origin and must never be the assertion
target.

**Derived, not measured by the pipeline.** Every number below comes from `data/BTCUSDT_1h_gaps.csv`
(27 downtime blocks, 122 missing bars) plus the design constants in `CLAUDE.md` §6.2 and §8.1. It is
committed so that a Stage 2 run has something to be checked *against*, rather than something to be
tuned *to*. Regenerate it whenever the gap catalogue or the origin grid changes.

## Constants

| Symbol | Value | Source |
|---|---|---|
| `L` / `H` | 96 / 24 | §6.2 |
| `L + H − 1` | **119** — window starts destroyed per break | §4.3 |
| Training window | 24 months, rolling | §8.1 |
| Training sub-block | first 21 months (validation is the final 3) | §8.1, `D25` |
| Origin spacing / count | **5 months / 15** | §8.1, `D26` |
| Test blocks | 6 × 30 days = 720 window starts each | §8.1 |
| Break cost | `119 × breaks + missing_bars` | §4.3 |

## Per-origin training sub-block (21 months)

`windows kept` is the assertion target: `kept(origin) == (bars − 119) − [119 × breaks + missing]`.

| # | Origin | Training sub-block | Breaks | Missing bars | Windows kept | Loss | Test-block loss B1…B6 |
|---:|---|---|---:|---:|---:|---:|---|
|  1 | 2020-01-01 | 2018-01-01 → 2019-10-01 | 10 | 86 | 13,917 | 8.4% | 0 / 244 / 120 / 121 / 0 / 122 |
|  2 | 2020-06-01 | 2018-06-01 → 2020-03-01 | 12 | 62 | 13,727 | 9.8% | 122 / 0 / 0 / 0 / 0 / 0 |
|  3 | 2020-11-01 | 2018-11-01 → 2020-08-01 | 11 | 47 | 13,861 | 8.9% | 120 / 243 / 0 / 120 / 120 / 243 |
|  4 | 2021-04-01 | 2019-04-01 → 2021-01-01 | 12 | 40 | 13,797 | 9.6% | 243 / 0 / 0 / 0 / 123 / 0 |
|  5 | 2021-09-01 | 2019-09-01 → 2021-06-01 | 14 | 29 | 13,522 | 11.1% | 121 / 0 / 0 / 0 / 0 / 0 |
|  6 | 2022-02-01 | 2020-02-01 → 2021-11-01 | 14 | 31 | **13,520** | **11.2%** | 0 / 0 / 0 / 0 / 0 / 0 |
|  7 | 2022-07-01 | 2020-07-01 → 2022-04-01 | 9 | 19 | 14,127 | 7.2% | 0 / 0 / 0 / 0 / 0 / 0 |
|  8 | 2022-12-01 | 2020-12-01 → 2022-09-01 | 8 | 18 | 14,247 | 6.4% | 0 / 0 / 0 / 120 / 0 / 0 |
|  9 | 2023-05-01 | 2021-05-01 → 2023-02-01 | 2 | 6 | 15,021 | 1.6% | 0 / 0 / 0 / 0 / 0 / 0 |
| 10 | 2023-10-01 | 2021-10-01 → 2023-07-01 | 1 | 1 | 15,073 | 0.8% | 0 / 0 / 0 / 0 / 0 / 0 |
| 11 | 2024-03-01 | 2022-03-01 → 2023-12-01 | 1 | 1 | 15,121 | 0.8% | 0 / 0 / 0 / 0 / 0 / 0 |
| 12 | 2024-08-01 | 2022-08-01 → 2024-05-01 | 1 | 1 | 15,097 | 0.8% | 0 / 0 / 0 / 0 / 0 / 0 |
| 13 | 2025-01-01 | 2023-01-01 → 2024-10-01 | 1 | 1 | 15,097 | 0.8% | 0 / 0 / 0 / 0 / 0 / 0 |
| 14 | 2025-06-01 | 2023-06-01 → 2025-03-01 | 0 | 0 | **15,217** | **0.0%** | 0 / 0 / 0 / 0 / 0 / 0 |
| 15 | 2025-11-01 | 2023-11-01 → 2025-08-01 | 0 | 0 | **15,217** | **0.0%** | 0 / 0 / 0 / 0 / 0 / 0 |

**Pooled:** 96 break-incidences, 342 missing-bar-incidences across the fifteen overlapping training
windows, 5.2% pooled loss. The counts exceed the catalogue's 27 blocks / 122 bars because training
windows overlap by 79.2% — the same gap is charged to several origins, which is correct: each origin
trains its own model and each loses those windows.

## What the spread means

- **Training volume is not constant across origins**, which is what §8.1's fixed rolling window was
  chosen to guarantee. The range is 13,520 … 15,217 windows — a **12.5% spread**, monotone in
  calendar time, hence correlated with the origin index and so with everything compared across
  origins. **Control:** subsample every origin's training set to **13,520** windows. Report the
  uncontrolled version as the sensitivity.
- **Test-block loss is 0% at 74 of 90 cells and reaches 33.9%** (origin 1, B2: 244 of 720). Blocks
  carrying any loss: origin 1 (B2, B3, B4, B6), origin 2 (B1), origin 3 (B1, B2, B4, B5, B6),
  origin 4 (B1, B5), origin 5 (B1), origin 8 (B4). **Every origin from 9 onward is clean.** Report
  the surviving count next to every `A(i,b)` and `D(i,b)` in Table 5.
- **The loss is not random.** Binance outages cluster on stress (2018-02-08 is 33 bars, 2020-02-19
  and 2020-12-21 sit on large moves), so dropped targets are disproportionately high-volatility ones,
  and survival is conditioned on *future* gaps — information unavailable at the forecast origin. This
  is a stated limitation, not a fixable defect.

## Not yet measured

Two quantities feed §4.3's arithmetic and are **not** in this table because they have never been
measured:

1. the **`H == L` bar count** — the segment law (§4.3) breaks the series there, but the report
   schema records only `zero_volume_bars` and `zero_trade_bars`;
2. whether the 3 zero-volume and 3 zero-trade bars are the **same** 3 bars.

Emit both at Stage 2, add them to the report schema, then regenerate this table. Until then every
figure here is a lower bound on breaks and an upper bound on windows kept.

## Regenerating

The table is a pure function of `data/BTCUSDT_1h_gaps.csv` and the constants above: for each origin,
count gap blocks whose `gap_start` falls in the span, sum their `missing_bars`, and apply
`119 × breaks + missing`. Gap blocks are attributed by `gap_start`; a block straddling a span
boundary is charged to the span containing its start, which is the same convention the window
validity rule in §4.3 enforces.

The figures above were produced against the artifact **before** the `2026-08-01T00:00` boundary bar
is dropped (`D33`). That bar lies outside every training and test span in the grid, so no number here
changes when it is removed — but re-run the derivation after the artifact is re-emitted and confirm.
