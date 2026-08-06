# Origin window budget

Per-origin and per-block window accounting for the walk-forward grid. Required by `D45`: §11's
rejected-window assertion is an **exact equality per origin**, and an exact assertion needs an exact
table. The pooled figure survives only as §4.3's feasibility argument for not narrowing the analysis
window — it is wrong at almost every individual origin and must never be the assertion target.

**Measured from the artifact, 2026-08-06 — this table supersedes the derived one (`D51`).** The
previous version was computed by hand from `data/raw/BTCUSDT_1h_gaps.csv` plus the design constants,
and it diverged from the artifact at **twelve of fifteen origins**. Every number below now comes from
`src/itransformer_btc/budget.py` run against `BTCUSDT_1h.parquet`
(sha256 `8270a84b07c2923b…`), and `tests/test_data_plane.py` asserts it. Regenerate with
`format_markdown()` whenever the artifact or the origin grid changes, and update
`budget.py::COMMITTED_TRAIN_BUDGET` in the same commit — never one alone.

## Constants

| Symbol | Value | Source |
|---|---|---|
| `L` / `H` | 96 / 24 | §6.2 |
| `L + H − 1` | **119** — window starts destroyed per break | §4.3 |
| Training window | 24 months, rolling | §8.1 |
| Training sub-block | first 21 months (validation is the final 3) | §8.1, `D25` |
| Origin spacing / count | **5 months / 15** | §8.1, `D26` |
| Test blocks | 6 × 30 days = **720 forecast origins each** | §8.1, `D51` |

## Per-origin training sub-block (21 months)

`Breaks` counts maximal runs of excluded calendar positions; `Excluded` counts the positions
themselves, missing **and** unusable. `Windows kept` is the assertion target and is computed
segment-wise, `Σ max(0, nᵢ − 119)`.

| # | Origin | Training sub-block | Breaks | Excluded | Windows kept | Loss | Surviving starts B1…B6 (of 720) |
|---:|---|---|---:|---:|---:|---:|---|
|  1 | 2020-01-01 | 2018-01-01 → 2019-10-01 | 11 | 87 | 13,934 | 8.3% | 720 / 439 / 637 / 599 / 720 / 598 |
|  2 | 2020-06-01 | 2018-06-01 → 2020-03-01 | 13 | 63 | 13,701 | 10.0% | 598 / 720 / 720 / 720 / 720 / 655 |
|  3 | 2020-11-01 | 2018-11-01 → 2020-08-01 | 12 | 48 | 13,741 | 9.7% | 600 / 516 / 720 / 599 / 600 / 477 |
|  4 | 2021-04-01 | 2019-04-01 → 2021-01-01 | 13 | 41 | 13,716 | 10.1% | 477 / 720 / 720 / 720 / 597 / 632 |
|  5 | 2021-09-01 | 2019-09-01 → 2021-06-01 | 14 | 30 | 13,560 | 10.9% | 599 / 720 / 720 / 720 / 720 / 720 |
|  6 | 2022-02-01 | 2020-02-01 → 2021-11-01 | 14 | 32 | **13,558** | **10.9%** | 720 / 720 / 720 / 720 / 720 / 720 |
|  7 | 2022-07-01 | 2020-07-01 → 2022-04-01 | 9 | 20 | 14,165 | 6.9% | 720 / 720 / 720 / 720 / 720 / 720 |
|  8 | 2022-12-01 | 2020-12-01 → 2022-09-01 | 8 | 19 | 14,285 | 6.1% | 720 / 720 / 720 / 599 / 720 / 720 |
|  9 | 2023-05-01 | 2021-05-01 → 2023-02-01 | 2 | 6 | 15,021 | 1.6% | 720 / 720 / 720 / 720 / 720 / 720 |
| 10 | 2023-10-01 | 2021-10-01 → 2023-07-01 | 1 | 2 | 15,072 | 0.8% | 720 / 720 / 720 / 720 / 720 / 720 |
| 11 | 2024-03-01 | 2022-03-01 → 2023-12-01 | 1 | 2 | 15,120 | 0.8% | 720 / 720 / 720 / 720 / 720 / 720 |
| 12 | 2024-08-01 | 2022-08-01 → 2024-05-01 | 1 | 2 | 15,096 | 0.8% | 720 / 720 / 720 / 720 / 720 / 720 |
| 13 | 2025-01-01 | 2023-01-01 → 2024-10-01 | 1 | 2 | 15,096 | 0.8% | 720 / 720 / 720 / 720 / 720 / 720 |
| 14 | 2025-06-01 | 2023-06-01 → 2025-03-01 | 0 | 0 | **15,217** | **0.0%** | 720 / 720 / 720 / 720 / 720 / 720 |
| 15 | 2025-11-01 | 2023-11-01 → 2025-08-01 | 0 | 0 | **15,217** | **0.0%** | 720 / 720 / 720 / 720 / 720 / 720 |

Training-window range **13,558 … 15,217** — a **10.9%** spread, monotone in calendar time. Largest
training tensor: `15,217 × 96 × 12 × 4 B` = **70.12 MB**.

## What changed against the derived table, and why

Three independent causes, none of them arithmetic slips:

1. **The three unusable bars were never counted.** The segment law breaks at zero-volume and
   `H == L` bars as well as at downtime (`D14`), and the derived table used only the 27 gap blocks.
   Measured: **exactly 3 unusable bars**, at `2019-06-07T21:00`, `2021-02-11T03:00` and
   `2023-03-24T12:00`. They add one break run and one excluded position wherever they land.
2. **Gap blocks straddling a span boundary were charged whole.** The derivation attributed a block to
   the span containing its `gap_start`; the measurement counts only positions actually inside the
   span. This moves counts in **both** directions, which is why some origins measured higher than
   derived and others lower.
3. **The closed form is not an identity.** `(bars − 119) − [119 × breaks + excluded]` equals the
   segment-wise count only while every segment clears 120 bars. Origin 2022-02 contains a segment of
   **80 bars**: it contributes zero windows, but the closed form charges it `80 − 119 = −39` and
   absorbs the negative silently. Seven origins are affected, understated by 39 … 137 windows.
   **Use the segment-wise count; keep the closed form only as an upper-bound sanity check.**

## Both previously-unmeasured quantities, answered

The earlier version listed two open items. Both are now measured, and the answer to the second is
stronger than the question assumed:

1. **`H == L` bar count = 3.**
2. **The 3 zero-volume, 3 zero-trade and 3 `H == L` bars are the same 3 bars.** Total unusable is
   therefore **3**, not 9 — which follows mechanically: no volume means no trades, and no trades
   means the bar's high and low never separate. §4.3's "thirty breaks = 27 downtime blocks + 3
   zero-volume bars" is right in total for the wrong reason, and right only because the three
   coincide.

## What the spread means

- **Training volume is not constant across origins**, which is what §8.1's fixed rolling window was
  chosen to guarantee. 13,558 … 15,217, monotone in calendar time and so correlated with the origin
  index and with everything compared across origins. **Control:** subsample every origin's training
  set to **13,558**. Report the uncontrolled version as the sensitivity.
- **Test-block loss is 0% at 74 of 90 cells and reaches 39.0%** (origin 1, B2: 439 of 720). Every
  origin from 9 onward is clean. Report the surviving count next to every `A(i,b)` and `D(i,b)` in
  Table 5.
- **The loss is not random.** Binance outages cluster on stress (2018-02-08 is 33 bars; 2020-02-19
  and 2020-12-21 sit on large moves), so dropped targets are disproportionately high-volatility ones,
  and survival is conditioned on *future* gaps — information unavailable at the forecast origin. A
  stated limitation, not a fixable defect.

## Test blocks hold 720 forecast origins, not 601 (`D51`)

A **training** window must lie wholly inside its span: its target may not cross into validation
(§8.2). A **test** window may not — §8.3 states that a window's 96-bar lookback reaching back across
the boundary is past information legitimately available to a forecaster at that moment, and that
blocking it "would make the evaluation unrealistically pessimistic". Every one of a block's 720 hours
is therefore an admissible forecast origin; what disqualifies one is a break inside the 120 bars it
spans, wherever those bars fall.

Counting test blocks the training way returns **601 of 720 on a perfectly clean block** — a 16.5%
phantom loss that would read as outage damage and would enter §9.2's block-coverage covariate as pure
noise. The two semantics differ by exactly `L + H − 1`.

## Regenerating

```bash
python -m pytest tests/test_data_plane.py -q      # asserts this table
python -c "import sys; sys.path.insert(0,'src'); \
  from itransformer_btc.segments import load_bars, usable_mask; \
  from itransformer_btc.budget import budget_table, format_markdown; \
  print(format_markdown(budget_table(usable_mask(load_bars()))))"
```

The table is a pure function of `data/raw/BTCUSDT_1h.parquet` and the constants above. It is
committed so a Stage 2 run has something to be checked *against* rather than something to be tuned
*to* — which is why `budget.py` pins the values instead of recomputing them inside the test: with
both sides computed the same way, a regression would agree with itself and pass.
