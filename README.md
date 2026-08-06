# invertedTransformer

**Nominal Variates or Effective Dimensionality? A Walk-Forward Evaluation of iTransformer for
Hourly Bitcoin Forecasting.**

A research study, not a product. The deliverable is a manuscript; the model is an experimental
instrument. There is no production inference path.

| | |
|---|---|
| Data | BTCUSDT **spot only**, 1-hour klines, Binance, 2018-01-01 → 2026-08-01 UTC (end exclusive) |
| Artifacts | `data/raw/` — 75,094 bars, 99.8378% coverage, 122 missing in 27 downtime blocks. **Immutable** |
| Model | iTransformer (PyTorch only), `L=96`, `H=24`, `d_model=128` |
| Variates | 12, laddered at K ∈ {1, 4, 8, 12} across five information families |
| Evaluation | Rolling-origin walk-forward, **15 origins at 5-month spacing**, purge at both boundaries, 6 × 30-day test blocks, no retraining |
| Grid | **789 runs** — 534 iTransformer (main, uniform-attention, falsification, horizon sweep) + 195 baselines + 60 ridge. Not 837: 48 sweep cells share a `run_id` with the main grid (`D53e`) |
| Hardware | Kaggle 2 × T4, 12 h per session, 30 GPU-h per week |

Three questions: whether added-variate benefit tracks nominal count **K** or effective
dimensionality **K_eff**; whether the multivariate advantage **decays** with time since training;
and what **retraining cadence** follows, under a threshold fixed in advance.

## Start here

**`CLAUDE.md` is the project law.** Read it before touching anything — data contract, variate
ladder, walk-forward protocol, statistical specification, Kaggle execution and resume protocol,
anti-leakage checklist, traceability and paper contracts.

**`USAGE.md` is the operational companion** — install, the pipeline stage by stage, every command,
the artifact schemas, expected numbers to check a break against, and troubleshooting.

- `docs/DIVERGENCE_REGISTER.md` — **corrections `D01`–`D53f`**, each with its evidence and the
  manuscript section that must disclose it. `D01–D22` correct the source specifications; `D23–D50`
  correct `CLAUDE.md` itself, found by a later adversarial audit; `D51`–`D53` were found by *running
  the code*, and no amount of re-reading would have produced them.
- `docs/ORIGIN_WINDOW_BUDGET.md` — per-origin and per-block window accounting. Committed **before**
  any run so the pipeline's assertions have a target they cannot be tuned to.
- `paper/CLAUDE.md` — writing posture, and the **only** directory-local rule file. `src/CLAUDE.md`
  and `notebooks/CLAUDE.md` were deleted on 2026-08-06: they restated the root at 55–65% overlap, and
  a subdirectory `CLAUDE.md` loads only when a file in that subtree is touched — so a prohibition
  living there is absent exactly when it is most needed. Rules whose violation is catastrophic belong
  in the root, which is always loaded. See §15.
- `research_specification_itransformer_btc.md`, `reference_library_itransformer_btc.md` — source
  inputs. **Not authority**: where they disagree with `CLAUDE.md`, `CLAUDE.md` wins.

## Note

Before 2026-08-05 this repository pursued a different project — a production 1-minute BTC/USDT model
fusing gold, dollar-index, and macro data. It is superseded in full; see `CLAUDE.md` §17. Nothing
from it is authoritative.
