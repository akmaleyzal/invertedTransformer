# invertedTransformer

**Nominal Variates or Effective Dimensionality? A Walk-Forward Evaluation of iTransformer for
Hourly Bitcoin Forecasting.**

A research study, not a product. The deliverable is a manuscript; the model is an experimental
instrument. There is no production inference path.

| | |
|---|---|
| Data | BTCUSDT **spot only**, 1-hour klines, Binance, 2018-01-01 → 2026-08-01 UTC |
| Model | iTransformer (PyTorch only), `L=96`, `H=24`, `d_model=128` |
| Variates | 12, laddered at K ∈ {1, 4, 8, 12} across five information families |
| Evaluation | Rolling-origin walk-forward, 13 origins, H-step purge, 6 × 30-day test blocks, no retraining |
| Hardware | Kaggle 2 × T4, 12 h per session, 30 GPU-h per week |

Three questions: whether added-variate benefit tracks nominal count **K** or effective
dimensionality **K_eff**; whether the multivariate advantage **decays** with time since training;
and what **retraining cadence** follows, under a threshold fixed in advance.

## Start here

**`CLAUDE.md` is the project law.** Read it before touching anything — data contract, variate
ladder, walk-forward protocol, statistical specification, Kaggle execution and resume protocol,
anti-leakage checklist, traceability and paper contracts.

- `docs/DIVERGENCE_REGISTER.md` — the 22 corrections applied to the source specifications, each with
  its evidence and the manuscript section that must disclose it.
- `src/CLAUDE.md`, `notebooks/CLAUDE.md`, `paper/CLAUDE.md` — directory-local rules.
- `research_specification_itransformer_btc.md`, `reference_library_itransformer_btc.md` — source
  inputs. **Not authority**: where they disagree with `CLAUDE.md`, `CLAUDE.md` wins.

## Note

Before 2026-08-05 this repository pursued a different project — a production 1-minute BTC/USDT model
fusing gold, dollar-index, and macro data. It is superseded in full; see `CLAUDE.md` §17. Nothing
from it is authoritative.
