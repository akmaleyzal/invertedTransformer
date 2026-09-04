"""Encoder-only iTransformer: each variate is a token, attention runs across them.

Root §6.1. The inversion is the whole point — attention operates over the
**variate** axis, not the time axis, so the sequence the attention sees is
``N <= 12`` long rather than ``L = 96``. That is why ``d_model = 128`` and not
the reference implementation's 512: at ~14,000 training samples per origin, 512
over-parameterises badly, and the usual justification for a wide model — a long
attention sequence — does not apply here (`D25`).

**No causal mask.** Masking applies to the time axis; this attention runs over
the variate axis, where all tokens are contemporaneous. Causality is enforced
upstream, in the features and the windowing.

**K=1 is a designed control, not a degenerate bug** (`D50`). At ``N = 1``,
softmax over a single token returns weight 1, so attention reduces to
``W_O W_V x + x`` — the value and output projections and the residual **remain**;
it is not a bare identity. Parameter count is identical at every rung, because K
changes the token count and not a single weight shape. Say so in the
methodology: unexplained, an examiner reads it as an implementation error.

Upstream
--------
**Reimplemented, not vendored.** No file here is a copy of an upstream file:
the architecture is written from the published description against this study's
own tensor contract. That is a narrower claim than a fork and a stronger one to
defend, because an examiner can check it by reading both sides.

- Y. Liu, T. Hu, H. Zhang, H. Wu, S. Wang, L. Ma, and M. Long, "iTransformer:
  Inverted transformers are effective for time series forecasting," in *Proc.
  12th Int. Conf. Learn. Represent. (ICLR)*, 2024. arXiv:2310.06625. Official
  code: https://github.com/thuml/iTransformer (THUML, Tsinghua University;
  MIT; accessed 2026-09-03).
- T. Kim, J. Kim, Y. Tae, C. Park, J.-H. Choi, and J. Choo, "Reversible
  instance normalization for accurate time-series forecasting against
  distribution shift," in *Proc. 10th Int. Conf. Learn. Represent. (ICLR)*,
  2022 — the operation ``use_norm=True`` performs. Official code:
  https://github.com/ts-kim/RevIN (MIT; accessed 2026-09-03). The upstream
  README dates the paper 2021; dblp records ``conf/iclr/KimKTPCC22``, and 2022
  is used here.

Deliberate departures, each with the ID that forced it: ``d_model`` 512 -> 128
(`D25`); target-channel loss against the reference's all-channel default
(`D39`); ``lr`` halved every four epochs rather than every epoch (`D47`); and a
runtime ``capture`` attribute for the attention maps that consumes no RNG, so a
captured run stays bit-identical (`D62d`). Everything else is adopted unchanged
and never tuned (`D38`).
:data:`itransformer_btc.config.SOURCE_PROVENANCE` carries this row and the
fifteen others in the same form.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class ITransformerConfig:
    """Hyperparameters, adopted unchanged from Liu et al. (2024) bar ``d_model``.

    **Nothing here is tuned, deliberately** (`D38`). Holding capacity fixed is
    what makes the rungs comparable; per-rung tuning would confound the ladder
    with model selection. Root §11's checklist item on validation-based
    hyperparameter selection therefore applies to ARIMA order and ridge alpha
    only — those are the two models where selection actually happens.
    """

    seq_len: int = 96
    pred_len: int = 24
    d_model: int = 128
    d_ff: int = 256
    e_layers: int = 2
    n_heads: int = 8
    dropout: float = 0.1
    use_norm: bool = True
    #: Force attention weights uniform — the third main-grid arm (`D50`).
    #: K=1 vs K=8 differs in *information* and in *whether attention is active*
    #: at the same time, so a decaying A(b) is equally consistent with a
    #: capacity story as with the information story RQ2 claims. This separates
    #: them, at runs Figure 5 needs anyway.
    uniform_attention: bool = False

    # -- the Architecture protocol (`D56`) ----------------------------------
    #
    # Methods, not fields. ``write_artifacts`` records ``asdict(cfg)``, so
    # anything added here as a *field* would enter every iTransformer
    # ``meta/*.json`` and change bytes the 534-run grid has already produced.

    def build(self) -> "ITransformer":
        """A fresh module for this configuration."""
        return ITransformer(self)

    def loss_target(self) -> str:
        """``"target"`` — MSE on the target channel only, at every rung (`D39`).

        A constant rather than a field, deliberately. Standard iTransformer
        implementations compute the loss over all N channels, which would make
        K=12 a 12-task problem and K=1 a 1-task problem: auxiliary supervision
        varying with the study's own independent variable, and K=1 no longer the
        stated control but a different learning problem. Root §11 carries this as
        a verifiable assertion, and a field would be one edit away from failing
        it silently.
        """
        return "target"

    def schedule(self) -> "TrainSchedule":
        """Root §6.2's optimisation budget: 30 epochs, patience 5, LR halved every 4.

        A method rather than a field, and that is load-bearing for the same reason
        ``build`` and ``loss_target`` are: ``write_artifacts`` records
        ``asdict(cfg)``, so a field added here would enter every iTransformer
        ``meta/*.json`` and change bytes the 684-run grid has already produced.
        The schedule is logged under its own ``meta`` key instead.
        """
        from itransformer_btc.train import TrainSchedule

        return TrainSchedule()

    def fit(
        self,
        tensors: "OriginTensors",
        spec: "RunSpec",
        *,
        device: "torch.device | None" = None,
    ) -> tuple["ITransformer", "ITransformerConfig", "TrainOutcome"]:
        """Train one cell and hand back this config **unchanged**.

        Nothing is selected here: every hyperparameter is fixed a priori and
        identical at every rung (`D38`), which is what makes the rungs
        comparable. Ridge is the contrast — its alpha is chosen on the validation
        sub-block, so its ``fit`` returns a different config than it received.

        The import is deferred because ``train`` owns the protocol this
        satisfies, and importing it at module scope would be a cycle. In the
        flattened notebook there are no modules at all and ``train_one`` is
        simply a name a later cell defines, bound by the time this is called.
        """
        from itransformer_btc.train import train_one

        model, outcome = train_one(tensors, spec, self, device=device)
        return model, self, outcome


class LongScheduleConfig(ITransformerConfig):
    """`D62c` --- the exploratory arm that answers "you under-trained".

    The most obvious attack on a null result, and until now it had no answer. The
    document implied the 30-epoch cap was the budget that bound; measured, **0 of
    444** iTransformer runs reached it, mean 10.49, maximum 26. What bound was the
    learning-rate schedule, so widening the cap alone would have changed nothing.
    This widens the schedule: LR halved every 8 epochs instead of 4, 60 epochs
    instead of 30, patience 10 instead of 5.

    A **plain subclass adding no field**. ``asdict`` therefore returns exactly the
    parent's dict, ``meta['config']`` is unchanged, and the difference lives in
    the ``meta['schedule']`` key and in the ``run_id`` tag ``itrl``. Nothing about
    the ladder's confirmatory numbers moves: this arm is **exploratory**, declared
    as such under §13.2's confirmatory-versus-exploratory rule, and reported in
    its own table because §12 forbids two code vintages sharing one.
    """

    def schedule(self) -> "TrainSchedule":
        from itransformer_btc.train import TrainSchedule

        return TrainSchedule(max_epochs=60, patience=10, lr_halve_every=8)


@dataclass(frozen=True, slots=True)
class TunedConfig(ITransformerConfig):
    """`D70`'s tuned arm --- the winning configuration, **learning rate included**.

    `D76`: the arm shipped without this class, and that was the defect. The search
    space is ``d_model x e_layers x lr`` and
    :func:`~itransformer_btc.runner.tune_on_validation` ranked all eighteen points
    on origin 1's validation --- but it returned a bare
    :class:`ITransformerConfig`, which carries no learning rate. The winner's
    ``lr = 1e-3`` was therefore *selected and then discarded*, and the arm ran the
    winner's architecture under the default ``1e-4``: a point the search had also
    evaluated and had **not** ranked first. Root §12 requires a number to resolve
    to the decision that produced it, the documented decision and the executed run
    disagreed, and the arm did not answer the referee's question it exists for.

    ``lr`` is a **field**, unlike every other member of the Architecture protocol,
    and the exception is deliberate. ``write_artifacts`` records ``asdict(cfg)``,
    so a field on :class:`ITransformerConfig` would enter every iTransformer
    ``meta/*.json``; a field on a subclass one arm uses enters only that arm's,
    where recording the rate that ran is exactly what §12 asks for. It reaches the
    trainer through :meth:`schedule`, the same route `D62c`'s ``itrl`` takes.
    """

    #: The learning rate the validation search selected. The default is root
    #: §6.2's, so an un-tuned construction of this class is the main arm.
    lr: float = 1e-4

    def schedule(self) -> "TrainSchedule":
        from itransformer_btc.train import TrainSchedule

        return TrainSchedule(lr=self.lr)


class InvertedEmbedding(nn.Module):
    """Embed each variate's entire lookback: ``Linear(L -> d_model)``.

    With ``d_model = 128 > L = 96`` this projection is generically injective, so
    the whole lookback survives it. Root §5.3 leans on that: the reason for
    excluding moving averages is not that they are unrecoverable — a linear
    function of the lookback is recoverable in principle — but that adding one
    raises nominal K without raising information, which is precisely the
    phenomenon RQ1 exists to measure.
    """

    def __init__(self, seq_len: int, d_model: int, dropout: float) -> None:
        super().__init__()
        self.projection = nn.Linear(seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """``(B, L, N) -> (B, N, d_model)``."""
        return self.dropout(self.projection(x.permute(0, 2, 1)))


class VariateAttention(nn.Module):
    """Multi-head attention over the variate axis, optionally forced uniform."""

    def __init__(self, d_model: int, n_heads: int, dropout: float, uniform: bool) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model {d_model} not divisible by n_heads {n_heads}")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.uniform = uniform
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        #: Runtime capture of the attention weights -- Figure 5's only input
        #: (`D62d`). A plain module attribute, **never** an
        #: :class:`ITransformerConfig` field: ``write_artifacts`` records
        #: ``asdict(cfg)``, so a field here would appear in every iTransformer
        #: ``meta/*.json`` and change bytes the 684-run grid has already
        #: produced. Off by default, and the branch consumes no RNG, so a run
        #: with capture on is bit-identical to one without --- which is what
        #: makes the maps describe the model whose numbers the paper reports
        #: rather than a second model that merely resembles it.
        self.capture: bool = False
        self.last_weights: Tensor | None = None

    def forward(self, x: Tensor) -> Tensor:
        b, n, d = x.shape
        shape = (b, n, self.n_heads, self.head_dim)
        v = self.v(x).view(shape).transpose(1, 2)

        if self.uniform:
            # Every variate attends equally to every variate. W_V, W_O and the
            # parameter count stay intact, so the arm isolates *what attention
            # selects* rather than how much capacity the model has.
            context = v.mean(dim=2, keepdim=True).expand(-1, -1, n, -1)
        else:
            q = self.q(x).view(shape).transpose(1, 2)
            k = self.k(x).view(shape).transpose(1, 2)
            scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            weights = torch.softmax(scores, dim=-1)
            if self.capture:
                # Averaged over heads: Figure 5 shows variate-level reliance, and
                # eight per-head panels per layer would be a different figure.
                self.last_weights = weights.detach().mean(dim=1)
            context = self.dropout(weights) @ v

        return self.out(context.transpose(1, 2).reshape(b, n, d))


class EncoderLayer(nn.Module):
    """Attention over variates, then a position-wise FFN. Post-norm, as in the paper."""

    def __init__(self, cfg: ITransformerConfig) -> None:
        super().__init__()
        self.attention = VariateAttention(
            cfg.d_model, cfg.n_heads, cfg.dropout, cfg.uniform_attention
        )
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff, cfg.d_model),
        )
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm1(x + self.dropout(self.attention(x)))
        return self.norm2(x + self.dropout(self.ffn(x)))


class ITransformer(nn.Module):
    """``(B, L, N) -> (B, H)`` on the target channel.

    The output is the target channel alone even though the projection produces
    all N, because the **loss is single-channel** (`D39`). Standard
    implementations compute it over all N, which would make K=12 a 12-task
    problem and K=1 a 1-task problem — auxiliary supervision varying with the
    study's own independent variable, and K=1 no longer the stated control but a
    different learning problem. The reference implementation defaults to the
    option that breaks the design, so root §11 carries this as an assertion.
    """

    def __init__(self, cfg: ITransformerConfig, target_index: int = 0) -> None:
        super().__init__()
        self.cfg = cfg
        self.target_index = target_index
        self.embedding = InvertedEmbedding(cfg.seq_len, cfg.d_model, cfg.dropout)
        self.layers = nn.ModuleList(EncoderLayer(cfg) for _ in range(cfg.e_layers))
        self.projection = nn.Linear(cfg.d_model, cfg.pred_len)

    def forward(self, x: Tensor) -> Tensor:
        """Args: ``x`` of shape ``(B, L, N)``. Returns ``(B, H)``."""
        mean = std = None
        if self.cfg.use_norm:
            # Per-channel instance normalisation over time. Root §6.3: this is
            # what makes the outer StandardScaler cancel algebraically — and it
            # is itself a nonlinearity, which is why the F2 estimators
            # contribute *shape* and not *level*, the confound `D04` requires be
            # disclosed in Limitations whatever RQ1 returns.
            mean = x.mean(dim=1, keepdim=True)
            x = x - mean
            std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + 1e-5)
            x = x / std

        h = self.embedding(x)
        for layer in self.layers:
            h = layer(h)
        out = self.projection(h).permute(0, 2, 1)  # (B, H, N)

        if self.cfg.use_norm:
            out = out * std[:, 0, :].unsqueeze(1) + mean[:, 0, :].unsqueeze(1)

        return out[:, :, self.target_index]

    def forecast_target(self, x: Tensor) -> Tensor:
        """``(B, L, N) -> (B, H)`` — here, identical to ``forward``.

        The method exists because ``preds/*.parquet`` holds the target channel
        for **every** model in the study, and a channel-independent baseline's
        ``forward`` returns all N (`D56`). Declaring the projection on the model
        that wrote the file beats sniffing the rank of an output tensor: the
        prediction file's meaning then rests on something a model said, not on a
        shape a reader has to reverse-engineer.
        """
        return self(x)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
