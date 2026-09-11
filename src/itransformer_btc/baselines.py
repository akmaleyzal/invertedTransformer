"""Baseline configurations compared with iTransformer on common target calendars.

DLinear and PatchTST default to target-only loss, including checkpoint selection.
Their target-only path consumes target history alone (effective_input_channels=1)
despite the supplied K=8 tensor. Separate all-channel arms expose auxiliary-task
supervision. Ridge and LSTM can use the supplied multivariate history directly.

The post-audit PatchTST port uses its own BatchNorm/residual-attention encoder,
shared patch projection and head, affine-free RevIN, no patch padding, and no
head dropout. It is a declared configuration, not an exact copy of all upstream
defaults. Baseline LR sensitivity uses only origin-1 validation. A 120-epoch cap
with patience 12 is recorded, and reaching or avoiding it is not proof of global
optimization convergence. All output metrics concern the return channel.

ARIMA remains outside this study's implemented scope; ADF does not establish
that AIC would select an ARIMA(0,0,0). LSTM and naive comparators ARE implemented.

Upstream
--------
**Two of these are reimplemented from published architectures; three have no
upstream code at all.** That distinction is the answer to "where did this come
from", so it is drawn per model rather than left to a blanket acknowledgement.

- **DLinear**, ``SeriesDecomposition`` — A. Zeng, M. Chen, L. Zhang, and Q. Xu,
  "Are transformers effective for time series forecasting?," in *Proc. 37th
  AAAI Conf. Artif. Intell.*, 2023, pp. 11121-11128. arXiv:2205.13504.
  Official code: https://github.com/cure-lab/LTSF-Linear (Apache-2.0; accessed
  2026-09-03). Reimplemented; the published all-channel objective, shared
  weights and centred moving average are kept as published (`D40`, `D56`).
- **PatchTST** — Y. Nie, N. H. Nguyen, P. Sinthong, and J. Kalagnanam, "A time
  series is worth 64 words: Long-term forecasting with transformers," in *Proc.
  11th Int. Conf. Learn. Represent. (ICLR)*, 2023. arXiv:2211.14730. Official
  code: https://github.com/yuqinie98/PatchTST (Apache-2.0; accessed
  2026-09-03). Reimplemented on this study's own
  :class:`itransformer_btc.model.EncoderLayer`, so it differs from iTransformer
  in what a token is and in nothing else. Patch 16 / stride 8 as published.
- **LSTMForecaster** — ``torch.nn.LSTM``
  (https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html,
  BSD-3-Clause; accessed 2026-09-03) is called directly and only the forecasting
  head is written here. S. Hochreiter and J. Schmidhuber, "Long short-term
  memory," *Neural Computation*, vol. 9, no. 8, pp. 1735-1780, 1997.
- **RidgeForecaster** — **not scikit-learn.** The normal equations are solved
  here in ``float64``. A. E. Hoerl and R. W. Kennard, "Ridge regression: Biased
  estimation for nonorthogonal problems," *Technometrics*, vol. 12, no. 1,
  pp. 55-67, 1970.
- **NaiveForecaster**, Naive-RW — closed forms, no upstream code.
  R. J. Hyndman and G. Athanasopoulos, *Forecasting: Principles and Practice*,
  3rd ed. OTexts, 2021.

:data:`itransformer_btc.config.SOURCE_PROVENANCE` carries every row in the same
form, with the licence and the full list of departures."""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from itransformer_btc.config import PRED_LEN, SEQ_LEN
from itransformer_btc.features import TARGET_INDEX
from itransformer_btc.metrics import assert_same_windows, load_predictions
from itransformer_btc.model import EncoderLayer, ITransformerConfig
from itransformer_btc.splits import OriginTensors
from itransformer_btc.train import (
    RunSpec,
    TrainOutcome,
    pick_device,
    SEED_LOCK,
    set_seed,
    train_one,
)
from itransformer_btc.train import TrainSchedule


class BaselineModule(nn.Module):
    """Shared parameter counting and explicit target-only input selection for channel-independent baselines."""

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forecast_target(self, x: Tensor) -> Tensor:
        """``(B, L, N) -> (B, H)`` on the target channel."""
        return self(x[:, :, TARGET_INDEX:TARGET_INDEX+1] if self.cfg.loss_target() == "target" else x)[:, :, 0 if self.cfg.loss_target() == "target" else TARGET_INDEX]


# -- ridge -------------------------------------------------------------------


#: Root §11: ridge alpha is selected on the validation sub-block, and with ARIMA
#: outside the minimal set it is the **only** hyperparameter selected anywhere in
#: this study (`D38`). The solve is unnormalised — ``(X'X + a I) W = X'Y`` — so
#: the scale that matters is ``diag(X'X) ~ n``, about 1.4e4 at these origins; the
#: grid spans five orders below it and two above.
RIDGE_ALPHAS: tuple[float, ...] = (1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6)


@dataclass(frozen=True, slots=True)
class RidgeConfig:
    """L2-regularised linear map from the flattened window to the H-step target.

    `D17`: K=1 iTransformer controls for *architecture* — it answers "does
    cross-variate attention help?" It does not answer "is a transformer needed at
    all?" Ridge on the same K features separates *does the information help* from
    *does attention help*, at seconds per run, and closes a question a reviewer
    asks otherwise.

    ``k`` is a field here and nowhere else among the study's configs.
    iTransformer's parameter count is identical at every rung because K changes
    the token count and not a weight shape; ridge's weight matrix is
    ``(L*K, H)``, so K is part of its geometry and belongs in ``meta['config']``.
    """

    seq_len: int = SEQ_LEN
    pred_len: int = PRED_LEN
    k: int = 8
    alphas: tuple[float, ...] = RIDGE_ALPHAS
    #: Chosen by :meth:`fit` on the validation sub-block. ``None`` in an unfitted
    #: config and never in a written ``meta/*.json`` — root §12 cannot regenerate
    #: a number whose only free parameter went unrecorded.
    alpha: float | None = None

    def build(self) -> "RidgeForecaster":
        return RidgeForecaster(self)

    def loss_target(self) -> str:
        """``"target"``: ridge predicts the target channel and nothing else."""
        return "target"

    def fit(
        self,
        tensors: OriginTensors,
        spec: RunSpec,
        *,
        device: torch.device | None = None,
    ) -> tuple["RidgeForecaster", "RidgeConfig", TrainOutcome]:
        """Solve the normal equations once, then pick alpha on validation.

        The Gram matrix and the right-hand side are built **once** and reused for
        every alpha, so the sweep costs one solve per candidate rather than a
        refit. In ``float64``: at ``L*K = 1152`` the design is conditioned badly
        enough that a ``float32`` Gram would make the smallest alphas report
        noise, and showing what an essentially unregularised linear map does is
        the whole reason the small alphas are in the grid.

        The intercept is fitted by centring and is **not** penalised. Shrinking
        it toward zero would shrink the forecast toward zero *in scaler space*,
        which is ``r = mu_g`` — the constant-drift model `D31` spent a section
        removing from the Naive-RW baseline.
        """
        device = device or pick_device()
        # A solve consumes no RNG. Seeded anyway, so a ridge run and an
        # iTransformer run of the same cell are reproducible under one rule
        # (root §16) rather than two.
        started = time.perf_counter()

        # Seeding and construction under one lock, as in ``train_one``: both draw
        # from the CPU generator, which every worker shares (`D68`).
        with SEED_LOCK:
            set_seed(spec.seed, device)
            model = self.build().to(device)
        x_tr = self._design(tensors.train.x, device)
        y_tr = torch.from_numpy(tensors.train.y).to(device).double()
        x_va = self._design(tensors.val.x, device)
        y_va = torch.from_numpy(tensors.val.y).to(device).double()

        x_mean, y_mean = x_tr.mean(0), y_tr.mean(0)
        x_tr -= x_mean
        y_tr -= y_mean
        gram = x_tr.T @ x_tr
        rhs = x_tr.T @ y_tr
        eye = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)

        best: tuple[float, float, Tensor] | None = None
        for alpha in self.alphas:
            weight = torch.linalg.solve(gram + alpha * eye, rhs)
            residual = (x_va - x_mean) @ weight + y_mean - y_va
            val_mse = float(residual.pow(2).mean())
            if best is None or val_mse < best[0]:
                best = (val_mse, float(alpha), weight)
        if best is None:
            raise ValueError("no ridge alpha to select; `alphas` is empty")

        val_mse, alpha, weight = best
        if len(self.alphas) > 1 and alpha in (self.alphas[0], self.alphas[-1]):
            # Not a failure. An alpha pinned at the top of the grid says the
            # least-squares fit is worthless and the best linear predictor is the
            # training mean, which is a finding. It is warned about because a
            # boundary selection is also what an unbracketed grid looks like, and
            # the two are indistinguishable from the number alone.
            warnings.warn(
                f"{spec.run_id}: ridge alpha {alpha:g} sits at the edge of "
                f"{self.alphas}; the grid may not bracket the optimum",
                stacklevel=2,
            )

        with torch.no_grad():
            model.weight.copy_(weight.to(torch.float32))
            model.bias.copy_((y_mean - x_mean @ weight).to(torch.float32))
        train_mse = float((x_tr @ weight - y_tr).pow(2).mean())

        return (
            model,
            replace(self, alpha=alpha),
            TrainOutcome(
                run_id=spec.run_id,
                # A solve, not a loop. Zero is the honest number, and it is what
                # tells a reader of Table 3 why this row has no epochs-to-stop.
                epochs_run=0,
                best_val_mse=val_mse,
                train_loss=train_mse,
                wall_time_s=time.perf_counter() - started,
                n_parameters=model.n_parameters(),
                device=str(device),
            ),
        )

    @staticmethod
    def _design(x: np.ndarray, device: torch.device) -> Tensor:
        """``(n, L, K) -> (n, L*K)`` in float64, on the device.

        Row-major, so a column is one (lag, variate) pair. Nothing depends on
        which ordering it is, only that this and
        :meth:`RidgeForecaster.forward` agree — which they do by both being
        ``reshape``.
        """
        return torch.from_numpy(x).to(device).reshape(len(x), -1).double()


class RidgeForecaster(nn.Module):
    """``y_hat = vec(x) @ W + b``, fitted in closed form.

    ``W`` and ``b`` are **buffers**, not parameters: nothing here is trained by
    gradient descent, and registering them as parameters would put them in front
    of an optimiser that must never see them. That is why :meth:`n_parameters`
    counts them explicitly — the usual sum over ``self.parameters()`` would
    report zero, and root §12 would record a model with no coefficients.
    """

    def __init__(self, cfg: RidgeConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.register_buffer(
            "weight",
            torch.zeros(cfg.seq_len * cfg.k, cfg.pred_len, dtype=torch.float32),
        )
        self.register_buffer("bias", torch.zeros(cfg.pred_len, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        """``(B, L, K) -> (B, H)``."""
        return x.reshape(len(x), -1) @ self.weight + self.bias

    def forecast_target(self, x: Tensor) -> Tensor:
        return self(x)

    def n_parameters(self) -> int:
        return self.weight.numel() + self.bias.numel()


# -- DLinear -----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DLinearConfig:
    """Shared trend/seasonal linear maps; target objective by default, all-channel sensitivity explicit. The validation-selected learning rate and resolved schedule are saved."""

    seq_len: int = SEQ_LEN
    pred_len: int = PRED_LEN
    #: Odd, so the decomposition's padding is symmetric. 25 is the published
    #: default and is not tuned here (`D38`, extended to the baselines).
    moving_avg: int = 25
    #: Recorded in ``meta/*.json`` rather than left to be inferred: see the
    #: header. A reader who does not know the objective cannot read
    #: ``best_val_mse``, which is an all-channel figure for this model and a
    #: target-channel one for the ladder.
    loss_channels: str = "target"
    channel_independent: bool = True

    lr: float = 1e-3
    max_epochs: int = 120
    patience: int = 12
    lr_halve_every: int = 20

    def schedule(self) -> "TrainSchedule":
        return TrainSchedule(lr=self.lr, max_epochs=self.max_epochs,
                             patience=self.patience, lr_halve_every=self.lr_halve_every)

    def build(self) -> "DLinear":
        return DLinear(self)

    def loss_target(self) -> str:
        return "all" if self.loss_channels == "all" else "target"

    def fit(
        self,
        tensors: OriginTensors,
        spec: RunSpec,
        *,
        device: torch.device | None = None,
    ) -> tuple["DLinear", "DLinearConfig", TrainOutcome]:
        """Root §6.2's schedule; nothing is selected, so the config returns as given."""
        model, outcome = train_one(tensors, spec, self, device=device)
        return model, self, outcome



class SeriesDecomposition(nn.Module):
    """Moving-average trend and the residual seasonal component.

    **This is a rolling window inside a model, and root §5.3's ban is on rolling
    *features*. The distinction is not a technicality, so here is the argument.**
    The ban exists because a rolling feature computed over the full series can let
    a later bar reach an earlier feature value — the ``center=True`` leak class —
    and root §8.3's no-embargo justification rests on no feature having one. This
    average is computed at inference time from the 96 bars of the window itself,
    every one of which precedes the first forecast hour, and the padding
    replicates the window's own endpoints rather than reaching outside it. No
    test-period bar can therefore influence any training-set value, which is the
    property §8.3 actually needs, and it holds even though the average is centred
    **within** the window, as the published DLinear's is. Reproducing the
    published decomposition matters: a causal variant would be a different model,
    and the question this baseline exists to answer is about DLinear.
    """

    def __init__(self, kernel: int) -> None:
        super().__init__()
        self.kernel = kernel
        self.average = nn.AvgPool1d(kernel, stride=1, padding=0)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """``(B, L, N) -> (seasonal, trend)``, both ``(B, L, N)``."""
        front_pad = (self.kernel - 1) // 2
        padded = torch.cat(
            [
                x[:, :1, :].repeat(1, front_pad, 1),
                x,
                x[:, -1:, :].repeat(1, self.kernel - 1 - front_pad, 1),
            ],
            dim=1,
        )
        trend = self.average(padded.permute(0, 2, 1)).permute(0, 2, 1)
        return x - trend, trend


class DLinear(BaselineModule):
    """``(B, L, N) -> (B, H, N)``: decompose, map each part linearly, add.

    No instance normalisation, as published — which is exactly the case root §6.3
    says the outer ``StandardScaler`` exists to serve, so this model reads the
    same scaler space as every other and needs nothing of its own.
    """

    def __init__(self, cfg: DLinearConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.decomposition = SeriesDecomposition(cfg.moving_avg)
        self.seasonal = nn.Linear(cfg.seq_len, cfg.pred_len)
        self.trend = nn.Linear(cfg.seq_len, cfg.pred_len)

    def forward(self, x: Tensor) -> Tensor:
        seasonal, trend = self.decomposition(x)
        out = self.seasonal(seasonal.permute(0, 2, 1)) + self.trend(
            trend.permute(0, 2, 1)
        )
        return out.permute(0, 2, 1)


# -- PatchTST ----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PatchTSTConfig:
    """Shared patched encoder with BatchNorm and residual attention. The supplied tensor has K channels; target-only training uses just the target. RevIN affine is disabled, padding is absent, head dropout is zero."""

    seq_len: int = SEQ_LEN
    pred_len: int = PRED_LEN
    #: Root §7's committed geometry. ``(96 - 16) / 8 + 1 = 11`` patches, with no
    #: end-padding patch: the published option that adds one is a convenience for
    #: lookbacks the stride does not divide evenly, and 96 is not one of those.
    patch_len: int = 16
    stride: int = 8
    d_model: int = 128
    d_ff: int = 256
    e_layers: int = 2
    n_heads: int = 8
    dropout: float = 0.1
    #: Reversible instance normalisation, as published. It is the **same**
    #: operation ``use_norm=True`` applies in :class:`ITransformer` — per window,
    #: per channel — so the two models are normalised alike and root §6.3's
    #: cross-model scale consistency holds.
    revin: bool = True
    loss_channels: str = "target"
    channel_independent: bool = True

    @property
    def n_patches(self) -> int:
        return (self.seq_len - self.patch_len) // self.stride + 1

    lr: float = 1e-3
    max_epochs: int = 120
    patience: int = 12
    lr_halve_every: int = 20

    def schedule(self) -> "TrainSchedule":
        return TrainSchedule(lr=self.lr, max_epochs=self.max_epochs,
                             patience=self.patience, lr_halve_every=self.lr_halve_every)

    def build(self) -> "PatchTST":
        return PatchTST(self)

    def loss_target(self) -> str:
        return "all" if self.loss_channels == "all" else "target"

    def fit(
        self,
        tensors: OriginTensors,
        spec: RunSpec,
        *,
        device: torch.device | None = None,
    ) -> tuple["PatchTST", "PatchTSTConfig", TrainOutcome]:
        """Root §6.2's schedule; nothing is selected, so the config returns as given."""
        model, outcome = train_one(tensors, spec, self, device=device)
        return model, self, outcome




class PatchEncoderLayer(nn.Module):
    """PatchTST post-norm encoder with BatchNorm and residual attention scores.

    Follows the official supervised backbone configuration (GELU, post-norm,
    residual attention); no reuse of the iTransformer layer. Attention dropout
    is zero, residual/FFN dropout is cfg.dropout. RevIN affine is disabled.
    """
    def __init__(self, cfg: PatchTSTConfig):
        super().__init__()
        self.heads = cfg.n_heads
        self.width = cfg.d_model // cfg.n_heads
        if cfg.d_model % cfg.n_heads:
            raise ValueError("PatchTST d_model must divide into n_heads")
        self.q = nn.Linear(cfg.d_model, cfg.d_model)
        self.k = nn.Linear(cfg.d_model, cfg.d_model)
        self.v = nn.Linear(cfg.d_model, cfg.d_model)
        self.out = nn.Linear(cfg.d_model, cfg.d_model)
        self.norm1 = nn.BatchNorm1d(cfg.d_model)
        self.norm2 = nn.BatchNorm1d(cfg.d_model)
        self.projection_dropout = nn.Dropout(cfg.dropout)
        self.dropout = nn.Dropout(cfg.dropout)
        self.ffn = nn.Sequential(nn.Linear(cfg.d_model, cfg.d_ff), nn.GELU(),
                                 nn.Dropout(cfg.dropout), nn.Linear(cfg.d_ff, cfg.d_model))

    def forward(self, x: Tensor, previous_scores: Tensor | None = None):
        b, n, d = x.shape
        def heads(layer):
            return layer(x).reshape(b, n, self.heads, self.width).transpose(1, 2)
        q, k, v = heads(self.q), heads(self.k), heads(self.v)
        scores = q @ k.transpose(-2, -1) / self.width**.5
        if previous_scores is not None:
            scores = scores + previous_scores
        context = (scores.softmax(dim=-1) @ v).transpose(1, 2).reshape(b, n, d)
        x = self.norm1((x + self.dropout(self.projection_dropout(self.out(context)))).transpose(1, 2)).transpose(1, 2)
        x = self.norm2((x + self.dropout(self.ffn(x))).transpose(1, 2)).transpose(1, 2)
        return x, scores


class PatchTST(BaselineModule):
    """``(B, L, N) -> (B, H, N)``, each channel processed as its own sequence."""

    def __init__(self, cfg: PatchTSTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        if (cfg.seq_len - cfg.patch_len) % cfg.stride:
            raise ValueError(
                f"seq_len {cfg.seq_len} and patch_len {cfg.patch_len} leave "
                f"{(cfg.seq_len - cfg.patch_len) % cfg.stride} bars uncovered at "
                f"stride {cfg.stride}. Dropping the tail of every window would "
                f"make this model's lookback shorter than the ladder's, and the "
                f"comparison would no longer be on identical information."
            )
        self.embedding = nn.Linear(cfg.patch_len, cfg.d_model)
        self.position = nn.Parameter(torch.zeros(cfg.n_patches, cfg.d_model))
        nn.init.uniform_(self.position, -0.02, 0.02)
        self.dropout = nn.Dropout(cfg.dropout)
        self.layers = nn.ModuleList(PatchEncoderLayer(cfg) for _ in range(cfg.e_layers))
        self.head = nn.Linear(cfg.n_patches * cfg.d_model, cfg.pred_len)

    def forward(self, x: Tensor) -> Tensor:
        b, length, n = x.shape
        mean = std = None
        if self.cfg.revin:
            mean = x.mean(dim=1, keepdim=True).detach()
            x = x - mean
            std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
            x = x / std

        # (B, L, N) -> (B*N, L). Folding the channels into the batch **is**
        # channel independence: from here on nothing in the network sees two
        # variates at once, which is the property under test.
        series = x.permute(0, 2, 1).reshape(b * n, length)
        patches = series.unfold(1, self.cfg.patch_len, self.cfg.stride)
        h = self.dropout(self.embedding(patches) + self.position)
        scores = None
        for layer in self.layers:
            h, scores = layer(h, scores)
        out = self.head(h.transpose(1, 2).reshape(b * n, -1)).reshape(b, n, self.cfg.pred_len)
        out = out.permute(0, 2, 1)

        if self.cfg.revin:
            out = out * std[:, 0, :].unsqueeze(1) + mean[:, 0, :].unsqueeze(1)
        return out


# -- LSTM --------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LSTMConfig:
    """Two layers, hidden 128, dropout 0.1 — root §7, adopted not tuned (`D38`).

    **Multivariate, not channel-independent, and the distinction is the point.**
    DLinear and PatchTST wear their K=8 label through an all-channel objective
    with shared weights: they are *trained on* eight channels but predict the
    target from its own history alone (`D56`). An LSTM reads all K channels of
    every timestep and emits the target directly, so its K=8 means what ridge's
    and iTransformer's mean. That makes it the only recurrent point of comparison
    on the same information set the ladder uses.

    It is here because it is the model the crypto forecasting literature this
    paper argues against reaches for first. Claiming "no deep model beats
    Naive-RW" while leaving the most-cited deep model untested is a hole a
    reviewer finds in one pass.
    """

    seq_len: int = SEQ_LEN
    pred_len: int = PRED_LEN
    k: int = 8
    hidden: int = 128
    layers: int = 2
    dropout: float = 0.1
    #: Target-channel, like the ladder (`D39`) and unlike the two
    #: channel-independent baselines. Logged so a reader of Table 3 can see that
    #: this model's ``best_val_mse`` *is* comparable to the ladder's.
    loss_channels: str = "target"
    channel_independent: bool = False

    def build(self) -> "LSTMForecaster":
        return LSTMForecaster(self)

    def loss_target(self) -> str:
        return "target"

    def fit(
        self,
        tensors: OriginTensors,
        spec: RunSpec,
        *,
        device: torch.device | None = None,
    ) -> tuple["LSTMForecaster", "LSTMConfig", TrainOutcome]:
        """Root §6.2's schedule; nothing is selected, so the config returns as given."""
        model, outcome = train_one(tensors, spec, self, device=device)
        return model, self, outcome


class LSTMForecaster(nn.Module):
    """``(B, L, K) -> (B, H)`` from the last hidden state.

    No instance normalisation, as is standard for this baseline — root §6.3's
    outer ``StandardScaler`` is what serves that case, so this model reads the
    same scaler space as every other and needs nothing of its own.
    """

    def __init__(self, cfg: LSTMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.lstm = nn.LSTM(
            input_size=cfg.k,
            hidden_size=cfg.hidden,
            num_layers=cfg.layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.layers > 1 else 0.0,
        )
        self.head = nn.Linear(cfg.hidden, cfg.pred_len)

    def forward(self, x: Tensor) -> Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])

    def forecast_target(self, x: Tensor) -> Tensor:
        return self(x)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# -- the two naive comparators -----------------------------------------------


#: Hours in the seasonal cycle a daily pattern would repeat on.
SEASONAL_PERIOD: int = 24


@dataclass(frozen=True, slots=True)
class NaiveConfig:
    """``persist`` and ``seasonal`` — root §7's two secondary naive comparators.

    Neither trains and neither has a parameter, so both cost microseconds and
    exist purely to close a hole: root §7 listed them and `D56` recorded, in
    writing, that nobody had built them. Two rows marked *deferred* in a results
    table read as unfinished work, and these are the cheapest rows in the study.

    Both read the target channel and nothing else, so their honest K is **1**
    (`D40` requires the label; it does not require the label to be large). They
    are distinct from Naive-RW, which forecasts ``y_hat_raw = 0`` and needs no run
    at all (`D31`):

    - ``persist`` repeats the last observed return for all H steps. Root §7 calls
      it a weaker baseline than Naive-RW and that is the expected result; the
      point is to show it rather than assert it.
    - ``seasonal`` repeats the return one daily cycle back, step for step. It is
      the comparator for any claim that hourly crypto carries a daily pattern.
    """

    mode: str = "persist"
    seq_len: int = SEQ_LEN
    pred_len: int = PRED_LEN
    k: int = 1
    loss_channels: str = "target"
    channel_independent: bool = False

    def build(self) -> "NaiveForecaster":
        return NaiveForecaster(self)

    def loss_target(self) -> str:
        return "target"

    def fit(
        self,
        tensors: OriginTensors,
        spec: RunSpec,
        *,
        device: torch.device | None = None,
    ) -> tuple["NaiveForecaster", "NaiveConfig", TrainOutcome]:
        """No fit. The two split losses are still measured and reported.

        ``epochs_run=0`` is the honest number and it is what tells a reader of
        Table 3 why these rows have no epochs-to-stop, exactly as for ridge.
        """
        device = device or pick_device()
        # Consumes no RNG. Seeded anyway so every arm is reproducible under one
        # rule rather than two (root §16).
        started = time.perf_counter()

        with SEED_LOCK:
            set_seed(spec.seed, device)
            model = self.build().to(device)

        def split_mse(split) -> float:
            x = torch.from_numpy(split.x).to(device)
            y = torch.from_numpy(split.y).to(device)
            with torch.no_grad():
                return float((model.forecast_target(x) - y).pow(2).mean())

        return (
            model,
            self,
            TrainOutcome(
                run_id=spec.run_id,
                epochs_run=0,
                best_val_mse=split_mse(tensors.val),
                train_loss=split_mse(tensors.train),
                wall_time_s=time.perf_counter() - started,
                n_parameters=0,
                device=str(device),
            ),
        )


class NaiveForecaster(nn.Module):
    """``(B, L, K) -> (B, H)`` by copying a past value of the target channel.

    Stateless: no parameters, no buffers, nothing to move to a device beyond the
    module shell. ``n_parameters`` returns 0 and that is the truthful figure —
    unlike ridge, where the coefficients are buffers and the usual sum would
    under-report (root §12).
    """

    def __init__(self, cfg: NaiveConfig) -> None:
        super().__init__()
        if cfg.mode not in ("persist", "seasonal"):
            raise ValueError(f"unknown naive mode {cfg.mode!r}")
        self.cfg = cfg

    def forward(self, x: Tensor) -> Tensor:
        target = x[:, :, TARGET_INDEX]
        if self.cfg.mode == "persist":
            return target[:, -1:].expand(-1, self.cfg.pred_len)
        length = target.shape[1]
        # Modulo, so a horizon longer than one cycle repeats the cycle instead of
        # indexing past the lookback. At the H=24 arms it is the identity.
        index = [
            length - SEASONAL_PERIOD + (h % SEASONAL_PERIOD)
            for h in range(self.cfg.pred_len)
        ]
        return target[:, index]

    def forecast_target(self, x: Tensor) -> Tensor:
        return self(x)

    def n_parameters(self) -> int:
        return 0


# -- the `D45` assertion -----------------------------------------------------


def assert_baseline_alignment(
    baseline_run_id: str, reference_run_id: str, roots: list[Path]
) -> None:
    """`D45` — a baseline may only be scored on its comparator's exact windows.

    Root §7: "Baselines are scored on exactly the same surviving windows." Unless
    that holds, RelMSE is a ratio across two samples rather than a ratio, and the
    two samples would differ systematically rather than randomly: test-window
    survival is conditioned on *future* gaps (root §4.3) and Binance outages
    cluster on stress, so the windows one model kept and the other dropped are
    disproportionately the high-volatility ones.

    Here the sets are equal by construction — both come from
    :func:`itransformer_btc.splits.window_starts` with the same origin, span and
    ``"origin"`` semantics — and that is exactly why the assertion is cheap, and
    why it is the only thing that would notice if it ever stopped being true.
    Root §4.3 names positional-index drift after a row drop as the
    highest-probability silent bug in this pipeline; this is its detector on the
    cross-model axis.

    Raises:
        ValueError: If the evaluated ``(block, timestamp)`` sets differ.
        FileNotFoundError: If either run is absent from ``roots``.
    """
    assert_same_windows(
        load_predictions(baseline_run_id, roots),
        load_predictions(reference_run_id, roots),
        f"{baseline_run_id} vs {reference_run_id}",
    )
