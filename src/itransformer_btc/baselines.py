"""Ridge, DLinear and PatchTST — the comparators root §7 calls mandatory (`D56`).

Until the 534-run grid finished on 2026-08-08 this module did not exist, and
neither did any other baseline class. Root §7 says DLinear and PatchTST are "not
optional", root §10.2 budgets 255 baseline runs, and
:func:`itransformer_btc.metrics.dm_nonnested` has been sitting ready for input
that was never produced — so §10.2's 789 was never executable and Table 6 had no
inputs. The consequence is not bookkeeping: with Naive-RW the only comparator,
"iTransformer has no edge" rests on one contrast, and §6.2/`D38` says the
hyperparameters were adopted unchanged and never tuned. A referee reads that null
as an immature configuration rather than a finding. These three models are the
minimum that answers the question deciding what the negative result means:
**did iTransformer fail, or did the whole LTSF class fail here?**

What is built, and what each one is for:

=========  ==========  =====================================================
Model      K           Question it answers
=========  ==========  =====================================================
Ridge      1, 4, 8, 12 `D17` — is a transformer needed at all? Linear and
                       genuinely multivariate, so it separates *does the
                       information help* from *does attention help*
DLinear    8           root §7 — the first thing an LTSF-literate reviewer
                       looks for. Linear, decomposition-based, ~4.7k weights
PatchTST   8           root §7 — SOTA and channel-independent, the other side
                       of the debate §13.1 makes a Related Work pillar
=========  ==========  =====================================================

**What "K = 8" means for a channel-independent model, stated because it is not
what it means elsewhere in this study.** DLinear and PatchTST forecast each
channel from that channel's own history; that is the architecture's claim, not a
shortcoming of this implementation. They are therefore trained with their
published **all-channel** objective and their weights are **shared across
channels**, which is the only route by which the other seven variates reach the
target's forecast at all. Trained on the target channel alone they would be K=1
wearing a K=8 label — the exact collapse `D40` was written to prevent — and the
paper's central architectural comparison would quietly become
univariate-versus-multivariate again. Both facts are recorded as fields in every
``meta/*.json`` (``loss_channels``, ``channel_independent``) so no reader has to
infer them. Ridge carries no such caveat: every one of its ``L x K`` inputs
enters the target's forecast, so its K label means what K means in §5.2.

**Capacity is held fixed rather than tuned.** PatchTST takes iTransformer's
``d_model``, ``d_ff``, ``e_layers``, ``n_heads`` and ``dropout`` and reuses
:class:`itransformer_btc.model.EncoderLayer` itself, so the two models differ in
**what a token is** — a patch of one variate against the whole lookback of each
variate — and in nothing else. That is the cleanest available form of the
contrast, and it extends §6.2/`D38`'s no-tuning posture to the baselines instead
of quietly exempting them.

**Scale space is identical to the ladder's** (root §6.3, §11). Every model reads
the same ``StandardScaler`` output fitted on the same 21-month sub-block. What
differs is internal normalisation, and it differs the way the published models
do: PatchTST normalises per window (RevIN — the same operation ``use_norm=True``
applies), DLinear and ridge do not, which is precisely the case §6.3 says the
outer scaler exists to serve.

**Not built here, and not silently dropped.** ARIMA, LSTM, naive-persist and
seasonal-naive are deferred from this minimal set. Naive-RW needs no run at all —
:func:`itransformer_btc.metrics.block_metrics` computes it from ``naive_rw_z`` on
exactly the rows the model was scored on. If the other four are eventually cut,
root §7 must be edited with a written reason rather than left standing over
models nobody built.
"""

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
    set_seed,
    train_one,
)


class BaselineModule(nn.Module):
    """Shared plumbing for the two channel-independent baselines.

    ``forward`` returns all N channels because that is what the published
    objective supervises; ``forecast_target`` is the projection root §10.4's
    prediction file actually holds. Channel ``TARGET_INDEX`` is ``r`` at every
    rung — the ladder pins it there, see
    :data:`itransformer_btc.features.VARIATE_ORDER` — so this is one constant
    rather than a lookup.
    """

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forecast_target(self, x: Tensor) -> Tensor:
        """``(B, L, N) -> (B, H)`` on the target channel."""
        return self(x)[:, :, TARGET_INDEX]


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
        set_seed(spec.seed)
        started = time.perf_counter()

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
    """Trend-seasonal decomposition plus two linear maps (Zeng et al., 2023).

    Root §7 calls it mandatory for a reason worth stating: a missing DLinear is
    the first thing a reviewer familiar with the LTSF literature flags, because
    it is the model that showed a linear map beating several transformers on the
    standard benchmarks. Against a null result it does more than that — if ~4.7k
    weights also fail here, the failure belongs to the problem and not to
    attention.

    Weights are **shared across channels**, never per-channel. The published
    implementation offers both; only the shared form lets the all-channel
    objective carry information from the other seven variates into the target's
    forecast, which is what makes the K=8 label true (see this module's header).
    """

    seq_len: int = SEQ_LEN
    pred_len: int = PRED_LEN
    #: Odd, so the decomposition's padding is symmetric. 25 is the published
    #: default and is not tuned here (`D38`, extended to the baselines).
    moving_avg: int = 25
    #: Recorded in ``meta/*.json`` rather than left to be inferred: see the
    #: header. A reader who does not know the objective cannot read
    #: ``best_val_mse``, which is an all-channel figure for this model and a
    #: target-channel one for the ladder.
    loss_channels: str = "all"
    channel_independent: bool = True

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
    """Patched, channel-independent transformer (Nie et al., 2023).

    The other side of the channel-independence debate root §13.1 makes a Related
    Work pillar, and the comparison `D40` says an LTSF-literate reviewer wants:
    iTransformer at K=8 against PatchTST at K=8, on the same information, the same
    windows and the same scale space.

    Capacity is iTransformer's, field for field, and the encoder block is
    literally :class:`itransformer_btc.model.EncoderLayer`. The two models
    therefore differ in **what a token is** — a patch of one variate here, one
    variate's entire lookback there — and in nothing else. Root §6.2/`D38`'s
    no-tuning rule applies unchanged.
    """

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
    loss_channels: str = "all"
    channel_independent: bool = True

    @property
    def n_patches(self) -> int:
        return (self.seq_len - self.patch_len) // self.stride + 1

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
        # EncoderLayer reads d_model, d_ff, n_heads, dropout and
        # uniform_attention; its remaining fields are inert here and stay at
        # their defaults. Reusing the block rather than reimplementing it is what
        # makes "same capacity, different tokenisation" a fact and not a claim.
        block = ITransformerConfig(
            d_model=cfg.d_model,
            d_ff=cfg.d_ff,
            n_heads=cfg.n_heads,
            dropout=cfg.dropout,
        )
        self.embedding = nn.Linear(cfg.patch_len, cfg.d_model)
        self.position = nn.Parameter(torch.zeros(cfg.n_patches, cfg.d_model))
        nn.init.normal_(self.position, std=0.02)
        self.dropout = nn.Dropout(cfg.dropout)
        self.layers = nn.ModuleList(EncoderLayer(block) for _ in range(cfg.e_layers))
        self.head = nn.Linear(cfg.n_patches * cfg.d_model, cfg.pred_len)

    def forward(self, x: Tensor) -> Tensor:
        b, length, n = x.shape
        mean = std = None
        if self.cfg.revin:
            mean = x.mean(dim=1, keepdim=True)
            x = x - mean
            std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + 1e-5)
            x = x / std

        # (B, L, N) -> (B*N, L). Folding the channels into the batch **is**
        # channel independence: from here on nothing in the network sees two
        # variates at once, which is the property under test.
        series = x.permute(0, 2, 1).reshape(b * n, length)
        patches = series.unfold(1, self.cfg.patch_len, self.cfg.stride)
        h = self.dropout(self.embedding(patches) + self.position)
        for layer in self.layers:
            h = layer(h)
        out = self.head(h.reshape(b * n, -1)).reshape(b, n, self.cfg.pred_len)
        out = out.permute(0, 2, 1)

        if self.cfg.revin:
            out = out * std[:, 0, :].unsqueeze(1) + mean[:, 0, :].unsqueeze(1)
        return out


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
