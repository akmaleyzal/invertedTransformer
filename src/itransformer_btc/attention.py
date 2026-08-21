"""Figure 5's input: attention maps, binned by volatility regime.

`D60g`: attention weights were never persisted. ``preds/`` holds forecasts only,
so Figure 5 and §13.2's interpretability claim rested on ``A_attn`` alone --- a
scalar that says whether attention helps, never what it attends to. This module
captures the maps; ``runner``'s ``attention`` arm re-runs the K=8 cells to
produce them.

**The regimes are data-determined** (`D48`): calm is the bottom tercile and
stress the top tercile of realised volatility across all of that origin's test
blocks. Picking the windows after seeing the maps would make the paper's
interpretability claim a free parameter, which is the same defect §3 forbids for
tau and `D49` forbids for the equivalence margin.

**Volatility is measured on the lookback, not on the forecast period.** The
attention map is a deterministic function of the 96-bar input window, so the
conditioning variable has to be a property of that same window or the figure
plots a map against something it does not depend on. The measure is the standard
deviation of the target channel over the lookback, in scaler space, which is
what ``use_norm`` divides by before the embedding sees anything.

**Attention is not explanation.** Root §13.2 admits these maps only as
*descriptive evidence of variate reliance*, validated for stability across seeds
and paired with the uniform-attention ablation (Jain & Wallace 2019; Wiegreffe &
Pinter 2019) --- which is why the arm carries three seeds rather than one, and
why ``A_attn`` stays in the paper beside the figure. The debate is also scoped to
RNN-era NLP and its transfer to variate-level attention in long-term forecasting
is genuinely open; that is itself a limitation sentence, not a footnote.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import polars as pl
import torch

from itransformer_btc.model import ITransformer
from itransformer_btc.splits import OriginTensors, SplitTensors

#: Tercile labels, coldest first. Root §13.4 names calm and stress; the middle
#: band is carried because a figure showing only the extremes invites a reader to
#: assume the relationship between them is monotone.
TERCILES: Final[tuple[str, str, str]] = ("calm", "mid", "stress")

#: Inference batch. Larger than training's 32 because nothing here backpropagates.
CAPTURE_BATCH: Final = 512


def lookback_volatility(split: SplitTensors, target_index: int = 0) -> np.ndarray:
    """Standard deviation of the target channel over each window's lookback.

    Scaler-space, so the figure's regimes are the regimes ``use_norm`` sees.
    """
    if len(split) == 0:
        return np.empty(0, dtype=np.float64)
    return split.x[:, :, target_index].std(axis=1).astype(np.float64)


def tercile_edges(volatility: np.ndarray) -> tuple[float, float]:
    """The 33.3rd and 66.7th percentiles, computed **across all test blocks**.

    Per-block edges would make "stress" mean a different thing in each block and
    the panels incomparable, which is the opposite of what the figure is for.
    """
    return float(np.quantile(volatility, 1 / 3)), float(np.quantile(volatility, 2 / 3))


def _capture_batch(model: ITransformer, x: torch.Tensor) -> list[np.ndarray]:
    """One forward pass with capture on; per-layer ``(B, N, N)`` weight arrays.

    The flag is cleared in a ``finally`` so an exception cannot leave a model
    quietly accumulating detached tensors for the rest of a session.
    """
    for layer in model.layers:
        layer.attention.capture = True
    try:
        with torch.no_grad():
            model(x)
        return [
            layer.attention.last_weights.cpu().numpy().astype(np.float64)
            for layer in model.layers
        ]
    finally:
        for layer in model.layers:
            layer.attention.capture = False
            layer.attention.last_weights = None


def tercile_maps(
    model: ITransformer,
    tensors: OriginTensors,
    device: torch.device,
    target_index: int = 0,
    batch: int = CAPTURE_BATCH,
) -> pl.DataFrame:
    """Mean attention map per (tercile, layer) over one origin's test blocks.

    Two passes, because the tercile edges are a property of the whole test span
    and a single streaming pass would have to guess them: the first collects the
    volatility of every test window, the second accumulates maps into the bins
    those volatilities define.

    Args:
        model: A trained iTransformer. Set to ``eval`` here, so dropout is off
            and the captured weights are the ones inference actually uses.
        tensors: The origin whose test blocks to sweep.
        device: Where the model lives.
        target_index: Channel the volatility is measured on. 0 is ``r``.
        batch: Inference batch size.

    Returns:
        ``tercile, layer, i, j, weight, n_windows, vol_low, vol_high`` --- one row
        per (tercile, layer, variate pair). At K=8 with two layers that is 384
        rows, so persisting it costs nothing beside ``preds/``.

    Raises:
        ValueError: If the origin has no test window at all, which would leave
            the terciles undefined rather than merely empty.
    """
    model.eval()
    splits = [s for s in tensors.test_blocks if len(s) > 0]
    if not splits:
        raise ValueError(f"origin {tensors.origin.label} has no test window to sweep")

    volatility = np.concatenate([lookback_volatility(s, target_index) for s in splits])
    low, high = tercile_edges(volatility)

    n_layers = len(model.layers)
    n_variates = tensors.k
    totals = np.zeros((len(TERCILES), n_layers, n_variates, n_variates))
    counts = np.zeros(len(TERCILES), dtype=np.int64)

    for split in splits:
        # Two edges, three bins: below low, between, at or above high.
        bins = np.digitize(lookback_volatility(split, target_index), [low, high])
        x_all = torch.from_numpy(split.x).to(device)
        for start in range(0, len(split), batch):
            weights = _capture_batch(model, x_all[start : start + batch])
            chunk_bins = bins[start : start + batch]
            for tercile in range(len(TERCILES)):
                mask = chunk_bins == tercile
                if not mask.any():
                    continue
                counts[tercile] += int(mask.sum())
                for layer, w in enumerate(weights):
                    totals[tercile, layer] += w[mask].sum(axis=0)

    rows: list[dict[str, float | int | str]] = []
    for tercile, name in enumerate(TERCILES):
        if counts[tercile] == 0:
            continue
        for layer in range(n_layers):
            mean_map = totals[tercile, layer] / counts[tercile]
            for i in range(n_variates):
                for j in range(n_variates):
                    rows.append(
                        {
                            "tercile": name,
                            "layer": layer,
                            "i": i,
                            "j": j,
                            "weight": float(mean_map[i, j]),
                            "n_windows": int(counts[tercile]),
                            "vol_low": low,
                            "vol_high": high,
                        }
                    )
    return pl.DataFrame(rows)
