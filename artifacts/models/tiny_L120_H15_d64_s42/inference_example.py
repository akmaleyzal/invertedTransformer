# Runnable inference example for the exported iTransformer bundle.
#
# INPUT CONTRACT
#     shape  (B, L, N) float32
#     values already scaled: x = (winsorise(raw) - scaler.mean) / scaler.std
#     order  exactly feature_manifest.json["feature_order"] - assert it, never assume it
#     L, N   from feature_manifest.json (seq_len, n_variates)
#
# OUTPUT CONTRACT
#     shape  (B, H, 1) float32, in STANDARDISED log-return space
#     to raw one-minute log returns:  r = out[..., 0] * std[target_index] + mean[target_index]
#     to an h-minute cumulative return: r[:, :h].sum(axis=1)
#     to a price path: price_{t+k} = close_t * exp(cumsum(r)[k])
#
# STALENESS POLICY
#     gold      may be forward-filled up to 3 days (a normal weekend). Beyond that, refuse.
#     usd index may be forward-filled up to 5 business days.
#     macro     may be forward-filled up to 45 days past its release date.
#     Exceeding any of these means the staleness features leave their training support,
#     and the prediction should be refused rather than served.

import json
from pathlib import Path

import numpy as np
import torch

BUNDLE = Path(__file__).parent
manifest = json.loads((BUNDLE / "feature_manifest.json").read_text())
scaler = json.loads((BUNDLE / "scaler.json").read_text())

L, N = manifest["seq_len"], manifest["n_variates"]
H = manifest["pred_len"]
ti = manifest["target_index"]
mean = np.asarray(scaler["mean"], dtype=np.float32)
std = np.asarray(scaler["std"], dtype=np.float32)


def scale(raw: np.ndarray, feature_order: list) -> np.ndarray:
    # raw: (B, L, N) unscaled, columns in `feature_order`.
    assert list(feature_order) == manifest["feature_order"], (
        "feature order mismatch - the model will produce plausible-looking garbage"
    )
    lo = np.asarray([v if v is not None else -np.inf for v in scaler["winsor_lo"]], np.float32)
    hi = np.asarray([v if v is not None else np.inf for v in scaler["winsor_hi"]], np.float32)
    return ((np.clip(raw, lo, hi) - mean) / std).astype(np.float32)


model = torch.jit.load(str(BUNDLE / "model_scripted.pt"))
model.eval()

x = torch.randn(1, L, N)                      # substitute real, scaled features here
with torch.no_grad():
    out = model(x)                            # (1, H, 1) standardised log returns

r = out[..., 0].numpy() * std[ti] + mean[ti]  # raw one-minute log returns
print("output", tuple(out.shape))
for h in (1, 5, 15, 30, 60):
    if h <= H:
        print(f"  cumulative {h:>3}-min log return: {r[0, :h].sum():+.6f}")

close_t = 60_000.0
print(f"  implied price path from {close_t:,.0f}: "
      f"{(close_t * np.exp(np.cumsum(r[0])))[[0, min(H, 60) - 1]].round(2).tolist()}")
