"""Read-only checks behind the 2026-09-09 research workflow review.

Run from the repository: .venv/Scripts/python.exe .research/research_audit_2026_09_09.py
Writes only its own JSON evidence; does not train or change study artifacts.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
import torch
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from itransformer_btc.config import ORIGINS
from itransformer_btc.economics import positions, summarise
from itransformer_btc.features import build_features
from itransformer_btc.metrics import decay, directional_accuracy, j_test, minimum_detectable_beta1
from itransformer_btc.segments import load_bars, usable_mask
from itransformer_btc.runner import RunCell, execute, pending
from itransformer_btc.model import ITransformer, ITransformerConfig

ART = ROOT / "notebooks/outputs/artifacts"
HOUR = 3_600_000


def iso(ms: int) -> str:
    return datetime.fromtimestamp(int(ms) / 1000, timezone.utc).isoformat()


def read_run(tag: str, origin: int = 1, seed: int = 42, k: int = 8):
    name = f"{tag}_o{origin:02d}_K{k:02d}_H024_s{seed}"
    meta = json.loads((ART / "meta" / f"{name}.json").read_text())
    pred = pl.read_parquet(ART / "preds" / f"{name}.parquet").sort(
        ["block", "timestamp", "step"]
    )
    return meta, pred


def run() -> dict:
    report = json.loads((ROOT / "paper/paper_numbers.json").read_text())
    metas = [json.loads(p.read_text()) for p in (ART / "meta").glob("*.json")]
    metas = [m for m in metas if "run_id" in m]
    features = build_features(usable_mask(load_bars(ROOT / "data/raw/BTCUSDT_1h.parquet")))
    raw_by_ts = dict(features.select("ts_ms", "r").iter_rows())
    evidence = {
        "runs": len(metas),
        "status": dict(Counter(m["status"] for m in metas)),
        "code_hashes": dict(Counter(m["code_sha256"] for m in metas)),
        "input_hashes": dict(Counter(m["input_sha256"] for m in metas)),
        "actual_input_hash": hashlib.sha256((ROOT / "data/raw/BTCUSDT_1h.parquet").read_bytes()).hexdigest(),
        "grid_json_hash": hashlib.sha256((ART / "paper_numbers.json").read_bytes()).hexdigest(),
        "paper_derived_from": report["derived_from"],
        "paper_runs_complete": report["runs_complete"],
        "n_train_main": sorted({m["n_train"] for m in metas if m["spec"]["model"] == "itr" and m["spec"]["pred_len"] == 24}),
    }

    dates, targets = {}, {}
    for tag in ("l048", "itr", "l192"):
        meta, pred = read_run(tag)
        row = pred.row(0, named=True)
        actual_ts = int(row["timestamp"]) + (meta["config"]["seq_len"] + int(row["step"]) - 1) * HOUR
        raw_target = row["y_true"] * meta["sigma_g"] + meta["mu_g"]
        assert np.isclose(raw_target, raw_by_ts[actual_ts], atol=1e-8)
        final_b1 = pred.filter(pl.col("block") == 1).row(-1, named=True)
        dates[tag] = {
            "stored_timestamp": iso(row["timestamp"]),
            "first_target_bar": iso(actual_ts),
            "last_target_bar_block1": iso(final_b1["timestamp"] + (meta["config"]["seq_len"] + final_b1["step"] - 1) * HOUR),
            "raw_target_first_row": raw_target,
            "origin": ORIGINS[0].origin.isoformat(),
            "nominal_block1_end": ORIGINS[0].block(1)[1].isoformat(),
        }
        targets[tag] = pred.select("block", "timestamp", "step").with_columns(
            pl.lit(meta["config"]["seq_len"]).alias("L")
        )
    evidence["time_alignment"] = dates
    evidence["training_cutoff_origin1"] = ORIGINS[0].train_sub_end.isoformat()

    avg = pl.read_parquet(ART / "seed_averaged_cells.parquet")
    keff = pl.read_parquet(ART / "keff_table.parquet")
    main = avg.filter((pl.col("model") == "itr") & (pl.col("pred_len") == 24))
    race = main.join(keff.select("origin", "k", "pr_raw"), on=["origin", "k"]).sort(["origin_index", "block", "k"])
    group = race["origin_index"].to_numpy() * 100 + race["block"].to_numpy()
    cluster = race["origin_index"].to_numpy()
    y = race["mse"].to_numpy().astype(float)
    results = {}
    for a_name, b_name in (("k", "pr_raw"), ("pr_raw", "k")):
        a, b = race[a_name].to_numpy().astype(float), race[b_name].to_numpy().astype(float)
        legacy_t, legacy_p = j_test(y, a, b, group)
        yd, ad, bd = y.copy(), a.copy(), b.copy()
        for g in np.unique(group):
            keep = group == g
            yd[keep] -= y[keep].mean()
            ad[keep] -= a[keep].mean()
            bd[keep] -= b[keep].mean()
        fitted_b = bd * (bd @ yd / (bd @ bd))
        design = np.column_stack([ad, fitted_b])
        coef = np.linalg.lstsq(design, yd, rcond=None)[0]
        residual = yd - design @ coef
        bread = np.linalg.pinv(design.T @ design)
        scores = np.stack([design[cluster == c].T @ residual[cluster == c] for c in np.unique(cluster)])
        n, g, p = len(y), len(np.unique(cluster)), len(np.unique(group)) + 2
        covariance = g / (g - 1) * (n - 1) / (n - p) * bread @ (scores.T @ scores) @ bread
        clustered_t = float(coef[1] / np.sqrt(covariance[1, 1]))
        results[f"{a_name}_augmented_by_{b_name}"] = {
            "implemented_t": legacy_t, "implemented_p": legacy_p,
            "origin_CR1_t_diagnostic": clustered_t,
            "origin_CR1_p_t14_diagnostic": float(2 * stats.t.sf(abs(clustered_t), g - 1)),
        }
    evidence["j_test"] = results
    slopes = np.array(report["rq2"]["within_slopes"])
    evidence["mde"] = {
        "published": report["rq2"]["minimum_detectable_beta1"],
        "recomputed_from_test_slopes": minimum_detectable_beta1(slopes),
    }
    assert math.isclose(evidence["mde"]["published"], evidence["mde"]["recomputed_from_test_slopes"], abs_tol=1e-12)

    seed_results = []
    for origin in (1, 8, 15):
        preds = [read_run("itr", origin, seed)[1] for seed in range(42, 47)]
        y = preds[0]["y_true"].to_numpy().astype(float)
        f = np.stack([p["y_pred"].to_numpy().astype(float) for p in preds])
        mean_loss = float(np.square(f - y).mean())
        ensemble_loss = float(np.square(f.mean(axis=0) - y).mean())
        variance = float(f.var(axis=0).mean())
        assert np.isclose(mean_loss - ensemble_loss, variance)
        seed_results.append({"origin": origin, "mean_seed_MSE": mean_loss, "ensemble_MSE": ensemble_loss, "difference_pct_of_mean_seed_MSE": 100 * variance / mean_loss})
    evidence["seed_estimands"] = seed_results

    torch.manual_seed(42)
    net_model = ITransformer(ITransformerConfig()).eval()
    sample_x = torch.randn(2, 96, 1)
    shift = 0.2
    with torch.no_grad():
        shift_error = float((net_model(sample_x + shift) - net_model(sample_x) - shift).abs().max())
    evidence["instance_norm_shift_equivariance"] = {
        "target_window_shift": shift, "max_abs_f_x_plus_shift_minus_f_x_minus_shift": shift_error,
        "implication": "The architectural shift equivariance prevents a globally constant forecast function; nesting of Naive-RW is not implied by additional input variables.",
    }
    assert shift_error < 1e-5

    direction = []
    for origin in (1, 8, 15):
        meta, pred = read_run("itr", origin)
        raw = pred.with_columns((pl.col(c) * meta["sigma_g"] + meta["mu_g"]).alias(c) for c in ("y_true", "y_pred"))
        implemented_da, raw_da = directional_accuracy(pred), directional_accuracy(raw)
        p = positions(pred, meta["sigma_g"], meta["mu_g"])
        raw_sign = np.sign(p["forecast_raw"].to_numpy() + 24 * meta["mu_g"])
        direction.append({
            "origin": origin, "mu_over_sigma": meta["mu_over_sigma"],
            "implemented_da_h1": implemented_da.da_h1, "raw_da_h1": raw_da.da_h1,
            "implemented_da_cum": implemented_da.da_cum, "raw_da_cum": raw_da.da_cum,
            "position_sign_changes_after_inverse_scaler": int(np.sum(p["position"].to_numpy() != raw_sign)),
            "positions": p.height,
        })
    evidence["direction_scale"] = direction
    example_net = np.array([-0.01, -0.01, 0.02, 0.02])
    example = summarise(np.ones(4), example_net + np.array([0.0006, 0, 0, 0]), 0.0002, 0, mdd_interval=False)
    conventional = float(example_net.mean() / np.sqrt(np.mean(np.minimum(example_net, 0) ** 2)) * np.sqrt(365))
    evidence["sortino_counterexample"] = {
        "net_returns": example_net.tolist(),
        "implemented": None if not np.isfinite(example.sortino_annualised) else example.sortino_annualised,
        "zero_MAR_full_sample_downside_deviation": conventional,
    }

    improving = pl.DataFrame({
        "origin": ["synthetic"] * 6, "origin_index": [1] * 6,
        "block": list(range(1, 7)), "model": ["itr"] * 6,
        "k": [8] * 6, "pred_len": [24] * 6, "n_windows": [720] * 6,
        "r2_oos": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
    })
    improving_decay = decay(improving)
    evidence["improving_skill_counterexample"] = {
        "skill": improving["r2_oos"].to_list(),
        "computed_D": improving_decay.table["D"].to_list(),
        "b_star": improving_decay.b_star(0.05).to_dicts(),
    }
    assert improving_decay.b_star(0.05)["b_star"][0] == 1

    spec = importlib.util.spec_from_file_location("audit_notebook_generator", ROOT / "tools/build_notebook.py")
    generator = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = generator
    spec.loader.exec_module(generator)
    def_cell = {"cell_type": "code", "metadata": {"itbtc": {"role": "module", "module": "example", "section": "f"}}, "outputs": []}
    call_cell = {"cell_type": "code", "metadata": {"itbtc": {"role": "step", "step": "example"}}, "source": ["print(f())\n"]}
    previous = {"cells": [{**def_cell, "source": ["def f(): return 1\n"]}, {**call_cell, "execution_count": 2, "outputs": [{"output_type": "stream", "name": "stdout", "text": ["1\n"]}]}]}
    current = {"cells": [{**def_cell, "source": ["def f(): return 2\n"]}, {**call_cell, "outputs": []}]}
    carried = generator.carry_outputs(current, previous)
    evidence["notebook_upstream_change_counterexample"] = {"carried_changed_missing": list(carried), "output_after_upstream_function_changed": current["cells"][1]["outputs"]}
    assert current["cells"][1]["outputs"][0]["text"] == ["1\n"]

    temporary = tempfile.TemporaryDirectory(prefix="audit_resume_", dir=ROOT / ".research")
    scratch = Path(temporary.name).resolve()
    try:
        (scratch / "preds").mkdir()
        (scratch / "meta").mkdir()
        cell = RunCell("main", 1, 8, 24, 42)
        (scratch / "preds" / f"{cell.run_id}.parquet").write_bytes(b"synthetic placeholder")
        (scratch / "meta" / f"{cell.run_id}.json").write_text(json.dumps({"status": "complete", "code_sha256": "superseded-audit-example"}))
        candidates = pending([cell], [scratch])
        outcome = execute(candidates, pl.DataFrame(), out_root=scratch, roots=[scratch], log=lambda _: None)
        evidence["resume_vintage_counterexample"] = {
            "pending_before_execute": len(candidates), "completed": outcome.completed,
            "skipped": outcome.skipped, "remaining": outcome.remaining,
        }
        assert len(candidates) == 1 and outcome.skipped == 1 and outcome.remaining == 1
    finally:
        assert scratch.is_relative_to((ROOT / ".research").resolve())
        temporary.cleanup()

    # Re-score existing forecasts on common actual issuance hours. This is a
    # diagnostic on a reduced sample, not a replacement preregistered result.
    aligned_rows = []
    for origin in range(1, 16):
        common = None
        for tag in ("l048", "itr", "l192"):
            runs = [read_run(tag, origin, seed) for seed in range(42, 47)]
            meta, first = runs[0]
            actual = first["y_true"].to_numpy().astype(float) * meta["sigma_g"] + meta["mu_g"]
            forecasts = np.stack([p["y_pred"].to_numpy().astype(float) for _, p in runs]) * meta["sigma_g"] + meta["mu_g"]
            windows = first.filter(pl.col("step") == 1)["timestamp"].to_numpy() + meta["config"]["seq_len"] * HOUR
            losses = np.square(forecasts - actual).mean(axis=0).reshape(-1, 24).mean(axis=1)
            null_losses = np.square(actual).reshape(-1, 24).mean(axis=1)
            frame = pl.DataFrame({"issue_ts": windows, f"loss_{tag}": losses, f"naive_{tag}": null_losses})
            common = frame if common is None else common.join(frame, on="issue_ts", how="inner")
        assert np.allclose(common["naive_itr"], common["naive_l048"], atol=1e-10)
        assert np.allclose(common["naive_itr"], common["naive_l192"], atol=1e-10)
        ratios = {tag: float(common[f"loss_{tag}"].sum() / common["naive_itr"].sum()) for tag in ("l048", "itr", "l192")}
        aligned_rows.append({"origin": origin, "common_windows": common.height, **ratios})
    evidence["lookback_on_common_actual_targets"] = {
        "per_origin": aligned_rows,
        "mean_delta_l192_minus_itr": float(np.mean([r["l192"] - r["itr"] for r in aligned_rows])),
        "mean_delta_l048_minus_itr": float(np.mean([r["l048"] - r["itr"] for r in aligned_rows])),
        "caveat": "Exploratory intersection of surviving actual issue times; no new training, no official tables replaced.",
    }
    return evidence


if __name__ == "__main__":
    result = run()
    output = ROOT / ".research/research_audit_2026_09_09.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))
