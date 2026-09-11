from pathlib import Path
exec(Path(__file__).with_name('applied_first_batch.py').read_text(encoding='utf-8').split('# A13:')[0])

function('metrics.py', 'gather_grid', '''
def evaluation_windows(run_ids: list[str], roots: list[Path]) -> pl.DataFrame:
    """Common forecast times per origin, horizon and block across supplied runs.

    The intersection is exploratory for legacy artifacts. It does not recover
    forecasts absent from an arm or outcomes absent from the data.
    """
    common = {}
    vintages = set()
    for run_id in sorted(run_ids):
        parts = parse_run_id(run_id)
        meta = load_meta(run_id, roots)
        vintage = (meta.get("input_sha256"), meta.get("code_sha256"))
        if any(not v or v == "unknown" for v in vintage):
            raise ValueError(f"{run_id}: missing analysis provenance")
        vintages.add(vintage)
        frame = load_predictions(run_id, roots)
        labels = meta.get("block_labels", [1, 2, 3, 4, 5, 6])
        for b in labels:
            key = (parts["origin_index"], parts["pred_len"], int(b))
            stamps = set(frame.filter(pl.col("block") == b)["timestamp"].unique().to_list())
            common[key] = common[key] & stamps if key in common else stamps
    if len(vintages) != 1:
        raise ValueError("analysis mixes code or input vintages")
    if any(not stamps for stamps in common.values()):
        raise ValueError("a required origin/horizon/block has no common forecast times")
    return pl.DataFrame([
        {"origin_index": i, "pred_len": h, "block": b, "timestamp": t}
        for (i, h, b), stamps in sorted(common.items()) for t in sorted(stamps)
    ])


def gather_grid(run_ids: list[str], roots: list[Path], *,
                windows: pl.DataFrame | None = None) -> pl.DataFrame:
    """Mean step errors on identical actual targets; average seeds afterwards.

    Common times and per-block hashes make downstream sample equality checkable.
    Forecast files and their original metadata are never modified.
    """
    import hashlib
    if windows is None:
        windows = evaluation_windows(run_ids, roots)
    rows = []
    raw_targets = {}
    for run_id in sorted(run_ids):
        parts = parse_run_id(run_id)
        meta = load_meta(run_id, roots)
        keep = windows.filter((pl.col("origin_index") == parts["origin_index"]) &
                              (pl.col("pred_len") == parts["pred_len"]))
        frame = load_predictions(run_id, roots).join(
            keep.select("block", "timestamp"), on=["block", "timestamp"], how="semi"
        ).sort(["block", "timestamp", "step"])
        hashes = []
        for (b,), block_frame in frame.group_by("block", maintain_order=True):
            key = (parts["origin_index"], parts["pred_len"], b)
            values = block_frame["y_true"].to_numpy().astype(np.float64) * float(meta["sigma_g"]) + float(meta["mu_g"])
            if key in raw_targets and not np.allclose(values, raw_targets[key], rtol=1e-5, atol=1e-8):
                raise ValueError(f"{run_id}: actual raw targets disagree")
            raw_targets[key] = values
            keys = block_frame.select("timestamp", "step", "target_timestamp").to_numpy().astype("<i8")
            hashes.append({"block": int(b), "evaluation_keys_sha256": hashlib.sha256(keys.tobytes()).hexdigest()})
        rows.append(block_metrics(frame, float(meta["naive_rw_z"])).join(
            pl.DataFrame(hashes), on="block"
        ).with_columns(
            pl.lit(run_id).alias("run_id"), pl.lit(str(parts["model"])).alias("model"),
            pl.lit(int(parts["origin_index"])).cast(pl.Int32).alias("origin_index"),
            pl.lit(str(meta["origin"])).alias("origin"),
            pl.lit(int(parts["k"])).cast(pl.Int32).alias("k"),
            pl.lit(int(parts["pred_len"])).cast(pl.Int32).alias("pred_len"),
            pl.lit(int(parts["seed"])).cast(pl.Int32).alias("seed"),
            pl.lit(float(meta["sigma_g"])).alias("sigma_g"),
        ))
    return pl.concat(rows)
''')
replace('metrics.py', '    return (\n        grid.group_by(["model", "origin_index", "origin", "k", "pred_len", "block"])', '''    identity = ["model", "origin_index", "origin", "k", "pred_len", "block"]
    extra = []
    if "evaluation_keys_sha256" in grid.columns:
        if grid.group_by(identity).agg(pl.col("evaluation_keys_sha256").n_unique().alias("n")).filter(pl.col("n") != 1).height:
            raise ValueError("seeds were evaluated on different target calendars")
        extra = [pl.col("evaluation_keys_sha256").first()]
    return (
        grid.group_by(identity)''')
replace('metrics.py', '            pl.col("mse").count().alias("n_seeds"),', '            pl.col("mse").count().alias("n_seeds"),\n            *extra,')
function('metrics.py', 'per_origin_relmse', '''
def per_origin_relmse(seed_avg: pl.DataFrame, model: str, k: int | None = None) -> pl.DataFrame:
    """Equal-weight block RelMSE, matching the headline and comparison panel."""
    part = seed_avg.filter((pl.col("model") == model) & (pl.col("pred_len") == PRED_LEN))
    if k is not None:
        part = part.filter(pl.col("k") == k)
    return part.group_by("origin").agg(
        pl.col("rel_mse").mean(), pl.col("n_windows").sum()
    ).sort("origin")
''')
replace('metrics.py', '    a = per_origin_relmse(seed_avg, left[0], left[1])', '''    if "evaluation_keys_sha256" in seed_avg.columns:
        parts = []
        for tag, k in (left, right):
            part = seed_avg.filter((pl.col("model") == tag) & (pl.col("pred_len") == PRED_LEN))
            if k is not None:
                part = part.filter(pl.col("k") == k)
            parts.append(part.select("origin_index", "block", "evaluation_keys_sha256"))
        check = parts[0].join(parts[1], on=["origin_index", "block"], suffix="_right")
        if check.filter(pl.col("evaluation_keys_sha256") != pl.col("evaluation_keys_sha256_right")).height:
            raise ValueError("paired contrast uses different actual targets")
    a = per_origin_relmse(seed_avg, left[0], left[1])''')
replace('metrics.py', '        "left_better": int((diff < 0).sum()),', '        "left_better": int((diff < 0).sum()),\n        "inference_status": "exploratory; independent-origin t approximation only",')
replace('comparisons.py', '    origin_indices: tuple[int, ...] | None = None,\n) -> PredictionPanel:', '    origin_indices: tuple[int, ...] | None = None,\n    *, windows: pl.DataFrame | None = None,\n) -> PredictionPanel:')
replace('comparisons.py', '                frame = load_predictions(run_id, roots)\n                sig =', '''                frame = load_predictions(run_id, roots)
                if windows is not None:
                    keep = windows.filter((pl.col("origin_index") == index) & (pl.col("pred_len") == pred_len))
                    frame = frame.join(keep.select("block", "timestamp"), on=["block", "timestamp"], how="semi").sort(["block", "timestamp", "step"])
                sig =''')
replace('comparisons.py', '                vintages.add((meta.get("input_sha256"), meta.get("code_sha256")))', '''                vintage = (meta.get("input_sha256"), meta.get("code_sha256"))
                if any(not v or v == "unknown" for v in vintage):
                    raise ValueError(f"{run_id}: missing analysis provenance")
                vintages.add(vintage)''')

function('runner.py', 'PilotResult', '''
class PilotResult:
    """Validation-only descriptive gate; it does not establish statistical power."""
    val_mse: dict[int, float]
    n_val: int
    passed: bool

    def __str__(self) -> str:
        rungs = "  ".join(f"K={k}: {v:.6f}" for k, v in sorted(self.val_mse.items()))
        return f"validation mean seed/step MSE  {rungs}\\nK=8 lower validation loss: {self.passed}; descriptive selection event, no CW claim"
''')
replace('runner.py', '        val_mse[k] = float(_np.mean((y_val - mean_pred) ** 2))', '        val_mse[k] = float(_np.mean((_np.stack(stacked) - y_val[None, :, :]) ** 2))')
c, pilot_node = cell_for('runner.py', 'stage5_pilot')
s = ''.join(c['source'])
a = s.index('    small, large = min(rungs), 8')
s = s[:a] + '    return PilotResult(val_mse=val_mse, n_val=len(y_val),\n                       passed=bool(8 in val_mse and val_mse[8] < val_mse[min(rungs)]))\n' + ''.join(s.splitlines(keepends=True)[pilot_node.end_lineno:])
c['source'] = s.splitlines(keepends=True)

def set_step(slug, source):
    matches=[c for c in nb['cells'] if c.get('metadata',{}).get('itbtc',{}).get('step')==slug]
    assert len(matches)==1, slug
    matches[0]['source']=(source.strip()+'\n').splitlines(keepends=True)

# Additional report/step changes follow before the single final notebook write.
summary_function = '''
def research_summary(seed_avg: pl.DataFrame, keff_tbl: pl.DataFrame, *,
                     B: int = 9999, seed: int = 42) -> dict:
    """Exploratory reanalysis; p-values assume independent origins.

    Origins overlap in training and test calendars. These diagnostics are not
    confirmatory evidence. MDE uses observed TEST slopes, not pre-test pilot data.
    """
    main = seed_avg.filter((pl.col("model") == "itr") & (pl.col("pred_len") == PRED_LEN))
    origin = main.group_by("origin", "k").agg(pl.col("mse").mean(), pl.col("rel_mse").mean(), pl.col("r2_oos").mean())
    rung = origin.group_by("k").agg(
        pl.col("mse").mean().alias("MSE"), pl.col("rel_mse").mean().alias("RelMSE"),
        (pl.col("rel_mse").std()/pl.len().sqrt()).alias("SE_across_origins"),
        pl.col("r2_oos").mean().alias("R2_oos"), pl.len().alias("n_origins"),
    ).sort("k")
    wide = {k: origin.filter(pl.col("k") == k).sort("origin")["rel_mse"].to_numpy() for k in K_LADDER}
    d48, d812 = wide[4]-wide[8], wide[8]-wide[12]
    margin = .25 * abs(float(d48.mean()))
    race = main.join(keff_tbl.select("origin", "k", "pr_raw"), on=["origin", "k"])
    groups = race["origin_index"].to_numpy()*100 + race["block"].to_numpy()
    clusters = race["origin_index"].to_numpy()
    y, k, pr = [race[n].to_numpy().astype(float) for n in ("rel_mse", "k", "pr_raw")]
    t_ab, p_ab = j_test(y, k, pr, groups, clusters=clusters)
    t_ba, p_ba = j_test(y, pr, k, groups, clusters=clusters)
    amp = amplification(seed_avg)
    beta = panel_beta1(amp, B=B, seed=seed)
    sensitivity = []
    for offset in range(5):
        labels = [o.label for o in ORIGINS[offset::5]]
        part = amp.filter(pl.col("origin").is_in(labels))
        if part.height == len(labels)*6 and len(labels) >= 2:
            sub = panel_beta1(part, B=B, seed=seed)
            sensitivity.append({"origins": labels, "G": sub.n_clusters, "beta1": sub.beta1,
                                "p_diagnostic": sub.headline_p})
    dec = decay(seed_avg, k=8)
    crossing = []
    for tau in TAU_SENSITIVITY:
        bs = dec.b_star(tau)
        km = kaplan_meier(bs["b_star"].to_numpy(), bs["event"].to_numpy()) if bs.height else None
        crossing.append({"tau": tau, "status": "descriptive" if bs.height else "undefined",
                         "median_b_star": km.median if km else None, "ci_low": None, "ci_high": None,
                         "events": int(bs["event"].sum()) if bs.height else 0,
                         "censored": int((~bs["event"]).sum()) if bs.height else 0,
                         "n_origins": bs.height})
    return {
        "analysis_schema_version": 2,
        "inference_status": "exploratory; cross-origin dependence unresolved; no confirmatory rejection",
        "evaluation_population": "common surviving forecast times per origin, horizon and block",
        "estimand": "mean seed step-squared-error; block RelMSE; equal block and origin weights",
        "rq1": {"rung_effects": rung.to_dicts(), "contrast_metric": "RelMSE", "delta_4_to_8": float(d48.mean()),
                "delta_8_to_12": float(d812.mean()), "tost_margin": margin,
                "tost": str(tost_equivalence(d812, margin)),
                "j_test_k_augmented_by_keff": {"t": t_ab, "p": p_ab},
                "j_test_keff_augmented_by_k": {"t": t_ba, "p": p_ba},
                "covariance": "CR1 by origin; fixed effects by origin/block; t(G-1)",
                "causal_keff_claim": False},
        "rq2": {"beta1": beta.beta1, "t": beta.t_statistic, "cluster_se": beta.cluster_se,
                "p_rademacher": beta.p_rademacher, "p_webb": beta.p_webb, "headline_p": beta.headline_p,
                "G": beta.n_clusters, "N": beta.n_observations, "B": beta.B,
                "minimum_detectable_beta1": minimum_detectable_beta1(beta.within_slopes),
                "mde_source": "post-analysis TEST within-origin slopes; not prospective power",
                "within_slopes": beta.within_slopes.tolist(), "stride5_sensitivity": sensitivity,
                "consecutive_origin_overlap_pct": 79.2,
                "age_definition": "time since selection/deployment; gradient training cutoff is three months earlier",
                "fresh_intervention": "training and validation windows both moved; one fresh versus five aged seeds"},
        "rq3": {"tau_headline": TAU_HEADLINE, "reference": "block 1 skill (exploratory post-audit definition)",
                "b_star": crossing, "excluded_origins": list(dec.excluded_origins),
                "optimal_cadence_estimated": False, "logrank_status": "withheld: paired and dependent origins",
                "interval_status": "withheld: independent-subject confidence bands not justified"},
    }


'''
c, node = cell_for('report.py','build_report')
s = ''.join(c['source'])
node = next(n for n in ast.parse(s).body if getattr(n,'name',None)=='build_report')
lines=s.splitlines(keepends=True)
lines[node.lineno-1:node.lineno-1]=[summary_function]
c['source']=''.join(lines).splitlines(keepends=True)
c['metadata']['itbtc']['projection_imports'] = [
    'from itransformer_btc.metrics import evaluation_windows, j_test, panel_beta1, minimum_detectable_beta1, decay, kaplan_meier, tost_equivalence',
    'from itransformer_btc.config import TAU_HEADLINE, TAU_SENSITIVITY',
    'from itransformer_btc.keff import keff_table',
    'from itransformer_btc.train import code_sha256',
]
replace('report.py', '    raw = gather_grid(run_ids, roots)', '    windows = evaluation_windows(run_ids, roots)\n    raw = gather_grid(run_ids, roots, windows=windows)\n    log(f"report: common calendar has {windows.height} forecast times")')
replace('report.py', '    panel = build_panel(comparison_keys, roots)', '    panel = build_panel(comparison_keys, roots, windows=windows)')
replace('report.py', '    grid_r2 = {int(row["k"]): float(row["R2_oos"]) for row in grid["rq1"]["rung_effects"]}', '''    analysis = research_summary(seed_avg, keff_table(features), B=bootstrap_b, seed=seed)
    grid_r2 = {int(row["k"]): float(row["R2_oos"]) for row in analysis["rq1"]["rung_effects"]}''')
replace('report.py', '        "rq1": grid["rq1"],\n        "rq2": grid["rq2"],\n        "rq3": grid["rq3"],', '''        **analysis,
        "analysis_code_sha256": code_sha256(),
        "prediction_code_sha256": sorted({load_meta(r, roots)["code_sha256"] for r in run_ids}),
        "publication_status": "exploratory corrected reanalysis; new experimental controls remain unrun",''')

set_step('pilot', '''pilot = stage5_pilot(features, out_root=ARTIFACTS, device=device)
print(pilot)
print("Validation selection event only; no CW nesting claim or prospective-power estimate.")''')
set_step('rq1', '''if not GRID_COMPLETE:
    print("RQ1 skipped: current-vintage manifest is incomplete. Save and resume.")
else:
    done = sorted(c.run_id for c in ALL)
    grid = gather_grid(done, discover_roots(ARTIFACTS))
    seed_avg = seed_average(grid)
    research_results = research_summary(seed_avg, keff_tbl, B=99_999, seed=42)
    print(research_results["inference_status"])
    print(pl.DataFrame(research_results["rq1"]["rung_effects"]))
    print(research_results["rq1"])
    print("Feature subsets change content as well as PR. This does not identify a causal K_eff effect.")''')
set_step('rq2', '''if not GRID_COMPLETE:
    print("RQ2 skipped: current-vintage manifest is incomplete. Save and resume.")
else:
    amp = amplification(seed_avg)
    print(research_results["rq2"])
    print("MDE uses observed TEST slopes: a post-analysis sensitivity diagnostic.")
    print("Overlap affects origin-cluster inference; stride-5 subsets have only three origins.")
    print(falsification_relmse(seed_avg))
    print("Fresh-minus-aged is a combined training/validation update, with unequal seed counts and calendar confounding.")''')
set_step('rq3', '''if not GRID_COMPLETE:
    print("RQ3 skipped: current-vintage manifest is incomplete. Save and resume.")
else:
    dec = decay(seed_avg, k=8)
    print(dec.table)
    print(research_results["rq3"])
    print("Threshold crossing is descriptive and uses block 1 as reference. It does not estimate an optimal retraining cadence.")''')
set_step('save', '''if not GRID_COMPLETE:
    print("Analysis save skipped: current-vintage manifest is incomplete; run artifacts remain resumable.")
else:
    _digest, _digest_source = _input_sha256(PARQUET)
    paper_numbers = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input_parquet": str(PARQUET), "input_sha256": _digest,
        "input_sha256_source": _digest_source, "code_sha256": code_sha256(),
        "runs_complete": len(done), "runs_in_manifest": len(ALL),
        "keff": {"gate_pr_k8_pre_first_origin": gate, "gate_floor": GATE_PR_FLOOR,
                 "corr_k_keff": corr_k_keff(keff_tbl), "per_rung": rung_view.to_dicts()},
        **research_results,
    }
    out = ARTIFACTS / "paper_numbers.json"
    out.write_text(json.dumps(paper_numbers, indent=2, default=float), encoding="utf-8")
    seed_avg.write_parquet(ARTIFACTS / "seed_averaged_cells.parquet")
    grid.write_parquet(ARTIFACTS / "run_block_metrics.parquet")
    amp.write_parquet(ARTIFACTS / "amplification_panel.parquet")
    dec.table.write_parquet(ARTIFACTS / "decay_panel.parquet")
    print(f"wrote {out}; save the session output to resume later")''')

for c in nb['cells']:
    if c['cell_type']=='code':
        ast.parse(''.join(c['source']))
        c['outputs']=[]; c['execution_count']=None
PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False)+'\n', encoding='utf-8')
print('Applied common-calendar aggregation, pilot estimand and research-summary corrections')
