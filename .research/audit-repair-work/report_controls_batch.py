from pathlib import Path
exec(Path(__file__).with_name('applied_first_batch.py').read_text(encoding='utf-8').split('# A13:')[0])

c,n=cell_for('runner.py','validation_fit')
c['metadata']['itbtc'].setdefault('projection_imports',[]).append('from dataclasses import asdict')
replace('runner.py','    "orthogonal", "redundant", "look048", "look192", "tuned",','    "orthogonal", "redundant", "look048", "look192", "tuned",\n    "valrefresh", "repr_identity", "repr_whiten", "repr_correlate", "dlinear_all", "patchtst_all",')
replace('runner.py','    "main", "uniform", "fresh", "horizon", *BASELINE_ARMS, *ROBUSTNESS_ARMS,\n    "valrefresh", "repr_identity", "repr_whiten", "repr_correlate",\n    "dlinear_all", "patchtst_all",','    "main", "uniform", "fresh", "horizon", *BASELINE_ARMS, *ROBUSTNESS_ARMS,')
c,n=cell_for('runner.py','manifest');s=''.join(c['source'])
s=s.replace('def manifest(arms: tuple[str, ...] = ALL_ARMS) -> list[RunCell]:','def manifest(arms: tuple[str, ...] = ALL_ARMS, *, historical: bool = False) -> list[RunCell]:')
s=s.replace('    return unique','''    if historical:
        added = {"valrefresh", "repr_identity", "repr_whiten", "repr_correlate", "dlinear_all", "patchtst_all"}
        unique = [c for c in unique if c.arm not in added and (c.arm != "fresh" or c.seed == SEEDS[0])]
    return unique''')
c['source']=s.splitlines(keepends=True)

c,n=cell_for('report.py','build_report');s=''.join(c['source'])
s=s.replace('    missing_runs = sorted({c.run_id for c in manifest()} - set(run_ids))','''    # A declared new grid and the preserved historical grid are distinct studies.
    historical = "manifest_run_ids" not in grid
    required = set(grid.get("manifest_run_ids", [c.run_id for c in manifest(historical=True)]))
    if not historical and required != {c.run_id for c in manifest()}:
        raise ValueError("declared manifest differs from the current controlled-rerun protocol")
    missing_runs = sorted(required - set(run_ids))''')
s=s.replace('    windows = evaluation_windows(run_ids, roots)','    run_ids = sorted(required)\n    windows = evaluation_windows(run_ids, roots)')
s=s.replace('    return ReportInputs(\n', '''    numbers["training_protocol"] = "historical 1620-run grid" if historical else "post-audit 2130-run controlled rerun"
    numbers["publication_status"] = "exploratory historical reanalysis" if historical else "completed post-audit exploratory grid; inspect controls and budget diagnostics"
    numbers["manifest_run_ids"] = run_ids
    new_tags = {"itrv": "validation_refresh", "repi": "representation_identity",
                "repw": "representation_whiten", "repc": "representation_correlate",
                "dlina": "dlinear_all_channel_sensitivity", "ptsta": "patchtst_all_channel_sensitivity"}
    control_rows = seed_avg.filter(pl.col("model").is_in(list(new_tags)))
    numbers["experimental_controls"] = {
        "status": "not run in this historical vintage" if control_rows.is_empty() else "run",
        "interpretation": "Representation effects preserve information but do not identify a PR-only causal effect. Refresh contrasts compare training/selection procedures.",
        "seed_averaged_blocks": control_rows.to_dicts(),
        "paired": [],
    }
    for left, right, question in (
        ("itrv", "itr", "updated validation with original training, B4--B6"),
        ("itrf", "itrv", "updated training with the same refreshed validation, B4--B6"),
        ("repw", "repi", "whitening of identical information, all use_norm=False"),
        ("repc", "repw", "invertible correlated coordinates, all use_norm=False"),
        ("dlina", "dlin", "DLinear all-channel versus target-only objective"),
        ("ptsta", "ptst", "PatchTST all-channel versus target-only objective"),
    ):
        if left not in seed_avg["model"].unique().to_list():
            continue
        pair = seed_avg.filter(pl.col("block").is_in([4,5,6])) if left in ("itrv", "itrf") else seed_avg
        numbers["experimental_controls"]["paired"].append({
            **paired_contrast(pair, (left, 8), (right, 8)), "question": question})
    metadata = [load_meta(run_id, roots) for run_id in run_ids]
    selected_counts = [int(m["n_train"]) for m in metadata]
    numbers["training_sample"] = {"min": min(selected_counts), "max": max(selected_counts),
        "equal_count_achieved": len(set(selected_counts)) == 1,
        "protocol_count": None if historical else 11500,
        "scope": "surviving continuous training windows; no outcome imputation"}
    if not historical and set(selected_counts) != {11500}:
        raise ValueError("new grid violates its fixed training-window budget")
    numbers["representation_diagnostics"] = [
        {"run_id": m["run_id"], **m["representation"]} for m in metadata if m.get("representation")]
    numbers["optimization_status"] = {
        "capped_runs": sum(bool(m.get("reached_epoch_cap", False)) for m in metadata),
        "ranking_claim": "configuration comparison only; early stopping or a higher cap is not proof of optimizer convergence",
    }
    numbers["falsification"]["note"] = "Same actual B4--B6 targets, scale-free loss; see refresh-control decomposition and achieved seed counts."
    numbers["attention_amplification"]["note"] = "Same features, Q/K inactive under uniform attention; active capacity differs. " + ("Historical dropout also differs." if historical else "Attention-weight dropout is shared in the new implementation.")
    numbers["coverage"]["note"] = "Conditional on surviving continuous windows; gap causes unverified and missing outcomes unrecovered. See training_sample for achieved counts."
    return ReportInputs(
''')
assert 'experimental_controls' in s
c['source']=s.splitlines(keepends=True)

c,n=cell_for('report.py','research_summary');s=''.join(c['source'])
s=s.replace('"fresh_intervention": "training and validation windows both moved; one fresh versus five aged seeds",','''"fresh_intervention": "training and validation windows both moved; validation-refresh control separates the two updates in the new grid",
            "fresh_seed_counts": seed_avg.filter(pl.col("model") == "itrf")["n_seeds"].unique().to_list() if "n_seeds" in seed_avg.columns else [],''')
c['source']=s.splitlines(keepends=True)

c,n=cell_for('report.py','_architecture_section');s=''.join(c['source'])
s=s.replace('            "n_parameters": meta.get("n_parameters"),','''            "n_parameters": meta.get("n_parameters"),
            "n_allocated_parameters": meta.get("n_allocated_parameters"),
            "effective_input_channels": meta.get("effective_input_channels"),
            "loss_target": meta.get("loss_target", "historical; inspect config"),''')
c['source']=s.splitlines(keepends=True)

# Preserve the existing single-title/folding style; put the audit notice in the
# opening cell instead of adding an extra phase or changing all cell indices.
status=next((c for c in nb['cells'] if c.get('metadata',{}).get('itbtc',{}).get('role')=='audit_status'),None)
if status:
    notice=''.join(status['source']).split('\n',1)[1]
    nb['cells'][0]['source'] += ('\n\n**Status audit A01–A15 — 9 September 2026**\n'+notice).splitlines(keepends=True)
    nb['cells'].remove(status)

for c in nb['cells']:
    if c['cell_type']=='code':
        ast.parse(''.join(c['source']))
        c['outputs']=[];c['execution_count']=None
PATH.write_text(json.dumps(nb,indent=1,ensure_ascii=False)+'\n',encoding='utf-8')
print('Report separates historical and new manifests and reports every new control')
