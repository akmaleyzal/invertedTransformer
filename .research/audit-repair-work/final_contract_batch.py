"""One-time notebook edits after the final report/resume review."""
from pathlib import Path
exec(Path('.research/audit-repair-work/applied_first_batch.py').read_text(encoding='utf-8').split('# A13:')[0])

replace('report.py', '    main = seed_avg.filter((pl.col("model") == "itr") & (pl.col("pred_len") == PRED_LEN))',
'''    fresh_counts = sorted(seed_avg.filter(pl.col("model") == "itrf")["n_seeds"].unique().to_list())
    main = seed_avg.filter((pl.col("model") == "itr") & (pl.col("pred_len") == PRED_LEN))''')
replace('report.py', '"fresh_intervention": "training and validation windows both moved; one fresh versus five aged seeds"',
'''"fresh_intervention": "training and validation windows both moved; inspect fresh_seed_counts",
                "fresh_seed_counts": fresh_counts''')
replace('report.py', 'if left not in seed_avg["model"].unique().to_list():',
        'if not {left, right}.issubset(set(seed_avg["model"].unique().to_list())):')
replace('report.py', '"capped_runs": sum(bool(m.get("reached_epoch_cap", False)) for m in metadata),',
'''"capped_runs": sum(row["epochs_at_cap"] for row in architecture["cells"] if row["max_epochs"] > 0),
        "source": "actual epochs versus recorded schedule; historical missing schedules use their documented 30-epoch cap",''')
replace('report.py', '    numbers["falsification"]["note"] = "Same actual B4--B6 targets, scale-free loss; see refresh-control decomposition and achieved seed counts."',
'''    numbers["falsification"]["note"] = "Same actual B4--B6 targets, scale-free loss; see refresh-control decomposition and achieved seed counts."
    numbers["falsification"]["fresh_seed_counts"] = analysis["rq2"]["fresh_seed_counts"]''')
replace('runner.py', '\n        if cell.arm in BASELINE_ARMS:',
        '\n        if cell.arm in (*BASELINE_ARMS, "dlinear_all", "patchtst_all"):')
# Parallel and serial executors are separate notebook cells.
replace('runner.py', '\n            if cell.arm in BASELINE_ARMS:',
        '\n            if cell.arm in (*BASELINE_ARMS, "dlinear_all", "patchtst_all"):')
replace('runner.py', 'def tensor_key(self) -> tuple[bool, int, int, int]:',
        'def tensor_key(self) -> tuple:')
replace('splits.py', '    # Named columns override the rung, so a matched-K arm can hold K fixed and\n    # move only the effective rank (`D70`). ``k`` still has to agree with the set:',
        '    # Named columns override the rung; the historical matched-K arm changes\n    # feature identity as well as PR. ``k`` still has to agree with the set:')
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        ast.parse(''.join(cell['source']))
        cell['outputs'] = []
        cell['execution_count'] = None
PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False)+'\n', encoding='utf-8')
print('Updated notebook report contracts and both baseline alignment gates')
