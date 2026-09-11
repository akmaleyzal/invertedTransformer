from pathlib import Path
exec(Path('.research/audit-repair-work/applied_first_batch.py').read_text(encoding='utf-8').split('# A13:')[0])
replace('report.py', 'headline[("ridge", 4)]', 'headline[("rdg", 4)]')
replace('report.py', '        "BetaSlope": fmt(numbers["rq2"]["beta1"], 6),',
'''        "DecayExcluded": fmt(len(numbers["rq3"]["excluded_origins"]), 0),
        "DecayDefined": fmt(next(r["n_origins"] for r in numbers["rq3"]["b_star"] if r["tau"] == numbers["rq3"]["tau_headline"]), 0),
        "DecayMedian": fmt(next(r["median_b_star"] for r in numbers["rq3"]["b_star"] if r["tau"] == numbers["rq3"]["tau_headline"]), 0),
        "BetaSlope": fmt(numbers["rq2"]["beta1"], 6),''')
for c in nb['cells']:
    if c['cell_type'] == 'code':
        ast.parse(''.join(c['source']))
        c['outputs'] = []
        c['execution_count'] = None
PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False)+'\n', encoding='utf-8')
