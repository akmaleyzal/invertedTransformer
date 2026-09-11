from pathlib import Path
exec(Path(__file__).with_name('applied_first_batch.py').read_text(encoding='utf-8').split('# A13:')[0])

replace('train.py', '        "prediction_schema_version": 2,', '        "prediction_schema_version": 2,\n        "predictions_sha256": hashlib.sha256(preds_path.read_bytes()).hexdigest(),')
replace('train.py', '        if cfg is None or meta.get("requested_config") != asdict(cfg):', '        if cfg is None or meta.get("requested_config") != json.loads(json.dumps(asdict(cfg))):')
replace('train.py', '        if meta.get("schedule") != schedule:', '        if meta.get("schedule") != json.loads(json.dumps(schedule)):')
replace('train.py', '        pl.read_parquet_schema(preds)\n        return True', '''        required = {"block", "step", "timestamp", "forecast_origin", "input_start",
                    "target_timestamp", "y_true", "y_pred"}
        if meta.get("prediction_schema_version") != 2 or not required <= set(pl.read_parquet_schema(preds)):
            return False
        if meta.get("predictions_sha256") != hashlib.sha256(preds.read_bytes()).hexdigest():
            return False
        return True''')
replace('train.py', '    meta_path.write_text(json.dumps(meta, indent=2))', '    staging_meta = meta_path.with_suffix(".json.tmp")\n    staging_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")\n    staging_meta.replace(meta_path)')
# The writable output must shadow any read-only previous session.
for c in nb['cells']:
    if c.get('metadata',{}).get('itbtc',{}).get('module')=='runner.py':
        s=''.join(c['source'])
        s=s.replace('    roots = roots or discover_roots(out_root)', '    roots = list(dict.fromkeys([Path(out_root), *(roots or discover_roots(out_root))]))')
        c['source']=s.splitlines(keepends=True)
replace('splits.py', '    if ts.ndim != 1 or np.any(np.diff(ts) <= 0):', '    if ts.ndim != 1 or np.any(np.diff(ts) <= 0) or np.any(ts % HOUR_MS != 0):')
replace('splits.py', 'timestamps must be strictly increasing and unique', 'timestamps must be unique, increasing UTC hour boundaries')
function('metrics.py','load_meta','''
def load_meta(run_id: str, roots: list[Path]) -> dict:
    """Read metadata paired with the selected prediction file, never another root."""
    path = _locate(run_id, roots, "preds", ".parquet").parent.parent / "meta" / f"{run_id}.json"
    return json.loads(path.read_text(encoding="utf-8"))
''')
replace('metrics.py','    counts = frame.group_by("timestamp").agg(','''    required = ["block", "timestamp", "step", "forecast_origin", "input_start", "target_timestamp", "y_true", "y_pred"]
    if frame.is_empty() or any(frame[n].null_count() for n in required):
        raise ValueError(f"{run_id}: empty predictions or null prediction keys")
    counts = frame.group_by("timestamp").agg(''')
for c in nb['cells']:
    if c['cell_type']=='code':
        ast.parse(''.join(c['source']))
        c['outputs']=[];c['execution_count']=None
PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False)+'\n',encoding='utf-8')
print('Applied paired artifact provenance and crash-safe completion metadata')
