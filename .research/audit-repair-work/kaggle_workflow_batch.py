from pathlib import Path
exec(Path(__file__).with_name('applied_first_batch.py').read_text(encoding='utf-8').split('# A13:')[0])

# One bounded validation cache; no test predictions are evaluated or written.
c,n=cell_for('runner.py','tune_on_validation')
lines=''.join(c['source']).splitlines(keepends=True)
lines[n.lineno-1:n.lineno-1]=['''
def validation_fit(tensors, spec, cfg, *, device, out_root=None, roots=None, **schedule):
    """Cache a validation-only fit by data/code/config, including its budget."""
    from itransformer_btc.train import _input_sha256
    identity = json.loads(json.dumps({
        "spec": asdict(spec), "config": asdict(cfg), "schedule_overrides": schedule,
        "code_sha256": code_sha256(), "input_sha256": _input_sha256()[0],
        "torch": str(torch.__version__), "device_type": device.type,
        "training_selection": tensors.training_selection,
    }))
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    destination = Path(out_root) / "validation" / f"{key}.json" if out_root else None
    row = None
    if destination:
        for root in dict.fromkeys([Path(out_root), *(roots or [])]):
            path = root / "validation" / f"{key}.json"
            if path.exists():
                cached = json.loads(path.read_text(encoding="utf-8"))
                if cached.get("identity") == identity and np.isfinite(cached.get("val_mse", np.nan)):
                    row = cached
                    break
    if row is None:
        _, outcome = train_one(tensors, spec, cfg, device=device, **schedule)
        row = {"identity": identity, "val_mse": outcome.best_val_mse,
               "epochs_run": outcome.epochs_run, "wall_time_s": outcome.wall_time_s,
               "n_val": len(tensors.val)}
    if destination:
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = destination.with_suffix(".json.tmp")
        staging.write_text(json.dumps(row, indent=2), encoding="utf-8")
        staging.replace(destination)
        (Path(out_root) / "checkpoints" / f"{spec.run_id}.pt").unlink(missing_ok=True)
    return row


''']
c['source']=''.join(lines).splitlines(keepends=True)
c['metadata']['itbtc'].setdefault('projection_imports',[]).extend(['import hashlib','import numpy as np'])
function('runner.py','tune_on_validation','''
def tune_on_validation(features: pl.DataFrame, *, origin_index=1, k=8,
                       device=None, out_root=None, roots=None, log=print):
    """Select the exploratory iTransformer config on origin-1 validation only."""
    device = device or pick_device()
    tensors = _TensorCache(features, size=1).get(RunCell("tuned", origin_index, k, PRED_LEN, SEEDS[0]))
    rows = []
    for index, point in enumerate(TUNING_GRID):
        cfg = ITransformerConfig(pred_len=PRED_LEN, d_model=int(point["d_model"]), e_layers=int(point["e_layers"]))
        spec = RunSpec(f"probeit{index}", origin_index, k, PRED_LEN, SEEDS[0])
        row = validation_fit(tensors, spec, cfg, device=device, out_root=out_root, roots=roots,
                             max_epochs=TUNING_EPOCHS, patience=TUNING_EPOCHS, lr=float(point["lr"]))
        rows.append({**point, "val_mse": row["val_mse"], "epochs_run": row["epochs_run"]})
        log(f"validation probe {index+1}/{len(TUNING_GRID)}: {rows[-1]}")
    rows.sort(key=lambda r: r["val_mse"])
    best = rows[0]
    return TunedConfig(pred_len=PRED_LEN, d_model=int(best["d_model"]),
                       e_layers=int(best["e_layers"]), lr=float(best["lr"])), rows


def tune_baselines_on_validation(features, *, device=None, out_root=None, roots=None, log=print):
    """Fixed LR sensitivity, no test feedback; report every candidate and cap."""
    device = device or pick_device()
    tensors = _TensorCache(features, size=1).get(RunCell("dlinear", 1, 8, PRED_LEN, SEEDS[0]))
    configs, table = {}, []
    for arm in ("dlinear", "patchtst", "dlinear_all", "patchtst_all"):
        rows = []
        for index, lr in enumerate((1e-4, 1e-3, 1e-2)):
            cfg = replace(RunCell(arm, 1, 8, PRED_LEN, SEEDS[0]).model_config(), lr=lr)
            spec = RunSpec(f"probe{arm}{index}", 1, 8, PRED_LEN, SEEDS[0])
            result = validation_fit(tensors, spec, cfg, device=device, out_root=out_root, roots=roots)
            rows.append({"arm": arm, "lr": lr, "val_mse": result["val_mse"],
                         "epochs_run": result["epochs_run"], "cap_reached": result["epochs_run"] >= cfg.max_epochs})
            log(f"baseline validation: {rows[-1]}")
        best = min(rows, key=lambda r: r["val_mse"])
        configs[arm] = replace(cfg, lr=best["lr"])
        table.extend(rows)
    return configs, table
''')
function('runner.py','stage5_pilot','''
def stage5_pilot(features: pl.DataFrame, *, origin_index=1, rungs=K_LADDER,
                 seeds=SEEDS[:3], out_root=None, roots=None, device=None, log=print):
    """Mean validation step-MSE across seeds; no test artifact or power claim."""
    device = device or pick_device()
    cache = _TensorCache(features, size=1)
    val_mse = {}
    for k in rungs:
        cell = RunCell("main", origin_index, k, PRED_LEN, seeds[0])
        tensors = cache.get(cell)
        losses = []
        for seed in seeds:
            spec = RunSpec("pilotitr", origin_index, k, PRED_LEN, seed)
            result = validation_fit(tensors, spec, cell.model_config(), device=device,
                                    out_root=out_root, roots=roots)
            losses.append(result["val_mse"])
            log(f"pilot {spec.run_id}: validation MSE {losses[-1]:.6f}")
        val_mse[k] = float(np.mean(losses))
    return PilotResult(val_mse=val_mse, n_val=len(tensors.val),
                       passed=bool(8 in val_mse and val_mse[8] < val_mse[min(rungs)]))
''')

# Keep one attachable output bundle across sessions. Copy only completed runs
# accepted by the same strict gate; never consume untrusted pickle objects.
c,n=cell_for('runner.py','execute_parallel')
lines=''.join(c['source']).splitlines(keepends=True)
lines[n.lineno-1:n.lineno-1]=['''
def consolidate_resume_outputs(cells, roots, out_root, configs=None):
    """Carry accepted prior runs forward so the next session needs one bundle."""
    import shutil
    out_root = Path(out_root)
    copied = 0
    for cell in cells:
        cfg = cell.model_config(configs)
        for root in roots:
            root = Path(root)
            if root == out_root:
                continue
            if not is_complete(cell.run_id, root, strict=True, cfg=cfg, columns=cell.columns()):
                continue
            if is_complete(cell.run_id, out_root, strict=True, cfg=cfg, columns=cell.columns()):
                break
            # Metadata is copied last, after every file it certifies.
            for folder, suffix in (("preds", ".parquet"), ("weights", ".pt"),
                                   ("attn", ".parquet"), ("meta", ".json")):
                source = root / folder / f"{cell.run_id}{suffix}"
                if source.exists():
                    destination = out_root / folder / source.name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    staging = destination.with_suffix(destination.suffix + ".tmp")
                    shutil.copyfile(source, staging)
                    staging.replace(destination)
            copied += 1
            break
    # Pending epochs and validation caches are small. Their own identity gates
    # are checked at consumption, even when copied from an older code vintage.
    for root in roots:
        if Path(root) == out_root:
            continue
        for folder, pattern in (("checkpoints", "*.pt"), ("validation", "*.json")):
            for source in (Path(root) / folder).glob(pattern):
                destination = out_root / folder / source.name
                if not destination.exists():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    staging = destination.with_suffix(destination.suffix + ".tmp")
                    shutil.copyfile(source, staging)
                    staging.replace(destination)
    return copied


''']
c['source']=''.join(lines).splitlines(keepends=True)

# Failed alignment must return to the main thread, not disappear in stderr.
c,n=cell_for('runner.py','execute_parallel');s=''.join(c['source'])
s=s.replace('    state = threading.Lock()','    state = threading.Lock()\n    fatal = []')
s=s.replace('            if cursor >= len(queue) or not guard.may_start():','            if fatal or cursor >= len(queue) or not guard.may_start():')
s=s.replace('            if cell.arm in BASELINE_ARMS:\n                _assert_alignment(cell, roots, log)','''            if cell.arm in BASELINE_ARMS:
                try:
                    _assert_alignment(cell, roots, log)
                except Exception as exc:
                    with state:
                        fatal.append(exc)
                    return''')
s=s.replace('    if cursor < len(queue):','    if fatal:\n        raise RuntimeError("fatal cross-model target alignment failure") from fatal[0]\n    if cursor < len(queue):')
c['source']=s.splitlines(keepends=True)

# Artifact completion includes recoverable weights; both bytes and metadata are
# atomic, so an interrupted write can never qualify as a finished run.
replace('train.py','    preds.write_parquet(preds_path)','    staging_preds = preds_path.with_suffix(".parquet.tmp")\n    preds.write_parquet(staging_preds)\n    staging_preds.replace(preds_path)')
replace('train.py','        if meta.get("predictions_sha256") != hashlib.sha256(preds.read_bytes()).hexdigest():','''        weights = root / "weights" / f"{run_id}.pt"
        if not weights.is_file() or meta.get("weights_sha256") != hashlib.sha256(weights.read_bytes()).hexdigest():
            return False
        if meta.get("predictions_sha256") != hashlib.sha256(preds.read_bytes()).hexdigest():''')

for c in nb['cells']:
    if c['cell_type']=='code':
        ast.parse(''.join(c['source']))
        c['outputs']=[];c['execution_count']=None
PATH.write_text(json.dumps(nb,indent=1,ensure_ascii=False)+'\n',encoding='utf-8')
print('Applied cached validation, baseline LR selection and consolidated resume bundle')
