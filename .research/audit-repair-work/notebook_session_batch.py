from pathlib import Path
exec(Path(__file__).with_name('applied_first_batch.py').read_text(encoding='utf-8').split('# A13:')[0])

replace('runner.py','        self, budget_h: float = SESSION_BUDGET_H, reserve_h: float = RESERVE_H\n','        self, budget_h: float = SESSION_BUDGET_H, reserve_h: float = RESERVE_H,\n        *, started_at: float | None = None,\n')
replace('runner.py','        self.deadline = time.perf_counter() + (budget_h - reserve_h) * 3600.0','''        if not np.isfinite(budget_h + reserve_h) or min(budget_h, reserve_h) < 0:
            raise ValueError("budget and reserve must be finite, nonnegative hours")
        self.deadline = (time.perf_counter() if started_at is None else started_at) + (budget_h - reserve_h) * 3600.0''')
replace('train.py','    protocol = cfg.schedule() if hasattr(cfg, "schedule") else TrainSchedule()','''    active = getattr(_TRAINING_CONTEXT, "state", None)
    if active is not None and time.perf_counter() >= active[2]:
        raise SessionBudgetExhausted(f"{spec.run_id}: session budget exhausted before fitting")
    protocol = cfg.schedule() if hasattr(cfg, "schedule") else TrainSchedule()''')
# Validation-only sessions still discover input bundles that contain no preds.
replace('runner.py','    if kaggle_input.exists():\n        roots += sorted(p for p in kaggle_input.iterdir() if (p / "preds").is_dir())','''    if kaggle_input.exists():
        for folder in ("validation", "checkpoints"):
            roots += sorted(p.parent for p in kaggle_input.glob(f"*/{folder}") if p.is_dir())
            roots += sorted(p.parent for p in kaggle_input.glob(f"*/*/{folder}") if p.is_dir())
        roots += sorted(p for p in kaggle_input.iterdir() if (p / "preds").is_dir())''')

def set_step(slug, source):
    c=next(c for c in nb['cells'] if c.get('metadata',{}).get('itbtc',{}).get('step')==slug)
    c['source']=(source.strip()+'\n').splitlines(keepends=True)
    return c

c=next(c for c in nb['cells'] if c.get('metadata',{}).get('itbtc',{}).get('step')=='setup')
s=''.join(c['source'])
s=s.replace('SESSION_T0 = time.perf_counter()', '''# Preserve the clock when the first cell is rerun in the same kernel.
SESSION_T0 = globals().get("SESSION_T0", time.perf_counter())
# EDIT before Run All: read the remaining quota from Kaggle's accelerator meter.
# The notebook has no authenticated API for your account's remaining weekly quota.
WEEKLY_GPU_HOURS_REMAINING = None
SESSION_ALREADY_USED_H = 0.0  # idle/use before the first cell, if this is not a fresh session
SESSION_LIMIT_H = 11.5        # below the user's 12-hour session ceiling
SAVE_RESERVE_H = 0.75        # 45 minutes for saving outputs and stopping the session
ANALYSIS_ONLY = False        # True on Kaggle CPU to render a completed saved grid''')
s=s.replace('\n\ndef looks_like_parquet', '''

if not ON_KAGGLE:
    raise RuntimeError("Full notebook execution is configured for Kaggle. Local pytest runs only small checks.")
if not ANALYSIS_ONLY:
    if WEEKLY_GPU_HOURS_REMAINING is None:
        raise ValueError("Set WEEKLY_GPU_HOURS_REMAINING from Kaggle's current quota meter, then Run All.")
    if not 0 <= float(WEEKLY_GPU_HOURS_REMAINING) <= 30 or not 0 <= SESSION_ALREADY_USED_H < 12:
        raise ValueError("Use remaining quota in [0,30] hours and elapsed session time in [0,12).")
    import torch
    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if len(names) != 2 or not all("T4" in name for name in names):
        raise RuntimeError(f"Select GPU T4 x2 in Kaggle settings before training. Detected: {names}")


def looks_like_parquet''')
c['source']=s.splitlines(keepends=True)

c=next(c for c in nb['cells'] if c.get('metadata',{}).get('itbtc',{}).get('step')=='code_digest')
c['source']+='''

# One deadline covers validation probes, training and analysis. T4 x2 is used
# through one independent run per device, without splitting a minibatch.
roots = discover_roots(ARTIFACTS)
SESSION_GUARD = BudgetGuard(
    0.0 if ANALYSIS_ONLY else min(float(WEEKLY_GPU_HOURS_REMAINING),
                                 max(0.0, SESSION_LIMIT_H - SESSION_ALREADY_USED_H)),
    SAVE_RESERVE_H, started_at=SESSION_T0,
)
consolidate_resume_outputs([], roots, ARTIFACTS)
PRELUDE_COMPLETE = True
print(f"Training time remaining before save margin: {max(0, SESSION_GUARD.remaining_s)/3600:.2f} h")
'''.splitlines(keepends=True)

c=next(c for c in nb['cells'] if c.get('metadata',{}).get('itbtc',{}).get('step')=='invariants')
s=''.join(c['source'])
# Tiny invariant checks run on CPU during analysis-only sessions.
s=s.replace('device = pick_device()','device = torch.device("cpu") if ANALYSIS_ONLY else pick_device()')
s=s.replace('build_origin_tensors(features, o, 1)','build_origin_tensors(features, o, 1, train_window_limit=11_500)')
c['source']=s.splitlines(keepends=True)

set_step('pilot','''
try:
    with TrainingSession(ARTIFACTS, roots, SESSION_GUARD.deadline):
        pilot = stage5_pilot(features, out_root=ARTIFACTS, roots=roots, device=device,
                             log=lambda msg: print(msg, flush=True))
    print(pilot)
except SessionBudgetExhausted as exc:
    PRELUDE_COMPLETE = False
    print(f"PAUSED during validation: {exc}. Save output; attach it next session.")
''')['metadata']['itbtc']['writes']=['artifacts/validation/*.json','artifacts/checkpoints/*.pt']

set_step('tune','''
GRID_CONFIGS = {}
if PRELUDE_COMPLETE:
    try:
        with TrainingSession(ARTIFACTS, roots, SESSION_GUARD.deadline):
            TUNED_CONFIG, TUNING_TABLE = tune_on_validation(
                features, device=device, out_root=ARTIFACTS, roots=roots,
                log=lambda msg: print(msg, flush=True))
            BASELINE_CONFIGS, BASELINE_TUNING_TABLE = tune_baselines_on_validation(
                features, device=device, out_root=ARTIFACTS, roots=roots,
                log=lambda msg: print(msg, flush=True))
        GRID_CONFIGS = {"tuned": TUNED_CONFIG, **BASELINE_CONFIGS}
        selection = {"code_sha256": code_sha256(), "input_sha256": _input_sha256()[0],
                     "scope": "origin 1 validation only; exploratory post-audit search",
                     "itr_candidates": TUNING_TABLE, "baseline_candidates": BASELINE_TUNING_TABLE,
                     "selected": {name: asdict(cfg) for name, cfg in GRID_CONFIGS.items()}}
        (ARTIFACTS / "meta").mkdir(parents=True, exist_ok=True)
        (ARTIFACTS / "meta" / "tuning_selection.json").write_text(
            json.dumps(selection, indent=2), encoding="utf-8")
    except SessionBudgetExhausted as exc:
        PRELUDE_COMPLETE = False
        print(f"PAUSED during tuning: {exc}. Completed probes and partial epochs are saved.")
else:
    print("Tuning deferred until the validation prelude resumes.")
''')

set_step('grid','''
for _name in ("probe", "plumb", "xs", "ys", "opt", "loss"):
    globals().pop(_name, None)
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

ALL = manifest()
roots = discover_roots(ARTIFACTS)
summary = None
GRID_COMPLETE = False
if PRELUDE_COMPLETE:
    carried = consolidate_resume_outputs(ALL, roots, ARTIFACTS, GRID_CONFIGS)
    todo = pending(ALL, roots, configs=GRID_CONFIGS)
    print(f"Manifest: {len(ALL)} runs; carried forward {carried}; pending {len(todo)}")
    if not ANALYSIS_ONLY:
        DEVICES = visible_devices()
        summary = execute_parallel(todo, features, devices=DEVICES,
            configs=GRID_CONFIGS, out_root=ARTIFACTS, roots=roots, guard=SESSION_GUARD,
            log=lambda msg: print(msg, flush=True))
        print(summary)
    left = pending(ALL, [ARTIFACTS], configs=GRID_CONFIGS)
    GRID_COMPLETE = not left
else:
    left = ALL
ANALYSIS_READY = GRID_COMPLETE and (ANALYSIS_ONLY or SESSION_GUARD.remaining_s > 600)
status = {
    "code_sha256": code_sha256(), "input_sha256": _input_sha256()[0],
    "manifest_runs": len(ALL), "prelude_complete": PRELUDE_COMPLETE,
    "grid_complete": GRID_COMPLETE, "analysis_ready": ANALYSIS_READY,
    "pending_run_ids": [c.run_id for c in left],
    "partial_checkpoints": [p.name for p in (ARTIFACTS / "checkpoints").glob("*.pt")],
    "elapsed_session_h": (time.perf_counter()-SESSION_T0)/3600 + SESSION_ALREADY_USED_H,
    "weekly_quota_entered_h": WEEKLY_GPU_HOURS_REMAINING,
    "summary": asdict(summary) if summary else None,
    "next": "Save Version output, stop accelerator/session, attach that output to the next session, recheck quota, Run All.",
}
(ARTIFACTS / "session_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
print(f"Grid complete: {GRID_COMPLETE}; analysis ready: {ANALYSIS_READY}")
if not ANALYSIS_READY:
    print(status["next"])
    print("Completed runs and validation probes are reused; interrupted training resumes at its last saved epoch.")
    print("After a complete grid, ANALYSIS_ONLY=True can render on Kaggle CPU without GPU quota.")
if summary and summary.completed:
    print(f"Observed throughput: {summary.completed/max(summary.wall_time_s,1)*3600:.1f} runs/hour this session.")
    print("New model and arm runtimes differ; this is not a promise that the full grid fits one weekly quota.")
''')['metadata']['itbtc']['writes'] += ['artifacts/checkpoints/*.pt','artifacts/weights/*.pt','artifacts/session_status.json']

for c in nb['cells']:
    m=c.get('metadata',{}).get('itbtc',{})
    if m.get('step') in ('rq1','rq2','rq3','save','report'):
        c['source']=''.join(c['source']).replace('if not GRID_COMPLETE:', 'if not ANALYSIS_READY:').splitlines(keepends=True)
        m['guarded_on']='ANALYSIS_READY'
    if m.get('step')=='save':
        s=''.join(c['source']).replace('"runs_complete": len(done),','"runs_complete": len(done),\n        "manifest_run_ids": sorted(c.run_id for c in ALL),\n        "training_protocol": "post-audit controlled rerun",')
        assert 'manifest_run_ids' in s
        c['source']=s.splitlines(keepends=True)

for c in nb['cells']:
    if c['cell_type']=='code':
        ast.parse(''.join(c['source']))
        c['outputs']=[];c['execution_count']=None
PATH.write_text(json.dumps(nb,indent=1,ensure_ascii=False)+'\n',encoding='utf-8')
print('Applied Kaggle quota preflight, shared deadline, resumable stages and analysis gating')
