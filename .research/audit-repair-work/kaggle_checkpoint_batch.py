from pathlib import Path
exec(Path(__file__).with_name('applied_first_batch.py').read_text(encoding='utf-8').split('# A13:')[0])

context_source = '''
_TRAINING_CONTEXT = threading.local()


class SessionBudgetExhausted(RuntimeError):
    """A recoverable pause, not a failed experiment."""


class TrainingSession:
    """Per-worker checkpoint roots and a monotonic deadline."""
    def __init__(self, out_root: Path, roots: list[Path], deadline: float):
        self.state = (Path(out_root), [Path(r) for r in roots], deadline)

    def __enter__(self):
        self.previous = getattr(_TRAINING_CONTEXT, "state", None)
        _TRAINING_CONTEXT.state = self.state
        return self

    def __exit__(self, *exc):
        _TRAINING_CONTEXT.state = self.previous


'''
c,node=cell_for('train.py','train_one')
lines=''.join(c['source']).splitlines(keepends=True)
lines[node.lineno-1:node.lineno-1]=[context_source]
c['source']=''.join(lines).splitlines(keepends=True)
function('train.py','train_one','''
def train_one(
    tensors: OriginTensors, spec: RunSpec, cfg: Architecture, *,
    device: torch.device | None = None, max_epochs: int | None = None,
    patience: int | None = None, lr: float | None = None,
    lr_halve_every: int | None = None, batch_size: int = 32,
) -> tuple[nn.Module, TrainOutcome]:
    """Train on one device, checkpoint every epoch, stop before session expiry.

    A checkpoint records optimizer/scheduler, best weights, epoch, patience and
    device RNG. An interrupted epoch is redone from its last completed boundary.
    Identity includes actual train indices, input/code/config and resolved schedule.
    Only weights_only=True loads are accepted; no arbitrary Python object loading.
    """
    device = device or pick_device()
    protocol = cfg.schedule() if hasattr(cfg, "schedule") else TrainSchedule()
    max_epochs = protocol.max_epochs if max_epochs is None else max_epochs
    patience = protocol.patience if patience is None else patience
    lr = protocol.lr if lr is None else lr
    lr_halve_every = protocol.lr_halve_every if lr_halve_every is None else lr_halve_every
    if min(max_epochs, patience, lr_halve_every, batch_size) < 1 or lr <= 0:
        raise ValueError("positive training schedule and batch size required")
    if not len(tensors.train) or not len(tensors.val):
        raise ValueError("training and validation must be nonempty")
    with SEED_LOCK:
        set_seed(spec.seed, device)
        model = cfg.build().to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    schedule = torch.optim.lr_scheduler.StepLR(optimiser, step_size=lr_halve_every, gamma=.5)
    loss_target = cfg.loss_target()
    x_tr, y_tr = _to_device(tensors.train, device, target=loss_target)
    x_va, y_va = _to_device(tensors.val, device, target=loss_target)
    best_val, best_state, stale = float("inf"), None, 0
    epochs_run, train_loss, prior_seconds = 0, float("nan"), 0.
    started = time.perf_counter()
    context = getattr(_TRAINING_CONTEXT, "state", None)
    checkpoint = None
    identity = None
    deadline = float("inf")
    if context is not None:
        out_root, roots, deadline = context
        checkpoint = out_root / "checkpoints" / f"{spec.run_id}.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        identity = json.loads(json.dumps({
            "spec": asdict(spec), "config": asdict(cfg), "code": code_sha256(),
            "input": _input_sha256()[0], "torch": str(torch.__version__),
            "device_type": device.type, "batch_size": batch_size,
            "schedule": [max_epochs, patience, lr, lr_halve_every],
            "train_times": hashlib.sha256(tensors.train.ts.tobytes()).hexdigest(),
        }))
        for root in dict.fromkeys([out_root, *roots]):
            candidate = root / "checkpoints" / f"{spec.run_id}.pt"
            if not candidate.exists():
                continue
            try:
                saved = torch.load(candidate, map_location="cpu", weights_only=True)
            except (OSError, RuntimeError, EOFError) as exc:
                raise ValueError(f"{candidate}: unreadable training checkpoint") from exc
            if saved.get("identity") != identity:
                continue
            model.load_state_dict(saved["model"])
            optimiser.load_state_dict(saved["optimizer"])
            schedule.load_state_dict(saved["scheduler"])
            best_state, best_val = saved["best_state"], saved["best_val"]
            epochs_run, stale = saved["epoch"], saved["stale"]
            train_loss, prior_seconds = saved["train_loss"], saved["wall_time_s"]
            if device.type == "cuda":
                torch.cuda.set_rng_state(saved["rng"], device=device)
            else:
                torch.set_rng_state(saved["rng"])
            break

    def save_boundary():
        if checkpoint is None:
            return
        staging = checkpoint.with_suffix(".pt.tmp")
        torch.save({
            "identity": identity, "model": model.state_dict(),
            "optimizer": optimiser.state_dict(), "scheduler": schedule.state_dict(),
            "best_state": best_state, "best_val": best_val, "epoch": epochs_run,
            "stale": stale, "train_loss": train_loss,
            "wall_time_s": prior_seconds + time.perf_counter() - started,
            "rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else torch.get_rng_state(),
        }, staging)
        staging.replace(checkpoint)

    # Epoch zero is recoverable even if the first epoch hits the deadline.
    if epochs_run == 0:
        save_boundary()
    for epoch in range(epochs_run + 1, max_epochs + 1):
        if stale >= patience:
            break
        if time.perf_counter() >= deadline:
            raise SessionBudgetExhausted(f"{spec.run_id}: resume from epoch {epochs_run}")
        model.train()
        order = torch.randperm(len(x_tr), device=device)
        running = 0.
        for i in range(0, len(order), batch_size):
            if time.perf_counter() >= deadline:
                raise SessionBudgetExhausted(f"{spec.run_id}: resume from epoch {epochs_run}")
            idx = order[i:i+batch_size]
            optimiser.zero_grad(set_to_none=True)
            fitted = model.forecast_target(x_tr[idx]) if loss_target == "target" else model(x_tr[idx])
            loss = nn.functional.mse_loss(fitted, y_tr[idx])
            if not torch.isfinite(loss):
                raise ValueError(f"{spec.run_id}: non-finite training loss")
            loss.backward()
            optimiser.step()
            running += loss.item()*len(idx)
        schedule.step()
        epochs_run = epoch
        train_loss = running/len(x_tr)
        val = _mean_loss(model, x_va, y_va, target=loss_target)
        if not np.isfinite(val):
            raise ValueError(f"{spec.run_id}: non-finite validation loss")
        if val < best_val - 1e-9:
            best_val, stale = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        save_boundary()
    if best_state is None:
        raise ValueError(f"{spec.run_id}: no finite validation checkpoint")
    model.load_state_dict(best_state)
    return model, TrainOutcome(
        run_id=spec.run_id, epochs_run=epochs_run, best_val_mse=best_val,
        train_loss=train_loss, wall_time_s=prior_seconds + time.perf_counter()-started,
        n_parameters=model.n_parameters(), device=str(device),
    )
''')
# Every fitted model is available for later inference; checkpoints are removed
# only after completed predictions and metadata have been durably written.
replace('train.py','    input_parquet = resolve_input_parquet()\n    input_digest, input_provenance = _input_sha256(input_parquet)', '''    weights_path = root / "weights" / f"{spec.run_id}.pt"
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    staging_weights = weights_path.with_suffix(".pt.tmp")
    torch.save(model.state_dict(), staging_weights)
    staging_weights.replace(weights_path)
    input_parquet = resolve_input_parquet()
    input_digest, input_provenance = _input_sha256(input_parquet)''')
replace('train.py','        "prediction_schema_version": 2,', '''        "prediction_schema_version": 2,
        "weights_sha256": hashlib.sha256(weights_path.read_bytes()).hexdigest(),
        "torch_version": str(torch.__version__),''')
replace('train.py','    staging_meta.replace(meta_path)\n    return preds_path, meta_path', '''    staging_meta.replace(meta_path)
    (root / "checkpoints" / f"{spec.run_id}.pt").unlink(missing_ok=True)
    return preds_path, meta_path''')
replace('runner.py','        return self.remaining_s > self.mean_run_s','        return self.remaining_s > max(120.0, 1.5 * max(self.durations[-20:], default=120.0))')
# Wrap only fitting: artifact saving has the separately reserved session margin.
for name in ('execute','execute_parallel'):
    c,node=cell_for('runner.py',name)
    s=''.join(c['source'])
    old='model, cfg, outcome = cell.model_config(configs).fit(\n'
    offset=s.index(old); indent=s[s.rfind('\n',0,offset)+1:offset]
    lines=s.splitlines(keepends=True)
    call=next(n for n in ast.walk(ast.parse(s)) if isinstance(n,ast.Assign) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='fit')
    chunk=''.join(lines[call.lineno-1:call.end_lineno])
    lines[call.lineno-1:call.end_lineno]=[indent+'with TrainingSession(out_root, roots, guard.deadline):\n'+''.join('    '+line for line in chunk.splitlines(keepends=True))]
    s=''.join(lines)
    marker=indent[:-4]+'except Exception as exc:'
    assert marker in s, name
    s=s.replace(marker,indent[:-4]+'except SessionBudgetExhausted as exc:\n'+indent+'log(f"PAUSED: {exc}; save this session output and attach it next session")\n'+indent+'break\n'+marker,1)
    c['source']=s.splitlines(keepends=True)
    c['metadata']['itbtc'].setdefault('projection_imports',[]).append('from itransformer_btc.train import TrainingSession, SessionBudgetExhausted')

for c in nb['cells']:
    if c['cell_type']=='code':
        ast.parse(''.join(c['source']))
        c['outputs']=[];c['execution_count']=None
PATH.write_text(json.dumps(nb,indent=1,ensure_ascii=False)+'\n',encoding='utf-8')
print('Applied per-worker epoch checkpoints, deadline pause and persisted best weights')
