from pathlib import Path
exec(Path(__file__).with_name('applied_first_batch.py').read_text(encoding='utf-8').split('# A13:')[0])

# A08: hold training fixed while updating validation/checkpoint selection.
c,n=cell_for('config.py','FalsificationOrigin')
c['source'] += '''

@dataclass(frozen=True, slots=True)
class ValidationRefreshOrigin(FalsificationOrigin):
    """Original training span, refreshed validation, same B4--B6 targets."""
    @property
    def train_start(self) -> datetime:
        return self.base.train_start

    @property
    def train_sub_end(self) -> datetime:
        return self.base.train_sub_end

    @property
    def val_start(self) -> datetime:
        return self.base.train_sub_end + self._shift

    @property
    def label(self) -> str:
        return f"{self.base.label}+validation{self.offset_days}d"
'''.splitlines(keepends=True)

# Defaults remain unrestricted for data-plane diagnostics; ALL runner arms use
# the fixed protocol below. Sampling is based only on training timestamps.
replace('splits.py','    block_labels: tuple[int, ...]\n','    block_labels: tuple[int, ...]\n    training_selection: dict | None = None\n    representation: dict | None = None\n')
c,n=cell_for('splits.py','build_origin_tensors')
s=''.join(c['source'])
s=s.replace('    columns: tuple[str, ...] | None = None,','    columns: tuple[str, ...] | None = None,\n    train_window_limit: int | None = None,\n    selection_seed: int = 1729,\n    representation: str = "identity",')
assert 'train_window_limit:' in s
s=s.replace('    scaled = scaler.transform(values)\n','''    scaled = scaler.transform(values)
    n_available = len(train_idx)
    fit_rows = scaled[train_idx[0]:train_idx[-1] + span].astype(np.float64)
    representation_meta = None
    if representation != "identity":
        if k != 8 or representation not in ("repr_identity", "whiten", "correlate"):
            raise ValueError("representation controls require the same K=8 base columns")
        covariance = np.cov(fit_rows[:, 1:], rowvar=False, ddof=0)
        eig, vectors = np.linalg.eigh(covariance)
        floor = max(float(eig.max()) * 1e-6, 1e-8)
        transform = np.eye(k)
        if representation != "repr_identity":
            transform[1:, 1:] = (vectors * (1. / np.sqrt(np.maximum(eig, floor)))) @ vectors.T
        if representation == "correlate":
            corr = .95 * np.ones((k-1, k-1)) + .05 * np.eye(k-1)
            ev, q = np.linalg.eigh(corr)
            transform[1:, 1:] = transform[1:, 1:] @ ((q * np.sqrt(ev)) @ q.T)
        if np.linalg.matrix_rank(transform) != k or np.linalg.cond(transform) > 1e6:
            raise ValueError("representation is not safely invertible")
        represented = fit_rows @ transform
        eigenvalues = np.linalg.eigvalsh(np.corrcoef(represented, rowvar=False))
        representation_meta = {
            "name": representation, "matrix": transform.tolist(),
            "inverse": np.linalg.inv(transform).tolist(),
            "condition_number": float(np.linalg.cond(transform)),
            "training_pr": float(eigenvalues.sum()**2 / (eigenvalues**2).sum()),
            "fit_scope": "purged training rows only", "target_preserved": True,
            "use_norm": False, "eigenvalue_floor": floor,
        }
        scaled = (scaled @ transform).astype(np.float32)
    if train_window_limit is not None:
        if train_window_limit < 1 or n_available < train_window_limit:
            raise ValueError(f"{origin.label}: {n_available} training windows < required {train_window_limit}")
        selected = np.random.default_rng(selection_seed).choice(n_available, train_window_limit, replace=False)
        train_idx = train_idx[np.sort(selected)]
    selection = {"available": n_available, "selected": len(train_idx),
                 "limit": train_window_limit, "seed": selection_seed,
                 "forecast_times_sha256": hashlib.sha256(ts[train_idx + seq_len].tobytes()).hexdigest(),
                 "scaler_fit": "all purged training rows before subsampling"}
''')
s=s.replace('        block_labels=tuple(label for label, _, _ in blocks),','        block_labels=tuple(label for label, _, _ in blocks),\n        training_selection=selection, representation=representation_meta,')
assert 'training_selection=selection' in s
c['source']=s.splitlines(keepends=True)
c['metadata']['itbtc'].setdefault('projection_imports',[]).append('import hashlib')

replace('runner.py','    "fresh": "itrf",','    "valrefresh": "itrv",\n    "repr_identity": "repi",\n    "repr_whiten": "repw",\n    "repr_correlate": "repc",\n    "dlinear_all": "dlina",\n    "patchtst_all": "ptsta",\n    "fresh": "itrf",')
replace('runner.py','    "main", "uniform", "fresh", "horizon", *BASELINE_ARMS, *ROBUSTNESS_ARMS,','    "main", "uniform", "fresh", "horizon", *BASELINE_ARMS, *ROBUSTNESS_ARMS,\n    "valrefresh", "repr_identity", "repr_whiten", "repr_correlate",\n    "dlinear_all", "patchtst_all",')
c,n=cell_for('runner.py','RunCell')
s=''.join(c['source']).replace('self.arm == "fresh",','self.arm if self.arm in ("fresh", "valrefresh") else "base",')
s=s.replace('self.seq_len, self.subset)','self.seq_len, self.subset, self.representation())')
s=s.replace('    def columns(self)', '''    def representation(self) -> str:
        return {"repr_identity": "repr_identity", "repr_whiten": "whiten",
                "repr_correlate": "correlate"}.get(self.arm, "identity")

    def columns(self)''')
s=s.replace('        return FalsificationOrigin(base) if self.arm == "fresh" else base','        if self.arm == "valrefresh":\n            return ValidationRefreshOrigin(base)\n        return FalsificationOrigin(base) if self.arm == "fresh" else base')
s=s.replace('        if self.arm == "dlinear":\n            return DLinearConfig(pred_len=self.pred_len)','''        if self.arm in ("dlinear", "dlinear_all"):
            selected = (overrides or {}).get(self.arm, DLinearConfig(pred_len=self.pred_len))
            return replace(selected, loss_channels="all" if self.arm.endswith("_all") else "target")''')
s=s.replace('        if self.arm == "patchtst":\n            return PatchTSTConfig(pred_len=self.pred_len)','''        if self.arm in ("patchtst", "patchtst_all"):
            selected = (overrides or {}).get(self.arm, PatchTSTConfig(pred_len=self.pred_len))
            return replace(selected, loss_channels="all" if self.arm.endswith("_all") else "target")''')
s=s.replace('        if self.arm == "longsched":','        if self.arm.startswith("repr_"):\n            return ITransformerConfig(pred_len=self.pred_len, use_norm=False)\n        if self.arm == "longsched":')
c['source']=s.splitlines(keepends=True)
c['metadata']['itbtc'].setdefault('projection_imports',[]).extend(['from itransformer_btc.config import ValidationRefreshOrigin','from dataclasses import replace'])

c,n=cell_for('runner.py','manifest'); s=''.join(c['source'])
s=s.replace('RunCell("fresh", o.index, 8, PRED_LEN, SEEDS[0]) for o in ORIGINS','RunCell("fresh", o.index, 8, PRED_LEN, s) for o in ORIGINS for s in SEEDS')
s=s.replace('    seen: set[str] = set()', '''    for arm in ("valrefresh", "repr_identity", "repr_whiten", "repr_correlate", "dlinear_all", "patchtst_all"):
        if arm in arms:
            cells += [RunCell(arm, o.index, 8, PRED_LEN, s) for o in ORIGINS for s in SEEDS]
    seen: set[str] = set()''')
c['source']=s.splitlines(keepends=True)

c,n=cell_for('runner.py','_TensorCache')
s=''.join(c['source']).replace('            columns=cell.columns(),','            columns=cell.columns(), train_window_limit=11_500,\n            representation=cell.representation(),')
assert 'train_window_limit=11_500' in s
c['source']=s.splitlines(keepends=True)

replace('train.py','        "n_train": len(tensors.train),','        "n_train": len(tensors.train),\n        "training_selection": tensors.training_selection,\n        "representation": tensors.representation,\n        "effective_input_channels": 1 if getattr(cfg, "channel_independent", False) and cfg.loss_target() == "target" else tensors.k,')

# Target-only CI training must not mix other channels through BatchNorm stats.
replace('baselines.py','        return self(x)[:, :, TARGET_INDEX]','        return self(x[:, :, TARGET_INDEX:TARGET_INDEX+1] if self.cfg.loss_target() == "target" else x)[:, :, 0 if self.cfg.loss_target() == "target" else TARGET_INDEX]')
c,n=cell_for('baselines.py','PatchTST');s=''.join(c['source'])
t=ast.parse(s); old=next(n for n in ast.walk(t) if isinstance(n,ast.Assign) and any(isinstance(v,ast.Name) and v.id=='block' for v in n.targets))
lines=s.splitlines(keepends=True); del lines[old.lineno-1:old.end_lineno]
s=''.join(lines).replace('nn.init.normal_(self.position, std=0.02)','nn.init.uniform_(self.position, -0.02, 0.02)')
s=s.replace('mean = x.mean(dim=1, keepdim=True)','mean = x.mean(dim=1, keepdim=True).detach()')
c['source']=s.splitlines(keepends=True)

for c in nb['cells']:
    if c['cell_type']=='code':
        ast.parse(''.join(c['source']))
        c['outputs']=[];c['execution_count']=None
PATH.write_text(json.dumps(nb,indent=1,ensure_ascii=False)+'\n',encoding='utf-8')
print('Added controlled training counts, refresh decomposition and invertible representation arms')
