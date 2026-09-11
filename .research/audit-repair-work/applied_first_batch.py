import ast
import copy
import json
import textwrap
from pathlib import Path

ROOT = Path('D:/pythonProject/invertedTransformer')
PATH = ROOT / 'notebooks/iTransformer.ipynb'
nb = json.loads(PATH.read_text(encoding='utf-8'))
previous = copy.deepcopy(nb)

def cell_for(module, name):
    for cell in nb['cells']:
        tag = cell.get('metadata', {}).get('itbtc', {})
        if tag.get('module') != module:
            continue
        for node in ast.parse(''.join(cell['source'])).body:
            if getattr(node, 'name', None) == name:
                return cell, node
    raise AssertionError((module, name))

def function(module, name, source):
    cell, node = cell_for(module, name)
    lines = ''.join(cell['source']).splitlines(keepends=True)
    lines[node.lineno - 1:node.end_lineno] = [textwrap.dedent(source).strip() + '\n']
    cell['source'] = ''.join(lines).splitlines(keepends=True)

def replace(module, old, new):
    matches = [c for c in nb['cells'] if c.get('metadata', {}).get('itbtc', {}).get('module') == module and old in ''.join(c['source'])]
    assert len(matches) == 1, (module, old[:80], len(matches))
    c = matches[0]
    c['source'] = ''.join(c['source']).replace(old, new).splitlines(keepends=True)

def step(slug, old, new):
    matches = [c for c in nb['cells'] if c.get('metadata', {}).get('itbtc', {}).get('step') == slug]
    assert len(matches) == 1
    c = matches[0]
    assert old in ''.join(c['source']), (slug, old)
    c['source'] = ''.join(c['source']).replace(old, new).splitlines(keepends=True)

# A13: one strict decision feeds pending, both executors, and remaining counts.
function('train.py', 'is_complete', '''
def is_complete(
    run_id: str, root: Path = ARTIFACTS, *, strict: bool = False,
    cfg: Architecture | None = None, columns: tuple[str, ...] | None = None,
) -> bool:
    """Read completion, or require the current request/code/input for resume.

    A13: missing provenance fails closed on the execution path. Historical
    readers may inspect complete artifacts without claiming they match today.
    """
    preds = root / "preds" / f"{run_id}.parquet"
    meta_path = root / "meta" / f"{run_id}.json"
    if not (preds.is_file() and meta_path.is_file()):
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("status") != "complete":
            return False
        if not strict:
            return True
        digest, _ = _input_sha256()
        if digest == "unknown" or meta.get("input_sha256") != digest:
            return False
        if meta.get("code_sha256") != code_sha256() or meta.get("run_id") != run_id:
            return False
        if cfg is None or meta.get("requested_config") != asdict(cfg):
            return False
        schedule = asdict(cfg.schedule()) if hasattr(cfg, "schedule") else None
        if meta.get("schedule") != schedule:
            return False
        if columns is not None and meta.get("variates") != list(columns):
            return False
        pl.read_parquet_schema(preds)
        return True
    except (OSError, ValueError, TypeError, pl.exceptions.PolarsError):
        return False
''')
function('train.py', '_input_sha256', '''
def _input_sha256(parquet: Path | str | None = None) -> tuple[str, str]:
    """Hash the actual input bytes; a sibling report is not proof of identity."""
    try:
        return hashlib.sha256(resolve_input_parquet(parquet).read_bytes()).hexdigest(), "file-digest"
    except OSError:
        return "unknown", "unresolved"
''')
replace('train.py', '    attention: "pl.DataFrame | None" = None,\n', '    attention: "pl.DataFrame | None" = None,\n    requested_config: Architecture | None = None,\n')
replace('train.py', '        "config": asdict(cfg),\n', '        "config": asdict(cfg),\n        "requested_config": asdict(requested_config or cfg),\n')
function('runner.py', 'pending', '''
def pending(
    cells: list[RunCell], roots: list[Path], *,
    configs: dict[str, Architecture] | None = None,
) -> list[RunCell]:
    """Requests not complete for the current code, input, and resolved options.

    The notebook supplies its validation-selected config. Without that config
    a tuned request is unresolved, so it cannot be declared complete.
    """
    todo = []
    for cell in cells:
        if cell.arm == "tuned" and (not configs or "tuned" not in configs):
            todo.append(cell)
            continue
        cfg = cell.model_config(configs)
        # First existing artifact wins, as it does for the prediction reader.
        candidates = [root for root in roots if
                      (root / "preds" / f"{cell.run_id}.parquet").exists() or
                      (root / "meta" / f"{cell.run_id}.json").exists()]
        if not candidates or not is_complete(
            cell.run_id, candidates[0], strict=True, cfg=cfg, columns=cell.columns()
        ):
            todo.append(cell)
    return todo
''')
replace('runner.py', '            if wanted and meta.get("code_sha256") not in (None, wanted):', '            if wanted and meta.get("code_sha256") != wanted:')
for name in ('execute', 'execute_parallel'):
    c, n = cell_for('runner.py', name)
    source = ''.join(c['source'])
    source = source.replace('    done = completed_run_ids(roots)', '    pending_ids = {c.run_id for c in pending(cells, roots, configs=configs)}')
    source = source.replace('if cell.run_id in done or is_complete(cell.run_id, out_root):', 'if cell.run_id not in pending_ids:')
    source = source.replace('remaining=len(pending(queue, discover_roots(out_root))),', 'remaining=len(pending(queue, roots, configs=configs)),')
    source = source.replace('root=out_root, attention=maps,', 'root=out_root, attention=maps,\n                requested_config=cell.model_config(configs),')
    c['source'] = source.splitlines(keepends=True)
for slug in ('grid', 'save'):
    c = next(c for c in nb['cells'] if c.get('metadata', {}).get('itbtc', {}).get('step') == slug)
    source = ''.join(c['source']).replace('pending(ALL, roots)', 'pending(ALL, roots, configs=GRID_CONFIGS)').replace('pending(ALL, discover_roots(ARTIFACTS))', 'pending(ALL, discover_roots(ARTIFACTS), configs=GRID_CONFIGS)')
    c['source'] = source.splitlines(keepends=True)

# A10: inverse scaling happens before any step/cumulative sign calculation.
replace('metrics.py', 'def directional_accuracy(frame: pl.DataFrame) -> DirectionalAccuracy:', 'def directional_accuracy(\n    frame: pl.DataFrame, *, sigma_g: float = 1.0, mu_g: float = 0.0,\n) -> DirectionalAccuracy:')
replace('metrics.py', '    last_step = int(frame.get_column("step").max())', '    if not np.isfinite(sigma_g) or sigma_g <= 0 or not np.isfinite(mu_g):\n        raise ValueError("directional accuracy requires a finite mean and positive scale")\n    frame = frame.with_columns(\n        (pl.col("y_true") * sigma_g + mu_g).alias("y_true"),\n        (pl.col("y_pred") * sigma_g + mu_g).alias("y_pred"),\n    )\n    last_step = int(frame.get_column("step").max())')
replace('metrics.py', '        da = directional_accuracy(load_predictions(run_id, roots))', '        da = directional_accuracy(\n            load_predictions(run_id, roots),\n            sigma_g=float(meta["sigma_g"]), mu_g=float(meta["mu_g"]),\n        )')

# A11: MAR=0 downside deviation uses all periods, including zero contributions.
replace('economics.py', '    downside = net[net < 0.0]\n    downside_sd = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0', '    downside_sd = float(np.sqrt(np.mean(np.minimum(net, 0.0) ** 2)))')
replace('economics.py', '    peak = np.maximum.accumulate(equity)', '    peak = np.maximum.accumulate(np.maximum(equity, 1.0))')

# A03: fixed effects and sampling clusters are separate explicit inputs.
function('metrics.py', 'j_test', '''
def j_test(
    y: np.ndarray, x_a: np.ndarray, x_b: np.ndarray, groups: np.ndarray,
    *, clusters: np.ndarray | None = None,
) -> tuple[float, float]:
    """Davidson-MacKinnon J diagnostic with CR1 covariance and t(G-1).

    ``groups`` identify fixed effects; ``clusters`` identify origins. If omitted,
    each fixed-effect group is a cluster. Overlap BETWEEN origins remains a
    design limitation: this diagnostic is not confirmatory inference (A03).
    """
    y, x_a, x_b = [np.asarray(v, dtype=np.float64) for v in (y, x_a, x_b)]
    groups = np.asarray(groups)
    clusters = groups if clusters is None else np.asarray(clusters)
    if any(v.ndim != 1 or len(v) != len(y) for v in (y, x_a, x_b, groups, clusters)):
        raise ValueError("J-test arrays must be one-dimensional with equal lengths")
    if not all(np.all(np.isfinite(v)) for v in (y, x_a, x_b)):
        raise ValueError("J-test inputs must be finite")
    def _demean(v: np.ndarray) -> np.ndarray:
        out = v.copy()
        for group in np.unique(groups):
            mask = groups == group
            out[mask] -= v[mask].mean()
        return out
    yd, ad, bd = [_demean(v) for v in (y, x_a, x_b)]
    n, g = len(y), len(np.unique(clusters))
    dof = n - 2 - len(np.unique(groups))
    if g < 2 or dof <= 0 or not bd @ bd > 0:
        return float("nan"), float("nan")
    fitted_b = bd * float((bd @ yd) / (bd @ bd))
    design = np.column_stack([ad, fitted_b])
    if np.linalg.matrix_rank(design) < 2:
        return float("nan"), float("nan")
    coef = np.linalg.lstsq(design, yd, rcond=None)[0]
    resid = yd - design @ coef
    bread = np.linalg.pinv(design.T @ design)
    scores = np.array([design[clusters == c].T @ resid[clusters == c]
                       for c in np.unique(clusters)])
    cov = (g / (g - 1)) * ((n - 1) / dof) * bread @ (scores.T @ scores) @ bread
    se = math.sqrt(max(float(cov[1, 1]), 0.0))
    if se <= 0:
        return float("nan"), float("nan")
    statistic = float(coef[1] / se)
    return statistic, 2.0 * _upper_tail(abs(statistic), g - 1)
''')
step('rq1', 'race["k_eff"].to_numpy(), groups)', 'race["k_eff"].to_numpy(), groups,\n                        clusters=race["origin_index"].to_numpy())')
step('rq1', 'race["k"].to_numpy().astype(float), groups)', 'race["k"].to_numpy().astype(float), groups,\n                        clusters=race["origin_index"].to_numpy())')

# A07: explicit exploratory change; future blocks cannot set the reference.
replace('metrics.py', '    reference = sel.group_by("origin").agg(pl.col("r2_oos").mean().alias("r2_ref"))', '    if sel.group_by("origin").agg(pl.col("block").n_unique().alias("n")).filter(pl.col("n") != 6).height:\n        raise ValueError("decay requires all six blocks per origin")\n    reference = sel.filter(pl.col("block") == 1).select(\n        "origin", pl.col("r2_oos").alias("r2_ref")\n    )\n    if reference.height != sel.get_column("origin").n_unique():\n        raise ValueError("decay requires exactly one first-block reference per origin")')

# Save the original evidence separately, then invalidate outputs of the changed program.
archive = ROOT / 'notebooks/outputs/iTransformer_before_A01_A15.ipynb'
if not archive.exists():
    archive.write_bytes(PATH.read_bytes())
for c in nb['cells']:
    if c['cell_type'] == 'code':
        ast.parse(''.join(c['source']))
        c['outputs'] = []
        c['execution_count'] = None
PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
print('Edited notebook only; previous executed notebook archived at', archive)
