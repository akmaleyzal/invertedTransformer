from pathlib import Path
exec(Path(__file__).with_name('applied_first_batch.py').read_text(encoding='utf-8').split('# A13:')[0])

function('splits.py', 'window_starts', '''
def window_starts(
    ts: np.ndarray, start: datetime, end: datetime, semantics: Semantics,
    span: int = WINDOW_SPAN, *, seq_len: int = SEQ_LEN,
) -> np.ndarray:
    """Input-start indices; test membership uses the FIRST TARGET timestamp (A01).

    A forecast at t consumes [t-L, t) and predicts [t, t+H). Training and
    validation keep their complete windows inside their respective spans.
    """
    if semantics not in ("contained", "origin") or span < 2:
        raise ValueError("invalid window semantics or span")
    if semantics == "origin" and not 0 < seq_len < span:
        raise ValueError("origin semantics require 0 < seq_len < span")
    ts = np.asarray(ts)
    if ts.ndim != 1 or np.any(np.diff(ts) <= 0):
        raise ValueError("timestamps must be strictly increasing and unique")
    lo, hi = int(start.timestamp() * 1000), int(end.timestamp() * 1000)
    if hi <= lo:
        raise ValueError("window end must follow start")
    first = np.arange(max(0, len(ts) - span + 1), dtype=np.int64)
    contiguous = (ts[first + span - 1] - ts[first]) == (span - 1) * HOUR_MS
    if semantics == "contained":
        inside = (ts[first] >= lo) & (ts[first + span - 1] < hi)
    else:
        issued = ts[first + seq_len]
        inside = (issued >= lo) & (issued < hi)
    return first[contiguous & inside]
''')
replace('splits.py', '        ts=ts[starts],', '        ts=ts[starts + seq_len],')
replace('splits.py', '    ts: np.ndarray     # (n,)      int64, window start — for traceability', '    ts: np.ndarray     # (n,) int64, forecast origin = first target bar open (UTC)')
replace('splits.py', 'window_starts(ts, lo, hi, "origin", span)', 'window_starts(ts, lo, hi, "origin", span, seq_len=seq_len)')
replace('budget.py', 'if all((start + k * HOUR_MS) in usable for k in range(WINDOW_SPAN)):', 'if all((start + (k - SEQ_LEN) * HOUR_MS) in usable for k in range(WINDOW_SPAN)):')
replace('train.py', '                    "timestamp": np.repeat(split.ts, h),', '                    "timestamp": np.repeat(split.ts, h),\n                    "forecast_origin": np.repeat(split.ts, h),\n                    "input_start": np.repeat(split.ts - cfg.seq_len * 3_600_000, h),\n                    "target_timestamp": (split.ts[:, None] + np.arange(h) * 3_600_000).reshape(-1),')
replace('train.py', '                "timestamp": pl.Int64,', '                "timestamp": pl.Int64,\n                "forecast_origin": pl.Int64,\n                "input_start": pl.Int64,\n                "target_timestamp": pl.Int64,')
replace('train.py', '        "run_id": spec.run_id,\n        "spec": asdict(spec),', '        "run_id": spec.run_id,\n        "prediction_schema_version": 2,\n        "timestamp_semantics": "forecast_origin",\n        "forecast_origin_definition": "first target bar open, UTC",\n        "evaluation_population": "surviving contiguous windows",\n        "selection_time_ms": int(tensors.origin.test_start.timestamp() * 1000),\n        "training_cutoff_ms": int(tensors.origin.train_sub_end.timestamp() * 1000),\n        "spec": asdict(spec),')

function('metrics.py', 'load_predictions', '''
def load_predictions(run_id: str, roots: list[Path]) -> pl.DataFrame:
    """Read predictions with explicit UTC target times and forecast-based blocks.

    A01: legacy input-start labels are corrected in memory, never on disk. Only
    existing forecasts survive: the first L missing forecast hours cannot be
    recovered by relabelling. ``legacy_block`` retains the original attribution.
    """
    from datetime import datetime, timezone
    path = _locate(run_id, roots, "preds", ".parquet")
    meta_path = path.parent.parent / "meta" / f"{run_id}.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    frame = pl.read_parquet(path)
    length = int(meta["config"]["seq_len"])
    horizon = int(meta["spec"]["pred_len"])
    if length < 1 or horizon < 1:
        raise ValueError(f"{run_id}: invalid lookback or horizon")
    if meta.get("timestamp_semantics") == "forecast_origin":
        required = {"input_start", "forecast_origin", "target_timestamp"}
        if not required <= set(frame.columns):
            raise ValueError(f"{run_id}: incomplete timestamp schema")
    elif "timestamp_semantics" not in meta:
        frame = frame.with_columns(
            pl.col("block").alias("legacy_block"),
            pl.col("timestamp").alias("input_start"),
            (pl.col("timestamp") + length * HOUR_MS).alias("forecast_origin"),
        ).with_columns(pl.col("forecast_origin").alias("timestamp"))
        frame = frame.with_columns(
            (pl.col("timestamp") + (pl.col("step").cast(pl.Int64) - 1) * HOUR_MS)
            .alias("target_timestamp")
        )
        # All declared base origins are month starts; the fresh label appends +90d.
        origin = datetime.fromisoformat(str(meta["origin"])[:7] + "-01").replace(tzinfo=timezone.utc)
        origin_ms = int(origin.timestamp() * 1000)
        frame = frame.with_columns(
            (((pl.col("timestamp") - origin_ms) // (BLOCK_HOURS * HOUR_MS)) + 1)
            .cast(pl.Int32).alias("block")
        ).filter(pl.col("block").is_in(meta.get("block_labels", [1, 2, 3, 4, 5, 6])))
    else:
        raise ValueError(f"{run_id}: unknown timestamp semantics")
    if frame.filter(
        (pl.col("timestamp") != pl.col("forecast_origin")) |
        (pl.col("forecast_origin") - pl.col("input_start") != length * HOUR_MS) |
        (pl.col("target_timestamp") != pl.col("timestamp") +
         (pl.col("step").cast(pl.Int64) - 1) * HOUR_MS)
    ).height:
        raise ValueError(f"{run_id}: inconsistent target timestamps")
    counts = frame.group_by("timestamp").agg(
        pl.len().alias("n"), pl.col("step").n_unique().alias("unique"),
        pl.col("step").min().alias("first"), pl.col("step").max().alias("last"),
    )
    if counts.filter((pl.col("n") != horizon) | (pl.col("unique") != horizon) |
                     (pl.col("first") != 1) | (pl.col("last") != horizon)).height:
        raise ValueError(f"{run_id}: incomplete or duplicated forecast horizon")
    if frame.select(pl.any_horizontal(pl.col("y_true", "y_pred").is_null() |
                                     ~pl.col("y_true", "y_pred").is_finite()).any()).item():
        raise ValueError(f"{run_id}: non-finite predictions")
    return frame.sort(["block", "timestamp", "step"])
''')
function('metrics.py', 'assert_same_windows', '''
def assert_same_windows(left: pl.DataFrame, right: pl.DataFrame, what: str) -> None:
    """Compare every forecast/step key, including actual target times (A01)."""
    columns = ["block", "timestamp", "step"]
    if "target_timestamp" in left.columns or "target_timestamp" in right.columns:
        if not all("target_timestamp" in f.columns for f in (left, right)):
            raise ValueError(f"{what}: target timestamp contract missing on one side")
        columns.append("target_timestamp")
    a, b = [f.select(columns).sort(columns) for f in (left, right)]
    if a.is_duplicated().any() or b.is_duplicated().any() or not a.equals(b):
        raise ValueError(f"{what}: evaluated window sets differ ({a.height} vs {b.height} points)")
''')

# A04: no model-nesting theorem has been established for the fitted procedures.
function('comparisons.py', 'nesting_order', '''
def nesting_order(left: ModelKey, right: ModelKey) -> tuple[ModelKey, ModelKey] | None:
    """No automatic CW classification for this study's fitted procedures (A04).

    Nested feature sets do not establish nested neural function classes or the
    sampling assumptions of CW. Comparisons use unadjusted forecast losses.
    """
    return None
''')
# A05: retain seed-average losses separately from ensemble predictions.
replace('comparisons.py', '    pred_len: int\n', '    pred_len: int\n    seed_losses: dict[tuple[ModelKey, int], np.ndarray] | None = None\n')
function('comparisons.py', 'build_panel', '''
def build_panel(
    keys: list[ModelKey], roots: list[Path], pred_len: int = PRED_LEN,
    origin_indices: tuple[int, ...] | None = None,
) -> PredictionPanel:
    """A01/A05: identical actual targets; seed-average losses are the estimand.

    Ensemble predictions are retained for inspection but are not substituted
    for single-training-procedure loss. Naive loss provides scale-free contrasts.
    """
    if not any(key != NAIVE for key in keys):
        raise ValueError("a comparison panel needs a persisted forecast")
    indices = origin_indices or tuple(o.index for o in ORIGINS)
    block, y_true, y_pred, seed_losses = {}, {}, {}, {}
    vintages = set()
    for index in indices:
        signature = None
        naive_z = None
        for key in keys:
            if key == NAIVE:
                continue
            runs = _run_ids(key, index, roots, pred_len)
            if not runs:
                raise FileNotFoundError(f"{key} has no run at origin {index} (H={pred_len})")
            stacked = []
            for run_id in runs:
                meta = load_meta(run_id, roots)
                vintages.add((meta.get("input_sha256"), meta.get("code_sha256")))
                frame = load_predictions(run_id, roots)
                sig = frame.select("block", "timestamp", "step", "target_timestamp")
                actual = frame["y_true"].to_numpy().astype(np.float64)
                if signature is None:
                    signature = sig
                    block[index] = frame["block"].to_numpy()
                    y_true[index] = actual
                    naive_z = float(meta["naive_rw_z"])
                elif not sig.equals(signature):
                    raise ValueError(f"{run_id}: evaluated window sets differ at origin {index}")
                elif not np.allclose(actual, y_true[index], rtol=1e-6, atol=1e-8):
                    raise ValueError(f"{run_id}: target values or scaler differ")
                stacked.append(frame["y_pred"].to_numpy().astype(np.float64))
            y_pred[key, index] = np.mean(stacked, axis=0)
            seed_losses[key, index] = np.mean(
                np.square(y_true[index][None, :] - np.stack(stacked)), axis=0
            )
        y_pred[NAIVE, index] = np.full(len(y_true[index]), naive_z)
        seed_losses[NAIVE, index] = np.square(y_true[index] - naive_z)
    if len(vintages) != 1:
        raise ValueError("comparison panel mixes code or input vintages")
    return PredictionPanel(
        keys=tuple(keys), origin_indices=tuple(indices),
        origins=tuple(ORIGINS[i-1].label for i in indices), block=block,
        y_true=y_true, y_pred=y_pred, pred_len=pred_len, seed_losses=seed_losses,
    )
''')
function('comparisons.py', 'differential', '''
def differential(panel: PredictionPanel, left: ModelKey, right: ModelKey,
                 origin_index: int) -> np.ndarray:
    """Per-forecast mean step loss: average losses across seeds BEFORE comparing.

    No CW adjustment: A04 does not establish the required model nesting.
    """
    def loss(key: ModelKey) -> np.ndarray:
        if panel.seed_losses is not None:
            return panel.seed_losses[key, origin_index]
        return np.square(panel.y_true[origin_index] - panel.y_pred[key, origin_index])
    return _per_window(loss(left) - loss(right), panel.pred_len)
''')
function('comparisons.py', 'per_origin_loss', '''
def per_origin_loss(panel: PredictionPanel, key: ModelKey) -> np.ndarray:
    """Equal-weight block RelMSE from seed-average step loss (A05)."""
    means = []
    for index in panel.origin_indices:
        loss = (panel.seed_losses[key, index] if panel.seed_losses is not None else
                np.square(panel.y_true[index] - panel.y_pred[key, index]))
        values = []
        for b in np.unique(panel.block[index]):
            mask = panel.block[index] == b
            denominator = (panel.seed_losses[NAIVE, index][mask].mean()
                           if panel.seed_losses is not None else 1.0)
            if denominator <= 0:
                raise ValueError("Naive-RW block MSE must be positive")
            values.append(loss[mask].mean() / denominator)
        means.append(np.mean(values))
    return np.asarray(means)
''')
function('comparisons.py', 'per_origin_differential', '''
def per_origin_differential(panel: PredictionPanel, left: ModelKey,
                            right: ModelKey) -> np.ndarray:
    """Equal-weight block contrast, matching the report's aggregation (A05)."""
    return per_origin_loss(panel, left) - per_origin_loss(panel, right)
''')

# Scope all inferential output; CR1/iid-origin resampling cannot prove independence.
replace('comparisons.py', '            "statistic_name": "Clark-West" if nested else "DM-HLN",', '            "statistic_name": "unadjusted forecast-loss diagnostic",\n            "inference_status": "exploratory; cross-origin dependence unresolved",\n            "estimand": "mean seed loss, then equal block means",')

for c in nb['cells']:
    if c['cell_type']=='code':
        ast.parse(''.join(c['source']))
        c['outputs']=[]; c['execution_count']=None
PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False)+'\n', encoding='utf-8')
print('Edited timing and forecast-loss contracts in the notebook')
