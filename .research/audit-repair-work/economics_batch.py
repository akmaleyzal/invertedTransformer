from pathlib import Path
exec(Path(__file__).with_name('applied_first_batch.py').read_text(encoding='utf-8').split('# A13:')[0])

function('economics.py', 'positions', '''
def positions(preds: pl.DataFrame, sigma_g: float, mu_g: float) -> pl.DataFrame:
    """Exploratory long/cash daily decisions from inverse-scaled forecasts.

    Observed complete targets are selected retrospectively. This conditional
    simulation cannot establish executable performance across missing outcomes.
    """
    if not np.isfinite(sigma_g) or sigma_g <= 0 or not np.isfinite(mu_g):
        raise ValueError("finite mean and positive scale are required")
    per_window = preds.group_by("timestamp").agg(
        pl.col("y_pred").sum().alias("_f_z"), pl.col("y_true").sum().alias("_a_z"),
        pl.len().alias("_n_steps"), pl.col("step").n_unique().alias("_unique"),
        pl.col("step").min().alias("_first"), pl.col("step").max().alias("_last"),
        pl.col("block").first().alias("block"),
    ).sort("timestamp")
    if per_window.filter((pl.col("_n_steps") != 24) | (pl.col("_unique") != 24) |
                         (pl.col("_first") != 1) | (pl.col("_last") != 24)).height:
        raise ValueError("economics requires complete 24-step daily forecasts")
    if preds.select(pl.any_horizontal(pl.col("y_true", "y_pred").is_null() |
                                     ~pl.col("y_true", "y_pred").is_finite()).any()).item():
        raise ValueError("non-finite economic inputs")
    keep = non_overlapping_mask(per_window["timestamp"].to_numpy())
    return per_window.filter(pl.Series(keep)).with_columns(
        (pl.col("_f_z") * sigma_g + 24 * mu_g).alias("forecast_raw"),
        (pl.col("_a_z") * sigma_g + 24 * mu_g).alias("realised_raw"),
    ).with_columns((pl.col("forecast_raw") > 0).cast(pl.Float64).alias("position")).select(
        "timestamp", "block", "position", "realised_raw", "forecast_raw"
    )
''')
function('economics.py', 'net_returns', '''
def net_returns(position: np.ndarray, realised: np.ndarray, slippage_per_side: float) -> np.ndarray:
    """Self-financing log equity increments of separate daily long/cash trades.

    Every long trade opens from cash and closes after 24h, including the final
    trade and trades beside missing days. Buy at P*(1+c), sell at P'*(1-c):
    wealth multiplier = exp(realised)*(1-c)/(1+c). Cash earns zero.
    This explicit round-trip policy replaces the old unfinanced short ledger.
    """
    position, realised = np.asarray(position), np.asarray(realised)
    if position.ndim != 1 or position.shape != realised.shape or not len(position):
        raise ValueError("position and realised must be nonempty equal 1D arrays")
    if not np.isfinite(position).all() or not np.isfinite(realised).all():
        raise ValueError("non-finite ledger input")
    if not np.isin(position, [0., 1.]).all():
        raise ValueError("this spot simulation permits only long/cash positions")
    if not np.isfinite(slippage_per_side) or slippage_per_side < 0:
        raise ValueError("slippage must be finite and nonnegative")
    cost = TAKER_FEE_PER_SIDE + slippage_per_side
    if cost >= 1:
        raise ValueError("per-side cost must be below one")
    return position * (realised + math.log1p(-cost) - math.log1p(cost))
''')
replace('economics.py', '    mean = float(net.mean())\n    sd = float(net.std(ddof=1))\n    downside_sd = float(np.sqrt(np.mean(np.minimum(net, 0.0) ** 2)))', '''    simple = np.expm1(net)
    mean = float(simple.mean())
    sd = float(simple.std(ddof=1))
    downside_sd = float(np.sqrt(np.mean(np.minimum(simple, 0.0) ** 2)))''')
replace('economics.py', 'turnover_per_period=float(np.abs(np.diff(position, prepend=0.0)).mean() / 2.0),', 'turnover_per_period=float(np.mean(position)),')
function('economics.py', '_flat_days', '''
def _flat_days(frame: pl.DataFrame) -> int:
    """Unavailable daily slots, NOT a bound on unobserved loss or drawdown.

    The legacy result field n_flat_days is retained for file compatibility;
    it counts excluded slots, not verified live decisions to remain in cash.
    """
    return int(frame["block"].n_unique() * BLOCK_DAYS - frame.height)
''')
function('economics.py', 'economics_table', '''
def economics_table(
    roots: list[Path], keys: list[tuple[str, int]], origin_indices: tuple[int, ...],
    slippages: tuple[float, ...] = SLIPPAGE_BAND, pred_len: int = PRED_LEN, seed: int = 42,
) -> pl.DataFrame:
    """Conditional long/cash simulation; inferential trading claims withheld."""
    if pred_len != 24:
        raise ValueError("the economic protocol requires H=24")
    rows = []
    for origin_index in origin_indices:
        n_trials = len(_origin_run_ids(roots, origin_index, pred_len))
        for model, k in keys:
            run_id = f"{model}_o{origin_index:02d}_K{k:02d}_H{pred_len:03d}_s42"
            meta = load_meta(run_id, roots)
            frame = positions(load_predictions(run_id, roots), float(meta["sigma_g"]), float(meta["mu_g"]))
            pos, realised = [frame[n].to_numpy().astype(np.float64) for n in ("position", "realised_raw")]
            for slippage in slippages:
                result = summarise(pos, realised, slippage, _flat_days(frame), mdd_interval=False, seed=seed)
                hold = summarise(np.ones(len(pos)), realised, slippage, _flat_days(frame), mdd_interval=False)
                rows.append({
                    "model": f"{model}-K{k}", "origin_index": origin_index, "origin": str(meta["origin"]),
                    "slippage_per_side": slippage, **asdict(result),
                    "hold_sharpe_annualised": hold.sharpe_annualised, "hold_net_total_return": hold.net_total_return,
                    "jk_memmel_z": float("nan"), "jk_memmel_p": float("nan"), "dsr": float("nan"),
                    "dsr_n_trials": n_trials, "dsr_var_sharpe": float("nan"),
                    "evaluation_status": "conditional on future target availability; not an executable backtest",
                    "policy": "long/cash; each daily trade opens and closes; terminal and both-side costs included",
                    "risk_return_scale": "simple daily return; 365-period annualisation is a conditional diagnostic",
                    "inference_status": "JK/DSR and MDD confidence intervals withheld",
                })
    return pl.DataFrame(rows)
''')

# Short descriptions beside the implementation replace incorrect legacy claims.
for name, doc in {
    'summarise': 'Summarise conditional daily round trips. Sharpe/Sortino use simple returns; equity and MDD use log increments. MAR=0 uses all periods. Annualisation does not correct missing calendar outcomes.',
    'run_strategy': 'Conditional long/cash daily round-trip simulation for one saved forecast run.',
    'buy_and_hold': 'Always-long DAILY ROUND TRIPS on observed slots; retained API name, not uninterrupted buy-and-hold.',
}.items():
    c,node=cell_for('economics.py',name)
    lines=''.join(c['source']).splitlines(keepends=True); docnode=node.body[0]
    assert isinstance(docnode,ast.Expr) and isinstance(docnode.value,ast.Constant)
    lines[docnode.lineno-1:docnode.end_lineno]=['    """'+doc+'"""\n']
    c['source']=''.join(lines).splitlines(keepends=True)

for c in nb['cells']:
    if c['cell_type']=='code':
        ast.parse(''.join(c['source']))
        c['outputs']=[]; c['execution_count']=None
PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False)+'\n', encoding='utf-8')
print('Applied explicit conditional long/cash ledger and risk metric corrections')
