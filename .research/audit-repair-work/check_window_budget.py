"""Verify every declared training span without constructing or fitting a model."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / 'src'))
from itransformer_btc.runner import manifest
from itransformer_btc.features import build_features
from itransformer_btc.segments import load_bars, usable_mask
from itransformer_btc.splits import window_starts
from itransformer_btc.budget import budget_table, format_markdown

bars = usable_mask(load_bars(root / 'data/raw/BTCUSDT_1h.parquet'))
ts = build_features(bars)['ts_ms'].to_numpy()
rows = {}
for cell in manifest():
    origin = cell.origin()
    key = (origin.train_start.isoformat(), origin.train_sub_end.isoformat(), cell.seq_len, cell.pred_len)
    if key in rows:
        continue
    indices = window_starts(ts, origin.train_start, origin.train_sub_end, 'contained', cell.seq_len+cell.pred_len)
    assert len(indices) >= 11500, (cell.run_id, len(indices))
    selected = indices[np.sort(np.random.default_rng(1729).choice(len(indices), 11500, replace=False))]
    rows[key] = {'example_run_id': cell.run_id, 'train_start': key[0], 'train_end': key[1],
        'seq_len': cell.seq_len, 'pred_len': cell.pred_len, 'available': len(indices), 'selected': len(selected),
        'forecast_times_sha256': hashlib.sha256(ts[selected+cell.seq_len].tobytes()).hexdigest()}
evidence = {'manifest_runs': len(manifest()), 'unique_training_spans': len(rows),
    'minimum_available': min(r['available'] for r in rows.values()),
    'maximum_available': max(r['available'] for r in rows.values()), 'rows': list(rows.values())}
(root / '.research/audit-repair-work/window_budget.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
(root / '.research/audit-repair-work/raw_window_budget.md').write_text(format_markdown(budget_table(bars)), encoding='utf-8')
print({k:v for k,v in evidence.items() if k != 'rows'})
