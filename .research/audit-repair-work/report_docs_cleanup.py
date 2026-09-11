"""Remove stale methodological explanations after their implementation changed."""
from pathlib import Path
exec(Path('.research/audit-repair-work/applied_first_batch.py').read_text(encoding='utf-8').split('# A13:')[0])
for name, doc in {
 '_figure2': 'Draw the recorded iTransformer architecture for this prediction vintage.',
 '_figure7': 'Conditional long/cash daily round-trip wealth and always-long daily comparator. Average over a fixed set of origins on the retained-observation index; this is not a calendar-executable backtest.'
}.items():
    c, node = cell_for('report.py', name)
    lines = ''.join(c['source']).splitlines(keepends=True)
    old = node.body[0]
    assert isinstance(old, ast.Expr) and isinstance(old.value, ast.Constant)
    lines[old.lineno-1:old.end_lineno] = ['    """'+doc+'"""\n']
    c['source'] = lines
for c in nb['cells']:
    if c['cell_type'] == 'markdown':
        text = ''.join(c['source']).replace('jadi kelas kebocoran <code>center=True</code> tak terwakili (root §5.3).',
            'sehingga fitur di sini hanya memakai bar yang sudah selesai. Polars sendiri mendukung <code>center=True</code>; keamanan berasal dari pilihan fitur dan pemeriksaan kronologi.')
        c['source'] = text.splitlines(keepends=True)
    else:
        ast.parse(''.join(c['source']))
        c['outputs'] = []
        c['execution_count'] = None
PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False)+'\n', encoding='utf-8')
print('Cleaned explanatory text')
