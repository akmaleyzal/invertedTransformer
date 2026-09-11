from pathlib import Path
exec(Path(__file__).with_name('applied_first_batch.py').read_text(encoding='utf-8').split('# A13:')[0])

def documentation(module, name, text):
    if name:
        c,n=cell_for(module,name)
        node=n.body[0]
    else:
        c=next(c for c in nb['cells'] if c.get('metadata',{}).get('itbtc',{}).get('module')==module)
        node=ast.parse(''.join(c['source'])).body[0]
    assert isinstance(node,ast.Expr) and isinstance(node.value,ast.Constant)
    lines=''.join(c['source']).splitlines(keepends=True)
    indent=' '*node.col_offset
    lines[node.lineno-1:node.end_lineno]=[indent+'"""'+text.strip()+'"""\n']
    c['source']=''.join(lines).splitlines(keepends=True)

documentation('__init__.py',None,'''Generated projection of notebooks/iTransformer.ipynb.

Edit the notebook and export only through its final cell. CLAUDE.md governs the
study. Polars is the data-plane implementation choice; it supports centered
rolling windows. Safety comes from per-bar features, chronology tests and purged
splits, not from an alleged limitation of the library.''')

# Preserve upstream citations while replacing the obsolete methodological header.
c=next(c for c in nb['cells'] if c.get('metadata',{}).get('itbtc',{}).get('module')=='baselines.py')
old=ast.parse(''.join(c['source'])).body[0].value.value
upstream=old[old.index('Upstream\n'):]
documentation('baselines.py',None,'''Baseline configurations compared with iTransformer on common target calendars.

DLinear and PatchTST default to target-only loss, including checkpoint selection.
Their target-only path consumes target history alone (effective_input_channels=1)
despite the supplied K=8 tensor. Separate all-channel arms expose auxiliary-task
supervision. Ridge and LSTM can use the supplied multivariate history directly.

The post-audit PatchTST port uses its own BatchNorm/residual-attention encoder,
shared patch projection and head, affine-free RevIN, no patch padding, and no
head dropout. It is a declared configuration, not an exact copy of all upstream
defaults. Baseline LR sensitivity uses only origin-1 validation. A 120-epoch cap
with patience 12 is recorded, and reaching or avoiding it is not proof of global
optimization convergence. All output metrics concern the return channel.

ARIMA remains outside this study's implemented scope; ADF does not establish
that AIC would select an ARIMA(0,0,0). LSTM and naive comparators ARE implemented.

'''+upstream)
documentation('baselines.py','BaselineModule', 'Shared parameter counting and explicit target-only input selection for channel-independent baselines.')
documentation('baselines.py','DLinearConfig','Shared trend/seasonal linear maps; target objective by default, all-channel sensitivity explicit. The validation-selected learning rate and resolved schedule are saved.')
documentation('baselines.py','PatchTSTConfig','Shared patched encoder with BatchNorm and residual attention. The supplied tensor has K channels; target-only training uses just the target. RevIN affine is disabled, padding is absent, head dropout is zero.')
documentation('splits.py','SplitTensors','Inputs, target returns, all-channel targets and first-target timestamps. The default objective consumes y; all-channel sensitivity consumes y_all. Every output artifact predicts the return channel.')
documentation('runner.py','BudgetGuard','One monotonic deadline for a session. Starting a run requires a buffer based on recent maximum duration; train_one also checks the deadline within every epoch. Reserved time is for saving and stopping the Kaggle session.')
documentation('train.py','write_artifacts','Atomically publish predictions and best weights, then completion metadata with byte hashes. Remove this output root\'s partial checkpoint only after metadata is published. Prior attached outputs are read-only.')
documentation('report.py','_architecture_section','Report actual allocated/active parameters, objectives, effective inputs, schedules and cap counts from each run\'s metadata. These are configuration diagnostics, not a proof of architecture parity or optimization convergence.')

# Official PatchTST applies projection dropout AND residual dropout after attention.
c,n=cell_for('baselines.py','PatchEncoderLayer');s=''.join(c['source'])
s=s.replace('        self.dropout = nn.Dropout(cfg.dropout)','        self.projection_dropout = nn.Dropout(cfg.dropout)\n        self.dropout = nn.Dropout(cfg.dropout)',1)
s=s.replace('self.dropout(self.out(context))','self.dropout(self.projection_dropout(self.out(context)))')
s=s.replace('''        # EncoderLayer reads d_model, d_ff, n_heads, dropout and
        # uniform_attention; its remaining fields are inert here and stay at
        # their defaults. Reusing the block rather than reimplementing it is what
        # makes "same capacity, different tokenisation" a fact and not a claim.
''','')
# RevIN's window statistics are detached, as in the upstream affine=False mode.
s=s.replace('std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + 1e-5)','std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + 1e-5).detach()')
c['source']=s.splitlines(keepends=True)

replace('train.py','        "training_cutoff_ms": int(tensors.origin.train_sub_end.timestamp() * 1000),','        "training_cutoff_ms": int(tensors.origin.train_sub_end.timestamp() * 1000),\n        "latest_training_target_ms": int(tensors.train.ts.max() + (cfg.pred_len-1)*3600000),')

# Update the adapted prose in structured provenance, leaving unrelated citations intact.
c,n=cell_for('config.py','Upstream');s=''.join(c['source']);tree=ast.parse(s)
for node in reversed(list(ast.walk(tree))):
    if not isinstance(node,ast.Call) or not isinstance(node.func,ast.Name) or node.func.id!='Upstream':continue
    kws={kw.arg:kw for kw in node.keywords}
    component=ast.literal_eval(kws['component'].value)
    if component.startswith('ITransformer'):
        adapted='Post-audit final encoder LayerNorm restored; target-only loss; study-sized encoder and declared schedules. Uniform arm applies the same attention dropout but freezes unused Q/K. Three invertible-representation controls disable instance norm; all other iTransformer arms retain it. Hyperparameter sensitivity is explicit and validation-only.'
    elif component.startswith('PatchTST'):
        adapted='Post-audit BatchNorm/residual-attention encoder with both projection and residual dropout, correct channel/patch flatten order and uniform positional initialization. Shared head; RevIN affine=False, no patch padding, head dropout=0. Target-only default and separate all-channel sensitivity; validation LR search and 120-epoch cap recorded.'
    elif component.startswith('DLinear'):
        adapted='Shared channel maps, target-only default and separate all-channel sensitivity. Target-only effective input K=1 is disclosed. Validation LR search and 120-epoch cap recorded; no claim of equivalent optimization across model families.'
    else:continue
    v=kws['adapted'].value;lines=s.splitlines(True)
    # Keyword values span multiple lines; replace only their source positions.
    start=sum(len(x) for x in lines[:v.lineno-1])+v.col_offset
    end=sum(len(x) for x in lines[:v.end_lineno-1])+v.end_col_offset
    s=s[:start]+repr(adapted)+s[end:]
c['source']=s.splitlines(keepends=True)

for c in nb['cells']:
    s=''.join(c['source'])
    if c['cell_type']=='markdown':
        s=s.replace('dan cadence retraining (RQ3), dengan protokol walk-forward yang dipra-registrasi.',
                    'dan crossing ambang skill (RQ3), dengan protokol walk-forward terdokumentasi dan revisi eksploratori setelah audit.')
        s=s.replace('pilih GPU, lalu gunakan', 'pilih GPU T4 x2, isi sisa kuota mingguan pada sel setup, lalu gunakan')
        s=s.replace('Impor baru ditambahkan di src/, kemudian notebook dibangun ulang. Perubahan source dan notebook disimpan bersama.',
                    'Impor baru ditulis pada sel Library dan metadata itbtc.projection_imports untuk modul tujuan. Ekspor melalui sel terakhir; jangan membangun ulang notebook dari template lama.')
        s=s.replace('RQ3 — cadence retraining optimal?', 'RQ3 — crossing ambang skill (deskriptif)')
        s=s.replace('dan yang channel-independent membawa objektif all-channel terbitannya sendiri',
                    'objektif default adalah target-only; sensitivitas all-channel memiliki arm terpisah')
        s=s.replace('Kontrol model/objective,\njumlah training window, dan mekanisme memerlukan eksperimen baru.',
                    'Notebook ini memuat kontrol model/objective, jumlah training window, dan representasi invertibel untuk rerun baru di Kaggle; hasilnya belum tersedia.')
    if c.get('metadata',{}).get('itbtc',{}).get('step')=='sync_back':
        start=s.index('#   3. Impor baru');end=s.index('#\n# Sesudahnya',start)
        s=s[:start]+'''#   3. Impor baru: tambahkan pada Library untuk namespace notebook, serta
#      metadata itbtc.projection_imports pada sel modul untuk proyeksi paket.
#      Hapus impor lama melalui projection_remove_imports bila diperlukan.
#      Jangan menyunting src/ atau membangun ulang notebook dari template lama.
'''+s[end:]
    s=s.replace('with no rolling\nwindow anywhere, the ``center=True`` leak class is unrepresentable (root §5.3).',
                'the implemented per-bar features use no centered windows; chronology tests enforce that choice. Polars itself supports center=True.')
    s=s.replace('what makes the ``center=True`` leakage class structurally unrepresentable and\nlicenses root §8.3\'s no-embargo argument (`D15`).',
                'a constraint enforced by these feature definitions and chronology tests. Polars itself supports centered rolling windows.')
    c['source']=s.splitlines(keepends=True)
    if c['cell_type']=='code':
        ast.parse(s);c['outputs']=[];c['execution_count']=None
PATH.write_text(json.dumps(nb,indent=1,ensure_ascii=False)+'\n',encoding='utf-8')
print('Updated model parity details, provenance and notebook authority narrative')
