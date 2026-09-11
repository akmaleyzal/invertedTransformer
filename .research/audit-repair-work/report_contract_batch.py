from pathlib import Path
exec(Path(__file__).with_name('applied_first_batch.py').read_text(encoding='utf-8').split('# A13:')[0])

def doc(module, name, text):
    c,node=cell_for(module,name)
    lines=''.join(c['source']).splitlines(keepends=True)
    d=node.body[0]
    assert isinstance(d,ast.Expr) and isinstance(d.value,ast.Constant)
    lines[d.lineno-1:d.end_lineno]=['    """'+text+'"""\n']
    c['source']=''.join(lines).splitlines(keepends=True)

def replace_note(name, source):
    c,node=cell_for('report.py',name)
    note=next(n for n in ast.walk(node) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='note' for t in n.targets))
    lines=''.join(c['source']).splitlines(keepends=True)
    lines[note.lineno-1:note.end_lineno]=['    note = '+source+'\n']
    c['source']=''.join(lines).splitlines(keepends=True)

doc('metrics.py','decay', '''Exploratory skill loss relative to the FIRST test block (A07).

    D(i,b) = (R2(i,1)-R2(i,b))/R2(i,1). The block-1 estimate is noisy and
    shared by later ratios. Non-positive reference skill is excluded and named.
    This post-audit definition cannot estimate an optimal retraining policy.''')
doc('metrics.py','directional_accuracy', '''Return direction after inverse scaling y = z*sigma_g + mu_g.

    H-step and cumulative primary diagnostics use non-overlapping forecast
    origins. Non-overlap alone does not establish temporal independence for PT.
    Defaults describe already-raw input; artifact callers must pass the scaler.''')
doc('metrics.py','falsification_relmse', '''Aged-minus-fresh block RelMSE on a verified common target calendar.

    Different training scalers require scale-free losses. The fresh arm changes
    BOTH training and validation windows and has one seed versus five aged.
    This combined intervention does not identify a pure causal ageing effect.''')
doc('metrics.py','paired_contrast', '''Exploratory paired contrast of mean seed loss, equal blocks and origins.

    Positive means left is worse. Calendar hashes must agree when supplied.
    The t(G-1) interval/p-value assumes independent origins, which the overlapping
    study does not establish. Feature content and PR both change in matched-K.''')
doc('metrics.py','minimum_detectable_beta1', '''Plug-in sensitivity of beta1 under independent-origin assumptions.

    The caller supplies slopes. In this study these are observed TEST slopes,
    so the computed value is post-analysis sensitivity, not prospective power.''')
doc('comparisons.py','PredictionPanel', '''Aligned forecast points with mean-seed losses as the estimand.

    y_pred stores the ensemble for inspection only. Production loss comparisons
    read seed_losses, then equally average block RelMSE and origin values.''')
doc('runner.py','stage5_pilot', '''Descriptive validation-only pilot at one origin.

    Square step errors for each seed, then average. No automatic CW test,
    no test prediction persistence and no prospective power claim. out_root
    remains accepted for call compatibility but the pilot writes no artifacts.''')

replace_note('_table4', repr('Exploratory reanalysis on common forecast times. Mean seed squared error is aggregated into block RelMSE with equal block and origin weights. Dispersion is across origins; it is not a dependence-corrected confidence interval. MCS membership is an independent-origin bootstrap diagnostic. These are local architecture adaptations with differing objectives and convergence histories, not controlled tests of architecture alone.'))
replace_note('_table5', repr('Exploratory first-block reference: D(i,b)=(R2(i,1)-R2(i,b))/R2(i,1). The decay estimand is undefined for non-positive block-1 skill; this is not right-censored evidence. Positive-reference origins without a crossing are right-censored at six blocks. No optimal retraining cadence is estimated. Log-rank inference is withheld because the arms are paired and origins dependent; the confirmatory H3 claim is untestable with this procedure. Confidence intervals are withheld. Excluded origins: ')+' + ", ".join(rq3["excluded_origins"])')
replace_note('_table6', '''(
        "Unadjusted mean-seed forecast-loss contrasts; no automatic Clark--West nesting. "
        "Positive t means left is worse. Block RelMSE has equal block and origin weights. "
        "All p-values and Romano--Wolf adjustments assume independent origins; training "
        "and test calendars overlap, so these are exploratory diagnostics only. "
        "Family adjustments are post-hoc. No confirmatory rejection is claimed."
    )''')
replace('report.py', '"CW" if row["statistic_name"].startswith("Clark") else "DM",', '"Loss",')
replace_note('_table8', repr('Exploratory conditional long/cash simulation from inverse-scaled raw cumulative forecasts. Each observed daily trade opens at 00:00 UTC and closes after 24 hours, paying entry and exit costs including the terminal trade. Execution cost c is the stated fee plus slippage, applied to purchase and sale prices. No borrowing or shorts. Missing future target periods are excluded retrospectively: this is not an executable backtest and unavailable-day counts do not bound missing losses. The comparator makes always-long daily round trips. Sharpe/Sortino use simple returns, MAR=0, and a conditional 365-period annualisation. JK/DSR and MDD confidence intervals are withheld.'))
replace_note('_table9', repr('Exploratory contrasts on the same forecast-time intersection and mean-seed/equal-block estimand. Matched K=8 changes feature identity as well as PR and cannot identify a causal dimensionality effect. Uniform attention also removes Q/K use and attention-weight dropout. L=48/96/192 are scored on actual shared targets. Missing early legacy forecasts remain absent. Reported t p-values assume independent origins and carry no confirmatory interpretation. Local iTransformer omits the official final encoder norm; local PatchTST reuses the iTransformer encoder. Training objectives and epoch-cap sensitivities remain limitations.'))
replace('report.py','"RQ3: retraining cadence at each pre-registered threshold."', '"RQ3: exploratory first-block skill threshold crossings."')
replace('report.py','"Pairwise forecast comparison with family-wise error control."', '"Exploratory forecast-loss diagnostics; independence assumptions unresolved."')
replace('report.py','"Economic evaluation across the pre-registered slippage band."', '"Conditional long/cash simulation across the stated cost assumptions."')
replace('economics.py','HOLD_LABEL: Final = "Buy & hold"', 'HOLD_LABEL: Final = "Always-long daily trades"')
replace('report.py','    windows = evaluation_windows(run_ids, roots)', '''    missing_runs = sorted({c.run_id for c in manifest()} - set(run_ids))
    if missing_runs:
        raise ValueError(f"report requires the complete declared manifest; missing {len(missing_runs)} runs, e.g. {missing_runs[:3]}")
    windows = evaluation_windows(run_ids, roots)''')
c,node=cell_for('report.py','build_report')
c['metadata']['itbtc'].setdefault('projection_imports',[]).append('from itransformer_btc.runner import manifest')
replace('report.py','    return ReportInputs(\n        numbers=numbers,', '''    numbers["comparisons"]["inference_status"] = analysis["inference_status"]
    numbers["directional_accuracy"]["note"] = "Raw returns after inverse scaling. PT p-values are diagnostics: non-overlap does not establish independence."
    numbers["economics"]["note"] = "Conditional long/cash daily round trips; future availability selection prevents executable-backtest claims."
    numbers["falsification"]["note"] = "Same actual targets, scale-free RelMSE; combined training/validation intervention, one fresh versus five aged seeds."
    numbers["attention_amplification"]["note"] = "Same features; uniform branch also removes Q/K use and attention-weight dropout. Effective capacity is not matched."
    numbers["coverage"]["note"] = "Conditional on surviving contiguous windows. Gap causes are unverified; coverage cannot recover missing outcomes. Equal-count training control has not run."
    numbers["main_results"]["note"] = analysis["estimand"] + "; overlapping origins make marginal SEs descriptive."
    return ReportInputs(
        numbers=numbers,''')

for c in nb['cells']:
    s=''.join(c['source'])
    s=s.replace('pre-registered','documented').replace('Pre-Registered','Documented').replace('preregistered','documented')
    s=s.replace('PR is the only thing that moves', 'feature content and PR both move')
    s=s.replace('participation ratio the only thing that moves', 'participation ratio and feature content both changing')
    s=s.replace('moves only the participation ratio', 'changes participation ratio and feature content together')
    s=s.replace('only what attention selects', 'attention selection, Q/K use and attention-weight dropout')
    c['source']=s.splitlines(keepends=True)

status = {
    'cell_type':'markdown', 'metadata':{'itbtc':{'role':'audit_status'}},
    'source':'''## Status setelah audit A01–A15 (9 September 2026)

Notebook ini adalah sumber utama proyek. Edit implementasi di sini; src/ hanya
dihasilkan melalui sel sinkronisasi terakhir. build_notebook.py adalah template
lama dan tidak boleh menimpa notebook ini.

Kode evaluasi telah dikoreksi, tetapi hasil grid lama bukan hasil training ulang.
Reanalisis memakai irisan waktu forecast yang benar-benar sama, mengukur rata-rata
loss per seed, dan bersifat **eksploratori**. Dependensi antar-origin belum
terselesaikan; tidak ada klaim kausal K_eff, daya uji prospektif, cadence retraining
optimal, atau backtest trading yang dapat dieksekusi. Kontrol model/objective,
jumlah training window, dan mekanisme memerlukan eksperimen baru.

Output eksekusi lama disimpan di outputs/iTransformer_before_A01_A15.ipynb.
Lihat ../docs/AUDIT_REMEDIATION_2026-09-09.md untuk bukti, batasan, dan syarat
penutupan setiap temuan. Simpan pekerjaan per batch; checkpoint pemulihan ada
di ../docs/AUDIT_REMEDIATION_CHECKPOINT.md.
'''.splitlines(keepends=True)}
nb['cells'].insert(1,status)
for c in nb['cells']:
    if c['cell_type']=='code':
        ast.parse(''.join(c['source']))
        c['outputs']=[];c['execution_count']=None
PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False)+'\n',encoding='utf-8')
print('Applied truthful report captions, scientific scope and notebook entry status')
