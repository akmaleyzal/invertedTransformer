"""Replace assertions that encoded audited defects, retaining regression coverage."""
import ast
from pathlib import Path

def change(file, name, source):
    path=Path('tests')/file
    text=path.read_text(encoding='utf-8')
    node=next(n for n in ast.parse(text).body if getattr(n,'name',None)==name)
    lines=text.splitlines(keepends=True)
    lines[node.lineno-1:node.end_lineno]=[source.strip()+'\n']
    path.write_text(''.join(lines),encoding='utf-8',newline='\n')

change('test_comparisons.py','test_nested_pairs_are_recognised','''
def test_feature_subsets_do_not_automatically_establish_model_nesting():
    for a, b in [(("itr", 1), ("itr", 8)), (("rdg", 1), ("rdg", 8)),
                 (("itr", 8), NAIVE), (NAIVE, ("dlin", 8))]:
        assert not is_nested(a, b)
''')
change('test_comparisons.py','test_nested_differential_carries_the_clark_west_adjustment','''
def test_differential_compares_mean_seed_losses_without_cw_adjustment():
    panel = build_panel([("itr", 1), ("itr", 8)], [ARTIFACTS], origin_indices=(1,))
    actual = differential(panel, ("itr", 1), ("itr", 8), 1)
    expected = (panel.seed_losses[("itr", 1), 1] - panel.seed_losses[("itr", 8), 1]).reshape(-1, 24).mean(axis=1)
    np.testing.assert_allclose(actual, expected)
''')
change('test_comparisons.py','test_nesting_order_puts_the_restricted_model_first','''
def test_nesting_order_never_invents_restrictions_for_the_studied_procedures():
    from itransformer_btc.comparisons import nesting_order
    for pair in [(("itr", 8), NAIVE), (NAIVE, ("itr", 8)),
                 (("itr", 8), ("itr", 1)), (("rdg", 1), ("rdg", 8))]:
        assert nesting_order(*pair) is None
''')
change('test_comparisons.py','test_differential_refuses_a_misoriented_nested_pair','''
def test_unadjusted_differential_changes_sign_when_the_pair_is_reversed():
    panel = build_panel([("itr", 8), NAIVE], [ARTIFACTS], origin_indices=(1,))
    a = differential(panel, ("itr", 8), NAIVE, 1)
    b = differential(panel, NAIVE, ("itr", 8), 1)
    np.testing.assert_array_equal(a, -b)
    assert len(a) == len(panel.y_true[1]) // 24
''')
path=Path('tests/test_comparisons.py');s=path.read_text(encoding='utf-8')
s=s.replace('assert n == 88_992','assert n == 88_560  # A01: remove forecasts issued after the six-block endpoint')
s=s.replace('assert named[("itr-K1", "itr-K8")] == "Clark-West"', 'assert named[("itr-K1", "itr-K8")] == "unadjusted forecast-loss diagnostic"')
s=s.replace('assert named[("Naive-RW", "itr-K1")] == "Clark-West"', 'assert named[("itr-K1", "Naive-RW")] == "unadjusted forecast-loss diagnostic"')
s=s.replace('assert named[("itr-K8", "ptst-K8")] == "DM-HLN"', 'assert named[("itr-K8", "ptst-K8")] == "unadjusted forecast-loss diagnostic"\n    assert all("exploratory" in x for x in table["inference_status"])')
s=s.replace('assert (table.get_column("se_across_origins").to_numpy() > 0).all()', 'assert (table.get_column("se_across_origins").to_numpy() >= 0).all()\n    assert table.filter(table["model"] == "Naive-RW")["mean_loss"].item() == pytest.approx(1.)')
s=s.replace('assert against_naive.get_column("left").to_list() == ["Naive-RW", "Naive-RW"]', 'assert against_naive.get_column("right").to_list() == ["Naive-RW", "Naive-RW"]')
path.write_text(s,encoding='utf-8',newline='\n')
path=Path('tests/test_experiment_plane.py');s=path.read_text(encoding='utf-8')
s=s.replace('# Mean skill is 0.05, so D(1) = (0.05 - 0.10)/0.05 = -1 and D(6) = +1.', '# A07: block-1 skill is 0.10; D(1)=0 and D(6)=1.')
s=s.replace('assert d[0] == pytest.approx(-1.0)', 'assert d[0] == pytest.approx(0.0)')
path.write_text(s,encoding='utf-8',newline='\n')

change('test_economics.py','test_realised_return_carries_the_drift_the_forecast_does_not','''
def test_forecast_and_realised_both_use_the_inverse_scaler():
    preds = _preds(5, seed=9)
    with_mean = positions(preds, META["sigma_g"], mu_g=.01)
    without = positions(preds, META["sigma_g"], mu_g=0.)
    for name in ("forecast_raw", "realised_raw"):
        np.testing.assert_allclose(with_mean[name].to_numpy() - without[name].to_numpy(), .24)
''')
change('test_economics.py','test_a_held_position_is_charged_once_not_every_period','''
def test_daily_round_trips_pay_entry_exit_and_terminal_costs():
    position = np.array([1., 1., 0., 1.])
    cost = TAKER_FEE_PER_SIDE + .0005
    np.testing.assert_allclose(net_returns(position, np.zeros(4), .0005),
                               position * math.log((1-cost)/(1+cost)))
    with pytest.raises(ValueError, match="long/cash"):
        net_returns(np.array([-1.]), np.array([0.]), .0005)
''')
path=Path('tests/test_economics.py');s=path.read_text(encoding='utf-8')
s=s.replace('assert np.array_equal(np.sign(forecast), position)', 'assert np.array_equal((forecast > 0).astype(float), position)')
s=s.replace('<= {-1.0, 0.0, 1.0}', '<= {0.0, 1.0}')
s=s.replace('assert hold.turnover_per_period == pytest.approx(0.5 / hold.n_periods)', 'assert hold.turnover_per_period == pytest.approx(1.)  # one complete round trip per observed day')
path.write_text(s,encoding='utf-8',newline='\n')
path=Path('tests/test_audit_repairs.py');s=path.read_text(encoding='utf-8')
s=s.replace('realised = np.array([-.01 + economics.TAKER_FEE_PER_SIDE, -.01, .02, .02])',
'''cost = economics.TAKER_FEE_PER_SIDE
    realised = np.log1p(np.array([-.01, -.01, .02, .02])) - math.log((1-cost)/(1+cost))''')
path.write_text(s,encoding='utf-8',newline='\n')

# Tests select stable notebook step metadata, not old printed section markers.
path=Path('tests/test_notebook_cells.py');s=path.read_text(encoding='utf-8')
start=s.index('# -- RQ3:')
end=s.index('#: The smallest byte string')
s=s[:start]+'''
def evaluation_step(slug):
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return next("".join(c["source"]) for c in nb["cells"]
                if c.get("metadata", {}).get("itbtc", {}).get("step") == slug)


@pytest.mark.parametrize("slug", ["rq1", "rq2", "rq3", "save"])
def test_estimator_cells_skip_cleanly_on_a_partial_grid(slug):
    out, _ = run_cell(evaluation_step(slug), GRID_COMPLETE=False)
    assert "skipped" in out.lower()


@pytest.mark.parametrize("skills,excluded", [({1: -.02, 8: -.018}, 15),
                                          ({1: .004, 8: .004}, 0)])
def test_rq3_cell_reports_reference_and_withholds_optimal_cadence(skills, excluded):
    out, ns = run_cell(evaluation_step("rq3"), seed_avg=panel(skills),
                      research_results={"rq3": {"reference": "block 1",
                       "logrank_status": "withheld", "optimal_cadence_estimated": False}})
    assert len(ns["dec"].excluded_origins) == excluded
    assert "block 1" in out
    assert "withheld" in out
    assert "does not estimate an optimal retraining cadence" in out
    assert "chi2=nan" not in out


'''+s[end:]
path.write_text(s,encoding='utf-8',newline='\n')
print('Updated tests to corrected audit contracts')
