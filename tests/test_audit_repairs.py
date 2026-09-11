"""Counterexamples from A01-A15; run against the notebook's exported projection."""

import copy
import importlib.util
import json
import math
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
import torch

from itransformer_btc import economics, metrics, runner, train
from itransformer_btc.model import ITransformerConfig


@pytest.mark.parametrize("parallel", [False, True])
def test_a13_stale_run_reaches_executor(tmp_path, monkeypatch, parallel):
    cell = runner.RunCell("main", 1, 8, 24, 42)
    (tmp_path / "preds").mkdir()
    (tmp_path / "meta").mkdir()
    (tmp_path / "preds" / f"{cell.run_id}.parquet").write_bytes(b"PAR1")
    (tmp_path / "meta" / f"{cell.run_id}.json").write_text(json.dumps({
        "status": "complete", "code_sha256": "old", "input_sha256": "input",
    }))
    called = []
    def fit(self, tensors, spec, *, device=None):
        called.append(spec.run_id)
        return object(), self, SimpleNamespace(epochs_run=0, best_val_mse=0.)
    monkeypatch.setattr(ITransformerConfig, "fit", fit)
    monkeypatch.setattr(runner._TensorCache, "get", lambda *a: SimpleNamespace(train=[0]))
    monkeypatch.setattr(runner, "write_artifacts", lambda *a, **k: None)
    assert runner.pending([cell], [tmp_path]) == [cell]
    kwargs = {"devices": [torch.device("cpu"), torch.device("cpu")]} if parallel else {}
    execute = runner.execute_parallel if parallel else runner.execute
    result = execute([cell], pl.DataFrame(), roots=[tmp_path], out_root=tmp_path,
                     log=lambda *a: None, **kwargs)
    assert called == [cell.run_id]
    assert (result.completed, result.skipped, result.failed) == (1, 0, 0)


def test_a13_resume_requires_input_config_and_code(tmp_path, monkeypatch):
    cell = runner.RunCell("main", 1, 8, 24, 42)
    cfg = cell.model_config()
    (tmp_path / "preds").mkdir()
    (tmp_path / "meta").mkdir()
    pred_path = tmp_path / "preds" / f"{cell.run_id}.parquet"
    pl.DataFrame({name: [1] for name in ("block", "step", "timestamp", "forecast_origin",
                                      "input_start", "target_timestamp", "y_true", "y_pred")}).write_parquet(pred_path)
    monkeypatch.setattr(train, "_input_sha256", lambda *a: ("input", "file-digest"))
    (tmp_path / "weights").mkdir()
    weight_path = tmp_path / "weights" / f"{cell.run_id}.pt"
    torch.save(cfg.build().state_dict(), weight_path)
    metadata = {"run_id": cell.run_id, "status": "complete", "code_sha256": train.code_sha256(),
                "input_sha256": "input", "requested_config": asdict(cfg),
                "schedule": asdict(cfg.schedule()), "prediction_schema_version": 2,
                "weights_sha256": __import__("hashlib").sha256(weight_path.read_bytes()).hexdigest(),
                "predictions_sha256": __import__("hashlib").sha256(pred_path.read_bytes()).hexdigest()}
    path = tmp_path / "meta" / f"{cell.run_id}.json"
    path.write_text(json.dumps(metadata))
    assert train.is_complete(cell.run_id, tmp_path, strict=True, cfg=cfg)
    assert not train.is_complete(cell.run_id, tmp_path, strict=True, cfg=replace(cfg, d_ff=512))
    for field in ("input_sha256", "code_sha256", "requested_config"):
        bad = dict(metadata)
        bad.pop(field)
        path.write_text(json.dumps(bad))
        assert not train.is_complete(cell.run_id, tmp_path, strict=True, cfg=cfg)


def test_a13_dependency_change_invalidates_caller_output():
    path = Path(__file__).resolve().parents[1] / "tools/build_notebook.py"
    import sys
    spec = importlib.util.spec_from_file_location("audit_nb_generator", path)
    generator = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = generator
    spec.loader.exec_module(generator)
    previous = {"cells": [
        {"cell_type": "code", "source": ["def f():\n    return 1\n"], "outputs": [],
         "metadata": {"itbtc": {"role": "module", "module": "probe.py", "section": "f"}}},
        {"cell_type": "code", "source": ["print(f())\n"], "execution_count": 2,
         "outputs": [{"output_type": "stream", "name": "stdout", "text": ["1\n"]}],
         "metadata": {"itbtc": {"role": "step", "step": "probe"}}},
    ]}
    current = copy.deepcopy(previous)
    current["cells"][0]["source"] = ["def f():\n    return 2\n"]
    assert generator.carry_outputs(current, previous)[0] == 0
    assert current["cells"][1]["outputs"] == []


def test_a03_cr1_matches_full_dummy_regression():
    import statsmodels.api as sm
    from statsmodels.stats.sandwich_covariance import cov_cluster
    from scipy.stats import t
    rng = np.random.default_rng(43)
    groups = np.repeat(np.arange(30), 4)
    clusters = groups // 6
    a = np.tile([1., 4., 8., 12.], 30)
    b = a + rng.normal(size=len(a))
    y = .2 * b + rng.normal(size=len(a)) + np.repeat(rng.normal(size=30), 4)
    demean = lambda v: v - v.reshape(-1, 4).mean(axis=1).repeat(4)
    yd, ad, bd = map(demean, (y, a, b))
    fitted = bd * ((bd @ yd) / (bd @ bd))
    design = np.column_stack([ad, fitted, np.eye(30)[groups]])
    model = sm.OLS(y, design).fit()
    statistic = model.params[1] / math.sqrt(cov_cluster(model, clusters)[1, 1])
    actual = metrics.j_test(y, a, b, groups, clusters=clusters)
    np.testing.assert_allclose(actual, [statistic, 2 * t.sf(abs(statistic), 4)], rtol=1e-9)


def test_a10_direction_uses_inverse_scaling_before_summing():
    stamps = np.repeat(np.arange(3) * 24 * metrics.HOUR_MS, 24)
    frame = pl.DataFrame({"timestamp": stamps, "step": np.tile(np.arange(1, 25), 3),
                          "y_true": np.full(72, .2), "y_pred": np.full(72, -.1)})
    result = metrics.directional_accuracy(frame, sigma_g=1., mu_g=1.)
    assert result.da_h1 == result.da_hH == result.da_cum == 1.
    assert metrics.directional_accuracy(frame).da_cum == 0.


def test_a11_sortino_and_initial_drawdown():
    cost = economics.TAKER_FEE_PER_SIDE
    realised = np.log1p(np.array([-.01, -.01, .02, .02])) - math.log((1-cost)/(1+cost))
    result = economics.summarise(np.ones(4), realised, 0., 0, mdd_interval=False)
    expected = .005 / math.sqrt(.0002 / 4) * math.sqrt(365)
    assert result.sortino_annualised == pytest.approx(expected)
    assert economics.max_drawdown(np.array([-.1])) == pytest.approx(1 - math.exp(-.1))


def test_a07_improving_skill_does_not_trigger_decay():
    rows = [{"model": "itr", "k": 8, "pred_len": 24, "origin": "2020-01",
             "origin_index": 1, "block": b, "r2_oos": .01*b, "n_windows": 720}
            for b in range(1, 7)]
    result = metrics.decay(pl.DataFrame(rows))
    assert result.table["D"][0] == 0.
    assert not result.b_star(.05)["event"][0]


def test_a01_all_lookbacks_issue_at_the_same_time_and_never_read_a_target():
    from datetime import datetime, timezone, timedelta
    from itransformer_btc.splits import window_starts, _gather
    hours = np.arange(1000, dtype=np.int64)
    ts = hours * metrics.HOUR_MS
    start = datetime.fromtimestamp(300*3600, tz=timezone.utc)
    expected = ts[300:330]
    for length in (48, 96, 192):
        idx = window_starts(ts, start, start+timedelta(hours=30), "origin", length+24, seq_len=length)
        split = _gather(hours[:, None], idx, ts, length, 24)
        np.testing.assert_array_equal(split.ts, expected)
        np.testing.assert_array_equal(split.y[:, 0], hours[300:330])
        assert np.all(split.x[:, -1, 0] < split.y[:, 0])


def test_a01_legacy_labels_are_corrected_without_mutating_artifacts(tmp_path):
    from datetime import datetime, timezone
    run_id = "itr_o01_K08_H002_s42"
    origin = int(datetime(2020, 1, 1, tzinfo=timezone.utc).timestamp()*1000)
    (tmp_path/"preds").mkdir()
    (tmp_path/"meta").mkdir()
    data = pl.DataFrame({"block": [1, 1], "timestamp": [origin, origin],
                         "step": [1, 2], "y_true": [1., 2.], "y_pred": [1., 2.]})
    path = tmp_path/"preds"/f"{run_id}.parquet"
    data.write_parquet(path)
    (tmp_path/"meta"/f"{run_id}.json").write_text(json.dumps({
        "origin": "2020-01", "config": {"seq_len": 96}, "spec": {"pred_len": 2}
    }))
    before = path.read_bytes()
    result = metrics.load_predictions(run_id, [tmp_path])
    assert result["forecast_origin"].to_list() == [origin+96*metrics.HOUR_MS]*2
    assert result["target_timestamp"].to_list() == [origin+96*metrics.HOUR_MS, origin+97*metrics.HOUR_MS]
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match="window sets differ"):
        metrics.assert_same_windows(result, result.with_columns(pl.col("target_timestamp")+1), "test")


def test_a05_panel_measures_seed_loss_not_ensemble_loss(monkeypatch):
    from itransformer_btc import comparisons
    monkeypatch.setattr(comparisons, "_run_ids", lambda *a: ["s1", "s2"])
    monkeypatch.setattr(comparisons, "load_meta", lambda *a: {
        "code_sha256": "code", "input_sha256": "input", "naive_rw_z": 0.})
    def prediction(run_id, roots):
        return pl.DataFrame({"block": [1, 1], "timestamp": [0, 0],
                             "target_timestamp": [0, metrics.HOUR_MS], "step": [1, 2],
                             "y_true": [1., 1.], "y_pred": [-1., -1.] if run_id == "s1" else [3., 3.]})
    monkeypatch.setattr(comparisons, "load_predictions", prediction)
    panel = comparisons.build_panel([("itr", 8)], [], pred_len=2, origin_indices=(1,))
    assert np.mean((panel.y_true[1]-panel.y_pred[("itr", 8), 1])**2) == 0.
    assert np.mean(panel.seed_losses[("itr", 8), 1]) == 4.
    assert comparisons.per_origin_loss(panel, ("itr", 8)).item() == 4.


def test_a05_pilot_averages_seed_losses_and_caches_without_test_inference(monkeypatch, tmp_path):
    from itransformer_btc.splits import SplitTensors
    val = SplitTensors(np.zeros((3, 96, 1), dtype=np.float32), np.zeros((3, 24)),
                       np.zeros((3, 24, 1)), np.arange(3))
    tensors = SimpleNamespace(val=val, training_selection={"selected": 10})
    monkeypatch.setattr(runner._TensorCache, "get", lambda *a: tensors)
    calls = []
    def fitted(tensors, spec, cfg, **kwargs):
        calls.append(spec.run_id)
        return object(), SimpleNamespace(best_val_mse=1. if spec.seed == 42 else 3., wall_time_s=0., epochs_run=2)
    monkeypatch.setattr(runner, "train_one", fitted)
    monkeypatch.setattr(train, "predict", lambda *a: pytest.fail("no test inference in pilot"))
    monkeypatch.setattr(runner, "write_artifacts", lambda *a, **k: pytest.fail("pilot must not write test predictions"))
    result = runner.stage5_pilot(pl.DataFrame(), rungs=(1, 8), seeds=(42, 43),
                                 out_root=tmp_path, device=torch.device("cpu"), log=lambda *a: None)
    again = runner.stage5_pilot(pl.DataFrame(), rungs=(1, 8), seeds=(42, 43),
                               out_root=tmp_path, device=torch.device("cpu"), log=lambda *a: None)
    assert result.val_mse == again.val_mse == {1: 2., 8: 2.}
    assert len(calls) == 4
    assert not result.passed


def test_kaggle_mid_epoch_resume_matches_uninterrupted_training(tmp_path, monkeypatch):
    from itransformer_btc.config import ORIGINS
    from itransformer_btc.splits import OriginTensors, SplitTensors, Scaler
    rng = np.random.default_rng(12)
    x = rng.normal(size=(8, 4, 1)).astype(np.float32)
    y = rng.normal(size=(8, 2)).astype(np.float32)
    split = SplitTensors(x, y, y[:, :, None], np.arange(8, dtype=np.int64))
    tensors = OriginTensors(ORIGINS[0], 1, Scaler(np.zeros(1), np.ones(1), ("r",)),
                            split, split, (split,), (1,))
    cfg = ITransformerConfig(seq_len=4, pred_len=2, d_model=8, d_ff=16,
                             n_heads=2, e_layers=1, dropout=.2)
    spec = train.RunSpec("itr", 1, 1, 2, 42)
    settings = dict(device=torch.device("cpu"), max_epochs=3, patience=3, batch_size=4)
    monkeypatch.setattr(train, "_input_sha256", lambda *a: ("synthetic-input", "fixture"))
    expected, expected_outcome = train.train_one(tensors, spec, cfg, **settings)
    clock = [0.]
    monkeypatch.setattr(train, "time", SimpleNamespace(perf_counter=lambda: clock[0]))
    original_step = torch.optim.Adam.step
    calls = [0]
    def interrupt_step(self, *args, **kwargs):
        result = original_step(self, *args, **kwargs)
        calls[0] += 1
        if calls[0] == 3:  # first minibatch of epoch two
            clock[0] = 100.
        return result
    monkeypatch.setattr(torch.optim.Adam, "step", interrupt_step)
    first = tmp_path / "session1"
    with train.TrainingSession(first, [], 50.):
        with pytest.raises(train.SessionBudgetExhausted, match="epoch 1"):
            train.train_one(tensors, spec, cfg, **settings)
    checkpoint = first / "checkpoints" / f"{spec.run_id}.pt"
    payload = torch.load(checkpoint, weights_only=True)
    assert payload["epoch"] == 1
    monkeypatch.setattr(torch.optim.Adam, "step", original_step)
    clock[0] = 0.
    with train.TrainingSession(tmp_path / "session2", [first], 50.):
        actual, outcome = train.train_one(tensors, spec, cfg, **settings)
    for name, value in expected.state_dict().items():
        torch.testing.assert_close(actual.state_dict()[name], value, rtol=0, atol=0)
    assert outcome.best_val_mse == expected_outcome.best_val_mse
    # A mismatched identity is never loaded, even if its state is unusable.
    payload["identity"]["config"]["d_ff"] = 999
    payload["model"] = {"invalid": torch.ones(1)}
    torch.save(payload, checkpoint)
    with train.TrainingSession(tmp_path / "session3", [first], 50.):
        fresh, _ = train.train_one(tensors, spec, cfg, **settings)
    for name, value in expected.state_dict().items():
        torch.testing.assert_close(fresh.state_dict()[name], value, rtol=0, atol=0)


def test_a02_a12_training_only_invertible_controls_and_fixed_samples():
    from datetime import datetime, timezone
    from itransformer_btc.features import ladder_columns
    from itransformer_btc.splits import build_origin_tensors
    date = lambda hour: datetime.fromtimestamp(hour*3600, tz=timezone.utc)
    origin = SimpleNamespace(label="synthetic", train_start=date(0), train_sub_end=date(1000),
                             val_start=date(1000), val_end=date(1300),
                             blocks=lambda: [(1, date(1300), date(1400))])
    rng = np.random.default_rng(18)
    values = rng.normal(size=(1500, 8))
    values[:, 2:] += 2*values[:, 1:2]
    frame = pl.DataFrame({"ts_ms": np.arange(1500, dtype=np.int64)*metrics.HOUR_MS,
                          **{name: values[:, i] for i, name in enumerate(ladder_columns(8))}})
    base = build_origin_tensors(frame, origin, 8, seq_len=12, pred_len=4, train_window_limit=100)
    for mode in ("repr_identity", "whiten", "correlate"):
        part = build_origin_tensors(frame, origin, 8, seq_len=12, pred_len=4,
                                     train_window_limit=100, representation=mode)
        assert len(part.train) == 100
        np.testing.assert_array_equal(part.train.ts, base.train.ts)
        np.testing.assert_array_equal(part.train.y, base.train.y)
        np.testing.assert_allclose(part.train.x @ np.array(part.representation["inverse"]),
                                   base.train.x, atol=2e-5)
        future = frame.with_columns([pl.when(pl.col("ts_ms") >= 1000*metrics.HOUR_MS)
            .then(pl.col(col)*100).otherwise(pl.col(col)).alias(col) for col in ladder_columns(8)])
        changed = build_origin_tensors(future, origin, 8, seq_len=12, pred_len=4,
                                        train_window_limit=100, representation=mode)
        assert changed.representation == part.representation
    with pytest.raises(ValueError, match="required"):
        build_origin_tensors(frame, origin, 8, train_window_limit=2000)


def test_a08_refresh_protocol_preserves_train_and_pairs_five_seeds():
    from itransformer_btc.config import ORIGINS
    cells = runner.manifest(("fresh", "valrefresh"))
    assert len(cells) == 2*15*5
    for cell in cells:
        base, origin = ORIGINS[cell.origin_index-1], cell.origin()
        assert origin.train_sub_end <= origin.val_start < origin.val_end == origin.test_start
        assert origin.blocks() == [(b, *base.block(b)) for b in (4, 5, 6)]
        if cell.arm == "valrefresh":
            assert origin.train_start == base.train_start
            assert origin.train_sub_end == base.train_sub_end
            assert origin.val_start > base.val_start


def test_kaggle_two_workers_stop_on_deadline_and_surface_alignment(tmp_path, monkeypatch):
    """A pause stays pending; a scientific alignment failure reaches the caller."""
    cell = runner.RunCell("main", 1, 8, 24, 42)
    monkeypatch.setattr(runner._TensorCache, "get", lambda *a: SimpleNamespace(train=[0]))
    def pause(*a, **kw):
        raise train.SessionBudgetExhausted("deadline fixture")
    monkeypatch.setattr(ITransformerConfig, "fit", pause)
    result = runner.execute_parallel([cell], pl.DataFrame(), roots=[tmp_path], out_root=tmp_path,
        devices=[torch.device("cpu"), torch.device("cpu")], log=lambda *a: None)
    assert (result.completed, result.failed, result.remaining) == (0, 0, 1)
    from itransformer_btc.baselines import DLinearConfig
    monkeypatch.setattr(DLinearConfig, "fit", lambda *a, **kw:
                        (None, None, SimpleNamespace(epochs_run=1, best_val_mse=1.)))
    monkeypatch.setattr(runner, "write_artifacts", lambda *a, **kw: None)
    def mismatch(*a, **kw):
        raise AssertionError("actual target calendars differ")
    monkeypatch.setattr(runner, "_assert_alignment", mismatch)
    for arm in ("dlinear", "dlinear_all"):
        with pytest.raises(RuntimeError, match="fatal cross-model target alignment"):
            runner.execute_parallel([runner.RunCell(arm, 1, 8, 24, 42)], pl.DataFrame(),
                roots=[tmp_path], out_root=tmp_path,
                devices=[torch.device("cpu"), torch.device("cpu")], log=lambda *a: None)


def test_visible_devices_reports_every_cuda_device_the_session_has(monkeypatch):
    """`D68`'s entry point, checked where the 894-run session failed silently.

    That session used one of two T4s for 7.8 hours. Nothing raised, nothing was
    logged, and the grid finished — a no-op is indistinguishable from correct
    behaviour unless something asserts the count. The suite never could: every
    machine it runs on reports ``cuda.is_available() is False``, so the branch
    that matters on Kaggle is dead code here. Faking the CUDA report is the only
    way to exercise it before the next twelve-hour session pays for it.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    assert [str(d) for d in runner.visible_devices()] == ["cuda:0", "cuda:1"]

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    assert [str(d) for d in runner.visible_devices()] == ["cuda:0"]

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert [str(d) for d in runner.visible_devices()] == ["cpu"]


def test_parallel_grid_routes_work_to_both_devices(tmp_path, monkeypatch):
    """Two *distinct* devices, which the existing coverage never used.

    The executor test in ``test_experiment_plane`` passes ``[cpu, cpu]``, so it
    proves the queue discipline and nothing about routing: a worker that ignored
    its device argument entirely would pass it. Here the two differ, so the
    device each cell was fitted on is observable.

    A real run is ~30 s. The fake fit sleeps, because without a per-run cost the
    first thread drains the queue before the second is scheduled and the test
    would measure thread start-up latency instead of the split. The thread names
    are asserted separately, and that half is independent of scheduling.
    """
    import threading
    import time as _time

    seen: list[tuple[str, str]] = []
    lock = threading.Lock()

    def fit(self, tensors, spec, *, device=None):
        _time.sleep(0.004)
        with lock:
            seen.append((str(device), spec.run_id))
        return object(), self, SimpleNamespace(
            epochs_run=1, best_val_mse=0.0, train_loss=0.0,
            wall_time_s=0.0, n_parameters=0, device=str(device))

    written: list[str] = []
    monkeypatch.setattr(ITransformerConfig, "fit", fit)
    monkeypatch.setattr(runner._TensorCache, "get", lambda *a: SimpleNamespace(train=[0]))
    monkeypatch.setattr(runner, "is_complete", lambda *a, **k: False)
    monkeypatch.setattr(runner, "completed_run_ids", lambda roots, code_digest="": set())
    monkeypatch.setattr(runner, "write_artifacts",
                        lambda model, tensors, spec, *a, **k: written.append(spec.run_id))

    spawned: list[str] = []
    real_thread = threading.Thread

    class Recording(real_thread):
        def __init__(self, *a, **kw):
            if str(kw.get("name", "")).startswith("grid-"):
                spawned.append(kw["name"])
            super().__init__(*a, **kw)

    monkeypatch.setattr(runner.threading, "Thread", Recording)

    cells = runner.manifest(("main",))[:40]
    summary = runner.execute_parallel(
        cells, pl.DataFrame(),
        devices=[torch.device("cuda", 0), torch.device("cuda", 1)],
        out_root=tmp_path, roots=[tmp_path], log=lambda *a: None,
    )

    assert sorted(spawned) == ["grid-cuda:0", "grid-cuda:1"], "one worker per device"
    assert {d for d, _ in seen} == {"cuda:0", "cuda:1"}, "a device sat idle"
    assert summary.completed == len(cells)
    assert sorted(r for _, r in seen) == sorted(c.run_id for c in cells)
    assert sorted(written) == sorted(c.run_id for c in cells), "a cell was dropped or doubled"


def test_the_grid_cell_hands_the_notebook_both_devices():
    """The wiring `D68` depends on, asserted on the committed notebook bytes.

    ``execute_parallel`` defaults to ``visible_devices()``, so a grid cell that
    passed ``devices=[pick_device()]`` — or nothing at all through a helper that
    picks one — would still run, still finish, and still use half the hardware.
    `D55` is the precedent: testing the generator's intent rather than the bytes
    is how a stale cell reached Kaggle.
    """
    notebook = json.loads(
        (Path(__file__).resolve().parents[1] / "notebooks/iTransformer.ipynb")
        .read_text(encoding="utf-8"))
    callers = [src for cell in notebook["cells"] if cell["cell_type"] == "code"
               for src in ["".join(cell["source"])]
               if "execute_parallel(" in src and "def execute_parallel" not in src]

    assert len(callers) == 1, f"expected one grid cell, found {len(callers)}"
    body = callers[0]
    assert "visible_devices()" in body, "the grid cell picks devices some other way"
    assert "devices=DEVICES" in body, "visible_devices() is computed and not passed"
    assert "guard=SESSION_GUARD" in body, "both workers must share one deadline"


def test_kaggle_consolidation_preserves_existing_checkpoint_and_copies_valid_bundle(tmp_path, monkeypatch):
    cell = runner.RunCell("main", 1, 8, 24, 42)
    prior, out = tmp_path / "previous", tmp_path / "current"
    for folder, name, payload in (("preds", cell.run_id+".parquet", b"predictions"),
        ("weights", cell.run_id+".pt", b"weights"), ("meta", cell.run_id+".json", b"metadata"),
        ("checkpoints", "pending.pt", b"old epoch"), ("validation", "probe.json", b"cached fit")):
        (prior / folder).mkdir(exist_ok=True, parents=True)
        (prior / folder / name).write_bytes(payload)
    (out / "checkpoints").mkdir(parents=True)
    (out / "checkpoints/pending.pt").write_bytes(b"new epoch")
    calls = []
    def accepted(run_id, root, **kwargs):
        calls.append(kwargs)
        return root == prior
    monkeypatch.setattr(runner, "is_complete", accepted)
    assert runner.consolidate_resume_outputs([cell], [out, prior], out_root=out) == 1
    assert all(call["strict"] and "cfg" in call and "columns" in call for call in calls)
    assert (out / "preds" / (cell.run_id+".parquet")).read_bytes() == b"predictions"
    assert (out / "meta" / (cell.run_id+".json")).read_bytes() == b"metadata"
    assert (out / "checkpoints/pending.pt").read_bytes() == b"new epoch"
    assert (out / "validation/probe.json").read_bytes() == b"cached fit"
