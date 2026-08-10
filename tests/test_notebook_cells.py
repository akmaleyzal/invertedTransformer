"""Execute the notebook's evaluation cells — the surface that had no coverage.

`test_notebook_sync` proves the notebook's package cells match ``src/``. Nothing
proved anything about the *evaluation* cells, which are the only code in this
repository that exists solely inside the notebook, and that gap is how `D55`
reached Kaggle: ``b_star`` returned a schema-less frame, the RQ3 cell indexed a
column that was not there, and a twelve-hour session was marked failed at the
moment its grid output was the only thing worth keeping.

So these tests run the **bytes the notebook actually contains**, not the
generator constants they were built from — the notebook is what Kaggle executes,
and a test of the constants would pass while a stale notebook still crashed.

Scope, stated rather than implied. Covered here: the RQ3 cell across the three
panel states it must tell apart, and the ``GRID_COMPLETE`` guard `D54e` added.
**Not covered: the RQ1, RQ2 and paper_numbers cells**, whose scopes need
bootstrap results and K_eff tables that a synthetic fixture would have to invent
— and an invented fixture that passes tells you nothing. They remain the same
kind of gap this file was written to close, one size smaller.
"""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from itransformer_btc import metrics

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "notebooks" / "iTransformer.ipynb"

ORIGINS = ("2020-01", "2020-06", "2020-11", "2021-04", "2021-09", "2022-02",
           "2022-07", "2022-12", "2023-05", "2023-10", "2024-03", "2024-08",
           "2025-01", "2025-06", "2025-11")


@pytest.fixture(scope="module")
def cells() -> list[str]:
    assert NOTEBOOK.exists(), f"{NOTEBOOK} is missing; run tools/build_notebook.py"
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def cell_containing(cells: list[str], marker: str) -> str:
    """The one code cell holding ``marker``.

    Exactly one, deliberately: two matches would mean the marker no longer
    identifies the cell and the test would silently drift onto the wrong one.
    """
    hits = [c for c in cells if marker in c]
    assert len(hits) == 1, f"{len(hits)} cells contain {marker!r}, expected 1"
    return hits[0]


def panel(r2_by_k: dict[int, float]) -> pl.DataFrame:
    """A seed-averaged panel: 15 origins x 6 blocks, one skill level per rung."""
    return pl.DataFrame([
        {
            "model": "itr", "origin_index": i, "origin": origin, "k": k,
            "pred_len": 24, "block": block, "mse": 1.0 - r2, "mse_naive": 1.0,
            "n_windows": 720, "r2_oos": r2,
        }
        for i, origin in enumerate(ORIGINS, start=1)
        for k, r2 in r2_by_k.items()
        for block in range(1, 7)
    ])


def run_cell(source: str, **scope: object) -> tuple[str, dict]:
    """Execute a cell body and return everything it printed plus its namespace.

    The scope is **flat**, mirroring the kernel the notebook actually builds: the
    module cells define ``decay`` and ``kaplan_meier`` as bare names in one shared
    namespace, so the evaluation cells call them unqualified. Injecting
    ``metrics`` as a module would test a notebook that no longer exists.
    """
    ns: dict = {
        **{k: v for k, v in vars(metrics).items() if not k.startswith("__")},
        "pl": pl,
        "np": np,
        "GRID_COMPLETE": True,
        **scope,
    }
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        exec(compile(source, "<notebook-cell>", "exec"), ns)
    return buffer.getvalue(), ns


# -- RQ3: the three panel states it has to tell apart ------------------------


def test_rq3_reports_undefined_when_every_origin_lacks_an_edge(cells) -> None:
    """`D55`, and the shape the completed grid actually returned.

    R2_oos was negative at all four rungs and all fifteen origins, so ``decay``
    excluded every origin. The cell must print the pre-registered null instead of
    raising, and must **not** call it censoring: a censored origin has an edge
    that never decays past tau, whereas here there is no edge to take a
    proportion of. One wording for both would claim skill the grid never found.
    """
    out, ns = run_cell(cell_containing(cells, "--- RQ3"),
                       seed_avg=panel({1: -0.0205, 8: -0.0180}))

    # Assert the claim, not the spelling: the cell *quotes* the censored wording
    # in order to forbid it, so a bare substring check would fail on the very
    # sentence that gets this right.
    assert "UNDEFINED" in out
    assert "RQ3 RETURNS NO ANSWER" in out
    assert "the decay estimand is undefined" in out
    assert "censored >6" not in out          # the per-tau result lines
    assert "crossings" not in out            # the estimated branch's format
    assert {r["status"] for r in ns["b_star_rows"]} == {"undefined"}
    assert len(ns["b_star_rows"]) == len(metrics.TAU_SENSITIVITY)

    # Every tau, including the headline, and the excluded origins named.
    for tau in metrics.TAU_SENSITIVITY:
        assert f"tau={tau:>6.1%}" in out
    assert "2020-01" in out and "2025-11" in out


def test_rq3_reports_censoring_when_an_edge_exists_but_never_decays(cells) -> None:
    """The state the estimator was designed for, and the wording root §3 fixes.

    Positive skill, flat across blocks: b* is right-censored at 6 in every
    origin. Here — and only here — "no decay detected within 180 days" is the
    honest sentence.
    """
    out, ns = run_cell(cell_containing(cells, "--- RQ3"),
                       seed_avg=panel({1: 0.004, 8: 0.004}))

    assert "censored >6" in out
    assert "no decay detected within 180 days" in out
    assert "UNDEFINED" not in out
    assert {r["status"] for r in ns["b_star_rows"]} == {"estimated"}
    assert all(r["n_origins"] == len(ORIGINS) for r in ns["b_star_rows"])


def test_rq3_never_prints_a_nan_as_though_it_were_a_statistic(cells) -> None:
    """With zero crossings in both arms the log-rank statistic is 0/0.

    Printing ``chi2=nan p=nan`` there is the same defect as `D55`: a number that
    reads like a computed result but is not one, which root §12 calls a
    documented failure rather than a footnote.

    The check is on nan *formatted as a statistic*, not on the three letters:
    the cell's own explanation says the word, which is the sentence doing this
    correctly.
    """
    for skills in ({1: -0.0205, 8: -0.0180}, {1: 0.004, 8: 0.004}):
        out, _ = run_cell(cell_containing(cells, "--- RQ3"), seed_avg=panel(skills))
        assert "chi2=nan" not in out
        assert "p=nan" not in out
        assert "log-rank K=8 vs K=1 UNAVAILABLE" in out


def test_rq3_computes_the_log_rank_when_both_arms_have_crossings(cells) -> None:
    """The positive path still runs — the guards must not have disabled H3.

    Skill decays within each origin, so D(i,b) crosses tau and b* is an event
    rather than a censoring in both arms.
    """
    rows = [
        {
            "model": "itr", "origin_index": i, "origin": origin, "k": k,
            "pred_len": 24, "block": block, "mse": 1.0 - r2, "mse_naive": 1.0,
            "n_windows": 720, "r2_oos": r2,
        }
        for i, origin in enumerate(ORIGINS, start=1)
        for k in (1, 8)
        # Decays 0.010 -> 0.001 across the six blocks, so late blocks sit far
        # below the within-origin mean and D crosses every pre-registered tau.
        for block, r2 in enumerate([0.010, 0.008, 0.006, 0.004, 0.002, 0.001], start=1)
    ]
    out, ns = run_cell(cell_containing(cells, "--- RQ3"), seed_avg=pl.DataFrame(rows))

    assert "log-rank K=8 vs K=1 at tau=5%" in out
    assert "UNAVAILABLE" not in out
    assert all(r["status"] == "estimated" for r in ns["b_star_rows"])
    assert any(r["events"] > 0 for r in ns["b_star_rows"])


# -- the D54e guard ----------------------------------------------------------


@pytest.mark.parametrize("marker", ["--- RQ1", "--- RQ2", "--- RQ3"])
def test_estimator_cells_skip_cleanly_on_a_partial_grid(cells, marker) -> None:
    """`D54e`. A partial session is the expected case, and it must exit clean.

    The estimators stay strict — a half-panel beta1 is a different estimand, not
    a noisier one — so the cell must not call them at all. Passing no panel at
    all is the strongest form of the check: if the guard leaked, the body would
    raise ``NameError`` on ``seed_avg`` rather than print.
    """
    out, _ = run_cell(cell_containing(cells, marker), GRID_COMPLETE=False)
    assert "SKIPPED" in out
    assert "Resume in the next session" in out
