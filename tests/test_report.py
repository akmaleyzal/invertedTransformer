"""The report generator, tested against the real artifacts.

`D60g` is the defect this file exists to keep closed: the grid produced every
number the manuscript needs and left every table and figure ungenerated, and
nothing in the repository noticed for nine days. A test that only exercised
synthetic frames would have passed throughout. So these run against the committed
684-run output, and skip --- rather than pass --- where it is absent.

Two of the assertions here are about *wording*, not arithmetic, and they are not
decoration. `D60b` fixes RQ3's phrasing because the two available sentences make
different claims and only one is true; `D60i` fixes the falsification arm's
metric because the shipped figure was 99.7% scaler drift with its sign reversed.
Both are the kind of defect that survives review by looking like prose.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

from itransformer_btc.features import build_features
from itransformer_btc.report import (
    ROBUSTNESS_TAGS,
    build_report,
    fmt,
    render_figures,
    render_tables,
    tabular,
)
from itransformer_btc.segments import load_bars, usable_mask

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "notebooks" / "outputs" / "artifacts"
PARQUET = ROOT / "data" / "raw" / "BTCUSDT_1h.parquet"
GENERATOR = ROOT / "tools" / "build_report.py"

#: ``nan`` as a *token*, so "Binance" does not trip it. A NaN printed into a
#: table reads as a measurement and is the absence of one.
NON_VALUE = re.compile(r"(?<![A-Za-z])(nan|inf|None)(?![A-Za-z])", re.IGNORECASE)

#: Bootstrap draws for the tests. Far below the 9,999 the paper uses: the floor
#: on a p-value is 1/(1+B), and these assertions are about structure rather than
#: about a p-value's fourth decimal.
TEST_B = 99

requires_grid = pytest.mark.skipif(
    not (ARTIFACTS / "paper_numbers.json").exists(),
    reason="the 684-run grid output is not in this checkout",
)


# -- the pieces that need no artifacts ---------------------------------------


def test_fmt_never_emits_a_non_value():
    """An em-dash makes a gap visible; ``nan`` disguises one as a measurement."""
    assert fmt(float("nan")) == "---"
    assert fmt(float("inf")) == "---"
    assert fmt(None) == "---"
    assert fmt(0.123456, 3) == "0.123"
    assert fmt(75094, 0) == "75,094"


def test_tabular_is_balanced_and_names_its_generator():
    out = tabular("Caption", "tab:x", ["A", "B"], [["1", "2"]], "rr", note="note")
    assert out.count(r"\begin{tabular}") == out.count(r"\end{tabular}") == 1
    assert out.count(r"\begin{table}") == out.count(r"\end{table}") == 1
    assert "tools/build_report.py" in out, "a generated file must name its generator"


def test_generated_latex_never_names_the_package():
    """`D59`: a string literal mentioning ``itransformer_btc`` survives flattening
    into the notebook, and the generator cannot tell a comment about the package
    from a reference to it --- so it refuses the cell. The provenance line
    therefore names the command instead, which is more useful anyway.
    """
    out = tabular("Caption", "tab:x", ["A"], [["1"]], "r")
    assert "itransformer_btc" not in out


# -- the report itself -------------------------------------------------------


@pytest.fixture(scope="module")
def inputs():
    if not (ARTIFACTS / "paper_numbers.json").exists() or not PARQUET.exists():
        pytest.skip("the 684-run grid output is not in this checkout")
    bars = usable_mask(load_bars(PARQUET))
    features = build_features(bars)
    return build_report(
        ARTIFACTS, bars, features, bootstrap_b=TEST_B, log=lambda *_: None
    )


@requires_grid
def test_every_section_the_manuscript_reads_is_present(inputs):
    numbers = inputs.numbers
    for section in (
        "derived_from", "dataset", "architecture", "keff", "keff_rolling",
        "rq1", "rq2", "rq3", "efficiency", "comparisons", "main_results",
        "economics", "directional_accuracy", "horizons", "falsification",
        "attention_amplification", "raw_scale", "coverage", "robustness",
    ):
        assert section in numbers, f"{section} missing from the manuscript's source"


@requires_grid
def test_provenance_names_the_grid_file_by_digest(inputs):
    """Root §12: two artifacts that must agree, with something that checks."""
    provenance = inputs.numbers["derived_from"]
    assert len(provenance["grid_paper_numbers_sha256"]) == 64
    assert inputs.numbers["input_sha256"].startswith("8270a84b07c2923b")
    assert inputs.numbers["runs_complete"] >= 684


def test_a_short_grid_is_refused(tmp_path):
    """`D54e`: a partial panel is a different estimand, not a noisier one."""
    (tmp_path / "paper_numbers.json").write_text("{}", encoding="utf-8")
    (tmp_path / "preds").mkdir()
    (tmp_path / "meta").mkdir()
    with pytest.raises(ValueError, match="below the 684-run grid"):
        build_report(tmp_path, None, None, log=lambda *_: None)


@requires_grid
def test_falsification_is_reported_on_relmse(inputs):
    """`D60i`. The mean must reproduce the corrected figure, and the metric name
    must say which scale it is on --- the defect was a number on the wrong scale,
    not a number computed wrongly.
    """
    falsification = inputs.numbers["falsification"]
    assert falsification["metric"] == "RelMSE"
    assert abs(falsification["mean_gap_rel_mse"] - 0.000828) < 5e-6
    assert falsification["n_origins"] == 15


@requires_grid
def test_coverage_check_reports_its_own_unavailability(inputs):
    """`D45`: restricting to well-covered blocks unbalances the panel, and that
    the check cannot run IS the finding. A number here would mean the estimator
    had been loosened until it produced one.
    """
    coverage = inputs.numbers["coverage"]
    assert coverage["full"]["G"] == 15
    if coverage["restricted"] is None:
        assert coverage["restricted_unavailable_reason"]


@requires_grid
def test_absent_robustness_arms_report_a_status_rather_than_raising(inputs):
    """They run after the grid; a report that waits for a GPU session is a report
    nobody regenerates.
    """
    robustness = inputs.numbers["robustness"]
    assert set(robustness) == set(ROBUSTNESS_TAGS.values())
    for arm in robustness.values():
        assert arm["status"] in {"run", "not run"}


# -- rendered output ---------------------------------------------------------


@requires_grid
def test_every_table_is_balanced_latex_with_no_non_values(inputs, tmp_path):
    paths = render_tables(inputs.numbers, tmp_path)
    # Nine of root §13.4, plus Table 9 --- the exploratory arms in their own table,
    # which §13.2 requires and which nothing rendered before (`D64`, `D70`).
    assert len(paths) == 10
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert text.count(r"\begin{tabular}") == text.count(r"\end{tabular}") == 1
        assert text.count(r"\begin{table}") == text.count(r"\end{table}") == 1
        hits = NON_VALUE.findall(text)
        assert not hits, f"{path.name} prints {hits}, which are gaps not values"


@requires_grid
def test_rq3_uses_the_undefined_wording_not_the_censored_one(inputs, tmp_path):
    """`D60b`, and the two sentences are not interchangeable. "No decay detected
    within 180 days" is §3's RIGHT-CENSORED phrasing: it asserts an edge that
    decayed slowly. All fifteen origins have mean R2_oos <= 0, so there is no
    edge and the estimand is undefined --- large, small and absent are all wrong.
    """
    render_tables(inputs.numbers, tmp_path)
    body = (tmp_path / "table5_decay.tex").read_text(encoding="utf-8").lower()
    assert "undefined" in body
    assert "untestable" in body

    # The note deliberately *quotes* the forbidden sentence in order to disown
    # it, which is worth keeping: a referee reading Table 5 is better served by
    # being told which claim is not being made than by silence. So the assertion
    # is about context, not presence --- every occurrence must sit inside a
    # negating clause, and a naive `not in` check could not tell the difference.
    forbidden = "no decay detected"
    for match in re.finditer(re.escape(forbidden), body):
        window = body[max(0, match.start() - 80):match.start()]
        assert "not" in window, (
            "Table 5 states the right-censored wording without disowning it; "
            "it asserts an edge the data does not contain (`D60b`)"
        )
    assert "right-censored" in body


@requires_grid
def test_figure5_is_skipped_by_name_when_attention_was_never_persisted(inputs, tmp_path):
    """An empty axes labelled as a figure reads as a measurement of nothing."""
    messages: list[str] = []
    paths = render_figures(inputs, tmp_path, log=messages.append)
    stems = {path.stem for path in paths}
    if inputs.attention is None:
        assert not any(stem.startswith("figure5") for stem in stems)
        assert any("figure5" in message and "SKIPPED" in message for message in messages)
    else:
        assert any(stem.startswith("figure5") for stem in stems)
    # Everything that does not depend on the attention arm must be there.
    for stem in ("figure2b_rolling", "figure3_decay", "figure4_relmse",
                 "figure6_horizons", "figure7_equity"):
        assert stem in stems


@requires_grid
def test_generator_check_flag_agrees_with_the_committed_report():
    """The drift guard `D54d` added for the notebook, on the second artifact.

    Skipped rather than failed when ``paper/paper_numbers.json`` has not been
    rendered yet: the guard is about drift, and there is no drift before a first
    render.
    """
    if not (ROOT / "paper" / "paper_numbers.json").exists():
        pytest.skip("no report rendered yet")
    result = subprocess.run(
        [sys.executable, str(GENERATOR), "--check", "--quiet"],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert result.returncode == 0, result.stdout or result.stderr
