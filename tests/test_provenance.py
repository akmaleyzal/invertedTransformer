"""The provenance table and the module docstrings are two copies. Bind them.

Root §12 requires every number in the manuscript to resolve to a persisted
artifact, a config hash and a documented decision. It says nothing about where
the *algorithms* came from, and until now neither did the code: every docstring
named its paper by author-year and not one named a repository, a licence or a
release. `D16` is what that costs -- this project already carries mis-dated
references assembled from memory, and an examiner who asks where a transformer
implementation came from is asking a question author-year cannot answer.

:data:`itransformer_btc.config.SOURCE_PROVENANCE` is the answer, and the module
docstrings repeat the part of it a reader meets inside the cell. That is two
artifacts required to agree, which is exactly the shape of `D54a` (notebook and
code Dataset kept in step by hand) and `D69` (``AGENTS.md`` as a stale copy of
``CLAUDE.md``). Both were defects because **nothing checked**. This module is
the check.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from itransformer_btc.config import SOURCE_PROVENANCE, Upstream

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "itransformer_btc"

#: The three values ``status`` may take. ``reimplemented`` means written here
#: from the published description, ``library`` means imported and called, and
#: ``own`` means no upstream code exists at all -- the algorithm is defined in a
#: paper and implemented here. The distinction is the whole point of the table:
#: a reader asking "did you copy this?" gets a different answer for each.
STATUSES = frozenset({"reimplemented", "library", "own"})

#: Rows whose repository page or indexing entry was read on 2026-09-03. Every
#: other row carries ``verified=False`` and root §13.3 still owes it a check.
#: The set is asserted rather than merely counted, so flipping a flag without
#: doing the reading fails here.
VERIFIED_REPOS = {
    "https://github.com/thuml/iTransformer",
    "https://github.com/cure-lab/LTSF-Linear",
    "https://github.com/yuqinie98/PatchTST",
    "https://github.com/ts-kim/RevIN",
}


def _module_source(name: str) -> str:
    return (PACKAGE / name).read_text(encoding="utf-8")


def _module_docstring(name: str) -> str:
    return ast.get_docstring(ast.parse(_module_source(name))) or ""


def test_every_row_names_a_module_that_exists() -> None:
    """A row describing a module that is not there describes nothing."""
    for row in SOURCE_PROVENANCE:
        assert (PACKAGE / row.module).is_file(), (
            f"{row.component!r} claims to live in {row.module}, which is not a "
            f"file in src/itransformer_btc/"
        )


def test_status_is_one_of_the_three_declared_values() -> None:
    for row in SOURCE_PROVENANCE:
        assert row.status in STATUSES, (
            f"{row.component!r} has status {row.status!r}. A fourth value blurs "
            f"the copied/called/written distinction the table exists to draw."
        )


def test_repo_url_appears_in_the_module_it_describes() -> None:
    """The binding. A URL in the table and nowhere else drifts silently.

    This is the assertion `D54a` and `D69` would have needed. It is deliberately
    an exact substring test against the module *source*, not a fuzzy match: a
    URL edited in one place and not the other is precisely the failure being
    prevented, and a tolerant comparison would pass it.
    """
    for row in SOURCE_PROVENANCE:
        if not row.repo:
            continue
        assert row.repo in _module_source(row.module), (
            f"{row.repo} is in SOURCE_PROVENANCE for {row.component!r} but not "
            f"in {row.module}. A reader who opens that cell sees no source at "
            f"all, which is the state this table was added to end."
        )


def test_a_row_without_a_repo_is_written_here_and_says_so() -> None:
    """No repo means no upstream code, and only ``own`` may claim that."""
    for row in SOURCE_PROVENANCE:
        if row.repo:
            continue
        assert row.status == "own", (
            f"{row.component!r} is {row.status!r} but names no repository. "
            f"reimplemented and library both assert upstream code exists, so "
            f"one of the two fields is wrong."
        )


def test_third_party_rows_carry_a_licence_and_an_access_date() -> None:
    """Attribution is a licence term, not a courtesy, wherever code is reused."""
    for row in SOURCE_PROVENANCE:
        if not row.repo:
            continue
        assert row.licence, f"{row.component!r} names {row.repo} with no licence"
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", row.accessed), (
            f"{row.component!r} has accessed={row.accessed!r}; an IEEE software "
            f"citation carries an access date, and ISO is the one format that "
            f"cannot be read two ways"
        )


def test_verified_flag_matches_what_was_actually_read() -> None:
    """``verified`` is a record, not an aspiration (root §13.3, `D16`)."""
    claimed = {row.repo for row in SOURCE_PROVENANCE if row.verified}
    assert claimed == VERIFIED_REPOS, (
        f"verified=True rows are {sorted(claimed)} but the reading log says "
        f"{sorted(VERIFIED_REPOS)}. Flipping the flag is not the same as "
        f"opening the page."
    )


def test_every_reimplemented_row_states_what_was_changed() -> None:
    """Reimplemented with an empty ``adapted`` is an unfinished claim.

    If nothing was changed the honest word is ``library``. The departures are
    what an examiner comparing this code against the upstream release finds
    anyway; stating them first is the difference between a design decision and a
    discrepancy.
    """
    for row in SOURCE_PROVENANCE:
        if row.status != "reimplemented":
            continue
        assert row.adapted.strip(), (
            f"{row.component!r} is reimplemented but lists no departures"
        )


@pytest.mark.parametrize(
    "module",
    sorted({row.module for row in SOURCE_PROVENANCE}),
)
def test_module_docstring_carries_an_upstream_section(module: str) -> None:
    """A reader lands on the Header cell. The answer has to be there.

    The notebook is segmented so that a module's docstring is its first cell
    (`D63`), which is where someone scrolling to ``PatchTST`` or ``dm_test``
    arrives. A provenance table forty cells above is not where they are looking.
    """
    assert "Upstream" in _module_docstring(module), (
        f"{module}'s module docstring has no Upstream section, so its first "
        f"notebook cell answers the one question this work exists to answer "
        f"only by pointing somewhere else."
    )


def test_the_ladder_and_the_baselines_are_all_accounted_for() -> None:
    """Every model the grid runs has a row. A silent gap is `D74` again.

    `D74` is the precedent: the manifest grew to 1,620 runs while the reporting
    layer still knew 684, so 480 runs produced no row anywhere and nothing
    failed. The same failure here would leave a model in the results table whose
    provenance nobody wrote down.
    """
    covered = " ".join(row.component for row in SOURCE_PROVENANCE)
    for name in (
        "ITransformer",
        "DLinear",
        "PatchTST",
        "LSTMForecaster",
        "RidgeForecaster",
        "NaiveForecaster",
    ):
        assert name in covered, f"{name} runs in the grid and has no provenance row"


def test_rows_are_unique_per_component() -> None:
    seen: set[str] = set()
    for row in SOURCE_PROVENANCE:
        assert row.component not in seen, f"{row.component!r} appears twice"
        seen.add(row.component)


def test_upstream_is_immutable() -> None:
    """Frozen, so no caller can edit provenance at runtime and then report it."""
    row = SOURCE_PROVENANCE[0]
    assert isinstance(row, Upstream)
    with pytest.raises((AttributeError, TypeError)):
        row.repo = "https://example.invalid"  # type: ignore[misc]
