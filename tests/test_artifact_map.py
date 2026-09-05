"""A cell may not claim an artefact its code does not write (`D87`).

The notebook is the artefact an examiner opens, and until `D87` it could not
answer the first question anyone asks of it: *which cell makes the figures, and
which one writes the metric parquets?* One hundred and forty-four definition
cells carried ``metadata.itbtc``; the seventeen cells that actually produce
something carried nothing at all, so the producers were unfindable except by
reading 354 cells in order.

The manifest fixes that, and this module is what stops it from becoming a
second thing to keep in step by hand -- which is `D54a` and `D69`, twice over.
A manifest nobody checks is worse than no manifest: it *looks* authoritative
while pointing at a cell that stopped writing figures three commits ago.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "iTransformer.ipynb"

#: Steps whose artefacts are written by a package function rather than by a
#: literal in the cell body, with the reason each one is unreachable by grep.
#:
#: Kept explicit and short. An allowlist that grows without argument is how a
#: guard stops guarding, so each entry names the callee that does the writing --
#: a reader can follow it, and an entry that stops being true stands out.
INDIRECT_WRITERS: dict[str, str] = {
    "grid": (
        "execute_parallel -> train_one -> write_artifacts writes preds/ and "
        "meta/; the cell itself names neither path"
    ),
    "pilot": (
        "stage5_pilot writes through the same write_artifacts; the cell passes "
        "out_root and nothing else"
    ),
}

#: Slugs every reader is entitled to find, whatever else the notebook grows.
#:
#: These two are the question `D87` was raised by, quoted almost verbatim: where
#: are the figures made, and where are the metric parquets written.
REQUIRED_PRODUCERS = {"report", "save"}


@pytest.fixture(scope="module")
def cells() -> list[dict]:
    assert NOTEBOOK.exists(), f"{NOTEBOOK} is missing; run tools/build_notebook.py"
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]


def _steps(cells: list[dict]) -> list[tuple[int, dict, str]]:
    out = []
    for index, cell in enumerate(cells):
        tag = cell.get("metadata", {}).get("itbtc", {})
        if tag.get("role") == "step":
            out.append((index, tag, "".join(cell["source"])))
    return out


def _witness(pattern: str) -> str:
    """The most specific literal a declared path can be looked for by.

    A concrete basename is the strongest witness available. A glob has none, so
    fall back to the directory that contains it: ``paper/figures/*.pdf`` is
    evidenced by the cell mentioning ``figures``. Weaker, and still enough to
    catch a manifest pointing at a cell that has nothing to do with the path.
    """
    name = pattern.rsplit("/", 1)[-1]
    if "*" not in name:
        return name
    parts = [p for p in pattern.split("/") if "*" not in p]
    return parts[-1] if parts else pattern


def test_every_step_cell_has_a_slug(cells: list[dict]) -> None:
    """Position is not identity: a slug survives inserting a phase, an index does not."""
    for index, tag, _ in _steps(cells):
        assert tag.get("step"), f"cell {index} has role=step and no slug"


def test_slugs_are_unique(cells: list[dict]) -> None:
    seen: set[str] = set()
    for index, tag, _ in _steps(cells):
        slug = tag["step"]
        assert slug not in seen, f"cell {index} reuses the slug {slug!r}"
        seen.add(slug)


def test_declared_writes_are_evidenced_by_the_cell_body(cells: list[dict]) -> None:
    """The binding. A manifest that outruns its code is worse than none."""
    for index, tag, source in _steps(cells):
        slug = tag["step"]
        if slug in INDIRECT_WRITERS:
            continue
        for pattern in tag.get("writes", []):
            witness = _witness(pattern)
            assert witness in source, (
                f"cell {index} [{slug}] declares it writes {pattern!r}, but "
                f"{witness!r} appears nowhere in its body. Either the cell "
                f"stopped writing it, or the manifest was copied from a "
                f"neighbour. If a package function does the writing, say so in "
                f"INDIRECT_WRITERS with the callee named."
            )


def test_indirect_writers_still_declare_something(cells: list[dict]) -> None:
    """An allowlist entry is a promise the cell writes *something*."""
    slugs = {tag["step"]: tag for _, tag, _ in _steps(cells)}
    for slug, reason in INDIRECT_WRITERS.items():
        assert slug in slugs, f"INDIRECT_WRITERS names {slug!r}, which no cell is"
        assert slugs[slug].get("writes"), (
            f"{slug!r} is allowlisted as an indirect writer but declares no "
            f"writes at all, so the exemption covers nothing"
        )
        assert reason.strip(), f"{slug!r} is allowlisted with no reason given"


def test_the_two_questions_a_reader_arrives_with_are_answerable(
    cells: list[dict],
) -> None:
    """Figures and metric parquets each resolve to exactly one named cell."""
    producers = {
        tag["step"]: (index, tag)
        for index, tag, _ in _steps(cells)
        if tag.get("writes")
    }
    assert REQUIRED_PRODUCERS <= set(producers), (
        f"missing producers {sorted(REQUIRED_PRODUCERS - set(producers))}; the "
        f"notebook can no longer say where its deliverables come from"
    )

    figure_cells = [
        slug for slug, (_, tag) in producers.items()
        if any(w.endswith((".pdf", ".png")) for w in tag["writes"])
    ]
    assert figure_cells == ["report"], (
        f"figures should be produced by exactly one cell; found {figure_cells}"
    )

    parquet_cells = [
        slug for slug, (_, tag) in producers.items()
        if any(
            w.endswith(".parquet") and w.startswith("artifacts/")
            for w in tag["writes"]
        )
    ]
    assert "save" in parquet_cells, (
        f"no cell claims the metric panels; producers of artifacts/*.parquet "
        f"are {parquet_cells}"
    )


def test_declared_paths_are_relative_and_glob_shaped(cells: list[dict]) -> None:
    """No absolute paths and no backslashes: these strings are read by people."""
    for index, tag, _ in _steps(cells):
        for pattern in list(tag.get("writes", [])) + list(tag.get("reads", [])):
            assert not pattern.startswith(("/", "C:", "\\")), (
                f"cell {index} declares the absolute path {pattern!r}; the "
                f"notebook runs on Kaggle and locally and neither root is fixed"
            )
            assert "\\" not in pattern, f"cell {index}: {pattern!r} uses backslashes"
            assert re.fullmatch(r"[\w./*\-]+", pattern), (
                f"cell {index}: {pattern!r} is not a plain relative path or glob"
            )


def test_the_artifact_map_cell_lists_every_producer(cells: list[dict]) -> None:
    """The index at the top and the manifest below it are one source (`D87`).

    The map is generated from the emitted cells rather than from ``PHASES``, so
    this checks the property that makes that worth doing: every producing cell
    appears in it, at its real index.
    """
    map_source = next(
        (
            "".join(c["source"])
            for c in cells
            if c.get("metadata", {}).get("itbtc", {}).get("step") == "artifact_map"
        ),
        None,
    )
    assert map_source is not None, "no cell carries the artifact_map step"
    for index, tag, _ in _steps(cells):
        if not tag.get("writes"):
            continue
        entry = f"({index}, {tag['step']!r}"
        assert entry in map_source, (
            f"cell {index} [{tag['step']}] writes {tag['writes']} but the "
            f"artefact map does not list it at that index. The map is built "
            f"from the emitted cells, so this means the notebook was hand-edited."
        )
