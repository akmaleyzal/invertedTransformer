"""The notebook carries the package; these tests stop the two copies drifting.

`D54` made ``notebooks/iTransformer.ipynb`` self-contained: it materialises
``itransformer_btc/`` from ``%%writefile`` cells before importing it, so a Kaggle
session needs the notebook and the data artifact and nothing else. That buys
independence at the cost of a second copy of 4,000 lines, and an unpoliced second
copy is a worse defect than the dependency it removes — the notebook would keep
running old code, and every number it produced would be traceable to a version
nobody could find.

So the copy is **generated** by ``tools/build_notebook.py`` and asserted here to
be byte-identical to ``src/``. Editing ``src/`` without re-running the generator
fails :func:`test_notebook_is_not_stale`, which is the only moment anyone would
otherwise notice.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "src" / "itransformer_btc"
NOTEBOOK = ROOT / "notebooks" / "iTransformer.ipynb"
GENERATOR = ROOT / "tools" / "build_notebook.py"

WRITEFILE = re.compile(r"^%%writefile (itransformer_btc/[\w.]+)\n")


@pytest.fixture(scope="module")
def notebook() -> dict:
    assert NOTEBOOK.exists(), f"{NOTEBOOK} is missing; run tools/build_notebook.py"
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _sources(notebook: dict) -> list[str]:
    return ["".join(cell["source"]) for cell in notebook["cells"]]


def _code_sources(notebook: dict) -> list[str]:
    """Code cells only.

    The markdown cells name both ``itransformer_btc`` and
    ``src/itransformer_btc/`` on purpose — the title block has to say where the
    materialised copy came from, or a reader edits the ``%%writefile`` cells by
    hand and the generator silently reverts the edit on its next run.
    """
    return [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]


def _materialised(notebook: dict) -> dict[str, str]:
    """Target path to written body, for every ``%%writefile`` cell."""
    out: dict[str, str] = {}
    for source in _sources(notebook):
        match = WRITEFILE.match(source)
        if match:
            out[match.group(1)] = source[match.end() :]
    return out


def test_notebook_materialises_every_module(notebook: dict) -> None:
    """Every module in the package is written, and nothing else is."""
    expected = {f"itransformer_btc/{p.name}" for p in PACKAGE.glob("*.py")}
    assert _materialised(notebook).keys() == expected


def test_materialised_source_is_byte_identical(notebook: dict) -> None:
    """The written body equals the file on disk, character for character.

    Not "equivalent", not "equal after formatting": identical. A single stale
    line here is a run executing code that is not in the repository, and root
    §12 has no way to detect that after the fact.
    """
    for target, body in sorted(_materialised(notebook).items()):
        on_disk = (PACKAGE / Path(target).name).read_text(encoding="utf-8")
        assert body == on_disk, f"{target} differs; run tools/build_notebook.py"


def test_notebook_imports_nothing_from_src(notebook: dict) -> None:
    """No cell reaches into ``src/`` — that dependency is the point of `D54`.

    The old launcher globbed for ``src/itransformer_btc/__init__.py`` and pushed
    the hit onto ``sys.path``. Nothing may do that any more: the package the
    notebook imports must be the one its own cells wrote, which the notebook
    additionally asserts at runtime by checking ``__file__``.
    """
    for source in _code_sources(notebook):
        if WRITEFILE.match(source):
            continue  # the package's own docstrings legitimately mention src/
        assert "src/itransformer_btc" not in source
        assert 'find_one("src"' not in source
        assert "sys.path.insert" not in source or "WORK" in source


def test_package_is_importable_before_first_use(notebook: dict) -> None:
    """Every ``%%writefile`` cell precedes the first ``import itransformer_btc``.

    Cell order is the only thing keeping this true, and *Save & Run All* executes
    in cell order — so an import placed above the writes would fail on Kaggle and
    nowhere else.
    """
    imports = re.compile(r"^(?:from|import) itransformer_btc\b", re.MULTILINE)
    sources = _code_sources(notebook)
    last_write = max(i for i, s in enumerate(sources) if WRITEFILE.match(s))
    first_import = min(
        i
        for i, s in enumerate(sources)
        if not WRITEFILE.match(s) and imports.search(s)
    )
    assert last_write < first_import


def test_notebook_is_valid_nbformat_with_gpu_metadata(notebook: dict) -> None:
    assert notebook["nbformat"] == 4 and notebook["nbformat_minor"] >= 5
    assert notebook["metadata"]["kernelspec"]["name"] == "python3"
    # Kaggle reads this to pre-select the accelerator; without it the notebook
    # opens on CPU and the 534-run grid quietly becomes a 100-hour job.
    assert notebook["metadata"]["accelerator"] == "GPU"
    for cell in notebook["cells"]:
        assert cell["cell_type"] in {"code", "markdown"}
        assert isinstance(cell["source"], list)
        assert "id" in cell
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [], "committed outputs go stale; strip them"


def test_notebook_is_not_stale() -> None:
    """``build_notebook.py --check`` agrees the committed file is current."""
    result = subprocess.run(
        [sys.executable, str(GENERATOR), "--check"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr or result.stdout
