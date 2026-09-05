"""The return leg has to be lossless, or it is worse than not existing (`D88`).

``tools/notebook_to_src.py`` is the only tool in this repository that writes
``src/``, and it writes it from a file a human has been editing by hand. A
round trip that is *nearly* right would silently drop an import block or move a
docstring, and the damage would surface as a ``NameError`` hours into a Kaggle
session -- which is `D59`, exactly, and cost a twelve-hour version once.

So the property this module pins is not "it works" but **identity**: rebuilding
a module from the cells the generator just emitted for it must reproduce the
file on disk byte for byte. Nothing here writes to ``src/``; the reconstruction
is compared in memory, because a test that has to clean up after itself is a
test that leaves damage behind when it fails.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "iTransformer.ipynb"
REVERSE = ROOT / "tools" / "notebook_to_src.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def reverse():
    return _load(REVERSE, "notebook_to_src")


@pytest.fixture(scope="module")
def generator(reverse):
    return reverse._load_generator()


@pytest.fixture(scope="module")
def notebook() -> dict:
    assert NOTEBOOK.exists(), f"{NOTEBOOK} is missing; run tools/build_notebook.py"
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def test_every_module_is_recoverable_from_the_notebook(
    reverse, generator, notebook: dict
) -> None:
    """No module may be missing its cells: recovery from nothing is truncation."""
    grouped = reverse.cells_by_module(notebook)
    missing = [m for m in generator.MODULE_ORDER if m not in grouped]
    assert not missing, f"the notebook carries no cells for {missing}"


def test_cells_rejoin_to_the_flattened_body(reverse, generator, notebook: dict) -> None:
    """The forward direction's guarantee, asserted from the reverse side.

    ``tests/test_notebook_sync.py`` checks this going out. Checking it coming
    back is not redundant: this reads the cells the way the writer does, through
    ``metadata.itbtc``, so a metadata change that silently regrouped cells would
    pass there and fail here.
    """
    grouped = reverse.cells_by_module(notebook)
    for module in generator.MODULE_ORDER:
        assert "".join(grouped[module]) == generator.flatten_module_body(module), (
            f"{module}'s cells no longer rejoin to its flattened body"
        )


def test_rebuild_reproduces_every_module_byte_for_byte(
    reverse, generator, notebook: dict
) -> None:
    """**The property the whole tool rests on.**

    Feed ``rebuild_module`` the body the generator would emit and it must return
    the file that is on disk -- same bytes, same import block, same trailing
    guard. Anything less means an unchanged module would be rewritten into a
    *different* unchanged module, and every ``code_sha256`` in the study would
    move for no reason at all.
    """
    grouped = reverse.cells_by_module(notebook)
    for module in generator.MODULE_ORDER:
        body = "".join(grouped[module])
        rebuilt = reverse.rebuild_module(generator, module, body)
        on_disk = (generator.PACKAGE / module).read_text(encoding="utf-8")
        assert rebuilt == on_disk, (
            f"rebuilding {module} from its cells does not reproduce the file on "
            f"disk. The import block or the trailing guard is landing in the "
            f"wrong place, and writing it would corrupt the module."
        )


def test_rebuilt_modules_still_parse(reverse, generator, notebook: dict) -> None:
    """A reconstruction that does not compile is the worst possible outcome."""
    grouped = reverse.cells_by_module(notebook)
    for module in generator.MODULE_ORDER:
        rebuilt = reverse.rebuild_module(generator, module, "".join(grouped[module]))
        ast.parse(rebuilt, filename=module)


def test_an_import_added_in_a_cell_is_detected(reverse) -> None:
    """Imports cannot be added from a cell, and the refusal must be specific.

    Module-level imports are stripped by the flattening and re-emitted once in
    the Library cell, which is generated *from* the modules (`D66`). A cell
    therefore has no import lines to edit, so one appearing in a cell body is
    new text with no correct place to go. Detecting it is what turns a silent
    corruption into a message naming ``src/``.
    """
    body = '"""Docstring."""\n\nimport itertools\n\n\ndef f():\n    return 1\n'
    assert reverse._module_level_imports_in(body, "probe.py") == ["import itertools"]

    clean = '"""Docstring."""\n\n\ndef f():\n    import itertools\n    return 1\n'
    assert reverse._module_level_imports_in(clean, "probe.py") == [], (
        "a function-local import is not a module-level one and must not be "
        "refused: report._pyplot defers matplotlib on purpose"
    )


def test_the_notebook_carries_the_sync_cell_and_it_is_inert(notebook: dict) -> None:
    """The activation cell exists, is last, and executes nothing as committed.

    Fully commented is not a stylistic choice. On Kaggle there is no ``tools/``
    and no ``src/``: an active cell there either raises in the last cell of a
    twelve-hour session or writes rubbish into the session's working directory.
    Activation has to be a deliberate act in a local checkout.
    """
    last = notebook["cells"][-1]
    assert last["cell_type"] == "code"
    assert last["metadata"]["itbtc"]["step"] == "sync_back"
    source = "".join(last["source"])
    assert ast.parse(source).body == [], "the sync cell executes something"
    assert "notebook_to_src.py" in source, "the sync cell does not name its tool"
