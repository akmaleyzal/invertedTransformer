"""The notebook carries the package; these tests stop the two copies drifting.

``notebooks/iTransformer.ipynb`` is self-contained: it **defines** every module
in plain cells and calls them by name, so a Kaggle session needs the
notebook and the data artifact and nothing else. That buys independence at the
cost of a second copy of 4,000 lines, and an unpoliced second copy is a worse
defect than the dependency it removes — the notebook would keep running old code,
and every number it produced would be traceable to a version nobody could find.

So the copy is **generated** by ``tools/build_notebook.py`` and asserted here to
match ``src/`` under exactly one declared transformation: intra-package imports
removed, the ``__main__`` guard removed, and the two subprocess-only functions in
``runner`` removed. Editing ``src/`` without re-running the generator fails
:func:`test_notebook_is_not_stale`, which is the only moment anyone would
otherwise notice.

Two of these tests are deliberately **not** expressed in terms of the generator.
Comparing a cell against ``flatten_module_source`` proves the committed file is
current but says nothing about whether the transformation is *correct*, because
both sides come from the same function. :func:`test_no_executable_package_refs`
and :func:`test_definition_cells_execute_in_one_namespace` re-derive the property
that matters — that these cells run on a machine with no ``itransformer_btc``
installed — from the notebook alone.
"""

from __future__ import annotations

import ast
import importlib.util
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

BANNER = re.compile(r"^# ═+ (?P<name>[\w.]+) ═+\n")


def _load_generator():
    """Import ``tools/build_notebook.py`` without making ``tools`` a package."""
    spec = importlib.util.spec_from_file_location("build_notebook", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def generator():
    return _load_generator()


@pytest.fixture(scope="module")
def notebook() -> dict:
    assert NOTEBOOK.exists(), f"{NOTEBOOK} is missing; run tools/build_notebook.py"
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _strip_magics(source: str) -> str:
    """Blank out IPython line magics so ``ast.parse`` can read a scaffolding cell.

    ``%pip install`` and friends are not Python and would raise SyntaxError.
    Blanking rather than deleting keeps line numbers aligned with the cell, so an
    assertion message still points at the right line.
    """
    return "\n".join(
        "" if line.lstrip().startswith(("%", "!")) else line
        for line in source.splitlines()
    )


def _code_sources(notebook: dict) -> list[str]:
    return [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]


def _module_cells(notebook: dict) -> dict[str, str]:
    """Module name to cell body, for every banner-headed cell, in notebook order."""
    out: dict[str, str] = {}
    for source in _code_sources(notebook):
        match = BANNER.match(source)
        if match:
            out[match.group("name")] = source[match.end():]
    return out


def test_notebook_defines_every_module(notebook: dict, generator) -> None:
    """Every module in the package gets a cell, nothing else does, order preserved.

    Order is asserted because it is now execution order: these cells run, so a
    cell naming something a later cell defines fails at once. Under
    ``%%writefile`` the same list was only a reading convenience.
    """
    cells = _module_cells(notebook)
    assert set(cells) == {p.name for p in PACKAGE.glob("*.py")}
    assert list(cells) == list(generator.MODULE_ORDER)


def test_module_cell_matches_flattened_source(notebook: dict, generator) -> None:
    """Each cell body equals the module under the declared transformation.

    Not "equivalent", not "equal after formatting": identical. A single stale line
    here is a run executing code that is not in the repository, and root §12 has
    no way to detect that after the fact.
    """
    for name, body in sorted(_module_cells(notebook).items()):
        assert body == generator.flatten_module_source(name), (
            f"{name} differs from src/; run tools/build_notebook.py"
        )


def test_flattening_only_removes_what_it_declares(generator) -> None:
    """The transformation is subtractive, and only over the declared categories.

    Guards the generator itself: a future edit that started *rewriting* module
    source rather than deleting whole statements would still satisfy the
    byte-comparison above, because both sides would move together.
    """
    for name in generator.MODULE_ORDER:
        original = (PACKAGE / name).read_text(encoding="utf-8").splitlines()
        flattened = generator.flatten_module_source(name).splitlines()
        assert len(flattened) <= len(original)
        # Every surviving line must appear in the original, unmodified.
        remaining = iter(original)
        for line in flattened:
            assert any(line == candidate for candidate in remaining), (
                f"{name}: {line!r} is not a verbatim line of the original, so the "
                f"flattening rewrote source instead of only deleting statements"
            )


def test_no_executable_package_refs(notebook: dict) -> None:
    """No **module** cell executably names ``itransformer_btc``.

    Re-derived from the notebook rather than from the generator: the whole point
    of the format is that these cells run where no such package exists, and a
    test that asked the generator would only confirm the generator agrees with
    itself.

    Docstrings are stripped first because the modules legitimately *mention* the
    package in prose — ``:mod:`` cross-references, ``Importers:`` notes — and
    ``ast.unparse`` drops comments on its own.

    Scoped to module cells on purpose. The notebook's own scaffolding names the
    package in a string literal, in the ``"itransformer_btc" not in sys.modules``
    guard, which is the assertion that an installed copy did **not** shadow these
    definitions. Forbidding the name everywhere would forbid the check that
    enforces this very property.
    """
    for name, body in _module_cells(notebook).items():
        tree = ast.parse(body)
        for node in ast.walk(tree):
            if isinstance(
                node,
                (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
            ) and ast.get_docstring(node):
                node.body = node.body[1:] or [ast.Pass()]
        executable = ast.unparse(ast.fix_missing_locations(tree))
        assert "itransformer_btc" not in executable, (
            f"{name} still executably references the package, so the cell would "
            f"fail on a machine with no such package installed"
        )


def test_no_writefile_and_no_main_guard(notebook: dict) -> None:
    """Neither of the two things the old format relied on survives.

    ``%%writefile`` would materialise a package this notebook must not have, and
    a ``__main__`` guard is live code in a cell — ``__name__`` *is*
    ``"__main__"`` there, so ``runner``'s guard would launch the whole grid the
    moment its definition cell ran.
    """
    for source in _code_sources(notebook):
        assert "%%writefile" not in source
        assert '__name__ == "__main__"' not in source
        assert "sys.path.insert" not in source

    # The repository dependency this format removes is an **import**, not a
    # spelling. `src/itransformer_btc` is legitimately named in prose by the
    # transcribed modules, and by the provenance cell recording the directory its
    # digest was taken from; forbidding the string fails on exactly the cells that
    # document this correctly. What no cell may do is import the package, which
    # would not exist on the machine running it.
    #
    # Module cells are additionally held to the stronger rule in
    # :func:`test_no_executable_package_refs` — no executable mention at all.
    for source in _code_sources(notebook):
        for node in ast.walk(ast.parse(_strip_magics(source))):
            if isinstance(node, ast.Import):
                offenders = [a.name for a in node.names
                             if a.name.split(".")[0] == "itransformer_btc"]
                assert not offenders, f"cell imports {offenders}"
            elif isinstance(node, ast.ImportFrom):
                root = (node.module or "").split(".")[0]
                assert root != "itransformer_btc", f"cell imports from {node.module}"

    assert _module_cells(notebook), "no module cells found; the banner format changed"


def test_definitions_precede_the_first_call(notebook: dict, generator) -> None:
    """Every module cell comes before the cell that first calls into them.

    Cell order is the only thing keeping this true, and *Save & Run All* executes
    in cell order — so a call placed above the definitions would fail on Kaggle
    and nowhere else.
    """
    sources = _code_sources(notebook)
    last_definition = max(i for i, s in enumerate(sources) if BANNER.match(s))
    first_call = min(
        i for i, s in enumerate(sources) if "CODE_SHA256_OVERRIDE = " in s
    )
    assert last_definition < first_call


def test_pinned_digest_matches_the_package(notebook: dict, generator) -> None:
    """The digest in the notebook is the one ``src/`` produces.

    Root §12 asks a run to name the code that produced it, and the answer has to
    be the same number whether that code ran from a checkout or from this
    notebook. Drift here reads as a phantom change of code vintage — a false
    positive on the one check §12 exists to make possible.
    """
    expected = generator.package_digest()
    pinned = [
        s for s in _code_sources(notebook) if "CODE_SHA256_OVERRIDE = " in s
    ]
    assert len(pinned) == 1
    assert f'CODE_SHA256_OVERRIDE = "{expected}"' in pinned[0]


def test_definition_cells_execute_in_one_namespace(notebook: dict) -> None:
    """The definition cells actually run, together, and produce the committed figures.

    The strongest test here and the only one that exercises the format rather
    than describing it. Flattened cells are *executed*: decorators run, dataclass
    field types resolve, module constants evaluate, and two module-level names
    collide across modules (``DEFAULT_PARQUET``, ``HOUR_MS`` — same value in both
    definitions, so last-cell-wins is harmless). None of that is visible to a
    parse check.

    ``__name__`` is set to ``"__main__"`` deliberately: that is what a real cell
    sees, and it is the condition that made ``runner``'s guard dangerous.
    """
    torch = pytest.importorskip("torch")
    namespace: dict = {"__name__": "__main__"}

    for name, body in _module_cells(notebook).items():
        exec(compile(body, f"<cell:{name}>", "exec"), namespace)

    # Committed expected values — CLAUDE.md §6.2 and `D52`.
    model = namespace["ITransformer"](namespace["ITransformerConfig"]())
    assert model.n_parameters() == 280_472
    assert len(namespace["ORIGINS"]) == 15
    assert len(namespace["VARIATE_ORDER"]) == 12
    assert len(namespace["manifest"]()) == 684
    assert namespace["ladder_columns"](1) == ["r"]

    # Every arm must be able to *build and run its model* here, not merely be
    # listed. `runner` reaches the baselines by bare name for exactly this
    # reason: one merged namespace has no `baselines` module to attribute off, so
    # `baselines.RidgeConfig` would satisfy every parse-level check in this file
    # and then raise NameError hours into a Kaggle session (`D56`, `D58`).
    cells = namespace["manifest"]()
    for arm in namespace["ALL_ARMS"]:
        cell = next(c for c in cells if c.arm == arm)
        built = cell.model_config().build().eval()
        out = built.forecast_target(torch.randn(2, 96, max(cell.k, 1)))
        assert out.shape == (2, cell.pred_len), f"{arm} built the wrong horizon"
    del torch

    # The subprocess path must not have survived: it spawns
    # `python -m itransformer_btc.runner`, which cannot resolve here.
    assert "launch_workers" not in namespace
    assert "_main" not in namespace


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
