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

import __future__
import ast
import builtins
import importlib.util
import json
import re
import subprocess
import symtable
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "src" / "itransformer_btc"
NOTEBOOK = ROOT / "notebooks" / "iTransformer.ipynb"
GENERATOR = ROOT / "tools" / "build_notebook.py"



#: Key under which a code cell records the module and section it came from.
#: Cells carry their identity here rather than in a banner comment, so a cell
#: body stays a byte-exact slice of its module and the concatenation check below
#: needs no normalisation step (`D63`).
CELL_TAG = "itbtc"


def _load_generator():
    """Import ``tools/build_notebook.py`` without making ``tools`` a package."""
    spec = importlib.util.spec_from_file_location("build_notebook", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # Registered before execution because ``dataclasses`` resolves a frozen
    # class's annotations through ``sys.modules[cls.__module__]``; an unregistered
    # module makes that lookup return None and the decorator raises.
    sys.modules[spec.name] = module
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


def _code_cells(notebook: dict) -> list[dict]:
    return [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]


def _module_of(cell: dict) -> str | None:
    """The module a code cell was cut from, or ``None`` for a scaffolding cell."""
    return (cell.get("metadata", {}).get(CELL_TAG) or {}).get("module")


def _module_cells(notebook: dict) -> dict[str, list[str]]:
    """Module name to its ordered cell bodies, in notebook order."""
    out: dict[str, list[str]] = {}
    for cell in _code_cells(notebook):
        name = _module_of(cell)
        if name:
            out.setdefault(name, []).append("".join(cell["source"]))
    return out


def _library_cell(notebook: dict) -> str:
    """The one cell that carries every import the package makes (`D66`)."""
    for cell in _code_cells(notebook):
        if (cell.get("metadata", {}).get(CELL_TAG) or {}).get("role") == "library":
            return "".join(cell["source"])
    raise AssertionError("no library cell; run tools/build_notebook.py")


def _inherited_flags(source: str) -> int:
    """``__future__`` compiler flags a cell leaves behind for the cells after it.

    IPython accumulates these across a session — ``InteractiveShell.compile.flags``
    is OR-ed with every future import a cell executes, and Kaggle runs the
    notebook through papermill to ipykernel to ``run_cell``, the same path. That
    is why the library cell can carry ``from __future__ import annotations`` once
    for the whole notebook and no definition cell repeats it (`D67`).

    Measured on IPython 9.13: ``compile.flags`` goes ``16896 -> 16794112`` after
    that cell, and a following cell with no future import of its own defines a
    forward annotation without raising. Plain :func:`compile` does **not** inherit,
    so a test that used it would be modelling a different interpreter than the one
    the notebook runs on.
    """
    flags = 0
    for node in ast.parse(source).body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            for alias in node.names:
                flags |= getattr(__future__, alias.name).compiler_flag
    return flags


def _module_source(notebook: dict, generator) -> dict[str, str]:
    """Module name to its cells rejoined — what the module body must equal.

    Nothing to undo: a cell is a slice of ``flatten_module_body`` and only a
    slice. The imports, the ``__future__`` directive among them, live in the
    library cell (`D66`, `D67`).
    """
    return {
        name: "".join(bodies)
        for name, bodies in _module_cells(notebook).items()
    }


def test_notebook_defines_every_module(notebook: dict, generator) -> None:
    """Every module in the package gets a cell, nothing else does, order preserved.

    Order is asserted because it is now execution order: these cells run, so a
    cell naming something a later cell defines fails at once. Under
    ``%%writefile`` the same list was only a reading convenience.
    """
    cells = _module_cells(notebook)
    assert set(cells) == {p.name for p in PACKAGE.glob("*.py")}
    assert list(cells) == list(generator.MODULE_ORDER)
    assert set(cells) == set(generator.SECTION_MAP), (
        "SECTION_MAP and MODULE_ORDER disagree, so some module ships unsegmented"
    )
    for name, bodies in cells.items():
        assert len(bodies) == len(generator.SECTION_MAP[name]), (
            f"{name}: {len(bodies)} cells against {len(generator.SECTION_MAP[name])} "
            f"sections; run tools/build_notebook.py"
        )


def test_module_cells_concatenate_to_flattened_source(notebook, generator) -> None:
    """A module's cells, rejoined in order, equal the module itself.

    Not "equivalent", not "equal after formatting": identical. A single stale line
    here is a run executing code that is not in the repository, and root §12 has
    no way to detect that after the fact.

    Segmentation moved this guarantee from per-cell to per-module (`D63`) and
    weakened nothing: the cells are contiguous, exhaustive line ranges, so a line
    that went missing between two of them fails here exactly as a changed line
    would.
    """
    for name, body in sorted(_module_source(notebook, generator).items()):
        assert body == generator.flatten_module_body(name), (
            f"{name} differs from src/; run tools/build_notebook.py"
        )


def test_every_code_cell_has_a_markdown_heading_above_it(notebook: dict) -> None:
    """No code cell appears without prose introducing it.

    The rule the notebook exists to satisfy now that it is the artefact a
    reviewer reads rather than a launcher nobody opens (`D63`). A cell that
    arrives unannounced is one the reader has to reverse-engineer.
    """
    cells = notebook["cells"]
    naked = [
        i
        for i, cell in enumerate(cells)
        if cell["cell_type"] == "code"
        and (i == 0 or cells[i - 1]["cell_type"] != "markdown")
    ]
    assert not naked, f"code cells with no heading above them: {naked}"


def test_section_map_covers_every_top_level_statement(generator) -> None:
    """Splitting loses nothing, and every anchor still names a real definition.

    ``split_module_cells`` asserts the partition itself; this reaches it for every
    module, so a definition that moves in ``src/`` without its SECTION_MAP entry
    moving fails here rather than shipping inside the wrong heading.
    """
    for name in generator.MODULE_ORDER:
        cells = generator.split_module_cells(name)
        assert len(cells) == len(generator.SECTION_MAP[name])
        assert all(body.strip() for _, body in cells), f"{name}: empty cell"


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


def test_no_executable_package_refs(notebook: dict, generator) -> None:
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
    for name, body in _module_source(notebook, generator).items():
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

    assert _module_cells(notebook), (
        "no tagged definition cells found; the cell metadata contract changed"
    )


def test_definitions_precede_the_first_call(notebook: dict, generator) -> None:
    """Every module cell comes before the cell that first calls into them.

    Cell order is the only thing keeping this true, and *Save & Run All* executes
    in cell order — so a call placed above the definitions would fail on Kaggle
    and nowhere else.
    """
    cells = _code_cells(notebook)
    last_definition = max(i for i, c in enumerate(cells) if _module_of(c))
    first_call = min(
        i for i, c in enumerate(cells) if "CODE_SHA256_OVERRIDE = " in "".join(c["source"])
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

    # The library cell first: the definition cells carry no imports, so this is
    # what binds ``np``, ``pl``, ``torch`` and the rest (`D66`) — and what leaves
    # the ``__future__`` flags every later cell compiles under (`D67`).
    library = _library_cell(notebook)
    flags = _inherited_flags(library)
    exec(compile(library, "<cell:library>", "exec"), namespace)

    for index, cell in enumerate(_code_cells(notebook)):
        name = _module_of(cell)
        if name:
            body = "".join(cell["source"])
            code = compile(body, f"<cell:{index}:{name}>", "exec", flags=flags)
            exec(code, namespace)

    # Committed expected values — CLAUDE.md §6.2 and `D52`.
    model = namespace["ITransformer"](namespace["ITransformerConfig"]())
    assert model.n_parameters() == 280_472
    assert len(namespace["ORIGINS"]) == 15
    assert len(namespace["VARIATE_ORDER"]) == 12
    # 684 confirmatory + `D62`'s 210 exploratory + `D64`'s 75 deferred baselines
    assert len(namespace["manifest"]()) == 1_620
    assert namespace["ladder_columns"](1) == ["r"]

    # Every arm must be able to *build and run its model* here, not merely be
    # listed. `runner` reaches the baselines by bare name for exactly this
    # reason: one merged namespace has no `baselines` module to attribute off, so
    # `baselines.RidgeConfig` would satisfy every parse-level check in this file
    # and then raise NameError hours into a Kaggle session (`D56`, `D58`).
    cells = namespace["manifest"]()
    # The tuned arm has no default config on purpose (`D70`): it refuses rather
    # than silently running the untuned one, so it is exercised with the config the
    # notebook's prelude would hand it.
    overrides = {"tuned": namespace["ITransformerConfig"](pred_len=24)}
    for arm in namespace["ALL_ARMS"]:
        cell = next(c for c in cells if c.arm == arm)
        built = cell.model_config(overrides).build().eval()
        # ``cell.seq_len``, not 96: `D70`'s lookback arms are the reason this
        # cannot be a constant any more.
        out = built.forecast_target(torch.randn(2, cell.seq_len, max(cell.k, 1)))
        assert out.shape == (2, cell.pred_len), f"{arm} built the wrong horizon"
    del torch

    # The subprocess path must not have survived: it spawns
    # `python -m itransformer_btc.runner`, which cannot resolve here.
    assert "launch_workers" not in namespace
    assert "_main" not in namespace


def test_flattening_rejects_imports_that_bind_a_module_object(generator) -> None:
    """The generator can tell a dropped *name* import from a dropped *module* one.

    Both forms vanish when the cells are flattened, and only one of them is
    lossless. ``from itransformer_btc.metrics import clark_west_test`` binds a
    function another cell defines; ``from itransformer_btc import metrics`` binds
    a module object no cell defines, leaving every ``metrics.x`` in the file
    dangling. That second form shipped, and cost a Kaggle session (`D59`).

    Asserted on synthetic source rather than through
    :func:`flatten_module_source`, which reads real files: the point is that the
    two halves of the guard classify correctly, including the case that would
    make a name-matching check cry wolf — a *local* called ``metrics``.
    """

    def bindings(source: str) -> set[str]:
        return {
            name
            for node in ast.walk(ast.parse(source))
            if generator._intra_package_import(node)
            for name in generator._module_object_bindings(node)
        }

    assert bindings("from itransformer_btc import metrics") == {"metrics"}
    assert bindings("from itransformer_btc import metrics as m") == {"m"}
    assert bindings("from . import metrics") == {"metrics"}
    assert bindings("import itransformer_btc.metrics") == {"itransformer_btc"}
    # Binds a definition, not a module: harmless, and the common form here.
    assert bindings("from itransformer_btc.metrics import clark_west_test") == set()
    assert bindings("import numpy as np") == set()

    reads = generator.unbound_global_reads
    assert reads("cw = metrics.clark_west_test(y)", "<t>", {"metrics"}) == {"metrics"}
    # An import inside the flattened source really does bind it, so the guard
    # must not fire on a name the cell rebinds for itself.
    assert reads("import metrics\nmetrics.f()", "<t>", {"metrics"}) == set()
    assert reads(
        "def f():\n    return metrics.clark_west_test(y)\n", "<t>", {"metrics"}
    ) == {"metrics"}
    # A local of the same name is not a dangling global — this is why the check
    # is a symbol-table question rather than a spelling one.
    assert reads(
        "def f():\n    metrics = {}\n    return metrics['x']\n", "<t>", {"metrics"}
    ) == set()


def test_flatten_refuses_the_defect_that_reached_kaggle(generator, tmp_path, monkeypatch) -> None:
    """``flatten_module_source`` raises on the exact form that shipped.

    The classification test above proves the two helpers answer correctly; this
    proves they are *wired in*. Without it, deleting the call from
    :func:`flatten_module_source` leaves every test in this file green and puts
    the notebook back where it was on 2026-08-11.

    ``PACKAGE`` is redirected at a throwaway pair of modules so the case can be
    written as it was rather than reconstructed by editing ``src/``.
    """
    (tmp_path / "metrics.py").write_text(
        "def clark_west_test(y, small, large, h=24, name=''):\n    return 0.0\n",
        encoding="utf-8",
    )
    (tmp_path / "runner.py").write_text(
        "from itransformer_btc import metrics\n"
        "\n"
        "\n"
        "def gate(y, small, large):\n"
        "    return metrics.clark_west_test(y, small, large)\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(generator, "PACKAGE", tmp_path)
    monkeypatch.setattr(generator, "FLATTEN_DROP_FUNCTIONS", {})

    with pytest.raises(AssertionError, match="metrics"):
        generator.flatten_module_source("runner.py")

    # The same module written the way the package writes it flattens cleanly.
    (tmp_path / "runner.py").write_text(
        "from itransformer_btc.metrics import clark_west_test\n"
        "\n"
        "\n"
        "def gate(y, small, large):\n"
        "    return clark_west_test(y, small, large)\n",
        encoding="utf-8",
    )
    flattened = generator.flatten_module_source("runner.py")
    assert "clark_west_test(y, small, large)" in flattened
    assert "import" not in flattened


#: Global reads that stay unbound in the notebook **on purpose**, as
#: ``(module cell, scope, name)``.
#:
#: ``code_sha256`` hashes the ``*.py`` beside ``__file__``, and a definition cell
#: has no file. The lookup sits *after* the ``CODE_SHA256_OVERRIDE`` check for
#: exactly that reason (``train.py``), and the notebook always pins the override
#: (`D54b`), so the line is unreachable there. Listing it rather than allowing
#: ``__file__`` everywhere keeps the net closed around the one known hole.
ALLOWED_UNBOUND: set[tuple[str, str, str]] = {
    ("train.py", "code_sha256", "__file__"),
}


def _unbound_reads(source: str, label: str, defined: set[str]) -> list[tuple[str, str]]:
    """``(scope, name)`` for every global this source reads but never binds."""
    out: list[tuple[str, str]] = []
    stack = [symtable.symtable(source, label, "exec")]
    while stack:
        table = stack.pop()
        stack.extend(table.get_children())
        for sym in table.get_symbols():
            if (
                sym.is_referenced()
                and sym.is_global()
                and not _binds(sym)
                and sym.get_name() not in defined
            ):
                out.append((table.get_name(), sym.get_name()))
    return out


def _binds(sym: symtable.Symbol) -> bool:
    """Does this scope give the name a value?

    ``is_assigned()`` alone does not: an ``import gc`` sets a different flag, so
    reading ``gc`` two lines later would look unbound.
    """
    return sym.is_assigned() or sym.is_imported()


def _assigned_names(source: str, label: str) -> set[str]:
    """Module-level names a cell binds, which the cells below it may then read."""
    return {
        sym.get_name()
        for sym in symtable.symtable(source, label, "exec").get_symbols()
        if _binds(sym)
    }


def test_every_name_the_notebook_reads_is_defined_somewhere(notebook: dict) -> None:
    """No cell reads a global that nothing binds. The general net under `D59`.

    The defect that reached Kaggle was not a syntax error, not a stale cell and
    not a surviving package reference — it was a *name*, read in one function
    body, that the flattening had quietly unbound. Every other test in this file
    asks a question that defect answers correctly, which is why it shipped, and
    it surfaced only when the interpreter reached that line: past the data stage,
    past K_eff, past twelve training runs, six minutes in.

    So this asks the interpreter's own question instead, without running anything
    expensive. Definition cells are executed to get the real namespace; every
    code cell is then walked with :mod:`symtable`, which knows a local from a
    global and so does not cry wolf over a variable that shares a module's name.
    Scaffolding cells accumulate: each may read what the cells above it bound and
    nothing more, which is the rule *Save & Run All* enforces anyway.
    """
    pytest.importorskip("torch")
    namespace: dict = {"__name__": "__main__"}
    library = _library_cell(notebook)
    flags = _inherited_flags(library)
    exec(compile(library, "<cell:library>", "exec"), namespace)
    definition_cells = [
        (index, _module_of(cell), "".join(cell["source"]))
        for index, cell in enumerate(_code_cells(notebook))
        if _module_of(cell)
    ]
    for index, name, body in definition_cells:
        exec(compile(body, f"<cell:{index}:{name}>", "exec", flags=flags), namespace)

    defined = set(namespace) | set(dir(builtins))
    offenders: list[tuple[str, str, str]] = []

    for index, name, body in definition_cells:
        for scope, symbol in _unbound_reads(body, f"<cell:{name}>", defined):
            if (name, scope, symbol) not in ALLOWED_UNBOUND:
                offenders.append((name, scope, symbol))

    running = set(defined)
    for index, cell in enumerate(_code_cells(notebook)):
        if _module_of(cell):
            continue
        source = "".join(cell["source"])
        stripped = _strip_magics(source)
        label = f"cell[{index}]"
        # A cell's own bindings count before it is checked: ``find_parquet``
        # legitimately closes over ``WORK`` from the lines above it, and the grid
        # cell imports ``gc`` at its top. What stays forbidden is reading forward
        # into a cell that has not run yet, which is the ordering *Save & Run
        # All* fixes and a reader cannot see.
        running |= _assigned_names(stripped, label)
        for scope, symbol in _unbound_reads(stripped, label, running):
            offenders.append((label, scope, symbol))

    assert not offenders, (
        "these names are read but never bound, so the notebook raises NameError "
        f"when execution first reaches them: {sorted(offenders)}"
    )


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


#: Names a scaffolding cell may rebind even though a definition cell defines them.
#: Every entry is a value the notebook must resolve for *this* session rather than
#: the package default, and each is checked by eye rather than assumed.
ALLOWED_SHADOWS: frozenset[str] = frozenset({
    # train.py's repo-relative default; the notebook resolves /kaggle/working.
    "ARTIFACTS",
    # segments.py and train.py both define it; the notebook finds the real file.
    "DEFAULT_PARQUET",
    # The pinned digest. ``code_sha256()`` needs a ``__file__`` no cell has, so
    # the generator computes it from ``src/`` and the provenance cell binds it
    # (root §12, §15).
    "CODE_SHA256_OVERRIDE",
    # `D54f`: the 12 h wall runs from cell 0, so the grid is handed what is
    # *left* of the budget after the prelude rather than runner.py's constant.
    "SESSION_BUDGET_H",
})


def _imported_names(source: str) -> set[str]:
    """Names a cell binds with ``import``. Excluded from the shadowing check.

    Re-importing ``numpy as np`` binds the same module object the package cell
    bound, so it is a shadow by name and not by value — there is nothing for a
    later cell to read differently. What the check is for is an *assignment* that
    replaces a definition with something of another kind, which is what
    ``_digest, _provenance = ...`` did to ``report._provenance``.
    """
    out: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            out |= {a.asname or a.name.split(".")[0] for a in node.names}
    return out


def test_no_scaffolding_cell_shadows_a_package_name(notebook: dict, generator) -> None:
    """An evaluation cell may not rebind a name the package defines (`D65`).

    The mirror of :func:`test_flattening_rejects_imports_that_bind_a_module_object`.
    `D59` was a name the flattening *unbound*; this is a name an evaluation cell
    *rebinds*, and it cost the same thing: ``_digest, _provenance = ...`` in the
    save cell shadowed ``report._provenance`` — a function ``build_report`` calls
    one cell later — with a string, and the last cell of the 894-run session died
    with ``TypeError: 'str' object is not callable`` after 7.8 hours of grid.

    §15 already noted ``DEFAULT_PARQUET`` and ``HOUR_MS`` collide across modules
    and called them harmless *because the values are equal in both definitions*.
    That is a property of those two names, not a property of the format, and
    nothing was checking which kind any new collision was. This checks.
    """
    package_names: set[str] = set()
    for name in generator.MODULE_ORDER:
        for _, body in generator.split_module_cells(name):
            package_names |= _assigned_names(body, f"<cell:{name}>")

    offenders: list[tuple[int, str]] = []
    for index, cell in enumerate(_code_cells(notebook)):
        if _module_of(cell):
            continue
        source = _strip_magics("".join(cell["source"]))
        bound = _assigned_names(source, f"cell[{index}]") - _imported_names(source)
        shadowed = bound & package_names
        offenders += [(index, n) for n in sorted(shadowed - ALLOWED_SHADOWS)]

    assert not offenders, (
        "scaffolding cells rebind names the package defines, and in a flattened "
        f"notebook that is last-write-wins across the whole kernel: {offenders}"
    )


def test_every_cell_parses_under_kaggle_python(notebook: dict) -> None:
    """Every cell is syntax this notebook's declared interpreter can read.

    The notebook's own ``language_info`` says 3.11 and Kaggle's image supplies it;
    a developer's machine may be several minors ahead, and nothing else here would
    notice. Syntax accepted locally and rejected there is a failure that appears
    only after upload — the same shape as `D59` and the `_provenance` shadow
    (`D65`), and the same cost: hours of grid time paid before the error shows.

    ``feature_version`` bounds *syntax*, not runtime behaviour. It would not have
    caught the future-import hazard `D63` fixed, which is a semantic difference
    between 3.11 and 3.14 rather than a syntactic one. It is a floor, not a
    guarantee.
    """
    declared = notebook["metadata"]["language_info"]["version"]
    major, minor = (int(part) for part in declared.split(".")[:2])

    offenders: list[tuple[int, str]] = []
    for index, cell in enumerate(_code_cells(notebook)):
        source = _strip_magics("".join(cell["source"]))
        try:
            ast.parse(source, feature_version=(major, minor))
        except SyntaxError as exc:
            offenders.append((index, str(exc)))

    assert not offenders, (
        f"cells use syntax newer than the declared Python {declared}, so they "
        f"would fail on Kaggle and nowhere locally: {offenders}"
    )
