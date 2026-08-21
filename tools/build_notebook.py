"""Assemble ``notebooks/iTransformer.ipynb`` from ``src/itransformer_btc/``.

Root §15 says logic lives in the package and a notebook is a launcher. That rule
is unchanged. What changed is *how the notebook carries the package*: it used to
write twelve files with ``%%writefile`` and import them back, and now it
**defines them directly** — one cell per module, plain ``def``/``class``/constant
bodies that the cells below call by name. There is no ``itransformer_btc``
package on the running machine and nothing to import.

Why that became possible. The file-based form existed for exactly one reason
(`D54a`): the grid ran as two subprocesses, one pinned per GPU, and a subprocess
inherits none of the kernel's namespace, so it could reach the code only by
importing it from disk. Measured on Kaggle, the completed grid was **534 runs in
2.31 h at ~30 s per run** against a §10.3 estimate of 60–100 s and 10–20 h
(`D57`). One process running the grid in sequence is therefore ~4.5 h — inside
the 11 h session budget with room — so the subprocesses stopped buying anything
the budget needs, and the whole justification for materialising files fell with
them.

What the flattening does to each module, and nothing else:

* **intra-package imports are dropped.** Every module reaches its siblings
  through ``from itransformer_btc.x import y``; in one shared namespace those
  names are already bound, and the import would fail for want of a package.
  Removal is by :mod:`ast` node span rather than by line matching, which is what
  makes the parenthesised multi-line imports and the deferred ones nested inside
  function bodies come out right.
* **the ``if __name__ == "__main__":`` guard is dropped.** In a notebook cell
  ``__name__`` *is* ``"__main__"``, so ``runner.py``'s trailing guard would
  launch the entire grid the moment its definition cell ran. Verified, not
  assumed.

Everything else is verbatim — docstrings, comments, blank lines, third-party
imports. ``from __future__ import annotations`` stays where it sits: a leading
string literal is the compile unit's docstring, so the future import still counts
as the first statement and compiles. Verified too.

Two module-level names collide once the namespaces merge —
``DEFAULT_PARQUET`` (``segments``, ``train``) and ``HOUR_MS`` (``segments``,
``metrics``). Both are the **same value** in both definitions, differing only in
type annotation, so last-cell-wins is harmless. Anything else colliding would not
be, which is why :func:`flatten_module_source` compiles what it returns and the
sync tests re-derive the set.

**The notebook is generated, never hand-edited.** Two copies of 4,000 lines
drifting apart is a worse defect than the one this solves, so the copy inside the
notebook is written from ``src/`` by this script and
``tests/test_notebook_sync.py`` asserts the two agree under exactly the
transformation above. Edit ``src/``, re-run this, commit both.

Usage::

    python tools/build_notebook.py            # writes notebooks/iTransformer.ipynb
    python tools/build_notebook.py --check    # exit 1 if the notebook is stale
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import symtable
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "src" / "itransformer_btc"
NOTEBOOK = ROOT / "notebooks" / "iTransformer.ipynb"
PKG_NAME = "itransformer_btc"

#: **Execution order, and it is now load-bearing.** Under ``%%writefile`` these
#: cells only wrote bytes, so ordering was a courtesy to the reader and could not
#: break anything. Flattened, each cell is *executed*: decorators run, dataclass
#: field types resolve, module constants evaluate. A cell naming something a
#: later cell defines fails immediately rather than at call time.
#:
#: ``config.py`` therefore leads and ``__init__.py`` follows it instead of
#: leading as it did. Stripping ``__init__``'s imports leaves only a docstring
#: and ``__all__`` — harmless anywhere — but placing it after the module it
#: depends on keeps the cell honest about the dependency. It is kept rather than
#: dropped so every module inside ``code_sha256`` also appears in the notebook: a
#: digest naming a file the reader cannot see is worse than a redundant cell.
MODULE_ORDER: tuple[str, ...] = (
    "config.py",
    "__init__.py",
    "segments.py",
    "windows.py",
    "budget.py",
    "features.py",
    "splits.py",
    "model.py",
    "train.py",
    "keff.py",
    # Consumes the feature frame and root §4.5's stats boundary; nothing in the
    # package consumes it, so its position only has to follow `config`.
    "efficiency.py",
    "metrics.py",
    # After `metrics`, which it imports `assert_same_windows` from, and before
    # `runner`, which imports its three configs by name (`D56`).
    "baselines.py",
    # After `metrics`, whose hln_test, load_predictions and parse_run_id it
    # imports by name. Nothing in the package imports it back.
    "comparisons.py",
    # Reads metrics' non_overlapping_mask and the two normal helpers by name.
    "economics.py",
    # After `model` and `splits`, whose ITransformer and SplitTensors it reads,
    # and before `runner`, which imports tercile_maps by name for the attention
    # arm (`D62d`).
    "attention.py",
    "runner.py",
)

#: Header comment opening each module cell. ``tests/test_notebook_sync.py``
#: locates module cells by this, so its pattern must track this format. With the
#: ``%%writefile`` line gone there is otherwise nothing naming the module a
#: reader is looking at.
MODULE_BANNER = "# ═══ {name} " + "═" * 8

#: Definitions dropped from the flattened cells because they **cannot work
#: there**, per module.
#:
#: Both entries are the subprocess path. ``launch_workers`` spawns
#: ``python -m itransformer_btc.runner``, and ``_main`` is the CLI on the other
#: end of that pipe — reachable only through the ``if __name__ == "__main__":``
#: guard this generator already removes. In a flattened notebook there is no
#: ``itransformer_btc`` module for the child to import, so a call would fail with
#: ``No module named itransformer_btc`` after forking two processes: a confusing
#: failure, hours in, for a function the notebook has no reason to call.
#:
#: They stay in ``src/`` untouched, where ``python -m itransformer_btc.runner``
#: is a real and supported entry point. Dropping them here is a statement about
#: this launcher, not about the package — and it is declared rather than silent
#: because a reader comparing the cell to the file must be able to see why the
#: two differ.
FLATTEN_DROP_FUNCTIONS: dict[str, tuple[str, ...]] = {
    "runner.py": ("launch_workers", "_main"),
}


# -- flattening --------------------------------------------------------------


def _is_main_guard(node: ast.stmt) -> bool:
    """``if __name__ == "__main__":`` at module level, however it is spelled."""
    if not isinstance(node, ast.If):
        return False
    test = node.test
    return (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "__name__"
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.Eq)
    )


def _intra_package_import(node: ast.AST) -> bool:
    if isinstance(node, ast.ImportFrom):
        # ``node.level > 0`` is the relative form, ``from . import x``, whose
        # ``module`` is None. The package writes absolute imports throughout, but
        # a relative one left in place would not merely be untidy: there is no
        # parent package in a flattened cell, so it raises ImportError on the
        # first run rather than being quietly equivalent.
        return node.level > 0 or (node.module or "").split(".")[0] == PKG_NAME
    if isinstance(node, ast.Import):
        return any(a.name.split(".")[0] == PKG_NAME for a in node.names)
    return False


def _module_object_bindings(node: ast.AST) -> list[str]:
    """Names a dropped import would have bound to a **module object**.

    The distinction decides whether flattening is lossless.
    ``from itransformer_btc.metrics import clark_west_test`` binds a *function*,
    and that function is defined by another cell, so deleting the import costs
    nothing. ``from itransformer_btc import metrics`` binds the *module*, and no
    cell defines any module object at all — so ``metrics.clark_west_test`` is
    left dangling and raises NameError the first time that line is reached.

    Which is exactly what happened (`D59`): the reference sat inside
    ``stage5_pilot``, six minutes into a Kaggle session, past the data stage, the
    K_eff stage and twelve training runs. Every check the repository had passed,
    because each one asked a question this defect does not answer to — the cell
    parses, it compiles, it matches ``src/`` byte for byte, and it names no
    surviving ``itransformer_btc``.
    """
    bound: list[str] = []
    if isinstance(node, ast.ImportFrom):
        if node.level > 0 or (node.module or "") == PKG_NAME:
            for alias in node.names:
                if (PACKAGE / f"{alias.name}.py").exists():
                    bound.append(alias.asname or alias.name)
    elif isinstance(node, ast.Import):
        for alias in node.names:
            if alias.name.split(".")[0] == PKG_NAME:
                bound.append(alias.asname or alias.name.split(".")[0])
    return bound


def unbound_global_reads(source: str, label: str, names: set[str]) -> set[str]:
    """Which of ``names`` the source *reads* as a global it never binds.

    :mod:`symtable` rather than an ``ast.Name`` walk because the question is
    about scope, not spelling: a local variable that happens to be called
    ``metrics`` is a Load node too, and would make a name-matching check cry wolf
    on correct code. The symbol table knows which scope each name belongs to,
    handles comprehensions, class bodies and ``global``/``nonlocal`` correctly,
    and is in the standard library.
    """
    hits: set[str] = set()
    stack = [symtable.symtable(source, label, "exec")]
    while stack:
        table = stack.pop()
        stack.extend(table.get_children())
        for sym in table.get_symbols():
            if (
                sym.get_name() in names
                and sym.is_referenced()
                and sym.is_global()
                # ``is_assigned()`` alone would miss a rebinding by import, which
                # sets a different flag.
                and not (sym.is_assigned() or sym.is_imported())
            ):
                hits.add(sym.get_name())
    return hits


def _executable_source(source: str) -> str:
    """Source with docstrings stripped — what actually runs.

    The modules legitimately *mention* ``itransformer_btc`` in prose: ``:mod:``
    cross-references, ``Importers:`` notes, the ``D54`` discussion in
    ``runner``. Only executable references can break a machine with no such
    package, and going through the AST is how to tell the two apart without a
    comment regex that would be wrong on the first docstring containing a colon.
    """
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ) and ast.get_docstring(node):
            node.body = node.body[1:] or [ast.Pass()]
    return ast.unparse(ast.fix_missing_locations(tree))


def flatten_module_source(name: str) -> str:
    """The module verbatim, minus intra-package imports and the ``__main__`` guard.

    Spans come from the AST rather than from line matching, which is what makes
    the parenthesised multi-line imports and the deferred imports inside function
    bodies come out right — ``ast.walk`` reaches the nested ones, and the
    module-level scan handles the guard.

    The result is compiled before it is returned, and then re-parsed to confirm no
    executable reference to the package survived. Deleting an import that was the
    only statement in its block would leave a syntax error, and a notebook is the
    worst place to discover that: twelve hours into a session, in a cell nobody
    re-reads.
    """
    text = (PACKAGE / name).read_text(encoding="utf-8")
    tree = ast.parse(text, filename=name)

    spans: list[tuple[int, int]] = [
        (node.lineno, node.end_lineno or node.lineno)
        for node in ast.walk(tree)
        if _intra_package_import(node)
    ]
    module_objects = {
        binding
        for node in ast.walk(tree)
        if _intra_package_import(node)
        for binding in _module_object_bindings(node)
    }
    spans += [
        (node.lineno, node.end_lineno or node.lineno)
        for node in tree.body
        if _is_main_guard(node)
    ]

    unusable = FLATTEN_DROP_FUNCTIONS.get(name, ())
    found = {
        node.name: (node.lineno, node.end_lineno or node.lineno)
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in unusable
    }
    absent = sorted(set(unusable) - found.keys())
    assert not absent, (
        f"{name}: FLATTEN_DROP_FUNCTIONS names {absent}, which no longer exist. "
        f"A stale entry silently drops nothing, so the notebook would ship the "
        f"subprocess path again — review why the function moved before editing "
        f"the list."
    )
    # A dropped function's own decorator lines sit above node.lineno in older
    # Python; take the earliest so nothing is orphaned.
    for fn_name, (lo, hi) in found.items():
        node = next(n for n in tree.body
                    if getattr(n, "name", None) == fn_name)
        starts = [d.lineno for d in getattr(node, "decorator_list", [])] + [lo]
        spans.append((min(starts), hi))

    dropped = {n for lo, hi in spans for n in range(lo, hi + 1)}
    source = "".join(
        line
        for number, line in enumerate(text.splitlines(keepends=True), start=1)
        if number not in dropped
    )

    compile(source, f"<cell:{name}>", "exec")
    assert PKG_NAME not in _executable_source(source), (
        f"{name}: an executable reference to {PKG_NAME} survived flattening, so "
        f"the cell would fail on a machine with no such package installed"
    )
    dangling = unbound_global_reads(source, f"<cell:{name}>", module_objects)
    assert not dangling, (
        f"{name}: {sorted(dangling)} is still read after its import was dropped, "
        f"and it named a module rather than a definition — nothing in the merged "
        f"namespace will bind it, so the cell raises NameError when that line is "
        f"first reached and not before. Import the names instead: "
        f"`from {PKG_NAME}.{sorted(dangling)[0]} import <name>`, then call "
        f"`<name>(...)` rather than `{sorted(dangling)[0]}.<name>(...)` (`D59`)."
    )
    return source


def package_digest() -> str:
    """Byte-for-byte the digest :func:`itransformer_btc.train.code_sha256` returns.

    Computed from ``src/`` and deliberately not from the flattened cells: §12 asks
    a run to name *the code that produced it*, and the answer has to be the same
    number whether that code ran from a checkout or from this notebook. Drift
    between the two implementations would surface as a phantom change of code
    vintage — a false positive on the one check §12 exists to make possible.
    """
    digest = hashlib.sha256()
    for path in sorted(PACKAGE.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


# -- prose -------------------------------------------------------------------

MD_TITLE = """<div style="background: linear-gradient(135deg, #0b1021, #14213d, #1b2a4a); border-radius: 16px; padding: 36px 40px; margin-bottom: 8px;">
  <h1 style="color: #8ecae6; font-size: 2.3em; font-weight: 800; margin: 0 0 10px 0; letter-spacing: 0.5px;">
    &#9889; iTransformer &middot; Walk-Forward BTCUSDT 1h
  </h1>
  <p style="color: #ffb703; font-size: 1.12em; margin: 0 0 18px 0; font-weight: 500;">
    Nominal Variates or Effective Dimensionality? &mdash; 15 origins &middot; 684 runs &middot; 2 &times; T4
  </p>
  <hr style="border: none; border-top: 1px solid #2a4365; margin: 16px 0;">
  <p style="color: #a8c0dd; font-size: 0.97em; margin: 0 0 12px 0;">
    <strong>Self-contained.</strong> This notebook needs exactly two things: itself, and
    <code>BTCUSDT_1h.parquet</code> attached as a Kaggle Dataset. Every definition it uses is
    <em>in</em> it &mdash; the cells below are ordinary <code>def</code>, <code>class</code> and
    constant bodies, run top to bottom. Nothing is written to disk to be imported back, there is
    no <code>itransformer_btc</code> package on this machine, and no <code>src/</code> on
    <code>sys.path</code>.
  </p>
  <p style="color: #a8c0dd; font-size: 0.97em; margin: 0 0 12px 0;">
    <strong>It is still a launcher, not a program.</strong> Every definition &mdash; the twelve
    variates, the segment law, the window semantics, the scaler, the model, the metrics &mdash; is
    authored in <code>src/itransformer_btc/</code> and unit-tested on CPU; the cells below are a
    transcription, not an authoring surface. <strong>Do not hand-edit the <em>Definitions</em>
    cells:</strong> they are generated by <code>tools/build_notebook.py</code>,
    <code>tests/test_notebook_sync.py</code> fails the moment they diverge from the package, and
    the next generator run reverts the edit.
  </p>
  <p style="color: #7f9cc0; font-size: 0.93em; margin: 0;">
    Answers <strong>RQ1</strong> (does benefit track K or K<sub>eff</sub>?),
    <strong>RQ2</strong> (does the multivariate gap narrow with model age?),
    <strong>RQ3</strong> (what retraining cadence?) &mdash; all three pre-registered before any
    model ran, and none of them changeable now without declaring a new experiment.
  </p>
</div>"""

MD_SETUP = """<div style="background: linear-gradient(90deg, #0b1021, #112240); border-left: 4px solid #8ecae6; border-radius: 8px; padding: 18px 24px;">
  <h2 style="color: #8ecae6; margin: 0 0 8px 0;">&#128295; 0 &middot; Setup</h2>
  <p style="color: #b8c7e0; margin: 0;">Find the immutable artifact by globbing, never by dataset slug. Install only what the Kaggle image lacks.</p>
  <p style="color: #7f9cc0; margin: 12px 0 0 0; font-size: 0.9em;">Kaggle ships its own torch and
    pyarrow; pinning them against a local venv is forbidden. <code>/kaggle/input</code> is read-only,
    everything is written to <code>/kaggle/working</code>. The parquet is <strong>not</strong>
    re-downloaded here even though Stage 1 could: a fresh download is a new vintage, and &sect;12
    forbids numbers from two vintages sharing a table.</p>
</div>"""

MD_DEFINE = """<div style="background: linear-gradient(90deg, #0a1a12, #0f2a1c); border-left: 4px solid #52b788; border-radius: 8px; padding: 18px 24px;">
  <h2 style="color: #52b788; margin: 0 0 8px 0;">&#128230; 0b &middot; Definitions</h2>
  <p style="color: #b8c7e0; margin: 0;">Thirteen cells, one per module, in dependency order. Plain definitions &mdash; run them and every name the rest of the notebook calls exists. Generated from <code>src/</code>; do not hand-edit.</p>
  <ul style="color: #b7e4c7; margin: 10px 0 0 0; padding-left: 20px; font-size: 0.92em;">
    <li><strong>Definitions, not files.</strong> The earlier launcher wrote these modules to
    disk with <code>%%writefile</code> and imported them back, because the grid ran as two
    subprocesses and a subprocess inherits none of this kernel's namespace. Measured on Kaggle, the
    completed grid was <strong>534 runs in 2.31 h at ~30 s per run</strong> against a &sect;10.3
    estimate of 60&ndash;100 s and 10&ndash;20 h (<code>D57</code>), so one process running the grid
    in sequence is ~4.5 h &mdash; inside the 11 h budget with room. The subprocesses stopped paying
    for themselves, and the files existed only to feed them.</li>
    <li><strong>Order is load-bearing now.</strong> These cells are <em>executed</em>, not merely
    written: decorators run, dataclass field types resolve, module constants evaluate. A cell naming
    something a later cell defines fails at once rather than at call time. <em>Save Version &rarr;
    Save &amp; Run All</em> runs them in order, which is the only order that works.</li>
    <li><strong>Two edits, and only two.</strong> Intra-package imports are removed &mdash; the names
    are already in this namespace, and the import would fail for want of a package. And
    <code>runner</code>'s <code>if __name__ == "__main__":</code> guard is removed: in a notebook
    cell <code>__name__</code> <em>is</em> <code>"__main__"</code>, so it would launch the entire
    grid the instant its definition cell ran. Everything else is the package, character for
    character.</li>
    <li><strong>The digest is the provenance.</strong> There is no git repository on Kaggle and now
    no package files either, so the cell after these pins <code>code_sha256</code> to the digest of
    <code>src/itransformer_btc/</code> taken at generation time &mdash; the same number a local
    checkout of the same source reports (<code>D54b</code>), so a notebook run and a repository run
    do not look like different code vintages.</li>
  </ul>
</div>"""

MD_DATA = """<div style="background: linear-gradient(90deg, #1a1200, #2b1d00); border-left: 4px solid #ffb703; border-radius: 8px; padding: 18px 24px;">
  <h2 style="color: #ffb703; margin: 0 0 8px 0;">&#128202; 1 &middot; Data &amp; integrity &mdash; Stage 2</h2>
  <p style="color: #b8c7e0; margin: 0;">Load the immutable artifact and assert the window budget per origin, by exact equality.</p>
  <ul style="color: #e0c68a; margin: 10px 0 0 0; padding-left: 20px; font-size: 0.92em;">
    <li><strong>No imputation anywhere.</strong> When the exchange is down no price forms, so
    imputation is <em>undefined</em>, not merely risky &mdash; Rubin's taxonomy applies to values
    that exist but went unobserved.</li>
    <li><strong>Per origin, exact equality</strong> (<code>D45</code>). Asserted against the pooled
    4.9% it fires spuriously at fourteen of fifteen origins, gets loosened until it passes, and then
    can no longer distinguish positional drift from ordinary between-origin variation.</li>
    <li>Test blocks hold <strong>720</strong> forecast origins, not 601 (<code>D51b</code>): a test
    window's lookback may cross backwards, a training window's target may not cross forwards.</li>
    </ul>
</div>"""

MD_VARIATES = """<div style="background: linear-gradient(90deg, #001a1a, #002b2b); border-left: 4px solid #48cae4; border-radius: 8px; padding: 18px 24px;">
  <h2 style="color: #48cae4; margin: 0 0 8px 0;">&#129514; 2 &middot; The twelve variates</h2>
  <p style="color: #b8c7e0; margin: 0;">Per-bar functions of the current bar, except r, which uses the current and previous close. No rolling window anywhere.</p>
  <p style="color: #a5e8f0; margin: 10px 0 0 0; font-size: 0.92em;">That is a structural safety
    property, not a style choice: with no rolling window in the pipeline, the
    <code>center=True</code> leak class is <em>unrepresentable</em>. Column order is ladder order, so
    rung K is exactly the first K columns and <code>r</code> is channel 0 at every rung &mdash; which
    makes the single-channel loss one constant rather than a lookup.</p>
</div>"""

MD_KEFF = """<div style="background: linear-gradient(90deg, #150029, #22003d); border-left: 4px solid #c77dff; border-radius: 8px; padding: 18px 24px;">
  <h2 style="color: #c77dff; margin: 0 0 8px 0;">&#128209; 3 &middot; Effective dimensionality &mdash; Stage 3b</h2>
  <p style="color: #b8c7e0; margin: 0;">K_eff is RQ1's independent variable and it is measured before a single epoch runs.</p>
  <ul style="color: #d8b4fe; margin: 10px 0 0 0; padding-left: 20px; font-size: 0.92em;">
    <li><strong>Per origin, on that origin's own 21-month training sub-block</strong>
    (<code>D44</code>). A full-sample PR would be estimated on the same data as the outcome, making
    RQ1 partly circular &mdash; the one leakage path that survived every checklist item, because
    &sect;11 audits only the gate.</li>
    <li><strong>The gate reads the pre-first-origin span alone</strong> (<code>D02</code>), trigger
    pre-registered at PR &lt; 5.0. Its action is <em>disclosure, not a re-cut</em>
    (<code>D48</code>): <code>D01</code> leaves no second consistent cut over F1&ndash;F5, so
    "re-cut the ladder" named no reachable alternative.</li>
    <li>Reported on <strong>window-normalised</strong> features too (<code>D04</code>) &mdash;
    <code>use_norm</code> strips volatility <em>level</em>, so the 8&rarr;12 rung can flatten for a
    reason that has nothing to do with redundancy.</li>
    </ul>
</div>"""

MD_INVARIANTS = """<div style="background: linear-gradient(90deg, #002200, #003300); border-left: 4px solid #7ae582; border-radius: 8px; padding: 18px 24px;">
  <h2 style="color: #7ae582; margin: 0 0 8px 0;">&#128295; 4 &middot; Pre-flight invariants &mdash; Stage 4</h2>
  <p style="color: #b8c7e0; margin: 0;">Three checks that must pass before the grid. Each failed the first time it ran.</p>
  <ul style="color: #b7e4c7; margin: 10px 0 0 0; padding-left: 20px; font-size: 0.92em;">
    <li><code>MSE(c&middot;x)/c&sup2; == MSE(x)</code>, <strong>not</strong>
    <code>MSE(c&middot;x) == MSE(x)</code> (<code>D03</code>) &mdash; the target is a channel of the
    same array, so it scales too and the loss scales by c&sup2;. The source specification's version
    cannot pass.</li>
    <li>Single-batch overfit with <strong><code>dropout=0.0</code></strong> (<code>D52d</code>). With
    the configured 0.1 still on, the loss floors near 7e-2 and a reader following the instruction
    literally concludes the plumbing is broken when it is not.</li>
    <li>The <strong>Naive-RW baseline is computed first</strong>, before any model trains, and it is
    <code>&#375;<sub>z</sub> = &minus;&mu;<sub>g</sub>/&sigma;<sub>g</sub></code> &mdash; never 0
    (<code>D31</code>), which would silently be a constant-drift model wearing the EMH baseline's
    name.</li>
    </ul>
</div>"""

MD_GATE = """<div style="background: linear-gradient(90deg, #2b0a00, #3d1000); border-left: 4px solid #ff7b54; border-radius: 8px; padding: 18px 24px;">
  <h2 style="color: #ff7b54; margin: 0 0 8px 0;">&#128737; 5 &middot; Stage 5 gate &mdash; validation only</h2>
  <p style="color: #b8c7e0; margin: 0;">Origin 1, 4 K x 3 seeds, scored on the validation sub-block. The test blocks stay shut.</p>
  <p style="color: #ffc4a3; margin: 10px 0 0 0; font-size: 0.92em;">
    &sect;11 requires the test blocks be opened once, after the design is frozen, so a gate that
    repositions the title on a test-block result cannot coexist with it (<code>D27</code>). The
    statistic is <strong>Clark&ndash;West, not DM</strong> (<code>D29</code>): K=1's feature set is a
    strict subset of K=8's under the same architecture and sample, and standard DM is systematically
    undersized against exactly the alternative being tested. The gate is
    <strong>K=1 vs K=8, never K=12</strong> &mdash; K=12 is built to be redundant, and gating on it
    would kill a viable paper for the wrong reason. The twelve cells are ordinary main-grid
    <code>run_id</code>s, so the grid below skips them.</p>
</div>"""

MD_GRID = """<div style="background: linear-gradient(90deg, #001233, #001845); border-left: 4px solid #4cc9f0; border-radius: 8px; padding: 18px 24px;">
  <h2 style="color: #4cc9f0; margin: 0 0 8px 0;">&#128640; 6 &middot; The grid</h2>
  <p style="color: #b8c7e0; margin: 0;">684 unique runs across seven arms, executed in this kernel. Resume automatic, budget guard at run boundaries.</p>
  <ul style="color: #a9d6f5; margin: 10px 0 0 0; padding-left: 20px; font-size: 0.92em;">
    <li><strong>main 300</strong> &middot; <strong>uniform 75</strong> (<code>D50</code>) &middot;
    <strong>fresh 15</strong> (falsification) &middot; <strong>horizon 144</strong>. The sweep's
    H=24 slice shares 48 <code>run_id</code>s with the main grid, so 582 nominal cells are 534 real
    runs &mdash; executing one twice would mean two files racing for one path.</li>
    <li><strong>ridge 60</strong> (<code>D17</code>, K=1/4/8/12) &middot;
    <strong>dlinear 45</strong> &middot; <strong>patchtst 45</strong> &mdash; the &sect;7 comparators,
    absent from every earlier manifest (<code>D56</code>). Without them "iTransformer has no edge"
    rests on Naive-RW alone, and a referee reads the null as an untuned configuration rather than a
    finding. They run <em>after</em> the ladder so a short session leaves RQ1&ndash;RQ3's inputs
    complete, and so each baseline's <code>D45</code> window-alignment assertion finds its comparator
    already on disk.</li>
    <li><strong>One process, one GPU, and a second T4 idles.</strong> A real cost, stated rather than
    buried: it roughly doubles wall time against the two-worker form. It is what the notebook format
    costs &mdash; definitions live in this namespace and a subprocess inherits none of it. The
    measured arithmetic says it still fits: <strong>534 &times; ~30 s &asymp; 4.5 h</strong> for the
    ladder (<code>D57</code>).</li>
    <li><strong>PatchTST is the expensive arm, by a factor nobody guessed.</strong> Measured on CPU at
    origin 1, K=8: iTransformer 113 s over 10 epochs, ridge 0.5 s, DLinear 24 s, PatchTST
    <strong>1810 s</strong>. Per epoch that is 5.3&times; iTransformer &mdash; it folds channels into
    the batch, so a step processes B&times;N=256 sequences rather than 32 &mdash; and both
    channel-independent baselines run the full 30 epochs because early stopping never fires. Scaling
    <code>D57</code>'s 30 s/run gives ridge and DLinear ~10 min combined and PatchTST
    <strong>~6 h</strong>, putting the whole manifest near 11 h. <strong>Two sessions is therefore the
    expected case, not the exception.</strong> That is survivable precisely because the baselines run
    last: an overrun costs comparators, never RQ1&ndash;RQ3's inputs, and they resume by
    <code>run_id</code> like anything else.
    Threads are <em>not</em> the way to reclaim the second GPU:
    <code>torch.manual_seed</code> seeds <em>every</em> CUDA device, so two threads would clobber
    each other's generator mid-run and &sect;12's reproducibility contract would be
    unenforceable.</li>
    <li><strong>No <code>DataLoader</code>.</strong> At ~280k parameters the run is dominated by data
    movement and Python overhead, which a per-item loader maximises &mdash; roughly 10&times; worse,
    which puts the grid outside the 30 h weekly quota outright.</li>
    <li>Run <em>Save Version &rarr; Save &amp; Run All</em>, never the editor: the 20-minute idle
    timeout kills interactive sessions, and hitting the 12 h wall interactively loses
    <code>/kaggle/working</code> entirely.</li>
    <li><strong>Resume granularity is one run, ~30 s.</strong> A run counts as complete only when
    both <code>preds/</code> and <code>meta/</code> exist and <code>meta.status ==
    "complete"</code>, so a session cut short at run 200 of 684 loses at most the one run in flight.
    The next session subtracts what is done and continues &mdash; there is no bookkeeping to do by
    hand and no state beyond the files themselves. Intra-run checkpointing is deliberately omitted:
    at ~30 s per run it costs more complexity than it saves.</li>
    <li><strong>The budget guard bounds the session, not the grid call.</strong> Kaggle's 12 h wall
    starts at cell 0, so the prelude &mdash; data, K<sub>eff</sub>, invariants, the twelve pilot
    runs &mdash; is subtracted before the guard is built. Counting from the grid's own start would
    let the two clocks drift apart by exactly however long the prelude took.</li>
  </ul>
</div>"""

MD_EVAL = """<div style="background: linear-gradient(90deg, #2d0036, #4a0060); border-left: 4px solid #e0aaff; border-radius: 8px; padding: 18px 24px;">
  <h2 style="color: #e0aaff; margin: 0 0 8px 0;">&#128200; 7 &middot; Evaluation &mdash; RQ1, RQ2, RQ3</h2>
  <p style="color: #b8c7e0; margin: 0;">Every number below resolves to a persisted prediction file and a config hash, or it does not enter the manuscript.</p>
  <p style="color: #d8b4fe; margin: 10px 0 0 0; font-size: 0.92em;">Ratio metrics are formed
    from <strong>seed-averaged MSEs</strong>, never from an average of per-seed ratios
    (<code>D42</code>): the two differ by Jensen, and the second would require pairing seed 42 at K=1
    with seed 42 at K=8 &mdash; independent training runs of different models, where any of 5!
    orderings gives a different answer.</p>
</div>"""

MD_SAVE = """<div style="background: linear-gradient(90deg, #150029, #240046); border-left: 4px solid #bf5af2; border-radius: 8px; padding: 18px 24px;">
  <h2 style="color: #bf5af2; margin: 0 0 8px 0;">&#128190; 8 &middot; Save</h2>
  <p style="color: #b8c7e0; margin: 0;">Every table and figure is generated FROM paper_numbers.json, never transcribed.</p>
  <p style="color: #c77dff; margin: 10px 0 0 0; font-size: 0.92em;">Numbers produced under
    different input-artifact hashes are not comparable and must not share a table, so the parquet
    digest travels with them &mdash; and so does <code>code_sha256</code>, which is what identifies
    the code off-repo. A number that cannot be regenerated is a documented failure, not a
    footnote.</p>
</div>"""


# -- code cells --------------------------------------------------------------

CODE_SETUP = r'''import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Kaggle's 12 h wall runs from HERE, not from the moment the grid starts. The
# budget guard counts from whatever it is handed, so the prelude — data, K_eff,
# invariants, the twelve pilot runs — would sit outside the budget entirely and
# the two clocks would drift apart by however long it took. The grid cell
# subtracts this. Losing /kaggle/working to the wall costs the whole session's
# runs, so the margin is not somewhere to be approximate.
SESSION_T0 = time.perf_counter()

ON_KAGGLE = Path("/kaggle/working").exists()
WORK = (Path("/kaggle/working") if ON_KAGGLE else Path.cwd()).resolve()
ARTIFACTS = WORK / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

# Relative paths inside the definitions below resolve against the process working
# directory, so artifacts have to land beside it whichever machine this is.
# Kaggle already starts in /kaggle/working; a local run started from notebooks/
# does not. Nothing is added to sys.path — there is no package to import, and an
# entry there could only serve to shadow these cells with someone else's copy.
if Path.cwd().resolve() != WORK:
    os.chdir(WORK)


def ensure(module: str, pip_name: str | None = None) -> None:
    """Install only what is genuinely missing.

    Kaggle ships torch, numpy, pyarrow and usually polars. Pinning any of them
    against a local venv is forbidden by root section 16 — the image wins.
    """
    try:
        __import__(module)
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", pip_name or module]
        )


for _mod in ("polars", "pyarrow", "numpy", "torch"):
    ensure(_mod)


def find_parquet() -> Path:
    """Locate BTCUSDT_1h.parquet by globbing — never by Kaggle dataset slug.

    Root section 10.5: discovery is by glob so the Dataset can be renamed without
    editing anything. Both upload shapes are covered — the four Stage 1 files
    uploaded flat, and the whole repository uploaded with data/raw/ inside it.
    """
    patterns = (
        "data/raw/BTCUSDT_1h.parquet",
        "*/data/raw/BTCUSDT_1h.parquet",
        "BTCUSDT_1h.parquet",
        "*/BTCUSDT_1h.parquet",
        "*/*/BTCUSDT_1h.parquet",
    )
    roots = [WORK, Path("/kaggle/input")] if ON_KAGGLE else [WORK, WORK.parent]
    for root in roots:
        if not root.exists():
            continue
        for pattern in patterns:
            for hit in sorted(root.glob(pattern)):
                return hit.resolve()
    raise FileNotFoundError(
        "BTCUSDT_1h.parquet not found under "
        f"{[str(r) for r in roots]}. Attach data/raw/ as a Kaggle Dataset. "
        "It is NOT re-downloaded here on purpose: a fresh download is a new "
        "vintage, and section 12 forbids numbers from two vintages sharing a table."
    )


PARQUET = find_parquet()
# Every meta/*.json records the digest of the artifact its run consumed (section
# 12), and can only find it if told.
os.environ["ITBTC_PARQUET"] = str(PARQUET)

import numpy as np
import polars as pl
import torch

print(f"work      {WORK}")
print(f"parquet   {PARQUET}  ({PARQUET.stat().st_size / 1e6:.1f} MB)")
print(f"artifacts {ARTIFACTS}")
print(f"polars {pl.__version__} | torch {torch.__version__} | numpy {np.__version__}")
print(f"CUDA devices: {torch.cuda.device_count()}")
for _i in range(torch.cuda.device_count()):
    _cap = torch.cuda.get_device_capability(_i)
    print(f"  cuda:{_i}  {torch.cuda.get_device_name(_i)}  sm_{_cap[0]}{_cap[1]}")

# Root section 10.3: never gate precision on torch.cuda.is_bf16_supported(). It
# defaults to including_emulation=True and returns True on a T4 (sm_75),
# selecting an emulated bf16 path *slower than fp32*. Gate on capability.
if torch.cuda.is_available():
    print(f"native bf16: {torch.cuda.get_device_capability(0)[0] >= 8}  "
          f"(is_bf16_supported() says {torch.cuda.is_bf16_supported()} "
          f"and is not to be trusted here)")
'''

#: ``{digest}`` is substituted at generation time — see :func:`package_digest`.
CODE_PROVENANCE = r'''# Root section 12 asks a run to name the code that produced it, and names the git
# sha as the way. There is no git repository on Kaggle and — the cells above
# being definitions rather than files — nothing on disk to hash either. So
# the digest is taken from src/itransformer_btc/ when this notebook is generated
# and pinned here (D54b). It is the SAME number a local checkout of the same
# source reports, which is the point: a run from the notebook and a run from the
# repository must not look like different code vintages.
CODE_SHA256_OVERRIDE = "{digest}"

# These cells are EXECUTED, not written, so a skipped one leaves a hole rather
# than a stale file — and the hole surfaces hours later, inside the grid. One
# sentinel per module, in MODULE_ORDER: cheap here, unbounded there.
_sentinels = (
    "ORIGINS", "__all__", "build_segments", "count_windows", "budget_table",
    "build_features", "build_origin_tensors", "ITransformer", "code_sha256",
    "keff_table", "seed_average", "PatchTST", "manifest",
)
_missing = [name for name in _sentinels if name not in globals()]
assert not _missing, (
    f"{len(_missing)} of {len(_sentinels)} definition cells have not run: "
    f"{_missing}. Run the Definitions cells above in order, top to bottom."
)
assert "itransformer_btc" not in sys.modules, (
    "an installed or on-path itransformer_btc package was imported. This notebook "
    "must run its OWN definitions, or every number it produces is traceable to "
    "code that is not in the cells above — exactly the dependency this format "
    "exists to remove."
)

print(f"modules defined in-kernel: {len(MODULE_NAMES)}  {MODULE_NAMES}")
print(f"code_sha256 {code_sha256()}")
print("\nThat digest goes into every meta/*.json. There is no git repository on "
      "Kaggle, so it is what the traceability contract has to name the code with "
      "— and it is the better half of the pair anyway: it identifies the code "
      "that ran, not the commit someone was standing on with a dirty tree.")
'''

CODE_DATA = r'''bars = usable_mask(load_bars(PARQUET))
print(f"bars {bars.height:,}  usable {int(bars['usable'].sum()):,}  "
      f"unusable {int((~bars['usable']).sum())}")

# D51c: the same 3 bars are zero-volume, zero-trade and H == L. No volume means
# no trades, and no trades means high and low never separate.
print(bars.filter(~pl.col("usable")).select(
    ["open_time", "zero_volume", "flat_bar", "zero_trades"]))

budgets = budget_table(bars)
drift = [
    (b.label, b.summary.break_runs, b.summary.excluded_positions, b.windows_measured,
     COMMITTED_TRAIN_BUDGET[b.label])
    for b in budgets
    if (b.summary.break_runs, b.summary.excluded_positions, b.windows_measured)
    != COMMITTED_TRAIN_BUDGET[b.label]
]
assert not drift, f"budget drift against the committed table: {drift}"
print(f"\nwindow budget matches docs/ORIGIN_WINDOW_BUDGET.md at all "
      f"{len(budgets)} origins (exact equality, D45)")

coverage = pl.DataFrame([
    {"origin": b.label, "train_windows": b.windows_measured,
     "loss_pct": round(b.loss_pct, 2), "closed_form_agrees": b.closed_form_agrees,
     **{f"B{i}": n for i, n in enumerate(b.test_block_starts, start=1)}}
    for b in budgets
])
print(coverage)
print(f"\ntraining range {coverage['train_windows'].min():,} … "
      f"{coverage['train_windows'].max():,} windows (raw-bar frame)")
print("The closed form disagrees wherever a segment is shorter than one window "
      "(D51a) and is kept only as an upper bound.")
'''

CODE_FEATURES = r'''features = build_features(bars)
print(f"feature frame {features.height:,} rows x {len(VARIATE_ORDER)} variates")
print(f"dropped {int(bars['usable'].sum()) - features.height} rows = one per segment "
      f"(D52c: r is per segment, so each segment's first bar has no predecessor)")

for k in (1, 4, 8, 12):
    print(f"  K={k:>2}: {ladder_columns(k)}")

print(features.select(VARIATE_ORDER).describe().filter(
    pl.col("statistic").is_in(["mean", "std", "min", "max"])))

# D52a: Rogers-Satchell is NOT strictly positive — it vanishes on a shadowless
# (marubozu) bar, of which 33 exist. log(RS + 1e-9) puts log kappa = -20.7 inside
# the measured support rather than leaving 33 out-of-support spikes that would
# distort the instance normalisation of every window containing one.
rs = features["log_rogers_satchell"]
print(f"\nlog_rogers_satchell: min {rs.min():.3f}  q0.1% {rs.quantile(0.001):.3f}  "
      f"median {rs.median():.3f}  at-floor {int((rs <= np.log(1e-9) + 1e-9).sum())}")
assert features.select([pl.col(c).is_finite().all() for c in VARIATE_ORDER]).row(0) \
    == tuple([True] * 12), "a variate is non-finite; the segment law did not run"
'''

CODE_KEFF = r'''gate = gate_pr(features, k=8)
print(gate_verdict(gate))

t0 = time.perf_counter()
keff_tbl = keff_table(features)          # 15 origins x 4 rungs, training spans only
print(f"\nmeasured in {time.perf_counter() - t0:.0f}s")

rung_view = (
    keff_tbl.group_by("k")
    .agg(
        pl.col("pr_raw").mean().alias("PR_raw"),
        pl.col("pr_raw").std().alias("PR_raw_sd"),
        pl.col("pr_window_norm").mean().alias("PR_windownorm"),
        pl.col("stable_rank_lookback").mean().alias("stable_rank"),
        pl.col("pr_lookback_ratio").mean().alias("crosslag_share"),
        pl.col("divergence").mean().alias("divergence"),
    )
    .sort("k")
)
print(rung_view)
print(f"\ncorr(K, K_eff) = {corr_k_keff(keff_tbl):.4f}")
print("A reader is entitled to that before reading the K-vs-K_eff horse race: near "
      "1 means the two theories are close to collinear and there is little to "
      "separate, whatever the p-value says.")
print("Section 5.2 expected 1 / ~3.5 / ~6.5 / ~7, reasoned from family structure "
      "and not measured. Fix the hypothesis to the measurement, never the reverse.")

keff_tbl.write_parquet(ARTIFACTS / "keff_table.parquet")
'''

CODE_INVARIANTS = r'''device = pick_device()
print(f"device {device}")

set_seed(42)
probe = ITransformer(ITransformerConfig()).to(device).eval()
base, scaled = scale_invariance_check(
    probe,
    torch.randn(64, 96, 8, device=device),
    torch.randn(64, 24, device=device),
    c=100.0,
)
rel = abs(base - scaled) / base
print(f"use_norm invariance: {base:.8f} vs {scaled:.8f}  rel={rel:.2e}")
assert rel < 1e-3, "FATAL (D03): use_norm inactive, or the scaler no longer cancels"
print(f"parameters {probe.n_parameters():,} — identical at every rung by construction")

set_seed(42)
plumb = ITransformer(ITransformerConfig(dropout=0.0)).to(device).train()
xs, ys = torch.randn(8, 96, 8, device=device), torch.randn(8, 24, device=device)
opt = torch.optim.Adam(plumb.parameters(), lr=1e-3)
for _ in range(200):
    opt.zero_grad(set_to_none=True)
    loss = torch.nn.functional.mse_loss(plumb(xs), ys)
    loss.backward()
    opt.step()
print(f"single-batch overfit (dropout=0.0): {loss.item():.3e}")
assert loss.item() < 1e-3, "plumbing broken"

print("\nNaive-RW in scaler space, per origin (D31 / D52b):")
naive = pl.DataFrame([
    {"origin": o.label, **{
        "mu_g": (t := build_origin_tensors(features, o, 1)).scaler.mean[0],
        "sigma_g": t.scaler.std[0],
        "mu_over_sigma": t.scaler.target_mu_over_sigma,
        "naive_rw_z": t.naive_rw_z,
        "n_train": len(t.train),
    }}
    for o in ORIGINS
])
print(naive)
print(f"mu_g/sigma_g spans {naive['mu_over_sigma'].min():+.5f} … "
      f"{naive['mu_over_sigma'].max():+.5f} and CHANGES SIGN, so the tilt is not a "
      f"constant a reader could subtract. It tracks the same bull/bear cycle H2 "
      f"invokes as its own mechanism, which is why it is confounded with the "
      f"effect of interest and does not wash out.")
naive.write_parquet(ARTIFACTS / "naive_rw_by_origin.parquet")
'''

CODE_PILOT = r'''pilot = stage5_pilot(features, out_root=ARTIFACTS, device=device)
print(pilot)

if not pilot.passed:
    print("\n*** Reposition the title to the descriptive variant NOW, not in week "
          "nine (root section 8.5). ***")
print("\nDisclose the pilot in section 13.2 as a SELECTION EVENT, stated separately "
      "from the DSR trial count — the DSR does not correct for selection over a "
      "paper's conclusion.")
'''

CODE_GRID = r'''import gc

# The pre-flight probe and the pilot allocated on this same device, and the grid
# is about to. Hand the memory back before it starts rather than carrying two
# dead models through 684 runs.
for _name in ("probe", "plumb", "xs", "ys", "opt", "loss"):
    globals().pop(_name, None)
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

ALL = manifest()
roots = discover_roots(ARTIFACTS)
todo = pending(ALL, roots)

by_arm = {}
for cell in ALL:
    by_arm[cell.arm] = by_arm.get(cell.arm, 0) + 1
already_done = len(ALL) - len(todo)
print(f"manifest {len(ALL)} unique runs {by_arm}")
print(f"roots searched: {[str(r) for r in roots]}")
print(f"already complete: {already_done}   pending: {len(todo)}")
print("Completeness is per run_id: both preds/ and meta/ present and "
      "meta.status == 'complete'. A run interrupted mid-training leaves no meta, "
      "so it is redone — losing at most one run, ~30 s (root §10.5).")

# The budget is what is LEFT of the session, not a fresh 11 h.
SESSION_BUDGET_H = 11.0
elapsed_h = (time.perf_counter() - SESSION_T0) / 3600.0
budget_h = max(0.25, SESSION_BUDGET_H - elapsed_h)
print(f"\nprelude took {elapsed_h * 60:.0f} min -> grid gets {budget_h:.2f} h "
      f"of the {SESSION_BUDGET_H:.1f} h session budget, guard reserves 0.5 h more")

# In-kernel and sequential. The definitions live in THIS namespace and nowhere
# else, so a subprocess could not reach them — and at ~30 s per run measured
# (D57) the 534 iTransformer cells are ~4.5 h, which fits without a second
# worker. The 150 baseline cells (D56) are not measured on a T4; they run last,
# so an overrun costs the comparators and never RQ1-RQ3's inputs. The second GPU
# idles; that is the price of the format, and it is stated rather than hidden.
if torch.cuda.device_count() > 1:
    print(f"NOTE: {torch.cuda.device_count()} GPUs visible, using {device} only. "
          f"Threads are not the fix — torch.manual_seed seeds EVERY CUDA device, "
          f"so two threads would clobber each other's generator mid-run.")

t0 = time.perf_counter()
summary = execute(
    todo, features,
    out_root=ARTIFACTS,
    roots=roots,
    guard=BudgetGuard(budget_h, 0.5),
    device=device,
    log=lambda msg: print(msg, flush=True),
)
print(summary)
print(f"grid finished after {(time.perf_counter() - t0) / 3600:.2f} h")

left = pending(ALL, discover_roots(ARTIFACTS))
GRID_COMPLETE = not left
print(f"remaining after this session: {len(left)} of {len(ALL)}")
if left:
    # The evaluation below is GATED on this. A partial grid is an unbalanced
    # panel, and §9.1's estimators refuse one by design — `amplification` raises
    # rather than silently comparing K=1 at eleven origins against K=8 at ten.
    # Letting that exception reach Kaggle would mark the version failed at the
    # exact moment its output is the only thing worth keeping.
    print("\nEvaluation is SKIPPED this session — the panel is incomplete, and a")
    print("half-panel beta1 is not a smaller answer but a different estimand.")
    print("Nothing is lost: preds/ and meta/ are on disk and resume is by run_id.")
    print("\n  1. Save Version now (its output IS the session's work)")
    print("  2. Attach that output as the next session's input Dataset")
    print("  3. Run this notebook again — completed runs are skipped automatically")

    ran = len(ALL) - len(left) - already_done
    if ran > 0:
        mean_s = (time.perf_counter() - t0) / ran
        print(f"\n  {ran} runs this session at ~{mean_s:.0f}s each -> about "
              f"{len(left) * mean_s / 3600:.1f} h of wall still to do, "
              f"{len(left) * mean_s / 3600 / 10.5:.1f} more sessions")
'''

CODE_RQ1 = r'''done = sorted(completed_run_ids(discover_roots(ARTIFACTS)))
grid = gather_grid(done, discover_roots(ARTIFACTS))
seed_avg = seed_average(grid)
print(f"gathered {len(done)} runs -> {grid.height} run-block rows -> "
      f"{seed_avg.height} seed-averaged cells")

main = seed_avg.filter((pl.col("model") == "itr") & (pl.col("pred_len") == 24))

# Root section 9.2 / D30: any number aggregated across origins carries the SE
# ACROSS ORIGINS, never the seed std. Seed dispersion measures re-initialisation
# noise on one fixed dataset; origin dispersion measures the sampling variability
# of the estimand, and in walk-forward crypto evaluation the second is typically
# an order of magnitude larger. Reporting the first as "+/-" on an aggregated row
# understates the headline uncertainty by roughly that factor — reintroducing,
# through the reporting convention, the overstated precision the wild cluster
# bootstrap was added to prevent.
per_origin = main.group_by(["origin", "k"]).agg(
    pl.col("mse").mean().alias("mse"), pl.col("r2_oos").mean().alias("r2_oos")
)
rung = (
    per_origin.group_by("k")
    .agg(
        pl.col("mse").mean().alias("MSE"),
        (pl.col("mse").std() / pl.col("mse").count().sqrt()).alias("SE_across_origins"),
        pl.col("r2_oos").mean().alias("R2_oos"),
        pl.col("mse").count().alias("n_origins"),
    )
    .sort("k")
)
print("\n--- RQ1: free rung effects ---")
print(rung)
print(f"seed std, a Monte-Carlo diagnostic only: {main['mse_seed_std'].mean():.6f} "
      f"mean across cells at n={main['n_seeds'][0]} seeds per cell")

wide = {int(k): per_origin.filter(pl.col("k") == k).sort("origin")["mse"].to_numpy()
        for k in (1, 4, 8, 12)}
d_4_8, d_8_12 = wide[4] - wide[8], wide[8] - wide[12]
margin = 0.25 * abs(float(d_4_8.mean()))
print(f"\ndelta MSE 4->8 = {d_4_8.mean():+.6f}   8->12 = {d_8_12.mean():+.6f}")
print("D49's margin is 0.25 x delta(4->8), fixed in advance: a non-significant "
      "delta is a failure to reject, not evidence of equivalence, and choosing the "
      "margin after seeing the rung is the p-hacking section 3 forbids for tau.")
print(tost_equivalence(d_8_12, margin))

# D32: RQ1 is a NON-NESTED comparison, not an OLS on three points. Four rungs give
# three deltas, and stacking 360 rows creates no information about a slope that
# varies only between rungs. K_eff measured per origin is what makes the regressor
# vary at all — and it is leak-free because the span is training-only.
keff_join = keff_tbl.select(["origin", "k", "pr_raw"]).rename({"pr_raw": "k_eff"})
race = main.join(keff_join, on=["origin", "k"], how="inner")
groups = race["origin_index"].to_numpy() * 100 + race["block"].to_numpy()
t_ab, p_ab = j_test(race["mse"].to_numpy(),
                    race["k"].to_numpy().astype(float),
                    race["k_eff"].to_numpy(), groups)
t_ba, p_ba = j_test(race["mse"].to_numpy(), race["k_eff"].to_numpy(),
                    race["k"].to_numpy().astype(float), groups)
print(f"\nJ-test  K augmented by K_eff: t={t_ab:+.3f} p={p_ab:.4f}   |   "
      f"K_eff augmented by K: t={t_ba:+.3f} p={p_ba:.4f}")
print("Both reject -> neither explanation alone suffices. Neither rejects -> the "
      "data cannot separate them, which at corr(K, K_eff) near 1 is the outcome to "
      "expect and to report plainly rather than to spin.")
'''

CODE_RQ2 = r'''print("--- RQ2: does the multivariate gap narrow with model age? ---")
amp = amplification(seed_avg, k_small=1, k_large=8)
print(amp.select(["origin", "block", "mse_small", "mse_large", "A"]).head(12))

beta = panel_beta1(amp, value="A", B=99_999, seed=42)
print(f"\n{beta}")
mde = minimum_detectable_beta1(beta.within_slopes)
print(f"\nminimum detectable beta1 at 80% power, alpha=0.05: {mde:+.6f}")
print(f"observed {beta.beta1:+.6f} is "
      f"{'INSIDE (undetectable)' if abs(beta.beta1) < abs(mde) else 'outside'} it")
print("If the MDE exceeds the plausible magnitude of A, RQ2 must be repositioned as "
      "descriptive BEFORE the grid: a non-significant beta1 is otherwise "
      "indistinguishable from a design that could never have detected decay.")

# D28: consecutive origins share 79.2% of their training data, so the clusters are
# NOT independent draws and the bootstrapped p is anticonservative by an
# unquantified amount. Windows become disjoint only at stride 5.
print("\ntraining-window overlap: 79.2% at stride 1, 58.3% at 2, 37.5% at 3, "
      "16.7% at 4. Disjoint only at stride 5, which leaves G=3.")
for offset in range(5):
    triple = [ORIGINS[i].label for i in range(offset, len(ORIGINS), 5)]
    subset = amp.filter(pl.col("origin").is_in(triple))
    if subset.height == len(triple) * 6:
        sub = panel_beta1(subset, value="A", B=9_999, seed=42)
        print(f"  {triple}: beta1={sub.beta1:+.6f}  p={sub.headline_p:.4f}  (G=3)")
print("At G=3 these will very likely be inconclusive, and THAT IS THE FINDING — it "
      "bounds what the full-panel p-value can honestly claim.")

# D50: K=1 vs K=8 differs in information AND in whether attention is active, at the
# same time. This holds information fixed and varies only what attention selects.
try:
    attn = attention_amplification(seed_avg, k=8)
    print(f"\nA_attn (uniform-attention control, D50):")
    print(panel_beta1(attn, value="A_attn", B=99_999, seed=42))
    print("Reporting both decompositions answers information-versus-attention "
          "directly, at runs Figure 5 needs anyway.")
except (ValueError, KeyError) as exc:
    print(f"\nuniform-attention arm not complete yet: {exc}")

# The falsification arm: the only design that identifies decay directly.
aged = main.filter((pl.col("k") == 8) & (pl.col("block") >= 4)).select(
    ["origin_index", "block", "mse"]).rename({"mse": "mse_aged"})
fresh = seed_avg.filter(pl.col("model") == "itrf").select(
    ["origin_index", "block", "mse"]).rename({"mse": "mse_fresh"})
falsify = aged.join(fresh, on=["origin_index", "block"], how="inner")
if falsify.height:
    gap = (falsify["mse_aged"] - falsify["mse_fresh"]).to_numpy()
    print(f"\nfalsification arm: mean(aged - fresh) = {gap.mean():+.6f} over "
          f"{len(gap)} (origin, block) cells")
    print("Positive means the fresh model really is better, so decay is age. Near "
          "zero while beta1 < 0 means beta1 is CALENDAR, not age, and RQ2's "
          "headline is an artefact.")
else:
    print("\nfalsification arm not complete yet")
'''

CODE_RQ3 = r'''print("--- RQ3: what retraining cadence? ---")
dec = decay(seed_avg, k=8)
print(dec.table)
if dec.excluded_origins:
    print(f"\nEXCLUDED, mean R2_oos <= 0 so there is no edge to lose a proportion "
          f"of: {list(dec.excluded_origins)}")
    print("Named, never silently dropped. Root section 10.3's first measured run "
          "returned R2_oos = -0.0183, so this guard may be the common case rather "
          "than the edge case section 9.1 assumed.")

print("\ntau sensitivity — headline 5%, pre-registered before the curve was seen:")
b_star_rows = []

if not dec.table.height:
    # Every origin failed the R2_oos > 0 guard, so D(i,b) has no denominator
    # anywhere and b* has nothing to estimate (D55). This is NOT censoring: a
    # censored origin has an edge that never decays past tau within 180 days,
    # whereas here there is no edge to lose a proportion of. Reporting the two
    # in one wording would claim skill the grid never found.
    for tau in TAU_SENSITIVITY:
        flag = "   <-- HEADLINE" if abs(tau - TAU_HEADLINE) < 1e-9 else ""
        print(f"  tau={tau:>6.1%}  UNDEFINED — no origin has positive mean skill{flag}")
        b_star_rows.append({"tau": tau, "status": "undefined", "median_b_star": None,
                            "ci_low": None, "ci_high": None, "events": 0,
                            "censored": 0, "n_origins": 0})
    print(f"\nRQ3 RETURNS NO ANSWER, and that is the finding. D(i,b) is a proportion "
          f"of skill lost; all {len(dec.excluded_origins)} origins have mean "
          f"R2_oos <= 0, so the proportion is undefined rather than large or small.")
    print("Root section 9.1's guard was written for an edge case and is here the "
          "ONLY case. Report it as 'the decay estimand is undefined under "
          "non-positive out-of-sample skill' — never as 'no decay detected within "
          "180 days', which is the right-censored wording and asserts an edge.")
else:
    for tau in TAU_SENSITIVITY:
        bs = dec.b_star(tau)
        km = kaplan_meier(bs["b_star"].to_numpy(), bs["event"].to_numpy())
        lo, hi = km.median_interval
        median = "censored >6" if km.median == float("inf") else f"{km.median:.0f}"
        interval = "censored" if lo == float("inf") else f"[{lo:.0f}, {hi:.0f}]"
        flag = "   <-- HEADLINE" if abs(tau - TAU_HEADLINE) < 1e-9 else ""
        print(f"  tau={tau:>6.1%}  crossings {km.n_events}/"
              f"{km.n_events + km.n_censored}  median b* {median}  CI {interval}{flag}")
        b_star_rows.append({"tau": tau, "status": "estimated",
                            "median_b_star": km.median, "ci_low": lo,
                            "ci_high": hi, "events": km.n_events,
                            "censored": km.n_censored, "n_origins": bs.height})

    print("\nb* resolves only to 30-day granularity and only out to 180 days. If no "
          "block crosses tau, the honest answer is 'no decay detected within 180 days' "
          "— a right-censored result, not a missing one. Say it in those words, and put "
          "the INTERVAL in the abstract, never a bare integer.")

# H3: larger K decays faster. Needs surviving origins AND at least one crossing:
# with zero events in both arms the log-rank variance is zero and the statistic
# is 0/0, which prints as nan and reads like a computed result. Section 12 calls
# a number that cannot be regenerated a documented failure, so say why instead.
a = dec.b_star(TAU_HEADLINE)
b = decay(seed_avg, k=1).b_star(TAU_HEADLINE)
events = (int(a["event"].sum()) if a.height else 0,
          int(b["event"].sum()) if b.height else 0)
if a.height and b.height and sum(events):
    chi2, p = logrank(a["b_star"].to_numpy(), a["event"].to_numpy(),
                      b["b_star"].to_numpy(), b["event"].to_numpy())
    print(f"\nlog-rank K=8 vs K=1 at tau=5%: chi2={chi2:.3f}  p={p:.4f}  "
          f"(H3; crossings K=8 {events[0]}, K=1 {events[1]})")
elif not (a.height and b.height):
    print(f"\nlog-rank K=8 vs K=1 UNAVAILABLE: surviving origins K=8 {a.height}, "
          f"K=1 {b.height}. H3 compares decay RATES, so it needs an edge in both "
          "arms; with none, H3 is untestable rather than rejected.")
else:
    print(f"\nlog-rank K=8 vs K=1 UNAVAILABLE: zero crossings in both arms "
          f"({a.height} and {b.height} origins, all censored at 6). The statistic "
          "is 0/0 here, not a large p-value — H3 is untestable, and reporting a "
          "nan as though it were computed would be the same defect as D55.")
'''

CODE_SAVE = r'''_digest, _provenance = _input_sha256(PARQUET)

paper_numbers = {
    "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "input_parquet": str(PARQUET),
    "input_sha256": _digest,
    "input_sha256_source": _provenance,
    "code_sha256": code_sha256(),
    "runs_complete": len(done),
    "runs_in_manifest": len(ALL),
    "keff": {
        "gate_pr_k8_pre_first_origin": gate,
        "gate_floor": GATE_PR_FLOOR,
        "corr_k_keff": corr_k_keff(keff_tbl),
        "per_rung": rung_view.to_dicts(),
    },
    "rq1": {
        "rung_effects": rung.to_dicts(),
        "delta_4_to_8": float(d_4_8.mean()),
        "delta_8_to_12": float(d_8_12.mean()),
        "tost_margin": margin,
        "tost": str(tost_equivalence(d_8_12, margin)),
        "j_test_k_augmented_by_keff": {"t": t_ab, "p": p_ab},
        "j_test_keff_augmented_by_k": {"t": t_ba, "p": p_ba},
    },
    "rq2": {
        "beta1": beta.beta1, "t": beta.t_statistic, "cluster_se": beta.cluster_se,
        "p_rademacher": beta.p_rademacher, "p_webb": beta.p_webb,
        "headline_p": beta.headline_p, "G": beta.n_clusters,
        "N": beta.n_observations, "B": beta.B,
        "minimum_detectable_beta1": mde,
        "within_slopes": beta.within_slopes.tolist(),
        "effective_independent_training_sets": 4,
        "consecutive_origin_overlap_pct": 79.2,
    },
    "rq3": {
        "tau_headline": TAU_HEADLINE,
        "b_star": b_star_rows,
        "excluded_origins": list(dec.excluded_origins),
    },
}

out = ARTIFACTS / "paper_numbers.json"
out.write_text(json.dumps(paper_numbers, indent=2, default=float))
seed_avg.write_parquet(ARTIFACTS / "seed_averaged_cells.parquet")
grid.write_parquet(ARTIFACTS / "run_block_metrics.parquet")
amp.write_parquet(ARTIFACTS / "amplification_panel.parquet")
dec.table.write_parquet(ARTIFACTS / "decay_panel.parquet")

print(f"wrote {out}")
for path in sorted(ARTIFACTS.glob("*.parquet")):
    print(f"  {path.name}  {path.stat().st_size / 1e6:.2f} MB")
print(f"\npreds {len(list((ARTIFACTS / 'preds').glob('*.parquet')))} files  |  "
      f"meta {len(list((ARTIFACTS / 'meta').glob('*.json')))} files")
print(f"remaining runs: {len(pending(ALL, discover_roots(ARTIFACTS)))}")
print("\nSave Version now, then attach this output as the next session's input "
      "Dataset. Nothing else needs doing by hand.")
'''


# -- assembly ----------------------------------------------------------------


def guarded(body: str, what: str) -> str:
    """Run ``body`` only when the grid is complete; say why, clearly, when not.

    A partial grid is an **unbalanced panel**, and §9.1's estimators refuse one by
    design: ``amplification`` raises rather than compare K=1 at eleven origins
    against K=8 at ten, and RQ1's ``wide[4] - wide[8]`` would broadcast-error.
    That is correct behaviour and must not be softened — a half-panel β₁ is a
    different estimand, not a noisier one.

    What is wrong is *where* the exception lands. A 12-hour Kaggle session that
    stops at run 200 of 684 would raise here, in the last cells, and mark the
    version failed at the exact moment its output is the only thing worth
    keeping. So the estimators stay strict and the notebook simply does not call
    them until the panel exists.

    Indented mechanically rather than by hand, so the guarded and unguarded
    sources cannot drift.
    """
    head = (
        f"if not GRID_COMPLETE:\n"
        f'    print("{what}: SKIPPED — the grid is incomplete, so the panel is "\n'
        f'          "unbalanced and §9.1\'s estimators refuse it by design. "\n'
        f'          "Resume in the next session; nothing is recomputed.")\n'
        f"else:\n"
    )
    return head + textwrap.indent(body, "    ")


def _lines(text: str) -> list[str]:
    """nbformat's canonical source form: a list of lines, newlines kept."""
    return text.splitlines(keepends=True)


def _markdown(index: int, text: str) -> dict:
    return {
        "id": f"md-{index:02d}",
        "cell_type": "markdown",
        "metadata": {},
        "source": _lines(text),
    }


def _code(index: int, text: str) -> dict:
    return {
        "id": f"code-{index:02d}",
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _lines(text),
    }


def module_cell_source(name: str) -> str:
    """Banner comment plus the flattened module.

    ``tests/test_notebook_sync.py`` locates module cells by the banner and
    compares what follows against :func:`flatten_module_source`, so the two
    cannot drift without a test failing. The banner also gives a reader scrolling
    the notebook the module's name, which the ``%%writefile`` line used to supply
    and nothing else now does.
    """
    return f"{MODULE_BANNER.format(name=name)}\n{flatten_module_source(name)}"


def build() -> dict:
    """The whole notebook, as an nbformat 4.5 dictionary."""
    cells: list[dict] = []
    counter = 0

    def md(text: str) -> None:
        nonlocal counter
        cells.append(_markdown(counter, text))
        counter += 1

    def code(text: str) -> None:
        nonlocal counter
        cells.append(_code(counter, text))
        counter += 1

    md(MD_TITLE)
    md(MD_SETUP)
    code(CODE_SETUP)

    md(MD_DEFINE)
    for name in MODULE_ORDER:
        code(module_cell_source(name))
    code("MODULE_NAMES = [\n" + "".join(
        f"    {name!r},\n" for name in MODULE_ORDER) + "]\n")
    code(CODE_PROVENANCE.replace("{digest}", package_digest()))

    md(MD_DATA)
    code(CODE_DATA)
    md(MD_VARIATES)
    code(CODE_FEATURES)
    md(MD_KEFF)
    code(CODE_KEFF)
    md(MD_INVARIANTS)
    code(CODE_INVARIANTS)
    md(MD_GATE)
    code(CODE_PILOT)
    md(MD_GRID)
    code(CODE_GRID)
    md(MD_EVAL)
    code(guarded(CODE_RQ1, "RQ1"))
    code(guarded(CODE_RQ2, "RQ2"))
    code(guarded(CODE_RQ3, "RQ3"))
    md(MD_SAVE)
    code(guarded(CODE_SAVE, "paper_numbers.json"))

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def render(notebook: dict) -> str:
    """UTF-8 JSON with a trailing newline — the shape nbformat itself writes."""
    return json.dumps(notebook, indent=1, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build notebooks/iTransformer.ipynb")
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit 1 if the committed notebook differs from what src/ implies",
    )
    args = parser.parse_args(argv)

    missing = [n for n in MODULE_ORDER if not (PACKAGE / n).exists()]
    if missing:
        raise SystemExit(f"missing package modules: {missing}")
    extra = sorted(p.name for p in PACKAGE.glob("*.py") if p.name not in MODULE_ORDER)
    if extra:
        raise SystemExit(
            f"{extra} exist in src/itransformer_btc/ but are absent from "
            f"MODULE_ORDER, so the notebook would define an incomplete package "
            f"while code_sha256 still named every file. Add them here, in "
            f"dependency order — which is now execution order and must be right."
        )

    notebook = build()
    text = render(notebook)

    if args.check:
        current = NOTEBOOK.read_text(encoding="utf-8") if NOTEBOOK.exists() else ""
        if current != text:
            print(
                f"{NOTEBOOK} is stale against src/itransformer_btc/. "
                f"Run: python tools/build_notebook.py",
                file=sys.stderr,
            )
            return 1
        print(f"{NOTEBOOK} is current")
        return 0

    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    # newline="\n" explicitly: Windows would otherwise translate to CRLF and the
    # generator would emit platform-dependent bytes for identical content, so the
    # same `src/` would produce a whole-file diff depending on who ran it.
    NOTEBOOK.write_text(text, encoding="utf-8", newline="\n")
    n_code = sum(1 for c in notebook["cells"] if c["cell_type"] == "code")
    print(
        f"wrote {NOTEBOOK}  "
        f"({len(notebook['cells'])} cells, {n_code} code, "
        f"{len(text) / 1e3:.0f} kB)  code_sha256 {package_digest()[:16]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
