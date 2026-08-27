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
import dataclasses
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
    # Last, and it has to be: it reads `completed_run_ids` from `runner` and a
    # driver from every analysis module above it. Nothing imports it back, so no
    # cell below depends on a name it defines.
    "report.py",
)

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


# -- sections ----------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Section:
    """One notebook cell's worth of a module, plus the heading above it.

    ``starts_at`` names the first module-level definition the section owns; the
    first section of every module carries ``None`` and starts at line one. Cells
    are the half-open line ranges between consecutive ``starts_at`` anchors, so
    they are **contiguous and exhaustive by construction** — the concatenation of
    a module's cells is its flattened source byte for byte, with no normalisation
    step for a reader to distrust (`D63`).
    """

    title: str
    emoji: str
    blurb: str
    starts_at: str | None = None


#: Per-module colour, so a reader navigating a ~300-cell notebook can tell which
#: module they are inside without reading the heading text. ``(from, to, accent,
#: heading, body)`` — the five slots `markdown-example.ipynb` uses.
MODULE_THEME: dict[str, tuple[str, str, str, str, str]] = {
    "config.py": ("#0b1021", "#14213d", "#8ecae6", "#8ecae6", "#a8c7d8"),
    "__init__.py": ("#101010", "#1c1c1c", "#9e9e9e", "#d0d0d0", "#a8a8a8"),
    "segments.py": ("#1a1200", "#2b1d00", "#ffb703", "#ffd60a", "#ffca7a"),
    "windows.py": ("#1a1200", "#2b1d00", "#fb8500", "#ffb703", "#ffca7a"),
    "budget.py": ("#231400", "#3a2200", "#f48c06", "#ffba08", "#ffd08a"),
    "features.py": ("#001a1a", "#002b2b", "#48cae4", "#90e0ef", "#ade8f4"),
    "splits.py": ("#001233", "#001845", "#4cc9f0", "#8ecae6", "#a9d6e5"),
    "model.py": ("#150029", "#22003d", "#c77dff", "#e0aaff", "#cbb2e8"),
    "train.py": ("#1b0033", "#2d0052", "#bf5af2", "#e0aaff", "#cbb2e8"),
    "keff.py": ("#150029", "#240046", "#9d4edd", "#e0aaff", "#c8a2e0"),
    "efficiency.py": ("#002200", "#003300", "#7ae582", "#95d5b2", "#b7e4c7"),
    "metrics.py": ("#03071e", "#370617", "#e94560", "#f5a623", "#ffd6a5"),
    "baselines.py": ("#0a1a12", "#0f2a1c", "#52b788", "#95d5b2", "#b7e4c7"),
    "comparisons.py": ("#2b0a00", "#3d1000", "#ff7b54", "#ffb4a2", "#ffd8c2"),
    "economics.py": ("#0d1b2a", "#1b263b", "#00b4d8", "#90e0ef", "#ade8f4"),
    "attention.py": ("#2d0036", "#4a0060", "#e0aaff", "#e0aaff", "#c77dff"),
    "runner.py": ("#012a4a", "#013a63", "#48cae4", "#90e0ef", "#caf0f8"),
    "report.py": ("#001a0d", "#003317", "#52b788", "#95d5b2", "#b7e4c7"),
}

#: What each module is, in one line, under its ``##`` heading.
MODULE_BLURB: dict[str, str] = {
    "config.py": "Setiap angka protokol yang root §8.1 kunci, sebagai konstanta bernama. Tidak ada magic number di modul lain.",
    "__init__.py": "Permukaan paket. Di notebook ia tidak mengimpor apa pun — seluruh nama sudah hidup di namespace kernel yang sama.",
    "segments.py": "Hukum segmen root §4.3: gap memutus deret, dan bar tak-layak-pakai memutusnya juga. Tidak ada imputasi di mana pun.",
    "windows.py": "Jendela divalidasi lewat <strong>timestamp</strong>, tidak pernah lewat indeks posisional — bug paling senyap di pipeline ini.",
    "budget.py": "Anggaran jendela per origin, ditegakkan dengan kesamaan persis terhadap <code>docs/ORIGIN_WINDOW_BUDGET.md</code> (<code>D45</code>).",
    "features.py": "Dua belas variat F1–F5, seluruhnya fungsi per-bar. Tidak satu pun memakai rolling window (root §5.3).",
    "splits.py": "Purge H langkah di <strong>kedua</strong> batas, dan scaler yang dipasang hanya pada sub-blok 21 bulan (<code>D24</code>).",
    "model.py": "iTransformer encoder-only. Attention berjalan lintas variat, bukan lintas waktu; kausalitas ditegakkan di hulu.",
    "train.py": "Loop latih GPU-resident, tanpa <code>DataLoader</code>, plus kontrak keterlacakan root §12 yang setiap run tulis.",
    "keff.py": "Dimensionalitas efektif — regresor RQ1. Dihitung hanya pada rentang latih, per origin (<code>D02</code>, <code>D44</code>).",
    "efficiency.py": "Uji efisiensi pasar root §4.5: ADF, variance ratio, Hurst. Full sample dan per sub-blok latih.",
    "metrics.py": "Seluruh estimator paper: metrik, DM/Clark–West, survival, dan bootstrap klaster liar untuk β₁.",
    "baselines.py": "Ridge, DLinear, PatchTST — masing-masing dengan K eksplisit (<code>D40</code>) dan objektif terbitannya sendiri (<code>D56</code>).",
    "comparisons.py": "Matriks pasangan, stepdown Romano–Wolf, dan Model Confidence Set (<code>D35</code>).",
    "economics.py": "Evaluasi ekonomi root §13.5: fase 00:00 UTC, non-overlapping, per segmen, DSR per origin.",
    "attention.py": "Peta attention per tercile volatilitas untuk Figure 5. Cabang <code>capture</code> tidak mengonsumsi RNG (<code>D62d</code>).",
    "runner.py": "Manifes 969 run, penemuan resume lewat glob, penjaga anggaran sesi, dan eksekutor grid.",
    "report.py": "Sembilan tabel dan enam figure, seluruhnya di-render dari <code>paper_numbers.json</code> — tidak pernah disalin tangan.",
}

#: Where every module is cut, and what the heading above each cut says.
#:
#: ``main()`` refuses to run when a module is missing from this table, for the
#: same reason it refuses on a missing ``MODULE_ORDER`` entry: a silently
#: unsegmented module would ship as one unreadable cell and nothing would say so.
SECTION_MAP: dict[str, tuple[Section, ...]] = {
    "config.py": (
        Section("Header", "📄", "Docstring modul. Impornya hidup di sel Library di atas."),
        Section("Kontrak data & jendela", "📐", "Angka terukur dari laporan Stage 1, plus L=96 dan H=24. Root §4.1 dan §6.2.", "DATA_START"),
        Section("Protokol walk-forward", "🗓️", "24 bulan latih, 21/3 latih-validasi, enam blok 30 hari, spasi lima bulan (<code>D26</code>).", "TRAIN_MONTHS"),
        Section("Origin", "📍", "Satu titik latih-ulang: batas latih, validasi, uji, seluruhnya epoch-based UTC.", "Origin"),
        Section("Origin falsifikasi", "🧪", "Model fresh di <code>o + 90 hari</code>, dievaluasi pada blok kalender yang sama (root §8.1).", "FalsificationOrigin"),
        Section("Grid lima belas origin", "🧭", "2020-01 hingga 2025-11. Spasi lima bulan koprima terhadap 12, jadi indeks blok tidak kolinear dengan bulan kalender.", "OriginLike"),
    ),
    "__init__.py": (
        Section("Permukaan paket", "📦", "Di checkout ia mengekspor nama; di notebook impornya dibuang dan <code>__all__</code> tinggal dokumentasi."),
    ),
    "segments.py": (
        Section("Header & konstanta", "📄", "Jam dalam milidetik dan path parquet bawaan."),
        Section("Segmen", "🧱", "Deretan maksimal bar per-jam yang bersambung dan layak pakai.", "Segment"),
        Section("Memuat bar & masker layak-pakai", "📥", "Bar bervolume nol atau ber-<code>H == L</code> dikecualikan — tiga bar, dan ketiganya bar yang sama (<code>D51c</code>).", "load_bars"),
        Section("Membangun segmen", "✂️", "Gap memutus deret. Tidak ada ffill, tidak ada reindex ke grid jam penuh (root §4.2).", "build_segments"),
        Section("Ringkasan break", "📊", "Break per origin — target asersi kesamaan-persis yang <code>D45</code> tuntut.", "BreakSummary"),
    ),
    "windows.py": (
        Section("Header", "📄", "Docstring modul."),
        Section("Enumerasi jendela", "🪟", "<strong>Divalidasi lewat timestamp, tidak pernah lewat indeks posisional.</strong> Setelah baris manapun jatuh, sliding posisional menutup gap tanpa terlihat.", "enumerate_windows"),
        Section("Menghitung jendela", "🔢", "Bentuk tertutup hanyalah batas atas; hitungan sebenarnya per segmen (<code>D51a</code>).", "count_windows"),
    ),
    "budget.py": (
        Section("Header & anggaran terkomit", "📄", "Angka per origin yang <code>docs/ORIGIN_WINDOW_BUDGET.md</code> kunci."),
        Section("Anggaran per origin", "💰", "Jendela latih yang bertahan, break, dan bar hilang untuk satu origin.", "OriginBudget"),
        Section("Start blok uji yang bertahan", "🎯", "720 origin ramalan per blok bersih, bukan 601 — akuntansi uji memakai semantik uji (<code>D51b</code>).", "surviving_block_starts"),
        Section("Tabel anggaran", "📋", "Lima belas baris yang Tabel 1 render, diukur bukan disalin.", "origin_budget"),
    ),
    "features.py": (
        Section("Header, tangga variat, konstanta", "📄", "Dua belas variat berurutan, target <code>r</code>, dan penstabil κ = 1e-9 untuk Rogers–Satchell (<code>D52a</code>)."),
        Section("Kolom per rung", "🪜", "K ∈ {1, 4, 8, 12} — satu-satunya potongan konsisten yang <code>D01</code> tetapkan.", "ladder_columns"),
        Section("Membangun dua belas variat", "🔬", "Seluruhnya fungsi per-bar. <strong>Tidak satu pun memakai rolling window</strong>, jadi kelas kebocoran <code>center=True</code> tak terwakili (root §5.3).", "build_features"),
    ),
    "splits.py": (
        Section("Header & semantik", "📄", "Semantik latih lawan uji — yang membedakan 601 dari 720."),
        Section("Start jendela per semantik", "🪟", "Enumerasi hingga <code>val_start − L − H</code>: purge H langkah di batas latih→validasi (<code>D24</code>).", "window_starts"),
        Section("Scaler", "⚖️", "StandardScaler dipasang <strong>hanya</strong> pada sub-blok 21 bulan. Memindahkan <code>train_end</code> adalah kebocoran, bukan ketidakcocokan.", "Scaler"),
        Section("Tensor split", "🧮", "Tensor GPU-resident per split, plus <code>mu_g</code>/<code>sigma_g</code> yang Naive-RW butuh (<code>D31</code>).", "SplitTensors"),
        Section("Merakit tensor origin", "🏗️", "Satu origin, tiga split, tanpa <code>DataLoader</code> di mana pun (root §10.3).", "_gather"),
    ),
    "model.py": (
        Section("Header", "📄", "Hanya torch yang dipakai modul ini. TensorFlow, Keras, dan JAX terlarang (root §2)."),
        Section("Konfigurasi arsitektur", "🎛️", "<code>d_model=128</code>, bukan 512 — panjang urutan attention adalah N ≤ 12, dan 512 akan over-parameterise terhadap ~14.000 sampel (<code>D25</code>).", "ITransformerConfig"),
        Section("Jadwal panjang", "⏱️", "Arm <code>longsched</code>: <code>lr_halve_every=8</code>, 60 epoch, patience 10 (<code>D62c</code>).", "LongScheduleConfig"),
        Section("Blok encoder", "🧠", "Embedding terbalik <code>Linear(L → d_model)</code>, attention lintas variat, FFN. Tanpa causal mask — semua token sezaman.", "InvertedEmbedding"),
        Section("ITransformer", "🔁", "Encoder-only. Loss dihitung hanya pada kanal target, di setiap rung (<code>D39</code>).", "ITransformer"),
    ),
    "train.py": (
        Section("Header & konstanta", "📄", "Path artefak dan <code>CODE_SHA256_OVERRIDE</code> — digest yang dipin generator (root §12)."),
        Section("Seed & pemilihan perangkat", "🎲", "Presisi digerbangi <code>get_device_capability(0)[0] >= 8</code>, tidak pernah <code>is_bf16_supported()</code>, yang mengembalikan True palsu di T4.", "set_seed"),
        Section("Spesifikasi run & hasil latih", "📝", "Satu sel grid: model, origin, K, H, seed — komponen <code>run_id</code> root §10.4.", "RunSpec"),
        Section("Arsitektur & protokol Forecaster", "🏛️", "Antarmuka yang iTransformer dan ketiga baseline sama-sama penuhi.", "Architecture"),
        Section("Helper batch & prediksi", "⚡", "Batch dengan mengiris indeks tensor yang sudah di GPU. Data tidak bergerak setelah muat awal.", "_to_device"),
        Section("Loop latih", "🔥", "Early stopping patience 5 pada MSE validasi; LR dibelah tiap 4 epoch (<code>D47</code>).", "train_one"),
        Section("Uji invarian skala", "🔍", "<code>MSE(c·x)/c² == MSE(x)</code> — bentuk yang benar; versi loss-identik tidak mungkin lolos (<code>D03</code>).", "scale_invariance_check"),
        Section("Provenance kode & input", "🔐", "<code>code_sha256</code> ada karena <code>git_sha</code> berbunyi unknown di Kaggle — persis tempat grid berjalan (<code>D54b</code>).", "_git_sha"),
        Section("Menulis artefak run", "💾", "Prediksi mentah selalu, bukan cuma metrik. Tanpanya DM, analisis rezim, dan evaluasi ekonomi mustahil.", "write_artifacts"),
        Section("Idempotensi", "✅", "Run lengkap hanya bila kedua berkas ada <strong>dan</strong> <code>meta.status</code> berbunyi complete.", "is_complete"),
    ),
    "keff.py": (
        Section("Header & konstanta", "📄", "Lantai gerbang PR = 5,0, dipra-registrasi sebelum apa pun berjalan (<code>D02</code>)."),
        Section("Participation ratio", "📐", "<code>PR = (Σλ)² / Σλ²</code> pada matriks korelasi. Diukur juga pada fitur ternormalisasi-jendela — confound instance-norm <code>D04</code>.", "participation_ratio"),
        Section("Stable rank & spektrum sadar-lookback", "📊", "Standardisasi dalam jendela lebih dulu; tanpa itu satu baris mendominasi kedua norma dan angkanya jadi artefak satuan (<code>D53a</code>, <code>D53b</code>).", "stable_rank"),
        Section("Baris K_eff per origin", "🧾", "Regresor RQ1, dihitung <strong>hanya pada sub-blok latih 21 bulan</strong> origin itu (<code>D44</code>).", "KeffRow"),
        Section("Tabel K_eff & korelasi", "🔗", "<code>corr(K, K_eff) = 0,828</code>, bukan ≈0,97 yang diantisipasi — pacuannya jadi <em>lebih</em> teridentifikasi.", "keff_table"),
        Section("Gerbang Stage 3b", "🚧", "PR di K=8 terukur 4,393 &lt; 5,0. Aksinya disklosur, bukan re-cut (<code>D48</code>).", "gate_pr"),
        Section("PR bergulir — deskriptif saja", "🌀", "Jendela 90 hari, 2018–2026. <strong>Tidak boleh menginformasikan satu pun keputusan desain</strong> (root §5.4).", "ROLLING_WINDOW_DAYS"),
        Section("OLS R² bergulir", "📉", "Premis H2 diuji sebelum satu epoch pun berjalan — Figure 2b.", "rolling_ols_r2"),
    ),
    "efficiency.py": (
        Section("Header & konstanta", "📄", "Lag variance ratio dan ambang minimum Hurst."),
        Section("Baris hasil", "🧾", "Bentuk keluaran untuk VR dan ADF.", "VarianceRatioRow"),
        Section("Eksponen Hurst (R/S)", "🌊", "H ≈ 0,5 berarti tanpa memori panjang. Terukur 0,5515 full sample.", "hurst_rs"),
        Section("Variance ratio Lo–MacKinlay", "📏", "VR ≈ 1 konsisten dengan random walk.", "variance_ratios"),
        Section("ADF", "🧪", "Log-return stasioner. Terukur −38,43.", "adf"),
        Section("Tabel efisiensi", "📋", "Full sample <strong>dan</strong> tiap sub-blok latih — klaimnya soal <em>variasi</em>, dan satu baris tidak bisa menunjukkannya.", "_row"),
    ),
    "baselines.py": (
        Section("Header & protokol baseline", "📄", "Setiap baseline membawa K eksplisit (<code>D40</code>), dan yang channel-independent membawa objektif all-channel terbitannya sendiri (<code>D56</code>)."),
        Section("Ridge — konfigurasi", "📏", "α dipilih pada sub-blok validasi. <strong>Satu-satunya hiperparameter yang dipilih di mana pun dalam studi ini.</strong>", "RIDGE_ALPHAS"),
        Section("Ridge — forecaster", "➗", "L2 pada fitur K yang sama. Menjawab pertanyaan <code>D17</code>: apakah transformer dibutuhkan sama sekali?", "RidgeForecaster"),
        Section("DLinear — konfigurasi", "📉", "Dekomposisi tren–musiman plus linear.", "DLinearConfig"),
        Section("DLinear — dekomposisi & model", "🪚", "Moving average-nya <em>terpusat</em> — dan §8.3 selamat: rata-ratanya dihitung dari 96 bar jendela itu sendiri, seluruhnya mendahului jam ramalan pertama.", "SeriesDecomposition"),
        Section("PatchTST — konfigurasi", "🧩", "Patch 16, stride 8.", "PatchTSTConfig"),
        Section("PatchTST — model", "🔷", "Memakai ulang blok encoder dan kapasitas iTransformer <strong>verbatim</strong>, jadi keduanya berbeda hanya pada <em>apa itu token</em>.", "PatchTST"),
        Section("LSTM — konfigurasi", "🔁", "Dua layer, hidden 128, dropout 0,1 — diadopsi dari root §7, tidak di-tune. <strong>Multivariat</strong>, bukan channel-independent: K=8-nya berarti apa yang ridge dan iTransformer maksud (<code>D64</code>).", "LSTMConfig"),
        Section("LSTM — forecaster", "🧬", "Hidden state terakhir langsung ke <code>Linear(hidden, H)</code>. Model deep paling sering disitir literatur crypto yang paper ini lawan.", "LSTMForecaster"),
        Section("Dua komparator naif", "🪶", "<code>persist</code> mengulang return terakhir; <code>seasonal</code> mengulang return satu siklus harian ke belakang. Nol parameter, K=1, dan berbeda dari Naive-RW yang meramalkan <code>y_raw = 0</code> (<code>D31</code>).", "SEASONAL_PERIOD"),
        Section("Penyelarasan jendela baseline", "🔗", "Timestamp yang dievaluasi diasersi sama sebelum RelMSE dihitung — fatal untuk setiap pasangan selain Naive-RW.", "assert_baseline_alignment"),
    ),
    "metrics.py": (
        Section("Header & konstanta", "📄", "τ headline 5% dan sensitivitasnya, plus <code>B_STAR_SCHEMA</code> — skema dideklarasikan supaya kasus semua-origin-dikecualikan tetap mengembalikan frame berkolom (<code>D55</code>)."),
        Section("Memuat run", "📥", "Parse <code>run_id</code> dan muat prediksi/meta dari akar mana pun yang ditemukan lewat glob.", "parse_run_id"),
        Section("Metrik inti", "📐", "MSE, MAE, RelMSE, dan <code>R²_oos = 1 − RelMSE</code> (<code>D20</code>). RMSE mentah dilaporkan berdampingan supaya dua skala dapat direkonsiliasi.", "mse"),
        Section("Jendela non-overlap & Pesaran–Timmermann", "🎯", "Di H=24 target tumpang-tindih 23 dari 24 jam, jadi PT over-reject parah; versi tumpang-tindih deskriptif saja (<code>D21</code>).", "non_overlapping_mask"),
        Section("Akurasi arah", "🧭", "DA di h=1, h=24, dan pada return kumulatif 24 jam — tiga rezim pengujian berbeda.", "_hit_rate"),
        Section("Metrik per blok", "🧱", "Baseline dinilai pada <strong>himpunan jendela yang persis sama</strong> dengan pembandingnya (<code>D45</code>).", "assert_same_windows"),
        Section("Perakitan grid & rata-rata seed", "🧮", "Rata-rata seed <strong>lebih dulu</strong>, rasio kedua — dibalik keduanya berbeda karena Jensen (<code>D42</code>).", "gather_grid"),
        Section("Amplifikasi A dan A_attn", "📶", "<code>A</code> variabel terikat RQ2, hanya K=1 lawan K=8. <code>A_attn</code> memisahkan <em>attention</em> dari <em>informasi</em> (<code>D50</code>).", "amplification"),
        Section("Decay dan b*", "⏳", "<code>D(i,b)</code> di skala skill (<code>D23</code>). Origin ber-<code>R²_oos ≤ 0</code> dikecualikan dengan disebut namanya — dan itu <strong>seluruh</strong> lima belasnya.", "DecayResult"),
        Section("Kurva survival & kuantil normal", "📈", "<code>b*</code> right-censored di 6 — data survival berinterval, bukan bilangan bulat telanjang (<code>D41</code>).", "SurvivalCurve"),
        Section("Kaplan–Meier & log-rank", "🩺", "Estimator dan uji H3. Tidak tersedia di sini: tidak satu arm pun punya origin bertahan.", "kaplan_meier"),
        Section("Varians jangka panjang", "📊", "Estimator <strong>rektangular</strong> terpotong pada lag h−1, bukan bobot Bartlett — Bartlett menyusutkan γ̂₂₂ ~92% dan menghasilkan p terlalu optimistis (<code>D34</code>).", "_rectangular_lrv"),
        Section("Koreksi Harvey–Leybourne–Newbold", "🔧", "Dirujuk ke Student-t dengan T−1 dof, bukan normal baku. Faktornya diasersi positif sebelum dipakai.", "TestResult"),
        Section("DM & Clark–West", "⚔️", "Pasangan <strong>bersarang</strong> memakai Clark–West; DM baku undersized justru terhadap alternatif yang studi ini uji (<code>D29</code>).", "hln_test"),
        Section("Rugi per origin", "🧾", "Rugi per origin per model, masukan matriks pasangan.", "per_origin_loss"),
        Section("Bobot bootstrap & matriks panel", "🎲", "Rademacher dan Webb 6-titik, keduanya dilaporkan. Panel tak seimbang ditolak keras — reduksi β₁ ke rata-rata within-slope hanya berlaku di panel seimbang.", "Beta1Result"),
        Section("β₁ dengan bootstrap klaster liar", "🧬", "WCR dengan null dipaksakan, mem-bootstrap <em>t</em> bukan β̂, B = 99.999, satu sisi, klaster = origin. p memakai <code>(1 + count)/(1 + B)</code> (<code>D53d</code>).", "panel_beta1"),
        Section("TOST & uji-J non-nested", "⚖️", "TOST menguji <em>ekuivalensi</em> rung 8→12; uji-J memacu penjelasan K lawan K_eff (<code>D32</code>, <code>D49</code>).", "EquivalenceResult"),
        Section("Efek minimum terdeteksi", "🔬", "MDE dipublikasi <strong>sebelum</strong> blok tes dibuka. Tanpa itu null tidak terbedakan dari desain yang memang tak sanggup mendeteksi (root §9.2 syarat 6).", "minimum_detectable_beta1"),
        Section("Tabel akurasi arah & skala mentah", "📋", "DA per rung dan rekonsiliasi MSE-scaler dengan RMSE mentah.", "directional_accuracy_table"),
        Section("RelMSE falsifikasi & β₁ dengan cakupan", "🧯", "Arm falsifikasi dilaporkan pada RelMSE, tidak pernah MSE ruang-scaler — 99,7% angka mentahnya adalah drift scaler (<code>D60i</code>, <code>D62e</code>).", "falsification_relmse"),
    ),
    "comparisons.py": (
        Section("Header & konstanta", "📄", "Level MCS 90% dan 75%, dan B bootstrap bawaan."),
        Section("Nesting & label model", "🪆", "Tangga bersifat kumulatif dan Naive-RW bersarang di setiap model — itu yang menentukan statistik mana yang sah per pasangan.", "nesting_order"),
        Section("Panel prediksi", "🗂️", "Prediksi seluruh model pada himpunan jendela bersama.", "PredictionPanel"),
        Section("Membangun panel", "🏗️", "Memuat setiap <code>preds/*.parquet</code> dan menegakkan keselarasan jendela lintas model.", "build_panel"),
        Section("Diferensial rugi", "➖", "Diferensial per jendela, per (origin, blok) — T ≈ 720, h = 24, lag pemotongan 23.", "_per_window"),
        Section("Bootstrap klaster & Romano–Wolf", "🪜", "Stepdown mengendalikan FWER lintas <em>seluruh</em> pasangan. Ia menghapus kedelapan penolakan mentah terhadap Naive-RW (<code>D35</code>, <code>D62a</code>).", "_studentised"),
        Section("Model Confidence Set", "🏅", "Siapa yang tak terbedakan dari yang terbaik. Di 90% maupun 75%: Naive-RW dan keempat rung ridge, tanpa satu pun model deep.", "model_confidence_set"),
        Section("Diagnostik per sel", "🔎", "T, h, dan apakah fallback Bartlett menyala — dilaporkan per pasangan.", "_cell_diagnostics"),
        Section("Matriks pasangan", "🧮", "66 pasangan, statistik disebut namanya per pasangan, T dicantumkan di samping tiap p.", "pair_matrix"),
        Section("Tabel MCS", "📋", "Kolom keanggotaan untuk Tabel 4 dan Tabel 6.", "mcs_table"),
    ),
    "economics.py": (
        Section("Header, biaya, band slippage", "📄", "Fee taker 0,04% per sisi, slippage 0,02/0,05/0,10% — band dipra-registrasi, ketiganya dilaporkan (root §13.5)."),
        Section("Posisi & return bersih", "📊", "Tanda ramalan kumulatif H-langkah pada return <strong>mentah bebas-drift</strong>, non-overlapping, fase 00:00 UTC.", "StrategyResult"),
        Section("Max drawdown & bootstrap-nya", "📉", "Dari ~180 observasi MDD tak terinterpretasi tanpa interval; bootstrap stasioner memasoknya.", "max_drawdown"),
        Section("Ringkasan strategi", "🧾", "Sharpe, Sortino, MDD, turnover — masing-masing dengan interval.", "summarise"),
        Section("Menjalankan strategi & buy-and-hold", "🏃", "Periode holding yang melintasi break <strong>dilewati</strong>: return lintas-gap tidak terdefinisi (<code>D46</code>).", "run_strategy"),
        Section("Uji Jobson–Korkie–Memmel", "⚖️", "Selisih Sharpe terhadap strategi naif. p ≈ 0,53 untuk <code>itr-K8</code>.", "jobson_korkie_memmel"),
        Section("Deflated Sharpe Ratio", "🎈", "Dihitung <strong>per origin</strong> dari Sharpe per-periode, bukan yang disetahunkan — memberi yang disetahunkan menggelembungkannya √(periode per tahun).", "deflated_sharpe"),
        Section("Run id per origin", "🔖", "Konfigurasi yang bersaing pada rentang uji origin itu — N untuk DSR.", "_origin_run_ids"),
        Section("Tabel ekonomi", "📋", "Tabel 8, ketiga tingkat slippage. Tiga angka dilaporkan bersama: strategi, buy-and-hold, DSR.", "economics_table"),
        Section("Kurva ekuitas", "📈", "Figure 7, sebelum dan sesudah biaya.", "equity_curves"),
    ),
    "attention.py": (
        Section("Header & konstanta", "📄", "Tercile dan ukuran batch penangkapan."),
        Section("Volatilitas lookback & tercile", "🌡️", "Calm dan stress ditentukan <strong>data</strong>, bukan dipilih setelah melihat petanya (<code>D48</code>).", "lookback_volatility"),
        Section("Penangkap batch", "🎥", "Atribut runtime, <strong>tidak pernah</strong> field config — cabangnya tidak mengonsumsi RNG, jadi run tertangkap identik bit-per-bit (<code>D62d</code>).", "_capture_batch"),
        Section("Peta attention per tercile", "🗺️", "Masukan Figure 5. Terukur: praktis uniform, dan kontras calm-vs-stress tenggelam di derau seed.", "tercile_maps"),
    ),
    "runner.py": (
        Section("Header, arm, konstanta sesi", "📄", "Sepuluh arm, dan <code>SESSION_BUDGET_H = 11,0</code> dengan cadangan setengah jam."),
        Section("Sel run", "🔲", "Satu sel grid, dan <code>run_id</code> deterministiknya. Mengubah komponen mana pun <strong>meng-orphan</strong> keluaran lama, bukan diam-diam memakainya ulang.", "RunCell"),
        Section("Manifes 969 run", "📜", "684 konfirmatori, 210 robustness <code>D62</code>, 75 baseline yang <code>D64</code> bangun. Irisan H=24 sweep dideduplikasi terhadap grid utama (<code>D53e</code>).", "manifest"),
        Section("Penemuan & resume", "🔍", "Ditemukan lewat glob, <strong>tidak pernah lewat slug Dataset yang dikodekan mati</strong>, jadi nama Dataset Kaggle bebas berubah.", "discover_roots"),
        Section("Penjaga anggaran sesi", "⏰", "Diperiksa di batas run, bukan batas epoch. <code>SESSION_T0</code> distempel di sel 0 supaya prelude ikut terhitung (<code>D54f</code>).", "BudgetGuard"),
        Section("Cache tensor per origin", "🗄️", "Tensor origin dimuat sekali dan dipakai ulang lintas rung dan seed.", "_TensorCache"),
        Section("Ringkasan eksekusi & penyelarasan", "🧾", "Selesai, dilewati, gagal — dan asersi keselarasan sebelum satu pun perbandingan dibentuk.", "ExecutionSummary"),
        Section("Eksekutor grid", "🚀", "Berurutan pada satu <code>cuda:0</code>. Terukur pada manifes 894-run: 0 gagal, 7,79 jam, rata-rata 31,8 s/run.", "execute"),
        Section("Dua GPU, tingkat-run", "🖥️", "Satu worker per device dari satu antrean bersama. <strong>Tingkat-run, tidak pernah tingkat-batch</strong> — <code>DataParallel</code> ditolak root §10.3 dengan alasan terukur, dan DDP lebih buruk lagi untuk 969 run ~32 detik. Determinisme dijaga: seed ber-scope device, prolog di bawah satu lock (<code>D68</code>).", "visible_devices"),
        Section("Grid tuning & pemilih konfigurasi", "🎛️", "Delapan belas konfigurasi, <strong>dideklarasikan sebelum dijalankan</strong>, diranking pada sub-blok <strong>validasi</strong> origin 1 — persis tempat <code>D27</code> menaruh gerbang Stage 5, dan karena alasan yang sama. Menjawab satu-satunya serangan yang §6.2 biarkan terbuka: <em>kalian tidak mencoba</em> (<code>D70</code>).", "TUNING_GRID"),
        Section("Pilot Stage 5", "🚧", "Berjalan di <strong>validasi</strong>, tidak pernah di uji (<code>D27</code>). Hasilnya <code>S* = +0,8759, p = 0,1906</code> — gerbang <strong>gagal</strong>, dan judul direposisi.", "PilotResult"),
        Section("Frame fitur", "🧊", "Satu pemanggilan yang notebook pakai untuk membangun frame fitur.", "build_feature_frame"),
    ),
    "report.py": (
        Section("Header & konstanta", "📄", "Nama model, kunci perbandingan, dan tag robustness."),
        Section("Format angka & LaTeX", "🔤", "Satu jalur format, jadi tiap tabel <code>.tex</code> memakai presisi yang sama.", "fmt"),
        Section("Helper kecil", "🧰", "Digest, standard error lintas origin, dan penanda signifikansi.", "_sha256"),
        Section("Input laporan & provenance", "🔐", "<code>paper/paper_numbers.json</code> menyebut berkas grid lewat sha256, jadi keduanya tidak bisa diam-diam berbeda (<code>D60f</code>, <code>D62a</code>).", "ReportInputs"),
        Section("Bagian dataset & arsitektur", "🧱", "Di sinilah 0 dari 444 run menyentuh cap epoch terlihat (<code>D62c</code>).", "_dataset_section"),
        Section("Bagian horizon & robustness", "🔭", "Sweep horizon dibatasi ke empat origin bernama di <strong>setiap</strong> horizon supaya kolomnya berbagi sampel.", "_horizon_section"),
        Section("Memuat peta attention", "🗺️", "Membaca 45 parquet attention arm <code>D62d</code>.", "_load_attention"),
        Section("build_report", "🏗️", "Pengumpul tunggal. Setiap angka manuskrip lahir di sini, lalu tabel dan figure di-render <em>dari</em> berkas itu — tidak pernah disalin tangan (root §12).", "build_report"),
        Section("paper_numbers.json", "💾", "Menulis berkas yang seluruh deliverable baca.", "build_paper_numbers"),
        Section("Tabel 1 & 2", "📋", "Profil dataset dan gap; deskriptif plus ADF, VR, Hurst.", "_table1"),
        Section("Tabel 2b & 3", "📋", "Eigenspektrum dan PR per rung per origin; hiperparameter dan K seluruh model.", "_table2b"),
        Section("Tabel 4 & 5", "📋", "Hasil utama dengan SE <strong>lintas origin</strong> (<code>D30</code>); <code>D(i,b)</code> per blok yang membawa kata undefined (<code>D60b</code>).", "_table4"),
        Section("Tabel 6", "📋", "Matriks DM — statistik disebut namanya per pasangan, T dicantumkan, kolom Romano–Wolf dan MCS.", "_table6"),
        Section("Tabel 7, 8, dan render", "📋", "Sweep horizon dan ekonomi, lalu seluruhnya ditulis sebagai <code>.tex</code>.", "_table7"),
        Section("Helper plot", "🎨", "Impor matplotlib ditunda dan penyimpan PDF/PNG.", "_pyplot"),
        Section("Figure 2b, 3, 4", "📈", "Figure 3 memikul seluruh paper: <strong>satu</strong> seri, lima belas garis per-origin, dengan MDE digambar di samping fit-nya (<code>D36</code>).", "_figure2b"),
        Section("Figure 5 & 6", "📈", "Peta attention per tercile dan sensitivitas horizon.", "_figure5"),
        Section("Figure 7 & render", "📈", "Kurva ekuitas di tiga tingkat slippage, lalu seluruhnya ditulis ke disk.", "_figure7"),
    ),
}


def _module_level_import_lines(source: str) -> set[int]:
    """1-based lines of every **module-level** import in ``source``.

    Module level only. ``report._pyplot`` imports matplotlib inside the function
    on purpose — the package must import cleanly without a plotting backend — and
    a walk over the whole tree would hoist that decision into a cell that runs at
    session start.
    """
    out: set[int] = set()
    for node in ast.parse(source).body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            out.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))
    return out


def flatten_module_body(name: str) -> str:
    """The flattened module with its module-level imports removed.

    The third declared subtractive category (`D66`). Every import the package
    makes is emitted once by :func:`library_cell` at the top of the notebook, so
    repeating them inside 137 definition cells is noise in the artefact a reader
    examines. This is what a module's cells must rejoin to, and
    ``tests/test_notebook_sync.py`` compares against it rather than against
    :func:`flatten_module_source` — the guarantee is unchanged, its reference
    moved.
    """
    source = flatten_module_source(name)
    lines = source.splitlines(keepends=True)
    dropped = _module_level_import_lines(source)

    # The blank lines that *followed* an import block go with it; the ones that
    # preceded it stay and become the separator. Done positionally rather than by
    # collapsing blank runs afterwards, because a docstring may legitimately hold
    # three blank lines in a row and a text-level rule would eat those too.
    for number in sorted(dropped):
        after = number + 1
        while after <= len(lines) and not lines[after - 1].strip():
            dropped.add(after)
            after += 1

    return "".join(
        line
        for number, line in enumerate(lines, start=1)
        if number not in dropped
    )


def _named_label(node: ast.stmt) -> str | None:
    """The module-level name a statement binds, or ``None`` if it binds nothing.

    Only these are eligible as a :class:`Section` anchor. Imports and the module
    docstring bind nothing a heading could sensibly point at, so they ride along
    with whichever section is open when they appear.
    """
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return node.name
    if isinstance(node, ast.Assign):
        names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        return names[0] if len(names) == 1 else None
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return node.target.id
    return None


def _first_line(node: ast.stmt) -> int:
    """1-based line the statement starts on, decorators included."""
    return min([d.lineno for d in getattr(node, "decorator_list", [])] + [node.lineno])


def _boundary(lines: list[str], first: int) -> int:
    """0-based cut point at or above ``first``, moved up past blanks and comments.

    A ``# -- section ---`` divider or a leading explanatory comment belongs to the
    definition it introduces, and the AST cannot say so. Walking up puts it at the
    top of the new cell instead of orphaning it at the bottom of the previous one.
    The cut only ever moves *up*, so the ranges stay contiguous and exhaustive.
    """
    index = first - 1
    while index > 0:
        previous = lines[index - 1].strip()
        if previous and not previous.startswith("#"):
            break
        index -= 1
    return index


def split_module_cells(name: str) -> list[tuple[Section, str]]:
    """One ``(section, source)`` pair per cell, covering the module exactly once.

    The pairs partition :func:`flatten_module_source` — every line lands in
    exactly one cell, in order — so ``"".join(source for _, source in ...)``
    reproduces the flattened module byte for byte. ``tests/test_notebook_sync.py``
    asserts that, and it is the whole reason the cells carry their identity in
    ``cell.metadata`` rather than in a banner comment the join would have to
    strip (`D63`).

    **Nothing is added.** A cell is a slice and only a slice. The
    ``from __future__ import annotations`` these modules open with lives in the
    library cell alone (`D66`, `D67`): IPython accumulates ``__future__`` compiler
    flags across a session, so the directive reaches every later cell without
    being repeated in any of them. Measured on IPython 9.13 --- ``compile.flags``
    goes ``16896 -> 16794112`` after that cell, and a following cell with no
    future import of its own defines a forward annotation without raising.
    """
    sections = SECTION_MAP.get(name)
    if not sections:
        raise ValueError(
            f"{name} has no SECTION_MAP entry, so it would ship as one "
            f"unsegmented cell while every other module is readable. Add its "
            f"sections beside its MODULE_ORDER entry."
        )
    if sections[0].starts_at is not None:
        raise ValueError(f"{name}: the first section must start at line one")

    source = flatten_module_body(name)
    lines = source.splitlines(keepends=True)
    tree = ast.parse(source)
    anchors = {
        label: _first_line(node)
        for node in tree.body
        if (label := _named_label(node)) is not None
    }

    cuts: list[int] = [0]
    for section in sections[1:]:
        anchor = section.starts_at
        if anchor not in anchors:
            raise ValueError(
                f"{name}: section {section.title!r} anchors on {anchor!r}, which "
                f"is not a module-level definition there. A stale anchor would "
                f"silently merge two sections into one cell."
            )
        cut = _boundary(lines, anchors[anchor])
        if cut <= cuts[-1]:
            raise ValueError(
                f"{name}: section {section.title!r} anchors on {anchor!r}, which "
                f"comes at or before the previous section. SECTION_MAP has to "
                f"follow the module's own order."
            )
        cuts.append(cut)
    cuts.append(len(lines))

    cells = [
        (section, "".join(lines[start:end]))
        for section, start, end in zip(sections, cuts, cuts[1:])
    ]
    empty = [s.title for s, body in cells if not body.strip()]
    if empty:
        raise ValueError(
            f"{name}: sections {empty} hold nothing once the imports move to the "
            f"library cell (`D66`). Merge them into a neighbour rather than shipping a "
            f"cell whose heading introduces an empty body."
        )
    assert "".join(body for _, body in cells) == source, (
        f"{name}: the cells no longer partition the module. This is the one "
        f"invariant the notebook's traceability rests on."
    )
    return cells


def _html_module(name: str) -> str:
    """The ``##``-level banner opening a module's run of cells."""
    start, end, accent, head, body = MODULE_THEME[name]
    return (
        f'<div style="background: linear-gradient(135deg, {start}, {end}); '
        f'border-radius: 14px; padding: 26px 32px; margin-bottom: 8px; '
        f'border: 1px solid {accent}33;">\n'
        f'  <h2 style="color: {head}; margin: 0 0 8px 0; font-size: 1.6em; '
        f'font-weight: 700; letter-spacing: 0.5px;">\n'
        f"    📘 {name}\n"
        f"  </h2>\n"
        f'  <p style="color: {body}; margin: 0; font-size: 1.0em;">'
        f"{MODULE_BLURB[name]}</p>\n"
        f"</div>"
    )


def _html_section(name: str, section: Section) -> str:
    """The ``###``-level banner directly above one code cell."""
    start, end, accent, head, body = MODULE_THEME[name]
    return (
        f'<div style="background: linear-gradient(90deg, {start}, {end}); '
        f"border-left: 4px solid {accent}; border-radius: 8px; "
        f'padding: 14px 20px;">\n'
        f'  <h3 style="color: {head}; margin: 0 0 6px 0; font-size: 1.15em;">'
        f"{section.emoji} {section.title}</h3>\n"
        f'  <p style="color: {body}; margin: 0; font-size: 0.94em;">'
        f"{section.blurb}</p>\n"
        f"</div>"
    )


def _html_step(emoji: str, title: str, blurb: str, theme: str) -> str:
    """A ``###``-level banner for an orchestration cell, themed like a module."""
    start, end, accent, head, body = MODULE_THEME[theme]
    return (
        f'<div style="background: linear-gradient(90deg, {start}, {end}); '
        f"border-left: 4px solid {accent}; border-radius: 8px; "
        f'padding: 14px 20px;">\n'
        f'  <h3 style="color: {head}; margin: 0 0 6px 0; font-size: 1.15em;">'
        f"{emoji} {title}</h3>\n"
        f'  <p style="color: {body}; margin: 0; font-size: 0.94em;">{blurb}</p>\n'
        f"</div>"
    )


MD_MODULE_NAMES = _html_step(
    "🧾",
    "Inventaris modul",
    "Delapan belas nama yang sel-sel di atas definisikan, dicatat supaya sel "
    "provenance dapat menyebut apa yang sebenarnya berjalan.",
    "__init__.py",
)

MD_PROVENANCE = _html_step(
    "🔐",
    "Provenance kode & input",
    "Root §12 meminta tiap run menyebut kode yang menghasilkannya. "
    "<code>git_sha</code> berbunyi unknown di Kaggle, jadi <code>code_sha256</code> "
    "yang memikulnya (<code>D54b</code>).",
    "train.py",
)

MD_RQ1 = _html_step(
    "1️⃣",
    "RQ1 — K nominal atau K_eff?",
    "ΔMSE per rung, TOST terhadap margin pra-registrasi, dan uji-J non-nested "
    "yang memacu kedua penjelasan (<code>D32</code>, <code>D49</code>).",
    "metrics.py",
)

MD_RQ2 = _html_step(
    "2️⃣",
    "RQ2 — apakah gap menyempit seiring umur model?",
    "β₁ dengan origin fixed effects, klaster per origin, WCR B = 99.999 — dan "
    "MDE dicetak di sampingnya, karena null tanpa daya tidak berarti apa-apa "
    "(root §9.2 syarat 6).",
    "metrics.py",
)

MD_RQ3 = _html_step(
    "3️⃣",
    "RQ3 — cadence retraining optimal?",
    "Bercabang ke <strong>undefined</strong>, tidak pernah ke no decay detected: "
    "frasa kedua adalah bentuk right-censored dan ia menyiratkan edge yang data "
    "ini tidak punya (<code>D55</code>, <code>D60b</code>).",
    "metrics.py",
)


MD_LIBRARY = _html_step(
    "📚",
    "Library",
    "Setiap impor yang paket ini pakai, dimuat sekali di satu tempat. Sel-sel "
    "definisi di bawah <strong>tidak membawa satu impor pun</strong> — termasuk "
    "<code>from __future__ import annotations</code>, karena IPython mengakumulasi "
    "flag <code>__future__</code> lintas sel dan direktif di sini berlaku untuk "
    "seluruh sesi (<code>D67</code>). Satu-satunya sel lain yang mengimpor adalah "
    "Setup di atas, dan itu tidak bisa dihindari: ia yang <em>memasang</em> paket "
    "yang sel ini impor, lalu memeriksanya lewat laporan perangkat.",
    "__init__.py",
)


def library_cell() -> str:
    """Every module-level import the package makes, deduplicated and grouped.

    Read from the flattened modules rather than typed out, so a dependency that
    appears in ``src/`` cannot go missing here. Intra-package imports are already
    gone by the time :func:`flatten_module_source` returns, which is why this can
    take the whole set without filtering.
    """
    # What the evaluation cells need and no module imports. ``gc`` is the grid
    # cell's, called between runs so a finished model's tensors are freed before
    # the next one allocates.
    plain: set[str] = {"import gc"}
    grouped: dict[str, set[tuple[str, str | None]]] = {}
    for name in MODULE_ORDER:
        tree = ast.parse(flatten_module_source(name))
        for node in tree.body:
            if isinstance(node, ast.Import):
                plain.add(ast.unparse(node))
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                grouped.setdefault(module, set()).update(
                    (a.name, a.asname) for a in node.names
                )

    def render(module: str) -> str:
        names = ", ".join(
            f"{n} as {alias}" if alias else n
            for n, alias in sorted(grouped[module])
        )
        return f"from {module} import {names}"

    modules = sorted(m for m in grouped if m != "__future__")
    lines = (
        ([render("__future__")] if "__future__" in grouped else [])
        + sorted(plain)
        + [render(m) for m in modules]
    )
    return "\n".join(lines) + "\n"


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

MD_TUNE = _html_step(
    "🎛️",
    "Pemilihan konfigurasi pada validasi",
    "Delapan belas konfigurasi diranking pada sub-blok validasi origin 1, lalu "
    "pemenangnya dilatih penuh di lima belas origin sebagai arm <code>itrt</code>. "
    "Grid-nya dideklarasikan di <code>TUNING_GRID</code> sebelum dijalankan; ruang "
    "pencarian yang dipilih setelah melihat pemenangnya bukan pencarian. Arm ini "
    "<strong>eksploratori</strong>, tidak masuk perbandingan tangga RQ1, dan "
    "jumlah percobaannya masuk hitungan development trial root §13.5 "
    "(<code>D70</code>).",
    "model.py",
)

CODE_TUNE = r'''TUNED_CONFIG, TUNING_TABLE = tune_on_validation(
    features, origin_index=1, k=8, device=device,
    log=lambda msg: print(msg, flush=True),
)
GRID_CONFIGS = {"tuned": TUNED_CONFIG}

(ARTIFACTS / "meta").mkdir(parents=True, exist_ok=True)
(ARTIFACTS / "meta" / "tuning_selection.json").write_text(
    json.dumps({"grid": list(TUNING_GRID), "ranked": TUNING_TABLE,
                "selected": {"d_model": TUNED_CONFIG.d_model,
                             "e_layers": TUNED_CONFIG.e_layers}}, indent=1),
    encoding="utf-8",
)
print(f"tuned arm will run {TUNED_CONFIG.d_model=} {TUNED_CONFIG.e_layers=}")
'''

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

CODE_SETUP = r'''import os
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


# torch and the data plane run the grid; the rest are the reporting pass, which
# root section 16 confines to one named stats boundary. Kaggle ships scipy,
# statsmodels and matplotlib and usually not `arch`, so this list is the whole
# difference between a session that renders Table 2 and one that raises on it.
for _mod in ("polars", "pyarrow", "numpy", "torch",
             "scipy", "statsmodels", "arch", "matplotlib"):
    ensure(_mod)


def looks_like_parquet(path: Path) -> bool:
    """True only for a file that begins and ends with the parquet magic.

    Four bytes at each end. A parquet file opens with ``PAR1`` and closes with
    ``PAR1`` after its footer, so this catches a truncated upload, a Git LFS
    pointer, an HTML error page saved under the right name, and anything else
    that merely occupies the path.

    It exists because the alternative is what happened: something that was not a
    parquet was accepted here, and the failure surfaced three cells later as
    ``ComputeError: File out of specification`` from inside polars, naming
    neither the file nor the reason it was chosen. A check at the point of
    selection can say both.
    """
    try:
        with path.open("rb") as handle:
            if handle.read(4) != b"PAR1":
                return False
            handle.seek(-4, 2)
            return handle.read(4) == b"PAR1"
    except OSError:
        return False


def find_parquet() -> Path:
    """Locate BTCUSDT_1h.parquet by globbing — never by Kaggle dataset slug.

    Root section 10.5: discovery is by glob so the Dataset can be renamed without
    editing anything. The shallow patterns come first because they express a
    *preference* -- data/raw/ over a flat upload -- and because they are cheap.

    The recursive pass exists because a fixed depth is a hard-coded assumption
    wearing a glob. Kaggle mounts a dataset at /kaggle/input/<slug>/, but the
    path a user copies out of the web UI can carry the owner and the datasets/
    segment too, and a repository uploaded whole nests data/raw/ one level
    deeper again. Any of those is three or four levels down, and the previous
    patterns stopped at two -- so the file was there and discovery said it was
    not. Depth is not a thing to enumerate.
    """
    patterns = (
        "data/raw/BTCUSDT_1h.parquet",
        "*/data/raw/BTCUSDT_1h.parquet",
        "BTCUSDT_1h.parquet",
        "*/BTCUSDT_1h.parquet",
        "*/*/BTCUSDT_1h.parquet",
    )
    roots = [WORK, Path("/kaggle/input")] if ON_KAGGLE else [WORK, WORK.parent]
    rejected: list[str] = []

    def accept(candidate: Path):
        """The candidate, or None once it has failed the magic check."""
        if candidate.is_file() and looks_like_parquet(candidate):
            return candidate.resolve()
        rejected.append(str(candidate))
        return None

    for root in roots:
        if not root.exists():
            continue
        for pattern in patterns:
            for hit in sorted(root.glob(pattern)):
                if (found := accept(hit)) is not None:
                    return found
    # Depth-independent fallback. Preferring data/raw/ still, then anything.
    for root in roots:
        if not root.exists():
            continue
        hits = [h for h in sorted(root.rglob("BTCUSDT_1h.parquet")) if h.is_file()]
        valid = [h for h in hits if looks_like_parquet(h)]
        rejected += [str(h) for h in hits if h not in valid]
        if valid:
            preferred = [h for h in valid if h.parent.name == "raw"]
            chosen = (preferred or valid)[0]
            if len(valid) > 1:
                print(f"note: {len(valid)} copies of BTCUSDT_1h.parquet under "
                      f"{root}; using {chosen}. Section 12 forbids two vintages "
                      f"in one table -- check they are the same file.")
            return chosen.resolve()

    if rejected:
        raise FileNotFoundError(
            "found candidates but none is a parquet file -- each failed the "
            f"PAR1 magic check at one or both ends: {rejected}. A truncated "
            "upload, a Git LFS pointer, or the wrong file under the right name."
        )
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

CODE_GRID = r'''
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
# Both GPUs, one worker each, off a shared queue (`D68`). The note that used to
# stand here said threads were not the fix because torch.manual_seed seeds EVERY
# CUDA device — true of the seeding as it was, and that is what changed:
# set_seed now takes the device and seeds only that one, and the CPU generator is
# shared under a single lock across seeding and module construction. Parallelism
# stays at the RUN level; root §10.3 rejects nn.DataParallel with a measured
# reason, and DDP pays a process group per ~32 s run.
DEVICES = visible_devices()
print(f"devices for the grid: {[str(d) for d in DEVICES]}")

t0 = time.perf_counter()
summary = execute_parallel(
    todo, features,
    devices=DEVICES,
    configs=GRID_CONFIGS,
    out_root=ARTIFACTS,
    roots=roots,
    guard=BudgetGuard(budget_h, 0.5),
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

CODE_SAVE = r'''# ``_digest_source``, never ``_provenance``: in a flattened notebook every module
# shares one kernel namespace, and ``report._provenance`` is a *function*
# ``build_report`` calls one cell later. Binding that name to a string here
# killed the last cell of the 894-run session with ``TypeError: 'str' object is
# not callable`` --- after the grid had run for 7.8 hours and every artifact was
# already safe on disk. `D59` from the opposite direction: not an import the
# flattening dropped, but a package-private name an evaluation cell shadowed (`D65`).
_digest, _digest_source = _input_sha256(PARQUET)

paper_numbers = {
    "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "input_parquet": str(PARQUET),
    "input_sha256": _digest,
    "input_sha256_source": _digest_source,
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


MD_REPORT = """<div style="background: linear-gradient(90deg, #001a0d, #003317); border-left: 4px solid #52b788; border-radius: 8px; padding: 18px 24px;">
  <h2 style="color: #52b788; margin: 0 0 8px 0;">&#128202; 9 &middot; Tables and figures</h2>
  <p style="color: #b8c7e0; margin: 0;">Everything section 13.4 promises, rendered from
    <code>paper_numbers.json</code> and never transcribed.</p>
  <p style="color: #95d5b2; margin: 10px 0 0 0; font-size: 0.92em;">This is the cell
    <code>D60g</code> was about. The grid produced the numbers and left every table and figure
    ungenerated; four of them &mdash; the DM matrix, the economic evaluation, the attention
    heatmap and the equity curve &mdash; had no inputs at all. Three of those four are computed
    here from the 684 prediction files already on disk. <b>Figure 5 is the exception</b>: attention
    weights were never persisted, so it needs the <code>attention</code> arm and is skipped
    <i>by name</i> until that arm runs &mdash; an empty axes labelled as a figure reads as a
    measurement of nothing.</p>
</div>"""


CODE_REPORT = r'''PAPER = WORK / "paper"

# The manuscript's single source. The grid's own paper_numbers.json written above
# stays immutable evidence; this reads it, adds every analysis pass the grid never
# ran — section 4.5's efficiency tests, the DM/Romano-Wolf/MCS matrix, the
# economic evaluation, directional accuracy, the horizon aggregation, the raw
# metric scale, D60i's RelMSE falsification gap and D45's coverage check — and
# names the grid file by digest so the two cannot silently diverge (root §12).
report_inputs = build_report(
    ARTIFACTS,
    bars,
    features,
    roots=discover_roots(ARTIFACTS),
    bootstrap_b=9_999,
    seed=42,
    log=lambda message: print(message, flush=True),
)

PAPER.mkdir(parents=True, exist_ok=True)
numbers_path = PAPER / "paper_numbers.json"
numbers_path.write_text(json.dumps(report_inputs.numbers, indent=2, default=float))
print(f"\nwrote {numbers_path}")

print("\ntables:")
for path in render_tables(report_inputs.numbers, PAPER / "tables"):
    print(f"  {path.name}")

print("\nfigures:")
for path in render_figures(report_inputs, PAPER / "figures", log=print):
    print(f"  {path.name}")

# The frames a figure reads, persisted beside the numbers so a plot can be redone
# without re-running the whole aggregation. Figure 2b alone is ~3,000 points and
# has no business inside a JSON file a human is expected to read.
panels = PAPER / "panels"
panels.mkdir(parents=True, exist_ok=True)
for _name, _frame in (
    ("seed_averaged_cells", report_inputs.seed_avg),
    ("amplification_panel", report_inputs.amplification),
    ("rolling_pr", report_inputs.rolling_pr),
    ("rolling_ols_r2", report_inputs.rolling_r2),
    ("equity_curves", report_inputs.equity),
):
    _frame.write_parquet(panels / f"{_name}.parquet")
if report_inputs.attention is not None:
    report_inputs.attention.write_parquet(panels / "attention_maps.parquet")
print(f"\nwrote panels to {panels}")

_robustness = report_inputs.numbers["robustness"]
print("\nD62 robustness arms:")
for _arm, _state in _robustness.items():
    print(f"  {_arm:12s} {_state['status']}")
print(
    "\nAn arm reading 'not run' is a pre-registered exploratory arm awaiting its "
    "runs, not a failure. Whatever it returns goes in the paper: an arm reported "
    "only when it agrees with the headline is not a robustness arm (root §13.2)."
)
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


def _code(index: int, text: str, metadata: dict | None = None) -> dict:
    """A code cell, optionally tagged with the module and section it came from.

    The tag is what ``tests/test_notebook_sync.py`` groups by. Carrying it in
    ``metadata`` rather than in a banner comment is what keeps the cell source a
    byte-exact slice of the module, so the concatenation check needs no
    normalisation step (`D63`).
    """
    return {
        "id": f"code-{index:02d}",
        "cell_type": "code",
        "execution_count": None,
        "metadata": dict(metadata or {}),
        "outputs": [],
        "source": _lines(text),
    }


def build() -> dict:
    """The whole notebook, as an nbformat 4.5 dictionary."""
    cells: list[dict] = []
    counter = 0

    def md(text: str) -> None:
        nonlocal counter
        cells.append(_markdown(counter, text))
        counter += 1

    def code(text: str, metadata: dict | None = None) -> None:
        nonlocal counter
        cells.append(_code(counter, text, metadata))
        counter += 1

    md(MD_TITLE)
    md(MD_SETUP)
    code(CODE_SETUP)

    md(MD_LIBRARY)
    code(library_cell(), {"itbtc": {"role": "library"}})

    md(MD_DEFINE)
    for name in MODULE_ORDER:
        md(_html_module(name))
        for section, body in split_module_cells(name):
            md(_html_section(name, section))
            code(body, {"itbtc": {"module": name, "section": section.title}})
    md(MD_MODULE_NAMES)
    code("MODULE_NAMES = [\n" + "".join(
        f"    {name!r},\n" for name in MODULE_ORDER) + "]\n")
    md(MD_PROVENANCE)
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
    md(MD_TUNE)
    code(CODE_TUNE)

    md(MD_GRID)
    code(CODE_GRID)
    md(MD_EVAL)
    md(MD_RQ1)
    code(guarded(CODE_RQ1, "RQ1"))
    md(MD_RQ2)
    code(guarded(CODE_RQ2, "RQ2"))
    md(MD_RQ3)
    code(guarded(CODE_RQ3, "RQ3"))
    md(MD_SAVE)
    code(guarded(CODE_SAVE, "paper_numbers.json"))
    md(MD_REPORT)
    code(guarded(CODE_REPORT, "tables and figures"))

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

    unsegmented = [n for n in MODULE_ORDER if n not in SECTION_MAP]
    if unsegmented:
        raise SystemExit(
            f"{unsegmented} have MODULE_ORDER entries but no SECTION_MAP ones, "
            f"so each would ship as one unsegmented cell while every other "
            f"module is readable. Add their sections here (`D63`)."
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
