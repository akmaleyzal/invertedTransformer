"""Write ``src/itransformer_btc/`` back from the notebook's cells (`D88`).

The forward direction has always existed: ``tools/build_notebook.py`` turns
eighteen modules into 367 cells. This is the return leg, and it exists because
root §1's deliverable made the notebook the artefact people actually open and
edit, while ``src/`` stayed the only place a change could be made. Editing a
cell and then re-typing it into a module is two copies kept in step by hand,
which is `D54a` and `D69` in a new costume.

**What this does not change.** ``src/`` remains what the tests import and what
``code_sha256`` hashes, so it is still the vintage every ``meta/*.json`` names.
The notebook is where you type; ``src/`` is the tested projection. Both
directions are policed by the same byte-exact comparison, and this script
refuses to write unless that comparison passes *after* the write.

**The one thing you cannot do from a cell: add an import.** Module-level imports
are stripped by the flattening and re-emitted once in the Library cell, which is
generated *from* the modules (`D66`). A cell has no import lines to edit, so an
import appearing in one is new text this script cannot place. It refuses,
loudly, and names ``src/`` as the place to add it. Everything else -- a
docstring, a constant, a function body, a whole new class -- flows back.

Usage::

    python tools/notebook_to_src.py notebooks/iTransformer.ipynb
    python tools/notebook_to_src.py notebooks/iTransformer.ipynb --dry-run
"""

from __future__ import annotations

import argparse
import ast
import difflib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "tools" / "build_notebook.py"


def _load_generator():
    """Import the forward generator as a module.

    Imported rather than duplicated: the flattening rules are subtle enough
    (`D63`, `D66`, `D67`) that a second implementation of them would be a second
    thing to keep correct, and the whole argument of this file is against that.
    """
    spec = importlib.util.spec_from_file_location("build_notebook", GENERATOR)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_notebook"] = module
    spec.loader.exec_module(module)
    return module


def cells_by_module(notebook: dict) -> dict[str, list[str]]:
    """Every definition cell's source, grouped by module, in notebook order."""
    grouped: dict[str, list[str]] = {}
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        tag = cell.get("metadata", {}).get("itbtc", {})
        if tag.get("role") == "module" or (
            tag.get("role") is None and tag.get("module")
        ):
            grouped.setdefault(tag["module"], []).append("".join(cell["source"]))
    return grouped


def dropped_runs(generator, module: str) -> dict[int, list[str]]:
    """Lines the flattening removed, keyed by where they sit in the body.

    The removed lines are not one block at the top. They are the module-level
    imports, **every intra-package import wherever it lives** — ``model.py``
    defers one inside a function on purpose — and ``runner.py``'s
    ``if __name__ == "__main__":`` guard, which in a cell would launch the whole
    grid because ``__name__`` *is* ``"__main__"`` there. Collecting them into a
    single header and pasting it at the top puts an indented import at module
    level, which is an ``IndentationError`` in the file this tool just wrote.

    So each removed run is anchored to the number of body lines that precede it,
    and :func:`rebuild_module` puts it back at that anchor. The anchors are
    found by **alignment**: the flattening only ever deletes, so the flattened
    body is a subsequence of the file and what the alignment does not consume is
    what was removed. That does not depend on re-deriving the flattening rules
    correctly — only on deletions being deletions.

    Returns:
        ``{body_lines_before: [removed lines]}``. The key ``len(body)`` holds
        anything trailing the last kept line.
    """
    lines = (generator.PACKAGE / module).read_text(encoding="utf-8").splitlines(
        keepends=True
    )
    body = generator.flatten_module_body(module).splitlines(keepends=True)

    runs: dict[int, list[str]] = {}
    cursor = 0
    for line in lines:
        if cursor < len(body) and line == body[cursor]:
            cursor += 1
        else:
            runs.setdefault(cursor, []).append(line)
    if cursor != len(body):
        raise SystemExit(
            f"{module}: its flattened body is not a subsequence of the file on "
            f"disk, so the flattening is doing more than deleting lines and "
            f"this reconstruction cannot be trusted. Refusing rather than "
            f"guessing."
        )
    return runs


def _map_anchor(old: list[str], new: list[str], anchor: int) -> int:
    """Where ``anchor`` in the old body lands in the new one.

    A removed line sits between two body lines; editing the body moves them. The
    anchor is carried across on the unchanged stretches, which is what keeps an
    import block above the code it serves after the code below it has been
    edited.
    """
    if anchor >= len(old):
        return len(new)
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
        None, old, new, autojunk=False
    ).get_opcodes():
        if i1 <= anchor < i2:
            return j1 + (anchor - i1) if tag == "equal" else j1
    return len(new)


def rebuild_module(generator, module: str, body: str) -> str:
    """The full ``src/`` text for ``module`` given its flattened body.

    Every removed run goes back at its anchor, so an unchanged body reproduces
    the file byte for byte and an edited one keeps its imports where they were.
    Byte-for-byte matters more than it sounds: rewriting an unchanged module
    into a *differently formatted* unchanged module moves ``code_sha256``, and
    root §12 hangs the traceability of all 1,620 runs on that number.
    """
    runs = dropped_runs(generator, module)
    old = generator.flatten_module_body(module).splitlines(keepends=True)
    new = body.splitlines(keepends=True)

    insertions: dict[int, list[str]] = {}
    for anchor in sorted(runs):
        target = _map_anchor(old, new, anchor)
        insertions.setdefault(target, []).extend(runs[anchor])

    out: list[str] = []
    for index in range(len(new) + 1):
        out.extend(insertions.get(index, []))
        if index < len(new):
            out.append(new[index])
    return "".join(out)


def _module_level_imports_in(body: str, module: str) -> list[str]:
    """Import statements a cell body carries, which no cell body may."""
    return [
        ast.unparse(node)
        for node in ast.parse(body, filename=module).body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]


def sync(notebook_path: Path, dry_run: bool = False) -> int:
    generator = _load_generator()
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    grouped = cells_by_module(notebook)

    missing = [m for m in generator.MODULE_ORDER if m not in grouped]
    if missing:
        print(
            f"refusing to write: the notebook has no cells for {missing}. A "
            f"module that lost its cells would be truncated to nothing here, "
            f"which is silent and total. Regenerate the notebook first.",
            file=sys.stderr,
        )
        return 1

    changed: list[str] = []
    for module in generator.MODULE_ORDER:
        body = "".join(grouped[module])
        if body == generator.flatten_module_body(module):
            continue

        offending = _module_level_imports_in(body, module)
        if offending:
            print(
                f"refusing to write {module}: its cells carry module-level "
                f"imports {offending}. Imports live in the Library cell, which "
                f"is generated *from* the modules (`D66`), so a cell has no "
                f"import lines to edit and these are text this script cannot "
                f"place. Add the dependency in src/ and rebuild the notebook.",
                file=sys.stderr,
            )
            return 1

        target = generator.PACKAGE / module
        original = target.read_text(encoding="utf-8")
        rebuilt = rebuild_module(generator, module, body)
        if dry_run:
            changed.append(module)
            continue

        target.write_text(rebuilt, encoding="utf-8", newline="\n")
        # Verify by re-flattening what was just written. Anything short of
        # byte-equality means the import block landed in the wrong place, and a
        # module that merely *looks* right is exactly the failure mode `D59`
        # cost a session to. Restore rather than leave it.
        if generator.flatten_module_body(module) != body:
            target.write_text(original, encoding="utf-8", newline="\n")
            print(
                f"refusing to write {module}: re-flattening what was written "
                f"did not reproduce the cells byte for byte, so the round trip "
                f"is not sound here. The file has been restored unchanged.",
                file=sys.stderr,
            )
            return 1
        changed.append(module)

    if not changed:
        print("src/itransformer_btc/ already matches the notebook; nothing written")
        return 0
    verb = "would rewrite" if dry_run else "rewrote"
    print(f"{verb} {len(changed)} module(s): {', '.join(changed)}")
    if not dry_run:
        print(
            "now run: python tools/build_notebook.py --check\n"
            "and commit src/ and the notebook together -- they are one change."
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Write src/itransformer_btc/ back from the notebook (`D88`)"
    )
    parser.add_argument(
        "notebook",
        nargs="?",
        default=str(ROOT / "notebooks" / "iTransformer.ipynb"),
        help="the notebook to read; defaults to the committed one",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report which modules would change and write nothing",
    )
    args = parser.parse_args(argv)

    path = Path(args.notebook)
    if not path.exists():
        raise SystemExit(f"{path} does not exist")
    return sync(path, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
