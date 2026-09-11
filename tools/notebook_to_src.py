"""Export the primary notebooks/iTransformer.ipynb to its tested src projection.

Invoke through the notebook's final, locally enabled sync cell. The older
build_notebook template must not overwrite notebook edits. Its flattening helpers
are reused only to restore existing imports and verify an exact body round trip.

New imports belong in the notebook Library cell and module metadata
itbtc.projection_imports. Remove obsolete package imports through
projection_remove_imports. Successful export refreshes the artifact map, pinned
package digest and ordered notebook program digest, invalidating stale outputs.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import importlib.util
import hashlib
import json
import re
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
    # This map is derived from notebook metadata, never from the legacy template.
    for cell in notebook["cells"]:
        if cell.get("metadata", {}).get("itbtc", {}).get("step") == "artifact_map":
            source = "".join(cell["source"])
            node = next(n for n in ast.parse(source).body if isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == "_ARTIFACT_MAP" for t in n.targets))
            rows = []
            for index, item in enumerate(notebook["cells"]):
                tag = item.get("metadata", {}).get("itbtc", {})
                if tag.get("role") == "step":
                    rows.append((index, tag["step"], tag.get("writes", []), tag.get("reads", [])))
            lines = source.splitlines(keepends=True)
            lines[node.lineno-1:node.end_lineno] = ["_ARTIFACT_MAP = [\n" + "".join(f"    {row!r},\n" for row in rows) + "]\n"]
            cell["source"] = "".join(lines).splitlines(keepends=True)
    grouped = cells_by_module(notebook)

    missing = [m for m in generator.MODULE_ORDER if m not in grouped]
    if missing:
        print(
            f"refusing to write: the notebook has no cells for {missing}. A "
            f"module that lost its cells would be truncated to nothing here, "
            f"which is silent and total. Restore the missing notebook cells from version history first.",
            file=sys.stderr,
        )
        return 1

    changed: list[str] = []
    for module in generator.MODULE_ORDER:
        body = "".join(grouped[module])

        offending = _module_level_imports_in(body, module)
        if offending:
            print(
                f"refusing to write {module}: its cells carry module-level "
                f"imports {offending}. Imports live in the Library cell, which "
                f"is generated *from* the modules (`D66`), so a cell has no "
                f"import lines to edit and these are text this script cannot "
                f"place. Use a function-local external import, or declare a "
                f"package-only import in metadata.itbtc.projection_imports.",
                file=sys.stderr,
            )
            return 1

        target = generator.PACKAGE / module
        original = target.read_text(encoding="utf-8")
        rebuilt = rebuild_module(generator, module, body)
        # Notebook metadata declares imports needed only by the exported package.
        # In the notebook, sibling functions already share the kernel namespace.
        extra = []
        remove_imports = set()
        for cell in notebook["cells"]:
            tag = cell.get("metadata", {}).get("itbtc", {})
            if tag.get("module") == module:
                extra.extend(tag.get("projection_imports", []))
                remove_imports.update(tag.get("projection_remove_imports", []))
        if remove_imports:
            lines = rebuilt.splitlines(keepends=True)
            for node in reversed(ast.parse(rebuilt).body):
                if isinstance(node, (ast.Import, ast.ImportFrom)) and ast.unparse(node) in remove_imports:
                    del lines[node.lineno - 1:node.end_lineno]
            rebuilt = "".join(lines)
        tree = ast.parse(rebuilt)
        existing = {ast.unparse(n) for n in tree.body
                    if isinstance(n, (ast.Import, ast.ImportFrom))}
        additions = []
        for statement in extra:
            parsed = ast.parse(statement).body
            if len(parsed) != 1 or not isinstance(parsed[0], (ast.Import, ast.ImportFrom)):
                raise ValueError(f"{module}: invalid projection import {statement!r}")
            normal = ast.unparse(parsed[0])
            if normal not in existing:
                additions.append(normal + "\n")
                existing.add(normal)
        if additions:
            # External imports must join the external block. Appending one after
            # package imports would swallow a notebook separator during flattening.
            end = max(n.end_lineno for n in tree.body
                      if isinstance(n, (ast.Import, ast.ImportFrom))
                      and not generator._intra_package_import(n))
            lines = rebuilt.splitlines(keepends=True)
            lines[end:end] = additions
            rebuilt = "".join(lines)
        ast.parse(rebuilt, filename=module)
        if rebuilt == original:
            continue
        if dry_run:
            changed.append(module)
            continue

        target.write_text(rebuilt, encoding="utf-8", newline="\n")
        # Verify by re-flattening what was just written. Anything short of
        # byte-equality means the import block landed in the wrong place, and a
        # module that merely *looks* right is exactly the failure mode `D59`
        # cost a session to. Restore rather than leave it.
        if generator.flatten_module_body(module) != body:
            mismatch = list(difflib.unified_diff(
                body.splitlines(), generator.flatten_module_body(module).splitlines(),
                fromfile="notebook", tofile="projection", n=2))
            target.write_text(original, encoding="utf-8", newline="\n")
            print(
                f"refusing to write {module}: re-flattening what was written "
                f"did not reproduce the cells byte for byte, so the round trip "
                f"is not sound here. The file has been restored unchanged.",
                file=sys.stderr,
            )
            print("\n".join(mismatch[:40]), file=sys.stderr)
            return 1
        changed.append(module)

    if not dry_run:
        # The pinned digest is export metadata, never an excuse to rebuild the
        # user's notebook from the older presentation template (A13, A15).
        digest = generator.package_digest()
        for cell in notebook["cells"]:
            if cell.get("metadata", {}).get("itbtc", {}).get("step") == "code_digest":
                source = "".join(cell["source"])
                source = re.sub(r"CODE_SHA256_OVERRIDE = ['\"][0-9a-f]{64}['\"]",
                                f'CODE_SHA256_OVERRIDE = "{digest}"', source)
                cell["source"] = source.splitlines(keepends=True)
        program = json.dumps(["".join(c["source"]) for c in notebook["cells"]
                              if c.get("cell_type") == "code"], ensure_ascii=False)
        program_digest = hashlib.sha256(program.encode("utf-8")).hexdigest()
        if notebook["metadata"].get("itbtc_exported_program_sha256") != program_digest:
            for cell in notebook["cells"]:
                if cell.get("cell_type") == "code":
                    cell["outputs"] = []
                    cell["execution_count"] = None
        notebook["metadata"]["itbtc_exported_program_sha256"] = program_digest
        notebook_path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
                                 encoding="utf-8", newline="\n")

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


def check_projection(notebook_path: Path) -> int:
    """Validate the notebook and its projection without consulting old prose."""
    generator = _load_generator()
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    grouped = cells_by_module(notebook)
    errors = [m for m in generator.MODULE_ORDER
              if "".join(grouped.get(m, [])) != generator.flatten_module_body(m)]
    code = ["".join(c["source"]) for c in notebook["cells"]
            if c.get("cell_type") == "code"]
    for source in code:
        ast.parse(source, feature_version=(3, 11))
    pinned = [s for s in code if re.search(r"^CODE_SHA256_OVERRIDE =", s, re.M)]
    if len(pinned) != 1 or generator.package_digest() not in pinned[0]:
        errors.append("pinned package digest")
    program = hashlib.sha256(json.dumps(code, ensure_ascii=False).encode("utf-8")).hexdigest()
    if notebook.get("metadata", {}).get("itbtc_exported_program_sha256") != program:
        errors.append("ordered notebook program digest")
    if errors:
        print(f"Projection is stale: {errors}. Save the notebook, then use its final sync cell.",
              file=sys.stderr)
        return 1
    print("Notebook source, ordered program digest and exported src projection agree")
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
