"""Render ``notebooks/iTransformer.ipynb`` as a navigable Markdown map.

CLAUDE.md section 15 makes the notebook the primary surface and ``src/`` its
tested projection. Nothing outside the notebook could say so: ``.ipynb`` is not
a format the documentation tooling or the knowledge graph reads, so the one
artefact an examiner opens was the one artefact with no entry anywhere. This
writes that entry.

What it emits is deliberately *not* the notebook's code. The definition cells
are byte-exact slices of ``src/itransformer_btc/`` -- ``D63``, enforced by
``tests/test_notebook_sync.py`` -- so transcribing them would duplicate every
module under a second name and say nothing new. What the notebook holds
uniquely is its *shape*: the phases, the orchestration steps that exist in no
module, and ``D87``'s ``reads``/``writes`` manifest naming which cell produces
which artefact. That is what this renders.

The metadata is read from the notebook itself rather than from
``tools/build_notebook.py``'s ``PHASES``/``SECTION_MAP`` constants, and the
reason is ``D55``: a generator constant describes the notebook someone meant to
build, while the file on disk is the notebook that ran. When they disagree the
file wins, and only reading the file can show the disagreement.

Usage::

    python tools/notebook_map.py            # writes docs/NOTEBOOK_MAP.md
    python tools/notebook_map.py --check    # exit 1 if the file is stale
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "notebooks" / "iTransformer.ipynb"
OUTPUT = ROOT / "docs" / "NOTEBOOK_MAP.md"

_TAG = re.compile(r"<[^>]+>")
_HEADING = re.compile(r"<h([1-4])[^>]*>(.*?)</h\1>", re.S | re.I)
_MARKDOWN_HEADING = re.compile(r"^(#{1,4})[ \t]+(.+)$", re.M)
_PARAGRAPH = re.compile(r"<p[^>]*>(.*?)</p>", re.S | re.I)


def _plain(markup: str) -> str:
    """Strip the banner HTML down to the text a reader would see."""
    return " ".join(html.unescape(_TAG.sub("", markup)).split())


@dataclass
class Heading:
    """One banner: its level, its title, and its blurb if it carries one."""

    level: int
    title: str
    blurb: str = ""


@dataclass
class Entry:
    """One code cell, with the banner that introduces it."""

    index: int
    itbtc: dict
    heading: Heading | None

    @property
    def role(self) -> str:
        return str(self.itbtc.get("role", "?"))


@dataclass
class Phase:
    """One ``##`` phase of the notebook and every code cell beneath it."""

    title: str
    blurb: str
    entries: list[Entry] = field(default_factory=list)


def parse(notebook: Path) -> tuple[list[Phase], Counter]:
    """Walk the notebook once, grouping code cells under their phase banner.

    Args:
        notebook: Path to the ``.ipynb`` file.

    Returns:
        The phases in document order, and a count of cells by cell type.
    """
    doc = json.loads(notebook.read_text(encoding="utf-8"))
    phases: list[Phase] = []
    pending: Heading | None = None
    counts: Counter = Counter()

    for index, cell in enumerate(doc["cells"]):
        counts[cell["cell_type"]] += 1
        if cell["cell_type"] == "markdown":
            source = "".join(cell["source"])
            native = _MARKDOWN_HEADING.search(source)
            found = native or _HEADING.search(source)
            if not found:
                continue
            paragraphs = _PARAGRAPH.findall(source)
            heading = Heading(
                level=len(found.group(1)) if native else int(found.group(1)),
                title=_plain(found.group(2)),
                blurb=_plain(paragraphs[0]) if paragraphs else "",
            )
            if heading.level <= 2:
                # h1 is the notebook title and h2 opens a phase; both start a
                # new bucket, so the h1's bucket collects whatever precedes the
                # first real phase instead of it being dropped on the floor.
                phases.append(Phase(title=heading.title, blurb=heading.blurb))
                pending = None
            else:
                pending = heading
            continue

        itbtc = cell.get("metadata", {}).get("itbtc")
        if itbtc is None:
            continue
        if not phases:
            phases.append(Phase(title="(sebelum banner pertama)", blurb=""))
        phases[-1].entries.append(Entry(index=index, itbtc=itbtc, heading=pending))
        pending = None

    return phases, counts


def _steps(phases: list[Phase]) -> list[tuple[Phase, Entry]]:
    """Every producing cell, paired with the phase it sits in."""
    return [(p, e) for p in phases for e in p.entries if e.role == "step"]


def _modules(phases: list[Phase]) -> dict[str, list[tuple[Phase, Entry]]]:
    """Definition cells grouped by the module they are a slice of."""
    out: dict[str, list[tuple[Phase, Entry]]] = {}
    for phase in phases:
        for entry in phase.entries:
            if entry.role == "module":
                key = str(entry.itbtc.get("module", "?"))
                out.setdefault(key, []).append((phase, entry))
    return out


def _paths(entry: Entry, key: str) -> str:
    """Format a ``reads``/``writes`` list as inline code, or an em dash."""
    values = entry.itbtc.get(key) or []
    return ", ".join(f"`{v}`" for v in values) if values else "—"


def render(phases: list[Phase], counts: Counter) -> str:
    """Build the Markdown map. Prose is Indonesian, matching the notebook."""
    steps = _steps(phases)
    modules = _modules(phases)
    # phases[0] is the h1 title banner, not a phase of the pipeline.
    numbered = phases[1:] if phases else []
    lines: list[str] = []
    add = lines.append

    add("# Peta notebook — `notebooks/iTransformer.ipynb`")
    add("")
    add(
        "**Digenerate oleh `tools/notebook_map.py`. Jangan disunting tangan** — "
        "jalankan ulang generatornya setelah `tools/build_notebook.py`."
    )
    add("")
    add(
        "`notebooks/iTransformer.ipynb` adalah **deliverable utama** repositori "
        "ini (`CLAUDE.md` §15): notebook mandiri yang membawa seluruh paket "
        "`src/itransformer_btc/` sebagai sel definisi, ditambah sel orkestrasi "
        "yang tidak ada di modul mana pun. `src/` adalah proyeksinya yang diuji, "
        "bukan sebaliknya. Sel definisi adalah irisan byte-exact dari modulnya "
        "(`D63`, dijaga `tests/test_notebook_sync.py`), sehingga peta ini "
        "menyebut modul yang diproyeksikan alih-alih mengulang kodenya."
    )
    add("")
    add(
        f"- Sel: **{sum(counts.values())}** "
        f"({counts.get('code', 0)} kode, {counts.get('markdown', 0)} markdown)"
    )
    add(f"- Fase: **{len(numbered)}**")
    add(f"- Langkah produksi (`role: step`): **{len(steps)}**")
    add(
        f"- Modul yang diproyeksikan: **{len(modules)}** dalam "
        f"{sum(len(v) for v in modules.values())} sel definisi"
    )
    add("")

    add("## Peta artefak — sel mana menghasilkan apa")
    add("")
    add(
        "`D87` mewajibkan tiap sel produksi mendeklarasikan yang dibaca dan "
        "ditulisnya. Tanpa tabel ini, *sel mana yang menulis figure* hanya bisa "
        "dijawab dengan membaca seluruh sel berurutan — dan yang tidak bisa "
        "ditemukan tidak bisa diverifikasi."
    )
    add("")
    add("| Langkah | Sel | Fase | Membaca | Menulis |")
    add("| --- | --: | --- | --- | --- |")
    for phase, entry in steps:
        slug = entry.itbtc.get("step", "?")
        add(
            f"| `{slug}` | {entry.index} | {phase.title} | "
            f"{_paths(entry, 'reads')} | {_paths(entry, 'writes')} |"
        )
    add("")

    add("## Modul yang dibawa notebook")
    add("")
    add(
        "Tiap baris adalah satu modul `src/itransformer_btc/`, dipotong menjadi "
        "sel per kelompok logis (`SECTION_MAP` di `tools/build_notebook.py`)."
    )
    add("")
    add("| Modul | Sel | Fase | Bagian |")
    add("| --- | --: | --- | --- |")
    for module in sorted(modules):
        pairs = modules[module]
        phase_names = sorted({p.title for p, _ in pairs})
        sections = [str(e.itbtc.get("section", "?")) for _, e in pairs]
        add(
            f"| `src/itransformer_btc/{module}` | {len(pairs)} | "
            f"{'; '.join(phase_names)} | {' · '.join(sections)} |"
        )
    add("")

    add("## Fase")
    add("")
    for phase in numbered:
        add(f"### {phase.title}")
        if phase.blurb:
            add("")
            add(phase.blurb)
        add("")
        if not phase.entries:
            add("_Tidak ada sel kode._")
            add("")
            continue
        for entry in phase.entries:
            label = entry.heading.title if entry.heading else ""
            if entry.role == "step":
                slug = entry.itbtc.get("step", "?")
                detail = f"**Langkah `{slug}`** (sel {entry.index})"
                if label:
                    detail += f" — {label}"
                reads, writes = _paths(entry, "reads"), _paths(entry, "writes")
                if writes != "—":
                    detail += f" · menulis {writes}"
                if reads != "—":
                    detail += f" · membaca {reads}"
            elif entry.role == "module":
                module = entry.itbtc.get("module", "?")
                section = entry.itbtc.get("section", "?")
                detail = (
                    f"Modul `src/itransformer_btc/{module}` § {section} "
                    f"(sel {entry.index})"
                )
            else:
                detail = f"`{entry.role}` (sel {entry.index})"
                if label:
                    detail += f" — {label}"
            add(f"- {detail}")
        add("")

    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    """Write the map, or verify the committed one is current."""
    parser = argparse.ArgumentParser(description="Render the notebook map.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit 1 if docs/NOTEBOOK_MAP.md differs from the notebook",
    )
    args = parser.parse_args(argv)

    phases, counts = parse(NOTEBOOK)
    rendered = render(phases, counts)
    relative = OUTPUT.relative_to(ROOT).as_posix()

    if args.check:
        if not OUTPUT.exists():
            print(f"{relative} is missing", file=sys.stderr)
            return 1
        if OUTPUT.read_text(encoding="utf-8") != rendered:
            print(
                f"{relative} is stale — run `python tools/notebook_map.py`",
                file=sys.stderr,
            )
            return 1
        print(f"{relative} is current")
        return 0

    OUTPUT.write_text(rendered, encoding="utf-8")
    print(
        f"wrote {relative} — {len(phases) - 1} fase, "
        f"{len(_steps(phases))} langkah, {len(_modules(phases))} modul"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
