"""Resumable driver for the graphify build. Survives a usage-limit interruption.

The build has exactly one expensive stage -- semantic extraction, which runs as
LLM subagents, one per chunk of documents -- and several cheap deterministic
ones. A usage limit or a crash can only ever interrupt the expensive stage, so
the whole point of this file is that finished extraction work is never redone.

Two things make that true:

``checkpoint``
    Banks every chunk that has landed into graphify's own semantic cache, keyed
    by content hash and by the extraction prompt. Run it after every batch of
    agents, not only at the end. A chunk file on disk survives a crash; a cached
    file additionally survives someone clearing the chunk files, and it is what
    the next run's ``check_semantic_cache`` consults in order to skip a document
    entirely.

    Scoping is the part to get right. ``save_semantic_cache`` stamps a file as
    done, so a file may be stamped only when a chunk actually produced output
    for it -- otherwise a batch that died halfway marks its unreached files
    complete and their content is lost for good (graphify #2015). The covered
    set is therefore derived from the chunks' own ``source_file`` values, never
    from a hardcoded list, so it cannot drift from what was really extracted.

``status``
    Prints the documents that still need an agent. That is the only input needed
    to resume: dispatch agents for exactly those files and nothing else.

``finish``
    Everything after extraction -- merge, build, cluster, analyse, export. It
    reads the cache rather than the chunk files, because after a checkpoint the
    cache is the complete record and also carries work from earlier sessions.
    Deterministic and cheap, so it is always safe to re-run.

Usage::

    python graphify-out/resume.py status
    python graphify-out/resume.py checkpoint
    python graphify-out/resume.py finish
"""

from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "graphify-out"

#: The extraction prompt that produced the cache entries. Cache reads and writes
#: are attributed to it, so a read under a different prompt than the write lands
#: where the next run will not look (graphify #1939).
SPEC = Path.home() / ".claude" / "skills" / "graphify" / "references" / "extraction-spec.md"


def _load_chunks(verbose: bool) -> tuple[list, list, list, set[str]]:
    """Every valid chunk on disk, plus the source files they actually cover."""
    nodes: list = []
    edges: list = []
    hyper: list = []
    covered: set[str] = set()
    for path in sorted(glob.glob(str(OUT / ".graphify_chunk_*.json"))):
        name = os.path.basename(path)
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        except Exception as exc:  # a half-written chunk is not fatal; a BOM is
            # tolerated above because Windows tooling emits one and a chunk
            # skipped here would leave its files unstamped and re-extracted.
            if verbose:
                print(f"  SKIP {name}: unreadable ({exc})")
            continue
        if "nodes" not in data or "edges" not in data:
            if verbose:
                print(f"  SKIP {name}: missing nodes/edges")
            continue
        nodes += data["nodes"]
        edges += data["edges"]
        hyper += data.get("hyperedges", [])
        for item in data["nodes"] + data["edges"] + data.get("hyperedges", []):
            if item.get("source_file"):
                covered.add(item["source_file"])
        if verbose:
            print(f"  OK   {name}: {len(data['nodes'])} nodes, {len(data['edges'])} edges")
    return nodes, edges, hyper, covered


def _pending() -> list[str]:
    """The documents this build still owes an extraction pass."""
    path = OUT / ".graphify_uncached.txt"
    if not path.exists():
        return []
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line]


def _report_remaining(covered: set[str]) -> int:
    remaining = [f for f in _pending() if f not in covered]
    print(f"\nstill to extract: {len(remaining)}")
    for f in remaining:
        print("  -", os.path.relpath(f, ROOT).replace(os.sep, "/"))
    return 0


def status() -> int:
    """Report the documents that still need an extraction agent."""
    _, _, _, covered = _load_chunks(verbose=False)
    return _report_remaining(covered)


def checkpoint() -> int:
    """Bank landed chunks into the semantic cache, scoped to what they cover."""
    from graphify.cache import save_semantic_cache

    nodes, edges, hyper, covered = _load_chunks(verbose=True)
    if not nodes:
        print("nothing to checkpoint")
        return 0
    allow = [f for f in _pending() if f in covered]
    saved = save_semantic_cache(
        nodes, edges, hyper, root=str(ROOT),
        allowed_source_files=allow, prompt_file=str(SPEC),
    )
    print(f"\ncheckpointed {saved} file(s) into the semantic cache")
    return _report_remaining(covered)


def finish() -> int:
    """Merge, build, cluster, analyse, export. Deterministic; safe to re-run."""
    from graphify.analyze import god_nodes, suggest_questions, surprising_connections
    from graphify.build import build_from_json
    from graphify.cache import check_semantic_cache
    from graphify.cluster import cluster, score_all
    from graphify.export import to_json
    from graphify.report import generate

    detect = json.loads((OUT / ".graphify_detect.json").read_text(encoding="utf-8"))
    docs = [f for cat in ("document", "paper", "image") for f in detect["files"].get(cat, [])]
    nodes, edges, hyper, uncached = check_semantic_cache(
        docs, root=str(ROOT), prompt_file=str(SPEC))
    if uncached:
        print(f"ERROR: {len(uncached)} document(s) unextracted; checkpoint then resume",
              file=sys.stderr)
        for f in uncached:
            print("  -", os.path.relpath(f, ROOT).replace(os.sep, "/"), file=sys.stderr)
        return 1

    ast = json.loads((OUT / ".graphify_ast.json").read_text(encoding="utf-8"))
    seen = {n["id"] for n in ast["nodes"]}
    merged_nodes = list(ast["nodes"])
    for node in nodes:
        if node["id"] not in seen:
            merged_nodes.append(node)
            seen.add(node["id"])
    extraction = {
        "nodes": merged_nodes,
        "edges": ast["edges"] + edges,
        "hyperedges": hyper,
        "input_tokens": 0,
        "output_tokens": 0,
    }
    (OUT / ".graphify_extract.json").write_text(
        json.dumps(extraction, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Merged: {len(merged_nodes)} nodes, {len(extraction['edges'])} edges "
          f"({len(ast['nodes'])} AST + {len(nodes)} semantic)")

    graph = build_from_json(extraction, root=str(ROOT), directed=False)
    if graph.number_of_nodes() == 0:
        print("ERROR: graph is empty", file=sys.stderr)
        return 1
    communities = cluster(graph)
    cohesion = score_all(graph, communities)
    gods = god_nodes(graph)
    surprises = surprising_connections(graph, communities)
    labels = {cid: f"Community {cid}" for cid in communities}
    questions = suggest_questions(graph, communities, labels)

    # Export first: to_json returns False rather than shrinking an existing
    # graph (graphify #479), and the report must never describe a graph that
    # graph.json does not contain.
    if not to_json(graph, communities, str(OUT / "graph.json")):
        print("ERROR: refused to shrink graph.json (graphify #479)", file=sys.stderr)
        return 1
    (OUT / "GRAPH_REPORT.md").write_text(
        generate(graph, communities, cohesion, labels, gods, surprises, detect,
                 {"input": 0, "output": 0}, str(ROOT), suggested_questions=questions),
        encoding="utf-8")
    (OUT / ".graphify_analysis.json").write_text(json.dumps({
        "communities": {str(k): v for k, v in communities.items()},
        "cohesion": {str(k): v for k, v in cohesion.items()},
        "gods": gods,
        "surprises": surprises,
        "questions": questions,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges, "
          f"{len(communities)} communities")
    return 0


COMMANDS = {"status": status, "checkpoint": checkpoint, "finish": finish}

if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "status"
    if command not in COMMANDS:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(COMMANDS[command]())
