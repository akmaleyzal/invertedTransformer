"""Build `01_preprocess.ipynb` and `02_train.ipynb` by partitioning `iTransformer.ipynb`.

Source cells are copied **byte-identically**; the script never rewrites their contents.
Two cells are cut in half because they mix concerns that land on opposite sides of the
split, and the cut is made at a text anchor so a source edit that moves the boundary
fails loudly here instead of silently producing a broken notebook:

  * cell 21 -> `21a` (`UTC`, `ts`: no data dependency, needed by both notebooks)
               `21b` (master grid: needs `btc_raw`, preprocessing only)
  * cell 43 -> `43a` (`gate_shift_test`: tests `build_features`, so it lives with it)
               `43b` (`gate_split_test`, `gate_scaler_test`: test the training side)

New cells - the artifact contract, the freeze step, the loader, and the section
markdown - come from `tools/split_cells/` and from the palette constants below.

    python tools/build_split_notebooks.py
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SRC = ROOT / "notebooks" / "iTransformer.ipynb"
CELLS = HERE / "split_cells"
OUT_01 = ROOT / "notebooks" / "01_preprocess.ipynb"
OUT_02 = ROOT / "notebooks" / "02_train.ipynb"

# ---- palettes, taken verbatim from the source notebook -------------------------
# One family per concern. No new palette is introduced: a reader who learned the
# colour scheme from the single notebook keeps it across both halves.
TITLE = "linear-gradient(135deg, #0f0c29, #302b63, #24243e)"
META = ("linear-gradient(90deg, #10002b, #240046)", "#c77dff", "#e0aaff", "#9d8cff")
CONFIG = ("linear-gradient(90deg, #1a1a2e, #16213e)", "#e94560", "#f5a623", "#ffd6a5")
LOAD = ("linear-gradient(90deg, #03071e, #370617)", "#f48c06", "#ffd60a", "#ffb703")
GATE = ("linear-gradient(90deg, #0d1b2a, #1b263b)", "#00b4d8", "#90e0ef", "#ade8f4")
EXPORT = ("linear-gradient(90deg, #2d0036, #4a0060)", "#bf5af2", "#e0aaff", "#c77dff")


def section(pal: tuple[str, str, str, str], heading: str, lead: str,
            bullets: list[str]) -> str:
    """A section header card in the source notebook's house style."""
    bg, bl, h2, tx = pal
    items = "\n".join(f"    <li>{b}</li>" for b in bullets)
    return (
        f'<div style="background: {bg}; border-left: 4px solid {bl}; '
        f'border-radius: 8px; padding: 18px 24px;">\n'
        f'  <h2 style="color: {h2}; margin: 0 0 10px 0;">{heading}</h2>\n'
        f'  <p style="color: {tx}; margin: 0 0 8px 0;">{lead}</p>'
        f'  <ul style="color: {tx}; margin: 0; padding-left: 20px;">\n{items}\n'
        f'  </ul>\n</div>'
    )


def banner(heading: str, sub: str, body: str) -> str:
    """The title card, matching source cell 0."""
    return (
        f'<div style="background: {TITLE}; border-radius: 16px; padding: 36px 40px; '
        f'margin-bottom: 8px;">\n'
        f'  <h1 style="color: #e0aaff; font-size: 2.4em; font-weight: 800; '
        f'margin: 0 0 10px 0; letter-spacing: 1px;">\n    {heading}\n  </h1>\n'
        f'  <p style="color: #c77dff; font-size: 1.15em; margin: 0 0 18px 0; '
        f'font-weight: 500;">\n    {sub}\n  </p>\n'
        f'  <hr style="border: none; border-top: 1px solid #7b2d8b; margin: 16px 0;">\n'
        f'  <p style="color: #9d8cff; font-size: 0.97em; margin: 0;">\n    {body}\n  </p>\n'
        f'</div>'
    )


A = "<code>data/processed/features_&lt;profile&gt;/</code>"

MD = {}

MD["01_title"] = banner(
    "🧊 iTransformer · Stage 1 — Preprocessing",
    "Raw parquet → aligned master grid → engineered features → one frozen, hash-verified artifact",
    "This notebook does every part of the pipeline that <strong>does not need a GPU</strong>: "
    "load, validate, resolve the gold timezone, apply publication lags, align four sampling "
    "frequencies onto the BTC minute grid, engineer features, and fit train-only statistics. "
    "It ends by writing a frozen artifact to " + A + " together with the hashes that bind it to "
    "the raw files it came from. <strong>Run it on your own machine</strong> — it costs nothing "
    "and consumes no Kaggle GPU quota. Then run <code>02_train.ipynb</code>, which consumes the "
    "artifact and never opens the raw data at all.")

MD["01_howto"] = section(
    META, "🗂️ How to run this",
    "Designed for a <strong>local machine</strong>, not a Kaggle session. Preprocessing is "
    "CPU-bound and it is the memory ceiling of the whole project, so it belongs where RAM is "
    "free and GPU quota is not being spent. A Kaggle CPU session works as a fallback.",
    ["<strong>Local run.</strong> Working directory is <code>notebooks/</code>; the raw files "
     "are found at <code>../data/raw</code> automatically.",
     "<strong>Pick a profile.</strong> <code>'tiny'</code> = three months, runs in minutes and "
     "exists to verify the notebook itself. <code>'smoke'</code> = six months. "
     "<code>'full'</code> = the whole 2018–2026 grid.",
     "<strong>One artifact per profile.</strong> " + A + " is keyed by profile, so a "
     "<code>tiny</code> artifact can never be mistaken for a <code>full</code> one.",
     "<strong>Publish once.</strong> Upload the <code>full</code> artifact as a Kaggle Dataset "
     "and every training session afterwards reads it instead of rebuilding it.",
     "<strong>Re-run only when features change.</strong> Model, optimiser and evaluation knobs "
     "live in <code>02_train.ipynb</code> and never require touching this notebook."])

MD["01_map"] = section(
    META, "🧭 Notebook map",
    "Sections §1–§8 are unchanged from the original single notebook; §9–§17 moved to "
    "<code>02_train.ipynb</code>. The three closing sections are new and exist only because "
    "the pipeline is now split.",
    ["<strong>§1 Setup</strong> — imports, device detection, seeding, plot theme",
     "<strong>§2 Configuration</strong> — one <code>CFG</code> object; profiles; publication-lag table",
     "<strong>§3 Loading</strong> — typed loaders, string→float casting, UTC normalisation",
     "<strong>§4 Validation</strong> — gap census, OHLC sanity, extreme-return census",
     "<strong>§5 Gold timezone</strong> — resolved empirically, not assumed",
     "<strong>§6 Alignment</strong> — publication lags, backward as-of joins, staleness features",
     "<strong>§7 Features</strong> — price · volume · cross-asset · macro · temporal blocks",
     "<strong>§8 Hygiene</strong> — train-only winsorisation, collinearity pruning, scaler",
     "<strong>Causality gate</strong> — shift equivariance and future-perturbation invariance",
     "<strong>Freeze</strong> — six artifact files plus the hashes that tie them to the raw data",
     "<strong>Verify</strong> — the artifact re-read through the consumer's own rejection rules"])

MD["01_gate"] = section(
    GATE, "🚦 Causality gate — run before anything is frozen",
    "<strong>The one gate that belongs to feature construction rather than to training.</strong> "
    "It travels with <code>build_features</code> because it tests <code>build_features</code>; "
    "freezing a matrix that fails it would bake look-ahead into every session downstream.",
    ["<strong>Shift equivariance.</strong> Delaying the whole input by <em>k</em> minutes must "
     "delay every feature by exactly <em>k</em>.",
     "<strong>Future-perturbation invariance</strong> — the stronger property. Corrupting the "
     "input from row <em>j</em> onward must leave every feature before <em>j</em> bit-identical. "
     "A centred window, a backward fill, or a reversed index fails this immediately.",
     "The remaining gates (split, scaler, overfit-a-batch, leakage) test the training side and "
     "run in <code>02_train.ipynb</code>."])

MD["01_freeze"] = section(
    EXPORT, "🧊 Freezing the feature matrix",
    "Everything above is deterministic given the raw files and <code>CFG</code>. Writing it "
    "down once turns that determinism into something checkable: two training sessions can now "
    "<em>prove</em> they used the same inputs instead of assuming it.",
    ["<code>features.npy</code> — <code>float32 (T, N)</code>, standardised, variate order fixed",
     "<code>close.npy</code> — <code>float64 (T,)</code> raw close. Price reconstruction must "
     "never be derived from a standardised column",
     "<code>timestamps.npy</code> — <code>int64</code> epoch-microseconds UTC. An integer cannot "
     "carry a timezone it forgot to declare",
     "<code>scaler.json</code> — mean, std and winsorisation bounds, with the split they were "
     "fitted on recorded alongside them",
     "<code>feature_manifest.json</code> — variate order, groups, target index, and the "
     "data-dependent choices (frac-diff order, PCA rank, gold offset) that cannot be recomputed "
     "without the raw data",
     "<code>prep_metadata.json</code> — hashes of the raw files, of the matrix, of the manifest "
     "and of the scaler, plus the <strong>frozen config fields</strong> a training session must match"])

MD["01_verify"] = section(
    GATE, "🔍 Verifying the frozen artifact",
    "The producer is checked by the consumer's own rules, in the producer's own session. "
    "An artifact that fails its own verification has not been produced — and learning that here "
    "costs seconds, whereas learning it on Kaggle costs a session.",
    ["The six rejection rules <code>02_train.ipynb</code> applies, applied here first",
     "<strong>Round-trip.</strong> <code>features.npy</code> re-read from disk must be "
     "bit-identical to the matrix in memory — mapped read-only, so this is a streaming "
     "comparison rather than a second full-size allocation",
     "Timestamps, raw close, and the scaler's fitted-row count all checked against what this "
     "session actually computed"])

MD["01_close"] = (
    f'<div style="background: {TITLE}; border-radius: 16px; padding: 28px 34px;">\n'
    '  <h2 style="color: #e0aaff; margin: 0 0 12px 0;">✅ Next — hand the artifact to the GPU</h2>\n'
    '  <ol style="color: #c8b6ff; margin: 0; padding-left: 22px; line-height: 1.7;">\n'
    '    <li><strong>Confirm every gate printed PASS.</strong> A failing gate means the artifact '
    'is not publishable, whatever the numbers downstream look like.</li>\n'
    '    <li><strong>Run <code>02_train.ipynb</code> at the same <code>PROFILE</code>.</strong> '
    'Locally it finds the artifact by itself; the frozen-field check rejects any mismatch.</li>\n'
    '    <li><strong>For the <code>full</code> profile, publish once.</strong> Upload '
    f'{A} as a Kaggle Dataset, then attach it to every training session. Preprocessing never '
    'runs on Kaggle again, and every session trains on a matrix with the same hash.</li>\n'
    '    <li><strong>Re-run this notebook only when the feature definitions change</strong> — '
    'and expect the hash to change with them. That is the mechanism working, not a fault.</li>\n'
    '  </ol>\n</div>')

MD["02_title"] = banner(
    "⚡ iTransformer · Stage 2 — Training &amp; Evaluation",
    "Frozen feature artifact → purged splits → iTransformer + baselines → gates → evaluation → export",
    "This notebook consumes the artifact produced by <code>01_preprocess.ipynb</code> and "
    "<strong>never opens the raw data</strong>. It refuses to start unless the matrix hash, the "
    "manifest hash and every frozen config field match what the artifact records — so a GPU "
    "session cannot silently train on inputs it does not describe. Everything a session is free "
    "to change (model size, optimiser, loss, seed, which stages to run) lives here; everything "
    "that shapes the feature matrix does not.")

MD["02_howto"] = section(
    META, "🗂️ How to run this on Kaggle",
    "Written for a <strong>Kaggle T4 ×2</strong> session: <strong>12 h per session</strong>, "
    "<strong>30 GPU-hours per week</strong>, <strong>20 GB</strong> of auto-saved "
    "<code>/kaggle/working</code>, ~29 GB RAM, 4 CPU cores, and a <strong>20-minute idle "
    "timeout</strong> while editing interactively. <code>docs/KAGGLE_GUIDE.md</code> covers all "
    "of it; the short version:",
    ["<strong>Attach the artifact, not the raw data.</strong> One Kaggle Dataset holding "
     "<code>features_&lt;profile&gt;/</code> from <code>01_preprocess.ipynb</code>. The "
     "discovery cell <em>searches</em> <code>/kaggle/input</code>, so the slug does not have to "
     "match anything.",
     "<strong>Settings → Accelerator → GPU T4 ×2.</strong> <strong>Internet: Off</strong> keeps "
     "the run reproducible — turn it on for one run only if you want the ONNX parity check.",
     "<strong><code>PROFILE</code> must match the artifact.</strong> It is a frozen field; a "
     "mismatch stops the session rather than training on the wrong matrix.",
     "<strong>One stage per session.</strong> The full profile does not fit in 12 h. Use "
     "<code>run_baselines</code>, <code>run_ablation</code>, <code>run_walkforward</code>.",
     "<strong>Resuming is built in.</strong> Training stops <code>reserve_hours</code> before "
     "<code>session_budget_hours</code> so the checkpoint is written instead of killed at the "
     "wall. Save the version, attach that output to the next session, and set "
     "<code>KAGGLE_RESUME_DIR</code> to its <code>checkpoints/&lt;run_id&gt;</code> folder."])

MD["02_map"] = section(
    META, "🧭 Notebook map",
    "Sections §9–§17 are unchanged from the original single notebook. §1–§8 — loading, "
    "validation, alignment, feature engineering, hygiene — now live in "
    "<code>01_preprocess.ipynb</code> and reach this notebook as a frozen artifact.",
    ["<strong>§1 Setup</strong> — imports, device/AMP detection, seeding, plot theme",
     "<strong>Configuration &amp; artifact</strong> — <code>CFG</code>, artifact discovery, and "
     "the six rejection rules that gate the whole session",
     "<strong>§9 Splits</strong> — purged + embargoed chronological splits",
     "<strong>§10 Dataset</strong> — windowing with a precomputed validity mask",
     "<strong>§11 Model</strong> — iTransformer, and the baselines it must beat",
     "<strong>§12 Sanity gates</strong> — split, scaler, overfit-a-batch, leakage",
     "<strong>§13 Training</strong> — DataParallel + AMP + resumable checkpointing",
     "<strong>§14 Evaluation</strong> — metrics, Diebold–Mariano, cost-aware backtest, diagnostics",
     "<strong>§15 Ablation &amp; walk-forward</strong> — which exogenous blocks earn their place",
     "<strong>§16 Export</strong> — state_dict · TorchScript · ONNX, with a verified parity check",
     "<strong>§17 Report</strong> — results table and every known limitation, stated plainly"])

MD["02_config"] = section(
    CONFIG, "🎛️ Configuration &amp; the artifact contract",
    "<code>CFG</code> is still the single source of truth, but its fields are now in two "
    "classes. <strong>Frozen</strong> fields shape the feature matrix and must match the "
    "artifact exactly. <strong>Free</strong> fields are whatever this session wants to try.",
    ["<strong>Frozen</strong> — <code>profile</code>, the grid and split boundaries, "
     "<code>seq_len</code>, <code>pred_len</code>, the feature blocks, the frac-diff and "
     "winsorisation settings, the gold offset. A mismatch <em>stops the session</em>.",
     "<code>train_end</code> is frozen because <strong>the scaler was fitted on rows "
     "<code>t &lt;= train_end</code></strong>. Moving it here would be a data leak, not a "
     "configuration change.",
     "<code>seq_len</code> is frozen because warm-up truncation is computed as "
     "<code>1440 + seq_len + 60</code>, so it decides which rows exist at all.",
     "<strong>Free</strong> — <code>d_model</code>, <code>n_heads</code>, "
     "<code>e_layers</code>, <code>d_ff</code>, <code>dropout</code>, <code>lr</code>, "
     "<code>batch_size</code>, <code>epochs</code>, <code>loss</code>, <code>seed</code>, and "
     "every stage and session-budget switch.",
     "The <code>Config</code> dataclass is duplicated from <code>01_preprocess.ipynb</code> "
     "<em>on purpose</em>: that duplication is what makes the frozen-field check possible, and "
     "the check turns any future drift into a visible failure instead of a wrong result."])

MD["02_artifact"] = section(
    LOAD, "📥 Loading the frozen artifact",
    "<strong>This replaces §1–§8 of the original notebook.</strong> Nothing below reads "
    "<code>data/raw</code>; the matrix, the scaler and the manifest arrive already built, and "
    "six rules decide whether they are allowed to be used at all.",
    ["<strong>1–2 Hashes.</strong> <code>features.npy</code> and the manifest must hash to what "
     "<code>prep_metadata.json</code> recorded. A truncated upload or an edited manifest fails here.",
     "<strong>3 Frozen fields.</strong> Every frozen <code>CFG</code> field must equal the "
     "artifact's. The failure message names the field and both values.",
     "<strong>4 Shape agreement.</strong> Rows, timestamps and close must agree; columns must "
     "match the manifest's variate count.",
     "<strong>5 Target variate.</strong> <code>feature_order[target_index]</code> must be "
     "<code>btc_logret_1</code>. A reordered matrix produces plausible-looking garbage, which is "
     "the worst possible failure mode.",
     "<strong>6 Scaler hash.</strong> Added beyond the original five: a scaler that drifted "
     "without detection is a leak, so five rules are the floor rather than the ceiling.",
     "<code>train_row</code> and <code>n_tr</code> are <strong>recomputed from "
     "<code>CFG</code></strong> and then checked against the scaler. Reading them from the "
     "artifact would make §12's scaler gate pass unconditionally."])


# ---- cell surgery -------------------------------------------------------------
def _anchor(lines: list[str], needle: str) -> int:
    hits = [i for i, ln in enumerate(lines) if ln.startswith(needle)]
    assert len(hits) == 1, f"anchor {needle!r} found {len(hits)} times, expected exactly 1"
    return hits[0]


def split_21(src: str) -> dict[str, str]:
    """`UTC`/`ts` (both notebooks) away from the master-grid build (preprocessing only)."""
    lines = src.splitlines(keepends=True)
    cut = _anchor(lines, "# Warm-up prefix:")
    return {"21a": "".join(lines[:cut]).rstrip("\n") + "\n",
            "21b": "".join(lines[cut:])}


def split_43(src: str) -> dict[str, str]:
    """The causality gate away from the two gates that test the training side."""
    lines = src.splitlines(keepends=True)
    i_synth = _anchor(lines, "def _synthetic_master")
    i_gate2 = _anchor(lines, "# ---- gate 2: split test")
    i_print = _anchor(lines, 'print("Sanity gates")')
    i_shift = _anchor(lines, 'GATES["shift"]')
    i_split = _anchor(lines, 'GATES["split"]')
    i_scale = _anchor(lines, 'GATES["scaler"]')
    head = lines[:i_synth]                      # `GATES = {}` and the blank lines after it
    return {"43a": "".join(head + lines[i_synth:i_gate2] + [lines[i_print], lines[i_shift]]),
            "43b": "".join(head + lines[i_gate2:i_print]
                           + [lines[i_print], lines[i_split], lines[i_scale]])}


# ---- assembly ------------------------------------------------------------------
SEQ_01: list[tuple[str, object]] = [
    ("md", "01_title"), ("md", "01_howto"), ("md", "01_map"),
    *[("src", i) for i in range(3, 21)],
    ("frag", "21a"), ("frag", "21b"),
    *[("src", i) for i in range(22, 33)],
    ("md", "01_gate"), ("frag", "43a"),
    ("md", "01_freeze"), ("py", "shared_artifact_io"), ("py", "01_freeze"),
    ("md", "01_verify"), ("py", "01_verify"),
    ("md", "01_close"),
]

SEQ_02: list[tuple[str, object]] = [
    ("md", "02_title"), ("md", "02_howto"), ("md", "02_map"),
    *[("src", i) for i in range(3, 9)],
    ("frag", "21a"),
    ("md", "02_config"), ("py", "shared_artifact_io"), ("py", "02_discover"), ("src", 11),
    ("md", "02_artifact"), ("py", "02_load"),
    *[("src", i) for i in range(33, 43)],
    ("frag", "43b"),
    *[("src", i) for i in range(44, 70)],
]

# Setup, `Config` and the time helpers appear in both notebooks on purpose: the
# duplication of `Config` is exactly what makes the frozen-field check possible.
DUPLICATED = {3, 4, 5, 6, 7, 8, 11}
# Title, how-to-run and notebook map are rewritten per notebook, not copied.
REPLACED = {0, 1, 2}
SPLIT = {21, 43}


def build(seq: list[tuple[str, object]], src_cells: list[dict], frags: dict[str, str],
          prefix: str) -> list[dict]:
    out: list[dict] = []
    for kind, key in seq:
        if kind == "src":
            cell = src_cells[key]
            ctype, source = cell["cell_type"], "".join(cell["source"])
        elif kind == "frag":
            ctype, source = "code", frags[key]
        elif kind == "md":
            ctype, source = "markdown", MD[key]
        else:
            ctype, source = "code", (CELLS / f"{key}.py").read_text(encoding="utf-8")
        c: dict = {"cell_type": ctype, "metadata": {},
                   "id": f"{prefix}-{len(out):03d}",
                   "source": source.splitlines(keepends=True)}
        if ctype == "code":
            c["execution_count"] = None
            c["outputs"] = []
        out.append(c)
    return out


def main() -> int:
    nb = json.loads(SRC.read_text(encoding="utf-8"))
    src_cells = nb["cells"]
    n = len(src_cells)

    frags = {**split_21("".join(src_cells[21]["source"])),
             **split_43("".join(src_cells[43]["source"]))}

    used01 = {i for k, i in SEQ_01 if k == "src"}
    used02 = {i for k, i in SEQ_02 if k == "src"}
    covered = used01 | used02 | REPLACED | SPLIT
    assert covered == set(range(n)), (
        f"source cells unaccounted for: {sorted(set(range(n)) - covered)} - every cell must be "
        f"routed to a notebook, split, or explicitly replaced")
    stray = (used01 & used02) - DUPLICATED
    assert not stray, f"cells copied into both notebooks without being exempt: {sorted(stray)}"

    for path, seq, prefix in ((OUT_01, SEQ_01, "prep"), (OUT_02, SEQ_02, "train")):
        cells = build(seq, src_cells, frags, prefix)
        bad = []
        for c in cells:
            if c["cell_type"] != "code":
                continue
            try:
                ast.parse("".join(c["source"]))
            except SyntaxError as e:
                bad.append(f"{c['id']}: {e}")
        assert not bad, "code cells failed to parse:\n  " + "\n  ".join(bad)
        path.write_text(json.dumps(
            {"cells": cells, "metadata": nb["metadata"],
             "nbformat": nb["nbformat"], "nbformat_minor": nb["nbformat_minor"]},
            indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        n_code = sum(c["cell_type"] == "code" for c in cells)
        print(f"{path.relative_to(ROOT).as_posix():<32} {len(cells):>3} cells "
              f"({n_code} code, {len(cells) - n_code} markdown)")

    print(f"\nsource cells {n}: {len(used01 - DUPLICATED)} -> 01 only, "
          f"{len(used02 - DUPLICATED)} -> 02 only, {len(DUPLICATED)} duplicated, "
          f"{len(SPLIT)} split, {len(REPLACED)} replaced")
    return 0


if __name__ == "__main__":
    sys.exit(main())
