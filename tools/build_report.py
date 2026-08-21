"""Render every table and figure from the artifacts already on disk.

Root §12 admits no number into the manuscript that does not resolve to a
prediction file, a config hash and a documented decision. This is the command
that turns those three into LaTeX: it reads ``notebooks/outputs/artifacts/``,
writes ``paper/paper_numbers.json`` --- the manuscript's single source --- and
renders every table and figure **from that file**, so nothing is transcribed by
hand.

CPU only, and deliberately so. Nothing here needs a GPU or a Kaggle session: the
684 prediction files are committed, a Romano--Wolf bootstrap iterates in seconds,
and putting the analysis inside a twelve-hour session would make every iteration
cost a session. The notebook calls the same functions from
``itransformer_btc.report``; this is the local driver for them, in the same sense
``tools/build_notebook.py`` is the local driver for the launcher.

Usage::

    python tools/build_report.py
    python tools/build_report.py --artifacts notebooks/outputs/artifacts --out paper
    python tools/build_report.py --check      # fail if regenerating would change a byte

``--check`` is the drift guard `D54d` added for the notebook, applied to the
second generated artifact. It ignores the one field that legitimately moves ---
the generation timestamp --- and nothing else.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from itransformer_btc.features import build_features  # noqa: E402
from itransformer_btc.report import (  # noqa: E402
    build_report,
    render_figures,
    render_tables,
)
from itransformer_btc.segments import load_bars, usable_mask  # noqa: E402

#: Where the grid's output actually lives. Repo-root ``artifacts/`` holds ONE
#: stale 2026-08-06 CPU smoke run and is not the results directory (`D60f`).
DEFAULT_ARTIFACTS = ROOT / "notebooks" / "outputs" / "artifacts"
DEFAULT_OUT = ROOT / "paper"
DEFAULT_PARQUET = ROOT / "data" / "raw" / "BTCUSDT_1h.parquet"

#: The one field that moves on every run and means nothing to a reader. Excluded
#: from ``--check`` so a re-render does not read as drift; everything else is
#: compared.
VOLATILE = ("generated_utc",)

#: Relative tolerance for ``--check``'s float comparison.
#:
#: Exact string equality is the wrong instrument here and measurably so: polars
#: aggregates ``group_by`` in parallel, so summation order varies between runs and
#: a mean over float32 cells lands on a different eighth significant digit each
#: time. Two consecutive builds in one process differ on 28 of ~8,000 numbers at a
#: relative ~1e-7 --- below the precision the underlying float32 columns carry at
#: all, and five orders below anything the tables print.
#:
#: 1e-6 is therefore tight enough that any change a reader could notice fails the
#: check, and loose enough that thread scheduling does not. **Structure is still
#: compared exactly**: a new key, a dropped key, a changed string or a changed
#: list length all fail regardless of tolerance.
FLOAT_RTOL = 1e-6


def resolve_parquet(argument: str | None) -> Path:
    """Argument, then ``ITBTC_PARQUET``, then the committed artifact (`D54c`)."""
    if argument:
        return Path(argument)
    from_env = os.environ.get("ITBTC_PARQUET")
    return Path(from_env) if from_env else DEFAULT_PARQUET


def stable(numbers: dict) -> dict:
    """The numbers with the volatile timestamp removed --- ``--check``'s subject."""
    copy = json.loads(json.dumps(numbers, default=float))
    for field in VOLATILE:
        copy.get("derived_from", {}).pop(field, None)
    return copy


def first_difference(left, right, path: str = "") -> str | None:
    """The first place two reports disagree, or ``None`` --- ``--check``'s answer.

    Structure is compared exactly and floats within :data:`FLOAT_RTOL`. Returning
    the *path* rather than a bare boolean is what makes a failed check actionable:
    "stale" alone would send a reader to diff an 8,000-line JSON file by hand.
    """
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            missing = sorted(set(left) ^ set(right))
            return f"{path or '<root>'}: keys differ, e.g. {missing[:4]}"
        for key in left:
            found = first_difference(left[key], right[key], f"{path}.{key}")
            if found:
                return found
        return None
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return f"{path}: length {len(left)} vs {len(right)}"
        for index, (a, b) in enumerate(zip(left, right)):
            found = first_difference(a, b, f"{path}[{index}]")
            if found:
                return found
        return None
    if isinstance(left, float) and isinstance(right, float):
        if left == right:
            return None
        if math.isnan(left) and math.isnan(right):
            # Both absent, and absent equals absent. Ridge is a solve rather than
            # an optimisation, so its one seed gives an undefined seed std ---
            # correctly, and `report.fmt` renders it as an em-dash. Without this
            # branch the check would report drift on every run, because NaN is
            # not equal to itself.
            return None
        scale = max(abs(left), abs(right), 1e-30)
        if abs(left - right) / scale <= FLOAT_RTOL:
            return None
        return f"{path}: {left!r} vs {right!r}"
    if left != right:
        return f"{path}: {left!r} vs {right!r}"
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the paper's tables and figures.")
    parser.add_argument("--artifacts", default=str(DEFAULT_ARTIFACTS))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--parquet", default=None)
    parser.add_argument("--bootstrap-b", type=int, default=9_999)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--check", action="store_true",
        help="exit 1 if regenerating would change paper_numbers.json",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    log = (lambda *_: None) if args.quiet else print
    artifacts = Path(args.artifacts)
    out_dir = Path(args.out)
    parquet = resolve_parquet(args.parquet)

    log(f"reading {parquet}")
    bars = usable_mask(load_bars(parquet))
    features = build_features(bars)
    log(f"bars {bars.height:,}  features {features.height:,}")

    inputs = build_report(
        artifacts, bars, features,
        bootstrap_b=args.bootstrap_b, seed=args.seed, log=log,
    )

    numbers_path = out_dir / "paper_numbers.json"
    rendered = stable(inputs.numbers)

    if args.check:
        if not numbers_path.exists():
            print(f"{numbers_path} is absent. Run: python tools/build_report.py")
            return 1
        current = stable(json.loads(numbers_path.read_text(encoding="utf-8")))
        difference = first_difference(current, rendered)
        if difference:
            print(
                f"{numbers_path} is stale against {artifacts}: {difference}\n"
                f"Run: python tools/build_report.py"
            )
            return 1
        log("report is current")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    numbers_path.write_text(
        json.dumps(inputs.numbers, indent=2, default=float), encoding="utf-8"
    )
    log(f"wrote {numbers_path}")

    tables = render_tables(inputs.numbers, out_dir / "tables")
    log(f"wrote {len(tables)} tables to {out_dir / 'tables'}")

    figures = render_figures(inputs, out_dir / "figures", log=log)
    log(f"wrote {len(figures)} figure files to {out_dir / 'figures'}")

    # The panels a figure reads, persisted beside the numbers so a reader can
    # redo a plot without re-running the whole aggregation (root §12).
    panels = out_dir / "panels"
    panels.mkdir(parents=True, exist_ok=True)
    for name, frame in (
        ("seed_averaged_cells", inputs.seed_avg),
        ("amplification_panel", inputs.amplification),
        ("rolling_pr", inputs.rolling_pr),
        ("rolling_ols_r2", inputs.rolling_r2),
        ("equity_curves", inputs.equity),
    ):
        frame.write_parquet(panels / f"{name}.parquet")
    if inputs.attention is not None:
        inputs.attention.write_parquet(panels / "attention_maps.parquet")
    log(f"wrote panels to {panels}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
