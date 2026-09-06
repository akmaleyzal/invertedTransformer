"""Download the open-access reference PDFs into paper/references/.

One-shot, idempotent, and deliberately conservative about duplicates: the user
curated eighteen PDFs by hand before this script existed, and re-downloading any
of them under a different filename is the failure this guard exists to prevent.

Every DOI below was resolved against Crossref and every free-full-text URL
against Unpaywall or the hosting repository. Entries with no legal free full
text are listed in ``PAYWALLED`` and are never fetched -- they carry a resolved
DOI and an acquisition route instead, which is what CLAUDE.md 12 asks of a
number it cannot regenerate.

Note the standard this does NOT meet: CLAUDE.md 13.3 requires a verified DOI
*and the source read*. This closes the first half only. ``SOURCE_PROVENANCE``
flags in ``config.py`` are untouched on purpose, so the two do not blur.

Usage::

    python tools/fetch_references.py            # download
    python tools/fetch_references.py --dry-run  # report only
    python tools/fetch_references.py --audit    # duplicate scan of what is there
"""
from __future__ import annotations

import argparse
import difflib
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REFS = ROOT / "paper" / "references"

# Category folders, ordered as a reader meets the study: what the model is,
# what it is compared against, the domain, how it is evaluated, the tests that
# decide, and how the variates were built.
CATEGORIES: dict[str, str] = {
    "transformer-based":
        "LTSF transformer architectures, the channel-independence debate, and "
        "the attention-as-explanation dispute",
    "baselines-and-components":
        "Non-transformer comparators, plus the components the model is built "
        "from (instance normalisation, optimiser)",
    "crypto-market":
        "Bitcoin and cryptocurrency forecasting, market microstructure, and "
        "market efficiency",
    "evaluation-protocol":
        "How out-of-sample performance is measured: rolling-origin evaluation, "
        "concept drift, backtest overfitting",
    "statistical-tests":
        "The tests the code actually runs on the panel: predictive accuracy "
        "under nesting, multiplicity control, clustered inference, directional "
        "accuracy",
    "feature-construction":
        "Estimators the variates are built from (family F2) and the random "
        "matrix theory behind K_eff",
}
BUCKETS = tuple(CATEGORIES)

DUP_RATIO = 0.75

# Several hosts return 403 to a bare urllib User-Agent even for content they
# serve free. These headers are what a browser sends; they do not defeat any
# paywall, and anything still refused is recorded rather than worked around.
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/126.0.0.0 Safari/537.36"),
    "Accept": "application/pdf,text/html;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# name, bucket, url, why it is in the study
# The filename follows the descriptive style already used in paper/references/.
DOWNLOADS: list[tuple[str, str, str, str]] = [
    (
        "The Capacity and Robustness Trade-off, Revisiting the Channel "
        "Independent Strategy (Han, TKDE 2024). arXiv 2304.05206.pdf",
        "transformer-based",
        "https://arxiv.org/pdf/2304.05206",
        "CLAUDE.md 3 -- the theoretical statement of H2; capacity vs robustness",
    ),
    (
        "Attention Is Not Explanation (Jain and Wallace, NAACL 2019). "
        "arXiv 1902.10186.pdf",
        "transformer-based",
        "https://arxiv.org/pdf/1902.10186",
        "CLAUDE.md 13.2 -- mandatory disclosure on attention maps",
    ),
    (
        "Attention Is Not Not Explanation (Wiegreffe and Pinter, EMNLP 2019). "
        "arXiv 1908.04626.pdf",
        "transformer-based",
        "https://arxiv.org/pdf/1908.04626",
        "CLAUDE.md 13.2 -- the rebuttal, and source of the uniform-attention test",
    ),
    (
        "Adam, A Method for Stochastic Optimization (Kingma and Ba, ICLR 2015). "
        "arXiv 1412.6980.pdf",
        "baselines-and-components",
        "https://arxiv.org/pdf/1412.6980",
        "SOURCE_PROVENANCE -- the optimiser train_one uses",
    ),
    (
        "Noise Dressing of Financial Correlation Matrices (Laloux, PRL 1999). "
        "arXiv cond-mat 9810255.pdf",
        "feature-construction",
        "https://arxiv.org/pdf/cond-mat/9810255",
        "CLAUDE.md 5.4 -- random matrix provenance for K_eff",
    ),
    (
        "A Random Matrix Approach to Cross-Correlations in Financial Data "
        "(Plerou, PRE 2002). arXiv cond-mat 0108023.pdf",
        "feature-construction",
        "https://arxiv.org/pdf/cond-mat/0108023",
        "CLAUDE.md 5.4 -- source of the participation ratio",
    ),
    (
        "Tests of Equal Forecast Accuracy and Encompassing for Nested Models "
        "(Clark and McCracken, 2001).pdf",
        "statistical-tests",
        "http://fmwww.bc.edu/RePEc/es2000/0319.pdf",
        "CLAUDE.md 9.2 -- why standard DM is invalid on nested pairs",
    ),
    (
        "A Survey on Concept Drift Adaptation (Gama, ACM CSUR 2014).pdf",
        "evaluation-protocol",
        "http://eprints.bournemouth.ac.uk/22491/1/ACM%20computing%20surveys.pdf",
        "RQ3 -- blind versus informed retraining",
    ),
    (
        "Cluster-Robust Inference, A Guide to Empirical Practice (MacKinnon, "
        "Nielsen and Webb, J Econometrics 2023).pdf",
        "statistical-tests",
        "https://arxiv.org/pdf/2205.03285",
        "CLAUDE.md 9.2 req.3 -- WCR versus WCU at small G",
    ),
    (
        "Trading and Arbitrage in Cryptocurrency Markets (Makarov and Schoar, "
        "JFE 2020).pdf",
        "crypto-market",
        "https://researchonline.lse.ac.uk/id/eprint/100409/1/"
        "Cryptocurrency_Markets_JFE_Accepted.pdf",
        "CLAUDE.md 5.1 -- theoretical justification for family F4 order flow",
    ),
    (
        "Estimating Variance From High, Low and Closing Prices (Rogers and "
        "Satchell, Ann Appl Prob 1991).pdf",
        "feature-construction",
        "https://projecteuclid.org/journals/annals-of-applied-probability/"
        "volume-1/issue-4/Estimating-Variance-From-High-Low-and-Closing-Prices/"
        "10.1214/aoap/1177005835.pdf",
        "CLAUDE.md 5.1 -- the F2 estimator that vanishes on marubozu bars",
    ),
    (
        "Stepwise Multiple Testing as Formalized Data Snooping (Romano and "
        "Wolf, Econometrica 2005).pdf",
        "statistical-tests",
        "https://www.econ.uzh.ch/dam/jcr:ffffffff-935a-b0d6-ffff-ffffa286d4d1/"
        "etca.pdf",
        "CLAUDE.md 9.2 -- the FWER stepdown behind Table 6",
    ),
    (
        "A Note on the Validity of Cross-Validation for Evaluating "
        "Autoregressive Time Series Prediction (Bergmeir, Hyndman and Koo, "
        "CSDA 2018). Monash WP 10-15.pdf",
        "evaluation-protocol",
        "https://robjhyndman.com/papers/cv-wp.pdf",
        "CLAUDE.md 8.1 -- cited AGAINST our own argument: states the "
        "conditions under which K-fold CV IS valid for time series. The "
        "author's Monash working-paper copy, not the Elsevier version",
    ),
    (
        "Evaluating Time Series Forecasting Models, An Empirical Study on "
        "Performance Estimation Methods (Cerqueira, Torgo and Mozetic, "
        "Machine Learning 2020). arXiv 1905.11744.pdf",
        "evaluation-protocol",
        "https://arxiv.org/pdf/1905.11744",
        "CLAUDE.md 8.1 -- the empirical answer to bergmeir2018note: on real "
        "non-stationary series, out-of-sample estimators dominate CV",
    ),
]

# Free to read, but the host refuses scripted access (Cloudflare / Elsevier bot
# walls return 403 or an HTML challenge). Distinct from PAYWALLED: no
# subscription is needed, only a browser. Fetch these by hand into the named
# bucket, keeping the filename so the duplicate guard recognises them.
MANUAL: list[tuple[str, str, str, str]] = [
    (
        "RevIN, Reversible Instance Normalization against Distribution Shift "
        "(Kim, ICLR 2022).pdf",
        "baselines-and-components",
        "https://openreview.net/pdf?id=cGDAkQo1C0p",
        "CLAUDE.md 6.3 -- origin of use_norm. ICLR mints no DOI; OpenReview "
        "is the version of record. Blocked by bot challenge, not by paywall",
    ),
    (
        "The Inefficiency of Bitcoin. urquhart2016.pdf",
        "crypto-market",
        "https://www.sciencedirect.com/science/article/pii/S0165176516303640",
        "CLAUDE.md 4.5 -- market efficiency. Unpaywall reports this OA in the "
        "Elsevier open archive; ScienceDirect returns 403 to a script",
    ),
]

# Resolved DOI, no legal free full text found, and not on disk. Never
# fabricated, never fetched. Everything else once listed here has since been
# obtained by hand and now lives under a category folder.
PAYWALLED: list[tuple[str, str, str]] = [
    ("Brownlees & Gallo 2006, CSDA 51(4):2232-2245",
     "10.1016/j.csda.2006.09.030", "CLAUDE.md 4.2 high-frequency data handling"),
    ("Garman & Klass 1980, J Business 53(1):67-78",
     "10.1086/296072", "CLAUDE.md 5.1 family F2 -- the one F2 estimator missing"),
    ("Rubin 1976, Biometrika 63(3):581-592",
     "10.1093/biomet/63.3.581", "CLAUDE.md 4.2 -- cited to argue it does NOT apply"),
    ("Lopez de Prado 2018, Advances in Financial Machine Learning (Wiley)",
     "book, ISBN 978-1-119-48208-6", "CLAUDE.md 8.2 purging, 8.4 CPCV rejection"),
]

# Two entries were listed here on 2026-09-06 on Unpaywall verdicts of closed,
# and both verdicts were correct: neither file was on disk at the time.
# Tashman 2000 (10.1016/S0169-2070(00)00065-0) and Arian, Norouzi Mobarekeh &
# Seco 2024 (10.1016/j.knosys.2024.112477, ScienceDirect PII
# S0950705124011110) were both supplied by hand later the same day through
# institutional access and now carry file= fields in the bib. check_bib()
# links them, so this list must not claim they are missing.

_STOP = {"a", "an", "the", "of", "for", "and", "on", "in", "to", "with", "is",
         "are", "from", "against", "not"}


def normalise(name: str) -> str:
    """Filename or title -> comparable token string.

    Drops the extension, arXiv/DOI tails, punctuation, digits and stopwords, so
    that two names for the same paper collapse onto the same string even when
    one carries a version suffix and the other a venue.
    """
    s = re.sub(r"\.pdf$", "", name, flags=re.I).lower()
    s = re.sub(r"arxiv[\s:]*[\w./-]+", " ", s)
    s = re.sub(r"10\.\d{4,9}/\S+", " ", s)
    s = re.sub(r"[^a-z ]+", " ", s)
    return " ".join(w for w in s.split() if w not in _STOP and len(w) > 2)


def existing() -> list[tuple[Path, str]]:
    out = []
    for b in BUCKETS:
        for p in sorted((REFS / b).glob("*.pdf")):
            out.append((p, normalise(p.name)))
    return out


def duplicate_of(name: str, index: list[tuple[Path, str]]) -> Path | None:
    """Return the file `name` duplicates, or None. Never renames around a hit."""
    key = normalise(name)
    for path, other in index:
        if difflib.SequenceMatcher(None, key, other).ratio() >= DUP_RATIO:
            return path
    return None


def download(url: str, dest: Path) -> tuple[bool, str]:
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            blob = r.read()
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    if not blob.startswith(b"%PDF"):
        head = blob[:60].decode("utf-8", "replace").replace("\n", " ")
        return False, f"not a PDF (starts {head!r}, {len(blob)} bytes)"
    dest.write_bytes(blob)
    try:
        from pypdf import PdfReader
        if len(PdfReader(str(dest)).pages) < 1:
            dest.unlink()
            return False, "PDF has zero pages"
    except Exception as exc:
        dest.unlink()
        return False, f"pypdf refused it: {type(exc).__name__}: {exc}"
    return True, f"{len(blob) // 1024} kB"


def audit() -> int:
    """Report every pair of on-disk PDFs that look like the same paper."""
    idx = existing()
    print(f"{len(idx)} PDFs under paper/references/")
    hits = 0
    for i, (pa, ka) in enumerate(idx):
        for pb, kb in idx[i + 1:]:
            ratio = difflib.SequenceMatcher(None, ka, kb).ratio()
            if ratio >= DUP_RATIO:
                hits += 1
                print(f"  DUPLICATE {ratio:.2f}\n    {pa.name}\n    {pb.name}")
    print(f"duplicate pairs: {hits}")
    return 1 if hits else 0


BIB = REFS / "references.bib"
INDEX = REFS / "README.md"

_ENTRY = re.compile(r"@(\w+)\{([^,]+),(.*?)\n\}", re.S)


def parse_bib() -> list[dict]:
    """Minimal BibTeX reader for the one file this repository owns.

    Not a general parser: it assumes the layout tools/ writes, where every
    entry ends with a closing brace at column 0.
    """
    text = BIB.read_text(encoding="utf-8")
    out = []
    for kind, key, body in _ENTRY.findall(text):
        rec = {"kind": kind, "key": key.strip()}
        for field in ("title", "journal", "booktitle", "year", "doi",
                      "eprint", "note", "file", "author"):
            m = re.search(rf"\n\s*{field}\s*=\s*\{{(.*?)\}},?\s*\n\s*(?:\w+\s*=|$)",
                          body + "\n", re.S)
            if m:
                rec[field] = " ".join(m.group(1).split())
        note = rec.get("note", "")
        vm = re.search(r"verified=([a-z-]+)", note)
        rec["verified"] = vm.group(1) if vm else "unknown"
        rec["has_pdf"] = bool(rec.get("file"))
        out.append(rec)
    return out


def check_bib(entries: list[dict]) -> int:
    """Every file= must exist; every PDF on disk must have an entry."""
    problems = 0
    claimed = set()
    for e in entries:
        rel = e.get("file", "")
        if not rel:
            continue
        claimed.add(rel)
        if not (REFS / rel).exists():
            print(f"  MISSING FILE  {e['key']}: {rel}")
            problems += 1
    on_disk = {f"{b}/{p.name}" for b in BUCKETS
               for p in (REFS / b).glob("*.pdf")}
    for orphan in sorted(on_disk - claimed):
        print(f"  NO BIB ENTRY  {orphan}")
        problems += 1
    print(f"bib entries {len(entries)} | PDFs on disk {len(on_disk)} | "
          f"problems {problems}")
    return problems


def write_index(entries: list[dict]) -> None:
    tiers = {"read": "read in full", "doi-resolved": "DOI resolved",
             "artifact": "identity from the PDF itself",
             "screened": "search result only", "unknown": "unclassified"}
    lines = [
        "# `paper/references/` — reference library",
        "",
        "**Generated. Do not hand-edit** — run `python tools/fetch_references.py "
        "--index`; the source is `references.bib`.",
        "",
        "`CLAUDE.md` §13.3 forbids a citation without a **verified DOI and the "
        "source read**. The `verified` tier below records how far each entry "
        "got; only `read` clears §13.3 in full.",
        "",
    ]
    counts: dict[str, int] = {}
    for e in entries:
        counts[e["verified"]] = counts.get(e["verified"], 0) + 1
    lines += ["| Tier | Meaning | Count |", "|---|---|---|"]
    for t, meaning in tiers.items():
        if counts.get(t):
            lines.append(f"| `{t}` | {meaning} | {counts[t]} |")
    have = sum(1 for e in entries if e["has_pdf"])
    lines += ["", f"**{len(entries)} entries — {have} with a PDF on disk, "
                  f"{len(entries) - have} metadata-only.**", ""]

    sections: list[tuple[str, str, str]] = [
        (cat, cat.replace("-", " ").capitalize(), blurb)
        for cat, blurb in CATEGORIES.items()
    ]
    sections.append(
        ("", "Not on disk",
         "Resolved DOI, no legal free full text found. Obtain through the "
         "UNESA library or the DOI — never cite one of these without reading "
         "it first (§13.3)."))

    for bucket, heading, blurb in sections:
        rows = [e for e in entries
                if (e.get("file", "").startswith(bucket + "/") if bucket
                    else not e.get("file"))]
        if not rows:
            continue
        lines += [f"## {heading}", "", f"*{blurb}*", "",
                  f"`paper/references/{bucket}/` — {len(rows)} entries" if bucket
                  else f"{len(rows)} entries", "",
                  "| Key | Reference | Year | Identifier | Tier |",
                  "|---|---|---|---|---|"]
        for e in sorted(rows, key=lambda r: (r.get("year", ""), r["key"])):
            ident = (f"`{e['doi']}`" if e.get("doi")
                     else f"arXiv:{e['eprint']}" if e.get("eprint") else "—")
            # strip BibTeX brace-protection: {TimesNet} -> TimesNet
            title = e.get("title", e["key"]).replace("|", "/")
            title = re.sub(r"\{([^{}]*)\}", r"\1", title)
            venue = e.get("journal") or e.get("booktitle") or ""
            venue = venue.replace("\\&", "&").replace("|", "/")
            ref = f"{title}" + (f" — *{venue}*" if venue else "")
            lines.append(f"| `{e['key']}` | {ref} | {e.get('year', '')} | "
                         f"{ident} | `{e['verified']}` |")
        lines.append("")

    lines += [
        "## Free to read, but not fetchable by script",
        "",
        "No subscription needed — only a browser, because the host answers a "
        "script with 403 or a bot challenge. Save into the named category "
        "folder **keeping the filename**, so the duplicate guard recognises it.",
        "",
    ]
    for name, bucket, url, why in MANUAL:
        state = "already saved" if (REFS / bucket / name).exists() else "MISSING"
        lines += [f"- **{bucket}/**`{name}` — {state}", f"  - {url}",
                  f"  - {why}"]
    lines += [
        "", "## Rules this directory follows", "",
        "- **Nothing is ever duplicated.** Files were regrouped into the "
        "category folders above on 2026-09-05 by same-volume rename — never by "
        "copy — and the migration refused to run until its table accounted for "
        "every PDF on disk with no colliding destination. "
        "`--audit` reports **0 duplicate pairs**.",
        "- **Four files were renamed, each for a factual reason.** The largest: "
        "a file named *Financial econometric analysis at ultra-high frequency* "
        "was, on its own first page, **Clark & West (2007)** — the study's "
        "headline statistic filed under someone else's title.",
        "- **No file is fabricated.** An entry with no legal free full text "
        "carries a resolved DOI and an acquisition route, never a placeholder.",
        "- **The artifact outranks the search result.** Crossref returned a "
        "*different paper* for 9 of the 18 originally curated PDFs; where a "
        "filename, a search hit and page 1 of the PDF disagree, page 1 wins. "
        "See `D89`.",
        "- **`verified` in this directory is not `verified` in "
        "`SOURCE_PROVENANCE`.** That flag means *read*, and flipping it "
        "requires reading the paper, not resolving its DOI.",
        "",
    ]
    INDEX.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {INDEX.relative_to(ROOT)} ({len(entries)} entries)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would happen, download nothing")
    ap.add_argument("--audit", action="store_true",
                    help="scan what is already on disk for duplicates and exit")
    ap.add_argument("--index", action="store_true",
                    help="regenerate README.md from references.bib and exit")
    args = ap.parse_args()

    if args.audit:
        return audit()

    if args.index:
        entries = parse_bib()
        problems = check_bib(entries)
        write_index(entries)
        return 1 if problems else 0

    for b in BUCKETS:
        (REFS / b).mkdir(parents=True, exist_ok=True)

    index = existing()
    print(f"baseline: {len(index)} PDFs already present -- none will be moved "
          f"or overwritten\n")

    got = skipped = failed = 0
    for name, bucket, url, why in DOWNLOADS:
        dest = REFS / bucket / name
        dup = duplicate_of(name, index)
        if dup is not None:
            print(f"SKIP dup   {name[:62]}\n           already have {dup.name}")
            skipped += 1
            continue
        if dest.exists():
            print(f"SKIP have  {name[:62]}")
            skipped += 1
            continue
        if args.dry_run:
            print(f"WOULD GET  {name[:62]}\n           {url}")
            continue
        ok, detail = download(url, dest)
        if ok:
            print(f"GOT  {detail:>9}  {name[:60]}")
            index.append((dest, normalise(dest.name)))
            got += 1
        else:
            print(f"FAIL       {name[:62]}\n           {detail[:110]}")
            failed += 1

    print(f"\ndownloaded {got} | skipped {skipped} | failed {failed}")
    print(f"paywalled, DOI recorded, not fetched: {len(PAYWALLED)}")
    if MANUAL:
        print(f"\nfree to read but bot-walled -- save these by hand into the "
              f"named bucket, keeping the filename:")
        for name, bucket, url, _why in MANUAL:
            have = "  [have]" if (REFS / bucket / name).exists() else ""
            print(f"  {bucket}/{name}{have}\n    {url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
