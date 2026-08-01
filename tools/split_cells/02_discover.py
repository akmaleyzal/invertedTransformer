# ==========================================================================
#  EDIT THIS ONE LINE to match the Kaggle Dataset holding the frozen artifact.
#  Local runs fall back to ../data/processed automatically.
# ==========================================================================
KAGGLE_ARTIFACT_DIR = Path("/kaggle/input/itransformer-btc-features")

# Resuming across Kaggle sessions: attach the PREVIOUS session's output as an input
# dataset and point this at the checkpoint folder inside it, e.g.
#   "/kaggle/input/itransformer-session-1/checkpoints/full_L1440_H60_d512_s42"
# Leave as None for a fresh run. See docs/KAGGLE_GUIDE.md.
KAGGLE_RESUME_DIR: str | None = None

PROFILE = "smoke"          # must match the profile 01_preprocess froze the artifact at

ON_KAGGLE = Path("/kaggle/input").exists()


def _complete_artifact(d: Path) -> bool:
    return d.is_dir() and not [f for f in ARTIFACT_FILES if not (d / f).exists()]


def _discover_artifact_dir(profile: str) -> Path:
    """Locate the frozen artifact for `profile`.

    One notebook has to run in two places: on Kaggle the artifact arrives as a
    read-only Dataset under /kaggle/input/<slug>, and the slug is whatever the
    uploader named it; locally it sits where 01_preprocess wrote it. Search both
    instead of hard-coding either. Accept a directory that either contains
    features_<profile>/ or *is* features_<profile>.
    """
    leaf = f"features_{profile}"
    cands = [KAGGLE_ARTIFACT_DIR / leaf, KAGGLE_ARTIFACT_DIR,
             Path("../data/processed") / leaf, Path("data/processed") / leaf]
    root = Path("/kaggle/input")
    if root.exists():
        tops = sorted(p for p in root.iterdir() if p.is_dir())
        mids = [q for p in tops for q in sorted(p.iterdir()) if q.is_dir()]
        cands += [p / leaf for p in tops] + tops + [q / leaf for q in mids] + mids
    return next((c for c in cands if _complete_artifact(c)), KAGGLE_ARTIFACT_DIR / leaf)


ART_DIR = _discover_artifact_dir(PROFILE)
WORK_DIR = Path("/kaggle/working") if ON_KAGGLE else Path("../artifacts")
WORK_DIR.mkdir(parents=True, exist_ok=True)

_missing = [f for f in ARTIFACT_FILES if not (ART_DIR / f).exists()]
assert not _missing, (
    f"ART_DIR={ART_DIR} is missing {len(_missing)} artifact file(s): {_missing}\n"
    f"Run notebooks/01_preprocess.ipynb at PROFILE={PROFILE!r} to produce it, or attach "
    f"the Kaggle Dataset holding features_{PROFILE}/ and set KAGGLE_ARTIFACT_DIR at the "
    f"top of this cell to its mount path."
)
print(f"artifact -> {ART_DIR}   ({len(ARTIFACT_FILES)} files present)")
print(f"work     -> {WORK_DIR}")
print("raw data -> never opened by this notebook; that is the point of the split")
