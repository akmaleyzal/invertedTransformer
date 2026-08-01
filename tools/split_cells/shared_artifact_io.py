# ==========================================================================
#  The frozen-artifact contract.
#
#  This cell is byte-identical in 01_preprocess and 02_train. The producer and
#  the consumer must not be able to disagree about what a valid artifact is, so
#  the rules live in one cell that both notebooks carry rather than in two
#  descriptions that drift apart.
# ==========================================================================

# Config fields that shape the feature matrix itself. A training session that
# disagrees with the artifact on any of these is training on inputs it does not
# describe. `train_end` belongs here because the scaler is fitted on rows
# t <= train_end - changing it on the training side is a data leak, not a
# mismatch. `seq_len` belongs here because the warm-up truncation is computed as
# 1440 + seq_len + 60, so it decides which rows exist at all.
FROZEN_FIELDS = (
    "profile", "grid_start", "grid_end", "train_end", "val_end", "test_end",
    "seq_len", "pred_len", "blocks", "macro_n_pca", "fracdiff_grid",
    "fracdiff_width", "winsor_q", "collinear_thresh", "gold_utc_offset_h",
)

ARTIFACT_FILES = ("features.npy", "close.npy", "timestamps.npy",
                  "scaler.json", "feature_manifest.json", "prep_metadata.json")


def _sha256_file(p: Path, chunk: int = 1 << 23) -> str:
    """Hash a file in 8 MB chunks - `features.npy` is ~1 GB at the full profile."""
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def _sha256_json(obj: dict) -> str:
    """Hash a JSON object by content, not by formatting, so indent changes are invisible."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def _frozen_of(cfg) -> dict:
    """Frozen fields with tuples normalised to lists, so a JSON round-trip is a no-op."""
    out: dict = {}
    for f in FROZEN_FIELDS:
        v = getattr(cfg, f)
        out[f] = list(v) if isinstance(v, tuple) else v
    return out


def artifact_dir(profile: str) -> Path:
    """One directory per profile, never shared: `tiny` and `full` are different matrices."""
    base = Path("/kaggle/working/processed") if ON_KAGGLE else Path("../data/processed")
    return base / f"features_{profile}"


def write_artifact(art_dir: Path, cfg, X: np.ndarray, close: np.ndarray,
                   t_us: np.ndarray, scaler: dict, manifest: dict,
                   raw_hashes: dict) -> dict:
    """Write the six artifact files and return the metadata that binds them together.

    `prep_metadata.json` is written last because it carries the hashes of the files
    written before it.
    """
    art_dir.mkdir(parents=True, exist_ok=True)
    assert X.dtype == np.float32, f"feature matrix must be float32, got {X.dtype}"
    assert close.dtype == np.float64, f"close must stay float64, got {close.dtype}"
    assert t_us.dtype == np.int64, f"timestamps must be int64 epoch-us, got {t_us.dtype}"
    assert len(X) == len(close) == len(t_us), (
        f"row counts disagree: X={len(X)}, close={len(close)}, t={len(t_us)}")

    np.save(art_dir / "features.npy", X)
    np.save(art_dir / "close.npy", close)
    np.save(art_dir / "timestamps.npy", t_us)
    (art_dir / "scaler.json").write_text(json.dumps(scaler, indent=2))
    (art_dir / "feature_manifest.json").write_text(json.dumps(manifest, indent=2))

    meta = {
        "created_utc": datetime.now(UTC).isoformat(),
        "profile": cfg.profile,
        "frozen": _frozen_of(cfg),
        "raw_sha256": raw_hashes,
        "features_sha256": _sha256_file(art_dir / "features.npy"),
        "manifest_sha256": _sha256_json(manifest),
        "scaler_sha256": _sha256_json(scaler),
        "shape": list(X.shape),
        "dtype": str(X.dtype),
        "t_first_us": int(t_us[0]),
        "t_last_us": int(t_us[-1]),
        "versions": {"python": sys.version.split()[0],
                     "numpy": np.__version__, "polars": pl.__version__},
        "peak_rss_gb": round(peak_rss_gb(), 3),
    }
    (art_dir / "prep_metadata.json").write_text(json.dumps(meta, indent=2))
    return meta


def load_artifact(art_dir: Path, cfg, mmap: bool = False) -> dict:
    """Load a frozen artifact, enforcing every rejection rule before returning.

    Each rule is printed first and asserted second. A bare AssertionError names the
    rule but not the state that broke it; printing the whole table means a rejected
    session says which value disagreed with which.

    `mmap=True` maps the matrix read-only, which is right for verification. Training
    needs `mmap=False`: the leakage gate overwrites the target column in place and
    restores it afterwards, and a read-only mapping would raise instead.
    """
    missing = [f for f in ARTIFACT_FILES if not (art_dir / f).exists()]
    assert not missing, (
        f"{art_dir} is not a frozen feature artifact - missing {missing}.\n"
        f"Run 01_preprocess.ipynb at this profile, or point at the Dataset holding it."
    )

    manifest = json.loads((art_dir / "feature_manifest.json").read_text())
    scaler = json.loads((art_dir / "scaler.json").read_text())
    meta = json.loads((art_dir / "prep_metadata.json").read_text())

    feat_hash = _sha256_file(art_dir / "features.npy")
    man_hash = _sha256_json(manifest)
    scl_hash = _sha256_json(scaler)

    t_us = np.load(art_dir / "timestamps.npy")
    close = np.load(art_dir / "close.npy")
    X = np.load(art_dir / "features.npy", mmap_mode="r" if mmap else None)

    want, got = _frozen_of(cfg), meta["frozen"]
    drift = {k: {"artifact": got.get(k), "session": want[k]}
             for k in want if got.get(k) != want[k]}
    n_feat = len(manifest["feature_order"])
    shape_ok = X.shape == (len(t_us), n_feat) and len(close) == len(t_us)
    tgt = manifest["feature_order"][manifest["target_index"]]

    checks = [
        ("1 features sha256", feat_hash == meta["features_sha256"],
         f"{feat_hash[:16]}... vs recorded {meta['features_sha256'][:16]}..."),
        ("2 manifest sha256", man_hash == meta["manifest_sha256"],
         f"{man_hash[:16]}... vs recorded {meta['manifest_sha256'][:16]}..."),
        ("3 frozen config", not drift,
         f"{len(FROZEN_FIELDS)} fields identical" if not drift else f"differs {drift}"),
        ("4 shape agreement", shape_ok,
         f"X{X.shape}  t={len(t_us):,}  close={len(close):,}  features={n_feat}"),
        ("5 target variate", tgt == "btc_logret_1",
         f"feature_order[{manifest['target_index']}] = {tgt}"),
        ("+ scaler sha256", scl_hash == meta.get("scaler_sha256"),
         f"{scl_hash[:16]}... vs recorded {str(meta.get('scaler_sha256'))[:16]}..."),
    ]

    print(f"artifact  {art_dir}")
    print(f"  built    {meta['created_utc']}  by numpy {meta['versions']['numpy']} / "
          f"polars {meta['versions']['polars']}")
    for name, ok, detail in checks:
        print(f"  {name:<20} {'PASS' if ok else 'FAIL'}   {detail}")
    for name, ok, detail in checks:
        assert ok, f"artifact REJECTED - rule {name}: {detail}"

    return {"X": X, "close": close, "t_us": t_us, "scaler": scaler,
            "manifest": manifest, "metadata": meta, "checks": checks,
            "features_sha256": feat_hash, "manifest_sha256": man_hash}


print(f"artifact contract loaded: {len(ARTIFACT_FILES)} files, "
      f"{len(FROZEN_FIELDS)} frozen config fields, 6 rejection rules")
