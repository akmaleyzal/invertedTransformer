t0 = time.time()
ART_DIR = artifact_dir(CFG.profile)

# The raw-file hashes travel with the artifact. Without them a refreshed data vintage
# can be trained against a matrix built from the previous one, and nothing complains.
RAW_HASHES = {f: _sha256_file(RAW_DIR / f) for f in REQUIRED_FILES}
print(f"hashed {len(RAW_HASHES)} raw files in {time.time() - t0:.1f}s")

# Superset of the export manifest in section 16: everything that notebook needs in order
# to report the pipeline it trained on, without reopening the raw data.
PREP_MANIFEST = {
    "feature_order": FEATURE_NAMES,
    "n_variates": N_VARIATES,
    "target_index": TARGET_IDX,
    "target_name": TARGET_NAME,
    "seq_len": CFG.seq_len,
    "pred_len": CFG.pred_len,
    "groups": {n: FEATURE_GROUP.get(n, "other") for n in FEATURE_NAMES},
    "fracdiff_d": FRAC_D,
    "gold_utc_offset_hours": GOLD_OFFSET_H,
    "release_lag_months_days": {k: list(v) for k, v in RELEASE_LAG.items()},
    "dxy_lag_days": DXY_LAG_DAYS,
    "dropped_columns": sorted(MACRO_DROP),
    "macro_pca_components": int(K),
    "winsor_quantile": CFG.winsor_q,
    "collinearity_threshold": CFG.collinear_thresh,
    # ---- beyond the export manifest -------------------------------------
    "rows": int(T),
    "train_rows": int(n_tr),
    "sigma_target": SIGMA_TARGET,
    "raw_feature_count": len(FEATURE_NAMES_RAW),
    "pruned_features": sorted(set(FEATURE_NAMES_RAW) - set(FEATURE_NAMES)),
}

# int64 epoch-microseconds, not a string and not a timezone-aware object: the split
# boundaries downstream are integer comparisons, and an integer cannot carry a
# timezone it forgot to declare.
T_US = t_all.to_numpy().astype("datetime64[us]").astype(np.int64)

PREP_METADATA = write_artifact(ART_DIR, CFG, X, close_all, T_US,
                               SCALER, PREP_MANIFEST, RAW_HASHES)

print(f"\nartifact -> {ART_DIR.resolve()}")
_total = 0.0
for f in ARTIFACT_FILES:
    _mb = (ART_DIR / f).stat().st_size / 1024**2
    _total += _mb
    print(f"  {f:<24} {_mb:>10,.2f} MB")
print(f"  {'TOTAL':<24} {_total:>10,.2f} MB")

print(f"\nfeatures sha256  {PREP_METADATA['features_sha256']}")
print(f"manifest sha256  {PREP_METADATA['manifest_sha256']}")
print(f"scaler   sha256  {PREP_METADATA['scaler_sha256']}")
print(f"frozen fields    {', '.join(FROZEN_FIELDS)}")
print(f"\nwritten in {time.time() - t0:.1f}s   peak RSS {peak_rss_gb():.2f} GB")
