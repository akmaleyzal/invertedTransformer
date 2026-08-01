# mmap=False, deliberately: the leakage gate in section 12 overwrites the target
# column of X in place and restores it in a `finally`, which a read-only mapping
# would refuse. A full read costs ~1 GB at the full profile - well inside budget,
# since the whole reason this notebook exists is that it no longer holds the
# polars frames that preprocessing needed.
_A = load_artifact(ART_DIR, CFG, mmap=False)

X = _A["X"]
close_all = _A["close"]                      # raw float64 close, for price reconstruction
SCALER = _A["scaler"]
PREP_MANIFEST = _A["manifest"]
PREP_METADATA = _A["metadata"]

# Timestamps come back as int64 epoch-microseconds and are rebuilt into the same
# tz-aware polars Series the preprocessing notebook held, so every downstream
# comparison against `ts(...)` behaves identically on both sides of the split.
T_US = _A["t_us"]
t_all = pl.Series("t", T_US.astype("datetime64[us]")).dt.replace_time_zone("UTC")
T = len(X)

FEATURE_NAMES = list(PREP_MANIFEST["feature_order"])
FEATURE_GROUP = dict(PREP_MANIFEST["groups"])
N_VARIATES = int(PREP_MANIFEST["n_variates"])
TARGET_NAME = PREP_MANIFEST["target_name"]
TARGET_IDX = int(PREP_MANIFEST["target_index"])
assert N_VARIATES == X.shape[1] and FEATURE_NAMES[TARGET_IDX] == TARGET_NAME

mu = np.asarray(SCALER["mean"], dtype=np.float64)
sd = np.asarray(SCALER["std"], dtype=np.float64)

# Scaled space -> raw log return. Every threshold quoted in basis points (the directional
# epsilon, the backtest thresholds) is a raw-return quantity; applying it to standardised
# values would make it ~1/SIGMA_TARGET times too small and it would stop filtering.
SIGMA_TARGET = float(sd[TARGET_IDX])
assert abs(SIGMA_TARGET / PREP_MANIFEST["sigma_target"] - 1) < 1e-12, (
    f"scaler std and manifest sigma disagree: {SIGMA_TARGET} vs "
    f"{PREP_MANIFEST['sigma_target']} - the artifact is internally inconsistent"
)

# Alignment-stage constants that section 16's export manifest reports. All of them are
# data-dependent - the frac-diff order was chosen by an ADF sweep, K by the rank of the
# macro block, the gold offset by a correlation scan - so they are read back from the
# artifact. Recomputing them here would require the raw data this notebook never opens.
FRAC_D = PREP_MANIFEST["fracdiff_d"]
GOLD_OFFSET_H = int(PREP_MANIFEST["gold_utc_offset_hours"])
RELEASE_LAG = {k: tuple(v) for k, v in PREP_MANIFEST["release_lag_months_days"].items()}
DXY_LAG_DAYS = int(PREP_MANIFEST["dxy_lag_days"])
MACRO_DROP = set(PREP_MANIFEST["dropped_columns"])
K = int(PREP_MANIFEST["macro_pca_components"])

# Recomputed, NOT read from the artifact. The scaler gate exists to prove the scaler was
# fitted on the training split alone; if it compared the artifact's own row count against
# the artifact's own claim it would pass unconditionally. Deriving the boundary from CFG
# and checking it against the artifact is what makes the gate mean something.
TRAIN_END_TS = ts(CFG.train_end)
train_row = (t_all <= TRAIN_END_TS).to_numpy()
n_tr = int(train_row.sum())
assert n_tr == SCALER["fitted_on"]["rows"], (
    f"train boundary disagrees: CFG.train_end={CFG.train_end} gives {n_tr:,} rows, "
    f"the scaler was fitted on {SCALER['fitted_on']['rows']:,}"
)

print(f"\nfeature matrix  X {X.shape}  {X.nbytes / 1024**3:.2f} GB  dtype={X.dtype}")
print(f"grid            {t_all[0]}  ->  {t_all[-1]}")
print(f"target variate  index {TARGET_IDX} = {TARGET_NAME}")
print(f"target sigma    {SIGMA_TARGET:.6e}   (1 bp = {1e-4 / SIGMA_TARGET:.4f} in scaled units)")
print(f"train rows      {n_tr:,} of {T:,} ({100 * n_tr / T:.1f}%)  scaler fitted here only")
print(f"frac-diff d     {FRAC_D}    macro PCA components {K}    gold offset {GOLD_OFFSET_H:+d}h")
print(f"blocks          {', '.join(sorted(set(FEATURE_GROUP.values())))}")
print(f"\npeak RSS so far {peak_rss_gb():.2f} GB   (preprocessing peaked at "
      f"{PREP_METADATA['peak_rss_gb']:.2f} GB, in the other notebook)")
