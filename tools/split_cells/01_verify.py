# Verify the artifact the way 02_train will: re-read it from disk and apply every
# rejection rule. A producer that cannot pass its own consumer's checks has not
# produced anything usable, and finding that out here costs seconds rather than a
# Kaggle session.
_A = load_artifact(ART_DIR, CFG, mmap=True)

# Round-trip: the matrix on disk must be bit-identical to the one in memory. The
# mapping keeps this a streaming comparison instead of a second full-size copy.
GATES["artifact_roundtrip"] = bool(np.array_equal(_A["X"], X))
GATES["artifact_order"] = _A["manifest"]["feature_order"] == FEATURE_NAMES
GATES["artifact_time"] = bool(np.array_equal(_A["t_us"], T_US))
GATES["artifact_close"] = bool(np.array_equal(_A["close"], close_all, equal_nan=True))
GATES["artifact_scaler"] = _A["scaler"]["fitted_on"]["rows"] == n_tr

print()
print(f"  {'artifact_roundtrip':<20} {'PASS' if GATES['artifact_roundtrip'] else 'FAIL'}"
      f"   features.npy is bit-identical to X in memory")
print(f"  {'artifact_order':<20} {'PASS' if GATES['artifact_order'] else 'FAIL'}"
      f"   {len(FEATURE_NAMES)} variates in the order the model will receive them")
print(f"  {'artifact_time':<20} {'PASS' if GATES['artifact_time'] else 'FAIL'}"
      f"   {_A['t_us'][0]} -> {_A['t_us'][-1]} epoch-us")
print(f"  {'artifact_close':<20} {'PASS' if GATES['artifact_close'] else 'FAIL'}"
      f"   raw close preserved in float64 for price reconstruction")
print(f"  {'artifact_scaler':<20} {'PASS' if GATES['artifact_scaler'] else 'FAIL'}"
      f"   scaler fitted on {n_tr:,} train rows of {T:,}")

del _A
gc.collect()

print("\n" + "=" * 62)
for k, v in GATES.items():
    print(f"  {k:<20} {'PASS' if v else 'FAIL'}")
ALL_GATES_PASS = all(GATES.values())
print(f"  {'ALL GATES':<20} {'PASS' if ALL_GATES_PASS else 'FAIL'}")
print("=" * 62)
if not ALL_GATES_PASS:
    print("\n  Do NOT publish this artifact. A failing gate here means every number "
          "trained on it would be invalid.")
else:
    print(f"\n  Artifact frozen and verified. Next: 02_train.ipynb at "
          f"PROFILE = {CFG.profile!r}.")
