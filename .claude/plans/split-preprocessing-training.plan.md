# Plan: Pemisahan Preprocessing dan Training (Kaggle dua-notebook)

**Source PRD**: `.claude/prds/split-preprocessing-training.prd.md`
**Selected Milestone**: 0 (Muat di mesin lokal) + 1 (Kontrak artifact) + 2 (Notebook preprocessing) + 3 (Notebook training)
**Complexity**: Large

> Milestone 1–3 direncanakan sebagai satu unit karena tidak terpisahkan: kontrak tanpa produsen
> dan konsumen tidak bisa divalidasi, dan notebook mana pun tanpa kontrak hanya memindahkan
> masalah. Milestone 4 (panduan) dan 5 (validasi ekuivalensi) ikut sebagai Task 7–8 karena
> keduanya murah dan justru menjadi bukti bahwa 1–3 benar.
>
> **Milestone 0 adalah prasyarat keras.** Mesin lokal punya **14,8 GB RAM total, ~4,2 GB bebas** —
> lebih kecil daripada Kaggle (~29 GB). Puncak preprocessing profil `full` dengan kode sekarang
> ~7–9 GB, jadi tanpa Task 0 seluruh rencana ini tidak bisa dijalankan di mesin peneliti.

---

## Summary

Pecah `notebooks/iTransformer.ipynb` (70 sel) menjadi `notebooks/01_preprocess.ipynb` (**dijalankan
di mesin lokal**, menghasilkan artifact fitur beku di `data/processed/`) dan
`notebooks/02_train.ipynb` (sesi GPU Kaggle, mengonsumsinya). Pemisahan dilakukan lewat **skrip
pembangun yang mem-partisi sel yang sudah ada secara byte-identik**, bukan dengan menyalin ulang
kode — supaya tidak ada perilaku yang berubah diam-diam saat pemisahan. Kontrak antar-notebook
dipaksakan lewat rantai hash: notebook training menolak berjalan bila matriks fitur, manifest, atau
field konfigurasi beku tidak cocok.

**Di mana masing-masing berjalan:**

| Notebook | Tempat | Alasan |
|---|---|---|
| `01_preprocess` | **Mesin lokal** (12 core logis) | Preprocessing CPU-bound; lokal punya 3× core Kaggle dan nol biaya kuota. Sesi CPU Kaggle tetap jadi fallback |
| `02_train` | **Kaggle GPU T4 ×2** | Satu-satunya tahap yang butuh GPU |

Artifact ditulis ke `data/processed/features_{profile}/` — persis lokasi yang sudah dituju
CLAUDE.md §3 (`processed/ # windowed tensors / memmapped .npy + scalers`), jadi ini kembali ke
rencana asal, bukan penyimpangan darinya.

---

## Patterns to Mirror

Tidak ada `src/`, `tests/`, atau `configs/` di repo — pola diambil dari notebook itu sendiri.

| Category | Source | Pattern |
|---|---|---|
| Markdown seksi | `iTransformer.ipynb` cell 14 | `<div style="background: linear-gradient(90deg, #03071e, #370617); border-left: 4px solid #f48c06; border-radius: 8px; padding: 18px 24px;">` + `<h2 style="color: #ffd60a;">` + `<ul>`. Satu palet per seksi, tidak pernah dicampur |
| Markdown judul | `iTransformer.ipynb` cell 0 | `linear-gradient(135deg, #0f0c29, #302b63, #24243e)`, `border-radius: 16px`, `padding: 28px 34px` |
| Penamaan knob | `iTransformer.ipynb` cell 10 | `KAGGLE_*` untuk path yang diedit manusia, blok komentar `EDIT THIS ONE LINE` di paling atas |
| Penemuan path | `iTransformer.ipynb` cell 10 `_discover_raw_dir()` | Daftar kandidat → `next((c for c in cands if _complete(c)), fallback)`; assertion menyebut file yang hilang |
| Error handling | `iTransformer.ipynb` cell 15 `_assert_schema()` | `assert` dengan pesan menyebut nama kolom/file konkret; gagal keras, tidak pernah `warnings.warn` |
| Logging | `iTransformer.ipynb` cell 46 | `print` sejajar kolom + JSONL per epoch ke `RUN_DIR`; tanpa layanan eksternal |
| Gate | `iTransformer.ipynb` cell 43 | `GATES["nama"] = fungsi()` mengembalikan `bool`, mencetak `PASS`/`FAIL`, diagregasi jadi `ALL_GATES_PASS` |
| Hash | `iTransformer.ipynb` cell 65 | `hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()` |
| Verifikasi | scratchpad `run_tiny.py` | Eksekusi sel berurutan dalam satu namespace `exec`, berhenti di kegagalan pertama |

---

## Files to Change

| File | Action | Why |
|---|---|---|
| `notebooks/iTransformer.ipynb` | UPDATE (Task 0) | Optimisasi memori tahap hygiene + helper `peak_rss_gb()`. Tetap jadi sumber partisi dan referensi ekuivalensi |
| `notebooks/01_preprocess.ipynb` | CREATE | Lokal: load → validate → gold tz → align → features → hygiene → freeze artifact |
| `notebooks/02_train.ipynb` | CREATE | Kaggle GPU: load artifact → verify → split → train → eval → backtest → export |
| `data/processed/features_{profile}/` | CREATE (runtime) | Artifact beku; diunggah ke Kaggle sebagai Dataset sekali |
| `tools/build_split_notebooks.py` | CREATE | Skrip pembangun: mem-partisi sel, menyuntik sel baru, menulis kedua notebook |
| `docs/KAGGLE_GUIDE.md` | UPDATE | Alur dua-notebook menggantikan alur satu-notebook (§2, §3, §5, §6, §10) |
| `CLAUDE.md` | UPDATE | §3 layout, §3.1 "di mana pipeline hidup", kontrak artifact sebagai fakta proyek |
| `.claude/prds/split-preprocessing-training.prd.md` | UPDATE | Milestone 1–3 → `in-progress`, kolom Plan diisi |

---

## Kontrak Artifact (Milestone 1 — keputusan desain)

Isi direktori `features_{profile}/`:

| File | Isi | Alasan |
|---|---|---|
| `features.npy` | `float32 (T, N)` | Matriks fitur terstandardisasi. `float32` karena itu yang dikonsumsi model; `mmap_mode='r'` di sisi training |
| `close.npy` | `float64 (T,)` | Harga close mentah — dibutuhkan rekonstruksi harga; tidak boleh diturunkan dari fitur terstandardisasi |
| `timestamps.npy` | `int64 (T,)` epoch-µs UTC | Basis semua batas split; integer agar bebas timezone |
| `scaler.json` | mean/std/winsor per fitur + `fitted_on` | Kontrak yang sudah ada di bundle export, dipakai ulang apa adanya |
| `feature_manifest.json` | urutan variat, grup, `target_index`, `fracdiff_d`, offset gold, tabel release lag, kolom yang di-drop, jumlah komponen PCA | Superset dari manifest bundle yang sudah ada |
| `prep_metadata.json` | hash file mentah, `features_sha256`, `manifest_sha256`, **frozen fields**, versi polars/numpy, `created_utc` | Rantai bukti; ini yang diverifikasi notebook training |

**Frozen fields** — field `Config` yang membentuk matriks. Notebook training **wajib** memakai
nilai yang sama; ketidakcocokan menggagalkan sesi:

```
profile, grid_start, grid_end, train_end, val_end, test_end,
seq_len, pred_len, blocks, macro_n_pca, fracdiff_grid, fracdiff_width,
winsor_q, collinear_thresh, gold_utc_offset_h
```

`seq_len` ikut dibekukan karena warm-up truncation dihitung dari `1440 + seq_len + 60`;
`train_end` ikut dibekukan karena **scaler difit pada baris `t <= train_end`** — mengubahnya di
sisi training adalah kebocoran data, bukan sekadar ketidakcocokan.

**Free fields** — bebas per sesi training: `d_model, n_heads, e_layers, d_ff, dropout, lr,
weight_decay, batch_size, epochs, loss, seed, run_*`, dan semua knob sesi/anggaran.

**Aturan penolakan** (semuanya `assert`, gagal keras):
1. `sha256(features.npy)` ≠ `prep_metadata.features_sha256`
2. `sha256(feature_manifest.json)` ≠ `prep_metadata.manifest_sha256`
3. Ada frozen field yang berbeda antara `CFG` dan `prep_metadata.frozen`
4. `features.shape` ≠ `(len(timestamps), len(feature_order))`
5. `feature_order[target_index] != "btc_logret_1"`

---

## Pembagian Sel

Partisi dari `iTransformer.ipynb` (indeks sel sumber):

| Notebook | Sel sumber | Sel baru |
|---|---|---|
| `01_preprocess` | 3–8 (setup/device/theme), 9–13 (config), 14–32 (load → hygiene), bagian `gate_shift_test` dari 43 | judul, cara pakai, peta, **freeze artifact**, **laporan verifikasi**, penutup |
| `02_train` | 4–8 (setup/device/theme), 33–42, bagian `gate_split_test`+`gate_scaler_test` dari 43, 44–69 | judul, cara pakai, peta, **muat + verifikasi artifact** (menggantikan 9–32) |

**Sel 43 dipecah.** `gate_shift_test` menguji `build_features`, jadi ikut ke `01` — gate kausalitas
harus hidup bersama kode yang diujinya. `gate_split_test` dan `gate_scaler_test` ke `02`, dengan
tambahan: `02` memverifikasi ulang `scaler.fitted_on` dari artifact, bukan dari variabel lokal.

**Yang diduplikasi dengan sengaja:** sel 4–8 (imports, deteksi device, tema plot) dan
`dataclass Config`. Duplikasi inilah yang membuat pengecekan frozen-field mungkin, dan biayanya
nol karena keduanya tidak mengandung logika domain.

---

## Tasks

### Task 0: Turunkan puncak memori sampai muat di mesin lokal — **prasyarat**
- **Action**: Enam perubahan pada sel hygiene `iTransformer.ipynb`, semuanya lokal ke satu sel:
  1. `feat_df` di-cast ke `Float32` **di dalam polars** sebelum `to_numpy()`, menghilangkan array float64 antara (2,04 GB).
  2. Buang `.astype(np.float64)` yang menyalin tanpa guna.
  3. Split train adalah **prefix kontigu** (`t_all` terurut, mask `[True…True, False…False]`) → ganti lima `X_raw[train_row]` dengan slice view `X_raw[:n_tr]`.
  4. `np.clip(..., out=X_raw)` in-place, bukan array baru.
  5. Standardisasi in-place dengan `mu`/`sd` float32, tetapi **dihitung dengan akumulator float64** (`mean(axis=0, dtype=np.float64)`) supaya presisi scaler tidak turun.
  6. `del` + `gc.collect()` di setiap batas tahap.
- **Ditolak dengan sengaja**: menurunkan `master` ke float32. Harga BTC ~60.000 dalam float32 hanya menyisakan ~3 digit signifikan pada log-return 1 menit (~1e-3) — merusak target itu sendiri.
- **Tambahan**: helper `peak_rss_gb()` tanpa dependensi (Windows `psapi.GetProcessMemoryInfo`, Linux `/proc/self/status`), dicetak di akhir tahap hygiene.
- **Mirror**: gaya komentar "kenapa, bukan apa" di cell 32; assertion eksplisit seperti `_assert_schema`.
- **Validate**: run `tiny` tetap `ALL CELLS OK` + `ALL GATES PASS`; `peak_rss_gb()` tercetak; jalankan profil `full` di lokal dan konfirmasi puncak < 4 GB.
- **Konsekuensi**: Task 0 mengubah bit `X` (kuantil dan scaler kini dihitung di float32). Karena itu **ekuivalensi Task 8 diukur terhadap `iTransformer.ipynb` setelah Task 0**, bukan versi sebelumnya.

### Task 1: Skrip pembangun + partisi
- **Action**: Buat `tools/build_split_notebooks.py`. Baca `iTransformer.ipynb`, salin sel per indeks **tanpa mengubah isinya**, pecah sel 43, sisipkan sel baru, tulis kedua notebook.
- **Mirror**: pola `exec` berurutan dari `run_tiny.py`; assertion eksplisit seperti `_assert_schema`.
- **Validate**: skrip meng-assert setiap sel sumber muncul tepat sekali di salah satu keluaran; AST-parse semua sel kode di kedua notebook.

### Task 2: Sel freeze artifact di `01`
- **Action**: Sel baru sesudah hygiene. Hitung hash file mentah, tulis enam file artifact ke `data/processed/features_{profile}/` (atau `/kaggle/working/` bila kebetulan dijalankan di Kaggle), cetak ukuran dan hash.
- **Mirror**: gaya penulisan bundle di cell 65 (`write_text(json.dumps(..., indent=2))`), pola hash di cell 65.
- **Validate**: jalankan `01` profil `tiny`; artifact ada, `features.npy` bershape `(T, 62)`, hash tercetak.

### Task 3: Sel verifikasi di `01`
- **Action**: Baca ulang artifact yang baru ditulis, terapkan kelima aturan penolakan pada dirinya sendiri, cetak laporan.
- **Mirror**: format tabel `GATES` di cell 43.
- **Validate**: semua cek `PASS` di run `tiny`.

### Task 4: Sel muat + verifikasi artifact di `02`
- **Action**: Penemuan `features_{profile}/` dengan urutan `/kaggle/input/*/` → `/kaggle/input/*/*/` → `../data/processed/` → `data/processed/` (pola `_discover_raw_dir`), sehingga notebook yang sama jalan di Kaggle dan lokal. Lalu `np.load(mmap_mode='r')`, terapkan kelima aturan penolakan, rekonstruksi `FEATURE_NAMES`, `FEATURE_GROUP`, `mu`, `sd`, `SIGMA_TARGET`, `TARGET_IDX`, `N_VARIATES`, `t_np`, `close_all`, `T`.
- **Mirror**: `_discover_raw_dir()` cell 10; pesan assertion menyebut file konkret.
- **Validate**: `02` profil `tiny` mencapai sel split tanpa menyentuh `data/raw`.

### Task 5: Rewire sel hilir di `02`
- **Action**: Sel 65 (manifest export) mengambil `fracdiff_d`, `gold_utc_offset_hours`, `release_lag`, `dropped_columns`, `macro_pca_components` dari manifest artifact, bukan dari variabel yang kini tidak ada. Gate scaler diverifikasi terhadap `scaler.json`.
- **Mirror**: struktur `MANIFEST` cell 65 apa adanya.
- **Validate**: `02` selesai sampai bundle export, parity TorchScript `< 1e-4`.

### Task 6: Markdown seksi baru
- **Action**: Judul, cara pakai, peta notebook, dan header seksi baru untuk kedua notebook. Palet mengikuti tabel *Patterns to Mirror* — `01` memakai keluarga oranye/biru (loading, alignment), `02` memakai keluarga ungu/teal (model, gates, eval).
- **Mirror**: cell 0, 1, 2, 14, 20 apa adanya.
- **Validate**: setiap sel markdown baru memakai `border-left` dari palet yang terdaftar; tidak ada palet baru.

### Task 7: Perbarui `docs/KAGGLE_GUIDE.md` dan `CLAUDE.md`
- **Action**: Guide §2/§3/§5/§6/§10 jadi alur dua-notebook, termasuk publikasi artifact sebagai Dataset dan sesi CPU untuk `01`. CLAUDE.md §3 layout + §3.1.
- **Mirror**: gaya tabel dan nada guide yang sudah ada; Bahasa Indonesia.
- **Validate**: guide menyebut `01_preprocess`/`02_train` di setiap tempat yang sebelumnya menyebut satu notebook.

### Task 8: Validasi ekuivalensi (Milestone 5)
- **Action**: Jalankan `01` lalu `02` pada profil `tiny`. Bandingkan dengan run satu-notebook.
- **Validate**:
  - **Keras**: `sha256(features.npy)` dari `01` = hash `X` yang dihitung ulang notebook tunggal.
  - **Lunak**: `best_val_mse` dan MSE test sama dalam toleransi `1e-6`. Perbedaan diselidiki, tidak diasumsikan wajar.

---

## Validation

```bash
PY=".venv/Scripts/python.exe"

# Task 1 — bangun kedua notebook dari sumber
$PY tools/build_split_notebooks.py notebooks/iTransformer.ipynb

# AST check kedua notebook
$PY -c "import ast,json,sys;[ast.parse(''.join(c['source'])) for f in sys.argv[1:] for c in json.load(open(f,encoding='utf-8'))['cells'] if c['cell_type']=='code']" \
   notebooks/01_preprocess.ipynb notebooks/02_train.ipynb

# Task 2-3 — jalankan preprocessing profil tiny, artifact terbentuk
$PY <scratchpad>/run_tiny.py notebooks/01_preprocess.ipynb

# Task 4-5 — jalankan training profil tiny dari artifact
$PY <scratchpad>/run_tiny.py notebooks/02_train.ipynb

# Task 8 — ekuivalensi terhadap notebook tunggal
$PY <scratchpad>/run_tiny.py notebooks/iTransformer.ipynb
```

Kriteria lulus: `01` berakhir dengan seluruh cek artifact `PASS`; `02` berakhir dengan
`ALL CELLS OK`, `ALL GATES PASS`, dan `parity torchscript OK`.

---

## Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Sel yang dipindah kehilangan variabel yang didefinisikan di sel yang tidak ikut | **Tinggi** | Skrip mem-partisi per indeks, lalu run `tiny` mengeksekusi berurutan — `NameError` muncul segera, bukan di Kaggle |
| Pemecahan sel 43 mengubah urutan pemanggilan `set_seed`, sehingga metrik tidak lagi identik | Sedang | Ekuivalensi keras diukur pada hash `X`, bukan metrik; metrik dibandingkan dengan toleransi dan diselidiki bila berbeda |
| Duplikasi `Config` di dua notebook menyimpang seiring waktu | **Tinggi** | Frozen-field check menggagalkan sesi saat menyimpang — drift jadi kegagalan yang terlihat, bukan hasil yang salah |
| `mmap_mode='r'` melambat karena akses acak lintas 1 GB | Sedang | Worker `fork` berbagi mapping; bila lambat, `np.load` tanpa mmap tetap hanya 1,02 GB |
| Notebook belum masuk git, restrukturisasi tanpa undo | **Tinggi** | Skrip **hanya menulis file baru**; `iTransformer.ipynb` tidak pernah dimodifikasi |
| Guide dan CLAUDE.md tidak sinkron dengan notebook | Sedang | Task 7 satu paket dengan implementasi, bukan menyusul |

---

## Acceptance

- [ ] **Task 0**: puncak RSS preprocessing profil `full` terukur **< 4 GB** di mesin lokal
- [ ] `tools/build_split_notebooks.py` menghasilkan kedua notebook secara deterministik dari sumber
- [ ] Setiap sel sumber muncul tepat sekali di salah satu keluaran (kecuali 4–8 dan `Config` yang sengaja diduplikasi)
- [ ] `01_preprocess` profil `tiny` menulis enam file artifact dan seluruh cek verifikasi `PASS`
- [ ] `02_train` profil `tiny` berjalan sampai export tanpa menyentuh `data/raw`
- [ ] Kelima aturan penolakan terbukti menggagalkan sesi saat sengaja dilanggar
- [ ] Hash `features.npy` identik dengan `X` dari notebook tunggal
- [ ] `docs/KAGGLE_GUIDE.md` dan `CLAUDE.md` konsisten dengan alur baru
- [ ] Gaya markdown mengikuti palet per seksi yang terdaftar, tidak ada palet baru

---

## Catatan Implementasi (2026-07-30) — hasil pengukuran, bukan rencana

Empat hal di bawah **mengoreksi rencana di atas**. Semuanya ditemukan lewat analisis
dependensi statis (`ast`) atas 70 sel, bukan lewat pembacaan biasa.

### 1. Kontrak batas = 24 nama, terukur

Partisi `02_train` = sel `4-8, 11, 34-42, 44-69` meninggalkan **tepat 24 nama tidak
terdefinisi**. Itulah yang wajib direkonstruksi sel pemuat:

| Sumber | Nama | Direkonstruksi dari |
|---|---|---|
| sel 10 | `PROFILE`, `ON_KAGGLE`, `WORK_DIR`, `KAGGLE_RESUME_DIR` | sel penemuan baru (`02_discover.py`) |
| sel 13 | `RELEASE_LAG`, `DXY_LAG_DAYS` | `feature_manifest.json` |
| sel 19 | `GOLD_OFFSET_H` | `feature_manifest.json` |
| sel 21 | `UTC`, `ts` | **sel 21 dipecah** — lihat (2) |
| sel 23 | `MACRO_DROP` | `feature_manifest.json` |
| sel 25 | `K` | `feature_manifest.json` |
| sel 29 | `FRAC_D` | `feature_manifest.json` |
| sel 30 | `FEATURE_GROUP` | `feature_manifest.json` (`groups`) |
| sel 32 | `X`, `t_all`, `T`, `FEATURE_NAMES`, `N_VARIATES`, `TARGET_IDX`, `TARGET_NAME`, `SIGMA_TARGET`, `mu`, `sd`, `SCALER` | artifact + `scaler.json` |
| sel 43 | `GATES` | sel `43b` |

Ditambah `TRAIN_END_TS`, `train_row`, `n_tr` yang dipakai `gate_scaler_test` — ketiganya
**dihitung ulang dari `CFG`**, tidak dibaca dari artifact, supaya gate-nya tidak tautologis.

Partisi `01_preprocess` (`4-8, 10-32, 43`) hanya kehilangan `SPLITS` dan `span`, keduanya
milik `gate_split_test` — konfirmasi bahwa pemecahan sel 43 memang tepat di garis itu.

### 2. Sel 21 juga harus dipecah — rencana asal melewatkan ini

`ts()` dan `UTC` didefinisikan di sel 21 **bersama** pembangunan master grid yang butuh
`btc_raw`. `02` memakai `ts()` di sel 34 dan 62, jadi sel 21 dipecah seperti sel 43:

- `21a` = `UTC` + `def ts(...)` — nol dependensi data, **masuk ke kedua notebook**
- `21b` = `WARMUP_MIN` ke bawah (grid + master) — **hanya `01`**

### 3. `mmap_mode='r'` di sisi training TIDAK bisa dipakai

`gate_leakage_test` (sel 47) melakukan `X[:, TARGET_IDX] = ...` lalu memulihkannya di
`finally`. Mapping read-only menolak penulisan itu. Karena itu `02` memuat penuh
(`mmap=False`, ~1,02 GB) — tetap jauh di bawah target < 4 GB, karena notebook ini tidak
lagi memegang frame polars milik preprocessing. `mmap=True` hanya dipakai sel verifikasi
di `01`, di mana perbandingan round-trip justru ingin streaming.

### 4. Task 5 dikerjakan tanpa menyentuh sel 65

Alih-alih mengedit sel 65 agar membaca manifest, **sel pemuat mengikat ulang nama-nama
yang sama** (`FRAC_D`, `GOLD_OFFSET_H`, `RELEASE_LAG`, `DXY_LAG_DAYS`, `MACRO_DROP`, `K`)
dari `feature_manifest.json`. Sel 65 tetap byte-identik, dan `MANIFEST` hasil export
otomatis mereproduksi nilai artifact persis — sehingga ekuivalensi Task 8 jadi eksak,
bukan kira-kira.

### Tambahan pada kontrak

- Aturan penolakan ke-**6**: `sha256(scaler.json)` vs `prep_metadata.scaler_sha256`.
  Scaler yang menyimpang tanpa terdeteksi adalah kebocoran, jadi lima aturan itu batas
  bawah, bukan batas atas.
- `close.npy` **belum dikonsumsi** `02` hari ini — di notebook tunggal `close_all` pun
  dibuat lalu tidak pernah dipakai lagi setelah sel 32. Tetap dibekukan: rekonstruksi
  harga tidak boleh diturunkan dari matriks terstandardisasi, dan 35 MB itu murah.
- Sel setup yang **sengaja diduplikasi**: `3-8` (imports/device/tema), `11` (`Config`),
  `21a`, dan sel kontrak `shared_artifact_io`. Sel `0-2` (judul/cara pakai/peta)
  **diganti**, bukan disalin.

### Status file (2026-07-30)

| File | Status |
|---|---|
| `tools/split_cells/shared_artifact_io.py` | selesai — `FROZEN_FIELDS`, `write_artifact`, `load_artifact` |
| `tools/split_cells/01_freeze.py` | selesai (Task 2) |
| `tools/split_cells/01_verify.py` | selesai (Task 3) |
| `tools/split_cells/02_discover.py` | selesai (Task 4a) |
| `tools/split_cells/02_load.py` | selesai (Task 4b, sekaligus Task 5) |
| `tools/build_split_notebooks.py` | selesai (Task 1) — markdown seksi jadi konstanta di dalamnya, bukan file `.md` terpisah (Task 6) |
| `notebooks/01_preprocess.ipynb` | selesai — 42 sel (23 kode, 19 markdown) |
| `notebooks/02_train.ipynb` | selesai — 53 sel (28 kode, 25 markdown) |
| `docs/KAGGLE_GUIDE.md`, `CLAUDE.md` | selesai (Task 7) — guide §1–§10 dan `CLAUDE.md` §3/§3.1/§3.2 |
| Run `tiny` dua notebook + ekuivalensi | selesai (Task 8) — lihat di bawah |

### Hasil verifikasi (2026-07-30, profil `tiny`)

| Yang diuji | Hasil |
|---|---|
| Pemetaan sel | 70 sel: 22 → `01`, 36 → `02`, 7 diduplikasi, 2 dipecah, 3 diganti. Skrip meng-assert tidak ada sel tersisa |
| `01_preprocess` | `ALL CELLS OK` + `ALL GATES PASS` (6 gate); artifact 32,28 MB; puncak RSS **1,63 GB** |
| 6 aturan penolakan, diterapkan `01` ke dirinya sendiri | semua `PASS` |
| `02_train` | `ALL CELLS OK`; sampai tabel ablasi + export; **`data/raw` tidak dibuka** |
| **Ekuivalensi keras** | `features.npy` **bit-identik** dengan `X` notebook tunggal — `sha256 c3ad8cd397d9fb88ca5c11372dd2b1d7` di kedua sisi; `timestamps` dan `scaler` juga sama |

### Penyimpangan dari rencana, dan alasannya

1. **Sel markdown tidak jadi file `.md` terpisah.** Semuanya jadi konstanta `MD[...]` di
   `build_split_notebooks.py`, dibangun lewat helper `section()`/`banner()` yang memaksa palet
   terdaftar. Palet baru jadi tidak mungkin diselipkan tanpa mengubah helper-nya.
2. **Sel 43b tidak diberi ringkasan `ALL_GATES_PASS` sendiri** — sel 47 sudah melakukannya di
   `02`. Di `01`, agregasi itu ada di sel verifikasi artifact.

### Yang masih terbuka

- **Task 0**: puncak RSS profil `full` di mesin lokal belum diukur (`tiny` = 1,63 GB).
  Kriteria terima `< 4 GB` belum terbukti.
- **Ekuivalensi lunak** (`best_val_mse` dan MSE test dalam toleransi `1e-6` terhadap notebook
  tunggal) belum diuji; hanya ekuivalensi keras yang dijalankan.
- Belum ada yang di-commit.

---
*Status: SELESAI kecuali dua pengukuran di atas.*
