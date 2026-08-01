# Panduan Menjalankan Pipeline Dua-Notebook di Kaggle

Pipeline ini dipecah menjadi **dua notebook**, dan pembagiannya mengikuti satu garis:
apa yang butuh GPU dan apa yang tidak.

| Notebook | Dijalankan di | Menghasilkan / memakai |
| --- | --- | --- |
| `notebooks/01_preprocess.ipynb` | **mesin lokal Anda** | Membaca `data/raw/`, menulis artifact beku ke `data/processed/features_<profile>/` |
| `notebooks/02_train.ipynb` | **Kaggle GPU T4 ×2** | Memakai artifact itu; **tidak pernah membuka data mentah** |

Kenapa dipisah, dua alasan yang keduanya terukur:

1. **Kuota.** Tahap preprocessing tidak menyentuh GPU sama sekali, tetapi dalam alur satu
   notebook ia dijalankan ulang di **setiap** sesi GPU — dan programnya butuh 5+ sesi.
2. **Reproducibility.** Kalau matriks fitur dibangun ulang setiap sesi, tidak ada bukti bahwa
   sesi ke-2 melatih di atas matriks yang identik dengan sesi ke-1. Model utama, baseline, dan
   tabel ablasi dilatih di sesi berbeda lalu dibandingkan seolah-olah inputnya sama. Sekarang
   input itu **dibekukan dan diverifikasi hash**, jadi kesamaannya dibuktikan, bukan diasumsikan.

> `notebooks/iTransformer.ipynb` (satu notebook, 70 sel) **masih ada** sebagai referensi dan
> sebagai tolok ukur ekuivalensi. Kedua notebook baru dibangun darinya lewat
> `tools/build_split_notebooks.py`, yang menyalin sel secara byte-identik — jadi keduanya bukan
> tulisan ulang, melainkan partisi. Sudah diuji: `features.npy` dari `01` **bit-identik** dengan
> matriks `X` notebook tunggal pada profil `tiny`.

Semua angka batasan di bawah diverifikasi dari dokumentasi Kaggle (lihat §11 Sumber).
Semua estimasi waktu adalah **estimasi**, bukan janji — §5.3 menjelaskan cara mengukurnya
sendiri setelah epoch pertama.

---

## 1. Batasan Kaggle yang menentukan seluruh alur kerja

Semua batas di bawah berlaku untuk **`02_train.ipynb`**. `01_preprocess.ipynb` berjalan di
mesin Anda sendiri, jadi tidak terkena satu pun dari ini.

| Batas | Nilai | Dampak ke sesi training |
| --- | --- | --- |
| Runtime maksimum per sesi | **12 jam** (CPU & GPU), 9 jam TPU | Profil `full` **tidak muat** dalam satu sesi. Harus dipecah per tahap. |
| Kuota GPU mingguan | **30 jam/minggu per akun** | Program lengkap (model + baseline + ablasi + walk-forward) ≈ 25–40 jam GPU → **butuh ±2 minggu**. Preprocessing tidak lagi memakan jatah ini. |
| Idle timeout saat editing | **20 menit** tanpa aktivitas | Jangan jalankan training panjang di mode interaktif sambil ditinggal. Pakai **Save & Run All**. |
| Disk `/kaggle/working` | **20 GB**, otomatis tersimpan sebagai output versi | Checkpoint semua tahap ≈ 2–4 GB. Artifact masuk lewat `/kaggle/input`, bukan ditulis ke sini. |
| RAM | **±29 GB**, 4 CPU core | Sekarang lapang: puncak sesi training tinggal ±1 GB matriks fitur + model, bukan 8–12 GB frame polars. |
| `/kaggle/input` | **read-only** | Artifact **dan** checkpoint lama dibaca dari sini. Notebook tidak pernah menulis ke sana. |

> **Kunci yang harus dipahami:** kalau sesi mati di jam ke-12 dalam mode interaktif tanpa
> Anda menyimpan versi, **isi `/kaggle/working` hilang**. Karena itu notebook sekarang
> berhenti sendiri sebelum batas (lihat §4, `session_budget_hours`).

---

## 2. Persiapan sekali saja

### 2.0 Bangun kedua notebook dari sumbernya

Kedua notebook adalah **hasil build**, bukan file yang diedit tangan. Kalau
`iTransformer.ipynb` berubah, bangun ulang:

```powershell
$PY = "D:\pythonProject\invertedTransformer\.venv\Scripts\python.exe"
& $PY tools/build_split_notebooks.py
```

Skrip itu meng-assert setiap sel sumber tersalurkan ke salah satu notebook (atau dipecah, atau
sengaja diganti), lalu AST-parse semua sel kode. Kalau ada sel sumber yang tidak
terpetakan, skrip gagal — bukan diam-diam menghasilkan notebook cacat.

### 2.1 Jalankan `01_preprocess.ipynb` di mesin lokal

```powershell
# working directory = notebooks/ ; data/raw ditemukan otomatis di ../data/raw
& $PY -m jupyter lab notebooks/01_preprocess.ipynb
```

Setel `PROFILE`, Run All, dan tunggu sampai seluruh gate mencetak **PASS**. Hasilnya:

```
data/processed/features_<profile>/
├── features.npy            float32 (T, N)   matriks terstandardisasi
├── close.npy               float64 (T,)     harga close mentah
├── timestamps.npy          int64            epoch-mikrodetik UTC
├── scaler.json             mean/std/winsor + split tempat ia difit
├── feature_manifest.json   urutan variat, grup, target_index, d frac-diff, offset gold, ...
└── prep_metadata.json      hash data mentah + hash matriks + hash manifest + frozen fields
```

Ukuran: ±32 MB pada `tiny`, ±1,0 GB pada `full`.

**Tahap ini tidak memakai sesi Kaggle sama sekali.** Kalau mesin lokal Anda tidak cukup,
sesi **CPU** Kaggle (bukan GPU — tidak memotong kuota 30 jam) bisa dipakai sebagai fallback:
attach dataset data mentah, jalankan `01`, lalu ambil `features_<profile>/` dari output versi.

### 2.2 Upload artifact sebagai Kaggle Dataset

Buat satu Dataset berisi folder `features_<profile>/` dari langkah 2.1. Visibility Private.

**Nama slug bebas.** `02_train.ipynb` *mencari* folder yang berisi enam file itu di
`/kaggle/input` (termasuk satu level subfolder, dan menerima folder yang *berisi*
`features_<profile>/` maupun folder yang *adalah* `features_<profile>/`). Kalau pencarian
gagal, assertion-nya menyebutkan file mana yang hilang.

> **Hanya artifact `full` yang perlu dipublikasikan.** `tiny` dan `smoke` murah dibangun ulang
> secara lokal, jadi tidak layak memakan kuota Dataset.
>
> **Data mentah tidak perlu diunggah lagi** kecuali Anda memakai fallback sesi CPU di §2.1.

### 2.3 Setting notebook training

- **Settings → Accelerator → GPU T4 ×2**
- **Settings → Internet → Off**
  Off membuat run reproducible dan tidak memakan izin jaringan. Satu-satunya yang butuh
  internet adalah `onnxruntime` untuk cek parity ONNX — kalau tidak ada, notebook
  melewatinya dengan status `SKIPPED` dan tidak gagal. Nyalakan internet hanya untuk
  satu sesi export kalau Anda memang butuh file `.onnx` terverifikasi.
- **Add Input → Datasets →** pilih dataset artifact dari §2.2.

---

## 3. Tiga profil dan kegunaannya

`PROFILE` ada di sel bertanda `EDIT THIS ONE LINE` — **di kedua notebook**, dan **keduanya
harus sama**. `profile` adalah salah satu *frozen field* (§4.1): kalau berbeda, `02` menolak
berjalan alih-alih melatih di atas matriks yang salah.

| Profil | Rentang data | L / H | Model | Artifact | Waktu training | Untuk apa |
| --- | --- | --- | --- | --- | --- | --- |
| `tiny` | 3 bulan (2021 Q1) | 120 / 15 | d_model 64, 1 epoch | ±32 MB | ±3–5 menit **di CPU** | Membuktikan kedua notebook jalan tanpa memakai kuota GPU |
| `smoke` | 6 bulan (2021 H1) | 480 / 60 | d_model 128, 2 epoch | ±120 MB | ±15 menit di T4 ×2 | Membuktikan semua sanity gate PASS dengan data nyata |
| `full` | 2018-01-02 → 2026-05-31 | 1440 / 60 | d_model 512, 30 epoch | ±1,0 GB | **bertahap, lihat §5** | Hasil yang dilaporkan |

> Angka dari `tiny` dan `smoke` **tidak punya arti ilmiah**. Keduanya menguji pipa, bukan model.
> Jangan pernah melaporkan MASE atau Sharpe dari profil itu.

Alur wajib: `smoke` **PASS dulu**, baru `full`. Kalau ada gate yang FAIL di `smoke`,
menjalankan `full` hanya membuang beberapa jam kuota GPU.

**Satu artifact per profil, saling terisolasi.** Direktorinya diberi kunci nama profil, jadi
artifact `tiny` tidak mungkin dikira `full` — dan kalaupun path-nya dipaksa, pemeriksaan
frozen-field menangkapnya.

---

## 4. Tombol yang perlu Anda sentuh

### 4.1 Frozen vs free — pembagian yang menentukan segalanya

`CFG` masih satu sumber kebenaran, tetapi field-nya sekarang punya dua kelas.

**Frozen** — membentuk matriks fitur. `02_train` **wajib** memakai nilai yang sama dengan
artifact; kalau tidak, sesi berhenti dan pesannya menyebut field mana dan kedua nilainya.

```
profile, grid_start, grid_end, train_end, val_end, test_end,
seq_len, pred_len, blocks, macro_n_pca, fracdiff_grid, fracdiff_width,
winsor_q, collinear_thresh, gold_utc_offset_h
```

Dua di antaranya sering disalahpahami:

- **`train_end` ikut dibekukan** karena **scaler difit pada baris `t <= train_end`**.
  Mengubahnya di sisi training bukan perbedaan konfigurasi — itu **kebocoran data**.
- **`seq_len` ikut dibekukan** karena warm-up truncation dihitung `1440 + seq_len + 60`,
  jadi ia menentukan baris mana yang eksis sama sekali.

**Free** — bebas berubah tiap sesi training, tanpa membangun ulang artifact:
`d_model`, `n_heads`, `e_layers`, `d_ff`, `dropout`, `lr`, `weight_decay`, `batch_size`,
`epochs`, `loss`, `seed`, dan semua tombol tahap/anggaran sesi.

> `dataclass Config` **diduplikasi** di kedua notebook, dengan sengaja. Duplikasi itulah yang
> membuat pemeriksaan frozen-field mungkin: kalau kelak keduanya menyimpang, sesi **gagal
> terlihat**, bukan menghasilkan angka yang salah diam-diam.

### 4.2 Tombol di `01_preprocess.ipynb`

```python
# --- sel "EDIT THIS ONE LINE" ---
KAGGLE_RAW_DIR = Path("/kaggle/input/itransformer-btc-raw")  # hanya untuk fallback sesi CPU
PROFILE        = "smoke"     # "tiny" | "smoke" | "full"
```

Selain itu tidak ada yang perlu disentuh. Setiap field frozen lainnya sudah punya nilai yang
dipilih dengan alasan; mengubahnya berarti membangun artifact baru.

### 4.3 Tombol di `02_train.ipynb`

```python
# --- sel "EDIT THIS ONE LINE" ---
KAGGLE_ARTIFACT_DIR = Path("/kaggle/input/itransformer-btc-features")  # fallback; auto-discovery duluan
KAGGLE_RESUME_DIR   = None        # diisi saat melanjutkan sesi (§6)
PROFILE             = "smoke"     # WAJIB sama dengan artifact

# --- dataclass Config ---
run_baselines           = True   # AR(60), DLinear, PatchTST, TimeXer, VanillaTF
run_vanilla_transformer = True   # baseline paling mahal (attention L×L)
run_ablation            = True   # tabel 5 baris: BTC only ... BTC + all exog
run_walkforward         = False  # ±5× runtime; bukti utama untuk laporan akhir
session_budget_hours    = 11.0   # berhenti sendiri sebelum tembok 12 jam
reserve_hours           = 0.5    # sisakan segini untuk menulis checkpoint & simpan versi
```

**Cara kerja pengaman waktu.** Setiap akhir epoch trainer mengecek sisa anggaran. Kalau
tersisa ≤ `reserve_hours`, training berhenti dengan pesan jelas, checkpoint sudah tertulis,
dan sel-sel berikutnya tetap jalan. Ini bedanya antara "lanjut di sesi berikutnya" dan
"mulai dari nol". Setel `session_budget_hours = 0` untuk mematikan pengaman (tidak disarankan
di Kaggle).

### 4.4 Enam aturan penolakan artifact

Sebelum menyentuh GPU, `02` menjalankan enam pemeriksaan dan **gagal keras** di yang pertama
tidak lolos. Semuanya dicetak dulu sebagai tabel `PASS`/`FAIL`, jadi sesi yang ditolak
memberi tahu nilai mana yang berbeda dari mana.

| # | Aturan | Menangkap |
| --- | --- | --- |
| 1 | `sha256(features.npy)` = yang dicatat `prep_metadata.json` | upload terpotong, file tertukar |
| 2 | `sha256(feature_manifest.json)` cocok | manifest diedit tangan |
| 3 | semua frozen field `CFG` = milik artifact | notebook menyimpang dari artifact |
| 4 | `features.shape` = `(len(timestamps), len(feature_order))`, `close` sepanjang itu juga | artifact tidak konsisten |
| 5 | `feature_order[target_index]` = `btc_logret_1` | matriks tertukar urutan variat |
| 6 | `sha256(scaler.json)` cocok | scaler menyimpang tanpa terdeteksi |

Aturan 5 yang paling penting dipahami: matriks dengan urutan variat tertukar **tidak error** —
ia menghasilkan angka yang tampak wajar. Itu mode kegagalan terburuk yang ada di proyek ini.

---

## 5. Rencana multi-sesi untuk `PROFILE = "full"`

### 5.1 Kenapa tidak bisa satu sesi

Pada profil `full`:

- grid master ±4,42 juta menit; split train 2018→2023 ≈ 3,15 juta menit
- `train_stride = 5` → **±630 ribu window per epoch**
- batch efektif 128 × 2 GPU = 256 → **±2.460 step per epoch**

Setiap model — termasuk baseline sesederhana `DLinear` — membaca window yang sama, jadi
biaya per epoch didominasi **pemindahan data (±91 MB per batch)**, bukan ukuran model.
Konsekuensinya: baseline kecil **tidak** jauh lebih murah daripada model utama.

| Tahap | Model dilatih | Perkiraan |
| --- | --- | --- |
| iTransformer (utama) | 1 | 1,5–3 jam |
| Baseline terlatih | 5–6 | 6–9 jam |
| Ablasi | 5 | 7–15 jam |
| Walk-forward | ±5 fold | 8–15 jam |
| **Total** | | **±25–40 jam GPU** |

Dengan kuota 30 jam/minggu, program lengkap memakan waktu sekitar **dua minggu kalender**.

### 5.2 Pembagian sesi yang disarankan

| Sesi | Di mana | Setelan | Isi | Perkiraan |
| --- | --- | --- | --- | --- |
| **L** | **lokal** | `01_preprocess`, `PROFILE="full"` | Bangun + verifikasi artifact, unggah sebagai Dataset | sekali saja, **0 jam GPU** |
| **0** | Kaggle | `PROFILE="smoke"` | Semua gate PASS | 15 menit |
| **1** | Kaggle | `PROFILE="full"`, `run_baselines=False`, `run_ablation=False` | Model utama saja | 2–4 jam |
| **2** | Kaggle | resume sesi 1, `run_baselines=True`, `run_ablation=False` | Semua baseline | 6–10 jam |
| **3** | Kaggle | resume sesi 2, `run_ablation=True` | Tabel ablasi | 8–11 jam (mungkin butuh 2 sesi) |
| **4+** | Kaggle | resume, `run_walkforward=True` | Walk-forward | 8–15 jam (2 sesi) |
| **5+** | Kaggle | ulangi sesi 1 dengan `seed ∈ {1, 7, 13, 2024}` | Variansi seed | 2–4 jam per seed |

Sesi **L** dijalankan **satu kali** dan tidak diulang. Sesi 0 sampai 5+ semuanya memakai
artifact yang sama, dengan hash yang sama — itulah yang membuat perbandingan antar-sesi
(model utama vs baseline vs ablasi vs seed) sah, karena inputnya terbukti identik.

Di setiap sesi Kaggle, model dari tahap sebelumnya **di-resume dari checkpoint dalam hitungan
detik** (karena epoch-nya sudah selesai semua), lalu tahap barunya yang dilatih. Baseline yang
belum sempat selesai di sesi 2 akan dilanjutkan di sesi 3 dari epoch terakhirnya.

> Sesi 0 untuk `smoke` butuh artifact `smoke`. Bangun secara lokal dan unggah, **atau** cukup
> jalankan `01` + `02` profil `smoke` sepenuhnya di lokal kalau mesin Anda sanggup — sesi 0
> memang hanya membuktikan gate, bukan menghasilkan angka yang dilaporkan.

### 5.3 Cara mengukur sendiri, bukan menebak

Setelah epoch pertama trainer mencetak baris seperti:

```
  [itransformer] ep  1/30  train 0.981234  val_mse 0.996120  lr 3.00e-04  gnorm 0.84  212s  6.4GB
```

`212s` itu detik per epoch. Kalikan dengan `epochs` untuk memperkirakan total tahap tersebut.
Kalau hasilnya melewati `session_budget_hours`, turunkan salah satu:

- `CFG.epochs` (misal 30 → 15 untuk tahap baseline dan ablasi),
- `CFG.train_stride` (5 → 15 memangkas window per epoch jadi ±1/3),
- `CFG.eval_max_windows` (200.000 → 100.000 mempercepat validasi tiap epoch).

Ketiganya melemahkan hasil, jadi terapkan pada tahap pembanding (baseline/ablasi) dan
**biarkan model utama pakai setelan penuh**. Kalau `eval_max_windows` diubah, catat nilainya:
jarak antar-window evaluasi ikut berubah dan notebook mencetak jarak itu di header backtest.

---

## 6. Prosedur resume antar-sesi — langkah demi langkah

Ini bagian terpenting dari dokumen ini. Ikuti persis.

### Sesi yang sedang berjalan (misalnya sesi 1)

1. Biarkan notebook berhenti sendiri, atau tunggu selesai.
2. Klik **Save Version → Save & Run All (Commit)** — atau **Quick Save** kalau Anda sudah
   menjalankan semuanya secara interaktif dan hanya ingin menyimpan output apa adanya.
3. Tunggu sampai versi selesai. Isi `/kaggle/working` sekarang menjadi **output versi** dan
   permanen di akun Anda.

Struktur output yang tersimpan:

```
/kaggle/working/
├── checkpoints/full_L1440_H60_d512_s42/
│   ├── itransformer_last.pt      <- untuk melanjutkan
│   ├── itransformer_best.pt      <- bobot terbaik
│   ├── DLinear_last.pt, TimeXer_last.pt, abl_BTC_only_last.pt, ...
├── runs/full_L1440_H60_d512_s42/
│   └── itransformer_metrics.jsonl
└── models/full_L1440_H60_d512_s42/
    ├── model.pt, model_scripted.pt, model.onnx
    ├── scaler.json, feature_manifest.json, config.json, metadata.json
    └── inference_example.py
```

### Sesi berikutnya (sesi 2)

4. Buka notebook, **Add Input → Notebook Output →** pilih versi dari langkah 3.
   Kaggle me-mount-nya di `/kaggle/input/<nama-notebook>/`.

   > **Dataset artifact harus tetap ter-attach.** Sesi baru sekarang punya **dua** input:
   > artifact fitur (§2.2) dan output sesi sebelumnya. Kalau artifact-nya lepas, notebook
   > berhenti di sel penemuan artifact sebelum menyentuh GPU — jadi kesalahan ini murah, tetapi
   > tetap membuang waktu antrian.

5. Isi knob resume dengan path folder checkpoint di dalam mount itu:

```python
KAGGLE_RESUME_DIR = "/kaggle/input/itransformer-btc-1min/checkpoints/full_L1440_H60_d512_s42"
```

> Cara memastikan path-nya benar: jalankan satu sel `!ls -R /kaggle/input | head -50`,
> atau lihat panel **Input** di sisi kanan editor.

6. Ubah tombol tahap sesuai rencana (§5.2), lalu jalankan.
7. Pastikan log mencetak baris seperti ini:

```
resume from  /kaggle/input/.../checkpoints/full_L1440_H60_d512_s42   (previous session)
[itransformer] resumed from epoch 17 (best val 0.994210)
```

Kalau baris `resumed from epoch ...` **tidak muncul**, resume gagal dan model dilatih dari
nol. Penyebab tersering: `run_id` berbeda. `run_id` dibentuk dari
`{profile}_L{seq_len}_H{pred_len}_d{d_model}_s{seed}` — mengubah salah satu dari lima nilai
itu menghasilkan folder checkpoint yang berbeda, dan itu memang disengaja: checkpoint dari
arsitektur lain tidak boleh dipakai diam-diam.

**Checkpoint dibaca dari `RESUME_DIR` dan selalu ditulis ke `CKPT_DIR`.** Jadi `/kaggle/input`
tetap read-only dan sesi baru menghasilkan output-nya sendiri, yang menjadi input untuk
sesi berikutnya lagi. Rantai ini bisa diteruskan sepanjang yang Anda butuhkan.

---

## 7. Memori, VRAM, dan disk

**RAM host (±29 GB).** Sesi training tidak lagi memegang frame polars milik preprocessing.
Puncaknya sekarang tinggal matriks fitur (`float32`, ±1,0 GB pada `full`) plus model dan batch
— dengan target **di bawah 4 GB**. Matriks dimuat penuh, **bukan** `mmap`: sanity gate
kebocoran menulis ulang kolom target di tempat lalu memulihkannya, dan mapping read-only akan
menolak penulisan itu.

**RAM saat preprocessing (mesin lokal).** Ini yang dulu menjadi plafon proyek. Setelah
optimisasi tahap hygiene — cast ke `Float32` di dalam polars sebelum `to_numpy()`, split train
sebagai *view* alih-alih salinan, `np.clip(..., out=...)` in-place, standardisasi in-place
dengan akumulator `float64` — `01_preprocess` mencetak `peak_rss_gb()` di akhir tahapnya
sendiri. Terukur **1,63 GB pada profil `tiny`**; angka profil `full` belum diukur dan itu
satu-satunya bagian §2.1 yang masih terbuka.

> Yang **ditolak dengan sengaja**: menurunkan `master` ke `float32`. Harga BTC ~60.000 dalam
> `float32` hanya menyisakan ~3 digit signifikan pada log-return 1 menit (~1e-3) — itu merusak
> target itu sendiri, bukan menghemat memori.

**VRAM (15 GB per T4).** Model utama memakai ±6–8 GB pada batch 256. Yang paling rawan adalah
`VanillaTransformer` (attention L×L = 1440×1440); notebook sudah memperkecil batch-nya
seperempat. Kalau tetap OOM:

```python
CFG.batch_size = 64                   # per GPU; batch efektif = 64 × 2
CFG.run_vanilla_transformer = False   # atau matikan saja baseline ini
```

**Disk 20 GB.** Setiap checkpoint menyimpan bobot + state optimizer + riwayat, ±80–150 MB
untuk model utama. Dengan ±13 tag (`_last` + `_best`) totalnya ±2–4 GB. Kalau mendekati batas,
hapus checkpoint tahap yang sudah selesai sebelum menyimpan versi:

```python
# hapus HANYA setelah tahap tersebut selesai dan hasilnya sudah tercatat di metadata.json
for f in CKPT_DIR.glob("abl_*_last.pt"):
    f.unlink()
```

---

## 8. Interaktif vs Save & Run All

| | Interaktif (Run All di editor) | Save Version → Save & Run All |
| --- | --- | --- |
| Batas waktu | 12 jam, **tapi mati setelah 20 menit idle** | 12 jam penuh, headless |
| Bisa ditinggal | Tidak | Ya |
| Output tersimpan | Hanya kalau Anda klik Save / Quick Save | Otomatis |
| Cocok untuk | `tiny`, `smoke`, debugging, mengecek path | semua tahap `full` |

Pola yang disarankan: jalankan `smoke` secara interaktif untuk memastikan path dan gate benar,
lalu setiap tahap `full` dijalankan lewat **Save & Run All** dan ditinggal.

---

## 9. Troubleshooting

| Gejala | Penyebab | Tindakan |
| --- | --- | --- |
| `AssertionError: ART_DIR=... is missing N artifact file(s)` | Dataset artifact belum di-attach, atau `PROFILE` tidak cocok dengan folder yang diunggah | Add Input → Datasets. Pastikan folder bernama `features_<profile>/` dan berisi 6 file. |
| `artifact REJECTED - rule 1 features sha256` | `features.npy` tidak utuh — upload terpotong, atau file dari artifact lain | Unggah ulang artifact. Jangan mengedit isi folder artifact dengan tangan. |
| `artifact REJECTED - rule 3 frozen config` | Field frozen di `02` berbeda dari artifact | Samakan nilainya dengan yang disebut pesan errornya. Kalau memang ingin berubah, **bangun ulang artifact** di `01` — jangan paksa di sisi training. |
| `artifact REJECTED - rule 5 target variate` | Urutan variat matriks tertukar | Bangun ulang artifact. Jangan lanjut: matriks ini menghasilkan angka yang tampak wajar tetapi salah. |
| `train boundary disagrees: CFG.train_end=...` | `train_end` diubah di sisi training padahal scaler difit pada batas lama | Kembalikan `train_end`, atau bangun ulang artifact. Ini kebocoran data, bukan ketidakcocokan. |
| `AssertionError: RAW_DIR=... is missing N file(s)` | Hanya muncul di `01_preprocess` (atau fallback sesi CPU): dataset data mentah belum ada | Pastikan `data/raw/` berisi 12 file, atau Add Input → Datasets untuk fallback. |
| `AssertionError: anchor '...' found 0 times` | `iTransformer.ipynb` diedit sampai batas pemecahan sel bergeser | Perbarui anchor di `tools/build_split_notebooks.py`, lalu bangun ulang kedua notebook. |
| `resumed from epoch ...` tidak muncul | `run_id` beda, atau `KAGGLE_RESUME_DIR` salah path | Samakan `profile`, `seq_len`, `pred_len`, `d_model`, `seed`. Cek `!ls` pada path resume. |
| Gate `shift` atau `future-perturbation` FAIL | Ada fitur yang melihat masa depan | **Jangan lanjut training.** Ini bug kebocoran data, bukan gangguan kecil. |
| Gate `overfit-1-batch` FAIL | Plumbing model/data rusak | Jangan jalankan run panjang. Periksa bentuk tensor dan learning rate. |
| Gate `leakage` FAIL | Fitur mengandung target | Cari fitur yang dihitung dari `close` masa depan. |
| `CUDA out of memory` | Batch terlalu besar, biasanya di VanillaTF | Turunkan `CFG.batch_size`, atau `run_vanilla_transformer = False`. |
| Sesi mati mendadak tanpa output | Kena tembok 12 jam di mode interaktif | Turunkan `session_budget_hours`, dan pakai Save & Run All. |
| `ONNX export/verify unavailable` | `onnxruntime` tidak terpasang (Internet Off) | Normal. TorchScript tetap diverifikasi. Nyalakan internet satu sesi kalau butuh `.onnx`. |
| DataLoader lambat / GPU idle | 4 core CPU jadi bottleneck | `CFG.num_workers = 3`, atau naikkan `train_stride`. |
| Kuota GPU habis | 30 jam/minggu terpakai | Sisa minggu itu jalankan `tiny` di CPU, atau tunggu reset kuota. |

---

## 10. Checklist sebelum menekan Run

**Sekali, di mesin lokal (`01_preprocess.ipynb`):**

- [ ] `python tools/build_split_notebooks.py` sudah dijalankan setelah perubahan terakhir pada `iTransformer.ipynb`
- [ ] `01` selesai dengan **ALL GATES PASS**, termasuk `artifact_roundtrip`
- [ ] `features_<profile>/` berisi 6 file; hash `features.npy` dicatat
- [ ] Folder itu sudah diunggah sebagai Kaggle Dataset (khusus profil `full`)

**Setiap sesi Kaggle (`02_train.ipynb`):**

- [ ] Accelerator = **GPU T4 ×2** (bukan P100, bukan None)
- [ ] Dataset **artifact** sudah di-Add Input (bukan data mentah)
- [ ] `PROFILE` **sama** dengan profil artifact
- [ ] Tidak ada field frozen (§4.1) yang diubah di notebook ini
- [ ] Kalau melanjutkan: output sesi sebelumnya sudah di-Add Input **dan** `KAGGLE_RESUME_DIR` terisi, **dan** dataset artifact tetap ter-attach
- [ ] Tombol tahap (`run_baselines`, `run_ablation`, `run_walkforward`) sesuai §5.2
- [ ] `session_budget_hours` ≤ 11
- [ ] Sesi `full` dijalankan lewat **Save Version → Save & Run All**, bukan ditinggal di editor
- [ ] Enam aturan penolakan artifact semuanya **PASS** di awal log
- [ ] Setelah selesai: cek `ALL GATES PASS` sebelum mempercayai satu angka pun
- [ ] Hash `features.npy` yang dicetak sesi ini **sama** dengan sesi sebelumnya — kalau berbeda, perbandingan antar-sesi tidak sah

Dan satu aturan yang tidak bisa ditawar: **kalau hasilnya terlihat terlalu bagus, itu
kebocoran data sampai terbukti sebaliknya.** Sharpe di atas 3 pada frekuensi 1 menit, atau
R² di atas 0,05, adalah alasan untuk mencari bug — bukan untuk merayakan.

---

## 11. Sumber

- [Kaggle Notebooks Documentation](https://www.kaggle.com/docs/notebooks) — 12 jam eksekusi CPU/GPU, 9 jam TPU, 20 GB `/kaggle/working`, idle timeout 20 menit, Save & Run All
- [Efficient GPU Usage Tips](https://www.kaggle.com/docs/efficient-gpu-usage) — kuota 30 jam GPU / 20 jam TPU per minggu
- [Notebooks update: New GPU (T4s) options & more CPU RAM](https://www.kaggle.com/product-feedback/361104) — T4 ×2, ±29 GB RAM, 4 CPU core
- [Tips for troubleshooting out-of-memory errors in Kaggle Notebooks](https://www.kaggle.com/getting-started/188347)
