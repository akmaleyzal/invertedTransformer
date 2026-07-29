# Panduan Menjalankan `notebooks/iTransformer.ipynb` di Kaggle

Dokumen ini menjelaskan cara menjalankan notebook dari awal sampai menghasilkan artifact
yang bisa dipakai, **dengan asumsi Anda memakai Kaggle gratisan (GPU T4 ×2)**. Fokus utama:
bagaimana menyiasati batas 12 jam per sesi tanpa kehilangan progres.

Semua angka batasan di bawah diverifikasi dari dokumentasi Kaggle (lihat §11 Sumber).
Semua estimasi waktu adalah **estimasi**, bukan janji — §5.3 menjelaskan cara mengukurnya
sendiri setelah epoch pertama.

---

## 1. Batasan Kaggle yang menentukan seluruh alur kerja

| Batas | Nilai | Dampak ke notebook ini |
| --- | --- | --- |
| Runtime maksimum per sesi | **12 jam** (CPU & GPU), 9 jam TPU | Profil `full` **tidak muat** dalam satu sesi. Harus dipecah per tahap. |
| Kuota GPU mingguan | **30 jam/minggu per akun** | Program lengkap (model + baseline + ablasi + walk-forward) ≈ 25–40 jam GPU → **butuh ±2 minggu**. |
| Idle timeout saat editing | **20 menit** tanpa aktivitas | Jangan jalankan training panjang di mode interaktif sambil ditinggal. Pakai **Save & Run All**. |
| Disk `/kaggle/working` | **20 GB**, otomatis tersimpan sebagai output versi | Checkpoint semua tahap ≈ 2–4 GB. Aman, tapi jangan menulis dataset besar ke sini. |
| RAM | **±29 GB**, 4 CPU core (sesi GPU) | Cukup. Titik puncak ada di tahap feature engineering (§7). |
| `/kaggle/input` | **read-only** | Notebook tidak pernah menulis ke sana. Checkpoint lama dibaca dari sini saat resume. |

> **Kunci yang harus dipahami:** kalau sesi mati di jam ke-12 dalam mode interaktif tanpa
> Anda menyimpan versi, **isi `/kaggle/working` hilang**. Karena itu notebook sekarang
> berhenti sendiri sebelum batas (lihat §4, `session_budget_hours`).

---

## 2. Persiapan sekali saja

### 2.1 Upload data sebagai Kaggle Dataset

Buat satu Dataset berisi **12 file** dari `data/raw/`:

```
btc_usdt_binance_2018.parquet   ...   btc_usdt_binance_2026.parquet   (9 file)
xauusd_2018_2026.parquet
US_Dollar_Index_2018_2026.parquet
fed_economic_data_2018_2026.parquet
```

Upload sebagai Dataset (bukan Utility Script), visibility Private.

**Nama slug bebas.** Notebook sekarang *mencari* folder yang berisi 12 file itu di
`/kaggle/input` (termasuk satu level subfolder, untuk kasus upload berbentuk folder ter-zip).
Kalau pencarian gagal, assertion-nya menyebutkan file mana yang hilang.

### 2.2 Setting notebook

- **Settings → Accelerator → GPU T4 ×2**
- **Settings → Internet → Off**
  Off membuat run reproducible dan tidak memakan izin jaringan. Satu-satunya yang butuh
  internet adalah `onnxruntime` untuk cek parity ONNX — kalau tidak ada, notebook
  melewatinya dengan status `SKIPPED` dan tidak gagal. Nyalakan internet hanya untuk
  satu sesi export kalau Anda memang butuh file `.onnx` terverifikasi.
- **Add Input → Datasets →** pilih dataset dari §2.1.

---

## 3. Tiga profil dan kegunaannya

`PROFILE` ada di sel konfigurasi (sel kode ke-2, bertanda `EDIT THIS ONE LINE`).

| Profil | Rentang data | L / H | Model | Perkiraan waktu | Untuk apa |
| --- | --- | --- | --- | --- | --- |
| `tiny` | 3 bulan (2021 Q1) | 120 / 15 | d_model 64, 1 epoch | ±3–5 menit **di CPU** | Membuktikan notebook jalan tanpa memakai kuota GPU |
| `smoke` | 6 bulan (2021 H1) | 480 / 60 | d_model 128, 2 epoch | ±15 menit di T4 ×2 | Membuktikan semua sanity gate PASS dengan data nyata |
| `full` | 2018-01-02 → 2026-05-31 | 1440 / 60 | d_model 512, 30 epoch | **bertahap, lihat §5** | Hasil yang dilaporkan |

> Angka dari `tiny` dan `smoke` **tidak punya arti ilmiah**. Keduanya menguji pipa, bukan model.
> Jangan pernah melaporkan MASE atau Sharpe dari profil itu.

Alur wajib: `smoke` **PASS dulu**, baru `full`. Kalau ada gate yang FAIL di `smoke`,
menjalankan `full` hanya membuang beberapa jam kuota GPU.

---

## 4. Tombol yang perlu Anda sentuh

Semuanya ada di dua sel pertama bagian §2 Configuration.

```python
# --- sel "EDIT THIS ONE LINE" ---
KAGGLE_RAW_DIR    = Path("/kaggle/input/itransformer-btc-raw")  # fallback; auto-discovery jalan duluan
KAGGLE_RESUME_DIR = None        # diisi saat melanjutkan sesi (§6)
PROFILE           = "smoke"     # "tiny" | "smoke" | "full"

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

| Sesi | Setelan | Isi | Perkiraan |
| --- | --- | --- | --- |
| **0** | `PROFILE="smoke"` | Semua gate PASS | 15 menit |
| **1** | `PROFILE="full"`, `run_baselines=False`, `run_ablation=False` | Model utama saja | 2–4 jam |
| **2** | resume sesi 1, `run_baselines=True`, `run_ablation=False` | Semua baseline | 6–10 jam |
| **3** | resume sesi 2, `run_ablation=True` | Tabel ablasi | 8–11 jam (mungkin butuh 2 sesi) |
| **4+** | resume, `run_walkforward=True` | Walk-forward | 8–15 jam (2 sesi) |
| **5+** | ulangi sesi 1 dengan `seed ∈ {1, 7, 13, 2024}` | Variansi seed | 2–4 jam per seed |

Di setiap sesi, model dari tahap sebelumnya **di-resume dari checkpoint dalam hitungan detik**
(karena epoch-nya sudah selesai semua), lalu tahap barunya yang dilatih. Baseline yang belum
sempat selesai di sesi 2 akan dilanjutkan di sesi 3 dari epoch terakhirnya.

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

**RAM host (±29 GB).** Puncaknya di sel hygiene fitur (§8 notebook): matriks fitur dibuat
`float64` untuk perhitungan kuantil, lalu dipotong ke `float32`. Perkiraan puncak 8–12 GB
pada profil `full`. Aman, tapi jangan menjalankan sel berat lain secara paralel.

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
| `AssertionError: RAW_DIR=... is missing N file(s)` | Dataset belum ditambahkan, atau isinya kurang | Add Input → Datasets. Pastikan jumlah file = 12. |
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

- [ ] Accelerator = **GPU T4 ×2** (bukan P100, bukan None)
- [ ] Dataset 12 file sudah di-Add Input
- [ ] `PROFILE` sesuai rencana sesi
- [ ] Kalau melanjutkan: output sesi sebelumnya sudah di-Add Input **dan** `KAGGLE_RESUME_DIR` terisi
- [ ] Tombol tahap (`run_baselines`, `run_ablation`, `run_walkforward`) sesuai §5.2
- [ ] `session_budget_hours` ≤ 11
- [ ] Sesi `full` dijalankan lewat **Save Version → Save & Run All**, bukan ditinggal di editor
- [ ] Setelah selesai: cek `ALL GATES PASS` sebelum mempercayai satu angka pun

Dan satu aturan yang tidak bisa ditawar: **kalau hasilnya terlihat terlalu bagus, itu
kebocoran data sampai terbukti sebaliknya.** Sharpe di atas 3 pada frekuensi 1 menit, atau
R² di atas 0,05, adalah alasan untuk mencari bug — bukan untuk merayakan.

---

## 11. Sumber

- [Kaggle Notebooks Documentation](https://www.kaggle.com/docs/notebooks) — 12 jam eksekusi CPU/GPU, 9 jam TPU, 20 GB `/kaggle/working`, idle timeout 20 menit, Save & Run All
- [Efficient GPU Usage Tips](https://www.kaggle.com/docs/efficient-gpu-usage) — kuota 30 jam GPU / 20 jam TPU per minggu
- [Notebooks update: New GPU (T4s) options & more CPU RAM](https://www.kaggle.com/product-feedback/361104) — T4 ×2, ±29 GB RAM, 4 CPU core
- [Tips for troubleshooting out-of-memory errors in Kaggle Notebooks](https://www.kaggle.com/getting-started/188347)
