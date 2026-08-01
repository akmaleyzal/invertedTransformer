# Pemisahan Preprocessing dan Training (Kaggle dua-notebook)

## Problem

Seluruh pipeline berjalan dalam satu notebook, sehingga setiap sesi training Kaggle membangun
ulang matriks fitur dari data mentah sebelum menyentuh GPU. Tahap itu tidak membutuhkan GPU
sama sekali, tetapi tetap membakar jam dari kuota **30 jam GPU per minggu** — dan program
lengkap direncanakan berjalan lintas **5+ sesi**. Tahap yang sama juga merupakan puncak
pemakaian RAM sesi, sekitar **3× lebih tinggi** daripada saat training berjalan.

Konsekuensi kedua yang lebih serius: karena matriks fitur dibangun ulang setiap sesi, tidak ada
jaminan bahwa sesi ke-2 melatih di atas matriks yang identik dengan sesi ke-1. Model utama,
baseline, dan tabel ablasi dilatih di sesi yang berbeda-beda, lalu dibandingkan seolah-olah
inputnya sama.

## Evidence

**Terukur:**
- `data/raw` = 240 MB parquet, mengembang menjadi puncak memori **~7–9 GB** pada profil `full`
  (4,42 juta menit × 62 variat; frame polars ~3,1 GB + salinan numpy float64 2,04 GB × beberapa).
- Saat training berjalan, pemakaian turun ke **~2,5 GB** — jadi tahap preprocessing yang
  menentukan batas RAM, bukan modelnya.
- `metadata.json` yang ditulis notebook **tidak memuat hash data maupun hash matriks fitur**,
  meski CLAUDE.md §12.1 memintanya. Tidak ada mekanisme apa pun untuk membuktikan dua sesi
  memakai input yang sama.
- Rencana eksekusi resmi (`docs/KAGGLE_GUIDE.md` §5.2) adalah **5+ sesi**, dan setiap sesi
  mengulang tahap preprocessing yang identik.

**Asumsi — perlu divalidasi lewat pengukuran di Kaggle:**
- Durasi tahap preprocessing pada profil `full` diperkirakan ~10 menit; belum pernah diukur di
  Kaggle. Total pemborosan kuota GPU diperkirakan ~1 jam untuk 5–6 sesi.

## Users

- **Primary**: pemilik proyek — peneliti tunggal yang menjalankan seluruh eksperimen di Kaggle
  free tier (GPU T4 ×2), terikat kuota 30 jam GPU/minggu dan 12 jam per sesi. Kebutuhan muncul
  setiap kali sesi baru dimulai untuk melanjutkan tahap berikutnya.
- **Not for**: penyajian model produksi, inferensi real-time, atau kolaborasi banyak orang.
  Kontrak inferensi dan bundle export sudah ditangani terpisah dan tidak diubah di sini.

## Hypothesis

Kami percaya **membekukan matriks fitur menjadi artifact terpisah yang diverifikasi hash** akan
**menghilangkan pekerjaan CPU dari sesi GPU dan menjamin setiap tahap eksperimen dilatih di atas
input yang identik** untuk **peneliti yang menjalankan program multi-sesi di Kaggle**.

Kami tahu ini benar ketika **sesi GPU tidak lagi menjalankan tahap preprocessing sama sekali,
puncak RAM sesi training turun di bawah 4 GB, dan dua sesi training berbeda melaporkan hash
matriks fitur yang sama persis**.

## Success Metrics

| Metric | Target | Cara diukur |
| --- | --- | --- |
| Sesi Kaggle terpakai untuk preprocessing | **0** | Preprocessing dijalankan di mesin lokal; tidak ada sesi Kaggle sama sekali untuk tahap ini |
| Puncak RSS preprocessing lokal, profil `full` | **< 4 GB** | `peak_rss_gb()` dicetak di akhir tahap hygiene; mesin lokal punya 14,8 GB total, ~4,2 GB bebas |
| Puncak RAM sesi training | **< 4 GB** | Meter RAM Kaggle + `peak_rss_gb()` di akhir tahap |
| Hash matriks fitur lintas sesi | **identik** | Hash dicetak di setiap sesi dan tercatat di `metadata.json` |
| Sesi training dengan artifact tidak cocok | **100% ditolak** | Sesi berhenti, bukan melanjutkan diam-diam |
| Waktu dari Run sampai epoch 1 | **< 3 menit** | Timestamp log; baseline saat ini TBD — perlu diukur di Kaggle |

## Scope

**MVP** — Dua notebook terpisah dengan kontrak artifact yang dipaksakan:

0. **Prasyarat** — puncak memori tahap preprocessing diturunkan sampai muat di mesin lokal
   (14,8 GB total, ~4,2 GB bebas). Tanpa ini langkah 1 tidak bisa dijalankan sama sekali.
1. Notebook preprocessing dijalankan **di mesin lokal**, menghasilkan artifact beku di
   `data/processed/features_{profile}/` berisi matriks fitur, parameter scaler, manifest fitur,
   dan hash yang mengikat keduanya ke data mentah yang dipakai.
2. Notebook training mengonsumsi artifact itu dan **tidak pernah menyentuh data mentah**. Ia
   menemukan artifact di `/kaggle/input` bila ada, jika tidak jatuh ke `data/processed/` lokal —
   satu notebook, dua tempat.
3. Notebook training **menolak berjalan** bila hash atau manifest tidak cocok — gagal keras,
   bukan peringatan.
4. Ketiga profil (`tiny`, `smoke`, `full`) didukung, masing-masing menghasilkan artifact sendiri
   yang saling terisolasi.
5. Alur dua-notebook terdokumentasi di panduan Kaggle, menggantikan alur satu-notebook.

**Out of scope**

- **Memmap/streaming inkremental** (menulis `features.npy` per blok agar puncak turun di bawah
  1,5 GB) — jalur cadangan bila prasyarat 0 ternyata belum cukup. Tidak direncanakan sebelum
  angka puncak sesungguhnya terukur.
- **Menurunkan `master` ke float32** — ditolak: harga BTC ~60.000 dalam float32 hanya menyisakan
  ~3 digit signifikan pada log-return 1 menit (~1e-3), yang merusak target itu sendiri.
- **Penambahan blok fitur baru** (order flow, funding rate, open interest, cross-exchange) —
  pekerjaan terpisah dengan nilai lebih tinggi, tetapi menambah variabel bila digabung ke sini.
- **Otomatisasi publikasi Dataset lewat Kaggle API** — langkah manual dulu.
- **Menggabungkan kembali menjadi satu notebook** sebagai fallback — keputusan sudah diambil:
  pisah keras.
- **Ekstraksi `src/` dan test suite** — arah jangka panjang yang berbeda; PRD ini tidak
  mengasumsikan atau menghalanginya.

## Delivery Milestones

<!-- Business outcomes, not engineering tasks. /plan turns each into a plan. -->
<!-- Status: pending | in-progress | complete -->

| # | Milestone | Outcome | Status | Plan |
|---|---|---|---|---|
| 0 | Preprocessing muat di mesin lokal | Peneliti dapat menjalankan tahap preprocessing profil `full` di laptopnya sendiri tanpa swap, puncak RSS terukur di bawah 4 GB | in-progress | `.claude/plans/split-preprocessing-training.plan.md` (Task 0) |
| 1 | Kontrak artifact | Isi artifact, aturan hash, dan syarat penolakan terdefinisi dan disepakati sebelum ada kode ditulis | complete | `.claude/plans/split-preprocessing-training.plan.md` |
| 2 | Notebook preprocessing | Peneliti dapat menghasilkan artifact beku untuk profil mana pun **di mesin lokal**, tanpa sesi Kaggle sama sekali | complete | `.claude/plans/split-preprocessing-training.plan.md` |
| 3 | Notebook training | Sesi GPU melatih langsung dari artifact, menolak artifact tidak cocok, puncak RAM di bawah 4 GB | complete | `.claude/plans/split-preprocessing-training.plan.md` |
| 4 | Panduan Kaggle diperbarui | Alur dua-notebook — publikasi artifact dan prosedur resume — menggantikan alur lama | complete | `.claude/plans/split-preprocessing-training.plan.md` (Task 7) |
| 5 | Validasi ekuivalensi | Terbukti hasil alur dua-notebook identik dengan alur satu-notebook pada profil `tiny` | complete | `.claude/plans/split-preprocessing-training.plan.md` (Task 8) |

**Bukti per milestone (profil `tiny`, dijalankan 2026-07-30):**

| Milestone | Bukti |
|---|---|
| 1 | 6 file artifact + 6 aturan penolakan di `tools/split_cells/shared_artifact_io.py`; 15 frozen field |
| 2 | `01_preprocess.ipynb` (42 sel) berakhir `ALL CELLS OK` + `ALL GATES PASS`; artifact 32,28 MB, `X (128160, 62)` |
| 3 | `02_train.ipynb` (53 sel) berakhir `ALL CELLS OK`, mencapai tabel ablasi + export, **tanpa membuka `data/raw`** |
| 4 | `docs/KAGGLE_GUIDE.md` §1–§10 memakai alur dua-notebook; `CLAUDE.md` §3, §3.1, §3.2 baru |
| 5 | `features.npy` **bit-identik** dengan `X` notebook tunggal — `sha256 c3ad8cd397d9fb88ca5c11372dd2b1d7`; timestamp dan scaler juga sama |

**Milestone 0 masih terbuka pada satu angka**: puncak RSS profil `full` di mesin lokal belum
diukur. Yang sudah terukur: **1,63 GB pada profil `tiny`**. Kriteria terima `< 4 GB` untuk
`full` karena itu belum terbukti — dan itulah satu-satunya klaim di dokumen ini yang masih
berupa perkiraan.

## Open Questions

- [ ] Berapa puncak RSS sebenarnya pada profil `full` setelah Task 0? Target < 4 GB. Bila masih
      di atas itu, jalur cadangan `np.memmap` inkremental harus diaktifkan.
- [ ] Berapa lama tahap preprocessing profil `full` di mesin lokal (12 core logis) dibanding
      Kaggle (4 core)? Belum diukur.
- [ ] Berapa lama waktu unggah artifact 1,02 GB ke Kaggle Dataset dari koneksi peneliti? Ini
      biaya sekali yang diamortisasi lintas 5+ sesi, tetapi belum diketahui.
- [ ] Apakah artifact `tiny` dan `smoke` perlu dipublikasikan sebagai Dataset, atau cukup
      dibangun ulang setiap kali karena murah?
- [ ] Bagaimana artifact di-versioning ketika definisi fitur berubah? Hash mendeteksi
      ketidakcocokan, tetapi belum ada aturan penamaan atau kebijakan retensi.
- [ ] Berapa banyak artifact yang boleh hidup bersamaan mengingat batas 20 GB `/kaggle/working`
      dan batas ukuran Dataset?
- [ ] Apakah sesi CPU Kaggle memakai batas 12 jam yang sama, dan apakah itu cukup untuk profil
      `full`? Perlu dikonfirmasi.
- [ ] Setelah pemisahan, apa yang menjadi sumber kebenaran definisi fitur agar dua notebook
      tidak menyimpang (*drift*)?

## Risks

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| Definisi fitur terduplikasi di dua notebook lalu menyimpang | Tinggi | Tinggi — model dilatih pada fitur yang tidak sesuai manifestnya | Satu sumber kebenaran untuk definisi fitur; hash manifest diverifikasi di kedua sisi |
| Artifact basi dipakai setelah data mentah diperbarui | Sedang | Tinggi — hasil salah tanpa gejala | Hash data mentah ikut dibekukan ke dalam artifact; ketidakcocokan menggagalkan sesi |
| Iterasi fitur melambat: setiap perubahan menuntut menjalankan ulang notebook preprocessing | Tinggi | Sedang | Profil `tiny` tetap cepat sebagai loop pengembangan; terima biaya ini untuk profil `full` |
| Batas ukuran/kuota Dataset terlampaui saat beberapa profil dan versi hidup bersamaan | Sedang | Sedang | Kebijakan retensi; hanya artifact `full` yang dipublikasikan |
| Pemisahan memperkenalkan kebocoran data baru (mis. scaler dipakai ulang lintas rentang berbeda) | Rendah | **Kritis** | Parameter scaler dibekukan bersama batas split yang dipakai untuk memfitnya; sanity gate tetap berjalan di sisi training |
| Penghematan kuota GPU jauh lebih kecil dari perkiraan | Sedang | Sedang — pekerjaan tetap bernilai karena reproducibility, tetapi justifikasi utamanya berubah | Ukur durasi preprocessing lebih dulu sebelum menulis kode |
| **Mesin lokal hanya punya 14,8 GB RAM (4,2 GB bebas) — lebih kecil dari Kaggle (~29 GB)** | **Terjadi** | **Tinggi** — tanpa Task 0, preprocessing lokal langsung swap dan tidak selesai | Task 0 jadi prasyarat keras; puncak diukur, bukan diperkirakan; Kaggle CPU session tetap tersedia sebagai fallback |
| Task 0 mengubah bit matriks fitur, sehingga hash tidak lagi cocok dengan run lama | Terjadi (disengaja) | Rendah | Ekuivalensi diukur terhadap notebook tunggal **setelah** Task 0, bukan terhadap versi sebelumnya |

---
*Status: DRAFT — requirements only. Implementation planning pending via /plan.*
