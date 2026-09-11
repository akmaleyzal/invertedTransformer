# Peta notebook — `notebooks/iTransformer.ipynb`

**Digenerate oleh `tools/notebook_map.py`. Jangan disunting tangan** — jalankan ulang generatornya setelah `tools/build_notebook.py`.

`notebooks/iTransformer.ipynb` adalah **deliverable utama** repositori ini (`CLAUDE.md` §15): notebook mandiri yang membawa seluruh paket `src/itransformer_btc/` sebagai sel definisi, ditambah sel orkestrasi yang tidak ada di modul mana pun. `src/` adalah proyeksinya yang diuji, bukan sebaliknya. Sel definisi adalah irisan byte-exact dari modulnya (`D63`, dijaga `tests/test_notebook_sync.py`), sehingga peta ini menyebut modul yang diproyeksikan alih-alih mengulang kodenya.

- Sel: **357** (163 kode, 194 markdown)
- Fase: **12**
- Langkah produksi (`role: step`): **18**
- Modul yang diproyeksikan: **18** dalam 144 sel definisi

## Peta artefak — sel mana menghasilkan apa

`D87` mewajibkan tiap sel produksi mendeklarasikan yang dibaca dan ditulisnya. Tanpa tabel ini, *sel mana yang menulis figure* hanya bisa dijawab dengan membaca seluruh sel berurutan — dan yang tidak bisa ditemukan tidak bisa diverifikasi.

| Langkah | Sel | Fase | Membaca | Menulis |
| --- | --: | --- | --- | --- |
| `artifact_map` | 3 | 🔧 01 · Persiapan lingkungan dan konfigurasi | — | — |
| `setup` | 5 | 🔧 01 · Persiapan lingkungan dan konfigurasi | `data/raw/BTCUSDT_1h.parquet` | — |
| `provenance_sources` | 27 | 🔧 01 · Persiapan lingkungan dan konfigurasi | — | — |
| `data` | 57 | 📥 02 · Muat data dan audit kualitas | `data/raw/BTCUSDT_1h.parquet` | — |
| `features` | 67 | 🧪 03 · Feature engineering dan eksplorasi | — | — |
| `keff` | 111 | 🪟 04 · Split walk-forward, scaling, dan K_eff | — | `artifacts/keff_table.parquet` |
| `module_names` | 328 | ⚙️ 06 · Persiapan evaluasi dan eksekutor | — | — |
| `code_digest` | 330 | ⚙️ 06 · Persiapan evaluasi dan eksekutor | — | — |
| `invariants` | 333 | 🛠️ 07 · Pemeriksaan sebelum training | — | — |
| `pilot` | 336 | 🛡️ 08 · Validasi dan pemilihan konfigurasi | — | `artifacts/validation/*.json`, `artifacts/checkpoints/*.pt` |
| `tune` | 338 | 🛡️ 08 · Validasi dan pemilihan konfigurasi | — | `artifacts/meta/tuning_selection.json` |
| `grid` | 341 | 🚀 09 · Training grid walk-forward | `data/raw/BTCUSDT_1h.parquet` | `artifacts/preds/*.parquet`, `artifacts/meta/*.json`, `artifacts/attn/*.parquet`, `artifacts/checkpoints/*.pt`, `artifacts/weights/*.pt`, `artifacts/session_status.json` |
| `rq1` | 344 | 📈 10 · Evaluasi model dan research questions | `artifacts/preds/*.parquet`, `artifacts/meta/*.json` | — |
| `rq2` | 346 | 📈 10 · Evaluasi model dan research questions | `artifacts/preds/*.parquet`, `artifacts/meta/*.json` | — |
| `rq3` | 348 | 📈 10 · Evaluasi model dan research questions | `artifacts/preds/*.parquet`, `artifacts/meta/*.json` | — |
| `save` | 351 | 💾 11 · Simpan hasil, tabel, dan figure | `artifacts/preds/*.parquet`, `artifacts/meta/*.json` | `artifacts/paper_numbers.json`, `artifacts/run_block_metrics.parquet`, `artifacts/seed_averaged_cells.parquet`, `artifacts/amplification_panel.parquet`, `artifacts/decay_panel.parquet` |
| `report` | 353 | 💾 11 · Simpan hasil, tabel, dan figure | `artifacts/paper_numbers.json`, `artifacts/preds/*.parquet`, `artifacts/attn/*.parquet` | `paper/paper_numbers.json`, `paper/tables/*.tex`, `paper/figures/*.pdf`, `paper/figures/*.png`, `paper/panels/*.parquet` |
| `sync_back` | 356 | 🔁 12 · Lampiran — sinkronisasi lokal | — | — |

## Modul yang dibawa notebook

Tiap baris adalah satu modul `src/itransformer_btc/`, dipotong menjadi sel per kelompok logis (`SECTION_MAP` di `tools/build_notebook.py`).

| Modul | Sel | Fase | Bagian |
| --- | --: | --- | --- |
| `src/itransformer_btc/__init__.py` | 1 | 🔧 01 · Persiapan lingkungan dan konfigurasi | Permukaan paket |
| `src/itransformer_btc/attention.py` | 4 | ⚙️ 06 · Persiapan evaluasi dan eksekutor | Header & konstanta · Volatilitas lookback & tercile · Penangkap batch · Peta attention per tercile |
| `src/itransformer_btc/baselines.py` | 11 | 🧠 05 · Model, baseline, dan fungsi training | Header & protokol baseline · Ridge — konfigurasi · Ridge — forecaster · DLinear — konfigurasi · DLinear — dekomposisi & model · PatchTST — konfigurasi · PatchTST — model · LSTM — konfigurasi · LSTM — forecaster · Dua komparator naif · Penyelarasan jendela baseline |
| `src/itransformer_btc/budget.py` | 4 | 📥 02 · Muat data dan audit kualitas | Header & anggaran terkomit · Anggaran per origin · Start blok uji yang bertahan · Tabel anggaran |
| `src/itransformer_btc/comparisons.py` | 10 | ⚙️ 06 · Persiapan evaluasi dan eksekutor | Header & konstanta · Nesting & label model · Panel prediksi · Ketersediaan & membangun panel · Diferensial rugi · Bootstrap klaster & Romano–Wolf · Model Confidence Set · Diagnostik per sel · Matriks pasangan · Tabel MCS |
| `src/itransformer_btc/config.py` | 7 | 🔧 01 · Persiapan lingkungan dan konfigurasi | Header · Kontrak data & jendela · Protokol walk-forward · Origin · Origin falsifikasi · Grid lima belas origin · Provenance source code |
| `src/itransformer_btc/economics.py` | 10 | ⚙️ 06 · Persiapan evaluasi dan eksekutor | Header, biaya, band slippage · Posisi & return bersih · Max drawdown & bootstrap-nya · Ringkasan strategi · Menjalankan strategi & buy-and-hold · Uji Jobson–Korkie–Memmel · Deflated Sharpe Ratio · Run id per origin · Tabel ekonomi · Kurva ekuitas |
| `src/itransformer_btc/efficiency.py` | 6 | 🧪 03 · Feature engineering dan eksplorasi | Header & konstanta · Baris hasil · Eksponen Hurst (R/S) · Variance ratio Lo–MacKinlay · ADF · Tabel efisiensi |
| `src/itransformer_btc/features.py` | 3 | 🧪 03 · Feature engineering dan eksplorasi | Header, tangga variat, konstanta · Kolom per rung · Membangun dua belas variat |
| `src/itransformer_btc/keff.py` | 8 | 🪟 04 · Split walk-forward, scaling, dan K_eff | Header & konstanta · Participation ratio · Stable rank & spektrum sadar-lookback · Baris K_eff per origin · Tabel K_eff & korelasi · Gerbang Stage 3b · PR bergulir — deskriptif saja · OLS R² bergulir |
| `src/itransformer_btc/metrics.py` | 21 | 🧠 05 · Model, baseline, dan fungsi training | Header & konstanta · Memuat run · Metrik inti · Jendela non-overlap & Pesaran–Timmermann · Akurasi arah · Metrik per blok · Perakitan grid & rata-rata seed · Amplifikasi A dan A_attn · Decay dan b* · Kurva survival & kuantil normal · Kaplan–Meier & log-rank · Varians jangka panjang · Koreksi Harvey–Leybourne–Newbold · DM & Clark–West · Rugi per origin · Bobot bootstrap & matriks panel · β₁ dengan bootstrap klaster liar · TOST & uji-J non-nested · Efek minimum terdeteksi · Tabel akurasi arah & skala mentah · RelMSE falsifikasi & β₁ dengan cakupan |
| `src/itransformer_btc/model.py` | 6 | 🧠 05 · Model, baseline, dan fungsi training | Header · Konfigurasi arsitektur · Jadwal panjang · Konfigurasi ter-tuning · Blok encoder · ITransformer |
| `src/itransformer_btc/report.py` | 18 | ⚙️ 06 · Persiapan evaluasi dan eksekutor | Header & konstanta · Format angka & LaTeX · Helper kecil · Input laporan & provenance · Bagian dataset & arsitektur · Bagian horizon & robustness · Memuat peta attention · build_report · paper_numbers.json · Tabel 1 & 2 · Tabel 2b & 3 · Tabel 4 & 5 · Tabel 6 · Tabel 7, 8, 9, dan render · Helper plot · Figure 2b, 3, 4 · Figure 5 & 6 · Figure 7 & render |
| `src/itransformer_btc/runner.py` | 12 | ⚙️ 06 · Persiapan evaluasi dan eksekutor | Header, arm, konstanta sesi · Sel run · Manifes eksperimen · Penemuan & resume · Penjaga anggaran sesi · Cache tensor per origin · Ringkasan eksekusi & penyelarasan · Eksekutor grid · Dua GPU, tingkat-run · Grid tuning & pemilih konfigurasi · Pilot Stage 5 · Frame fitur |
| `src/itransformer_btc/segments.py` | 5 | 📥 02 · Muat data dan audit kualitas | Header & konstanta · Segmen · Memuat bar & masker layak-pakai · Membangun segmen · Ringkasan break |
| `src/itransformer_btc/splits.py` | 5 | 🪟 04 · Split walk-forward, scaling, dan K_eff | Header & semantik · Start jendela per semantik · Scaler · Tensor split · Merakit tensor origin |
| `src/itransformer_btc/train.py` | 10 | 🧠 05 · Model, baseline, dan fungsi training | Header & konstanta · Seed & pemilihan perangkat · Spesifikasi run & hasil latih · Arsitektur & protokol Forecaster · Helper batch & prediksi · Loop latih · Uji invarian skala · Provenance kode & input · Menulis artefak run · Idempotensi |
| `src/itransformer_btc/windows.py` | 3 | 📥 02 · Muat data dan audit kualitas | Header · Enumerasi jendela · Menghitung jendela |

## Fase

### 🔧 01 · Persiapan lingkungan dan konfigurasi

Siapkan dependensi, temukan dataset, lalu muat konfigurasi penelitian dan sumber algoritma.

- **Langkah `artifact_map`** (sel 3) — 🗺️ Peta artefak — sel mana memproduksi apa
- **Langkah `setup`** (sel 5) — 🧰 Siapkan sesi · membaca `data/raw/BTCUSDT_1h.parquet`
- `library` (sel 7) — 📚 Library — impor bersama
- Modul `src/itransformer_btc/config.py` § Header (sel 10)
- Modul `src/itransformer_btc/config.py` § Kontrak data & jendela (sel 12)
- Modul `src/itransformer_btc/config.py` § Protokol walk-forward (sel 14)
- Modul `src/itransformer_btc/config.py` § Origin (sel 16)
- Modul `src/itransformer_btc/config.py` § Origin falsifikasi (sel 18)
- Modul `src/itransformer_btc/config.py` § Grid lima belas origin (sel 20)
- Modul `src/itransformer_btc/config.py` § Provenance source code (sel 22)
- Modul `src/itransformer_btc/__init__.py` § Permukaan paket (sel 25)
- **Langkah `provenance_sources`** (sel 27) — 📚 Provenance source code — dari mana tiap algoritma berasal

### 📥 02 · Muat data dan audit kualitas

Baca bar BTCUSDT, tandai bar yang tidak layak, dan periksa kontinuitas serta jumlah jendela per origin.

- Modul `src/itransformer_btc/segments.py` § Header & konstanta (sel 31)
- Modul `src/itransformer_btc/segments.py` § Segmen (sel 33)
- Modul `src/itransformer_btc/segments.py` § Memuat bar & masker layak-pakai (sel 35)
- Modul `src/itransformer_btc/segments.py` § Membangun segmen (sel 37)
- Modul `src/itransformer_btc/segments.py` § Ringkasan break (sel 39)
- Modul `src/itransformer_btc/windows.py` § Header (sel 42)
- Modul `src/itransformer_btc/windows.py` § Enumerasi jendela (sel 44)
- Modul `src/itransformer_btc/windows.py` § Menghitung jendela (sel 46)
- Modul `src/itransformer_btc/budget.py` § Header & anggaran terkomit (sel 49)
- Modul `src/itransformer_btc/budget.py` § Anggaran per origin (sel 51)
- Modul `src/itransformer_btc/budget.py` § Start blok uji yang bertahan (sel 53)
- Modul `src/itransformer_btc/budget.py` § Tabel anggaran (sel 55)
- **Langkah `data`** (sel 57) — 📊 Jalankan Stage 2 — muat bar & asersi anggaran jendela · membaca `data/raw/BTCUSDT_1h.parquet`

### 🧪 03 · Feature engineering dan eksplorasi

Bangun dua belas variat F1–F5, tinjau statistiknya, lalu siapkan fungsi diagnostik efisiensi pasar.

- Modul `src/itransformer_btc/features.py` § Header, tangga variat, konstanta (sel 61)
- Modul `src/itransformer_btc/features.py` § Kolom per rung (sel 63)
- Modul `src/itransformer_btc/features.py` § Membangun dua belas variat (sel 65)
- **Langkah `features`** (sel 67) — 🔬 Bangun frame fitur
- Modul `src/itransformer_btc/efficiency.py` § Header & konstanta (sel 70)
- Modul `src/itransformer_btc/efficiency.py` § Baris hasil (sel 72)
- Modul `src/itransformer_btc/efficiency.py` § Eksponen Hurst (R/S) (sel 74)
- Modul `src/itransformer_btc/efficiency.py` § Variance ratio Lo–MacKinlay (sel 76)
- Modul `src/itransformer_btc/efficiency.py` § ADF (sel 78)
- Modul `src/itransformer_btc/efficiency.py` § Tabel efisiensi (sel 80)

### 🪟 04 · Split walk-forward, scaling, dan K_eff

Siapkan pembagian train–validation–test, tensor per origin, dan pengukuran dimensionalitas efektif sebelum training.

- Modul `src/itransformer_btc/splits.py` § Header & semantik (sel 84)
- Modul `src/itransformer_btc/splits.py` § Start jendela per semantik (sel 86)
- Modul `src/itransformer_btc/splits.py` § Scaler (sel 88)
- Modul `src/itransformer_btc/splits.py` § Tensor split (sel 90)
- Modul `src/itransformer_btc/splits.py` § Merakit tensor origin (sel 92)
- Modul `src/itransformer_btc/keff.py` § Header & konstanta (sel 95)
- Modul `src/itransformer_btc/keff.py` § Participation ratio (sel 97)
- Modul `src/itransformer_btc/keff.py` § Stable rank & spektrum sadar-lookback (sel 99)
- Modul `src/itransformer_btc/keff.py` § Baris K_eff per origin (sel 101)
- Modul `src/itransformer_btc/keff.py` § Tabel K_eff & korelasi (sel 103)
- Modul `src/itransformer_btc/keff.py` § Gerbang Stage 3b (sel 105)
- Modul `src/itransformer_btc/keff.py` § PR bergulir — deskriptif saja (sel 107)
- Modul `src/itransformer_btc/keff.py` § OLS R² bergulir (sel 109)
- **Langkah `keff`** (sel 111) — 📐 Jalankan Stage 3b — ukur K_eff · menulis `artifacts/keff_table.parquet`

### 🧠 05 · Model, baseline, dan fungsi training

Definisikan iTransformer, loop training, metrik, serta model baseline pada informasi dan skala yang dinyatakan eksplisit.

- Modul `src/itransformer_btc/model.py` § Header (sel 115)
- Modul `src/itransformer_btc/model.py` § Konfigurasi arsitektur (sel 117)
- Modul `src/itransformer_btc/model.py` § Jadwal panjang (sel 119)
- Modul `src/itransformer_btc/model.py` § Konfigurasi ter-tuning (sel 121)
- Modul `src/itransformer_btc/model.py` § Blok encoder (sel 123)
- Modul `src/itransformer_btc/model.py` § ITransformer (sel 125)
- Modul `src/itransformer_btc/train.py` § Header & konstanta (sel 128)
- Modul `src/itransformer_btc/train.py` § Seed & pemilihan perangkat (sel 130)
- Modul `src/itransformer_btc/train.py` § Spesifikasi run & hasil latih (sel 132)
- Modul `src/itransformer_btc/train.py` § Arsitektur & protokol Forecaster (sel 134)
- Modul `src/itransformer_btc/train.py` § Helper batch & prediksi (sel 136)
- Modul `src/itransformer_btc/train.py` § Loop latih (sel 138)
- Modul `src/itransformer_btc/train.py` § Uji invarian skala (sel 140)
- Modul `src/itransformer_btc/train.py` § Provenance kode & input (sel 142)
- Modul `src/itransformer_btc/train.py` § Menulis artefak run (sel 144)
- Modul `src/itransformer_btc/train.py` § Idempotensi (sel 146)
- Modul `src/itransformer_btc/metrics.py` § Header & konstanta (sel 149)
- Modul `src/itransformer_btc/metrics.py` § Memuat run (sel 151)
- Modul `src/itransformer_btc/metrics.py` § Metrik inti (sel 153)
- Modul `src/itransformer_btc/metrics.py` § Jendela non-overlap & Pesaran–Timmermann (sel 155)
- Modul `src/itransformer_btc/metrics.py` § Akurasi arah (sel 157)
- Modul `src/itransformer_btc/metrics.py` § Metrik per blok (sel 159)
- Modul `src/itransformer_btc/metrics.py` § Perakitan grid & rata-rata seed (sel 161)
- Modul `src/itransformer_btc/metrics.py` § Amplifikasi A dan A_attn (sel 163)
- Modul `src/itransformer_btc/metrics.py` § Decay dan b* (sel 165)
- Modul `src/itransformer_btc/metrics.py` § Kurva survival & kuantil normal (sel 167)
- Modul `src/itransformer_btc/metrics.py` § Kaplan–Meier & log-rank (sel 169)
- Modul `src/itransformer_btc/metrics.py` § Varians jangka panjang (sel 171)
- Modul `src/itransformer_btc/metrics.py` § Koreksi Harvey–Leybourne–Newbold (sel 173)
- Modul `src/itransformer_btc/metrics.py` § DM & Clark–West (sel 175)
- Modul `src/itransformer_btc/metrics.py` § Rugi per origin (sel 177)
- Modul `src/itransformer_btc/metrics.py` § Bobot bootstrap & matriks panel (sel 179)
- Modul `src/itransformer_btc/metrics.py` § β₁ dengan bootstrap klaster liar (sel 181)
- Modul `src/itransformer_btc/metrics.py` § TOST & uji-J non-nested (sel 183)
- Modul `src/itransformer_btc/metrics.py` § Efek minimum terdeteksi (sel 185)
- Modul `src/itransformer_btc/metrics.py` § Tabel akurasi arah & skala mentah (sel 187)
- Modul `src/itransformer_btc/metrics.py` § RelMSE falsifikasi & β₁ dengan cakupan (sel 189)
- Modul `src/itransformer_btc/baselines.py` § Header & protokol baseline (sel 192)
- Modul `src/itransformer_btc/baselines.py` § Ridge — konfigurasi (sel 194)
- Modul `src/itransformer_btc/baselines.py` § Ridge — forecaster (sel 196)
- Modul `src/itransformer_btc/baselines.py` § DLinear — konfigurasi (sel 198)
- Modul `src/itransformer_btc/baselines.py` § DLinear — dekomposisi & model (sel 200)
- Modul `src/itransformer_btc/baselines.py` § PatchTST — konfigurasi (sel 202)
- Modul `src/itransformer_btc/baselines.py` § PatchTST — model (sel 204)
- Modul `src/itransformer_btc/baselines.py` § LSTM — konfigurasi (sel 206)
- Modul `src/itransformer_btc/baselines.py` § LSTM — forecaster (sel 208)
- Modul `src/itransformer_btc/baselines.py` § Dua komparator naif (sel 210)
- Modul `src/itransformer_btc/baselines.py` § Penyelarasan jendela baseline (sel 212)

### ⚙️ 06 · Persiapan evaluasi dan eksekutor

Siapkan fungsi perbandingan, ekonomi, attention, laporan, dan eksekutor; catat provenance sebelum training dimulai.

- Modul `src/itransformer_btc/comparisons.py` § Header & konstanta (sel 216)
- Modul `src/itransformer_btc/comparisons.py` § Nesting & label model (sel 218)
- Modul `src/itransformer_btc/comparisons.py` § Panel prediksi (sel 220)
- Modul `src/itransformer_btc/comparisons.py` § Ketersediaan & membangun panel (sel 222)
- Modul `src/itransformer_btc/comparisons.py` § Diferensial rugi (sel 224)
- Modul `src/itransformer_btc/comparisons.py` § Bootstrap klaster & Romano–Wolf (sel 226)
- Modul `src/itransformer_btc/comparisons.py` § Model Confidence Set (sel 228)
- Modul `src/itransformer_btc/comparisons.py` § Diagnostik per sel (sel 230)
- Modul `src/itransformer_btc/comparisons.py` § Matriks pasangan (sel 232)
- Modul `src/itransformer_btc/comparisons.py` § Tabel MCS (sel 234)
- Modul `src/itransformer_btc/economics.py` § Header, biaya, band slippage (sel 237)
- Modul `src/itransformer_btc/economics.py` § Posisi & return bersih (sel 239)
- Modul `src/itransformer_btc/economics.py` § Max drawdown & bootstrap-nya (sel 241)
- Modul `src/itransformer_btc/economics.py` § Ringkasan strategi (sel 243)
- Modul `src/itransformer_btc/economics.py` § Menjalankan strategi & buy-and-hold (sel 245)
- Modul `src/itransformer_btc/economics.py` § Uji Jobson–Korkie–Memmel (sel 247)
- Modul `src/itransformer_btc/economics.py` § Deflated Sharpe Ratio (sel 249)
- Modul `src/itransformer_btc/economics.py` § Run id per origin (sel 251)
- Modul `src/itransformer_btc/economics.py` § Tabel ekonomi (sel 253)
- Modul `src/itransformer_btc/economics.py` § Kurva ekuitas (sel 255)
- Modul `src/itransformer_btc/attention.py` § Header & konstanta (sel 258)
- Modul `src/itransformer_btc/attention.py` § Volatilitas lookback & tercile (sel 260)
- Modul `src/itransformer_btc/attention.py` § Penangkap batch (sel 262)
- Modul `src/itransformer_btc/attention.py` § Peta attention per tercile (sel 264)
- Modul `src/itransformer_btc/runner.py` § Header, arm, konstanta sesi (sel 267)
- Modul `src/itransformer_btc/runner.py` § Sel run (sel 269)
- Modul `src/itransformer_btc/runner.py` § Manifes eksperimen (sel 271)
- Modul `src/itransformer_btc/runner.py` § Penemuan & resume (sel 273)
- Modul `src/itransformer_btc/runner.py` § Penjaga anggaran sesi (sel 275)
- Modul `src/itransformer_btc/runner.py` § Cache tensor per origin (sel 277)
- Modul `src/itransformer_btc/runner.py` § Ringkasan eksekusi & penyelarasan (sel 279)
- Modul `src/itransformer_btc/runner.py` § Eksekutor grid (sel 281)
- Modul `src/itransformer_btc/runner.py` § Dua GPU, tingkat-run (sel 283)
- Modul `src/itransformer_btc/runner.py` § Grid tuning & pemilih konfigurasi (sel 285)
- Modul `src/itransformer_btc/runner.py` § Pilot Stage 5 (sel 287)
- Modul `src/itransformer_btc/runner.py` § Frame fitur (sel 289)
- Modul `src/itransformer_btc/report.py` § Header & konstanta (sel 292)
- Modul `src/itransformer_btc/report.py` § Format angka & LaTeX (sel 294)
- Modul `src/itransformer_btc/report.py` § Helper kecil (sel 296)
- Modul `src/itransformer_btc/report.py` § Input laporan & provenance (sel 298)
- Modul `src/itransformer_btc/report.py` § Bagian dataset & arsitektur (sel 300)
- Modul `src/itransformer_btc/report.py` § Bagian horizon & robustness (sel 302)
- Modul `src/itransformer_btc/report.py` § Memuat peta attention (sel 304)
- Modul `src/itransformer_btc/report.py` § build_report (sel 306)
- Modul `src/itransformer_btc/report.py` § paper_numbers.json (sel 308)
- Modul `src/itransformer_btc/report.py` § Tabel 1 & 2 (sel 310)
- Modul `src/itransformer_btc/report.py` § Tabel 2b & 3 (sel 312)
- Modul `src/itransformer_btc/report.py` § Tabel 4 & 5 (sel 314)
- Modul `src/itransformer_btc/report.py` § Tabel 6 (sel 316)
- Modul `src/itransformer_btc/report.py` § Tabel 7, 8, 9, dan render (sel 318)
- Modul `src/itransformer_btc/report.py` § Helper plot (sel 320)
- Modul `src/itransformer_btc/report.py` § Figure 2b, 3, 4 (sel 322)
- Modul `src/itransformer_btc/report.py` § Figure 5 & 6 (sel 324)
- Modul `src/itransformer_btc/report.py` § Figure 7 & render (sel 326)
- **Langkah `module_names`** (sel 328) — 🧾 Inventaris modul
- **Langkah `code_digest`** (sel 330) — 🔐 Provenance kode & input

### 🛠️ 07 · Pemeriksaan sebelum training

Periksa invariansi skala, overfit satu batch, dan nilai Naive-RW pada setiap origin.

- **Langkah `invariants`** (sel 333) — 🛠️ Jalankan Stage 4 — tiga invarian pra-terbang

### 🛡️ 08 · Validasi dan pemilihan konfigurasi

Jalankan pilot pada validation origin pertama, lalu pilih konfigurasi untuk arm tuning eksploratori.

- **Langkah `pilot`** (sel 336) — 🛡️ Jalankan gerbang Stage 5 — pada validasi, bukan uji · menulis `artifacts/validation/*.json`, `artifacts/checkpoints/*.pt`
- **Langkah `tune`** (sel 338) — 🎛️ Pemilihan konfigurasi pada validasi · menulis `artifacts/meta/tuning_selection.json`

### 🚀 09 · Training grid walk-forward

Latih seluruh manifes dengan resume otomatis, satu worker per device, dan batas waktu sesi.

- **Langkah `grid`** (sel 341) — 🚀 Jalankan grid — 1.620 run, dua T4 · menulis `artifacts/preds/*.parquet`, `artifacts/meta/*.json`, `artifacts/attn/*.parquet`, `artifacts/checkpoints/*.pt`, `artifacts/weights/*.pt`, `artifacts/session_status.json` · membaca `data/raw/BTCUSDT_1h.parquet`

### 📈 10 · Evaluasi model dan research questions

Baca prediksi tersimpan untuk menjawab RQ1, RQ2, dan RQ3 pada grid yang lengkap.

- **Langkah `rq1`** (sel 344) — 1️⃣ RQ1 — K nominal atau K_eff? · membaca `artifacts/preds/*.parquet`, `artifacts/meta/*.json`
- **Langkah `rq2`** (sel 346) — 2️⃣ RQ2 — apakah gap menyempit seiring umur model? · membaca `artifacts/preds/*.parquet`, `artifacts/meta/*.json`
- **Langkah `rq3`** (sel 348) — 3️⃣ RQ3 — crossing ambang skill (deskriptif) · membaca `artifacts/preds/*.parquet`, `artifacts/meta/*.json`

### 💾 11 · Simpan hasil, tabel, dan figure

Simpan panel metrik dan paper_numbers.json, lalu render tabel, figure, dan analisis pendukung manuskrip.

- **Langkah `save`** (sel 351) — 🧊 Simpan panel metrik dan paper_numbers.json · menulis `artifacts/paper_numbers.json`, `artifacts/run_block_metrics.parquet`, `artifacts/seed_averaged_cells.parquet`, `artifacts/amplification_panel.parquet`, `artifacts/decay_panel.parquet` · membaca `artifacts/preds/*.parquet`, `artifacts/meta/*.json`
- **Langkah `report`** (sel 353) — 🖼️ Render deliverable — sepuluh tabel dan delapan figure · menulis `paper/paper_numbers.json`, `paper/tables/*.tex`, `paper/figures/*.pdf`, `paper/figures/*.png`, `paper/panels/*.parquet` · membaca `artifacts/paper_numbers.json`, `artifacts/preds/*.parquet`, `artifacts/attn/*.parquet`

### 🔁 12 · Lampiran — sinkronisasi lokal

Sinkronkan perubahan definisi dari notebook yang sudah disimpan ke src/ pada checkout lokal.

- **Langkah `sync_back`** (sel 356) — 🔁 Sinkron balik ke src/ — nonaktif, aktifkan sendiri
