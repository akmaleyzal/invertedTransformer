# Perbaikan audit A01–A15

Pekerjaan dimulai 9 September 2026; pemeriksaan dan dokumentasi diperbarui 10 September 2026. Dokumen ini menindaklanjuti [audit asli](RESEARCH_WORKFLOW_AUDIT_2026-09-09.md), tanpa mengubah bukti audit tersebut.

**Perbaikan implementasi telah dibuat pada notebook utama. Hasil eksperimen baru belum tersedia.** Ini bukan sertifikat “zero vulnerability”: audit membahas validitas riset dan alur eksperimen, dan pengujian lokal tidak membuktikan seluruh perilaku CUDA atau semua kemungkinan cacat. Temuan yang memerlukan observasi baru tetap memiliki syarat penutupan empiris yang dinyatakan di bawah.

## Otoritas dan bukti

- [iTransformer.ipynb](../notebooks/iTransformer.ipynb) adalah implementasi utama, termasuk model, analisis, dan orkestrasi Kaggle. `src/` hanya proyeksi yang diekspor melalui sel terakhir. Tidak ada perbaikan yang ditulis langsung ke `src/`.
- `tools/build_notebook.py` menolak menimpa notebook dari template lama. Mode `--check` memeriksa proyeksi. Digests program dan ekspor mencegah output lama tampil sebagai hasil dependency baru.
- Notebook yang sudah dieksekusi disimpan pada [arsip sebelum perbaikan](../notebooks/outputs/iTransformer_before_A01_A15.ipynb). Notebook aktif memiliki 357 sel dan output dikosongkan untuk rerun.
- Sebanyak 1.620 run lama di `notebooks/outputs/artifacts/`, beserta `paper/paper_numbers.json` dan hasil aslinya, dipertahankan. Prediksi tersebut memakai code SHA-256 `bfb43f21028da322e123402837625815164a32b70f24f6fa50bed22f6679cadb`.
- Input parquet SHA-256: `8270a84b07c2923bc885782a8ba4e1898133d18ee3b260f157fcee3fd6923b4e`.
- [Reanalisis historis](../paper/reanalysis_2026-09-09/paper_numbers.json) menyimpan hash kode analisis dan hash kode prediksi secara terpisah. Ia tidak mengubah model yang menghasilkan prediksi lama.
- Manifest baru memiliki **2.130 run unik**. Perubahan arsitektur, kalender prediksi, objective, dan sampling memerlukan rerun; run lama sengaja tidak lolos gerbang resume baru.

## Disposisi setiap temuan

| ID | Perbaikan yang tersedia | Status bukti |
|---|---|---|
| A01 | Forecast origin = bar target pertama; input/target timestamp eksplisit; evaluasi pada irisan kalender aktual | Kontrak dan counterexample diperiksa; prediksi yang belum pernah disimpan perlu rerun |
| A02 | Kontrol representasi invertibel dengan informasi K8 sama; klaim kausal PR-only ditarik | Desain dan invariansi diuji; hasil tiga arm menunggu Kaggle |
| A03 | J-test CR1 berkelompok origin; fixed effects origin×block; inferensi origin dependen dibatasi | Numerik dibandingkan dengan regresi dummy lengkap; validitas confirmatory tidak diklaim |
| A04 | K ladder tidak otomatis dianggap nested; CW tidak menjadi keputusan headline | Jalur perbandingan dan fixture diperbarui |
| A05 | Mean loss masing-masing seed, step-level MSE, rasio per block, bobot block/origin sama | Regresi agregasi dan laporan diperiksa |
| A06 | Pilot validation-only; MDE diberi label post-analysis TEST; preregistration tanpa bukti tidak diklaim | Perilaku pilot/cache diuji; status eksploratori dinyatakan |
| A07 | Decay terhadap skill block 1; guard baseline positif; tanpa cadence optimal/log-rank/CI yang tidak sah | Counterexample skill meningkat lolos |
| A08 | Dua definisi age; fresh lima seed; kontrol validation-refresh dengan training tetap | Span dan pairing diperiksa; hasil intervensi menunggu Kaggle |
| A09 | Final encoder norm iTransformer; PatchTST diperbaiki; target-only dan all-channel dipisah; budget baseline diperpanjang | Delapan forward-parity check lolos; ranking baru menunggu Kaggle |
| A10 | DA dan sinyal ekonomi memakai inverse scaling sebelum sign/sum | Counterexample drift diperiksa; reanalisis tersedia |
| A11 | Long/cash daily round trips, kedua sisi biaya, Sortino dan MDD benar; scope conditional | Numerik diperiksa; bukan bukti strategi spot yang dapat dieksekusi |
| A12 | Setiap run memakai 11.500 training windows; metadata sampling; populasi surviving windows | Semua 72 span cukup; hasil training terkontrol menunggu Kaggle |
| A13 | Satu gerbang strict completion; atomic writes; epoch checkpoint; deadline dan bundle resume | Resume CPU ekuivalen, stale-vintage, dua worker dan konsolidasi diuji; T4 belum dijalankan |
| A14 | Alasan Polars, uniform attention, ARIMA dan jumlah parameter diperjelas | Dokumentasi/implementasi dikoreksi; effective capacity tidak diklaim setara |
| A15 | Pintu masuk, naskah, source log, pemisahan historical/new, dan hukum proyek diperbarui | Naskah tetap draft riset; hasil eksperimen baru dan keputusan venue bukan hasil yang telah diperoleh |

### A01 — waktu penerbitan dan sampel pembanding

Untuk lookback L dan horizon H, `timestamp == forecast_origin` adalah waktu pembukaan target pertama. `input_start = forecast_origin − L jam`; `target_timestamp = forecast_origin + (step−1) jam`. Input terakhir sudah selesai pada waktu forecast diterbitkan. Train/validation tetap contained: target terakhir tidak boleh menyentuh split berikutnya. Test block dipilih menurut waktu penerbitan, bukan awal input.

Reader legacy menggeser timestamp sesuai L dan mengalokasikan ulang block **di memori**. Evaluator mengambil irisan waktu yang benar-benar tersedia per origin/H/block untuk seluruh run yang dibandingkan, memeriksa horizon lengkap dan kesamaan actual return setelah inverse scaling. Dengan demikian L48/L96/L192 tidak dinilai pada target berbeda. Irisan ini mengurangi sampel dan tetap conditional pada ketersediaan seluruh target. Forecast awal yang tidak pernah disimpan tidak dapat direkonstruksi oleh relabeling.

### A02 — identifikasi dimensionalitas

Subset fitur orthogonal/redundant yang lama tetap dilaporkan sebagai perubahan identitas fitur sekaligus PR. Tiga arm baru (`repi`, `repw`, `repc`) menggunakan informasi K8 yang sama: identity, whitening, dan transformasi korelasi invertibel. Ketiganya menjaga target channel 0 dan menonaktifkan instance normalization. Transformasi dan PR hanya dipasang pada training; matrix, inverse, condition number, eigenvalue floor dan PR disimpan.

Uji memeriksa inversi, target yang tidak berubah, dan ketidakpekaan transformasi terhadap perubahan data masa depan. Intervensi ini memisahkan informasi dari koordinat representasinya. Ia masih mengubah conditioning/optimisasi dan geometri fungsi model: hasilnya **bukan efek kausal PR saja**. PR tidak diperlakukan sebagai ukuran informasi prediktif.

### A03–A06 — estimand, inferensi dan waktu keputusan

Loss headline adalah rerata squared error per step untuk setiap seed, kemudian rerata seed, RelMSE terhadap forecast raw return nol pada setiap block, lalu bobot block dan origin yang sama. MSE dari prediksi ensemble merupakan objek lain dan tidak dipakai diam-diam. Kalender evaluasi ikut di-hash untuk paired contrasts. Satu `research_summary` dipakai oleh notebook dan laporan agar definisi RQ tidak bercabang.

J-test memakai covariance CR1 dengan cluster origin dan fixed effects origin×block; statistik dibandingkan terhadap implementasi regresi dummy lengkap. K berbeda pada nonlinear learned model tidak cukup untuk mengasumsikan struktur nesting Clark–West. Pair matrix menggunakan loss differential tanpa koreksi nesting otomatis.

Origin memiliki training/test calendar yang bertumpang tindih. Clustered SE, independent-origin bootstrap, TOST, PT, RW dan MCS yang masih dihitung adalah **diagnostik eksploratori**, bukan dasar penolakan confirmatory. Koreksi multipel tidak memperbaiki dependensi yang tidak dimodelkan. Sensitivitas stride-5 memakai lima triplet G=3, bukan mengarang effective G atau menganggap jumlah kecil tersebut memberi presisi kuat.

Pilot hanya memakai validation dan mean best target-validation MSE antar-seed. Cache fit validation tidak memanggil test inference. MDE dihitung dari slope TEST yang sudah diamati; labelnya post-analysis, bukan prospective power. Margin equivalence berbasis hasil juga eksploratori. Tidak ada bukti registrasi bertanggal eksternal yang diberikan untuk mendukung judul “pre-registered”; perubahan pasca-audit dinyatakan sebagai protokol eksploratori baru.

### A07–A08 — decay dan refresh

`D(i,b) = [R²(i,1) − R²(i,b)] / R²(i,1)` hanya didefinisikan ketika reference skill block 1 positif. Skill yang terus naik tidak boleh menghasilkan event decay. Crossing threshold 5% (sensitivitas 2,5/10/50%) adalah deskripsi diskret enam block; ia bukan kebijakan retraining optimal. Interval independent-subject dan log-rank untuk arm berpasangan ditahan. Reference nonpositif berarti undefined, berbeda dari reference positif yang tidak crossing hingga akhir observasi.

Metadata membedakan selection/deployment time, cutoff training yang direncanakan, dan target training aktual terakhir. Fresh memperbarui training serta validation pada +90 hari dan memakai lima seed. Validation-refresh mempertahankan training lama tetapi memperbarui validation/selection ke kalender yang sama dengan fresh. Semua dibandingkan pada target asli B4–B6. Selisih ini membandingkan prosedur seleksi/training; ia tidak mengidentifikasi penyebab perubahan pasar. Training baru dengan validation lama yang tumpang tindih tidak ditambahkan karena akan membocorkan informasi.

### A09 — model dan kesempatan optimisasi

iTransformer kini menyertakan LayerNorm akhir encoder dan mean normalization yang detached. Target loss tetap return channel. Uniform attention melewati dropout bobot attention yang sama; Q/K yang tidak digunakan dibekukan. Jumlah parameter teralokasi sama tidak berarti jumlah trainable parameter, apalagi effective capacity, sama.

PatchTST menggunakan BatchNorm, residual attention logits, GELU, urutan flatten yang benar, inisialisasi posisi yang sesuai dan kedua dropout projection/residual. Adaptasi `affine=False`, attention dropout 0, tanpa padding akhir dan head dropout 0 disebutkan. Target-only DLinear/PatchTST menerima channel target saja pada jalur model, termasuk BatchNorm; effective input K=1 meski tensor yang disuplai K8. Sensitivitas all-channel memiliki ID tersendiri.

Baseline neural terkait memiliki cap 120 epoch, patience 12, LR dibagi dua setiap 20 epoch, dengan pencarian LR 1e-4/1e-3/1e-2 pada validation origin pertama secara terpisah untuk objective target/all. Kandidat dan cap tercatat. Early stopping ataupun cap yang lebih besar bukan bukti konvergensi. Semua ranking baru wajib dibaca bersama achieved epochs dan cap; ranking historis hanya berlaku bagi port/objective/budget lama.

[Pemeriksaan parity](../.research/audit-repair-work/check_upstream_parity.py) mencocokkan bobot terhadap definisi upstream yang dipin: iTransformer `c2426e68ca13f74aaec08045c5c724d8ad328124`, PatchTST `204c21efe0b39603ad6e2ca640ef5896646ab1a9`. Empat kasus iTransformer eval memiliki max absolute error paling besar 7,16e-7; empat kasus PatchTST target/all × eval/train cocok tepat pada probe tersebut. Toleransi 2e-5. Ini forward parity CPU sintetis, bukan kesamaan training atau verifikasi CUDA.

### A10–A11 — arah, biaya dan risiko

Return mentah direkonstruksi sebagai `r = zσ + μ`; cumulative forecast H langkah adalah `σ sum(z) + Hμ`. Tanda diambil sesudah transformasi tersebut. PT tetap diagnostik karena non-overlap sendiri tidak membuktikan independence.

Simulator memakai posisi long/cash, issuance tengah malam UTC dan 24 target berurutan. Setiap posisi long adalah daily round trip sehingga biaya masuk dan keluar selalu dibebankan, termasuk akhir sampel dan observasi di dekat gap. Log wealth menggunakan `p × [r + log(1−c) − log(1+c)]`, dengan `c = fee + slippage` per sisi; cash menghasilkan nol. Pembanding adalah always-long daily trades, bukan buy-and-hold kontinu.

Sharpe/Sortino menggunakan simple returns dari log wealth. Downside deviation memakai RMS bagian negatif terhadap MAR=0, dengan seluruh periode dalam denominator. MDD memasukkan modal awal. JK/DSR dan confidence interval MDD yang tidak didukung tidak ditampilkan sebagai inferensi sah. Outcome window yang hilang tetap tidak diketahui; mengevaluasi hanya window lengkap adalah seleksi berdasarkan ketersediaan masa depan. Tidak ada klaim executable backtest atau profitabilitas seluruh kalender.

### A12 — ukuran training dan missingness

Runner memilih 11.500 window tanpa replacement memakai seed 1729 dan indeks waktu training saja. Scaler dipasang sebelum subsampling pada bar training yang dipurge. Jumlah available/selected, seed dan digest timestamp dicatat. Pemilihan identik ketika hanya K berubah dan semua waktu validnya sama; perubahan L/H dapat mengubah daftar kandidat dan window terpilih.

[Pemeriksaan seluruh span](../.research/audit-repair-work/window_budget.json) menemukan 72 konfigurasi training unik dalam 2.130 run, dengan available **11.689–15.265**. Seluruhnya memenuhi 11.500. Pemeriksaan ini tidak melakukan fitting. Durasi kalender sama tidak menjamin jumlah observasi sama; kontrol baru diperlukan karena run lama belum memakai jumlah tetap. Keterangan gap diubah menjadi missing-bar gaps yang penyebabnya belum diverifikasi. Coverage tidak mengembalikan outcome hilang dan tidak membuktikan ignorability.

### A13 — continuation tanpa menerima hasil basi

`pending`, executor serial/parallel dan hitungan remaining memakai gerbang yang sama: code/input aktual, requested config, resolved schedule, kolom, schema serta hash prediksi/weights. Metadata completion ditulis terakhir sesudah file atomik. Checkpoint memuat model/best model, optimizer, scheduler, RNG, epoch, patience dan identitas fit; load menggunakan `weights_only=True`. Hash membuktikan kecocokan byte, bukan identitas pembuat: gunakan output milik studi ini.

Deadline monotonic tunggal mencakup pilot, tuning dan grid. Training memeriksa deadline pada batas minibatch, menyimpan setiap epoch selesai dan mengulang hanya partial epoch dari boundary terakhir. Uji CPU menunjukkan interrupted/resumed training cocok tepat dengan uninterrupted training. Dua worker mempertahankan status pending saat pause dan meneruskan alignment failure ke caller. Konsolidasi membawa hasil lengkap yang diterima serta cache/checkpoint ke satu bundle baru tanpa menimpa checkpoint lokal yang lebih baru.

Untuk batas pengguna 12 jam/session dan 30 jam/minggu: default ceiling 11,5 jam dengan reserve 45 menit; budget efektif dapat lebih pendek menurut quota aktual dan waktu session yang sudah terpakai. Pengguna harus memasukkan meter quota tersisa, menyimpan output dan menghentikan session. Python tidak dapat menyelamatkan output interaktif yang tidak disimpan atau melepaskan alokasi GPU secara otomatis. Petunjuk lengkap berada di [USAGE](../USAGE.md).

### A14–A15 — alasan dan klaim yang dapat dipertanggungjawabkan

Polars mendukung `center=True`; proteksi datang dari fitur per-bar dan pengujian kronologi. Uniform attention tidak mempunyai effective capacity yang terjamin sama. ARIMA tetap di luar scope; ADF/variance-ratio tidak membuktikan bahwa AIC pasti memilih (0,0,0).

README/USAGE menunjuk notebook dan manifest baru. Naskah membahas hasil historis yang dihitung ulang, keterbatasan, dan protokol eksperimen baru secara terpisah; angka prosa dihasilkan dari JSON laporan. Source specifications lama tidak boleh menghidupkan kembali judul, cadence, atau prioritas ilmiah yang ditarik. [Log sumber dan pencarian terbatas](LITERATURE_SCOPE_2026-09-10.md) menyebut ruang lingkup bacaan dan tidak menyamar sebagai systematic review. Sudah ada karya primer tentang iTransformer, hourly BTC dan walk-forward; klaim “pertama” tidak dipertahankan.

## Pemeriksaan dan langkah empiris berikutnya

Evidence mesin: [hasil pytest per file](../.research/audit-repair-work/pytest_results.json), [parity](../.research/audit-repair-work/upstream_parity.json), [window budget](../.research/audit-repair-work/window_budget.json). Test dijalankan dalam proses terpisah agar RAM lokal tidak terus bertambah; TEMP/TMP berada di D:. Tidak dilakukan full local training.

Selesainya implementasi tidak menghasilkan observasi GPU baru. Pada Kaggle, jalankan notebook aktif dengan T4 x2, simpan setiap output session, dan lanjutkan hingga semua 2.130 ID lolos validasi. Bila quota habis, lanjutkan setelah quota tersedia; tidak ada bypass atau jaminan seluruh grid selesai dalam satu minggu. Setelah grid lengkap, baca `experimental_controls`, `training_sample`, `representation_diagnostics`, `optimization_status` dan seed counts. Laporkan semua arm, termasuk jika kontrol memperburuk hasil. Analisis bisa diselesaikan dalam session CPU Kaggle.

Klaim yang tetap tidak tersedia sesudah sekadar rerun: efek kausal PR-only, bukti tidak adanya predictability di seluruh pasar Bitcoin, cadence optimal, validitas inferensi dari origin independen, penyebab semua gap, serta profitabilitas strategi yang dapat dieksekusi pada seluruh kalender. Klaim tersebut memerlukan desain/bukti lain, bukan perubahan label atau lebih banyak seed saja.

## Sumber metodologis yang diperiksa

- [Kode resmi iTransformer](https://github.com/thuml/iTransformer) dan [PatchTST](https://github.com/yuqinie98/PatchTST): definisi implementasi yang dipin pada evidence parity.
- [Polars rolling_mean](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.rolling_mean.html): API memiliki opsi centered window.
- [Clark–West working paper](https://www.kansascityfed.org/documents/5368/pdf-RWP05-05.pdf): konteks nested forecast models, bukan lisensi menerapkan nesting otomatis pada seluruh K ladder.
- [MacKinnon, Nielsen dan Webb](https://arxiv.org/abs/2205.03285): clustered inference dan kehati-hatian terhadap struktur dependensi.
- [Rollinger dan Hoffman, Sortino](https://www.cmegroup.com/education/files/rr-sortino-a-sharper-ratio.pdf): downside deviation terhadap target pada seluruh observasi.
- [Kaggle GPU guidance](https://www.kaggle.com/docs/efficient-gpu-usage): operasi accelerator; angka 12/30 jam di konfigurasi adalah batas yang diberikan pengguna dan meter akun tetap harus diperiksa.
