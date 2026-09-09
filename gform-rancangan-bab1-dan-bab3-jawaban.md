# Jawaban Google Form — Rancangan Bab 1 dan Bab 3

Disusun 2026-09-06. Siap salin-tempel per nomor.
Angka dataset diambil dari `data/raw/BTCUSDT_1h_report.json` (lihat `CLAUDE.md` §4.1),
referensi dari `paper/references/references.bib`. Tidak ada hasil Bab 4 di sini —
form ini hanya meminta rancangan Bab 1 dan Bab 3.

---

## 1. Proyeksi Judul

Evaluasi Walk-Forward Praregistrasi iTransformer terhadap Baseline Linier untuk Peramalan Return Bitcoin Per Jam

---

## 2. Latar Belakang Kasus

Bitcoin diperdagangkan dua puluh empat jam tanpa jeda dan datanya tersedia bebas pada resolusi menit, sehingga menjadi objek favorit literatur peramalan berbasis pembelajaran mendalam. Namun sebagian besar publikasi di bidang ini melaporkan akurasi pada level harga menggunakan MAPE atau RMSE, membandingkan model hanya terhadap model pembelajaran mesin lain, dan membagi data secara acak atau dengan K-fold. Ketiga praktik tersebut membuat angka yang dilaporkan tampak meyakinkan tanpa pernah diuji terhadap pertanyaan yang sesungguhnya, yaitu apakah model itu mengalahkan random walk pada data yang belum pernah dilihatnya.

Penelitian terdahulu yang paling dekat dengan studi ini telah memperbaiki sebagian masalah tersebut, tetapi menyisakan lubang yang jelas. Bysik dan Ślepaczuk (2026) mengevaluasi BTC/USDT per jam dari Binance dengan dua puluh tujuh lipatan walk-forward dan menyertakan iTransformer sebagai salah satu model pembanding, namun sasaran mereka adalah performa perdagangan setelah biaya transaksi, bukan skill statistik: mereka tidak melaporkan koefisien determinasi out-of-sample terhadap random walk, tidak mengukur peluruhan performa seiring bertambahnya umur model, dan tidak mengunci ambang keputusannya sebelum hasil dilihat. Han, Ye, dan Zhan (2024) menyediakan pernyataan teoretis mengenai pertukaran antara kapasitas dan ketangguhan pada strategi channel-dependent, namun mengujinya pada benchmark peramalan jangka panjang yang jauh lebih stasioner daripada aset kripto. Liu dkk. (2024) memperkenalkan iTransformer, tetapi suite evaluasinya mencampur data berdimensi kecil dengan data berdimensi ratusan tanpa membedakan kedua rezim tersebut.

Tiga hal belum terselesaikan dan saling terkait. Pertama, belum jelas apakah manfaat penambahan variat ditentukan oleh jumlah nominalnya atau oleh dimensionalitas efektifnya, padahal keduanya bergerak bersama pada tangga variat yang lazim dipakai. Kedua, belum ada pengukuran eksplisit seberapa cepat keunggulan sebuah model meluruh setelah pelatihan, padahal komposisi pelaku pasar kripto berubah drastis sepanjang 2018 hingga 2026. Ketiga, tanpa ambang yang dikunci sebelum hasil dilihat, sebuah temuan nol tidak dapat dibedakan dari desain yang memang tidak pernah berdaya mendeteksi efek apa pun.

---

## 3. Pertanyaan Penelitian

Apakah arsitektur peramalan deret waktu multivariat berbasis attention benar-benar menghasilkan keunggulan prediktif out-of-sample atas baseline naif pada return Bitcoin per jam — dan jika keunggulan itu ada, apakah ia ditentukan oleh jumlah variat nominal atau oleh dimensionalitas efektifnya, serta berapa lama ia bertahan setelah model dilatih?

---

## 4. Rumusan Masalah

1. Apakah iTransformer menghasilkan *forecast error* yang lebih kecil daripada
   *baseline* Naive-RW, ridge, DLinear, PatchTST, dan LSTM ketika seluruhnya
   dinilai *out-of-sample* pada himpunan *window* dan skala metrik yang persis
   sama?
2. Apakah manfaat penambahan variat mikrostruktur pada iTransformer diatur oleh
   *nominal variate count* (K) atau oleh *effective dimensionality* (K_eff) yang
   diukur melalui *participation ratio*?
3. Apakah *gap* akurasi antara model delapan variat dan model satu variat
   menyempit seiring bertambahnya *time-since-training*?
4. Berapa *retraining cadence* yang optimal di bawah *degradation threshold* yang
   telah ditetapkan di muka, dan apakah *cadence* tersebut bergantung pada jumlah
   variat?

---

## 5. Tujuan Penelitian

1. Membandingkan iTransformer dengan seluruh *baseline* pada metrik RelMSE dan
   R²_oos terhadap Naive-RW, diuji dengan Clark–West untuk pasangan *nested* dan
   Diebold–Mariano berkoreksi Harvey–Leybourne–Newbold untuk pasangan *non-nested*,
   dengan koreksi multiplisitas Romano–Wolf dan keanggotaan Model Confidence Set,
   seluruhnya dinilai pada himpunan *window* yang identik.
2. Mengukur *participation ratio* pada setiap anak tangga variat menggunakan hanya
   sub-blok pelatihan tiap *origin*, lalu membandingkan penjelasan berbasis K
   dengan penjelasan berbasis K_eff sebagai dua model *non-nested*.
3. Mengestimasi *decay slope* dari *gap* delapan-variat terhadap satu-variat
   melalui regresi panel dengan *origin fixed effects* dan galat yang di-*cluster*
   per *origin*, serta mempublikasikan *minimum detectable effect* sebelum blok
   uji dibuka.
4. Menentukan blok pertama yang melewati *degradation threshold* beserta selang
   kepercayaannya, dan menyatakan secara eksplisit apabila *estimand* tersebut
   tidak terdefinisi karena syarat penyebutnya tidak terpenuhi.

---

## 6. Referensi Penelitian Terdahulu (1) — artikel jurnal

https://doi.org/10.1109/TKDE.2024.3400008

Han, L., Ye, H.-J., & Zhan, D.-C. (2024). The Capacity and Robustness Trade-off:
Revisiting the Channel Independent Strategy for Multivariate Time Series
Forecasting. *IEEE Transactions on Knowledge and Data Engineering*, 36(11),
7129–7142.

---

## 7. Referensi Penelitian Terdahulu (2) — artikel jurnal atau prosiding

https://arxiv.org/abs/2310.06625

Liu, Y., Hu, T., Zhang, H., Wu, H., Wang, S., Ma, L., & Long, M. (2024).
iTransformer: Inverted Transformers Are Effective for Time Series Forecasting.
*International Conference on Learning Representations (ICLR) 2024*, spotlight.

---

## 8. Algoritma / Libraries atau Tools yang Digunakan

**Algoritma dan model.** iTransformer (encoder-only, *inverted tokenization*,
*attention* berjalan lintas variat bukan lintas waktu) sebagai model utama;
PatchTST, DLinear, dan LSTM sebagai pembanding; Ridge regression sebagai pembanding
linier; Naive-RW, naive-persist, dan seasonal-naive sebagai *baseline*
deterministik; serta satu ablasi iTransformer dengan *attention* dipaksa seragam
untuk memisahkan sumbangan informasi dari sumbangan *attention*.

**Uji dan estimator statistik.** Clark–West untuk pasangan *nested*;
Diebold–Mariano dengan koreksi Harvey–Leybourne–Newbold dan *long-run variance*
rektangular untuk pasangan *non-nested*; Romano–Wolf *stepdown* untuk multiplisitas;
Model Confidence Set; *restricted wild cluster bootstrap* dengan bobot Rademacher
dan Webb; Pesaran–Timmermann untuk akurasi arah; TOST untuk klaim ekuivalensi;
Davidson–MacKinnon J-test untuk perbandingan model *non-nested*; estimator
Kaplan–Meier/Turnbull untuk data *interval-censored*; dan Deflated Sharpe Ratio
untuk evaluasi ekonomi. Untuk uji efisiensi pasar: ADF, Variance Ratio
Lo–MacKinlay, dan eksponen Hurst. Dimensionalitas efektif diukur dengan
*participation ratio* atas spektrum matriks korelasi.

**Framework deep learning.** PyTorch, satu-satunya. TensorFlow, Keras, dan JAX
tidak dipakai sama sekali.

**Data plane.** polars sebagai pustaka utama untuk segmentasi, rekayasa fitur,
pembentukan *window*, dan pemisahan data; pyarrow dan numpy sebagai pendukung.
pandas hanya dipakai di dua tempat yang disebutkan namanya: skrip pengambilan data
Stage 1, dan batas tempat data menyeberang ke statsmodels, arch, dan wildboottest.

**Pustaka statistik.** scipy, statsmodels, arch, wildboottest.

**Visualisasi.** matplotlib.

**Infrastruktur dan perkakas.** Kaggle Notebook dengan dua GPU NVIDIA T4 untuk
menjalankan 1.620 *run*; Binance REST `/api/v3/klines` melalui requests untuk
pengambilan data; format Parquet untuk seluruh artefak; Git untuk versi; pytest
untuk pengujian; serta LaTeX kelas IEEEtran dan BibTeX untuk naskah.

**Yang sengaja tidak dipakai, karena kemungkinan ditanyakan.** scikit-learn tidak
dipakai — Ridge diselesaikan lewat persamaan normal dalam float64 dan penskalaan
ditulis sendiri, agar keduanya berada pada ruang skala yang sama persis dengan
model lainnya. `DataLoader` PyTorch juga tidak dipakai: seluruh tensor pelatihan
satu *origin* muat dalam memori GPU (paling besar 70,12 MB), sehingga data dimuat
sekali lalu dibatch dengan pengirisan indeks. Tanpa itu satu *grid* yang selesai
dalam hitungan jam akan berubah menjadi puluhan jam.

---

# Lampiran — ringkasan Bab 3 (tidak diminta form, bahan konsultasi)

Form menanyakan delapan butir di atas; tujuh pertama materi Bab 1 dan butir
kedelapan menyentuh Bab 3 hanya sebagai daftar. Ringkasan berikut disiapkan bila
pembimbing menanyakan rancangan Bab 3 lebih jauh dalam pertemuan.

**Data.** Spot BTCUSDT interval satu jam dari Binance REST `/api/v3/klines`,
jendela 2018-01-01T00:00:00+00:00 sampai 2026-08-01T00:00:00+00:00 (akhir
eksklusif). Terukur: 75.216 bar diharapkan, 75.094 bar terkumpul, cakupan 99,8378
persen, 122 bar hilang dalam 27 blok, nol cap waktu duplikat, nol pelanggaran OHLC.
Tiga bar tidak dapat dipakai karena volumenya nol dan high sama dengan low. Sebelas
kolom kline dipertahankan seluruhnya karena `quote_asset_volume`,
`number_of_trades`, dan `taker_buy_base_volume` membawa informasi yang tidak dapat
diturunkan dari OHLC.

**Segmentasi dan pembentukan *window*.** Ini yang semula menjadi butir pertama
rumusan masalah lalu dipindahkan ke sini karena ia kendala metode, bukan pertanyaan
penelitian. Deret dipecah pada setiap *bar* hilang dan pada setiap *bar* bervolume
nol atau ber-*high* sama dengan *low*, sehingga tidak ada *window* yang melintasi
*gap*. Tidak ada *imputation* dalam bentuk apa pun: saat bursa berhenti tidak ada
harga yang terbentuk, sehingga tidak ada nilai yang dapat diduga dan *forward fill*
justru akan mencetak *return* nol palsu yang diikuti lompatan dua jam. Setiap
*window* divalidasi berdasarkan *timestamp*, bukan indeks posisi — jika tidak,
penggeseran posisional akan menutup *gap* secara tak kasatmata setelah baris apa pun
dibuang. Jumlah *window* yang ditolak dicatat per *origin* dan dicocokkan secara
persis dengan tabel *break* per *origin*.

**Variat.** Dua belas variat hasil rekayasa dalam lima famili: lintasan harga,
estimator volatilitas per bar, intensitas perdagangan, aliran order, dan lokasi
intrabar. Tangga variat K sebesar 1, 4, 8, dan 12. Tidak ada indikator teknis, tidak
ada statistik rolling multi-bar, tidak ada dummy kalender.

**Model.** iTransformer encoder-only, panjang lookback 96, horizon utama 24,
`d_model` 128, `d_ff` 256, dua lapis encoder, delapan kepala attention. Seluruh
hiperparameter diadopsi dari Liu dkk. (2024) tanpa penalaan per anak tangga, kecuali
`d_model` yang diturunkan dari 512 karena ukuran sampel. Pembanding: Naive-RW,
naive-persist, seasonal-naive, ridge pada empat anak tangga, DLinear, PatchTST, dan
LSTM.

**Protokol.** Walk-forward rolling-origin dengan lima belas origin berjarak lima
bulan, jendela pelatihan 24 bulan tetap, sub-blok pelatihan 21 bulan dan validasi
tiga bulan, purge sepanjang horizon pada kedua batas, serta enam blok uji tiga puluh
hari tanpa pelatihan ulang. Landasan setiap elemen dipetakan satu per satu ke
literatur di `docs/WALK_FORWARD_FOUNDATION.md`.

**Metrik dan uji.** MSE dan MAE pada log-return terstandardisasi, RelMSE dan
koefisien determinasi out-of-sample terhadap Naive-RW, serta akurasi arah.
Clark–West untuk pasangan bersarang, Diebold–Mariano berkoreksi
Harvey–Leybourne–Newbold dengan estimator varians rektangular untuk pasangan tidak
bersarang, Romano–Wolf untuk multiplisitas, Model Confidence Set untuk keanggotaan,
dan wild cluster bootstrap terestriksi untuk kemiringan peluruhan.

---

# Catatan keputusan

- Judul memakai kata **"praregistrasi"**, bukan "terpraregistrasi", karena bentuk
  kedua terlalu kaku dalam bahasa Indonesia. Konsepnya tetap ada di judul karena
  praregistrasilah yang membedakan temuan nol dari eksperimen yang gagal: ambang,
  margin ekuivalensi, dan efek minimum yang dapat dideteksi semuanya dikunci
  sebelum blok pengujian dibuka.
- Judul di form **tidak** membawa subjudul temuan yang ada pada judul manuskrip.
  Form ini adalah rancangan Bab 1 dan Bab 3, sehingga hasil tidak dilaporkan di
  sini. Judul manuskrip di `CLAUDE.md` §1 tidak berubah.
- Frasa "Baseline Linier" dipertahankan sesuai judul manuskrip. Perlu diingat saat
  menulis Bab 4: LSTM juga masuk Model Confidence Set, sehingga penjelasan di dalam
  naskah harus menyebut arsitektur berbasis attention, bukan pembelajaran mendalam
  secara umum.
- Rumusan masalah butir 2, 3, dan 4 adalah RQ1, RQ2, dan RQ3 yang telah
  dipraregistrasi di `CLAUDE.md` §3, dengan urutan dan isi yang tidak diubah.
- Butir 1, perbandingan terhadap *baseline*, **bukan** pertanyaan penelitian
  pra-registrasi dan harus dinyatakan demikian di Bab 3. Ia tetap dimasukkan karena
  ia temuan utama skripsi ini: §13.2 menjadikannya disclosure wajib, dan §7
  menyuruh melaporkannya sebelum RQ1 karena itulah bingkai tempat semua hasil lain
  dibaca. Ia juga bukan pertanyaan post-hoc — daftar baseline dikunci di §7,
  metriknya di §9.1, ujinya di §9.2, dan gerbang Stage 5 dijalankan pada sub-blok
  validasi, bukan pada blok uji. Yang tidak boleh dilakukan adalah menyebutnya RQ4.
- Butir soal dataset dikeluarkan dari daftar dan dipindahkan ke lampiran Bab 3. Ia
  kendala metode (§4.2, §4.3), bukan pertanyaan penelitian, sehingga menempatkannya
  sejajar dengan RQ akan mengaburkan mana yang dipraregistrasi dan mana yang tidak.
- Butir 3 dibatasi pada **delapan variat lawan satu variat**, bukan "multivariat
  lawan univariat" secara umum. §3 menyatakan RQ2 membandingkan K=1 dengan K=8 dan
  tidak pernah dengan K=12, karena K=12 sengaja dibuat redundan sehingga akan
  mencampuradukkan peluruhan dengan redundansi tersebut.
- Tujuan penelitian berjumlah empat dan berpasangan satu-satu dengan rumusan
  masalah pada urutan yang sama.
- Rumusan masalah tidak memuat hipotesis maupun ambang. H1, H2, H3 beserta
  τ = 5 persen, margin ekuivalensi sebesar 0,25 kali selisih MSE anak tangga
  keempat ke kedelapan, dan efek minimum yang dapat dideteksi semuanya ada di §3
  dan harus muncul di badan Bab 1 atau Bab 3. Tanpa itu, kata "praregistrasi" di
  judul tidak punya isi.
- Prior art terdekat, Bysik dan Ślepaczuk (2026), disebut di paragraf kedua tetapi
  **tidak** dipakai untuk soal 6 maupun 7. Halaman satu PDF-nya dibaca 2026-09-06:
  arXiv 2606.00060v1, tanpa journal-ref, tanpa DOI, tanpa baris "submitted to".
  Masih preprint, sehingga bukan artikel jurnal maupun prosiding dan tidak
  memenuhi §13.3. Kalau versi jurnalnya terbit sebelum sidang, ia kandidat kuat
  untuk menggantikan salah satu dari dua slot itu.
- Kemiripan sampel mereka perlu diketahui pembimbing sejak awal: sekitar 70.000
  observasi per jam 2018–2026, hampir sama dengan 75.094 bar pada studi ini, dengan
  iTransformer termasuk model yang mereka uji. Yang memisahkan kedua studi adalah
  sasaran: mereka mengukur performa perdagangan setelah biaya sepuluh basis poin,
  studi ini mengukur skill statistik terhadap random walk beserta peluruhannya.
  Karena itu klaim kontribusi yang bertahan adalah pengukuran peluruhan yang
  eksplisit, bukan klaim "pertama yang mengevaluasi iTransformer pada aset kripto".
