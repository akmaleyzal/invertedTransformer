# Dosir Keputusan Topik — iTransformer untuk Peramalan Return Bitcoin Jam-an

| | |
|---|---|
| **Peneliti** | Akmaley Zal (akmal.23078@mhs.unesa.ac.id) |
| **Repositori** | `D:\pythonProject\invertedTransformer` · branch `main` · HEAD `dac969c` |
| **Tanggal dosir** | 2026-08-26 |
| **Status proyek** | Instrumen selesai (894 run, 0 gagal). Manuskrip belum ditulis. |
| **Basis bukti** | 894 `preds/` + 894 `meta/` + 45 `attn/`; `paper/paper_numbers.json`; 163 tes (161 lulus, 2 gagal) |
| **Mutu penelusuran pustaka** | Tingkat penyaringan (*screening-grade*), bukan tinjauan sistematis |

**Kalimat pembingkai.** Proyek ini bukan lagi kandidat topik — ia adalah eksperimen yang sudah
tuntas dijalankan dan yang setiap gerbang pra-registrasinya gagal. Karena itu pertanyaan yang
relevan bukan *"apakah gap ini layak dikejar?"* melainkan *"klaim mana dari tiga klaim yang
tersedia yang sanggup menopang sebuah manuskrip, dan apa yang secara konkret masih kurang?"*

---

## 1. Ringkasan keputusan

### Kandidat A — Evaluasi walk-forward iTransformer pada BTC jam-an dengan pengukuran decay

| | |
|---|---|
| **Apakah celahnya masih terbuka?** | Sebagian tertutup — ada makalah 2026 yang sangat berdekatan |
| **Apakah ini kontribusi?** | Lemah — pembeda utamanya justru yang gagal diukur |
| **Apakah layak dikerjakan?** | Sudah dikerjakan; tinggal penulisan |
| **Verdikt** | **Jangan dijadikan klaim utama** |

### Kandidat B — Jumlah variat nominal (K) versus dimensionalitas efektif (K_eff)

| | |
|---|---|
| **Apakah celahnya masih terbuka?** | Ya, sejauh penelusuran ini |
| **Apakah ini kontribusi?** | Ya, tetapi terdegradasi oleh skill negatif di semua rung |
| **Apakah layak dikerjakan?** | Sudah dikerjakan; tinggal penulisan |
| **Verdikt** | **Bersyarat — kontribusi metodologis paling orisinal yang Anda pegang** |

### Kandidat C — Tidak ada model yang mengalahkan Naive-RW pada jumlah variat mana pun

| | |
|---|---|
| **Apakah celahnya masih terbuka?** | Ya, dan berlawanan arus dengan pustaka arus utama |
| **Apakah ini kontribusi?** | Ya — paling kuat dari ketiganya |
| **Apakah layak dikerjakan?** | Sudah dikerjakan; tinggal penulisan |
| **Verdikt** | **Jadikan klaim utama** |

**Ketidakpastian kunci — dan ini bukan soal teknis.** `bimbingan-skripsi.pdf` (masuk repo
2026-08-24) mendefinisikan ruang lingkup skripsi S1 kelompok riset sebagai *knowledge graph, GNN,
graph database, GraphRAG, NER/IE, arsitektur penyimpanan, Spark/Kafka, benchmarking, BI* — dengan
IEEE Keywords Level 1 seluruhnya bertema graf/KG/LLM/NoSQL. Peramalan deret waktu BTC dengan
iTransformer **tidak muncul di stream primer maupun sekunder, dan tidak dapat mengklaim satu pun
kata kunci Level 1.** Dosir ini tidak dapat menyelesaikan pertanyaan itu; hanya Anda dan pembimbing
yang bisa. Lihat §6.

---

## 2. Definisi kandidat

**Kandidat A — "Evaluasi walk-forward iTransformer dengan pengukuran decay eksplisit."**
Klaim kontribusi (1) di CLAUDE.md §3: *"first walk-forward evaluation of iTransformer on a crypto
asset with explicit decay measurement"*, digabung dengan kontribusi (3), *"evidence-based
retraining cadence under a pre-registered degradation threshold"*.
*Tipe celah:* **belum ada yang menerapkan** — celah aplikasi yang belum ditempati.

**Kandidat B — "K nominal versus K_eff sebagai penjelas gain lintas-variat."**
Kontribusi (2): memisahkan jumlah variat nominal dari dimensionalitas efektif sebagai dua
penjelasan yang bersaing, diuji sebagai perbandingan model non-bersarang.
*Tipe celah:* **metode yang ada tidak menjawab** — literatur LTSF melaporkan K tanpa pernah
mengukur berapa derajat kebebasan yang sebenarnya dibawa K itu.

**Kandidat C — "Tidak ada model yang mengalahkan Naive-RW pada jumlah variat mana pun."**
Bukan kontribusi pra-registrasi. Ia muncul dari data dan kini menjadi judul (`D60a`, 2026-08-20).
*Tipe celah:* **metode yang ada tidak menjawab** — klaim positif di pustaka crypto sebagian besar
tidak diuji terhadap baseline yang benar pada skala yang benar.

---

## 3. Kartu skor keputusan

### Kandidat A

| Gerbang | Skor (1–5) | Alasan |
|---|---|---|
| Celah masih terbuka | **2** | arXiv 2606.00060 (2026) menjalankan iTransformer walk-forward pada BTC/USDT jam-an 2017-12→2026-01, 27 fold, dan menyebut "controlled comparison of XGBoost, LSTM, and iTransformer on hourly cryptocurrency data within a common walk-forward framework" sebagai kontribusinya sendiri |
| Merupakan kontribusi | **2** | Pembeda Anda dari makalah itu adalah pengukuran decay — dan estimand decay kembali **tidak terdefinisi** di 15 dari 15 origin |
| Layak dikerjakan | **5** | Sudah selesai: 894 run, 0 gagal, 7,79 jam satu sesi |
| **Verdikt** | | **Jangan dijadikan klaim utama** |

### Kandidat B

| Gerbang | Skor (1–5) | Alasan |
|---|---|---|
| Celah masih terbuka | **4** | Tidak ditemukan makalah yang memasangkan tangga K nominal dengan participation ratio sebagai regresor bersaing di LTSF. Yang terdekat, arXiv 2401.00230, mereduksi dimensi kanal *di dalam* attention — pertanyaan metode, bukan pertanyaan pengukuran |
| Merupakan kontribusi | **3** | Uji-J memihak K_eff (`t = −0,348, p = 0,7281` untuk K_eff yang diaugmentasi K; `t = +3,293, p = 0,0011` untuk arah sebaliknya) dan `corr(K, K_eff) = 0,828` membuat pacuan itu teridentifikasi. Terdegradasi karena setiap rung ber-`R²_oos` negatif |
| Layak dikerjakan | **5** | Sudah selesai; `keff_table.parquet` ada di disk |
| **Verdikt** | | **Bersyarat — perlu perumusan ulang yang jujur** |

### Kandidat C

| Gerbang | Skor (1–5) | Alasan |
|---|---|---|
| Celah masih terbuka | **4** | Penelusuran mengembalikan pustaka yang didominasi hasil positif (`R²_oos` +4,855%, +2,75%, dan satu R² = 0,9897 pada level harga). Tidak ditemukan null walk-forward crypto yang dipra-registrasi dengan MCS dan Romano–Wolf |
| Merupakan kontribusi | **4** | Falsifikasi, bukan inkremental: ambang pra-registrasi, MDE dipublikasikan sebelum blok tes dibuka, dua arm serangan-referee dijalankan dan keduanya memberatkan model sendiri, dan reproduksi bit-per-bit |
| Layak dikerjakan | **5** | Sudah selesai |
| **Verdikt** | | **Jadikan klaim utama** |

---

## 4. Basis bukti

### 4.1 Corong penelusuran

| Tahap | Jumlah | Catatan |
|---|---|---|
| Formulasi kueri | 3 | dijalankan manual — CLI `research-hub` tidak terpasang di mesin ini |
| Rekaman dikembalikan | 20 | |
| Relevan setelah penyaringan judul/abstrak | 9 | |
| Diambil dan dibaca isinya | 1 | arXiv 2606.00060, diambil penuh via WebFetch |
| **Keyakinan recall** | **RENDAH–SEDANG** | Tanpa penelusuran adversarial, tanpa gerbang relevansi BM25. Satu makalah yang terlewat dapat membalik verdikt "terbuka" mana pun di dosir ini |

### 4.2 Klasifikasi karya terdahulu

| Karya | Hubungan dengan proyek |
|---|---|
| **arXiv 2606.00060** — *Machine Learning-Based Bitcoin Trading Under Transaction Costs* | **Tumpang tindih paling berbahaya.** BTC/USDT **futures** jam-an, 2017-12→2026-01, ±70.000 observasi, Binance REST, 27 fold walk-forward (latih 12 bln / validasi 3 bln / uji 3 bln, maju 3 bln), membandingkan XGBoost, LSTM, **iTransformer**, ±104 fitur. **Tidak** melaporkan `R²_oos` atau RelMSE terhadap random walk; **tidak** mengukur decay; **tidak** menyentuh K atau dimensionalitas efektif; **tidak** dipra-registrasi. Kontribusinya adalah aturan eksekusi sadar-biaya |
| **arXiv 2401.00230** — *Transformer Multivariate Forecasting: Less is More?* | Bersebelahan dengan Kandidat B, tetapi mereduksi dimensi kanal sebagai **metode**; Anda mengukur dimensionalitas efektif sebagai **variabel penjelas**. Berbeda pertanyaan |
| **MSPCIFormer** (Expert Systems, 2026) | Transformer channel-independent untuk peramalan harga crypto — menyentuh pilar channel-independence Related Work Anda |
| Kelompok hasil positif (ScienceDirect S0275531923000314; S2667096824000405; Helformer, J. Big Data 2025; MDPI *Symmetry* 18(1):32; CryptoMamba arXiv 2501.01010) | Konteks untuk Kandidat C. Salah satunya melaporkan R² = 0,9897 — angka yang hanya mungkin pada **level harga**, persis metrik yang CLAUDE.md §9.1 larang karena random walk secara trivial memberi R² ≈ 0,99 di sana |

### 4.3 Karya terdekat per kandidat

- **Kandidat A → arXiv 2606.00060.** Klaim prioritas "pertama" versi tanpa kualifikasi kini
  **dapat dibantah oleh satu hasil penelusuran**, persis yang CLAUDE.md §13.2 peringatkan. Yang
  menyelamatkannya hanya tiga kualifikasi: spot bukan futures, pengukuran decay, dan pra-registrasi.
- **Kandidat B → arXiv 2401.00230.** Tidak tumpang tindih pada pertanyaan intinya.
- **Kandidat C → seluruh kelompok hasil positif.** Tidak ada yang menjalankan MCS atau
  Romano–Wolf, sehingga tidak satu pun dapat menyatakan model mana yang *tak terbedakan* dari
  baseline.

---

## 5. Penilaian per gerbang

### Gerbang 1 — Apakah celahnya masih terbuka?

**Skor: 2 / 4 / 4** (A / B / C)

**Bukti.** arXiv 2606.00060 mencakup aset, interval, rentang tanggal, protokol, dan model yang
sama dengan Kandidat A. Ekstraksi terstruktur dari HTML makalah itu menyatakan ia tidak mengukur
decay model, tidak melaporkan `R²_oos` terhadap random walk, dan tidak mempelajari jumlah variat.

**Interpretasi.** Kalimat kontribusi (1) sebagaimana ditulis sekarang tidak selamat. "Evaluasi
walk-forward iTransformer pada aset crypto" sudah ditempati. Yang belum ditempati adalah
gabungan **spot + pra-registrasi + `R²_oos` terhadap Naive-RW + tangga K**.

**Risiko.** Keyakinan recall rendah–sedang. Tiga kueri manual bukan penelusuran adversarial, dan
absennya sebuah makalah dari korpus ini bukan bukti absennya dari pustaka.

**Tindakan yang dibutuhkan.**
1. Sitasi arXiv 2606.00060 di Related Work dan **nyatakan deltanya secara eksplisit** dalam satu
   paragraf. Makalah yang tidak menyebutnya akan dikembalikan oleh referee mana pun yang mencari.
2. Tulis protokol penelusuran §2 yang CLAUDE.md §13.2 wajibkan — basis data, string kueri,
   tanggal, kriteria inklusi — dan turunkan klaim prioritas menjadi *"sepanjang pengetahuan kami"*.
3. Jalankan ulang gerbang ini dengan penelusuran adversarial sebelum submit.

### Gerbang 2 — Apakah ini kontribusi?

**Skor: 2 / 3 / 4**

**Bukti.**

| Klaim | Angka terukur | Status |
|---|---|---|
| RQ1 — gain melacak K_eff | ΔMSE 4→8 = **+0,000636**; 8→12 = **−0,000437**; TOST vs margin ±0,000159 → p = (0,9734, 0,0002), **tidak terbukti ekuivalen** | Bertahan di pacuannya sendiri |
| RQ2 — celah menyempit seiring usia model | β₁ = **+0,000256** (tanda salah), t = 0,717, SE klaster 0,000358, WCR satu sisi p = **0,7381**, MDE = **−0,000920**, G = 15, N = 90 | **Deskriptif**, bukan konfirmatori |
| RQ3 — cadence retraining optimal | `b*` **tidak terdefinisi** di keempat τ; 15 dari 15 origin dikecualikan karena `R²_oos ≤ 0`; `decay_panel.parquet` **nol baris** | **Tidak terjawab** |
| Headline | `R²_oos`: itr-K1 **−0,02045**, K4 −0,01868, K8 **−0,01799**, K12 −0,01857, uniform-attn −0,01772, ridge **−0,000568**, PatchTST −0,01631, DLinear −0,02625 | Setiap model kalah dari Naive-RW |

**Interpretasi.** Ini kunci seluruh dosir: **satu-satunya hal yang membuat Kandidat A masih
terbuka di Gerbang 1 adalah pengukuran decay, dan pengukuran decay itulah yang gagal.** Sirkularitas
tersebut membuat A tidak dapat menopang manuskrip. Kandidat C tidak punya masalah ini — ia justru
*diperkuat* oleh setiap kegagalan: dua arm robustness (`longsched`, `capacity`) menjawab dua
serangan referee paling jelas dan keduanya membuat hasil **lebih buruk**, di 15/15 dan 14/15 origin.

**Riwayat jalan buntu.** Ini bukan celah yang ditinggalkan karena tak terpecahkan; ini celah yang
ditinggalkan karena **tidak menarik untuk dipublikasikan**. Pustaka crypto menerbitkan positif.
Itu menjadikan Kandidat C sebagai kontribusi *pemecah masalah*, bukan inkremental: ia mengoreksi
praktik pengukuran, bukan menambah satu arsitektur lagi.

**Risiko.** Referee jurnal Sinta informatika lazim mengharapkan artefak positif. Null yang
dipra-registrasi harus dijual sebagai kontribusi protokol evaluasi, bukan sebagai eksperimen gagal.

**Tindakan yang dibutuhkan.** Tulis §4 Results dengan urutan **C → B → A**, bukan RQ1 → RQ2 → RQ3.
Urutan pra-registrasi adalah urutan *pengujian*, bukan urutan *pelaporan*.

### Gerbang 3 — Apakah layak dikerjakan?

**Skor: 5 / 5 / 5 pada sumber daya — tetapi lihat peringatan kelembagaan**

**Bukti.** 894 run selesai, 0 gagal, satu sesi Kaggle 7,79 jam pada satu T4, rata-rata 31,8 s/run.
Data 75.216 bar diharapkan / 75.094 aktual / 75.091 layak pakai / 27 blok gap. Kuota GPU tidak
pernah menjadi kendala. 161 dari 163 tes lulus.

**Interpretasi.** Kelayakan sumber daya bukan lagi pertanyaan — ia sudah menjadi sejarah. Sisa
pekerjaan seluruhnya CPU dan penulisan.

**Risiko.** Yang tersisa adalah **kelayakan kelembagaan**, dan ini belum terselesaikan. Lihat §6.

---

## 6. Risiko dan uji naik-kelas / uji-bunuh

### Risiko 1 — Kesesuaian ruang lingkup kelompok riset *(paling mendesak; tidak dapat diselesaikan dosir ini)*

`bimbingan-skripsi.pdf` mencantumkan ruang lingkup S1 sebagai konstruksi KG, GNN, graph database,
GraphRAG, NER/IE, arsitektur penyimpanan, Spark/Kafka/Hadoop, benchmarking, dan aplikasi BI. IEEE
Keywords Level 1 yang diwajibkan: *Knowledge graphs · Graph neural networks · Ontologies · Semantic
Web · Metadata · Retrieval augmented · Large language models · NoSQL databases*. Proyek ini tidak
memenuhi satu pun.

**Uji-bunuh:** tanyakan pembimbing satu pertanyaan — *"apakah topik ini sudah disetujui di luar
daftar ruang lingkup, atau perlu dipetakan ke salah satu stream?"* Jawabannya menentukan segalanya
di bawah ini. Jangan menulis satu paragraf manuskrip pun sebelum jawaban itu ada.

**Jalur penyelamatan bila jawabannya "harus masuk ruang lingkup":** aset instrumen ini —
protokol walk-forward, inferensi terklaster, kontrak keterlacakan, register divergensi — dapat
dipindahkan ke *Pengembangan model prediktif spasio-temporal berbasis GNN pada data geospasial
deret waktu* (stream sekunder #4) tanpa membuang metodologi. Yang hilang hanya datasetnya.

### Risiko 2 — Validitas konstruk pada K_eff

`D44` sudah menutup versi terburuknya (PR kontemporer buta terhadap struktur lintas-lag, sehingga
ukuran sadar-lookback ditambahkan). Yang tersisa: gerbang Stage 3b **tidak lolos** — PR pada K=8
terukur **4,393 < 5,0** — dan K=12 ber-PR **di bawah** K=8. Ini didisklosur, bukan ditutup.

**Uji naik-kelas:** laporkan PR kontemporer dan PR sadar-lookback berdampingan di Tabel 2b dengan
`corr(K, K_eff) = 0,828` di sebelahnya, dan nyatakan di §4.1b bahwa rung K=12 ternyata **lebih**
redundan daripada rancangannya — kontrolnya jadi lebih kuat, bukan lebih lemah.

### Risiko 3 — Kebaruan Kandidat C

"Model deep tidak mengalahkan baseline linier pada deret finansial" adalah wilayah yang ramai
(garis Zeng dkk. 2023 / DLinear). Yang *tidak* ramai adalah versi yang dipra-registrasi, dengan
MDE dipublikasikan sebelum blok tes dibuka, MCS, dan Romano–Wolf.

**Uji naik-kelas:** jadikan **prosedurnya** yang diklaim, bukan hasilnya. Kalimat yang menjual:
Romano–Wolf menghapus **seluruh** penolakan mentah terhadap Naive-RW (adjusted p ≥ 0,336),
sementara kolom mentah akan mengklaim delapan hasil yang studi ini tidak punya.

### Risiko 4 — Reproduktifitas dan kesegaran artefak *(dapat ditutup hari ini)*

Deliverable `paper/` dibangun dari vintage **684 run**; artefak di disk kini **894 run**. Suite
menyatakannya merah pada dua titik. Rinciannya di §7.

---

## 7. Langkah berikutnya

Jawaban langsung atas pertanyaan Anda — *"apa gap dan langkah pemenuhannya sudah terselesaikan
secara konkret?"* — terbagi dua, dan keduanya harus dibaca bersama.

**Yang sudah selesai secara konkret.** Instrumennya utuh. Manifes 894 run tuntas tanpa satu pun
kegagalan, seluruh `preds/` dan `meta/` ada di disk, ketiga gerbang tahapan sudah dijalankan dan
hasilnya tercatat, ketiga pertanyaan penelitian punya jawaban terukur, dan register divergensi
D01–D62 mendokumentasikan setiap penyimpangan. Ini bukan pekerjaan setengah jadi; ini pekerjaan
eksperimental yang sudah tutup buku.

**Yang belum selesai secara konkret.** Enam langkah yang dokumen analisis Anda sendiri tetapkan
pada 2026-08-24 — **nol dari enam telah dijalankan**, dan verifikasi ulang hari ini
mengonfirmasinya:

1. **Tabrakan `_provenance` masih ada.** `src/itransformer_btc/report.py:259` mendefinisikan
   `def _provenance(...)`, dipanggil di `report.py:619`; sel `CODE_SAVE` di
   `tools/build_notebook.py:1140` menimpanya dengan sebuah string. Dalam notebook yang diratakan
   seluruh modul berbagi satu namespace, sehingga `build_report` mati di sel terakhir setiap sesi
   Kaggle. Ganti nama binding menjadi `_digest_source`, lalu `python tools/build_notebook.py`.
2. **Pemeriksaan tabrakan generator belum diperluas** ke arah modul-lawan-sel-evaluasi.
3. **`paper/` masih vintage 684 run.** `paper/paper_numbers.json` mencatat `runs_complete: 684` dan
   `grid_paper_numbers_sha256: 1d6f08a6…`, sementara artefak di disk memberi `5dc0960a…`.
   `tests/test_report.py::test_generator_check_flag_agrees_with_the_committed_report` gagal karena
   itu. Konsekuensi yang terlihat: **`paper/figures/figure5_attention` tidak ada** meskipun 45
   parquet attention sudah tersedia, dan bagian robustness masih berbunyi *"not run"* untuk tiga
   arm yang sudah berjalan. Jalankan `python tools/build_report.py`.
4. **Cetakan arm falsifikasi belum diperbaiki** agar RelMSE (**+0,000828**, tanda tidak stabil,
   berbalik di 6 dari 15 origin) menjadi headline, bukan MSE mentah yang 99,7%-nya adalah drift
   skaler.
5. **Notebook masih hasil ekspor Kaggle.**
   `tests/test_notebook_sync.py::test_notebook_is_not_stale` merah: *"iTransformer.ipynb is stale
   against src/itransformer_btc/"*.
6. **Manuskripnya belum ada sama sekali.** `paper/` berisi `paper_numbers.json`, sembilan
   `tables/*.tex`, sepuluh berkas figure, lima panel — dan **nol baris prosa**. Tidak ada berkas
   §1 Introduction, tidak ada §2 Related Work, tidak ada §3 Methodology. CLAUDE.md §1 menyatakan
   *"The deliverable is a manuscript, not a model."* Menurut definisi dokumen itu sendiri,
   deliverable proyek ini belum berwujud.

Perbedaan antara kedua daftar itu adalah jawaban ringkas: **gap-nya sudah dipenuhi secara
eksperimental, belum secara tekstual.** Yang Anda miliki adalah instrumen yang selesai dan bukti
yang lengkap; yang belum Anda miliki adalah dokumen yang mengubahnya menjadi klaim.

**Urutan yang disarankan.** Selesaikan Risiko 1 lebih dulu — satu pertanyaan ke pembimbing, dan
jangan menulis sebelum terjawab. Sesudah itu langkah 1, 3, dan 5 adalah pekerjaan satu sesi dan
menghijaukan suite sekaligus memunculkan Figure 5. Baru kemudian menulis, dengan Kandidat C sebagai
klaim utama, Kandidat B sebagai kontribusi metodologis, dan Kandidat A diturunkan menjadi deskripsi
protokol yang menyitir arXiv 2606.00060 secara terbuka.

**Batas dosir ini.** Dosir menyusun bukti untuk tiga gerbang. Apakah topik ini *layak dilanjutkan*
adalah keputusan Anda dan pembimbing — bukan keputusan yang boleh diambil dokumen ini.

---

## Lampiran A. Protokol penelusuran dan penyaringan

| | |
|---|---|
| Tanggal penelusuran | 2026-08-26 |
| Basis data | Web (indeks umum). Tanpa akses Scopus/WoS/IEEE Xplore pada sesi ini |
| CLI `research-hub` | **Tidak terpasang** — mode adversarial dan gerbang relevansi BM25 tidak dijalankan |
| Keluarga kueri | (1) `iTransformer cryptocurrency Bitcoin forecasting walk-forward evaluation negative R2 out-of-sample`; (2) `"effective dimensionality" OR "participation ratio" number of variates multivariate transformer forecasting nominal channel count gains`; (3) `deep learning transformer models fail to beat random walk naive baseline cryptocurrency return forecasting negative out-of-sample R2 replication` |
| Rekaman dikembalikan | 20 |
| Relevan setelah penyaringan | 9 |
| Diambil isinya | 1 (arXiv 2606.00060) |
| Kriteria inklusi | Peramalan crypto dengan model deret waktu deep, atau pengukuran dimensionalitas efektif pada peramalan multivariat |
| Keterbatasan diketahui | Tiga kueri manual; tanpa penelusuran mundur sitasi; tanpa penyaringan dua penilai; satu makalah dibaca isinya, delapan hanya pada tingkat judul/abstrak |
| **Keyakinan recall** | **RENDAH–SEDANG.** Verdikt "terbuka" di dosir ini membawa peringatan itu, dan harus dijalankan ulang secara adversarial sebelum submit |

**Catatan verifikasi.** Selain arXiv 2606.00060, entri di §4.2 belum diverifikasi terhadap sumber
aslinya. CLAUDE.md §13.3 melarang sitasi tanpa DOI terverifikasi dan sumber yang dibaca — entri
tersebut **belum** memenuhi ambang itu dan tidak boleh masuk manuskrip sebelum diverifikasi.

## Lampiran B. Daftar berkas deliverable

```
.research/
├── topic_dossier.md        # dokumen ini
├── topic_dossier.bib       # rujukan Gerbang 1, tingkat penyaringan
└── topic_dossier.gaps.yml  # kandidat + verdikt + pertanyaan terbuka, terstruktur
```

Bukti yang dirujuk dosir ini, di dalam repositori:

```
notebooks/outputs/artifacts/   894 preds/ · 894 meta/ · 45 attn/ · 6 panel parquet
paper/                          paper_numbers.json (vintage 684) · 9 tabel · 10 figure · 5 panel
notebooks/analysis_result.md    analisis grid 894-run, 2026-08-24 (belum di-commit)
docs/DIVERGENCE_REGISTER.md     bukti panjang D01–D62
bimbingan-skripsi.pdf           ruang lingkup kelompok riset (belum di-commit)
```
