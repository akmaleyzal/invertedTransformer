**Audit workflow dan validitas riset — 9 September 2026**

Penilaian utama: proyek ini memiliki infrastruktur eksperimen yang cukup disiplin, tetapi beberapa kesimpulan ilmiahnya belum ditopang oleh implementasi evaluasi yang sesuai dengan metodologi tertulis. Persoalan paling mendesak adalah keselarasan waktu target, definisi objek yang dibandingkan, asumsi pengujian statistik, dan makna decay. Menambah run sebelum membereskan hal tersebut akan memperbanyak hasil yang menjawab pertanyaan yang berbeda dari RQ.

Audit ini membaca kondisi working tree saat ini, bukan menganggap catatan audit lama masih berlaku. Orientasi dimulai melalui `graphify query`, kemudian aturan di `CLAUDE.md`, `USAGE.md`, kode data/model/analisis, generator notebook dan laporan, artefak hasil, serta scaffold manuscript ditelusuri. Rujukan metodologis dan implementasi resmi diperiksa secara terarah. Ini merupakan audit riset dengan penelusuran literatur terbatas, bukan systematic literature review atau pembuktian klaim novelty.

Bukti komputasional tersedia dalam [skrip audit](../.research/research_audit_2026_09_09.py) dan [hasil JSON](../.research/research_audit_2026_09_09.json). Skrip membaca eksperimen yang sudah ada, melakukan perhitungan ulang serta counterexample sintetis, dan menulis hanya bukti audit. ID A01–A15 di bawah merupakan ID temuan audit; bukan penggantian atau penutupan ID dalam divergence register.

**Kondisi yang berhasil diverifikasi**

- Terdapat **1.620 run**, seluruhnya berstatus `complete`, dengan satu `code_sha256`: `bfb43f21028da322e123402837625815164a32b70f24f6fa50bed22f6679cadb`.
- Seluruh run mencatat satu input hash; digest parquet aktual cocok: `8270a84b07c2923bc885782a8ba4e1898133d18ee3b260f157fcee3fd6923b4e`.
- `paper/paper_numbers.json` saat ini mencatat 1.620 run dan menunjuk digest grid JSON yang benar. Pernyataan lama bahwa laporan masih tertinggal pada 894 run **tidak berlaku untuk artefak laporan saat audit ini**.
- Test suite selesai: **229 passed dalam 308,77 detik**. Pemeriksaan audit tambahan juga selesai. Kelulusan tersebut mengonfirmasi banyak kontrak implementasi, tetapi beberapa kontrak yang diuji masih berbeda dari pertanyaan ilmiah yang seharusnya dijawab.
- Tidak dilakukan training ulang grid untuk audit ini. Perbandingan ulang lookback menggunakan prediksi yang telah tersimpan.

**Workflow yang berjalan**

| Tahap | Implementasi dan keluaran | Penilaian audit |
|---|---|---|
| Definisi studi | `CLAUDE.md`, `config.py`; tiga RQ, K ladder, origin, horizon, seed | Scope jelas; sebagian klaim pre-registration dan penjelasan statistik perlu dikoreksi |
| Ingest | `spot_klines_btc.py`; JSONL, parquet, gap CSV, laporan integritas | Retensi data mentah dan checksum mendukung penelusuran ulang |
| Segmentation | `segments.py`, `windows.py`, `budget.py` | Tidak melakukan imputation; kontinuitas waktu diperiksa; alasan penyebab setiap gap belum dibuktikan hanya oleh deteksi gap |
| Feature engineering | `features.py`; 12 variates dalam lima keluarga | Transformasi per bar jelas; eksperimen K_eff belum mengisolasi dimensionalitas dari identitas fitur |
| Split dan scaling | `splits.py`; train 21 bulan, validation 3 bulan, enam blok test | Purging training diperiksa; penetapan waktu test memakai awal input, bukan waktu penerbitan forecast |
| Pengukuran sebelum model | `keff.py`, `efficiency.py` | PR training-only tersedia; indikator deskriptif belum merupakan pengujian mekanisme H2 |
| Pilot dan tuning | `runner.stage5_pilot`, `tune_on_validation` | Pilot memakai validation, tetapi statistik pilot mengubah agregasi horizon; MDE justru dihitung setelah test |
| Training dan eksekusi | `model.py`, `baselines.py`, `train.py`, `runner.py` | GPU-resident, banyak seed, metadata; ada ketimpangan objective/budget baseline dan celah resume |
| Analisis RQ dan perbandingan | `metrics.py`, `comparisons.py` | Ada perbedaan estimand antar tabel, J-test tanpa clustering, dan asumsi nested yang bermasalah |
| Economics dan attention | `economics.py`, `attention.py` | Attention layak sebagai deskripsi; DA/P&L memerlukan koreksi definisi dan beberapa perhitungan |
| Notebook dan manuscript | `build_notebook.py`, `notebook_to_src.py`, `report.py`, `paper/manuscript.tex` | Sinkronisasi source diuji, tetapi preservasi output tidak mengikuti perubahan dependency; prose manuscript masih banyak TODO |

**A01 — Waktu evaluasi bergeser sebesar lookback, sehingga perbandingan L tidak memakai jam target yang sama. Prioritas: kritis. Status: terverifikasi pada prediksi aktual.**

Di [splits.py](../src/itransformer_btc/splits.py), `window_starts(..., semantics="origin")` memilih `ts[first]` di dalam blok test. `_gather()` kemudian mengambil target mulai `first + seq_len`, sementara [train.py](../src/itransformer_btc/train.py) menyimpan `split.ts` sebagai `timestamp`. Timestamp itu merupakan **awal input**. Dokumen justru menjelaskan blok test berdasarkan forecast origin dan membolehkan lookback menjangkau periode sebelumnya.

Contoh nyata pada origin pertama, seed 42, K=8, H=24:

| Arm | Timestamp tersimpan pertama | Bar target pertama yang benar-benar dinilai |
|---|---|---|
| L=48 | 2020-01-01 00:00 UTC | **2020-01-03 00:00 UTC** |
| L=96 | 2020-01-01 00:00 UTC | **2020-01-05 00:00 UTC** |
| L=192 | 2020-01-01 00:00 UTC | **2020-01-09 00:00 UTC** |

Target tersebut diverifikasi kembali terhadap return dari parquet mentah setelah inverse scaling. Jadi persamaan label origin/block tidak menjamin persamaan target. Masalah ini memengaruhi interpretasi umur model, batas blok, horizon pengamatan, dan terutama kesimpulan tentang panjang lookback. Ini **bukan bukti bahwa training membaca label test**: input tetap mendahului target. Cacatnya adalah kontrak waktu dan kesetaraan sampel evaluasi.

Pemeriksaan pembanding yang adil juga dilakukan: prediksi tiga L disejajarkan menggunakan waktu target/issuance aktual yang sama, dengan intersection surviving windows pada 15 origin dan lima seed. Pada sampel bersama itu, mean ΔRelMSE **L192−L96 = −0,004164**; semua 15 origin mengarah ke L192 yang lebih baik. Angka paired sebelumnya sekitar **−0,003506** menggunakan sampel masing-masing arm. Dengan demikian arah manfaat L192 masih bertahan pada pemeriksaan ini; besaran dan dasar perbandingan perlu dilaporkan ulang. Pemeriksaan intersection ini eksploratori, bukan pengganti otomatis hasil protokol awal.

Yang kurang: kontrak eksplisit untuk `input_start`, `forecast_origin`, dan `target_timestamp`, serta assertion bahwa pasangan model menilai target aktual yang sama. Perbaikannya dimulai dari definisi waktu itu, kemudian relabel/re-score prediksi yang tersedia. Forecast yang sama sekali belum dibuat tidak dapat dipulihkan hanya dengan mengganti label.

**A02 — Matched-K belum mengisolasi pengaruh K_eff. Prioritas: tinggi. Status: kelemahan desain yang terlihat langsung dari konfigurasi.**

Komentar di [features.py](../src/itransformer_btc/features.py) dan penjelasan `D70` menyatakan bahwa PR merupakan satu-satunya hal yang berubah pada dua subset K=8. Padahal `redundant` memuat ketiga volatility estimators dan `log_mean_trade_size`, sedangkan `orthogonal` memuat `upper_shadow`, `lower_shadow`, dan `vwap_location` sebagai penggantinya.

Artinya, **isi informasi berubah bersamaan dengan PR**. Hasil yang mengunggulkan `orthogonal` dapat berasal dari fitur yang lebih relevan untuk target, representasi yang lebih mudah dipelajari, pengaruh instance normalisation, atau redundansi yang berkurang. Dua subset tidak memisahkan penjelasan itu. Label `orthogonal` juga tidak berarti vektor fiturnya benar-benar ortogonal secara matematis.

Selain itu, PR merupakan ukuran spektrum korelasi, bukan ukuran informasi prediktif mengenai return mendatang. PR tinggi bisa berasal dari noise independen; fitur ber-PR rendah bisa memuat sinyal target yang berguna. Tabel 2b sendiri memperlihatkan bahwa ukuran contemporaneous dan cross-lag memberikan struktur yang berbeda.

Yang dapat diklaim sekarang: **subset dengan komposisi tertentu dan PR lebih tinggi mempunyai error lebih rendah pada konfigurasi yang diuji**. Yang belum dapat diklaim: peningkatan K_eff menyebabkan peningkatan akurasi, atau K_eff secara umum lebih menentukan daripada identitas fitur.

Yang kurang untuk penelitian lanjutan: beberapa kontras yang mengontrol keluarga/isi informasi, kontrol redundansi yang mempertahankan informasi dasar, dan ukuran hubungan fitur–target yang diukur hanya pada training. Eksperimen itu perlu dinyatakan sebagai eksperimen baru; hasil sekarang tidak boleh ditulis ulang seolah desain tersebut sudah dijalankan.

**A03 — J-test RQ1 tidak menerapkan clustering yang dinyatakan metodologi. Prioritas: kritis untuk klaim inferensial RQ1. Status: terverifikasi.**

[metrics.j_test](../src/itransformer_btc/metrics.py) melakukan demeaning berdasarkan origin×block, tetapi covariance-nya adalah `sigma2 * pinv(X.T @ X)`. Itu merupakan covariance OLS konvensional. Fixed effects **tidak otomatis menghasilkan clustered standard errors**.

Perhitungan ulang dengan covariance CR1 per origin dan distribusi t(14), pada data dan regresi yang sama, memberikan diagnostik berikut:

| Pengujian | p implementasi | p diagnostik CR1 per origin |
|---|---:|---:|
| K ditambah fitted K_eff | 0,001123 | **0,006249** |
| K_eff ditambah fitted K | 0,728132 | **0,429679** |

Arah keputusan pada ambang 5% tidak berubah dalam diagnostik ini. Temuannya adalah p-value yang sekarang dilaporkan bukan p-value dari prosedur yang dijanjikan. Diagnostik CR1 tersebut pun belum menyelesaikan ketergantungan antar-origin.

Training windows antar-origin saling tumpang tindih, dan test spans origin berdekatan juga berbagi kalender. Asumsi independensi antarklaster perlu diperiksa pada RQ1, paired contrasts, Romano–Wolf, dan MCS, bukan hanya dicantumkan untuk RQ2. Literatur clustered inference memang membolehkan dependence di dalam cluster dengan asumsi independensi antarklaster; jumlah cluster kecil menambah masalah finite-sample. Lihat [MacKinnon, Nielsen, dan Webb, versi kerja 2022](https://arxiv.org/pdf/2205.03285).

Yang kurang: estimator yang sesuai dengan unit sampling, sensitivity terhadap dependence kalender, dan penjelasan bahwa perkiraan “sekitar empat training sets independen” dalam dokumen merupakan heuristik overlap, bukan estimasi formal effective sample size.

**A04 — Asumsi nested untuk Clark–West tidak mengikuti hanya dari subset fitur; Naive-RW bahkan bermasalah secara struktural pada instance-normalised model. Prioritas: kritis. Status: analisis matematis dan counterexample implementasi.**

[comparisons.nesting_order](../src/itransformer_btc/comparisons.py) menggolongkan Naive-RW versus setiap model, serta sebagian pasangan K, sebagai nested. Namun nested information sets tidak sama dengan nested parametric models. K=1→K=8 mengubah jumlah token dan operasi attention; diperlukan pembuktian bahwa model kecil dapat diperoleh melalui pembatasan parameter model besar.

Ada masalah yang lebih mendasar untuk Naive-RW. Pada K=1, normalisasi dan denormalisasi di [model.py](../src/itransformer_btc/model.py) memberikan sifat:

`f(x + c) = f(x) + c`.

Normalised input tidak berubah saat seluruh input window digeser dengan konstanta c, sedangkan mean yang ditambahkan kembali berubah sebesar c. Forecast konstan nol tidak memenuhi sifat ini. Dengan demikian, fungsi Naive-RW bukan otomatis anggota kelas fungsi iTransformer ber-instance-normalisation tersebut. Cek numerik dengan c=0,2 memberikan selisih dari identitas tersebut hanya sekitar **2,83×10⁻⁷**; alasan utamanya tetap aljabar arsitektur, bukan satu contoh numerik.

Clark–West memperbaiki bias estimasi dalam perbandingan model yang memang nested; hasil teorinya tidak otomatis berlaku untuk seluruh pasangan neural network hanya karena satu pasangan memiliki fitur lebih banyak. Lihat [Clark dan West, working paper 2005](https://www.kansascityfed.org/documents/5368/pdf-RWP05-05.pdf). Penilaian bahwa asumsi itu belum terpenuhi di proyek ini berasal dari audit implementasi, bukan klaim bahwa paper tersebut membahas iTransformer.

Yang kurang: pembenaran nesting per pasangan dan validasi ukuran uji pada DGP yang menyerupai protokol proyek. Alternatifnya adalah merumuskan perbandingan atas forecast/loss dari prosedur yang benar-benar dijalankan, dengan inferensi yang mempertahankan dependence-nya. Nilai R²_oos negatif tetap merupakan pengukuran deskriptif; p-value Clark–West tidak boleh diberi jaminan teoretis yang belum ditunjukkan.

**A05 — Tabel utama, MCS/DM, dan pilot membandingkan objek statistik yang berbeda. Prioritas: tinggi. Status: terverifikasi.**

Tabel utama dibentuk dari **rata-rata MSE antar-seed**. Sebaliknya, [comparisons.build_panel](../src/itransformer_btc/comparisons.py) merata-ratakan **prediksi antar-seed**, baru menghitung loss dan MCS. Yang kedua adalah performa ensemble, sedangkan yang pertama mengestimasi performa rata-rata model hasil satu training.

Identitas yang menjelaskan bedanya:

`mean_seed[(y − prediction_seed)²] = (y − mean_seed[prediction_seed])² + variance_seed(prediction_seed)`.

Pada iTransformer K8 origin 1, audit mendapatkan MSE rata-rata seed **1,319134**, sedangkan MSE ensemble **1,310948**. Selisihnya sekitar **0,62%** dari MSE rata-rata seed. Pada origin 8 dan 15 selisihnya sekitar 0,43% dan 0,37%. Besaran ini relevan ketika efek yang ingin dibaca hanya sepersekian persen. MCS membership yang ditempel ke baris rata-rata run karenanya berasal dari objek forecast berbeda.

Pilot juga berbeda lagi: `stage5_pilot()` memberikan `y_val.mean(axis=1)` dan rata-rata forecast sepanjang H kepada Clark–West. Itu menguji error **rata-rata/cumulative H-step return**, bukan rata-rata error kuadrat tiap step yang menjadi objective utama. Error antar-step dapat saling meniadakan sebelum dikuadratkan.

Yang kurang: satu definisi estimand yang digunakan secara konsisten, atau pelaporan terpisah dan jelas untuk single-run procedure, seed ensemble, dan cumulative-return forecast. Jumlah seed saja tidak menyelesaikan perbedaan tersebut.

**A06 — MDE yang disebut berasal dari pilot sebenarnya dihitung dari slope test; status pre-registration perlu diluruskan. Prioritas: kritis untuk kredibilitas pelaporan. Status: terverifikasi pada alur kode dan angka.**

Di [tools/build_notebook.py](../tools/build_notebook.py), `CODE_RQ2` menjalankan `panel_beta1(amp)` setelah grid dikumpulkan, lalu menghitung `minimum_detectable_beta1(beta.within_slopes)`. `amp` adalah panel test. Audit menghitung ulang dari 15 slope test yang tersimpan dan memperoleh **−0,0009202919371959043**, persis sama dengan MDE yang dipublikasikan.

Sebaliknya, Stage 5 hanya memakai origin 1. Satu origin tidak dapat menyediakan dispersi **antar-15-origin** sebagaimana disebut dalam narasi pilot. Artinya, angka yang tersedia adalah diagnostik sensitivitas berdasarkan hasil test, bukan bukti prospective power analysis sebelum test dibuka.

Ini tidak membuktikan semua keputusan riset dibuat setelah melihat hasil. Formula MDE atau hipotesis bisa saja ditentukan sebelumnya. Akan tetapi, asal-usul angka dan urutan pengukurannya tidak boleh diganti oleh pernyataan di dokumentasi.

Dalam sumber proyek yang diperiksa, belum ditemukan identitas registry dan snapshot pre-analysis yang secara langsung menopang label “Pre-Registered” pada judul. Mungkin ada bukti di luar repository; audit ini tidak mengasumsikan bukti itu tidak ada. Penjelasan [Center for Open Science](https://www.cos.io/initiatives/prereg) membedakan rencana sebelum analisis dari keputusan yang dipengaruhi hasil, termasuk pada existing datasets.

Yang kurang: kronologi berbukti, pemisahan pre-specified versus exploratory, serta penyebutan MDE saat ini sebagai diagnostik post-analysis. Mengulang grid pada data yang sudah dilihat tidak mengubah analisis tersebut menjadi prospective.

**A07 — Definisi RQ3 dapat mendeteksi “decay” ketika skill justru meningkat, dan belum mengestimasi cadence optimal. Prioritas: kritis secara konseptual. Status: counterexample terverifikasi.**

[metrics.decay](../src/itransformer_btc/metrics.py) memakai mean skill dari **seluruh enam blok test** sebagai referensi, kemudian `b_star()` memilih blok pertama yang lebih buruk dari referensi itu sebesar τ. Referensi tersebut mengandung masa depan relatif terhadap blok awal.

Counterexample sederhana: skill per blok **1%, 2%, 3%, 4%, 5%, 6%**. Model membaik monoton. Mean-nya 3,5%, sehingga D pada blok pertama adalah `(3,5%−1%)/3,5% = 71,4%`. Kode menghasilkan **b*=1, event=True pada τ=5%**. Estimator ini mendeteksi posisi di bawah rata-rata keseluruhan, bukan secara khusus hilangnya skill sejak model baru dilatih.

Pada hasil nyata, guard mengeluarkan seluruh origin ladder karena mean skill tidak positif, sehingga laporan “estimand undefined” memang sesuai output kode. Counterexample di atas tidak mengubah angka undefined menjadi angka cadence; ia menunjukkan bahwa mengganti dataset sampai skill positif pun belum membuat definisinya valid untuk makna decay yang dikehendaki.

Lebih jauh, waktu crossing threshold bukan otomatis **cadence optimal**. Optimal memerlukan objective keputusan, biaya retraining, dan perbandingan beberapa kebijakan retraining pada kalender evaluasi yang sama. Arm fresh pada +90 hari hanya menguji satu intervensi terbatas.

Yang kurang: referensi skill yang tidak bergantung pada blok masa depan dan definisi keputusan retraining. Untuk studi sekarang, laporkan keterbatasan estimand dan tidak tersedianya rekomendasi cadence. Evaluasi kebijakan retraining merupakan eksperimen lanjutan tersendiri.

**A08 — RQ2 mencampurkan beberapa pengertian umur model dan masih lemah untuk klaim mekanisme. Prioritas: tinggi. Status: keterbatasan desain.**

Origin 2020-01 memiliki training cutoff **2019-10-01**, lalu validation tiga bulan sebelum test. Bobot yang dipilih pada Januari menggunakan gradient-training data yang berakhir sekitar tiga bulan sebelumnya. Waktu sejak deployment/selection dan waktu sejak observasi training terbaru adalah dua besaran berbeda. Pergeseran timestamp pada A01 menambah offset lagi.

Arm fresh memperbarui training window dan validation window sekaligus, sementara fresh memakai **satu seed per origin** dan aged main arm dirata-ratakan atas lima seed. Perbedaan itu tidak otomatis menimbulkan bias pada mean antar-run, tetapi meningkatkan Monte-Carlo uncertainty pada arm yang justru diharapkan mendeteksi efek kecil.

Origin fixed effects mengontrol level kesulitan rata-rata per origin. Ia tidak dengan sendirinya memisahkan age dari perubahan kalender di dalam origin. Demikian juga naik-turunnya rolling in-sample OLS R² belum membuktikan bahwa conditional relationship berubah; variasi estimator, volatilitas, dan komposisi surviving sample juga bisa berperan.

Yang kurang: definisi age yang eksplisit, ketidakpastian fresh-versus-aged, dan pengujian yang dapat membedakan perubahan informasi, optimisasi, dan kondisi pasar. Angka β₁ yang tidak signifikan mendukung laporan deskriptif, bukan bukti bahwa decay tidak terjadi.

**A09 — Perbandingan baseline belum cukup mengisolasi arsitektur dari objective dan optimisasi; port tidak sepenuhnya sama dengan implementasi resmi. Prioritas: tinggi. Status: kode dan metadata terverifikasi.**

iTransformer dan LSTM dioptimalkan untuk target return. DLinear dan PatchTST memakai loss seluruh channel, termasuk engineered features yang distribusi dan persistensinya berbeda. Early stopping juga mengikuti objective masing-masing. Karena itu, memilih checkpoint terbaik untuk delapan output tidak identik dengan memilih checkpoint terbaik untuk return. Alasan ingin mempertahankan label K=8 tidak menghapus confounding objective tersebut.

Masalah budget terlihat pada metadata terbaru: **DLinear 56/75 run** dan **PatchTST 39/75 run** mencapai cap 30 epoch; iTransformer headline 0/300. Cap tercapai tidak membuktikan training lebih lama pasti membantu, tetapi belum ada bukti pembanding bahwa ranking tersebut stabil setelah kesempatan optimisasi yang memadai. Memberi seluruh arsitektur angka epoch/LR yang sama bukan jaminan fairness optimisasi.

Ada pula perbedaan port yang perlu dinamai. [Implementasi resmi iTransformer](https://raw.githubusercontent.com/thuml/iTransformer/main/model/iTransformer.py) memasang LayerNorm setelah keseluruhan encoder; `ITransformer` lokal langsung memproyeksikan keluaran daftar layer tanpa final encoder norm tersebut. [Implementasi resmi PatchTST](https://raw.githubusercontent.com/yuqinie98/PatchTST/main/PatchTST_supervised/layers/PatchTST_backbone.py) menyediakan BatchNorm dan residual attention dalam encoder, sedangkan versi lokal memakai `EncoderLayer` iTransformer. Itu merupakan adaptasi eksperimental yang perlu dijelaskan, bukan bukti bahwa hasilnya otomatis salah.

Yang kurang: tabel perbedaan implementasi, parity check untuk bagian yang diklaim direplikasi, dan sensitivity terhadap target-channel selection serta budget training baseline. Uji konvergensi tambahan perlu menjadi analisis eksploratori yang dilaporkan apa pun hasilnya. Temuan sekarang hanya mendukung perbandingan konfigurasi dan prosedur training yang benar-benar dijalankan.

**A10 — Directional accuracy dihitung dalam scaler space, bukan arah return aktual. Prioritas: tinggi. Status: terverifikasi pada prediksi aktual.**

[metrics.directional_accuracy](../src/itransformer_btc/metrics.py) menggunakan tanda `y_true` dan `y_pred` langsung, padahal keduanya tersimpan dalam z-score. Tanda `(r−μ)/σ` menyatakan apakah return di atas/bawah mean training, bukan apakah harga naik/turun. Inverse scaling yang relevan adalah `r = zσ + μ`.

Contoh origin 15, iTransformer K8, seed 42: DA cumulative non-overlapping yang dihitung kode adalah **44,44%**, sedangkan setelah inverse scaling menjadi **47,22%**. Pada contoh yang sama, menambahkan kembali drift scaler ke cumulative forecast mengubah **12 dari 180** tanda posisi. Angka ini merupakan contoh satu run, bukan revisi headline seluruh grid.

`economics.positions()` memang secara sengaja memakai `sum(pred_z)σ` sebagai forecast yang di-demean, sementara realised return memakai `sum(actual_z)σ + Hμ`. Itu dapat menjadi strategi tersendiri bila didefinisikan demikian, tetapi tidak sama dengan strategi berdasarkan sign dari prediksi return mentah. DA dan economics perlu menyatakan objek yang sama atau membedakannya dengan terang.

Yang kurang: kontrak scale pada fungsi evaluasi dan penghitungan ulang seluruh DA/PT setelah target arah dipastikan. Jangan menggunakan istilah “raw” untuk dua transformasi berbeda.

**A11 — Economic evaluation belum layak dibaca sebagai backtest spot yang dapat dieksekusi; Sortino juga salah formula. Prioritas: tinggi untuk klaim ekonomi. Status: kombinasi cacat teruji dan keterbatasan desain.**

Di [economics.py](../src/itransformer_btc/economics.py), posisi dapat bernilai −1. Data spot sah dipakai untuk simulasi long/short, tetapi implementasi short membutuhkan asumsi inventori/borrowing dan biaya yang tidak tercakup dalam simulator ini. [Dokumentasi Binance Margin](https://developers.binance.com/en/docs/products/margin-trading/Introduction) menjelaskan borrowing dan dukungan short pada margin. Karena itu, hasil sekarang perlu disebut simulasi P&L dengan asumsi tersebut, bukan langsung kinerja strategi spot tanpa leverage.

Window yang melintasi gap dikeluarkan berdasarkan kejadian masa depan. Untuk evaluasi forecasting bersyarat pada target tersedia, ini dapat dinyatakan sebagai keterbatasan cakupan. Untuk strategi trading, keputusan menjadi flat sebelum gap tidak tersedia bagi trader tanpa oracle. Tambahan masalah: `net_returns()` hanya menerima posisi surviving periods. Ia tidak menerima kalender gap, sehingga tidak dapat membebankan penutupan dan pembukaan kembali ketika narasi menyebut strategi flat selama gap. Biaya likuidasi posisi terminal juga tidak dimodelkan.

Sortino mempunyai cacat numerik yang terpisah. Kode memakai standard deviation dari return negatif saja. Target downside deviation pada MAR=0 seharusnya dihitung relatif terhadap nol, dengan kontribusi nol untuk periode di atas target. Contoh return `[-1%, -1%, +2%, +2%]` menghasilkan **Sortino undefined** dalam kode karena dua kerugian identik mempunyai standard deviation nol, padahal downside risk tidak nol. Dengan downside deviation pada seluruh sampel dan annualisation yang dipakai proyek, nilainya **13,5093**. Definisi dan contoh metode dapat diperiksa pada [Rollinger dan Hoffman, Sortino: A Sharper Ratio](https://www.cmegroup.com/education/files/rr-sortino-a-sharper-ratio.pdf).

Yang kurang: kontrak self-financing/cash, pemilihan long/cash atau asumsi short yang eksplisit, kalender eksekusi, biaya transaksi lengkap, dan pembetulan risk metrics. P&L positif tidak dapat digunakan untuk menutup kelemahan predictive skill sebelum lapisan ini benar.

**A12 — Selection bias dari gap belum hilang hanya karena coverage ditambahkan; kontrol ukuran training yang dijanjikan juga belum berjalan. Prioritas: tinggi untuk generalisasi dan RQ2. Status: metadata dan kode terverifikasi.**

Jumlah window main training aktual adalah **13.545–15.217**, berbeda sekitar 12,3% relatif terhadap jumlah minimum. `build_origin_tensors()` memakai seluruh training indices dan manifest tidak memuat arm yang menyamakan jumlahnya. Jadi kontrol subsampling yang disebut dalam §8.1 dan window budget belum merupakan hasil eksperimen yang tersedia.

Untuk ladder pada origin yang sama, jumlah data tetap sama antar-K; ini **tidak membatalkan kontras within-origin secara otomatis**. Kekhawatirannya adalah generalisasi antar-origin dan klaim bahwa fixed-duration windows menghilangkan seluruh variasi sample size/training exposure.

Coverage covariate memang sudah diimplementasikan; audit tidak menganggap `D80` masih terbuka. Namun covariate berupa proporsi data yang selamat tidak mengembalikan outcome yang dibuang atau membuktikan missingness dapat diabaikan. Hubungan outage dengan stress yang dijadikan alasan juga memerlukan bukti, bukan hanya gap CSV. Absennya bar dari artifact REST sendiri belum membedakan maintenance, trading halt, atau kehilangan di jalur data.

Yang kurang: definisi populasi hasil sebagai surviving continuous windows, analisis sensitivitas yang mempertahankan kalender, serta bukti penyebab gap bila disebut exchange downtime. Kontrol ukuran training dapat dilakukan sebagai eksperimen baru bila diperlukan untuk mempertahankan klaim tersebut; tidak perlu mengubah kebijakan no-imputation.

**A13 — Resume dan preservasi output notebook masih dapat mempertahankan hasil lama setelah kode berubah. Prioritas: kritis sebelum rerun atau revisi eksperimen. Status: dua counterexample terverifikasi.**

Pertama, [runner.pending](../src/itransformer_btc/runner.py) menyaring vintage code dengan ketat, tetapi `execute()` dan `execute_parallel()` kembali memanggil `completed_run_ids(roots)` tanpa code digest dan juga memeriksa `is_complete()` yang tidak membandingkan vintage. Pemeriksaan awal dapat meminta run diulang, kemudian eksekusinya tetap melewati run tersebut.

Counterexample menggunakan direktori sementara dan metadata kode lama, tanpa fitting model, menghasilkan: **pending=1 → completed=0, skipped=1, remaining=1**. Jadi `D85` belum menutup seluruh alur pemakaiannya. Ini merupakan masalah workflow nyata; bukan bukti bahwa 1.620 run saat ini bercampur vintage, karena audit hash justru menunjukkan satu vintage yang konsisten.

Kedua, [build_notebook.carry_outputs](../tools/build_notebook.py) memeriksa source cell pemanggil, tetapi tidak perubahan dependency-nya. Contoh `f()` semula mengembalikan 1 lalu diubah menjadi 2; cell `print(f())` tetap sama. Generator tetap membawa output **1** ke notebook baru. Kesamaan teks cell pemanggil tidak membuktikan output masih berasal dari fungsi yang sekarang tersedia.

Yang kurang: satu keputusan completeness yang konsisten sampai eksekusi, validasi resolved config/input vintage, dan invalidation output notebook berdasarkan dependency atau digest seluruh program/input yang menghasilkan output. Bukti yang telah tersimpan boleh dipertahankan sebagai hasil vintage lama, tetapi tidak boleh ditampilkan sebagai hasil kode baru tanpa penanda tersebut.

**A14 — Beberapa justifikasi metodologis terlalu absolut atau tidak sesuai kemampuan library. Prioritas: sedang, tetapi penting untuk pertahanan metodologi. Status: dokumentasi resmi dan inspeksi.**

`CLAUDE.md`, `pyproject.toml`, dan komentar kode berulang kali menyatakan bahwa Polars membuat `center=True` tidak dapat direpresentasikan. Padahal API resmi `Expr.rolling_mean` **memiliki parameter `center` dan contoh `center=True`**. Lihat [dokumentasi Polars](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.rolling_mean.html). Perlindungan proyek saat ini berasal dari pilihan membangun fitur per bar dan pemeriksaan kronologi, bukan ketidakmampuan library membuat centered windows. Tetap memakai Polars merupakan keputusan yang sah; alasan keamanannya perlu akurat.

Demikian pula, parameter count yang sama pada uniform-attention dan learned-attention tidak berarti effective capacity sama: Q/K pada uniform branch tidak dipakai, dan dropout bobot attention di branch learned tidak berjalan pada uniform branch. Ablation itu tetap berguna, tetapi menguji paket perubahan tersebut, bukan seleksi attention yang steril dari perubahan regularisation/effective capacity.

ARIMA boleh tetap deferred sesuai scope keputusan. Akan tetapi, ADF atau variance ratio tidak membuktikan bahwa AIC pasti memilih (0,0,0). Pernyataan demikian sebaiknya dinyatakan sebagai dugaan atau alasan pragmatis, bukan hasil yang sudah diperoleh.

Yang kurang: membedakan batas scope, hipotesis, sifat matematis, dan jaminan implementasi. Dokumen yang sangat rinci tetap memerlukan pengujian terhadap alasan yang dipakai untuk membenarkan setiap aturan.

**A15 — Manuscript, novelty, dan batas kesimpulan belum tuntas. Prioritas: tinggi untuk deliverable riset. Status: terverifikasi pada dokumen.**

[paper/manuscript.tex](../paper/manuscript.tex) sudah menghubungkan banyak tabel/figure, tetapi abstract, introduction, related work, sejumlah methodology, limitations, dan conclusion masih berisi TODO. Bagian search protocol untuk novelty juga belum diisi. Repository memiliki hasil dan scaffold, tetapi belum memiliki argumentasi manuscript yang selesai.

README masih memakai judul comparative lama dan grid 684; USAGE mencampurkan jumlah run/cell dari beberapa fase; RUN_ANALYSIS mempertahankan sejumlah tabel historis di bawah catatan bahwa keadaan sudah berubah. Artefak JSON saat ini telah konsisten, sehingga kelemahannya adalah **panduan membaca hasil**, bukan alasan untuk menuduh hasil terbaru masih menggunakan vintage lama.

Frasa “no out-of-sample skill” perlu dibatasi pada model, target, horizon, preprocessing, sampel, dan agregasi yang diuji. Mean R²_oos negatif tidak membuktikan ketiadaan predictability pada seluruh pasar Bitcoin atau seluruh kelas attention. Beberapa origin baseline dapat mempunyai skill positif walaupun rata-rata keseluruhannya negatif. Klaim “pertama” memerlukan log pencarian literatur yang dapat diperiksa; audit ini tidak menetapkan novelty hanya dari repository.

Yang kurang: manuscript yang menghubungkan RQ dengan estimand yang benar, pembaruan pintu masuk dokumentasi, bukti pencarian novelty, dan batas klaim. Penelitian tetap dapat bernilai sebagai evaluasi negatif yang transparan setelah masalah evaluasinya diselesaikan.

**Apa yang tetap dapat dipertahankan saat ini**

Pada agregasi yang saat ini digunakan laporan, pola perbandingan deskriptifnya jelas:

| Model | Mean R²_oos terhadap Naive-RW | Catatan |
|---|---:|---|
| Naive-RW | 0 | Forecast nol pada raw return |
| Ridge K4 | −0,000292 | Deterministik |
| LSTM K8 | −0,001559 | Lima seed per origin |
| PatchTST K8 | −0,016353 | Lima seed; objective seluruh channel |
| iTransformer K8 | −0,017993 | Lima seed; target-channel objective |
| DLinear K8 | −0,026209 | Lima seed; objective seluruh channel |

Sumber: `paper/paper_numbers.json`, `main_results.by_model`; 15 origin, H=24. Angka ini menggambarkan evaluasi yang tersedia dan tetap perlu dibaca bersama A01–A15. MCS dan p-value tidak ikut dianggap tervalidasi hanya karena mean-nya dapat direproduksi.

Hasil negatif terhadap Naive-RW tidak membuat RQ1 tidak bermakna: perbedaan error antara dua model yang sama-sama kalah masih dapat diteliti. Yang perlu dibatasi adalah penafsiran perbedaan itu sebagai skill yang menguntungkan, mekanisme sebab-akibat K_eff, atau rekomendasi trading/retraining.

**Urutan perbaikan yang saya sarankan**

| Urutan | Hasil yang perlu dicapai | Mengapa didahulukan |
|---|---|---|
| 1 | Kontrak waktu target dan definisi single-run/ensemble/cumulative loss konsisten | Semua perbandingan sesudahnya bergantung pada objek dan sampel yang benar |
| 2 | Nested/non-nested reasoning, J-test, dependence-aware uncertainty, dan asal MDE dikoreksi | Menentukan apa yang sah disebut bukti inferensial atau pre-specified |
| 3 | DA dan economics dihitung pada definisi yang jelas; RQ3 dibatasi sesuai estimand | Mencegah salah penafsiran arah, risiko, dan cadence |
| 4 | Jalur resume serta dependency invalidation notebook diperbaiki dan diberi regression check | Rerun berikutnya harus benar-benar menghasilkan dan menampilkan vintage baru |
| 5 | Nilai kebutuhan sensitivity baseline dan kontrol fitur/sample size sebagai eksperimen baru | Menguatkan interpretasi arsitektur tanpa menyamarkan analisis post-hoc |
| 6 | Selesaikan manuscript dan protokol novelty dari klaim yang sudah dapat dibuktikan | Deliverable ilmiah memerlukan argumen dan batas kesimpulan, bukan hanya tabel |

Tidak seluruh tindakan membutuhkan GPU. Koreksi banyak metrik, definisi aggregasi, pelaporan MDE, sebagian alignment prediksi, dan penulisan dapat menggunakan artefak yang sudah ada. Forecast untuk timestamp yang belum pernah dievaluasi, perubahan arsitektur, atau sensitivity training memang membutuhkan run baru; itu dikerjakan setelah kontraknya benar, dengan provenance dan status eksploratori yang jelas.

**Catatan penelusuran sumber luar**

Penelusuran dilakukan 9 September 2026, bahasa Inggris, berfokus pada sumber primer/otoritatif yang langsung berkaitan dengan temuan. Sumber dipilih berdasarkan naskah penulis, repository resmi penulis model, dokumentasi library resmi, dan dokumentasi platform. Search snippets dipakai sebagai petunjuk; kutipan substansi di atas merujuk halaman/file yang dibuka. Tidak dilakukan penghitungan hits atau screening sistematis untuk klaim kebaruan.

| Fokus | Contoh query atau URL yang diperiksa | Penggunaan |
|---|---|---|
| Clustered inference | `MacKinnon Nielsen Webb cluster robust inference guide empirical practice`; arXiv 2205.03285 | Asumsi cluster dan finite-sample uncertainty, A03 |
| Pre-registration | `site.cos.io preregistration existing data before analysis` | Pemisahan rencana sebelum analisis dan post-analysis, A06 |
| Nested forecast tests | `Clark West approximately normal tests equal predictive accuracy nested models` | Asumsi pengujian CW, A04 |
| Fidelity model | Repository resmi `thuml/iTransformer` dan `yuqinie98/PatchTST` | Perbedaan port encoder, A09 |
| Economics | Dokumen Sortino di CME dan Binance Margin introduction | Risk metric serta asumsi eksekusi short, A11 |
| Safety claim library | API resmi `polars.Expr.rolling_mean` | Verifikasi keberadaan centered window, A14 |

Perintah verifikasi lokal:

```powershell
.venv/Scripts/python.exe -m pytest tests/ -q
.venv/Scripts/python.exe .research/research_audit_2026_09_09.py
```

Perhitungan CR1, intersection lookback, perubahan DA, dan counterexample dalam audit ini adalah diagnostik audit. Mereka tidak diberi status hasil confirmatory baru dan tidak menggantikan angka pada tabel manuscript secara diam-diam.
