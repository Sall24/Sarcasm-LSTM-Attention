# Sarcasm-LSTM-Attention

**Aspect-Based Sentiment Analysis (ABSA) dengan Deteksi Sarkasme pada Komentar YouTube Bilingual (Indonesia-Inggris) menggunakan BiLSTM dengan Incongruity-Aware Attention.**

Repository ini berisi kode eksperimen skripsi S1 Informatika, Universitas Amikom Yogyakarta. Model diuji pada komentar YouTube suporter Liverpool FC yang bercampur bahasa Indonesia dan Inggris (code-mixed), lalu dibandingkan dengan baseline RNN, GRU, dan LSTM.

## Ringkasan

- **Tugas:** klasifikasi 4 kelas (Negatif, Netral, Positif, Sarkasme) dengan konteks aspek
- **Aspek:** pemain, pelatih, wasit, tim, kiper, umum
- **Model utama:** BiLSTM + Incongruity-Aware Attention
- **Baseline:** RNN, GRU, LSTM (bidirectional)
- **Embedding:** FastText (`cc.id.300` dan `cc.en.300`)
- **Framework:** PyTorch

## Incongruity-Aware Attention

Attention biasa hanya menyorot token yang penting secara umum. Pada model ini, attention menghitung dua context vector:

- `ctx_global`: attention standar (Bahdanau) pada token penting
- `ctx_contrast`: attention pada token yang paling menyimpang (divergen) dari konteks keseluruhan kalimat, sebagai sinyal ketidaksesuaian (incongruity) yang khas pada sarkasme

Input ke classifier adalah gabungan `[ctx_global | ctx_contrast | h_last]`, sehingga sinyal kontras ikut dipakai langsung untuk membedakan sarkasme dari sentimen negatif biasa.

## Dataset

| Tahap | Jumlah |
|-------|--------|
| Komentar mentah (22 video YouTube, Nov 2025 - Mei 2026) | 10.659 |
| Setelah pembersihan | 7.502 |
| Setelah augmentasi | 7.942 |
| Setelah ekspansi ABSA (satu sampel per aspek) | 10.444 |

Pembagian data: **train 7.310 / validasi 1.567 / test 1.567** (141 sampel sarkasme di test).

**Pelabelan (weak supervision):**
- Sentimen: `cardiffnlp/twitter-roberta-base-sentiment-latest`
- Sarkasme: zero-shot `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` dengan rule-gate
- Ketidakseimbangan kelas ditangani dengan augmentasi teks, oversampling pada data latih, dan class weight

**Validasi label:** 121 sampel stratified dianotasi manual (`sampel_validasi_annotated_v2_kiw - sampel_validasi_annotated_v2.csv`). Hasilnya agreement **64,46%** dan Cohen's Kappa **0,527** (kategori moderate).

> Data komentar mentah tidak dipublikasikan di repository ini karena memuat identitas pengguna YouTube.

## Hasil (Test Set)

Threshold klasifikasi sarkasme dicari dari data validasi.

| Model | Akurasi | F1-Macro | F1-Sarkasme |
|-------|---------|----------|-------------|
| RNN | 0,6599 | 0,5841 | 0,5251 |
| GRU | 0,6905 | 0,6128 | 0,5526 |
| LSTM | 0,6847 | 0,6283 | 0,6034 |
| **BiLSTM + Incongruity-Aware Attention** | **0,7154** | **0,6423** | **0,6195** |

Model usulan unggul pada ketiga metrik dibanding semua baseline.

**Laporan per kelas (BiLSTM + Incongruity-Aware Attention):**

| Kelas | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Negatif | 0,7918 | 0,8447 | 0,8174 | 824 |
| Netral | 0,6910 | 0,5809 | 0,6312 | 408 |
| Positif | 0,5079 | 0,4948 | 0,5013 | 194 |
| Sarkasme | 0,5897 | 0,6525 | 0,6195 | 141 |

## Konfigurasi

| Parameter | Nilai |
|-----------|-------|
| Vocabulary maksimum | 30.000 |
| Panjang sekuens maksimum | 120 |
| Dimensi embedding | 300 |
| Hidden dimension | 256 (2 layer) |
| Dropout | 0,4 |
| Epoch maksimum / patience | 50 / 10 |
| Optimizer | AdamW |

## Struktur Repository

```
.
├── notebookfe3a16d76d.ipynb                      # pipeline lengkap: preprocessing, pelabelan, ABSA, training, evaluasi
├── sampel_validasi_annotated_v2_kiw - ....csv    # sampel validasi anotasi manual
└── README.md
```

## Cara Menjalankan

Notebook dikembangkan di **Kaggle** dengan GPU Tesla T4.

1. Unggah notebook ke Kaggle (atau Colab) dan aktifkan GPU.
2. Siapkan dataset komentar YouTube dengan kolom `video_id`, `author`, `text`, `like_count`, `published_at`, lalu sesuaikan `CSV_PATH` di notebook.
3. Jalankan sel pertama untuk memasang library dan mengunduh embedding FastText (`cc.id.300.vec`, `cc.en.300.vec`). File ini berukuran besar dan tidak disertakan di repository.
4. Jalankan sel secara berurutan.

Library utama: `torch`, `transformers`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `wordcloud`, `py3langid`.

## Keterbatasan

- Label dihasilkan secara semi-otomatis (weak supervision), dengan agreement terhadap anotasi manual sebesar 64,46%, sehingga hasil evaluasi mengukur kesesuaian dengan label tersebut.
- Kelas Positif dan Sarkasme memiliki jumlah data lebih sedikit sehingga performanya lebih rendah dibanding kelas Negatif.
- Cakupan embedding FastText pada vocabulary sekitar 55%, karena banyak kata slang dan campuran bahasa.

## Author

**Faisal Akbar Ilham Arya Ardhi**
Informatika, Universitas Amikom Yogyakarta
GitHub: [@Sall24](https://github.com/Sall24)
