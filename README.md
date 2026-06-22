# 🤖 Deep Learning Regression — MLP dengan Optuna, MLflow & LIME

> **Final Term Machine Learning | TK46GAB**
> Muhamad Kevin · NIM 101032300243

---

## 📌 Tujuan Repository

Repository ini merupakan bagian dari tugas **Final Term Machine Learning** yang bertujuan membangun dan mengevaluasi pipeline regresi berbasis **Deep Learning (MLP)** secara end-to-end — mulai dari preprocessing, hyperparameter tuning otomatis, experiment tracking, hingga interpretasi prediksi.

---

## 📋 Gambaran Umum Proyek

Proyek ini menggunakan dataset tabular dengan 90 fitur numerik untuk memprediksi satu variabel kontinu (regresi). Seluruh pipeline dibangun di atas ekosistem Python modern:

| Komponen | Tool |
|---|---|
| Model | PyTorch MLP (Multilayer Perceptron) |
| Hyperparameter Tuning | Optuna (TPE Sampler) |
| Experiment Tracking | MLflow |
| Explainability | LIME + SHAP (DeepExplainer) |
| Preprocessing | Scikit-learn (StandardScaler) |

---

## 🗂️ Struktur Repository

```
📦 dl-regression/
├── 📓 DL_Regression.ipynb          # Notebook utama (MLP pipeline)
├── 📄 README.md                    # Dokumentasi ini
├── 📄 Kesimpulan_DL_Regression.md  # Kesimpulan & analisis hasil
├── 📊 hasil_prediksi_mlp_18kolom.csv   # Output prediksi test set
├── 🖼️ lime_single_instance.png     # Visualisasi LIME per instance
└── 📊 lime_feature_importance.csv  # Agregasi bobot fitur LIME
```

---

## 🔬 Deskripsi Model

### Arsitektur: TabularMLP (PyTorch)

Model MLP fleksibel yang dibangun dinamis berdasarkan hyperparameter hasil Optuna. Setiap hidden layer terdiri dari:

```
Linear → [BatchNorm1d] → Activation → Dropout
```

**Parameter yang di-tuning Optuna:**

| Hyperparameter | Range |
|---|---|
| Jumlah hidden layer | 1 — 4 |
| Ukuran hidden unit | 32 — 512 (log scale) |
| Dropout rate | 0.0 — 0.5 |
| BatchNorm | True / False |
| Activation function | ReLU / GELU / LeakyReLU |
| Learning rate | 1e-4 — 1e-2 (log scale) |
| Weight decay | 1e-6 — 1e-2 (log scale) |
| Batch size | 256 / 512 / 1024 / 2048 |

### Preprocessing Data

Sebelum masuk ke model, 90 fitur asli dikelompokkan menjadi **18 fitur** dengan cara merata-ratakan setiap 5 kolom berurutan, lalu dilakukan **StandardScaler** pada X dan y secara terpisah.

---

## 📊 Hasil Evaluasi Model

Evaluasi dilakukan pada **test set (10%)** setelah training dengan hyperparameter terbaik dari Optuna.

| Metrik | Nilai |
|---|---|
| **RMSE** | *(lihat MLflow run)* |
| **MAE** | *(lihat MLflow run)* |
| **R² Score** | *(lihat MLflow run)* |
| **CV RMSE (best trial)** | *(lihat MLflow run)* |

> Jalankan `mlflow ui --port 5000` setelah eksekusi notebook untuk melihat nilai lengkap beserta perbandingan antar trial.

### Top Fitur Berpengaruh (LIME)

Berdasarkan agregasi LIME pada 18 instance test, fitur yang paling mempengaruhi prediksi model dapat dilihat di `lime_feature_importance.csv`.

---

## 🚀 Cara Menjalankan

### 1. Clone Repository

```bash
git clone https://github.com/<username>/dl-regression.git
cd dl-regression
```

### 2. Install Dependencies

```bash
pip install torch optuna mlflow lime shap scikit-learn pandas numpy matplotlib seaborn
```

### 3. Siapkan Dataset

Letakkan file `midterm-regresi-dataset.csv` di direktori yang sama dengan notebook.

### 4. Jalankan Notebook

```bash
jupyter notebook DL_Regression.ipynb
```

Urutan eksekusi cell:
1. **Section 1** — Import library
2. **Section 2** — Load & preprocessing data (feature grouping, train/test split)
3. **Section 3** — Optuna hyperparameter tuning (10 trials, 3-Fold CV)
4. **Section 4** — MLflow experiment tracking & evaluasi final
5. **Section 5** — LIME explainability (per-instance & agregasi)
6. **Section 6** — Ringkasan hasil & export prediksi

### 5. Lihat Hasil di MLflow

```bash
mlflow ui --port 5000
```

Buka browser: `http://localhost:5000`

---

## 📁 Panduan Navigasi Notebook

| Section | Cell | Isi |
|---|---|---|
| **1. Import** | Awal | Semua library yang dibutuhkan |
| **2. Preprocessing** | Load CSV → Feature Grouping → Split | Transformasi 90 → 18 fitur, heatmap korelasi |
| **3. Optuna** | Class MLP → `objective()` → `study.optimize()` | Definisi model, training loop, tuning |
| **3. Visualisasi Optuna** | Setelah tuning | `plot_optimization_history`, `plot_param_importances`, dll |
| **4. MLflow** | `mlflow.start_run()` | Log params, metrics, dan model artifact |
| **5. LIME** | `LimeTabularExplainer` → loop 18 instance | Penjelasan lokal & agregasi fitur |
| **6. Ringkasan** | Akhir | Print summary + export CSV prediksi |

---

## 🔍 Analisis & Catatan

- **Feature grouping** dipilih untuk mengurangi dimensi dan noise dari 90 fitur mentah sebelum masuk ke MLP yang sensitif terhadap skala.
- **Early stopping** (patience=15) diterapkan agar training tidak overfit dan efisien secara komputasi.
- R² yang relatif rendah mengindikasikan bahwa mayoritas fitur bersifat noise — seleksi fitur berbasis SHAP direkomendasikan untuk eksperimen lanjutan.

---

## 👤 Identitas

| | |
|---|---|
| **Nama** | Muhamad Kevin |
| **Kelas** | TK46GAB |
| **NIM** | 101032300243 |
| **Mata Kuliah** | Machine Learning (Final Term) |

---

*Dibuat sebagai bagian dari Final Term Machine Learning — TK46GAB.*
