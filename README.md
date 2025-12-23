# Prediksi Customer Churn E-Commerce  
**Capstone Project (CP) Module 3 – Machine Learning**

**Nama:** Muhammad Arief Munazat  
**Program:** JCDSAH 024  
**Topik:** Supervised Learning – Binary Classification  

---

## 1. Gambaran Umum Proyek

Proyek ini bertujuan untuk membangun **model machine learning klasifikasi** yang dapat memprediksi **customer churn** (pelanggan yang berhenti bertransaksi) pada perusahaan e-commerce.

Dataset sudah memiliki label target (`Churn`), sehingga pendekatan yang digunakan adalah **Supervised Learning** dengan tipe **Binary Classification**:
- `1` → Customer Churn  
- `0` → Customer Tidak Churn  

Hasil model diharapkan dapat membantu bisnis dalam:
- Mengidentifikasi pelanggan berisiko churn
- Menyusun strategi retensi yang lebih tepat sasaran

---

## 2. Load Dataset & Library

### Kode
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data_ecommerce_customer_churn.csv')
df

Penjelasan

pandas dan numpy digunakan untuk manipulasi data

matplotlib dan seaborn digunakan untuk visualisasi

pd.read_csv() membaca file CSV menjadi DataFrame

df digunakan untuk preview awal data


Tujuan tahap ini adalah memastikan data berhasil dimuat dan strukturnya sesuai ekspektasi.


---

3. Data Overview & Struktur Data

Kode

df.shape
df.info()

Penjelasan

df.shape → Mengetahui jumlah baris (observasi) dan kolom (fitur)

df.info() → Melihat:

Nama kolom

Tipe data

Jumlah data non-null

Indikasi missing values



Insight utama:

Dataset berisi 3.941 baris & 11 kolom

Terdapat numerical, categorical, ordinal, dan binary features

Beberapa kolom memiliki missing values



---

4. Duplicate Data Handling

Kode

df.duplicated().sum()
df = df.drop_duplicates()
df.shape

Penjelasan

df.duplicated().sum() menghitung jumlah baris duplikat

drop_duplicates() menghapus baris dengan nilai identik di semua kolom


Alasan:

Data duplikat dapat menyebabkan bias

Model bisa “menghafal” pola tertentu secara berlebihan

Evaluasi model menjadi tidak realistis


Setelah pembersihan, data berkurang menjadi 3.270 baris unik.


---

5. Missing Value Analysis

Kode

df.isnull().sum()

Penjelasan

Kode ini digunakan untuk:

Mengidentifikasi kolom mana yang memiliki nilai kosong

Menentukan strategi preprocessing yang tepat


Kolom dengan missing values:

Tenure

WarehouseToHome

DaySinceLastOrder



---

6. Target Variable Analysis (Churn)

Kode

df['Churn'].value_counts()
df['Churn'].value_counts(normalize=True)

Penjelasan

Menghitung jumlah customer churn dan non-churn

Menghitung proporsinya


Hasil:

Non-Churn ≈ 83.7%

Churn ≈ 16.3%


Artinya dataset imbalanced, sehingga:

Accuracy saja tidak cukup

Fokus evaluasi pada Recall & F1-score



---

7. Exploratory Data Analysis (EDA)

a. Statistik Deskriptif

Kode

df.describe()

Penjelasan

Menampilkan:

Mean, median, std

Min & max

Distribusi numerik


Tujuan:

Mengidentifikasi outlier

Melihat skala data

Menentukan kebutuhan scaling dan capping



---

b. Univariate Analysis

Kode

for col in num_cols:
    df[col].hist(bins=20)
    plt.title(col)
    plt.show()

Penjelasan

Histogram digunakan untuk melihat distribusi setiap fitur numerik

Membantu mendeteksi skewness & outlier


Untuk fitur kategorikal:

df[col].value_counts()

Digunakan untuk melihat dominasi kategori tertentu.


---

c. Bivariate Analysis (Feature vs Churn)

Kode

df.groupby('Churn')[num_cols].mean()

Penjelasan

Membandingkan rata-rata fitur antara churn dan non-churn

Mengidentifikasi fitur yang memiliki perbedaan signifikan


Contoh insight:

Customer churn memiliki tenure lebih pendek

Cashback lebih kecil → risiko churn lebih tinggi



---

d. Correlation Analysis

Kode

corr_matrix = df[num_cols + ['Churn']].corr()
sns.heatmap(corr_matrix, annot=True)

Penjelasan

Mengukur hubungan antar fitur numerik

Mendeteksi multicollinearity

Membantu seleksi fitur pada model linear



---

8. Handling Missing Values

Kode

numeric_cols = df.select_dtypes(include=['int64','float64']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

categorical_cols = df.select_dtypes(include=['object']).columns
df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])

Penjelasan

Strategi imputasi:

Numerik → Median

Lebih robust terhadap outlier


Kategorikal → Mode

Mengisi kategori paling sering muncul



Hasil:

Tidak ada missing values

Semua data dapat digunakan untuk modeling



---

9. Encoding Categorical Variables

Kode

from sklearn.preprocessing import LabelEncoder

label_encoders = {}
cols_to_encode = ['MaritalStatus', 'PreferedOrderCat']

for col in cols_to_encode:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

Penjelasan

Model ML hanya bisa memproses data numerik

LabelEncoder mengubah kategori menjadi angka

Cocok untuk tree-based models (Random Forest, GB)


Mapping disimpan untuk:

Interpretasi

Reproduksibilitas



---

10. Feature & Target Separation

Kode

X = df.drop('Churn', axis=1)
y = df['Churn']

Penjelasan

X → fitur input model

y → target yang ingin diprediksi


Struktur ini wajib sebelum:

Scaling

Train-test split

Model training



---

11. Feature Scaling & Train-Test Split

Kode

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

Penjelasan

StandardScaler menormalkan fitur agar berada pada skala yang sama

train_test_split:

80% data training

20% data testing


stratify=y menjaga proporsi churn tetap konsisten



---

12. Handling Imbalanced Data (SMOTE)

Kode

from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

Penjelasan

SMOTE membuat data sintetis untuk kelas minoritas (churn)

Menghindari model bias ke kelas mayoritas

Meningkatkan Recall & F1-score



---

13. Modeling & Hyperparameter Tuning

Random Forest

Kode

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

rf = RandomForestClassifier(random_state=42)
grid_rf = GridSearchCV(rf, param_grid, cv=5, scoring='f1')
grid_rf.fit(X_train_res, y_train_res)
best_rf = grid_rf.best_estimator_

Penjelasan

Random Forest cocok untuk data tabular & non-linear

GridSearchCV mencari kombinasi parameter terbaik

Metric f1 dipilih karena data imbalanced



---

14. Model Evaluation

Kode

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

y_pred = best_rf.predict(X_test)

Penjelasan

Metrik evaluasi:

Accuracy → keseluruhan prediksi benar

Precision → kualitas prediksi churn

Recall → kemampuan menangkap churner

F1-score → keseimbangan precision & recall


Random Forest memberikan performa terbaik.


---

15. Feature Importance

Kode

best_rf.feature_importances_

Penjelasan

Menunjukkan fitur paling berpengaruh dalam prediksi churn

Insight utama:

Tenure

CashbackAmount

DaySinceLastOrder



Insight ini digunakan sebagai dasar business recommendation.


---

16. Kesimpulan

Dataset layak untuk supervised learning

Random Forest menjadi model terbaik

Faktor utama churn berkaitan dengan:

Loyalitas pelanggan

Aktivitas transaksi

Insentif (cashback)



Model ini dapat digunakan sebagai early warning system untuk strategi retensi pelanggan.

---

Jika kamu mau, **langkah berikutnya** aku bisa bantu:
- Versi README **lebih ringkas (executive summary)**
- **Business Recommendation section** berbasis feature importance
- **Slide / script video capstone (≤15 menit)**
- **Diagram alur ML (flowchart) untuk README**

Tinggal bilang mau lanjut ke bagian mana. 
