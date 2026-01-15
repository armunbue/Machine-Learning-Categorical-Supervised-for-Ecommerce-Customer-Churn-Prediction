# **E-COMMERCE CUSTOMER CHURN PREDICTION**

**Nama:** Muhammad Arief Munazat  
**Program:** JCDSAH 024  
**Topik:** Supervised Learning – Binary Classification  

---

## **PROJEK OVERVIEW & BUSINESS CONTEXT**

### **Business Problem**

Perusahaan e-commerce ingin **memprediksi pelanggan yang berpotensi churn** (berhenti berlangganan/membeli) untuk:

- Memberikan penawaran atau promosi yang sesuai
- Mengoptimalkan strategi retensi pelanggan
- Mengurangi biaya akuisisi pelanggan baru
- Meningkatkan Customer Lifetime Value (CLV)

### **Dataset Overview**

- **Sumber:** `data_ecommerce_customer_churn.csv`
- **Jumlah Data:** 3.941 observasi (pelanggan)
- **Jumlah Fitur:** 11 kolom (10 features + 1 target)
- **Target Variable:** `Churn` (Binary: 0 = Aktif, 1 = Churn)
- **Distribusi Target:**
    - Non-Churn: 82.9% (3,267 pelanggan)
    - Churn: 17.1% (674 pelanggan) → **Imbalanced Dataset**

### **Feature Definition Detail**
Fitur-fitur dalam dataset mencakup aspek demografis, perilaku transaksi, kepuasan layanan, serta histori interaksi pelanggan, seperti lama berlangganan (Tenure), jarak logistik (WarehouseToHome), intensitas penggunaan platform, hingga indikator ketidakpuasan seperti komplain dan jarak waktu sejak pesanan terakhir.
| No. | Feature Name | Data Type | Description | Business Significance |
| --- | --- | --- | --- | --- |
| 1 | Tenure | Numerical (Discrete) | Lama menjadi pelanggan (bulan) | Loyalitas & engagement |
| 2 | WarehouseToHome | Numerical (Discrete) | Jarak gudang ke rumah (km/mile) | Pengalaman logistik |
| 3 | NumberOfDeviceRegistered | Numerical (Discrete) | Jumlah perangkat terdaftar | Multi-device engagement |
| 4 | PreferedOrderCat | Categorical (Nominal) | Kategori pesanan favorit | Preferensi belanja |
| 5 | SatisfactionScore | Numerical (Ordinal) | Skor kepuasan (1-5) | Customer experience |
| 6 | MaritalStatus | Categorical (Nominal) | Status pernikahan | Segmentasi demografi |
| 7 | NumberOfAddress | Numerical (Discrete) | Jumlah alamat terdaftar | Fleksibilitas pengiriman |
| 8 | Complain | Binary | Ada komplain bulan lalu | Service quality indicator |
| 9 | DaySinceLastOrder | Numerical (Discrete) | Hari sejak pesanan terakhir | Recency metric |
| 10 | CashbackAmount | Numerical (Continuous) | Rata-rata cashback | Insentif & reward |
| 11 | Churn | Binary (Target) | Status churn | Target prediksi |

### **Atribute Information Dari Dataset:**
Setiap atribut diklasifikasikan berdasarkan jenis data statistik dan kebutuhan machine learning, mencakup numerical, categorical, ordinal, dan binary features. Variabel Churn ditetapkan sebagai target label dan tidak digunakan sebagai input model.
| Attribute Name           | Jenis Data (Statistik) | Jenis Data (ML) |
| ------------------------ | ---------------------- | --------------- |
| Tenure                   | Numerik (Diskrit)      | Numerical       |
| WarehouseToHome          | Numerik (Diskrit)      | Numerical       |
| NumberOfDeviceRegistered | Numerik (Diskrit)      | Numerical       |
| PreferedOrderCat         | Kategorikal (Nominal)  | Categorical     |
| SatisfactionScore        | Ordinal                | Ordinal         |
| MaritalStatus            | Kategorikal (Nominal)  | Categorical     |
| NumberOfAddress          | Numerik (Diskrit)      | Numerical       |
| Complain                 | Biner                  | Binary          |
| DaySinceLastOrder        | Numerik (Diskrit)      | Numerical       |
| CashbackAmount           | Numerik (Kontinu)      | Numerical       |
| Churn                    | Biner (Target)         | Label           |

**Kategori Value:**
- PreferOrderCat :
    - Laptop & Accessory
    - Mobile
    - Fashion
    - Mobile Phone
    - Grocery
    - Others
- MaritalStatus :
    - Single
    - Married
    - Divorced

## **Blueprint Summary**
- Blueprint ini menggambarkan alur end-to-end machine learning untuk memprediksi customer churn pada bisnis e-commerce, mulai dari pemahaman data hingga interpretasi bisnis.
- Proyek menggunakan supervised binary classification, dengan variabel target Churn (aktif vs churn). Data dibagi secara stratified 80:20 untuk memastikan proporsi churn tetap konsisten antara data training dan testing.
- Seluruh proses preprocessing (handling missing values, encoding kategorikal, dan scaling numerik) dibangun dalam satu pipeline dan hanya di-fit pada data training untuk mencegah data leakage. Pendekatan ini memastikan evaluasi model bersifat objektif dan dapat direplikasi.
- Beberapa model ensemble dan tree-based (Decision Tree, Random Forest, AdaBoost, Gradient Boosting, dan XGBoost) dievaluasi menggunakan cross-validation, dengan fokus pada F1-score, precision, dan recall karena churn merupakan masalah klasifikasi tidak seimbang.
- Model terbaik kemudian melalui hyperparameter tuning untuk menyeimbangkan kemampuan mendeteksi churn (recall) dan ketepatan prediksi (precision). Evaluasi akhir dilakukan pada data test yang benar-benar tidak terlihat, menggunakan metrik komprehensif seperti ROC-AUC dan Precision–Recall Curve.
- Hasil model diterjemahkan ke dalam insight bisnis, termasuk identifikasi faktor utama penyebab churn dan rekomendasi strategis untuk customer retention, churn mitigation, dan early warning system.

---

# E-Commerce Customer Churn Prediction - Code Explanation

## 📌 **Project Overview**
```python
# Import semua library yang diperlukan
# Data Manipulation
import pandas as pd  # Untuk manipulasi data dalam bentuk DataFrame
import numpy as np   # Untuk operasi numerik dan array
import seaborn as sns # Untuk visualisasi data yang menarik
import matplotlib.pyplot as plt # Untuk plotting grafik

# Data Splitting & Model Validation
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, cross_val_score, cross_validate
# train_test_split: membagi data menjadi training dan testing
# RandomizedSearchCV: mencari hyperparameter terbaik secara acak
# StratifiedKFold: cross-validation yang menjaga proporsi kelas
# cross_val_score: mengevaluasi model dengan cross-validation

# Pipeline & Feature Transformation
from sklearn.compose import ColumnTransformer # Mentransformasi kolom berbeda dengan cara berbeda
from sklearn.pipeline import Pipeline # Menggabungkan beberapa langkah preprocessing/model
from sklearn.impute import SimpleImputer # Menangani missing values

# Encoding & Scaling
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
# OneHotEncoder: mengubah kategori menjadi variabel dummy
# StandardScaler: menormalisasi data numerik (mean=0, std=1)
import category_encoders as ce # Library tambahan untuk encoding kategorikal

# Model Classification
from sklearn.tree import DecisionTreeClassifier, plot_tree # Decision tree
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
# Ensemble methods: RandomForest, AdaBoost, GradientBoosting
from xgboost import XGBClassifier # XGBoost algorithm

# Model Evaluation
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, precision_recall_curve, auc
# Metrics: precision, recall, f1, ROC-AUC, PR-AUC
```

## 📊 **1. Exploratory Data Analysis (EDA)**
```python
# Load dataset dari file CSV
df = pd.read_csv('data_ecommerce_customer_churn.csv')

# Preview data - melihat 5 baris pertama
df.head()

# Membuat baseline model: prediksi semua customer tidak churn
from sklearn.metrics import confusion_matrix
y_actual = df['Churn']  # Target actual
y_pred_baseline = [0] * len(y_actual)  # Prediksi semua 0 (non-churn)

# Confusion matrix untuk baseline
cm_baseline = confusion_matrix(y_actual, y_pred_baseline)
print("Confusion Matrix - Baseline (all non-churn):\n", cm_baseline)

# Visualisasi confusion matrix baseline
plt.figure(figsize=(5,4))
sns.heatmap(
    cm_baseline, 
    annot=True,  # Menampilkan angka dalam cell
    fmt='d',     # Format integer
    cmap=sns.light_palette("#006400", as_cmap=True),  # Warna hijau
    xticklabels=['Non-Churn','Churn'], 
    yticklabels=['Non-Churn','Churn']
)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Before Modelling')
plt.show()

# Informasi struktur data
df.info()  # Menampilkan tipe data dan missing values

# Statistik deskriptif
df.describe()  # Mean, std, min, max, quartiles

# Cek duplikat
df.duplicated().sum()  # Jumlah baris duplikat
```

## 📈 **2. Visualisasi Distribusi Data**
```python
# Mendefinisikan fitur numerik
num_features = [
    'Tenure',
    'WarehouseToHome',
    'NumberOfDeviceRegistered',
    'SatisfactionScore',
    'NumberOfAddress',
    'Complain',
    'DaySinceLastOrder',
    'CashbackAmount'
]

target = 'Churn'

# Visualisasi distribusi churn dengan persentase
plt.figure(figsize=(4, 5))

# Hitung jumlah dan persentase churn
churn_counts = df['Churn'].value_counts()
churn_percentage = df['Churn'].value_counts(normalize=True) * 100

# Buat bar chart
bars = plt.bar(['Non-Churn (0)', 'Churn (1)'], 
               churn_counts.values,
               color=['#006400', '#9B111E'],  # Hijau dan merah
               alpha=0.7)

# Tambahkan persentase di atas bar
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2,
             height + max(churn_counts.values)*0.01,
             f'{churn_percentage[i]:.1f}%',  # Format 1 desimal
             ha='center',
             va='bottom',
             fontsize=10)

plt.title("Churn Distribution with Percentage")
plt.ylabel("Number of Customers")
plt.xticks(rotation=0)
plt.grid(axis='y', alpha=0)
plt.tight_layout()
plt.show()
```

Distribution Result (Bar Chart):

<img width="390" height="490" alt="image" src="https://github.com/user-attachments/assets/27f11658-5c57-4851-bd4f-90d0ea7cb7b0" />

Interpretasi: Menunjukan distribusi customer churn vs non-churn dalam bentuk bar chart vertikal:
- Non-Churn distribution    : 82.9%
- Churn distribution        : 17.1%

> Artinya, sebagian besar customer tetap aktif.

```python
# KDE Plot untuk fitur numerik (distribusi berdasarkan churn)
fig, axes = plt.subplots(2, 4, figsize=(18, 8))  # 2x4 grid
axes = axes.flatten()  # Ubah menjadi array 1D

for i, col in enumerate(num_features[:8]):
    sns.kdeplot(
        data=df,
        x=col,
        hue="Churn",  # Warna berbeda untuk churn/non-churn
        common_norm=False,  # Normalisasi terpisah
        fill=True,  # Area terisi
        alpha=0.4,  # Transparansi
        ax=axes[i]
    )
    axes[i].set_title(f'{col}')
    axes[i].set_xlabel('Churn')
    axes[i].set_ylabel(col)

plt.suptitle('Distribusi KDE Fitur Numerik terhadap Churn', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
```
KDE Plot Result:

<img width="1789" height="789" alt="image" src="https://github.com/user-attachments/assets/b72e71fe-328a-4b38-9d80-7e89c353fdcb" />

**Interpretasi Distribusi Fitur Numerik terhadap Churn:**

| Fitur                        | Pola Distribusi                                                                                                        | Interpretasi Utama                                                                                                                                                                |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Tenure**                   | Distribusi churn (1) sangat terkonsentrasi pada tenure rendah, sementara non-churn (0) lebih menyebar ke tenure tinggi | Customer baru memiliki risiko churn jauh lebih tinggi. Semakin lama pelanggan bertahan, semakin rendah probabilitas churn. Tenure merupakan indikator loyalitas yang sangat kuat. |
| **WarehouseToHome**          | Distribusi churn sedikit bergeser ke kanan dibanding non-churn                                                         | Customer dengan jarak warehouse lebih jauh cenderung memiliki risiko churn lebih tinggi, kemungkinan akibat pengalaman pengiriman yang kurang optimal.                            |
| **NumberOfDeviceRegistered** | Non-churn dominan pada 3–4 device, churn lebih tersebar dan relatif lebih tinggi pada device rendah                    | Pengguna dengan lebih banyak device terdaftar menunjukkan engagement yang lebih tinggi dan lebih “sticky”. Single-device user lebih rentan churn.                                 |
| **SatisfactionScore**        | Churn lebih dominan pada skor kepuasan rendah hingga menengah, non-churn lebih terkonsentrasi di skor lebih tinggi     | Kepuasan pelanggan berhubungan negatif dengan churn. Skor kepuasan rendah meningkatkan probabilitas churn secara signifikan.                                                      |
| **NumberOfAddress**          | Distribusi relatif mirip, namun non-churn sedikit lebih dominan pada jumlah alamat rendah–menengah                     | Banyak alamat mencerminkan fleksibilitas dan aktivitas penggunaan. Customer yang lebih aktif cenderung lebih bertahan.                                                            |
| **Complain**                 | Churn memiliki puncak yang jelas pada nilai complain = 1, non-churn dominan pada 0                                     | Riwayat komplain merupakan sinyal churn yang kuat. Customer yang pernah complain jauh lebih berisiko untuk churn.                                                                 |
| **DaySinceLastOrder**        | Distribusi churn bergeser ke kanan (hari lebih lama sejak order terakhir)                                              | Semakin lama customer tidak melakukan transaksi, semakin tinggi risiko churn. Ini adalah indikator recency klasik yang sangat kuat.                                               |
| **CashbackAmount**           | Non-churn memiliki distribusi cashback yang lebih tinggi dan lebih lebar                                               | Cashback berperan sebagai insentif retensi. Customer dengan cashback lebih rendah cenderung lebih mudah churn.                                                                    |

***Ringkasan Insight Utama:***
- Berdasarkan seluruh distribusi KDE:
    - Fitur behavioral dan transactional (Tenure, DaySinceLastOrder, Complain, CashbackAmount) menunjukkan pemisahan distribusi yang jelas antara churn dan non-churn
    - Fitur-fitur tersebut memberikan early warning signal yang kuat terhadap churn
    - Faktor pengalaman pelanggan dan engagement jauh lebih menentukan dibandingkan karakteristik statis

> *Analisis distribusi KDE menunjukkan bahwa pelanggan yang churn cenderung memiliki tenure lebih rendah, jarak waktu lebih lama sejak transaksi terakhir, riwayat komplain, serta menerima insentif cashback yang lebih rendah. Hal ini menegaskan bahwa churn lebih dipengaruhi oleh faktor perilaku dan pengalaman pelanggan dibandingkan karakteristik demografis.*

```python
# Boxplot untuk melihat outliers dan distribusi
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.flatten()

for i, col in enumerate(num_features):
    sns.boxplot(
        x=target,
        y=col,
        color='#006400',  # Warna hijau
        data=df,
        ax=axes[i]
    )
    axes[i].set_title(f'{col}')
    axes[i].set_xlabel('Churn')
    axes[i].set_ylabel(col)

plt.suptitle('Boxplot Distribusi Fitur Numerik terhadap Churn', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# Heatmap korelasi
plt.figure(figsize=(10, 6))
corr_matrix = df[num_features + [target]].corr()  # Hitung korelasi

sns.heatmap(
    corr_matrix,
    annot=True,  # Tampilkan nilai korelasi
    fmt=".2f",   # Format 2 desimal
    cmap="YlGn",  # Warna kuning-hijau
    linewidths=0.5  # Garis pemisah
)

plt.title("Correlation Between Customer Behavior Metrics and Churn")
plt.tight_layout()
plt.show()
```
Boxplot Result:

<img width="1789" height="789" alt="image" src="https://github.com/user-attachments/assets/a83ebeb9-8d3b-4b62-8ebb-59ab5ce486fc" />

**Interpretasi Boxplot Distribusi Fitur Numerik terhadap Churn:**

| Fitur                        | Perbandingan Churn vs Non-Churn                                                 | Pola Outlier                                 | Interpretasi Bisnis                                                                                                              |
| ---------------------------- | ------------------------------------------------------------------------------- | -------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| **Tenure**                   | Customer churn memiliki **median tenure jauh lebih rendah** dibanding non-churn | Outlier tenure tinggi dominan pada non-churn | Churn paling banyak terjadi pada **customer dengan masa aktif awal**. Tenure rendah adalah **indikator churn yang sangat kuat**. |
| **WarehouseToHome**          | Median churn sedikit **lebih tinggi** dibanding non-churn                       | Outlier ekstrem muncul di kedua kelas        | Jarak gudang ke rumah berpotensi memengaruhi pengalaman logistik, namun **bukan faktor utama tunggal** churn.                    |
| **NumberOfDeviceRegistered** | Customer churn cenderung memiliki **jumlah device lebih sedikit**               | Outlier tinggi muncul pada non-churn         | Semakin banyak device terdaftar, semakin tinggi engagement dan **risiko churn lebih rendah**.                                    |
| **SatisfactionScore**        | Customer churn memiliki **skor kepuasan lebih rendah**                          | Distribusi non-churn lebih stabil            | Kepuasan pelanggan berbanding terbalik dengan churn. **Skor rendah → risiko churn meningkat**.                                   |
| **NumberOfAddress**          | Distribusi relatif mirip pada kedua kelas                                       | Outlier alamat tinggi muncul di kedua kelas  | Jumlah alamat tidak menunjukkan pemisahan kelas yang kuat; **pengaruh terhadap churn relatif lemah**.                            |
| **Complain**                 | Churn sangat dominan pada nilai **1 (pernah komplain)**                         | Tidak relevan (binary feature)               | Riwayat komplain merupakan **sinyal churn yang sangat kuat** dan indikator ketidakpuasan langsung.                               |
| **DaySinceLastOrder**        | Customer churn memiliki **median lebih tinggi**                                 | Outlier tinggi muncul pada churn             | Semakin lama customer tidak melakukan transaksi, semakin tinggi **risiko churn (recency effect)**.                               |
| **CashbackAmount**           | Distribusi overlap cukup besar                                                  | Outlier tinggi di kedua kelas                | Cashback tinggi tidak menjamin customer bertahan; **cashback bukan faktor penentu tunggal churn**.                               |

```python
# Correlation Matrix (Numerical Features)
plt.figure(figsize=(10, 6))

corr_matrix = df[num_features + [target]].corr()

sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="YlGn",
    linewidths=0.5
)

plt.title("Correlation Between Customer Behavior Metrics and Churn")
plt.tight_layout()
plt.show()
```
Correlation Matrix Result:

<img width="935" height="590" alt="image" src="https://github.com/user-attachments/assets/316f8991-efaa-40af-8edd-09b37dd67ff3" />

**Interpretasi Correlation Matrix (Customer Behavior and Churn):**
| Fitur                        | Korelasi dengan Churn | Interpretasi                                                                                                                                       |
| ---------------------------- | --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Tenure**                   | **-0.36**             | Korelasi negatif paling kuat. Semakin lama customer aktif, semakin kecil kemungkinan churn. Ini adalah **predictor paling penting secara linear**. |
| **Complain**                 | **+0.26**             | Korelasi positif sedang. Customer yang pernah komplain memiliki kecenderungan churn lebih tinggi.                                                  |
| **DaySinceLastOrder**        | **-0.16**             | Korelasi lemah–sedang. Semakin lama tidak order, kecenderungan churn meningkat (arah hubungan perlu dibaca bersama konteks churn label).           |
| **CashbackAmount**           | **-0.16**             | Korelasi lemah. Cashback tinggi tidak cukup kuat untuk mencegah churn.                                                                             |
| **NumberOfDeviceRegistered** | +0.11                 | Korelasi sangat lemah. Jumlah device bukan faktor dominan secara linear.                                                                           |
| **SatisfactionScore**        | +0.11                 | Korelasi lemah secara linear meskipun secara distribusi berpengaruh.                                                                               |
| **WarehouseToHome**          | +0.07                 | Hampir tidak ada hubungan linear dengan churn.                                                                                                     |
| **NumberOfAddress**          | +0.04                 | Tidak signifikan terhadap churn.                                                                                                                   |

**Tidak Ada Multikolinearitas Tinggi**
- Tidak ada pasangan fitur dengan korelasi > |0.7|
- Artinya aman untuk model linear (Logistic Regression)
- Tidak perlu feature removal karena multikolinearitas

**Korelasi ≠ Kepentingan Model**
Beberapa fitur seperti:
- SatisfactionScore
- DaySinceLastOrder
- CashbackAmount
> *memiliki korelasi linear rendah, namun terbukti penting di model tree-based karena:*
- Hubungan non-linear
- Interaksi antar fitur
- Threshold effect

**Hubungan antar feature**
| Pasangan Fitur                         | Korelasi | Makna                                                                |
| -------------------------------------- | -------- | -------------------------------------------------------------------- |
| **Tenure ↔ CashbackAmount**            | **0.46** | Customer lama cenderung menerima cashback lebih besar                |
| **DaySinceLastOrder ↔ CashbackAmount** | 0.34     | Cashback sering diberikan pada customer yang lama tidak bertransaksi |
| **Tenure ↔ NumberOfAddress**           | 0.22     | Customer lama memiliki lebih banyak alamat                           |


## 🏗️ **3. Data Preparation & Preprocessing**
```python
# Memisahkan fitur dan target
X = df.drop(columns="Churn")    # Semua kolom kecuali Churn
y = df["Churn"]                 # Hanya kolom Churn

print("X shape:", X.shape)
print("y shape:", y.shape)

# Mengelompokkan fitur berdasarkan tipe
onehot_cols = ["MaritalStatus", "PreferedOrderCat"]  # Kategorikal untuk one-hot encoding
binary_cols = ["Complain"]  # Fitur biner (0/1)
num_cols = [
    "Tenure",
    "WarehouseToHome",
    "NumberOfDeviceRegistered",
    "NumberOfAddress",
    "DaySinceLastOrder",
    "CashbackAmount",
    "SatisfactionScore"
]  # Fitur numerik

print("One-hot Columns:", onehot_cols)
print("Binary Columns:", binary_cols)
print("Numerical Columns:", num_cols)

# Membagi data dengan stratifikasi (menjaga proporsi churn)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,         # 20% data testing
    stratify=y,            # Proporsi churn sama di train dan test
    random_state=42)       # Reproducible split

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("\ny_train distribution:\n", y_train.value_counts(normalize=True))
print("\ny_test distribution:\n", y_test.value_counts(normalize=True))

# Cek missing values
print(df.isnull().sum())
```

## ⚙️ **4. Pipeline Construction**
```python
# Pipeline untuk fitur numerik:
# 1. Imputasi nilai missing dengan median
# 2. Standardisasi (mean=0, std=1)
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Pipeline untuk fitur biner:
# 1. Imputasi dengan mode (nilai paling sering)
# 2. Tidak perlu scaling untuk data biner
binary_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent"))
])

# Pipeline untuk fitur kategorikal:
# 1. Imputasi dengan mode
# 2. One-hot encoding (membuat dummy variables)
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))  # Ignore jika ada kategori baru di test
])

# ColumnTransformer: menerapkan pipeline berbeda ke kolom berbeda
preprocess = ColumnTransformer([
    ("num", num_pipeline, num_cols),      # Pipeline numerik ke num_cols
    ("bin", binary_pipeline, binary_cols), # Pipeline biner ke binary_cols
    ("cat", cat_pipeline, onehot_cols)    # Pipeline kategorikal ke onehot_cols
])
```
**Proses diatas:**
- Preprocessing dilakukan menggunakan Pipeline dan ColumnTransformer untuk menangani fitur numerik, biner, dan kategorikal secara terpisah. 
- Missing value diimputasi sesuai karakteristik data, fitur numerik diskalakan, dan fitur kategorikal di-encode menggunakan one-hot encoding. 
- Pendekatan ini memastikan proses preprocessing konsisten, scalable, dan bebas data leakage.

```python
# Transformasi data
X_train_processed = preprocess.fit_transform(X_train)  # Fit + transform training
X_test_processed = preprocess.transform(X_test)        # Hanya transform testing

print("Type:", type(X_train_processed))  # Biasanya sparse matrix atau numpy array
print("X_train shape setelah preprocessing:", X_train_processed.shape)
print("X_test shape setelah preprocessing:", X_test_processed.shape)
```
Result:
Hasil X Test: (789, 17)
Hasil X Train: (3152, 17)

**Interpretasi Hasil Dimensi Data**

| Output       | Artinya                                 |
| ------------ | --------------------------------------- |
| `(3152, 17)` | **Data training setelah preprocessing** |
| `(789, 17)`  | **Data testing setelah preprocessing**  |

- X_train_processed memiliki bentuk (3152, 17) yang merepresentasikan data training setelah preprocessing.
- X_test_processed memiliki bentuk (789, 17) yang merepresentasikan data testing setelah preprocessing.
- Jumlah kolom yang sama (17 fitur) menunjukkan bahwa preprocessing diterapkan secara konsisten pada data training dan testing.

**Interpretasi Struktur Nilai Data**

- Nilai numerik dengan rentang negatif dan desimal menunjukkan bahwa fitur numerik telah melalui proses standardisasi (StandardScaler).
- Nilai biner 0 dan 1 pada kolom tertentu menunjukkan hasil imputasi fitur biner serta one-hot encoding pada fitur kategorikal.
- Tidak terdapat nilai NaN, yang mengonfirmasi bahwa proses imputasi missing value telah berhasil.

> Kesiapan Modeling → Data telah dikonversi ke dalam format NumPy array dan sepenuhnya numerik, sehingga siap digunakan langsung untuk proses training dan evaluasi model machine learning.

```python
# Konversi ke DataFrame untuk inspeksi
X_train_processed_df = pd.DataFrame(
    X_train_processed,
    columns=preprocess.get_feature_names_out(),  # Nama kolom setelah preprocessing
    index=X_train.index  # Pertahankan index asli
)
```
Langkah ini:
- Mengubah array → DataFrame
- Mengembalikan:
    - Nama fitur hasil transformasi
    - Index asli data
- Ini penting:
    - Untuk inspeksi data
    - Untuk feature importance analysis
    - Untuk debugging preprocessing

Dan dilakukan *index=X_train.index*
Ini memastikan:
- Tidak ada fitur “misterius”
- Model benar-benar tahu apa yang dipelajari
- Setelah nya menjaga index asli data
- Manfaat:
    - Mudah trace kembali ke pelanggan asli
- Aman untuk:
    - Error analysis
    - Confusion matrix investigation
    - Audit hasil prediksi

```python
X_test_processed_df = pd.DataFrame(
    X_test_processed,
    columns=preprocess.get_feature_names_out(),
    index=X_test.index
)
```
Ini untuk menjamin:
- Struktur fitur konsisten antara train dan test
- Tidak ada data leakage
- Tidak ada perbedaan jumlah atau urutan kolom

```python
# Cek apakah masih ada missing values
print("Missing values in train:", X_train_processed_df.isnull().sum().sum())
print("Missing values in test:", X_test_processed_df.isnull().sum().sum())
```
Proses data quality validation untuk cek missing value.
- Tujuan untuk memastikan:
    - Imputasi berjalan dengan benar
    - Tidak ada NaN tersisa
- Menghindari:
    - Error saat training
    - Bias model
    - Kegagalan saat cross-validation
- Jika hasil = 0, maka, Dataset siap untuk modeling ✅

```python
# Fungsi untuk mendapatkan nama fitur setelah preprocessing
def get_feature_names(preprocess):
    feature_names = []    # Proses untuk menyiapkan wadah untuk menyusun ulang nama fitur akhir yang akan di gunakan model setelah seluruh preprocessing selesai.
                          # Tujuan nya untuk interpretabilitas & validasi struktur data
    
    # Fitur numerik (tidak berubah nama)
    feature_names.extend(num_cols)    # Process fitur passthrough (tidak di encode, tidak pecah, nama kolom tetap sama sebelum dan sesudah preprocessing).
    
    # Fitur biner (tidak berubah nama)
    feature_names.extend(binary_cols)    # Menambah fitur Biner (Nilai mungkin di-impute atau di scale, namun tidak mengubah struktur kolom).
    
    # Fitur kategorikal setelah one-hot encoding
    # Mengambil dari onehot encoder dalam pipeline
    ohe = preprocess.named_transformers_["cat"] \
                    .named_steps["onehot"] \
                    .get_feature_names_out(onehot_cols) # Extract encode feature names dari pipeline
                                                        # 1. Mengakses ColumnTransformer → "cat"
                                                        # 2. Masuk ke Pipeline → "onehot"
                                                        # 3. Mengambil nama kolom hasil one-hot encoding

    feature_names.extend(ohe)    # Final feature space assembly, untuk menghasilkan urutan fitur persis sama dengan data hasil preprocessing
    
    return feature_names

feature_names = get_feature_names(preprocess)
print("Jumlah fitur:", len(feature_names))    # Process sanity check dimensionality, agar: jumlah fitur sama dengan kolom input model & tidak ada fitur yang hilang/dovel.
                                              # Mencegah mismatch saat interpretasi model.
```

## 🤖 **5. Model Definition & Baseline Evaluation**
```python
# Mendefinisikan semua model yang akan dievaluasi
models = {
    "Logistic Regression": Pipeline([
        ("preprocess", preprocess),  # Preprocessing pipeline
        ("model", LogisticRegression(
            max_iter=1000,           # Maksimum iterasi untuk konvergensi
            class_weight="balanced", # Menyeimbangkan kelas imbalance
            solver="liblinear",      # Algoritma optimization
            random_state=42
        ))
    ]),

    "Decision Tree": Pipeline([
        ("preprocess", preprocess),
        ("model", DecisionTreeClassifier(
            max_depth=None,          # Tidak membatasi depth
            class_weight="balanced", # Menangani imbalance
            random_state=42
        ))
    ]),

    "Random Forest": Pipeline([
        ("preprocess", preprocess),
        ("model", RandomForestClassifier(
            n_estimators=100,        # 100 pohon
            class_weight="balanced",
            random_state=42,
            n_jobs=-1                # Gunakan semua core CPU
        ))
    ]),

    "AdaBoost": Pipeline([
        ("preprocess", preprocess),
        ("model", AdaBoostClassifier(
            n_estimators=100,        # 100 estimator
            learning_rate=1.0,       # Learning rate
            random_state=42
        ))
    ]),

    "Gradient Boosting": Pipeline([
        ("preprocess", preprocess),
        ("model", GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42
        ))
    ]),

    "XGBoost": Pipeline([
        ("preprocess", preprocess),
        ("model", XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,           # 80% data untuk setiap pohon
            colsample_bytree=0.8,    # 80% fitur untuk setiap pohon
            objective="binary:logistic",  # Untuk klasifikasi biner
            eval_metric="logloss",   # Metric evaluasi
            random_state=42,
            n_jobs=-1
        ))
    ])
}
```
Ini untuk mendefinisikan kumpulan pipeline machine learning yang berisi beberapa model klasifikasi untuk membandingkan performa prediksi churn secara adil dan konsisten.
Setiap model menggunakan preprocessing yang sama dan dirancang untuk menangani data churn yang tidak seimbang, sehingga memudahkan benchmarking, pemilihan model terbaik, dan evaluasi yang objektif sebelum dilakukan hyperparameter tuning.

```python
# Stratified K-Fold Cross Validation (5 fold)
skf = StratifiedKFold(
    n_splits=5,          # 5 fold
    shuffle=True,        # Acak data
    random_state=42      # Reproducible
)

# Metrics untuk evaluasi
scoring = {
    "roc_auc": "roc_auc",          # Area Under ROC Curve
    "pr_auc": "average_precision", # Area Under Precision-Recall Curve
    "f1": "f1",                    # F1-Score
    "recall": "recall"             # Recall
}

# Evaluasi baseline model dengan cross-validation
baseline_results = []

for name, model in models.items():
    # Cross-validation untuk setiap model
    cv = cross_validate(
        model,
        X_train,
        y_train,
        cv=skf,           # Stratified K-Fold
        scoring=scoring,  # Multiple metrics
        n_jobs=-1         # Paralel processing
    )
    
    # Kumpulkan hasil rata-rata
    baseline_results.append({
        "Model": name,
        "ROC_AUC": cv["test_roc_auc"].mean(),
        "PR_AUC": cv["test_pr_auc"].mean(),
        "F1": cv["test_f1"].mean(),
        "Recall": cv["test_recall"].mean()
    })

# Konversi ke DataFrame dan urutkan berdasarkan ROC-AUC
baseline_results = pd.DataFrame(baseline_results)
baseline_results.sort_values("ROC_AUC", ascending=False, inplace=True)
print("Baseline Model Performance:")
print(baseline_results)
```
Kode ini digunakan untuk mengevaluasi performa baseline dari setiap model klasifikasi menggunakan Stratified K-Fold Cross-Validation.
Evaluasi dilakukan dengan metrik yang relevan untuk kasus churn yang tidak seimbang (ROC-AUC, PR-AUC, F1, dan Recall), kemudian hasil rata-rata tiap model dirangkum dalam sebuah tabel untuk membandingkan dan menentukan model terbaik sebelum tuning.

Dengan hasil *Model Performance Comparison:*
| No | Model                | ROC_AUC  | PR_AUC  | F1 Score | Recall  |
|----|----------------------|----------|---------|----------|---------|
| 2  | Random Forest        | 0.956938 | 0.851917| 0.742995 | 0.660471 |
| 5  | XGBoost              | 0.949454 | 0.831305| 0.734479 | 0.675424 |
| 4  | Gradient Boosting    | 0.929161 | 0.782368| 0.699655 | 0.630755 |
| 3  | AdaBoost             | 0.901851 | 0.706207| 0.638474 | 0.562080 |
| 0  | Logistic Regression  | 0.881066 | 0.681731| 0.569770 | 0.820007 |
| 1  | Decision Tree        | 0.835550 | 0.587475| 0.733465 | 0.721616 |

Catatan:
- ROC-AUC & PR-AUC digunakan sebagai metrik utama karena dataset churn bersifat imbalanced
- XGBoost dan Random Forest menunjukkan performa paling seimbang
- Logistic Regression memiliki recall tinggi namun dengan trade-off pada precision/F1

**Interpretasi Hasil Evaluasi Model**
| Model                   | Observasi Utama                                                                                                                                                                                                                                                                                                  |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Random Forest**       | Menunjukkan kemampuan diskriminasi terbaik dengan nilai ROC-AUC (0.957) dan PR-AUC (0.852) tertinggi. Model ini memiliki keseimbangan yang baik antara precision dan recall, yang mengindikasikan kemampuan generalisasi yang kuat.                                                                              |
| **XGBoost**             | Memiliki performa yang kompetitif dengan ROC-AUC dan PR-AUC sedikit lebih rendah dibandingkan Random Forest, namun dengan recall yang sedikit lebih tinggi. Hal ini menunjukkan bahwa XGBoost mampu menangkap pola churn secara efektif dan memiliki potensi peningkatan performa melalui hyperparameter tuning. |
| **Gradient Boosting**   | Menunjukkan performa yang solid, namun dengan nilai recall dan PR-AUC yang lebih rendah, sehingga berisiko melewatkan sebagian pelanggan churn dibandingkan dua model teratas.                                                                                                                                   |
| **AdaBoost**            | Memiliki kemampuan diskriminasi yang moderat, tetapi performanya lebih rendah pada recall dan F1-score, sehingga kurang optimal untuk kasus churn di mana kesalahan false negative memiliki dampak bisnis yang besar.                                                                                            |
| **Logistic Regression** | Memiliki nilai recall tertinggi (0.82), yang menunjukkan sensitivitas tinggi dalam mengidentifikasi pelanggan churn. Namun, hal ini dicapai dengan mengorbankan precision dan F1-score, sehingga kualitas klasifikasi secara keseluruhan lebih rendah.                                                           |
| **Decision Tree**       | Menunjukkan recall dan F1-score yang relatif tinggi, tetapi memiliki ROC-AUC dan PR-AUC yang jauh lebih rendah, yang mengindikasikan model kurang stabil dan kemampuan generalisasi yang terbatas pada pembagian data yang berbeda.                                                                              |


## 🔧 **6. Hyperparameter Tuning**
```python
# Membuat pipeline khusus untuk tuning
rf_pipeline = Pipeline([
    ("preprocess", preprocess),
    ("model", RandomForestClassifier(random_state=42))
])

xgb_pipeline = Pipeline([
    ("preprocess", preprocess),
    ("model", XGBClassifier(random_state=42, eval_metric='logloss'))
])

# Grid parameter untuk Random Forest
rf_param_grid = {
    "model__n_estimators": [100, 200, 300],      # Jumlah pohon
    "model__max_depth": [None, 10, 20],         # Kedalaman maksimum
    "model__min_samples_split": [2, 5],         # Min samples untuk split
    "model__min_samples_leaf": [1, 2]           # Min samples di leaf
}

# Grid parameter untuk XGBoost
xgb_param_grid = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [3, 5, 7],              # Kedalaman pohon
    "model__learning_rate": [0.01, 0.05, 0.1],  # Learning rate
    "model__subsample": [0.8, 1.0],             # Proporsi data training
    "model__colsample_bytree": [0.8, 1.0]       # Proporsi fitur
}

# Konfigurasi tuning
tuning_configs = {
    "Random Forest": (rf_pipeline, rf_param_grid),
    "XGBoost": (xgb_pipeline, xgb_param_grid)
}

# Stratified K-Fold untuk tuning
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Dictionary untuk menyimpan model terbaik
best_models = {}

# Tuning untuk setiap model
for name, (pipeline, param_grid) in tuning_configs.items():
    print(f"Running RandomizedSearchCV for {name}...")
    
    # Randomized Search Cross Validation
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        n_iter=20,           # Coba 20 kombinasi parameter acak
        scoring='roc_auc',   # Metric utama untuk optimasi
        cv=skf,              # 5-fold cross-validation
        n_jobs=-1,           # Paralel processing
        verbose=1,           # Tampilkan progress
        random_state=42      # Reproducible
    )
    
    # Fit model dengan data training
    search.fit(X_train, y_train)
    
    print(f"{name} - Best ROC-AUC: {search.best_score_:.4f}")
    print(f"{name} - Best Parameters: {search.best_params_}\n")
    
    # Simpan model terbaik
    best_models[name] = search.best_estimator_
```
Proses ini bertujuan untuk memperoleh model terbaik dengan performa optimal dan evaluasi yang adil melalui hyperparameter tuning terintegrasi dalam pipeline preprocessing. Dengan menggunakan Randomized Search dan Stratified 5-Fold Cross-Validation, setiap kandidat model (Random Forest dan XGBoost) dievaluasi secara konsisten berdasarkan ROC-AUC, sehingga hasil yang diperoleh robust terhadap ketidakseimbangan kelas dan minim risiko data leakage. Model dengan kombinasi hyperparameter terbaik kemudian dipilih dan disimpan sebagai final candidate model yang siap digunakan untuk evaluasi lanjutan, interpretasi, dan pengambilan keputusan bisnis.

## 📊 **7. Model Evaluation on Test Set**
```python
# Evaluasi model terbaik pada data test
for name, model in best_models.items():
    # Prediksi pada test set
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]  # Probabilitas kelas positif
    
    print(f"Evaluation for {name} on Test Set:")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
    print(f"PR-AUC: {average_precision_score(y_test, y_prob):.4f}")
    print(f"F1-score: {f1_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}\n")
```
Tahapan ini merupakan evaluasi performa model terbaik pada data test yang belum pernah dilihat sebelumnya. Setiap model yang telah dipilih melalui hyperparameter tuning diuji dengan:
- Prediksi label (y_pred) untuk mengukur akurasi klasifikasi,
- Prediksi probabilitas (y_prob) untuk menghitung metrik berbasis probabilitas seperti ROC-AUC dan PR-AUC.

Dengan cara ini, kita mendapatkan estimasi performa sebenarnya di dunia nyata, termasuk kemampuan model untuk:
- Membedakan churn vs non-churn (ROC-AUC),
- Menangani ketidakseimbangan kelas (PR-AUC),
- Keseimbangan presisi dan recall (F1-score),
- Menangkap churn yang sebenarnya terjadi (Recall).

> Hasil evaluasi ini menjadi dasar untuk pemilihan model final dan rekomendasi bisnis terkait strategi retensi pelanggan.

**Random Forest vs XGBoost — Overview & interpretasi:**
| Metric   | Random Forest | XGBoost | Insight                                                                                                         |
| -------- | ------------- | ------- | --------------------------------------------------------------------------------------------------------------- |
| ROC-AUC  | 0.9622        | 0.9569  | Keduanya sangat bagus dalam membedakan churn vs non-churn (nilai >0.95 sangat tinggi). RF sedikit lebih tinggi. |
| PR-AUC   | 0.9004        | 0.8901  | PR-AUC fokus pada kelas positif (churn). RF sedikit lebih unggul.                                               |
| F1-score | 0.8083        | 0.8379  | F1 menggabungkan precision & recall. XGBoost lebih seimbang → lebih baik menangani trade-off.                   |
| Recall   | 0.7185        | 0.7852  | Recall penting untuk bisnis: semakin tinggi, semakin sedikit churn yang terlewat. XGBoost lebih unggul.         |

**Interpretasi:**
- ROC-AUC tinggi di kedua model → Model sangat baik memisahkan pelanggan yang churn vs yang aktif.
- PR-AUC tinggi → Model tetap bagus meski dataset imbalance (misal churn lebih sedikit).
- F1-score lebih tinggi di XGBoost → Model ini lebih seimbang antara false positives & false negatives.
- Recall lebih tinggi di XGBoost → Lebih banyak churn yang berhasil ditangkap, ini penting jika tujuan bisnis mencegah churn dengan intervensi (misal promo, outreach).
- Random Forest punya sedikit keunggulan di ROC-AUC, tapi untuk bisnis, recall & F1 lebih penting karena kita ingin mengurangi churn sebanyak mungkin.

## 💾 **8. Model Persistence**
```python
import joblib

# Save model terbaik ke file
joblib.dump(best_models["Random Forest"], "rf_pipeline_best.pkl")
joblib.dump(best_models["XGBoost"], "xgb_pipeline_best.pkl")

# Untuk load model: model = joblib.load("rf_pipeline_best.pkl")
```
Kode ini digunakan untuk menyimpan model yang sudah dilatih ke dalam file agar bisa digunakan kembali tanpa harus melatih ulang.
- joblib.dump(model, "filename.pkl") → menyimpan objek model (pipeline lengkap termasuk preprocessing dan estimator) ke file .pkl.
- Dalam kasus ini:
    - best_models["Random Forest"] disimpan sebagai "rf_pipeline_best.pkl"
    - best_models["XGBoost"] disimpan sebagai "xgb_pipeline_best.pkl"

Tujuannya:
- Memudahkan deploy model ke production
- Bisa memuat kembali model kapan saja dengan joblib.load("filename.pkl")
- Menjamin reproducibility hasil prediksi tanpa perlu retraining.

## 📈 **9. Performance Comparison**
```python
# DataFrame untuk hasil baseline
baseline_results.sort_values("ROC_AUC", ascending=False, inplace=True)

# DataFrame untuk hasil tuned (dari evaluasi test set)
tuned_results = pd.DataFrame([
    {
        "Model": "Random Forest",
        "ROC_AUC": 0.9622,
        "PR_AUC": 0.9004,
        "F1": 0.8083,
        "Recall": 0.7185
    },
    {
        "Model": "XGBoost",
        "ROC_AUC": 0.9569,
        "PR_AUC": 0.8901,
        "F1": 0.8379,
        "Recall": 0.7852
    }
])

# Gabungkan dan bandingkan
comparison_df = pd.merge(
    baseline_results,
    tuned_results,
    on="Model",
    suffixes=("_Baseline", "_Tuned")  # Suffix untuk kolom
)

# Hitung improvement setelah tuning
metrics = ["ROC_AUC", "PR_AUC", "F1", "Recall"]
for metric in metrics:
    comparison_df[f"{metric}_Diff"] = comparison_df[f"{metric}_Tuned"] - comparison_df[f"{metric}_Baseline"]

print("Performance Comparison (Baseline vs Tuned):")
print(comparison_df)
```
Ini digunakan untuk membandingkan performa model sebelum dan setelah hyperparameter tuning. Prosesnya:
- Baseline Results – Hasil evaluasi model awal sebelum tuning (cross-validation).
- Tuned Results – Hasil model terbaik setelah hyperparameter tuning diuji pada data test.
- Merge / Compare – Menggabungkan kedua hasil menjadi satu DataFrame berdasarkan nama model.
- Hitung Improvement – Menambahkan kolom baru yang menunjukkan selisih (perbaikan) performa untuk tiap metrik (ROC-AUC, PR-AUC, F1, Recall).

Tujuan:
- Memudahkan visualisasi seberapa besar peningkatan performa model setelah tuning.
- Membantu memilih model final yang optimal dan stabil untuk prediksi churn.

***Interpretasi hasil perbandingan antara Tuned vs Baseline:***
| Model                        | ROC-AUC   | PR-AUC    | F1        | Recall    | Interpretasi Churn                                  |
| ---------------------------- | --------- | --------- | --------- | --------- | --------------------------------------------------- |
| **Random Forest (Baseline)** | 0.957     | 0.852     | 0.743     | 0.660     | Stabil sejak awal, masih ada churner terlewat       |
| **Random Forest (Tuned)**    | **0.962** | **0.900** | 0.808     | 0.718     | Lebih konservatif, lebih banyak churner tertangkap  |
| **XGBoost (Baseline)**       | 0.949     | 0.831     | 0.734     | 0.675     | Sudah bagus, performa seimbang                      |
| **XGBoost (Tuned)**          | 0.957     | 0.890     | **0.838** | **0.785** | Peningkatan terbesar, optimal untuk churn detection |

***Ringkasan Kedua Model***
| Model                  | Ringkasan Performa                                            | Dampak Hyperparameter Tuning                          | Interpretasi Bisnis                                                     |
| ---------------------- | ------------------------------------------------------------- | ----------------------------------------------------- | ------------------------------------------------------------------------|
| **Random Forest**      | ROC-AUC & PR-AUC tinggi sejak baseline, F1-score moderat      | Peningkatan F1 dan recall sedang (+0.065, +0.058)     | Model stabil sejak awal, peningkatan tuning ada tetapi lebih kecil, Aman untuk **production deployment**, cocok sebagai benchmark      |
| **XGBoost Classifier** | ROC-AUC dan PR-AUC tinggi, F1-score sudah baik sejak baseline | Peningkatan F1 dan recall signifikan (+0.103, +0.110) | Model paling meningkat performanya setelah tuning, terutama F1 & recall, cocok sebagai **primary churn detector**; tuning efektif untuk coverage churn |

**Kesimpulan Utama:**
> Best model yang cocok sebagai *primary churn detector* adalah **XGBooster**.

> Model ini dipilih sebagai alat utama untuk mendeteksi pelanggan yang berpotensi churn.

> Setelah hyperparameter tuning, model ini lebih baik dalam “menangkap” churner (Recall tinggi), artinya jumlah pelanggan yang benar-benar churn dan terdeteksi meningkat.

> Implikasi bisnis:
- Ideal untuk strategi intervensi agresif, misal menawarkan promo atau retention campaign ke pelanggan yang terdeteksi akan churn.
- Cocok jika tujuan bisnis mengurangi churn sebanyak mungkin, walaupun mungkin ada beberapa false positive (pelanggan dikira churn tapi sebenarnya tidak).

## 📊 **10. Model Analysis & Visualization**
```python
# XGBoost dengan parameter tuned
xgb_tuned = Pipeline([
    ("preprocess", preprocess),
    ("model", XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    ))
])

# Fit model tuned
xgb_tuned.fit(X_train, y_train)
```
Ini mendefinisikan dan melatih **model XGBoost final yang sudah dituning** menggunakan **pipeline lengkap**:
1. **Pipeline**
   * `"preprocess"` → semua preprocessing (imputer, scaler, encoding) dijalankan secara otomatis pada data input.
   * `"model"` → XGBClassifier dengan **hyperparameter hasil tuning** (`n_estimators=300`, `max_depth=7`, `learning_rate=0.1`, dll).
2. **Fitting**
   * `xgb_tuned.fit(X_train, y_train)` → model belajar dari **data training mentah**, pipeline memastikan preprocessing dijalankan sebelum training.

**Tujuan:**
- Membuat **model final siap pakai** untuk prediksi churn.
- Hyperparameter sudah dioptimalkan agar **performanya maksimal** (misal F1-score dan recall tinggi pada data tidak seimbang).
- Pipeline memastikan **reproducibility** dan **tidak ada data leakage**.

```python
# Ambil model baseline XGBoost
xgb_baseline = models["XGBoost"]
xgb_baseline.fit(X_train, y_train)

# Fungsi untuk plot ROC dan Precision-Recall curves
def plot_roc_pr(before_model, after_model, X_test, y_test, model_name="XGBoost"):
    # Probabilitas prediksi
    y_proba_before = before_model.predict_proba(X_test)[:,1]
    y_proba_after = after_model.predict_proba(X_test)[:,1]
    
    # ROC Curve
    fpr_before, tpr_before, _ = roc_curve(y_test, y_proba_before)
    fpr_after, tpr_after, _ = roc_curve(y_test, y_proba_after)
    roc_auc_before = auc(fpr_before, tpr_before)
    roc_auc_after = auc(fpr_after, tpr_after)
    
    # Precision-Recall Curve
    precision_before, recall_before, _ = precision_recall_curve(y_test, y_proba_before)
    precision_after, recall_after, _ = precision_recall_curve(y_test, y_proba_after)
    pr_auc_before = average_precision_score(y_test, y_proba_before)
    pr_auc_after = average_precision_score(y_test, y_proba_after)
    
    # Plot
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(12,5))
    
    # ROC Curve
    ax_roc.plot(fpr_before, tpr_before, '-', label=f'Baseline (AUC={roc_auc_before:.3f})', color='red')
    ax_roc.plot(fpr_after, tpr_after, '-', label=f'Tuned (AUC={roc_auc_after:.3f})', color='blue')
    ax_roc.plot([0,1],[0,1], ':', color='gray')  # Diagonal random
    ax_roc.set_title(f'ROC Curve - {model_name}')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.legend()
    ax_roc.grid(True)
    
    # Precision-Recall Curve
    ax_pr.plot(recall_before, precision_before, '-', label=f'Baseline (AP={pr_auc_before:.3f})', color='red')
    ax_pr.plot(recall_after, precision_after, '-', label=f'Tuned (AP={pr_auc_after:.3f})', color='blue')
    ax_pr.set_title(f'Precision–Recall Curve - {model_name}')
    ax_pr.set_xlabel('Recall')
    ax_pr.set_ylabel('Precision')
    ax_pr.legend()
    ax_pr.grid(True)
    
    plt.tight_layout()
    plt.show()

# Plot comparison curves
plot_roc_pr(xgb_baseline, xgb_tuned, X_test, y_test, model_name="XGBoost")
```
Ini digunakan untuk **membandingkan performa model XGBoost sebelum dan setelah tuning** menggunakan **ROC Curve** dan **Precision–Recall Curve**:
1. **`xgb_baseline.fit(X_train, y_train)`** → Melatih model XGBoost awal (baseline) tanpa hyperparameter tuning.
2. **`plot_roc_pr`** → Fungsi yang:
   * Menghitung probabilitas prediksi (`predict_proba`) untuk baseline dan model tuning.
   * Menghitung ROC curve & AUC, serta Precision–Recall curve & Average Precision (AP).
   * Membuat **plot perbandingan** kedua model secara visual (baseline merah, tuned biru).
3. **Tujuan:**
   * Memvisualisasikan **peningkatan performa** setelah tuning.
   * Menilai model secara **komprehensif pada data test** untuk kasus churn yang tidak seimbang.

Hasil grafik nya:

<img width="1189" height="490" alt="image" src="https://github.com/user-attachments/assets/2d314ce9-4254-4c09-9c41-3a57ffeb6fc8" />

***Ringkasan interpretasi gambar kurva:***
- Baseline PR Curve (merah): AP = 0.860
- Tuned PR Curve (biru): AP = 0.890 → lebih baik
- Titik awal kiri (Recall ~1, Precision rendah) → threshold rendah (~0.2–0.3)
- Titik kanan atas (Precision ~1, Recall rendah) → threshold tinggi (~0.8–0.9)

***Kesimpulan:***
- Untuk menangkap lebih banyak churn, gunakan threshold 0.3–0.4 (recall tinggi, precision masih cukup).
- Untuk lebih konservatif, gunakan threshold ~0.5 (default).

## 🎯 **11. Threshold Analysis**
```python
# Analisis threshold yang berbeda untuk prediksi
models = {"XGBoost": xgb_tuned}
thresholds = [0.5, 0.4, 0.3, 0.2]  # Threshold yang akan diuji

for name, model in models.items():
    # Probabilitas prediksi
    y_prob = model.predict_proba(X_test)[:,1]
    
    # Buat figure 2x2 untuk confusion matrices
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, thresh in enumerate(thresholds):
        # Prediksi biner berdasarkan threshold
        y_pred = (y_prob >= thresh).astype(int)
        
        # Hitung metrics
        print(f"===== {name} (Threshold = {thresh}) =====")
        print(f"Accuracy : {accuracy_score(y_test, y_pred):.3f}")
        print(f"Precision: {precision_score(y_test, y_pred):.3f}")
        print(f"Recall   : {recall_score(y_test, y_pred):.3f}")
        print(f"F1 Score : {f1_score(y_test, y_pred):.3f}\n")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(
            cm, annot=True, 
            fmt='d', 
            cmap=sns.light_palette("#006400", as_cmap=True),
            xticklabels=["Active (0)", "Churn (1)"],
            yticklabels=["Active (0)", "Churn (1)"],
            ax=axes[i]
        )
        axes[i].set_title(f"Threshold = {thresh}")
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")
    
    plt.suptitle(f"{name} Confusion Matrices for Different Thresholds", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
```
Ini digunakan untuk **menguji efek perubahan threshold probabilitas** pada **model XGBoost yang sudah dituning** dan menilai performa klasifikasi churn secara lebih fleksibel:
1. **Prediksi probabilitas** (`predict_proba`) dari model XGBoost untuk setiap data test.
2. **Thresholds berbeda** (`0.5, 0.4, 0.3, 0.2`) digunakan untuk mengubah probabilitas menjadi label biner (`0` atau `1`).
3. **Evaluasi metrics** untuk setiap threshold:
   * Accuracy
   * Precision
   * Recall
   * F1-score
4. **Visualisasi Confusion Matrix** di setiap threshold menggunakan heatmap.
5. **Tujuan:**
   * Menentukan threshold optimal untuk **menyeimbangkan precision dan recall**, terutama pada kasus churn yang tidak seimbang.
   * Membantu membuat **keputusan bisnis** tentang strategi retensi berdasarkan risiko churn yang diukur secara probabilistik.

Hasil visual CM threshold:

<img width="1163" height="985" alt="image" src="https://github.com/user-attachments/assets/d1213900-ec3c-4347-aabb-0ca637be27ad" />

**Interpretasi Confusion Matrices dan Metrics**

| Threshold | Accuracy | Precision | Recall | F1 Score | Observasi Utama                                                                                                                                                                                                                                          |
| --------- | -------- | --------- | ------ | -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **0.5**   | 0.948    | 0.898     | 0.785  | 0.838    | Threshold default 0.5 menghasilkan **Precision tinggi**, artinya prediksi Churn yang positif cenderung benar (few false positives), tapi Recall lebih rendah, sehingga **beberapa Churn terlewat (false negatives)**.                                    |
| **0.4**   | 0.944    | 0.864     | 0.800  | 0.831    | Sedikit penurunan Accuracy & Precision, tetapi Recall meningkat. Artinya model mulai **menangkap lebih banyak Churn**, dengan trade-off sedikit lebih banyak false positives.                                                                            |
| **0.3**   | 0.943    | 0.841     | 0.822  | 0.831    | Recall hampir sama dengan Precision yang menurun sedikit. Threshold lebih rendah membuat model lebih “agresif” mendeteksi Churn, menyeimbangkan trade-off antara **false positives & false negatives**.                                                  |
| **0.2**   | 0.940    | 0.810     | 0.852  | 0.830    | Threshold rendah → model menangkap **paling banyak Churn** (Recall tertinggi), tapi Precision turun karena lebih banyak false positives. Cocok jika prioritas adalah **mendeteksi Churn sebanyak mungkin**, meskipun ada risiko salah tandai user aktif. |

***Insight dari Confusion Matrices:***
- True Positives (TP) meningkat saat threshold menurun → lebih banyak pelanggan Churn yang berhasil dideteksi.
- False Positives (FP) meningkat saat threshold menurun → beberapa pelanggan aktif salah dikategorikan sebagai Churn.
- Trade-off: Threshold tinggi → lebih aman (Precision tinggi), tapi risiko kehilangan Churn. Threshold rendah → lebih agresif (Recall tinggi), tapi beberapa aktif user salah terdeteksi.

*Ringkasan Angka per Threshold 0.5 vs 0.4 vs 0.3 vs 0.2 (After Tuning)*
--
**Threshold = 0.5**

|                   | Pred Active | Pred Churn |
| ----------------- | ----------- | ---------- |
| **Actual Active** | TN = 645    | FP = 9     |
| **Actual Churn**  | FN = 37     | TP = 98    |

* **FN tinggi (37)** → banyak churn **tidak terdeteksi**
* Model **terlalu konservatif**
* Cocok hanya jika biaya FP sangat mahal (bukan kasus churn)

**Threshold = 0.4**

|                   | Pred Active | Pred Churn |
| ----------------- | ----------- | ---------- |
| **Actual Active** | TN = 630    | FP = 24    |
| **Actual Churn**  | FN = 26     | TP = 109   |

* FN **turun signifikan** (37 → 26)
* TP naik (98 → 109)
* FP naik, tapi **masih wajar**
* **Trade-off paling seimbang**

**Threshold = 0.3**

|                   | Pred Active | Pred Churn |
| ----------------- | ----------- | ---------- |
| **Actual Active** | TN = 618    | FP = 36    |
| **Actual Churn**  | FN = 20     | TP = 115   |

* FN **paling rendah**
* TP **paling tinggi**
* FP naik cukup signifikan
* Lebih agresif → cocok jika promo murah & automated

**Threshold = 0.2**

|                   | Pred Active | Pred Churn |
| ----------------- | ----------- | ---------- |
| **Actual Active** | TN = 600    | FP = 54    |
| **Actual Churn**  | FN = 16     | TP = 119   |

* FN **terendah (16)** → hampir semua churn terdeteksi
* TP **paling tinggi (119)**
* FP naik signifikan → biaya intervensi lebih besar
* Strategi **sangat agresif**, cocok jika goal adalah **maximal churn capture**

*Fokus Utama: False Negative (FN)*
-
> **FN = customer churn tapi tidak terdeteksi → LOSS**

| Threshold | FN | Perubahan |
| --------- | -- | --------- |
| 0.5       | 37 | baseline  |
| 0.4       | 26 | ↓ **29%** |
| 0.3       | 20 | ↓ **46%** |
| 0.2       | 16 | ↓ **57%** |

**Insight penting:**
Menurunkan threshold secara nyata **mengurangi churn yang “lolos” dari sistem**, tapi FP naik → biaya retensi meningkat.

*Trade-off Bisnis (FP vs FN)*
-

| Threshold | FN (Loss)       | FP (Cost Promo) | Karakter       |
| --------- | --------------- | --------------- | -------------- |
| 0.5       | ❌ Tinggi        | ✅ Rendah        | Terlalu aman   |
| 0.4       | ✅ Rendah        | ⚠️ Sedikit naik | **Balanced**   |
| 0.3       | ✅ Sangat rendah | ❌ Lebih tinggi  | Agresif        |
| 0.2       | ✅ Maksimal      | ❌ Signifikan    | Sangat agresif |


*Kesimpulan Utama*
-

> **Threshold 0.4–0.3 adalah pilihan paling rasional untuk bisnis churn prediction**

**Kenapa bukan 0.5?**

* Terlalu banyak churn tidak terdeteksi → potensi **revenue loss tinggi**

**Kenapa bukan 0.2?**

* FP meningkat drastis → biaya retensi/promo bisa membengkak
* Strategi terlalu agresif untuk sebagian bisnis

**Kenapa threshold 0.4–0.3?**

* Menangkap **lebih banyak churn** → FN turun signifikan
* FP masih dapat dikelola
* Cocok untuk **retention campaign bertarget**
* Threshold 0.4 → lebih seimbang antara biaya & retensi
* Threshold 0.3 → lebih agresif, bisa dipilih jika goal adalah **maximal churn capture** dan biaya tambahan masih bisa ditoleransi

> Analisis ini menunjukkan bahwa threshold **0.4–0.3** memberikan **trade-off optimal** antara menurunkan False Negative dan mengendalikan False Positive untuk strategi prediksi churn di e-commerce.

## 🔑 **12. Feature Importance Analysis**
```python
# Ambil model XGBoost dari pipeline
xgb_model = xgb_tuned.named_steps['model']

# Ambil feature importance (gain)
feature_importance = xgb_model.get_booster().get_score(importance_type='gain')

# Dapatkan nama fitur setelah preprocessing
# 1. Fitur numerik
num_features = num_cols

# 2. Fitur biner
bin_features = binary_cols

# 3. Fitur kategorikal setelah one-hot encoding
cat_pipeline = preprocess.named_transformers_['cat']
ohe = cat_pipeline.named_steps['onehot']
cat_features = ohe.get_feature_names_out(onehot_cols)

# Gabungkan semua nama fitur
feature_names = np.concatenate([num_features, bin_features, cat_features])

# Buat DataFrame feature importance
fi_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': [feature_importance.get(f'f{i}', 0) for i in range(len(feature_names))]
})

# Urutkan berdasarkan importance
fi_df = fi_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)

# Normalisasi importance scores
fi_df['Importance_norm'] = fi_df['Importance'] / fi_df['Importance'].sum()

print("===== Feature Importance =====")
print(fi_df)

# Visualisasi semua feature importance
plt.figure(figsize=(10,6))
sns.barplot(data=fi_df, x='Importance_norm', y='Feature', palette='viridis')
plt.title('XGBoost Feature Importance (Gain)')
plt.xlabel('Importance Score')
plt.ylabel('')
plt.tight_layout()
plt.show()
```
Tahapan ini merupakan evaluasi performa model terbaik pada data test yang belum pernah dilihat sebelumnya. Setiap model yang telah dipilih melalui hyperparameter tuning diuji dengan:
- Prediksi label (y_pred) untuk mengukur akurasi klasifikasi,
- Prediksi probabilitas (y_prob) untuk menghitung metrik berbasis probabilitas seperti ROC-AUC dan PR-AUC.

Dengan cara ini, kita mendapatkan estimasi performa sebenarnya di dunia nyata, termasuk kemampuan model untuk:
- Membedakan churn vs non-churn (ROC-AUC),
- Menangani ketidakseimbangan kelas (PR-AUC),
- Keseimbangan presisi dan recall (F1-score),
- Menangkap churn yang sebenarnya terjadi (Recall).

> Hasil evaluasi ini menjadi dasar untuk pemilihan model final dan rekomendasi bisnis terkait strategi retensi pelanggan.

Hasil Feature importance (Gain):

<img width="989" height="590" alt="image" src="https://github.com/user-attachments/assets/b4efdf65-0774-41ab-b01e-3b2df35775ee" />

**Interpretasi Feature Importance:**
| Rank | Feature                             | Importance | Importance_norm |
| ---- | ----------------------------------- | ---------- | --------------- |
| 1    | Tenure                              | 3.967232   | 0.179215        |
| 2    | Complain                            | 2.303027   | 0.104036        |
| 3    | PreferedOrderCat_Laptop & Accessory | 1.569810   | 0.070914        |
| 4    | PreferedOrderCat_Fashion            | 1.387434   | 0.062676        |
| 5    | MaritalStatus_Single                | 1.307897   | 0.059083        |
| 6    | PreferedOrderCat_Grocery            | 1.262979   | 0.057053        |
| 7    | NumberOfAddress                     | 1.180740   | 0.053338        |
| 8    | DaySinceLastOrder                   | 1.131525   | 0.051115        |
| 9    | MaritalStatus_Married               | 1.095072   | 0.049469        |
| 10   | SatisfactionScore                   | 0.997108   | 0.045043        |
| 11   | PreferedOrderCat_Others             | 0.968589   | 0.043755        |
| 12   | NumberOfDeviceRegistered            | 0.967729   | 0.043716        |
| 13   | CashbackAmount                      | 0.899975   | 0.040655        |
| 14   | MaritalStatus_Divorced              | 0.847012   | 0.038263        |
| 15   | WarehouseToHome                     | 0.794865   | 0.035907        |
| 16   | PreferedOrderCat_Mobile Phone       | 0.783597   | 0.035398        |
| 17   | PreferedOrderCat_Mobile             | 0.672156   | 0.030364        |

```python
# Ambil top 10 features
top10_fi = fi_df.head(10)
print("===== Top 10 Feature Importance =====")
print(top10_fi)

# Visualisasi top 10 dengan warna hijau gradient
import matplotlib.colors as mcolors

# Normalisasi untuk gradient color
norm = mcolors.Normalize(
    vmin=top10_fi['Importance_norm'].min(),
    vmax=top10_fi['Importance_norm'].max()
)

# Generate warna hijau berdasarkan importance
colors = [plt.cm.Greens(norm(v*1.3)) for v in top10_fi['Importance_norm']]

plt.figure(figsize=(8,5))
sns.barplot(
    x=top10_fi['Importance_norm'],
    y=top10_fi['Feature'],
    palette=colors
)

plt.title('Top 10 Feature Importance – XGBoost (Gain)')
plt.xlabel('Normalized Importance Score')
plt.ylabel('')
plt.tight_layout()
plt.show()
```
Top 10 most important features impacting customer churn are visualized using a green gradient bar chart, highlighting key factors for business retention strategy.

<img width="790" height="490" alt="image" src="https://github.com/user-attachments/assets/511b66a8-9976-455f-a568-0812d9445879" />

**Interpretasi Top 10 Importance Features - XGBoost**
| Rank  | Feature                                                                              | Importance (norm) | Pola / Insight Bisnis                                                                               |
| ----- | ------------------------------------------------------------------------------------ | ----------------- | --------------------------------------------------------------------------------------------------- |
| 1     | **Tenure**                                                                           | 0.179             | Customer dengan **Tenure lebih rendah** cenderung churn lebih tinggi → fokus retention early-stage. |
| 2     | **Complain**                                                                         | 0.104             | Adanya **keluhan/complaint** sangat mempengaruhi churn → prioritaskan penyelesaian keluhan cepat.   |
| 3     | **PreferedOrderCat_Laptop & Accessory**                                              | 0.071             | Pelanggan yang sering membeli kategori ini **lebih loyal**, churn lebih rendah.                     |
| 4     | **PreferedOrderCat_Fashion**                                                         | 0.063             | Fashion buyers punya pengaruh signifikan → bisa dibuat promo khusus retention.                      |
| 5     | **MaritalStatus_Single**                                                             | 0.059             | Pelanggan single cenderung lebih berisiko churn → segmentasi campaign khusus.                       |
| 6     | **PreferedOrderCat_Grocery**                                                         | 0.057             | Grocery buyers lebih aktif → potensi retention tinggi jika ditarget.                                |
| 7     | **NumberOfAddress**                                                                  | 0.053             | Banyak alamat terdaftar → loyal, tapi churn bisa lebih tinggi jika alamat tidak valid.              |
| 8     | **DaySinceLastOrder**                                                                | 0.051             | Lama sejak order terakhir → predictor churn → kirim reminder / promo.                               |
| 9     | **MaritalStatus_Married**                                                            | 0.049             | Married customers → sedikit lebih loyal, tapi tidak terlalu signifikan dibanding Tenure.            |
| 10    | **SatisfactionScore**                                                                | 0.045             | Skor kepuasan rendah → risiko churn tinggi → customer survey & retention plan.                      |
| 11–16 | Fitur lain (Others, DeviceRegistered, CashbackAmount, WarehouseToHome, Mobile Phone) | 0.03–0.044        | Masih relevan tapi kontribusi lebih kecil; bisa digunakan untuk fine-tuning strategi campaign.      |

***Insight Utama***
- Tenure & Complain adalah driver utama churn (FN risk tinggi jika tidak ditangani).
- Preferred Order Category (Laptop, Fashion, Grocery) bisa dipakai untuk targeted retention campaign (Misal promo atau voucher khusus kategori ini.)
- Demografi (MaritalStatus) membantu segmentasi campaign: single vs married.
- DaySinceLastOrder & SatisfactionScore → indikator perilaku & pengalaman → bisa digunakan untuk alert sistem churn.
- Fitur lain → tambahan insight, bisa untuk personalized campaign atau prediksi micro-segmen.

***Rekomendasi Bisnis Berdasarkan Feature Importance***
- Fokus awal pada early churners → Tenure rendah + Complaint aktif.
- Segmentasi promo berdasarkan kategori produk → gunakan PreferedOrderCat.
- Behavior-based trigger → kirim promo / reminder berdasarkan DaySinceLastOrder dan SatisfactionScore.
- Demografi untuk targeting → Single lebih berisiko churn → prioritaskan retention.

---
# **Business Insight & Recommendation**
---
## *Key Churn Drivers (Berdasarkan Feature Importance – XGBoost)*

| No | Churn Driver                            | Penjelasan                                                                                                                                                                   | Insight Utama                                                                          |
| -- | --------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| 1  | **Tenure (Durasi Menjadi Customer)**    | Faktor paling dominan dalam prediksi churn. Customer dengan tenure rendah memiliki risiko churn lebih tinggi, menunjukkan kegagalan pada fase early engagement & onboarding. | Customer belum membentuk kebiasaan dan loyalitas terhadap platform.                    |
| 2  | **Complain**                            | Customer yang pernah mengajukan keluhan memiliki risiko churn lebih tinggi, menandakan masalah pada customer service dan pengalaman pengguna.                                | Keluhan yang tidak ditangani dengan baik mempercepat churn.                            |
| 3  | **PreferedOrderCat_Laptop & Accessory** | Pelanggan yang sering membeli kategori ini lebih loyal, sementara yang jarang membeli berisiko churn lebih tinggi.                                                           | Segmentasi campaign berbasis kategori produk meningkatkan retensi.                     |
| 4  | **PreferedOrderCat_Fashion**            | Fashion buyers memiliki pengaruh signifikan terhadap churn.                                                                                                                  | Bisa dibuat promo & reminder spesifik untuk kategori fashion.                          |
| 5  | **MaritalStatus_Single**                | Pelanggan single cenderung lebih berisiko churn dibanding married.                                                                                                           | Segmentasi campaign berdasarkan status marital dapat meningkatkan efektivitas retensi. |
| 6  | **DaySinceLastOrder**                   | Semakin lama sejak order terakhir, semakin tinggi probabilitas churn.                                                                                                        | Indikator “silent churn”, bisa diatasi dengan reactivation reminder.                   |
| 7  | **SatisfactionScore**                   | Skor kepuasan rendah → pelanggan lebih mungkin churn.                                                                                                                        | Monitoring skor kepuasan membantu proaktif mencegah churn.                             |
| 8  | **NumberOfAddress**                     | Banyak alamat terdaftar → loyalitas lebih tinggi, tetapi alamat yang tidak valid bisa memicu churn karena masalah delivery.                                                  | Perlu strategi pengiriman fleksibel dan verifikasi alamat.                             |
| 9  | **CashbackAmount**                      | Cashback rendah atau tidak konsisten cenderung membuat pelanggan merasa kurang dihargai → risiko churn meningkat.                                                            | Personalization promo & cashback meningkatkan perceived value.                         |
| 10 | **WarehouseToHome**                     | Jarak warehouse ke pelanggan jauh → delivery lebih lambat → meningkatkan churn.                                                                                              | Transparansi SLA & free shipping terbatas bisa mengurangi friction.                    |
## *Retention Strategy Recommendations (Data-Driven)*
| No | Strategi Retensi                         | Target Customer                               | Action Plan                                                                                         | Tujuan                                                     |
| -- | ---------------------------------------- | --------------------------------------------- | --------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| 1  | **Early Lifecycle Retention Program**    | Customer dengan tenure rendah                 | Onboarding campaign 7–14 hari pertama, voucher first repeat order, edukasi fitur & manfaat platform | Meningkatkan habit formation sejak awal                    |
| 2  | **Service Recovery Program**             | Customer yang pernah complain                 | Prioritas CS untuk high-risk churn, apology voucher, follow-up pasca penyelesaian keluhan           | Mengubah pengalaman negatif menjadi loyalitas              |
| 3  | **Personalized Category Promo**          | Customer yang jarang membeli kategori favorit | Promo & reminder spesifik kategori (Laptop, Fashion, Grocery)                                       | Meningkatkan engagement dan repeat purchase                |
| 4  | **Reactivation Campaign**                | Customer dengan DaySinceLastOrder tinggi      | Push notification / email reminder, time-limited promo, rekomendasi produk personal                 | Mengaktifkan kembali customer pasif sebelum churn permanen |
| 5  | **Satisfaction Monitoring Program**      | Customer dengan SatisfactionScore rendah      | Survey follow-up, value reminder, loyalty perks                                                     | Menurunkan risiko churn karena pengalaman buruk            |
| 6  | **Cashback & Incentive Personalization** | Customer dengan cashback rendah               | Cashback berbasis histori belanja, dynamic promo, reminder sesuai preferensi produk                 | Meningkatkan perceived value tanpa over-spending           |
| 7  | **Logistic-Sensitive Strategy**          | Customer dengan jarak warehouse jauh          | Free shipping area tertentu, SLA delivery transparan, alternative fulfillment                       | Mengurangi friction pada proses pengiriman                 |

## *Early Warning System – XGBoost Variable–Driven Action Mapping*
*Sebagai tambahan inisght berdasarkan actual business cases pada industri e-commerce*

| Variable                      | Risk Signal (Early Warning)          | Business Interpretation              | Strategic Action                                                     | Rationale (Consultant View)                                 | Expected Business Impact                   |
| ----------------------------- | ------------------------------------ | ------------------------------------ | -------------------------------------------------------------------- | ----------------------------------------------------------- | ------------------------------------------ |
| **Tenure**                    | Tenure rendah (< 30 hari)            | Customer belum membentuk habit       | Early lifecycle onboarding, feature education, first repeat reminder | Churn awal lebih dipicu *lack of engagement*, bukan harga   | Meningkatkan early repeat & lifetime value |
| **DaySinceLastOrder**         | Tidak order > threshold              | Awareness & intent mulai hilang      | Reactivation reminder + urgency message                              | Churn adalah proses gradual, bukan event                    | Menghentikan drift menuju churn permanen   |
| **CashbackAmount**            | Cashback rendah / tidak pernah pakai | Perceived value rendah               | Personalized cashback berbasis histori                               | Insentif kecil tapi relevan lebih efektif dari diskon besar | Efisiensi biaya promo                      |
| **Complain**                  | Pernah komplain                      | Trust rusak, emosi negatif           | Service recovery + reassurance + kompensasi terbatas                 | Emosional churn tidak bisa diselesaikan dengan promo saja   | Recovery trust & retention                 |
| **WarehouseToHome**           | Jarak warehouse jauh                 | Friksi operasional, delivery risk    | Free shipping terbatas / SLA transparan                              | Customer churn karena effort, bukan price                   | Mengurangi friction non-price              |
| **NumberOfDeviceRegistered**  | Banyak device                        | Multi-touch user, higher expectation | Konsistensi experience & cross-device messaging                      | Experience inconsistency meningkatkan churn                 | Stabilitas engagement                      |
| **SatisfactionScore**         | Skor rendah                          | Early dissatisfaction                | Preventive outreach + value reminder                                 | Early dissatisfaction adalah leading indicator churn        | Menurunkan complaint-driven churn          |
| **NumberOfAddress**           | Banyak alamat                        | Mobilitas tinggi                     | Flexible delivery & pickup options                                   | Logistic rigidity meningkatkan drop-off                     | Adaptasi ke customer behavior              |
| **Churn Probability (Model)** | ≥ threshold (ex: 0.4)                | Risiko churn terkonfirmasi           | Escalation ke targeted intervention                                  | Model menggabungkan semua signal                            | Prioritas resource lebih akurat            |

## **Catatan (Penutup)**
Secara keseluruhan, hasil analisis ini menegaskan bahwa **churn bukan semata-mata dipicu oleh satu faktor**, melainkan merupakan akumulasi dari **pfase awal customer lifecycle yang lemah, penurunan engagement (silent churn), friksi pada layanan, nilai promo yang tidak tepat sasaran, serta kendala logistik**. Model XGBoost berhasil memvalidasi pola ini secara kuantitatif dan sejalan dengan realitas bisnis e-commerce.

Insight yang paling krusial adalah bahwa **fase awal pelanggan (Tenure rendah) dan periode sebelum churn** merupakan window of opportunity paling strategis untuk intervensi. Oleh karena itu, strategi retensi yang efektif **harus bersifat proaktif, tersegmentasi, dan berbasis perilaku pelanggan**, bukan generik.

Implementasi rekomendasi ini diharapkan dapat:
- Menurunkan churn rate secara signifikan.
- Meningkatkan Customer Lifetime Value (CLV) dengan menahan pelanggan berisiko tinggi.
- Mengoptimalkan biaya promo melalui insentif yang lebih tepat sasaran.
- Memperkuat fondasi loyalitas jangka panjang melalui pengalaman pelanggan yang konsisten dan personal.

*Logika Strategis*
-
- Setiap variabel penting (Tenure, Complain, DaySinceLastOrder, SatisfactionScore, kategori produk, dll.) bukan sekadar fitur model, tetapi indikator kegagalan yang nyata dalam pengalaman pelanggan.
- Titik kegagalan ini mencakup:
    - Engagement awal (Tenure)
    - Pengalaman negatif & keluhan (Complain, SatisfactionScore)
    - Perilaku transaksional (DaySinceLastOrder, PreferedOrderCat)
    - Friksi operasional & logistik (WarehouseToHome, NumberOfAddress)
- Tindakan retensi harus selaras dengan titik kegagalan ini → intervensi yang tidak sesuai akan menghasilkan pemborosan biaya dan kehilangan efektivitas.

*Nilai strategis yang dihasilkan* 
-
Dengan mapping ini, perusahaan ecommerce dapat:
- Menghindari penggunaan promo massal (blanket promotion) yangv tidak efisien.
- Menurunkan biaya churn secara lebih terkontrol.
- Menjadikan model churn sebagai **mesin pengambilan keputusan operasional**, bukan sekadar output analitik.
- Memprioritaskan pelanggan dengan resiko tinggi untuk **intervensi yang lebih terukur**.

*Pengembangan Selanjutnya*
-
Hasil ini membuka peluang untuk:
- Churn Risk Scoring → memberi setiap pelanggan skor risiko churn untuk prioritas retensi.
- Trigger-based Campaign Automation → notifikasi / promo otomatis untuk pelanggan berisiko.
- A/B Testing Retention Strategies → uji efektivitas intervensi berbasis variabel spesifik.
- Predictive Segmentation → menyesuaikan strategi retention berdasarkan kategori produk, demografi, dan perilaku pelanggan.
- Monitoring KPI Retensi → integrasi ke dashboard real-time untuk mengukur dampak intervensi secara kuantitatif.

*Kesimpulan Strategis*
- 
> Analisis ini menegaskan bahwa retensi pelanggan efektif membutuhkan kombinasi pendekatan behavior-driven dan data-driven, fokus pada early engagement, deteksi risiko dini, pengalaman pelanggan, serta mitigasi friksi operasional. Model XGBoost memberikan kerangka kerja untuk memprioritaskan tindakan bisnis yang tepat, sehingga intervensi retensi menjadi lebih efisien, terukur, dan berkelanjutan.

## **Kesimpulan dari Proses & Analisis**
### **Alur Kerja Utama:**
1. **EDA & Data Understanding** - Memahami distribusi data dan hubungan antar fitur
2. **Preprocessing** - Menangani missing values, encoding, dan scaling
3. **Baseline Modeling** - Membandingkan beberapa algoritma dasar
4. **Hyperparameter Tuning** - Optimasi parameter untuk model terpilih
5. **Evaluation** - Evaluasi menyeluruh dengan berbagai metrics
6. **Analysis** - Analisis feature importance dan threshold optimization

### **Best Practices yang Diimplementasikan:**
1. **Pipeline** - Menggabungkan preprocessing dan modeling
2. **Cross-Validation** - Evaluasi robust dengan stratified k-fold
3. **Class Imbalance Handling** - Menggunakan class_weight dan metrics yang tepat
4. **Reproducibility** - Random state konsisten di semua step
5. **Model Persistence** - Menyimpan model untuk deployment

### **Key Insights dari Code:**
- XGBoost menunjukkan performa terbaik untuk masalah churn prediction
- Feature importance menunjukkan tenure sebagai faktor paling penting
- Threshold tuning dapat menyeimbangkan precision dan recall sesuai kebutuhan bisnis
- Pipeline memastikan konsistensi preprocessing antara training dan inference

---

## *Progress Storaging*

| Item                 | Link / Reference                                                                                                                |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| Visual Code          | [[Visual Code] Ecommerce Customer Churn Prediction](https://drive.google.com/file/d/1djhT5kg2vyGRXK3Uj-rlnuf-PDY0ME4s/view?usp=sharing)|
| Dashboard            | [Looker Studio Ecommerce Churn](https://lookerstudio.google.com/reporting/7b7052ad-08ae-4ee9-9c05-0c6c820e9052/page/TlJ0C/edit) |
| GitHub Repository    | [Machine_Learning_Categorical_Supervised_for_Ecommerce_Customer_Churn_Prediction](https://github.com/armunbue/Machine-Learning-Categorical-Supervised-for-Ecommerce-Customer-Churn-Prediction/tree/main)|
| Presentation (PPT)   | [PPT Ecommerce Churn Prediction](https://docs.google.com/presentation/d/1EWQ3_182yuFrFVrkyL_6dCgQmqYKTmeJ/edit?usp=sharing&ouid=102760215644664585607&rtpof=true&sd=true)            |
| Google Drive Storage | [Capstone Project 3 ML Storage](https://drive.google.com/drive/folders/1-UT9YRnpPjpkZWp5AFFpwvkuwsp4VSre)                          |
