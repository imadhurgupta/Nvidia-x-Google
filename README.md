<div align="center">

<img src="https://img.shields.io/badge/NVIDIA-76B900?style=for-the-badge&logo=nvidia&logoColor=white" alt="NVIDIA"/> <img src="https://img.shields.io/badge/Google_Cloud-4285F4?style=for-the-badge&logo=google-cloud&logoColor=white" alt="Google Cloud"/> <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/> <img src="https://img.shields.io/badge/RAPIDS-84BD00?style=for-the-badge&logo=nvidia&logoColor=white" alt="RAPIDS"/> <img src="https://img.shields.io/badge/XGBoost-004B87?style=for-the-badge" alt="XGBoost"/>

# 🚀 Accelerated Data Science with Google Cloud & NVIDIA

### GPU-Powered Machine Learning Pipelines using RAPIDS cuDF, cuML, and XGBoost

*Notebook Authors: **Will Hill**, **Jeff Nelson*** · *Platform: Google Colab Enterprise + NVIDIA L4 GPU*

*AI/ML Engineer: **Madhur Gupta** · B.Tech 2026 · [GitHub](https://github.com/imadhurgupta) · [LinkedIn](https://www.linkedin.com/in/imadhurgupta)*

---

[![Open in BQ Studio](https://img.shields.io/badge/Open_in-BigQuery_Studio-4285F4?style=flat-square&logo=google-cloud)](https://console.cloud.google.com/bigquery/import?url=https://github.com/GoogleCloudPlatform/ai-ml-recipes/blob/main/notebooks/regression/gpu_accelerated_regression/gpu_accelerated_regression.ipynb)
[![Open in Colab Enterprise](https://img.shields.io/badge/Open_in-Colab_Enterprise-F9AB00?style=flat-square&logo=google-colab)](https://console.cloud.google.com/vertex-ai/colab/import/https:%2F%2Fraw.githubusercontent.com%2FGoogleCloudPlatform%2Fai-ml-recipes%2Fmain%2Fnotebooks/regression/gpu_accelerated_regression/gpu_accelerated_regression.ipynb)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=flat-square)](https://www.apache.org/licenses/LICENSE-2.0)
[![CUDA](https://img.shields.io/badge/CUDA-13.0-76B900?style=flat-square&logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [GPU Environment](#-gpu-environment)
- [Dataset](#-dataset)
- [Pipeline Breakdown](#-pipeline-breakdown)
  - [1. Data Loading & Cleaning](#1-data-loading--cleaning)
  - [2. Feature Engineering](#2-feature-engineering)
  - [3. Model Training](#3-model-training)
  - [4. Ensemble Evaluation](#4-ensemble-evaluation)
- [CPU vs. GPU Benchmark](#-cpu-vs-gpu-benchmark)
- [Model Performance Results](#-model-performance-results)
- [Profiling & Optimization](#-profiling--optimization)
- [Best Practices](#-best-practices)
- [Setup & Requirements](#-setup--requirements)
- [Services & Costs](#-services--costs)
- [Reference Documentation](#-reference-documentation)

---

## 🌐 Overview

This project demonstrates how to dramatically accelerate end-to-end data science and machine learning workflows on large datasets using **NVIDIA CUDA-X for Data Science** on **Google Cloud's Colab Enterprise** platform.

The core workflow builds a complete ML pipeline to **predict NYC taxi tip amounts** across ~2.3 million clean records. The key insight: by adding just two magic commands to a standard Jupyter notebook, you can unlock full GPU acceleration — **zero code changes required**.

```python
%load_ext cudf.pandas   # ← GPU-accelerates your entire pandas workflow
%load_ext cuml.accel    # ← GPU-accelerates scikit-learn models
```

### What's demonstrated:

| Goal | Technology |
|---|---|
| Accelerate DataFrame operations | **NVIDIA cuDF** (GPU pandas) |
| Accelerate ML model training | **NVIDIA cuML** (GPU scikit-learn) |
| Train gradient-boosted trees on GPU | **XGBoost** with `device='cuda'` |
| Profile CPU vs. GPU execution | `%%cudf.pandas.line_profile` |
| Build an ensemble model | Weighted Linear Stacking |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   Google Colab Enterprise                        │
│                 (GPU-Enabled Runtime: NVIDIA L4)                 │
├──────────────────────────┬──────────────────────────────────────┤
│       CPU (Host)         │         GPU (Device)                 │
│                          │                                       │
│  Standard Python Logic   │  ┌─────────────────────────────┐    │
│  Control Flow            │  │  cuDF  - DataFrame Engine    │    │
│  I/O Operations          │  │  cuML  - ML Algorithm Engine │    │
│  Custom .apply() funcs   │  │  XGBoost (hist / cuda)       │    │
│                          │  └─────────────────────────────┘    │
│                          │                                       │
│                          │  ✅ Vectorized Ops (GPU CUDA Kernels) │
│                          │  ✅ GroupBy / Merge / Sort            │
│                          │  ✅ Linear Regression (cuML)          │
│                          │  ✅ Random Forest (cuML)              │
│                          │  ✅ XGBoost tree building             │
└──────────────────────────┴──────────────────────────────────────┘
           ↕ cuDF.pandas auto-fallback for unsupported ops
```

### Zero-Code-Change Acceleration Flow

```
Your existing pandas code
        ↓
 %load_ext cudf.pandas
        ↓
   cuDF intercepts calls
   ┌────────────────────┐
   │  GPU supported?    │
   │  YES → CUDA Kernel │  ← 10–100x faster
   │  NO  → CPU pandas  │  ← Seamless fallback
   └────────────────────┘
        ↓
  Same output, same API
```

---

## 🖥️ GPU Environment

The notebook is designed to run on **Colab Enterprise** with a GPU-enabled runtime.

```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.159.03       Driver Version: 580.159.03     CUDA Version: 13.0          |
+-------------------------------------------+------------------------+--------------------+
| GPU  Name               Persistence-M     | Bus-Id         Disp.A  | Volatile Uncorr.   |
| Fan  Temp   Perf        Pwr:Usage/Cap     |         Memory-Usage   | GPU-Util Compute M.|
|===========================================+========================+====================|
|   0  NVIDIA L4                   Off      | 00000000:00:03.0 Off   |                0   |
|  N/A  55C    P8           14W /  72W      |     0MiB / 23034MiB    |    0%   Default    |
+-------------------------------------------+------------------------+--------------------+
```

**Verify your GPU** by running in a notebook cell:
```bash
!nvidia-smi
```

> **Supported GPU Runtimes:** T4, L4, A100, and any CUDA-capable Colab Enterprise instance.

---

## 📊 Dataset

**NYC Taxi & Limousine Commission (TLC) Trip Record Data — Year 2024**

> Source: [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) — Publicly available Parquet format

| Property | Value |
|---|---|
| **Format** | Apache Parquet (columnar, efficient) |
| **Scope** | All 12 months of 2024 Yellow Taxi data |
| **Raw Records (Jan)** | **2,964,624 trips** |
| **Clean Records (Jan)** | **2,298,343 trips** |
| **Target Variable** | `tip_amount` (in USD) |

### Schema (Key Features)

| Column | Type | Description |
|---|---|---|
| `tpep_pickup_datetime` | timestamp | Trip start timestamp |
| `tpep_dropoff_datetime` | timestamp | Trip end timestamp |
| `trip_distance` | float32 | Distance in miles |
| `fare_amount` | float32 | Base fare in USD |
| `tip_amount` | float32 | **Target variable** — tip in USD |
| `payment_type` | int32 | `1` = Credit Card (tips recorded) |
| `PULocationID` | int32 | Pickup taxi zone ID |
| `DOLocationID` | int32 | Dropoff taxi zone ID |
| `passenger_count` | int32 | Number of passengers |

### Data Cleaning Filters

```python
df = df[
    (df['fare_amount'] > 0)   & (df['fare_amount'] < 500)  &  # Outlier removal
    (df['trip_distance'] > 0) & (df['trip_distance'] < 100) & # Invalid trips
    (df['tip_amount'] >= 0)   & (df['tip_amount'] < 100)   &  # Unrealistic tips
    (df['payment_type'] == 1)                                   # Credit card only
]
```

**Result:** ~22% of records removed (anomalies, cash payments, sensor errors).

---

## ⚙️ Pipeline Breakdown

### 1. Data Loading & Cleaning

```python
# cuDF.pandas is active — this reads Parquet directly into GPU memory
import time, glob
start = time.perf_counter()

df = pd.concat(
    [pd.read_parquet(f) for f in glob.glob("nyc_taxi_data/*-01.parquet")],
    ignore_index=True
)

# Memory optimization — reduce VRAM footprint by 50%
df[df.select_dtypes('float64').columns] = df.select_dtypes('float64').astype('float32')
df[df.select_dtypes('int64').columns]   = df.select_dtypes('int64').astype('int32')

print(f"Load & Clean Time: {time.perf_counter() - start:.2f}s")
# Output → Load and Clean Time: 3.35 seconds  (on GPU)
```

### 2. Feature Engineering

15 features were engineered from the raw data, all executed as GPU CUDA kernels:

```python
# ── Time Features ─────────────────────────────────────────────────────
df['hour']        = df['tpep_pickup_datetime'].dt.hour
df['dow']         = df['tpep_pickup_datetime'].dt.dayofweek
df['is_weekend']  = (df['dow'] >= 5).astype(int)
df['is_rush_hour']= (
    ((df['hour'] >= 7)  & (df['hour'] <= 9)) |
    ((df['hour'] >= 17) & (df['hour'] <= 19))
).astype(int)

# ── Fare Features ──────────────────────────────────────────────────────
df['fare_log']      = np.log1p(df['fare_amount'])       # Log-transform for skew
df['fare_decimal']  = (df['fare_amount'] % 1 * 100).astype(int)
df['is_round_fare'] = (df['fare_amount'] % 5 == 0).astype(int)

# ── Route Features ─────────────────────────────────────────────────────
df['route_id']        = df['PULocationID'].astype(str) + '_' + df['DOLocationID'].astype(str)
df['route_frequency'] = df['route_id'].map(df['route_id'].value_counts())

# ── Location Aggregation ───────────────────────────────────────────────
pu_stats = df.groupby('PULocationID')['tip_amount'].agg(['mean','std']).reset_index()
pu_stats.columns = ['PULocationID', 'pu_tip_mean', 'pu_tip_std']
df = df.merge(pu_stats, on='PULocationID', how='left')
```

> **GPU Time: 0.77 seconds** for all feature engineering across 2.3M records.

### 3. Model Training

#### XGBoost — GPU Gradient Boosting

```python
import xgboost as xgb

params = {
    'objective':        'reg:squarederror',
    'eval_metric':      'rmse',
    'max_depth':        5,
    'learning_rate':    0.1,
    'device':           'cuda',             # ← GPU Acceleration
    'tree_method':      'hist',             # GPU histogram binning
    'max_bin':          256,
    'subsample':        0.8,
    'colsample_bytree': 0.8,
    'sampling_method':  'gradient_based',   # GPU-optimized sampling
    'verbosity':        0
}

model = xgb.train(
    params, dtrain,
    num_boost_round=1000,
    evals=[(dval, 'val')],
    early_stopping_rounds=50
)
# Training Time: 12.84 seconds (3-fold CV, 2.3M rows)
```

#### Linear Regression — GPU-Accelerated via cuML

```python
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Transparently runs on GPU via cuml.accel
scaler = StandardScaler()
model  = LinearRegression()

X_scaled = scaler.fit_transform(X_train)
model.fit(X_scaled, y_train)
# Training Time: 20.13 seconds (3-fold CV, 2.3M rows)
```

#### Random Forest — GPU-Accelerated via cuML

```python
from sklearn.ensemble import RandomForestRegressor

# Runs on GPU via cuml.accel (100 trees, depth 10)
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    n_jobs=-1,
    max_features='sqrt',
    random_state=42
)
model.fit(X_train, y_train)
# Training Time: 32.08 seconds (3-fold CV, 2.3M rows)
```

### 4. Ensemble Evaluation

A **Linear Stacking Ensemble** was created by fitting a constrained linear regression on the out-of-fold predictions from all three models.

```python
from sklearn.linear_model import LinearRegression

# Learn optimal model blend weights (constrained positive, no intercept)
meta_learner = LinearRegression(positive=True, fit_intercept=False)
meta_learner.fit(np.c_[xgb_preds, rf_preds, linreg_preds], y)

# Normalize weights to sum to 1.0
ensemble_weights = meta_learner.coef_ / meta_learner.coef_.sum()

ensemble_preds = np.c_[xgb_preds, rf_preds, linreg_preds] @ ensemble_weights
```

---

## 📈 CPU vs. GPU Benchmark

A full end-to-end comparison was performed by running the identical pipeline once with standard CPU pandas/scikit-learn, and once with RAPIDS GPU acceleration.

### Timing Results

```
Pipeline Stage          CPU Time    GPU Time    Speedup
──────────────────────────────────────────────────────
Load Data               18.50s       3.15s       5.9x ⚡
Clean Data               5.20s       0.20s      26.0x ⚡
Feature Engineering      2.10s       0.15s      14.0x ⚡
Train Random Forest     45.40s      12.10s       3.8x ⚡
Train XGBoost            9.27s       1.28s       7.2x ⚡
──────────────────────────────────────────────────────
TOTAL PIPELINE          80.47s      16.88s       4.8x ⚡
```

### Relative Speedup Chart

```
Data Cleaning       ████████████████████████████  26.0x
Feature Eng.        ██████████████               14.0x
Train XGBoost       ███████                       7.2x
Load Data           █████▌                        5.9x
Total Pipeline      ████▊                         4.8x
Train Rand.Forest   ███▊                          3.8x
                    0   5   10   15   20   25   30
```

> **Key Insight:** Data cleaning saw the highest speedup (26x) because it involves massively parallelizable boolean masking and type conversion operations perfectly suited for GPU SIMD cores.

---

## 🏆 Model Performance Results

All models were evaluated using 3-Fold Cross-Validation on 2,298,343 clean records.

```
╔══════════════════════════════════════════════╗
║         Model Comparison Summary             ║
╠══════════════════════════╦═══════════════════╣
║ Model                    ║ RMSE              ║
╠══════════════════════════╬═══════════════════╣
║ Linear Regression        ║ $2.4319           ║
║ Random Forest            ║ $2.3142           ║
║ XGBoost                  ║ $2.2810           ║
╠══════════════════════════╬═══════════════════╣
║ Ensemble (Stacked)       ║ $2.2796  ★ Best   ║
╚══════════════════════════╩═══════════════════╝
  Ensemble Lift over XGBoost: $0.0014
```

### Feature Set (15 Features)

| Feature | Category | Importance |
|---|---|---|
| `pu_tip_mean` | Location Stats | 🔴 High |
| `fare_amount` | Trip Fare | 🔴 High |
| `fare_log` | Derived Fare | 🔴 High |
| `trip_distance` | Trip Metadata | 🟡 Medium |
| `PULocationID` | Location | 🟡 Medium |
| `DOLocationID` | Location | 🟡 Medium |
| `route_frequency` | Route Stats | 🟡 Medium |
| `pu_tip_std` | Location Stats | 🟡 Medium |
| `hour` | Time | 🟢 Low-Med |
| `dow` | Time | 🟢 Low-Med |
| `is_rush_hour` | Time | 🟢 Low |
| `is_weekend` | Time | 🟢 Low |
| `fare_decimal` | Derived Fare | 🟢 Low |
| `is_round_fare` | Derived Fare | 🟢 Low |
| `passenger_count` | Trip Metadata | 🟢 Low |

---

## 🔍 Profiling & Optimization

### Line-Level Profiling

Use the `%%cudf.pandas.line_profile` magic to inspect which lines run on the GPU vs. fall back to the CPU:

```python
%%cudf.pandas.line_profile
import pandas as pd, glob

df = pd.concat([pd.read_parquet(f) for f in glob.glob("nyc_taxi_data/*-01.parquet")], ignore_index=True)
df = df.sample(1_000)

df['hour'] = df['tpep_pickup_datetime'].dt.hour     # ← GPU: 0.004s
df['time_of_day'] = df['hour'].apply(my_func)       # ← CPU: 0.001s (fallback!)
```

**Sample output:**
```
┏━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ Line no. ┃ Line                            ┃ GPU TIME(s) ┃ CPU TIME(s) ┃
┡━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│ 5        │ df = pd.concat(...)             │ 0.076565    │             │
│ 6        │ df = df.sample(1_000)           │ 0.289549    │             │
│ 15       │ df['hour'] = df[...].dt.hour    │ 0.004409    │             │
│ 17       │ df['time_of_day'] = df.apply()  │             │ 0.001320    │ ← CPU!
└──────────┴─────────────────────────────────┴─────────────┴─────────────┘
```

### Eliminate CPU Fallbacks

> ❌ **Slow Pattern (forces CPU):**
> ```python
> def categorize_hour(hour):
>     return 'Morning' if hour < 12 else 'Afternoon/Evening'
>
> df['time_of_day'] = df['hour'].apply(categorize_hour)  # Iterates each row on CPU
> ```
>
> ✅ **Fast Pattern (stays on GPU):**
> ```python
> df['time_of_day'] = pd.cut(
>     df['hour'],
>     bins=[-1, 11, 24],
>     labels=['Morning', 'Afternoon/Evening']  # Vectorized CUDA kernel
> )
> ```

---

## 💡 Best Practices

### 1. Always Vectorize Operations

| Pattern | Performance |
|---|---|
| `.apply(lambda x: ...)` | 🔴 CPU-bound, slow |
| `.iterrows()` / `for` loops | 🔴 CPU-bound, very slow |
| `np.select()` / `np.where()` | ✅ GPU vectorized |
| `pd.cut()` / `pd.get_dummies()` | ✅ GPU vectorized |
| `.str` / `.dt` accessors | ✅ GPU vectorized |
| `.groupby().agg()` | ✅ GPU vectorized |

### 2. Manage VRAM Like a Pro

```python
import gc

# After training, free GPU memory explicitly
del X_train, X_val, df
gc.collect()

# Downcast dtypes to halve memory usage
df[float_cols] = df[float_cols].astype('float32')  # 64→32 bit = 50% savings
df[int_cols]   = df[int_cols].astype('int32')
```

### 3. XGBoost GPU Configuration

```python
params = {
    'device':          'cuda',   # Enable GPU
    'tree_method':     'hist',   # GPU histogram — fastest option
    'max_bin':         256,      # Histogram bins (higher = more precise)
    'sampling_method': 'gradient_based',  # GPU-optimized sampling
    'subsample':       0.8,      # Fraction of rows per tree
    'colsample_bytree': 0.8,     # Fraction of features per tree
}
```

### 4. Environment Safety Check

```python
import sys

def check_rapids():
    extensions = ['cudf.pandas', 'cuml.accel']
    for ext in extensions:
        try:
            get_ipython().run_line_magic('load_ext', ext)
            print(f"✅ {ext} loaded successfully")
        except Exception as e:
            print(f"⚠️  {ext} unavailable — using CPU fallback")

check_rapids()
```

---

## 🛠️ Setup & Requirements

### Prerequisites

- **Google Cloud Project** with Colab Enterprise access
- **GPU-Enabled Runtime** (T4 / L4 / A100 recommended)
- **Python** ≥ 3.10

### Required Libraries

```
rapids-colab        # RAPIDS ecosystem
cudf               # GPU DataFrames
cuml               # GPU ML algorithms
xgboost            # GPU gradient boosting
scikit-learn       # ML utilities
pandas             # CPU pandas (fallback)
numpy              # Numerical computing
matplotlib         # Visualization
tqdm               # Progress bars
requests           # HTTP data download
pyarrow            # Parquet file reading
```

### Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/GoogleCloudPlatform/ai-ml-recipes.git

# 2. Open in Colab Enterprise (GPU runtime required)
#    Navigate to: Accelerated Machine Learning/gpu_accelerated_regression.ipynb

# 3. Verify GPU availability
!nvidia-smi

# 4. Run all cells in order
```

---

## 💰 Services & Costs

This project uses the following billable Google Cloud components:

| Service | Purpose | Pricing |
|---|---|---|
| **Colab Enterprise** | GPU-enabled notebook runtime | [Colab Pricing](https://cloud.google.com/colab/pricing) |
| **Google Cloud Storage** | Parquet dataset storage (optional) | [GCS Pricing](https://cloud.google.com/storage/pricing) |

> Use the [Google Cloud Pricing Calculator](https://cloud.google.com/products/calculator) to estimate costs for your specific usage.

### Cost Optimization Tips
- **Stop the runtime** when not actively using the notebook — GPU runtimes bill per minute.
- Use the notebook's **cleanup cell** to delete the NYC taxi dataset after the session (`!rm -rf nyc_taxi_data`).
- Start with **January data only** (the notebook default) before experimenting with the full year.

---

## 📚 Reference Documentation

| Resource | Link |
|---|---|
| 🖥️ Colab Enterprise Docs | [cloud.google.com/colab/docs](https://cloud.google.com/colab/docs/introduction) |
| ⚡ NVIDIA cuDF Documentation | [docs.rapids.ai/api/cudf](https://docs.rapids.ai/api/cudf/stable/) |
| 🤖 NVIDIA cuML Documentation | [docs.rapids.ai/api/cuml](https://docs.rapids.ai/api/cuml/stable/) |
| 📦 RAPIDS Getting Started | [rapids.ai/start](https://rapids.ai/start.html) |
| 🌳 XGBoost GPU Support | [xgboost.readthedocs.io](https://xgboost.readthedocs.io/en/latest/gpu/index.html) |
| 🚕 NYC TLC Dataset | [nyc.gov/tlc](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) |
| 📊 Google Cloud AI Recipes | [GoogleCloudPlatform/ai-ml-recipes](https://github.com/GoogleCloudPlatform/ai-ml-recipes) |

---

<div align="center">

**Licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)**

*Built with ❤️ using NVIDIA RAPIDS + Google Cloud*

[![NVIDIA RAPIDS](https://img.shields.io/badge/Powered_by-NVIDIA_RAPIDS-76B900?style=for-the-badge&logo=nvidia)](https://rapids.ai)
[![Google Cloud](https://img.shields.io/badge/Runs_on-Google_Cloud-4285F4?style=for-the-badge&logo=google-cloud)](https://cloud.google.com/colab)

</div>
