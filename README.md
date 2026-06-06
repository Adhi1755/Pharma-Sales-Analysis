# Pharma Sales Data Analysis & Forecasting

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Adhi1755/Pharma-Sales-Analysis/blob/main/Pharma_sales_data_analysis_and_forecasting.ipynb)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![pandas](https://img.shields.io/badge/pandas-2.2-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-Classifier-FF6600?style=flat-square)](https://xgboost.readthedocs.io)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-LSTM-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://tensorflow.org)

End-to-end time-series analysis and demand classification on 6 years of pharmaceutical sales data — covering EDA, stationarity testing, ARIMA forecasting, and a three-model ML ensemble (Random Forest · XGBoost · LSTM).

---

## Dataset

The dataset contains approximately **600,000 transactional records** from a single pharmacy spanning **2014–2019**, exported from a Point-of-Sale system. Sales of 57 pharmaceutical drugs are aggregated into **8 ATC Classification System categories** and pre-resampled into four temporal resolutions:

| File | Resolution | Rows |
|---|---|---|
| `salesdaily.csv` | Daily | 2,106 |
| `saleshourly.csv` | Hourly | — |
| `salesweekly.csv` | Weekly | — |
| `salesmonthly.csv` | Monthly | — |

The data is pre-processed for outlier detection and missing value imputation — no raw cleaning required.

### Drug Categories (ATC Classification)

| Code | Category |
|---|---|
| **M01AB** | Anti-inflammatory / antirheumatic — Acetic acid derivatives |
| **M01AE** | Anti-inflammatory / antirheumatic — Propionic acid derivatives |
| **N02BA** | Analgesics & antipyretics — Salicylic acid derivatives |
| **N02BE** | Analgesics & antipyretics — Pyrazolones & Anilides |
| **N05B** | Psycholeptics — Anxiolytic drugs |
| **N05C** | Psycholeptics — Hypnotics & sedatives |
| **R03** | Drugs for obstructive airway diseases |
| **R06** | Antihistamines for systemic use |

---

## Notebook Structure

### 1. Data Preparation
- Load all four resampled datasets into pandas DataFrames
- Data quality audit: shape, dtypes, missing values, duplicates
- Convert `datum` column to `datetime64` for time-based indexing
- Temporal gap analysis to confirm strict daily frequency with no missing timestamps

### 2. Exploratory Data Analysis

**Drug Proportions Analysis**
Bar chart of each category's share of total sales. **N02BE dominates** (highest consumption); **N05C has the lowest** overall volume.

**Coefficient of Variation (CV)**
Sales variability per category. N05C has the highest CV (~1.84) — most unpredictable. M01AB, M01AE, and N02BE have the lowest CVs (~0.52–0.54) — most stable and forecastable.

**Monthly Sales Trend**
Line plot of monthly mean sales revealing distinct seasonal patterns:
- N02BE and R03 peak in **cold months (Oct–Apr)** — driven by cold/flu and respiratory illness seasons
- R06 peaks in **spring/summer (Mar–May)** — aligned with allergy season
- N05B and N05C show relatively flat year-round demand

**Weekly Sales Trend**
Box plots of weekly distributions per drug, highlighting intra-year variability, outlier weeks, and stable vs. volatile demand profiles.

**Correlation Matrix**
Heatmap of pairwise correlations between drug categories to identify co-movement patterns in sales.

**Hourly Sales Patterns**
Within-day demand analysis using the hourly dataset to identify peak dispensing hours.

### 3. Stationarity Testing & Forecasting

- **ADF Test** (Augmented Dickey-Fuller) — tests for unit root / non-stationarity per drug series
- **KPSS Test** (Kwiatkowski–Phillips–Schmidt–Shin) — complements ADF for trend-stationarity
- **ARIMA(1,0,1)** fitted on the N02BE daily series, selected after stationarity confirmation; model summary with coefficient p-values, Ljung-Box, Jarque-Bera, AIC/BIC

### 4. ML Demand Classification

All three classifiers predict a **3-class demand label** (Low / Medium / High) derived from tertile-binning of sales quantity. Training uses an **80/20 time-based split** (no random shuffle) to prevent data leakage.

**Feature Set (12 features):**

| Feature | Description |
|---|---|
| `time_idx` | Sequential integer index |
| `day_of_week` | 0–6 |
| `month` | 1–12 |
| `month_sin` / `month_cos` | Cyclical encoding of month |
| `sales_lag_1` | Previous day's sales |
| `sales_lag_7` | Sales 7 days prior |
| `rolling_mean_7` | 7-day rolling average |
| `day_of_year` | 1–365 |
| `week_of_year` | 1–52 |
| `is_weekend` | Binary flag |
| `category_encoded` | Label-encoded drug category |

**Random Forest Classifier**
`n_estimators=300`, `max_depth=15`, `min_samples_split=5`, `n_jobs=-1`.
Evaluated with accuracy, classification report (precision / recall / F1 per class), confusion matrix, and feature importance ranking.

**XGBoost Classifier**
`objective=multi:softprob`, `n_estimators=500`, `learning_rate=0.05`, `max_depth=6`, `subsample=0.8`, `colsample_bytree=0.8`.
Same evaluation suite plus feature importance comparison against Random Forest.

**LSTM Classifier**
Architecture: `LSTM(50, relu) → Dropout(0.2) → Dense(3, softmax)`.
Input scaled with `StandardScaler` and reshaped to `(samples, 1, features)`.
Trained for 20 epochs, `batch_size=32`, `validation_split=0.2`, `Adam(lr=0.001)`, `sparse_categorical_crossentropy` loss.

---

## Key Findings

- **N02BE** (analgesics/antipyretics) is the highest-volume and most stable drug category — ideal anchor for inventory baseline models.
- **N05C** (hypnotics/sedatives) shows the highest relative variability (CV ~1.84) despite its low volume, making it the hardest to forecast.
- **Strong seasonal demand** for respiratory (R03) and anti-inflammatory (M01AB, M01AE) drugs in winter; antihistamines (R06) peak in spring.
- **sales_lag_1**, **rolling_mean_7**, and **time_idx** consistently rank as the top predictive features across both tree-based models.

---

## Libraries

| Library | Purpose |
|---|---|
| `pandas 2.2`, `numpy` | Data loading, feature engineering, aggregation |
| `matplotlib`, `seaborn` | EDA visualisations |
| `statsmodels` | ADF test, KPSS test, ARIMA modelling |
| `scikit-learn` | Random Forest, train/test split, metrics, StandardScaler |
| `xgboost` | Gradient-boosted classifier |
| `tensorflow / keras` | LSTM sequence model |

---

## Running the Notebook

**Option A — Google Colab (recommended)**

Click the badge at the top. Mount your Google Drive and place the four CSV files at:
```
MyDrive/Google Colab/Pharma_Sales/
├── salesdaily.csv
├── saleshourly.csv
├── salesmonthly.csv
└── salesweekly.csv
```

Then run all cells in order.

**Option B — Local**

```bash
git clone https://github.com/Adhi1755/Pharma-Sales-Analysis.git
cd Pharma-Sales-Analysis

pip install pandas==2.2.2 numpy matplotlib seaborn statsmodels \
            scikit-learn xgboost tensorflow meteostat

jupyter notebook Pharma_sales_data_analysis_and_forecasting.ipynb
```

Update the CSV paths in the *Data preparation* section to point to your local files.

---

## Author

**Adithya**  
B.Tech CSE (Data Science) — Dayananda Sagar University, Bangalore  
[GitHub](https://github.com/Adhi1755) · [LinkedIn](https://linkedin.com/in/adithyanagamuneendran)
