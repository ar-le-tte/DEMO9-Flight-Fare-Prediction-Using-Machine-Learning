# Flight Fare Prediction Using Machine Learning: End-to-End ML Pipeline Report
Airlines and travel platforms want to estimate ticket prices based on route, airline, and travel date to help with pricing strategy and dynamic recommendations.

## 0. Project Overview
**Business goal:** Predict flight ticket prices to support pricing strategy and recommendation systems.  
**ML framing:** Supervised regression.

    - **Inputs (X):** route and trip characteristics such as airline, source, destination, departure date/time (engineered into month/day features), duration, stopovers, class, booking source, seasonality, and days before departure.  
    - **Target (Y):** `Total Fare (BDT)`  
**Main evaluation metrics:** 
We will evaluate models using:

    - **`R²`** (**Coefficient of Determination**): how much variance in fare is explained by the model.

    - **Mean Absolute Error** (**`MAE`**): average absolute prediction error (in BDT).

    - **Root Mean Squared Error** (**`RMSE`**): error magnitude that penalizes large mistakes more (in BDT).

**Dataset:** [Flight Price Dataset of Bangladesh](https://www.kaggle.com/datasets/mahatiratusher/flight-price-dataset-of-bangladesh)

(Kaggle), 57,000 rows.

---

## 1. Data Understanding (Step 1)
### Key columns
- **Categorical:** Airline, Source, Destination, Class, Stopovers, Booking Source, Seasonality, Aircraft Type  
- **Datetime:** Departure Date & Time, Arrival Date & Time  
- **Numeric:** Duration (hrs), Days Before Departure, Base Fare (BDT), Tax & Surcharge (BDT), Total Fare (BDT)

### Initial observations
- High granularity in departure timestamps suggests strong time-related pricing effects.
- Fare variables show wide ranges; modeling benefits from handling skewness (log-transform).

---

## 2. Data Quality + Preprocessing (Step 2) — DE mindset (Bronze → Silver → Gold)

### Bronze (raw validation)
**Script:** `scripts/00_profile_raw.py`  
**Artifact:** `reports/bronze_quality.json`

**Result summary**
- Missing columns: none  
- Extra columns: none  
- Missing values: 0% for all columns  
- Duplicate rows: 0  
- Negative values in key numeric fields: 0

### Silver (typed + standardized + audit)
**Script:** `scripts/01_build_silver.py`  
**Artifact:** `data/interim/flight_fares_silver.parquet`, `reports/silver_checks.json`

**Transformations**
- Parsed datetimes: Departure/Arrival timestamps
- Enforced numeric types for duration and fares
- Normalized text in categorical columns
- Added derived fields: `dep_month`, `dep_dayofweek`, `route`
- Added fare audit fields:
  - `Total Fare Calc (BDT) = Base Fare + Tax & Surcharge`
  - `Fare Diff (BDT) = Total Fare - Total Fare Calc`
  - `Fare Mismatch Flag` (tolerance = 1 BDT)

**Audit finding**
- About **4.42%** of rows have `Total Fare (BDT) != Base + Tax` (beyond tolerance), with large max absolute differences.
- Decision: keep `Total Fare (BDT)` as the ground-truth target; treat audit fields as metadata (not model features).

### Gold (model-ready, leakage-free)
**Script:** `scripts/03_build_gold.py`  
**Artifact:** `data/processed/flight_fares_gold.parquet`

**Leakage policy**
- Dropped fare components (Base Fare, Tax & Surcharge) from features to avoid target leakage.
- Final features used:
  - Categorical: Airline, Source, Destination, Stopovers, Aircraft Type, Class, Booking Source, Seasonality
  - Numeric: Duration (hrs), Days Before Departure, dep_month, dep_dayofweek

---

## 3. Exploratory Data Analysis (Step 3)
EDA is provided in: `notebooks/02_eda.ipynb` (plots + KPIs).

Key plots:
- Fare distribution histogram
- Average fare by airline (bar chart)
- Fare variation by seasonality (boxplot)
- Average fare by month
- Route frequency & top expensive routes
- Correlation heatmap (numeric features)

---

## 4. Baseline Model (Step 4)
### Baseline: Linear Regression
**Script:** `scripts/04_train_baseline.py`  
**Artifacts:** `models/baseline_linear.joblib`, `reports/metrics_baseline.json`

**Important improvement**
- Modeled `log1p(Total Fare)` as target to handle skewness.
- Reported errors back in BDT scale using `expm1()`.

**Baseline performance (test set)**
- R² (log): ~0.8896  
- MAE (BDT): ~28,605  
- RMSE (BDT): ~48,262

---

## 5. Advanced Modeling & Optimization (Step 5)
### Model comparison (full dataset)
**Script:** `scripts/05_compare_models.py`  
**Artifact:** `reports/model_comparison_full.csv`

Compared:
- Linear Regression
- Ridge
- Lasso / ElasticNet (not consistently beneficial here)
- Decision Tree
- Random Forest (light config)
- Gradient Boosting Regressor (GBR)

**Best performing family (full data):** GBR (log-target)

### Hyperparameter tuning
**Script:** `scripts/06b_tune_gbr_logtarget.py`  
**Artifacts:** `models/best_gbr_logtarget.joblib`, `reports/best_gbr_logtarget_report.json`

**Tuned GBR performance (test set)**
- R² (log): ~0.8943  
- MAE (BDT): ~28,405  
- RMSE (BDT): ~47,643

---

## 6. Model Interpretation & Insights (Step 6)
### Feature importance
**Script:** `scripts/09_permutation_importance.py`  
**Artifact:** `reports/permutation_importance.csv`

(Top drivers typically include booking lead time, duration, airline, route, and class.)

### Stakeholder insights (high level)
- Fares are strongly influenced by **airline**, **route**, **class**, and **booking lead time**.
- Seasonal effects exist but are smaller compared to route/airline/class differences (data-dependent).
- Longer lead time generally correlates with lower fares (direction validated in interpretation section).

---

## 7. Deliverables
- **Data layers**
  - Bronze: `data/raw/Flight_Price_Dataset_of_Bangladesh.csv`
  - Silver: `data/interim/flight_fares_silver.parquet`
  - Gold: `data/processed/flight_fares_gold.parquet`
- **Models**
  - Baseline: `models/baseline_linear.joblib`
  - Best tuned: `models/best_gbr_logtarget.joblib`
- **Reports**
  - Quality: `reports/bronze_quality.json`, `reports/silver_checks.json`
  - Model comparison: `reports/model_comparison_full.csv`
  - Best model: `reports/best_gbr_logtarget_report.json`
  - Feature importance: `reports/permutation_importance.csv`
- [**Notebook (EDA + visuals)**](notebooks/02_eda.ipynb)
