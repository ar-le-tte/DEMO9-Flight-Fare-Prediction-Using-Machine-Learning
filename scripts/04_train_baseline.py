import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.models.eval import metrics_for_log_target


DATA_PATH = "data/processed/flight_fares_gold.parquet"
TARGET = "Total Fare (BDT)"

CATEGORICAL = ["Airline", "Source", "Destination", "Stopovers",
    "Aircraft Type", "Class", "Booking Source", "Seasonality"]
NUMERIC = ["Days Before Departure", "Duration (hrs)", "dep_month", "dep_dayofweek"]

def main() -> None:
    df = pd.read_parquet(DATA_PATH)

    X = df[CATEGORICAL + NUMERIC]
    y = np.log1p(df[TARGET])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=2605
    )

    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),  # with_mean=False plays nice with sparse matrices
    ])

    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC),
            ("cat", categorical_pipe, CATEGORICAL),
        ],
        remainder="drop"
    )

    model = LinearRegression()

    pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model),
    ])

    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)

    m = metrics_for_log_target(y_test, preds)

    metrics = {
        "model": "LinearRegression (log-target)",
        "rows_train": int(len(X_train)),
        "rows_test": int(len(X_test)),
        **m
    }


    Path("reports").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)

    Path("reports/metrics_baseline.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    joblib.dump(pipe, "models/baseline_linear.joblib")

    print(json.dumps(metrics, indent=2))
    print("Saved model: models/baseline_linear.joblib")
    print("Saved metrics: reports/metrics_baseline.json")

if __name__ == "__main__":
    main()
