# scripts/09_permutation_importance.py
from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

DATA_PATH = "data/processed/flight_fares_gold.parquet"
MODEL_PATH = "models/best_gbr_logtarget.joblib"
TARGET = "Total Fare (BDT)"

CATEGORICAL = [
    "Airline", "Source", "Destination", "Stopovers",
    "Aircraft Type", "Class", "Booking Source", "Seasonality"
]
NUMERIC = ["Days Before Departure", "Duration (hrs)", "dep_month", "dep_dayofweek"]

def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    X = df[CATEGORICAL + NUMERIC]
    y = np.log1p(df[TARGET])

    # same fixed split
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=2605)

    pipe = joblib.load(MODEL_PATH)

    # Permutation on raw columns (NOT one-hot expanded).
    r = permutation_importance(
        pipe, X_test, y_test,
        n_repeats=5,
        random_state=2605,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1
    )

    imp = pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean": r.importances_mean,
        "importance_std": r.importances_std,
    }).sort_values("importance_mean", ascending=False)

    Path("reports").mkdir(exist_ok=True)
    imp.to_csv("reports/permutation_importance.csv", index=False)

    print(imp.head(15).to_string(index=False))
    print("Saved: reports/permutation_importance.csv")

if __name__ == "__main__":
    main()
