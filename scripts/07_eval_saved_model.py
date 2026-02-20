# scripts/07b_eval_saved_model.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.models.eval import metrics_for_log_target


DATA_PATH = "data/processed/flight_fares_gold.parquet"
TARGET = "Total Fare (BDT)"

CATEGORICAL = [
    "Airline", "Source", "Destination", "Stopovers",
    "Aircraft Type", "Class", "Booking Source", "Seasonality"
]
NUMERIC = ["Days Before Departure", "Duration (hrs)", "dep_month", "dep_dayofweek"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to .joblib pipeline")
    args = p.parse_args()

    df = pd.read_parquet(DATA_PATH)
    X = df[CATEGORICAL + NUMERIC]
    y = np.log1p(df[TARGET])


    # fixed split 
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=2605)

    pipe = joblib.load(args.model)
    preds_log = pipe.predict(X_test)

    metrics = metrics_for_log_target(y_test, preds_log)

    Path("reports").mkdir(exist_ok=True)
    Path("reports/eval_saved_model.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    print("Saved: reports/eval_saved_model.json")


if __name__ == "__main__":
    main()
