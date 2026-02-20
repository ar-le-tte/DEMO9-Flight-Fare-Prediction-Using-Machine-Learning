from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from src.models.registry import get_models
from src.models.pipeline import make_preprocessor, build_pipeline
from src.models.eval import save_artifacts, metrics_for_log_target

DATA_PATH = "data/processed/flight_fares_gold.parquet"
TARGET = "Total Fare (BDT)"

CATEGORICAL = [
    "Airline", "Source", "Destination", "Stopovers",
    "Aircraft Type", "Class", "Booking Source", "Seasonality"
]
NUMERIC = ["Days Before Departure", "Duration (hrs)", "dep_month", "dep_dayofweek"]


def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    DEV_SAMPLE_N = None
    dev_tag = "full"
    if DEV_SAMPLE_N is not None and len(df) > DEV_SAMPLE_N:
        df = df.sample(n=DEV_SAMPLE_N, random_state=2605).reset_index(drop=True)
        dev_tag = f"sample_{DEV_SAMPLE_N}"
        print(f"[DEV MODE] Using sample of {DEV_SAMPLE_N} rows")

    X = df[CATEGORICAL + NUMERIC]
    y = np.log1p(df[TARGET])


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=2605
    )

    preprocessor = make_preprocessor(NUMERIC, CATEGORICAL)
    models = get_models(random_state=2605)

    results = []
    for name, estimator in models.items():
        print(f"\n=== Training {name} ===")
        pipe = build_pipeline(estimator, preprocessor)
        pipe.fit(X_train, y_train)

        preds = pipe.predict(X_test)
        m = metrics_for_log_target(y_test, preds)
        row = {"model": name, **m}
        results.append(row)
        save_artifacts(pipe, m, name)

    res_df = pd.DataFrame(results).sort_values("rmse_bdt", ascending=True)

    Path("reports").mkdir(exist_ok=True)
    out_csv = f"reports/model_comparison_{dev_tag}.csv"
    res_df.to_csv(out_csv, index=False) 

    print(res_df.to_string(index=False))
    print(f"Saved: {out_csv}")
    print("Saved: models/*.joblib and reports/*_metrics.json")


if __name__ == "__main__":
    main()
