from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split

from src.models.pipeline import make_preprocessor, build_pipeline
from src.models.eval import metrics_for_log_target

DATA_PATH = "data/processed/flight_fares_gold.parquet"
TARGET = "Total Fare (BDT)"

CATEGORICAL = [
    "Airline", "Source", "Destination", "Stopovers",
    "Aircraft Type", "Class", "Booking Source", "Seasonality"
]
NUMERIC = ["Days Before Departure", "Duration (hrs)", "dep_month", "dep_dayofweek"]


def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    X = df[CATEGORICAL + NUMERIC]
    y = np.log1p(df[TARGET])  # <-- log target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=2605
    )

    pre = make_preprocessor(NUMERIC, CATEGORICAL)
    base = GradientBoostingRegressor(random_state=2605)
    pipe = build_pipeline(base, pre)

    param_dist = {
        "model__n_estimators": [200, 400, 800],
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__max_depth": [2, 3, 4],
        "model__min_samples_leaf": [5, 10, 20],
        "model__subsample": [0.6, 0.8, 1.0],
        "model__max_features": [None, "sqrt", "log2"],
    }

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=20,
        scoring="neg_root_mean_squared_error",  # RMSE in log-space for CV
        cv=3,
        random_state=2605,
        n_jobs=-1,
        verbose=1,
    )

    search.fit(X_train, y_train)

    best_pipe = search.best_estimator_
    preds_log = best_pipe.predict(X_test)

    test_metrics = metrics_for_log_target(y_test, preds_log)

    out = {
        "model": "GradientBoostingRegressor (log-target)",
        "best_params": search.best_params_,
        "cv_best_rmse_log": float(-search.best_score_),
        "test_metrics": test_metrics,
    }

    Path("models").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)

    joblib.dump(best_pipe, "models/best_gbr_logtarget.joblib")
    Path("reports/best_gbr_logtarget_report.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(json.dumps(out, indent=2))
    print("Saved: models/best_gbr_logtarget.joblib")
    print("Saved: reports/best_gbr_logtarget_report.json")


if __name__ == "__main__":
    main()
