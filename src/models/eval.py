from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def regression_metrics(y_true, y_pred) -> dict:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def save_artifacts(pipe, metrics: dict, model_name: str) -> None:
    Path("models").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)

    joblib.dump(pipe, f"models/{model_name}.joblib")
    Path(f"reports/{model_name}_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
