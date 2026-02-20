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
def metrics_for_log_target(y_true_log, y_pred_log) -> dict:
    """
    Evaluate both:
      - performance in log space (R2 on log)
      - performance back in BDT space (MAE/RMSE on original scale)
    """
    # log-space metrics
    r2_log = float(r2_score(y_true_log, y_pred_log))

    # invert back to BDT
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    return {"r2_log": r2_log, "mae_bdt": mae, "rmse_bdt": rmse}

def save_artifacts(pipe, metrics: dict, model_name: str) -> None:
    Path("models").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)

    joblib.dump(pipe, f"models/{model_name}.joblib")
    Path(f"reports/{model_name}_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )


