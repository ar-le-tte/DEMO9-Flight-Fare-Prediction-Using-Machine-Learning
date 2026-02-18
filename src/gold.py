from __future__ import annotations
import pandas as pd

TARGET = "Total Fare (BDT)"

FEATURE_COLS = [
    "Airline",
    "Source",
    "Destination",
    "Stopovers",
    "Aircraft Type",
    "Class",
    "Booking Source",
    "Seasonality",
    "Days Before Departure",
    "Duration (hrs)",
    "dep_month",
    "dep_dayofweek",
]

def build_gold(df_silver: pd.DataFrame) -> pd.DataFrame:
    df = df_silver.copy()

    # Keep only columns we need
    keep = FEATURE_COLS + [TARGET]
    out = df[keep].copy()

    # Drop rows with essential nulls
    out = out.dropna(subset=[TARGET])

    # Ensure types
    out["Days Before Departure"] = pd.to_numeric(out["Days Before Departure"], errors="coerce")
    out["Duration (hrs)"] = pd.to_numeric(out["Duration (hrs)"], errors="coerce")

    return out
