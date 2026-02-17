from __future__ import annotations
import pandas as pd
from .config import RAW_COLUMNS, NON_NEGATIVE_COLS

def quality_report(df: pd.DataFrame) -> dict:
    report = {}

    # schema
    missing_cols = [c for c in RAW_COLUMNS if c not in df.columns]
    extra_cols = [c for c in df.columns if c not in RAW_COLUMNS]
    report["missing_columns"] = missing_cols
    report["extra_columns"] = extra_cols

    # basic stats
    report["rows"] = int(df.shape[0])
    report["cols"] = int(df.shape[1])
    report["duplicate_rows"] = int(df.duplicated().sum())

    # missingness
    na = df.isna().mean().sort_values(ascending=False)
    report["missingness_pct"] = (na * 100).round(2).to_dict()

    # non-negative checks
    negatives = {}
    for c in NON_NEGATIVE_COLS:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            negatives[c] = int((s < 0).sum())
    report["negative_counts"] = negatives

    return report


def assert_bronze_ok(df: pd.DataFrame, *, dup_fail_ratio: float = 0.33) -> None:
    r = quality_report(df)

    if r["missing_columns"]:
        raise ValueError(f"Missing required columns: {r['missing_columns']}")

    # Duplicates count
    rows = r["rows"]
    dup = r["duplicate_rows"]

    if rows == 0:
        raise ValueError("Raw dataset has 0 rows (empty file).")

    dup_ratio = dup / rows
    if dup_ratio >= dup_fail_ratio:
        raise ValueError(
            f"Too many duplicate rows: {dup}/{rows} "
            f"({dup_ratio:.1%}) >= {dup_fail_ratio:.0%}. "
            "This suggests a broken extraction or duplicated ingestion."
        )

    # Fail if any negative values in key numeric columns
    neg = {k: v for k, v in r["negative_counts"].items() if v > 0}
    if neg:
        raise ValueError(f"Found negative values in numeric columns: {neg}")
def silver_checks(df: pd.DataFrame) -> dict:
    issues = {}

    # datetime parse
    issues["bad_departure_dt"] = int(df["Departure Date & Time"].isna().sum())
    issues["bad_arrival_dt"] = int(df["Arrival Date & Time"].isna().sum())
    # Duplicates
    issues["potential_duplicates"] = int(df["is_potential_duplicate"].sum())
    issues["max_dup_group_size"] = int(df["dup_group_size"].max())

    # arrival before departure
    bad_order = (df["Arrival Date & Time"] < df["Departure Date & Time"]).sum()
    issues["arrival_before_departure"] = int(bad_order)

    # fare mismatch metrics 
    issues["fare_mismatch_gt_1_bdt"] = int(df["Fare Mismatch Flag"].sum())
    issues["fare_mismatch_max_abs_diff"] = float(df["Fare Diff (BDT)"].abs().max())


    return issues
