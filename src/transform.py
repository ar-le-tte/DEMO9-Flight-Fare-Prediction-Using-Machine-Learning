from __future__ import annotations
import pandas as pd

def normalize_text(s: pd.Series) -> pd.Series:
    # keep it simple and safe: strip + collapse whitespace
    return (
        s.astype("string")
         .str.strip()
         .str.replace(r"\s+", " ", regex=True)
    )

def build_silver(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # --- normalize categorical text columns ---
    cat_cols = ["Airline","Source","Source Name","Destination","Destination Name",
        "Stopovers","Aircraft Type","Class","Booking Source","Seasonality"]
    for c in cat_cols:
        out[c] = normalize_text(out[c])
    
     # datetimes
    out["Departure Date & Time"] = pd.to_datetime(out["Departure Date & Time"], errors="coerce")
    out["Arrival Date & Time"] = pd.to_datetime(out["Arrival Date & Time"], errors="coerce")

    # numerics 
    num_cols = ["Duration (hrs)", "Base Fare (BDT)", "Tax & Surcharge (BDT)", "Total Fare (BDT)"]
    for c in num_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["Days Before Departure"] = pd.to_numeric(out["Days Before Departure"], errors="coerce").astype("Int64")

    # duplicates 
    dedupe_cols = ["Airline", "Source", "Destination", "Departure Date & Time", "Class", "Booking Source", "Stopovers"]
    out["dup_key"] = out[dedupe_cols].astype("string").agg("|".join, axis=1)
    out["is_potential_duplicate"] = out.duplicated(subset=dedupe_cols, keep=False)
    out["dup_group_size"] = out.groupby(dedupe_cols)["Airline"].transform("size")

    # fare audit 
    out["Total Fare Calc (BDT)"] = out["Base Fare (BDT)"] + out["Tax & Surcharge (BDT)"]
    out["Fare Diff (BDT)"] = out["Total Fare (BDT)"] - out["Total Fare Calc (BDT)"]
    out["Fare Mismatch Flag"] = out["Fare Diff (BDT)"].abs() > 1

    # derived
    out["dep_date"] = out["Departure Date & Time"].dt.date
    out["dep_month"] = out["Departure Date & Time"].dt.month.astype("Int64")
    out["dep_dayofweek"] = out["Departure Date & Time"].dt.dayofweek.astype("Int64")
    out["route"] = out["Source"].astype("string") + "-" + out["Destination"].astype("string")

    return out