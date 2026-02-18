import pandas as pd

SILVER_PATH = "data/interim/flight_fares_silver.parquet"

def main() -> None:
    df = pd.read_parquet(SILVER_PATH)

    df["abs_diff"] = df["Fare Diff (BDT)"].abs()
    mism = df[df["Fare Mismatch Flag"]].copy()

    print(f"rows={len(df)} mismatches={len(mism)} pct={len(mism)/len(df)*100:.2f}%")
    print("abs_diff describe:")
    print(mism["abs_diff"].describe())

    # Top 20 worst mismatches
    top = mism.sort_values("abs_diff", ascending=False).head(20)
    cols = [
        "Airline","Source","Destination","Departure Date & Time","Arrival Date & Time",
        "Class","Booking Source","Stopovers","Duration (hrs)",
        "Base Fare (BDT)","Tax & Surcharge (BDT)","Total Fare (BDT)",
        "Total Fare Calc (BDT)","Fare Diff (BDT)","abs_diff","Seasonality","Days Before Departure"
    ]
    top[cols].to_csv("reports/top_20_fare_mismatches.csv", index=False)
    print("Saved reports/top_20_fare_mismatches.csv")

    # Where mismatches concentrate
    for c in ["Booking Source", "Airline", "Class", "Seasonality"]:
        share = (mism[c].value_counts() / len(mism) * 100).head(10)
        print(f"\nTop mismatch share by {c}:")
        print(share.round(2).to_string())

if __name__ == "__main__":
    main()
