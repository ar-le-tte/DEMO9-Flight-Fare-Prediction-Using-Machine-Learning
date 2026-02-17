import json
from pathlib import Path

from src.io import load_raw
from src.transform import build_silver
from src.quality import assert_bronze_ok, silver_checks

RAW_PATH = "data/raw/Flight_Price_Dataset_of_Bangladesh.csv"
OUT_PATH = "data/interim/flight_fares_silver.parquet"

def main() -> None:
    df_raw = load_raw(RAW_PATH)
    assert_bronze_ok(df_raw)

    df_silver = build_silver(df_raw)

    checks = silver_checks(df_silver)
    print(json.dumps(checks, indent=2))

    Path("data/interim").mkdir(parents=True, exist_ok=True)
    df_silver.to_parquet(OUT_PATH, index=False)

    Path("reports").mkdir(exist_ok=True)
    Path("reports/silver_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")

    print(f"Saved: {OUT_PATH} rows={len(df_silver)} cols={df_silver.shape[1]}")

if __name__ == "__main__":
    main()
