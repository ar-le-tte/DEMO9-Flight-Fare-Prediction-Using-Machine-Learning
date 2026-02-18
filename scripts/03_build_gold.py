from pathlib import Path
import pandas as pd
from src.gold import build_gold

SILVER_PATH = "data/interim/flight_fares_silver.parquet"
OUT_PATH = "data/processed/flight_fares_gold.parquet"

def main() -> None:
    df_silver = pd.read_parquet(SILVER_PATH)
    df_gold = build_gold(df_silver)

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    df_gold.to_parquet(OUT_PATH, index=False)

    print(f"Saved: {OUT_PATH} rows={len(df_gold)} cols={df_gold.shape[1]}")
    print(df_gold.dtypes)

if __name__ == "__main__":
    main()
