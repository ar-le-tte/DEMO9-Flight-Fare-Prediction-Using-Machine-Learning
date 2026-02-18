from __future__ import annotations

import numpy as np
import pandas as pd

DATA_PATH = "data/processed/flight_fares_gold.parquet"
TARGET = "Total Fare (BDT)"

def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    y = df[TARGET].dropna()

    # Basic shape diagnostics
    skew = float(y.skew())
    kurt = float(y.kurt())

    # Percentiles
    q = y.quantile([0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0])

    # Compare original vs log1p distribution (skew)
    y_log = np.log1p(y)
    skew_log = float(pd.Series(y_log).skew())
    kurt_log = float(pd.Series(y_log).kurt())

    print(f"Rows: {len(y)}")
    print(f"{TARGET} skewness: {skew:.4f} | kurtosis: {kurt:.4f}")
    print("\nPercentiles (BDT):")
    for idx, val in q.items():
        print(f"  p{int(idx*100):02d}: {val:,.2f}")

    print("\nAfter log1p transform:")
    print(f"log1p({TARGET}) skewness: {skew_log:.4f} | kurtosis: {kurt_log:.4f}")

if __name__ == "__main__":
    main()
