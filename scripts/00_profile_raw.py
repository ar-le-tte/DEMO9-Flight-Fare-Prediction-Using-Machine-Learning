import json
from pathlib import Path
from src.io import load_raw  
from src.quality import quality_report, assert_bronze_ok

RAW_PATH = "data/raw/Flight_Price_Dataset_of_Bangladesh.csv"

def main() -> None:
    df = load_raw(RAW_PATH)

    report = quality_report(df)
    print(json.dumps(report, indent=2))

    # fail fast
    assert_bronze_ok(df)

    # persist report
    Path("reports").mkdir(exist_ok=True)
    Path("reports/bronze_quality.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
