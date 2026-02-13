from __future__ import annotations
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def load_raw(path: str, *, encoding: str = "utf-8") -> pd.DataFrame:
    """
    Load the raw flights CSV from data/raw/ without modifying it.
    """
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if not p.exists():
        raise FileNotFoundError(
            f"Raw data file not found at: {path}\n"
            "Put the CSV in data/raw/ (create the folders if needed)."
        )
    df = pd.read_csv(p, encoding=encoding)
    return df
