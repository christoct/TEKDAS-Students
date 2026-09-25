"""Validasi, pemeriksaan kualitas, dan split data pendapatan kedai kopi."""
from pathlib import Path
import json

import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "coffee_shop_revenue.csv"
TARGET = "Daily_Revenue"
FEATURES = [
    "Number_of_Customers_Per_Day", "Average_Order_Value",
    "Operating_Hours_Per_Day", "Number_of_Employees",
    "Marketing_Spend_Per_Day", "Location_Foot_Traffic",
]


def load_data() -> pd.DataFrame:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)
    required = FEATURES + [TARGET]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Kolom wajib tidak tersedia: {missing}")
    df = df[required].copy()
    for column in required:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    if df[required].isna().any().any():
        bad = df[required].isna().sum()
        raise ValueError(f"Nilai kosong/non-numerik ditemukan:\n{bad[bad > 0]}")
    if len(df) < 10:
        raise ValueError("Data terlalu sedikit untuk split train/test yang bermakna.")
    return df


def main() -> None:
    df = load_data()
    train, test = train_test_split(df, test_size=0.20, random_state=42)
    out = BASE_DIR / "data" / "processed"
    out.mkdir(parents=True, exist_ok=True)
    train.to_csv(out / "train.csv", index=False)
    test.to_csv(out / "test.csv", index=False)
    report = {
        "rows": len(df), "columns": len(df.columns), "features": FEATURES,
        "target": TARGET, "train_rows": len(train), "test_rows": len(test),
        "missing_values": df.isna().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "negative_revenue_rows": int((df[TARGET] < 0).sum()),
        "feature_summary": df[FEATURES].describe().round(3).to_dict(),
        "revenue_summary": df[TARGET].describe().round(3).to_dict(),
        "split": "Random 80/20, random_state=42; no date column is present.",
        "warning": "Negative revenue is retained and flagged; verify its business meaning.",
    }
    (out / "data_quality.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Selesai: {len(train)} baris train, {len(test)} baris test")
    print(f"Pendapatan negatif ditandai: {report['negative_revenue_rows']} baris")


if __name__ == "__main__":
    main()
