import pandas as pd
from pathlib import Path

csv_path = Path(__file__).parent / "customers.csv"
df = pd.read_csv(csv_path)

print("Ukuran_data:",df.shape)
print("\nTipe Kolom:\n", df.dtypes)
print("\n5 Baris Pertama:\n",df.head())
print("\nPresentase nilai kosong tertinggi:\n",
    df.isna().mean().sort_values(ascending=False).head()*100)