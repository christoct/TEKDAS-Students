print("Halo mahasiswa!")
print("Pertemuan 3: Data Semantik dan Business Intelligence")
print("\n")
import pandas as pd
print("Pandas berhasil digunakan!")
print("Versi Pandas:", pd.__version__)
print("\n")
from pathlib import Path


print("\nPraktik 1 Membaca order dan menentukan arti satu baris")
folder = Path(__file__).parent
orders_path = folder / "orders.csv"
customers_path = folder / "customers.csv"

orders = pd.read_csv(orders_path)
customers = pd.read_csv(customers_path)

print(orders.head())

print("\n")
for nomor, nama_kolom in enumerate(orders.columns, start=1):
    print(nomor, nama_kolom)
print("\n")
print("Ukuran orders:", orders.shape)
print("Order ID berbeda:", orders["order_id"].nunique())
print("Customer ID berbeda:", orders["customer_id"].nunique())


df = orders.merge(
    customers[["customer_id", "membership_tier", "country"]],
    on="customer_id",
    how="left"
)

print("\n")
print(customers["customer_id"].is_unique)
print(orders["order_id"].is_unique)
print(orders["customer_id"].isin(customers["customer_id"]).all())
print("\n")
delivered = orders[orders["order_status"].eq("Delivered")]
monthly = (
    delivered
    .groupby(["year", "month"], as_index=False)
    .agg(
    orders=("order_id", "count"),
    revenue_usd=("total_amount_usd", "sum")
    )
)

print("\nPraktik 2 Membandingkan data transaksi dengan agregat bulanan\n")
monthly = pd.read_csv("monthly_revenue.csv")
print("Ukuran monthly_revenue:", monthly.shape)
print(monthly[["year", "month", "orders", "revenue_usd"]].head())
print("\nPraktik 3 Membuktikan definisi metrik bulanan\n")

januari_2020 = orders[
    (orders["year"] == 2020) & (orders["month"] == 1)
]
print("Semua order Januari:", len(januari_2020))
print(januari_2020["order_status"].value_counts())
terkirim_januari = januari_2020[
    januari_2020["order_status"] == "Delivered"
]
print("Delivered Januari:", len(terkirim_januari))
print("Revenue Delivered Januari:",
    round(terkirim_januari["total_amount_usd"].sum(), 2))

print("\nPraktik 4 Mengenal pelanggan dan melakukan join\n")
customers = pd.read_csv("customers.csv")
print("Ukuran customers:", customers.shape)
print("Customer ID berbeda:", customers["customer_id"].nunique())
print(customers[["customer_id", "country", "membership_tier"]].head())

print("\n")
orders_pelanggan = orders.merge(
    customers[["customer_id", "country", "membership_tier"]],
    on="customer_id",
    how="left",
    validate="many_to_one"
)
print("Ukuran setelah join:", orders_pelanggan.shape)
print(orders_pelanggan[
    ["order_id", "customer_id", "country", "membership_tier"]
].head())

print("\n")
print("Country kosong:", orders_pelanggan["country"].isna().sum())
print("Tier kosong:", orders_pelanggan["membership_tier"].isna().sum())

print("\nPraktik 5 Menjawab pertanyaan Business Intelligence\n")
order_terkirim = orders_pelanggan[
    orders_pelanggan["order_status"] == "Delivered"
]
pendapatan_negara = (
    order_terkirim.groupby("country")["total_amount_usd"]
    .sum()
    .sort_values(ascending=False)
)
print(pendapatan_negara.head(5))

print("\nPraktik 6 Membandingkan jumlah dan komposisi status\n")
status_negara = pd.crosstab(
    orders_pelanggan["country"],
    orders_pelanggan["order_status"]
).reindex(
    columns=["Delivered", "Processing", "Cancelled", "Returned"],
    fill_value=0
)
status_negara["Total"] = status_negara.sum(axis=1)
status_negara["Persen Delivered"] = (
 status_negara["Delivered"] / status_negara["Total"] * 100
).round(1)
print(status_negara.sort_values("Total", ascending=False).head(5).to_string())

print("\n\nPraktik 7 Nilai kosong tidak sama dengan nol\n")
print("Rating kosong:", orders["customer_rating"].isna().sum())
print("Rating nol:", orders["customer_rating"].eq(0).sum())
print("Rata-rata rating tercatat:", orders["customer_rating"].mean())

print("\n\nPraktik 8 Memahami agregat per produk\n")
produk = pd.read_csv("product_summary.csv")
print("Ukuran product_summary:", produk.shape)
print("Produk berbeda:", produk["product_name"].nunique())
print(produk[
    ["category", "product_name", "total_orders", "total_revenue_usd"]
].head())
