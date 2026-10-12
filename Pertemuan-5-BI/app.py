from datetime import timedelta
import json
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from decision_support import build_business_prompt, forecast_daily_sales, generate_business_analysis


st.set_page_config(page_title="Coffee Shop Sales | BI", page_icon="☕", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data_prepared" / "coffee_sales_prepared.csv"
DAY_NAMES = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
COLORS = ["#087F8C", "#7656B5", "#D98B39", "#32936F", "#D2645A", "#6886A5"]


@st.cache_data(ttl="1h", max_entries=3)
def load_data(path, modified_time):
    data = pd.read_csv(path, parse_dates=["transaction_date", "transaction_datetime"])
    data["day_index"] = data["transaction_date"].dt.dayofweek
    data["day_name"] = data["day_index"].map(dict(enumerate(DAY_NAMES)))
    return data


def format_number(value, decimals=2):
    if pd.isna(value):
        return "—"
    return f"{value:,.{decimals}f}".replace(",", "_").replace(".", ",").replace("_", ".")


def percent_change(current, previous):
    if pd.isna(previous) or previous == 0:
        return None
    return (current / previous - 1) * 100


def metric_delta(current, previous):
    change = percent_change(current, previous)
    return f"{change:+.1f}%" if change is not None else None


def filter_rows(data, start_date, end_date, stores, categories, weekdays, hours):
    start_stamp = pd.Timestamp(start_date)
    end_stamp = pd.Timestamp(end_date) + pd.Timedelta(days=1)
    return data.loc[
        data["transaction_date"].ge(start_stamp)
        & data["transaction_date"].lt(end_stamp)
        & data["store_location"].isin(stores)
        & data["product_category"].isin(categories)
        & data["day_name"].isin(weekdays)
        & data["hour"].between(hours[0], hours[1])
    ].copy()


def prepare_plot(fig, height=380):
    fig.update_layout(
        height=height,
        margin=dict(l=12, r=12, t=48, b=12),
        legend_title_text="",
        hoverlabel=dict(font_size=13),
    )
    return fig


st.title("Coffee Shop Sales")
st.caption("Business intelligence untuk pemantauan penjualan, produk, gerai, dan pola transaksi.")

if not DATA_PATH.exists():
    st.error("Data belum tersedia. Jalankan audit_prepare_coffee.py terlebih dahulu.")
    st.stop()

df = load_data(str(DATA_PATH), DATA_PATH.stat().st_mtime_ns)
min_date = df["transaction_date"].min().date()
max_date = df["transaction_date"].max().date()

with st.sidebar:
    st.header("Filter dashboard")
    date_selection = st.date_input(
        "Rentang tanggal",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        format="DD/MM/YYYY",
    )
    if len(date_selection) != 2:
        st.info("Pilih tanggal mulai dan tanggal akhir.")
        st.stop()
    start_date, end_date = date_selection
    if start_date > end_date:
        st.error("Tanggal mulai harus sebelum tanggal akhir.")
        st.stop()

    stores = sorted(df["store_location"].dropna().unique().tolist())
    categories = sorted(df["product_category"].dropna().unique().tolist())
    selected_stores = st.multiselect("Gerai", stores, default=stores)
    selected_categories = st.multiselect("Kategori produk", categories, default=categories)
    selected_weekdays = st.multiselect("Hari dalam minggu", DAY_NAMES, default=DAY_NAMES)
    min_hour, max_hour = int(df["hour"].min()), int(df["hour"].max())
    selected_hours = st.slider("Rentang jam transaksi", min_hour, max_hour, (min_hour, max_hour))
    monthly_target = st.number_input(
        "Target penjualan bulanan (opsional)",
        min_value=0.0,
        value=0.0,
        step=1000.0,
        help="Masukkan target dalam satuan nilai yang sama dengan kolom sales. Target tidak disimpan permanen.",
    )
    st.caption("Semua visual dan ekspor mengikuti filter ini. Mata uang sumber tidak dinyatakan.")

if not selected_stores or not selected_categories or not selected_weekdays:
    st.info("Pilih minimal satu gerai, kategori, dan hari.")
    st.stop()

current = filter_rows(
    df, start_date, end_date, selected_stores, selected_categories, selected_weekdays, selected_hours
)
if current.empty:
    st.warning("Tidak ada transaksi pada kombinasi filter ini. Ubah filter untuk melihat analisis.")
    st.stop()

period_days = (end_date - start_date).days + 1
comparison_end = start_date - timedelta(days=1)
comparison_start = comparison_end - timedelta(days=period_days - 1)
comparison = filter_rows(
    df, comparison_start, comparison_end, selected_stores, selected_categories, selected_weekdays, selected_hours
)

sales = current["sales"].sum()
transactions = current["transaction_id"].nunique()
units = current["transaction_qty"].sum()
avg_transaction = sales / transactions if transactions else 0
previous_sales = comparison["sales"].sum()
previous_transactions = comparison["transaction_id"].nunique()
previous_units = comparison["transaction_qty"].sum()
previous_avg = previous_sales / previous_transactions if previous_transactions else 0

st.caption(
    f"Cakupan: {start_date:%d %b %Y}–{end_date:%d %b %Y} · "
    f"{len(selected_stores)} gerai · {len(selected_categories)} kategori · "
    f"pembanding: {comparison_start:%d %b}–{comparison_end:%d %b %Y}"
)

with st.container(horizontal=True):
    st.metric("Total penjualan", format_number(sales), metric_delta(sales, previous_sales),
              delta_description="vs. periode sebelumnya", border=True)
    st.metric("Transaksi tercatat", format_number(transactions, 0),
              metric_delta(transactions, previous_transactions),
              delta_description="vs. periode sebelumnya", border=True)
    st.metric("Unit terjual", format_number(units, 0), metric_delta(units, previous_units),
              delta_description="vs. periode sebelumnya", border=True)
    st.metric("Rata-rata nilai per transaksi", format_number(avg_transaction),
              metric_delta(avg_transaction, previous_avg),
              delta_description="vs. periode sebelumnya", border=True)

if comparison.empty:
    st.info("Belum ada data pada periode pembanding. Perubahan KPI baru tersedia jika periode sebelumnya tercakup dalam data.")
else:
    st.caption("Pembanding memakai rentang hari dengan durasi sama tepat sebelum tanggal mulai terpilih, dengan filter non-tanggal yang sama.")

if monthly_target > 0:
    selected_month = current["transaction_date"].max().strftime("%Y-%m")
    selected_month_data = current[current["transaction_date"].dt.strftime("%Y-%m").eq(selected_month)]
    month_sales = selected_month_data["sales"].sum()
    target_progress = min(month_sales / monthly_target, 1.0)
    with st.container(border=True):
        st.subheader(f"Pantau target · {selected_month}")
        target_col, progress_col = st.columns([1, 2])
        target_col.metric("Penjualan bulan terakhir dalam cakupan", format_number(month_sales))
        progress_col.progress(target_progress, text=f"{target_progress:.1%} dari target {format_number(monthly_target)}")
        if month_sales >= monthly_target:
            st.success("Target tercapai pada bulan tersebut.")
        else:
            st.info(f"Sisa menuju target: {format_number(monthly_target - month_sales)}.")
        st.caption("Perhitungan mengikuti filter aktif; bulan dengan rentang tanggal parsial hanya memakai tanggal yang terpilih.")

filter_signature = (
    start_date.isoformat(), end_date.isoformat(), tuple(selected_stores),
    tuple(selected_categories), tuple(selected_weekdays), tuple(selected_hours),
)

overview_tab, product_tab, operations_tab, ml_tab, llm_tab, data_tab = st.tabs(
    ["Ikhtisar", "Produk & kategori", "Pola operasional", "Forecast ML", "Analisis LLM", "Data & ekspor"]
)

with overview_tab:
    st.subheader("Tren penjualan")
    trend_option = st.segmented_control(
        "Granularitas tren", ["Harian", "Mingguan", "Bulanan"], default="Mingguan"
    )
    trend_data = current.copy()
    if trend_option == "Bulanan":
        trend_data["periode"] = trend_data["transaction_date"].dt.to_period("M").astype(str)
    elif trend_option == "Mingguan":
        trend_data["periode"] = trend_data["transaction_date"].dt.to_period("W-MON").astype(str)
    else:
        trend_data["periode"] = trend_data["transaction_date"].dt.strftime("%Y-%m-%d")
    trend = trend_data.groupby("periode", as_index=False).agg(
        total_sales=("sales", "sum"),
        transactions=("transaction_id", "nunique"),
        units=("transaction_qty", "sum"),
    )
    fig_trend = px.line(
        trend, x="periode", y="total_sales", markers=True,
        hover_data={"transactions": True, "units": True},
        labels={"periode": "Periode", "total_sales": "Penjualan", "transactions": "Transaksi", "units": "Unit"},
    )
    fig_trend.update_traces(line_color=COLORS[0])
    fig_trend.update_layout(xaxis_type="category", yaxis_rangemode="tozero")
    st.plotly_chart(prepare_plot(fig_trend, 420), width="stretch", alt="Tren penjualan pada periode dan filter terpilih")

    store_summary = current.groupby("store_location", as_index=False).agg(
        total_sales=("sales", "sum"),
        transactions=("transaction_id", "nunique"),
        units=("transaction_qty", "sum"),
        active_days=("transaction_date", "nunique"),
    )
    store_summary["sales_per_recorded_day"] = store_summary["total_sales"] / store_summary["active_days"]
    store_summary = store_summary.sort_values("total_sales", ascending=False)
    category_summary = current.groupby("product_category", as_index=False).agg(
        total_sales=("sales", "sum"),
        units=("transaction_qty", "sum"),
        transactions=("transaction_id", "nunique"),
    ).sort_values("total_sales", ascending=False)
    category_summary["contribution_pct"] = category_summary["total_sales"] / sales * 100

    left, right = st.columns(2)
    with left:
        st.subheader("Performa gerai")
        fig_store = px.bar(
            store_summary.sort_values("total_sales"), x="total_sales", y="store_location",
            orientation="h", color="store_location", text_auto=".3s",
            hover_data={"transactions": True, "units": True, "active_days": True,
                        "sales_per_recorded_day": ":.2f"},
            labels={"total_sales": "Penjualan", "store_location": "Gerai",
                    "sales_per_recorded_day": "Penjualan per hari tercatat"},
            color_discrete_sequence=COLORS,
        )
        fig_store.update_layout(showlegend=False)
        st.plotly_chart(prepare_plot(fig_store), width="stretch", alt="Perbandingan penjualan dan aktivitas antar gerai")
    with right:
        st.subheader("Kontribusi kategori")
        fig_category = px.bar(
            category_summary, x="product_category", y="total_sales", color="product_category",
            text=category_summary["contribution_pct"].map(lambda value: f"{value:.1f}%"),
            hover_data={"total_sales": ":.2f", "units": True, "transactions": True,
                        "contribution_pct": ":.2f"},
            labels={"product_category": "Kategori", "total_sales": "Penjualan",
                    "contribution_pct": "Kontribusi (%)"}, color_discrete_sequence=COLORS,
        )
        fig_category.update_layout(showlegend=False)
        st.plotly_chart(prepare_plot(fig_category), width="stretch", alt="Penjualan dan kontribusi tiap kategori produk")

    st.subheader("Ringkasan per gerai")
    store_table = store_summary.rename(columns={
        "store_location": "Gerai", "total_sales": "Penjualan", "transactions": "Transaksi",
        "units": "Unit terjual", "active_days": "Hari dengan catatan",
        "sales_per_recorded_day": "Penjualan per hari tercatat",
    })
    st.dataframe(store_table, hide_index=True, width="stretch", alt="Tabel KPI dan produktivitas penjualan per gerai")
    st.caption("Hari aktif berarti tanggal dengan transaksi dalam CSV, bukan bukti gerai buka setiap hari.")

with product_tab:
    st.subheader("Produk: nilai penjualan, unit, dan kontribusi")
    product_summary = current.groupby(["product_id", "product_detail"], as_index=False).agg(
        total_sales=("sales", "sum"),
        units=("transaction_qty", "sum"),
        transactions=("transaction_id", "nunique"),
    ).sort_values("total_sales", ascending=False)
    product_summary["sales_share_pct"] = product_summary["total_sales"] / sales * 100
    product_summary["cumulative_share_pct"] = product_summary["sales_share_pct"].cumsum()
    pareto = product_summary.head(20).sort_values("total_sales", ascending=True)
    fig_products = px.bar(
        pareto, x="total_sales", y="product_detail", orientation="h", color_discrete_sequence=[COLORS[1]],
        hover_data={"product_id": True, "units": True, "transactions": True,
                    "sales_share_pct": ":.2f", "cumulative_share_pct": ":.2f"},
        labels={"total_sales": "Penjualan", "product_detail": "Produk",
                "product_id": "ID produk", "units": "Unit", "transactions": "Transaksi",
                "sales_share_pct": "Kontribusi penjualan (%)", "cumulative_share_pct": "Kontribusi kumulatif (%)"},
    )
    fig_products.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(prepare_plot(fig_products, 560), width="stretch", alt="Dua puluh produk dengan penjualan tertinggi pada filter aktif")
    st.caption("Peringkat memakai nilai sales; data tidak memuat biaya sehingga grafik tidak menunjukkan laba atau margin.")

    left, right = st.columns(2)
    with left:
        st.subheader("Campuran produk: penjualan vs unit")
        scatter_data = product_summary.copy()
        fig_scatter = px.scatter(
            scatter_data, x="units", y="total_sales", size="transactions", color="total_sales",
            hover_name="product_detail", hover_data={"product_id": True, "sales_share_pct": ":.2f"},
            labels={"units": "Unit terjual", "total_sales": "Penjualan", "transactions": "Transaksi"},
            color_continuous_scale="Tealgrn",
        )
        st.plotly_chart(prepare_plot(fig_scatter), width="stretch", alt="Hubungan jumlah unit, transaksi, dan nilai penjualan per produk")
    with right:
        st.subheader("Kontribusi kategori")
        fig_donut = px.pie(category_summary, names="product_category", values="total_sales", hole=0.58)
        fig_donut.update_traces(textinfo="none", hovertemplate=(
            "%{label}<br>Penjualan: %{value:,.2f}<br>Kontribusi: %{percent}<extra></extra>"
        ))
        st.plotly_chart(prepare_plot(fig_donut), width="stretch", alt="Bagian kontribusi penjualan dari tiap kategori")

    st.subheader("Daftar produk lengkap")
    st.dataframe(product_summary, hide_index=True, width="stretch", alt="Peringkat seluruh produk berdasarkan penjualan, unit, dan kontribusi")

with operations_tab:
    st.subheader("Pola transaksi sepanjang hari dan minggu")
    activity = current.groupby(["day_index", "day_name", "hour"], as_index=False).agg(
        transactions=("transaction_id", "nunique"), total_sales=("sales", "sum")
    )
    heatmap_metric = st.segmented_control(
        "Isi peta panas", ["Transaksi", "Penjualan"], default="Transaksi"
    )
    value_column = "transactions" if heatmap_metric == "Transaksi" else "total_sales"
    heat = activity.pivot(index="day_name", columns="hour", values=value_column)
    heat = heat.reindex(index=[day for day in DAY_NAMES if day in heat.index])
    fig_heat = px.imshow(
        heat, aspect="auto", color_continuous_scale="Tealgrn",
        labels={"x": "Jam transaksi", "y": "Hari", "color": heatmap_metric},
    )
    fig_heat.update_xaxes(dtick=1)
    st.plotly_chart(prepare_plot(fig_heat, 390), width="stretch", alt=f"Peta panas {heatmap_metric.lower()} menurut hari dan jam")

    day_summary = current.groupby(["day_index", "day_name"], as_index=False).agg(
        transactions=("transaction_id", "nunique"), total_sales=("sales", "sum"),
        units=("transaction_qty", "sum"), observed_dates=("transaction_date", "nunique"),
    ).sort_values("day_index")
    hour_summary = current.groupby("hour", as_index=False).agg(
        transactions=("transaction_id", "nunique"), total_sales=("sales", "sum"),
    ).sort_values("hour")
    busiest_day = day_summary.sort_values("transactions", ascending=False).iloc[0]
    busiest_hour = hour_summary.sort_values("transactions", ascending=False).iloc[0]
    left, right = st.columns(2)
    with left:
        st.metric("Hari dengan transaksi terbanyak", busiest_day["day_name"],
                  f"{format_number(busiest_day['transactions'], 0)} transaksi")
        fig_day = px.bar(day_summary, x="day_name", y="transactions", color="day_name",
                         color_discrete_sequence=COLORS, labels={"day_name": "Hari", "transactions": "Transaksi"})
        fig_day.update_layout(showlegend=False, xaxis={"categoryorder": "array", "categoryarray": DAY_NAMES})
        st.plotly_chart(prepare_plot(fig_day), width="stretch", alt="Jumlah transaksi tercatat untuk setiap hari dalam minggu")
    with right:
        st.metric("Jam dengan transaksi terbanyak", f"{int(busiest_hour['hour']):02d}:00",
                  f"{format_number(busiest_hour['transactions'], 0)} transaksi")
        fig_hour = px.bar(hour_summary, x="hour", y="transactions", color="transactions",
                          color_continuous_scale="Oranges", labels={"hour": "Jam", "transactions": "Transaksi"})
        fig_hour.update_layout(coloraxis_showscale=False, xaxis={"dtick": 1})
        st.plotly_chart(prepare_plot(fig_hour), width="stretch", alt="Jumlah transaksi tercatat untuk setiap jam")
    st.warning("Pola transaksi membantu merencanakan jadwal pemeriksaan operasional, tetapi bukan ukuran antrean. Dataset tidak berisi waktu tunggu, durasi layanan, jumlah pelanggan yang menunggu, atau jadwal staf.")

with data_tab:
    st.subheader("Ekspor data sesuai filter")
    current_download = current.drop(columns=["day_index", "day_name"], errors="ignore").to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Unduh transaksi terfilter (CSV)", current_download,
        file_name=f"coffee_sales_{start_date}_{end_date}.csv", mime="text/csv",
    )
    monthly_export = current.assign(month=current["transaction_date"].dt.to_period("M").astype(str)).groupby("month", as_index=False).agg(
        total_sales=("sales", "sum"),
        total_transactions=("transaction_id", "nunique"),
        total_units=("transaction_qty", "sum"),
    )
    monthly_export["average_transaction_value"] = monthly_export["total_sales"] / monthly_export["total_transactions"]
    monthly_export["growth_pct"] = monthly_export["total_sales"].pct_change() * 100
    monthly_download = monthly_export.to_csv(index=False).encode("utf-8-sig")
    st.download_button("Unduh ringkasan bulanan (CSV)", monthly_download,
                       file_name="coffee_sales_ringkasan_bulanan.csv", mime="text/csv")

    st.subheader("Transaksi terfilter")
    search_product = st.text_input("Cari nama produk", placeholder="Contoh: Latte")
    visible_data = current
    if search_product.strip():
        visible_data = visible_data[visible_data["product_detail"].str.contains(search_product.strip(), case=False, na=False)]
    st.caption(f"Menampilkan {min(len(visible_data), 2000):,} dari {len(visible_data):,} baris yang cocok. Unduhan memuat semua baris pada filter utama.")
    st.dataframe(visible_data.head(2000), hide_index=True, width="stretch", height=460,
                 alt="Baris transaksi sesuai filter utama dan pencarian produk")

with ml_tab:
    st.subheader("Forecast penjualan harian")
    st.write(
        "Model Random Forest memproyeksikan nilai penjualan harian dari pola waktu dan penjualan "
        "historis. Hasil dibandingkan dengan baseline musiman sederhana yang mengulang penjualan "
        "hari yang sama pada minggu sebelumnya."
    )
    forecast_horizon = st.select_slider(
        "Horizon forecast (hari)", options=[7, 14, 21, 28], value=14, key="forecast_horizon"
    )
    st.caption(
        f"Pelatihan memakai seluruh riwayat yang cocok dengan gerai, kategori, hari, dan jam terpilih "
        f"sampai {end_date:%d %b %Y}; tanggal mulai filter tidak memotong riwayat model."
    )

    history_rows = filter_rows(
        df, min_date, end_date, selected_stores, selected_categories, selected_weekdays, selected_hours
    )
    history_dates = pd.date_range(min_date, end_date, freq="D")
    daily_sales = history_rows.groupby("transaction_date")["sales"].sum().reindex(history_dates, fill_value=0.0)
    daily_sales.index.name = "tanggal"

    if st.button("Latih dan buat forecast", type="primary", key="run_forecast"):
        try:
            with st.spinner("Melatih model dan mengevaluasi pada bagian waktu yang ditahan..."):
                forecast_result = forecast_daily_sales(daily_sales, horizon=forecast_horizon)
            st.session_state["coffee_forecast_result"] = forecast_result
            st.session_state["coffee_forecast_signature"] = filter_signature
        except ImportError:
            st.error("Scikit-learn belum tersedia. Pasang dependensi dengan python -m pip install -r requirements.txt dari folder Pertemuan-5-BI.")
        except ValueError as exc:
            st.warning(str(exc))
        except Exception as exc:
            st.error(f"Forecast gagal dibuat: {exc}")

    forecast_result = st.session_state.get("coffee_forecast_result")
    forecast_is_current = st.session_state.get("coffee_forecast_signature") == filter_signature
    if forecast_result and forecast_is_current:
        model_metrics = forecast_result["metrics"]
        random_forest_metrics = model_metrics["random_forest"]
        baseline_metrics = model_metrics["weekly_baseline"]
        st.caption(
            f"Backtest urutan waktu: {model_metrics['validation_days']} hari, "
            f"{model_metrics['validation_start']}–{model_metrics['validation_end']}; "
            f"training memakai {model_metrics['training_days']} hari sebelumnya."
        )
        metric_cols = st.columns(4)
        metric_cols[0].metric("MAE Random Forest", format_number(random_forest_metrics["mae"]))
        metric_cols[1].metric("MAE baseline mingguan", format_number(baseline_metrics["mae"]))
        metric_cols[2].metric("RMSE Random Forest", format_number(random_forest_metrics["rmse"]))
        metric_cols[3].metric("RMSE baseline", format_number(baseline_metrics["rmse"]))
        if random_forest_metrics["mae"] <= baseline_metrics["mae"]:
            st.success("Pada periode backtest ini, Random Forest memiliki MAE lebih rendah daripada baseline mingguan.")
        else:
            st.warning("Pada periode backtest ini, baseline mingguan lebih akurat daripada Random Forest. Perlakukan forecast ML dengan ekstra hati-hati.")

        forecast_data = forecast_result["forecast"]
        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(
            x=forecast_data["tanggal"], y=forecast_data["forecast_random_forest"],
            mode="lines+markers", name="Random Forest", line={"color": COLORS[0], "width": 3},
        ))
        fig_forecast.add_trace(go.Scatter(
            x=forecast_data["tanggal"], y=forecast_data["forecast_baseline_mingguan"],
            mode="lines+markers", name="Baseline minggu lalu", line={"color": COLORS[2], "dash": "dash"},
        ))
        fig_forecast.update_layout(
            title=f"Forecast {forecast_horizon} hari setelah {forecast_result['history_end']}",
            xaxis_title="Tanggal", yaxis_title="Nilai penjualan", yaxis_rangemode="tozero",
        )
        st.plotly_chart(prepare_plot(fig_forecast, 420), width="stretch",
                        alt="Perbandingan forecast Random Forest dengan baseline penjualan mingguan")
        forecast_summary = forecast_data.rename(columns={
            "tanggal": "Tanggal", "forecast_random_forest": "Forecast Random Forest",
            "forecast_baseline_mingguan": "Baseline minggu lalu",
        })
        st.dataframe(forecast_summary, hide_index=True, width="stretch", alt="Nilai forecast per hari dari dua pendekatan")
        forecast_csv = forecast_data.to_csv(index=False).encode("utf-8-sig")
        st.download_button("Unduh forecast (CSV)", forecast_csv,
                           file_name="coffee_sales_forecast.csv", mime="text/csv")
        st.warning(
            "Backtest mengukur prediksi satu hari ke depan dengan observasi aktual yang tersedia "
            "untuk lag hari sebelumnya. Akurasi forecast multi-hari dapat lebih rendah. "
            "Enam bulan data historis belum menangkap musim tahunan, hari libur, promosi, stok, "
            "perubahan harga, ataupun penutupan gerai. Gunakan sebagai skenario perencanaan, bukan janji penjualan."
        )
    elif forecast_result:
        st.info("Filter berubah sejak forecast dibuat. Buat ulang forecast agar sesuai dengan cakupan aktif.")
    else:
        st.info("Klik tombol untuk melatih model dan melihat hasil backtest serta forecast.")

with llm_tab:
    st.subheader("Asisten keputusan bisnis")
    st.write(
        "LLM menyusun ringkasan evidence dan opsi pemeriksaan berdasarkan filter BI aktif. "
        "Inferensi diarahkan ke Ollama lokal; konfigurasi endpoint Ollama jarak jauh dapat mengubah lokasi pemrosesan."
    )
    model_name = st.text_input(
        "Nama model Ollama", value=os.getenv("OLLAMA_MODEL", "llama3.2"), key="business_llm_model"
    ).strip()

    llm_categories = current.groupby("product_category")["sales"].sum().sort_values(ascending=False)
    llm_stores = current.groupby("store_location")["sales"].sum().sort_values(ascending=False)
    llm_products = current.groupby("product_detail")["sales"].sum().sort_values(ascending=False).head(5)
    llm_monthly = current.assign(month=current["transaction_date"].dt.to_period("M").astype(str)).groupby("month")["sales"].sum()
    llm_peak_hour = current.groupby("hour")["transaction_id"].nunique().sort_values(ascending=False)
    llm_peak_day = current.groupby("day_name")["transaction_id"].nunique().sort_values(ascending=False)
    evidence = {
        "filter": {
            "date_start": start_date.isoformat(), "date_end": end_date.isoformat(),
            "stores": selected_stores, "categories": selected_categories,
            "weekdays": selected_weekdays, "hours": list(selected_hours),
        },
        "measured_bi": {
            "sales_value_currency_unknown": round(float(sales), 2),
            "unique_transaction_ids": int(transactions), "units_sold": int(units),
            "average_sales_per_transaction_id": round(float(avg_transaction), 2),
            "monthly_sales_values": {str(key): round(float(value), 2) for key, value in llm_monthly.items()},
            "sales_by_store": {str(key): round(float(value), 2) for key, value in llm_stores.items()},
            "sales_by_category": {str(key): round(float(value), 2) for key, value in llm_categories.items()},
            "top_five_products_by_sales": {str(key): round(float(value), 2) for key, value in llm_products.items()},
            "busiest_hour_by_transaction_count": int(llm_peak_hour.index[0]),
            "busiest_hour_transaction_count": int(llm_peak_hour.iloc[0]),
            "busiest_weekday_by_transaction_count": str(llm_peak_day.index[0]),
            "busiest_weekday_transaction_count": int(llm_peak_day.iloc[0]),
            "prior_equal_length_period_sales": round(float(previous_sales), 2) if not comparison.empty else None,
            "prior_equal_length_period_transactions": int(previous_transactions) if not comparison.empty else None,
        },
        "data_caveats": [
            "Satu baris adalah satu transaction_id dan satu produk; transaction_id unik bukan pelanggan unik atau seluruh struk.",
            "Mata uang, harga pokok, margin, biaya, promosi, persediaan, waktu tunggu, serta data staf tidak tersedia.",
            "Asosiasi dan pola waktu tidak membuktikan sebab-akibat.",
        ],
    }
    if forecast_result and forecast_is_current:
        forecast_values = forecast_result["forecast"]
        evidence["ml_forecast"] = {
            "history_through": forecast_result["history_end"],
            "horizon_days": len(forecast_values),
            "random_forest_forecast_sum": round(float(forecast_values["forecast_random_forest"].sum()), 2),
            "weekly_baseline_forecast_sum": round(float(forecast_values["forecast_baseline_mingguan"].sum()), 2),
            "random_forest_backtest_mae": round(float(random_forest_metrics["mae"]), 2),
            "weekly_baseline_backtest_mae": round(float(baseline_metrics["mae"]), 2),
        }
    prompt = build_business_prompt(evidence)
    with st.expander("Lihat evidence yang akan diberikan kepada LLM"):
        st.json(evidence, expanded=False)
        st.code(prompt, language="text")

    if st.button("Buat analisis keputusan", type="primary", key="generate_business_analysis"):
        if not model_name:
            st.error("Masukkan nama model Ollama yang tersedia secara lokal.")
        else:
            try:
                with st.spinner(f"Meminta analisis dari Ollama ({model_name})..."):
                    answer = generate_business_analysis(prompt, model=model_name)
                st.session_state["coffee_llm_answer"] = answer
                st.session_state["coffee_llm_signature"] = (
                    filter_signature, model_name, json.dumps(evidence, sort_keys=True, default=str)
                )
            except ImportError:
                st.error("Pustaka Ollama belum terpasang. Jalankan python -m pip install -r requirements.txt dari folder Pertemuan-5-BI.")
            except Exception as exc:
                st.error(f"Ollama belum merespons. Pastikan Ollama berjalan dan model {model_name} sudah diunduh. Detail: {exc}")

    active_llm_signature = (filter_signature, model_name, json.dumps(evidence, sort_keys=True, default=str))
    if st.session_state.get("coffee_llm_signature") == active_llm_signature:
        st.markdown("### Analisis untuk ditinjau")
        st.markdown(st.session_state["coffee_llm_answer"])
        st.caption("Gunakan rekomendasi sebagai bahan pemeriksaan dan validasi manajemen, bukan keputusan otomatis.")
    elif st.session_state.get("coffee_llm_answer"):
        st.info("Filter, forecast, atau model berubah sejak analisis dibuat. Buat ulang agar jawabannya memakai evidence terbaru.")
    else:
        st.caption("Analisis dibuat hanya saat diminta. Jalankan ollama serve dan ollama pull llama3.2 untuk model default.")

st.divider()
st.caption(
    "Batas data: setiap baris adalah satu transaction_id dan satu produk. Transaksi unik bukan pelanggan unik; "
    "rata-rata nilai transaksi belum tentu nilai seluruh isi struk. Data tidak memuat biaya, laba, "
    "mata uang, waktu tunggu, atau jadwal staf."
)
