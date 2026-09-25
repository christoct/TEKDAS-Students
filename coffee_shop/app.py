"""Dashboard BI + ML untuk eksplorasi pendapatan harian kedai kopi."""
from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st

from data_prep import FEATURES, TARGET, load_data

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "artifacts" / "model.joblib"
METRICS_PATH = BASE_DIR / "artifacts" / "metrics.json"

st.set_page_config(page_title="Coffee Shop Revenue", layout="wide")
st.title("Coffee Shop Revenue — BI dan Prediksi Pendapatan")
st.caption("Eksplorasi faktor operasional dan estimasi pendapatan harian.")

try:
    data = load_data()
except Exception as exc:
    st.error(f"Dataset tidak dapat dibaca: {exc}")
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Catatan harian", f"{len(data):,}")
c2.metric("Rata-rata pendapatan", f"{data[TARGET].mean():,.2f}")
c3.metric("Pendapatan negatif", f"{(data[TARGET] < 0).sum():,}")
st.caption("Mata uang dan arti pendapatan negatif belum terdokumentasi; angka ditampilkan apa adanya.")

tab_data, tab_bi, tab_ml, tab_llm = st.tabs(["Data", "BI", "ML", "Analyst LLM"])
with tab_data:
    st.subheader("Pratinjau data")
    st.dataframe(data.head(100), use_container_width=True)
    st.markdown("**Ringkasan statistik**")
    st.dataframe(data.describe().T, use_container_width=True)
    st.markdown("**Korelasi terhadap pendapatan** (asosiasi, bukan sebab-akibat)")
    corr = data.corr(numeric_only=True)[TARGET].drop(TARGET).sort_values(ascending=False)
    st.bar_chart(corr)

with tab_bi:
    st.subheader("Pendapatan dan faktor operasional")
    selected = st.selectbox("Bandingkan pendapatan dengan", FEATURES)
    st.scatter_chart(data, x=selected, y=TARGET)
    st.markdown("**Rata-rata pendapatan per kelompok nilai faktor**")
    bins = st.slider("Jumlah kelompok", 3, 15, 8)
    grouped = data.assign(group=pd.qcut(data[selected], q=bins, duplicates="drop")).groupby(
        "group", observed=True
    )[TARGET].mean()
    st.bar_chart(grouped)

with tab_ml:
    if not MODEL_PATH.exists() or not METRICS_PATH.exists():
        st.info("Model belum dibuat. Jalankan `python data_prep.py`, lalu `python train.py`.")
    else:
        model = joblib.load(MODEL_PATH)
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        st.write(f"Model terpilih: **{metrics['selected_model']}**; dipilih berdasarkan MAE terkecil.")
        cols = st.columns(len(metrics["models"]))
        for col, (name, scores) in zip(cols, metrics["models"].items()):
            col.metric(f"{name} · MAE", f"{scores['mae']:,.2f}")
            col.caption(f"RMSE {scores['rmse']:,.2f} · R² {scores['r2']:.3f}")
        st.caption("MAE/RMSE dalam satuan target yang belum diberi label mata uang; R² bukan proporsi sebab-akibat.")
        st.subheader("Coba estimasi skenario")
        values = {}
        defaults = data[FEATURES].median()
        for feature in FEATURES:
            values[feature] = st.number_input(feature, value=float(defaults[feature]))
        if st.button("Estimasi pendapatan"):
            row = pd.DataFrame([values], columns=FEATURES)
            st.metric("Estimasi pendapatan harian", f"{model.predict(row)[0]:,.2f}")
            st.caption("Estimasi model bukan jaminan hasil aktual dan sebaiknya digunakan dalam rentang data yang diamati.")
        imp_path = BASE_DIR / "artifacts" / "feature_importance.csv"
        if imp_path.exists():
            st.subheader("Feature importance model")
            st.bar_chart(pd.read_csv(imp_path).set_index("feature"))

with tab_llm:
    st.subheader("Ringkasan evidence untuk pengelola")
    if not METRICS_PATH.exists():
        st.info("Jalankan pipeline data dan training terlebih dahulu.")
    else:
        if st.button("Buat analisis dengan Ollama"):
            try:
                from llm import build_prompt, DEFAULT_MODEL
                from ollama import chat
                metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
                summary = {
                    "rows": len(data), "mean_revenue": round(float(data[TARGET].mean()), 2),
                    "negative_revenue_rows": int((data[TARGET] < 0).sum()),
                    "feature_means": data[FEATURES].mean().round(2).to_dict(),
                }
                prompt = build_prompt(summary, metrics)
                with st.expander("Periksa evidence dan prompt"):
                    st.code(prompt)
                answer = chat(model=DEFAULT_MODEL, messages=[{"role": "user", "content": prompt}])
                st.markdown(answer["message"]["content"])
            except Exception as exc:
                st.error(f"Ollama tidak tersedia atau gagal merespons: {exc}")
        else:
            st.caption("LLM opsional. Pastikan Ollama berjalan dan model lokal tersedia.")
