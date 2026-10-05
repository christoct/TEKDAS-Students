"""Dashboard and three-class student outcome classifier."""
from pathlib import Path
import json
import os

import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "student_doa.csv"
TARGET = "Target"
CATEGORICAL_COLUMNS = [
    "Marital status",
    "Application mode",
    "Course",
    "Daytime/evening attendance",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International",
]
EXPECTED_COLUMNS = [
    "Marital status", "Application mode", "Application order", "Course",
    "Daytime/evening attendance", "Previous qualification", "Nacionality",
    "Mother's qualification", "Father's qualification", "Mother's occupation",
    "Father's occupation", "Displaced", "Educational special needs", "Debtor",
    "Tuition fees up to date", "Gender", "Scholarship holder", "Age at enrollment",
    "International", "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)", "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)", "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)", "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)", "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)", "Curricular units 2nd sem (without evaluations)",
    "Unemployment rate", "Inflation rate", "GDP", TARGET,
]

st.set_page_config(page_title="Student DOA | Prediksi status mahasiswa", layout="wide")
st.title("Prediksi status mahasiswa")
st.caption("Eksplorasi dan klasifikasi status: Dropout, Enrolled, atau Graduate")


@st.cache_data
def load_data(path_string):
    path = Path(path_string)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {path}. Pastikan student_doa.csv berada satu folder dengan app.py."
        )
    frame = pd.read_csv(path)
    missing = [column for column in EXPECTED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Kolom dataset tidak lengkap: {missing}")
    if TARGET not in frame.columns:
        raise ValueError(f"Kolom target '{TARGET}' tidak ditemukan di CSV.")
    if frame.empty:
        raise ValueError("Dataset student_doa.csv kosong.")

    feature_columns = [column for column in frame.columns if column != TARGET]
    missing = [column for column in feature_columns + [TARGET] if column not in frame]
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan: {missing}")
    frame = frame.dropna(subset=[TARGET]).copy()
    if frame[TARGET].nunique() < 2:
        raise ValueError("Kolom Target harus memiliki minimal dua kelas untuk melatih model.")
    for column in feature_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=feature_columns, how="all")
    if frame[TARGET].value_counts().min() < 2:
        raise ValueError("Setiap kelas Target memerlukan setidaknya dua baris untuk pembagian train/test.")
    return frame


@st.cache_resource
def train_model(data):
    features = [column for column in data.columns if column != TARGET]
    categorical = [column for column in CATEGORICAL_COLUMNS if column in features]
    numeric = [column for column in features if column not in categorical]
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical),
    ])
    classifier = RandomForestClassifier(
        n_estimators=250,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model = Pipeline([("preprocessing", preprocessing), ("classifier", classifier)])
    x = data[features]
    y = data[TARGET].astype(str)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.20, random_state=42, stratify=y
    )
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    labels = sorted(y.unique())
    metrics = {
        "accuracy": accuracy_score(y_test, prediction),
        "balanced_accuracy": balanced_accuracy_score(y_test, prediction),
        "macro_f1": f1_score(y_test, prediction, average="macro"),
        "report": classification_report(y_test, prediction, labels=labels, output_dict=True, zero_division=0),
        "confusion": confusion_matrix(y_test, prediction, labels=labels),
        "labels": labels,
        "y_test": y_test,
        "prediction": prediction,
    }
    return model, metrics, features, numeric, categorical


try:
    df = load_data(str(DATA_PATH))
except (FileNotFoundError, ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
    st.error(f"Tidak bisa membaca dataset: {exc}")
    st.stop()

model, metrics, feature_columns, numeric_columns, categorical_columns = train_model(df)

with st.container():
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Jumlah mahasiswa", f"{len(df):,}")
    col2.metric("Jumlah fitur", f"{len(feature_columns)}")
    col3.metric("Jumlah kelas", str(df[TARGET].nunique()))
    col4.metric("Akurasi uji", f"{metrics['accuracy']:.1%}")

tab_bi, tab_ml, tab_llm, tab_data = st.tabs(
    ["BI · Analisis data", "ML · Prediksi", "LLM · Insight", "Dataset"]
)

with tab_bi:
    st.subheader("Business intelligence: pola dalam data")
    st.caption("BI merangkum pola cohort; hubungan pada data tidak membuktikan sebab-akibat.")
    count_col, age_col = st.columns(2)
    with count_col:
        st.markdown("**Jumlah mahasiswa menurut status**")
        counts = df[TARGET].value_counts().rename_axis("Status").to_frame("Mahasiswa")
        st.bar_chart(counts)
    with age_col:
        st.markdown("**Status menurut kelompok usia saat mendaftar**")
        age_groups = pd.cut(
            df["Age at enrollment"],
            bins=[0, 20, 25, 30, 40, float("inf")],
            labels=["≤20", "21–25", "26–30", "31–40", ">40"],
        )
        age_counts = pd.crosstab(age_groups, df[TARGET])
        st.bar_chart(age_counts)
    grade_columns = [
        "Curricular units 1st sem (grade)", "Curricular units 2nd sem (grade)",
    ]
    approved_columns = [
        "Curricular units 1st sem (approved)", "Curricular units 2nd sem (approved)",
    ]
    summary = df.groupby(TARGET)[grade_columns + approved_columns].mean().round(2)
    st.markdown("**Rata-rata nilai dan mata kuliah lulus per status**")
    st.dataframe(summary)
    st.markdown("**Distribusi status berdasarkan pembayaran biaya kuliah**")
    tuition = pd.crosstab(
        df["Tuition fees up to date"].map({0: "Belum lunas", 1: "Tercatat lunas"}),
        df[TARGET], normalize="index",
    ).mul(100).round(1)
    st.bar_chart(tuition)

with tab_ml:
    st.subheader("Evaluasi pada data uji")
    st.caption("Pembagian acak 80/20 dengan stratifikasi kelas dan random_state=42.")
    m1, m2, m3 = st.columns(3)
    m1.metric("Akurasi", f"{metrics['accuracy']:.1%}")
    m2.metric("Akurasi seimbang", f"{metrics['balanced_accuracy']:.1%}")
    m3.metric("Macro F1", f"{metrics['macro_f1']:.1%}")
    st.markdown("**Confusion matrix** (baris = status sebenarnya, kolom = prediksi)")
    matrix = pd.DataFrame(
        metrics["confusion"],
        index=[f"Aktual: {label}" for label in metrics["labels"]],
        columns=[f"Prediksi: {label}" for label in metrics["labels"]],
    )
    st.dataframe(matrix)
    report = pd.DataFrame(metrics["report"]).T
    st.markdown("**Precision, recall, dan F1 per kelas**")
    st.dataframe(report.round(3))
    preprocessing = model.named_steps["preprocessing"]
    names = preprocessing.get_feature_names_out()
    importances = model.named_steps["classifier"].feature_importances_
    top = pd.DataFrame({"Fitur": names, "Kepentingan": importances}).nlargest(15, "Kepentingan")
    st.markdown("**15 fitur dengan kepentingan tertinggi**")
    st.bar_chart(top.set_index("Fitur"))

    st.divider()
    st.subheader("Coba prediksi status")
    st.warning(
        "Model ini memakai informasi akademik semester 1 dan 2. Gunakan prediksi untuk penilaian "
        "setelah data dua semester tersedia; hasilnya tidak menggambarkan prediksi sejak mahasiswa baru mendaftar."
    )
    sample_index = st.selectbox(
        "Isi awal dari baris dataset (boleh diubah)",
        options=list(range(len(df))),
        format_func=lambda index: f"Baris {index + 1} — status tercatat: {df.iloc[index][TARGET]}",
    )
    sample = df.iloc[sample_index]
    with st.form("student_prediction_form"):
        st.caption("Nilai dimulai dari baris yang dipilih. Ubah nilai sesuai profil yang ingin dinilai.")
        entered = {}
        for start in range(0, len(feature_columns), 3):
            group = feature_columns[start:start + 3]
            cols = st.columns(len(group))
            for container, column in zip(cols, group):
                with container:
                    if column in categorical_columns:
                        choices = sorted(df[column].dropna().unique().tolist())
                        current = sample[column]
                        default_index = choices.index(current) if current in choices else 0
                        entered[column] = st.selectbox(
                            column,
                            choices,
                            index=default_index,
                            format_func=lambda value: str(int(value)) if float(value).is_integer() else str(value),
                            key=f"input_{sample_index}_{column}",
                        )
                    else:
                        median = float(df[column].median()) if df[column].notna().any() else 0.0
                        value = float(sample[column]) if pd.notna(sample[column]) else median
                        step = 0.1 if any(word in column.lower() for word in ["rate", "gdp", "grade"]) else 1.0
                        entered[column] = st.number_input(
                            column,
                            value=value,
                            step=step,
                            key=f"input_{sample_index}_{column}",
                        )
        submitted = st.form_submit_button("Prediksi status", type="primary")

    if submitted:
        input_frame = pd.DataFrame([entered], columns=feature_columns)
        predicted = str(model.predict(input_frame)[0])
        probabilities = model.predict_proba(input_frame)[0]
        probability_frame = pd.DataFrame({"Status": model.classes_, "Probabilitas": probabilities})
        probability_frame = probability_frame.sort_values("Probabilitas", ascending=False)
        st.session_state["last_prediction_profile"] = entered.copy()
        st.session_state["last_prediction_label"] = predicted
        st.session_state["last_prediction_probabilities"] = {
            str(label): float(probability)
            for label, probability in zip(model.classes_, probabilities)
        }
        st.success(f"Prediksi model: **{predicted}**")
        st.markdown("**Skor probabilitas model per kelas**")
        st.bar_chart(probability_frame.set_index("Status"))
        st.caption("Probabilitas adalah skor model, bukan kepastian atau keputusan akademik.")

with tab_llm:
    st.subheader("LLM analyst: interpretasi berbasis bukti")
    st.write(
        "LLM menerima profil ringkas, skor prediksi ML, dan metrik evaluasi untuk menyusun "
        "interpretasi. LLM tidak melatih atau menggantikan model ML."
    )
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2")
    st.caption(f"Provider: Ollama · model: {ollama_model}")
    if "last_prediction_profile" in st.session_state:
        llm_profile = st.session_state["last_prediction_profile"]
        llm_label = st.session_state["last_prediction_label"]
        llm_probabilities = st.session_state["last_prediction_probabilities"]
        st.info(f"Menggunakan hasil terakhir dari tab ML: **{llm_label}**")
    else:
        row_index = st.selectbox(
            "Pilih baris profil untuk analisis",
            options=list(range(len(df))),
            format_func=lambda index: f"Baris {index + 1} — status tercatat: {df.iloc[index][TARGET]}",
            key="llm_sample_index",
        )
        llm_profile = df.iloc[row_index][feature_columns].to_dict()
        llm_row = pd.DataFrame([llm_profile], columns=feature_columns)
        llm_prob_values = model.predict_proba(llm_row)[0]
        llm_probabilities = {
            str(label): float(probability)
            for label, probability in zip(model.classes_, llm_prob_values)
        }
        llm_label = str(model.classes_[llm_prob_values.argmax()])
        st.info(f"Menggunakan prediksi ML untuk baris terpilih: **{llm_label}**")

    important_features = model.named_steps["classifier"].feature_importances_
    encoded_features = model.named_steps["preprocessing"].get_feature_names_out()
    top_features = sorted(
        zip(encoded_features, important_features), key=lambda item: item[1], reverse=True
    )[:8]
    key_fields = [
        "Age at enrollment", "Tuition fees up to date", "Debtor", "Scholarship holder",
        "Curricular units 1st sem (approved)", "Curricular units 1st sem (grade)",
        "Curricular units 2nd sem (approved)", "Curricular units 2nd sem (grade)",
    ]
    evidence_profile = {key: llm_profile.get(key) for key in key_fields if key in llm_profile}
    prompt = (
        "Anda adalah asisten analis data pendidikan. Jawab dalam bahasa Indonesia.\n"
        "Gunakan hanya bukti yang diberikan. Jangan mengarang fakta, menyatakan sebab-akibat, "
        "atau menyebut probabilitas sebagai kepastian. Jelaskan bahwa fitur penting adalah "
        "kepentingan model, bukan bukti penyebab. Jangan menyarankan keputusan akademik otomatis "
        "terhadap individu.\n\n"
        f"Profil pendidikan ringkas: {json.dumps(evidence_profile, ensure_ascii=False, default=str)}\n"
        f"Hasil ML: prediksi={llm_label}; probabilitas={json.dumps(llm_probabilities, ensure_ascii=False)}\n"
        f"Evaluasi ML pada data uji: akurasi={metrics['accuracy']:.3f}; "
        f"balanced_accuracy={metrics['balanced_accuracy']:.3f}; macro_F1={metrics['macro_f1']:.3f}\n"
        f"Fitur dengan kepentingan model tertinggi: {json.dumps(top_features, ensure_ascii=False, default=str)}\n\n"
        "Tuliskan: (1) arti skor prediksi secara hati-hati, (2) hingga tiga observasi yang "
        "benar-benar didukung data, (3) saran tindak lanjut dukungan mahasiswa yang manusiawi, "
        "dan (4) batasan interpretasi. Sebutkan bahwa data memuat informasi sampai semester 2."
    )
    with st.expander("Periksa konteks yang akan dikirim ke LLM"):
        st.code(prompt, language="text")
    st.caption("Permintaan dikirim ke server Ollama pada endpoint yang dikonfigurasi di komputer ini.")
    if st.button("Buat interpretasi dengan LLM", type="primary"):
        try:
            import ollama

            with st.spinner("LLM sedang menyusun interpretasi..."):
                client = ollama.Client(timeout=120)
                response = client.chat(
                    model=ollama_model,
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": 0.2},
                )
            answer = response["message"]["content"]
            st.markdown(answer)
        except ImportError:
            st.error("Pustaka Ollama belum terpasang. Jalankan `python -m pip install -r requirements.txt`.")
        except Exception as exc:
            st.error(
                f"Tidak dapat menghubungi model Ollama '{ollama_model}'. Pastikan Ollama aktif, "
                f"model tersedia, dan OLLAMA_MODEL sesuai. Detail: {exc}"
            )

with tab_data:
    st.subheader("Dataset student_doa.csv")
    st.write(
        "Satu baris mewakili seorang mahasiswa. Terdapat fitur pendaftaran, demografi, "
        "kemajuan akademik semester 1 dan 2, serta kondisi ekonomi. Kolom kode seperti "
        "Course dan Application mode diperlakukan sebagai kategori, bukan besaran angka."
    )
    st.dataframe(df.head(100), hide_index=True)
    st.download_button(
        "Unduh CSV dataset",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="student_doa.csv",
        mime="text/csv",
    )

st.divider()
st.caption("Aplikasi pembelajaran. Evaluasi di data uji dan interpretasi konteks diperlukan sebelum penggunaan nyata.")
