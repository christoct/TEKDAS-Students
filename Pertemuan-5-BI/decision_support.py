"""Forecasting and evidence-grounded business analysis for the coffee dashboard."""

from __future__ import annotations

import json
import os
from datetime import timedelta

import pandas as pd


def forecast_daily_sales(daily_sales: pd.Series, horizon: int = 14) -> dict:
    """Backtest a random forest against a weekly-naive baseline, then forecast."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    series = daily_sales.sort_index().astype(float).asfreq("D", fill_value=0.0)
    if len(series) < 70:
        raise ValueError("Forecast memerlukan setidaknya 70 hari data historis setelah filter.")
    if horizon < 1 or horizon > 60:
        raise ValueError("Horizon forecast harus antara 1 dan 60 hari.")

    dates = series.index
    values = series.to_numpy()
    feature_rows = []
    targets = []
    target_dates = []
    for position in range(7, len(series)):
        day = dates[position]
        history = values[:position]
        feature_rows.append({
            "trend_day": position,
            "weekday": day.dayofweek,
            "month": day.month,
            "day_of_year": day.dayofyear,
            "lag_1": history[-1],
            "lag_7": history[-7],
            "rolling_mean_7": history[-7:].mean(),
        })
        targets.append(values[position])
        target_dates.append(day)

    features = pd.DataFrame(feature_rows)
    target = pd.Series(targets, dtype=float)
    dates_target = pd.DatetimeIndex(target_dates)
    validation_rows = max(14, int(len(features) * 0.2))
    split_position = len(features) - validation_rows
    if split_position < 42:
        raise ValueError("Riwayat terlalu pendek untuk pemisahan training dan evaluasi waktu.")

    model = RandomForestRegressor(
        n_estimators=250,
        min_samples_leaf=3,
        max_features=0.9,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(features.iloc[:split_position], target.iloc[:split_position])
    ml_validation = model.predict(features.iloc[split_position:])
    actual_validation = target.iloc[split_position:].to_numpy()
    naive_validation = features.iloc[split_position:]["lag_7"].to_numpy()

    def score(actual, predicted):
        return {
            "mae": float(mean_absolute_error(actual, predicted)),
            "rmse": float(mean_squared_error(actual, predicted) ** 0.5),
        }

    metrics = {
        "random_forest": score(actual_validation, ml_validation),
        "weekly_baseline": score(actual_validation, naive_validation),
        "validation_days": len(actual_validation),
        "training_days": split_position,
        "validation_start": dates_target[split_position].date().isoformat(),
        "validation_end": dates_target[-1].date().isoformat(),
    }

    full_model = RandomForestRegressor(
        n_estimators=250,
        min_samples_leaf=3,
        max_features=0.9,
        random_state=42,
        n_jobs=-1,
    )
    full_model.fit(features, target)
    recursive_history = list(values)
    baseline_history = list(values)
    forecast_rows = []
    last_day = dates[-1]
    for step in range(1, horizon + 1):
        future_day = last_day + timedelta(days=step)
        history = recursive_history
        row = pd.DataFrame([{
            "trend_day": len(values) + step - 1,
            "weekday": future_day.dayofweek,
            "month": future_day.month,
            "day_of_year": future_day.dayofyear,
            "lag_1": history[-1],
            "lag_7": history[-7],
            "rolling_mean_7": sum(history[-7:]) / 7,
        }])
        random_forest_value = max(0.0, float(full_model.predict(row)[0]))
        weekly_value = max(0.0, float(baseline_history[-7]))
        forecast_rows.append({
            "tanggal": future_day,
            "forecast_random_forest": random_forest_value,
            "forecast_baseline_mingguan": weekly_value,
        })
        recursive_history.append(random_forest_value)
        baseline_history.append(weekly_value)

    return {
        "metrics": metrics,
        "forecast": pd.DataFrame(forecast_rows),
        "history_end": dates[-1].date().isoformat(),
        "history_days": len(series),
    }


def build_business_prompt(evidence: dict) -> str:
    """Build a short prompt whose only factual source is the supplied evidence."""
    encoded_evidence = json.dumps(evidence, ensure_ascii=False, indent=2, default=str)
    return f"""Anda adalah analis bisnis untuk pemilik dan manajemen kedai kopi.

Gunakan hanya EVIDENCE_JSON berikut. Angka pada kolom sales adalah nilai penjualan; mata uang tidak tersedia.
Jangan mengarang penyebab, margin/laba, data pelanggan, stok, promosi, antrean, ataupun jumlah staf.
Bedakan fakta terukur dari hipotesis. Forecast ML adalah estimasi, bukan kepastian atau target.

EVIDENCE_JSON:
{encoded_evidence}

Tulis jawaban ringkas dalam bahasa Indonesia dengan format:
1. Ringkasan keadaan bisnis dan maksimal tiga fakta terpenting beserta angkanya.
2. Maksimal tiga tindakan yang bisa dipertimbangkan; untuk tiap tindakan sebutkan data yang perlu diperiksa sebelum keputusan.
3. Risiko atau keterbatasan analisis, termasuk informasi penting yang tidak tersedia.
Jangan menyebut korelasi atau forecast sebagai bukti sebab-akibat. Jika data tidak mendukung rekomendasi tertentu, katakan demikian."""


def generate_business_analysis(prompt: str, model: str | None = None) -> str:
    """Generate grounded management notes through a local Ollama model."""
    import ollama

    selected_model = model or os.getenv("OLLAMA_MODEL", "llama3.2")
    response = ollama.chat(
        model=selected_model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.2},
    )
    return response["message"]["content"]
