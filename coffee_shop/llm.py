"""Analisis opsional dengan Ollama; prompt hanya memakai ringkasan evidence."""
import argparse
import json
import os
import ollama

DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")


def build_prompt(summary: dict, metrics: dict) -> str:
    evidence = json.dumps({"data_summary": summary, "test_metrics": metrics}, indent=2)
    return f"""Anda adalah analis operasional kedai kopi. Gunakan hanya EVIDENCE di bawah.
Jangan menyimpulkan sebab-akibat atau mengarang konteks yang tidak tercatat.
Jelaskan temuan utama, keterbatasan prediksi, dan maksimal tiga hal yang layak diperiksa pengelola.
Sebutkan bahwa mata uang dan arti nilai pendapatan negatif belum dikonfirmasi.

EVIDENCE:\n{evidence}"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-only", action="store_true")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()
    from data_prep import load_data, FEATURES, TARGET
    df = load_data()
    metrics_path = os.path.join(os.path.dirname(__file__), "artifacts", "metrics.json")
    metrics = json.load(open(metrics_path, encoding="utf-8"))
    prompt = build_prompt({
        "rows": len(df), "feature_means": df[FEATURES].mean().round(2).to_dict(),
        "revenue_mean": round(float(df[TARGET].mean()), 2),
        "revenue_min": round(float(df[TARGET].min()), 2),
        "negative_revenue_rows": int((df[TARGET] < 0).sum()),
    }, metrics)
    if args.prompt_only:
        print(prompt)
        return
    response = ollama.chat(model=args.model, messages=[{"role": "user", "content": prompt}])
    print(response["message"]["content"])


if __name__ == "__main__":
    main()
