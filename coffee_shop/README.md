# Coffee Shop Revenue — proyek BI + ML + LLM

Proyek ini mengadaptasi pola folder `simulation/` untuk `coffee_shop_revenue.csv`. Karena target `Daily_Revenue` bernilai numerik, tugas ML yang digunakan adalah **regresi**, bukan klasifikasi churn.

## Isi

- `coffee_shop_revenue.csv`: data yang dianalisis (satu baris diperlakukan sebagai satu hari; konfirmasi definisi ini kepada pemilik data).
- `data_prep.py`: validasi kolom, pemeriksaan kualitas, dan split acak 80/20.
- `train.py`: baseline prediksi rata-rata, Random Forest, dan HistGradientBoosting; pemilihan model memakai MAE pada test set.
- `llm.py`: analyst opsional melalui Ollama dengan prompt berbasis ringkasan evidence.
- `app.py`: dashboard Streamlit untuk inspeksi data, BI, estimasi skenario ML, dan analyst.
- `data/processed/`: split dan laporan kualitas yang dihasilkan.
- `artifacts/`: model terpilih, metrik, prediksi, dan feature importance jika tersedia.

## Menjalankan

Dari folder `coffee_shop/`:

```bash
python -m pip install -r requirements.txt
python data_prep.py
python train.py
streamlit run app.py
```

Analyst Ollama bersifat opsional:

```bash
ollama serve
ollama pull llama3.2
python llm.py --prompt-only
python llm.py
```

## Interpretasi dan batasan

MAE dan RMSE dilaporkan dalam satuan `Daily_Revenue`; mata uang belum diketahui. Baris dengan pendapatan negatif dipertahankan dan ditandai, karena maknanya perlu dikonfirmasi (misalnya refund, koreksi, atau kesalahan). Kolom tanggal tidak tersedia, jadi model memakai split acak dan tidak menguji generalisasi ke periode mendatang. Korelasi dan feature importance tidak membuktikan sebab-akibat. Sebelum pemakaian nyata, dokumentasikan sumber/lisensi, definisi kolom, mata uang, arti observasi, serta keputusan yang akan didukung.
