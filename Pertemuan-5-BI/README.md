# Coffee Shop Sales BI, ML, dan LLM

Dashboard Streamlit untuk analisis penjualan, forecast harian, dan dukungan keputusan berbasis evidence.

## Menjalankan

Dari folder `Pertemuan-5-BI`:

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

## Forecast ML

Tab **Forecast ML** melatih Random Forest dengan fitur kalender dan lag penjualan harian. Pengujian memakai pemisahan kronologis: bagian akhir riwayat menjadi holdout, lalu MAE/RMSE dibandingkan dengan baseline yang memakai penjualan hari yang sama pada minggu sebelumnya. Forecast berikutnya dapat diekspor sebagai CSV.

Model mengikuti filter gerai, kategori, hari, dan jam, serta memakai riwayat sampai tanggal akhir yang dipilih. Forecast multi-hari bersifat rekursif; error dapat meningkat seiring horizon. Riwayat sumber hanya enam bulan dan tidak menyediakan promosi, stok, biaya, kalender hari libur, maupun status operasional gerai. Gunakan hasil sebagai skenario perencanaan, bukan target pasti.

## Analisis LLM

Tab **Analisis LLM** mengirim ringkasan agregat BI dan, bila tersedia, ringkasan forecast ke model Ollama yang berjalan lokal. Prompt dapat ditinjau sebelum permintaan dibuat. Seluruh rekomendasi perlu divalidasi oleh manajemen.

```powershell
ollama serve
ollama pull llama3.2
```

Model dapat diganti dari kolom **Nama model Ollama** atau melalui variabel lingkungan `OLLAMA_MODEL`. Pastikan Ollama diarahkan ke endpoint lokal jika data harus tetap diproses di mesin ini.

## Batas interpretasi

Kolom `sales` tidak menyebut mata uang dan bukan laba. Satu `transaction_id` adalah satu baris produk, bukan identitas pelanggan atau jaminan seluruh isi struk. Dataset tidak memiliki harga pokok, margin, persediaan, promosi, waktu antre, atau jadwal staf; BI, ML, dan LLM tidak dapat memastikan dampaknya tanpa data tambahan.
