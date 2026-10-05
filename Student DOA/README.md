# Student DOA

Aplikasi Streamlit untuk menjelajahi `student_doa.csv`, mengevaluasi model klasifikasi tiga status mahasiswa, dan mencoba prediksi dari profil yang dapat diedit.

## Menjalankan di Windows

Buka terminal pada folder `Student DOA`, lalu jalankan:

```powershell
python -m pip install -r requirements.txt
ollama pull llama3.2
python -m streamlit run app.py
```

Pastikan Ollama sudah terpasang dan berjalan, serta `student_doa.csv` berada satu folder dengan `app.py`. Jika memakai model lain, set environment variable `OLLAMA_MODEL` sesuai nama model tersebut sebelum menjalankan aplikasi. Model ML dilatih otomatis saat aplikasi dibuka, kemudian hasilnya disimpan dalam cache selama aplikasi berjalan.

## Isi aplikasi

- **BI · Analisis data:** jumlah mahasiswa per status, pola menurut kelompok usia, ringkasan capaian akademik, dan distribusi status menurut pembayaran kuliah.
- **ML · Prediksi:** Random Forest multiclass dengan pembagian stratified 80/20, metrik evaluasi, confusion matrix, kepentingan fitur, serta form profil yang bisa diedit.
- **LLM · Insight:** Ollama menyusun interpretasi berbahasa Indonesia dari profil ringkas, prediksi dan metrik ML. Prompt dapat diperiksa sebelum dikirim; jika Ollama/model tidak tersedia, aplikasi menampilkan petunjuk koneksi.
- **Dataset:** pratinjau dan unduhan CSV.

## Catatan interpretasi

Model menggunakan seluruh 34 fitur prediktor, termasuk nilai dan kemajuan akademik semester 1 dan 2. Karena itu aplikasi dimaksudkan untuk memperkirakan status setelah informasi akademik tersebut tersedia. Hasil ini bukan prediksi risiko sejak pendaftaran, bukan kepastian, dan tidak seharusnya menjadi satu-satunya dasar keputusan akademik.

Fitur kategori yang dikodekan angka (misalnya program studi atau metode pendaftaran) diproses sebagai kategori oleh pipeline, bukan diperlakukan sebagai besaran numerik. Imputasi dan one-hot encoding dipelajari hanya dari data latih.

Tab LLM memerlukan koneksi ke server Ollama yang berjalan dan model yang sudah diunduh. Bagian BI dan ML tetap bisa digunakan tanpa Ollama.
