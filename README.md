# Timestamp Generator V7.7 (Streamlit)

Watermark timestamp gaya GPS Map Camera dengan font AVHershey Simplex Medium, OCR Gemini untuk mode satu foto, dan mode **Batch NPSN Online**.

Dibuat oleh **Matthew Artur Panahatan Sitorus** & **Nikita Adriella Virginia Jacob**.

## Mode 1 — Satu Foto

Fitur versi sebelumnya tetap tersedia:

- Upload satu foto JPG/JPEG/PNG.
- Input timestamp manual 3 baris atau OCR screenshot melalui Gemini.
- Normalisasi orientasi EXIF.
- Auto-fit font jika baris lokasi terlalu panjang.
- Download foto hasil.

## Mode 2 — Batch NPSN Online (V7.7)

Tidak membutuhkan `master_npsn.xlsx` atau CSV.

Alur penggunaan:

1. Upload foto bernama `1.jpg`, `2.jpg`, `3.jpg`, dst.
2. Nomor harus mulai dari 1, unik, dan tidak boleh loncat agar urutan aman.
3. Masukkan satu NPSN untuk satu batch.
4. Klik **Cek NPSN & Ambil Lokasi Online**.
5. Aplikasi mengambil data langsung dari halaman **Referensi Data Kemendikdasmen** untuk NPSN tersebut.
6. Data yang dipakai hanya:
   - Lintang → latitude
   - Bujur → longitude
   - Desa/Kelurahan
   - Kabupaten/Kota
   - Provinsi
7. Nama sekolah tidak dimasukkan ke timestamp.
8. Lokasi diformat menjadi gaya seperti `Cipaisan, Purwakarta, Jawa Barat`.
9. Koordinat dicek lagi menggunakan reverse geocoding OpenStreetMap/Nominatim sebagai validator tambahan.
10. Jika validator menemukan konflik jelas pada Kabupaten/Kota atau Provinsi, tombol generate diblokir.
11. Jika reverse-geocoder sedang tidak tersedia, aplikasi tetap memakai data resmi Kemendikdasmen dan menampilkan peringatan.
12. Pilih tanggal, jam mulai, jam selesai, dan zona waktu.
13. Jam dibuat acak tetapi selalu naik mengikuti file `1 → 2 → 3 → ...`.
14. Hasil diunduh sebagai ZIP, disertai `timestamp_log.csv` untuk audit.

Contoh overlay:

```text
-6.5500687, 107.4378152
Cipaisan, Purwakarta, Jawa Barat
07/10/2026 13:12:08 WIB
```

> Penting: tanggal/jam pada mode batch adalah overlay yang dibuat berdasarkan input pengguna dan **bukan** metadata waktu pengambilan asli/EXIF foto.

## Akurasi Data Lokasi

- Lookup NPSN harus exact-match dengan NPSN yang tampil pada halaman resmi.
- Generate tidak diizinkan jika Desa/Kelurahan, Kabupaten/Kota, Provinsi, Lintang, atau Bujur resmi tidak lengkap.
- Latitude wajib berada pada rentang -90 sampai 90 dan longitude -180 sampai 180.
- Prefix administrasi (`KAB.`, `KOTA`, `PROV.`) dibersihkan hanya pada field yang tepat, sehingga nama desa yang memang mengandung kata seperti `KOTA LINTANG` tidak rusak.
- Koordinat ditampilkan 7 angka desimal agar menyerupai template GPS Camera.

## Cara Deploy ke Streamlit Cloud

1. Push seluruh folder ini ke GitHub.
2. Buka `https://share.streamlit.io` → **New app**.
3. Pilih repository/branch dan set main file ke `app.py`.
4. Untuk fitur OCR mode satu foto, isi Streamlit Secrets:

   ```toml
   GEMINI_API_KEY = "api-key-gemini-kamu"
   ```

5. Deploy/reboot aplikasi.

Mode Batch NPSN Online tidak membutuhkan API Google Maps. Akses internet dari server Streamlit diperlukan agar dapat membuka sumber resmi Kemendikdasmen dan validator OpenStreetMap.

## Cara Jalan Lokal

```bash
pip install -r requirements.txt
streamlit run app.py
```

Jika ingin memakai OCR Gemini, buat `.streamlit/secrets.toml` dan isi `GEMINI_API_KEY`.
