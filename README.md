# AutoTimestamp Streamlit — V7.7

Aplikasi Streamlit untuk membuat overlay timestamp gaya GPS Map Camera.

## Mode

### 1. Satu Foto
Fitur lama tetap tersedia:
- upload 1 foto,
- input timestamp 3 baris secara manual,
- OCR screenshot timestamp menggunakan Gemini,
- normalisasi orientasi EXIF,
- auto-fit font Hershey agar teks panjang tidak terpotong.

### 2. Batch NPSN Online
Fitur baru sesuai workflow batch:
1. Upload foto langsung (tanpa ZIP input) dengan nama `1.jpg`, `2.jpg`, `3.jpg`, dst.
2. Masukkan satu NPSN.
3. Klik **CEK NPSN & AMBIL LOKASI ONLINE**.
4. Program melakukan exact-match NPSN ke **Referensi Data Kemendikdasmen**.
5. Program mengambil:
   - Lintang / latitude,
   - Bujur / longitude,
   - Desa/Kelurahan (fallback Kecamatan),
   - Kabupaten/Kota,
   - Provinsi.
6. Nama sekolah **tidak dimasukkan** ke overlay.
7. Koordinat dicek lagi menggunakan reverse-geocoding OpenStreetMap/Nominatim.
   - jika wilayah jelas tidak cocok, proses Generate diblokir;
   - jika validator kedua sedang tidak tersedia, data resmi Kemendikdasmen tetap ditampilkan sebagai sumber utama.
8. Pilih tanggal, jam mulai, jam selesai, dan zona waktu.
9. Program membuat jam random tetapi tetap berurutan sesuai nomor foto.
10. Output berupa ZIP foto hasil + `timestamp_log.csv`.

Format overlay batch:

```text
-6.5500687, 107.4378152
Cipaisan, Purwakarta, Jawa Barat
7/10/2026 13:12:08 WIB
```

> Catatan: waktu pada mode batch dibuat dari rentang input dan bukan waktu asli/EXIF. Metadata EXIF foto tidak diubah. Log CSV menyimpan sumber koordinat dan penanda bahwa waktu dihasilkan oleh program.

## Sumber data lokasi

Sumber utama:

`https://referensi.data.kemendikdasmen.go.id/pendidikan/npsn/<NPSN>`

Program tidak memakai nama sekolah untuk overlay. NPSN hanya dipakai sebagai kunci exact-match ke data resmi.

Validasi kedua menggunakan OpenStreetMap/Nominatim untuk mengecek kecocokan Kabupaten/Kota dan Provinsi terhadap titik koordinat resmi.

## Instalasi

```bash
pip install -r requirements.txt
```

## Jalankan

```bash
streamlit run app.py
```

## Streamlit Secrets

Untuk OCR Gemini pada mode satu foto, isi `.streamlit/secrets.toml` atau Streamlit Cloud Secrets:

```toml
GEMINI_API_KEY = "YOUR_API_KEY"
```

Fitur Batch NPSN Online tidak membutuhkan Google Maps API key.
