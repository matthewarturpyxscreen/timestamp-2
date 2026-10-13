# Timestamp Generator (Streamlit)

Watermark timestamp gaya GPS Map Camera, font AVHershey Simplex Medium,
dengan opsi OCR otomatis pakai Gemini AI.

Dibuat oleh **Matthew Artur Panahatan Sitorus** & **Nikita Adriella Virginia Jacob**.

## Cara Deploy ke Streamlit Cloud

1. Push folder ini ke repo GitHub kamu (isi minimal: `app.py`, `requirements.txt`).
   **Jangan** push file `.streamlit/secrets.toml` (sudah di-`.gitignore`, hanya
   `secrets.toml.example` yang boleh ikut ke repo).
2. Buka https://share.streamlit.io → **New app**.
3. Pilih repo, branch, dan set **Main file path** ke `app.py`.
4. Sebelum/sesudah deploy, buka **App settings → Secrets**, lalu isi:

   ```toml
   GEMINI_API_KEY = "api-key-gemini-kamu"
   ```

   Ambil key gratis di https://aistudio.google.com/apikey
5. Klik **Deploy** (atau **Save** kalau app sudah jalan, lalu reboot app).

API key disimpan terenkripsi oleh Streamlit Cloud, **tidak pernah ada di
kode atau repo GitHub**. User yang membuka app tidak perlu (dan tidak bisa)
melihat atau memasukkan key ini — fitur OCR langsung jalan otomatis.

## Cara Jalan Lokal

```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# edit .streamlit/secrets.toml, isi GEMINI_API_KEY dengan key asli
streamlit run app.py
```

## Catatan

- Font AVHershey Simplex Medium di-clone otomatis dari GitHub
  (`yangcht/Hershey_font_TTF`) saat app pertama kali jalan, lalu di-cache
  (`@st.cache_resource`) supaya tidak clone ulang tiap interaksi.
- Fitur OCR pakai Gemini AI, key-nya diset SEKALI oleh pemilik app lewat
  Streamlit Secrets — user yang buka app tidak perlu input key apapun.
- Model Gemini yang dipakai: `gemini-3.6-flash` (ganti di `GEMINI_MODEL_NAME`
  kalau Google merilis versi baru / model ini di-deprecate).
- Orientasi foto (EXIF) dinormalisasi otomatis sebelum digambar, jadi
  timestamp selalu lurus/horizontal walaupun foto sumbernya dari HP yang
  menyimpan foto dengan tag rotasi (potret/landscape).
