import io
import hashlib
import math
import os
import re
import secrets
import zipfile
from datetime import datetime, date, time, timedelta

import cv2
import numpy as np
import pandas as pd
from PIL import ImageOps
from streamlit_paste_button import paste_image_button as pbutton

try:
    import streamlit as st
except ImportError:
    st = None

BATCH_MANUAL_API_VERSION = "8.8"


# ============================================================
# HELPERS
# ============================================================


def _collapse_spaces(value):
    return re.sub(r"\s+", " ", str(value or "").strip())


def _photo_sort_key(uploaded_file):
    stem = os.path.splitext(os.path.basename(uploaded_file.name))[0].strip()
    if stem.isdigit():
        return (0, int(stem), uploaded_file.name.lower())

    parts = re.split(r"(\d+)", stem.lower())
    natural = tuple(int(part) if part.isdigit() else part for part in parts)
    return (1, natural, uploaded_file.name.lower())


def validate_numeric_photo_names(uploaded_files):
    invalid = []
    duplicates = []
    seen = set()

    for uploaded_file in uploaded_files or []:
        stem = os.path.splitext(os.path.basename(uploaded_file.name))[0].strip()
        if not stem.isdigit():
            invalid.append(uploaded_file.name)
            continue
        number = int(stem)
        if number in seen:
            duplicates.append(number)
        seen.add(number)

    errors = []
    if invalid:
        errors.append(
            "Nama file harus angka saja, contoh 1.jpg, 2.jpg, 3.jpg. Tidak valid: "
            + ", ".join(invalid[:8])
        )
    if duplicates:
        errors.append(
            "Nomor foto duplikat: " + ", ".join(str(x) for x in sorted(set(duplicates)))
        )
    return errors


def generate_ordered_random_datetimes(selected_date, start_time, end_time, count):
    if count <= 0:
        return []

    start_dt = datetime.combine(selected_date, start_time)
    end_dt = datetime.combine(selected_date, end_time)

    if end_dt <= start_dt:
        raise ValueError("Jam selesai harus lebih besar daripada jam mulai.")

    total_seconds = int((end_dt - start_dt).total_seconds())
    if total_seconds + 1 < count:
        raise ValueError("Rentang waktu terlalu pendek untuk membuat jam unik untuk semua foto.")

    offsets = sorted(secrets.SystemRandom().sample(range(total_seconds + 1), count))
    return [start_dt + timedelta(seconds=offset) for offset in offsets]


def _encode_pil_image_bytes(pil_image, original_filename):
    _, ext = os.path.splitext(original_filename)
    ext = ext.lower()
    if ext not in [".jpg", ".jpeg", ".png"]:
        ext = ".jpg"

    rgb = np.array(pil_image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    if ext in [".jpg", ".jpeg"]:
        ok, encoded = cv2.imencode(ext, bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        mime = "image/jpeg"
    else:
        ok, encoded = cv2.imencode(ext, bgr)
        mime = "image/png"

    if not ok:
        raise ValueError(f"Gagal encode hasil foto: {original_filename}")

    return encoded.tobytes(), mime


def _jitter_single_coordinate(base_latitude, base_longitude, max_radius_meters):
    if max_radius_meters <= 0:
        return base_latitude, base_longitude, 0.0

    random_gen = secrets.SystemRandom()
    angle = random_gen.uniform(0, 2 * math.pi)
    distance_m = random_gen.uniform(0, max_radius_meters)

    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = max(111_320.0 * math.cos(math.radians(base_latitude)), 1e-9)

    delta_lat = (distance_m * math.cos(angle)) / meters_per_deg_lat
    delta_lon = (distance_m * math.sin(angle)) / meters_per_deg_lon

    return (
        base_latitude + delta_lat,
        base_longitude + delta_lon,
        distance_m,
    )


def _parse_coordinate_pair(text):
    text = _collapse_spaces(text)
    matches = re.findall(r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)", text)
    if len(matches) < 2:
        raise ValueError("Koordinat harus berisi latitude dan longitude. Contoh: -6.5500687, 107.4378152")

    latitude = float(matches[0])
    longitude = float(matches[1])

    if not (-90 <= latitude <= 90):
        raise ValueError("Latitude di luar rentang yang valid.")
    if not (-180 <= longitude <= 180):
        raise ValueError("Longitude di luar rentang yang valid.")

    return latitude, longitude


def _timezone_options():
    return [
        "WIB", "WITA", "WIT", "GMT", "UTC",
        "GMT-12", "GMT-11", "GMT-10", "GMT-9", "GMT-8", "GMT-7", "GMT-6",
        "GMT-5", "GMT-4", "GMT-3", "GMT-2", "GMT-1", "GMT+0", "GMT+1", "GMT+2",
        "GMT+3", "GMT+4", "GMT+5", "GMT+6", "GMT+7", "GMT+8", "GMT+9", "GMT+10",
        "GMT+11", "GMT+12", "GMT+13", "GMT+14",
        "UTC-12", "UTC-11", "UTC-10", "UTC-9", "UTC-8", "UTC-7", "UTC-6",
        "UTC-5", "UTC-4", "UTC-3", "UTC-2", "UTC-1", "UTC+0", "UTC+1", "UTC+2",
        "UTC+3", "UTC+4", "UTC+5", "UTC+6", "UTC+7", "UTC+8", "UTC+9", "UTC+10",
        "UTC+11", "UTC+12", "UTC+13", "UTC+14",
    ]


def _extract_timezone_label(text):
    """Ambil label zona waktu dari timestamp acuan/OCR tanpa mengubah format labelnya."""
    raw = str(text or "")
    if not raw.strip():
        return ""

    # Prioritaskan label Indonesia, lalu GMT/UTC dengan offset.
    match = re.search(r"\b(WITA|WIB|WIT)\b", raw, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()

    match = re.search(
        r"\b(GMT|UTC)\s*([+-])\s*(1[0-4]|[0-9])(?:\s*:\s*(00|15|30|45))?\b",
        raw,
        flags=re.IGNORECASE,
    )
    if match:
        prefix = match.group(1).upper()
        sign = match.group(2)
        hour = int(match.group(3))
        minute = match.group(4)
        if minute and minute != "00":
            return f"{prefix}{sign}{hour}:{minute}"
        return f"{prefix}{sign}{hour}"

    # Label GMT/UTC tanpa offset tetap dipertahankan jika memang tertulis begitu.
    match = re.search(r"\b(GMT|UTC)\b", raw, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()

    return ""



def _timezone_info_from_label(label):
    """Ubah label zona waktu ke offset + pilihan label ekuivalen."""
    label = str(label or "").strip().upper()
    if not label:
        return None

    indonesia_offsets = {"WIB": 7, "WITA": 8, "WIT": 9}
    if label in indonesia_offsets:
        offset = indonesia_offsets[label]
        return {
            "source_label": label,
            "offset": offset,
            "indonesia_label": label,
            "gmt_label": f"GMT+{offset}",
            "utc_label": f"UTC+{offset}",
        }

    match = re.fullmatch(r"(GMT|UTC)([+-])(1[0-4]|[0-9])(?::(00|15|30|45))?", label)
    if not match:
        if label in {"GMT", "UTC"}:
            return {
                "source_label": label,
                "offset": 0,
                "indonesia_label": None,
                "gmt_label": "GMT+0",
                "utc_label": "UTC+0",
            }
        return None

    sign = 1 if match.group(2) == "+" else -1
    hour = int(match.group(3))
    minute = int(match.group(4) or 0)
    offset = sign * (hour + minute / 60)

    indonesia_label = None
    if minute == 0:
        if offset == 7:
            indonesia_label = "WIB"
        elif offset == 8:
            indonesia_label = "WITA"
        elif offset == 9:
            indonesia_label = "WIT"

    if minute:
        abs_offset = abs(offset)
        whole = int(abs_offset)
        mins = int(round((abs_offset - whole) * 60))
        sign_text = "+" if offset >= 0 else "-"
        gmt_label = f"GMT{sign_text}{whole}:{mins:02d}"
        utc_label = f"UTC{sign_text}{whole}:{mins:02d}"
    else:
        sign_text = "+" if offset >= 0 else "-"
        whole = int(abs(offset))
        gmt_label = f"GMT{sign_text}{whole}"
        utc_label = f"UTC{sign_text}{whole}"

    return {
        "source_label": label,
        "offset": offset,
        "indonesia_label": indonesia_label,
        "gmt_label": gmt_label,
        "utc_label": utc_label,
    }


def _normalize_location_for_timezone(value):
    value = _collapse_spaces(value).lower()
    value = value.replace("d.k.i.", "dki").replace("d.i.", "di")
    value = re.sub(r"[^a-z0-9 ]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _infer_indonesia_timezone_from_location(location_text):
    """Fallback auto timezone untuk Batch Manual bila nama provinsi tampak di teks lokasi."""
    text = _normalize_location_for_timezone(location_text)
    if not text:
        return None

    province_aliases = {
        "aceh": "WIB", "sumatera utara": "WIB", "sumatera barat": "WIB",
        "riau": "WIB", "kepulauan riau": "WIB", "jambi": "WIB",
        "bengkulu": "WIB", "sumatera selatan": "WIB", "kepulauan bangka belitung": "WIB",
        "lampung": "WIB", "banten": "WIB", "dki jakarta": "WIB", "jakarta": "WIB",
        "jawa barat": "WIB", "jawa tengah": "WIB", "di yogyakarta": "WIB",
        "yogyakarta": "WIB", "jawa timur": "WIB", "kalimantan barat": "WIB",
        "kalimantan tengah": "WIB",

        "bali": "WITA", "nusa tenggara barat": "WITA", "nusa tenggara timur": "WITA",
        "kalimantan selatan": "WITA", "kalimantan timur": "WITA", "kalimantan utara": "WITA",
        "sulawesi utara": "WITA", "gorontalo": "WITA", "sulawesi tengah": "WITA",
        "sulawesi barat": "WITA", "sulawesi selatan": "WITA", "sulawesi tenggara": "WITA",

        "maluku": "WIT", "maluku utara": "WIT", "papua": "WIT", "papua barat": "WIT",
        "papua barat daya": "WIT", "papua tengah": "WIT", "papua pegunungan": "WIT",
        "papua selatan": "WIT",
    }

    # Cari nama terpanjang dulu supaya "papua barat daya" tidak tertangkap sebagai "papua".
    for province in sorted(province_aliases, key=len, reverse=True):
        if re.search(rf"\b{re.escape(province)}\b", text):
            info = _timezone_info_from_label(province_aliases[province])
            if info:
                info["detected_province"] = province.title()
                return info
    return None


# ============================================================
# BATCH FONT SIZE HELPERS
# ============================================================


def _calculate_locked_font_size_px(uploaded_files, load_image_fn, font_size_mode, font_size_percent, fixed_font_size_px):
    if font_size_mode == "fixed_px":
        return max(14, int(round(fixed_font_size_px))), []

    basis_values = []
    dimensions = []
    for uploaded_file in uploaded_files or []:
        uploaded_file.seek(0)
        image = load_image_fn(uploaded_file)
        width, height = image.size
        dimensions.append((uploaded_file.name, width, height))
        basis_values.append(min(width, height) if font_size_mode == "short_edge" else width)
        uploaded_file.seek(0)

    if not basis_values:
        return None, dimensions

    median_basis = float(np.median(basis_values))
    locked_px = max(14, int(round(median_basis * (font_size_percent / 100.0))))
    return locked_px, dimensions


# ============================================================
# STREAMLIT UI
# ============================================================


def render_batch_mode(uploader_version, generate_timestamp_fn, load_image_fn, ocr_extract_text_fn):
    if st is None:
        raise RuntimeError("Streamlit belum terpasang. Jalankan pip install -r requirements.txt")

    state_key = "batch_manual_generated_result"
    ocr_signature_key = "batch_manual_last_ocr_signature"
    timestamp_input_key = f"batch_manual_timestamp_input_{uploader_version}"
    if state_key not in st.session_state:
        st.session_state[state_key] = None
    if ocr_signature_key not in st.session_state:
        st.session_state[ocr_signature_key] = None

    st.markdown(
        '<div class="hero-badge">⚡ V9.1 • Batch Manual + Auto Timezone Lengkap</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="hero-title">🗂️ Batch Manual Timestamp</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="hero-desc">Upload foto bernama <b>1, 2, 3, ...</b>, isi timestamp acuan manual atau scan OCR, '
        'atur tanggal serta rentang jam seperti mode Batch NPSN Online, lalu generate timestamp untuk semua foto.</div>',
        unsafe_allow_html=True,
    )

    st.warning(
        "⚠️ Jam pada mode ini dibuat dari rentang yang kamu pilih, bukan waktu pengambilan asli/EXIF. "
        "Program hanya menggambar overlay visual dan tidak mengubah metadata EXIF asli."
    )

    st.markdown("### 1️⃣ Upload Foto Batch")
    with st.container(border=True):
        batch_files = st.file_uploader(
            "Pilih beberapa foto (JPG, JPEG, PNG)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key=f"batch_manual_photo_uploader_{uploader_version}",
            help="Upload langsung beberapa foto. Tidak perlu ZIP. Nama file: 1.jpg, 2.jpg, 3.jpg, dst.",
        )

        sorted_files = sorted(batch_files or [], key=_photo_sort_key)
        name_errors = validate_numeric_photo_names(sorted_files)

        for error_text in name_errors:
            st.error("❌ " + error_text)

        if sorted_files and not name_errors:
            st.success(f"✅ {len(sorted_files)} foto siap. Urutan proses mengikuti nomor file.")
            st.dataframe(
                pd.DataFrame(
                    [{"Urutan": i, "Nama File": f.name} for i, f in enumerate(sorted_files, start=1)]
                ),
                use_container_width=True,
                hide_index=True,
            )
        elif not sorted_files:
            st.info("Upload foto yang sudah dinamai 1, 2, 3, 4, ...")

    st.markdown("### 2️⃣ Masukkan Timestamp Acuan")
    with st.container(border=True):
        tab_manual, tab_ocr = st.tabs(
            [
                "✍️ Ketik Manual",
                "🔍 OCR Screenshot (Gemini AI)",
            ]
        )

        with tab_manual:
            st.caption(
                "Masukkan minimal 2 baris: **baris 1 koordinat** dan **baris 2 lokasi**. "
                "Kalau kamu menempel timestamp 3 baris lengkap, baris tanggal/jamnya hanya menjadi acuan "
                "dan tidak dipakai untuk batch karena tanggal serta range jam diatur pada Langkah 3."
            )

        with tab_ocr:
            gemini_api_key = st.secrets.get("GEMINI_API_KEY", "")

            if not gemini_api_key:
                st.warning(
                    "⚠️ **`GEMINI_API_KEY` belum dikonfigurasi di Streamlit Secrets.** "
                    "Fitur OCR otomatis membutuhkan API key di pengaturan Secrets."
                )
            else:
                st.caption(
                    "✨ **Gemini OCR Siap.** Screenshot bagian timestamp, lalu paste dengan Ctrl+V "
                    "atau upload screenshot seperti pada mode Satu Foto."
                )

            col_paste_btn, col_upload_ocr = st.columns([1, 1])

            with col_paste_btn:
                paste_result = pbutton(
                    label="📋 Klik lalu Tekan Ctrl+V (Paste)",
                    key=f"batch_manual_screenshot_paste_{uploader_version}",
                )

            with col_upload_ocr:
                screenshot_file = st.file_uploader(
                    "Atau upload screenshot timestamp",
                    type=["jpg", "jpeg", "png"],
                    key=f"batch_manual_screenshot_uploader_{uploader_version}",
                    label_visibility="collapsed",
                )

            screenshot_image = None

            if paste_result.image_data is not None:
                try:
                    screenshot_image = ImageOps.exif_transpose(
                        paste_result.image_data
                    ).convert("RGB")
                except Exception as error:
                    st.error(f"❌ Screenshot hasil paste tidak dapat dibaca: {error}")

            elif screenshot_file is not None:
                try:
                    screenshot_image = load_image_fn(screenshot_file)
                except Exception as error:
                    st.error(f"❌ Screenshot tidak dapat dibaca: {error}")

            if screenshot_image is not None:
                st.image(
                    screenshot_image,
                    caption="Preview Screenshot untuk OCR",
                    use_container_width=True,
                )

                image_buffer_for_hash = io.BytesIO()
                screenshot_image.save(image_buffer_for_hash, format="PNG")
                screenshot_signature = hashlib.md5(
                    image_buffer_for_hash.getvalue()
                ).hexdigest()

                already_processed = (
                    st.session_state.get(ocr_signature_key) == screenshot_signature
                )

                col_ocr_status, col_ocr_button = st.columns([3, 1])

                with col_ocr_button:
                    rerun_ocr_clicked = st.button(
                        "🔁 OCR Ulang",
                        type="secondary",
                        disabled=not gemini_api_key,
                        use_container_width=True,
                        key=f"batch_manual_ocr_rerun_{uploader_version}",
                    )

                should_run_ocr = (
                    gemini_api_key
                    and (not already_processed or rerun_ocr_clicked)
                )

                if should_run_ocr:
                    try:
                        with col_ocr_status:
                            with st.spinner("⏳ Meminta Gemini membaca teks timestamp..."):
                                ocr_result = ocr_extract_text_fn(
                                    screenshot_image,
                                    gemini_api_key,
                                )

                        st.session_state[ocr_signature_key] = screenshot_signature

                        if ocr_result:
                            st.session_state[timestamp_input_key] = ocr_result
                            st.success(
                                "✅ OCR berhasil! Koordinat dan lokasi sudah dimasukkan ke Timestamp Acuan."
                            )
                            st.rerun()
                        else:
                            st.warning("⚠️ Gemini tidak mengembalikan teks apapun.")

                    except Exception as error:
                        st.error(f"❌ ERROR OCR GEMINI: {error}")

                elif already_processed:
                    st.caption("✅ Screenshot ini sudah berhasil di-OCR.")

        st.markdown("---")

        timestamp_reference_text = st.text_area(
            "Timestamp Acuan",
            height=130,
            placeholder=(
                "-6.5500687, 107.4378152\n"
                "Cipaisan, Purwakarta, Jawa Barat\n"
                "7/10/2026 13:12:08 WIB"
            ),
            key=timestamp_input_key,
            help=(
                "Baris 1: koordinat | Baris 2: lokasi. Baris 3 bila ada dianggap waktu acuan dan tidak dipakai "
                "untuk generate jam batch. Baris ke-4 dan seterusnya akan menjadi baris tambahan."
            ),
        )

        reference_lines = [
            line.strip()
            for line in timestamp_reference_text.splitlines()
            if line.strip()
        ]

        manual_location_ready = False
        manual_latitude = None
        manual_longitude = None
        location_text_input = ""
        reference_time_text = ""
        extra_lines = []

        if reference_lines:
            if len(reference_lines) < 2:
                st.error("❌ Timestamp acuan minimal harus memiliki 2 baris: koordinat dan lokasi.")
            else:
                try:
                    manual_latitude, manual_longitude = _parse_coordinate_pair(reference_lines[0])
                    location_text_input = _collapse_spaces(reference_lines[1])
                    if not location_text_input:
                        raise ValueError("Daerah/lokasi belum diisi.")

                    if len(reference_lines) >= 3:
                        reference_time_text = reference_lines[2]
                    if len(reference_lines) >= 4:
                        extra_lines = reference_lines[3:]

                    manual_location_ready = True
                    st.success("✅ Timestamp acuan valid dan siap dipakai untuk Batch Manual.")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.metric("Latitude / Lintang", f"{manual_latitude:.7f}")
                    with c2:
                        st.metric("Longitude / Bujur", f"{manual_longitude:.7f}")
                    st.markdown(f"📍 **Daerah untuk timestamp:** `{location_text_input}`")

                    if reference_time_text:
                        detected_reference_timezone = _extract_timezone_label(reference_time_text)
                        detected_note = (
                            f" Zona waktu terdeteksi otomatis: **{detected_reference_timezone}**."
                            if detected_reference_timezone
                            else " Zona waktu belum terdeteksi dari teks acuan."
                        )
                        st.info(
                            f"🕒 Waktu acuan terbaca: `{reference_time_text}` — tanggal/jam acuan **tidak dipakai**."
                            + detected_note
                            + " Tanggal dan range jam hasil batch mengikuti Langkah 3."
                        )
                    if extra_lines:
                        st.caption(
                            "Baris tambahan yang akan ikut pada semua foto: "
                            + " | ".join(extra_lines)
                        )

                except Exception as error:
                    st.error(f"❌ {error}")

    st.markdown("### 3️⃣ Atur Tanggal & Rentang Jam")
    with st.container(border=True):
        c_date, c_start, c_end = st.columns(3)
        with c_date:
            batch_date = st.date_input(
                "Tanggal",
                value=date.today(),
                key=f"batch_manual_date_{uploader_version}",
                format="DD/MM/YYYY",
            )
        with c_start:
            batch_start = st.time_input(
                "Jam Mulai",
                value=time(13, 0, 0),
                key=f"batch_manual_start_{uploader_version}",
                step=60,
            )
        with c_end:
            batch_end = st.time_input(
                "Jam Selesai",
                value=time(16, 0, 0),
                key=f"batch_manual_end_{uploader_version}",
                step=60,
            )

        detected_timezone = _extract_timezone_label(timestamp_reference_text)
        detected_timezone_info = _timezone_info_from_label(detected_timezone)
        location_timezone_info = _infer_indonesia_timezone_from_location(location_text_input)

        timezone_source_options = []
        if detected_timezone_info:
            timezone_source_options.append(
                f"Auto dari Timestamp/OCR — {detected_timezone_info['source_label']}"
            )
        if location_timezone_info:
            timezone_source_options.append(
                "Auto dari Lokasi — "
                f"{location_timezone_info['indonesia_label']} "
                f"({location_timezone_info['gmt_label']})"
            )
        timezone_source_options.append("Pilih Manual")

        timezone_source = st.radio(
            "Sumber Zona Waktu",
            timezone_source_options,
            horizontal=True,
            key=f"batch_manual_timezone_source_{uploader_version}",
            help=(
                "Prioritas paling praktis: ikuti Timestamp/OCR. Jika tidak ada label timezone, "
                "program bisa membaca provinsi Indonesia dari teks lokasi. Manual override tetap tersedia."
            ),
        )

        auto_timezone_info = None
        auto_source_label = ""

        if timezone_source.startswith("Auto dari Timestamp/OCR"):
            auto_timezone_info = detected_timezone_info
            auto_source_label = "Timestamp/OCR"
        elif timezone_source.startswith("Auto dari Lokasi"):
            auto_timezone_info = location_timezone_info
            auto_source_label = "Lokasi"

        if auto_timezone_info:
            label_choices = [
                f"Ikuti Sumber — {auto_timezone_info['source_label']}"
                if auto_source_label == "Timestamp/OCR"
                else f"Indonesia — {auto_timezone_info['indonesia_label']}",
            ]

            if auto_timezone_info.get("indonesia_label"):
                indonesia_choice = f"Indonesia — {auto_timezone_info['indonesia_label']}"
                if indonesia_choice not in label_choices:
                    label_choices.append(indonesia_choice)

            label_choices.extend([
                f"GMT — {auto_timezone_info['gmt_label']}",
                f"UTC — {auto_timezone_info['utc_label']}",
            ])

            label_format = st.radio(
                "Format Label Zona Waktu",
                label_choices,
                horizontal=True,
                key=f"batch_manual_timezone_label_format_{uploader_version}",
                help=(
                    "Offset waktunya tidak diubah. Pilihan ini hanya menentukan tulisan pada foto, "
                    "misalnya WITA, GMT+8, atau UTC+8."
                ),
            )

            if label_format.startswith("Ikuti Sumber"):
                timezone_text = auto_timezone_info["source_label"]
            elif label_format.startswith("Indonesia"):
                timezone_text = auto_timezone_info["indonesia_label"]
            elif label_format.startswith("GMT"):
                timezone_text = auto_timezone_info["gmt_label"]
            else:
                timezone_text = auto_timezone_info["utc_label"]

            st.success(
                f"🕒 Zona waktu otomatis dari **{auto_source_label}**: **{timezone_text}**"
            )
        else:
            timezone_options = _timezone_options()
            manual_default = "WIB"
            if detected_timezone in timezone_options:
                manual_default = detected_timezone
            elif location_timezone_info and location_timezone_info.get("indonesia_label") in timezone_options:
                manual_default = location_timezone_info["indonesia_label"]

            timezone_text = st.selectbox(
                "Zona Waktu Manual / Override",
                timezone_options,
                index=timezone_options.index(manual_default),
                key=f"batch_manual_tz_{uploader_version}",
                help="Pilihan manual lengkap: WIB, WITA, WIT, GMT, UTC, serta offset GMT/UTC.",
            )

        if not detected_timezone_info and not location_timezone_info:
            st.info(
                "ℹ️ Zona waktu belum bisa dideteksi otomatis dari Timestamp/OCR maupun teks lokasi. "
                "Gunakan pilihan manual."
            )

        st.caption(
            "Kamu cukup mengatur **tanggal + jam mulai + jam selesai**. Zona waktu bisa otomatis dari OCR, "
            "fallback dari provinsi pada teks lokasi, atau di-override manual. Format label WIB/WITA/WIT, GMT, dan UTC tetap bisa dipilih."
        )
        st.caption("Jam diacak di dalam rentang lalu diurutkan naik sesuai file 1 → 2 → 3 → seterusnya.")

    st.markdown("### 4️⃣ Atur Gaya Timestamp & Variasi Koordinat")
    with st.container(border=True):
        st.caption(
            "Font yang dipakai: **AVHershey Simplex Medium**. Bold sekarang bisa diatur tingkat ketebalannya sendiri."
        )

        size_mode_label = st.selectbox(
            "Metode Ukuran Timestamp",
            [
                "Smart Uniform (Sisi Terpendek) — Recommended",
                "% Lebar Foto (Legacy)",
                "Fixed Pixel",
            ],
            index=0,
            key=f"batch_manual_size_mode_{uploader_version}",
            help="Smart Uniform membuat portrait dan landscape dengan resolusi setara terlihat konsisten.",
        )

        if size_mode_label.startswith("Smart Uniform"):
            batch_font_size_mode = "short_edge"
        elif size_mode_label.startswith("% Lebar"):
            batch_font_size_mode = "width"
        else:
            batch_font_size_mode = "fixed_px"

        style_col1, style_col2 = st.columns([2, 1])
        with style_col1:
            if batch_font_size_mode == "fixed_px":
                batch_fixed_font_size_px = st.number_input(
                    "Ukuran Timestamp (px)",
                    min_value=14,
                    max_value=400,
                    value=80,
                    step=1,
                    key=f"batch_manual_fixed_size_px_{uploader_version}",
                )
                batch_font_size_percent = 5.0
            else:
                batch_font_size_percent = st.slider(
                    "Ukuran Timestamp (%)",
                    min_value=2.0,
                    max_value=8.0,
                    value=5.0,
                    step=0.1,
                    key=f"batch_manual_font_size_percent_{uploader_version}",
                    help="Smart Uniform = persen dari sisi terpendek. Legacy = persen dari lebar foto.",
                )
                batch_fixed_font_size_px = 80

        with style_col2:
            batch_bold_enabled = st.toggle(
                "Bold",
                value=False,
                key=f"batch_manual_bold_enabled_{uploader_version}",
                help="Aktifkan penebalan font. Ketebalannya bisa diatur di bawah.",
            )

        lock_batch_font_size = st.toggle(
            "Kunci ukuran karakter seluruh batch (True Uniform)",
            value=True,
            key=f"batch_manual_lock_size_{uploader_version}",
            help="Recommended. Semua foto memakai ukuran karakter px yang sama. Jika baris terlalu panjang pada foto sempit, teks dibungkus ke baris berikutnya, bukan dikecilkan.",
        )

        if batch_bold_enabled:
            batch_bold_strength_percent = st.slider(
                "Ketebalan Bold",
                min_value=1,
                max_value=100,
                value=25,
                step=1,
                key=f"batch_manual_bold_strength_{uploader_version}",
                help="1 = sangat tipis, 100 = paling tebal.",
            )
        else:
            batch_bold_strength_percent = 25

        jitter_enabled = st.toggle(
            "Acak sedikit koordinat per foto",
            value=False,
            key=f"batch_manual_jitter_enabled_{uploader_version}",
            help="Menambahkan offset kecil di sekitar titik dasar supaya koordinat antar foto tidak identik.",
        )
        if jitter_enabled:
            jitter_radius_m = st.slider(
                "Radius maksimum acak koordinat (meter)",
                min_value=1,
                max_value=30,
                value=8,
                step=1,
                key=f"batch_manual_jitter_radius_{uploader_version}",
            )
        else:
            jitter_radius_m = 0

        size_preview = (
            f"{int(batch_fixed_font_size_px)} px"
            if batch_font_size_mode == "fixed_px"
            else f"{batch_font_size_percent:.1f}%"
        )
        st.caption(
            f"Preview setting: Font **AVHershey Simplex Medium** | Metode **{size_mode_label}** | Ukuran **{size_preview}** | "
            f"True Uniform: **{'Aktif' if lock_batch_font_size else 'Nonaktif'}** | "
            f"Bold: **{'Aktif' if batch_bold_enabled else 'Nonaktif'}**"
            + (f" | Ketebalan: **{batch_bold_strength_percent}%**" if batch_bold_enabled else "")
            + f" | Variasi koordinat: **{'Aktif' if jitter_enabled else 'Nonaktif'}**"
        )

    st.markdown("### 5️⃣ Generate Batch")
    batch_ready = bool(sorted_files) and not name_errors and manual_location_ready

    with st.container(border=True):
        if not batch_ready:
            st.info(
                "Lengkapi foto bernomor serta koordinat/lokasi manual yang valid untuk mengaktifkan generate."
            )

        clicked = st.button(
            "🚀 GENERATE SEMUA FOTO",
            type="primary",
            disabled=not batch_ready,
            use_container_width=True,
            key=f"batch_manual_generate_{uploader_version}",
        )

        if clicked:
            try:
                random_datetimes = generate_ordered_random_datetimes(
                    batch_date, batch_start, batch_end, len(sorted_files)
                )

                locked_font_size_px = None
                if lock_batch_font_size:
                    locked_font_size_px, _ = _calculate_locked_font_size_px(
                        sorted_files,
                        load_image_fn,
                        batch_font_size_mode,
                        batch_font_size_percent,
                        batch_fixed_font_size_px,
                    )

                zip_buffer = io.BytesIO()
                log_rows = []
                first_preview = None
                first_preview_name = None

                with st.spinner(f"🖌️ Memproses {len(sorted_files)} foto..."):
                    with zipfile.ZipFile(
                        zip_buffer,
                        mode="w",
                        compression=zipfile.ZIP_DEFLATED,
                    ) as zip_file:
                        for uploaded_file, generated_dt in zip(sorted_files, random_datetimes):
                            uploaded_file.seek(0)
                            photo_pil = load_image_fn(uploaded_file)
                            photo_cv = np.array(photo_pil)

                            actual_latitude, actual_longitude, actual_offset_m = _jitter_single_coordinate(
                                manual_latitude,
                                manual_longitude,
                                jitter_radius_m if jitter_enabled else 0,
                            )
                            coordinate_text = f"{actual_latitude:.7f}, {actual_longitude:.7f}"
                            date_text = (
                                f"{generated_dt.day}/{generated_dt.month}/{generated_dt.year} "
                                f"{generated_dt.strftime('%H:%M:%S')} {timezone_text}"
                            )

                            lines = [
                                coordinate_text,
                                _collapse_spaces(location_text_input),
                                date_text,
                            ] + extra_lines
                            raw_timestamp = "\n".join(lines)

                            result_pil, info = generate_timestamp_fn(
                                photo_cv,
                                raw_timestamp,
                                font_size_percent=batch_font_size_percent,
                                font_size_mode=batch_font_size_mode,
                                fixed_font_size_px=batch_fixed_font_size_px,
                                font_size_px_override=locked_font_size_px if lock_batch_font_size else None,
                                preserve_locked_font_size=lock_batch_font_size,
                                bold_enabled=batch_bold_enabled,
                                bold_strength_percent=batch_bold_strength_percent,
                            )
                            encoded_bytes, _ = _encode_pil_image_bytes(result_pil, uploaded_file.name)
                            zip_file.writestr(uploaded_file.name, encoded_bytes)

                            if first_preview is None:
                                first_preview = encoded_bytes
                                first_preview_name = uploaded_file.name

                            log_rows.append(
                                {
                                    "Nama File": uploaded_file.name,
                                    "Latitude": f"{actual_latitude:.7f}",
                                    "Longitude": f"{actual_longitude:.7f}",
                                    "Latitude Dasar": f"{manual_latitude:.7f}",
                                    "Longitude Dasar": f"{manual_longitude:.7f}",
                                    "Offset Koordinat (meter)": round(actual_offset_m, 2),
                                    "Daerah": _collapse_spaces(location_text_input),
                                    "Tanggal": generated_dt.strftime("%d/%m/%Y"),
                                    "Jam": generated_dt.strftime("%H:%M:%S"),
                                    "Zona Waktu": timezone_text,
                                    "Jumlah Baris": info.get("line_count", len(lines)),
                                    "Font": "AVHershey Simplex Medium",
                                    "Bold": "Ya" if batch_bold_enabled else "Tidak",
                                    "Ketebalan Bold (%)": batch_bold_strength_percent if batch_bold_enabled else 0,
                                    "Ukuran Timestamp (%)": batch_font_size_percent,
                                    "Baris Tambahan": " | ".join(extra_lines),
                                    "Sumber Lokasi": "Input manual user",
                                    "Sumber Waktu": "Generated/random dari rentang input; bukan EXIF asli",
                                }
                            )

                st.session_state[state_key] = {
                    "zip_bytes": zip_buffer.getvalue(),
                    "zip_name": "hasil_timestamp_batch_manual.zip",
                    "log_rows": log_rows,
                    "preview_bytes": first_preview,
                    "preview_name": first_preview_name,
                }
                st.success(f"✅ Selesai memproses {len(sorted_files)} foto.")
            except Exception as error:
                st.error(f"❌ Gagal generate batch: {error}")

    result = st.session_state.get(state_key)
    if result is not None:
        st.markdown("---")
        st.markdown("### ✨ Hasil Batch")
        with st.container(border=True):
            st.success("Timestamp batch manual berhasil dibuat. Foto siap diunduh dalam ZIP.")

            result_df = pd.DataFrame(result["log_rows"])
            st.dataframe(
                result_df[
                    [
                        "Nama File",
                        "Tanggal",
                        "Jam",
                        "Latitude",
                        "Longitude",
                        "Offset Koordinat (meter)",
                        "Daerah",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

            if result.get("preview_bytes"):
                st.image(
                    result["preview_bytes"],
                    caption=f"Preview foto pertama: {result['preview_name']}",
                    use_container_width=True,
                )

            c_zip, c_reset = st.columns([3, 1])
            with c_zip:
                st.download_button(
                    "⬇️ DOWNLOAD SEMUA FOTO (ZIP)",
                    data=result["zip_bytes"],
                    file_name=result["zip_name"],
                    mime="application/zip",
                    type="primary",
                    use_container_width=True,
                )
            with c_reset:
                if st.button("🔄 Reset Batch", use_container_width=True, key=f"batch_manual_reset_{uploader_version}"):
                    st.session_state[state_key] = None
                    st.session_state.uploader_version += 1
                    st.rerun()
