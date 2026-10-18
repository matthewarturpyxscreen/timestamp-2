import io
import math
import os
import re
import secrets
import zipfile
from datetime import datetime, date, time, timedelta
from urllib.parse import quote

import cv2
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

try:
    import streamlit as st
except ImportError:  # hanya untuk test helper di environment tanpa Streamlit
    st = None


# ============================================================
# ONLINE NPSN CONFIG
# ============================================================

KEMENDIKDASMEN_NPSN_URL = (
    "https://referensi.data.kemendikdasmen.go.id/pendidikan/npsn/{npsn}"
)
NOMINATIM_REVERSE_URL = "https://nominatim.openstreetmap.org/reverse"
HTTP_TIMEOUT_SECONDS = 15
HTTP_HEADERS = {
    "User-Agent": "AutoTimestamp-NPSN/8.3.1 (Streamlit; NPSN location lookup)",
    "Accept-Language": "id-ID,id;q=0.9,en;q=0.6",
}


# ============================================================
# NORMALIZATION HELPERS
# ============================================================


def _normalize_npsn(value):
    if value is None:
        return ""
    value = str(value).strip().upper()
    return re.sub(r"\s+", "", value)


def _validate_npsn_format(npsn):
    npsn = _normalize_npsn(npsn)
    if not npsn:
        raise ValueError("NPSN belum diisi.")
    if not re.fullmatch(r"[A-Z0-9]{8}", npsn):
        raise ValueError(
            "Format NPSN harus 8 karakter huruf/angka tanpa spasi. "
            "Contoh: 20212345."
        )
    return npsn


def _collapse_spaces(value):
    return re.sub(r"\s+", " ", str(value or "").strip())


def _normalize_label(value):
    return re.sub(r"[^a-z0-9]+", "", _collapse_spaces(value).lower())


def _clean_region_prefix(value):
    value = _collapse_spaces(value)
    if not value:
        return ""

    # Prefix mengikuti format yang umum tampil di Referensi Data.
    # Dilakukan berulang supaya kombinasi seperti "PROV. D.K.I. ..."
    # tetap bersih tanpa menghapus bagian nama wilayahnya.
    patterns = [
        r"^PROVINSI\s+",
        r"^PROPINSI\s+",
        r"^PROV\.\s*",
        r"^KABUPATEN\s+",
        r"^KAB\.\s*",
        r"^KAB\s+",
        r"^KOTA\s+ADMINISTRASI\s+",
        r"^KOTA\s+ADM\.\s*",
        r"^KOTA\s+",
        r"^KECAMATAN\s+",
        r"^KEC\.\s*",
        r"^KEC\s+",
        r"^KELURAHAN\s+",
        r"^KEL\.\s*",
        r"^DESA\s+",
    ]

    previous = None
    while previous != value:
        previous = value
        for pattern in patterns:
            value = re.sub(pattern, "", value, flags=re.IGNORECASE).strip()

    return value


def _pretty_region(value):
    value = _clean_region_prefix(value)
    if not value:
        return ""
    # title() mempertahankan bentuk seperti D.K.I. Jakarta dengan cukup baik,
    # sekaligus mengubah "PURWAKARTA" menjadi "Purwakarta".
    return value.title()


def _parse_coordinate(value, field_name):
    raw = _collapse_spaces(value)
    if not raw:
        raise ValueError(f"{field_name} kosong pada data NPSN resmi.")

    # Beberapa sumber menulis koma sebagai pemisah desimal.
    if "," in raw and "." not in raw:
        raw = raw.replace(",", ".")

    match = re.search(r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)", raw)
    if not match:
        raise ValueError(f"{field_name} tidak valid: {value}")

    number = float(match.group(0))

    if field_name.lower().startswith("lat") and not (-90 <= number <= 90):
        raise ValueError(f"Latitude di luar rentang: {number}")
    if field_name.lower().startswith("long") and not (-180 <= number <= 180):
        raise ValueError(f"Longitude di luar rentang: {number}")

    return number


def _extract_token_value(tokens, aliases):
    """Ambil nilai setelah label dari token HTML yang sudah dibersihkan.

    Halaman Referensi Data kadang memisahkan label, ':', dan nilai menjadi
    elemen HTML terpisah. Helper ini mendukung kedua bentuk tersebut.
    """
    alias_norms = [(_normalize_label(alias), alias) for alias in aliases]

    for index, token in enumerate(tokens):
        token_text = _collapse_spaces(token)
        token_norm = _normalize_label(token_text)

        for alias_norm, alias in alias_norms:
            # Bentuk: "Lintang: -6.1234" / "NPSN : 20212345"
            if token_norm.startswith(alias_norm) and len(token_norm) > len(alias_norm):
                match = re.match(
                    rf"^{re.escape(alias)}\s*:?\s*(.+)$",
                    token_text,
                    flags=re.IGNORECASE,
                )
                if match and _collapse_spaces(match.group(1)):
                    return _collapse_spaces(match.group(1))

            # Bentuk: token label sendiri, lalu ':' dan nilai di token berikutnya.
            if token_norm == alias_norm:
                for next_index in range(index + 1, min(index + 5, len(tokens))):
                    candidate = _collapse_spaces(tokens[next_index])
                    if not candidate or candidate in {":", "-"}:
                        continue
                    if _normalize_label(candidate) in {norm for norm, _ in alias_norms}:
                        break
                    return candidate.lstrip(":").strip()

    return ""


# ============================================================
# OFFICIAL KEMENDIKDASMEN LOOKUP
# ============================================================


def _parse_kemendikdasmen_npsn_html(html_text, requested_npsn, source_url):
    soup = BeautifulSoup(html_text, "html.parser")
    tokens = list(soup.stripped_strings)
    page_text = "\n".join(tokens)

    page_npsn = _extract_token_value(tokens, ["NPSN"])
    page_npsn = _normalize_npsn(page_npsn)

    if not page_npsn:
        raise ValueError(
            "NPSN tidak ditemukan pada halaman Referensi Data. "
            "Kemungkinan NPSN tidak valid atau struktur situs sedang berubah."
        )

    if page_npsn != requested_npsn:
        raise ValueError(
            f"Validasi exact-match gagal. Diminta {requested_npsn}, "
            f"tetapi halaman mengembalikan {page_npsn}."
        )

    desa = _extract_token_value(
        tokens,
        ["Desa/Kelurahan", "Desa / Kelurahan", "Kelurahan/Desa"],
    )
    kecamatan = _extract_token_value(
        tokens,
        ["Kecamatan/Kota (LN)", "Kecamatan", "Kec."],
    )
    kabupaten_kota = _extract_token_value(
        tokens,
        [
            "Kab.-Kota/Negara (LN)",
            "Kabupaten/Kota",
            "Kab.-Kota",
            "Kabupaten",
            "Kota",
        ],
    )
    provinsi = _extract_token_value(
        tokens,
        ["Propinsi/Luar Negeri (LN)", "Provinsi/Luar Negeri (LN)", "Provinsi", "Propinsi"],
    )

    # Koordinat pada halaman resmi berada di bagian Peta dan biasanya berbentuk
    # "Lintang: -6.x" / "Bujur: 107.x". Regex menjadi fallback paling kuat.
    lat_match = re.search(
        r"Lintang\s*:?\s*([+-]?(?:\d+(?:\.\d+)?|\.\d+))",
        page_text,
        flags=re.IGNORECASE,
    )
    lon_match = re.search(
        r"Bujur\s*:?\s*([+-]?(?:\d+(?:\.\d+)?|\.\d+))",
        page_text,
        flags=re.IGNORECASE,
    )

    if not lat_match:
        lat_value = _extract_token_value(tokens, ["Lintang", "Latitude"])
    else:
        lat_value = lat_match.group(1)

    if not lon_match:
        lon_value = _extract_token_value(tokens, ["Bujur", "Longitude"])
    else:
        lon_value = lon_match.group(1)

    latitude = _parse_coordinate(lat_value, "Latitude")
    longitude = _parse_coordinate(lon_value, "Longitude")

    # 0,0 adalah placeholder/residu yang tidak layak dijadikan acuan lokasi.
    if abs(latitude) < 1e-12 and abs(longitude) < 1e-12:
        raise ValueError(
            "Koordinat resmi NPSN masih 0,0 / belum valid. "
            "Program tidak akan membuat timestamp dari titik tersebut."
        )

    desa_pretty = _pretty_region(desa)
    kecamatan_pretty = _pretty_region(kecamatan)
    kab_pretty = _pretty_region(kabupaten_kota)
    prov_pretty = _pretty_region(provinsi)

    first_part = desa_pretty or kecamatan_pretty
    location_parts = [part for part in [first_part, kab_pretty, prov_pretty] if part]

    if len(location_parts) < 2:
        raise ValueError(
            "Data wilayah dari Referensi Data belum cukup untuk membentuk lokasi timestamp."
        )

    location = ", ".join(location_parts)

    return {
        "npsn": requested_npsn,
        "latitude": latitude,
        "longitude": longitude,
        "location": location,
        "desa": desa_pretty,
        "kecamatan": kecamatan_pretty,
        "kabupaten_kota": kab_pretty,
        "provinsi": prov_pretty,
        "source_url": source_url,
        "source_name": "Referensi Data Kemendikdasmen",
    }


def lookup_npsn_online(npsn):
    npsn = _validate_npsn_format(npsn)
    source_url = KEMENDIKDASMEN_NPSN_URL.format(npsn=quote(npsn, safe=""))

    try:
        response = requests.get(
            source_url,
            headers=HTTP_HEADERS,
            timeout=HTTP_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        raise ValueError(
            "Tidak dapat terhubung ke Referensi Data Kemendikdasmen. "
            "Periksa koneksi internet lalu coba lagi."
        ) from exc

    if response.status_code != 200:
        raise ValueError(
            f"Referensi Data Kemendikdasmen mengembalikan HTTP {response.status_code}. "
            "Coba lagi beberapa saat."
        )

    return _parse_kemendikdasmen_npsn_html(
        response.text,
        requested_npsn=npsn,
        source_url=source_url,
    )


# ============================================================
# SECOND-LAYER LOCATION VALIDATION (OPENSTREETMAP / NOMINATIM)
# ============================================================


def _normalize_region_for_compare(value):
    value = _collapse_spaces(value).lower()
    value = value.replace("d.k.i.", "dki").replace("d.i.", "di")
    value = re.sub(r"[^a-z0-9 ]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()

    aliases = {
        "dki jakarta": "jakarta",
        "daerah khusus ibukota jakarta": "jakarta",
        "daerah khusus jakarta": "jakarta",
        "di yogyakarta": "yogyakarta",
        "daerah istimewa yogyakarta": "yogyakarta",
    }
    if value in aliases:
        return aliases[value]

    stop_words = {
        "kabupaten",
        "kab",
        "kota",
        "provinsi",
        "propinsi",
        "province",
        "regency",
        "city",
        "municipality",
    }
    words = [word for word in value.split() if word not in stop_words]
    return " ".join(words)


def _region_matches(official_value, reverse_value):
    official = _normalize_region_for_compare(official_value)
    reverse = _normalize_region_for_compare(reverse_value)
    if not official or not reverse:
        return None
    return official == reverse or official in reverse or reverse in official


def reverse_validate_location(
    latitude,
    longitude,
    official_desa,
    official_kecamatan,
    official_kabupaten,
    official_provinsi,
):
    params = {
        "format": "jsonv2",
        "lat": f"{latitude:.8f}",
        "lon": f"{longitude:.8f}",
        "zoom": 10,
        "addressdetails": 1,
        "accept-language": "id",
    }

    try:
        response = requests.get(
            NOMINATIM_REVERSE_URL,
            params=params,
            headers=HTTP_HEADERS,
            timeout=HTTP_TIMEOUT_SECONDS,
        )
        if response.status_code != 200:
            raise requests.RequestException(f"HTTP {response.status_code}")
        payload = response.json()
    except Exception as exc:
        return {
            "status": "unavailable",
            "message": "Validasi peta kedua sedang tidak tersedia.",
            "reverse_desa": "",
            "reverse_kecamatan": "",
            "reverse_kabupaten": "",
            "reverse_provinsi": "",
            "display_name": "",
            "error": str(exc),
        }

    address = payload.get("address") or {}
    reverse_desa = (
        address.get("village")
        or address.get("suburb")
        or address.get("hamlet")
        or address.get("neighbourhood")
        or ""
    )
    reverse_kecamatan = (
        address.get("city_district")
        or address.get("subdistrict")
        or address.get("municipality")
        or address.get("district")
        or ""
    )
    reverse_kabupaten = (
        address.get("county")
        or address.get("city")
        or address.get("municipality")
        or address.get("town")
        or address.get("state_district")
        or ""
    )
    reverse_provinsi = address.get("state") or address.get("region") or ""
    display_name = _collapse_spaces(payload.get("display_name", ""))

    desa_match = _region_matches(official_desa, reverse_desa)
    kec_match = _region_matches(official_kecamatan, reverse_kecamatan)
    kab_match = _region_matches(official_kabupaten, reverse_kabupaten)
    prov_match = _region_matches(official_provinsi, reverse_provinsi)

    explicit_mismatches = []
    if desa_match is False:
        explicit_mismatches.append("desa/kelurahan")
    if kec_match is False:
        explicit_mismatches.append("kecamatan")
    if kab_match is False:
        explicit_mismatches.append("kabupaten/kota")
    if prov_match is False:
        explicit_mismatches.append("provinsi")

    if explicit_mismatches:
        return {
            "status": "mismatch",
            "message": (
                "Koordinat resmi tidak cocok dengan hasil reverse-geocoding pada "
                + " dan ".join(explicit_mismatches)
                + ". Generate diblokir agar tidak memakai lokasi yang meragukan."
            ),
            "reverse_desa": reverse_desa,
            "reverse_kecamatan": reverse_kecamatan,
            "reverse_kabupaten": reverse_kabupaten,
            "reverse_provinsi": reverse_provinsi,
            "display_name": display_name,
            "desa_match": desa_match,
            "kec_match": kec_match,
            "kab_match": kab_match,
            "prov_match": prov_match,
        }

    if kab_match is True and prov_match is True:
        status = "verified"
        message = (
            "Kabupaten/kota dan provinsi cocok dengan posisi koordinat. "
            "Kecamatan/desa tidak menunjukkan konflik."
        )
    else:
        status = "partial"
        message = (
            "Validator peta belum bisa memverifikasi seluruh wilayah wajib. "
            "Generate diblokir agar hanya data yang benar-benar terverifikasi yang dipakai."
        )

    return {
        "status": status,
        "message": message,
        "reverse_desa": reverse_desa,
        "reverse_kecamatan": reverse_kecamatan,
        "reverse_kabupaten": reverse_kabupaten,
        "reverse_provinsi": reverse_provinsi,
        "display_name": display_name,
        "desa_match": desa_match,
        "kec_match": kec_match,
        "kab_match": kab_match,
        "prov_match": prov_match,
    }


if st is not None:
    # Cache mengurangi request berulang saat Streamlit rerun.
    lookup_npsn_online_cached = st.cache_data(ttl=21600, show_spinner=False)(lookup_npsn_online)
    reverse_validate_location_cached = st.cache_data(ttl=21600, show_spinner=False)(
        reverse_validate_location
    )
else:
    lookup_npsn_online_cached = lookup_npsn_online
    reverse_validate_location_cached = reverse_validate_location


# ============================================================
# PHOTO / TIME HELPERS
# ============================================================


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


# ============================================================
# COORDINATE JITTER HELPERS
# ============================================================


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


# ============================================================
# STREAMLIT BATCH MODE
# ============================================================


def render_batch_mode(uploader_version, load_image_fn, generate_timestamp_fn):
    if st is None:
        raise RuntimeError("Streamlit belum terpasang. Jalankan pip install -r requirements.txt")

    state_key = "batch_generated_result"
    lookup_state_key = "batch_npsn_online_result"

    if state_key not in st.session_state:
        st.session_state[state_key] = None
    if lookup_state_key not in st.session_state:
        st.session_state[lookup_state_key] = None

    st.markdown(
        '<div class="hero-badge">⚡ V8.3.1 • Batch NPSN Online + Validasi Lokasi Ketat</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="hero-title">🗂️ Batch Timestamp dari NPSN</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="hero-desc">Upload foto bernama <b>1, 2, 3, ...</b>, masukkan satu NPSN, '
        'ambil koordinat & daerah secara online dari <b>Referensi Data Kemendikdasmen</b>, '
        'validasi posisi, atur gaya tulisan, lalu buat jam acak yang tetap berurutan.</div>',
        unsafe_allow_html=True,
    )

    st.warning(
        "⚠️ Jam pada mode ini dibuat dari rentang yang kamu pilih, bukan waktu pengambilan "
        "asli/EXIF. Program hanya menggambar overlay visual dan tidak mengubah metadata EXIF asli."
    )

    # --------------------------------------------------------
    # 1. PHOTO UPLOAD
    # --------------------------------------------------------
    st.markdown("### 1️⃣ Upload Foto Batch")
    with st.container(border=True):
        batch_files = st.file_uploader(
            "Pilih beberapa foto (JPG, JPEG, PNG)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key=f"batch_photo_uploader_{uploader_version}",
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

    # --------------------------------------------------------
    # 2. ONLINE NPSN LOOKUP
    # --------------------------------------------------------
    st.markdown("### 2️⃣ Ambil Lokasi dari NPSN Online")
    with st.container(border=True):
        npsn_value = st.text_input(
            "Masukkan NPSN",
            placeholder="Contoh: 20212345",
            key=f"batch_npsn_{uploader_version}",
            help="NPSN dipakai sebagai exact-match ke Referensi Data Kemendikdasmen. Nama sekolah tidak ditampilkan pada foto.",
        )
        current_npsn = _normalize_npsn(npsn_value)

        lookup_clicked = st.button(
            "🔎 CEK NPSN & AMBIL LOKASI ONLINE",
            type="secondary",
            disabled=not bool(current_npsn),
            use_container_width=True,
            key=f"batch_lookup_npsn_{uploader_version}",
        )

        if lookup_clicked:
            st.session_state[state_key] = None
            try:
                with st.spinner("Mengambil data resmi Kemendikdasmen dan memvalidasi koordinat..."):
                    official_info = lookup_npsn_online_cached(current_npsn)
                    validation = reverse_validate_location_cached(
                        official_info["latitude"],
                        official_info["longitude"],
                        official_info["desa"],
                        official_info["kecamatan"],
                        official_info["kabupaten_kota"],
                        official_info["provinsi"],
                    )
                    official_info["validation"] = validation
                    st.session_state[lookup_state_key] = official_info
            except Exception as error:
                st.session_state[lookup_state_key] = None
                st.error(f"❌ Gagal mengambil NPSN: {error}")

        npsn_info = st.session_state.get(lookup_state_key)
        if npsn_info and npsn_info.get("npsn") != current_npsn:
            st.info("NPSN berubah. Klik **CEK NPSN & AMBIL LOKASI ONLINE** untuk memuat data baru.")
            npsn_info = None

        validation_ok = False
        if npsn_info is not None:
            validation = npsn_info.get("validation", {})
            validation_status = validation.get("status", "unavailable")
            validation_ok = validation_status == "verified"

            st.success("✅ NPSN exact-match ditemukan pada Referensi Data Kemendikdasmen.")
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Latitude / Lintang", f"{npsn_info['latitude']:.7f}")
            with c2:
                st.metric("Longitude / Bujur", f"{npsn_info['longitude']:.7f}")

            st.markdown(f"📍 **Daerah untuk timestamp:** `{npsn_info['location']}`")
            st.caption(
                "Format daerah: Desa/Kelurahan (fallback Kecamatan) + Kabupaten/Kota + Provinsi. "
                "Nama sekolah tidak dimasukkan."
            )

            if validation_status == "verified":
                st.success("🗺️ Validasi lokasi kedua: " + validation.get("message", "Cocok."))
            elif validation_status == "partial":
                st.warning("🗺️ Validasi lokasi kedua: " + validation.get("message", "Sebagian."))
            elif validation_status == "unavailable":
                st.warning(
                    "🗺️ Validasi lokasi kedua tidak tersedia saat ini. "
                    "Generate diblokir sampai validator kedua berhasil agar akurasi tetap ketat."
                )
            elif validation_status == "mismatch":
                st.error("🛑 " + validation.get("message", "Lokasi tidak cocok."))
                if validation.get("reverse_kabupaten") or validation.get("reverse_provinsi"):
                    st.caption(
                        "Hasil validator peta: "
                        f"{validation.get('reverse_kabupaten', '-')}, "
                        f"{validation.get('reverse_provinsi', '-')}"
                    )

            st.caption(f"Sumber utama: {npsn_info['source_url']}")
            st.caption(
                "Hasil validator kedua → "
                f"Desa: {validation.get('reverse_desa', '-') or '-'} | "
                f"Kecamatan: {validation.get('reverse_kecamatan', '-') or '-'} | "
                f"Kab/Kota: {validation.get('reverse_kabupaten', '-') or '-'} | "
                f"Provinsi: {validation.get('reverse_provinsi', '-') or '-'}"
            )

    # --------------------------------------------------------
    # 3. DATE / TIME RANGE
    # --------------------------------------------------------
    st.markdown("### 3️⃣ Atur Tanggal & Rentang Jam")
    with st.container(border=True):
        c_date, c_start, c_end = st.columns(3)
        with c_date:
            batch_date = st.date_input(
                "Tanggal",
                value=date.today(),
                key=f"batch_date_{uploader_version}",
                format="DD/MM/YYYY",
            )
        with c_start:
            batch_start = st.time_input(
                "Jam Mulai",
                value=time(13, 0, 0),
                key=f"batch_start_{uploader_version}",
                step=60,
            )
        with c_end:
            batch_end = st.time_input(
                "Jam Selesai",
                value=time(16, 0, 0),
                key=f"batch_end_{uploader_version}",
                step=60,
            )

        timezone_text = st.selectbox(
            "Zona Waktu",
            ["WIB", "WITA", "WIT"],
            index=0,
            key=f"batch_tz_{uploader_version}",
        )
        st.caption(
            "Jam diacak di dalam rentang lalu diurutkan naik sesuai file 1 → 2 → 3 → seterusnya."
        )

    # --------------------------------------------------------
    # 4. STYLE + COORDINATE OPTIONS
    # --------------------------------------------------------
    st.markdown("### 4️⃣ Atur Gaya Timestamp & Variasi Koordinat")
    with st.container(border=True):
        st.caption(
            "Font yang dipakai: **AVHershey Simplex Medium**. Bold sekarang bisa diatur tingkat ketebalannya sendiri."
        )

        style_col1, style_col2 = st.columns([2, 1])
        with style_col1:
            batch_font_size_percent = st.slider(
                "Ukuran Timestamp (% dari lebar foto)",
                min_value=2.0,
                max_value=8.0,
                value=5.0,
                step=0.1,
                key=f"batch_font_size_percent_{uploader_version}",
                help="Semakin besar nilainya, semakin besar tulisan timestamp. Auto-fit tetap aktif jika teks terlalu panjang.",
            )
        with style_col2:
            batch_bold_enabled = st.toggle(
                "Bold",
                value=False,
                key=f"batch_bold_enabled_{uploader_version}",
                help="Aktifkan penebalan font. Ketebalannya bisa diatur di bawah.",
            )

        if batch_bold_enabled:
            batch_bold_strength_percent = st.slider(
                "Ketebalan Bold",
                min_value=1,
                max_value=100,
                value=25,
                step=1,
                key=f"batch_bold_strength_{uploader_version}",
                help="1 = sangat tipis, 100 = paling tebal. Default 25 dibuat lebih halus dari versi sebelumnya.",
            )
        else:
            batch_bold_strength_percent = 25

        jitter_enabled = st.toggle(
            "Acak sedikit koordinat per foto",
            value=False,
            key=f"batch_jitter_enabled_{uploader_version}",
            help="Menambahkan offset kecil di sekitar titik resmi NPSN supaya koordinat antar foto tidak identik.",
        )
        if jitter_enabled:
            jitter_radius_m = st.slider(
                "Radius maksimum acak koordinat (meter)",
                min_value=1,
                max_value=30,
                value=8,
                step=1,
                key=f"batch_jitter_radius_{uploader_version}",
                help="Gunakan radius kecil agar koordinat tetap dekat dengan titik resmi NPSN.",
            )
            st.caption(
                "Offset koordinat akan dibuat acak dalam radius kecil dari titik resmi NPSN. Cocok jika kamu ingin setiap foto tidak memakai koordinat yang persis sama."
            )
        else:
            jitter_radius_m = 0
            st.caption("Koordinat akan memakai titik resmi NPSN secara persis untuk semua foto.")

        st.caption(
            f"Preview setting: Font **AVHershey Simplex Medium** | Ukuran **{batch_font_size_percent:.1f}%** | "
            f"Bold: **{'Aktif' if batch_bold_enabled else 'Nonaktif'}**"
            + (f" | Ketebalan: **{batch_bold_strength_percent}%**" if batch_bold_enabled else "")
            + f" | Variasi koordinat: **{'Aktif' if jitter_enabled else 'Nonaktif'}**"
        )

    # --------------------------------------------------------
    # 5. GENERATE
    # --------------------------------------------------------
    st.markdown("### 5️⃣ Generate Batch")
    batch_ready = (
        bool(sorted_files)
        and not name_errors
        and npsn_info is not None
        and validation_ok
    )

    with st.container(border=True):
        if not batch_ready:
            if npsn_info is not None and not validation_ok:
                st.error("Generate dinonaktifkan karena koordinat gagal validasi wilayah.")
            else:
                st.info(
                    "Lengkapi foto bernomor dan cek NPSN online sampai data lokasi valid "
                    "untuk mengaktifkan generate."
                )

        clicked = st.button(
            "🚀 GENERATE SEMUA FOTO",
            type="primary",
            disabled=not batch_ready,
            use_container_width=True,
        )

        if clicked:
            try:
                random_datetimes = generate_ordered_random_datetimes(
                    batch_date, batch_start, batch_end, len(sorted_files)
                )

                zip_buffer = io.BytesIO()
                log_rows = []
                first_preview = None
                first_preview_name = None
                validation = npsn_info.get("validation", {})

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
                                npsn_info["latitude"],
                                npsn_info["longitude"],
                                jitter_radius_m if jitter_enabled else 0,
                            )
                            date_text = (
                                f"{generated_dt.day}/{generated_dt.month}/{generated_dt.year} "
                                f"{generated_dt.strftime('%H:%M:%S')} {timezone_text}"
                            )

                            # Dibentuk langsung dari koordinat hasil per foto supaya tidak
                            # bergantung pada variabel sementara `coordinate_text`.
                            raw_timestamp = "\n".join(
                                [
                                    f"{actual_latitude:.7f}, {actual_longitude:.7f}",
                                    npsn_info["location"],
                                    date_text,
                                ]
                            )

                            result_pil, _ = generate_timestamp_fn(
                                photo_cv,
                                raw_timestamp,
                                font_size_percent=batch_font_size_percent,
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
                                    "NPSN": npsn_info["npsn"],
                                    "Latitude": f"{actual_latitude:.7f}",
                                    "Longitude": f"{actual_longitude:.7f}",
                                    "Offset Koordinat (meter)": round(actual_offset_m, 2),
                                    "Daerah": npsn_info["location"],
                                    "Tanggal": generated_dt.strftime("%d/%m/%Y"),
                                    "Jam": generated_dt.strftime("%H:%M:%S"),
                                    "Zona Waktu": timezone_text,
                                    "Bold": "Ya" if batch_bold_enabled else "Tidak",
                                    "Ketebalan Bold (%)": batch_bold_strength_percent if batch_bold_enabled else 0,
                                    "Ukuran Timestamp (%)": batch_font_size_percent,
                                    "Sumber Koordinat/Daerah": npsn_info["source_name"],
                                    "URL Sumber": npsn_info["source_url"],
                                    "Status Validasi Lokasi": validation.get("status", "unavailable"),
                                    "Sumber Waktu": "Generated/random dari rentang input; bukan EXIF asli",
                                }
                            )

                st.session_state[state_key] = {
                    "zip_bytes": zip_buffer.getvalue(),
                    "zip_name": f"hasil_timestamp_npsn_{npsn_info['npsn']}.zip",
                    "log_rows": log_rows,
                    "preview_bytes": first_preview,
                    "preview_name": first_preview_name,
                }
                st.success(f"✅ Selesai memproses {len(sorted_files)} foto.")
            except Exception as error:
                st.error(f"❌ Gagal generate batch: {error}")

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------
    result = st.session_state.get(state_key)
    if result is not None:
        st.markdown("---")
        st.markdown("### ✨ Hasil Batch")
        with st.container(border=True):
            st.success(
                "NPSN dan nama sekolah tidak ditampilkan pada foto. Overlay hanya berisi "
                "koordinat, daerah, tanggal, dan jam."
            )

            result_df = pd.DataFrame(result["log_rows"])
            st.dataframe(
                result_df[["Nama File", "Tanggal", "Jam", "Latitude", "Longitude", "Offset Koordinat (meter)", "Daerah"]],
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
                if st.button("🔄 Reset Batch", use_container_width=True):
                    st.session_state[state_key] = None
                    st.session_state[lookup_state_key] = None
                    st.session_state.uploader_version += 1
                    st.rerun()
