import io
import os
import re
import secrets
import zipfile
from datetime import datetime, date, time, timedelta
from difflib import SequenceMatcher
from urllib.parse import quote

import cv2
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

try:
    import streamlit as st
except ImportError:  # helper tetap bisa dites tanpa Streamlit
    st = None


OFFICIAL_NPSN_BASE_URL = "https://referensi.data.kemendikdasmen.go.id/pendidikan/npsn/"
NOMINATIM_REVERSE_URL = "https://nominatim.openstreetmap.org/reverse"
HTTP_TIMEOUT = 18
HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/154 Safari/537.36 AutoTimestampNPSN/7.7"
    )
}


# ============================================================
# NORMALISASI / PARSING DATA NPSN RESMI
# ============================================================


def _clean_cell(value):
    if value is None:
        return ""
    value = re.sub(r"\s+", " ", str(value)).strip()
    if value.lower() in {"nan", "none", "null", "nat", "-"}:
        return ""
    return value


def _normalize_npsn(value):
    value = _clean_cell(value).upper()
    if re.fullmatch(r"\d+\.0", value):
        value = value[:-2]
    return re.sub(r"[^A-Z0-9]", "", value)


def _normalize_label(value):
    value = _clean_cell(value).lower()
    value = value.replace("propinsi", "provinsi")
    value = value.replace("kab.-kota", "kabupaten kota")
    value = value.replace("kab-kota", "kabupaten kota")
    value = value.replace("/negara (ln)", "")
    value = value.replace("/luar negeri (ln)", "")
    value = value.replace("/kota (ln)", "")
    return re.sub(r"[^a-z0-9]+", "", value)


def _extract_table_value(soup, labels):
    wanted = {_normalize_label(label) for label in labels}

    # Jalur utama: tabel HTML resmi.
    for row in soup.find_all("tr"):
        cells = [
            _clean_cell(cell.get_text(" ", strip=True))
            for cell in row.find_all(["th", "td"])
        ]
        for idx, cell in enumerate(cells):
            if _normalize_label(cell) in wanted:
                for candidate in cells[idx + 1 :]:
                    candidate = _clean_cell(candidate)
                    if candidate and candidate != ":":
                        return candidate

    # Fallback: beberapa perubahan layout bisa membuat label/value
    # terpisah menjadi baris-baris teks, jadi scan urutan string.
    strings = [_clean_cell(x) for x in soup.stripped_strings]
    for idx, value in enumerate(strings):
        if _normalize_label(value) in wanted:
            for candidate in strings[idx + 1 : idx + 5]:
                candidate = _clean_cell(candidate)
                if candidate and candidate != ":":
                    return candidate
    return ""


def _extract_coordinate_from_html(soup, html, labels):
    # Coba elemen teks satu per satu dulu supaya tidak tertukar angka lain.
    label_pattern = "|".join(re.escape(x) for x in labels)
    coord_regex = re.compile(
        rf"(?:{label_pattern})\s*:?\s*([-+]?\d{{1,3}}(?:[\.,]\d+)?)",
        re.IGNORECASE,
    )

    for text_node in soup.stripped_strings:
        match = coord_regex.search(_clean_cell(text_node))
        if match:
            return match.group(1)

    plain_text = soup.get_text(" ", strip=True)
    match = coord_regex.search(plain_text)
    if match:
        return match.group(1)

    # Fallback paling akhir ke HTML mentah (tag mungkin berada di tengah).
    html_text = BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
    match = coord_regex.search(html_text)
    return match.group(1) if match else ""


def _parse_coordinate(value, field_name):
    raw = _clean_cell(value)
    if not raw:
        raise ValueError(f"{field_name} kosong pada data resmi NPSN.")

    # Desimal Indonesia kadang memakai koma.
    if "," in raw and "." not in raw:
        raw = raw.replace(",", ".")

    raw = re.sub(r"[^0-9+\-.]", "", raw)
    try:
        number = float(raw)
    except Exception as exc:
        raise ValueError(f"{field_name} tidak valid: {value}") from exc

    if field_name.lower().startswith("lat") and not (-90 <= number <= 90):
        raise ValueError(f"Latitude di luar rentang valid: {number}")
    if field_name.lower().startswith("long") and not (-180 <= number <= 180):
        raise ValueError(f"Longitude di luar rentang valid: {number}")
    return number


def _strip_region_prefix(value, kind="generic"):
    value = _clean_cell(value)

    # Prefix dibersihkan sesuai JENIS field. Ini penting karena kata seperti
    # "KOTA" bisa benar-benar menjadi bagian nama desa (contoh: KOTA LINTANG).
    if kind == "village":
        patterns = [
            r"^KELURAHAN\s+",
            r"^KEL\.?\s+",
            r"^DESA\s+",
        ]
    elif kind == "regency":
        patterns = [
            r"^KABUPATEN\s+",
            r"^KAB\.?\s+",
            r"^KOTA\s+ADMINISTRASI\s+",
            r"^KOTA\s+ADM\.?\s+",
            r"^KOTA\s+",
        ]
    elif kind == "province":
        patterns = [
            r"^PROV(?:INSI)?\.?\s+",
            r"^PROPINSI\.?\s+",
        ]
    else:
        patterns = [
            r"^PROV(?:INSI)?\.?\s+",
            r"^PROPINSI\.?\s+",
            r"^KABUPATEN\s+",
            r"^KAB\.?\s+",
            r"^KOTA\s+ADMINISTRASI\s+",
            r"^KOTA\s+ADM\.?\s+",
            r"^KOTA\s+",
            r"^KECAMATAN\s+",
            r"^KEC\.?\s+",
            r"^KELURAHAN\s+",
            r"^KEL\.?\s+",
            r"^DESA\s+",
        ]

    for pattern in patterns:
        value = re.sub(pattern, "", value, flags=re.IGNORECASE)
    return value.strip(" ,.-")


def _smart_region_case(value, kind="generic"):
    value = _strip_region_prefix(value, kind=kind)
    if not value:
        return ""

    # Rapikan singkatan provinsi yang umum muncul pada data resmi.
    compact_upper = re.sub(r"[^A-Z0-9]", "", value.upper())
    special = {
        "DKIJAKARTA": "DKI Jakarta",
        "DAERAHKHUSUSIBUKOTAJAKARTA": "DKI Jakarta",
        "DIYOGYAKARTA": "DI Yogyakarta",
        "DAERAHISTIMEWAYOGYAKARTA": "DI Yogyakarta",
    }
    if compact_upper in special:
        return special[compact_upper]

    result = value.lower().title()
    replacements = {
        "D.I.": "DI",
        "D. I.": "DI",
        "Dki": "DKI",
        "Di Yogyakarta": "DI Yogyakarta",
    }
    for old, new in replacements.items():
        result = result.replace(old, new)
    return result


def _parse_official_npsn_html(html, target_npsn):
    if not html or len(html) < 100:
        raise ValueError("Respons situs NPSN kosong/tidak lengkap.")

    soup = BeautifulSoup(html, "html.parser")

    page_npsn = _extract_table_value(soup, ["NPSN"])
    page_npsn = _normalize_npsn(page_npsn)
    target = _normalize_npsn(target_npsn)

    if not page_npsn:
        # Fallback jika NPSN ditulis sebagai teks biasa.
        text = soup.get_text(" ", strip=True)
        match = re.search(r"\bNPSN\s*:?\s*([A-Z0-9]{6,16})\b", text, re.I)
        if match:
            page_npsn = _normalize_npsn(match.group(1))

    if not page_npsn:
        raise ValueError("NPSN tidak ditemukan pada halaman resmi.")
    if page_npsn != target:
        raise ValueError(
            f"NPSN pada halaman resmi ({page_npsn}) tidak sama dengan input ({target})."
        )

    village_raw = _extract_table_value(
        soup,
        ["Desa/Kelurahan", "Desa Kelurahan", "Kelurahan/Desa"],
    )
    regency_raw = _extract_table_value(
        soup,
        [
            "Kab.-Kota/Negara (LN)",
            "Kab-Kota/Negara (LN)",
            "Kabupaten/Kota",
            "Kabupaten Kota",
        ],
    )
    province_raw = _extract_table_value(
        soup,
        [
            "Propinsi/Luar Negeri (LN)",
            "Provinsi/Luar Negeri (LN)",
            "Propinsi",
            "Provinsi",
        ],
    )

    latitude_raw = _extract_coordinate_from_html(
        soup, html, ["Lintang", "Latitude"]
    )
    longitude_raw = _extract_coordinate_from_html(
        soup, html, ["Bujur", "Longitude"]
    )

    latitude = _parse_coordinate(latitude_raw, "Latitude")
    longitude = _parse_coordinate(longitude_raw, "Longitude")

    village = _smart_region_case(village_raw, kind="village")
    regency_city = _smart_region_case(regency_raw, kind="regency")
    province = _smart_region_case(province_raw, kind="province")

    missing = [
        name
        for name, value in [
            ("Desa/Kelurahan", village),
            ("Kabupaten/Kota", regency_city),
            ("Provinsi", province),
        ]
        if not value
    ]
    if missing:
        raise ValueError(
            "Data wilayah resmi belum lengkap: " + ", ".join(missing) + "."
        )

    location = f"{village}, {regency_city}, {province}"

    return {
        "npsn": target,
        "latitude": latitude,
        "longitude": longitude,
        "village": village,
        "regency_city": regency_city,
        "province": province,
        "location": location,
        "source_url": OFFICIAL_NPSN_BASE_URL + quote(target, safe=""),
        "source_name": "Referensi Data Kemendikdasmen",
    }


def _fetch_official_npsn_location_uncached(npsn):
    target = _normalize_npsn(npsn)
    if not target:
        raise ValueError("NPSN belum diisi.")
    if len(target) < 6 or len(target) > 16:
        raise ValueError("Format NPSN tidak valid.")

    url = OFFICIAL_NPSN_BASE_URL + quote(target, safe="")
    try:
        response = requests.get(
            url,
            headers=HTTP_HEADERS,
            timeout=HTTP_TIMEOUT,
            allow_redirects=True,
        )
    except requests.RequestException as exc:
        raise ValueError(
            "Tidak dapat terhubung ke situs Referensi Data Kemendikdasmen. "
            "Periksa koneksi internet lalu coba lagi."
        ) from exc

    if response.status_code == 404:
        raise ValueError(f"NPSN {target} tidak ditemukan pada situs resmi.")
    if response.status_code != 200:
        raise ValueError(
            f"Situs NPSN mengembalikan HTTP {response.status_code}. Coba beberapa saat lagi."
        )

    return _parse_official_npsn_html(response.text, target)


# Cache dibuat hanya ketika Streamlit tersedia.
if st is not None:
    fetch_official_npsn_location = st.cache_data(ttl=6 * 60 * 60, show_spinner=False)(
        _fetch_official_npsn_location_uncached
    )
else:
    fetch_official_npsn_location = _fetch_official_npsn_location_uncached


# ============================================================
# VALIDASI KOORDINAT DENGAN REVERSE GEOCODING (PELAPIS TAMBAHAN)
# ============================================================


def _normalize_region_for_compare(value):
    value = _clean_cell(value).lower()
    value = value.replace("daerah khusus ibukota", "dki")
    value = value.replace("daerah istimewa", "di")
    value = re.sub(
        r"\b(provinsi|propinsi|kabupaten|kab\.?|kota administrasi|kota adm\.?|kota|"
        r"kecamatan|kec\.?|kelurahan|kel\.?|desa|regency|city|special region)\b",
        " ",
        value,
    )
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _region_similar(a, b):
    a = _normalize_region_for_compare(a)
    b = _normalize_region_for_compare(b)
    if not a or not b:
        return None
    if a == b or a in b or b in a:
        return True

    ta, tb = set(a.split()), set(b.split())
    if ta and tb:
        jaccard = len(ta & tb) / len(ta | tb)
        if jaccard >= 0.67:
            return True

    return SequenceMatcher(None, a, b).ratio() >= 0.74


def _reverse_geocode_osm_uncached(latitude, longitude):
    params = {
        "format": "jsonv2",
        "lat": f"{float(latitude):.8f}",
        "lon": f"{float(longitude):.8f}",
        "zoom": 14,
        "addressdetails": 1,
        "accept-language": "id",
    }
    headers = dict(HTTP_HEADERS)
    headers["User-Agent"] = "AutoTimestampNPSN/7.7 Streamlit location-validator"

    try:
        response = requests.get(
            NOMINATIM_REVERSE_URL,
            params=params,
            headers=headers,
            timeout=HTTP_TIMEOUT,
        )
        if response.status_code != 200:
            return {"available": False, "reason": f"HTTP {response.status_code}"}
        data = response.json()
    except Exception as exc:
        return {"available": False, "reason": str(exc)[:180]}

    address = data.get("address") or {}
    village_candidates = [
        address.get("village"),
        address.get("suburb"),
        address.get("hamlet"),
        address.get("neighbourhood"),
        address.get("town"),
    ]
    regency_candidates = [
        address.get("county"),
        address.get("city"),
        address.get("municipality"),
        address.get("state_district"),
    ]
    province_candidates = [address.get("state"), address.get("region")]

    village_candidates = [_clean_cell(x) for x in village_candidates if _clean_cell(x)]
    regency_candidates = [_clean_cell(x) for x in regency_candidates if _clean_cell(x)]
    province_candidates = [_clean_cell(x) for x in province_candidates if _clean_cell(x)]

    return {
        "available": True,
        "village": village_candidates[0] if village_candidates else "",
        "regency_city": regency_candidates[0] if regency_candidates else "",
        "province": province_candidates[0] if province_candidates else "",
        "village_candidates": village_candidates,
        "regency_candidates": regency_candidates,
        "province_candidates": province_candidates,
        "display_name": _clean_cell(data.get("display_name")),
    }


if st is not None:
    reverse_geocode_osm = st.cache_data(ttl=24 * 60 * 60, show_spinner=False)(
        _reverse_geocode_osm_uncached
    )
else:
    reverse_geocode_osm = _reverse_geocode_osm_uncached


def validate_official_location_with_reverse(official, reverse_data):
    if not reverse_data or not reverse_data.get("available"):
        return {
            "status": "official_only",
            "conflict": False,
            "message": (
                "Data resmi NPSN lengkap. Cross-check peta sedang tidak tersedia; "
                "koordinat tetap memakai sumber resmi Kemendikdasmen."
            ),
        }

    def match_any(official_value, candidates):
        candidates = [x for x in (candidates or []) if _clean_cell(x)]
        if not _clean_cell(official_value) or not candidates:
            return None
        checks = [_region_similar(official_value, candidate) for candidate in candidates]
        return True if True in checks else False

    province_candidates = reverse_data.get("province_candidates") or [
        reverse_data.get("province")
    ]
    regency_candidates = reverse_data.get("regency_candidates") or [
        reverse_data.get("regency_city")
    ]
    village_candidates = reverse_data.get("village_candidates") or [
        reverse_data.get("village")
    ]

    province_match = match_any(official.get("province"), province_candidates)
    regency_match = match_any(official.get("regency_city"), regency_candidates)
    village_match = match_any(official.get("village"), village_candidates)

    # Konflik jelas pada provinsi atau kabupaten/kota harus menghentikan generate.
    # Jika salah satu level tidak tersedia dari reverse geocoder, level itu tidak
    # dianggap konflik (sumber utama tetap data resmi NPSN).
    conflict = province_match is False or regency_match is False

    if conflict:
        return {
            "status": "conflict",
            "conflict": True,
            "province_match": province_match,
            "regency_match": regency_match,
            "village_match": village_match,
            "message": (
                "Koordinat resmi terdeteksi tidak konsisten dengan wilayah hasil reverse geocoding. "
                "Generate diblokir agar lokasi tidak salah."
            ),
        }

    return {
        "status": "verified",
        "conflict": False,
        "province_match": province_match,
        "regency_match": regency_match,
        "village_match": village_match,
        "message": (
            "Koordinat konsisten dengan Kabupaten/Kota dan Provinsi pada pengecekan peta."
        ),
    }


# ============================================================
# FOTO, WAKTU, DAN OUTPUT
# ============================================================


def _photo_sort_key(uploaded_file):
    stem = os.path.splitext(os.path.basename(uploaded_file.name))[0].strip()
    if stem.isdigit():
        return (0, int(stem), uploaded_file.name.lower())
    return (1, stem.lower(), uploaded_file.name.lower())


def validate_numeric_photo_names(uploaded_files):
    invalid = []
    duplicates = []
    seen = []

    for uploaded_file in uploaded_files or []:
        stem = os.path.splitext(os.path.basename(uploaded_file.name))[0].strip()
        if not stem.isdigit():
            invalid.append(uploaded_file.name)
            continue
        number = int(stem)
        if number in seen:
            duplicates.append(number)
        seen.append(number)

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

    if seen and not invalid and not duplicates:
        numbers = sorted(seen)
        expected = list(range(1, len(numbers) + 1))
        if numbers != expected:
            errors.append(
                "Nomor foto harus berurutan mulai 1 tanpa loncat. "
                f"Ditemukan: {', '.join(map(str, numbers))}; seharusnya: {', '.join(map(str, expected))}."
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
# STREAMLIT BATCH UI
# ============================================================


def render_batch_mode(uploader_version, load_image_fn, generate_timestamp_fn):
    if st is None:
        raise RuntimeError("Streamlit belum terpasang. Jalankan pip install -r requirements.txt")

    result_key = "batch_generated_result"
    lookup_key = "batch_online_npsn_result"
    validation_key = "batch_online_validation"

    for key in [result_key, lookup_key, validation_key]:
        if key not in st.session_state:
            st.session_state[key] = None

    st.markdown(
        '<div class="hero-badge">⚡ V7.7 • NPSN Online Resmi + Validasi Lokasi</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="hero-title">🗂️ Batch Timestamp dari NPSN</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="hero-desc">Upload foto bernama <b>1, 2, 3, ...</b>, masukkan satu NPSN, '
        'ambil koordinat & daerah <b>langsung dari Referensi Data Kemendikdasmen</b>, lalu '
        'buat jam acak yang tetap berurutan.</div>',
        unsafe_allow_html=True,
    )

    st.warning(
        "⚠️ Waktu pada mode ini dibuat dari rentang yang kamu pilih dan bukan waktu "
        "pengambilan asli/EXIF. Overlay adalah informasi yang ditambahkan setelah foto dibuat."
    )

    # --------------------------------------------------------
    # 1. FOTO
    # --------------------------------------------------------
    st.markdown("### 1️⃣ Upload Foto Batch")
    with st.container(border=True):
        batch_files = st.file_uploader(
            "Pilih beberapa foto (JPG, JPEG, PNG)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key=f"batch_photo_uploader_{uploader_version}",
            help="Tidak perlu ZIP. Nama foto wajib 1.jpg, 2.jpg, 3.jpg, dst.",
        )

        sorted_files = sorted(batch_files or [], key=_photo_sort_key)
        name_errors = validate_numeric_photo_names(sorted_files)
        for error_text in name_errors:
            st.error("❌ " + error_text)

        if sorted_files and not name_errors:
            st.success(f"✅ {len(sorted_files)} foto siap. Urutan mengikuti nomor file.")
            st.dataframe(
                pd.DataFrame(
                    [{"Urutan": i, "Nama File": f.name} for i, f in enumerate(sorted_files, 1)]
                ),
                use_container_width=True,
                hide_index=True,
            )
        elif not sorted_files:
            st.info("Upload foto yang sudah dinamai 1, 2, 3, 4, ...")

    # --------------------------------------------------------
    # 2. NPSN ONLINE
    # --------------------------------------------------------
    st.markdown("### 2️⃣ Ambil Lokasi dari NPSN Online")
    with st.container(border=True):
        npsn_value = st.text_input(
            "Masukkan NPSN",
            placeholder="Contoh: 10257564",
            key=f"batch_npsn_{uploader_version}",
            help="NPSN digunakan hanya sebagai kunci lookup. Nama sekolah tidak dimasukkan ke timestamp.",
        )

        current_npsn = _normalize_npsn(npsn_value)
        lookup_clicked = st.button(
            "🔎 CEK NPSN & AMBIL LOKASI ONLINE",
            type="primary",
            disabled=not bool(current_npsn),
            use_container_width=True,
        )

        if lookup_clicked:
            try:
                with st.spinner("Mengambil data resmi NPSN dari Kemendikdasmen..."):
                    official = fetch_official_npsn_location(current_npsn)

                with st.spinner("Memeriksa konsistensi koordinat dengan peta..."):
                    reverse_data = reverse_geocode_osm(
                        official["latitude"], official["longitude"]
                    )
                    validation = validate_official_location_with_reverse(
                        official, reverse_data
                    )

                st.session_state[lookup_key] = official
                st.session_state[validation_key] = {
                    **validation,
                    "reverse_data": reverse_data,
                }
                st.session_state[result_key] = None
            except Exception as error:
                st.session_state[lookup_key] = None
                st.session_state[validation_key] = None
                st.error(f"❌ {error}")

        official = st.session_state.get(lookup_key)
        validation = st.session_state.get(validation_key)

        # Jangan gunakan hasil lama jika user mengganti NPSN setelah lookup.
        if official and official.get("npsn") != current_npsn:
            official = None
            validation = None
            st.info("NPSN berubah. Klik **Cek NPSN** lagi untuk mengambil lokasi yang baru.")

        npsn_info = None
        if official:
            st.success("✅ Data NPSN resmi ditemukan dan field lokasi lengkap.")
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Latitude", f"{official['latitude']:.7f}")
            with c2:
                st.metric("Longitude", f"{official['longitude']:.7f}")

            st.markdown(f"📍 **Format daerah timestamp:** `{official['location']}`")
            st.caption(
                "Format otomatis: Desa/Kelurahan, Kabupaten/Kota, Provinsi. "
                "Prefix KAB./KOTA/PROV. dibersihkan agar mirip template GPS Camera."
            )
            st.markdown(
                f"🔗 [Buka sumber resmi NPSN]({official['source_url']})"
            )

            if validation:
                if validation.get("status") == "verified":
                    st.success("✅ Validasi lokasi: " + validation.get("message", ""))
                elif validation.get("status") == "official_only":
                    st.warning("⚠️ " + validation.get("message", ""))
                elif validation.get("conflict"):
                    st.error("⛔ " + validation.get("message", ""))
                    reverse_data = validation.get("reverse_data") or {}
                    if reverse_data.get("display_name"):
                        st.caption("Hasil peta: " + reverse_data["display_name"])

            if not (validation and validation.get("conflict")):
                npsn_info = official

            st.code(
                f"{official['latitude']:.7f}, {official['longitude']:.7f}\n"
                f"{official['location']}\n"
                "DD/MM/YYYY HH:MM:SS WIB",
                language=None,
            )

    # --------------------------------------------------------
    # 3. TANGGAL & JAM
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
            "Jam dibuat acak di dalam rentang, lalu diurutkan naik sesuai file 1 → 2 → 3 → dst."
        )

    # --------------------------------------------------------
    # 4. GENERATE
    # --------------------------------------------------------
    st.markdown("### 4️⃣ Generate Batch")
    batch_ready = bool(sorted_files) and not name_errors and npsn_info is not None

    with st.container(border=True):
        if not batch_ready:
            st.info(
                "Lengkapi foto bernomor + NPSN online yang valid. Jika validator menemukan konflik "
                "wilayah/koordinat, tombol generate otomatis diblokir."
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
                coordinate_text = (
                    f"{npsn_info['latitude']:.7f}, {npsn_info['longitude']:.7f}"
                )

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

                            date_text = (
                                generated_dt.strftime("%d/%m/%Y %H:%M:%S")
                                + f" {timezone_text}"
                            )
                            raw_timestamp = (
                                coordinate_text
                                + "\n"
                                + npsn_info["location"]
                                + "\n"
                                + date_text
                            )

                            result_pil, _ = generate_timestamp_fn(photo_cv, raw_timestamp)
                            encoded_bytes, _ = _encode_pil_image_bytes(
                                result_pil, uploaded_file.name
                            )
                            zip_file.writestr(uploaded_file.name, encoded_bytes)

                            if first_preview is None:
                                first_preview = encoded_bytes
                                first_preview_name = uploaded_file.name

                            validation_status = (
                                (validation or {}).get("status", "official_only")
                            )
                            log_rows.append(
                                {
                                    "Nama File": uploaded_file.name,
                                    "NPSN": npsn_info["npsn"],
                                    "Latitude": f"{npsn_info['latitude']:.7f}",
                                    "Longitude": f"{npsn_info['longitude']:.7f}",
                                    "Daerah": npsn_info["location"],
                                    "Tanggal": generated_dt.strftime("%d/%m/%Y"),
                                    "Jam": generated_dt.strftime("%H:%M:%S"),
                                    "Zona Waktu": timezone_text,
                                    "Sumber Lokasi": npsn_info["source_name"],
                                    "URL Sumber": npsn_info["source_url"],
                                    "Validasi Lokasi": validation_status,
                                    "Sumber Waktu": "Manual/random dari rentang input",
                                }
                            )

                        log_df = pd.DataFrame(log_rows)
                        log_csv = log_df.to_csv(index=False).encode("utf-8-sig")
                        zip_file.writestr("timestamp_log.csv", log_csv)

                st.session_state[result_key] = {
                    "zip_bytes": zip_buffer.getvalue(),
                    "zip_name": f"hasil_timestamp_npsn_{npsn_info['npsn']}.zip",
                    "log_rows": log_rows,
                    "log_csv": log_csv,
                    "preview_bytes": first_preview,
                    "preview_name": first_preview_name,
                }
                st.success(f"✅ Selesai memproses {len(sorted_files)} foto.")
            except Exception as error:
                st.error(f"❌ Gagal generate batch: {error}")

    # --------------------------------------------------------
    # HASIL
    # --------------------------------------------------------
    result = st.session_state.get(result_key)
    if result is not None:
        st.markdown("---")
        st.markdown("### ✨ Hasil Batch")
        with st.container(border=True):
            st.success(
                "Nama sekolah dan NPSN tidak ditampilkan pada foto. Overlay hanya berisi "
                "koordinat, daerah, tanggal, dan jam."
            )

            result_df = pd.DataFrame(result["log_rows"])
            st.dataframe(
                result_df[
                    ["Nama File", "Tanggal", "Jam", "Latitude", "Longitude", "Daerah"]
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

            c_zip, c_log, c_reset = st.columns([2, 1, 1])
            with c_zip:
                st.download_button(
                    "⬇️ DOWNLOAD SEMUA FOTO (ZIP)",
                    data=result["zip_bytes"],
                    file_name=result["zip_name"],
                    mime="application/zip",
                    type="primary",
                    use_container_width=True,
                )
            with c_log:
                st.download_button(
                    "📄 Download Log CSV",
                    data=result["log_csv"],
                    file_name="timestamp_log.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with c_reset:
                if st.button("🔄 Reset Batch", use_container_width=True):
                    st.session_state[result_key] = None
                    st.session_state[lookup_key] = None
                    st.session_state[validation_key] = None
                    st.session_state.uploader_version += 1
                    st.rerun()
