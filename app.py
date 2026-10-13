# ============================================================
# TIMESTAMP GENERATOR V7.5 - STREAMLIT UI (ENHANCED UI + AUTOFIT)
# ============================================================
#
# FONT:
# AVHershey Simplex Medium
#
# SOURCE:
# https://github.com/yangcht/Hershey_font_TTF
#
# Port dari versi Google Colab (V7.4 New Input freetext +
# Gemini OCR). Fungsi inti (normalisasi teks, parsing
# timestamp, render font, OCR Gemini) TIDAK diubah logikanya
# -- hanya lapisan UI/IO yang disempurnakan tampilannya.
#
# V7.5 (baru):
# - AUTO-FIT font: baris teks yang kepanjangan (misal nama
#   lokasi panjang) otomatis mengecilkan ukuran font supaya
#   TIDAK kepotong di tepi kanan foto. Ukuran font tetap sama
#   rata untuk ketiga baris (bukan per-baris beda ukuran).
#
# Credit:
# Matthew Artur Panahatan Sitorus
# Nikita Adriella Virginia Jacob
# ============================================================


# ============================================================
# IMPORT
# ============================================================

import os
import re
import io
import glob
import hashlib
import secrets
import subprocess

import cv2
import numpy as np
import streamlit as st

from PIL import (
    Image,
    ImageDraw,
    ImageFont,
    ImageOps
)

from google import genai
from google.genai import types as genai_types

from streamlit_paste_button import paste_image_button as pbutton


# ============================================================
# ============================================================
# CONFIGURATION (SAMA PERSIS DENGAN VERSI COLAB)
# ============================================================
# ============================================================


# ============================================================
# MASTER WIDTH
# ============================================================

REFERENCE_WIDTH = 2048


# ============================================================
# FONT SIZE
# ============================================================

REFERENCE_FONT_SIZE = 70

# Ukuran timestamp dinyatakan sebagai % dari lebar foto, supaya
# otomatis mengikuti resolusi/rasio foto apapun yang diupload.
# Diatur langsung di sini oleh program (bukan lewat UI) --
# tinggal ganti angka ini kalau mau lebih besar/kecil lagi.
DEFAULT_FONT_SIZE_PERCENT = 5.0

# --------------------------------------------------------
# Batas bawah font size (px) saat auto-fit mengecilkan teks.
# Ini jaring pengaman supaya teks tidak jadi mikroskopis kalau
# baris lokasinya sangat-sangat panjang -- kalau sudah mentok
# batas ini, teks dibiarkan sedikit mepet ke tepi daripada
# jadi tidak terbaca sama sekali.
# --------------------------------------------------------

MIN_FONT_SIZE_PX = 14


# ============================================================
# TEXT COLOR
# ============================================================

TEXT_COLOR = (
    255,
    255,
    255
)


# ============================================================
# OUTLINE
# ============================================================

REFERENCE_OUTLINE_WIDTH = 2


OUTLINE_COLOR = (
    0,
    0,
    0
)


# ============================================================
# OUTLINE OFFSET
# ============================================================

REFERENCE_OUTLINE_OFFSET_X = 0

REFERENCE_OUTLINE_OFFSET_Y = 0


# ============================================================
# LETTER SPACING
# ============================================================

REFERENCE_LETTER_SPACING = -1


# ============================================================
# SPACE SCALE
# ============================================================

SPACE_SCALE = 0.75


# ============================================================
# POSISI
# ============================================================

REFERENCE_LEFT_MARGIN = 32

REFERENCE_BOTTOM_MARGIN = 88

REFERENCE_LINE_SPACING = 78


# ============================================================
# TEXT ANCHOR
# ============================================================

TEXT_ANCHOR = "ls"


# ============================================================
# ============================================================
# FONT SETUP (DENGAN CACHE STREAMLIT)
# ============================================================
# ============================================================

REPO_DIR = "Hershey_font_TTF"


# ============================================================
# DAFTAR KATA UNTUK NAMA FILE RANDOM
# ============================================================
#
# Dipakai buat bikin nama file download unik & gampang dibaca,
# contoh: FOTO_hangatpelangi.jpg
# ============================================================

RANDOM_WORD_LIST_A = [
    "hangat", "cerah", "biru", "senja", "kilat", "lembut",
    "gemilang", "rindang", "damai", "megah", "riang", "sejuk",
    "terang", "elok", "anggun", "cepat", "tenang", "ceria",
]

RANDOM_WORD_LIST_B = [
    "pelangi", "senja", "camar", "kabut", "elang", "bintang",
    "ombak", "awan", "bukit", "kilau", "embun", "cakrawala",
    "merpati", "fajar", "angin", "karang", "hutan", "samudra",
]


@st.cache_resource(show_spinner="Menyiapkan font Hershey...")
def get_font_path():

    if not os.path.exists(REPO_DIR):

        subprocess.run(
            [
                "git",
                "clone",
                "-q",
                "https://github.com/yangcht/Hershey_font_TTF.git",
                REPO_DIR
            ],
            check=True
        )

    font_candidates = glob.glob(
        REPO_DIR + "/**/*Simplex*Medium*.ttf",
        recursive=True
    )

    if len(font_candidates) == 0:

        raise Exception(
            "Font AVHershey Simplex Medium tidak ditemukan."
        )

    return font_candidates[0]


# ------------------------------------------------------------
# Cache objek font per (path, size) supaya file .ttf tidak
# di-parse ulang dari disk setiap kali generate dipanggil --
# ImageFont.truetype() lumayan mahal kalau dipanggil berulang.
# ------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_font(font_path, font_size):

    return ImageFont.truetype(
        font_path,
        font_size
    )


# ============================================================
# ============================================================
# OCR PAKAI GEMINI AI (LOGIKA SAMA DENGAN VERSI COLAB)
# ============================================================
# ============================================================

GEMINI_MODEL_NAME = "gemini-flash-lite-latest"


# ============================================================
# LOAD FOTO DENGAN FIX ORIENTASI EXIF
# ============================================================
#
# cv2.imread() TIDAK membaca tag EXIF Orientation, jadi foto
# dari HP yang disimpan dengan tag "rotate 90/180/270" akan
# terbaca apa adanya (raw sensor orientation) oleh OpenCV.
#
# Akibatnya timestamp yang kita gambar (selalu lurus/horizontal
# di array mentah) jadi ikut miring begitu foto ditampilkan di
# viewer lain yang MENGHORMATI tag EXIF tersebut.
#
# Fix: normalisasi orientasi dulu pakai PIL (ImageOps.exif_transpose)
# SEBELUM digambar, supaya pixel-nya benar2 sudah tegak sesuai
# tampilan aslinya, dan text yang kita gambar horizontal akan
# selalu terlihat lurus di manapun foto ini dibuka.
# ============================================================

def load_pil_image_fixed_orientation(uploaded_file):

    pil_image = Image.open(
        uploaded_file
    )

    pil_image = ImageOps.exif_transpose(
        pil_image
    )

    pil_image = pil_image.convert(
        "RGB"
    )

    return pil_image


GEMINI_OCR_PROMPT = """
Gambar ini adalah screenshot yang berisi overlay timestamp
GPS kamera (koordinat, lokasi, tanggal & jam).

Baca teks overlay timestamp tersebut PERSIS seperti yang
tertulis di gambar, tanpa mengubah, membetulkan, atau
menerka-nerka isinya.

Keluarkan HANYA 3 baris berikut, tanpa penjelasan tambahan,
tanpa markdown, tanpa tanda kutip:

Baris 1: koordinat (lintang, bujur)
Baris 2: nama lokasi
Baris 3: tanggal dan jam (dan WIB/zona waktu jika ada)
""".strip()


def ocr_extract_text_gemini(
    pil_image,
    api_key
):

    client = genai.Client(
        api_key=api_key
    )

    # --------------------------------------------------------
    # Convert PIL image ke bytes PNG
    # --------------------------------------------------------

    image_buffer = io.BytesIO()

    pil_image.save(
        image_buffer,
        format="PNG"
    )

    image_bytes = image_buffer.getvalue()

    # --------------------------------------------------------
    # Kirim ke Gemini
    # --------------------------------------------------------

    response = client.models.generate_content(

        model=GEMINI_MODEL_NAME,

        contents=[

            genai_types.Part.from_bytes(
                data=image_bytes,
                mime_type="image/png"
            ),

            GEMINI_OCR_PROMPT
        ]
    )

    raw_text = response.text or ""

    # --------------------------------------------------------
    # Bersihkan kemungkinan markdown code fence
    # kalau Gemini iseng bungkus jawabannya
    # --------------------------------------------------------

    raw_text = re.sub(
        r"^```[a-zA-Z]*\n?",
        "",
        raw_text.strip()
    )

    raw_text = re.sub(
        r"```$",
        "",
        raw_text.strip()
    )

    lines = [
        line.strip()
        for line in raw_text.splitlines()
        if line.strip()
    ]

    return "\n".join(lines)


# ============================================================
# ============================================================
# NORMALISASI TEXT (LOGIKA SAMA DENGAN VERSI COLAB)
# ============================================================
# ============================================================

def normalize_coordinate(text):

    text = re.sub(
        r"\s+",
        " ",
        text.strip()
    )

    text = re.sub(
        r"\s*,",
        ",",
        text
    )

    text = re.sub(
        r",\s*",
        ",  ",
        text
    )

    return text


def normalize_location(text):

    text = re.sub(
        r"\s+",
        " ",
        text.strip()
    )

    text = re.sub(
        r"\s*,",
        ",",
        text
    )

    text = re.sub(
        r",\s*",
        ",  ",
        text
    )

    return text


def normalize_datetime(text):

    text = re.sub(
        r"\s+",
        " ",
        text.strip()
    )

    time_match = re.search(
        r"\b(\d{1,2}:\d{2}:\d{2})\b",
        text
    )

    if time_match:

        time_value = time_match.group(1)

        before = text[
            :time_match.start()
        ].strip()

        after = text[
            time_match.end():
        ].strip()

        after = re.sub(
            r"^\s*WIB\s*$",
            "WIB",
            after,
            flags=re.IGNORECASE
        )

        if after:

            return (
                before
                + " "
                + time_value
                + "  "
                + after
            )

        else:

            return (
                before
                + " "
                + time_value
            )

    return text


# ============================================================
# PARSE TIMESTAMP
# ============================================================

def parse_timestamp(raw_text):

    lines = raw_text.splitlines()

    lines = [
        line.strip()
        for line in lines
        if line.strip()
    ]

    if len(lines) != 3:

        raise ValueError(
            "Input harus terdiri dari tepat 3 baris."
        )

    line1 = normalize_coordinate(
        lines[0]
    )

    line2 = normalize_location(
        lines[1]
    )

    line3 = normalize_datetime(
        lines[2]
    )

    return [
        line1,
        line2,
        line3
    ]


# ============================================================
# ============================================================
# DRAW TIMESTAMP (LOGIKA SAMA DENGAN VERSI COLAB)
# ============================================================
# ============================================================

def draw_timestamp(
    draw,
    text,
    x,
    baseline,
    font,
    outline_width,
    letter_spacing,
    space_scale,
    outline_offset_x,
    outline_offset_y
):

    current_x = float(x)

    # Lebar spasi sama untuk semua karakter " " di baris yang
    # sama, jadi dihitung sekali saja sebelum loop
    space_width = draw.textlength(
        " ",
        font=font
    ) * space_scale

    for char in text:

        if char == " ":

            current_x += (
                space_width +
                letter_spacing
            )

            continue

        draw.text(

            (
                round(
                    current_x +
                    outline_offset_x
                ),

                round(
                    baseline +
                    outline_offset_y
                )
            ),

            char,

            font=font,

            fill=TEXT_COLOR,

            stroke_width=outline_width,

            stroke_fill=OUTLINE_COLOR,

            anchor=TEXT_ANCHOR
        )

        char_width = draw.textlength(
            char,
            font=font
        )

        current_x += (
            char_width +
            letter_spacing
        )


# ============================================================
# ============================================================
# UKUR LEBAR TEKS (BUAT AUTO-FIT, TANPA MENGGAMBAR APAPUN)
# ============================================================
# ============================================================
#
# Rumus lebar ini SENGAJA dibuat identik dengan logika loop
# di draw_timestamp() di atas (char demi char + letter_spacing,
# spasi pakai space_scale) supaya angka yang diukur benar-benar
# merepresentasikan lebar hasil gambar yang sebenarnya.
# ============================================================

def measure_line_width(
    measure_draw,
    text,
    font,
    letter_spacing,
    space_scale
):

    space_width = measure_draw.textlength(
        " ",
        font=font
    ) * space_scale

    total_width = 0.0

    for char in text:

        if char == " ":

            total_width += (
                space_width +
                letter_spacing
            )

        else:

            total_width += (
                measure_draw.textlength(
                    char,
                    font=font
                ) +
                letter_spacing
            )

    return total_width


# ============================================================
# ============================================================
# GENERATE TIMESTAMP (LOGIKA SAMA, INPUT/OUTPUT PAKAI BYTES)
# ============================================================
# ============================================================

def generate_timestamp_image(
    image_cv,
    raw_timestamp_text,
    font_size_percent=DEFAULT_FONT_SIZE_PERCENT
):

    lines = parse_timestamp(
        raw_timestamp_text
    )

    line1 = lines[0]

    line2 = lines[1]

    line3 = lines[2]

    height, width = image_cv.shape[:2]

    # --------------------------------------------------------
    # Font size AWAL dihitung dari % lebar foto (seperti biasa)
    # --------------------------------------------------------

    font_size = max(
        1,
        round(
            width *
            (font_size_percent / 100)
        )
    )

    font_path = get_font_path()

    # --------------------------------------------------------
    # AUTO-FIT
    #
    # font_size di atas cuma mempertimbangkan lebar FOTO, belum
    # mempertimbangkan lebar TEKS. Kalau baris terpanjang
    # (biasanya baris lokasi) lebih lebar dari ruang yang
    # tersedia (lebar foto - margin kiri - margin kanan), maka
    # font dikecilkan proporsional sampai muat, lalu diukur
    # ulang -- diulang beberapa kali sampai konvergen (biasanya
    # cukup 1-2 kali karena lebar teks kurang-lebih linear
    # terhadap font_size).
    #
    # font_size tetap SATU ANGKA untuk ketiga baris, biar
    # tampilannya tetap konsisten & rapi.
    # --------------------------------------------------------

    measure_draw = ImageDraw.Draw(
        Image.new(
            "RGB",
            (1, 1)
        )
    )

    scale_ratio = None
    left_margin = None
    right_margin = None
    letter_spacing = None
    font = None

    for _ in range(6):

        scale_ratio = (
            font_size /
            REFERENCE_FONT_SIZE
        )

        left_margin = round(
            REFERENCE_LEFT_MARGIN *
            scale_ratio
        )

        # Margin kanan dibuat sama dengan margin kiri sebagai
        # ruang aman simetris, supaya teks tidak mepet ke tepi
        # kanan foto.
        right_margin = left_margin

        letter_spacing = (
            REFERENCE_LETTER_SPACING *
            scale_ratio
        )

        font = get_font(
            font_path,
            font_size
        )

        max_line_width = max(
            measure_line_width(
                measure_draw, line1, font, letter_spacing, SPACE_SCALE
            ),
            measure_line_width(
                measure_draw, line2, font, letter_spacing, SPACE_SCALE
            ),
            measure_line_width(
                measure_draw, line3, font, letter_spacing, SPACE_SCALE
            ),
        )

        available_width = width - left_margin - right_margin

        if max_line_width <= available_width:

            break

        if font_size <= MIN_FONT_SIZE_PX:

            # Sudah mentok batas bawah -- biarkan sedikit mepet
            # daripada teks jadi tidak terbaca sama sekali.
            break

        shrink_ratio = available_width / max_line_width

        # Faktor 0.97 = sedikit buffer aman supaya tidak
        # langsung pas-pasan di tepi banget setelah dibulatkan.
        font_size = max(
            MIN_FONT_SIZE_PX,
            int(font_size * shrink_ratio * 0.97)
        )

    auto_fit_applied = (
        font_size <
        max(1, round(width * (font_size_percent / 100)))
    )

    # --------------------------------------------------------
    # Lanjut seperti biasa dengan font_size FINAL hasil auto-fit
    # --------------------------------------------------------

    outline_width = max(
        1,
        round(
            REFERENCE_OUTLINE_WIDTH *
            scale_ratio
        )
    )

    bottom_margin = round(
        REFERENCE_BOTTOM_MARGIN *
        scale_ratio
    )

    line_spacing = round(
        REFERENCE_LINE_SPACING *
        scale_ratio
    )

    outline_offset_x = round(
        REFERENCE_OUTLINE_OFFSET_X *
        scale_ratio
    )

    outline_offset_y = round(
        REFERENCE_OUTLINE_OFFSET_Y *
        scale_ratio
    )

    pil_image = Image.fromarray(
        image_cv
    )

    draw = ImageDraw.Draw(
        pil_image
    )

    baseline3 = (
        height -
        bottom_margin
    )

    baseline2 = (
        baseline3 -
        line_spacing
    )

    baseline1 = (
        baseline2 -
        line_spacing
    )

    draw_timestamp(
        draw,
        line1,
        left_margin,
        baseline1,
        font,
        outline_width,
        letter_spacing,
        SPACE_SCALE,
        outline_offset_x,
        outline_offset_y
    )

    draw_timestamp(
        draw,
        line2,
        left_margin,
        baseline2,
        font,
        outline_width,
        letter_spacing,
        SPACE_SCALE,
        outline_offset_x,
        outline_offset_y
    )

    draw_timestamp(
        draw,
        line3,
        left_margin,
        baseline3,
        font,
        outline_width,
        letter_spacing,
        SPACE_SCALE,
        outline_offset_x,
        outline_offset_y
    )

    info = {
        "line1": line1,
        "line2": line2,
        "line3": line3,
        "font_size": font_size,
        "scale_ratio": scale_ratio,
        "outline_width": outline_width,
        "letter_spacing": letter_spacing,
        "left_margin": left_margin,
        "bottom_margin": bottom_margin,
        "line_spacing": line_spacing,
        "auto_fit_applied": auto_fit_applied
    }

    return pil_image, info


# ============================================================
# ============================================================
# STREAMLIT UI CONFIG & MODERN STYLING
# ============================================================
# ============================================================

st.set_page_config(
    page_title="Timestamp Generator | GPS Camera Style",
    page_icon="🕒",
    layout="centered",
    initial_sidebar_state="collapsed"
)


# ------------------------------------------------------------
# CUSTOM CSS UNTUK TAMPILAN MODERN & CLEAN
# ------------------------------------------------------------

st.markdown(
    """
    <style>
    /* Card Container & Shadow */
    div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 14px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.04);
        border: 1px solid rgba(140, 140, 140, 0.16);
        transition: all 0.2s ease-in-out;
    }
    
    /* Hero Header */
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin-bottom: 0.2rem;
    }
    .hero-badge {
        display: inline-block;
        font-size: 0.75rem;
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 20px;
        background: linear-gradient(135deg, #2563eb, #3b82f6);
        color: white !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.6rem;
    }
    .hero-desc {
        color: var(--text-color, #6b7280);
        font-size: 0.95rem;
        margin-bottom: 1.2rem;
    }

    /* Stepper Status Bar */
    .stepper-container {
        display: flex;
        gap: 8px;
        margin-bottom: 1.5rem;
        flex-wrap: wrap;
    }
    .step-pill {
        flex: 1 1 auto;
        min-width: 140px;
        padding: 8px 12px;
        border-radius: 10px;
        font-size: 0.82rem;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 6px;
        border: 1px solid rgba(140, 140, 140, 0.2);
        background: rgba(140, 140, 140, 0.06);
    }
    .step-pill.active {
        border-color: #3b82f6;
        background: rgba(59, 130, 246, 0.1);
        color: #2563eb;
    }
    .step-pill.done {
        border-color: #10b981;
        background: rgba(16, 185, 129, 0.1);
        color: #059669;
    }

    /* Button Styling */
    div[data-testid="stDownloadButton"] button,
    div[data-testid="stButton"] button {
        border-radius: 10px;
        font-weight: 600;
        padding: 0.55rem 1rem;
        transition: transform 0.1s ease, box-shadow 0.2s ease;
    }
    div[data-testid="stButton"] button:hover {
        transform: translateY(-1px);
    }

    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: rgba(140, 140, 140, 0.07);
        border: 1px solid rgba(140, 140, 140, 0.14);
        border-radius: 12px;
        padding: 10px 14px;
    }
    div[data-testid="stMetric"] label {
        font-weight: 600;
        font-size: 0.8rem;
    }

    /* Info Badge Box */
    .info-card {
        padding: 12px 16px;
        border-radius: 10px;
        background: rgba(140, 140, 140, 0.06);
        border-left: 4px solid #3b82f6;
        margin: 10px 0;
        font-size: 0.9rem;
    }

    /* Footer */
    .footer-text {
        text-align: center;
        font-size: 0.82rem;
        opacity: 0.75;
        padding-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ------------------------------------------------------------
# SESSION STATE AWAL
# ------------------------------------------------------------

DEFAULTS = {
    "uploader_version": 0,
    "generated_result": None,
    "last_ocr_signature": None,
}

for state_key, default_value in DEFAULTS.items():

    if state_key not in st.session_state:

        st.session_state[state_key] = default_value


def reset_app():

    st.session_state.uploader_version += 1

    st.session_state.last_ocr_signature = None

    st.session_state.generated_result = None


uploader_version = st.session_state.uploader_version

timestamp_key = f"timestamp_box_{uploader_version}"


# ------------------------------------------------------------
# HERO HEADER & STEP STATUS BAR
# ------------------------------------------------------------

st.markdown('<div class="hero-badge">⚡ V7.5 • AVHershey Font Engine + Auto-Fit</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">🕒 Timestamp Generator</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-desc">Watermark timestamp presisi tinggi gaya <b>GPS Map Camera</b> '
    'dengan normalisasi orientasi EXIF, auto-fit font, & OCR bertenaga <b>Gemini AI</b>.</div>',
    unsafe_allow_html=True
)


# ------------------------------------------------------------
# 1. UPLOAD FOTO UTAMA
# ------------------------------------------------------------

st.markdown("### 1️⃣ Upload Foto Utama")

image_cv = None
photo_file = None

with st.container(border=True):

    photo_file = st.file_uploader(
        "Pilih file foto (JPG, JPEG, PNG)",
        type=["jpg", "jpeg", "png"],
        key=f"photo_uploader_{uploader_version}",
        help="Foto akan dinormalisasi orientasi EXIF-nya secara otomatis agar watermark selalu tegak lurus."
    )

    if photo_file is not None:

        try:

            photo_pil = load_pil_image_fixed_orientation(
                photo_file
            )

            image_cv = np.array(
                photo_pil
            )

        except Exception as error:

            st.error(
                f"❌ Foto tidak dapat dibaca: {error}"
            )

            image_cv = None

        if image_cv is not None:

            height, width = image_cv.shape[:2]

            col_name, col_res, col_aspect = st.columns(3)

            with col_name:

                st.metric(
                    "📄 File",
                    photo_file.name if len(photo_file.name) <= 18 else photo_file.name[:15] + "..."
                )

            with col_res:

                st.metric(
                    "📐 Resolusi",
                    f"{width} × {height} px"
                )

            with col_aspect:

                aspect = width / height if height > 0 else 1.0
                aspect_label = "Landscape" if aspect > 1.1 else ("Portrait" if aspect < 0.9 else "Square")

                st.metric(
                    "🧭 Orientasi",
                    f"{aspect_label} ({aspect:.2f})"
                )

            st.image(
                image_cv,
                caption="Preview Foto Asli (Orientasi EXIF Normal)",
                use_container_width=True
            )

    else:

        st.info(
            "📷 **Belum ada foto yang dipilih.** Silakan unggah foto utama yang ingin diberi watermark timestamp."
        )


# ------------------------------------------------------------
# 2. MASUKKAN TIMESTAMP
# ------------------------------------------------------------

st.markdown("### 2️⃣ Masukkan Teks Timestamp")

with st.container(border=True):

    tab_manual, tab_ocr = st.tabs(
        [
            "✍️ Ketik Manual",
            "🔍 OCR Screenshot (Gemini AI)"
        ]
    )

    with tab_manual:

        st.caption(
            "💡 Masukkan tepat **3 baris**: baris 1 untuk koordinat, "
            "baris 2 untuk nama lokasi, dan baris 3 untuk tanggal & jam. "
            "Baris yang kepanjangan akan otomatis dikecilkan ukuran fontnya "
            "supaya tidak kepotong (lihat detail setelah Generate)."
        )

    with tab_ocr:

        gemini_api_key = st.secrets.get(
            "GEMINI_API_KEY",
            ""
        )

        if not gemini_api_key:

            st.warning(
                "⚠️ **`GEMINI_API_KEY` belum dikonfigurasi di Streamlit Secrets.** "
                "Fitur OCR otomatis membutuhkan API key di pengaturan Secrets."
            )
        else:

            st.caption(
                "✨ **Gemini OCR Siap.** Screenshot area timestamp (Win+Shift+S / Cmd+Shift+4), "
                "lalu paste di bawah atau unggah filenya."
            )

        col_paste_btn, col_upload_ocr = st.columns([1, 1])

        with col_paste_btn:

            paste_result = pbutton(
                label="📋 Klik lalu Tekan Ctrl+V (Paste)",
                key=f"screenshot_paste_{uploader_version}"
            )

        with col_upload_ocr:

            screenshot_file = st.file_uploader(
                "Atau upload screenshot",
                type=["jpg", "jpeg", "png"],
                key=f"screenshot_uploader_{uploader_version}",
                label_visibility="collapsed"
            )

        screenshot_image = None

        if paste_result.image_data is not None:

            try:

                screenshot_image = ImageOps.exif_transpose(
                    paste_result.image_data
                ).convert("RGB")

            except Exception as error:

                st.error(
                    f"❌ Screenshot hasil paste tidak dapat dibaca: {error}"
                )

                screenshot_image = None

        elif screenshot_file is not None:

            try:

                screenshot_image = load_pil_image_fixed_orientation(
                    screenshot_file
                )

            except Exception as error:

                st.error(
                    f"❌ Screenshot tidak dapat dibaca: {error}"
                )

                screenshot_image = None

        if screenshot_image is not None:

            st.image(
                screenshot_image,
                caption="Preview Screenshot untuk OCR",
                use_container_width=True
            )

            # ----------------------------------------------------
            # Jalankan OCR OTOMATIS begitu ada screenshot baru
            # ----------------------------------------------------

            image_buffer_for_hash = io.BytesIO()

            screenshot_image.save(
                image_buffer_for_hash,
                format="PNG"
            )

            screenshot_signature = hashlib.md5(
                image_buffer_for_hash.getvalue()
            ).hexdigest()

            already_processed = (
                st.session_state.last_ocr_signature == screenshot_signature
            )

            col_ocr_status, col_ocr_button = st.columns([3, 1])

            with col_ocr_button:

                rerun_ocr_clicked = st.button(
                    "🔁 OCR Ulang",
                    type="secondary",
                    disabled=not gemini_api_key,
                    use_container_width=True
                )

            should_run_ocr = (
                gemini_api_key
                and (not already_processed or rerun_ocr_clicked)
            )

            if should_run_ocr:

                try:

                    with col_ocr_status:

                        with st.spinner("⏳ Meminta Gemini membaca teks timestamp..."):

                            ocr_result = ocr_extract_text_gemini(
                                screenshot_image,
                                gemini_api_key
                            )

                    st.session_state.last_ocr_signature = screenshot_signature

                    if ocr_result:

                        st.session_state[timestamp_key] = ocr_result

                        st.success(
                            "✅ OCR Berhasil! Teks sudah dimasukkan ke kotak di bawah."
                        )

                        st.rerun()

                    else:

                        st.warning("⚠️ Gemini tidak mengembalikan teks apapun.")

                except Exception as error:

                    st.error(f"❌ ERROR OCR GEMINI: {error}")

            elif already_processed:

                st.caption("✅ Screenshot ini sudah berhasil di-OCR.")

    # --------------------------------------------------------
    # KOTAK INPUT TEXT TIMESTAMP DENGAN LIVE VALIDATOR
    # --------------------------------------------------------

    st.markdown("---")

    timestamp_text = st.text_area(
        "Isi Teks Timestamp (Wajib 3 Baris)",
        height=130,
        placeholder=(
            "-5.7297568, 105.6231008\n"
            "Sukaratu, Lampung Selatan, Lampung\n"
            "18/8/2026 10:12:35 WIB"
        ),
        key=timestamp_key,
        help="Baris 1: Koordinat | Baris 2: Nama Tempat | Baris 3: Tanggal, Jam & Zona Waktu"
    )

    # Validasi visual baris input
    raw_lines = [line.strip() for line in timestamp_text.splitlines() if line.strip()]
    
    if len(raw_lines) == 3:
        st.caption("🟢 **Format valid**: Terdeteksi 3 baris lengkap siap digenerate.")
    elif len(raw_lines) > 0:
        st.caption(f"🟡 **Format belum pas**: Terdeteksi {len(raw_lines)} baris. Dibutuhkan tepat 3 baris teks.")


# ------------------------------------------------------------
# 3. GENERATE ACTION
# ------------------------------------------------------------

st.markdown("### 3️⃣ Generate Watermark")

photo_ready = image_cv is not None
timestamp_ready = bool(timestamp_text.strip())

with st.container(border=True):

    if not photo_ready or not timestamp_ready:

        missing = []

        if not photo_ready:
            missing.append("Upload Foto Utama (Langkah 1)")

        if not timestamp_ready:
            missing.append("Teks Timestamp (Langkah 2)")

        st.info("⏳ **Menunggu input**: Lengkapi " + " & ".join(missing) + " untuk mengaktifkan tombol.")

    generate_clicked = st.button(
        "🚀 GENERATE TIMESTAMP SEKARANG",
        type="primary",
        disabled=not (photo_ready and timestamp_ready),
        use_container_width=True
    )

    if generate_clicked:

        try:

            with st.spinner("🖌️ Sedang merender font Hershey Simplex & menggambar timestamp..."):

                pil_image, info = generate_timestamp_image(
                    image_cv,
                    timestamp_text
                )

            # ----------------------------------------------------
            # SIAPKAN FILE UNTUK DOWNLOAD
            # ----------------------------------------------------

            name, ext = os.path.splitext(
                photo_file.name
            )

            if not ext:

                ext = ".jpg"

            random_word = (
                secrets.choice(RANDOM_WORD_LIST_A) +
                secrets.choice(RANDOM_WORD_LIST_B)
            )

            output_file_name = (
                name +
                "_" +
                random_word +
                ext
            )

            result_rgb = np.array(
                pil_image
            )

            result_cv = cv2.cvtColor(
                result_rgb,
                cv2.COLOR_RGB2BGR
            )

            if ext.lower() in [
                ".jpg",
                ".jpeg"
            ]:

                encode_success, encoded_bytes = cv2.imencode(
                    ext,
                    result_cv,
                    [
                        cv2.IMWRITE_JPEG_QUALITY,
                        95
                    ]
                )

                mime_type = "image/jpeg"

            else:

                encode_success, encoded_bytes = cv2.imencode(
                    ext,
                    result_cv
                )

                mime_type = "image/png"

            if encode_success:

                st.session_state.generated_result = {
                    "pil_image": pil_image,
                    "info": info,
                    "output_file_name": output_file_name,
                    "encoded_bytes": encoded_bytes.tobytes(),
                    "mime_type": mime_type,
                }

                st.rerun()

            else:

                st.error("❌ Gagal encode gambar hasil.")

        except Exception as error:

            st.error(f"❌ Terjadi kesalahan: {error}")


# ------------------------------------------------------------
# 4. HASIL (PERSIST DI SESSION STATE)
# ------------------------------------------------------------

result = st.session_state.generated_result

if result is not None:

    st.markdown("---")
    st.markdown("### ✨ Foto Hasil Timestamp")

    with st.container(border=True):

        st.success("🎉 **Timestamp Berhasil Digambar!** Foto siap diunduh dengan kualitas tinggi.")

        if result["info"].get("auto_fit_applied"):

            st.info(
                "📏 **Auto-fit aktif**: salah satu baris teksnya cukup panjang, "
                "jadi ukuran font otomatis dikecilkan sedikit supaya semua teks "
                "tetap muat & tidak kepotong."
            )

        st.image(
            result["pil_image"],
            caption=f"Preview Hasil: {result['output_file_name']}",
            use_container_width=True
        )

        with st.expander("📊 Detail Teks & Parameter Rendering", expanded=True):

            info = result["info"]

            st.markdown(
                f"""
                <div class="info-card">
                    📍 <b>Koordinat:</b> <code>{info['line1']}</code><br>
                    🏢 <b>Lokasi:</b> <code>{info['line2']}</code><br>
                    ⏰ <b>Waktu:</b> <code>{info['line3']}</code>
                </div>
                """,
                unsafe_allow_html=True
            )

            m_col1, m_col2, m_col3, m_col4 = st.columns(4)

            with m_col1:
                st.metric("Font Size", f"{info['font_size']} px")
            with m_col2:
                st.metric("Outline Width", f"{info['outline_width']} px")
            with m_col3:
                st.metric("Margin Kiri", f"{info['left_margin']} px")
            with m_col4:
                st.metric("Margin Bawah", f"{info['bottom_margin']} px")

            st.caption(
                f"• Font: AVHershey Simplex Medium | Scale Ratio: {info['scale_ratio']:.4f} | "
                f"Line Spacing: {info['line_spacing']}px | Letter Spacing: {info['letter_spacing']:.2f}px | "
                f"Auto-fit: {'Ya' if info['auto_fit_applied'] else 'Tidak perlu'}"
            )

        col_download, col_reset = st.columns([2, 1])

        with col_download:

            st.download_button(
                label=f"⬇️ DOWNLOAD FOTO ({result['output_file_name']})",
                data=result["encoded_bytes"],
                file_name=result["output_file_name"],
                mime=result["mime_type"],
                type="primary",
                use_container_width=True
            )

        with col_reset:

            if st.button(
                "🔄 Reset & Buat Baru",
                type="secondary",
                use_container_width=True
            ):

                reset_app()
                st.rerun()


# ------------------------------------------------------------
# FOOTER & CREDITS
# ------------------------------------------------------------

st.markdown("---")

st.markdown(
    """
    <div class="footer-text">
        Dibuat oleh <b>Matthew Artur Panahatan Sitorus</b> & <b>Nikita Adriella Virginia Jacob</b><br>
        <span style="font-size:0.75rem; opacity:0.8;">AVHershey Simplex Medium • Gemini AI Vision OCR • OpenCV Engine</span>
    </div>
    """,
    unsafe_allow_html=True
)
