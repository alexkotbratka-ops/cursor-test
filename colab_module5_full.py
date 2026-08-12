# =============================================================================
# МОДУЛЬ 5 — Полный анализ тендера (единый код для Google Colab)
# Скопируйте ВЕСЬ файл в ОДНУ ячейку Colab и выполните (без правок).
#
# Последовательно:
#   1) установка зависимостей
#   2) API-ключ DeepSeek (getpass)
#   3) загрузка файлов (files.upload)
#   4) OCR / извлечение / индексация (дедуп, chunk=600)
#   5) анализ 86 вопросов + классификация + «Согласования»
#   6) TXT-отчёт + files.download + сводка по времени
#
# OCR (смешанный контент):
#   - dpi=200, lang=rus+eng, psm=6 (страница целиком)
#   - PDF: OCR если мало текста ИЛИ есть изображения/графика
#   - DOCX: OCR всех встроенных изображений
#   - цифровой текст сохраняется; OCR дополняет без дублей
#   - прогресс: «OCR: страница X из Y — файл»
# =============================================================================

import time as _time_boot

_TIMINGS = {}
_T0_ALL = _time_boot.time()


def _mark(name: str):
    _TIMINGS[name] = _time_boot.time()


def _fmt_dur(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    if seconds < 60:
        return f"{seconds} сек."
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{m} мин. {s} сек."
    h, m = divmod(m, 60)
    return f"{h} ч. {m} мин."


def _elapsed(name_from: str, name_to: str = None) -> float:
    a = _TIMINGS.get(name_from, _T0_ALL)
    b = _TIMINGS.get(name_to, _time_boot.time()) if name_to else _time_boot.time()
    return b - a


_CURRENT_STATUS = "инициализация"


def _status(msg: str) -> None:
    """Текущий статус выполнения + сколько уже прошло с старта модуля 5."""
    global _CURRENT_STATUS
    _CURRENT_STATUS = msg
    print(
        f"🔄 СТАТУС: {msg}  | ⏱ прошло {_fmt_dur(_elapsed('start'))}",
        flush=True,
    )


def _manual_download_link(filename: str) -> str:
    base = os.path.basename(filename) if "os" in dir() else filename
    try:
        import os as _os
        base = _os.path.basename(filename)
        if _os.path.isdir("/content"):
            abs_path = _os.path.abspath(filename)
            if abs_path.startswith("/content/"):
                return abs_path
            return f"/content/{base}"
        return _os.path.abspath(filename)
    except Exception:
        return f"/content/{filename}"


_mark("start")
_mark("deps_start")
_status("этап 1/5 — установка зависимостей и инициализация")

# =============================================================================
# ЭТАП 1/5 — Установка зависимостей и инициализация
# =============================================================================
print("=" * 70)
print("🚀 МОДУЛЬ 5 — Полный анализ тендера (единый код)")
print("ЭТАП 1/5 — Установка зависимостей и инициализация")
print("=" * 70)

# --- 1. Установка системных пакетов и Python-библиотек ---
import subprocess
import sys


def _run(cmd, check=False):
    return subprocess.run(
        cmd,
        check=check,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


print("📦 Установка системных пакетов...")
_run(["apt-get", "update", "-qq"])
_run(
    [
        "apt-get",
        "install",
        "-y",
        "-qq",
        "antiword",
        "tesseract-ocr",
        "tesseract-ocr-rus",
        "tesseract-ocr-eng",
        "poppler-utils",
        "unrar",
        "p7zip-full",
    ]
)

print("📦 Установка Python-библиотек...")
subprocess.check_call(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "pymupdf",
        "python-docx",
        "openpyxl",
        "xlrd",
        "odfpy",
        "striprtf",
        "beautifulsoup4",
        "lxml",
        "pytesseract",
        "pdf2image",
        "Pillow",
        "scikit-learn",
        "requests",
        "rarfile",
        "py7zr",
        "ezdxf",
        "python-pptx",
        "numpy",
        "extract-msg",
    ],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.STDOUT,
)
print("✅ Зависимости установлены.\n")


# --- 2. Импорты ---
import csv
import gzip
import io
import json
import os
import re
import tarfile
import tempfile
import time
import zipfile
import xml.etree.ElementTree as ET
from getpass import getpass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import fitz  # pymupdf
import numpy as np
import py7zr
import pytesseract
import rarfile
import requests
from bs4 import BeautifulSoup
from docx import Document
from openpyxl import load_workbook
from pdf2image import convert_from_bytes
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from pptx import Presentation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from striprtf.striprtf import rtf_to_text

try:
    import extract_msg
except ImportError:
    extract_msg = None

try:
    import email
    from email import policy as email_policy
except ImportError:
    email = None
    email_policy = None

try:
    import ezdxf
    from ezdxf import recover as ezdxf_recover
except ImportError:
    ezdxf = None
    ezdxf_recover = None

try:
    from odf.opendocument import load as odf_load
    from odf import text as odf_text
    from odf import table as odf_table
    from odf.teletype import extractText as odf_extract_text
except ImportError:
    odf_load = None

try:
    import xlrd
except ImportError:
    xlrd = None

# --- 3. API-ключ DeepSeek (сохраняется в переменной сессии) ---
DEEPSEEK_API_KEY = getpass("🔑 Введите API-ключ DeepSeek: ").strip()
os.environ["DEEPSEEK_API_KEY"] = DEEPSEEK_API_KEY

DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"

# --- Настройки RAG / OCR (ключевые константы) ---
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
TOP_K = 5
OCR_LANG = "rus+eng"                 # русский + английский
DEEPSEEK_TIMEOUT = 60                # сек. на ответ API

# Пороги OCR для смешанных тендерных документов (текст + сканы)
OCR_MIN_CHARS_PER_PAGE = 30          # мало текста → страница почти наверняка скан
OCR_MIN_ALPHA_RATIO = 0.35           # доля букв среди непробельных символов
OCR_IMAGE_AREA_RATIO = 0.08          # любая заметная картинка на странице → OCR
OCR_DPI = 200                        # dpi рендера PDF для OCR
OCR_PDF_FORCE_FULL_IF_AVG_BELOW = 15 # средний символов/стр. → полный OCR всего PDF
OCR_OVERLAP_SKIP = 0.75              # совпадение токенов OCR↔цифровой текст → пропуск дубля

SUPPORTED_DOCS = {
    # документы
    ".docx",
    ".doc",
    ".rtf",
    ".odt",
    ".pdf",
    ".txt",
    ".log",
    ".md",
    ".html",
    ".htm",
    ".xml",
    ".json",
    # почта
    ".eml",
    ".msg",
    # таблицы
    ".xlsx",
    ".xls",
    ".csv",
    ".ods",
    # изображения (всегда полный OCR)
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".gif",
    ".tif",
    ".tiff",
    ".webp",
    ".jfif",
    ".heic",
    # специфические
    ".dwg",
    ".dxf",
    ".ppt",
    ".pptx",
}

SUPPORTED_ARCHIVES = {
    ".zip",
    ".rar",
    ".7z",
    ".tar",
    ".gz",
    ".tgz",
    ".tar.gz",
}

SUPPORTED_EXTENSIONS = SUPPORTED_DOCS | SUPPORTED_ARCHIVES


def file_ext(filename: str) -> str:
    """Расширение с учётом составных (.tar.gz) и Windows-путей '\\'."""
    name = str(filename).replace("\\", "/").split("/")[-1].lower()
    if name.endswith(".tar.gz"):
        return ".tar.gz"
    if name.endswith(".tar.bz2"):
        return ".tar.bz2"
    return Path(name).suffix.lower()


# =============================================================================
# OCR helpers — смешанный контент: цифровой текст + сканы/печати/схемы
# =============================================================================

def _preprocess_for_ocr(img: Image.Image) -> Image.Image:
    """Улучшение читаемости сканов писем/штампов перед Tesseract."""
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    # увеличиваем мелкий текст (типично для фото/сканов A4)
    w, h = img.size
    if max(w, h) < 1600:
        scale = 1600 / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    gray = ImageOps.exif_transpose(img).convert("L")
    gray = ImageOps.autocontrast(gray)
    gray = ImageEnhance.Contrast(gray).enhance(1.4)
    gray = ImageEnhance.Sharpness(gray).enhance(1.2)
    gray = gray.filter(ImageFilter.MedianFilter(size=3))
    return gray


def ocr_pil_image(img: Image.Image, filename: str = "") -> str:
    """OCR PIL-изображения: один основной проход psm=6 (без тройного PSM)."""
    try:
        processed = _preprocess_for_ocr(img)
        # Режим «страница целиком» — один раз (не гоняем psm 4/3 дополнительно)
        text = pytesseract.image_to_string(
            processed, lang=OCR_LANG, config="--oem 3 --psm 6"
        ).strip()
        # Запасной проход только если почти пусто
        if len(text) < 20:
            try:
                raw = img.convert("RGB") if img.mode != "RGB" else img
                alt = pytesseract.image_to_string(
                    raw, lang=OCR_LANG, config="--oem 3 --psm 6"
                ).strip()
                if len(alt) > len(text):
                    text = alt
            except Exception:
                pass
        return text
    except Exception as e:
        print(f"  ⚠️ OCR изображение {filename}: {e}")
        return ""


def _text_quality_stats(text: str) -> Dict[str, float]:
    text = text or ""
    chars = len(text.strip())
    nonspace = [c for c in text if not c.isspace()]
    alpha = sum(1 for c in nonspace if c.isalpha())
    alpha_ratio = (alpha / len(nonspace)) if nonspace else 0.0
    return {"chars": float(chars), "alpha_ratio": float(alpha_ratio)}


def _page_image_area_ratio(page) -> float:
    """Оценка доли площади страницы, занятой встроенными изображениями."""
    try:
        rect = page.rect
        page_area = abs(rect.width * rect.height) or 1.0
        img_area = 0.0
        for info in page.get_image_info(xrefs=True) or []:
            bbox = info.get("bbox")
            if not bbox:
                continue
            x0, y0, x1, y1 = bbox
            img_area += abs((x1 - x0) * (y1 - y0))
        return min(img_area / page_area, 1.0)
    except Exception:
        # fallback: есть ли картинки вообще
        try:
            return 0.8 if page.get_images(full=True) else 0.0
        except Exception:
            return 0.0


def _normalize_for_ocr_overlap(text: str) -> str:
    t = (text or "").lower()
    return re.sub(r"\s+", " ", t).strip()


def _ocr_token_set(text: str) -> set:
    return set(re.findall(r"[а-яёa-z0-9]{3,}", _normalize_for_ocr_overlap(text), flags=re.I))


def ocr_supplement_text(digital: str, ocr: str) -> str:
    """
    Возвращает OCR-текст, который дополняет цифровой слой (без дублей).
    Если OCR почти целиком уже есть в digital — "".
    """
    ocr = (ocr or "").strip()
    if not ocr:
        return ""
    digital = (digital or "").strip()
    if not digital:
        return ocr

    ocr_norm = _normalize_for_ocr_overlap(ocr)
    dig_norm = _normalize_for_ocr_overlap(digital)
    if len(ocr_norm) >= 40 and ocr_norm in dig_norm:
        return ""

    ocr_tok = _ocr_token_set(ocr)
    if not ocr_tok:
        return ""
    dig_tok = _ocr_token_set(digital)
    overlap = len(ocr_tok & dig_tok) / len(ocr_tok)
    if overlap >= float(globals().get("OCR_OVERLAP_SKIP", 0.75) or 0.75):
        return ""
    return ocr


def pdf_page_has_visual_content(page) -> bool:
    """
    Есть ли на странице сканы/фото/подписи/печати/схемы/чертежи.
    Любое встроенное изображение или заметная векторная графика → True.
    """
    try:
        if page.get_images(full=True):
            return True
    except Exception:
        pass
    try:
        if page.get_image_info(xrefs=True):
            return True
    except Exception:
        pass
    try:
        drawings = page.get_drawings() or []
        # таблицы/схемы/штампы часто как векторные path'ы без text layer
        if len(drawings) >= 5:
            return True
    except Exception:
        pass
    return _page_image_area_ratio(page) >= float(
        globals().get("OCR_IMAGE_AREA_RATIO", 0.08) or 0.08
    )


def page_needs_ocr(page, digital_text: str) -> bool:
    """
    OCR для смешанных тендерных PDF:
    - мало цифрового текста (< OCR_MIN_CHARS_PER_PAGE), ИЛИ
    - на странице есть изображения / графика / схемы.
    Цифровой текст сохраняется; OCR только дополняет (см. ocr_supplement_text).
    """
    stats = _text_quality_stats(digital_text)
    if stats["chars"] < OCR_MIN_CHARS_PER_PAGE:
        return True
    return pdf_page_has_visual_content(page)


def ocr_pdf_page(page, page_no: int, filename: str = "", page_count: int = 0) -> str:
    """Рендер страницы PDF целиком → OCR (dpi=OCR_DPI, rus+eng, psm=6)."""
    try:
        total = page_count if page_count else "?"
        print(f"  🔍 OCR: страница {page_no} из {total} — {filename}", flush=True)
        zoom = OCR_DPI / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        text = ocr_pil_image(img, f"{filename}#p{page_no}")
        return text
    except Exception as e:
        print(f"  ⚠️ OCR страницы {page_no} ({filename}): {e}")
        return ""


# Кэш результата PDF + защита от параллельного повторного OCR
_PDF_TEXT_CACHE: Dict[str, str] = {}
_PDF_OCR_INFLIGHT: Dict[str, Any] = {}
try:
    import threading as _threading_pdf
    _PDF_TEXT_LOCK = _threading_pdf.Lock()
except Exception:
    _threading_pdf = None  # type: ignore
    _PDF_TEXT_LOCK = None


def _pdf_cache_get(cache_key: str) -> Optional[str]:
    if _PDF_TEXT_LOCK is None:
        return _PDF_TEXT_CACHE.get(cache_key)
    with _PDF_TEXT_LOCK:
        return _PDF_TEXT_CACHE.get(cache_key)


def _pdf_cache_put(cache_key: str, value: str) -> str:
    if _PDF_TEXT_LOCK is None:
        _PDF_TEXT_CACHE[cache_key] = value
        return value
    with _PDF_TEXT_LOCK:
        _PDF_TEXT_CACHE[cache_key] = value
        ev = _PDF_OCR_INFLIGHT.pop(cache_key, None)
        if ev is not None:
            try:
                ev.set()
            except Exception:
                pass
    return value


def _pdf_begin_work(cache_key: str):
    """
    Возвращает (cached_text | None, wait_event | None, is_owner).
    is_owner=True — этот поток должен выполнить OCR.
    """
    if _PDF_TEXT_LOCK is None or _threading_pdf is None:
        cached = _PDF_TEXT_CACHE.get(cache_key)
        return cached, None, cached is None
    with _PDF_TEXT_LOCK:
        cached = _PDF_TEXT_CACHE.get(cache_key)
        if cached is not None:
            return cached, None, False
        if cache_key in _PDF_OCR_INFLIGHT:
            return None, _PDF_OCR_INFLIGHT[cache_key], False
        ev = _threading_pdf.Event()
        _PDF_OCR_INFLIGHT[cache_key] = ev
        return None, None, True


def extract_text_from_pdf(file_bytes: bytes, filename: str = "") -> str:
    """
    PDF: ровно ОДИН проход OCR на документ (флаг ocr_done).

    Алгоритм:
      1) Быстрый обзор — только цифровой текст (OCR ещё нет).
      2) Если PDF — скан → один полный OCR, ocr_done=True, выход.
      3) Если смешанный → OCR только нужных страниц, ocr_done=True.
      4) После ocr_done=True полный OCR больше НЕ вызывается
         (нет повторной проверки в конце; полный OCR вне except-retry).
    """
    import hashlib as _hashlib

    cache_key = _hashlib.md5(file_bytes).hexdigest()
    cached, wait_ev, is_owner = _pdf_begin_work(cache_key)
    if cached is not None:
        print(f"  ⏭️ PDF уже извлечён (кэш, без повторного OCR): {filename}", flush=True)
        return cached
    if not is_owner and wait_ev is not None:
        print(f"  ⏳ PDF уже OCR-ится другим потоком, ждём: {filename}", flush=True)
        try:
            wait_ev.wait(timeout=3600)
        except Exception:
            pass
        cached2 = _pdf_cache_get(cache_key)
        return cached2 if cached2 is not None else ""

    ocr_done = False
    need_full_ocr = False
    result = ""
    page_count = 0
    parts: List[str] = []
    ocr_pages = 0

    # --- Анализ PDF (без полного OCR внутри try, чтобы except не делал retry) ---
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        page_count = int(doc.page_count or 0)
        print(
            f"  📄 PDF {filename}: {page_count} стр. — сначала цифровой слой (без OCR)…",
            flush=True,
        )

        digital_list: List[str] = []
        need_ocr_flags: List[bool] = []
        digital_chars_total = 0
        pages_with_digital = 0

        for page in doc:
            digital = (page.get_text("text") or "").strip()
            digital_list.append(digital)
            digital_chars_total += len(digital)
            if digital:
                pages_with_digital += 1
            need_ocr_flags.append(page_needs_ocr(page, digital))

        avg = (digital_chars_total / page_count) if page_count else 0.0
        force_below = float(globals().get("OCR_PDF_FORCE_FULL_IF_AVG_BELOW", 15) or 15)
        is_full_scan = bool(page_count) and avg < force_below and pages_with_digital == 0

        if is_full_scan:
            doc.close()
            print(
                f"  🔍 PDF похож на скан (avg={avg:.0f} симв/стр) — "
                f"нужен один полный OCR: {filename}",
                flush=True,
            )
            need_full_ocr = True
        else:
            pages_to_ocr = sum(1 for flag in need_ocr_flags if flag)
            if pages_to_ocr:
                print(
                    f"  🔍 PDF смешанный: OCR {pages_to_ocr} из {page_count} стр. — {filename}",
                    flush=True,
                )
            else:
                print(
                    f"  ✅ PDF {filename}: OCR не нужен (есть цифровой текст, нет сканов)",
                    flush=True,
                )

            for i, page in enumerate(doc, start=1):
                digital = digital_list[i - 1]
                page_bits: List[str] = []
                if digital:
                    page_bits.append(digital)

                if need_ocr_flags[i - 1]:
                    ocr_text = ocr_pdf_page(page, i, filename, page_count=page_count)
                    if ocr_text:
                        extra = ocr_supplement_text(digital, ocr_text)
                        if extra:
                            ocr_pages += 1
                            label = "[OCR-дополнение]" if digital else "[OCR]"
                            page_bits.append(f"{label}\n{extra}")

                if page_bits:
                    parts.append(f"[Страница {i}]\n" + "\n".join(page_bits))
                else:
                    print(f"  ⚠️ Страница {i} без текста даже после OCR: {filename}")

            doc.close()
            if pages_to_ocr > 0:
                ocr_done = True  # постраничный OCR уже был — полный запрещён
            result = "\n\n".join(parts)
            if ocr_pages:
                print(
                    f"  ✅ PDF {filename}: OCR выполнен один раз на {ocr_pages}/{page_count} стр.",
                    flush=True,
                )
    except Exception as e:
        print(f"  ❌ Ошибка PDF {filename}: {e}")
        # Только если OCR ещё не делали — пометим необходимость полного OCR
        if not ocr_done:
            need_full_ocr = True

    # --- Полный OCR максимум один раз, ВНЕ except-retry ---
    if need_full_ocr and not ocr_done:
        print(f"  🔍 Полный OCR запускается один раз: {filename}", flush=True)
        result = ocr_pdf_bytes(file_bytes, filename)
        ocr_done = True
    elif need_full_ocr and ocr_done:
        print(
            f"  ⏭️ OCR уже выполнен (ocr_done=True) — повторный полный OCR пропущен: {filename}",
            flush=True,
        )

    return _pdf_cache_put(cache_key, result)


def extract_text_from_docx(file_bytes: bytes, filename: str = "") -> str:
    """
    DOCX: цифровой текст + OCR ВСЕХ встроенных изображений
    (сканы, подписи, печати, таблицы-картинки). OCR дополняет, не дублирует.
    """
    parts: List[str] = []
    digital_parts: List[str] = []
    try:
        doc = Document(io.BytesIO(file_bytes))
        for para in doc.paragraphs:
            if para.text.strip():
                digital_parts.append(para.text)
        for table in doc.tables:
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells if c.text.strip()]
                if cells:
                    digital_parts.append(" | ".join(cells))
        parts.extend(digital_parts)
        digital_blob = "\n".join(digital_parts)

        # встроенные картинки — всегда OCR, если есть
        try:
            image_rels = []
            for rel in doc.part.rels.values():
                try:
                    if "image" in getattr(rel, "reltype", ""):
                        image_rels.append(rel)
                except Exception:
                    continue
            total_imgs = len(image_rels)
            if total_imgs:
                print(f"  🖼️ DOCX {filename}: найдено изображений {total_imgs} — запускаем OCR")
            for img_idx, rel in enumerate(image_rels, start=1):
                try:
                    blob = rel.target_part.blob
                    print(f"  🔍 OCR: изображение {img_idx} из {total_imgs} — {filename}")
                    ocr_text = extract_text_from_image(blob, f"{filename}#img{img_idx}")
                    extra = ocr_supplement_text(digital_blob, ocr_text)
                    if extra:
                        parts.append(f"[Изображение {img_idx} | OCR]\n{extra}")
                        # учитываем уже добавленный OCR при следующих картинках
                        digital_blob = digital_blob + "\n" + extra
                except Exception:
                    continue
        except Exception as e:
            print(f"  ⚠️ DOCX images OCR {filename}: {e}")
    except Exception as e:
        print(f"  ❌ Ошибка DOCX {filename}: {e}")
    return "\n".join(parts)


def extract_text_from_doc(file_bytes: bytes, filename: str = "") -> str:
    """Извлекает текст из старых .doc (Word 97–2003) через antiword."""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".doc", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        result = subprocess.run(
            ["antiword", "-m", "UTF-8.txt", tmp_path],
            capture_output=True,
            check=False,
        )
        raw = result.stdout or b""
        if not raw:
            result = subprocess.run(
                ["antiword", tmp_path],
                capture_output=True,
                check=False,
            )
            raw = result.stdout or b""

        if not raw:
            err = (result.stderr or b"").decode("utf-8", errors="replace").strip()
            print(f"  ❌ antiword не извлёк текст из {filename}" + (f": {err}" if err else ""))
            return ""

        for encoding in ("utf-8", "cp1251", "latin-1"):
            try:
                return raw.decode(encoding).strip()
            except UnicodeDecodeError:
                continue
        return raw.decode("latin-1", errors="replace").strip()
    except FileNotFoundError:
        print(f"  ❌ antiword не установлен — не удалось прочитать {filename}")
        return ""
    except Exception as e:
        print(f"  ❌ Ошибка DOC {filename}: {e}")
        return ""
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def extract_text_from_rtf(file_bytes: bytes, filename: str = "") -> str:
    try:
        raw = None
        for encoding in ("utf-8", "cp1251", "latin-1"):
            try:
                raw = file_bytes.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        if raw is None:
            raw = file_bytes.decode("latin-1", errors="replace")
        return (rtf_to_text(raw) or "").strip()
    except Exception as e:
        print(f"  ❌ Ошибка RTF {filename}: {e}")
        return ""


def extract_text_from_odt(file_bytes: bytes, filename: str = "") -> str:
    if odf_load is None:
        print(f"  ❌ odfpy не установлен — пропуск {filename}")
        return ""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".odt", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        doc = odf_load(tmp_path)
        parts: List[str] = []
        for el in doc.getElementsByType(odf_text.P) + doc.getElementsByType(odf_text.H):
            t = odf_extract_text(el).strip()
            if t:
                parts.append(t)
        for table in doc.getElementsByType(odf_table.Table):
            for row in table.getElementsByType(odf_table.TableRow):
                cells = []
                for cell in row.getElementsByType(odf_table.TableCell):
                    ct = odf_extract_text(cell).strip()
                    if ct:
                        cells.append(ct)
                if cells:
                    parts.append(" | ".join(cells))
        return "\n".join(parts)
    except Exception as e:
        print(f"  ❌ Ошибка ODT {filename}: {e}")
        return ""
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def extract_text_from_txt(file_bytes: bytes, filename: str = "") -> str:
    for encoding in ("utf-8", "utf-16", "cp1251", "latin-1"):
        try:
            return file_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    return file_bytes.decode("latin-1", errors="replace")


def extract_text_from_html(file_bytes: bytes, filename: str = "") -> str:
    try:
        raw = extract_text_from_txt(file_bytes, filename)
        soup = BeautifulSoup(raw, "lxml")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        lines = [ln.strip() for ln in text.splitlines()]
        return "\n".join(ln for ln in lines if ln)
    except Exception as e:
        print(f"  ❌ Ошибка HTML {filename}: {e}")
        return ""


def extract_text_from_xml(file_bytes: bytes, filename: str = "") -> str:
    try:
        raw = extract_text_from_txt(file_bytes, filename)
        root = ET.fromstring(raw)
        parts: List[str] = []

        def walk(node, path=""):
            tag = node.tag.split("}")[-1] if isinstance(node.tag, str) else str(node.tag)
            cur = f"{path}/{tag}" if path else tag
            if node.text and node.text.strip():
                parts.append(f"{cur}: {node.text.strip()}")
            for child in list(node):
                walk(child, cur)
            if node.tail and node.tail.strip():
                parts.append(node.tail.strip())

        walk(root)
        return "\n".join(parts)
    except Exception as e:
        print(f"  ❌ Ошибка XML {filename}: {e}")
        # fallback: как текст
        return extract_text_from_txt(file_bytes, filename)


def extract_text_from_json(file_bytes: bytes, filename: str = "") -> str:
    try:
        raw = extract_text_from_txt(file_bytes, filename)
        data = json.loads(raw)
        return json.dumps(data, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"  ❌ Ошибка JSON {filename}: {e}")
        return extract_text_from_txt(file_bytes, filename)


def extract_text_from_xlsx(file_bytes: bytes, filename: str = "") -> str:
    parts: List[str] = []
    try:
        wb = load_workbook(io.BytesIO(file_bytes), data_only=True, read_only=True)
        for sheet in wb.worksheets:
            parts.append(f"[Лист: {sheet.title}]")
            for row in sheet.iter_rows(values_only=True):
                cells = [str(c).strip() for c in row if c is not None and str(c).strip()]
                if cells:
                    parts.append(" | ".join(cells))
        wb.close()
    except Exception as e:
        print(f"  ❌ Ошибка XLSX {filename}: {e}")
    return "\n".join(parts)


def extract_text_from_xls(file_bytes: bytes, filename: str = "") -> str:
    if xlrd is None:
        print(f"  ❌ xlrd не установлен — пропуск {filename}")
        return ""
    parts: List[str] = []
    try:
        book = xlrd.open_workbook(file_contents=file_bytes)
        for sheet in book.sheets():
            parts.append(f"[Лист: {sheet.name}]")
            for r in range(sheet.nrows):
                cells = []
                for c in range(sheet.ncols):
                    val = sheet.cell_value(r, c)
                    if val is None or val == "":
                        continue
                    cells.append(str(val).strip())
                if cells:
                    parts.append(" | ".join(cells))
    except Exception as e:
        print(f"  ❌ Ошибка XLS {filename}: {e}")
    return "\n".join(parts)


def extract_text_from_csv(file_bytes: bytes, filename: str = "") -> str:
    try:
        raw = extract_text_from_txt(file_bytes, filename)
        # автоопределение разделителя
        sample = raw[:4096]
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=";,\t|")
        except Exception:
            dialect = csv.excel
            dialect.delimiter = ";" if sample.count(";") > sample.count(",") else ","
        reader = csv.reader(io.StringIO(raw), dialect)
        rows = []
        for row in reader:
            cells = [c.strip() for c in row if c and c.strip()]
            if cells:
                rows.append(" | ".join(cells))
        return "\n".join(rows)
    except Exception as e:
        print(f"  ❌ Ошибка CSV {filename}: {e}")
        return extract_text_from_txt(file_bytes, filename)


def extract_text_from_ods(file_bytes: bytes, filename: str = "") -> str:
    if odf_load is None:
        print(f"  ❌ odfpy не установлен — пропуск {filename}")
        return ""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".ods", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        doc = odf_load(tmp_path)
        parts: List[str] = []
        for table in doc.getElementsByType(odf_table.Table):
            name = table.getAttribute("name") or "Sheet"
            parts.append(f"[Лист: {name}]")
            for row in table.getElementsByType(odf_table.TableRow):
                cells = []
                for cell in row.getElementsByType(odf_table.TableCell):
                    ct = odf_extract_text(cell).strip()
                    if ct:
                        cells.append(ct)
                if cells:
                    parts.append(" | ".join(cells))
        return "\n".join(parts)
    except Exception as e:
        print(f"  ❌ Ошибка ODS {filename}: {e}")
        return ""
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def extract_text_from_image(file_bytes: bytes, filename: str = "") -> str:
    """Полный OCR для любого изображения (скан письма, фото, штамп)."""
    try:
        img = Image.open(io.BytesIO(file_bytes))
        # многостраничный TIFF
        texts = []
        try:
            n = getattr(img, "n_frames", 1)
        except Exception:
            n = 1
        for frame in range(max(n, 1)):
            try:
                if n > 1:
                    img.seek(frame)
                frame_img = img.copy()
            except Exception:
                frame_img = img
            t = ocr_pil_image(frame_img, filename if n == 1 else f"{filename}#f{frame+1}")
            if t:
                if n > 1:
                    texts.append(f"[Кадр {frame + 1}]\n{t}")
                else:
                    texts.append(t)
        return "\n\n".join(texts).strip()
    except Exception as e:
        print(f"  ❌ Ошибка OCR изображения {filename}: {e}")
        return ""


def ocr_pdf_bytes(file_bytes: bytes, filename: str = "") -> str:
    """Полный OCR PDF через pdf2image @ OCR_DPI + Tesseract (страница целиком). Один проход."""
    parts: List[str] = []
    try:
        images = convert_from_bytes(file_bytes, dpi=OCR_DPI)
        total = len(images)
        print(f"  🔍 Полный OCR PDF ({total} стр. @ {OCR_DPI} dpi): {filename}", flush=True)
        for i, img in enumerate(images, start=1):
            print(f"  🔍 OCR: страница {i} из {total} — {filename}", flush=True)
            text = ocr_pil_image(img, f"{filename}#p{i}")
            if text:
                parts.append(f"[Страница {i} | OCR]\n{text}")
            else:
                print(f"  ⚠️ OCR пуст на стр. {i}: {filename}")
    except Exception as e:
        print(f"  ❌ OCR PDF {filename}: {e}")
    return "\n\n".join(parts)


def extract_text_from_eml(file_bytes: bytes, filename: str = "") -> str:
    """Разбор .eml: заголовки, тело, вложения (в т.ч. сканы)."""
    if email is None:
        return extract_text_fallback(file_bytes, filename)
    parts: List[str] = []
    try:
        msg = email.message_from_bytes(file_bytes, policy=email_policy.default)
        headers = []
        for h in ("From", "To", "Cc", "Subject", "Date", "Message-ID"):
            val = msg.get(h)
            if val:
                headers.append(f"{h}: {val}")
        if headers:
            parts.append("[Заголовки письма]\n" + "\n".join(headers))

        body_texts = []
        attachments = []
        if msg.is_multipart():
            for part in msg.walk():
                ctype = part.get_content_type()
                disp = str(part.get_content_disposition() or "")
                name = part.get_filename() or ""
                payload = part.get_payload(decode=True)
                if payload is None:
                    continue
                if disp == "attachment" or name:
                    attachments.append((name or f"attachment-{ctype}", payload))
                elif ctype == "text/plain":
                    body_texts.append(payload.decode(part.get_content_charset() or "utf-8", errors="replace"))
                elif ctype == "text/html":
                    body_texts.append(extract_text_from_html(payload, name or "body.html"))
                elif ctype.startswith("image/"):
                    attachments.append((name or f"inline.{ctype.split('/')[-1]}", payload))
        else:
            payload = msg.get_payload(decode=True) or b""
            ctype = msg.get_content_type()
            if ctype == "text/html":
                body_texts.append(extract_text_from_html(payload, filename))
            else:
                body_texts.append(payload.decode(msg.get_content_charset() or "utf-8", errors="replace"))

        body = "\n".join(t.strip() for t in body_texts if t and t.strip())
        if body:
            parts.append("[Тело письма]\n" + body)

        for att_name, data in attachments:
            print(f"  📎 EML вложение: {att_name}")
            nested = extract_text(data, f"{filename}/{att_name}")
            if nested.strip():
                parts.append(f"[Вложение: {att_name}]\n{nested}")
            else:
                # принудительный OCR, если похоже на картинку
                ocr = extract_text_from_image(data, att_name)
                if ocr:
                    parts.append(f"[Вложение OCR: {att_name}]\n{ocr}")
    except Exception as e:
        print(f"  ❌ Ошибка EML {filename}: {e}")
    return "\n\n".join(parts)


def extract_text_from_msg(file_bytes: bytes, filename: str = "") -> str:
    """Разбор Outlook .msg: тема, тело, вложения (сканы писем/согласований)."""
    if extract_msg is None:
        print(f"  ❌ extract-msg не установлен — пропуск {filename}")
        return ""
    tmp_path = None
    parts: List[str] = []
    try:
        with tempfile.NamedTemporaryFile(suffix=".msg", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        msg = extract_msg.Message(tmp_path)
        headers = []
        for label, attr in (
            ("From", "sender"),
            ("To", "to"),
            ("Cc", "cc"),
            ("Subject", "subject"),
            ("Date", "date"),
        ):
            val = getattr(msg, attr, None)
            if val:
                headers.append(f"{label}: {val}")
        if headers:
            parts.append("[Заголовки письма]\n" + "\n".join(headers))
        body = (getattr(msg, "body", None) or "").strip()
        if body:
            parts.append("[Тело письма]\n" + body)
        # HTML-тело если body пуст
        if not body:
            html = getattr(msg, "htmlBody", None) or getattr(msg, "htmlBodyMsg", None)
            if html:
                if isinstance(html, bytes):
                    parts.append("[Тело HTML]\n" + extract_text_from_html(html, filename))
                else:
                    parts.append("[Тело HTML]\n" + extract_text_from_html(str(html).encode("utf-8"), filename))

        for att in getattr(msg, "attachments", []) or []:
            try:
                att_name = getattr(att, "longFilename", None) or getattr(att, "shortFilename", None) or "attachment"
                data = getattr(att, "data", None)
                if not data:
                    continue
                print(f"  📎 MSG вложение: {att_name}")
                nested = extract_text(data, f"{filename}/{att_name}")
                if nested.strip():
                    parts.append(f"[Вложение: {att_name}]\n{nested}")
                else:
                    ocr = extract_text_from_image(data, att_name)
                    if ocr:
                        parts.append(f"[Вложение OCR: {att_name}]\n{ocr}")
            except Exception as e:
                print(f"  ⚠️ MSG attachment error: {e}")
        try:
            msg.close()
        except Exception:
            pass
    except Exception as e:
        print(f"  ❌ Ошибка MSG {filename}: {e}")
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    return "\n\n".join(parts)


def _looks_like_image(file_bytes: bytes) -> bool:
    if len(file_bytes) < 8:
        return False
    magic = [
        b"\xff\xd8\xff",            # jpeg
        b"\x89PNG\r\n\x1a\n",       # png
        b"GIF87a", b"GIF89a",       # gif
        b"BM",                      # bmp
        b"II*\x00", b"MM\x00*",     # tiff
        b"RIFF",                    # webp (need WEBP later)
    ]
    return any(file_bytes.startswith(m) for m in magic)


def extract_text_fallback(file_bytes: bytes, filename: str = "") -> str:
    """Fallback: картинка → OCR; иначе попытка как текст."""
    # неизвестный формат, но это изображение — полный OCR
    if _looks_like_image(file_bytes):
        print(f"  🖼️ Неизвестное расширение, но файл-изображение — OCR: {filename}")
        return extract_text_from_image(file_bytes, filename)

    print(f"  ⚠️ Неизвестный формат {filename} — пробую как текст")
    text = extract_text_from_txt(file_bytes, filename).strip()
    if not text:
        # последняя попытка — OCR (вдруг скан без расширения)
        ocr = extract_text_from_image(file_bytes, filename)
        if ocr:
            print(f"  🔍 Fallback OCR сработал: {filename}")
            return ocr
        return ""
    printable = sum(1 for ch in text if ch.isprintable() or ch in "\n\r\t")
    if printable / max(len(text), 1) < 0.7:
        print(f"  🖼️ Похоже на бинарный файл — пробую OCR: {filename}")
        ocr = extract_text_from_image(file_bytes, filename)
        if ocr:
            return ocr
        print(f"  ⏭️ OCR не дал текста, пропуск: {filename}")
        return ""
    return text


def extract_text(file_bytes: bytes, filename: str) -> str:
    """Универсальное извлечение текста по расширению файла."""
    ext = file_ext(filename)

    if ext == ".pdf":
        return extract_text_from_pdf(file_bytes, filename)
    if ext == ".docx":
        return extract_text_from_docx(file_bytes, filename)
    if ext == ".doc":
        return extract_text_from_doc(file_bytes, filename)
    if ext == ".rtf":
        return extract_text_from_rtf(file_bytes, filename)
    if ext == ".odt":
        return extract_text_from_odt(file_bytes, filename)
    if ext in {".txt", ".log", ".md"}:
        return extract_text_from_txt(file_bytes, filename)
    if ext in {".html", ".htm"}:
        return extract_text_from_html(file_bytes, filename)
    if ext == ".xml":
        return extract_text_from_xml(file_bytes, filename)
    if ext == ".json":
        return extract_text_from_json(file_bytes, filename)
    if ext == ".xlsx":
        return extract_text_from_xlsx(file_bytes, filename)
    if ext == ".xls":
        return extract_text_from_xls(file_bytes, filename)
    if ext == ".csv":
        return extract_text_from_csv(file_bytes, filename)
    if ext == ".ods":
        return extract_text_from_ods(file_bytes, filename)
    if ext in {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp", ".jfif"}:
        return extract_text_from_image(file_bytes, filename)
    if ext in {".eml"}:
        return extract_text_from_eml(file_bytes, filename)
    if ext in {".msg"}:
        return extract_text_from_msg(file_bytes, filename)
    if ext in {".dwg", ".dxf"}:
        return extract_text_from_dwg(file_bytes, filename)
    if ext == ".pptx":
        return extract_text_from_pptx(file_bytes, filename)
    if ext == ".ppt":
        return extract_text_from_ppt(file_bytes, filename)

    # fallback: текст или OCR
    return extract_text_fallback(file_bytes, filename)


def extract_text_from_pptx(file_bytes: bytes, filename: str = "") -> str:
    parts: List[str] = []
    try:
        prs = Presentation(io.BytesIO(file_bytes))
        for i, slide in enumerate(prs.slides, start=1):
            slide_parts: List[str] = []
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text and shape.text.strip():
                    slide_parts.append(shape.text.strip())
                if shape.has_table:
                    table = shape.table
                    for row in table.rows:
                        cells = [c.text.strip() for c in row.cells if c.text and c.text.strip()]
                        if cells:
                            slide_parts.append(" | ".join(cells))
            if slide_parts:
                parts.append(f"[Слайд {i}]\n" + "\n".join(slide_parts))
    except Exception as e:
        print(f"  ❌ Ошибка PPTX {filename}: {e}")
    return "\n\n".join(parts)


def extract_text_from_ppt(file_bytes: bytes, filename: str = "") -> str:
    """
    Старый .ppt: python-pptx не читает бинарный PPT.
    Пробуем catppt (если есть) или fallback как текст.
    """
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".ppt", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        result = subprocess.run(
            ["catppt", tmp_path],
            capture_output=True,
            check=False,
        )
        raw = result.stdout or b""
        if raw:
            for encoding in ("utf-8", "cp1251", "latin-1"):
                try:
                    return raw.decode(encoding).strip()
                except UnicodeDecodeError:
                    continue
            return raw.decode("latin-1", errors="replace").strip()
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"  ⚠️ catppt недоступен для {filename}: {e}")
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    # python-pptx иногда открывает только pptx — пробуем на всякий случай
    text = extract_text_from_pptx(file_bytes, filename)
    if text.strip():
        return text
    print(f"  ⚠️ Старый .ppt может читаться неполно: {filename}")
    return extract_text_from_txt(file_bytes, filename)


def _dxf_collect_text(doc) -> List[str]:
    parts: List[str] = []
    try:
        msp = doc.modelspace()
    except Exception:
        return parts

    for entity in msp:
        try:
            dxftype = entity.dxftype()
            if dxftype == "TEXT":
                t = (entity.dxf.text or "").strip()
                if t:
                    parts.append(t)
            elif dxftype == "MTEXT":
                t = (entity.text or entity.plain_text() if hasattr(entity, "plain_text") else "").strip()
                if not t and hasattr(entity, "plain_text"):
                    t = entity.plain_text().strip()
                if t:
                    parts.append(t)
            elif dxftype == "ATTRIB":
                t = (entity.dxf.text or "").strip()
                if t:
                    parts.append(t)
            elif dxftype == "ATTDEF":
                t = (entity.dxf.text or "").strip()
                if t:
                    parts.append(t)
            elif dxftype == "INSERT":
                for attrib in getattr(entity, "attribs", []):
                    t = (attrib.dxf.text or "").strip()
                    if t:
                        parts.append(t)
        except Exception:
            continue
    return parts


def extract_text_from_dwg(file_bytes: bytes, filename: str = "") -> str:
    """Извлечение текста из DWG/DXF через ezdxf (DWG — best-effort)."""
    if ezdxf is None:
        print(f"  ❌ ezdxf не установлен — пропуск {filename}")
        return ""

    ext = file_ext(filename) or ".dxf"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=ext if ext in {".dwg", ".dxf"} else ".dxf", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        doc = None
        try:
            doc = ezdxf.readfile(tmp_path)
        except Exception:
            if ezdxf_recover is not None:
                try:
                    doc, auditor = ezdxf_recover.readfile(tmp_path)
                except Exception as e:
                    print(f"  ❌ Не удалось открыть DWG/DXF {filename}: {e}")
                    print("     (для нативных .dwg часто нужен ODA File Converter)")
                    return ""
            else:
                return ""

        parts = _dxf_collect_text(doc)
        return "\n".join(parts)
    except Exception as e:
        print(f"  ❌ Ошибка DWG/DXF {filename}: {e}")
        return ""
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# =============================================================================
# Распаковка архивов (рекурсивно)
# =============================================================================

def _iter_archive_members_zip(file_bytes: bytes) -> List[Tuple[str, bytes]]:
    members: List[Tuple[str, bytes]] = []
    with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
        for name in zf.namelist():
            if name.endswith("/") or "__MACOSX" in name or Path(name).name.startswith("."):
                continue
            try:
                members.append((name, zf.read(name)))
            except Exception as e:
                print(f"  ⚠️ Не удалось прочитать из ZIP: {name}: {e}")
    return members


def _iter_archive_members_rar(file_bytes: bytes) -> List[Tuple[str, bytes]]:
    members: List[Tuple[str, bytes]] = []
    with tempfile.NamedTemporaryFile(suffix=".rar", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        with rarfile.RarFile(tmp_path) as rf:
            for info in rf.infolist():
                name = info.filename
                if info.is_dir() or "__MACOSX" in name or Path(name).name.startswith("."):
                    continue
                try:
                    members.append((name, rf.read(info)))
                except Exception as e:
                    print(f"  ⚠️ Не удалось прочитать из RAR: {name}: {e}")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
    return members


def _iter_archive_members_7z(file_bytes: bytes) -> List[Tuple[str, bytes]]:
    members: List[Tuple[str, bytes]] = []
    with tempfile.NamedTemporaryFile(suffix=".7z", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        with py7zr.SevenZipFile(tmp_path, mode="r") as archive:
            with tempfile.TemporaryDirectory() as extract_dir:
                archive.extractall(path=extract_dir)
                root = Path(extract_dir)
                for path in root.rglob("*"):
                    if not path.is_file():
                        continue
                    rel = str(path.relative_to(root))
                    if "__MACOSX" in rel or path.name.startswith("."):
                        continue
                    members.append((rel, path.read_bytes()))
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
    return members


def _iter_archive_members_tar(file_bytes: bytes) -> List[Tuple[str, bytes]]:
    members: List[Tuple[str, bytes]] = []
    with tarfile.open(fileobj=io.BytesIO(file_bytes), mode="r:*") as tf:
        for info in tf.getmembers():
            if not info.isfile():
                continue
            name = info.name
            if "__MACOSX" in name or Path(name).name.startswith("."):
                continue
            f = tf.extractfile(info)
            if f is None:
                continue
            members.append((name, f.read()))
    return members


def _iter_gzip_member(file_bytes: bytes, filename: str) -> List[Tuple[str, bytes]]:
    """Одиночный .gz (не tar.gz): распаковать и вернуть внутренний файл."""
    try:
        data = gzip.decompress(file_bytes)
    except Exception as e:
        print(f"  ❌ Ошибка gzip {filename}: {e}")
        return []
    inner_name = Path(filename).name
    if inner_name.lower().endswith(".gz"):
        inner_name = inner_name[:-3]
    if not inner_name:
        inner_name = "uncompressed.bin"
    return [(inner_name, data)]


def unpack_archive(file_bytes: bytes, filename: str) -> List[Tuple[str, bytes]]:
    """Распаковывает ZIP / RAR / 7Z / TAR / GZ и возвращает список (имя, байты)."""
    ext = file_ext(filename)
    try:
        if ext == ".zip":
            return _iter_archive_members_zip(file_bytes)
        if ext == ".rar":
            return _iter_archive_members_rar(file_bytes)
        if ext == ".7z":
            return _iter_archive_members_7z(file_bytes)
        if ext in {".tar", ".tar.gz", ".tgz"}:
            return _iter_archive_members_tar(file_bytes)
        if ext == ".gz":
            # .tar.gz уже обработан выше; чистое .gz
            return _iter_gzip_member(file_bytes, filename)
    except Exception as e:
        print(f"  ❌ Ошибка распаковки {filename}: {e}")
    return []


def extract_documents_from_bytes(file_bytes: bytes, filename: str) -> List[Tuple[str, str]]:
    """
    Извлекает документы из файла или архива (рекурсивно).
    Возвращает список (источник, текст).
    """
    ext = file_ext(filename)
    results: List[Tuple[str, str]] = []

    if ext in SUPPORTED_ARCHIVES:
        for inner_name, data in unpack_archive(file_bytes, filename):
            nested = extract_documents_from_bytes(data, f"{filename}/{inner_name}")
            results.extend(nested)
        return results

    # известный документ / неизвестный → extract_text (с fallback)
    text = extract_text(file_bytes, filename)
    if text.strip():
        results.append((filename, text))
    else:
        if ext and ext not in SUPPORTED_DOCS:
            print(f"  ⚠️ Не удалось извлечь текст из {filename}")
        else:
            print(f"  ⚠️ Пустой текст: {filename}")
    return results


# =============================================================================
# RAGIndex (TF-IDF)
# =============================================================================

def split_into_chunks(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    result: List[str] = []
    start = 0
    step = max(1, chunk_size - overlap)
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            result.append(chunk)
        if end >= len(text):
            break
        start += step
    return result


class RAGIndex:
    """Индекс документов на базе TF-IDF + cosine similarity."""

    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks: List[str] = []
        self.sources: List[str] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.matrix = None

    def build(self, documents: List[Tuple[str, str]]) -> int:
        """Строит индекс по списку (source, text). Возвращает число чанков."""
        self.chunks = []
        self.sources = []

        for source, text in documents:
            for part in split_into_chunks(text, self.chunk_size, self.chunk_overlap):
                self.chunks.append(part)
                self.sources.append(source)

        if not self.chunks:
            self.vectorizer = None
            self.matrix = None
            print("⚠️ Нет текста для индексации.")
            return 0

        self.vectorizer = TfidfVectorizer(
            max_features=50000,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self.matrix = self.vectorizer.fit_transform(self.chunks)
        print(
            f"✅ Индекс построен: {len(self.chunks)} чанков "
            f"из {len({s for s in self.sources})} источников."
        )
        return len(self.chunks)

    def search(self, query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
        """Ищет top_k релевантных фрагментов. Возвращает [{text, source, score}, ...]."""
        if not self.chunks or self.vectorizer is None or self.matrix is None:
            return []

        q = self.vectorizer.transform([query])
        scores = cosine_similarity(q, self.matrix).ravel()
        if scores.size == 0:
            return []

        k = min(top_k, len(scores))
        idx = np.argsort(scores)[::-1][:k]

        results: List[Dict[str, Any]] = []
        for i in idx:
            score = float(scores[i])
            if score <= 0:
                continue
            results.append(
                {
                    "text": self.chunks[i],
                    "source": self.sources[i],
                    "score": score,
                }
            )
        return results

    def clear(self) -> None:
        self.chunks = []
        self.sources = []
        self.vectorizer = None
        self.matrix = None


# Глобальное хранилище сессии:
#   uploaded_files — модуль 2 положит сюда {имя: bytes}
#   rag_index      — модуль 3 построит индекс после чтения
uploaded_files = {}
rag_index = RAGIndex()


# =============================================================================
# DeepSeek
# =============================================================================

def ask_deepseek(
    question: str,
    context: str = "",
    api_key: Optional[str] = None,
    model: str = DEEPSEEK_MODEL,
    temperature: float = 0.2,
    timeout: Optional[float] = None,
) -> str:
    """
    Отправляет вопрос в DeepSeek Chat Completions API.
    Если передан context — отвечает строго по контексту (RAG).
    """
    key = (api_key or os.environ.get("DEEPSEEK_API_KEY") or DEEPSEEK_API_KEY or "").strip()
    if not key:
        return "❌ API-ключ DeepSeek не задан. Перезапустите модуль 1 и введите ключ."

    if context.strip():
        system_prompt = (
            "Ты помощник по анализу документов. Отвечай только на основе "
            "предоставленного контекста. Если ответа в контексте нет — так и скажи. "
            "Отвечай на языке вопроса пользователя."
        )
        user_content = f"Контекст:\n{context}\n\nВопрос: {question}"
    else:
        system_prompt = "Ты полезный помощник. Отвечай кратко и по делу."
        user_content = question

    req_timeout = DEEPSEEK_TIMEOUT if timeout is None else timeout

    try:
        resp = requests.post(
            DEEPSEEK_API_URL,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                "temperature": temperature,
            },
            timeout=req_timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except requests.HTTPError as e:
        detail = ""
        try:
            detail = resp.text[:500]
        except Exception:
            pass
        return f"❌ Ошибка DeepSeek API ({e}): {detail}"
    except Exception as e:
        return f"❌ Ошибка запроса к DeepSeek: {e}"


class ProgressBar:
    """
    Простой прогресс-бар с ETA:
    [████████░░░░] 67% (3 мин. осталось)
    """

    def __init__(self, total: int, label: str = "", width: int = 12):
        self.total = max(1, int(total))
        self.label = label
        self.width = max(4, int(width))
        self.done = 0
        self.t0 = time.time()

    @staticmethod
    def _fmt_eta(seconds: float) -> str:
        if seconds < 0 or seconds != seconds:  # NaN
            return "—"
        seconds = int(round(seconds))
        if seconds < 60:
            return f"{seconds} сек."
        minutes = seconds // 60
        if minutes < 60:
            rem = seconds % 60
            return f"{minutes} мин." if rem < 15 else f"{minutes} мин. {rem} сек."
        hours = minutes // 60
        minutes = minutes % 60
        return f"{hours} ч. {minutes} мин."

    def render(self, done: Optional[int] = None, suffix: str = "") -> str:
        if done is not None:
            self.done = done
        n = min(self.done, self.total)
        pct = 100.0 * n / self.total
        filled = int(round(self.width * n / self.total))
        filled = min(self.width, max(0, filled))
        bar = "█" * filled + "░" * (self.width - filled)
        elapsed = time.time() - self.t0
        if n > 0:
            eta = elapsed * (self.total - n) / n
            eta_s = self._fmt_eta(eta)
        else:
            eta_s = "оценка…"
        label = f"{self.label} " if self.label else ""
        extra = f" {suffix}" if suffix else ""
        return f"{label}[{bar}] {pct:.0f}% ({eta_s} осталось){extra}"

    def tick(self, step: int = 1, suffix: str = "") -> str:
        self.done = min(self.total, self.done + step)
        line = self.render(suffix=suffix)
        print(line, flush=True)
        return line

    def finish(self, suffix: str = "готово") -> str:
        self.done = self.total
        elapsed = time.time() - self.t0
        label = f"{self.label} " if self.label else ""
        bar = "█" * self.width
        extra = f" {suffix}" if suffix else ""
        line = f"{label}[{bar}] 100% (заняло {self._fmt_eta(elapsed)}){extra}"
        print(line, flush=True)
        return line


def ask_with_rag(question: str, top_k: int = TOP_K) -> Dict[str, Any]:
    """Удобная обёртка: поиск по индексу + ответ DeepSeek."""
    hits = rag_index.search(question, top_k=top_k)
    if not hits:
        return {
            "answer": (
                "❌ Индекс пуст или ничего не найдено. "
                "Сначала выполните модуль 2–3 (Загрузка и индексация)."
            ),
            "sources": [],
        }
    context = "\n\n---\n\n".join(
        f"[Источник: {h['source']} | score={h['score']:.3f}]\n{h['text']}" for h in hits
    )
    answer = ask_deepseek(question, context=context)
    return {"answer": answer, "sources": hits}


print(
    "📎 Форматы: docx/doc/rtf/odt/pdf(+OCR)/txt/eml/msg/xlsx/xls/csv/ods/"
    "jpg/png/zip/rar/7z + др."
)
print(
    f"🔍 OCR dpi={OCR_DPI}, lang={OCR_LANG}, psm=6; "
    f"смешанный контент: OCR при изображениях/графике + цифровой текст без дублей; "
    f"chunk={CHUNK_SIZE}, TOP_K={TOP_K}, timeout={DEEPSEEK_TIMEOUT}с"
)
if DEEPSEEK_API_KEY:
    print("🔐 API-ключ DeepSeek сохранён.")
else:
    print("⚠️ API-ключ пустой — ответы DeepSeek будут недоступны.")


_mark("deps_end")
print(f"⏱️ Этап 1: {_fmt_dur(_elapsed('deps_start', 'deps_end'))}")

# =============================================================================
# ЭТАП 2/5 — Загрузка файлов
# =============================================================================
_mark("upload_start")
_status("этап 2/5 — загрузка файлов (files.upload)")
print("\n" + "=" * 70)
print("ЭТАП 2/5 — Загрузка файлов")
print("=" * 70)

try:
    from google.colab import files
except ImportError as e:
    raise RuntimeError(
        "❌ google.colab недоступен. Запускайте этот код в Google Colab."
    ) from e

print("📂 Выберите файлы для загрузки (можно несколько, включая архивы)...")

# Результат files.upload() — словарь {имя: bytes}; сохраняем сразу в uploaded_files
uploaded_files = files.upload()

if not uploaded_files:
    print("⚠️ Файлы не загружены. Запустите ячейку ещё раз и выберите файлы.")
else:
    print(f"\n📋 Загружено файлов: {len(uploaded_files)}\n")
    print(f"{'№':<4} {'Имя файла':<60} {'Размер'}")
    print("-" * 80)
    for i, (name, data) in enumerate(uploaded_files.items(), start=1):
        size = len(data)
        if size >= 1024 * 1024:
            size_str = f"{size / (1024 * 1024):.2f} MB ({size:,} байт)"
        elif size >= 1024:
            size_str = f"{size / 1024:.2f} KB ({size:,} байт)"
        else:
            size_str = f"{size} байт"
        display_name = name if len(name) <= 58 else name[:55] + "..."
        print(f"{i:<4} {display_name:<60} {size_str}")

print("\n✅ Данные загружены.")

_mark("upload_end")
print(f"⏱️ Этап 2: {_fmt_dur(_elapsed('upload_start', 'upload_end'))}")

if not uploaded_files:
    raise RuntimeError("❌ Файлы не загружены. Перезапустите ячейку и выберите файлы.")

# =============================================================================
# ЭТАП 3/5 — Извлечение текста, OCR, индексация
# =============================================================================
_mark("index_start")
_status("этап 3/5 — OCR / извлечение текста / индексация")
print("\n" + "=" * 70)
print("ЭТАП 3/5 — Извлечение текста, OCR, индексация")
print("=" * 70)

import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Set

# =============================================================================
# 1) OCR + размер чанков + TOP_K (переопределяем настройки модуля 1)
# =============================================================================

OCR_DPI = 200
OCR_PDF_FORCE_FULL_IF_AVG_BELOW = 15
OCR_MIN_CHARS_PER_PAGE = 30          # мало текста → OCR
OCR_MIN_ALPHA_RATIO = 0.25
OCR_IMAGE_AREA_RATIO = 0.08          # любая заметная картинка → OCR
OCR_OVERLAP_SKIP = 0.75              # порог дубля OCR↔цифровой текст

CHUNK_SIZE = 600                     # было 800
CHUNK_OVERLAP = 100
TOP_K = 5                            # единый режим: TOP_K=5


def page_needs_ocr(page, digital_text: str) -> bool:
    """
    OCR при смешанном контенте: мало текста ИЛИ есть изображения/графика.
    Цифровой слой сохраняется; OCR дополняет через ocr_supplement_text.
    """
    text = (digital_text or "").strip()
    chars = len(re.sub(r"\s+", "", text))
    if chars < OCR_MIN_CHARS_PER_PAGE:
        return True
    try:
        if page.get_images(full=True):
            return True
    except Exception:
        pass
    try:
        if page.get_image_info(xrefs=True):
            return True
    except Exception:
        pass
    try:
        if len(page.get_drawings() or []) >= 5:
            return True
    except Exception:
        pass
    try:
        rect = page.rect
        page_area = abs(rect.width * rect.height) or 1.0
        img_area = 0.0
        for info in page.get_image_info(xrefs=True) or []:
            bbox = info.get("bbox")
            if not bbox:
                continue
            x0, y0, x1, y1 = bbox
            img_area += abs((x1 - x0) * (y1 - y0))
        if (img_area / page_area) >= OCR_IMAGE_AREA_RATIO:
            return True
    except Exception:
        pass
    return False


# Пересоздаём индекс с новым размером чанка
rag_index = RAGIndex(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

print(
    f"⚙️ OCR: dpi={OCR_DPI}, lang=rus+eng, psm=6; "
    f"запуск при изображениях/графике или тексте < {OCR_MIN_CHARS_PER_PAGE} симв/стр "
    f"(цифровой текст + OCR-дополнение без дублей)"
)
print(f"⚙️ Индекс: chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}, TOP_K={TOP_K}")


# =============================================================================
# Утилиты
# =============================================================================

_PRINT_LOCK = threading.Lock()


def _log(*args, **kwargs):
    with _PRINT_LOCK:
        print(*args, **kwargs)


def _progress_line(done: int, total: int, t0: float, label: str = "") -> str:
    total = max(1, total)
    pct = 100.0 * done / total
    width = 12
    filled = min(width, max(0, int(round(width * done / total))))
    bar = "█" * filled + "░" * (width - filled)
    elapsed = time.time() - t0
    if done > 0:
        eta = elapsed * (total - done) / done
        if eta < 60:
            eta_s = f"{int(round(eta))} сек."
        else:
            eta_s = f"{int(round(eta / 60))} мин."
    else:
        eta_s = "оценка…"
    prefix = f"{label} " if label else ""
    return f"{prefix}[{bar}] {pct:.0f}% ({eta_s} осталось)"


def _fmt_size(n: int) -> str:
    if n >= 1024 * 1024:
        return f"{n / (1024 * 1024):.2f} MB ({n:,} байт)"
    if n >= 1024:
        return f"{n / 1024:.2f} KB ({n:,} байт)"
    return f"{n} байт"


def _safe_ext(filename: str) -> str:
    if "file_ext" in globals():
        normalized = str(filename).replace("\\", "/")
        ext = file_ext(normalized)
        if ext:
            return ext
    name = str(filename).replace("\\", "/").split("/")[-1].lower().strip()
    if name.endswith(".tar.gz"):
        return ".tar.gz"
    if name.endswith(".tar.bz2"):
        return ".tar.bz2"
    if "." not in name:
        return ""
    return "." + name.rsplit(".", 1)[-1]


def _basename(filename: str) -> str:
    return str(filename).replace("\\", "/").split("/")[-1]


def _canon_basename(filename: str) -> str:
    """file (2).docx → file.docx (для дедупа копий Colab)."""
    name = _basename(filename)
    name = re.sub(r"\s*\(\d+\)(?=\.\w+$)", "", name)
    return name.lower().strip()


def _md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


# =============================================================================
# 2) Дедупликация загруженных файлов
# =============================================================================

def dedupe_uploaded(files: Dict[str, bytes]) -> Dict[str, bytes]:
    """
    Убирает дубликаты по MD5 содержимого и по каноническому имени.
    Печатает: ⏭️ Пропуск дубликата: ...
    """
    unique: Dict[str, bytes] = {}
    seen_hash: Set[str] = set()
    seen_name: Set[str] = set()
    skipped = 0

    # стабильный порядок
    for name, data in files.items():
        h = _md5(data)
        cname = _canon_basename(name)

        if h in seen_hash:
            _log(f"⏭️ Пропуск дубликата: {name} (уже обработан, тот же хэш)")
            skipped += 1
            continue
        if cname in seen_name:
            _log(f"⏭️ Пропуск дубликата: {name} (уже обработан как «{cname}»)")
            skipped += 1
            continue

        seen_hash.add(h)
        seen_name.add(cname)
        unique[name] = data

    _log(f"🧹 Дедупликация: было {len(files)} → уникальных {len(unique)} (пропущено {skipped})")
    return unique


# Глобальные множества для дедупа внутри архивов (на весь прогон модуля 3)
_SEEN_INNER_HASHES: Set[str] = set()
_SEEN_INNER_NAMES: Set[str] = set()
_DEDUP_LOCK = threading.Lock()


def _register_or_skip_inner(name: str, data: bytes) -> bool:
    """
    Дедуп ТОЛЬКО для файлов внутри архивов.
    True = это дубликат, пропустить.
    False = новый файл, зарегистрирован.
    Не использовать для загрузок верхнего уровня (их чистит dedupe_uploaded).
    """
    h = _md5(data)
    cname = _canon_basename(name)
    with _DEDUP_LOCK:
        if h in _SEEN_INNER_HASHES:
            return True
        if cname in _SEEN_INNER_NAMES:
            return True
        _SEEN_INNER_HASHES.add(h)
        _SEEN_INNER_NAMES.add(cname)
        return False


# =============================================================================
# Извлечение текста
# =============================================================================

def _read_document_bytes(file_bytes: bytes, filename: str) -> str:
    ext = _safe_ext(filename)
    clean_name = _basename(filename) or f"document{ext}"

    if ext == ".doc":
        text = ""
        if "extract_text_from_doc" in globals():
            try:
                text = extract_text_from_doc(file_bytes, clean_name) or ""
            except Exception as e:
                _log(f"  ⚠️ extract_text_from_doc({clean_name}): {e}")
        if not (text or "").strip() and "extract_text" in globals():
            try:
                text = extract_text(file_bytes, clean_name) or ""
            except Exception as e:
                _log(f"  ⚠️ extract_text({clean_name}): {e}")
        return (text or "").strip()

    if "extract_text" in globals():
        try:
            return (extract_text(file_bytes, clean_name) or "").strip()
        except Exception as e:
            _log(f"  ⚠️ extract_text({clean_name}): {e}")
            return ""
    return ""


def _status_label(ext: str, ok: bool) -> str:
    if ok:
        if ext == ".doc":
            return "прочитан (.doc / antiword)"
        if "SUPPORTED_DOCS" in globals() and ext in SUPPORTED_DOCS:
            return "прочитан"
        return "прочитан (fallback)"
    if ext == ".doc":
        return "пропущен (.doc: antiword не извлёк текст)"
    return "пропущен (пустой текст / не удалось прочитать)"


def _extract_from_any(file_bytes: bytes, filename: str) -> List[Tuple[str, str]]:
    """Рекурсивно извлекает документы; дубликаты внутри архивов пропускает."""
    ext = _safe_ext(filename)
    results: List[Tuple[str, str]] = []

    if ext in SUPPORTED_ARCHIVES:
        try:
            members = unpack_archive(file_bytes, filename)
        except Exception as e:
            _log(f"  ❌ Ошибка распаковки {filename}: {e}")
            return results
        for inner_name, data in members:
            inner_norm = str(inner_name).replace("\\", "/")
            full = f"{filename}/{inner_norm}"
            if _register_or_skip_inner(inner_norm, data):
                _log(f"  ⏭️ Пропуск дубликата: {inner_norm} (уже обработан)")
                continue
            results.extend(_extract_from_any(data, full))
        return results

    # обычный файл (уже прошёл дедуп на уровне архива, либо это вложенный вызов)
    text = _read_document_bytes(file_bytes, filename)
    if text:
        results.append((filename, text))
    return results


def _process_uploaded_file(filename: str, file_bytes: bytes) -> List[Tuple[str, str]]:
    """Читает один загруженный файл/архив. Сначала листинг архива, потом OCR."""
    ext = _safe_ext(filename)
    size = len(file_bytes)
    extracted: List[Tuple[str, str]] = []

    lines = [
        "=" * 80,
        f"📄 Файл: {filename}",
        f"   Размер: {_fmt_size(size)}",
        f"   Тип: {'архив ' + ext if ext in SUPPORTED_ARCHIVES else ext or '(без расширения)'}",
    ]

    if ext in SUPPORTED_ARCHIVES:
        lines.append("   Содержимое архива:")
        try:
            members = unpack_archive(file_bytes, filename)
        except Exception as e:
            lines.append(f"   ❌ Не удалось открыть архив: {e}")
            members = []

        if not members:
            lines.append("   ⚠️ Архив пуст или не удалось прочитать.")
            _log("\n".join(lines))
            return extracted

        # 1) Полный листинг БЕЗ OCR (чтобы OCR не шёл «во время распаковки»)
        work_items: List[Tuple[str, bytes, str]] = []
        for inner_name, data in members:
            inner_norm = str(inner_name).replace("\\", "/")
            inner_ext = _safe_ext(inner_norm)
            prefix = f"   • {inner_norm} [{_fmt_size(len(data))}]"
            source = f"{filename}/{inner_norm}"

            if _register_or_skip_inner(inner_norm, data):
                lines.append(f"{prefix} → ⏭️ Пропуск дубликата: {inner_norm} (уже обработан)")
                continue

            if inner_ext in SUPPORTED_ARCHIVES:
                lines.append(f"{prefix} → вложенный архив (после листинга)")
            else:
                lines.append(f"{prefix} → в очереди на чтение/OCR")
            work_items.append((inner_norm, data, source))

        lines.append(f"   Распаковка завершена: к обработке {len(work_items)} файл(ов)")
        _log("\n".join(lines))

        # 2) Только после листинга — извлечение текста / OCR
        for inner_norm, data, source in work_items:
            inner_ext = _safe_ext(inner_norm)
            if inner_ext in SUPPORTED_ARCHIVES:
                nested = _extract_from_any(data, source)
                extracted.extend(nested)
                if nested:
                    chars = sum(len(t) for _, t in nested)
                    _log(
                        f"   ✅ {inner_norm}: вложенный архив → {len(nested)} док., {chars:,} символов"
                    )
                else:
                    _log(f"   ⚠️ {inner_norm}: вложенный архив пуст / только дубли")
                continue

            text = _read_document_bytes(data, inner_norm)
            if text:
                extracted.append((source, text))
                _log(
                    f"   ✅ {inner_norm}: {_status_label(inner_ext, True)} ({len(text):,} символов)"
                )
            else:
                _log(f"   ⚠️ {inner_norm}: {_status_label(inner_ext, False)}")

        docs_count = len(extracted)
        chars_total = sum(len(t) for _, t in extracted)
        _log(f"   Итого по архиву «{filename}»: документов={docs_count}, символов={chars_total:,}")
        return extracted

    # Файл верхнего уровня: уже уникален после dedupe_uploaded (MD5 + basename).
    text = _read_document_bytes(file_bytes, filename)
    if text:
        extracted = [(filename, text)]
    elif ext == ".pdf":
        # PDF уже прошёл extract_text_from_pdf — не вызываем повторно через fallback
        extracted = []
    else:
        try:
            extracted = extract_documents_from_bytes(file_bytes, filename) or []
        except Exception:
            extracted = []

    docs_count = len(extracted)
    chars_total = sum(len(t) for _, t in extracted)
    lines.append(f"   Статус: {_status_label(ext, docs_count > 0)}")
    lines.append(f"   Документов извлечено: {docs_count}")
    lines.append(f"   Символов: {chars_total:,}")
    _log("\n".join(lines))
    return extracted


# =============================================================================
# 3) Параллельный запуск
# =============================================================================

_doc_ok = ".doc" in SUPPORTED_DOCS if "SUPPORTED_DOCS" in globals() else False
_antiword_fn = "extract_text_from_doc" in globals()

print(f"🔍 Чтение и индексация")
print(f"   .doc в SUPPORTED_DOCS: {'да' if _doc_ok else 'нет — перезапустите модуль 1'}")
print(f"   extract_text_from_doc: {'да' if _antiword_fn else 'нет — перезапустите модуль 1'}")
print()

# дедуп верхнего уровня — ТОЛЬКО по MD5 / каноническому basename (реальные дубли)
unique_files = dedupe_uploaded(dict(uploaded_files))

# множества внутренних дублей — ТОЛЬКО для содержимого архивов (пусто на старте)
_SEEN_INNER_HASHES.clear()
_SEEN_INNER_NAMES.clear()
# кэш PDF с прошлого запуска модуля 5 в той же сессии — сбрасываем
if "_PDF_TEXT_CACHE" in globals() and isinstance(_PDF_TEXT_CACHE, dict):
    _PDF_TEXT_CACHE.clear()
if "_PDF_OCR_INFLIGHT" in globals() and isinstance(_PDF_OCR_INFLIGHT, dict):
    _PDF_OCR_INFLIGHT.clear()

MAX_WORKERS = min(4, max(1, len(unique_files)))
print(f"🚀 Параллельная обработка: {len(unique_files)} файлов, workers={MAX_WORKERS}\n")

documents: List[Tuple[str, str]] = []
errors = []
_t0 = time.time()
_done = 0
_total_files = max(1, len(unique_files))

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
    futures = {
        pool.submit(_process_uploaded_file, name, data): name
        for name, data in unique_files.items()
    }
    for fut in as_completed(futures):
        name = futures[fut]
        try:
            docs = fut.result()
            documents.extend(docs)
        except Exception as e:
            errors.append((name, str(e)))
            _log(f"❌ Ошибка обработки {name}: {e}")
        _done += 1
        _log(_progress_line(_done, _total_files, _t0, label="Индексация"))

# дополнительная дедупликация документов по (basename, hash текста)
_final_docs: List[Tuple[str, str]] = []
_seen_doc: Set[str] = set()
dup_docs = 0
for source, text in documents:
    key = _canon_basename(source) + "::" + hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()
    if key in _seen_doc:
        dup_docs += 1
        continue
    _seen_doc.add(key)
    _final_docs.append((source, text))
documents = _final_docs
if dup_docs:
    print(f"\n🧹 Убрано дублирующих документов после извлечения: {dup_docs}")

print("\n" + "=" * 80)
print("📚 Сборка индекса TF-IDF...")

if not documents:
    rag_index.clear()
    print("⚠️ Не извлечено ни одного документа — индекс пуст.")
else:
    n_chunks = rag_index.build(documents)

    n_sources = len(set(rag_index.sources))
    vocab_size = 0
    if rag_index.vectorizer is not None:
        try:
            vocab_size = len(rag_index.vectorizer.vocabulary_)
        except Exception:
            vocab_size = 0

    doc_sources = [s for s in set(rag_index.sources) if _safe_ext(s) == ".doc"]
    print("\n📊 Итоговая статистика")
    print("-" * 40)
    print(f"   Загружено (модуль 2):        {len(uploaded_files)}")
    print(f"   Уникальных после дедупа:     {len(unique_files)}")
    print(f"   Извлечено документов:        {len(documents)}")
    print(f"   Источников в индексе:        {n_sources}")
    print(f"   из них .doc:                 {len(doc_sources)}")
    print(f"   Чанков:                      {n_chunks}")
    print(f"   chunk_size / overlap:         {CHUNK_SIZE} / {CHUNK_OVERLAP}")
    print(f"   TOP_K (сессия):               {TOP_K}")
    print(f"   Размер словаря TF-IDF:       {vocab_size:,}")
    print(f"   Всего символов текста:       {sum(len(t) for _, t in documents):,}")
    print(f"   Время извлечения:            {time.time() - _t0:.1f} сек.")
    if errors:
        print(f"   Ошибок обработки:            {len(errors)}")
    if doc_sources:
        print("   .doc источники:")
        for s in sorted(doc_sources):
            print(f"      • {s}")

print("\n✅ Индекс построен.")

_mark("index_end")
print(f"⏱️ Этап 3: {_fmt_dur(_elapsed('index_start', 'index_end'))}")

# =============================================================================
# ЭТАП 4/5 — Анализ 86 вопросов + классификация + согласования
# =============================================================================
_mark("analysis_start")
_status("этап 4/5 — анализ тендера (86 вопросов)")
print("\n" + "=" * 70)
print("ЭТАП 4/5 — Анализ тендера (86 вопросов)")
print("=" * 70)

# colab_files уже импортирован как files — алиас для скачивания
colab_files = files

if not getattr(rag_index, "chunks", None):
    raise RuntimeError("❌ Индекс пуст — не удалось извлечь текст из загруженных файлов.")

from collections import defaultdict
from datetime import datetime

ANALYSIS_DATE = datetime.now()
TOP_K = 5                            # было 12 — меньше контекста, быстрее ответ
TOP_K_FORCED = 8
MAX_CTX = 12000
DEEPSEEK_TIMEOUT = 60                # было 120 сек.

# =============================================================================
# Ожидаемые 14 файлов
# =============================================================================

EXPECTED_FILES = [
    "Извещение.docx",
    "Закупочная документация.docx",
    "Приложение № 1 - Техническое задание.doc",
    "Приложение № 3 - График производства работ.doc",
    "Приложение № 4 - График освоения и финансирования денежных средств.docx",
    "Приложение № 5 - Акт окончания работ.docx",
    "Проект Договора СМР-ПНР по САУГПТ и ЕСУМИС.docx",
    "Техническое задание.pdf",
    "ВОР Раздел ПД №12 ЛСР (02-01-01) САУГПТ.xlsx",
    "ВОР Раздел ПД №12 ЛСР (02-01-02) ЕСУМИС.xlsx",
    "Приложение № 5 к ТЗ - ЛСР САУГПТ.xlsx",
    "Приложение № 6 к ТЗ - ЛСР ЕСУМИС.xlsx",
    "14-27-00987 от 03.02.2026 РД САУГПТ.pdf",
    "14-27-03537 от 20.04.2026 РД ЕСУМИС.pdf",
]

MASK_TECH = ("рд", "14-27-", "техническое задание", "техзадани", "/тз", "тз.", "рабочая документация", "саугпт", "есумис")
MASK_ESTIMATE = ("лср", "вор")
MASK_CONTRACT = ("договор",)
MASK_NOTICE = ("извещение", "закупочная")
MASK_SCHEDULE = ("график",)
MASK_FINANCE_SCHED = ("освоения", "финансирования")

# =============================================================================
# Дедуп индекса
# =============================================================================

def _basename(source: str) -> str:
    name = str(source).replace("\\", "/").split("/")[-1]
    name = re.sub(r"\s*\(\d+\)(?=\.\w+$)", "", name)
    return name


def _norm(name: str) -> str:
    return re.sub(r"\s+", " ", _basename(name).lower().strip())


def _hash(text: str) -> str:
    return hashlib.md5(re.sub(r"\s+", " ", (text or "").strip().lower()).encode("utf-8", errors="ignore")).hexdigest()


def _match_expected(actual: str, expected: str) -> bool:
    a, e = _norm(actual), _norm(expected)
    if a == e or e in a or a in e:
        return True
    m = re.search(r"14-27-\d+", e)
    if m and m.group(0) in a:
        return True
    pairs = [
        ("график производства", "график производства"),
        ("график освоения", "график освоения"),
        ("акт окончания", "акт окончания"),
        ("закупочная документация", "закупочная документация"),
        ("извещение", "извещение"),
    ]
    for pe, pa in pairs:
        if pe in e and pa in a:
            return True
    if "лср" in e and "лср" in a:
        if ("саугпт" in e and "саугпт" in a) or ("есумис" in e and "есумис" in a):
            return True
    if "вор" in e and "вор" in a:
        if ("саугпт" in e and "саугпт" in a) or ("есумис" in e and "есумис" in a):
            return True
    if "проект договора" in e and "договор" in a and "смр" in a:
        return True
    if "приложение № 1" in e and "техническое задание" in a and a.endswith(".doc"):
        return True
    if e.startswith("техническое задание") and a.startswith("техническое задание") and a.endswith(".pdf"):
        return True
    return False


def build_dedup():
    items = list(enumerate(zip(rag_index.chunks, rag_index.sources)))

    def rank(src: str):
        s = src.lower()
        pen = (2 if "процедуре" in s else 0) + (1 if s.count(".zip/") + s.count(".rar/") > 1 else 0)
        return (pen, len(s))

    items.sort(key=lambda it: rank(it[1][1]))
    seen: Set[str] = set()
    chunks, sources, orig = [], [], []
    for i, (t, s) in items:
        h = _hash(t)
        if h in seen:
            continue
        seen.add(h)
        chunks.append(t)
        sources.append(s)
        orig.append(i)
    return chunks, sources, orig


print("🔧 Дедупликация индекса...")
D_CHUNKS, D_SOURCES, D_ORIG = build_dedup()
print(f"   Чанков: {len(rag_index.chunks)} → {len(D_CHUNKS)}")
print(f"   Уник. basename: {len({_norm(s) for s in D_SOURCES})}")

_VECT = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True)
_MAT = _VECT.fit_transform(D_CHUNKS) if D_CHUNKS else None

# группы по basename
FILE_GROUPS: Dict[str, Dict[str, Any]] = {}
for i, (t, s) in enumerate(zip(D_CHUNKS, D_SOURCES)):
    key = _norm(s)
    g = FILE_GROUPS.setdefault(key, {"display": _basename(s), "paths": set(), "idxs": [], "chars": 0})
    g["paths"].add(s)
    g["idxs"].append(i)
    g["chars"] += len(t)


def map_expected() -> List[Tuple[str, Optional[str]]]:
    used: Set[str] = set()
    out: List[Tuple[str, Optional[str]]] = []
    for exp in EXPECTED_FILES:
        found = None
        for key, g in FILE_GROUPS.items():
            if key in used:
                continue
            if _match_expected(key, exp) or _match_expected(g["display"], exp):
                found = key
                break
        if found:
            used.add(found)
        out.append((exp, found))
    for key in FILE_GROUPS:
        if key not in used:
            out.append((FILE_GROUPS[key]["display"], key))
    return out


FILE_PLAN = map_expected()


# =============================================================================
# Классификация документов + реквизиты писем/согласований
# =============================================================================

DOC_TYPE_LETTER = "Письмо-согласование"
DOC_TYPE_RD = "Рабочая документация"
DOC_TYPE_ESTIMATE = "Смета"
DOC_TYPE_TZ = "Техническое задание"
DOC_TYPE_CONTRACT = "Договор / приложение к договору"
DOC_TYPE_NOTICE = "Извещение / закупочная документация"
DOC_TYPE_SCHEDULE = "График"
DOC_TYPE_ACT = "Акт"
DOC_TYPE_OTHER = "Прочий документ"
DOC_TYPE_SCAN = "Скан (тип не определён)"

# Явные маркеры из ТЗ + расширения для OCR/типовых формулировок
LETTER_KW = (
    "письмо", "согласовани", "таможн", "таможен", "фтс", "обращение",
    "уведомлени", "разрешени", "заключаем", "не возражаем", "рассмотрев",
)
RD_KW = (
    "рабочая документация", "шифр", "альбом рд", "том рд",
    "чертеж", "спецификац", "ведомость рабочих чертежей",
)
EST_KW = (
    "смета", "сметн", "лср", "вор", "локальн",
    "единичн расцен", "итого по смете",
)
TZ_KW = ("техническое задание", "предмет закупки", "требования к выполнению")
CONTRACT_KW = ("договор", "подрядчик", "заказчик обязуется", "неустойк", "гарантийный срок")
NOTICE_KW = ("извещение", "запрос предложений", "закупочная документация", "нмцк")
SCHEDULE_KW = ("график производства", "график освоения", "этап работ")
ACT_KW = ("акт окончания", "акт сдачи", "приёмк")

# Ключевые слова ТЗ — достаточно одного явного маркера письма
LETTER_CORE = ("письмо", "согласовани", "таможн", "таможен", "фтс", "обращение")


def _file_sample_text(group_key: str, max_chars: int = 12000) -> str:
    """Собрать текст файла из чанков (начало + середина) для классификации."""
    g = FILE_GROUPS.get(group_key)
    if not g:
        return ""
    idxs = g["idxs"]
    parts = []
    total = 0
    # первые чанки + равномерно ещё несколько
    pick = list(idxs[:4])
    if len(idxs) > 8:
        step = max(1, len(idxs) // 6)
        pick.extend(idxs[4::step][:6])
    elif len(idxs) > 4:
        pick.extend(idxs[4:8])
    seen = set()
    for i in pick:
        if i in seen:
            continue
        seen.add(i)
        t = D_CHUNKS[i]
        if total + len(t) > max_chars and parts:
            break
        parts.append(t)
        total += len(t)
    return "\n".join(parts)


def _score_keywords(text_low: str, keywords: Sequence[str]) -> int:
    return sum(1 for kw in keywords if kw in text_low)


def _has_rd_token(text_low: str) -> bool:
    """«РД» как отдельный токен (не часть другого слова)."""
    return bool(re.search(r"(?<![a-zа-я0-9])рд(?![a-zа-я0-9])", text_low))


def classify_document(display_name: str, text: str) -> Tuple[str, List[str]]:
    """
    Классификация по СОДЕРЖИМОМУ (приоритетнее имени файла).
    Возвращает (тип, список сработавших признаков).

    Правила ТЗ:
      - письмо / согласование / таможня / ФТС / обращение → Письмо-согласование
      - рабочая документация / РД / шифр → Рабочая документация
      - смета / ЛСР / ВОР → Смета
    """
    name_low = _norm(display_name)
    text_low = (text or "").lower()
    blob = name_low + "\n" + text_low
    hits: List[str] = []

    letter_score = _score_keywords(text_low, LETTER_KW)
    # Контент важнее имени: «14-27-… РД …» часто письмо таможни, а не альбом РД
    if letter_score >= 1 and any(k in text_low for k in LETTER_CORE):
        for kw in LETTER_KW:
            if kw in text_low:
                hits.append(kw)
        return DOC_TYPE_LETTER, hits[:8]

    rd_score = _score_keywords(blob, RD_KW)
    if _has_rd_token(text_low):
        rd_score += 1
    est_score = _score_keywords(blob, EST_KW)
    tz_score = _score_keywords(blob, TZ_KW)
    contract_score = _score_keywords(blob, CONTRACT_KW)
    notice_score = _score_keywords(blob, NOTICE_KW)
    sched_score = _score_keywords(blob, SCHEDULE_KW)
    act_score = _score_keywords(blob, ACT_KW)

    # эвристики по имени (без 14-27-… — это часто исходящий № письма)
    if "лср" in name_low or "вор" in name_low or "смет" in name_low:
        est_score += 3
    if "рабочая документация" in name_low or re.search(r"(?:^|[^a-zа-я0-9])рд(?:[^a-zа-я0-9]|$)", name_low):
        # только если нет явных маркеров письма в тексте
        if not any(k in text_low for k in LETTER_CORE):
            rd_score += 2
    if "техническое задание" in name_low or (name_low.endswith(".doc") and "приложение № 1" in name_low):
        tz_score += 2
    if "договор" in name_low:
        contract_score += 2
    if "извещение" in name_low or "закупочная" in name_low:
        notice_score += 3
    if "график" in name_low:
        sched_score += 3
    if "акт" in name_low:
        act_score += 3

    ranked = [
        (est_score, DOC_TYPE_ESTIMATE, EST_KW),
        (rd_score, DOC_TYPE_RD, RD_KW),
        (tz_score, DOC_TYPE_TZ, TZ_KW),
        (contract_score, DOC_TYPE_CONTRACT, CONTRACT_KW),
        (notice_score, DOC_TYPE_NOTICE, NOTICE_KW),
        (sched_score, DOC_TYPE_SCHEDULE, SCHEDULE_KW),
        (act_score, DOC_TYPE_ACT, ACT_KW),
        (letter_score, DOC_TYPE_LETTER, LETTER_KW),
    ]
    ranked.sort(key=lambda x: x[0], reverse=True)
    best_score, best_type, best_kws = ranked[0]
    if best_score <= 0:
        # OCR-текст есть, но тип неясен
        if len((text or "").strip()) > 40:
            return DOC_TYPE_SCAN, ["текст есть, ключевых маркеров нет"]
        return DOC_TYPE_OTHER, ["пусто / мало текста"]

    for kw in best_kws:
        if kw in blob:
            hits.append(kw)
    if best_type == DOC_TYPE_RD and _has_rd_token(text_low) and "рд" not in hits:
        hits.append("рд")
    return best_type, hits[:8]


LETTER_EXTRACT_PROMPT = """Это текст документа. Определи, является ли он письмом / согласованием / разрешением.
Если да — извлеки реквизиты СТРОГО в формате:

Тип: Письмо-согласование
Дата письма: ...
Номер письма: ...
Отправитель: ...
Получатель: ...
Суть согласования: ...
Результат (согласовано/отказано/с замечаниями): ...

Если какого-то поля нет — напиши «Не указано».
Если это НЕ письмо/согласование — первой строкой напиши: Тип: не письмо
и кратко что это за документ.
"""


def extract_letter_requisites(display_name: str, text: str) -> Dict[str, str]:
    """Извлечение реквизитов письма через DeepSeek (+ regex-подсказки)."""
    result = {
        "Тип": DOC_TYPE_LETTER,
        "Дата письма": "Не указано",
        "Номер письма": "Не указано",
        "Отправитель": "Не указано",
        "Получатель": "Не указано",
        "Суть согласования": "Не указано",
        "Результат": "Не указано",
        "Файл": display_name,
    }
    # regex-подсказки из имени файла: «14-27-03537 от 20.04.2026 …»
    m_num = re.search(r"(14-27-\d+)", display_name)
    m_date = re.search(r"от\s+(\d{2}[.\-]\d{2}[.\-]\d{4})", display_name, re.I)
    if m_num:
        result["Номер письма"] = m_num.group(1)
    if m_date:
        result["Дата письма"] = m_date.group(1).replace("-", ".")

    if not (text or "").strip():
        return result

    try:
        raw = ask_deepseek(LETTER_EXTRACT_PROMPT, context=text[:14000], timeout=DEEPSEEK_TIMEOUT)
    except Exception as e:
        result["Суть согласования"] = f"Ошибка извлечения: {e}"
        return result

    raw = (raw or "").strip()
    if re.search(r"тип:\s*не письмо", raw, re.I):
        result["Тип"] = "не письмо"
        result["Суть согласования"] = raw
        return result

    def _field(patterns: Sequence[str]) -> Optional[str]:
        for pat in patterns:
            m = re.search(pat, raw, flags=re.I | re.M)
            if m:
                val = m.group(1).strip().strip(" .;")
                if val and val.lower() not in ("не указано", "-", "нет"):
                    return val
        return None

    date = _field([r"Дата письма:\s*(.+)", r"Дата:\s*(.+)"])
    number = _field([r"Номер письма:\s*(.+)", r"№\s*([^\n]+)", r"Исх\.?\s*№?\s*([^\n]+)"])
    sender = _field([r"Отправитель:\s*(.+)", r"От кого:\s*(.+)"])
    receiver = _field([r"Получатель:\s*(.+)", r"Кому:\s*(.+)"])
    essence = _field([r"Суть согласования:\s*(.+)", r"Суть:\s*(.+)"])
    outcome = _field([r"Результат[^:]*:\s*(.+)", r"Решение:\s*(.+)"])

    if date:
        result["Дата письма"] = date
    if number:
        result["Номер письма"] = number
    if sender:
        result["Отправитель"] = sender
    if receiver:
        result["Получатель"] = receiver
    if essence:
        result["Суть согласования"] = essence
    if outcome:
        result["Результат"] = outcome

    # доп. regex по самому тексту, если LLM не нашёл
    if result["Дата письма"] == "Не указано":
        m = re.search(r"\b(\d{2}[.\-/]\d{2}[.\-/]\d{4})\b", text[:2000])
        if m:
            result["Дата письма"] = m.group(1).replace("-", ".").replace("/", ".")
    if result["Номер письма"] == "Не указано":
        m = re.search(r"(?:исх\.?\s*№?|№)\s*([A-Za-zА-Яа-я0-9\-_/]+)", text[:2000], re.I)
        if m:
            result["Номер письма"] = m.group(1)

    return result


print("🏷️ Классификация документов по содержимому...")
_status("классификация документов и извлечение реквизитов писем")
FILE_CLASSIFICATION: Dict[str, Dict[str, Any]] = {}  # group_key -> meta
APPROVALS: List[Dict[str, str]] = []

for exp, key in FILE_PLAN:
    if key is None:
        FILE_CLASSIFICATION[exp] = {
            "key": None,
            "display": exp,
            "doc_type": "Не найден в индексе",
            "signals": [],
            "sample_chars": 0,
        }
        continue
    display = FILE_GROUPS[key]["display"]
    sample = _file_sample_text(key)
    doc_type, signals = classify_document(display, sample)
    meta = {
        "key": key,
        "display": display,
        "label": exp,
        "doc_type": doc_type,
        "signals": signals,
        "sample_chars": len(sample),
        "sample": sample,
    }
    FILE_CLASSIFICATION[key] = meta
    print(f"   • {display}: {doc_type} [{', '.join(signals[:4]) or '—'}]")

    # письма / согласования — извлекаем реквизиты
    if doc_type == DOC_TYPE_LETTER or (
        letter_score := _score_keywords(sample.lower(), LETTER_KW)
    ) >= 2:
        # даже если имя «РД …», при сильных признаках письма — извлекаем
        if doc_type != DOC_TYPE_LETTER and letter_score >= 2:
            doc_type = DOC_TYPE_LETTER
            meta["doc_type"] = DOC_TYPE_LETTER
            print(f"     ↳ переклассифицирован в «{DOC_TYPE_LETTER}» по содержимому")
        print(f"     ✉️ Извлечение реквизитов письма...")
        req = extract_letter_requisites(display, sample)
        meta["letter"] = req
        if req.get("Тип") != "не письмо":
            APPROVALS.append(req)

# дедуп согласований по номер+дата+файл
_seen_appr = set()
_uniq_appr: List[Dict[str, str]] = []
for a in APPROVALS:
    key = (a.get("Номер письма"), a.get("Дата письма"), a.get("Файл"))
    if key in _seen_appr:
        continue
    _seen_appr.add(key)
    _uniq_appr.append(a)
APPROVALS = _uniq_appr
print(f"   Найдено писем/согласований: {len(APPROVALS)}")


# =============================================================================
# Поиск
# =============================================================================

def _mask_hit(source: str, masks: Sequence[str]) -> bool:
    s = source.lower().replace("\\", "/")
    b = _norm(source)
    return any(m.lower() in s or m.lower() in b for m in masks)


def search(
    query: str,
    top_k: int = TOP_K,
    masks: Optional[Sequence[str]] = None,
    mask_only: bool = False,
) -> List[Dict[str, Any]]:
    if _MAT is None:
        return []
    q = _VECT.transform([query])
    scores = cosine_similarity(q, _MAT).ravel()
    out = []
    for i, sc in enumerate(scores):
        src = D_SOURCES[i]
        matched = _mask_hit(src, masks) if masks else False
        if mask_only and masks and not matched:
            continue
        boost = 0.5 if matched else 0.0
        if not masks:
            for kw, b in (("извещение", 0.3), ("закупочная", 0.28), ("договор", 0.28),
                          ("лср", 0.4), ("вор", 0.4), ("график", 0.35), ("рд", 0.25),
                          ("техническое задание", 0.3)):
                if kw in src.lower():
                    boost = max(boost, b)
        sc = float(sc) + boost
        if sc <= 0:
            continue
        out.append({
            "text": D_CHUNKS[i],
            "source": src,
            "score": sc,
            "chunk": D_ORIG[i] + 1,
            "base": _basename(src),
        })
    out.sort(key=lambda x: x["score"], reverse=True)
    return out[:top_k]


def merge_hits(lists: List[List[Dict[str, Any]]], top_k: int = TOP_K) -> List[Dict[str, Any]]:
    best: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for hits in lists:
        for h in hits:
            key = (h["source"], h["text"][:160])
            if key not in best or h["score"] > best[key]["score"]:
                best[key] = h
    return sorted(best.values(), key=lambda x: x["score"], reverse=True)[:top_k]


def fmt_ctx(hits: List[Dict[str, Any]], max_chars: int = MAX_CTX) -> str:
    parts, n = [], 0
    for i, h in enumerate(hits, 1):
        block = f"[Фрагмент {i} | {h['source']} | чанк #{h.get('chunk')} | {h['score']:.3f}]\n{h['text']}"
        if n + len(block) > max_chars and parts:
            break
        parts.append(block)
        n += len(block)
    return "\n\n---\n\n".join(parts)


def _empty(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    low = t.lower()
    marks = ["не указано", "не найдено", "нет информации", "в контексте нет",
             "информация отсутствует", "не удалось", "ответить невозможно"]
    if len(t) < 200 and any(m in low for m in marks):
        if re.search(r"\d{3,}", t) and ("руб" in low or "%" in t or "инн" in low):
            return False
        return True
    return False


def clean(text: str) -> str:
    text = (text or "").strip()
    return "Не указано" if _empty(text) else text


PRIORITY_PROMPT = (
    "Отвечай только по контексту. Выпиши конкретные факты: числа, даты, суммы, перечни, нормы. "
    "Не пиши «не указано», если в контексте есть хотя бы частичный ответ. "
    "Если данные противоречивы — укажи оба варианта и источники. Ответ на русском."
)


def ask_rag(
    question: str,
    variants: Optional[List[str]] = None,
    masks: Optional[Sequence[str]] = None,
    mask_only: bool = False,
    top_k: int = TOP_K,
) -> Tuple[str, List[Dict[str, Any]]]:
    variants = variants or []
    # Для скорости: основной вопрос + максимум 1 вариант (не все)
    queries = [question] + (variants[:1] if variants else [])
    lists = []
    for q in queries:
        lists.append(search(q, top_k=top_k, masks=masks, mask_only=False))
        if masks:
            lists.append(search(q, top_k=TOP_K_FORCED, masks=masks, mask_only=True))
    hits = merge_hits(lists, top_k=top_k)
    if not hits:
        return "Не указано", []
    ans = clean(ask_deepseek(
        f"{PRIORITY_PROMPT}\n\nВопрос: {question}",
        context=fmt_ctx(hits),
        timeout=DEEPSEEK_TIMEOUT,
    ))
    if ans == "Не указано" and variants:
        rq = variants[-1] + " Приведи любые найденные факты из контекста."
        rh = merge_hits(
            [search(rq, top_k=top_k, masks=masks, mask_only=bool(masks)), hits],
            top_k=top_k,
        )
        if rh:
            ans2 = clean(ask_deepseek(
                f"{PRIORITY_PROMPT}\n\nВопрос: {rq}",
                context=fmt_ctx(rh),
                timeout=DEEPSEEK_TIMEOUT,
            ))
            if ans2 != "Не указано":
                return ans2, rh
    return ans, hits


# =============================================================================
# 86 вопросов: (id, section, title, question, variants, masks, mask_only, kind)
# kind: rag | meta | synthesize
# =============================================================================

# Вопросы 67-78 и 85-86 — synthesize на основе предыдущих ответов
# 79-84 — meta по статистике файлов

QUESTIONS: List[Dict[str, Any]] = []

def Q(num, section, title, question, variants=None, masks=None, mask_only=False, kind="rag"):
    QUESTIONS.append({
        "num": num,
        "section": section,
        "title": title,
        "question": question,
        "variants": variants or [],
        "masks": masks,
        "mask_only": mask_only,
        "kind": kind,
    })

# --- 1. Общая информация ---
Q(1, "1. ОБЩАЯ ИНФОРМАЦИЯ", "Номер тендера / ИКЗ",
  "Каков полный номер тендера / извещения / процедуры / ИКЗ? Укажи номер целиком (включая все цифры после B).",
  ["Номер процедуры вида B… из Извещения.", "ИКЗ или номер закупки."], MASK_NOTICE)
Q(2, "1. ОБЩАЯ ИНФОРМАЦИЯ", "Полное название объекта закупки",
  "Каково полное название / наименование объекта закупки или предмета лота?",
  ["Наименование закупки / лота из Извещения."], MASK_NOTICE)
Q(3, "1. ОБЩАЯ ИНФОРМАЦИЯ", "Краткое описание проекта (аннотация)",
  "Кратко опиши проект / закупку (аннотация 3–6 предложений): что делается и зачем.",
  ["Суть работ САУГПТ и ЕСУМИС."], MASK_NOTICE + MASK_TECH)
Q(4, "1. ОБЩАЯ ИНФОРМАЦИЯ", "Заказчик (название)",
  "Кто заказчик? Полное наименование организации.",
  ["Наименование заказчика."], MASK_NOTICE)
Q(5, "1. ОБЩАЯ ИНФОРМАЦИЯ", "ИНН заказчика",
  "Какой ИНН заказчика?",
  ["ИНН организации-заказчика."], MASK_NOTICE)
Q(6, "1. ОБЩАЯ ИНФОРМАЦИЯ", "Контакты заказчика (телефон, email, адрес)",
  "Контакты заказчика: телефоны, email, адрес, контактные лица.",
  ["Телефон и email заказчика, ФИО контактов."], MASK_NOTICE)
Q(7, "1. ОБЩАЯ ИНФОРМАЦИЯ", "Адрес объекта",
  "Какой адрес объекта?",
  ["Адрес площадки / порта / сооружения."], MASK_NOTICE + MASK_TECH)
Q(8, "1. ОБЩАЯ ИНФОРМАЦИЯ", "Место выполнения работ",
  "Каково место выполнения работ?",
  ["Место производства работ."], MASK_NOTICE + MASK_TECH)
Q(9, "1. ОБЩАЯ ИНФОРМАЦИЯ", "Объект (здание/помещение)",
  "В каком здании / помещении выполняются работы (пункт пропуска и т.п.)?",
  ["Наименование здания / сооружения объекта."], MASK_TECH + MASK_NOTICE)
Q(10, "1. ОБЩАЯ ИНФОРМАЦИЯ", "Источник финансирования",
  "Какой источник финансирования?",
  ["Собственные средства заказчика или иной источник."], MASK_NOTICE)
Q(11, "1. ОБЩАЯ ИНФОРМАЦИЯ", "Вид процедуры (запрос предложений, аукцион, конкурс)",
  "Какой вид закупочной процедуры?",
  ["Открытый запрос предложений / аукцион / конкурс."], MASK_NOTICE)

# --- 2. Требования к участникам ---
Q(12, "2. ТРЕБОВАНИЯ К УЧАСТНИКАМ", "Требуемая лицензия (СРО, МЧС)",
  "Какие лицензии и допуски нужны (СРО, МЧС)?",
  ["Лицензия МЧС и членство в СРО."], MASK_NOTICE)
Q(13, "2. ТРЕБОВАНИЯ К УЧАСТНИКАМ", "Требования к опыту работы",
  "Какие требования к опыту работы / аналогичным договорам?",
  ["Справка об аналогичных договорах."], MASK_NOTICE)
Q(14, "2. ТРЕБОВАНИЯ К УЧАСТНИКАМ", "Минимальный срок работы на рынке",
  "Какой минимальный срок работы на рынке / опыт в годах?",
  ["Релевантный опыт не менее N лет."], MASK_NOTICE)
Q(15, "2. ТРЕБОВАНИЯ К УЧАСТНИКАМ", "Требования к персоналу",
  "Какие требования к персоналу и квалификации?",
  ["Квалификация специалистов, ИТР."], MASK_NOTICE)
Q(16, "2. ТРЕБОВАНИЯ К УЧАСТНИКАМ", "Требования к финансовым ресурсам",
  "Какие требования к финансовым ресурсам участника?",
  ["Финансовая устойчивость, обороты, справки банка."], MASK_NOTICE)
Q(17, "2. ТРЕБОВАНИЯ К УЧАСТНИКАМ", "Требования к материально-техническим ресурсам",
  "Какие требования к материально-техническим ресурсам?",
  ["Справка о МТР, оборудование участника."], MASK_NOTICE)
Q(18, "2. ТРЕБОВАНИЯ К УЧАСТНИКАМ", "Требования к деловой репутации",
  "Какие требования к деловой репутации / РНП / отказам от договоров?",
  ["Реестр недобросовестных поставщиков, отказы от заключения."], MASK_NOTICE)
Q(19, "2. ТРЕБОВАНИЯ К УЧАСТНИКАМ", "Перечень документов для заявки",
  "Какой перечень документов заявки (формы №…)?",
  ["Формы №1–№5 и иные документы предложения."], MASK_NOTICE)
Q(20, "2. ТРЕБОВАНИЯ К УЧАСТНИКАМ", "Размер обеспечения заявки",
  "Какой размер обеспечения заявки?",
  ["Обеспечение предложения: сумма или %."], MASK_NOTICE)
Q(21, "2. ТРЕБОВАНИЯ К УЧАСТНИКАМ", "Запрет/ограничения на участие",
  "Какие запреты и ограничения на участие в закупке?",
  ["Ограничения для участников, аффилированность, санкции."], MASK_NOTICE)

# --- 3. Техника (приоритет РД/ТЗ) ---
Q(22, "3. ТЕХНИЧЕСКИЕ ТРЕБОВАНИЯ", "Тип системы пожаротушения",
  "Какой тип системы пожаротушения (САУГПТ и др.)? Упомяни ЕСУМИС.",
  ["Газовое пожаротушение САУГПТ."], MASK_TECH, False)
Q(23, "3. ТЕХНИЧЕСКИЕ ТРЕБОВАНИЯ", "Огнетушащее вещество (ГОТВ)",
  "Какое ГОТВ / огнетушащее вещество (тип, масса заправки)? Ищи в ТЗ, ЛСР, РД.",
  ["Тип газа ГОТВ и масса заправки модуля."], MASK_TECH + MASK_ESTIMATE, False)
Q(24, "3. ТЕХНИЧЕСКИЕ ТРЕБОВАНИЯ", "Модули пожаротушения (перечень)",
  "Перечень модулей пожаротушения: марки, типы, количество (из ЛСР/ВОР/ТЗ/РД).",
  ["Модули МГП / газового ПТ с количеством."], MASK_TECH + MASK_ESTIMATE, False)
Q(25, "3. ТЕХНИЧЕСКИЕ ТРЕБОВАНИЯ", "Оборудование в системе (полный перечень)",
  "Полный перечень оборудования САУГПТ и ЕСУМИС с количествами из ЛСР/ВОР/ТЗ/РД.",
  ["Спецификация оборудования по смете и РД."], MASK_TECH + MASK_ESTIMATE, False)
Q(26, "3. ТЕХНИЧЕСКИЕ ТРЕБОВАНИЯ", "Технические характеристики оборудования",
  "Какие технические характеристики оборудования указаны (напряжение, интерфейсы, параметры)?",
  ["Характеристики приборов и модулей из ТЗ/РД."], MASK_TECH, False)
Q(27, "3. ТЕХНИЧЕСКИЕ ТРЕБОВАНИЯ", "Схемы и чертежи (наличие, количество)",
  "Какие схемы и чертежи есть в РД/ТЗ (наличие, обозначения, количество листов/альбомов)?",
  ["Альбомы РД, количество чертежей, обозначения."], MASK_TECH, False)
Q(28, "3. ТЕХНИЧЕСКИЕ ТРЕБОВАНИЯ", "Спецификации оборудования (марки, модели)",
  "Какие марки и модели оборудования указаны в спецификациях?",
  ["Марки приборов: Рубеж и др."], MASK_TECH + MASK_ESTIMATE, False)
Q(29, "3. ТЕХНИЧЕСКИЕ ТРЕБОВАНИЯ", "Нормативные документы (ГОСТ, СП, СНиП)",
  "Какие ГОСТ, СП, СНиП, ПУЭ, ФЗ указаны (с номерами)?",
  ["Перечень НТД с номерами."], MASK_TECH + MASK_CONTRACT, False)
Q(30, "3. ТЕХНИЧЕСКИЕ ТРЕБОВАНИЯ", "Требования к монтажу",
  "Какие требования к монтажу?",
  ["Условия СМР, размещение оборудования."], MASK_TECH + MASK_CONTRACT, False)
Q(31, "3. ТЕХНИЧЕСКИЕ ТРЕБОВАНИЯ", "Требования к кабельным трассам",
  "Какие требования к кабельным трассам / прокладке кабелей?",
  ["Кабельные трассы, типы кабелей, способы прокладки."], MASK_TECH, False)
Q(32, "3. ТЕХНИЧЕСКИЕ ТРЕБОВАНИЯ", "Требования к заземлению и молниезащите",
  "Какие требования к заземлению и молниезащите?",
  ["Заземление, уравнивание потенциалов, молниезащита."], MASK_TECH, False)
Q(33, "3. ТЕХНИЧЕСКИЕ ТРЕБОВАНИЯ", "Требования к автоматизации и диспетчеризации",
  "Какие требования к автоматизации, диспетчеризации, ЕСУМИС / мониторингу?",
  ["Диспетчеризация и удалённый мониторинг."], MASK_TECH, False)
Q(34, "3. ТЕХНИЧЕСКИЕ ТРЕБОВАНИЯ", "Требования к ПНР (пусконаладочным работам)",
  "Какие требования к ПНР?",
  ["Состав и программа пуско-наладки."], MASK_TECH + MASK_CONTRACT, False)
Q(35, "3. ТЕХНИЧЕСКИЕ ТРЕБОВАНИЯ", "Требования к испытаниям и проверкам",
  "Какие требования к испытаниям, проверкам, сдаче систем?",
  ["Испытания, комплексное опробование, проверки."], MASK_TECH + MASK_CONTRACT, False)
Q(36, "3. ТЕХНИЧЕСКИЕ ТРЕБОВАНИЯ", "Требования по пожарной безопасности",
  "Какие требования по пожарной безопасности на объекте / при работах?",
  ["Инструктаж, режим ПБ, огнезащита."], MASK_TECH + MASK_CONTRACT + MASK_NOTICE, False)
Q(37, "3. ТЕХНИЧЕСКИЕ ТРЕБОВАНИЯ", "Требования к гарантийному обслуживанию",
  "Какие требования к гарантийному обслуживанию / сервису?",
  ["Гарантийное обслуживание оборудования и работ."], MASK_CONTRACT + MASK_TECH, False)

# --- 4. Условия контракта (приоритет Договор) ---
Q(38, "4. УСЛОВИЯ КОНТРАКТА", "Срок выполнения работ (этапы)",
  "Каковы этапы и сроки выполнения работ? Смотри график производства работ и ТЗ.",
  ["Стадийность и сроки этапов СМР/ПНР."], MASK_SCHEDULE + MASK_TECH + MASK_CONTRACT, False)
Q(39, "4. УСЛОВИЯ КОНТРАКТА", "Срок выполнения работ (конечная дата)",
  "Какова конечная дата / общий срок выполнения работ по графику или договору?",
  ["Дата окончания работ, продолжительность."], MASK_SCHEDULE + MASK_CONTRACT + MASK_NOTICE, False)
Q(40, "4. УСЛОВИЯ КОНТРАКТА", "Срок гарантии",
  "Каков срок гарантии (месяцев) и точка отсчёта?",
  ["Гарантия с даты акта окончания работ."], MASK_CONTRACT + MASK_NOTICE, False)
Q(41, "4. УСЛОВИЯ КОНТРАКТА", "Срок подачи заявок",
  "До какой даты подаются заявки?",
  ["Дата окончания подачи предложений."], MASK_NOTICE, False)
Q(42, "4. УСЛОВИЯ КОНТРАКТА", "Время окончания подачи заявок",
  "В какое время (часы:минуты) заканчивается приём заявок?",
  ["Время дедлайна подачи оферт."], MASK_NOTICE, False)
Q(43, "4. УСЛОВИЯ КОНТРАКТА", "Срок оплаты",
  "Каков срок оплаты? Если разные в Извещении и договоре — укажи оба.",
  ["Срок расчёта в рабочих/календарных днях."], MASK_NOTICE + MASK_CONTRACT, False)
Q(44, "4. УСЛОВИЯ КОНТРАКТА", "Поэтапное выполнение работ",
  "Предусмотрено ли поэтапное выполнение? Перечень этапов.",
  ["Этапы работ по ТЗ и графику."], MASK_SCHEDULE + MASK_TECH + MASK_CONTRACT, False)
Q(45, "4. УСЛОВИЯ КОНТРАКТА", "Порядок сдачи-приёмки работ",
  "Какой порядок сдачи-приёмки (акты, сроки)?",
  ["Акт окончания работ, приёмка."], MASK_CONTRACT, False)
Q(46, "4. УСЛОВИЯ КОНТРАКТА", "Ответственность за просрочку",
  "Какая ответственность за просрочку? Ищи в договоре.",
  ["Неустойка за просрочку исполнения."], MASK_CONTRACT, True)
Q(47, "4. УСЛОВИЯ КОНТРАКТА", "Штрафы и пени",
  "Какой размер штрафов и пеней в договоре (%/суммы)?",
  ["Пени % в день, штрафы по договору."], MASK_CONTRACT, True)
Q(48, "4. УСЛОВИЯ КОНТРАКТА", "Возможность расторжения контракта",
  "Условия расторжения / одностороннего отказа по договору.",
  ["Основания расторжения договора."], MASK_CONTRACT, True)
Q(49, "4. УСЛОВИЯ КОНТРАКТА", "Форс-мажорные обстоятельства",
  "Что указано о форс-мажоре в договоре?",
  ["Непреодолимая сила: уведомление, сроки, расторжение."], MASK_CONTRACT, True)
Q(50, "4. УСЛОВИЯ КОНТРАКТА", "Порядок разрешения споров",
  "Порядок разрешения споров по договору (претензия, суд).",
  ["Претензионный порядок и суд."], MASK_CONTRACT, True)
Q(51, "4. УСЛОВИЯ КОНТРАКТА", "Арбитражная оговорка",
  "Какой арбитражный суд указан в договоре?",
  ["Наименование арбитражного суда."], MASK_CONTRACT, True)
Q(52, "4. УСЛОВИЯ КОНТРАКТА", "Состав ЗИП (запасных частей)",
  "Состав ЗИП и условия передачи заказчику (ТЗ/ЛСР/РД/договор).",
  ["Перечень запасных частей и ЗИП."], MASK_TECH + MASK_ESTIMATE + MASK_CONTRACT, False)
Q(53, "4. УСЛОВИЯ КОНТРАКТА", "Запрет субподряда",
  "Запрещён ли субподряд / нужно ли согласие заказчика?",
  ["Условия привлечения субподрядчиков."], MASK_CONTRACT + MASK_TECH, False)

# --- 5. Финансы ---
Q(54, "5. ФИНАНСОВЫЕ УСЛОВИЯ", "Начальная (максимальная) цена контракта",
  "Какая НМЦК в рублях? Если нет — скажи «Не указано», не подменяй сметой.",
  ["НМЦК из Извещения / закупочной документации."], MASK_NOTICE + MASK_CONTRACT, False)
Q(55, "5. ФИНАНСОВЫЕ УСЛОВИЯ", "Сметная стоимость (ориентир)",
  "Итоговые суммы по ЛСР/ВОР (САУГПТ и ЕСУМИС). Пометь как ориентир, НЕ НМЦК.",
  ["Итого/Всего по локальным сметам ЛСР."], MASK_ESTIMATE, True)
Q(56, "5. ФИНАНСОВЫЕ УСЛОВИЯ", "Сводка из ЛСР (топ позиций)",
  "Топ-10 позиций из ЛСР/ВОР с наименованием, кол-вом и ценой (если есть) + строка Итого.",
  ["Крупнейшие позиции спецификации ЛСР САУГПТ и ЕСУМИС."], MASK_ESTIMATE, True)
Q(57, "5. ФИНАНСОВЫЕ УСЛОВИЯ", "Порядок оплаты",
  "Порядок оплаты, удержания, этапы платежей.",
  ["Условия оплаты и гарантийные удержания."], MASK_NOTICE + MASK_CONTRACT, False)
Q(58, "5. ФИНАНСОВЫЕ УСЛОВИЯ", "Аванс",
  "Аванс: %, условия, банковская гарантия.",
  ["Максимальный размер аванса."], MASK_NOTICE + MASK_CONTRACT, False)
Q(59, "5. ФИНАНСОВЫЕ УСЛОВИЯ", "Обеспечение заявки",
  "Обеспечение заявки: размер и способ.",
  ["Обеспечение предложения."], MASK_NOTICE, False)
Q(60, "5. ФИНАНСОВЫЕ УСЛОВИЯ", "Обеспечение контракта",
  "Обеспечение исполнения договора (не путать с удержанием 5%).",
  ["Банковская гарантия / депозит обеспечения договора."], MASK_CONTRACT + MASK_NOTICE, False)
Q(61, "5. ФИНАНСОВЫЕ УСЛОВИЯ", "Валюта контракта",
  "В какой валюте контракт / расчёты?",
  ["Валюта договора."], MASK_NOTICE + MASK_CONTRACT, False)
Q(62, "5. ФИНАНСОВЫЕ УСЛОВИЯ", "Порядок индексации цены",
  "Предусмотрена ли индексация цены? Как?",
  ["Индексация / изменение цены договора."], MASK_CONTRACT + MASK_NOTICE, False)
Q(63, "5. ФИНАНСОВЫЕ УСЛОВИЯ", "График финансирования (детально)",
  "Детали графика освоения и финансирования денежных средств (этапы, суммы, сроки).",
  ["График финансирования по приложению к договору."], MASK_FINANCE_SCHED + MASK_CONTRACT + MASK_SCHEDULE, False)
Q(64, "5. ФИНАНСОВЫЕ УСЛОВИЯ", "Банковское сопровождение контракта",
  "Требуется ли банковское сопровождение контракта?",
  ["Банковское сопровождение / отдельный счёт."], MASK_CONTRACT + MASK_NOTICE, False)
Q(65, "5. ФИНАНСОВЫЕ УСЛОВИЯ", "Возможность снижения цены",
  "Есть ли возможность / порядок снижения цены?",
  ["Уменьшение цены договора."], MASK_CONTRACT + MASK_NOTICE, False)
Q(66, "5. ФИНАНСОВЫЕ УСЛОВИЯ", "Критерии оценки заявок",
  "Критерии оценки заявок / выбора победителя.",
  ["Оценка по стоимости лота / иные критерии."], MASK_NOTICE, False)

# --- 6. Риски (synthesize) ---
for num, title, prompt in [
    (67, "Основные риски для исполнителя", "Основные риски для исполнителя."),
    (68, "Риски по срокам", "Риски по срокам подачи и выполнения."),
    (69, "Риски по цене", "Финансовые риски (НМЦК vs смета, оплата, пени, обеспечения)."),
    (70, "Риски по качеству", "Риски по качеству, ТЗ, РД, оборудованию, гарантии."),
    (71, "Что не указано в документах", "Что важное отсутствует или неполно."),
    (72, "Что проверить перед подачей заявки", "Чек-лист перед подачей заявки."),
    (73, "Что можно улучшить в документации", "Недостатки документации для улучшения."),
    (74, "Рекомендация по участию", "Рекомендация: участвовать / осторожно / нет — с обоснованием."),
    (75, "Ключевые выводы", "Ключевые выводы (5–8 пунктов)."),
    (76, "Анализ конкурентной среды", "Какие выводы о конкурентной среде можно сделать по документации (критерий цены, барьеры входа, лицензии)?"),
    (77, "Оценка экономической целесообразности", "Оцени экономическую целесообразность участия при имеющихся данных."),
    (78, "Рекомендация по цене предложения", "Какую стратегию цены предложения рекомендовать (если НМЦК нет — опиши подход от сметы)?"),
]:
    Q(num, "6. РИСКИ И РЕКОМЕНДАЦИИ", title, prompt, kind="synthesize")

# --- 7. Мета / статус файлов ---
Q(79, "7. СТАТИСТИКА И ПОЛНОТА АНАЛИЗА", "Всего загружено файлов", "", kind="meta")
Q(80, "7. СТАТИСТИКА И ПОЛНОТА АНАЛИЗА", "Всего уникальных файлов (после дедупликации)", "", kind="meta")
Q(81, "7. СТАТИСТИКА И ПОЛНОТА АНАЛИЗА", "Статус обработки каждого файла", "", kind="meta")
Q(82, "7. СТАТИСТИКА И ПОЛНОТА АНАЛИЗА", "Какие файлы использованы в ответах", "", kind="meta")
Q(83, "7. СТАТИСТИКА И ПОЛНОТА АНАЛИЗА", "Какие файлы НЕ использованы", "", kind="meta")
Q(84, "7. СТАТИСТИКА И ПОЛНОТА АНАЛИЗА", "Причины неиспользования файлов", "", kind="meta")
Q(85, "7. СТАТИСТИКА И ПОЛНОТА АНАЛИЗА", "Рекомендации по дообработке файлов",
  "На основе статуса файлов и пробелов в ответах дай рекомендации по дообработке (какие файлы перечитать, что улучшить в OCR/парсинге).",
  kind="synthesize")
Q(86, "7. СТАТИСТИКА И ПОЛНОТА АНАЛИЗА", "Общий вывод по полноте анализа",
  "Общий вывод: насколько полон анализ (в % условно), какие зоны закрыты хорошо, какие слабые.",
  kind="synthesize")

assert len(QUESTIONS) == 86, f"Ожидалось 86 вопросов, получено {len(QUESTIONS)}"


# =============================================================================
# Номер тендера
# =============================================================================

TENDER_RE = re.compile(r"(B\d{10,})", re.IGNORECASE)


def find_tender_no() -> str:
    names = []
    if "uploaded_files" in globals() and uploaded_files:
        names.extend(uploaded_files.keys())
    names.extend(D_SOURCES)
    for n in names:
        m = TENDER_RE.search(str(n))
        if m:
            return m.group(1).upper()
    return ""


# =============================================================================
# Запуск
# =============================================================================

print("🚀 Анализ: 86 вопросов по всем файлам тендера (ускоренный режим)")
print(f"   Уник. файлов: {len(FILE_GROUPS)}")
print(f"   Чанков после дедупа: {len(D_CHUNKS)}")
print(f"   Вопросов: {len(QUESTIONS)}")
print(f"   TOP_K={TOP_K}, DeepSeek timeout={DEEPSEEK_TIMEOUT}с")
print()
_status("этап 4/5 — ответы на 86 вопросов")

tender_no = find_tender_no()
if tender_no:
    print(f"🔖 Номер из имён файлов: {tender_no}")

answers: Dict[int, str] = {}
sources_used: Dict[int, List[str]] = {}
files_touched: Set[str] = set()
per_file_hits: Dict[str, int] = defaultdict(int)

TOTAL = 86


def _fmt_eta(seconds: float) -> str:
    return _fmt_dur(seconds)


def _progress_bar(done: int, total: int, t0: float, title: str = "") -> str:
    total = max(1, total)
    pct = 100.0 * done / total
    width = 12
    filled = min(width, max(0, int(round(width * done / total))))
    bar = "█" * filled + "░" * (width - filled)
    elapsed = time.time() - t0
    if done > 0:
        eta_s = _fmt_eta(elapsed * (total - done) / done)
    else:
        eta_s = "оценка…"
    short = (title[:36] + "…") if len(title) > 37 else title
    passed = _fmt_dur(_elapsed("start"))
    return (
        f"🔄 СТАТУС: вопрос {min(done + 1, total)}/{total} — {short}  | "
        f"[{bar}] {pct:.0f}% (осталось {eta_s}, прошло {passed})"
    )


_t0_analysis = time.time()

for q in QUESTIONS:
    n = q["num"]
    print(_progress_bar(n - 1, TOTAL, _t0_analysis, q["title"]), flush=True)

    if q["kind"] == "rag":
        # спец: номер тендера из имени файла
        if n == 1 and tender_no:
            answers[n] = tender_no
            sources_used[n] = ["(имя загруженного файла)"]
            print(f"   ↳ из имени файла: {tender_no}")
            continue

        ans, hits = ask_rag(
            q["question"],
            variants=q["variants"],
            masks=q["masks"],
            mask_only=q["mask_only"],
            top_k=TOP_K,
        )
        answers[n] = ans
        bases = []
        for h in hits:
            b = h.get("base") or _basename(h["source"])
            bases.append(b)
            files_touched.add(_norm(b))
            per_file_hits[_norm(b)] += 1
        sources_used[n] = list(dict.fromkeys(bases))[:12]

        if n == 1 and not tender_no:
            m = TENDER_RE.search(ans)
            if m:
                tender_no = m.group(1).upper()
                answers[n] = tender_no

    elif q["kind"] == "synthesize":
        # контекст = собранные ответы 1-66 (и уже готовые synth)
        facts = []
        for qq in QUESTIONS:
            if qq["num"] >= n:
                break
            if qq["kind"] == "meta":
                continue
            if qq["num"] in answers:
                facts.append(f"{qq['num']}. {qq['title']}: {answers[qq['num']]}")
        ctx = "\n\n".join(facts)
        if len(ctx) > 30000:
            ctx = ctx[:30000] + "\n\n[...усечено...]"
        prompt = (
            "Ты эксперт по тендерам. Опирайся ТОЛЬКО на собранные факты ниже. "
            "Не противоречь им. Различай НМЦК и сметный ориентир.\n\n"
            f"Задание: {q['question']}"
        )
        answers[n] = clean(ask_deepseek(prompt, context=ctx or "Факты отсутствуют.", timeout=DEEPSEEK_TIMEOUT))
        sources_used[n] = ["(синтез ответов 1–N)"]

    elif q["kind"] == "meta":
        # заполним после цикла статусом — placeholder
        answers[n] = "<<META>>"
        sources_used[n] = ["(статистика модуля 4)"]

print(_progress_bar(TOTAL, TOTAL, _t0_analysis, "основной проход"), flush=True)
print(f"⏱️ Основной проход: {_fmt_eta(time.time() - _t0_analysis)}")

# --- Статус файлов ---
file_status_rows = []
for exp, key in FILE_PLAN:
    if key is None:
        cls = FILE_CLASSIFICATION.get(exp, {})
        file_status_rows.append({
            "file": exp,
            "in_index": "Нет",
            "used": "Нет",
            "hits": 0,
            "chunks": 0,
            "chars": 0,
            "doc_type": cls.get("doc_type", "Не найден в индексе"),
            "signals": cls.get("signals", []),
            "reason": "Не найден в индексе после дедупликации",
        })
        continue
    g = FILE_GROUPS[key]
    hits = per_file_hits.get(key, 0)
    used = "Да" if hits > 0 else "Нет"
    reason = "" if used == "Да" else "Ни один ответ не сослался на чанки этого файла (низкая релевантность / маска поиска)"
    cls = FILE_CLASSIFICATION.get(key, {})
    file_status_rows.append({
        "file": exp if exp in EXPECTED_FILES else g["display"],
        "in_index": "Да",
        "used": used,
        "hits": hits,
        "chunks": len(g["idxs"]),
        "chars": g["chars"],
        "doc_type": cls.get("doc_type", DOC_TYPE_OTHER),
        "signals": cls.get("signals", []),
        "reason": reason,
    })

uploaded_count = len(uploaded_files) if "uploaded_files" in globals() and uploaded_files else len(FILE_GROUPS)
unique_count = len(FILE_GROUPS)
used_files = sorted({r["file"] for r in file_status_rows if r["used"] == "Да"})
unused_files = [r for r in file_status_rows if r["used"] == "Нет"]

# meta answers
answers[79] = str(uploaded_count)
answers[80] = str(unique_count)
answers[81] = "См. раздел «Статус обработки файлов» ниже."
answers[82] = "; ".join(used_files) if used_files else "Не указано"
answers[83] = "; ".join(r["file"] for r in unused_files) if unused_files else "Нет — все учтённые файлы использованы"
answers[84] = "\n".join(
    f"- {r['file']}: {r['reason']}" for r in unused_files
) if unused_files else "Все файлы использованы в ответах."

# synthesize 85-86 now that meta ready
for q in QUESTIONS:
    if q["num"] in (85, 86):
        print(_progress_bar(q["num"] - 1, TOTAL, _t0_analysis, q["title"] + " (финальный синтез)"), flush=True)
        facts = []
        for qq in QUESTIONS:
            if qq["num"] == q["num"]:
                break
            if qq["num"] in answers:
                facts.append(f"{qq['num']}. {qq['title']}: {answers[qq['num']]}")
        facts.append("Статус файлов:\n" + "\n".join(
            f"- {r['file']}: type={r.get('doc_type')}, index={r['in_index']}, "
            f"used={r['used']}, hits={r['hits']}, reason={r['reason']}"
            for r in file_status_rows
        ))
        if APPROVALS:
            facts.append("Согласования:\n" + "\n".join(
                f"- {a.get('Файл')}: №{a.get('Номер письма')}, {a.get('Дата письма')}, "
                f"{a.get('Отправитель')} → {a.get('Суть согласования')}"
                for a in APPROVALS
            ))
        ctx = "\n\n".join(facts)
        if len(ctx) > 30000:
            ctx = ctx[:30000]
        answers[q["num"]] = clean(ask_deepseek(
            f"{PRIORITY_PROMPT}\n\nЗадание: {q['question']}",
            context=ctx,
            timeout=DEEPSEEK_TIMEOUT,
        ))
        sources_used[q["num"]] = ["(синтез + статус файлов)"]

print(_progress_bar(TOTAL, TOTAL, _t0_analysis, "анализ завершён"), flush=True)
print(f"⏱️ Полный анализ: {_fmt_eta(time.time() - _t0_analysis)}")

# =============================================================================
# Отчёт
# =============================================================================

def build_name(num: str) -> str:
    d = ANALYSIS_DATE.strftime("%Y%m%d")
    t = ANALYSIS_DATE.strftime("%H%M%S")
    if num:
        safe = re.sub(r"[^\w\-]+", "_", num)
        return f"Анализ_тендера_{safe}_{d}.txt"
    return f"Анализ_тендера_{d}_{t}.txt"


def format_report() -> str:
    lines = []
    lines.append("=" * 70)
    lines.append("📊 ПОЛНЫЙ АНАЛИЗ ТЕНДЕРНОЙ ДОКУМЕНТАЦИИ (86 вопросов)")
    lines.append("=" * 70)
    lines.append(f"Номер тендера / ИКЗ: {tender_no or answers.get(1, 'Не указано')}")
    lines.append(f"Дата анализа: {ANALYSIS_DATE.strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"Загружено файлов (upload): {uploaded_count}")
    lines.append(f"Уникальных файлов (дедуп): {unique_count}")
    lines.append(f"Чанков исходных / после дедупа: {len(rag_index.chunks)} / {len(D_CHUNKS)}")
    lines.append(f"Писем/согласований найдено: {len(APPROVALS)}")
    lines.append("")

    # --- Согласования и разрешения (сразу после шапки) ---
    lines.append("=" * 70)
    lines.append("СОГЛАСОВАНИЯ И РАЗРЕШЕНИЯ")
    lines.append("=" * 70)
    if not APPROVALS:
        lines.append("Письма-согласования / разрешения не обнаружены "
                     "(или OCR не дал достаточно текста для классификации).")
    else:
        for i, a in enumerate(APPROVALS, start=1):
            lines.append("")
            lines.append(f"--- Согласование {i} ---")
            lines.append(f"Файл: {a.get('Файл', 'Не указано')}")
            lines.append(f"Тип: {a.get('Тип', DOC_TYPE_LETTER)}")
            lines.append(f"Дата письма: {a.get('Дата письма', 'Не указано')}")
            lines.append(f"Номер письма: {a.get('Номер письма', 'Не указано')}")
            lines.append(f"Отправитель: {a.get('Отправитель', 'Не указано')}")
            lines.append(f"Получатель: {a.get('Получатель', 'Не указано')}")
            lines.append(f"Суть согласования: {a.get('Суть согласования', 'Не указано')}")
            lines.append(f"Результат: {a.get('Результат', 'Не указано')}")
    lines.append("")

    cur_sec = None
    for q in QUESTIONS:
        if q["section"] != cur_sec:
            if cur_sec is not None:
                lines.append("")
            lines.append("=" * 70)
            lines.append(q["section"])
            lines.append("=" * 70)
            cur_sec = q["section"]
        lines.append("")
        lines.append(f"{q['num']}. {q['title']}")
        lines.append("-" * 70)
        lines.append(answers.get(q["num"], "Не указано"))
        srcs = sources_used.get(q["num"]) or []
        if srcs:
            lines.append("Источники: " + "; ".join(srcs[:10]))

    lines.append("")
    lines.append("=" * 70)
    lines.append("СТАТУС ОБРАБОТКИ ФАЙЛОВ")
    lines.append("=" * 70)
    lines.append(
        f"{'Файл':<48} {'Тип документа':<28} {'Индекс':<8} {'Исп.':<6} {'Hits':<5} {'Чанков'}"
    )
    lines.append("-" * 120)
    for r in file_status_rows:
        name = r["file"] if len(r["file"]) <= 46 else r["file"][:43] + "..."
        dtype = str(r.get("doc_type", ""))
        if len(dtype) > 26:
            dtype = dtype[:23] + "..."
        lines.append(
            f"{name:<48} {dtype:<28} {r['in_index']:<8} {r['used']:<6} {r['hits']:<5} {r['chunks']}"
        )
        if r.get("signals"):
            lines.append(f"   признаки: {', '.join(r['signals'][:6])}")
        if r["used"] == "Нет":
            lines.append(f"   причина: {r['reason']}")
        lines.append(f"   символов текста: {r['chars']:,}")

    lines.append("=" * 70)
    return "\n".join(lines) + "\n"


_status("формирование TXT-отчёта")

_mark("analysis_end")
# end отметим после скачивания — чтобы итог включал запись/download
_t_deps = _elapsed("deps_start", "deps_end")
_t_up = _elapsed("upload_start", "upload_end")
_t_idx = _elapsed("index_start", "index_end")
_t_an = _elapsed("analysis_start", "analysis_end")

report_text = format_report()
report_text = report_text.rstrip() + (
    f"\n\n{'=' * 70}\n"
    f"СВОДКА ВРЕМЕНИ ВЫПОЛНЕНИЯ\n"
    f"{'=' * 70}\n"
    f"1. Зависимости / init:   {_fmt_dur(_t_deps)}\n"
    f"2. Загрузка файлов:      {_fmt_dur(_t_up)}\n"
    f"3. OCR / индексация:     {_fmt_dur(_t_idx)}\n"
    f"4. Анализ 86 вопросов:   {_fmt_dur(_t_an)}\n"
)

report_filename = build_name(tender_no)
with open(report_filename, "w", encoding="utf-8") as f:
    f.write(report_text)

tender_report = report_text
tender_report_filename = report_filename
tender_number = tender_no
tender_answers = answers
tender_file_status = file_status_rows

_manual_link = _manual_download_link(report_filename)

_status("скачивание отчёта")
if colab_files is not None:
    colab_files.download(report_filename)

_mark("end")
_t_all = _elapsed("start", "end")
tender_timings = {
    "deps": _t_deps,
    "upload": _t_up,
    "index": _t_idx,
    "analysis": _t_an,
    "total": _t_all,
}

# допишем ИТОГО в файл
try:
    with open(report_filename, "a", encoding="utf-8") as f:
        f.write(f"ИТОГО:                   {_fmt_dur(_t_all)}\n")
except Exception:
    pass

print()
print("✅ Отчёт сформирован!")
print(f"📁 Файл: {report_filename}")
print(f"📊 Вопросов: {TOTAL}")
print(f"⚡ Режим: TOP_K={TOP_K}, timeout={DEEPSEEK_TIMEOUT}с, ETA-бар включён")
print(f"📂 Файлов использовано: {sum(1 for r in file_status_rows if r['used']=='Да')} / {len(file_status_rows)}")
if tender_no:
    print(f"🔖 Номер тендера: {tender_no}")

print(f"\n📥 Файл скачан автоматически." if colab_files is not None else "\n💾 Автоскачивание недоступно (не Colab).")
print(f"📁 Если скачивание не началось, скачайте вручную:")
print(f"   🔗 {_manual_link}")

# =============================================================================
# ЭТАП 5/5 — Сводка по времени
# =============================================================================
_status("этап 5/5 — сводка по времени")
print("\n" + "=" * 70)
print("ЭТАП 5/5 — Сводка по времени выполнения")
print("=" * 70)
print(f"   1. Зависимости / init:   {_fmt_dur(_t_deps)}")
print(f"   2. Загрузка файлов:      {_fmt_dur(_t_up)}")
print(f"   3. OCR / индексация:     {_fmt_dur(_t_idx)}")
print(f"   4. Анализ 86 вопросов:   {_fmt_dur(_t_an)}")
print(f"   ─────────────────────────────")
print(f"   ИТОГО:                   {_fmt_dur(_t_all)}")
print(f"   Финальный статус:        {_CURRENT_STATUS}")
print()
_status("модуль 5 завершён")
print("✅ Модуль 5 завершён.")
