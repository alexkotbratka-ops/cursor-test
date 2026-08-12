# =============================================================================
# МОДУЛЬ 1 — Запуск системы анализа документов (Google Colab)
# Поддержка форматов, типичных для тендерной документации.
# Скопируйте ВЕСЬ код ниже в одну ячейку Colab и выполните.
# =============================================================================

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
from typing import Any, Dict, List, Optional, Tuple

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

CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
TOP_K = 5
OCR_LANG = "rus+eng"
DEEPSEEK_TIMEOUT = 60                # сек. на ответ API (было 120)

# Пороги детекции сканов / «бедного» текста
# Быстрая проверка: если цифровой текст уже есть (≥ порога) — OCR не запускаем.
OCR_MIN_CHARS_PER_PAGE = 30          # меньше → страница считается сканом (было 80)
OCR_MIN_ALPHA_RATIO = 0.35           # доля букв среди непробельных символов
OCR_IMAGE_AREA_RATIO = 0.55          # доля площади страницы под картинками
OCR_DPI = 200                        # dpi для pdf2image / pixmap (было 300)
DOCX_OCR_MIN_TEXT_CHARS = 200       # если в DOCX уже ≥ N символов — OCR картинок не нужен
ARCHIVE_MAX_DEPTH = 4                 # макс. глубина вложенных архивов

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
# OCR helpers — полное распознавание сканов и изображений
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
    """OCR PIL-изображения: несколько PSM + предобработка."""
    try:
        processed = _preprocess_for_ocr(img)
        configs = [
            "--oem 3 --psm 6",   # блок текста (письмо)
            "--oem 3 --psm 4",   # одна колонка
            "--oem 3 --psm 3",   # авто
        ]
        best = ""
        for cfg in configs:
            try:
                text = pytesseract.image_to_string(
                    processed, lang=OCR_LANG, config=cfg
                ).strip()
            except Exception:
                text = ""
            if len(text) > len(best):
                best = text
        # запасной проход без препроцесса
        if len(best) < 20:
            try:
                raw = img.convert("RGB") if img.mode != "RGB" else img
                alt = pytesseract.image_to_string(raw, lang=OCR_LANG).strip()
                if len(alt) > len(best):
                    best = alt
            except Exception:
                pass
        return best
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


def page_needs_ocr(page, digital_text: str) -> bool:
    """
    Решение: нужен ли OCR для страницы.
    Быстрая проверка: если цифровой текст уже есть — OCR НЕ запускаем.
    OCR только для пустых / почти пустых страниц.
    """
    stats = _text_quality_stats(digital_text)
    # Главный ускоритель: текст уже извлечён → пропускаем OCR
    if stats["chars"] >= OCR_MIN_CHARS_PER_PAGE:
        return False
    if not (digital_text or "").strip():
        return True
    # Очень бедный текст + картинка на весь лист → скан
    if (
        stats["chars"] < OCR_MIN_CHARS_PER_PAGE
        and _page_image_area_ratio(page) >= OCR_IMAGE_AREA_RATIO
    ):
        return True
    return stats["chars"] < OCR_MIN_CHARS_PER_PAGE


def ocr_pdf_page(page, page_no: int, filename: str = "") -> str:
    """Рендер страницы PDF → OCR."""
    try:
        zoom = OCR_DPI / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        text = ocr_pil_image(img, f"{filename}#p{page_no}")
        return text
    except Exception as e:
        print(f"  ⚠️ OCR страницы {page_no} ({filename}): {e}")
        return ""


def extract_text_from_pdf(file_bytes: bytes, filename: str = "") -> str:
    """
    PDF: цифровой текст; OCR только если текста нет / почти нет.
    Если в PDF уже есть текст — OCR не запускается (ускорение).
    """
    parts: List[str] = []
    ocr_pages = 0
    digital_chars_total = 0
    page_count = 0
    digital_pages = 0

    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        page_count = doc.page_count
        for i, page in enumerate(doc, start=1):
            digital = (page.get_text("text") or "").strip()
            digital_chars_total += len(digital)
            need_ocr = page_needs_ocr(page, digital)

            page_bits: List[str] = []
            if digital and not need_ocr:
                # Быстрый путь: текст есть → без OCR
                digital_pages += 1
                page_bits.append(digital)
            elif need_ocr:
                ocr_text = ocr_pdf_page(page, i, filename)
                if ocr_text:
                    ocr_pages += 1
                    if digital and len(digital) > 10:
                        # гибрид только если OCR реально богаче
                        if len(ocr_text) > len(digital) * 1.2:
                            page_bits = [f"[OCR]\n{ocr_text}"]
                        else:
                            page_bits = [digital, f"[OCR-дополнение]\n{ocr_text}"]
                    else:
                        page_bits.append(f"[OCR]\n{ocr_text}")
                elif digital:
                    page_bits.append(digital)

            if page_bits:
                parts.append(f"[Страница {i}]\n" + "\n".join(page_bits))
            else:
                print(f"  ⚠️ Страница {i} без текста даже после OCR: {filename}")
        doc.close()
    except Exception as e:
        print(f"  ❌ Ошибка PDF {filename}: {e}")
        return ocr_pdf_bytes(file_bytes, filename)

    result = "\n\n".join(parts)

    # Полный OCR только если почти нет цифрового текста
    avg = (digital_chars_total / page_count) if page_count else 0
    if page_count and avg < OCR_PDF_FORCE_FULL_IF_AVG_BELOW and digital_pages == 0:
        print(f"  🔍 PDF похож на скан (avg={avg:.0f} симв/стр, OCR-стр={ocr_pages}) — полный OCR: {filename}")
        full = ocr_pdf_bytes(file_bytes, filename)
        if len(full) > len(result):
            return full

    return result


def extract_text_from_docx(file_bytes: bytes, filename: str = "") -> str:
    """
    DOCX: сначала цифровой текст; OCR встроенных изображений —
    только если текста мало (иначе сканы-логотипы сильно замедляют).
    """
    parts: List[str] = []
    try:
        doc = Document(io.BytesIO(file_bytes))
        for para in doc.paragraphs:
            if para.text.strip():
                parts.append(para.text)
        for table in doc.tables:
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells if c.text.strip()]
                if cells:
                    parts.append(" | ".join(cells))

        digital = "\n".join(parts).strip()
        # ускорение: если текста достаточно — не гоняем OCR по всем картинкам
        min_chars = int(globals().get("DOCX_OCR_MIN_TEXT_CHARS", 200) or 200)
        if len(digital) >= min_chars:
            return digital

        # мало текста → возможно скан внутри DOCX
        try:
            img_idx = 0
            for rel in doc.part.rels.values():
                try:
                    if "image" not in getattr(rel, "reltype", ""):
                        continue
                    blob = rel.target_part.blob
                    img_idx += 1
                    ocr_text = extract_text_from_image(blob, f"{filename}#img{img_idx}")
                    if ocr_text:
                        parts.append(f"[Изображение {img_idx} | OCR]\n{ocr_text}")
                except Exception:
                    continue
            if img_idx:
                print(f"  🖼️ DOCX {filename}: OCR для {img_idx} изображений (мало текста: {len(digital)} симв.)")
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
    """Полный OCR PDF через pdf2image @ OCR_DPI + Tesseract."""
    parts: List[str] = []
    try:
        images = convert_from_bytes(file_bytes, dpi=OCR_DPI)
        print(f"  🔍 Полный OCR PDF ({len(images)} стр. @ {OCR_DPI} dpi): {filename}")
        for i, img in enumerate(images, start=1):
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

def _fix_zip_member_name(name: str, info: "zipfile.ZipInfo") -> str:
    """
    Имена в ZIP с Windows (часто CP866) без UTF-8 флага приходят как mojibake.
    Пробуем восстановить кириллицу.
    """
    name = name.replace("\\", "/")
    # Бит 11 = UTF-8
    if info.flag_bits & 0x800:
        return name
    try:
        raw = name.encode("cp437", errors="strict")
    except Exception:
        return name
    for enc in ("cp866", "cp1251", "utf-8"):
        try:
            decoded = raw.decode(enc)
            # предпочитаем вариант с кириллицей / читаемыми символами
            if any("а" <= ch.lower() <= "я" or ch in "ёЁ" for ch in decoded):
                return decoded
            if enc == "utf-8":
                return decoded
        except Exception:
            continue
    return name


def _iter_archive_members_zip(file_bytes: bytes) -> List[Tuple[str, bytes]]:
    """Распаковка ZIP с поддержкой кириллических имён (CP866/CP1251)."""
    members: List[Tuple[str, bytes]] = []
    try:
        zf = zipfile.ZipFile(io.BytesIO(file_bytes))
    except Exception as e:
        print(f"  ❌ ZIP не открылся: {e}")
        return members

    with zf:
        infos = zf.infolist()
        print(f"  📦 ZIP: записей в архиве = {len(infos)}")
        for info in infos:
            name = _fix_zip_member_name(info.filename, info)
            if name.endswith("/") or info.is_dir():
                print(f"     · (папка) {name}")
                continue
            if "__MACOSX" in name or Path(name).name.startswith("."):
                continue
            try:
                data = zf.read(info)
                members.append((name, data))
                print(f"     · {name} [{len(data):,} байт]")
            except Exception as e:
                print(f"  ⚠️ Не удалось прочитать из ZIP: {name}: {e}")
    print(f"  📦 ZIP: извлечено файлов = {len(members)}")
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


def extract_documents_from_bytes(
    file_bytes: bytes,
    filename: str,
    depth: int = 0,
) -> List[Tuple[str, str]]:
    """
    Извлекает документы из файла или архива (рекурсивно).
    depth — уровень вложенности архива (0 = исходный файл).
    ARCHIVE_MAX_DEPTH ограничивает распаковку (по умолчанию 2).
    """
    max_depth = int(globals().get("ARCHIVE_MAX_DEPTH", 4) or 4)
    ext = file_ext(filename)
    results: List[Tuple[str, str]] = []

    if ext in SUPPORTED_ARCHIVES:
        if depth >= max_depth:
            print(f"  ⏭️ Архив глубже {max_depth} ур. — пропуск: {filename}")
            return results
        for inner_name, data in unpack_archive(file_bytes, filename):
            nested = extract_documents_from_bytes(
                data, f"{filename}/{inner_name}", depth=depth + 1
            )
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


# --- Готово ---
print(
    "📎 Поддерживаемые форматы: "
    "docx/doc/rtf/odt/pdf(+OCR сканов)/txt/log/md/html/xml/json, "
    "eml/msg (+вложения OCR), "
    "xlsx/xls/csv/ods, "
    "jpg/png/bmp/gif/tif/webp (полный OCR), "
    "zip/rar/7z/tar/gz, "
    "dwg/dxf, ppt/pptx + fallback OCR для бинарных/без расширения."
)
print(
    f"🔍 OCR: порог {OCR_MIN_CHARS_PER_PAGE} симв/стр (текст есть → без OCR), "
    f"dpi={OCR_DPI}, timeout DeepSeek={DEEPSEEK_TIMEOUT}с, chunk={CHUNK_SIZE}"
)
if DEEPSEEK_API_KEY:
    print("🔐 API-ключ DeepSeek сохранён в сессии.")
else:
    print("⚠️ API-ключ пустой — ask_deepseek потребует ключ позже.")

print("✅ Система запущена. Теперь выполните модуль 2 (Загрузка данных).")
