# =============================================================================
# МОДУЛЬ 1 — Запуск системы анализа документов (Google Colab)
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


print("📦 Установка системных пакетов (tesseract, poppler, unrar, p7zip)...")
_run(["apt-get", "update", "-qq"])
_run(
    [
        "apt-get",
        "install",
        "-y",
        "-qq",
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
        "pytesseract",
        "pdf2image",
        "Pillow",
        "scikit-learn",
        "requests",
        "rarfile",
        "py7zr",
        "numpy",
    ],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.STDOUT,
)
print("✅ Зависимости установлены.\n")

# --- 2. Импорты ---
import io
import os
import re
import zipfile
import tempfile
from getpass import getpass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz  # pymupdf
import numpy as np
import py7zr
import pytesseract
import rarfile
import requests
from docx import Document
from openpyxl import load_workbook
from pdf2image import convert_from_bytes
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- 3. API-ключ DeepSeek (сохраняется в переменной сессии) ---
DEEPSEEK_API_KEY = getpass("🔑 Введите API-ключ DeepSeek: ").strip()
os.environ["DEEPSEEK_API_KEY"] = DEEPSEEK_API_KEY

DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 5
OCR_LANG = "rus+eng"

SUPPORTED_DOCS = {".pdf", ".docx", ".xlsx", ".xls", ".txt", ".md", ".csv", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
SUPPORTED_ARCHIVES = {".zip", ".rar", ".7z"}
SUPPORTED_EXTENSIONS = SUPPORTED_DOCS | SUPPORTED_ARCHIVES


# =============================================================================
# 4. Извлечение текста
# =============================================================================

def extract_text_from_pdf(file_bytes: bytes, filename: str = "") -> str:
    """Извлекает текст из PDF через PyMuPDF; при пустых страницах — OCR."""
    parts: List[str] = []
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        for i, page in enumerate(doc, start=1):
            text = (page.get_text("text") or "").strip()
            if text:
                parts.append(f"[Страница {i}]\n{text}")
            else:
                try:
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    ocr_text = pytesseract.image_to_string(img, lang=OCR_LANG).strip()
                    if ocr_text:
                        parts.append(f"[Страница {i} | OCR]\n{ocr_text}")
                except Exception as e:
                    print(f"  ⚠️ OCR страницы {i} ({filename}): {e}")
        doc.close()
    except Exception as e:
        print(f"  ❌ Ошибка PDF {filename}: {e}")
        # запасной путь: весь PDF через pdf2image + OCR
        ocr = ocr_pdf_bytes(file_bytes, filename)
        if ocr:
            return ocr
    return "\n\n".join(parts)


def extract_text_from_docx(file_bytes: bytes, filename: str = "") -> str:
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
    except Exception as e:
        print(f"  ❌ Ошибка DOCX {filename}: {e}")
    return "\n".join(parts)


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


def extract_text_from_txt(file_bytes: bytes, filename: str = "") -> str:
    for encoding in ("utf-8", "cp1251", "latin-1"):
        try:
            return file_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    return file_bytes.decode("latin-1", errors="replace")


def extract_text_from_image(file_bytes: bytes, filename: str = "") -> str:
    """OCR для изображений (PNG/JPG/TIFF и т.д.)."""
    try:
        img = Image.open(io.BytesIO(file_bytes))
        return pytesseract.image_to_string(img, lang=OCR_LANG).strip()
    except Exception as e:
        print(f"  ❌ Ошибка OCR изображения {filename}: {e}")
        return ""


def ocr_pdf_bytes(file_bytes: bytes, filename: str = "") -> str:
    """Полный OCR PDF через pdf2image + Tesseract."""
    parts: List[str] = []
    try:
        images = convert_from_bytes(file_bytes, dpi=200)
        for i, img in enumerate(images, start=1):
            text = pytesseract.image_to_string(img, lang=OCR_LANG).strip()
            if text:
                parts.append(f"[Страница {i} | OCR]\n{text}")
    except Exception as e:
        print(f"  ❌ OCR PDF {filename}: {e}")
    return "\n\n".join(parts)


def extract_text(file_bytes: bytes, filename: str) -> str:
    """Универсальное извлечение текста по расширению файла."""
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_bytes, filename)
    if ext == ".docx":
        return extract_text_from_docx(file_bytes, filename)
    if ext in {".xlsx", ".xls"}:
        return extract_text_from_xlsx(file_bytes, filename)
    if ext in {".txt", ".md", ".csv"}:
        return extract_text_from_txt(file_bytes, filename)
    if ext in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}:
        return extract_text_from_image(file_bytes, filename)
    return ""


# =============================================================================
# Распаковка архивов
# =============================================================================

def _iter_archive_members_zip(file_bytes: bytes) -> List[Tuple[str, bytes]]:
    members: List[Tuple[str, bytes]] = []
    with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
        for name in zf.namelist():
            if name.endswith("/") or "__MACOSX" in name or Path(name).name.startswith("."):
                continue
            members.append((name, zf.read(name)))
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
                members.append((name, rf.read(info)))
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


def unpack_archive(file_bytes: bytes, filename: str) -> List[Tuple[str, bytes]]:
    """Распаковывает ZIP / RAR / 7Z и возвращает список (имя, байты)."""
    ext = Path(filename).suffix.lower()
    try:
        if ext == ".zip":
            return _iter_archive_members_zip(file_bytes)
        if ext == ".rar":
            return _iter_archive_members_rar(file_bytes)
        if ext == ".7z":
            return _iter_archive_members_7z(file_bytes)
    except Exception as e:
        print(f"  ❌ Ошибка распаковки {filename}: {e}")
    return []


def extract_documents_from_bytes(file_bytes: bytes, filename: str) -> List[Tuple[str, str]]:
    """
    Извлекает документы из файла или архива.
    Возвращает список (источник, текст).
    """
    ext = Path(filename).suffix.lower()
    results: List[Tuple[str, str]] = []

    if ext in SUPPORTED_ARCHIVES:
        for inner_name, data in unpack_archive(file_bytes, filename):
            inner_ext = Path(inner_name).suffix.lower()
            if inner_ext in SUPPORTED_ARCHIVES:
                # вложенный архив
                nested = extract_documents_from_bytes(data, f"{filename}/{inner_name}")
                results.extend(nested)
                continue
            if inner_ext not in SUPPORTED_DOCS:
                print(f"  ⏭️ Пропуск: {filename}/{inner_name}")
                continue
            text = extract_text(data, inner_name)
            if text.strip():
                results.append((f"{filename}/{inner_name}", text))
            else:
                print(f"  ⚠️ Пустой текст: {filename}/{inner_name}")
        return results

    if ext not in SUPPORTED_DOCS:
        print(f"⏭️ Неподдерживаемый формат: {filename}")
        return []

    text = extract_text(file_bytes, filename)
    if text.strip():
        results.append((filename, text))
    else:
        print(f"  ⚠️ Не удалось извлечь текст из {filename}")
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
        """
        Строит индекс по списку (source, text).
        Возвращает число чанков.
        """
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
        print(f"✅ Индекс построен: {len(self.chunks)} чанков из {len({s for s in self.sources})} источников.")
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
            timeout=120,
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


def ask_with_rag(question: str, top_k: int = TOP_K) -> Dict[str, Any]:
    """Удобная обёртка: поиск по индексу + ответ DeepSeek."""
    hits = rag_index.search(question, top_k=top_k)
    if not hits:
        return {
            "answer": "❌ Индекс пуст или ничего не найдено. Сначала выполните модуль 2 (Загрузка данных).",
            "sources": [],
        }
    context = "\n\n---\n\n".join(
        f"[Источник: {h['source']} | score={h['score']:.3f}]\n{h['text']}" for h in hits
    )
    answer = ask_deepseek(question, context=context)
    return {"answer": answer, "sources": hits}


# --- Готово ---
if DEEPSEEK_API_KEY:
    print("🔐 API-ключ DeepSeek сохранён в сессии.")
else:
    print("⚠️ API-ключ пустой — ask_deepseek потребует ключ позже.")

print("✅ Система запущена. Теперь выполните модуль 2 (Загрузка данных).")
