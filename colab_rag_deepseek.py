# =============================================================================
# RAG-система для анализа документов (Google Colab, одна ячейка)
# Скопируйте весь код ниже в одну ячейку Colab и запустите.
# =============================================================================

# --- Установка зависимостей (лёгкие библиотеки, без torch / sentence-transformers) ---
import subprocess
import sys

def _pip_install(*packages):
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", *packages],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )

print("📦 Установка зависимостей...")
try:
    _pip_install(
        "pdfplumber",
        "python-docx",
        "openpyxl",
        "openai",
        "scikit-learn",
        "rarfile",
    )
except Exception as e:
    print(f"⚠️ Ошибка установки pip-пакетов: {e}")

# unrar нужен для .rar (в Colab обычно доступен apt)
try:
    subprocess.run(
        ["apt-get", "install", "-y", "-qq", "unrar"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
except Exception:
    pass

print("✅ Зависимости готовы.\n")

# --- Импорты ---
import io
import os
import re
import zipfile
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import pdfplumber
from docx import Document
from openpyxl import load_workbook
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

try:
    from google.colab import files  # type: ignore
except ImportError:
    files = None
    print("⚠️ google.colab недоступен — загрузка через files.upload() не сработает вне Colab.")

try:
    import rarfile
except ImportError:
    rarfile = None

# --- Конфигурация ---
# Вставьте DeepSeek API key ниже (или задайте переменную окружения DEEPSEEK_API_KEY).
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "sk-ВАШ_КЛЮЧ_DEEPSEEK").strip()
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

if not DEEPSEEK_API_KEY or "ВАШ_КЛЮЧ" in DEEPSEEK_API_KEY:
    print(
        "⚠️ Укажите DEEPSEEK_API_KEY в коде или через os.environ "
        "перед вызовом ask_question()."
    )

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 5

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".txt", ".zip", ".rar"}

# Глобальное хранилище
chunks: List[str] = []
chunk_sources: List[str] = []
vectorizer: Optional[TfidfVectorizer] = None
tfidf_matrix = None

client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)


# =============================================================================
# Извлечение текста
# =============================================================================

def extract_text_from_pdf(file_bytes: bytes, filename: str) -> str:
    texts = []
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                try:
                    page_text = page.extract_text() or ""
                    if page_text.strip():
                        texts.append(page_text)
                except Exception as e:
                    print(f"  ⚠️ PDF страница {i} ({filename}): {e}")
    except Exception as e:
        print(f"  ❌ Ошибка чтения PDF {filename}: {e}")
    return "\n".join(texts)


def extract_text_from_docx(file_bytes: bytes, filename: str) -> str:
    texts = []
    try:
        doc = Document(io.BytesIO(file_bytes))
        for para in doc.paragraphs:
            if para.text.strip():
                texts.append(para.text)
        for table in doc.tables:
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells if c.text.strip()]
                if cells:
                    texts.append(" | ".join(cells))
    except Exception as e:
        print(f"  ❌ Ошибка чтения DOCX {filename}: {e}")
    return "\n".join(texts)


def extract_text_from_xlsx(file_bytes: bytes, filename: str) -> str:
    texts = []
    try:
        wb = load_workbook(io.BytesIO(file_bytes), data_only=True, read_only=True)
        for sheet in wb.worksheets:
            texts.append(f"[Лист: {sheet.title}]")
            for row in sheet.iter_rows(values_only=True):
                cells = [str(c).strip() for c in row if c is not None and str(c).strip()]
                if cells:
                    texts.append(" | ".join(cells))
        wb.close()
    except Exception as e:
        print(f"  ❌ Ошибка чтения XLSX {filename}: {e}")
    return "\n".join(texts)


def extract_text_from_txt(file_bytes: bytes, filename: str) -> str:
    for encoding in ("utf-8", "cp1251", "latin-1"):
        try:
            return file_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    print(f"  ⚠️ Не удалось декодировать TXT {filename}, использую latin-1 с заменой")
    return file_bytes.decode("latin-1", errors="replace")


def extract_text_from_bytes(file_bytes: bytes, filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_bytes, filename)
    if ext == ".docx":
        return extract_text_from_docx(file_bytes, filename)
    if ext == ".xlsx":
        return extract_text_from_xlsx(file_bytes, filename)
    if ext == ".txt":
        return extract_text_from_txt(file_bytes, filename)
    return ""


def extract_from_zip(file_bytes: bytes, filename: str) -> List[Tuple[str, str]]:
    results = []
    try:
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
            for name in zf.namelist():
                if name.endswith("/") or Path(name).name.startswith("."):
                    continue
                # пропускаем служебные файлы macOS
                if "__MACOSX" in name or name.startswith("."):
                    continue
                ext = Path(name).suffix.lower()
                if ext not in {".pdf", ".docx", ".xlsx", ".txt"}:
                    print(f"  ⏭️ Пропуск внутри ZIP: {name}")
                    continue
                try:
                    data = zf.read(name)
                    text = extract_text_from_bytes(data, name)
                    if text.strip():
                        results.append((f"{filename}/{name}", text))
                    else:
                        print(f"  ⚠️ Пустой текст: {filename}/{name}")
                except Exception as e:
                    print(f"  ❌ Ошибка файла в ZIP ({name}): {e}")
    except Exception as e:
        print(f"  ❌ Ошибка чтения ZIP {filename}: {e}")
    return results


def extract_from_rar(file_bytes: bytes, filename: str) -> List[Tuple[str, str]]:
    results = []
    if rarfile is None:
        print(f"  ❌ rarfile не установлен, пропускаю {filename}")
        return results
    try:
        with tempfile.NamedTemporaryFile(suffix=".rar", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        try:
            with rarfile.RarFile(tmp_path) as rf:
                for info in rf.infolist():
                    name = info.filename
                    if info.is_dir() or Path(name).name.startswith("."):
                        continue
                    if "__MACOSX" in name:
                        continue
                    ext = Path(name).suffix.lower()
                    if ext not in {".pdf", ".docx", ".xlsx", ".txt"}:
                        print(f"  ⏭️ Пропуск внутри RAR: {name}")
                        continue
                    try:
                        data = rf.read(info)
                        text = extract_text_from_bytes(data, name)
                        if text.strip():
                            results.append((f"{filename}/{name}", text))
                        else:
                            print(f"  ⚠️ Пустой текст: {filename}/{name}")
                    except Exception as e:
                        print(f"  ❌ Ошибка файла в RAR ({name}): {e}")
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    except Exception as e:
        print(f"  ❌ Ошибка чтения RAR {filename}: {e}")
        print("     Убедитесь, что установлен unrar (apt-get install unrar).")
    return results


def process_uploaded_file(filename: str, file_bytes: bytes) -> List[Tuple[str, str]]:
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        print(f"⏭️ Неподдерживаемый формат: {filename}")
        return []

    print(f"📄 Обработка: {filename}")
    if ext == ".zip":
        return extract_from_zip(file_bytes, filename)
    if ext == ".rar":
        return extract_from_rar(file_bytes, filename)

    text = extract_text_from_bytes(file_bytes, filename)
    if text.strip():
        return [(filename, text)]
    print(f"  ⚠️ Не удалось извлечь текст из {filename}")
    return []


# =============================================================================
# Чанкинг и индекс TF-IDF
# =============================================================================

def split_into_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    result = []
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


def build_index(documents: List[Tuple[str, str]]) -> None:
    global chunks, chunk_sources, vectorizer, tfidf_matrix

    chunks = []
    chunk_sources = []

    for source, text in documents:
        parts = split_into_chunks(text)
        for part in parts:
            chunks.append(part)
            chunk_sources.append(source)

    if not chunks:
        vectorizer = None
        tfidf_matrix = None
        print("❌ Нет текстовых чанков для индексации.")
        return

    try:
        vectorizer = TfidfVectorizer(
            max_features=50000,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        tfidf_matrix = vectorizer.fit_transform(chunks)
        print(f"✅ Индекс построен: {len(chunks)} чанков из {len(documents)} документ(ов).")
    except Exception as e:
        vectorizer = None
        tfidf_matrix = None
        print(f"❌ Ошибка построения TF-IDF индекса: {e}")


def retrieve_relevant_chunks(question: str, top_k: int = TOP_K) -> List[Dict]:
    if vectorizer is None or tfidf_matrix is None or not chunks:
        return []
    try:
        q_vec = vectorizer.transform([question])
        scores = cosine_similarity(q_vec, tfidf_matrix).flatten()
        top_idx = scores.argsort()[::-1][:top_k]
        results = []
        for i in top_idx:
            if scores[i] <= 0:
                continue
            results.append(
                {
                    "text": chunks[i],
                    "source": chunk_sources[i],
                    "score": float(scores[i]),
                }
            )
        return results
    except Exception as e:
        print(f"❌ Ошибка поиска: {e}")
        return []


# =============================================================================
# Генерация ответа (DeepSeek)
# =============================================================================

def generate_answer(question: str, context_chunks: List[Dict]) -> str:
    if not context_chunks:
        return "Не удалось найти релевантный контекст в загруженных документах."

    context_blocks = []
    for i, ch in enumerate(context_chunks, start=1):
        context_blocks.append(
            f"[Источник {i}: {ch['source']} | релевантность={ch['score']:.3f}]\n{ch['text']}"
        )
    context = "\n\n---\n\n".join(context_blocks)

    system_prompt = (
        "Ты — ассистент для анализа документов. Отвечай на вопрос пользователя "
        "строго на основе предоставленного контекста. Если ответа нет в контексте — "
        "честно скажи об этом. Отвечай на русском языке, кратко и по делу. "
        "В конце укажи использованные источники."
    )
    user_prompt = (
        f"Контекст из документов:\n\n{context}\n\n"
        f"Вопрос: {question}\n\n"
        "Сформулируй ответ по контексту и перечисли источники."
    )

    try:
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=1500,
        )
        answer = (response.choices[0].message.content or "").strip()
        if not answer:
            return "Модель вернула пустой ответ."
        return answer
    except Exception as e:
        return f"Ошибка обращения к DeepSeek API: {e}"


def ask_question(question: str) -> str:
    """Задаёт вопрос по загруженным документам. Выводит и возвращает ответ."""
    question = (question or "").strip()
    if not question:
        output = "Вопрос — Укажите непустой вопрос."
        print(output)
        return output

    if not chunks or vectorizer is None:
        formatted = f"{question} — Индекс не готов. Сначала загрузите документы."
        print(formatted)
        return formatted

    relevant = retrieve_relevant_chunks(question)
    answer = generate_answer(question, relevant)

    # Источники (уникальные)
    if relevant:
        unique_sources = []
        for ch in relevant:
            if ch["source"] not in unique_sources:
                unique_sources.append(ch["source"])
        sources_line = "; ".join(unique_sources)
        full_answer = f"{answer}\n\nИсточники: {sources_line}"
    else:
        full_answer = answer

    formatted = f"{question} — {full_answer}"
    print(formatted)
    return formatted


# =============================================================================
# Загрузка файлов и запуск
# =============================================================================

def upload_and_index() -> None:
    if files is None:
        print("❌ Функция files.upload() доступна только в Google Colab.")
        return

    print(
        "📁 Выберите файлы для загрузки "
        "(PDF, DOCX, XLSX, TXT, ZIP, RAR)..."
    )
    try:
        uploaded = files.upload()
    except Exception as e:
        print(f"❌ Ошибка загрузки файлов: {e}")
        return

    if not uploaded:
        print("⚠️ Файлы не выбраны.")
        return

    all_docs: List[Tuple[str, str]] = []
    for filename, content in uploaded.items():
        try:
            docs = process_uploaded_file(filename, content)
            all_docs.extend(docs)
        except Exception as e:
            print(f"❌ Необработанная ошибка для {filename}: {e}")

    if not all_docs:
        print("❌ Не удалось извлечь текст ни из одного файла.")
        return

    build_index(all_docs)
    print("\n🎉 Готово! Используйте ask_question(\"Ваш вопрос\").")


# --- Автозапуск пайплайна ---
upload_and_index()

# Пример (раскомментируйте после загрузки документов):
# ask_question("О чём этот документ?")
