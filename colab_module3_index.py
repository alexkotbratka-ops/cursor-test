# =============================================================================
# МОДУЛЬ 3 — Чтение и индексация (оптимизированный)
# Выполняйте строго ПОСЛЕ модуля 2.
# Оптимизации:
#   1) дедупликация по basename + MD5 до обработки
#   2) OCR: dpi=200, порог 30 симв — если текст есть, OCR не запускаем
#   3) chunk_size=600, TOP_K=6
#   4) параллельная обработка + прогресс-бар с ETA
# =============================================================================

import hashlib
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Set, Tuple


# --- Проверки сессии ---
if "uploaded_files" not in globals() or not uploaded_files:
    raise RuntimeError(
        "❌ uploaded_files пуст. Сначала выполните модуль 2 (Загрузка данных)."
    )

if "rag_index" not in globals() or not isinstance(rag_index, RAGIndex):
    raise RuntimeError(
        "❌ rag_index не найден. Сначала выполните модуль 1 (Запуск системы)."
    )


# =============================================================================
# 1) OCR + размер чанков + TOP_K (переопределяем настройки модуля 1)
# =============================================================================

OCR_DPI = 200
OCR_PDF_FORCE_FULL_IF_AVG_BELOW = 15
OCR_MIN_CHARS_PER_PAGE = 30          # как в модуле 1 (было 80)
OCR_MIN_ALPHA_RATIO = 0.25
OCR_IMAGE_AREA_RATIO = 0.70

CHUNK_SIZE = 600                     # было 800
CHUNK_OVERLAP = 100
TOP_K = 6                            # 5–8 для поиска в сессии


def page_needs_ocr(page, digital_text: str) -> bool:
    """
    Быстрая проверка: если цифровой текст уже есть — OCR не нужен.
    """
    text = (digital_text or "").strip()
    if len(text) >= OCR_MIN_CHARS_PER_PAGE:
        return False
    if not text:
        return True
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
        ratio = min(img_area / page_area, 1.0)
    except Exception:
        ratio = 0.0
    return ratio >= OCR_IMAGE_AREA_RATIO and len(text) < OCR_MIN_CHARS_PER_PAGE


# Пересоздаём индекс с новым размером чанка
rag_index = RAGIndex(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

print(
    f"⚙️ OCR: dpi={OCR_DPI}, OCR только если текст < {OCR_MIN_CHARS_PER_PAGE} симв/стр"
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
    True = это дубликат, пропустить.
    False = новый файл, зарегистрирован.
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

    # обычный файл
    if _register_or_skip_inner(filename, file_bytes):
        _log(f"  ⏭️ Пропуск дубликата: {_basename(filename)} (уже обработан)")
        return results

    text = _read_document_bytes(file_bytes, filename)
    if text:
        results.append((filename, text))
    return results


def _process_uploaded_file(filename: str, file_bytes: bytes) -> List[Tuple[str, str]]:
    """Читает один загруженный файл/архив. Потокобезопасно по логам."""
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
        else:
            for inner_name, data in members:
                inner_norm = str(inner_name).replace("\\", "/")
                inner_ext = _safe_ext(inner_norm)
                prefix = f"   • {inner_norm} [{_fmt_size(len(data))}]"
                source = f"{filename}/{inner_norm}"

                if _register_or_skip_inner(inner_norm, data):
                    lines.append(f"{prefix} → ⏭️ Пропуск дубликата: {inner_norm} (уже обработан)")
                    continue

                if inner_ext in SUPPORTED_ARCHIVES:
                    nested = _extract_from_any(data, source)
                    extracted.extend(nested)
                    if nested:
                        chars = sum(len(t) for _, t in nested)
                        lines.append(
                            f"{prefix} → вложенный архив, извлечено {len(nested)} док., {chars:,} символов"
                        )
                    else:
                        lines.append(f"{prefix} → вложенный архив, пусто / только дубли")
                    continue

                text = _read_document_bytes(data, inner_norm)
                if text:
                    extracted.append((source, text))
                    lines.append(f"{prefix} → {_status_label(inner_ext, True)} ({len(text):,} символов)")
                else:
                    lines.append(f"{prefix} → {_status_label(inner_ext, False)}")

        docs_count = len(extracted)
        chars_total = sum(len(t) for _, t in extracted)
        lines.append(f"   Итого по архиву: документов={docs_count}, символов={chars_total:,}")
        _log("\n".join(lines))
        return extracted

    # обычный файл
    if _register_or_skip_inner(filename, file_bytes):
        lines.append(f"   ⏭️ Пропуск дубликата: {filename} (уже обработан)")
        _log("\n".join(lines))
        return []

    text = _read_document_bytes(file_bytes, filename)
    if text:
        extracted = [(filename, text)]
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

# дедуп верхнего уровня
unique_files = dedupe_uploaded(dict(uploaded_files))

# сбрасываем множества внутренних дублей и регистрируем уже принятые upload-хэши
_SEEN_INNER_HASHES.clear()
_SEEN_INNER_NAMES.clear()
for _name, _data in unique_files.items():
    _SEEN_INNER_HASHES.add(_md5(_data))
    _SEEN_INNER_NAMES.add(_canon_basename(_name))

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

print("\n✅ Индекс построен. Теперь выполните модуль 4 (Задать вопросы).")
