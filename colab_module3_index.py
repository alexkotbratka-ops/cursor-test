# =============================================================================
# МОДУЛЬ 3 — Чтение и индексация (Google Colab)
# Выполняйте ПОСЛЕ модуля 2.
# Использует функции модуля 1: dedupe_uploaded, _process_uploaded_file, RAGIndex…
# =============================================================================

import time
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Set, Tuple

# --- Проверки сессии ---
if "uploaded_files" not in globals() or not uploaded_files:
    raise RuntimeError(
        "❌ uploaded_files пуст. Сначала выполните модуль 2 (Загрузка данных)."
    )
if "rag_index" not in globals() or "RAGIndex" not in globals():
    raise RuntimeError(
        "❌ rag_index / RAGIndex не найдены. Сначала выполните модуль 1."
    )
if "dedupe_uploaded" not in globals() or "_process_uploaded_file" not in globals():
    raise RuntimeError(
        "❌ Функции индексации не найдены. Перезапустите модуль 1."
    )

_t0_mod = time.time()
if "_mark" in globals():
    _mark("index_start")
print("=" * 70)
print("🚀 МОДУЛЬ 3 — Чтение и индексация")
print("=" * 70)

# Настройки (как в эталоне модуле 5)
OCR_DPI = int(globals().get("OCR_DPI", 200) or 200)
OCR_PDF_FORCE_FULL_IF_AVG_BELOW = 15
OCR_MIN_CHARS_PER_PAGE = int(globals().get("OCR_MIN_CHARS_PER_PAGE", 30) or 30)
OCR_IMAGE_AREA_RATIO = float(globals().get("OCR_IMAGE_AREA_RATIO", 0.08) or 0.08)
CHUNK_SIZE = int(globals().get("CHUNK_SIZE", 600) or 600)
CHUNK_OVERLAP = int(globals().get("CHUNK_OVERLAP", 100) or 100)
TOP_K = 5

# ВАЖНО: page_needs_ocr() НЕ переопределяем — единая из модуля 1.

rag_index = RAGIndex(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

print(
    f"⚙️ OCR: dpi={OCR_DPI}, lang=rus+eng, psm=6; "
    f"единая page_needs_ocr(); OCR строго 1 раз на PDF"
)
print(f"⚙️ Индекс: chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}, TOP_K={TOP_K}")

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
if "_PDF_OCR_STATUS" in globals() and isinstance(_PDF_OCR_STATUS, dict):
    _PDF_OCR_STATUS.clear()
print("🔒 Контроль OCR: единая page_needs_ocr + _PDF_OCR_STATUS (1 раз на PDF)")

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

if "_mark" in globals():
    _mark("index_end")
_elapsed = time.time() - _t0_mod
if "_fmt_dur" in globals():
    print(f"⏱️ Модуль 3: {_fmt_dur(_elapsed)}")
else:
    print(f"⏱️ Модуль 3: {int(_elapsed)} сек.")
print("✅ Индекс построен. Теперь выполните модуль 4 (Задать вопросы).")
