# =============================================================================
# МОДУЛЬ 3 — Чтение и индексация (Google Colab)
# Выполняйте строго ПОСЛЕ модуля 2. Использует:
#   uploaded_files, extract_documents_from_bytes, unpack_archive,
#   rag_index, SUPPORTED_DOCS, SUPPORTED_ARCHIVES, Path
# Автономный: без ручного ввода.
# =============================================================================


def _fmt_size(n: int) -> str:
    if n >= 1024 * 1024:
        return f"{n / (1024 * 1024):.2f} MB ({n:,} байт)"
    if n >= 1024:
        return f"{n / 1024:.2f} KB ({n:,} байт)"
    return f"{n} байт"


def _process_uploaded_file(filename: str, file_bytes: bytes):
    """
    Читает один загруженный файл (или архив).
    Возвращает список (source, text) и печатает подробный отчёт.
    """
    ext = Path(filename).suffix.lower()
    size = len(file_bytes)
    extracted = []

    print("=" * 80)
    print(f"📄 Файл: {filename}")
    print(f"   Размер: {_fmt_size(size)}")
    print(f"   Тип: {'архив ' + ext if ext in SUPPORTED_ARCHIVES else ext or '(без расширения)'}")

    if ext in SUPPORTED_ARCHIVES:
        print("   Содержимое архива:")
        try:
            members = unpack_archive(file_bytes, filename)
        except Exception as e:
            print(f"   ❌ Не удалось открыть архив: {e}")
            members = []

        if not members:
            print("   ⚠️ Архив пуст или не удалось прочитать.")
        else:
            for inner_name, data in members:
                inner_ext = Path(inner_name).suffix.lower()
                prefix = f"   • {inner_name} [{_fmt_size(len(data))}]"

                if inner_ext in SUPPORTED_ARCHIVES:
                    nested = extract_documents_from_bytes(data, f"{filename}/{inner_name}")
                    if nested:
                        extracted.extend(nested)
                        chars = sum(len(t) for _, t in nested)
                        print(f"{prefix} → вложенный архив, извлечено {len(nested)} док., {chars:,} символов")
                    else:
                        print(f"{prefix} → вложенный архив, пропущен (пусто / ошибка)")
                    continue

                if inner_ext not in SUPPORTED_DOCS:
                    print(f"{prefix} → пропущен (неподдерживаемый формат)")
                    continue

                text = extract_text(data, inner_name)
                if text.strip():
                    source = f"{filename}/{inner_name}"
                    extracted.append((source, text))
                    print(f"{prefix} → прочитан ({len(text):,} символов)")
                else:
                    print(f"{prefix} → пропущен (пустой текст / не удалось прочитать)")

        docs_count = len(extracted)
        chars_total = sum(len(t) for _, t in extracted)
        print(f"   Итого по архиву: документов={docs_count}, символов={chars_total:,}")
        return extracted

    # Обычный (неархивный) файл
    extracted = extract_documents_from_bytes(file_bytes, filename)
    docs_count = len(extracted)
    chars_total = sum(len(t) for _, t in extracted)
    if docs_count:
        status = "прочитан"
    elif ext not in SUPPORTED_DOCS:
        status = "пропущен (неподдерживаемый формат)"
    else:
        status = "пропущен (пустой текст / не удалось прочитать)"
    print(f"   Статус: {status}")
    print(f"   Документов извлечено: {docs_count}")
    print(f"   Символов: {chars_total:,}")
    return extracted


# --- Проверки сессии ---
if "uploaded_files" not in globals() or not uploaded_files:
    raise RuntimeError(
        "❌ uploaded_files пуст. Сначала выполните модуль 2 (Загрузка данных)."
    )

if "rag_index" not in globals() or not isinstance(rag_index, RAGIndex):
    raise RuntimeError(
        "❌ rag_index не найден. Сначала выполните модуль 1 (Запуск системы)."
    )

print(f"🔍 Чтение и индексация: файлов к обработке — {len(uploaded_files)}\n")

documents = []
for name, data in uploaded_files.items():
    docs = _process_uploaded_file(name, data)
    documents.extend(docs)

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

    print("\n📊 Итоговая статистика")
    print("-" * 40)
    print(f"   Загружено файлов (модуль 2): {len(uploaded_files)}")
    print(f"   Извлечено документов:        {len(documents)}")
    print(f"   Источников в индексе:        {n_sources}")
    print(f"   Чанков:                      {n_chunks}")
    print(f"   Размер словаря TF-IDF:       {vocab_size:,}")
    print(f"   Всего символов текста:       {sum(len(t) for _, t in documents):,}")

print("\n✅ Индекс построен. Теперь выполните модуль 4 (Задать вопросы).")
