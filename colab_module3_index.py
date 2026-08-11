# =============================================================================
# МОДУЛЬ 3 — Чтение и индексация (Google Colab)
# Выполняйте строго ПОСЛЕ модуля 2. Использует функции/переменные модуля 1:
#   uploaded_files, extract_text, extract_text_from_doc, extract_documents_from_bytes,
#   unpack_archive, rag_index, RAGIndex, SUPPORTED_DOCS, SUPPORTED_ARCHIVES
# Автономный: без ручного ввода.
# Исправлено: корректная обработка .doc (в т.ч. внутри архивов с путями Windows).
# =============================================================================


def _fmt_size(n: int) -> str:
    if n >= 1024 * 1024:
        return f"{n / (1024 * 1024):.2f} MB ({n:,} байт)"
    if n >= 1024:
        return f"{n / 1024:.2f} KB ({n:,} байт)"
    return f"{n} байт"


def _safe_ext(filename: str) -> str:
    """
    Надёжное расширение файла.
    Учитывает составные (.tar.gz) и пути Windows с '\\' внутри ZIP/RAR.
    """
    if "file_ext" in globals():
        # нормализуем слэши до вызова file_ext из модуля 1
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


def _read_document_bytes(file_bytes: bytes, filename: str) -> str:
    """
    Извлекает текст из одного документа.
    Для .doc явно вызывает extract_text_from_doc (antiword),
    чтобы не зависеть от устаревшей версии extract_text в сессии.
    """
    ext = _safe_ext(filename)
    # для antiword/временных файлов лучше короткое имя с правильным суффиксом
    clean_name = _basename(filename) or f"document{ext}"

    if ext == ".doc":
        text = ""
        # 1) явный вызов antiword-экстрактора из модуля 1
        if "extract_text_from_doc" in globals():
            try:
                text = extract_text_from_doc(file_bytes, clean_name) or ""
            except Exception as e:
                print(f"  ⚠️ extract_text_from_doc({clean_name}): {e}")
        # 2) запасной путь через универсальный extract_text
        if not (text or "").strip() and "extract_text" in globals():
            try:
                text = extract_text(file_bytes, clean_name) or ""
            except Exception as e:
                print(f"  ⚠️ extract_text({clean_name}): {e}")
        return (text or "").strip()

    if "extract_text" in globals():
        try:
            return (extract_text(file_bytes, clean_name) or "").strip()
        except Exception as e:
            print(f"  ⚠️ extract_text({clean_name}): {e}")
            return ""
    return ""


def _status_label(ext: str, ok: bool) -> str:
    if ok:
        if ext == ".doc":
            return "прочитан (.doc / antiword)"
        if ext in SUPPORTED_DOCS:
            return "прочитан"
        return "прочитан (fallback)"
    if ext == ".doc":
        return "пропущен (.doc: antiword не извлёк текст — проверьте модуль 1 / antiword)"
    return "пропущен (пустой текст / не удалось прочитать)"


def _extract_from_any(file_bytes: bytes, filename: str):
    """
    Рекурсивно извлекает документы из файла или архива.
    Не зависит от Path.suffix и корректно обрабатывает .doc.
    """
    ext = _safe_ext(filename)
    results = []

    if ext in SUPPORTED_ARCHIVES:
        try:
            members = unpack_archive(file_bytes, filename)
        except Exception as e:
            print(f"  ❌ Ошибка распаковки {filename}: {e}")
            return results
        for inner_name, data in members:
            inner_norm = str(inner_name).replace("\\", "/")
            results.extend(_extract_from_any(data, f"{filename}/{inner_norm}"))
        return results

    text = _read_document_bytes(file_bytes, filename)
    if text:
        results.append((filename, text))
    return results


def _process_uploaded_file(filename: str, file_bytes: bytes):
    """
    Читает один загруженный файл (или архив).
    Возвращает список (source, text) и печатает подробный отчёт.
    """
    ext = _safe_ext(filename)
    size = len(file_bytes)
    extracted = []

    print("=" * 80)
    print(f"📄 Файл: {filename}")
    print(f"   Размер: {_fmt_size(size)}")
    print(f"   Тип: {'архив ' + ext if ext in SUPPORTED_ARCHIVES else ext or '(без расширения)'}")

    # ----- Архив -----
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
                inner_norm = str(inner_name).replace("\\", "/")
                inner_ext = _safe_ext(inner_norm)
                prefix = f"   • {inner_norm} [{_fmt_size(len(data))}]"
                source = f"{filename}/{inner_norm}"

                if inner_ext in SUPPORTED_ARCHIVES:
                    nested = _extract_from_any(data, source)
                    extracted.extend(nested)
                    if nested:
                        chars = sum(len(t) for _, t in nested)
                        print(
                            f"{prefix} → вложенный архив, извлечено {len(nested)} док., {chars:,} символов"
                        )
                    else:
                        print(f"{prefix} → вложенный архив, пропущен (пусто / ошибка)")
                    continue

                text = _read_document_bytes(data, inner_norm)
                if text:
                    extracted.append((source, text))
                    print(f"{prefix} → {_status_label(inner_ext, True)} ({len(text):,} символов)")
                else:
                    print(f"{prefix} → {_status_label(inner_ext, False)}")

        docs_count = len(extracted)
        chars_total = sum(len(t) for _, t in extracted)
        print(f"   Итого по архиву: документов={docs_count}, символов={chars_total:,}")
        return extracted

    # ----- Обычный файл (включая .doc) -----
    text = _read_document_bytes(file_bytes, filename)
    if text:
        extracted = [(filename, text)]
    else:
        # запасной путь через модуль 1
        try:
            extracted = extract_documents_from_bytes(file_bytes, filename) or []
        except Exception:
            extracted = []

    docs_count = len(extracted)
    chars_total = sum(len(t) for _, t in extracted)
    print(f"   Статус: {_status_label(ext, docs_count > 0)}")
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

# Диагностика поддержки .doc в сессии (модуль 1)
_doc_ok = ".doc" in SUPPORTED_DOCS if "SUPPORTED_DOCS" in globals() else False
_antiword_fn = "extract_text_from_doc" in globals()
print(f"🔍 Чтение и индексация: файлов к обработке — {len(uploaded_files)}")
print(f"   .doc в SUPPORTED_DOCS: {'да' if _doc_ok else 'нет — перезапустите модуль 1'}")
print(f"   extract_text_from_doc: {'да' if _antiword_fn else 'нет — перезапустите модуль 1'}")
print()

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

    # отдельно покажем, сколько .doc попало в индекс
    doc_sources = [s for s in set(rag_index.sources) if _safe_ext(s) == ".doc"]
    print("\n📊 Итоговая статистика")
    print("-" * 40)
    print(f"   Загружено файлов (модуль 2): {len(uploaded_files)}")
    print(f"   Извлечено документов:        {len(documents)}")
    print(f"   Источников в индексе:        {n_sources}")
    print(f"   из них .doc:                 {len(doc_sources)}")
    print(f"   Чанков:                      {n_chunks}")
    print(f"   Размер словаря TF-IDF:       {vocab_size:,}")
    print(f"   Всего символов текста:       {sum(len(t) for _, t in documents):,}")
    if doc_sources:
        print("   .doc источники:")
        for s in sorted(doc_sources):
            print(f"      • {s}")

print("\n✅ Индекс построен. Теперь выполните модуль 4 (Задать вопросы).")
