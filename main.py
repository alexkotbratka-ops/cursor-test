#!/usr/bin/env python3
"""OP System — загрузка и анализ документов через RAG + DeepSeek."""

import os
import sys

from rag import RAGSystem


def main() -> int:
    # API-ключ: переменная окружения DEEPSEEK_API_KEY (предпочтительно)
    # или передайте через аргумент окружения перед запуском.
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        print(
            "❌ Не задан DEEPSEEK_API_KEY.\n"
            "   export DEEPSEEK_API_KEY='ваш_ключ'\n"
            "   python main.py"
        )
        return 1

    rag = RAGSystem(api_key=api_key)

    print("=" * 60)
    print("🚀 OP SYSTEM — ЗАГРУЗКА И АНАЛИЗ ДОКУМЕНТОВ")
    print("=" * 60)

    # --- ВАРИАНТ 1: указать путь вручную (раскомментируйте) ---
    # file_path = r"/path/to/document.pdf"

    # --- ВАРИАНТ 2: ввести путь в консоли ---
    file_path = input(
        "📁 Введите полный путь к файлу (PDF, DOCX, XLSX, ZIP, RAR): "
    ).strip().strip('"').strip("'")

    if not os.path.exists(file_path):
        print(f"❌ Файл не найден: {file_path}")
        return 1

    print(f"\n📄 Загружаю: {os.path.basename(file_path)}")
    try:
        result = rag.load_file(file_path)
    except Exception as exc:  # noqa: BLE001
        print(f"❌ Ошибка загрузки: {exc}")
        return 1

    if result and result.get("chunks_count", 0) > 0:
        print(f"\n✅ Документ загружен! Добавлено {result['chunks_count']} чанков.")
        if result.get("sources"):
            print("   Источники:", ", ".join(result["sources"]))
    else:
        print("\n⚠️ Не удалось извлечь текст из файла.")
        return 1

    print("\n" + "=" * 60)
    print("💬 ЗАДАЙ ВОПРОС ПО ДОКУМЕНТУ (пустая строка — выход)")
    print("=" * 60)

    while True:
        question = input("\n❓ Введите вопрос: ").strip()
        if not question:
            print("Выход.")
            break

        result = rag.ask(question)

        print("\n" + "=" * 60)
        print("🤖 ОТВЕТ:")
        print("=" * 60)
        print(result["answer"])
        print("=" * 60)

        if result.get("sources"):
            print("\n📚 ИСТОЧНИКИ:")
            for i, src in enumerate(result["sources"][:3], 1):
                preview = src["text"][:200].replace("\n", " ")
                print(f"{i}. [{src.get('source', '?')}] {preview}...")

    return 0


if __name__ == "__main__":
    sys.exit(main())
