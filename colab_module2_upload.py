# =============================================================================
# МОДУЛЬ 2 — Загрузка данных (Google Colab)
# Выполняйте строго ПОСЛЕ модуля 1, в отдельной ячейке.
# Использует окружение сессии из модуля 1. Только загрузка файлов —
# без извлечения текста и без индексации.
# =============================================================================

from google.colab import files

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

print("\n✅ Данные загружены. Теперь выполните модуль 3 (Чтение и индексация).")
