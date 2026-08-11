# =============================================================================
# МОДУЛЬ 2 — Загрузка данных (Google Colab)
# Выполняйте СТРОГО после модуля 1.
# Скопируйте ВЕСЬ код ниже в отдельную ячейку Colab и выполните.
# Использует окружение сессии из модуля 1; не ставит зависимости заново.
# Не извлекает текст и не строит индекс — только загрузка файлов.
# =============================================================================

from google.colab import files


def _format_size(num_bytes: int) -> str:
    """Человекочитаемый размер файла."""
    if num_bytes < 1024:
        return f"{num_bytes} Б"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} КБ"
    return f"{num_bytes / (1024 * 1024):.2f} МБ"


print("📂 Выберите файлы для загрузки (можно несколько)...")

# Результат files.upload() сразу в глобальную переменную сессии
# (имя -> байты). Отдельный uploaded не создаём.
uploaded_files = files.upload()

print()
if not uploaded_files:
    print("⚠️ Файлы не выбраны — uploaded_files пуст.")
else:
    print(f"📋 Загружено файлов: {len(uploaded_files)}")
    for filename, content in uploaded_files.items():
        print(f"  • {filename} — {_format_size(len(content))}")

print("✅ Данные загружены. Теперь выполните модуль 3 (Чтение и индексация).")
