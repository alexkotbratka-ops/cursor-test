# =============================================================================
# МОДУЛЬ 2 — Загрузка файлов (Google Colab)
# Выполняйте ПОСЛЕ модуля 1.
# =============================================================================

import time as _time_m2

if "uploaded_files" not in globals():
    raise RuntimeError(
        "❌ Сначала выполните модуль 1 (Инициализация). "
        "Глобальная переменная uploaded_files не найдена."
    )

_t0 = _time_m2.time()
if "_mark" in globals():
    _mark("upload_start")
print("=" * 70)
print("🚀 МОДУЛЬ 2 — Загрузка файлов")
print("=" * 70)

try:
    from google.colab import files
except ImportError as e:
    raise RuntimeError(
        "❌ google.colab недоступен. Запускайте этот код в Google Colab."
    ) from e

print("📂 Выберите файлы для загрузки (можно несколько, включая архивы)...")

# Результат files.upload() — словарь {имя: bytes}
uploaded_files = files.upload()

if not uploaded_files:
    raise RuntimeError("❌ Файлы не загружены. Перезапустите модуль 2 и выберите файлы.")

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

if "_mark" in globals():
    _mark("upload_end")
_elapsed = _time_m2.time() - _t0
if "_fmt_dur" in globals():
    print(f"\n⏱️ Модуль 2: {_fmt_dur(_elapsed)}")
else:
    print(f"\n⏱️ Модуль 2: {int(_elapsed)} сек.")

print("✅ Данные загружены. Теперь выполните модуль 3 (Чтение и индексация).")
