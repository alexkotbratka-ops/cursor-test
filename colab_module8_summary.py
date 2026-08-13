# =============================================================================
# МОДУЛЬ 8 — Сводная таблица по тендерам с выбором (Google Colab)
# Выполняйте ПОСЛЕ модулей 4/5 (когда в /content/ есть Анализ_тендера_*.txt).
#
# Вход:  Анализ_тендера_*.txt в /content/
# Выход: Сводная_таблица_по_тендерам_YYYYMMDD_HHMMSS.txt + files.download()
#
# Выбор: ввод номеров через запятую или 'all'.
# Для автотеста/скрипта можно заранее задать:
#   MODULE8_SELECTION = "all"   # или "1,3"
# =============================================================================

import glob
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_T0 = datetime.now()
print("=" * 70)
print("🚀 МОДУЛЬ 8 — Сводная таблица по тендерам (с выбором)")
print("=" * 70)

NA = "—"
CONTENT_DIRS: List[str] = []
if os.path.isdir("/content"):
    CONTENT_DIRS.append("/content")
# локальный fallback (отладка вне Colab)
CONTENT_DIRS.append(os.path.abspath("."))

# Регулярки из ТЗ + совместимость с блоками вопросов модулей 4/5
patterns = {
    "tender": r"Номер тендера / ИКЗ:\s*(.+?)(?:\n|$)",
    "customer": r"Заказчик \(название\):\s*(.+?)(?:\n|$)",
    "subject": r"Полное название объекта закупки:\s*(.+?)(?:\n|$)",
    "nmck": r"Начальная \(максимальная\) цена контракта:\s*(.+?)(?:\n|$)",
    "deadline": r"Срок подачи заявок:\s*(.+?)(?:\n|$)",
    "verdict": r"Рекомендация по участию:\s*(.+?)(?:\n|$)",
    "date": r"Дата анализа:\s*(.+?)(?:\n|$)",
    "total_time": r"ИТОГО:\s*(.+?)(?:\n|$)",
}

RE_QA = re.compile(
    r"(?m)^(\d{1,2})\.\s+([^\n]+)\n-{3,}\n(.*?)(?=\n\d{1,2}\.\s+[^\n]+\n-{3,}|\n={3,}|\Z)",
    re.S,
)
RE_B_NUM = re.compile(r"(B\d{10,})", re.IGNORECASE)
RE_IKZ = re.compile(r"\b(\d{18,36})\b")
RE_FROM_FNAME = re.compile(
    r"Анализ_тендера_(.+?)_(\d{8})(?:_\d{6})?\.txt$", re.IGNORECASE
)


# =============================================================================
# 1) Поиск отчётов
# =============================================================================

def find_report_files() -> List[str]:
    found: List[str] = []
    seen = set()
    for root in CONTENT_DIRS:
        pattern = os.path.join(root, "Анализ_тендера_*.txt")
        for path in glob.glob(pattern):
            ap = os.path.abspath(path)
            if ap in seen or not os.path.isfile(ap):
                continue
            if os.path.basename(ap).startswith("Сводная_таблица"):
                continue
            seen.add(ap)
            found.append(ap)
    found.sort(key=lambda p: (os.path.getmtime(p), os.path.basename(p)))
    return found


# =============================================================================
# 2) Извлечение полей
# =============================================================================

def _clean(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"(?m)^Источники:.*$", "", s).strip()
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n+", " ", s).strip()
    return s


def _or_dash(s: str) -> str:
    s = _clean(s)
    if not s or s.lower() in {"не указано", "не применимо", "none", "null", "nan", "-", "—"}:
        return NA
    return s


def _first_line(s: str, max_len: int = 120) -> str:
    s = _or_dash(s)
    if s == NA:
        return NA
    part = re.split(r"[;\n]", s)[0].strip()
    if len(part) > max_len:
        part = part[: max_len - 1].rstrip() + "…"
    return part or NA


def _parse_qa_blocks(text: str) -> Dict[int, Tuple[str, str]]:
    out: Dict[int, Tuple[str, str]] = {}
    for m in RE_QA.finditer(text):
        num = int(m.group(1))
        if 1 <= num <= 86:
            out[num] = (m.group(2).strip(), _clean(m.group(3)))
    return out


def _qa_get(qa: Dict[int, Tuple[str, str]], num: int) -> str:
    return qa[num][1] if num in qa else ""


def _qa_by_title(qa: Dict[int, Tuple[str, str]], *keywords: str) -> str:
    for _n, (title, body) in qa.items():
        t = title.lower()
        if any(k.lower() in t for k in keywords):
            return body
    return ""


def _re_get(text: str, key: str) -> str:
    try:
        m = re.search(patterns[key], text, flags=re.IGNORECASE | re.MULTILINE)
    except re.error:
        return ""
    if not m:
        return ""
    return m.group(1).strip()


def _short_tender_no(raw: str, filename: str) -> str:
    s = _or_dash(raw)
    if s != NA and len(s) < 50:
        return s
    m = RE_B_NUM.search(raw or "")
    if m:
        return m.group(1).upper()
    m = RE_IKZ.search(raw or "")
    if m:
        return m.group(1)
    m = RE_FROM_FNAME.search(os.path.basename(filename))
    if m:
        return m.group(1)
    if s != NA:
        return s[:40]
    return NA


def _format_nmck(raw: str) -> str:
    s = _or_dash(raw)
    if s == NA:
        return NA
    m = re.search(
        r"(\d{1,3}(?:[ \u00a0]\d{3})+(?:[.,]\d+)?|\d+(?:[.,]\d+)?)\s*(млн|тыс|руб|₽)?",
        s,
        re.I,
    )
    if not m:
        return _first_line(s, 24)
    num_s = m.group(1).replace("\u00a0", " ").replace(" ", "").replace(",", ".")
    try:
        val = float(num_s)
    except ValueError:
        return _first_line(s, 24)
    unit = (m.group(2) or "").lower()
    if unit.startswith("млн"):
        val *= 1_000_000
    elif unit.startswith("тыс"):
        val *= 1_000
    if val >= 1_000_000:
        x = val / 1_000_000
        return f"{x:.0f}M" if abs(x - round(x)) < 1e-9 else f"{x:.1f}M"
    if val >= 1_000:
        return f"{val / 1_000:.0f}K"
    return f"{val:.0f}"


def _classify_verdict(text: str) -> str:
    t = (text or "").lower()
    if not t or t in {"—", "не указано"}:
        return NA
    skip = (
        "не участвовать", "не рекоменду", "пропустить", "отказаться",
        "нецелесообраз", "не стоит",
    )
    go = ("участвовать", "рекомендуется участие", "можно участвовать", "к участию")
    mid = ("осторож", "рассмотр", "условн", "с оговорк", "требует уточн", "требует дополнительн")
    has_skip = any(k in t for k in skip)
    has_go = any(k in t for k in go)
    has_mid = any(k in t for k in mid)
    if has_skip:
        return "Пропустить"
    if has_go and has_mid:
        return "Осторожно"
    if has_mid and "осторож" in t:
        return "Осторожно"
    if has_mid:
        return "Рассмотр."
    if has_go:
        return "Участвовать"
    return "Рассмотр."


def _parse_date(text: str, filename: str) -> str:
    raw = _re_get(text, "date")
    if raw:
        raw = _clean(raw)
        for fmt, n in (
            ("%Y-%m-%d %H:%M", 16),
            ("%Y-%m-%d", 10),
            ("%d.%m.%Y %H:%M", 16),
            ("%d.%m.%Y", 10),
        ):
            try:
                return datetime.strptime(raw[:n], fmt).strftime("%d.%m.%Y")
            except ValueError:
                continue
        return raw[:16]
    m = RE_FROM_FNAME.search(os.path.basename(filename))
    if m:
        try:
            return datetime.strptime(m.group(2), "%Y%m%d").strftime("%d.%m.%Y")
        except ValueError:
            pass
    try:
        return datetime.fromtimestamp(os.path.getmtime(filename)).strftime("%d.%m.%Y")
    except Exception:
        return NA


def _parse_total_time(text: str) -> Tuple[str, float]:
    """
    Возвращает (метка для таблицы, секунды для суммирования).
    Ищет «ИТОГО: …» в сводке времени; иначе сумму строк 1–4.
    """
    raw = _re_get(text, "total_time")
    if not raw:
        # запасной вариант: сложить пункты сводки
        parts = re.findall(
            r"(?m)^\d\.\s+[^:]+:\s*(.+)$",
            text,
        )
        # только блок СВОДКА ВРЕМЕНИ
        block = re.search(
            r"СВОДКА ВРЕМЕНИ ВЫПОЛНЕНИЯ[\s\S]*?(?=ИТОГО:|\Z)",
            text,
            re.I,
        )
        if block:
            parts = re.findall(r"(?m)^\d\.\s+[^:\n]+:\s*(.+)$", block.group(0))
            secs = sum(_duration_to_seconds(p) for p in parts)
            if secs > 0:
                return _seconds_to_label(secs), secs
        return NA, 0.0
    secs = _duration_to_seconds(raw)
    return _seconds_to_label(secs) if secs > 0 else _first_line(raw, 16), secs


def _duration_to_seconds(raw: str) -> float:
    s = _clean(raw).lower()
    if not s or s == NA:
        return 0.0
    total = 0.0
    # «1 ч 23 мин», «15 мин», «90 сек», «1:23:05»
    m = re.match(r"(\d+):(\d{2}):(\d{2})$", s)
    if m:
        return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
    m = re.match(r"(\d+):(\d{2})$", s)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2))
    for num, unit in re.findall(r"(\d+(?:[.,]\d+)?)\s*(ч|час|мин|сек|s|m|h)", s):
        v = float(num.replace(",", "."))
        if unit.startswith("ч") or unit == "h":
            total += v * 3600
        elif unit.startswith("мин") or unit == "m":
            total += v * 60
        else:
            total += v
    if total == 0 and re.fullmatch(r"\d+", s):
        total = float(s)  # голые секунды
    return total


def _seconds_to_label(secs: float) -> str:
    secs = int(round(secs))
    if secs <= 0:
        return NA
    if secs < 60:
        return f"{secs} сек"
    mins = secs // 60
    rem = secs % 60
    if mins < 60:
        return f"{mins} мин" if rem == 0 else f"{mins} мин {rem} сек"
    hours = mins // 60
    mins = mins % 60
    if mins == 0:
        return f"{hours} ч"
    return f"{hours} ч {mins} мин"


def parse_report(path: str) -> Dict[str, object]:
    """Извлекает поля одной строки сводки."""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        print(f"  ⚠️ Не прочитан {os.path.basename(path)}: {e}")
        return {
            "tender_no": NA,
            "customer": NA,
            "subject": NA,
            "nmck": NA,
            "deadline": NA,
            "verdict": NA,
            "verdict_raw": "",
            "date": NA,
            "time_label": NA,
            "time_sec": 0.0,
            "path": path,
            "file": os.path.basename(path),
        }

    qa = _parse_qa_blocks(text)

    tender_raw = _re_get(text, "tender") or _qa_get(qa, 1) or _qa_by_title(
        qa, "номер тендера", "икз"
    )
    customer = _re_get(text, "customer") or _qa_get(qa, 4) or _qa_by_title(qa, "заказчик")
    # В формате 4/5 «Заказчик (название)» — это заголовок вопроса, не «ключ: значение».
    # Паттерн ТЗ сработает только если ответ в той же строке; иначе берём блок Q4.
    subject = _re_get(text, "subject") or _qa_get(qa, 2) or _qa_by_title(
        qa, "объект закупки", "предмет", "наименование"
    )
    nmck_raw = _re_get(text, "nmck") or _qa_get(qa, 54) or _qa_by_title(
        qa, "нмцк", "начальная", "цена контракта"
    )
    deadline = _re_get(text, "deadline") or _qa_get(qa, 41) or _qa_by_title(
        qa, "срок подачи", "окончания подачи"
    )
    verdict_raw = _re_get(text, "verdict") or _qa_get(qa, 74) or _qa_by_title(
        qa, "рекомендация по участию", "рекомендация"
    )
    if not verdict_raw or verdict_raw.lower() == "не указано":
        verdict_raw = _qa_get(qa, 77) or _qa_get(qa, 86) or verdict_raw

    time_label, time_sec = _parse_total_time(text)

    return {
        "tender_no": _short_tender_no(tender_raw, path),
        "customer": _first_line(customer, 40),
        "subject": _first_line(subject, 36),
        "nmck": _format_nmck(nmck_raw),
        "deadline": _first_line(deadline, 16),
        "verdict": _classify_verdict(verdict_raw),
        "verdict_raw": _first_line(verdict_raw, 80),
        "date": _parse_date(text, path),
        "time_label": time_label,
        "time_sec": float(time_sec),
        "path": path,
        "file": os.path.basename(path),
    }


# =============================================================================
# 3) Выбор тендеров
# =============================================================================

def _pad(s: str, width: int) -> str:
    s = NA if s is None else str(s)
    if len(s) > width:
        return s[: width - 1] + "…"
    return s.ljust(width)


def print_found_list(rows: List[Dict[str, object]]) -> None:
    print()
    print("📋 НАЙДЕННЫЕ ОТЧЁТЫ:")
    print("=" * 70)
    print()
    headers = (("№", 3), ("Номер тендера", 18), ("Заказчик", 28), ("Дата анализа", 12))
    print("\t".join(_pad(h, w) for h, w in headers))
    print("-" * 70)
    for i, r in enumerate(rows, start=1):
        print(
            "\t".join(
                _pad(v, w)
                for v, w in (
                    (str(i), 3),
                    (str(r["tender_no"]), 18),
                    (str(r["customer"]), 28),
                    (str(r["date"]), 12),
                )
            )
        )
    print("=" * 70)


def parse_selection(raw: str, n: int) -> Optional[List[int]]:
    """
    Разбор ввода: all | 1,3,5 | 1
    Возвращает список 1-based индексов или None при ошибке.
    """
    s = (raw or "").strip().lower()
    if not s:
        return None
    if s in {"all", "все", "*"}:
        return list(range(1, n + 1))
    parts = re.split(r"[,\s;]+", s)
    idxs: List[int] = []
    for p in parts:
        if not p:
            continue
        if not p.isdigit():
            return None
        i = int(p)
        if i < 1 or i > n:
            return None
        if i not in idxs:
            idxs.append(i)
    return idxs or None


def ask_selection(n: int) -> List[int]:
    """
    Запрос выбора. Если в сессии задан MODULE8_SELECTION — используем его
    (удобно для автотеста / неинтерактивного запуска).
    """
    preset = globals().get("MODULE8_SELECTION")
    if preset is not None and str(preset).strip():
        parsed = parse_selection(str(preset), n)
        if parsed is not None:
            print(f"📌 Выбор из MODULE8_SELECTION = {preset!r} → {parsed}")
            return parsed
        print(f"⚠️ MODULE8_SELECTION={preset!r} некорректен — запрашиваем ввод.")

    while True:
        try:
            raw = input(
                "📌 Выберите тендеры для отчёта "
                "(введите номера через запятую, или 'all' для всех): "
            )
        except EOFError:
            print("⚠️ Ввод недоступен — выбираем все тендеры.")
            return list(range(1, n + 1))
        except KeyboardInterrupt:
            print("\n⚠️ Прервано пользователем — выбираем все тендеры.")
            return list(range(1, n + 1))
        parsed = parse_selection(raw, n)
        if parsed is not None:
            return parsed
        print(f"❌ Неверный ввод: {raw!r}. Примеры: all  |  1  |  1,3,5  (диапазон 1–{n})")


# =============================================================================
# 4) Формирование отчёта
# =============================================================================

def format_summary(rows: List[Dict[str, object]], formed_at: datetime) -> str:
    lines: List[str] = []
    lines.append("=" * 70)
    lines.append("📊 СВОДНАЯ ТАБЛИЦА ПО ТЕНДЕРАМ")
    lines.append("=" * 70)
    lines.append(f"Дата формирования: {formed_at.strftime('%d.%m.%Y %H:%M')}")
    lines.append(f"Всего тендеров: {len(rows)}")
    lines.append("")

    headers = (
        ("№", 3),
        ("Номер тендера", 18),
        ("Заказчик", 20),
        ("Предмет", 16),
        ("НМЦК", 8),
        ("Срок подачи", 12),
        ("Вердикт", 12),
        ("Время", 14),
        ("Дата анализа", 12),
    )
    head = "\t".join(_pad(h, w) for h, w in headers)
    lines.append(head)
    lines.append("-" * min(140, max(70, len(head) + 8)))

    for i, r in enumerate(rows, start=1):
        vals = (
            (str(i), 3),
            (str(r["tender_no"]), 18),
            (str(r["customer"]), 20),
            (str(r["subject"]), 16),
            (str(r["nmck"]), 8),
            (str(r["deadline"]), 12),
            (str(r["verdict"]), 12),
            (str(r["time_label"]), 14),
            (str(r["date"]), 12),
        )
        lines.append("\t".join(_pad(v, w) for v, w in vals))

    n_go = sum(1 for r in rows if r["verdict"] == "Участвовать")
    n_care = sum(1 for r in rows if r["verdict"] == "Осторожно")
    n_rev = sum(1 for r in rows if r["verdict"] == "Рассмотр.")
    n_skip = sum(1 for r in rows if r["verdict"] == "Пропустить")
    total_sec = sum(float(r.get("time_sec") or 0) for r in rows)

    lines.append("=" * 70)
    lines.append("📊 СТАТИСТИКА:")
    lines.append("=" * 70)
    lines.append(f"Всего тендеров: {len(rows)}")
    lines.append(f"Рекомендовано участвовать: {n_go}")
    lines.append(f"Рекомендовано с осторожностью: {n_care}")
    lines.append(f"Рекомендовано рассмотреть: {n_rev}")
    lines.append(f"Рекомендовано пропустить: {n_skip}")
    lines.append(f"Общее время анализа: {_seconds_to_label(total_sec) if total_sec else NA}")
    lines.append("")
    lines.append("📋 СПИСОК ОТЧЁТОВ:")
    lines.append("=" * 70)
    lines.append("")
    for r in rows:
        tlab = r["time_label"] if r["time_label"] != NA else "?"
        lines.append(f"{r['file']} ({tlab})")
        lines.append("")
    lines.append("📁 ВСЕ ОТЧЁТЫ ДОСТУПНЫ ДЛЯ СКАЧИВАНИЯ")
    lines.append("=" * 70)
    lines.append("")
    lines.append("Пути к файлам:")
    for r in rows:
        lines.append(f"  - {r['path']}")
    lines.append("")
    return "\n".join(lines)


# =============================================================================
# 5) Запуск
# =============================================================================

summary_text = ""
summary_filename = ""
tender_summary_rows: List[Dict[str, object]] = []
tender_summary_all_rows: List[Dict[str, object]] = []
tender_summary_text = ""
tender_summary_filename = ""

report_files = find_report_files()
print(f"📂 Каталоги: {', '.join(CONTENT_DIRS)}")
print(f"📄 Найдено отчётов: {len(report_files)}")

if not report_files:
    print("⚠️ Отчёты не найдены. Сначала выполните анализ тендеров.")
else:
    all_rows: List[Dict[str, object]] = []
    for path in report_files:
        print(f"  • {os.path.basename(path)}")
        all_rows.append(parse_report(path))

    def _sort_key(r: Dict[str, object]):
        d = str(r.get("date") or "")
        try:
            return (datetime.strptime(d, "%d.%m.%Y"), str(r.get("tender_no") or ""))
        except ValueError:
            return (datetime.min, str(r.get("file") or ""))

    all_rows.sort(key=_sort_key)
    tender_summary_all_rows = all_rows

    print_found_list(all_rows)
    selected_idxs = ask_selection(len(all_rows))
    rows = [all_rows[i - 1] for i in selected_idxs]
    print(f"✅ Выбрано тендеров: {len(rows)} → {[r['tender_no'] for r in rows]}")

    formed = datetime.now()
    summary_text = format_summary(rows, formed)
    out_name = f"Сводная_таблица_по_тендерам_{formed.strftime('%Y%m%d_%H%M%S')}.txt"
    if os.path.isdir("/content"):
        summary_path = os.path.join("/content", out_name)
    else:
        summary_path = os.path.abspath(out_name)

    try:
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary_text)
    except OSError as e:
        # запасной путь при проблемах с /content
        summary_path = os.path.abspath(out_name)
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary_text)
        print(f"⚠️ Сохранено в текущую папку из-за ошибки пути: {e}")

    tender_summary_rows = rows
    tender_summary_text = summary_text
    tender_summary_filename = summary_path
    summary_filename = summary_path

    print()
    print(summary_text)

    try:
        from google.colab import files as colab_files  # type: ignore
        colab_files.download(summary_path)
        print("📥 Сводная таблица скачана автоматически.")
    except Exception:
        print("💾 Автоскачивание недоступно (не Colab) — файл сохранён локально.")

    elapsed = int((datetime.now() - _T0).total_seconds())
    print()
    print("=" * 70)
    print("✅ Сводная таблица сформирована!")
    print(f"📁 Файл: {summary_path}")
    print(f"📊 Тендеров в отчёте: {len(rows)} / найдено {len(all_rows)}")
    print(f"⏱ Время модуля: {elapsed} сек.")
    print("=" * 70)
    print("Модуль 8 завершён.")
