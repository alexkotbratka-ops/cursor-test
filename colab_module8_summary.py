# =============================================================================
# МОДУЛЬ 8 — Сводная таблица по всем проанализированным тендерам (Google Colab)
# Выполняйте ПОСЛЕ модулей 4/5 (когда в /content/ есть Анализ_тендера_*.txt).
#
# Вход:  все файлы Анализ_тендера_*.txt в /content/ (и текущей папке — для отладки)
# Выход: Сводная_таблица_по_тендерам_YYYYMMDD_HHMMSS.txt + автоскачивание
# =============================================================================

import glob
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_T0 = datetime.now()
print("=" * 70)
print("🚀 МОДУЛЬ 8 — Сводная таблица по тендерам")
print("=" * 70)

NA = "—"
CONTENT_DIRS = []
if os.path.isdir("/content"):
    CONTENT_DIRS.append("/content")
CONTENT_DIRS.append(os.path.abspath("."))


# =============================================================================
# 1) Поиск отчётов
# =============================================================================

def find_report_files() -> List[str]:
    """Все Анализ_тендера_*.txt (без дублей по inode/пути)."""
    found: List[str] = []
    seen = set()
    for root in CONTENT_DIRS:
        for path in glob.glob(os.path.join(root, "Анализ_тендера_*.txt")):
            ap = os.path.abspath(path)
            if ap in seen or not os.path.isfile(ap):
                continue
            # не включаем саму сводную таблицу, если имя похоже
            if os.path.basename(ap).startswith("Сводная_таблица"):
                continue
            seen.add(ap)
            found.append(ap)
    found.sort(key=lambda p: (os.path.getmtime(p), os.path.basename(p)))
    return found


# =============================================================================
# 2) Извлечение полей из TXT
# =============================================================================

# Шапка отчёта (модули 4/5)
RE_HEADER_NUM = re.compile(r"(?m)^Номер тендера\s*/\s*ИКЗ:\s*(.+)$")
RE_HEADER_DATE = re.compile(r"(?m)^Дата анализа:\s*(.+)$")

# Пользовательские паттерны «ключ: значение» (на случай альтернативного формата)
RE_KV = {
    "tender_no": re.compile(
        r"(?mi)^(?:Номер тендера\s*/\s*ИКЗ|Номер тендера):\s*(.+)$"
    ),
    "customer": re.compile(
        r"(?mi)^Заказчик\s*\(название\):\s*(.+)$"
    ),
    "subject": re.compile(
        r"(?mi)^Полное название объекта закупки:\s*(.+)$"
    ),
    "nmck": re.compile(
        r"(?mi)^(?:Начальная\s*\(максимальная\)\s*цена контракта|НМЦК):\s*(.+)$"
    ),
    "deadline": re.compile(
        r"(?mi)^Срок подачи заявок:\s*(.+)$"
    ),
    "verdict": re.compile(
        r"(?mi)^Рекомендация по участию:\s*(.+)$"
    ),
    "date": re.compile(r"(?mi)^Дата анализа:\s*(.+)$"),
}

# Блоки вопросов модулей 4/5: «N. Заголовок\n----\nответ»
RE_QA = re.compile(
    r"(?m)^(\d{1,2})\.\s+([^\n]+)\n-{3,}\n(.*?)(?=\n\d{1,2}\.\s+[^\n]+\n-{3,}|\n={3,}|\Z)",
    re.S,
)

# Короткие номера из длинного текста
RE_B_NUM = re.compile(r"(B\d{10,})", re.IGNORECASE)
RE_IKZ = re.compile(r"\b(\d{18,36})\b")
RE_FROM_FNAME = re.compile(
    r"Анализ_тендера_(.+?)_(\d{8})(?:_\d{6})?\.txt$", re.IGNORECASE
)


def _clean(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"(?m)^Источники:.*$", "", s).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _first_line(s: str, max_len: int = 120) -> str:
    s = _clean(s)
    if not s or s.lower() in {"не указано", "none", "null", "nan", "-", "—"}:
        return NA
    # первая осмысленная строка / предложение
    part = re.split(r"[;\n]", s)[0].strip()
    if len(part) > max_len:
        part = part[: max_len - 1].rstrip() + "…"
    return part or NA


def _parse_qa_blocks(text: str) -> Dict[int, Tuple[str, str]]:
    """num -> (title, answer)."""
    out: Dict[int, Tuple[str, str]] = {}
    for m in RE_QA.finditer(text):
        num = int(m.group(1))
        if num < 1 or num > 86:
            continue
        title = m.group(2).strip()
        body = _clean(m.group(3))
        out[num] = (title, body)
    return out


def _answer_by_num(qa: Dict[int, Tuple[str, str]], num: int) -> str:
    if num in qa:
        return qa[num][1]
    return ""


def _answer_by_title(qa: Dict[int, Tuple[str, str]], *keywords: str) -> str:
    for _num, (title, body) in qa.items():
        t = title.lower()
        if any(k.lower() in t for k in keywords):
            return body
    return ""


def _short_tender_no(raw: str, filename: str) -> str:
    s = _clean(raw)
    if s and s != NA and len(s) < 50:
        return s
    m = RE_B_NUM.search(s or "")
    if m:
        return m.group(1).upper()
    m = RE_IKZ.search(s or "")
    if m:
        return m.group(1)
    m = RE_FROM_FNAME.search(os.path.basename(filename))
    if m:
        return m.group(1)
    if s and s != NA:
        return s[:40]
    return NA


def _format_nmck(raw: str) -> str:
    """Компактный вид: 5 000 000 → 5.0M, иначе укороченная строка."""
    s = _clean(raw)
    if not s or s == NA or s.lower() in {"не указано", "не применимо"}:
        return NA
    # вытащить число
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
        return f"{val / 1_000_000:.1f}M".replace(".0M", "M") if val % 1_000_000 == 0 else f"{val / 1_000_000:.1f}M"
    if val >= 1_000:
        return f"{val / 1_000:.0f}K"
    return f"{val:.0f}"


def _classify_verdict(text: str) -> str:
    """Короткая метка для таблицы: Участвовать / Осторожно / Рассмотр. / Пропустить."""
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
    m = RE_HEADER_DATE.search(text) or RE_KV["date"].search(text)
    if m:
        raw = _clean(m.group(1))
        # 2026-08-13 14:30 → 13.08.2026
        for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d", "%d.%m.%Y %H:%M", "%d.%m.%Y"):
            try:
                return datetime.strptime(raw[:16], fmt).strftime("%d.%m.%Y")
            except ValueError:
                continue
        return raw[:16]
    m = RE_FROM_FNAME.search(os.path.basename(filename))
    if m:
        d = m.group(2)
        try:
            return datetime.strptime(d, "%Y%m%d").strftime("%d.%m.%Y")
        except ValueError:
            pass
    try:
        return datetime.fromtimestamp(os.path.getmtime(filename)).strftime("%d.%m.%Y")
    except Exception:
        return NA


def parse_report(path: str) -> Dict[str, str]:
    """Извлекает поля сводной строки из одного TXT-отчёта."""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        print(f"  ⚠️ Не прочитан {path}: {e}")
        return {
            "tender_no": NA,
            "customer": NA,
            "subject": NA,
            "nmck": NA,
            "deadline": NA,
            "verdict": NA,
            "verdict_raw": "",
            "date": NA,
            "path": path,
            "file": os.path.basename(path),
        }

    qa = _parse_qa_blocks(text)

    # --- номер ---
    tender_raw = ""
    m = RE_HEADER_NUM.search(text) or RE_KV["tender_no"].search(text)
    if m:
        tender_raw = m.group(1)
    if not tender_raw:
        tender_raw = _answer_by_num(qa, 1) or _answer_by_title(qa, "номер тендера", "икз")
    tender_no = _short_tender_no(tender_raw, path)

    # --- заказчик ---
    customer = ""
    m = RE_KV["customer"].search(text)
    if m:
        customer = m.group(1)
    if not customer:
        customer = _answer_by_num(qa, 4) or _answer_by_title(qa, "заказчик")

    # --- предмет ---
    subject = ""
    m = RE_KV["subject"].search(text)
    if m:
        subject = m.group(1)
    if not subject:
        subject = _answer_by_num(qa, 2) or _answer_by_title(
            qa, "объект закупки", "предмет", "наименование"
        )

    # --- НМЦК ---
    nmck_raw = ""
    m = RE_KV["nmck"].search(text)
    if m:
        nmck_raw = m.group(1)
    if not nmck_raw:
        nmck_raw = _answer_by_num(qa, 54) or _answer_by_title(
            qa, "нмцк", "начальная", "цена контракта"
        )

    # --- срок подачи ---
    deadline = ""
    m = RE_KV["deadline"].search(text)
    if m:
        deadline = m.group(1)
    if not deadline:
        deadline = _answer_by_num(qa, 41) or _answer_by_title(
            qa, "срок подачи", "окончания подачи"
        )
    # добавить время (вопрос 42), если есть
    t42 = _answer_by_num(qa, 42)
    if deadline and t42 and t42.lower() not in {"не указано", "—"}:
        d1 = _first_line(deadline, 40)
        if d1 != NA and _first_line(t42, 20) not in d1:
            deadline = f"{d1}"

    # --- вердикт ---
    verdict_raw = ""
    m = RE_KV["verdict"].search(text)
    if m:
        verdict_raw = m.group(1)
    if not verdict_raw:
        verdict_raw = _answer_by_num(qa, 74) or _answer_by_title(
            qa, "рекомендация по участию", "рекомендация"
        )
    # запасной — вопрос 77/86
    if not verdict_raw or verdict_raw.lower() in {"не указано"}:
        verdict_raw = _answer_by_num(qa, 77) or _answer_by_num(qa, 86) or verdict_raw

    return {
        "tender_no": tender_no,
        "customer": _first_line(customer, 40),
        "subject": _first_line(subject, 36),
        "nmck": _format_nmck(nmck_raw),
        "deadline": _first_line(deadline, 16),
        "verdict": _classify_verdict(verdict_raw),
        "verdict_raw": _first_line(verdict_raw, 80),
        "date": _parse_date(text, path),
        "path": path,
        "file": os.path.basename(path),
    }


# =============================================================================
# 3) Таблица и статистика
# =============================================================================

def _pad(s: str, width: int) -> str:
    s = s if s is not None else NA
    # визуальная ширина ≈ len для кириллицы в моноширинных шрифтах Colab
    if len(s) > width:
        return s[: width - 1] + "…"
    return s.ljust(width)


def format_summary(rows: List[Dict[str, str]], formed_at: datetime) -> str:
    lines: List[str] = []
    lines.append("=" * 70)
    lines.append("📊 СВОДНАЯ ТАБЛИЦА ПО ТЕНДЕРАМ")
    lines.append("=" * 70)
    lines.append(f"Дата формирования: {formed_at.strftime('%d.%m.%Y %H:%M')}")
    lines.append(f"Всего тендеров: {len(rows)}")
    lines.append("")

    # Заголовок таблицы
    headers = (
        ("№", 3),
        ("Номер тендера", 18),
        ("Заказчик", 22),
        ("Предмет", 18),
        ("НМЦК", 8),
        ("Срок подачи", 12),
        ("Вердикт", 12),
        ("Дата анализа", 12),
    )
    head = "\t".join(_pad(h, w) for h, w in headers)
    lines.append(head)
    lines.append("-" * min(120, max(70, len(head) + 10)))

    for i, r in enumerate(rows, start=1):
        vals = (
            (str(i), 3),
            (r["tender_no"], 18),
            (r["customer"], 22),
            (r["subject"], 18),
            (r["nmck"], 8),
            (r["deadline"], 12),
            (r["verdict"], 12),
            (r["date"], 12),
        )
        lines.append("\t".join(_pad(v, w) for v, w in vals))

    lines.append("=" * 70)
    lines.append("📊 СТАТИСТИКА:")
    lines.append("=" * 70)
    n_go = sum(1 for r in rows if r["verdict"] == "Участвовать")
    n_care = sum(1 for r in rows if r["verdict"] == "Осторожно")
    n_rev = sum(1 for r in rows if r["verdict"] == "Рассмотр.")
    n_skip = sum(1 for r in rows if r["verdict"] == "Пропустить")
    lines.append(f"Всего тендеров: {len(rows)}")
    lines.append(f"Рекомендовано участвовать: {n_go}")
    lines.append(f"Рекомендовано с осторожностью: {n_care}")
    lines.append(f"Рекомендовано рассмотреть: {n_rev}")
    lines.append(f"Рекомендовано пропустить: {n_skip}")
    lines.append("")
    lines.append("📋 СПИСОК ОТЧЁТОВ:")
    lines.append("=" * 70)
    lines.append("")
    for r in rows:
        lines.append(r["file"])
        lines.append("")
    lines.append("📁 ВСЕ ОТЧЁТЫ ДОСТУПНЫ ДЛЯ СКАЧИВАНИЯ")
    lines.append("=" * 70)
    lines.append("")
    # детальные пути (для Colab)
    lines.append("Пути к файлам:")
    for r in rows:
        lines.append(f"  - {r['path']}")
    lines.append("")
    return "\n".join(lines)


# =============================================================================
# 4) Запуск
# =============================================================================

report_files = find_report_files()
print(f"📂 Каталоги: {', '.join(CONTENT_DIRS)}")
print(f"📄 Найдено отчётов: {len(report_files)}")

if not report_files:
    print("⚠️ Отчёты не найдены. Сначала выполните анализ тендеров.")
    summary_text = ""
    summary_filename = ""
    tender_summary_rows = []
else:
    rows: List[Dict[str, str]] = []
    for path in report_files:
        print(f"  • {os.path.basename(path)}")
        rows.append(parse_report(path))

    def _sort_key(r: Dict[str, str]):
        # дата анализа ДД.ММ.ГГГГ → для хронологии; иначе имя файла
        d = r.get("date") or ""
        try:
            return (datetime.strptime(d, "%d.%m.%Y"), r.get("tender_no") or "")
        except ValueError:
            return (datetime.min, r.get("file") or "")

    rows.sort(key=_sort_key)

    formed = datetime.now()
    summary_text = format_summary(rows, formed)
    summary_filename = f"Сводная_таблица_по_тендерам_{formed.strftime('%Y%m%d_%H%M%S')}.txt"
    if os.path.isdir("/content"):
        summary_path = os.path.join("/content", summary_filename)
    else:
        summary_path = os.path.abspath(summary_filename)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    # сессионные переменные (совместимость с модулями 1–7)
    tender_summary_rows = rows
    tender_summary_text = summary_text
    tender_summary_filename = summary_path
    summary_filename = summary_path

    print()
    print(summary_text)

    try:
        from google.colab import files as colab_files
        colab_files.download(summary_path)
        print("📥 Сводная таблица скачана автоматически.")
    except Exception:
        print("💾 Автоскачивание недоступно (не Colab) — файл сохранён локально.")

    print()
    print("=" * 70)
    print("✅ Сводная таблица сформирована!")
    print(f"📁 Файл: {summary_path}")
    print(f"📊 Тендеров: {len(rows)}")
    print(f"⏱ Время: {int((datetime.now() - _T0).total_seconds())} сек.")
    print("=" * 70)
    print("Модуль 8 завершён.")
