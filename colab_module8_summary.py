# =============================================================================
# МОДУЛЬ 8 — Сводная таблица по тендерам с выбором через галочки (Google Colab)
# Выполняйте ПОСЛЕ модулей 4/5 (когда есть Анализ_тендера_*.txt).
#
# - Список тендеров с Checkbox (ipywidgets), fallback — текстовый ввод
# - Сводка печатается ВНУТРИ ячейки
# - Скачивание сводки / ZIP отчётов — по выбранному действию
#
# Автотест / неинтерактивный режим:
#   MODULE8_SELECTION = "all"   # или "1,3"
#   MODULE8_ACTION = 1          # 1=сводка, 2=ZIP, 3=сводка+ZIP
# =============================================================================

from __future__ import annotations

import glob
import io
import os
import re
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

_T0 = datetime.now()
print("=" * 70)
print("🚀 МОДУЛЬ 8 — Сводная таблица (галочки / выбор действия)")
print("=" * 70)

NA = "—"
CONTENT_DIRS: List[str] = []
if os.path.isdir("/content"):
    CONTENT_DIRS.append("/content")
CONTENT_DIRS.append(os.path.abspath("."))

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

# Сессионные результаты
summary_text = ""
summary_filename = ""
zip_filename = ""
tender_summary_rows: List[Dict[str, object]] = []
tender_summary_all_rows: List[Dict[str, object]] = []
tender_summary_text = ""
tender_summary_filename = ""
tender_reports_zip = ""


# =============================================================================
# 1) Поиск / парсинг
# =============================================================================

def find_report_files() -> List[str]:
    found: List[str] = []
    seen = set()
    for root in CONTENT_DIRS:
        for path in glob.glob(os.path.join(root, "Анализ_тендера_*.txt")):
            ap = os.path.abspath(path)
            if ap in seen or not os.path.isfile(ap):
                continue
            if os.path.basename(ap).startswith("Сводная_таблица"):
                continue
            seen.add(ap)
            found.append(ap)
    found.sort(key=lambda p: (os.path.getmtime(p), os.path.basename(p)))
    return found


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
        if any(k.lower() in title.lower() for k in keywords):
            return body
    return ""


def _re_get(text: str, key: str) -> str:
    try:
        m = re.search(patterns[key], text, flags=re.IGNORECASE | re.MULTILINE)
    except re.error:
        return ""
    return m.group(1).strip() if m else ""


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
    return s[:40] if s != NA else NA


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


def _duration_to_seconds(raw: str) -> float:
    s = _clean(raw).lower()
    if not s or s == NA:
        return 0.0
    m = re.match(r"(\d+):(\d{2}):(\d{2})$", s)
    if m:
        return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
    m = re.match(r"(\d+):(\d{2})$", s)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2))
    total = 0.0
    for num, unit in re.findall(r"(\d+(?:[.,]\d+)?)\s*(ч|час|мин|сек|s|m|h)", s):
        v = float(num.replace(",", "."))
        if unit.startswith("ч") or unit == "h":
            total += v * 3600
        elif unit.startswith("мин") or unit == "m":
            total += v * 60
        else:
            total += v
    if total == 0 and re.fullmatch(r"\d+", s):
        total = float(s)
    return total


def _seconds_to_label(secs: float) -> str:
    secs = int(round(secs))
    if secs <= 0:
        return NA
    if secs < 60:
        return f"{secs} сек"
    mins, rem = divmod(secs, 60)
    if mins < 60:
        return f"{mins} мин" if rem == 0 else f"{mins} мин {rem} сек"
    hours, mins = divmod(mins, 60)
    return f"{hours} ч" if mins == 0 else f"{hours} ч {mins} мин"


def _parse_total_time(text: str) -> Tuple[str, float]:
    raw = _re_get(text, "total_time")
    if not raw:
        block = re.search(r"СВОДКА ВРЕМЕНИ ВЫПОЛНЕНИЯ[\s\S]*?(?=ИТОГО:|\Z)", text, re.I)
        if block:
            parts = re.findall(r"(?m)^\d\.\s+[^:\n]+:\s*(.+)$", block.group(0))
            secs = sum(_duration_to_seconds(p) for p in parts)
            if secs > 0:
                return _seconds_to_label(secs), secs
        return NA, 0.0
    secs = _duration_to_seconds(raw)
    return (_seconds_to_label(secs) if secs > 0 else _first_line(raw, 16)), secs


def parse_report(path: str) -> Dict[str, object]:
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        print(f"  ⚠️ Не прочитан {os.path.basename(path)}: {e}")
        return {
            "tender_no": NA, "customer": NA, "subject": NA, "nmck": NA,
            "deadline": NA, "verdict": NA, "date": NA, "time_label": NA,
            "time_sec": 0.0, "path": path, "file": os.path.basename(path),
        }

    qa = _parse_qa_blocks(text)
    tender_raw = _re_get(text, "tender") or _qa_get(qa, 1) or _qa_by_title(qa, "номер тендера", "икз")
    customer = _re_get(text, "customer") or _qa_get(qa, 4) or _qa_by_title(qa, "заказчик")
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
        "date": _parse_date(text, path),
        "time_label": time_label,
        "time_sec": float(time_sec),
        "path": path,
        "file": os.path.basename(path),
    }


# =============================================================================
# 2) Форматирование / файлы
# =============================================================================

def _pad(s: object, width: int) -> str:
    s = NA if s is None else str(s)
    if len(s) > width:
        return s[: width - 1] + "…"
    return s.ljust(width)


def _out_dir() -> str:
    return "/content" if os.path.isdir("/content") else os.path.abspath(".")


def _manual_download_link(path: str) -> str:
    """HTML-ссылка для ручного скачивания в Colab (если доступно)."""
    try:
        import base64
        data = Path(path).read_bytes()
        b64 = base64.b64encode(data).decode("ascii")
        name = os.path.basename(path)
        href = f"data:application/octet-stream;base64,{b64}"
        return f'<a download="{name}" href="{href}" target="_blank">📥 Скачать {name}</a>'
    except Exception:
        return f"📥 Скачать: {path}"


def _try_colab_download(path: str) -> bool:
    try:
        from google.colab import files as colab_files  # type: ignore
        colab_files.download(path)
        return True
    except Exception:
        return False


def format_summary(rows: Sequence[Dict[str, object]], formed_at: datetime) -> str:
    lines: List[str] = []
    lines.append("=" * 70)
    lines.append("📊 СВОДНАЯ ТАБЛИЦА ПО ТЕНДЕРАМ")
    lines.append("=" * 70)
    lines.append(f"Дата формирования: {formed_at.strftime('%d.%m.%Y %H:%M')}")
    lines.append(f"Выбрано тендеров: {len(rows)}")
    lines.append("")

    headers = (
        ("№", 3),
        ("Номер тендера", 18),
        ("Заказчик", 22),
        ("НМЦК", 8),
        ("Срок подачи", 12),
        ("Вердикт", 12),
        ("Время", 14),
    )
    lines.append("\t".join(_pad(h, w) for h, w in headers))
    lines.append("-" * 100)

    for i, r in enumerate(rows, start=1):
        vals = (
            (str(i), 3),
            (r["tender_no"], 18),
            (r["customer"], 22),
            (r["nmck"], 8),
            (r["deadline"], 12),
            (r["verdict"], 12),
            (r["time_label"], 14),
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
    lines.append("=" * 70)
    return "\n".join(lines)


def save_summary(rows: Sequence[Dict[str, object]]) -> Tuple[str, str]:
    formed = datetime.now()
    text = format_summary(rows, formed)
    name = f"Сводная_таблица_по_тендерам_{formed.strftime('%Y%m%d_%H%M%S')}.txt"
    path = os.path.join(_out_dir(), name)
    try:
        Path(path).write_text(text, encoding="utf-8")
    except OSError:
        path = os.path.abspath(name)
        Path(path).write_text(text, encoding="utf-8")
    return text, path


def make_reports_zip(rows: Sequence[Dict[str, object]]) -> str:
    formed = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"Отчёты_тендеров_{formed}.zip"
    path = os.path.join(_out_dir(), name)
    try:
        zf_path = path
        with zipfile.ZipFile(zf_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for r in rows:
                src = str(r["path"])
                if os.path.isfile(src):
                    zf.write(src, arcname=os.path.basename(src))
        return zf_path
    except OSError:
        zf_path = os.path.abspath(name)
        with zipfile.ZipFile(zf_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for r in rows:
                src = str(r["path"])
                if os.path.isfile(src):
                    zf.write(src, arcname=os.path.basename(src))
        return zf_path


def run_action(rows: Sequence[Dict[str, object]], action: int) -> None:
    """
    action:
      1 — сформировать сводку (показать в ячейке, сохранить TXT)
      2 — скачать ZIP выбранных отчётов
      3 — сводка + ZIP
    """
    global summary_text, summary_filename, zip_filename
    global tender_summary_rows, tender_summary_text, tender_summary_filename, tender_reports_zip

    if not rows:
        print("⚠️ Ничего не выбрано. Отметьте хотя бы один тендер.")
        return

    tender_summary_rows = list(rows)
    do_summary = action in (1, 3)
    do_zip = action in (2, 3)

    if do_summary:
        summary_text, summary_path = save_summary(rows)
        summary_filename = summary_path
        tender_summary_text = summary_text
        tender_summary_filename = summary_path
        # ОБЯЗАТЕЛЬНО внутри ячейки
        print()
        print(summary_text)
        print()
        print(f"📥 Скачать сводку: {summary_path}")
        try:
            from IPython.display import HTML, display  # type: ignore
            display(HTML(_manual_download_link(summary_path)))
        except Exception:
            pass
        # автоскачивание только если явно просили ZIP+сводку/скачивание —
        # для action=1 не форсим files.download (по желанию пользователя)
        if action == 3:
            _try_colab_download(summary_path)

    if do_zip:
        zip_path = make_reports_zip(rows)
        zip_filename = zip_path
        tender_reports_zip = zip_path
        print()
        print(f"📁 Скачать отчёты (ZIP): {zip_path}")
        print(f"   Файлов в архиве: {len(rows)}")
        try:
            from IPython.display import HTML, display  # type: ignore
            display(HTML(_manual_download_link(zip_path)))
        except Exception:
            pass
        _try_colab_download(zip_path)

    print()
    print("=" * 70)
    print("✅ Готово.")
    print(f"📊 Выбрано тендеров: {len(rows)}")
    if do_summary:
        print(f"📄 Сводка: {summary_filename}")
    if do_zip:
        print(f"🗜 ZIP: {zip_filename}")
    print("=" * 70)


# =============================================================================
# 3) Выбор: ipywidgets или текстовый fallback
# =============================================================================

def parse_selection(raw: str, n: int) -> Optional[List[int]]:
    s = (raw or "").strip().lower()
    if not s:
        return None
    if s in {"all", "все", "*"}:
        return list(range(1, n + 1))
    idxs: List[int] = []
    for p in re.split(r"[,\s;]+", s):
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


def parse_action(raw: object) -> Optional[int]:
    s = str(raw or "").strip()
    if s in {"1", "2", "3"}:
        return int(s)
    return None


def _ipywidgets_available() -> bool:
    try:
        import ipywidgets  # noqa: F401
        from IPython.display import display  # noqa: F401
        return True
    except Exception:
        return False


def _preset_selection_and_action(n: int) -> Optional[Tuple[List[int], int]]:
    """Неинтерактивный режим через MODULE8_SELECTION / MODULE8_ACTION."""
    sel = globals().get("MODULE8_SELECTION")
    act = globals().get("MODULE8_ACTION", 1)
    if sel is None or not str(sel).strip():
        return None
    idxs = parse_selection(str(sel), n)
    action = parse_action(act)
    if idxs is None or action is None:
        print(f"⚠️ Некорректный пресет MODULE8_SELECTION={sel!r} / MODULE8_ACTION={act!r}")
        return None
    print(f"📌 Пресет: тендеры {idxs}, действие {action}")
    return idxs, action


def ui_with_widgets(all_rows: List[Dict[str, object]]) -> None:
    import ipywidgets as widgets
    from IPython.display import clear_output, display

    print()
    print("📋 ВЫБЕРИТЕ ТЕНДЕРЫ ДЛЯ СВОДКИ:")
    print("=" * 70)

    checks: List[widgets.Checkbox] = []
    for i, r in enumerate(all_rows, start=1):
        label = f"{i}. {r['tender_no']} — {r['customer']} ({r['date']})"
        cb = widgets.Checkbox(value=True, description=label, indent=False, layout=widgets.Layout(width="100%"))
        cb.style.description_width = "initial"
        checks.append(cb)

    select_all = widgets.Checkbox(value=True, description="Выбрать все", indent=False)

    def _on_select_all(change):
        if change["name"] == "value":
            for cb in checks:
                cb.value = bool(change["new"])

    select_all.observe(_on_select_all, names="value")

    action = widgets.RadioButtons(
        options=[
            ("1. Сформировать сводку по выбранным", 1),
            ("2. Скачать все отчёты выбранных тендеров (ZIP)", 2),
            ("3. Сформировать сводку + скачать все отчёты", 3),
        ],
        value=1,
        description="Действие:",
        layout=widgets.Layout(width="100%"),
        style={"description_width": "70px"},
    )
    btn = widgets.Button(description="Выполнить", button_style="primary", icon="check")
    out = widgets.Output()

    def _on_click(_btn):
        with out:
            clear_output(wait=True)
            selected = [all_rows[i] for i, cb in enumerate(checks) if cb.value]
            if not selected:
                print("⚠️ Отметьте хотя бы один тендер галочкой.")
                return
            print(f"✅ Выбрано: {len(selected)} → {[r['tender_no'] for r in selected]}")
            run_action(selected, int(action.value))

    btn.on_click(_on_click)

    box = widgets.VBox(
        [
            widgets.HTML("<b>📋 ВЫБЕРИТЕ ТЕНДЕРЫ ДЛЯ СВОДКИ</b>"),
            select_all,
            widgets.VBox(checks),
            widgets.HTML("<hr><b>Выберите действие:</b>"),
            action,
            btn,
            out,
        ]
    )
    display(box)
    print("(отметьте галочки и нажмите «Выполнить»)")


def ui_text_fallback(all_rows: List[Dict[str, object]]) -> None:
    print()
    print("📋 ВЫБЕРИТЕ ТЕНДЕРЫ ДЛЯ СВОДКИ:")
    print("=" * 70)
    print("(ipywidgets недоступен — текстовый режим)")
    print()
    for i, r in enumerate(all_rows, start=1):
        mark = "[x]"  # подсказка: по умолчанию можно выбрать all
        print(f"{mark} {i}. {r['tender_no']} — {r['customer']} ({r['date']})")
    print("=" * 70)
    print("Выберите действие:")
    print("  1 — Сформировать сводку по выбранным")
    print("  2 — Скачать все отчёты выбранных тендеров (ZIP)")
    print("  3 — Сформировать сводку + скачать все отчёты")
    print()

    n = len(all_rows)
    while True:
        try:
            sel_raw = input("📌 Номера тендеров через запятую (или 'all'): ")
        except EOFError:
            sel_raw = "all"
            print("⚠️ Ввод недоступен — all")
        idxs = parse_selection(sel_raw, n)
        if idxs is not None:
            break
        print(f"❌ Неверный ввод. Примеры: all | 1 | 1,3  (1–{n})")

    while True:
        try:
            act_raw = input("📌 Номер действия (1/2/3): ")
        except EOFError:
            act_raw = "1"
            print("⚠️ Ввод недоступен — действие 1")
        action = parse_action(act_raw)
        if action is not None:
            break
        print("❌ Введите 1, 2 или 3.")

    selected = [all_rows[i - 1] for i in idxs]
    print(f"✅ Выбрано: {len(selected)} → {[r['tender_no'] for r in selected]}")
    run_action(selected, action)


# =============================================================================
# 4) Запуск
# =============================================================================

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

    preset = _preset_selection_and_action(len(all_rows))
    if preset is not None:
        idxs, action = preset
        selected = [all_rows[i - 1] for i in idxs]
        run_action(selected, action)
    elif _ipywidgets_available():
        try:
            ui_with_widgets(all_rows)
        except Exception as e:
            print(f"⚠️ ipywidgets UI не удалось показать ({e}) — текстовый режим.")
            ui_text_fallback(all_rows)
    else:
        ui_text_fallback(all_rows)

print(f"⏱ Модуль 8: {int((datetime.now() - _T0).total_seconds())} сек.")
