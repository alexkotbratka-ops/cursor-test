# =============================================================================
# МОДУЛЬ 7 — PDF-отчёт по тендеру (Google Colab)
# Выполняйте ПОСЛЕ модуля 4 или модуля 5.
#
# Вход:
#   - ответы на 86 вопросов из сессии (answers / tender_answers)
#   - QUESTIONS, номер тендера, TXT-отчёт (если есть)
# Выход:
#   - Отчёт_по_тендеру_[НОМЕР]_[ДАТА].pdf (строгий ч/б стиль)
# =============================================================================

import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_T0 = time.time()
print("=" * 70)
print("🚀 МОДУЛЬ 7 — Формирование PDF-отчёта по тендеру")
print("=" * 70)

# =============================================================================
# 1) Зависимости
# =============================================================================

print("📦 Установка reportlab и шрифтов…")


def _run(cmd: List[str]) -> None:
    subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _font_present() -> bool:
    candidates = [
        "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/times.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf",
    ]
    return any(os.path.isfile(p) for p in candidates)


if not _font_present():
    _run(["apt-get", "update", "-qq"])
    # Liberation Serif — метрический аналог Times New Roman (открытый)
    _run(["apt-get", "install", "-y", "-qq", "fonts-liberation", "fonts-liberation2"])
    # Попытка поставить настоящий Times New Roman (может требовать EULA — не критично)
    _run(["apt-get", "install", "-y", "-qq", "ttf-mscorefonts-installer"])

try:
    import reportlab  # noqa: F401
except ImportError:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "reportlab"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

print("✅ Зависимости готовы.\n")

# =============================================================================
# 2) Шрифты: Times New Roman (или Liberation Serif как замена)
# =============================================================================

COMPANY_NAME = 'ООО «Пожарная Автоматика»'
FONT_REG = "TimesNewRoman"
FONT_BOLD = "TimesNewRoman-Bold"


def _register_times_fonts() -> Tuple[str, str]:
    """Регистрирует Times New Roman или Liberation Serif под теми же именами."""
    candidates_reg = [
        "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/times.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf",
    ]
    candidates_bold = [
        "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Bold.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/timesbd.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSerif-Bold.ttf",
    ]
    reg_path = next((p for p in candidates_reg if os.path.isfile(p)), None)
    bold_path = next((p for p in candidates_bold if os.path.isfile(p)), None)
    if not reg_path or not bold_path:
        raise RuntimeError(
            "❌ Не найдены TTF-шрифты Times/Liberation Serif. "
            "Установите fonts-liberation в Colab и перезапустите модуль 7."
        )
    pdfmetrics.registerFont(TTFont(FONT_REG, reg_path))
    pdfmetrics.registerFont(TTFont(FONT_BOLD, bold_path))
    src = "Times New Roman" if "msttcore" in reg_path or "times" in reg_path.lower() else "Liberation Serif → Times New Roman"
    print(f"🖋 Шрифт: {src}")
    print(f"   regular: {reg_path}")
    print(f"   bold:    {bold_path}")
    return FONT_REG, FONT_BOLD


_register_times_fonts()

# =============================================================================
# 3) Данные из сессии / TXT
# =============================================================================

NA = "Не указано"


def _na(val: Any) -> str:
    s = str(val or "").strip()
    if not s or s.lower() in {"none", "null", "nan", "-", "—"}:
        return NA
    # унификация пустых/служебных ответов
    low = s.lower()
    if low in {"n/a", "na", "не применимо", "неприменимо", "не задано"}:
        return "Не применимо" if "примен" in low else NA
    return s


def _get_answers() -> Dict[int, str]:
    """answers / tender_answers из модулей 4 или 5."""
    g = globals()
    raw = g.get("tender_answers")
    if not isinstance(raw, dict):
        raw = g.get("answers")
    if not isinstance(raw, dict):
        return {}
    out: Dict[int, str] = {}
    for k, v in raw.items():
        try:
            out[int(k)] = _na(v)
        except Exception:
            continue
    return out


def _get_questions() -> List[Dict[str, Any]]:
    qs = globals().get("QUESTIONS")
    if isinstance(qs, list) and qs:
        return list(qs)
    # Минимальный каркас, если QUESTIONS нет в сессии
    sections = [
        (range(1, 12), "1. ОБЩАЯ ИНФОРМАЦИЯ"),
        (range(12, 22), "2. ТРЕБОВАНИЯ К УЧАСТНИКАМ"),
        (range(22, 38), "3. ТЕХНИЧЕСКИЕ ТРЕБОВАНИЯ"),
        (range(38, 54), "4. УСЛОВИЯ КОНТРАКТА"),
        (range(54, 67), "5. ФИНАНСОВЫЕ УСЛОВИЯ"),
        (range(67, 79), "6. РИСКИ И РЕКОМЕНДАЦИИ"),
        (range(79, 87), "7. СТАТИСТИКА И ПОЛНОТА АНАЛИЗА"),
    ]
    titles = {
        1: "Номер тендера / ИКЗ",
        2: "Полное название объекта закупки",
        4: "Заказчик (название)",
        5: "ИНН заказчика",
        6: "Контакты заказчика",
        11: "Вид процедуры",
        41: "Срок подачи заявок",
        42: "Время окончания подачи заявок",
        54: "НМЦК",
        74: "Рекомендация по участию",
        75: "Ключевые выводы",
        77: "Оценка экономической целесообразности",
        78: "Рекомендация по цене предложения",
        86: "Общий вывод по полноте анализа",
    }
    out = []
    for rng, sec in sections:
        for n in rng:
            out.append({
                "num": n,
                "section": sec,
                "title": titles.get(n, f"Вопрос {n}"),
                "question": "",
            })
    return out


def _extract_short_tender_no(raw: str) -> str:
    """Достаёт короткий номер (B… / ИКЗ) из длинного ответа на вопрос 1."""
    s = (raw or "").strip()
    if not s or s == NA:
        return ""
    if len(s) < 50:
        return s
    m = re.search(r"(B\d{10,})", s, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b(\d{18,36})\b", s)
    if m:
        return m.group(1)
    return ""


def _get_tender_no(ans: Dict[int, str]) -> str:
    g = globals()
    for key in ("tender_number", "tender_no"):
        v = g.get(key)
        if v and str(v).strip():
            short = _extract_short_tender_no(str(v).strip())
            return short or str(v).strip()[:40]
    raw = str(ans.get(1, "") or "").strip()
    short = _extract_short_tender_no(raw)
    if short:
        return short
    return _na(raw[:40] if raw else "")


def build_filename(tender_num: str = None) -> str:
    """Формирует безопасное имя файла (макс. 100 символов)."""
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    if tender_num and tender_num != NA and len(tender_num) < 50:
        # Очищаем номер тендера от недопустимых символов
        safe_num = re.sub(r"[^\w\-]+", "_", tender_num)[:30]
        return f"Отчёт_по_тендеру_{safe_num}_{date_str}.pdf"
    return f"Отчёт_по_тендеру_{date_str}.pdf"


def _find_txt_report() -> Optional[str]:
    """Ищет TXT-отчёт модуля 4/5 в сессии или в /content."""
    for key in ("tender_report_filename", "report_filename"):
        p = globals().get(key)
        if p and os.path.isfile(str(p)):
            return str(p)
    # glob
    roots = []
    if os.path.isdir("/content"):
        roots.append(Path("/content"))
    roots.append(Path("."))
    found: List[Path] = []
    for root in roots:
        found.extend(root.glob("Анализ_тендера_*.txt"))
        found.extend(root.glob("**/Анализ_тендера_*.txt"))
    if not found:
        return None
    found.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(found[0])


def _parse_answers_from_txt(path: str) -> Tuple[Dict[int, str], Dict[int, str]]:
    """Разбор TXT-отчёта: «N. Заголовок» + текст ответа до следующего вопроса.

    Возвращает (answers, titles).
    """
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        print(f"  ⚠️ Не удалось прочитать TXT: {e}")
        return {}, {}
    # Блоки вида: "12. Название\n----\nответ"
    # Важно: заголовок без перевода строки (иначе «2. ТРЕБОВАНИЯ…\\n====\\n\\n12. …\\n----»
    # ошибочно съедает вопрос 12). Следующий вопрос — только с подчёркиванием ----.
    pattern = re.compile(
        r"(?m)^(\d{1,2})\.\s+([^\n]+)\n-{3,}\n(.*?)(?=\n\d{1,2}\.\s+[^\n]+\n-{3,}|\n={3,}|\Z)",
        re.S,
    )
    out: Dict[int, str] = {}
    titles: Dict[int, str] = {}
    for m in pattern.finditer(text):
        num = int(m.group(1))
        if num < 1 or num > 86:
            continue
        titles[num] = m.group(2).strip()
        body = m.group(3).strip()
        # убрать строку источников
        body = re.sub(r"(?m)^Источники:.*$", "", body).strip()
        out[num] = _na(body)
    return out, titles


def _get_report_date() -> datetime:
    ad = globals().get("ANALYSIS_DATE")
    if isinstance(ad, datetime):
        return ad
    return datetime.now()


ANSWERS = _get_answers()
TXT_TITLES: Dict[int, str] = {}
TXT_PATH = _find_txt_report()
if TXT_PATH:
    print(f"📄 TXT-отчёт: {TXT_PATH}")
    parsed, TXT_TITLES = _parse_answers_from_txt(TXT_PATH)
    # TXT дополняет пустые ответы сессии
    for k, v in parsed.items():
        if k not in ANSWERS or ANSWERS[k] == NA:
            ANSWERS[k] = v
else:
    print("📄 TXT-отчёт не найден — используем данные сессии.")

if not ANSWERS:
    raise RuntimeError(
        "❌ Нет данных для PDF. Сначала выполните модуль 4 или модуль 5 "
        "(нужны answers / tender_answers или файл Анализ_тендера_*.txt)."
    )

QUESTIONS_M7 = _get_questions()
# Подтянуть названия вопросов из TXT, если в сессии нет полного QUESTIONS
if TXT_TITLES:
    by_num = {int(q["num"]): q for q in QUESTIONS_M7 if "num" in q}
    for n, title in TXT_TITLES.items():
        if n in by_num:
            cur = str(by_num[n].get("title", ""))
            if (not cur) or cur.startswith("Вопрос "):
                by_num[n]["title"] = title
TENDER_NO = _get_tender_no(ANSWERS)
REPORT_DATE = _get_report_date()
print(f"🔖 Номер тендера: {TENDER_NO}")
print(f"📊 Ответов: {len(ANSWERS)}")

# =============================================================================
# 4) Вердикт
# =============================================================================


# Метки вердикта: цветные круги (emoji) для консоли Colab;
# в PDF Times/Liberation emoji нет — рисуем ● + цветную рамку.
_VERDICT_GO = ("🟢 Участвовать", "● Участвовать", "GO", colors.HexColor("#006400"))
_VERDICT_REVIEW = ("🟡 Рассмотреть", "● Рассмотреть", "REVIEW", colors.HexColor("#8B6914"))
_VERDICT_SKIP = ("🔴 Пропустить", "● Пропустить", "SKIP", colors.HexColor("#8B0000"))


def _classify_verdict(text: str) -> Tuple[str, str, str, colors.Color]:
    """
    Возвращает (метка_emoji, метка_pdf, код, цвет_рамки).
    🟢 Участвовать / 🟡 Рассмотреть / 🔴 Пропустить
    """
    t = (text or "").lower()
    # негатив раньше позитива (иначе «не участвовать» попадёт в «участвовать»)
    skip_kw = (
        "не участвовать", "не рекоменду", "пропустить", "отказаться",
        "нецелесообраз", "высокий риск без", "не стоит",
    )
    go_kw = (
        "участвовать", "рекомендуется участие", "рекомендую участвовать",
        "целесообразно участвовать", "можно участвовать", "к участию",
    )
    mid_kw = (
        "осторож", "рассмотр", "условн", "при уточнении", "после проверки",
        "с оговорк", "требует уточн", "требует дополнительн",
    )
    has_skip = any(k in t for k in skip_kw)
    has_go = any(k in t for k in go_kw)
    has_mid = any(k in t for k in mid_kw)
    if has_skip:
        return _VERDICT_SKIP
    # «участвовать с осторожностью» → рассмотреть
    if has_go and has_mid:
        return _VERDICT_REVIEW
    if has_go:
        return _VERDICT_GO
    if has_mid:
        return _VERDICT_REVIEW
    # эвристика по экономической оценке
    eco = _na(ANSWERS.get(77, "")).lower()
    if "нецелесообраз" in eco or "не рекомендуется" in eco:
        return _VERDICT_SKIP
    if "осторож" in eco or "риск" in eco:
        return _VERDICT_REVIEW
    if text and text != NA:
        return _VERDICT_REVIEW
    return _VERDICT_REVIEW


VERDICT_TEXT = _na(ANSWERS.get(74, ""))
VERDICT_LABEL, VERDICT_LABEL_PDF, VERDICT_CODE, VERDICT_COLOR = _classify_verdict(VERDICT_TEXT)

# =============================================================================
# 5) Стили
# =============================================================================

styles = getSampleStyleSheet()

style_title = ParagraphStyle(
    "M7Title",
    fontName=FONT_BOLD,
    fontSize=16,
    leading=20,
    alignment=TA_CENTER,
    spaceAfter=8,
    textColor=colors.black,
)
style_company = ParagraphStyle(
    "M7Company",
    fontName=FONT_BOLD,
    fontSize=11,
    leading=14,
    alignment=TA_CENTER,
    spaceAfter=2,
    textColor=colors.black,
)
style_meta = ParagraphStyle(
    "M7Meta",
    fontName=FONT_REG,
    fontSize=11,
    leading=14,
    alignment=TA_CENTER,
    spaceAfter=2,
    textColor=colors.black,
)
style_h1 = ParagraphStyle(
    "M7H1",
    fontName=FONT_BOLD,
    fontSize=12,
    leading=15,
    alignment=TA_LEFT,
    spaceBefore=12,
    spaceAfter=8,
    textColor=colors.black,
)
style_q_title = ParagraphStyle(
    "M7QTitle",
    fontName=FONT_BOLD,
    fontSize=11,
    leading=14,
    alignment=TA_LEFT,
    spaceBefore=8,
    spaceAfter=2,
    textColor=colors.black,
)
style_body = ParagraphStyle(
    "M7Body",
    fontName=FONT_REG,
    fontSize=11,
    leading=14,
    alignment=TA_JUSTIFY,
    spaceAfter=4,
    textColor=colors.black,
)
style_cell = ParagraphStyle(
    "M7Cell",
    fontName=FONT_REG,
    fontSize=11,
    leading=13,
    alignment=TA_LEFT,
    textColor=colors.black,
)
style_cell_bold = ParagraphStyle(
    "M7CellBold",
    fontName=FONT_BOLD,
    fontSize=11,
    leading=13,
    alignment=TA_LEFT,
    textColor=colors.black,
)
style_verdict = ParagraphStyle(
    "M7Verdict",
    fontName=FONT_BOLD,
    fontSize=12,
    leading=15,
    alignment=TA_CENTER,
    textColor=colors.black,
)
style_footer = ParagraphStyle(
    "M7Footer",
    fontName=FONT_REG,
    fontSize=9,
    leading=11,
    alignment=TA_CENTER,
    textColor=colors.black,
)
style_label = ParagraphStyle(
    "M7Label",
    fontName=FONT_BOLD,
    fontSize=11,
    leading=14,
    alignment=TA_LEFT,
    spaceBefore=6,
    spaceAfter=1,
    textColor=colors.black,
)
style_answer = ParagraphStyle(
    "M7Answer",
    fontName=FONT_REG,
    fontSize=11,
    leading=14,
    alignment=TA_JUSTIFY,
    leftIndent=8,
    spaceAfter=6,
    textColor=colors.black,
)

# Ячейка Table не умеет разрываться между страницами.
# Порог: ~0.7 страницы текста при leading=13–14 (~45 строк × ~90 символов).
_MAX_TABLE_CHARS = 2800
_CHUNK_CHARS = 1800  # куски для сверхдлинных ответов (параграфы)


def _escape_xml(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br/>")
    )


def _p(text: Any, style: ParagraphStyle = style_body) -> Paragraph:
    return Paragraph(_escape_xml(_na(text)), style)


def _split_long_text(text: str, max_chars: int = _CHUNK_CHARS) -> List[str]:
    """Режет длинный текст по абзацам/предложениям, чтобы flowable помещался на страницу."""
    s = _na(text)
    if len(s) <= max_chars:
        return [s]
    parts: List[str] = []
    # сначала по пустым строкам
    blocks = re.split(r"\n\s*\n", s)
    buf = ""
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        if len(buf) + len(block) + 2 <= max_chars:
            buf = f"{buf}\n\n{block}" if buf else block
            continue
        if buf:
            parts.append(buf)
            buf = ""
        if len(block) <= max_chars:
            buf = block
            continue
        # жёсткая нарезка длинного блока
        start = 0
        while start < len(block):
            end = min(start + max_chars, len(block))
            if end < len(block):
                # откат к пробелу/переносу
                cut = max(block.rfind("\n", start, end), block.rfind(" ", start, end))
                if cut > start + max_chars // 3:
                    end = cut
            parts.append(block[start:end].strip())
            start = end
        buf = ""
    if buf:
        parts.append(buf)
    return [p for p in parts if p] or [NA]


def _append_paragraph_answer(story: list, title: str, answer: str) -> None:
    """Вопрос/ответ параграфами — текст свободно переносится на следующую страницу."""
    q = Paragraph(_escape_xml(f"Вопрос: {title}"), style_label)
    chunks = _split_long_text(answer)
    if len(chunks) == 1 and len(chunks[0]) <= 900:
        # короткий блок держим вместе; длинный — по кускам (без LayoutError)
        story.append(KeepTogether([q, Paragraph(_escape_xml(chunks[0]), style_answer)]))
        return
    story.append(q)
    for chunk in chunks:
        story.append(Paragraph(_escape_xml(chunk), style_answer))


def _one_row_table(key: str, value: str, col_widths: Optional[Sequence[float]] = None) -> Table:
    """Одна строка таблицы — может переехать на следующую страницу целиком."""
    w = list(col_widths) if col_widths else [55 * mm, 125 * mm]
    t = Table([[_p(key, style_cell_bold), _p(value, style_cell)]], colWidths=w, hAlign="LEFT")
    t.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), FONT_REG),
                ("FONTSIZE", (0, 0), (-1, -1), 11),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.6, colors.black),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("BACKGROUND", (0, 0), (0, 0), colors.Color(0.95, 0.95, 0.95)),
            ]
        )
    )
    return t


def _kv_table(rows: List[Tuple[str, str]], col_widths: Optional[Sequence[float]] = None) -> list:
    """
    Сводная/короткая карточка: каждая строка — отдельная Table.
    Длинные значения (>_MAX_TABLE_CHARS) выводятся параграфами (без LayoutError).
    """
    flowables: list = []
    for k, v in rows:
        val = _na(v)
        if len(val) > _MAX_TABLE_CHARS:
            _append_paragraph_answer(flowables, k, val)
        else:
            flowables.append(_one_row_table(k, val, col_widths))
    return flowables


def _section_as_table(story: list, q_from: int, q_to: int, title: str) -> None:
    """Разделы 1–5: таблицы вопрос/ответ (по одной строке на вопрос)."""
    story.append(Paragraph(title, style_h1))
    by_num = {int(q["num"]): q for q in QUESTIONS_M7 if "num" in q}
    for n in range(q_from, q_to + 1):
        q = by_num.get(n, {"num": n, "title": f"Вопрос {n}"})
        q_title = f"{n}. {_na(q.get('title', f'Вопрос {n}'))}"
        ans = _na(ANSWERS.get(n, NA))
        if len(ans) > _MAX_TABLE_CHARS:
            # длинный ответ — параграфы, иначе LayoutError
            _append_paragraph_answer(story, q_title, ans)
        else:
            story.append(_one_row_table(q_title, ans))
            story.append(Spacer(1, 2 * mm))


def _section_as_paragraphs(story: list, q_from: int, q_to: int, title: str) -> None:
    """Разделы 6–7 и длинные тексты: параграфы с переносом страниц."""
    story.append(Paragraph(title, style_h1))
    by_num = {int(q["num"]): q for q in QUESTIONS_M7 if "num" in q}
    for n in range(q_from, q_to + 1):
        q = by_num.get(n, {"num": n, "title": f"Вопрос {n}"})
        q_title = f"{n}. {_na(q.get('title', f'Вопрос {n}'))}"
        _append_paragraph_answer(story, q_title, _na(ANSWERS.get(n, NA)))


# =============================================================================
# 6) Сборка PDF
# =============================================================================

PDF_NAME = build_filename(None if (not TENDER_NO or TENDER_NO == NA) else TENDER_NO)
if len(PDF_NAME) > 100:
    # страховка: даже при странном номере не превышаем лимит имени
    PDF_NAME = build_filename(None)
if os.path.isdir("/content"):
    PDF_PATH = os.path.join("/content", PDF_NAME)
else:
    PDF_PATH = os.path.abspath(PDF_NAME)

print(f"📝 Формирование PDF: {PDF_PATH}")


def _add_page_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont(FONT_REG, 9)
    canvas.setFillColor(colors.black)
    y = 12 * mm
    canvas.drawCentredString(
        A4[0] / 2,
        y,
        f"{COMPANY_NAME}  |  дата формирования: {REPORT_DATE.strftime('%d.%m.%Y %H:%M')}  |  стр. {doc.page}",
    )
    canvas.setStrokeColor(colors.black)
    canvas.setLineWidth(0.5)
    canvas.line(20 * mm, y + 5 * mm, A4[0] - 20 * mm, y + 5 * mm)
    canvas.restoreState()


doc = SimpleDocTemplate(
    PDF_PATH,
    pagesize=A4,
    leftMargin=20 * mm,
    rightMargin=20 * mm,
    topMargin=18 * mm,
    bottomMargin=20 * mm,
    title=f"Отчёт по тендеру {TENDER_NO}",
    author=COMPANY_NAME,
)

story: list = []

# --- Заголовок и шапка ---
story.append(Paragraph("ОТЧЁТ ПО ТЕНДЕРУ", style_title))
story.append(Paragraph(COMPANY_NAME, style_company))
story.append(Paragraph(f"Дата анализа: {REPORT_DATE.strftime('%d.%m.%Y %H:%M')}", style_meta))
story.append(Paragraph(f"Номер тендера / ИКЗ: {_na(TENDER_NO)}", style_meta))
story.append(Spacer(1, 8 * mm))

# --- Сводная карточка ---
story.append(Paragraph("СВОДНАЯ КАРТОЧКА ТЕНДЕРА", style_h1))

# Площадка / ссылки — ищем по ключевым словам в ответах 1–20, иначе Н/Д
def _find_answer_by_title_keywords(keywords: Sequence[str]) -> str:
    for q in QUESTIONS_M7:
        title = str(q.get("title", "")).lower()
        if any(k in title for k in keywords):
            return _na(ANSWERS.get(int(q["num"]), NA))
    return NA


platform = _find_answer_by_title_keywords(("площадк", "сайт", "еис", "этп", "электронн"))
links = _find_answer_by_title_keywords(("ссылк", "url", "адрес извещ"))
# Срок подачи — вопросы 41–42 (эталон модулей 1/4/5)
deadline_date = _na(ANSWERS.get(41, NA))
deadline_time = _na(ANSWERS.get(42, NA))
if deadline_date != NA and deadline_time != NA:
    deadline = f"{deadline_date}, {deadline_time}"
elif deadline_date != NA:
    deadline = deadline_date
else:
    deadline = _find_answer_by_title_keywords(
        ("срок подачи", "окончания подачи", "дата подачи", "приёма заявок", "приема заявок")
    )

summary_rows = [
    ("Номер тендера / ИКЗ", _na(TENDER_NO)),
    ("Заказчик", _na(ANSWERS.get(4, NA))),
    ("ИНН заказчика", _na(ANSWERS.get(5, NA))),
    ("Контакты заказчика", _na(ANSWERS.get(6, NA))),
    ("Предмет закупки", _na(ANSWERS.get(2, NA))),
    ("НМЦК", _na(ANSWERS.get(54, NA))),
    ("Срок подачи заявки", deadline if deadline != NA else "Не указано"),
    ("Способ отбора", _na(ANSWERS.get(11, NA))),
    ("Площадка", platform if platform != NA else "Не указано"),
    ("Ссылки", links if links != NA else "Не указано"),
    ("Вердикт", VERDICT_LABEL_PDF),
]
story.extend(_kv_table(summary_rows))
story.append(Spacer(1, 4 * mm))

# Рамка вердикта: заголовок в таблице; длинный текст — параграфами (без LayoutError)
verdict_hdr = Table(
    [[Paragraph(f"ИТОГОВЫЙ ВЕРДИКТ: {VERDICT_LABEL_PDF}", style_verdict)]],
    colWidths=[180 * mm],
)
verdict_hdr.setStyle(
    TableStyle(
        [
            ("BOX", (0, 0), (-1, -1), 1.5, VERDICT_COLOR),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("BACKGROUND", (0, 0), (-1, -1), colors.Color(0.93, 0.93, 0.93)),
        ]
    )
)
story.append(verdict_hdr)
_vtxt = VERDICT_TEXT if VERDICT_TEXT != NA else "Рекомендация по участию не сформирована."
for chunk in _split_long_text(_vtxt):
    story.append(_p(chunk, style_body))
story.append(Spacer(1, 6 * mm))

# --- Разделы 1–5: таблицы; 6–7: параграфы (длинные synthesize-ответы) ---
_section_as_table(story, 1, 11, "1. ОБЩАЯ ИНФОРМАЦИЯ")
_section_as_table(story, 12, 21, "2. ТРЕБОВАНИЯ К УЧАСТНИКАМ")
_section_as_table(story, 22, 37, "3. ТЕХНИЧЕСКИЕ ТРЕБОВАНИЯ")
_section_as_table(story, 38, 53, "4. УСЛОВИЯ КОНТРАКТА")
_section_as_table(story, 54, 66, "5. ФИНАНСОВЫЕ УСЛОВИЯ")
_section_as_paragraphs(story, 67, 78, "6. РИСКИ И РЕКОМЕНДАЦИИ")
_section_as_paragraphs(story, 79, 86, "7. СТАТИСТИКА И ПОЛНОТА АНАЛИЗА")

# --- Итоговый вердикт (развёрнутый) — только параграфы ---
story.append(Paragraph("ИТОГОВЫЙ ВЕРДИКТ", style_h1))
for label, val in (
    ("Рекомендация", VERDICT_LABEL_PDF),
    ("Обоснование", VERDICT_TEXT),
    ("Ключевые выводы", _na(ANSWERS.get(75, NA))),
    ("Экономическая целесообразность", _na(ANSWERS.get(77, NA))),
    ("Рекомендация по цене", _na(ANSWERS.get(78, NA))),
    ("Полнота анализа", _na(ANSWERS.get(86, NA))),
):
    _append_paragraph_answer(story, label, val)
story.append(Spacer(1, 4 * mm))
story.append(_p(
    "Рекомендации по участию: руководствуйтесь разделом 6 (риски) и вердиктом выше. "
    "При статусе «Рассмотреть» уточните недостающие данные до подачи заявки. "
    "При статусе «Пропустить» участие не рекомендуется без изменения условий закупки.",
    style_body,
))

# --- Футер на последней странице (дублирует колонтитул) ---
story.append(Spacer(1, 10 * mm))
story.append(Paragraph("—" * 40, style_meta))
story.append(Paragraph(COMPANY_NAME, style_company))
story.append(Paragraph(
    f"Дата формирования отчёта: {REPORT_DATE.strftime('%d.%m.%Y %H:%M')}",
    style_meta,
))

doc.build(story, onFirstPage=_add_page_footer, onLaterPages=_add_page_footer)

# Сохраняем в сессию
tender_pdf_filename = PDF_PATH
tender_pdf_verdict = VERDICT_LABEL

# Скачивание в Colab
try:
    from google.colab import files as colab_files
    colab_files.download(PDF_PATH)
    print("📥 PDF скачан автоматически.")
except Exception:
    print("💾 Автоскачивание недоступно (не Colab) — файл сохранён локально.")

elapsed = time.time() - _T0
print()
print("=" * 70)
print("✅ PDF-отчёт сформирован!")
print(f"📁 Файл: {PDF_PATH}")
print(f"🔖 Тендер: {TENDER_NO}")
print(f"⚖ Вердикт: {VERDICT_LABEL}")
print(f"⏱ Время: {int(round(elapsed))} сек.")
print("=" * 70)
print("Модуль 7 завершён.")
