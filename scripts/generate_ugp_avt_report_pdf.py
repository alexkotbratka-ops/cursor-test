#!/usr/bin/env python3
"""Генерация PDF-отчёта УГП-Авт: сводная таблица линейки + Гант дорожной карты."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
        PageBreak,
    )
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Установите reportlab: pip install reportlab\n" + str(exc)
    ) from exc


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "reports"
OUTPUT_FILE = OUTPUT_DIR / "UGP-Avt-roadmap-2026.pdf"

# --- Данные линейки (утверждено коммюнике 11.08.2026) ---
LINEUP = [
    ["УГП-Авт-П-Х-1", "А635165.036-ПХ.01", "1 м³", "1 кг (Хладон 227еа)", "4 м"],
    ["УГП-Авт-П-Х-2", "требует заполнения", "2 м³", "требует заполнения", "требует заполнения"],
    ["УГП-Авт-П-ФК-1", "требует заполнения", "1 м³", "требует заполнения", "требует заполнения"],
    ["УГП-Авт-П-ФК-2", "требует заполнения", "2 м³", "требует заполнения", "требует заполнения"],
    ["УГП-Авт-К-У-0,7", "требует заполнения", "0,7 м³", "требует заполнения (CO₂)", "требует заполнения"],
    ["УГП-Авт-К-У-1,4", "требует заполнения", "1,4 м³", "требует заполнения (CO₂)", "требует заполнения"],
    ["УГП-Авт-К-У-2,1", "требует заполнения", "2,1 м³", "требует заполнения (CO₂)", "требует заполнения"],
    ["УГП-Авт-К-У-3,5", "требует заполнения", "3,5 м³", "требует заполнения (CO₂)", "требует заполнения"],
    ["УГП-Авт-К-У-7,0", "требует заполнения", "7,0 м³", "требует заполнения (CO₂)", "требует заполнения"],
]

# --- Блоки дорожной карты: (название, start, end, роль) ---
ROADMAP = [
    ("Блок 1. Подготовка к презентации", date(2026, 8, 18), date(2026, 8, 24), "продукт / маркетинг"),
    ("Блок 2. Презентация линейки", date(2026, 8, 25), date(2026, 8, 28), "продукт / продажи"),
    ("Блок 3. Совещание по продвижению", date(2026, 8, 31), date(2026, 9, 1), "продажи / маркетинг / финансы"),
    ("Блок 4. Запуск продаж и система идей", date(2026, 9, 1), date(2026, 9, 30), "продажи / техподдержка / продукт"),
    ("Блок 5. Масштабирование", date(2026, 10, 1), date(2026, 12, 31), "продажи / маркетинг"),
]


def _styles():
    base = getSampleStyleSheet()
    title = ParagraphStyle(
        "TitleRU",
        parent=base["Heading1"],
        fontSize=16,
        alignment=TA_CENTER,
        spaceAfter=8,
    )
    h2 = ParagraphStyle(
        "H2RU",
        parent=base["Heading2"],
        fontSize=12,
        spaceBefore=10,
        spaceAfter=6,
    )
    body = ParagraphStyle(
        "BodyRU",
        parent=base["Normal"],
        fontSize=9,
        leading=12,
        alignment=TA_LEFT,
    )
    small = ParagraphStyle(
        "SmallRU",
        parent=base["Normal"],
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#475569"),
    )
    return title, h2, body, small


def _gantt_table(roadmap, canvas_start: date, canvas_end: date):
    """Простая табличная Гант-диаграмма по неделям."""
    total_days = (canvas_end - canvas_start).days or 1
    header = ["Этап", "Срок", "Роли", "Диаграмма"]
    rows = [header]
    bar_width = 80  # символов условной шкалы

    for name, start, end, role in roadmap:
        left = int((start - canvas_start).days / total_days * bar_width)
        right = int((end - canvas_start).days / total_days * bar_width)
        left = max(0, min(bar_width - 1, left))
        right = max(left + 1, min(bar_width, right))
        bar = "." * left + "#" * (right - left) + "." * (bar_width - right)
        period = f"{start.strftime('%d.%m')}–{end.strftime('%d.%m.%Y')}"
        rows.append([name, period, role, bar])

    table = Table(rows, colWidths=[70 * mm, 35 * mm, 45 * mm, 95 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("FONTNAME", (3, 1), (3, -1), "Courier"),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def build_pdf(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    title, h2, body, small = _styles()

    doc = SimpleDocTemplate(
        str(path),
        pagesize=landscape(A4),
        leftMargin=12 * mm,
        rightMargin=12 * mm,
        topMargin=12 * mm,
        bottomMargin=12 * mm,
    )
    story = []

    story.append(Paragraph("УГП-Авт — дорожная карта и линейка продуктов", title))
    story.append(
        Paragraph(
            f"ООО «Пожарная Автоматика» · отчёт для РОП · сформировано {datetime.now().strftime('%d.%m.%Y %H:%M')} · "
            "только роли, без ФИО · данные из внутренних материалов",
            small,
        )
    )
    story.append(Spacer(1, 6 * mm))

    story.append(Paragraph("1. Уточнённая линейка (коммюнике 11.08.2026)", h2))
    story.append(
        Paragraph(
            "П → только Хладон 227еа и ФК-5-1-12; К → только CO₂. Исключены: П+CO₂, К+хладон/ФК, объёмы вне утверждённого ряда.",
            body,
        )
    )
    story.append(Spacer(1, 3 * mm))

    lineup_header = ["Модель", "Артикул", "Объём", "Масса ГОТВ", "Длина трубки"]
    lineup_table = Table(
        [lineup_header] + LINEUP,
        colWidths=[40 * mm, 45 * mm, 25 * mm, 55 * mm, 45 * mm],
    )
    lineup_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2563eb")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eff6ff")]),
                ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#dcfce7")),
            ]
        )
    )
    story.append(lineup_table)
    story.append(Spacer(1, 2 * mm))
    story.append(
        Paragraph(
            "Зелёная строка — подтверждённые данные (КП ЦНИИ). Остальные поля — «требует заполнения» до 20.08.2026.",
            small,
        )
    )

    story.append(PageBreak())
    story.append(Paragraph("2. Гант-диаграмма дорожной карты (авг–дек 2026)", h2))
    story.append(
        Paragraph(
            "Блоки: подготовка → презентация → совещание по продвижению → запуск продаж и система идей → масштабирование. "
            "KPI: 200 комплектов до 31.12.2026; система идей ≤ 30.09.2026.",
            body,
        )
    )
    story.append(Spacer(1, 4 * mm))
    story.append(_gantt_table(ROADMAP, date(2026, 8, 1), date(2026, 12, 31)))
    story.append(Spacer(1, 4 * mm))

    story.append(Paragraph("3. KPI по блокам (кратко)", h2))
    kpi_data = [
        ["Блок", "KPI"],
        ["1. Подготовка", "Брошюра/ТП/РЭ без ошибок; КП на 9 моделей; сайт без МПТИ"],
        ["2. Презентация", "≥90% менеджеров продаж; ≥1 ответственный в техподдержке"],
        ["3. Совещание", "План Q4/Q1; бюджет; топ-3 сегмента"],
        ["4. Запуск + идеи", "≥40 шт. в сентябре; реестр идей ≤30.09"],
        ["5. Масштабирование", "Накопительно 200 шт. к 31.12; ≥2 сегментных КП"],
    ]
    kpi_table = Table(kpi_data, colWidths=[50 * mm, 180 * mm])
    kpi_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ]
        )
    )
    story.append(kpi_table)
    story.append(Spacer(1, 6 * mm))
    story.append(
        Paragraph(
            "Сценарии: пессимистичный ≤100 шт.; реалистичный 150–200 шт.; оптимистичный ≥200 шт. + дилерские контракты. "
            "Полный текстовый отчёт: reports/UGP-Avt-analiz-dorozhnaya-karta-2026.md",
            small,
        )
    )

    doc.build(story)
    return path


def main() -> None:
    out = build_pdf(OUTPUT_FILE)
    print(f"PDF создан: {out}")


if __name__ == "__main__":
    main()
