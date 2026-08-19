#!/usr/bin/env python3
"""Генерация PDF-отчёта УПА: карта игроков категории + Гант дорожной карты."""

from __future__ import annotations

from datetime import date
from pathlib import Path

try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
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


FONT_REG = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
pdfmetrics.registerFont(TTFont("DejaVu", FONT_REG))
pdfmetrics.registerFont(TTFont("DejaVuBold", FONT_BOLD))

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "reports"
OUTPUT_FILE = OUTPUT_DIR / "UPA-roadmap-2026.pdf"

# Карта игроков категории УПА
PLAYERS = [
    ["НТО Пламя", "УПА-Г", "0,03–18 м³", "до 7 ГОТВ; колба/транслятор", "10 лет"],
    ["AFESPRO", "УПА 25-…", "0,25–6 л", "Х125/227еа/ФК; колба/эл.пуск", "уточнить"],
    ["Спецавтоматика", "F-Line", "до 1,5 / 3 м³", "ФК; трубка до 10 м", "≥10 лет"],
    ["ПироХимика", "Парабола", "~1,1 м³", "хладон, прямое", "уточнить"],
    ["СКБ Тензор", "АУП-01Ф", "до ~10,6 м³", "Х227еа; FireDetec", "уточнить"],
    ["Эпотос", "УГП Эол", "0,25–1 м³", "газоген. + эл.пуск", "уточнить"],
    ["Пож. Автоматика", "УГП-Авт", "П 1–2; К 0,7–7 м³", "П: 227еа/ФК; К: CO₂; 140°C", "П-Х-1: 12 мес."],
]

ROADMAP = [
    ("Блок 1. Язык категории УПА", date(2026, 8, 19), date(2026, 8, 24), "продукт / маркетинг"),
    ("Блок 2. Презентация УГП-Авт как УПА", date(2026, 8, 25), date(2026, 8, 28), "продукт / продажи"),
    ("Блок 3. Сегменты и каналы УПА", date(2026, 8, 31), date(2026, 9, 1), "продажи / маркетинг / финансы"),
    ("Блок 4. Продажи + идея 0,03–1 м³", date(2026, 9, 1), date(2026, 9, 30), "продажи / техподдержка / продукт"),
    ("Блок 5. Масштабирование → 200 шт.", date(2026, 10, 1), date(2026, 12, 31), "продажи / маркетинг"),
]


def _styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "TitleRU",
            parent=base["Heading1"],
            fontName="DejaVuBold",
            fontSize=16,
            alignment=TA_CENTER,
            spaceAfter=8,
        ),
        "h2": ParagraphStyle(
            "H2RU",
            parent=base["Heading2"],
            fontName="DejaVuBold",
            fontSize=12,
            spaceBefore=10,
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "BodyRU",
            parent=base["Normal"],
            fontName="DejaVu",
            fontSize=9,
            leading=12,
            alignment=TA_LEFT,
        ),
        "small": ParagraphStyle(
            "SmallRU",
            parent=base["Normal"],
            fontName="DejaVu",
            fontSize=8,
            leading=10,
        ),
        "cell": ParagraphStyle(
            "CellRU",
            parent=base["Normal"],
            fontName="DejaVu",
            fontSize=7.5,
            leading=9,
        ),
        "cellb": ParagraphStyle(
            "CellBRU",
            parent=base["Normal"],
            fontName="DejaVuBold",
            fontSize=7.5,
            leading=9,
        ),
    }


def _p(text: str, style) -> Paragraph:
    return Paragraph(str(text).replace("\n", "<br/>"), style)


def build_players_table(styles):
    header = [
        _p("Игрок", styles["cellb"]),
        _p("Продукт", styles["cellb"]),
        _p("Объёмы", styles["cellb"]),
        _p("ГОТВ / обнаружение", styles["cellb"]),
        _p("Срок службы", styles["cellb"]),
    ]
    data = [header]
    for row in PLAYERS:
        data.append([_p(c, styles["cell"]) for c in row])
    col_w = [38 * mm, 32 * mm, 35 * mm, 70 * mm, 30 * mm]
    t = Table(data, colWidths=col_w, repeatRows=1)
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e79")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#e8f0fe")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#888888")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return t


def build_gantt(styles):
    start = date(2026, 8, 19)
    end = date(2026, 12, 31)
    total_days = (end - start).days + 1
    bar_w = 160 * mm
    label_w = 70 * mm
    role_w = 45 * mm

    header = [
        _p("Блок", styles["cellb"]),
        _p("Роль", styles["cellb"]),
        _p("Авг", styles["cellb"]),
        _p("Сен", styles["cellb"]),
        _p("Окт", styles["cellb"]),
        _p("Ноя", styles["cellb"]),
        _p("Дек", styles["cellb"]),
    ]
    # Simplified month columns for readability + textual dates
    rows = [header]
    month_spans = [
        ("Авг", date(2026, 8, 1), date(2026, 8, 31)),
        ("Сен", date(2026, 9, 1), date(2026, 9, 30)),
        ("Окт", date(2026, 10, 1), date(2026, 10, 31)),
        ("Ноя", date(2026, 11, 1), date(2026, 11, 30)),
        ("Дек", date(2026, 12, 1), date(2026, 12, 31)),
    ]

    def mark(b_start: date, b_end: date, m_start: date, m_end: date) -> str:
        if b_end < m_start or b_start > m_end:
            return ""
        return "████"

    for name, b_start, b_end, role in ROADMAP:
        row = [
            _p(f"{name}<br/><font size='6'>{b_start.strftime('%d.%m')}–{b_end.strftime('%d.%m.%Y')}</font>", styles["cell"]),
            _p(role, styles["cell"]),
        ]
        for _, m_s, m_e in month_spans:
            row.append(_p(mark(b_start, b_end, m_s, m_e), styles["cell"]))
        rows.append(row)

    t = Table(
        rows,
        colWidths=[label_w, role_w, 22 * mm, 22 * mm, 22 * mm, 22 * mm, 22 * mm],
        repeatRows=1,
    )
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e79")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#888888")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (2, 1), (-1, -1), "CENTER"),
                ("BACKGROUND", (2, 1), (-1, -1), colors.HexColor("#f7fafc")),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    # silence unused
    _ = (total_days, bar_w)
    return t


def build_verdict(styles):
    data = [
        [
            _p("Зона", styles["cellb"]),
            _p("Оценка", styles["cellb"]),
            _p("Комментарий", styles["cellb"]),
        ],
        [
            _p("Нормативный драйвер УПА", styles["cell"]),
            _p("Зелёный", styles["cellb"]),
            _p("СП 486/485 закрепляют спрос", styles["cell"]),
        ],
        [
            _p("Язык УПА в материалах УГП-Авт", styles["cell"]),
            _p("Жёлтый", styles["cellb"]),
            _p("Доработать до 24.08", styles["cell"]),
        ],
        [
            _p("Позиция 0,03–1 м³", styles["cell"]),
            _p("Красный", styles["cellb"]),
            _p("Дыра vs УПА-Г / AFESPRO", styles["cell"]),
        ],
        [
            _p("Позиция 1–7 м³", styles["cell"]),
            _p("Жёлтый", styles["cellb"]),
            _p("Есть продукт и склад", styles["cell"]),
        ],
        [
            _p("План 200 шт. до 31.12.2026", styles["cell"]),
            _p("Жёлтый", styles["cellb"]),
            _p("Достижим при Блоках 1–5", styles["cell"]),
        ],
    ]
    t = Table(data, colWidths=[55 * mm, 25 * mm, 95 * mm])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e79")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#888888")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BACKGROUND", (1, 1), (1, 1), colors.HexColor("#c6f6d5")),
                ("BACKGROUND", (1, 2), (1, 2), colors.HexColor("#fefcbf")),
                ("BACKGROUND", (1, 3), (1, 3), colors.HexColor("#fed7d7")),
                ("BACKGROUND", (1, 4), (1, 4), colors.HexColor("#fefcbf")),
                ("BACKGROUND", (1, 5), (1, 5), colors.HexColor("#fefcbf")),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return t


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    styles = _styles()
    doc = SimpleDocTemplate(
        str(OUTPUT_FILE),
        pagesize=landscape(A4),
        leftMargin=12 * mm,
        rightMargin=12 * mm,
        topMargin=12 * mm,
        bottomMargin=12 * mm,
        title="УПА — анализ категории 2026",
        author="ООО Пожарная Автоматика",
    )
    story = []
    story.append(_p("УПА — анализ категории и дорожная карта", styles["title"]))
    story.append(
        _p(
            "ООО «Пожарная Автоматика» · УГП-Авт в сегменте УПА · отчёт 19.08.2026 · "
            "склад 200 комплектов → цель до 31.12.2026",
            styles["body"],
        )
    )
    story.append(Spacer(1, 4 * mm))
    story.append(_p("1. Карта игроков категории УПА", styles["h2"]))
    story.append(
        _p(
            "УПА — нормативная категория (ТР ЕАЭС 043/2017; СП 486/485). "
            "УГП-Авт — продукт компании внутри категории. Строка компании выделена.",
            styles["small"],
        )
    )
    story.append(Spacer(1, 2 * mm))
    story.append(build_players_table(styles))
    story.append(Spacer(1, 4 * mm))
    story.append(_p("2. Вердикт по зонам", styles["h2"]))
    story.append(build_verdict(styles))

    story.append(PageBreak())
    story.append(_p("3. Дорожная карта сегмента УПА (Гант)", styles["title"]))
    story.append(
        _p(
            "KPI — артефакты и даты. Помесячный план продаж утверждается в Блоке 3, "
            "не задаётся заранее. Жёсткая цель: 200 шт. УГП-Авт к 31.12.2026.",
            styles["body"],
        )
    )
    story.append(Spacer(1, 3 * mm))
    story.append(build_gantt(styles))
    story.append(Spacer(1, 5 * mm))
    story.append(_p("Ключевые решения Блока 3", styles["h2"]))
    story.append(
        _p(
            "• Приоритеты сегментов: щитовики, ЦОД/стойки, АСУ ТП, транспорт, склады<br/>"
            "• Каналы: проектировщики, монтажники МЧС, дилеры, тендеры, маркетплейсы<br/>"
            "• Решение по дыре 0,03–1 м³: свой SKU / OEM / партнёр (go / hold / partner)<br/>"
            "• План Q4 2026 и Q1 2027 — цифры только из протокола совещания",
            styles["body"],
        )
    )
    story.append(Spacer(1, 4 * mm))
    story.append(
        _p(
            "Полный текст: reports/UPA-analiz-dorozhnaya-karta-2026.md · "
            "Структура: reports/UPA-dorozhnaya-karta-struktura-2026.md",
            styles["small"],
        )
    )
    doc.build(story)
    print(f"Written: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
