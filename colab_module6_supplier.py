# =============================================================================
# МОДУЛЬ 6 — Анализ поставщика и сверка с тендером (Google Colab)
# Выполняйте ПОСЛЕ модуля 4 или модуля 5.
#
# Важно:
#   • Сайт поставщика парсится ОДИН РАЗ за сессию.
#   • Результат сохраняется в SUPPLIER_DATA.
#   • Повторный запуск использует кэш (без повторного парсинга),
#     пока не выставите FORCE_REFRESH_SUPPLIER = True.
# =============================================================================

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

try:
    from google.colab import files as colab_files
except ImportError:
    colab_files = None
    print("⚠️ google.colab недоступен — отчёт сохранится локально без автоскачивания.")

# -----------------------------------------------------------------------------
# Настройки
# -----------------------------------------------------------------------------

FORCE_REFRESH_SUPPLIER = False  # True → принудительно перепарсить сайт
SUPPLIER_WEBSITE = "https://pozhavt.ru"
SUPPLIER_CONTACT_URL = "https://pozhavt.ru/kontakty/"
SUPPLIER_NAME = 'ООО «Пожарная Автоматика»'
HTTP_TIMEOUT = 30
USER_AGENT = (
    "Mozilla/5.0 (compatible; TenderSupplierBot/1.0; +https://pozhavt.ru) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# Канонический каталог актуальных линеек (источник истины для сверки)
CANONICAL_MODULES: Dict[str, Dict[str, Any]] = {
    "МПТХ2": {
        "gotv": ["Хладон 125", "Хладон 227еа"],
        "status": "актуально",
        "aliases": ["мптх2", "мпт х2", "mptx2"],
    },
    "МПТУ2": {
        "gotv": ["CO₂"],
        "status": "актуально",
        "aliases": ["мпту2", "мпт у2", "mptu2"],
    },
    "МПТИ": {
        "gotv": ["Инерген"],
        "status": "актуально",
        "aliases": ["мпти", "mpti"],
    },
    "УГП-Авт": {
        "gotv": ["различные"],
        "status": "актуально",
        "aliases": ["угп-авт", "угп авт", "угпавt", "ugp-avt", "автономн"],
    },
}

OBSOLETE_MODULES = ["FireDETEC", "МПТХ", "МПТУ", "Inerex"]
OBSOLETE_ALIASES: Dict[str, List[str]] = {
    "FireDETEC": ["firedetec", "fire detec", "фаердетек"],
    "МПТХ": ["мптх", "mptx"],       # без «2» — устаревшая
    "МПТУ": ["мпту", "mptu"],       # без «2» — устаревшая
    "Inerex": ["inerex", "инерекс"],
}

CANONICAL_GOTV = ["Хладон 125", "Хладон 227еа", "CO₂", "Инерген"]
GOTV_PATTERNS: List[Tuple[str, List[str]]] = [
    ("Хладон 125", [r"хладон\s*125", r"hfc[-\s]?125", r"r[-\s]?125"]),
    ("Хладон 227еа", [r"хладон\s*227\s*е?а?", r"hfc[-\s]?227", r"227ea", r"r[-\s]?227"]),
    ("CO₂", [r"двуокис[ьи]\s+углерод", r"\bco[\s₂2]\b", r"углекислот", r"carbon\s+dioxide"]),
    ("Инерген", [r"инерген", r"inergen", r"ig[-\s]?541"]),
]

CANONICAL_SERVICES = [
    "Обследование",
    "Проектирование",
    "Поставка",
    "Монтаж и ПНР",
    "Обслуживание",
]
SERVICE_PATTERNS: List[Tuple[str, List[str]]] = [
    ("Обследование", [r"обследован", r"предпроектн"]),
    ("Проектирование", [r"проектирован", r"проектно[-\s]?сметн"]),
    ("Поставка", [r"поставк", r"комплексн\w*\s+поставк"]),
    ("Монтаж и ПНР", [r"монтаж", r"пусконалад", r"\bпнр\b"]),
    ("Обслуживание", [r"обслуживан", r"техобслуживан", r"техническ\w*\s+обслуживан"]),
]


# =============================================================================
# Утилиты
# =============================================================================

def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def _norm(text: str) -> str:
    text = (text or "").lower().replace("ё", "е")
    text = text.replace("₂", "2").replace("²", "2")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _fetch_html(url: str) -> str:
    resp = requests.get(
        url,
        headers={"User-Agent": USER_AGENT, "Accept-Language": "ru,en;q=0.8"},
        timeout=HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding or "utf-8"
    return resp.text


def _html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()
    return re.sub(r"\s+", " ", soup.get_text(" ", strip=True))


def _find_emails(text: str) -> List[str]:
    found = re.findall(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", text)
    # фильтр мусора
    out = []
    for e in found:
        el = e.lower()
        if any(x in el for x in ("example.com", "sentry", "wixpress", "png", "jpg")):
            continue
        if el not in out:
            out.append(el)
    return out


def _find_phones(text: str) -> List[str]:
    pats = [
        r"\+7\s*\(?\d{3}\)?\s*\d{3}[-\s]?\d{2}[-\s]?\d{2}",
        r"8\s*\(?\d{3}\)?\s*\d{3}[-\s]?\d{2}[-\s]?\d{2}",
    ]
    out: List[str] = []
    for pat in pats:
        for m in re.finditer(pat, text):
            phone = re.sub(r"\s+", " ", m.group(0)).strip()
            if phone not in out:
                out.append(phone)
    return out


def _detect_gotv(text: str) -> List[str]:
    low = _norm(text)
    found = []
    for name, pats in GOTV_PATTERNS:
        if any(re.search(p, low, re.I) for p in pats):
            found.append(name)
    return found


def _detect_services(text: str) -> List[str]:
    low = _norm(text)
    found = []
    for name, pats in SERVICE_PATTERNS:
        if any(re.search(p, low, re.I) for p in pats):
            found.append(name)
    return found


def _module_mentioned(name: str, text_low: str, aliases: List[str]) -> bool:
    """
    Аккуратная проверка упоминания линейки.
    Для МПТХ / МПТУ без «2» — не считаем совпадением МПТХ2 / МПТУ2.
    """
    for alias in aliases:
        a = _norm(alias)
        if not a:
            continue
        if name in ("МПТХ", "МПТУ"):
            # устаревшие: «мптх» / «мпту», но НЕ «мптх2» / «мпту2»
            if re.search(rf"(?<![a-zа-я0-9]){re.escape(a)}(?!2)(?![a-zа-я0-9])", text_low):
                return True
        else:
            if a in text_low or re.search(rf"(?<![a-zа-я0-9]){re.escape(a)}(?![a-zа-я0-9])", text_low):
                return True
    return False


def _detect_modules(text: str) -> Tuple[List[str], List[str]]:
    """Возвращает (актуальные, устаревшие), найденные в тексте."""
    low = _norm(text)
    actual, obsolete = [], []
    for name, meta in CANONICAL_MODULES.items():
        aliases = [name] + list(meta.get("aliases") or [])
        if _module_mentioned(name, low, aliases):
            actual.append(name)
    for name in OBSOLETE_MODULES:
        aliases = [name] + list(OBSOLETE_ALIASES.get(name, []))
        if _module_mentioned(name, low, aliases):
            obsolete.append(name)
    return actual, obsolete


# =============================================================================
# Парсинг сайта (один раз)
# =============================================================================

def parse_supplier_website() -> Dict[str, Any]:
    """Скачивает главную + контакты и собирает SUPPLIER_DATA."""
    print(f"🌐 Парсинг сайта поставщика: {SUPPLIER_WEBSITE}")
    pages: Dict[str, str] = {}
    errors: List[str] = []

    for label, url in (("home", SUPPLIER_WEBSITE), ("contacts", SUPPLIER_CONTACT_URL)):
        try:
            html = _fetch_html(url)
            pages[label] = _html_to_text(html)
            print(f"   ✅ {label}: {url} ({len(pages[label]):,} символов)")
        except Exception as e:
            errors.append(f"{url}: {e}")
            print(f"   ⚠️ Не удалось загрузить {url}: {e}")

    blob = "\n".join(pages.values())
    if not blob.strip():
        raise RuntimeError(
            "❌ Не удалось получить текст сайта поставщика. "
            "Проверьте сеть / доступ к pozhavt.ru."
        )

    # ГОТВ / услуги — из сайта, с дополнением каноническим списком
    gotv_found = _detect_gotv(blob)
    services_found = _detect_services(blob)
    actual_found, obsolete_found = _detect_modules(blob)

    # Контакты
    emails = _find_emails(blob)
    phones = _find_phones(blob)
    # приоритет info@ / общий московский
    email = next((e for e in emails if e.startswith("info@")), emails[0] if emails else "")
    phone = next((p for p in phones if "495" in p.replace(" ", "")), phones[0] if phones else "")

    # Модули: канонические актуальные + подтверждение с сайта
    modules: Dict[str, Dict[str, Any]] = {}
    for name, meta in CANONICAL_MODULES.items():
        confirmed = name in actual_found
        modules[name] = {
            "gotv": list(meta["gotv"]),
            "status": "актуально",
            "confirmed_on_site": confirmed,
        }

    # ГОТВ: объединяем найденные с каноном (канон — полный актуальный перечень)
    gotv = list(CANONICAL_GOTV)
    for g in gotv_found:
        if g not in gotv:
            gotv.append(g)

    services = list(CANONICAL_SERVICES)
    for s in services_found:
        if s not in services:
            services.append(s)

    data = {
        "supplier_name": SUPPLIER_NAME,
        "gotv": gotv,
        "modules": modules,
        "obsolete_modules": list(OBSOLETE_MODULES),
        "services": services,
        "contacts": {
            "phone": phone or "+7 (495) 730-02-02",
            "email": email or "info@pozhavt.ru",
            "website": SUPPLIER_WEBSITE,
            "phone_spb": next((p for p in phones if "812" in p.replace(" ", "")), "+7 (812) 426-12-90"),
            "email_spb": next((e for e in emails if e.startswith("spb@")), "spb@pozhavt.ru"),
            "all_phones": phones,
            "all_emails": emails,
        },
        "site_confirmed_modules": actual_found,
        "site_found_obsolete": obsolete_found,
        "source_urls": [SUPPLIER_WEBSITE, SUPPLIER_CONTACT_URL],
        "parse_errors": errors,
        "last_update": _now_str(),
        "from_cache": False,
    }
    return data


def get_or_load_supplier_data() -> Dict[str, Any]:
    """
    Возвращает SUPPLIER_DATA из сессии, либо парсит сайт один раз.
    """
    global SUPPLIER_DATA  # noqa: PLW0603 — переменная сессии Colab

    cached = globals().get("SUPPLIER_DATA")
    if (
        not FORCE_REFRESH_SUPPLIER
        and isinstance(cached, dict)
        and cached.get("modules")
        and cached.get("gotv")
    ):
        data = dict(cached)
        data["from_cache"] = True
        print("♻️ SUPPLIER_DATA уже есть в сессии — повторный парсинг сайта НЕ выполняется.")
        print(f"   last_update: {data.get('last_update', '—')}")
        SUPPLIER_DATA = data
        return SUPPLIER_DATA

    if FORCE_REFRESH_SUPPLIER:
        print("🔄 FORCE_REFRESH_SUPPLIER=True — принудительный повторный парсинг.")

    data = parse_supplier_website()
    SUPPLIER_DATA = data
    print(f"💾 SUPPLIER_DATA сохранён в сессии (last_update={data['last_update']})")
    return SUPPLIER_DATA


# =============================================================================
# Сбор текста тендера для сверки
# =============================================================================

def collect_tender_corpus() -> Tuple[str, Dict[str, Any]]:
    """
    Собирает текст тендера из результатов модулей 4/5 (и при необходимости RAG).
    """
    parts: List[str] = []
    meta: Dict[str, Any] = {
        "has_tender_report": False,
        "has_tender_answers": False,
        "has_rag": False,
        "tender_number": "",
    }

    if "tender_number" in globals() and tender_number:
        meta["tender_number"] = str(tender_number)
        parts.append(f"Номер тендера: {tender_number}")

    if "tender_report" in globals() and tender_report:
        meta["has_tender_report"] = True
        parts.append(str(tender_report))

    if "tender_answers" in globals() and isinstance(tender_answers, dict):
        meta["has_tender_answers"] = True
        for num, ans in sorted(tender_answers.items(), key=lambda x: int(x[0]) if str(x[0]).isdigit() else 0):
            parts.append(f"Q{num}: {ans}")

    # fallback: куски индекса
    if len("\n".join(parts)) < 500 and "rag_index" in globals() and getattr(rag_index, "chunks", None):
        meta["has_rag"] = True
        # берём ограниченный объём
        chunks = list(rag_index.chunks)[:400]
        parts.append("\n".join(chunks))

    text = "\n".join(parts)
    return text, meta


# =============================================================================
# Сверка поставщик ↔ тендер
# =============================================================================

def reconcile_supplier_with_tender(
    supplier: Dict[str, Any],
    tender_text: str,
) -> Dict[str, Any]:
    tender_gotv = _detect_gotv(tender_text)
    tender_services = _detect_services(tender_text)
    tender_modules, tender_obsolete = _detect_modules(tender_text)

    supplier_gotv = list(supplier.get("gotv") or [])
    supplier_modules = list((supplier.get("modules") or {}).keys())
    supplier_services = list(supplier.get("services") or [])

    gotv_ok = [g for g in tender_gotv if g in supplier_gotv]
    gotv_missing = [g for g in tender_gotv if g not in supplier_gotv]

    modules_ok = [m for m in tender_modules if m in supplier_modules]
    modules_unknown = [m for m in tender_modules if m not in supplier_modules]

    services_ok = [s for s in tender_services if s in supplier_services]
    services_missing = [s for s in tender_services if s not in supplier_services]

    # Совместимость по ГОТВ → какие линейки поставщика покрывают потребность тендера
    covering_modules: Dict[str, List[str]] = {}
    for g in tender_gotv or supplier_gotv:
        covering_modules[g] = []
        for mname, meta in (supplier.get("modules") or {}).items():
            mg = meta.get("gotv") or []
            if g in mg or (g == "CO₂" and any("co" in _norm(x) for x in mg)):
                covering_modules[g].append(mname)
            if g in ("Хладон 125", "Хладон 227еа") and mname == "МПТХ2":
                if mname not in covering_modules[g]:
                    covering_modules[g].append(mname)
            if g == "Инерген" and mname == "МПТИ":
                if mname not in covering_modules[g]:
                    covering_modules[g].append(mname)
            if mname == "УГП-Авт" and "различн" in " ".join(_norm(x) for x in mg):
                if mname not in covering_modules[g]:
                    covering_modules[g].append(mname)

    # Оценка
    score = 100
    risks: List[str] = []
    recommendations: List[str] = []

    if tender_obsolete:
        score -= 25 * min(2, len(tender_obsolete))
        risks.append(
            "В тендерной документации упоминаются устаревшие линейки: "
            + ", ".join(tender_obsolete)
            + ". Их нельзя предлагать — только актуальные аналоги."
        )
        mapping = {
            "МПТХ": "МПТХ2 (Хладон 125 / 227еа)",
            "МПТУ": "МПТУ2 (CO₂)",
            "FireDETEC": "актуальные линейки МПТХ2 / МПТУ2 / МПТИ / УГП-Авт (по ГОТВ объекта)",
            "Inerex": "МПТИ (Инерген) или иная актуальная линейка по ГОТВ",
        }
        for obs in tender_obsolete:
            recommendations.append(
                f"Заменить «{obs}» на: {mapping.get(obs, 'актуальную линейку из каталога поставщика')}."
            )

    if gotv_missing:
        score -= 15 * len(gotv_missing)
        risks.append("ГОТВ из тендера не подтверждены в каталоге поставщика: " + ", ".join(gotv_missing))
    if not tender_gotv:
        recommendations.append(
            "В тексте тендера ГОТВ явно не найдены — уточните ТЗ / РД "
            "(Хладон 125/227еа, CO₂, Инерген) перед формированием КП."
        )
        score -= 5

    if modules_unknown:
        score -= 10
        risks.append("В тендере указаны модули вне актуального каталога: " + ", ".join(modules_unknown))

    if services_missing:
        score -= 5 * len(services_missing)
        recommendations.append(
            "Услуги из тендера, которые стоит явно подтвердить в КП: " + ", ".join(services_missing)
        )

    # Позитив
    matches: List[str] = []
    if gotv_ok:
        matches.append("ГОТВ совпадают: " + ", ".join(gotv_ok))
    if modules_ok:
        matches.append("Актуальные линейки упомянуты в тендере: " + ", ".join(modules_ok))
    if services_ok:
        matches.append("Услуги покрываются: " + ", ".join(services_ok))

    # Если тендер молчит о линейках, но ГОТВ есть — рекомендуем покрытие
    if tender_gotv and not modules_ok:
        for g in tender_gotv:
            cov = covering_modules.get(g) or []
            if cov:
                recommendations.append(f"Для ГОТВ «{g}» предложить линейку: {', '.join(cov)}.")
            else:
                recommendations.append(f"Для ГОТВ «{g}» подобрать актуальную линейку из каталога поставщика.")

    score = max(0, min(100, score))
    if score >= 80:
        verdict = "Высокая совместимость — поставщик может закрыть тендер актуальными линейками."
    elif score >= 55:
        verdict = "Средняя совместимость — требуется уточнение ГОТВ/линеек и аккуратная замена устаревших позиций."
    else:
        verdict = "Низкая совместимость / много рисков — нужна ручная проверка ТЗ и КП."

    return {
        "tender_gotv": tender_gotv,
        "tender_modules_actual": tender_modules,
        "tender_modules_obsolete": tender_obsolete,
        "tender_services": tender_services,
        "gotv_matched": gotv_ok,
        "gotv_missing": gotv_missing,
        "modules_matched": modules_ok,
        "modules_unknown": modules_unknown,
        "services_matched": services_ok,
        "services_missing": services_missing,
        "covering_modules": covering_modules,
        "matches": matches,
        "risks": risks,
        "recommendations": recommendations,
        "score": score,
        "verdict": verdict,
    }


# =============================================================================
# Отчёт
# =============================================================================

def format_supplier_report(
    supplier: Dict[str, Any],
    reconciliation: Dict[str, Any],
    tender_meta: Dict[str, Any],
) -> str:
    lines: List[str] = []
    lines.append("=" * 70)
    lines.append("МОДУЛЬ 6 — АНАЛИЗ ПОСТАВЩИКА И СВЕРКА С ТЕНДЕРОМ")
    lines.append("=" * 70)
    lines.append(f"Поставщик: {supplier.get('supplier_name', SUPPLIER_NAME)}")
    lines.append(f"Сайт: {supplier.get('contacts', {}).get('website', SUPPLIER_WEBSITE)}")
    lines.append(f"Данные сайта: {'из кэша сессии' if supplier.get('from_cache') else 'свежий парсинг'}")
    lines.append(f"last_update: {supplier.get('last_update', '—')}")
    lines.append(f"Номер тендера: {tender_meta.get('tender_number') or 'не указан'}")
    lines.append(f"Дата отчёта: {_now_str()}")
    lines.append("")

    lines.append("=" * 70)
    lines.append("1. SUPPLIER_DATA (актуальный каталог)")
    lines.append("=" * 70)
    lines.append("")
    lines.append("ГОТВ:")
    for g in supplier.get("gotv") or []:
        lines.append(f"  • {g}")
    lines.append("")
    lines.append("Актуальные линейки модулей:")
    for name, meta in (supplier.get("modules") or {}).items():
        conf = "подтверждено на сайте" if meta.get("confirmed_on_site") else "по канону / каталогу"
        lines.append(
            f"  • {name}: ГОТВ={', '.join(meta.get('gotv') or [])}; "
            f"статус={meta.get('status')}; {conf}"
        )
    lines.append("")
    lines.append("НЕ учитывать (устаревшее):")
    for name in supplier.get("obsolete_modules") or OBSOLETE_MODULES:
        lines.append(f"  ✗ {name}")
    lines.append("")
    lines.append("Услуги:")
    for s in supplier.get("services") or []:
        lines.append(f"  • {s}")
    lines.append("")
    contacts = supplier.get("contacts") or {}
    lines.append("Контакты:")
    lines.append(f"  Телефон: {contacts.get('phone', '—')}")
    lines.append(f"  Email:   {contacts.get('email', '—')}")
    lines.append(f"  Сайт:    {contacts.get('website', '—')}")
    if contacts.get("phone_spb"):
        lines.append(f"  СПб тел: {contacts.get('phone_spb')}")
    if contacts.get("email_spb"):
        lines.append(f"  СПб mail:{contacts.get('email_spb')}")
    lines.append("")

    lines.append("=" * 70)
    lines.append("2. СВЕРКА С ТЕНДЕРОМ")
    lines.append("=" * 70)
    lines.append(f"Оценка совместимости: {reconciliation.get('score', 0)} / 100")
    lines.append(f"Вердикт: {reconciliation.get('verdict', '—')}")
    lines.append("")
    lines.append("Найдено в тендере:")
    lines.append(f"  ГОТВ:              {', '.join(reconciliation.get('tender_gotv') or []) or 'не найдено'}")
    lines.append(
        f"  Линейки (актуал.): {', '.join(reconciliation.get('tender_modules_actual') or []) or 'не найдено'}"
    )
    lines.append(
        f"  Линейки (устар.):  {', '.join(reconciliation.get('tender_modules_obsolete') or []) or 'нет'}"
    )
    lines.append(
        f"  Услуги:            {', '.join(reconciliation.get('tender_services') or []) or 'не найдено'}"
    )
    lines.append("")
    lines.append("Совпадения:")
    if reconciliation.get("matches"):
        for m in reconciliation["matches"]:
            lines.append(f"  ✅ {m}")
    else:
        lines.append("  — явных совпадений по маркерам не найдено")
    lines.append("")
    lines.append("Покрытие ГОТВ актуальными линейками:")
    covering = reconciliation.get("covering_modules") or {}
    if covering:
        for g, mods in covering.items():
            lines.append(f"  • {g} → {', '.join(mods) if mods else 'нет прямой линейки'}")
    else:
        lines.append("  — нет данных")
    lines.append("")
    lines.append("Риски:")
    if reconciliation.get("risks"):
        for r in reconciliation["risks"]:
            lines.append(f"  ⚠ {r}")
    else:
        lines.append("  — критичных рисков не выявлено")
    lines.append("")
    lines.append("Рекомендации:")
    if reconciliation.get("recommendations"):
        for r in reconciliation["recommendations"]:
            lines.append(f"  → {r}")
    else:
        lines.append("  → Можно готовить КП на актуальных линейках МПТХ2 / МПТУ2 / МПТИ / УГП-Авт.")
    lines.append("")

    lines.append("=" * 70)
    lines.append("3. SUPPLIER_DATA (JSON)")
    lines.append("=" * 70)
    # компактный JSON без служебных полей парсинга
    export = {
        "gotv": supplier.get("gotv"),
        "modules": {
            k: {"gotv": v.get("gotv"), "status": v.get("status")}
            for k, v in (supplier.get("modules") or {}).items()
        },
        "obsolete_modules": supplier.get("obsolete_modules"),
        "services": supplier.get("services"),
        "contacts": {
            "phone": contacts.get("phone"),
            "email": contacts.get("email"),
            "website": contacts.get("website"),
        },
        "last_update": supplier.get("last_update"),
    }
    lines.append(json.dumps(export, ensure_ascii=False, indent=2))
    lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines) + "\n"


def build_report_name(tender_no: str = "") -> str:
    d = datetime.now().strftime("%Y%m%d_%H%M%S")
    if tender_no:
        safe = re.sub(r"[^\w\-]+", "_", str(tender_no))
        return f"Сверка_поставщика_{safe}_{d}.txt"
    return f"Сверка_поставщика_{d}.txt"


# =============================================================================
# Запуск
# =============================================================================

print("=" * 70)
print("🚀 МОДУЛЬ 6 — Анализ поставщика и сверка с тендером")
print("=" * 70)
print(f"Сайт: {SUPPLIER_WEBSITE}")
print(f"FORCE_REFRESH_SUPPLIER = {FORCE_REFRESH_SUPPLIER}")
print()

_t0 = time.time()

# 1) SUPPLIER_DATA (кэш или парсинг)
SUPPLIER_DATA = get_or_load_supplier_data()

print("\n📦 SUPPLIER_DATA:")
print(f"   ГОТВ: {', '.join(SUPPLIER_DATA.get('gotv') or [])}")
print(f"   Линейки: {', '.join((SUPPLIER_DATA.get('modules') or {}).keys())}")
print(f"   Устаревшие (игнор): {', '.join(SUPPLIER_DATA.get('obsolete_modules') or [])}")
print(f"   Услуги: {', '.join(SUPPLIER_DATA.get('services') or [])}")
c = SUPPLIER_DATA.get("contacts") or {}
print(f"   Контакты: {c.get('phone')} | {c.get('email')} | {c.get('website')}")

# 2) Текст тендера
tender_text, tender_meta = collect_tender_corpus()
if not tender_text.strip():
    print(
        "\n⚠️ Текст тендера не найден в сессии "
        "(нет tender_report / tender_answers / rag_index).\n"
        "   Сверка будет ограничена — сначала выполните модуль 4 или 5."
    )
else:
    src = []
    if tender_meta.get("has_tender_report"):
        src.append("tender_report")
    if tender_meta.get("has_tender_answers"):
        src.append("tender_answers")
    if tender_meta.get("has_rag"):
        src.append("rag_index")
    print(f"\n📄 Корпус тендера: {len(tender_text):,} символов ({', '.join(src) or '—'})")

# 3) Сверка
print("\n🔍 Сверка поставщик ↔ тендер...")
SUPPLIER_RECONCILIATION = reconcile_supplier_with_tender(SUPPLIER_DATA, tender_text)
print(f"   Оценка: {SUPPLIER_RECONCILIATION['score']} / 100")
print(f"   {SUPPLIER_RECONCILIATION['verdict']}")
if SUPPLIER_RECONCILIATION.get("tender_modules_obsolete"):
    print(
        "   ⚠ Устаревшие линейки в тендере: "
        + ", ".join(SUPPLIER_RECONCILIATION["tender_modules_obsolete"])
    )

# 4) Отчёт
supplier_report = format_supplier_report(SUPPLIER_DATA, SUPPLIER_RECONCILIATION, tender_meta)
supplier_report_filename = build_report_name(tender_meta.get("tender_number") or "")
with open(supplier_report_filename, "w", encoding="utf-8") as f:
    f.write(supplier_report)

if colab_files is not None:
    colab_files.download(supplier_report_filename)

elapsed = time.time() - _t0
print()
print("=" * 70)
print("✅ Модуль 6 завершён")
print(f"📁 Отчёт: {supplier_report_filename}")
print(f"💾 SUPPLIER_DATA в сессии (from_cache={SUPPLIER_DATA.get('from_cache')})")
print(f"⏱️ Время этапа: {elapsed:.1f} сек.")
if colab_files is not None:
    print("📥 Файл скачан.")
else:
    print(f"💾 {os.path.abspath(supplier_report_filename)}")
print()
print("Подсказка: повторный запуск модуля 6 возьмёт SUPPLIER_DATA из сессии.")
print("Чтобы перепарсить сайт: FORCE_REFRESH_SUPPLIER = True")
