# =============================================================================
# МОДУЛЬ 4 — Автоматический анализ тендера и формирование отчёта
# Версия после самопроверки: regex номера, дедуп источников, forced retrieve,
# сводка ЛСР, поиск пени/расторжения только в Договоре, новые поля.
# Выполняйте строго ПОСЛЕ модуля 3.
# Использует: rag_index, ask_deepseek, uploaded_files (модули 1–2).
# =============================================================================

import hashlib
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

try:
    from google.colab import files as colab_files
except ImportError:
    colab_files = None
    print("⚠️ google.colab недоступен — файл сохранится локально без автоскачивания.")


# --- Проверки сессии ---
if "rag_index" not in globals() or rag_index is None:
    raise RuntimeError("❌ rag_index не найден. Сначала выполните модули 1–3.")

if not getattr(rag_index, "chunks", None):
    raise RuntimeError("❌ Индекс пуст. Сначала выполните модуль 3 (Чтение и индексация).")

if "ask_deepseek" not in globals():
    raise RuntimeError("❌ ask_deepseek не найден. Сначала выполните модуль 1.")


# =============================================================================
# Константы
# =============================================================================

TOP_K = 12
ANALYSIS_DATE = datetime.now()

PRIORITY_PROMPT = (
    "Если информация есть в Извещении, Закупочной документации, "
    "Проекте договора, Техническом задании, ЛСР или ВОР — используй её в первую очередь. "
    "Если в контексте есть число, дата, перечень или точная формулировка — выпиши их. "
    "Не отвечай «не указано»/«нет в контексте», если данные есть хотя бы в одном фрагменте. "
    "Отвечай кратко и по делу на русском языке."
)

# Маски для forced retrieve (подстрока в имени файла, lower)
MASK_LSR = ("лср", "вор")
MASK_SCHEDULE = ("график",)
MASK_RD = ("рд ", " рд", "/рд", "рд_", "альбом рд", "14-27-")  # рабочие чертежи / альбомы РД
MASK_CONTRACT = ("договор",)
MASK_NOTICE = ("извещение",)
MASK_TZ = ("техническое задание", "техзадани", "/тз", "тз.", "тз/")

# Поля → принудительные маски источников
FIELD_SOURCE_MASKS: Dict[str, Tuple[str, ...]] = {
    "Срок выполнения работ (конечная дата)": MASK_SCHEDULE + ("договор", "извещение", "техническое задание"),
    "Поэтапное выполнение работ": MASK_SCHEDULE + ("техническое задание", "договор"),
    "Штрафы/пени": MASK_CONTRACT,
    "Ответственность за просрочку": MASK_CONTRACT,
    "Возможность расторжения контракта": MASK_CONTRACT,
    "Арбитражная оговорка (суд)": MASK_CONTRACT,
    "Форс-мажорные обстоятельства": MASK_CONTRACT,
    "Состав ЗИП (запасных частей)": MASK_TZ + MASK_LSR + MASK_RD,
    "Модули пожаротушения (перечень)": MASK_LSR + MASK_TZ + MASK_RD,
    "Оборудование в системе (полный перечень)": MASK_LSR + MASK_TZ + MASK_RD,
    "Огнетушащее вещество": MASK_LSR + MASK_TZ + MASK_RD,
    "Начальная (максимальная) цена контракта": MASK_NOTICE + ("закупочн",) + MASK_LSR + MASK_CONTRACT,
    "Сметная стоимость (ориентир)": MASK_LSR,
    "Сводка из ЛСР (топ позиций)": MASK_LSR,
}

CRITICAL_FIELDS = {
    "Номер тендера / ИКЗ",
    "Предмет закупки",
    "Объект (здание/помещение)",
    "Место выполнения работ",
    "Адрес объекта",
    "Начальная (максимальная) цена контракта",
    "Сметная стоимость (ориентир)",
    "Срок выполнения работ (конечная дата)",
    "Срок подачи заявок",
    "Время окончания подачи заявок",
    "Размер обеспечения заявки",
    "Обеспечение заявки",
    "Обеспечение контракта (размер)",
    "Штрафы/пени",
    "Ответственность за просрочку",
    "Возможность расторжения контракта",
    "Арбитражная оговорка (суд)",
    "Состав ЗИП (запасных частей)",
    "Оборудование в системе (полный перечень)",
    "Модули пожаротушения (перечень)",
}


# =============================================================================
# Дедупликация чанков индекса
# =============================================================================

def _basename_key(source: str) -> str:
    """Канонический ключ файла: нижний регистр basename без копии '(2)'."""
    name = str(source).replace("\\", "/").split("/")[-1].lower().strip()
    name = re.sub(r"\s*\(\d+\)(?=\.\w+$)", "", name)  # file (2).docx → file.docx
    return name


def _content_hash(text: str) -> str:
    norm = re.sub(r"\s+", " ", (text or "").strip().lower())
    return hashlib.md5(norm.encode("utf-8", errors="ignore")).hexdigest()


def build_deduped_index():
    """
    Строит дедуплицированные списки чанков.
    Убирает тройные копии одних и тех же документов из разных ZIP.
    Возвращает (chunks, sources, indices_in_original).
    """
    seen_content: Set[str] = set()
    seen_source_chunk: Set[Tuple[str, str]] = set()
    chunks: List[str] = []
    sources: List[str] = []
    orig_idx: List[int] = []

    # предпочитаем более «короткий» путь (без вложенного дубля процедуры)
    candidates = list(enumerate(zip(rag_index.chunks, rag_index.sources)))

    def path_rank(src: str) -> Tuple[int, int]:
        s = src.lower()
        # меньше — лучше: отдельные файлы / лот №1 предпочтительнее «процедуре»
        penalty = 0
        if "процедуре" in s:
            penalty += 2
        if s.count(".zip/") + s.count(".rar/") > 1:
            penalty += 1
        return (penalty, len(s))

    candidates.sort(key=lambda it: path_rank(it[1][1]))

    for i, (text, source) in candidates:
        h = _content_hash(text)
        bkey = _basename_key(source)
        pair = (bkey, h)
        if h in seen_content or pair in seen_source_chunk:
            continue
        seen_content.add(h)
        seen_source_chunk.add(pair)
        chunks.append(text)
        sources.append(source)
        orig_idx.append(i)

    return chunks, sources, orig_idx


print("🔧 Дедупликация индекса...")
DEDUP_CHUNKS, DEDUP_SOURCES, DEDUP_ORIG_IDX = build_deduped_index()
print(
    f"   Было чанков: {len(rag_index.chunks)} → после дедупа: {len(DEDUP_CHUNKS)} "
    f"(уникальных basename: {len(set(_basename_key(s) for s in DEDUP_SOURCES))})"
)

# TF-IDF по дедуплицированному корпусу (локальный поиск модуля 4)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

_DEDUP_VECTORIZER = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True)
_DEDUP_MATRIX = _DEDUP_VECTORIZER.fit_transform(DEDUP_CHUNKS) if DEDUP_CHUNKS else None


# =============================================================================
# Поиск с масками и бустом
# =============================================================================

def _match_mask(source: str, masks: Sequence[str]) -> bool:
    s = source.lower().replace("\\", "/")
    base = _basename_key(source)
    for m in masks:
        m = m.lower()
        if m in s or m.strip() in base:
            return True
    return False


def _source_boost(source: str, extra_masks: Optional[Sequence[str]] = None) -> float:
    s = source.lower()
    bonus = 0.0
    keywords = [
        ("извещение", 0.35),
        ("закупочная", 0.32),
        ("договор", 0.30),
        ("техническое задание", 0.28),
        ("лср", 0.40),
        ("вор", 0.40),
        ("график", 0.38),
        ("рд", 0.25),
    ]
    for kw, val in keywords:
        if kw in s:
            bonus = max(bonus, val)
    if extra_masks and _match_mask(source, extra_masks):
        bonus = max(bonus, 0.45)
    return bonus


def search_dedup(
    query: str,
    top_k: int = TOP_K,
    source_masks: Optional[Sequence[str]] = None,
    mask_only: bool = False,
) -> List[Dict[str, Any]]:
    """Поиск по дедуплицированному индексу с опциональной маской источников."""
    if _DEDUP_MATRIX is None or not DEDUP_CHUNKS:
        return []

    q = _DEDUP_VECTORIZER.transform([query])
    scores = cosine_similarity(q, _DEDUP_MATRIX).ravel()

    results = []
    for i, score in enumerate(scores):
        src = DEDUP_SOURCES[i]
        if source_masks:
            matched = _match_mask(src, source_masks)
            if mask_only and not matched:
                continue
            if matched:
                score = float(score) + 0.5  # сильный boost
            else:
                score = float(score) + _source_boost(src)
        else:
            score = float(score) + _source_boost(src)

        if score <= 0:
            continue
        results.append(
            {
                "text": DEDUP_CHUNKS[i],
                "source": src,
                "score": float(score),
                "chunk": DEDUP_ORIG_IDX[i] + 1,
            }
        )

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def force_retrieve_by_mask(
    masks: Sequence[str],
    query: str,
    top_k: int = TOP_K,
) -> List[Dict[str, Any]]:
    """Принудительный поиск только внутри файлов, подходящих под маску."""
    return search_dedup(query, top_k=top_k, source_masks=masks, mask_only=True)


def merge_hits(hit_lists: List[List[Dict[str, Any]]], top_k: int = TOP_K) -> List[Dict[str, Any]]:
    best: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for hits in hit_lists:
        for h in hits:
            key = (h.get("source", ""), h.get("text", "")[:200])
            if key not in best or h["score"] > best[key]["score"]:
                best[key] = h
    return sorted(best.values(), key=lambda x: x["score"], reverse=True)[:top_k]


def format_context(hits: List[Dict[str, Any]]) -> str:
    parts = []
    for i, h in enumerate(hits, start=1):
        parts.append(
            f"[Фрагмент {i} | Источник: {h.get('source', '?')} | "
            f"чанк #{h.get('chunk', '?')} | score={h.get('score', 0):.3f}]\n"
            f"{h.get('text', '')}"
        )
    return "\n\n---\n\n".join(parts)


# =============================================================================
# Ответы LLM
# =============================================================================

def _is_empty_answer(text: str) -> bool:
    text = (text or "").strip()
    if not text:
        return True
    low = text.lower()
    markers = [
        "не указано",
        "не найден",
        "нет информации",
        "не удалось",
        "в контексте нет",
        "в предоставленном контексте нет",
        "в предоставленном контексте не",
        "информация отсутствует",
        "ответить на вопрос невозможно",
        "невозможно определить",
        "не могу найти",
        "не могу ответить",
    ]
    if len(text) < 220 and any(m in low for m in markers):
        if re.search(r"\d{3,}", text) and ("руб" in low or "инн" in low or "%" in text):
            return False
        return True
    return False


def _clean_answer(text: str) -> str:
    text = (text or "").strip()
    if _is_empty_answer(text):
        return "Не указано"
    return text


def ask_with_variants(
    question: str,
    variants: Optional[List[str]] = None,
    top_k: int = TOP_K,
    source_masks: Optional[Sequence[str]] = None,
    mask_only: bool = False,
    retry_if_empty: bool = True,
) -> Tuple[str, List[Dict[str, Any]]]:
    variants = variants or []
    queries = [question] + list(variants)

    hit_lists = []
    for q in queries:
        hit_lists.append(
            search_dedup(q, top_k=top_k, source_masks=source_masks, mask_only=mask_only)
        )
        # доп. forced retrieve, если заданы маски
        if source_masks:
            hit_lists.append(force_retrieve_by_mask(source_masks, q, top_k=top_k))

    hits = merge_hits(hit_lists, top_k=top_k)
    if not hits:
        return "Не указано", []

    context = format_context(hits)
    prompt = f"{PRIORITY_PROMPT}\n\nВопрос: {question}"
    answer = _clean_answer(ask_deepseek(prompt, context=context))

    if retry_if_empty and answer == "Не указано" and (variants or source_masks):
        retry_q = (variants[-1] if variants else question) + (
            " Ответь только фактами из контекста. Если есть частичный ответ — приведи его."
        )
        retry_hits = merge_hits(
            [
                search_dedup(retry_q, top_k=top_k, source_masks=source_masks, mask_only=mask_only),
                force_retrieve_by_mask(source_masks, retry_q, top_k=top_k) if source_masks else [],
                hits,
            ],
            top_k=top_k,
        )
        if retry_hits:
            retry_ans = _clean_answer(
                ask_deepseek(f"{PRIORITY_PROMPT}\n\nВопрос: {retry_q}", context=format_context(retry_hits))
            )
            if retry_ans != "Не указано":
                return retry_ans, retry_hits

    return answer, hits


def record_sources(field: str, hits: List[Dict[str, Any]], store: Dict[str, List[Dict[str, Any]]]):
    seen = set()
    items = []
    for h in hits:
        src = h.get("source", "?")
        key = (src, h.get("chunk"))
        if key in seen:
            continue
        seen.add(key)
        items.append(
            {
                "source": src,
                "chunk": h.get("chunk"),
                "score": round(float(h.get("score", 0.0)), 4),
            }
        )
    store[field] = items


# =============================================================================
# Номер тендера
# =============================================================================

TENDER_NUM_RE = re.compile(r"(B\d{10,})", re.IGNORECASE)


def extract_tender_number_from_filenames() -> str:
    names: List[str] = []
    if "uploaded_files" in globals() and uploaded_files:
        names.extend(list(uploaded_files.keys()))
    names.extend(DEDUP_SOURCES)

    for name in names:
        m = TENDER_NUM_RE.search(str(name))
        if m:
            return m.group(1).upper()
        m2 = re.search(r"\b(\d{18,36})\b", str(name))
        if m2:
            return m2.group(1)
    return ""


def extract_tender_number_from_text(raw: str) -> str:
    if not raw or raw == "Не указано":
        return ""
    m = TENDER_NUM_RE.search(raw)
    if m:
        return m.group(1).upper()
    m2 = re.search(r"\b(\d{18,36})\b", raw)
    if m2:
        return m2.group(1)
    m3 = re.search(r"\b([A-Za-z]\d{10,})\b", raw)
    if m3:
        return m3.group(1).upper()
    return ""


def build_report_filename(tender_no: str) -> str:
    date_s = ANALYSIS_DATE.strftime("%Y%m%d")
    time_s = ANALYSIS_DATE.strftime("%H%M%S")
    if tender_no:
        safe = re.sub(r"[^\w\-]+", "_", tender_no).strip("_")
        return f"Анализ_тендера_{safe}_{date_s}.txt"
    return f"Анализ_тендера_{date_s}_{time_s}.txt"


# =============================================================================
# Структура отчёта
# =============================================================================

REPORT_FIELDS: List[Tuple[str, str, str, List[str]]] = [
    # ----- 1 -----
    (
        "1. ОБЩАЯ ИНФОРМАЦИЯ О ТЕНДЕРЕ",
        "Номер тендера / ИКЗ",
        "Каков номер тендера, извещения, закупки, процедуры или ИКЗ? Укажи точный номер целиком.",
        [
            "Найди номер извещения вида B… (все цифры после B).",
            "Какой номер процедуры / лота в Извещении или Закупочной документации?",
        ],
    ),
    (
        "1. ОБЩАЯ ИНФОРМАЦИЯ О ТЕНДЕРЕ",
        "Предмет закупки",
        "Каков предмет закупки? Кратко опиши работы (САУГПТ, ЕСУМИС и т.п.).",
        [
            "Предмет договора / закупки: СМР, ПНР, автоматизация.",
            "Системы пожаротушения и мониторинга в предмете закупки.",
        ],
    ),
    (
        "1. ОБЩАЯ ИНФОРМАЦИЯ О ТЕНДЕРЕ",
        "Объект (здание/помещение)",
        "В каком здании / помещении / сооружении выполняются работы (пункт пропуска, терминал и т.п.)?",
        [
            "Наименование объекта капитального строительства / здания для монтажа.",
            "Здание морского пункта пропуска или иное помещение объекта.",
        ],
    ),
    (
        "1. ОБЩАЯ ИНФОРМАЦИЯ О ТЕНДЕРЕ",
        "Заказчик",
        "Кто заказчик (полное наименование организации)?",
        ["Наименование заказчика / организатора закупки."],
    ),
    (
        "1. ОБЩАЯ ИНФОРМАЦИЯ О ТЕНДЕРЕ",
        "ИНН заказчика",
        "Какой ИНН заказчика?",
        ["ИНН организации-заказчика."],
    ),
    (
        "1. ОБЩАЯ ИНФОРМАЦИЯ О ТЕНДЕРЕ",
        "Контакты заказчика (телефон, email)",
        "Какие контакты заказчика указаны (телефон, email, контактные лица)?",
        ["Телефон и email заказчика, контактные лица."],
    ),
    (
        "1. ОБЩАЯ ИНФОРМАЦИЯ О ТЕНДЕРЕ",
        "Адрес объекта",
        "Какой адрес объекта / места выполнения работ?",
        [
            "Адрес площадки / порта / сооружения.",
            "Приморский край, Шкотовский район — уточни полный адрес.",
        ],
    ),
    (
        "1. ОБЩАЯ ИНФОРМАЦИЯ О ТЕНДЕРЕ",
        "Место выполнения работ",
        "Каково место выполнения работ (площадка, объект, регион)?",
        ["Место выполнения работ по Извещению / договору / ТЗ."],
    ),
    (
        "1. ОБЩАЯ ИНФОРМАЦИЯ О ТЕНДЕРЕ",
        "Источник финансирования",
        "Какой источник финансирования закупки / контракта?",
        [
            "За счёт каких средств финансируется закупка?",
            "Собственные средства заказчика или иной источник финансирования.",
        ],
    ),

    # ----- 2 -----
    (
        "2. ТРЕБОВАНИЯ К УЧАСТНИКАМ",
        "Требуемая лицензия",
        "Какая лицензия требуется (МЧС, СРО и др.)?",
        ["СРО в строительстве и лицензия МЧС на пожарную безопасность."],
    ),
    (
        "2. ТРЕБОВАНИЯ К УЧАСТНИКАМ",
        "Требования к опыту работы",
        "Какие требования к опыту работы участника (аналогичные договоры, годы, объёмы)?",
        ["Справка об аналогичных договорах, минимальный опыт."],
    ),
    (
        "2. ТРЕБОВАНИЯ К УЧАСТНИКАМ",
        "Минимальный срок работы на рынке",
        "Какой минимальный срок работы на рынке требуется?",
        ["Релевантный опыт не менее N лет."],
    ),
    (
        "2. ТРЕБОВАНИЯ К УЧАСТНИКАМ",
        "Требования к персоналу",
        "Какие требования к персоналу и квалификации указаны?",
        ["Квалификация специалистов, ИТР, аттестованный персонал."],
    ),
    (
        "2. ТРЕБОВАНИЯ К УЧАСТНИКАМ",
        "Перечень документов для заявки",
        "Какой полный перечень документов заявки (формы №…)?",
        ["Формы №1–№5 и иные обязательные документы предложения."],
    ),
    (
        "2. ТРЕБОВАНИЯ К УЧАСТНИКАМ",
        "Размер обеспечения заявки",
        "Какой размер обеспечения заявки (сумма или % от НМЦК)?",
        ["Обеспечение заявки / предложения: размер и способ."],
    ),

    # ----- 3 -----
    (
        "3. ТЕХНИЧЕСКИЕ ТРЕБОВАНИЯ",
        "Тип системы пожаротушения",
        "Какой тип системы пожаротушения (САУГПТ / газовое и т.д.)? Упомяни ЕСУМИС, если есть.",
        ["Система автоматической установки газового пожаротушения."],
    ),
    (
        "3. ТЕХНИЧЕСКИЕ ТРЕБОВАНИЯ",
        "Огнетушащее вещество",
        "Какое ГОТВ / огнетушащее вещество требуется (тип газа, масса заправки)? Ищи в ЛСР и ТЗ.",
        [
            "Тип газового огнетушащего вещества и масса заправки модуля.",
            "Хладон, инертный газ, состав ГОТВ из ЛСР / спецификации.",
        ],
    ),
    (
        "3. ТЕХНИЧЕСКИЕ ТРЕБОВАНИЯ",
        "Модули пожаротушения (перечень)",
        "Перечень модулей пожаротушения (марки, типы, количество) из ЛСР/ВОР/ТЗ.",
        [
            "Модули газового пожаротушения: наименование и количество.",
            "Позиции МГП / модулей из ведомости объёмов и ЛСР.",
        ],
    ),
    (
        "3. ТЕХНИЧЕСКИЕ ТРЕБОВАНИЯ",
        "Оборудование в системе (полный перечень)",
        "Полный перечень оборудования из ЛСР/ВОР/ТЗ/РД с количеством (приборы, датчики, оповещатели, кабели).",
        [
            "Основные позиции оборудования САУГПТ и ЕСУМИС из ЛСР с количеством.",
            "Спецификация оборудования по смете / ВОР.",
        ],
    ),
    (
        "3. ТЕХНИЧЕСКИЕ ТРЕБОВАНИЯ",
        "Нормативные документы (ГОСТ, СП)",
        "Какие конкретные ГОСТ, СП, СНиП, ПУЭ, ФЗ указаны (с номерами)?",
        ["Перечень НТД с номерами из ТЗ и договора."],
    ),
    (
        "3. ТЕХНИЧЕСКИЕ ТРЕБОВАНИЯ",
        "Требования к монтажу",
        "Какие требования к монтажу указаны в ТЗ / договоре?",
        ["Условия СМР, прокладка трасс, размещение оборудования."],
    ),
    (
        "3. ТЕХНИЧЕСКИЕ ТРЕБОВАНИЯ",
        "Требования к пусконаладочным работам (ПНР)",
        "Какие требования к ПНР указаны?",
        ["Состав и сроки пуско-наладки, программа ПНР."],
    ),
    (
        "3. ТЕХНИЧЕСКИЕ ТРЕБОВАНИЯ",
        "Требования по пожарной безопасности",
        "Какие требования по пожарной безопасности указаны?",
        ["Инструктаж, лицензия МЧС, огнезащита, режим на площадке."],
    ),

    # ----- 4 -----
    (
        "4. УСЛОВИЯ КОНТРАКТА",
        "Срок выполнения работ (конечная дата)",
        "Какой срок / конечная дата выполнения работ? Обязательно смотри График производства работ.",
        [
            "Дата начала и окончания работ по графику производства работ.",
            "Продолжительность работ в днях / конечный срок по договору.",
        ],
    ),
    (
        "4. УСЛОВИЯ КОНТРАКТА",
        "Срок гарантии (месяцев)",
        "Какой срок гарантии (в месяцах)?",
        ["Гарантийный срок с даты акта окончания работ."],
    ),
    (
        "4. УСЛОВИЯ КОНТРАКТА",
        "Срок подачи заявок",
        "До какой даты нужно подать заявку / предложение?",
        [
            "Дата окончания срока подачи предложений.",
            "Срок приёма заявок из Извещения.",
        ],
    ),
    (
        "4. УСЛОВИЯ КОНТРАКТА",
        "Время окончания подачи заявок",
        "В какое время (часы:минуты, часовой пояс) заканчивается приём заявок?",
        [
            "Время окончания подачи предложений в Извещении.",
            "Часы и минуты дедлайна подачи оферт.",
        ],
    ),
    (
        "4. УСЛОВИЯ КОНТРАКТА",
        "Срок оплаты (дней)",
        "Какой срок оплаты после сдачи работ? Если в документах разные сроки — укажи оба и источники.",
        ["Срок расчёта в рабочих и/или календарных днях из Извещения и договора."],
    ),
    (
        "4. УСЛОВИЯ КОНТРАКТА",
        "Поэтапное выполнение работ",
        "Предусмотрено ли поэтапное выполнение? Какие этапы и сроки (смотри график и ТЗ)?",
        ["Стадийность СМР и ПНР, сроки этапов."],
    ),
    (
        "4. УСЛОВИЯ КОНТРАКТА",
        "Порядок сдачи-приёмки работ",
        "Какой порядок сдачи-приёмки (акты, сроки)?",
        ["Акт окончания работ, порядок приёмки."],
    ),
    (
        "4. УСЛОВИЯ КОНТРАКТА",
        "Ответственность за просрочку",
        "Какая ответственность за просрочку работ? Ищи ТОЛЬКО в проекте договора: пени, неустойка, %.",
        [
            "Неустойка / пени за нарушение сроков в договоре.",
            "Размер ответственности подрядчика за просрочку исполнения.",
        ],
    ),
    (
        "4. УСЛОВИЯ КОНТРАКТА",
        "Штрафы/пени",
        "Какой размер штрафов и пеней (% в день / сумма) в ДОГОВОРЕ?",
        [
            "Пени за просрочку в процентах от цены договора.",
            "Штрафные санкции по проекту договора СМР-ПНР.",
        ],
    ),
    (
        "4. УСЛОВИЯ КОНТРАКТА",
        "Запрет субподряда",
        "Нужно ли согласие заказчика на субподряд? Запрещён ли субподряд?",
        ["Условия привлечения третьих лиц / субподрядчиков."],
    ),
    (
        "4. УСЛОВИЯ КОНТРАКТА",
        "Возможность расторжения контракта",
        "Основания и порядок расторжения договора (ищи в проекте договора).",
        [
            "Односторонний отказ и расторжение договора.",
            "Условия прекращения договора по инициативе сторон.",
        ],
    ),
    (
        "4. УСЛОВИЯ КОНТРАКТА",
        "Форс-мажорные обстоятельства",
        "Что указано про форс-мажор в договоре?",
        ["Непреодолимая сила: уведомление, сроки, расторжение."],
    ),
    (
        "4. УСЛОВИЯ КОНТРАКТА",
        "Порядок разрешения споров",
        "Какой порядок споров по ДОГОВОРУ (претензия + какой суд)?",
        [
            "Претензионный порядок и арбитражный суд в договоре.",
            "Подсудность споров по проекту договора.",
        ],
    ),
    (
        "4. УСЛОВИЯ КОНТРАКТА",
        "Арбитражная оговорка (суд)",
        "Какой суд указан в договоре для разрешения споров (арбитражный суд какого субъекта)?",
        [
            "Арбитражный суд в договоре (наименование).",
            "Место рассмотрения споров по договору подряда.",
        ],
    ),
    (
        "4. УСЛОВИЯ КОНТРАКТА",
        "Состав ЗИП (запасных частей)",
        "Какой состав ЗИП / запасных частей и инструментов подлежит передаче заказчику? Ищи в ТЗ, ЛСР, РД.",
        [
            "Перечень ЗИП, запасных частей, поставка на склад заказчика.",
            "Комплект запасных изделий и принадлежностей.",
        ],
    ),

    # ----- 5 -----
    (
        "5. ФИНАНСОВЫЕ УСЛОВИЯ",
        "Начальная (максимальная) цена контракта",
        "Какая НМЦК в рублях? Если суммы нет — так и скажи (не подменяй сметой).",
        [
            "НМЦК / начальная максимальная цена из Извещения или закупочной документации.",
            "Цена договора цифрами в извещении (если заполнена).",
        ],
    ),
    (
        "5. ФИНАНСОВЫЕ УСЛОВИЯ",
        "Сметная стоимость (ориентир)",
        "Найди итоговую сумму по ЛСР/ВОР/смете (строки «Итого», «Всего»). Это ориентир, не НМЦК. Укажи сумму и файл-источник.",
        [
            "Итого по локальному сметному расчёту ЛСР САУГПТ / ЕСУМИС.",
            "Всего по смете / ВОР в рублях с НДС или без.",
        ],
    ),
    (
        "5. ФИНАНСОВЫЕ УСЛОВИЯ",
        "Сводка из ЛСР (топ позиций)",
        "По ЛСР/ВОР выпиши топ-10 позиций оборудования/работ с наименованием и количеством (и ценой, если есть). В конце — строка Итого по смете.",
        [
            "Крупнейшие позиции спецификации из ЛСР САУГПТ и ЕСУМИС.",
            "Перечень основных материалов и оборудования из ведомости объёмов.",
        ],
    ),
    (
        "5. ФИНАНСОВЫЕ УСЛОВИЯ",
        "Порядок оплаты",
        "Какой порядок оплаты (по факту, удержания, этапы)? Если сроки в документах различаются — укажи оба.",
        ["Условия оплаты и гарантийные удержания."],
    ),
    (
        "5. ФИНАНСОВЫЕ УСЛОВИЯ",
        "Аванс (если есть)",
        "Предусмотрен ли аванс? Размер и условия (банковская гарантия)?",
        ["Максимальный процент аванса."],
    ),
    (
        "5. ФИНАНСОВЫЕ УСЛОВИЯ",
        "Обеспечение заявки",
        "Размер и способ обеспечения заявки (отдельно от обеспечения договора)?",
        ["Обеспечение заявки / предложения."],
    ),
    (
        "5. ФИНАНСОВЫЕ УСЛОВИЯ",
        "Обеспечение контракта (размер)",
        "Размер обеспечения исполнения договора (БГ, депозит, %). Не путай с гарантийным удержанием.",
        ["Обеспечение исполнения обязательств по договору."],
    ),
    (
        "5. ФИНАНСОВЫЕ УСЛОВИЯ",
        "Возможность снижения цены",
        "Есть ли условие снижения цены контракта?",
        ["Изменение (уменьшение) цены договора."],
    ),
    (
        "5. ФИНАНСОВЫЕ УСЛОВИЯ",
        "Критерии оценки заявок",
        "Какие критерии оценки заявок (цена, квалификация, веса)?",
        ["Как выбирается победитель запроса предложений?"],
    ),
]

SECTION6_FIELDS: List[Tuple[str, str]] = [
    ("6. РИСКИ И РЕКОМЕНДАЦИИ", "Основные риски для исполнителя"),
    ("6. РИСКИ И РЕКОМЕНДАЦИИ", "Риски по срокам"),
    ("6. РИСКИ И РЕКОМЕНДАЦИИ", "Риски по цене"),
    ("6. РИСКИ И РЕКОМЕНДАЦИИ", "Риски по качеству"),
    ("6. РИСКИ И РЕКОМЕНДАЦИИ", "Что не указано в документах"),
    ("6. РИСКИ И РЕКОМЕНДАЦИИ", "Что проверить перед подачей заявки"),
    ("6. РИСКИ И РЕКОМЕНДАЦИИ", "Что можно улучшить в документации"),
    ("6. РИСКИ И РЕКОМЕНДАЦИИ", "Рекомендация по участию"),
    ("6. РИСКИ И РЕКОМЕНДАЦИИ", "Ключевые выводы"),
]

SECTION6_PROMPTS = {
    "Основные риски для исполнителя":
        "На основе собранных фактов перечисли основные риски для исполнителя.",
    "Риски по срокам":
        "Оцени риски по срокам выполнения работ и подачи заявки.",
    "Риски по цене":
        "Оцени финансовые риски: НМЦК, сметный ориентир, оплата, аванс, удержания, обеспечение, пени.",
    "Риски по качеству":
        "Оцени риски по качеству, ТЗ, лицензиям, гарантии, составу оборудования.",
    "Что не указано в документах":
        "Что важное отсутствует или неполно в собранных ответах?",
    "Что проверить перед подачей заявки":
        "Чек-лист обязательных проверок перед подачей заявки.",
    "Что можно улучшить в документации":
        "Недостатки закупочной документации, которые можно улучшить.",
    "Рекомендация по участию":
        "Рекомендация: участвовать / осторожно / не участвовать — с обоснованием.",
    "Ключевые выводы":
        "Ключевые выводы по тендеру в 5–8 пунктах.",
}


def build_section6(answers: Dict[str, str]) -> Dict[str, str]:
    facts = "\n".join(
        f"[{section}] {field}: {answers.get(field, 'Не указано')}"
        for section, field, _q, _v in REPORT_FIELDS
    )
    results = {}
    total = len(SECTION6_FIELDS)
    for i, (_section, field) in enumerate(SECTION6_FIELDS, start=1):
        print(f"[S6 {i}/{total}] Анализ: {field}...")
        prompt = (
            "Ты эксперт по тендерному анализу. Ниже — уже извлечённые факты. "
            "Опирайся ТОЛЬКО на них. Не противоречь установленным фактам. "
            "Если НМЦК нет, но есть сметный ориентир — различай их.\n\n"
            f"Вопрос: {SECTION6_PROMPTS[field]}"
        )
        results[field] = _clean_answer(ask_deepseek(prompt, context=facts))
    return results


def postprocess_nmck_and_estimate(answers: Dict[str, str], sources_used: Dict[str, List[Dict[str, Any]]]):
    """Если НМЦК пуста — усиленно ищем сметный итог в ЛСР/ВОР."""
    nmck = answers.get("Начальная (максимальная) цена контракта", "Не указано")
    estimate = answers.get("Сметная стоимость (ориентир)", "Не указано")

    if estimate == "Не указано" or nmck == "Не указано":
        print("   ↳ Доп. поиск итоговых сумм в ЛСР/ВОР...")
        q = (
            "Найди строки «Итого», «Всего по смете», итоговые суммы в рублях в ЛСР и ВОР. "
            "Перечисли найденные суммы с указанием файла. Это сметный ориентир, не НМЦК."
        )
        ans, hits = ask_with_variants(
            q,
            variants=[
                "Итоговая стоимость локального сметного расчёта САУГПТ и ЕСУМИС.",
                "Всего с НДС / без НДС по ЛСР.",
            ],
            source_masks=MASK_LSR,
            mask_only=True,
            top_k=TOP_K,
        )
        if ans != "Не указано":
            labeled = (
                f"Сметная стоимость (ориентир, НЕ НМЦК): {ans}"
            )
            answers["Сметная стоимость (ориентир)"] = labeled
            record_sources("Сметная стоимость (ориентир)", hits, sources_used)
            if nmck == "Не указано":
                answers["Начальная (максимальная) цена контракта"] = (
                    "Не указано как НМЦК. См. поле «Сметная стоимость (ориентир)» "
                    "(сумма из ЛСР/ВОР — ориентир, не начальная максимальная цена контракта)."
                )


# =============================================================================
# Формат отчёта
# =============================================================================

def format_report(
    answers: Dict[str, str],
    tender_no: str,
    sources_used: Dict[str, List[Dict[str, Any]]],
) -> str:
    n_docs = len(set(_basename_key(s) for s in DEDUP_SOURCES))
    n_chunks = len(DEDUP_CHUNKS)
    date_str = ANALYSIS_DATE.strftime("%Y-%m-%d %H:%M")

    lines: List[str] = []
    lines.append("=" * 60)
    lines.append("📊 АНАЛИЗ ТЕНДЕРНОЙ ДОКУМЕНТАЦИИ")
    lines.append("=" * 60)
    lines.append(f"Номер тендера / ИКЗ: {tender_no or answers.get('Номер тендера / ИКЗ', 'Не указано')}")
    lines.append(f"Дата анализа: {date_str}")
    lines.append(f"Всего документов (уник. после дедупа): {n_docs}")
    lines.append(f"Всего чанков (после дедупа): {n_chunks}")
    lines.append(f"Чанков в исходном индексе: {len(rag_index.chunks)}")
    lines.append("")

    current_section = None
    for section, field, _q, _v in REPORT_FIELDS:
        if section != current_section:
            if current_section is not None:
                lines.append("")
            lines.append("=" * 60)
            lines.append(section)
            lines.append("=" * 60)
            current_section = section
        lines.append(f"{field}: {answers.get(field, 'Не указано')}")

    lines.append("")
    lines.append("=" * 60)
    lines.append("6. РИСКИ И РЕКОМЕНДАЦИИ")
    lines.append("=" * 60)
    for _section, field in SECTION6_FIELDS:
        lines.append(f"{field}: {answers.get(field, 'Не указано')}")

    lines.append("")
    lines.append("=" * 60)
    lines.append("7. ИСПОЛЬЗОВАННЫЕ ИСТОЧНИКИ ПО ПУНКТАМ")
    lines.append("=" * 60)
    for field, items in sources_used.items():
        if not items:
            lines.append(f"{field}: источники не найдены")
            continue
        lines.append(f"{field}:")
        for it in items[:8]:
            chunk = f", чанк #{it['chunk']}" if it.get("chunk") else ""
            lines.append(f"  • {it['source']}{chunk} (score={it.get('score', 0)})")

    all_sources = sorted(
        {
            it["source"]
            for items in sources_used.values()
            for it in items
            if not str(it["source"]).startswith("(")
        },
        key=lambda s: _basename_key(s),
    )
    unique_bases = sorted({_basename_key(s) for s in all_sources})
    all_bases = sorted({_basename_key(s) for s in DEDUP_SOURCES})
    unused = [b for b in all_bases if b not in unique_bases]

    lines.append("")
    lines.append("Уникальные файлы-источники (basename):")
    for b in unique_bases:
        lines.append(f"  • {b}")
    lines.append("")
    lines.append("Файлы индекса, НЕ использованные в ответах (basename):")
    if unused:
        for b in unused:
            lines.append(f"  • {b}")
    else:
        lines.append("  • (нет — все basename задействованы)")

    lines.append("=" * 60)
    return "\n".join(lines) + "\n"


# =============================================================================
# Запуск
# =============================================================================

print("🚀 Автоматический анализ тендерной документации (модуль 4, post-QA)")
print(f"   Уник. документов (basename): {len(set(_basename_key(s) for s in DEDUP_SOURCES))}")
print(f"   Чанков после дедупа: {len(DEDUP_CHUNKS)}")
print(f"   Пунктов RAG (1–5): {len(REPORT_FIELDS)}")
print(f"   Пунктов раздела 6: {len(SECTION6_FIELDS)}")
print(f"   top_k = {TOP_K}")
print()

answers: Dict[str, str] = {}
sources_used: Dict[str, List[Dict[str, Any]]] = {}

# --- Номер из имён файлов ---
print("[0] Поиск номера тендера в именах файлов...")
tender_no = extract_tender_number_from_filenames()
if tender_no:
    print(f"   ✅ Номер из имени файла: {tender_no}")
    answers["Номер тендера / ИКЗ"] = tender_no
    sources_used["Номер тендера / ИКЗ"] = [
        {"source": "(имя загруженного файла)", "chunk": None, "score": 1.0}
    ]
else:
    print("   ⚠️ В именах не найден — будет RAG-запрос")

# --- Разделы 1–5 ---
total_rag = len(REPORT_FIELDS)
for i, (section, field, question, variants) in enumerate(REPORT_FIELDS, start=1):
    print(f"[{i}/{total_rag}] Обработка: {field}...")

    if field == "Номер тендера / ИКЗ" and tender_no:
        print("   ↳ использован номер из имени файла")
        continue

    masks = FIELD_SOURCE_MASKS.get(field)
    # пени / расторжение / арбитраж — ТОЛЬКО договор
    mask_only = field in {
        "Штрафы/пени",
        "Ответственность за просрочку",
        "Возможность расторжения контракта",
        "Арбитражная оговорка (суд)",
        "Сметная стоимость (ориентир)",
        "Сводка из ЛСР (топ позиций)",
    }

    answer, hits = ask_with_variants(
        question,
        variants=variants,
        top_k=TOP_K,
        source_masks=masks,
        mask_only=mask_only,
        retry_if_empty=True,
    )
    answers[field] = answer
    record_sources(field, hits, sources_used)

    if field == "Номер тендера / ИКЗ" and not tender_no:
        extracted = extract_tender_number_from_text(answer)
        if extracted:
            tender_no = extracted
            answers[field] = extracted
            print(f"   ✅ Номер из RAG: {tender_no}")

# --- Доп. обработка НМЦК / сметы ---
print("\n💰 Проверка НМЦК и сметного ориентира...")
postprocess_nmck_and_estimate(answers, sources_used)

# --- Раздел 6 ---
print("\n📊 Формирование раздела 6 (риски) на основе собранных фактов...")
section6 = build_section6(answers)
answers.update(section6)
for field in section6:
    sources_used[field] = [
        {"source": "(синтез ответов разделов 1–5)", "chunk": None, "score": 1.0}
    ]

# --- Сохранение ---
report_text = format_report(answers, tender_no, sources_used)
report_filename = build_report_filename(tender_no)

with open(report_filename, "w", encoding="utf-8") as f:
    f.write(report_text)

tender_report = report_text
tender_report_filename = report_filename
tender_number = tender_no

if colab_files is not None:
    colab_files.download(report_filename)

processed = len(REPORT_FIELDS) + len(SECTION6_FIELDS)
print()
print("✅ Отчёт сформирован!")
print(f"📁 Файл: {report_filename}")
print(f"📊 Всего обработано: {processed} пунктов")
if tender_no:
    print(f"🔖 Номер тендера: {tender_no}")
if colab_files is not None:
    print("📥 Файл скачан.")
else:
    print(f"💾 Файл сохранён локально: {os.path.abspath(report_filename)}")
