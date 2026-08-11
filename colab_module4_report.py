# =============================================================================
# МОДУЛЬ 4 — Полный анализ тендера: 86 вопросов по ВСЕМ файлам
# Выполняйте строго ПОСЛЕ модуля 3.
# Использует: rag_index, ask_deepseek, uploaded_files (модули 1–2).
# =============================================================================

import hashlib
import os
import re
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

try:
    from google.colab import files as colab_files
except ImportError:
    colab_files = None
    print("⚠️ google.colab недоступен — файл сохранится локально без автоскачивания.")

if "rag_index" not in globals() or rag_index is None:
    raise RuntimeError("❌ rag_index не найден. Сначала выполните модули 1–3.")
if not getattr(rag_index, "chunks", None):
    raise RuntimeError("❌ Индекс пуст. Сначала выполните модуль 3.")
if "ask_deepseek" not in globals():
    raise RuntimeError("❌ ask_deepseek не найден. Сначала выполните модуль 1.")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

ANALYSIS_DATE = datetime.now()
TOP_K = 12
TOP_K_FORCED = 16
MAX_CTX = 16000

# =============================================================================
# Ожидаемые 14 файлов
# =============================================================================

EXPECTED_FILES = [
    "Извещение.docx",
    "Закупочная документация.docx",
    "Приложение № 1 - Техническое задание.doc",
    "Приложение № 3 - График производства работ.doc",
    "Приложение № 4 - График освоения и финансирования денежных средств.docx",
    "Приложение № 5 - Акт окончания работ.docx",
    "Проект Договора СМР-ПНР по САУГПТ и ЕСУМИС.docx",
    "Техническое задание.pdf",
    "ВОР Раздел ПД №12 ЛСР (02-01-01) САУГПТ.xlsx",
    "ВОР Раздел ПД №12 ЛСР (02-01-02) ЕСУМИС.xlsx",
    "Приложение № 5 к ТЗ - ЛСР САУГПТ.xlsx",
    "Приложение № 6 к ТЗ - ЛСР ЕСУМИС.xlsx",
    "14-27-00987 от 03.02.2026 РД САУГПТ.pdf",
    "14-27-03537 от 20.04.2026 РД ЕСУМИС.pdf",
]

MASK_TECH = ("рд", "14-27-", "техническое задание", "техзадани", "/тз", "тз.", "рабочая документация", "саугпт", "есумис")
MASK_ESTIMATE = ("лср", "вор")
MASK_CONTRACT = ("договор",)
MASK_NOTICE = ("извещение", "закупочная")
MASK_SCHEDULE = ("график",)
MASK_FINANCE_SCHED = ("освоения", "финансирования")

# =============================================================================
# Дедуп индекса
# =============================================================================

def _basename(source: str) -> str:
    name = str(source).replace("\\", "/").split("/")[-1]
    name = re.sub(r"\s*\(\d+\)(?=\.\w+$)", "", name)
    return name


def _norm(name: str) -> str:
    return re.sub(r"\s+", " ", _basename(name).lower().strip())


def _hash(text: str) -> str:
    return hashlib.md5(re.sub(r"\s+", " ", (text or "").strip().lower()).encode("utf-8", errors="ignore")).hexdigest()


def _match_expected(actual: str, expected: str) -> bool:
    a, e = _norm(actual), _norm(expected)
    if a == e or e in a or a in e:
        return True
    m = re.search(r"14-27-\d+", e)
    if m and m.group(0) in a:
        return True
    pairs = [
        ("график производства", "график производства"),
        ("график освоения", "график освоения"),
        ("акт окончания", "акт окончания"),
        ("закупочная документация", "закупочная документация"),
        ("извещение", "извещение"),
    ]
    for pe, pa in pairs:
        if pe in e and pa in a:
            return True
    if "лср" in e and "лср" in a:
        if ("саугпт" in e and "саугпт" in a) or ("есумис" in e and "есумис" in a):
            return True
    if "вор" in e and "вор" in a:
        if ("саугпт" in e and "саугпт" in a) or ("есумис" in e and "есумис" in a):
            return True
    if "проект договора" in e and "договор" in a and "смр" in a:
        return True
    if "приложение № 1" in e and "техническое задание" in a and a.endswith(".doc"):
        return True
    if e.startswith("техническое задание") and a.startswith("техническое задание") and a.endswith(".pdf"):
        return True
    return False


def build_dedup():
    items = list(enumerate(zip(rag_index.chunks, rag_index.sources)))

    def rank(src: str):
        s = src.lower()
        pen = (2 if "процедуре" in s else 0) + (1 if s.count(".zip/") + s.count(".rar/") > 1 else 0)
        return (pen, len(s))

    items.sort(key=lambda it: rank(it[1][1]))
    seen: Set[str] = set()
    chunks, sources, orig = [], [], []
    for i, (t, s) in items:
        h = _hash(t)
        if h in seen:
            continue
        seen.add(h)
        chunks.append(t)
        sources.append(s)
        orig.append(i)
    return chunks, sources, orig


print("🔧 Дедупликация индекса...")
D_CHUNKS, D_SOURCES, D_ORIG = build_dedup()
print(f"   Чанков: {len(rag_index.chunks)} → {len(D_CHUNKS)}")
print(f"   Уник. basename: {len({_norm(s) for s in D_SOURCES})}")

_VECT = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True)
_MAT = _VECT.fit_transform(D_CHUNKS) if D_CHUNKS else None

# группы по basename
FILE_GROUPS: Dict[str, Dict[str, Any]] = {}
for i, (t, s) in enumerate(zip(D_CHUNKS, D_SOURCES)):
    key = _norm(s)
    g = FILE_GROUPS.setdefault(key, {"display": _basename(s), "paths": set(), "idxs": [], "chars": 0})
    g["paths"].add(s)
    g["idxs"].append(i)
    g["chars"] += len(t)


def map_expected() -> List[Tuple[str, Optional[str]]]:
    used: Set[str] = set()
    out: List[Tuple[str, Optional[str]]] = []
    for exp in EXPECTED_FILES:
        found = None
        for key, g in FILE_GROUPS.items():
            if key in used:
                continue
            if _match_expected(key, exp) or _match_expected(g["display"], exp):
                found = key
                break
        if found:
            used.add(found)
        out.append((exp, found))
    for key in FILE_GROUPS:
        if key not in used:
            out.append((FILE_GROUPS[key]["display"], key))
    return out


FILE_PLAN = map_expected()


# =============================================================================
# Поиск
# =============================================================================

def _mask_hit(source: str, masks: Sequence[str]) -> bool:
    s = source.lower().replace("\\", "/")
    b = _norm(source)
    return any(m.lower() in s or m.lower() in b for m in masks)


def search(
    query: str,
    top_k: int = TOP_K,
    masks: Optional[Sequence[str]] = None,
    mask_only: bool = False,
) -> List[Dict[str, Any]]:
    if _MAT is None:
        return []
    q = _VECT.transform([query])
    scores = cosine_similarity(q, _MAT).ravel()
    out = []
    for i, sc in enumerate(scores):
        src = D_SOURCES[i]
        matched = _mask_hit(src, masks) if masks else False
        if mask_only and masks and not matched:
            continue
        boost = 0.5 if matched else 0.0
        if not masks:
            for kw, b in (("извещение", 0.3), ("закупочная", 0.28), ("договор", 0.28),
                          ("лср", 0.4), ("вор", 0.4), ("график", 0.35), ("рд", 0.25),
                          ("техническое задание", 0.3)):
                if kw in src.lower():
                    boost = max(boost, b)
        sc = float(sc) + boost
        if sc <= 0:
            continue
        out.append({
            "text": D_CHUNKS[i],
            "source": src,
            "score": sc,
            "chunk": D_ORIG[i] + 1,
            "base": _basename(src),
        })
    out.sort(key=lambda x: x["score"], reverse=True)
    return out[:top_k]


def merge_hits(lists: List[List[Dict[str, Any]]], top_k: int = TOP_K) -> List[Dict[str, Any]]:
    best: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for hits in lists:
        for h in hits:
            key = (h["source"], h["text"][:160])
            if key not in best or h["score"] > best[key]["score"]:
                best[key] = h
    return sorted(best.values(), key=lambda x: x["score"], reverse=True)[:top_k]


def fmt_ctx(hits: List[Dict[str, Any]], max_chars: int = MAX_CTX) -> str:
    parts, n = [], 0
    for i, h in enumerate(hits, 1):
        block = f"[Фрагмент {i} | {h['source']} | чанк #{h.get('chunk')} | {h['score']:.3f}]\n{h['text']}"
        if n + len(block) > max_chars and parts:
            break
        parts.append(block)
        n += len(block)
    return "\n\n---\n\n".join(parts)


def _empty(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    low = t.lower()
    marks = ["не указано", "не найдено", "нет информации", "в контексте нет",
             "информация отсутствует", "не удалось", "ответить невозможно"]
    if len(t) < 200 and any(m in low for m in marks):
        if re.search(r"\d{3,}", t) and ("руб" in low or "%" in t or "инн" in low):
            return False
        return True
    return False


def clean(text: str) -> str:
    text = (text or "").strip()
    return "Не указано" if _empty(text) else text


PRIORITY_PROMPT = (
    "Отвечай только по контексту. Выпиши конкретные факты: числа, даты, суммы, перечни, нормы. "
    "Не пиши «не указано», если в контексте есть хотя бы частичный ответ. "
    "Если данные противоречивы — укажи оба варианта и источники. Ответ на русском."
)


def ask_rag(
    question: str,
    variants: Optional[List[str]] = None,
    masks: Optional[Sequence[str]] = None,
    mask_only: bool = False,
    top_k: int = TOP_K,
) -> Tuple[str, List[Dict[str, Any]]]:
    variants = variants or []
    queries = [question] + variants
    lists = []
    for q in queries:
        lists.append(search(q, top_k=top_k, masks=masks, mask_only=False))
        if masks:
            lists.append(search(q, top_k=TOP_K_FORCED, masks=masks, mask_only=True))
    hits = merge_hits(lists, top_k=top_k)
    if not hits:
        return "Не указано", []
    ans = clean(ask_deepseek(f"{PRIORITY_PROMPT}\n\nВопрос: {question}", context=fmt_ctx(hits)))
    if ans == "Не указано" and (variants or masks):
        rq = (variants[-1] if variants else question) + " Приведи любые найденные факты из контекста."
        rh = merge_hits(
            [search(rq, top_k=top_k, masks=masks, mask_only=bool(masks)), hits],
            top_k=top_k,
        )
        if rh:
            ans2 = clean(ask_deepseek(f"{PRIORITY_PROMPT}\n\nВопрос: {rq}", context=fmt_ctx(rh)))
            if ans2 != "Не указано":
                return ans2, rh
    return ans, hits


# =============================================================================
# 86 вопросов: (id, section, title, question, variants, masks, mask_only, kind)
# kind: rag | meta | synthesize
# =============================================================================

# Вопросы 67-78 и 85-86 — synthesize на основе предыдущих ответов
# 79-84 — meta по статистике файлов

QUESTIONS: List[Dict[str, Any]] = []

def Q(num, section, title, question, variants=None, masks=None, mask_only=False, kind="rag"):
    QUESTIONS.append({
        "num": num,
        "section": section,
        "title": title,
        "question": question,
        "variants": variants or [],
        "masks": masks,
        "mask_only": mask_only,
        "kind": kind,
    })

# --- 1. Общая информация ---
Q(1, "1. ОБЩАЯ ИНФОРМАЦИЯ", "Номер тендера / ИКЗ",
  "Каков полный номер тендера / извещения / процедуры / ИКЗ? Укажи номер целиком (включая все цифры после B).",
  ["Номер процедуры вида B… из Извещения.", "ИКЗ или номер закупки."], MASK_NOTICE)
Q(2, "1. ОБЩАЯ ИНФОРМАЦИЯ", "Полное название объекта закупки",
  "Каково полное название / наименование объекта закупки или предмета лота?",
  ["Наименование закупки / лота из Извещения."], MASK_NOTICE)
Q(3, "1. ОБЩАЯ ИНФОРМАЦИЯ", "Краткое описание проекта (аннотация)",
  "Кратко опиши проект / закупку (аннотация 3–6 предложений): что делается и зачем.",
  ["Суть работ САУГПТ и ЕСУМИС."], MASK_NOTICE + MASK_TECH)
Q(4, "1. ОБЩАЯ ИНФОРМАЦИЯ", "Заказчик (название)",
  "Кто заказчик? Полное наименование организации.",
  ["Наименование заказчика."], MASK_NOTICE)
Q(5, "1. ОБЩАЯ ИНФОРМАЦИЯ", "ИНН заказчика",
  "Какой ИНН заказчика?",
  ["ИНН организации-заказчика."], MASK_NOTICE)
Q(6, "1. ОБЩАЯ ИНФОРМАЦИЯ", "Контакты заказчика (телефон, email, адрес)",
  "Контакты заказчика: телефоны, email, адрес, контактные лица.",
  ["Телефон и email заказчика, ФИО контактов."], MASK_NOTICE)
Q(7, "1. ОБЩАЯ ИНФОРМАЦИЯ", "Адрес объекта",
  "Какой адрес объекта?",
  ["Адрес площадки / порта / сооружения."], MASK_NOTICE + MASK_TECH)
Q(8, "1. ОБЩАЯ ИНФОРМАЦИЯ", "Место выполнения работ",
  "Каково место выполнения работ?",
  ["Место производства работ."], MASK_NOTICE + MASK_TECH)
Q(9, "1. ОБЩАЯ ИНФОРМАЦИЯ", "Объект (здание/помещение)",
  "В каком здании / помещении выполняются работы (пункт пропуска и т.п.)?",
  ["Наименование здания / сооружения объекта."], MASK_TECH + MASK_NOTICE)
Q(10, "1. ОБЩАЯ ИНФОРМАЦИЯ", "Источник финансирования",
  "Какой источник финансирования?",
  ["Собственные средства заказчика или иной источник."], MASK_NOTICE)
Q(11, "1. ОБЩАЯ ИНФОРМАЦИЯ", "Вид процедуры (запрос предложений, аукцион, конкурс)",
  "Какой вид закупочной процедуры?",
  ["Открытый запрос предложений / аукцион / конкурс."], MASK_NOTICE)

# --- 2. Требования к участникам ---
Q(12, "2. ТРЕБОВАНИЯ К УЧАСТНИКАМ", "Требуемая лицензия (СРО, МЧС)",
  "Какие лицензии и допуски нужны (СРО, МЧС)?",
  ["Лицензия МЧС и членство в СРО."], MASK_NOTICE)
Q(13, "2. ТРЕБОВАНИЯ К УЧАСТНИКАМ", "Требования к опыту работы",
  "Какие требования к опыту работы / аналогичным договорам?",
  ["Справка об аналогичных договорах."], MASK_NOTICE)
Q(14, "2. ТРЕБОВАНИЯ К УЧАСТНИКАМ", "Минимальный срок работы на рынке",
  "Какой минимальный срок работы на рынке / опыт в годах?",
  ["Релевантный опыт не менее N лет."], MASK_NOTICE)
Q(15, "2. ТРЕБОВАНИЯ К УЧАСТНИКАМ", "Требования к персоналу",
  "Какие требования к персоналу и квалификации?",
  ["Квалификация специалистов, ИТР."], MASK_NOTICE)
Q(16, "2. ТРЕБОВАНИЯ К УЧАСТНИКАМ", "Требования к финансовым ресурсам",
  "Какие требования к финансовым ресурсам участника?",
  ["Финансовая устойчивость, обороты, справки банка."], MASK_NOTICE)
Q(17, "2. ТРЕБОВАНИЯ К УЧАСТНИКАМ", "Требования к материально-техническим ресурсам",
  "Какие требования к материально-техническим ресурсам?",
  ["Справка о МТР, оборудование участника."], MASK_NOTICE)
Q(18, "2. ТРЕБОВАНИЯ К УЧАСТНИКАМ", "Требования к деловой репутации",
  "Какие требования к деловой репутации / РНП / отказам от договоров?",
  ["Реестр недобросовестных поставщиков, отказы от заключения."], MASK_NOTICE)
Q(19, "2. ТРЕБОВАНИЯ К УЧАСТНИКАМ", "Перечень документов для заявки",
  "Какой перечень документов заявки (формы №…)?",
  ["Формы №1–№5 и иные документы предложения."], MASK_NOTICE)
Q(20, "2. ТРЕБОВАНИЯ К УЧАСТНИКАМ", "Размер обеспечения заявки",
  "Какой размер обеспечения заявки?",
  ["Обеспечение предложения: сумма или %."], MASK_NOTICE)
Q(21, "2. ТРЕБОВАНИЯ К УЧАСТНИКАМ", "Запрет/ограничения на участие",
  "Какие запреты и ограничения на участие в закупке?",
  ["Ограничения для участников, аффилированность, санкции."], MASK_NOTICE)

# --- 3. Техника (приоритет РД/ТЗ) ---
Q(22, "3. ТЕХНИЧЕСКИЕ ТРЕБОВАНИЯ", "Тип системы пожаротушения",
  "Какой тип системы пожаротушения (САУГПТ и др.)? Упомяни ЕСУМИС.",
  ["Газовое пожаротушение САУГПТ."], MASK_TECH, False)
Q(23, "3. ТЕХНИЧЕСКИЕ ТРЕБОВАНИЯ", "Огнетушащее вещество (ГОТВ)",
  "Какое ГОТВ / огнетушащее вещество (тип, масса заправки)? Ищи в ТЗ, ЛСР, РД.",
  ["Тип газа ГОТВ и масса заправки модуля."], MASK_TECH + MASK_ESTIMATE, False)
Q(24, "3. ТЕХНИЧЕСКИЕ ТРЕБОВАНИЯ", "Модули пожаротушения (перечень)",
  "Перечень модулей пожаротушения: марки, типы, количество (из ЛСР/ВОР/ТЗ/РД).",
  ["Модули МГП / газового ПТ с количеством."], MASK_TECH + MASK_ESTIMATE, False)
Q(25, "3. ТЕХНИЧЕСКИЕ ТРЕБОВАНИЯ", "Оборудование в системе (полный перечень)",
  "Полный перечень оборудования САУГПТ и ЕСУМИС с количествами из ЛСР/ВОР/ТЗ/РД.",
  ["Спецификация оборудования по смете и РД."], MASK_TECH + MASK_ESTIMATE, False)
Q(26, "3. ТЕХНИЧЕСКИЕ ТРЕБОВАНИЯ", "Технические характеристики оборудования",
  "Какие технические характеристики оборудования указаны (напряжение, интерфейсы, параметры)?",
  ["Характеристики приборов и модулей из ТЗ/РД."], MASK_TECH, False)
Q(27, "3. ТЕХНИЧЕСКИЕ ТРЕБОВАНИЯ", "Схемы и чертежи (наличие, количество)",
  "Какие схемы и чертежи есть в РД/ТЗ (наличие, обозначения, количество листов/альбомов)?",
  ["Альбомы РД, количество чертежей, обозначения."], MASK_TECH, False)
Q(28, "3. ТЕХНИЧЕСКИЕ ТРЕБОВАНИЯ", "Спецификации оборудования (марки, модели)",
  "Какие марки и модели оборудования указаны в спецификациях?",
  ["Марки приборов: Рубеж и др."], MASK_TECH + MASK_ESTIMATE, False)
Q(29, "3. ТЕХНИЧЕСКИЕ ТРЕБОВАНИЯ", "Нормативные документы (ГОСТ, СП, СНиП)",
  "Какие ГОСТ, СП, СНиП, ПУЭ, ФЗ указаны (с номерами)?",
  ["Перечень НТД с номерами."], MASK_TECH + MASK_CONTRACT, False)
Q(30, "3. ТЕХНИЧЕСКИЕ ТРЕБОВАНИЯ", "Требования к монтажу",
  "Какие требования к монтажу?",
  ["Условия СМР, размещение оборудования."], MASK_TECH + MASK_CONTRACT, False)
Q(31, "3. ТЕХНИЧЕСКИЕ ТРЕБОВАНИЯ", "Требования к кабельным трассам",
  "Какие требования к кабельным трассам / прокладке кабелей?",
  ["Кабельные трассы, типы кабелей, способы прокладки."], MASK_TECH, False)
Q(32, "3. ТЕХНИЧЕСКИЕ ТРЕБОВАНИЯ", "Требования к заземлению и молниезащите",
  "Какие требования к заземлению и молниезащите?",
  ["Заземление, уравнивание потенциалов, молниезащита."], MASK_TECH, False)
Q(33, "3. ТЕХНИЧЕСКИЕ ТРЕБОВАНИЯ", "Требования к автоматизации и диспетчеризации",
  "Какие требования к автоматизации, диспетчеризации, ЕСУМИС / мониторингу?",
  ["Диспетчеризация и удалённый мониторинг."], MASK_TECH, False)
Q(34, "3. ТЕХНИЧЕСКИЕ ТРЕБОВАНИЯ", "Требования к ПНР (пусконаладочным работам)",
  "Какие требования к ПНР?",
  ["Состав и программа пуско-наладки."], MASK_TECH + MASK_CONTRACT, False)
Q(35, "3. ТЕХНИЧЕСКИЕ ТРЕБОВАНИЯ", "Требования к испытаниям и проверкам",
  "Какие требования к испытаниям, проверкам, сдаче систем?",
  ["Испытания, комплексное опробование, проверки."], MASK_TECH + MASK_CONTRACT, False)
Q(36, "3. ТЕХНИЧЕСКИЕ ТРЕБОВАНИЯ", "Требования по пожарной безопасности",
  "Какие требования по пожарной безопасности на объекте / при работах?",
  ["Инструктаж, режим ПБ, огнезащита."], MASK_TECH + MASK_CONTRACT + MASK_NOTICE, False)
Q(37, "3. ТЕХНИЧЕСКИЕ ТРЕБОВАНИЯ", "Требования к гарантийному обслуживанию",
  "Какие требования к гарантийному обслуживанию / сервису?",
  ["Гарантийное обслуживание оборудования и работ."], MASK_CONTRACT + MASK_TECH, False)

# --- 4. Условия контракта (приоритет Договор) ---
Q(38, "4. УСЛОВИЯ КОНТРАКТА", "Срок выполнения работ (этапы)",
  "Каковы этапы и сроки выполнения работ? Смотри график производства работ и ТЗ.",
  ["Стадийность и сроки этапов СМР/ПНР."], MASK_SCHEDULE + MASK_TECH + MASK_CONTRACT, False)
Q(39, "4. УСЛОВИЯ КОНТРАКТА", "Срок выполнения работ (конечная дата)",
  "Какова конечная дата / общий срок выполнения работ по графику или договору?",
  ["Дата окончания работ, продолжительность."], MASK_SCHEDULE + MASK_CONTRACT + MASK_NOTICE, False)
Q(40, "4. УСЛОВИЯ КОНТРАКТА", "Срок гарантии",
  "Каков срок гарантии (месяцев) и точка отсчёта?",
  ["Гарантия с даты акта окончания работ."], MASK_CONTRACT + MASK_NOTICE, False)
Q(41, "4. УСЛОВИЯ КОНТРАКТА", "Срок подачи заявок",
  "До какой даты подаются заявки?",
  ["Дата окончания подачи предложений."], MASK_NOTICE, False)
Q(42, "4. УСЛОВИЯ КОНТРАКТА", "Время окончания подачи заявок",
  "В какое время (часы:минуты) заканчивается приём заявок?",
  ["Время дедлайна подачи оферт."], MASK_NOTICE, False)
Q(43, "4. УСЛОВИЯ КОНТРАКТА", "Срок оплаты",
  "Каков срок оплаты? Если разные в Извещении и договоре — укажи оба.",
  ["Срок расчёта в рабочих/календарных днях."], MASK_NOTICE + MASK_CONTRACT, False)
Q(44, "4. УСЛОВИЯ КОНТРАКТА", "Поэтапное выполнение работ",
  "Предусмотрено ли поэтапное выполнение? Перечень этапов.",
  ["Этапы работ по ТЗ и графику."], MASK_SCHEDULE + MASK_TECH + MASK_CONTRACT, False)
Q(45, "4. УСЛОВИЯ КОНТРАКТА", "Порядок сдачи-приёмки работ",
  "Какой порядок сдачи-приёмки (акты, сроки)?",
  ["Акт окончания работ, приёмка."], MASK_CONTRACT, False)
Q(46, "4. УСЛОВИЯ КОНТРАКТА", "Ответственность за просрочку",
  "Какая ответственность за просрочку? Ищи в договоре.",
  ["Неустойка за просрочку исполнения."], MASK_CONTRACT, True)
Q(47, "4. УСЛОВИЯ КОНТРАКТА", "Штрафы и пени",
  "Какой размер штрафов и пеней в договоре (%/суммы)?",
  ["Пени % в день, штрафы по договору."], MASK_CONTRACT, True)
Q(48, "4. УСЛОВИЯ КОНТРАКТА", "Возможность расторжения контракта",
  "Условия расторжения / одностороннего отказа по договору.",
  ["Основания расторжения договора."], MASK_CONTRACT, True)
Q(49, "4. УСЛОВИЯ КОНТРАКТА", "Форс-мажорные обстоятельства",
  "Что указано о форс-мажоре в договоре?",
  ["Непреодолимая сила: уведомление, сроки, расторжение."], MASK_CONTRACT, True)
Q(50, "4. УСЛОВИЯ КОНТРАКТА", "Порядок разрешения споров",
  "Порядок разрешения споров по договору (претензия, суд).",
  ["Претензионный порядок и суд."], MASK_CONTRACT, True)
Q(51, "4. УСЛОВИЯ КОНТРАКТА", "Арбитражная оговорка",
  "Какой арбитражный суд указан в договоре?",
  ["Наименование арбитражного суда."], MASK_CONTRACT, True)
Q(52, "4. УСЛОВИЯ КОНТРАКТА", "Состав ЗИП (запасных частей)",
  "Состав ЗИП и условия передачи заказчику (ТЗ/ЛСР/РД/договор).",
  ["Перечень запасных частей и ЗИП."], MASK_TECH + MASK_ESTIMATE + MASK_CONTRACT, False)
Q(53, "4. УСЛОВИЯ КОНТРАКТА", "Запрет субподряда",
  "Запрещён ли субподряд / нужно ли согласие заказчика?",
  ["Условия привлечения субподрядчиков."], MASK_CONTRACT + MASK_TECH, False)

# --- 5. Финансы ---
Q(54, "5. ФИНАНСОВЫЕ УСЛОВИЯ", "Начальная (максимальная) цена контракта",
  "Какая НМЦК в рублях? Если нет — скажи «Не указано», не подменяй сметой.",
  ["НМЦК из Извещения / закупочной документации."], MASK_NOTICE + MASK_CONTRACT, False)
Q(55, "5. ФИНАНСОВЫЕ УСЛОВИЯ", "Сметная стоимость (ориентир)",
  "Итоговые суммы по ЛСР/ВОР (САУГПТ и ЕСУМИС). Пометь как ориентир, НЕ НМЦК.",
  ["Итого/Всего по локальным сметам ЛСР."], MASK_ESTIMATE, True)
Q(56, "5. ФИНАНСОВЫЕ УСЛОВИЯ", "Сводка из ЛСР (топ позиций)",
  "Топ-10 позиций из ЛСР/ВОР с наименованием, кол-вом и ценой (если есть) + строка Итого.",
  ["Крупнейшие позиции спецификации ЛСР САУГПТ и ЕСУМИС."], MASK_ESTIMATE, True)
Q(57, "5. ФИНАНСОВЫЕ УСЛОВИЯ", "Порядок оплаты",
  "Порядок оплаты, удержания, этапы платежей.",
  ["Условия оплаты и гарантийные удержания."], MASK_NOTICE + MASK_CONTRACT, False)
Q(58, "5. ФИНАНСОВЫЕ УСЛОВИЯ", "Аванс",
  "Аванс: %, условия, банковская гарантия.",
  ["Максимальный размер аванса."], MASK_NOTICE + MASK_CONTRACT, False)
Q(59, "5. ФИНАНСОВЫЕ УСЛОВИЯ", "Обеспечение заявки",
  "Обеспечение заявки: размер и способ.",
  ["Обеспечение предложения."], MASK_NOTICE, False)
Q(60, "5. ФИНАНСОВЫЕ УСЛОВИЯ", "Обеспечение контракта",
  "Обеспечение исполнения договора (не путать с удержанием 5%).",
  ["Банковская гарантия / депозит обеспечения договора."], MASK_CONTRACT + MASK_NOTICE, False)
Q(61, "5. ФИНАНСОВЫЕ УСЛОВИЯ", "Валюта контракта",
  "В какой валюте контракт / расчёты?",
  ["Валюта договора."], MASK_NOTICE + MASK_CONTRACT, False)
Q(62, "5. ФИНАНСОВЫЕ УСЛОВИЯ", "Порядок индексации цены",
  "Предусмотрена ли индексация цены? Как?",
  ["Индексация / изменение цены договора."], MASK_CONTRACT + MASK_NOTICE, False)
Q(63, "5. ФИНАНСОВЫЕ УСЛОВИЯ", "График финансирования (детально)",
  "Детали графика освоения и финансирования денежных средств (этапы, суммы, сроки).",
  ["График финансирования по приложению к договору."], MASK_FINANCE_SCHED + MASK_CONTRACT + MASK_SCHEDULE, False)
Q(64, "5. ФИНАНСОВЫЕ УСЛОВИЯ", "Банковское сопровождение контракта",
  "Требуется ли банковское сопровождение контракта?",
  ["Банковское сопровождение / отдельный счёт."], MASK_CONTRACT + MASK_NOTICE, False)
Q(65, "5. ФИНАНСОВЫЕ УСЛОВИЯ", "Возможность снижения цены",
  "Есть ли возможность / порядок снижения цены?",
  ["Уменьшение цены договора."], MASK_CONTRACT + MASK_NOTICE, False)
Q(66, "5. ФИНАНСОВЫЕ УСЛОВИЯ", "Критерии оценки заявок",
  "Критерии оценки заявок / выбора победителя.",
  ["Оценка по стоимости лота / иные критерии."], MASK_NOTICE, False)

# --- 6. Риски (synthesize) ---
for num, title, prompt in [
    (67, "Основные риски для исполнителя", "Основные риски для исполнителя."),
    (68, "Риски по срокам", "Риски по срокам подачи и выполнения."),
    (69, "Риски по цене", "Финансовые риски (НМЦК vs смета, оплата, пени, обеспечения)."),
    (70, "Риски по качеству", "Риски по качеству, ТЗ, РД, оборудованию, гарантии."),
    (71, "Что не указано в документах", "Что важное отсутствует или неполно."),
    (72, "Что проверить перед подачей заявки", "Чек-лист перед подачей заявки."),
    (73, "Что можно улучшить в документации", "Недостатки документации для улучшения."),
    (74, "Рекомендация по участию", "Рекомендация: участвовать / осторожно / нет — с обоснованием."),
    (75, "Ключевые выводы", "Ключевые выводы (5–8 пунктов)."),
    (76, "Анализ конкурентной среды", "Какие выводы о конкурентной среде можно сделать по документации (критерий цены, барьеры входа, лицензии)?"),
    (77, "Оценка экономической целесообразности", "Оцени экономическую целесообразность участия при имеющихся данных."),
    (78, "Рекомендация по цене предложения", "Какую стратегию цены предложения рекомендовать (если НМЦК нет — опиши подход от сметы)?"),
]:
    Q(num, "6. РИСКИ И РЕКОМЕНДАЦИИ", title, prompt, kind="synthesize")

# --- 7. Мета / статус файлов ---
Q(79, "7. СТАТИСТИКА И ПОЛНОТА АНАЛИЗА", "Всего загружено файлов", "", kind="meta")
Q(80, "7. СТАТИСТИКА И ПОЛНОТА АНАЛИЗА", "Всего уникальных файлов (после дедупликации)", "", kind="meta")
Q(81, "7. СТАТИСТИКА И ПОЛНОТА АНАЛИЗА", "Статус обработки каждого файла", "", kind="meta")
Q(82, "7. СТАТИСТИКА И ПОЛНОТА АНАЛИЗА", "Какие файлы использованы в ответах", "", kind="meta")
Q(83, "7. СТАТИСТИКА И ПОЛНОТА АНАЛИЗА", "Какие файлы НЕ использованы", "", kind="meta")
Q(84, "7. СТАТИСТИКА И ПОЛНОТА АНАЛИЗА", "Причины неиспользования файлов", "", kind="meta")
Q(85, "7. СТАТИСТИКА И ПОЛНОТА АНАЛИЗА", "Рекомендации по дообработке файлов",
  "На основе статуса файлов и пробелов в ответах дай рекомендации по дообработке (какие файлы перечитать, что улучшить в OCR/парсинге).",
  kind="synthesize")
Q(86, "7. СТАТИСТИКА И ПОЛНОТА АНАЛИЗА", "Общий вывод по полноте анализа",
  "Общий вывод: насколько полон анализ (в % условно), какие зоны закрыты хорошо, какие слабые.",
  kind="synthesize")

assert len(QUESTIONS) == 86, f"Ожидалось 86 вопросов, получено {len(QUESTIONS)}"


# =============================================================================
# Номер тендера
# =============================================================================

TENDER_RE = re.compile(r"(B\d{10,})", re.IGNORECASE)


def find_tender_no() -> str:
    names = []
    if "uploaded_files" in globals() and uploaded_files:
        names.extend(uploaded_files.keys())
    names.extend(D_SOURCES)
    for n in names:
        m = TENDER_RE.search(str(n))
        if m:
            return m.group(1).upper()
    return ""


# =============================================================================
# Запуск
# =============================================================================

print("🚀 Модуль 4: 86 вопросов по всем файлам тендера")
print(f"   Уник. файлов: {len(FILE_GROUPS)}")
print(f"   Чанков после дедупа: {len(D_CHUNKS)}")
print(f"   Вопросов: {len(QUESTIONS)}")
print()

tender_no = find_tender_no()
if tender_no:
    print(f"🔖 Номер из имён файлов: {tender_no}")

answers: Dict[int, str] = {}
sources_used: Dict[int, List[str]] = {}
files_touched: Set[str] = set()
per_file_hits: Dict[str, int] = defaultdict(int)

TOTAL = 86

for q in QUESTIONS:
    n = q["num"]
    print(f"[{n}/{TOTAL}] {q['title']}...")

    if q["kind"] == "rag":
        # спец: номер тендера из имени файла
        if n == 1 and tender_no:
            answers[n] = tender_no
            sources_used[n] = ["(имя загруженного файла)"]
            print(f"   ↳ из имени файла: {tender_no}")
            continue

        ans, hits = ask_rag(
            q["question"],
            variants=q["variants"],
            masks=q["masks"],
            mask_only=q["mask_only"],
            top_k=TOP_K,
        )
        answers[n] = ans
        bases = []
        for h in hits:
            b = h.get("base") or _basename(h["source"])
            bases.append(b)
            files_touched.add(_norm(b))
            per_file_hits[_norm(b)] += 1
        sources_used[n] = list(dict.fromkeys(bases))[:12]

        if n == 1 and not tender_no:
            m = TENDER_RE.search(ans)
            if m:
                tender_no = m.group(1).upper()
                answers[n] = tender_no

    elif q["kind"] == "synthesize":
        # контекст = собранные ответы 1-66 (и уже готовые synth)
        facts = []
        for qq in QUESTIONS:
            if qq["num"] >= n:
                break
            if qq["kind"] == "meta":
                continue
            if qq["num"] in answers:
                facts.append(f"{qq['num']}. {qq['title']}: {answers[qq['num']]}")
        ctx = "\n\n".join(facts)
        if len(ctx) > 30000:
            ctx = ctx[:30000] + "\n\n[...усечено...]"
        prompt = (
            "Ты эксперт по тендерам. Опирайся ТОЛЬКО на собранные факты ниже. "
            "Не противоречь им. Различай НМЦК и сметный ориентир.\n\n"
            f"Задание: {q['question']}"
        )
        answers[n] = clean(ask_deepseek(prompt, context=ctx or "Факты отсутствуют."))
        sources_used[n] = ["(синтез ответов 1–N)"]

    elif q["kind"] == "meta":
        # заполним после цикла статусом — placeholder
        answers[n] = "<<META>>"
        sources_used[n] = ["(статистика модуля 4)"]


# --- Статус файлов ---
file_status_rows = []
for exp, key in FILE_PLAN:
    if key is None:
        file_status_rows.append({
            "file": exp,
            "in_index": "Нет",
            "used": "Нет",
            "hits": 0,
            "chunks": 0,
            "chars": 0,
            "reason": "Не найден в индексе после дедупликации",
        })
        continue
    g = FILE_GROUPS[key]
    hits = per_file_hits.get(key, 0)
    used = "Да" if hits > 0 else "Нет"
    reason = "" if used == "Да" else "Ни один ответ не сослался на чанки этого файла (низкая релевантность / маска поиска)"
    file_status_rows.append({
        "file": exp if exp in EXPECTED_FILES else g["display"],
        "in_index": "Да",
        "used": used,
        "hits": hits,
        "chunks": len(g["idxs"]),
        "chars": g["chars"],
        "reason": reason,
    })

uploaded_count = len(uploaded_files) if "uploaded_files" in globals() and uploaded_files else len(FILE_GROUPS)
unique_count = len(FILE_GROUPS)
used_files = sorted({r["file"] for r in file_status_rows if r["used"] == "Да"})
unused_files = [r for r in file_status_rows if r["used"] == "Нет"]

# meta answers
answers[79] = str(uploaded_count)
answers[80] = str(unique_count)
answers[81] = "См. раздел «Статус обработки файлов» ниже."
answers[82] = "; ".join(used_files) if used_files else "Не указано"
answers[83] = "; ".join(r["file"] for r in unused_files) if unused_files else "Нет — все учтённые файлы использованы"
answers[84] = "\n".join(
    f"- {r['file']}: {r['reason']}" for r in unused_files
) if unused_files else "Все файлы использованы в ответах."

# synthesize 85-86 now that meta ready
for q in QUESTIONS:
    if q["num"] in (85, 86):
        print(f"[{q['num']}/{TOTAL}] {q['title']} (финальный синтез)...")
        facts = []
        for qq in QUESTIONS:
            if qq["num"] == q["num"]:
                break
            if qq["num"] in answers:
                facts.append(f"{qq['num']}. {qq['title']}: {answers[qq['num']]}")
        facts.append("Статус файлов:\n" + "\n".join(
            f"- {r['file']}: index={r['in_index']}, used={r['used']}, hits={r['hits']}, reason={r['reason']}"
            for r in file_status_rows
        ))
        ctx = "\n\n".join(facts)
        if len(ctx) > 30000:
            ctx = ctx[:30000]
        answers[q["num"]] = clean(ask_deepseek(
            f"{PRIORITY_PROMPT}\n\nЗадание: {q['question']}",
            context=ctx,
        ))
        sources_used[q["num"]] = ["(синтез + статус файлов)"]


# =============================================================================
# Отчёт
# =============================================================================

def build_name(num: str) -> str:
    d = ANALYSIS_DATE.strftime("%Y%m%d")
    t = ANALYSIS_DATE.strftime("%H%M%S")
    if num:
        safe = re.sub(r"[^\w\-]+", "_", num)
        return f"Анализ_тендера_{safe}_{d}.txt"
    return f"Анализ_тендера_{d}_{t}.txt"


def format_report() -> str:
    lines = []
    lines.append("=" * 70)
    lines.append("📊 ПОЛНЫЙ АНАЛИЗ ТЕНДЕРНОЙ ДОКУМЕНТАЦИИ (86 вопросов)")
    lines.append("=" * 70)
    lines.append(f"Номер тендера / ИКЗ: {tender_no or answers.get(1, 'Не указано')}")
    lines.append(f"Дата анализа: {ANALYSIS_DATE.strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"Загружено файлов (upload): {uploaded_count}")
    lines.append(f"Уникальных файлов (дедуп): {unique_count}")
    lines.append(f"Чанков исходных / после дедупа: {len(rag_index.chunks)} / {len(D_CHUNKS)}")
    lines.append("")

    cur_sec = None
    for q in QUESTIONS:
        if q["section"] != cur_sec:
            if cur_sec is not None:
                lines.append("")
            lines.append("=" * 70)
            lines.append(q["section"])
            lines.append("=" * 70)
            cur_sec = q["section"]
        lines.append("")
        lines.append(f"{q['num']}. {q['title']}")
        lines.append("-" * 70)
        lines.append(answers.get(q["num"], "Не указано"))
        srcs = sources_used.get(q["num"] ) or []
        if srcs:
            lines.append("Источники: " + "; ".join(srcs[:10]))

    lines.append("")
    lines.append("=" * 70)
    lines.append("СТАТУС ОБРАБОТКИ ФАЙЛОВ")
    lines.append("=" * 70)
    lines.append(f"{'Файл':<70} {'В индексе':<10} {'Использован':<12} {'Hits':<6} {'Чанков'}")
    lines.append("-" * 110)
    for r in file_status_rows:
        name = r["file"] if len(r["file"]) <= 68 else r["file"][:65] + "..."
        lines.append(f"{name:<70} {r['in_index']:<10} {r['used']:<12} {r['hits']:<6} {r['chunks']}")
        if r["used"] == "Нет":
            lines.append(f"   причина: {r['reason']}")
        lines.append(f"   символов текста: {r['chars']:,}")

    lines.append("=" * 70)
    return "\n".join(lines) + "\n"


report_text = format_report()
report_filename = build_name(tender_no)
with open(report_filename, "w", encoding="utf-8") as f:
    f.write(report_text)

tender_report = report_text
tender_report_filename = report_filename
tender_number = tender_no
tender_answers = answers
tender_file_status = file_status_rows

if colab_files is not None:
    colab_files.download(report_filename)

print()
print("✅ Отчёт сформирован!")
print(f"📁 Файл: {report_filename}")
print(f"📊 Вопросов: {TOTAL}")
print(f"📂 Файлов использовано: {sum(1 for r in file_status_rows if r['used']=='Да')} / {len(file_status_rows)}")
if tender_no:
    print(f"🔖 Номер тендера: {tender_no}")
if colab_files is not None:
    print("📥 Файл скачан.")
else:
    print(f"💾 {os.path.abspath(report_filename)}")
