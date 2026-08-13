# =============================================================================
# МОДУЛЬ 4 — Анализ и отчёт (Google Colab)
# Выполняйте ПОСЛЕ модуля 3.
# Использует функции модуля 1: classify_document, ask_rag, clean, build_dedup…
# =============================================================================

import time
import re
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

# --- Проверки сессии ---
if "rag_index" not in globals() or not getattr(rag_index, "chunks", None):
    raise RuntimeError(
        "❌ Индекс пуст или не найден. Сначала выполните модули 1–3."
    )
for _fn in ("ask_rag", "classify_document", "extract_letter_requisites",
            "clean", "build_dedup", "map_expected", "search"):
    if _fn not in globals():
        raise RuntimeError(
            f"❌ Функция {_fn} не найдена. Перезапустите модуль 1."
        )

_t0_mod = time.time()
if "_mark" in globals():
    _mark("analysis_start")
print("=" * 70)
print("🚀 МОДУЛЬ 4 — Анализ тендера (86 вопросов) и отчёт")
print("=" * 70)

try:
    from google.colab import files as colab_files
except ImportError:
    colab_files = None

ANALYSIS_DATE = datetime.now()
TOP_K = 5
TOP_K_FORCED = 8
MAX_CTX = 12000
DEEPSEEK_TIMEOUT = int(globals().get("DEEPSEEK_TIMEOUT", 60) or 60)

# --- Дедуп индекса и план файлов (функции — из модуля 1) ---
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



# --- Классификация документов (функции — из модуля 1) ---
print("🏷️ Классификация документов по содержимому...")
_status("классификация документов и извлечение реквизитов писем")
FILE_CLASSIFICATION: Dict[str, Dict[str, Any]] = {}  # group_key -> meta
APPROVALS: List[Dict[str, str]] = []

for exp, key in FILE_PLAN:
    if key is None:
        FILE_CLASSIFICATION[exp] = {
            "key": None,
            "display": exp,
            "doc_type": "Не найден в индексе",
            "signals": [],
            "sample_chars": 0,
        }
        continue
    display = FILE_GROUPS[key]["display"]
    sample = _file_sample_text(key)
    doc_type, signals = classify_document(display, sample)
    meta = {
        "key": key,
        "display": display,
        "label": exp,
        "doc_type": doc_type,
        "signals": signals,
        "sample_chars": len(sample),
        "sample": sample,
    }
    FILE_CLASSIFICATION[key] = meta
    print(f"   • {display}: {doc_type} [{', '.join(signals[:4]) or '—'}]")

    # письма / согласования — извлекаем реквизиты
    if doc_type == DOC_TYPE_LETTER or (
        letter_score := _score_keywords(sample.lower(), LETTER_KW)
    ) >= 2:
        # даже если имя «РД …», при сильных признаках письма — извлекаем
        if doc_type != DOC_TYPE_LETTER and letter_score >= 2:
            doc_type = DOC_TYPE_LETTER
            meta["doc_type"] = DOC_TYPE_LETTER
            print(f"     ↳ переклассифицирован в «{DOC_TYPE_LETTER}» по содержимому")
        print(f"     ✉️ Извлечение реквизитов письма...")
        req = extract_letter_requisites(display, sample)
        meta["letter"] = req
        if req.get("Тип") != "не письмо":
            APPROVALS.append(req)

# дедуп согласований по номер+дата+файл
_seen_appr = set()
_uniq_appr: List[Dict[str, str]] = []
for a in APPROVALS:
    key = (a.get("Номер письма"), a.get("Дата письма"), a.get("Файл"))
    if key in _seen_appr:
        continue
    _seen_appr.add(key)
    _uniq_appr.append(a)
APPROVALS = _uniq_appr
print(f"   Найдено писем/согласований: {len(APPROVALS)}")



# --- 86 вопросов ---
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
# Запуск анализа (find_tender_no / _progress_bar — из модуля 1)
# =============================================================================

print("🚀 Анализ: 86 вопросов по всем файлам тендера (ускоренный режим)")
print(f"   Уник. файлов: {len(FILE_GROUPS)}")
print(f"   Чанков после дедупа: {len(D_CHUNKS)}")
print(f"   Вопросов: {len(QUESTIONS)}")
print(f"   TOP_K={TOP_K}, DeepSeek timeout={DEEPSEEK_TIMEOUT}с")
print()
if "_status" in globals():
    _status("модуль 4 — ответы на 86 вопросов")

_t0_analysis = time.time()

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
    print(_progress_bar(n - 1, TOTAL, _t0_analysis, q["title"]), flush=True)

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
        answers[n] = clean(ask_deepseek(prompt, context=ctx or "Факты отсутствуют.", timeout=DEEPSEEK_TIMEOUT))
        sources_used[n] = ["(синтез ответов 1–N)"]

    elif q["kind"] == "meta":
        # заполним после цикла статусом — placeholder
        answers[n] = "<<META>>"
        sources_used[n] = ["(статистика модуля 4)"]

print(_progress_bar(TOTAL, TOTAL, _t0_analysis, "основной проход"), flush=True)
print(f"⏱️ Основной проход: {_fmt_eta(time.time() - _t0_analysis)}")

# --- Статус файлов ---
file_status_rows = []
for exp, key in FILE_PLAN:
    if key is None:
        cls = FILE_CLASSIFICATION.get(exp, {})
        file_status_rows.append({
            "file": exp,
            "in_index": "Нет",
            "used": "Нет",
            "hits": 0,
            "chunks": 0,
            "chars": 0,
            "doc_type": cls.get("doc_type", "Не найден в индексе"),
            "signals": cls.get("signals", []),
            "reason": "Не найден в индексе после дедупликации",
        })
        continue
    g = FILE_GROUPS[key]
    hits = per_file_hits.get(key, 0)
    used = "Да" if hits > 0 else "Нет"
    reason = "" if used == "Да" else "Ни один ответ не сослался на чанки этого файла (низкая релевантность / маска поиска)"
    cls = FILE_CLASSIFICATION.get(key, {})
    file_status_rows.append({
        "file": exp if exp in EXPECTED_FILES else g["display"],
        "in_index": "Да",
        "used": used,
        "hits": hits,
        "chunks": len(g["idxs"]),
        "chars": g["chars"],
        "doc_type": cls.get("doc_type", DOC_TYPE_OTHER),
        "signals": cls.get("signals", []),
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
        print(_progress_bar(q["num"] - 1, TOTAL, _t0_analysis, q["title"] + " (финальный синтез)"), flush=True)
        facts = []
        for qq in QUESTIONS:
            if qq["num"] == q["num"]:
                break
            if qq["num"] in answers:
                facts.append(f"{qq['num']}. {qq['title']}: {answers[qq['num']]}")
        facts.append("Статус файлов:\n" + "\n".join(
            f"- {r['file']}: type={r.get('doc_type')}, index={r['in_index']}, "
            f"used={r['used']}, hits={r['hits']}, reason={r['reason']}"
            for r in file_status_rows
        ))
        if APPROVALS:
            facts.append("Согласования:\n" + "\n".join(
                f"- {a.get('Файл')}: №{a.get('Номер письма')}, {a.get('Дата письма')}, "
                f"{a.get('Отправитель')} → {a.get('Суть согласования')}"
                for a in APPROVALS
            ))
        ctx = "\n\n".join(facts)
        if len(ctx) > 30000:
            ctx = ctx[:30000]
        answers[q["num"]] = clean(ask_deepseek(
            f"{PRIORITY_PROMPT}\n\nЗадание: {q['question']}",
            context=ctx,
            timeout=DEEPSEEK_TIMEOUT,
        ))
        sources_used[q["num"]] = ["(синтез + статус файлов)"]

print(_progress_bar(TOTAL, TOTAL, _t0_analysis, "анализ завершён"), flush=True)
print(f"⏱️ Полный анализ: {_fmt_eta(time.time() - _t0_analysis)}")



# --- Отчёт ---
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
    lines.append(f"Писем/согласований найдено: {len(APPROVALS)}")
    lines.append("")

    # --- Согласования и разрешения (сразу после шапки) ---
    lines.append("=" * 70)
    lines.append("СОГЛАСОВАНИЯ И РАЗРЕШЕНИЯ")
    lines.append("=" * 70)
    if not APPROVALS:
        lines.append("Письма-согласования / разрешения не обнаружены "
                     "(или OCR не дал достаточно текста для классификации).")
    else:
        for i, a in enumerate(APPROVALS, start=1):
            lines.append("")
            lines.append(f"--- Согласование {i} ---")
            lines.append(f"Файл: {a.get('Файл', 'Не указано')}")
            lines.append(f"Тип: {a.get('Тип', DOC_TYPE_LETTER)}")
            lines.append(f"Дата письма: {a.get('Дата письма', 'Не указано')}")
            lines.append(f"Номер письма: {a.get('Номер письма', 'Не указано')}")
            lines.append(f"Отправитель: {a.get('Отправитель', 'Не указано')}")
            lines.append(f"Получатель: {a.get('Получатель', 'Не указано')}")
            lines.append(f"Суть согласования: {a.get('Суть согласования', 'Не указано')}")
            lines.append(f"Результат: {a.get('Результат', 'Не указано')}")
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
        srcs = sources_used.get(q["num"]) or []
        if srcs:
            lines.append("Источники: " + "; ".join(srcs[:10]))

    lines.append("")
    lines.append("=" * 70)
    lines.append("СТАТУС ОБРАБОТКИ ФАЙЛОВ")
    lines.append("=" * 70)
    lines.append(
        f"{'Файл':<48} {'Тип документа':<28} {'Индекс':<8} {'Исп.':<6} {'Hits':<5} {'Чанков'}"
    )
    lines.append("-" * 120)
    for r in file_status_rows:
        name = r["file"] if len(r["file"]) <= 46 else r["file"][:43] + "..."
        dtype = str(r.get("doc_type", ""))
        if len(dtype) > 26:
            dtype = dtype[:23] + "..."
        lines.append(
            f"{name:<48} {dtype:<28} {r['in_index']:<8} {r['used']:<6} {r['hits']:<5} {r['chunks']}"
        )
        if r.get("signals"):
            lines.append(f"   признаки: {', '.join(r['signals'][:6])}")
        if r["used"] == "Нет":
            lines.append(f"   причина: {r['reason']}")
        lines.append(f"   символов текста: {r['chars']:,}")

    lines.append("=" * 70)
    return "\n".join(lines) + "\n"


_status("формирование TXT-отчёта")

_mark("analysis_end")
# end отметим после скачивания — чтобы итог включал запись/download
_t_deps = _elapsed("deps_start", "deps_end")
_t_up = _elapsed("upload_start", "upload_end")
_t_idx = _elapsed("index_start", "index_end")
_t_an = _elapsed("analysis_start", "analysis_end")

report_text = format_report()
report_text = report_text.rstrip() + (
    f"\n\n{'=' * 70}\n"
    f"СВОДКА ВРЕМЕНИ ВЫПОЛНЕНИЯ\n"
    f"{'=' * 70}\n"
    f"1. Зависимости / init:   {_fmt_dur(_t_deps)}\n"
    f"2. Загрузка файлов:      {_fmt_dur(_t_up)}\n"
    f"3. OCR / индексация:     {_fmt_dur(_t_idx)}\n"
    f"4. Анализ 86 вопросов:   {_fmt_dur(_t_an)}\n"
)

report_filename = build_name(tender_no)
with open(report_filename, "w", encoding="utf-8") as f:
    f.write(report_text)

tender_report = report_text
tender_report_filename = report_filename
tender_number = tender_no
tender_answers = answers
tender_file_status = file_status_rows

_manual_link = _manual_download_link(report_filename)

_status("скачивание отчёта")
if colab_files is not None:
    colab_files.download(report_filename)

_mark("end")
_t_all = _elapsed("start", "end")
tender_timings = {
    "deps": _t_deps,
    "upload": _t_up,
    "index": _t_idx,
    "analysis": _t_an,
    "total": _t_all,
}

# допишем ИТОГО в файл
try:
    with open(report_filename, "a", encoding="utf-8") as f:
        f.write(f"ИТОГО:                   {_fmt_dur(_t_all)}\n")
except Exception:
    pass

print()
print("✅ Отчёт сформирован!")
print(f"📁 Файл: {report_filename}")
print(f"📊 Вопросов: {TOTAL}")
print(f"⚡ Режим: TOP_K={TOP_K}, timeout={DEEPSEEK_TIMEOUT}с, ETA-бар включён")
print(f"📂 Файлов использовано: {sum(1 for r in file_status_rows if r['used']=='Да')} / {len(file_status_rows)}")
if tender_no:
    print(f"🔖 Номер тендера: {tender_no}")

print(f"\n📥 Файл скачан автоматически." if colab_files is not None else "\n💾 Автоскачивание недоступно (не Colab).")
print(f"📁 Если скачивание не началось, скачайте вручную:")
print(f"   🔗 {_manual_link}")



# --- Сводка по времени модуля 4 ---
_elapsed = time.time() - _t0_mod
print()
print("=" * 70)
print("Сводка модуля 4")
print("=" * 70)
if "_fmt_dur" in globals():
    print(f"   Анализ + отчёт: {_fmt_dur(_elapsed)}")
else:
    print(f"   Анализ + отчёт: {int(_elapsed)} сек.")
print()
print("✅ Отчёт сформирован!")
