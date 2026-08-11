# =============================================================================
# МОДУЛЬ 4 — Полный анализ ВСЕХ файлов тендера + структурированный отчёт
# Принцип: пройти по КАЖДОМУ уникальному файлу → извлечь ВСЕ данные →
#           собрать ответы на 24 ключевых вопроса → TXT + download.
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

# --- Проверки сессии ---
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
TOP_K_FILE = 20          # чанков на файл при полном разборе
TOP_K_Q = 12             # чанков на целевой вопрос
MAX_CHARS_PER_FILE_CTX = 14000  # лимит контекста на файл для LLM


# =============================================================================
# Ожидаемые файлы (для статуса) + дедупликация
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


def _norm_name(name: str) -> str:
    name = str(name).replace("\\", "/").split("/")[-1].lower().strip()
    name = re.sub(r"\s*\(\d+\)(?=\.\w+$)", "", name)
    name = re.sub(r"\s+", " ", name)
    return name


def _content_hash(text: str) -> str:
    norm = re.sub(r"\s+", " ", (text or "").strip().lower())
    return hashlib.md5(norm.encode("utf-8", errors="ignore")).hexdigest()


def _match_expected(basename: str, expected: str) -> bool:
    b = _norm_name(basename)
    e = _norm_name(expected)
    if b == e:
        return True
    # мягкое сравнение: ключевые токены
    if e in b or b in e:
        return True
    # РД по номеру
    m = re.search(r"14-27-\d+", e)
    if m and m.group(0) in b:
        return True
    # ЛСР/ВОР САУГПТ / ЕСУМИС
    if "лср" in e and "лср" in b:
        if "саугпт" in e and "саугпт" in b:
            return True
        if "есумис" in e and "есумис" in b:
            return True
    if "вор" in e and "вор" in b:
        if "саугпт" in e and "саугпт" in b:
            return True
        if "есумис" in e and "есумис" in b:
            return True
    if "график производства" in e and "график производства" in b:
        return True
    if "график освоения" in e and "график освоения" in b:
        return True
    if "акт окончания" in e and "акт окончания" in b:
        return True
    if "проект договора" in e and "проект договора" in b and "смр" in b:
        return True
    if e.startswith("техническое задание") and b.startswith("техническое задание"):
        return True
    if e.startswith("извещение") and b.startswith("извещение"):
        return True
    if "закупочная документация" in e and "закупочная документация" in b:
        return True
    if "приложение № 1" in e and "техническое задание" in e and "техническое задание" in b and b.endswith(".doc"):
        return True
    return False


def build_file_groups() -> Dict[str, Dict[str, Any]]:
    """
    Группирует чанки по каноническому basename (дедуп копий).
    Возвращает: canon_name -> {paths, chunks:[(text, source, orig_i)], chars}
    """
    # сначала собираем все чанки с предпочтением коротких путей
    items = list(enumerate(zip(rag_index.chunks, rag_index.sources)))

    def path_rank(src: str) -> Tuple[int, int]:
        s = src.lower()
        penalty = 0
        if "процедуре" in s:
            penalty += 2
        if s.count(".zip/") + s.count(".rar/") > 1:
            penalty += 1
        return (penalty, len(s))

    items.sort(key=lambda it: path_rank(it[1][1]))

    # content-dedup глобально, но группируем по basename
    seen_hash: Set[str] = set()
    groups: Dict[str, Dict[str, Any]] = {}

    for orig_i, (text, source) in items:
        h = _content_hash(text)
        if h in seen_hash:
            continue
        seen_hash.add(h)
        base = _norm_name(source)
        # каноническое отображаемое имя — basename исходного пути
        display = str(source).replace("\\", "/").split("/")[-1]
        display = re.sub(r"\s*\(\d+\)(?=\.\w+$)", "", display)

        if base not in groups:
            groups[base] = {
                "display": display,
                "paths": set(),
                "chunks": [],
                "chars": 0,
            }
        groups[base]["paths"].add(source)
        groups[base]["chunks"].append({"text": text, "source": source, "chunk": orig_i + 1})
        groups[base]["chars"] += len(text)

    return groups


print("🔧 Группировка и дедупликация файлов...")
FILE_GROUPS = build_file_groups()
print(f"   Уникальных файлов (basename): {len(FILE_GROUPS)}")
print(f"   Чанков после дедупа: {sum(len(g['chunks']) for g in FILE_GROUPS.values())}")


def resolve_expected_to_groups() -> List[Tuple[str, Optional[str]]]:
    """Сопоставляет EXPECTED_FILES с ключами FILE_GROUPS. [(expected, group_key|None)]"""
    used_keys: Set[str] = set()
    mapping: List[Tuple[str, Optional[str]]] = []
    for exp in EXPECTED_FILES:
        found = None
        for key in FILE_GROUPS:
            if key in used_keys:
                continue
            if _match_expected(key, exp) or _match_expected(FILE_GROUPS[key]["display"], exp):
                found = key
                break
        if found:
            used_keys.add(found)
        mapping.append((exp, found))
    # добавить файлы индекса, не попавшие в expected
    extras = [k for k in FILE_GROUPS if k not in used_keys]
    for k in extras:
        mapping.append((FILE_GROUPS[k]["display"], k))
    return mapping


FILE_PLAN = resolve_expected_to_groups()


# =============================================================================
# Поиск внутри файла / глобально
# =============================================================================

def chunks_for_file(group_key: str) -> List[Dict[str, Any]]:
    return FILE_GROUPS[group_key]["chunks"]


def select_chunks_from_file(group_key: str, query: str, top_k: int = TOP_K_FILE) -> List[Dict[str, Any]]:
    """TF-IDF ранжирование чанков одного файла относительно query; если мало — взять все (усечённо)."""
    chs = chunks_for_file(group_key)
    if not chs:
        return []
    texts = [c["text"] for c in chs]
    if len(texts) <= top_k:
        # всё равно отсортируем по релевантности если возможно
        try:
            vect = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), sublinear_tf=True)
            mat = vect.fit_transform(texts + [query])
            scores = cosine_similarity(mat[-1], mat[:-1]).ravel()
            order = np.argsort(scores)[::-1]
            return [
                {**chs[i], "score": float(scores[i])}
                for i in order
            ]
        except Exception:
            return [{**c, "score": 1.0} for c in chs]

    try:
        vect = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), sublinear_tf=True)
        mat = vect.fit_transform(texts)
        qv = vect.transform([query])
        scores = cosine_similarity(qv, mat).ravel()
        order = np.argsort(scores)[::-1][:top_k]
        return [{**chs[i], "score": float(scores[i])} for i in order]
    except Exception:
        return [{**c, "score": 1.0} for c in chs[:top_k]]


def build_context(hits: List[Dict[str, Any]], max_chars: int = MAX_CHARS_PER_FILE_CTX) -> str:
    parts = []
    total = 0
    for i, h in enumerate(hits, start=1):
        block = (
            f"[Фрагмент {i} | {h.get('source', '?')} | чанк #{h.get('chunk', '?')}]\n"
            f"{h.get('text', '')}"
        )
        if total + len(block) > max_chars and parts:
            break
        parts.append(block)
        total += len(block)
    return "\n\n---\n\n".join(parts)


def _clean(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return "Не указано"
    low = text.lower()
    empty = [
        "не указано", "нет информации", "не найден", "в контексте нет",
        "информация отсутствует", "не удалось извлечь", "пустой файл",
    ]
    if len(text) < 160 and any(m in low for m in empty):
        return "Не указано"
    return text


def ask(question: str, context: str) -> str:
    if not context.strip():
        return "Не указано"
    prompt = (
        "Ты аналитик тендерной документации. Отвечай ТОЛЬКО по контексту. "
        "Выпиши все конкретные факты: числа, даты, суммы, перечни, нормы, ФИО, ИНН. "
        "Не пиши «не указано», если в контексте есть хотя бы частичный ответ. "
        "Отвечай на русском, структурировано.\n\n"
        f"Вопрос: {question}"
    )
    return _clean(ask_deepseek(prompt, context=context))


# =============================================================================
# Шаг 1: разбор КАЖДОГО файла
# =============================================================================

FILE_EXTRACT_PROMPT = """Проанализируй фрагменты ОДНОГО файла тендерной документации и извлеки ВСЁ полезное.

Выдай ответ строго в структуре (если пункта нет в файле — напиши «нет в этом файле»):

1) Тип документа и краткое назначение
2) Ключевые идентификаторы (номер тендера/лота/ИКЗ, даты, реквизиты)
3) Стороны / контакты / ИНН (если есть)
4) Предмет, объект, адрес / место работ
5) Техника: системы (САУГПТ/ЕСУМИС), ГОТВ, модули, оборудование, количества
6) Нормативы (ГОСТ/СП/СНиП/ПУЭ с номерами)
7) Требования: монтаж, ПНР, пожарная безопасность, лицензии, опыт, персонал
8) Сроки: подачи заявок (дата+время), выполнения работ, этапы, гарантия, оплата
9) Финансы: НМЦК, сметные итоги, аванс, обеспечения, удержания, пени/штрафы
10) Юридическое: расторжение, форс-мажор, споры/суд, субподряд
11) Документы для заявки / приложения / ЗИП
12) Важные цифры и перечни (списком)
13) Краткое резюме (1–3 предложения): что даёт этот файл для анализа тендера
"""


print("\n📂 Этап 1/3: анализ каждого файла...")
file_analyses: Dict[str, Dict[str, Any]] = {}  # expected_or_display -> result
file_status: List[Dict[str, Any]] = []

total_files = len(FILE_PLAN)
for idx, (label, key) in enumerate(FILE_PLAN, start=1):
    print(f"[{idx}/{total_files}] Файл: {label}")

    if key is None or key not in FILE_GROUPS:
        print("   ❌ Не найден в индексе")
        file_status.append({
            "file": label,
            "used": "Нет",
            "reason": "Файл не найден в индексе после дедупликации",
            "extracted": "—",
            "chunks": 0,
            "chars": 0,
        })
        file_analyses[label] = {
            "ok": False,
            "analysis": "Файл отсутствует в индексе.",
            "key": None,
        }
        continue

    group = FILE_GROUPS[key]
    n_ch = len(group["chunks"])
    print(f"   чанков: {n_ch}, символов: {group['chars']:,}")

    # Берём релевантные + «обзорные» чанки (начало документа тоже важно)
    overview_q = (
        "извлечение всех ключевых данных: заказчик ИНН контакты предмет объект адрес "
        "сроки цена смета оборудование модули ГОТВ гарантия оплата пени договор лицензия "
        "график ПНР монтаж ГОСТ ЗИП итог сумма"
    )
    hits = select_chunks_from_file(key, overview_q, top_k=TOP_K_FILE)
    # добавить первые чанки файла (шапка), если их нет
    head = group["chunks"][:3]
    existing = {(h["chunk"], h["text"][:80]) for h in hits}
    for h in head:
        k2 = (h["chunk"], h["text"][:80])
        if k2 not in existing:
            hits.insert(0, {**h, "score": 1.0})
            existing.add(k2)

    context = build_context(hits, max_chars=MAX_CHARS_PER_FILE_CTX)
    if not context.strip():
        print("   ⚠️ Пустой контекст")
        file_status.append({
            "file": label,
            "used": "Нет",
            "reason": "Нет текстовых чанков (возможно, пустой/бинарный разбор)",
            "extracted": "—",
            "chunks": n_ch,
            "chars": group["chars"],
        })
        file_analyses[label] = {"ok": False, "analysis": "Пустой текст.", "key": key}
        continue

    analysis = ask(FILE_EXTRACT_PROMPT, context)
    # краткое резюме для статуса
    summary = ask(
        "В 1–2 предложениях: какие ГЛАВНЫЕ данные удалось извлечь из этого файла? "
        "Перечисли конкретику (даты, суммы, перечни), без воды.",
        context=analysis if analysis != "Не указано" else context[:4000],
    )

    used = "Да" if analysis != "Не указано" else "Нет"
    reason = "" if used == "Да" else "Модель не извлекла фактов из доступных чанков"
    print(f"   ✅ извлечено" if used == "Да" else f"   ⚠️ {reason}")

    file_analyses[label] = {
        "ok": used == "Да",
        "analysis": analysis,
        "summary": summary,
        "key": key,
        "display": group["display"],
        "paths": sorted(group["paths"])[:3],
        "chunks": n_ch,
        "chars": group["chars"],
    }
    file_status.append({
        "file": label,
        "used": used,
        "reason": reason,
        "extracted": summary,
        "chunks": n_ch,
        "chars": group["chars"],
    })


# =============================================================================
# Шаг 2: ответы на 24 вопроса по объединённым выдержкам (+ точечный поиск)
# =============================================================================

QUESTIONS: List[Tuple[str, str, str, Optional[Tuple[str, ...]]]] = [
    # (id, title, question, preferred_file_keywords)
    ("q1", "Заказчик (название, ИНН, контакты)",
     "Кто заказчик? Укажи полное наименование, ИНН, телефоны, email, контактных лиц.",
     ("извещение", "закупочная")),
    ("q2", "Предмет закупки",
     "Что является предметом закупки? Опиши САУГПТ/ЕСУМИС и состав работ (СМР, ПНР).",
     ("извещение", "закупочная", "техническое задание", "договор")),
    ("q3", "Местоположение объекта",
     "Где находится объект? Адрес, район, площадка, здание/помещение.",
     ("извещение", "закупочная", "техническое задание")),
    ("q4", "Система пожаротушения (тип, ГОТВ, модули)",
     "Какая система пожаротушения требуется? Тип, ГОТВ/огнетушащее вещество, модули (марки, кол-во).",
     ("техническое задание", "лср", "вор", "рд", "саугпт")),
    ("q5", "Оборудование в системе (полный перечень)",
     "Какое оборудование входит в систему? Собери полный перечень из ТЗ/ЛСР/ВОР/РД с количествами.",
     ("лср", "вор", "техническое задание", "рд", "саугпт", "есумис")),
    ("q6", "Нормативные документы (ГОСТ, СП, СНиП)",
     "Какие нормативные документы указаны? Перечисли ГОСТ, СП, СНиП, ПУЭ, ФЗ с номерами.",
     ("техническое задание", "договор", "рд")),
    ("q7", "Требования к монтажу",
     "Какие требования к монтажу указаны?",
     ("техническое задание", "договор", "рд")),
    ("q8", "Требования к ПНР",
     "Какие требования к пусконаладочным работам (ПНР)?",
     ("техническое задание", "договор", "извещение")),
    ("q9", "Требования по пожарной безопасности",
     "Какие требования по пожарной безопасности указаны?",
     ("техническое задание", "договор", "извещение")),
    ("q10", "Сроки выполнения работ (по этапам)",
     "Каковы сроки выполнения работ по этапам? Даты начала/окончания, длительность. Смотри график производства работ.",
     ("график производства", "техническое задание", "договор", "извещение")),
    ("q11", "Срок гарантии",
     "Каков срок гарантии (месяцев) и с какой даты считается?",
     ("договор", "извещение")),
    ("q12", "Срок подачи заявок",
     "Каков срок подачи заявок? Дата и время окончания приёма предложений.",
     ("извещение", "закупочная")),
    ("q13", "Срок оплаты",
     "Каков срок оплаты? Если в документах разные значения — укажи оба и источники.",
     ("извещение", "закупочная", "договор")),
    ("q14", "Начальная цена (НМЦК)",
     "Какая начальная (максимальная) цена контракта (НМЦК)? Если не указана — явно скажи.",
     ("извещение", "закупочная", "договор")),
    ("q15", "Суммы по сметам (ЛСР и ВОР)",
     "Каковы итоговые суммы по сметам ЛСР и ВОР (САУГПТ и ЕСУМИС)? Укажи «Итого/Всего» и файл.",
     ("лср", "вор")),
    ("q16", "Аванс",
     "Какой аванс предусмотрен (%, условия, банковская гарантия)?",
     ("извещение", "закупочная", "договор")),
    ("q17", "Обеспечение заявки и контракта",
     "Какое обеспечение заявки и обеспечение исполнения контракта? Размеры и способы. Не путай с гарантийным удержанием.",
     ("закупочная", "извещение", "договор")),
    ("q18", "Штрафы и пени",
     "Какие штрафы и пени указаны в договоре (% в день, суммы)?",
     ("договор",)),
    ("q19", "Условия расторжения договора",
     "Какие условия расторжения договора / одностороннего отказа?",
     ("договор",)),
    ("q20", "Форс-мажор",
     "Какие форс-мажорные обстоятельства и последствия указаны?",
     ("договор",)),
    ("q21", "Порядок разрешения споров",
     "Какой порядок разрешения споров и какой суд указан в договоре?",
     ("договор", "закупочная")),
    ("q22", "Требования к участникам",
     "Какие требования к участникам: лицензии (МЧС, СРО), опыт, персонал?",
     ("извещение", "закупочная")),
    ("q23", "Документы для заявки",
     "Какие документы нужно подать в заявке (формы №…)?",
     ("закупочная",)),
    ("q24", "Состав ЗИП",
     "Что входит в состав ЗИП (запасных частей/инструментов) и условия передачи?",
     ("техническое задание", "лср", "рд", "договор")),
]


def gather_context_for_question(title: str, question: str, keywords: Optional[Tuple[str, ...]]) -> Tuple[str, List[str]]:
    """Собирает контекст: выдержки file_analyses + чанки из релевантных файлов."""
    used_files: List[str] = []
    blocks: List[str] = []

    # 1) готовые разборы файлов (приоритет по keywords)
    scored_analyses = []
    for label, data in file_analyses.items():
        if not data.get("ok"):
            continue
        score = 0
        low = label.lower()
        if keywords:
            for kw in keywords:
                if kw.lower() in low:
                    score += 3
        scored_analyses.append((score, label, data["analysis"]))
    scored_analyses.sort(key=lambda x: (-x[0], x[1]))

    for score, label, analysis in scored_analyses[:8]:
        if score > 0 or not keywords:
            blocks.append(f"=== Выдержка анализа файла: {label} ===\n{analysis}")
            used_files.append(label)

    # 2) сырые чанки из релевантных файлов
    for label, key in FILE_PLAN:
        if key is None:
            continue
        low = label.lower()
        if keywords and not any(kw.lower() in low for kw in keywords):
            continue
        hits = select_chunks_from_file(key, question + " " + title, top_k=8)
        if not hits:
            continue
        ctx = build_context(hits, max_chars=6000)
        blocks.append(f"=== Чанки файла: {label} ===\n{ctx}")
        if label not in used_files:
            used_files.append(label)

    # если keywords слишком узкие и пусто — взять топ разборов всех файлов
    if not blocks:
        for label, data in list(file_analyses.items())[:10]:
            if data.get("ok"):
                blocks.append(f"=== Выдержка анализа файла: {label} ===\n{data['analysis']}")
                used_files.append(label)

    combined = "\n\n".join(blocks)
    # усечение общего контекста
    if len(combined) > 28000:
        combined = combined[:28000] + "\n\n[... контекст усечён ...]"
    return combined, used_files


print("\n📋 Этап 2/3: ответы на 24 ключевых вопроса...")
answers: Dict[str, str] = {}
answer_sources: Dict[str, List[str]] = {}

for i, (qid, title, question, keywords) in enumerate(QUESTIONS, start=1):
    print(f"[{i}/24] {title}...")
    ctx, used = gather_context_for_question(title, question, keywords)
    ans = ask(question, ctx)
    # retry без маски keywords, если пусто
    if ans == "Не указано":
        ctx2, used2 = gather_context_for_question(title, question, None)
        ans2 = ask(
            question + " Если данные разрознены — собери из всех фрагментов.",
            ctx2,
        )
        if ans2 != "Не указано":
            ans, used = ans2, used2
    answers[qid] = ans
    answer_sources[qid] = used


# =============================================================================
# Номер тендера для имени файла
# =============================================================================

TENDER_RE = re.compile(r"(B\d{10,})", re.IGNORECASE)


def find_tender_number() -> str:
    names = []
    if "uploaded_files" in globals() and uploaded_files:
        names.extend(uploaded_files.keys())
    for g in FILE_GROUPS.values():
        names.append(g["display"])
        names.extend(g["paths"])
    for n in names:
        m = TENDER_RE.search(str(n))
        if m:
            return m.group(1).upper()
    # из ответов
    blob = " ".join(answers.values())
    m = TENDER_RE.search(blob)
    if m:
        return m.group(1).upper()
    return ""


tender_no = find_tender_number()
print(f"\n🔖 Номер тендера: {tender_no or '(не найден)'}")


# =============================================================================
# Шаг 3: раздел рисков на основе собранных ответов
# =============================================================================

print("\n📊 Этап 3/3: риски и рекомендации...")

facts_blob = "\n\n".join(
    f"{title}:\n{answers[qid]}" for qid, title, _q, _kw in QUESTIONS
)

RISK_PROMPTS = [
    ("Основные риски для исполнителя",
     "Перечисли основные риски для исполнителя на основе фактов ниже."),
    ("Риски по срокам",
     "Оцени риски по срокам (подача заявки и выполнение работ)."),
    ("Риски по цене",
     "Оцени финансовые риски (НМЦК/смета, оплата, аванс, обеспечения, пени). Различай НМЦК и сметный ориентир."),
    ("Риски по качеству / технике",
     "Оцени риски по составу оборудования, ТЗ, РД, гарантии, лицензиям."),
    ("Что проверить перед подачей заявки",
     "Составь чек-лист проверок перед подачей заявки."),
    ("Рекомендация по участию",
     "Дай рекомендацию: участвовать / осторожно / не участвовать — с обоснованием."),
    ("Ключевые выводы",
     "Ключевые выводы по тендеру (5–8 пунктов)."),
]

risk_answers: Dict[str, str] = {}
for title, prompt in RISK_PROMPTS:
    print(f"   • {title}")
    risk_answers[title] = ask(
        "Ты эксперт по тендерам. Опирайся ТОЛЬКО на собранные факты. Не противоречь им.\n" + prompt,
        facts_blob[:28000],
    )


# =============================================================================
# Сборка TXT
# =============================================================================

def build_report_filename(num: str) -> str:
    d = ANALYSIS_DATE.strftime("%Y%m%d")
    t = ANALYSIS_DATE.strftime("%H%M%S")
    if num:
        safe = re.sub(r"[^\w\-]+", "_", num)
        return f"Анализ_тендера_{safe}_{d}.txt"
    return f"Анализ_тендера_{d}_{t}.txt"


def format_report() -> str:
    lines: List[str] = []
    lines.append("=" * 70)
    lines.append("📊 ПОЛНЫЙ АНАЛИЗ ТЕНДЕРНОЙ ДОКУМЕНТАЦИИ (по всем файлам)")
    lines.append("=" * 70)
    lines.append(f"Номер тендера / ИКЗ: {tender_no or 'Не указано'}")
    lines.append(f"Дата анализа: {ANALYSIS_DATE.strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"Уникальных файлов в индексе: {len(FILE_GROUPS)}")
    lines.append(f"Файлов в плане анализа: {len(FILE_PLAN)}")
    lines.append(f"Чанков исходных: {len(rag_index.chunks)}")
    lines.append("")

    # --- A. Ответы на 24 вопроса ---
    lines.append("=" * 70)
    lines.append("A. КЛЮЧЕВЫЕ ВОПРОСЫ И ОТВЕТЫ (агрегация по всем файлам)")
    lines.append("=" * 70)
    for qid, title, _q, _kw in QUESTIONS:
        lines.append("")
        lines.append("-" * 70)
        lines.append(f"{title}")
        lines.append("-" * 70)
        lines.append(answers.get(qid, "Не указано"))
        srcs = answer_sources.get(qid) or []
        if srcs:
            lines.append("Источники: " + "; ".join(srcs[:10]))

    # --- B. Риски ---
    lines.append("")
    lines.append("=" * 70)
    lines.append("B. РИСКИ И РЕКОМЕНДАЦИИ")
    lines.append("=" * 70)
    for title, _p in RISK_PROMPTS:
        lines.append("")
        lines.append(f"{title}:")
        lines.append(risk_answers.get(title, "Не указано"))

    # --- C. Пофайловые выдержки ---
    lines.append("")
    lines.append("=" * 70)
    lines.append("C. РАЗБОР ПО КАЖДОМУ ФАЙЛУ")
    lines.append("=" * 70)
    for label, key in FILE_PLAN:
        data = file_analyses.get(label, {})
        lines.append("")
        lines.append("#" * 70)
        lines.append(f"Файл: {label}")
        if data.get("display") and data["display"] != label:
            lines.append(f"В индексе как: {data['display']}")
        lines.append(f"Чанков: {data.get('chunks', 0)}, символов: {data.get('chars', 0):,}")
        lines.append("#" * 70)
        lines.append(data.get("analysis", "Нет данных."))

    # --- D. Статус обработки ---
    lines.append("")
    lines.append("=" * 70)
    lines.append("D. СТАТУС ОБРАБОТКИ ФАЙЛОВ")
    lines.append("=" * 70)
    used_yes = sum(1 for s in file_status if s["used"] == "Да")
    lines.append(f"Использовано файлов: {used_yes} / {len(file_status)}")
    lines.append("")
    for s in file_status:
        lines.append("-" * 70)
        lines.append(f"Имя файла: {s['file']}")
        lines.append(f"Использован: {s['used']}")
        lines.append(f"Чанков / символов: {s['chunks']} / {s['chars']:,}")
        if s["used"] == "Да":
            lines.append(f"Какие данные извлечены: {s['extracted']}")
        else:
            lines.append(f"Причина: {s.get('reason') or 'неизвестно'}")

    # ожидаемые, но отсутствующие
    missing = [exp for exp, key in FILE_PLAN[:len(EXPECTED_FILES)] if key is None]
    if missing:
        lines.append("")
        lines.append("Ожидаемые файлы, НЕ найденные в индексе:")
        for m in missing:
            lines.append(f"  • {m}")

    lines.append("=" * 70)
    return "\n".join(lines) + "\n"


report_text = format_report()
report_filename = build_report_filename(tender_no)
with open(report_filename, "w", encoding="utf-8") as f:
    f.write(report_text)

tender_report = report_text
tender_report_filename = report_filename
tender_number = tender_no
tender_file_status = file_status
tender_answers = answers

if colab_files is not None:
    colab_files.download(report_filename)

print()
print("✅ Отчёт сформирован!")
print(f"📁 Файл: {report_filename}")
print(f"📂 Файлов проанализировано: {sum(1 for s in file_status if s['used']=='Да')} / {len(file_status)}")
print(f"❓ Вопросов обработано: {len(QUESTIONS)}")
if tender_no:
    print(f"🔖 Номер тендера: {tender_no}")
if colab_files is not None:
    print("📥 Файл скачан.")
else:
    print(f"💾 Сохранено: {os.path.abspath(report_filename)}")
