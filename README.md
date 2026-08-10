# OP System — RAG для анализа документов

Консольная система загрузки документов и ответов на вопросы по их содержимому с помощью **DeepSeek** и локального TF-IDF поиска.

## Возможности

- Загрузка: **PDF, DOCX, XLSX, TXT/MD, CSV, ZIP, RAR**
- Разбиение текста на чанки и индексация
- Поиск релевантных фрагментов (TF-IDF + cosine similarity)
- Генерация ответа через DeepSeek Chat API
- Показ источников к ответу

## Быстрый старт

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

export DEEPSEEK_API_KEY='ваш_ключ'   # Windows PowerShell: $env:DEEPSEEK_API_KEY='ваш_ключ'
python main.py
```

Скрипт спросит путь к файлу и вопрос по документу.

## Программный API

```python
from rag import RAGSystem

rag = RAGSystem(api_key="sk-...")  # или DEEPSEEK_API_KEY в окружении
result = rag.load_file("document.pdf")
print(result["chunks_count"])

answer = rag.ask("О чём этот документ?")
print(answer["answer"])
for src in answer["sources"]:
    print(src["source"], src["text"][:120])
```

## Структура

| Файл | Назначение |
|------|------------|
| `main.py` | Интерактивный CLI |
| `rag.py` | Класс `RAGSystem` (загрузка, индекс, ask) |
| `requirements.txt` | Зависимости |
| `.env.example` | Пример переменной с API-ключом |

## Примечания

- **Не коммитьте API-ключ** в репозиторий. Используйте `DEEPSEEK_API_KEY`.
- Для RAR нужны пакет `rarfile` и системная утилита `unrar`.
- Ответы строятся только по извлечённому тексту; сканы без OCR дадут пустой результат.
