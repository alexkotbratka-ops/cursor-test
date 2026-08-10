# OP System — RAG для анализа документов (один файл)

Скопируйте и запустите **`op_system.py`**. Файл сам поставит зависимости, загрузит документ и ответит на вопросы через DeepSeek.

## Быстрый старт

```bash
python op_system.py
```

При первом запуске установятся пакеты (`requests`, `pypdf`, `python-docx`, `openpyxl`, `scikit-learn`, …).

API-ключ:
```bash
export DEEPSEEK_API_KEY='sk-...'
python op_system.py
```
или введите ключ в консоли, если переменная не задана.

## Возможности

- Форматы: **PDF, DOCX, XLSX, TXT/MD, CSV, ZIP, RAR**
- Несколько файлов за сессию, цикл вопросов
- Поиск фрагментов: TF-IDF
- Ответы: DeepSeek Chat API
- Показ источников

## Структура

| Файл | Назначение |
|------|------------|
| **`op_system.py`** | Полная версия в одном файле (главное) |
| `main.py` | Короткий алиас → `op_system.main()` |
| `rag.py` | Реэкспорт `RAGSystem` для импорта `from rag import RAGSystem` |
| `requirements.txt` | Зависимости (опционально, есть автоустановка) |
| `test_rag.py` | Smoke-тесты загрузки/поиска |

## Программный API

```python
from op_system import RAGSystem  # или: from rag import RAGSystem

rag = RAGSystem(api_key="sk-...")
print(rag.load_file("document.pdf"))
print(rag.ask("О чём документ?")["answer"])
```

## Примечания

- Не коммитьте API-ключ в репозиторий.
- Для RAR нужны `rarfile` и системный `unrar`.
- Сканы PDF без OCR дадут пустой текст.
