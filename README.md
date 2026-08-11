# Система анализа документов (Google Colab)

Однофайловый скрипт для Google Colab: загрузка документов, извлечение текста (включая OCR), чанкинг, поиск TF-IDF и ответы через DeepSeek с источниками.

## Как запустить

1. Откройте `colab_document_analyzer.ipynb` в Google Colab **или** скопируйте содержимое `colab_document_analyzer.py` в одну ячейку.
2. Запустите ячейку.
3. Загрузите файлы через диалог `files.upload()`.
4. Задавайте вопросы в интерактивном цикле.

## Поддерживаемые форматы

- PDF (текст + сканы через `pytesseract` + `pdf2image`)
- DOCX, XLSX, TXT
- Изображения: PNG, JPG, TIFF, BMP (OCR)
- Архивы: ZIP, RAR, 7z

## Параметры

- Размер чанка: 800 символов
- Перекрытие: 150 символов
- Поиск: TF-IDF + cosine similarity (без torch / sentence-transformers)
- API: DeepSeek (`deepseek-chat`)
