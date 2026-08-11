# Colab RAG (DeepSeek + TF-IDF)

Готовый к запуску скрипт для Google Colab: `colab_rag_deepseek.py`.

Скопируйте содержимое файла в **одну ячейку** Colab и запустите.

## Возможности

- Безопасный ввод API-ключа DeepSeek через `getpass()` (без хардкода)
- Автоустановка библиотек и системных пакетов (OCR, архивы)
- Поддержка PDF, DOCX, XLSX, TXT, JPG/PNG/TIFF/BMP
- Архивы ZIP / RAR / 7z с рекурсивной распаковкой
- OCR для сканов PDF и изображений (`pytesseract`)
- Чанки 800 / overlap 150, индекс TF-IDF (без torch)
- Цикл вопросов с ответами DeepSeek и списком источников
