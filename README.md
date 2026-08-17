# HTML-шаблон письма МПТХ2 для Mail.ru

Файл **`email-mptkh2.html`** (он же `index.html`) — HTML-письмо по странице [МПТХ2](https://pozhavt.ru/oborudovanie/mptkh2/), подготовленное для отправки через Mail.ru.

## Что внутри
- Table-вёрстка 600px + инлайн-стили
- Адаптив через `@media` (мобильный Mail.ru / Apple Mail / Android)
- Preheader, кнопки с VML-fallback для Outlook
- Абсолютные URL картинок с pozhavt.ru
- Без JavaScript и без SVG
- Блоки: hero, ГОТВ, модули по ЗПУ, характеристики, контроль, CTA, футер

## Как отправить в Mail.ru
1. Откройте `email-mptkh2.html` в браузере и проверьте вид.
2. В Mail.ru: **Написать** → значок «три точки» / HTML / вставка исходного кода (или через сервис рассылок с HTML-полем).
3. Вставьте содержимое файла целиком.
4. Перед массовой рассылкой отправьте тест себе на Mail.ru / Gmail / Яндекс.

## Превью локально
```bash
python3 -m http.server 8080
```
Откройте http://localhost:8080/email-mptkh2.html
