# Test

Демо приёма со скролл-анимацией, как на mobile-версии ralphlauren.global:
полноэкранные секции, фон движется медленнее страницы (параллакс), заголовок
и подпись всплывают при входе секции в кадр.

Открыть: `index.html` (обычный статичный файл, сборка не нужна).

## Что внутри

| Файл | Зачем |
| --- | --- |
| `index.html` | Демо-страница: три секции + описание приёма |
| `assets/parallax.css` | Стили эффекта, все классы с префиксом `px-` |
| `assets/parallax.js` | Расчёт сдвига и появление текста, ~90 строк, без зависимостей |

## Как встроить в существующий сайт

1. Скопировать `assets/parallax.css` и `assets/parallax.js`.
2. Подключить: `<link rel="stylesheet" href="assets/parallax.css">` в `<head>`
   и `<script src="assets/parallax.js"></script> `в конце `<body>`.
3. Добавить секцию:

```html
<section class="px-panel" data-depth="0.18">
  <div class="px-panel__media">
    <img src="/img/autumn.jpg" alt="">
    <!-- или <video autoplay muted loop playsinline><source src="/video/hero.mp4"></video> -->
  </div>
  <div class="px-panel__scrim"></div>
  <div class="px-panel__copy">
    <p class="eyebrow" data-reveal>Коллекция</p>
    <h2 data-reveal>Заголовок</h2>
    <p data-reveal>Описание в одну-две строки.</p>
    <a class="cta" href="/catalog" data-reveal>В каталог</a>
  </div>
</section>
```

Всё, что помечено `data-reveal`, появляется по очереди — порядок задаётся
самой разметкой.

## Настройки

- `data-depth` на секции — глубина параллакса. `0.10` — сдержанно,
  `0.18` — как в примере, `0.25` — заметно. `0` выключает.
- Фон картинкой без `<img>`: `<div class="px-panel__still" style="--px-media:url(/img/hero.jpg)"></div>`.
- Класс `px-ambient` на `.px-panel__still` — медленный наезд камеры
  (замена видео на статичном кадре).

## Совместимость

Считается в `requestAnimationFrame` по `getBoundingClientRect()`, поэтому
работает в Safari на iOS, где `background-attachment: fixed` не работает.
Без JS текст виден сразу, ничего не ломается. При системной настройке
«Уменьшение движения» вся анимация отключается.

VESNA — вымышленная марка, взята только для демонстрации.
