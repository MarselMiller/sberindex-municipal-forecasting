# Финальная презентация

12 слайдов на русском языке, формат 16:9. Основа — реальные данные и forecasting; quantitative detection и controlled early warning явно отмечены как synthetic benchmark.

- [PDF для показа](presentation.pdf).
- [Markdown outline](presentation.md).
- [Редактируемый HTML-исходник](presentation.html), включая стили и схему pipeline.
- [Скрипт локального PDF-экспорта](export_pdf.ps1).
- [Результаты финальных проверок](verification.json).

Источники: [README проекта](../../../README.md), [методологический отчёт](../METHODOLOGY_REPORT.md), [сводка результатов](../RESULTS_SUMMARY.md), [JSON результатов](../results_summary.json), [figure manifest](../figure_manifest.csv).

## Содержание

1. Прогнозирование потребительских расходов и обнаружение структурных изменений.
2. Три разные задачи: forecasting, detection, early warning.
3. Данные и временная оценка.
4. Pipeline и доступная информация.
5. Forecasting: основная holdout MAE-таблица.
6. Пример последовательных прогнозов.
7. Online/offline change-point detection.
8. News и внешние данные: ограниченное coverage.
9. Реальный early warning: недостаточно независимых дат.
10. Synthetic early warning: проверка механизма.
11. Основные выводы.
12. Ограничения и следующие шаги.

## Использованные рисунки

| Слайд | Рисунок | Область интерпретации |
| --- | --- | --- |
| 6 | [rolling_forecast.png](../figures/rolling_forecast.png) | Реальные прогнозы, МО 21, h=1 |
| 7 | [online_comparison.png](../figures/online_comparison.png) | Synthetic detection TEST |
| 9 | [real_warning_sufficiency.png](../figures/real_warning_sufficiency.png) | Достаточность реальных данных; классификатор не обучался |
| 10 | [synthetic_event_performance.png](../figures/synthetic_event_performance.png) | Synthetic early-warning TEST |

На слайде 5 использована основная MAE-таблица вместо `forecast_mae.png`. PNG подключены относительными путями из существующего каталога `figures/`; их SHA256 сверены с manifest.

## Локальный экспорт

Для просмотра откройте PDF или HTML. HTML использует системный Arial, встроенные стили и SVG; внешних библиотек и ресурсов у него нет. Outline и HTML содержат согласованные формулировки; изменение текста в одном файле требует соответствующей правки в другом.

Из корня проекта, при наличии установленного Microsoft Edge:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File reports/final/presentation/export_pdf.ps1
```

`ExecutionPolicy Bypass` действует только для этого процесса. Скрипт печатает локальный HTML в `presentation.pdf`, отключает фоновые сетевые функции браузера и направляет HTTP-запросы на локальный неработающий proxy. Временный профиль находится внутри каталога презентации и удаляется после экспорта. Используемые параметры headless-рендера позволяют работать в данной ограниченной среде; пакеты не устанавливаются.

PDF проверен локально: 12 страниц по 960 × 540 pt, доступный для поиска русский текст, существующие PNG, отсутствие обрезания содержимого и пересечений с нижними подписями. Подробности — в `verification.json`.

F4 подготовлена из сохранённых финальных материалов. Числа округляются для показа, исследовательские результаты не пересчитываются. Существующие README, отчёты, docs, configs и scripts сохранены; модели, новые эксперименты, pytest, builder, commit и push не запускались. Каталог `reports/` игнорируется действующим `.gitignore`; презентация локальная и в Git не добавлялась.
