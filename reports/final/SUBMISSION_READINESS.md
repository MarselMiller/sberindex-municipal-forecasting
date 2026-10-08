# Готовность проекта к сдаче

Дата: 2026-10-09. Номинация: прогнозирование. Проверена рабочая версия на ветке
`research/forecast-robustness`, исходный HEAD `e0f5ad5` — интеграция устойчивости
прогнозов. До аудита рабочее дерево было чистым. Текущие исправления не закоммичены.

**Содержательная и техническая готовность: PASS WITH LIMITATION.** Требования к
материалам выполнены; доступ проверяющего к приватному репозиторию и приложениям
нужно обеспечить перед сдачей. Публичный сайт не опубликован, анонимный доступ
по внешним URL не проверялся. Этот аудит не является новым clean-clone запуском.

## Требования организаторов

| Требование | Статус | Подтверждение и границы |
| --- | --- | --- |
| Документированный код, GitHub, запуск | PASS WITH LIMITATION | [README](../../README.md), `src/`, `scripts/`, `tests/`; публичные проверки работают без исходных расходов. Реальные backtests требуют приватных inputs/outputs; доступ жюри к приватному GitHub не проверен. |
| Сохранённые YAML/JSON-конфигурации | PASS | 23 YAML в `configs/` и 21 ранее tracked JSON корректно разбираются; версии зависимостей и команды сохранены. |
| Методологический отчёт на русском | PASS | [Методология](METHODOLOGY_REPORT.md), §§1–14: цель, данные, временной протокол, модели, параметры, метрики, результаты и ограничения; подробности и глоссарий в приложениях. |
| Сравнение forecasting models | PASS | [PDF](presentation/presentation.pdf), слайд 5; [HTML](../../docs/index.html#forecasting); [сводка](RESULTS_SUMMARY.md), раздел A. Девять основных стратегий, оба Prophet и Chronos-2; h1/3/6/12, native/fallback разделены. |
| Сравнение structural detection methods | PASS | PDF, слайд 7; HTML, раздел detection; [методология](METHODOLOGY_REPORT.md), §§7–8. Online/offline сравнения и численные таблицы всех пяти методов есть в HTML/отчёте; synthetic качество отделено от real диагностики. |
| Метрики оценки | PASS | PDF, слайды 2/5/7/9/10; [forecasting_metrics.csv](forecasting_metrics.csv), [detection_metrics.csv](detection_metrics.csv). MAE в рублях; отсутствующие real показатели остаются неизвестными, а не нулевыми. |
| Архитектура непосредственно в PDF | PASS | Слайд 4: источники, календарь доступности, история/признаки, forecasting, residuals, online/offline, внешние новости/ставка/FX и real early-warning feasibility. Offline не ведёт в признаки раннего предупреждения; реальный классификатор не обучался, synthetic проверка отдельно. |
| Интерпретация реальных примеров | PASS | PDF, слайд 6; HTML, пример МО 21; методология, §5, рисунок 2. Объяснены даты выпуска/цели, знак ошибки и мартовская warmup-граница сегментации; экономическая причина не установлена. |

## Выполненные исправления

- README/HTML объясняют цель как средние безналичные расходы жителей каждого
  муниципального образования за месяц. Утверждение «на одного человека» не добавлено.
- Вместо «контрольных тревог» объяснены ложные сигналы и реальная процедура:
  параметры, включая пороги, выбирались по фиксированной синтетической сетке;
  бюджет проверялся отдельно на рядах без изменений и с одиночными выбросами.
- Выводы synthetic early warning поясняют улучшение при добавлении искусственных
  внешних предвестников относительно вариантов без них. Часть событий предвестников
  не имела; способность предупреждать реальные экономические изменения не доказана.
- В PDF дополнены архитектура, реальный пример, ограничения origin-bootstrap и
  описательный результат Persistence; сохранены 12 слайдов и 36 MAE-ячеек таблицы.
  Уточнены термины, подписи и вёрстка; существующие PNG не пересоздавались.
- HTML обновлён через `build_project_report.py`; дизайн, таблицы, переключатели,
  75 canonical определений и их смысл сохранены. Новые пояснения не создают второй
  источник результатов. В README разделены проверки после clone и реальные запуски.
- Удалено устаревшее утверждение методологии, будто презентация и предыдущий
  clean-clone audit ещё предстоят. Исторические аудиты сохранены.

## Реальный пример и права на данные

МО 21 выбран по минимальному числовому ID среди оцениваемых рядов, а не по качеству
прогноза или выразительности сигнала. Существующий рисунок и 12 его ранее включённых
строк сверены с `outputs/prophet_comparison_v1/predictions.csv.gz` и
[figure_manifest.csv](figure_manifest.csv), группа 2: даты, факты, прогнозы и SHA256
совпадают. Январский прогноз ниже факта; все шесть прогнозов июля–декабря выше факта,
поэтому residual «факт минус прогноз» отрицателен. Сохранённые diagnostic residuals
согласованы с этим рядом.

PELT и Binary Segmentation для того же МО выделили март 2024 при анализе ряда до
декабря. Это `warmup`, до мониторинга, с `eligible_evaluation=False`; граница не
входит в monitoring-оценку и не означает предупреждение, доступное в марте.
Дата уже присутствовала в [tracked отчёте сегментации](../results/E06a_offline_detection.md).
Диагностический сигнал указывает на изменение поведения ряда, но причина изменения
независимо не подтверждена. Новых построчных данных, графиков или событий не опубликовано.

Использованы только существующий проверенный рисунок и уже включённая в отчёт
метаинформация. Решения [аудита публикации](DATA_PUBLICATION_AUDIT.md) сохранены:
исходные parquet не добавлены; право их перераспространения остаётся `UNCLEAR` /
`PUBLISH_METADATA_ONLY`, условия справочника рассматриваются отдельно.
Повторное использование существующего примера не подтверждает общую лицензию на
исходные данные. Перед внешней публикацией допустимость выбранного пакета нужно
проверить отдельно; новых прав или разрешений этот аудит не устанавливает.

## Неизменность исследовательских выводов

`results_summary.json`, основной состав 72 строк `forecasting_metrics.csv`,
остальные метрики, конфигурации и research-артефакты сохранены побайтово.
270 ячеек HTML-таблиц не изменились. National/Local + LightGBM: holdout MAE
799.55 / 1212.73 / 1897.22 на h1/3/6 — минимум среди девяти стратегий общей
benchmark-выборки; устойчивое превосходство над SeasonalNaiveYoY не подтверждено.
ΔMAE = baseline − candidate: holdout +121.15 / +27.68 / +8.79;
validation −15.65 / −351.52 / −748.18. На holdout h3/6 origin-интервалы включают
ноль; даже h1 с шестью датами выпуска не даёт сильного статистического вывода.
Persistence закрывает 68–76% наблюдаемого Direct→NL разрыва: это описательное
сравнение, не причинный вклад. Сохранены 63 оцениваемых МО, 24 месяца, h12 с одной
origin и fallback, недостаточность real early-warning labels и отрицательные
результаты проверенных news/financial возможностей.

## Выполненные проверки

- 143 relevant unit / final artifact consistency tests прошли: семь
  `tests/test_final_*.py` и два теста повторяемости редакционного HTML-builder.
  Использованы синтетические fixtures; модели не обучались.
- `build_project_report.py --check`, `build_final_summary.py
  --forecasting-ablation-only --check` и `--forecasting-robustness-only --check`:
  PASS. JSON/data bundle остались прежними; побочных изменений source SHA нет.
- `scripts/check_data.py`: публичный manifest PASS, 12 файлов; `pip check`: PASS.
  Три команды `--help` из README и `scripts/run_chronos.py --help` проверены.
  Установки зависимостей, Chronos preflight/inference и полные backtests не выполнялись.
- 65 браузерных проверок основного HTML: PASS, desktop, mobile 390/320 px,
  light/dark, клавиатура, все переключатели, глоссарий, отсутствие JavaScript-ошибок
  и внешних запросов, работа без JavaScript. Repository links/anchors/assets: PASS.
- 222 относительные ссылки и anchors в девяти изменённых Markdown-документах:
  PASS; новые файлы проверены как подготовленные к review, без `git add`.
- Изолированный пакет сайта: интерфейс и реальный пример работают при отсутствии
  остального репозитория; 70 из 71 браузерной проверки PASS, проверка наличия
  приложений Persistence вне `docs/` — FAIL. Внешние приложения остаются
  ограничением, перечисленным ниже; общий статус этого сценария PASS WITH LIMITATION.
- Штатный `presentation/export_pdf.ps1`: выполнен. PDF: 12 страниц 16:9,
  все страницы отрендерены и просмотрены; текст, 36 MAE-ячеек, границы страниц,
  изображения и колонтитулы проверены. Текущее подтверждение —
  [presentation/verification.json](presentation/verification.json), старый audit
  F4 сохранён внутри как историческая запись.
- Проверки секретов, абсолютных локальных путей и состава runtime-пакета: PASS.
  Новых raw/private data нет; исходные расходы и predictions не изменялись.
  `git diff --check`: PASS. Локальные доказательства: `outputs/submission_readiness_checks/`
  (ignored; не являются публичными данными или новым экспериментом).

## Доступность материалов

Все зависимости HTML находятся в `docs/`. Проверенный минимальный пакет —
`index.html`, `assets/` и `data/` (10 файлов), размещённые как корень статического
сайта. `PROJECT_CONTEXT.md`, `TASKS.md` и служебная `README_SITE.md` для страницы
не нужны. Приватный исходный репозиторий может оставаться приватным.

12 уникальных ссылок требуют доступа к приватному GitHub. Для строк с путём общий
префикс URL — `https://github.com/MarselMiller/sberindex-municipal-forecasting/blob/main/`:

| Материал | Путь после префикса / отдельный URL |
| --- | --- |
| Исходный код | `https://github.com/MarselMiller/sberindex-municipal-forecasting` |
| Методология | `reports/final/METHODOLOGY_REPORT.md` |
| PDF | `reports/final/presentation/presentation.pdf` |
| Forecasting metrics | `reports/final/forecasting_metrics.csv` |
| Financial source audit | `reports/results/E08a_leading_financial_indicators_audit.md` |
| Financial features | `reports/results/E08b_leading_financial_features.md` |
| Financial forecasting | `reports/results/E08c_leading_financial_forecasting.md` |
| Financial warning diagnostic | `reports/results/E08d_real_financial_early_warning.md` |
| Reproduction audit | `reports/final/REPRODUCTION_AUDIT.md` |
| Data publication audit | `reports/final/DATA_PUBLICATION_AUDIT.md` |
| Canonical glossary | `reports/final/terminology.md` |
| Results summary | `reports/final/RESULTS_SUMMARY.md` |

Ещё шесть уникальных ссылок выходят из будущего site root и отсутствуют в пакете
только `docs/`: `../reports/results/national_local_persistence.md`,
`../reports/results/national_local_persistence/gap_analysis.csv`,
`../reports/results/forecast_robustness.md`,
`../reports/results/forecast_robustness/pairwise_metrics.csv`,
`../reports/results/forecast_robustness/bootstrap_origin.csv`,
`../reports/results/E06a_offline_detection.md`.
Ссылки не удалялись и автоматически не перенаправлялись. В текущем полном
репозитории они ведут к существующим tracked файлам; внешние GitHub URL также
указывают на `main`, а этот аудит выполнялся в research-ветке, без merge.

Минимальное решение после отдельного согласования: предоставить жюри доступ к
репозиторию либо опубликовать проверенный runtime-пакет и отдельно разрешённые
приложения внутри site root, затем обновить ссылки. Сам HTML, таблицы и полный
глоссарий не требуют GitHub-авторизации. Создание репозитория, изменение visibility
и размещение на хостинге не выполнялись.

## Приоритет оставшихся замечаний

- **BLOCKER перед сдачей:** не подтверждён доступ проверяющего к приватному коду
  и финальным материалам. Нужно обеспечить доступ или согласовать отдельный пакет;
  объявлять внешнюю доставку завершённой сейчас нельзя.
- **IMPORTANT:** полное воспроизведение real backtests из одного clone ограничено
  приватными данными и весами; независимая временная оценка коротка и holdout просмотрен.
  Сохранённые отчёты и проверки агрегатов доступны без нового обучения.
- **IMPORTANT перед публичной публикацией:** проверить права выбранного пакета;
  перенести согласованные приложения в site root или дать доступ к GitHub.
- **OPTIONAL:** дальнейшая проверка устойчивости на новой истории и независимая
  экономическая разметка. Это будущая исследовательская задача, а не часть аудита.

Submission review complete. Ready for final approval.
NO NEW MODEL FITS / NO METRIC CHANGES / NO SYNTHETIC-AS-REAL CLAIMS /
NO COMPETITOR REFERENCES / NO RAW DATA PUBLICATION / NO SITE DEPLOYMENT /
NO COMMIT/PUSH/MERGE.
