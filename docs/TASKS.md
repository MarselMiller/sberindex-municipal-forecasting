# План работ и критерии готовности

Дата начала этого списка: 2026-10-05.
Актуализация: 2026-10-07, по локальным артефактам и проверкам.
Первоначальные задачи настройки E01/E02/E03 переименованы в ENV01/ENV02/GIT01,
чтобы E01 и E02 обозначали согласованные исследовательские эксперименты.

| ID | Задача | Статус | Критерий завершения |
|---|---|---|---|
| ENV01 | Проверить среду | Частично подтверждена | CPU/RAM/revision E03 сохранены; отдельная .venv-chronos; один полный pytest F1: 1461 passed, 1 прежнее warning; дополнительная проверка исправленного figure cohort и сборщика: 31 passed; версии сохранены, установок нет; прежний pip check E06a успешен; точное расширение VS Code не проверено |
| ENV02 | Разобрать один ряд и минимальный запуск | Объяснение автором не проверено | Отдельный исходный критерий объяснения цели, горизонта и MAE сохраняется |
| GIT01 | Проверенный начальный Git-репозиторий | Git существует; аудит публикации не подтверждён | HEAD на начало E02a: ab3aa50; проверку всего состава публичной публикации не приписывать этому шагу |
| E01 | Локальный пилот Prophet и диагностика сопоставимости | Выполнены; диагностика принята | Совпадают множества ключей шести моделей; переход 64→63 объяснён, метрики/покрытие сверены, ошибки/резервы проверены |
| E02a | Сбор прямых пар, проверка утечки и аудит доступности | Выполнено: код, полный pytest, сохранённый аудит | 80 passed; 96 независимых сверок множеств пар; 12 точных сравнений A/h=1/L=0; 3794 строки покрытия ключей E01; защищённые файлы неизменны |
| E02b | Реализовать CatBoostDirect и выполнить первый опыт legacy на протоколе E01 | Выполнено | 98 passed; 12 выпусков, 29 fit; 1833 native / 64 fallback / 0 failed; две области метрик, точные ключи, h=1 в допуске; отчёт сохранён |
| E02 | Обучить и оценить прямой CatBoost | Первый эксперимент legacy выполнен как E02b | configs/catboost_direct.yaml → outputs/catboost_direct_v1/; MAE macro/micro, R² и покрытие проверены; годовая обученная модель не оценена из-за нуля пар; strict12 не запускался |
| E03 | Подключить Chronos-2 zero-shot и выполнить сопоставимый опыт | Выполнено | 137 passed; 39 тестов в новом окружении; smoke 2 МО × 12 месяцев успешен; 12 выпусков / 1897 native / 0 failed; годовой резерв Direct исключён; MAE macro/micro, R², coverage, revision и отчёт сохранены |
| E04a | Сравнить CUSUM, EWMA и BOCPD на синтетике; диагностика ошибок E01 | Выполнена | 212 passed; 1260 validation + 1260 test, три рабочие точки в бюджете; one-to-one/задержки/CI/исключения; 63 МО и 504 месяца после warmup; 312 групп метрик независимо сверены; отчёт/артефакты сохранены, 469 прежних файлов неизменны |
| E05 | Тренд, инфляция и заработная плата | E05a/b/c/d выполнены; обычный поиск моделей закрыт; региональная часть/поправка на цены не выполнены | Отдельные опыты компонентов; исходная MAE, те же ключи, нет двойного учёта роста YoY; незавершённые ИПЦ/зарплата/B/дефлирование не запускаются автоматически после закрытия поиска моделей |
| E05a | Аудит макроданных, причинные признаки и спецификация E05b | Код, аудит и полный pytest выполнены; региональные источники недоступны | Реальные официальные загрузки/ошибки и SHA256, классы A/B/C, даты/версии, однозначная география, отдельное подтверждённое/сценарное покрытие 64 МО и исторических примеров, тесты без сети; модели не обучались |
| E05b | Проверить сезонные тренды и представления месяца K0/K1/K2 | Выполнен: код, тесты, smoke/full, метрики и отчёт | 392 passed; 12 выпусков, 58/58 новых fit, 0 failed; точные E01 ключи/факты; отдельные strategy/native оценки, явный годовой резерв, схемы19/17/19, реальные categorical+one_hot12; старые результаты сохранены |
| E05c | Проверить вклад доступных национальных макропрогнозов | Выполнен: код, тесты, smoke/full, независимая сверка и отчёт | 501 passed; 73 A строки проверены офлайн; K0/19→24→29 признаков, исторический join/исходные пары сохранены; 12/12 выпусков, 58/58 fit, 0 failed; три области MAE/R² и provenance; смешанный результат без смены протокола |
| E05d | Финальные National/Local decomposition и LightGBM benchmark | Выполнен: код, pytest, smoke/full, метрики, независимые сверки и отчёт | 666 passed; 7 вариантов/12 выпусков/87 новых fit/0 failed; C0 и ориентиры без fit; 19 признаков, точные исходные/ratio пары, A/B, national MAE, 4 контраста/robustness; 2495226 lineage-строк/217 артефактов/2525 прежних файлов сверены; отчёт 12 таблиц PASS |
| E06a | PELT + Binary Segmentation | Выполнен: код, pytest, smoke, synthetic, real/prefix, проверки и отчёт | 865 passed; прежние 1260 validation + 1260 test рядов E04a; оба penalty=4 в бюджете; 208 групп метрик/728 CI; 63 МО/160 полных кандидатов/1512 префиксов; отдельная online-справка, ограничение native PELT сохранено, 3975 прежних файлов неизменны |
| E06b | News/events | Выполнен: код, pytest, сбор, реальные events/features, аудит и отчёт; историческое покрытие ограничено | 1064 passed; 93 документов / 93 canonical events / 768 keys / 26 features+flags; 4 historical PDF по archive-trust, regional historical=0; независимая сверка PASS; E07 classifier не запускался |
| E07a | Подготовка данных и слабой разметки | Выполнена: код, реальные таблицы, аудит, отчёт и один полный pytest | Source audit; отдельная news v3; зафиксированный конечный weak criterion; 1536 cases, причинные признаки, temporal availability, manual queue; без classifier fit |
| E07b | Full-panel early warning feasibility; условные B0–B4 | Выполнен: полная разметка, признаки, аудит, отчёт и один полный pytest; условное обучение пропущено по gate | 2190 МО, 73 weak events/6 onset-дат/71 МО; k1 train14/test4 positives, train1/test2 onset-даты; k3 train/test0; оба gate отказали, все 24 календарных варианта отказали; 138 числовых признаков, 1247 passed, 213 защищённых файлов неизменны |
| E07c | Controlled synthetic early-warning benchmark | Completed: код, unit tests, smoke/full, метрики, bootstrap, 3 графика, независимая сверка и один полный pytest; research phase closed | 600/200/300 ряда по24месяца;360/120/180событий;TRAIN fit/VALIDATION threshold/TEST;1367passed/1warning;160защищённыхфайлов и frozenYAML сохранены;результаты только synthetic |
| E08a | Leading Financial Indicators: source / availability / vintage audit | Выполнен; полный vintage/availability ledger остаётся E08b | Ветка research/e08-leading-indicators; 10 series A2/B7/C1, шесть shortlist families / восемь representations; primary key rate+official FX; B только sensitivity, unsecured C и top10 исключены; metadata/схемы/ссылки/causal assumptions проверены; без fits, feature matrix, metrics, commit/push |
| F1 | Final Results Summary | Выполнена: единая сводка, builder, тесты, проверка чисел и графиков | 72 forecasting / 10 detection / 68 early-warning записей; real/synthetic/diagnostic/not_evaluated; точные ключи и MAE/R²; один builder и один full pytest 1461 passed/1 прежнее warning; 990 финальных сверок; 9 групп/11 PNG; после исправления TEST-filter 31 targeted passed; новых fits/experiments нет |
| F2 | Конкурсный README на русском | Выполнена: переписан из F1; редакторский проход завершён | 2664 слова / 2172 вне таблиц после сокращения; воспроизведение −38%; 4 результата, 9 моделей, отдельные online/offline и real/synthetic warning; 5 таблиц/4 PNG/AI disclosure/ссылки сохранены; 98 ячеек сверены; Mermaid syntax проверен; F1 artifacts неизменны; clean-clone не выполнялся |
| F3 | Самостоятельный методологический отчёт | Выполнена: Markdown, table provenance, проверки и финальная редактура | 6665 слов вне таблиц после редакторского сокращения на 9.71%; 15 предметных разделов и приложения A–F; 12 таблиц/9 существующих PNG; 167 числовых/null ячеек сверены; 78 relative links; README и 44 прежних reports побайтово сохранены; 106 F1 source hashes проверены; новых fits/experiments нет; PDF и clean-clone не выполнялись |
| F4 | Финальная презентация | Выполнена; доступна в проверенном GitHub commit 6ada526 | 12 русских слайдов, Markdown outline, HTML source и PDF 16:9; одна MAE-таблица и четыре существующих PNG; SHA и доступность повторно проверены в F5 |
| F5 | Clean-clone reproduction audit | Выполнен: PASS WITH LIMITATION | Новый GitHub clone 6ada526; Python 3.12.10/pip 25.0.1; fresh install + pip check; 1432 passed/34 deselected/1 warning без новых fit; CLI/imports/links/figures/PDF PASS; clone clean; private data/full experiments NOT TESTED; исторические provenance paths сохранены |
| F6 | Финальный audit и cleanup публичного репозитория | Выполнен: PASS WITH LIMITATION | Base3371a44 синхронизирован с GitHub; навигация и шесть private/ignored links исправлены; claims/AI/135 file links+4anchors/11PNG/PDF/manifests/security PASS; 23 historical paths сохранены с SHA-обоснованием; .vscode/build/temp ignored; docs-only, без pytest/fits/experiments/commit/push |
| F7 | Data publication and reproducibility audit | Выполнен: PASS WITH LIMITATION | 90 решений/1877 inventory entries; 12 public artifacts (2.39 MB), SYNTHETIC byte-copy demo; primary terms/provenance; stdlib preflight public12PASS; новый GitHub base e50e044 + reviewed overlay; research9/sources5MISSING в clone; source rights/full experiments не подтверждены; старые local-only links сохранены; без fits/experiments/pytest/commit/push |
| R01 | Зафиксировать протокол проверки | Для E02b/E03/E04a/E05b/E05c/E05d/E06a зафиксирован; независимая реальная проверка требует решения | Временные границы/выборка E01 сохранены, L=0/vintages неподтверждены; A принимает датированный архив; holdout просмотрен; Chronos checkpoint позже backtest; E04a split seed/ID раздельны, E06a повторяет просмотренный synthetic test; offline future access явно отделён от online |
| F01 | Выполнить сопоставимый пилот Prophet | Выполнен как E01 | outputs/prophet_comparison_v1/ и reports/results/E01_prophet_comparison.md; это пилот, не полный набор МО |
| F02 | Улучшить прогнозирование и проверить фундаментальную модель | Chronos-2 проверен в E03; обычный model search закрыт после E05d | Сопоставимые опыты сохранены с ограничениями; прирост E05d на просмотренном holdout не является независимым подтверждением или выбором окончательного победителя |
| C01 | Определить шок и протокол разметки | Синтетический E04a и weak E07a подготовлены; независимых реальных меток нет | E07a размечает трёхмесячный сдвиг сохранённой ошибки SeasonalNaiveYoY, не уровень расходов и не истинный экономический шок; human review pending |
| C02 | Сравнить онлайн-детекторы и ретроспективную сегментацию отдельно | Онлайн E04a и offline E06a выполнены; реальная оценка без меток не выполнена | Раздельные таблицы/доступ к будущему; precision/recall/F1/FAR/miss/локализация+whole-series CI; real/prefix hindsight и native-library ограничения; общий победитель/early warning не объявлены |
| N01 | Подготовить новости с датами доступности и территорией | E06b сохранён; E07a добавляет ограниченную v3 | 17 official decision cores под явным archive trust; 4 PDF объединены с релизами; исторических regional/municipal событий0; полная новостная интенсивность не установлена |
| N02 | Проверить риск будущих шоков и вклад новостей | Feasibility E07a/E07b выполнен; predictive utility не оценена из-за недостатка дат/меток | Для продолжения нужны достаточные причинные train/test периоды; baseline, оценка предупреждений, время упреждения и сравнение без/с новостями |
| I01 | Проверить объединённую модель и интерпретировать ошибки | Не начата | Проверен вклад компонентов, показаны успех, ложная тревога и пропуск при их наличии |
| D01 | Подготовить русский отчёт и PDF-презентацию | F1 сводка, F2 README, F3 методологический отчёт и F4 презентация готовы | reports/final/RESULTS_SUMMARY.md и CSV/JSON — единый источник; 12 слайдов и PDF доступны в GitHub clone; F5 не пересобирал результаты |
| D02 | Проверить финальную воспроизводимость и публикацию | F5 code audit и F6 финальный audit выполнены с ограничениями; полное воспроизведение и права на данные не подтверждены | Код/навигация/claims/security/figures PASS; private data/full experiment reproduction NOT TESTED; historical paths документированы как исключение; права на исходные/производные данные NOT VERIFIED; финальный commit отдельно |

## Карточка текущей задачи
ID: E08a — Leading Financial Indicators source/availability/vintage audit.
Выполнен 2026-10-07 только в research/e08-leading-indicators, base 70c95d6.
Main и результаты E01–E07/F1–F7, README/final reports/presentation заморожены.
Проверено 10 series в 6 семействах: A=2 (key rate/USD-RUB), B=7, C=1 (unsecured monthly).
Shortlist: 6 families / 8 series; USE для A, USE_WITH_CAVEAT только exploratory для B;
top10 max deposit rate B и полный unsecured C — DO_NOT_USE в текущем shortlist.
Созданы E08a report и два metadata catalog files. JSON/CSV parity, evidence IDs,
обязательные поля, shortlist, относительные links/content scan проверены локально.
Полное покрытие information set на всех origins и historical vintages не проверено;
current bank XLSX/BBS parity и OFZ first-release timing остаются открытыми.
Никаких feature matrices, correlations, forecasts/MAE, classifier/PR-AUC/F1,
packages, model fits, новых outputs или commit/push. Main не изменён.
Минимальные official HTTP format probes — только память; MOEX filter не подтверждён
(17972 rows на однодатный запрос), current calendar query не доказал старый schedule.
Действующий /reports/* скрывает новый report; .gitignore и index не менялись.
Следующий конкретный шаг E08b: отдельно зафиксировать acquisition/as-of specification
и проверить primary key-rate/FX loader, own-r availability, anchors/праздники/coverage.
Для B нужен отдельный archive/sensitivity gate; обучение этим аудитом не разрешено.
[E08a audit](../reports/results/E08a_leading_financial_indicators_audit.md).

## F7: сохранённая предыдущая задача
ID: FINALIZATION / F7 — data publication and reproducibility audit. Выполнена 2026-10-07.
Base: main, e50e044; рабочая папка до F7 чиста.
Проверяется только data/documentation layer. Research phase закрыта.
Новые fits, ML-эксперименты, пересчёт результатов, commit/push запрещены.
Verdict: PASS WITH LIMITATION для proposed public data/documentation layer.
19 новых / 5 изменённых файлов, 12 data artifacts; metadata без news bodies,
numeric macro values и справочника. SYNTHETIC demo: 76 рядов, 1824 наблюдения,
46 событий. Public preflight: 12 PASS; локальные research: 9 PASS; sources: 5 MISSING.
Новый GitHub clone содержит 236 base files и 24 проверенных overlay files.
Public: 12 PASS из clone/parent CWD; research: 9 MISSING; sources: 5 MISSING.
Private inputs/outputs/weights/venv не копировались. Все 86 relative links F7
проверены; 18 старых local-only links в 3 архивных отчётах ограничивают global check.
Права на Sber Parquet и dictionary требуют independent official confirmation;
уже tracked rolling example с 12 real y_true также требует проверки прав.
228 защищённых файлов и 1815 inventory hashes неизменны; whitelist/security/size
и git diff --check PASS; staged files: 0. Full real reproduction NOT TESTED.

Final review перед commit: PASS для 19 новых / 5 изменённых F7 файлов.
Raw/private/restricted payloads отсутствуют; UNCLEAR + PUBLISH = 0.
Derived visual/report artifacts сохранены и отдельно классифицированы;
это не подтверждение прав на raw source data. 18 local-only links документированы:
восемь целей нужны повторной F1-сборке, не inference/public preflight.
110 relative links F7 без пропусков; public12PASS; security/path scan PASS;
пять SYNTHETIC demo files совпали с сохранённым smoke, 228 защищённых files неизменны.
Новых sources/fits/experiments/pytest/builder/commit/push нет. F7 ready for commit.
[F7 audit](../reports/final/DATA_PUBLICATION_AUDIT.md) содержит evidence и команды.
Следующий шаг: review кандидатов и подтверждение официального grant,
включая отдельный dictionary и права уже tracked real example.

## F6: сохранённая предыдущая задача
ID: FINALIZATION / F6 — финальный audit и cleanup. Выполнена 2026-10-07.
Verdict: PASS WITH LIMITATION. Base3371a44 синхронизирован с origin/main и GitHub.
README верхняя навигация ведёт к отчёту/PDF/сводке/ограничениям; report5 private
hyperlinks сняты, summaryAIlink исправлен; presentationREADME tracked-статус верен.
Claims и disclosure согласованы; 135 file links/4 anchors, 11 PNG/PDF SHA,
35 figure source SHA/21 table source SHA проверены; таблицы/формулы сохранены.
Security235tracked: рабочих секретов нет; 23 строки C historical provenance
в9файлах оставлены побайтово, runtime A и current instruction B после cleanup0.
.gitignore исключает .vscode/build/dist/temp; только FINAL_AUDIT.md разрешён
как новый report. Data/full reproduction NOT TESTED, права NOT VERIFIED.
[F6 audit](../reports/final/FINAL_AUDIT.md) содержит перечень и обоснование.
Git diff --check PASS; research/models/metrics/configs/tests/outputs не менялись;
pytest/fits/experiments/PDF export/commit/push отсутствуют.
Следующий шаг: финальный commit проверенных файлов; приватные данные не добавлять.

## F5: сохранённая предыдущая задача
ID: FINALIZATION / F5 — clean-clone reproduction audit. Выполнена 2026-10-07.
Verdict: PASS WITH LIMITATION. Cloned commit 6ada52638299415936742669b24d12b6792f64d7.
Новый отдельный sberindex_python_mvp_repro; tracked-tree clean до/после install,
тестов и CLI. Python 3.12.10, pip 25.0.1, 26 runtime/test distributions;
requirements.txt + requirements-ruptures.txt установлены без старых venv/cache,
pip check PASS. Optional Prophet/LightGBM/Chronos не устанавливались.
No-fit pytest: 1432 passed, 34 deselected, 1 warning, 276.25 с, exit 0;
27 префиксов исключают настоящие estimator/Ruptures fit, полный pytest NOT TESTED.
Четыре help работают также из parent CWD; 13 imports, 18 YAML, 37 README links
и 11 PNG SHA проверены. PDF прочитан из clone, соответствует сохранённому SHA.
Chronos preflight exit 1 из-за отсутствия приватного CSV; data/full reproduction
NOT TESTED. Исходные данные, outputs и weights не копировались; новых fits нет.
Статическая проверка 231 tracked-файла: рабочих секретов не найдено; operational
paths переносимы. 23 исторические строки машинных путей в девяти report/JSON
файлах сохранены; глобальное требование отсутствия абсолютных paths не выполнено.
README/DATA_NOTICE уточнены и сверены с фактическим clone; code/config/models/tests,
цифры и методология не менялись. [F5 audit](../reports/final/REPRODUCTION_AUDIT.md)
содержит команды, границы проверки и remaining limitations.
Git diff --check PASS; .vscode/ не затронута; commit/push отсутствуют.
Следующий шаг: отдельный финальный аудит материалов и условий публикации.

## F3: сохранённая предыдущая задача
ID: FINALIZATION / F3 — единый методологический отчёт. Выполнена 2026-10-07.
Финальный editorial pass выполнен: 7382 → 6665 слов вне таблиц (−9.71%).
Отредактированы §§1–14 и вводный текст; §12 выделяет пять научных выводов.
Все 12 таблиц, 5 display-формул, 78 relative links, 9 figure paths,
source-строки, AI disclosure и приложения A–F сохранены побайтово.
Небольшие изменения подписей рисунков 4/5 не меняют статус или интерпретацию.
Проверены numeric/claim consistency, Markdown, ссылки, SHA и git diff --check;
запрошенные запрещённые выражения не найдены. В этом проходе не выполнялись
pytest, model fits, builder, PDF, презентация, clean-clone или commit/push.

Начальный HEAD: 2940ba9 (F2); F1: f8541d5. Созданы
[METHODOLOGY_REPORT.md](../reports/final/METHODOLOGY_REPORT.md) и
[report_table_manifest.csv](../reports/final/report_table_manifest.csv).
Самостоятельное изложение: постановка, данные, temporal protocol, модели,
forecasting results/ablations, online/offline, news, real feasibility,
controlled synthetic warning, интерпретация, ограничения, воспроизведение и AI.
Текущий объём — 6665 whitespace-delimited слов вне pipe-таблиц, включая
заголовки, подписи и приложения A–F; 585 строк. 12 таблиц, 9 прежних PNG, 5 формул.
Table manifest содержит 21 source-запись для 12 таблиц, фильтры, статус,
единицы, округление и SHA. 167 числовых/undefined-ячеек совпадают с источниками;
54 разных numeric tokens в тексте найдены в frozen sources, контекст claims
дополнительно проверен. Настройки и основные comparative claims сверены.
78 вхождений relative links / 50 уникальных targets существуют; 9 PNG
совпадают с figure manifest. Markdown parse/render подтверждает таблицы,
images и fences; отдельный LaTeX/PDF/browser renderer не запускался.
Forbidden competition/jury/score/русские эквиваленты и абсолютные машинные ссылки отсутствуют.
Реальный classifier не обучался; synthetic механизм не перенесён на real
quality; h12 direct fallback и просмотренный holdout явно сохранены.
README, 31 прежний final-файл и 13 results-отчётов побайтово неизменны;
106 исходных provenance-файлов F1 совпадают с сохранёнными SHA.
Реально выполнены read-only PowerShell/rg/git, локальные Python CSV/JSON/hash
сверки и Markdown parse; git diff --check PASS. Новых fit, experiments,
tuning, pytest, builder, сетевых обращений, установок, PDF, clean-clone,
commit/push не было. Исходная .vscode/ не затронута.
Новые final-файлы пока исключены действующим /reports/ в .gitignore;
правило и Git index не менялись. Публичный состав требует отдельного аудита.
Следующий конкретный шаг: презентация на основе F1 и F3; clean-clone — F5.

## F2: сохранённая предыдущая задача
ID: FINALIZATION / F2 — конкурсная витрина README. Выполнена 2026-10-07.
Редакторское уточнение F2 выполнено отдельным проходом: 3166 → 2664 слова
(−15.86%); вне таблиц 2674 → 2172 (−18.77%); воспроизведение 324 → 201 (−37.96%).
Порядок и названия разделов, все 5 таблиц, 4 PNG, AI disclosure, ограничения,
команды и все 41 вхождение относительных ссылок сохранены. Убраны смешанные
формулировки об оценке early warning и вкладе news; доступность истории
SeasonalNaiveYoY объяснена при первом употреблении. Mermaid содержит только
компоненты: history + online state + external features → early warning;
offline не входит в признаки. Фраза о выполнении foundation-критерия удалена.
Синтаксис всех строк простого flowchart проверен; браузерный renderer не запускался.
98 ячеек сверены с final CSV, исследовательские числа в тексте сохранены;
31 файл Source of Truth побайтово неизменен. В этом проходе выполнялись только
текстовые/числовые/link/hash проверки и git diff --check: PASS; CLI, pytest,
model fits, builder, установки, сеть, commit/push не запускались.
Следующий шаг остаётся прежним: методологический отчёт из F1, clean-clone — F5.

Исходная версия README сохранена в Git HEAD f8541d5; до редактирования
побайтовое совпадение с HEAD проверено, дублирующая копия не создавалась.
README полностью переписан на русском из reports/final/ — единственного
Source of Truth. Структура по результатам, без хронологии исследовательских этапов:
4 результата, Mermaid, данные/backtest, 9 forecast-стратегий, Chronos-2,
online/offline detection, news, реальные ограничения warning и synthetic S0–S3,
статусы проверки, 8 ограничений, воспроизведение, структура и AI disclosure.
MAE округлена непосредственно из полного CSV до 1 знака; synthetic метрики
до 3 знаков. 98 табличных ячеек, 8 k-значений, fallback/bold и 24 группы
inline/comparative claims сверены; исходные CSV/JSON согласованы по 150 строкам.
Использованы только прежние forecast_mae, online_comparison,
real_warning_sufficiency и synthetic_event_performance PNG. Реальные сигналы
diagnostic, real classifier не обучался; просмотренный holdout, годовой fallback,
L=0/архивные допущения, weak truth и Chronos overlap явно обозначены.
Все относительные ссылки/anchors существуют; file links и directory contents
подтверждены Git. Локальный Markdown parse/render проверил 5 таблиц, 4 code
fences и 4 images; Mermaid проверена логически, отдельный graph renderer не запускался.
Все 4 CLI-примера реально выполнены без fit: run.py --help,
run_national_local_lightgbm.py --help, run_early_warning_synthetic.py --help,
run_chronos.py --config configs/chronos_zero_shot.yaml --preflight-only; exit 0.
Preflight подтвердил 64 исходных МО / 1897 raw / 1890 evaluable keys.
Quickstart документирует Python 3.12 и requirements-ruptures.txt для обязательных
offline-тестов; команды установки/pytest при F2 не выполнялись.
31 файл reports/final/ сохранён побайтово, прежний код/config/outputs не менялись;
исходная .vscode/ не затронута. Новых fits/experiments/tuning, сети, установок,
сборки F1, commit/push нет. Изменены только README и два документа статуса.
Непроверены GitHub/browser rendering Mermaid, clean-clone и публичный состав.
Следующий конкретный шаг: методологический отчёт из F1; clean-clone отдельно в F5.

## F1: сохранённая предыдущая задача
ID: FINALIZATION / F1 — Final Results Summary. Выполнена 2026-10-07.
Research phase закрыта на be6d993; новые fits, источники, tuning, holdout changes
и исследовательские эксперименты запрещены. Старые reports/results/ и outputs
сохранены; README/.gitignore/исходная .vscode/ не менялись, commit/push нет.
Новый scripts/build_final_summary.py и final_*.py читают сохранённые artifacts,
проверяют одинаковые forecasting keys/truth, пересчитывают MAE macro/micro/R²,
detection matching/counts и early-warning метрики; существенное расхождение
останавливает сборку. Созданы все 9 запрошенных файлов reports/final/,
criteria CSV, AI disclosure draft, проверочные JSON и 9 групп/11 figures.
94 targeted tests перед сборкой PASS. Один builder: exit 0, 72/10/68 records.
Один full pytest: 1461 passed, 1 прежнее RuntimeWarning, 334.87 s, exit 0.
При визуальной проверке исправлена figure 9: metrics_event.csv содержит
VALIDATION и TEST; график/figure_data теперь используют 8 TEST rows, 180 events.
Builder повторно не запускался; только этот график и его export пересохранены
из прежних metrics. Добавлены 5 synthetic cohort regressions; первый расширенный
targeted: 98 passed / 1 failure из-за текста сообщения ошибки, затем уточнён контекст
сообщения без изменения expectations; конечный figure + summary targeted: 31 PASS.
990 source/hash/serialization/link/null checks PASS, 11 figures просмотрены.
Original builder SHA и final code SHA сохранены раздельно; full pytest начат
до узкой plot correction, конечные затронутые F1 правила проверены дополнительно.
Единственное старое текстовое расхождение: Prophet YAML «НЕ ЗАПУСКАЛОСЬ»;
сохранённые результаты подтверждают завершённый E01, старый файл не исправлен.
Неоценённые real classifier metrics = null; real early warning не обучался.
Непроверены clean-copy reproduction, публичный состав и реальные shock labels.
Следующий конкретный шаг: F2 — обновить README из reports/final/, затем отчёт
и презентацию; ни один из этих следующих материалов в F1 не создавался.

## E07c: сохранённая предыдущая задача
ID: E07c — synthetic controlled benchmark, не оценка реальных экономических шоков.
HEAD на начало df580ac; E01–E07b/README/исходная .vscode/ не меняются.
Новая configs/early_warning_synthetic.yaml зафиксирована до TEST evaluation,
SHA d7277fd317e64d543224ccb034f960edbe0a555203fbc8ddb10c90e97f6dde61.
600/200/300 независимых cohorts по24месяца; event fraction0.6, unanticipated0.25
среди событий, false cues0.35 среди контролей; stochastic channels/noise,
onset17…21, lead1…3, direct generator truth (не новый weak criterion).
S0–S3: fixedC1/no classweight/seed42/max_iter2000; sklearn отсутствует,
используется установленный SciPy для эквивалентной L2logistic objective.
TRAIN-only E07b preprocessor; validation F1 с monthly control budget≤1/12,
ties higher threshold, TEST не выбирает параметры/порог/генератор.
120целевых tests PASS; первый115passed с pandasFutureWarning, затем
добавлены API/metadata regression и исправлен concat пустых frames без
изменения expectations; второй120passed безwarnings.
Smoke40/16/20 на отдельныхseeds завершён exit0;8model entries/8thresholds,
peak0.142GiB. Полный --stage full завершён exit0 за37.363s/peak0.177GiB,
CPU/BLAS/jobs1, dense numeric estimate0.00746GiB. Сохранены26400observations,
1100series,660events,14300feature rows/70numericcandidates,28600label auditrows.
TRAIN/VAL/TEST events360/120/180; unanticipated90/30/45; false-control cues84/28/42.
TEST monitored k1=2722(180positive)/k3=2482(540positive), classbalance6.613%/21.757%.
S3 eventrecall77/180=.42778(k1),111/180=.61667(k3); alertprecision.37379/.51628,
medianlead1/2,meanlead1/2.063; falsealerts129/104, FAR.56870/.50282 per12monitoredmonths.
S2−S1 AP+.0343/+.0618, eventrecall−.05/+.1333; S3−S2 AP+.1851/+.1482,
eventrecall+.2611/+.2167. Incremental utility относится только к генератору.
S3 recall withcue.54074/.74074, withoutcue.08889/.24444;75%cue reference
не строгая граница из-за chance/prioralerts. Stressfalsecontrols FAR1.119/1.6,
validationbudget не гарантирован на TEST subsets. Все5scenarios сохранены.
48whole-series bootstrap95%intervals(B500) сохранены; S3eventrecallCI
k1[.35399,.50276],k3[.54611,.68927]. Автоматические TESTexamples:
test_000002 success/lead1, test_000000 control falsealert, test_000006 miss;
правило S3k3/lexicographicfirst зафиксировано,3PNG построены и проверены.
Независимые saved-file comparisons38034 PASS(labels27/models119/evaluation37888);
probabilities восстановлены поJSON/rawfeatures, пороги и метрики пересчитаны
без проектных scoring APIs; это численные сверки, не независимые real labels.
Один полный pytest:1367passed/1warning/in299.69s(0:04:59),exit0; предупреждение
из прежнего test_direct_training сnonfinite values, новыеE07c tests безwarnings.
154code/test/config файла неизменны во времяpytest. 160защищённыхпрежнихфайлов,
HEADdf580ac и SHA новойYAML неизменны; весьстарыйoutputtree не пересканирован.
Команды реально выполнены: targeted pytest(двапрогона послеузкихfixes),
run_early_warning_synthetic.py --stage smoke / --stage full / --stage report,
independent raw audits, один fullpytest и preservation check; exit0.
Точныекоманды/seed/версии/Gitdirty/code/artifactSHA вmanifest, логи ипроверки
вoutputs/e07c_checks. Сети/установок/commit/push/измененийстарыхoutputs нет.
Результаты outputs/early_warning_synthetic_v1/, отдельный smoke output,
отчёт reports/results/E07c_synthetic_early_warning.md.
E07c completed; experimental research phase closed. FINALIZATION ONLY — README,
методологический отчёт, сводка результатов, презентация, clean reproduction.

## E07b: сохранённая предыдущая задача
ID: E07b — full-panel feasibility по неизменному E07a, условные B0–B4 только
при достаточных данных. HEAD на начало bf99872; исходная .vscode/ не изменяется.
Новая configs/early_warning_full_panel.yaml зафиксирована до полного подсчёта,
SHA1fed3b0ae32855c70999263fb9950614c653de85641806684fbeec71cff54dc6.
Результаты outputs/early_warning_full_panel_v1/; отчёт
reports/results/E07b_early_warning_full_panel.md сформирован из сохранённых CSV.
Все2190UID×12origin сохранены, eligibility на каждом prefix; не future cohort.
Weak criterion/цензурирование/L0/active exclusion/k1,k3 не изменялись.
26280 cases на k, 52560 суммарно; registry73events/6onset-дат/71МО(3.242%).
k1 rawknown12183(73pos/12110neg), unknown14097; at-risk-evaluable12069
(71pos/11998neg). k3 rawknown8101(156pos/7945neg), unknown18179;
at-risk-evaluable8087(154pos/7933neg). Unknown не становится negative.
Cutoff2024-07/testoriginsAug–Nov сохранён: k1train2023(14pos/2009neg),
1warning/1positive-onset date; test3974(4pos/3970neg),2warning/2onset dates.
k3train0/test0. Все12cutoff×2k сохранены, ни один не проходит
30/10positive rows и3/2positive onset dates. Обучение B0–B4 не разрешено
gate; sklearn отсутствует, установки нет. Feasibility full выполнен exit0,
629.966s/peak0.239GiB; equivalence768pilot residual keys PASS.
Целевые новые unit tests80passed (adapter3/labels19/features20/models24/audit14).
Full features выполнен exit0: 26280строк, 138числовых candidates A40/B36/C10/D52,
70provenance files; 1513.989s/peak0.215GiB, оценка памяти0.569GiB, CPU/jobs1.
Finite value-cell coverage (без missing flags): A90.162%, B74.855%, C87.5%,
D82.051%; зависимостей позже собственной O и отсутствующих coverage-records0.
Train-only audit k1 оставил13A; B/C/D0 после allmissing/constant/exactduplicate
filter; k3 no training rows. Во всей панели7temporal news vectors, в train0
varying news values/flags, в test6values/3flags и2информативные даты. Национальные
копии не независимы. B0–B4/инкрементальные/event metrics и оценённые примеры
не вычислялись; пустые файлы имеют схемы и skip-status, это не нулевые метрики.
Pilot feature equivalence768rows/126numeric columns PASS; независимая сверка
разметки118checks и135проверок сохранённых таблиц/artifact hashes PASS.
Первый буквальный JSON-evidence check отказал из-за последних float digits;
decoded numeric check использует исходный atol1e-8, данные/criterion не менялись.
Full feature run выдал pandas DtypeWarning на смешанных evidence metadata;
reader до финального pytest исправлен на low_memory=False, числовая сверка PASS.
Один полный pytest: 1247 passed, 1 warning in270.73s; exit0, код/тесты/config
неизменны во время проверки. Warning из прежнего test_direct_training с inf.
213защищённых файлов E01–E07a и SHA новой YAML неизменны; большой аудит9005
старых файлов повторно не выполнялся. Проверки/точные команды/логи сохранены
в outputs/e07b_checks/, seed42/versions/Git dirty/команды/время/память/SHA —
в manifest.json. Реально выполнены smoke/features-smoke, feasibility, features,
baselines (skip до fit), audit-report и preserve-after. Commit/push/сети/
установок/обучения нет; README и исходная .vscode/ не изменялись.
Следующий отдельный этап после E07b: финальная сборка отчёта/презентации/README.

## E07a: сохранённая предыдущая задача
ID: E07a — подготовка и feasibility, без обучения классификатора.
Новая конфигурация configs/early_warning_feasibility.yaml зафиксирована до
подсчёта событий, SHA3027d57dce91fd87d83dbb7d5f1ef462019fd0e20853466d4ce51f58445a2c65.
configs/news_events_v3.yaml задаёт отдельную политику admission;
news_events_v1/v2 и старые отчёты сохранены.

Source audit:180saved snapshots/93URL documents; два новых GET,
157889bytes; 17/17 официальных решений ЦБ2023–2024 (9unchanged/8increase).
Исторически допускается только первая фраза решения под явным доверием
датированному оригинальному архиву, без независимого oldSHA. Формат HTML/PDF
не критерий. Lenta retrieval-only;46updated>published и25updated==published
статей разделены. Publication/event/article-update/catalogue-update/retrieval
не смешиваются. Новые сетевые файлы изолированы от старого cache.
Четыре PDF E05c объединены с теми же решениями, не независимые новые источники.
20.12.2024 позже последнего O и не увеличивает исторические features.

news v3:197snapshot/core versions,93documents,89canonical groups;
17decision events,20historical URL-documents к последней O (16release+4PDF).
768исходных ключей64МО×12O сохранены. 8O/512cases с news30d>0;
7уникальных temporal news vectors против6v2. Regional/municipal historical=0;
source_archive_complete=false. Ноль частичного корпуса не означает нет события;
национальные копии64МО не независимы.

Weak criterion: четыре непосредственно прошлых календарных residuals;
median center; scale=max(1.4826MAD,.03median(abs(saved prediction)),1рубль);
три будущих residuals одного signed направления ≥3scale от center.
Все7месяцев finite, без fill. Сезонность — сохранённый causal h1SeasonalNaiveYoY
с прежней поправкой YoY; цель — устойчивый сдвиг ошибки, не уровень расходов.
Same-direction gap≤2 или unrecovered regime — continuation; recovery два
signed z<1.5 месяца по frozen baseline. Opposite onset может означать возврат
ошибки к прежнему baseline, а не истинное экономическое потрясение.
Три даты onset T/confirmation T+2/warning O разделены. Метки known не раньше
O+k+2; при delayed history также после зависимости regime prefix; right end
censored, не0. Active risk filter использует только подтверждённый prefix state.

4weak events на3onset dates;1536cases (k1/k3). Split cutoff2024-07,
testO2024-08…11: k1train63(1pos/62neg) на1O/1event date,
test-evaluable120(0pos/120neg) на2O; k3train0/test-evaluable0.
Код classifier технически можно было бы подогнать на1positive, но надёжной
валидации/оценки recall, пропусков, упреждения и news increment здесь нет.
Доступна описательная проверка FP на negative test; она сейчас не выполнялась.

Собраны expense/residual/online-prefix/macro/news candidates с provenance;
online E04fixed params, без refit; macroA current calendar year forecasts,
не месячные факты. Offline breakpoints только diagnostic, неfeatures/labels.
Все52news колонки описаны, не выбраны автоматически; отдельный train-only audit.
8 строк очереди (7 уникальных случаев) включают кандидаты/control/missing;
pending_human_review, независимых человеческих меток0.
outputs/early_warning_feasibility_v1/, outputs/news_events_v3/,
outputs/early_warning_news_audit_v1/; отчёт
reports/results/E07a_early_warning_feasibility.md.

Целевые синтетические tests:103passed (labels34/features40/sources26/integration3).
Один финальный полный pytest: 1167 passed, 1 warning in 203.43s (0:03:23); exit=0.
763 проверки новых artifacts PASS; 292 старых файлов и protocol SHA неизменны.
132 code/test/config файлов не изменились во время полного pytest.
В train k1 все52news столбца постоянны (3полностью пусты); вклад news не идентифицируется.
Проверка сохранности ограничена292используемыми старыми файлами, без повтора
большого9005file audit. Первый integration guard обнаружил10savedE04placeholder
строк без forecast metadata; сохранение1471иmissing покрыто regression.
Начальная новая сборка сохранена отдельно *_initial_integration; окончательная
повторена после исправления смешанных дат и label provenance, без смены критерия.
Установок/обучения/commit/push/публикации нет. Следующий шаг — человеческая
проверка очереди и решение по дополнительной временной истории, не auto E07b.

## Предыдущая завершённая задача E06b
ID: E06b — выполнен 2026-10-07 как воспроизводимый pipeline и ограниченный
реальный корпус: event table и feature table действительно построены.
configs/news_events_v2.yaml; scripts/build_news_events.py; шесть news_* модулей
и шесть test_news_* файлов. Окончательные результаты outputs/news_events_v2/;
синтетический smoke outputs/news_events_smoke_v2/; отчёт
reports/results/E06b_news_events.md. Forecasting model search остаётся закрыт;
E07 classifier и реальное объединение с detector/macro states не выполнялись.

Сбор: 112 запросов / 6376900 учтённых байтов;
24 фиксированных архивных дня Lenta (по 15-му числу каждого месяца), до трёх
статей на день; календарь ЦБ и выбранные rate releases. Сохранены 180
снимков, 93 уникальных документов (CBR 21, Lenta 72) и
93 canonical events; объединено точных дублей документов 0.
Группы заголовок+день не являются независимо размеченными экономическими событиями.
Ошибки сохранены: Pravo TLS EOF, по одному разрыву соединения Lenta/CBR;
МЧС HTTP 200 только как access probe, исторический адаптер не подтверждён.

Исторически допустимы к последней O только 4 PDF ЦБ за 2024 год:
16 февраля, 26 апреля, 26 июля, 25 октября. Это ранее принятое доверие
датированному оригинальному официальному архиву E05a, с проверкой PDF/text/archive
SHA и заголовка; независимых SHA-снимков 2024 нет. Использованы только заголовок
и три строки, macro_policy/announcement/unknown/national. Современные HTML/index
версии retrieval-only доступны в 2026 и исключены из всех origin E01.

768 неизменных ключей: 64 МО × 12 O (2023-12…2024-11), включая МО 1471.
26 признаков + 26 missing flags; definitions/windows/units/geography/availability
в feature_dictionary.csv; 312 строк покрытия feature×O. Origin с regional news:
0; только national: 4; полностью без допущенных news 30d: 8.
Случаев только national: 256; без news 30d: 512.
source_archive_complete=false; nonmissing и нулевые counts не доказывают полноту
исторической новостной базы или отсутствие события. Safe merge отклоняет
future detector states и offline/hindsight PELT/BinSeg; это проверенная функция,
а не выполненный merge с реальными детекторами.

Первая сборка v1 и её конфигурация сохранены: пять снимков ошибочно относили имена
Путин/Зеленский к городу Владимир. v2 исправляет collision aliases с требованием
непосредственного территориального квалификатора; regional context сам по себе
не превращает имя человека в название МО. Frozen corpus сохранён, features E01
не изменились; geography_fix_audit.json сохраняет исправленные snapshot IDs.

Финальный pytest: 1064 passed, 1 warning in 209.41s (0:03:29), exit=0; SHA code/config/tests неизменны в run5.
Run3 перед сбором: 1036 passed; затем precision обновления отделена от публикации,
добавлено 9 regression cases; целевые tests 52 passed. Первому целевому прогону
мешал внешний TEMP (WinError 5), повтор внутри проекта прошёл без смены ожиданий.
Старый процесс collector уже импортировал parser; frozen corpus не менялся,
все его updates содержат exact timestamps и не превышают retrieval.
Runtime collect 765.836 s; build 50.388 s;
общий pytest 212.183 s с обёрткой. Независимая проверка:
42469 сверок/assertions и ячеек, 19968 values +
19968 flags, max difference 0.0;
проверены saved coverage/provenance/SHA и семь report-summary полей.
Контроль 9005 прежних файлов PASS; 33 пакета и Git HEAD
неизменны; .vscode сохранена. В .gitignore добавлено только исключение raw E06b.
Нет установок, экспериментального обучения, commit/push или публикации.

Ограничения: архив частичный; historical regional/municipal news нет;
география/темы не проверены по независимой разметке; sentiment причинность не доказаны;
исторические варианты названий МО не ограничены датой действия, fuzzy отсутствует.
E07 данные готовы по схеме, но их историческое news-покрытие недостаточно для
обоснованного сравнения предупреждений с новостями/без. Следующий отдельный шаг:
протокол E07 с независимой разметкой будущих событий и решением по историческим
версиям/достаточности новостей. Неподтверждённые версии не включать автоматически.

## Предыдущая завершённая задача E06a
ID: E06a — выполнен 2026-10-07: код, полный pytest, smoke, synthetic
validation/test, real/prefix диагностика, независимые сверки и отчёт.
Отдельные configs/offline_detection.yaml, scripts/run_offline_detection.py,
outputs/offline_detection_v1/ и reports/results/E06a_offline_detection.md.
Четыре новых модуля offline_* и четыре файла tests/test_offline_*; прежний код
прогнозирования/детектирования и E01–E05d не изменены. requirements-ruptures.txt
фиксирует ruptures==1.1.10; проверен Windows CP312 wheel. Версии 32 прежних
пакетов неизменны, всего 33 пакета; pip check завершился с кодом 0.

Сохранены генератор и seeds E04a: 1260 validation + 1260 test рядов.
Основное событие — изменение уровня продолжительностью >=3 месяцев;
изменения наклона/дисперсии дополнительные, no_change/outlier — контроли.
Причинный SeasonalNaiveYoY h=1; e=y_true−y_pred, c/s по первым четырём ошибкам,
как в E04a. Offline fit использует все 12 месяцев ошибок; оценка/FAR — только
восемь monitoring-месяцев. Warmup-точки сохранены вне оценки. Пропуски не
заполняются; календарные даты сохраняются при вычислении cost по конечным
наблюдениям. BP — первый наблюдаемый месяц правого сегмента; terminal n исключён.
Метки событий не передаются fit. Фиксированы l2, min_size=2, jump=1,
penalties=[.5,1,2,4,8,16]. Выбор только на validation, бюджет <=1 FP/12
наблюдаемых месяцев отдельно на обоих контролях, без расширения бюджета
или обязательного BP. Оба выбраны с penalty=4: validation FAR no_change/outlier
PELT .425/.825, BinSeg .350/.608333. Selection seal предшествует test;
после test параметры неизменны.

Основной test: 360 событий. PELT P/R/F1=.449393/.308333/.365733,
miss=.691667, FP/12=.566667; BinSeg .440758/.258333/.325744,
miss=.741667, FP/12=.491667. Для обоих медианы абсолютной ошибки локализации
и смещения BP равны 0 только среди обнаруженных событий; это не online delay.
F1 CI [.336120,.392915]/[.295944,.351567]: 500 bootstrap-ресэмплов целых рядов,
seed=300000. CI условны на выбранные параметры, пересекаются; парная значимость
не проверена. PELT имеет большую точечную F1 только в offline-сравнении.
Online CUSUM/EWMA/BOCPD из E04a показаны отдельно (F1 .484321/.450450/.451977),
общий победитель не выбран. Сценарии/сила/шум, контрольные FAR и исключения сохранены.

Реальные данные: 63 МО, полных BP 81/79, из них monitoring 59/57 и warmup 22/22.
Все 126 полных сегментаций complete, failed=0. Исходные sample/coverage на 64 МО
сохранены; МО 1471 без конечных warmup-ошибок не сегментируется.
1512 диагностик префиксов: 1134 complete, 378 not_ready, failed=0.
Fit не выполняется до четырёх конечных ошибок; каждый префикс использует только
собственные доступные данные. Сопоставление с полными BP проводится после fit,
one-to-one с допуском +/-1 месяц. Медиана first-prefix-month−full-BP-month=1;
даты пересматривались у 22/81 PELT и 22/79 BinSeg кандидатов, исчезновения
после обнаружения — у 9/81 и 12/79; максимальный разброс дат — один месяц.
Временных prefix-only появлений 33/39, всех prefix-оценок 458/440,
строк траекторий 1920. Это ретроспективная диагностика устойчивости,
не online detector или предупреждение. Без независимой реальной разметки
real precision/recall/F1 не вычислялись.

Обнаружено ограничение native PELT 1.1.10 при min_size=2: 86 objective gaps
в 4896 проверках произвольных коротких сигналов; synthetic — 93/10080,
включая selected validation 6/test 11; новые real/prefix проверки — 0/630.
Native pruning воспроизведён независимо. Непрореженный DP служит нижней границей
cost и не заменяет модель в benchmark. Все gaps/IDs/penalties/objectives
сохранены; глобальный optimum не обещан. В отчёт добавлена диагностика из
сохранённых JSON; исходный текст/SHA, добавленный фрагмент и команда сохранены.

Полный pytest: 865 passed, 1 warning, 263.16 s, exit=0 (прежний Inf-тест E02a).
11 файлов кода/тестов/конфигурации не менялись после pytest. Smoke: 42 исходных
replicate=0, 84 пары метод/ряд, 28.1002 s, failed=0; без selection/test/real.
Synthetic: 752.1687 s, failed=0; real: 110.6777 s до графиков/отчёта;
сумма этих двух стадий 862.8465 s. Независимая synthetic-проверка PASS:
1435192 сверки, 208 групп метрик/728 CI/15120 candidate-series/5040 selected-series,
max difference=5.0023e-12. Полная проверка PASS: 114408 новых/общих сверок,
max difference=9.09495e-13; успешная synthetic-проверка повторно не вычислялась,
подтверждены неизменные SHA 36 файлов и исходного checker. Независимо сверены
prefix calibration/partitions/costs/dates/trajectories; полная повторная сверка
каждого вторичного поля каждого prefix BP не заявлена. Bootstrap CI повторно
не вычислялись при аудите: проверены оценки, границы и единица ресэмплирования.
Численная проверка финального отчёта PASS: 3141 сверка, 11 таблиц, 2356 ячеек,
1019 числовых ячеек, восемь PNG и их источники/SHA; максимальное отличие .0005
соответствует округлению до трёх знаков. Примеры МО 21/25/37 и test_000120
выбраны по ID, не по качеству. 3975 прежних файлов, включая .vscode, сохранены;
HEAD=313cf851a8ca26d5077ba2fefb675e57412f5457. Commit/push/публикации и обучения
прогнозных моделей не было.

Точные команды:
```powershell
.\.venv\Scripts\python.exe -B -X utf8 -m pytest -p no:cacheprovider --basetemp=outputs/e06a_checks/pytest_full_run1
.\.venv\Scripts\python.exe -B -X utf8 scripts/run_offline_detection.py --config configs/offline_detection.yaml --stage smoke
.\.venv\Scripts\python.exe -B -X utf8 outputs/e06a_checks/verify_smoke.py
.\.venv\Scripts\python.exe -B -X utf8 scripts/run_offline_detection.py --config configs/offline_detection.yaml --stage synthetic
.\.venv\Scripts\python.exe -B -X utf8 outputs/e06a_checks/verify_results.py --stage synthetic
.\.venv\Scripts\python.exe -B -X utf8 scripts/run_offline_detection.py --config configs/offline_detection.yaml --stage real
.\.venv\Scripts\python.exe -B -X utf8 outputs/e06a_checks/verify_results.py --stage full
.\.venv\Scripts\python.exe -B -X utf8 outputs/e06a_checks/finalize_report_audit.py
.\.venv\Scripts\python.exe -B -X utf8 outputs/e06a_checks/verify_report.py
.\.venv\Scripts\python.exe -B -X utf8 outputs/e06a_checks/verify_final_preservation.py
```
Доказательства в outputs/e06a_checks/: test_result.json, smoke_verification.json,
synthetic_verification.json, full_verification.json, report_verification.json,
final_preservation.json, диагностика библиотеки и неизменяемые записи запусков.
L=0 и vintages неподтверждены; 12 месяцев ошибок, четыре warmup и восемь
monitoring-месяцев, Gaussian synthetic и адаптация forecaster ограничивают вывод.
E04a synthetic test и реальный holdout уже просмотрены; новой независимой
проверкой этот опыт не назван. Следующий шаг — отдельный E06b news/events,
затем E07 early warning; оба запланированы, не выполнены. Возврат к подбору
прогнозных моделей не предусмотрен.

## E05d: завершённая предыдущая задача
ID: E05d — выполнен 2026-10-07: код, полный pytest, smoke/full, метрики,
независимые построчная/отчётная проверки и отчёт.
Фиксированы C0 — сохранённый CatBoostDirect K0 E02b, L0 — LightGBMDirect,
CN/LN — CatBoost/LightGBM на R=y/N. Три ориентира — сохранённые YoY,
ProphetAuto/ProphetYearly. Ни C0, ни ориентиры не переобучались.
Отдельные configs/national_local_lightgbm.yaml,
scripts/run_national_local_lightgbm.py, outputs/national_local_lightgbm_v1/
и reports/results/E05d_national_local_lightgbm.md.

N — медиана конечных фактов всех доступных МО текущего месяца, вся панель
2190 МО. Знаменатель доли пропусков — МО, наблюдавшиеся к этому месяцу;
будущая полнота не определяет старый состав. 24 месяца, used=2072…2149,
N>0 во всех месяцах, 50521 конечный R; распределения/пропуски сохранены.
Национальный forecaster — заранее фиксированный неизменный SeasonalNaiveYoY
через адаптер baseline_predict, шаг h+L. Actual future N только для диагностики.
19 K0 признаков/legacy/параметры E02/seed/допуск/64 ID и временные границы E01
сохранены; MO1471 не исключён заранее. Macro/one-hot/trend/news/weather/new
calendar не добавлены. CN/LN: delta_R=R_target−anchor_R;
y_hat=max(0,N_hat×(anchor_R+predicted_delta_R)). Исторические X до r−L,
labels/target N только при r+h+L≤O; построчное происхождение сохранено.
relative_delta_1 сохраняет старый порог 1 на собственной шкале.

LightGBM 4.6.0, проверенный Windows wheel, установлен единственной новой
зависимостью; 31 прежний пакет основной .venv сохранил версии, pip check код 0.
CPU/native Dataset/train, sklearn не добавлялся. Фиксированы regression_l1,
300 итераций, lr=0.05, leaves=31, seed=42, threads=2; deterministic/force_col_wise
для воспроизводимости, остальные defaults и полный resolved footer сохранены.
Requirements только requirements-lightgbm.txt; tuning отсутствует.

Полный pytest: 666 passed, 1 warning, 168.77s, код 0 (прежний Inf-тест E02a).
13 проверенных файлов не менялись после pytest. Smoke март: 189 ключей/модель,
9/9 fit, 0 failed, 35.4063s раздел/60.9009s вызов; независимая проверка PASS.
Полный запуск возобновил март без fit: 12/12 выпусков, 87/87 новых fit,
90 сравнений подписей ключей/X/labels, 0 failed, код 0.
Runtime разделов включая smoke 388.5208s; полный вызов до отчёта 409.1861s.
Точные команды:
```powershell
.\.venv\Scripts\python.exe -B -X utf8 -m pytest -p no:cacheprovider --basetemp=outputs/e05d_checks/pytest_full_run1
.\.venv\Scripts\python.exe -B -X utf8 scripts/run_national_local_lightgbm.py --config configs/national_local_lightgbm.yaml --origins 2024-03
.\.venv\Scripts\python.exe -B -X utf8 outputs/e05d_checks/verify_smoke.py
.\.venv\Scripts\python.exe -B -X utf8 scripts/run_national_local_lightgbm.py --config configs/national_local_lightgbm.yaml
.\.venv\Scripts\python.exe -B -X utf8 outputs/e05d_checks/verify_results.py
.\.venv\Scripts\python.exe -B -X utf8 outputs/e05d_checks/verify_report.py
```

Все семь моделей имеют те же 1897 raw/1890 конечных ключей E01.
Каждый C0/L0/CN/LN: 1833 native / 64 fallback / 0 failed;
с фактом 1827/63/0. Годовые декабрь2023/h12: 0 пар, только expenseSeasonalNaive,
никакого fit; MAE 4322.14 — резерв, не native. Ориентиры native на всех 1897.
A=1890 и B=1827; 63 годовых резерва исключены явно только из B.
MAE macro/micro и pooled R², метрики по МО/датам/покрытие/резервы сохранены.
Validation MAE C0/L0/CN/LN: h1 3104.94/2733.30/1391.41/1370.55;
h3 2918.37/2880.02/2789.24/2819.74; h6 2570.63/6378.07/5162.10/5525.34
(одна дата выпуска); годовых validation нет.
Holdout h1 986.22/1003.28/823.42/799.55;
h3 1939.17/1449.82/1225.69/1212.73; h6 3349.50/2463.21/2132.67/1897.22.
Национальная MAE отдельная, один вес на O/h: validation h1/3/6
1053.14/2347.72/3975.00; holdout h1/3/6/12 325.06/430.62/1448.86/3958.00.

Algorithm effect смешанный: L0−C0 holdout h1 +17.06, h3 −489.35, h6 −886.29;
LN−CN −23.87/−12.97/−235.45. Decomposition снижает holdout агрегаты:
CN−C0 −162.80/−713.48/−1216.83; LN−L0 −203.73/−237.09/−565.99.
L0−C0/h3 имеет 3 улучшенных/3 ухудшенных даты, сентябрь даёт 98.51% суммы
снижений; медианная Δ +40.45. LN−L0/h1 улучшает 2 даты и ухудшает 4,
август даёт 67.74% суммы снижений, медианная Δ +31.87.
Единственная улучшающая дата только validation/h6/LN−L0, декабрь2023,
который является единственной оценочной датой этой группы. Значимость
и причинность не проверены; holdout просмотрен, независимый тест не объявлен.

Независимая проверка результатов PASS: 98 агрегатных/420 групп по датам,
5733 строки по МО/98 покрытия, все 30 national forecasts/120 training записей,
2495226 строк lineage, 87 fit, 217 артефактов; max_abs=2.91e-11.
C0/три ориентира совпали с сохранёнными прогнозами без численной разницы.
Attempt1 checker остановился на object-типе пустого годового CSV; исправлено
только явное преобразование типа checker, attempt1/успешный attempt2 сохранены.
Отчёт PASS с первой попытки: 12 таблиц, 8585 проверок, 5777 ячеек/4801 числовой
элемент. 2525 прежних файлов и версии 31 старого пакета не изменились.
Доказательства: outputs/e05d_checks/. Источники E01–E05c/.gitignore/.vscode/
сохранены, commit/push отсутствуют. Данные/веса/построчные outputs не публикуются.
Сопоставимость C0 основана на реконструкции пар по неизменным данным/коду:
E02b первоначально не сохранял построчные хеши training.
L=0/vintages расходов не подтверждены; N зависит от состава наблюдаемых МО;
decomposition изменяет шкалу признаков/ошибки и национальный forecaster вместе.

После финальной сверки E05d обычный forecasting model search закрыт.
E06a (PELT + Binary Segmentation) выполнен отдельно выше; далее E06b (news/events)
и E07 (early warning) запланированы, не выполнены; ретроспективная сегментация не подменяет
предупреждение будущего события. Региональные CPI/зарплата/B/дефлирование
остаются незавершёнными; закрытие поиска моделей не объявляет их выполненными.

## E05c: завершённая предыдущая задача
ID: E05c — выполнен 2026-10-07: код, полный pytest, smoke/full, метрики и отчёт.
Фиксированное сравнение: M0 — сохранённый CatBoostDirect K0 E02b; M1 +
национальная инфляция A; M2 + реальное потребление A. Month/sin/cos и исходные
параметры/seed/legacy/delta/anchor не менялись; 19/24/29 числовых признаков.
Отдельные configs/macro_forecast.yaml, scripts/run_macro_forecast.py,
outputs/macro_forecast_v1/ и reports/results/E05c_macro_forecast.md.

До fit офлайн проверены все 73 A строки (45/28), определения/единицы/годы/даты
и 27/6/43 SHA артефактов/кода/raw E05a. Ошибок значений/дат нет, E05a не менялся.
Инфляция — медиана участников опроса, Dec/Dec, %, не собственный прогноз ЦБ;
потребление — прогноз ЦБ годового реального объёма. Производная середина
диапазона явно помечена; официальный центр → midpoint → NaN. Для каждого
показателя добавлены значение, ширина, midpoint_used, missing, publication_age_days.
На собственную r/O выбирается последний доступный прогноз на год целевого
месяца; факт/другой год/будущая версия/B не подставляются. Строки с NaN не удаляются.

Полный pytest: 501 passed, 1 warning, 118.95s, код 0; прежний Inf-тест E02a.
Точные команды:
```powershell
.\.venv\Scripts\python.exe -B -X utf8 -m pytest -p no:cacheprovider --basetemp=outputs/e05c_checks/pytest_full_run1
.\.venv\Scripts\python.exe -B -X utf8 scripts/run_macro_forecast.py --config configs/macro_forecast.yaml --origins 2024-03
.\.venv\Scripts\python.exe -B -X utf8 scripts/run_macro_forecast.py --config configs/macro_forecast.yaml
.\.venv\Scripts\python.exe -B -X utf8 outputs/e05c_checks/verify_results.py
.\.venv\Scripts\python.exe -B -X utf8 outputs/e05c_checks/verify_report.py
```
Smoke март: 189 случаев/модель, 6/6 fit, 0 failed, native=macro-available=189,
PASS; полный запуск возобновил март без fit. Full: 12/12 выпусков, 58/58 новых
fit, 0 failed, код 0. M0/YoY/ProphetAuto/ProphetYearly не переобучались.
1897 raw/1890 конечных ключей на каждую из шести моделей, 64 ID и все факты
совпали точно с E01/E02b, МО1471 не исключено заранее. В каждом M-варианте
1833 native / 64 fallback / 0 failed (с фактом 1827/63/0). Годовые 64 запроса
только SeasonalNaive: h12/декабрь2023 имеет 0 пар, fit не вызывается.

Три области: A=1890, B=C=1827 точных ключей; 63 годовых резерва явно исключены
из B/C, остаются в A. На h1/3/6 ключи/метрики всех трёх областей совпали.
M1/M2 имеют макроданные на всех 1897 текущих ключах, native без макро=0;
64 годовых случая с макроданными всё равно fallback. MAE macro/micro в
номинальных рублях, pooled R², показатели по МО/датам, разности и покрытие сохранены.
Holdout MAE macro M0/M1/M2:
h1 986.22/958.49/1245.64; h3 1939.17/2322.28/2020.66;
h6 3349.50/3403.00/3111.59; h12 4322.14 у всех, только резерв.
Validation: h1 3104.94/3481.44/3036.65; h3 2918.37/3124.05/3005.42;
h6 2570.63/2397.75/2072.36 (одна дата); годовых validation нет.
М1 улучшает holdout h1, ухудшает h3/6; M2 после M1 улучшает h3/6, ухудшает h1.
M2 относительно M0 хуже на holdout h1/3, лучше на h6 на 237.91 руб.
YoY лучше всех M-вариантов на holdout h1/3/6. Общего устойчивого выигрыша,
статистической значимости или причинного влияния не установлено.

На модель 831742 повторно суммируемые пары по 29 непустым O/h. Инфляция
доступна в 100%, потребление в 92.586% (770077 пар; задачи 83.224…95.371%).
На задачу 6…22 исторические даты, 6…20/2…8 публикаций инфляции/потребления,
4…13/2…8 значений. Источник национальный, числа МО не являются независимыми
макронаблюдениями. Ширина инфляции всегда NaN, её midpoint/missing постоянно 0;
column_states всех признаков сохранены. Пары, порядок, delta и исходные19
совпали в 30 O/h; 60 сравнениях M1/M2 с реконструированным M0. Ограничение:
E02b первоначально не сохранял построчных подписей, контроль основан на его
неизменных данных/коде/счётчиках; сохранённые прогнозы M0 совпали точно.

Независимая проверка PASS: 126 агрегатных/540 групп по датам/7182 строк по МО,
42 разности, покрытие, реальные58fit, augmentedX и 2495226 training/5691 forecast
lineage-строк; max_abs=3.64e-12. Проверены 187 артефактов, 1517 защищённых
файлов, 31 пакет, HEAD и все SHA. Отчёт сверён отдельно: 13 таблиц/4077 проверок,
PASS. Доказательства: verification_result_attempt2.json и report_verification.json
в outputs/e05c_checks/. Attempt1 остановился на неверном расширении старого
журнала E02b в diagnostic checker (JSON вместо CSV); исправлен только путь,
неудачная попытка сохранена, моделей/результатов ошибка не затронула.
Сохранены config/seed/версии/Git HEAD+dirty/команды/SHA/происхождение.
Код/config/тесты не менялись после pytest. E01–E05b, .gitignore/.vscode и
окружения не трогались; установки, сеть, тюнинг, commit/push отсутствуют.

Только национальные A с доверием датированному архиву; независимых старых
снимков содержимого нет. Региональные ИПЦ/зарплата, B и дефлирование не исследованы.
История короткая, макропубликаций мало, одна годовая дата и ноль native h12;
сроки публикации/пересмотры расходов неподтверждены, L0 остаётся допущением.
Holdout уже просмотрен, не новый независимый тест. Следующий конкретный шаг:
обсудить фиксированный результат и отдельный протокол с более длинной историей
и подтверждёнными датами расходов; региональные источники/дефлятор требуют подготовки.

## E05b: завершённая предыдущая задача
ID: E05b — выполнен 2026-10-06: код, полный pytest, smoke/full и отчёт.
Два вопроса: фиксированные SeasonalTrendLinear/Damped и представления месяца
K0/K1/K2. Макрофакторы не подключены; источники повторно не скачивались.
Конфигурация configs/trend_calendar.yaml; scripts/run_trend_calendar.py;
результаты outputs/trend_calendar_v1/, отчёт reports/results/E05b_trend_calendar.md.
Протокол E01 сохранён: 64 ID, 1897 raw / 1890 конечных ключей на модель,
все ключи/факты/допуск/разбиение/хеши проверены; МО 1471 не исключено заранее.
Полный pytest: 392 passed, 1 warning, 48.52s, код 0. Точная команда:
.\.venv\Scripts\python.exe -B -X utf8 -m pytest -p no:cacheprovider --basetemp=outputs/e05b_checks/pytest_full_run1.
Smoke --origins 2024-03: 189 случаев на модель, 6/6 fit, 0 failed, код 0.
Полный запуск без --origins: 12/12 выпусков, 58/58 fit K1/K2, 0 failed, код 0;
март использован по сохранённым проверенным хешам без повторных fit.
K0/YoY взяты из E02b/E01 и не переобучались. Все параметры/seed/legacy/delta
не менялись. K0/K1/K2 имеют 19/17/19 признаков; month_category строка месяца
цели, cat_features явно задан, реальный one_hot_max_size12; days_in_month,
time_index и прочие значения/типы/NaN сохранены. Для 30 O/h проверены ключи,
delta/X signatures и старые счётчики; исходный E02b построчных подписей
не сохранял, K0 реконструирован по неизменному коду/данным.
Тренды: native/fallback/failed на h=1/3/6/12 — 567/191/0, 441/191/0,
252/191/0, 0/64/0; общих нативных 1260, исключённых 630 конечных случаев.
K0/K1/K2: 758/0/0, 632/0/0, 443/0/0, 0/64/0. На h=12 нет native;
validation h6 также не имеет native трендов. Полная MAE h12 относится
к резервам, не обученной годовой модели. Короткая история/одна годовая дата
и неподтверждённый L0 сохраняются как ограничения; holdout не слепой.
Тренды лучше YoY на holdout h1/3/6, но хуже на validation h1/3.
K1 ухудшает holdout h1/6; K2 улучшает h3/6, ухудшает h1 относительно K0.
Общее преимущество one-hot/статистическая значимость не доказаны.
Сохранены обе области MAE macro/micro/R², метрики по МО/датам, покрытие,
исключённые ключи, реальные fit-параметры/признаки, manifest/versions/dirty/SHA.
Независимо проверены формулы 3794 строк трендов, 11 таблиц отчёта и 58 fit;
пересчитаны 84 агрегатные группы, 360 групп по датам и 4725 строк по МО,
max_abs=7.28e-12; проверены 30 наборов пар и все 105 артефактов.
1153 защищённых файла, 31 пакет и HEAD сохранены. Проверки: outputs/e05b_checks/.
Код/конфигурация/тесты после полного pytest не менялись; зависимости,
.gitignore/.vscode и E01–E05a не трогались, commit/push/тюнинг отсутствуют.
Следующим отдельно согласованным шагом стал E05c на фиксированном K0:
сначала национальная инфляция A, затем потребление A; каждый вклад отдельно.
B допускается только отдельным сценарием; региональные ИПЦ/зарплата/дефлятор
остаются незавершёнными. Национальные A можно проверять без ожидания
региональной части; не называть их региональными и не складывать механически
с ростом расходов. Последующий E05c выполнен отдельно выше; артефакты E05b сохранены.

## E05a: завершённый аудит с неполными источниками
ID: E05a — подготовка данных и спецификации, без обучения прогнозных моделей.
Разрешены официальные источники, публичные загрузки, код признаков, расчёты и тесты.
Результаты: outputs/macro_data_audit_v1/; отчёт reports/results/E05a_macro_data_audit.md.
Конфигурация/команда: configs/macro_data_audit.yaml и scripts/audit_macro_data.py.
География проверена: 2190 МО / 77 регионов; E01: 64 МО / 37 регионов.
Росстат: 0 загрузок/чисел, статус C, TLS EOF и web timeout/502; обхода доступа нет.
Банк России: 38 файлов источников, 227 прогнозных строк A73/B154; общий
Excel остаётся B. Исторический архив A принимается с явно указанным доверием
датированной официальной версии, независимых старых SHA-снимков нет.
На 1890 конечных случаях E01 инфляция Dec/Dec и потребление доступны строго;
среднегодовая инфляция/номинальная зарплата — только сценарно. Региональные
ИПЦ/зарплата/реальный рост отсутствуют на всех датах; дефлятор не готов.
3072 строки сетки/64 МО сохранены, 1897 ключей E01 проверены точно.
Обучение global legacy: 48 O/h, 1206017 повторяющихся пар, 2451 макрозапрос
на собственную r; пар/МО/дат столько же, сколько в E02a.
Даты/версии фильтруются на собственную дату исторического примера r; сценарные
лаги B отделены от подтверждённой доступности. Происхождение/возраст/NaN сохраняются.
Полный pytest: 298 passed, 1 warning, 29.56s, код 0;
точные команды в PROJECT_CONTEXT и отчёте. Аудит, manifest и независимая
проверка готовы: 550 файлов/31 пакет/HEAD сохранены, 1104 join-проверки совпали.
Обучение, подбор параметров, MAE E05, commit/push не выполнялись.
На момент окончания E05a E05b не выполнен: формулы/окна/phi0.9/резервы
фиксированы в отчёте и PROJECT_CONTEXT; последующий E05b выполнен отдельно выше.
Региональная часть требует проверенных выгрузок и сведений о vintages;
строгие CPI/зарплатные варианты не готовы, временные границы сохраняются.
Непроверенные региональные vintages не блокируют опыт тренда без внешних данных,
но строгий региональный макроэксперимент сейчас нельзя объявить готовым;
отдельный национальный опыт A выполнен как E05c.

## E04a: завершённая предыдущая задача
ID: E04a — выполнена 2026-10-06: код, тесты, запуск, метрики и диагностика.
Цель: обнаружение уже начавшихся изменений; не предупреждение будущих шоков.
Результаты: outputs/online_detection_v1/; отчёт reports/results/E04a_online_detection.md.
Конфигурация: configs/online_detection.yaml; минимальный прогон отдельно v2.
Полный pytest: 212 passed, 1 warning, 15.56 s, код 0; прежний Inf-тест E02a.
Точная команда: .\.venv\Scripts\python.exe -B -X utf8 -m pytest -p no:cacheprovider --basetemp=outputs/e04a_checks/pytest_full_run3.
Полный эксперимент: .\.venv\Scripts\python.exe -B -X utf8 scripts/run_online_detection.py --config configs/online_detection.yaml;
код 0, 493.235 s, 1260 рядов на каждый synthetic split, 18 validation-кандидатов.
Main test: 360 устойчивых сдвигов; F1 CUSUM/EWMA/BOCPD
0.484321/0.450450/0.451977; пропуски 61.39%/65.28%/66.67%, задержка 1/0/1.
Все рабочие точки соответствуют фиксированному validation-бюджету.
Реальный мониторинг май–декабрь 2024: 63/64 МО, 504 месяца на метод,
252 warmup ошибки, 12 календарных пропусков; сигналов 104/97/14,
совпадений всех трёх в одном МО/месяце — 1; реальные шоки не размечены.
Независимо сверены 312 групп метрик (max_abs=1.11e-16), календарь/one-to-one,
61992 причинные калибровки и 20664 BOCPD posterior; 469 файлов/31 пакет
не изменились. SHA256/seed/UTC/команды/git dirty/артефакты сохранены.
Старый код/данные/config/outputs E01–E03 и .vscode/ не трогались, commit/push нет.
Макропризнаки исключены из E04a. После неё отдельно согласован E05a;
на момент окончания E04a обучение и оценка E05b не начинались;
последующие E05b и E05c выполнены отдельно и описаны выше.

## E03: завершённый Chronos-2 zero-shot
ID: E03 — выполнена 2026-10-06.
Цель: проверить amazon/chronos-2 zero-shot с независимыми МО на протоколе E01.
Модель/revision: 29ec3766d36d6f73f0696f85560a422f50e8498c, пакет 2.3.2;
CPU/float32/batch_size=1/seed=42, cross_learning=False, точка=q0.5.
Данные/даты/64 ID/допуск/разделение E01 сохранены; МО 1471 не исключено заранее.
Соперники — сохранённые E01/E02, чистый Direct без годового резерва.
Окружение .venv-chronos Python 3.12.10 создано отдельно; CPU torch 2.8.0+cpu,
transformers 5.18.0; 47 пакетов зафиксированы в requirements-chronos-lock.txt.
Обычный полный pytest в прежней .venv без Chronos: 137 passed, 1 warning,
17.25 s; новый набор в .venv-chronos: 39 passed, 4.28 s; все команды — код 0.
Проверены cutoff/календарь/NaN/давность, q0.5, независимость МО, failed без
резерва, точные ключи, полная и общая успешная области, отсутствие годовой
обученной модели Direct. Обычные тесты веса не скачивают.
Smoke: ID 21/25 заранее по порядку, декабрь 2023, конечный путь 12×2,
0.43 s инференс, 53.07 s загрузка, пик процесса 0.78 GiB.
Полный запуск: 12 выпусков, 1897 native, 0 failed, 0 fallback; 1890 случаев
с фактом, семь отсутствующих фактов сохранены. Общая успешная область полная.
Инференс 147.62 s, загрузка 25.97 s, до итогового run_status 186.77 s.
Независимо пересчитаны 82 числовые группы метрик/5166 строк МО и сверены
41 ячейка MAE/21 дополнительная ячейка отчёта; max_abs=1.82e-12.
3735 месячных позиций проверены; 12 журналов ошибок пусты. SHA256 всех
295 защищённых файлов и состав 30 пакетов исходной .venv не изменились.
Результаты: outputs/chronos_zero_shot_v1/; smoke в отдельной папке;
отчёт: reports/results/E03_chronos_zero_shot.md.
Holdout MAE Chronos h=1/3/6/12: 1947.35 / 2526.33 / 4150.59 / 9163.47.
Validation: 2316.65 / 3064.93 / 2754.10, годовых случаев нет.
Превосходство не подтверждено, результаты не изменены ради выигрыша.
Публикация модели 20.10.2025 позже backtest 2023–2024; отсутствие пересечения
предобучения с данными не доказано. Holdout уже просмотрен; не слепой тест.
Прежние .venv/requirements/код/данные/E01/E02 и .vscode/ не менялись;
.gitignore добавляет только /.venv-chronos/. Fit/fine-tuning, CUDA/WSL,
системные компоненты, подбор параметров, commit/push не выполнялись.
Следующий шаг для обсуждения: анализ ошибок по сохранённым прогнозам и
отдельное решение о новом протоколе проверки, без переноса старых границ.

## E02b: завершённый прямой CatBoost
ID: E02b — выполнена 2026-10-06.
Цель: проверить прямой глобальный CatBoost на неизменном протоколе E01.
Реализация: отдельные direct_model.py, direct_experiment.py, direct_evaluation.py,
direct_report.py, scripts/run_catboost_direct.py, конфигурация и два файла тестов.
Legacy и параметры E01 фиксированы; strict12 и подбор параметров не выполнялись.
Все 64 ID взяты из E01; соперники — сохранённые прогнозы, без повторного обучения.
Две оценки: A — вся общая область E01 с явным резервом; B — только native,
каждый соперник на точно таких же ключах. Failed не скрывается исключением строк.
Результаты: outputs/catboost_direct_v1/; отчёт: reports/results/E02_catboost_direct.md.
Полный pytest до обучения: 98 passed, 1 warning, 8.40 s, код 0.
Команды тестирования, минимального и полного запуска — в PROJECT_CONTEXT и отчёте.
Эксперимент завершён: 12 выпусков, 29 fit, 1897 запрошенных ключей;
1833 native, 64 fallback_no_training_pairs, 0 failed. На 1890 оцениваемых
ключах E01: 1827 native и 63 fallback. Все h=12 резервные, fit не вызван.
Годовая MAE обученного direct отсутствует, резерв не выдан за обученную модель.
h=1: все 758 ключей, max_abs=3.637978807091713e-12, допуск 1e-8, превышений 0.
49 групп метрик A и 42 группы B независимо пересчитаны из прогнозов;
91 числовая ячейка MAE отчёта сверена, 12 журналов ошибок пусты.
172 защищённых файла совпали по SHA256; E01, baseline_v1, E02a и данные не менялись.
Holdout direct хуже Recursive на h=3, лучше на h=6, но YoY лучше direct
на h=1/3/6. Это исследовательский результат, не новая независимая проверка.
Не выполнялись: установка зависимостей, настройка параметров, commit/push,
изменения .gitignore и посторонних файлов, включая .vscode/.
Следующий шаг для согласования: разбор ошибок h=3/h=6 по сохранённым прогнозам;
новый опыт не должен молча менять временную оценку или требования к истории.

## E02a: завершённый аудит обучающих пар
ID: E02a
Цель: подготовить прямые обучающие пары и проверить временную доступность.
Входы: текущий код признаков/обучения, данные, артефакты E01 и его принятая диагностика.
Разрешённые изменения: отдельный модуль, тесты, конфигурация/скрипт аудита,
outputs/e02_data_audit/, docs/PROJECT_CONTEXT.md и docs/TASKS.md.
Не входит: изменение E01, экспериментальное обучение, оценка новых прогнозов,
установка зависимостей, commit/push, посторонние изменения (включая .vscode/).
Режим A (legacy) повторяет фактические условия старого сборщика, strict12
добавляет только порог 12 фактов; давность обучающего примера задаётся отдельно.
Оба режима фиксированы, не подбираются по уже просмотренному holdout.
Критерии: тесты временной доступности и календаря, точные ключи h=1/L=0,
96 строк доступности (2 режима × 12 выпусков × 4 горизонта), построчное
покрытие случаев E01 и явное сохранение нулевых наборов.
Результаты: outputs/e02_data_audit/{training_availability.csv,
e01_case_availability.csv.gz,e01_evaluation_summary.csv,legacy_h1_comparison.csv,
config_resolved.yaml,run_manifest.json}. Команды и ограничения записаны
в docs/PROJECT_CONTEXT.md. Аудит не обучал модели и не рассчитывал MAE E02.
Полный pytest: 80 passed, 1 warning; старый синтетический тест выполняет
два fit CatBoost по 5 итераций, экспериментальное обучение не запускалось.
Следующий шаг после E02a был отдельно согласован и выполнен как E02b;
сохранённые аудит и правила сборщика E02a не изменены.

## Шаблон записи эксперимента
ID / исследовательский вопрос / Git commit / незакоммиченные изменения /
конфигурация и версия данных / дата запуска / команда / среда и seed /
выборка, даты и горизонты / путь к результатам / метрики и покрытие /
вывод / ограничения / статус проверки.

## Правило принятия результата
Сначала проверяются корректность и сопоставимость; только затем улучшение метрик.
Не менять правила теста ради выигрыша. Неудачные эксперименты сохранять.
По окончании задачи обновлять статус и указывать реальный путь к подтверждающему результату.
