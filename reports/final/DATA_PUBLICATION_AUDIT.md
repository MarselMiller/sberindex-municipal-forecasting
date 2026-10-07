# F7 — data publication and reproducibility audit

Дата: **2026-10-07**. Base: `main`,
`e50e0440f74fe8484d33d9662d9ecc09aa4918a5`; GitHub main совпал с этим commit.
Рабочая папка до F7 была чистой. Сначала была открыта другая локальная ветка
на том же commit; перед правками выполнен `git switch main`.
F7 меняет data publication artifacts, read-only preflight и документацию.
Research phase закрыта; результаты, протоколы и Source of Truth не пересчитываются.

## Решение

**Verdict: PASS WITH LIMITATION для предлагаемого public data/documentation layer.**
Полное воспроизведение реальных
исторических experiments из одного clone невозможно. Необходимые private
inputs/outputs отсутствуют в Git, а права на весь реальный комплект не подтверждены.
Это audit доступности и условий публикации, не новый ML-эксперимент.

**Можно ли публиковать исходные SberIndex Parquet?** F7 не подтверждает
безусловное разрешение. Сохранённый PDF отдельно указывает CC BY-SA 4.0 для
`consumption.parquet`, `market_access.parquet` и `connection.parquet`;
SHA PDF совпадает с прежним input manifest. Однако независимая официальная
связь grant с предоставленным комплектом не установлена. Поэтому статус всех
трёх — **UNCLEAR**, решение — **PUBLISH_METADATA_ONLY**. Это не вывод о запрете
и не утверждение, что лицензии нет. После подтверждения официального grant
распространение возможно с выполнением условий CC; справочник требует отдельного
разрешения. Официальная
[страница набора](https://sberindex.ru/ru/research/data-sense-opisanie-nabora-dannikh-khakatona-sberindeksa-po-munitsipalnim-dannim),
[grant evidence](../../data/metadata/sberindex_terms.json) и
[CC legal code](https://creativecommons.org/licenses/by-sa/4.0/legalcode.ru).

## Inventory и границы доказательств

Просмотрены README, DATA_NOTICE, configs, scripts, src, reports/final,
reports/results и tests; фактические локальные data/outputs сопоставлены с refs.
[Manifest](../../data/manifest.csv) содержит **90 решений**:
12 PUBLISH, 18 PUBLISH_METADATA_ONLY, 59 DO_NOT_PUBLISH, 1 NEEDS_CONFIRMATION.
[Пофайловый inventory](../../data/inventory_files.csv) содержит **1877 записей**:
1815 файлов с измеренными размерами/SHA, 51 output group и 11 исторических
отсутствующих источников. Technical fixtures и model cache учтены как группы;
их payloads не публикуются и не хешировались пофайлово. 1052 CSV сопоставлены с
[200 уникальными заголовками](../../data/schemas/csv_headers.json), без значений строк.

Для отсутствующих архивов/Parquet/справочника размеры и hashes — исторические
из `reports/input_manifest.json`, counts — из `reports/audit_report.md`.
Это не новые проверки отсутствующих оригиналов. Неизвестные SHA/размеры старых
`territory_coverage.csv` и `audit_summary.json` оставлены пустыми.
Имена/типы неизвестных полей не выдуманы. Header inventory не доказывает dtypes,
nullability или корректность всех ключей; preflight проверяет более строгие
контракты только для выбранного public/research data layer.

Размер группы в manifest — сумма размеров её файлов. Group SHA, если указан,
вычислен по отсортированному compact UTF-8 JSON списка `{path, sha256}`
инвентаризированных членов (`sort_keys=True`, separators `,`/`:`).
Он не является checksum единого файла. Физические file SHA и logical CSV row
counts отделены: одинаковые распакованные CSV могут иметь разные gzip SHA.
Восемь сохранённых E04/E06a synthetic CSV проверены на byte equality после
распаковки; все совпадают. Уровень lineage в inventory — консервативный
file/group audit, не исчерпывающий AST dependency graph.

| Dataset | Used in project | Source | License/terms | Redistribution | Published? | Method | Reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| consumption.parquet | Источник target, 303126 строк / 6 категорий | СберИндекс | Bundled CC BY-SA 4.0; official binding pending | UNCLEAR | Metadata only | Имена, схема, count, historical SHA | Raw отсутствует; grant не подтверждён независимо |
| market_access.parquet | Первоначальный аудит, 2571 строка; не forecasting feature | СберИндекс | То же, отдельный grant в PDF | UNCLEAR | Metadata only | Схема/SHA/размер | Не требуется для текущего численного forecast input |
| connection.parquet | Аудит metadata, 5942364 строк; не model feature | СберИндекс | То же, отдельный grant в PDF | UNCLEAR | Metadata only | Схема/SHA/размер | Полные значения не анализировались |
| XLSX/GPKG муниципалитетов | Year-aware enrichment; геометрия только audit | СберИндекс, отдельный набор | Явное разрешение не найдено | UNCLEAR | Metadata only | 3101 записей / 2660 объектов, SHA/placement | Лицензия трёх Parquet не покрывает справочник |
| Prepared target CSV | Основной вход; 50521 строка × 11 полей | Target + справочник | Наследует непроверенные права | DERIVED_RESTRICTED | No | Контракт и SHA; private input сохранён | Есть исходный target и поля справочника |
| Forecast inputs/outputs и residuals | Baselines/Prophet/boosting/Chronos/detection | Project + target | Source conditions inherited | DERIVED_RESTRICTED | No new rows | Per-file metadata, schemas, hashes | `y_true`, error и лаги могут восстановить target |
| Weak labels/history features | Реальная feasibility, не обучение classifier | Project + target/macro/news | Source conditions inherited | DERIVED_RESTRICTED | No | Metadata/контракты, без строк | Производный файл не становится свободным автоматически |
| CBR macro numeric/raw | Четыре прогнозные серии, 227 записей | Банк России | Перепечатка с source link; commercial restriction | UNCLEAR для unrestricted package | Metadata only | Provenance, dates/vintages/hashes; values исключены | Не заявлено безусловное право модификации/перелицензирования |
| CBR/Lenta news raw | 93 URL-documents, 197 snapshots, 89 canonical groups | Банк России / Lenta.ru | CBR notice; Lenta permission не установлено | UNCLEAR для text corpus | Metadata only | 24 поля whitelist, без title/snippet/body/geo names | Attribution сама по себе не разрешает corpus publication |
| МЧС / Pravo / Росстат | Probe/failed attempts, 0 принятых событий/observations | Официальные порталы | Conflicting scope / не установлено для этих attempted tables | UNCLEAR | Attempt metadata only | Terms URLs и recorded failures | Не являются реально использованными числовыми источниками |
| Synthetic detection / warning | Quantitative benchmarks; без реальных raw inputs | Project generators | OWN_GENERATED provenance | OWN_GENERATED | Demo + generator/config; full rows metadata only | Пять byte copies старого smoke | Нет новых генераций, моделей или оценки |
| News/macro metadata | F7 reproducibility/provenance | Факты источников + собственные IDs/codes | Original rights не перенесены на source-free состав | DERIVED_SAFE | Proposed PUBLISH | Whitelist projection | Без текстов, numerical macro values и справочника |
| Manifests/schemas/configs/test fixtures | Зафиксированные протоколы и synthetic checks | Project | Own metadata; не новый blanket license | OWN_GENERATED / DERIVED_SAFE | Tracked или proposed metadata | Inventory/header contracts | Original manifests с machine paths не копируются |
| Curated real rolling export | Уже tracked F1 example: 12 real `y_true` | Project / СберИндекс | Original source rights pending | DERIVED_RESTRICTED | Already tracked; NEEDS_CONFIRMATION | Только отметка о риске; F1 bytes сохранены | Это не source-free aggregate; F7 не подтверждает права автоматически |
| Chronos weights/cache | Model dependency, не external data series | Amazon model card | Current Apache-2.0; pinned revision scope отдельно | Не подтверждалось для перепубликации pinned weights | No | Revision в конфигурации, group size metadata | Большой binary; F7 не скачивал веса |

CBR series сохранены раздельно: `forecast_inflation_dec_dec_pct`,
`forecast_inflation_annual_average_pct`, `forecast_nominal_wage_growth_annual_pct`
от участников опроса и `forecast_consumption_growth_annual_pct` Банка России.
Это прогнозы, а не фактические месячные ИПЦ/зарплаты Росстата. Статусы A=73/B=154
сохранены; range midpoint остаётся проектным преобразованием, не объявляется
официальным центральным прогнозом.

## License/terms provenance

Полные primary URLs, короткие точные цитаты и дата **2026-10-07** сохранены в
[source_terms.json](../../data/metadata/source_terms.json) и SberIndex terms JSON.
[DATA_NOTICE](../../DATA_NOTICE.md) описывает attribution и условия.
Число цитируемых слов каждой web-страницы ограничено; тексты новостей не цитируются.

- SberIndex: локальный PDF 538791 байт / 4 страницы, hash-bound к старому manifest.
  Он явно именует три Parquet. Live HTML — application shell; official public API
  list доступен, но matching grant/official original PDF/direct download не получен.
  Это ограничение проверки, не доказательство отсутствия лицензии.
- [Банк России](https://www.cbr.ru/about/): checked current reproduction notice
  имеет attribution и commercial restriction; unlimited open grant не установлен.
  Current terms не являются историческим license snapshot 2023–2024.
- Lenta: official HTML указывает
  [terms](https://lenta.ru/info/posts/terms_of_use/) / [legal](https://lenta.ru/legal/);
  доступный первичный grant не получен. Сторонние пересказы не использованы как разрешение.
- [МЧС agreement](https://mchs.gov.ru/chek-listy/informaciya-o-sayte/ob-ispolzovanii-informacii-sayta-i-obrabotke-personalnyh-dannyh)
  содержит CC footer и ограничительные пункты; scope не разрешён в пользу blanket grant.
- [Росстат open data](https://rosstat.gov.ru/opendata/) имеет reuse conditions,
  но attempted wage/CPI tables не получены и применимость к ним не установлена.
  Pravo не дал accepted source files. Неудавшиеся источники не названы used numerical data.

## PUBLISHED / METADATA ONLY / NOT PUBLISHED / UNCLEAR LICENSE

**PUBLISHED (proposed F7; commit/push отсутствуют):** 12 data artifacts,
**2392393 байта** суммарно. Пять SYNTHETIC файлов — **73584 байта**:
observations 43658; series metadata 20794; events 4457; cohort IDs 3126;
generator config 1549. Metadata news 152172, macro 158724; inventory 1470344;
CSV header bundle 506700; Sber schema 1644; external terms 19426; Sber terms 9799.
Список индивидуальных SHA — в manifest/preflight. README demo явно маркирует
76 smoke series / 1824 observations / 46 events и seeds 421101/421201/421301.

**METADATA ONLY:** оригинальные Parquet, XLSX/GPKG, архивы/лицензионный PDF,
исторические missing preparation/audit files, full synthetic inputs,
news/macro sources. Все field projections сохранены без обновления snapshots.

**NOT PUBLISHED:** enriched real target, real row predictions/residuals/features/
weak labels, source article corpus, numeric macro/raw documents, technical
fixtures/caches, weights и исходные архивы. Эти файлы не добавлены и не staging.

**UNCLEAR LICENSE:** официальная provenance grant трёх Sber Parquet; отдельный
dictionary XLSX/GPKG; Lenta text; scope CBR commercial/adaptation use; МЧС
contradictory notices. Неполученные Росстат/Pravo не создают published source rows.
У уже public curated real rolling export остаётся inherited rights limitation.
F7 не удаляет и не меняет existing Source of Truth; текущая открытая копия
не получает blanket legal clearance этим audit.

### Уже tracked производные отчётные материалы

Raw SberIndex data не публикуются. В репозитории сохранены отдельные
**DERIVED VISUALIZATION / DERIVED REPORT OUTPUT**; классификация описывает
форму материала, а не подтверждение прав на его распространение. Их наличие
не подтверждает redistribution rights для raw source data. F7 не добавляет
новых реальных значений и не удаляет уже tracked отчётные материалы.

| Уже tracked artifact | Классификация | Содержимое и решение review |
| --- | --- | --- |
| [rolling_forecast.png](figures/rolling_forecast.png) | DERIVED VISUALIZATION | Реальные target и прогноз одного МО; сохранить без изменений, source rights не подтверждены |
| [rolling_forecast.csv](figure_data/rolling_forecast.csv) | DERIVED REPORT OUTPUT | 12 реальных `y_true` и прогнозы; сохранить, manifest: DERIVED_RESTRICTED / NEEDS_CONFIRMATION |
| [forecast_mae.png](figures/forecast_mae.png), [forecast_mae.csv](figure_data/forecast_mae.csv), [forecasting_metrics.csv](forecasting_metrics.csv) | DERIVED VISUALIZATION / DERIVED REPORT OUTPUT | Агрегированные ошибки на реальных данных; сохранить без изменений |
| [real_diagnostic.png](figures/real_diagnostic.png) | DERIVED VISUALIZATION | Реальные residuals и ретроспективные кандидаты change points; диагностика, не подтверждённые события |
| [real_warning_sufficiency.png](figures/real_warning_sufficiency.png), [real_warning_sufficiency.csv](figure_data/real_warning_sufficiency.csv) | DERIVED VISUALIZATION / DERIVED REPORT OUTPUT | Агрегированные counts реальных weak labels, не качество обученного classifier |
| [detection_metrics.csv](detection_metrics.csv), [early_warning_metrics.csv](early_warning_metrics.csv), [results_summary.json](results_summary.json) | DERIVED REPORT OUTPUT | Real diagnostics/feasibility и отдельно маркированные synthetic metrics; сохранить различие статусов |
| [news_timeline.png](figures/news_timeline.png), [news_timeline.csv](figure_data/news_timeline.csv) | DERIVED VISUALIZATION / DERIVED REPORT OUTPUT | Counts канонических новостных событий, не значения target СберИндекса |

Статусы сверены с [figure manifest](figure_manifest.csv). В частности,
[offline_example.png](figures/offline_example.png) — **SYNTHETIC**, как и
online comparison и synthetic warning examples; их нельзя относить к реальным
значениям СберИндекса. Все перечисленные прежние файлы остаются побайтово прежними.

## Размеры и Git

Все предлагаемые новые файлы меньше 50 MiB; самый большой data artifact —
inventory, 1470344 байта. Historical connection — 21904682 байта, consumption —
2373939, market access — 25163; все меньше 50 MiB, но отсутствуют и не добавлены.
GPKG — 74334208 байт, больше 50 MiB; права unclear. Cached weight blob —
477930472 байта, больше 100 MiB, не кандидат normal Git.
[GitHub](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)
предупреждает при файлах больше 50 MiB и блокирует больше 100 MiB.
Для больших разрешённых файлов возможны official download, Release asset или
отдельно согласованный LFS; F7 выбирает metadata/hash. Direct download не
подтверждён, Releases не создавались, LFS не включён.

Предлагаются 19 новых файлов: `.gitattributes`, `data/README.md`, manifest,
inventory, preflight JSON, четыре metadata files, два schema files, demo README
и пять его data files, `scripts/check_data.py`, этот audit. Изменены только
README/DATA_NOTICE, `.gitignore` и две документационные карточки docs.
`.gitattributes` точечно отключает EOL conversion для hash-pinned data artifacts.
Raw/private/restricted paths добавлены в ignore без смены experiment paths.
Ни одного `git add`, commit или push не выполнялось.

## Проверки и воспроизводимость

Workspace preflight: public **12/12 PASS**, research **9/9 PASS**;
sources — **5 MISSING**, exit 1. Research проверяет основные уже сохранённые
входы, не означает новое воспроизведение моделей. Sources отсутствуют;
XLSX/GPKG structure validator не реализован, Parquet требует доступный pyarrow.
Public profile использует только stdlib, показывает filename/schema/hash/size,
не выводит содержимое private data и ничего не меняет/скачивает.
Девять focused stdlib checks для path traversal, missing, SHA/schema/size,
duplicate keys/nonfinite, privacy и manifest 1:1/unsafe publication прошли;
full pytest не выполнялся.
Отдельный review исправил проверку 1:1 manifest PUBLISH ↔ public contract.

Clean-clone data/documentation check: **PASS WITH LIMITATION**. Новый GitHub
clone создан в `tmp/f7_clean_clone_e50e044`, HEAD совпадает с base; все **236**
base tracked files доступны, clone был чистым до overlay. Наложены только
24 reviewed F7 files: 19 новых / 5 изменённых. F7 не committed/pushed, поэтому
это проверка proposed publication tree, не обновлённого remote commit.
Python **3.12.10**, `-I -B`, только stdlib; пакеты не устанавливались,
private data/outputs/weights/venv не копировались. Public preflight **12/12 PASS**,
exit 0, также из parent CWD. Research **9 MISSING**, sources **5 MISSING**,
оба exit 1; это ожидаемые отсутствующие зависимости, не скрытый полный PASS.
Схемы demo и 1052 inventory header references согласованы, mismatches 0.

Команды действительно выполнены; workspace preflight использовал существующий
локальный Python 3.12, clone — системный launcher:

```powershell
git clone --depth 1 --branch main https://github.com/MarselMiller/sberindex_python_mvp.git tmp/f7_clean_clone_e50e044
# После явно перечисленного metadata-only overlay, из clone:
py -3.12 -I -B -X utf8 scripts/check_data.py --profile public --json
py -3.12 -I -B -X utf8 scripts/check_data.py --profile research --json
py -3.12 -I -B -X utf8 scripts/check_data.py --profile sources --json
# Из tmp, проверка независимости от CWD:
py -3.12 -I -B -X utf8 f7_clean_clone_e50e044/scripts/check_data.py --json
git diff --check
```

Первоначальная F7-проверка README и всех семи новых/изменённых Markdown материалов: **86 relative links
PASS**, missing 0. Широкий scan 31 documentation file обнаружил **18 прежних
local-only links** в `reports/results/E04a_online_detection.md`,
`E06a_offline_detection.md`, `E07c_synthetic_early_warning.md` на игнорируемые
`outputs`: 15 PNG и 3 JSON. Они существовали до F7, не являются missing tracked
files и не входят в новый data layer. Полный historical report trail требует
private/saved outputs; глобальный критерий «все ссылки всего repo доступны
из clone» **не выполнен**. Архивные отчёты и Source of Truth не переписаны;
добавление реальных diagnostic PNG до проверки source rights не сделано.

### Documented historical exception: 18 local-only links

Все цели ниже находятся в ignored `outputs`, не являются tracked файлами
и не добавляются в предлагаемый diff. README ссылается на существующий каталог
`reports/results/`, но непосредственно на эти 18 целей не ссылается.
Current README/data docs и public preflight не зависят от этих локальных ссылок.
Архивные отчёты сохранены без косметического переписывания.

| Historical report | Local-only targets относительно корня проекта | Число |
| --- | --- | ---: |
| [E04a_online_detection.md](../results/E04a_online_detection.md) | `outputs/online_detection_v1/figures/municipality_21.png`, `municipality_25.png`, `municipality_37.png`, `municipality_1471.png` в том же каталоге | 4 |
| [E06a_offline_detection.md](../results/E06a_offline_detection.md) | `outputs/offline_detection_v1/figures/offline_metrics_ci.png`, `municipality_21.png`, `prefix_stability_21.png`, `municipality_25.png`, `prefix_stability_25.png`, `municipality_37.png`, `prefix_stability_37.png`, `synthetic_level_example.png` в том же каталоге; `outputs/e06a_checks/pelt_pruning_diagnostic.json`, `pelt_optimality_synthetic.json`, `pelt_optimality_full.json` в том же каталоге | 11 |
| [E07c_synthetic_early_warning.md](../results/E07c_synthetic_early_warning.md) | `outputs/early_warning_synthetic_v1/plots/first_successful_warning.png`, `first_false_alert.png`, `first_missed_event.png` в том же каталоге | 3 |

**Восемь целей также нужны повторной сборке F1 из сохранённых outputs**:
`pelt_optimality_full.json`, четыре offline PNG (`synthetic_level_example.png`,
`offline_metrics_ci.png`, `municipality_21.png`, `prefix_stability_21.png`)
и три synthetic warning PNG. Это реальные зависимости архивного builder:
[final_detection.py](../../src/sberforecast/final_detection.py) читает JSON
и проверяет hashes PNG, [final_figures.py](../../src/sberforecast/final_figures.py)
копирует выбранные PNG, [final_evidence.py](../../src/sberforecast/final_evidence.py)
фиксирует provenance. Они не нужны для чтения готовых итогов, model inference
или [public data preflight](../../scripts/check_data.py). Полная повторная сборка
F1 требует saved/private outputs; audit не объявляет её доступной из public clone
и не заменяет её входы отчётными PNG. Текущих broken README/data-doc
Markdown links на эти цели не обнаружено; пути архивного builder не менялись.

Scan 24 кандидатов: secrets/tokens/private email/absolute private paths — 0;
news/macro columns строго соответствуют whitelist, article text/numeric macro
values отсутствуют. Byte copies demo совпали с оригинальным smoke. Git filters
для 12 hash-pinned artifacts сохраняют байты благодаря `.gitattributes`;
CRLF учитывается как line ending, остальные whitespace checks сохранены.
Шесть ignore probes закрывают raw/private/restricted/input/news/outputs paths;
staged files 0. `git diff --check` PASS в workspace и clone; отдельно
`git diff --no-index --check` проверил 18 новых текстовых файлов, ошибок 0.
Проверено отсутствие изменений в **228** защищённых старых files и совпадение
**1815** inventory file hashes; shared protected SHA:
`be3f70b30b9ccf5f39d75836b93742566385915387ba5ca77ac32355f289dbf0`.
Старые **23 absolute provenance lines в 9 files** из F6 оставлены побайтово;
в новых artifacts абсолютных runtime paths нет. No model fits, генераций,
F1 builder, пересчёта метрик, PDF, full pytest, commit/push.

### Final review перед commit

**PASS для предложенного F7 diff с документированными историческими исключениями.**
Состав: **19 новых / 5 изменённых файлов**, staged 0. Raw Parquet, исходные
справочники, real target, real predictions/residual tables и private/restricted
payloads в diff отсутствуют. Исходные SberIndex и справочник остаются
**UNCLEAR / PUBLISH_METADATA_ONLY**; случаев **UNCLEAR + PUBLISH — 0**.

Повторный public preflight: **12/12 PASS**, manifest/schema/hash/size PASS,
exit 0. Проверено **110 relative links** в семи Markdown-файлах F7 против
предлагаемого tracked/public tree, missing 0. Более широкий scan 30 Markdown
файлов проверил 251 relative links и сохранил ровно 18 перечисленных выше
local-only исключений. Это не общий all-links PASS и не подтверждение полной
повторной F1-сборки из public clone.

Security/path scan всех 24 кандидатов, включая распакованный SYNTHETIC CSV:
секретов, API keys/tokens/cookies/authorization values, email, private URL
и абсолютных локальных filesystem paths не обнаружено. Отдельно проверены
inventory, manifest и четыре metadata files: 2391 CSV-строка и оба terms JSON.
Публичные API routes в metadata — пути запросов, не filesystem paths.
Historical source identifiers — логические ключи, а реальные file paths
portable/relative. Новые metadata не содержат news bodies или macro values.

Все **пять SYNTHETIC demo files** совпадают по SHA/size с сохранённым smoke
и manifest, без новой генерации. Маркировка есть в demo README и каждой из пяти
manifest-записей. Все 76 сценариев имеют origin
`controlled_synthetic_generator_not_real_economic_data`; generator использует
числовую конфигурацию и независимый RNG, без чтения SberIndex/реального target.
Demo не содержит реальные значения источника и не восстанавливает real target.

Все **228** защищённых файлов, включая прежние derived figures/tables,
сохранили SHA. Research results и Source of Truth не менялись.
`git diff --check` и whitespace check 18 новых текстовых файлов прошли;
`git status --short` подтверждает только 24 согласованных F7 кандидата.
В этом review выполнены только локальные проверки и уточнение документации;
новых источников, скачиваний, экспериментов, model fits, builder, pytest,
commit/push не было. **F7 ready for commit** в указанном scope.

Следующий конкретный шаг после review: получить независимый official grant
для указанного Sber комплекта и отдельные условия dictionary; проверить права
уже tracked real example. Только затем расширять real data publication и
планировать полное повторение исторических experiments отдельной задачей.

NO NEW MODEL FITS.
NO NEW EXPERIMENTS.
NO COMMIT/PUSH.
