# Данные и воспроизводимость

F7, проверено 2026-10-07. Этот каталог описывает фактические входы и решения о
публикации. Он не содержит полный исходный рабочий комплект.

## Что входит в public-пакет

- [manifest.csv](manifest.csv): источник, условия, решение, размер и SHA256.
- [inventory_files.csv](inventory_files.csv): пофайловая metadata inventory,
  сохранённые outputs и исторические отсутствующие источники. Значения строк
  не экспортированы. Технические fixtures и веса учтены как группы.
- [csv_headers.json](schemas/csv_headers.json): 200 уникальных заголовков CSV;
  lookup по `schema_id` после `#` в inventory. Это header contracts, а не
  подтверждение типов, nullability или уникальности ключей всех старых таблиц.
- [sberindex_sources.json](schemas/sberindex_sources.json): известные схемы и
  исторические row counts оригиналов; неизвестные имена полей не выдуманы.
- [news_snapshots.csv](metadata/news_snapshots.csv): 197 snapshots / 93 документов /
  89 canonical groups — URL, даты, IDs, hashes, статусы и собственные коды тем.
  Titles, snippets, тела статей и географические имена исключены.
- [macro_provenance.csv](metadata/macro_provenance.csv): происхождение 227 записей
  Банка России, без числовых значений. Статусы A=73/B=154 и vintages сохранены.
- [source_terms.json](metadata/source_terms.json) и
  [sberindex_terms.json](metadata/sberindex_terms.json): первичные URL, короткие
  цитаты, дата проверки и ограничения применимости условий.
- [SYNTHETIC demo](synthetic/e07c_demo/README.md): пять побайтовых копий старого
  smoke dataset, 76 рядов × 24 месяца; реальные данные не используются.

`PUBLISH` в manifest означает **предлагается добавить в проверяемый пакет F7**.
Commit/push не выполнены. Для исходного набора `PUBLISH_METADATA_ONLY` означает,
что опубликованы только его описание/схема/hash, а не значения.
Manifest и preflight каталог не хешируют сами себя; все PUBLISH data artifacts
имеют отдельные byte-pinned contracts. `.gitattributes` сохраняет их физические
байты при checkout, включая BOM/переводы строк исходного demo.

## Проверка без обучения

Используйте Python 3.12, из корня проекта:

```powershell
py -3.12 scripts/check_data.py
py -3.12 scripts/check_data.py --profile research
py -3.12 scripts/check_data.py --profile sources
```

- `public` (по умолчанию): manifest, все добавляемые data artifacts, SHA/размер,
  JSON structure, точные CSV headers, counts и необходимые ключи demo/metadata.
  Использует только стандартную библиотеку, ничего не скачивает и не меняет.
- `research`: основные сохранённые входы исторического pipeline. В обычном clone
  они **MISSING**; это не ошибка public-пакета и не полный аудит всех зависимостей
  прежних experiments. Полный состав групп и файлов дан в inventory.
- `sources`: точные имена и исторические SHA пяти исходных файлов. Они **MISSING**
  в clone. Parquet schema проверяется только при уже доступном `pyarrow`; скрипт
  его не устанавливает. Для XLSX/GPKG структурная проверка не реализована:
  `NOT CHECKED` не считается PASS.

Каждая строка проверки показывает FOUND/MISSING, ожидаемый относительный filename,
schema, hash и size status. Exit 0 означает PASS выбранного профиля; exit 1 —
неполный или ошибочный профиль. `--json` выдаёт те же статусы в JSON, без содержимого
исходных строк и машинных абсолютных путей. Команда работает и из другого CWD,
если указан путь к самому скрипту.

## Что получить отдельно и куда положить

Автоматического скачивания **нет**. Официальный допустимый direct download
исходного комплекта не подтверждён; новый скачанный snapshot не заменяет
историческую версию. Получайте файлы из официального источника или от проверенного
правообладателя на его условиях. Исторические SHA ниже взяты из старого input
manifest и не были заново измерены по отсутствующим оригиналам.

| Ожидаемый файл | Локальное размещение | Исторические строки / объекты | Байты |
| --- | --- | ---: | ---: |
| consumption.parquet | `data/raw/sberindex/consumption.parquet` | 303126 | 2373939 |
| market_access.parquet | `data/raw/sberindex/market_access.parquet` | 2571 | 25163 |
| connection.parquet | `data/raw/sberindex/connection.parquet` | 5942364 | 21904682 |
| t_dict_municipal_districts.xlsx | `data/raw/municipal/t_dict_municipal_districts.xlsx` | 3101 | 322024 |
| t_dict_municipal_districts_poly.gpkg | `data/raw/municipal/t_dict_municipal_districts_poly.gpkg` | 2660 | 74334208 |

Официальные страницы: [три таблицы СберИндекса](https://sberindex.ru/ru/research/data-sense-opisanie-nabora-dannikh-khakatona-sberindeksa-po-munitsipalnim-dannim)
и [муниципальный справочник](https://sberindex.ru/ru/news/dataset-borders-and-changes-of-municipalities).
Подготовленный исторический вход располагается в
`data/input/consumption_all_categories.csv`: 50521 строка, 10236634 байта,
SHA256 `9802b7b2448b42afbb4152c16a12346ebcb5d074ff6eedda8783669abf35a63c`.
Loader требует `date`, `territory_id`, `value`; `category` при наличии отбирается
как «Все категории». Ключ МО × месяц уникален, значения конечны и неотрицательны.
Оригинал содержит 11 полей вместе со справочником. Произвольный новый CSV может
быть пригоден loader, но не проходит проверку исторической byte provenance.
Скрипт не создаёт, не заменяет и не пересохраняет этот вход.

Другие основные зависимости — сохранённые E01 sample IDs/predictions,
online residuals/selected parameters, macro table/source IDs и news v3
documents/geography dictionary. Их точные paths/схемы/SHA — в
[preflight.json](preflight.json); полная metadata — в inventory.
Весь исходный news corpus, macro raw и прежние outputs остаются отдельными
локальными артефактами. Получение современных публикаций по тем же URL не
восстанавливает их прежний hash и historical availability.

## Условия и ограничения

Источник показателей — **СберИндекс**. Сохранённый документ отдельно указывает
CC BY-SA 4.0 для трёх Parquet, но независимая официальная связь этого grant
с комплектом не подтверждена: статус F7 **UNCLEAR**, значения не добавлены.
При подтверждённом распространении нужны название соответствующего dataset,
СберИндекс, официальный URL, реальная дата скачивания, CC notice/link,
обозначение изменений и ShareAlike; неизвестную дату скачивания нельзя заменить
датой F7. Лицензия и атрибуция справочника проверяются отдельно.
Подробности и условия остальных источников — в [DATA_NOTICE.md](../DATA_NOTICE.md).

Реальные prepared CSV, predictions/residuals/history features и weak labels
не добавлены: они могут сохранять target или наследовать непроверенные права.
Отдельно отмечен уже tracked реальный rolling example; F7 не изменяет Source of
Truth и не подтверждает права на этот прежний export автоматически.
Тексты Lenta, исходные HTML/PDF/XLSX и числовые macro snapshots не добавлены.
Raw/private/restricted paths игнорируются Git; новые пустые папки не создаются.

У полноценных synthetic datasets нет внешних raw inputs. Уже tracked
[генератор online benchmark](../src/sberforecast/online_synthetic.py),
[генератор early warning](../src/sberforecast/early_warning_synthetic.py),
[online config](../configs/online_detection.yaml) и
[early-warning config](../configs/early_warning_synthetic.yaml) сохраняют seeds
и правила. Offline benchmark использует те же сохранённые synthetic данные.
Полные generated rows не дублируются в Git; demo достаточен для проверки data
format, но не заменяет полный benchmark и не подтверждает реальные метрики.

Результат F7 — воспроизводимый public data/documentation layer с явными missing
inputs. **Полное воспроизведение реальных исторических experiments не проверено
и из одного clone пока невозможно.** Проверки F7 и remaining limitations — в
[audit](../reports/final/DATA_PUBLICATION_AUDIT.md).
