# Publication rights review — 2026-10-09

## Решение по rolling_forecast.csv

**PUBLICATION PERMITTED при соблюдении CC BY-SA 4.0.** Это решение относится
к существующему 12-строчному примеру, а не к публикации private research archive.
Публикация и изменение видимости здесь не выполнялись.

Файл содержит `municipality_id`, `forecast_origin`, `target_period`, `y_true`,
`y_pred`, `split`: МО 21, категория «Все категории», месяцы цели январь–декабрь
2024; шесть validation и шесть holdout. `y_true` — неизменённые исходные наблюдения,
`y_pred` — сохранённый SeasonalNaiveYoY, h=1. Следовательно, CSV одновременно
перераспространяет наблюдения и содержит производные результаты. Названий МО,
регионов и полей справочника в нём нет.

Выборка создавалась в [final_figures.py](../../src/sberforecast/final_figures.py):
сохранённые прогнозы `outputs/prophet_comparison_v1/predictions.csv.gz`, конечный
факт, SeasonalNaiveYoY, h=1, минимальный числовой ID, сортировка по месяцу цели.
Новая проверка всех 12 строк против этого prediction artifact и значений
`data/input/consumption_all_categories.csv` прошла; прогнозы не пересчитывались.
Исторический manifest связывает подготовленный target с `consumption.parquet`.
Исходный Parquet при этой проверке не найден внутри проекта: связь с ним основана
на сохранённых input manifest и отчёте подготовки, а не новой сверке Parquet.

## Первичные лицензионные основания

Локальный `reports/Данные_СберИндекс_лицензия.pdf` прочитан: четыре страницы,
538791 байт, SHA256
`030269e726f969250b911a1fcb72131a4b61c3952aad10c8470ce9c0736c472f`.
На странице 2 отдельно указан CC BY-SA 4.0 для муниципальных потребительских
расходов; на странице 1 указан `consumption.parquet` и период 2023–2024.
SHA совпадает с историческим input manifest.

2026-10-09 получен HTTP 200 от
[официального API описания набора](https://sberindex.ru/api/researches/v1/data-sense-opisanie-nabora-dannikh-khakatona-sberindeksa-po-munitsipalnim-dannim).
Запись `33fd3aed-4f44-4214-ad8c-182b34a68047` содержит в `template.main[].inner`
отдельное указание CC BY-SA 4.0 для муниципальных расходов за январь 2023 —
декабрь 2024, официальную атрибуцию и ссылку на архив Сбербанка.
Это независимый первичный источник grant; README других участников не использовались.
[Публичный адрес описания](https://sberindex.ru/ru/research/data-sense-opisanie-nabora-dannikh-khakatona-sberindeksa-po-munitsipalnim-dannim)
отдельно получить не удалось; соответствующее содержимое официального API получено.
Архив с данными не скачивался. Побайтовое совпадение локального PDF с официальным
asset не проверялось и не заявляется.

По [юридическому тексту CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/legalcode.ru)
распространение и адаптация допустимы при сохранении атрибуции, ссылки на лицензию,
описания изменений и условий ShareAlike для адаптированного материала.
Дата исходного скачивания неизвестна; дата этой проверки её не заменяет.
[Metadata доказательств](../../data/metadata/publication_rights_review.json)
содержит SHA ответа, файлов и точные границы решения, без исходных наблюдений.

[README примера](figure_data/README.md) фиксирует согласованную атрибуцию
и CC BY-SA 4.0 для CSV и иллюстрации, включая соответствующий вклад автора.
Собственному коду лицензия не назначена; MIT не добавлена. Это не лицензия муниципального
справочника, обогащённого target, news corpus или других private inputs.
Исторические решения F7 и audit reports сохранены без изменения.

## Git history publication verdict

**PASS — existing history accepted by owner, subject to no other blockers.** Подтверждённые права
устраняют правовую неопределённость именно для CSV в текущем дереве и его
исторического blob `75b4a28dd28f75b10ef8e70e258308cceaaca533`.
2026-10-09 автор явно принял раскрытие historical local paths и Git
author/committer identities при отсутствии иных блокирующих данных.
Эвристика не обнаружила реальных секретов;
она не доказывает отсутствие всех secrets/PII. Необъявленные remote refs,
dangling objects и GitHub releases/attachments не проверены.

Auditor допускает только точное проверенное содержание CSV под исходным именем
при наличии нового metadata review и attribution notice, оставляя severity REVIEW.
CRLF исходного Windows CSV и LF Git blob сравниваются без изменения файла.
Другой путь/содержание сохраняет BLOCKER; secret/path checks не отключаются.
Эвристический `PASS_WITH_REVIEW` сохраняется как результат сканера;
решение владельца закрывает известные замечания о путях/identities и атрибуции.
Это не разрешает новые непроверенные данные и не выполняет публикацию.

Принятое решение: публикуется существующий `sberindex-municipal-forecasting`,
Git history сохраняется. Второй репозиторий и новый публичный snapshot
не создаются, история не переписывается. Private archive остаётся локальным.

## Предыдущая техническая проверка кандидата до решения автора

Безопасный `reports/results/e08b/feature_coverage.csv` добавлен в index:
303 байта, 10 строк агрегированных счётчиков, без наблюдаемых значений,
municipality rows, secrets или local paths; SHA совпал с сохранённым E08b output.
Файл необходим явному publication allowlist. Данные не пересчитывались.

**CLEAN CLONE REVIEW-CANDIDATE PASS:** новый GitHub clone, reviewed candidate
в его Git index без commit, экспорт Git tree с отключённой CRLF-конверсией,
проверка каждого файла против blob, новое Python 3.12 окружение, 361 tracked input,
никаких private inputs. Выполнены `pip check`, summary editorial check,
HTML builder check, publication build/check, 57 no-fit tests и strict audit.
Пакет: 74 файла, 383 локальные ссылки/anchors, воспроизводимый ZIP.
Browser QA этого Git-object пакета под `/sberindex-municipal-forecasting/`:
89 проверок PASS (desktop/mobile, controls, PDF, приложения, glossary, no-JS).
Текущий GitHub-only main clone остаётся FAIL из-за отсутствующего coverage-файла;
исправления по инструкции не commit/push. После разрешённого push нужно повторить
GitHub-only проверку. Validation tree не является опубликованным commit.

PDF: штатный `export_pdf.ps1`, 12 страниц 960 × 540 pt. Только подпись
`pilot holdout` заменена на `holdout общей выборки`. Все числовые tokens и
36 MAE cells сохранены, остальные 11 страниц попиксельно идентичны прежнему
экспорту; visual review всех 12 страниц, 17 layout и 9 PDF checks PASS.
Архитектура, формулы, h12 fallback, реальные примеры и PNG сохранены.
Новые проверки и прежние historical records доступны в
[presentation verification](presentation/verification.json).

Локальные доказательства — ignored `outputs/publication_blockers_review/`;
никаких fits, новых экспериментов, raw publication, deploy, history rewrite,
изменений visibility или commit/push не выполнялось.

## Окончательная подготовка существующего репозитория

Решение владельца закрывает выбор репозитория, раскрытие исторических путей/identities
и атрибуцию соответствующего адаптированного материала. Окончательный кандидат
готовится к review перед commit/merge. Изолированный Git export используется
только для проверки tracked inputs и не создаёт публичный snapshot.

Финальный staged candidate: 362 tracked Git inputs, clean build/check и 57 no-fit
tests PASS; PDF неизменён относительно исправленного экспорта. После добавления
указателя на существующий реальный пример в README пакет пересобран: 74 файла,
385 локальных links/anchors. README покрывает code/run/configs, методологию
и results/architecture/real examples. Доказательства текущего этапа — ignored
`outputs/final_public_preparation/`; старые проверки выше сохранены как provenance.

Команды commit, push, merge без squash/rebase, GitHub-only verification
и отдельная инструкция visibility/Pages — в
[PUBLISH_GITHUB_PAGES.md](../../docs/PUBLISH_GITHUB_PAGES.md).
GitHub-only проверка нового commit возможна после согласованного push;
фактическая публикация проверяется после отдельного manual deployment.
Ни одно из этих действий в подготовке не выполнялось.
