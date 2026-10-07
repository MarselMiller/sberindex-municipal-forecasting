# E08a — Leading Financial Indicators: source / availability / vintage audit

Дата аудита: **2026-10-07**. Ветка: `research/e08-leading-indicators`.
Исходный HEAD и локальный `main`: `70c95d6d4686a991aa225d7e0649dda0d8e9b0ff`;
рабочая папка перед E08a была чистой. Main, результаты E01–E07/F1–F7,
README, итоговые отчёты, презентация, модели, configs и прежние outputs не меняются.

**Результат исследования источников:** проверены 10 candidate series из шести
обязательных семейств: **A=2, B=7, C=1**. Shortlist содержит шесть семейств,
из них два основных и четыре только exploratory/sensitivity. Полезность для
forecasting или предупреждения реальных событий **не оценена**.

## 1. Вопрос и граница вывода

Могут ли финансовые показатели, действительно доступные до forecast origin,
дать дополнительный сигнал о будущих изменениях месячного значения категории
«Все категории» по МО? Цель — оценка средних безналичных расходов жителей
в номинальных рублях. Аудит определяет допустимые источники и information set;
он не устанавливает опережающую предиктивную связь или экономический причинный эффект.

Для causal backtest 2023–2024 **без очевидной временной утечки подходят
ключевая ставка и официальный USD/RUB** при доверии официальному nonrevision
archive, правильном различении publication/effective dates и явном cutoff.
Это условная пригодность источника, а не уже выполненная полная реконструкция.
Для остальных рядов first-release timing или vintage semantics не подтверждены
достаточно: нельзя включать current XLSX/curve history в основной causal backtest
лишь добавлением произвольного лага.

## 2. Что сохранено из существующего проекта

Прочитаны README, DATA_NOTICE, PROJECT_CONTEXT, TASKS, RESULTS_SUMMARY,
METHODOLOGY_REPORT, limitations и DATA_PUBLICATION_AUDIT; дополнительно
`configs/macro_forecast.yaml`, `configs/news_events_v3.yaml` и код availability.
Target CSV, реальные строки outputs и приватные источники для E08a не открывались.

- История цели — январь 2023…декабрь 2024, только 24 месяца; национальный
  финансовый feature, скопированный по 2190 МО, остаётся одной временной
  реализацией. Новые финансовые источники не добавляют target months.
- Forecasting origins существующего протокола — декабрь 2023…ноябрь 2024;
  horizons 1/3/6/12 календарных месяцев. MAE в исходной шкале остаётся обязательной.
  Validation задаётся target months до июня 2024; holdout июля…декабря уже просмотрен.
- `release_lag_months=0` остаётся неподтверждённым допущением доступности цели.
  Неизвестные target vintages не исправляются аудитом финансовых источников.
- Для historical training pair с origin `r` признаки доступны на собственную
  `r`, обучающая метка допускается при `r+h+L <= O`. Более позднее знание
  на общей training origin `O` нельзя переносить назад.
- E05c использовал только national annual `forecast_inflation_dec_dec_pct`
  и `forecast_consumption_growth_annual_pct`, с опубликованным значением,
  шириной диапазона, midpoint flag, missing flag и publication age.
  Selector требовал одновременно `published_at`, `available_at` и
  `forecast_issue_date <= own r`, статус A и год целевого месяца.
  Ни один из 10 финансовых рядов этого каталога не совпадает с этими indicators.
- Current-vintage macro B, региональные ИПЦ/зарплата и deflation в E05c не участвовали.
  E08 не переименовывает их в новые финансовые leading features.
- News v3 уже использует доверенные ядра key-rate decisions. Поэтому числовой
  уровень/накопленное изменение ставки — новая репрезентация тех же событий,
  а не независимый новый news corpus.
- Real early warning остаётся feasibility-задачей с недостаточными независимыми
  onset dates и известными к cutoff метками. Метка окна известна не раньше
  `O+k+2`; unknown/censored не становятся negatives. Новые financial features
  не снимают отказ прежнего gate и не разрешают обучение classifier.

Проектные основания: [temporal protocol и macro/EW](../final/METHODOLOGY_REPORT.md),
[зафиксированные результаты](../final/RESULTS_SUMMARY.md),
[ограничения](../final/limitations.md), [контекст](../../docs/PROJECT_CONTEXT.md).

## 3. Даты, vintage и рекомендуемая конвенция E08b

Сохранять отдельно `reference_period`, `observed_at`, `published_at`,
`effective_at`, `available_at_upper_bound`, `vintage_id`, `retrieved_at`,
`source_url/hash`, точность времени и основание допуска. Нынешний HTTP
Last-Modified, имя файла с периодом или время торгов не заменяют first publication.

Важная найденная особенность: E05c работает с month-end dates без времени,
а `src/sberforecast/news_features.py` преобразует origin в **00:00
Europe/Moscow последнего дня месяца**, затем UTC, а не конец этого дня.
Для финансового слоя E08b рекомендуется явно сохранить этот midnight cutoff.
Это спецификация к следующему этапу: существующий код/разбиение не изменены,
матрица признаков не построена. Перенос cutoff на 23:59 требует отдельного решения.

Если доступен только день реально подтверждённого выпуска, консервативная
доступность — **00:00 следующего календарного дня**, без выдуманного intraday time.
Если FX опубликован до effective date, её midnight можно использовать как
консервативную верхнюю границу availability: это не actual published_at.
Курс, уже действующий ровно в этот cutoff, допустим; будущий effective курс
не превращается в текущий уровень. Для key rate объявленная будущая ставка
может существовать только как отдельный announcement feature; in-force rate
дополнительно требует `effective_at <= O`.

**A** — официальный исторический маршрут с достаточным timing/nonrevision или
vintage evidence, при явном archive trust без независимых old SHA.
**B** — historical values есть, но исторический timing/version не полностью
подтверждён: только exploratory/sensitivity.
**C** — требуемый historical information set не реконструирован: исключить из
final causal backtest. Fixed lag сам по себе не повышает B до A.

## 4. Полный каталог проверенных рядов

Подробные 21 обязательное поле, proposed features, coverage basis, evidence IDs
и decision каждого series — в [CSV](../../data/metadata/leading_financial_series.csv).
[JSON](../../data/metadata/leading_financial_sources.json) содержит тот же каталог,
source URLs, dated evidence, результаты format probes и research assumptions.
Коды USE обозначают следующий исследовательский допуск, а не измеренное качество.

| Series ID | Проверенное определение / источник | Класс | Решение для E08 |
| --- | --- | --- | --- |
| `cbr_key_rate` | [ЦБ: ключевая ставка](https://www.cbr.ru/hd_base/KeyRate/), отдельно in-force/announcement | A | USE |
| `cbr_usd_rub_official` | [ЦБ: официальный USD/RUB](https://www.cbr.ru/currency_base/dynamics/), не intraday close | A | USE |
| `cbr_individual_credit_stock_rub` | [Задолженность ФЛ-резидентов, рубли](https://www.cbr.ru/vfs/statistics/BankSector/Mortgage/02_05_Debt_ind.xlsx), включая ипотеку | B | USE_WITH_CAVEAT |
| `cbr_unsecured_consumer_credit_monthly` | [Необеспеченный кредит: месячные narrative reports](https://www.cbr.ru/Collection/Collection/File/43486/razv_bs_22_10.pdf); full compatible series не подтверждён | C | DO_NOT_USE |
| `cbr_individual_deposits_excl_escrow_rub` | [Вклады, депозиты и другие средства ФЛ, без эскроу, рубли](https://www.cbr.ru/vfs/statistics/banksector/borrowings/02_27_Dep_ind_excluding_escrow.xlsx), включая текущие счета | B | USE_WITH_CAVEAT |
| `cbr_individual_deposit_rate_rub_le1y_excl_demand` | [Weighted RUB deposit rate ФЛ до 1 года, кроме до востребования](https://www.cbr.ru/vfs/statistics/pdko/int_rat/deposits.xlsx) | B | USE_WITH_CAVEAT |
| `cbr_individual_loan_rate_rub_le1y` | [Weighted RUB loan rate ФЛ до 1 года, включая до востребования](https://www.cbr.ru/vfs/statistics/pdko/int_rat/loans_ind.xlsx), не unsecured-only | B | USE_WITH_CAVEAT |
| `cbr_top10_max_deposit_rate_rub` | [Средняя максимальная ставка top-10, декадная](https://www.cbr.ru/statistics/avgprocstav/); другой показатель, чем weighted rate | B | DO_NOT_USE: отложен вне shortlist |
| `cbr_ofz_zcyc_1y` | [ЦБ/MOEX: fitted zero-coupon yield, 1 год](https://www.cbr.ru/hd_base/zcyc_params/) | B | USE_WITH_CAVEAT |
| `cbr_ofz_zcyc_10y` | Тот же официальный источник, 10 лет; slope 10y−1y | B | USE_WITH_CAVEAT |

Все выбранные маршруты — national. Региональные строки workbook не подключаются
без отдельной проверки publication timing и исторического mapping МО→регион.
Municipal financial history не заявлена. Optional inflation expectations,
money supply, real wages и unemployment не добавлялись: шесть обязательных
финансовых семейств уже покрывают scope, сильное дополнительное vintage evidence
и новый тип информации для расширения пока не установлены.

## 5. Publication lag и revision findings

### Monetary policy

[Официальные SDDS metadata](https://www.cbr.ru/statichtml/file/105133/ps_b.pdf)
сообщают об отсутствии уточнений опубликованной ключевой ставки.
[16.09.2022](https://www.cbr.ru/press/pr/?file=16092022_133000Key.htm)
объявление в 13:30 предшествовало effective date 19.09;
[15.08.2023](https://www.cbr.ru/press/pr/?file=15082023_103000Key.htm)
внеочередное решение опубликовано в 10:30 со вступлением в силу в тот же день.
[26.07.2024](https://www.cbr.ru/press/pr/?file=26072024_133000key.htm)
даёт ещё один timestamped original core. Нельзя всем событиям назначать 13:30
или использовать позднее резюме вместо первого объявления.

### FX

[FX SDDS](https://www.cbr.ru/vfs/statistics/ssrd/meth/exr_b.pdf)
разделяет установление/публикацию и подтверждает отсутствие уточнений.
[FAQ](https://www.cbr.ru/faq/foreign_exchange_market/) с ответами,
обновлёнными 07.05.2024, указывает вступление в силу следующим календарным днём.
Проверенные historical pages показывают effective/update пары
11.01.2022/10.01.2022 и 01.07.2023/30.06.2023; запрос 31.12.2024 возвращает
действующее с 29.12.2024 значение. Полнота всех effective dates пока не проверена.

Bare и www copies metadata вернули разные ссылки на методологические акты:
это текущие copies, а не два независимо подтверждённых historical snapshots.
FAQ «обычно до 18:00» обновлён в январе 2025; переносить это время назад нельзя.
[Релиз 13.06.2024](https://cbr.ru/press/pr/?id=39834) подтверждает переход
USD/EUR на банковскую отчётность о внебиржевых операциях.
Указанное 15:30 — cutoff исходных сделок, не timestamp публикации курса.
Изменение метода нужно сохранять как regime metadata.

### Credit / household funds

Current XLSX labels подтверждают историю: credit 2019-02-01…2026-09-01,
household funds 2019-07-01…2026-09-01. Credit workbook содержит все monthly
snapshot labels 2022–2024 и 2025-01-01. Это **проверка текущего history layout,
не historical vintages**. Snapshot 01.02 соответствует завершённому январю,
но не означает, что январская информация была доступна 31.01.

Архив [BBS](https://www.cbr.ru/statistics/bbs/) содержит publication tooltips:
[№3/2022](https://www.cbr.ru/Collection/Collection/File/40970/Bbs2203r.pdf)
размещён 18.05.2022,
[№3/2023](https://www.cbr.ru/Collection/Collection/File/43890/Bbs2303r.pdf)
— 07.04.2023,
[№3/2024](https://www.cbr.ru/Collection/Collection/File/49066/Bbs2403r.pdf)
— 05.04.2024. Issue month не publication month. Эти даты относятся к BBS
и могут дать позднюю консервативную доступность именно его версии;
они не доказанный first-release lag каждого показателя.

В current sors HTML файл с reference date 20240301 имеет placement date
30.09.2025; файлы 20220301/20230301 — 30.08.2023.
Поэтому date-specific filename не гарантирует первый historical vintage.
[Выпуск за сентябрь 2022](https://www.cbr.ru/Collection/Collection/File/43415/razv_bs_22_09.pdf)
прямо отмечает исправление previous-month growth после замены отчётности банков.

Для credit требуется сверка current XLSX с BBS 6.3.8, а не таблицей 4.3.2
«кредиты и прочие средства» с иным perimeter. Для household funds — сверка
0409302/current XLSX с 0409101/BBS и состава текущих счетов, нерезидентов,
ВЭБ, процентов и эскроу. BBS deposits table 4.2.1 в 2023 стала 4.2.2 в 2024.
Рублёвая серия выбрана, чтобы не смешивать growth с FX revaluation;
stock ratios не равны опубликованным adjusted growth из narrative reports.

### Retail rates

Current files содержат deposits Jan2014…Jul2026 и archived household loans
Jan2014…Dec2025. Современный loans_ind_new с layout 2026 для этого окна не нужен.
В [current calendar](https://www.cbr.ru/calendar/) rates за месяц m публикуются
в m+2, но actual first releases 2023–2024 полностью не восстановлены.
Январские ставки присутствуют в BBS, размещённом 07.04.2023 / 05.04.2024:
это консервативная bulletin availability, а не единый исторический лаг.

[Current methodology](https://www.cbr.ru/Content/Document/File/135988/meth_rates.pdf)
описывает annualised transaction-weighted rates и exclusions.
Поздние детали методологии нельзя автоматически переносить назад.
Household loan rate не равен цене всех revolving lines или unsecured credit.
Top-10 quote имеет меняющийся набор банков, иной estimator и неопределённые
per-decade publication dates; история значений не решает эти вопросы.

### OFZ

[MOEX methodology page](https://www.moex.com/a3642) описывает fitted curve и
closing parameters, [архив](https://www.moex.com/ru/marketdata/indices/state/g-curve/archive/)
сохраняет closing history. Historical CBR pages за 10.01.2022,
29.12.2023 и 30.12.2024 проверены; полнота всех торговых дней не проверялась.

CBR SDDS задаёт **следующий рабочий день** распространения.
Календарь расчётной базы меняется ежемесячно, описан 3m retrospective window;
это не доказывает фактического overwriting прошлых кривых, но immutable
first-release semantics также не подтверждены.
[Объявление новой базы 13.09.2023](https://www.moex.com/n63910) с effective date
15.09 подтверждает chronology состава, а не vintage неизменность yields.
`tradedate/tradetime` ISS не считать publication timestamp.

## 6. Причинная агрегация: спецификация, без расчёта features

На каждой собственной origin сначала выбрать допустимую версию; затем считать
агрегаты. Не вычислять rolling/returns на current full history и только потом
отрезать будущие строки. Для B это правило устраняет очевидный time-order leakage,
но не устраняет unknown revisions.

- **Key rate:** действующее as-of значение; signed change в п.п.;
  сумма effective changes в предыдущих 3/6 календарных месяцах;
  дни с последнего изменения, без reset на unchanged decision.
  Announcement features отделены от effective changes.
- **FX:** last effective available, mean/close доступного месячного префикса,
  1m/3m log change и realized volatility по новым установлениям курса.
  Не принимать официальный курс за intraday close; повторённые выходные не
  являются независимыми returns. Prefix close при midnight cutoff может
  отличаться от полного closing value календарного месяца.
- **Stocks:** MoM/YoY и acceleration по последним опубликованным reference
  months, из одного совместимого vintage, с missing/age flags.
  Для ранних 2023 origins задержка публикации и YoY могут потребовать denominators
  из 2021, а не только 2022; при их отсутствии сохранить missingness.
- **Retail rates:** last published, изменения в п.п., spread относительно key
  rate в **том же reference month**, а не слепое вычитание текущей policy rate
  из старого статистического месяца; публикационный возраст сохранять отдельно.
- **OFZ:** fixed 1y/10y tenors, доступные closing yields, изменение за месяц,
  slope 10y−1y по одной дате/совместимому vintage.
  Sensitivity availability: следующий российский рабочий день и до следующего
  calendar midnight при неизвестном intraday времени. Это допущение, не повышение B.

Не использовать future month close, interpolation будущих выпусков, full-sample
нормализацию/отбор по корреляции или перемешивание nationwide rows.
Для производного feature все operands и окно должны быть доступны на r.
Нулевое значение, отсутствующая публикация и stale observation различаются.

## 7. Shortlist и отклонения

Выбрано **шесть семейств**, восемь series representations. Подходы ранжированы
по economic interpretation и availability; target correlation не считалась.

| Приоритет | Семейство | Решение | Допуск |
| --- | --- | --- | --- |
| 1 | Key rate | USE / A | Основной causal loader, decision/effective ledger |
| 1 | Official USD/RUB | USE / A | Основной loader, conservative effective-date bound |
| 2 | Все кредиты ФЛ-резидентов, рубли | USE_WITH_CAVEAT / B | Vintage recovery / отдельная exploratory ветка |
| 2 | Средства ФЛ без эскроу, рубли | USE_WITH_CAVEAT / B | Definition parity и archive-vintage recovery |
| 3 | Weighted deposit/household-loan rates | USE_WITH_CAVEAT / B | Lagged transmission sensitivity, historical exclusions |
| 3 | OFZ 1y/10y curve | USE_WITH_CAVEAT / B | Publication calendar и vintage check; отдельная sensitivity |

Unsecured monthly series — **C / DO_NOT_USE**: отдельные preliminary narratives
не образуют проверенный полный ряд. Нельзя достраивать его вычитанием total−mortgage−car
без одинаковых perimeters или интерполировать квартальные данные назад.
Top-10 max deposit rate — **B / DO_NOT_USE в текущем shortlist**:
отложен как возможная замена weighted rate лишь после dated-decade release audit.
B не означает обязательного включения каждого доступного источника.

## 8. Что реально выполнено и что не проверено

Локально выполнены read-only PowerShell/rg/git чтения, проверка ветки/HEAD/status,
чтение configs/source и SHA256 snapshot **255 tracked files**.
Python 3.12.10, PowerShell 5.1.26100.1882, Git 2.53.0.windows.2;
seed неприменим: случайных вычислений нет.

Веб-поиск/чтение официальных страниц и PDF выполнены. Минимальные HTTP probes
через `Invoke-WebRequest -UseBasicParsing` с timeout выполнялись в памяти:

| Запрос / формат | Реальный результат |
| --- | --- |
| XML_dynamic, USD, 10–11.01.2022 | HTTP 200; ValCurs/Record, 1 запись effective 11.01; windows-1251 |
| SOAP KeyRate, 27–30.12.2024 | HTTP 200; Envelope/KR, 3 observation dates; DT/Rate |
| CBR curve HTML, 30.12.2024 | HTTP 200; одна таблица с датой и tenors |
| MOEX ISS zcyc JSON, запрос одной даты 29.12.2023 | HTTP 200, schema params; **17972 rows, фильтр НЕ подтверждён** |
| Четыре current банковских XLSX | MIME/ZIP и sheets/period labels разобраны без сохранения численных series |
| Historical year=2023 calendar query | Вернул rendered даты 2026; historical schedule не подтверждён |

Точные endpoints, SOAP action, request dates и metadata результатов — в JSON.
Запрос MOEX намеренно был однодатным, но сервер вернул больший диапазон; он не
сохранён, повторного collection не было. Не выдавать это за успешный filtered loader.
Начальные sandbox transport failures преодолены разрешёнными read-only запросами;
они не означают отсутствие данных. Первые schema/encoding checks были исправлены
и повторены, не скрыты. Formal MOEX PDF binary не прочитан — его содержание не evidence.

Не проверены: полный ежедневный/месячный availability ledger, все source vintages
и coverage на каждой forecast origin, эквивалентность BBS/current XLSX,
вклад признаков, real early-warning predictive utility, новое temporal split.
Нет feature matrix, корреляций, MAE/PR-AUC/F1 или новых model/forecasting runs.
Конфигурация/loader file не нужны: probes использовали уже имеющиеся инструменты.

### Локальные проверки артефактов

**PASS для локальных metadata/protocol checks.** JSON/CSV: 10 одинаковых строк,
28 полей, включая 21 обязательное; 40 уникальных evidence/probe IDs разрешаются.
Shortlist: 6 families / 8 series; classifications/decisions согласованы;
relative links существуют. Content scan и осмотр собственных source-free artifacts
не нашли абсолютных drive paths, credential patterns, raw observations/target rows.
Независимый read-only review подтвердил согласованность с temporal protocol;
он не является повторным независимым web-аудитом каждого источника.

Из 255 исходных tracked-файлов 253 совпадают по SHA256; изменены только
PROJECT_CONTEXT и TASKS. HEAD/main совпадают с исходным base, ветка сохранена.
`git status --short`: два изменённых docs, два новых metadata; report ignored.
`git diff --check`: exit 0, без вывода. `git diff --stat`: 2 файла, 68 добавленных строк.
`git diff --cached --stat`: пусто. Report дополнительно сравнен с NUL через
`git diff --no-index --stat`: exit 1 означает различие нового файла, не ошибку;
единственное warning касается потенциальной LF→CRLF конвертации Git.
Команды и границы проверок сохранены в JSON. Pytest не запускался:
research/production code change нет, выполнена stdlib artifact validation.

Созданы только этот отчёт и два metadata файла; обновляются E08a statuses в
PROJECT_CONTEXT/TASKS. Действующее `/reports/*` скрывает новый results report:
`.gitignore` и index не меняются, отчёт следует просматривать напрямую.
Current source availability не является разрешением публиковать numeric/raw data;
применена metadata-only граница [F7](../final/DATA_PUBLICATION_AUDIT.md).

## 9. Точная рекомендация E08b

Сначала зафиксировать отдельную acquisition/as-of спецификацию **только для
key rate и official USD/RUB**, без обучения: восстановить initial anchors,
decision/effective ledger и FX settings за 2022–2024, сохранить provenance и
проверить допустимую историю на всех прежних origins и собственных r historical pairs.
Явно закрепить timezone/midnight cutoff, date-only availability bounds,
праздники/пустые ответы и regime change FX. Полную регулярность не предполагать.

После этого отдельным gate решать archive reconstruction C/D/E и OFZ sensitivity.
Без reconciliation они не входят в основной A-layer. Unsecured и top-10 пока
исключены. Ни шестичленный shortlist, ни этот аудит не разрешают model fits.
Будущие эксперименты требуют отдельного протокола, одинаковых forecast cases,
MAE в исходной шкале и прежних календарных horizons; просмотренный holdout
остаётся ретроспективным. Real classifier нельзя запускать без достаточных
доступных меток и независимых onset periods.

NO MODEL FITS

NO NEW FORECASTING EXPERIMENTS

NO EARLY-WARNING TRAINING

NO COMMIT/PUSH

E08a source/vintage audit complete.
Ready for E08b: acquisition/as-of specification; model training requires a separate decision.
