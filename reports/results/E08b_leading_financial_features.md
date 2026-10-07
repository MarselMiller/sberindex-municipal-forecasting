# E08b — causal loading and origin-aware leading financial features

Ветка `research/e08-leading-indicators`, исходный HEAD `f6f0cef87889d01d3dd121b2858bf123eabf7264`. Seed 42. Команда: `.\.venv\Scripts\python.exe -B -X utf8 scripts/build_leading_financial_features.py --config configs/leading_financial_e08b.yaml`.

**E08b complete. Ready for E08c — causal feature layer при указанных archive assumptions.** Код подготовлен; источники загружены; матрица и автоматические audit checks выполнены; 65 synthetic tests passed. Предиктивная полезность не оценивалась.

## Источники и реконструкция

Запрошено 2021-01-01…2024-12-31. Ключевая ставка: 1016 daily observations (2021-01-04…2024-12-30), 23 реальных изменений + initial anchor. USD/RUB: 990 settings (2021-01-01…2024-12-29). После декабря 2024 значений нет.

Ключевая ставка сверена с архивными решениями: decision_date, announced_at, effective_date, rate, source URL/hash и точность времени сохранены отдельно в cache ledger. Effective date берётся из прямого текста либо наблюдаемого перехода официального daily history, с явным basis; понедельник не вычисляется по расписанию. Для каждого изменения требуется единственное совпадение rate/effective date; unmatched change останавливает сборку. Точный timestamp используется только при прямом подтверждении; date-only decision доступен со следующего calendar midnight. In-force level требует одновременно availability и effective≤O. Первый daily anchor допускается со следующего дня, с неизвестным prior delta; он не устанавливает дату последнего изменения. Unchanged decisions не сбрасывают last change.

Ledger: 36 решений, 13 unchanged; точное время 20 / date-only 16. Среди 23 changes: 9 exact / 14 date-only. Effective date 3 изменений подтверждена прямо в core, 20 — официальным daily transition. Неподтверждённые 00:00 footer не считаются доказанным midnight publication; их причина неизвестна. Внеочередное 28.02.2022 допускается только 01.03, 15.08.2023 сохраняет фактическое 10:30.

FX Record.Date — effective date. available_at = effective midnight Europe/Moscow — консервативная верхняя граница при официальной публикации до вступления в силу, не выдуманный actual published_at. Рубли за 1 USD нормализуются через Nominal; VunitRate, дубли, неположительные значения проверяются. Holidays/weekends используют последний допустимый setting, без future backfill. Снимки архива получены сейчас; допуск A опирается на nonrevision/archive trust E08a, независимых historical SHA нет.

[Ставка](https://www.cbr.ru/hd_base/KeyRate/), [календарь решений](https://www.cbr.ru/dkp/cal_mp/), [SOAP KeyRate](https://www.cbr.ru/DailyInfoWebServ/DailyInfo.asmx?op=KeyRate), [FX XML documentation](https://www.cbr.ru/development/SXML/), [FX FAQ](https://www.cbr.ru/faq/foreign_exchange_market/), [E08a evidence/catalog](../../data/metadata/leading_financial_sources.json).

## Origins и признаки

Матрица содержит 12 существующих forecasting origins 2023-12-31 00:00:00+03:00…2024-11-30 00:00:00+03:00. Они получены через backtest.origins_from_config и data.period_end из неизменного configs/macro_forecast.yaml; target не читается. Cutoff — 00:00 Москвы последнего календарного дня, как в news feature layer. Intraday manual anchors ниже — проверки источников, не новые backtest origins. Горизонты/выборка/разбиение не менялись.

| Признак | Определение |
| --- | --- |
| key_rate_level | Последняя известная действующая ставка, % |
| key_rate_delta_last | Последнее известное ненулевое изменение, процентные пункты |
| key_rate_change_3m / key_rate_change_6m | level(O) − level(O−3/6 календарных месяцев); нижний anchor доступен на собственную дату |
| months_since_rate_change | Разница календарных month indices O и последнего effective change; не число дней/30 |
| usd_rub_last | Последний действующий/допустимый курс, RUB за 1 USD |
| usd_rub_change_1m / usd_rub_change_3m | ln(last(O)/last(O−1/3 календарных месяцев)); без умножения на 100 |
| usd_rub_vol_1m / usd_rub_vol_3m | std(ddof=1) log returns между соседними settings с ending effective date в (lower,O]; без annualisation |

Для change окон нужны оба causal endpoints; для volatility дополнительно минимум 2 returns и historical anchor≤lower. Начало первого return может предшествовать lower, поскольку return датируется вторым setting. Holiday daily zeros не добавляются. При недостатке истории — NaN и отдельный <feature>_missing; замена нулём или future value отсутствует. Полнота источника предполагает полноту официального ответа: отсутствие setting не доказывает торговый день без публикации. Max staleness не придумывается; anchor dates и return counts видны в аудите. DateOffset сохраняет номер календарного дня с ограничением длиной месяца: lower для 29.02−1m=29.01, а не предыдущая monthly origin 31.01. delta_last пересчитывается из двух доступных уровней на собственную origin, не копируется из глобального precomputed delta; оба operands включены в audit.

| feature | origins | nonmissing | missing |
| --- | --- | --- | --- |
| key_rate_level | 12 | 12 | 0 |
| key_rate_delta_last | 12 | 12 | 0 |
| key_rate_change_3m | 12 | 12 | 0 |
| key_rate_change_6m | 12 | 12 | 0 |
| months_since_rate_change | 12 | 12 | 0 |
| usd_rub_last | 12 | 12 | 0 |
| usd_rub_change_1m | 12 | 12 | 0 |
| usd_rub_change_3m | 12 | 12 | 0 |
| usd_rub_vol_1m | 12 | 12 | 0 |
| usd_rub_vol_3m | 12 | 12 | 0 |

Эти десять национальных признаков повторяются для МО на одной origin. Дублирование по 2190 МО не добавляет независимых финансовых наблюдений. Репликация проверена на двух synthetic MO identifiers, без чтения реального target/panel.

## Июнь 2024 и audit operands

[Релиз ЦБ 13.06.2024 14:40](https://cbr.ru/press/pr/?id=39834) задаёт изменение установления USD/EUR с 13 июня. При правиле next-calendar-day effective граница settings 13.06 соответствует first effective 14.06; это явно отмеченный вывод из двух правил, не timestamp XML publication. До объявления boundary metadata не раскрывается. Окна, пересекающие режимы, не корректируются и не отбрасываются; сравнимость volatility через границу ограничена. 15:30 в релизе — cutoff исходных сделок, не время публикации курса.

Аудит сохраняет last available/effective dates и timestamps для обоих источников, нижние anchors, return counts, regime и maxima effective/availability **всех использованных operands**. Требование ≤ own origin проверяется для обеих maxima.

| forecast_origin | key_rate_last_available_date | key_rate_level | usd_rub_last_available_date | usd_rub_last | max_source_date_used |
| --- | --- | --- | --- | --- | --- |
| 2023-12-31 00:00:00+03:00 | 2023-12-18 00:00:00+03:00 | 16 | 2023-12-30 00:00:00+03:00 | 89.6883 | 2023-12-30 00:00:00+03:00 |
| 2024-01-31 00:00:00+03:00 | 2023-12-18 00:00:00+03:00 | 16 | 2024-01-31 00:00:00+03:00 | 89.2887 | 2024-01-31 00:00:00+03:00 |
| 2024-02-29 00:00:00+03:00 | 2023-12-18 00:00:00+03:00 | 16 | 2024-02-29 00:00:00+03:00 | 91.8692 | 2024-02-29 00:00:00+03:00 |
| 2024-03-31 00:00:00+03:00 | 2023-12-18 00:00:00+03:00 | 16 | 2024-03-30 00:00:00+03:00 | 92.366 | 2024-03-30 00:00:00+03:00 |
| 2024-04-30 00:00:00+03:00 | 2023-12-18 00:00:00+03:00 | 16 | 2024-04-28 00:00:00+03:00 | 91.7791 | 2024-04-28 00:00:00+03:00 |
| 2024-05-31 00:00:00+03:00 | 2023-12-18 00:00:00+03:00 | 16 | 2024-05-31 00:00:00+03:00 | 89.7869 | 2024-05-31 00:00:00+03:00 |
| 2024-06-30 00:00:00+03:00 | 2023-12-18 00:00:00+03:00 | 16 | 2024-06-29 00:00:00+03:00 | 85.748 | 2024-06-29 00:00:00+03:00 |
| 2024-07-31 00:00:00+03:00 | 2024-07-29 00:00:00+03:00 | 18 | 2024-07-31 00:00:00+03:00 | 86.33 | 2024-07-31 00:00:00+03:00 |
| 2024-08-31 00:00:00+03:00 | 2024-07-29 00:00:00+03:00 | 18 | 2024-08-31 00:00:00+03:00 | 91.1868 | 2024-08-31 00:00:00+03:00 |
| 2024-09-30 00:00:00+03:00 | 2024-09-16 00:00:00+03:00 | 19 | 2024-09-28 00:00:00+03:00 | 92.7126 | 2024-09-28 00:00:00+03:00 |
| 2024-10-31 00:00:00+03:00 | 2024-10-28 00:00:00+03:00 | 21 | 2024-10-31 00:00:00+03:00 | 97.053 | 2024-10-31 00:00:00+03:00 |
| 2024-11-30 00:00:00+03:00 | 2024-10-28 00:00:00+03:00 | 21 | 2024-11-30 00:00:00+03:00 | 107.741 | 2024-11-30 00:00:00+03:00 |

## Ручные anchors

Каждая строка рассчитана по сохранённым sources, не вписана вручную. Таблица показывает, какой effective setting известен на cutoff, и часть значений; в manual_anchors.csv сохранены все десять features и missing flags.

| anchor_case | forecast_origin | key_rate_level | key_rate_delta_last | key_rate_change_3m | key_rate_change_6m | usd_rub_last | usd_rub_change_1m | usd_rub_vol_1m | usd_rub_last_effective_date | usd_rub_method_regime |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2022_date_only_decision_day | 2022-02-28 00:00:00+03:00 | 9.5 | 1 | 2 | 3 | 83.5485 | 0.0566506 | 0.0241625 | 2022-02-26 00:00:00+03:00 | pre_june_2024_setting_method |
| 2022_date_only_available_next_day | 2022-03-01 00:00:00+03:00 | 20 | 10.5 | 12.5 | 13.5 | 93.5589 | 0.188698 | 0.0349265 | 2022-03-01 00:00:00+03:00 | pre_june_2024_setting_method |
| 2023_same_day_before | 2023-08-15 10:29:00+03:00 | 8.5 | 1 | 1 | 1 | 101.04 | 0.114384 | 0.0085365 | 2023-08-15 00:00:00+03:00 | pre_june_2024_setting_method |
| 2023_same_day_after | 2023-08-15 10:30:00+03:00 | 12 | 3.5 | 4.5 | 4.5 | 101.04 | 0.114384 | 0.0085365 | 2023-08-15 00:00:00+03:00 | pre_june_2024_setting_method |
| 2023_weekend_month_end_actual_origin | 2023-12-31 00:00:00+03:00 | 16 | 1 | 3 | 8.5 | 89.6883 | 0.00900705 | 0.00815068 | 2023-12-30 00:00:00+03:00 | pre_june_2024_setting_method |
| 2024_fx_before_announcement | 2024-06-13 00:00:00+03:00 | 16 | 1 | 0 | 1 | 89.0214 | -0.0309958 | 0.00470568 | 2024-06-12 00:00:00+03:00 | pre_june_2024_setting_method |
| 2024_fx_announcement_not_yet_effective | 2024-06-13 14:40:00+03:00 | 16 | 1 | 0 | 1 | 89.0214 | -0.0309958 | 0.00470568 | 2024-06-12 00:00:00+03:00 | pre_june_2024_setting_method |
| 2024_fx_first_effective | 2024-06-14 00:00:00+03:00 | 16 | 1 | 0 | 1 | 88.208 | -0.0380938 | 0.00498878 | 2024-06-14 00:00:00+03:00 | bank_otc_reporting |
| 2024_weekend_month_end_actual_origin | 2024-06-30 00:00:00+03:00 | 16 | 1 | 0 | 0 | 85.748 | -0.0401284 | 0.0183438 | 2024-06-29 00:00:00+03:00 | bank_otc_reporting |
| 2024_rate_announcement_future_effective | 2024-07-26 13:30:00+03:00 | 16 | 1 | 0 | 0 | 85.41 | -0.0216238 | 0.00955538 | 2024-07-26 00:00:00+03:00 | bank_otc_reporting |
| 2024_rate_effective | 2024-07-29 00:00:00+03:00 | 18 | 2 | 2 | 2 | 85.565 | -0.00213644 | 0.00623206 | 2024-07-27 00:00:00+03:00 | bank_otc_reporting |
| 2024_month_end_actual_origin | 2024-07-31 00:00:00+03:00 | 18 | 2 | 2 | 2 | 86.33 | 0.0067644 | 0.00645616 | 2024-07-31 00:00:00+03:00 | bank_otc_reporting |

## Автоматические проверки

```json
{
  "all_operands_at_or_before_own_origin": true,
  "deterministic_including_input_permutation": true,
  "future_mutation_invariance_origins": 12,
  "future_removal_invariance_no_backfill_origins": 12,
  "national_features_equal_across_two_synthetic_municipalities": true,
  "june_method_boundary_in_audit": true,
  "cached_loader_deterministic": true
}
```

Future mutation меняет future key levels/deltas и FX values на каждой actual origin, затем сравнивает всю feature/audit row. Дополнительно будущие строки удаляются целиком. Loader и feature builder повторно читаются/выполняются, перестановка inputs проверяется на точное совпадение. Это проверки явной утечки, не доказательство полноты historical archive.

## Артефакты, ограничения и следующий этап

Raw XML/HTML, ledger и нормализованные public financial snapshots остаются в ignored outputs/leading_financial_e08b_v1/. Матрица/audit и report остаются в ignored reports/results/. Code/config/tests и две project docs пригодны для review; новые raw финансовые наблюдения в Git не добавляются. Manifest содержит hashes, относительные пути, seed, версии, command, commit, dirty state и coverage. Секреты/absolute local paths не сохраняются.

E08b не оценивает leading predictive value или causal economic effect. Ограничения цели сохраняются: лишь 24 target months, L=0 — допущение, target vintages неизвестны, holdout уже просмотрен. Национальные features не снимают feasibility gate реального early warning. Announcement features для ещё не действующей ставки отдельно не добавлены. Будущая сборка training pair должна вызывать этот builder на собственной historical r, а не переносить row общей training origin O назад.

Следующий конкретный шаг после прохождения tests: отдельно согласовать E08c, фиксированный forecasting ablation на прежних cases с features на own r. До согласования моделей/метрик/подбора не выполнялось.

NO MODEL FITS; NO FORECASTING METRICS; NO EARLY-WARNING TRAINING; NO COMMIT/PUSH.

## Итоговая проверка выполнения

Минимальная команда `.\.venv\Scripts\python.exe -B -X utf8 scripts/build_leading_financial_features.py --config configs/leading_financial_e08b.yaml --smoke` выполнена: 2 origins × 10 features, missing 0, code 0. Полная команда без `--smoke` выполнена: 12 origins × 10, missing 0, code 0; повтор после проверки URL/cache provenance дал те же CSV hashes. Финальный manifest содержит текущие code/config hashes. Python 3.12.10, pandas 2.2.3, numpy 2.3.5, PyYAML 6.0.3; seed 42. Source retrieval completion time для двух первых XML отмечено как file completion approximation, не publication time. Каждый HTML core имеет собственные HTTP provenance/retrieved_at/hash.

Релевантный тестовый запуск:

```powershell
.\.venv\Scripts\python.exe -B -X utf8 -m pytest tests/test_leading_financial.py tests/test_leading_financial_sources.py -p no:cacheprovider --basetemp=outputs/e08b_checks/pytest_final_v1
```

**65 passed in 1.92 s, code 0.** Basetemp расположен внутри проекта; его parent создан перед запуском. При воспроизведении выбрать новый leaf, чтобы не удалять предыдущий test output. Первые unit проверки выявили несовпадение exception type malformed XML: parsers исправлены на единый ValueError. Предварительные source tests получили setup errors из-за недоступной внешней tmp_path; первая общая попытка — 60 passed / 5 setup errors из-за отсутствующего parent нового basetemp. Ожидания tests не ослаблялись; после исправления путей все проверки прошли. Full model-fit suite не запускался.

Независимый reference использует stdlib XML/CSV/JSON и Python loops, без вызова feature builder. Пересчитаны 120 matrix cells и 120 manual anchor cells: **240 comparisons, 0 mismatches, max_abs=3.469446951953614e-18**, tolerance 1e-10. Сверены 12 origins по исходному config, оба source maxima, June publication metadata и шесть конкретных rate anchors. В actual origins окна FX содержат 16…23 returns для 1m и 57…66 для 3m; missing 0 не является искусственным zero fill.

`git diff --check` — PASS. Дополнительно проверены whitespace шести новых untracked review files через `git diff --no-index --check`; лишняя blank line at EOF в test file удалена. Content scan новых files/report на absolute user paths и типичные private-key/token patterns — без совпадений. Из 258 исходных tracked files 256 побайтово неизменны; изменены только две project docs. SHA256 всех 80 проверенных файлов data/input и baseline_v1 сохранены; HEAD/main и пустой index не изменились. Проверены 101 ignored artifact/cache/report paths и совпадение report CSV с output CSV. Исходные расходы для features/anchors не открывались: preservation читает только bytes для SHA. Все новые numeric financial snapshots остаются ignored; публикация/добавление в index не выполнялись.

Для review созданы `configs/leading_financial_e08b.yaml`, `scripts/build_leading_financial_features.py`, `src/sberforecast/leading_financial.py`, `src/sberforecast/leading_financial_sources.py`, `tests/test_leading_financial.py`, `tests/test_leading_financial_sources.py`; обновлены только `docs/PROJECT_CONTEXT.md` и `docs/TASKS.md`. Локальные результаты: `reports/results/e08b/{financial_features_by_origin.csv,manual_anchors.csv,feature_coverage.csv,causality_checks.json}`, `outputs/leading_financial_e08b_v1/run_manifest.json`; записи проверки — `outputs/e08b_checks/`.

Causal реконструкция ключевой ставки и official FX допускается при official nonrevision archive trust и консервативных availability bounds. **E08b complete; Ready for E08c** относится к этому слою данных. Следующий конкретный шаг — отдельно согласованный fixed forecasting ablation на прежних cases с own-r features. Результаты прежних экспериментов, target protocol и real-EW feasibility вывод не изменены.
