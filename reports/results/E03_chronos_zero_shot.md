# E03 — Chronos-2, zero-shot

Дата запуска: 2026-10-06. Конфигурация: configs/chronos_zero_shot.yaml.
Результаты: outputs/chronos_zero_shot_v1/. Git commit: 4eee7762112b2866e3efbf0010be951a3751f96f.
Незакоммиченные изменения на запуске: True.

## Постановка и сохранение протокола

Прогнозируется месячное значение категории «Все категории» по МО в исходных рублях — оценка средних безналичных расходов жителей.
Zero-shot: веса не дообучаются, подбор параметров по validation или holdout не выполняется. На каждом выпуске модель получает только исторический префикс до O−L, без будущих фактов и прогнозов других выпусков.
Сохранены даты выпуска декабрь 2023 — ноябрь 2024, горизонты 1/3/6/12 календарных месяцев, validation по июнь 2024 и holdout июль–декабрь 2024. Требования прогнозной истории ≥12 наблюдений и давности ≤1 месяца сохранены. Лаг публикации L=0 остаётся неподтверждённым допущением.
Выборка — исходные 64 идентификатора E01, без исключения МО 1471 по будущему отсутствию фактов; 63 МО имеют оцениваемые факты. Календарные пропуски остаются NaN и не сдвигают месяцы.
CatBoostDirect, CatBoostRecursive, SeasonalNaiveYoY и оба Prophet взяты из сохранённых прогнозов E01/E02b. Соперники повторно не обучаются. Годовой резерв E02b исключён из чистого CatBoostDirect: для h=12 на декабрь 2023 нет обучающих пар и нет годовой MAE обученной прямой модели.

## Модель и происхождение весов

Checkpoint: amazon/chronos-2; revision: 29ec3766d36d6f73f0696f85560a422f50e8498c. Версия пакета: 2.3.2.
Device=cpu, dtype=float32, batch_size=1, seed=42; cross_learning=False, quantile=0.5.
Точечный прогноз — квантиль 0.5. Постобработка отсутствует: отрицательные значения не обрезаются. Длина совместного прогноза соответствует наибольшему нужному календарному горизонту от cutoff.
Chronos-2 — модель примерно на 120 млн параметров, опубликованная 20 октября 2025 года. Этот checkpoint не существовал в исторических датах оценки 2023–2024; E03 — ретроспективное исследовательское сравнение с современной моделью. [Официальный репозиторий](https://github.com/amazon-science/chronos-forecasting).
Предобучение включает публичные временные ряды из наборов Chronos/GIFT и синтетические данные. Отсутствие пересечения предобучения с нашими данными и информацией за оцениваемый период не доказано. Ограничение префикса в адаптере проверяет доступность локального входа, но не историческую чистоту весов. [Карточка модели: training data](https://huggingface.co/amazon/chronos-2#training-data), [статья, раздел 4](https://arxiv.org/html/2510.15821v1#S4).

## Команды и статус

```powershell
.\.venv\Scripts\python.exe -B -X utf8 -m venv .venv-chronos
.\.venv-chronos\Scripts\python.exe -B -X utf8 -m pip install --disable-pip-version-check --cache-dir outputs/e03_checks/pip_cache --retries 0 --timeout 20 --report outputs/e03_checks/install_report.json -r requirements-chronos.txt
.\.venv-chronos\Scripts\python.exe -B -X utf8 -m pip check
.\.venv\Scripts\python.exe -B -X utf8 -m pytest -p no:cacheprovider --basetemp=outputs/e03_checks/pytest_full_run3
.\.venv-chronos\Scripts\python.exe -B -X utf8 -m pytest tests/test_chronos_model.py tests/test_chronos_evaluation.py -p no:cacheprovider --basetemp=outputs/e03_checks/pytest_chronos_run1
.\.venv-chronos\Scripts\python.exe -B -X utf8 scripts/run_chronos.py --config configs/chronos_zero_shot.yaml --preflight-only
.\.venv-chronos\Scripts\python.exe -B -X utf8 scripts/run_chronos.py --config configs/chronos_zero_shot.yaml --smoke
C:\Users\user\Desktop\Classes\sberindex_python_mvp\.venv-chronos\Scripts\python.exe -B -X utf8 scripts/run_chronos.py --config configs/chronos_zero_shot.yaml
```

Все команды завершились с кодом 0. Полный pytest перед smoke: **137 passed, 1 warning in 17.25s**; предупреждение относится к прежней синтетической проверке Inf в E02a. В отдельном окружении Chronos: **39 passed in 4.28s**, без загрузки весов. `pip check`: No broken requirements found.
Python 3.12.10; torch 2.8.0+cpu, chronos-forecasting 2.3.2, transformers 5.18.0, accelerate 1.15.0, huggingface_hub 1.33.0, safetensors 0.8.0. Все 47 установленных пакетов сохранены в requirements-chronos-lock.txt; прежняя .venv и requirements не обновлялись.

Все выпуски рассчитаны: True; полное сравнение: True. Выпусков завершено: 12 из 12.
Chronos: запрошено 1897, native=1897, failed=0, без целевого факта=7.
Fallback отсутствует. Ошибка инференса сохраняется как failed с NaN-прогнозом и причиной, а не заменяется другой моделью или исчезает из полной оценки.

## Покрытие

Полная область определяется наличием факта, независимо от успешности нового прогноза. Равенство выборок проверяется по municipality_id + forecast_origin + target_period + horizon.

| split | model | horizon | n_requested | n_actual_available | n_forecast_available | n_native | n_failed | n_unavailable | n_fallback | n_e01_cases | forecast_coverage | failed_share | fallback_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| holdout | CatBoostDirect | 1 | 378 | 378 | 378 | 378 | 0 | 0 | 0 | 378 | 100.00% | 0.00% | 0.00% |
| holdout | CatBoostDirect | 3 | 378 | 378 | 378 | 378 | 0 | 0 | 0 | 378 | 100.00% | 0.00% | 0.00% |
| holdout | CatBoostDirect | 6 | 379 | 378 | 379 | 379 | 0 | 0 | 0 | 378 | 100.00% | 0.00% | 0.00% |
| holdout | CatBoostDirect | 12 | 64 | 63 | 0 | 0 | 0 | 64 | 0 | 63 | 0.00% | 0.00% | 0.00% |
| holdout | CatBoostRecursive | 1 | 378 | 378 | 378 | 378 | 0 | 0 | 0 | 378 | 100.00% | 0.00% | 0.00% |
| holdout | CatBoostRecursive | 3 | 378 | 378 | 378 | 378 | 0 | 0 | 0 | 378 | 100.00% | 0.00% | 0.00% |
| holdout | CatBoostRecursive | 6 | 379 | 378 | 379 | 379 | 0 | 0 | 0 | 378 | 100.00% | 0.00% | 0.00% |
| holdout | CatBoostRecursive | 12 | 64 | 63 | 64 | 64 | 0 | 0 | 0 | 63 | 100.00% | 0.00% | 0.00% |
| holdout | Chronos-2 | 1 | 378 | 378 | 378 | 378 | 0 | 0 | 0 | 378 | 100.00% | 0.00% | 0.00% |
| holdout | Chronos-2 | 3 | 378 | 378 | 378 | 378 | 0 | 0 | 0 | 378 | 100.00% | 0.00% | 0.00% |
| holdout | Chronos-2 | 6 | 379 | 378 | 379 | 379 | 0 | 0 | 0 | 378 | 100.00% | 0.00% | 0.00% |
| holdout | Chronos-2 | 12 | 64 | 63 | 64 | 64 | 0 | 0 | 0 | 63 | 100.00% | 0.00% | 0.00% |
| holdout | ProphetAuto | 1 | 378 | 378 | 378 | 378 | 0 | 0 | 0 | 378 | 100.00% | 0.00% | 0.00% |
| holdout | ProphetAuto | 3 | 378 | 378 | 378 | 378 | 0 | 0 | 0 | 378 | 100.00% | 0.00% | 0.00% |
| holdout | ProphetAuto | 6 | 379 | 378 | 379 | 379 | 0 | 0 | 0 | 378 | 100.00% | 0.00% | 0.00% |
| holdout | ProphetAuto | 12 | 64 | 63 | 64 | 64 | 0 | 0 | 0 | 63 | 100.00% | 0.00% | 0.00% |
| holdout | ProphetYearly | 1 | 378 | 378 | 378 | 378 | 0 | 0 | 0 | 378 | 100.00% | 0.00% | 0.00% |
| holdout | ProphetYearly | 3 | 378 | 378 | 378 | 378 | 0 | 0 | 0 | 378 | 100.00% | 0.00% | 0.00% |
| holdout | ProphetYearly | 6 | 379 | 378 | 379 | 379 | 0 | 0 | 0 | 378 | 100.00% | 0.00% | 0.00% |
| holdout | ProphetYearly | 12 | 64 | 63 | 64 | 64 | 0 | 0 | 0 | 63 | 100.00% | 0.00% | 0.00% |
| holdout | SeasonalNaiveYoY | 1 | 378 | 378 | 378 | 378 | 0 | 0 | 0 | 378 | 100.00% | 0.00% | 0.00% |
| holdout | SeasonalNaiveYoY | 3 | 378 | 378 | 378 | 378 | 0 | 0 | 0 | 378 | 100.00% | 0.00% | 0.00% |
| holdout | SeasonalNaiveYoY | 6 | 379 | 378 | 379 | 379 | 0 | 0 | 0 | 378 | 100.00% | 0.00% | 0.00% |
| holdout | SeasonalNaiveYoY | 12 | 64 | 63 | 64 | 64 | 0 | 0 | 0 | 63 | 100.00% | 0.00% | 0.00% |
| validation | CatBoostDirect | 1 | 380 | 378 | 380 | 380 | 0 | 0 | 0 | 378 | 100.00% | 0.00% | 0.00% |
| validation | CatBoostDirect | 3 | 254 | 252 | 254 | 254 | 0 | 0 | 0 | 252 | 100.00% | 0.00% | 0.00% |
| validation | CatBoostDirect | 6 | 64 | 63 | 64 | 64 | 0 | 0 | 0 | 63 | 100.00% | 0.00% | 0.00% |
| validation | CatBoostRecursive | 1 | 380 | 378 | 380 | 380 | 0 | 0 | 0 | 378 | 100.00% | 0.00% | 0.00% |
| validation | CatBoostRecursive | 3 | 254 | 252 | 254 | 254 | 0 | 0 | 0 | 252 | 100.00% | 0.00% | 0.00% |
| validation | CatBoostRecursive | 6 | 64 | 63 | 64 | 64 | 0 | 0 | 0 | 63 | 100.00% | 0.00% | 0.00% |
| validation | Chronos-2 | 1 | 380 | 378 | 380 | 380 | 0 | 0 | 0 | 378 | 100.00% | 0.00% | 0.00% |
| validation | Chronos-2 | 3 | 254 | 252 | 254 | 254 | 0 | 0 | 0 | 252 | 100.00% | 0.00% | 0.00% |
| validation | Chronos-2 | 6 | 64 | 63 | 64 | 64 | 0 | 0 | 0 | 63 | 100.00% | 0.00% | 0.00% |
| validation | ProphetAuto | 1 | 380 | 378 | 380 | 380 | 0 | 0 | 0 | 378 | 100.00% | 0.00% | 0.00% |
| validation | ProphetAuto | 3 | 254 | 252 | 254 | 254 | 0 | 0 | 0 | 252 | 100.00% | 0.00% | 0.00% |
| validation | ProphetAuto | 6 | 64 | 63 | 64 | 64 | 0 | 0 | 0 | 63 | 100.00% | 0.00% | 0.00% |
| validation | ProphetYearly | 1 | 380 | 378 | 380 | 380 | 0 | 0 | 0 | 378 | 100.00% | 0.00% | 0.00% |
| validation | ProphetYearly | 3 | 254 | 252 | 254 | 254 | 0 | 0 | 0 | 252 | 100.00% | 0.00% | 0.00% |
| validation | ProphetYearly | 6 | 64 | 63 | 64 | 64 | 0 | 0 | 0 | 63 | 100.00% | 0.00% | 0.00% |
| validation | SeasonalNaiveYoY | 1 | 380 | 378 | 380 | 380 | 0 | 0 | 0 | 378 | 100.00% | 0.00% | 0.00% |
| validation | SeasonalNaiveYoY | 3 | 254 | 252 | 254 | 254 | 0 | 0 | 0 | 252 | 100.00% | 0.00% | 0.00% |
| validation | SeasonalNaiveYoY | 6 | 64 | 63 | 64 | 64 | 0 | 0 | 0 | 63 | 100.00% | 0.00% | 0.00% |

Общее успешное пересечение и размер исключений по группам:

| split | horizon | n_full | n_common_success | n_excluded | incomplete | participants | direct_status | common_success_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| holdout | 1 | 378 | 378 | 0 | False | Chronos-2 / CatBoostDirect / SeasonalNaiveYoY / CatBoostRecursive / ProphetAuto / ProphetYearly | included | 100.00% |
| holdout | 3 | 378 | 378 | 0 | False | Chronos-2 / CatBoostDirect / SeasonalNaiveYoY / CatBoostRecursive / ProphetAuto / ProphetYearly | included | 100.00% |
| holdout | 6 | 378 | 378 | 0 | False | Chronos-2 / CatBoostDirect / SeasonalNaiveYoY / CatBoostRecursive / ProphetAuto / ProphetYearly | included | 100.00% |
| holdout | 12 | 63 | 63 | 0 | False | Chronos-2 / SeasonalNaiveYoY / CatBoostRecursive / ProphetAuto / ProphetYearly | no_training_pairs | 100.00% |
| validation | 1 | 378 | 378 | 0 | False | Chronos-2 / CatBoostDirect / SeasonalNaiveYoY / CatBoostRecursive / ProphetAuto / ProphetYearly | included | 100.00% |
| validation | 3 | 252 | 252 | 0 | False | Chronos-2 / CatBoostDirect / SeasonalNaiveYoY / CatBoostRecursive / ProphetAuto / ProphetYearly | included | 100.00% |
| validation | 6 | 63 | 63 | 0 | False | Chronos-2 / CatBoostDirect / SeasonalNaiveYoY / CatBoostRecursive / ProphetAuto / ProphetYearly | included | 100.00% |

Для h=12 участники успешного сравнения — Chronos и четыре модели E01; чистый Direct обозначен как unavailable/no_training_pairs. Это структурное отсутствие обучения, а не сбой Chronos.

## MAE macro на полной области

MAE macro — среднее MAE муниципалитетов с равными весами. Если модель имеет failed или unavailable в полной группе, её MAE этой группы не вычисляется. Ошибочные случаи не удаляются молча.

### Holdout

| Модель | h=1 | h=3 | h=6 | h=12 |
| --- | --- | --- | --- | --- |
| Chronos-2 | 1947.35 | 2526.33 | 4150.59 | 9163.47 |
| CatBoostDirect | 986.22 | 1939.17 | 3349.50 | нет обучающих пар |
| CatBoostRecursive | 986.22 | 1693.72 | 3759.86 | 12215.92 |
| SeasonalNaiveYoY | 920.70 | 1240.40 | 1906.01 | 4322.14 |
| ProphetAuto | 1868.56 | 2041.31 | 2165.78 | 3054.18 |
| ProphetYearly | 1682.31 | 1679.62 | 1937.65 | 8496.01 |

### Validation

| Модель | h=1 | h=3 | h=6 | h=12 |
| --- | --- | --- | --- | --- |
| Chronos-2 | 2316.65 | 3064.93 | 2754.10 | нет случаев |
| CatBoostDirect | 3104.94 | 2918.37 | 2570.63 | нет случаев |
| CatBoostRecursive | 3104.94 | 5408.99 | 7875.25 | нет случаев |
| SeasonalNaiveYoY | 1354.90 | 2468.22 | 4777.16 | нет случаев |
| ProphetAuto | 1730.34 | 1436.84 | 1981.26 | нет случаев |
| ProphetYearly | 2736.68 | 3209.01 | 8362.49 | нет случаев |

MAE micro и pooled R² Chronos-2 в исходной шкале:

| split | horizon | n_predictions | n_success | n_failed | n_unavailable | mae_macro | mae_micro | r2_pooled | metric_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| holdout | 1 | 378 | 378 | 0 | 0 | 1947.3527 | 1947.3527 | 0.9326 | complete |
| holdout | 3 | 378 | 378 | 0 | 0 | 2526.3331 | 2526.3331 | 0.8936 | complete |
| holdout | 6 | 378 | 378 | 0 | 0 | 4150.5933 | 4150.5933 | 0.7887 | complete |
| holdout | 12 | 63 | 63 | 0 | 0 | 9163.4710 | 9163.4710 | 0.3447 | complete |
| validation | 1 | 378 | 378 | 0 | 0 | 2316.6547 | 2316.6547 | 0.9143 | complete |
| validation | 3 | 252 | 252 | 0 | 0 | 3064.9326 | 3064.9326 | 0.8789 | complete |
| validation | 6 | 63 | 63 | 0 | 0 | 2754.0982 | 2754.0982 | 0.9037 | complete |

Общее успешное пересечение совпадает с полной областью для всех участвующих моделей; его метрики также сохранены отдельно в metrics_common_success.csv.

## Smoke и затраты ресурсов

Smoke: passed=True, рядов=2, prediction_length=12, инференс=0.4302501999991364 с, загрузка=53.06818000000021 с.
Полный запуск: загрузка модели=25.970910100000765 с, инференс=147.62055500000315 с, общее время=186.76854079999976 с. Эти времена относятся к зафиксированной CPU-среде и включают только явно указанные фазы.

| seconds | n_series | n_calendar_history_months | n_missing_history_values | n_history_observations | prediction_length | history_cutoff | path_first_month | path_last_month | quantile | cross_learning | forecast_origin | ram_total_bytes | ram_available_bytes | disk_free_bytes | process_rss_bytes | process_peak_rss_bytes | cpu_logical_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 9.111501800000042 | 64 | 12 | 0 | 768 | 12 | 2023-12-01 | 2024-01-01 | 2024-12-01 | 0.5 | False | 2023-12-31 | 7448702976 | 1068630016 | 278680461312 | 823705600 | 833908736 | 8 |
| 10.215127200001008 | 64 | 13 | 1 | 831 | 6 | 2024-01-01 | 2024-02-01 | 2024-07-01 | 0.5 | False | 2024-01-31 | 7448702976 | 1045798912 | 278680391680 | 823791616 | 833908736 | 8 |
| 10.163096999998745 | 63 | 14 | 0 | 882 | 6 | 2024-02-01 | 2024-03-01 | 2024-08-01 | 0.5 | False | 2024-02-29 | 7448702976 | 1062043648 | 278680322048 | 823873536 | 833908736 | 8 |
| 10.091352299999926 | 63 | 15 | 0 | 945 | 6 | 2024-03-01 | 2024-04-01 | 2024-09-01 | 0.5 | False | 2024-03-31 | 7448702976 | 1039245312 | 278679203840 | 823869440 | 833908736 | 8 |
| 10.441680900001302 | 63 | 16 | 0 | 1008 | 6 | 2024-04-01 | 2024-05-01 | 2024-10-01 | 0.5 | False | 2024-04-30 | 7448702976 | 1367379968 | 278679064576 | 823910400 | 833908736 | 8 |
| 13.857118900001296 | 63 | 17 | 0 | 1071 | 6 | 2024-05-01 | 2024-06-01 | 2024-11-01 | 0.5 | False | 2024-05-31 | 7448702976 | 1327460352 | 278678994944 | 827330560 | 833908736 | 8 |
| 14.046645000000352 | 63 | 18 | 0 | 1134 | 6 | 2024-06-01 | 2024-07-01 | 2024-12-01 | 0.5 | False | 2024-06-30 | 7448702976 | 1318961152 | 278678925312 | 827838464 | 833908736 | 8 |
| 14.61048469999878 | 63 | 19 | 0 | 1197 | 3 | 2024-07-01 | 2024-08-01 | 2024-10-01 | 0.5 | False | 2024-07-31 | 7448702976 | 1277210624 | 278678851584 | 828166144 | 833908736 | 8 |
| 14.314854899999771 | 63 | 20 | 0 | 1260 | 3 | 2024-08-01 | 2024-09-01 | 2024-11-01 | 0.5 | False | 2024-08-31 | 7448702976 | 1250250752 | 278678781952 | 828080128 | 833908736 | 8 |
| 13.120414000000892 | 63 | 21 | 0 | 1323 | 3 | 2024-09-01 | 2024-10-01 | 2024-12-01 | 0.5 | False | 2024-09-30 | 7448702976 | 1242251264 | 278678646784 | 828379136 | 833908736 | 8 |
| 13.5366122000014 | 63 | 22 | 0 | 1386 | 1 | 2024-10-01 | 2024-11-01 | 2024-11-01 | 0.5 | False | 2024-10-31 | 7448702976 | 1231671296 | 278678573056 | 828063744 | 833908736 | 8 |
| 14.111666099999638 | 63 | 23 | 0 | 1449 | 1 | 2024-11-01 | 2024-12-01 | 2024-12-01 | 0.5 | False | 2024-11-30 | 7448702976 | 1219424256 | 278678503424 | 828522496 | 833908736 | 8 |

| Этап | Момент | Сохранённые ресурсы |
| --- | --- | --- |
| smoke | before | {"cpu_logical_count": 8, "disk_free_bytes": 279159934976, "process_peak_rss_bytes": 100839424, "process_rss_bytes": 82112512, "ram_available_bytes": 1718927360, "ram_total_bytes": 7448702976} |
| smoke | after | {"cpu_logical_count": 8, "disk_free_bytes": 278681755648, "process_peak_rss_bytes": 841244672, "process_rss_bytes": 822464512, "ram_available_bytes": 1152815104, "ram_total_bytes": 7448702976} |
| полный запуск | before | {"cpu_logical_count": 8, "disk_free_bytes": 278680711168, "process_peak_rss_bytes": 100319232, "process_rss_bytes": 80703488, "ram_available_bytes": 1813733376, "ram_total_bytes": 7448702976} |
| полный запуск | after | {"cpu_logical_count": 8, "disk_free_bytes": 278677626880, "process_peak_rss_bytes": 840204288, "process_rss_bytes": 829857792, "ram_available_bytes": 1207635968, "ram_total_bytes": 7448702976} |

## Вывод и ограничения

На holdout h=1/3/6 Chronos проиграл всем пяти сравнимым соперникам. На h=12 он лучше Recursive, но хуже YoY и обоих Prophet; pure Direct здесь не оценён. Превосходство zero-shot Chronos на этом фиксированном пилоте не подтверждено.
На validation Chronos лучше Recursive и ProphetYearly на всех доступных горизонтах, однако хуже ProphetAuto. Параметры, медиана, история, календарь и временные границы после просмотра результатов не изменялись.

h=1: MAE Chronos-2 1947.35; минимальная MAE среди доступных соперников — SeasonalNaiveYoY, 920.70. Ошибка Chronos больше.
h=3: MAE Chronos-2 2526.33; минимальная MAE среди доступных соперников — SeasonalNaiveYoY, 1240.40. Ошибка Chronos больше.
h=6: MAE Chronos-2 4150.59; минимальная MAE среди доступных соперников — SeasonalNaiveYoY, 1906.01. Ошибка Chronos больше.
h=12: MAE Chronos-2 9163.47; минимальная MAE среди доступных соперников — ProphetAuto, 3054.18. Ошибка Chronos больше.

Эти числа описывают фиксированный пилот. Уже просмотренный holdout не является новой независимой проверкой, и результат не используется для смены дат, выборки или настройки модели.
Месячная история короткая: число МО и прогнозных строк не заменяет число различных дат. Годовой прогноз проверяется на одной дате выпуска; validation h=6 также имеет одну дату. Рейтинг нельзя переносить на полную панель МО или устойчивость прогноза будущих шоков.
Фактические даты публикации и vintages не подтверждены; возможные пересечения предобучения и недоступность текущего checkpoint в 2023–2024 ограничивают вывод о временной чистоте. Преимущество фундаментальной модели требует отдельной сопоставимой проверки, а не следует из её класса.

## Воспроизводимость и источники чисел

predictions.csv.gz сохраняет факты, прогнозы, статусы, effective_model и причины. metrics_full.csv и metrics_common_success.csv содержат MAE macro/micro, pooled R², размеры и статусы групп; evaluation_keys_*.csv и excluded_common_success_keys.csv задают области и исключения.
coverage*.csv и comparison_coverage.csv сохраняют покрытие, timings.csv — время по выпуску, partitions/errors_*.json — ошибки. Smoke сохранён отдельно. Manifest фиксирует команду, seed, версии, Git/status, хеши входа, кода, исходных результатов и скачанных весов.
Исходные данные, outputs/baseline_v1/, E01 и E02b сохраняются отдельно от E03. Построчные результаты и веса остаются в outputs и автоматически в Git не добавляются.

Независимый расчёт проверил 82 числовые группы MAE macro/micro/R², 5166 строк метрик МО, 41 ячейку MAE и 21 ячейку дополнительных метрик отчёта; max_abs=1.82e-12. Проверены все 3735 месячных позиций, хеши кода, входа, исходных результатов и весов. Все 295 защищённых файлов и 30 пакетов прежней .venv сохранены; тестировавшийся код не менялся. Диагностика: outputs/e03_checks/{independent_verification,preservation_checks,test_validation}.json.

Хеши весов из сохранённого manifest:

```text
model.safetensors: ddcda3c7508bf2528087723e98a20707cc04b7f370ae275a9fd88078ddba4f42
```
