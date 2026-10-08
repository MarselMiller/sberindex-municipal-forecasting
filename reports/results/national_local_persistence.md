# National/Local Persistence: фиксированная методологическая ablation

## Исследовательский вопрос

Проверить, какая часть улучшения National/Local + LightGBM связана с самим разложением ряда на общую национальную и локальную компоненты, а какая — с обучаемой моделью локальной динамики.

## Существующее разложение National/Local

N_t — медиана конечных значений полной панели всех доступных МО, а не только пилотной выборки; R_i,t = y_i,t / N_t при конечном положительном N_t. Переиспользованы без изменений build_national, ratio_panel и national_forecast. Национальный прогноз — прежний SeasonalNaiveYoY с теми же параметрами роста и шагом h+L. Протокол, sample, eligibility, факты, split и ключи взяты из [конфигурации E05d](../../configs/national_local_lightgbm.yaml) и сверены с сохранёнными predictions.

## Определение Persistence

Для каждого origin используется последний конечный causal-доступный R по t≤O−L; y_hat = N_hat × R_last для всех h. Пропущенный последний ratio заменяется только более ранним известным ratio, без сглаживания или нового ограничения давности. Нет обучения, tuning, clipping ratio и поиска признаков. Невалидный национальный прогноз или отсутствие anchor сохраняют прежнюю failed-политику всей партии, без подстановки другой модели.

У Persistence нет требования наличия обучающих пар. Поэтому на h=12 он применяет ту же формулу, а LightGBMDirect и National/Local + LightGBM сохраняют SeasonalNaive fallback. h=12 исключён из основного вывода.

## Reproduction gate

**PASS.** Использован пересчёт сохранённых predictions, а не новый fit LightGBM. Проверены source SHA, параметры/seed, фиксированные ключи и факты, strategy/native метрики и origin-метрики, национальные forecasts. Численный допуск — 1e−8; подробности находятся в [run_manifest.json](national_local_persistence/run_manifest.json).

## Validation: MAE macro, исходные рубли

| Strategy | h1 | h3 | h6 |
| --- | --- | --- | --- |
| LastValue | 2394.74 | 2476.35 | 1496.14 |
| SeasonalNaiveYoY | 1354.90 | 2468.22 | 4777.16 |
| LightGBMDirect | 2733.30 | 2880.02 | 6378.07 |
| National/Local Persistence | 1437.28 | 2816.00 | 4994.06 |
| National/Local + LightGBM | 1370.55 | 2819.74 | 5525.34 |

## Holdout: MAE macro, исходные рубли

| Strategy | h1 | h3 | h6 |
| --- | --- | --- | --- |
| LastValue | 1998.76 | 2528.44 | 3371.34 |
| SeasonalNaiveYoY | 920.70 | 1240.40 | 1906.01 |
| LightGBMDirect | 1003.28 | 1449.82 | 2463.21 |
| National/Local Persistence | 864.08 | 1285.65 | 2031.59 |
| National/Local + LightGBM | 799.55 | 1212.73 | 1897.22 |

Все стратегии оценены на прежних конечных фактах. Missing/failed не исключаются для улучшения метрик; coverage и статусы сохранены в [metrics.csv](national_local_persistence/metrics.csv). h12 и pooled R² приведены в этом же CSV.

## Устойчивость по forecast origins

Положительный Δ означает снижение MAE при переходе from → to. Каждая календарная origin учитывается один раз; среднее origin-Δ не подменяет общую macro MAE.

| split | horizon | from_model | to_model | n_origins | n_evaluable_origins | n_improved | mean_origin_reduction | median_origin_reduction | best_origin | worst_origin |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| holdout | 1 | LastValue | NationalLocalPersistence | 6 | 6 | 4 | 1134.67 | 436.75 | 2024-11-30 | 2024-10-31 |
| holdout | 1 | LightGBMDirect | NationalLocalPersistence | 6 | 6 | 3 | 139.20 | 14.52 | 2024-08-31 | 2024-11-30 |
| holdout | 1 | NationalLocalPersistence | NationalLocalLightGBM | 6 | 6 | 4 | 64.53 | 85.47 | 2024-11-30 | 2024-09-30 |
| holdout | 1 | SeasonalNaiveYoY | NationalLocalPersistence | 6 | 6 | 4 | 56.61 | 108.30 | 2024-09-30 | 2024-06-30 |
| holdout | 3 | LastValue | NationalLocalPersistence | 6 | 6 | 6 | 1242.79 | 456.96 | 2024-09-30 | 2024-07-31 |
| holdout | 3 | LightGBMDirect | NationalLocalPersistence | 6 | 6 | 5 | 164.17 | 181.95 | 2024-06-30 | 2024-09-30 |
| holdout | 3 | NationalLocalPersistence | NationalLocalLightGBM | 6 | 6 | 5 | 72.92 | 36.77 | 2024-09-30 | 2024-04-30 |
| holdout | 3 | SeasonalNaiveYoY | NationalLocalPersistence | 6 | 6 | 3 | -45.25 | -79.89 | 2024-06-30 | 2024-04-30 |
| holdout | 6 | LastValue | NationalLocalPersistence | 6 | 6 | 4 | 1339.75 | 846.13 | 2024-01-31 | 2024-03-31 |
| holdout | 6 | LightGBMDirect | NationalLocalPersistence | 6 | 6 | 3 | 431.62 | 309.41 | 2024-02-29 | 2024-06-30 |
| holdout | 6 | NationalLocalPersistence | NationalLocalLightGBM | 6 | 6 | 5 | 134.37 | 168.34 | 2024-03-31 | 2024-01-31 |
| holdout | 6 | SeasonalNaiveYoY | NationalLocalPersistence | 6 | 6 | 3 | -125.58 | -4.74 | 2024-05-31 | 2024-02-29 |
| validation | 1 | LastValue | NationalLocalPersistence | 6 | 6 | 4 | 957.47 | 778.49 | 2023-12-31 | 2024-03-31 |
| validation | 1 | LightGBMDirect | NationalLocalPersistence | 6 | 6 | 3 | 1296.02 | 209.08 | 2023-12-31 | 2024-03-31 |
| validation | 1 | NationalLocalPersistence | NationalLocalLightGBM | 6 | 6 | 5 | 66.73 | 10.99 | 2024-02-29 | 2023-12-31 |
| validation | 1 | SeasonalNaiveYoY | NationalLocalPersistence | 6 | 6 | 3 | -82.38 | -51.41 | 2023-12-31 | 2024-03-31 |
| validation | 3 | LastValue | NationalLocalPersistence | 4 | 4 | 2 | -339.65 | -633.06 | 2024-01-31 | 2023-12-31 |
| validation | 3 | LightGBMDirect | NationalLocalPersistence | 4 | 4 | 2 | 64.02 | 58.09 | 2024-01-31 | 2023-12-31 |
| validation | 3 | NationalLocalPersistence | NationalLocalLightGBM | 4 | 4 | 1 | -3.74 | -28.85 | 2023-12-31 | 2024-02-29 |
| validation | 3 | SeasonalNaiveYoY | NationalLocalPersistence | 4 | 4 | 1 | -347.78 | -575.92 | 2023-12-31 | 2024-02-29 |
| validation | 6 | LastValue | NationalLocalPersistence | 1 | 1 | 0 | -3497.91 | -3497.91 | 2023-12-31 | 2023-12-31 |
| validation | 6 | LightGBMDirect | NationalLocalPersistence | 1 | 1 | 1 | 1384.01 | 1384.01 | 2023-12-31 | 2023-12-31 |
| validation | 6 | NationalLocalPersistence | NationalLocalLightGBM | 1 | 1 | 0 | -531.28 | -531.28 | 2023-12-31 | 2023-12-31 |
| validation | 6 | SeasonalNaiveYoY | NationalLocalPersistence | 1 | 1 | 0 | -216.90 | -216.90 | 2023-12-31 | 2023-12-31 |

Полные best/worst Δ и counts доступны в [origin_stability.csv](national_local_persistence/origin_stability.csv), отдельные даты — в [origin_deltas.csv](national_local_persistence/origin_deltas.csv).

## Диагностика по МО

Описательная доля МО, где Persistence имеет меньшую MAE; равенства не считаются выигрышами. Это не significance test. Идентификаторы и построчные расходы не публикуются.

| split | horizon | reference | n_municipalities | n_evaluable | n_better | share_better | median_municipality_reduction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| holdout | 1 | SeasonalNaiveYoY | 63 | 63 | 36 | 0.57 | 75.13 |
| holdout | 1 | LightGBMDirect | 63 | 63 | 47 | 0.75 | 126.23 |
| holdout | 1 | NationalLocalLightGBM | 63 | 63 | 23 | 0.37 | -58.05 |
| holdout | 3 | SeasonalNaiveYoY | 63 | 63 | 37 | 0.59 | 133.63 |
| holdout | 3 | LightGBMDirect | 63 | 63 | 32 | 0.51 | 18.87 |
| holdout | 3 | NationalLocalLightGBM | 63 | 63 | 28 | 0.44 | -15.46 |
| holdout | 6 | SeasonalNaiveYoY | 63 | 63 | 36 | 0.57 | 145.23 |
| holdout | 6 | LightGBMDirect | 63 | 63 | 50 | 0.79 | 291.43 |
| holdout | 6 | NationalLocalLightGBM | 63 | 63 | 31 | 0.49 | -15.21 |
| validation | 1 | SeasonalNaiveYoY | 63 | 63 | 25 | 0.40 | -106.12 |
| validation | 1 | LightGBMDirect | 63 | 63 | 63 | 1.00 | 1227.60 |
| validation | 1 | NationalLocalLightGBM | 63 | 63 | 25 | 0.40 | -21.11 |
| validation | 3 | SeasonalNaiveYoY | 63 | 63 | 22 | 0.35 | -219.78 |
| validation | 3 | LightGBMDirect | 63 | 63 | 38 | 0.60 | 102.41 |
| validation | 3 | NationalLocalLightGBM | 63 | 63 | 38 | 0.60 | 67.84 |
| validation | 6 | SeasonalNaiveYoY | 63 | 63 | 30 | 0.48 | -56.60 |
| validation | 6 | LightGBMDirect | 63 | 63 | 42 | 0.67 | 1825.79 |
| validation | 6 | NationalLocalLightGBM | 63 | 63 | 56 | 0.89 | 286.63 |

## Decomposition gap analysis

ΔMAE = MAE_from − MAE_to, relative Δ = 100×ΔMAE/MAE_from. Доля observed MAE gap = (MAE_Direct−MAE_Persistence)/(MAE_Direct−MAE_NL_LGBM), только при положительном знаменателе; отрицательные значения и значения >1 не обрезаются. Это описательная доля наблюдаемого разрыва, не causal contribution.

| split | horizon | from_model | to_model | reduction_mae | reduction_pct | observed_share_of_mae_gap |
| --- | --- | --- | --- | --- | --- | --- |
| holdout | 1 | LastValue | NationalLocalPersistence | 1134.67 | 56.77 | NA |
| holdout | 1 | SeasonalNaiveYoY | NationalLocalPersistence | 56.61 | 6.15 | NA |
| holdout | 1 | LightGBMDirect | NationalLocalPersistence | 139.20 | 13.87 | 0.68 |
| holdout | 1 | NationalLocalPersistence | NationalLocalLightGBM | 64.53 | 7.47 | NA |
| holdout | 3 | LastValue | NationalLocalPersistence | 1242.79 | 49.15 | NA |
| holdout | 3 | SeasonalNaiveYoY | NationalLocalPersistence | -45.25 | -3.65 | NA |
| holdout | 3 | LightGBMDirect | NationalLocalPersistence | 164.17 | 11.32 | 0.69 |
| holdout | 3 | NationalLocalPersistence | NationalLocalLightGBM | 72.92 | 5.67 | NA |
| holdout | 6 | LastValue | NationalLocalPersistence | 1339.75 | 39.74 | NA |
| holdout | 6 | SeasonalNaiveYoY | NationalLocalPersistence | -125.58 | -6.59 | NA |
| holdout | 6 | LightGBMDirect | NationalLocalPersistence | 431.62 | 17.52 | 0.76 |
| holdout | 6 | NationalLocalPersistence | NationalLocalLightGBM | 134.37 | 6.61 | NA |
| validation | 1 | LastValue | NationalLocalPersistence | 957.47 | 39.98 | NA |
| validation | 1 | SeasonalNaiveYoY | NationalLocalPersistence | -82.38 | -6.08 | NA |
| validation | 1 | LightGBMDirect | NationalLocalPersistence | 1296.02 | 47.42 | 0.95 |
| validation | 1 | NationalLocalPersistence | NationalLocalLightGBM | 66.73 | 4.64 | NA |
| validation | 3 | LastValue | NationalLocalPersistence | -339.65 | -13.72 | NA |
| validation | 3 | SeasonalNaiveYoY | NationalLocalPersistence | -347.78 | -14.09 | NA |
| validation | 3 | LightGBMDirect | NationalLocalPersistence | 64.02 | 2.22 | 1.06 |
| validation | 3 | NationalLocalPersistence | NationalLocalLightGBM | -3.74 | -0.13 | NA |
| validation | 6 | LastValue | NationalLocalPersistence | -3497.91 | -233.80 | NA |
| validation | 6 | SeasonalNaiveYoY | NationalLocalPersistence | -216.90 | -4.54 | NA |
| validation | 6 | LightGBMDirect | NationalLocalPersistence | 1384.01 | 21.70 | 1.62 |
| validation | 6 | NationalLocalPersistence | NationalLocalLightGBM | -531.28 | -10.64 | NA |

Полная точность: [gap_analysis.csv](national_local_persistence/gap_analysis.csv).

## Категория интерпретации

**D: NO STABLE DECOMPOSITION BENEFIT**.

На holdout baseline без learned local model закрывает 68.32%, 69.24% и 76.26%
наблюдаемого MAE-разрыва LightGBMDirect → National/Local + LightGBM на h=1/3/6.
Однако относительно SeasonalNaiveYoY улучшение есть только на holdout h=1;
на validation Persistence уступает ему на всех трёх горизонтах. Добавление
LightGBM к Persistence снижает holdout MAE на 7.47%, 5.67% и 6.61%, но на
validation h=3/6 ухудшает результат. Направление и распределение выигрыша
по датам не подтверждают устойчивое преимущество по заданному правилу.
Поэтому итог — D, при наблюдаемом снижении MAE Persistence относительно
LightGBMDirect. Доли 106.20% и 162.30% на validation h=3/6
означают, что Persistence лучше learned NL на этих случаях; это не causal share.


Правило зафиксировано в [новой конфигурации](../../configs/national_local_persistence.yaml) до расчёта baseline: существенное улучшение ≥5% к LastValue и SeasonalNaiveYoY; близость к learned MAE — не хуже более чем на 5%; для «most» требуется закрыть ≥50% положительного Direct→NL gap. Строгое большинство origins — >1/2. Сначала D при смене направления между split либо отсутствии строгого большинства origins, согласующихся с направлением агрегированной разницы. Далее A требует существенного улучшения простых baselines, близости к learned MAE и большинства положительного gap; B — такого же улучшения и дополнительного снижения MAE с learner минимум на 5%; C — улучшения менее 5% к каждому простому baseline в обоих split и снижения MAE с learner минимум на 5% относительно Persistence и raw baselines. Остальные смешанные случаи относятся к D. Общая A/B/C допустима только при одинаковой категории на всех h=1/3/6, иначе D. Это описательная классификация, не статистическая значимость.

| horizon | category | instability | substantial_simple_baseline_benefit | close_to_learned | little_simple_baseline_benefit | closes_most_positive_direct_gap | learned_adds_value |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | D | True | False | False | False | True | False |
| 3 | D | True | False | False | False | True | False |
| 6 | D | True | False | False | False | True | False |

## Ограничения

На h=12 MAE всех пяти стратегий совпадает до приведённой точности:
4322.142857 руб. Максимальное различие Persistence и сезонного ориентира
в отдельных predictions — около 7.28e−12 руб., из-за вычислений с float.
Единичные strict wins/losses по МО на h=12 отражают эту численную точность,
а не экономически значимый эффект; годовой горизонт не участвует в выводе.

Holdout уже просмотрен; этот эксперимент не является новым blind test. История целевого показателя — 24 месяца (2023–2024), сохранена пилотная выборка и категория «Все категории». Доступность при release_lag_months=0 остаётся допущением. Для validation h=6 есть одна origin, для holdout h=1/3/6 — шесть; h12 имеет одну origin и только описательное значение. Число МО не создаёт независимые временные наблюдения. Разложение включает national forecast и сохраняемое относительное положение; доля MAE gap не изолирует causal contribution. Пересчёт saved predictions подтверждает метрики и артефакты, но не является свежим переобучением reference learners.

Новых model fits: **0**. Runtime полного запуска: **189.357 s**. Seed=42; версии и source hashes сохранены в manifest. Построчные predictions, ratio-provenance, resolved config и диагностические таблицы остаются в ignored output_dir.

[Агрегаты по МО](national_local_persistence/municipality_win_rates.csv) · [Метаданные запуска](national_local_persistence/run_manifest.json)
