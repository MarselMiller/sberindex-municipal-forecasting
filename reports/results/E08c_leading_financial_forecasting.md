# E08c — forecasting ablation with leading financial indicators

**E08c forecasting ablation complete. Primary conclusion: NO STABLE FORECASTING UPLIFT.**

Ветка `research/e08-leading-indicators`, исходный HEAD `cb4db21a48031761f930876b21119a69b4433b35`, seed 42. Код подготовлен, relevant tests пройдены, fixed model commands выполнены, метрики получены из сохранённых predictions.

## Вопрос и фиксированный протокол

Цель — месячное значение категории «Все категории» по МО; исходная шкала и определение показателя сохранены из E05d. MAE приведена в рублях.

Дают ли causal key-rate/official-FX indicators дополнительную forecasting value сверх E05d и сохраняется ли знак улучшения по origins? Primary — LightGBMDirectNationalLocal (LN), sensitivity — LightGBMDirect (L0). Использован исходный native LightGBM adapter E05d, не другой estimator API. Национальный прогноз SeasonalNaiveYoY не изменён. Ни CatBoost, ни Prophet, ни Chronos не переобучались. Других learners или parameter search нет.

Параметры совпадают с E05d: objective=regression_l1, 300 rounds, learning_rate=0.05, num_leaves=31, random_state=42, n_jobs=2, CPU, deterministic=true, force_col_wise=true, use_missing=true, zero_as_missing=false. Native requested/actual/resolved params каждого fit сохранены в training diagnostics.

F0 — исходные 19 K0 features; F1 добавляет только пять key-rate features; F2 — только пять FX features; F3 — ровно десять E08b features. Missing flags, regime metadata и новые derived financial features не подаются learner.

```text
F0: no financial features
F1: key_rate_level, key_rate_delta_last, key_rate_change_3m, key_rate_change_6m, months_since_rate_change
F2: usd_rub_last, usd_rub_change_1m, usd_rub_change_3m, usd_rub_vol_1m, usd_rub_vol_3m
F3: key_rate_level, key_rate_delta_last, key_rate_change_3m, key_rate_change_6m, months_since_rate_change, usd_rub_last, usd_rub_change_1m, usd_rub_change_3m, usd_rub_vol_1m, usd_rub_vol_3m
```

Сохранены E05d data/category/target filters, legacy pair admission, training windows, origins Dec2023…Nov2024, validation target months до June2024 включительно, holdout July…Dec2024, horizons 1/3/6/12. Обучение использует все допустимые МО панели, оценка — неизменные 64 pilot IDs с прежней календарной eligibility. На variant: 1897 raw forecast keys, 1890 cases с фактом; missing actual сохранены в raw coverage. Неудачные forecasts не заменяются fallback и не удаляются молча.

## Success criterion до результатов

F3 получает PROMISING, только если macro MAE ниже F0 одновременно на validation и holdout как минимум на двух из h=1/3/6. На каждом таком horizon и каждом split с >=2 origins должны улучшаться >=2 origins и mean origin MAE delta после удаления лучшей origin оставаться отрицательной. Численный epsilon — 1e-8 руб.; h12 не участвует. Validation h6 имеет одну origin: концентрация UNDETERMINED, а не доказанная стабильность. Overall conclusion определяется только primary LN; результат L0 не используется для выбора победителя. Операционализация концентрации записана в YAML/manifest до первого E08c fit.

| model | conclusion | qualifying_horizons | concentration_pass | reasons |
| --- | --- | --- | --- | --- |
| LightGBMDirectNationalLocal | NO STABLE FORECASTING UPLIFT | [6] | True | ["F3 must improve both validation and holdout at >=2 of h=1/3/6"] |
| LightGBMDirect | NO STABLE FORECASTING UPLIFT | [] | False | ["F3 must improve both validation and holdout at >=2 of h=1/3/6"] |

## F0 reproduction gate

**PASS:** обе семьи воспроизведены до любых финансовых fits. Max absolute F0 prediction difference=0 руб. Max absolute aggregate metric difference=4.55e-13; atol=1e-8, rtol=0. По каждому partition сверены predictions/anchor/ratio/N_hat, keys/truths, split/history cutoff, statuses/reasons/fallback, K0 feature names/dtypes, training keys/order/targets/X signatures и requested/actual/native params. Resume требует matching fingerprint, completion marker и hashes всех partition artifacts; partial/failed checkpoint останавливается до нового fit.

## Validation

| model | variant | horizon | mae_macro | delta_mae_macro_vs_F0 | relative_delta_mae_macro_pct_vs_F0 | r2_pooled | n_origins_improved | n_origins_worsened | n_origins |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LightGBMDirectNationalLocal | F0 | 1 | 1370.546 | 0.000 | 0.000 | 0.968 | 0 | 0 | 6 |
| LightGBMDirectNationalLocal | F1 | 1 | 1342.897 | -27.649 | -2.017 | 0.969 | 5 | 1 | 6 |
| LightGBMDirectNationalLocal | F2 | 1 | 1409.037 | 38.491 | 2.808 | 0.966 | 2 | 4 | 6 |
| LightGBMDirectNationalLocal | F3 | 1 | 1399.356 | 28.810 | 2.102 | 0.966 | 3 | 3 | 6 |
| LightGBMDirectNationalLocal | F0 | 3 | 2819.738 | 0.000 | 0.000 | 0.898 | 0 | 0 | 4 |
| LightGBMDirectNationalLocal | F1 | 3 | 2870.138 | 50.400 | 1.787 | 0.895 | 0 | 4 | 4 |
| LightGBMDirectNationalLocal | F2 | 3 | 2874.910 | 55.172 | 1.957 | 0.895 | 1 | 3 | 4 |
| LightGBMDirectNationalLocal | F3 | 3 | 2919.530 | 99.792 | 3.539 | 0.891 | 1 | 3 | 4 |
| LightGBMDirectNationalLocal | F0 | 6 | 5525.335 | 0.000 | 0.000 | 0.616 | 0 | 0 | 1 |
| LightGBMDirectNationalLocal | F1 | 6 | 5525.335 | 0.000 | 0.000 | 0.616 | 0 | 0 | 1 |
| LightGBMDirectNationalLocal | F2 | 6 | 5433.189 | -92.146 | -1.668 | 0.637 | 1 | 0 | 1 |
| LightGBMDirectNationalLocal | F3 | 6 | 5433.189 | -92.146 | -1.668 | 0.637 | 1 | 0 | 1 |
| LightGBMDirect | F0 | 1 | 2733.299 | 0.000 | 0.000 | 0.832 | 0 | 0 | 6 |
| LightGBMDirect | F1 | 1 | 2416.468 | -316.831 | -11.592 | 0.855 | 5 | 1 | 6 |
| LightGBMDirect | F2 | 1 | 2578.028 | -155.271 | -5.681 | 0.842 | 5 | 1 | 6 |
| LightGBMDirect | F3 | 1 | 2511.807 | -221.492 | -8.103 | 0.851 | 5 | 1 | 6 |
| LightGBMDirect | F0 | 3 | 2880.023 | 0.000 | 0.000 | 0.872 | 0 | 0 | 4 |
| LightGBMDirect | F1 | 3 | 2690.013 | -190.010 | -6.598 | 0.888 | 4 | 0 | 4 |
| LightGBMDirect | F2 | 3 | 2796.527 | -83.495 | -2.899 | 0.878 | 2 | 2 | 4 |
| LightGBMDirect | F3 | 3 | 2770.556 | -109.467 | -3.801 | 0.882 | 1 | 3 | 4 |
| LightGBMDirect | F0 | 6 | 6378.066 | 0.000 | 0.000 | 0.555 | 0 | 0 | 1 |
| LightGBMDirect | F1 | 6 | 6378.066 | 0.000 | 0.000 | 0.555 | 0 | 0 | 1 |
| LightGBMDirect | F2 | 6 | 5602.292 | -775.775 | -12.163 | 0.643 | 1 | 0 | 1 |
| LightGBMDirect | F3 | 6 | 5602.292 | -775.775 | -12.163 | 0.643 | 1 | 0 | 1 |

## Holdout — уже просмотренный

| model | variant | horizon | mae_macro | delta_mae_macro_vs_F0 | relative_delta_mae_macro_pct_vs_F0 | r2_pooled | n_origins_improved | n_origins_worsened | n_origins |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LightGBMDirectNationalLocal | F0 | 1 | 799.550 | 0.000 | 0.000 | 0.991 | 0 | 0 | 6 |
| LightGBMDirectNationalLocal | F1 | 1 | 817.862 | 18.312 | 2.290 | 0.990 | 2 | 4 | 6 |
| LightGBMDirectNationalLocal | F2 | 1 | 803.192 | 3.641 | 0.455 | 0.990 | 2 | 4 | 6 |
| LightGBMDirectNationalLocal | F3 | 1 | 791.627 | -7.923 | -0.991 | 0.990 | 4 | 2 | 6 |
| LightGBMDirectNationalLocal | F0 | 3 | 1212.726 | 0.000 | 0.000 | 0.972 | 0 | 0 | 6 |
| LightGBMDirectNationalLocal | F1 | 3 | 1254.425 | 41.699 | 3.438 | 0.971 | 2 | 4 | 6 |
| LightGBMDirectNationalLocal | F2 | 3 | 1225.266 | 12.540 | 1.034 | 0.971 | 2 | 4 | 6 |
| LightGBMDirectNationalLocal | F3 | 3 | 1259.966 | 47.240 | 3.895 | 0.970 | 1 | 5 | 6 |
| LightGBMDirectNationalLocal | F0 | 6 | 1897.219 | 0.000 | 0.000 | 0.942 | 0 | 0 | 6 |
| LightGBMDirectNationalLocal | F1 | 6 | 1891.531 | -5.689 | -0.300 | 0.942 | 2 | 1 | 6 |
| LightGBMDirectNationalLocal | F2 | 6 | 1857.112 | -40.108 | -2.114 | 0.944 | 4 | 2 | 6 |
| LightGBMDirectNationalLocal | F3 | 6 | 1863.474 | -33.745 | -1.779 | 0.944 | 4 | 2 | 6 |
| LightGBMDirect | F0 | 1 | 1003.280 | 0.000 | 0.000 | 0.985 | 0 | 0 | 6 |
| LightGBMDirect | F1 | 1 | 1078.374 | 75.093 | 7.485 | 0.982 | 5 | 1 | 6 |
| LightGBMDirect | F2 | 1 | 1616.133 | 612.853 | 61.085 | 0.954 | 1 | 5 | 6 |
| LightGBMDirect | F3 | 1 | 1634.663 | 631.382 | 62.932 | 0.960 | 1 | 5 | 6 |
| LightGBMDirect | F0 | 3 | 1449.820 | 0.000 | 0.000 | 0.962 | 0 | 0 | 6 |
| LightGBMDirect | F1 | 3 | 1895.430 | 445.609 | 30.735 | 0.940 | 4 | 2 | 6 |
| LightGBMDirect | F2 | 3 | 1468.644 | 18.823 | 1.298 | 0.962 | 3 | 3 | 6 |
| LightGBMDirect | F3 | 3 | 2027.480 | 577.660 | 39.844 | 0.932 | 2 | 4 | 6 |
| LightGBMDirect | F0 | 6 | 2463.212 | 0.000 | 0.000 | 0.904 | 0 | 0 | 6 |
| LightGBMDirect | F1 | 6 | 2470.929 | 7.717 | 0.313 | 0.904 | 2 | 2 | 6 |
| LightGBMDirect | F2 | 6 | 3506.088 | 1042.876 | 42.338 | 0.788 | 2 | 4 | 6 |
| LightGBMDirect | F3 | 6 | 3513.649 | 1050.437 | 42.645 | 0.787 | 2 | 4 | 6 |

Отрицательная delta означает уменьшение ошибки; relative delta приведена в процентах. Macro MAE — средняя MAE по МО, MAE micro и pooled R² сохранены рядом; их определения не изменены. Последующий holdout не новый blind test.

## Компактное F0/F1/F2/F3 сравнение

| model | variant | horizon | validation_MAE | delta_validation_vs_F0 | holdout_MAE | delta_holdout_vs_F0 | n_origins_improved | n_origins_worsened | median_holdout_origin_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LightGBMDirectNationalLocal | F0 | 1 | 1370.546 | 0.000 | 799.550 | 0.000 | 0 | 0 | 0.000 |
| LightGBMDirectNationalLocal | F1 | 1 | 1342.897 | -27.649 | 817.862 | 18.312 | 2 | 4 | 3.275 |
| LightGBMDirectNationalLocal | F2 | 1 | 1409.037 | 38.491 | 803.192 | 3.641 | 2 | 4 | 3.008 |
| LightGBMDirectNationalLocal | F3 | 1 | 1399.356 | 28.810 | 791.627 | -7.923 | 4 | 2 | -9.657 |
| LightGBMDirectNationalLocal | F0 | 3 | 2819.738 | 0.000 | 1212.726 | 0.000 | 0 | 0 | 0.000 |
| LightGBMDirectNationalLocal | F1 | 3 | 2870.138 | 50.400 | 1254.425 | 41.699 | 2 | 4 | 24.675 |
| LightGBMDirectNationalLocal | F2 | 3 | 2874.910 | 55.172 | 1225.266 | 12.540 | 2 | 4 | 1.122 |
| LightGBMDirectNationalLocal | F3 | 3 | 2919.530 | 99.792 | 1259.966 | 47.240 | 1 | 5 | 33.509 |
| LightGBMDirectNationalLocal | F0 | 6 | 5525.335 | 0.000 | 1897.219 | 0.000 | 0 | 0 | 0.000 |
| LightGBMDirectNationalLocal | F1 | 6 | 5525.335 | 0.000 | 1891.531 | -5.689 | 2 | 1 | 0.000 |
| LightGBMDirectNationalLocal | F2 | 6 | 5433.189 | -92.146 | 1857.112 | -40.108 | 4 | 2 | -10.404 |
| LightGBMDirectNationalLocal | F3 | 6 | 5433.189 | -92.146 | 1863.474 | -33.745 | 4 | 2 | -11.462 |
| LightGBMDirect | F0 | 1 | 2733.299 | 0.000 | 1003.280 | 0.000 | 0 | 0 | 0.000 |
| LightGBMDirect | F1 | 1 | 2416.468 | -316.831 | 1078.374 | 75.093 | 5 | 1 | -12.958 |
| LightGBMDirect | F2 | 1 | 2578.028 | -155.271 | 1616.133 | 612.853 | 1 | 5 | 38.725 |
| LightGBMDirect | F3 | 1 | 2511.807 | -221.492 | 1634.663 | 631.382 | 1 | 5 | 52.015 |
| LightGBMDirect | F0 | 3 | 2880.023 | 0.000 | 1449.820 | 0.000 | 0 | 0 | 0.000 |
| LightGBMDirect | F1 | 3 | 2690.013 | -190.010 | 1895.430 | 445.609 | 4 | 2 | -6.066 |
| LightGBMDirect | F2 | 3 | 2796.527 | -83.495 | 1468.644 | 18.823 | 3 | 3 | 7.591 |
| LightGBMDirect | F3 | 3 | 2770.556 | -109.467 | 2027.480 | 577.660 | 2 | 4 | 51.540 |
| LightGBMDirect | F0 | 6 | 6378.066 | 0.000 | 2463.212 | 0.000 | 0 | 0 | 0.000 |
| LightGBMDirect | F1 | 6 | 6378.066 | 0.000 | 2470.929 | 7.717 | 2 | 2 | 0.000 |
| LightGBMDirect | F2 | 6 | 5602.292 | -775.775 | 3506.088 | 1042.876 | 2 | 4 | 60.319 |
| LightGBMDirect | F3 | 6 | 5602.292 | -775.775 | 3513.649 | 1050.437 | 2 | 4 | 61.592 |

## Origin consistency и концентрация

На одной origin macro MAE равна среднему absolute error по доступным МО. Origin deltas считаются на одинаковых cases; временные blocks — origins, а не 2190 независимых финансовых наблюдений. Mean/median делtas не обязаны точно совпадать с aggregate macro delta при неодинаковом coverage.

| model | variant | split | horizon | n_origins_improved | n_origins_worsened | median_delta_mae_macro | mean_delta_mae_macro | best_origin | best_origin_delta | worst_origin | worst_origin_delta | leave_best_origin_out_mean_delta | concentration_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LightGBMDirect | F1 | holdout | 1 | 5 | 1 | -12.958 | 75.093 | 2024-06-30 | -51.506 | 2024-10-31 | 554.524 | 100.413 | FAIL |
| LightGBMDirect | F1 | holdout | 3 | 4 | 2 | -6.066 | 445.609 | 2024-06-30 | -203.535 | 2024-09-30 | 2827.498 | 575.438 | FAIL |
| LightGBMDirect | F1 | holdout | 6 | 2 | 2 | 0.000 | 7.717 | 2024-05-31 | -29.990 | 2024-01-31 | 68.996 | 15.259 | FAIL |
| LightGBMDirect | F1 | validation | 1 | 5 | 1 | -168.930 | -316.831 | 2024-01-31 | -1149.641 | 2024-02-29 | 126.151 | -150.269 | PASS |
| LightGBMDirect | F1 | validation | 3 | 4 | 0 | -172.355 | -190.010 | 2024-01-31 | -376.534 | 2024-03-31 | -38.796 | -127.835 | PASS |
| LightGBMDirect | F1 | validation | 6 | 0 | 0 | 0.000 | 0.000 | 2023-12-31 | 0.000 | 2023-12-31 | 0.000 | — | SINGLE ORIGIN UNDETERMINED |
| LightGBMDirect | F2 | holdout | 1 | 1 | 5 | 38.725 | 612.853 | 2024-10-31 | -266.940 | 2024-11-30 | 3786.789 | 788.812 | FAIL |
| LightGBMDirect | F2 | holdout | 3 | 3 | 3 | 7.591 | 18.823 | 2024-06-30 | -85.240 | 2024-08-31 | 159.893 | 39.636 | FAIL |
| LightGBMDirect | F2 | holdout | 6 | 2 | 4 | 60.319 | 1042.876 | 2024-04-30 | -91.720 | 2024-01-31 | 5813.241 | 1269.795 | FAIL |
| LightGBMDirect | F2 | validation | 1 | 5 | 1 | -157.698 | -155.271 | 2024-01-31 | -2814.959 | 2024-02-29 | 2488.792 | 376.666 | FAIL |
| LightGBMDirect | F2 | validation | 3 | 2 | 2 | -59.911 | -83.495 | 2024-02-29 | -275.066 | 2024-03-31 | 60.907 | -19.638 | PASS |
| LightGBMDirect | F2 | validation | 6 | 1 | 0 | -775.775 | -775.775 | 2023-12-31 | -775.775 | 2023-12-31 | -775.775 | — | SINGLE ORIGIN UNDETERMINED |
| LightGBMDirect | F3 | holdout | 1 | 1 | 5 | 52.015 | 631.382 | 2024-09-30 | -45.079 | 2024-11-30 | 2851.133 | 766.675 | FAIL |
| LightGBMDirect | F3 | holdout | 3 | 2 | 4 | 51.540 | 577.660 | 2024-06-30 | -146.335 | 2024-09-30 | 3455.646 | 722.459 | FAIL |
| LightGBMDirect | F3 | holdout | 6 | 2 | 4 | 61.592 | 1050.437 | 2024-03-31 | -85.391 | 2024-01-31 | 5813.241 | 1277.603 | FAIL |
| LightGBMDirect | F3 | validation | 1 | 5 | 1 | -191.937 | -221.492 | 2024-01-31 | -2866.684 | 2024-02-29 | 2333.127 | 307.546 | FAIL |
| LightGBMDirect | F3 | validation | 3 | 1 | 3 | 23.724 | -109.467 | 2024-02-29 | -591.585 | 2024-03-31 | 106.270 | 51.239 | FAIL |
| LightGBMDirect | F3 | validation | 6 | 1 | 0 | -775.775 | -775.775 | 2023-12-31 | -775.775 | 2023-12-31 | -775.775 | — | SINGLE ORIGIN UNDETERMINED |
| LightGBMDirectNationalLocal | F1 | holdout | 1 | 2 | 4 | 3.275 | 18.312 | 2024-07-31 | -11.408 | 2024-11-30 | 71.682 | 24.256 | FAIL |
| LightGBMDirectNationalLocal | F1 | holdout | 3 | 2 | 4 | 24.675 | 41.699 | 2024-07-31 | -21.161 | 2024-05-31 | 187.346 | 54.270 | FAIL |
| LightGBMDirectNationalLocal | F1 | holdout | 6 | 2 | 1 | 0.000 | -5.689 | 2024-06-30 | -28.185 | 2024-04-30 | 13.382 | -1.189 | PASS |
| LightGBMDirectNationalLocal | F1 | validation | 1 | 5 | 1 | -25.533 | -27.649 | 2024-01-31 | -54.795 | 2023-12-31 | 0.406 | -22.220 | PASS |
| LightGBMDirectNationalLocal | F1 | validation | 3 | 0 | 4 | 49.324 | 50.400 | 2024-02-29 | 3.045 | 2024-03-31 | 99.907 | 66.185 | FAIL |
| LightGBMDirectNationalLocal | F1 | validation | 6 | 0 | 0 | 0.000 | 0.000 | 2023-12-31 | 0.000 | 2023-12-31 | 0.000 | — | SINGLE ORIGIN UNDETERMINED |
| LightGBMDirectNationalLocal | F2 | holdout | 1 | 2 | 4 | 3.008 | 3.641 | 2024-09-30 | -71.027 | 2024-11-30 | 68.722 | 18.575 | FAIL |
| LightGBMDirectNationalLocal | F2 | holdout | 3 | 2 | 4 | 1.122 | 12.540 | 2024-05-31 | -14.799 | 2024-04-30 | 81.009 | 18.008 | FAIL |
| LightGBMDirectNationalLocal | F2 | holdout | 6 | 4 | 2 | -10.404 | -40.108 | 2024-02-29 | -167.909 | 2024-01-31 | 58.468 | -14.547 | PASS |
| LightGBMDirectNationalLocal | F2 | validation | 1 | 2 | 4 | 42.402 | 38.491 | 2024-01-31 | -74.648 | 2024-04-30 | 132.358 | 61.119 | FAIL |
| LightGBMDirectNationalLocal | F2 | validation | 3 | 1 | 3 | 35.657 | 55.172 | 2024-01-31 | -106.813 | 2024-03-31 | 256.185 | 109.166 | FAIL |
| LightGBMDirectNationalLocal | F2 | validation | 6 | 1 | 0 | -92.146 | -92.146 | 2023-12-31 | -92.146 | 2023-12-31 | -92.146 | — | SINGLE ORIGIN UNDETERMINED |
| LightGBMDirectNationalLocal | F3 | holdout | 1 | 4 | 2 | -9.657 | -7.923 | 2024-09-30 | -95.323 | 2024-11-30 | 78.675 | 9.557 | FAIL |
| LightGBMDirectNationalLocal | F3 | holdout | 3 | 1 | 5 | 33.509 | 47.240 | 2024-07-31 | -8.148 | 2024-05-31 | 146.168 | 58.317 | FAIL |
| LightGBMDirectNationalLocal | F3 | holdout | 6 | 4 | 2 | -11.462 | -33.745 | 2024-02-29 | -167.909 | 2024-01-31 | 56.074 | -6.913 | PASS |
| LightGBMDirectNationalLocal | F3 | validation | 1 | 3 | 3 | 18.327 | 28.810 | 2024-01-31 | -47.522 | 2024-03-31 | 147.079 | 44.077 | FAIL |
| LightGBMDirectNationalLocal | F3 | validation | 3 | 1 | 3 | 72.201 | 99.792 | 2024-01-31 | -22.072 | 2024-03-31 | 276.837 | 140.413 | FAIL |
| LightGBMDirectNationalLocal | F3 | validation | 6 | 1 | 0 | -92.146 | -92.146 | 2023-12-31 | -92.146 | 2023-12-31 | -92.146 | — | SINGLE ORIGIN UNDETERMINED |

Primary F3, каждый origin:

| split | horizon | forecast_origin | origin_mae_F0 | origin_mae | delta_mae_macro_vs_F0 |
| --- | --- | --- | --- | --- | --- |
| validation | 1 | 2023-12-31 | 1991.642 | 1980.930 | -10.711 |
| validation | 3 | 2023-12-31 | 3067.124 | 3197.690 | 130.565 |
| validation | 6 | 2023-12-31 | 5525.335 | 5433.189 | -92.146 |
| holdout | 6 | 2024-01-31 | 2933.497 | 2989.571 | 56.074 |
| validation | 1 | 2024-01-31 | 926.901 | 879.379 | -47.522 |
| validation | 3 | 2024-01-31 | 3100.934 | 3078.862 | -22.072 |
| holdout | 6 | 2024-02-29 | 2684.052 | 2516.143 | -167.909 |
| validation | 1 | 2024-02-29 | 942.098 | 896.495 | -45.603 |
| validation | 3 | 2024-02-29 | 2660.530 | 2674.367 | 13.837 |
| holdout | 6 | 2024-03-31 | 1223.886 | 1232.994 | 9.109 |
| validation | 1 | 2024-03-31 | 2105.792 | 2252.871 | 147.079 |
| validation | 3 | 2024-03-31 | 2450.366 | 2727.203 | 276.837 |
| holdout | 3 | 2024-04-30 | 1612.796 | 1669.045 | 56.249 |
| holdout | 6 | 2024-04-30 | 1372.775 | 1366.893 | -5.882 |
| validation | 1 | 2024-04-30 | 1365.313 | 1447.566 | 82.254 |
| holdout | 3 | 2024-05-31 | 1024.004 | 1170.172 | 146.168 |
| holdout | 6 | 2024-05-31 | 1146.470 | 1129.429 | -17.041 |
| validation | 1 | 2024-05-31 | 891.532 | 938.897 | 47.365 |
| holdout | 1 | 2024-06-30 | 952.482 | 911.872 | -40.610 |
| holdout | 3 | 2024-06-30 | 1023.724 | 1100.167 | 76.443 |
| holdout | 6 | 2024-06-30 | 2022.637 | 1945.814 | -76.823 |
| holdout | 1 | 2024-07-31 | 570.465 | 564.423 | -6.041 |
| holdout | 3 | 2024-07-31 | 1041.666 | 1033.518 | -8.148 |
| holdout | 1 | 2024-08-31 | 785.817 | 772.544 | -13.273 |
| holdout | 3 | 2024-08-31 | 1352.485 | 1354.444 | 1.958 |
| holdout | 1 | 2024-09-30 | 818.084 | 722.761 | -95.323 |
| holdout | 3 | 2024-09-30 | 1221.681 | 1232.448 | 10.768 |
| holdout | 1 | 2024-10-31 | 705.290 | 734.327 | 29.036 |
| holdout | 1 | 2024-11-30 | 965.162 | 1043.837 | 78.675 |

## F1/F2/F3 интерпретация

- LightGBMDirectNationalLocal / F1 (key rate alone): validation improvements 1/3, holdout 1/3; paired improving horizons []. Это описание фиксированной ablation, не выбор features по holdout.
- LightGBMDirectNationalLocal / F2 (FX alone): validation improvements 1/3, holdout 1/3; paired improving horizons [6]. Это описание фиксированной ablation, не выбор features по holdout.
- LightGBMDirectNationalLocal / F3 (combination): validation improvements 1/3, holdout 2/3; paired improving horizons [6]. Это описание фиксированной ablation, не выбор features по holdout.
- LightGBMDirect / F1 (key rate alone): validation improvements 2/3, holdout 0/3; paired improving horizons []. Это описание фиксированной ablation, не выбор features по holdout.
- LightGBMDirect / F2 (FX alone): validation improvements 3/3, holdout 0/3; paired improving horizons []. Это описание фиксированной ablation, не выбор features по holdout.
- LightGBMDirect / F3 (combination): validation improvements 3/3, holdout 0/3; paired improving horizons []. Это описание фиксированной ablation, не выбор features по holdout.

В данном фиксированном прогоне устойчивой пользы key rate alone для primary LN не подтверждено: F1/h1 даёт validation delta -2.02%, но holdout delta +2.29%; paired improving horizons отсутствуют.

FX alone даёт ограниченный прирост на h6: F2 validation delta -1.67%, holdout -2.11%. На h1/h3 обе части ухудшаются; h6 validation представлен одной origin.

Для combination F3 обе части улучшаются только на h6. На h1 validation delta +2.10%, а holdout -0.99%; после удаления лучшей holdout origin средняя delta становится +9.557 руб. — concentration FAIL. F3/h6 validation MAE совпадает с F2, а holdout MAE выше F2; добавка key rate к FX здесь не даёт дополнительного h6 выигрыша. Эффекты по горизонту, split и origins не согласованы по знаку.

Sensitivity L0 не подтверждает перенос прироста: F1 улучшает два validation горизонта, F2/F3 — три, но каждый финансовый вариант ухудшает holdout на всех h1/3/6. Это отдельная sensitivity, а не основание менять primary model.

Feature importance сохранена только descriptive; она не используется как доказательство usefulness или как success criterion.

## Temporal / integrity checks

E08b cache и decision ledger SHA проверены без сети; тот же builder материализует 23 monthly own origins Jan2023…Nov2024 из сохранённых training ranges и existing evaluation origins. На 12 frozen E08b origins числовые features совпадают в пределах 1e-12, source maxima — точно. Историческая row присоединяет features на собственной r, forecast row — на O. В каждом join effective/availability bounds≤own r≤O проверяются до learner creation; target label availability≤O также проверяется исходным E05d validator. Нет future fill, row dropping или new source research.

Проверено 4990452 financial training rows и 11382 forecast rows F1–F3; cutoff violations=0. Данные national: повторение по МО не увеличивает число независимых временных observations. Для LN actual historical N_target — только доступная метка; прогноз восстанавливается через неизменный forecast N_hat, без actual future N.

83 relevant synthetic tests passed; независимый пересчёт PASS для 15176 predictions в 8 arms, 10 сохранённых таблиц. Проверены 1631 artifact SHA. Доказательства записаны в outputs/e08c_checks/. F0 без financial covariates выполняет исходный E05d path; source reconstruction и extended matrix проверены до любых fits. Full unrelated model suite не запускался.

## h12 — DESCRIPTIVE ONLY

| model | variant | split | mae_macro | n_origins | n_native | n_fallback | evaluation_role |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LightGBMDirectNationalLocal | F0 | holdout | 4322.143 | 1 | 0 | 63 | DESCRIPTIVE ONLY |
| LightGBMDirectNationalLocal | F1 | holdout | 4322.143 | 1 | 0 | 63 | DESCRIPTIVE ONLY |
| LightGBMDirectNationalLocal | F2 | holdout | 4322.143 | 1 | 0 | 63 | DESCRIPTIVE ONLY |
| LightGBMDirectNationalLocal | F3 | holdout | 4322.143 | 1 | 0 | 63 | DESCRIPTIVE ONLY |
| LightGBMDirect | F0 | holdout | 4322.143 | 1 | 0 | 63 | DESCRIPTIVE ONLY |
| LightGBMDirect | F1 | holdout | 4322.143 | 1 | 0 | 63 | DESCRIPTIVE ONLY |
| LightGBMDirect | F2 | holdout | 4322.143 | 1 | 0 | 63 | DESCRIPTIVE ONLY |
| LightGBMDirect | F3 | holdout | 4322.143 | 1 | 0 | 63 | DESCRIPTIVE ONLY |

h12 имеет единственную origin Dec2023 и ноль обучающих пар. Все восемь strategies используют прежний expense SeasonalNaive fallback; ни одного h12 fit, обученного годового результата или вклада в success criterion нет.

## Команды, runtime и ограничения

```powershell
.\.venv\Scripts\python.exe -B -X utf8 scripts/run_leading_financial_forecasting.py --config configs/e08c_leading_financial_forecasting.yaml --stage smoke
.\.venv\Scripts\python.exe -B -X utf8 scripts/run_leading_financial_forecasting.py --config configs/e08c_leading_financial_forecasting.yaml --stage f0
.\.venv\Scripts\python.exe -B -X utf8 scripts/run_leading_financial_forecasting.py --config configs/e08c_leading_financial_forecasting.yaml --stage ablation
```

Successful model fits: 232 из 232; runtime стадий по manifest 4249.700 s (таймер фиксируется после partition loop, до итоговой агрегации/записи отчёта). Smoke F0 March2024 h1 используется повторно в полном F0 gate без повторного fit. Runtime отдельных tasks и full native params сохранены; веса не публикуются.

Ограничения: 24 target months, L=0 остаётся неподтверждённой availability цели; target vintages неизвестны, FX/key archive trust условен, June2024 methodology boundary ограничивает сопоставимость FX volatility. Уже просмотренный holdout, немного независимых origins и weak coverage не позволяют делать вывод об экономическом причинном эффекте или generalisation на новое blind test. Real-EW feasibility gate не меняется. h6 validation с одной origin имеет отдельное ограничение устойчивости.

Результат не обновляет F1–F7 Source of Truth, README, final report или presentation. Numeric predictions/features/metrics/cache/report остаются в существующих ignored outputs и reports/results; index не меняется.

**NO STABLE FORECASTING UPLIFT** — вывод по заранее объявленному primary criterion, без выбора лучшего learner/variant после результата. Следующий шаг — отдельно обсудить этот результат; новые модели, sources или early-warning training этим запуском не разрешаются.

NO TUNING; NO NEW MODEL FAMILY; NO SOURCE OF TRUTH CHANGES; NO COMMIT/PUSH.
