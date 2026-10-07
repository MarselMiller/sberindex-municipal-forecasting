# E07c — synthetic early-warning benchmark

**Синтетический controlled benchmark. Результаты не являются оценкой способности предсказывать реальные экономические шоки.**

Контролируемый опыт выполнен на независимом TEST; генератор и параметры моделей не менялись по его результатам.

## 1. Генератор и задача

Задача: вероятность нового истинного structural level shift в O+1…O+k, k=1/3, по наблюдениям ≤O. Каждый ряд содержит 24 месячных наблюдения, случайный уровень, слабый тренд, годовую сезонность, неодинаковый шум и возможный устойчивый shift до конца ряда. Onset выбирается из zero-based indices [16, 20] (месяцы 17…21). Это генераторная истинная разметка; weak residual criterion E07a/E07b не переопределён и не используется как synthetic truth.

Генератор зафиксирован до TEST в `configs/early_warning_synthetic.yaml`, SHA256 `d7277fd317e64d543224ccb034f960edbe0a555203fbc8ddb10c90e97f6dde61`. Exact fractions округляются по заранее заданному правилу: event fraction 0.6, unanticipated fraction 0.25 среди событий, false precursor fraction 0.35 среди контролей. У предсказуемых событий за 1…3 месяца могут возникать стохастические slope/volatility/residual и external macro/news процессы; сила меняется, минимум один канал активен. Ложные эпизоды контролей имеют ту же распределённую силу/длительность, но shift за ними не следует. Нет признака countdown или прямой даты события.

Полная неизменная спецификация: `generator_config.json` и `config_resolved.yaml` в новом output_dir. Все значения y/каналов — синтетические условные единицы, не реальные расходы, публикации или макропрогнозы.

## 2. Независимые cohorts и метки

| cohort | series | events | false_precursor_controls |
| --- | --- | --- | --- |
| test | 300 | 180 | 42 |
| train | 600 | 360 | 84 |
| validation | 200 | 120 | 28 |

Cohort seeds: `{"train": {"size": 600, "seed": 420101}, "validation": {"size": 200, "seed": 420201}, "test": {"size": 300, "seed": 420301}}`. `series_seed=cohort_seed×100000+index`; IDs/seeds не пересекаются. Независимость обеспечивается между целыми рядами; одинаковая относительная календарная сетка не означает проверку переноса во времени на реальные vintages.

| cohort | is_anticipated | precursor_class | noise_class | events |
| --- | --- | --- | --- | --- |
| test | False | none | high | 12 |
| test | False | none | low | 33 |
| test | True | strong | high | 9 |
| test | True | strong | low | 55 |
| test | True | weak | high | 17 |
| test | True | weak | low | 54 |
| train | False | none | high | 20 |
| train | False | none | low | 70 |
| train | True | strong | high | 40 |
| train | True | strong | low | 85 |
| train | True | weak | high | 40 |
| train | True | weak | low | 105 |
| validation | False | none | high | 8 |
| validation | False | none | low | 22 |
| validation | True | strong | high | 10 |
| validation | True | strong | low | 33 |
| validation | True | weak | high | 14 |
| validation | True | weak | low | 33 |

Origins после 12 наблюдений. Label1: истинный onset в (O,O+k]; label0 только при полном конечном будущем окне; край/пропуски — unknown. Label availability O+k; дополнительное weak confirmation T+2 не требуется, поскольку истина задана генератором. Уже начавшийся regime исключён из target по oracle onset; это контролируемое допущение, а не доказанная причинная возможность такого исключения на реальных данных. Oracle metadata не поступает в модели.

| cohort | k | audit_rows | positives | negatives | unknown | active | monitored | monitored_positives | monitored_rows | monitored_positive_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | 1 | 3900 | 180 | 3420 | 300 | 1058 | 2722 | 180 | 2722 | 0.0661278 |
| test | 3 | 3900 | 540 | 2460 | 900 | 1058 | 2482 | 540 | 2482 | 0.217566 |
| train | 1 | 7800 | 360 | 6840 | 600 | 2143 | 5417 | 360 | 5417 | 0.0664574 |
| train | 3 | 7800 | 1080 | 4920 | 1800 | 2143 | 4937 | 1080 | 4937 | 0.218756 |
| validation | 1 | 2600 | 120 | 2280 | 200 | 745 | 1775 | 120 | 1775 | 0.0676056 |
| validation | 3 | 2600 | 360 | 1640 | 600 | 745 | 1615 | 360 | 1615 | 0.22291 |

## 3. Причинные признаки и модели

History: лаги, изменения, rolling statistics, local slope/volatility, прошлые causal SeasonalNaiveYoY errors и residual trend. Detector state: существующие CUSUM/EWMA/BOCPD классы на доступном prefix, causal warmup из трёх конечных ошибок, затем фиксированный center/scale; alarms не сбрасывают state. External: stochastic macro pressure/news intensity и trailing changes. Все группы перечислены в `feature_dictionary.csv`/`feature_groups.json`; onset/countdown/future/offline breakpoints исключены.

S0 — train positive rate; S1 — History; S2 — History+Detector; S3 — History+Detector+External. Filter all-missing/constant/exact duplicates, median и population standardization обучены только на TRAIN прежним `TrainOnlyPreprocessor` E07b. C=1, class_weight=None, seed42, max_iter=2000, без tuning. В установленном окружении sklearn отсутствует; backend — SciPy L-BFGS-B для той же бинарной logistic objective sum(logloss)+||w||²/(2C), intercept не штрафуется. Установок нет. Численная эквивалентность конкретной версии sklearn не заявляется; optimizer status и gradient diagnostics сохранены.

| model | k | backend | status | converged | iterations | training_rows | retained_features | optimizer_message |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S0 | 1 | analytic_constant_prior | fitted | True | 0 | 5417 | 0 | TRAIN positive rate |
| S1 | 1 | scipy | fitted | True | 108 | 5417 | 19 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| S2 | 1 | scipy | fitted | True | 164 | 5417 | 31 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| S3 | 1 | scipy | fitted | True | 192 | 5417 | 39 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| S0 | 3 | analytic_constant_prior | fitted | True | 0 | 4937 | 0 | TRAIN positive rate |
| S1 | 3 | scipy | fitted | True | 157 | 4937 | 19 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| S2 | 3 | scipy | fitted | True | 231 | 4937 | 31 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |
| S3 | 3 | scipy | fitted | True | 217 | 4937 | 39 | CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH |

## 4. Threshold только VALIDATION

Максимум validation row F1 при false monthly alerts ≤1 на 12 monitored CONTROL-months. Candidates — наблюдаемые validation probabilities, 0.5 и 1.0; равный F1 → более высокий threshold; при отсутствии допустимого threshold fallback0.5 с явным статусом. Контрольный бюджет не является гарантией на TEST. TEST labels не используются для обучения, фильтра, scaler/imputer или threshold.

| threshold | validation_f1 | false_control_alerts | false_alerts_per_12_control_months | constraint_met | cohort | model | k | admitted_cases | control_monitored_months | control_monitored_years | candidates_evaluated | candidates_meeting_constraint | fallback_used | fallback_reason | selection_rule | threshold_candidates | truth_definition |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0 | 0 | 0 | True | validation | S0 | 1 | 1775 | 960 | 80 | 3 | 2 | False | — | validation_row_f1_max_control_false_alert_budget_ties_higher_threshold | unique_validation_probabilities_plus_0.5_1.0 | E07c_independent_synthetic_generator_onset_not_real_shock_truth |
| 1 | 0 | 0 | 0 | True | validation | S0 | 3 | 1615 | 800 | 66.6667 | 3 | 2 | False | — | validation_row_f1_max_control_false_alert_budget_ties_higher_threshold | unique_validation_probabilities_plus_0.5_1.0 | E07c_independent_synthetic_generator_onset_not_real_shock_truth |
| 0.111144 | 0.137405 | 77 | 0.9625 | True | validation | S1 | 1 | 1775 | 960 | 80 | 1777 | 148 | False | — | validation_row_f1_max_control_false_alert_budget_ties_higher_threshold | unique_validation_probabilities_plus_0.5_1.0 | E07c_independent_synthetic_generator_onset_not_real_shock_truth |
| 0.358414 | 0.174274 | 64 | 0.96 | True | validation | S1 | 3 | 1615 | 800 | 66.6667 | 1617 | 128 | False | — | validation_row_f1_max_control_false_alert_budget_ties_higher_threshold | unique_validation_probabilities_plus_0.5_1.0 | E07c_independent_synthetic_generator_onset_not_real_shock_truth |
| 0.151533 | 0.176471 | 64 | 0.8 | True | validation | S2 | 1 | 1775 | 960 | 80 | 1777 | 147 | False | — | validation_row_f1_max_control_false_alert_budget_ties_higher_threshold | unique_validation_probabilities_plus_0.5_1.0 | E07c_independent_synthetic_generator_onset_not_real_shock_truth |
| 0.399352 | 0.22449 | 65 | 0.975 | True | validation | S2 | 3 | 1615 | 800 | 66.6667 | 1617 | 133 | False | — | validation_row_f1_max_control_false_alert_budget_ties_higher_threshold | unique_validation_probabilities_plus_0.5_1.0 | E07c_independent_synthetic_generator_onset_not_real_shock_truth |
| 0.17483 | 0.433824 | 58 | 0.725 | True | validation | S3 | 1 | 1775 | 960 | 80 | 1777 | 187 | False | — | validation_row_f1_max_control_false_alert_budget_ties_higher_threshold | unique_validation_probabilities_plus_0.5_1.0 | E07c_independent_synthetic_generator_onset_not_real_shock_truth |
| 0.424239 | 0.44 | 65 | 0.975 | True | validation | S3 | 3 | 1615 | 800 | 66.6667 | 1617 | 193 | False | — | validation_row_f1_max_control_false_alert_budget_ties_higher_threshold | unique_validation_probabilities_plus_0.5_1.0 | E07c_independent_synthetic_generator_onset_not_real_shock_truth |

## 5. Row-level TEST и calibration

PR-AUC = average precision, step integral с grouped ties. ROC-AUC дополнительна; Brier/logloss proper scores. Precision/F1 при отсутствии alerts следуют явной zero-division convention; event precision/lead без denominator остаются undefined.

| model | k | cases | positives | negatives | pr_auc | roc_auc | brier_score | log_loss | precision | recall | f1 | threshold | calibration_ece |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S0 | 1 | 2722 | 180 | 2542 | 0.0661278 | 0.5 | 0.0617551 | 0.243507 | 0 | 0 | 0 | 1 | 0.000329602 |
| S0 | 3 | 2482 | 540 | 1942 | 0.217566 | 0.5 | 0.170233 | 0.523815 | 0 | 0 | 0 | 1 | 0.00118985 |
| S1 | 1 | 2722 | 180 | 2542 | 0.121066 | 0.701792 | 0.0595447 | 0.215663 | 0.164557 | 0.216667 | 0.18705 | 0.111144 | 0.00235562 |
| S1 | 3 | 2482 | 540 | 1942 | 0.327963 | 0.702985 | 0.150882 | 0.442021 | 0.364078 | 0.138889 | 0.201072 | 0.358414 | 0.0187498 |
| S2 | 1 | 2722 | 180 | 2542 | 0.155395 | 0.78819 | 0.0581188 | 0.20147 | 0.159574 | 0.166667 | 0.163043 | 0.151533 | 0.00794046 |
| S2 | 3 | 2482 | 540 | 1942 | 0.389735 | 0.756137 | 0.147448 | 0.433976 | 0.449761 | 0.174074 | 0.251001 | 0.399352 | 0.0135128 |
| S3 | 1 | 2722 | 180 | 2542 | 0.340535 | 0.854249 | 0.051822 | 0.180193 | 0.373786 | 0.427778 | 0.398964 | 0.17483 | 0.0120822 |
| S3 | 3 | 2482 | 540 | 1942 | 0.53794 | 0.804511 | 0.133648 | 0.40408 | 0.642612 | 0.346296 | 0.45006 | 0.424239 | 0.0299415 |

S3/k3 reliability table ниже; полные 10bins по всем моделям/горизонтам сохранены в `calibration.csv`. Calibration model отдельно не обучался.

| bin | lower | upper | count | mean_probability | positive_fraction |
| --- | --- | --- | --- | --- | --- |
| 1 | 0 | 0.1 | 777 | 0.0180986 | 0.019305 |
| 2 | 0.1 | 0.2 | 357 | 0.15471 | 0.162465 |
| 3 | 0.2 | 0.3 | 602 | 0.251825 | 0.232558 |
| 4 | 0.3 | 0.4 | 401 | 0.343026 | 0.274314 |
| 5 | 0.4 | 0.5 | 153 | 0.442856 | 0.555556 |
| 6 | 0.5 | 0.6 | 87 | 0.543516 | 0.655172 |
| 7 | 0.6 | 0.7 | 43 | 0.648343 | 0.651163 |
| 8 | 0.7 | 0.8 | 30 | 0.749748 | 0.766667 |
| 9 | 0.8 | 0.9 | 21 | 0.838313 | 0.714286 |
| 10 | 0.9 | 1 | 11 | 0.933312 | 0.818182 |

## 6. Event-level early warning и упреждение

Успех только за 1 месяц (k1) или 1…3 месяца (k3) ДО onset. Earliest alert и lead сохранены для каждого event. Каждый event считается успешным максимум один раз; repeated warnings записаны отдельно и не увеличивают recall/числитель precision. Alert precision = successes/(successes+false monthly alerts), повторы успешного предупреждения исключены. FAR = false monthly alerts/(fully-known at-risk monitored months/12); target exclusion сокращает exposure событийных рядов, не контролей. Alert в onset или позже никогда не считается early warning.

| model | k | eligible_events | warned_events | event_recall | alert_precision | median_lead_time_months | mean_lead_time_months | miss_rate | false_alert_count | false_alerts_per_12_monitored_months | monitored_cases | repeated_alert_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S0 | 1 | 180 | 0 | 0 | — | — | — | 1 | 0 | 0 | 2722 | 0 |
| S0 | 3 | 180 | 0 | 0 | — | — | — | 1 | 0 | 0 | 2482 | 0 |
| S1 | 1 | 180 | 39 | 0.216667 | 0.164557 | 1 | 1 | 0.783333 | 198 | 0.872888 | 2722 | 0 |
| S1 | 3 | 180 | 48 | 0.266667 | 0.268156 | 2 | 2.02083 | 0.733333 | 131 | 0.63336 | 2482 | 27 |
| S2 | 1 | 180 | 30 | 0.166667 | 0.159574 | 1 | 1 | 0.833333 | 158 | 0.696547 | 2722 | 0 |
| S2 | 3 | 180 | 72 | 0.4 | 0.385027 | 2 | 1.94444 | 0.6 | 115 | 0.556003 | 2482 | 22 |
| S3 | 1 | 180 | 77 | 0.427778 | 0.373786 | 1 | 1 | 0.572222 | 129 | 0.568699 | 2722 | 0 |
| S3 | 3 | 180 | 111 | 0.616667 | 0.516279 | 2 | 2.06306 | 0.383333 | 104 | 0.50282 | 2482 | 76 |

95% intervals: 500 bootstrap replicates целых TEST series_id с multiplicity, seed420401; все строки одного ряда остаются вместе. Это условная неопределённость данного fixed-generator опыта, без утверждения независимости месячных строк. Undefined replicates не заменены нулями.

| cohort | model | k | metric | estimate | lower | upper | ci_level | bootstrap_resamples | defined_resamples | series_count | bootstrap_seed | bootstrap_unit | undefined_resample_policy | truth_definition |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | S0 | 1 | event_recall | 0 | 0 | 0 | 0.95 | 500 | 500 | 300 | 420401 | whole_series_id_with_multiplicity | omit_undefined_metric_and_report_defined_resamples | E07c_independent_synthetic_generator_onset_not_real_shock_truth |
| test | S0 | 1 | alert_precision | — | — | — | 0.95 | 500 | 0 | 300 | 420401 | whole_series_id_with_multiplicity | omit_undefined_metric_and_report_defined_resamples | E07c_independent_synthetic_generator_onset_not_real_shock_truth |
| test | S0 | 1 | median_lead_time_months | — | — | — | 0.95 | 500 | 0 | 300 | 420401 | whole_series_id_with_multiplicity | omit_undefined_metric_and_report_defined_resamples | E07c_independent_synthetic_generator_onset_not_real_shock_truth |
| test | S0 | 1 | mean_lead_time_months | — | — | — | 0.95 | 500 | 0 | 300 | 420401 | whole_series_id_with_multiplicity | omit_undefined_metric_and_report_defined_resamples | E07c_independent_synthetic_generator_onset_not_real_shock_truth |
| test | S0 | 1 | false_alerts_per_12_monitored_months | 0 | 0 | 0 | 0.95 | 500 | 500 | 300 | 420401 | whole_series_id_with_multiplicity | omit_undefined_metric_and_report_defined_resamples | E07c_independent_synthetic_generator_onset_not_real_shock_truth |
| test | S0 | 1 | miss_rate | 1 | 1 | 1 | 0.95 | 500 | 500 | 300 | 420401 | whole_series_id_with_multiplicity | omit_undefined_metric_and_report_defined_resamples | E07c_independent_synthetic_generator_onset_not_real_shock_truth |
| test | S0 | 3 | event_recall | 0 | 0 | 0 | 0.95 | 500 | 500 | 300 | 420401 | whole_series_id_with_multiplicity | omit_undefined_metric_and_report_defined_resamples | E07c_independent_synthetic_generator_onset_not_real_shock_truth |
| test | S0 | 3 | alert_precision | — | — | — | 0.95 | 500 | 0 | 300 | 420401 | whole_series_id_with_multiplicity | omit_undefined_metric_and_report_defined_resamples | E07c_independent_synthetic_generator_onset_not_real_shock_truth |
| test | S0 | 3 | median_lead_time_months | — | — | — | 0.95 | 500 | 0 | 300 | 420401 | whole_series_id_with_multiplicity | omit_undefined_metric_and_report_defined_resamples | E07c_independent_synthetic_generator_onset_not_real_shock_truth |
| test | S0 | 3 | mean_lead_time_months | — | — | — | 0.95 | 500 | 0 | 300 | 420401 | whole_series_id_with_multiplicity | omit_undefined_metric_and_report_defined_resamples | E07c_independent_synthetic_generator_onset_not_real_shock_truth |
| test | S0 | 3 | false_alerts_per_12_monitored_months | 0 | 0 | 0 | 0.95 | 500 | 500 | 300 | 420401 | whole_series_id_with_multiplicity | omit_undefined_metric_and_report_defined_resamples | E07c_independent_synthetic_generator_onset_not_real_shock_truth |
| test | S0 | 3 | miss_rate | 1 | 1 | 1 | 0.95 | 500 | 500 | 300 | 420401 | whole_series_id_with_multiplicity | omit_undefined_metric_and_report_defined_resamples | E07c_independent_synthetic_generator_onset_not_real_shock_truth |
| test | S1 | 1 | event_recall | 0.216667 | 0.1582 | 0.276979 | 0.95 | 500 | 500 | 300 | 420401 | whole_series_id_with_multiplicity | omit_undefined_metric_and_report_defined_resamples | E07c_independent_synthetic_generator_onset_not_real_shock_truth |
| test | S1 | 1 | alert_precision | 0.164557 | 0.119001 | 0.219298 | 0.95 | 500 | 500 | 300 | 420401 | whole_series_id_with_multiplicity | omit_undefined_metric_and_report_defined_resamples | E07c_independent_synthetic_generator_onset_not_real_shock_truth |
| test | S1 | 1 | median_lead_time_months | 1 | 1 | 1 | 0.95 | 500 | 500 | 300 | 420401 | whole_series_id_with_multiplicity | omit_undefined_metric_and_report_defined_resamples | E07c_independent_synthetic_generator_onset_not_real_shock_truth |
| test | S1 | 1 | mean_lead_time_months | 1 | 1 | 1 | 0.95 | 500 | 500 | 300 | 420401 | whole_series_id_with_multiplicity | omit_undefined_metric_and_report_defined_resamples | E07c_independent_synthetic_generator_onset_not_real_shock_truth |
| test | S1 | 1 | false_alerts_per_12_monitored_months | 0.872888 | 0.682225 | 1.0787 | 0.95 | 500 | 500 | 300 | 420401 | whole_series_id_with_multiplicity | omit_undefined_metric_and_report_defined_resamples | E07c_independent_synthetic_generator_onset_not_real_shock_truth |
| test | S1 | 1 | miss_rate | 0.783333 | 0.723021 | 0.8418 | 0.95 | 500 | 500 | 300 | 420401 | whole_series_id_with_multiplicity | omit_undefined_metric_and_report_defined_resamples | E07c_independent_synthetic_generator_onset_not_real_shock_truth |
| test | S1 | 3 | event_recall | 0.266667 | 0.204543 | 0.331462 | 0.95 | 500 | 500 | 300 | 420401 | whole_series_id_with_multiplicity | omit_undefined_metric_and_report_defined_resamples | E07c_independent_synthetic_generator_onset_not_real_shock_truth |
| test | S1 | 3 | alert_precision | 0.268156 | 0.200814 | 0.354454 | 0.95 | 500 | 500 | 300 | 420401 | whole_series_id_with_multiplicity | omit_undefined_metric_and_report_defined_resamples | E07c_independent_synthetic_generator_onset_not_real_shock_truth |
| test | S1 | 3 | median_lead_time_months | 2 | 2 | 2 | 0.95 | 500 | 500 | 300 | 420401 | whole_series_id_with_multiplicity | omit_undefined_metric_and_report_defined_resamples | E07c_independent_synthetic_generator_onset_not_real_shock_truth |
| test | S1 | 3 | mean_lead_time_months | 2.02083 | 1.8201 | 2.22398 | 0.95 | 500 | 500 | 300 | 420401 | whole_series_id_with_multiplicity | omit_undefined_metric_and_report_defined_resamples | E07c_independent_synthetic_generator_onset_not_real_shock_truth |
| test | S1 | 3 | false_alerts_per_12_monitored_months | 0.63336 | 0.445361 | 0.817654 | 0.95 | 500 | 500 | 300 | 420401 | whole_series_id_with_multiplicity | omit_undefined_metric_and_report_defined_resamples | E07c_independent_synthetic_generator_onset_not_real_shock_truth |
| test | S1 | 3 | miss_rate | 0.733333 | 0.668538 | 0.795457 | 0.95 | 500 | 500 | 300 | 420401 | whole_series_id_with_multiplicity | omit_undefined_metric_and_report_defined_resamples | E07c_independent_synthetic_generator_onset_not_real_shock_truth |
| test | S2 | 1 | event_recall | 0.166667 | 0.10933 | 0.220061 | 0.95 | 500 | 500 | 300 | 420401 | whole_series_id_with_multiplicity | omit_undefined_metric_and_report_defined_resamples | E07c_independent_synthetic_generator_onset_not_real_shock_truth |
| test | S2 | 1 | alert_precision | 0.159574 | 0.102659 | 0.213069 | 0.95 | 500 | 500 | 300 | 420401 | whole_series_id_with_multiplicity | omit_undefined_metric_and_report_defined_resamples | E07c_independent_synthetic_generator_onset_not_real_shock_truth |
| test | S2 | 1 | median_lead_time_months | 1 | 1 | 1 | 0.95 | 500 | 500 | 300 | 420401 | whole_series_id_with_multiplicity | omit_undefined_metric_and_report_defined_resamples | E07c_independent_synthetic_generator_onset_not_real_shock_truth |
| test | S2 | 1 | mean_lead_time_months | 1 | 1 | 1 | 0.95 | 500 | 500 | 300 | 420401 | whole_series_id_with_multiplicity | omit_undefined_metric_and_report_defined_resamples | E07c_independent_synthetic_generator_onset_not_real_shock_truth |
| test | S2 | 1 | false_alerts_per_12_monitored_months | 0.696547 | 0.573467 | 0.816157 | 0.95 | 500 | 500 | 300 | 420401 | whole_series_id_with_multiplicity | omit_undefined_metric_and_report_defined_resamples | E07c_independent_synthetic_generator_onset_not_real_shock_truth |
| test | S2 | 1 | miss_rate | 0.833333 | 0.779939 | 0.89067 | 0.95 | 500 | 500 | 300 | 420401 | whole_series_id_with_multiplicity | omit_undefined_metric_and_report_defined_resamples | E07c_independent_synthetic_generator_onset_not_real_shock_truth |
| test | S2 | 3 | event_recall | 0.4 | 0.326556 | 0.469287 | 0.95 | 500 | 500 | 300 | 420401 | whole_series_id_with_multiplicity | omit_undefined_metric_and_report_defined_resamples | E07c_independent_synthetic_generator_onset_not_real_shock_truth |
| test | S2 | 3 | alert_precision | 0.385027 | 0.311976 | 0.469621 | 0.95 | 500 | 500 | 300 | 420401 | whole_series_id_with_multiplicity | omit_undefined_metric_and_report_defined_resamples | E07c_independent_synthetic_generator_onset_not_real_shock_truth |
| test | S2 | 3 | median_lead_time_months | 2 | 1.5 | 2 | 0.95 | 500 | 500 | 300 | 420401 | whole_series_id_with_multiplicity | omit_undefined_metric_and_report_defined_resamples | E07c_independent_synthetic_generator_onset_not_real_shock_truth |
| test | S2 | 3 | mean_lead_time_months | 1.94444 | 1.73644 | 2.1486 | 0.95 | 500 | 500 | 300 | 420401 | whole_series_id_with_multiplicity | omit_undefined_metric_and_report_defined_resamples | E07c_independent_synthetic_generator_onset_not_real_shock_truth |
| test | S2 | 3 | false_alerts_per_12_monitored_months | 0.556003 | 0.442719 | 0.677166 | 0.95 | 500 | 500 | 300 | 420401 | whole_series_id_with_multiplicity | omit_undefined_metric_and_report_defined_resamples | E07c_independent_synthetic_generator_onset_not_real_shock_truth |
| test | S2 | 3 | miss_rate | 0.6 | 0.530713 | 0.673444 | 0.95 | 500 | 500 | 300 | 420401 | whole_series_id_with_multiplicity | omit_undefined_metric_and_report_defined_resamples | E07c_independent_synthetic_generator_onset_not_real_shock_truth |
| test | S3 | 1 | event_recall | 0.427778 | 0.353992 | 0.502757 | 0.95 | 500 | 500 | 300 | 420401 | whole_series_id_with_multiplicity | omit_undefined_metric_and_report_defined_resamples | E07c_independent_synthetic_generator_onset_not_real_shock_truth |
| test | S3 | 1 | alert_precision | 0.373786 | 0.318268 | 0.430282 | 0.95 | 500 | 500 | 300 | 420401 | whole_series_id_with_multiplicity | omit_undefined_metric_and_report_defined_resamples | E07c_independent_synthetic_generator_onset_not_real_shock_truth |
| test | S3 | 1 | median_lead_time_months | 1 | 1 | 1 | 0.95 | 500 | 500 | 300 | 420401 | whole_series_id_with_multiplicity | omit_undefined_metric_and_report_defined_resamples | E07c_independent_synthetic_generator_onset_not_real_shock_truth |
| test | S3 | 1 | mean_lead_time_months | 1 | 1 | 1 | 0.95 | 500 | 500 | 300 | 420401 | whole_series_id_with_multiplicity | omit_undefined_metric_and_report_defined_resamples | E07c_independent_synthetic_generator_onset_not_real_shock_truth |
| test | S3 | 1 | false_alerts_per_12_monitored_months | 0.568699 | 0.468529 | 0.671112 | 0.95 | 500 | 500 | 300 | 420401 | whole_series_id_with_multiplicity | omit_undefined_metric_and_report_defined_resamples | E07c_independent_synthetic_generator_onset_not_real_shock_truth |
| test | S3 | 1 | miss_rate | 0.572222 | 0.497243 | 0.646008 | 0.95 | 500 | 500 | 300 | 420401 | whole_series_id_with_multiplicity | omit_undefined_metric_and_report_defined_resamples | E07c_independent_synthetic_generator_onset_not_real_shock_truth |
| test | S3 | 3 | event_recall | 0.616667 | 0.54611 | 0.689266 | 0.95 | 500 | 500 | 300 | 420401 | whole_series_id_with_multiplicity | omit_undefined_metric_and_report_defined_resamples | E07c_independent_synthetic_generator_onset_not_real_shock_truth |
| test | S3 | 3 | alert_precision | 0.516279 | 0.430465 | 0.59669 | 0.95 | 500 | 500 | 300 | 420401 | whole_series_id_with_multiplicity | omit_undefined_metric_and_report_defined_resamples | E07c_independent_synthetic_generator_onset_not_real_shock_truth |
| test | S3 | 3 | median_lead_time_months | 2 | 2 | 2 | 0.95 | 500 | 500 | 300 | 420401 | whole_series_id_with_multiplicity | omit_undefined_metric_and_report_defined_resamples | E07c_independent_synthetic_generator_onset_not_real_shock_truth |
| test | S3 | 3 | mean_lead_time_months | 2.06306 | 1.92281 | 2.20929 | 0.95 | 500 | 500 | 300 | 420401 | whole_series_id_with_multiplicity | omit_undefined_metric_and_report_defined_resamples | E07c_independent_synthetic_generator_onset_not_real_shock_truth |
| test | S3 | 3 | false_alerts_per_12_monitored_months | 0.50282 | 0.390447 | 0.639232 | 0.95 | 500 | 500 | 300 | 420401 | whole_series_id_with_multiplicity | omit_undefined_metric_and_report_defined_resamples | E07c_independent_synthetic_generator_onset_not_real_shock_truth |
| test | S3 | 3 | miss_rate | 0.383333 | 0.310734 | 0.45389 | 0.95 | 500 | 500 | 300 | 420401 | whole_series_id_with_multiplicity | omit_undefined_metric_and_report_defined_resamples | E07c_independent_synthetic_generator_onset_not_real_shock_truth |

## 7. Ablation S2−S1 и S3−S2

Все модели оцениваются на одинаковых случаях; threshold каждой модели выбран на VALIDATION по единому правилу. Разности показывают synthetic predictive utility; detector при уже богатой истории может не добавить информации. External channels специально созданы связанными с риском; это не доказательство пользы реальных новостей или реальных макропрогнозов.

| cohort | k | comparison | metric | earlier_model | later_model | earlier_value | later_value | difference | preferred_direction | interpretation | truth_definition |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | 1 | S2_minus_S1 | pr_auc | S1 | S2 | 0.121066 | 0.155395 | 0.0343282 | higher | descriptive_fixed_ablation_no_model_selection | E07c_independent_synthetic_generator_onset_not_real_shock_truth |
| test | 1 | S2_minus_S1 | roc_auc | S1 | S2 | 0.701792 | 0.78819 | 0.0863974 | higher | descriptive_fixed_ablation_no_model_selection | E07c_independent_synthetic_generator_onset_not_real_shock_truth |
| test | 1 | S2_minus_S1 | brier_score | S1 | S2 | 0.0595447 | 0.0581188 | -0.00142596 | lower | descriptive_fixed_ablation_no_model_selection | E07c_independent_synthetic_generator_onset_not_real_shock_truth |
| test | 1 | S2_minus_S1 | log_loss | S1 | S2 | 0.215663 | 0.20147 | -0.014193 | lower | descriptive_fixed_ablation_no_model_selection | E07c_independent_synthetic_generator_onset_not_real_shock_truth |
| test | 1 | S2_minus_S1 | f1 | S1 | S2 | 0.18705 | 0.163043 | -0.0240069 | higher | descriptive_fixed_ablation_no_model_selection | E07c_independent_synthetic_generator_onset_not_real_shock_truth |
| test | 1 | S2_minus_S1 | event_recall | S1 | S2 | 0.216667 | 0.166667 | -0.05 | higher | descriptive_fixed_ablation_no_model_selection | E07c_independent_synthetic_generator_onset_not_real_shock_truth |
| test | 1 | S2_minus_S1 | alert_precision | S1 | S2 | 0.164557 | 0.159574 | -0.00498249 | higher | descriptive_fixed_ablation_no_model_selection | E07c_independent_synthetic_generator_onset_not_real_shock_truth |
| test | 1 | S2_minus_S1 | false_alerts_per_12_monitored_months | S1 | S2 | 0.872888 | 0.696547 | -0.176341 | lower | descriptive_fixed_ablation_no_model_selection | E07c_independent_synthetic_generator_onset_not_real_shock_truth |
| test | 1 | S2_minus_S1 | median_lead_time_months | S1 | S2 | 1 | 1 | 0 | higher | descriptive_fixed_ablation_no_model_selection | E07c_independent_synthetic_generator_onset_not_real_shock_truth |
| test | 1 | S3_minus_S2 | pr_auc | S2 | S3 | 0.155395 | 0.340535 | 0.18514 | higher | descriptive_fixed_ablation_no_model_selection | E07c_independent_synthetic_generator_onset_not_real_shock_truth |
| test | 1 | S3_minus_S2 | roc_auc | S2 | S3 | 0.78819 | 0.854249 | 0.0660591 | higher | descriptive_fixed_ablation_no_model_selection | E07c_independent_synthetic_generator_onset_not_real_shock_truth |
| test | 1 | S3_minus_S2 | brier_score | S2 | S3 | 0.0581188 | 0.051822 | -0.00629677 | lower | descriptive_fixed_ablation_no_model_selection | E07c_independent_synthetic_generator_onset_not_real_shock_truth |
| test | 1 | S3_minus_S2 | log_loss | S2 | S3 | 0.20147 | 0.180193 | -0.0212763 | lower | descriptive_fixed_ablation_no_model_selection | E07c_independent_synthetic_generator_onset_not_real_shock_truth |
| test | 1 | S3_minus_S2 | f1 | S2 | S3 | 0.163043 | 0.398964 | 0.23592 | higher | descriptive_fixed_ablation_no_model_selection | E07c_independent_synthetic_generator_onset_not_real_shock_truth |
| test | 1 | S3_minus_S2 | event_recall | S2 | S3 | 0.166667 | 0.427778 | 0.261111 | higher | descriptive_fixed_ablation_no_model_selection | E07c_independent_synthetic_generator_onset_not_real_shock_truth |
| test | 1 | S3_minus_S2 | alert_precision | S2 | S3 | 0.159574 | 0.373786 | 0.214212 | higher | descriptive_fixed_ablation_no_model_selection | E07c_independent_synthetic_generator_onset_not_real_shock_truth |
| test | 1 | S3_minus_S2 | false_alerts_per_12_monitored_months | S2 | S3 | 0.696547 | 0.568699 | -0.127847 | lower | descriptive_fixed_ablation_no_model_selection | E07c_independent_synthetic_generator_onset_not_real_shock_truth |
| test | 1 | S3_minus_S2 | median_lead_time_months | S2 | S3 | 1 | 1 | 0 | higher | descriptive_fixed_ablation_no_model_selection | E07c_independent_synthetic_generator_onset_not_real_shock_truth |
| test | 3 | S2_minus_S1 | pr_auc | S1 | S2 | 0.327963 | 0.389735 | 0.0617712 | higher | descriptive_fixed_ablation_no_model_selection | E07c_independent_synthetic_generator_onset_not_real_shock_truth |
| test | 3 | S2_minus_S1 | roc_auc | S1 | S2 | 0.702985 | 0.756137 | 0.0531525 | higher | descriptive_fixed_ablation_no_model_selection | E07c_independent_synthetic_generator_onset_not_real_shock_truth |
| test | 3 | S2_minus_S1 | brier_score | S1 | S2 | 0.150882 | 0.147448 | -0.00343418 | lower | descriptive_fixed_ablation_no_model_selection | E07c_independent_synthetic_generator_onset_not_real_shock_truth |
| test | 3 | S2_minus_S1 | log_loss | S1 | S2 | 0.442021 | 0.433976 | -0.00804545 | lower | descriptive_fixed_ablation_no_model_selection | E07c_independent_synthetic_generator_onset_not_real_shock_truth |
| test | 3 | S2_minus_S1 | f1 | S1 | S2 | 0.201072 | 0.251001 | 0.0499289 | higher | descriptive_fixed_ablation_no_model_selection | E07c_independent_synthetic_generator_onset_not_real_shock_truth |
| test | 3 | S2_minus_S1 | event_recall | S1 | S2 | 0.266667 | 0.4 | 0.133333 | higher | descriptive_fixed_ablation_no_model_selection | E07c_independent_synthetic_generator_onset_not_real_shock_truth |
| test | 3 | S2_minus_S1 | alert_precision | S1 | S2 | 0.268156 | 0.385027 | 0.11687 | higher | descriptive_fixed_ablation_no_model_selection | E07c_independent_synthetic_generator_onset_not_real_shock_truth |
| test | 3 | S2_minus_S1 | false_alerts_per_12_monitored_months | S1 | S2 | 0.63336 | 0.556003 | -0.077357 | lower | descriptive_fixed_ablation_no_model_selection | E07c_independent_synthetic_generator_onset_not_real_shock_truth |
| test | 3 | S2_minus_S1 | median_lead_time_months | S1 | S2 | 2 | 2 | 0 | higher | descriptive_fixed_ablation_no_model_selection | E07c_independent_synthetic_generator_onset_not_real_shock_truth |
| test | 3 | S3_minus_S2 | pr_auc | S2 | S3 | 0.389735 | 0.53794 | 0.148206 | higher | descriptive_fixed_ablation_no_model_selection | E07c_independent_synthetic_generator_onset_not_real_shock_truth |
| test | 3 | S3_minus_S2 | roc_auc | S2 | S3 | 0.756137 | 0.804511 | 0.0483741 | higher | descriptive_fixed_ablation_no_model_selection | E07c_independent_synthetic_generator_onset_not_real_shock_truth |
| test | 3 | S3_minus_S2 | brier_score | S2 | S3 | 0.147448 | 0.133648 | -0.0137999 | lower | descriptive_fixed_ablation_no_model_selection | E07c_independent_synthetic_generator_onset_not_real_shock_truth |
| test | 3 | S3_minus_S2 | log_loss | S2 | S3 | 0.433976 | 0.40408 | -0.0298957 | lower | descriptive_fixed_ablation_no_model_selection | E07c_independent_synthetic_generator_onset_not_real_shock_truth |
| test | 3 | S3_minus_S2 | f1 | S2 | S3 | 0.251001 | 0.45006 | 0.199059 | higher | descriptive_fixed_ablation_no_model_selection | E07c_independent_synthetic_generator_onset_not_real_shock_truth |
| test | 3 | S3_minus_S2 | event_recall | S2 | S3 | 0.4 | 0.616667 | 0.216667 | higher | descriptive_fixed_ablation_no_model_selection | E07c_independent_synthetic_generator_onset_not_real_shock_truth |
| test | 3 | S3_minus_S2 | alert_precision | S2 | S3 | 0.385027 | 0.516279 | 0.131252 | higher | descriptive_fixed_ablation_no_model_selection | E07c_independent_synthetic_generator_onset_not_real_shock_truth |
| test | 3 | S3_minus_S2 | false_alerts_per_12_monitored_months | S2 | S3 | 0.556003 | 0.50282 | -0.0531829 | lower | descriptive_fixed_ablation_no_model_selection | E07c_independent_synthetic_generator_onset_not_real_shock_truth |
| test | 3 | S3_minus_S2 | median_lead_time_months | S2 | S3 | 2 | 2 | 0 | higher | descriptive_fixed_ablation_no_model_selection | E07c_independent_synthetic_generator_onset_not_real_shock_truth |

k=1: S2−S1 PR-AUC +0.0343, event recall -0.0500; S3−S2 PR-AUC +0.1851, event recall +0.2611. k=3: S2−S1 PR-AUC +0.0618, event recall +0.1333; S3−S2 PR-AUC +0.1482, event recall +0.2167. Это разности на сохранённых TEST predictions при отдельных validation thresholds; рост ranking metric не гарантирует рост recall при выбранном alert budget.

## 8. Anticipated / intentionally unanticipated

В TEST 135 событий с внедрённым предвестником из 180; остальные intentionally unanticipated. Для рядов без cue до onset отсутствует внедрённая сигнализирующая информация; возможны случайные/prior warnings. Доля 75% — ориентир recall для идеального распознавания только cue-bearing событий, а не строгий математический потолок любого alert rule. Разбиение по anticipated/unanticipated сохранено в scenario_metrics; paired controls сохраняют сопоставимость false-alert exposure.

## 9. Stress scenarios

Предзаданные срезы weak/strong precursor, high noise и controls with false precursor; все показаны, лучший scenario не выбирался. Oracle scenario metadata использована только после прогнозирования для оценки.

| scenario | model | k | metadata_series_count | monitored_series_count | eligible_events | warned_events | event_recall | alert_precision | median_lead_time_months | mean_lead_time_months | miss_rate | false_alert_count | false_alerts_per_12_monitored_months | pr_auc | f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base_all | S0 | 1 | 300 | 300 | 180 | 0 | 0 | — | — | — | 1 | 0 | 0 | 0.0661278 | 0 |
| weak_precursor | S0 | 1 | 94 | 94 | 71 | 0 | 0 | — | — | — | 1 | 0 | 0 | 0.0896465 | 0 |
| strong_precursor | S0 | 1 | 83 | 83 | 64 | 0 | 0 | — | — | — | 1 | 0 | 0 | 0.0945347 | 0 |
| high_noise | S0 | 1 | 63 | 63 | 38 | 0 | 0 | — | — | — | 1 | 0 | 0 | 0.0655172 | 0 |
| controls_false_precursor | S0 | 1 | 42 | 42 | 0 | 0 | — | — | — | — | — | 0 | 0 | — | 0 |
| anticipated_events | S0 | 1 | 255 | 255 | 135 | 0 | 0 | — | — | — | 1 | 0 | 0 | 0.0561331 | 0 |
| unanticipated_events | S0 | 1 | 165 | 165 | 45 | 0 | 0 | — | — | — | 1 | 0 | 0 | 0.0256118 | 0 |
| base_all | S0 | 3 | 300 | 300 | 180 | 0 | 0 | — | — | — | 1 | 0 | 0 | 0.217566 | 0 |
| weak_precursor | S0 | 3 | 94 | 94 | 71 | 0 | 0 | — | — | — | 1 | 0 | 0 | 0.285523 | 0 |
| strong_precursor | S0 | 3 | 83 | 83 | 64 | 0 | 0 | — | — | — | 1 | 0 | 0 | 0.300469 | 0 |
| high_noise | S0 | 3 | 63 | 63 | 38 | 0 | 0 | — | — | — | 1 | 0 | 0 | 0.215094 | 0 |
| controls_false_precursor | S0 | 3 | 42 | 42 | 0 | 0 | — | — | — | — | — | 0 | 0 | — | 0 |
| anticipated_events | S0 | 3 | 255 | 255 | 135 | 0 | 0 | — | — | — | 1 | 0 | 0 | 0.187067 | 0 |
| unanticipated_events | S0 | 3 | 165 | 165 | 45 | 0 | 0 | — | — | — | 1 | 0 | 0 | 0.0889914 | 0 |
| base_all | S1 | 1 | 300 | 300 | 180 | 39 | 0.216667 | 0.164557 | 1 | 1 | 0.783333 | 198 | 0.872888 | 0.121066 | 0.18705 |
| weak_precursor | S1 | 1 | 94 | 94 | 71 | 14 | 0.197183 | 0.175 | 1 | 1 | 0.802817 | 66 | 1 | 0.153675 | 0.18543 |
| strong_precursor | S1 | 1 | 83 | 83 | 64 | 14 | 0.21875 | 0.245614 | 1 | 1 | 0.78125 | 43 | 0.762186 | 0.204062 | 0.231405 |
| high_noise | S1 | 1 | 63 | 63 | 38 | 27 | 0.710526 | 0.144385 | 1 | 1 | 0.289474 | 160 | 3.31034 | 0.133315 | 0.24 |
| controls_false_precursor | S1 | 1 | 42 | 42 | 0 | 0 | — | 0 | — | — | — | 54 | 1.28571 | — | 0 |
| anticipated_events | S1 | 1 | 255 | 255 | 135 | 28 | 0.207407 | 0.135922 | 1 | 1 | 0.792593 | 178 | 0.88815 | 0.102799 | 0.164223 |
| unanticipated_events | S1 | 1 | 165 | 165 | 45 | 11 | 0.244444 | 0.0714286 | 1 | 1 | 0.755556 | 143 | 0.976665 | 0.046929 | 0.110553 |
| base_all | S1 | 3 | 300 | 300 | 180 | 48 | 0.266667 | 0.268156 | 2 | 2.02083 | 0.733333 | 131 | 0.63336 | 0.327963 | 0.201072 |
| weak_precursor | S1 | 3 | 94 | 94 | 71 | 18 | 0.253521 | 0.327273 | 2 | 2.11111 | 0.746479 | 37 | 0.595174 | 0.391684 | 0.188406 |
| strong_precursor | S1 | 3 | 83 | 83 | 64 | 17 | 0.265625 | 0.447368 | 2 | 1.88235 | 0.734375 | 21 | 0.394366 | 0.509232 | 0.210084 |
| high_noise | S1 | 3 | 63 | 63 | 38 | 33 | 0.868421 | 0.244444 | 2 | 2.27273 | 0.131579 | 102 | 2.30943 | 0.344102 | 0.423358 |
| controls_false_precursor | S1 | 3 | 42 | 42 | 0 | 0 | — | 0 | — | — | — | 38 | 1.08571 | — | 0 |
| anticipated_events | S1 | 3 | 255 | 255 | 135 | 35 | 0.259259 | 0.221519 | 2 | 2 | 0.740741 | 123 | 0.681755 | 0.273727 | 0.176166 |
| unanticipated_events | S1 | 3 | 165 | 165 | 45 | 13 | 0.288889 | 0.104839 | 2 | 2.07692 | 0.711111 | 111 | 0.878049 | 0.13997 | 0.177778 |
| base_all | S2 | 1 | 300 | 300 | 180 | 30 | 0.166667 | 0.159574 | 1 | 1 | 0.833333 | 158 | 0.696547 | 0.155395 | 0.163043 |
| weak_precursor | S2 | 1 | 94 | 94 | 71 | 9 | 0.126761 | 0.145161 | 1 | 1 | 0.873239 | 53 | 0.80303 | 0.182749 | 0.135338 |
| strong_precursor | S2 | 1 | 83 | 83 | 64 | 16 | 0.25 | 0.253968 | 1 | 1 | 0.75 | 47 | 0.833087 | 0.259128 | 0.251969 |
| high_noise | S2 | 1 | 63 | 63 | 38 | 13 | 0.342105 | 0.149425 | 1 | 1 | 0.657895 | 74 | 1.53103 | 0.171796 | 0.208 |
| controls_false_precursor | S2 | 1 | 42 | 42 | 0 | 0 | — | 0 | — | — | — | 47 | 1.11905 | — | 0 |
| anticipated_events | S2 | 1 | 255 | 255 | 135 | 25 | 0.185185 | 0.149701 | 1 | 1 | 0.814815 | 142 | 0.708524 | 0.136746 | 0.165563 |
| unanticipated_events | S2 | 1 | 165 | 165 | 45 | 5 | 0.111111 | 0.0454545 | 1 | 1 | 0.888889 | 105 | 0.717131 | 0.0557996 | 0.0645161 |
| base_all | S2 | 3 | 300 | 300 | 180 | 72 | 0.4 | 0.385027 | 2 | 1.94444 | 0.6 | 115 | 0.556003 | 0.389735 | 0.251001 |
| weak_precursor | S2 | 3 | 94 | 94 | 71 | 23 | 0.323944 | 0.396552 | 2 | 2.17391 | 0.676056 | 35 | 0.563003 | 0.468666 | 0.202899 |
| strong_precursor | S2 | 3 | 83 | 83 | 64 | 30 | 0.46875 | 0.491803 | 2 | 1.83333 | 0.53125 | 31 | 0.58216 | 0.544705 | 0.304183 |
| high_noise | S2 | 3 | 63 | 63 | 38 | 29 | 0.763158 | 0.337209 | 2 | 2.06897 | 0.236842 | 57 | 1.29057 | 0.378098 | 0.386792 |
| controls_false_precursor | S2 | 3 | 42 | 42 | 0 | 0 | — | 0 | — | — | — | 40 | 1.14286 | — | 0 |
| anticipated_events | S2 | 3 | 255 | 255 | 135 | 53 | 0.392593 | 0.323171 | 2 | 1.98113 | 0.607407 | 111 | 0.615242 | 0.339372 | 0.232877 |
| unanticipated_events | S2 | 3 | 165 | 165 | 45 | 19 | 0.422222 | 0.175926 | 2 | 1.84211 | 0.577778 | 89 | 0.704021 | 0.163335 | 0.208 |
| base_all | S3 | 1 | 300 | 300 | 180 | 77 | 0.427778 | 0.373786 | 1 | 1 | 0.572222 | 129 | 0.568699 | 0.340535 | 0.398964 |
| weak_precursor | S3 | 1 | 94 | 94 | 71 | 23 | 0.323944 | 0.359375 | 1 | 1 | 0.676056 | 41 | 0.621212 | 0.318031 | 0.340741 |
| strong_precursor | S3 | 1 | 83 | 83 | 64 | 50 | 0.78125 | 0.431034 | 1 | 1 | 0.21875 | 66 | 1.16987 | 0.518968 | 0.555556 |
| high_noise | S3 | 1 | 63 | 63 | 38 | 11 | 0.289474 | 0.2 | 1 | 1 | 0.710526 | 44 | 0.910345 | 0.238327 | 0.236559 |
| controls_false_precursor | S3 | 1 | 42 | 42 | 0 | 0 | — | 0 | — | — | — | 47 | 1.11905 | — | 0 |
| anticipated_events | S3 | 1 | 255 | 255 | 135 | 73 | 0.540741 | 0.370558 | 1 | 1 | 0.459259 | 124 | 0.618711 | 0.382803 | 0.439759 |
| unanticipated_events | S3 | 1 | 165 | 165 | 45 | 4 | 0.0888889 | 0.0547945 | 1 | 1 | 0.911111 | 69 | 0.471258 | 0.042845 | 0.0677966 |
| base_all | S3 | 3 | 300 | 300 | 180 | 111 | 0.616667 | 0.516279 | 2 | 2.06306 | 0.383333 | 104 | 0.50282 | 0.53794 | 0.45006 |
| weak_precursor | S3 | 3 | 94 | 94 | 71 | 40 | 0.56338 | 0.547945 | 2 | 2.05 | 0.43662 | 33 | 0.530831 | 0.56598 | 0.376238 |
| strong_precursor | S3 | 3 | 83 | 83 | 64 | 60 | 0.9375 | 0.625 | 2 | 2.11667 | 0.0625 | 36 | 0.676056 | 0.71441 | 0.666667 |
| high_noise | S3 | 3 | 63 | 63 | 38 | 23 | 0.605263 | 0.348485 | 3 | 2.21739 | 0.394737 | 43 | 0.973585 | 0.448387 | 0.397959 |
| controls_false_precursor | S3 | 3 | 42 | 42 | 0 | 0 | — | 0 | — | — | — | 56 | 1.6 | — | 0 |
| anticipated_events | S3 | 3 | 255 | 255 | 135 | 100 | 0.740741 | 0.492611 | 2 | 2.09 | 0.259259 | 103 | 0.570901 | 0.545369 | 0.503682 |
| unanticipated_events | S3 | 3 | 165 | 165 | 45 | 11 | 0.244444 | 0.107843 | 2 | 1.81818 | 0.755556 | 91 | 0.719842 | 0.130562 | 0.132231 |

На срезах возможны превышения validation budget: бюджет задавался на всех VALIDATION controls, не на каждом TEST stress subset. Отсутствие event у controls означает undefined event recall, а не нулевую способность предупреждения.

## 10. Фиксированно выбранные TEST examples

Правило зафиксировано заранее: S3/k3, затем k1 если тип отсутствует; первый series_id лексикографически, затем earliest origin. Успех/false alert/missed event берутся из сохранённого `examples.csv`; отсутствующий тип не создаётся вручную. На графиках показаны y, onset, наблюдаемые precursor channels, risk, validation threshold, alerts и lead.

**successful_warning — test_000002**, S3, k=3.

![successful_warning](../../outputs/early_warning_synthetic_v1/plots/first_successful_warning.png)
**false_alert — test_000000**, S3, k=3.

![false_alert](../../outputs/early_warning_synthetic_v1/plots/first_false_alert.png)
**missed_event — test_000006**, S3, k=3.

![missed_event](../../outputs/early_warning_synthetic_v1/plots/first_missed_event.png)

## 11. Выполненные команды, runtime и проверки

Command: `C:\Users\user\Desktop\Classes\sberindex_python_mvp\.venv\Scripts\python.exe -B -X utf8 scripts/run_early_warning_synthetic.py --stage full`. Experiment runtime 37.363s, peak process working set 0.177GiB; CPU, jobs1/BLAS1, лимит5GiB. Seed, versions, Git commit/dirty, config/code/artifact SHA сохранены в manifest. Это память процесса, не всей системы.

```json
{
  "final_validation": {
    "full_pytest_executions": 1,
    "command": "C:\\Users\\user\\Desktop\\Classes\\sberindex_python_mvp\\.venv\\Scripts\\python.exe -B -X utf8 -m pytest -p no:cacheprovider --basetemp=outputs/e07c_checks/pytest_full_once",
    "exit_code": 0,
    "passed": 1367,
    "warnings": 1,
    "summary": "1367 passed, 1 warning in 299.69s (0:04:59)",
    "runtime_seconds": 302.48946350000006,
    "stdout_path": "outputs\\e07c_checks\\pytest_full.txt",
    "stdout_sha256": "5b06828ed5f4457de175d60e2a27bdf535e6b3043384b1980291bd4dc29cfb5e",
    "code_test_config_unchanged_during_pytest": true,
    "changed_code": [],
    "code_test_config_files": 154,
    "targeted_tests": 120,
    "actual_classifier_backend": "existing SciPy; no installation or mock classifier",
    "independent_audit": 38034
  },
  "preservation_after": {
    "passed": true,
    "checked_files": 160,
    "changed": [],
    "config_unchanged": true,
    "git_commit_unchanged": true,
    "scope": "Tracked old code/config/tests/dependencies/README plus E07a/E07b reports and three audit metadata files; no full old-output-tree recheck"
  },
  "independent_audit": {
    "passed": true,
    "checks": 38034,
    "sections": {
      "labels": {
        "passed": true,
        "checks": 27
      },
      "models": {
        "passed": true,
        "checks": 119
      },
      "evaluation": {
        "passed": true,
        "checks": 37888
      }
    },
    "evaluation_runtime_seconds": 69.71887489999972,
    "audit_helper_sha256": {
      "outputs\\e07c_checks\\audit_evaluation.py": "b1d096d15c4d028c916ac2a5d03ccad6a2a72862af9d682d3dbd03ca51354d69",
      "outputs\\e07c_checks\\audit_labels.py": "5b7dc75d2e335446d32a82293821c67f68770d91ec9c059466e4e1fcfb5b2ded",
      "outputs\\e07c_checks\\audit_models.py": "67400a40f3490d6301a14ef793bb2de8a7184e846d9bf84f23d4d15d643a4395"
    }
  }
}
```

Full pytest выполняется один раз после smoke/full experiment; повторное report stage только читает сохранённые результаты и не повторяет fit/TEST evaluation. Сети, установки, commit/push и изменения старых E01–E07b outputs не выполнялись; большой старый artifact audit не повторялся.

## 12. Ограничения и завершение research phase

Синтетические процессы намеренно связаны с событием и содержат точно известный onset; эта информация и независимые длинные cohorts отсутствуют в реальной E07a/E07b истории. Размер отдельных рядов остаётся24, однако cross-series обучение даёт много независимых генераторных повторений. Seasonal baseline, короткий warmup, ложные cues, неравный шум, oracle active exclusion и fixed onset range ограничивают интерпретацию. Detector scores описывают уже наблюдённое изменение; их использование до будущего shift не превращает detection delay в early-warning lead. Никаких выводов о настоящих экономических шоках, реальных news/macro или superiority над Prophet из этого опыта нет.

E07c completed. Experimental research phase closed. Следующий этап — FINALIZATION ONLY: README, methodological report, results summary, presentation, clean reproduction check. Новые модели/источники/tuning не предлагаются.
