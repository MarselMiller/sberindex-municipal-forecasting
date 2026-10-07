# E07b — full-panel early warning feasibility

Результат: панель расширена до **2190 МО**, разметка и причинные признаки рассчитаны. Оба feasibility gate отказали; classifier fit и метрик раннего предупреждения нет. Это завершённый feasibility-опыт с предусмотренным отказом условной ветки, а не доказательство отсутствия предсказуемых экономических событий.

## 1. Протокол и реальные counts

Месячное значение категории «Все категории» по муниципальным образованиям; определение целевой величины E01 сохранено. Источник 2023-01…2024-12, forecast origins 2023-12…2024-11. Все municipality_id исходной панели сохраняются на каждой дате, включая ineligible audit rows. Eligibility определяется только prefix: не менее 12 доступных наблюдений и staleness не более одного календарного месяца. L=0 — прежнее неподтверждённое допущение о доступности расходов, не восстановленный архив vintages.

Новые full-panel h1 SeasonalNaiveYoY прогнозы E07b построены прежней функцией baseline_predict на каждом prefix и сохранены без обучения прогнозной модели; часть для прежних 64 МО сверена с E07a. Criterion E07a не меняется: четыре предыдущих календарных residuals, median center, scale=max(1.4826×MAD, 0.03×median|past prediction|, 1 рубль); три последовательных z=(e−center)/scale должны все быть ≥3 либо все ≤−3, все семь месяцев конечные. Merge gap≤2 и непрошедший recovery того же направления — продолжение; recovery требует двух месяцев direction_sign×z<1.5 относительно замороженной базы. Метка известна не ранее O+k+2 и доступности всех необходимых фактов/истории режима. Missing и правое цензурирование остаются unknown.

Реестр: **73** уникальных муниципальных слабых событий, **6** onset-дат, **71/2190 МО (3.242%)** хотя бы с одним событием. Counts событий, связанных с evaluable warning cases, ниже отличаются от полного реестра: события без полного окна не создают оцениваемую метку.

`scope=all` — вся панель; `eligible` — prefix eligibility; `at_risk_evaluable` дополнительно требует полную метку и отсутствие известного активного режима. `train`/`test` уже применяют календарь и доступность метки. `censored_or_missing` объединяет неполную прошлую/будущую информацию; отдельный right-censoring сохранён в feasibility.csv и label_counts.csv. `positive_event_onset_dates` считает onset-месяцы, `positive_warning_origin_dates` — даты выпуска предупреждений; муниципальные копии не увеличивают число дат.

| k | scope | cases | municipalities | forecast_origin_dates | fully_known | positives | negatives | censored_or_missing | unique_events | positive_event_onset_dates | positive_warning_origin_dates | known_active_cases | positive_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | all | 26280 | 2190 | 12 | 12183 | 73 | 12110 | 14097 | 73 | 6 | 6 | 229 | 0.00599196 |
| 1 | eligible | 24749 | 2119 | 12 | 12183 | 73 | 12110 | 12566 | 73 | 6 | 6 | 229 | 0.00599196 |
| 1 | at_risk_evaluable | 12069 | 2045 | 6 | 12069 | 71 | 11998 | 0 | 71 | 6 | 6 | 0 | 0.00588284 |
| 1 | train | 2023 | 2023 | 1 | 2023 | 14 | 2009 | 0 | 14 | 1 | 1 | 0 | 0.00692042 |
| 1 | test | 3974 | 1999 | 2 | 3974 | 4 | 3970 | 0 | 4 | 2 | 2 | 0 | 0.00100654 |
| 3 | all | 26280 | 2190 | 12 | 8101 | 156 | 7945 | 18179 | 73 | 6 | 4 | 229 | 0.0192569 |
| 3 | eligible | 24749 | 2119 | 12 | 8101 | 156 | 7945 | 16648 | 73 | 6 | 4 | 229 | 0.0192569 |
| 3 | at_risk_evaluable | 8087 | 2036 | 4 | 8087 | 154 | 7933 | 0 | 73 | 6 | 4 | 0 | 0.0190429 |
| 3 | train | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | — |
| 3 | test | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | — |

Статусы SeasonalNaiveYoY h1 сохранены; ineligible audit placeholders не являются прогнозами. Ошибку можно вычислить только при конечных прогнозе и target. Подробные counts по дате выпуска — `forecast_coverage.csv`.

| forecast_status | cases | finite_predictions | finite_targets | fallback_cases |
| --- | --- | --- | --- | --- |
| ineligible_no_forecast | 1531 | 0 | 0 | 0 |
| native | 24667 | 24667 | 24537 | 0 |
| seasonal_fallback | 82 | 82 | 75 | 82 |

## 2. Gate и временной split

Пороги записаны в новой YAML **до полного подсчёта**: train positives≥30, test positives≥10, положительных onset-дат train≥3/test≥2, отдельно для k=1/3. Hash исходной E07b YAML и неизменного E07a протокола сохранён в preservation_before.json и manifest. После просмотра counts/метрик пороги, окна и даты не изменялись.

| k | train_positives | test_positives | train_positive_onset_dates | test_positive_onset_dates | train_origin_dates | test_origin_dates | gate_pass | failure_reasons |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 14 | 4 | 1 | 2 | 1 | 2 | False | train_positives=14<30;test_positives=4<10;train_positive_onset_dates=1<3 |
| 3 | 0 | 0 | 0 | 0 | 0 | 0 | False | train_positives=0<30;test_positives=0<10;train_positive_onset_dates=0<3;test_positive_onset_dates=0<2 |

Исходная граница E07a сохранена: label-information cutoff июль 2024, test origins август–ноябрь 2024. Train допускает только полные eligible at-risk метки, известные к cutoff; test truth доступен лишь для последующей ретроспективной оценки. Random split не использован. Для k=3 исходный test пуст; ниже сохранены все календарные альтернативы без classifier metrics. Ни один вариант не проходит gate, поэтому замена границы не обоснована.

| cutoff_period | k | train_cases | test_cases | train_positives | test_positives | train_positive_onset_dates | test_positive_onset_dates | gate_pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2023-12 | 1 | 0 | 12069 | 0 | 71 | 0 | 6 | False |
| 2023-12 | 3 | 0 | 8087 | 0 | 154 | 0 | 6 | False |
| 2024-01 | 1 | 0 | 12069 | 0 | 71 | 0 | 6 | False |
| 2024-01 | 3 | 0 | 8087 | 0 | 154 | 0 | 6 | False |
| 2024-02 | 1 | 0 | 12069 | 0 | 71 | 0 | 6 | False |
| 2024-02 | 3 | 0 | 8087 | 0 | 154 | 0 | 6 | False |
| 2024-03 | 1 | 0 | 12069 | 0 | 71 | 0 | 6 | False |
| 2024-03 | 3 | 0 | 8087 | 0 | 154 | 0 | 6 | False |
| 2024-04 | 1 | 0 | 10046 | 0 | 57 | 0 | 5 | False |
| 2024-04 | 3 | 0 | 6069 | 0 | 94 | 0 | 5 | False |
| 2024-05 | 1 | 0 | 8023 | 0 | 30 | 0 | 4 | False |
| 2024-05 | 3 | 0 | 4047 | 0 | 41 | 0 | 4 | False |
| 2024-06 | 1 | 0 | 5995 | 0 | 11 | 0 | 3 | False |
| 2024-06 | 3 | 0 | 2020 | 0 | 11 | 0 | 3 | False |
| 2024-07 | 1 | 2023 | 3974 | 14 | 4 | 1 | 2 | False |
| 2024-07 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | False |
| 2024-08 | 1 | 4046 | 1979 | 41 | 2 | 2 | 1 | False |
| 2024-08 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | False |
| 2024-09 | 1 | 6074 | 0 | 60 | 0 | 3 | 0 | False |
| 2024-09 | 3 | 2018 | 0 | 60 | 0 | 3 | 0 | False |
| 2024-10 | 1 | 8095 | 0 | 67 | 0 | 4 | 0 | False |
| 2024-10 | 3 | 4040 | 0 | 113 | 0 | 4 | 0 | False |
| 2024-11 | 1 | 10090 | 0 | 69 | 0 | 5 | 0 | False |
| 2024-11 | 3 | 6067 | 0 | 143 | 0 | 5 | 0 | False |

Причина структурная: первое возможное residual наблюдение — январь 2024, первое возможное onset после четырёх прошлых residuals — май. Чтобы для k=1 иметь ≥3 train onset-месяца с подтверждением, cutoff должен быть не раньше сентября; чтобы иметь ≥2 последующих test onset-месяца до последнего подтверждаемого октября — не позже июля. Совместной границы нет. Для k=3 первая полностью доступная train метка появляется в сентябре (origin апрель+5), тогда как последний evaluable origin — июль (конец данных декабрь−5). Увеличение числа МО не удлиняет календарь.

## 3. Причинные активные состояния

| k | forecast_origin | cases | eligible_cases | known_active_cases | state_uncertain_cases | retrospective_inside_cases |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 2023-12-31 | 2190 | 2075 | 0 | 2190 | 0 |
| 1 | 2024-01-31 | 2190 | 2083 | 0 | 159 | 0 |
| 1 | 2024-02-29 | 2190 | 2044 | 0 | 151 | 0 |
| 1 | 2024-03-31 | 2190 | 2053 | 0 | 146 | 0 |
| 1 | 2024-04-30 | 2190 | 2054 | 0 | 139 | 0 |
| 1 | 2024-05-31 | 2190 | 2057 | 0 | 139 | 14 |
| 1 | 2024-06-30 | 2190 | 2061 | 0 | 142 | 41 |
| 1 | 2024-07-31 | 2190 | 2060 | 14 | 138 | 60 |
| 1 | 2024-08-31 | 2190 | 2063 | 41 | 137 | 66 |
| 1 | 2024-09-30 | 2190 | 2066 | 59 | 130 | 64 |
| 1 | 2024-10-31 | 2190 | 2066 | 60 | 131 | 57 |
| 1 | 2024-11-30 | 2190 | 2067 | 55 | 126 | 55 |
| 3 | 2023-12-31 | 2190 | 2075 | 0 | 2190 | 0 |
| 3 | 2024-01-31 | 2190 | 2083 | 0 | 159 | 0 |
| 3 | 2024-02-29 | 2190 | 2044 | 0 | 151 | 0 |
| 3 | 2024-03-31 | 2190 | 2053 | 0 | 146 | 0 |
| 3 | 2024-04-30 | 2190 | 2054 | 0 | 139 | 0 |
| 3 | 2024-05-31 | 2190 | 2057 | 0 | 139 | 14 |
| 3 | 2024-06-30 | 2190 | 2061 | 0 | 142 | 41 |
| 3 | 2024-07-31 | 2190 | 2060 | 14 | 138 | 60 |
| 3 | 2024-08-31 | 2190 | 2063 | 41 | 137 | 66 |
| 3 | 2024-09-30 | 2190 | 2066 | 59 | 130 | 64 |
| 3 | 2024-10-31 | 2190 | 2066 | 60 | 131 | 57 |
| 3 | 2024-11-30 | 2190 | 2067 | 55 | 126 | 55 |

`known_active_cases` — режим подтверждён и известен на O, только он исключается причинным фильтром. `retrospective_inside_cases` — аудит с доступом к полной истории; он не входит в признаки, eligibility или обучение. State uncertainty сохраняется отдельно. Counts представлены по k, поэтому сумма по двум k удваивает одни и те же MO/origin.

## 4. Признаки и coverage

Fixed groups: A history/lag/dynamics/rolling/residual/volatility, B prefix CUSUM/EWMA/BOCPD, C admitted annual macro forecasts E05, D historically admitted news E06/E07a. В числовой матрице 138 кандидатов (A40/B36/C10/D52 с имеющимися missing flags); три текстовых detector phase только диагностические. Offline PELT/BinSeg breakpoints запрещены. Новые rolling volatility требуют все три конечных календарных месяца, ddof=0, без заполнения пропусков.

| group | role | feature_columns | cells | finite_cells | finite_cell_coverage | observed_varying_columns | dependency_violation_records | missing_dependency_coverage_records | temporal_dependency_check_passed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | value | 20 | 525600 | 473889 | 0.901615 | 20 | 0 | 0 | True |
| A | missing_flag | 20 | 525600 | 525600 | 1 | 17 | 0 | 0 | True |
| B | value | 18 | 473040 | 354096 | 0.748554 | 18 | 0 | 0 | True |
| B | missing_flag | 18 | 473040 | 473040 | 1 | 12 | 0 | 0 | True |
| C | value | 8 | 210240 | 183960 | 0.875 | 4 | 0 | 0 | True |
| C | missing_flag | 2 | 52560 | 52560 | 1 | 0 | 0 | 0 | True |
| D | value | 26 | 683280 | 560640 | 0.820513 | 9 | 0 | 0 | True |
| D | missing_flag | 26 | 683280 | 683280 | 1 | 4 | 0 | 0 | True |

Coverage рассчитано по полной audit-панели, поэтому включает ineligible строки. `feature_coverage.csv` содержит каждую дату/признак, finite/nonmissing, число различных значений и dependency audit; lineage расходов/ошибок/макро записан пакетами на диск. Все зависимости проверяются по собственному O. Нулевая violation count отражает принятые правила timestamp, а не доказанную историческую публикацию расходов. Параметры E04 ранее выбраны и зафиксированы: prefix replay причинен при этих параметрах, но историческая доступность самого выбора гиперпараметров на O не подтверждена.

На train выполнен отдельный аудит удаления all-missing, constants и точных numeric duplicates; median/mean/scales записаны в `feature_filter_train.csv`. На k=3 train отсутствует, фильтр помечен `no_training_rows`. Классификаторы от этого аудита не запускались. При допустимом gate код применяет один train-list и train-median/scaler к test, C=1, class_weight=None, threshold=0.5; выбора по test нет.

| k | model | group | candidates | retained | all_missing | constant | exact_duplicate | no_training_rows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | B1 | A | 40 | 13 | 0 | 25 | 2 | 0 |
| 1 | B2 | A | 40 | 13 | 0 | 25 | 2 | 0 |
| 1 | B2 | B | 36 | 0 | 9 | 27 | 0 | 0 |
| 1 | B3 | A | 40 | 13 | 0 | 25 | 2 | 0 |
| 1 | B3 | B | 36 | 0 | 9 | 27 | 0 | 0 |
| 1 | B3 | C | 10 | 0 | 1 | 9 | 0 | 0 |
| 1 | B4 | A | 40 | 13 | 0 | 25 | 2 | 0 |
| 1 | B4 | B | 36 | 0 | 9 | 27 | 0 | 0 |
| 1 | B4 | C | 10 | 0 | 1 | 9 | 0 | 0 |
| 1 | B4 | D | 52 | 0 | 3 | 49 | 0 | 0 |
| 3 | B1 | A | 40 | 0 | 0 | 0 | 0 | 40 |
| 3 | B2 | A | 40 | 0 | 0 | 0 | 0 | 40 |
| 3 | B2 | B | 36 | 0 | 0 | 0 | 0 | 36 |
| 3 | B3 | A | 40 | 0 | 0 | 0 | 0 | 40 |
| 3 | B3 | B | 36 | 0 | 0 | 0 | 0 | 36 |
| 3 | B3 | C | 10 | 0 | 0 | 0 | 0 | 10 |
| 3 | B4 | A | 40 | 0 | 0 | 0 | 0 | 40 |
| 3 | B4 | B | 36 | 0 | 0 | 0 | 0 | 36 |
| 3 | B4 | C | 10 | 0 | 0 | 0 | 0 | 10 |
| 3 | B4 | D | 52 | 0 | 0 | 0 | 0 | 52 |

## 5. News и вклад групп

| k | scope | cases | forecast_origins | varying_value_columns | varying_missing_flag_columns | unique_temporal_vectors | maximum_spatial_full_vectors_per_origin | origins_with_news_information |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | all | 26280 | 12 | 9 | 4 | 7 | 1 | 12 |
| 1 | train | 2023 | 1 | 0 | 0 | 1 | 1 | 1 |
| 1 | test | 3974 | 2 | 6 | 3 | 2 | 1 | 2 |
| 3 | all | 26280 | 12 | 9 | 4 | 7 | 1 | 12 |
| 3 | train | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 3 | test | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

News counts по train/test относятся только к допустимым cases этого split. Реально varying значения и existing missing flags показаны раздельно. Temporal vectors дедуплицированы по значениям, независимо от числа MO-копий; пространственная вариативность отдельно показывает национальный характер. News information определяется наличием admitted документов в 90-дневном окне. Корпус ограничен: официальные key-rate cores и PDF под прежним archive-trust, исторических региональных/муниципальных публикаций нет; полная интенсивность новостей не восстановлена. Macro — доступные национальные годовые прогнозы, не причинный эффект, не региональная зарплата и не дефлирование цели.

**B2 vs B1 (детекторы), B3 vs B2 (macro), B4 vs B3 (news) не оценены.** Вклад ни одной группы не установлен. Даже при varying test-признаках нет допустимого обучения и оценки.

## 6. B0–B4 и event-level оценка

Для **k=1 и k=3 B0–B4 не обучались**: оба gate отказали до fit. Пустые файлы с заголовками `predictions.csv.gz`, `metrics_row.csv`, `metrics_event.csv`, `calibration.csv`, `model_coefficients.csv`, `alerts.csv` и `examples.csv` не содержат выдуманных результатов. Причины по k/model сохранены в `model_status.csv`. В основной .venv sklearn отсутствует; установки нет. Это дополнительное ограничение условной ветки, а основание отказа в данном опыте — достаточность данных.

Подготовленная условная оценка PR-AUC определяется как average precision (step integral с группировкой ties); дополнительно ROC-AUC, Brier, log loss, precision/recall/F1 и 10 заранее фиксированных calibration bins. Event matching — хронологический earliest alert за 1…k месяцев до onset, один event/alert максимум один match; повторы для уже предупреждённого события отдельно, после onset — не early warning. Denominator event recall — события, для которых существует хотя бы один полностью известный at-risk pre-onset monitored origin. Earliest/count warnings относятся к назначенным one-to-one alerts. False-alert exposure — число полностью известных at-risk monitored MO-months/12; ложные MO-month alerts считаются по отдельности. Event precision, recall, lead time и false alerts в этом запуске **не получены**.

## 7. Реальные примеры и ограничения weak labels

Успешные предупреждения, false alerts и missed events классификатора не оценивались: предупреждения не выпускались. Реестр слабых событий сохранён для человеческой проверки; его строки нельзя выдавать за примеры успешного early warning.

Weak truth — устойчивое изменение ошибки SeasonalNaiveYoY, не экспертная разметка экономического шока. Ошибки baseline, сезонность, пропуски и возврат к прежнему режиму могут создавать onset; противоположный знак может означать восстановление ошибки, а не отдельный экономический шок. Подтверждение T+2 требует будущего, поэтому onset/confirmation/warning origin различаются. Человеческое подтверждение E07a pending. Известные timestamps расходов и архив полноты новостей не восстановлены. Эти ограничения нельзя устранить большим числом муниципальных строк. Нового слепого теста здесь нет, преимущества над Prophet или способности предсказывать экономические шоки не установлены.

## 8. Выполненные команды и проверки

| stage | runtime_seconds | peak_working_set_gib | command |
| --- | --- | --- | --- |
| feasibility | 629.966 | 0.239365 | C:\Users\user\Desktop\Classes\sberindex_python_mvp\.venv\Scripts\python.exe scripts/run_early_warning_full_panel.py --stage feasibility |
| baselines-skipped | 0.0124192 | 0.0654984 | C:\Users\user\Desktop\Classes\sberindex_python_mvp\.venv\Scripts\python.exe -B -X utf8 scripts/run_early_warning_full_panel.py --stage baselines |
| features | 1513.99 | 0.214996 | C:\Users\user\Desktop\Classes\sberindex_python_mvp\.venv\Scripts\python.exe scripts/run_early_warning_full_panel.py --stage features |
| audit-report | 5.51337 | 0.183823 | C:\Users\user\Desktop\Classes\sberindex_python_mvp\.venv\Scripts\python.exe -B -X utf8 scripts/run_early_warning_full_panel.py --stage audit-report |
| audit-report | 5.96577 | 0.184029 | C:\Users\user\Desktop\Classes\sberindex_python_mvp\.venv\Scripts\python.exe -B -X utf8 scripts/run_early_warning_full_panel.py --stage audit-report |

Целевые синтетические тесты новых adapter/labels/features/models: 3+19+20+24 passed; технический smoke использует 5 реальных МО и не является синтетическим исследовательским экспериментом. Дополнительные audit unit tests и результат **одного полного pytest**: `{"full_pytest_executions": 1, "command": "C:\\Users\\user\\Desktop\\Classes\\sberindex_python_mvp\\.venv\\Scripts\\python.exe -B -X utf8 -m pytest -p no:cacheprovider --basetemp=outputs/e07b_checks/pytest_full_once", "exit_code": 0, "passed": 1247, "warnings": 1, "runtime_seconds": 273.5615921000008, "summary": "1247 passed, 1 warning in 270.73s (0:04:30)", "stdout_path": "outputs\\e07b_checks\\pytest_full.txt", "stdout_sha256": "98992bcbbd66b6a95e3d7d6076692ebce868e584fc9f941f51e78ba2b0d1463c", "code_test_config_unchanged_during_pytest": true, "changed_code": [], "targeted_tests": {"adapter": 3, "labels": 19, "features": 20, "models": 24, "audit": 14, "total": 80}, "classifier_unit_tests": "synthetic explicit mock; actual sklearn absent; no real classifier fit", "independent_label_checks": 118, "pytest_reported_runtime_seconds": 270.73, "saved_artifact_checks": 135, "full_feature_run_notes": {"warning": "pandas DtypeWarning on mixed evidence metadata in label_cases.csv.gz during the full features run; numerical feature and pilot checks passed", "reader_resolution": "The reader uses low_memory=False in the code frozen for final pytest; no labels or numerical feature values were changed", "metric_status": "No classifier fit and no classification or event-level metrics"}}`. Classifier tests при отсутствующем sklearn используют явно синтетический mock, не свидетельствуют о реальном sklearn fit.

Pilot residual equivalence: `{
  "passed": true,
  "rows": 768,
  "max_abs_difference": {
    "y_true": 0.0,
    "y_pred": 7.275957614183426e-12
  },
  "target_or_history_used_for_eligibility": false,
  "forecast_model_fit": false
}`.
Pilot feature equivalence: `{
  "passed": true,
  "rows": 768,
  "numeric_columns": 126
}`.
Независимая сверка разметки: 118 проверок raw CSV/finite windows/cutoffs/pilot в `outputs/e07b_checks/independent_label_check.json`. Первоначальное буквальное сравнение JSON evidence различало последние float digits; после сравнения декодированных чисел с заранее используемым допуском1e−8 maxdifference7.28e−12. Исходный отказ строковой проверки сохранён; данные, weak criterion, labels и состояния не изменены.
Защищённые прежние файлы и frozen E07b config: `{"passed": true, "checked_files": 213, "changed": [], "config_unchanged": true}`.
Git HEAD `bf99872b02daf379a736926093399bbf485d277c`; dirty status, seed42, versions, command, input/code/artifact SHA сохранены. CPU, sequential jobs=1, процесс останавливается при >5.5GiB; сети, установки, fullforecast training, commit/push не было. PeakWorkingSet измеряет память данного Python-процесса, не весь ноутбук.

## 9. Следующий шаг

Финальная сборка отчёта, презентации и README по сохранённым E01–E07b артефактам — отдельная задача. Возможная будущая early-warning оценка требует более длинной истории/проверенной разметки и отдельного решения, без ослабления текущего критерия или подбора по уже просмотренному test.
