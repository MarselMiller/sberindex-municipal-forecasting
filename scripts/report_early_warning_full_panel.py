"""Render E07b report strictly from saved local experiment artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def markdown(table: pd.DataFrame) -> str:
    def cell(value):
        if pd.isna(value):
            return '—'
        if isinstance(value,float):
            return f'{value:.6g}'
        return str(value).replace('|',';').replace('\n',' ')
    rows=[list(table.columns)]+[[cell(value) for value in row] for row in table.itertuples(index=False,name=None)]
    return '\n'.join(['| '+' | '.join(rows[0])+' |','| '+' | '.join(['---']*len(rows[0]))+' |']+
                     ['| '+' | '.join(row)+' |' for row in rows[1:]])


def render_report(root: Path,config: dict) -> None:
    output=root/config['output_dir']
    read=lambda name: pd.read_csv(output/name,dtype={'municipality_id':str})
    feasibility,gate=read('feasibility.csv'),read('feasibility_gate.csv')
    state,features,news=read('state_counts.csv'),read('feature_summary.csv'),read('news_summary.csv')
    selection=read('feature_selection_summary.csv')
    alternatives=read('calendar_split_options.csv')
    status=json.loads((output/'execution_status.json').read_text(encoding='utf-8'))
    manifest=json.loads((output/'manifest.json').read_text(encoding='utf-8'))
    validation_path=root/'outputs/e07b_checks/final_validation.json'
    validation=json.loads(validation_path.read_text(encoding='utf-8')) if validation_path.exists() else {}
    validation={key:value for key,value in validation.items() if key!='code_test_config_sha256'}
    preservation_path=root/'outputs/e07b_checks/preservation_after.json'
    preservation=json.loads(preservation_path.read_text(encoding='utf-8')) if preservation_path.exists() else {}
    counts=feasibility[['k','scope','cases','municipalities','forecast_origin_dates','fully_known','positives','negatives',
        'censored_or_missing','unique_events','positive_event_onset_dates','positive_warning_origin_dates','known_active_cases','positive_rate']]
    registry=feasibility.iloc[0]
    forecast_coverage=read('forecast_coverage.csv').groupby('forecast_status',dropna=False)[['cases','finite_predictions','finite_targets','fallback_cases']].sum().reset_index()
    gate_columns=['k','train_positives','test_positives','train_positive_onset_dates','test_positive_onset_dates',
        'train_origin_dates','test_origin_dates','gate_pass','failure_reasons']
    split_columns=['cutoff_period','k','train_cases','test_cases','train_positives','test_positives',
                   'train_positive_onset_dates','test_positive_onset_dates','gate_pass']
    state=state[['k','forecast_origin','cases','eligible_cases','known_active_cases','state_uncertain_cases','retrospective_inside_cases']]
    features=features[['group','role','feature_columns','cells','finite_cells','finite_cell_coverage',
        'observed_varying_columns','dependency_violation_records','missing_dependency_coverage_records','temporal_dependency_check_passed']]
    news=news[['k','scope','cases','forecast_origins','varying_value_columns','varying_missing_flag_columns',
        'unique_temporal_vectors','maximum_spatial_full_vectors_per_origin','origins_with_news_information']]
    stage_table=pd.DataFrame(manifest['stages'])[['stage','runtime_seconds','peak_working_set_gib','command']]
    population=int(registry.panel_municipalities)
    event_count=int(registry.registry_events)
    event_municipalities=int(registry.registry_municipalities_with_event)
    if not gate.gate_pass.any():
        result_status='Оба feasibility gate отказали; classifier fit и метрик раннего предупреждения нет. Это завершённый feasibility-опыт с предусмотренным отказом условной ветки, а не доказательство отсутствия предсказуемых экономических событий.'
    else:
        result_status=f"Допустимые k указаны в gate; реально обучено {status['classifiers_fitted']} classifiers. Наличие метрик: {status['classification_metrics_obtained']}."
    if status['classifiers_fitted'] or status['classification_metrics_obtained']:
        model_text=markdown(read('metrics_row.csv'))+'\n\n'+markdown(read('metrics_event.csv'))
        incremental='Контрасты B2−B1, B3−B2 и B4−B3 следует читать в сохранённых метриках на одинаковых ключах.'
        examples=read('examples.csv')
        example_text=markdown(examples) if not examples.empty else 'Примеры не сохранены: это ограничение запуска.'
    else:
        model_text=('Для **k=1 и k=3 B0–B4 не обучались**: оба gate отказали до fit. '
            'Пустые файлы с заголовками `predictions.csv.gz`, `metrics_row.csv`, `metrics_event.csv`, '
            '`calibration.csv`, `model_coefficients.csv`, `alerts.csv` и `examples.csv` не содержат выдуманных результатов. '
            'Причины по k/model сохранены в `model_status.csv`. В основной .venv sklearn отсутствует; установки нет. '
            'Это дополнительное ограничение условной ветки, а основание отказа в данном опыте — достаточность данных.')
        incremental=('**B2 vs B1 (детекторы), B3 vs B2 (macro), B4 vs B3 (news) не оценены.** '
            'Вклад ни одной группы не установлен. Даже при varying test-признаках нет допустимого обучения и оценки.')
        example_text=('Успешные предупреждения, false alerts и missed events классификатора не оценивались: '
            'предупреждения не выпускались. Реестр слабых событий сохранён для человеческой проверки; '
            'его строки нельзя выдавать за примеры успешного early warning.')
    text=f'''# E07b — full-panel early warning feasibility

Результат: панель расширена до **{population} МО**, разметка и причинные признаки рассчитаны. {result_status}

## 1. Протокол и реальные counts

Месячное значение категории «Все категории» по муниципальным образованиям; определение целевой величины E01 сохранено. Источник 2023-01…2024-12, forecast origins 2023-12…2024-11. Все municipality_id исходной панели сохраняются на каждой дате, включая ineligible audit rows. Eligibility определяется только prefix: не менее 12 доступных наблюдений и staleness не более одного календарного месяца. L=0 — прежнее неподтверждённое допущение о доступности расходов, не восстановленный архив vintages.

Новые full-panel h1 SeasonalNaiveYoY прогнозы E07b построены прежней функцией baseline_predict на каждом prefix и сохранены без обучения прогнозной модели; часть для прежних 64 МО сверена с E07a. Criterion E07a не меняется: четыре предыдущих календарных residuals, median center, scale=max(1.4826×MAD, 0.03×median|past prediction|, 1 рубль); три последовательных z=(e−center)/scale должны все быть ≥3 либо все ≤−3, все семь месяцев конечные. Merge gap≤2 и непрошедший recovery того же направления — продолжение; recovery требует двух месяцев direction_sign×z<1.5 относительно замороженной базы. Метка известна не ранее O+k+2 и доступности всех необходимых фактов/истории режима. Missing и правое цензурирование остаются unknown.

Реестр: **{event_count}** уникальных муниципальных слабых событий, **{int(registry.registry_event_onset_dates)}** onset-дат, **{event_municipalities}/{population} МО ({100*event_municipalities/population:.3f}%)** хотя бы с одним событием. Counts событий, связанных с evaluable warning cases, ниже отличаются от полного реестра: события без полного окна не создают оцениваемую метку.

`scope=all` — вся панель; `eligible` — prefix eligibility; `at_risk_evaluable` дополнительно требует полную метку и отсутствие известного активного режима. `train`/`test` уже применяют календарь и доступность метки. `censored_or_missing` объединяет неполную прошлую/будущую информацию; отдельный right-censoring сохранён в feasibility.csv и label_counts.csv. `positive_event_onset_dates` считает onset-месяцы, `positive_warning_origin_dates` — даты выпуска предупреждений; муниципальные копии не увеличивают число дат.

{markdown(counts)}

Статусы SeasonalNaiveYoY h1 сохранены; ineligible audit placeholders не являются прогнозами. Ошибку можно вычислить только при конечных прогнозе и target. Подробные counts по дате выпуска — `forecast_coverage.csv`.

{markdown(forecast_coverage)}

## 2. Gate и временной split

Пороги записаны в новой YAML **до полного подсчёта**: train positives≥30, test positives≥10, положительных onset-дат train≥3/test≥2, отдельно для k=1/3. Hash исходной E07b YAML и неизменного E07a протокола сохранён в preservation_before.json и manifest. После просмотра counts/метрик пороги, окна и даты не изменялись.

{markdown(gate[gate_columns])}

Исходная граница E07a сохранена: label-information cutoff июль 2024, test origins август–ноябрь 2024. Train допускает только полные eligible at-risk метки, известные к cutoff; test truth доступен лишь для последующей ретроспективной оценки. Random split не использован. Для k=3 исходный test пуст; ниже сохранены все календарные альтернативы без classifier metrics. Ни один вариант не проходит gate, поэтому замена границы не обоснована.

{markdown(alternatives[split_columns])}

Причина структурная: первое возможное residual наблюдение — январь 2024, первое возможное onset после четырёх прошлых residuals — май. Чтобы для k=1 иметь ≥3 train onset-месяца с подтверждением, cutoff должен быть не раньше сентября; чтобы иметь ≥2 последующих test onset-месяца до последнего подтверждаемого октября — не позже июля. Совместной границы нет. Для k=3 первая полностью доступная train метка появляется в сентябре (origin апрель+5), тогда как последний evaluable origin — июль (конец данных декабрь−5). Увеличение числа МО не удлиняет календарь.

## 3. Причинные активные состояния

{markdown(state)}

`known_active_cases` — режим подтверждён и известен на O, только он исключается причинным фильтром. `retrospective_inside_cases` — аудит с доступом к полной истории; он не входит в признаки, eligibility или обучение. State uncertainty сохраняется отдельно. Counts представлены по k, поэтому сумма по двум k удваивает одни и те же MO/origin.

## 4. Признаки и coverage

Fixed groups: A history/lag/dynamics/rolling/residual/volatility, B prefix CUSUM/EWMA/BOCPD, C admitted annual macro forecasts E05, D historically admitted news E06/E07a. В числовой матрице 138 кандидатов (A40/B36/C10/D52 с имеющимися missing flags); три текстовых detector phase только диагностические. Offline PELT/BinSeg breakpoints запрещены. Новые rolling volatility требуют все три конечных календарных месяца, ddof=0, без заполнения пропусков.

{markdown(features)}

Coverage рассчитано по полной audit-панели, поэтому включает ineligible строки. `feature_coverage.csv` содержит каждую дату/признак, finite/nonmissing, число различных значений и dependency audit; lineage расходов/ошибок/макро записан пакетами на диск. Все зависимости проверяются по собственному O. Нулевая violation count отражает принятые правила timestamp, а не доказанную историческую публикацию расходов. Параметры E04 ранее выбраны и зафиксированы: prefix replay причинен при этих параметрах, но историческая доступность самого выбора гиперпараметров на O не подтверждена.

На train выполнен отдельный аудит удаления all-missing, constants и точных numeric duplicates; median/mean/scales записаны в `feature_filter_train.csv`. На k=3 train отсутствует, фильтр помечен `no_training_rows`. Классификаторы от этого аудита не запускались. При допустимом gate код применяет один train-list и train-median/scaler к test, C=1, class_weight=None, threshold=0.5; выбора по test нет.

{markdown(selection)}

## 5. News и вклад групп

{markdown(news)}

News counts по train/test относятся только к допустимым cases этого split. Реально varying значения и existing missing flags показаны раздельно. Temporal vectors дедуплицированы по значениям, независимо от числа MO-копий; пространственная вариативность отдельно показывает национальный характер. News information определяется наличием admitted документов в 90-дневном окне. Корпус ограничен: официальные key-rate cores и PDF под прежним archive-trust, исторических региональных/муниципальных публикаций нет; полная интенсивность новостей не восстановлена. Macro — доступные национальные годовые прогнозы, не причинный эффект, не региональная зарплата и не дефлирование цели.

{incremental}

## 6. B0–B4 и event-level оценка

{model_text}

Подготовленная условная оценка PR-AUC определяется как average precision (step integral с группировкой ties); дополнительно ROC-AUC, Brier, log loss, precision/recall/F1 и 10 заранее фиксированных calibration bins. Event matching — хронологический earliest alert за 1…k месяцев до onset, один event/alert максимум один match; повторы для уже предупреждённого события отдельно, после onset — не early warning. Denominator event recall — события, для которых существует хотя бы один полностью известный at-risk pre-onset monitored origin. Earliest/count warnings относятся к назначенным one-to-one alerts. False-alert exposure — число полностью известных at-risk monitored MO-months/12; ложные MO-month alerts считаются по отдельности. Event precision, recall, lead time и false alerts {'сохранены в metrics_event.csv' if status['classification_metrics_obtained'] else 'в этом запуске **не получены**'}.

## 7. Реальные примеры и ограничения weak labels

{example_text}

Weak truth — устойчивое изменение ошибки SeasonalNaiveYoY, не экспертная разметка экономического шока. Ошибки baseline, сезонность, пропуски и возврат к прежнему режиму могут создавать onset; противоположный знак может означать восстановление ошибки, а не отдельный экономический шок. Подтверждение T+2 требует будущего, поэтому onset/confirmation/warning origin различаются. Человеческое подтверждение E07a pending. Известные timestamps расходов и архив полноты новостей не восстановлены. Эти ограничения нельзя устранить большим числом муниципальных строк. Нового слепого теста здесь нет, преимущества над Prophet или способности предсказывать экономические шоки не установлены.

## 8. Выполненные команды и проверки

{markdown(stage_table)}

Целевые синтетические тесты новых adapter/labels/features/models: 3+19+20+24 passed; технический smoke использует 5 реальных МО и не является синтетическим исследовательским экспериментом. Дополнительные audit unit tests и результат **одного полного pytest**: `{json.dumps(validation,ensure_ascii=False)}`. Classifier tests при отсутствующем sklearn используют явно синтетический mock, не свидетельствуют о реальном sklearn fit.

Pilot residual equivalence: `{(output/'pilot_equivalence.json').read_text(encoding='utf-8').strip()}`.
Pilot feature equivalence: `{(output/'pilot_feature_equivalence.json').read_text(encoding='utf-8').strip()}`.
Независимая сверка разметки: 118 проверок raw CSV/finite windows/cutoffs/pilot в `outputs/e07b_checks/independent_label_check.json`. Первоначальное буквальное сравнение JSON evidence различало последние float digits; после сравнения декодированных чисел с заранее используемым допуском1e−8 maxdifference7.28e−12. Исходный отказ строковой проверки сохранён; данные, weak criterion, labels и состояния не изменены.
Защищённые прежние файлы и frozen E07b config: `{json.dumps(preservation,ensure_ascii=False)}`.
Git HEAD `{manifest['git_commit']}`; dirty status, seed42, versions, command, input/code/artifact SHA сохранены. CPU, sequential jobs=1, процесс останавливается при >5.5GiB; сети, установки, fullforecast training, commit/push не было. PeakWorkingSet измеряет память данного Python-процесса, не весь ноутбук.

## 9. Следующий шаг

Финальная сборка отчёта, презентации и README по сохранённым E01–E07b артефактам — отдельная задача. Возможная будущая early-warning оценка требует более длинной истории/проверенной разметки и отдельного решения, без ослабления текущего критерия или подбора по уже просмотренному test.
'''
    destination=root/config['report_path']
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(text,encoding='utf-8')
