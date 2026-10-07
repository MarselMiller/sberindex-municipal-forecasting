"""Saved-artifact reporting and fixed-rule examples for the E07c benchmark."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


WARNING = ('Синтетический controlled benchmark. Результаты не являются оценкой '
           'способности предсказывать реальные экономические шоки.')


def markdown(table: pd.DataFrame) -> str:
    def cell(value):
        if pd.isna(value):
            return '—'
        if isinstance(value, (float, np.floating)):
            return f'{value:.6g}'
        return str(value).replace('|', ';').replace('\n', ' ')
    if table.empty:
        return 'Нет наблюдений для этой таблицы; значения не заменены нулями.'
    rows = [list(table.columns)] + [[cell(value) for value in row] for row in table.itertuples(index=False, name=None)]
    return '\n'.join(['| ' + ' | '.join(rows[0]) + ' |', '| ' + ' | '.join(['---'] * len(rows[0])) + ' |'] +
                     ['| ' + ' | '.join(row) + ' |' for row in rows[1:]])


def _choose(table: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return table[[name for name in columns if name in table]]


def plot_examples(output: Path) -> list[dict]:
    """Every example comes from the evaluator's saved deterministic selection."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    read = lambda name: pd.read_csv(output / name, dtype={'series_id': str}, low_memory=False)
    examples = read('examples.csv')
    if examples.empty:
        return []
    observations, predictions, events = read('observations.csv.gz'), read('predictions.csv.gz'), read('true_events.csv')
    plots = output / 'plots'
    plots.mkdir(exist_ok=True)
    saved = []
    kind_column = next(name for name in ('example_type', 'type', 'kind') if name in examples)
    for example in examples.to_dict('records'):
        identifier, kind = str(example['series_id']), str(example[kind_column])
        k, model = int(example.get('k', 3)), str(example.get('model', 'S3'))
        obs = observations.loc[observations.series_id.eq(identifier)].sort_values('month_index')
        pred = predictions.loc[predictions.series_id.eq(identifier) & predictions.k.eq(k) & predictions.model.eq(model)].sort_values('month_index')
        own_events = events.loc[events.series_id.eq(identifier)]
        figure, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
        x = obs.month_index.to_numpy() + 1
        axes[0].plot(x, obs.y, color='navy', marker='.', label='Наблюдаемый synthetic y')
        axes[0].set_ylabel('Условные единицы y')
        axes[1].plot(x, obs.macro_pressure, color='darkorange', label='Synthetic macro pressure')
        axes[1].plot(x, obs.news_intensity, color='teal', label='Synthetic news intensity')
        axes[1].set_ylabel('Условные каналы')
        px = pred.month_index.to_numpy() + 1
        axes[2].plot(px, pred.probability, color='purple', marker='.', label=f'{model}, k={k}: риск')
        if not pred.empty:
            threshold = float(pred.threshold.iloc[0])
            axes[2].axhline(threshold, color='grey', linestyle=':', label=f'Validation threshold={threshold:.3f}')
            alerted = pred.alert.astype(bool)
            axes[2].scatter(px[alerted], pred.loc[alerted, 'probability'], color='red', marker='^', label='Alert', zorder=4)
        for event in own_events.itertuples():
            for axis in axes:
                axis.axvline(event.onset_index + 1, color='black', linestyle='--', label='Истинный onset')
        lead = example.get('lead_time_months', np.nan)
        lead_text = f'; lead={int(lead)} мес.' if pd.notna(lead) else '; успешное упреждение отсутствует'
        figure.suptitle(f'{identifier}: {kind}{lead_text}')
        axes[2].set_ylim(-0.03, 1.03)
        axes[2].set_ylabel('Вероятность')
        axes[2].set_xlabel('Месяц синтетического ряда (1…24)')
        for axis in axes:
            axis.grid(alpha=0.2)
            axis.legend(loc='upper left', fontsize=8)
        destination = plots / ('first_' + kind + '.png')
        figure.savefig(destination, dpi=140)
        plt.close(figure)
        saved.append(dict(example_type=kind, series_id=identifier, model=model, k=k,
            lead_time_months=float(lead) if pd.notna(lead) else None, path=str(destination.relative_to(output))))
    return saved


def render_report(root: Path, config: dict, *, smoke: bool = False) -> Path:
    output = root / config['output_dir']
    read = lambda name: pd.read_csv(output / name, dtype={'series_id': str}, low_memory=False)
    manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    actual = yaml.safe_load((output / 'config_resolved.yaml').read_text(encoding='utf-8'))
    metadata, events, counts = read('series_metadata.csv'), read('true_events.csv'), read('label_counts.csv')
    rows, event_metrics = read('metrics_row.csv'), read('metrics_event.csv')
    thresholds, bootstrap = read('selected_thresholds.csv'), read('bootstrap_intervals.csv')
    scenarios, ablation = read('scenario_metrics.csv'), read('ablation.csv')
    calibration, statuses = read('calibration.csv'), read('model_status.csv')
    cohorts = metadata.groupby('cohort', as_index=False).agg(series=('series_id', 'size'),
        events=('has_event', 'sum'), false_precursor_controls=('false_precursor', 'sum'))
    event_stats = events.groupby(['cohort', 'is_anticipated', 'precursor_class', 'noise_class'], dropna=False).size().reset_index(name='events')
    test_rows = rows.loc[rows.cohort.eq('test')] if 'cohort' in rows else rows
    test_events = event_metrics.loc[event_metrics.cohort.eq('test')] if 'cohort' in event_metrics else event_metrics
    test_scenarios = scenarios.loc[scenarios.cohort.eq('test')] if 'cohort' in scenarios else scenarios
    test_ablation = ablation.loc[ablation.cohort.eq('test')] if 'cohort' in ablation else ablation
    ablation_text = []
    for k in (1, 3):
        ranked = test_rows.loc[test_rows.k.eq(k)].set_index('model')
        matched = test_events.loc[test_events.k.eq(k)].set_index('model')
        if {'S1', 'S2', 'S3'}.issubset(ranked.index) and {'S1', 'S2', 'S3'}.issubset(matched.index):
            ablation_text.append(f"k={k}: S2−S1 PR-AUC {ranked.at['S2','pr_auc']-ranked.at['S1','pr_auc']:+.4f}, "
                f"event recall {matched.at['S2','event_recall']-matched.at['S1','event_recall']:+.4f}; "
                f"S3−S2 PR-AUC {ranked.at['S3','pr_auc']-ranked.at['S2','pr_auc']:+.4f}, "
                f"event recall {matched.at['S3','event_recall']-matched.at['S2','event_recall']:+.4f}.")
    checks_folder = root / 'outputs/e07c_checks'
    checks = {}
    if not smoke:
        for name in ('final_validation', 'preservation_after', 'independent_audit'):
            path = checks_folder / (name + '.json')
            if path.exists():
                value = json.loads(path.read_text(encoding='utf-8'))
                checks[name] = {key:item for key,item in value.items() if key not in ('code_test_config_sha256', 'details', 'files')}
    plots = plot_examples(output)
    (output / 'example_plots.json').write_text(json.dumps(plots, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')
    destination = output / 'synthetic_smoke_report.md' if smoke else root / config['report_path']
    destination.parent.mkdir(parents=True, exist_ok=True)
    plot_markdown = []
    for plot in plots:
        target = output / plot['path']
        import os
        relative = Path(os.path.relpath(target, destination.parent)).as_posix()
        plot_markdown.append(f"**{plot['example_type']} — {plot['series_id']}**, {plot['model']}, k={plot['k']}.\n\n![{plot['example_type']}]({relative})")
    if not plots:
        plot_markdown.append('Оценённых примеров нет; графики не выдуманы.')
    absent = set(('successful_warning', 'false_alert', 'missed_event')) - {plot['example_type'] for plot in plots}
    if absent:
        plot_markdown.append('Типы без сохранённого примера: ' + ', '.join(sorted(absent)) + '.')
    generator = actual['generator']
    calibration_s3 = calibration.loc[calibration.model.eq('S3') & calibration.k.eq(3)]
    if 'cohort' in calibration_s3:
        calibration_s3 = calibration_s3.loc[calibration_s3.cohort.eq('test')]
    validation_status = ('Один полный pytest и проверки сохранности ещё не выполнены.'
        if 'final_validation' not in checks else json.dumps(checks, ensure_ascii=False, indent=2))
    complete = (not smoke and checks.get('final_validation', {}).get('exit_code') == 0
        and checks.get('preservation_after', {}).get('passed', False)
        and checks.get('independent_audit', {}).get('passed', False))
    research_status = ('E07c completed. Experimental research phase closed.' if complete else
        'Финальные критерии готовности ещё не подтверждены; research phase пока не помечена закрытой.')
    report = f'''# E07c — synthetic early-warning benchmark

**{WARNING}**

{('Технический smoke на отдельных seeds; это не итоговый TEST.' if smoke else 'Контролируемый опыт выполнен на независимом TEST; генератор и параметры моделей не менялись по его результатам.')}

## 1. Генератор и задача

Задача: вероятность нового истинного structural level shift в O+1…O+k, k=1/3, по наблюдениям ≤O. Каждый ряд содержит 24 месячных наблюдения, случайный уровень, слабый тренд, годовую сезонность, неодинаковый шум и возможный устойчивый shift до конца ряда. Onset выбирается из zero-based indices {generator['onset_index_range']} (месяцы 17…21). Это генераторная истинная разметка; weak residual criterion E07a/E07b не переопределён и не используется как synthetic truth.

Генератор зафиксирован до TEST в `configs/early_warning_synthetic.yaml`, SHA256 `{manifest['source_config_sha256']}`. Exact fractions округляются по заранее заданному правилу: event fraction {generator['event_fraction']}, unanticipated fraction {generator['unanticipated_fraction']} среди событий, false precursor fraction {generator['controls_false_precursor_fraction']} среди контролей. У предсказуемых событий за 1…3 месяца могут возникать стохастические slope/volatility/residual и external macro/news процессы; сила меняется, минимум один канал активен. Ложные эпизоды контролей имеют ту же распределённую силу/длительность, но shift за ними не следует. Нет признака countdown или прямой даты события.

Полная неизменная спецификация: `generator_config.json` и `config_resolved.yaml` в новом output_dir. Все значения y/каналов — синтетические условные единицы, не реальные расходы, публикации или макропрогнозы.

## 2. Независимые cohorts и метки

{markdown(cohorts)}

Cohort seeds: `{json.dumps(actual['cohorts'], ensure_ascii=False)}`. `series_seed=cohort_seed×100000+index`; IDs/seeds не пересекаются. Независимость обеспечивается между целыми рядами; одинаковая относительная календарная сетка не означает проверку переноса во времени на реальные vintages.

{markdown(event_stats)}

Origins после 12 наблюдений. Label1: истинный onset в (O,O+k]; label0 только при полном конечном будущем окне; край/пропуски — unknown. Label availability O+k; дополнительное weak confirmation T+2 не требуется, поскольку истина задана генератором. Уже начавшийся regime исключён из target по oracle onset; это контролируемое допущение, а не доказанная причинная возможность такого исключения на реальных данных. Oracle metadata не поступает в модели.

{markdown(counts)}

## 3. Причинные признаки и модели

History: лаги, изменения, rolling statistics, local slope/volatility, прошлые causal SeasonalNaiveYoY errors и residual trend. Detector state: существующие CUSUM/EWMA/BOCPD классы на доступном prefix, causal warmup из трёх конечных ошибок, затем фиксированный center/scale; alarms не сбрасывают state. External: stochastic macro pressure/news intensity и trailing changes. Все группы перечислены в `feature_dictionary.csv`/`feature_groups.json`; onset/countdown/future/offline breakpoints исключены.

S0 — train positive rate; S1 — History; S2 — History+Detector; S3 — History+Detector+External. Filter all-missing/constant/exact duplicates, median и population standardization обучены только на TRAIN прежним `TrainOnlyPreprocessor` E07b. C=1, class_weight=None, seed42, max_iter=2000, без tuning. В установленном окружении sklearn отсутствует; backend — SciPy L-BFGS-B для той же бинарной logistic objective sum(logloss)+||w||²/(2C), intercept не штрафуется. Установок нет. Численная эквивалентность конкретной версии sklearn не заявляется; optimizer status и gradient diagnostics сохранены.

{markdown(_choose(statuses, ['model','k','backend','status','converged','iterations','training_rows','retained_features','optimizer_message']))}

## 4. Threshold только VALIDATION

Максимум validation row F1 при false monthly alerts ≤1 на 12 monitored CONTROL-months. Candidates — наблюдаемые validation probabilities, 0.5 и 1.0; равный F1 → более высокий threshold; при отсутствии допустимого threshold fallback0.5 с явным статусом. Контрольный бюджет не является гарантией на TEST. TEST labels не используются для обучения, фильтра, scaler/imputer или threshold.

{markdown(thresholds)}

## 5. Row-level TEST и calibration

PR-AUC = average precision, step integral с grouped ties. ROC-AUC дополнительна; Brier/logloss proper scores. Precision/F1 при отсутствии alerts следуют явной zero-division convention; event precision/lead без denominator остаются undefined.

{markdown(_choose(test_rows, ['model','k','cases','positives','negatives','pr_auc','roc_auc','brier_score','log_loss','precision','recall','f1','threshold','calibration_ece']))}

S3/k3 reliability table ниже; полные 10bins по всем моделям/горизонтам сохранены в `calibration.csv`. Calibration model отдельно не обучался.

{markdown(_choose(calibration_s3, ['bin','lower','upper','count','mean_probability','positive_fraction']))}

## 6. Event-level early warning и упреждение

Успех только за 1 месяц (k1) или 1…3 месяца (k3) ДО onset. Earliest alert и lead сохранены для каждого event. Каждый event считается успешным максимум один раз; repeated warnings записаны отдельно и не увеличивают recall/числитель precision. Alert precision = successes/(successes+false monthly alerts), повторы успешного предупреждения исключены. FAR = false monthly alerts/(fully-known at-risk monitored months/12); target exclusion сокращает exposure событийных рядов, не контролей. Alert в onset или позже никогда не считается early warning.

{markdown(_choose(test_events, ['model','k','eligible_events','warned_events','event_recall','alert_precision','median_lead_time_months','mean_lead_time_months','miss_rate','false_alert_count','false_alerts_per_12_monitored_months','monitored_cases','repeated_alert_count']))}

95% intervals: {actual['evaluation']['bootstrap_resamples']} bootstrap replicates целых TEST series_id с multiplicity, seed{actual['evaluation']['bootstrap_seed']}; все строки одного ряда остаются вместе. Это условная неопределённость данного fixed-generator опыта, без утверждения независимости месячных строк. Undefined replicates не заменены нулями.

{markdown(bootstrap)}

## 7. Ablation S2−S1 и S3−S2

Все модели оцениваются на одинаковых случаях; threshold каждой модели выбран на VALIDATION по единому правилу. Разности показывают synthetic predictive utility; detector при уже богатой истории может не добавить информации. External channels специально созданы связанными с риском; это не доказательство пользы реальных новостей или реальных макропрогнозов.

{markdown(test_ablation)}

{' '.join(ablation_text)} Это разности на сохранённых TEST predictions при отдельных validation thresholds; рост ranking metric не гарантирует рост recall при выбранном alert budget.

## 8. Anticipated / intentionally unanticipated

В TEST {int(events.loc[events.cohort.eq('test'), 'is_anticipated'].sum())} событий с внедрённым предвестником из {int(events.cohort.eq('test').sum())}; остальные intentionally unanticipated. Для рядов без cue до onset отсутствует внедрённая сигнализирующая информация; возможны случайные/prior warnings. Доля 75% — ориентир recall для идеального распознавания только cue-bearing событий, а не строгий математический потолок любого alert rule. Разбиение по anticipated/unanticipated сохранено в scenario_metrics; paired controls сохраняют сопоставимость false-alert exposure.

## 9. Stress scenarios

Предзаданные срезы weak/strong precursor, high noise и controls with false precursor; все показаны, лучший scenario не выбирался. Oracle scenario metadata использована только после прогнозирования для оценки.

{markdown(_choose(test_scenarios, ['scenario','model','k','metadata_series_count','monitored_series_count','eligible_events','warned_events','event_recall','alert_precision','median_lead_time_months','mean_lead_time_months','miss_rate','false_alert_count','false_alerts_per_12_monitored_months','pr_auc','f1']))}

На срезах возможны превышения validation budget: бюджет задавался на всех VALIDATION controls, не на каждом TEST stress subset. Отсутствие event у controls означает undefined event recall, а не нулевую способность предупреждения.

## 10. Фиксированно выбранные TEST examples

Правило зафиксировано заранее: S3/k3, затем k1 если тип отсутствует; первый series_id лексикографически, затем earliest origin. Успех/false alert/missed event берутся из сохранённого `examples.csv`; отсутствующий тип не создаётся вручную. На графиках показаны y, onset, наблюдаемые precursor channels, risk, validation threshold, alerts и lead.

{chr(10).join(plot_markdown)}

## 11. Выполненные команды, runtime и проверки

Command: `{manifest.get('experiment_command',manifest['command'])}`. Experiment runtime {manifest.get('experiment_runtime_seconds',manifest['runtime_seconds']):.3f}s, peak process working set {manifest.get('experiment_peak_working_set_gib',manifest['peak_working_set_gib']):.3f}GiB; CPU, jobs1/BLAS1, лимит5GiB. Seed, versions, Git commit/dirty, config/code/artifact SHA сохранены в manifest. Это память процесса, не всей системы.

```json
{validation_status}
```

Full pytest выполняется один раз после smoke/full experiment; повторное report stage только читает сохранённые результаты и не повторяет fit/TEST evaluation. Сети, установки, commit/push и изменения старых E01–E07b outputs не выполнялись; большой старый artifact audit не повторялся.

## 12. Ограничения и завершение research phase

Синтетические процессы намеренно связаны с событием и содержат точно известный onset; эта информация и независимые длинные cohorts отсутствуют в реальной E07a/E07b истории. Размер отдельных рядов остаётся24, однако cross-series обучение даёт много независимых генераторных повторений. Seasonal baseline, короткий warmup, ложные cues, неравный шум, oracle active exclusion и fixed onset range ограничивают интерпретацию. Detector scores описывают уже наблюдённое изменение; их использование до будущего shift не превращает detection delay в early-warning lead. Никаких выводов о настоящих экономических шоках, реальных news/macro или superiority над Prophet из этого опыта нет.

{research_status} Следующий этап — FINALIZATION ONLY: README, methodological report, results summary, presentation, clean reproduction check. Новые модели/источники/tuning не предлагаются.
'''
    destination.write_text(report, encoding='utf-8')
    return destination
