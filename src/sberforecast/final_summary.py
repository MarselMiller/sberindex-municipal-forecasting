"""Read-only assembly of F1 evidence; rendering writes exclusively reports/final."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess
import sys

import pandas as pd

from .final_forecasting import collect_forecasting
from .final_detection import collect_detection
from .final_evidence import collect_evidence

DATA_STATUSES = {'real', 'synthetic', 'diagnostic', 'not_evaluated'}
LIMITATIONS = [
    ('history', 'Только 24 месяца истории цели (январь 2023 — декабрь 2024). Число МО не заменяет число независимых календарных дат.'),
    ('annual', 'h=12 имеет одну forecast origin — декабрь 2023. Validation отсутствует; годовые direct-стратегии полностью используют fallback, устойчивый ranking невозможен.'),
    ('availability', 'release_lag_months=0 — допущение. Исторические даты публикации и vintages расходов не подтверждены; доверие датированным официальным макро/news-архивам также является явной предпосылкой.'),
    ('holdout', 'Holdout был просмотрен в ходе исследования. Последующие после E01 сравнения не являются новой полностью независимой проверкой. E06a также повторно использует synthetic test E04a.'),
    ('truth', 'Нет независимого ground truth реальных structural shocks. Сигналы и breakpoints — диагностика; weak events обозначают изменения ошибки baseline, а не подтверждённые экономические шоки.'),
    ('news', 'Исторический news-корпус ограничен и преимущественно national. Региональных/муниципальных исторических событий нет; нулевое наблюдаемое число не доказывает отсутствие новостей.'),
    ('real_warning', 'Для real early warning недостаточно положительных событий и независимых train/test onset-дат. B0–B4 не обучались; отсутствующие метрики не являются нулями.'),
    ('transfer', 'Synthetic early warning не переносится автоматически на реальные экономические шоки. Chronos-2 также является ретроспективным backtest checkpoint, выпущенного после дат оценки; пересечение pretraining не исключено.'),
]
AI_DISCLOSURE = ('При разработке проекта использовались инструменты ИИ-ассистирования ChatGPT / Codex '
                 'для помощи в написании и проверке кода, организации экспериментов и подготовке документации. '
                 'Методологические решения, запуск экспериментов, проверка результатов и итоговые выводы '
                 'контролировались автором проекта.')


def sha256(path: Path) -> str:
    with path.open('rb') as handle:
        digest = hashlib.sha256()
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def clean_json(value):
    """Standard JSON: undefined metrics are null, never NaN or artificial zero."""
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if hasattr(value, 'item'):
        return clean_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return value.as_posix()
    return value


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(clean_json(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def output_path(root: Path, value: str) -> Path:
    resolved = (root / value).resolve()
    permitted = (root / 'reports/final').resolve()
    if not resolved.is_relative_to(permitted):
        raise ValueError('F1 can only write inside reports/final; frozen reports/outputs are protected')
    return resolved


def validate_records(records: list[dict], component: str) -> None:
    if not records:
        raise ValueError(f'{component}: no evidence records')
    for row in records:
        if row.get('data_status') not in DATA_STATUSES:
            raise ValueError(f'{component}: missing/invalid data_status')
        if row['data_status'] in {'diagnostic', 'not_evaluated'}:
            for field in ('mae_macro', 'mae_micro', 'r2_pooled', 'precision', 'recall', 'f1',
                          'pr_auc', 'row_f1', 'event_recall', 'alert_precision'):
                value = row.get(field)
                if value is not None and pd.notna(value):
                    raise ValueError(f'{component}: unevaluated quality {field} must be null')


def fmt(value, digits=3) -> str:
    if value is None or (isinstance(value, (float, int)) and not math.isfinite(value)):
        return '—'
    if isinstance(value, bool):
        return 'да' if value else 'нет'
    if isinstance(value, int):
        return str(value)
    return f'{value:.{digits}f}' if isinstance(value, float) else str(value)


def table(headers: list[str], rows: list[list]) -> str:
    def escape(x):
        return str(x).replace('|', '\\|').replace('\n', ' ')
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(escape(x) for x in row) + ' |' for row in rows])


def forecasting_table(records: list[dict], split: str) -> str:
    models = list(dict.fromkeys(r['model'] for r in records))
    lookup = {(r['model'], r['split'], r['horizon']): r for r in records}
    rows = []
    for model in models:
        values = []
        for h in (1, 3, 6, 12):
            r = lookup[(model, split, h)]
            values.append(fmt(r['mae_macro'], 2) + ('†' if r['fallback_count'] else ''))
        rows.append([model] + values)
    return table(['Модель', 'h=1', 'h=3', 'h=6', 'h=12'], rows)


def criteria() -> list[dict]:
    values = [
        ('Методология', 10, 'Причинные признаки, календарные горизонты, общие ключи; различены прогноз, detection и warning',
         'RESULTS_SUMMARY.md; results_summary.json; terminology.md', 'completed', 'L=0 и доверие архивам — допущения; независимой real shock-разметки нет'),
        ('Forecasting + Prophet + 1/3/6/12', 20, '9 моделей/стратегий; одинаковые 63 МО, validation и holdout',
         'forecasting_metrics.csv; figures/forecast_mae.png', 'completed', 'h12: один origin, direct fallback; holdout уже просмотрен'),
        ('Change-point detection', 20, 'Online CUSUM/EWMA/BOCPD и offline PELT/BinSeg: synthetic качество и раздельная real диагностика',
         'detection_metrics.csv; figures/offline_example.png; figures/real_diagnostic.png', 'partially_completed', 'Качество на реальных МО не оценено без ground truth; offline использует весь отрезок'),
        ('Foundation models', 15, 'Chronos-2 zero-shot оценён на том же историческом пилоте, MAE/R² и покрытие сохранены',
         'forecasting_metrics.csv; outputs/chronos_zero_shot_v1/run_manifest.json', 'completed', 'Улучшение не подтверждено; checkpoint позже backtest и pretraining overlap не исключён'),
        ('News integration', 15, 'Временное согласование и canonical grouping реализованы; historic coverage проверено',
         'results_summary.json#news; figures/news_timeline.png', 'limited_by_data', 'Incremental predictive contribution news на real early warning не оценён'),
        ('MAE / R²', 10, 'MAE macro/micro в рублях и pooled R² пересчитаны из сохранённых прогнозов',
         'forecasting_metrics.csv; results_summary.json#forecasting', 'completed', 'Pooled R² не изолирует качество временной динамики каждого МО'),
        ('Interpretation / reproducibility', 10, 'SHA источников, определения метрик, команды, manifests, ограничения и выбранные примеры',
         'results_summary.json#provenance; figure_manifest.csv; limitations.md', 'partially_completed', 'Воспроизведение из чистой копии и состав публичной публикации ещё не проверены'),
    ]
    return [dict(zip(('criterion', 'weight', 'evidence', 'artifact', 'status', 'limitation'), row)) for row in values]


def terminology_text() -> str:
    terms = [
        ('МО и цель', 'Муниципальное образование; месячное значение «Все категории» — оценка средних безналичных расходов жителей в исходных номинальных рублях.'),
        ('forecast origin / horizon', 'Дата выпуска O и число календарных месяцев h до цели. Forecast origin хранится последним днём месяца; точное время доступности описано в исходном протоколе.'),
        ('backtest / validation / holdout', 'Историческая проверка; validation — цели до июня 2024, holdout — июль–декабрь 2024. Этот holdout уже просмотрен.'),
        ('baseline / feature / pipeline', 'Простой ориентир; доступный на дату выпуска признак; последовательность подготовки данных, расчёта и оценки.'),
        ('fallback / native coverage', 'Fallback — заранее назначенный резерв вместо основного алгоритма. Native coverage — доля прогнозов, выданных собственным алгоритмом без резерва. Область сравнения по таким прогнозам требует одинаковых ключей всех моделей.'),
        ('MAE macro / micro / pooled R²', 'MAE macro: среднее MAE по МО с равными весами. Micro: среднее по прогнозным случаям. Pooled R²: 1−SSE/SST по общей группе фактов; это не среднее индивидуальных R².'),
        ('online / offline detection, change point', 'Online использует текущий и прошлые месяцы; offline сегментирует весь анализируемый отрезок. Change point — кандидат границы изменения, не подтверждённый экономический шок.'),
        ('prefix stability', 'Сегментация повторяется на сохранённых исторических префиксах; проверяется повторное появление/сдвиг кандидата при удлинении ряда. Сопоставление с итоговой сегментацией является ретроспективной диагностикой.'),
        ('weak labels / active regime / censoring', 'Weak labels получены из устойчивого изменения ошибки SeasonalNaiveYoY, требуют трёх месяцев подтверждения. Из warning исключается только уже подтверждённый на O активный режим. Censoring: полного будущего окна нет; такая метка unknown, а не negative.'),
        ('row-level / event-level metrics', 'Row-level оценивает каждый МО/ряд × origin × k. Event-level сопоставляет одному событию одно первое успешное предупреждение; повторы учитываются отдельно.'),
        ('PR-AUC / lead time / false alerts', 'В E07c PR-AUC — average precision с группировкой равных вероятностей. Lead time — месяцы от первого предупреждения до onset. False alerts/12 — число ложных месячных предупреждений ×12 / известные at-risk месяцы.'),
        ('alert precision', 'E07c: предупреждённые события / (предупреждённые события + ложные месячные alerts), после удаления повторов успешного warning. Не равна row precision при k=3.'),
        ('coverage / calibration', 'Coverage описывает конечные значения и доступность информации, а не качество. Calibration сравнивает предсказанные вероятности с частотой событий.'),
        ('zero-shot / ablation', 'Zero-shot — применение pretrained весов без дообучения. Ablation — сравнение фиксированных наборов компонентов/признаков на одной области.'),
        ('snapshots / canonical events / исторический допуск', 'Snapshot — конкретная сохранённая версия/извлечённое ядро. Canonical group объединяет дубли одного события. Исторический допуск требует доступности к O по записанной политике, а не только даты в URL.'),
        ('data_status', 'real: forecasting метрики непосредственно измерены на СберИндексе; coverage news/макро относится к реально сохранённому корпусу/источникам. synthetic: controlled benchmark; diagnostic: реальные данные без ground truth; not_evaluated: качество не оценено. Неопределённые метрики представлены null/пустой CSV-ячейкой.'),
    ]
    return '# Термины итоговых материалов\n\n' + table(['Термин', 'Значение в проекте'], [[a, b] for a, b in terms]) + '\n'


def git_output(root: Path, *args) -> str:
    return subprocess.run(['git', *args], cwd=root, capture_output=True, text=True, encoding='utf-8', check=True).stdout.strip()


def protected_files(root: Path) -> dict[str, str]:
    """Small explicit scope: tracked source/config plus top-level frozen artifacts, no weights/raw data."""
    paths = {root / p for p in git_output(root, 'ls-files').splitlines()
             if p == 'README.md' or p == '.gitignore' or p.startswith(('configs/', 'src/', 'scripts/', 'tests/', 'requirements', 'pyproject'))}
    paths.update((root / 'reports/results').glob('E0*.md'))
    dirs = ['baseline_v1', 'prophet_comparison_v1', 'catboost_direct_v1', 'chronos_zero_shot_v1', 'trend_calendar_v1',
            'macro_data_audit_v1', 'macro_forecast_v1', 'national_local_lightgbm_v1', 'online_detection_v1',
            'offline_detection_v1', 'news_events_v1', 'news_events_v2', 'news_events_v3',
            'early_warning_feasibility_v1', 'early_warning_full_panel_v1', 'early_warning_synthetic_v1']
    for name in dirs:
        paths.update(p for p in (root / 'outputs' / name).glob('*') if p.is_file())
    return {p.relative_to(root).as_posix(): sha256(p) for p in sorted(paths) if p.is_file()}


def build_final_summary(root: Path, destination='reports/final', command=None) -> dict:
    root = root.resolve()
    out = output_path(root, destination)
    before = protected_files(root)
    reports = sorted((root / 'reports/results').glob('E0*.md'))
    if len(reports) != 13:
        raise ValueError('Expected all 13 frozen final E01–E07 reports')
    forecasting = clean_json(collect_forecasting(root))
    detection = clean_json(collect_detection(root))
    evidence = clean_json(collect_evidence(root))
    for name, section in [('forecasting', forecasting), ('detection', detection), ('early_warning', evidence)]:
        if section.get('passed', True) is not True:
            raise ValueError(f'{name}: source audit failed')
        validate_records(section['records'], name)
    source_provenance = []
    for section in (forecasting, detection, evidence):
        source_provenance.extend(section.get('provenance', []))
    source_provenance.extend(dict(path=p.relative_to(root).as_posix(), sha256=sha256(p), role='frozen_final_report') for p in reports)
    out.mkdir(parents=True, exist_ok=True)
    for filename, records in [('forecasting_metrics.csv', forecasting['records']),
                              ('detection_metrics.csv', detection['records']), ('early_warning_metrics.csv', evidence['records'])]:
        pd.DataFrame(records).to_csv(out / filename, index=False)
    mapping = criteria()
    pd.DataFrame(mapping).to_csv(out / 'competition_criteria.csv', index=False)
    (out / 'competition_criteria.md').write_text('# Соответствие конкурсным критериям\n\nСтатусы описывают выполненную работу; баллы жюри не присваиваются.\n\n' +
        table(['Критерий', 'Вес, %', 'Подтверждение', 'Артефакт', 'Статус', 'Ограничение'],
              [[r[k] for k in ('criterion', 'weight', 'evidence', 'artifact', 'status', 'limitation')] for r in mapping]) + '\n', encoding='utf-8')
    (out / 'limitations.md').write_text('# Основные ограничения\n\n' + '\n\n'.join(f'{i}. {t}' for i, (_, t) in enumerate(LIMITATIONS, 1)) + '\n', encoding='utf-8')
    (out / 'terminology.md').write_text(terminology_text(), encoding='utf-8')
    (out / 'ai_disclosure_draft.md').write_text('# Подготовленная формулировка AI disclosure\n\n' + AI_DISCLOSURE +
        '\n\nФормулировка подготовлена для последующего использования; в README, отчёт и презентацию автоматически не вставлялась. Фактическое использование Claude Code не подтверждено и не указано.\n', encoding='utf-8')
    from .final_figures import build_figures
    figures = build_figures(root, out, forecasting, detection, evidence)
    pd.DataFrame(figures).to_csv(out / 'figure_manifest.csv', index=False)
    after = protected_files(root)
    if before != after:
        raise ValueError('Protected research files changed during F1 assembly')
    summary = dict(schema_version='F1.1', language='ru', research_phase='closed',
        target={'data_status': 'real', 'definition': 'Месячное значение «Все категории» по МО: оценка средних безналичных расходов жителей, исходные номинальные рубли',
                'history': ['2023-01', '2024-12'], 'history_months': 24, 'municipalities': 2190,
                'provenance': 'docs/PROJECT_CONTEXT.md; outputs/prophet_comparison_v1/data_audit.json'},
        forecasting=forecasting, detection=detection, **{k: v for k, v in evidence.items() if k not in {'records', 'facts', 'provenance', 'discrepancies', 'passed'}},
        early_warning_records=evidence['records'], evidence_facts=evidence.get('facts'),
        verified_components=verified_components(), competition_criteria=mapping,
        limitations=[dict(id=k, text=t) for k, t in LIMITATIONS], ai_disclosure_draft=AI_DISCLOSURE,
        figures=figures, provenance=source_provenance,
        discrepancies=forecasting.get('discrepancies', []) + detection.get('discrepancies', []) + evidence.get('discrepancies', []),
        reproducibility=dict(generated_at=datetime.now(timezone.utc).isoformat(), research_head=git_output(root, 'rev-parse', 'be6d993'),
            git_head=git_output(root, 'rev-parse', 'HEAD'), git_status=git_output(root, 'status', '--short'), command=command or sys.orig_argv,
            python=sys.version, versions={p: importlib.metadata.version(p) for p in ('numpy', 'pandas', 'matplotlib', 'pytest', 'PyYAML')},
            model_fitting=False, new_experiments=False, network=False, installs=False, full_pytest='run separately after builder',
            protected_file_count=len(before), protected_files_unchanged=True, protected_scope='tracked code/config/tests and top-level frozen artifacts; not complete raw/weight/partition audit',
            clean_copy_reproduction='not_checked', public_data_audit='not_checked'))
    summary['reproducibility']['builder_code_sha256'] = {p.relative_to(root).as_posix(): sha256(p)
        for p in sorted((root / 'src/sberforecast').glob('final_*.py')) + [root / 'scripts/build_final_summary.py']}
    from .final_render import render_summary
    (out / 'RESULTS_SUMMARY.md').write_text(render_summary(summary), encoding='utf-8')
    write_json(out / 'results_summary.json', summary)
    write_json(out / 'build_validation.json', dict(status='passed', protected_files_unchanged=True, before=before,
        after=after, source_provenance=source_provenance,
        model_fitting=False, new_experiments=False,
        final_files={p.relative_to(out).as_posix(): sha256(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'build_validation.json'}))
    return dict(status='passed', forecasting_records=len(forecasting['records']), detection_records=len(detection['records']),
                early_warning_records=len(evidence['records']), figures=len(figures))


def verified_components() -> list[dict]:
    return [dict(component=c, data_status=d, metric=m, status=s) for c, d, m, s in [
        ('Forecasting', 'real', 'MAE macro/micro; pooled R²', 'evaluated'),
        ('Online detection', 'synthetic', 'Precision/Recall/F1/FP/delay', 'evaluated'),
        ('Online detection', 'diagnostic', 'Real alert counts', 'diagnostic'),
        ('Offline detection', 'synthetic', 'F1/localisation/FP', 'evaluated'),
        ('Offline detection', 'diagnostic', 'Real candidates; prefix stability', 'diagnostic'),
        ('News pipeline', 'real', 'Historic coverage; temporal vectors', 'implemented'),
        ('Real early warning', 'not_evaluated', 'Data sufficiency checked; classifier metrics absent', 'limited_by_data'),
        ('Synthetic early warning', 'synthetic', 'PR-AUC; event recall; lead time; false alerts', 'evaluated'),
        ('Foundation model', 'real', 'Historic MAE/R² backtest', 'evaluated'),
    ]]
