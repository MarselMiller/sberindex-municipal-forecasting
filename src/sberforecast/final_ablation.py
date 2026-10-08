"""Add the completed persistence ablation without rebuilding the main benchmark.

Only public saved aggregates and verification metadata are read. No research
predictions, private observations, model code or experiment runners are imported.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path


SOURCE_DIR = 'reports/results/national_local_persistence'
REPORT = 'reports/results/national_local_persistence.md'
CSV_FILES = ('metrics.csv', 'gap_analysis.csv', 'origin_deltas.csv',
             'origin_stability.csv', 'municipality_win_rates.csv')
MODEL_LABELS = {
    'LastValue': 'LastValue',
    'SeasonalNaiveYoY': 'SeasonalNaiveYoY',
    'LightGBMDirect': 'LightGBMDirect',
    'NationalLocalPersistence': 'National/Local Persistence',
    'NationalLocalLightGBM': 'National/Local + LightGBM',
}
CONCLUSION = (
    'Дополнительная ablation показала, что сохранение последнего доступного отношения расходов МО '
    'к национальной медиане закрывает 68–76% наблюдаемого holdout-разрыва MAE между '
    'LightGBMDirect и National/Local + LightGBM на горизонтах 1/3/6 месяцев. '
    'Это указывает на существенную роль разложения National/Local в данном сравнении. '
    'При этом преимущество Persistence относительно SeasonalNaiveYoY и дополнительный выигрыш '
    'обучаемой локальной модели не полностью устойчивы между validation, holdout и forecast origins.'
)
LIMITATIONS = [
    'Holdout уже просмотрен: ablation является описательным анализом разложения и устойчивости, '
    'а не новой независимой слепой проверкой.',
    'История содержит 24 месяца; validation h=6 имеет одну дату выпуска, holdout h=1/3/6 — по шесть.',
    'h=12 — только описательный срез с одной датой выпуска и без validation; у learned references '
    'сохранён SeasonalNaive fallback, Persistence применяет собственную фиксированную формулу.',
    'Доля разрыва MAE относится к двум указанным стратегиям; причинное объяснение результата '
    'и универсальное преимущество разложения не установлены.',
    'Доступность расходов при release_lag_months=0 остаётся допущением исходного протокола.',
]


def text_sha(path: Path) -> str:
    return hashlib.sha256(path.read_text(encoding='utf-8-sig').encode('utf-8')).hexdigest()


def verify_sha(path: Path, expected: str) -> None:
    # Git may change checkout line endings. The original sealed LF content must match.
    actual = {hashlib.sha256(path.read_bytes()).hexdigest(), text_sha(path)}
    if expected not in actual:
        raise ValueError(f'Saved ablation source SHA mismatch: {path.name}')


def saved_rows(path: Path) -> list[dict]:
    strings = {'split', 'model', 'scope', 'metric_status', 'from_model', 'to_model', 'share_status'}
    integers = {'horizon', 'n_predictions', 'n_municipalities', 'n_origins', 'n_native',
                'n_fallback', 'n_failed', 'n_requested_all', 'n_rows'}
    result = []
    with path.open(encoding='utf-8-sig', newline='') as stream:
        for raw in csv.DictReader(stream):
            row = {}
            for key, value in raw.items():
                if key in strings:
                    row[key] = value
                elif key == 'descriptive_only':
                    if value not in {'True', 'False'}:
                        raise ValueError('Invalid saved descriptive_only flag')
                    row[key] = value == 'True'
                elif value in {'', 'NA'}:
                    row[key] = None
                else:
                    number = float(value)
                    if not math.isfinite(number) or (key in integers and not number.is_integer()):
                        raise ValueError(f'Invalid saved aggregate: {key}')
                    row[key] = int(number) if key in integers else number
            for key, label_key in [('model', 'model_label'), ('from_model', 'from_label'), ('to_model', 'to_label')]:
                if key in row:
                    row[label_key] = MODEL_LABELS[row[key]]
            result.append(row)
    return result


def collect_forecasting_ablation(root: Path) -> dict:
    directory = root / SOURCE_DIR
    manifest = json.loads((directory / 'run_manifest.json').read_text(encoding='utf-8-sig'))
    gate = manifest['reproduction_gate']
    if (gate['status'] != 'PASS' or manifest['n_model_fits'] != 0
            or manifest['n_model_fit_attempts'] != 0
            or manifest['no_tuning'] is not True or manifest['no_feature_search'] is not True):
        raise ValueError('Saved ablation verification or fixed-method gate failed')
    for filename in CSV_FILES:
        verify_sha(directory / filename, manifest['artifact_sha256'][filename])
    verify_sha(root / REPORT, manifest['report_sha256'])
    metrics = saved_rows(directory / 'metrics.csv')
    gaps = saved_rows(directory / 'gap_analysis.csv')
    expected_keys = {(split, h, m) for split in ('validation', 'holdout') for h in (1, 3, 6, 12)
                     for m in MODEL_LABELS if (split, h) != ('validation', 12)}
    if len(metrics) != 35 or {(r['split'], r['horizon'], r['model']) for r in metrics} != expected_keys:
        raise ValueError('Saved ablation must contain five strategies on the fixed split/horizon cases')
    contrasts = {('LastValue', 'NationalLocalPersistence'), ('SeasonalNaiveYoY', 'NationalLocalPersistence'),
                 ('LightGBMDirect', 'NationalLocalPersistence'), ('NationalLocalPersistence', 'NationalLocalLightGBM')}
    expected_gaps = {(s, h, a, b) for s, h, _ in expected_keys for a, b in contrasts}
    if len(gaps) != 28 or {(r['split'], r['horizon'], r['from_model'], r['to_model']) for r in gaps} != expected_gaps:
        raise ValueError('Saved ablation comparison keys differ')
    source_names = [REPORT] + [f'{SOURCE_DIR}/{name}' for name in (*CSV_FILES, 'run_manifest.json')]
    return {
        'schema_version': 1, 'id': 'national_local_persistence', 'data_status': 'real',
        'role': 'methodological_ablation', 'included_in_main_benchmark': False,
        'definition': 'Прежний национальный прогноз SeasonalNaiveYoY умножается на последнее конечное '
                      'отношение расходов МО к национальной медиане, доступное на дату выпуска.',
        'models': [{'id': key, 'label': value} for key, value in MODEL_LABELS.items()],
        'primary_horizons': [1, 3, 6], 'descriptive_horizons': [12],
        'metric': 'mae_macro', 'unit': 'nominal_RUB',
        'metrics': metrics, 'gap_analysis': gaps,
        'gap_share_definition': '(MAE Direct − MAE Persistence) / (MAE Direct − MAE NL+LightGBM); '
                                'описательная доля наблюдаемого разрыва, без ограничения диапазоном [0, 1]; '
                                'при неположительном или отсутствующем знаменателе — null.',
        'interpretation': {'category': manifest['interpretation']['overall'],
                           'label': manifest['interpretation']['label'],
                           'per_horizon': manifest['interpretation']['per_horizon'], 'conclusion': CONCLUSION},
        'limitations': LIMITATIONS,
        'verification': {'status': gate['status'], 'model_fits': manifest['n_model_fits'],
                         'tolerance_absolute': gate['tolerance_absolute'],
                         'maximum_strategy_metric_difference': gate['saved_metric_checks']['metrics_strategy']['max_abs_delta'],
                         'same_evaluation_cases': True, 'holdout_previously_inspected': True,
                         'new_experiments_during_integration': False},
        'sources': [{'path': name, 'sha256': text_sha(root / name),
                     'hash_basis': 'UTF-8 text normalized to LF, without BOM'} for name in source_names],
    }


def refresh_forecasting_ablation(root: Path, destination='reports/final', *, check=False) -> dict:
    """Refresh only the additive JSON object; preserve every previously published value."""
    root = root.resolve()
    out = (root / destination).resolve()
    if not out.is_relative_to((root / 'reports/final').resolve()):
        raise ValueError('Ablation refresh can only write inside reports/final')
    path = out / 'results_summary.json'
    summary = json.loads(path.read_text(encoding='utf-8-sig'))
    ablation = collect_forecasting_ablation(root)
    main = {(r['split'], r['horizon'], r['model']): r for r in summary['forecasting']['records']}
    for row in ablation['metrics']:
        key = row['split'], row['horizon'], row['model_label']
        if key in main:
            for field in ('mae_macro', 'mae_micro', 'r2_pooled', 'n_predictions', 'n_municipalities', 'n_origins'):
                if row[field] != main[key][field]:
                    raise ValueError(f'Ablation reference differs from the main benchmark: {key}, {field}')
    if check:
        if summary.get('forecasting_ablation') != ablation:
            raise ValueError('Published forecasting ablation is stale; rebuild with --forecasting-ablation-only')
    else:
        summary['forecasting_ablation'] = ablation
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return {'status': 'passed', 'forecasting_ablation_records': len(ablation['metrics']),
            'main_benchmark_changed': False, 'model_fitting': False, 'new_experiments': False}
