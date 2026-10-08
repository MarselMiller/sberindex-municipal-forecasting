"""Publish sealed robustness aggregates without running the research analysis."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import re

from .final_ablation import text_sha, verify_sha


SOURCE_DIR = 'reports/results/forecast_robustness'
REPORT = 'reports/results/forecast_robustness.md'
CSV_FILES = ('pairwise_metrics.csv', 'municipality_summary.csv', 'origin_deltas.csv',
             'origin_summary.csv', 'bootstrap_origin.csv', 'bootstrap_municipality.csv',
             'concentration.csv', 'applicability_checks.csv')
PAIRS = {
    'A': ('SeasonalNaiveYoY', 'National/Local + LightGBM'),
    'B': ('ProphetAuto', 'SeasonalNaiveYoY'),
    'C': ('ProphetYearly', 'SeasonalNaiveYoY'),
    'D': ('ProphetAuto', 'National/Local + LightGBM'),
    'E': ('ProphetYearly', 'National/Local + LightGBM'),
    'F': ('LightGBMDirect', 'National/Local + LightGBM'),
}
SIGN = 'MAE baseline − MAE candidate; positive = candidate better'
CONCLUSION = (
    'National/Local + LightGBM показывает минимальную holdout MAE среди девяти основных '
    'стратегий на h=1/3/6, но устойчивое преимущество над SeasonalNaiveYoY между периодами '
    'оценки не подтверждено: на validation ошибка больше на всех трёх горизонтах.'
)
LIMITATIONS = [
    'При 4–6 forecast origins интервалы описывают чувствительность к повторной выборке '
    'наблюдаемых дат; независимость соседних дат не установлена.',
    'При одной origin первичный интервал не оценивается; h=12 исключён из вывода об устойчивости.',
    'Общие национальные шоки связывают МО; bootstrap по МО служит дополнительной пространственной '
    'проверкой и не заменяет временную оценку.',
    'Прогнозы соседних origins используют пересекающуюся историю, а целевые месяцы разных '
    'горизонтов могут совпадать; bootstrap не устраняет эту зависимость.',
    'Holdout уже просмотрен; анализ не является новым слепым тестом и не подтверждает качество '
    'на будущих периодах. Доля положительных bootstrap draws не является p-value.',
    'История цели содержит 24 месяца; release_lag_months=0 и неизвестные historical vintages '
    'остаются допущениями исходного backtest.',
]


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def numeric(row: dict, field: str, *, integer=False):
    value = row[field]
    if value in ('', 'NA', None):
        return None
    result = float(value)
    if not math.isfinite(result) or (integer and not result.is_integer()):
        raise ValueError(f'Invalid saved robustness number: {field}')
    return int(result) if integer else result


def close(a, b):
    if a is None or b is None or abs(a - b) > 1e-8:
        raise ValueError('Saved robustness aggregate differs from its reference')


def interval(row: dict) -> dict:
    lower, upper = numeric(row, 'ci_lower'), numeric(row, 'ci_upper')
    status = row['interval_status']
    if status == 'DESCRIPTIVE_EMPIRICAL_SENSITIVITY':
        if lower is None or upper is None or lower > upper or numeric(row, 'n_origins') < 4:
            raise ValueError('Invalid descriptive origin interval')
    elif lower is not None or upper is not None:
        raise ValueError('An unestimated origin interval must remain null')
    return {'status': status, 'lower': lower, 'upper': upper,
            'includes_zero': None if lower is None else lower <= 0 <= upper,
            'role': 'descriptive_temporal_sensitivity', 'confidence_level': 0.95}


def collect_forecasting_robustness(root: Path, summary: dict) -> dict:
    directory = root / SOURCE_DIR
    manifest = json.loads((directory / 'run_manifest.json').read_text(encoding='utf-8-sig'))
    gate = manifest['comparability_gate']
    if (gate['status'] != 'PASS' or manifest['n_model_fits'] != 0 or manifest['new_predictions'] != 0
            or not all(manifest[k] is True for k in ('no_tuning', 'no_feature_search', 'no_primary_subset_selection'))
            or manifest['bootstrap_positive_share_is_p_value'] is not False
            or manifest['independent_verification']['status'] != 'PASS'
            or manifest['config']['primary_horizons'] != [1, 3, 6]
            or manifest['config']['descriptive_horizons'] != [12]
            or manifest['config']['delta_convention'] != 'baseline_mae_minus_candidate_mae'
            or manifest['delta_convention'] != SIGN
            or gate['tolerance_absolute'] != 1e-8 or gate['maximum_metric_difference'] > 1e-8):
        raise ValueError('Saved robustness comparability or fixed-method gate failed')
    for name in CSV_FILES:
        verify_sha(directory / name, manifest['artifact_sha256'][name])
    verify_sha(root / REPORT, manifest['post_run_report_review']['report_sha256'])
    groups = {(pair, split, h) for pair in PAIRS for split in ('validation', 'holdout')
              for h in (1, 3, 6, 12) if (split, h) != ('validation', 12)}
    tables = {}
    for name in CSV_FILES:
        rows = read_rows(directory / name)
        if name == 'origin_deltas.csv':
            keys = [(r['pair'], r['split'], int(r['horizon']), r['forecast_origin']) for r in rows]
            if len(keys) != len(set(keys)) or {k[:3] for k in keys} != groups:
                raise ValueError('Duplicate or missing saved origin keys')
            tables[name] = rows
            continue
        keys = [(r['pair'], r['split'], int(r['horizon'])) for r in rows]
        if len(keys) != len(groups) or set(keys) != groups:
            raise ValueError('Duplicate or missing saved robustness groups')
        tables[name] = dict(zip(keys, rows))
    main_rows = summary['forecasting']['records']
    main = {(r['model'], r['split'], r['horizon']): r for r in main_rows}
    if len(main_rows) != 72 or len(main) != 72 or len({k[0] for k in main}) != 9:
        raise ValueError('Main benchmark must retain nine strategies and 72 records')
    if {r['pair'] for r in gate['pairs']} != set(PAIRS):
        raise ValueError('Saved gate comparison pairs differ')
    for row in gate['pairs']:
        if (row['gate_status'] != 'PASS' or row['comparable_rows'] != 1890
                or row['n_municipalities'] != 63 or (row['baseline'], row['candidate']) != PAIRS[row['pair']]):
            raise ValueError('Saved gate evaluation scope differs')
    records = {}
    for key in sorted(groups):
        pair, split, h = key
        metric, mo, origin, boot, applicability = [tables[n][key] for n in
            ('pairwise_metrics.csv', 'municipality_summary.csv', 'origin_summary.csv',
             'bootstrap_origin.csv', 'applicability_checks.csv')]
        n_origins = (6 if split == 'holdout' and h != 12 else
                     {1: 6, 3: 4, 6: 1, 12: 1}[h])
        for name in CSV_FILES:
            if name == 'origin_deltas.csv':
                continue
            row = tables[name][key]
            if ((row['baseline'], row['candidate']) != PAIRS[pair]
                    or row['delta_convention'] != SIGN or row['descriptive_only'] != str(h == 12)):
                raise ValueError('Saved comparison labels, sign or inference scope differ')
        if (numeric(metric, 'n_comparable_rows') != 63 * n_origins
                or numeric(metric, 'n_municipalities') != 63
                or numeric(metric, 'n_origins') != n_origins):
            raise ValueError('Saved comparable case count differs')
        baseline_mae, candidate_mae, delta = [numeric(metric, f) for f in
                                             ('mae_baseline', 'mae_candidate', 'delta_mae')]
        close(delta, baseline_mae - candidate_mae)
        for model, mae, side in [(PAIRS[pair][0], baseline_mae, 'baseline'),
                                 (PAIRS[pair][1], candidate_mae, 'candidate')]:
            reference = main[(model, split, h)]
            close(mae, reference['mae_macro'])
            for field, value in [('n_predictions', 63 * n_origins), ('n_municipalities', 63),
                                 ('n_origins', n_origins), ('fallback_count', numeric(metric, side + '_fallback_count'))]:
                close(value, reference[field])
        for row, units in [(mo, 63), (origin, n_origins)]:
            close(numeric(row, 'n_units'), units)
            close(sum(numeric(row, f) for f in ('candidate_wins', 'candidate_losses', 'ties')), units)
            close(numeric(row, 'candidate_win_rate'), numeric(row, 'candidate_wins') / units)
            close(numeric(row, 'mean_delta_mae'), delta)
        origin_rows = [r for r in tables['origin_deltas.csv']
                       if (r['pair'], r['split'], int(r['horizon'])) == key]
        close(len(origin_rows), n_origins)
        close(sum(numeric(r, 'n_cases') for r in origin_rows), 63 * n_origins)
        close(sum(numeric(r, 'delta_mae') for r in origin_rows) / n_origins, delta)
        close(numeric(boot, 'observed_delta_mae'), delta)
        ci = interval(boot)
        expected_status = ('DESCRIPTIVE_ONLY_H12' if h == 12 else
                           'NOT_ESTIMABLE_ONE_ORIGIN' if n_origins == 1 else 'DESCRIPTIVE_EMPIRICAL_SENSITIVITY')
        if (ci['status'] != expected_status or applicability['dm_status'] != 'NOT_RELIABLE_NOT_ESTIMATED'
                or applicability['primary_ci_status'] != expected_status):
            raise ValueError('Saved temporal inference status differs')
        if h != 12:
            records[key] = {'split': split, 'horizon': h, 'n_comparable_rows': 63 * n_origins,
                'n_municipalities': 63, 'mae_baseline': baseline_mae, 'mae_candidate': candidate_mae,
                'delta_mae': delta, 'municipality_wins': numeric(mo, 'candidate_wins', integer=True),
                'municipality_win_rate': numeric(mo, 'candidate_win_rate'),
                'origin_wins': numeric(origin, 'candidate_wins', integer=True), 'n_origins': n_origins,
                'origin_bootstrap': ci}
    comparisons = []
    for pair in ('B', 'C'):
        comparisons.append({'baseline': PAIRS[pair][0], 'candidate': PAIRS[pair][1],
            'records': [{f: records[(pair, split, h)][f] for f in
                         ('split', 'horizon', 'delta_mae', 'origin_bootstrap')}
                        for split in ('validation', 'holdout') for h in (1, 3, 6)]})
    source_names = [REPORT] + [f'{SOURCE_DIR}/{name}' for name in (*CSV_FILES, 'run_manifest.json')]
    return {'schema_version': 1, 'id': 'forecasting_robustness', 'data_status': 'real',
        'role': 'saved_forecast_sensitivity_analysis', 'included_in_main_benchmark': False,
        'primary_horizons': [1, 3, 6], 'descriptive_horizons': [12], 'metric': 'mae_macro',
        'unit': 'nominal_RUB', 'delta_convention': SIGN,
        'comparable_cases': 'Одинаковые ключи МО × forecast origin × target month × horizon; '
            'совпадают конечные y_true и split, проверены статусы и cutoff; fallback не исключён.',
        'primary_pair': {'research_pair_id': 'A', 'baseline': PAIRS['A'][0], 'candidate': PAIRS['A'][1],
            'records': [records[('A', split, h)] for split in ('validation', 'holdout') for h in (1, 3, 6)]},
        'prophet_comparisons': comparisons,
        'prophet_conclusion': 'На holdout SeasonalNaiveYoY имеет меньшую MAE, чем оба проверенных '
            'Prophet на h=1/3/6. На validation ProphetAuto лучше на h=3/6, а ProphetYearly хуже '
            'на всех трёх горизонтах. Временная неопределённость остаётся большой.',
        'methods': {'primary_bootstrap': 'forecast_origin_cluster',
            'secondary_bootstrap': 'municipality_cluster_sensitivity_only',
            'saved_iterations': manifest['bootstrap_iterations'], 'saved_seed': manifest['seed'],
            'minimum_primary_origins': manifest['config']['minimum_primary_origins'],
            'bootstrap_positive_share_is_p_value': False, 'diebold_mariano': 'NOT_ESTIMATED',
            'new_bootstrap_runs_during_integration': 0},
        'interpretation': {'stable_advantage_across_splits': False, 'conclusion': CONCLUSION,
            'concentration': 'Положительное улучшение не ограничивается несколькими МО, '
                'но при h=3/6 положительные и отрицательные вклады почти компенсируются. '
                'Отсутствие влияния выбросов не установлено.'},
        'limitations': LIMITATIONS,
        'verification': {'status': 'PASS', 'comparable_cases_per_pair': 1890,
            'evaluable_municipalities': 63, 'tolerance_absolute': gate['tolerance_absolute'],
            'maximum_metric_difference': gate['maximum_metric_difference'], 'saved_model_fits': 0,
            'holdout_previously_inspected': True, 'new_experiments_during_integration': False},
        'sources': [{'path': name, 'sha256': text_sha(root / name),
                     'hash_basis': 'UTF-8 text normalized to LF, without BOM'} for name in source_names]}


def render_summary(block: dict) -> str:
    rows = block['primary_pair']['records']
    lines = ['### Устойчивость результатов прогнозирования', '',
        'National/Local + LightGBM сравнивается с SeasonalNaiveYoY на одинаковых прогнозных '
        'случаях у 63 оцениваемых МО. Validation и holdout рассматриваются отдельно; '
        'основные горизонты — 1/3/6 месяцев. ΔMAE = MAE baseline − MAE candidate: '
        'положительное значение означает меньшую ошибку National/Local + LightGBM.', '',
        '| Горизонт | Holdout ΔMAE, руб. | Доля МО с меньшей MAE | Выигрыши по origins | Описательный 95% origin-интервал, руб. |',
        '| --- | ---: | ---: | ---: | --- |']
    for row in rows:
        if row['split'] != 'holdout':
            continue
        ci = row['origin_bootstrap']
        lines.append(f"| h={row['horizon']} | {row['delta_mae']:+.2f} | {row['municipality_win_rate'] * 100:.2f}% | "
                     f"{row['origin_wins']}/{row['n_origins']} | [{ci['lower']:.2f}; {ci['upper']:.2f}] |")
    val = [r for r in rows if r['split'] == 'validation']
    deltas = ' / '.join(f"{r['delta_mae']:.2f}" for r in val)
    lines += ['', 'На holdout h=1 меньшая ошибка наблюдается у большинства МО и в пяти из шести '
        'origins; описательный интервал полностью положителен. На h=3/6 интервалы широкие '
        'и включают ноль. На validation ΔMAE составляет ' + deltas + ' руб. для h=1/3/6: '
        'преимущество не воспроизводится. Устойчивое преимущество National/Local + LightGBM '
        'над SeasonalNaiveYoY между периодами оценки не подтверждено. '
        'Всего 4–6 origins, а на validation h=6 — одна, поэтому интервалы являются описательной '
        'оценкой чувствительности; для одной даты интервал не оценивается. Положительное улучшение '
        'не ограничивается несколькими МО, но на h=3/6 положительные и отрицательные вклады '
        'почти компенсируются.', '', block['prophet_conclusion'], '',
        'Источники: [исследовательский отчёт](../results/forecast_robustness.md), '
        '[парные метрики](../results/forecast_robustness/pairwise_metrics.csv), '
        '[интервалы по датам](../results/forecast_robustness/bootstrap_origin.csv), '
        '[методика](METHODOLOGY_REPORT.md#51-устойчивость-сравнения-прогнозов). '
        'Эта проверка дополняет ablation разложения и не изменяет её результаты.']
    return '\n'.join(lines)


def refresh_forecasting_robustness(root: Path, destination='reports/final', *, check=False) -> dict:
    root = root.resolve()
    out = (root / destination).resolve()
    if not out.is_relative_to((root / 'reports/final').resolve()):
        raise ValueError('Robustness refresh can only write inside reports/final')
    path = out / 'results_summary.json'
    summary = json.loads(path.read_text(encoding='utf-8-sig'))
    block = collect_forecasting_robustness(root, summary)
    markdown = out / 'RESULTS_SUMMARY.md'
    content = markdown.read_text(encoding='utf-8-sig')
    begin, end = '<!-- forecasting-robustness:begin -->', '<!-- forecasting-robustness:end -->'
    if content.count(begin) != 1 or content.count(end) != 1:
        raise ValueError('Summary requires one robustness marker pair')
    rendered = re.sub(re.escape(begin) + r'.*?' + re.escape(end),
                      lambda _: begin + '\n' + render_summary(block) + '\n' + end, content, flags=re.S)
    if check:
        if summary.get('forecasting_robustness') != block or content != rendered:
            raise ValueError('Published forecasting robustness is stale; rebuild with --forecasting-robustness-only')
    else:
        summary['forecasting_robustness'] = block
        encoded = json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
        if path.read_text(encoding='utf-8-sig') != encoded:
            path.write_text(encoded, encoding='utf-8', newline='\n')
        if content != rendered:
            markdown.write_text(rendered, encoding='utf-8', newline='\n')
    return {'status': 'PASS', 'primary_comparison_records': len(block['primary_pair']['records']),
            'main_benchmark_changed': False, 'model_fitting': False, 'new_experiments': False}
