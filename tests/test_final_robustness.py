"""Synthetic saved aggregates: sign, scope, missingness and additive publication."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from sberforecast import final_robustness as module


@pytest.fixture
def saved_robustness(tmp_path):
    directory = tmp_path / module.SOURCE_DIR
    directory.mkdir(parents=True)
    report = tmp_path / module.REPORT
    report.write_text('Synthetic audit fixture\n', encoding='utf-8')
    tables = {name: [] for name in module.CSV_FILES}
    main = []
    models = sorted(set(m for pair in module.PAIRS.values() for m in pair)) + ['Other1', 'Other2', 'Other3', 'Other4']
    def mae(model, split):
        return {'SeasonalNaiveYoY': 100., 'National/Local + LightGBM': 110. if split == 'validation' else 90.,
                'ProphetAuto': 120., 'ProphetYearly': 130., 'LightGBMDirect': 140.}.get(model, 150.)
    for split in ('validation', 'holdout'):
        for h in (1, 3, 6, 12):
            n = 6 if split == 'holdout' and h != 12 else {1: 6, 3: 4, 6: 1, 12: 1}[h]
            for model in models:
                main.append(dict(model=model, split=split, horizon=h, mae_macro=mae(model, split),
                                 n_predictions=63*n, n_municipalities=63, n_origins=n, fallback_count=0))
            if (split, h) == ('validation', 12):
                continue
            for pair, (baseline, candidate) in module.PAIRS.items():
                common = dict(pair=pair, baseline=baseline, candidate=candidate, split=split,
                              horizon=h, descriptive_only=h == 12, delta_convention=module.SIGN)
                delta = mae(baseline, split) - mae(candidate, split)
                metric = dict(common, n_comparable_rows=63*n, n_municipalities=63, n_origins=n,
                              mae_baseline=mae(baseline, split), mae_candidate=mae(candidate, split),
                              delta_mae=delta, baseline_fallback_count=0, candidate_fallback_count=0)
                tables['pairwise_metrics.csv'].append(metric)
                for name, units in [('municipality_summary.csv', 63), ('origin_summary.csv', n)]:
                    wins = units if delta > 0 else 0
                    tables[name].append(dict(common, n_units=units, candidate_wins=wins,
                        candidate_losses=units-wins, ties=0, candidate_win_rate=wins/units, mean_delta_mae=delta))
                status = ('DESCRIPTIVE_ONLY_H12' if h == 12 else 'NOT_ESTIMABLE_ONE_ORIGIN' if n == 1
                          else 'DESCRIPTIVE_EMPIRICAL_SENSITIVITY')
                for name in ('bootstrap_origin.csv', 'bootstrap_municipality.csv'):
                    tables[name].append(dict(common, n_origins=n, observed_delta_mae=delta,
                        ci_lower=delta-2 if status == 'DESCRIPTIVE_EMPIRICAL_SENSITIVITY' else '',
                        ci_upper=delta+2 if status == 'DESCRIPTIVE_EMPIRICAL_SENSITIVITY' else '', interval_status=status))
                for i in range(n):
                    tables['origin_deltas.csv'].append(dict(common, forecast_origin=f'synthetic-{i}',
                                                          n_cases=63, delta_mae=delta))
                tables['concentration.csv'].append(common)
                tables['applicability_checks.csv'].append(dict(common, dm_status='NOT_RELIABLE_NOT_ESTIMATED',
                                                               primary_ci_status=status))
    for name, rows in tables.items():
        with (directory / name).open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    manifest = dict(comparability_gate=dict(status='PASS', tolerance_absolute=1e-8, maximum_metric_difference=0.,
        pairs=[dict(pair=p, baseline=a, candidate=b, gate_status='PASS', comparable_rows=1890, n_municipalities=63)
               for p, (a, b) in module.PAIRS.items()]), n_model_fits=0, new_predictions=0,
        no_tuning=True, no_feature_search=True, no_primary_subset_selection=True,
        bootstrap_positive_share_is_p_value=False, independent_verification=dict(status='PASS'),
        config=dict(primary_horizons=[1, 3, 6], descriptive_horizons=[12],
                    delta_convention='baseline_mae_minus_candidate_mae', minimum_primary_origins=4),
        delta_convention=module.SIGN, bootstrap_iterations=10000, seed=42,
        artifact_sha256={name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in tables},
        post_run_report_review=dict(report_sha256=hashlib.sha256(report.read_bytes()).hexdigest()))
    (directory / 'run_manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    original = dict(forecasting=dict(records=main), forecasting_ablation=dict(unchanged=[.6832, None]),
                    detection=dict(precision=None), old_metrics=[799.55, 1212.73, 1897.22])
    out = tmp_path / 'reports/final'
    out.mkdir(parents=True)
    (out / 'results_summary.json').write_text(json.dumps(original), encoding='utf-8')
    (out / 'RESULTS_SUMMARY.md').write_text('Existing results\n<!-- forecasting-robustness:begin -->\n'
                                          '<!-- forecasting-robustness:end -->\nExisting ablation\n', encoding='utf-8')
    (out / 'forecasting_metrics.csv').write_text('unchanged historical benchmark\n', encoding='utf-8')
    return tmp_path, original


def test_additive_refresh_preserves_all_old_values_and_sources(saved_robustness):
    root, original = saved_robustness
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    module.refresh_forecasting_robustness(root)
    current = json.loads((root / 'reports/final/results_summary.json').read_text())
    assert {k: v for k, v in current.items() if k != 'forecasting_robustness'} == original
    changed = {p.as_posix() for p, content in before.items() if (root / p).read_bytes() != content}
    assert changed == {'reports/final/results_summary.json', 'reports/final/RESULTS_SUMMARY.md'}
    assert current['forecasting_robustness']['included_in_main_benchmark'] is False
    module.refresh_forecasting_robustness(root, check=True)


def test_sign_split_scope_and_one_origin_missingness(saved_robustness):
    root, summary = saved_robustness
    block = module.collect_forecasting_robustness(root, summary)
    rows = {(r['split'], r['horizon']): r for r in block['primary_pair']['records']}
    assert len(rows) == 6 and {h for _, h in rows} == {1, 3, 6}
    assert rows['validation', 1]['delta_mae'] == -10.
    assert rows['holdout', 1]['delta_mae'] == 10.
    ci = rows['validation', 6]['origin_bootstrap']
    assert ci['status'] == 'NOT_ESTIMABLE_ONE_ORIGIN'
    assert ci['lower'] is ci['upper'] is ci['includes_zero'] is None
    assert block['descriptive_horizons'] == [12]
    assert [r['baseline'] for r in block['prophet_comparisons']] == ['ProphetAuto', 'ProphetYearly']
    assert block['interpretation']['stable_advantage_across_splits'] is False


@pytest.mark.parametrize('name', [*module.CSV_FILES, '../forecast_robustness.md'])
def test_tampered_sealed_source_is_rejected(saved_robustness, name):
    root, summary = saved_robustness
    path = root / module.SOURCE_DIR / name
    path.write_text(path.read_text() + 'tampered\n')
    with pytest.raises(ValueError, match='SHA mismatch'):
        module.collect_forecasting_robustness(root, summary)


@pytest.mark.parametrize('key,value', [('n_model_fits', 1), ('new_predictions', 1),
    ('no_tuning', False), ('no_feature_search', False), ('no_primary_subset_selection', False),
    ('bootstrap_positive_share_is_p_value', True), ('delta_convention', 'candidate minus baseline')])
def test_changed_method_or_sign_gate_is_rejected(saved_robustness, key, value):
    root, summary = saved_robustness
    path = root / module.SOURCE_DIR / 'run_manifest.json'
    manifest = json.loads(path.read_text())
    manifest[key] = value
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='fixed-method gate'):
        module.collect_forecasting_robustness(root, summary)


def test_changed_old_mae_or_duplicate_strategy_is_rejected(saved_robustness):
    root, summary = saved_robustness
    row = next(r for r in summary['forecasting']['records'] if r['model'] == 'SeasonalNaiveYoY')
    row['mae_macro'] += 1
    with pytest.raises(ValueError, match='differs from its reference'):
        module.collect_forecasting_robustness(root, summary)
    summary['forecasting']['records'].append(dict(row))
    with pytest.raises(ValueError, match='nine strategies'):
        module.collect_forecasting_robustness(root, summary)


def test_check_is_read_only_and_detects_missing_block(saved_robustness):
    root, _ = saved_robustness
    path = root / 'reports/final/results_summary.json'
    before = path.read_bytes()
    with pytest.raises(ValueError, match='stale'):
        module.refresh_forecasting_robustness(root, check=True)
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match='only write inside'):
        module.refresh_forecasting_robustness(root, 'reports/results')


def test_html_and_markdown_display_only_saved_holdout_numbers(saved_robustness):
    root, summary = saved_robustness
    block = module.collect_forecasting_robustness(root, summary)
    path = Path(__file__).resolve().parents[1] / 'scripts/build_project_report.py'
    spec = importlib.util.spec_from_file_location('robustness_report_fixture', path)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    html = builder.render_forecasting_robustness(block)
    assert html.count('<tr>') == 4 and html.count('10,00') == 3
    assert 'validation преимущество не воспроизводится' in html
    assert 'h = 12' not in html
    markdown = module.render_summary(block)
    assert markdown.count('| +10.00 |') == 3
    block['primary_pair']['records'].append(dict(block['primary_pair']['records'][0]))
    with pytest.raises(ValueError, match='separate validation and holdout'):
        builder.render_forecasting_robustness(block)
