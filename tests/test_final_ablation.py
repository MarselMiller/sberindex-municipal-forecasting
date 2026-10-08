"""Synthetic saved aggregates: additive publication, missingness and integrity."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from sberforecast import final_ablation as module


@pytest.fixture
def saved_ablation(tmp_path):
    directory = tmp_path / module.SOURCE_DIR
    directory.mkdir(parents=True)
    report = tmp_path / module.REPORT
    report.write_text('Synthetic verification fixture\n', encoding='utf-8')
    metrics, gaps = [], []
    for split in ('validation', 'holdout'):
        for h in (1, 3, 6, 12):
            if split == 'validation' and h == 12:
                continue
            for model in module.MODEL_LABELS:
                metrics.append(dict(split=split, horizon=h, model=model, mae_macro=20.0,
                                    mae_micro=20.0, r2_pooled=.5, n_predictions=1,
                                    n_municipalities=1, n_origins=1, descriptive_only=h == 12))
            for before, after in [('LastValue', 'NationalLocalPersistence'),
                                  ('SeasonalNaiveYoY', 'NationalLocalPersistence'),
                                  ('LightGBMDirect', 'NationalLocalPersistence'),
                                  ('NationalLocalPersistence', 'NationalLocalLightGBM')]:
                gaps.append(dict(split=split, horizon=h, from_model=before, to_model=after,
                                 observed_share_of_mae_gap=None if h == 12 else 1.5,
                                 reduction_pct=-2.5, descriptive_only=h == 12))
    for filename, rows in [('metrics.csv', metrics), ('gap_analysis.csv', gaps)]:
        with (directory / filename).open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    for name in module.CSV_FILES[2:]:
        (directory / name).write_text('synthetic_aggregate\n1\n', encoding='utf-8')
    manifest = {'reproduction_gate': {'status': 'PASS', 'tolerance_absolute': 1e-8,
                                    'saved_metric_checks': {'metrics_strategy': {'max_abs_delta': 0.0}}},
                'n_model_fits': 0, 'n_model_fit_attempts': 0, 'no_tuning': True, 'no_feature_search': True,
                'artifact_sha256': {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                                    for name in module.CSV_FILES},
                'report_sha256': hashlib.sha256(report.read_bytes()).hexdigest(),
                'interpretation': {'overall': 'D', 'label': 'SYNTHETIC FIXTURE', 'per_horizon': []}}
    (directory / 'run_manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    out = tmp_path / 'reports/final'
    out.mkdir()
    original = {'schema_version': 'existing', 'forecasting': {'records': []},
                'detection': {'precision': None}, 'historic_numbers': [799.55, 1212.73, 1897.22]}
    (out / 'results_summary.json').write_text(json.dumps(original), encoding='utf-8')
    (out / 'forecasting_metrics.csv').write_text('untouched_main_benchmark\n', encoding='utf-8')
    return tmp_path, original


def test_refresh_preserves_every_existing_value_and_writes_only_summary(saved_ablation):
    root, original = saved_ablation
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    result = module.refresh_forecasting_ablation(root)
    current = json.loads((root / 'reports/final/results_summary.json').read_text(encoding='utf-8'))
    assert {k: v for k, v in current.items() if k != 'forecasting_ablation'} == original
    assert result['main_benchmark_changed'] is False
    after = {p.relative_to(root): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    assert set(before) == set(after)
    assert [p.as_posix() for p in before if before[p] != after[p]] == ['reports/final/results_summary.json']
    first = after[Path('reports/final/results_summary.json')]
    module.refresh_forecasting_ablation(root)
    assert (root / 'reports/final/results_summary.json').read_bytes() == first
    module.refresh_forecasting_ablation(root, check=True)


def test_missing_share_remains_null_and_values_above_one_remain_unclipped(saved_ablation):
    root, _ = saved_ablation
    ablation = module.collect_forecasting_ablation(root)
    assert {r['observed_share_of_mae_gap'] for r in ablation['gap_analysis']} == {None, 1.5}
    assert all(r['reduction_pct'] == -2.5 for r in ablation['gap_analysis'])
    assert ablation['included_in_main_benchmark'] is False
    assert len(ablation['metrics']) == 35 and len(ablation['gap_analysis']) == 28
    assert ablation['primary_horizons'] == [1, 3, 6] and ablation['descriptive_horizons'] == [12]


@pytest.mark.parametrize('name', [*module.CSV_FILES, '../national_local_persistence.md'])
def test_changed_sealed_source_rejected_before_publication(saved_ablation, name):
    root, _ = saved_ablation
    path = root / module.SOURCE_DIR / name
    path.write_text(path.read_text(encoding='utf-8') + 'tampered\n', encoding='utf-8')
    with pytest.raises(ValueError, match='SHA mismatch'):
        module.refresh_forecasting_ablation(root)
    assert 'forecasting_ablation' not in json.loads((root / 'reports/final/results_summary.json').read_text())


@pytest.mark.parametrize('key,value', [('n_model_fits', 1), ('n_model_fit_attempts', 1),
                                      ('no_tuning', False), ('no_feature_search', False)])
def test_unapproved_method_metadata_rejected(saved_ablation, key, value):
    root, _ = saved_ablation
    path = root / module.SOURCE_DIR / 'run_manifest.json'
    manifest = json.loads(path.read_text())
    manifest[key] = value
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='fixed-method gate'):
        module.refresh_forecasting_ablation(root)


def test_main_reference_disagreement_rejected(saved_ablation):
    root, _ = saved_ablation
    path = root / 'reports/final/results_summary.json'
    summary = json.loads(path.read_text())
    summary['forecasting']['records'] = [dict(split='holdout', horizon=1, model='LightGBMDirect', mae_macro=999)]
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match='differs from the main benchmark'):
        module.refresh_forecasting_ablation(root)


def test_check_detects_missing_object_without_writing(saved_ablation):
    root, _ = saved_ablation
    path = root / 'reports/final/results_summary.json'
    before = path.read_bytes()
    with pytest.raises(ValueError, match='stale'):
        module.refresh_forecasting_ablation(root, check=True)
    assert path.read_bytes() == before


def test_destination_cannot_escape_final_scope(saved_ablation):
    root, _ = saved_ablation
    with pytest.raises(ValueError, match='only write inside'):
        module.refresh_forecasting_ablation(root, 'reports/results')


@pytest.fixture
def report_builder():
    path = Path(__file__).resolve().parents[1] / 'scripts/build_project_report.py'
    spec = importlib.util.spec_from_file_location('project_report_fixture', path)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    return builder


def synthetic_display_ablation(root):
    ablation = module.collect_forecasting_ablation(root)
    for row in ablation['gap_analysis']:
        row['share_status'] = 'estimable_descriptive'
        row['metric_status'] = 'complete'
        if row['split'] == 'holdout' and row['from_model'] == 'LightGBMDirect' and row['horizon'] != 12:
            row['observed_share_of_mae_gap'] = {1: .6832, 3: .6924, 6: .7626}[row['horizon']]
    return ablation


def test_report_builder_selects_only_primary_holdout_gap_not_validation_or_annual(saved_ablation, report_builder):
    root, _ = saved_ablation
    ablation = synthetic_display_ablation(root)
    fragment = report_builder.render_forecasting_ablation(ablation)
    assert [r['observed_share_of_mae_gap'] for r in report_builder.persistence_gap_rows(ablation)] == [.6832, .6924, .7626]
    assert all(value in fragment for value in ['68%', '69%', '76%'])
    assert '150%' not in fragment  # Synthetic validation shares above one remain outside this holdout block.
    assert 'validation' in fragment and 'SeasonalNaiveYoY' in fragment


@pytest.mark.parametrize('field,value', [('included_in_main_benchmark', True), ('data_status', 'synthetic'),
                                        ('primary_horizons', [1, 3, 6, 12]), ('descriptive_horizons', [])])
def test_report_builder_rejects_changed_ablation_scope(saved_ablation, report_builder, field, value):
    root, _ = saved_ablation
    ablation = synthetic_display_ablation(root)
    ablation[field] = value
    with pytest.raises(ValueError, match='separate from the nine-strategy benchmark'):
        report_builder.persistence_gap_rows(ablation)


def test_report_builder_duplicate_primary_case_is_not_silently_selected(saved_ablation, report_builder):
    root, _ = saved_ablation
    ablation = synthetic_display_ablation(root)
    duplicate = report_builder.persistence_gap_rows(ablation)[0].copy()
    ablation['gap_analysis'].append(duplicate)
    with pytest.raises(ValueError, match='exactly one'):
        report_builder.persistence_gap_rows(ablation)
