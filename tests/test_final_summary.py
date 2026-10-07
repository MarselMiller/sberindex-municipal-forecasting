"""F1 checks on synthetic fixtures; no real data/model-fitting dependencies."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from sberforecast.final_summary import (
    DATA_STATUSES, clean_json, criteria, forecasting_table, output_path,
    validate_records, verified_components, write_json,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('status', sorted(DATA_STATUSES))
def test_every_allowed_data_status_is_preserved(status):
    records = [dict(data_status=status)]
    validate_records(records, 'synthetic_fixture')
    assert records[0]['data_status'] == status


@pytest.mark.parametrize('row', [{}, {'data_status': 'real+synthetic'}, {'data_status': None}])
def test_missing_or_ambiguous_data_status_fails(row):
    with pytest.raises(ValueError, match='data_status'):
        validate_records([row], 'fixture')


@pytest.mark.parametrize('status', ['diagnostic', 'not_evaluated'])
@pytest.mark.parametrize('field', ['mae_macro', 'pr_auc', 'precision', 'row_f1', 'event_recall'])
def test_unmeasured_quality_must_not_be_zero(status, field):
    with pytest.raises(ValueError, match='must be null'):
        validate_records([{'data_status': status, field: 0}], 'fixture')
    validate_records([{'data_status': status, field: None}], 'fixture')


def test_standard_json_undefined_metrics_are_null(tmp_path):
    data = {'undefined': float('nan'), 'nested': [float('inf'), None, .25]}
    write_json(tmp_path / 'metrics.json', data)
    assert json.loads((tmp_path / 'metrics.json').read_text()) == {'undefined': None, 'nested': [None, None, .25]}
    assert 'NaN' not in (tmp_path / 'metrics.json').read_text()


@pytest.mark.parametrize('unsafe', ['reports/results', 'outputs/baseline_v1', '../elsewhere', 'reports/final/../../results'])
def test_builder_cannot_write_frozen_artifacts(tmp_path, unsafe):
    with pytest.raises(ValueError, match='only write'):
        output_path(tmp_path, unsafe)


def test_csv_summary_table_includes_explicit_annual_fallback():
    rows = [dict(model='Direct', split='holdout', horizon=h, mae_macro=11.125, fallback_count=1 if h == 12 else 0)
            for h in (1, 3, 6, 12)]
    rendered = forecasting_table(rows, 'holdout')
    assert rendered.count('†') == 1
    assert '11.12†' in rendered


def test_criteria_weights_and_evidence_are_not_jury_scores():
    mapping = criteria()
    assert [row['weight'] for row in mapping] == [10, 20, 20, 15, 15, 10, 10]
    assert sum(row['weight'] for row in mapping) == 100
    assert {row['status'] for row in mapping} <= {'completed', 'partially_completed', 'limited_by_data'}
    assert all(row['artifact'] and row['limitation'] for row in mapping)
    assert not any('score' in row for row in mapping)


def test_real_early_warning_is_not_evaluated_but_foundation_is_real():
    components = verified_components()
    assert next(row for row in components if row['component'] == 'Real early warning')['data_status'] == 'not_evaluated'
    assert next(row for row in components if row['component'] == 'Foundation model')['data_status'] == 'real'
    assert {row['data_status'] for row in components if row['component'] == 'Offline detection'} == {'synthetic', 'diagnostic'}


def test_entire_f1_import_and_call_graph_has_no_models_or_fitting():
    files = list((ROOT / 'src/sberforecast').glob('final_*.py')) + [ROOT / 'scripts/build_final_summary.py']
    forbidden_modules = {'catboost', 'lightgbm', 'sklearn', 'scipy', 'torch', 'chronos', 'ruptures', 'requests', 'urllib', 'httpx'}
    forbidden_calls = {'fit', 'train', 'predict', 'minimize', 'baseline_predict', 'generate_cohorts', 'run_experiment', 'urlopen'}
    for path in files:
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                modules = [node.module or ''] if isinstance(node, ast.ImportFrom) else [item.name for item in node.names]
                for module in modules:
                    assert module.split('.')[0] not in forbidden_modules, (path, module)
                    if module.startswith('sberforecast.'):
                        assert module.split('.')[1].startswith('final_'), (path, module)
                    if isinstance(node, ast.ImportFrom) and node.level:
                        assert module.startswith('final_'), (path, module)
            elif isinstance(node, ast.Call):
                name = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else ''
                assert name not in forbidden_calls, (path, name)
