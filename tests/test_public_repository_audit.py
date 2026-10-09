"""Synthetic audit fixtures only; never expose real secret values."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('repository_audit', ROOT / 'scripts/audit_public_repository.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_real_row_header_blocks_but_explicit_synthetic_demo_is_marked():
    data = b'municipality_id,y_true,y_pred\nsynthetic,1,2\n'
    assert any(f['severity'] == 'BLOCKER' for f in audit.inspect('reports/rows.csv', data))
    assert all(f['severity'] == 'INFO' for f in audit.inspect('data/synthetic/e07c_demo/observations.csv', data))


def test_secret_value_is_never_in_findings():
    value = b'ghp_' + b'x' * 36
    found = audit.inspect('example.py', value)
    assert found[0]['category'] == 'github_token'
    assert value.decode() not in str(found)


def test_only_named_rejected_fixture_has_dummy_url_exception():
    value = b'https://user:' + b'password' + b'@cbr.ru/example'
    assert audit.inspect('tests/test_macro_cbr.py', value)[0]['severity'] == 'INFO'
    assert audit.inspect('example.py', value)[0]['severity'] == 'BLOCKER'


def test_absolute_paths_are_counted_without_recording_values():
    data = bytes([67, 58, 92]) + b'Users' + bytes([92]) + b'synthetic' + bytes([92]) + b'example.json'
    assert audit.inspect('manifest.json', data) == [{'category': 'absolute_user_path', 'severity': 'REVIEW', 'count': 1}]


def test_reviewed_rows_exception_is_bound_to_content_and_path():
    import hashlib
    data = b'municipality_id,y_true,y_pred\nsynthetic,1,2\n'
    name = 'reports/final/figure_data/rolling_forecast.csv'
    reviewed = {name: hashlib.sha256(data).hexdigest()}
    assert audit.inspect(name, data, reviewed)[0]['severity'] == 'REVIEW'
    assert audit.inspect(name, data.replace(b',2', b',3'), reviewed)[0]['severity'] == 'BLOCKER'
    assert audit.inspect('reports/other.csv', data, reviewed)[0]['severity'] == 'BLOCKER'
    # A rights decision must not suppress a secret finding in the same file.
    bad = data + b'ghp_' + b'x' * 36
    assert any(f['category'] == 'github_token' and f['severity'] == 'BLOCKER'
               for f in audit.inspect(name, bad, {name: hashlib.sha256(bad).hexdigest()}))
