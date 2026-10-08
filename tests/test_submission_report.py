"""Synthetic HTML fixtures protect repeatable editorial builds and existing content."""
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def builder():
    path = Path(__file__).resolve().parents[1] / 'scripts/build_project_report.py'
    spec = importlib.util.spec_from_file_location('submission_report_fixture', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_repeated_editorial_build_preserves_tables_definitions_and_assets(builder):
    protected = ('<table><tr><td>1.25</td><td>NA</td></tr></table>'
                 '<details id="forecast-origin">Synthetic canonical definition</details>'
                 '<input id="glossary-search"><select id="forecast-horizon"></select>')
    figure = '<figure class="forecast-example"><img src="assets/figure.png"></figure>'
    toolbar = '<div class="chart-toolbar enhanced-control"><label class="select-label">Показатель</label></div>'
    first = builder.render_editorial_text(protected + figure + toolbar)
    assert builder.render_editorial_text(first) == first
    assert protected in first and figure in first and toolbar in first
    assert first.count('id="real-detection-example"') == 1
    assert first.count('id="detector-selection-note"') == 1
    assert 'href="../reports/results/E06a_offline_detection.md"' in first
    assert 'warmup' in first and 'Это не предупреждение' in first


def test_missing_forecast_section_fails_instead_of_silently_losing_case(builder):
    with pytest.raises(ValueError, match='existing real forecast example'):
        builder.render_editorial_text('<main>Incomplete synthetic template</main>')
