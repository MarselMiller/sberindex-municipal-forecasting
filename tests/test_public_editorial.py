"""Public wording must preserve every table and research addition; no fits."""
from pathlib import Path
import json
import re

from sberforecast.final_render import public_editorial, refresh_public_editorial

ROOT = Path(__file__).resolve().parents[1]


def test_editorial_preserves_numeric_tables_research_blocks_and_limitations():
    summary = json.loads((ROOT / 'reports/final/results_summary.json').read_text(encoding='utf-8-sig'))
    current = (ROOT / 'reports/final/RESULTS_SUMMARY.md').read_text(encoding='utf-8-sig')
    old = current.replace('## Прогнозирование: сравнение моделей на общей выборке',
                          '## A. Forecasting: сопоставимый реальный пилот')
    old += '\n### Обнаруженные несовпадения и подготовленная формулировка\n\nНЕ ЗАПУСКАЛОСЬ\nAI disclosure draft\n'
    revised = public_editorial(old, summary)
    assert revised == public_editorial(revised, summary)
    assert re.findall(r'^\|.*$', revised, re.MULTILINE) == re.findall(r'^\|.*$', current, re.MULTILINE)
    assert 'для 63 доступны оцениваемые факты' in revised
    assert 'все 2190 МО' in revised
    assert '7 строк без конечного факта' in revised
    assert 'SeasonalNaive fallback' in revised and 'news-корпус ограничен' in revised
    assert '### Leading financial indicators' in revised
    assert 'Роль разложения National/Local' in revised
    assert re.search(r'<!-- forecasting-robustness:begin -->.*?<!-- forecasting-robustness:end -->', revised, re.DOTALL)[0] == re.search(r'<!-- forecasting-robustness:begin -->.*?<!-- forecasting-robustness:end -->', current, re.DOTALL)[0]
    assert not any(term in revised for term in ['Обнаруженные несовпадения', 'НЕ ЗАПУСКАЛОСЬ', 'pilot holdout', 'AI disclosure'])


def test_current_summary_is_generated_editorial_output():
    assert refresh_public_editorial(ROOT, check=True)['status'] == 'PASS'
