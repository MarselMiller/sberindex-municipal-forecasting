"""Synthetic regression for figure cohort selection, without any saved real data."""
from __future__ import annotations

import pandas as pd
import pytest

from sberforecast.final_figures import select_test_event_metrics


def fixture():
    return pd.DataFrame([dict(cohort=c, model=m, k=k, eligible_events=12 if c == 'test' else 8,
                              event_recall=.25 if c == 'test' else .75, alert_precision=None if m == 'S0' else .5)
                         for c in ('validation', 'test') for m in ('S0', 'S1', 'S2', 'S3') for k in (1, 3)])


def test_event_figure_uses_eight_test_rows_and_preserves_undefined_s0_precision():
    source = fixture()
    measured = select_test_event_metrics(source)
    assert len(source) == 16 and len(measured) == 8
    assert measured.cohort.eq('test').all()
    assert measured.eligible_events.eq(12).all()
    assert measured.event_recall.eq(.25).all()
    assert measured.loc[measured.model.eq('S0'), 'alert_precision'].isna().all()
    assert len(source) == 16  # The frozen source is not mutated.


@pytest.mark.parametrize('mutation', ['no_cohort', 'no_test', 'duplicate', 'denominator'])
def test_incomplete_ambiguous_or_inconsistent_figure_provenance_fails(mutation):
    source = fixture()
    if mutation == 'no_cohort':
        source = source.drop(columns='cohort')
    elif mutation == 'no_test':
        source = source.loc[source.cohort.ne('test')]
    elif mutation == 'duplicate':
        source = pd.concat([source, source.loc[source.cohort.eq('test')].iloc[[0]]])
    else:
        source.loc[source.cohort.eq('test') & source.model.eq('S3'), 'eligible_events'] = 13
    with pytest.raises(ValueError, match='figure'):
        select_test_event_metrics(source)
