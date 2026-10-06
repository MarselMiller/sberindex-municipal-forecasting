"""Synthetic event matching and whole-series uncertainty checks for E04a."""
import json

import numpy as np
import pandas as pd
import pytest

from sberforecast.online_evaluation import aggregate_metrics, bootstrap_metrics, evaluate_series


def example(alarms=(), events=("2024-06",), missing=(), lag=0, end="2024-12", event_ids=None):
    periods = pd.period_range("2024-01", end, freq="M")
    rows = []
    n_available = 0
    for period in periods:
        available = (period + lag).to_timestamp(how="end").normalize()
        phase = "missing" if str(period) in missing else "warmup" if n_available < 4 else "monitoring"
        n_available += int(phase != "missing")
        is_alarm = str(period) in alarms
        rows.append(dict(series_id="a", observation_period=str(period), availability_date=available,
                         phase=phase, is_alarm=is_alarm, signal_date=available if is_alarm else pd.NaT))
    signals = pd.DataFrame(rows)
    truths = pd.DataFrame([dict(series_id="a", event_id=event_ids[index] if event_ids else str(index),
                                event_type="level", event_start_period=period, strength=0.1)
                           for index, period in enumerate(events)],
                          columns=["series_id", "event_id", "event_type", "event_start_period", "strength"])
    info = pd.DataFrame([dict(series_id="a", scenario="level_up" if events else "no_change", strength=0.1,
                              noise_fraction=0.01, split="test", observation_start_period="2024-01",
                              observation_end_period=end, release_lag_months=lag)])
    return signals, truths, info


def test_inclusive_three_month_window_and_repeated_alarm_false():
    per, events, alarms = evaluate_series(*example(alarms=("2024-09", "2024-10")))
    assert (per.iloc[0].tp, per.iloc[0].fp, per.iloc[0].fn) == (1, 1, 0)
    assert events.iloc[0].status == "detected" and events.iloc[0].delay_months == 3
    assert alarms.status.tolist() == ["true_positive", "false_positive"]
    assert json.loads(per.iloc[0].delays_months) == [3]
    metrics = aggregate_metrics(per).iloc[0]
    assert metrics.precision == 0.5 and metrics.recall == 1
    assert metrics.f1 == pytest.approx(2 / 3)
    assert metrics.n_monitoring_months == 8 and metrics.false_alarms_per_12_months == 1.5


def test_pre_event_alarm_cannot_be_detection_or_early_warning():
    per, events, alarms = evaluate_series(*example(alarms=("2024-05",)))
    assert (per.iloc[0].tp, per.iloc[0].fp, per.iloc[0].fn) == (0, 1, 1)
    assert events.iloc[0].status == "missed"
    assert alarms.iloc[0].status == "false_positive"
    metrics = aggregate_metrics(per).iloc[0]
    assert metrics.recall == 0 and metrics.missed_fraction == 1 and np.isnan(metrics.median_delay)


def test_one_alarm_matches_one_event_earliest_deadline_then_later_event():
    per, events, alarms = evaluate_series(*example(alarms=("2024-07", "2024-08"), events=("2024-06", "2024-07")))
    assert per.iloc[0].tp == 2
    assert events.matched_alarm_id.nunique() == 2
    assert alarms.matched_event_id.tolist() == ["0", "1"]
    assert alarms.delay_months.tolist() == [1, 1]


def test_one_event_receives_only_one_alarm_inside_window():
    per, _, alarms = evaluate_series(*example(alarms=("2024-06", "2024-07", "2024-08")))
    assert per.iloc[0].tp == 1 and per.iloc[0].fp == 2
    assert alarms.status.tolist() == ["true_positive", "false_positive", "false_positive"]


def test_calendar_missing_month_does_not_compress_delay_or_denominator():
    per, events, _ = evaluate_series(*example(alarms=("2024-09",), missing=("2024-07", "2024-08")))
    assert per.iloc[0].n_monitoring_months == 6
    assert events.iloc[0].delay_months == 3
    assert events.iloc[0].window_end_period == "2024-09"


def test_publication_lag_uses_issue_date_instead_of_observation_month():
    per, events, _ = evaluate_series(*example(alarms=("2024-08",), lag=2))
    assert per.iloc[0].fn == 1 and per.iloc[0].fp == 1
    assert events.iloc[0].status == "missed"  # Issue October, outside June..September.
    per, events, _ = evaluate_series(*example(alarms=("2024-07",), lag=2))
    assert per.iloc[0].tp == 1 and events.iloc[0].delay_months == 3


def test_first_monitoring_event_remains_eligible_with_publication_lag():
    per, events, _ = evaluate_series(*example(alarms=("2024-05",), events=("2024-05",), lag=1))
    assert per.iloc[0].tp == 1
    assert events.iloc[0].eligibility == "full_window" and events.iloc[0].delay_months == 1


def test_partial_end_window_and_alarms_preserved_but_excluded():
    per, events, alarms = evaluate_series(*example(alarms=("2024-11", "2024-12"), events=("2024-11",)))
    assert events.iloc[0].eligibility == "partial_window" and events.iloc[0].status == "excluded"
    assert per.iloc[0].n_events_total == 1 and per.iloc[0].n_events == 0
    assert per.iloc[0].n_events_excluded == 1 and per.iloc[0].n_alarms_excluded == 2
    assert alarms.status.eq("excluded_censored_event").all()
    assert per.iloc[0].fp == 0
    metrics = aggregate_metrics(per).iloc[0]
    assert np.isnan(metrics.recall) and np.isnan(metrics.precision)


def test_full_calendar_end_is_used_when_final_observation_is_missing():
    per, events, _ = evaluate_series(*example(events=("2024-09",), missing=("2024-12",)))
    assert events.iloc[0].eligibility == "full_window"
    assert per.iloc[0].fn == 1 and per.iloc[0].n_monitoring_months == 7


def test_before_monitoring_and_outside_observation_events_are_explicit():
    per, events, alarms = evaluate_series(*example(alarms=("2024-05",),
                                                 events=("2024-03", "2023-12", "2025-01")))
    assert events.set_index("event_start_period").eligibility.to_dict() == {
        "2023-12": "outside_observation", "2024-03": "before_monitoring", "2025-01": "outside_observation"}
    assert per.iloc[0].n_events_excluded == 3 and per.iloc[0].n_events == 0
    assert alarms.iloc[0].status == "excluded_censored_event"


def test_no_monitoring_exposure_keeps_far_undefined():
    per, events, alarms = evaluate_series(*example(events=("2024-02",), end="2024-04"))
    assert events.iloc[0].eligibility == "before_monitoring"
    assert per.iloc[0].n_monitoring_months == 0 and alarms.empty
    assert np.isnan(aggregate_metrics(per).iloc[0].false_alarms_per_12_months)


def test_no_change_control_has_undefined_recall_but_real_false_alarm_rate():
    per, events, alarms = evaluate_series(*example(alarms=("2024-06",), events=()))
    assert events.empty and len(alarms) == 1
    metrics = aggregate_metrics(per).iloc[0]
    assert metrics.precision == 0 and np.isnan(metrics.recall) and np.isnan(metrics.missed_fraction)
    assert metrics.false_alarms_per_12_months == 1.5


def test_two_series_do_not_share_matching_and_group_metrics():
    one = example(alarms=("2024-06",))
    two = tuple(frame.assign(series_id="b") for frame in example(alarms=("2024-05",)))
    per, _, _ = evaluate_series(*(pd.concat([a, b], ignore_index=True) for a, b in zip(one, two)))
    metrics = aggregate_metrics(per, ["split"]).iloc[0]
    assert metrics.n_series == 2 and metrics.tp == 1 and metrics.fp == 1 and metrics.fn == 1
    assert metrics.precision == metrics.recall == metrics.f1 == 0.5


def test_bootstrap_resamples_whole_series_reproducibly_and_keeps_delay_clusters():
    a, _, _ = evaluate_series(*example(alarms=("2024-06",), events=("2024-06",)))
    b, _, _ = evaluate_series(*example(alarms=("2024-09",), events=("2024-06",)))
    c, _, _ = evaluate_series(*example(alarms=(), events=("2024-06",)))
    per = pd.concat([a, b.assign(series_id="b"), c.assign(series_id="c")], ignore_index=True)
    result = bootstrap_metrics(per, ["split"], n_bootstrap=100, seed=12)
    repeated = bootstrap_metrics(per, ["split"], n_bootstrap=100, seed=12)
    pd.testing.assert_frame_equal(result, repeated)
    recall = result.set_index("metric").loc["recall"]
    assert recall.estimate == pytest.approx(2 / 3)
    assert recall.ci_low <= recall.estimate <= recall.ci_high
    assert result.bootstrap_unit.eq("series").all() and result.n_series.eq(3).all()
    delay = result.set_index("metric").loc["median_delay"]
    assert delay.estimate == 1.5 and delay.ci_low == 0 and delay.ci_high == 3


def test_bootstrap_undefined_metrics_are_not_filled_with_zero():
    per, _, _ = evaluate_series(*example(events=()))
    result = bootstrap_metrics(per, n_bootstrap=10, seed=1).set_index("metric")
    assert np.isnan(result.loc["recall", "estimate"])
    assert result.loc["recall", "n_finite_bootstrap"] == 0
    assert result.loc["false_alarms_per_12_months", "ci_low"] == 0


@pytest.mark.parametrize("invalid", ["duplicate_month", "duplicate_series", "duplicate_event", "warmup_alarm", "early_signal"])
def test_invalid_inputs_are_rejected(invalid):
    signals, events, info = example(alarms=("2024-06",))
    if invalid == "duplicate_month":
        signals = pd.concat([signals, signals.iloc[[0]]])
    elif invalid == "duplicate_series":
        info = pd.concat([info, info])
    elif invalid == "duplicate_event":
        events = pd.concat([events, events])
    elif invalid == "warmup_alarm":
        signals.loc[0, "is_alarm"] = True
    elif invalid == "early_signal":
        signals.loc[signals.is_alarm, "signal_date"] = pd.Timestamp("2024-05-31")
    with pytest.raises(ValueError):
        evaluate_series(signals, events, info)
