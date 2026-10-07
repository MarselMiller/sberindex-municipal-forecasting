"""Calendar localization, fixed validation selection and whole-series CIs."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from sberforecast.offline_evaluation import (
    METRICS, aggregate_metrics, bootstrap_metrics, evaluate_series, select_candidate,
)
from sberforecast.online_synthetic import generate_benchmark


def example(points=(), events=("2024-06",), missing=(), lag=0, end="2024-12"):
    periods = pd.period_range("2024-01", end, freq="M")
    rows, n_observed = [], 0
    for period in periods:
        phase = "missing" if str(period) in missing else "warmup" if n_observed < 4 else "monitoring"
        n_observed += int(phase != "missing")
        rows.append(dict(series_id="a", observation_period=str(period),
                         availability_date=(period + lag).to_timestamp(how="end").normalize(),
                         phase=phase))
    stream = pd.DataFrame(rows)
    breakpoint_rows = [dict(series_id="a", method="PELT", breakpoint_id=f"a:{index}",
                            breakpoint_month=month, analyzed_end_period=end,
                            analyzed_available_date=(periods[-1] + lag).to_timestamp(how="end").normalize())
                       for index, month in enumerate(points)]
    breakpoints = pd.DataFrame(breakpoint_rows, columns=[
        "series_id", "method", "breakpoint_id", "breakpoint_month", "analyzed_end_period", "analyzed_available_date",
    ])
    truths = pd.DataFrame([dict(series_id="a", event_id=str(index), event_type="level",
                                event_start_period=month, strength=4.0, duration_months=7)
                          for index, month in enumerate(events)], columns=[
        "series_id", "event_id", "event_type", "event_start_period", "strength", "duration_months",
    ])
    info = pd.DataFrame([dict(series_id="a", scenario="level_up" if events else "no_change",
                              strength=4.0 if events else 0.0, noise_fraction=0.01, split="test",
                              observation_start_period="2024-01", observation_end_period=end,
                              release_lag_months=lag)])
    return stream, breakpoints, truths, info


def candidate(candidate_id="a", **updates):
    row = dict(split="validation", method="PELT", candidate_id=candidate_id,
               primary_f1=0.6, primary_recall=0.5,
               primary_median_absolute_localisation_error=1.0,
               no_change_far=0.1, outlier_far=0.2)
    return dict(row, **updates)


def test_inclusive_window_excess_point_false_and_observed_exposure():
    per, matches, points = evaluate_series(*example(points=("2024-09", "2024-10")))
    assert (per.iloc[0].tp, per.iloc[0].fp, per.iloc[0].fn) == (1, 1, 0)
    assert matches.iloc[0].breakpoint_offset_months == 3
    assert matches.iloc[0].absolute_localisation_error_months == 3
    assert matches.iloc[0].duration_months == 7
    assert points.status.tolist() == ["true_positive", "false_positive"]
    metrics = aggregate_metrics(per).iloc[0]
    assert metrics.precision == 0.5 and metrics.recall == 1
    assert metrics.f1 == pytest.approx(2 / 3) and metrics.miss_rate == 0
    assert metrics.n_monitoring_months == 8 and metrics.false_positives_per_12_months == 1.5
    assert metrics.median_absolute_localisation_error == metrics.median_breakpoint_offset == 3


def test_before_true_event_is_false_positive_never_a_warning():
    per, matches, points = evaluate_series(*example(points=("2024-05",)))
    assert (per.iloc[0].tp, per.iloc[0].fp, per.iloc[0].fn) == (0, 1, 1)
    assert matches.iloc[0].status == "missed" and points.iloc[0].status == "false_positive"
    metrics = aggregate_metrics(per).iloc[0]
    assert metrics.recall == 0 and metrics.miss_rate == 1
    assert np.isnan(metrics.median_absolute_localisation_error)


def test_chronological_one_to_one_earliest_closing_window():
    inputs = example(points=("2024-08", "2024-07"), events=("2024-07", "2024-06"))
    per, matches, points = evaluate_series(*inputs)
    assert per.iloc[0].tp == 2
    assert points.breakpoint_month.tolist() == ["2024-07", "2024-08"]
    assert points.matched_event_id.tolist() == ["1", "0"]
    assert matches.matched_breakpoint_id.nunique() == 2
    assert points.breakpoint_offset_months.tolist() == [1, 1]


def test_one_event_gets_only_first_point_even_inside_its_window():
    per, matches, points = evaluate_series(*example(points=("2024-06", "2024-07", "2024-08")))
    assert per.iloc[0].tp == 1 and per.iloc[0].fp == 2
    assert points.status.tolist() == ["true_positive", "false_positive", "false_positive"]
    assert matches.iloc[0].breakpoint_month == "2024-06"


def test_missing_months_preserve_localization_window_and_exposure():
    per, matches, _ = evaluate_series(*example(points=("2024-09",), missing=("2024-07", "2024-08")))
    assert per.iloc[0].n_monitoring_months == 6
    assert matches.iloc[0].window_end_period == "2024-09"
    assert matches.iloc[0].breakpoint_offset_months == 3
    per, matches, points = evaluate_series(*example(points=("2024-10",), missing=("2024-07", "2024-08")))
    assert per.iloc[0].tp == 0 and per.iloc[0].fp == 1 and matches.iloc[0].status == "missed"


def test_warmup_points_preserved_outside_evaluation_not_false_positives():
    per, matches, points = evaluate_series(*example(points=("2024-02", "2024-06")))
    assert points.status.tolist() == ["outside_evaluation_warmup", "true_positive"]
    assert per.iloc[0].n_breakpoints == 2 and per.iloc[0].n_breakpoints_excluded == 1
    assert per.iloc[0].tp == 1 and per.iloc[0].fp == 0


def test_missing_point_explicitly_excluded_without_compressing_calendar():
    per, _, points = evaluate_series(*example(points=("2024-07",), missing=("2024-07",)))
    assert points.iloc[0].status == "outside_evaluation_missing"
    assert per.iloc[0].n_breakpoints_excluded == 1 and per.iloc[0].fp == 0


def test_analysis_availability_and_publication_lag_do_not_change_localization():
    per, matches, points = evaluate_series(*example(points=("2024-09",), lag=2))
    assert per.iloc[0].tp == 1 and matches.iloc[0].breakpoint_offset_months == 3
    assert per.iloc[0].availability_end_period == "2025-02"
    assert per.iloc[0].localization_end_period == "2024-12"
    assert points.iloc[0].analyzed_available_date == pd.Timestamp("2025-02-28")
    source = list(example(points=("2024-09",), lag=2))
    source[1]["analyzed_available_date"] = pd.Timestamp("2027-12-31")
    repeated, repeated_matches, _ = evaluate_series(*source)
    pd.testing.assert_frame_equal(per, repeated)
    pd.testing.assert_frame_equal(matches, repeated_matches)


def test_publication_lag_does_not_extend_incomplete_observation_window():
    per, matches, points = evaluate_series(*example(points=("2024-11", "2024-12"), events=("2024-11",), lag=2))
    assert matches.iloc[0].eligibility == "partial_window" and matches.iloc[0].status == "excluded"
    assert points.status.eq("excluded_censored_event").all()
    assert per.iloc[0].n_events == 0 and per.iloc[0].n_events_excluded == 1
    assert per.iloc[0].n_breakpoints_excluded == 2 and per.iloc[0].fp == 0


def test_censored_warmup_and_end_windows_keep_points_explicit():
    per, matches, points = evaluate_series(*example(points=("2024-05", "2024-11", "2024-12"),
                                                  events=("2024-03", "2024-11")))
    assert matches.eligibility.tolist() == ["before_monitoring", "partial_window"]
    assert points.status.eq("excluded_censored_event").all()
    assert per.iloc[0].n_events_total == 2 and per.iloc[0].n_events == 0
    assert per.iloc[0].n_breakpoints_excluded == 3 and per.iloc[0].fp == 0
    metrics = aggregate_metrics(per).iloc[0]
    assert np.isnan(metrics.recall) and np.isnan(metrics.precision)


def test_full_calendar_end_used_even_if_last_month_missing():
    per, matches, _ = evaluate_series(*example(events=("2024-09",), missing=("2024-12",)))
    assert matches.iloc[0].eligibility == "full_window"
    assert per.iloc[0].fn == 1 and per.iloc[0].n_monitoring_months == 7


def test_zero_breakpoints_means_missed_event_not_success():
    per, matches, points = evaluate_series(*example())
    assert points.empty and per.iloc[0].tp == per.iloc[0].fp == 0 and per.iloc[0].fn == 1
    assert matches.iloc[0].status == "missed"
    metrics = aggregate_metrics(per).iloc[0]
    assert np.isnan(metrics.precision) and metrics.recall == metrics.f1 == 0 and metrics.miss_rate == 1


def test_no_change_no_events_undefined_recall_observed_false_positive_rate():
    per, matches, points = evaluate_series(*example(points=("2024-06",), events=()))
    assert matches.empty and len(points) == 1
    metrics = aggregate_metrics(per).iloc[0]
    assert metrics.precision == 0 and np.isnan(metrics.recall) and np.isnan(metrics.miss_rate)
    assert metrics.false_positives_per_12_months == 1.5


def test_no_monitoring_exposure_far_undefined_and_boundary_event_excluded():
    per, matches, points = evaluate_series(*example(events=("2024-02",), end="2024-04"))
    assert matches.iloc[0].eligibility == "before_monitoring" and points.empty
    assert per.iloc[0].n_monitoring_months == 0
    assert np.isnan(aggregate_metrics(per).iloc[0].false_positives_per_12_months)


def test_series_pools_are_independent_and_grouped_metrics_are_counts():
    one = example(points=("2024-06",))
    two = tuple(frame.assign(series_id="b") for frame in example(points=("2024-05",)))
    per, _, _ = evaluate_series(*(pd.concat([a, b], ignore_index=True) for a, b in zip(one, two)))
    metrics = aggregate_metrics(per, ["split"]).iloc[0]
    assert metrics.n_series == 2 and metrics.tp == metrics.fp == metrics.fn == 1
    assert metrics.precision == metrics.recall == metrics.f1 == 0.5


def test_bootstrap_reproducible_whole_series_clustered_offsets():
    a, _, _ = evaluate_series(*example(points=("2024-06",)))
    b, _, _ = evaluate_series(*example(points=("2024-09",)))
    c, _, _ = evaluate_series(*example())
    per = pd.concat([a, b.assign(series_id="b"), c.assign(series_id="c")], ignore_index=True)
    result = bootstrap_metrics(per, ["split"], n_bootstrap=100, seed=12)
    repeated = bootstrap_metrics(per, ["split"], n_bootstrap=100, seed=12)
    pd.testing.assert_frame_equal(result, repeated)
    recall = result.set_index("metric").loc["recall"]
    assert recall.estimate == pytest.approx(2 / 3) and recall.ci_low <= recall.estimate <= recall.ci_high
    assert result.bootstrap_unit.eq("series").all() and result.n_series.eq(3).all()
    assert result.conditional_on_selected_parameters.all()
    offset = result.set_index("metric").loc["median_breakpoint_offset"]
    assert offset.estimate == 1.5 and offset.ci_low == 0 and offset.ci_high == 3
    assert result.set_index("metric").loc["median_absolute_localisation_error", "estimate"] == 1.5


def test_bootstrap_preserves_stratum_mix_and_observed_exposure_as_series_units():
    a, _, _ = evaluate_series(*example(points=("2024-05",), events=(), missing=("2024-11", "2024-12")))
    b, _, _ = evaluate_series(*example(points=(), events=()))
    # One series in each noise stratum: every replicate retains both complete
    # series and their respective six/eight exposure months, not eight each.
    per = pd.concat([a, b.assign(series_id="b", noise_fraction=0.03)], ignore_index=True)
    result = bootstrap_metrics(per, n_bootstrap=25).set_index("metric")
    expected_far = 12 / 14
    assert result.loc["false_positives_per_12_months", "estimate"] == pytest.approx(expected_far)
    assert result.loc["false_positives_per_12_months", "ci_low"] == pytest.approx(expected_far)
    assert result.loc["false_positives_per_12_months", "ci_high"] == pytest.approx(expected_far)


def test_bootstrap_undefined_metrics_retained_and_duplicate_series_rejected():
    per, _, _ = evaluate_series(*example(events=()))
    result = bootstrap_metrics(per, n_bootstrap=10, seed=1).set_index("metric")
    assert np.isnan(result.loc["recall", "estimate"]) and result.loc["recall", "n_finite_bootstrap"] == 0
    assert result.loc["false_positives_per_12_months", "ci_low"] == 0
    with pytest.raises(ValueError, match="independent series"):
        bootstrap_metrics(pd.concat([per, per]), n_bootstrap=10)


def test_candidates_are_validation_only_and_each_control_meets_budget():
    rows = pd.DataFrame([candidate("high", primary_f1=0.9, outlier_far=1.1),
                         candidate("low", primary_f1=0.4, no_change_far=1.0, outlier_far=1.0)])
    assert select_candidate(rows, 1).candidate_id == "low"
    assert select_candidate(rows.iloc[:1], 1) is None
    with pytest.raises(ValueError, match="only on validation"):
        select_candidate(rows.assign(split="test"), 1)
    with pytest.raises(ValueError, match="only on validation"):
        select_candidate(pd.concat([rows, rows.iloc[[0]].assign(split="test", candidate_id="test")]), 1)


@pytest.mark.parametrize("a,b,expected", [
    ({"primary_f1": 0.7}, {}, "a"),
    ({"primary_recall": 0.6}, {}, "a"),
    ({"outlier_far": 0.1}, {}, "a"),
    ({"primary_median_absolute_localisation_error": 0.0}, {}, "a"),
    ({}, {}, "a"),
    ({"primary_median_absolute_localisation_error": np.nan}, {}, "b"),
])
def test_prefixed_selection_order_is_stable_and_independent_of_input_order(a, b, expected):
    rows = pd.DataFrame([candidate("b", **b), candidate("a", **a)])
    assert select_candidate(rows, 1).candidate_id == expected
    assert select_candidate(rows.iloc[::-1], 1).candidate_id == expected


def test_undefined_or_unbounded_control_cannot_be_selected():
    rows = pd.DataFrame([candidate("a", no_change_far=np.nan), candidate("b", outlier_far=np.inf)])
    assert select_candidate(rows, 1) is None
    with pytest.raises(ValueError, match="disagrees"):
        select_candidate(pd.DataFrame([candidate(max_control_far=0.0)]), 1)
    assert select_candidate(pd.DataFrame([candidate(max_control_far=0.2)]), 1).candidate_id == "a"


@pytest.mark.parametrize("invalid", ["duplicate_month", "duplicate_series", "duplicate_event", "duplicate_point",
                                          "point_outside", "point_absent", "wrong_phase", "wrong_eligibility", "mixed_methods",
                                          "unknown_series", "negative_lag", "fractional_lag", "negative_availability", "invalid_phase"])
def test_invalid_evaluation_inputs_rejected(invalid):
    stream, points, events, info = example(points=("2024-06",))
    if invalid == "duplicate_month":
        stream = pd.concat([stream, stream.iloc[[0]]])
    elif invalid == "duplicate_series":
        info = pd.concat([info, info])
    elif invalid == "duplicate_event":
        events = pd.concat([events, events])
    elif invalid == "duplicate_point":
        points = pd.concat([points, points.assign(breakpoint_id="other")])
    elif invalid == "point_outside":
        points["breakpoint_month"] = "2025-01"
    elif invalid == "point_absent":
        stream = stream.loc[stream.observation_period.ne("2024-06")]
    elif invalid == "wrong_phase":
        points["phase"] = "warmup"
    elif invalid == "wrong_eligibility":
        points["eligible_evaluation"] = False
    elif invalid == "mixed_methods":
        points = pd.concat([points, points.assign(method="BinSeg", breakpoint_id="other", breakpoint_month="2024-07")])
    elif invalid == "unknown_series":
        points["series_id"] = "unknown"
    elif invalid == "negative_lag":
        info["release_lag_months"] = -1
    elif invalid == "fractional_lag":
        info["release_lag_months"] = 1.5
    elif invalid == "negative_availability":
        stream.loc[0, "availability_date"] = pd.Timestamp("2023-12-31")
    elif invalid == "invalid_phase":
        stream.loc[0, "phase"] = "other"
    with pytest.raises(ValueError):
        evaluate_series(stream, points, events, info)


@pytest.mark.parametrize("window", [-1, 1.5, True])
def test_invalid_window_rejected(window):
    with pytest.raises(ValueError):
        evaluate_series(*example(), window_months=window)


@pytest.mark.parametrize("replicates", [0, -1, 1.5, True])
def test_invalid_bootstrap_count_rejected(replicates):
    per, _, _ = evaluate_series(*example())
    with pytest.raises(ValueError):
        bootstrap_metrics(per, n_bootstrap=replicates)


def test_no_metric_field_confuses_retrospective_offsets_with_issue_times():
    per, matches, points = evaluate_series(*example(points=("2024-06",)))
    assert "delay" not in " ".join([*METRICS, *per.columns, *matches.columns, *points.columns])
    assert json.loads(per.iloc[0].breakpoint_offsets_months) == [0]


def test_existing_generator_seeds_ids_and_protocol_are_disjoint_repeatable():
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "configs/online_detection.yaml").read_text(encoding="utf-8"))
    generator = dict(cfg["generator"], replicates_per_cell=1)
    val = generate_benchmark(generator, "validation", cfg["forecast"], 0)
    test = generate_benchmark(generator, "test", cfg["forecast"], 0)
    repeated = generate_benchmark(generator, "validation", cfg["forecast"], 0)
    assert len(val[2]) == len(test[2]) == 42
    assert not set(val[2].seed) & set(test[2].seed)
    assert not set(val[2].series_id) & set(test[2].series_id)
    for first, second in zip(val, repeated):
        pd.testing.assert_frame_equal(first, second)
    primary_ids = set(val[2].loc[val[2].scenario.isin(["level_up", "level_down"]), "series_id"])
    assert val[1].loc[val[1].series_id.isin(primary_ids), "duration_months"].ge(3).all()
