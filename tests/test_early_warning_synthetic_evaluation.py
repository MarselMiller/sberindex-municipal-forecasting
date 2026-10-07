"""Synthetic, integer-indexed E07c evaluation and temporal-boundary regressions."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from sberforecast.early_warning_synthetic_evaluation import (
    EVENT_METRICS, EXAMPLE_RULE, TRUTH, evaluate_predictions, select_threshold,
)


def _config():
    return dict(generator=dict(months=24), labels=dict(min_history=12, horizons=[1, 3]),
                threshold=dict(objective="validation_row_f1", tie_break="higher_threshold",
                               false_alerts_per_12_control_months_max=1.0, fallback=.5),
                evaluation=dict(calibration_bins=10, bootstrap_resamples=80,
                                bootstrap_seed=420401, ci_level=.95,
                                scenarios=["base_all", "weak_precursor", "strong_precursor", "high_noise", "controls_false_precursor"],
                                examples=dict(model="S3", primary_k=3)))


def _metadata(cohort="test"):
    return pd.DataFrame([
        dict(series_id="A", cohort=cohort, has_event=True, is_anticipated=True,
             false_precursor=False, precursor_class="strong", noise_class="low"),
        dict(series_id="B", cohort=cohort, has_event=True, is_anticipated=False,
             false_precursor=False, precursor_class="none", noise_class="high"),
        dict(series_id="C", cohort=cohort, has_event=False, is_anticipated=False,
             false_precursor=True, precursor_class="weak", noise_class="high"),
        dict(series_id="D", cohort=cohort, has_event=False, is_anticipated=False,
             false_precursor=False, precursor_class="none", noise_class="low"),
    ])


def _events(cohort="test"):
    return pd.DataFrame([dict(series_id="A", cohort=cohort, event_id="A_event", onset_index=16),
                         dict(series_id="B", cohort=cohort, event_id="B_event", onset_index=18)])


def _predictions(rows=None, *, k=3, model="S3", threshold=.5, cohort="test"):
    if rows is None:
        rows = [("A", 11, .1), ("A", 13, .8), ("A", 14, .9), ("A", 15, .85),
                ("B", 11, .1), ("B", 15, .1), ("B", 17, .1),
                ("C", 11, .7), ("C", 12, .2), ("C", 13, .2), ("D", 11, .2)]
    onsets = {"A": 16, "B": 18}
    return pd.DataFrame([dict(series_id=series, cohort=cohort, month_index=origin,
                              k=k, model=model, probability=probability,
                              label=int(origin < onsets.get(series, 100) <= origin + k),
                              threshold=threshold, alert=probability >= threshold,
                              fully_known=True, at_risk=True, eligible=True)
                         for series, origin, probability in rows])


def _run(predictions=None, events=None, metadata=None, config=None):
    return evaluate_predictions(_predictions() if predictions is None else predictions,
                                _events() if events is None else events,
                                _metadata() if metadata is None else metadata,
                                _config() if config is None else config)


def test_event_warning_repeats_are_audited_but_not_extra_precision_successes():
    result = _run()
    row = result["metrics_event"].iloc[0]
    assert row.monitored_cases == 11
    assert row.monitored_series_years == pytest.approx(11 / 12)
    assert row.eligible_events == 2 and row.warned_events == 1
    assert row.event_recall == .5 and row.miss_rate == .5
    assert row.alert_precision == .5
    assert row.raw_alert_count == 4 and row.deduplicated_alert_count == 2
    assert row.repeated_alert_count == 2 and row.false_alert_count == 1
    assert row.false_alerts_per_12_monitored_months == pytest.approx(12 / 11)
    assert row.mean_lead_time_months == row.median_lead_time_months == 3
    match = result["event_matches"].set_index("event_id").loc["A_event"]
    assert match.earliest_alert_index == 13 and match.warning_count_before_event == 3
    assert match.successful_warning_count == 1
    assert result["alerts"].status.tolist() == ["false_alert_no_eligible_future_event",
                                                "successful_pre_onset_warning",
                                                "repeated_pre_onset_warning",
                                                "repeated_pre_onset_warning"]


def test_at_and_after_onset_are_never_warning_successes_and_not_monitored_when_active():
    prediction = _predictions([("A", 15, .1), ("A", 16, .99), ("A", 17, .99)])
    prediction.loc[prediction.month_index.ge(16), "at_risk"] = False
    prediction.loc[prediction.month_index.ge(16), "active_regime"] = True
    result = _run(prediction)
    assert result["metrics_event"].iloc[0].warned_events == 0
    assert result["metrics_event"].iloc[0].monitored_cases == 1
    assert result["alerts"].empty
    assert not result["event_matches"].iloc[0].warning_observed
    incorrect = prediction.copy()
    incorrect["at_risk"] = True
    with pytest.raises(ValueError, match="Already active"):
        _run(incorrect)


def test_events_without_a_preonset_monitored_origin_are_not_fabricated_as_misses():
    result = _run(_predictions([("A", 11, .99), ("C", 11, .1)]))
    assert result["metrics_event"].iloc[0].eligible_events == 0
    assert result["event_matches"].empty
    assert np.isnan(result["metrics_event"].iloc[0].event_recall)
    assert result["alerts"].iloc[0].false_alert


def test_one_alert_warns_at_most_one_event_in_an_overlapping_future_window():
    events = _events()
    events = pd.concat([events, pd.DataFrame([dict(series_id="A", cohort="test", event_id="A_second", onset_index=17)])], ignore_index=True)
    result = _run(_predictions([("A", 14, .9)]), events)
    row = result["metrics_event"].iloc[0]
    assert row.eligible_events == 2 and row.warned_events == 1 and row.event_recall == .5
    assert result["alerts"].iloc[0].event_id == "A_event"
    assert result["event_matches"].successful_warning_count.sum() == 1


def test_censored_ineligible_and_active_rows_do_not_become_negative_monitored_cases():
    prediction = _predictions([("A", 13, .8), ("C", 10, .99), ("C", 21, .99), ("B", 19, .99)])
    prediction.loc[prediction.month_index.eq(21), ["fully_known", "label"]] = [False, np.nan]
    prediction.loc[prediction.month_index.eq(10), "eligible"] = False
    prediction.loc[prediction.month_index.eq(19), ["at_risk", "label"]] = [False, np.nan]
    result = _run(prediction)
    assert result["metrics_row"].iloc[0].cases == 1
    assert result["metrics_row"].iloc[0].positives == 1
    assert result["metrics_event"].iloc[0].monitored_cases == 1
    incorrect = prediction.copy()
    incorrect.loc[incorrect.month_index.eq(21), "label"] = 0
    with pytest.raises(ValueError, match="Unknown/censored"):
        _run(incorrect)


def test_zero_based_min_history_admits_origin_eleven_not_twelve():
    result = _run(_predictions([("C", 10, .8), ("C", 11, .8)], k=1))
    assert result["metrics_row"].iloc[0].cases == 1
    assert result["alerts"].month_index.tolist() == [11]


@pytest.mark.parametrize("column,value,message", [
    ("month_index", 11.5, "integer"), ("k", 2, "horizons"),
    ("probability", np.inf, "Probabilities"), ("label", 0, "Label disagrees"),
    ("alert", False, "Stored alert"), ("threshold", -1, "Threshold"),
])
def test_invalid_integer_keys_labels_and_probabilities_are_rejected(column, value, message):
    prediction = _predictions([("A", 13, .8)])
    prediction[column] = value
    with pytest.raises(ValueError, match=message):
        _run(prediction)


def test_future_window_must_be_observed_even_for_a_negative_case():
    with pytest.raises(ValueError, match="complete O\\+k"):
        _run(_predictions([("C", 21, .1)]))


def test_duplicate_monthly_keys_and_inconsistent_event_registry_are_rejected():
    prediction = _predictions([("A", 13, .8)])
    with pytest.raises(ValueError, match="Duplicate keyed"):
        _run(pd.concat([prediction, prediction], ignore_index=True))
    metadata = _metadata()
    metadata.loc[metadata.series_id.eq("A"), "has_event"] = False
    with pytest.raises(ValueError, match="has_event"):
        _run(metadata=metadata)


def test_tied_average_precision_and_selected_threshold_confusion_reuse_fixed_ten_bins():
    result = _run(_predictions([("A", 13, .7), ("C", 11, .7)], threshold=.8))
    row = result["metrics_row"].iloc[0]
    assert row.pr_auc == .5 and row.roc_auc == .5
    assert row.brier_score == pytest.approx(.29)
    assert row.threshold == .8 and row.f1 == 0 and row.precision == 0 and row.recall == 0
    assert row.true_positives == 0 and row.false_negatives == 1
    assert "ties" in row.pr_auc_definition and row.truth_definition == TRUTH
    bins = result["calibration"]
    assert len(bins) == 10 and bins["count"].sum() == 2
    assert bins.loc[bins["count"].eq(2), "positive_fraction"].item() == .5


def _validation_budget_fixture():
    rows = [("C", origin, .8 if origin == 11 else .1) for origin in range(11, 23)]
    rows += [("A", 15, .6), ("A", 14, .4)]
    return _predictions(rows, cohort="validation", k=1), _events("validation"), _metadata("validation")


def test_validation_threshold_maximizes_f1_under_monthly_control_budget_with_higher_tie():
    predictions, events, metadata = _validation_budget_fixture()
    selected = select_threshold(predictions, events, metadata, _config())
    assert selected["threshold"] == .6  # 0.5 gives the same alerts, higher tie wins.
    assert selected["validation_f1"] == pytest.approx(2 / 3)
    assert selected["control_monitored_months"] == 12 and selected["control_monitored_years"] == 1
    assert selected["false_control_alerts"] == 1
    assert selected["false_alerts_per_12_control_months"] == 1
    assert selected["constraint_met"] and not selected["fallback_used"]
    assert selected["candidates_evaluated"] == 6


def test_control_budget_counts_monthly_alerts_not_series_or_deduplicated_event_alerts():
    predictions, events, metadata = _validation_budget_fixture()
    predictions.loc[predictions.series_id.eq("C") & predictions.month_index.eq(12), "probability"] = .8
    selected = select_threshold(predictions, events, metadata, _config())
    assert selected["threshold"] == 1.0
    assert selected["false_control_alerts"] == 0 and selected["validation_f1"] == 0


def test_threshold_fallback_records_unmet_budget_and_no_control_exposure():
    predictions, events, metadata = _validation_budget_fixture()
    predictions.loc[predictions.series_id.eq("C"), "probability"] = 1.0
    selected = select_threshold(predictions, events, metadata, _config())
    assert selected["threshold"] == .5 and selected["fallback_used"] and not selected["constraint_met"]
    assert selected["fallback_reason"] == "no_candidate_meets_control_alert_budget"
    predictions = predictions.loc[predictions.series_id.eq("A")]
    selected = select_threshold(predictions, events, metadata, _config())
    assert selected["fallback_used"] and selected["fallback_reason"] == "no_control_exposure"
    assert np.isnan(selected["false_alerts_per_12_control_months"])


@pytest.mark.parametrize("frame_number", [0, 1, 2])
def test_threshold_selector_rejects_test_or_train_in_any_input(frame_number):
    frames = list(_validation_budget_fixture())
    for forbidden in ("test", "train"):
        altered = [frame.copy() for frame in frames]
        altered[frame_number].loc[altered[frame_number].index[0], "cohort"] = forbidden
        with pytest.raises(ValueError, match="validation only"):
            select_threshold(*altered, _config())


def test_threshold_selection_does_not_read_or_modify_test_predictions():
    predictions, events, metadata = _validation_budget_fixture()
    before = [frame.copy(deep=True) for frame in (predictions, events, metadata)]
    first = select_threshold(predictions, events, metadata, _config())
    unrelated_test = _predictions()
    unrelated_test["label"] = 1 - unrelated_test.label
    unrelated_test["probability"] = .999
    second = select_threshold(predictions, events, metadata, _config())
    assert first == second
    for actual, expected in zip((predictions, events, metadata), before):
        assert_frame_equal(actual, expected)


def test_threshold_requires_single_model_horizon_and_uses_defensive_case_admission():
    predictions, events, metadata = _validation_budget_fixture()
    censored = _predictions([("C", 23, 1)], cohort="validation", k=1)
    censored["fully_known"], censored["label"] = False, np.nan
    combined = pd.concat([predictions, censored], ignore_index=True)
    selected = select_threshold(combined, events, metadata, _config())
    assert selected["control_monitored_months"] == 12 and selected["threshold"] == .6
    other = predictions.copy()
    other["model"] = "S2"
    with pytest.raises(ValueError, match="one validation model/k"):
        select_threshold(pd.concat([predictions, other]), events, metadata, _config())


def test_bootstrap_samples_whole_series_with_multiplicity_and_reproduces_manual_intervals():
    predictions = _predictions([("A", 15, .9), ("B", 15, .9),
                                ("C", 11, .9), ("C", 12, .9), ("C", 13, .9), ("D", 11, .1)])
    config = _config()
    result = _run(predictions, config=config)
    intervals = result["bootstrap_intervals"].set_index("metric")
    assert set(intervals.index) == set(EVENT_METRICS)
    assert intervals.series_count.eq(4).all()
    assert intervals.bootstrap_unit.eq("whole_series_id_with_multiplicity").all()
    # A: one event, lead 1; B: one event, lead 3; C: three false monthly
    # alerts and three months; D: one negative control month. Repeated sampled
    # controls retain all THREE C alerts, which row bootstrap would not do.
    weights = np.random.default_rng(420401).multinomial(4, np.full(4, .25), size=80)
    manual = {metric: [] for metric in EVENT_METRICS}
    for a, b, c, d in weights:
        warned, false, monitored = a + b, 3 * c, a + b + 3 * c + d
        leads = [1] * a + [3] * b
        manual["event_recall"].append(1.0 if warned else np.nan)
        manual["miss_rate"].append(0.0 if warned else np.nan)
        manual["alert_precision"].append(warned / (warned + false) if warned + false else np.nan)
        manual["false_alerts_per_12_monitored_months"].append(false * 12 / monitored)
        manual["median_lead_time_months"].append(float(np.median(leads)) if leads else np.nan)
        manual["mean_lead_time_months"].append(float(np.mean(leads)) if leads else np.nan)
    for metric, values in manual.items():
        finite = np.asarray(values)[np.isfinite(values)]
        low, high = np.quantile(finite, [.025, .975])
        assert intervals.loc[metric, "lower"] == pytest.approx(low)
        assert intervals.loc[metric, "upper"] == pytest.approx(high)
        assert intervals.loc[metric, "defined_resamples"] == len(finite)
    assert intervals.loc["event_recall", "defined_resamples"] < 80


def test_validation_has_metrics_but_no_test_bootstrap_or_examples():
    result = _run(_predictions(cohort="validation"), _events("validation"), _metadata("validation"))
    assert not result["metrics_event"].empty
    assert result["bootstrap_intervals"].empty and result["examples"].empty


def test_complete_registry_train_metadata_are_ignored_by_evaluation():
    expected = _run()
    train_events, train_metadata = _events("train"), _metadata("train")
    for frame in (train_events, train_metadata):
        frame["series_id"] = "train_" + frame.series_id
    train_events["event_id"] = "train_" + train_events.event_id
    actual = _run(events=pd.concat([_events(), train_events], ignore_index=True),
                  metadata=pd.concat([_metadata(), train_metadata], ignore_index=True))
    for key in expected:
        assert_frame_equal(actual[key], expected[key])


def test_scenarios_preserve_common_controls_and_separate_event_subsets():
    scenarios = _run()["scenario_metrics"].set_index("scenario")
    assert len(scenarios) == 7
    assert scenarios.loc["anticipated_events", "eligible_events"] == 1
    assert scenarios.loc["anticipated_events", "event_recall"] == 1
    assert scenarios.loc["unanticipated_events", "eligible_events"] == 1
    assert scenarios.loc["unanticipated_events", "event_recall"] == 0
    assert scenarios.loc["anticipated_events", "control_monitored_months"] == 4
    assert scenarios.loc["unanticipated_events", "control_monitored_months"] == 4
    assert scenarios.loc["weak_precursor", "monitored_cases"] == 3
    assert scenarios.loc["controls_false_precursor", "monitored_cases"] == 3
    assert scenarios.loc["controls_false_precursor", "false_alert_count"] == 1
    assert np.isnan(scenarios.loc["controls_false_precursor", "event_recall"])
    assert scenarios.loc["high_noise", "monitored_cases"] == 6


def test_ablation_records_raw_fixed_comparisons_without_model_selection():
    prediction = _predictions()
    earlier = prediction.copy()
    earlier["model"] = "S1"
    earlier["probability"] = .1
    earlier["alert"] = False
    middle = earlier.copy()
    middle["model"] = "S2"
    result = _run(pd.concat([earlier, middle, prediction], ignore_index=True))
    ablation = result["ablation"]
    assert set(ablation.comparison) == {"S2_minus_S1", "S3_minus_S2"}
    recall = ablation.loc[ablation.comparison.eq("S3_minus_S2") & ablation.metric.eq("event_recall")].iloc[0]
    assert recall.earlier_value == 0 and recall.later_value == .5 and recall.difference == .5
    assert ablation.interpretation.eq("descriptive_fixed_ablation_no_model_selection").all()


def test_examples_are_lexicographic_fixed_model_horizon_and_absent_types_not_invented():
    result = _run()
    examples = result["examples"].set_index("example_type")
    assert examples.loc["successful_warning", "series_id"] == "A"
    assert examples.loc["successful_warning", "month_index"] == 13
    assert examples.loc["false_alert", "series_id"] == "C"
    assert examples.loc["missed_event", "series_id"] == "B"
    assert examples.loc["missed_event", "month_index"] == 18
    assert examples.selection_rule.eq(EXAMPLE_RULE).all() and examples.k.eq(3).all()
    only_one_type = _run(_predictions([("A", 13, .9)]))["examples"]
    assert only_one_type.example_type.tolist() == ["successful_warning"]


def test_examples_fall_back_to_k1_only_for_a_type_absent_in_primary_k3():
    primary = _predictions([("A", 13, .9)])
    fallback = _predictions([("A", 15, .1), ("C", 11, .9)], k=1)
    examples = _run(pd.concat([primary, fallback], ignore_index=True))["examples"].set_index("example_type")
    assert examples.loc["successful_warning", "k"] == 3
    assert examples.loc["false_alert", "k"] == 1
    assert examples.loc["missed_event", "k"] == 1


def test_empty_admitted_group_keeps_zero_exposure_and_unknown_event_metrics():
    prediction = _predictions([("C", 23, .9)], k=1)
    prediction["fully_known"], prediction["label"] = False, np.nan
    result = _run(prediction)
    assert result["metrics_row"].iloc[0].cases == 0
    assert result["metrics_event"].iloc[0].monitored_cases == 0
    assert np.isnan(result["metrics_event"].iloc[0].event_recall)
    assert result["bootstrap_intervals"].series_count.eq(0).all()
    assert result["bootstrap_intervals"].defined_resamples.eq(0).all()
    assert result["event_matches"].empty and result["alerts"].empty and result["examples"].empty


def test_evaluation_is_deterministic_order_invariant_and_does_not_mutate_inputs():
    predictions, events, metadata, config = _predictions(), _events(), _metadata(), _config()
    old = (predictions.copy(deep=True), events.copy(deep=True), metadata.copy(deep=True), deepcopy(config))
    first = evaluate_predictions(predictions, events, metadata, config)
    second = evaluate_predictions(predictions.iloc[::-1], events.iloc[::-1], metadata.iloc[::-1], config)
    for key in first:
        assert_frame_equal(first[key], second[key])
    for actual, expected in zip((predictions, events, metadata), old[:3]):
        assert_frame_equal(actual, expected)
    assert config == old[3]


def test_evaluation_cannot_accept_train_rows_or_change_locked_scenario_list():
    with pytest.raises(ValueError, match="validation/test"):
        _run(_predictions(cohort="train"), _events("train"), _metadata("train"))
    config = _config()
    config["evaluation"]["scenarios"] = ["strong_precursor"]
    with pytest.raises(ValueError, match="five locked"):
        _run(config=config)
