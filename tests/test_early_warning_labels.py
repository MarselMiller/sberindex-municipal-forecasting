"""Synthetic saved-error examples; no real shock counts or classifier fits."""
import copy

import numpy as np
import pandas as pd
import pytest

from sberforecast.early_warning_labels import admit_training_labels, build_early_warning_labels


@pytest.fixture
def config():
    return {"data": {"release_lag_months": 0}, "weak_label": {
        "past_window_months": 4, "confirmation_months": 3,
        "relative_scale_floor": 0.03, "absolute_scale_floor": 1.0,
        "strength_threshold": 3.0, "merge_gap_months": 2,
        "recovery_months": 2, "recovery_strength": 1.5,
    }, "early_warning": {"horizons_months": [1, 3]}}


def residuals(errors, uid="synthetic", prediction=100.0, start="2024-01"):
    periods = pd.period_range(start, periods=len(errors), freq="M").astype(str)
    return pd.DataFrame({"series_id": uid, "observation_period": periods,
        "y_true": prediction + np.asarray(errors, dtype=float), "y_pred": prediction,
        "available_period": periods, "model": "SeasonalNaiveYoY", "horizon": 1})


def queries(*periods, uid="synthetic"):
    return pd.DataFrame({"municipality_id": uid, "forecast_origin": [str(pd.Period(period, freq="M").end_time.normalize().date()) for period in periods]})


def case(result, origin, k):
    table = result["cases"]
    return table.loc[table.warning_origin_period.eq(origin) & table.k.eq(k)].iloc[0]


def test_onset_confirmation_warning_origin_and_frozen_scale_are_separate(config):
    result = build_early_warning_labels(residuals([0] * 4 + [12] * 6), queries("2024-04"), config)
    event = result["events"].iloc[0]
    assert event.onset_period == "2024-05"
    assert event.confirmation_period == "2024-07"
    assert event.confirmation_at == "2024-07-31"
    assert event.baseline_center == 0 and event.baseline_scale == 3
    row = case(result, "2024-04", 1)
    assert row.label == 1 and row.label_known_at == "2024-07-31"
    assert row.forecast_origin == "2024-04-30"
    assert row.evidence_start_period == "2024-01"
    assert row.evidence_end_period == "2024-07" and row.evidence_month_count == 7


def test_constant_spending_can_have_error_shift_target(config):
    table = residuals([0] * 4 + [12] * 5)
    table["y_true"] = 100.0
    table["y_pred"] = [100] * 4 + [88] * 5
    result = build_early_warning_labels(table, queries("2024-04"), config)
    assert table.y_true.nunique() == 1
    assert len(result["events"]) == 1
    assert result["events"].direction.iloc[0] == "positive"


@pytest.mark.parametrize("future", [[12, 12, -12], [12, 0, 12], [12, 12, np.nan], [12, 12, 8.9]])
def test_all_three_signed_finite_threshold_residuals_are_required(config, future):
    result = build_early_warning_labels(residuals([0] * 4 + future), queries("2024-04"), config)
    assert result["events"].empty


def test_inclusive_event_threshold_and_strict_recovery_threshold(config):
    result = build_early_warning_labels(residuals([0] * 4 + [9] * 3 + [4.5, 4.5, 0, 0]), queries("2024-09", "2024-10", "2024-11"), config)
    assert result["events"].onset_period.tolist() == ["2024-05"]
    states = result["origin_states"].set_index("warning_origin_period")
    assert states.loc["2024-09", "known_active_at_origin"]
    assert states.loc["2024-10", "recovery_streak_months"] == 1
    assert not states.loc["2024-11", "known_active_at_origin"]


def test_mad_and_absolute_floor_are_from_only_four_prior_errors(config):
    table = residuals([-2, 2, -2, 2, 12, 12, 12], prediction=1)
    result = build_early_warning_labels(table, queries("2024-04"), config)
    event = result["events"].iloc[0]
    assert event.baseline_center == 0
    assert event.baseline_mad == 2
    assert event.baseline_scale == pytest.approx(2.9652)
    tiny = build_early_warning_labels(residuals([0] * 4 + [3] * 3, prediction=1), queries("2024-04"), config)
    assert tiny["events"].baseline_scale.iloc[0] == 1


def test_overlapping_candidates_merge_and_long_unrecovered_shift_is_not_new_onset(config):
    result = build_early_warning_labels(residuals([0] * 4 + [12] * 7 + [60] * 6), queries("2024-11"), config)
    candidates = result["candidates"]
    assert len(result["events"]) == 1
    assert candidates.loc[candidates.is_candidate, "disposition"].eq("same_direction_continuation").sum() >= 2
    assert result["events"].continuation_count.iloc[0] >= 2
    assert case(result, "2024-11", 1).label == 0
    assert not case(result, "2024-11", 1).at_risk


def test_confirmed_active_mode_differs_from_retrospective_onset_membership(config):
    result = build_early_warning_labels(residuals([0] * 4 + [12] * 6), queries("2024-05", "2024-06", "2024-07"), config)
    states = result["origin_states"].set_index("warning_origin_period")
    assert states.loc["2024-05", "retrospective_inside_regime_at_origin"]
    assert not states.loc["2024-05", "known_active_at_origin"]
    assert not states.loc["2024-06", "known_active_at_origin"]
    assert states.loc["2024-07", "known_active_at_origin"]
    assert case(result, "2024-05", 1).at_risk


def test_missing_month_breaks_recovery_consecutiveness_and_marks_state_uncertain(config):
    result = build_early_warning_labels(residuals([0] * 4 + [12] * 4 + [0, np.nan, 0, 0]), queries("2024-10", "2024-11", "2024-12"), config)
    states = result["origin_states"].set_index("warning_origin_period")
    assert states.loc["2024-10", "known_active_at_origin"]
    assert states.loc["2024-10", "state_uncertain"]
    assert states.loc["2024-11", "recovery_streak_months"] == 1
    assert states.loc["2024-11", "known_active_at_origin"]
    assert not states.loc["2024-12", "known_active_at_origin"]
    assert result["regimes"].recovery_start_period.iloc[0] == "2024-11"


def test_opposite_onset_is_a_new_event_with_confirmation_after_old_recovery(config):
    result = build_early_warning_labels(residuals([0] * 4 + [20] * 7 + [-20] * 6), queries("2024-11", "2025-01"), config)
    assert result["events"].direction.tolist() == ["positive", "negative"]
    assert result["events"].onset_period.tolist() == ["2024-05", "2024-12"]
    assert result["events"].confirmation_period.tolist() == ["2024-07", "2025-02"]


def test_recovery_then_later_same_direction_shift_is_a_separate_event(config):
    # A gradual return recovers the original anchor without meeting the
    # separate opposite-shift criterion against the moving four-month past.
    result = build_early_warning_labels(residuals([0] * 4 + [12] * 4 + [6, 6] + [0] * 5 + [12] * 4), queries("2024-04"), config)
    assert result["events"].onset_period.tolist() == ["2024-05", "2025-04"]
    assert result["events"].direction.tolist() == ["positive", "positive"]


def test_abrupt_return_to_old_error_baseline_can_be_an_opposite_weak_onset(config):
    # The target is a saved-error shift relative to its recent past. It does
    # not distinguish economic shocks from reversals of forecast error.
    result = build_early_warning_labels(residuals([0] * 4 + [12] * 4 + [0] * 5), queries("2024-08"), config)
    assert result["events"].onset_period.tolist() == ["2024-05", "2024-09"]
    assert result["events"].direction.tolist() == ["positive", "negative"]
    assert result["regimes"].recovery_start_period.iloc[0] == "2024-09"


def test_future_truncation_and_future_values_do_not_change_causal_prefix_state(config):
    original = residuals([0] * 4 + [12] * 6 + [0, 0, -30, -30, -30])
    samples = queries("2024-05", "2024-06", "2024-07", "2024-09")
    full = build_early_warning_labels(original, samples, config)["origin_states"]
    truncated = build_early_warning_labels(original.iloc[:9], samples, config)["origin_states"]
    changed = original.copy()
    changed.loc[9:, "y_true"] = 7777
    altered = build_early_warning_labels(changed, samples, config)["origin_states"]
    causal_columns = [column for column in full if column != "retrospective_inside_regime_at_origin"]
    pd.testing.assert_frame_equal(full[causal_columns], truncated[causal_columns])
    pd.testing.assert_frame_equal(full[causal_columns], altered[causal_columns])


def test_positive_label_does_not_become_known_early_for_longer_horizon(config):
    partial = build_early_warning_labels(residuals([0] * 4 + [12] * 3), queries("2024-04"), config)
    assert case(partial, "2024-04", 1).label == 1
    long = case(partial, "2024-04", 3)
    assert not long.fully_known and np.isnan(long.label)
    assert long.right_censored and long.label_known_at == ""


def test_positive_and_negative_labels_have_same_uniform_window_availability(config):
    source = pd.concat([residuals([0] * 4 + [12] * 6, uid="positive"), residuals([0] * 10, uid="negative")])
    samples = pd.concat([queries("2024-04", uid="positive"), queries("2024-04", uid="negative")])
    cases = build_early_warning_labels(source, samples, config)["cases"]
    assert cases.loc[cases.k.eq(1), "label"].tolist() == [1, 0]
    assert cases.loc[cases.k.eq(1), "label_known_at"].tolist() == ["2024-07-31"] * 2
    assert cases.loc[cases.k.eq(3), "label_known_at"].tolist() == ["2024-09-30"] * 2


def test_training_admission_requires_uniform_label_confirmation_and_at_risk(config):
    result = build_early_warning_labels(residuals([0] * 4 + [12] * 6), queries("2024-04", "2024-07"), config)
    assert admit_training_labels(result["cases"], "2024-06").empty
    admitted = admit_training_labels(result["cases"], "2024-07")
    assert len(admitted) == 1
    assert admitted.k.iloc[0] == 1 and admitted.forecast_origin.iloc[0] == "2024-04-30"


def test_explicit_fit_date_never_admits_later_same_month_label(config):
    result = build_early_warning_labels(residuals([0] * 4 + [12] * 6), queries("2024-04"), config)
    assert admit_training_labels(result["cases"], "2024-07-01").empty
    assert admit_training_labels(result["cases"], "2024-07-30T23:59:59+03:00").empty
    assert len(admit_training_labels(result["cases"], "2024-07-31T00:00:00+03:00")) == 1
    assert len(admit_training_labels(result["cases"], "2024-07")) == 1


def test_left_history_missing_past_missing_future_and_right_censor_are_distinct(config):
    source = residuals([0] * 12)
    source.loc[5, "y_pred"] = np.nan
    result = build_early_warning_labels(source, queries("2024-02", "2024-04", "2024-06", "2024-10"), config)
    assert case(result, "2024-02", 1).left_insufficient
    assert case(result, "2024-04", 1).missing_future
    assert case(result, "2024-06", 1).missing_past
    assert case(result, "2024-10", 1).right_censored
    assert result["cases"].label.isna().all()


def test_missing_calendar_row_is_not_compressed_or_filled(config):
    source = residuals([0] * 4 + [12] * 5).drop(index=5)
    result = build_early_warning_labels(source, queries("2024-04"), config)
    row = result["candidates"].loc[lambda frame: frame.onset_period.eq("2024-05")].iloc[0]
    assert row.status == "missing_confirmation_residuals"
    assert np.isnan(case(result, "2024-04", 1).label)


def test_delayed_availability_cannot_leak_into_causal_origin_state(config):
    source = residuals([0] * 4 + [12] * 6)
    source.loc[6, "available_period"] = "2024-08"
    result = build_early_warning_labels(source, queries("2024-04", "2024-07", "2024-08"), config)
    states = result["origin_states"].set_index("warning_origin_period")
    assert not states.loc["2024-07", "known_active_at_origin"]
    assert states.loc["2024-08", "known_active_at_origin"]
    assert case(result, "2024-04", 1).label_known_at == "2024-08-31"
    assert admit_training_labels(result["cases"], "2024-07").empty


def test_unavailable_confirmation_values_cannot_change_earlier_prefix_state(config):
    source = residuals([0] * 4 + [12] * 6)
    source.loc[6, "available_period"] = "2024-09"
    changed = source.copy()
    changed.loc[6, "y_true"] = -1000
    samples = queries("2024-07", "2024-08")
    original_states = build_early_warning_labels(source, samples, config)["origin_states"]
    changed_states = build_early_warning_labels(changed, samples, config)["origin_states"]
    causal_columns = [column for column in original_states if column != "retrospective_inside_regime_at_origin"]
    pd.testing.assert_frame_equal(original_states[causal_columns], changed_states[causal_columns])


def test_old_delayed_regime_dependency_postpones_both_label_classes(config):
    # An unavailable May confirmation later establishes an old positive
    # regime. It can turn December's same-direction candidate into a
    # continuation although May is outside the local Aug..Feb evidence.
    old_regime = residuals([0] * 4 + [12] * 7 + [60] * 5, uid="old_regime")
    new_onset = residuals([0] * 11 + [60] * 5, uid="new_onset")
    old_regime.loc[4, "available_period"] = "2025-06"
    new_onset.loc[4, "available_period"] = "2025-06"
    samples = pd.concat([queries("2024-11", uid="old_regime"), queries("2024-11", uid="new_onset")], ignore_index=True)
    result = build_early_warning_labels(pd.concat([old_regime, new_onset]), samples, config)
    cases = result["cases"].loc[lambda frame: frame.k.eq(1)]
    assert cases.label.tolist() == [0, 1]
    assert cases.label_known_at.tolist() == ["2025-06-30"] * 2
    assert cases.evidence_start_period.tolist() == ["2024-08"] * 2
    assert cases.regime_history_dependency_start_period.tolist() == ["2024-01"] * 2
    assert admit_training_labels(result["cases"], "2025-02").empty
    assert len(admit_training_labels(result["cases"], "2025-06")) == 4


def test_every_sample_and_horizon_retained_including_absent_series(config):
    samples = pd.concat([queries("2024-04", "2024-10"), queries("2024-04", "2024-10", uid="absent")], ignore_index=True)
    result = build_early_warning_labels(residuals([0] * 12), samples, config)
    assert len(result["origin_states"]) == len(samples)
    assert len(result["cases"]) == len(samples) * 2
    assert result["cases"].loc[result["cases"].municipality_id.eq("absent"), "label"].isna().all()
    pd.testing.assert_frame_equal(result["origin_states"][["municipality_id", "forecast_origin"]], samples)


def test_error_is_recomputed_from_saved_prediction_and_inputs_unchanged(config):
    source = residuals([0] * 10)
    source["error"] = 999
    before = source.copy(deep=True)
    cfg_before = copy.deepcopy(config)
    assert build_early_warning_labels(source, queries("2024-04"), config)["events"].empty
    pd.testing.assert_frame_equal(source, before)
    assert config == cfg_before


def test_saved_residual_schema_allows_absent_forecast_placeholders(config):
    source = residuals([np.nan] * 4 + [0] * 9)
    source.loc[:3, ["y_pred", "horizon", "model"]] = np.nan
    source["source_prediction_present"] = source.y_pred.notna()
    source["status"] = "ok"
    source.loc[:3, "status"] = np.nan
    result = build_early_warning_labels(source, queries("2024-04", "2024-08"), config)
    assert case(result, "2024-04", 1).left_insufficient
    assert np.isnan(case(result, "2024-04", 1).label)
    assert case(result, "2024-08", 1).fully_known
    assert case(result, "2024-08", 1).label == 0


@pytest.mark.parametrize("missing_metadata", ["horizon", "model"])
def test_finite_saved_forecast_cannot_have_missing_identity(config, missing_metadata):
    source = residuals([0] * 10)
    source.loc[0, missing_metadata] = np.nan
    with pytest.raises(ValueError, match="missing metadata"):
        build_early_warning_labels(source, queries("2024-04"), config)


@pytest.mark.parametrize("change", ["duplicate", "horizon", "model", "backdated_availability"])
def test_invalid_saved_forecast_or_time_identity_rejected(config, change):
    source = residuals([0] * 10)
    if change == "duplicate":
        source = pd.concat([source, source.iloc[:1]])
    elif change == "horizon":
        source.loc[0, "horizon"] = 3
    elif change == "model":
        source.loc[0, "model"] = "OtherModel"
    else:
        source.loc[0, "available_period"] = "2023-12"
    with pytest.raises(ValueError):
        build_early_warning_labels(source, queries("2024-04"), config)


def test_missing_availability_needs_explicit_project_assumption(config):
    source = residuals([0] * 10).drop(columns="available_period")
    result = build_early_warning_labels(source, queries("2024-04"), config)
    assert "assumed_observation_period_plus_0" in result["cases"].availability_basis.iloc[0]
    bad = copy.deepcopy(config)
    bad["data"] = {}
    with pytest.raises(ValueError, match="explicit release_lag"):
        build_early_warning_labels(source, queries("2024-04"), bad)
