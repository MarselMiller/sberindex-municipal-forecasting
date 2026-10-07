"""Synthetic contracts for retrospective segmentation and prefix independence."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sberforecast import offline_detection as offline
from sberforecast.online_detection import run_stream


PREPARATION = dict(warmup_observations=4, relative_scale_floor=0.03,
                   absolute_scale_floor=1.0, release_lag_months=0, cooldown_months=2)
PARAMETERS = dict(model="l2", min_size=2, jump=1, penalty=4.0)
METHODS = ["PELT", "BinSeg"]


def residuals(errors=None, *, months=12, start="2024-01", series_id="synthetic"):
    if errors is None:
        errors = np.zeros(months)
    errors = np.asarray(errors, dtype=float)
    return pd.DataFrame({
        "series_id": series_id,
        "observation_period": pd.period_range(start, periods=len(errors), freq="M").astype(str),
        "y_true": 100.0 + errors, "y_pred": 100.0,
    })


@pytest.mark.parametrize("method", METHODS)
def test_known_level_shift_detected_and_cost_explained(method):
    result = offline.run_offline(residuals([0.0] * 6 + [12.0] * 6), method, PARAMETERS, PREPARATION)
    assert result.errors.empty
    assert result.series_diagnostics["status"].tolist() == ["complete"]
    assert result.breakpoints["breakpoint_month"].tolist() == ["2024-07"]
    point = result.breakpoints.iloc[0]
    assert point["finite_boundary_index"] == 6
    assert point["estimated_shift"] == pytest.approx(12.0)
    assert point["shift_units"] == "RUB"
    assert point["eligible_evaluation"]
    assert point["gain_l2"] == pytest.approx(48.0)
    assert point["total_sse_l2"] == pytest.approx(0.0)
    assert point["penalized_objective"] == pytest.approx(4.0)
    assert point["penalized_gain_l2"] == pytest.approx(44.0)
    assert point["before_start_period"] == "2024-01"
    assert point["before_end_period"] == "2024-06"
    assert point["after_start_period"] == "2024-07"
    assert point["after_end_period"] == "2024-12"
    assert point["analyzed_end_period"] == "2024-12"
    assert result.segments["n_observations"].tolist() == [6, 6]


@pytest.mark.parametrize("method", METHODS)
def test_no_change_has_no_mandatory_breakpoint(method):
    result = offline.run_offline(residuals(), method, PARAMETERS, PREPARATION)
    assert result.breakpoints.empty
    assert len(result.segments) == 1
    assert result.series_diagnostics.iloc[0]["n_breakpoints"] == 0
    assert result.series_diagnostics.iloc[0]["penalized_objective"] == 0.0


@pytest.mark.parametrize("method", METHODS)
def test_fixed_parameters_reproduce_exact_results(method):
    data = residuals(np.random.default_rng(42).normal(size=12) + np.r_[np.zeros(6), np.ones(6) * 10])
    first = offline.run_offline(data, method, PARAMETERS, PREPARATION)
    second = offline.run_offline(data, method, PARAMETERS, PREPARATION)
    for field in ["stream", "breakpoints", "segments", "series_diagnostics", "errors"]:
        pd.testing.assert_frame_equal(getattr(first, field), getattr(second, field))


def test_calibration_and_monitoring_exposure_equal_e04a():
    data = residuals([4, -2, 1, 3, 10, 5, 0, 2, 3, -1, 2, 7])
    data.loc[5, "y_true"] = np.nan
    actual = offline.prepare_streams(data, PREPARATION)
    reference = run_stream(data, "CUSUM", dict(allowance=0.5, threshold=1000), PREPARATION)
    assert actual["phase"].tolist() == reference["phase"].tolist()
    monitoring = actual["phase"] == "monitoring"
    np.testing.assert_array_equal(actual.loc[monitoring, "center"], reference.loc[monitoring, "center"])
    np.testing.assert_array_equal(actual.loc[monitoring, "scale"], reference.loc[monitoring, "scale"])
    np.testing.assert_array_equal(actual.loc[monitoring, "z"], reference.loc[monitoring, "z"])
    assert np.isfinite(actual.loc[actual["phase"] == "warmup", "z"]).all()
    assert reference.loc[reference["phase"] == "warmup", "z"].isna().all()
    assert actual["center"].unique().tolist() == [2.0]
    assert actual["scale"].unique().tolist() == [3.0]


def test_later_predictions_cannot_set_scale_floor():
    data = residuals([0.0] * 12)
    before = offline.prepare_streams(data, PREPARATION)
    data.loc[4:, ["y_true", "y_pred"]] = 1e12
    after = offline.prepare_streams(data, PREPARATION)
    np.testing.assert_array_equal(before["center"], after["center"])
    np.testing.assert_array_equal(before["scale"], after["scale"])
    assert after["scale"].iloc[0] == 3.0


@pytest.mark.parametrize("method", METHODS)
def test_missing_months_preserve_right_segment_calendar_date(method):
    data = residuals([0.0] * 6 + [12.0] * 6).drop(index=6)
    result = offline.run_offline(data, method, PARAMETERS, PREPARATION)
    assert result.stream["observation_period"].tolist() == pd.period_range("2024-01", "2024-12", freq="M").astype(str).tolist()
    missing = result.stream[result.stream["observation_period"] == "2024-07"].iloc[0]
    assert missing["phase"] == "missing"
    assert np.isnan(missing["error"]) and np.isnan(missing["z"])
    point = result.breakpoints.iloc[0]
    assert point["breakpoint_month"] == "2024-08"
    assert point["finite_boundary_index"] == 6
    assert point["before_end_period"] == "2024-06"
    assert point["after_start_period"] == "2024-08"
    assert point["boundary_gap_calendar_months"] == 1
    assert result.series_diagnostics.iloc[0]["n_finite"] == 11


@pytest.mark.parametrize("method", METHODS)
def test_warmup_breakpoints_kept_separately_not_benchmark_false_positives(method, monkeypatch):
    monkeypatch.setattr(offline, "_fit_boundaries", lambda *args: [2, 12])
    result = offline.run_offline(residuals([0.0] * 2 + [10.0] * 10), method, PARAMETERS, PREPARATION)
    assert result.breakpoints["breakpoint_month"].tolist() == ["2024-03"]
    assert result.breakpoints["phase"].tolist() == ["warmup"]
    assert not result.breakpoints.iloc[0]["eligible_evaluation"]
    diagnostic = result.series_diagnostics.iloc[0]
    assert diagnostic["n_warmup_breakpoints"] == 1
    assert diagnostic["n_evaluation_breakpoints"] == 0


def test_terminal_index_is_not_a_breakpoint_and_boundary_is_not_left_month(monkeypatch):
    monkeypatch.setattr(offline, "_fit_boundaries", lambda *args: [6, 12])
    result = offline.run_offline(residuals(), "PELT", PARAMETERS, PREPARATION)
    assert result.breakpoints["breakpoint_month"].tolist() == ["2024-07"]
    assert result.breakpoints["finite_boundary_index"].tolist() == [6]
    assert len(result.segments) == 2


@pytest.mark.parametrize("length", [0, 1, 2, 3])
def test_insufficient_calibration_does_not_fit(length, monkeypatch):
    def forbidden(*args):
        raise AssertionError("No fit should be attempted")
    monkeypatch.setattr(offline, "_fit_boundaries", forbidden)
    result = offline.run_offline(residuals(months=length), "PELT", PARAMETERS, PREPARATION)
    assert result.errors.empty and result.breakpoints.empty and result.segments.empty
    if length:
        assert result.series_diagnostics["status"].tolist() == ["not_ready"]
        assert result.stream["z"].isna().all()
    else:
        assert result.stream.empty and result.series_diagnostics.empty
    for frame, columns in [(result.stream, offline.STREAM_COLUMNS),
                           (result.breakpoints, offline.BREAKPOINT_COLUMNS),
                           (result.segments, offline.SEGMENT_COLUMNS)]:
        assert list(frame.columns) == columns


@pytest.mark.parametrize("method", METHODS)
def test_four_warmup_observations_can_fit_but_have_no_monitoring_exposure(method):
    result = offline.run_offline(residuals(months=4), method, PARAMETERS, PREPARATION)
    assert result.series_diagnostics["status"].tolist() == ["complete"]
    assert result.series_diagnostics["n_monitoring"].tolist() == [0]
    assert result.stream["phase"].tolist() == ["warmup"] * 4


def test_fit_receives_only_numeric_residuals_no_true_labels(monkeypatch):
    seen = []
    def fit(values, method, penalty, min_size, jump):
        assert isinstance(values, np.ndarray)
        assert values.dtype == np.float64 and values.ndim == 1
        assert len(values) == 12 and np.isfinite(values).all()
        seen.append(values.copy())
        return [len(values)]
    monkeypatch.setattr(offline, "_fit_boundaries", fit)
    plain = residuals()
    labelled = plain.assign(event_start_period="2024-07", future_event_label=999999,
                            scenario="level_up", split="test", error=1e12)
    a = offline.run_offline(plain, "PELT", PARAMETERS, PREPARATION)
    b = offline.run_offline(labelled, "PELT", PARAMETERS, PREPARATION)
    pd.testing.assert_frame_equal(a.stream, b.stream)
    pd.testing.assert_frame_equal(a.series_diagnostics, b.series_diagnostics)
    np.testing.assert_array_equal(seen[0], seen[1])


def test_nonfinite_truth_prediction_and_overflow_are_missing_errors():
    data = residuals()
    data.loc[1, "y_true"] = np.nan
    data.loc[6, "y_true"] = np.inf
    data.loc[7, "y_pred"] = -np.inf
    data.loc[8, ["y_true", "y_pred"]] = [np.finfo(float).max, -np.finfo(float).max]
    result = offline.prepare_streams(data, PREPARATION)
    assert result.loc[[1, 6, 7, 8], "phase"].tolist() == ["missing"] * 4
    assert result.loc[[1, 6, 7, 8], "error"].isna().all()
    assert result.loc[[1, 6, 7, 8], "finite_index"].isna().all()
    assert result.loc[5, "phase"] == "monitoring"


def test_series_are_independent_and_keys_are_sorted():
    data = pd.concat([residuals(series_id="b"), residuals(np.arange(12), series_id="a")])
    actual = offline.prepare_streams(data.sample(frac=1.0, random_state=42), PREPARATION)
    for sid in ["a", "b"]:
        expected = offline.prepare_streams(data[data["series_id"] == sid], PREPARATION)
        pd.testing.assert_frame_equal(actual[actual["series_id"] == sid].reset_index(drop=True), expected)
    assert actual["series_id"].unique().tolist() == ["a", "b"]


@pytest.mark.parametrize("case", ["missing_column", "missing_id", "missing_month", "duplicate"])
def test_invalid_residual_schema_is_rejected(case):
    data = residuals()
    if case == "missing_column":
        data = data.drop(columns="y_true")
    elif case == "missing_id":
        data.loc[0, "series_id"] = None
    elif case == "missing_month":
        data.loc[0, "observation_period"] = None
    else:
        data = pd.concat([data, data.iloc[[0]]])
    with pytest.raises(ValueError):
        offline.prepare_streams(data, PREPARATION)


@pytest.mark.parametrize("preparation", [
    dict(warmup_observations=0), dict(warmup_observations=True),
    dict(release_lag_months=-1), dict(relative_scale_floor=-0.1),
    dict(relative_scale_floor=np.nan), dict(absolute_scale_floor=0),
    dict(absolute_scale_floor=np.inf), dict(as_of="NaT"),
])
def test_invalid_preparation_is_rejected_even_on_empty_input(preparation):
    with pytest.raises(ValueError):
        offline.prepare_streams(residuals(months=0), dict(PREPARATION, **preparation))


@pytest.mark.parametrize("params", [
    dict(penalty=0), dict(penalty=np.inf), dict(model="rbf"),
    dict(min_size=1), dict(min_size=3), dict(jump=2), dict(n_bkps=2),
])
def test_unapproved_cost_or_stopping_parameters_rejected(params):
    with pytest.raises(ValueError):
        offline.run_offline(residuals(months=0), "PELT", dict(PARAMETERS, **params), PREPARATION)


def test_unknown_method_rejected_before_empty_input():
    with pytest.raises(ValueError, match="Unknown offline"):
        offline.run_offline(residuals(months=0), "CUSUM", PARAMETERS, PREPARATION)


@pytest.mark.parametrize("endpoints", [[], [6], [6, 6, 12], [1, 12], [True, 12], [6.5, 12]])
def test_invalid_library_boundaries_are_visible_failures(endpoints, monkeypatch):
    monkeypatch.setattr(offline, "_fit_boundaries", lambda *args: endpoints)
    result = offline.run_offline(residuals(), "PELT", PARAMETERS, PREPARATION)
    assert result.series_diagnostics["status"].tolist() == ["failed"]
    assert len(result.errors) == 1
    assert result.errors.iloc[0]["stage"] == "fit"
    assert result.breakpoints.empty and result.segments.empty


def test_fit_error_is_explicit_without_reserve(monkeypatch):
    def broken(*args):
        raise RuntimeError("library failure")
    monkeypatch.setattr(offline, "_fit_boundaries", broken)
    result = offline.run_offline(residuals(), "BinSeg", PARAMETERS, PREPARATION)
    assert result.series_diagnostics.iloc[0]["status"] == "failed"
    assert result.errors.iloc[0]["error"] == "library failure"
    assert result.errors.iloc[0]["error_type"] == "RuntimeError"
    assert result.breakpoints.empty


def test_publication_lag_limits_calibration_and_errors_at_boundary_month():
    data = residuals(np.arange(12))
    preparation = dict(PREPARATION, release_lag_months=1, as_of="2024-05")
    actual = offline.prepare_streams(data, preparation)
    assert actual["phase"].tolist() == ["warmup"] * 4 + ["unavailable"] * 8
    assert actual["calibration_ready"].all()
    assert actual["calibration_end_period"].unique().tolist() == ["2024-04"]
    assert actual.loc[3, "availability_date"] == "2024-05-31"
    assert actual.loc[4:, "error"].isna().all()
    assert actual.loc[4:, "z"].isna().all()
    before_boundary = offline.prepare_streams(data, dict(preparation, as_of="2024-05-30"))
    assert before_boundary["phase"].tolist() == ["warmup"] * 3 + ["unavailable"] * 9
    assert not before_boundary["calibration_ready"].any()


def test_explicit_later_fact_availability_is_respected():
    data = residuals()
    data["fact_available_date"] = pd.period_range("2024-01", periods=12, freq="M").end_time.normalize()
    data.loc[0, "fact_available_date"] = pd.Timestamp("2024-06-30")
    actual = offline.prepare_streams(data, dict(PREPARATION, as_of="2024-05"))
    assert actual.loc[0, "phase"] == "unavailable"
    assert actual.loc[4, "phase"] == "warmup"
    assert actual["calibration_end_period"].unique().tolist() == ["2024-05"]


def test_explicit_fact_cannot_precede_configured_release_lag():
    data = residuals()
    data["fact_available_date"] = pd.period_range("2024-01", periods=12, freq="M").end_time.normalize()
    with pytest.raises(ValueError, match="precedes"):
        offline.prepare_streams(data, dict(PREPARATION, release_lag_months=1))


@pytest.mark.parametrize("method", METHODS)
def test_prefix_refits_each_history_and_retains_unready_prefixes(method, monkeypatch):
    lengths = []
    def fit(values, *args):
        lengths.append(len(values))
        return [6, len(values)] if len(values) >= 8 else [len(values)]
    monkeypatch.setattr(offline, "_fit_boundaries", fit)
    actual = offline.prefix_stability(residuals([0.0] * 6 + [12.0] * 6), method, PARAMETERS, PREPARATION)
    assert lengths == [12] + list(range(4, 13))
    assert actual.prefix_diagnostics["status"].tolist() == ["not_ready"] * 3 + ["complete"] * 9
    assert actual.summary["full_sample_breakpoint"].tolist() == ["2024-07"]
    row = actual.summary.iloc[0]
    assert row["first_prefix_where_detected"] == "2024-08"
    assert row["first_prefix_offset_months"] == 1
    assert row["n_matched_prefixes"] == 5
    assert row["revision_span_months"] == 0
    assert row["n_presence_losses"] == 0
    assert row["is_hindsight_diagnostic"]
    assert actual.trajectories["prefix_end_period"].tolist() == pd.period_range("2024-01", "2024-12", freq="M").astype(str).tolist()


@pytest.mark.parametrize("method", METHODS)
def test_future_values_do_not_change_historical_prefix_fits(method):
    original = residuals([0.0] * 6 + [12.0] * 6)
    changed = original.copy()
    changed.loc[9:, "y_true"] += 1000.0
    first = offline.prefix_stability(original, method, PARAMETERS, PREPARATION)
    second = offline.prefix_stability(changed, method, PARAMETERS, PREPARATION)
    for field in ["prefix_diagnostics", "prefix_breakpoints"]:
        a, b = getattr(first, field), getattr(second, field)
        # Full-sample association legitimately changes with the future; the
        # historical segmentation and calibration themselves must not change.
        columns = [c for c in a if c != "is_temporary_without_full_match"]
        pd.testing.assert_frame_equal(a.loc[a["prefix_end_period"] <= "2024-09", columns].reset_index(drop=True),
                                      b.loc[b["prefix_end_period"] <= "2024-09", columns].reset_index(drop=True))


def test_prefix_fit_sees_only_available_facts_and_its_own_first_four_scaler(monkeypatch):
    observed = []
    def fit(values, *args):
        observed.append(values.copy())
        return [len(values)]
    monkeypatch.setattr(offline, "_fit_boundaries", fit)
    data = residuals([2, 4, 6, 8] + [100.0] * 8)
    actual = offline.prefix_stability(data, "PELT", PARAMETERS,
                                     dict(PREPARATION, release_lag_months=1))
    assert [len(v) for v in observed] == [12] + list(range(4, 13))
    assert actual.prefix_diagnostics.iloc[0]["n_finite"] == 0
    assert actual.prefix_diagnostics.iloc[3]["status"] == "not_ready"
    assert actual.prefix_diagnostics.iloc[4]["status"] == "complete"
    for values in observed[1:]:
        np.testing.assert_array_equal(values, observed[0][:len(values)])
    assert actual.prefix_diagnostics["prefix_end_period"].iloc[-1] == "2025-01"


def test_future_calibration_data_cannot_make_short_prefix_ready(monkeypatch):
    calls = []
    monkeypatch.setattr(offline, "_fit_boundaries", lambda values, *args: calls.append(len(values)) or [len(values)])
    data = residuals([1, 2, 3, 10000, 20000, 30000])
    actual = offline.prefix_stability(data, "PELT", PARAMETERS, PREPARATION)
    assert calls == [6, 4, 5, 6]
    early = actual.prefix_diagnostics.iloc[:3]
    assert early["status"].tolist() == ["not_ready"] * 3
    assert early["center"].isna().all() and early["scale"].isna().all()


def test_prefix_as_of_does_not_analyze_later_calendar_ends():
    actual = offline.prefix_stability(residuals(), "PELT", PARAMETERS, dict(PREPARATION, as_of="2024-06"))
    assert actual.prefix_diagnostics["prefix_end_period"].tolist() == pd.period_range("2024-01", "2024-06", freq="M").astype(str).tolist()
    assert actual.prefix_diagnostics["n_finite"].max() == 6


def test_full_reference_matching_is_one_to_one_and_uses_calendar_offsets():
    full = pd.DataFrame([dict(breakpoint_id="a", breakpoint_month="2024-06"),
                         dict(breakpoint_id="b", breakpoint_month="2024-08")])
    estimates = pd.DataFrame([dict(breakpoint_id="p", breakpoint_month="2024-07")])
    matches = offline._matching(full, estimates, 1)
    assert list(matches) == ["a"]
    assert matches["a"]["offset"] == 1
    assert offline._matching(full, estimates, 0) == {}


def test_prefix_records_revision_presence_loss_and_temporary_points(monkeypatch):
    def fit(values, *args):
        n = len(values)
        if n in {8, 9}:
            return [5, n]
        if n == 10:
            return [2, n]
        if n >= 11:
            return [6, n]
        return [n]
    monkeypatch.setattr(offline, "_fit_boundaries", fit)
    result = offline.prefix_stability(residuals(), "PELT", PARAMETERS, PREPARATION)
    row = result.summary.iloc[0]
    assert row["first_prefix_where_detected"] == "2024-08"
    assert row["first_breakpoint_estimate"] == "2024-06"
    assert row["minimum_estimated_month"] == "2024-06"
    assert row["maximum_estimated_month"] == "2024-07"
    assert row["revision_span_months"] == 1
    assert row["n_date_revisions"] == 1
    assert row["n_presence_losses"] == 1
    temporary = result.prefix_breakpoints[result.prefix_breakpoints["prefix_end_period"] == "2024-10"].iloc[0]
    assert temporary["is_temporary_without_full_match"]
    assert temporary["breakpoint_month"] == "2024-03"


def test_failed_prefix_cannot_be_treated_as_absence_loss(monkeypatch):
    def fit(values, *args):
        n = len(values)
        if n == 10:
            raise RuntimeError("prefix failure")
        return [6, n] if n >= 8 else [n]
    monkeypatch.setattr(offline, "_fit_boundaries", fit)
    result = offline.prefix_stability(residuals(), "PELT", PARAMETERS, PREPARATION)
    assert result.prefix_diagnostics.loc[result.prefix_diagnostics["prefix_end_period"] == "2024-10", "status"].tolist() == ["failed"]
    assert not result.trajectories.loc[result.trajectories["prefix_end_period"] == "2024-10", "comparison_available"].any()
    assert result.summary.iloc[0]["n_presence_losses"] == 0


def test_empty_prefix_result_preserves_all_schemas():
    result = offline.prefix_stability(residuals(months=0), "PELT", PARAMETERS, PREPARATION)
    assert result.full_breakpoints.empty and result.prefix_diagnostics.empty
    assert result.prefix_breakpoints.empty and result.trajectories.empty and result.summary.empty
    assert list(result.trajectories) == offline.TRAJECTORY_COLUMNS
    assert list(result.summary) == offline.SUMMARY_COLUMNS


def test_negative_matching_tolerance_rejected():
    with pytest.raises(ValueError):
        offline.prefix_stability(residuals(), "PELT", PARAMETERS, PREPARATION, matching_tolerance_months=-1)
