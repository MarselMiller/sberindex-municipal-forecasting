"""Synthetic tests of the independent E04a causal detector implementation."""
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from sberforecast.online_detection import BOCPD, CUSUM, EWMA, run_stream


PREPARATION = {
    "warmup_observations": 4, "relative_scale_floor": 0.03,
    "absolute_scale_floor": 1.0, "release_lag_months": 0,
    "cooldown_months": 2,
}
METHODS = [
    ("CUSUM", {"allowance": 0.5, "threshold": 3.0}),
    ("EWMA", {"alpha": 0.3, "threshold": 3.0}),
    ("BOCPD", {"hazard": 1 / 24, "threshold": 0.4}),
]


def stream(errors, *, start="2024-01", series_id="1", prediction=100.0):
    errors = np.asarray(errors, dtype=float)
    return pd.DataFrame({
        "series_id": series_id,
        "observation_period": pd.period_range(start, periods=len(errors), freq="M").astype(str),
        "y_true": prediction + errors,
        "y_pred": prediction,
    })


@pytest.mark.parametrize("method,params", METHODS)
def test_future_changes_do_not_change_past_signal_or_preparation(method, params):
    original = stream([0.0, 1.0, -1.0, 0.0, 3.0, 4.0, 5.0, 1.0, 0.0, 0.0])
    changed = original.copy()
    changed.loc[7:, "y_true"] = [1e7, -1e7, 1e6]
    changed.loc[7:, "y_pred"] = [1e6, 1e4, 1e3]
    before = run_stream(original, method, params, PREPARATION)
    after = run_stream(changed, method, params, PREPARATION)
    assert_frame_equal(before.iloc[:7], after.iloc[:7])


@pytest.mark.parametrize("method,params", METHODS)
def test_current_error_cannot_change_its_precalibration(method, params):
    original = stream([1.0, 2.0, 3.0, 4.0, 5.0, 0.0])
    changed = original.copy()
    changed.loc[4, ["y_true", "y_pred"]] = [1e7, 1e6]
    before = run_stream(original, method, params, PREPARATION)
    after = run_stream(changed, method, params, PREPARATION)
    columns = ["center", "scale", "calibration_end_period", "calibration_count"]
    assert_frame_equal(before.loc[4:4, columns], after.loc[4:4, columns])
    assert before.loc[4, "center"] == 2.5
    assert before.loc[4, "scale"] == 3.0  # 3% of prior warmup predictions only.
    assert before.loc[4, "calibration_end_period"] == "2024-04"
    assert np.isnan(before.loc[3, "center"])
    assert np.isnan(before.loc[3, "score"])


@pytest.mark.parametrize("method,params", METHODS)
def test_missing_months_are_preserved_and_warmup_counts_available_errors(method, params):
    data = stream([0.0, 0.0, np.nan, 0.0, 0.0, 0.0, 0.0]).drop(index=[1])
    result = run_stream(data, method, params, PREPARATION)
    assert result.observation_period.tolist() == [f"2024-{month:02d}" for month in range(1, 8)]
    assert result.phase.tolist() == [
        "warmup", "missing", "missing", "warmup", "warmup", "warmup", "monitoring",
    ]
    assert result.loc[[1, 2], "error"].isna().all()
    assert not result.loc[[1, 2], "is_alarm"].any()
    assert result.loc[6, "n_updates_since_reset"] == 1
    assert result.loc[6, "calibration_end_period"] == "2024-06"


@pytest.mark.parametrize("method,params", METHODS)
def test_publication_lag_and_exact_as_of_boundary(method, params):
    data = stream([0.0] * 4 + [50.0, 50.0], start="2023-10")
    config = {**PREPARATION, "release_lag_months": 1, "as_of": "2024-03-30"}
    before_release = run_stream(data, method, params, config)
    assert before_release.loc[4, "observation_period"] == "2024-02"
    assert before_release.loc[4, "availability_date"] == "2024-03-31"
    assert before_release.loc[4, "phase"] == "unavailable"
    assert np.isnan(before_release.loc[4, "error"])
    after_release = run_stream(data, method, params, {**config, "as_of": "2024-03-31"})
    assert after_release.loc[4, "phase"] == "monitoring"
    assert after_release.loc[5, "phase"] == "unavailable"
    alarms = after_release[after_release.is_alarm]
    assert alarms.signal_date.eq(alarms.availability_date).all()
    assert after_release.loc[3, "availability_date"] == "2024-02-29"


def test_state_updates_during_calendar_cooldown_and_survives_missing_month():
    data = stream([0.0] * 4 + [100.0, 100.0, np.nan, 100.0])
    result = run_stream(data, "CUSUM", {"allowance": 0.5, "threshold": 3.0}, PREPARATION)
    assert result.loc[4, "is_alarm"]
    assert result.loc[4, "state_reset"]
    assert result.loc[5, "cooldown_suppressed"]
    assert result.loc[5, "crossed_threshold"]
    assert not result.loc[5, "is_alarm"]
    assert result.loc[5, "n_updates_since_reset"] == 1
    assert result.loc[6, "phase"] == "missing"
    # Two suppressed calendar months are June and July, even though July is missing.
    assert result.loc[7, "is_alarm"]
    assert not result.loc[7, "cooldown_suppressed"]
    assert result.loc[7, "n_updates_since_reset"] == 2


def test_scaling_zero_variance_has_explicit_absolute_floor():
    result = run_stream(stream([0.0] * 6, prediction=0.0), "EWMA", {}, PREPARATION)
    assert result.loc[4:, "center"].eq(0.0).all()
    assert result.loc[4:, "scale"].eq(1.0).all()
    assert result.loc[4:, "z"].eq(0.0).all()
    assert not result.is_alarm.any()


@pytest.mark.parametrize("method,params", METHODS)
def test_series_are_calibrated_and_processed_independently(method, params):
    one = stream([1.0, 2.0, 3.0, 4.0, 20.0, 30.0], series_id="1")
    two = stream([100.0] * 6, series_id="2", prediction=1e5)
    isolated = run_stream(one, method, params, PREPARATION)
    joint = run_stream(pd.concat([two, one]).sample(frac=1, random_state=7), method, params, PREPARATION)
    assert_frame_equal(isolated, joint[joint.series_id.eq("1")].reset_index(drop=True))


def test_cusum_matches_nist_tabular_recursion_in_both_directions():
    positive = CUSUM(allowance=0.5, threshold=2.0)
    negative = CUSUM(allowance=0.5, threshold=2.0)
    scores = [positive.update(z).score for z in [1.0, -0.2, 2.0, 2.0]]
    assert scores == pytest.approx([0.5, 0.0, 1.5, 3.0])
    assert [negative.update(-z).score for z in [1.0, -0.2, 2.0, 2.0]] == pytest.approx(scores)
    equality = CUSUM(allowance=0.0, threshold=1.0).update(1.0)
    assert not equality.crossed_threshold  # Strict > is fixed for all methods.


def test_ewma_matches_finite_start_variance_formula():
    detector = EWMA(alpha=0.3, threshold=3.0)
    update = detector.update(2.0)
    assert update.diagnostics["ewma_value"] == pytest.approx(0.6)
    assert update.diagnostics["ewma_sd"] == pytest.approx(0.3)
    assert update.score == pytest.approx(2.0)
    second = detector.update(0.0)
    expected_sd = np.sqrt(0.3 / 1.7 * (1 - 0.7 ** 4))
    assert second.score == pytest.approx(0.42 / expected_sd)
    assert EWMA(alpha=1.0).update(-2.0).score == 2.0


@pytest.mark.parametrize("values", [
    [0.0] * 20,
    [0.0] * 6 + [1e150, -1e150, 1e150, 0.0],
    [0.1, -0.1, 0.2, -0.2] * 4 + [5.0] * 8,
])
def test_bocpd_posterior_is_normalized_and_constant_hazard_cp_is_not_score(values):
    detector = BOCPD(hazard=1 / 24)
    for value in values:
        update = detector.update(value)
        assert np.isfinite(detector.posterior).all()
        assert detector.posterior.sum() == pytest.approx(1.0, abs=1e-12)
        assert (detector.posterior >= 0).all()
        assert update.diagnostics["posterior_cp_probability"] == pytest.approx(1 / 24, abs=1e-12)
        assert np.isfinite(update.score)
        assert 0 <= update.score <= 1


def test_bocpd_recent_run_score_responds_to_data_and_shift():
    steady = BOCPD(hazard=1 / 24, threshold=0.4)
    changed = BOCPD(hazard=1 / 24, threshold=0.4)
    prefix = [0.1, -0.1, 0.2, -0.2] * 3
    for value in prefix:
        assert steady.update(value).score == changed.update(value).score
    steady_updates = [steady.update(value) for value in [0.1, -0.1, 0.2, -0.2]]
    changed_updates = [changed.update(value) for value in [5.0] * 4]
    assert max(update.score for update in changed_updates) > max(update.score for update in steady_updates)
    assert any(update.crossed_threshold for update in changed_updates)
    assert not any(update.crossed_threshold for update in steady_updates)


def test_bocpd_initialization_and_alarm_reset_cannot_create_automatic_recent_run_alarm():
    detector = BOCPD(threshold=0.001)
    for value in [0.0, 1.0]:
        update = detector.update(value)
        assert not update.crossed_threshold
        assert not update.diagnostics["bocpd_armed"]
        assert update.score == 0.0
    detector.update(2.0)
    detector.reset()
    update = detector.update(10.0)
    assert not update.crossed_threshold
    assert update.diagnostics["n_updates_since_reset"] == 1


@pytest.mark.parametrize("constructor", [CUSUM, EWMA, BOCPD])
def test_detectors_reject_nonfinite_inputs(constructor):
    for value in [np.nan, np.inf, -np.inf]:
        with pytest.raises(ValueError, match="finite"):
            constructor().update(value)


@pytest.mark.parametrize("method,params", [
    ("CUSUM", {"allowance": -1.0}),
    ("EWMA", {"alpha": 0.0}),
    ("EWMA", {"alpha": 1.1}),
    ("BOCPD", {"hazard": 1.0}),
    ("BOCPD", {"threshold": 1.1}),
    ("BOCPD", {"prior_beta": 0.0}),
])
def test_invalid_parameters_are_rejected(method, params):
    with pytest.raises(ValueError):
        run_stream(stream([0.0] * 5), method, params, PREPARATION)


def test_duplicate_month_keys_are_rejected_and_empty_stream_is_well_formed():
    data = stream([0.0])
    with pytest.raises(ValueError, match="Duplicate"):
        run_stream(pd.concat([data, data]), "CUSUM", {}, PREPARATION)
    empty = run_stream(data.iloc[:0], "CUSUM", {}, PREPARATION)
    assert empty.empty
    assert {"series_id", "observation_period", "phase", "is_alarm"} <= set(empty.columns)
