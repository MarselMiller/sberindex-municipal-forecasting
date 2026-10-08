"""Synthetic-only chronology and semantics tests; no learners or real inputs."""
import numpy as np
import pandas as pd
import pytest

from sberforecast import national_local_persistence_model as persistence
from sberforecast.data import get_prefix, period_end
from sberforecast.models import baseline_predict
from sberforecast.national_local import (
    build_local_training, build_national, national_forecast,
)
from sberforecast.national_local_persistence_model import (
    MODEL_NAME, NationalLocalPersistence, PersistenceResult,
)


BASELINE = {"yearly_growth_window": 3, "yearly_growth_bounds": [0.5, 2.0]}


@pytest.fixture
def toy_panel():
    months = pd.date_range("2023-01-01", periods=24, freq="MS")
    base = 100.0 + np.arange(24)
    return pd.DataFrame({"a": base, "b": 2 * base, "c": 4 * base,
                         "never": np.nan}, index=months)


def predict(panel, origin="2023-12", horizon=3, ids=None, lag=0, national=None):
    national = build_national(panel, lag) if national is None else national
    return NationalLocalPersistence(national, lag).predict(
        panel, origin, horizon, ids or ["a", "b"], BASELINE,
    )


def test_exact_full_panel_national_component_and_causal_forecaster_reused(toy_panel):
    national = build_national(toy_panel)
    result = predict(toy_panel, ids=["a"], national=national)
    assert isinstance(result, PersistenceResult)
    assert national.N.iloc[0] == 200.0
    assert build_national(toy_panel[["a"]]).N.iloc[0] == 100.0
    expected = national_forecast(national, "2023-12", 3, BASELINE)
    assert result.national_forecast == expected
    assert expected["N_hat"] == 204.0
    assert result.forecasts.y_pred.iloc[0] == 102.0


@pytest.mark.parametrize("horizon", [1, 3, 6, 12])
def test_last_ratio_not_smoothed_and_exact_nominal_reconstruction(toy_panel, horizon):
    toy_panel.loc["2023-12-01", "a"] = 888.0
    result = predict(toy_panel, horizon=horizon, ids=["a", "b", "c"])
    rows = result.forecasts
    np.testing.assert_array_equal(rows.anchor, [2.0, 0.5, 1.0])
    np.testing.assert_array_equal(rows.ratio_hat, rows.anchor)
    np.testing.assert_array_equal(rows.y_pred, rows.national_hat * rows.anchor)
    assert rows.last_ratio_period.eq(pd.Timestamp("2023-12-01")).all()
    assert rows.ratio_national_denominator.eq(444.0).all()
    assert rows.status.eq("native").all()
    assert rows.effective_model.eq(MODEL_NAME).all()
    assert not rows.fit_called.any()


@pytest.mark.parametrize("lag,origin", [(0, "2023-12"), (1, "2024-01"), (2, "2024-02")])
def test_future_expenses_and_future_only_municipality_do_not_change_forecast(toy_panel, lag, origin):
    cutoff = (pd.Period(origin, "M") - lag).to_timestamp()
    before = predict(toy_panel, origin=origin, lag=lag)
    changed = toy_panel.copy()
    changed.loc[changed.index > cutoff] *= 10000
    changed["future_only"] = np.nan
    changed.loc[changed.index > cutoff, "future_only"] = 1e9
    after = predict(changed, origin=origin, lag=lag)
    pd.testing.assert_frame_equal(before.forecasts, after.forecasts)
    assert before.national_forecast["N_hat"] == after.national_forecast["N_hat"]
    assert before.national_forecast["N_actual"] != after.national_forecast["N_actual"]
    assert (after.forecasts.ratio_available_at <= period_end(pd.Period(origin, "M"))).all()
    assert (after.forecasts.ratio_national_available_at <= period_end(pd.Period(origin, "M"))).all()


def test_future_national_fact_is_evaluation_only(toy_panel):
    national = build_national(toy_panel)
    before = predict(toy_panel, national=national)
    changed = national.copy()
    changed.loc["2024-03-01", "N"] = 1e9
    after = predict(toy_panel, national=changed)
    pd.testing.assert_frame_equal(before.forecasts, after.forecasts)
    assert before.national_forecast["N_actual"] != after.national_forecast["N_actual"]


@pytest.mark.parametrize("missing", [np.nan, np.inf])
def test_missing_latest_expense_uses_earlier_joint_ratio(toy_panel, missing):
    toy_panel.loc["2023-11-01", "a"] = 660.0
    toy_panel.loc["2023-12-01", "a"] = missing
    result = predict(toy_panel, ids=["a"])
    row = result.forecasts.iloc[0]
    assert row.last_ratio_period == pd.Timestamp("2023-11-01")
    assert row.ratio_expense_value == 660.0
    assert row.ratio_national_denominator == 440.0
    assert row.ratio_hat == 1.5
    assert row.ratio_staleness_months == 1
    assert row.status == "native"


@pytest.mark.parametrize("invalid", [0.0, -1.0, np.nan])
def test_invalid_latest_denominator_uses_last_earlier_valid_ratio(toy_panel, invalid):
    national = build_national(toy_panel)
    national.loc["2023-12-01", "N"] = invalid
    result = predict(toy_panel, ids=["a"], national=national)
    row = result.forecasts.iloc[0]
    assert row.last_ratio_period == pd.Timestamp("2023-11-01")
    assert row.ratio_hat == 0.5
    assert row.status == "native"


def test_infinite_latest_national_history_preserves_inherited_forecast_failure(toy_panel):
    national = build_national(toy_panel)
    national.loc["2023-12-01", "N"] = np.inf
    result = predict(toy_panel, national=national)
    assert result.national_forecast["status"] == "failed"
    assert result.forecasts.status.eq("failed").all()
    assert result.forecasts.failure_stage.eq("national_forecast").all()
    assert result.forecasts.y_pred.isna().all()
    assert result.forecasts.ratio_hat.isna().all()
    assert result.forecasts.last_ratio_period.eq(pd.Timestamp("2023-11-01")).all()
    np.testing.assert_array_equal(result.forecasts.anchor, [0.5, 1.0])


def test_no_finite_ratio_fails_entire_batch_without_dropping_rows(toy_panel):
    result = predict(toy_panel, ids=["a", "never"])
    rows = result.forecasts
    assert rows.municipality_id.tolist() == ["a", "never"]
    assert rows.status.eq("failed").all()
    assert rows.y_pred.isna().all() and rows.ratio_hat.isna().all()
    assert rows.failure_stage.eq("local_ratio").all()
    assert rows.ratio_status.tolist() == ["available", "missing"]
    assert rows.reason.tolist() == ["forecast_batch_contains_missing_causal_ratio", "no_causal_ratio"]
    assert rows.fallback_baseline_status.eq("").all()
    assert rows.anchor.iloc[0] == 0.5


def test_annual_forecast_remains_native_when_learned_training_pairs_are_empty(toy_panel):
    national = build_national(toy_panel)
    X, delta, trace = build_local_training(
        toy_panel, national, "2023-12", 12,
        release_lag_months=0, mode="legacy",
    )
    assert X.empty and len(delta) == 0 and trace.empty
    result = predict(toy_panel, horizon=12, national=national)
    assert result.forecasts.status.eq("native").all()
    assert result.forecasts.effective_model.eq(MODEL_NAME).all()
    assert result.forecasts.reason.eq("").all()
    assert not result.forecasts.fit_called.any()
    np.testing.assert_array_equal(result.forecasts.y_pred,
                                  result.forecasts.national_hat * result.forecasts.ratio_hat)
    expense_fallback, _ = baseline_predict(
        get_prefix(toy_panel[["a", "b"]], pd.Period("2023-12", "M"), 0),
        12, "SeasonalNaive", BASELINE,
    )
    # Numeric equality on this particular fixture does not make the parameter-
    # free decomposition a learned direct strategy's empty-training fallback.
    np.testing.assert_array_equal(result.forecasts.y_pred, expense_fallback[-1])


def test_national_last_value_fallback_is_separate_from_local_native_status(toy_panel):
    result = predict(toy_panel.iloc[:4], origin="2023-04", horizon=1)
    assert result.national_forecast["status"] == "fallback_last_value"
    assert result.national_forecast["effective_model"] == "LastValue"
    assert result.forecasts.status.eq("native").all()
    assert result.forecasts.national_forecast_status.eq("fallback_last_value").all()
    assert result.forecasts.fallback_baseline_status.eq("").all()


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf])
def test_invalid_national_prediction_fails_batch_without_expense_fallback(toy_panel, monkeypatch, bad):
    expected = national_forecast(build_national(toy_panel), "2023-12", 3, BASELINE)
    monkeypatch.setattr(persistence, "national_forecast", lambda *args, **kwargs: dict(expected, N_hat=bad))
    result = predict(toy_panel)
    assert result.forecasts.status.eq("failed").all()
    assert result.forecasts.y_pred.isna().all()
    assert result.forecasts.failure_stage.eq("national_forecast").all()
    assert result.forecasts.fallback_baseline_status.eq("").all()


def test_lag_metadata_mismatch_stays_explicit_failed_national_forecast(toy_panel):
    result = predict(toy_panel, origin="2024-01", lag=1, national=build_national(toy_panel, 0))
    assert result.forecasts.status.eq("failed").all()
    assert result.forecasts.failure_stage.eq("national_forecast").all()
    assert "release_lag_months" in result.national_forecast["reason"]


def test_no_additional_minimum_history_or_ratio_staleness_rule():
    panel = pd.DataFrame({"a": [30.0, np.nan, np.nan, np.nan],
                          "b": [20.0, 30.0, 40.0, 50.0],
                          "c": [40.0, 60.0, 80.0, 100.0]},
                         index=pd.date_range("2023-01-01", periods=4, freq="MS"))
    result = predict(panel, origin="2023-04", horizon=1, ids=["a"])
    row = result.forecasts.iloc[0]
    assert row.status == "native"
    assert row.last_ratio_period == pd.Timestamp("2023-01-01")
    assert row.n_ratio_observations == 1 and row.ratio_staleness_months == 3
    assert row.ratio_hat == 1.0


def test_extreme_finite_ratio_and_zero_expense_are_not_clipped(toy_panel):
    toy_panel.loc["2023-12-01", "a"] = 1e6
    result = predict(toy_panel, ids=["a"])
    assert result.forecasts.ratio_hat.iloc[0] == 1e6 / 444.0
    assert result.forecasts.y_pred.iloc[0] == result.national_forecast["N_hat"] * (1e6 / 444.0)
    toy_panel.loc["2023-12-01", "a"] = 0.0
    zero = predict(toy_panel, ids=["a"])
    assert zero.forecasts.status.iloc[0] == "native"
    assert zero.forecasts.ratio_hat.iloc[0] == zero.forecasts.y_pred.iloc[0] == 0.0


def test_restoration_overflow_fails_entire_batch(toy_panel, monkeypatch):
    toy_panel.loc["2023-12-01", "a"] = 1e308
    expected = national_forecast(build_national(toy_panel), "2023-12", 3, BASELINE)
    monkeypatch.setattr(persistence, "national_forecast", lambda *args, **kwargs: dict(expected, N_hat=1e308))
    result = predict(toy_panel)
    assert result.forecasts.status.eq("failed").all()
    assert result.forecasts.y_pred.isna().all()
    assert result.forecasts.failure_stage.eq("restore").all()


def test_deterministic_order_and_no_input_mutation(toy_panel):
    original = toy_panel.copy(deep=True)
    national = build_national(toy_panel)
    original_national = national.copy(deep=True)
    model = NationalLocalPersistence(national)
    first = model.predict(toy_panel, "2023-12", 3, ["c", "a"], BASELINE)
    second = model.predict(toy_panel, "2023-12", 3, ["c", "a"], BASELINE)
    pd.testing.assert_frame_equal(first.forecasts, second.forecasts, check_exact=True)
    assert first.national_forecast == second.national_forecast
    assert first.forecasts.municipality_id.tolist() == ["c", "a"]
    pd.testing.assert_frame_equal(toy_panel, original, check_exact=True)
    pd.testing.assert_frame_equal(national, original_national, check_exact=True)


@pytest.mark.parametrize("horizon", [True, 0, -1, 1.0])
def test_invalid_horizon_rejected(toy_panel, horizon):
    with pytest.raises(ValueError):
        predict(toy_panel, horizon=horizon)


@pytest.mark.parametrize("lag", [True, -1, 1.0])
def test_invalid_publication_lag_rejected(toy_panel, lag):
    with pytest.raises(ValueError):
        NationalLocalPersistence(build_national(toy_panel), lag)


def test_duplicate_and_unknown_forecast_keys_are_rejected(toy_panel):
    with pytest.raises(ValueError, match="unique"):
        predict(toy_panel, ids=["a", "a"])
    with pytest.raises(ValueError, match="absent"):
        predict(toy_panel, ids=["unknown"])


def test_negative_historical_expenses_rejected_without_ratio_clipping(toy_panel):
    toy_panel.loc["2023-12-01", "a"] = -1.0
    with pytest.raises(ValueError, match="nonnegative"):
        predict(toy_panel)
