"""Синтетические проверки E05c; сеть и реальные экспериментальные fit запрещены."""
import numpy as np
import pandas as pd
import pytest

from sberforecast import macro_forecast_model
from sberforecast.calendar_models import training_signature
from sberforecast.data import get_prefix
from sberforecast.direct_model import CatBoostDirect, TemporalIntegrityError
from sberforecast.direct_training import build_direct_training
from sberforecast.macro_forecast_model import (
    CONSUMPTION, INFLATION, MacroCatBoostDirect, ordered_training_key_signature,
)


class RecordingRegressor:
    instances = []

    def __init__(self, **parameters):
        self.parameters = parameters
        self.instances.append(self)

    def fit(self, X, y):
        self.training_X = X.copy(deep=True)
        self.training_y = y.copy()
        self.feature_names_ = X.columns.tolist()
        self.feature_importances_ = np.zeros(len(X.columns))

    def predict(self, X):
        self.prediction_X = X.copy(deep=True)
        return np.zeros(len(X))

    def get_all_params(self):
        return dict(self.parameters)

    def get_cat_feature_indices(self):
        return []


@pytest.fixture
def recorder(monkeypatch):
    RecordingRegressor.instances = []
    monkeypatch.setattr(macro_forecast_model, "CatBoostRegressor", RecordingRegressor)
    return RecordingRegressor


def publication(indicator, year, issued="2023-01-15", value=5.0, vintage=None):
    inflation = indicator == INFLATION
    return {
        "region_id": "RU", "geographic_level": "national", "indicator": indicator,
        "reference_period": str(year), "target_period": str(year), "value": value,
        "unit": "percent_growth", "published_at": issued, "available_at": issued,
        "forecast_issue_date": issued, "vintage_id": vintage or "synthetic-" + issued,
        "availability_status": "A", "source_url": "https://cbr.ru/synthetic/" + issued,
        "is_forecast": True, "forecast_producer": "survey_participants" if inflation else "Bank of Russia",
        "aggregation_basis": "december_to_december" if inflation else "annual_real_volume_growth",
        "forecast_central": value,
        "forecast_low": np.nan if inflation else value - 0.5,
        "forecast_high": np.nan if inflation else value + 0.5,
        "central_method": "quoted_rounded_respondent_median" if inflation else "interval_midpoint_not_official_central",
        "forecast_interval_kind": "not_reported" if inflation else "official_forecast_range",
    }


@pytest.fixture
def macro():
    return pd.DataFrame([publication(indicator, year)
                         for indicator in (INFLATION, CONSUMPTION) for year in (2023, 2024)])


def model(cfg, macro, variant, lag=0):
    return MacroCatBoostDirect(cfg["models"]["catboost"], cfg["seed"], variant,
                              macro, release_lag_months=lag)


def pairs(panel, origin="2023-12", horizon=1, lag=0):
    return build_direct_training(get_prefix(panel, pd.Period(origin, "M"), lag), origin,
                                 horizon, release_lag_months=lag, mode="legacy")


@pytest.mark.parametrize("variant, count", [("M0", 19), ("M1", 24), ("M2", 29)])
def test_original_features_keys_labels_types_order_and_global_panel_are_exact(
        panel, cfg, macro, recorder, variant, count):
    original_panel = panel.copy(deep=True)
    original_macro = macro.copy(deep=True)
    X, delta, trace = pairs(panel, horizon=3)
    result = model(cfg, macro, variant).predict(panel, pd.Period("2023-12", "M"), 3,
                                              ["1"], cfg["models"])
    estimator = recorder.instances[-1]
    assert len(estimator.training_X.columns) == count
    pd.testing.assert_frame_equal(estimator.training_X[X.columns], X)
    np.testing.assert_array_equal(estimator.training_y, delta)
    for key, value in training_signature(trace, X, delta).items():
        assert result.training[key] == value
    assert result.training["training_ordered_key_sha256"] == ordered_training_key_signature(trace)
    assert result.training["n_training_rows"] == len(trace)
    assert result.training["n_training_municipalities"] == 2
    assert result.training["cat_features"] == []
    assert estimator.parameters == dict(cfg["models"]["catboost"], random_seed=cfg["seed"],
                                       allow_writing_files=False, verbose=False)
    assert result.forecasts.status.eq("native").all()
    pd.testing.assert_frame_equal(panel, original_panel)
    pd.testing.assert_frame_equal(macro, original_macro)
    assert estimator.prediction_X.month.iloc[0] == 3


def test_ordered_signature_detects_reordered_pairs(panel):
    X, delta, trace = pairs(panel)
    assert training_signature(trace, X, delta)["training_key_sha256"] == training_signature(
        trace.iloc[::-1], X, delta)["training_key_sha256"]
    assert ordered_training_key_signature(trace) != ordered_training_key_signature(trace.iloc[::-1])


@pytest.mark.parametrize("variant", ["M1", "M2"])
def test_missing_macro_values_never_remove_training_or_forecast_rows(
        panel, cfg, macro, recorder, variant):
    empty_macro = macro.iloc[:0].copy()
    X, delta, trace = pairs(panel)
    result = model(cfg, empty_macro, variant).predict(panel, pd.Period("2023-12", "M"), 1,
                                                    ["1", "2"], cfg["models"])
    estimator = recorder.instances[-1]
    assert len(estimator.training_X) == len(trace) == len(delta)
    pd.testing.assert_frame_equal(estimator.training_X[X.columns], X)
    assert estimator.training_X[INFLATION].isna().all()
    assert estimator.training_X[INFLATION + "_missing"].eq(1.0).all()
    assert len(result.forecasts) == 2 and result.forecasts.status.eq("native").all()
    assert not result.forecasts.macro_features_available.any()
    for diagnostic in result.macro_diagnostics:
        assert diagnostic["n_available"] == 0 and diagnostic["all_missing"]
        assert diagnostic["n_unique_publications"] == 0
        if diagnostic["indicator"] == INFLATION:
            assert diagnostic["column_states"][INFLATION + "_missing"]["constant_finite"]


def test_future_expenses_and_publications_do_not_change_training_on_O(
        panel, cfg, macro, recorder):
    O = pd.Period("2024-01", "M")
    baseline = model(cfg, macro, "M2", lag=1).predict(panel, O, 3, ["1", "2"], cfg["models"])
    old_training = recorder.instances[-1].training_X.copy()
    old_y = recorder.instances[-1].training_y.copy()
    old_future = recorder.instances[-1].prediction_X.copy()
    changed = panel.copy()
    changed.loc["2024-01-01":] *= 1000
    augmented_macro = pd.concat([macro, pd.DataFrame([
        publication(indicator, 2024, "2024-02-01", 999.0)
        for indicator in (INFLATION, CONSUMPTION)])], ignore_index=True)
    after = model(cfg, augmented_macro, "M2", lag=1).predict(changed, O, 3,
                                                          ["1", "2"], cfg["models"])
    pd.testing.assert_frame_equal(recorder.instances[-1].training_X, old_training)
    pd.testing.assert_frame_equal(recorder.instances[-1].prediction_X, old_future)
    np.testing.assert_array_equal(recorder.instances[-1].training_y, old_y)
    assert baseline.training["training_augmented_X_sha256"] == after.training["training_augmented_X_sha256"]
    np.testing.assert_array_equal(baseline.forecasts.y_pred, after.forecasts.y_pred)


def test_historical_example_uses_own_r_future_y_may_change(panel, cfg, macro, recorder):
    O = pd.Period("2023-12", "M")
    model(cfg, macro, "M1").predict(panel, O, 3, ["1"], cfg["models"])
    original_X = recorder.instances[-1].training_X.copy()
    original_y = recorder.instances[-1].training_y.copy()
    _, _, trace = pairs(panel, horizon=3)
    selected = trace.historical_origin.eq(pd.Timestamp("2023-01-31"))
    changed = panel.copy()
    changed.loc["2023-02-01":] *= 2
    late_macro = pd.concat([macro, pd.DataFrame([
        publication(INFLATION, 2023, "2023-02-01", 77.0)])], ignore_index=True)
    model(cfg, late_macro, "M1").predict(changed, O, 3, ["1"], cfg["models"])
    pd.testing.assert_frame_equal(recorder.instances[-1].training_X.loc[selected], original_X.loc[selected])
    assert not np.array_equal(recorder.instances[-1].training_y[selected], original_y[selected])
    # The later publication really does affect later examples, never the January ones.
    assert recorder.instances[-1].training_X.loc[~selected, INFLATION].eq(77.0).any()


def test_macro_as_of_is_r_not_expense_cutoff_and_target_is_origin_plus_h(
        panel, cfg, macro, recorder):
    macro = pd.concat([macro, pd.DataFrame([
        publication(INFLATION, 2024, "2024-01-15", 8.0)])], ignore_index=True)
    result = model(cfg, macro, "M1", lag=1).predict(panel, pd.Period("2024-01", "M"), 3,
                                                 ["1"], cfg["models"])
    future = recorder.instances[-1].prediction_X
    assert future.month.iloc[0] == 4 and future.lag_1.iloc[0] == 111
    assert future[INFLATION].iloc[0] == 8.0  # Jan publication available at O despite expense cutoff Dec.
    provenance = result.training_provenance
    assert (provenance.publication_date <= provenance.historical_origin).all()
    assert provenance.target_available_at.max() <= pd.Timestamp("2024-01-31")
    assert provenance.target_period.dt.to_period("M").equals(
        provenance.historical_origin.dt.to_period("M") + 3)


@pytest.mark.parametrize("variant", ["M0", "M1", "M2"])
def test_first_annual_case_stays_explicit_seasonal_fallback_even_with_macro(
        panel, cfg, macro, recorder, variant):
    result = model(cfg, macro, variant).predict(panel, pd.Period("2023-12", "M"), 12,
                                              ["1", "2"], cfg["models"])
    assert not recorder.instances and not result.training["fit_called"]
    assert result.training["n_training_rows"] == 0
    assert result.forecasts.status.eq("fallback_no_training_pairs").all()
    assert result.forecasts.effective_model.eq("SeasonalNaive").all()
    assert result.forecasts.reason.eq("no_training_pairs").all()
    np.testing.assert_array_equal(result.forecasts.y_pred, [111.0, 222.0])
    if variant != "M0":
        assert result.forecasts.macro_features_available.all()


@pytest.mark.parametrize("phase", ["fit", "predict"])
def test_fit_predict_errors_are_failed_without_hidden_reserve(
        panel, cfg, macro, recorder, monkeypatch, phase):
    def fail(*args):
        raise RuntimeError("synthetic " + phase + " failure")
    monkeypatch.setattr(recorder, phase, fail)
    monkeypatch.setattr(macro_forecast_model, "baseline_predict", lambda *args: pytest.fail("hidden fallback"))
    result = model(cfg, macro, "M2").predict(panel, pd.Period("2023-12", "M"), 1,
                                           ["1"], cfg["models"])
    assert result.forecasts.status.eq("failed").all() and result.forecasts.y_pred.isna().all()
    assert result.errors[0]["stage"] == phase
    assert result.training["fit_called"]
    assert result.training["fit_succeeded"] == (phase == "predict")


def test_bad_target_availability_stops_before_fit(panel, cfg, macro, recorder, monkeypatch):
    original = macro_forecast_model.build_direct_training
    def invalid(*args, **kwargs):
        X, y, trace = original(*args, **kwargs)
        trace.target_available_at = pd.Timestamp("2024-12-31")
        return X, y, trace
    monkeypatch.setattr(macro_forecast_model, "build_direct_training", invalid)
    with pytest.raises(TemporalIntegrityError):
        model(cfg, macro, "M1").predict(panel, pd.Period("2023-12", "M"), 1,
                                       ["1"], cfg["models"])
    assert not recorder.instances


def test_nominal_anchor_restoration_and_negative_clipping(panel, cfg, macro, recorder, monkeypatch):
    monkeypatch.setattr(recorder, "predict", lambda self, X: np.array([-1000.0, 7.0]))
    result = model(cfg, macro, "M2").predict(panel, pd.Period("2023-12", "M"), 3,
                                           ["1", "2"], cfg["models"])
    np.testing.assert_array_equal(result.forecasts.y_pred, [0.0, 229.0])


def test_M0_actual_synthetic_predictions_equal_original_catboost(panel, cfg, macro):
    origin = pd.Period("2023-12", "M")
    previous = CatBoostDirect(cfg["models"]["catboost"], cfg["seed"]).predict(
        panel, origin, 3, ["1", "2"], cfg["models"])
    identity = model(cfg, macro, "M0").predict(panel, origin, 3, ["1", "2"], cfg["models"])
    assert not previous.errors and not identity.errors
    np.testing.assert_array_equal(identity.forecasts.y_pred, previous.forecasts.y_pred)


def test_actual_small_catboost_fits_with_all_missing_macro_without_categories(panel, cfg, macro):
    result = model(cfg, macro.iloc[:0], "M2").predict(panel, pd.Period("2023-12", "M"), 1,
                                                  ["1", "2"], cfg["models"])
    assert not result.errors and result.forecasts.status.eq("native").all()
    assert result.training["fit_actual_cat_feature_indices"] == []
    assert len(result.training["fit_actual_feature_names"]) == 29
    assert result.training["fit_actual_parameters"]["iterations"] == 5
    assert not result.forecasts.macro_features_available.any()


def test_diagnostics_distinguish_panel_rows_dates_publications_and_constants(
        panel, cfg, macro, recorder):
    result = model(cfg, macro, "M2").predict(panel, pd.Period("2023-12", "M"), 1,
                                           ["1", "2"], cfg["models"])
    training = [row for row in result.macro_diagnostics if row["scope"] == "training"]
    assert len(training) == 2
    for row in training:
        assert row["n_rows"] == 22 and row["n_historical_origins"] == 11
        assert row["n_unique_publications"] == 1 and row["n_unique_finite_values"] == 1
        assert row["available_fraction"] == 1.0 and row["constant_finite_value"]
        assert row["n_dates_with_within_date_feature_variation"] == 0
    inflation = next(row for row in training if row["indicator"] == INFLATION)
    assert inflation["column_states"][INFLATION + "_range_width_pp"]["all_missing"]


@pytest.mark.parametrize("variant, mode, extra", [
    ("BAD", "legacy", {}), ("M1", "strict12", {}), ("M2", "legacy", {"cat_features": ["month"]}),
])
def test_unfixed_variant_mode_or_calendar_parameters_rejected(cfg, macro, variant, mode, extra):
    with pytest.raises(ValueError):
        MacroCatBoostDirect(dict(cfg["models"]["catboost"], **extra), cfg["seed"], variant,
                           macro, training_mode=mode)
