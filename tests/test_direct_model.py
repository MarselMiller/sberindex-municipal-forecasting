"""Синтетические проверки модели и обработки резерва/ошибок."""
import numpy as np
import pandas as pd
import pytest

from sberforecast import direct_model
from sberforecast.data import get_prefix
from sberforecast.direct_model import CatBoostDirect, TemporalIntegrityError
from sberforecast.models import catboost_predict


class RecordingRegressor:
    instances = []

    def __init__(self, **parameters):
        self.parameters = parameters
        self.instances.append(self)

    def fit(self, X, y):
        self.training_X = X.copy()
        self.training_y = y.copy()
        self.feature_importances_ = np.zeros(len(X.columns))

    def predict(self, X):
        self.prediction_X = X.copy()
        return np.zeros(len(X))


@pytest.fixture
def recorder(monkeypatch):
    RecordingRegressor.instances = []
    monkeypatch.setattr(direct_model, "CatBoostRegressor", RecordingRegressor)
    return RecordingRegressor


def model(cfg, lag=0):
    return CatBoostDirect(cfg["models"]["catboost"], cfg["seed"], lag)


def test_horizons_have_separate_fits_and_do_not_modify_history(panel, cfg, recorder):
    original = panel.copy(deep=True)
    predictor = model(cfg)
    one = predictor.predict(panel, pd.Period("2023-12", "M"), 1, ["1", "2"], cfg["models"])
    three = predictor.predict(panel, pd.Period("2023-12", "M"), 3, ["1", "2"], cfg["models"])
    again = predictor.predict(panel, pd.Period("2023-12", "M"), 1, ["1", "2"], cfg["models"])
    assert len(recorder.instances) == 3
    np.testing.assert_array_equal(one.forecasts.y_pred, again.forecasts.y_pred)
    pd.testing.assert_frame_equal(panel, original)
    assert recorder.instances[0].prediction_X.month.eq(1).all()
    assert recorder.instances[1].prediction_X.month.eq(3).all()
    assert recorder.instances[1].prediction_X.days_in_month.eq(31).all()
    np.testing.assert_array_equal(recorder.instances[1].prediction_X.lag_1, [111, 222])
    assert three.forecasts.status.eq("native").all()


def test_target_calendar_with_publication_lag(panel, cfg, recorder):
    model(cfg, lag=1).predict(panel, pd.Period("2024-01", "M"), 3, ["1"], cfg["models"])
    X = recorder.instances[0].prediction_X
    assert X.month.iloc[0] == 4 and X.days_in_month.iloc[0] == 30
    assert X.lag_1.iloc[0] == 111  # cutoff=декабрь, цель=апрель.


def test_scale_restoration_and_nonnegative_postprocessing(panel, cfg, recorder, monkeypatch):
    monkeypatch.setattr(recorder, "predict", lambda self, X: np.array([-1000., 7.]))
    result = model(cfg).predict(panel, pd.Period("2023-12", "M"), 3, ["1", "2"], cfg["models"])
    np.testing.assert_array_equal(result.forecasts.y_pred, [0., 229.])
    assert result.forecasts.status.eq("native").all()
    parameters = recorder.instances[0].parameters
    assert parameters == dict(cfg["models"]["catboost"], random_seed=cfg["seed"],
                              allow_writing_files=False, verbose=False)


def test_global_training_is_not_restricted_to_forecast_sample(panel, cfg, recorder):
    result = model(cfg).predict(panel, pd.Period("2023-12", "M"), 1, ["1"], cfg["models"])
    assert len(recorder.instances[0].training_X) == 22
    assert result.training["n_training_municipalities"] == 2
    assert result.training["n_forecast_series"] == 1


def test_no_pairs_uses_explicit_seasonal_reserve_without_constructing_model(panel, cfg, recorder):
    result = model(cfg).predict(panel, pd.Period("2023-12", "M"), 12, ["1", "2"], cfg["models"])
    assert not recorder.instances
    assert result.training["n_training_rows"] == 0
    assert not result.training["fit_called"]
    assert result.forecasts.status.eq("fallback_no_training_pairs").all()
    assert result.forecasts.effective_model.eq("SeasonalNaive").all()
    assert result.forecasts.reason.eq("no_training_pairs").all()
    np.testing.assert_array_equal(result.forecasts.y_pred, [111, 222])
    assert not result.errors


def test_fit_error_is_not_no_pairs_and_never_uses_reserve(panel, cfg, recorder, monkeypatch):
    def fail_fit(self, X, y):
        raise RuntimeError("synthetic fit failure")
    monkeypatch.setattr(recorder, "fit", fail_fit)
    monkeypatch.setattr(direct_model, "baseline_predict", lambda *args: pytest.fail("hidden fallback"))
    result = model(cfg).predict(panel, pd.Period("2023-12", "M"), 1, ["1", "2"], cfg["models"])
    assert result.training["n_training_rows"] == 22 and result.training["fit_called"]
    assert not result.training["fit_succeeded"]
    assert result.forecasts.status.eq("failed").all()
    assert result.forecasts.y_pred.isna().all()
    assert result.errors[0]["stage"] == "fit"
    assert "synthetic fit failure" in result.errors[0]["error"]


@pytest.mark.parametrize("failure", ["exception", "nan", "positive_infinity"])
def test_predict_failures_are_logged_without_reserve(panel, cfg, recorder, monkeypatch, failure):
    def predict(self, X):
        if failure == "exception":
            raise RuntimeError("synthetic predict failure")
        return np.full(len(X), np.nan if failure == "nan" else np.inf)
    monkeypatch.setattr(recorder, "predict", predict)
    monkeypatch.setattr(direct_model, "baseline_predict", lambda *args: pytest.fail("hidden fallback"))
    result = model(cfg).predict(panel, pd.Period("2023-12", "M"), 1, ["1", "2"], cfg["models"])
    assert result.training["fit_succeeded"]
    assert result.forecasts.status.eq("failed").all()
    assert result.forecasts.y_pred.isna().all()
    assert result.errors[0]["stage"] == "predict"


def test_temporal_integrity_violation_stops_before_fit(panel, cfg, recorder, monkeypatch):
    original = direct_model.build_direct_training
    def bad_trace(*args, **kwargs):
        X, y, meta = original(*args, **kwargs)
        meta["target_available_at"] = pd.Timestamp("2024-12-31")
        return X, y, meta
    monkeypatch.setattr(direct_model, "build_direct_training", bad_trace)
    with pytest.raises(TemporalIntegrityError):
        model(cfg).predict(panel, pd.Period("2023-12", "M"), 1, ["1"], cfg["models"])
    assert not recorder.instances


def test_real_h1_matches_recursive_on_synthetic_panel(panel, cfg):
    origin = pd.Period("2023-12", "M")
    prefix = get_prefix(panel, origin, 0)
    recursive, _, _ = catboost_predict(prefix, prefix, 1, cfg["models"]["catboost"], cfg["seed"])
    result = model(cfg).predict(panel, origin, 1, ["1", "2"], cfg["models"])
    assert result.forecasts.status.eq("native").all()
    np.testing.assert_allclose(result.forecasts.y_pred, recursive[0], rtol=0, atol=1e-8)


def test_future_values_do_not_change_real_direct_forecast(panel, cfg):
    origin = pd.Period("2023-12", "M")
    changed = panel.copy()
    changed.loc["2024-01-01":] *= 1000
    a = model(cfg).predict(panel, origin, 3, ["1", "2"], cfg["models"])
    b = model(cfg).predict(changed, origin, 3, ["1", "2"], cfg["models"])
    assert a.forecasts.status.eq("native").all() and b.forecasts.status.eq("native").all()
    np.testing.assert_allclose(a.forecasts.y_pred, b.forecasts.y_pred, rtol=0, atol=0)
