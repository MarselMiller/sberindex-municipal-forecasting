"""Синтетические проверки E05b: меняется только представление месяца цели."""
import numpy as np
import pandas as pd
import pytest

from sberforecast import calendar_models
from sberforecast.calendar_models import (
    CalendarCatBoostDirect, training_signature, transform_month_features,
)
from sberforecast.data import get_prefix
from sberforecast.direct_model import CatBoostDirect, TemporalIntegrityError
from sberforecast.direct_training import build_direct_training, direct_features


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
        return [self.feature_names_.index(name) for name in self.parameters.get("cat_features", [])]


@pytest.fixture
def recorder(monkeypatch):
    RecordingRegressor.instances = []
    monkeypatch.setattr(calendar_models, "CatBoostRegressor", RecordingRegressor)
    return RecordingRegressor


def model(cfg, encoding, lag=0):
    return CalendarCatBoostDirect(cfg["models"]["catboost"], cfg["seed"], encoding, lag)


def training(panel, origin="2023-12", horizon=1):
    return build_direct_training(panel, origin, horizon, release_lag_months=0, mode="legacy")


def test_k0_identity_preserves_features_values_types_index_and_input(panel):
    X, _, _ = training(panel)
    original = X.copy(deep=True)
    transformed = transform_month_features(X, "K0")
    pd.testing.assert_frame_equal(transformed, original)
    transformed.iloc[0, 0] = -123.0
    pd.testing.assert_frame_equal(X, original)


@pytest.mark.parametrize("encoding", ["K1", "K2"])
def test_categorical_columns_types_and_other_features_unchanged(panel, encoding):
    X, _, _ = training(panel)
    original = X.copy(deep=True)
    transformed = transform_month_features(X, encoding)
    removed = ["month", "month_sin", "month_cos"] if encoding == "K1" else ["month"]
    expected = set(X.columns) - set(removed) | {"month_category"}
    assert set(transformed.columns) == expected
    assert transformed.month_category.dtype == object
    assert transformed.month_category.map(type).eq(str).all()
    assert transformed.month_category.tolist() == [f"{int(month):02d}" for month in X.month]
    pd.testing.assert_frame_equal(
        transformed.drop(columns="month_category"), X.drop(columns=removed))
    assert transformed.lag_12.isna().equals(X.lag_12.isna())
    pd.testing.assert_frame_equal(X, original)


@pytest.mark.parametrize("encoding", ["K1", "K2"])
def test_empty_category_still_has_string_compatible_type(panel, encoding):
    X, _, _ = training(panel, horizon=12)
    transformed = transform_month_features(X, encoding)
    assert transformed.empty and transformed.month_category.dtype == object


@pytest.mark.parametrize("bad_month", [0, 13, 1.5, np.nan, np.inf])
def test_invalid_target_month_is_rejected(panel, bad_month):
    X, _, _ = training(panel)
    X.loc[0, "month"] = bad_month
    with pytest.raises(ValueError, match="1..12"):
        transform_month_features(X, "K1")


def test_unknown_encoding_is_rejected(panel, cfg):
    X, _, _ = training(panel)
    with pytest.raises(ValueError):
        transform_month_features(X, "unknown")
    with pytest.raises(ValueError):
        model(cfg, "unknown")


@pytest.mark.parametrize("encoding", ["K1", "K2"])
def test_target_month_not_origin_month_and_december_january(panel, cfg, recorder, encoding):
    model(cfg, encoding).predict(panel, pd.Period("2023-12", "M"), 1, ["1"], cfg["models"])
    january = recorder.instances[-1].prediction_X
    assert january.month_category.tolist() == ["01"]
    assert january.days_in_month.iloc[0] == 31
    assert january.time_index.iloc[0] == 12
    model(cfg, encoding).predict(panel, pd.Period("2024-01", "M"), 3, ["1"], cfg["models"])
    assert recorder.instances[-1].prediction_X.month_category.tolist() == ["04"]


def test_target_month_is_correct_with_publication_lag(panel, cfg, recorder):
    model(cfg, "K1", lag=1).predict(panel, pd.Period("2024-01", "M"), 3, ["1"], cfg["models"])
    X = recorder.instances[-1].prediction_X
    assert X.month_category.tolist() == ["04"]
    assert X.lag_1.iloc[0] == 111.0  # cutoff декабрь, цель апрель.


@pytest.mark.parametrize("encoding", ["K1", "K2"])
def test_actual_catboost_category_and_one_hot_parameter_and_unseen_month(panel, cfg, encoding):
    X, _, trace = training(panel)
    assert not trace.target_period.dt.month.eq(1).any()
    changed = panel.copy(deep=True)
    changed.loc["2024-01-01":] = 999999.0
    a = model(cfg, encoding).predict(panel, pd.Period("2023-12", "M"), 1, ["1", "2"], cfg["models"])
    b = model(cfg, encoding).predict(changed, pd.Period("2023-12", "M"), 1, ["1", "2"], cfg["models"])
    assert a.forecasts.status.eq("native").all() and not a.errors
    assert a.training["fit_actual_parameters"]["one_hot_max_size"] == 12
    index = a.training["fit_actual_feature_names"].index("month_category")
    assert a.training["fit_actual_cat_feature_indices"] == [index]
    assert a.training["feature_dtypes"]["month_category"] == "object"
    np.testing.assert_array_equal(a.forecasts.y_pred, b.forecasts.y_pred)
    assert a.training["training_raw_X_sha256"] == b.training["training_raw_X_sha256"]


def test_k0_actual_predictions_equal_unchanged_e02b_adapter(panel, cfg):
    old = CatBoostDirect(cfg["models"]["catboost"], cfg["seed"])
    original = panel.copy(deep=True)
    a = old.predict(panel, pd.Period("2023-12", "M"), 3, ["1", "2"], cfg["models"])
    b = model(cfg, "K0").predict(panel, pd.Period("2023-12", "M"), 3, ["1", "2"], cfg["models"])
    assert b.forecasts.status.eq("native").all()
    np.testing.assert_allclose(a.forecasts.y_pred, b.forecasts.y_pred, atol=0, rtol=0)
    pd.testing.assert_frame_equal(panel, original)


def test_all_encodings_keep_identical_global_pairs_targets_and_raw_features(panel, cfg, recorder):
    original = panel.copy(deep=True)
    X, delta, trace = training(panel, horizon=3)
    signature = training_signature(trace, X, delta)
    results = [model(cfg, encoding).predict(panel, pd.Period("2023-12", "M"), 3,
                                          ["1"], cfg["models"]) for encoding in ("K0", "K1", "K2")]
    for result, estimator in zip(results, recorder.instances):
        assert result.training["n_training_municipalities"] == 2
        assert result.training["n_training_rows"] == len(trace)
        for key, value in signature.items():
            assert result.training[key] == value
        np.testing.assert_array_equal(estimator.training_y, delta)
        assert result.forecasts.status.eq("native").all()
        # Only category declarations differ from fixed E02b CatBoost parameters.
        params = dict(estimator.parameters)
        if result.training["month_encoding"] != "K0":
            assert params.pop("cat_features") == ["month_category"]
            assert params.pop("one_hot_max_size") == 12
        assert params == dict(cfg["models"]["catboost"], random_seed=cfg["seed"],
                              allow_writing_files=False, verbose=False)
    pd.testing.assert_frame_equal(panel, original)


def test_signature_covers_raw_X_delta_keys_and_target_availability(panel):
    X, delta, trace = training(panel)
    original = training_signature(trace, X, delta)
    changed_X = X.copy()
    changed_X.loc[0, "days_in_month"] += 1
    assert training_signature(trace, changed_X, delta)["training_raw_X_sha256"] != original["training_raw_X_sha256"]
    changed_delta = delta.copy()
    changed_delta[0] += 1
    assert training_signature(trace, X, changed_delta)["training_delta_sha256"] != original["training_delta_sha256"]
    changed_trace = trace.copy()
    changed_trace.loc[0, "target_available_at"] += pd.offsets.MonthEnd(1)
    assert training_signature(changed_trace, X, delta)["training_key_sha256"] != original["training_key_sha256"]
    reordered_trace = trace.iloc[::-1]
    assert training_signature(reordered_trace, X, delta)["training_key_sha256"] == original["training_key_sha256"]


@pytest.mark.parametrize("encoding", ["K0", "K1", "K2"])
def test_first_annual_case_is_explicit_seasonal_reserve_without_fit(panel, cfg, recorder, encoding):
    result = model(cfg, encoding).predict(panel, pd.Period("2023-12", "M"), 12, ["1", "2"], cfg["models"])
    assert not recorder.instances and not result.training["fit_called"]
    assert result.training["n_training_rows"] == 0
    assert result.forecasts.status.eq("fallback_no_training_pairs").all()
    assert result.forecasts.effective_model.eq("SeasonalNaive").all()
    assert result.forecasts.reason.eq("no_training_pairs").all()
    np.testing.assert_array_equal(result.forecasts.y_pred, [111.0, 222.0])


def test_fit_error_never_becomes_no_pairs_fallback(panel, cfg, recorder, monkeypatch):
    def fail(self, X, y):
        raise RuntimeError("synthetic failure")
    monkeypatch.setattr(recorder, "fit", fail)
    monkeypatch.setattr(calendar_models, "baseline_predict", lambda *args: pytest.fail("hidden fallback"))
    result = model(cfg, "K1").predict(panel, pd.Period("2023-12", "M"), 1, ["1"], cfg["models"])
    assert result.training["fit_called"] and not result.training["fit_succeeded"]
    assert result.forecasts.status.eq("failed").all() and result.forecasts.y_pred.isna().all()
    assert result.errors[0]["stage"] == "fit"


def test_temporal_violation_stops_before_fit(panel, cfg, recorder, monkeypatch):
    original = calendar_models.build_direct_training
    def invalid_trace(*args, **kwargs):
        X, delta, trace = original(*args, **kwargs)
        trace["target_available_at"] = pd.Timestamp("2024-12-31")
        return X, delta, trace
    monkeypatch.setattr(calendar_models, "build_direct_training", invalid_trace)
    with pytest.raises(TemporalIntegrityError):
        model(cfg, "K2").predict(panel, pd.Period("2023-12", "M"), 1, ["1"], cfg["models"])
    assert not recorder.instances


def test_nominal_anchor_restoration_and_nonnegative_prediction(panel, cfg, recorder, monkeypatch):
    monkeypatch.setattr(recorder, "predict", lambda self, X: np.array([-1000.0, 7.0]))
    result = model(cfg, "K2").predict(panel, pd.Period("2023-12", "M"), 3, ["1", "2"], cfg["models"])
    np.testing.assert_array_equal(result.forecasts.y_pred, [0.0, 229.0])
