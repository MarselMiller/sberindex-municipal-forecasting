"""E05d learner tests: synthetic panels only; ordinary checks need no network."""
import numpy as np
import pandas as pd
import pytest

from sberforecast import direct_model, national_local_model
from sberforecast.calendar_models import training_signature
from sberforecast.data import get_prefix
from sberforecast.direct_model import CatBoostDirect, TemporalIntegrityError
from sberforecast.direct_training import build_direct_training, direct_features
from sberforecast.national_local import build_local_training, build_national, national_forecast, ratio_panel
from sberforecast.national_local_model import NationalLocalDirect, ordered_training_key_signature


class RecordingRegressor:
    instances = []

    def __init__(self, **parameters):
        self.parameters = parameters
        self.instances.append(self)

    def fit(self, X, y):
        self.training_X = X.copy(deep=True)
        self.training_y = y.copy()
        self.feature_names_ = X.columns.tolist()
        self.feature_name_ = X.columns.tolist()
        self.feature_importances_ = np.zeros(len(X.columns))

    def predict(self, X):
        self.prediction_X = X.copy(deep=True)
        return np.zeros(len(X))

    def get_all_params(self):
        return dict(self.parameters)

    def get_params(self, deep=True):
        return dict(self.parameters)

    def get_cat_feature_indices(self):
        return []


@pytest.fixture
def recorder(monkeypatch):
    RecordingRegressor.instances = []
    monkeypatch.setattr(national_local_model, "CatBoostRegressor", RecordingRegressor)
    monkeypatch.setattr(national_local_model, "_lightgbm_regressor", RecordingRegressor)
    return RecordingRegressor


def lgb_parameters(seed=42):
    return {"objective": "regression_l1", "n_estimators": 5, "learning_rate": .05,
            "num_leaves": 31, "random_state": seed, "n_jobs": 2,
            "deterministic": True, "force_col_wise": True}


def model(panel, cfg, variant, lag=0, national=None):
    parameters = cfg["models"]["catboost"] if variant == "CN" else lgb_parameters(cfg["seed"])
    national = build_national(panel, lag) if national is None else national
    return NationalLocalDirect(parameters, cfg["seed"], variant, national,
                               release_lag_months=lag)


def old_pairs(panel, origin="2023-12", horizon=1, lag=0):
    return build_direct_training(get_prefix(panel, pd.Period(origin, "M"), lag), origin, horizon,
                                 release_lag_months=lag, mode="legacy")


@pytest.mark.parametrize("horizon", [1, 3, 6])
def test_L0_gets_exact_original_training_keys_features_NaNs_labels_and_anchor(
        panel, cfg, recorder, horizon):
    panel = panel.copy()
    panel.loc["2023-03-01", "1"] = np.nan
    original_panel = panel.copy(deep=True)
    X, delta, trace = old_pairs(panel, horizon=horizon)
    original_future, anchor = direct_features(panel.loc[:"2023-12-01", ["1"]],
                                              pd.Timestamp("2023-12-01") + pd.DateOffset(months=horizon))
    result = model(panel, cfg, "L0").predict(panel, pd.Period("2023-12", "M"), horizon,
                                            ["1"], cfg["models"])
    estimator = recorder.instances[-1]
    pd.testing.assert_frame_equal(estimator.training_X, X)
    np.testing.assert_array_equal(estimator.training_y, delta)
    pd.testing.assert_frame_equal(estimator.prediction_X, original_future)
    np.testing.assert_array_equal(result.forecasts.anchor, anchor)
    assert result.training["n_training_municipalities"] == 2  # Global, not forecast-only.
    assert result.training["n_pairs_excluded_by_ratio_validity"] == 0
    assert result.training["training_ordered_key_sha256"] == ordered_training_key_signature(trace)
    for key, value in training_signature(trace, X, delta).items():
        assert result.training[key] == value
    assert estimator.parameters["use_missing"] is True
    assert estimator.parameters["zero_as_missing"] is False
    assert "territory_id" not in estimator.training_X
    assert len(estimator.training_X.columns) == 19
    assert all(dtype == np.dtype("float64") for dtype in estimator.training_X.dtypes)
    pd.testing.assert_frame_equal(panel, original_panel)


def test_C0_L0_training_keys_X_and_y_match_actual_old_model_pipeline(panel, cfg, recorder, monkeypatch):
    monkeypatch.setattr(direct_model, "CatBoostRegressor", RecordingRegressor)
    C0 = CatBoostDirect(cfg["models"]["catboost"], cfg["seed"]).predict(
        panel, pd.Period("2023-12", "M"), 3, ["1", "2"], cfg["models"])
    original = recorder.instances[-1]
    L0 = model(panel, cfg, "L0").predict(panel, pd.Period("2023-12", "M"), 3,
                                       ["1", "2"], cfg["models"])
    new = recorder.instances[-1]
    pd.testing.assert_frame_equal(new.training_X, original.training_X)
    np.testing.assert_array_equal(new.training_y, original.training_y)
    pd.testing.assert_frame_equal(new.prediction_X, original.prediction_X)
    np.testing.assert_array_equal(L0.forecasts.y_pred, C0.forecasts.y_pred)
    _, _, trace = old_pairs(panel, horizon=3)
    pd.testing.assert_frame_equal(L0.training_provenance[trace.columns], trace)


@pytest.mark.parametrize("horizon,lag", [(1, 0), (3, 0), (6, 1)])
def test_CN_LN_same_ratio_training_keys_X_y_forecast_keys_and_national_forecast(
        panel, cfg, recorder, horizon, lag):
    panel = panel.copy()
    panel.loc["2023-03-01", "1"] = np.nan
    national = build_national(panel, lag)
    origin = pd.Period("2024-01", "M")
    CN = model(panel, cfg, "CN", lag, national).predict(panel, origin, horizon, ["1", "2"], cfg["models"])
    cat = recorder.instances[-1]
    LN = model(panel, cfg, "LN", lag, national).predict(panel, origin, horizon, ["1", "2"], cfg["models"])
    light = recorder.instances[-1]
    pd.testing.assert_frame_equal(light.training_X, cat.training_X)
    np.testing.assert_array_equal(light.training_y, cat.training_y)
    pd.testing.assert_frame_equal(light.prediction_X, cat.prediction_X)
    for key in ("training_key_sha256", "training_ordered_key_sha256", "training_raw_X_sha256",
                "training_delta_sha256", "forecast_raw_X_sha256", "forecast_anchor_sha256"):
        assert CN.training[key] == LN.training[key]
    pd.testing.assert_series_equal(CN.forecasts.municipality_id, LN.forecasts.municipality_id)
    np.testing.assert_array_equal(CN.forecasts.y_pred, LN.forecasts.y_pred)
    assert CN.national_forecast == LN.national_forecast
    X, y, trace = build_local_training(panel, national, origin, horizon,
                                        release_lag_months=lag, mode="legacy")
    pd.testing.assert_frame_equal(cat.training_X, X)
    np.testing.assert_array_equal(cat.training_y, y)
    pd.testing.assert_frame_equal(CN.training_provenance[trace.columns], trace)
    assert (trace.target_national_available_at <= origin.to_timestamp(how="end")).all()
    assert (trace.anchor_national_available_at <= trace.historical_origin).all()
    np.testing.assert_allclose(y, trace.ratio_target_value - trace.ratio_anchor, rtol=0, atol=0)


@pytest.mark.parametrize("variant", ["L0", "CN", "LN"])
def test_changes_after_O_minus_L_leave_training_features_labels_and_forecast_unchanged(
        panel, cfg, recorder, variant):
    origin = pd.Period("2024-01", "M")
    before = model(panel, cfg, variant, lag=1).predict(panel, origin, 3, ["1", "2"], cfg["models"])
    old = recorder.instances[-1]
    changed = panel.copy()
    changed.loc["2024-01-01":] *= 1000
    after = model(changed, cfg, variant, lag=1).predict(changed, origin, 3,
                                                      ["1", "2"], cfg["models"])
    new = recorder.instances[-1]
    pd.testing.assert_frame_equal(new.training_X, old.training_X)
    pd.testing.assert_frame_equal(new.prediction_X, old.prediction_X)
    np.testing.assert_array_equal(new.training_y, old.training_y)
    np.testing.assert_array_equal(after.forecasts.y_pred, before.forecasts.y_pred)
    if variant != "L0":
        assert after.national_forecast["N_actual"] != before.national_forecast["N_actual"]
        assert after.national_forecast["N_hat"] == before.national_forecast["N_hat"]


@pytest.mark.parametrize("variant", ["CN", "LN"])
def test_future_actual_N_is_diagnostic_only_never_used_for_nominal_restoration(
        panel, cfg, recorder, variant):
    origin = pd.Period("2023-12", "M")
    national = build_national(panel)
    before = model(panel, cfg, variant, national=national).predict(panel, origin, 3,
                                                                  ["1", "2"], cfg["models"])
    changed = national.copy()
    changed.loc["2024-03-01", "N"] *= 999
    after = model(panel, cfg, variant, national=changed).predict(panel, origin, 3,
                                                               ["1", "2"], cfg["models"])
    assert after.national_forecast["N_actual"] != before.national_forecast["N_actual"]
    np.testing.assert_array_equal(after.forecasts.y_pred, before.forecasts.y_pred)
    np.testing.assert_array_equal(after.forecasts.y_pred,
                                  np.maximum(0, after.forecasts.national_hat * after.forecasts.ratio_hat))


def test_individual_historical_r_features_do_not_use_later_y_but_target_can_change(panel, cfg, recorder):
    panel = panel.copy()
    panel["3"] = np.arange(24, dtype=float) + 400
    before = model(panel, cfg, "CN").predict(panel, pd.Period("2023-12", "M"), 3,
                                            ["1", "2"], cfg["models"])
    old = recorder.instances[-1]
    selected = before.training_provenance.historical_origin.eq(pd.Timestamp("2023-01-31"))
    changed = panel.copy()
    changed.loc["2023-02-01":, "1"] *= 10
    model(changed, cfg, "CN").predict(changed, pd.Period("2023-12", "M"), 3,
                                      ["1", "2"], cfg["models"])
    new = recorder.instances[-1]
    pd.testing.assert_frame_equal(new.training_X.loc[selected], old.training_X.loc[selected])
    assert not np.array_equal(new.training_y[selected], old.training_y[selected])


@pytest.mark.parametrize("variant", ["L0", "CN", "LN"])
def test_annual_empty_pairs_explicit_original_expense_fallback_no_learner_constructed(
        panel, cfg, recorder, variant):
    result = model(panel, cfg, variant).predict(panel, pd.Period("2023-12", "M"), 12,
                                              ["1", "2"], cfg["models"])
    assert not recorder.instances
    assert result.training["n_training_rows"] == 0
    assert not result.training["fit_called"]
    assert result.forecasts.status.eq("fallback_no_training_pairs").all()
    assert result.forecasts.effective_model.eq("SeasonalNaive").all()
    assert result.forecasts.reason.eq("no_training_pairs").all()
    np.testing.assert_array_equal(result.forecasts.y_pred, [111.0, 222.0])
    assert result.forecasts.ratio_hat.isna().all()
    assert not result.errors


@pytest.mark.parametrize("variant", ["L0", "CN", "LN"])
@pytest.mark.parametrize("phase", ["fit", "predict"])
def test_learner_errors_are_failed_and_do_not_call_reserve(panel, cfg, recorder, monkeypatch, variant, phase):
    def failure(*args):
        raise RuntimeError("synthetic " + phase + " failure")
    monkeypatch.setattr(recorder, phase, failure)
    monkeypatch.setattr(national_local_model, "baseline_predict", lambda *args: pytest.fail("hidden fallback"))
    result = model(panel, cfg, variant).predict(panel, pd.Period("2023-12", "M"), 1,
                                              ["1", "2"], cfg["models"])
    assert result.forecasts.status.eq("failed").all()
    assert result.forecasts.y_pred.isna().all()
    assert result.errors[0]["stage"] == phase
    assert result.training["fit_succeeded"] == (phase == "predict")


@pytest.mark.parametrize("variant", ["L0", "CN", "LN"])
@pytest.mark.parametrize("bad_value", [np.nan, np.inf])
def test_nonfinite_learner_output_fails_without_reserve(panel, cfg, recorder, monkeypatch, variant, bad_value):
    monkeypatch.setattr(recorder, "predict", lambda self, X: np.full(len(X), bad_value))
    result = model(panel, cfg, variant).predict(panel, pd.Period("2023-12", "M"), 1,
                                              ["1", "2"], cfg["models"])
    assert result.forecasts.status.eq("failed").all()
    assert result.errors[0]["stage"] == "predict"


@pytest.mark.parametrize("N_hat", [np.nan, np.inf, 0.0, -1.0])
def test_invalid_national_forecast_fails_without_fit_or_constant(panel, cfg, recorder, monkeypatch, N_hat):
    old = national_forecast(build_national(panel), "2023-12", 1, cfg["models"])
    monkeypatch.setattr(national_local_model, "national_forecast", lambda *args, **kwargs: dict(old, N_hat=N_hat))
    result = model(panel, cfg, "CN").predict(panel, pd.Period("2023-12", "M"), 1,
                                            ["1", "2"], cfg["models"])
    assert not recorder.instances and not result.training["fit_called"]
    assert result.forecasts.status.eq("failed").all()
    assert result.errors[0]["stage"] == "national_forecast"


def test_bad_national_label_availability_stops_before_fit(panel, cfg, recorder, monkeypatch):
    original = national_local_model.build_local_training
    def bad(*args, **kwargs):
        X, delta, trace = original(*args, **kwargs)
        trace["target_national_available_at"] = pd.Timestamp("2024-12-31")
        return X, delta, trace
    monkeypatch.setattr(national_local_model, "build_local_training", bad)
    with pytest.raises(TemporalIntegrityError, match="знаменатель"):
        model(panel, cfg, "CN").predict(panel, pd.Period("2023-12", "M"), 1, ["1"], cfg["models"])
    assert not recorder.instances


def test_negative_local_ratio_clips_only_after_nominal_restore(panel, cfg, recorder, monkeypatch):
    monkeypatch.setattr(recorder, "predict", lambda self, X: np.array([-1000.0, 1.0]))
    result = model(panel, cfg, "CN").predict(panel, pd.Period("2023-12", "M"), 1,
                                            ["1", "2"], cfg["models"])
    assert result.forecasts.ratio_hat.iloc[0] < 0 and result.forecasts.y_pred.iloc[0] == 0
    assert result.forecasts.y_pred.iloc[1] == result.forecasts.national_hat.iloc[1] * result.forecasts.ratio_hat.iloc[1]


def test_invalid_denominator_remains_NaN_and_pair_loss_is_explicit(panel, cfg, recorder):
    panel = panel.copy()
    panel.loc["2023-04-01"] = 0
    result = model(panel, cfg, "LN").predict(panel, pd.Period("2023-12", "M"), 1,
                                            ["1", "2"], cfg["models"])
    assert result.training["n_pairs_excluded_by_ratio_validity"] == 2
    assert result.training["n_training_rows"] == result.training["n_nominal_source_pairs"] - 2
    april = next(row for row in result.local_ratio_diagnostics if row["period"] == pd.Timestamp("2023-04-01"))
    assert april["n_ratio_missing_invalid_denominator"] == 2
    assert april["denominator_status"] == "national_denominator_nonpositive"
    estimator = recorder.instances[-1]
    assert estimator.training_X.isna().any().any()


def test_actual_small_native_LightGBM_records_all_resolved_parameters(panel, cfg):
    pytest.importorskip("lightgbm")
    result = model(panel, cfg, "L0").predict(panel, pd.Period("2023-12", "M"), 1,
                                            ["1", "2"], cfg["models"])
    assert not result.errors and result.forecasts.status.eq("native").all()
    params = result.training["fit_actual_parameters"]
    resolved = result.training["fit_native_resolved_parameters"]
    assert params["objective"] == "regression_l1"
    assert params["num_threads"] == 2 and params["seed"] == 42
    assert params["num_iterations"] == 5
    assert resolved["objective"] == "regression_l1"
    assert resolved["device_type"] == "cpu"
    assert resolved["use_missing"] == "1" and resolved["zero_as_missing"] == "0"
    assert resolved["deterministic"] == "1" and resolved["force_col_wise"] == "1"
    assert len(resolved) > 80
    assert result.training["fit_actual_cat_feature_indices"] == []
    assert len(result.training["fit_actual_feature_names"]) == 19
    assert "tree_info" not in result.training["fit_booster_dump_metadata"]


@pytest.mark.parametrize("variant,extra", [("C0", {}), ("bad", {}), ("L0", {"device_type": "gpu"}),
                                         ("LN", {"zero_as_missing": True}),
                                         ("L0", {"random_state": 999}), ("CN", {"cat_features": ["month"]})])
def test_unplanned_variant_categorical_gpu_or_missing_policy_rejected(panel, cfg, variant, extra):
    parameters = dict(cfg["models"]["catboost"] if variant == "CN" else lgb_parameters(), **extra)
    with pytest.raises(ValueError):
        NationalLocalDirect(parameters, cfg["seed"], variant, build_national(panel))


def test_missing_national_series_and_strict_mode_are_rejected(cfg):
    with pytest.raises(ValueError):
        NationalLocalDirect(cfg["models"]["catboost"], 42, "CN")
    with pytest.raises(ValueError):
        NationalLocalDirect(lgb_parameters(), 42, "L0", training_mode="strict12")
