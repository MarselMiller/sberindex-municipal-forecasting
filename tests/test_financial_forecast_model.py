"""Synthetic E08c checks: recording learner only, no real fits or source data."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sberforecast import financial_forecast_model as financial
from sberforecast import national_local_model as original
from sberforecast.data import get_prefix
from sberforecast.direct_model import TemporalIntegrityError
from sberforecast.direct_training import build_direct_training, direct_features
from sberforecast.leading_financial import FEATURE_COLUMNS, origin_timestamp
from sberforecast.national_local import build_local_training, build_national


class RecordingLearner:
    instances = []

    def __init__(self, **parameters):
        self.parameters = dict(parameters)
        self.instances.append(self)

    def fit(self, X, y):
        self.training_X = X.copy(deep=True)
        self.training_y = np.asarray(y).copy()
        self.feature_name_ = X.columns.tolist()
        self.feature_importances_ = np.zeros(len(X.columns))
        return self

    def predict(self, X):
        self.forecast_X = X.copy(deep=True)
        return np.zeros(len(X))

    def get_params(self, deep=True):
        return dict(self.parameters)


@pytest.fixture
def recorder(monkeypatch):
    RecordingLearner.instances = []
    monkeypatch.setattr(original, "_lightgbm_regressor", RecordingLearner)
    monkeypatch.setattr(original, "CatBoostRegressor", lambda **kwargs: pytest.fail("CatBoost forbidden"))
    return RecordingLearner


@pytest.fixture
def matrix():
    origins = pd.period_range("2023-01", "2025-06", freq="M")
    stamps = [origin_timestamp(origin) for origin in origins]
    result = pd.DataFrame({"forecast_origin": stamps})
    for position, name in enumerate(FEATURE_COLUMNS):
        result[name] = np.arange(len(origins), dtype=float) * 10 + position + 1
        result[f"{name}_missing"] = False
    for name in ("max_source_date_used", "max_source_available_at_used",
                 "key_rate_last_available_at", "key_rate_last_effective_date",
                 "usd_rub_last_available_at", "usd_rub_last_effective_date"):
        result[name] = stamps
    result["usd_rub_method_regime"] = "synthetic_regime"
    result["source_url"] = "https://example.invalid/synthetic-financial"
    return result


@pytest.fixture
def parameters():
    return {"objective": "regression_l1", "n_estimators": 300, "learning_rate": .05,
            "num_leaves": 31, "random_state": 42, "n_jobs": 2, "device_type": "cpu",
            "deterministic": True, "force_col_wise": True, "verbosity": -1}


def make_model(panel, parameters, matrix, base_variant="LN", variant="F3", lag=0):
    return financial.FinancialDirect(
        parameters, 42, base_variant, matrix, financial.FINANCIAL_VARIANTS[variant],
        build_national(panel, lag) if base_variant == "LN" else None,
        release_lag_months=lag)


def predict(model, panel, cfg, horizon=1, origin="2023-12", ids=None):
    return model.predict(panel, pd.Period(origin, freq="M"), horizon,
                         ["2", "1"] if ids is None else ids, cfg["models"])


@pytest.mark.parametrize("base_variant", ["L0", "LN"])
def test_F0_calls_original_predict_and_preserves_complete_baseline(
        panel, cfg, parameters, recorder, monkeypatch, base_variant):
    calls = []
    old_predict = original.NationalLocalDirect.predict

    def record_call(self, *args, **kwargs):
        calls.append(self.variant)
        return old_predict(self, *args, **kwargs)

    monkeypatch.setattr(original.NationalLocalDirect, "predict", record_call)
    national = build_national(panel) if base_variant == "LN" else None
    baseline = predict(original.NationalLocalDirect(parameters, 42, base_variant, national), panel, cfg, 3)
    learner_before = recorder.instances[-1]
    # F0 has no financial dependency, even on an empty matrix without a schema.
    result = predict(make_model(panel, parameters, pd.DataFrame(), base_variant, "F0"), panel, cfg, 3)
    learner_after = recorder.instances[-1]
    assert calls == [base_variant, base_variant]
    pd.testing.assert_frame_equal(result.forecasts, baseline.forecasts)
    pd.testing.assert_frame_equal(result.training_provenance, baseline.training_provenance)
    pd.testing.assert_frame_equal(learner_after.training_X, learner_before.training_X)
    pd.testing.assert_frame_equal(learner_after.forecast_X, learner_before.forecast_X)
    np.testing.assert_array_equal(learner_after.training_y, learner_before.training_y)
    assert learner_after.parameters == learner_before.parameters
    for key, value in baseline.training.items():
        if key != "seconds":
            assert result.training[key] == value
    assert result.national_forecast == baseline.national_forecast
    assert result.financial_training_provenance.empty


@pytest.mark.parametrize("base_variant", ["L0", "LN"])
@pytest.mark.parametrize("variant", ["F1", "F2", "F3"])
@pytest.mark.parametrize("horizon", [1, 3, 6])
def test_ablation_adds_exact_subset_at_own_r_preserving_pairs_and_labels(
        panel, cfg, parameters, matrix, recorder, base_variant, variant, horizon):
    panel = panel.copy()
    panel.loc["2023-03-01", "1"] = np.nan
    result = predict(make_model(panel, parameters, matrix, base_variant, variant), panel, cfg, horizon)
    learner = recorder.instances[-1]
    prefix = get_prefix(panel, pd.Period("2023-12", "M"), 0)
    if base_variant == "LN":
        X, y, trace = build_local_training(prefix, build_national(panel), "2023-12", horizon,
                                            release_lag_months=0, mode="legacy")
    else:
        X, y, trace = build_direct_training(prefix, "2023-12", horizon,
                                             release_lag_months=0, mode="legacy")
    selected = financial.FINANCIAL_VARIANTS[variant]
    assert learner.training_X.columns.tolist() == X.columns.tolist() + list(selected)
    assert len(X.columns) == 19
    pd.testing.assert_frame_equal(learner.training_X[X.columns], X)
    np.testing.assert_array_equal(learner.training_y, y)
    pd.testing.assert_frame_equal(result.training_provenance[trace.columns], trace)
    assert result.training["training_ordered_key_sha256"] == original.ordered_training_key_signature(trace)
    expected = matrix.set_index("forecast_origin").reindex(
        trace.historical_origin.map(origin_timestamp))[list(selected)]
    np.testing.assert_array_equal(learner.training_X[list(selected)].to_numpy(), expected.to_numpy())
    own = pd.DatetimeIndex(result.training_provenance.financial_feature_origin)
    assert own.equals(pd.DatetimeIndex(trace.historical_origin.map(origin_timestamp)))
    assert (own <= origin_timestamp("2023-12")).all()
    expected_forecast = matrix.loc[matrix.forecast_origin.eq(origin_timestamp("2023-12")), list(selected)]
    np.testing.assert_array_equal(learner.forecast_X[list(selected)].to_numpy(),
                                  np.repeat(expected_forecast.to_numpy(), 2, axis=0))
    assert result.forecasts.municipality_id.tolist() == ["2", "1"]
    assert result.financial_forecast_provenance.municipality_id.tolist() == ["2", "1"]
    assert result.training["financial_training_provenance_sha256"] == original.frame_signature(
        result.financial_training_provenance)
    assert result.training["financial_forecast_provenance_sha256"] == original.frame_signature(
        result.financial_forecast_provenance)
    assert not any("missing" in name or "regime" in name or "source" in name
                   or "origin" in name for name in learner.training_X)
    assert result.audit_counts["training_rows_checked"] == len(trace)
    assert result.audit_counts["forecast_rows_checked"] == 2
    assert result.audit_counts["financial_cutoff_violations"] == 0
    assert learner.parameters == dict(parameters, use_missing=True, zero_as_missing=False)


@pytest.mark.parametrize("base_variant", ["L0", "LN"])
def test_release_lag_joins_historical_origin_not_target_or_expense_cutoff(
        panel, cfg, parameters, matrix, recorder, base_variant):
    result = predict(make_model(panel, parameters, matrix, base_variant, "F1", lag=1),
                     panel, cfg, 3, origin="2024-01")
    trace = result.training_provenance
    selected = financial.FINANCIAL_VARIANTS["F1"]
    actual = recorder.instances[-1].training_X[list(selected)].to_numpy()
    lookup = matrix.set_index("forecast_origin")
    expected = lookup.reindex(trace.historical_origin.map(origin_timestamp))[list(selected)].to_numpy()
    wrong_cutoff = lookup.reindex(trace.feature_cutoff.map(origin_timestamp))[list(selected)].to_numpy()
    wrong_target = lookup.reindex(trace.target_period.dt.to_period("M").map(origin_timestamp))[list(selected)].to_numpy()
    np.testing.assert_array_equal(actual, expected)
    assert not np.array_equal(actual, wrong_cutoff)
    assert not np.array_equal(actual, wrong_target)


@pytest.mark.parametrize("scope", ["training", "forecast"])
@pytest.mark.parametrize("column", ["max_source_date_used", "max_source_available_at_used",
                                     "key_rate_last_available_at", "usd_rub_last_effective_date"])
def test_future_source_or_availability_stops_before_learner_construction(
        panel, cfg, parameters, matrix, recorder, scope, column):
    corrupted = matrix.copy(deep=True)
    own = origin_timestamp("2023-02" if scope == "training" else "2023-12")
    corrupted.loc[corrupted.forecast_origin.eq(own), column] = own + pd.Timedelta(hours=1)
    with pytest.raises(TemporalIntegrityError, match="own financial origin"):
        predict(make_model(panel, parameters, corrupted), panel, cfg)
    assert not recorder.instances


@pytest.mark.parametrize("scope", ["training", "forecast"])
def test_missing_own_origin_is_an_error_before_fit_not_row_deletion_or_fallback(
        panel, cfg, parameters, matrix, recorder, scope):
    own = origin_timestamp("2023-02" if scope == "training" else "2023-12")
    incomplete = matrix.loc[~matrix.forecast_origin.eq(own)]
    with pytest.raises(TemporalIntegrityError, match="Missing own-origin"):
        predict(make_model(panel, parameters, incomplete), panel, cfg)
    assert not recorder.instances


def test_duplicate_normalized_financial_origin_is_rejected_before_fit(
        panel, cfg, parameters, matrix, recorder):
    duplicate = matrix.iloc[[0]].copy()
    duplicate["forecast_origin"] = duplicate.forecast_origin.dt.tz_convert("UTC")
    bad = pd.concat([matrix, duplicate], ignore_index=True)
    with pytest.raises(ValueError, match="unique after normalization"):
        predict(make_model(panel, parameters, bad), panel, cfg)
    assert not recorder.instances


def test_missing_numeric_features_preserve_rows_and_are_not_promoted_to_flags(
        panel, cfg, parameters, matrix, recorder):
    matrix = matrix.copy(deep=True)
    matrix.loc[matrix.forecast_origin.eq(origin_timestamp("2023-02")), "key_rate_change_6m"] = np.nan
    matrix.loc[matrix.forecast_origin.eq(origin_timestamp("2023-02")), "key_rate_change_6m_missing"] = True
    result = predict(make_model(panel, parameters, matrix, variant="F1"), panel, cfg)
    rows = result.training_provenance.historical_origin.eq(pd.Timestamp("2023-02-28"))
    assert rows.sum() == 2
    assert recorder.instances[-1].training_X.loc[rows, "key_rate_change_6m"].isna().all()
    assert result.audit_counts["training"]["n_missing_feature_values"] == 2
    assert "key_rate_change_6m_missing" not in recorder.instances[-1].training_X
    assert result.training_provenance.loc[rows, "financial_key_rate_change_6m_missing"].all()


@pytest.mark.parametrize("base_variant", ["L0", "LN"])
def test_future_matrix_append_removal_and_mutation_leave_past_training_and_prediction_unchanged(
        panel, cfg, parameters, matrix, recorder, base_variant):
    current = origin_timestamp("2023-12")
    before = predict(make_model(panel, parameters, matrix, base_variant), panel, cfg, 3)
    old = recorder.instances[-1]
    changed = matrix.copy(deep=True)
    future = changed.forecast_origin.gt(current)
    changed.loc[future, list(FEATURE_COLUMNS)] = -999.0
    changed.loc[future, "max_source_available_at_used"] = origin_timestamp("2050-12")
    for candidate in (changed, changed.loc[~future]):
        after = predict(make_model(panel, parameters, candidate, base_variant), panel, cfg, 3)
        new = recorder.instances[-1]
        pd.testing.assert_frame_equal(new.training_X, old.training_X)
        pd.testing.assert_frame_equal(new.forecast_X, old.forecast_X)
        np.testing.assert_array_equal(new.training_y, old.training_y)
        pd.testing.assert_frame_equal(after.forecasts, before.forecasts)
        pd.testing.assert_frame_equal(after.training_provenance, before.training_provenance)
        for key in ("training_key_sha256", "training_delta_sha256", "training_raw_X_sha256",
                    "forecast_raw_X_sha256", "financial_training_provenance_sha256"):
            assert after.training[key] == before.training[key]


def test_matrix_order_and_input_mutation_do_not_change_model_alignment(
        panel, cfg, parameters, matrix, recorder):
    original_panel, original_matrix = panel.copy(deep=True), matrix.copy(deep=True)
    before = predict(make_model(panel, parameters, matrix), panel, cfg, 3)
    old = recorder.instances[-1]
    after = predict(make_model(panel, parameters, matrix.sample(frac=1, random_state=123)), panel, cfg, 3)
    new = recorder.instances[-1]
    pd.testing.assert_frame_equal(new.training_X, old.training_X)
    pd.testing.assert_frame_equal(new.forecast_X, old.forecast_X)
    pd.testing.assert_frame_equal(after.training_provenance, before.training_provenance)
    pd.testing.assert_frame_equal(panel, original_panel)
    pd.testing.assert_frame_equal(matrix, original_matrix)


@pytest.mark.parametrize("base_variant", ["L0", "LN"])
@pytest.mark.parametrize("variant", ["F0", "F1", "F2", "F3"])
def test_h12_no_pairs_preserves_original_expense_fallback_without_any_learner(
        panel, cfg, parameters, matrix, recorder, base_variant, variant):
    old = original.NationalLocalDirect(parameters, 42, base_variant,
                                       build_national(panel) if base_variant == "LN" else None)
    expected = predict(old, panel, cfg, 12)
    result = predict(make_model(panel, parameters, matrix, base_variant, variant), panel, cfg, 12)
    assert not recorder.instances
    assert not result.training["fit_called"]
    assert result.training["n_training_rows"] == 0
    pd.testing.assert_frame_equal(result.forecasts, expected.forecasts)
    assert result.forecasts.status.eq("fallback_no_training_pairs").all()
    assert result.forecasts.effective_model.eq("SeasonalNaive").all()
    assert result.forecasts.ratio_hat.isna().all()
    np.testing.assert_array_equal(result.forecasts.y_pred, [222.0, 111.0])


@pytest.mark.parametrize("base_variant", ["L0", "LN"])
@pytest.mark.parametrize("stage", ["fit", "predict"])
def test_learner_failures_remain_failed_without_hidden_reserve(
        panel, cfg, parameters, matrix, recorder, monkeypatch, base_variant, stage):
    def fail(*args):
        raise RuntimeError(f"synthetic {stage} failure")

    monkeypatch.setattr(recorder, stage, fail)
    monkeypatch.setattr(financial, "baseline_predict", lambda *args: pytest.fail("hidden fallback"))
    result = predict(make_model(panel, parameters, matrix, base_variant), panel, cfg)
    assert result.forecasts.status.eq("failed").all()
    assert result.forecasts.y_pred.isna().all()
    assert result.errors[0]["stage"] == stage
    assert result.training["fit_succeeded"] == (stage == "predict")


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_nonfinite_learner_predictions_remain_failed(
        panel, cfg, parameters, matrix, recorder, monkeypatch, bad):
    monkeypatch.setattr(recorder, "predict", lambda self, X: np.full(len(X), bad))
    result = predict(make_model(panel, parameters, matrix), panel, cfg)
    assert result.forecasts.status.eq("failed").all()
    assert result.errors[0]["stage"] == "predict"


@pytest.mark.parametrize("features", [
    ["key_rate_level"], list(reversed(FEATURE_COLUMNS)),
    list(FEATURE_COLUMNS) + ["usd_rub_last_missing"], ["usd_rub_method_regime"],
    ["invented_financial"], list(FEATURE_COLUMNS[:5]) + ["key_rate_level"],
])
def test_unplanned_or_derived_financial_feature_sets_are_rejected(panel, parameters, matrix, features):
    with pytest.raises(ValueError, match="ordered F0/F1/F2/F3"):
        financial.FinancialDirect(parameters, 42, "LN", matrix, features, build_national(panel))


@pytest.mark.parametrize("variant", ["CN", "C0", "unknown"])
def test_no_other_model_family_is_allowed(panel, parameters, matrix, variant):
    with pytest.raises(ValueError, match="L0 and LN"):
        financial.FinancialDirect(parameters, 42, variant, matrix, FEATURE_COLUMNS, build_national(panel))


@pytest.mark.parametrize("column", list(financial.SOURCE_AUDIT_COLUMNS))
def test_populated_features_require_audit_dates(panel, cfg, parameters, matrix, recorder, column):
    matrix = matrix.copy(deep=True)
    matrix.loc[matrix.forecast_origin.eq(origin_timestamp("2023-02")), column] = pd.NaT
    with pytest.raises(TemporalIntegrityError, match="lack source audit dates"):
        predict(make_model(panel, parameters, matrix), panel, cfg)
    assert not recorder.instances


def test_pure_augmentation_preserves_nonrange_index_and_all_missing_history(matrix):
    X = pd.DataFrame({"base": [np.nan, 2.0]}, index=["municipality_b", "municipality_a"])
    matrix = matrix.copy(deep=True)
    own = origin_timestamp("2023-01")
    selected = financial.FINANCIAL_VARIANTS["F1"]
    matrix.loc[matrix.forecast_origin.eq(own), list(selected)] = np.nan
    matrix.loc[matrix.forecast_origin.eq(own), list(financial.SOURCE_AUDIT_COLUMNS)] = pd.NaT
    augmented, provenance, audit = financial.augment_financial_features(
        X, ["2023-01", "2023-01"], matrix, selected, "2023-12")
    pd.testing.assert_frame_equal(augmented[["base"]], X)
    assert augmented[list(selected)].isna().all().all()
    assert provenance.index.equals(X.index)
    assert audit["n_missing_feature_values"] == 10


def test_pure_augmentation_rejects_own_origin_after_model_origin(matrix):
    with pytest.raises(TemporalIntegrityError, match="later than model"):
        financial.augment_financial_features(pd.DataFrame({"base": [1.0]}), ["2024-01"],
                                             matrix, FEATURE_COLUMNS, "2023-12")
