"""Numerical and cohort-isolation checks, using only synthetic test fixtures."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import sberforecast.early_warning_synthetic_models as module


CONFIG = {"models": dict(C=1.0, max_iter=2000, random_state=42, class_weight=None,
                          n_jobs=1, backend=module.BACKEND_POLICY)}
GROUPS = dict(history=["history_x"], detector=["detector_score"], external=["external_pressure"])


def data(cohort="train", *, constant_extra=False):
    """Both horizons share independent synthetic units, not municipal data."""
    x = np.r_[np.full(10, -1.0), np.full(10, 1.0)]
    y = np.r_[np.ones(2), np.zeros(8), np.ones(8), np.zeros(2)]
    prefix = cohort + "_"
    features = pd.DataFrame(dict(series_id=[f"{prefix}{i:02}" for i in range(20)],
                                 cohort=cohort, month_index=11, history_x=x,
                                 detector_score=0.0 if constant_extra else np.cos(np.arange(20)),
                                 external_pressure=0.0 if constant_extra else np.sin(np.arange(20))))
    cases = pd.concat([features[list(module.KEYS)].assign(k=k, label=y, fully_known=True,
                       eligible=True, at_risk=True, active_regime=False) for k in (1, 3)], ignore_index=True)
    return cases, features


@pytest.fixture(scope="module")
def fitted():
    cases, features = data()
    return module.fit_training_models(cases, features, GROUPS, CONFIG)


def test_analytic_gradient_matches_independent_finite_differences():
    X = np.array([[1.0, -2.0], [0.0, 3.0], [-1.5, 0.2], [2.0, 1.0]])
    y = np.array([1.0, 0.0, 0.0, 1.0])
    params = np.array([0.3, -0.7, 0.4])
    loss, gradient = module.logistic_objective_and_gradient(params, X, y)
    step = 1e-6
    numeric = []
    for index in range(len(params)):
        delta = np.eye(len(params))[index] * step
        plus = module.logistic_objective_and_gradient(params + delta, X, y)[0]
        minus = module.logistic_objective_and_gradient(params - delta, X, y)[0]
        numeric.append((plus - minus) / (2 * step))
    assert np.isfinite(loss)
    np.testing.assert_allclose(gradient, numeric, rtol=1e-7, atol=1e-8)


def test_intercept_is_unpenalized_and_extreme_logits_stay_finite():
    X, y = np.zeros((4, 1)), np.array([0, 1, 0, 1])
    first = module.logistic_objective_and_gradient(np.array([2.0, 1000.0]), X, y)
    second = module.logistic_objective_and_gradient(np.array([2.0, 1000.0]), X, y, C=2.0)
    assert np.isfinite(first[0]) and np.isfinite(first[1]).all()
    assert first[0] - second[0] == pytest.approx(1.0)
    assert first[1][-1] == second[1][-1] == pytest.approx(2.0)
    assert first[1][0] == pytest.approx(2.0)
    assert second[1][0] == pytest.approx(1.0)


def test_logistic_fit_matches_separately_solved_balanced_one_dimensional_optimum():
    from scipy.optimize import brentq
    cases, features = data(constant_extra=True)
    fit = module.fit_training_models(cases, features, GROUPS, CONFIG)
    # 20 balanced observations: derivative is 20*(sigmoid(w)-0.8)+w.
    expected = brentq(lambda w: 20 * (1 / (1 + np.exp(-w)) - 0.8) + w, 0, 4)
    parameter = next(row for row in fit["parameters"] if row["model"] == "S1" and row["k"] == 1)
    assert parameter["weights"][0] == pytest.approx(expected, abs=2e-5)
    assert parameter["intercept"] == pytest.approx(0, abs=2e-5)
    prediction = module.predict_models(fit, *data("test", constant_extra=True))
    s1 = prediction.loc[(prediction.model == "S1") & (prediction.k == 1)]
    assert s1.iloc[:10].probability.max() < s1.iloc[10:].probability.min()
    assert fit["model_status"].converged.all()
    assert fit["classifier_fit_count"] == 6
    assert fit["constant_fit_count"] == 2


def test_no_information_intercept_models_equal_train_positive_rate():
    cases, features = data(constant_extra=True)
    features["history_x"] = 7.0
    for k in (1, 3):
        indices = cases.index[cases.k.eq(k)]
        cases.loc[indices, "label"] = np.r_[np.ones(6), np.zeros(14)]
    fit = module.fit_training_models(cases, features, GROUPS, CONFIG)
    prediction = module.predict_models(fit, *data("test"))
    np.testing.assert_allclose(prediction.probability, 0.3, atol=1e-15)
    assert fit["classifier_fit_count"] == 0
    assert fit["model_status"].retained_features.eq(0).all()
    assert set(fit["model_status"].backend) == {"analytic_constant_prior", "analytic_intercept_only"}


def test_prediction_and_training_parameters_ignore_test_labels_and_cohort_values(fitted):
    cases, features = data("test")
    before = json.dumps(fitted["parameters"], sort_keys=True, allow_nan=False)
    expected = module.predict_models(fitted, cases, features)
    changed = cases.copy()
    changed["label"] = "TEST labels must never be inspected by inference"
    changed["event_onset_index"] = -100000
    actual = module.predict_models(fitted, changed, features)
    np.testing.assert_array_equal(actual.probability, expected.probability)
    np.testing.assert_array_equal(actual.alert, expected.alert)
    assert actual.label.eq("TEST labels must never be inspected by inference").all()
    assert json.dumps(fitted["parameters"], sort_keys=True, allow_nan=False) == before


def test_train_only_median_scaler_filter_and_existing_missing_indicator():
    cases, features = data()
    features["history_partial"] = np.arange(20, dtype=float)
    features.loc[:4, "history_partial"] = np.nan
    features["history_partial_missing"] = features.history_partial.isna().astype(int)
    features["history_duplicate"] = features.history_x
    groups = {**GROUPS, "history": GROUPS["history"] + ["history_partial", "history_partial_missing", "history_duplicate"]}
    untouched = features.copy(deep=True)
    fit = module.fit_training_models(cases, features, groups, CONFIG)
    pd.testing.assert_frame_equal(features, untouched)
    parameter = next(row for row in fit["parameters"] if row["model"] == "S1" and row["k"] == 1)
    assert parameter["medians"]["history_partial"] == 12.0
    expected = features.history_partial.fillna(12)
    assert parameter["means"]["history_partial"] == pytest.approx(expected.mean())
    assert parameter["scales"]["history_partial"] == pytest.approx(expected.std(ddof=0))
    assert "history_partial_missing" in parameter["feature_names"]
    assert "history_duplicate" not in parameter["feature_names"]
    audit = fit["feature_filter"]
    assert audit.loc[audit.feature.eq("history_duplicate"), "drop_reason"].eq("exact_numeric_duplicate_on_train").all()
    test_cases, test_features = data("test")
    test_features["history_partial"] = np.nan
    test_features["history_partial_missing"] = 1
    test_features["history_duplicate"] = np.linspace(100000, 200000, 20)
    before = json.dumps(fit["parameters"], sort_keys=True, allow_nan=False)
    prediction = module.predict_models(fit, test_cases, test_features)
    assert np.isfinite(prediction.probability).all()
    assert json.dumps(fit["parameters"], sort_keys=True, allow_nan=False) == before


def test_external_varying_only_on_test_is_removed_and_cannot_change_s3():
    train_cases, train_features = data()
    train_features["external_pressure"] = 1.0
    train_features["external_missing"] = np.nan
    groups = {**GROUPS, "external": ["external_pressure", "external_missing"]}
    fit = module.fit_training_models(train_cases, train_features, groups, CONFIG)
    cases, features = data("test")
    features["external_pressure"] = np.linspace(-1e8, 1e8, 20)
    features["external_missing"] = np.arange(20)
    prediction = module.predict_models(fit, cases, features)
    for k in (1, 3):
        s2 = prediction.loc[prediction.model.eq("S2") & prediction.k.eq(k), "probability"]
        s3 = prediction.loc[prediction.model.eq("S3") & prediction.k.eq(k), "probability"]
        np.testing.assert_array_equal(s2.to_numpy(), s3.to_numpy())
        parameter = next(row for row in fit["parameters"] if row["model"] == "S3" and row["k"] == k)
        assert all(not name.startswith("external_") for name in parameter["feature_names"])
    external = fit["feature_filter"].loc[lambda frame: frame.group.eq("external")]
    assert not external.selected.any()
    assert set(external.drop_reason) == {"all_missing_on_train", "constant_after_training_median_imputation"}


def test_serializable_inference_roundtrip_and_original_units_coefficients(fitted):
    restored = json.loads(json.dumps({"parameters": fitted["parameters"]}, allow_nan=False, sort_keys=True))
    cases, features = data("test")
    first = module.predict_models(fitted, cases, features)
    second = module.predict_models(restored, cases, features)
    pd.testing.assert_frame_equal(first, second)
    parameter = next(row for row in restored["parameters"] if row["model"] == "S3" and row["k"] == 1)
    coefficients = fitted["coefficients"].loc[lambda frame: frame.model.eq("S3") & frame.k.eq(1)].set_index("feature")
    names = parameter["feature_names"]
    score = np.full(len(features), coefficients.loc["intercept", "coefficient_original_units"])
    for name in names:
        score += features[name].fillna(parameter["medians"][name]) * coefficients.loc[name, "coefficient_original_units"]
    expected = 1 / (1 + np.exp(-score))
    actual = first.loc[first.model.eq("S3") & first.k.eq(1), "probability"]
    np.testing.assert_allclose(expected, actual, atol=1e-14, rtol=1e-14)


@pytest.mark.parametrize("cohort", ["validation", "test"])
def test_fit_refuses_validation_and_test_cases(cohort):
    cases, features = data(cohort)
    with pytest.raises(ValueError, match="TRAIN cohort only"):
        module.fit_training_models(cases, features, GROUPS, CONFIG)


def test_training_features_cannot_include_another_cohort():
    cases, features = data()
    features.loc[0, "cohort"] = "test"
    with pytest.raises(ValueError, match="TRAIN only"):
        module.fit_training_models(cases, features, GROUPS, CONFIG)


@pytest.mark.parametrize("name", ["label", "history_label", "history_event_onset", "external_is_anticipated",
                                  "detector_offline_cusum", "history_time_to_event", "detector_pelt_score",
                                  "history_future_residual", "external_has_event", "history_event_id"])
def test_target_event_and_noncausal_fields_cannot_be_group_features(name):
    cases, features = data()
    groups = {**GROUPS, "history": [name]}
    features[name] = 0.0
    with pytest.raises(ValueError, match="Noncausal/target metadata"):
        module.fit_training_models(cases, features, groups, CONFIG)


def test_observed_external_event_pressure_namespace_is_permitted():
    cases, features = data()
    features = features.rename(columns={"external_pressure": "external_event_pressure"})
    groups = {**GROUPS, "external": ["external_event_pressure"]}
    fit = module.fit_training_models(cases, features, groups, CONFIG)
    assert "external_event_pressure" in fit["feature_filter"].feature.tolist()


def test_unknown_active_ineligible_and_not_at_risk_training_rows_are_excluded():
    cases, features = data()
    extra_cases, extra_features = [], []
    for name, changes in [
        ("unknown", dict(fully_known=False, label=np.nan)),
        ("active", dict(active_regime=True, label=1)),
        ("ineligible", dict(eligible=False, label=1)),
        ("not_risk", dict(at_risk=False, label=1)),
    ]:
        one = features.iloc[[0]].copy()
        one["series_id"] = "train_" + name
        extra_features.append(one)
        for k in (1, 3):
            row = cases.iloc[[0]].copy().assign(series_id="train_" + name, k=k)
            for key, value in changes.items():
                row[key] = value
            extra_cases.append(row)
    fit = module.fit_training_models(pd.concat([cases, *extra_cases], ignore_index=True),
                                     pd.concat([features, *extra_features], ignore_index=True), GROUPS, CONFIG)
    assert fit["model_status"].training_rows.eq(20).all()
    assert fit["model_status"].training_positives.eq(10).all()
    assert all("unknown" not in name and "active" not in name for name in fit["training_ids"])


def test_censored_training_row_cannot_silently_become_negative():
    cases, features = data()
    cases.loc[0, ["fully_known", "label"]] = [False, 0]
    with pytest.raises(ValueError, match="Unknown/censored"):
        module.fit_training_models(cases, features, GROUPS, CONFIG)


def test_unconverged_optimizer_aborts_with_diagnostics(monkeypatch):
    import scipy.optimize
    monkeypatch.setattr(module, "_backend", lambda: "scipy")
    monkeypatch.setattr(scipy.optimize, "minimize", lambda function, initial, **kwargs:
                        SimpleNamespace(x=initial, success=False, nit=2000, status=1, message="Synthetic iteration-limit fixture"))
    with pytest.raises(module.ModelConvergenceError) as error:
        module.fit_training_models(*data(), GROUPS, CONFIG)
    diagnostics = error.value.diagnostics
    assert not diagnostics["converged"]
    assert diagnostics["model"] == "S1" and diagnostics["k"] == 1
    assert diagnostics["iterations"] == 2000
    assert np.isfinite(diagnostics["objective_value"])


@pytest.mark.parametrize("key,value", [("C", 2.0), ("max_iter", 3000), ("random_state", 43),
                                       ("class_weight", "balanced"), ("n_jobs", 2)])
def test_fixed_model_configuration_cannot_be_tuned(key, value):
    config = {"models": {**CONFIG["models"], key: value}}
    with pytest.raises(ValueError, match="fixed model parameter"):
        module.fit_training_models(*data(), GROUPS, config)


def test_fixed_fit_is_deterministic():
    first = module.fit_training_models(*data(), GROUPS, CONFIG)
    second = module.fit_training_models(*data(), GROUPS, CONFIG)
    assert first["parameters"] == second["parameters"]
    pd.testing.assert_frame_equal(first["coefficients"], second["coefficients"])


def test_threshold_application_changes_only_decisions_and_preserves_identical_cases(fitted):
    cases, features = data("validation")
    default = module.predict_models(fitted, cases, features)
    thresholds = pd.DataFrame([dict(model=model, k=k, threshold=0.9) for model in module.MODEL_GROUPS for k in (1, 3)])
    changed = module.predict_models(fitted, cases, features, thresholds)
    np.testing.assert_array_equal(changed.probability, default.probability)
    assert changed.threshold.eq(0.9).all()
    assert changed.alert.equals(changed.probability.ge(0.9))
    assert default.threshold_status.eq("temporary_default_for_validation").all()
    assert changed.threshold_status.eq("provided_validation_selection").all()
    for (_, k), group in changed.groupby(["model", "k"]):
        assert len(group) == 20 and group.cohort.eq("validation").all()
        assert set(group.series_id) == set(cases.loc[cases.k.eq(k), "series_id"])


def test_prediction_audit_rows_keep_unknown_and_active_flags_without_issuing_alerts(fitted):
    cases, features = data("test")
    cases.loc[0, ["fully_known", "label"]] = [False, np.nan]
    cases.loc[1, "active_regime"] = True
    thresholds = {(model, k): 0.0 for model in module.MODEL_GROUPS for k in (1, 3)}
    prediction = module.predict_models(fitted, cases, features, thresholds)
    unknown = prediction.loc[prediction.series_id.eq(cases.loc[0, "series_id"]) & prediction.k.eq(1)]
    active = prediction.loc[prediction.series_id.eq(cases.loc[1, "series_id"]) & prediction.k.eq(1)]
    assert unknown.label.isna().all() and not unknown.alert.any()
    assert active.active_regime.all() and not active.alert.any()
    assert np.isfinite(prediction.probability).all()


def test_duplicate_features_missing_join_and_infinite_selected_values_are_refused(fitted):
    cases, features = data("test")
    with pytest.raises(ValueError, match="Duplicate"):
        module.predict_models(fitted, cases, pd.concat([features, features.iloc[[0]]]))
    with pytest.raises(ValueError, match="own-origin"):
        module.predict_models(fitted, cases, features.iloc[1:])
    features.loc[0, "history_x"] = np.inf
    with pytest.raises(ValueError, match="Infinite"):
        module.predict_models(fitted, cases, features)


def test_partial_threshold_table_cannot_silently_apply_default_to_test(fitted):
    with pytest.raises(ValueError, match="VALIDATION-selected threshold"):
        module.predict_models(fitted, *data("test"), thresholds={("S0", 1): 0.6})


def test_single_class_train_does_not_pretend_to_fit_logistic_models():
    cases, features = data()
    cases["label"] = 0
    with pytest.raises(ValueError, match="both TRAIN classes"):
        module.fit_training_models(cases, features, GROUPS, CONFIG)
