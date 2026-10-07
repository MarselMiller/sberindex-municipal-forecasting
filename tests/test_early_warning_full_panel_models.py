"""Synthetic train/test data and mock classifiers; no installation or real fit."""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from sberforecast import early_warning_full_panel_models as models


def synthetic_gate_data():
    train, test, registry, features = [], [], [], []
    for split, origins in [("train", ["2024-01", "2024-02", "2024-03"]),
                           ("test", ["2024-07", "2024-08"])]:
        for origin in origins:
            period = pd.Period(origin, freq="M")
            for index in range(20):
                uid = str(index)
                positive = index < (10 if split == "train" else 5)
                event_id = f"synthetic_{uid}_{period + 1}"
                row = dict(municipality_id=uid, forecast_origin=str(period.end_time.normalize().date()), k=1,
                           label=int(positive), fully_known=True, at_risk=True,
                           positive_event_ids=event_id if positive else "",
                           label_known_at=str((period + 3).end_time.normalize().date()))
                (train if split == "train" else test).append(row)
                if positive:
                    registry.append(dict(municipality_id=uid, event_id=event_id, onset_period=str(period + 1)))
                features.append(dict(municipality_id=uid, forecast_origin=row["forecast_origin"],
                                     expense_value=index + period.month / 100,
                                     residual_value=np.nan if index == 1 else index % 3,
                                     residual_value_missing=index == 1,
                                     online_score=(index + period.month) % 4,
                                     macro_value=float(period.month),
                                     news_count=float(period.month % 2)))
    return pd.DataFrame(train), pd.DataFrame(test), pd.DataFrame(registry), pd.DataFrame(features)


GROUPS = dict(history=["expense_value", "residual_value", "residual_value_missing"],
              detector=["online_score"], macro=["macro_value"], news=["news_count"])


class SyntheticClassifier:
    """Deterministic test double, never claimed to be sklearn or real metrics."""
    calls = []
    def __init__(self, **kwargs): self.kwargs = kwargs
    def fit(self, matrix, labels):
        self.__class__.calls.append((matrix.copy(), labels.copy(), self.kwargs.copy()))
        self.coef_ = np.array([np.mean(matrix * (labels - labels.mean())[:, None], axis=0)])
        self.intercept_ = np.array([0.0])
        return self
    def predict_proba(self, matrix):
        positive = 1 / (1 + np.exp(-(matrix @ self.coef_[0] + self.intercept_[0])))
        return np.column_stack([1 - positive, positive])


def test_gate_counts_unique_event_onset_dates_not_municipality_rows():
    train, test, events, _ = synthetic_gate_data()
    gate = models.feasibility_gate(train, test, events, 1)
    assert gate["passed"] and gate["train_positives"] == 30 and gate["test_positives"] == 10
    assert gate["train_positive_event_dates"] == 3 and gate["test_positive_event_dates"] == 2
    assert gate["train_positive_event_ids"] == 30 and gate["test_positive_event_ids"] == 10
    assert gate["requirements"] == models.GATE_REQUIREMENTS


def test_gate_cannot_be_overridden_and_refuses_before_any_fit(monkeypatch):
    train, test, events, features = synthetic_gate_data()
    train = train.loc[train.forecast_origin.ne("2024-01-31")]
    def prohibited(*args, **kwargs): raise AssertionError("No fit/import before gate")
    monkeypatch.setattr(models, "_logistic_regression_class", prohibited)
    monkeypatch.setattr(models.TrainOnlyPreprocessor, "fit", prohibited)
    result = models.fit_baselines(train, test, features, events, GROUPS, 1, gate={"passed": True})
    assert result["fit_count"] == 0 and result["status"] == "gate_failed"
    for name in ["predictions", "metrics_row", "metrics_event", "calibration", "coefficients", "alerts"]:
        assert result[name].empty and len(result[name].columns) > 0
    assert len(result["model_status"]) == 5 and not result["model_status"].fit_performed.any()


def test_gate_refuses_late_training_labels_and_overlap_origins():
    train, test, events, _ = synthetic_gate_data()
    delayed = train.copy(); delayed.loc[delayed.index[0], "label_known_at"] = "2024-07-31"
    assert "training_labels_not_available_by_fit_cutoff" in models.feasibility_gate(delayed, test, events, 1)["reasons"]
    assert "fit_cutoff_not_before_first_test_origin" in models.feasibility_gate(train, test, events, 1, fit_cutoff="2024-07")["reasons"]


def test_unknown_labels_never_silently_become_negative():
    train, test, events, _ = synthetic_gate_data()
    train.loc[train.index[0], "fully_known"] = False
    with pytest.raises(ValueError, match="Unknown/censored"):
        models.feasibility_gate(train, test, events, 1)


def test_active_cases_excluded_causally_from_gate_counts():
    train, test, events, _ = synthetic_gate_data()
    train.loc[train.label.eq(1), "at_risk"] = False
    gate = models.feasibility_gate(train, test, events, 1)
    assert not gate["passed"] and gate["train_positives"] == 0


def test_origin_ineligible_case_does_not_contribute_to_gate_or_training(monkeypatch):
    train, test, events, features = synthetic_gate_data()
    train["eligible_at_origin"] = True
    train.loc[train.index[0], "eligible_at_origin"] = False
    def prohibited(*args, **kwargs): raise AssertionError("Ineligible positive must not cause a gate pass")
    monkeypatch.setattr(models, "_logistic_regression_class", prohibited)
    result = models.fit_baselines(train, test, features, events, GROUPS, 1)
    assert not result["gate"]["passed"]
    assert result["gate"]["train_positives"] == 29 and result["gate"]["train_cases"] == 59
    assert result["fit_count"] == 0 and result["predictions"].empty


def test_missing_sklearn_reported_without_fit_or_fake_predictions(monkeypatch):
    train, test, events, features = synthetic_gate_data()
    def missing(): raise ImportError("Synthetic unavailable sklearn")
    monkeypatch.setattr(models, "_logistic_regression_class", missing)
    result = models.fit_baselines(train, test, features, events, GROUPS, 1)
    assert result["gate"]["passed"] and result["status"] == "sklearn_unavailable"
    assert result["fit_count"] == 0 and result["predictions"].empty and result["metrics_row"].empty


def test_imputation_scaling_and_filtering_use_only_train():
    train = pd.DataFrame(dict(x=[1., np.nan, 3., 5.], x_missing=[False, True, False, False],
                              duplicate=[1., np.nan, 3., 5.], constant=[2., 2., np.nan, 2.],
                              absent=[np.nan] * 4))
    test = pd.DataFrame(dict(x=[1000., np.nan], x_missing=[False, True],
                             duplicate=[-2000., -3000.], constant=[99., 100.], absent=[1., 2.]))
    processor = models.TrainOnlyPreprocessor(list(train)).fit(train)
    assert processor.selected_columns_ == ["x", "x_missing"]
    assert processor.medians_["x"] == processor.means_["x"] == 3
    assert processor.scales_["x"] == pytest.approx(np.sqrt(2))
    before = (processor.medians_.copy(), processor.means_.copy(), processor.scales_.copy())
    output = processor.transform(test)
    assert output[0, 0] == pytest.approx((1000 - 3) / np.sqrt(2)) and output[1, 0] == 0
    for saved, actual in zip(before, [processor.medians_, processor.means_, processor.scales_]):
        pd.testing.assert_series_equal(saved, actual)
    reasons = processor.audit().set_index("feature").reason
    assert processor.audit().selected.equals(processor.audit().kept)
    assert processor.audit().loc[processor.audit().selected, "drop_reason"].eq("").all()
    assert reasons["absent"] == "all_missing_on_train"
    assert reasons["constant"] == "constant_after_training_median_imputation"
    assert reasons["duplicate"] == "exact_numeric_duplicate_on_train"


@pytest.mark.parametrize("name", ["label", "offline_pelt_score", "binseg_state", "breakpoint_month", "future_expense"])
def test_target_and_offline_fields_cannot_enter_feature_matrix(name):
    with pytest.raises(ValueError, match="Noncausal"):
        models.TrainOnlyPreprocessor([name])


def test_transform_before_train_fit_and_infinite_features_refused():
    with pytest.raises(ValueError, match="fitted on train first"):
        models.TrainOnlyPreprocessor(["x"]).transform(pd.DataFrame({"x": [1]}))
    with pytest.raises(ValueError, match="Infinite"):
        models.TrainOnlyPreprocessor(["x"]).fit(pd.DataFrame({"x": [1, np.inf]}))


def test_probability_metrics_AP_ties_ROC_proper_scores_and_fixed_calibration():
    metrics, bins = models.evaluate_probabilities([0, 1, 0, 1], [.1, .4, .4, .9])
    assert metrics["pr_auc"] == pytest.approx(1 / 2 + (1 / 2) * (2 / 3))
    assert metrics["roc_auc"] == pytest.approx(7 / 8)
    assert metrics["brier_score"] == pytest.approx((.01 + .36 + .16 + .01) / 4)
    assert metrics["precision"] == 1 and metrics["recall"] == .5 and metrics["f1"] == pytest.approx(2 / 3)
    assert len(bins) == 10 and bins["count"].sum() == 4
    assert "average_precision" in metrics["pr_auc_definition"]
    boundary, calibration = models.evaluate_probabilities([0, 1], [0, 1])
    assert calibration.iloc[0]["count"] == calibration.iloc[-1]["count"] == 1
    assert boundary["log_loss"] < 1e-12


def test_absent_class_ranking_metrics_undefined_not_invented():
    metrics, _ = models.evaluate_probabilities([0, 0], [.2, .8])
    assert np.isnan(metrics["pr_auc"]) and np.isnan(metrics["roc_auc"]) and np.isnan(metrics["recall"])
    assert metrics["brier_score"] == pytest.approx(.34)
    with pytest.raises(ValueError, match="fixed at 0.5"):
        models.evaluate_probabilities([0, 1], [.2, .8], threshold=.3)


def prediction(uid, origin, probability=.9, *, k=3, label=1):
    return dict(municipality_id=uid, forecast_origin=origin, probability=probability,
                fully_known=True, at_risk=True, label=label, k=k, model="synthetic")


def test_event_matching_earliest_alert_repeats_late_and_false_exposure():
    cases = pd.DataFrame([prediction("a", "2024-01-31"), prediction("a", "2024-02-29"),
                          prediction("a", "2024-03-31", label=0), prediction("a", "2024-04-30", label=0),
                          prediction("b", "2024-01-31", probability=.1)])
    events = pd.DataFrame([dict(municipality_id="a", event_id="synthetic_a", onset_period="2024-03"),
                           dict(municipality_id="b", event_id="synthetic_b", onset_period="2024-03")])
    metrics, matched, alerts = models.evaluate_event_alerts(cases, events, 3)
    assert metrics["eligible_events"] == 2 and metrics["warned_events"] == 1
    assert metrics["event_recall"] == .5 and metrics["median_lead_time_months"] == 2
    assert metrics["raw_alert_count"] == 4 and metrics["repeated_alert_count"] == 1
    assert metrics["false_alert_count"] == 2 and metrics["alert_precision"] == pytest.approx(1 / 3)
    assert metrics["false_alerts_per_municipality_year"] == pytest.approx(2 * 12 / 5)
    a = matched.set_index("event_id").loc["synthetic_a"]
    assert a.earliest_alert_origin == "2024-01-31" and a.warning_count_before_event == 2
    assert a.successful_warning_count == 1
    assert alerts.successful_warning.sum() == 1
    assert alerts.status.tolist()[2:] == ["late_or_at_onset_false_alert", "late_or_at_onset_false_alert"]


def test_chronological_one_to_one_two_events_and_deterministic_shuffling():
    cases = pd.DataFrame([prediction("a", "2024-01-31"), prediction("a", "2024-02-29"),
                          prediction("a", "2024-03-31")])
    events = pd.DataFrame([dict(municipality_id="a", event_id="synthetic_a1", onset_period="2024-03"),
                           dict(municipality_id="a", event_id="synthetic_a2", onset_period="2024-04")])
    first = models.evaluate_event_alerts(cases, events, 3)
    second = models.evaluate_event_alerts(cases.iloc[::-1], events.iloc[::-1], 3)
    assert first[0] == second[0]
    pd.testing.assert_frame_equal(first[1], second[1])
    pd.testing.assert_frame_equal(first[2], second[2])
    assert first[2].successful_warning.sum() == 2 and first[2].repeated_warning.sum() == 1
    assert first[2].loc[first[2].successful_warning, "event_id"].is_unique


def test_k1_alert_two_months_before_event_does_not_warn():
    cases = pd.DataFrame([prediction("a", "2024-01-31", k=1, label=0),
                          prediction("a", "2024-02-29", probability=.1, k=1)])
    events = pd.DataFrame([dict(municipality_id="a", event_id="synthetic_a", onset_period="2024-03")])
    metrics, matched, alerts = models.evaluate_event_alerts(cases, events, 1)
    assert metrics["event_recall"] == 0 and not matched.warning_observed.any()
    assert alerts.false_alert.all()


def test_censored_and_active_rows_not_monitored_exposure():
    cases = pd.DataFrame([prediction("a", "2024-01-31"), prediction("a", "2024-02-29"), prediction("a", "2024-03-31")])
    cases.loc[1, "fully_known"] = False
    cases.loc[1, "label"] = np.nan
    cases.loc[2, "at_risk"] = False
    events = pd.DataFrame([dict(municipality_id="a", event_id="synthetic_a", onset_period="2024-03")])
    assert models.evaluate_event_alerts(cases, events, 3)[0]["monitored_cases"] == 1


def test_origin_ineligible_predictions_do_not_add_alerts_or_monitored_exposure():
    cases = pd.DataFrame([prediction("a", "2024-01-31"), prediction("a", "2024-02-29")])
    cases["eligible_at_origin"] = [False, True]
    events = pd.DataFrame([dict(municipality_id="a", event_id="synthetic_a", onset_period="2024-03")])
    metrics, matched, alerts = models.evaluate_event_alerts(cases, events, 3)
    assert metrics["monitored_cases"] == metrics["raw_alert_count"] == 1
    assert matched.earliest_alert_origin.tolist() == ["2024-02-29"]
    assert alerts.forecast_origin.tolist() == ["2024-02-29"]


def test_conditional_nested_models_fixed_parameters_identical_cases_and_test_label_immutability(monkeypatch):
    train, test, events, features = synthetic_gate_data()
    monkeypatch.setattr(models, "_logistic_regression_class", lambda: SyntheticClassifier)
    SyntheticClassifier.calls = []
    original_train, original_features = train.copy(deep=True), features.copy(deep=True)
    first = models.fit_baselines(train, test, features, events, GROUPS, 1)
    assert first["fit_count"] == 5 and len(SyntheticClassifier.calls) == 4
    for _, _, kwargs in SyntheticClassifier.calls:
        assert kwargs == dict(C=1.0, class_weight=None, random_state=42, solver="lbfgs", max_iter=2000, n_jobs=1)
    for _, subset in first["predictions"].groupby("model", sort=False):
        assert subset[list(models.KEYS)].reset_index(drop=True).equals(test[list(models.KEYS)].reset_index(drop=True))
    changed = test.copy()
    changed_events = events.copy()
    for origin, group in changed.groupby("forecast_origin"):
        positive = group.index[group.municipality_id.eq("0")][0]
        negative = group.index[group.municipality_id.eq("5")][0]
        original_id = changed.loc[positive, "positive_event_ids"]
        new_id = original_id.replace("synthetic_0_", "synthetic_5_")
        changed.loc[positive, ["label", "positive_event_ids"]] = [0, ""]
        changed.loc[negative, ["label", "positive_event_ids"]] = [1, new_id]
        event_row = changed_events.index[changed_events.event_id.eq(original_id)][0]
        changed_events.loc[event_row, ["municipality_id", "event_id"]] = ["5", new_id]
    second = models.fit_baselines(train, changed, features, changed_events, GROUPS, 1)
    pd.testing.assert_frame_equal(first["coefficients"], second["coefficients"])
    np.testing.assert_array_equal(first["predictions"].probability, second["predictions"].probability)
    pd.testing.assert_frame_equal(train, original_train)
    pd.testing.assert_frame_equal(features, original_features)
    third = models.fit_baselines(train, test, features, events, copy.deepcopy(GROUPS), 1)
    pd.testing.assert_frame_equal(first["predictions"], third["predictions"])
    pd.testing.assert_frame_equal(first["coefficients"], third["coefficients"])


def test_ABCD_group_aliases_preserve_nested_models(monkeypatch):
    train, test, events, features = synthetic_gate_data()
    monkeypatch.setattr(models, "_logistic_regression_class", lambda: SyntheticClassifier)
    aliases = {key: GROUPS[name] for key, name in zip("ABCD", ["history", "detector", "macro", "news"])}
    result = models.fit_baselines(train, test, features, events, aliases, 1)
    assert result["fit_count"] == 5
    assert set(result["coefficients"].group) == {"A", "B", "C", "D", "constant", "intercept"}


def test_train_constant_or_missing_news_never_selected_from_varying_test(monkeypatch):
    train, test, events, features = synthetic_gate_data()
    monkeypatch.setattr(models, "_logistic_regression_class", lambda: SyntheticClassifier)
    is_train = features.forecast_origin.isin(train.forecast_origin)
    features["news_constant"] = np.where(is_train, 2., np.arange(len(features), dtype=float))
    features["news_absent"] = np.where(is_train, np.nan, np.arange(len(features), dtype=float) ** 2)
    groups = dict(GROUPS, news=["news_constant", "news_absent"])
    result = models.fit_baselines(train, test, features, events, groups, 1)
    assert result["gate"]["passed"] and result["fit_count"] == 5
    audit = result["feature_filter"]
    news_audit = audit.loc[audit.model.eq("B4") & audit.group.eq("news")]
    assert not news_audit.selected.any()
    assert set(news_audit.drop_reason) == {"constant_after_training_median_imputation", "all_missing_on_train"}
    b3 = audit.loc[audit.model.eq("B3") & audit.selected, "feature"].tolist()
    b4 = audit.loc[audit.model.eq("B4") & audit.selected, "feature"].tolist()
    assert b3 == b4
    predictions = result["predictions"]
    np.testing.assert_array_equal(predictions.loc[predictions.model.eq("B3"), "probability"].to_numpy(),
                                  predictions.loc[predictions.model.eq("B4"), "probability"].to_numpy())
