"""Synthetic offline checks of E05c paired regions and macro coverage."""
import numpy as np
import pandas as pd
import pytest

from sberforecast.metrics import KEY
from sberforecast.macro_forecast_evaluation import MODELS, evaluate_macro


def examples():
    expected = pd.DataFrame([
        ("1", "2023-12-31", "2024-12-01", 12, 20.0, "holdout"),
        ("2", "2023-12-31", "2024-12-01", 12, np.nan, "holdout"),
        ("1", "2024-06-30", "2024-07-01", 1, 10.0, "holdout"),
        ("2", "2024-06-30", "2024-07-01", 1, 30.0, "holdout"),
        ("1", "2024-05-31", "2024-06-01", 1, 8.0, "validation"),
        ("1", "2024-07-31", "2024-08-01", 1, 12.0, "holdout"),
    ], columns=KEY + ["y_true", "split"])
    expected["history_cutoff"] = expected.forecast_origin
    expected["availability_assumption"] = "month_end_plus_0_months"
    frames = []
    for model, error in zip(MODELS, [1, 2, 3, 4, 5, 6]):
        frame = expected.copy()
        frame["model"] = model
        frame["y_pred"] = frame.y_true.fillna(20) - error
        frame["status"] = "native"
        frame["effective_model"] = model
        frame["reason"] = ""
        frame["macro_features_available"] = True if model in ("M1", "M2") else np.nan
        if model in ("M0", "M1", "M2"):
            annual = frame.horizon.eq(12)
            frame.loc[annual, "status"] = "fallback_no_training_pairs"
            frame.loc[annual, "effective_model"] = "SeasonalNaive"
            frame.loc[annual, "reason"] = "no_training_pairs"
        frames.append(frame)
    return pd.concat(frames, ignore_index=True), expected


def selected(frame, model, municipality="2", origin="2024-06-30"):
    mask = frame.model.eq(model) & frame.municipality_id.eq(municipality) & frame.forecast_origin.eq(origin)
    assert mask.sum() == 1
    return mask


def record(table, model, horizon=1, split="holdout"):
    return table.loc[table.model.eq(model) & table.horizon.eq(horizon) & table.split.eq(split)].iloc[0]


def test_macro_unavailable_does_not_remove_strategy_or_native_cases():
    forecasts, expected = examples()
    forecasts.loc[selected(forecasts, "M2"), "macro_features_available"] = False
    tables = evaluate_macro(forecasts, expected)
    assert record(tables["metrics_strategy"], "M2").n_predictions == 3
    assert record(tables["metrics_native"], "M2").n_predictions == 3
    macro = tables["metrics_macro_available"].query("split == 'holdout' and horizon == 1")
    assert macro.n_predictions.eq(2).all()
    assert set(macro.model) == set(MODELS)
    excluded = tables["excluded_macro_available_keys"].query("horizon == 1").iloc[0]
    assert excluded.municipality_id == "2" and excluded.reason == "M2:missing_macro_forecast_for_target_year"


def test_macro_region_is_intersection_on_actual_keys():
    forecasts, expected = examples()
    forecasts.loc[selected(forecasts, "M1"), "macro_features_available"] = False
    forecasts.loc[selected(forecasts, "M2", "1", "2024-07-31"), "macro_features_available"] = False
    tables = evaluate_macro(forecasts, expected)
    keys = tables["macro_available_keys"].query("split == 'holdout' and horizon == 1")
    assert len(keys) == 1
    assert keys.iloc[0].municipality_id == "1" and keys.iloc[0].forecast_origin == "2024-06-30"
    assert tables["metrics_macro_available"].query("split == 'holdout' and horizon == 1").n_predictions.eq(1).all()
    assert len(tables["macro_available_keys"]) + len(tables["excluded_macro_available_keys"]) == len(tables["strategy_keys"])


def test_control_is_recomputed_on_macro_keys_not_its_own_support():
    forecasts, expected = examples()
    forecasts.loc[selected(forecasts, "M0"), "y_pred"] = 0.0
    forecasts.loc[selected(forecasts, "M2"), "macro_features_available"] = False
    tables = evaluate_macro(forecasts, expected)
    assert record(tables["metrics_strategy"], "M0").mae_macro == 15.5
    assert record(tables["metrics_macro_available"], "M0").mae_macro == 1.0


def test_failed_is_preserved_and_delta_is_not_reported_as_complete():
    forecasts, expected = examples()
    mask = selected(forecasts, "M1")
    forecasts.loc[mask, ["status", "effective_model", "reason", "y_pred"]] = ["failed", "none", "fit_error", np.nan]
    tables = evaluate_macro(forecasts, expected)
    full = record(tables["metrics_strategy"], "M1")
    assert full.n_predictions == 3 and full.n_failed == 1
    assert full.metric_status == "incomplete_failed" and np.isnan(full.mae_macro)
    assert record(tables["metrics_macro_available"], "M0").n_predictions == 2
    explanation = tables["excluded_macro_available_keys"].query("horizon == 1").iloc[0].reason
    assert "M1:failed (fit_error)" in explanation
    delta = tables["metric_deltas"].query("scope == 'strategy' and split == 'holdout' and horizon == 1")
    assert delta.metric_status.eq("incomplete_failed").all() and delta.delta_mae_macro.isna().all()


def test_fallback_with_available_macro_data_stays_outside_native_and_C():
    forecasts, expected = examples()
    tables = evaluate_macro(forecasts, expected)
    for scope in ("native", "macro_available"):
        annual = tables[f"metrics_{scope}"].query("horizon == 12")
        assert annual.n_predictions.eq(0).all() and annual.mae_macro.isna().all()
    assert tables["metrics_macro_available"].query("horizon == 12").metric_status.eq("no_macro_available_native_forecasts").all()
    coverage = record(tables["macro_coverage_by_model"], "M2", horizon=12)
    assert coverage.n_requested == 2 and coverage.n_evaluable == 1
    assert coverage.n_macro_available == 2 and coverage.n_fallback_with_macro == 2
    assert coverage.n_native_with_macro == 0
    assert record(tables["metrics_strategy"], "M2", horizon=12).mae_macro == 3.0


def test_empty_macro_region_preserves_zero_metric_groups():
    forecasts, expected = examples()
    forecasts.loc[forecasts.model.isin(["M1", "M2"]), "macro_features_available"] = False
    tables = evaluate_macro(forecasts, expected)
    assert tables["macro_available_keys"].empty
    assert tables["metrics_macro_available"].metric_status.eq("no_macro_available_native_forecasts").all()
    assert tables["metrics_macro_available_by_origin"].n_predictions.eq(0).all()
    assert tables["metrics_macro_available_by_municipality"].empty
    assert len(tables["excluded_macro_available_keys"]) == len(tables["strategy_keys"])


def test_raw_and_evaluable_macro_counts_remain_distinct():
    forecasts, expected = examples()
    forecasts.loc[selected(forecasts, "M1"), "macro_features_available"] = False
    tables = evaluate_macro(forecasts, expected)
    coverage = record(tables["macro_coverage_by_model"], "M1")
    assert coverage.n_requested == coverage.n_evaluable == 3
    assert coverage.n_native_with_macro == 2 and coverage.n_native_without_macro == 1
    assert coverage.macro_available_share == pytest.approx(2 / 3)
    annual = record(tables["macro_coverage_by_model"], "M1", horizon=12)
    assert annual.n_requested == 2 and annual.n_evaluable == 1
    control = record(tables["macro_coverage_by_model"], "M0")
    assert not control.macro_applicable and control.n_macro_missing == control.n_native_without_macro == 0
    assert np.isnan(control.macro_available_share)


def test_sequential_differences_share_keys_and_signed_direction():
    forecasts, expected = examples()
    tables = evaluate_macro(forecasts, expected)
    differences = tables["metric_deltas"]
    assert set(differences.comparison) == {"M1-M0", "M2-M1"}
    assert set(differences.scope) == {"strategy", "native", "macro_available"}
    usable = differences.query("horizon == 1")
    assert usable.delta_mae_macro.eq(1).all() and usable.delta_mae_micro.eq(1).all()
    assert usable.loc[usable.split.eq("holdout"), "delta_r2_pooled"].lt(0).all()
    assert usable.loc[usable.split.eq("validation"), "delta_r2_pooled"].isna().all()
    annual = differences.query("horizon == 12 and scope != 'strategy'")
    assert annual.delta_mae_macro.isna().all() and annual.n_predictions.eq(0).all()


def test_macro_scope_contains_origin_and_municipality_metrics():
    forecasts, expected = examples()
    tables = evaluate_macro(forecasts, expected)
    by_mo = tables["metrics_macro_available_by_municipality"].query("model == 'M2' and split == 'holdout' and horizon == 1")
    assert by_mo.set_index("municipality_id").mae.to_dict() == {"1": 3.0, "2": 3.0}
    by_origin = tables["metrics_macro_available_by_origin"].query("model == 'M0' and split == 'holdout' and horizon == 1")
    assert by_origin.set_index("forecast_origin").n_predictions.to_dict() == {"2024-06-30": 2, "2024-07-31": 1}


@pytest.mark.parametrize("value", [np.nan, "False", "True", "0", 2, -1])
def test_ambiguous_macro_flags_are_rejected(value):
    forecasts, expected = examples()
    forecasts["macro_features_available"] = forecasts.macro_features_available.astype(object)
    forecasts.loc[selected(forecasts, "M1"), "macro_features_available"] = value
    with pytest.raises(ValueError, match="bool либо 0/1"):
        evaluate_macro(forecasts, expected)


def test_missing_macro_flag_column_is_rejected():
    forecasts, expected = examples()
    with pytest.raises(ValueError, match="явный macro_features_available"):
        evaluate_macro(forecasts.drop(columns="macro_features_available"), expected)


def test_all_six_models_must_retain_same_keys():
    forecasts, expected = examples()
    forecasts.loc[selected(forecasts, "M2"), "municipality_id"] = "other"
    with pytest.raises(ValueError, match="Несовпадение ключей"):
        evaluate_macro(forecasts, expected)


def test_reference_failed_does_not_shrink_strategy_silently():
    forecasts, expected = examples()
    mask = selected(forecasts, "ProphetAuto")
    forecasts.loc[mask, ["status", "effective_model", "reason", "y_pred"]] = ["failed", "none", "predict_error", np.nan]
    tables = evaluate_macro(forecasts, expected)
    assert record(tables["metrics_strategy"], "ProphetAuto").n_predictions == 3
    assert record(tables["metrics_macro_available"], "M0").n_predictions == 2
    assert "ProphetAuto:failed" in tables["excluded_macro_available_keys"].query("horizon == 1").iloc[0].reason


def test_evaluation_does_not_mutate_forecasts_or_control():
    forecasts, expected = examples()
    before, expected_before = forecasts.copy(deep=True), expected.copy(deep=True)
    evaluate_macro(forecasts, expected)
    pd.testing.assert_frame_equal(forecasts, before)
    pd.testing.assert_frame_equal(expected, expected_before)
