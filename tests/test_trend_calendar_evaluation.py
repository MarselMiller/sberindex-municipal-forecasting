"""Синтетические проверки неизменной области и двух оценок E05b."""
import numpy as np
import pandas as pd
import pytest

from sberforecast.metrics import KEY
from sberforecast.trend_calendar_evaluation import evaluate_group


MODELS = ["Baseline", "VariantA", "VariantB"]
NATIVE = MODELS[1:]


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
    for model, error in zip(MODELS, [1.0, 2.0, 3.0]):
        frame = expected.copy()
        frame["model"] = model
        frame["y_pred"] = frame.y_true.fillna(20.0) - error
        frame["status"] = "native"
        frame["effective_model"] = model
        frame["reason"] = ""
        if model in NATIVE:
            annual = frame.horizon.eq(12)
            frame.loc[annual, "status"] = "fallback_insufficient_history"
            frame.loc[annual, "effective_model"] = "SeasonalNaiveYoY"
            frame.loc[annual, "reason"] = "insufficient_annual_pairs"
        frames.append(frame)
    return pd.concat(frames, ignore_index=True), expected


def change_case(frame, model, municipality_id, origin, *, failed=False):
    selected = (frame.model.eq(model) & frame.municipality_id.eq(municipality_id) &
                frame.forecast_origin.eq(origin))
    assert selected.sum() == 1
    frame.loc[selected, "status"] = "failed" if failed else "fallback_missing_template"
    frame.loc[selected, "effective_model"] = "none" if failed else "SeasonalNaiveYoY"
    frame.loc[selected, "reason"] = "execution_error" if failed else "incomplete_seasonal_template"
    if failed:
        frame.loc[selected, "y_pred"] = np.nan


def row(table, model, *, horizon=1, split="holdout"):
    return table.loc[table.model.eq(model) & table.horizon.eq(horizon) & table.split.eq(split)].iloc[0]


def test_same_counts_with_different_keys_are_rejected():
    forecasts, expected = examples()
    forecasts.loc[forecasts.model.eq("VariantA") & (forecasts.index == 6), "municipality_id"] = "other"
    with pytest.raises(ValueError, match="Несовпадение ключей"):
        evaluate_group(forecasts, expected, MODELS, NATIVE)


def test_duplicate_keys_are_rejected():
    forecasts, expected = examples()
    forecasts = pd.concat([forecasts, forecasts.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="повторяющиеся"):
        evaluate_group(forecasts, expected, MODELS, NATIVE)


def test_truth_mismatch_is_rejected_including_missing_truth():
    forecasts, expected = examples()
    forecasts.loc[forecasts.model.eq("VariantA") & forecasts.municipality_id.eq("2") &
                  forecasts.horizon.eq(12), "y_true"] = 20.0
    with pytest.raises(ValueError, match="Целевые факты"):
        evaluate_group(forecasts, expected, MODELS, NATIVE)


@pytest.mark.parametrize("field,value", [
    ("history_cutoff", "2024-08-31"), ("split", "validation"),
    ("availability_assumption", "month_end_plus_1_months"),
])
def test_protocol_field_mismatch_is_rejected(field, value):
    forecasts, expected = examples()
    forecasts.loc[forecasts.model.eq("VariantA") & forecasts.forecast_origin.eq("2024-06-30"), field] = value
    with pytest.raises(ValueError, match=f"Поле {field}"):
        evaluate_group(forecasts, expected, MODELS, NATIVE)


def test_failed_is_retained_in_full_strategy_and_explained_in_native_exclusions():
    forecasts, expected = examples()
    change_case(forecasts, "VariantA", "2", "2024-06-30", failed=True)
    result = evaluate_group(forecasts, expected, MODELS, NATIVE)
    full = row(result["metrics_strategy"], "VariantA")
    assert full.n_predictions == 3 and full.n_failed == 1
    assert full.metric_status == "incomplete_failed" and np.isnan(full.mae_macro)
    assert row(result["metrics_strategy"], "Baseline").n_predictions == 3
    native = result["metrics_native"].query("split == 'holdout' and horizon == 1")
    assert native.n_predictions.eq(2).all()
    excluded = result["excluded_native_keys"].query("horizon == 1").iloc[0]
    assert excluded.municipality_id == "2" and "VariantA:failed (execution_error)" in excluded.reason
    mo = result["metrics_strategy_by_municipality"]
    failed_mo = mo.query("model == 'VariantA' and municipality_id == '2' and horizon == 1").iloc[0]
    assert failed_mo.n == 1 and np.isnan(failed_mo.mae) and failed_mo.metric_status == "incomplete_failed"


def test_fallback_is_scored_in_full_strategy_but_excluded_from_common_native():
    forecasts, expected = examples()
    change_case(forecasts, "VariantA", "2", "2024-06-30")
    result = evaluate_group(forecasts, expected, MODELS, NATIVE)
    assert row(result["metrics_strategy"], "VariantA").mae_macro == 2.0
    assert row(result["metrics_strategy"], "VariantA").n_predictions == 3
    assert row(result["metrics_native"], "Baseline").n_predictions == 2
    exclusions = result["excluded_native_keys"].query("horizon == 1")
    assert len(exclusions) == 1 and "fallback_missing_template" in exclusions.iloc[0].reason
    coverage = row(result["coverage_evaluable"], "VariantA")
    assert coverage.n_requested == 3 and coverage.n_native == 2 and coverage.n_fallback == 1


def test_intersection_uses_exact_keys_and_keeps_individual_native_coverage():
    forecasts, expected = examples()
    change_case(forecasts, "VariantA", "2", "2024-06-30")
    change_case(forecasts, "VariantB", "2", "2024-06-30")
    change_case(forecasts, "VariantB", "1", "2024-07-31")
    result = evaluate_group(forecasts, expected, MODELS, NATIVE)
    common = result["common_native_keys"].query("split == 'holdout' and horizon == 1")
    assert len(common) == 1
    assert common.iloc[0].municipality_id == "1" and common.iloc[0].forecast_origin == "2024-06-30"
    coverage = result["native_coverage_by_model"]
    assert row(coverage, "VariantA").n_native_evaluable == 2
    assert row(coverage, "VariantB").n_native_evaluable == 1
    assert row(coverage, "Baseline").n_common_native_evaluable == 1
    own = result["own_native_keys"].query("model == 'VariantA' and split == 'holdout' and horizon == 1")
    assert len(own) == 2
    assert result["metrics_native"].query("split == 'holdout' and horizon == 1").n_predictions.eq(1).all()


def test_annual_reserve_never_becomes_native_mae_even_for_reference():
    forecasts, expected = examples()
    result = evaluate_group(forecasts, expected, MODELS, NATIVE)
    native = result["metrics_native"].query("horizon == 12")
    assert set(native.model) == set(MODELS)
    assert native.metric_status.eq("no_native_forecasts").all()
    assert native.n_predictions.eq(0).all() and native.mae_macro.isna().all()
    assert result["metrics_native_by_origin"].query("horizon == 12").mae_micro.isna().all()
    assert row(result["metrics_strategy"], "VariantA", horizon=12).mae_macro == 2.0
    assert result["excluded_native_keys"].query("horizon == 12").shape[0] == 1


def test_raw_and_evaluable_coverage_preserve_missing_actual_cases():
    forecasts, expected = examples()
    result = evaluate_group(forecasts, expected, MODELS, NATIVE)
    raw = row(result["coverage"], "VariantA", horizon=12)
    evaluable = row(result["coverage_evaluable"], "VariantA", horizon=12)
    assert raw.n_requested == 2 and raw.n_actual_available == 1 and raw.n_fallback == 2
    assert evaluable.n_requested == 1 and evaluable.n_fallback == 1
    assert len(result["strategy_keys"]) == 5
    assert len(result["common_native_keys"]) + len(result["excluded_native_keys"]) == 5


def test_reference_internal_fallback_does_not_reduce_variants_native_intersection():
    forecasts, expected = examples()
    change_case(forecasts, "Baseline", "2", "2024-06-30")
    result = evaluate_group(forecasts, expected, MODELS, NATIVE)
    assert row(result["metrics_native"], "VariantA").n_predictions == 3
    assert row(result["metrics_native"], "Baseline").n_fallback == 1


def test_failed_reference_does_reduce_common_keys_without_shrinking_full_strategy():
    forecasts, expected = examples()
    change_case(forecasts, "Baseline", "2", "2024-06-30", failed=True)
    result = evaluate_group(forecasts, expected, MODELS, NATIVE)
    assert row(result["metrics_strategy"], "Baseline").n_predictions == 3
    assert row(result["metrics_native"], "VariantA").n_predictions == 2
    assert "Baseline:failed" in result["excluded_native_keys"].query("horizon == 1").iloc[0].reason


def test_macro_micro_r2_and_municipality_date_metrics_match_observations():
    forecasts, expected = examples()
    selected = forecasts.model.eq("VariantA") & forecasts.horizon.eq(1) & forecasts.split.eq("holdout")
    forecasts.loc[selected, "y_pred"] = [9.0, 20.0, 9.0]
    result = evaluate_group(forecasts, expected, MODELS, NATIVE)
    summary = row(result["metrics_strategy"], "VariantA")
    assert summary.mae_macro == 6.0  # MO1: (1+3)/2=2; MO2: 10; average 6.
    assert summary.mae_micro == pytest.approx(14.0 / 3)
    truth = np.array([10.0, 30.0, 12.0])
    assert summary.r2_pooled == pytest.approx(1 - 110.0 / np.sum((truth - truth.mean()) ** 2))
    per_mo = result["metrics_strategy_by_municipality"].query("model == 'VariantA' and split == 'holdout' and horizon == 1")
    assert per_mo.set_index("municipality_id").mae.to_dict() == {"1": 2.0, "2": 10.0}
    per_origin = result["metrics_strategy_by_origin"].query("model == 'VariantA' and horizon == 1 and split == 'holdout'")
    assert per_origin.set_index("forecast_origin").mae_macro.to_dict() == {"2024-06-30": 5.5, "2024-07-31": 3.0}


def test_only_missing_actuals_yield_explicit_zero_coverage_and_no_metric():
    forecasts, expected = examples()
    expected["y_true"] = np.nan
    forecasts["y_true"] = np.nan
    result = evaluate_group(forecasts, expected, MODELS, NATIVE)
    assert result["metrics_strategy"].metric_status.eq("no_evaluable_cases").all()
    assert result["coverage_evaluable"].n_requested.eq(0).all()
    assert result["native_coverage_by_model"].n_evaluable.eq(0).all()
    assert result["native_coverage_by_model"].native_share_evaluable.isna().all()


@pytest.mark.parametrize("bad_field,bad_value,message", [
    ("status", "unknown", "Неизвестный статус"),
    ("y_pred", np.nan, "Конечность"),
    ("effective_model", "", "effective_model"),
])
def test_invalid_success_status_is_rejected(bad_field, bad_value, message):
    forecasts, expected = examples()
    forecasts.loc[forecasts.index == 0, bad_field] = bad_value
    with pytest.raises(ValueError, match=message):
        evaluate_group(forecasts, expected, MODELS, NATIVE)


def test_failed_requires_nan_and_reason_and_does_not_allow_silent_reserve():
    forecasts, expected = examples()
    change_case(forecasts, "VariantA", "2", "2024-06-30", failed=True)
    selected = forecasts.model.eq("VariantA") & forecasts.municipality_id.eq("2") & forecasts.horizon.eq(1)
    forecasts.loc[selected, "reason"] = ""
    with pytest.raises(ValueError, match="причина"):
        evaluate_group(forecasts, expected, MODELS, NATIVE)
    forecasts.loc[selected, "reason"] = "failed"
    forecasts.loc[selected, "y_pred"] = 0.0
    with pytest.raises(ValueError, match="Конечность"):
        evaluate_group(forecasts, expected, MODELS, NATIVE)


def test_evaluation_does_not_mutate_input_frames():
    forecasts, expected = examples()
    before, before_expected = forecasts.copy(deep=True), expected.copy(deep=True)
    evaluate_group(forecasts, expected, MODELS, NATIVE)
    pd.testing.assert_frame_equal(forecasts, before)
    pd.testing.assert_frame_equal(expected, before_expected)


def test_invalid_declared_model_lists_are_rejected():
    forecasts, expected = examples()
    with pytest.raises(ValueError, match="разных моделей"):
        evaluate_group(forecasts, expected, ["Baseline", "Baseline"], ["Baseline"])
    with pytest.raises(ValueError, match="подмножеством"):
        evaluate_group(forecasts, expected, MODELS, ["Unknown"])
    with pytest.raises(ValueError, match="Состав моделей"):
        evaluate_group(forecasts, expected, ["Baseline", "VariantA"], ["VariantA"])
