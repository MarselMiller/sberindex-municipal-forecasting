"""Синтетические проверки E03, без загрузки checkpoint и импорта CatBoost."""
import numpy as np
import pandas as pd
import pytest

from sberforecast.chronos_evaluation import (
    DIRECT, E01_MODELS, MODEL, evaluate_chronos, require_same_keys, validate_predictions,
)
from sberforecast.metrics import KEY, compute_metrics


def examples():
    current = pd.DataFrame([
        dict(municipality_id="1", forecast_origin="2024-06-30", target_period="2024-07-01", horizon=1, y_true=10., y_pred=9.),
        dict(municipality_id="2", forecast_origin="2024-06-30", target_period="2024-07-01", horizon=1, y_true=30., y_pred=24.),
        dict(municipality_id="1", forecast_origin="2023-12-31", target_period="2024-12-01", horizon=12, y_true=20., y_pred=18.),
        dict(municipality_id="2", forecast_origin="2023-12-31", target_period="2024-12-01", horizon=12, y_true=40., y_pred=37.),
        dict(municipality_id="1471", forecast_origin="2023-12-31", target_period="2024-12-01", horizon=12, y_true=np.nan, y_pred=37.),
    ]).assign(model=MODEL, status="native", effective_model=MODEL, reason="", split="holdout",
              availability_assumption="month_end_plus_0_months")
    current["history_cutoff"] = current.forecast_origin
    e01 = pd.concat([current.assign(model=name, y_pred=current.y_true.fillna(21.)) for name in E01_MODELS], ignore_index=True)
    e02 = current.assign(model="CatBoostDirect+SeasonalNaive", effective_model=DIRECT, y_pred=current.y_true.fillna(21.) + 3)
    e02.loc[e02.horizon.eq(12), ["status", "effective_model", "reason"]] = ["fallback_no_training_pairs", "SeasonalNaive", "no_training_pairs"]
    return current, e01, e02


def metric(result, scope, model, horizon):
    return result[f"metrics_{scope}"].loc[lambda x: x.model.eq(model) & x.horizon.eq(horizon)].iloc[0]


def test_full_support_and_annual_comparison_keep_chronos_and_exclude_reserve():
    current, e01, e02 = examples()
    result = evaluate_chronos(current, e01, e02)
    assert len(result["evaluation_keys_full"]) == 4
    assert len(result["evaluation_keys_common_success"]) == 4
    assert result["excluded_common_success_keys"].empty
    for scope in ("full", "common_success"):
        annual = metric(result, scope, DIRECT, 12)
        assert annual.metric_status == "no_training_pairs" and np.isnan(annual.mae_macro)
        assert annual.n_success == 0
        assert metric(result, scope, MODEL, 12).mae_macro == 2.5
        assert metric(result, scope, "ProphetAuto", 12).n_predictions == 2
    annual_scope = result["comparison_coverage"].query("horizon == 12").iloc[0]
    assert DIRECT not in annual_scope.participants and MODEL in annual_scope.participants
    pure_coverage = result["coverage"].query("model == @DIRECT and horizon == 12").iloc[0]
    assert pure_coverage.n_requested == 3 and pure_coverage.n_e01_cases == 2
    assert pure_coverage.n_unavailable == 3 and pure_coverage.n_fallback == 0


def test_one_failed_is_visible_in_full_and_all_common_metrics_use_identical_keys():
    current, e01, e02 = examples()
    current.loc[current.municipality_id.eq("2") & current.horizon.eq(1), ["y_pred", "status", "reason"]] = [np.nan, "failed", "inference exception"]
    result = evaluate_chronos(current, e01, e02)
    full = metric(result, "full", MODEL, 1)
    assert full.n_predictions == 2 and full.n_failed == 1
    assert full.metric_status == "incomplete_failed" and np.isnan(full.mae_macro)
    assert metric(result, "full", "ProphetAuto", 1).n_predictions == 2
    shared = result["metrics_common_success"].query("horizon == 1")
    assert shared.n_predictions.eq(1).all()
    assert metric(result, "common_success", MODEL, 1).mae_macro == 1
    exclusion = result["excluded_common_success_keys"].iloc[0]
    assert exclusion.municipality_id == "2" and "Chronos-2:failed" in exclusion.reason
    coverage = result["comparison_coverage"].query("horizon == 1").iloc[0]
    assert coverage.n_full == 2 and coverage.n_common_success == 1 and coverage.incomplete


def test_all_failed_does_not_remove_chronos_from_participants():
    current, e01, e02 = examples()
    current.loc[:, ["y_pred", "status", "reason"]] = [np.nan, "failed", "load failed"]
    result = evaluate_chronos(current, e01, e02)
    assert result["evaluation_keys_common_success"].empty
    assert result["comparison_coverage"].participants.str.contains(MODEL, regex=False).all()
    assert metric(result, "common_success", DIRECT, 12).metric_status == "no_training_pairs"
    assert metric(result, "common_success", "ProphetAuto", 12).metric_status == "no_common_success_keys"
    assert len(result["excluded_common_success_keys"]) == 4


def test_truth_missing_stays_in_raw_coverage_and_failed_missing_truth_does_not_invalidate_metrics():
    current, e01, e02 = examples()
    current.loc[current.municipality_id.eq("1471"), ["y_pred", "status", "reason"]] = [np.nan, "failed", "inference failure"]
    result = evaluate_chronos(current, e01, e02)
    annual = metric(result, "full", MODEL, 12)
    assert annual.metric_status == "complete" and annual.n_predictions == 2
    raw = result["coverage"].query("model == @MODEL and horizon == 12").iloc[0]
    assert raw.n_requested == 3 and raw.n_actual_available == 2 and raw.n_failed == 1
    assert "1471" not in set(result["evaluation_keys_full"].municipality_id)


def test_equal_counts_with_different_keys_and_duplicates_are_rejected():
    current, e01, e02 = examples()
    changed = current.copy()
    changed.loc[0, "municipality_id"] = "other"
    with pytest.raises(ValueError, match="Несовпадение ключей"):
        evaluate_chronos(changed, e01, e02)
    with pytest.raises(ValueError, match="повторяющиеся"):
        require_same_keys(pd.concat([current, current.iloc[:1]]), current)
    with pytest.raises(ValueError, match="Несовпадение ключей"):
        evaluate_chronos(current.iloc[1:], e01, e02)


@pytest.mark.parametrize("field,value", [
    ("y_true", 123.), ("history_cutoff", "2020-01-31"),
    ("split", "validation"), ("availability_assumption", "month_end_plus_1_months"),
])
def test_changed_facts_or_protocol_are_rejected(field, value):
    current, e01, e02 = examples()
    e02.loc[0, field] = value
    with pytest.raises(ValueError):
        evaluate_chronos(current, e01, e02)


def test_nan_fact_mismatch_is_rejected():
    current, e01, e02 = examples()
    current.loc[current.municipality_id.eq("1471"), "y_true"] = 22.
    with pytest.raises(ValueError, match="Целевые факты"):
        evaluate_chronos(current, e01, e02)


@pytest.mark.parametrize("status,pred,reason", [
    ("fallback_last_value", 10., "reserve"), ("native", np.nan, ""),
    ("native", np.inf, ""), ("failed", np.inf, "error"),
    ("failed", 10., "error"), ("failed", np.nan, ""),
])
def test_invalid_chronos_status_value_or_reason_is_rejected(status, pred, reason):
    current, e01, _ = examples()
    current.loc[0, ["status", "y_pred", "reason"]] = [status, pred, reason]
    with pytest.raises(ValueError):
        validate_predictions(current, e01.loc[e01.model.eq(E01_MODELS[0])])


def test_unexpected_nonannual_no_pairs_is_not_removed_from_participants():
    current, e01, e02 = examples()
    e02.loc[e02.horizon.eq(1), ["status", "effective_model"]] = ["fallback_no_training_pairs", "SeasonalNaive"]
    result = evaluate_chronos(current, e01, e02)
    group = result["comparison_coverage"].query("horizon == 1").iloc[0]
    assert DIRECT in group.participants and group.n_common_success == 0
    assert metric(result, "full", DIRECT, 1).metric_status == "no_training_pairs"


def test_reference_failure_is_explicit_and_empty_per_municipality_table_has_columns():
    current, e01, e02 = examples()
    e01.loc[e01.model.eq("ProphetAuto"), ["y_pred", "status", "reason"]] = [np.nan, "failed", "reference failure"]
    result = evaluate_chronos(current, e01, e02)
    assert metric(result, "full", "ProphetAuto", 1).metric_status == "incomplete_failed"
    assert result["evaluation_keys_common_success"].empty
    per_mo = result["metrics_common_success_by_municipality"]
    assert per_mo.empty and set(["municipality_id", "mae", "scope"]).issubset(per_mo.columns)


def test_metric_numbers_can_be_recomputed_from_explicit_saved_key_scope():
    current, e01, e02 = examples()
    result = evaluate_chronos(current, e01, e02)
    observed = current.merge(result["evaluation_keys_full"][KEY], on=KEY, validate="one_to_one")
    independently_computed, _ = compute_metrics(observed)
    actual = result["metrics_full"].loc[lambda x: x.model.eq(MODEL)]
    fields = ["split", "model", "horizon", "n_predictions", "mae_macro", "mae_micro", "r2_pooled"]
    pd.testing.assert_frame_equal(actual[fields].reset_index(drop=True), independently_computed[fields].reset_index(drop=True), check_dtype=False)
