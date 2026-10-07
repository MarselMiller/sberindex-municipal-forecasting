"""Synthetic tests of fixed-case E08c metrics and the declared success gate."""
import json

import numpy as np
import pandas as pd
import pytest

from sberforecast.financial_forecast_evaluation import (
    EPSILON_RUB, MODEL_FAMILIES, VARIANTS, evaluate_financial_forecast,
)
from sberforecast.metrics import KEY


def examples(*, families=(MODEL_FAMILIES[0],), missing_annual=False):
    rows = []
    dates = {
        ("validation", 1): ["2024-01", "2024-02", "2024-03"],
        ("validation", 3): ["2024-01", "2024-02", "2024-03"],
        ("validation", 6): ["2023-12"],
        ("holdout", 1): ["2024-06", "2024-07", "2024-08"],
        ("holdout", 3): ["2024-04", "2024-05", "2024-06"],
        ("holdout", 6): ["2024-01", "2024-02", "2024-03"],
        ("holdout", 12): ["2023-12"],
    }
    for (split, horizon), months in dates.items():
        for month in months:
            period = pd.Period(month, freq="M")
            origin = str(period.end_time.normalize().date())
            for uid, truth in (("SYNTHETIC_A", 10.0), ("SYNTHETIC_B", 30.0)):
                if missing_annual and horizon == 12 and uid == "SYNTHETIC_B":
                    truth = np.nan
                rows.append((uid, origin, str((period + horizon).start_time.date()), horizon, truth, split))
    expected = pd.DataFrame(rows, columns=KEY + ["y_true", "split"])
    expected["history_cutoff"] = expected.forecast_origin
    expected["availability_assumption"] = "month_end_plus_0_months"
    frames = []
    for family in families:
        for variant, error in zip(VARIANTS, (10.0, 9.0, 8.0, 7.0)):
            frame = expected.copy()
            frame["model"], frame["variant"] = family, variant
            frame["y_pred"] = frame.y_true.fillna(30.0) - error
            frame["status"], frame["effective_model"], frame["reason"] = "native", family, ""
            annual = frame.horizon.eq(12)
            frame.loc[annual, "y_pred"] = frame.loc[annual, "y_true"].fillna(30.0) - 10
            frame.loc[annual, ["status", "effective_model", "reason"]] = [
                "fallback_no_training_pairs", "SeasonalNaive", "no_training_pairs"]
            frames.append(frame)
    return pd.concat(frames, ignore_index=True), expected


def set_errors(predictions, variant, errors, *, split=None, horizon=None, family=None):
    mask = predictions.variant.eq(variant)
    if split is not None:
        mask &= predictions.split.eq(split)
    if horizon is not None:
        mask &= predictions.horizon.eq(horizon)
    if family is not None:
        mask &= predictions.model.eq(family)
    selected = predictions.loc[mask]
    origins = sorted(selected.forecast_origin.unique())
    values = dict(zip(origins, errors)) if isinstance(errors, list) else dict.fromkeys(origins, errors)
    predictions.loc[mask, "y_pred"] = selected.y_true.fillna(30) - selected.forecast_origin.map(values)


def test_paired_metrics_relative_units_and_origin_gate():
    predictions, expected = examples()
    result = evaluate_financial_forecast(predictions, expected)
    f3 = result["forecasting_metrics"].query("variant == 'F3' and horizon == 1 and split == 'holdout'").iloc[0]
    assert f3.mae_macro == 7 and f3.delta_mae_macro_vs_F0 == -3
    assert f3.relative_delta_mae_macro_vs_F0 == -0.3
    assert f3.relative_delta_mae_macro_pct_vs_F0 == -30
    consistency = result["origin_consistency"].query("variant == 'F3' and horizon == 1 and split == 'holdout'").iloc[0]
    assert consistency.n_origins_improved == 3 and consistency.n_origins_worsened == 0
    assert consistency.mean_delta_mae_macro == consistency.median_delta_mae_macro == -3
    assert consistency.leave_best_origin_out_mean_delta == -3
    assert consistency.largest_improvement_share == pytest.approx(1 / 3)
    success = result["success_criterion"].iloc[0]
    assert success.conclusion == "PROMISING" and json.loads(success.qualifying_horizons) == [1, 3, 6]
    assert json.loads(success.single_origin_concentration_undetermined) == ["validation/h=6"]
    assert not success.h12_used_for_success


def test_macro_micro_and_pooled_r2_reuse_original_definition_on_unbalanced_cases():
    expected = pd.DataFrame([
        ("SYNTHETIC_A", "2024-06-30", "2024-07-01", 1, 10., "holdout"),
        ("SYNTHETIC_B", "2024-06-30", "2024-07-01", 1, 30., "holdout"),
        ("SYNTHETIC_A", "2024-07-31", "2024-08-01", 1, 12., "holdout"),
    ], columns=KEY + ["y_true", "split"])
    expected["history_cutoff"] = expected.forecast_origin
    expected["availability_assumption"] = "month_end_plus_0_months"
    frames = []
    for variant in VARIANTS:
        frame = expected.assign(model=MODEL_FAMILIES[0], variant=variant,
                                status="native", effective_model=MODEL_FAMILIES[0], reason="")
        frame["y_pred"] = frame.y_true - [0., 10., 0.]
        frames.append(frame)
    result = evaluate_financial_forecast(pd.concat(frames, ignore_index=True), expected)
    metric = result["forecasting_metrics"].query("variant == 'F0'").iloc[0]
    assert metric.mae_macro == 5 and metric.mae_micro == pytest.approx(10 / 3)
    assert metric.r2_pooled == pytest.approx(107 / 182)


def test_holdout_only_gain_is_not_promising():
    predictions, expected = examples()
    set_errors(predictions, "F3", 12, split="validation")
    result = evaluate_financial_forecast(predictions, expected)
    assert result["success_criterion"].iloc[0].conclusion == "NO STABLE FORECASTING UPLIFT"
    assert json.loads(result["success_criterion"].iloc[0].qualifying_horizons) == []


@pytest.mark.parametrize("paired_horizons, conclusion", [([6], "NO STABLE FORECASTING UPLIFT"), ([1, 3], "PROMISING")])
def test_at_least_two_same_horizons_must_improve_both_splits(paired_horizons, conclusion):
    predictions, expected = examples()
    for horizon in (1, 3, 6):
        if horizon not in paired_horizons:
            # Opposite split gains cannot be combined into a paired horizon.
            split = "holdout" if horizon == 1 else "validation"
            set_errors(predictions, "F3", 12, split=split, horizon=horizon)
    success = evaluate_financial_forecast(predictions, expected)["success_criterion"].iloc[0]
    assert json.loads(success.qualifying_horizons) == paired_horizons
    assert success.conclusion == conclusion


@pytest.mark.parametrize("errors, improved", [([0., 11., 11.], 1), ([0., 9., 18.], 2)])
def test_gain_concentrated_in_best_origin_rejected_even_with_negative_aggregate(errors, improved):
    predictions, expected = examples()
    for horizon in (1, 3):
        set_errors(predictions, "F3", errors, split="holdout", horizon=horizon)
    result = evaluate_financial_forecast(predictions, expected)
    metric = result["forecasting_metrics"].query("variant == 'F3' and horizon == 1 and split == 'holdout'").iloc[0]
    assert metric.delta_mae_macro_vs_F0 < 0
    consistency = result["origin_consistency"].query("variant == 'F3' and horizon == 1 and split == 'holdout'").iloc[0]
    assert consistency.n_origins_improved == improved
    assert consistency.leave_best_origin_out_mean_delta > 0 and consistency.concentration_status == "FAIL"
    assert result["success_criterion"].iloc[0].conclusion == "NO STABLE FORECASTING UPLIFT"


def test_failed_evaluable_row_retained_and_unknown_truth_visible_in_coverage():
    predictions, expected = examples(missing_annual=True)
    failed = predictions.variant.eq("F3") & predictions.horizon.eq(1) & predictions.split.eq("holdout")
    index = predictions.loc[failed].index[0]
    predictions.loc[index, ["y_pred", "status", "effective_model", "reason"]] = [np.nan, "failed", "none", "synthetic_fit_failure"]
    result = evaluate_financial_forecast(predictions, expected)
    metric = result["forecasting_metrics"].query("variant == 'F3' and horizon == 1 and split == 'holdout'").iloc[0]
    assert metric.n_predictions == 6 and metric.n_failed == 1
    assert metric.metric_status == "incomplete_failed" and np.isnan(metric.mae_macro)
    assert np.isnan(metric.delta_mae_macro_vs_F0)
    native = result["metrics_native"].query("variant == 'F0' and horizon == 1 and split == 'holdout'").iloc[0]
    assert native.n_predictions == 5
    raw = result["coverage"].query("variant == 'F0' and horizon == 12").iloc[0]
    valid = result["coverage_evaluable"].query("variant == 'F0' and horizon == 12").iloc[0]
    assert raw.n_requested == raw.n_fallback == 2 and raw.n_actual_available == 1
    assert valid.n_requested == 1
    assert result["forecasting_metrics"].query("horizon == 12").evaluation_role.eq("DESCRIPTIVE ONLY").all()
    assert result["metrics_native"].query("horizon == 12").metric_status.eq("no_native_forecasts").all()


def test_annual_only_gain_excluded_from_success():
    predictions, expected = examples()
    set_errors(predictions, "F3", 12)
    set_errors(predictions, "F3", 0, horizon=12)
    result = evaluate_financial_forecast(predictions, expected)
    annual = result["forecasting_metrics"].query("variant == 'F3' and horizon == 12").iloc[0]
    assert annual.delta_mae_macro_vs_F0 == -10
    assert result["success_criterion"].iloc[0].conclusion == "NO STABLE FORECASTING UPLIFT"


def test_epsilon_ties_and_zero_baseline_relative_delta_missing():
    predictions, expected = examples()
    set_errors(predictions, "F3", 10 - EPSILON_RUB / 2)
    result = evaluate_financial_forecast(predictions, expected)
    part = result["origin_consistency"].query("variant == 'F3' and horizon == 1 and split == 'holdout'").iloc[0]
    assert part.n_origins_tied == 3 and part.n_origins_improved == 0
    assert result["success_criterion"].iloc[0].conclusion == "NO STABLE FORECASTING UPLIFT"
    set_errors(predictions, "F0", 0)
    result = evaluate_financial_forecast(predictions, expected)
    assert result["forecasting_metrics"].relative_delta_mae_macro_vs_F0.isna().all()


def test_primary_and_secondary_conclusions_do_not_select_a_family_and_inputs_not_mutated():
    predictions, expected = examples(families=MODEL_FAMILIES)
    set_errors(predictions, "F3", 12, family=MODEL_FAMILIES[0])
    original, reference = predictions.copy(deep=True), expected.copy(deep=True)
    result = evaluate_financial_forecast(predictions, expected)
    conclusions = result["success_criterion"].set_index("model")
    assert conclusions.loc[MODEL_FAMILIES[0], "conclusion"] == "NO STABLE FORECASTING UPLIFT"
    assert conclusions.loc[MODEL_FAMILIES[1], "conclusion"] == "PROMISING"
    assert conclusions.loc[MODEL_FAMILIES[0], "family_role"] == "PRIMARY"
    assert conclusions.loc[MODEL_FAMILIES[1], "family_role"] == "SECONDARY SENSITIVITY"
    pd.testing.assert_frame_equal(predictions, original)
    pd.testing.assert_frame_equal(expected, reference)


@pytest.mark.parametrize("damage", ["key", "truth", "split", "duplicate", "missing_variant", "family", "inf_truth", "silent_failed"])
def test_fixed_keys_truth_protocol_and_failures_guarded(damage):
    predictions, expected = examples()
    if damage == "key":
        predictions.loc[0, "municipality_id"] = "OTHER_SYNTHETIC_ID"
    elif damage == "truth":
        predictions.loc[0, "y_true"] = 999
    elif damage == "split":
        predictions.loc[0, "split"] = "holdout"
    elif damage == "duplicate":
        predictions = pd.concat([predictions, predictions.iloc[[0]]], ignore_index=True)
    elif damage == "missing_variant":
        predictions = predictions.loc[~predictions.variant.eq("F1")]
    elif damage == "family":
        predictions["model"] = "UnapprovedModel"
    elif damage == "inf_truth":
        expected.loc[0, "y_true"] = np.inf
    else:
        predictions.loc[0, "y_pred"] = np.nan
    with pytest.raises(ValueError):
        evaluate_financial_forecast(predictions, expected)
