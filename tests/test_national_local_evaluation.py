"""Paired E05d metrics on deliberately unbalanced synthetic cases."""
import numpy as np
import pandas as pd
import pytest

from sberforecast.metrics import KEY
from sberforecast.national_local_evaluation import MODELS, evaluate_national_local


def examples():
    expected = pd.DataFrame([
        ("1", "2024-06-30", "2024-07-01", 1, 10.0, "holdout"),
        ("2", "2024-06-30", "2024-07-01", 1, 30.0, "holdout"),
        ("1", "2024-07-31", "2024-08-01", 1, 12.0, "holdout"),
        ("1", "2024-05-31", "2024-06-01", 1, 8.0, "validation"),
        ("1", "2023-12-31", "2024-12-01", 12, 20.0, "holdout"),
        ("2", "2023-12-31", "2024-12-01", 12, np.nan, "holdout"),
    ], columns=KEY + ["y_true", "split"])
    expected["history_cutoff"] = expected.forecast_origin
    expected["availability_assumption"] = "month_end_plus_0_months"
    frames = []
    for index, model in enumerate(MODELS):
        frame = expected.copy()
        frame["model"] = model
        frame["y_pred"] = frame.y_true.fillna(20) - (index + 1)
        frame["status"], frame["effective_model"], frame["reason"] = "native", model, ""
        if model in ("C0", "L0", "CN", "LN"):
            frame.loc[frame.horizon.eq(12), ["status", "effective_model", "reason"]] = [
                "fallback_no_training_pairs", "SeasonalNaive", "no_training_pairs"]
        frames.append(frame)
    national = expected[["forecast_origin", "target_period", "horizon", "split"]].drop_duplicates()
    national["n_hat"] = [18.0, 8.0, 9.0, 19.0]
    national["n_actual"] = [20.0, 10.0, 8.0, 20.0]
    national["national_forecast_status"] = "native"
    return pd.concat(frames, ignore_index=True), expected, national


def change_case(predictions, model, *, failed=False, origin="2024-06-30", mo="2"):
    selected = predictions.model.eq(model) & predictions.forecast_origin.eq(origin) & predictions.municipality_id.eq(mo)
    predictions.loc[selected, ["status", "effective_model", "reason"]] = [
        "failed" if failed else "fallback_missing_template", "none" if failed else "LastValue", "synthetic_failure"]
    if failed:
        predictions.loc[selected, "y_pred"] = np.nan


def test_exact_model_keys_and_four_contrasts_for_both_scopes():
    predictions, expected, national = examples()
    tables = evaluate_national_local(predictions, expected, national)
    assert len(tables) == 18
    deltas = tables["metric_deltas"]
    assert set(deltas.comparison) == {"L0-C0", "LN-CN", "CN-C0", "LN-L0"}
    assert set(deltas.scope) == {"strategy", "native"}
    assert set(deltas.hypothesis) == {"algorithm", "decomposition"}
    first = deltas.query("split == 'holdout' and horizon == 1 and scope == 'strategy'").set_index("comparison")
    assert first.delta_mae_micro.to_dict() == {"L0-C0": 1.0, "LN-CN": 1.0, "CN-C0": 2.0, "LN-L0": 2.0}


def test_strategy_failure_keeps_case_but_marks_metrics_incomplete():
    predictions, expected, national = examples()
    change_case(predictions, "LN", failed=True)
    tables = evaluate_national_local(predictions, expected, national)
    strategy = tables["metrics_strategy"].query("model == 'LN' and horizon == 1 and split == 'holdout'").iloc[0]
    assert strategy.n_predictions == 3 and strategy.metric_status == "incomplete_failed" and np.isnan(strategy.mae_macro)
    deltas = tables["metric_deltas"].query("scope == 'strategy' and split == 'holdout' and horizon == 1")
    assert deltas.loc[deltas.comparison.isin(["LN-CN", "LN-L0"]), "delta_mae_micro"].isna().all()
    assert deltas.loc[deltas.comparison.eq("L0-C0"), "delta_mae_micro"].iloc[0] == 1.0
    assert tables["metrics_native"].query("split == 'holdout' and horizon == 1").n_predictions.eq(2).all()
    assert "LN:failed" in tables["excluded_native_keys"].query("horizon == 1").iloc[0].reason


def test_same_count_different_keys_rejected_and_truth_protocol_guarded():
    predictions, expected, national = examples()
    predictions.loc[predictions.model.eq("L0") & predictions.municipality_id.eq("2"), "municipality_id"] = "3"
    with pytest.raises(ValueError):
        evaluate_national_local(predictions, expected, national)
    predictions, expected, national = examples()
    predictions.loc[predictions.index == 0, "y_true"] = 999
    with pytest.raises(ValueError, match="Целевые факты"):
        evaluate_national_local(predictions, expected, national)


def test_all_annual_direct_reserves_have_no_native_metric_or_delta():
    predictions, expected, national = examples()
    tables = evaluate_national_local(predictions, expected, national)
    annual = tables["metrics_native"].query("horizon == 12")
    assert annual.metric_status.eq("no_native_forecasts").all()
    assert annual.n_predictions.eq(0).all() and annual.mae_micro.isna().all()
    assert tables["metric_deltas"].query("horizon == 12 and scope == 'native'").delta_mae_micro.isna().all()
    coverage = tables["coverage"].query("model == 'C0' and horizon == 12").iloc[0]
    assert coverage.n_requested == 2 and coverage.n_fallback == 2
    assert tables["coverage_evaluable"].query("model == 'C0' and horizon == 12").iloc[0].n_requested == 1


def test_national_error_is_not_weighted_by_number_of_municipalities():
    predictions, expected, national = examples()
    national.loc[national.forecast_origin.eq("2024-07-31"), "n_hat"] = 0
    tables = evaluate_national_local(predictions, expected, national)
    result = tables["national_metrics"].query("split == 'holdout' and horizon == 1").iloc[0]
    assert result.n_evaluable_origins == 2 and result.mae_national == 6.0  # (2+10)/2, not (2+2+10)/3.
    assert len(tables["national_errors"]) == 4
    assert tables["national_errors"].actual_usage.eq("evaluation_only_never_forecast_feature").all()


@pytest.mark.parametrize("damage", ["duplicate", "missing", "horizon", "split", "silent_reserve", "inf_actual", "unknown_status", "alias"])
def test_invalid_national_diagnostic_keys_and_status_rejected(damage):
    predictions, expected, national = examples()
    if damage == "duplicate":
        national = pd.concat([national, national.iloc[[0]]], ignore_index=True)
    elif damage == "missing":
        national = national.iloc[1:]
    elif damage == "horizon":
        national.loc[national.index[0], "target_period"] = "2024-08-01"
    elif damage == "split":
        national.loc[national.index[0], "split"] = "validation"
    elif damage == "silent_reserve":
        national.loc[national.index[0], "national_forecast_status"] = "failed"
    elif damage == "inf_actual":
        national.loc[national.index[0], "n_actual"] = np.inf
    elif damage == "unknown_status":
        national["national_forecast_status"] = "unknown"
    else:
        national["N_hat"] = national.n_hat + 1
    with pytest.raises(ValueError):
        evaluate_national_local(predictions, expected, national)


def test_missing_future_national_actual_not_backfilled_and_failed_not_shrunk():
    predictions, expected, national = examples()
    national.loc[national.horizon.eq(12), "n_actual"] = np.nan
    national.loc[national.forecast_origin.eq("2024-06-30"), ["n_hat", "national_forecast_status"]] = [np.nan, "failed"]
    tables = evaluate_national_local(predictions, expected, national)
    annual = tables["national_metrics"].query("horizon == 12").iloc[0]
    assert annual.metric_status == "no_actual_national_target" and np.isnan(annual.mae_national)
    monthly = tables["national_metrics"].query("split == 'holdout' and horizon == 1").iloc[0]
    assert monthly.n_actual_available == 2 and monthly.n_evaluable_origins == 1
    assert monthly.metric_status == "incomplete_failed" and np.isnan(monthly.mae_national)


def test_single_origin_improvement_explicit_descriptive_robustness():
    predictions, expected, national = examples()
    # C0 errors 10,10,1; L0 errors 1,1,2. Gains come only from June.
    for model, errors in (("C0", [10, 10, 1]), ("L0", [1, 1, 2])):
        selected = predictions.model.eq(model) & predictions.horizon.eq(1) & predictions.split.eq("holdout")
        predictions.loc[selected, "y_pred"] = predictions.loc[selected, "y_true"] - errors
    tables = evaluate_national_local(predictions, expected, national)
    result = tables["origin_robustness"].query("scope == 'strategy' and split == 'holdout' and horizon == 1 and comparison == 'L0-C0'").iloc[0]
    assert result.n_origins_improved == 1 and result.n_origins_worsened == 1
    assert result.improvement_only_one_origin and result.largest_improvement_share == 1.0
    assert result.largest_improvement_origin == "2024-06-30"
    assert result.pooled_delta_mae_micro < 0
    assert result.interpretation == "descriptive_no_significance_test_no_holdout_selection"


def test_origin_and_municipality_metrics_preserved_and_inputs_not_mutated():
    predictions, expected, national = examples()
    before = [frame.copy(deep=True) for frame in (predictions, expected, national)]
    tables = evaluate_national_local(predictions, expected, national)
    assert set(tables["metrics_native_by_origin"].model) == set(MODELS)
    assert set(tables["metrics_strategy_by_municipality"].model) == set(MODELS)
    for actual, original in zip((predictions, expected, national), before):
        pd.testing.assert_frame_equal(actual, original)
