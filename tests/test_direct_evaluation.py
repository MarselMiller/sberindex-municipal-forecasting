"""Синтетические проверки областей сравнения и сохранения failed."""
import numpy as np
import pandas as pd
import pytest

from sberforecast.direct_evaluation import evaluate_direct, h1_consistency, require_same_keys
from sberforecast.direct_experiment import expected_cases
from sberforecast.direct_model import NATIVE_MODEL, STRATEGY
from sberforecast.metrics import KEY


def examples(include_failure=False):
    rows = [dict(municipality_id="1", forecast_origin="2024-06-30", target_period="2024-07-01",
                 horizon=1, y_true=10., y_pred=9., split="holdout", status="native",
                 effective_model=NATIVE_MODEL),
            dict(municipality_id="2", forecast_origin="2023-12-31", target_period="2024-12-01",
                 horizon=12, y_true=20., y_pred=10., split="holdout", status="fallback_no_training_pairs",
                 effective_model="SeasonalNaive")]
    if include_failure:
        rows.append(dict(municipality_id="3", forecast_origin="2024-06-30", target_period="2024-07-01",
                         horizon=1, y_true=30., y_pred=np.nan, split="holdout", status="failed",
                         effective_model="none"))
    direct = pd.DataFrame(rows).assign(model=STRATEGY)
    reference = pd.concat([direct.assign(model=name, y_pred=direct.y_true) for name in
                           ("CatBoostRecursive", "SeasonalNaive")], ignore_index=True)
    return direct, reference


def test_strategy_and_native_metrics_are_separate_and_paired():
    direct, ref = examples()
    result = evaluate_direct(direct, ref, ["CatBoostRecursive", "SeasonalNaive"])
    strategy = result["metrics_strategy"]
    native = result["metrics_native"]
    reserve = strategy.loc[strategy.model.eq(STRATEGY) & strategy.horizon.eq(12)].iloc[0]
    pure = native.loc[native.model.eq(NATIVE_MODEL) & native.horizon.eq(12)].iloc[0]
    assert reserve.mae_macro == 10 and reserve.n_predictions == 1
    assert pure.metric_status == "no_training_pairs" and pure.n_predictions == 0
    assert np.isnan(pure.mae_macro)
    assert len(result["evaluation_keys_strategy"]) == 2
    pd.testing.assert_frame_equal(result["evaluation_keys_native"].reset_index(drop=True), direct.iloc[:1][KEY])
    h1 = native.loc[native.horizon.eq(1)]
    assert h1.n_predictions.eq(1).all()  # Все соперники на том же одном ключе.


def test_failed_does_not_silently_shrink_full_strategy():
    direct, ref = examples(include_failure=True)
    result = evaluate_direct(direct, ref, ["CatBoostRecursive", "SeasonalNaive"])
    full = result["metrics_strategy"].query("model == @STRATEGY and horizon == 1").iloc[0]
    assert full.metric_status == "incomplete_failed"
    assert full.n_predictions == 2 and full.n_failed == 1 and np.isnan(full.mae_macro)
    native = result["metrics_native"].query("horizon == 1")
    assert native.n_predictions.eq(1).all()
    coverage = result["coverage_e01"].query("horizon == 1").iloc[0]
    assert coverage.n_e01_cases == 2 and coverage.n_failed == 1 and coverage.failed_share == 0.5


def test_equal_counts_with_different_identifiers_are_rejected():
    direct, ref = examples()
    ref.loc[ref.municipality_id.eq("1"), "municipality_id"] = "different"
    with pytest.raises(ValueError, match="Несовпадение ключей"):
        evaluate_direct(direct, ref, ["CatBoostRecursive", "SeasonalNaive"])


def test_duplicate_prediction_key_is_rejected():
    direct, _ = examples()
    with pytest.raises(ValueError, match="повторяющиеся"):
        require_same_keys(pd.concat([direct, direct]), direct)


def test_h1_tolerance_is_absolute_and_failed_count_is_explicit():
    direct, ref = examples(include_failure=True)
    ref.loc[(ref.model == "CatBoostRecursive") & (ref.municipality_id == "1"), "y_pred"] = 9. + 1e-10
    check, _ = h1_consistency(direct, ref, 1e-8)
    assert check["n_expected_cases"] == 2 and check["n_compared_native_cases"] == 1
    assert check["n_uncompared_cases"] == 1 and not check["complete"]
    assert check["n_exceeds_tolerance"] == 0 and check["relative_tolerance"] == 0
    ref.loc[(ref.model == "CatBoostRecursive") & (ref.municipality_id == "1"), "y_pred"] = 9.01
    assert h1_consistency(direct, ref, 1e-8)[0]["n_exceeds_tolerance"] == 1


def test_current_cohort_does_not_preexclude_future_missing_municipality(panel, cfg):
    changed = panel.rename(columns={"1": "1471"}).copy()
    changed.loc["2024-01-01":, "1471"] = np.nan
    cases = expected_cases(changed, cfg, {"1471", "2"})
    tracked = cases.loc[cases.municipality_id.eq("1471")]
    assert set(tracked.forecast_origin) == {"2023-12-31", "2024-01-31"}
    assert tracked.y_true.isna().all() and len(tracked) == 7
