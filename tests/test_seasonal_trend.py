"""Синтетические проверки фиксированного сезонного тренда; без реальных данных."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sberforecast.models import baseline_predict
from sberforecast.seasonal_trend import predict_seasonal_trend


def synthetic_panel(*, b: float = 8.0, start: str = "2022-01",
                    end: str = "2025-12", base: float = 1000.0) -> pd.DataFrame:
    periods = pd.period_range(start, end, freq="M")
    season = np.array([20.0, -10.0, 0.0, 10.0, -30.0, 40.0,
                       -5.0, 25.0, -15.0, 5.0, -20.0, -20.0])
    values = base + b * np.arange(len(periods)) + season[periods.month - 1]
    return pd.DataFrame({"b_mo": values, "a_mo": values * 2}, index=periods.to_timestamp())


def predict(panel: pd.DataFrame, origin: str = "2024-12", horizon: int = 1,
            *, variant: str = "SeasonalTrendLinear", lag: int = 0, phi: float = 0.9):
    return predict_seasonal_trend(
        panel, pd.Period(origin, freq="M"), horizon, release_lag_months=lag,
        variant=variant, phi=phi,
    )


@pytest.mark.parametrize("variant", ["SeasonalTrendLinear", "SeasonalTrendDamped"])
def test_future_values_do_not_change_b_template_or_prediction(variant):
    panel = synthetic_panel()
    before, diagnostic_before = predict(panel, "2024-10", 6, variant=variant, lag=2)
    changed = panel.copy()
    changed.loc[changed.index > pd.Timestamp("2024-08-01")] = -1e12
    after, diagnostic_after = predict(changed, "2024-10", 6, variant=variant, lag=2)
    np.testing.assert_array_equal(before, after)
    pd.testing.assert_frame_equal(diagnostic_before, diagnostic_after)


@pytest.mark.parametrize("horizon", [1, 3, 6, 12])
@pytest.mark.parametrize("lag", [0, 2])
def test_linear_recovers_additive_trend_and_season_without_noise(horizon, lag):
    panel = synthetic_panel()
    predictions, diagnostics = predict(panel, "2024-06", horizon, lag=lag)
    target = (pd.Period("2024-06", freq="M") + horizon).to_timestamp()
    np.testing.assert_allclose(predictions, panel.loc[target].to_numpy(), atol=1e-10)
    np.testing.assert_allclose(diagnostics["b"], [8.0, 16.0], atol=1e-12)
    assert diagnostics["status"].eq("native").all()
    assert diagnostics["n_annual_pairs"].eq(6).all()
    assert diagnostics["distance_from_cutoff"].eq(horizon + lag).all()
    assert diagnostics["target_period"].eq(target).all()


def test_lag_moves_cutoff_but_target_stays_relative_to_origin():
    panel = synthetic_panel()
    predictions, diagnostics = predict(panel, "2024-05", 1, lag=2)
    assert diagnostics["feature_cutoff"].eq(pd.Timestamp("2024-03-31")).all()
    assert diagnostics["target_period"].eq(pd.Timestamp("2024-06-01")).all()
    np.testing.assert_allclose(predictions, panel.loc["2024-06-01"], atol=1e-10)


def test_phi_one_reproduces_linear_forecast():
    panel = synthetic_panel()
    linear, linear_diagnostics = predict(panel, "2024-09", 6, lag=1)
    damped, damped_diagnostics = predict(
        panel, "2024-09", 6, lag=1, variant="SeasonalTrendDamped", phi=1.0)
    np.testing.assert_array_equal(linear, damped)
    np.testing.assert_array_equal(linear_diagnostics["b"], damped_diagnostics["b"])
    assert damped_diagnostics["effective_model"].eq("SeasonalTrendDamped").all()


def test_damping_starts_at_power_one_not_zero():
    panel = synthetic_panel()
    linear, _ = predict(panel, "2024-12", 1)
    damped, _ = predict(panel, "2024-12", 1, variant="SeasonalTrendDamped")
    np.testing.assert_allclose(linear - damped, [0.8, 1.6], atol=1e-10)
    linear_three, _ = predict(panel, "2024-12", 3)
    damped_three, _ = predict(panel, "2024-12", 3, variant="SeasonalTrendDamped")
    np.testing.assert_allclose(
        linear_three - damped_three, np.array([8.0, 16.0]) * (3 - (0.9 + 0.81 + 0.729)),
        atol=1e-10,
    )


@pytest.mark.parametrize("variant", ["SeasonalTrendLinear", "SeasonalTrendDamped"])
def test_zero_trend_reproduces_same_calendar_season(variant):
    panel = synthetic_panel(b=0.0)
    predictions, diagnostics = predict(panel, "2024-09", 6, variant=variant)
    np.testing.assert_array_equal(predictions, panel.loc["2025-03-01"])
    np.testing.assert_array_equal(diagnostics["b"], [0.0, 0.0])
    seasonal_columns = [f"seasonal_{month:02d}" for month in range(1, 13)]
    np.testing.assert_allclose(diagnostics[seasonal_columns].mean(axis=1), 0.0, atol=1e-12)


def test_fewer_than_three_pairs_gives_same_existing_yoy_fallback():
    panel = synthetic_panel(start="2023-01")
    predictions, diagnostics = predict(panel, "2024-02", 3)
    history = panel.loc[:"2024-02-01"]
    expected, _ = baseline_predict(history, 3, "SeasonalNaiveYoY", {})
    np.testing.assert_array_equal(predictions, expected[-1])
    assert diagnostics["n_annual_pairs"].eq(2).all()
    assert diagnostics["status"].eq("fallback").all()
    assert diagnostics["reason"].eq("insufficient_annual_pairs").all()
    assert diagnostics["effective_model"].eq("SeasonalNaiveYoY").all()


def test_incomplete_template_gives_explicit_yoy_with_internal_lastvalue():
    panel = synthetic_panel()
    panel.loc["2024-01-01"] = np.nan
    predictions, diagnostics = predict(panel, "2024-12", 1)
    expected, flags = baseline_predict(panel.loc[:"2024-12-01"], 1, "SeasonalNaiveYoY", {})
    np.testing.assert_array_equal(predictions, expected[-1])
    np.testing.assert_array_equal(diagnostics["baseline_internal_fallback"], flags[-1])
    assert diagnostics["n_annual_pairs"].eq(6).all()
    assert diagnostics["seasonal_template_complete"].eq(False).all()
    assert diagnostics["reason"].eq("incomplete_seasonal_template").all()
    assert diagnostics["baseline_effective_model"].eq("LastValue").all()


def test_fixed_six_calendar_positions_do_not_search_older_nonempty_pairs():
    panel = synthetic_panel()
    panel.loc["2023-07-01":"2023-10-01"] = np.nan
    _, diagnostics = predict(panel, "2024-12", 3)
    # Older 2024-01...2024-06 pairs are finite, but outside the fixed window.
    assert diagnostics["n_annual_pairs"].eq(2).all()
    assert diagnostics["seasonal_template_complete"].all()
    assert diagnostics["status"].eq("fallback").all()


def test_sparse_months_preserve_calendar_annual_pairs():
    panel = synthetic_panel()
    sparse = panel.drop(pd.Timestamp("2023-09-01"))
    predictions, diagnostics = predict(sparse, "2024-12", 1)
    assert diagnostics["n_annual_pairs"].eq(5).all()
    np.testing.assert_array_equal(diagnostics["b"], [8.0, 16.0])
    np.testing.assert_allclose(predictions, panel.loc["2025-01-01"], atol=1e-10)


def test_sparse_month_in_template_is_not_filled_from_next_observation():
    panel = synthetic_panel()
    _, diagnostics = predict(panel.drop(pd.Timestamp("2024-04-01")), "2024-12", 1)
    assert diagnostics["n_annual_pairs"].eq(6).all()
    assert diagnostics["status"].eq("fallback").all()
    assert diagnostics["reason"].eq("incomplete_seasonal_template").all()


def test_exactly_three_pairs_are_enough_with_complete_template():
    panel = synthetic_panel(start="2023-01")
    _, diagnostics = predict(panel, "2024-03", 1)
    assert diagnostics["n_annual_pairs"].eq(3).all()
    assert diagnostics["status"].eq("native").all()


def test_negative_trend_forecast_is_clipped_to_zero():
    panel = synthetic_panel(b=-50.0, base=3000.0)
    predictions, diagnostics = predict(panel, "2024-12", 48)
    assert diagnostics["status"].eq("native").all()
    np.testing.assert_array_equal(diagnostics["b"], [-50.0, -100.0])
    np.testing.assert_array_equal(predictions, [0.0, 0.0])


@pytest.mark.parametrize("variant", ["SeasonalTrendLinear", "SeasonalTrendDamped"])
def test_first_annual_forecast_is_not_native(variant):
    panel = synthetic_panel(start="2023-01")
    predictions, diagnostics = predict(panel, "2023-12", 12, variant=variant)
    expected, _ = baseline_predict(panel.loc[:"2023-12-01"], 12, "SeasonalNaiveYoY", {})
    np.testing.assert_array_equal(predictions, expected[-1])
    assert diagnostics["n_annual_pairs"].eq(0).all()
    assert diagnostics["status"].eq("fallback").all()
    assert diagnostics["effective_model"].eq("SeasonalNaiveYoY").all()


def test_fallback_with_publication_lag_uses_h_plus_l_steps():
    panel = synthetic_panel(start="2023-01")
    predictions, diagnostics = predict(panel, "2024-02", 3, lag=2)
    expected, _ = baseline_predict(panel.loc[:"2023-12-01"], 5, "SeasonalNaiveYoY", {})
    np.testing.assert_array_equal(predictions, expected[-1])
    assert diagnostics["target_period"].eq(pd.Timestamp("2024-05-01")).all()


def test_prefix_only_and_full_panel_give_identical_outputs_and_keep_column_order():
    panel = synthetic_panel()
    full_predictions, full_diagnostics = predict(panel, "2024-12", 3)
    prefix_predictions, prefix_diagnostics = predict(panel.loc[:"2024-12-01"], "2024-12", 3)
    np.testing.assert_array_equal(full_predictions, prefix_predictions)
    pd.testing.assert_frame_equal(full_diagnostics, prefix_diagnostics)
    assert full_diagnostics["municipality_id"].tolist() == ["b_mo", "a_mo"]


def test_execution_error_is_not_disguised_as_planned_fallback():
    panel = synthetic_panel(start="2023-01")
    panel["b_mo"] = np.nan
    with pytest.raises(ValueError, match="хотя бы один факт"):
        predict(panel, "2023-12", 12)


@pytest.mark.parametrize("change", ["duplicate_month", "duplicate_municipality", "empty_history"])
def test_invalid_panel_raises_without_fallback(change):
    panel = synthetic_panel()
    if change == "duplicate_month":
        panel = pd.concat([panel, panel.iloc[[0]]])
    elif change == "duplicate_municipality":
        panel.columns = ["same", "same"]
    else:
        panel = panel.loc[:"2022-01-01"]
    with pytest.raises(ValueError):
        predict(panel, "2021-12", 1)


@pytest.mark.parametrize("invalid", [0.0, -0.1, 1.1, np.inf, np.nan])
def test_invalid_phi_raises(invalid):
    with pytest.raises(ValueError, match="phi"):
        predict(synthetic_panel(), phi=invalid)
