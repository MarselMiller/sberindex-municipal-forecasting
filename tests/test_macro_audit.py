"""Offline checks of cohort preservation and historical audit aggregation."""
import json

import numpy as np
import pandas as pd
import pytest

from sberforecast.macro_audit import (e01_cases, forecast_grid, training_groups,
                                     training_queries, training_coverage, trend_readiness)


def test_e01_equal_counts_do_not_hide_different_keys():
    predictions = pd.DataFrame({"municipality_id": ["1", "2"], "forecast_origin": ["2023-12-31"] * 2,
        "target_period": ["2024-01-01"] * 2, "horizon": [1, 1], "model": ["A", "B"], "y_true": [10., 10.]})
    with pytest.raises(ValueError, match="support/targets differ"):
        e01_cases(predictions)


def test_e01_targets_and_duplicate_keys_are_checked():
    frame = pd.DataFrame({"municipality_id": ["1", "1"], "forecast_origin": ["2023-12-31"] * 2,
        "target_period": ["2024-01-01"] * 2, "horizon": [1, 1], "model": ["A", "B"], "y_true": [10., np.nan]})
    with pytest.raises(ValueError, match="support/targets differ"):
        e01_cases(frame)
    frame.loc[1, "model"] = "A"
    with pytest.raises(ValueError, match="Duplicate"):
        e01_cases(frame)


def test_forecast_grid_retains_missing_municipality_and_future_cases():
    frame = pd.DataFrame({"municipality_id": ["1"], "forecast_origin": ["2023-12-31"],
        "target_period": ["2024-01-01"], "horizon": [1], "model": ["A"], "y_true": [np.nan]})
    mapping = pd.DataFrame({"municipality_id": ["1", "2"], "region_id": ["10", "20"]})
    grid = forecast_grid(["1", "2"], pd.period_range("2023-12", periods=1, freq="M"), [1, 12], e01_cases(frame), mapping)
    assert len(grid) == 4 and grid.municipality_id.nunique() == 2
    assert grid.e01_case.sum() == 1 and grid.e01_evaluable.sum() == 0
    assert grid.loc[grid.horizon.eq(12), "target_period"].eq(pd.Timestamp("2024-12-01")).all()


def test_training_groups_preserve_counts_own_dates_and_empty_horizon():
    panel = pd.DataFrame({"1": [1., 2., 3., 4.], "2": [1., np.nan, 3., 4.]},
                         index=pd.date_range("2023-01-01", periods=4, freq="MS"))
    mapping = pd.DataFrame({"municipality_id": ["1", "2"], "region_id": ["10", "10"]})
    groups, counts = training_groups(panel, pd.period_range("2023-04", periods=1, freq="M"), [1, 12], mapping,
        release_lag_months=1, mode="legacy", max_staleness_months=None)
    queries, groups = training_queries(groups, mapping)
    assert counts.set_index("horizon").loc[1, "n_pairs"] == 2
    assert counts.set_index("horizon").loc[12, "n_pairs"] == 0
    assert len(queries) == 1
    query = queries.iloc[0]
    assert query.as_of_date == pd.Timestamp("2023-02-28")
    assert query.feature_cutoff == pd.Timestamp("2023-01-31")
    assert query.as_of_date < groups.model_origin.iloc[0]
    assert json.loads(groups.municipality_ids.iloc[0]) == ["1", "2"]
    provenance = pd.DataFrame({"sample_id": [query.sample_id], "feature": ["f"], "missing": [False],
                               "source_status": ["A+B"], "availability_status": ["B"], "geographic_level": ["national"]})
    coverage = training_coverage(groups, counts, provenance, ["f"]).set_index("horizon")
    assert coverage.loc[1, "n_available_pairs"] == 2
    assert coverage.loc[1, "n_available_municipalities"] == 2
    assert coverage.loc[1, "n_available_historical_dates"] == 1
    assert coverage.loc[1, "n_A_pairs"] == 0 and coverage.loc[1, "n_B_pairs"] == 2
    assert coverage.loc[12, "n_available_pairs"] == 0


def test_trend_readiness_counts_calendar_annual_pairs_without_looking_ahead():
    panel = pd.DataFrame({"1": np.arange(1., 25.)}, index=pd.date_range("2023-01-01", periods=24, freq="MS"))
    # make_panel() gives the real pivot this named column index.
    panel.columns.name = "municipality_id"
    grid = pd.DataFrame({"municipality_id": ["1", "1"], "forecast_origin": pd.to_datetime(["2023-12-31", "2024-03-31"]),
                         "horizon": [12, 1], "sample_id": ["a", "b"]})
    before = trend_readiness(panel, grid, release_lag_months=0)
    assert before.n_annual_pairs_last6.tolist() == [0, 3]
    assert before.native_trend_ready.tolist() == [False, True]
    panel.loc["2024-04-01":] = 999999.
    pd.testing.assert_frame_equal(before, trend_readiness(panel, grid, release_lag_months=0))
    panel.loc["2024-02-01", "1"] = np.nan
    after = trend_readiness(panel, grid, release_lag_months=0)
    assert after.iloc[1].n_annual_pairs_last6 == 2
    assert after.iloc[1].n_seasonal_months == 11
    assert not after.iloc[1].native_trend_ready
