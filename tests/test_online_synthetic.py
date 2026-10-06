"""Synthetic benchmark truth, independent seeds and causal forecast checks."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from sberforecast.backtest import run_origin
from sberforecast.models import baseline_predict
from sberforecast.online_synthetic import forecast_residuals, generate_benchmark


FORECAST = {
    "yearly_growth_window": 3,
    "yearly_growth_bounds": [0.5, 2.0],
    "min_history_observations": 12,
    "max_staleness_months": 1,
}
GENERATOR = {
    "months": 24,
    "start_period": "2023-01",
    "base_level": 10000.0,
    "seasonality_fraction": 0.1,
    "trend_per_month_fraction": 0.004,
    "noise_fractions": [0.01],
    "strengths": [2.0],
    "replicates_per_cell": 2,
    "validation_seed": 100000,
    "test_seed": 200000,
}


@pytest.fixture(scope="module")
def generated():
    return generate_benchmark(GENERATOR, "validation", FORECAST)


def _expenses():
    return pd.Series(
        10000.0 + 100 * np.arange(24, dtype=float),
        index=pd.date_range("2023-01", periods=24, freq="MS"),
        name="example",
    )


def test_generation_is_reproducible_and_seeds_identifiers_are_disjoint(generated):
    repeated = generate_benchmark(GENERATOR, "validation", FORECAST)
    test = generate_benchmark(GENERATOR, "test", FORECAST)
    for one, two in zip(generated, repeated):
        pd.testing.assert_frame_equal(one, two, check_exact=True)
    assert not set(generated[2].seed).intersection(test[2].seed)
    assert not set(generated[2].series_id).intersection(test[2].series_id)
    assert not np.array_equal(generated[3].value.to_numpy(), test[3].value.to_numpy())


def test_forecasts_match_actual_e01_algorithm_and_calendar(panel, cfg):
    cfg = deepcopy(cfg)
    cfg["models"]["enabled"] = ["SeasonalNaiveYoY"]
    cfg["models"].update(FORECAST)
    cfg["backtest"]["horizons"] = [1]
    helper = forecast_residuals(panel["1"], cfg["models"])
    expected = []
    for origin in pd.period_range("2023-12", "2024-11", freq="M"):
        records = run_origin(panel, cfg, origin)[0]
        expected.append(records.loc[records.municipality_id.eq("1")].iloc[0])
    expected = pd.DataFrame(expected).reset_index(drop=True)
    np.testing.assert_array_equal(helper.y_pred.to_numpy(), expected.y_pred.to_numpy())
    assert helper.forecast_origin.tolist() == expected.forecast_origin.tolist()
    assert helper.history_cutoff.tolist() == expected.history_cutoff.tolist()
    assert helper.target_period.tolist() == expected.target_period.tolist()


def test_future_and_current_target_values_do_not_enter_forecasts():
    observations = _expenses()
    original = forecast_residuals(observations, FORECAST)
    changed = observations.copy()
    changed.loc["2024-07-01":] += 10000
    future_changed = forecast_residuals(changed, FORECAST)
    pd.testing.assert_frame_equal(
        original.loc[original.observation_period.lt("2024-07")],
        future_changed.loc[future_changed.observation_period.lt("2024-07")],
        check_exact=True,
    )
    july = original.observation_period.eq("2024-07")
    np.testing.assert_array_equal(original.loc[july, "y_pred"], future_changed.loc[july, "y_pred"])
    assert original.loc[july, "y_true"].iloc[0] != future_changed.loc[july, "y_true"].iloc[0]


def test_truth_labels_never_enter_forecaster(generated):
    residuals, events, info, observations = generated
    # Every scenario and noise stream agrees exactly with separate-column
    # calls; batching may not introduce cross-series information.
    for sid in info.series_id:
        source = observations.loc[observations.series_id.eq(sid)]
        independent = forecast_residuals(pd.Series(
            source.value.to_numpy(), index=pd.to_datetime(source.observation_period), name=sid,
        ), FORECAST)
        pd.testing.assert_frame_equal(
            residuals.loc[residuals.series_id.eq(sid)].reset_index(drop=True), independent,
            check_exact=True,
        )
    assert set(events.columns).isdisjoint({"y_pred", "error"})
    assert "event_start_period" not in residuals.columns


def test_release_lag_limits_prefix_and_preserves_horizon_and_availability():
    observations = _expenses()
    rows = forecast_residuals(observations, FORECAST, release_lag_months=1)
    january, february = rows.iloc[0], rows.iloc[1]
    assert january.observation_period == "2024-01"
    assert january.history_cutoff == "2023-11-30"
    assert january.status == "ineligible_history" and np.isnan(january.y_pred)
    assert february.forecast_origin == "2024-01-31"
    assert february.history_cutoff == "2023-12-31"
    assert february.target_period == "2024-02-01" and february.horizon == 1
    expected, _ = baseline_predict(observations.iloc[:12].to_frame(), 2, "SeasonalNaiveYoY", FORECAST)
    assert february.y_pred == expected[1, 0]
    assert february.fact_available_period == "2024-03"
    assert february.fact_available_date == "2024-03-31"
    altered = observations.copy()
    altered.loc["2024-01-01"] += 100000
    two = forecast_residuals(altered, FORECAST, release_lag_months=1)
    assert two.iloc[1].y_pred == february.y_pred


def test_batched_lagged_forecasts_match_separate_historical_prefixes():
    residuals, _, info, observations = generate_benchmark(GENERATOR, "validation", FORECAST, 1)
    for sid in info.series_id:
        source = observations.loc[observations.series_id.eq(sid)]
        independent = forecast_residuals(pd.Series(
            source.value.to_numpy(), index=pd.to_datetime(source.observation_period), name=sid,
        ), FORECAST, 1)
        pd.testing.assert_frame_equal(
            residuals.loc[residuals.series_id.eq(sid)].reset_index(drop=True), independent,
            check_exact=True,
        )


def test_missing_months_retain_calendar_and_never_become_zero_errors():
    original = _expenses()
    with_hole = original.drop(pd.Timestamp("2023-02-01"))
    with_hole.loc["2024-05-01"] = np.nan
    rows = forecast_residuals(with_hole, FORECAST)
    assert rows.observation_period.tolist() == list(pd.period_range("2024-01", "2024-12", freq="M").astype(str))
    assert rows.iloc[0].n_history == 11 and np.isnan(rows.iloc[0].y_pred)
    february = rows.iloc[1]
    assert february.status == "fallback_last_value"
    assert february.y_pred == original.loc["2024-01-01"]
    may = rows.loc[rows.observation_period.eq("2024-05")].iloc[0]
    assert np.isnan(may.y_true) and np.isnan(may.error)
    # With one trailing missing month, E01 permits the June forecast.
    june = rows.loc[rows.observation_period.eq("2024-06")].iloc[0]
    assert june.staleness_months == 1 and np.isfinite(june.y_pred)


def test_staleness_is_calendar_based_and_configured():
    observations = _expenses()
    observations.loc["2024-04-01":"2024-05-01"] = np.nan
    rows = forecast_residuals(observations, FORECAST)
    may = rows.loc[rows.observation_period.eq("2024-05")].iloc[0]
    june = rows.loc[rows.observation_period.eq("2024-06")].iloc[0]
    assert may.staleness_months == 1 and np.isfinite(may.y_pred)
    assert june.staleness_months == 2 and june.status == "ineligible_history"
    assert np.isnan(june.error)
    cfg = dict(FORECAST, max_staleness_months=0)
    strict = forecast_residuals(observations, cfg)
    assert strict.loc[strict.observation_period.eq("2024-05"), "status"].iloc[0] == "ineligible_history"


def test_main_and_edge_events_and_nonstructural_controls_are_explicit(generated):
    residuals, events, info, observations = generated
    assert len(info) == 18
    assert len(observations) == 18 * 24
    assert len(residuals) == 18 * 12
    main = events.loc[events.is_main_protocol]
    assert main.duration_months.ge(3).all()
    assert main.event_start_period.ge("2024-05").all()
    assert main.event_start_period.le("2024-09").all()
    assert set(main.event_type) == {"level", "slope", "variance"}
    controls = info.loc[info.scenario.isin(["no_change", "outlier"])]
    assert not controls.has_structural_event.any()
    assert not set(controls.series_id).intersection(events.series_id)
    edge = events.loc[~events.is_main_protocol]
    assert set(edge.event_window_class) == {"warmup", "end_partial"}
    assert set(edge.loc[edge.event_window_class.eq("end_partial"), "duration_months"]) == {2}


def test_generator_has_fixed_seasonality_trend_and_balanced_single_outliers(generated):
    _, _, info, observations = generated
    positions = np.arange(24, dtype=float)
    ordinary = GENERATOR["base_level"] * (
        1 + GENERATOR["trend_per_month_fraction"] * positions
        + GENERATOR["seasonality_fraction"] * np.sin(2 * np.pi * positions / 12)
    )
    controls = info.loc[info.scenario.eq("no_change")]
    for row in controls.itertuples():
        rng = np.random.default_rng(row.seed)
        rng.choice([16, 17, 18, 19, 20])
        noise = rng.normal(0, GENERATOR["base_level"] * row.noise_fraction, 24)
        actual = observations.loc[observations.series_id.eq(row.series_id), "value"].to_numpy()
        np.testing.assert_allclose(actual - noise, ordinary, atol=1e-10)
    outliers = info.loc[info.scenario.eq("outlier")]
    signs = []
    for row in outliers.itertuples():
        rng = np.random.default_rng(row.seed)
        event_index = int(rng.choice([16, 17, 18, 19, 20]))
        noise = rng.normal(0, GENERATOR["base_level"] * row.noise_fraction, 24)
        actual = observations.loc[observations.series_id.eq(row.series_id), "value"].to_numpy()
        effect = actual - ordinary - noise
        assert np.count_nonzero(np.abs(effect) > 1e-9) == 1
        signs.append(np.sign(effect[event_index]))
    assert signs == [1, -1]


def test_invalid_or_overlapping_generator_protocol_is_rejected():
    for changes in (
        {"months": 25}, {"event_start_indices": [15]},
        {"test_seed": GENERATOR["validation_seed"] + 1},
        {"main_scenarios": ["level_up"]}, {"warmup_event_index": 18},
    ):
        cfg = dict(GENERATOR, **changes)
        with pytest.raises(ValueError):
            generate_benchmark(cfg, "validation", FORECAST)
    with pytest.raises(ValueError):
        generate_benchmark(GENERATOR, "holdout", FORECAST)
    with pytest.raises(ValueError):
        forecast_residuals(_expenses(), FORECAST, release_lag_months=-1)
