"""Synthetic tests of National/Local chronology; no learner or network required."""
import numpy as np
import pandas as pd
import pytest

from sberforecast.data import get_prefix, period_end
from sberforecast.direct_training import PAIR_KEY, build_direct_training, direct_features
from sberforecast.models import baseline_predict
from sberforecast.national_local import (
    build_local_training, build_national, national_forecast, ratio_panel,
    restore_local_prediction,
)


@pytest.fixture
def panel():
    months = pd.date_range("2023-01-01", periods=30, freq="MS")
    base = np.arange(100.0, 130.0)
    return pd.DataFrame({"a": base, "b": 2 * base, "c": 3 * base}, index=months)


def test_national_uses_all_panel_not_selected_forecast_ids(panel):
    national = build_national(panel)
    assert national.N.iloc[0] == 200.0
    assert national.n_municipalities_used.eq(3).all()
    assert national.N.iloc[0] != build_national(panel[["a"]]).N.iloc[0]
    assert national.N.equals(national["median"])
    assert national.N.equals(national.national_y)


def test_national_uses_finite_values_in_same_month_only():
    panel = pd.DataFrame({"a": [10.0, 1000.0, 10.0],
                          "b": [20.0, np.inf, 20.0],
                          "c": [np.nan, 5000.0, 30.0]},
                         index=pd.date_range("2023-01-01", periods=3, freq="MS"))
    national = build_national(panel)
    assert national.N.tolist() == [15.0, 3000.0, 20.0]
    assert national.n_municipalities_used.tolist() == [2, 2, 3]
    changed = panel.copy()
    changed.iloc[1] = [np.nan, 99.0, 999999.0]
    pd.testing.assert_series_equal(build_national(changed).iloc[0], national.iloc[0])
    pd.testing.assert_series_equal(build_national(changed).iloc[2], national.iloc[2])


def test_future_values_do_not_change_old_national(panel):
    changed = panel.copy()
    changed.loc[changed.index > "2023-12-01"] = 1e8
    old = build_national(panel)
    new = build_national(changed)
    pd.testing.assert_frame_equal(old.loc[:"2023-12-01"], new.loc[:"2023-12-01"])


def test_future_municipality_does_not_enter_past_cohort(panel):
    old = build_national(panel)
    changed = panel.copy()
    changed["future_only"] = np.nan
    changed.loc["2024-04-01":, "future_only"] = 1e9
    new = build_national(changed)
    pd.testing.assert_frame_equal(old.loc[:"2024-03-01"], new.loc[:"2024-03-01"])
    assert new.loc["2024-04-01", "n_municipalities_known"] == 4
    # A column with no observed month is never in the known population.
    changed["never_observed"] = np.nan
    pd.testing.assert_frame_equal(new, build_national(changed))


def test_monthly_missing_share_uses_only_currently_known_population():
    panel = pd.DataFrame({"a": [np.nan, 10.0, np.nan],
                          "b": [np.nan, np.nan, 20.0],
                          "future": [np.nan, np.nan, np.nan]},
                         index=pd.date_range("2023-01-01", periods=3, freq="MS"))
    n = build_national(panel)
    assert n.n_municipalities_known.tolist() == [0, 1, 2]
    assert n.n_municipalities_used.tolist() == [0, 1, 1]
    assert n.n_missing.tolist() == [0, 0, 1]
    assert np.isnan(n.missing_share.iloc[0])
    assert n.missing_share.iloc[1:].tolist() == [0.0, 0.5]


def test_national_distribution_and_publication_date():
    panel = pd.DataFrame([[0.0, 10.0, 20.0, 30.0, 40.0]],
                         columns=list("abcde"), index=pd.to_datetime(["2024-02-01"]))
    n = build_national(panel, release_lag_months=2)
    assert n.iloc[0][["min", "p10", "q25", "median", "q75", "p90", "max"]].tolist() == [0, 4, 10, 20, 30, 36, 40]
    assert n.available_at.iloc[0] == pd.Timestamp("2024-04-30")


def test_national_preserves_missing_calendar_positions():
    panel = pd.DataFrame({"a": [10.0, 30.0]}, index=pd.to_datetime(["2023-01-01", "2023-03-01"]))
    n = build_national(panel)
    assert n.index.tolist() == list(pd.date_range("2023-01-01", periods=3, freq="MS"))
    assert np.isnan(n.N.iloc[1])
    assert n.n_municipalities_used.iloc[1] == 0
    assert n.n_municipalities_known.iloc[1] == 1
    assert n.missing_share.iloc[1] == 1


def test_ratio_exact_reconstruction_on_synthetic_data(panel):
    n = build_national(panel)
    ratios, diagnostics = ratio_panel(panel, n)
    restored = ratios.mul(n.N, axis=0)
    np.testing.assert_allclose(restored, panel)
    assert ratios.columns.equals(panel.columns)
    assert ratios.index.equals(panel.index)
    assert ratios.dtypes.eq("float64").all()
    assert diagnostics.n_ratio_finite.eq(3).all()
    assert diagnostics.denominator_status.eq("valid").all()


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf])
def test_invalid_national_denominator_remains_missing(panel, bad):
    n = build_national(panel)
    n.loc[n.index[5], "N"] = bad
    ratios, diagnostics = ratio_panel(panel, n)
    assert ratios.iloc[5].isna().all()
    assert len(ratios) == len(panel)
    assert diagnostics.n_ratio_missing_invalid_denominator.iloc[5] == 3
    assert diagnostics.denominator_status.iloc[5] != "valid"


def test_nonfinite_expense_not_used_as_ratio(panel):
    panel.iloc[1, 0] = np.inf
    n = build_national(panel)
    ratios, diagnostic = ratio_panel(panel, n)
    assert np.isnan(ratios.iloc[1, 0])
    assert n.n_municipalities_used.iloc[1] == 2
    assert diagnostic.n_expense_nonfinite.iloc[1] == 1


def test_ratio_overflow_is_reported_without_constant_substitution():
    panel = pd.DataFrame({"a": [1e308]}, index=pd.to_datetime(["2023-01-01"]))
    n = build_national(panel)
    n["N"] = 1e-308
    ratios, diagnostic = ratio_panel(panel, n)
    assert np.isnan(ratios.iloc[0, 0])
    assert diagnostic.n_ratio_overflow.iloc[0] == 1
    assert diagnostic.n_ratio_missing_invalid_denominator.iloc[0] == 0


@pytest.mark.parametrize("lag", [0, 1, 2])
@pytest.mark.parametrize("horizon", [1, 3, 6, 12])
def test_national_forecast_exact_baseline_adapter(panel, lag, horizon):
    n = build_national(panel, lag)
    origin = pd.Period("2024-05", freq="M")
    config = {"yearly_growth_window": 3, "yearly_growth_bounds": [0.5, 2.0]}
    history = get_prefix(n[["N"]], origin, lag)
    expected, flags = baseline_predict(history, lag + horizon, "SeasonalNaiveYoY", config)
    actual = national_forecast(n, origin, horizon, config, lag)
    assert actual["N_hat"] == expected[-1, 0]
    assert actual["feature_cutoff"] == period_end(origin - lag)
    assert actual["target_period"] == (origin + horizon).to_timestamp()
    assert actual["target_available_at"] == period_end(origin + horizon + lag)
    assert actual["status"] == ("fallback_last_value" if flags[-1, 0] else "native")
    assert actual["national_actual_used_for_prediction"] is False


def test_national_prediction_never_uses_actual_future_n(panel):
    origin = pd.Period("2024-05", freq="M")
    n = build_national(panel, 1)
    changed = panel.copy()
    changed.loc[changed.index > "2024-04-01"] = 1e8
    original = national_forecast(n, origin, 3, {}, 1)
    mutated = national_forecast(build_national(changed, 1), origin, 3, {}, 1)
    assert original["N_hat"] == mutated["N_hat"]
    assert original["status"] == mutated["status"]
    assert original["N_actual"] != mutated["N_actual"]
    r_hat = np.array([0.4, 1.8])
    np.testing.assert_array_equal(restore_local_prediction(original["N_hat"], r_hat),
                                  restore_local_prediction(mutated["N_hat"], r_hat))


def test_missing_national_history_has_explicit_failed_status(panel):
    panel[:] = np.nan
    result = national_forecast(build_national(panel), "2023-12", 1, {})
    assert result["status"] == "failed"
    assert result["effective_model"] == "none"
    assert np.isnan(result["N_hat"])
    assert result["reason"]


def test_national_fallback_is_explicit(panel):
    short = build_national(panel.iloc[:3])
    result = national_forecast(short, "2023-03", 1, {})
    assert result["status"] == "fallback_last_value"
    assert result["effective_model"] == "LastValue"


def test_national_mismatched_publication_lag_is_explicit_failure(panel):
    result = national_forecast(build_national(panel, 1), "2024-04", 1, {}, 0)
    assert result["status"] == "failed"
    assert np.isnan(result["N_hat"])
    assert "release_lag_months" in result["reason"]


def test_local_pairs_reuse_old_19_features_and_delta_formula(panel):
    n = build_national(panel)
    ratios, _ = ratio_panel(panel, n)
    expected_X, expected_y, expected_trace = build_direct_training(
        ratios, "2024-04", 3, release_lag_months=0, mode="legacy")
    X, delta, trace = build_local_training(
        panel, n, "2024-04", 3, release_lag_months=0, mode="legacy")
    pd.testing.assert_frame_equal(X, expected_X)
    np.testing.assert_array_equal(delta, expected_y)
    pd.testing.assert_frame_equal(trace[expected_trace.columns], expected_trace)
    assert len(X.columns) == 19
    assert "municipality_id" not in X
    assert "territory_id" not in X
    np.testing.assert_allclose(delta, trace.ratio_target_value - trace.ratio_anchor)
    np.testing.assert_allclose(trace.expense_target_value,
                               trace.target_value * trace.target_national_denominator)


def test_local_keys_match_expense_keys_when_all_denominators_valid(panel):
    _, _, expense_keys = build_direct_training(panel, "2024-06", 3, release_lag_months=0, mode="legacy")
    _, _, local_keys = build_local_training(panel, build_national(panel), "2024-06", 3,
                                          release_lag_months=0, mode="legacy")
    pd.testing.assert_frame_equal(expense_keys[PAIR_KEY], local_keys[PAIR_KEY])


@pytest.mark.parametrize("lag", [0, 1, 2])
def test_target_denominator_available_only_after_label_boundary(panel, lag):
    origin = pd.Period("2024-04", freq="M")
    n = build_national(panel, lag)
    _, _, trace = build_local_training(panel, n, origin, 1, release_lag_months=lag, mode="legacy")
    assert (trace.target_national_available_at <= period_end(origin)).all()
    assert (trace.anchor_national_available_at <= trace.historical_origin).all()
    assert trace.target_period.max() == (origin - lag).to_timestamp()
    assert trace.target_national_available_at.max() == period_end(origin)
    assert trace.target_national_available_at.equals(trace.target_available_at)
    assert (trace.target_period.dt.to_period("M") == trace.historical_origin.dt.to_period("M") + 1).all()
    assert (trace.feature_cutoff.dt.to_period("M") == trace.historical_origin.dt.to_period("M") - lag).all()


def test_change_later_than_current_cutoff_does_not_change_local_training(panel):
    origin = pd.Period("2024-04", freq="M")
    changed = panel.copy()
    changed.loc[changed.index > "2024-03-01"] = 9e8
    original = build_local_training(panel, build_national(panel, 1), origin, 3,
                                    release_lag_months=1, mode="legacy")
    mutated = build_local_training(changed, build_national(changed, 1), origin, 3,
                                   release_lag_months=1, mode="legacy")
    pd.testing.assert_frame_equal(original[0], mutated[0])
    np.testing.assert_array_equal(original[1], mutated[1])
    pd.testing.assert_frame_equal(original[2], mutated[2])


def test_historical_features_use_own_cutoff_not_current_origin(panel):
    n = build_national(panel, 1)
    changed = panel.copy()
    changed.loc[changed.index > "2023-06-01", "a"] *= 10
    X, _, trace = build_local_training(panel, n, "2024-04", 3, release_lag_months=1, mode="legacy")
    altered_X, _, altered_trace = build_local_training(changed, build_national(changed, 1), "2024-04", 3,
                                                      release_lag_months=1, mode="legacy")
    selected = trace.historical_origin.eq(pd.Timestamp("2023-07-31"))
    changed_selected = altered_trace.historical_origin.eq(pd.Timestamp("2023-07-31"))
    pd.testing.assert_frame_equal(X.loc[selected].reset_index(drop=True),
                                  altered_X.loc[changed_selected].reset_index(drop=True))
    # The now-known target ratio may change; this test makes no invariant y claim.


def test_ratio_missing_month_does_not_shift_features_or_horizon(panel):
    panel.loc["2023-02-01"] = 0.0  # N=0 -> all local ratios explicitly missing.
    n = build_national(panel)
    ratios, _ = ratio_panel(panel, n)
    X, anchor = direct_features(ratios.loc[:"2023-03-01"], pd.Timestamp("2023-06-01"))
    assert X.lag_2.isna().all()
    np.testing.assert_array_equal(X.lag_3.to_numpy(), ratios.loc["2023-01-01"].to_numpy())
    _, _, trace = build_local_training(panel, n, "2023-06", 3, release_lag_months=0, mode="legacy")
    selected = trace.historical_origin.eq(pd.Timestamp("2023-03-31"))
    assert trace.loc[selected, "target_period"].eq(pd.Timestamp("2023-06-01")).all()
    assert trace.loc[selected, "n_history_observations"].eq(2).all()
    assert np.isfinite(anchor).all()


@pytest.mark.parametrize("mode", ["legacy", "strict12"])
def test_annual_pairs_absent_in_december_2023(panel, mode):
    X, delta, trace = build_local_training(panel, build_national(panel), "2023-12", 12,
                                          release_lag_months=0, mode=mode)
    assert X.empty and len(delta) == 0 and trace.empty
    assert len(X.columns) == 19
    assert "target_national_available_at" in trace


def test_strict12_annual_pairs_absent_through_november_2024(panel):
    n = build_national(panel)
    _, _, trace = build_local_training(panel, n, "2024-11", 12, release_lag_months=0, mode="strict12")
    assert trace.empty
    _, _, next_trace = build_local_training(panel, n, "2024-12", 12, release_lag_months=0, mode="strict12")
    assert len(next_trace) == 3


def test_invalid_denominator_changes_pairs_transparently(panel):
    panel.loc["2023-02-01"] = 0.0
    n = build_national(panel)
    _, diag = ratio_panel(panel, n)
    _, _, trace = build_local_training(panel, n, "2023-04", 1, release_lag_months=0, mode="legacy")
    assert not trace.target_period.eq(pd.Timestamp("2023-02-01")).any()
    assert trace.target_national_denominator.gt(0).all()
    assert diag.loc["2023-02-01", "n_ratio_missing_invalid_denominator"] == 3


def test_denominator_availability_mismatch_is_rejected_before_training(panel):
    n = build_national(panel, 1)
    with pytest.raises(ValueError, match="publication lag"):
        build_local_training(panel, n, "2024-04", 1, release_lag_months=0, mode="legacy")
    n = build_national(panel)
    n.loc["2023-03-01", "available_at"] = pd.Timestamp("2025-01-31")
    with pytest.raises(ValueError, match="publication lag"):
        build_local_training(panel, n, "2024-04", 1, release_lag_months=0, mode="legacy")


def test_restoration_uses_only_national_forecast_and_clips_final_value():
    np.testing.assert_array_equal(restore_local_prediction(200.0, np.array([0.5, -0.3, 2])), [100.0, 0.0, 400.0])


@pytest.mark.parametrize("n_hat,ratios", [(np.nan, [1]), (np.inf, [1]), (-1, [1]), (1, [np.nan]), (1, [np.inf])])
def test_restoration_invalid_forecast_or_ratio_fails_explicitly(n_hat, ratios):
    with pytest.raises(ValueError, match="Restoration"):
        restore_local_prediction(n_hat, np.array(ratios))


def test_restoration_overflow_fails_explicitly():
    with pytest.raises(ValueError, match="overflow"):
        restore_local_prediction(1e308, np.array([1e308]))


@pytest.mark.parametrize("bad_lag", [-1, True, 0.5])
def test_invalid_release_lag_rejected(panel, bad_lag):
    with pytest.raises(ValueError, match="release_lag_months"):
        build_national(panel, bad_lag)


def test_national_bad_calendar_is_rejected(panel):
    with pytest.raises(ValueError, match="упорядочены"):
        build_national(panel.iloc[::-1])


def test_national_empty_panel_is_rejected(panel):
    with pytest.raises(ValueError, match="непустая"):
        build_national(panel.iloc[:0])
