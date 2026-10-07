"""E07a offline synthetic tests: prefix replay, availability and diagnostics."""
import copy

import numpy as np
import pandas as pd
import pytest

import sberforecast.early_warning_features as module
from sberforecast.early_warning_features import (
    DETECTORS, FEATURE_COLUMNS, PHASE_COLUMNS, audit_feature_columns,
    build_origin_macro_features, build_prefix_history_features, feature_dictionary,
)
from sberforecast.macro_forecast_features import CONSUMPTION, INFLATION
from sberforecast.news_features import FEATURE_COLUMNS as NEWS_COLUMNS, MISSING_COLUMNS
from sberforecast.online_detection import run_stream


PARAMS = {
    "CUSUM": {"selected": True, "parameters": {"allowance": .5, "threshold": 3}},
    "EWMA": {"selected": True, "parameters": {"alpha": .4, "threshold": 2.5}},
    "BOCPD": {"selected": True, "parameters": {
        "hazard": 1/24, "prior_alpha": 2, "prior_beta": 1, "prior_kappa": 1,
        "prior_mean": 0, "recent_run_max": 2, "threshold": .2}},
}
CONFIG = {"data": {"release_lag_months": 0}, "detector_preparation": {
    "warmup_observations": 4, "relative_scale_floor": .03,
    "absolute_scale_floor": 1, "cooldown_months": 2, "release_lag_months": 0}}


def samples(*origins, ids=None, index=None):
    return pd.DataFrame({"municipality_id": ids or ["21"] * len(origins),
                         "region_id": ["1"] * len(origins), "forecast_origin": list(origins)}, index=index)


def observations(values, start="2023-01", uid="21"):
    return pd.DataFrame({"municipality_id": [uid]*len(values),
                         "ds": pd.period_range(start, periods=len(values), freq="M").to_timestamp(), "y": np.asarray(values, dtype=float)})


def residuals(errors, start="2024-01", uid="21"):
    periods = pd.period_range(start, periods=len(errors), freq="M")
    return pd.DataFrame({"series_id": [uid]*len(errors), "observation_period": periods.astype(str),
        "y_true": np.asarray(errors, dtype=float)+100, "y_pred": [100.]*len(errors),
        "model": ["SeasonalNaiveYoY"]*len(errors), "horizon": [1]*len(errors),
        "forecast_origin": [(period-1).end_time.normalize() for period in periods],
        "history_cutoff": [(period-1).end_time.normalize() for period in periods]})


def build(obs=None, errors=None, query=None, config=None, params=None):
    return build_prefix_history_features(
        observations([100]*18) if obs is None else obs,
        residuals([0, 1, -1, 0, 10, 20]) if errors is None else errors,
        samples("2024-06-30") if query is None else query,
        CONFIG if config is None else config, PARAMS if params is None else params)


def forecast_row(indicator=INFLATION, year=2024, published="2024-01-15", **extra):
    row = dict(region_id="RU", geographic_level="national", indicator=indicator,
        reference_period=str(year), target_period=str(year), value=999., unit="percent_growth",
        published_at=published, available_at=published, vintage_id=f"edition-{year}-{published}",
        availability_status="A", source_url=f"https://official.invalid/{year}/{published}",
        forecast_issue_date=published, is_forecast=True,
        aggregation_basis="december_to_december" if indicator == INFLATION else "annual_real_volume_growth",
        forecast_low=np.nan, forecast_central=5., forecast_high=np.nan,
        forecast_producer="official", central_method="official_central")
    row.update(extra)
    return row


def test_empty_or_missing_series_preserves_every_key_and_index():
    query = samples("2023-12-31", "2024-06-30", "2024-06-30", ids=["21", "21", "1471"], index=[9, 4, 4])
    X, provenance = build(query=query)
    pd.testing.assert_frame_equal(X[query.columns], query)
    missing = X.iloc[-1]
    assert missing.expense_current_value_missing and missing.residual_current_value_missing
    assert missing.expense_observed_months_3m == missing.expense_n_history_observations == 0
    for method in DETECTORS:
        stem = "online_"+method.lower()+"_"
        assert missing[stem+"phase"] == "not_started"
        assert missing[stem+"calibration_count"] == missing[stem+"ready"] == 0
        assert missing[stem+"score_missing"] and missing[stem+"state_age_months_missing"]
    assert len(provenance) == len(query)*len(FEATURE_COLUMNS)
    assert provenance.groupby("sample_position").feature.nunique().eq(len(FEATURE_COLUMNS)).all()


def test_expense_calendar_ratios_strict_mean_and_last_age():
    obs = observations([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180])
    X, provenance = build(obs=obs)
    row = X.iloc[0]
    assert row.expense_current_value == row.expense_last_value == 180
    assert row.expense_last_age_months == 0 and row.expense_previous_month_value == 170
    assert row.expense_mom_ratio == pytest.approx(180/170)
    assert row.expense_yoy_ratio == 3
    assert row.expense_mean_3m == 170
    assert row.expense_observed_months_3m == 3 and row.expense_n_history_observations == 18
    assert row.residual_current_value == 20 and row.residual_mean_3m == 10
    available = provenance.max_available_at.notna()
    assert provenance.loc[available, "max_available_at"].le(provenance.loc[available, "as_of_utc"]).all()
    assert provenance.as_of_utc.iloc[0] == pd.Timestamp("2024-06-29T21:00:00Z")


def test_missing_current_month_is_never_filled_by_stale_value_or_short_mean():
    obs = observations([10, 20, np.nan], start="2024-01")
    errors = residuals([1, 2, np.nan])
    X, _ = build(obs=obs, errors=errors, query=samples("2024-03-31"))
    row = X.iloc[0]
    assert np.isnan(row.expense_current_value) and row.expense_last_value == 20
    assert row.expense_last_age_months == row.residual_last_age_months == 1
    assert np.isnan(row.expense_mean_3m) and np.isnan(row.residual_mean_3m)
    assert row.expense_observed_months_3m == row.residual_observed_months_3m == 2
    assert row.online_cusum_phase == "missing" and row.online_cusum_state_age_months == 1


def test_zero_denominator_is_missing_while_zero_current_is_observed():
    X, _ = build(obs=observations([0, 0], start="2024-01"), query=samples("2024-02-29"))
    row = X.iloc[0]
    assert row.expense_current_value == 0 and not row.expense_current_value_missing
    assert np.isnan(row.expense_mom_ratio) and row.expense_mom_ratio_missing
    assert np.isnan(row.expense_yoy_ratio) and row.expense_yoy_ratio_missing


def test_moscow_midnight_boundary_is_inclusive_with_utc_equivalent_origins():
    X, _ = build(obs=observations([50], start="2024-02"), errors=residuals([2], start="2024-02"),
                 query=samples("2024-02-28T20:59:59Z", "2024-02-28T21:00:00Z"))
    assert X.expense_current_value.isna().tolist() == [True, False]
    assert X.residual_current_value.isna().tolist() == [True, False]
    assert X.online_cusum_calibration_count.tolist() == [0, 1]


def test_future_values_and_future_missing_rows_do_not_change_features_or_provenance():
    obs, errors = observations([100]*18), residuals([0, 1, -1, 0, 10, 20])
    query = samples("2023-12-31", "2024-02-29", "2024-04-30")
    old = build(obs=obs, errors=errors, query=query)
    mutated_obs, mutated_errors = obs.copy(), errors.copy()
    mutated_obs.loc[mutated_obs.ds.ge("2024-05-01"), "y"] = [99999, np.nan]
    mutated_errors.loc[mutated_errors.observation_period.ge("2024-05"), "y_true"] = [88888, np.nan]
    changed = build(obs=mutated_obs, errors=mutated_errors, query=query)
    pd.testing.assert_frame_equal(old[0], changed[0])
    pd.testing.assert_frame_equal(old[1], changed[1])


def test_first_future_observation_for_missing_MO_never_creates_old_missing_facts():
    obs = observations([100]*18)
    query = samples("2024-03-31", ids=["1471"])
    old = build(obs=obs, query=query)
    changed = build(obs=pd.concat([obs, observations([500], start="2024-06", uid="1471")]), query=query)
    pd.testing.assert_frame_equal(old[0], changed[0])
    pd.testing.assert_frame_equal(old[1], changed[1])


def test_each_origin_replay_receives_only_its_available_prefix_and_never_future_warmup(monkeypatch):
    seen = []
    original = module.run_stream

    def checked(table, method, params, preparation):
        assert pd.PeriodIndex(table.observation_period, freq="M").max() <= preparation["as_of"].to_period("M")
        seen.append((method, preparation["as_of"], len(table)))
        return original(table, method, params, preparation)

    monkeypatch.setattr(module, "run_stream", checked)
    X, _ = build(query=samples("2024-02-29", "2024-04-30", "2024-05-31"))
    assert len(seen) == 9
    assert X.online_cusum_calibration_count.tolist() == [2, 4, 4]
    assert X.online_cusum_ready.tolist() == [0, 1, 1]
    assert X.online_cusum_phase.tolist() == ["warmup", "warmup", "monitoring"]
    assert X.online_cusum_score.isna().tolist() == [True, True, False]


@pytest.mark.parametrize("method", DETECTORS)
def test_monitoring_scores_equal_existing_e04_runner_on_that_prefix(method):
    errors = residuals([0, 1, -1, 0, 10, -20, 40])
    origin = "2024-06-30"
    direct = run_stream(errors.loc[errors.observation_period.le("2024-06")], method,
                        PARAMS[method]["parameters"], dict(CONFIG["detector_preparation"], as_of=origin))
    X, provenance = build(errors=errors, query=samples(origin))
    row, expected = X.iloc[0], direct.iloc[-1]
    stem = "online_"+method.lower()+"_"
    assert row[stem+"score"] == pytest.approx(expected.score)
    assert row[stem+"is_alarm"] == float(expected.is_alarm)
    assert row[stem+"phase"] == expected.phase
    assert row[stem+"last_score"] == row[stem+"score"]
    assert provenance.loc[provenance.source.eq("E04_prefix_detector"), "parameters_assumption"].str.contains("not verified historic parameter selection").all()


def test_absent_trailing_residual_keeps_last_score_with_explicit_staleness():
    X, _ = build(errors=residuals([0, 1, -1, 0, 10]), query=samples("2024-07-31"))
    row = X.iloc[0]
    assert row.online_cusum_phase == "missing_current_month"
    assert np.isnan(row.online_cusum_score) and np.isnan(row.online_cusum_is_alarm)
    assert np.isfinite(row.online_cusum_last_score) and row.online_cusum_state_age_months == 2
    assert row.residual_last_value == 10 and row.residual_last_age_months == 2


def test_saved_e04_missing_forecast_placeholders_keep_samples_and_do_not_invent_issue_dates():
    errors = residuals([0, 1]+[np.nan]*10, uid="1471")
    placeholders = errors.observation_period.ge("2024-03")
    errors.loc[placeholders, ["model", "horizon", "forecast_origin", "history_cutoff", "y_true", "y_pred"]] = None
    query = samples("2024-02-29", "2024-03-31", "2024-11-30", ids=["1471"]*3)
    X, lineage = build(obs=observations([20, 30], start="2024-01", uid="1471"), errors=errors, query=query)
    pd.testing.assert_frame_equal(X[query.columns], query)
    assert X.residual_current_value.isna().tolist() == [False, True, True]
    assert X.residual_last_value.tolist() == [1, 1, 1]
    assert X.online_cusum_calibration_count.tolist() == [2, 2, 2]
    assert X.online_cusum_ready.eq(0).all()
    assert X.online_cusum_state_age_months.tolist() == [0, 1, 9]
    normalized = module._residuals(errors, 0)
    assert normalized.loc[placeholders, "forecast_available_at"].isna().all()
    assert len(normalized) == 12
    without_placeholders = build(obs=observations([20, 30], start="2024-01", uid="1471"),
                                 errors=errors.loc[~placeholders], query=query)
    pd.testing.assert_frame_equal(X, without_placeholders[0])
    pd.testing.assert_frame_equal(lineage, without_placeholders[1])


@pytest.mark.parametrize("field,message", [("model", "SeasonalNaiveYoY"), ("horizon", "h=1"),
    ("forecast_origin", "issued before"), ("history_cutoff", "history cutoff")])
def test_finite_saved_prediction_still_requires_present_valid_metadata(field, message):
    errors = residuals([1])
    errors[field] = None
    with pytest.raises(ValueError, match=message):
        build(errors=errors)


def test_delayed_or_unknown_expense_and_residual_facts_are_excluded():
    obs = observations([10, 20, 30], start="2024-01")
    obs["available_at"] = ["2024-01-31", "2024-04-15", None]
    errors = residuals([0, 1, 2])
    errors["fact_available_at"] = ["2024-01-31", "2024-04-15", None]
    X, provenance = build(obs=obs, errors=errors, query=samples("2024-03-31", "2024-04-30"))
    assert X.expense_last_value.tolist() == [10, 20]
    assert X.residual_last_value.tolist() == [0, 1]
    assert X.online_cusum_calibration_count.tolist() == [1, 2]
    assert X.expense_n_history_observations.tolist() == [1, 2]
    assert provenance.loc[provenance.sample_position.eq(1), "max_available_at"].max() == pd.Timestamp("2024-04-14T21:00:00Z")


def test_same_values_long_expense_input_and_load_data_input_are_equivalent():
    obs = observations([100]*18)
    long = obs.rename(columns={"municipality_id": "series_id", "y": "value"})
    long["observation_period"] = long.ds.dt.to_period("M").astype(str)
    first, second = build(obs=obs), build(obs=long.drop(columns="ds"))
    pd.testing.assert_frame_equal(first[0], second[0])
    pd.testing.assert_frame_equal(first[1], second[1])


@pytest.mark.parametrize("column,value,message", [
    ("model", "Prophet", "SeasonalNaiveYoY"), ("horizon", 3, "h=1"),
    ("forecast_origin", "2024-07-31", "issued before"),
    ("history_cutoff", "2024-07-31", "history cutoff"),
    ("fact_available_at", "2023-01-01", "completed month"),
    ("breakpoint_month", "2024-02", "Offline breakpoints"),
    ("is_hindsight_diagnostic", True, "Hindsight"),
])
def test_conflicting_or_offline_residual_inputs_are_rejected(column, value, message):
    errors = residuals([0, 1, -1, 0, 10, 20])
    errors[column] = value
    with pytest.raises(ValueError, match=message):
        build(errors=errors)


def test_unverified_duplicate_observation_vintage_is_rejected():
    obs = observations([10], start="2024-01")
    with pytest.raises(ValueError, match="Duplicate expense"):
        build(obs=pd.concat([obs, obs]))


def test_selected_parameters_and_config_are_required_and_never_mutated():
    cfg, params = copy.deepcopy(CONFIG), copy.deepcopy(PARAMS)
    build(config=cfg, params=params)
    assert cfg == CONFIG and params == PARAMS
    wrong = copy.deepcopy(CONFIG)
    wrong["data"]["release_lag_months"] = 1
    with pytest.raises(ValueError, match="L=0"):
        build(config=wrong)
    with pytest.raises(ValueError, match="fixed E04 selected parameters"):
        build(params={"CUSUM": PARAMS["CUSUM"]})


def test_empty_samples_and_empty_sources_are_valid_without_model_fitting():
    query = samples()
    X, lineage = build(obs=observations([]), errors=residuals([]), query=query)
    assert X.empty and lineage.empty
    assert set(FEATURE_COLUMNS+PHASE_COLUMNS).issubset(X)
    one, _ = build(obs=observations([]), errors=residuals([]), query=samples("2024-01-31"))
    assert one.expense_current_value.isna().all() and one.online_cusum_ready.eq(0).all()


def test_macro_is_current_origin_year_A_only_and_reuses_E05c_source_ids():
    rows = [forecast_row(year=2023, published="2023-12-15", forecast_central=8),
            forecast_row(year=2024, published="2023-12-15", forecast_central=4),
            forecast_row(CONSUMPTION, published="2023-12-15", forecast_central=np.nan,
                         forecast_low=2, forecast_high=4),
            forecast_row(year=2024, published="2024-01-10", forecast_central=99, availability_status="B")]
    source_ids = {row["source_url"]: "e05c-"+str(index) for index, row in enumerate(rows)}
    query = samples("2023-12-31", "2024-01-31")
    X, lineage = build_origin_macro_features(query, pd.DataFrame(rows), source_ids)
    assert X["macro_current_year_"+INFLATION].tolist() == [8, 4]
    assert X["macro_current_year_"+CONSUMPTION].isna().tolist() == [True, False]
    assert X["macro_current_year_"+CONSUMPTION].iloc[-1] == 3  # annual growth, never divided by 12
    assert lineage.target_year.tolist() == [2023, 2023, 2024, 2024]
    assert lineage.loc[lineage.source_id.ne(""), "source_reused_from_E05c"].all()
    assert set(lineage.source_status) <= {"", "A"}
    known = lineage.max_available_at.notna()
    assert lineage.loc[known, "max_available_at"].le(lineage.loc[known, "as_of_utc"]).all()
    assert lineage.target_rule.str.contains("not_H_or_k").all()


def test_future_macro_revision_cannot_change_features_or_lineage_at_O():
    row = forecast_row()
    future = forecast_row(published="2024-03-01", forecast_central=99)
    query = samples("2024-01-31")
    old = build_origin_macro_features(query, pd.DataFrame([row]))
    changed = build_origin_macro_features(query, pd.DataFrame([row, future]))
    pd.testing.assert_frame_equal(old[0], changed[0])
    pd.testing.assert_frame_equal(old[1], changed[1])


def test_macro_missing_current_year_never_substitutes_next_year_or_actual():
    rows = [forecast_row(year=2025), forecast_row(is_forecast=False)]
    X, lineage = build_origin_macro_features(samples("2024-01-31"), pd.DataFrame(rows))
    assert X["macro_current_year_"+INFLATION].isna().all()
    assert X["macro_current_year_"+INFLATION+"_missing"].eq(1).all()
    assert lineage.source_id.eq("").all()


def test_macro_empty_samples_returns_empty_features_and_lineage():
    X, lineage = build_origin_macro_features(samples(), pd.DataFrame([forecast_row()]))
    assert X.empty and lineage.empty


def test_macro_auto_url_hash_is_not_claimed_as_E05c_registry_reuse():
    _, lineage = build_origin_macro_features(samples("2024-01-31"), pd.DataFrame([forecast_row()]))
    chosen = lineage.loc[lineage.source_id.ne("")]
    assert chosen.source_id.str.startswith("url_sha256_").all()
    assert not chosen.source_reused_from_E05c.any()


def test_dictionary_describes_every_added_history_phase_flag_and_macro_field():
    X, _ = build()
    macro, _ = build_origin_macro_features(samples("2024-01-31"), pd.DataFrame([forecast_row()]))
    dictionary = feature_dictionary().set_index("feature")
    assert dictionary.index.is_unique
    assert set(X.columns)-set(samples("2024-06-30").columns) <= set(dictionary.index)
    assert set(macro.columns)-set(samples("2024-01-31").columns) <= set(dictionary.index)
    assert dictionary.loc["expense_mean_3m", "definition"].find("all three") >= 0
    assert dictionary.loc["expense_mom_ratio", "definition"].find("positive denominator") >= 0
    assert dictionary.loc["macro_current_year_"+INFLATION, "availability_rule"].find("current calendar year of O") >= 0


def test_news_all_52_columns_are_audited_without_selection_or_independence_claims():
    feature = pd.concat([samples("2024-01-31", "2024-02-29", "2024-03-31", ids=[str(uid)]*3)
                         for uid in range(64)], ignore_index=True)
    for index, name in enumerate(NEWS_COLUMNS):
        feature[name] = np.tile([0., 1., 0.], 64) if index < 2 else np.nan if index == 2 else 0.
        feature[name+"_missing"] = feature[name].isna()
    unchanged = feature.copy(deep=True)
    audit = audit_feature_columns(feature, NEWS_COLUMNS+MISSING_COLUMNS).set_index("feature")
    pd.testing.assert_frame_equal(feature, unchanged)
    assert len(audit) == 52
    assert audit.origin_dates_counted_once.eq(3).all()
    assert audit.unique_temporal_feature_vectors.eq(2).all()
    assert audit.identical_across_municipalities_within_origin.all()
    assert audit.loc[NEWS_COLUMNS[1], "exact_duplicate_of"] == NEWS_COLUMNS[0]
    assert audit.loc[NEWS_COLUMNS[2], "all_missing"]
    assert audit.loc[NEWS_COLUMNS[3], "finite_constant"]
    assert audit.interpretation.str.contains("no column removed").all()
    assert not any("independent_temporal" in name for name in audit.columns)


def test_empty_training_audit_preserves_52_descriptions_and_zero_dates():
    empty = samples()
    for name in NEWS_COLUMNS+MISSING_COLUMNS:
        empty[name] = pd.Series(dtype=float)
    audit = audit_feature_columns(empty, NEWS_COLUMNS+MISSING_COLUMNS)
    assert len(audit) == 52
    assert audit.rows.eq(0).all() and audit.origin_dates_counted_once.eq(0).all()
    assert audit.all_missing.all() and ~audit.finite_constant.any()
    assert audit.unique_temporal_feature_vectors.eq(0).all()


def test_local_vectors_are_not_reported_as_one_national_vector_per_origin():
    query = samples("2024-01-31", "2024-01-31", ids=["21", "22"])
    query["regional"] = [1., 2.]
    audit = audit_feature_columns(query, ["regional"])
    assert not audit.identical_across_municipalities_within_origin.any()
    assert audit.origin_dates_counted_once.isna().all()
    assert audit.unique_temporal_feature_vectors.isna().all()


def test_numeric_boolean_duplicate_detection_requires_identical_missing_positions():
    frame = samples("2024-01-31", "2024-02-29")
    frame["zero"], frame["false"], frame["some_missing"] = [0., 0.], [False, False], [0., np.nan]
    audit = audit_feature_columns(frame, ["zero", "false", "some_missing"]).set_index("feature")
    assert audit.loc["false", "exact_duplicate_of"] == "zero"
    assert audit.loc["some_missing", "exact_duplicate_of"] == ""
    assert audit.loc["some_missing", "finite_constant"]
    assert not audit.loc["some_missing", "exact_constant_including_missing"]


def test_training_only_subset_audit_does_not_use_future_column_variation():
    frame = samples("2024-01-31", "2024-02-29", "2024-08-31")
    frame["news"] = [0., 0., 1.]
    train = audit_feature_columns(frame.iloc[:2], ["news"]).iloc[0]
    full = audit_feature_columns(frame, ["news"]).iloc[0]
    assert train.finite_constant and not full.finite_constant
    assert train.origin_dates_counted_once == 2 and full.origin_dates_counted_once == 3
