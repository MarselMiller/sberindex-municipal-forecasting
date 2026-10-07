"""E07b synthetic, offline checks of bounded-memory full-panel features."""
import copy

import numpy as np
import pandas as pd
import pytest

from sberforecast import early_warning_features as old
from sberforecast import early_warning_full_panel_features as module
from sberforecast.early_warning_full_panel_features import (
    EXTRA_COLUMNS, FEATURE_GROUPS, MACRO_COLUMNS, NUMERIC_FEATURE_COLUMNS,
    build_panel_features, feature_dictionary, feature_groups,
)
from sberforecast.macro_forecast_features import CONSUMPTION, INFLATION
from sberforecast.news_features import EVENT_COLUMNS, build_news_features


PARAMS = {
    "CUSUM": {"selected": True, "parameters": {"allowance": .5, "threshold": 3}},
    "EWMA": {"selected": True, "parameters": {"alpha": .4, "threshold": 2.5}},
    "BOCPD": {"selected": True, "parameters": {
        "hazard": 1/24, "prior_alpha": 2, "prior_beta": 1, "prior_kappa": 1,
        "prior_mean": 0, "recent_run_max": 2, "threshold": .2}},
}
CONFIG = {"data": {"release_lag_months": 0, "first_month": "2023-01"},
    "runtime": {"feature_batch_size": 2, "memory_stop_gib": 5.5},
    "detector_preparation": {"warmup_observations": 4, "relative_scale_floor": .03,
        "absolute_scale_floor": 1, "cooldown_months": 2, "release_lag_months": 0}}


def observations(values=None, uid="21", start="2023-01"):
    values = list(range(10, 190, 10)) if values is None else values
    return pd.DataFrame({"municipality_id": [uid]*len(values),
        "ds": pd.period_range(start, periods=len(values), freq="M").to_timestamp(),
        "y": np.asarray(values, dtype=float)})


def residuals(errors=None, uid="21", start="2024-01"):
    errors = [0, 1, -1, 0, 10, 20] if errors is None else errors
    months = pd.period_range(start, periods=len(errors), freq="M")
    return pd.DataFrame({"series_id": [uid]*len(errors), "observation_period": months.astype(str),
        "y_true": np.asarray(errors, dtype=float)+100, "y_pred": [100.]*len(errors),
        "model": ["SeasonalNaiveYoY"]*len(errors), "horizon": [1]*len(errors),
        "forecast_origin": [(month-1).end_time.normalize() for month in months],
        "history_cutoff": [(month-1).end_time.normalize() for month in months]})


def samples():
    return pd.DataFrame({"municipality_id": ["21", "1471", "21", "1471"],
        "region_id": ["1", "2", "1", "2"],
        "forecast_origin": ["2024-04-30", "2024-04-30", "2024-06-30", "2024-06-30"],
        "eligible_at_origin": [True, False, True, False]}, index=[8, 8, 3, 2])


def macro_row(indicator=INFLATION, published="2024-01-15", year=2024, **updates):
    result = dict(region_id="RU", geographic_level="national", indicator=indicator,
        reference_period=str(year), target_period=str(year), value=999., unit="percent_growth",
        published_at=published, available_at=published, vintage_id="issue-"+published,
        availability_status="A", source_url="https://official.invalid/"+published,
        forecast_issue_date=published, is_forecast=True,
        aggregation_basis="december_to_december" if indicator == INFLATION else "annual_real_volume_growth",
        forecast_low=np.nan, forecast_central=5., forecast_high=np.nan,
        forecast_producer="official", central_method="official_central")
    result.update(updates)
    return result


def document(identity="d1", available="2024-03-20T00:00:00Z", **updates):
    result = dict(document_id=identity, canonical_event_id="event-"+identity,
        published_at=available, available_at=available, geography_level="national",
        region_id="RU", municipality_id="", topic="macro_policy", event_type="decision",
        direction="negative", availability_status="confirmed_historical")
    result.update(updates)
    return result


def build(obs=None, errors=None, query=None, config=None, macro=None, documents=None, progress=None):
    return build_panel_features(observations() if obs is None else obs,
        residuals() if errors is None else errors, samples() if query is None else query,
        CONFIG if config is None else config, PARAMS,
        pd.DataFrame([macro_row()]) if macro is None else macro,
        None, pd.DataFrame([document()]) if documents is None else documents, progress)


def test_exact_frozen_pilot_features_and_every_input_key_survive():
    query = samples()
    wide, coverage, dictionary = build()
    baseline, _ = old.build_prefix_history_features(observations(), residuals(), query, CONFIG, PARAMS)
    pd.testing.assert_frame_equal(wide[baseline.columns], baseline)
    macro, _ = old.build_origin_macro_features(query, pd.DataFrame([macro_row()]))
    pd.testing.assert_frame_equal(wide[macro.columns], macro)
    frozen_news = build_news_features(pd.DataFrame([document()]), query)
    pd.testing.assert_frame_equal(wide[frozen_news.columns], frozen_news)
    pd.testing.assert_frame_equal(wide[query.columns], query)
    assert len(dictionary) == 141 and len(coverage) == 282
    assert coverage.dependency_violations.eq(0).all()
    known = coverage.max_dependency_available_at.notna()
    assert coverage.loc[known, "max_dependency_available_at"].le(coverage.loc[known, "as_of_utc"]).all()
    assert wide.attrs["no_future_cohort_filter"]


def test_calendar_changes_and_population_volatility_are_computed_without_fill():
    wide, _, _ = build()
    row = wide.loc[wide.municipality_id.eq("21")].iloc[-1]
    assert row.expense_change_1m == 10
    assert row.residual_previous_month_value == 10 and row.residual_change_1m == 10
    assert row.expense_std_3m == pytest.approx(np.std([160, 170, 180], ddof=0))
    assert row.residual_std_3m == pytest.approx(np.std([0, 10, 20], ddof=0))
    assert row.expense_cv_3m == pytest.approx(np.std([160, 170, 180], ddof=0)/170)
    assert row.online_cusum_ready == 1 and row.online_cusum_phase == "monitoring"


def test_missing_calendar_month_does_not_use_latest_stale_value_for_change_or_std():
    obs = observations([10, 20, np.nan], start="2024-04")
    errors = residuals([1, np.nan, 3], start="2024-04")
    wide, _, _ = build(obs=obs, errors=errors)
    row = wide.loc[wide.municipality_id.eq("21")].iloc[-1]
    assert row.expense_last_value == 20 and row.expense_last_age_months == 1
    assert np.isnan(row.expense_change_1m) and np.isnan(row.expense_std_3m)
    assert np.isnan(row.residual_change_1m) and np.isnan(row.residual_std_3m)
    assert row.residual_change_1m_missing and row.expense_std_3m_missing


def test_zero_expenses_have_zero_population_std_but_missing_coefficient_of_variation():
    wide, _, _ = build(obs=observations([0, 0, 0], start="2024-04"))
    row = wide.loc[wide.municipality_id.eq("21")].iloc[-1]
    assert row.expense_std_3m == row.expense_change_1m == 0
    assert np.isnan(row.expense_cv_3m) and row.expense_cv_3m_missing


def test_future_label_months_macro_and_news_cannot_change_past_rows_or_coverage():
    query = samples().iloc[:2]
    baseline = build(query=query)
    obs, errors = observations(), residuals()
    obs.loc[obs.ds.ge("2024-05-01"), "y"] = [9999, np.nan]
    errors.loc[errors.observation_period.ge("2024-05"), "y_true"] = [8888, np.nan]
    future_macro = macro_row(published="2024-05-01", forecast_central=100)
    future_news = document("future", available="2024-05-01T00:00:00Z")
    changed = build(obs=obs, errors=errors, query=query,
        macro=pd.DataFrame([macro_row(), future_macro]), documents=pd.DataFrame([document(), future_news]))
    for previous, later in zip(baseline, changed):
        pd.testing.assert_frame_equal(previous, later)


def test_first_future_row_for_absent_municipality_cannot_change_old_features_or_coverage():
    query = samples().iloc[:2]
    previous = build(query=query)
    changed = build(obs=pd.concat([observations(), observations([999], uid="1471", start="2024-06")]), query=query)
    for first, second in zip(previous, changed):
        pd.testing.assert_frame_equal(first, second)


def test_batch_sizes_and_repeated_runs_are_deterministic():
    one, two = copy.deepcopy(CONFIG), copy.deepcopy(CONFIG)
    one["runtime"]["feature_batch_size"] = 1
    two["runtime"]["feature_batch_size"] = 64
    first, second, repeated = build(config=one), build(config=two), build(config=one)
    for index in range(3):
        pd.testing.assert_frame_equal(first[index], second[index])
        pd.testing.assert_frame_equal(first[index], repeated[index])


def test_group_definitions_are_numeric_fixed_and_no_full_history_selection_occurs():
    wide, _, dictionary = build()
    assert {key: len(value) for key, value in feature_groups().items()} == dict(A=40, B=36, C=10, D=52)
    assert len(NUMERIC_FEATURE_COLUMNS) == len(set(NUMERIC_FEATURE_COLUMNS)) == 138
    assert set(NUMERIC_FEATURE_COLUMNS).issubset(wide)
    assert set(EXTRA_COLUMNS).issubset(FEATURE_GROUPS["A"])
    assert not set(old.PHASE_COLUMNS) & set(NUMERIC_FEATURE_COLUMNS)
    assert dictionary.model_candidate.sum() == 138
    assert dictionary.selection_rule.str.contains("training rows only").all()
    # Every candidate stays present, including fully missing/constant features.
    missing = wide.loc[wide.municipality_id.eq("1471")]
    assert missing.residual_current_value.isna().all()
    assert "regional_news_count_30d" in wide and wide.regional_news_count_30d.eq(0).all()


@pytest.mark.parametrize("source", ["observations", "residuals", "samples"])
def test_offline_breakpoint_fields_are_forbidden_everywhere(source):
    arguments = dict(obs=observations(), errors=residuals(), query=samples())
    key = dict(observations="obs", residuals="errors", samples="query")[source]
    arguments[key]["breakpoint_month"] = "2024-06"
    with pytest.raises(ValueError, match="Offline breakpoints"):
        build(**arguments)


def test_saved_missing_forecast_metadata_placeholders_remain_missing_without_future_calibration():
    errors = residuals([0, 1]+[np.nan]*10, uid="1471")
    absent = errors.observation_period.ge("2024-03")
    errors.loc[absent, ["model", "horizon", "forecast_origin", "history_cutoff", "y_true", "y_pred"]] = None
    wide, _, _ = build(errors=pd.concat([residuals(), errors]))
    missing = wide.loc[wide.municipality_id.eq("1471")]
    assert missing.residual_current_value.isna().all()
    assert missing.residual_change_1m.isna().all()
    assert missing.online_cusum_calibration_count.eq(2).all()
    assert missing.online_cusum_ready.eq(0).all()
    assert missing.online_cusum_score.isna().all()


def test_explicit_delayed_fact_availability_controls_extra_features_and_coverage():
    obs, errors = observations(), residuals()
    obs["available_at"] = obs.ds.dt.to_period("M").dt.end_time.dt.normalize()
    obs.loc[obs.ds.ge("2024-05-01"), "available_at"] = pd.Timestamp("2024-07-31")
    errors["fact_available_at"] = pd.PeriodIndex(errors.observation_period, freq="M").end_time.normalize()
    errors.loc[errors.observation_period.ge("2024-05"), "fact_available_at"] = pd.Timestamp("2024-07-31")
    wide, coverage, _ = build(obs=obs, errors=errors)
    current = wide.loc[wide.municipality_id.eq("21")].iloc[-1]
    assert np.isnan(current.residual_change_1m) and np.isnan(current.expense_std_3m)
    assert current.online_cusum_calibration_count == 4 and current.online_cusum_score_missing
    checked = coverage.loc[coverage.forecast_origin.eq("2024-06-30") & coverage.group.isin(["A", "B"])]
    assert checked.max_dependency_available_at.max() == pd.Timestamp("2024-04-29T21:00:00Z")


def test_provenance_batches_save_global_positions_and_exact_values_with_bounded_call_size(tmp_path, monkeypatch):
    cfg = copy.deepcopy(CONFIG)
    cfg["runtime"]["feature_batch_size"] = 1
    cfg["output_dir"] = str(tmp_path / "new_e07b_output")
    cfg["feature_assembly"] = {"provenance_dir": "feature_provenance"}
    seen, progress = [], []
    original = old.build_prefix_history_features

    def checked(observations, residuals, samples, config, selected_parameters):
        seen.append(samples.municipality_id.nunique())
        return original(observations, residuals, samples, config, selected_parameters)

    monkeypatch.setattr(old, "build_prefix_history_features", checked)
    wide, coverage, _ = build(config=cfg, progress=progress.append)
    assert seen == [1, 1]
    files = wide.attrs["provenance_files"]
    assert len(files) == 4 and coverage.attrs["provenance_files"] == files
    history_parts = [pd.read_csv(item["path"]) for item in files if item["kind"] == "history"]
    stored = pd.concat(history_parts, ignore_index=True)
    assert len(stored) == len(samples())*38
    assert stored.groupby("sample_position").feature.nunique().eq(38).all()
    assert sorted(stored.sample_position.unique()) == [0, 1, 2, 3]
    assert stored.loc[stored.sample_position.eq(2) & stored.feature.eq("residual_change_1m"), "value"].iloc[0] == 10
    assert progress[0]["stage"] == "feature_assembly_start"
    assert progress[-1]["stage"] == "feature_assembly_complete"
    # Existing experiment contents must not be overwritten on a second call.
    with pytest.raises(FileExistsError):
        build(config=cfg)


def test_no_disk_provenance_by_default_and_config_sources_are_not_mutated(tmp_path):
    cfg = copy.deepcopy(CONFIG)
    cfg["output_dir"] = str(tmp_path / "unused_output")
    obs, errors, query = observations(), residuals(), samples()
    unchanged = [frame.copy(deep=True) for frame in [obs, errors, query]]
    previous_config = copy.deepcopy(cfg)
    wide, _, _ = build(obs=obs, errors=errors, query=query, config=cfg)
    assert not (tmp_path / "unused_output").exists()
    assert wide.attrs["provenance_files"] == []
    assert cfg == previous_config
    for actual, expected in zip([obs, errors, query], unchanged):
        pd.testing.assert_frame_equal(actual, expected)


def test_disk_provenance_cannot_escape_new_experiment_output_dir(tmp_path):
    cfg = copy.deepcopy(CONFIG)
    cfg["output_dir"] = str(tmp_path / "new")
    cfg["feature_assembly"] = {"provenance_dir": "../old"}
    with pytest.raises(ValueError, match="child of the new output_dir"):
        build(config=cfg)
    assert not (tmp_path / "old").exists()


def test_memory_projection_stops_before_prefix_stream_or_disk_creation(tmp_path, monkeypatch):
    cfg = copy.deepcopy(CONFIG)
    cfg["runtime"]["memory_stop_gib"] = .01
    cfg["output_dir"] = str(tmp_path / "must_not_be_created")
    cfg["feature_assembly"] = {"provenance_dir": "provenance"}

    def forbidden(*args, **kwargs):
        pytest.fail("A prefix stream started after the projected memory limit")

    monkeypatch.setattr(old, "build_prefix_history_features", forbidden)
    with pytest.raises(MemoryError, match="Projected feature assembly"):
        build(config=cfg)
    assert not (tmp_path / "must_not_be_created").exists()


def test_news_dependency_bound_is_honest_scoped_and_cannot_use_future_or_unknown_geography():
    rows = [document(), document("unknown", "2024-06-25T00:00:00Z", geography_level="unknown"),
        document("other_region", "2024-06-26T00:00:00Z", geography_level="region", region_id="99"),
        document("future", "2024-07-01T00:00:00Z"),
        document("same_region", "2024-06-15T00:00:00Z", geography_level="region", region_id="1")]
    wide, coverage, _ = build(documents=pd.DataFrame(rows))
    checked = coverage.loc[coverage.group.eq("D") & coverage.forecast_origin.eq("2024-06-30")]
    assert checked.max_dependency_available_at.eq(pd.Timestamp("2024-06-15T00:00:00Z")).all()
    assert checked.dependency_audit.eq("conservative_scoped_first_known_snapshot_upper_bound").all()
    local = wide.loc[wide.forecast_origin.eq("2024-06-30")]
    assert local.regional_news_count_30d.tolist() == [1, 0]
    assert local.national_news_count_30d.eq(0).all()


def test_empty_sources_and_samples_preserve_complete_candidate_schema():
    query = samples().iloc[:0]
    wide, coverage, dictionary = build(obs=observations([]), errors=residuals([]), query=query,
        documents=pd.DataFrame(columns=EVENT_COLUMNS))
    assert wide.empty and coverage.empty
    assert set(NUMERIC_FEATURE_COLUMNS).issubset(wide)
    assert len(dictionary) == 141


def test_empty_partial_news_corpus_keeps_counts_and_missing_denominator_rules():
    wide, coverage, _ = build(documents=pd.DataFrame(columns=EVENT_COLUMNS))
    assert wide.news_count_30d.eq(0).all()
    assert wide.negative_share_30d.isna().all()
    checked = coverage.loc[coverage.group.eq("D")]
    assert checked.max_dependency_available_at.isna().all()
    assert checked.dependency_violations.eq(0).all()
