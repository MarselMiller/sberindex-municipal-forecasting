"""Synthetic, offline tests for the fixed E05c national forecast join."""
import numpy as np
import pandas as pd
import pytest

from sberforecast.macro_forecast_features import (
    CONSUMPTION, DERIVED_MIDPOINT, INFLATION, MACRO_INDICATORS,
    forecast_feature_names, macro_feature_dictionary, prepare_forecast_features,
)


def forecast_row(indicator=INFLATION, year=2024, published="2023-12-15", vintage="edition1", **updates):
    row = {
        "region_id": "RU", "geographic_level": "national", "indicator": indicator,
        "reference_period": str(year), "target_period": str(year), "value": 5.0,
        "unit": "percent_growth", "published_at": published, "available_at": published,
        "vintage_id": vintage, "availability_status": "A", "source_url": "https://official.example/edition1",
        "forecast_issue_date": published, "is_forecast": True,
        "aggregation_basis": "december_to_december" if indicator == INFLATION else "annual_real_volume_growth",
        "forecast_low": np.nan, "forecast_central": 5.0, "forecast_high": np.nan,
        "forecast_producer": "survey_participants" if indicator == INFLATION else "Bank of Russia",
        "central_method": "quoted_rounded_respondent_median" if indicator == INFLATION else "official_central",
    }
    row.update(updates)
    return row


def samples(*as_of, targets=None, index=None):
    if targets is None:
        targets = ["2024-01"] * len(as_of)
    return pd.DataFrame({
        "municipality_id": [str(100 + position) for position in range(len(as_of))],
        "as_of_date": list(as_of), "target_period": targets,
    }, index=index)


def prepare(query, rows, indicators=(INFLATION,), **kwargs):
    return prepare_forecast_features(query, pd.DataFrame(rows, columns=list(forecast_row())), indicators=indicators, **kwargs)


def test_future_publication_cannot_change_historical_features_or_lineage():
    query = samples("2023-12-31")
    old = forecast_row()
    future = forecast_row(published="2024-02-15", vintage="future", forecast_central=12, value=12)
    X, lineage = prepare(query, [old])
    with_future = prepare(query, [old, future])
    pd.testing.assert_frame_equal(X, with_future[0])
    pd.testing.assert_frame_equal(lineage, with_future[1])


def test_revision_enters_only_after_actual_availability_not_download_or_original_date():
    query = samples("2024-01-31", "2024-02-28", "2024-02-29")
    original = forecast_row(published="2024-01-10", vintage="initial")
    revision = forecast_row(published="2024-01-10", available_at="2024-02-29", vintage="revision", forecast_central=6.0)
    X, provenance = prepare(query, [original, revision])
    assert X[INFLATION].tolist() == [5, 5, 6]
    assert provenance.vintage_id.tolist() == ["initial", "initial", "revision"]
    assert provenance.available_at.iloc[-1] == pd.Timestamp("2024-02-29")


def test_publication_day_is_inclusive_and_both_date_gates_are_required():
    row = forecast_row(published="2024-01-31", available_at="2024-02-01")
    X, lineage = prepare(samples("2024-01-31", "2024-02-01"), [row])
    assert X[INFLATION].isna().tolist() == [True, False]
    assert X[f"{INFLATION}_publication_age_days"].iloc[1] == 1
    assert lineage.source_status.tolist() == ["", "A"]


def test_latest_publication_has_priority_over_forecast_issue_date():
    first = forecast_row(published="2024-01-05", forecast_issue_date="2024-01-05", vintage="issue5", forecast_central=6)
    later_publication = forecast_row(published="2024-01-10", forecast_issue_date="2024-01-02", vintage="issue2", forecast_central=7)
    X, lineage = prepare(samples("2024-01-31"), [first, later_publication])
    assert X[INFLATION].iloc[0] == 7
    assert lineage.vintage_id.iloc[0] == "issue2"


def test_year_transition_matches_target_year_not_issue_year():
    query = samples("2023-12-31", "2023-12-31", targets=["2023-12", "2024-01"])
    X, lineage = prepare(query, [forecast_row(year=2023, forecast_central=8), forecast_row(year=2024, forecast_central=4)])
    assert X[INFLATION].tolist() == [8, 4]
    assert lineage.target_year.tolist() == [2023, 2024]


def test_missing_target_year_is_not_replaced_by_stale_other_year_or_actual_fact():
    rows = [forecast_row(year=2023), forecast_row(year=2025), forecast_row(year=2024, is_forecast=False)]
    X, lineage = prepare(samples("2023-12-31"), rows)
    assert X[INFLATION].isna().all()
    assert X[f"{INFLATION}_missing"].eq(1.0).all()
    assert lineage.missing_reason.eq("no_available_forecast_for_target_year").all()


@pytest.mark.parametrize("status", ["B", "C"])
def test_non_A_values_never_substitute_for_archived_A_forecasts(status):
    X, lineage = prepare(samples("2023-12-31"), [forecast_row(availability_status=status)])
    assert X[INFLATION].isna().all() and lineage.source_id.eq("").all()


def test_regional_rows_never_substitute_for_national_forecasts():
    X, _ = prepare(samples("2023-12-31"), [forecast_row(region_id="1", geographic_level="regional")])
    assert X[INFLATION].isna().all()


def test_official_central_has_priority_over_midpoint_and_width_is_not_a_mean():
    row = forecast_row(forecast_low=2, forecast_central=5, forecast_high=9)
    X, lineage = prepare(samples("2023-12-31"), [row])
    assert X[INFLATION].iloc[0] == 5
    assert X[f"{INFLATION}_range_width_pp"].iloc[0] == 7
    assert X[f"{INFLATION}_midpoint_used"].iloc[0] == 0
    assert lineage.value_method.iloc[0] == "official_central"


def test_E05a_consumption_derived_central_is_marked_as_our_midpoint():
    row = forecast_row(CONSUMPTION, forecast_low=-2, forecast_central=-1.5, forecast_high=-1,
                       central_method=DERIVED_MIDPOINT, value=-1.5)
    X, lineage = prepare(samples("2023-12-31"), [row], indicators=(CONSUMPTION,))
    assert X[CONSUMPTION].iloc[0] == -1.5
    assert X[f"{CONSUMPTION}_range_width_pp"].iloc[0] == 1
    assert X[f"{CONSUMPTION}_midpoint_used"].iloc[0] == 1
    assert lineage.central_method.iloc[0] == DERIVED_MIDPOINT
    assert lineage.value_method.iloc[0] == "fixed_interval_midpoint_not_official"


def test_interval_without_central_computes_midpoint_and_keeps_annual_percent_growth():
    row = forecast_row(CONSUMPTION, forecast_low=12, forecast_central=np.nan, forecast_high=24, value=999)
    X, _ = prepare(samples("2023-12-31"), [row], indicators=(CONSUMPTION,))
    assert X[CONSUMPTION].iloc[0] == 18  # Never divided by 12, never uses value=999.
    assert X[f"{CONSUMPTION}_midpoint_used"].iloc[0] == 1


def test_no_central_or_complete_range_is_missing_even_if_generic_value_exists():
    row = forecast_row(forecast_central=np.nan, value=999)
    X, lineage = prepare(samples("2023-12-31"), [row])
    assert np.isnan(X[INFLATION].iloc[0])
    assert X[f"{INFLATION}_missing"].iloc[0] == 1
    assert X[f"{INFLATION}_publication_age_days"].iloc[0] == 16
    assert lineage.source_status.iloc[0] == "A"
    assert lineage.missing_reason.iloc[0] == "no_central_or_complete_published_range"


def test_absent_generic_value_is_not_imputed_as_forecast():
    row = forecast_row(forecast_central=np.nan, value=np.nan)
    X, lineage = prepare(samples("2023-12-31"), [row])
    assert X[INFLATION].isna().all() and lineage.value.isna().all()


def test_point_estimate_does_not_invent_zero_width_range():
    X, _ = prepare(samples("2023-12-31"), [forecast_row()])
    assert X[f"{INFLATION}_range_width_pp"].isna().all()


def test_single_range_bound_does_not_create_midpoint():
    X, _ = prepare(samples("2023-12-31"), [forecast_row(forecast_central=np.nan, forecast_low=1)])
    assert X[INFLATION].isna().all()
    assert X[f"{INFLATION}_range_width_pp"].isna().all()


@pytest.mark.parametrize("unit", ["percent_index", "rub", "index", "monthly_percent_growth"])
def test_forecast_units_reject_price_indices_levels_and_unconfirmed_monthly_units(unit):
    with pytest.raises(ValueError, match="growth percentages"):
        prepare(samples("2023-12-31"), [forecast_row(unit=unit)])


def test_wrong_aggregation_cannot_rename_average_inflation_as_december_inflation():
    with pytest.raises(ValueError, match="aggregation basis"):
        prepare(samples("2023-12-31"), [forecast_row(aggregation_basis="annual_average")])


def test_invalid_or_infinite_bounds_fail_visibly():
    with pytest.raises(ValueError, match="ordered"):
        prepare(samples("2023-12-31"), [forecast_row(forecast_low=6, forecast_central=5, forecast_high=8)])
    with pytest.raises(ValueError, match="finite"):
        prepare(samples("2023-12-31"), [forecast_row(forecast_high=np.inf)])


def test_unknown_extracted_central_method_requires_explicit_source_decision():
    with pytest.raises(ValueError, match="central_method"):
        prepare(samples("2023-12-31"), [forecast_row(central_method="unexplained_preprocessing")])


def test_unknown_forecast_target_frequency_is_rejected():
    with pytest.raises(ValueError, match="annual year"):
        prepare(samples("2023-12-31"), [forecast_row(target_period="2024-01")])


def test_conflicting_same_day_publications_are_not_arbitrarily_chosen():
    with pytest.raises(ValueError, match="indistinguishable"):
        prepare(samples("2023-12-31"), [forecast_row(), forecast_row(vintage="other", forecast_central=6)])


def test_row_order_duplicate_indices_national_broadcast_and_input_are_preserved():
    query = samples("2023-12-31", "2024-01-31", "2023-12-31", index=[9, 2, 9])
    query["sample_id"] = ["b", "a", "c"]
    query["raw_feature"] = [np.nan, 6.0, 1.0]
    original = query.copy(deep=True)
    source = pd.DataFrame([forecast_row()])
    source_before = source.copy(deep=True)
    X, lineage = prepare_forecast_features(query, source, indicators=(INFLATION,))
    pd.testing.assert_frame_equal(query, original)
    pd.testing.assert_frame_equal(source, source_before)
    assert X.index.tolist() == [9, 2, 9]
    assert lineage.sample_id.tolist() == ["b", "a", "c"]
    assert lineage.sample_position.tolist() == [0, 1, 2]
    assert lineage.sample_index.tolist() == [9, 2, 9]
    assert lineage.municipality_id.tolist() == ["100", "101", "102"]
    assert X[INFLATION].tolist() == [5.0, 5.0, 5.0]
    assert X.iloc[0].equals(X.iloc[2])  # MO does not create fake national variation.
    assert X[f"{INFLATION}_publication_age_days"].tolist() == [16, 47, 16]


def test_unique_sample_ids_are_required_when_supplied():
    query = samples("2023-12-31", "2023-12-31").assign(sample_id=["x", "x"])
    with pytest.raises(ValueError, match="sample_id"):
        prepare(query, [forecast_row()])


def test_source_id_comes_from_verified_catalogue_mapping():
    row = forecast_row()
    _, lineage = prepare(samples("2023-12-31"), [row], source_ids={row["source_url"]: "verified_archived_edition"})
    assert lineage.source_id.iloc[0] == "verified_archived_edition"
    with pytest.raises(ValueError, match="source_id mapping"):
        prepare(samples("2023-12-31"), [row], source_ids={})


def test_stable_exact_schema_all_float64_and_flags_numeric_zero_one():
    X, lineage = prepare(samples("2023-12-31"), [forecast_row()], indicators=MACRO_INDICATORS)
    assert X.columns.tolist() == forecast_feature_names(MACRO_INDICATORS)
    assert X.shape == (1, 10)
    assert X.dtypes.eq(np.dtype("float64")).all()
    assert X[f"{CONSUMPTION}_missing"].iloc[0] == 1.0
    assert len(lineage) == 2
    definitions = macro_feature_dictionary(MACRO_INDICATORS)
    assert [row["name"] for row in definitions] == X.columns.tolist()
    assert all({"name", "formula", "type", "unit", "source", "nan_rule"}.issubset(row) for row in definitions)


def test_empty_queries_and_missing_sources_have_fixed_schema_and_no_deleted_rows():
    empty, provenance = prepare(samples(), [forecast_row()], indicators=MACRO_INDICATORS)
    assert empty.shape == (0, 10) and provenance.empty
    missing, provenance = prepare(samples("2023-12-31", "2024-01-31"), [], indicators=MACRO_INDICATORS)
    assert missing.shape == (2, 10) and len(provenance) == 4
    assert missing[[INFLATION, CONSUMPTION]].isna().all().all()


@pytest.mark.parametrize("indicators", [(), (INFLATION, INFLATION), ("cpi_mom_index",)])
def test_only_fixed_unique_E05c_indicators_are_accepted(indicators):
    with pytest.raises(ValueError):
        forecast_feature_names(indicators)


def test_asof_timezone_and_target_year_queries_require_explicit_valid_convention():
    with pytest.raises(ValueError, match="timezone-free"):
        prepare(samples(pd.Timestamp("2023-12-31", tz="UTC")), [forecast_row()])
    with pytest.raises(ValueError, match="monthly target_period"):
        prepare(samples("2023-12-31", targets=["2024"]), [forecast_row()])


def test_within_one_model_training_examples_use_own_asof_dates():
    query = samples("2024-01-31", "2024-02-29", "2024-03-31")
    rows = [forecast_row(published="2024-01-10", vintage="jan", forecast_central=5),
            forecast_row(published="2024-03-01", vintage="mar", forecast_central=8)]
    X, lineage = prepare(query, rows)
    assert X[INFLATION].tolist() == [5, 5, 8]
    assert lineage.vintage_id.tolist() == ["jan", "jan", "mar"]
