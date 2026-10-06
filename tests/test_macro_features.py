"""Offline synthetic tests for macro definitions, vintages, and causality."""
import numpy as np
import pandas as pd
import pytest

from sberforecast.macro_features import (
    attach_regions,
    build_macro_features,
    build_price_index,
    build_region_mapping,
    normalize_region_id,
    validate_macro_table,
)


def macro_row(indicator="cpi_mom_index", period="2023-01", value=101.0,
              available="2023-02-10", vintage="v1", status="A", **extra):
    row = {
        "region_id": "01", "indicator": indicator, "reference_period": period,
        "value": value, "unit": "RUB" if indicator == "nominal_wage_rub" else "index_percent",
        "published_at": available, "available_at": available, "vintage_id": vintage,
        "availability_status": status, "source_url": "https://example.invalid/official-test",
        "aggregation_basis": "year_on_year" if indicator == "cpi_yoy_index" else "month_to_month" if indicator == "cpi_mom_index" else "month",
        "geographic_level": "regional", "is_forecast": False,
    }
    return dict(row, **extra)


def samples(*dates):
    return pd.DataFrame({
        "sample_id": [f"sample-{index}" for index in range(len(dates))],
        "municipality_id": ["21"] * len(dates), "region_id": ["1"] * len(dates),
        "as_of_date": list(dates), "target_period": ["2024-03"] * len(dates),
    })


def forecast_row(status="A", **extra):
    return macro_row(
        "forecast_inflation_dec_dec_pct", "2024", 5.0,
        "2023-11-07", status=status, region_id="RU", unit="growth_pct",
        is_forecast=True, geographic_level="national",
        forecast_issue_date="2023-11-07", target_period="2024",
        aggregation_basis="december_to_december", forecast_low=4.0,
        forecast_central=5.0, forecast_high=6.0,
        forecast_producer="survey_participants", **extra,
    )


def test_future_publication_does_not_change_past_features_or_provenance():
    query = samples("2023-02-28")
    old = pd.DataFrame([macro_row()])
    future = macro_row(period="2023-02", value=999.0, available="2023-03-15", vintage="future")
    before = build_macro_features(query, old)
    after = build_macro_features(query, pd.concat([old, pd.DataFrame([future])], ignore_index=True))
    pd.testing.assert_frame_equal(before[0], after[0])
    pd.testing.assert_frame_equal(before[1], after[1])


def test_revision_is_selected_only_when_that_version_becomes_available():
    rows = pd.DataFrame([
        macro_row(value=101.0),
        macro_row(value=102.0, available="2023-03-15", vintage="revision"),
    ])
    X, provenance = build_macro_features(samples("2023-03-14", "2023-03-15"), rows)
    assert X["cpi_mom_growth_pct"].tolist() == [1.0, 2.0]
    assert provenance.loc[provenance.feature.eq("cpi_mom_growth_pct"), "vintage_id"].tolist() == ["v1", "revision"]


def test_historical_examples_use_own_dates_not_later_model_fit_date():
    query = samples("2023-01-31", "2023-02-28", "2023-03-31")
    query["model_origin"] = "2024-11-30"
    table = pd.DataFrame([
        macro_row(), macro_row(period="2023-02", value=103.0, available="2023-03-10", vintage="feb"),
    ])
    X, provenance = build_macro_features(query, table)
    assert np.isnan(X.iloc[0]["cpi_mom_growth_pct"])
    assert X["cpi_mom_growth_pct"].iloc[1:].tolist() == [1.0, 3.0]
    assert X["cpi_mom_growth_pct_missing"].tolist() == [True, False, False]
    known = provenance.loc[~provenance.missing]
    assert (known.available_at <= known.as_of_date).all()


def test_no_backfill_and_explicit_staleness_after_last_publication():
    X, provenance = build_macro_features(samples("2023-02-09", "2023-05-31"), pd.DataFrame([macro_row()]))
    assert np.isnan(X.iloc[0]["cpi_mom_growth_pct"])
    assert X.iloc[1]["cpi_mom_growth_pct"] == 1.0
    assert X.iloc[1]["cpi_mom_growth_pct_age_months"] == 4
    record = provenance.loc[provenance.feature.eq("cpi_mom_growth_pct")].iloc[0]
    assert record.missing_reason == "no_available_observation"


def test_published_future_reference_month_is_not_an_actual_feature():
    table = pd.DataFrame([macro_row(period="2023-06", available="2023-01-15")])
    X, _ = build_macro_features(samples("2023-03-31"), table)
    assert X.cpi_mom_growth_pct.isna().all()


def test_status_B_requires_explicit_scenario_and_remains_B():
    table = pd.DataFrame([macro_row(available="2026-10-06", status="B")])
    query = samples("2023-02-27", "2023-02-28")
    strict, _ = build_macro_features(query, table)
    scenario, provenance = build_macro_features(query, table, scenario_lags={"cpi_mom_index": 1})
    assert strict.cpi_mom_growth_pct.isna().all()
    assert np.isnan(scenario.iloc[0]["cpi_mom_growth_pct"])
    assert scenario.iloc[1]["cpi_mom_growth_pct"] == 1.0
    record = provenance.loc[(provenance.sample_id == "sample-1") & provenance.feature.eq("cpi_mom_growth_pct")].iloc[0]
    assert record.availability_status == "B" and record.scenario
    assert record.scenario_available_at == pd.Timestamp("2023-02-28")
    assert record.available_at == pd.Timestamp("2026-10-06")


def test_status_C_is_excluded_even_with_a_scenario_lag():
    table = pd.DataFrame([macro_row(status="C")])
    X, _ = build_macro_features(samples("2024-01-31"), table, scenario_lags={"cpi_mom_index": 0})
    assert X.cpi_mom_growth_pct.isna().all()


@pytest.mark.parametrize("lag", [-1, 1.2, True])
def test_scenario_lag_is_explicit_nonnegative_integer(lag):
    with pytest.raises(ValueError, match="Scenario lag"):
        build_macro_features(samples("2023-03-31"), pd.DataFrame([macro_row()]), scenario_lags={"cpi_mom_index": lag})


def test_units_and_aligned_nominal_real_growth_are_distinct():
    table = pd.DataFrame([
        macro_row("nominal_wage_rub", "2022-01", 100.0, "2022-03-15", "w2022"),
        macro_row("nominal_wage_rub", "2023-01", 120.0, "2023-03-15", "w2023"),
        macro_row("cpi_yoy_index", "2023-01", 110.0, "2023-02-15", "p2023"),
        macro_row("cpi_yoy_index", "2023-02", 112.0, "2023-03-15", "p2023feb"),
        macro_row("cpi_mom_index", "2023-02", 101.0, "2023-03-15", "mom"),
    ])
    X, provenance = build_macro_features(samples("2023-03-31"), table)
    assert X.iloc[0]["cpi_mom_growth_pct"] == 1.0
    assert X.iloc[0]["cpi_yoy_growth_pct"] == 12.0
    assert X.iloc[0]["nominal_wage_rub"] == 120.0
    assert X.iloc[0]["nominal_wage_yoy_growth_pct"] == pytest.approx(20.0)
    assert X.iloc[0]["real_wage_yoy_growth_pct"] == pytest.approx(100 * (1.2 / 1.1 - 1))
    record = provenance.loc[provenance.feature.eq("real_wage_yoy_growth_pct")].iloc[0]
    assert "2023-01" in record.components and "2022-01" in record.components
    assert "p2023feb" not in record.components
    assert record.unit == "percent_growth" and record.aggregation_basis == "year_on_year"
    cpi_record = provenance.loc[provenance.feature.eq("cpi_mom_growth_pct")].iloc[0]
    assert cpi_record.unit == "percent_growth" and "index_percent" in cpi_record.components


def test_missing_aligned_CPI_does_not_use_a_different_month_for_real_wage():
    table = pd.DataFrame([
        macro_row("nominal_wage_rub", "2022-01", 100.0, "2022-03-15", "w2022"),
        macro_row("nominal_wage_rub", "2023-01", 120.0, "2023-03-15", "w2023"),
        macro_row("cpi_yoy_index", "2023-02", 110.0, "2023-03-15", "p2023feb"),
    ])
    X, _ = build_macro_features(samples("2023-03-31"), table)
    assert X.nominal_wage_yoy_growth_pct.iloc[0] == pytest.approx(20.0)
    assert X.real_wage_yoy_growth_pct.isna().all()


@pytest.mark.parametrize("change", [
    {"unit": "growth_pct"}, {"aggregation_basis": "year_to_date"},
    {"reference_period": "2023"}, {"value": 0.0},
])
def test_CPI_definition_mismatch_is_rejected(change):
    with pytest.raises(ValueError):
        validate_macro_table(pd.DataFrame([macro_row(**change)]))


def test_cumulative_wage_is_not_accepted_as_a_monthly_wage_level():
    table = pd.DataFrame([macro_row("nominal_wage_rub", value=50000, aggregation_basis="year_to_date")])
    with pytest.raises(ValueError, match="not interchangeable"):
        validate_macro_table(table)


def test_forecast_annual_value_is_not_divided_by_12_and_is_national():
    table = pd.DataFrame([forecast_row()])
    query = samples("2023-11-06", "2023-11-07")
    X, provenance = build_macro_features(query, table)
    assert np.isnan(X.iloc[0]["forecast_inflation_dec_dec_pct"])
    assert X.iloc[1]["forecast_inflation_dec_dec_pct"] == 5.0
    record = provenance.loc[provenance.feature.eq("forecast_inflation_dec_dec_pct")].iloc[1]
    assert record.region_id == "RU" and record.geographic_level == "national"
    assert record.aggregation_basis == "december_to_december"


def test_forecast_selects_latest_available_edition_and_exact_target_year():
    older = forecast_row()
    newer = dict(older, vintage_id="new", value=7.0, forecast_low=6.0, forecast_central=7.0,
                 forecast_high=8.0, forecast_issue_date="2024-01-15", published_at="2024-01-15", available_at="2024-01-15")
    wrong_year = dict(newer, vintage_id="2025", reference_period="2025", target_period="2025", value=90.0,
                      forecast_low=89.0, forecast_central=90.0, forecast_high=91.0)
    X, _ = build_macro_features(samples("2023-12-31", "2024-01-15"), pd.DataFrame([older, newer, wrong_year]))
    assert X.forecast_inflation_dec_dec_pct.tolist() == [5.0, 7.0]


def test_monthly_forecasts_do_not_match_other_months_in_the_same_year():
    row = forecast_row()
    row.update(reference_period="2024-02", target_period="2024-02", aggregation_basis="month_to_month")
    X, _ = build_macro_features(samples("2023-12-31"), pd.DataFrame([row]))
    assert X.forecast_inflation_dec_dec_pct.isna().all()


def test_unknown_survey_publication_day_only_works_in_explicit_B_scenario():
    row = forecast_row("B")
    row.update(forecast_issue_date=pd.NaT, survey_issue_period="2023-11", published_at=pd.NaT,
               available_at="2026-10-06")
    query = samples("2023-11-29", "2023-11-30")
    strict, _ = build_macro_features(query, pd.DataFrame([row]))
    scenario, provenance = build_macro_features(query, pd.DataFrame([row]),
                                              scenario_lags={"forecast_inflation_dec_dec_pct": 0})
    assert strict.forecast_inflation_dec_dec_pct.isna().all()
    assert np.isnan(scenario.forecast_inflation_dec_dec_pct.iloc[0])
    assert scenario.forecast_inflation_dec_dec_pct.iloc[1] == 5.0
    selected = provenance.loc[provenance.feature.eq("forecast_inflation_dec_dec_pct")].iloc[1]
    assert selected.source_status == "B" and selected.age_months == 0
    assert pd.isna(selected.published_at)


def test_price_index_uses_compounded_monthly_indices_and_breaks_at_missing_month():
    table = pd.DataFrame([
        macro_row("cpi_mom_index", "2023-01", 101.0, "2023-02-10", "jan"),
        macro_row("cpi_mom_index", "2023-03", 102.0, "2023-04-10", "mar"),
    ])
    index = build_price_index(table, "1", "2023-04-30")
    assert index.price_index.iloc[:2].tolist() == [100.0, 101.0]
    assert index.price_index.iloc[2:].isna().all()
    complete = pd.concat([table, pd.DataFrame([macro_row("cpi_mom_index", "2023-02", 102.0, "2023-03-10", "feb")])], ignore_index=True)
    before = build_price_index(complete, "1", "2023-03-31")
    assert before.price_index.iloc[2] == pytest.approx(103.02)
    assert np.isnan(before.price_index.iloc[3])  # March not yet published.


def test_price_index_future_revision_does_not_change_past_levels():
    old = pd.DataFrame([macro_row()])
    future = macro_row(value=110.0, available="2023-06-15", vintage="later")
    pd.testing.assert_frame_equal(
        build_price_index(old, "1", "2023-03-31"),
        build_price_index(pd.concat([old, pd.DataFrame([future])], ignore_index=True), "1", "2023-03-31"),
    )


def test_geography_join_is_many_to_one_and_preserves_unmapped_rows_and_indices():
    raw = pd.DataFrame({"territory_id": ["21", "21", "25"], "region_code": ["01", "1", "2"],
                        "region_name": ["Region One", "Region One", "Region Two"]})
    mapping = build_region_mapping(raw)
    query = pd.DataFrame({"municipality_id": ["21", "21", "25", "999"]}, index=[6, 3, 20, 2])
    joined = attach_regions(query, mapping)
    assert joined.index.tolist() == [6, 3, 20, 2] and len(joined) == len(query)
    assert joined.region_id.iloc[:3].tolist() == ["1", "1", "2"]
    assert pd.isna(joined.region_id.iloc[3])
    supplied = query.assign(region_id=["01", "1", "2", np.nan])
    assert len(attach_regions(supplied, mapping)) == len(query)


def test_geography_ambiguity_and_multiplying_mapping_are_explicit_errors():
    ambiguous = pd.DataFrame({"territory_id": ["21", "21"], "region_code": ["1", "2"], "region_name": ["A", "B"]})
    with pytest.raises(ValueError, match="Ambiguous"):
        build_region_mapping(ambiguous)
    mapping = pd.DataFrame({"municipality_id": ["21", "21"], "region_id": ["1", "1"]})
    with pytest.raises(ValueError, match="many-to-one"):
        attach_regions(samples("2023-01-31"), mapping)


def test_geography_disagreement_is_not_silently_overwritten():
    mapping = pd.DataFrame({"municipality_id": ["21"], "region_id": ["2"]})
    with pytest.raises(ValueError, match="disagrees"):
        attach_regions(samples("2023-01-31"), mapping)


@pytest.mark.parametrize("value,expected", [("01", "1"), (1.0, "1"), ("RU", "RU"), ("AB", "AB")])
def test_only_explicit_numeric_region_codes_are_normalized(value, expected):
    assert normalize_region_id(value) == expected


def test_A_requires_real_dates_and_version_cannot_precede_publication():
    with pytest.raises(ValueError, match="requires verified"):
        validate_macro_table(pd.DataFrame([macro_row(published_at=pd.NaT)]))
    with pytest.raises(ValueError, match="before publication"):
        validate_macro_table(pd.DataFrame([macro_row(published_at="2023-03-01")]))


def test_duplicate_version_and_same_day_conflicting_vintages_are_rejected():
    row = macro_row()
    with pytest.raises(ValueError, match="Duplicate"):
        validate_macro_table(pd.DataFrame([row, row]))
    with pytest.raises(ValueError, match="indistinguishable"):
        build_macro_features(samples("2023-03-31"), pd.DataFrame([row, dict(row, value=102.0, vintage_id="v2")]))


def test_missing_geo_keeps_all_samples_and_marks_missing_source():
    query = samples("2023-03-31").drop(columns="region_id")
    X, provenance = build_macro_features(query, pd.DataFrame([macro_row()]))
    assert X.index.tolist() == query.sample_id.tolist()
    record = provenance.loc[provenance.feature.eq("cpi_mom_growth_pct")].iloc[0]
    assert record.missing and record.missing_reason == "region_unmapped"


def test_cached_shared_regional_features_preserve_each_municipality_sample_key():
    query = samples("2023-03-31", "2023-03-31")
    query.loc[1, "municipality_id"] = "25"
    X, provenance = build_macro_features(query, pd.DataFrame([macro_row()]))
    assert X.cpi_mom_growth_pct.tolist() == [1.0, 1.0]
    known = provenance.loc[provenance.feature.eq("cpi_mom_growth_pct")]
    assert known.sample_id.tolist() == ["sample-0", "sample-1"]
    assert known.municipality_id.tolist() == ["21", "25"]


def test_empty_queries_and_empty_sources_have_stable_schema():
    query = samples()
    X, provenance = build_macro_features(query, pd.DataFrame([macro_row()]))
    assert X.empty and provenance.empty
    X, provenance = build_macro_features(samples("2023-03-31"), pd.DataFrame(columns=macro_row().keys()))
    assert len(X) == 1 and X.cpi_mom_growth_pct.isna().all()
    assert len(provenance) == 5


def test_explicit_absent_forecast_is_reported_as_missing_with_provenance():
    query = samples("2023-12-31", "2024-06-30")
    table = pd.DataFrame(columns=macro_row().keys())
    forecast = "forecast_real_wage_growth_annual_pct"
    X, provenance = build_macro_features(query, table, forecast_indicators=[forecast])
    assert X[forecast].isna().all()
    assert X[f"{forecast}_missing"].tolist() == [True, True]
    assert X[f"{forecast}_age_months"].isna().all()
    records = provenance.loc[provenance.feature.eq(forecast)]
    assert records.sample_id.tolist() == query.sample_id.tolist()
    assert records.missing.tolist() == [True, True]
    assert records.missing_reason.eq("no_available_forecast_for_target_year").all()
