"""Fixed E05c annual national forecast features with per-example availability.

Each query is evaluated at its own historical issue date, never the later fit
date. Published forecasts for the target calendar year remain annual growth
percentages. There is no actual-value substitution, backfill, annual-to-monthly
conversion, price deflation, or change to the original 19 expense features.

E05a's table validation is reused. Its selector orders forecast issue dates;
E05c explicitly orders publication dates, so selection is implemented here.
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from .macro_features import REQUIRED_COLUMNS, validate_macro_table

INFLATION = "forecast_inflation_dec_dec_pct"
CONSUMPTION = "forecast_consumption_growth_annual_pct"
MACRO_INDICATORS = (INFLATION, CONSUMPTION)
EXPECTED_BASIS = {
    INFLATION: "december_to_december",
    CONSUMPTION: "annual_real_volume_growth",
}
DERIVED_MIDPOINT = "interval_midpoint_not_official_central"
OFFICIAL_CENTRAL_METHODS = {
    "", "official_central", "official_central_estimate", "official_point_estimate",
    "official_median", "quoted_rounded_respondent_median",
}
SUFFIXES = ("", "_range_width_pp", "_midpoint_used", "_missing", "_publication_age_days")
LINEAGE_COLUMNS = (
    "sample_id", "sample_position", "sample_index", "municipality_id", "as_of_date",
    "target_period", "target_year", "indicator", "value", "range_width_pp",
    "midpoint_used", "missing", "publication_age_days", "source_id",
    "publication_date", "available_at", "vintage_id", "central_method",
    "value_method", "source_url", "source_status", "is_forecast", "geographic_level",
    "region_id", "aggregation_basis", "unit", "forecast_issue_date",
    "forecast_producer", "forecast_low", "forecast_central", "forecast_high",
    "missing_reason",
)


def _indicators(indicators: Sequence[str]) -> tuple[str, ...]:
    selected = tuple(indicators)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("Forecast indicators must be nonempty and unique.")
    if set(selected) - set(MACRO_INDICATORS):
        raise ValueError("E05c supports only fixed national inflation and consumption forecasts.")
    return selected


def forecast_feature_names(indicators: Sequence[str]) -> list[str]:
    """Five numeric, fixed-order feature columns per selected annual indicator."""
    return [indicator + suffix for indicator in _indicators(indicators) for suffix in SUFFIXES]


def macro_feature_dictionary(indicators: Sequence[str]) -> list[dict]:
    """Definitions for the additional columns; all model input types are float64."""
    definitions = []
    for indicator in _indicators(indicators):
        definition = (
            "National CPI December / previous December growth; published rounded median of survey participants."
            if indicator == INFLATION else
            "National annual real household consumption volume growth; Bank of Russia forecast."
        )
        formulas = (
            "Official forecast_central if documented as official; otherwise (forecast_low + forecast_high) / 2 when both bounds are finite; otherwise NaN.",
            "forecast_high - forecast_low when both published bounds are finite and ordered; otherwise NaN.",
            "1 if value uses our fixed interval midpoint; 0 otherwise, including a missing forecast.",
            "1 if selected annual forecast value is unavailable/nonfinite; 0 otherwise.",
            "(normalize(as_of_date) - normalize(publication_date)).days; publication must be available at as_of_date.",
        )
        units = ("annual percent growth", "percentage points", "binary 0/1", "binary 0/1", "calendar days")
        nan_rules = (
            "Keep NaN; do not use value, a fact, another target year, or a future publication.",
            "Keep NaN when either range bound is absent; a point estimate is not a zero-width published range.",
            "Never NaN: 0 when no interval midpoint is used.",
            "Never NaN: 1 when value is missing, preserving the expense training row.",
            "NaN when no eligible publication exists; known even when its estimate is missing.",
        )
        for suffix, formula, unit, rule in zip(SUFFIXES, formulas, units, nan_rules):
            definitions.append({
                "name": indicator + suffix, "indicator": indicator, "formula": formula,
                "type": "float64", "unit": unit, "source": "E05a archived official national publications, status A only",
                "source_definition": definition, "aggregation_basis": EXPECTED_BASIS[indicator],
                "target_year_rule": "calendar year of target_period, not issue year or last observed month",
                "availability_rule": "latest publication_date <= own as_of_date with available_at <= own as_of_date and forecast_issue_date <= own as_of_date",
                "nan_rule": rule, "annual_to_monthly_conversion": False,
            })
    return definitions


def _date_series(values: pd.Series, label: str) -> pd.Series:
    parsed = pd.to_datetime(values, errors="raise")
    if parsed.isna().any() or parsed.dt.tz is not None:
        raise ValueError(f"{label} must contain timezone-free dates.")
    return parsed


def _target_month(value: object) -> pd.Period:
    if isinstance(value, pd.Period):
        if value.freqstr != "M":
            raise ValueError("Samples require monthly target_period, not an annual forecast year.")
        return value
    if pd.isna(value):
        raise ValueError("Samples require a target month.")
    text = str(value)
    if len(text) == 4 and text.isdigit():
        raise ValueError("Samples require monthly target_period, not an annual forecast year.")
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).to_period("M")
    if len(text) < 7 or text[4] != "-":
        raise ValueError("Samples require calendar target_period in YYYY-MM/date form.")
    return pd.Period(text[:7], freq="M")


def _strict_forecasts(macro: pd.DataFrame, indicators: tuple[str, ...]) -> pd.DataFrame:
    """Reuse E05a validation, extending only absent estimate representation.

    E05a requires finite ``value``. Its already extracted ``value`` is not an
    E05c feature or a fallback estimate. A validation-only zero for an absent
    value allows a dated publication without a usable central/range estimate;
    the original missing value is restored and never fills a feature.
    """
    missing = set(REQUIRED_COLUMNS) - set(macro)
    if missing:
        raise ValueError(f"Missing macro columns: {sorted(missing)}")
    table = macro.copy(deep=True)
    if "is_forecast" not in table:
        table["is_forecast"] = False
    forecast = table["is_forecast"].map(lambda value: value is True or value == 1 or str(value).lower() in ("true", "1"))
    table = table.loc[
        table["availability_status"].eq("A") & table["region_id"].astype(str).eq("RU")
        & table["indicator"].isin(indicators) & forecast
    ].copy()
    original_values = pd.to_numeric(table["value"], errors="raise")
    if np.isinf(original_values.to_numpy(dtype=float)).any():
        raise ValueError("Macro source values cannot be infinite.")
    table["value"] = original_values.fillna(0.0)
    table = validate_macro_table(table)
    table["value"] = original_values.to_numpy(dtype=float)
    if "central_method" not in table:
        table["central_method"] = ""
    table["central_method"] = table["central_method"].fillna("").astype(str)
    if not table.empty:
        annual = table["target_period"].str.fullmatch(r"\d{4}")
        if not annual.all():
            raise ValueError("E05c forecast sources must target an explicit annual year.")
        expected = table["indicator"].map(EXPECTED_BASIS)
        if not table["aggregation_basis"].eq(expected).all():
            raise ValueError("Incompatible forecast aggregation basis; December/December and annual real volume growth cannot be mixed.")
        unknown_method = ~table["central_method"].isin(OFFICIAL_CENTRAL_METHODS | {DERIVED_MIDPOINT})
        if (unknown_method & table["forecast_central"].notna()).any():
            raise ValueError("Undocumented forecast central_method; do not assume an extracted central is official.")
    table["_target_year"] = pd.to_numeric(table["target_period"], errors="raise").astype(int)
    return table


def _value(row: pd.Series) -> tuple[float, float, bool, str]:
    low, central, high = (float(row[name]) if pd.notna(row[name]) else np.nan for name in ("forecast_low", "forecast_central", "forecast_high"))
    bounds = np.isfinite(low) and np.isfinite(high)
    width = high - low if bounds else np.nan
    if row["central_method"] != DERIVED_MIDPOINT and np.isfinite(central):
        return central, width, False, "official_central"
    if bounds:
        return (low + high) / 2.0, width, True, "fixed_interval_midpoint_not_official"
    return np.nan, width, False, "missing_central_and_complete_range"


def _choose(table: pd.DataFrame, as_of: pd.Timestamp, target_year: int, indicator: str) -> pd.Series | None:
    choices = table.loc[
        table["indicator"].eq(indicator) & table["_target_year"].eq(target_year)
        & table["published_at"].le(as_of) & table["available_at"].le(as_of)
        & table["forecast_issue_date"].le(as_of)
    ]
    if choices.empty:
        return None
    ordered = choices.sort_values(["published_at", "available_at", "vintage_id", "source_url"], kind="stable")
    chosen = ordered.iloc[-1]
    tied = ordered.loc[
        ordered["published_at"].eq(chosen["published_at"])
        & ordered["available_at"].eq(chosen["available_at"])
    ]
    if len(tied) > 1:
        estimates = pd.DataFrame([_value(row) for _, row in tied.iterrows()])
        if len(estimates.drop_duplicates()) > 1:
            raise ValueError("Conflicting forecasts have indistinguishable publication and availability dates.")
    return chosen


def prepare_forecast_features(
    samples: pd.DataFrame,
    macro: pd.DataFrame,
    *,
    indicators: Sequence[str] = MACRO_INDICATORS,
    source_ids: Mapping[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return macro columns on exactly the input index/order and long lineage.

    Queries are deduplicated by (own as-of timestamp, target calendar year),
    then broadcast across MO. ``sample_position`` remains unique even when the
    original DataFrame index is duplicated. No expense/base feature is read.
    """
    selected = _indicators(indicators)
    required = {"municipality_id", "as_of_date", "target_period"}
    if not required.issubset(samples):
        raise ValueError(f"Missing sample columns: {sorted(required - set(samples))}")
    queries = samples.copy(deep=True)
    if queries["municipality_id"].isna().any():
        raise ValueError("Samples require municipality_id.")
    queries["as_of_date"] = _date_series(queries["as_of_date"], "as_of_date")
    if "sample_id" in queries:
        if queries["sample_id"].isna().any() or queries["sample_id"].astype(str).duplicated().any() or queries["sample_id"].astype(str).str.strip().eq("").any():
            raise ValueError("sample_id must be nonempty and unique.")
        sample_ids = queries["sample_id"].astype(str).tolist()
    else:
        sample_ids = [str(position) for position in range(len(queries))]
    table = _strict_forecasts(macro, selected)
    columns = forecast_feature_names(selected)
    matrix = np.empty((len(queries), len(columns)), dtype=np.float64)
    records = []
    cache = {}
    for position, (sample_index, sample) in enumerate(queries.iterrows()):
        as_of = sample["as_of_date"]
        target = _target_month(sample["target_period"])
        query_key = (as_of, target.year)
        if query_key not in cache:
            cache[query_key] = {}
            for indicator in selected:
                row = _choose(table, as_of, target.year, indicator)
                if row is None:
                    cache[query_key][indicator] = {
                        "indicator": indicator, "value": np.nan, "range_width_pp": np.nan,
                        "midpoint_used": False, "missing": True, "publication_age_days": np.nan,
                        "source_id": "", "publication_date": pd.NaT, "available_at": pd.NaT,
                        "vintage_id": "", "central_method": "", "value_method": "",
                        "source_url": "", "source_status": "", "is_forecast": False,
                        "geographic_level": "national", "region_id": "RU",
                        "aggregation_basis": EXPECTED_BASIS[indicator], "unit": "percent_growth",
                        "forecast_issue_date": pd.NaT, "forecast_producer": "",
                        "forecast_low": np.nan, "forecast_central": np.nan, "forecast_high": np.nan,
                        "missing_reason": "no_available_forecast_for_target_year",
                    }
                    continue
                value, width, midpoint, method = _value(row)
                url = str(row["source_url"])
                if source_ids is not None:
                    if url not in source_ids or not str(source_ids[url]):
                        raise ValueError(f"Eligible publication is absent from source_id mapping: {url}")
                    source_id = str(source_ids[url])
                else:
                    source_id = "url_sha256_" + hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
                cache[query_key][indicator] = {
                    "indicator": indicator, "value": value, "range_width_pp": width,
                    "midpoint_used": midpoint, "missing": not np.isfinite(value),
                    "publication_age_days": float((as_of.normalize() - row["published_at"].normalize()).days),
                    "source_id": source_id, "publication_date": row["published_at"], "available_at": row["available_at"],
                    "vintage_id": row["vintage_id"], "central_method": row["central_method"], "value_method": method,
                    "source_url": url, "source_status": "A", "is_forecast": True,
                    "geographic_level": "national", "region_id": "RU",
                    "aggregation_basis": row["aggregation_basis"], "unit": row["unit"],
                    "forecast_issue_date": row["forecast_issue_date"], "forecast_producer": row["forecast_producer"],
                    "forecast_low": row["forecast_low"], "forecast_central": row["forecast_central"], "forecast_high": row["forecast_high"],
                    "missing_reason": "" if np.isfinite(value) else "no_central_or_complete_published_range",
                }
        for indicator_position, indicator in enumerate(selected):
            record = dict(cache[query_key][indicator])
            values = (record["value"], record["range_width_pp"], float(record["midpoint_used"]), float(record["missing"]), record["publication_age_days"])
            matrix[position, indicator_position * 5:(indicator_position + 1) * 5] = values
            record.update({
                "sample_id": sample_ids[position], "sample_position": position,
                "sample_index": sample_index, "municipality_id": str(sample["municipality_id"]),
                "as_of_date": as_of, "target_period": str(target), "target_year": target.year,
            })
            records.append(record)
    features = pd.DataFrame(matrix, index=samples.index.copy(), columns=columns, dtype=np.float64)
    lineage = pd.DataFrame(records, columns=LINEAGE_COLUMNS)
    return features, lineage
