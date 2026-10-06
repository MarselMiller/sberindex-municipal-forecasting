"""Point-in-time external features; this module never fits a forecast model.

Status A uses a verified publication/version availability date. Status B is
excluded unless a named fixed-lag scenario is explicitly supplied; such rows
remain B, because assigning a lag cannot establish historical vintages.
Status C is always excluded. Each query has its own ``as_of_date``.

Monthly CPI values are indices (101 means 1 percent growth), not price levels.
Nominal wage YoY = 100 * (w[t] / w[t-12] - 1). Real wage YoY =
100 * ((w[t] / w[t-12]) / (CPI_yoy[t] / 100) - 1), with aligned months.
Missing inputs remain NaN; no backward fill or annual-to-monthly conversion.
"""
from __future__ import annotations

import json
from numbers import Integral
from typing import Sequence

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = (
    "region_id", "indicator", "reference_period", "value", "unit",
    "published_at", "available_at", "vintage_id", "availability_status", "source_url",
)
ACTUAL_INDICATORS = ("cpi_mom_index", "cpi_yoy_index", "nominal_wage_rub")
DERIVED_FEATURES = ("nominal_wage_yoy_growth_pct", "real_wage_yoy_growth_pct")
INDEX_UNITS = {"%", "percent", "percent_index", "index_percent"}
WAGE_UNITS = {"rub", "rubles", "руб.", "руб", "rub/month"}
PROVENANCE_COLUMNS = (
    "sample_id", "municipality_id", "as_of_date", "feature", "value", "region_id",
    "geographic_level", "reference_period", "target_period", "published_at",
    "available_at", "scenario_available_at", "vintage_id", "source_status",
    "availability_status", "source_url", "aggregation_basis", "unit",
    "forecast_issue_date", "survey_issue_period", "forecast_producer",
    "age_months", "missing", "missing_reason", "scenario", "components",
)


def normalize_region_id(value: object) -> str:
    """Normalize only an explicit numeric code, never geographic names."""
    if pd.isna(value) or str(value).strip() == "":
        raise ValueError("Missing region identifier.")
    text = str(value).strip()
    if text == "RU":
        return text
    if text.isdigit():
        return str(int(text))
    if text.endswith(".0") and text[:-2].isdigit():
        return str(int(text[:-2]))
    return text


def build_region_mapping(raw: pd.DataFrame) -> pd.DataFrame:
    """Use the project's explicit municipality-to-region correspondence."""
    municipality = "municipality_id" if "municipality_id" in raw else "territory_id"
    required = {municipality, "region_code", "region_name"}
    if not required.issubset(raw):
        raise ValueError(f"Missing geography columns: {sorted(required - set(raw))}")
    if raw[list(required)].isna().any().any():
        raise ValueError("Missing municipality or region in geography correspondence.")
    mapping = raw[list(required)].copy().rename(columns={municipality: "municipality_id", "region_code": "region_id"})
    mapping["municipality_id"] = mapping["municipality_id"].astype(str)
    mapping["region_id"] = mapping["region_id"].map(normalize_region_id)
    mapping["region_name"] = mapping["region_name"].astype(str)
    mapping = mapping.drop_duplicates()
    if mapping.groupby("municipality_id")["region_id"].nunique().gt(1).any():
        raise ValueError("Ambiguous municipality-to-region correspondence; no fuzzy matching is permitted.")
    if mapping.groupby("region_id")["region_name"].nunique().gt(1).any():
        raise ValueError("A region code has multiple names; an explicit geography decision is required.")
    return mapping.drop_duplicates("municipality_id").sort_values("municipality_id").reset_index(drop=True)


def attach_regions(samples: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    """Left join geography, preserve query rows and report unmapped rows."""
    if "municipality_id" not in samples or not {"municipality_id", "region_id"}.issubset(mapping):
        raise ValueError("Geography join requires municipality_id and region_id.")
    if mapping["municipality_id"].astype(str).duplicated().any():
        raise ValueError("Region mapping is not many-to-one.")
    left = samples.copy()
    left["municipality_id"] = left["municipality_id"].astype(str)
    right = mapping.copy()
    right["municipality_id"] = right["municipality_id"].astype(str)
    if "region_id" in left:
        expected = left["region_id"].map(lambda value: normalize_region_id(value) if pd.notna(value) else np.nan)
        checked = left[["municipality_id"]].merge(right[["municipality_id", "region_id"]], on="municipality_id", how="left", validate="many_to_one", sort=False)
        mismatch = expected.notna().to_numpy() & checked["region_id"].notna().to_numpy() & expected.ne(checked["region_id"].to_numpy()).to_numpy()
        if mismatch.any():
            raise ValueError("Sample region disagrees with the explicit municipality mapping.")
        left = left.drop(columns=[column for column in ("region_id", "region_name") if column in left])
    joined = left.merge(right, on="municipality_id", how="left", validate="many_to_one", sort=False)
    if len(joined) != len(samples):
        raise AssertionError("Geography join changed the number of rows.")
    joined.index = samples.index
    return joined


def _boolean(value: object) -> bool:
    if pd.isna(value):
        return False
    if value is True or value == 1 or str(value).lower() in ("true", "1"):
        return True
    if value is False or value == 0 or str(value).lower() in ("false", "0"):
        return False
    raise ValueError(f"Invalid actual/forecast flag: {value!r}")


def _period(text: object) -> pd.Period:
    text = str(text).strip()
    if len(text) == 4 and text.isdigit():
        return pd.Period(text, freq="Y")
    if len(text) != 7 or text[4] != "-":
        raise ValueError(f"Expected YYYY-MM or YYYY reference/target period: {text!r}")
    return pd.Period(text, freq="M")


def validate_macro_table(table: pd.DataFrame) -> pd.DataFrame:
    """Validate definitions and versions, returning a normalized copy."""
    missing = set(REQUIRED_COLUMNS) - set(table)
    if missing:
        raise ValueError(f"Missing macro columns: {sorted(missing)}")
    result = table.copy()
    defaults = {
        "is_forecast": False, "geographic_level": "regional", "forecast_issue_date": pd.NaT,
        "target_period": "", "aggregation_basis": "", "forecast_low": np.nan,
        "forecast_central": np.nan, "forecast_high": np.nan, "forecast_producer": "",
        "survey_issue_period": "",
    }
    for column, default in defaults.items():
        if column not in result:
            result[column] = default
    if result.empty:
        return result
    for column in ("region_id", "indicator", "reference_period", "unit", "vintage_id", "availability_status", "source_url"):
        if result[column].isna().any() or result[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Missing macro definition: {column}.")
    result["region_id"] = result["region_id"].map(normalize_region_id)
    result["availability_status"] = result["availability_status"].astype(str)
    if not result["availability_status"].isin(["A", "B", "C"]).all():
        raise ValueError("Availability status must be A, B, or C.")
    result["is_forecast"] = result["is_forecast"].map(_boolean)
    result["value"] = pd.to_numeric(result["value"], errors="raise")
    if not np.isfinite(result["value"]).all():
        raise ValueError("Macro values must be finite; missing sources use missing rows, not fabricated values.")
    for column in ("published_at", "available_at", "forecast_issue_date"):
        result[column] = pd.to_datetime(result[column], errors="raise")
        if result[column].dt.tz is not None:
            raise ValueError("Macro dates must use a consistent timezone-free date convention.")
    verified = result["availability_status"].eq("A")
    if result.loc[verified, ["published_at", "available_at"]].isna().any().any():
        raise ValueError("Status A requires verified publication and version availability dates.")
    dated = result["published_at"].notna() & result["available_at"].notna()
    if (result.loc[dated, "available_at"] < result.loc[dated, "published_at"]).any():
        raise ValueError("A version cannot be available before publication.")
    result["reference_period"] = result["reference_period"].map(lambda value: str(_period(value)))
    result["target_period"] = result["target_period"].fillna("").astype(str)
    result["aggregation_basis"] = result["aggregation_basis"].fillna("").astype(str)
    result["geographic_level"] = result["geographic_level"].fillna("").astype(str)
    national = result["region_id"].eq("RU")
    if not result.loc[national, "geographic_level"].eq("national").all():
        raise ValueError("National values must be explicitly labelled geographic_level=national.")
    if result.loc[~national, "geographic_level"].eq("national").any():
        raise ValueError("A national forecast cannot be relabelled as a regional value.")
    for row in result.itertuples(index=False):
        reference = _period(row.reference_period)
        if row.is_forecast:
            month_only = row.availability_status == "B" and bool(str(row.survey_issue_period)) and pd.isna(row.forecast_issue_date)
            if (pd.isna(row.forecast_issue_date) and not month_only) or not row.target_period or not row.aggregation_basis:
                raise ValueError("Forecasts require issue date, target period, and aggregation basis.")
            if month_only and _period(row.survey_issue_period).freqstr != "M":
                raise ValueError("A month-only survey edition requires YYYY-MM survey_issue_period.")
            _period(row.target_period)
            if pd.notna(row.available_at) and pd.notna(row.forecast_issue_date) and row.forecast_issue_date > row.available_at:
                raise ValueError("A forecast cannot be available before its issue date.")
            if not row.forecast_producer:
                raise ValueError("Forecast producer must distinguish survey participants from the central bank.")
            if row.indicator.endswith("_pct") and str(row.unit).strip().lower() not in {"%", "percent", "growth_pct", "percent_growth", "percent_change"}:
                raise ValueError("Forecast growth percentages cannot be supplied as percent indices or levels.")
            bounds = [row.forecast_low, row.forecast_central, row.forecast_high]
            known_bounds = [float(value) for value in bounds if pd.notna(value)]
            if not np.isfinite(known_bounds).all() or known_bounds != sorted(known_bounds):
                raise ValueError("Forecast low/central/high must be finite and ordered when provided.")
        elif row.indicator in ACTUAL_INDICATORS:
            if reference.freqstr != "M":
                raise ValueError("Monthly CPI/wage features cannot use an annual reference period.")
            unit = str(row.unit).strip().lower()
            if row.indicator.startswith("cpi_"):
                if unit not in INDEX_UNITS or row.value <= 0:
                    raise ValueError("CPI must be a positive percent index, not a growth rate or price level.")
                allowed_basis = {"month_to_month", "month"} if row.indicator == "cpi_mom_index" else {"year_on_year"}
            else:
                if unit not in WAGE_UNITS or row.value <= 0:
                    raise ValueError("Nominal wage must be a positive monthly RUB level.")
                allowed_basis = {"month"}
            if row.aggregation_basis not in allowed_basis:
                raise ValueError("Monthly, year-on-year, and cumulative indicators are not interchangeable.")
    keys = ["region_id", "indicator", "reference_period", "vintage_id", "is_forecast", "target_period"]
    if result.duplicated(keys).any():
        raise ValueError("Duplicate macro version keys.")
    return result.reset_index(drop=True)


def _validate_scenario(scenario_lags: dict[str, int] | None) -> dict[str, int]:
    lags = dict(scenario_lags or {})
    for indicator, lag in lags.items():
        if isinstance(lag, bool) or not isinstance(lag, Integral) or lag < 0:
            raise ValueError(f"Scenario lag for {indicator} must be a nonnegative integer number of months.")
    return lags


def _eligible(table: pd.DataFrame, as_of: pd.Timestamp, lags: dict[str, int]) -> pd.DataFrame:
    subset = table.copy()
    subset["scenario_available_at"] = pd.NaT
    subset["_effective_at"] = subset["available_at"]
    scenario = subset["availability_status"].eq("B") & subset["indicator"].isin(lags)
    for index in subset.index[scenario]:
        row = subset.loc[index]
        if bool(row["is_forecast"]):
            if pd.notna(row["forecast_issue_date"]):
                reference = row["forecast_issue_date"].to_period("M")
            elif pd.notna(row["survey_issue_period"]) and str(row["survey_issue_period"]):
                reference = _period(row["survey_issue_period"])
            else:
                continue
        else:
            reference = _period(row["reference_period"])
        if reference.freqstr != "M":
            continue
        effective = (reference + lags[row["indicator"]]).to_timestamp(how="end").normalize()
        subset.loc[index, "scenario_available_at"] = effective
        subset.loc[index, "_effective_at"] = effective
    accepted_status = subset["availability_status"].eq("A") | (scenario & subset["scenario_available_at"].notna())
    eligible = accepted_status & subset["_effective_at"].notna() & subset["_effective_at"].le(as_of)
    actual = ~subset["is_forecast"]
    reference_not_future = subset["reference_period"].map(lambda value: _period(value).start_time <= as_of.to_period("M").start_time)
    eligible &= ~actual | reference_not_future
    # For a B edition only its month may be known; its assumed month-end
    # availability is separate from the verified issue/publication date.
    eligible &= actual | subset["forecast_issue_date"].le(as_of) | (scenario & subset["forecast_issue_date"].isna())
    subset["_issue_sort"] = subset["forecast_issue_date"]
    month_only = subset["is_forecast"] & subset["_issue_sort"].isna()
    for index in subset.index[month_only]:
        issue = subset.loc[index, "survey_issue_period"]
        if pd.notna(issue) and str(issue):
            subset.loc[index, "_issue_sort"] = _period(issue).to_timestamp(how="end").normalize()
    return subset.loc[eligible].copy()


def _latest_version(rows: pd.DataFrame) -> pd.Series | None:
    if rows.empty:
        return None
    # B only has a current vintage; ordering it uses that actual vintage's date,
    # not an invented chronology. The scenario date remains explicitly separate.
    ordered = rows.sort_values(["available_at", "published_at", "vintage_id"], na_position="first")
    chosen = ordered.iloc[-1]
    tied = ordered.loc[ordered["available_at"].eq(chosen["available_at"]) & ordered["published_at"].eq(chosen["published_at"])]
    if tied["value"].nunique() > 1:
        raise ValueError("Conflicting macro vintages have indistinguishable availability dates.")
    return chosen


def _actual_at(rows: pd.DataFrame, region: str | None, indicator: str, period: str | None = None) -> pd.Series | None:
    if region is None:
        return None
    choices = rows.loc[~rows["is_forecast"] & rows["region_id"].eq(region) & rows["indicator"].eq(indicator)]
    if period is not None:
        choices = choices.loc[choices["reference_period"].eq(period)]
    elif not choices.empty:
        latest = max(choices["reference_period"], key=lambda value: _period(value).start_time)
        choices = choices.loc[choices["reference_period"].eq(latest)]
    return _latest_version(choices)


def _feature_record(sample: pd.Series, feature: str, components: list[pd.Series], value: float, reason: str = "") -> dict:
    as_of = sample["as_of_date"]
    base = {
        "sample_id": sample["sample_id"], "municipality_id": sample["municipality_id"],
        "as_of_date": as_of, "feature": feature, "value": value,
        "region_id": sample.get("region_id", np.nan), "geographic_level": "regional",
        "reference_period": "", "target_period": "", "published_at": pd.NaT,
        "available_at": pd.NaT, "scenario_available_at": pd.NaT, "vintage_id": "",
        "source_status": "", "availability_status": "", "source_url": "",
        "aggregation_basis": "", "unit": "", "age_months": np.nan,
        "forecast_issue_date": pd.NaT, "survey_issue_period": "", "forecast_producer": "",
        "missing": not np.isfinite(value), "missing_reason": reason,
        "scenario": False, "components": "[]",
    }
    if not components:
        return base
    primary = components[0]
    base.update({key: primary[key] for key in (
        "region_id", "geographic_level", "reference_period", "target_period", "published_at",
        "available_at", "scenario_available_at", "vintage_id", "source_url", "aggregation_basis", "unit",
        "forecast_issue_date", "survey_issue_period", "forecast_producer",
    )})
    statuses = sorted({row["availability_status"] for row in components})
    base["source_status"] = "+".join(statuses)
    base["availability_status"] = "B" if "B" in statuses else "A"
    base["scenario"] = "B" in statuses
    for date_column in ("published_at", "available_at"):
        dates = [row[date_column] for row in components if pd.notna(row[date_column])]
        base[date_column] = max(dates) if dates else pd.NaT
    if base["scenario"]:
        base["scenario_available_at"] = max(row["_effective_at"] for row in components)
    if feature.endswith("_growth_pct"):
        base["unit"] = "percent_growth"
    if feature in DERIVED_FEATURES:
        base["aggregation_basis"] = "year_on_year"
    primary_period = _period(primary["reference_period"]).asfreq("M", how="end")
    if bool(primary["is_forecast"]):
        primary_period = primary["_issue_sort"].to_period("M")
    base["age_months"] = int(as_of.to_period("M").ordinal - primary_period.ordinal)
    details = []
    for row in components:
        details.append({key: str(row[key]) for key in (
            "indicator", "region_id", "reference_period", "value", "available_at",
            "scenario_available_at", "vintage_id", "availability_status", "source_url",
            "published_at", "unit", "aggregation_basis", "geographic_level", "forecast_producer",
        )})
    base["components"] = json.dumps(details, ensure_ascii=False, sort_keys=True)
    return base


def build_macro_features(
    samples: pd.DataFrame,
    macro: pd.DataFrame,
    mapping: pd.DataFrame | None = None,
    *,
    scenario_lags: dict[str, int] | None = None,
    indicators: list[str] | tuple[str, ...] | None = None,
    forecast_indicators: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return wide features and long provenance, without changing query keys.

    ``as_of_date`` is the historical example's r, never the later fit date O.
    National forecasts are explicitly broadcast as national annual features;
    their target year is matched to each query's ``target_period``. Annual
    values remain annual. Explicit ``forecast_indicators`` keep unavailable
    forecast sources visible, even when their table has no rows. Every feature
    also has ``_age_months`` and ``_missing``.
    """
    required = {"sample_id", "municipality_id", "as_of_date"}
    if not required.issubset(samples):
        raise ValueError(f"Missing sample columns: {sorted(required - set(samples))}")
    if samples["sample_id"].isna().any() or samples["sample_id"].duplicated().any():
        raise ValueError("sample_id must be nonempty and unique.")
    queries = attach_regions(samples, mapping) if mapping is not None else samples.copy()
    if "region_id" not in queries:
        queries["region_id"] = np.nan
    queries["region_id"] = queries["region_id"].map(lambda value: normalize_region_id(value) if pd.notna(value) else np.nan)
    queries["as_of_date"] = pd.to_datetime(queries["as_of_date"], errors="raise")
    if queries["as_of_date"].isna().any() or queries["as_of_date"].dt.tz is not None:
        raise ValueError("Samples require timezone-free as-of dates.")
    queries["municipality_id"] = queries["municipality_id"].astype(str)
    table = validate_macro_table(macro)
    lags = _validate_scenario(scenario_lags)
    selected = tuple(indicators) if indicators is not None else ACTUAL_INDICATORS
    forecasts = sorted(table.loc[table["is_forecast"], "indicator"].unique()) if forecast_indicators is None else list(forecast_indicators)
    if len(forecasts) != len(set(forecasts)) or any(not isinstance(name, str) or not name for name in forecasts):
        raise ValueError("Forecast feature names must be unique nonempty strings.")
    feature_names = [name.replace("_index", "_growth_pct") if name.startswith("cpi_") else name for name in selected]
    if "nominal_wage_rub" in selected:
        feature_names.extend(DERIVED_FEATURES)
    feature_names.extend(name for name in forecasts if name not in feature_names)
    records = []
    availability_cache = {}
    feature_cache = {}
    for _, sample in queries.iterrows():
        as_of = sample["as_of_date"]
        if as_of not in availability_cache:
            availability_cache[as_of] = _eligible(table, as_of, lags)
        available = availability_cache[as_of]
        region = sample["region_id"] if pd.notna(sample["region_id"]) else None
        target_key = str(sample.get("target_period", ""))
        cache_key = (as_of, region, target_key)
        if cache_key in feature_cache:
            records.extend(dict(record, sample_id=sample["sample_id"], municipality_id=sample["municipality_id"])
                           for record in feature_cache[cache_key])
            continue
        first_record = len(records)
        for indicator in selected:
            row = _actual_at(available, region, indicator)
            feature = indicator.replace("_index", "_growth_pct") if indicator.startswith("cpi_") else indicator
            value = np.nan if row is None else float(row["value"])
            if row is not None and indicator in ("cpi_mom_index", "cpi_yoy_index"):
                value -= 100.0
            reason = "" if row is not None else "region_unmapped" if region is None else "no_available_observation"
            records.append(_feature_record(sample, feature, [] if row is None else [row], value, reason))
        if "nominal_wage_rub" in selected:
            current = _actual_at(available, region, "nominal_wage_rub")
            previous = None if current is None else _actual_at(available, region, "nominal_wage_rub", str(_period(current["reference_period"]) - 12))
            components = [row for row in (current, previous) if row is not None]
            growth = np.nan if current is None or previous is None else 100.0 * (float(current["value"]) / float(previous["value"]) - 1.0)
            reason = "" if np.isfinite(growth) else "missing_aligned_wage_t_or_t_minus_12"
            records.append(_feature_record(sample, DERIVED_FEATURES[0], components, growth, reason))
            prices = None if current is None else _actual_at(available, region, "cpi_yoy_index", current["reference_period"])
            real = np.nan if not np.isfinite(growth) or prices is None else 100.0 * ((1.0 + growth / 100.0) / (float(prices["value"]) / 100.0) - 1.0)
            reason = "" if np.isfinite(real) else "missing_aligned_wage_or_cpi_yoy"
            records.append(_feature_record(sample, DERIVED_FEATURES[1], components + ([] if prices is None else [prices]), real, reason))
        for indicator in forecasts:
            choices = available.loc[available["is_forecast"] & available["indicator"].eq(indicator)]
            choices = choices.loc[choices["region_id"].eq("RU") | choices["region_id"].eq(region)]
            target = sample.get("target_period", "")
            if pd.isna(target) or not str(target):
                choices = choices.iloc[:0]
            else:
                target_period = pd.Timestamp(target).to_period("M") if isinstance(target, pd.Timestamp) else _period(str(target)[:7])
                choices = choices.loc[choices["target_period"].map(lambda value: _period(value).year == target_period.year if _period(value).freqstr.startswith("Y") else _period(value) == target_period)]
            if len(choices) and choices["aggregation_basis"].nunique() > 1:
                raise ValueError("Forecast feature combines incompatible aggregation bases; use separate indicator names.")
            if len(choices) and choices["geographic_level"].nunique() > 1:
                raise ValueError("National and regional forecasts need separate feature definitions.")
            row = None if choices.empty else _latest_version(choices.loc[choices["_issue_sort"].eq(choices["_issue_sort"].max())])
            value = np.nan if row is None else float(row["forecast_central"]) if pd.notna(row["forecast_central"]) else float(row["value"])
            records.append(_feature_record(sample, indicator, [] if row is None else [row], value, "" if row is not None else "no_available_forecast_for_target_year"))
        feature_cache[cache_key] = records[first_record:]
    provenance = pd.DataFrame(records, columns=PROVENANCE_COLUMNS)
    X = pd.DataFrame(index=pd.Index(queries["sample_id"], name="sample_id"))
    for feature in feature_names:
        subset = provenance.loc[provenance["feature"].eq(feature)].set_index("sample_id")
        X[feature] = subset["value"].reindex(X.index).astype(float)
        X[f"{feature}_age_months"] = subset["age_months"].reindex(X.index).astype(float)
        X[f"{feature}_missing"] = subset["missing"].reindex(X.index).fillna(True).astype(bool)
    return X, provenance


def build_price_index(
    macro: pd.DataFrame,
    region_id: str,
    as_of_date: str | pd.Timestamp,
    *,
    base_period: str = "2022-12",
    scenario_lags: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Price level = 100 * product(CPI_mom / 100), with a fixed base.

    One historical query uses only vintages available on its own as-of date.
    A missing month breaks the chain: all subsequent levels remain unavailable.
    This produces observed history, never a trajectory of future actual CPI.
    """
    table = validate_macro_table(macro)
    as_of = pd.Timestamp(as_of_date)
    base = _period(base_period)
    if base.freqstr != "M":
        raise ValueError("Price-index base must be a monthly period.")
    if base > as_of.to_period("M"):
        raise ValueError("Price-index base cannot be in the future.")
    rows = _eligible(table, as_of, _validate_scenario(scenario_lags))
    region = normalize_region_id(region_id)
    results = [{"reference_period": str(base), "price_index": 100.0, "missing": False, "missing_reason": "fixed_base", "availability_status": "base", "vintage_id": "", "source_url": ""}]
    level = 100.0
    chain_status = "A"
    broken = False
    for month in pd.period_range(base + 1, as_of.to_period("M"), freq="M"):
        row = _actual_at(rows, region, "cpi_mom_index", str(month))
        if row is None:
            broken = True
        if not broken:
            level *= float(row["value"]) / 100.0
            if row["availability_status"] == "B":
                chain_status = "B"
        results.append({
            "reference_period": str(month), "price_index": np.nan if broken else level,
            "missing": broken, "missing_reason": "missing_current_or_previous_month" if broken else "",
            "availability_status": "" if broken else chain_status,
            "vintage_id": "" if row is None else row["vintage_id"],
            "source_url": "" if row is None else row["source_url"],
        })
    return pd.DataFrame(results)
