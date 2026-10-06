"""E05a coverage diagnostics. No forecast model is fitted or evaluated here."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import get_prefix, period_end
from .direct_training import build_direct_training
from .macro_features import attach_regions, build_macro_features

KEY = ["municipality_id", "forecast_origin", "target_period", "horizon"]


def e01_cases(predictions: pd.DataFrame) -> pd.DataFrame:
    """Check actual key/value agreement across models, then keep one copy."""
    frame = predictions.copy()
    frame["municipality_id"] = frame["municipality_id"].astype(str)
    for column in ("forecast_origin", "target_period"):
        frame[column] = pd.to_datetime(frame[column])
    first = None
    for model, group in frame.groupby("model", sort=True):
        if group.duplicated(KEY).any():
            raise ValueError(f"Duplicate E01 keys for {model}.")
        cases = group.set_index(KEY)["y_true"].sort_index()
        if first is None:
            first = cases
        elif not cases.index.equals(first.index) or not np.allclose(cases, first, equal_nan=True):
            raise ValueError(f"E01 support/targets differ for {model}.")
    if first is None:
        raise ValueError("E01 predictions are empty.")
    result = first.rename("y_true").reset_index()
    result["e01_case"] = True
    result["e01_evaluable"] = np.isfinite(result["y_true"])
    return result


def forecast_grid(ids: list[str], origins: pd.PeriodIndex, horizons: list[int],
                  cases: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    """Keep all 64 municipalities, including absent/out-of-range E01 cases."""
    records = [{"municipality_id": str(uid), "forecast_origin": period_end(origin),
                "target_period": (origin + h).to_timestamp(), "horizon": h}
               for origin in origins for h in horizons for uid in ids]
    grid = pd.DataFrame(records)
    if grid.duplicated(KEY).any():
        raise ValueError("Duplicate forecast grid keys.")
    if len(cases.merge(grid[KEY], on=KEY, how="inner", validate="one_to_one")) != len(cases):
        raise ValueError("Configured forecast grid omits E01 cases.")
    grid = grid.merge(cases, on=KEY, how="left", validate="one_to_one")
    grid["e01_case"] = grid["e01_case"].eq(True)
    grid["e01_evaluable"] = grid["e01_evaluable"].eq(True)
    grid["sample_id"] = [f"forecast-{i}" for i in range(len(grid))]
    grid["as_of_date"] = grid["forecast_origin"]
    return attach_regions(grid, mapping)


def forecast_coverage(grid: pd.DataFrame, provenance: pd.DataFrame) -> pd.DataFrame:
    merged = provenance.merge(grid[["sample_id", "forecast_origin", "horizon", "region_id",
                                   "e01_case", "e01_evaluable"]].rename(columns={"region_id": "sample_region_id"}),
                              on="sample_id", how="left", validate="many_to_one")
    rows = []
    for (origin, horizon, feature), group in merged.groupby(["forecast_origin", "horizon", "feature"], sort=True):
        present = ~group["missing"]
        rows.append({"forecast_origin": origin, "horizon": horizon, "feature": feature,
                     "n_cohort_cases": len(group), "n_cohort_regions": group.sample_region_id.nunique(),
                     "n_available_cohort_cases": int(present.sum()),
                     "n_available_cohort_regions": group.loc[present, "sample_region_id"].nunique(),
                     "n_e01_cases": int(group.e01_case.sum()),
                     "n_available_e01_cases": int((present & group.e01_case).sum()),
                     "n_e01_evaluable": int(group.e01_evaluable.sum()),
                     "n_available_e01_evaluable": int((present & group.e01_evaluable).sum()),
                     "n_A_cohort_cases": int((present & group.availability_status.eq("A")).sum()),
                     "n_B_cohort_cases": int((present & group.availability_status.eq("B")).sum()),
                     "source_geography": ";".join(sorted(group.loc[present, "geographic_level"].unique()))})
    return pd.DataFrame(rows)


def training_groups(panel: pd.DataFrame, origins: pd.PeriodIndex, horizons: list[int],
                    mapping: pd.DataFrame, *, release_lag_months: int,
                    mode: str, max_staleness_months: int | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build genuine E02 pairs; collapse only identical macro queries afterwards.

    Each group retains the exact set of municipalities and count of pairs.
    Historical query dates are r; current model date O is only a grouping key.
    """
    groups, counts = [], []
    for origin in origins:
        for horizon in horizons:
            X, y, meta = build_direct_training(panel, origin, horizon,
                release_lag_months=release_lag_months, mode=mode,
                max_staleness_months=max_staleness_months)
            del X, y
            counts.append({"model_origin": period_end(origin), "horizon": horizon,
                           "mode": mode, "n_pairs": len(meta),
                           "n_municipalities": meta.municipality_id.nunique(),
                           "n_historical_dates": meta.historical_origin.nunique()})
            if meta.empty:
                continue
            meta = attach_regions(meta, mapping)
            if meta.region_id.isna().any():
                raise ValueError("Unmapped historical training municipality.")
            # No monthly forecast currently exists: configured forecasts are annual.
            meta["target_year"] = pd.to_datetime(meta.target_period).dt.year.astype(str)
            keys = ["region_id", "historical_origin", "feature_cutoff", "target_year"]
            part = meta.groupby(keys, sort=True).agg(
                n_pairs=("municipality_id", "size"),
                municipality_ids=("municipality_id", lambda values: json.dumps(sorted(set(values)), ensure_ascii=False)),
            ).reset_index()
            part["model_origin"] = period_end(origin)
            part["horizon"] = horizon
            part["mode"] = mode
            groups.append(part)
    columns = ["region_id", "historical_origin", "feature_cutoff", "target_year",
               "n_pairs", "municipality_ids", "model_origin", "horizon", "mode"]
    result = pd.concat(groups, ignore_index=True) if groups else pd.DataFrame(columns=columns)
    return result, pd.DataFrame(counts)


def training_queries(groups: pd.DataFrame, mapping: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["region_id", "historical_origin", "feature_cutoff", "target_year"]
    queries = groups[keys].drop_duplicates().sort_values(keys).reset_index(drop=True)
    queries["sample_id"] = [f"historical-{i}" for i in range(len(queries))]
    representatives = mapping.drop_duplicates("region_id").set_index("region_id")["municipality_id"]
    queries["municipality_id"] = queries.region_id.map(representatives)
    queries["representative_municipality_only"] = True
    queries["as_of_date"] = queries.historical_origin
    queries["target_period"] = queries.target_year
    with_ids = groups.merge(queries[keys + ["sample_id"]], on=keys, how="left", validate="many_to_one")
    return queries, with_ids


def training_coverage(groups: pd.DataFrame, counts: pd.DataFrame, provenance: pd.DataFrame,
                      features: list[str]) -> pd.DataFrame:
    rows = []
    for total in counts.itertuples(index=False):
        local = groups.loc[groups.model_origin.eq(total.model_origin) & groups.horizon.eq(total.horizon) & groups["mode"].eq(total.mode)]
        merged = local.merge(provenance[["sample_id", "feature", "missing", "availability_status", "geographic_level"]],
                             on="sample_id", how="left", validate="many_to_many")
        for feature in features:
            available = merged.loc[merged.feature.eq(feature) & merged.missing.eq(False)]
            municipalities = set()
            for ids in available.municipality_ids:
                municipalities.update(json.loads(ids))
            rows.append({"model_origin": total.model_origin, "horizon": total.horizon,
                         "mode": total.mode, "feature": feature, "n_pairs": total.n_pairs,
                         "n_municipalities": total.n_municipalities, "n_historical_dates": total.n_historical_dates,
                         "n_available_pairs": int(available.n_pairs.sum()),
                         "n_available_municipalities": len(municipalities),
                         "n_available_regions": available.region_id.nunique(),
                         "n_available_historical_dates": available.historical_origin.nunique(),
                         "n_A_pairs": int(available.loc[available.availability_status.eq("A"), "n_pairs"].sum()),
                         "n_B_pairs": int(available.loc[available.availability_status.eq("B"), "n_pairs"].sum()),
                         "source_geography": ";".join(sorted(available.geographic_level.dropna().unique()))})
    return pd.DataFrame(rows)


def trend_readiness(panel: pd.DataFrame, grid: pd.DataFrame, *, release_lag_months: int,
                    window_months: int = 6, min_pairs: int = 3) -> pd.DataFrame:
    """Count seasonal/annual support without estimating slopes or forecasts."""
    records = []
    for origin, samples in grid.groupby("forecast_origin"):
        prefix = get_prefix(panel, pd.Timestamp(origin).to_period("M"), release_lag_months)
        cutoff = prefix.index[-1].to_period("M")
        positions = pd.period_range(cutoff - window_months + 1, cutoff, freq="M")
        recent = prefix.reindex(positions.to_timestamp())
        prior = prefix.reindex((positions - 12).to_timestamp())
        pairs = (np.isfinite(recent.to_numpy()) & np.isfinite(prior.to_numpy())).sum(axis=0)
        seasonal = np.isfinite(prefix.reindex(pd.period_range(cutoff - 11, cutoff, freq="M").to_timestamp()).to_numpy()).sum(axis=0)
        info = pd.DataFrame({"municipality_id": prefix.columns.astype(str),
                             "n_annual_pairs_last6": pairs, "n_seasonal_months": seasonal})
        info["native_trend_ready"] = (info.n_annual_pairs_last6 >= min_pairs) & (info.n_seasonal_months == 12)
        info["fallback_reason"] = np.select([info.n_seasonal_months.lt(12), info.n_annual_pairs_last6.lt(min_pairs)],
                                           ["incomplete_calendar_seasonal_template", "insufficient_annual_pairs"], default="")
        part = samples.merge(info, on="municipality_id", how="left", validate="many_to_one")
        part["feature_cutoff"] = period_end(cutoff)
        records.append(part)
    return pd.concat(records, ignore_index=True)


def feature_dictionary(forecast_indicators: list[str]) -> dict:
    observed = {
        "cpi_mom_growth_pct": {"formula": "CPI_mom_index - 100", "basis": "month_to_month"},
        "cpi_yoy_growth_pct": {"formula": "CPI_yoy_index - 100", "basis": "year_on_year"},
        "nominal_wage_rub": {"formula": "latest published monthly wage level", "unit": "RUB", "meaning": "wage of employees, not all residents' income"},
        "nominal_wage_yoy_growth_pct": {"formula": "100*(w[t]/w[t-12]-1)", "basis": "same calendar month"},
        "real_wage_yoy_growth_pct": {"formula": "100*((w[t]/w[t-12])/(CPI_yoy[t]/100)-1)", "basis": "aligned monthly wages and CPI"},
    }
    for name in forecast_indicators:
        observed[name] = {"formula": "latest available annual forecast matching target year", "unit": "percent_growth",
                          "geography": "national", "basis": "annual_real_volume_growth" if "consumption" in name else "december_to_december" if "dec_dec" in name else "annual_average_to_annual_average",
                          "producer": "Bank of Russia own forecast" if "consumption" in name else "survey participants' median",
                          "note": "annual feature, no division by 12; consumption range midpoint is explicitly derived" if "consumption" in name else "annual feature, no division by 12"}
    return {"features": observed, "per_feature_companions": ["age_months", "missing"],
            "strict": "A only, latest version available on each query's own date",
            "scenario": "A plus B with configured assumed availability; B never becomes A",
            "price_index": "100*product(CPI_mom/100) from 2022-12; any gap breaks subsequent levels"}


def save_feature_audit(output: Path, name: str, samples: pd.DataFrame, macro: pd.DataFrame,
                       forecast_indicators: list[str], scenario_lags: dict | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    X, provenance = build_macro_features(samples, macro, scenario_lags=scenario_lags,
                                         forecast_indicators=forecast_indicators)
    X.to_csv(output / f"{name}_features.csv.gz", compression="gzip")
    provenance.to_csv(output / f"{name}_provenance.csv.gz", index=False, compression="gzip")
    return X, provenance
