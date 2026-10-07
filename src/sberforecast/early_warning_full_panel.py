"""E07b full-panel adapter to the frozen E01/E07a definitions, without fitting."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .data import eligibility, get_prefix, period_end
from .models import baseline_predict


def resolve_config(root: Path, config_path: Path) -> dict:
    new = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    inherited_path = (root / new["protocol"]["path"]).resolve()
    if not inherited_path.is_relative_to(root):
        raise ValueError("Protocol must remain inside project")
    if hashlib.sha256(inherited_path.read_bytes()).hexdigest() != new["protocol"]["sha256"]:
        raise ValueError("Frozen E07a protocol SHA changed")
    old = yaml.safe_load(inherited_path.read_text(encoding="utf-8"))
    cfg = copy.deepcopy(old)
    # These replace E07b execution options only; inherited label/data rules stay.
    for key in ("experiment","experiment_date","seed","protocol","panel","split","gate","models","features","runtime",
                "output_dir","smoke_output_dir","report_path"):
        cfg[key] = copy.deepcopy(new[key])
    cfg["detector_preparation"] = yaml.safe_load((root / new["features"]["detector_preparation"]).read_text(encoding="utf-8"))["preparation"]
    reference = yaml.safe_load((root / new["panel"]["eligibility_source"]).read_text(encoding="utf-8"))
    # Actual E01 values, not guessed settings, determine all new eligibility.
    source = reference.get("data",{})
    backtest = reference["backtest"]
    original_forecast = reference["models"]
    checks = {
        "release_lag_months": (cfg["data"]["release_lag_months"],source["release_lag_months"]),
        "first_origin": (cfg["panel"]["first_origin"],backtest["first_origin"]),
        "last_origin": (cfg["panel"]["last_origin"],backtest["last_origin"]),
        "min_history_observations": (cfg["panel"]["min_history_observations"],backtest["min_history_observations"]),
        "max_staleness_months": (cfg["panel"]["max_staleness_months"],backtest["max_staleness_months"]),
    }
    if source.get("category") and source["category"] != cfg["data"]["category"]:
        raise ValueError("E07b changes target category")
    if any(left != right for left,right in checks.values()):
        raise ValueError(f"E07b changes E01 calendar/eligibility: {checks}")
    for key in ("yearly_growth_window","yearly_growth_bounds"):
        if cfg["panel"][key] != original_forecast[key]:
            raise ValueError("E07b changes saved seasonal baseline growth")
    if cfg["runtime"]["parallel_jobs"] > 2 or cfg["runtime"]["memory_stop_gib"] > 5.5:
        raise ValueError("E07b laptop execution bounds changed")
    return cfg


def prepare_panel_residuals(panel: pd.DataFrame, config: dict, geography: pd.DataFrame | None = None,
                            progress=None) -> tuple[pd.DataFrame,pd.DataFrame]:
    """Every UID/O audit row; h1 prediction only for prefix-eligible UIDs.

    This invokes the existing non-fitted seasonal baseline, not an old backtest
    or trained forecasting model. The target fact is read after the prediction;
    it cannot determine eligibility, cohort or y_pred. Absent forecasts stay NaN.
    """
    lag = config["data"]["release_lag_months"]
    settings = config["panel"]
    if lag != 0 or settings["residual_model"] != "SeasonalNaiveYoY" or settings["residual_horizon"] != 1:
        raise ValueError("E07b retains E07a L0 and SeasonalNaiveYoY h1")
    if panel.columns.astype(str).duplicated().any():
        raise ValueError("Duplicate municipality identity")
    samples, residuals = [],[]
    ids = panel.columns.astype(str).tolist()
    for origin in pd.period_range(settings["first_origin"],settings["last_origin"],freq="M"):
        prefix = get_prefix(panel,origin,lag)
        cutoff = (origin-lag).to_timestamp()
        audit = eligibility(prefix,settings["min_history_observations"],settings["max_staleness_months"])
        valid = audit.loc[audit.eligible,"municipality_id"].tolist()
        pred = np.full(len(ids),np.nan)
        fallback = np.full(len(ids),False)
        if valid:
            values,flags = baseline_predict(prefix.loc[:,valid],1,"SeasonalNaiveYoY",settings)
            positions = panel.columns.get_indexer(valid)
            pred[positions],fallback[positions] = values[0],flags[0]
        target = origin+1
        truth = panel.loc[target.start_time].to_numpy(dtype=float) if target.start_time in panel.index else np.full(len(ids),np.nan)
        for position,row in enumerate(audit.itertuples(index=False)):
            issued = str(period_end(origin).date())
            sample = dict(municipality_id=str(row.municipality_id),forecast_origin=issued,
                eligible_at_origin=bool(row.eligible),n_history=int(row.n_history),staleness_months=int(row.staleness_months),
                eligibility_reason="eligible" if row.eligible else "insufficient_prefix_history" if row.n_history<settings["min_history_observations"] else "stale_prefix_history")
            samples.append(sample)
            residuals.append(dict(series_id=str(row.municipality_id),observation_period=str(target),
                y_true=float(truth[position]) if row.eligible else np.nan,y_pred=float(pred[position]),
                model="SeasonalNaiveYoY" if row.eligible else None,horizon=1 if row.eligible else np.nan,
                forecast_origin=issued if row.eligible else None,history_cutoff=str(cutoff.date()) if row.eligible else None,
                source_prediction_present=bool(row.eligible),forecast_fallback=bool(fallback[position]),
                forecast_status="seasonal_fallback" if row.eligible and fallback[position] else "native" if row.eligible else "ineligible_no_forecast",
                availability_assumption=config["data"]["availability_assumption"]))
        if progress:
            progress(dict(stage="residuals",origin=str(origin),municipalities=len(ids),eligible=len(valid)))
    queries = pd.DataFrame(samples)
    if geography is not None:
        mapping = geography[["municipality_id","region_id","region"]].drop_duplicates("municipality_id").copy()
        mapping["municipality_id"] = mapping.municipality_id.astype(str)
        queries = queries.merge(mapping,on="municipality_id",how="left",validate="many_to_one",sort=False)
        if queries.region_id.isna().any():
            raise ValueError("Some full-panel municipalities are absent from frozen geography dictionary")
    return queries,pd.DataFrame(residuals)


def compare_saved_pilot(residuals: pd.DataFrame, saved: pd.DataFrame) -> dict:
    """Exact pilot predictions/errors must remain unchanged when adding UIDs."""
    keys=["series_id","observation_period"]
    generated=residuals.merge(saved[keys],on=keys,how="inner",validate="one_to_one")
    comparison=saved.merge(generated,on=keys,how="left",suffixes=("_old","_new"),validate="one_to_one")
    if len(comparison)!=len(saved) or len(generated)!=len(saved):
        raise AssertionError("Full panel lost pilot residual keys")
    differences={}
    for column in ("y_true","y_pred"):
        left,right=comparison[column+"_old"].to_numpy(float),comparison[column+"_new"].to_numpy(float)
        if not np.array_equal(np.isnan(left),np.isnan(right)) or not np.allclose(left,right,rtol=0,atol=1e-8,equal_nan=True):
            raise AssertionError(f"Pilot {column} changed")
        finite=np.isfinite(left)&np.isfinite(right)
        differences[column]=float(np.max(np.abs(left[finite]-right[finite]))) if finite.any() else 0.
    return dict(passed=True,rows=len(saved),max_abs_difference=differences,
                target_or_history_used_for_eligibility=False,forecast_model_fit=False)
