"""Каждая точка выпуска обучается заново; истина присоединяется ПОСЛЕ прогнозов."""
from __future__ import annotations
from pathlib import Path
import time
import numpy as np
import pandas as pd
from .data import get_prefix, eligibility, period_end
from .models import baseline_predict, catboost_predict, prophet_predict, BASELINES


def origins_from_config(cfg: dict) -> list[pd.Period]:
    bt = cfg["backtest"]
    return list(pd.period_range(bt["first_origin"], bt["last_origin"], freq="M"))


def run_origin(panel: pd.DataFrame, cfg: dict, origin: pd.Period,
               sample_ids: set[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, list[dict], pd.DataFrame, list[dict]]:
    lag = int(cfg["data"]["release_lag_months"])
    bt = cfg["backtest"]
    prefix = get_prefix(panel, origin, lag)
    cohort = eligibility(prefix, bt["min_history_observations"], bt["max_staleness_months"])
    cohort["in_sample"] = True if sample_ids is None else cohort["municipality_id"].isin(sample_ids)
    cohort["selected"] = cohort["eligible"] & cohort["in_sample"]
    cohort["forecast_origin"] = str(period_end(origin).date())
    ids = cohort.loc[cohort["selected"], "municipality_id"].tolist()
    horizons = [int(h) for h in bt["horizons"] if (origin + int(h)).to_timestamp() <= panel.index.max()]
    if not ids or not horizons:
        return pd.DataFrame(), cohort, [], pd.DataFrame(), []
    steps = lag + max(horizons)
    history = prefix[ids]
    records, timings, imps, errors = [], [], [], []
    for name in cfg["models"]["enabled"]:
        started = time.perf_counter()
        meta = {"n_training_rows": int(history.notna().sum().sum()),
                "max_training_target": str(prefix.index[-1].date())}
        if name in BASELINES:
            predictions, flags = baseline_predict(history, steps, name, cfg["models"])
            status_type = "fallback_last_value"
        elif name == "CatBoostRecursive":
            # Обучение на всех известных МО; прогноз — на общем допустимом наборе.
            predictions, meta, importance = catboost_predict(prefix, history, steps,
                                                            cfg["models"]["catboost"], cfg["seed"])
            flags = np.zeros_like(predictions, dtype=bool)
            status_type = "failed"
            importance["forecast_origin"] = str(period_end(origin).date()); imps.append(importance)
        else:
            predictions, flags, model_errors = prophet_predict(history, steps, name,
                                                               cfg["models"]["prophet"], cfg["seed"])
            status_type = "failed"
            for error in model_errors:
                error["forecast_origin"] = str(period_end(origin).date())
            errors.extend(model_errors)
        timings.append({"forecast_origin": str(period_end(origin).date()), "model": name,
                        "seconds": time.perf_counter()-started, "n_forecast_series": len(ids), **meta})
        for h in horizons:
            idx = lag + h - 1
            target = (origin + h).to_timestamp()
            # Этот доступ к будущей истине используется ТОЛЬКО для последующей оценки.
            truth = panel.loc[target, ids].to_numpy(dtype=float)
            rec = pd.DataFrame({
                "municipality_id": ids,
                "forecast_origin": str(period_end(origin).date()),
                "history_cutoff": str(period_end(origin-lag).date()),
                "target_period": str(target.date()), "horizon": h,
                "model": name, "y_true": truth, "y_pred": predictions[idx],
                "status": np.where(flags[idx], status_type, "native"),
                "availability_assumption": f"month_end_plus_{lag}_months",
            })
            records.append(rec)
    return (pd.concat(records, ignore_index=True), cohort, timings,
            pd.concat(imps, ignore_index=True) if imps else pd.DataFrame(), errors)


def fixed_sample(panel: pd.DataFrame, cfg: dict) -> set[str] | None:
    limit = cfg["backtest"].get("max_series")
    if limit is None:
        return None
    first = origins_from_config(cfg)[0]
    prefix = get_prefix(panel, first, cfg["data"]["release_lag_months"])
    tab = eligibility(prefix, cfg["backtest"]["min_history_observations"], cfg["backtest"]["max_staleness_months"])
    ids = tab.loc[tab.eligible, "municipality_id"].to_numpy()
    chosen = np.random.default_rng(cfg["seed"]).choice(ids, size=min(len(ids), int(limit)), replace=False)
    return set(chosen.tolist())
