"""Synthetic monthly expenses and causal E01 SeasonalNaiveYoY residuals.

Event truth belongs to the generator output only. The forecaster receives the
monthly values and its fixed configuration, never the scenario or event dates.
The synthetic truth describes changes in expenses; it does not assert that the
adaptive forecast errors must exhibit a persistent change of the same duration.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from .data import get_prefix, period_end
from .models import baseline_predict


RESIDUAL_COLUMNS = [
    "series_id", "observation_period", "target_period", "forecast_origin",
    "history_cutoff", "fact_available_period", "fact_available_date", "horizon",
    "y_true", "y_pred", "error", "status", "n_history", "staleness_months",
]
EVENT_COLUMNS = [
    "series_id", "event_id", "event_type", "event_start_period", "strength",
    "duration_months", "direction", "is_main_protocol", "event_window_class",
]


def forecast_residuals(
    observations: pd.Series,
    forecast_cfg: Mapping,
    release_lag_months: int = 0,
    *,
    first_target_index: int = 12,
    min_history_observations: int | None = None,
    max_staleness_months: int | None = None,
) -> pd.DataFrame:
    """Issue h=1 forecasts before each target, keeping missing calendar cells.

For a target month T, origin O=T-1 and history cutoff C=O-L. Prediction
step L+1 from C is therefore still an h=1 forecast from O, exactly as in
the unchanged E01 backtest. A target fact is assumed available at T+L.
Eligibility uses the historical prefix and the E01 12-observation/one-month
staleness defaults. Ineligible rows retain their calendar date and a NaN
prediction; missing target facts always produce a NaN error.
"""
    if min_history_observations is None:
        min_history_observations = int(forecast_cfg.get("min_history_observations", 12))
    if max_staleness_months is None:
        max_staleness_months = int(forecast_cfg.get("max_staleness_months", 1))
    if release_lag_months < 0:
        raise ValueError("release_lag_months must be nonnegative")
    if first_target_index < 0 or min_history_observations < 1 or max_staleness_months < 0:
        raise ValueError("Invalid calendar start or historical eligibility")
    if observations.empty:
        return pd.DataFrame(columns=RESIDUAL_COLUMNS)
    monthly_index = pd.PeriodIndex(observations.index, freq="M").to_timestamp()
    if monthly_index.has_duplicates:
        raise ValueError("Duplicate synthetic observation month")
    values = observations.to_numpy(dtype=float)
    if np.isinf(values).any() or np.any(values[np.isfinite(values)] < 0):
        raise ValueError("Synthetic expenses must be finite nonnegative values or NaN")
    series_id = str(observations.name if observations.name is not None else "synthetic")
    history = pd.Series(values, index=monthly_index).sort_index()
    calendar = pd.date_range(history.index.min(), history.index.max(), freq="MS")
    panel = history.reindex(calendar).to_frame(name=series_id)
    records = []
    for target_date in calendar[first_target_index:]:
        target = target_date.to_period("M")
        origin = target - 1
        cutoff = origin - int(release_lag_months)
        n_history = 0
        staleness = np.nan
        prediction = np.nan
        status = "ineligible_history"
        if cutoff.to_timestamp() >= panel.index.min():
            # No future target access occurs before this copied causal prefix.
            prefix = get_prefix(panel, origin, int(release_lag_months))
            finite = np.isfinite(prefix[series_id].to_numpy(dtype=float))
            n_history = int(finite.sum())
            if n_history:
                staleness = int(len(prefix) - 1 - np.flatnonzero(finite)[-1])
            if n_history >= min_history_observations and staleness <= max_staleness_months:
                path, fallback = baseline_predict(
                    prefix, int(release_lag_months) + 1, "SeasonalNaiveYoY", dict(forecast_cfg),
                )
                prediction = float(path[int(release_lag_months), 0])
                status = "fallback_last_value" if fallback[int(release_lag_months), 0] else "native"
        # Truth is joined after prediction, as in E01; it is not a feature.
        truth = float(panel.at[target_date, series_id])
        error = truth - prediction if np.isfinite(truth) and np.isfinite(prediction) else np.nan
        records.append({
            "series_id": series_id,
            "observation_period": str(target),
            "target_period": str(target_date.date()),
            "forecast_origin": str(period_end(origin).date()),
            "history_cutoff": str(period_end(cutoff).date()),
            "fact_available_period": str(target + int(release_lag_months)),
            "fact_available_date": str(period_end(target + int(release_lag_months)).date()),
            "horizon": 1,
            "y_true": truth,
            "y_pred": prediction,
            "error": error,
            "status": status,
            "n_history": n_history,
            "staleness_months": staleness,
        })
    return pd.DataFrame.from_records(records, columns=RESIDUAL_COLUMNS)


def _cells(noise_fractions: list[float], strengths: list[float]) -> list[tuple]:
    cells = []
    for noise_fraction in noise_fractions:
        cells.append(("no_change", 0.0, noise_fraction))
        for scenario in ("outlier", "level_up", "level_down", "slope_up", "slope_down", "variance_up"):
            cells.extend((scenario, strength, noise_fraction) for strength in strengths)
        cells.extend((scenario, 4.0, noise_fraction) for scenario in ("level_warmup", "level_end"))
    return cells


def _batch_forecast_residuals(
    values: pd.DataFrame,
    forecast_cfg: Mapping,
    release_lag_months: int,
) -> pd.DataFrame:
    """Vectorize independent columns using the unchanged E01 baseline.

    Only the generator's complete finite monthly panels use this shortcut.
    It has no cross-series statistics: baseline_predict computes seasonal
    values and annual growth separately for every column. Target truth is
    joined after each causal prediction call, just as in forecast_residuals.
    """
    if release_lag_months < 0:
        raise ValueError("release_lag_months must be nonnegative")
    min_history = int(forecast_cfg.get("min_history_observations", 12))
    max_staleness = int(forecast_cfg.get("max_staleness_months", 1))
    if min_history < 1 or max_staleness < 0:
        raise ValueError("Invalid historical eligibility")
    records = []
    ids = values.columns.astype(str).tolist()
    for target_date in values.index[12:]:
        target = target_date.to_period("M")
        origin = target - 1
        cutoff = origin - release_lag_months
        n_history = max(0, int(cutoff.ordinal - values.index[0].to_period("M").ordinal + 1))
        prediction = np.full(len(ids), np.nan)
        flags = np.zeros(len(ids), dtype=bool)
        eligible = n_history >= min_history
        if eligible:
            prefix = get_prefix(values, origin, release_lag_months)
            path, fallback = baseline_predict(
                prefix, release_lag_months + 1, "SeasonalNaiveYoY", dict(forecast_cfg),
            )
            prediction = path[release_lag_months]
            flags = fallback[release_lag_months]
        truth = values.loc[target_date].to_numpy(dtype=float)
        records.append(pd.DataFrame({
            "series_id": ids,
            "observation_period": str(target),
            "target_period": str(target_date.date()),
            "forecast_origin": str(period_end(origin).date()),
            "history_cutoff": str(period_end(cutoff).date()),
            "fact_available_period": str(target + release_lag_months),
            "fact_available_date": str(period_end(target + release_lag_months).date()),
            "horizon": 1,
            "y_true": truth,
            "y_pred": prediction,
            "error": truth - prediction,
            "status": np.where(flags, "fallback_last_value", "native") if eligible else "ineligible_history",
            "n_history": n_history,
            "staleness_months": 0 if n_history else np.nan,
        }))
    return pd.concat(records, ignore_index=True).sort_values(
        ["series_id", "observation_period"], kind="stable",
    ).reset_index(drop=True)[RESIDUAL_COLUMNS]


def generate_benchmark(
    generator_cfg: Mapping,
    split: str,
    forecast_cfg: Mapping,
    release_lag_months: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return residuals, independent event truth, series metadata and observations.

    Main protocol: 24 months, fixed ordinary annual seasonality and a small
    linear trend, iid Gaussian noise. A shift of strength s has magnitude
    s*base_level*noise_fraction. A slope change reaches that extra amount at
    its third month; a variance change multiplies the noise standard deviation
    by s. Level/slope/variance changes persist to the end of the calendar.
    Outlier signs alternate by replicate and never enter structural event truth.

    Main events start in months 17--21 (indices 16--20): after four residuals
    for warmup and with a complete inclusive 0--3-month detection window.
    Warmup and end-window shifts are separately labelled edge scenarios.
    Validation and test have disjoint identifiers and disjoint per-series seeds.
    """
    if split not in {"validation", "test"}:
        raise ValueError("split must be validation or test")
    months = int(generator_cfg.get("months", 24))
    if months != 24:
        raise ValueError("The frozen E04a main protocol requires exactly 24 months")
    start_period = pd.Period(generator_cfg.get("start_period", "2023-01"), freq="M")
    base_level = float(generator_cfg.get("base_level", 10000.0))
    seasonal_fraction = float(generator_cfg.get("seasonality_fraction", 0.1))
    trend_fraction = float(generator_cfg.get("trend_per_month_fraction", 0.004))
    noise_fractions = [float(x) for x in generator_cfg.get("noise_fractions", [0.01, 0.03])]
    strengths = [float(x) for x in generator_cfg.get("strengths", [2, 4, 6])]
    replicates = int(generator_cfg.get("replicates_per_cell", 30))
    if base_level <= 0 or replicates < 1 or not noise_fractions or not strengths:
        raise ValueError("Invalid synthetic base, replicates or strata")
    if not np.isfinite([base_level, seasonal_fraction, trend_fraction, *noise_fractions, *strengths]).all():
        raise ValueError("Synthetic parameters must be finite")
    if any(x <= 0 for x in noise_fractions) or any(x <= 1 for x in strengths):
        raise ValueError("Noise fractions must be positive and strengths must exceed one")
    starts = [int(x) for x in generator_cfg.get("event_start_indices", [16, 17, 18, 19, 20])]
    if not starts or any(x < 16 or x > 20 for x in starts):
        raise ValueError("Main events must start after warmup and have a complete 0--3-month window")
    warmup_index = int(generator_cfg.get("warmup_event_index", 14))
    end_index = int(generator_cfg.get("end_event_index", 22))
    if not 12 <= warmup_index <= 15 or not 21 <= end_index <= 23:
        raise ValueError("Invalid explicit warmup/end edge event positions")
    main_scenarios = ["no_change", "outlier", "level_up", "level_down", "slope_up", "slope_down", "variance_up"]
    edge_scenarios = ["level_warmup", "level_end"]
    if list(generator_cfg.get("main_scenarios", main_scenarios)) != main_scenarios:
        raise ValueError("The frozen main scenario set/order must not change silently")
    if list(generator_cfg.get("secondary_edge_scenarios", edge_scenarios)) != edge_scenarios:
        raise ValueError("The frozen secondary scenario set/order must not change silently")
    cells = _cells(noise_fractions, strengths)
    number_series = len(cells) * replicates
    seeds = {
        "validation": int(generator_cfg.get("validation_seed", 100000)),
        "test": int(generator_cfg.get("test_seed", 200000)),
    }
    if min(seeds.values()) < 0 or abs(seeds["validation"] - seeds["test"]) < number_series:
        raise ValueError("Validation and test per-series seed intervals must be disjoint")

    periods = pd.period_range(start_period, periods=months, freq="M")
    dates = periods.to_timestamp()
    positions = np.arange(months, dtype=float)
    ordinary = base_level * (
        1.0 + trend_fraction * positions + seasonal_fraction * np.sin(2 * np.pi * positions / 12)
    )
    # The deterministic baseline above is shared by all strata and both splits.
    observation_frames = []
    expense_columns = {}
    events = []
    series_rows = []
    series_number = 0
    for scenario, strength, noise_fraction in cells:
        for replicate in range(replicates):
            seed = seeds[split] + series_number
            rng = np.random.default_rng(seed)
            series_id = f"{split}_{series_number:06d}"
            series_number += 1
            event_index = (
                warmup_index if scenario == "level_warmup" else
                end_index if scenario == "level_end" else int(rng.choice(starts))
            )
            noise_std = base_level * noise_fraction
            shocks = rng.normal(0.0, noise_std, months)
            values = ordinary.copy()
            amplitude = strength * noise_std
            direction = -1 if scenario in {"level_down", "slope_down"} else 1
            if scenario.startswith("level_"):
                values[event_index:] += direction * amplitude
            elif scenario.startswith("slope_"):
                values[event_index:] += direction * amplitude * np.arange(1, months - event_index + 1) / 3
            elif scenario == "variance_up":
                shocks[event_index:] *= strength
            elif scenario == "outlier":
                direction = 1 if replicate % 2 == 0 else -1
                values[event_index] += direction * amplitude
            values += shocks
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError("Generated invalid expenses; revise protocol before running the benchmark")
            is_main = scenario not in {"level_warmup", "level_end"}
            has_event = scenario not in {"no_change", "outlier"}
            event_start = str(periods[event_index]) if scenario != "no_change" else None
            event_window_class = "main" if is_main else (
                "warmup" if scenario == "level_warmup" else "end_partial"
            )
            if has_event:
                event_type = "level" if scenario.startswith("level_") else (
                    "slope" if scenario.startswith("slope_") else "variance"
                )
                events.append({
                    "series_id": series_id,
                    "event_id": f"{series_id}_event_0",
                    "event_type": event_type,
                    "event_start_period": event_start,
                    "strength": strength,
                    "duration_months": months - event_index,
                    "direction": direction,
                    "is_main_protocol": is_main,
                    "event_window_class": event_window_class,
                })
            series_rows.append({
                "series_id": series_id,
                "scenario": scenario,
                "strength": strength,
                "noise_fraction": noise_fraction,
                "split": split,
                "seed": seed,
                "replicate": replicate,
                "is_main_protocol": is_main,
                "has_structural_event": has_event,
                "event_start_index": event_index if scenario != "no_change" else np.nan,
                "event_start_period": event_start,
                "event_window_class": event_window_class,
                "observation_start_period": str(periods[0]),
                "observation_end_period": str(periods[-1]),
                "observation_months": months,
                "level_shift_fraction": direction * strength * noise_fraction if scenario.startswith("level_") else np.nan,
            })
            expense_columns[series_id] = values
            observation_frames.append(pd.DataFrame({
                "series_id": series_id, "observation_period": periods.astype(str), "value": values,
            }))
    # No event/scenario metadata enters this complete calendar expense panel.
    residuals = _batch_forecast_residuals(
        pd.DataFrame(expense_columns, index=dates), forecast_cfg, int(release_lag_months),
    )
    return (
        residuals,
        pd.DataFrame.from_records(events, columns=EVENT_COLUMNS),
        pd.DataFrame.from_records(series_rows),
        pd.concat(observation_frames, ignore_index=True),
    )
