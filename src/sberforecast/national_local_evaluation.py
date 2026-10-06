"""E05d: paired learner/decomposition contrasts and separate national errors.

Expense metrics reuse the fixed-key E05b evaluator. National errors contain
one row per release/horizon, never one repeated row per municipality.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .trend_calendar_evaluation import evaluate_group

MODELS = ["C0", "L0", "CN", "LN", "SeasonalNaiveYoY", "ProphetAuto", "ProphetYearly"]
NATIVE_MODELS = ["C0", "L0", "CN", "LN"]
SCOPES = ("strategy", "native")
CONTRASTS = (("C0", "L0", "algorithm"), ("CN", "LN", "algorithm"),
             ("C0", "CN", "decomposition"), ("L0", "LN", "decomposition"))
NATIONAL_KEY = ["forecast_origin", "target_period", "horizon"]
METRICS = ("mae_macro", "mae_micro", "r2_pooled")


def _deltas(tables: dict[str, pd.DataFrame], *, by_origin: bool = False) -> pd.DataFrame:
    records = []
    fields = ["split", "horizon"] + (["forecast_origin"] if by_origin else [])
    for scope in SCOPES:
        metrics = tables[f"metrics_{scope}" + ("_by_origin" if by_origin else "")]
        for values, frame in metrics.groupby(fields, sort=True, dropna=False):
            group = dict(zip(fields, values))
            indexed = frame.set_index("model")
            for earlier, later, hypothesis in CONTRASTS:
                left, right = indexed.loc[earlier], indexed.loc[later]
                counts = ("n_predictions", "n_municipalities", "n_origins")
                if any(left[field] != right[field] for field in counts):
                    raise ValueError("Разности требуют одинаковых прогнозных ключей и числа случаев.")
                complete = left.metric_status == right.metric_status == "complete"
                state = "complete" if complete else (
                    "incomplete_failed" if "incomplete_failed" in (left.metric_status, right.metric_status)
                    else right.metric_status)
                record = dict(group, scope=scope, comparison=f"{later}-{earlier}",
                              from_model=earlier, to_model=later, hypothesis=hypothesis,
                              metric_status=state, **{field: int(right[field]) for field in counts})
                for metric in METRICS:
                    record["delta_" + metric] = float(right[metric] - left[metric]) if complete else np.nan
                records.append(record)
    return pd.DataFrame(records)


def _origin_robustness(deltas: pd.DataFrame, pooled: pd.DataFrame) -> pd.DataFrame:
    """Describe date concentration without a test of significance or selection."""
    fields = ["scope", "split", "horizon", "comparison", "hypothesis"]
    records = []
    for values, group in deltas.groupby(fields, sort=True, dropna=False):
        row = dict(zip(fields, values))
        valid = group.loc[group.metric_status.eq("complete") & np.isfinite(group.delta_mae_micro)]
        change = valid.delta_mae_micro
        gains = valid.loc[change.lt(0)]
        losses = valid.loc[change.gt(0)]
        pooled_row = pooled.loc[
            pooled.scope.eq(row["scope"]) & pooled.split.eq(row["split"])
            & pooled.horizon.eq(row["horizon"]) & pooled.comparison.eq(row["comparison"])
        ].iloc[0]
        total_gain = float(-gains.delta_mae_micro.sum())
        largest = gains.loc[gains.delta_mae_micro.idxmin()] if len(gains) else None
        row.update(n_origins_requested=len(group), n_origins_evaluable=len(valid),
                   n_origins_improved=len(gains), n_origins_worsened=len(losses),
                   n_origins_tied=int(change.eq(0).sum()),
                   n_origins_incomplete=int(group.metric_status.eq("incomplete_failed").sum()),
                   n_origins_without_native=int(group.metric_status.eq("no_native_forecasts").sum()),
                   median_delta_mae_micro=float(change.median()) if len(valid) else np.nan,
                   min_delta_mae_micro=float(change.min()) if len(valid) else np.nan,
                   max_delta_mae_micro=float(change.max()) if len(valid) else np.nan,
                   sum_origin_mae_reductions=total_gain,
                   sum_origin_mae_increases=float(losses.delta_mae_micro.sum()),
                   largest_improvement_origin=str(largest.forecast_origin) if largest is not None else "",
                   largest_improvement_share=float(-largest.delta_mae_micro / total_gain) if total_gain else np.nan,
                   pooled_delta_mae_micro=float(pooled_row.delta_mae_micro),
                   improvement_only_one_origin=bool(pooled_row.metric_status == "complete"
                       and pooled_row.delta_mae_micro < 0 and len(gains) == 1),
                   interpretation="descriptive_no_significance_test_no_holdout_selection")
        records.append(row)
    return pd.DataFrame(records)


def _national_tables(national: pd.DataFrame, expected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = NATIONAL_KEY + ["split", "n_hat", "n_actual", "national_forecast_status"]
    if set(required) - set(national):
        raise ValueError("Неполная схема отдельного национального прогноза.")
    frame = national.copy()
    if frame.duplicated(NATIONAL_KEY).any():
        raise ValueError("Национальные ошибки должны содержать один случай O/h, без повторения по МО.")
    for field in ("n_hat", "n_actual"):
        frame[field] = pd.to_numeric(frame[field], errors="raise").astype(float)
    for canonical, alias in (("n_hat", "N_hat"), ("n_actual", "N_actual")):
        if alias in frame:
            alternate = pd.to_numeric(frame[alias], errors="raise").astype(float)
            if not (frame[canonical].eq(alternate) | (frame[canonical].isna() & alternate.isna())).all():
                raise ValueError("Псевдонимы национального прогноза/факта должны совпадать.")
    origin = pd.to_datetime(frame.forecast_origin, errors="raise").dt.to_period("M")
    target = pd.to_datetime(frame.target_period, errors="raise").dt.to_period("M")
    if (frame.horizon.lt(1).any() or not frame.horizon.eq(frame.horizon.astype(int)).all()
            or any(t != o + int(h) for o, t, h in zip(origin, target, frame.horizon))):
        raise ValueError("Национальный горизонт обязан сохранять календарную цель O+h.")
    reference = expected[NATIONAL_KEY + ["split"]].drop_duplicates()
    # Normalize dates only for comparison, retaining the exact saved metadata
    # in the national error file. There is no target substitution.
    normalized = frame[NATIONAL_KEY + ["split"]].copy()
    for current in (reference, normalized):
        current["forecast_origin"] = pd.to_datetime(current.forecast_origin).dt.to_period("M").astype(str)
        current["target_period"] = pd.to_datetime(current.target_period).dt.to_period("M").astype(str)
    paired = reference.merge(normalized, on=NATIONAL_KEY, how="left", suffixes=("", "_national"),
                             validate="one_to_one", indicator=True)
    if not paired._merge.eq("both").all() or not paired.split.eq(paired.split_national).all():
        raise ValueError("Национальная диагностика не покрывает точные даты/горизонты/разделы E01.")
    finite_prediction = np.isfinite(frame.n_hat)
    failed = frame.national_forecast_status.eq("failed")
    if not (frame.national_forecast_status.eq("native")
            | frame.national_forecast_status.astype(str).str.startswith("fallback") | failed).all():
        raise ValueError("Неизвестный статус отдельного национального прогноза.")
    if not ((finite_prediction & ~failed) | (frame.n_hat.isna() & failed)).all():
        raise ValueError("Ошибка национального прогноза должна оставаться failed/NaN без скрытой замены.")
    if np.isinf(frame.n_actual).any():
        raise ValueError("Национальный факт должен быть конечным либо явно отсутствовать.")
    frame["actual_available_for_diagnostics"] = np.isfinite(frame.n_actual)
    frame["prediction_available"] = finite_prediction
    frame["absolute_error"] = np.where(finite_prediction & np.isfinite(frame.n_actual),
                                      np.abs(frame.n_hat - frame.n_actual), np.nan)
    frame["signed_error"] = frame.n_hat - frame.n_actual
    frame["actual_usage"] = "evaluation_only_never_forecast_feature"
    rows = []
    for (split, horizon), group in frame.groupby(["split", "horizon"], sort=True):
        actual = group.actual_available_for_diagnostics
        complete = actual & group.prediction_available
        missing_forecast = int((actual & ~group.prediction_available).sum())
        state = "incomplete_failed" if missing_forecast else "complete" if complete.any() else "no_actual_national_target"
        rows.append({"split": split, "horizon": int(horizon), "n_requested_origins": len(group),
                     "n_actual_available": int(actual.sum()), "n_evaluable_origins": int(complete.sum()),
                     "n_forecast_available": int(group.prediction_available.sum()),
                     "n_failed": int(group.national_forecast_status.eq("failed").sum()),
                     "mae_national": float(group.loc[complete, "absolute_error"].mean()) if state == "complete" else np.nan,
                     "metric_status": state, "weighting": "one_equal_weight_per_origin_horizon"})
    columns = ["split", "horizon", "n_requested_origins", "n_actual_available", "n_evaluable_origins",
               "n_forecast_available", "n_failed", "mae_national", "metric_status", "weighting"]
    return pd.DataFrame(rows, columns=columns), frame


def evaluate_national_local(predictions: pd.DataFrame, expected: pd.DataFrame,
                            national_forecasts: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return fixed A/B expense scopes, four contrasts and national-only MAE."""
    tables = evaluate_group(predictions, expected, MODELS, NATIVE_MODELS)
    tables["metric_deltas"] = _deltas(tables)
    tables["metric_deltas_by_origin"] = _deltas(tables, by_origin=True)
    tables["origin_robustness"] = _origin_robustness(tables["metric_deltas_by_origin"], tables["metric_deltas"])
    tables["national_metrics"], tables["national_errors"] = _national_tables(national_forecasts, expected)
    return tables
