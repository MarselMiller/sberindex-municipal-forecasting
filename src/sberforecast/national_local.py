"""Calendar-causal National/Local decomposition without changing E02 code.

N[t] is the median of finite observations in the complete available panel for
month t. R[i,t] = y[i,t] / N[t] exists only for a finite, strictly positive N.
The unchanged 19-feature pipeline operates on R; the training target is
delta_R = R[target] - last_available_R. Prediction is restored exclusively by
the forecast N_hat, never by actual future N: max(0, N_hat * R_hat).

Monthly missingness uses municipalities first observed by that month, including
the current month. Future municipalities do not enter past denominators.
Availability is the explicit project assumption: end of month t + L.
"""
from __future__ import annotations

from numbers import Integral

import numpy as np
import pandas as pd

from .data import get_prefix, period_end
from .direct_training import (
    TrainingMode, _validate_panel, build_direct_training,
)
from .models import baseline_predict


def _lag(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError("release_lag_months должен быть целым числом >= 0.")
    return int(value)


def build_national(panel: pd.DataFrame, release_lag_months: int = 0) -> pd.DataFrame:
    """Return month-level statistics using only observations in the same month.

    The input may include future months for audit purposes. None of their values
    or column membership affects earlier monthly statistics. Missing calendar
    months remain explicit rows with NaN N, rather than shifting positions.
    """
    _validate_panel(panel)
    lag = _lag(release_lag_months)
    if panel.empty:
        raise ValueError("Для national series нужна непустая календарная панель.")
    history = panel.reindex(pd.date_range(panel.index[0], panel.index[-1], freq="MS"))
    values = history.to_numpy(dtype=float)
    finite = np.isfinite(values)
    known = np.maximum.accumulate(finite, axis=0)
    observed_counts = finite.sum(axis=1)
    known_counts = known.sum(axis=1)
    records = []
    for position, timestamp in enumerate(history.index):
        observed = values[position, finite[position]]
        quantiles = (np.quantile(observed, [0, .1, .25, .5, .75, .9, 1])
                     if len(observed) else np.full(7, np.nan))
        n_used = int(observed_counts[position])
        n_known = int(known_counts[position])
        n_missing = n_known - n_used
        record = dict(zip(("min", "p10", "q25", "median", "q75", "p90", "max"), quantiles))
        record.update({
            "period": str(timestamp.to_period("M")),
            "N": float(quantiles[3]), "national_y": float(quantiles[3]),
            "n_municipalities_used": n_used,
            "n_municipalities_known": n_known,
            "n_municipalities_available": n_known,
            "n_missing": n_missing,
            "missing_share": n_missing / n_known if n_known else np.nan,
            "available_at": period_end(timestamp.to_period("M") + lag),
            "release_lag_months": lag,
        })
        records.append(record)
    result = pd.DataFrame(records, index=history.index.copy())
    result.index.name = "ds"
    return result


def _validate_national(national: pd.DataFrame) -> None:
    if not {"N", "n_municipalities_used", "available_at"}.issubset(national.columns):
        raise ValueError("National table requires N, n_municipalities_used, available_at.")
    _validate_panel(national)
    dates = pd.to_datetime(national["available_at"], errors="raise")
    if dates.isna().any() or dates.dt.tz is not None:
        raise ValueError("Некорректные даты доступности national denominator.")


def ratio_panel(panel: pd.DataFrame, national: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ratios on the original calendar and transparent monthly diagnostics.

    Invalid or absent denominators produce NaN; nonfinite expense observations
    also produce NaN. No future fill, constant replacement, or row removal is
    performed. The caller must restrict the returned history to its own cutoff.
    """
    _validate_panel(panel)
    _validate_national(national)
    aligned = national.reindex(panel.index)
    denominator = aligned["N"].to_numpy(dtype=float)
    values = panel.to_numpy(dtype=float)
    usable_denominator = np.isfinite(denominator) & (denominator > 0)
    finite_expense = np.isfinite(values)
    ratios = np.full(values.shape, np.nan, dtype=float)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        np.divide(values, denominator[:, None], out=ratios,
                  where=finite_expense & usable_denominator[:, None])
    # Extreme finite numerator/denominator ratios must also remain missing.
    overflow = ~np.isfinite(ratios) & finite_expense & usable_denominator[:, None]
    ratios[~np.isfinite(ratios)] = np.nan
    result = pd.DataFrame(ratios, index=panel.index.copy(), columns=panel.columns.copy())
    reasons = np.select(
        [~np.isfinite(denominator), denominator <= 0],
        ["national_denominator_missing_or_nonfinite", "national_denominator_nonpositive"],
        default="valid",
    )
    diagnostics = pd.DataFrame({
        "period": panel.index.to_period("M").astype(str),
        "N": denominator,
        "national_available_at": aligned.available_at.to_numpy(),
        "n_municipalities_used": aligned.n_municipalities_used.to_numpy(),
        "denominator_status": reasons,
        "n_expense_finite": finite_expense.sum(axis=1),
        "n_expense_nonfinite": (~finite_expense).sum(axis=1),
        "n_ratio_finite": np.isfinite(ratios).sum(axis=1),
        "n_ratio_missing": (~np.isfinite(ratios)).sum(axis=1),
        "n_ratio_missing_invalid_denominator": (finite_expense & ~usable_denominator[:, None]).sum(axis=1),
        "n_ratio_overflow": overflow.sum(axis=1),
    }, index=panel.index.copy())
    diagnostics.index.name = "ds"
    return result.astype(float), diagnostics


def national_forecast(national: pd.DataFrame, origin: str | pd.Period, horizon: int,
                      baseline_config: dict, release_lag_months: int = 0) -> dict:
    """Adapt unchanged SeasonalNaiveYoY to N, preserving its fixed growth rules.

    The last prediction at step h+L corresponds to O+h. N_actual and its error
    are evaluation-only fields obtained after prediction; no prediction code
    receives them. Failures return NaN and an explicit failure status.
    """
    _validate_national(national)
    lag = _lag(release_lag_months)
    if isinstance(horizon, bool) or not isinstance(horizon, Integral) or horizon < 1:
        raise ValueError("horizon должен быть целым числом >= 1.")
    origin = pd.Period(origin, freq="M")
    target = (origin + int(horizon)).to_timestamp()
    record = {
        "forecast_origin": str(period_end(origin).date()), "horizon": int(horizon),
        "target_period": target, "feature_cutoff": period_end(origin - lag),
        "target_available_at": period_end(origin + int(horizon) + lag),
        "release_lag_months": lag,
        "N_hat": np.nan, "status": "failed", "effective_model": "none", "reason": "",
        "n_available_history_months": 0, "n_available_national_observations": 0,
        "n_recursive_fallback_steps": 0,
        "national_actual_used_for_prediction": False,
    }
    try:
        prefix = get_prefix(national[["N"]], origin, lag)
        selected = national.reindex(prefix.index)
        known_N = np.isfinite(prefix.N.to_numpy(dtype=float))
        expected_availability = pd.DatetimeIndex([
            period_end(month + lag) for month in prefix.index.to_period("M")
        ])
        if not np.array_equal(pd.to_datetime(selected.loc[known_N, "available_at"]).to_numpy(),
                              expected_availability.to_numpy()[known_N]):
            raise ValueError("National history availability does not match release_lag_months.")
        if (pd.to_datetime(selected.loc[known_N, "available_at"]) > period_end(origin)).any():
            raise ValueError("National history includes a denominator unavailable at O.")
        record["n_available_history_months"] = len(prefix)
        record["n_available_national_observations"] = int(known_N.sum())
        values, flags = baseline_predict(prefix, int(horizon) + lag, "SeasonalNaiveYoY", baseline_config)
        predicted = float(values[-1, 0])
        if not np.isfinite(predicted) or predicted < 0:
            raise ValueError("Национальный прогноз должен быть конечным и неотрицательным.")
        fallback = bool(flags[-1, 0])
        record.update({
            "N_hat": predicted, "status": "fallback_last_value" if fallback else "native",
            "effective_model": "LastValue" if fallback else "SeasonalNaiveYoY",
            "n_recursive_fallback_steps": int(flags[:, 0].sum()),
        })
    except Exception as exc:
        record["reason"] = repr(exc)
    # Actual is deliberately accessed only after the forecast has been produced.
    actual = float(national.loc[target, "N"]) if target in national.index else np.nan
    actual = actual if np.isfinite(actual) else np.nan
    record["N_actual"] = actual
    record["n_hat"] = record["N_hat"]
    record["n_actual"] = actual
    record["absolute_error"] = abs(actual - record["N_hat"]) if np.isfinite(actual) and np.isfinite(record["N_hat"]) else np.nan
    record["national_forecast_status"] = record["status"]
    return record


def build_local_training(
    expense_panel: pd.DataFrame, national: pd.DataFrame,
    model_origin: str | pd.Period, horizon: int, *, release_lag_months: int,
    mode: TrainingMode, max_staleness_months: int | None = None,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Reuse the unchanged direct builder on R and add denominator provenance.

    Actual N[target] is part of a historical label only after target+L <= O.
    For every retained pair, the anchor denominator is known by its own r.
    """
    _validate_panel(expense_panel)
    _validate_national(national)
    lag = _lag(release_lag_months)
    origin = pd.Period(model_origin, freq="M")
    # Restrict before division as well as before the old builder's feature loop.
    expense_prefix = expense_panel.loc[expense_panel.index <= (origin - lag).to_timestamp()].copy()
    selected = national.reindex(expense_prefix.index)
    expected_availability = pd.DatetimeIndex([
        period_end(month + lag) for month in expense_prefix.index.to_period("M")
    ])
    relevant = np.isfinite(selected.N.to_numpy(dtype=float))
    if not np.array_equal(pd.to_datetime(selected.loc[relevant, "available_at"]).to_numpy(),
                          expected_availability.to_numpy()[relevant]):
        raise ValueError("Даты доступности N не соответствуют publication lag обучения.")
    ratios, _ = ratio_panel(expense_prefix, national)
    X, delta, trace = build_direct_training(
        ratios, origin, horizon, release_lag_months=lag, mode=mode,
        max_staleness_months=max_staleness_months,
    )
    columns = [
        "target_national_denominator", "target_national_available_at",
        "target_national_n_municipalities_used", "n_municipalities_used",
        "anchor_national_denominator", "anchor_national_available_at",
        "anchor_national_n_municipalities_used", "ratio_target_value", "ratio_anchor",
        "expense_target_value", "expense_anchor_value",
    ]
    if trace.empty:
        for column in columns:
            trace[column] = pd.Series(dtype="datetime64[ns]" if column.endswith("available_at") else "float64")
        return X, delta, trace
    targets = national.reindex(pd.DatetimeIndex(trace.target_period))
    anchors = national.reindex(pd.DatetimeIndex(trace.last_observed_period))
    trace["target_national_denominator"] = targets.N.to_numpy(dtype=float)
    trace["target_national_available_at"] = pd.to_datetime(targets.available_at).to_numpy()
    trace["target_national_n_municipalities_used"] = targets.n_municipalities_used.to_numpy()
    trace["n_municipalities_used"] = targets.n_municipalities_used.to_numpy()
    trace["anchor_national_denominator"] = anchors.N.to_numpy(dtype=float)
    trace["anchor_national_available_at"] = pd.to_datetime(anchors.available_at).to_numpy()
    trace["anchor_national_n_municipalities_used"] = anchors.n_municipalities_used.to_numpy()
    trace["ratio_target_value"] = trace.target_value
    trace["ratio_anchor"] = trace.anchor
    trace["expense_target_value"] = trace.target_value * trace.target_national_denominator
    trace["expense_anchor_value"] = trace.anchor * trace.anchor_national_denominator
    r = trace.historical_origin.dt.to_period("M")
    target = trace.target_period.dt.to_period("M")
    anchor_period = trace.last_observed_period.dt.to_period("M")
    valid = (
        (trace.feature_cutoff.dt.to_period("M") == r - lag)
        & (target == r + int(horizon))
        & (trace.target_available_at.dt.to_period("M") == target + lag)
        & (trace.target_national_available_at.dt.to_period("M") == target + lag)
        & (trace.target_national_available_at <= period_end(origin))
        & (trace.anchor_national_available_at.dt.to_period("M") == anchor_period + lag)
        & (trace.anchor_national_available_at <= trace.historical_origin)
        & np.isfinite(trace.target_national_denominator)
        & (trace.target_national_denominator > 0)
        & np.isfinite(trace.anchor_national_denominator)
        & (trace.anchor_national_denominator > 0)
    )
    if not valid.all() or len(X) != len(delta) or len(X) != len(trace):
        raise ValueError("Нарушена доступность или конечность national/local training pairs.")
    return X, delta, trace


def restore_local_prediction(national_prediction: float, ratio_prediction: np.ndarray) -> np.ndarray:
    """Restore the original scale using a forecast, with explicit finite checks."""
    forecast = float(national_prediction)
    ratios = np.asarray(ratio_prediction, dtype=float)
    if not np.isfinite(forecast) or forecast < 0 or not np.isfinite(ratios).all():
        raise ValueError("Restoration requires a finite nonnegative N_hat and finite R_hat.")
    with np.errstate(over="ignore", invalid="ignore"):
        restored = np.maximum(0.0, forecast * ratios)
    if not np.isfinite(restored).all():
        raise ValueError("National/local restoration overflow.")
    return restored
