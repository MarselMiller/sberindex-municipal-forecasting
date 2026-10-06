"""Фиксированные сезонно-трендовые варианты E05b, без внешних факторов.

Тренд оценивается по годовым разностям последних шести календарных позиций.
Полный сезонный шаблон переносится на cutoff линейным трендом. Затухание
применяется только к будущему участку, начиная со степени phi**1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .models import baseline_predict


TREND_VARIANTS = ("SeasonalTrendLinear", "SeasonalTrendDamped")
ANNUAL_DIFFERENCE_WINDOW = 6
MIN_ANNUAL_PAIRS = 3
SEASONAL_TEMPLATE_MONTHS = 12


def _positive_integer(value: int, name: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} должен быть целым числом.")
    minimum = 0 if allow_zero else 1
    if value < minimum:
        raise ValueError(f"{name} должен быть >= {minimum}.")
    return int(value)


def _calendar_prefix(panel: pd.DataFrame, cutoff: pd.Period) -> pd.DataFrame:
    """Восстановить календарную сетку только условно доступного префикса."""
    if not isinstance(panel, pd.DataFrame) or panel.empty:
        raise ValueError("Для сезонного тренда нужна непустая панель.")
    if panel.columns.has_duplicates or panel.columns.astype(str).has_duplicates:
        raise ValueError("Идентификаторы МО должны быть уникальны.")
    if not isinstance(panel.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        raise ValueError("Индекс панели должен содержать календарные месяцы.")
    periods = (panel.index.asfreq("M") if isinstance(panel.index, pd.PeriodIndex)
               else panel.index.to_period("M"))
    if periods.has_duplicates or periods.hasnans:
        raise ValueError("Месяцы панели должны быть известны и уникальны.")
    prefix = panel.loc[periods <= cutoff].copy()
    if prefix.empty:
        raise ValueError("На cutoff нет доступной истории.")
    prefix.index = periods[periods <= cutoff].to_timestamp()
    prefix = prefix.sort_index()
    months = pd.date_range(prefix.index[0], cutoff.to_timestamp(), freq="MS")
    return prefix.reindex(months).astype(float)


def predict_seasonal_trend(
    panel: pd.DataFrame,
    origin: pd.Period,
    horizon: int,
    *,
    release_lag_months: int,
    variant: str,
    phi: float = 0.9,
    yearly_growth_window: int = 3,
    yearly_growth_bounds: tuple[float, float] = (0.5, 2.0),
) -> tuple[np.ndarray, pd.DataFrame]:
    """Вернуть прогноз O+h и диагностику, в исходном порядке столбцов МО.

    C=O-L, D=h+L. b=median((y[t]-y[t-12])/12) для t=C-5,...,C,
    минимум три конечные пары. Нужны все двенадцать значений C-11,...,C.
    a[m]=y[t_m]+b*(C-t_m), level=mean(a), seasonal[m]=a[m]-level.
    Линейный прирост равен b*D, затухающий — b*sum(phi**j, j=1,...,D).

    При нехватке пар/шаблона вызывается существующий SeasonalNaiveYoY.
    Его внутренний резерв LastValue отражается дополнительными полями.
    Ошибки входа и вычислений выбрасываются: это не предусмотренный резерв.
    """
    if variant not in TREND_VARIANTS:
        raise ValueError(f"Неизвестный сезонный тренд: {variant}.")
    if not isinstance(origin, pd.Period) or origin.freqstr != "M":
        raise ValueError("origin должен быть месячным pd.Period.")
    horizon = _positive_integer(horizon, "horizon")
    release_lag_months = _positive_integer(
        release_lag_months, "release_lag_months", allow_zero=True)
    yearly_growth_window = _positive_integer(yearly_growth_window, "yearly_growth_window")
    if not np.isfinite(phi) or not 0 < phi <= 1:
        raise ValueError("phi должен быть конечным и лежать в (0, 1].")
    bounds = np.asarray(yearly_growth_bounds, dtype=float)
    if (bounds.shape != (2,) or not np.isfinite(bounds).all()
            or bounds[0] <= 0 or bounds[0] > bounds[1]):
        raise ValueError("yearly_growth_bounds должны быть положительными lo <= hi.")

    cutoff = origin - release_lag_months
    target = origin + horizon
    distance = horizon + release_lag_months
    history = _calendar_prefix(panel, cutoff)
    n_municipalities = history.shape[1]
    recent_periods = pd.period_range(cutoff - 5, cutoff, freq="M")
    recent = history.reindex(recent_periods.to_timestamp()).to_numpy(dtype=float)
    previous = history.reindex((recent_periods - 12).to_timestamp()).to_numpy(dtype=float)
    valid_pairs = np.isfinite(recent) & np.isfinite(previous)
    n_pairs = valid_pairs.sum(axis=0)
    differences = np.full_like(recent, np.nan)
    with np.errstate(over="raise", invalid="raise"):
        np.subtract(recent, previous, out=differences, where=valid_pairs)
        differences /= 12.0
    b = np.full(n_municipalities, np.nan)
    any_pairs = n_pairs > 0
    if any_pairs.any():
        b[any_pairs] = np.nanmedian(differences[:, any_pairs], axis=0)

    template_periods = pd.period_range(cutoff - 11, cutoff, freq="M")
    template_values = history.reindex(template_periods.to_timestamp()).to_numpy(dtype=float)
    template_complete = np.isfinite(template_values).all(axis=0)
    native = (n_pairs >= MIN_ANNUAL_PAIRS) & template_complete
    predictions = np.full(n_municipalities, np.nan)
    level = np.full(n_municipalities, np.nan)
    seasonal = np.full((SEASONAL_TEMPLATE_MONTHS, n_municipalities), np.nan)

    if native.any():
        month_distance = np.arange(SEASONAL_TEMPLATE_MONTHS - 1, -1, -1, dtype=float)
        with np.errstate(over="raise", invalid="raise"):
            adjusted = template_values[:, native] + month_distance[:, None] * b[native]
            level[native] = adjusted.mean(axis=0)
            chronological_seasonal = adjusted - level[native]
            for position, period in enumerate(template_periods):
                seasonal[period.month - 1, native] = chronological_seasonal[position]
            future_distance = (float(distance) if variant == "SeasonalTrendLinear"
                               else float(np.power(phi, np.arange(1, distance + 1)).sum()))
            predictions[native] = np.maximum(
                level[native] + seasonal[target.month - 1, native] + b[native] * future_distance,
                0.0,
            )

    baseline_internal_fallback = np.zeros(n_municipalities, dtype=bool)
    fallback = ~native
    if fallback.any():
        fallback_predictions, fallback_flags = baseline_predict(
            history.loc[:, fallback], distance, "SeasonalNaiveYoY",
            {"yearly_growth_window": yearly_growth_window,
             "yearly_growth_bounds": bounds.tolist()},
        )
        predictions[fallback] = fallback_predictions[-1]
        baseline_internal_fallback[fallback] = fallback_flags[-1]
    if not np.isfinite(predictions).all():
        raise RuntimeError("Сезонный тренд или его резерв вернул неконечный прогноз.")

    reasons = np.full(n_municipalities, "", dtype=object)
    reasons[n_pairs < MIN_ANNUAL_PAIRS] = "insufficient_annual_pairs"
    reasons[(n_pairs >= MIN_ANNUAL_PAIRS) & ~template_complete] = "incomplete_seasonal_template"
    diagnostics = pd.DataFrame({
        "municipality_id": history.columns.astype(str),
        "forecast_origin": origin.to_timestamp(how="end").normalize(),
        "feature_cutoff": cutoff.to_timestamp(how="end").normalize(),
        "target_period": target.to_timestamp(),
        "horizon": horizon,
        "release_lag_months": release_lag_months,
        "distance_from_cutoff": distance,
        "status": np.where(native, "native", "fallback"),
        "reason": reasons,
        "effective_model": np.where(native, variant, "SeasonalNaiveYoY"),
        "b": b,
        "n_annual_pairs": n_pairs,
        "seasonal_template_complete": template_complete,
        "level": level,
        "baseline_internal_fallback": baseline_internal_fallback,
        "baseline_effective_model": np.where(
            native, "", np.where(baseline_internal_fallback, "LastValue", "SeasonalNaiveYoY")),
    })
    for month in range(1, 13):
        diagnostics[f"seasonal_{month:02d}"] = seasonal[month - 1]
    return predictions, diagnostics
