"""Прямые обучающие пары без обучения моделей и без изменения признаков E01.

Для исторического выпуска r, горизонта h, лага L и текущего выпуска O:
    cutoff = r - L; target = r + h; target + L <= O.
lag_k = y[cutoff - (k - 1) месяцев], как в next_features(). Неизвестные
лаги остаются NaN; ffill используется только для anchor = last_available.
Обучающая метка delta = y[target] - anchor, как в supervised_training().
Календарные признаки описывают target, а не следующий месяц после cutoff.

legacy (режим A): непустой календарный префикс, конечные anchor и цель;
нет обязательных 12 фактов или ограничения давности. При h=1, L=0 это
те же пары, признаки и метки, что у supervised_training() на месячной сетке.
strict12 (режим B): дополнительно >=12 конечных фактов в префиксе примера.
max_staleness_months — отдельное ограничение обучающих примеров, по умолчанию
отсутствует в обоих режимах. Давность измеряется от cutoff, как в eligibility().
Допуск МО для текущего прогнозирования сюда не передаётся: глобальное
обучение, как в E01, использует все доступные МО.
"""
from __future__ import annotations

from numbers import Integral
from typing import Literal

import numpy as np
import pandas as pd

from .data import period_end
from .features import next_features

TrainingMode = Literal["legacy", "strict12"]
PAIR_KEY = ["municipality_id", "historical_origin", "target_period", "horizon"]
METADATA_COLUMNS = [
    "municipality_id", "historical_origin", "feature_cutoff", "target_period",
    "target_available_at", "model_origin", "horizon", "release_lag_months",
    "mode", "n_history_observations", "staleness_months",
    "last_observed_period", "anchor", "target_value",
]


def _nonnegative_integer(value: int, name: str, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} должен быть целым числом >= {minimum}.")


def _validate_panel(panel: pd.DataFrame) -> None:
    if not isinstance(panel.index, pd.DatetimeIndex) or panel.index.tz is not None:
        raise ValueError("Ожидался DatetimeIndex без часового пояса.")
    if panel.index.hasnans or not panel.index.is_unique or not panel.index.is_monotonic_increasing:
        raise ValueError("Даты должны быть уникальны, упорядочены и непусты.")
    if not panel.index.equals(panel.index.to_period("M").to_timestamp()):
        raise ValueError("Даты должны обозначать начало календарного месяца.")
    if not panel.columns.is_unique or not panel.columns.astype(str).is_unique:
        raise ValueError("Идентификаторы МО должны быть уникальны.")


def _calendar_prefix(panel: pd.DataFrame, cutoff: pd.Period) -> pd.DataFrame:
    """Сначала отсекаем будущее, затем достраиваем пропуски только внутри истории."""
    prefix = panel.loc[panel.index <= cutoff.to_timestamp()]
    if prefix.empty:
        return prefix.astype(float)
    return prefix.reindex(pd.date_range(prefix.index[0], cutoff.to_timestamp(), freq="MS")).astype(float)


def direct_features(history: pd.DataFrame, target: pd.Timestamp) -> tuple[pd.DataFrame, np.ndarray]:
    """Те же исторические признаки E01, календарь для произвольной будущей цели.

    Вызывающий код обязан передать только доступный исторический префикс.
    Пропущенные календарные строки достраиваются NaN, значения не backfill.
    """
    _validate_panel(history)
    target = pd.Timestamp(target)
    if history.empty:
        raise ValueError("Пустая история.")
    if target != target.to_period("M").to_timestamp() or target <= history.index[-1]:
        raise ValueError("Цель должна быть будущим календарным месяцем.")
    history = _calendar_prefix(history, history.index[-1].to_period("M"))
    X, anchor = next_features(history, history.index[-1] + pd.offsets.MonthBegin(1))
    month = target.month
    X["month"] = float(month)
    X["month_sin"] = np.sin(2 * np.pi * month / 12)
    X["month_cos"] = np.cos(2 * np.pi * month / 12)
    X["days_in_month"] = float(target.days_in_month)
    X["time_index"] = float(target.year * 12 + month - (2023 * 12 + 1))
    return X, anchor


def build_direct_training(
    panel: pd.DataFrame,
    model_origin: str | pd.Period,
    horizon: int,
    *,
    release_lag_months: int,
    mode: TrainingMode,
    max_staleness_months: int | None = None,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Вернуть X, delta и построчные метаданные с общим RangeIndex.

    Пустое множество пар — нормальный результат: пустые X/delta/metadata,
    без автоматического ослабления требований или резервного прогноза.
    При L>0 legacy сохраняет условия допуска старого сборщика, но пары
    намеренно отличаются: метки теперь соответствуют горизонту от выпуска,
    а не от cutoff. Месячное значение условно доступно в конце target + L.
    """
    _validate_panel(panel)
    _nonnegative_integer(horizon, "horizon", 1)
    _nonnegative_integer(release_lag_months, "release_lag_months")
    if mode not in ("legacy", "strict12"):
        raise ValueError("mode должен быть явно задан: legacy или strict12.")
    if max_staleness_months is not None:
        _nonnegative_integer(max_staleness_months, "max_staleness_months")
    origin = pd.Period(model_origin, freq="M")
    lag = int(release_lag_months)
    available = _calendar_prefix(panel, origin - lag)
    # Схема пустого результата не зависит от будущих значений данных.
    template = pd.DataFrame(np.nan, index=pd.to_datetime(["2023-01-01"]), columns=panel.columns)
    empty_X = next_features(template, pd.Timestamp("2023-02-01"))[0].iloc[:0].reset_index(drop=True)
    if available.empty:
        return empty_X, np.empty(0, dtype=float), pd.DataFrame(columns=METADATA_COLUMNS)

    frames, labels, metadata = [], [], []
    first_r = available.index[0].to_period("M") + lag
    last_r = origin - int(horizon) - lag
    for r in pd.period_range(first_r, last_r, freq="M"):
        cutoff = r - lag
        target = r + int(horizon)
        history = available.loc[available.index <= cutoff.to_timestamp()]
        X, anchor = direct_features(history, target.to_timestamp())
        truth = available.loc[target.to_timestamp()].to_numpy(dtype=float)
        valid = np.isfinite(truth) & np.isfinite(anchor)
        finite = np.isfinite(history.to_numpy(dtype=float))
        n_history = finite.sum(axis=0)
        last_position = np.where(finite, np.arange(len(history))[:, None], -1).max(axis=0)
        staleness = len(history) - 1 - last_position
        if mode == "strict12":
            valid &= n_history >= 12
        if max_staleness_months is not None:
            valid &= staleness <= max_staleness_months
        if not valid.any():
            continue
        delta = truth[valid] - anchor[valid]
        if not np.isfinite(delta).all():
            raise ValueError("Неконечное изменение при конечных цели и anchor.")
        frames.append(X.loc[valid])
        labels.append(delta)
        metadata.append(pd.DataFrame({
            "municipality_id": history.columns[valid].astype(str),
            "historical_origin": period_end(r),
            "feature_cutoff": period_end(cutoff),
            "target_period": target.to_timestamp(),
            "target_available_at": period_end(target + lag),
            "model_origin": period_end(origin),
            "horizon": int(horizon), "release_lag_months": lag, "mode": mode,
            "n_history_observations": n_history[valid],
            "staleness_months": staleness[valid],
            "last_observed_period": history.index[last_position[valid]],
            "anchor": anchor[valid], "target_value": truth[valid],
        }))
    if not frames:
        return empty_X, np.empty(0, dtype=float), pd.DataFrame(columns=METADATA_COLUMNS)
    return (pd.concat(frames, ignore_index=True), np.concatenate(labels),
            pd.concat(metadata, ignore_index=True)[METADATA_COLUMNS])
