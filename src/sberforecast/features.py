"""Признаки для следующего месяца исключительно из переданного префикса."""
from __future__ import annotations
import numpy as np
import pandas as pd

LAGS = (1, 2, 3, 6, 12)


def next_features(history: pd.DataFrame, target: pd.Timestamp) -> tuple[pd.DataFrame, np.ndarray]:
    if history.empty:
        raise ValueError("Пустая история.")
    expected = history.index[-1] + pd.offsets.MonthBegin(1)
    if target != expected:
        raise ValueError(f"Ожидался следующий календарный месяц {expected}, получен {target}.")
    n = history.shape[1]
    out: dict[str, np.ndarray | float] = {}
    for lag in LAGS:
        out[f"lag_{lag}"] = history.iloc[-lag].to_numpy() if len(history) >= lag else np.full(n, np.nan)
    last = history.ffill().iloc[-1].to_numpy(dtype=float)
    out["last_available"] = last
    for window in (3, 6):
        tail = history.tail(window)
        out[f"mean_{window}"] = tail.mean().to_numpy()
        out[f"std_{window}"] = tail.std(ddof=0).to_numpy()
        out[f"count_{window}"] = tail.notna().sum().to_numpy()
    out["delta_1"] = out["lag_1"] - out["lag_2"]
    out["relative_delta_1"] = out["delta_1"] / np.maximum(np.abs(out["lag_2"]), 1.0)
    month = target.month
    out["month"] = np.full(n, month)
    out["month_sin"] = np.full(n, np.sin(2 * np.pi * month / 12))
    out["month_cos"] = np.full(n, np.cos(2 * np.pi * month / 12))
    out["days_in_month"] = np.full(n, target.days_in_month)
    # Известный заранее календарный индекс, не значение целевого ряда.
    out["time_index"] = np.full(n, target.year * 12 + month - (2023 * 12 + 1))
    X = pd.DataFrame(out, index=history.columns).astype(float)
    return X, last


def supervised_training(prefix: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Цель — изменение к последнему известному значению. Все метки <= cutoff."""
    frames, labels = [], []
    target_dates = []
    for idx in range(1, len(prefix)):
        date = prefix.index[idx]
        X, anchor = next_features(prefix.iloc[:idx], date)
        y = prefix.iloc[idx].to_numpy(dtype=float)
        valid = np.isfinite(y) & np.isfinite(anchor)
        if valid.any():
            frames.append(X.loc[valid])
            labels.append((y - anchor)[valid])
            target_dates.append(date)
    if not frames:
        raise ValueError("Недостаточно наблюдений для обучения общей модели.")
    meta = {"n_training_rows": int(sum(len(x) for x in frames)),
            "max_training_target": str(max(target_dates).date()),
            "min_training_target": str(min(target_dates).date())}
    return pd.concat(frames, ignore_index=True), np.concatenate(labels), meta
