"""Контрольные прогнозы, общая CatBoost и необязательный настоящий Prophet."""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from .features import next_features, supervised_training

BASELINES = ("LastValue", "SeasonalNaive", "SeasonalNaiveYoY")
SUPPORTED = (*BASELINES, "CatBoostRecursive", "ProphetAuto", "ProphetYearly")


def check_dependencies(names: list[str]) -> None:
    unknown = set(names) - set(SUPPORTED)
    if unknown:
        raise ValueError(f"Неизвестные модели: {unknown}")
    if any(name.startswith("Prophet") for name in names):
        try:
            import prophet  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("Prophet не установлен. Выполните: python -m pip install -r requirements-prophet.txt. "
                               "Он не будет заменён другой моделью под тем же именем.") from exc


def baseline_predict(history: pd.DataFrame, steps: int, model: str, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    if model not in BASELINES:
        raise ValueError(model)
    if steps < 1:
        raise ValueError("steps >= 1")
    history = history.copy()
    last = history.ffill().iloc[-1].to_numpy(dtype=float)
    if not np.isfinite(last).all():
        raise ValueError("Для прогноза нужен хотя бы один факт по каждому МО.")
    growth = np.ones(history.shape[1])
    if model == "SeasonalNaiveYoY" and len(history) > 12:
        ratios = history / history.shift(12)
        g = ratios.tail(cfg.get("yearly_growth_window", 3)).median().fillna(1.0)
        lo, hi = cfg.get("yearly_growth_bounds", [0.5, 2.0])
        growth = g.clip(lo, hi).to_numpy(dtype=float)
    preds, fallback = [], []
    for _ in range(steps):
        target = history.index[-1] + pd.offsets.MonthBegin(1)
        seasonal_date = target - pd.DateOffset(months=12)
        if model == "LastValue":
            y = last.copy(); fb = np.zeros(len(last), dtype=bool)
        else:
            season = (history.loc[seasonal_date].to_numpy(dtype=float)
                      if seasonal_date in history.index else np.full(len(last), np.nan))
            fb = ~np.isfinite(season)
            y = np.where(fb, last, season * growth)
        y = np.maximum(y, 0.0)
        history.loc[target] = y
        last = y
        preds.append(y); fallback.append(fb)
    return np.array(preds), np.array(fallback)


def catboost_predict(training_prefix: pd.DataFrame, forecast_history: pd.DataFrame, steps: int,
                     cfg: dict, seed: int) -> tuple[np.ndarray, dict, pd.DataFrame]:
    from catboost import CatBoostRegressor
    X, delta, meta = supervised_training(training_prefix)
    model = CatBoostRegressor(**cfg, random_seed=seed, allow_writing_files=False, verbose=False)
    model.fit(X, delta)
    history = forecast_history.copy()
    preds = []
    for _ in range(steps):
        target = history.index[-1] + pd.offsets.MonthBegin(1)
        X_future, anchor = next_features(history, target)
        y = np.maximum(anchor + model.predict(X_future), 0.0)
        if not np.isfinite(y).all():
            raise RuntimeError("CatBoost вернул неконечный прогноз.")
        history.loc[target] = y
        preds.append(y)
    importance = pd.DataFrame({"feature": X.columns, "importance": model.feature_importances_})
    return np.array(preds), meta, importance.sort_values("importance", ascending=False)


def prophet_predict(history: pd.DataFrame, steps: int, name: str, cfg: dict,
                    seed: int) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Адаптер; исполнение проверяется только при наличии установленного Prophet."""
    from prophet import Prophet
    for logger in ("cmdstanpy", "prophet"):
        logging.getLogger(logger).setLevel(logging.ERROR)
    dates = pd.date_range(history.index[-1] + pd.offsets.MonthBegin(1), periods=steps, freq="MS")
    result = np.full((steps, history.shape[1]), np.nan)
    failed = np.zeros_like(result, dtype=bool)
    errors = []
    for j, uid in enumerate(history.columns):
        train = history[uid].dropna().rename_axis("ds").reset_index(name="y")
        try:
            if len(train) < cfg["min_observations"]:
                raise ValueError("Короткая история для выбранной конфигурации Prophet")
            model = Prophet(
                daily_seasonality=False, weekly_seasonality=False,
                yearly_seasonality=("auto" if name == "ProphetAuto" else cfg["yearly_fourier_order"]),
                changepoint_prior_scale=cfg["changepoint_prior_scale"],
                seasonality_prior_scale=cfg["seasonality_prior_scale"],
                uncertainty_samples=0,
            )
            model.fit(train, seed=seed)
            result[:, j] = np.maximum(model.predict(pd.DataFrame({"ds": dates}))["yhat"].to_numpy(), 0.0)
        except Exception as exc:
            failed[:, j] = True
            errors.append({"municipality_id": str(uid), "model": name, "error": repr(exc)})
    # Не подменяем неудачу Prophet прогнозом другого алгоритма.
    return result, failed, errors
