"""CatBoostDirect: независимое обучение горизонта и явно отделённый резерв."""
from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from .data import get_prefix, period_end
from .direct_training import build_direct_training, direct_features
from .models import baseline_predict

STRATEGY = "CatBoostDirect+SeasonalNaive"
NATIVE_MODEL = "CatBoostDirect"


class TemporalIntegrityError(RuntimeError):
    """Нарушение доступности: эксперимент обязан остановиться до fit."""


@dataclass
class DirectResult:
    forecasts: pd.DataFrame
    training: dict
    importance: pd.DataFrame
    errors: list[dict]


class CatBoostDirect:
    def __init__(self, parameters: dict, seed: int, release_lag_months: int = 0,
                 training_mode: str = "legacy", max_staleness_months: int | None = None):
        self.parameters = dict(parameters)
        self.seed = seed
        self.lag = release_lag_months
        self.mode = training_mode
        self.max_staleness = max_staleness_months

    def predict(self, panel: pd.DataFrame, origin: pd.Period, horizon: int,
                forecast_ids: list[str], baseline_config: dict) -> DirectResult:
        """Каждый вызов создаёт отдельную модель; прогнозы других h не принимаются."""
        started = time.perf_counter()
        origin = pd.Period(origin, freq="M")
        prefix = get_prefix(panel, origin, self.lag)
        # Вся известная панель для обучения; forecast_ids ограничивают только predict.
        X, delta, trace = build_direct_training(
            prefix, origin, horizon, release_lag_months=self.lag,
            mode=self.mode, max_staleness_months=self.max_staleness,
        )
        if len(trace):
            r = trace.historical_origin.dt.to_period("M")
            valid = ((trace.feature_cutoff.dt.to_period("M") == r - self.lag) &
                     (trace.target_period.dt.to_period("M") == r + horizon) &
                     (trace.target_available_at.dt.to_period("M") == r + horizon + self.lag) &
                     (trace.target_available_at <= period_end(origin)))
            if not valid.all() or len(X) != len(delta) or len(X) != len(trace):
                raise TemporalIntegrityError("Нарушены даты или соответствие обучающих пар.")
        history = prefix[forecast_ids]
        target = (origin + horizon).to_timestamp()
        training = {
            "forecast_origin": str(period_end(origin).date()), "horizon": horizon,
            "training_mode": self.mode, "n_training_rows": len(trace),
            "n_training_municipalities": trace.municipality_id.nunique(),
            "n_historical_origins": trace.historical_origin.nunique(),
            "n_target_periods": trace.target_period.nunique(),
            "min_historical_origin": str(trace.historical_origin.min().date()) if len(trace) else None,
            "max_historical_origin": str(trace.historical_origin.max().date()) if len(trace) else None,
            "max_feature_cutoff": str(trace.feature_cutoff.max().date()) if len(trace) else None,
            "max_training_target": str(trace.target_period.max().date()) if len(trace) else None,
            "max_target_available_at": str(trace.target_available_at.max().date()) if len(trace) else None,
            "n_forecast_series": len(forecast_ids), "fit_called": False,
            "fit_succeeded": False,
        }
        forecasts = pd.DataFrame({"municipality_id": forecast_ids,
                                  "y_pred": np.nan, "status": "failed",
                                  "effective_model": "none", "reason": "",
                                  "failure_stage": "", "fallback_baseline_status": ""})
        importance = pd.DataFrame(columns=["feature", "importance"])
        errors = []
        phase = "fallback" if not len(trace) else "fit"
        try:
            if not len(trace):
                values, flags = baseline_predict(history, self.lag + horizon, "SeasonalNaive", baseline_config)
                forecasts["y_pred"] = values[self.lag + horizon - 1]
                forecasts["status"] = "fallback_no_training_pairs"
                forecasts["effective_model"] = "SeasonalNaive"
                forecasts["reason"] = "no_training_pairs"
                forecasts["fallback_baseline_status"] = np.where(
                    flags[self.lag + horizon - 1], "fallback_last_value", "native")
            else:
                model = CatBoostRegressor(**self.parameters, random_seed=self.seed,
                                          allow_writing_files=False, verbose=False)
                training["fit_called"] = True
                model.fit(X, delta)
                training["fit_succeeded"] = True
                forecasts["effective_model"] = NATIVE_MODEL
                phase = "predict"
                X_future, anchor = direct_features(history, target)
                # Совпадает с восстановлением и проверкой конечности в CatBoostRecursive.
                values = np.maximum(anchor + model.predict(X_future), 0.0)
                if not np.isfinite(values).all():
                    raise RuntimeError("CatBoost вернул неконечный прогноз.")
                forecasts["y_pred"] = values
                forecasts["status"] = "native"
                importance = pd.DataFrame({"feature": X.columns,
                                           "importance": model.feature_importances_})
        except Exception as exc:
            forecasts["y_pred"] = np.nan
            forecasts["status"] = "failed"
            forecasts["reason"] = repr(exc)
            forecasts["failure_stage"] = phase
            errors.append({"forecast_origin": training["forecast_origin"], "horizon": horizon,
                           "stage": phase, "error": repr(exc), "municipality_ids": forecast_ids})
        training["status"] = forecasts.status.iloc[0]
        training["seconds"] = time.perf_counter() - started
        return DirectResult(forecasts, training, importance, errors)
