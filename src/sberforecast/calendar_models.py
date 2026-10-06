"""E05b: только представление месяца цели в прежнем CatBoostDirect.

Сборщик пар, delta = y[target] - last_available, числовые NaN, глобальная
обучающая панель и резерв SeasonalNaive не меняются. Отдельный адаптер
нужен потому, что E02b не предоставляет фабрику регрессора или hook для X.
"""
from __future__ import annotations

import hashlib
import json
import time

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from .data import get_prefix, period_end
from .direct_model import CatBoostDirect, DirectResult, TemporalIntegrityError
from .direct_training import PAIR_KEY, build_direct_training, direct_features
from .models import baseline_predict

ENCODINGS = ("K0", "K1", "K2")


def training_signature(trace: pd.DataFrame, X: pd.DataFrame, delta: np.ndarray) -> dict:
    """Проверяемые подписи исходных пар, меток и X до кодирования месяца."""
    keys = PAIR_KEY + ["target_available_at"]
    ordered_keys = trace.loc[:, keys].sort_values(keys, kind="stable").reset_index(drop=True)
    key_bytes = pd.util.hash_pandas_object(ordered_keys, index=False).to_numpy().tobytes()
    schema = json.dumps({"columns": X.columns.tolist(), "dtypes": [str(t) for t in X.dtypes]},
                        ensure_ascii=False, sort_keys=True).encode("utf-8")
    X_bytes = pd.util.hash_pandas_object(X, index=True).to_numpy().tobytes()
    return {
        "training_key_sha256": hashlib.sha256(key_bytes).hexdigest(),
        "training_delta_sha256": hashlib.sha256(np.asarray(delta, dtype=np.float64).tobytes()).hexdigest(),
        "training_raw_X_sha256": hashlib.sha256(schema + X_bytes).hexdigest(),
        "source_feature_names": X.columns.tolist(),
        "source_feature_dtypes": {name: str(dtype) for name, dtype in X.dtypes.items()},
    }


def transform_month_features(X: pd.DataFrame, encoding: str) -> pd.DataFrame:
    """Копия X; K0 identity, K1 category, K2 category + прежние sin/cos.

    Входной month уже относится к дате цели: он сформирован неизменённым
    direct_features(). Все остальные значения, типы и порядок сохраняются.
    Категория -- строка 01..12. NaN в числовых признаках остаются NaN,
    как в E02b; смешанный DataFrame никогда не приводится целиком к float.
    """
    if encoding not in ENCODINGS:
        raise ValueError(f"encoding должен быть одним из {ENCODINGS}.")
    result = X.copy(deep=True)
    if encoding == "K0":
        return result
    required = {"month", "month_sin", "month_cos"}
    if not required.issubset(X.columns) or "month_category" in X.columns:
        raise ValueError("Ожидались исходные month, month_sin, month_cos без month_category.")
    months = pd.to_numeric(X["month"], errors="raise").to_numpy(dtype=float)
    if not (np.isfinite(months).all() and
            ((months >= 1) & (months <= 12) & (months == np.floor(months))).all()):
        raise ValueError("month должен содержать конечные целые месяцы 1..12.")
    result["month_category"] = pd.Series([f"{int(month):02d}" for month in months],
                                         index=result.index, dtype=object)
    removed = {"month", "month_sin", "month_cos"} if encoding == "K1" else {"month"}
    columns = []
    for column in X.columns:
        if column == "month":
            columns.append("month_category")
        elif column not in removed:
            columns.append(column)
    return result.loc[:, columns]


class CalendarCatBoostDirect(CatBoostDirect):
    """Отдельная модель каждого h с календарной заменой только на копиях X.

    Подготовка/допуск обучающих пар и резерв полностью переиспользуются.
    Старый predict не имеет точки подстановки regressor/feature-transform;
    здесь повторён его небольшой адаптер fit/predict без копии backtest.
    """

    def __init__(self, parameters: dict, seed: int, encoding: str,
                 release_lag_months: int = 0, training_mode: str = "legacy",
                 max_staleness_months: int | None = None):
        if encoding not in ENCODINGS:
            raise ValueError(f"encoding должен быть одним из {ENCODINGS}.")
        if "cat_features" in parameters or "one_hot_max_size" in parameters:
            raise ValueError("Исходные параметры E02b не должны задавать cat_features/one_hot_max_size.")
        super().__init__(parameters, seed, release_lag_months, training_mode, max_staleness_months)
        self.encoding = encoding

    def predict(self, panel: pd.DataFrame, origin: pd.Period, horizon: int,
                forecast_ids: list[str], baseline_config: dict) -> DirectResult:
        """Одна независимая fit; ошибки исполнения не заменяются резервом."""
        started = time.perf_counter()
        origin = pd.Period(origin, freq="M")
        prefix = get_prefix(panel, origin, self.lag)
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
        signature = training_signature(trace, X, delta)
        transformed = transform_month_features(X, self.encoding)
        categorical = [] if self.encoding == "K0" else ["month_category"]
        parameters = dict(self.parameters)
        if categorical:
            parameters.update(cat_features=categorical, one_hot_max_size=12)
        history = prefix[forecast_ids]
        target = (origin + horizon).to_timestamp()
        training = {
            "forecast_origin": str(period_end(origin).date()), "horizon": horizon,
            "training_mode": self.mode, "month_encoding": self.encoding,
            "n_training_rows": len(trace),
            "n_training_municipalities": trace.municipality_id.nunique(),
            "n_historical_origins": trace.historical_origin.nunique(),
            "n_target_periods": trace.target_period.nunique(),
            "min_historical_origin": str(trace.historical_origin.min().date()) if len(trace) else None,
            "max_historical_origin": str(trace.historical_origin.max().date()) if len(trace) else None,
            "max_feature_cutoff": str(trace.feature_cutoff.max().date()) if len(trace) else None,
            "max_training_target": str(trace.target_period.max().date()) if len(trace) else None,
            "max_target_available_at": str(trace.target_available_at.max().date()) if len(trace) else None,
            "n_forecast_series": len(forecast_ids), "fit_called": False, "fit_succeeded": False,
            **signature,
            "feature_names": transformed.columns.tolist(),
            "feature_dtypes": {name: str(dtype) for name, dtype in transformed.dtypes.items()},
            "cat_features": categorical, "one_hot_max_size": 12 if categorical else None,
            "fit_actual_parameters": None,
        }
        forecasts = pd.DataFrame({
            "municipality_id": forecast_ids, "y_pred": np.nan, "status": "failed",
            "effective_model": "none", "reason": "", "failure_stage": "",
            "fallback_baseline_status": "",
        })
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
                model = CatBoostRegressor(**parameters, random_seed=self.seed,
                                          allow_writing_files=False, verbose=False)
                training["fit_called"] = True
                model.fit(transformed, delta)
                training["fit_succeeded"] = True
                training["fit_actual_parameters"] = model.get_all_params()
                training["fit_actual_cat_feature_indices"] = model.get_cat_feature_indices()
                training["fit_actual_feature_names"] = model.feature_names_
                forecasts["effective_model"] = f"CatBoostDirect{self.encoding}"
                phase = "predict"
                X_future, anchor = direct_features(history, target)
                transformed_future = transform_month_features(X_future, self.encoding)
                values = np.maximum(anchor + model.predict(transformed_future), 0.0)
                if not np.isfinite(values).all():
                    raise RuntimeError("CatBoost вернул неконечный прогноз.")
                forecasts["y_pred"] = values
                forecasts["status"] = "native"
                importance = pd.DataFrame({"feature": transformed.columns,
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
