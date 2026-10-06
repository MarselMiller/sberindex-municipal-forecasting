"""E05d: один числовой K0 pipeline, два learner и явное восстановление масштаба.

L0 учится на прежней delta = y_target - anchor. CN/LN используют те же
19 формул, применённые к R = y/N, и delta_R = R_target - anchor_R.
Будущий фактический N сохраняется только как диагностика национального
прогноза: восстановление использует исключительно N_hat.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import time

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from .calendar_models import training_signature
from .data import get_prefix, period_end
from .direct_model import DirectResult, TemporalIntegrityError
from .direct_training import PAIR_KEY, build_direct_training, direct_features
from .models import baseline_predict
from .national_local import build_local_training, national_forecast, ratio_panel

VARIANTS = ("L0", "CN", "LN")
NATIVE_MODELS = {"L0": "LightGBMDirect", "CN": "CatBoostDirectNationalLocal",
                 "LN": "LightGBMDirectNationalLocal"}


@dataclass
class NationalLocalResult(DirectResult):
    training_provenance: pd.DataFrame
    local_ratio_diagnostics: list[dict]
    national_forecast: dict


def ordered_training_key_signature(trace: pd.DataFrame) -> str:
    rows = pd.util.hash_pandas_object(
        trace[PAIR_KEY + ["target_available_at"]], index=False).to_numpy().tobytes()
    return hashlib.sha256(rows).hexdigest()


def frame_signature(frame: pd.DataFrame) -> str:
    schema = json.dumps({"columns": frame.columns.tolist(),
                         "dtypes": [str(dtype) for dtype in frame.dtypes]},
                        sort_keys=True).encode("utf-8")
    rows = pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()
    return hashlib.sha256(schema + rows).hexdigest()


def vector_signature(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.float64).tobytes()).hexdigest()


class NativeLightGBMRegressor:
    """Тонкий native API adapter: не требует дополнительного scikit-learn.

    n_estimators соответствует num_boost_round, n_jobs -- num_threads,
    random_state -- seed. Матрица и метки передаются без преобразований.
    """
    def __init__(self, **parameters):
        self.requested_parameters = dict(parameters)
        self.rounds = int(parameters.get("n_estimators", 300))
        self.parameters = dict(parameters)
        self.parameters.pop("n_estimators", None)
        self.parameters["num_threads"] = self.parameters.pop("n_jobs", 2)
        self.parameters["seed"] = self.parameters.pop("random_state", 42)
        self.parameters.setdefault("device_type", "cpu")

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        import lightgbm as lgb
        dataset = lgb.Dataset(X, label=y, feature_name=X.columns.tolist())
        self.booster_ = lgb.train(self.parameters, dataset, num_boost_round=self.rounds)
        self.feature_name_ = self.booster_.feature_name()
        self.feature_importances_ = self.booster_.feature_importance(importance_type="split")
        return self

    def predict(self, X: pd.DataFrame):
        return self.booster_.predict(X)

    def get_params(self, deep=True):
        return dict(self.booster_.params)


def _lightgbm_regressor(**parameters):
    """Lazy import: прежние модели и синтетические fake-тесты не требуют LGB."""
    return NativeLightGBMRegressor(**parameters)


def _lightgbm_metadata(model) -> dict:
    """Полные native resolved параметры, без сохранения деревьев/весов.

    Блок parameters из model_to_string содержит фактические defaults
    native learner; обёртка сохраняет также запрошенные параметры.
    Метаданные dump_model дополняют проверку objective/feature names.
    """
    result = {"fit_actual_parameters": model.get_params(deep=True),
              "fit_actual_feature_names": list(model.feature_name_),
              "fit_actual_cat_feature_indices": []}
    if hasattr(model, "requested_parameters"):
        result["fit_requested_parameters"] = dict(model.requested_parameters)
    booster = getattr(model, "booster_", None)
    if booster is not None:
        dump = booster.dump_model()
        result["fit_booster_parameters"] = dict(booster.params)
        result["fit_booster_dump_metadata"] = {
            key: dump[key] for key in ("name", "version", "num_class", "num_tree_per_iteration",
                                      "label_index", "max_feature_idx", "objective", "average_output",
                                      "feature_names", "feature_infos", "pandas_categorical") if key in dump}
        result["fit_booster_dump_metadata"]["n_trees"] = booster.num_trees()
        resolved = {}
        parameter_block = booster.model_to_string().split("parameters:", 1)
        if len(parameter_block) == 2:
            for line in parameter_block[1].splitlines():
                if line.strip() == "end of parameters":
                    break
                if line.startswith("[") and line.endswith("]") and ": " in line:
                    name, value = line[1:-1].split(": ", 1)
                    resolved[name] = value
        result["fit_native_resolved_parameters"] = resolved
    return result


def _validate_trace(trace: pd.DataFrame, X: pd.DataFrame, delta: np.ndarray,
                    origin: pd.Period, horizon: int, lag: int, decomposition: bool) -> None:
    if len(X) != len(delta) or len(X) != len(trace):
        raise TemporalIntegrityError("Число строк X, меток и обучающих ключей различается.")
    if not len(trace):
        return
    r = trace.historical_origin.dt.to_period("M")
    valid = ((trace.feature_cutoff.dt.to_period("M") == r - lag) &
             (trace.target_period.dt.to_period("M") == r + horizon) &
             (trace.target_available_at.dt.to_period("M") == r + horizon + lag) &
             (trace.target_available_at <= period_end(origin)))
    if not valid.all() or not np.isfinite(delta).all():
        raise TemporalIntegrityError("Недоступная метка или нарушенный календарь прямой пары.")
    if decomposition:
        denominators = trace.target_national_denominator.to_numpy(dtype=float)
        available_at = pd.to_datetime(trace.target_national_available_at)
        if not (np.isfinite(denominators) & (denominators > 0)).all() or (
                available_at.isna().any() or (available_at > period_end(origin)).any()):
            raise TemporalIntegrityError("Национальный знаменатель метки недоступен на O.")


class NationalLocalDirect:
    """Новые варианты L0/CN/LN; реальный C0 только копируется из E02b.

    Резерв допускается только при нулевом числе прямых пар. Ошибки fit,
    predict или национального прогноза сохраняются как failed, без замены.
    """
    def __init__(self, parameters: dict, seed: int, variant: str,
                 national: pd.DataFrame | None = None, release_lag_months: int = 0,
                 training_mode: str = "legacy", max_staleness_months: int | None = None):
        if variant not in VARIANTS:
            raise ValueError("E05d обучает только L0, CN и LN; C0 берётся из E02b.")
        if training_mode != "legacy":
            raise ValueError("E05d сохраняет фиксированный допуск пар legacy.")
        if "cat_features" in parameters or "one_hot_max_size" in parameters:
            raise ValueError("E05d использует числовые K0 без категорий/one-hot.")
        if variant != "L0" and national is None:
            raise ValueError("Для CN/LN требуется национальный ряд всей панели.")
        if variant != "CN" and parameters.get("device_type", parameters.get("device", "cpu")) != "cpu":
            raise ValueError("E05d разрешает только CPU LightGBM.")
        if variant != "CN" and parameters.get("random_state", seed) != seed:
            raise ValueError("random_state LightGBM должен совпадать с seed.")
        if variant != "CN" and (parameters.get("use_missing", True) is not True or
                                parameters.get("zero_as_missing", False) is not False):
            raise ValueError("LightGBM сохраняет NaN как missing и нули как числа.")
        self.parameters = dict(parameters)
        self.seed = seed
        self.variant = variant
        self.national = None if national is None else national.copy(deep=True)
        self.lag = release_lag_months
        self.mode = training_mode
        self.max_staleness = max_staleness_months

    def predict(self, panel: pd.DataFrame, origin: pd.Period, horizon: int,
                forecast_ids: list[str], baseline_config: dict) -> NationalLocalResult:
        started = time.perf_counter()
        origin = pd.Period(origin, freq="M")
        prefix = get_prefix(panel, origin, self.lag)
        nominal_X, nominal_delta, nominal_trace = build_direct_training(
            prefix, origin, horizon, release_lag_months=self.lag,
            mode=self.mode, max_staleness_months=self.max_staleness)
        decomposition = self.variant != "L0"
        diagnostics = []
        national_record = {}
        if decomposition:
            X, delta, trace = build_local_training(
                prefix, self.national, origin, horizon, release_lag_months=self.lag,
                mode=self.mode, max_staleness_months=self.max_staleness)
            ratios, ratio_diagnostics = ratio_panel(prefix, self.national)
            ratio_diagnostics = ratio_diagnostics.copy()
            ratio_diagnostics["period"] = ratio_diagnostics.index
            ratio_diagnostics["forecast_origin"] = str(period_end(origin).date())
            ratio_diagnostics["horizon"] = int(horizon)
            ratio_diagnostics["variant"] = self.variant
            ratio_diagnostics["scope"] = "available_history"
            diagnostics = ratio_diagnostics.to_dict("records")
            model_history = ratios[forecast_ids]
            national_record = national_forecast(
                self.national, origin, horizon, baseline_config,
                release_lag_months=self.lag)
        else:
            X, delta, trace = nominal_X, nominal_delta, nominal_trace
            model_history = prefix[forecast_ids]
        horizon = int(horizon)
        _validate_trace(trace, X, delta, origin, horizon, self.lag, decomposition)
        target = (origin + horizon).to_timestamp()
        X_future, anchor = direct_features(model_history, target)
        nominal_future, nominal_anchor = direct_features(prefix[forecast_ids], target)
        training = {
            "forecast_origin": str(period_end(origin).date()), "horizon": horizon,
            "variant": self.variant, "training_mode": self.mode, "month_encoding": "K0",
            "learner": "CatBoost" if self.variant == "CN" else "LightGBM",
            "target_transformation": "delta_ratio" if decomposition else "delta_nominal",
            "n_training_rows": len(trace),
            "n_training_municipalities": int(trace.municipality_id.nunique()),
            "n_historical_origins": int(trace.historical_origin.nunique()),
            "n_target_periods": int(trace.target_period.nunique()),
            "min_historical_origin": str(trace.historical_origin.min().date()) if len(trace) else None,
            "max_historical_origin": str(trace.historical_origin.max().date()) if len(trace) else None,
            "max_feature_cutoff": str(trace.feature_cutoff.max().date()) if len(trace) else None,
            "max_training_target": str(trace.target_period.max().date()) if len(trace) else None,
            "max_target_available_at": str(trace.target_available_at.max().date()) if len(trace) else None,
            "n_forecast_series": len(forecast_ids), "fit_called": False, "fit_succeeded": False,
            **training_signature(trace, X, delta),
            "training_ordered_key_sha256": ordered_training_key_signature(trace),
            "forecast_raw_X_sha256": frame_signature(X_future),
            "forecast_anchor_sha256": vector_signature(anchor),
            "nominal_source_signature": training_signature(nominal_trace, nominal_X, nominal_delta),
            "nominal_source_ordered_key_sha256": ordered_training_key_signature(nominal_trace),
            "nominal_forecast_raw_X_sha256": frame_signature(nominal_future),
            "nominal_forecast_anchor_sha256": vector_signature(nominal_anchor),
            "n_nominal_source_pairs": len(nominal_trace),
            "n_pairs_excluded_by_ratio_validity": len(nominal_trace) - len(trace),
            "feature_names": X.columns.tolist(),
            "feature_dtypes": {name: str(dtype) for name, dtype in X.dtypes.items()},
            "cat_features": [], "fit_actual_parameters": None,
            "fit_actual_cat_feature_indices": [],
        }
        provenance = trace.copy(deep=True)
        provenance["training_row"] = np.arange(len(trace))
        provenance["variant"] = self.variant
        provenance["delta"] = delta
        forecasts = pd.DataFrame({
            "municipality_id": forecast_ids, "y_pred": np.nan, "status": "failed",
            "effective_model": "none", "reason": "", "failure_stage": "",
            "fallback_baseline_status": "", "variant": self.variant,
            "anchor": anchor, "ratio_hat": np.nan,
            "national_hat": national_record.get("N_hat", np.nan),
        })
        importance = pd.DataFrame(columns=["feature", "importance"])
        errors = []
        phase = "fallback" if not len(trace) else "national_forecast" if decomposition else "fit"
        try:
            if not len(trace):
                # Даже CN/LN используют прежний резерв в исходных рублях.
                values, flags = baseline_predict(prefix[forecast_ids], self.lag + horizon,
                                                "SeasonalNaive", baseline_config)
                forecasts["y_pred"] = values[self.lag + horizon - 1]
                forecasts["status"] = "fallback_no_training_pairs"
                forecasts["effective_model"] = "SeasonalNaive"
                forecasts["reason"] = "no_training_pairs"
                forecasts["fallback_baseline_status"] = np.where(
                    flags[self.lag + horizon - 1], "fallback_last_value", "native")
            else:
                N_hat = national_record.get("N_hat", np.nan)
                if decomposition and (not np.isfinite(N_hat) or N_hat <= 0):
                    raise RuntimeError("Национальный прогноз не является конечным положительным N_hat.")
                if not np.isfinite(anchor).all():
                    raise RuntimeError("Не хватает конечного опорного значения для прогноза.")
                phase = "fit"
                if self.variant == "CN":
                    model = CatBoostRegressor(**self.parameters, random_seed=self.seed,
                                              allow_writing_files=False, verbose=False)
                else:
                    parameters = dict(self.parameters)
                    parameters.setdefault("random_state", self.seed)
                    parameters.setdefault("use_missing", True)
                    parameters.setdefault("zero_as_missing", False)
                    model = _lightgbm_regressor(**parameters)
                training["fit_called"] = True
                model.fit(X, delta)
                training["fit_succeeded"] = True
                if self.variant == "CN":
                    training["fit_actual_parameters"] = model.get_all_params()
                    training["fit_actual_feature_names"] = list(model.feature_names_)
                    training["fit_actual_cat_feature_indices"] = model.get_cat_feature_indices()
                else:
                    training.update(_lightgbm_metadata(model))
                forecasts["effective_model"] = NATIVE_MODELS[self.variant]
                phase = "predict"
                restored = anchor + model.predict(X_future)
                if not np.isfinite(restored).all():
                    raise RuntimeError("Learner вернул неконечное восстановленное значение.")
                if decomposition:
                    forecasts["ratio_hat"] = restored
                    values = np.maximum(N_hat * restored, 0.0)
                else:
                    values = np.maximum(restored, 0.0)
                if not np.isfinite(values).all():
                    raise RuntimeError("Неконечный прогноз в исходных рублях.")
                forecasts["y_pred"] = values
                forecasts["status"] = "native"
                importance = pd.DataFrame({"feature": X.columns,
                                           "importance": model.feature_importances_})
        except Exception as exc:
            forecasts["y_pred"] = np.nan
            forecasts["ratio_hat"] = np.nan
            forecasts["status"] = "failed"
            forecasts["reason"] = repr(exc)
            forecasts["failure_stage"] = phase
            errors.append({"forecast_origin": training["forecast_origin"], "horizon": horizon,
                           "variant": self.variant, "stage": phase, "error": repr(exc),
                           "municipality_ids": forecast_ids})
        training["status"] = forecasts.status.iloc[0]
        training["seconds"] = time.perf_counter() - started
        return NationalLocalResult(forecasts, training, importance, errors,
                                   provenance, diagnostics, national_record)
