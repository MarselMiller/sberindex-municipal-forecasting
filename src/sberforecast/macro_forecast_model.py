"""E05c: неизменный K0 с исторически доступными годовыми макропрогнозами.

Исходные признаки и delta берутся из E02a/E02b. Макропубликация выбирается
на собственную дату примера r, а не на более позднюю дату обучения O.
NaN макропризнаков сохраняются; они никогда не сокращают обучающую панель.
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
from .direct_model import CatBoostDirect, DirectResult, TemporalIntegrityError
from .direct_training import PAIR_KEY, build_direct_training, direct_features
from .macro_forecast_features import prepare_forecast_features
from .models import baseline_predict

INFLATION = "forecast_inflation_dec_dec_pct"
CONSUMPTION = "forecast_consumption_growth_annual_pct"
VARIANT_INDICATORS = {"M0": (), "M1": (INFLATION,), "M2": (INFLATION, CONSUMPTION)}


@dataclass
class MacroDirectResult(DirectResult):
    training_provenance: pd.DataFrame
    forecast_provenance: pd.DataFrame
    macro_diagnostics: list[dict]


def ordered_training_key_signature(trace: pd.DataFrame) -> str:
    """Подпись именно последовательности пар, дополняющая sorted hash E05b."""
    values = pd.util.hash_pandas_object(
        trace[PAIR_KEY + ["target_available_at"]], index=False).to_numpy().tobytes()
    return hashlib.sha256(values).hexdigest()


def _frame_signature(frame: pd.DataFrame) -> str:
    schema = json.dumps({"columns": frame.columns.tolist(),
                         "dtypes": [str(dtype) for dtype in frame.dtypes]},
                        sort_keys=True).encode("utf-8")
    rows = pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()
    return hashlib.sha256(schema + rows).hexdigest()


def append_macro_features(raw_X: pd.DataFrame, samples: pd.DataFrame,
                          macrotable: pd.DataFrame, indicators: tuple[str, ...],
                          source_ids: dict[str, str] | None = None,
                          ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Вернуть raw19 + macro, сохранив исходные значения, типы и индекс."""
    if not raw_X.index.equals(samples.index) or len(raw_X) != len(samples):
        raise TemporalIntegrityError("Индексы исходных X и макрозапросов различаются.")
    if not indicators:
        return raw_X.copy(deep=True), pd.DataFrame(index=raw_X.index), pd.DataFrame()
    macro_X, lineage = prepare_forecast_features(
        samples, macrotable, indicators=indicators, source_ids=source_ids)
    if not macro_X.index.equals(raw_X.index) or len(macro_X) != len(raw_X):
        raise TemporalIntegrityError("Макросоединение изменило строки или их порядок.")
    if set(raw_X.columns) & set(macro_X.columns):
        raise TemporalIntegrityError("Макропризнаки перекрывают исходные признаки K0.")
    augmented = pd.concat([raw_X, macro_X], axis=1)
    pd.testing.assert_frame_equal(augmented.loc[:, raw_X.columns], raw_X)
    if len(lineage):
        dated = lineage.publication_date.notna() & lineage.available_at.notna()
        limits = samples.set_index("sample_id").as_of_date
        as_of = lineage.sample_id.map(limits)
        if as_of.isna().any() or (lineage.loc[dated, "publication_date"] > as_of[dated]).any() or (
                lineage.loc[dated, "available_at"] > as_of[dated]).any():
            raise TemporalIntegrityError("Макропубликация недоступна на дату своего примера.")
    return augmented, macro_X, lineage


def macro_diagnostics(samples: pd.DataFrame, macro_X: pd.DataFrame,
                      lineage: pd.DataFrame, indicators: tuple[str, ...], *,
                      origin: pd.Period, horizon: int, variant: str,
                      scope: str) -> list[dict]:
    """Число строк не заменяет число дат, публикаций и макросостояний."""
    records = []
    for indicator in indicators:
        columns = [column for column in macro_X if column == indicator or column.startswith(indicator + "_")]
        values = macro_X[indicator].to_numpy(dtype=float)
        available = np.isfinite(values)
        provenance = lineage.loc[lineage.indicator.eq(indicator)]
        used = provenance.loc[provenance.value.notna()]
        dates = samples.as_of_date.reset_index(drop=True)
        states = macro_X.loc[:, columns].reset_index(drop=True)
        states["as_of_date"] = dates
        grouped = states.groupby("as_of_date", sort=True)
        variations = grouped[columns].nunique(dropna=False).gt(1).any(axis=1)
        date_values = grouped[indicator].first()
        changes = date_values.ne(date_values.shift()) & ~(
            date_values.isna() & date_values.shift().isna())
        if len(changes):
            changes.iloc[0] = False
        column_states = {}
        for column in columns:
            finite = macro_X[column].dropna()
            unique = int(finite.nunique())
            column_states[column] = {
                "n_finite": len(finite), "n_unique_finite": unique,
                "all_missing": not len(finite), "constant_finite": bool(len(finite) and unique == 1),
                "constant_including_missing": bool(len(macro_X) and macro_X[column].nunique(dropna=False) == 1),
            }
        records.append({
            "forecast_origin": str(period_end(origin).date()), "horizon": int(horizon),
            "variant": variant, "scope": scope, "indicator": indicator,
            "n_rows": len(samples), "n_historical_origins": int(samples.as_of_date.nunique()),
            "n_available": int(available.sum()),
            "available_fraction": float(available.mean()) if len(samples) else None,
            "n_dates_with_available_macro": int(samples.loc[available, "as_of_date"].nunique()),
            "n_unique_publications": int(used[["source_id", "publication_date"]].drop_duplicates().shape[0]),
            "n_unique_vintages": int(used.vintage_id.nunique()),
            "n_unique_finite_values": int(pd.Series(values[available]).nunique()),
            "n_dates_with_within_date_feature_variation": int(variations.sum()),
            "n_dates_with_value_changes": int(changes.sum()),
            "n_unique_date_macro_states": int(states.drop(columns="as_of_date").drop_duplicates().shape[0]),
            "all_missing": not available.any(),
            "constant_finite_value": bool(available.any() and pd.Series(values[available]).nunique() == 1),
            "column_states": column_states,
        })
    return records


def _queries(trace: pd.DataFrame) -> pd.DataFrame:
    samples = trace[["municipality_id", "historical_origin", "target_period"]].copy()
    samples = samples.rename(columns={"historical_origin": "as_of_date"})
    samples["sample_id"] = [str(position) for position in range(len(samples))]
    return samples


def _attach_training_keys(lineage: pd.DataFrame, trace: pd.DataFrame,
                          origin: pd.Period, horizon: int, variant: str) -> pd.DataFrame:
    if not len(lineage):
        result = lineage.copy()
        for column in PAIR_KEY + ["feature_cutoff", "target_available_at", "training_row"]:
            if column not in result:
                result[column] = pd.Series(dtype=object)
    else:
        keys = trace[PAIR_KEY + ["feature_cutoff", "target_available_at"]].copy()
        keys["sample_id"] = [str(position) for position in range(len(keys))]
        keys["training_row"] = np.arange(len(keys))
        # municipality/target are already part of the macro provenance if supplied.
        overlap = [column for column in keys if column in lineage and column != "sample_id"]
        result = lineage.drop(columns=overlap).merge(keys, on="sample_id", how="left",
                                                   validate="many_to_one", sort=False)
        if result.training_row.isna().any():
            raise TemporalIntegrityError("Происхождение макропризнака потеряло обучающий ключ.")
    result["model_origin"] = str(period_end(origin).date())
    result["forecast_origin"] = str(period_end(origin).date())
    result["model_horizon"] = int(horizon)
    result["variant"] = variant
    result["scope"] = "training"
    return result


class MacroCatBoostDirect(CatBoostDirect):
    """Независимый горизонт K0; M0 применяется только в синтетических тестах.

    В эксперименте M0 копируется из E02b, без вызова этой ветви обучения.
    Ошибки fit/predict сохраняются как failed и не вызывают резерв.
    """
    def __init__(self, parameters: dict, seed: int, variant: str,
                 macrotable: pd.DataFrame, source_ids: dict[str, str] | None = None,
                 release_lag_months: int = 0, training_mode: str = "legacy",
                 max_staleness_months: int | None = None):
        if variant not in VARIANT_INDICATORS:
            raise ValueError("variant должен быть M0, M1 или M2.")
        if training_mode != "legacy":
            raise ValueError("E05c сохраняет фиксированный режим legacy.")
        if "cat_features" in parameters or "one_hot_max_size" in parameters:
            raise ValueError("E05c сохраняет числовой календарь K0 без categorical/one-hot.")
        super().__init__(parameters, seed, release_lag_months, training_mode, max_staleness_months)
        self.variant = variant
        self.indicators = VARIANT_INDICATORS[variant]
        self.macrotable = macrotable.copy(deep=True)
        self.source_ids = None if source_ids is None else dict(source_ids)

    def predict(self, panel: pd.DataFrame, origin: pd.Period, horizon: int,
                forecast_ids: list[str], baseline_config: dict) -> MacroDirectResult:
        started = time.perf_counter()
        origin = pd.Period(origin, freq="M")
        prefix = get_prefix(panel, origin, self.lag)
        raw_X, delta, trace = build_direct_training(
            prefix, origin, horizon, release_lag_months=self.lag,
            mode=self.mode, max_staleness_months=self.max_staleness)
        horizon = int(horizon)  # Builder has already validated the integer horizon.
        if len(trace):
            r = trace.historical_origin.dt.to_period("M")
            valid = ((trace.feature_cutoff.dt.to_period("M") == r - self.lag) &
                     (trace.target_period.dt.to_period("M") == r + horizon) &
                     (trace.target_available_at.dt.to_period("M") == r + horizon + self.lag) &
                     (trace.target_available_at <= period_end(origin)))
            if not valid.all() or len(raw_X) != len(delta) or len(raw_X) != len(trace):
                raise TemporalIntegrityError("Нарушены даты или соответствие прямых обучающих пар.")
        samples = _queries(trace)
        X, macro_X, training_lineage = append_macro_features(
            raw_X, samples, self.macrotable, self.indicators, self.source_ids)
        history = prefix[forecast_ids]
        target = (origin + horizon).to_timestamp()
        raw_future, anchor = direct_features(history, target)
        future_samples = pd.DataFrame({
            "municipality_id": forecast_ids, "as_of_date": period_end(origin),
            "target_period": target, "sample_id": [str(position) for position in range(len(forecast_ids))],
        }, index=raw_future.index)
        X_future, macro_future, forecast_lineage = append_macro_features(
            raw_future, future_samples, self.macrotable, self.indicators, self.source_ids)
        training = {
            "forecast_origin": str(period_end(origin).date()), "horizon": horizon,
            "training_mode": self.mode, "variant": self.variant, "month_encoding": "K0",
            "n_training_rows": len(trace), "n_training_municipalities": int(trace.municipality_id.nunique()),
            "n_historical_origins": int(trace.historical_origin.nunique()),
            "n_target_periods": int(trace.target_period.nunique()),
            "min_historical_origin": str(trace.historical_origin.min().date()) if len(trace) else None,
            "max_historical_origin": str(trace.historical_origin.max().date()) if len(trace) else None,
            "max_feature_cutoff": str(trace.feature_cutoff.max().date()) if len(trace) else None,
            "max_training_target": str(trace.target_period.max().date()) if len(trace) else None,
            "max_target_available_at": str(trace.target_available_at.max().date()) if len(trace) else None,
            "n_forecast_series": len(forecast_ids), "fit_called": False, "fit_succeeded": False,
            **training_signature(trace, raw_X, delta),
            "training_ordered_key_sha256": ordered_training_key_signature(trace),
            "training_augmented_X_sha256": _frame_signature(X),
            "forecast_raw_X_sha256": _frame_signature(raw_future),
            "forecast_augmented_X_sha256": _frame_signature(X_future),
            "feature_names": X.columns.tolist(),
            "feature_dtypes": {name: str(dtype) for name, dtype in X.dtypes.items()},
            "cat_features": [], "fit_actual_parameters": None,
        }
        forecasts = pd.DataFrame({
            "municipality_id": forecast_ids, "y_pred": np.nan, "status": "failed",
            "effective_model": "none", "reason": "", "failure_stage": "",
            "fallback_baseline_status": "", "variant": self.variant,
        })
        for column in macro_future:
            forecasts[column] = macro_future[column].to_numpy()
        available_flags = []
        for indicator, label in ((INFLATION, "inflation"), (CONSUMPTION, "consumption")):
            available = np.isfinite(macro_future[indicator].to_numpy()) if indicator in macro_future else np.zeros(len(forecasts), dtype=bool)
            forecasts[label + "_available"] = available
            if indicator in self.indicators:
                available_flags.append(available)
                selected = forecast_lineage.loc[forecast_lineage.indicator.eq(indicator)].set_index("sample_id")
                selected = selected.reindex(future_samples.sample_id.to_numpy())
                for source_field in ("publication_date", "source_id", "vintage_id", "target_year"):
                    forecasts[label + "_" + source_field] = selected[source_field].to_numpy()
        forecasts["macro_features_available"] = np.logical_and.reduce(available_flags) if available_flags else False
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
                training["fit_actual_parameters"] = model.get_all_params()
                training["fit_actual_cat_feature_indices"] = model.get_cat_feature_indices()
                training["fit_actual_feature_names"] = model.feature_names_
                forecasts["effective_model"] = "CatBoostDirect" if self.variant == "M0" else "CatBoostDirect" + self.variant
                phase = "predict"
                values = np.maximum(anchor + model.predict(X_future), 0.0)
                if not np.isfinite(values).all():
                    raise RuntimeError("CatBoost вернул неконечный прогноз.")
                forecasts["y_pred"] = values
                forecasts["status"] = "native"
                importance = pd.DataFrame({"feature": X.columns, "importance": model.feature_importances_})
        except Exception as exc:
            forecasts["y_pred"] = np.nan
            forecasts["status"] = "failed"
            forecasts["reason"] = repr(exc)
            forecasts["failure_stage"] = phase
            errors.append({"forecast_origin": training["forecast_origin"], "horizon": horizon,
                           "variant": self.variant, "stage": phase, "error": repr(exc),
                           "municipality_ids": forecast_ids})
        diagnostics = macro_diagnostics(samples, macro_X, training_lineage, self.indicators,
                                        origin=origin, horizon=horizon, variant=self.variant, scope="training")
        diagnostics += macro_diagnostics(future_samples, macro_future, forecast_lineage, self.indicators,
                                         origin=origin, horizon=horizon, variant=self.variant, scope="forecast")
        training_lineage = _attach_training_keys(training_lineage, trace, origin, horizon, self.variant)
        if len(forecast_lineage):
            future_keys = future_samples[["sample_id", "municipality_id", "target_period"]].copy()
            overlap = [column for column in future_keys if column in forecast_lineage and column != "sample_id"]
            forecast_lineage = forecast_lineage.drop(columns=overlap).merge(
                future_keys, on="sample_id", how="left", validate="many_to_one", sort=False)
        forecast_lineage["forecast_origin"] = str(period_end(origin).date())
        forecast_lineage["feature_cutoff"] = period_end(origin - self.lag)
        forecast_lineage["horizon"] = horizon
        forecast_lineage["variant"] = self.variant
        forecast_lineage["scope"] = "forecast"
        training["status"] = forecasts.status.iloc[0]
        training["seconds"] = time.perf_counter() - started
        return MacroDirectResult(forecasts, training, importance, errors,
                                 training_lineage, forecast_lineage, diagnostics)
