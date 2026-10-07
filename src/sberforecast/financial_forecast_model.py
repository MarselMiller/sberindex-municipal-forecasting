"""Fixed E08c LightGBM ablation over the unchanged E05d direct pairs.

Financial covariates are national and joined at each pair's own historical
origin, rather than at the current fitting origin. F0 delegates to E05d.
The original target, nominal/ratio histories, N_hat and fallback are reused.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable
import time

import numpy as np
import pandas as pd

from . import national_local_model as original
from .calendar_models import training_signature
from .data import get_prefix, period_end
from .direct_model import TemporalIntegrityError
from .direct_training import build_direct_training, direct_features
from .leading_financial import FEATURE_COLUMNS, origin_timestamp
from .models import baseline_predict
from .national_local import build_local_training, national_forecast, ratio_panel

FINANCIAL_VARIANTS = {
    "F0": (), "F1": FEATURE_COLUMNS[:5], "F2": FEATURE_COLUMNS[5:],
    "F3": FEATURE_COLUMNS,
}
SOURCE_AUDIT_COLUMNS = ("max_source_date_used", "max_source_available_at_used")


@dataclass
class FinancialResult(original.NationalLocalResult):
    financial_training_provenance: pd.DataFrame
    financial_forecast_provenance: pd.DataFrame
    audit_counts: dict


def _variant(features: Iterable[str]) -> tuple[str, tuple[str, ...]]:
    selected = tuple(features)
    for variant, expected in FINANCIAL_VARIANTS.items():
        if selected == expected:
            return variant, selected
    raise ValueError("E08c permits only the ordered F0/F1/F2/F3 feature lists from E08b.")


def _indexed_matrix(matrix: pd.DataFrame, selected: tuple[str, ...]) -> pd.DataFrame:
    if not matrix.columns.is_unique:
        raise ValueError("Duplicate financial matrix columns.")
    required = {"forecast_origin", *selected, *SOURCE_AUDIT_COLUMNS}
    if not required.issubset(matrix.columns):
        raise ValueError(f"Missing financial matrix columns: {sorted(required - set(matrix.columns))}")
    result = matrix.copy(deep=True)
    result["forecast_origin"] = result.forecast_origin.map(origin_timestamp)
    if result.forecast_origin.isna().any() or result.forecast_origin.duplicated().any():
        raise ValueError("Financial matrix origins must be nonmissing and unique after normalization.")
    return result.set_index("forecast_origin", drop=False)


def augment_financial_features(
    X: pd.DataFrame, own_origins: Iterable, financial_matrix: pd.DataFrame,
    financial_features: Iterable[str], model_origin,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Append only numeric covariates, preserving every existing row and column.

    Return the augmented X, a full E08b join ledger, and successful audit counts.
    Missing numeric values stay NaN. Missing matrix rows or future operands stop
    the experiment before fit; neither forward fill nor row deletion is used.
    """
    _, selected = _variant(financial_features)
    own = pd.DatetimeIndex([origin_timestamp(value) for value in own_origins], tz="Europe/Moscow")
    current = origin_timestamp(model_origin)
    if len(own) != len(X) or own.hasnans or pd.isna(current):
        raise TemporalIntegrityError("Financial origins must match the feature rows exactly.")
    if (own > current).any():
        raise TemporalIntegrityError("Financial feature origin is later than model forecast origin.")
    counts = {"n_rows_checked": len(X), "n_unique_financial_origins": len(own.unique()),
              "n_missing_feature_values": 0, "financial_cutoff_violations": 0,
              "financial_origin_mismatches": 0}
    if not selected:
        return X.copy(deep=True), pd.DataFrame(index=X.index), counts
    matrix = _indexed_matrix(financial_matrix, selected)
    if not own.isin(matrix.index).all():
        absent = own[~own.isin(matrix.index)].unique()
        raise TemporalIntegrityError(f"Missing own-origin financial rows: {absent.tolist()}")
    joined = matrix.reindex(own).reset_index(drop=True)
    joined.index = X.index
    if not pd.DatetimeIndex(joined.forecast_origin).equals(own):
        raise TemporalIntegrityError("Financial own-origin join changed row correspondence.")
    # The aggregate operands audit both rate-difference levels and all original
    # FX-setting return operands, not merely the last source observation.
    for column in SOURCE_AUDIT_COLUMNS:
        dates = joined[column].map(lambda value: pd.NaT if pd.isna(value) else origin_timestamp(value))
        late = np.array([False if pd.isna(value) else value > own[position]
                         for position, value in enumerate(dates)])
        if late.any():
            raise TemporalIntegrityError(f"{column} exceeds the row's own financial origin.")
    # Preserve and additionally check every populated per-source anchor/last
    # observation date. Method-switch metadata are not source operands.
    source_columns = [column for column in joined if (
        column.startswith("key_rate_anchor_") or column.startswith("usd_rub_anchor_")
        or (column.startswith(("key_rate_last_", "usd_rub_last_"))
            and column.endswith(("_date", "_at"))))]
    for column in source_columns:
        for position, value in enumerate(joined[column]):
            if not pd.isna(value) and origin_timestamp(value) > own[position]:
                raise TemporalIntegrityError(f"{column} exceeds the row's own financial origin.")
    result = X.copy(deep=True)
    for feature in selected:
        if feature in X:
            raise ValueError(f"Financial feature already present in base X: {feature}")
        if pd.api.types.is_bool_dtype(joined[feature]):
            raise ValueError(f"Financial covariate must be numeric, not boolean: {feature}")
        values = pd.to_numeric(joined[feature], errors="raise").to_numpy(dtype=float)
        if np.isinf(values).any():
            raise ValueError(f"Infinite financial covariate: {feature}")
        result[feature] = values
        counts["n_missing_feature_values"] += int(np.isnan(values).sum())
    # Any populated selected numeric feature needs both aggregate audit bounds;
    # NaT is legitimate only when the selected row has no observed operands.
    observed = result.loc[:, list(selected)].notna().any(axis=1)
    if (observed & joined.loc[:, list(SOURCE_AUDIT_COLUMNS)].isna().any(axis=1)).any():
        raise TemporalIntegrityError("Observed financial covariates lack source audit dates.")
    provenance = joined.rename(columns={
        name: ("financial_feature_origin" if name == "forecast_origin" else
               name if name in FEATURE_COLUMNS else f"financial_{name}")
        for name in joined.columns
    })
    provenance["financial_model_origin"] = current
    return result, provenance, counts


class FinancialDirect(original.NationalLocalDirect):
    """The LN primary and L0 secondary fixed learners, with F0 through F3."""
    def __init__(self, parameters: dict, seed: int, base_variant: str,
                 financial_matrix: pd.DataFrame, financial_features: Iterable[str],
                 national: pd.DataFrame | None = None, release_lag_months: int = 0,
                 training_mode: str = "legacy", max_staleness_months: int | None = None):
        if base_variant not in ("L0", "LN"):
            raise ValueError("E08c permits only fixed LightGBM L0 and LN.")
        self.financial_variant, self.financial_features = _variant(financial_features)
        self.financial_matrix = financial_matrix.copy(deep=True)
        super().__init__(parameters, seed, base_variant, national, release_lag_months,
                         training_mode, max_staleness_months)

    def predict(self, panel: pd.DataFrame, origin: pd.Period, horizon: int,
                forecast_ids: list[str], baseline_config: dict) -> FinancialResult:
        if not self.financial_features:
            baseline = super().predict(panel, origin, horizon, forecast_ids, baseline_config)
            training = dict(baseline.training, financial_variant="F0", financial_feature_names=[])
            return FinancialResult(baseline.forecasts, training, baseline.importance, baseline.errors,
                                   baseline.training_provenance, baseline.local_ratio_diagnostics,
                                   baseline.national_forecast, pd.DataFrame(), pd.DataFrame(),
                                   {"training_rows_checked": 0, "forecast_rows_checked": 0,
                                    "financial_cutoff_violations": 0, "financial_origin_mismatches": 0})
        started = time.perf_counter()
        origin = pd.Period(origin, freq="M")
        prefix = get_prefix(panel, origin, self.lag)
        nominal_X, nominal_delta, nominal_trace = build_direct_training(
            prefix, origin, horizon, release_lag_months=self.lag,
            mode=self.mode, max_staleness_months=self.max_staleness)
        decomposition = self.variant == "LN"
        diagnostics, national_record = [], {}
        if decomposition:
            base_X, delta, trace = build_local_training(
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
            national_record = national_forecast(self.national, origin, horizon, baseline_config,
                                                release_lag_months=self.lag)
        else:
            base_X, delta, trace = nominal_X, nominal_delta, nominal_trace
            model_history = prefix[forecast_ids]
        horizon = int(horizon)
        original._validate_trace(trace, base_X, delta, origin, horizon, self.lag, decomposition)
        target = (origin + horizon).to_timestamp()
        base_future, anchor = direct_features(model_history, target)
        nominal_future, nominal_anchor = direct_features(prefix[forecast_ids], target)
        X, financial_training, training_audit = augment_financial_features(
            base_X, trace.historical_origin, self.financial_matrix, self.financial_features, origin)
        X_future, financial_forecast, forecast_audit = augment_financial_features(
            base_future, [origin] * len(forecast_ids), self.financial_matrix,
            self.financial_features, origin)
        training = {
            "forecast_origin": str(period_end(origin).date()), "horizon": horizon,
            "variant": self.variant, "financial_variant": self.financial_variant,
            "training_mode": self.mode, "month_encoding": "K0", "learner": "LightGBM",
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
            "base_source_signature": training_signature(trace, base_X, delta),
            "training_ordered_key_sha256": original.ordered_training_key_signature(trace),
            "forecast_raw_X_sha256": original.frame_signature(X_future),
            "base_forecast_raw_X_sha256": original.frame_signature(base_future),
            "forecast_anchor_sha256": original.vector_signature(anchor),
            "nominal_source_signature": training_signature(nominal_trace, nominal_X, nominal_delta),
            "nominal_source_ordered_key_sha256": original.ordered_training_key_signature(nominal_trace),
            "nominal_forecast_raw_X_sha256": original.frame_signature(nominal_future),
            "nominal_forecast_anchor_sha256": original.vector_signature(nominal_anchor),
            "n_nominal_source_pairs": len(nominal_trace),
            "n_pairs_excluded_by_ratio_validity": len(nominal_trace) - len(trace),
            "financial_feature_names": list(self.financial_features),
            "feature_names": X.columns.tolist(),
            "feature_dtypes": {name: str(dtype) for name, dtype in X.dtypes.items()},
            "cat_features": [], "fit_actual_parameters": None, "fit_actual_cat_feature_indices": [],
            "financial_training_provenance_sha256": original.frame_signature(financial_training),
            "financial_forecast_provenance_sha256": original.frame_signature(financial_forecast),
        }
        provenance = trace.copy(deep=True)
        provenance["training_row"] = np.arange(len(trace))
        provenance["variant"] = self.variant
        provenance["delta"] = delta
        provenance = pd.concat([provenance, financial_training], axis=1)
        financial_forecast = financial_forecast.copy()
        financial_forecast.insert(0, "municipality_id", forecast_ids)
        training["financial_forecast_provenance_sha256"] = original.frame_signature(financial_forecast)
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
                parameters = dict(self.parameters)
                parameters.setdefault("random_state", self.seed)
                parameters.setdefault("use_missing", True)
                parameters.setdefault("zero_as_missing", False)
                model = original._lightgbm_regressor(**parameters)
                training["fit_called"] = True
                model.fit(X, delta)
                training["fit_succeeded"] = True
                training.update(original._lightgbm_metadata(model))
                forecasts["effective_model"] = original.NATIVE_MODELS[self.variant]
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
                           "variant": self.variant, "financial_variant": self.financial_variant,
                           "stage": phase, "error": repr(exc), "municipality_ids": forecast_ids})
        training["status"] = forecasts.status.iloc[0]
        training["seconds"] = time.perf_counter() - started
        audit_counts = {"training_rows_checked": training_audit["n_rows_checked"],
                       "forecast_rows_checked": forecast_audit["n_rows_checked"],
                       "financial_cutoff_violations": 0, "financial_origin_mismatches": 0,
                       "training": training_audit, "forecast": forecast_audit}
        return FinancialResult(forecasts, training, importance, errors, provenance,
                               diagnostics, national_record, financial_training,
                               financial_forecast, audit_counts)
