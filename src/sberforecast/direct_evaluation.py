"""Две явные области сравнения E02; failed не исчезает из полной оценки."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .direct_model import NATIVE_MODEL, STRATEGY
from .metrics import KEY, common_support, compute_metrics, coverage_table


def require_same_keys(left: pd.DataFrame, right: pd.DataFrame) -> None:
    for frame in (left, right):
        if frame[KEY].isna().any().any() or frame.duplicated(KEY).any():
            raise ValueError("Пустые или повторяющиеся прогнозные ключи.")
    a = set(map(tuple, left[KEY].to_numpy()))
    b = set(map(tuple, right[KEY].to_numpy()))
    if a != b:
        raise ValueError(f"Несовпадение ключей: отсутствует {len(b-a)}, лишних {len(a-b)}.")


def validate_direct_against_reference(direct: pd.DataFrame, reference: pd.DataFrame) -> None:
    for name, ref in reference.groupby("model"):
        require_same_keys(direct, ref)
        paired = direct.merge(ref, on=KEY, suffixes=("_direct", "_reference"), validate="one_to_one")
        a, b = paired.y_true_direct.to_numpy(), paired.y_true_reference.to_numpy()
        if not ((a == b) | (np.isnan(a) & np.isnan(b))).all():
            raise ValueError(f"Целевые факты отличаются от E01: {name}.")
        for field in ("history_cutoff", "split", "availability_assumption"):
            if field in direct and field in ref and not paired[f"{field}_direct"].equals(paired[f"{field}_reference"]):
                raise ValueError(f"Поле {field} отличается от E01: {name}.")
    native = direct.status.eq("native")
    fallback = direct.status.eq("fallback_no_training_pairs")
    failed = direct.status.eq("failed")
    if not (native | fallback | failed).all():
        raise ValueError("Неизвестный статус прямого прогноза.")
    if not direct.loc[native, "effective_model"].eq(NATIVE_MODEL).all():
        raise ValueError("Нативный прогноз имеет другую effective_model.")
    if not direct.loc[fallback, "effective_model"].eq("SeasonalNaive").all():
        raise ValueError("Резервный прогноз не обозначен как SeasonalNaive.")
    if not np.isfinite(direct.loc[~failed, "y_pred"]).all() or np.isfinite(direct.loc[failed, "y_pred"]).any():
        raise ValueError("Конечность прогноза не соответствует статусу.")


def h1_consistency(direct: pd.DataFrame, reference: pd.DataFrame,
                   absolute_tolerance: float) -> tuple[dict, pd.DataFrame]:
    ref = reference.loc[reference.model.eq("CatBoostRecursive") & reference.horizon.eq(1)]
    forecast = direct.loc[direct.horizon.eq(1)]
    require_same_keys(forecast, ref)
    paired = forecast.merge(ref[KEY + ["y_pred"]], on=KEY, suffixes=("", "_recursive"), validate="one_to_one")
    valid = paired.status.eq("native") & np.isfinite(paired.y_pred) & np.isfinite(paired.y_pred_recursive)
    compared = paired.loc[valid].copy()
    compared["absolute_difference"] = np.abs(compared.y_pred - compared.y_pred_recursive)
    compared["exceeds_tolerance"] = compared.absolute_difference > absolute_tolerance
    by_origin = compared.groupby("forecast_origin").agg(
        n_compared=("municipality_id", "size"), max_absolute_difference=("absolute_difference", "max"),
        n_exceeds_tolerance=("exceeds_tolerance", "sum"),
    ).reset_index()
    result = {
        "absolute_tolerance": absolute_tolerance, "relative_tolerance": 0.0,
        "n_expected_cases": len(paired), "n_compared_native_cases": len(compared),
        "n_uncompared_cases": len(paired) - len(compared),
        "max_absolute_difference": float(compared.absolute_difference.max()) if len(compared) else None,
        "n_exceeds_tolerance": int(compared.exceeds_tolerance.sum()),
        "complete": bool(len(compared) == len(paired)),
    }
    return result, by_origin


def _group_metrics(frame: pd.DataFrame, groups: pd.DataFrame, models: list[str],
                   scope: str, no_native_pairs: set[tuple]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, per_mo = [], []
    for split, horizon in groups.itertuples(index=False, name=None):
        for name in models:
            sub = frame.loc[frame.split.eq(split) & frame.horizon.eq(horizon) & frame.model.eq(name)]
            bad = int((~np.isfinite(sub.y_pred)).sum())
            if len(sub) and not bad:
                metrics, by_mo = compute_metrics(sub)
                row = metrics.iloc[0].to_dict()
                per_mo.append(by_mo.assign(scope=scope))
                state = "complete"
            else:
                row = {"split": split, "model": name, "horizon": int(horizon),
                       "n_predictions": len(sub), "n_municipalities": sub.municipality_id.nunique(),
                       "n_origins": sub.forecast_origin.nunique(),
                       "mae_macro": np.nan, "mae_micro": np.nan, "r2_pooled": np.nan}
                if bad:
                    state = "incomplete_failed"
                elif scope == "native" and name == NATIVE_MODEL and (split, horizon) in no_native_pairs:
                    state = "no_training_pairs"
                else:
                    state = "no_native_comparison_keys"
            rows.append(dict(row, scope=scope, metric_status=state, n_failed=bad))
    return pd.DataFrame(rows), pd.concat(per_mo, ignore_index=True) if per_mo else pd.DataFrame()


def evaluate_direct(direct: pd.DataFrame, reference: pd.DataFrame,
                    models: list[str]) -> dict[str, pd.DataFrame]:
    """A: все общие случаи E01; B: native E02 и соперники на тех же ключах."""
    validate_direct_against_reference(direct, reference)
    common = common_support(reference, sorted(reference.model.unique()))
    support = common[KEY].drop_duplicates()
    current = direct.merge(support, on=KEY, how="inner", validate="one_to_one")
    baseline = common.loc[common.model.isin(models)].copy()
    groups = current[["split", "horizon"]].drop_duplicates().sort_values(["split", "horizon"])
    strategy = current.copy()
    strategy["model"] = STRATEGY
    all_strategy = pd.concat([strategy, baseline], ignore_index=True)
    native = current.loc[current.status.eq("native")].copy()
    native["model"] = NATIVE_MODEL
    native_reference = baseline.merge(native[KEY], on=KEY, how="inner", validate="many_to_one")
    all_native = pd.concat([native, native_reference], ignore_index=True)
    no_pairs = {(s, int(h)) for (s, h), g in current.groupby(["split", "horizon"])
                if g.status.eq("fallback_no_training_pairs").all()}
    strategy_metrics, strategy_mo = _group_metrics(all_strategy, groups, [STRATEGY, *models], "strategy", no_pairs)
    native_metrics, native_mo = _group_metrics(all_native, groups, [NATIVE_MODEL, *models], "native", no_pairs)
    coverage = coverage_table(direct)
    status_counts = direct.groupby(["split", "horizon"]).agg(
        n_native=("status", lambda x: int(x.eq("native").sum())),
        n_failed_status=("status", lambda x: int(x.eq("failed").sum())),
    ).reset_index()
    coverage = coverage.merge(status_counts, on=["split", "horizon"], validate="one_to_one")
    coverage["fallback_share"] = coverage.n_fallback / coverage.n_requested
    coverage["failed_share"] = coverage.n_failed_status / coverage.n_requested
    by_status = direct.assign(actual_available=np.isfinite(direct.y_true),
                              forecast_available=np.isfinite(direct.y_pred)).groupby(
        ["forecast_origin", "split", "horizon", "status", "effective_model"], dropna=False,
    ).agg(n_requested=("municipality_id", "size"), n_actual_available=("actual_available", "sum"),
          n_forecast_available=("forecast_available", "sum")).reset_index()
    denominators = direct.groupby(["forecast_origin", "horizon"]).size().rename("n_origin_requested").reset_index()
    by_status = by_status.merge(denominators, on=["forecast_origin", "horizon"], validate="many_to_one")
    by_status["status_share"] = by_status.n_requested / by_status.n_origin_requested
    e01_coverage = current.groupby(["split", "horizon"]).agg(
        n_e01_cases=("municipality_id", "size"),
        n_native=("status", lambda x: int(x.eq("native").sum())),
        n_fallback=("status", lambda x: int(x.eq("fallback_no_training_pairs").sum())),
        n_failed=("status", lambda x: int(x.eq("failed").sum())),
    ).reset_index()
    e01_coverage["fallback_share"] = e01_coverage.n_fallback / e01_coverage.n_e01_cases
    e01_coverage["failed_share"] = e01_coverage.n_failed / e01_coverage.n_e01_cases
    return {"metrics_strategy": strategy_metrics, "metrics_native": native_metrics,
            "metrics_strategy_by_municipality": strategy_mo, "metrics_native_by_municipality": native_mo,
            "coverage": coverage, "coverage_by_origin_status": by_status,
            "coverage_e01": e01_coverage,
            "evaluation_keys_strategy": current[KEY], "evaluation_keys_native": native[KEY]}
