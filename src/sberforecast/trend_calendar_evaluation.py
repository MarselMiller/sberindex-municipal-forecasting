"""Парная оценка E05b: полная стратегия и общее нативное пересечение.

Область полной оценки определяется наличием факта в E01. Ошибка новой
модели остаётся строкой этой области и делает её итоговую метрику неполной.
Ориентиры на нативной области оцениваются по тем же ключам, что варианты.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .direct_evaluation import require_same_keys
from .metrics import KEY, compute_metrics, coverage_table


PROTOCOL_FIELDS = ["history_cutoff", "split", "availability_assumption"]
GROUP = ["split", "model", "horizon"]
SUMMARY_COLUMNS = GROUP + [
    "n_predictions", "n_municipalities", "n_origins", "mae_macro", "mae_micro",
    "r2_pooled", "scope", "metric_status", "n_native", "n_fallback", "n_failed",
]
MO_COLUMNS = GROUP + ["municipality_id", "n", "mae", "r2", "scope", "metric_status", "n_failed"]
COVERAGE_COLUMNS = GROUP + [
    "n_requested", "n_actual_available", "n_forecast_available", "n_failed",
    "n_fallback", "forecast_coverage", "n_native", "native_share", "fallback_share", "failed_share",
]


def _require_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    absent = sorted(set(columns) - set(frame.columns))
    if absent:
        raise ValueError(f"Нет обязательных полей {label}: {', '.join(absent)}.")


def _validate(predictions: pd.DataFrame, expected: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    _require_columns(expected, KEY + ["y_true"] + PROTOCOL_FIELDS, "эталона")
    _require_columns(predictions, KEY + [
        "model", "y_true", "y_pred", "status", "effective_model", "reason",
    ] + PROTOCOL_FIELDS, "прогнозов")
    require_same_keys(expected, expected)
    if set(predictions.model.unique()) != set(models):
        raise ValueError("Состав моделей не соответствует объявленной группе сравнения.")
    result = predictions.copy()
    for field in ("y_true", "y_pred"):
        result[field] = pd.to_numeric(result[field], errors="raise").astype(float)
    for name in models:
        current = result.loc[result.model.eq(name)]
        require_same_keys(current, expected)
        paired = current.merge(expected, on=KEY, suffixes=("", "_expected"), validate="one_to_one")
        truth = paired.y_true.to_numpy(dtype=float)
        reference = paired.y_true_expected.to_numpy(dtype=float)
        if not ((truth == reference) | (np.isnan(truth) & np.isnan(reference))).all():
            raise ValueError(f"Целевые факты отличаются от E01: {name}.")
        for field in PROTOCOL_FIELDS:
            if not paired[field].equals(paired[field + "_expected"]):
                raise ValueError(f"Поле {field} отличается от E01: {name}.")
    status = result.status.fillna("").astype(str)
    native = status.eq("native")
    fallback = status.str.startswith("fallback")
    failed = status.eq("failed")
    if not (native | fallback | failed).all():
        raise ValueError("Неизвестный статус прогноза E05b.")
    if not np.isfinite(result.loc[~failed, "y_pred"]).all() or not result.loc[failed, "y_pred"].isna().all():
        raise ValueError("Конечность прогноза не соответствует статусу E05b.")
    effective = result.effective_model.fillna("").astype(str).str.strip()
    if (effective.loc[~failed].eq("") | effective.loc[~failed].eq("none")).any():
        raise ValueError("Успешному прогнозу нужна фактическая effective_model.")
    reason = result.reason.fillna("").astype(str).str.strip()
    if reason.loc[fallback | failed].eq("").any():
        raise ValueError("Резервному или failed-прогнозу нужна явная причина.")
    result["status"] = status
    return result


def _summary(sub: pd.DataFrame, fields: dict, scope: str) -> dict:
    failed = int((~np.isfinite(sub.y_pred.to_numpy(dtype=float))).sum())
    if len(sub) and not failed:
        metrics, _ = compute_metrics(sub)
        row = metrics.iloc[0].to_dict()
        state = "complete"
    else:
        row = {
            "n_predictions": len(sub), "n_municipalities": sub.municipality_id.nunique(),
            "n_origins": sub.forecast_origin.nunique(),
            "mae_macro": np.nan, "mae_micro": np.nan, "r2_pooled": np.nan,
        }
        state = "incomplete_failed" if failed else (
            "no_native_forecasts" if scope == "native" else "no_evaluable_cases")
    return dict(row, **fields, scope=scope, metric_status=state, n_failed=failed,
                n_native=int(sub.status.eq("native").sum()),
                n_fallback=int(sub.status.str.startswith("fallback").sum()))


def _metric_tables(frame: pd.DataFrame, groups: pd.DataFrame, models: list[str], scope: str,
                   *, by_origin: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_columns = ["forecast_origin", "split", "horizon"] if by_origin else ["split", "horizon"]
    rows, per_mo = [], []
    for values in groups[group_columns].itertuples(index=False, name=None):
        group = dict(zip(group_columns, values))
        selected = pd.Series(True, index=frame.index)
        for field, value in group.items():
            selected &= frame[field].eq(value)
        for model in models:
            sub = frame.loc[selected & frame.model.eq(model)]
            rows.append(_summary(sub, dict(group, model=model), scope))
            if by_origin:
                continue
            for municipality_id, mo in sub.groupby("municipality_id", sort=True):
                failed = int((~np.isfinite(mo.y_pred.to_numpy(dtype=float))).sum())
                if not failed:
                    _, detail = compute_metrics(mo)
                    record = detail.iloc[0].to_dict()
                else:
                    record = dict(group, model=model, municipality_id=municipality_id,
                                  n=len(mo), mae=np.nan, r2=np.nan)
                per_mo.append(dict(record, scope=scope, n_failed=failed,
                                   metric_status="incomplete_failed" if failed else "complete"))
    columns = ["forecast_origin", *SUMMARY_COLUMNS] if by_origin else SUMMARY_COLUMNS
    return pd.DataFrame(rows, columns=columns), pd.DataFrame(per_mo, columns=MO_COLUMNS)


def _coverage(frame: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    """Использует прежний счётчик, сохраняя также нулевые группы без фактов."""
    if len(frame):
        table = coverage_table(frame)
        native = frame.groupby(GROUP, sort=True).status.agg(
            lambda status: int(status.eq("native").sum())).rename("n_native").reset_index()
        table = table.merge(native, on=GROUP, validate="one_to_one")
        table = groups.merge(table, on=GROUP, how="left", validate="one_to_one")
    else:
        table = groups.copy()
    counts = ["n_requested", "n_actual_available", "n_forecast_available", "n_failed", "n_fallback", "n_native"]
    for field in counts:
        if field not in table:
            table[field] = 0
        table[field] = table[field].fillna(0).astype(int)
    denominator = table.n_requested.replace(0, np.nan)
    table["forecast_coverage"] = table.n_forecast_available / denominator
    table["native_share"] = table.n_native / denominator
    table["fallback_share"] = table.n_fallback / denominator
    table["failed_share"] = table.n_failed / denominator
    return table.reindex(columns=COVERAGE_COLUMNS)


def evaluate_group(predictions: pd.DataFrame, expected: pd.DataFrame, models: list[str],
                   native_models: list[str]) -> dict[str, pd.DataFrame]:
    """Оценить фиксированную группу без скрытого удаления ошибок или резервов.

    ``native_models`` задаёт варианты, для которых обязательно нативное
    исполнение. Ориентир (например SeasonalNaiveYoY) должен иметь конечный
    прогноз на тех же ключах, но его внутренний резерв не сужает пересечение.
    Ни признаки, ни результаты прогнозирования эта функция не изменяет.
    """
    if not models or len(set(models)) != len(models):
        raise ValueError("Нужен непустой список разных моделей.")
    if not native_models or len(set(native_models)) != len(native_models) or set(native_models) - set(models):
        raise ValueError("Нативные варианты должны быть непустым подмножеством моделей.")
    raw = _validate(predictions, expected, models)
    support = expected.loc[np.isfinite(expected.y_true.to_numpy(dtype=float)), KEY + ["split"]].copy()
    full = raw.merge(support[KEY], on=KEY, how="inner", validate="many_to_one")
    eligible = full.loc[(~full.model.isin(native_models) | full.status.eq("native")) &
                        np.isfinite(full.y_pred.to_numpy(dtype=float))]
    counts = eligible.groupby(KEY, dropna=False).model.nunique()
    shared = counts.loc[counts.eq(len(models))].reset_index()[KEY]
    common_keys = support.merge(shared, on=KEY, how="inner", validate="one_to_one")
    common = full.merge(shared, on=KEY, how="inner", validate="many_to_one")
    excluded = support.merge(shared.assign(in_common_native=True), on=KEY, how="left", validate="one_to_one")
    excluded = excluded.loc[excluded.in_common_native.isna()].drop(columns="in_common_native")
    bad = full.loc[(full.model.isin(native_models) & ~full.status.eq("native")) |
                   ~np.isfinite(full.y_pred.to_numpy(dtype=float))].copy()
    if len(bad):
        bad["exclusion"] = bad.model.astype(str) + ":" + bad.status.astype(str)
        explanation = bad.reason.fillna("").astype(str)
        bad["exclusion"] += np.where(explanation.eq(""), "", " (" + explanation + ")")
        reasons = bad.groupby(KEY, dropna=False).exclusion.agg(
            lambda values: "; ".join(sorted(values))).rename("reason").reset_index()
        excluded = excluded.merge(reasons, on=KEY, how="left", validate="one_to_one")
    else:
        excluded["reason"] = pd.Series(dtype=str)
    if len(excluded) and excluded.reason.isna().any():
        raise RuntimeError("Не найдена причина исключения из нативной области.")

    groups = expected[["split", "horizon"]].drop_duplicates().sort_values(["split", "horizon"])
    origin_groups = expected[["forecast_origin", "split", "horizon"]].drop_duplicates().sort_values(
        ["forecast_origin", "split", "horizon"])
    strategy_metrics, strategy_mo = _metric_tables(full, groups, models, "strategy")
    native_metrics, native_mo = _metric_tables(common, groups, models, "native")
    strategy_origin, _ = _metric_tables(full, origin_groups, models, "strategy", by_origin=True)
    native_origin, _ = _metric_tables(common, origin_groups, models, "native", by_origin=True)
    coverage_groups = raw[GROUP].drop_duplicates().sort_values(GROUP)
    coverage = _coverage(raw, coverage_groups)
    evaluable_coverage = _coverage(full, coverage_groups)
    native_coverage = coverage[GROUP + ["n_requested", "n_native", "n_fallback", "n_failed"]].merge(
        evaluable_coverage[GROUP + ["n_requested", "n_native", "n_fallback", "n_failed"]].rename(columns={
            "n_requested": "n_evaluable", "n_native": "n_native_evaluable",
            "n_fallback": "n_fallback_evaluable", "n_failed": "n_failed_evaluable",
        }), on=GROUP, validate="one_to_one")
    native_coverage["native_share_requested"] = native_coverage.n_native / native_coverage.n_requested.replace(0, np.nan)
    native_coverage["native_share_evaluable"] = native_coverage.n_native_evaluable / native_coverage.n_evaluable.replace(0, np.nan)
    native_coverage["n_common_native_evaluable"] = native_coverage.merge(
        common_keys.groupby(["split", "horizon"]).size().rename("n_common").reset_index(),
        on=["split", "horizon"], how="left", validate="many_to_one").n_common.fillna(0).astype(int)
    own_native_keys = full.loc[full.status.eq("native") & np.isfinite(full.y_pred.to_numpy(dtype=float)),
                               KEY + ["split", "model"]].copy()
    return {
        "metrics_strategy": strategy_metrics, "metrics_native": native_metrics,
        "metrics_strategy_by_municipality": strategy_mo, "metrics_native_by_municipality": native_mo,
        "metrics_strategy_by_origin": strategy_origin, "metrics_native_by_origin": native_origin,
        "coverage": coverage, "coverage_evaluable": evaluable_coverage,
        "native_coverage_by_model": native_coverage,
        "strategy_keys": support, "common_native_keys": common_keys,
        "excluded_native_keys": excluded, "own_native_keys": own_native_keys,
    }
