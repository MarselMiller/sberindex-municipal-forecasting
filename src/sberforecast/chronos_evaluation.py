"""E03: полная область фактов и отдельное пересечение успешных прогнозов."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import KEY, compute_metrics

MODEL = "Chronos-2"
DIRECT = "CatBoostDirect"
E01_MODELS = ["SeasonalNaiveYoY", "CatBoostRecursive", "ProphetAuto", "ProphetYearly"]
MODELS = [MODEL, DIRECT, *E01_MODELS]


def require_same_keys(left: pd.DataFrame, right: pd.DataFrame) -> None:
    """Сравнить фактические множества; количество строк само по себе недостаточно."""
    for frame in (left, right):
        if frame[KEY].isna().any().any() or frame.duplicated(KEY).any():
            raise ValueError("Пустые или повторяющиеся прогнозные ключи.")
    a = set(map(tuple, left[KEY].to_numpy()))
    b = set(map(tuple, right[KEY].to_numpy()))
    if a != b:
        raise ValueError(f"Несовпадение ключей: отсутствует {len(b-a)}, лишних {len(a-b)}.")


def _validate_facts(current: pd.DataFrame, expected: pd.DataFrame) -> None:
    require_same_keys(current, expected)
    paired = current.merge(expected, on=KEY, suffixes=("_current", "_expected"), validate="one_to_one")
    a = paired.y_true_current.to_numpy(dtype=float)
    b = paired.y_true_expected.to_numpy(dtype=float)
    if not ((a == b) | (np.isnan(a) & np.isnan(b))).all():
        raise ValueError("Целевые факты отличаются от сохранённого эталона.")
    for field in ("history_cutoff", "split", "availability_assumption"):
        if field in current or field in expected:
            if field not in current or field not in expected:
                raise ValueError(f"Нет обязательного поля {field} для проверки сопоставимости.")
            if not paired[f"{field}_current"].equals(paired[f"{field}_expected"]):
                raise ValueError(f"Поле {field} отличается от сохранённого эталона.")


def validate_predictions(current: pd.DataFrame, expected: pd.DataFrame) -> None:
    """Chronos не заменяется резервом; failed обязан оставаться явной NaN-строкой."""
    _validate_facts(current, expected)
    if "model" in current and not current.model.eq(MODEL).all():
        raise ValueError("Ожидались только прогнозы Chronos-2.")
    native, failed = current.status.eq("native"), current.status.eq("failed")
    if not (native | failed).all():
        raise ValueError("Chronos-2 допускает только native и failed, без fallback.")
    if not np.isfinite(current.loc[native, "y_pred"].to_numpy(dtype=float)).all():
        raise ValueError("Нативный прогноз Chronos-2 должен быть конечным.")
    if not current.loc[failed, "y_pred"].isna().all():
        raise ValueError("Failed-прогноз Chronos-2 должен иметь y_pred=NaN.")
    if failed.any() and ("reason" not in current or
                         current.loc[failed, "reason"].fillna("").astype(str).str.strip().eq("").any()):
        raise ValueError("Failed-прогноз Chronos-2 должен иметь причину.")
    if "effective_model" in current and not current.loc[native, "effective_model"].eq(MODEL).all():
        raise ValueError("Нативный прогноз Chronos-2 имеет другую effective_model.")


def _pure_direct(e02: pd.DataFrame) -> pd.DataFrame:
    native = e02.status.eq("native")
    reserve = e02.status.eq("fallback_no_training_pairs")
    failed = e02.status.eq("failed")
    if not (native | reserve | failed).all():
        raise ValueError("Неизвестный статус E02.")
    if not e02.loc[native, "effective_model"].eq(DIRECT).all():
        raise ValueError("Native E02 имеет другую effective_model.")
    if not e02.loc[reserve, "effective_model"].eq("SeasonalNaive").all():
        raise ValueError("Резерв E02 должен явно обозначать SeasonalNaive.")
    if not np.isfinite(e02.loc[~failed, "y_pred"].to_numpy(dtype=float)).all():
        raise ValueError("Успешные прогнозы E02 должны быть конечными.")
    if not e02.loc[failed, "y_pred"].isna().all():
        raise ValueError("Failed E02 должен иметь y_pred=NaN.")
    result = e02.copy()
    result["model"] = DIRECT
    result.loc[reserve, "y_pred"] = np.nan
    result.loc[reserve, "status"] = "no_training_pairs"
    result.loc[reserve, "effective_model"] = "none"
    result.loc[reserve, "reason"] = "no_training_pairs; SeasonalNaive reserve excluded from pure Direct"
    return result


def _metric_tables(frame: pd.DataFrame, groups: pd.DataFrame, scope: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, per_mo = [], []
    for split, horizon in groups.itertuples(index=False, name=None):
        for model in MODELS:
            sub = frame.loc[frame.split.eq(split) & frame.horizon.eq(horizon) & frame.model.eq(model)]
            success = np.isfinite(sub.y_pred.to_numpy(dtype=float))
            n_unavailable = int(sub.status.eq("no_training_pairs").sum())
            n_failed = int((sub.status.eq("failed") | (~success & ~sub.status.eq("no_training_pairs"))).sum())
            if len(sub) and success.all():
                summary, mo = compute_metrics(sub)
                row = summary.iloc[0].to_dict()
                per_mo.append(mo.assign(scope=scope))
                status = "complete"
            else:
                row = {"split": split, "model": model, "horizon": int(horizon),
                       "n_predictions": len(sub), "n_municipalities": sub.municipality_id.nunique(),
                       "n_origins": sub.forecast_origin.nunique(),
                       "mae_macro": np.nan, "mae_micro": np.nan, "r2_pooled": np.nan}
                if n_failed:
                    status = "incomplete_failed"
                elif len(sub) and n_unavailable == len(sub):
                    status = "no_training_pairs"
                elif n_unavailable:
                    status = "incomplete_unavailable"
                else:
                    status = "no_common_success_keys" if scope == "common_success" else "no_evaluable_cases"
            rows.append(dict(row, scope=scope, metric_status=status,
                             n_success=int(success.sum()), n_failed=n_failed, n_unavailable=n_unavailable))
    empty_mo = pd.DataFrame(columns=["split", "model", "horizon", "municipality_id", "n", "mae", "r2", "scope"])
    return pd.DataFrame(rows), pd.concat(per_mo, ignore_index=True) if per_mo else empty_mo


def evaluate_chronos(current: pd.DataFrame, e01: pd.DataFrame, e02: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Соперники сохранены; полная оценка не удаляет ошибки новой модели."""
    if set(E01_MODELS) - set(e01.model.unique()):
        raise ValueError("В E01 отсутствуют обязательные соперники.")
    if e01.duplicated(KEY + ["model"]).any():
        raise ValueError("Повторяющиеся прогнозные ключи E01.")
    expected = e01.loc[e01.model.eq(E01_MODELS[0])].copy()
    for _, source in e01.groupby("model"):
        _validate_facts(source, expected)
    _validate_facts(e02, expected)
    validate_predictions(current, expected)
    pure = _pure_direct(e02)
    raw = pd.concat([current.assign(model=MODEL), pure, e01.loc[e01.model.isin(E01_MODELS)]], ignore_index=True)
    # Наличие факта задаёт область независимо от успешности любого прогноза.
    support = expected.loc[np.isfinite(expected.y_true.to_numpy(dtype=float)), KEY + ["split"]].copy()
    full = raw.merge(support[KEY], on=KEY, how="inner", validate="many_to_one")
    groups = expected[["split", "horizon"]].drop_duplicates().sort_values(["split", "horizon"])
    successful_parts, key_parts, excluded_parts, comparison_rows = [], [], [], []
    for split, horizon in groups.itertuples(index=False, name=None):
        sub = full.loc[full.split.eq(split) & full.horizon.eq(horizon)]
        group_keys = support.loc[support.split.eq(split) & support.horizon.eq(horizon)].copy()
        direct = sub.loc[sub.model.eq(DIRECT)]
        no_annual_training = bool(int(horizon) == 12 and len(direct) and direct.status.eq("no_training_pairs").all())
        participants = [MODEL, *E01_MODELS] if no_annual_training else MODELS
        successful = sub.loc[sub.model.isin(participants) & np.isfinite(sub.y_pred.to_numpy(dtype=float))]
        counts = successful.groupby(KEY, dropna=False).model.nunique()
        shared = counts.loc[counts.eq(len(participants))].reset_index()[KEY]
        common_keys = group_keys.merge(shared, on=KEY, how="inner", validate="one_to_one")
        # Сохраняем строку unavailable Direct даже на успешной годовой области.
        successful_parts.append(sub.merge(shared, on=KEY, how="inner", validate="many_to_one"))
        key_parts.append(common_keys)
        missing = group_keys.merge(shared.assign(in_common_success=True), on=KEY, how="left", validate="one_to_one")
        missing = missing.loc[missing.in_common_success.isna()].drop(columns="in_common_success")
        if len(missing):
            bad = sub.loc[sub.model.isin(participants) & ~np.isfinite(sub.y_pred.to_numpy(dtype=float))].copy()
            bad["exclusion"] = bad.model.astype(str) + ":" + bad.status.astype(str)
            if "reason" in bad:
                explanation = bad.reason.fillna("").astype(str)
                bad["exclusion"] += np.where(explanation.eq(""), "", " (" + explanation + ")")
            reasons = bad.groupby(KEY, dropna=False).exclusion.agg(lambda values: "; ".join(sorted(values))).rename("reason").reset_index()
            missing = missing.merge(reasons, on=KEY, how="left", validate="one_to_one")
            missing["reason"] = missing.reason.fillna("missing participant prediction")
        else:
            missing["reason"] = pd.Series(dtype=str)
        excluded_parts.append(missing)
        comparison_rows.append({"split": split, "horizon": int(horizon), "n_full": len(group_keys),
                                "n_common_success": len(common_keys), "n_excluded": len(missing),
                                "incomplete": bool(len(common_keys) != len(group_keys)),
                                "participants": "|".join(participants),
                                "direct_status": "no_training_pairs" if no_annual_training else "included",
                                "common_success_share": len(common_keys) / len(group_keys) if len(group_keys) else np.nan})
    common = pd.concat(successful_parts, ignore_index=True) if successful_parts else full.iloc[:0]
    common_keys = pd.concat(key_parts, ignore_index=True) if key_parts else support.iloc[:0]
    metrics_full, mo_full = _metric_tables(full, groups, "full")
    metrics_common, mo_common = _metric_tables(common, groups, "common_success")
    # Если Chronos весь failed, пустое пересечение не скрывает структурную невозможность Direct h12.
    for row in comparison_rows:
        if row["direct_status"] == "no_training_pairs":
            mask = metrics_common.split.eq(row["split"]) & metrics_common.horizon.eq(row["horizon"]) & metrics_common.model.eq(DIRECT)
            metrics_common.loc[mask, "metric_status"] = "no_training_pairs"
    coverage = raw.assign(actual_available=np.isfinite(raw.y_true.to_numpy(dtype=float)),
                          forecast_available=np.isfinite(raw.y_pred.to_numpy(dtype=float))).groupby(
        ["split", "model", "horizon"], dropna=False,
    ).agg(n_requested=("municipality_id", "size"), n_actual_available=("actual_available", "sum"),
          n_forecast_available=("forecast_available", "sum"),
          n_native=("status", lambda x: int(x.eq("native").sum())),
          n_failed=("status", lambda x: int(x.eq("failed").sum())),
          n_unavailable=("status", lambda x: int(x.eq("no_training_pairs").sum())),
          n_fallback=("status", lambda x: int(x.fillna("").str.startswith("fallback").sum()))).reset_index()
    e01_counts = support.groupby(["split", "horizon"]).size().rename("n_e01_cases").reset_index()
    coverage = coverage.merge(e01_counts, on=["split", "horizon"], how="left", validate="many_to_one")
    coverage["n_e01_cases"] = coverage.n_e01_cases.fillna(0).astype(int)
    coverage["forecast_coverage"] = coverage.n_forecast_available / coverage.n_requested
    coverage["failed_share"] = coverage.n_failed / coverage.n_requested
    coverage["fallback_share"] = coverage.n_fallback / coverage.n_requested
    by_status = raw.assign(actual_available=np.isfinite(raw.y_true.to_numpy(dtype=float)),
                           forecast_available=np.isfinite(raw.y_pred.to_numpy(dtype=float))).groupby(
        ["forecast_origin", "split", "model", "horizon", "status"], dropna=False,
    ).agg(n_requested=("municipality_id", "size"), n_actual_available=("actual_available", "sum"),
          n_forecast_available=("forecast_available", "sum")).reset_index()
    return {"metrics_full": metrics_full, "metrics_common_success": metrics_common,
            "metrics_full_by_municipality": mo_full, "metrics_common_success_by_municipality": mo_common,
            "coverage": coverage, "coverage_by_origin_status": by_status,
            "comparison_coverage": pd.DataFrame(comparison_rows),
            "evaluation_keys_full": support, "evaluation_keys_common_success": common_keys,
            "excluded_common_success_keys": pd.concat(excluded_parts, ignore_index=True) if excluded_parts else support.assign(reason="").iloc[:0]}
