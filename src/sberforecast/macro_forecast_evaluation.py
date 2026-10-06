"""E05c: fixed keys, explicit reserves/errors and a shared macro-available scope.

The existing E05b evaluator supplies the full strategy and common native
regions. A third region additionally requires genuinely available predictors
for both macro variants; the control and all references are rescored on exactly
those keys. National predictors do not create independent observations per MO.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import KEY
from .trend_calendar_evaluation import GROUP, _metric_tables, evaluate_group

MODELS = ["M0", "M1", "M2", "SeasonalNaiveYoY", "ProphetAuto", "ProphetYearly"]
NATIVE_MODELS = ["M0", "M1", "M2"]
MACRO_MODELS = ["M1", "M2"]
SCOPES = ("strategy", "native", "macro_available")
MACRO_FLAG = "macro_features_available"


def _macro_flags(predictions: pd.DataFrame) -> pd.DataFrame:
    if MACRO_FLAG not in predictions:
        raise ValueError("Для M1/M2 нужен явный macro_features_available.")
    result = predictions.copy()
    selected = result.model.isin(MACRO_MODELS)
    values = result.loc[selected, MACRO_FLAG]
    # Strings such as 'False' are truthy in Python and must never promote a
    # missing forecast to the macro-available comparison region.
    valid = values.map(lambda value: isinstance(value, (bool, np.bool_, int, np.integer, float, np.floating))
                       and pd.notna(value) and value in (0, 1))
    if not valid.all():
        raise ValueError("macro_features_available у M1/M2 должен быть bool либо 0/1 без NaN.")
    result["_macro_available"] = False
    result.loc[selected, "_macro_available"] = values.astype(bool)
    return result


def _macro_coverage(raw: pd.DataFrame, support: pd.DataFrame) -> pd.DataFrame:
    evaluable = raw.merge(support[KEY], on=KEY, how="inner", validate="many_to_one")
    groups = raw[GROUP].drop_duplicates().sort_values(GROUP)
    records = []
    for split, model, horizon in groups.itertuples(index=False, name=None):
        selected = raw.split.eq(split) & raw.model.eq(model) & raw.horizon.eq(horizon)
        selected_evaluable = evaluable.split.eq(split) & evaluable.model.eq(model) & evaluable.horizon.eq(horizon)
        row = {"split": split, "model": model, "horizon": horizon,
               "macro_applicable": model in MACRO_MODELS}
        for frame, mask, suffix in ((raw, selected, ""), (evaluable, selected_evaluable, "_evaluable")):
            current = frame.loc[mask]
            native = current.status.eq("native")
            fallback = current.status.astype(str).str.startswith("fallback")
            available = current["_macro_available"]
            applicable = model in MACRO_MODELS
            counts = {
                "n_requested": len(current),
                "n_macro_available": int(available.sum()),
                "n_macro_missing": int((~available).sum()) if applicable else 0,
                "n_native_with_macro": int((native & available).sum()),
                "n_native_without_macro": int((native & ~available).sum()) if applicable else 0,
                "n_fallback_with_macro": int((fallback & available).sum()),
                "n_fallback_without_macro": int((fallback & ~available).sum()) if applicable else 0,
                "n_failed_with_macro": int((current.status.eq("failed") & available).sum()),
                "n_failed_without_macro": int((current.status.eq("failed") & ~available).sum()) if applicable else 0,
            }
            for name, value in counts.items():
                row[name + suffix] = value
            row["macro_available_share" + suffix] = (
                float(available.mean()) if applicable and len(current) else np.nan)
        row["n_evaluable"] = row.pop("n_requested_evaluable")
        records.append(row)
    return pd.DataFrame(records)


def _deltas(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    records = []
    for scope in SCOPES:
        metrics = tables[f"metrics_{scope}"]
        for split, horizon in metrics[["split", "horizon"]].drop_duplicates().itertuples(index=False, name=None):
            group = metrics.loc[metrics.split.eq(split) & metrics.horizon.eq(horizon)].set_index("model")
            for earlier, later in (("M0", "M1"), ("M1", "M2")):
                left, right = group.loc[earlier], group.loc[later]
                counts = ("n_predictions", "n_municipalities", "n_origins")
                if any(left[field] != right[field] for field in counts):
                    raise ValueError("Разности метрик требуют одинаковой области и числа прогнозных случаев.")
                complete = left.metric_status == right.metric_status == "complete"
                state = "complete" if complete else "incomplete_failed" if "incomplete_failed" in (
                    left.metric_status, right.metric_status) else right.metric_status
                record = {"scope": scope, "split": split, "horizon": int(horizon),
                          "comparison": f"{later}-{earlier}", "from_model": earlier, "to_model": later,
                          "n_predictions": int(right.n_predictions),
                          "n_municipalities": int(right.n_municipalities), "n_origins": int(right.n_origins),
                          "metric_status": state}
                for metric in ("mae_macro", "mae_micro", "r2_pooled"):
                    record["delta_" + metric] = float(right[metric] - left[metric]) if complete else np.nan
                records.append(record)
    return pd.DataFrame(records)


def evaluate_macro(predictions: pd.DataFrame, expected: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return all A/B tables plus paired macro coverage, C metrics and deltas.

    ``macro_features_available`` means that every indicator required by a
    particular variant has a finite available forecast for its target year.
    It is recorded independently of native/fallback/failed execution. A macro
    publication cannot create annual training labels or make a reserve native.
    """
    tables = evaluate_group(predictions, expected, MODELS, NATIVE_MODELS)
    raw = _macro_flags(predictions)
    support = tables["strategy_keys"]
    macro_rows = raw.loc[raw.model.isin(MACRO_MODELS) & raw["_macro_available"]]
    counts = macro_rows.groupby(KEY, dropna=False).model.nunique()
    shared_macro = counts.loc[counts.eq(len(MACRO_MODELS))].reset_index()[KEY]
    keys = tables["common_native_keys"].merge(shared_macro, on=KEY, how="inner", validate="one_to_one")
    common = raw.merge(keys[KEY], on=KEY, how="inner", validate="many_to_one")
    groups = expected[["split", "horizon"]].drop_duplicates().sort_values(["split", "horizon"])
    origin_groups = expected[["forecast_origin", "split", "horizon"]].drop_duplicates().sort_values(
        ["forecast_origin", "split", "horizon"])
    metrics, by_mo = _metric_tables(common, groups, MODELS, "macro_available")
    by_origin, _ = _metric_tables(common, origin_groups, MODELS, "macro_available", by_origin=True)
    for table in (metrics, by_origin):
        table.loc[table.n_predictions.eq(0), "metric_status"] = "no_macro_available_native_forecasts"
    excluded = support.merge(keys[KEY].assign(in_macro_available=True), on=KEY,
                             how="left", validate="one_to_one")
    excluded = excluded.loc[excluded.in_macro_available.isna()].drop(columns="in_macro_available")
    reasons = tables["excluded_native_keys"].rename(columns={"reason": "native_reason"})
    excluded = excluded.merge(reasons[KEY + ["native_reason"]], on=KEY, how="left", validate="one_to_one")
    missing = raw.loc[raw.model.isin(MACRO_MODELS) & ~raw["_macro_available"]].copy()
    if len(missing):
        missing["macro_reason"] = missing.model.astype(str) + ":missing_macro_forecast_for_target_year"
        explanation = missing.groupby(KEY, dropna=False).macro_reason.agg(
            lambda values: "; ".join(sorted(values))).reset_index()
        excluded = excluded.merge(explanation, on=KEY, how="left", validate="one_to_one")
    else:
        excluded["macro_reason"] = ""
    excluded["reason"] = ["; ".join(value for value in (str(native), str(macro)) if value and value != "nan")
                          for native, macro in zip(excluded.native_reason, excluded.macro_reason)]
    if len(excluded) and excluded.reason.eq("").any():
        raise RuntimeError("Нет причины исключения из области с макропризнаками.")
    excluded = excluded.drop(columns=["native_reason", "macro_reason"])
    tables.update({
        "metrics_macro_available": metrics,
        "metrics_macro_available_by_municipality": by_mo,
        "metrics_macro_available_by_origin": by_origin,
        "macro_available_keys": keys,
        "excluded_macro_available_keys": excluded,
        "macro_coverage_by_model": _macro_coverage(raw, support),
    })
    tables["metric_deltas"] = _deltas(tables)
    return tables
