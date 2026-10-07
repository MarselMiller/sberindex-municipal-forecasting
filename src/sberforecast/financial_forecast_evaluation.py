"""Fixed E08c paired evaluation, with origins as the units of consistency.

No model selection or parameter fitting takes place here. Strategy metrics
reuse the existing evaluator, including failed rows and annual fallbacks.
Relative deltas are fractions; the explicitly named pct column is percent.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .trend_calendar_evaluation import evaluate_group

MODEL_FAMILIES = ("LightGBMDirectNationalLocal", "LightGBMDirect")
VARIANTS = ("F0", "F1", "F2", "F3")
PRIMARY_HORIZONS = (1, 3, 6)
EPSILON_RUB = 1e-8


def _with_deltas(table: pd.DataFrame, *, by_origin: bool = False) -> pd.DataFrame:
    fields = ["model", "split", "horizon"]
    if by_origin:
        fields.append("forecast_origin")
    metrics = ["mae_macro", "mae_micro", "r2_pooled"]
    counts = ["n_predictions", "n_municipalities", "n_origins"]
    baseline = table.loc[table.variant.eq("F0"), fields + metrics + counts + ["metric_status"]]
    baseline = baseline.rename(columns={name: name + "_F0" for name in metrics + counts + ["metric_status"]})
    out = table.merge(baseline, on=fields, how="left", validate="many_to_one")
    if any(not out[name].equals(out[name + "_F0"]) for name in counts):
        raise ValueError("Ablations require identical forecast cases and counts versus F0")
    complete = out.metric_status.eq("complete") & out.metric_status_F0.eq("complete")
    for name in metrics:
        out["delta_" + name + "_vs_F0"] = (out[name] - out[name + "_F0"]).where(complete)
    denominator = out.mae_macro_F0.where(out.mae_macro_F0.ne(0))
    out["relative_delta_mae_macro_vs_F0"] = out.delta_mae_macro_vs_F0 / denominator
    out["relative_delta_mae_macro_pct_vs_F0"] = 100 * out.relative_delta_mae_macro_vs_F0
    out["delta_metric_status"] = np.where(complete, "complete", "incomplete")
    out["evaluation_role"] = np.where(out.horizon.eq(12), "DESCRIPTIVE ONLY", "PRIMARY")
    if by_origin:
        out["origin_mae"] = out.mae_macro
        out["origin_mae_F0"] = out.mae_macro_F0
    return out


def _consistency(origins: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for values, group in origins.groupby(["model", "variant", "split", "horizon"], sort=True):
        fields = dict(zip(("model", "variant", "split", "horizon"), values))
        valid = group.loc[group.delta_metric_status.eq("complete") & np.isfinite(group.delta_mae_macro_vs_F0)]
        ordered = valid.sort_values("forecast_origin", kind="stable")
        delta = ordered.delta_mae_macro_vs_F0
        gains = ordered.loc[delta.lt(-EPSILON_RUB)]
        losses = ordered.loc[delta.gt(EPSILON_RUB)]
        best = ordered.loc[delta.idxmin()] if len(ordered) else None
        worst = ordered.loc[delta.idxmax()] if len(ordered) else None
        leave_best = delta.drop(best.name) if best is not None else delta
        leave_best_mean = float(leave_best.mean()) if len(leave_best) else np.nan
        gain_sum = float(-gains.delta_mae_macro_vs_F0.sum())
        if len(valid) != len(group):
            concentration_status, concentration_pass = "INCOMPLETE", False
        elif len(valid) < 2:
            concentration_status, concentration_pass = "SINGLE ORIGIN UNDETERMINED", None
        else:
            concentration_pass = bool(len(gains) >= 2 and leave_best_mean < -EPSILON_RUB)
            concentration_status = "PASS" if concentration_pass else "FAIL"
        rows.append(dict(
            **fields, n_origins_requested=len(group), n_origins_evaluable=len(valid),
            n_origins_incomplete=len(group) - len(valid),
            n_origins_improved=len(gains), n_origins_worsened=len(losses),
            n_origins_tied=int(delta.abs().le(EPSILON_RUB).sum()),
            mean_delta_mae_macro=float(delta.mean()) if len(delta) else np.nan,
            median_delta_mae_macro=float(delta.median()) if len(delta) else np.nan,
            best_origin=str(best.forecast_origin) if best is not None else None,
            best_origin_delta=float(best.delta_mae_macro_vs_F0) if best is not None else np.nan,
            worst_origin=str(worst.forecast_origin) if worst is not None else None,
            worst_origin_delta=float(worst.delta_mae_macro_vs_F0) if worst is not None else np.nan,
            largest_improvement_share=(float(-gains.delta_mae_macro_vs_F0.min()) / gain_sum if gain_sum else np.nan),
            leave_best_origin_out_mean_delta=leave_best_mean,
            concentration_status=concentration_status, concentration_pass=concentration_pass,
            evaluation_role="DESCRIPTIVE ONLY" if int(fields["horizon"]) == 12 else "PRIMARY",
            interpretation="descriptive_origin_blocks_no_significance_test_no_model_selection",
        ))
    return pd.DataFrame(rows)


def _success(metrics: pd.DataFrame, consistency: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family in MODEL_FAMILIES:
        selected = metrics.loc[metrics.model.eq(family) & metrics.variant.eq("F3")]
        if selected.empty:
            continue
        qualifying, reasons, undetermined = [], [], []
        for horizon in PRIMARY_HORIZONS:
            current = selected.loc[selected.horizon.eq(horizon)]
            paired = all(
                len(part := current.loc[current.split.eq(split)]) == 1
                and part.delta_metric_status.iloc[0] == "complete"
                and part.delta_mae_macro_vs_F0.iloc[0] < -EPSILON_RUB
                for split in ("validation", "holdout")
            )
            if paired:
                qualifying.append(horizon)
        concentration = True
        for horizon in qualifying:
            for split in ("validation", "holdout"):
                part = consistency.loc[
                    consistency.model.eq(family) & consistency.variant.eq("F3")
                    & consistency.horizon.eq(horizon) & consistency.split.eq(split)]
                if len(part) != 1:
                    concentration = False
                    reasons.append(f"h={horizon}/{split}: missing origin consistency")
                    continue
                record = part.iloc[0]
                # The inherited validation h=6 has one origin; report its
                # concentration as unknown, never as proven consistency.
                if (split == "validation" and horizon == 6
                        and record.concentration_status == "SINGLE ORIGIN UNDETERMINED"):
                    undetermined.append("validation/h=6")
                elif record.concentration_status != "PASS":
                    concentration = False
                    reasons.append(f"h={horizon}/{split}: concentration {record.concentration_status}")
        if len(qualifying) < 2:
            reasons.insert(0, "F3 must improve both validation and holdout at >=2 of h=1/3/6")
        promising = len(qualifying) >= 2 and concentration
        rows.append(dict(
            model=family, variant="F3",
            conclusion="PROMISING" if promising else "NO STABLE FORECASTING UPLIFT",
            qualifying_horizons=json.dumps(qualifying), n_qualifying_horizons=len(qualifying),
            concentration_pass=bool(qualifying and concentration),
            single_origin_concentration_undetermined=json.dumps(undetermined),
            reasons=json.dumps(reasons), epsilon_rub=EPSILON_RUB,
            h12_used_for_success=False,
            family_role="PRIMARY" if family == MODEL_FAMILIES[0] else "SECONDARY SENSITIVITY",
        ))
    return pd.DataFrame(rows)


def evaluate_financial_forecast(predictions: pd.DataFrame, expected: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Evaluate F0--F3 within each fixed family on exactly the expected keys.

    Every supplied family requires all four variants. Missing truth remains in
    raw coverage; failed predictions remain in the evaluable strategy scope.
    Native-only metrics are secondary and never drive the success criterion.
    """
    if predictions.empty or not {"model", "variant"}.issubset(predictions):
        raise ValueError("Financial forecasts require nonempty model/variant columns")
    families = set(predictions.model.unique())
    if not families.issubset(MODEL_FAMILIES):
        raise ValueError("Only the two fixed LightGBM families are permitted")
    if "y_true" not in expected or np.isinf(pd.to_numeric(expected.y_true, errors="raise")).any():
        raise ValueError("Expected truth must contain finite values or NaN, not infinity")
    collected = {name: [] for name in (
        "forecasting_metrics", "origin_deltas", "coverage", "coverage_evaluable",
        "metrics_native", "metrics_by_municipality", "strategy_keys", "common_native_keys",
    )}
    for family in MODEL_FAMILIES:
        frame = predictions.loc[predictions.model.eq(family)].copy()
        if frame.empty:
            continue
        if set(frame.variant.unique()) != set(VARIANTS):
            raise ValueError("Each family requires all four financial variants F0/F1/F2/F3")
        if np.isinf(pd.to_numeric(frame.y_true, errors="raise")).any():
            raise ValueError("Forecast truth cannot contain infinity")
        temporary = frame.assign(model=frame.variant)
        tables = evaluate_group(temporary, expected, list(VARIANTS), list(VARIANTS))
        mapping = {
            "forecasting_metrics": "metrics_strategy", "origin_deltas": "metrics_strategy_by_origin",
            "coverage": "coverage", "coverage_evaluable": "coverage_evaluable",
            "metrics_native": "metrics_native", "metrics_by_municipality": "metrics_strategy_by_municipality",
            "strategy_keys": "strategy_keys", "common_native_keys": "common_native_keys",
        }
        for name, source in mapping.items():
            table = tables[source].rename(columns={"model": "variant"}).copy()
            table.insert(0, "model", family)
            collected[name].append(table)
    result = {name: pd.concat(parts, ignore_index=True) for name, parts in collected.items()}
    result["forecasting_metrics"] = _with_deltas(result["forecasting_metrics"])
    result["origin_deltas"] = _with_deltas(result["origin_deltas"], by_origin=True)
    result["origin_consistency"] = _consistency(result["origin_deltas"])
    result["success_criterion"] = _success(result["forecasting_metrics"], result["origin_consistency"])
    return result
