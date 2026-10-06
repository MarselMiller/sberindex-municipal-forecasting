"""MAE в исходных рублях; единая область сравнения и отдельный учёт покрытия."""
from __future__ import annotations
import numpy as np
import pandas as pd

KEY = ["municipality_id", "forecast_origin", "target_period", "horizon"]


def safe_r2(y: np.ndarray, p: np.ndarray) -> float:
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    if len(y) < 2:
        return float("nan")
    denom = np.sum((y-y.mean())**2)
    return float(1-np.sum((y-p)**2)/denom) if denom > 0 else float("nan")


def attach_split(predictions: pd.DataFrame, validation_end: str) -> pd.DataFrame:
    df = predictions.copy()
    periods = pd.to_datetime(df.target_period).dt.to_period("M")
    df["split"] = np.where(periods <= pd.Period(validation_end, freq="M"), "validation", "holdout")
    return df


def common_support(predictions: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    if predictions.duplicated(KEY+["model"]).any():
        raise ValueError("Повторный прогноз по одному ключу и модели.")
    df = predictions[predictions.model.isin(models)].copy()
    df["valid"] = np.isfinite(df.y_true) & np.isfinite(df.y_pred)
    counts = df.groupby(KEY, observed=True).agg(n_models=("model", "nunique"), n_valid=("valid", "sum"))
    keys = counts.loc[(counts.n_models == len(models)) & (counts.n_valid == len(models))].reset_index()[KEY]
    return df.merge(keys, on=KEY, how="inner", validate="many_to_one").drop(columns="valid")


def compute_metrics(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_mo, rows = [], []
    for (split, model, horizon), g in df.groupby(["split", "model", "horizon"], sort=True):
        by_mo = []
        for uid, sub in g.groupby("municipality_id", sort=False):
            mae = float(np.abs(sub.y_true-sub.y_pred).mean())
            mo_r2 = safe_r2(sub.y_true.to_numpy(), sub.y_pred.to_numpy())
            by_mo.append(mae)
            per_mo.append({"split": split, "model": model, "horizon": int(horizon),
                           "municipality_id": uid, "n": len(sub), "mae": mae, "r2": mo_r2})
        rows.append({"split": split, "model": model, "horizon": int(horizon),
                     "n_predictions": len(g), "n_municipalities": len(by_mo),
                     "n_origins": g.forecast_origin.nunique(), "mae_macro": float(np.mean(by_mo)),
                     "mae_micro": float(np.abs(g.y_true-g.y_pred).mean()),
                     "r2_pooled": safe_r2(g.y_true.to_numpy(), g.y_pred.to_numpy())})
    return pd.DataFrame(rows), pd.DataFrame(per_mo)


def coverage_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, g in df.groupby(["split", "model", "horizon"]):
        finite_pred = np.isfinite(g.y_pred)
        rows.append(dict(zip(["split", "model", "horizon"], keys),
                         n_requested=len(g), n_actual_available=int(np.isfinite(g.y_true).sum()),
                         n_forecast_available=int(finite_pred.sum()),
                         n_failed=int((~finite_pred).sum()),
                         n_fallback=int(g.status.str.startswith("fallback").sum()),
                         forecast_coverage=float(finite_pred.mean())))
    return pd.DataFrame(rows)


def select_on_validation(metrics: pd.DataFrame, cfg: dict) -> dict:
    result = {}
    for h in cfg["backtest"]["horizons"]:
        available = metrics[(metrics.split == "validation") & (metrics.horizon == h)]
        if available.empty:
            result[str(h)] = {"model": cfg["models"]["unvalidated_horizon_fallback"],
                              "selection": "predeclared_fallback_no_validation", "validation_mae": None,
                              "selection_available_at": str(pd.Period(cfg["backtest"]["first_origin"], freq="M").to_timestamp(how="end").normalize().date())}
        else:
            best = available.sort_values(["mae_macro", "model"]).iloc[0]
            result[str(h)] = {"model": best.model, "selection": "validation_macro_MAE",
                              "validation_mae": float(best.mae_macro),
                              "validation_origins": int(best.n_origins),
                              "selection_available_at": str((pd.Period(cfg["backtest"]["validation_target_end"], freq="M") + cfg["data"]["release_lag_months"]).to_timestamp(how="end").normalize().date())}
    return result


def compare_with_baseline(per_mo: pd.DataFrame, baseline: str) -> pd.DataFrame:
    base = per_mo[per_mo.model.eq(baseline)][["split", "horizon", "municipality_id", "mae"]].rename(columns={"mae":"baseline_mae"})
    pairs = per_mo.merge(base, on=["split", "horizon", "municipality_id"], validate="many_to_one")
    rows = []
    for keys, g in pairs.groupby(["split", "model", "horizon"]):
        denom = g.baseline_mae.mean()
        rows.append(dict(zip(["split", "model", "horizon"], keys), baseline=baseline,
                         improvement_macro_pct=float(100*(denom-g.mae.mean())/denom) if denom else np.nan,
                         share_municipalities_better=float((g.mae < g.baseline_mae).mean()),
                         n_municipalities=len(g)))
    return pd.DataFrame(rows)


def selected_policy_predictions(df: pd.DataFrame, selection: dict) -> pd.DataFrame:
    """Нельзя применять выбор июня к прогнозу, выпущенному ещё в январе."""
    parts = []
    for h, item in selection.items():
        valid = ((df.split == "holdout") & (df.horizon == int(h)) &
                 (df.model == item["model"]) &
                 (pd.to_datetime(df.forecast_origin) >= pd.Timestamp(item["selection_available_at"])))
        parts.append(df.loc[valid].copy())
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
