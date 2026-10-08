"""Verify saved National/Local references without fitting or forecasting Persistence.

The learner gate rescores historical predictions; it is not a fresh LightGBM
fit. Only the existing parameter-free national and expense references are
recomputed. Returned data frames stay local; the audit record contains counts,
aggregate metrics and repository-relative hashes, never municipal rows.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .data import get_prefix, load_data, make_panel, sha256_file
from .direct_experiment import expected_cases, verify_reference
from .metrics import KEY
from .models import baseline_predict
from .national_local import build_national, national_forecast
from .national_local_evaluation import evaluate_national_local
from .trend_calendar_evaluation import evaluate_group

REFERENCE_MODELS = ["LastValue", "SeasonalNaiveYoY", "L0", "LN"]
METRIC_COLUMNS = {"mae_macro", "mae_micro", "r2_pooled", "mae_national"}
TOLERANCE = 1e-8
METRIC_TABLES = ("metrics_strategy", "metrics_native",
                 "metrics_strategy_by_origin", "metrics_native_by_origin")


def _path(root: Path, name: str | Path) -> Path:
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Reference path must remain inside the repository.")
    return path


def _json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"municipality_id": str})


def _verify_hashes(root: Path, hashes: dict, verified: dict) -> None:
    for name, digest in hashes.items():
        path = _path(root, name)
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"Historical reference SHA mismatch: {name}.")
        verified[path.relative_to(root).as_posix()] = digest


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _compare(actual: pd.DataFrame, saved: pd.DataFrame, keys: list[str],
             label: str, approximate: set[str] = frozenset()) -> dict:
    """Keep key/count/status checks exact, including explicit missing metrics."""
    if set(actual) != set(saved) or len(actual) != len(saved):
        raise ValueError(f"Reference table schema/row mismatch: {label}.")
    if actual.duplicated(keys).any() or saved.duplicated(keys).any():
        raise ValueError(f"Duplicate reference keys: {label}.")
    left = actual.sort_values(keys).reset_index(drop=True)
    right = saved.sort_values(keys).reset_index(drop=True)
    cells, maximum = 0, 0.0
    for column in right:
        empty_text = right[column].isna().all() and not pd.api.types.is_numeric_dtype(left[column])
        if pd.api.types.is_numeric_dtype(right[column]) and not empty_text:
            a, b = left[column].to_numpy(dtype=float), right[column].to_numpy(dtype=float)
            same = np.isclose(a, b, rtol=0, atol=TOLERANCE, equal_nan=True) if column in approximate else (
                (a == b) | (np.isnan(a) & np.isnan(b)))
            cells += len(a)
            finite = np.isfinite(a) & np.isfinite(b)
            if finite.any():
                maximum = max(maximum, float(np.abs(a[finite] - b[finite]).max()))
        else:
            same = left[column].fillna("").eq(right[column].fillna("")).to_numpy()
        if not same.all():
            raise ValueError(f"Reference table mismatch: {label}/{column}.")
    return {"rows": len(right), "numeric_cells": cells, "max_abs_delta": maximum}


def _verify_reference_tables(tables: dict, source: Path) -> dict:
    checks = {}
    for name in METRIC_TABLES:
        keys = ["split", "model", "horizon"] + (["forecast_origin"] if "by_origin" in name else [])
        checks[name] = _compare(tables[name], _csv(source / f"{name}.csv"), keys, name, METRIC_COLUMNS)
    for name in ("strategy_keys", "common_native_keys", "own_native_keys"):
        checks[name] = _compare(tables[name], _csv(source / f"{name}.csv"),
                                KEY + (["model"] if name == "own_native_keys" else []), name)
    for name in ("coverage", "coverage_evaluable", "national_metrics"):
        keys = ["split", "horizon"] + ([] if name == "national_metrics" else ["model"])
        checks[name] = _compare(tables[name], _csv(source / f"{name}.csv"), keys, name, METRIC_COLUMNS)
    return checks


def _fresh_national(national: pd.DataFrame, expected: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rows = []
    for (issue, horizon, split), _ in expected.groupby(["forecast_origin", "horizon", "split"], sort=True):
        row = national_forecast(national, pd.Period(issue, freq="M"), int(horizon),
                                cfg["models"], cfg["data"]["release_lag_months"])
        for column in ("target_period", "feature_cutoff", "target_available_at"):
            row[column] = str(pd.Timestamp(row[column]).date())
        rows.append(dict(row, split=split))
    return pd.DataFrame(rows)


def _fresh_expense_references(panel: pd.DataFrame, expected: pd.DataFrame,
                              reference: pd.DataFrame, cfg: dict) -> dict:
    rows = []
    lag = cfg["data"]["release_lag_months"]
    # The annual comparison remains an existing saved reference, not a new
    # baseline experiment. Fresh expense checks cover only primary horizons.
    for issue, cases in expected.loc[expected.horizon.isin([1, 3, 6])].groupby("forecast_origin", sort=True):
        origin = pd.Period(issue, freq="M")
        ids = cases.municipality_id.drop_duplicates().tolist()
        history = get_prefix(panel, origin, lag)[ids]
        for model in ("LastValue", "SeasonalNaiveYoY"):
            values, flags = baseline_predict(history, int(cases.horizon.max()) + lag, model, cfg["models"])
            for h, current in cases.groupby("horizon", sort=True):
                selected = [ids.index(uid) for uid in current.municipality_id]
                index = int(h) + lag - 1
                row = current[KEY].copy()
                row["model"] = model
                row["y_pred"] = values[index, selected]
                row["status"] = np.where(flags[index, selected], "fallback_last_value", "native")
                rows.append(row)
    columns = KEY + ["model", "y_pred", "status"]
    saved = reference.loc[reference.model.isin(["LastValue", "SeasonalNaiveYoY"])
                          & reference.horizon.isin([1, 3, 6]), columns]
    return _compare(pd.concat(rows, ignore_index=True), saved, KEY + ["model"],
                    "fresh_expense_references", {"y_pred"})


def load_verified_sources(root: Path, base_config_path: Path) -> dict:
    """Raise on a failed historical gate; return local inputs and public audit.

    No models are fitted, no Persistence predictions are constructed, and no
    files are written. Forecasts returned under ``predictions`` use the four
    original reference codes on every requested raw key, including NaN facts.
    """
    root = root.resolve()
    config_path = _path(root, base_config_path)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
    source = _path(root, cfg["output_dir"])
    manifest_path = source / "run_manifest.json"
    manifest = _json(manifest_path)
    resolved = yaml.safe_load((source / "config_resolved.yaml").read_text(encoding="utf-8-sig"))
    if cfg != resolved or cfg != manifest["config"] or manifest.get("complete") is not True:
        raise ValueError("Historical config/manifest is inconsistent or incomplete.")
    verified = {}
    _verify_hashes(root, {config_path.relative_to(root).as_posix(): manifest["config_sha256"],
                          cfg["data"]["path"]: manifest["data_sha256"],
                          cfg["validation_record"]: manifest["test_record_sha256"],
                          cfg["environment_record"]: manifest["environment_record_sha256"],
                          cfg["preservation_record"]: manifest["protected_snapshot_sha256"]}, verified)
    for collection in ("code_sha256", "source_e01_sha256", "source_e02_sha256"):
        _verify_hashes(root, manifest[collection], verified)
    artifact_names = {path.relative_to(source).as_posix() for path in source.rglob("*")
                      if path.is_file() and path.name != "run_manifest.json"}
    if artifact_names != set(manifest["artifact_sha256"]):
        raise ValueError("Historical artifact inventory differs from the saved manifest.")
    _verify_hashes(root, {str(source.relative_to(root) / name): digest
                          for name, digest in manifest["artifact_sha256"].items()}, verified)
    signature_fields = ("config", "config_sha256", "code_sha256", "data_sha256", "source_e01_sha256",
                        "source_e02_sha256", "test_record_sha256", "environment_record_sha256",
                        "protected_snapshot_sha256")
    signature = {field: manifest[field] for field in signature_fields}
    fingerprint = hashlib.sha256(json.dumps(signature, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if fingerprint != manifest["fingerprint"] or manifest.get("future_actual_N_used") is not False:
        raise ValueError("Historical signature or future-national usage marker is inconsistent.")
    verified[manifest_path.relative_to(root).as_posix()] = sha256_file(manifest_path)
    report_path = _path(root, cfg["report_path"])
    _verify_hashes(root, {report_path.relative_to(root).as_posix(): manifest["report_sha256"]}, verified)

    panel = make_panel(load_data(_path(root, cfg["data"]["path"]), cfg["data"]["category"]))
    sample, e01, expected, _ = verify_reference(root, panel, cfg)
    regenerated = expected_cases(panel, cfg, set(sample))
    _compare(expected, regenerated, KEY, "fresh_expected_cases")
    if _json(source / "sample_ids.json") != sample:
        raise ValueError("Historical municipality sample/order differs from the saved reference.")
    national = build_national(panel, cfg["data"]["release_lag_months"])
    fresh_series = national.reset_index()
    for column in ("ds", "available_at"):
        fresh_series[column] = pd.to_datetime(fresh_series[column]).dt.strftime("%Y-%m-%d")
    series_check = _compare(fresh_series, _csv(source / "national_series.csv"), ["ds"],
                            "national_series", set(fresh_series.select_dtypes("number")) - {
                                "n_municipalities_used", "n_municipalities_known", "n_municipalities_available",
                                "n_missing", "release_lag_months"})
    fresh_national = _fresh_national(national, expected, cfg)
    saved_national = _csv(source / "national_forecasts.csv")
    national_check = _compare(fresh_national, saved_national, ["forecast_origin", "target_period", "horizon"],
                              "national_forecasts", {"N_hat", "N_actual", "n_hat", "n_actual", "absolute_error"})
    saved = _csv(source / "predictions.csv.gz")
    tables = evaluate_national_local(saved, expected, saved_national)
    table_checks = _verify_reference_tables(tables, source)
    expense_check = _fresh_expense_references(panel, expected, e01, cfg)
    common_columns = KEY + ["y_true", "y_pred", "history_cutoff", "split", "availability_assumption", "status"]
    _compare(saved.loc[saved.model.eq("SeasonalNaiveYoY"), common_columns],
             e01.loc[e01.model.eq("SeasonalNaiveYoY"), common_columns], KEY,
             "saved_yearly_reference", {"y_pred"})
    last = e01.loc[e01.model.eq("LastValue")].copy()
    last["effective_model"] = "LastValue"
    last["reason"] = np.where(last.status.eq("native"), "", "baseline_internal_fallback")
    predictions = pd.concat([last, saved.loc[saved.model.isin(REFERENCE_MODELS)]], ignore_index=True)
    four = evaluate_group(predictions, expected, REFERENCE_MODELS, ["L0", "LN"])
    gate = _json_safe({
        "status": "PASS", "verification_mode": "saved_prediction_rescoring_not_fresh_LightGBM_fit",
        "model_fits": 0, "persistence_predictions_computed": False,
        "reference_models": REFERENCE_MODELS, "historical_git_commit": manifest["git_commit"],
        "historical_seed": cfg["seed"], "independent_blind_test": False,
        "expense_publication_dates_verified": False,
        "tolerance_absolute": TOLERANCE, "counts_and_statuses_exact": True,
        "sample_size": len(sample), "raw_requested_keys": len(expected),
        "finite_target_keys": len(tables["strategy_keys"]),
        "original_common_native_keys": len(tables["common_native_keys"]),
        "historical_artifacts_verified": len(artifact_names), "source_hashes_verified": len(verified),
        "saved_metric_checks": table_checks, "fresh_national_series_check": series_check,
        "fresh_national_forecast_check": national_check,
        "fresh_expense_reference_check": dict(expense_check, horizons=[1, 3, 6]),
        "annual_reference": "saved_only_one_origin_existing_expense_SeasonalNaive_fallback_no_native_L0_LN",
        "reference_strategy_metrics": four["metrics_strategy"].to_dict("records"),
        "reference_strategy_metrics_by_origin": four["metrics_strategy_by_origin"].to_dict("records"),
        "source_hashes": verified,
    })
    return {"cfg": cfg, "panel": panel, "national": national, "expected": expected,
            "predictions": predictions, "gate": gate, "source_hashes": verified}
