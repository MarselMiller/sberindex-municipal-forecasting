"""Thin E05b orchestration over the unchanged E01/E02 preparation and models."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import yaml

from .backtest import origins_from_config
from .calendar_models import CalendarCatBoostDirect, training_signature, transform_month_features
from .data import eligibility, get_prefix, load_data, make_panel, period_end, sha256_file
from .direct_evaluation import require_same_keys, validate_direct_against_reference
from .direct_experiment import project_path, save_json, verify_reference
from .direct_model import TemporalIntegrityError
from .direct_training import build_direct_training
from .metrics import KEY
from .seasonal_trend import TREND_VARIANTS, predict_seasonal_trend
from .trend_calendar_evaluation import evaluate_group

TREND_MODELS = ["SeasonalNaiveYoY", *TREND_VARIANTS]
CALENDAR_MODELS = ["CatBoostDirectK0", "CatBoostDirectK1", "CatBoostDirectK2"]


def tested_files(root: Path) -> dict[str, str]:
    paths = [root / "src/sberforecast" / name for name in (
        "calendar_models.py", "seasonal_trend.py", "trend_calendar_evaluation.py", "trend_calendar_experiment.py",
        "trend_calendar_report.py")]
    paths += [root / "scripts/run_trend_calendar.py", root / "configs/trend_calendar.yaml"]
    paths += sorted((root / "tests").glob("test_*calendar*.py")) + [root / "tests/test_seasonal_trend.py"]
    return {str(path.relative_to(root)): sha256_file(path) for path in paths}


def validate_specification(cfg: dict) -> None:
    fixed = {"variants": list(TREND_VARIANTS), "annual_difference_window_months": 6,
             "min_annual_pairs": 3, "seasonal_template_months": 12, "phi": .9,
             "fallback": "SeasonalNaiveYoY"}
    if cfg["trends"] != fixed:
        raise ValueError("E05b trend specification differs from the fixed E05a formulas.")
    calendar = cfg["calendar"]
    if (calendar["saved_reference"] != "K0" or calendar["train_encodings"] != ["K1", "K2"]
            or calendar["cat_features"] != ["month_category"] or calendar["one_hot_max_size"] != 12):
        raise ValueError("E05b month encodings are fixed K0/K1/K2.")
    if cfg["macro_features_enabled"] is not False or cfg["holdout_status"] != "previously_inspected_not_independent":
        raise ValueError("No macro factors or independent-test claim are permitted in E05b.")


def verify_saved_direct(root: Path, cfg: dict, reference: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    source = project_path(root, cfg["comparison"]["direct_source_dir"])
    manifest = json.loads((source / "run_manifest.json").read_text(encoding="utf-8"))
    old = yaml.safe_load((source / "config_resolved.yaml").read_text(encoding="utf-8"))
    status = json.loads((source / "run_status.json").read_text(encoding="utf-8"))
    tracked_config = yaml.safe_load((root / "configs/catboost_direct.yaml").read_text(encoding="utf-8"))
    if old != manifest["config"] or old != tracked_config or not status["complete"] or status["n_failed"]:
        raise ValueError("Saved E02 experiment is inconsistent or incomplete.")
    direct_ids = json.loads((source / "sample_ids.json").read_text(encoding="utf-8"))
    expected_ids = json.loads(project_path(root, cfg["comparison"]["sample_ids_path"]).read_text(encoding="utf-8"))
    if len(direct_ids) != 64 or len(set(direct_ids)) != 64 or set(direct_ids) != set(expected_ids):
        raise ValueError("E02 and E01 municipality identifiers differ.")
    for field in ("data", "backtest", "models", "seed", "direct_training"):
        if cfg[field] != old[field]:
            raise ValueError(f"E05b changes E02 protocol/parameters: {field}.")
    if sha256_file(project_path(root, cfg["data"]["path"])) != manifest["data_sha256"]:
        raise ValueError("Data differ from saved E02.")
    for collection in ("code_sha256", "source_e01_sha256"):
        for path, expected in manifest[collection].items():
            if sha256_file(project_path(root, path)) != expected:
                raise ValueError(f"Saved E02 provenance changed: {path}.")
    direct = pd.read_csv(source / "predictions.csv.gz", dtype={"municipality_id": str})
    validate_direct_against_reference(direct, reference)
    diagnostics = pd.read_csv(source / "training_diagnostics.csv")
    annual = diagnostics.loc[diagnostics.horizon.eq(12)]
    if len(annual) != 1 or annual.n_training_rows.ne(0).any() or annual.fit_called.any():
        raise ValueError("E02 annual fallback has been confused with a fitted model.")
    hashes = {str(path.relative_to(root)): sha256_file(path) for path in source.rglob("*") if path.is_file()}
    return direct, diagnostics, hashes


def require_test_record(root: Path, cfg: dict) -> dict:
    record = json.loads(project_path(root, cfg["validation_record"]).read_text(encoding="utf-8"))
    if record["exit_code"] != 0 or record.get("full_pytest") is not True or record["code_sha256"] != tested_files(root):
        raise ValueError("A successful full pytest for the current E05b files is required before training.")
    return record


def forecast_trend(panel: pd.DataFrame, origin: pd.Period, cases: pd.DataFrame,
                   name: str, cfg: dict) -> tuple[pd.DataFrame, list[dict]]:
    h = int(cases.horizon.iloc[0])
    ids = cases.municipality_id.tolist()
    try:
        values, diag = predict_seasonal_trend(panel[ids], origin, h,
            release_lag_months=cfg["data"]["release_lag_months"], variant=name, phi=cfg["trends"]["phi"],
            yearly_growth_window=cfg["models"]["yearly_growth_window"],
            yearly_growth_bounds=tuple(cfg["models"]["yearly_growth_bounds"]))
        diag = diag.drop(columns=["forecast_origin", "target_period", "horizon"])
        diag["y_pred"] = values
        diag.loc[diag.status.eq("fallback"), "status"] = "fallback_history"
        forecast = cases.merge(diag, on="municipality_id", how="left", validate="one_to_one")
        forecast["model"] = name
        return forecast, []
    except Exception as exc:
        error = {"forecast_origin": str(period_end(origin).date()), "horizon": h,
                 "model": name, "stage": "trend", "error": repr(exc), "municipality_ids": ids}
        return cases.assign(model=name, y_pred=np.nan, status="failed", effective_model="none",
                            reason=repr(exc), failure_stage="trend"), [error]


def check_training_match(signature: dict, training: dict, old: pd.Series,
                         raw_X: pd.DataFrame, encoding: str) -> None:
    for field in ("training_key_sha256", "training_delta_sha256", "training_raw_X_sha256"):
        if signature[field] != training[field]:
            raise TemporalIntegrityError(f"Training composition differs between K0 and {encoding}: {field}.")
    for field in ("n_training_rows", "n_training_municipalities", "n_historical_origins", "n_target_periods"):
        if old[field] != training[field]:
            raise TemporalIntegrityError(f"Training count differs from saved K0: {field}.")
    expected = transform_month_features(raw_X, encoding)
    if training["feature_names"] != expected.columns.tolist() or training["feature_dtypes"] != {col: str(dtype) for col, dtype in expected.dtypes.items()}:
        raise TemporalIntegrityError("Actual month feature schema differs from specification.")
    if training["fit_succeeded"]:
        actual = training["fit_actual_parameters"]
        if actual["one_hot_max_size"] != 12 or training["fit_actual_cat_feature_indices"] != [expected.columns.get_loc("month_category")]:
            raise RuntimeError("CatBoost did not apply categorical month and one_hot_max_size=12.")


def run_experiment(root: Path, config_path: Path, requested_origins: list[str] | None, command: str) -> dict:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_specification(cfg)
    tests = require_test_record(root, cfg)
    data_path = project_path(root, cfg["data"]["path"])
    panel = make_panel(load_data(data_path, cfg["data"]["category"]))
    sample, reference, expected, e01_hashes = verify_reference(root, panel, cfg)
    e02, e02_training, e02_hashes = verify_saved_direct(root, cfg, reference)
    output = project_path(root, cfg["output_dir"])
    if output != root / "outputs/trend_calendar_v1":
        raise ValueError("E05b must use its own outputs/trend_calendar_v1 directory.")
    all_origins = origins_from_config(cfg)
    selected = [pd.Period(value, freq="M") for value in requested_origins] if requested_origins else all_origins
    if not selected or len(set(selected)) != len(selected) or any(origin not in all_origins for origin in selected):
        raise ValueError("Duplicate or out-of-protocol forecast origin.")
    signature = {"config": cfg, "config_sha256": sha256_file(config_path), "code_sha256": tested_files(root),
                 "data_sha256": sha256_file(data_path), "source_e01_sha256": e01_hashes,
                 "source_e02_sha256": e02_hashes, "test_record_sha256": sha256_file(project_path(root, cfg["validation_record"]))}
    fingerprint = hashlib.sha256(json.dumps(signature, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    manifest_path = output / "run_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["fingerprint"] != fingerprint:
            raise ValueError("Code/configuration/source/test record changed: resume is prohibited.")
        for name, expected_hash in manifest.get("artifact_sha256", {}).items():
            artifact = (output / name).resolve()
            if not artifact.is_relative_to(output) or not artifact.is_file() or sha256_file(artifact) != expected_hash:
                raise ValueError(f"Saved E05b artifact is missing or changed: {name}.")
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("Output directory has no matching E05b manifest.")
        output.mkdir(parents=True, exist_ok=True)
        manifest = dict(signature, fingerprint=fingerprint, seed=cfg["seed"], python=platform.python_version(), python_executable=sys.executable,
            platform=platform.platform(), versions={d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
            git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            git_status=subprocess.check_output(["git", "status", "--short"], cwd=root, text=True).splitlines(),
            started_at=datetime.now(timezone.utc).isoformat(), run_commands=[], tests=tests,
            macro_features_enabled=False, independent_blind_test=False, source_models_refitted=False,
            training_scope="all available municipalities, unchanged legacy E02",
            publication_dates_verified=False)
        manifest["has_uncommitted_changes"] = bool(manifest["git_status"])
        save_json(output / "sample_ids.json", sample)
        (output / "config_resolved.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
        save_json(output / "preflight.json", {"e01_e02_data_protocol_keys_targets_verified": True,
            "sample_size": len(sample), "municipality_1471_not_preexcluded": "1471" in sample,
            "n_raw_keys": len(expected), "n_evaluable_keys": int(np.isfinite(expected.y_true).sum())})
    manifest["run_commands"].append(command)
    save_json(manifest_path, manifest)
    partitions = output / "partitions"
    partitions.mkdir(exist_ok=True)
    predictors = {encoding: CalendarCatBoostDirect(cfg["models"]["catboost"], cfg["seed"], encoding,
        cfg["data"]["release_lag_months"], cfg["direct_training"]["mode"], cfg["direct_training"]["max_staleness_months"])
        for encoding in cfg["calendar"]["train_encodings"]}
    for origin in selected:
        destination = partitions / f"predictions_{origin}.csv.gz"
        if destination.exists():
            for name in (f"predictions_{origin}.csv.gz", f"training_{origin}.json", f"errors_{origin}.json",
                         f"importance_{origin}.csv", f"cohort_{origin}.csv", f"timing_{origin}.json"):
                if f"partitions/{name}" not in manifest.get("artifact_sha256", {}) or not (partitions / name).is_file():
                    raise ValueError(f"Incomplete or unverified saved partition: {origin}.")
            print(f"{origin}: verified saved partition, no repeated fits", flush=True)
            continue
        started = time.perf_counter()
        issue = str(period_end(origin).date())
        origin_cases = expected.loc[expected.forecast_origin.eq(issue)].copy()
        saved_k0 = e02.loc[e02.forecast_origin.eq(issue)].copy().assign(model=CALENDAR_MODELS[0], month_encoding="K0")
        yoy = reference.loc[reference.forecast_origin.eq(issue) & reference.model.eq("SeasonalNaiveYoY")].copy()
        yoy["effective_model"] = np.where(yoy.status.eq("native"), "SeasonalNaiveYoY", "LastValue")
        yoy["reason"] = np.where(yoy.status.eq("native"), "", "baseline_internal_fallback")
        rows, training, importance, errors = [saved_k0, yoy], [], [], []
        for horizon, cases in origin_cases.groupby("horizon", sort=True):
            ids = cases.municipality_id.tolist()
            for name in TREND_VARIANTS:
                forecast, trend_errors = forecast_trend(panel, origin, cases, name, cfg)
                rows.append(forecast)
                errors.extend(trend_errors)
            X, delta, trace = build_direct_training(get_prefix(panel, origin, cfg["data"]["release_lag_months"]),
                origin, int(horizon), release_lag_months=cfg["data"]["release_lag_months"],
                mode=cfg["direct_training"]["mode"], max_staleness_months=cfg["direct_training"]["max_staleness_months"])
            source_signature = training_signature(trace, X, delta)
            old = e02_training.loc[e02_training.forecast_origin.eq(issue) & e02_training.horizon.eq(horizon)].iloc[0]
            training.append(dict(old.to_dict(), **source_signature, month_encoding="K0", source="saved_E02",
                                 fit_called_in_e05b=False, feature_names=X.columns.tolist(),
                                 feature_dtypes={col: str(dtype) for col, dtype in X.dtypes.items()}))
            for encoding, predictor in predictors.items():
                result = predictor.predict(panel, origin, int(horizon), ids, cfg["models"])
                check_training_match(source_signature, result.training, old, X, encoding)
                forecast = cases.merge(result.forecasts, on="municipality_id", how="left", validate="one_to_one")
                forecast["model"], forecast["month_encoding"] = f"CatBoostDirect{encoding}", encoding
                rows.append(forecast)
                training.append(dict(result.training, fit_called_in_e05b=result.training["fit_called"], source="E05b"))
                importance.append(result.importance.assign(forecast_origin=issue, horizon=int(horizon), model=f"CatBoostDirect{encoding}"))
                errors.extend(dict(error, model=f"CatBoostDirect{encoding}") for error in result.errors)
                print(f"{origin} h={horizon} {encoding}: {result.training['n_training_rows']} rows, {result.training['status']}", flush=True)
        forecasts = pd.concat(rows, ignore_index=True)
        # Reuse the comparison checker before writing each partition; full evaluator does the same.
        for _, current in forecasts.groupby("model"):
            require_same_keys(current, origin_cases)
        forecasts.to_csv(destination, index=False)
        save_json(partitions / f"training_{origin}.json", training)
        save_json(partitions / f"errors_{origin}.json", errors)
        pd.concat(importance, ignore_index=True).to_csv(partitions / f"importance_{origin}.csv", index=False)
        cohort = eligibility(get_prefix(panel, origin, cfg["data"]["release_lag_months"]),
            cfg["backtest"]["min_history_observations"], cfg["backtest"]["max_staleness_months"])
        cohort.assign(in_sample=cohort.municipality_id.isin(sample), forecast_origin=issue).to_csv(partitions / f"cohort_{origin}.csv", index=False)
        save_json(partitions / f"timing_{origin}.json", {"seconds": time.perf_counter() - started})
        manifest["artifact_sha256"] = {path.relative_to(output).as_posix(): sha256_file(path) for path in output.rglob("*")
            if path.is_file() and path != manifest_path}
        save_json(manifest_path, manifest)
    available = [origin for origin in all_origins if (partitions / f"predictions_{origin}.csv.gz").exists()]
    combined = pd.concat([pd.read_csv(partitions / f"predictions_{origin}.csv.gz", dtype={"municipality_id": str}) for origin in available], ignore_index=True)
    current_expected = expected.loc[expected.forecast_origin.isin(combined.forecast_origin.unique())]
    combined.to_csv(output / "predictions.csv.gz", index=False)
    for group_name, models, native_models in (("trends", TREND_MODELS, list(TREND_VARIANTS)), ("calendar", CALENDAR_MODELS, CALENDAR_MODELS)):
        tables = evaluate_group(combined.loc[combined.model.isin(models)], current_expected, models, native_models)
        for table_name, frame in tables.items():
            frame.to_csv(output / f"{group_name}_{table_name}.csv", index=False)
    records = [item for origin in available for item in json.loads((partitions / f"training_{origin}.json").read_text(encoding="utf-8"))]
    save_json(output / "training_diagnostics.json", records)
    save_json(output / "feature_sets.json", {item["month_encoding"]: {"names": item["feature_names"], "dtypes": item["feature_dtypes"]} for item in records})
    protected = {**e01_hashes, **e02_hashes}
    if any(sha256_file(root / path) != expected_hash for path, expected_hash in protected.items()):
        raise RuntimeError("Saved E01/E02 artifacts changed during E05b.")
    status = {"complete": len(available) == len(all_origins), "n_origins_completed": len(available),
        "n_origins_expected": len(all_origins), "n_models": combined.model.nunique(),
        "n_raw_keys_per_model": len(combined) // combined.model.nunique(), "n_failed": int(combined.status.eq("failed").sum()),
        "n_new_fits_called": sum(bool(item["fit_called_in_e05b"]) for item in records),
        "n_new_fits_succeeded": sum(bool(item["fit_succeeded"]) for item in records if item["source"] == "E05b"),
        "source_models_refitted": False, "sources_unchanged": True,
        "n_training_key_signature_matches": sum(item["source"] == "E05b" for item in records)}
    save_json(output / "run_status.json", status)
    manifest.update(complete=status["complete"], finished_at=datetime.now(timezone.utc).isoformat(),
        artifact_sha256={path.relative_to(output).as_posix(): sha256_file(path) for path in output.rglob("*") if path.is_file() and path != manifest_path})
    save_json(manifest_path, manifest)
    if status["complete"]:
        from .trend_calendar_report import write_report
        write_report(output, project_path(root, cfg["report_path"]), cfg)
    print(json.dumps(status, ensure_ascii=False), flush=True)
    return status
