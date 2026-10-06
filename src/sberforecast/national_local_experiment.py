"""E05d orchestration over unchanged E01/E02 cohorts, pairs and evaluation."""
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
from .calendar_models import training_signature
from .data import eligibility, get_prefix, load_data, make_panel, period_end, sha256_file
from .direct_evaluation import h1_consistency, require_same_keys
from .direct_experiment import project_path, verify_reference
from .direct_model import TemporalIntegrityError
from .direct_training import build_direct_training
from .macro_forecast_experiment import base_feature_dictionary
from .national_local import build_national, national_forecast, ratio_panel
from .national_local_model import NationalLocalDirect, ordered_training_key_signature
from .national_local_evaluation import evaluate_national_local
from .trend_calendar_experiment import verify_saved_direct

VARIANTS = ["C0", "L0", "CN", "LN"]
REFERENCES = ["SeasonalNaiveYoY", "ProphetAuto", "ProphetYearly"]
MODELS = VARIANTS + REFERENCES
LIGHTGBM = {"objective": "regression_l1", "n_estimators": 300, "learning_rate": .05,
            "num_leaves": 31, "random_state": 42, "n_jobs": 2, "device_type": "cpu",
            "deterministic": True, "force_col_wise": True, "verbosity": -1}
NATIONAL_LOCAL = {
    "variants": VARIANTS, "train_variants": ["L0", "CN", "LN"],
    "national_statistic": "median_finite_all_available_municipalities",
    "missingness_population": "first_observed_by_current_month_inclusive",
    "national_forecaster": "unchanged_SeasonalNaiveYoY",
    "national_forecast_step": "horizon_plus_release_lag",
    "local_ratio": "expense_divided_by_positive_finite_same_month_national_median",
    "local_target": "future_ratio_minus_last_available_ratio",
    "restore": "max_zero_national_forecast_times_predicted_ratio",
    "invalid_denominator": "nan_with_explicit_diagnostics",
    "feature_pipeline": "unchanged_direct_features_K0_19_on_own_history",
    "fallback": "expense_SeasonalNaive", "future_actual_national_for_prediction": False,
}


def _json_default(value):
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported saved diagnostic type: {type(value).__name__}")


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def tested_files(root: Path) -> dict[str, str]:
    paths = sorted((root / "src/sberforecast").glob("national_local*.py"))
    paths += sorted((root / "tests").glob("test_national_local*.py"))
    paths += [root / "configs/national_local_lightgbm.yaml", root / "scripts/run_national_local_lightgbm.py",
              root / "requirements-lightgbm.txt"]
    return {path.relative_to(root).as_posix(): sha256_file(path) for path in paths}


def validate_specification(cfg: dict) -> None:
    if cfg["national_local"] != NATIONAL_LOCAL or cfg["lightgbm"] != LIGHTGBM or cfg["lightgbm_version"] != "4.6.0":
        raise ValueError("Fixed E05d decomposition or learner specification changed.")
    if cfg["comparison"]["report_models"] != REFERENCES or cfg["smoke_origin"] != "2024-03":
        raise ValueError("Fixed E05d competitors or smoke origin changed.")
    disabled = ["macro_features_enabled", "one_hot_enabled", "trend_features_enabled",
                "news_features_enabled", "new_calendar_features_enabled"]
    if any(cfg[field] is not False for field in disabled):
        raise ValueError("E05d keeps only the original K0 features.")
    if cfg["holdout_status"] != "previously_inspected_not_independent" or cfg["forecasting_model_search_final"] is not True:
        raise ValueError("The viewed holdout and final forecasting experiment must be explicit.")


def require_checks(root: Path, cfg: dict) -> tuple[dict, dict, dict]:
    tests = json.loads(project_path(root, cfg["validation_record"]).read_text(encoding="utf-8"))
    if tests["exit_code"] != 0 or tests.get("full_pytest") is not True or tests["code_sha256"] != tested_files(root):
        raise ValueError("Successful full pytest of the current E05d files is required before training.")
    environment = json.loads(project_path(root, cfg["environment_record"]).read_text(encoding="utf-8"))
    initial = json.loads(project_path(root, cfg["preservation_record"]).read_text(encoding="utf-8"))
    versions = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
    expected = dict(initial["packages"], lightgbm=cfg["lightgbm_version"])
    if versions != expected or environment["pip_check_exit_code"] != 0 or environment["version"] != "4.6.0":
        raise ValueError("Only the authorized LightGBM installation may change the main environment.")
    return tests, environment, initial


def require_signature(expected: dict, actual: dict, ordered: str, label: str) -> None:
    for field in ("training_key_sha256", "training_delta_sha256", "training_raw_X_sha256"):
        if expected[field] != actual[field]:
            raise TemporalIntegrityError(f"{label} changes X/y/keys: {field}.")
    if actual["training_ordered_key_sha256"] != ordered:
        raise TemporalIntegrityError(f"{label} changes training row order.")


def artifact_hashes(output: Path) -> dict:
    return {p.relative_to(output).as_posix(): sha256_file(p) for p in output.rglob("*")
            if p.is_file() and p.name != "run_manifest.json"}


def run_experiment(root: Path, config_path: Path, requested_origins: list[str] | None, command: str) -> dict:
    started_run = time.perf_counter()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_specification(cfg)
    tests, environment, initial = require_checks(root, cfg)
    panel = make_panel(load_data(project_path(root, cfg["data"]["path"]), cfg["data"]["category"]))
    sample, reference, expected, e01_hashes = verify_reference(root, panel, cfg)
    saved_c0, old_training, e02_hashes = verify_saved_direct(root, cfg, reference)
    h1_check, _ = h1_consistency(saved_c0, reference, cfg["comparison"]["h1_absolute_tolerance"])
    if not h1_check["complete"] or h1_check["n_exceeds_tolerance"]:
        raise ValueError("Saved C0 does not agree with original E01 h=1.")
    for name, digest in initial["protected_files"].items():
        if sha256_file(root / name) != digest:
            raise ValueError(f"An old protected file changed before training: {name}.")
    national = build_national(panel, cfg["data"]["release_lag_months"])
    ratios, ratio_monthly = ratio_panel(panel, national)
    output = project_path(root, cfg["output_dir"])
    if output != root / "outputs/national_local_lightgbm_v1":
        raise ValueError("E05d requires its separate result directory.")
    all_origins = origins_from_config(cfg)
    selected = [pd.Period(value, freq="M") for value in requested_origins] if requested_origins else all_origins
    if not selected or len(set(selected)) != len(selected) or any(o not in all_origins for o in selected):
        raise ValueError("Duplicate or out-of-protocol forecast origin.")
    signature = {"config": cfg, "config_sha256": sha256_file(config_path), "code_sha256": tested_files(root),
        "data_sha256": sha256_file(project_path(root, cfg["data"]["path"])),
        "source_e01_sha256": e01_hashes, "source_e02_sha256": e02_hashes,
        "test_record_sha256": sha256_file(project_path(root, cfg["validation_record"])),
        "environment_record_sha256": sha256_file(project_path(root, cfg["environment_record"])),
        "protected_snapshot_sha256": sha256_file(project_path(root, cfg["preservation_record"]))}
    fingerprint = hashlib.sha256(json.dumps(signature, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    manifest_path = output / "run_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["fingerprint"] != fingerprint or artifact_hashes(output) != manifest["artifact_sha256"]:
            raise ValueError("Code/config/source/test marker or saved artifact set changed; resume prohibited.")
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("Nonempty E05d directory lacks its matching manifest.")
        output.mkdir(parents=True, exist_ok=True)
        manifest = dict(signature, fingerprint=fingerprint, seed=cfg["seed"], python=platform.python_version(),
            python_executable=sys.executable, platform=platform.platform(), tests=tests, environment=environment,
            versions={d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
            git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            git_status=subprocess.check_output(["git", "status", "--short"], cwd=root, text=True).splitlines(),
            started_at=datetime.now(timezone.utc).isoformat(), run_commands=[], run_runtimes=[],
            C0_refitted=False, source_models_refitted=False, independent_blind_test=False,
            expense_publication_dates_verified=False, publication_dates_verified=False,
            national_population="all finite observed municipalities in each month; no future-fixed cohort",
            local_target="R_target-last_available_R; actual target N only as an available historical label",
            reconstruction="max(0,N_hat*(anchor_R+predicted_delta_R))", future_actual_N_used=False)
        manifest["has_uncommitted_changes"] = bool(manifest["git_status"])
        (output / "config_resolved.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
        save_json(output / "sample_ids.json", sample)
        national.reset_index().to_csv(output / "national_series.csv", index=False)
        ratio_monthly.reset_index().to_csv(output / "local_ratio_monthly.csv", index=False)
        save_json(output / "preflight.json", {"sample_size": len(sample), "n_raw_keys": len(expected),
            "n_evaluable_keys": int(np.isfinite(expected.y_true).sum()), "municipality_1471_not_preexcluded": "1471" in sample,
            "n_national_months": len(national), "n_panel_municipalities": panel.shape[1],
            "n_invalid_national_months": int((~np.isfinite(national.N) | national.N.le(0)).sum()),
            "source_keys_targets_protocol_verified": True, "saved_C0_h1_check": h1_check})
    manifest["run_commands"].append(command)
    save_json(manifest_path, manifest)
    partitions = output / "partitions"
    partitions.mkdir(exist_ok=True)
    predictors = {name: NationalLocalDirect(cfg["models"]["catboost"] if name == "CN" else cfg["lightgbm"],
        cfg["seed"], name, national=national, release_lag_months=cfg["data"]["release_lag_months"],
        training_mode=cfg["direct_training"]["mode"], max_staleness_months=cfg["direct_training"]["max_staleness_months"])
        for name in cfg["national_local"]["train_variants"]}
    for origin in selected:
        destination = partitions / f"predictions_{origin}.csv.gz"
        if destination.exists():
            print(f"{origin}: verified saved partition; no repeated fit", flush=True)
            continue
        started = time.perf_counter()
        issue = str(period_end(origin).date())
        cases = expected.loc[expected.forecast_origin.eq(issue)].copy()
        control = saved_c0.loc[saved_c0.forecast_origin.eq(issue)].copy().assign(model="C0")
        competitors = reference.loc[reference.forecast_origin.eq(issue) & reference.model.isin(REFERENCES)].copy()
        competitors["effective_model"] = np.where(competitors.status.eq("native"), competitors.model, "LastValue")
        competitors["reason"] = np.where(competitors.status.eq("native"), "", "baseline_internal_fallback")
        rows, training, errors, importance, ratio_diagnostics, national_forecasts = [control, competitors], [], [], [], [], []
        for h, current in cases.groupby("horizon", sort=True):
            h = int(h)
            X, delta, trace = build_direct_training(get_prefix(panel, origin, cfg["data"]["release_lag_months"]),
                origin, h, release_lag_months=cfg["data"]["release_lag_months"], mode=cfg["direct_training"]["mode"],
                max_staleness_months=cfg["direct_training"]["max_staleness_months"])
            source, ordered = training_signature(trace, X, delta), ordered_training_key_signature(trace)
            old = old_training.loc[old_training.forecast_origin.eq(issue) & old_training.horizon.eq(h)].iloc[0]
            training.append(dict(old.to_dict(), **source, training_ordered_key_sha256=ordered, variant="C0",
                source="saved_E02", fit_called_in_e05d=False, learner="CatBoost", feature_names=X.columns.tolist(),
                feature_dtypes={name: str(dtype) for name, dtype in X.dtypes.items()}))
            nforecast = national_forecast(national, origin, h, cfg["models"], cfg["data"]["release_lag_months"])
            nforecast["target_period"] = str(pd.Timestamp(nforecast["target_period"]).date())
            nforecast["split"] = current.split.iloc[0]
            national_forecasts.append(nforecast)
            local_control = None
            for variant, predictor in predictors.items():
                try:
                    result = predictor.predict(panel, origin, h, current.municipality_id.tolist(), cfg["models"])
                    require_signature(source, dict(result.training["nominal_source_signature"],
                        training_ordered_key_sha256=result.training["nominal_source_ordered_key_sha256"]), ordered, variant + " nominal source")
                    if variant == "L0":
                        require_signature(source, result.training, ordered, "C0/L0")
                    else:
                        actual_n = result.national_forecast["N_hat"]
                        if not (actual_n == nforecast["N_hat"] or (pd.isna(actual_n) and pd.isna(nforecast["N_hat"]))):
                            raise TemporalIntegrityError("CN/LN use different national forecasts.")
                        if local_control is None:
                            local_control = result.training
                        else:
                            require_signature(local_control, result.training, local_control["training_ordered_key_sha256"], "CN/LN")
                            for field in ("forecast_raw_X_sha256", "forecast_anchor_sha256"):
                                if result.training[field] != local_control[field]:
                                    raise TemporalIntegrityError("CN/LN use different current X/anchor.")
                    for field in ("n_training_rows", "n_training_municipalities", "n_historical_origins", "n_target_periods"):
                        if old[field] != result.training[field]:
                            raise TemporalIntegrityError(f"Actual E05d pair counts differ from C0: {field}.")
                except TemporalIntegrityError as exc:
                    save_json(output / "run_status.json", {"complete": False, "stopped_temporal_integrity": True,
                        "forecast_origin": issue, "horizon": h, "error": str(exc)})
                    raise
                rows.append(current.merge(result.forecasts, on="municipality_id", validate="one_to_one").assign(model=variant))
                training.append(dict(result.training, source="E05d", fit_called_in_e05d=result.training["fit_called"]))
                errors.extend(dict(error, model=variant) for error in result.errors)
                importance.append(result.importance.assign(forecast_origin=issue, horizon=h, model=variant))
                ratio_diagnostics.extend(result.local_ratio_diagnostics)
                result.training_provenance.to_csv(partitions / f"training_provenance_{origin}_h{h}_{variant}.csv.gz", index=False)
                print(f"{origin} h={h} {variant}: {result.training['n_training_rows']} pairs / "
                    f"{result.training['n_historical_origins']} dates, {result.training['status']}", flush=True)
        forecasts = pd.concat(rows, ignore_index=True)
        for _, own in forecasts.groupby("model"):
            require_same_keys(own, cases)
        forecasts.to_csv(destination, index=False)
        save_json(partitions / f"training_{origin}.json", training)
        save_json(partitions / f"errors_{origin}.json", errors)
        save_json(partitions / f"local_ratio_{origin}.json", ratio_diagnostics)
        pd.DataFrame(national_forecasts).to_csv(partitions / f"national_forecasts_{origin}.csv", index=False)
        nonempty = [frame for frame in importance if not frame.empty]
        (pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame(columns=["feature", "importance", "forecast_origin", "horizon", "model"])).to_csv(partitions / f"importance_{origin}.csv", index=False)
        cohort = eligibility(get_prefix(panel, origin, cfg["data"]["release_lag_months"]),
            cfg["backtest"]["min_history_observations"], cfg["backtest"]["max_staleness_months"])
        cohort.assign(in_sample=cohort.municipality_id.isin(sample), forecast_origin=issue).to_csv(partitions / f"cohort_{origin}.csv", index=False)
        save_json(partitions / f"timing_{origin}.json", {"seconds": time.perf_counter() - started})
        manifest["artifact_sha256"] = artifact_hashes(output)
        save_json(manifest_path, manifest)
    available = [o for o in all_origins if (partitions / f"predictions_{o}.csv.gz").exists()]
    combined = pd.concat([pd.read_csv(partitions / f"predictions_{o}.csv.gz", dtype={"municipality_id": str}) for o in available], ignore_index=True)
    current_expected = expected.loc[expected.forecast_origin.isin(combined.forecast_origin.unique())]
    national_forecasts = pd.concat([pd.read_csv(partitions / f"national_forecasts_{o}.csv") for o in available], ignore_index=True)
    combined.to_csv(output / "predictions.csv.gz", index=False)
    national_forecasts.to_csv(output / "national_forecasts.csv", index=False)
    for name, frame in evaluate_national_local(combined, current_expected, national_forecasts).items():
        frame.to_csv(output / f"{name}.csv", index=False)
    records = [row for o in available for row in json.loads((partitions / f"training_{o}.json").read_text(encoding="utf-8"))]
    ratio_diagnostics = [row for o in available for row in json.loads((partitions / f"local_ratio_{o}.json").read_text(encoding="utf-8"))]
    save_json(output / "training_diagnostics.json", records)
    save_json(output / "local_ratio_diagnostics.json", ratio_diagnostics)
    save_json(output / "resolved_parameters.json", [row for row in records if row.get("fit_called_in_e05d")])
    save_json(output / "feature_sets.json", {row["variant"]: {"names": row["feature_names"], "dtypes": row["feature_dtypes"]} for row in records})
    raw_dictionary = base_feature_dictionary(next(row for row in records if row["variant"] == "C0")["feature_names"])
    save_json(output / "feature_dictionary.json", {name: [dict(item,
        source="unchanged E02 direct_features on historical R=y/N" if name in {"CN", "LN"} else item["source"],
        unit="dimensionless local ratio" if name in {"CN", "LN"} and item["unit"] == "nominal RUB" else item["unit"])
        for item in raw_dictionary] for name in VARIANTS})
    for name, digest in initial["protected_files"].items():
        if sha256_file(root / name) != digest:
            raise RuntimeError(f"Old protected file changed during E05d: {name}.")
    runtime = sum(json.loads((partitions / f"timing_{o}.json").read_text())["seconds"] for o in available)
    status = {"complete": len(available) == len(all_origins), "n_origins_completed": len(available),
        "n_origins_expected": len(all_origins), "n_models": combined.model.nunique(),
        "n_raw_keys_per_model": len(combined) // combined.model.nunique(), "n_failed": int(combined.status.eq("failed").sum()),
        "n_new_fits_called": sum(bool(row["fit_called_in_e05d"]) for row in records),
        "n_new_fits_succeeded": sum(bool(row["fit_succeeded"]) for row in records if row["source"] == "E05d"),
        "n_training_key_signature_matches": sum(row["source"] == "E05d" for row in records),
        "C0_refitted": False, "source_models_refitted": False, "sources_unchanged": True,
        "runtime_seconds": runtime, "command_runtime_seconds": time.perf_counter() - started_run}
    save_json(output / "run_status.json", status)
    manifest["run_runtimes"].append({"command": command, "seconds_before_report": status["command_runtime_seconds"]})
    manifest.update(complete=status["complete"], finished_at=datetime.now(timezone.utc).isoformat(), artifact_sha256=artifact_hashes(output))
    save_json(manifest_path, manifest)
    print(json.dumps(status, ensure_ascii=False), flush=True)
    if status["complete"]:
        from .national_local_report import write_report
        report_path = project_path(root, cfg["report_path"])
        write_report(output, report_path, cfg)
        manifest["report_sha256"] = sha256_file(report_path)
        save_json(manifest_path, manifest)
    return status
