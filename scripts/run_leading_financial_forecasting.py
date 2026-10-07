"""E08c fixed two-family ablation, with a complete F0 gate before financial fits."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sberforecast.backtest import origins_from_config
from sberforecast.data import load_data, make_panel, period_end, sha256_file
from sberforecast.direct_evaluation import require_same_keys
from sberforecast.direct_experiment import verify_reference
from sberforecast.financial_forecast_model import FinancialDirect
from sberforecast.financial_forecast_evaluation import evaluate_financial_forecast
from sberforecast.leading_financial import FEATURE_COLUMNS, build_financial_features, origin_timestamp, parse_key_rate_xml, parse_usd_rub_xml, reconstruct_key_events
from sberforecast.leading_financial_sources import collect_key_decisions
from sberforecast.metrics import KEY
from sberforecast.national_local import build_national
from sberforecast.national_local_experiment import validate_specification
from sberforecast.trend_calendar_evaluation import evaluate_group


def path(value: str) -> Path:
    result = (ROOT / value).resolve()
    if not result.is_relative_to(ROOT):
        raise ValueError("E08c paths must stay inside this project")
    return result


def git(*args) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True, encoding="utf-8").strip()


def save_json(destination: Path, value) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def code_hashes(cfg: dict) -> dict:
    names = ["scripts/run_leading_financial_forecasting.py", "src/sberforecast/financial_forecast_model.py",
             "src/sberforecast/financial_forecast_evaluation.py", "tests/test_financial_forecast_model.py",
             "tests/test_financial_forecast_evaluation.py", "configs/e08c_leading_financial_forecasting.yaml",
             "src/sberforecast/leading_financial.py", "src/sberforecast/leading_financial_sources.py",
             "src/sberforecast/national_local_model.py", "src/sberforecast/national_local.py",
             "src/sberforecast/direct_training.py", cfg["baseline_config"]]
    return {p: sha256_file(path(p)) for p in names}


def financial_matrix(cfg: dict, baseline: dict, old_training: list[dict]) -> pd.DataFrame:
    """Same E08b reconstruction, extended only to existing historical pair origins."""
    source = path(cfg["financial_output"])
    manifest = json.loads((source / "run_manifest.json").read_text(encoding="utf-8"))
    for p, digest in manifest["artifact_hashes"].items():
        if sha256_file(path(p)) != digest:
            raise ValueError("Frozen E08b numeric artifact changed")
    for item in manifest["source_catalog"]:
        if sha256_file(path(item["cache_path"])) != item["sha256"]:
            raise ValueError("Frozen E08b XML changed")
    ledger = manifest["decision_ledger"]
    if sha256_file(path(ledger["path"])) != ledger["sha256"]:
        raise ValueError("Frozen E08b decision ledger changed")
    cache = source / "cache"
    daily = parse_key_rate_xml((cache / "key_rate_2021_2024.xml").read_bytes())
    fx = parse_usd_rub_xml((cache / "usd_rub_2021_2024.xml").read_bytes())
    decisions = pd.DataFrame(collect_key_decisions(cache / "key_decisions", download=False))
    events = reconstruct_key_events(daily, decisions)
    dates = {period_end(o) for o in origins_from_config(baseline)}
    for record in old_training:
        if record["variant"] in cfg["models"].values() and record["n_training_rows"]:
            # These ranges come from E05d's existing training ledger, not a new
            # backtest schedule. Rows with no eligible pair remain unused.
            dates.update(period_end(p) for p in pd.period_range(record["min_historical_origin"], record["max_historical_origin"], freq="M"))
    matrix = build_financial_features(events, fx, sorted(dates))
    if matrix[list(FEATURE_COLUMNS)].isna().any().any():
        raise ValueError("Insufficient E08b financial prehistory; no row dropping or imputation")
    old = pd.read_csv(path(cfg["financial_audit"]))
    old["forecast_origin"] = old.forecast_origin.map(origin_timestamp)
    match = old.merge(matrix, on="forecast_origin", suffixes=("_old", "_new"), validate="one_to_one")
    if len(match) != len(old):
        raise ValueError("Extended matrix loses frozen E08b origins")
    for name in FEATURE_COLUMNS:
        if not np.allclose(match[name + "_old"], match[name + "_new"], atol=1e-12, rtol=0):
            raise ValueError("Extended matrix changes an E08b feature")
    for name in ("max_source_date_used", "max_source_available_at_used"):
        if not matrix[name].le(matrix.forecast_origin).all():
            raise ValueError("A financial source exceeds its own historical origin")
        if not pd.to_datetime(match[name + "_old"], utc=True).equals(pd.to_datetime(match[name + "_new"], utc=True)):
            raise ValueError("Extended matrix changes frozen source cutoffs")
    return matrix


def source_integrity(cfg: dict, base: dict, output: Path) -> dict:
    old = path(cfg["baseline_output"])
    manifest = json.loads((old / "run_manifest.json").read_text(encoding="utf-8"))
    saved = yaml.safe_load((old / "config_resolved.yaml").read_text(encoding="utf-8"))
    if saved != base or manifest["config"] != base:
        raise ValueError("Current E05d config differs from its frozen run")
    for p, digest in manifest["artifact_sha256"].items():
        if sha256_file(old / p) != digest:
            raise ValueError(f"Frozen E05d artifact differs: {p}")
    if importlib.metadata.version("lightgbm") != base["lightgbm_version"]:
        raise ValueError("E05d LightGBM version changed")
    checks = json.loads(path(cfg["validation_record"]).read_text(encoding="utf-8"))
    if checks["exit_code"] != 0 or checks["code_sha256"] != code_hashes(cfg):
        raise ValueError("Current E08c code needs successful relevant synthetic tests")
    if not output.is_relative_to(ROOT / "outputs") or output.exists() and not (output / "run_manifest.json").exists():
        raise ValueError("Use a new E08c output directory or its verified resumable manifest")
    return {"baseline_manifest_sha256": sha256_file(old / "run_manifest.json"),
            "financial_manifest_sha256": sha256_file(path(cfg["financial_output"]) / "run_manifest.json"),
            "tests_sha256": sha256_file(path(cfg["validation_record"]))}


def compare_f0(actual: pd.DataFrame, saved: pd.DataFrame, result, old_training: dict, tolerance: float) -> dict:
    require_same_keys(actual, saved)
    paired = actual.merge(saved, on=KEY, suffixes=("", "_old"), validate="one_to_one")
    for name in ["status", "split", "history_cutoff", "availability_assumption", "effective_model"]:
        if not paired[name].equals(paired[name + "_old"]):
            raise ValueError(f"F0 changes E05d {name}")
    for name in ["reason", "failure_stage", "fallback_baseline_status"]:
        if not paired[name].fillna("").equals(paired[name + "_old"].fillna("")):
            raise ValueError(f"F0 changes E05d {name}")
    for name in ["y_true", "y_pred", "anchor", "ratio_hat", "national_hat"]:
        if not np.allclose(paired[name], paired[name + "_old"], atol=tolerance, rtol=0, equal_nan=True):
            raise ValueError(f"F0 does not reproduce E05d {name}")
    for name in ["training_key_sha256", "training_ordered_key_sha256", "training_delta_sha256",
                 "training_raw_X_sha256", "forecast_raw_X_sha256", "forecast_anchor_sha256",
                 "feature_names", "feature_dtypes", "n_training_rows", "n_historical_origins",
                 "fit_actual_parameters", "fit_requested_parameters", "fit_native_resolved_parameters"]:
        if result.training.get(name) != old_training.get(name):
            raise ValueError(f"F0 changes E05d training specification: {name}")
    return dict(n_predictions=len(actual), max_abs_prediction_difference=float(np.nanmax(np.abs(paired.y_pred - paired.y_pred_old))),
                keys_status_protocol_features_labels_parameters_match=True)


def baseline_gate(output: Path, cfg: dict, expected: pd.DataFrame, saved: pd.DataFrame, old_training: list[dict]) -> dict:
    tasks = list(output.glob("partitions/*_F0.csv.gz"))
    if not tasks:
        raise ValueError("Complete F0 reproduction is required before financial fits")
    fingerprint = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))["fingerprint"]
    for task in tasks:
        stem = task.name[:-len(".csv.gz")]
        marker_path = task.parent / (stem + ".complete.json")
        if not marker_path.exists():
            raise ValueError("F0 gate rejects an unfinished/failed partition")
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker["fingerprint"] != fingerprint or marker["status"] == "failed":
            raise ValueError("F0 gate rejects different settings or failed status")
        for name, digest in marker["artifact_sha256"].items():
            if not (task.parent / name).is_file() or sha256_file(task.parent / name) != digest:
                raise ValueError("F0 gate rejects missing/changed checkpoint artifacts")
        training = json.loads((task.parent / (stem + ".training.json")).read_text(encoding="utf-8"))
        actual_task = pd.read_csv(task, dtype={"municipality_id": str})
        code, issue, horizon = training["variant"], training["forecast_origin"], training["horizon"]
        old_task = saved.loc[saved.model.eq(code) & saved.forecast_origin.eq(issue) & saved.horizon.eq(horizon)]
        old_record = next(r for r in old_training if r["variant"] == code and r["forecast_origin"] == issue and r["horizon"] == horizon)
        compare_f0(actual_task, old_task, SimpleNamespace(training=training), old_record,
                   cfg["baseline_reproduction"]["absolute_tolerance_rub"])
    actual = pd.concat([pd.read_csv(p, dtype={"municipality_id": str}) for p in tasks], ignore_index=True)
    old_metrics = pd.read_csv(path(cfg["baseline_output"]) / "metrics_strategy.csv")
    differences = []
    for model, code in cfg["models"].items():
        current = actual.loc[actual.model.eq(model)]
        require_same_keys(current, expected)
        tables = evaluate_group(current.assign(model="F0"), expected, ["F0"], ["F0"])
        for row in tables["metrics_strategy"].itertuples(index=False):
            reference = old_metrics.loc[old_metrics.model.eq(code) & old_metrics.split.eq(row.split) & old_metrics.horizon.eq(row.horizon)].iloc[0]
            for metric in ["mae_macro", "mae_micro", "r2_pooled"]:
                difference = float(getattr(row, metric) - reference[metric])
                if not np.isclose(getattr(row, metric), reference[metric], atol=cfg["baseline_reproduction"]["absolute_tolerance_rub"], rtol=0, equal_nan=True):
                    raise ValueError("F0 materially differs from saved E05d metrics; financial fits stopped")
                differences.append(dict(model=model, split=row.split, horizon=row.horizon, metric=metric, difference=difference))
    result = dict(passed=True, complete_both_families=True, financial_variant_fits_before_gate=0,
                  prediction_checks="each partition", aggregate_metric_checks=differences)
    save_json(output / "f0_reproduction.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/e08c_leading_financial_forecasting.yaml")
    parser.add_argument("--stage", choices=["smoke", "f0", "ablation"], required=True)
    args = parser.parse_args()
    if git("branch", "--show-current") != "research/e08-leading-indicators":
        raise ValueError("E08c is authorised only on the research branch")
    started = time.perf_counter()
    cfg = yaml.safe_load(path(args.config).read_text(encoding="utf-8"))
    base = yaml.safe_load(path(cfg["baseline_config"]).read_text(encoding="utf-8"))
    validate_specification(base)
    if cfg["seed"] != base["seed"] or cfg["variants"] != {"F0": [], "F1": list(FEATURE_COLUMNS[:5]), "F2": list(FEATURE_COLUMNS[5:]), "F3": list(FEATURE_COLUMNS)}:
        raise ValueError("Seed or fixed E08b financial variants changed")
    if cfg["models"] != {"LightGBMDirectNationalLocal": "LN", "LightGBMDirect": "L0"} or cfg["primary_model"] != "LightGBMDirectNationalLocal" or cfg["primary_horizons"] != [1, 3, 6] or cfg["descriptive_horizons"] != [12]:
        raise ValueError("The fixed model roles or horizons changed")
    if cfg["success_criterion"]["numerical_epsilon_rub"] != 1e-8 or cfg["success_criterion"]["minimum_qualifying_horizons"] != 2 or cfg["success_criterion"]["declared_before_results"] is not True:
        raise ValueError("The declared success criterion differs from the fixed evaluator")
    output = path(cfg["output_dir"])
    source_checks = source_integrity(cfg, base, output)
    panel = make_panel(load_data(path(base["data"]["path"]), base["data"]["category"]))
    sample, reference, expected, old_hashes = verify_reference(ROOT, panel, base)
    saved = pd.read_csv(path(cfg["baseline_output"]) / "predictions.csv.gz", dtype={"municipality_id": str})
    old_training = json.loads((path(cfg["baseline_output"]) / "training_diagnostics.json").read_text(encoding="utf-8"))
    financial = financial_matrix(cfg, base, old_training)
    national = build_national(panel, base["data"]["release_lag_months"])
    specification = {"config": cfg, "code_sha256": code_hashes(cfg), "sources": source_checks,
                     "data_sha256": sha256_file(path(base["data"]["path"])),
                     "versions": {name: importlib.metadata.version(name) for name in ["lightgbm", "numpy", "pandas", "PyYAML"]}}
    fingerprint = hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()
    mp = output / "run_manifest.json"
    if mp.exists():
        manifest = json.loads(mp.read_text(encoding="utf-8"))
        if manifest["fingerprint"] != fingerprint:
            raise ValueError("E08c source/code/specification changed; resume prohibited")
        for name, digest in manifest.get("artifact_sha256", {}).items():
            if sha256_file(output / name) != digest:
                raise ValueError("A completed E08c artifact changed")
    else:
        output.mkdir(parents=True)
        manifest = dict(specification, fingerprint=fingerprint, git_commit=git("rev-parse", "HEAD"),
                        main_commit=git("rev-parse", "main"), git_status_at_start=git("status", "--short"),
                        has_uncommitted_changes=bool(git("status", "--short")), seed=cfg["seed"],
                        python=sys.version.split()[0], started_at=datetime.now(timezone.utc).isoformat(),
                        success_criterion_declared_before_results=cfg["success_criterion"],
                        independent_blind_test=False, expense_release_lag_verified=False,
                        training_population="all available municipalities; unchanged E05d", evaluation_population="unchanged 64-MO pilot cases",
                        run_commands=[], run_runtimes=[], no_tuning=True, no_new_model_family=True,
                        no_source_of_truth_changes=True, no_commit_push=True)
        financial.to_csv(output / "financial_features_own_origins.csv", index=False)
        expected.to_csv(output / "expected_cases.csv", index=False)
        (output / "config_resolved.yaml").write_text(yaml.safe_dump({"e08c": cfg, "e05d_unchanged": base}, allow_unicode=True, sort_keys=False), encoding="utf-8")
        save_json(output / "sample_ids.json", sample)
    manifest["run_commands"].append(".\\.venv\\Scripts\\python.exe -B -X utf8 scripts/run_leading_financial_forecasting.py " + " ".join(sys.argv[1:]))
    save_json(mp, manifest)
    parts = output / "partitions"
    parts.mkdir(exist_ok=True)
    variants = ["F0"] if args.stage in ["smoke", "f0"] else ["F1", "F2", "F3"]
    if args.stage == "ablation":
        baseline_gate(output, cfg, expected, saved, old_training)
    for origin in origins_from_config(base):
        issue = str(period_end(origin).date())
        if args.stage == "smoke" and str(origin) != cfg["smoke_origin"]:
            continue
        cases = expected.loc[expected.forecast_origin.eq(issue)]
        for horizon, own in cases.groupby("horizon", sort=True):
            horizon = int(horizon)
            if args.stage == "smoke" and horizon != cfg["smoke_horizon"]:
                continue
            for model, code in cfg["models"].items():
                baseline_training = next(r for r in old_training if r["variant"] == code and r["forecast_origin"] == issue and r["horizon"] == horizon)
                for variant in variants:
                    stem = f"{origin}_h{horizon}_{code}_{variant}"
                    destination = parts / f"{stem}.csv.gz"
                    if destination.exists():
                        marker = parts / f"{stem}.complete.json"
                        if not marker.exists():
                            raise ValueError("Unfinished/failed partition is preserved; resume requires a complete marker")
                        completed = json.loads(marker.read_text(encoding="utf-8"))
                        if completed["fingerprint"] != fingerprint or completed["status"] == "failed":
                            raise ValueError("Partition specification/status cannot be resumed")
                        for name, digest in completed["artifact_sha256"].items():
                            if not (parts / name).is_file() or sha256_file(parts / name) != digest:
                                raise ValueError("Completed partition lost or changed an artifact")
                        continue
                    if list(parts.glob(stem + ".*")):
                        raise ValueError("Partial partition artifacts are preserved; investigation required before fit")
                    predictor = FinancialDirect(base["lightgbm"], cfg["seed"], code, financial,
                        cfg["variants"][variant], national=national, release_lag_months=base["data"]["release_lag_months"],
                        training_mode=base["direct_training"]["mode"], max_staleness_months=base["direct_training"]["max_staleness_months"])
                    result = predictor.predict(panel, origin, horizon, own.municipality_id.tolist(), base["models"])
                    forecast = own.merge(result.forecasts, on="municipality_id", validate="one_to_one").assign(model=model, variant=variant, base_variant=code)
                    require_same_keys(forecast, own)
                    if variant == "F0":
                        old = saved.loc[saved.model.eq(code) & saved.forecast_origin.eq(issue) & saved.horizon.eq(horizon)]
                        check = compare_f0(forecast, old, result, baseline_training, cfg["baseline_reproduction"]["absolute_tolerance_rub"])
                    else:
                        check = {"financial_source_audit": result.audit_counts}
                        f0 = json.loads((parts / f"{origin}_h{horizon}_{code}_F0.training.json").read_text(encoding="utf-8"))
                        for name in ["training_key_sha256", "training_ordered_key_sha256", "training_delta_sha256", "n_training_rows"]:
                            if result.training[name] != f0[name]:
                                raise ValueError("A financial variant changes E05d training keys/labels")
                        if result.training["base_source_signature"]["training_raw_X_sha256"] != f0["training_raw_X_sha256"] or result.training["base_forecast_raw_X_sha256"] != f0["forecast_raw_X_sha256"]:
                            raise ValueError("A financial variant changes the original E05d base features")
                        for name in ["fit_requested_parameters", "fit_actual_parameters", "fit_native_resolved_parameters", "forecast_anchor_sha256"]:
                            if result.training.get(name) != f0.get(name):
                                raise ValueError("A financial variant changes fixed parameters or anchor")
                    if forecast.status.eq("failed").any():
                        save_json(parts / f"{stem}.errors.json", result.errors)
                        forecast.to_csv(destination, index=False)
                        raise RuntimeError("A fixed ablation fit/prediction failed; saved explicitly, no fallback replacement")
                    forecast.to_csv(destination, index=False)
                    save_json(parts / f"{stem}.training.json", dict(result.training, ablation_variant=variant, model=model, e08c_checks=check))
                    result.training_provenance.to_csv(parts / f"{stem}.provenance.csv.gz", index=False)
                    if not result.financial_training_provenance.empty:
                        result.financial_training_provenance.to_csv(parts / f"{stem}.financial_training.csv.gz", index=False)
                    result.financial_forecast_provenance.to_csv(parts / f"{stem}.financial_forecast.csv", index=False)
                    result.importance.assign(model=model, variant=variant, horizon=horizon, forecast_origin=issue).to_csv(parts / f"{stem}.importance.csv", index=False)
                    marker = parts / f"{stem}.complete.json"
                    artifacts = {p.name: sha256_file(p) for p in parts.glob(stem + ".*") if p != marker}
                    save_json(marker, {"fingerprint": fingerprint, "status": result.training["status"],
                                       "fit_called": result.training["fit_called"], "artifact_sha256": artifacts})
                    manifest.setdefault("artifact_sha256", {}).update({"partitions/" + p.name: sha256_file(p) for p in parts.glob(stem + ".*")})
                    save_json(mp, manifest)
                    print(f"{issue} h={horizon} {code}/{variant}: pairs={result.training['n_training_rows']}, {result.training['status']}, seconds={result.training['seconds']:.2f}", flush=True)
    if args.stage == "f0":
        baseline_gate(output, cfg, expected, saved, old_training)
    records = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(parts.glob("*.training.json"))]
    manifest["model_fits_count"] = sum(bool(r["fit_called"]) for r in records)
    manifest["model_fits_succeeded"] = sum(bool(r["fit_succeeded"]) for r in records)
    manifest["run_runtimes"].append(dict(stage=args.stage, seconds=time.perf_counter()-started))
    manifest["runtime_seconds"] = sum(r["seconds"] for r in manifest["run_runtimes"])
    manifest["complete"] = args.stage == "ablation"
    if manifest["complete"]:
        predictions = pd.concat([pd.read_csv(p, dtype={"municipality_id": str}) for p in sorted(parts.glob("*.csv.gz")) if p.name.count(".") == 2], ignore_index=True)
        predictions.to_csv(output / "predictions.csv.gz", index=False)
        tables = evaluate_financial_forecast(predictions, expected)
        audit = path(cfg["audit_dir"])
        audit.mkdir(parents=True, exist_ok=True)
        for name, table in tables.items():
            table.to_csv(output / f"{name}.csv", index=False)
            table.to_csv(audit / f"{name}.csv", index=False)
        save_json(output / "training_diagnostics.json", records)
        manifest["raw_cases_per_variant"] = len(expected)
        manifest["evaluable_cases_per_variant"] = int(expected.y_true.notna().sum())
        manifest["no_failed_forecasts"] = bool(~predictions.status.eq("failed").any())
        manifest["conclusions"] = tables["success_criterion"].to_dict("records")
        manifest["primary_conclusion"] = tables["success_criterion"].loc[tables["success_criterion"].model.eq(cfg["primary_model"]), "conclusion"].iloc[0]
    manifest["artifact_sha256"] = {p.relative_to(output).as_posix(): sha256_file(p) for p in output.rglob("*") if p.is_file() and p != mp}
    save_json(mp, manifest)
    if manifest["complete"]:
        save_json(path(cfg["audit_dir"]) / "run_manifest.json", manifest)
    print(json.dumps({"stage": args.stage, "complete": manifest["complete"], "fits": manifest["model_fits_count"], "runtime_seconds": manifest["runtime_seconds"], "conclusion": manifest.get("primary_conclusion")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
