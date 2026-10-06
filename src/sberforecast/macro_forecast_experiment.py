"""Thin E05c orchestration: unchanged E02 pairs and E01 forecast support."""
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
from .direct_experiment import project_path, save_json, verify_reference
from .direct_model import TemporalIntegrityError
from .direct_training import PAIR_KEY, build_direct_training
from .macro_forecast_features import INFLATION, CONSUMPTION, SUFFIXES, macro_feature_dictionary
from .macro_forecast_model import MacroCatBoostDirect
from .macro_forecast_evaluation import evaluate_macro
from .trend_calendar_experiment import verify_saved_direct

VARIANTS = {"M0": [], "M1": [INFLATION], "M2": [INFLATION, CONSUMPTION]}
REFERENCES = ["SeasonalNaiveYoY", "ProphetAuto", "ProphetYearly"]
MODELS = [*VARIANTS, *REFERENCES]


def tested_files(root: Path) -> dict[str, str]:
    paths = sorted((root / "src/sberforecast").glob("macro_forecast_*.py"))
    paths += sorted((root / "tests").glob("test_macro_forecast_*.py"))
    paths += [root / "configs/macro_forecast.yaml", root / "scripts/run_macro_forecast.py"]
    return {path.relative_to(root).as_posix(): sha256_file(path) for path in paths}


def validate_specification(cfg: dict) -> None:
    macro = cfg["macro"]
    fixed = {"source_status": "A", "geography": "national", "publication_convention": "month_end",
        "target_year_rule": "calendar_year_of_target_month", "publication_rule": "latest_published_and_available_on_own_release",
        "value_rule": "official_central_else_published_range_midpoint_else_nan", "feature_suffixes": list(SUFFIXES),
        "variants": VARIANTS, "train_variants": ["M1", "M2"], "missing_training_rows": "preserve",
        "category_B_enabled": False, "deflation_enabled": False, "add_trend_enabled": False, "one_hot_enabled": False}
    if any(macro.get(key) != value for key, value in fixed.items()):
        raise ValueError("E05c fixed feature/source specification changed.")
    if cfg["comparison"]["report_models"] != REFERENCES or cfg["holdout_status"] != "previously_inspected_not_independent":
        raise ValueError("Fixed competitors or viewed holdout status changed.")
    if cfg["smoke_origin"] != "2024-03":
        raise ValueError("Smoke origin was fixed before reviewing E05c metrics.")


def require_checks(root: Path, cfg: dict) -> tuple[dict, dict]:
    tests = json.loads(project_path(root, cfg["validation_record"]).read_text(encoding="utf-8"))
    if tests["exit_code"] != 0 or tests.get("full_pytest") is not True or tests["code_sha256"] != tested_files(root):
        raise ValueError("A successful full pytest of the current E05c files is required before training.")
    audit = json.loads(project_path(root, cfg["macro"]["source_audit_record"]).read_text(encoding="utf-8"))
    if audit["status"] != "PASS" or audit["errors"] or audit["checks"]["audited_A_rows"] != 73:
        raise ValueError("Source audit failed: stop training and report E05a extraction/date errors.")
    return tests, audit


def verified_macro_table(root: Path, cfg: dict, audit: dict) -> tuple[pd.DataFrame, dict, dict]:
    source = project_path(root, cfg["macro"]["source_dir"])
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    for name, digest in manifest["artifact_sha256"].items():
        if sha256_file(source / name) != digest:
            raise ValueError(f"E05a artifact changed: {name}.")
    for name, digest in manifest["code_sha256"].items():
        if sha256_file(project_path(root, name)) != digest:
            raise ValueError(f"E05a code changed: {name}.")
    for name, info in manifest["raw_files"].items():
        if sha256_file(project_path(root, name)) != info["sha256"]:
            raise ValueError(f"E05a source cache changed: {name}.")
    table = pd.read_csv(source / "macro_table.csv.gz")
    selected = table.loc[table.availability_status.eq("A") & table.indicator.isin([INFLATION, CONSUMPTION])].copy()
    audited = {(r["indicator"], r["target_year"], r["vintage_id"], r["source_url"]): r for r in audit["audited_rows"]}
    if len(selected) != len(audited) or len(selected) != 73:
        raise ValueError("Audited A source rows differ from the current macro table.")
    source_ids = {}
    for row in selected.itertuples(index=False):
        evidence = audited[(row.indicator, int(row.target_period), row.vintage_id, row.source_url)]
        if (not row.is_forecast or row.region_id != "RU" or row.geographic_level != "national"
                or row.value != evidence["value"] or str(row.published_at) != evidence["published_at"]
                or row.available_at != row.published_at):
            raise ValueError("E05a A value/date/forecast classification differs from independent audit.")
        filename = evidence["file"]
        if sha256_file(source / "raw/cbr" / filename) != evidence["sha256"]:
            raise ValueError("Audited source content changed.")
        source_ids[row.source_url] = Path(filename).stem
    hashes = {path.relative_to(root).as_posix(): sha256_file(path) for path in source.rglob("*") if path.is_file()}
    return selected, source_ids, hashes


def ordered_pair_signature(trace: pd.DataFrame) -> str:
    keys = PAIR_KEY + ["target_available_at"]
    return hashlib.sha256(pd.util.hash_pandas_object(trace[keys], index=False).to_numpy().tobytes()).hexdigest()


def base_feature_dictionary(names: list[str]) -> list[dict]:
    definitions = []
    for name in names:
        unit, nan = "nominal RUB", "NaN when required historical calendar values are absent; no future fill."
        if name.startswith("lag_"):
            lag = int(name.split("_")[1])
            formula = f"y[expense_cutoff - {lag - 1} calendar months]"
        elif name == "last_available":
            formula, nan = "last finite y at or before expense_cutoff", "Finite anchor is required by unchanged pair admission."
        elif name.startswith(("mean_", "std_", "count_")):
            operation, window = name.split("_")
            formula = f"{operation} of last {window} calendar positions, ignoring NaN" + ("; ddof=0" if operation == "std" else "")
            if operation == "count":
                unit, nan = "observations", "Zero if window has no observations."
        elif name == "delta_1":
            formula = "lag_1 - lag_2"
        elif name == "relative_delta_1":
            formula, unit = "(lag_1-lag_2) / max(abs(lag_2), 1)", "dimensionless"
        else:
            formula = {"month": "month of target r+h", "month_sin": "sin(2*pi*target_month/12)",
                "month_cos": "cos(2*pi*target_month/12)", "days_in_month": "calendar days in target month",
                "time_index": "target_year*12+target_month-(2023*12+1)"}[name]
            unit, nan = ("calendar days" if name == "days_in_month" else "calendar months" if name in {"month", "time_index"} else "dimensionless"), "Never NaN: calendar is known in advance."
        definitions.append({"name": name, "formula": formula, "type": "float64", "unit": unit,
            "source": "unchanged E02b direct_features / historical expenses", "nan_rule": nan})
    return definitions


def validate_training(source_signature: dict, ordered: str, old: pd.Series, training: dict) -> None:
    for field in ("training_key_sha256", "training_delta_sha256", "training_raw_X_sha256"):
        if training[field] != source_signature[field]:
            raise TemporalIntegrityError(f"Macro variant changes source training pairs/delta/X: {field}.")
    if training["training_ordered_key_sha256"] != ordered:
        raise TemporalIntegrityError("Macro variant changes training row order.")
    for field in ("n_training_rows", "n_training_municipalities", "n_historical_origins", "n_target_periods"):
        if training[field] != old[field]:
            raise TemporalIntegrityError(f"Training counts differ from saved M0: {field}.")


def _artifact_hashes(output: Path) -> dict:
    return {p.relative_to(output).as_posix(): sha256_file(p) for p in output.rglob("*")
            if p.is_file() and p.name != "run_manifest.json"}


def run_experiment(root: Path, config_path: Path, requested_origins: list[str] | None, command: str) -> dict:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_specification(cfg)
    tests, audit = require_checks(root, cfg)
    macro, source_ids, macro_hashes = verified_macro_table(root, cfg, audit)
    panel = make_panel(load_data(project_path(root, cfg["data"]["path"]), cfg["data"]["category"]))
    sample, reference, expected, e01_hashes = verify_reference(root, panel, cfg)
    saved_m0, old_training, e02_hashes = verify_saved_direct(root, cfg, reference)
    h1_check, _ = h1_consistency(saved_m0, reference, cfg["comparison"]["h1_absolute_tolerance"])
    if not h1_check["complete"] or h1_check["n_exceeds_tolerance"]:
        raise ValueError("Saved M0 no longer matches the original E01 h=1 predictions.")
    output = project_path(root, cfg["output_dir"])
    if output != root / "outputs/macro_forecast_v1":
        raise ValueError("A separate outputs/macro_forecast_v1 directory is required.")
    all_origins = origins_from_config(cfg)
    selected = [pd.Period(value, freq="M") for value in requested_origins] if requested_origins else all_origins
    if not selected or len(set(selected)) != len(selected) or any(o not in all_origins for o in selected):
        raise ValueError("Duplicate or out-of-protocol issue month.")
    signature = {"config": cfg, "config_sha256": sha256_file(config_path), "code_sha256": tested_files(root),
        "data_sha256": sha256_file(project_path(root, cfg["data"]["path"])), "source_e01_sha256": e01_hashes,
        "source_e02_sha256": e02_hashes, "source_macro_sha256": macro_hashes,
        "test_record_sha256": sha256_file(project_path(root, cfg["validation_record"])),
        "source_audit_sha256": sha256_file(project_path(root, cfg["macro"]["source_audit_record"]))}
    fingerprint = hashlib.sha256(json.dumps(signature, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    manifest_path = output / "run_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["fingerprint"] != fingerprint:
            raise ValueError("Code/config/source/test marker changed; resume prohibited.")
        if _artifact_hashes(output) != manifest["artifact_sha256"]:
            raise ValueError("Saved artifact set or SHA changed; resume prohibited.")
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("Nonempty E05c output lacks its matching manifest.")
        output.mkdir(parents=True, exist_ok=True)
        manifest = dict(signature, fingerprint=fingerprint, seed=cfg["seed"], python=platform.python_version(),
            python_executable=sys.executable, platform=platform.platform(),
            versions={d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
            git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            git_status=subprocess.check_output(["git", "status", "--short"], cwd=root, text=True).splitlines(),
            started_at=datetime.now(timezone.utc).isoformat(), run_commands=[], tests=tests,
            source_audit=audit, source_models_refitted=False, M0_refitted=False, independent_blind_test=False,
            publication_dates_verified=False, expense_publication_dates_verified=False,
            macro_A_publication_dates_verified=True, independent_historical_source_snapshots=False,
            category_B_enabled=False, deflation_enabled=False,
            training_scope="unchanged global legacy panel; no macro-missing row removal")
        manifest["has_uncommitted_changes"] = bool(manifest["git_status"])
        save_json(output / "sample_ids.json", sample)
        save_json(output / "source_ids.json", source_ids)
        (output / "config_resolved.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
        save_json(output / "preflight.json", {"e01_e02_data_protocol_keys_targets_verified": True,
            "sample_size": len(sample), "municipality_1471_not_preexcluded": "1471" in sample,
            "n_raw_keys": len(expected), "n_evaluable_keys": int(np.isfinite(expected.y_true).sum()),
            "source_audit_passed": True, "audited_A_rows": len(macro), "saved_M0_h1_check": h1_check})
    manifest["run_commands"].append(command)
    save_json(manifest_path, manifest)
    partitions = output / "partitions"
    partitions.mkdir(exist_ok=True)
    predictors = {name: MacroCatBoostDirect(cfg["models"]["catboost"], cfg["seed"], name, macro,
        source_ids=source_ids, release_lag_months=cfg["data"]["release_lag_months"],
        training_mode=cfg["direct_training"]["mode"], max_staleness_months=cfg["direct_training"]["max_staleness_months"])
        for name in cfg["macro"]["train_variants"]}
    for origin in selected:
        destination = partitions / f"predictions_{origin}.csv.gz"
        if destination.exists():
            print(f"{origin}: verified saved partition, no repeated fits", flush=True)
            continue
        started = time.perf_counter()
        issue = str(period_end(origin).date())
        cases = expected.loc[expected.forecast_origin.eq(issue)].copy()
        control = saved_m0.loc[saved_m0.forecast_origin.eq(issue)].copy().assign(model="M0")
        competitors = reference.loc[reference.forecast_origin.eq(issue) & reference.model.isin(REFERENCES)].copy()
        competitors["effective_model"] = np.where(competitors.status.eq("native"), competitors.model, "LastValue")
        competitors["reason"] = np.where(competitors.status.eq("native"), "", "baseline_internal_fallback")
        rows, training, diagnostics, errors, importance, forecast_lineage = [control, competitors], [], [], [], [], []
        for h, current in cases.groupby("horizon", sort=True):
            X, delta, trace = build_direct_training(get_prefix(panel, origin, cfg["data"]["release_lag_months"]),
                origin, int(h), release_lag_months=cfg["data"]["release_lag_months"],
                mode=cfg["direct_training"]["mode"], max_staleness_months=cfg["direct_training"]["max_staleness_months"])
            source_signature, ordered = training_signature(trace, X, delta), ordered_pair_signature(trace)
            old = old_training.loc[old_training.forecast_origin.eq(issue) & old_training.horizon.eq(h)].iloc[0]
            training.append(dict(old.to_dict(), **source_signature, training_ordered_key_sha256=ordered,
                variant="M0", model_variant="M0", source="saved_E02", fit_called_in_e05c=False,
                feature_names=X.columns.tolist(), feature_dtypes={c: str(t) for c, t in X.dtypes.items()}))
            for variant, predictor in predictors.items():
                try:
                    result = predictor.predict(panel, origin, int(h), current.municipality_id.tolist(), cfg["models"])
                    validate_training(source_signature, ordered, old, result.training)
                except TemporalIntegrityError as exc:
                    save_json(output / "run_status.json", {"complete": False, "stopped_temporal_integrity": True,
                        "forecast_origin": issue, "horizon": int(h), "error": str(exc)})
                    raise
                rows.append(current.merge(result.forecasts, on="municipality_id", how="left", validate="one_to_one").assign(model=variant))
                training.append(dict(result.training, model_variant=variant,
                    fit_called_in_e05c=result.training["fit_called"], source="E05c"))
                diagnostics.extend(result.macro_diagnostics)
                errors.extend(dict(item, model=variant) for item in result.errors)
                importance.append(result.importance.assign(forecast_origin=issue, horizon=int(h), model=variant))
                result.training_provenance.to_csv(partitions / f"training_provenance_{origin}_h{h}_{variant}.csv.gz", index=False)
                forecast_lineage.append(result.forecast_provenance)
                print(f"{origin} h={h} {variant}: {result.training['n_training_rows']} rows / "
                    f"{result.training['n_historical_origins']} dates, {result.training['status']}", flush=True)
        forecasts = pd.concat(rows, ignore_index=True)
        for _, own in forecasts.groupby("model"):
            require_same_keys(own, cases)
        forecasts.to_csv(destination, index=False)
        save_json(partitions / f"training_{origin}.json", training)
        save_json(partitions / f"macro_diagnostics_{origin}.json", diagnostics)
        save_json(partitions / f"errors_{origin}.json", errors)
        nonempty_importance = [frame for frame in importance if not frame.empty]
        (pd.concat(nonempty_importance, ignore_index=True) if nonempty_importance else pd.DataFrame(columns=["feature", "importance", "forecast_origin", "horizon", "model"])).to_csv(partitions / f"importance_{origin}.csv", index=False)
        pd.concat(forecast_lineage, ignore_index=True).to_csv(partitions / f"forecast_provenance_{origin}.csv.gz", index=False)
        cohort = eligibility(get_prefix(panel, origin, cfg["data"]["release_lag_months"]),
            cfg["backtest"]["min_history_observations"], cfg["backtest"]["max_staleness_months"])
        cohort.assign(in_sample=cohort.municipality_id.isin(sample), forecast_origin=issue).to_csv(partitions / f"cohort_{origin}.csv", index=False)
        save_json(partitions / f"timing_{origin}.json", {"seconds": time.perf_counter() - started})
        manifest["artifact_sha256"] = _artifact_hashes(output)
        save_json(manifest_path, manifest)
    available = [o for o in all_origins if (partitions / f"predictions_{o}.csv.gz").exists()]
    combined = pd.concat([pd.read_csv(partitions / f"predictions_{o}.csv.gz", dtype={"municipality_id": str}) for o in available], ignore_index=True)
    current_expected = expected.loc[expected.forecast_origin.isin(combined.forecast_origin.unique())]
    combined.to_csv(output / "predictions.csv.gz", index=False)
    for name, frame in evaluate_macro(combined, current_expected).items():
        frame.to_csv(output / f"{name}.csv", index=False)
    records = [item for o in available for item in json.loads((partitions / f"training_{o}.json").read_text(encoding="utf-8"))]
    diagnostics = [item for o in available for item in json.loads((partitions / f"macro_diagnostics_{o}.json").read_text(encoding="utf-8"))]
    save_json(output / "training_diagnostics.json", records)
    save_json(output / "training_macro_diagnostics.json", diagnostics)
    save_json(output / "feature_sets.json", {item["model_variant"]: {"names": item["feature_names"], "dtypes": item["feature_dtypes"]} for item in records})
    base = base_feature_dictionary(next(item for item in records if item["model_variant"] == "M0")["feature_names"])
    save_json(output / "feature_dictionary.json", {name: base + (macro_feature_dictionary(indicators) if indicators else []) for name, indicators in VARIANTS.items()})
    pd.concat([pd.read_csv(partitions / f"forecast_provenance_{o}.csv.gz", dtype={"municipality_id": str}) for o in available], ignore_index=True).to_csv(output / "forecast_provenance.csv.gz", index=False)
    for path, digest in {**e01_hashes, **e02_hashes, **macro_hashes}.items():
        if sha256_file(root / path) != digest:
            raise RuntimeError(f"Protected source changed during E05c: {path}.")
    status = {"complete": len(available) == len(all_origins), "n_origins_completed": len(available), "n_origins_expected": len(all_origins),
        "n_models": combined.model.nunique(), "n_raw_keys_per_model": len(combined) // combined.model.nunique(),
        "n_failed": int(combined.status.eq("failed").sum()),
        "n_new_fits_called": sum(bool(item["fit_called_in_e05c"]) for item in records),
        "n_new_fits_succeeded": sum(bool(item["fit_succeeded"]) for item in records if item["source"] == "E05c"),
        "source_models_refitted": False, "M0_refitted": False, "sources_unchanged": True,
        "n_training_key_signature_matches": sum(item["source"] == "E05c" for item in records)}
    save_json(output / "run_status.json", status)
    manifest.update(complete=status["complete"], finished_at=datetime.now(timezone.utc).isoformat(), artifact_sha256=_artifact_hashes(output))
    save_json(manifest_path, manifest)
    print(json.dumps(status, ensure_ascii=False), flush=True)
    if status["complete"]:
        from .macro_forecast_report import write_report
        report_path = project_path(root, cfg["report_path"])
        write_report(output, report_path, cfg)
        manifest["report_sha256"] = sha256_file(report_path)
        save_json(manifest_path, manifest)
    return status
