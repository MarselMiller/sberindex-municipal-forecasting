"""Separate E06a stages over the frozen E04a generator and residual protocol."""
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

from .data import sha256_file
from .online_experiment import prepare_real, safe_path
from .online_synthetic import generate_benchmark
from .offline_detection import run_offline, prefix_stability
from .offline_evaluation import aggregate_metrics, bootstrap_metrics, evaluate_series, select_candidate

METHODS = ("PELT", "BinSeg")
PENALTIES = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
SHARED = ("source", "preparation", "forecast", "generator", "illustrations")


def _clean(value):
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_clean(v) for v in value]
    if isinstance(value, np.generic):
        return _clean(value.item())
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(_clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def tested_files(root: Path) -> dict:
    paths = sorted((root / "src/sberforecast").glob("offline_*.py"))
    paths += sorted((root / "tests").glob("test_offline*.py"))
    paths += [root / "configs/offline_detection.yaml", root / "scripts/run_offline_detection.py",
              root / "requirements-ruptures.txt"]
    return {p.relative_to(root).as_posix(): sha256_file(p) for p in paths}


def artifact_hashes(output: Path) -> dict:
    return {p.relative_to(output).as_posix(): sha256_file(p) for p in sorted(output.rglob("*"))
            if p.is_file() and p != output / "run_manifest.json"}


def validate_specification(root: Path, cfg: dict) -> dict:
    if cfg["reference"] != dict(e04_dir="outputs/online_detection_v1", e04_config="configs/online_detection.yaml",
                                e04_report="reports/results/E04a_online_detection.md"):
        raise ValueError("Use the unchanged saved E04a source and report.")
    old = yaml.safe_load(safe_path(root, cfg["reference"]["e04_config"]).read_text(encoding="utf-8"))
    if cfg["seed"] != old["seed"] or any(cfg[name] != old[name] for name in SHARED):
        raise ValueError("E06a keeps the exact E04a input, preparation, generator and illustrations.")
    keep = ("detection_window_months", "false_alarm_budget_per_12_months", "budget_controls", "budget_rule",
            "primary_scenarios", "bootstrap_replicates", "bootstrap_seed", "bootstrap_unit",
            "no_feasible_candidate", "incomplete_windows", "repeated_alarms")
    if any(cfg["evaluation"][name] != old["evaluation"][name] for name in keep):
        raise ValueError("The E04a event window, control budget and resampling protocol must not change.")
    if (cfg["evaluation"]["matching"] != "chronological_one_to_one_first_breakpoint_in_inclusive_observation_month_window" or
            cfg["evaluation"]["selection_rule"] != "highest_validation_primary_F1_then_recall_then_lowest_max_control_FAR_then_absolute_localisation_error_then_candidate_id"):
        raise ValueError("The declared matching and validation selection rules must match their implementation.")
    expected = {m: dict(model="l2", min_size=2, jump=1, penalty=PENALTIES) for m in METHODS}
    if cfg["detectors"] != expected or cfg["ruptures_version"] != "1.1.10":
        raise ValueError("Only the pre-specified L2 methods and penalty grid are allowed.")
    expected_analysis = dict(cost_model="l2", min_size=2, jump=1,
        input="full_available_residual_history_including_calibration_months",
        standardization="fixed_E04a_first_four_finite_residuals_and_predictions",
        evaluation="only_E04a_monitoring_month_breakpoints_and_exposure", warmup_breakpoints="saved_outside_evaluation",
        future_access="whole_available_series", missing="omit_nonfinite_for_cost_preserve_original_calendar_mapping_no_fill",
        breakpoint_month="first_observed_month_of_right_segment", terminal_index="not_a_change",
        labels_in_fit=False, online_cooldown=False, prefix_matching_tolerance_months=1,
        prefix_matching="nearest_offset_one_to_one_after_independent_prefix_fits",
        localisation_window_basis="observation_month_not_analysis_availability_date")
    if (cfg["offline_analysis"] != expected_analysis or
            cfg["forecasting_model_search_closed"] is not True or cfg["early_warning"] is not False or
            cfg["real_metrics"] != "unlabelled_retrospective_candidates_only" or
            cfg["holdout_status"] != "previously_inspected_not_independent"):
        raise ValueError("Retrospective access, evaluation calendar and closed forecasting search must be explicit.")
    if (cfg["output_dir"] != "outputs/offline_detection_v1" or
            cfg["smoke_output_dir"] != "outputs/offline_detection_smoke_v1" or
            cfg["report_path"] != "reports/results/E06a_offline_detection.md"):
        raise ValueError("Use the separate E06a result directories.")
    if cfg["smoke"] != dict(selection="first_original_replicate_per_scenario_strength_noise_cell",
                            penalty=4.0, metrics="technical_check_not_benchmark_or_selection"):
        raise ValueError("Smoke is a fixed technical subset, without parameter selection.")
    return old


def require_checks(root: Path, cfg: dict) -> tuple[dict, dict, dict]:
    read = lambda name: json.loads(safe_path(root, cfg[name]).read_text(encoding="utf-8"))
    tests, environment, initial = (read(name) for name in ("validation_record", "environment_record", "preservation_record"))
    if tests.get("full_pytest") is not True or tests["exit_code"] != 0 or tests["code_sha256"] != tested_files(root):
        raise ValueError("Successful full pytest of the current E06a files is required before running.")
    versions = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
    if (versions != dict(initial["packages"], ruptures="1.1.10") or
            environment["pip_check_exit_code"] != 0 or environment["version"].lstrip("v") != "1.1.10"):
        raise ValueError("Only the authorized ruptures installation may change the main environment.")
    for name, digest in initial["protected_files"].items():
        if sha256_file(root / name) != digest:
            raise ValueError(f"Protected old file changed: {name}.")
    return tests, environment, initial


def verify_source(root: Path, cfg: dict, old: dict) -> dict:
    directory = safe_path(root, cfg["reference"]["e04_dir"])
    manifest = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8"))
    if not manifest["complete"] or manifest["config"] != old:
        raise ValueError("The completed saved E04a protocol differs from its source configuration.")
    config = safe_path(root, cfg["reference"]["e04_config"])
    if manifest["signature"]["config_sha256"] != sha256_file(config):
        raise ValueError("The E04a configuration changed since its benchmark.")
    hashes = dict(manifest["signature"]["code_sha256"], **manifest["artifact_sha256"])
    for name, digest in hashes.items():
        if sha256_file(root / name) != digest:
            raise ValueError(f"An E04a code or artifact SHA changed: {name}.")
    if sha256_file(safe_path(root, cfg["source"]["data_path"])) != manifest["data_sha256"]:
        raise ValueError("The original expense data changed.")
    hashes[cfg["reference"]["e04_dir"] + "/run_manifest.json"] = sha256_file(directory / "run_manifest.json")
    hashes[cfg["reference"]["e04_report"]] = sha256_file(safe_path(root, cfg["reference"]["e04_report"]))
    return hashes


def require_same_frames(actual: pd.DataFrame, saved: pd.DataFrame, keys: list[str]) -> None:
    if list(actual.columns) != list(saved.columns) or actual.duplicated(keys).any() or saved.duplicated(keys).any():
        raise ValueError("Different source columns or duplicate source keys.")
    pd.testing.assert_frame_equal(actual.sort_values(keys).reset_index(drop=True),
                                  saved.sort_values(keys).reset_index(drop=True),
                                  check_dtype=False, check_exact=False, rtol=0, atol=1e-8)


def verified_synthetic(root: Path, cfg: dict, split: str) -> tuple:
    generated = generate_benchmark(cfg["generator"], split, cfg["forecast"], cfg["preparation"]["release_lag_months"])
    directory = safe_path(root, cfg["reference"]["e04_dir"])
    for name, frame, keys in zip(("residuals", "events", "series", "observations"), generated,
        (["series_id", "observation_period"], ["series_id", "event_id"], ["series_id"], ["series_id", "observation_period"])):
        saved = pd.read_csv(directory / f"synthetic_{split}_{name}.csv.gz", dtype={"series_id": str})
        require_same_frames(frame, saved, keys)
    return generated


def scored(generated: tuple, method: str, params: dict, cfg: dict):
    residuals, events, info, _ = generated
    result = run_offline(residuals, method, params, cfg["preparation"])
    if len(result.errors) or not result.series_diagnostics.status.eq("complete").all():
        raise RuntimeError(f"{method}: segmentation failed or not ready; no reduced successful-only metrics.")
    per, matches, classified = evaluate_series(result.stream, result.breakpoints, events, info,
                                                window_months=cfg["evaluation"]["detection_window_months"])
    return result, per, matches, classified


def tune(validation: tuple, cfg: dict, output: Path) -> dict:
    if not validation[2].split.eq("validation").all():
        raise ValueError("Candidate selection accepts synthetic validation only.")
    selected, rows, lineage = {}, [], {"per_series": [], "breakpoints": [], "matches": []}
    for method in METHODS:
        own = []
        for index, penalty in enumerate(PENALTIES):
            params = dict(model="l2", min_size=2, jump=1, penalty=penalty)
            result, per, matches, _ = scored(validation, method, params, cfg)
            candidate = f"{method}_{index:03d}"
            primary = aggregate_metrics(per.loc[per.scenario.isin(cfg["evaluation"]["primary_scenarios"])]).iloc[0]
            control = {name: float(aggregate_metrics(per.loc[per.scenario.eq(name)]).iloc[0].false_positives_per_12_months)
                       for name in cfg["evaluation"]["budget_controls"]}
            row = dict(method=method, candidate_id=candidate, split="validation", parameters=json.dumps(params, sort_keys=True),
                       primary_f1=primary.f1, primary_recall=primary.recall,
                       primary_median_absolute_localisation_error=primary.median_absolute_localisation_error,
                       no_change_far=control["no_change"], outlier_far=control["outlier"], max_control_far=max(control.values()))
            own.append(row)
            for name, frame in (("per_series", per), ("breakpoints", result.breakpoints), ("matches", matches)):
                lineage[name].append(frame.assign(method=method, candidate_id=candidate))
            print(f"validation {candidate} penalty={penalty}: F1={primary.f1:.4f}, max control FAR={max(control.values()):.4f}", flush=True)
        table = pd.DataFrame(own)
        winner = select_candidate(table, cfg["evaluation"]["false_alarm_budget_per_12_months"])
        if winner is None:
            selected[method] = dict(selected=False, reason="no_candidate_within_fixed_control_budget")
        else:
            selected[method] = dict(selected=True, candidate_id=winner.candidate_id, parameters=json.loads(winner.parameters),
                                    validation_primary_f1=winner.primary_f1,
                                    validation_control_far={n: winner[n + "_far"] for n in cfg["evaluation"]["budget_controls"]})
        table["selected"] = table.candidate_id.eq(None if winner is None else winner.candidate_id)
        rows.extend(table.to_dict("records"))
    pd.DataFrame(rows).to_csv(output / "validation_selection.csv", index=False)
    for name, frames in lineage.items():
        pd.concat(frames, ignore_index=True).to_csv(output / f"validation_candidate_{name}.csv.gz", index=False)
    save_json(output / "selected_parameters.json", selected)
    save_json(output / "selection_seal.json", dict(selected_at=datetime.now(timezone.utc).isoformat(),
              selected_on="synthetic_validation_only", selection_sha256=sha256_file(output / "selected_parameters.json"),
              candidate_table_sha256=sha256_file(output / "validation_selection.csv"), test_generated=False))
    return selected


def save_synthetic(split: str, generated: tuple, selected: dict, cfg: dict, output: Path) -> None:
    for name, frame in zip(("residuals", "events", "series", "observations"), generated):
        frame.to_csv(output / f"synthetic_{split}_{name}.csv.gz", index=False)
    frames = {name: [] for name in ("stream", "breakpoints", "segments", "series_diagnostics", "per_series", "matches", "classified_breakpoints")}
    metrics, uncertainty = [], []
    for method, choice in selected.items():
        if not choice["selected"]:
            continue
        result, per, matches, classified = scored(generated, method, choice["parameters"], cfg)
        for name, frame in (("stream", result.stream), ("breakpoints", result.breakpoints), ("segments", result.segments),
                            ("series_diagnostics", result.series_diagnostics), ("per_series", per),
                            ("matches", matches), ("classified_breakpoints", classified)):
            frames[name].append(frame.assign(method=method))
        primary = per.loc[per.scenario.isin(cfg["evaluation"]["primary_scenarios"])]
        for scope, data, grouping in (("primary_level", primary, []), ("by_scenario", per, ["scenario"]),
                                      ("by_scenario_strength_noise", per, ["scenario", "strength", "noise_fraction"])):
            table = aggregate_metrics(data, grouping).assign(method=method, scope=scope)
            if not grouping:
                table["scenario"] = "level_up_and_down"
            metrics.append(table)
            if split == "test":
                ci = bootstrap_metrics(data, grouping, cfg["evaluation"]["bootstrap_replicates"], cfg["evaluation"]["bootstrap_seed"])
                uncertainty.append(ci.assign(method=method, scope=scope))
        print(f"{split} {method}: {len(per)} series, {len(result.breakpoints)} all retrospective breakpoints", flush=True)
    if not metrics:
        raise RuntimeError("No offline method met the fixed validation control budget; no test or real winner is declared.")
    for name, pieces in frames.items():
        pd.concat(pieces, ignore_index=True).to_csv(output / f"synthetic_{split}_{name}.csv.gz", index=False)
    pd.concat(metrics, ignore_index=True).to_csv(output / f"synthetic_{split}_metrics.csv", index=False)
    if uncertainty:
        pd.concat(uncertainty, ignore_index=True).to_csv(output / "synthetic_test_ci.csv", index=False)


def prepare_real_source(root: Path, cfg: dict):
    residuals, info, observations, sample = prepare_real(root, cfg)
    old = safe_path(root, cfg["reference"]["e04_dir"])
    require_same_frames(residuals, pd.read_csv(old / "real_residuals.csv.gz", dtype={"series_id": str}), ["series_id", "observation_period"])
    require_same_frames(info, pd.read_csv(old / "real_series.csv", dtype={"series_id": str}), ["series_id"])
    coverage = pd.read_csv(old / "real_coverage.csv", dtype={"series_id": str})
    ids = coverage.loc[coverage.n_monitoring_months.gt(0), "series_id"].tolist()
    if len(ids) != 63 or int(coverage.n_monitoring_months.sum()) != 504 or "1471" in ids:
        raise ValueError("The saved 63 monitored municipalities or exposure changed.")
    selected = residuals.loc[residuals.series_id.isin(ids)].copy()
    selected["error"] = selected.y_true - selected.y_pred
    return selected, info.loc[info.series_id.isin(ids)].copy(), observations, sample, coverage


def real_diagnostics(real: tuple, selected: dict, cfg: dict, output: Path) -> dict:
    residuals, info, observations, sample, coverage = real
    residuals.to_csv(output / "real_residuals.csv.gz", index=False)
    info.to_csv(output / "real_series.csv", index=False)
    observations.to_csv(output / "real_observations.csv.gz", index=False)
    coverage.to_csv(output / "real_source_coverage.csv", index=False)
    save_json(output / "sample_ids.json", sample)
    containers = {n: [] for n in ("breakpoints", "segments", "series_diagnostics", "stream", "prefix_estimates", "prefix_matches", "prefix_summary", "prefix_diagnostics")}
    for method, choice in selected.items():
        if not choice["selected"]:
            continue
        result = run_offline(residuals, method, choice["parameters"], cfg["preparation"])
        if len(result.errors) or not result.series_diagnostics.status.eq("complete").all():
            raise RuntimeError(f"Real {method} failed; successful-only summaries are prohibited.")
        prefixes = prefix_stability(residuals, method, choice["parameters"], cfg["preparation"],
                                     matching_tolerance_months=cfg["offline_analysis"]["prefix_matching_tolerance_months"])
        require_same_frames(result.breakpoints, prefixes.full_breakpoints, ["series_id", "breakpoint_id"])
        if prefixes.prefix_diagnostics.status.eq("failed").any():
            raise RuntimeError(f"{method} prefix analysis failed; no hidden omission.")
        for name, frame in (("breakpoints", result.breakpoints), ("segments", result.segments),
                            ("series_diagnostics", result.series_diagnostics), ("stream", result.stream),
                            ("prefix_estimates", prefixes.prefix_breakpoints), ("prefix_matches", prefixes.trajectories),
                            ("prefix_summary", prefixes.summary), ("prefix_diagnostics", prefixes.prefix_diagnostics)):
            containers[name].append(frame.assign(method=method, municipality_id=frame.series_id.astype(str)))
        print(f"real {method}: {len(result.breakpoints)} full-sample candidates; {len(prefixes.prefix_diagnostics)} independent calendar prefixes", flush=True)
    for name, pieces in containers.items():
        pd.concat(pieces, ignore_index=True).to_csv(output / f"real_{name}.csv", index=False)
    bp = pd.concat(containers["breakpoints"], ignore_index=True)
    return dict(n_real_series=int(residuals.series_id.nunique()), n_real_breakpoints=len(bp),
                real_breakpoints_by_method=bp.groupby("method").size().to_dict(),
                n_real_monitoring_months_per_method=int(coverage.n_monitoring_months.sum()),
                n_real_precision_recall_computed=0)


def checkpoint(output: Path, manifest: dict, status: dict) -> None:
    save_json(output / "run_status.json", status)
    manifest.update(artifact_sha256=artifact_hashes(output), complete=bool(status.get("complete")),
                    last_checkpoint_at=datetime.now(timezone.utc).isoformat())
    save_json(output / "run_manifest.json", manifest)


def run_experiment(root: Path, config_path: Path, *, stage: str, command: str) -> dict:
    started = time.perf_counter()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    old = validate_specification(root, cfg)
    tests, environment, initial = require_checks(root, cfg)
    source_hashes = verify_source(root, cfg, old)
    if stage not in {"smoke", "synthetic", "real"}:
        raise ValueError("Choose smoke, synthetic, or real stage.")
    output = safe_path(root, cfg["smoke_output_dir"] if stage == "smoke" else cfg["output_dir"])
    signature = dict(config_sha256=sha256_file(config_path), code_sha256=tested_files(root),
                     source_e04_sha256=source_hashes, test_record_sha256=sha256_file(safe_path(root, cfg["validation_record"])),
                     environment_record_sha256=sha256_file(safe_path(root, cfg["environment_record"])),
                     initial_preservation_sha256=sha256_file(safe_path(root, cfg["preservation_record"])))
    fingerprint = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    path = output / "run_manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest["fingerprint"] != fingerprint or artifact_hashes(output) != manifest["artifact_sha256"]:
            raise ValueError("Code/config/source/test or saved artifact set changed; continuation prohibited.")
        status = json.loads((output / "run_status.json").read_text(encoding="utf-8"))
    else:
        if stage == "real":
            raise ValueError("Synthetic results and their independent verification are required before real diagnostics.")
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("E06a output directory must be empty or have its matching manifest.")
        output.mkdir(parents=True, exist_ok=True)
        manifest = dict(signature=signature, fingerprint=fingerprint, config=cfg, tests=tests, environment=environment,
            git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            git_status=subprocess.check_output(["git", "status", "--short"], cwd=root, text=True).splitlines(),
            versions={d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
            seed=cfg["seed"], python=platform.python_version(), python_executable=sys.executable, platform=platform.platform(),
            started_at=datetime.now(timezone.utc).isoformat(), complete=False, run_commands=[], run_runtimes=[],
            future_access="whole available residual interval; prefixes fitted on their own available data",
            labels_in_fit=False, parameters_selected_on="synthetic_validation_only", real_threshold_tuning=False,
            forecasting_models_refitted=False, forecasting_model_search_closed=True,
            real_independent_blind_test=False, early_warning=False, publication_dates_verified=False)
        manifest["has_uncommitted_changes"] = bool(manifest["git_status"])
        status = dict(complete=False, synthetic_complete=False, real_complete=False, n_validation_series=0, n_test_series=0)
        (output / "config_resolved.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
        save_json(output / "preflight.json", dict(source_e04_verified=True, source_generator_protocol_unchanged=True,
            offline_input_months=12, calibration_months=4, evaluated_monitoring_months=8,
            primary_test_events=360, old_files_protected=len(initial["protected_files"]), labels_in_fit=False,
            real_source_monitoring_municipalities=63, real_source_original_sample=64, municipality_1471_excluded_reason="no_finite_residual_warmup"))
        if stage != "smoke":
            source = safe_path(root, cfg["reference"]["e04_dir"])
            for old_name, new_name in (("synthetic_test_metrics.csv.gz", "online_reference_metrics.csv"),
                                      ("synthetic_test_uncertainty.csv.gz", "online_reference_ci.csv")):
                pd.read_csv(source / old_name).to_csv(output / new_name, index=False)
            save_json(output / "online_reference_selected.json", json.loads((source / "selected_parameters.json").read_text(encoding="utf-8")))
        checkpoint(output, manifest, status)
    if status.get("complete"):
        if stage != "smoke":
            report = safe_path(root, cfg["report_path"])
            if not report.exists() or manifest.get("report_sha256") != sha256_file(report):
                raise ValueError("The completed E06a report is missing or changed; continuation prohibited.")
        print("Verified already completed E06a; no repeated fitting.", flush=True)
        return status
    if stage == "synthetic" and status["synthetic_complete"]:
        print("Verified completed synthetic stage; no repeated fitting.", flush=True)
        return status
    if stage == "real":
        record = json.loads(safe_path(root, cfg["synthetic_verification_record"]).read_text(encoding="utf-8"))
        if (not status["synthetic_complete"] or record.get("passed") is not True or
                record["artifact_sha256"] != manifest["artifact_sha256"] or
                record["selection_sha256"] != sha256_file(output / "selected_parameters.json") or
                record["code_sha256"] != tested_files(root) or record["config_sha256"] != sha256_file(config_path)):
            raise ValueError("Independent verification of the unchanged synthetic results is required before real diagnostics.")
    if stage == "synthetic":
        smoke = json.loads(safe_path(root, cfg["smoke_record"]).read_text(encoding="utf-8"))
        if (smoke.get("passed") is not True or smoke["code_sha256"] != tested_files(root) or
                smoke.get("config_sha256") != sha256_file(config_path)):
            raise ValueError("Successful smoke of the current E06a code is required.")
    manifest["run_commands"].append(command)
    if stage in {"smoke", "synthetic"}:
        validation = verified_synthetic(root, cfg, "validation")
        if stage == "smoke":
            ids = validation[2].loc[validation[2].replicate.eq(0), "series_id"]
            subset = tuple(frame.loc[frame.series_id.isin(ids)].copy() for frame in validation)
            selected = {m: dict(selected=True, parameters=dict(model="l2", min_size=2, jump=1, penalty=cfg["smoke"]["penalty"])) for m in METHODS}
            save_synthetic("validation", subset, selected, cfg, output)
            status.update(complete=True, smoke_passed=True, n_validation_series=len(ids), n_methods=2,
                          n_failed=0, parameters_selected=False, real_complete=False, n_test_series=0)
        else:
            selected = tune(validation, cfg, output)
            save_synthetic("validation", validation, selected, cfg, output)
            sealed = sha256_file(output / "selected_parameters.json")
            test_generated_at = datetime.now(timezone.utc).isoformat()
            test = verified_synthetic(root, cfg, "test")
            if set(validation[2].series_id) & set(test[2].series_id) or set(validation[2].seed) & set(test[2].seed):
                raise ValueError("Validation and test seeds/identifiers overlap.")
            save_synthetic("test", test, selected, cfg, output)
            if sealed != sha256_file(output / "selected_parameters.json"):
                raise ValueError("Selected parameters changed after seeing test.")
            manifest.update(selection_sha256=sealed, test_generated_at=test_generated_at,
                            selected_parameters=selected, selected_parameters_sealed_before_test=True)
            status.update(synthetic_complete=True, n_validation_series=len(validation[2]), n_test_series=len(test[2]),
                          n_failed=0, selected_methods=[m for m in METHODS if selected[m]["selected"]])
    else:
        selected = json.loads((output / "selected_parameters.json").read_text(encoding="utf-8"))
        status.update(real_diagnostics(prepare_real_source(root, cfg), selected, cfg, output))
        status.update(complete=True, real_complete=True, n_failed=0)
    seconds = time.perf_counter() - started
    manifest["run_runtimes"].append(dict(stage=stage, command=command, seconds_before_report=seconds))
    status["runtime_seconds"] = sum(r["seconds_before_report"] for r in manifest["run_runtimes"])
    checkpoint(output, manifest, status)
    if stage == "real":
        from .offline_report import plot_saved, write_report
        plot_saved(output, cfg)
        checkpoint(output, manifest, status)
        report = safe_path(root, cfg["report_path"])
        write_report(output, report, cfg)
        manifest["report_sha256"] = sha256_file(report)
        save_json(path, manifest)
    for name, digest in initial["protected_files"].items():
        if sha256_file(root / name) != digest:
            raise RuntimeError(f"Protected old file changed during E06a: {name}.")
    print(json.dumps(status, ensure_ascii=False), flush=True)
    return status
