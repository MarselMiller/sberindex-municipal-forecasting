"""Run E08d from frozen real artifacts; no fits, new labels or network."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sberforecast.real_financial_early_warning import (CONCLUSION, build_origin_study, calculate_tables,
    load_real_inputs, month_date, read_json, sha256)

CODE = ["src/sberforecast/real_financial_early_warning.py",
        "scripts/run_real_financial_early_warning.py",
        "scripts/check_real_financial_early_warning.py",
        "scripts/report_real_financial_early_warning.py"]


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def code_hashes(config):
    return {name: sha256(ROOT / name) for name in CODE + [config] if (ROOT / name).is_file()}


def python_command():
    executable = Path(sys.executable).resolve()
    return executable.relative_to(ROOT).as_posix() if executable.is_relative_to(ROOT) else "python"


def checked_path(value):
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT) or Path(value).is_absolute():
        raise ValueError("E08d paths must remain relative to this repository")
    return path


def finalize(cfg, config, command):
    output, audit = checked_path(cfg["output_dir"]), checked_path(cfg["audit_dir"])
    manifest = read_json(output / "run_manifest.json")
    for name in ["real_validation.json", "independent_validation.json"]:
        proof = read_json(ROOT / "outputs/e08d_checks" / name)
        if proof.get("status") != "PASS":
            raise ValueError(f"Validation has not passed: {name}")
        manifest.setdefault("validation", {})[name] = {
            "path": "outputs/e08d_checks/"+name,
            "sha256": sha256(ROOT / "outputs/e08d_checks" / name), "status": "PASS"}
        proof_command = proof.get("command")
        if proof_command and proof_command not in manifest["run_commands"]:
            manifest["run_commands"].append(proof_command)
    report = checked_path(cfg["report_path"])
    if not report.is_file():
        raise ValueError("Report is absent")
    figures = sorted((output / "figures").glob("*.png"))
    if len(figures) > cfg["maximum_figures"] or len(figures) != 2:
        raise ValueError("Expected exactly two declared real-data figures")
    manifest["artifact_sha256"] = {p.relative_to(output).as_posix(): sha256(p)
        for p in sorted(output.rglob("*")) if p.is_file() and p.name != "run_manifest.json"}
    manifest["report"] = {"path": cfg["report_path"], "sha256": sha256(report)}
    manifest.setdefault("code_sha256_at_run", manifest["code_sha256"])
    manifest["code_sha256"] = code_hashes(config)
    manifest["git_status_at_completion"] = git("status", "--porcelain=v1")
    for path in [ROOT/cfg["smoke_output_dir"]/"run_manifest.json", ROOT/"outputs/e08d_checks/source_validation.json",
                 output/"figures/figure_manifest.json"]:
        record = read_json(path)
        recorded_commands = record.get("run_commands", [record.get("command")])
        for recorded in recorded_commands:
            if recorded and recorded not in manifest["run_commands"]:
                manifest["run_commands"].append(recorded)
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["run_commands"].append(command)
    manifest["status"] = "E08d real financial early-warning diagnostic complete."
    manifest["no_commit_push"] = True
    save_json(output / "run_manifest.json", manifest)
    save_json(audit / "run_manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "primary_conclusion": manifest["primary_conclusion"],
                      "figures": len(figures)}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/e08d_real_financial_early_warning.yaml")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    config = checked_path(args.config).relative_to(ROOT).as_posix()
    cfg = yaml.safe_load((ROOT / config).read_text(encoding="utf-8"))
    if git("branch", "--show-current") != cfg["branch"]:
        raise ValueError("E08d is restricted to its research branch")
    command = python_command()+" scripts/run_real_financial_early_warning.py --config "+config
    if args.finalize:
        if args.smoke:
            raise ValueError("Smoke and finalize are separate stages")
        return finalize(cfg, config, command+" --finalize")
    started = time.perf_counter()
    financial, events, cases, checks = load_real_inputs(ROOT, cfg)
    if args.smoke:
        selected = financial.loc[financial.forecast_origin.isin(["2023-12-31", "2024-04-30"])]
        tables = {"origin_event_study.csv": build_origin_study(selected, events, cases)}
    else:
        tables = calculate_tables(financial, events, cases, cfg)
    output = checked_path(cfg["smoke_output_dir"] if args.smoke else cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()) and not (output / "run_manifest.json").is_file():
        raise ValueError("New output directory is not empty and has no E08d manifest")
    if (output / "run_manifest.json").is_file() and read_json(output / "run_manifest.json").get("experiment") != cfg["experiment"]:
        raise ValueError("Refusing to replace another experiment")
    if args.smoke:
        # Two genuine saved dates, selected by calendar, never generated fixtures.
        command += " --smoke"
    else:
        # The frozen source has no date-level negative controls. Do not infer an
        # evidence grade from undefined contrasts or from an arbitrary p gate.
        if tables["eligibility_summary.csv"].n_negative.ne(0).any():
            raise ValueError("The source support changed; the E08d interpretation needs explicit review")
    for name, frame in tables.items():
        frame.to_csv(output / name, index=False, na_rep="NA", float_format="%.17g")
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    manifest = dict(experiment=cfg["experiment"], stage="REAL_MINIMAL_SMOKE" if args.smoke else "REAL_FULL_DIAGNOSTIC",
        status="REAL_CODE_RUN_COMPLETE_REPORT_PENDING", seed=cfg["seed"], config=config,
        statistical_unit="unique_forecast_origin", onset_dates=[month_date(v) for v in sorted(events.onset_period.unique())],
        eligibility_summary=[] if args.smoke else tables["eligibility_summary.csv"].to_dict("records"),
        primary_diagnostics=cfg["primary_diagnostics"], planned_holm_family_size=8,
        primary_conclusion=CONCLUSION, conclusion_reason="No known negative date controls; contrasts are not identified, absence of precursors is not established",
        real_data_only=True, no_synthetic_data=True, no_classifier_fits=True, no_threshold_tuning=True,
        no_network=True, no_commit_push=True, label_definition_changed=False, target_definition_changed=False,
        git_commit=git("rev-parse", "HEAD"), main_commit=git("rev-parse", "main"),
        git_status=git("status", "--porcelain=v1"), uncommitted_changes=bool(git("status", "--porcelain=v1")),
        versions={name: importlib.metadata.version(name) for name in ["numpy", "pandas", "PyYAML", "matplotlib"]},
        python_version=sys.version.split()[0], code_sha256=code_hashes(config), run_commands=[command],
        runtime_seconds=round(time.perf_counter()-started, 6),
        preservation_record=cfg["preservation_record"], **checks)
    manifest["artifact_sha256"] = {p.relative_to(output).as_posix(): sha256(p)
        for p in sorted(output.iterdir()) if p.is_file() and p.name != "run_manifest.json"}
    save_json(output / "run_manifest.json", manifest)
    if not args.smoke:
        audit = checked_path(cfg["audit_dir"])
        audit.mkdir(parents=True, exist_ok=True)
        for name, frame in tables.items():
            frame.to_csv(audit / name, index=False, na_rep="NA", float_format="%.17g")
        save_json(audit / "run_manifest.json", manifest)
    print(json.dumps({"stage": manifest["stage"], "runtime_seconds": manifest["runtime_seconds"],
        "source_counts": checks["source_counts"], "output_rows": {n: len(t) for n, t in tables.items()},
        "eligibility_summary": manifest["eligibility_summary"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
