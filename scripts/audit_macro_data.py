"""E05a public-source and temporal-coverage audit; does not fit models."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sberforecast.data import load_data, make_panel, sha256_file
from sberforecast.macro_audit import (e01_cases, feature_dictionary, forecast_coverage,
    forecast_grid, save_feature_audit, training_coverage, training_groups, training_queries, trend_readiness)
from sberforecast.macro_cbr import collect_cbr_sources
from sberforecast.macro_features import build_price_index, build_region_mapping, validate_macro_table
from sberforecast.macro_rosstat import audit_rosstat_sources


def project_path(name: str) -> Path:
    path = (ROOT / name).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Audit paths must stay inside the project.")
    return path


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def complete_raw_catalog(raw_dir: Path, catalog: list[dict]) -> None:
    """Account for duplicate downloads and derived caches as separate artifacts."""
    primary = list(catalog)
    for entry in primary:
        entry["catalog_role"] = "primary_source"
    indexed = {Path(entry["file"]).resolve() for entry in primary if entry.get("file")}
    for path in sorted(raw_dir.rglob("*")):
        if not path.is_file() or path.resolve() in indexed:
            continue
        digest = sha256_file(path)
        match = next((entry for entry in primary if digest == entry.get("sha256")), None)
        text_match = next((entry for entry in primary if digest == entry.get("text_sha256")), None)
        if match or text_match:
            entry = dict(match or text_match)
            entry.update({"file": str(path), "sha256": digest, "bytes": path.stat().st_size,
                          "primary_source_file": (match or text_match).get("file"),
                          "catalog_role": "duplicate_download" if match else "extracted_text",
                          "artifact_created_at": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()})
            if text_match:
                entry["original_pdf_sha256"] = text_match["sha256"]
                entry["downloaded_at"] = None
        else:
            entry = {"source": "Local provenance for Bank of Russia sources", "url": None,
                     "name": path.name, "file": str(path), "unit": None, "geographic_level": None,
                     "data_period": None, "published_at": None, "version_published_at": None,
                     "downloaded_at": None, "updated_at": None, "sha256": digest,
                     "availability_status": "B", "verification_status": "generated_provenance_metadata",
                     "catalog_role": "local_provenance", "bytes": path.stat().st_size,
                     "artifact_created_at": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()}
        catalog.append(entry)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/macro_data_audit.yaml")
    parser.add_argument("--download", action="store_true", help="Explicitly enable official public source downloads.")
    args = parser.parse_args()
    config_path = project_path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output = project_path(config["output_dir"])
    # Existing raw cache is allowed; completed processed results are never overwritten.
    if (output / "manifest.json").exists():
        raise FileExistsError("Completed audit exists: choose a new output_dir.")
    output.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    raw_dir = project_path(config["sources"]["raw_dir"])
    macro, catalog = collect_cbr_sources(raw_dir / "cbr", allow_network=args.download)
    failure_catalog = project_path(config["sources"]["rosstat_failure_catalog"])
    if args.download:
        rosstat = audit_rosstat_sources(raw_dir / "rosstat")
    elif failure_catalog.exists():
        rosstat = json.loads(failure_catalog.read_text(encoding="utf-8"))
    else:
        rosstat = [{"source": "Rosstat", "availability_status": "C", "status": "not_downloaded", "error": "Offline run, no verified source catalog."}]
    catalog.extend(rosstat)
    complete_raw_catalog(raw_dir, catalog)
    macro = validate_macro_table(macro)
    macro.to_csv(output / "macro_table.csv.gz", index=False, compression="gzip")
    write_json(output / "source_catalog.json", catalog)
    dictionary = feature_dictionary(config["forecast_indicators"])
    write_json(output / "feature_dictionary.json", dictionary)
    features = list(dictionary["features"])
    print(f"Read macro observations: {len(macro)}; statuses {macro.availability_status.value_counts().to_dict()}", flush=True)

    data_path = project_path(config["data"]["path"])
    data = load_data(data_path, config["data"]["category"])
    panel = make_panel(data)
    mapping = build_region_mapping(data)
    mapping.to_csv(output / "municipality_region_mapping.csv.gz", index=False, compression="gzip")
    source = project_path(config["comparison"]["source_dir"])
    original = yaml.safe_load((source / "config_resolved.yaml").read_text(encoding="utf-8"))
    for name in ("first_origin", "last_origin", "horizons"):
        if config["comparison"][name] != original["backtest"][name]:
            raise ValueError(f"Audit changes E01 evaluation bounds: {name}.")
    if config["data"]["release_lag_months"] != original["data"]["release_lag_months"]:
        raise ValueError("Audit changes E01 target publication-lag assumption.")
    ids = json.loads((source / "sample_ids.json").read_text(encoding="utf-8"))
    predictions_path = source / "predictions.csv.gz"
    cases = e01_cases(pd.read_csv(predictions_path, dtype={"municipality_id": str}))
    origins = pd.period_range(config["comparison"]["first_origin"], config["comparison"]["last_origin"], freq="M")
    horizons = config["comparison"]["horizons"]
    grid = forecast_grid(ids, origins, horizons, cases, mapping)
    grid.to_csv(output / "forecast_cases.csv.gz", index=False, compression="gzip")
    lag = config["data"]["release_lag_months"]
    groups, counts = training_groups(panel, origins, horizons, mapping, release_lag_months=lag,
                                     **config["direct_training"])
    queries, groups = training_queries(groups, mapping)
    groups.to_csv(output / "training_groups.csv.gz", index=False, compression="gzip")
    queries.to_csv(output / "historical_queries.csv.gz", index=False, compression="gzip")
    counts.to_csv(output / "training_pair_availability.csv", index=False)
    print(f"Forecast grid: {len(grid)}; E01 raw/evaluable: {len(cases)}/{cases.e01_evaluable.sum()}; historical macro queries: {len(queries)}", flush=True)
    coverage_summary = {}
    for scenario, lags in config["scenarios"].items():
        print(f"Joining {scenario} on own historical dates ...", flush=True)
        _, provenance = save_feature_audit(output, f"forecast_{scenario}", grid, macro, config["forecast_indicators"], lags)
        coverage = forecast_coverage(grid, provenance)
        coverage["scenario"] = scenario
        coverage.to_csv(output / f"forecast_coverage_{scenario}.csv", index=False)
        _, historical = save_feature_audit(output, f"historical_{scenario}", queries, macro, config["forecast_indicators"], lags)
        training = training_coverage(groups, counts, historical, features)
        training["scenario"] = scenario
        training.to_csv(output / f"training_coverage_{scenario}.csv", index=False)
        price_records = []
        regions = sorted(grid.region_id.unique())
        no_monthly_cpi = not ((~macro.is_forecast) & macro.indicator.eq("cpi_mom_index")).any()
        for origin in origins:
            # With no CPI observations, every region has the same broken chain.
            # Validate/build it once per date, without inventing price values.
            empty_chain = build_price_index(macro, regions[0], origin.end_time.normalize(),
                base_period=config["price_index"]["base_period"], scenario_lags=lags) if no_monthly_cpi else None
            for region in regions:
                price = empty_chain.copy() if no_monthly_cpi else build_price_index(macro, region, origin.end_time.normalize(),
                    base_period=config["price_index"]["base_period"], scenario_lags=lags)
                price["region_id"], price["forecast_origin"] = region, origin.end_time.normalize()
                price_records.append(price)
        pd.concat(price_records, ignore_index=True).to_csv(output / f"price_index_{scenario}.csv.gz", index=False, compression="gzip")
        coverage_summary[scenario] = coverage.groupby("feature")[["n_cohort_cases", "n_available_cohort_cases", "n_e01_evaluable", "n_available_e01_evaluable", "n_A_cohort_cases", "n_B_cohort_cases"]].sum().to_dict("index")
    readiness = trend_readiness(panel, grid, release_lag_months=lag,
        window_months=config["e05b_specification"]["trend_annual_difference_window_months"],
        min_pairs=config["e05b_specification"]["trend_min_annual_pairs"])
    readiness.to_csv(output / "trend_readiness.csv.gz", index=False, compression="gzip")
    trend = readiness.groupby(["forecast_origin", "horizon"]).agg(n_cases=("sample_id", "size"),
        n_native_ready=("native_trend_ready", "sum"), n_e01_cases=("e01_case", "sum"),
        n_e01_evaluable=("e01_evaluable", "sum")).reset_index()
    native_e01 = readiness.loc[readiness.e01_evaluable & readiness.native_trend_ready].groupby(["forecast_origin", "horizon"]).size()
    trend["n_native_ready_e01_evaluable"] = [int(native_e01.get((row.forecast_origin, row.horizon), 0)) for row in trend.itertuples()]
    trend.to_csv(output / "trend_readiness_summary.csv", index=False)
    summary = {"geography": {"panel_municipalities": len(mapping), "panel_regions": mapping.region_id.nunique(),
                               "cohort_municipalities": len(ids), "cohort_regions": grid.region_id.nunique()},
               "source_rows": macro.groupby(["availability_status", "indicator"]).size().to_dict(),
               "e01_cases": len(cases), "e01_evaluable": int(cases.e01_evaluable.sum()),
               "forecast_grid_cases": len(grid), "historical_queries": len(queries),
               "coverage": coverage_summary, "model_training_performed": False,
               "regional_observations_ready": bool((~macro.is_forecast & macro.region_id.ne("RU")).any()),
               "historical_macro_experiment_ready": False,
               "notes": ["Target publication lag zero is inherited, unverified E01 assumption.",
                         "National forecasts are broadcast, not regional observations.",
                         "A dated archive is accepted as historical-vintage evidence, not independently timestamped 2023 byte capture.",
                         "Training pairs are global legacy E02 pairs; each macro query uses its own r.",
                         "Trend readiness counts do not prove forecast quality; E05b has not run."]}
    summary["source_rows"] = {f"{key[0]}:{key[1]}": value for key, value in summary["source_rows"].items()}
    write_json(output / "summary.json", summary)
    (output / "config_resolved.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).splitlines()
    versions = {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()}
    raw_files = {str(path.relative_to(ROOT)): {"sha256": sha256_file(path), "bytes": path.stat().st_size}
                 for path in sorted(raw_dir.rglob("*")) if path.is_file()}
    code_files = [config_path, Path(__file__), *sorted((ROOT / "src/sberforecast").glob("macro_*.py"))]
    artifact_files = {str(path.relative_to(output)): sha256_file(path) for path in sorted(output.glob("*")) if path.is_file()}
    manifest = {"started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(),
                "command": subprocess.list2cmdline([sys.executable, *sys.argv]), "seed": config["seed"],
                "git_commit": git_commit, "git_dirty": bool(dirty), "git_status": dirty,
                "versions": versions, "network_enabled_for_this_command": args.download,
                "model_training_performed": False, "data_sha256": sha256_file(data_path),
                "e01_predictions_sha256": sha256_file(predictions_path),
                "e01_sample_ids_sha256": sha256_file(source / "sample_ids.json"),
                "code_sha256": {str(path.resolve().relative_to(ROOT)): sha256_file(path) for path in code_files},
                "raw_files": raw_files, "artifact_sha256": artifact_files}
    write_json(output / "manifest.json", manifest)
    print(f"Audit saved: {output}; no model training.", flush=True)


if __name__ == "__main__":
    main()
