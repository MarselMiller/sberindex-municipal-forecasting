"""Validate E08d only against its frozen REAL inputs and saved artifacts.

This is an artifact/replay check, not a synthetic fixture or model benchmark.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sberforecast.real_financial_early_warning import (FEATURES, PRIMARY, calculate_tables,
    load_real_inputs, month, read_json, sha256)


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def require(condition, description):
    if not condition:
        raise AssertionError(description)


def preserve(cfg):
    snapshot = read_json(ROOT / cfg["preservation_record"])
    require(git("rev-parse", "HEAD") == snapshot["head"], "HEAD changed")
    require(git("rev-parse", "main") == snapshot["main"], "main changed")
    require(not git("diff", "--cached", "--name-only"), "New staged files")
    allowed = {"docs/PROJECT_CONTEXT.md", "docs/TASKS.md"}
    protected = 0
    for category in ["tracked", "protected"]:
        for name, digest in snapshot[category].items():
            if category == "tracked" and name in allowed:
                continue
            require((ROOT/name).is_file() and sha256(ROOT/name) == digest, f"Preservation failed: {name}")
            protected += 1
    subprocess.run(["git", "diff", "--check"], cwd=ROOT, check=True)
    return protected


def compare_saved(expected, path):
    saved = pd.read_csv(path, keep_default_na=True)
    require(list(saved.columns) == list(expected.columns), f"Column schema differs: {path.name}")
    require(len(saved) == len(expected), f"Row count differs: {path.name}")
    for column in expected:
        left, right = expected[column], saved[column]
        require(np.array_equal(left.isna().to_numpy(), right.isna().to_numpy()) or
                np.array_equal(left.replace("", np.nan).isna().to_numpy(), right.isna().to_numpy()),
                f"Missingness differs: {path.name}/{column}")
        if pd.api.types.is_numeric_dtype(left.dtype) and not pd.api.types.is_bool_dtype(left.dtype):
            np.testing.assert_allclose(left.to_numpy(dtype=float, na_value=np.nan), right.to_numpy(dtype=float),
                                       rtol=1e-12, atol=1e-14, equal_nan=True)
        else:
            require(left.fillna("").astype(str).tolist() == right.fillna("").astype(str).tolist(),
                    f"Values differ: {path.name}/{column}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/e08d_real_financial_early_warning.yaml")
    parser.add_argument("--source-only", action="store_true")
    parser.add_argument("--complete", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    cfg = yaml.safe_load((ROOT/args.config).read_text(encoding="utf-8"))
    financial, events, cases, source = load_real_inputs(ROOT, cfg)
    require(len(cases) == 52560, "Frozen real case count")
    tables = calculate_tables(financial, events, cases, cfg)
    study = tables["origin_event_study.csv"]
    require(study.forecast_origin.is_unique and len(study) == 12, "Unique dates, not replicated MO rows")
    require(tables["eligibility_summary.csv"].to_dict("records") == [
        dict(k=1,n_origins=12,n_eligible=6,n_positive=6,n_negative=0,n_unknown=6),
        dict(k=3,n_origins=12,n_eligible=4,n_positive=4,n_negative=0,n_unknown=8)], "Real date support")
    for k in [1, 3]:
        unknown = ~study[f"known_label_k{k}"]
        for column in [f"event_next_{k}m", f"event_count_next_{k}m", f"affected_municipality_count_next_{k}m", f"affected_share_next_{k}m"]:
            require(study.loc[unknown, column].isna().all(), "Unknown labels/counts must remain NA")
        require(not study.loc[~unknown, f"event_count_complete_k{k}"].any(), "Incomplete monitored coverage is disclosed")
    require(len(tables["permutation_results.csv"]) == 8, "Eight planned primary tests")
    perm = tables["permutation_results.csv"]
    require(perm[["raw_exact_p","holm_adjusted_p","difference_in_means","rank_biserial"]].isna().all().all(), "No fabricated inferential statistics")
    require(perm.permutations_evaluated.eq(0).all() and perm.permutations_possible.eq(1).all(), "Degenerate assignments are not executed tests")
    loo = tables["leave_one_event_out.csv"]
    require(len(loo) == 84 and loo.n_negative.eq(0).all() and loo.removal_status.eq("REMOVED").all(), "LOO preserves original labels and deletes actual windows")
    require(loo[["effect_sign", "difference_in_means", "effect_ratio_to_full", "one_event_driven"]].isna().all().all(), "No unsupported stability claim")
    windows = tables["event_window_table.csv"]
    require(len(windows) == 18 and windows.financial_available.all(), "Six real event windows, three fixed leads each")
    for row in windows.to_dict("records"):
        require(month(row["forecast_origin"]) + row["lead_months"] == month(row["onset_date"]), "Calendar event lead")
    # Order replay reuses the same saved observations, never invented values.
    replay = calculate_tables(financial.iloc[::-1], events.iloc[::-1], cases.iloc[::-1], cfg)
    for name, frame in tables.items():
        pd.testing.assert_frame_equal(frame, replay[name], check_dtype=True, check_exact=True)
    if not args.source_only:
        output, audit = ROOT/cfg["output_dir"], ROOT/cfg["audit_dir"]
        manifest = read_json(output/"run_manifest.json")
        for name, frame in tables.items():
            compare_saved(frame, output/name)
            require(sha256(output/name) == manifest["artifact_sha256"][name], "Saved output SHA mismatch")
            require(sha256(output/name) == sha256(audit/name), "Audit table differs from output")
        require(sha256(output/"run_manifest.json") == sha256(audit/"run_manifest.json"), "Manifest mirrors differ")
        require(manifest["source_sha256"] == source["source_sha256"], "Frozen input manifest hashes")
        if args.complete:
            require(manifest["status"] == "E08d real financial early-warning diagnostic complete.", "Run not finalized")
            require(sha256(ROOT/cfg["report_path"]) == manifest["report"]["sha256"], "Report SHA")
            for name, digest in manifest["artifact_sha256"].items():
                require(sha256(output/name) == digest, "Postprocessing artifact SHA")
            for name, digest in manifest["code_sha256"].items():
                require(sha256(ROOT/name) == digest, "Code provenance changed")
            require(len(list((output/"figures").glob("*.png"))) <= 2, "Figure count")
    preserved = preserve(cfg)
    result = dict(status="PASS", stage="SOURCE_ONLY" if args.source_only else "COMPLETE" if args.complete else "REAL_ARTIFACTS",
        real_data_only=True, synthetic_inputs=0, classifier_fits=0, threshold_searches=0,
        source_counts=source["source_counts"], eligibility_summary=tables["eligibility_summary.csv"].to_dict("records"),
        checked_tables={name:len(t) for name,t in tables.items()}, source_parity_financial_values=120,
        preserved_hash_checks=preserved, deterministic_reversal_replay="PASS", git_diff_check="PASS",
        runtime_seconds=round(time.perf_counter()-started,6),
        command=(Path(sys.executable).resolve().relative_to(ROOT).as_posix() if Path(sys.executable).resolve().is_relative_to(ROOT) else "python")+
            " scripts/check_real_financial_early_warning.py --config "+args.config +
            (" --source-only" if args.source_only else " --complete" if args.complete else ""))
    target = ROOT/"outputs/e08d_checks"/("source_validation.json" if args.source_only else "completion_validation.json" if args.complete else "real_validation.json")
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False))


if __name__ == "__main__":
    main()
