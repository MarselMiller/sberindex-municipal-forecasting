"""Read-only E06b audit, optionally acquire missing original CBR rate releases."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sberforecast.early_warning_news import (
    CALENDAR_URL, DECISION_ARCHIVE_URL, RetrievalBudget, archive_release_evidence,
    audit_cached_news, extract_cbr_key_rate_decision, fetch_official,
    read_verified_cached, sha256_bytes,
)


def project_path(value):
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("E07a paths must stay inside the project")
    return path


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def git_state():
    result = {}
    for key, args in [("git_commit", ["git", "rev-parse", "HEAD"]),
                      ("git_status", ["git", "status", "--short"])]:
        run = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
        result[key] = run.stdout.strip() if run.returncode == 0 else None
    return result


def run(config_path: Path, *, download=False, output_override=None):
    started, clock = datetime.now(timezone.utc).isoformat(), time.perf_counter()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    news = config["news"]
    output = project_path(output_override or news["audit_output_dir"])
    raw_dir = project_path(news["raw_dir"])
    if output.exists():
        raise FileExistsError(f"Choose a unique audit output directory: {output}")
    if raw_dir != project_path("data/external/news_raw/e07a_key_rate_v1"):
        raise ValueError("New E07a downloads must use the isolated e07a_key_rate_v1 cache")
    if news["max_concurrent_requests"] != 1:
        raise ValueError("This CLI implements sequential retrieval only")
    old_corpus_path = project_path(news["old_corpus"])
    old_catalog_path = project_path(news["old_catalog"])
    documents_path = project_path(config["reference"]["e06b_dir"]) / "documents.csv.gz"
    inputs = {str(path.relative_to(ROOT)): sha256_bytes(path.read_bytes())
              for path in [config_path, old_corpus_path, old_catalog_path, documents_path,
                           Path(__file__), ROOT / "src/sberforecast/early_warning_news.py"]}
    corpus = json.loads(old_corpus_path.read_text(encoding="utf-8"))
    old_catalog = json.loads(old_catalog_path.read_text(encoding="utf-8"))
    documents = pd.read_csv(documents_path)
    sources = [dict(entry, e07a_retrieval_mode="unchanged_existing_catalog_entry") for entry in old_catalog]
    cached_by_url = {}
    # Existing index and all old article bodies are verified and read before HTTP.
    for index, entry in enumerate(old_catalog):
        if entry.get("source") != "cbr" or not str(entry.get("file", "")).endswith(".bin"):
            continue
        cached = read_verified_cached(entry)
        if cached:
            text, meta = cached
            cached_by_url[entry["source_url"]] = (text, meta)
            sources[index] = meta
            inputs[str(Path(meta["file"]).relative_to(ROOT))] = meta["sha256"]
    if CALENDAR_URL not in cached_by_url:
        raise ValueError("Verified existing CBR calendar cache required before any acquisition")
    calendar_text, calendar_meta = cached_by_url[CALENDAR_URL]
    archive = archive_release_evidence(calendar_text, CALENDAR_URL,
                                       calendar_meta["sha256"], calendar_meta["retrieved_at"])
    if not archive:
        raise ValueError("No dated original 2023–24 rate release links in cached calendar")
    budget = RetrievalBudget(max_requests=news["max_new_requests"], max_bytes=news["max_bytes"],
                             max_file_bytes=news["max_file_bytes"], timeout_seconds=news["timeout_seconds"],
                             max_retries=news["max_retries_per_url"],
                             stop_after_consecutive_blocks=news["stop_after_consecutive_blocks"])
    if download:
        # One official archive probe; its current text is not itself historical.
        text, meta = fetch_official(DECISION_ARCHIVE_URL, raw_dir, budget)
        meta["role"] = "official_archive_corrobation_probe_current_page_only"
        sources.append(meta)
        if text:
            extra = archive_release_evidence(text, DECISION_ARCHIVE_URL, meta["sha256"], meta["retrieved_at"])
            known = {row["release_url"] for row in archive}
            archive.extend(row for row in extra if row["release_url"] not in known)
        print(f"Archive original candidates={len(archive)}, existing cached releases="
              f"{sum(row['release_url'] in cached_by_url for row in archive)}", flush=True)
    decisions = []
    for evidence in sorted(archive, key=lambda row: row["archive_meeting_date"]):
        url = evidence["release_url"]
        cached = cached_by_url.get(url)
        if cached:
            text, meta = cached
        elif download and not budget.stopped:
            text, meta = fetch_official(url, raw_dir, budget)
            meta["role"] = "missing_original_rate_decision_release"
            sources.append(meta)
        else:
            text, meta = None, {"source_url": url, "parse_status": "not_downloaded", "error": "Explicit download required or stop bound reached"}
            sources.append(meta)
        if text:
            decision = extract_cbr_key_rate_decision(text, url, evidence,
                                                      meta["retrieved_at"], meta["sha256"])
        else:
            decision = {"source_url": url, "admitted": False, "reason": meta["parse_status"],
                        "archive_evidence": evidence, "retrieval_error": meta.get("error")}
        decisions.append(decision)
    origins = [str(month) for month in pd.period_range(config["reference"]["first_origin"],
                                                      config["reference"]["last_origin"], freq="M")]
    result = audit_cached_news(documents, corpus, sources, decisions, origins=origins)
    if len(result["snapshot_audit"]) != len(documents) or len(result["document_summary"]) != documents.document_id.nunique():
        raise AssertionError("Every old snapshot and document must appear in the audit")
    output.mkdir(parents=True)
    result["snapshot_audit"].to_csv(output / "snapshot_audit.csv", index=False)
    result["document_summary"].to_csv(output / "document_summary.csv", index=False)
    pd.DataFrame([{key: (json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value)
                   for key, value in row.items()} for row in decisions]).to_csv(output / "extracted_decisions.csv", index=False)
    save_json(output / "extracted_decisions.json", decisions)
    save_json(output / "core_trusted_records.json", result["core_trusted_records"])
    save_json(output / "archive_release_candidates.json", archive)
    save_json(output / "source_catalog.json", result["source_catalog"])
    save_json(output / "admission_rules.json", result["rules"])
    (output / "config_resolved.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    summary = {"old_snapshots": len(documents), "old_document_ids": documents.document_id.nunique(),
               "original_release_candidates": len(archive), "admitted_decision_cores": len(result["core_trusted_records"]),
               "refused_decisions": [dict(source_url=row["source_url"], reason=row["reason"]) for row in decisions if not row.get("admitted")],
               "e05c_overlapping_decisions": sum(row["e05c_overlap"] for row in result["core_trusted_records"]),
               "new_requests_total_ledger": budget.requests, "new_requests_this_command": len(budget.attempts),
               "new_bytes_total_ledger": budget.bytes_accounted, "stopped_after_blocks": budget.stopped,
               "strict_v2_historical_document_ids": int(result["document_summary"].strict_v2_historical_admitted.sum()),
               "v3_core_admitted_old_document_ids": int(result["document_summary"].v3_core_event_admitted.sum()),
               "independent_historical_SHA": False, "classifier_fitted": False, "features_built": False}
    save_json(output / "audit_summary.json", summary)
    # Re-check only inputs this narrow audit actually used, not the old full tree.
    for relative, expected in inputs.items():
        if sha256_bytes(project_path(relative).read_bytes()) != expected:
            raise AssertionError(f"Read-only input changed during audit: {relative}")
    artifacts = {str(path.relative_to(output)): sha256_bytes(path.read_bytes()) for path in output.iterdir() if path.is_file()}
    save_json(output / "run_manifest.json", {
        "experiment": "E07a_official_decision_core_news_audit", "started_at": started,
        "completed_at": datetime.now(timezone.utc).isoformat(), "runtime_seconds": time.perf_counter() - clock,
        "command": subprocess.list2cmdline([sys.executable, *sys.argv]), "download": download,
        "seed": config["seed"], "python": sys.version, **git_state(), "inputs": inputs,
        "artifacts": artifacts, "versions": {"pandas": pd.__version__, "PyYAML": importlib.metadata.version("PyYAML")},
        "new_request_attempts": budget.attempts, "summary": summary,
    })
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/early_warning_feasibility.yaml")
    parser.add_argument("--download", action="store_true", help="Explicitly acquire only missing original CBR rate releases and one official archive probe")
    parser.add_argument("--output-dir", help="Unique audit output override, inside the project")
    args = parser.parse_args()
    run(project_path(args.config), download=args.download, output_override=args.output_dir)


if __name__ == "__main__":
    main()
