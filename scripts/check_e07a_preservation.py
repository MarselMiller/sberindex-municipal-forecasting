"""Small E07a preservation check for the inputs actually reused in this task."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
CHECK = ROOT / "outputs/e07a_checks/preservation_before.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selected_files() -> list[Path]:
    paths = []
    for directory in ("outputs/news_events_v1", "outputs/news_events_v2"):
        paths.extend(p for p in (ROOT / directory).rglob("*") if p.is_file())
    names = ["data/external/news_raw/corpus.json", "data/external/news_raw/collection_catalog.json",
             "data/external/news_raw/retrieval_manifest.json", "outputs/online_detection_v1/real_residuals.csv.gz",
             "outputs/online_detection_v1/selected_parameters.json", "outputs/online_detection_v1/config_resolved.yaml",
             "outputs/offline_detection_v1/real_breakpoints.csv", "outputs/macro_forecast_v1/source_ids.json",
             "outputs/macro_data_audit_v1/macro_table.csv.gz", "outputs/macro_data_audit_v1/source_catalog.json",
             "outputs/prophet_comparison_v1/sample_ids.json", "outputs/prophet_comparison_v1/config_resolved.yaml"]
    names += [f"reports/results/{name}" for name in ("E04a_online_detection.md", "E05a_macro_data_audit.md",
              "E05c_macro_forecast.md", "E06a_offline_detection.md", "E06b_news_events.md")]
    paths.extend(ROOT / name for name in names if (ROOT / name).is_file())
    catalog = json.loads((ROOT / names[1]).read_text(encoding="utf-8"))
    for entry in catalog:
        for field in ("file", "text_file", "publication_evidence_file"):
            if entry.get(field):
                path = Path(entry[field]).resolve()
                if path.is_relative_to(ROOT) and path.is_file():
                    paths.append(path)
    # Track all pre-existing Git files without reading secrets or source data.
    tracked = subprocess.check_output(["git", "ls-files", "src", "scripts", "configs", "tests", "pyproject.toml", "requirements.txt", ".gitignore"], cwd=ROOT, text=True).splitlines()
    paths.extend(ROOT / name for name in tracked if (ROOT / name).is_file())
    return sorted(set(paths))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["before", "after"])
    args = parser.parse_args()
    if args.stage == "before":
        if CHECK.exists():
            raise FileExistsError(CHECK)
        CHECK.parent.mkdir(parents=True, exist_ok=True)
        value = {"recorded_at": datetime.now(timezone.utc).isoformat(),
                 "protocol_sha256": digest(ROOT / "configs/early_warning_feasibility.yaml"),
                 "scope": "Only reused E04/E05/E06 inputs, old news outputs, cached payloads and old tracked code; not a full old-output recheck",
                 "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                 "files": {str(p.relative_to(ROOT)): digest(p) for p in selected_files()}}
        CHECK.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"stage": "before", "files": len(value["files"]), "protocol_sha256": value["protocol_sha256"]}))
    else:
        before = json.loads(CHECK.read_text(encoding="utf-8"))
        changed = [name for name, sha in before["files"].items() if not (ROOT/name).is_file() or digest(ROOT/name) != sha]
        protocol_equal = digest(ROOT / "configs/early_warning_feasibility.yaml") == before["protocol_sha256"]
        result = {"checked_at": datetime.now(timezone.utc).isoformat(), "checked_files": len(before["files"]),
                  "changed": changed, "protocol_unchanged_since_before_counts": protocol_equal,
                  "passed": not changed and protocol_equal}
        (CHECK.parent / "preservation_after.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False))
        if not result["passed"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
