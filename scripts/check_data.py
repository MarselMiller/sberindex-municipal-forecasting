"""Read-only data preflight. Uses the standard library; never downloads or fits."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_COLUMNS = {
    "dataset_id", "file_or_group", "description", "used_in", "local_source",
    "authoritative_source", "source_url", "license_or_terms",
    "attribution_required", "redistribution_status", "publication_decision",
    "repository_destination", "size_bytes", "sha256", "notes",
}
REDISTRIBUTION = {
    "CONFIRMED_ALLOWED", "CONFIRMED_NOT_ALLOWED", "UNCLEAR", "OWN_GENERATED",
    "DERIVED_SAFE", "DERIVED_RESTRICTED",
}
DECISIONS = {
    "PUBLISH", "PUBLISH_METADATA_ONLY", "DOWNLOAD_SCRIPT_ONLY",
    "DO_NOT_PUBLISH", "NEEDS_CONFIRMATION",
}


def local_path(root: Path, name: str) -> Path:
    """Reject absolute paths, traversal and symlinks leaving the checkout."""
    if not name or "\\" in name or ":" in name:
        raise ValueError("expected a repository-relative path")
    path = Path(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("expected a repository-relative path")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("path leaves the checkout")
    return resolved


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_status(path: Path, schema: dict) -> str:
    opener = gzip.open if path.suffix == ".gz" else open
    keys: set[tuple[str, ...]] = set()
    count = 0
    with opener(path, "rt", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        if schema.get("columns") and columns != schema["columns"]:
            return "FAIL (columns)"
        if not set(schema.get("required_columns", [])).issubset(columns):
            return "FAIL (required columns)"
        for row in reader:
            count += 1
            if None in row or any(value is None for value in row.values()):
                return "FAIL (row width)"
            if any(not row.get(name, "").strip() for name in schema.get("nonempty_columns", [])):
                return "FAIL (empty required value)"
            for name, expected in schema.get("fixed_values", {}).items():
                if row.get(name) != expected:
                    return "FAIL (fixed value)"
            for name in schema.get("numeric_columns", []):
                value = row.get(name, "")
                if value == "" and name in schema.get("nullable_columns", []):
                    continue
                try:
                    number = float(value)
                except (ValueError, TypeError):
                    return "FAIL (numeric column)"
                if not math.isfinite(number):
                    return "FAIL (nonfinite value)"
                if name in schema.get("nonnegative_columns", []) and number < 0:
                    return "FAIL (negative value)"
            if schema.get("unique_key"):
                key = tuple(row[name] for name in schema["unique_key"])
                if key in keys:
                    return "FAIL (duplicate key)"
                keys.add(key)
    if schema.get("rows") is not None and count != schema["rows"]:
        return "FAIL (row count)"
    return "PASS"


def schema_status(root: Path, path: Path, contract: dict) -> str:
    schema = contract.get("schema", {})
    if isinstance(schema, str):
        schema = json.loads(local_path(root, schema).read_text(encoding="utf-8"))
    kind = contract.get("format")
    if kind == "csv":
        return csv_status(path, schema)
    if kind == "json":
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        expected_type = schema.get("type")
        if expected_type == "object" and not isinstance(payload, dict):
            return "FAIL (JSON type)"
        if expected_type == "array" and not isinstance(payload, list):
            return "FAIL (JSON type)"
        if schema.get("required_keys") and (
            not isinstance(payload, dict)
            or not set(schema["required_keys"]).issubset(payload)
        ):
            return "FAIL (JSON keys)"
        return "PASS"
    if kind == "parquet":
        # Optional: the public profile requires no third-party dependencies.
        try:
            import pyarrow.parquet as pq
        except ImportError:
            return "NOT CHECKED (pyarrow unavailable)"
        metadata = pq.read_metadata(path)
        names = metadata.schema.names
        if not set(schema.get("required_columns", [])).issubset(names):
            return "FAIL (required columns)"
        if schema.get("rows") is not None and metadata.num_rows != schema["rows"]:
            return "FAIL (row count)"
        return "PASS"
    return "NOT CHECKED (no schema validator)"


def check_contract(root: Path, contract: dict) -> dict:
    result = {
        "dataset_id": contract["dataset_id"], "filename": contract["path"],
        "status": "MISSING", "schema": "NOT CHECKED", "hash": "NOT CHECKED",
        "size": "NOT CHECKED",
    }
    try:
        path = local_path(root, contract["path"])
        if not path.is_file():
            return result
        result["status"] = "FOUND"
        expected_hash = contract.get("sha256")
        result["hash"] = (
            "PASS" if expected_hash and file_hash(path) == expected_hash
            else "FAIL" if expected_hash else "NOT CHECKED (no expected hash)"
        )
        expected_size = contract.get("size_bytes")
        result["size"] = (
            "PASS" if expected_size is not None and path.stat().st_size == expected_size
            else "FAIL" if expected_size is not None else "NOT CHECKED"
        )
        result["schema"] = schema_status(root, path, contract)
    except (OSError, ValueError, TypeError, KeyError, csv.Error, UnicodeError):
        # Never print exceptions: they may include a private row or machine path.
        result["schema"] = "FAIL (unreadable file or invalid contract)"
    return result


def check_manifest(root: Path, contracts: list[dict]) -> list[str]:
    errors: list[str] = []
    with local_path(root, "data/manifest.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        reader = csv.DictReader(handle)
        if not MANIFEST_COLUMNS.issubset(reader.fieldnames or []):
            return ["manifest required columns"]
        rows = list(reader)
    ids: set[str] = set()
    contract_ids: set[str] = set()
    contract_paths: set[str] = set()
    for contract in contracts:
        if contract["dataset_id"] in contract_ids or contract["path"] in contract_paths:
            errors.append("duplicate contract ID/path")
        contract_ids.add(contract["dataset_id"])
        contract_paths.add(contract["path"])
    public_contracts = {item["path"]: item for item in contracts if "public" in item["profiles"]}
    published_paths: set[str] = set()
    for row in rows:
        dataset_id = row["dataset_id"]
        if not dataset_id or dataset_id in ids:
            errors.append("manifest duplicate/empty dataset_id")
        ids.add(dataset_id)
        if row["redistribution_status"] not in REDISTRIBUTION:
            errors.append("manifest redistribution status")
        if row["publication_decision"] not in DECISIONS:
            errors.append("manifest publication decision")
        if row["publication_decision"] == "PUBLISH":
            if row["redistribution_status"] in {
                "UNCLEAR", "CONFIRMED_NOT_ALLOWED", "DERIVED_RESTRICTED"
            }:
                errors.append("manifest unsafe publication")
            destination = row["repository_destination"]
            try:
                local_path(root, destination)
            except ValueError:
                errors.append("manifest destination")
            if destination in published_paths:
                errors.append("duplicate published destination")
            published_paths.add(destination)
            contract = public_contracts.get(destination)
            if not contract:
                errors.append("published dataset without public contract")
            elif dataset_id != contract["dataset_id"]:
                errors.append("manifest/contract dataset_id mismatch")
            elif row["sha256"] != contract["sha256"] or row["size_bytes"] != str(contract["size_bytes"]):
                errors.append("manifest/contract hash or size mismatch")
    if published_paths != set(public_contracts):
        errors.append("manifest/public contracts coverage")
    return errors


def run(root: Path, profile: str) -> tuple[dict, int]:
    try:
        catalog = json.loads(local_path(root, "data/preflight.json").read_text(encoding="utf-8"))
        contracts = catalog["datasets"]
        errors = check_manifest(root, contracts)
        selected = [item for item in contracts if profile in item["profiles"]]
        if not selected:
            errors.append("empty profile")
        results = [check_contract(root, item) for item in selected]
    except (OSError, ValueError, TypeError, KeyError, csv.Error, UnicodeError):
        return {"profile": profile, "manifest": "FAIL", "errors": ["invalid catalog or manifest"], "datasets": []}, 1
    ok = not errors and all(
        item["status"] == "FOUND"
        and item["schema"] == "PASS"
        and item["hash"] == "PASS"
        and item["size"] == "PASS"
        for item in results
    )
    return {
        "profile": profile, "manifest": "PASS" if not errors else "FAIL",
        "errors": errors, "datasets": results, "verdict": "PASS" if ok else "INCOMPLETE",
    }, 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["public", "research", "sources"], default="public")
    parser.add_argument("--json", action="store_true", help="Print status metadata only.")
    args = parser.parse_args()
    report, code = run(ROOT, args.profile)
    if args.json:
        print(json.dumps(report, ensure_ascii=True, indent=2))
    else:
        print(f"MANIFEST {report['manifest']} | profile={args.profile}")
        for item in report["datasets"]:
            print(f"{item['status']} {item['filename']} | schema={item['schema']} | hash={item['hash']} | size={item['size']}")
        for error in report["errors"]:
            print(f"FAIL {error}")
        print(f"VERDICT {report.get('verdict', 'INCOMPLETE')}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
