"""E06b bounded acquisition and point-in-time news features; no model fitting."""
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

from sberforecast.data import load_data, sha256_file
from sberforecast.news_audit import assert_feature_keys, document_audit, feature_coverage, origin_coverage
from sberforecast.news_features import FEATURE_COLUMNS, build_news_features, feature_dictionary
from sberforecast.news_geo import build_geography_dictionary
from sberforecast.news_normalize import normalize_documents, canonical_events
from sberforecast.news_sources import collect_sources
from sberforecast.news_sources import reviewed_cbr_pdf_events


def project_path(value: str) -> Path:
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("E06b paths must remain inside this project")
    return path


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def pilot_samples(config: dict, dictionary: pd.DataFrame) -> pd.DataFrame:
    source = project_path(config["comparison"]["source_dir"])
    original = yaml.safe_load((source / "config_resolved.yaml").read_text(encoding="utf-8"))
    for bound in ["first_origin", "last_origin"]:
        if config["comparison"][bound] != original["backtest"][bound]:
            raise ValueError(f"News pipeline alters E01 {bound}")
    ids = [str(uid) for uid in json.loads((source / "sample_ids.json").read_text(encoding="utf-8"))]
    if len(ids) != 64 or len(set(ids)) != 64:
        raise ValueError("Expected the unchanged E01 64-municipality pilot")
    origins = pd.period_range(config["comparison"]["first_origin"], config["comparison"]["last_origin"], freq="M")
    if len(origins) != 12:
        raise ValueError("Expected E01's 12 forecast origins")
    samples = pd.DataFrame([dict(municipality_id=uid, forecast_origin=str(origin.end_time.normalize().date()))
                            for origin in origins for uid in ids])
    mapping = dictionary[["municipality_id", "region_id", "region"]].drop_duplicates("municipality_id")
    result = samples.merge(mapping, on="municipality_id", how="left", validate="many_to_one", sort=False)
    if result.region_id.isna().any():
        raise ValueError("Pilot contains unmapped municipalities")
    return result


def collection_entries(config: dict) -> list[dict]:
    entries = list(config["sources"]["entries"])
    for month in pd.period_range(config["period"]["start"], config["period"]["end"], freq="M"):
        for day in config["sources"]["archive_days"]:
            date = f"{month.year}/{month.month:02d}/{int(day):02d}"
            entries.append(dict(source="lenta", url=f"https://lenta.ru/{date}/", period=f"{month.start_time.date()}..{month.end_time.date()}",
                role="news_archive", terms_note="Public daily archive; partial fixed-day sample, historical headline version unconfirmed"))
    return entries


def collect(config: dict, download: bool) -> None:
    settings = config["sources"]
    raw_dir = project_path(settings["raw_dir"])
    corpus_path = project_path(settings["corpus_path"])
    if corpus_path.exists():
        raise FileExistsError("Corpus already frozen; use another configuration/cache for a new collection")
    docs, catalog = collect_sources(collection_entries(config), raw_dir,
        download=download, max_requests=settings["max_requests"], max_bytes=settings["max_bytes"],
        max_file_bytes=settings["max_file_bytes"], timeout=settings["timeout"])
    selected = []
    for url, group in pd.DataFrame(docs).groupby("retrieval_source_url", sort=True) if docs else []:
        records = group.sort_values("source_url", kind="stable").to_dict("records")
        if records[0]["source"] == "lenta":
            selected.extend(records[:settings["max_articles_per_archive_day"]])
        elif records[0]["source"] == "cbr":
            # The public calendar also links speeches and later discussions.
            # E06b's compact official sample consists of actual rate releases.
            rate_releases = [record for record in records if "/press/pr/" in record["source_url"]]
            selected.extend(rate_releases[:settings["max_cbr_articles"]])
        else:
            selected.extend(records)
    docs = selected
    article_entries = [dict(source=doc["source"], url=doc["source_url"],
        period=f"{config['period']['start']}..{config['period']['end']}", role="selected_article")
        for doc in docs if doc["source"] in {"cbr", "lenta"}]
    used_requests = sum(entry.get("requests_consumed", int(entry.get("retrieval_mode") == "downloaded")) for entry in catalog)
    used_bytes = sum(entry.get("bytes_accounted", entry.get("file_size") or 0) for entry in catalog)
    remaining_requests = settings["max_requests"] - used_requests
    remaining_bytes = settings["max_bytes"] - used_bytes
    if article_entries and remaining_requests > 0 and remaining_bytes > 0:
        articles, article_catalog = collect_sources(article_entries, raw_dir,
            download=download, max_requests=remaining_requests, max_bytes=remaining_bytes,
            max_file_bytes=settings["max_file_bytes"], timeout=settings["timeout"])
        docs.extend(articles)
        catalog.extend(article_catalog)
    reused_docs, reused_catalog = reviewed_cbr_pdf_events(project_path(settings["reviewed_cbr_pdf_dir"]))
    docs.extend(reused_docs)
    catalog.extend(reused_catalog)
    # Collection can inspect indexes without promoting current headlines to
    # confirmed historical text. Document article acquisition is bounded and
    # follows only actually discovered URLs in a separate, reviewable catalog.
    save_json(corpus_path, docs)
    save_json(raw_dir / "collection_catalog.json", catalog)
    print(f"Collected snapshots={len(docs)}, source requests={len(catalog)}", flush=True)


def synthetic_smoke_documents() -> list[dict]:
    return [dict(source="cbr", source_url="https://example.invalid/synthetic-news-smoke", title="Банк России принял решение о ключевой ставке",
        snippet="Синтетическая публикация для технической проверки", published_at="2023-12-15T13:30:00+03:00",
        available_at="2023-12-15T13:30:00+03:00", retrieved_at="2023-12-15T14:00:00+03:00",
        historical_version_confirmed=True, availability_evidence="Synthetic offline smoke fixture; not real source")]


def build(config: dict, config_path: Path, smoke: bool) -> None:
    started, clock = datetime.now(timezone.utc).isoformat(), time.perf_counter()
    output = project_path(config["smoke_output_dir"] if smoke else config["output_dir"])
    if output.exists():
        raise FileExistsError(f"Output exists; choose a unique output_dir: {output}")
    dictionary = build_geography_dictionary(load_data(project_path(config["data"]["path"]), config["data"]["category"]))
    samples = pilot_samples(config, dictionary)
    if smoke:
        samples = samples.iloc[:2].copy()
        raw = synthetic_smoke_documents()
    else:
        raw = json.loads(project_path(config["sources"]["corpus_path"]).read_text(encoding="utf-8"))
    period_start, period_end = config["period"]["start"], config["period"]["end"]
    normalized = normalize_documents(raw, dictionary)
    pub_days = pd.to_datetime(normalized.published_at, utc=True).dt.tz_convert("Europe/Moscow").dt.strftime("%Y-%m-%d")
    included = pub_days.between(period_start, period_end).fillna(False)
    excluded = normalized.loc[~included].copy()
    normalized = normalized.loc[included].reset_index(drop=True)
    normalized["is_canonical"] = ~normalized.canonical_event_id.duplicated()
    if not normalized.empty:
        normalized["duplicate_count"] = normalized.canonical_event_id.map(normalized.groupby("canonical_event_id").document_id.nunique()).astype(int)
    features = build_news_features(normalized, samples, history_start=config["availability"]["history_start"])
    assert_feature_keys(features, samples)
    # Completeness is source provenance, not the presence of a numeric count.
    features["source_archive_complete"] = bool(config["availability"]["corpus_is_complete_archive"])
    features["historical_version_evidence_available"] = [
        bool((normalized.availability_status.str.startswith("confirmed_") &
            pd.to_datetime(normalized.available_at, utc=True).le(pd.Timestamp(origin).tz_localize("Europe/Moscow").tz_convert("UTC"))).any())
        for origin in features.forecast_origin]
    output.mkdir(parents=True)
    normalized.to_csv(output / "documents.csv.gz", index=False, compression="gzip")
    canonical_events(normalized).to_csv(output / "events.csv.gz", index=False, compression="gzip")
    excluded.to_csv(output / "excluded_documents.csv.gz", index=False, compression="gzip")
    samples.to_csv(output / "forecast_cases.csv.gz", index=False, compression="gzip")
    features.to_csv(output / "news_features.csv.gz", index=False, compression="gzip")
    feature_dictionary().to_csv(output / "feature_dictionary.csv", index=False)
    dictionary.to_csv(output / "geography_dictionary.csv.gz", index=False, compression="gzip")
    from sberforecast.news_geo import AUDITED_REGION_ALIASES, AUDITED_PERSON_COLLISION_ALIASES, PERSON_COLLISION_QUALIFIER_PATTERNS
    from sberforecast.news_topics import TOPIC_RULES, EVENT_RULES
    save_json(output / "classification_rules.json", dict(region_aliases=AUDITED_REGION_ALIASES,
        person_collision_aliases=AUDITED_PERSON_COLLISION_ALIASES,
        person_collision_qualifier_patterns=PERSON_COLLISION_QUALIFIER_PATTERNS,
        topic_rules=TOPIC_RULES, event_rules=EVENT_RULES, confidence_interpretation="heuristic match strength, not probability"))
    feature_coverage(features, FEATURE_COLUMNS).to_csv(output / "feature_coverage.csv", index=False)
    origins = origin_coverage(features)
    origins.to_csv(output / "origin_coverage.csv.gz", index=False, compression="gzip")
    origin_summary = origins.groupby("forecast_origin").agg(cases=("municipality_id", "size"),
        regional_cases=("has_regional_30d", "sum"), national_only_cases=("only_national_30d", "sum"),
        no_news_cases=("no_observed_news_30d", "sum"), missing_window_cases=("news_window_missing", "sum"))
    origin_summary.to_csv(output / "origin_summary.csv")
    audit = document_audit(normalized, period_start, period_end)
    audit.update(feature_rows=len(features), feature_count=len(FEATURE_COLUMNS),
        pilot_municipalities=features.municipality_id.nunique(), pilot_origins=features.forecast_origin.nunique(),
        excluded_outside_period_or_missing_publication=len(excluded), synthetic_smoke=smoke,
        historical_documents_at_last_origin=int((pd.to_datetime(normalized.available_at,utc=True) <=
            pd.Timestamp(samples.forecast_origin.max()).tz_localize("Europe/Moscow").tz_convert("UTC")).sum()))
    save_json(output / "coverage_audit.json", audit)
    (output / "config_resolved.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    inputs = [config_path, project_path(config["data"]["path"]),
        project_path(config["comparison"]["source_dir"]) / "sample_ids.json",
        project_path(config["comparison"]["source_dir"]) / "config_resolved.yaml"]
    if not smoke:
        inputs += [project_path(config["sources"]["corpus_path"])]
        catalog_path = project_path(config["sources"]["raw_dir"]) / "collection_catalog.json"
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        save_json(output / "source_catalog.json", catalog)
        inputs.append(catalog_path)
        for entry in catalog:
            for field in ("file", "text_file", "publication_evidence_file"):
                if entry.get(field):
                    candidate = Path(entry[field]).resolve()
                    if candidate.is_relative_to(ROOT) and candidate.is_file() and candidate not in inputs:
                        inputs.append(candidate)
    code_files = sorted((ROOT / "src/sberforecast").glob("news_*.py")) + [Path(__file__)]
    manifest = dict(experiment=config["experiment"], started_at=started, completed_at=datetime.now(timezone.utc).isoformat(),
        runtime_seconds=time.perf_counter()-clock, seed=config["seed"], synthetic_smoke=smoke,
        command=subprocess.list2cmdline([sys.executable]+sys.argv), python=sys.version,
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"],cwd=ROOT,text=True).strip(),
        git_status=subprocess.check_output(["git", "status", "--short"],cwd=ROOT,text=True),
        versions={d.metadata["Name"]:d.version for d in importlib.metadata.distributions()},
        inputs={str(p.relative_to(ROOT)):sha256_file(p) for p in inputs},
        code={str(p.relative_to(ROOT)):sha256_file(p) for p in code_files},
        artifacts={p.name:dict(sha256=sha256_file(p),bytes=p.stat().st_size) for p in sorted(output.iterdir()) if p.is_file()},
        limitations=["Bounded corpus; incomplete archives", "Unverified current text available only at retrieval",
            "Dictionary aliases include source historical names; municipality version disambiguation is conservative",
            "Directions are lexical rules, not objective sentiment or causal effects", "No E07 classifier, no forecasting metrics or shock labels"])
    save_json(output / "run_manifest.json", manifest)
    print(json.dumps(dict(output=str(output.relative_to(ROOT)), **audit), ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/news_events.yaml")
    parser.add_argument("--stage", choices=["smoke", "collect", "build"], default="build")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    path = project_path(args.config)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if args.download and args.stage != "collect":
        parser.error("--download is only allowed with --stage collect")
    if args.stage == "collect":
        collect(config, args.download)
    else:
        build(config, path, smoke=args.stage == "smoke")


if __name__ == "__main__":
    main()
