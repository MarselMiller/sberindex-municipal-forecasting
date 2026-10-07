"""Conservative document versions and deterministic, auditable deduplication.

A publication date is not evidence that today's text existed on that date.
Unverified snapshots become available at retrieval. Historical availability
requires an explicit version confirmation and a description of its evidence.
All snapshots remain in the table; features deduplicate their own as-of prefix.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata

import pandas as pd

from .news_geo import resolve_geography
from .news_topics import classify_document

DOCUMENT_COLUMNS = [
    "document_id", "snapshot_id", "source", "source_url", "published_at",
    "event_date", "updated_at", "available_at", "retrieved_at", "title", "snippet",
    "geography_level", "region", "region_id", "municipality_id",
    "geography_confidence", "geography_rule", "topic", "event_type", "direction",
    "confidence", "classification_rule", "content_hash", "title_hash",
    "canonical_event_id", "duplicate_count", "is_canonical", "availability_status",
    "historical_version_status", "availability_evidence",
    "publication_date_precision", "updated_date_precision", "publication_date_evidence", "geography_evidence",
    "source_file_sha256", "retrieval_source_url",
]


def normalized_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold().replace("ё", "е")
    return " ".join(re.findall(r"\w+", text, flags=re.UNICODE))


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def timestamp(value: object) -> pd.Timestamp:
    if value is None or pd.isna(value) or str(value).strip() == "":
        return pd.NaT
    result = pd.Timestamp(value)
    if result.tzinfo is None:
        result = result.tz_localize("Europe/Moscow")
    return result.tz_convert("UTC")


def normalize_documents(documents: list[dict], dictionary: pd.DataFrame) -> pd.DataFrame:
    records = []
    for raw in documents:
        source, url = str(raw.get("source", "")), str(raw.get("source_url", ""))
        title, snippet = str(raw.get("title", "")).strip(), str(raw.get("snippet", "")).strip()
        if not source or not url or not title:
            raise ValueError("Document requires source, source_url and nonempty title")
        published, retrieved, updated = (timestamp(raw.get(name)) for name in
                                          ("published_at", "retrieved_at", "updated_at"))
        raw_update = raw.get("updated_at")
        raw_update_is_day = isinstance(raw_update, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw_update.strip()) is not None
        update_precision = raw.get("updated_date_precision") or ("day" if raw_update_is_day else "")
        if raw_update_is_day and pd.notna(updated):
            # A date alone never establishes a midnight update. Use the same
            # conservative day-end representation as the source readers.
            updated = (updated.tz_convert("Europe/Moscow").normalize()
                       + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)).tz_convert("UTC")
        if pd.isna(retrieved):
            raise ValueError("retrieved_at is required for every snapshot")
        if pd.notna(published) and published > retrieved:
            raise ValueError("Publication timestamp is after retrieval")
        if pd.notna(updated) and updated > retrieved:
            # Day-only update labels describe a day, not a verified future
            # clock time. A snapshot retrieved that same day remains retrieval
            # only; an actual later update day/time is inconsistent.
            same_local_day = updated.tz_convert("Europe/Moscow").date() == retrieved.tz_convert("Europe/Moscow").date()
            if not same_local_day or update_precision != "day":
                raise ValueError("Update timestamp is after retrieval")
        confirmed = raw.get("historical_version_confirmed") is True
        evidence = str(raw.get("availability_evidence", "")).strip()
        if confirmed:
            available = timestamp(raw.get("available_at"))
            if pd.isna(published) or pd.isna(available) or not evidence:
                raise ValueError("Historical version requires publication, availability and evidence")
            if available < published or available > retrieved:
                raise ValueError("Historical availability must be between publication and retrieval")
            if pd.notna(updated) and available < updated:
                raise ValueError("Updated text cannot be available before its update")
            status = "confirmed_historical" if pd.isna(updated) else "confirmed_updated_version"
        else:
            available, status = retrieved, "retrieval_only"
            evidence = evidence or "Current snapshot; historical text/version not independently confirmed"
        document_id = str(raw.get("document_id") or digest(source + "\n" + url))
        title_hash = digest(normalized_text(title))
        # Cross-source exact-title reprints on the same publication day. This
        # deliberately does not pretend to identify paraphrased reprints.
        publication_day = "unknown" if pd.isna(published) else str(published.tz_convert("Europe/Moscow").date())
        canonical = digest(title_hash + "\n" + publication_day)
        content_hash = digest(normalized_text(title) + "\n" + normalized_text(snippet))
        record = dict(document_id=document_id, snapshot_id=digest(document_id + content_hash + retrieved.isoformat()),
            source=source, source_url=url, title=title, snippet=snippet, published_at=published,
            retrieved_at=retrieved, updated_at=updated, event_date=timestamp(raw.get("event_date")),
            updated_date_precision=update_precision,
            available_at=available, content_hash=content_hash, title_hash=title_hash,
            canonical_event_id=canonical, availability_status=status,
            historical_version_status=str(raw.get("historical_version_status", "unknown")),
            availability_evidence=evidence)
        for name in ("publication_date_precision", "publication_date_evidence", "source_file_sha256", "retrieval_source_url"):
            record[name] = raw.get(name, "")
        record.update(resolve_geography(title, snippet, dictionary, source=source))
        record.update(classify_document(title, snippet, source=source))
        records.append(record)
    if not records:
        return pd.DataFrame(columns=DOCUMENT_COLUMNS)
    frame = pd.DataFrame(records).drop_duplicates("snapshot_id")
    frame = frame.sort_values(["available_at", "published_at", "document_id", "snapshot_id"], kind="stable")
    # A new title on a later version must not create a new economic event.
    first_ids = frame.drop_duplicates("document_id").set_index("document_id").canonical_event_id
    frame["canonical_event_id"] = frame.document_id.map(first_ids)
    counts = frame.groupby("canonical_event_id").document_id.nunique()
    frame["duplicate_count"] = frame.canonical_event_id.map(counts).astype(int)
    frame["is_canonical"] = ~frame.canonical_event_id.duplicated()
    return frame.reindex(columns=DOCUMENT_COLUMNS).reset_index(drop=True)


def canonical_events(documents: pd.DataFrame) -> pd.DataFrame:
    """Corpus-level summary only; never use its future counts in old features."""
    return documents.loc[documents.is_canonical.astype(bool)].copy().reset_index(drop=True)
