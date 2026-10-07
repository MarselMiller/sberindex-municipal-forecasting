"""E07a audit and narrowly scoped official-archive trust for decision cores.

The trust rule is an explicit research assumption, not a historical SHA check.
Current article commentary, archive headings, later summaries and translations
are never backdated by this module. Existing E06b inputs are read only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
from zoneinfo import ZoneInfo

import pandas as pd

from .news_sources import MONTHS, _HTML, _cbr_file_date, _iso_date


MOSCOW = ZoneInfo("Europe/Moscow")
CALENDAR_URL = "https://www.cbr.ru/dkp/cal_mp/"
DECISION_ARCHIVE_URL = "https://www.cbr.ru/dkp/mp_dec/decision_key_rate/"
TRUST_RULE = "official_dated_original_key_rate_decision_core_only_no_independent_old_SHA"
CORE_RULE = "first_board_key_rate_decision_sentence_v1"
DATE_RE = re.compile(r"\b(\d{1,2})\s+(" + "|".join(MONTHS) + r")\s+(\d{4})\b", re.I)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def document_id(url: str) -> str:
    return sha256_bytes(("cbr\n" + url).encode("utf-8"))


def canonical_event_id(day: str) -> str:
    return "cbr_key_rate_decision_" + day.replace("-", "_")


def _day(text: str) -> str | None:
    match = DATE_RE.search(text)
    if not match:
        return None
    try:
        return datetime(int(match[3]), MONTHS[match[2].lower()], int(match[1])).date().isoformat()
    except ValueError:
        return None


def _official(url: str) -> bool:
    parts = urlsplit(url)
    return parts.scheme == "https" and parts.hostname in {"cbr.ru", "www.cbr.ru"} and not parts.username


def _release_url(url: str) -> bool:
    return _official(url) and urlsplit(url).path == "/press/pr/" and _cbr_file_date(url)[0] is not None


def _timestamp(value: str) -> datetime:
    dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return dt.replace(tzinfo=MOSCOW) if dt.tzinfo is None else dt


def archive_release_evidence(html: str, source_url: str, source_sha256: str,
                             retrieved_at: str) -> list[dict]:
    """Require a dated meeting row linking directly to an original release.

    Archive page retrieval/update dates are provenance, never publication dates.
    Filename dates corroborate the row; they do not substitute for its label.
    """
    if not _official(source_url) or urlsplit(source_url).path not in {
        "/dkp/cal_mp/", "/dkp/mp_dec/decision_key_rate/"
    }:
        return []
    rows = {}
    tree = _HTML(html)
    catalog_updates = []
    for node in tree.root.walk():
        if "last-update" in str(node.attrs.get("class", "")):
            value, precision = _iso_date(node.text())
            if value:
                catalog_updates.append({"value": _timestamp(value).isoformat(), "precision": precision,
                                        "evidence": node.text(), "scope": "archive_catalogue_page_only"})
    for node in tree.root.walk():
        if node.tag != "a" or not node.attrs.get("href"):
            continue
        url = urljoin(source_url, str(node.attrs["href"]))
        file_date = _cbr_file_date(url)[0] if _release_url(url) else None
        if not file_date or not "2023-01-01" <= file_date[:10] <= "2024-12-31":
            continue
        ancestor = node.parent
        context, archive_day = "", None
        for _ in range(8):
            if ancestor is None:
                break
            text = ancestor.text()
            dates = DATE_RE.findall(text)
            # A bounded, unique dated row. Entire multi-year pages cannot qualify.
            if len(text) <= 2500 and len(dates) == 1:
                candidate = _day(text)
                if candidate == file_date[:10]:
                    context, archive_day = text, candidate
                    break
            ancestor = ancestor.parent
        if archive_day is None:
            continue
        rows[url] = {"release_url": url, "archive_url": source_url,
                     "archive_sha256": source_sha256, "archive_retrieved_at": retrieved_at,
                     "archive_meeting_date": archive_day, "archive_link_text": node.text(),
                     "archive_row_evidence": context, "catalogue_updates": catalog_updates}
    return [rows[url] for url in sorted(rows)]


def extract_cbr_key_rate_decision(html: str, source_url: str, archive_evidence: dict,
                                  retrieved_at: str, source_sha256: str = "") -> dict:
    """Return an admitted core or a machine-readable refusal, without inference.

    The first decision sentence must agree with the dated original archive link,
    header, exact publication footer, filename and heading's action/rate. The
    exceptional meeting date can come from the matching dated archive row when
    the decision sentence states only an effective date. No Monday rule is used.
    """
    base = {"source_url": source_url, "document_id": document_id(source_url),
            "source_file_sha256": source_sha256, "retrieved_at": retrieved_at,
            "admitted": False, "extraction_rule": CORE_RULE, "trust_rule": TRUST_RULE}

    def refuse(reason, **details):
        return {**base, "reason": reason, **details}

    if not _release_url(source_url):
        return refuse("not_original_official_rate_release")
    if (archive_evidence.get("release_url") != source_url
            or not _official(str(archive_evidence.get("archive_url", "")))
            or urlsplit(str(archive_evidence.get("archive_url", ""))).path not in {
                "/dkp/cal_mp/", "/dkp/mp_dec/decision_key_rate/"}
            or not archive_evidence.get("archive_sha256")
            or not archive_evidence.get("archive_row_evidence")):
        return refuse("missing_dated_official_archive_link")
    tree = _HTML(html)
    bodies = [n for n in tree.root.walk() if "landing-text" in str(n.attrs.get("class", "")).split()]
    if len(bodies) != 1:
        return refuse("missing_or_ambiguous_article_core")
    body = bodies[0]
    heading = next((n.text() for n in tree.root.walk() if n.tag == "h1"), "")
    paragraphs = [n for n in body.walk() if n.tag == "p" and n.text()
                  and "note" not in str(n.attrs.get("class", "")).split()]
    if not paragraphs:
        return refuse("missing_decision_paragraph")
    paragraph = paragraphs[0].text()
    # 'б.п.' contains periods, so a generic sentence splitter is unsafe here.
    match = re.match(r"(Совет директоров Банка России\b.*?годовых\.)", paragraph, re.I)
    if not match or "принял решение" not in match[1].lower():
        return refuse("first_paragraph_not_board_rate_decision")
    core = match[1]
    action_match = re.search(r"принял решение\s+(сохранить|повысить|снизить)\s+ключевую ставку", core, re.I)
    rate_match = re.search(r"(?:до|на уровне)\s*(\d+(?:[,.]\d+)?)\s*%\s*годовых", core, re.I)
    if not action_match or not rate_match:
        return refuse("unrecognized_decision_action_or_rate", core_evidence=core)
    action = {"сохранить": "unchanged", "повысить": "increase", "снизить": "decrease"}[action_match[1].lower()]
    rate = float(rate_match[1].replace(",", "."))
    heading_rate = re.search(r"(\d+(?:[,.]\d+)?)\s*%", heading)
    heading_action = {"unchanged": "сохранил", "increase": "повысил", "decrease": "снизил"}[action]
    if (not heading_rate or float(heading_rate[1].replace(",", ".")) != rate
            or not any(verb in heading.lower() for verb in [heading_action, action_match[1].lower()])):
        return refuse("heading_core_action_or_rate_disagreement", core_evidence=core)
    notes = [n.text() for n in body.walk() if "note" in str(n.attrs.get("class", "")).split()]
    exact = []
    for note in notes:
        value, precision = _iso_date(note)
        if precision == "timestamp" and re.fullmatch(r"\d{2}\.\d{2}\.\d{4}\s+\d{2}:\d{2}:\d{2}", note):
            exact.append((value, note))
    if len(exact) != 1:
        return refuse("missing_or_ambiguous_exact_publication_footer", core_evidence=core)
    published_at = _timestamp(exact[0][0]).isoformat()
    publication_day = published_at[:10]
    headers = [n.text() for n in tree.root.walk() if "news-info-line_date" in str(n.attrs.get("class", "")).split()]
    header_days = {_day(x) for x in headers}
    filename = _cbr_file_date(source_url)[0]
    if (header_days != {publication_day} or _timestamp(filename) != _timestamp(published_at)
            or archive_evidence.get("archive_meeting_date") != publication_day
            or _day(archive_evidence["archive_row_evidence"]) != publication_day
            or not "2023-01-01" <= publication_day <= "2024-12-31"):
        return refuse("publication_archive_header_filename_disagreement", core_evidence=core)
    before_action = core[:action_match.start()]
    decision_day = _day(before_action)
    decision_date_rule = "explicit_date_in_decision_core"
    if decision_day is None:
        decision_day = archive_evidence["archive_meeting_date"]
        decision_date_rule = "dated_original_decision_release_and_archive_meeting_row"
    if decision_day != publication_day:
        return refuse("decision_publication_date_disagreement", core_evidence=core)
    if _timestamp(published_at) > _timestamp(retrieved_at):
        return refuse("publication_after_retrieval", core_evidence=core)
    content_updates = []
    catalog_updates = []
    for node in tree.root.walk():
        if node.tag == "meta" and str(node.attrs.get("property", "")).lower() == "article:modified_time":
            value, precision = _iso_date(str(node.attrs.get("content", "")))
            if value:
                content_updates.append({"value": value, "precision": precision, "evidence_scope": "article_metadata"})
        if "last-update" in str(node.attrs.get("class", "")):
            value, precision = _iso_date(node.text())
            if value:
                # A generic page/catalog footer is recorded separately; it does
                # not assert that the first decision sentence was revised.
                catalog_updates.append({"value": value, "precision": precision, "evidence_scope": "page_footer_scope_unknown"})
    if any(_timestamp(x["value"]) > _timestamp(published_at) for x in content_updates):
        return refuse("declared_later_article_revision_core_not_independently_attested",
                      core_evidence=core, content_updates=content_updates, catalog_updates=catalog_updates)
    effective_match = re.search(r"ключевую ставку\s+с\s+(\d{1,2}\s+(?:" + "|".join(MONTHS) + r")\s+\d{4})", core, re.I)
    effective_day = _day(effective_match[1]) if effective_match else None
    bps = re.search(r"на\s+(\d+)\s*б\.\s*п\.", core, re.I)
    return {**base, "admitted": True, "reason": "core_only_explicit_official_archive_trust",
            "canonical_event_id": canonical_event_id(decision_day), "title": heading,
            "core_evidence": core, "key_rate_percent": rate, "action": action,
            "change_basis_points": int(bps[1]) if bps else (0 if action == "unchanged" else None),
            "decision_date": decision_day, "decision_date_rule": decision_date_rule,
            "effective_date": effective_day,
            "effective_date_status": "explicit_in_core" if effective_day else "not_stated_in_core_no_inference",
            "publication_date": publication_day, "published_at": published_at,
            "publication_date_precision": "timestamp", "publication_footer_evidence": exact[0][1],
            "publication_header_evidence": " | ".join(headers), "archive_evidence": archive_evidence,
            "content_updates": content_updates, "catalog_updates": catalog_updates,
            "independent_historical_sha_available": False,
            "e05c_overlap": decision_day in {"2024-02-16", "2024-04-26", "2024-07-26", "2024-10-25"}}


def core_trusted_record(decision: dict) -> dict:
    """Convert one admitted core to the existing normalization input contract."""
    if not decision.get("admitted"):
        raise ValueError("A refused decision cannot become a historical record")
    evidence = ("Explicit official dated original archive-trust assumption, decision core only; "
                "no independent historical SHA; current commentary excluded. "
                + decision["publication_footer_evidence"] + "; "
                + decision["archive_evidence"]["archive_row_evidence"])
    return {"source": "cbr", "source_url": decision["source_url"],
            "document_id": decision["document_id"], "canonical_event_id": decision["canonical_event_id"],
            "published_at": decision["published_at"], "publication_date_precision": "timestamp",
            "publication_date_evidence": decision["publication_footer_evidence"],
            "event_date": decision["decision_date"], "updated_at": None,
            "retrieved_at": decision["retrieved_at"], "available_at": decision["published_at"],
            "title": decision["title"], "snippet": decision["core_evidence"],
            "historical_version_confirmed": True,
            "historical_version_status": "official_dated_original_decision_core_archive_trust",
            "availability_status": "confirmed_historical", "availability_evidence": evidence,
            "source_file_sha256": decision["source_file_sha256"],
            "retrieval_source_url": decision["source_url"], "document_kind": "decision_core",
            "decision_date": decision["decision_date"], "effective_date": decision["effective_date"],
            "key_rate_percent": decision["key_rate_percent"], "decision_action": decision["action"],
            "archive_trust_assumption": TRUST_RULE, "independent_historical_sha_available": False,
            "e05c_overlap": decision["e05c_overlap"]}


def read_verified_cached(entry: dict) -> tuple[str, dict] | None:
    """Read an existing successful file; a current SHA is provenance, not vintage."""
    if not entry.get("file") or not entry.get("sha256"):
        return None
    path = Path(entry["file"])
    if not path.is_file():
        return None
    payload = path.read_bytes()
    if sha256_bytes(payload) != entry["sha256"] or len(payload) != entry.get("file_size"):
        raise ValueError(f"Cached provenance mismatch: {path}")
    return payload.decode("utf-8-sig"), {**entry, "e07a_retrieval_mode": "read_only_existing_verified_cache"}


@dataclass
class RetrievalBudget:
    max_requests: int = 100
    max_bytes: int = 20 * 1024 * 1024
    max_file_bytes: int = 2 * 1024 * 1024
    timeout_seconds: float = 20
    max_retries: int = 1
    stop_after_consecutive_blocks: int = 2
    requests: int = 0
    bytes_accounted: int = 0
    consecutive_blocks: int = 0
    stopped: bool = False
    attempts: list[dict] = field(default_factory=list)

    def __post_init__(self):
        if not 0 < self.max_requests <= 100 or not 0 <= self.max_retries <= 1:
            raise ValueError("E07a permits at most 100 requests and one retry per URL")
        if self.max_file_bytes <= 0 or self.max_bytes <= 0 or self.timeout_seconds <= 0:
            raise ValueError("Positive retrieval bounds required")


class _OfficialRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not _official(newurl):
            raise ValueError("Redirect outside official CBR HTTPS domains refused")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch_official(url: str, raw_dir: str | Path, budget: RetrievalBudget,
                   *, opener=None) -> tuple[str | None, dict]:
    """Sequential bounded urllib GET with durable success/failure provenance.

    Retries are only for transient transport errors/5xx, never access blocks.
    A ledger enforces the total request and per-URL attempt bound across runs.
    Cache hits issue no HTTP. Redirects are disabled so every HTTP request is
    accounted; an unexpected redirect is saved as an access failure.
    """
    if not _official(url):
        raise ValueError("Only official CBR HTTPS retrieval is authorized")
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    stem = sha256_bytes(url.encode("utf-8"))
    target = raw_dir / (stem + ".bin")
    sidecar = raw_dir / (stem + ".bin.json")
    ledger = raw_dir / "requests.jsonl"
    history = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()] if ledger.exists() else []
    budget.requests = max(budget.requests, len(history))
    budget.bytes_accounted = max(budget.bytes_accounted, sum(x.get("bytes_accounted", 0) for x in history))
    trailing_blocks = 0
    for old_attempt in reversed(history):
        if not old_attempt.get("blocked"):
            break
        trailing_blocks += 1
    budget.consecutive_blocks = max(budget.consecutive_blocks, trailing_blocks)
    budget.stopped = budget.stopped or budget.consecutive_blocks >= budget.stop_after_consecutive_blocks
    if target.is_file() and sidecar.is_file():
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        cached = read_verified_cached(meta)
        if cached:
            return cached
    url_attempts = sum(x.get("source_url") == url for x in history)

    class NoRedirect(_OfficialRedirect):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            if not _official(newurl):
                raise ValueError("Redirect outside official CBR HTTPS domains refused")
            return None

    client = opener or build_opener(NoRedirect())
    meta = {"source": "cbr", "source_url": url, "file": None, "sha256": None,
            "file_size": None, "http_status": None, "parse_status": "not_requested",
            "retrieval_mode": "new_E07a_bounded_urllib", "terms_note": "Public official original archive; redistribution not authorized.",
            "independent_historical_sha_available": False, "error": None}
    while url_attempts <= budget.max_retries and not budget.stopped:
        if budget.requests >= budget.max_requests or budget.bytes_accounted >= budget.max_bytes:
            meta.update(parse_status="budget_exhausted", error="Total request/byte bound reached")
            break
        started = datetime.now(timezone.utc).isoformat()
        size, blocked, transient = 0, False, False
        budget.requests += 1
        url_attempts += 1
        attempt = {"source_url": url, "attempt": url_attempts, "retrieved_at": started,
                   "http_status": None, "bytes_accounted": 0, "error": None}
        try:
            request = Request(url, headers={"User-Agent": "SberIndexResearch/0.1 bounded official archive audit"})
            with client.open(request, timeout=budget.timeout_seconds) as response:
                status = response.getcode()
                attempt["http_status"] = status
                meta.update(http_status=status, response_url=response.geturl(),
                            content_type=response.headers.get("Content-Type"),
                            http_last_modified=response.headers.get("Last-Modified"),
                            http_etag=response.headers.get("ETag"))
                if not _official(response.geturl()) or status != 200:
                    raise ValueError("Non-200 or outside-domain response refused")
                total_remaining = budget.max_bytes - budget.bytes_accounted
                read_limit = min(budget.max_file_bytes + 1, total_remaining)
                payload = response.read(read_limit)
                size = len(payload)
                # Never spend a sentinel byte beyond the total bound. An
                # unknown body filling the final allowance is refused because
                # its completion cannot be verified without another byte.
                declared_length = response.headers.get("Content-Length")
                declared_length = int(declared_length) if str(declared_length).isdigit() else None
                if size > budget.max_file_bytes or (size == total_remaining and declared_length != size):
                    raise ValueError("Response exceeds file or total byte bound")
            target.write_bytes(payload)
            meta.update(file=str(target.resolve()), sha256=sha256_bytes(payload), file_size=size,
                        retrieved_at=started, parse_status="downloaded_not_audited", error=None)
            text = payload.decode("utf-8-sig")
        except (OSError, HTTPError, URLError, ValueError, UnicodeError) as exc:
            text = None
            status = exc.code if isinstance(exc, HTTPError) else attempt["http_status"]
            attempt["http_status"] = status
            error = f"{type(exc).__name__}: {exc}"
            lowered = error.lower()
            blocked = status in {301, 302, 303, 307, 308, 401, 403, 429} or any(s in lowered for s in (
                "10013", "permission denied", "forbidden", "access denied", "captcha"))
            transient = isinstance(exc, (OSError, URLError)) and not blocked
            if isinstance(exc, HTTPError):
                transient = exc.code >= 500
            meta.update(retrieved_at=started, http_status=status,
                        parse_status="access_blocked" if blocked else "retrieval_failed", error=error)
            attempt["error"] = error
        budget.bytes_accounted += size
        budget.consecutive_blocks = budget.consecutive_blocks + 1 if blocked else 0
        budget.stopped = budget.consecutive_blocks >= budget.stop_after_consecutive_blocks
        attempt.update(bytes_accounted=size, blocked=blocked, parse_status=meta["parse_status"])
        budget.attempts.append(attempt)
        with ledger.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(attempt, ensure_ascii=False) + "\n")
        meta.update(requests_consumed_for_url=url_attempts, global_requests_consumed=budget.requests,
                    global_bytes_accounted=budget.bytes_accounted, stopped_after_blocks=budget.stopped)
        sidecar.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        if text is not None:
            return text, meta
        if not transient:
            break
    if meta["parse_status"] == "not_requested":
        meta.update(parse_status="requests_previously_exhausted_or_stop", error="No further request authorized by bounds")
    return None, meta


def audit_cached_news(documents: pd.DataFrame, corpus: list[dict], catalog: list[dict],
                       decisions: list[dict], *, origins: list[str]) -> dict:
    """Explain all existing snapshots/docs and core admission without editing v2."""
    admitted = {x["source_url"]: x for x in decisions if x.get("admitted")}
    by_day = {x["decision_date"]: x for x in admitted.values()}
    raw_lookup = {(x["source_url"], x["title"], str(x["retrieved_at"])): x for x in corpus}
    snapshots = []
    for row in documents.to_dict("records"):
        url = row["source_url"]
        raw = next((x for key, x in raw_lookup.items() if key[0] == url and key[1] == row["title"]), {})
        pdf = row.get("historical_version_status") == "reviewed_original_official_pdf"
        pub = str(row.get("published_at", ""))
        pub_day = _timestamp(pub).astimezone(MOSCOW).date().isoformat() if pub and pub != "nan" else None
        core = admitted.get(url) or (by_day.get(pub_day) if pdf else None)
        kind = raw.get("document_kind", "reviewed_pdf_heading" if pdf else "article")
        updated = row.get("updated_at")
        has_update = pd.notna(updated) and bool(str(updated))
        later_update = has_update and bool(pub_day) and _timestamp(str(updated)) > _timestamp(pub)
        if pdf:
            v2_reason = "reviewed_original_pdf_under_existing_explicit_archive_trust"
        elif row["source"] == "lenta":
            v2_reason = ("current_index_title_historical_version_unknown" if kind == "index_title" else
                         "current_article_declares_revision_original_version_not_attested" if later_update else
                         "current_article_historical_version_unknown")
        elif kind == "index_title":
            v2_reason = "current_index_title_not_original_release_core"
        else:
            v2_reason = "dated_original_release_eligible_but_v2_retrieval_only_no_old_SHA"
        available = _timestamp(str(row["available_at"]))
        origin_count = sum(available <= pd.Period(origin, freq="M").end_time.normalize().tz_localize(MOSCOW).to_pydatetime() for origin in origins)
        canonical = core["canonical_event_id"] if core else row["canonical_event_id"]
        snapshots.append({"document_id": row["document_id"], "snapshot_id": row["snapshot_id"],
                          "source": row["source"], "source_url": url, "document_kind": kind,
                          "published_at": row.get("published_at"), "event_date_v2": row.get("event_date"),
                          "content_updated_at": row.get("updated_at"),
                          "content_updated_date_precision": row.get("updated_date_precision"),
                          "catalogue_updated_at": (" | ".join(x["value"] for x in core["archive_evidence"].get("catalogue_updates", [])) or None) if core else None,
                          "retrieved_at": row["retrieved_at"], "available_at_v2": row["available_at"],
                          "availability_status_v2": row.get("availability_status"),
                          "publication_date_precision": row.get("publication_date_precision"),
                          "strict_v2_reason": v2_reason, "strict_v2_admitted_origin_count": origin_count,
                          "strict_v2_historical_admitted": pdf, "v3_whole_snapshot_admitted": pdf,
                          "v3_core_event_admitted": bool(core), "canonical_event_id_v2": row["canonical_event_id"],
                          "canonical_event_id_v3": canonical, "e05c_overlap": bool(core and core["e05c_overlap"]),
                          "v3_reason": ("existing_PDF_unified_with_same_decision_not_independent" if pdf and core else
                                        "replacement_decision_core_only_archive_trust_original_snapshot_not_backdated" if core else
                                        "unchanged_strict_retrieval_only"),
                          "availability_evidence_v2": row.get("availability_evidence"),
                          "source_file_sha256": row.get("source_file_sha256")})
    snapshot_frame = pd.DataFrame(snapshots)
    summary = []
    for doc_id, group in snapshot_frame.groupby("document_id", sort=True):
        group = group.sort_values("retrieved_at", kind="stable")
        first, latest = group.iloc[0], group.iloc[-1]
        summary.append({"document_id": doc_id, "source": first["source"], "source_url": first["source_url"],
                        "snapshot_count": len(group), "document_kinds": " | ".join(sorted(set(group["document_kind"]))),
                        "strict_v2_reasons": " | ".join(sorted(set(group["strict_v2_reason"]))),
                        "strict_v2_historical_admitted": bool(group["strict_v2_historical_admitted"].any()),
                        "v3_core_event_admitted": bool(group["v3_core_event_admitted"].any()),
                        "v3_reasons": " | ".join(sorted(set(group["v3_reason"]))),
                        "canonical_event_id_v3": first["canonical_event_id_v3"],
                        "e05c_overlap": bool(group["e05c_overlap"].any()),
                        "published_at_first_snapshot": first["published_at"],
                        "published_at_latest_snapshot": latest["published_at"],
                        "publication_date_precision_first_snapshot": first["publication_date_precision"],
                        "publication_date_precision_latest_snapshot": latest["publication_date_precision"],
                        "content_updated_at_first_snapshot": first["content_updated_at"],
                        "content_updated_at_latest_snapshot": latest["content_updated_at"],
                        "catalogue_updated_at": latest["catalogue_updated_at"],
                        "retrieved_at_first_snapshot": first["retrieved_at"],
                        "retrieved_at_latest_snapshot": latest["retrieved_at"],
                        "available_at_v2_first_snapshot": first["available_at_v2"],
                        "available_at_v2_latest_snapshot": latest["available_at_v2"],
                        "availability_status_v2_first_snapshot": first["availability_status_v2"],
                        "availability_status_v2_latest_snapshot": latest["availability_status_v2"],
                        "availability_evidence_v2_first_snapshot": first["availability_evidence_v2"],
                        "availability_evidence_v2_latest_snapshot": latest["availability_evidence_v2"]})
    return {"snapshot_audit": snapshot_frame, "document_summary": pd.DataFrame(summary),
            "extracted_decisions": decisions,
            "core_trusted_records": [core_trusted_record(x) for x in decisions if x.get("admitted")],
            "source_catalog": catalog,
            "rules": {"archive_trust": TRUST_RULE, "extraction_rule": CORE_RULE,
                      "independent_historical_sha_available": False,
                      "whole_current_HTML_or_index_backdated": False, "lenta_policy_relaxed": False,
                      "date_only_policy": "end_of_day_Moscow", "effective_date_inference": "none",
                      "event_deduplication": "decision_date_across_release_PDF_translation_snapshots",
                      "e05c_overlap": "same_official_information_not_independent_evidence"}}
