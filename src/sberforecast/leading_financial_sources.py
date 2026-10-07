"""Bounded official CBR acquisition for the two admitted E08 financial sources.

Raw documents and provenance stay in the ignored outputs tree. Historical
publication footers, effective dates and retrieval times remain separate.
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
import hashlib
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET
from urllib.parse import parse_qs, urljoin, urlsplit

from .macro_cbr import MONTHS, download_cbr
from .news_sources import _HTML

KEY_DECISION_CALENDAR = "https://www.cbr.ru/dkp/cal_mp/"
MOSCOW_OFFSET = timezone(timedelta(hours=3))
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_FILE_DATE = re.compile(r"^(\d{2})(\d{2})(\d{4})_\d{6}key\.htm$", re.I)
_DAY = re.compile(r"\b(\d{1,2})\s+(" + "|".join(MONTHS) + r")\s+(\d{4})\b", re.I)
_SCHEMA = "e08b-key-decisions-v1"


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _official(url: str) -> None:
    parts = urlsplit(url)
    if (parts.scheme != "https" or parts.hostname not in {"cbr.ru", "www.cbr.ru"}
            or parts.username or parts.password):
        raise ValueError("Only official public CBR HTTPS URLs are admitted.")


def _cache_path(path: Path) -> Path:
    path = Path(path).resolve()
    if not path.is_relative_to(_PROJECT_ROOT / "outputs"):
        raise ValueError("Downloaded CBR snapshots must stay in ignored outputs/.")
    return path


def cached_cbr_document(url: str, raw_path: Path, *, download: bool,
                        timeout: float = 25.0) -> tuple[bytes, dict]:
    """Read a hash-validated cache, or explicitly acquire a bounded public URL.

    No network request occurs for an existing valid cache or download=False.
    A cache without a provenance sidecar requires an explicitly enabled fetch.
    """
    _official(url)
    raw_path = _cache_path(raw_path)
    meta_path = raw_path.with_name(raw_path.name + ".metadata.json")
    if raw_path.exists() and meta_path.exists():
        content = raw_path.read_bytes()
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if metadata.get("source_url") != url or metadata.get("source_sha256") != _sha(content):
            raise ValueError("CBR cache source URL/hash mismatch.")
        return content, metadata
    if not download:
        raise FileNotFoundError(f"Missing verified CBR cache: {raw_path.name}")
    if not 0 < timeout <= 60:
        raise ValueError("CBR timeout must be bounded by 60 seconds.")
    content, response = download_cbr(url, timeout=timeout)
    metadata = {"source_url": url, **response,
                "source_sha256": _sha(content), "byte_count": len(content),
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "raw_path": raw_path.relative_to(_PROJECT_ROOT).as_posix(),
                "source_class": "A_trusted_official_archive_no_independent_historical_sha"}
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(content)
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return content, metadata


def _date(value: str) -> str | None:
    match = _DAY.search(value)
    if match:
        return datetime(int(match[3]), MONTHS[match[2].lower()], int(match[1])).date().isoformat()
    match = re.search(r"\b(\d{2})\.(\d{2})\.(\d{4})\b", value)
    if match:
        return datetime(int(match[3]), int(match[2]), int(match[1])).date().isoformat()
    return None


def key_decision_links(calendar_html: str) -> list[dict]:
    """Discover original release URLs for 2021-2024 from the official calendar."""
    tree = _HTML(calendar_html)
    rows = {}
    for node in tree.root.walk():
        if node.tag != "a" or not node.attrs.get("href"):
            continue
        url = urljoin(KEY_DECISION_CALENDAR, str(node.attrs["href"]))
        parts = urlsplit(url)
        filename = parse_qs(parts.query).get("file", [""])[0]
        match = _FILE_DATE.fullmatch(filename)
        if parts.path.lower() != "/press/pr/" or not match:
            continue
        _official(url)
        day = datetime(int(match[3]), int(match[2]), int(match[1])).date().isoformat()
        if not "2021-01-01" <= day <= "2024-12-31":
            continue
        context = node.parent
        evidence = None
        for _ in range(8):
            if context is None:
                break
            text = context.text()
            if len(text) <= 2500 and len(_DAY.findall(text)) == 1 and _date(text) == day:
                evidence = text
                break
            context = context.parent
        if evidence is None:
            raise ValueError("Calendar meeting date does not match linked original release.")
        rows[url] = {"source_url": url, "decision_date": day,
                     "archive_url": KEY_DECISION_CALENDAR,
                     "archive_row_evidence": evidence}
    result = sorted(rows.values(), key=lambda row: (row["decision_date"], row["source_url"]))
    if not result or len(result) > 40:
        raise ValueError("Unexpected bounded 2021-2024 decision calendar size.")
    if len({row["decision_date"] for row in result}) != len(result):
        raise ValueError("Duplicate original decision releases on a calendar date.")
    return result


def parse_key_decision(html: str, calendar_row: dict, provenance: dict) -> dict:
    """Extract actual footer precision and original decision core; no time inference."""
    tree = _HTML(html)
    bodies = [n for n in tree.root.walk()
              if "landing-text" in str(n.attrs.get("class", "")).split()]
    if len(bodies) != 1:
        raise ValueError("Missing/ambiguous CBR decision article.")
    body = bodies[0]
    paragraphs = [n.text() for n in body.walk() if n.tag == "p" and n.text()
                  and "note" not in str(n.attrs.get("class", "")).split()]
    core_match = re.match(r"(Совет директоров Банка России\b.*?годовых\.)", paragraphs[0] if paragraphs else "", re.I)
    if not core_match:
        raise ValueError("Original first paragraph is not a board key-rate decision.")
    core = core_match[1]
    action_match = re.search(r"принял решение\s+(сохранить|повысить|снизить)\s+ключевую ставку", core, re.I)
    rate_match = re.search(r"(?:до|на уровне)\s*(\d+(?:[,.]\d+)?)\s*%\s*годовых", core, re.I)
    if not action_match or not rate_match:
        raise ValueError("Unknown original key-rate action/rate.")
    day = calendar_row["decision_date"]
    headers = [n.text() for n in tree.root.walk()
               if "news-info-line_date" in str(n.attrs.get("class", "")).split()]
    if {_date(header) for header in headers} != {day}:
        raise ValueError("Decision header disagrees with official calendar.")
    notes = [n.text() for n in body.walk() if "note" in str(n.attrs.get("class", "")).split()]
    timestamps = [note for note in notes if re.fullmatch(r"\d{2}\.\d{2}\.\d{4}\s+\d{2}:\d{2}:\d{2}", note)]
    if len(timestamps) > 1:
        raise ValueError("Ambiguous historical publication footers.")
    announced_at = None
    precision = "date_only"
    if timestamps:
        parsed = datetime.strptime(timestamps[0], "%d.%m.%Y %H:%M:%S")
        if parsed.date().isoformat() != day:
            raise ValueError("Publication footer and calendar date disagree.")
        # A midnight footer alone does not corroborate a true midnight release.
        if parsed.time() != datetime.min.time():
            announced_at = parsed.replace(tzinfo=MOSCOW_OFFSET).isoformat()
            precision = "timestamp"
    explicit = re.search(r"ключевую ставку\s+с\s+(\d{1,2}\s+(?:" + "|".join(MONTHS) + r")\s+\d{4})", core, re.I)
    action = {"сохранить": "unchanged", "повысить": "increase", "снизить": "decrease"}[action_match[1].lower()]
    return {**calendar_row, "announced_at": announced_at, "time_precision": precision,
            "time_precision_reason": "exact_nonmidnight_official_footer" if announced_at else "intraday_time_uncorroborated_date_only",
            "publication_date": day, "publication_footer_evidence": timestamps[0] if timestamps else None,
            "effective_date": _date(explicit[1]) if explicit else None,
            "effective_date_basis": "explicit_in_decision_core" if explicit else "not_stated_in_core",
            "rate": float(rate_match[1].replace(",", ".")), "action": action,
            "core_evidence": core, "source_sha256": provenance["source_sha256"],
            "retrieved_at": provenance["retrieved_at"], "raw_path": provenance["raw_path"],
            "independent_historical_sha_available": False}


def resolve_effective_dates(decisions: list[dict], daily) -> list[dict]:
    """Match every observed rate transition to a unique preceding original decision.

    daily may be a DataFrame or records with DT/date/effective_date and Rate/rate.
    The initial observation is an anchor, not an inferred change. Unchanged
    releases retain null effective_date when the original core omits it.
    """
    records = daily.to_dict("records") if hasattr(daily, "to_dict") else list(daily)
    history = {}
    for record in records:
        day = next((record[k] for k in ("effective_date", "effective_at", "date", "DT") if k in record), None)
        value = next((record[k] for k in ("rate", "Rate", "value") if k in record), None)
        if day is None or value is None:
            raise ValueError("Daily rate history requires a date and rate.")
        day = str(day)[:10]
        datetime.fromisoformat(day)
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError("Daily rate history contains invalid values.")
        if day in history and history[day] != value:
            raise ValueError("Conflicting daily rates on one effective date.")
        history[day] = value
    ordered = sorted(history.items())
    if not ordered:
        raise ValueError("Empty official daily key-rate history.")
    transitions = [(day, rate) for index, (day, rate) in enumerate(ordered)
                   if index and rate != ordered[index - 1][1]]
    result = [dict(row) for row in decisions]
    changes = sorted((row for row in result if row["action"] != "unchanged"),
                     key=lambda row: row["decision_date"])
    if len(changes) != len(transitions):
        raise ValueError("Decision changes and observed daily transitions have different counts.")
    for index, (decision, (effective_day, rate)) in enumerate(zip(changes, transitions)):
        if (float(decision["rate"]) != rate or decision["decision_date"] > effective_day
                or (index + 1 < len(changes) and effective_day >= changes[index + 1]["decision_date"])):
            raise ValueError("Original decision cannot uniquely explain the daily rate transition.")
        if decision["effective_date"] and decision["effective_date"] != effective_day:
            raise ValueError("Explicit decision effective date conflicts with official daily history.")
        if decision["effective_date"] is None:
            decision["effective_date"] = effective_day
            decision["effective_date_basis"] = "observed_transition_in_official_daily_history"
        decision["effective_date_daily_crosscheck"] = "PASS"
    return result


def _daily_rate_records(content: bytes) -> list[dict]:
    rows = []
    for element in ET.fromstring(content).iter():
        if element.tag.rsplit("}", 1)[-1] == "KR":
            fields = {child.tag.rsplit("}", 1)[-1]: child.text for child in element}
            rows.append({"DT": fields["DT"], "Rate": fields["Rate"]})
    return rows


def collect_key_decisions(cache_dir: Path, download: bool) -> list[dict]:
    """Collect all original 2021-2024 decisions, including unchanged decisions.

    Change effective dates are checked against the sibling official SOAP cache
    key_rate_2021_2024.xml; never infer Mondays from meeting dates.
    Offline reruns validate every raw snapshot and the derived JSON digest.
    """
    cache_dir = _cache_path(cache_dir)
    calendar, provenance = cached_cbr_document(KEY_DECISION_CALENDAR, cache_dir / "calendar.html", download=download)
    links = key_decision_links(calendar.decode("utf-8-sig"))
    result = []
    for link in links:
        content, source = cached_cbr_document(link["source_url"], cache_dir / (link["decision_date"] + ".html"), download=download)
        row = parse_key_decision(content.decode("utf-8-sig"), link, source)
        row["archive_sha256"] = provenance["source_sha256"]
        result.append(row)
    history_path = cache_dir.parent / "key_rate_2021_2024.xml"
    history_content = history_path.read_bytes()
    result = resolve_effective_dates(result, _daily_rate_records(history_content))
    history_sha = _sha(history_content)
    for row in result:
        row["daily_history_sha256"] = history_sha
        row["daily_history_raw_path"] = history_path.relative_to(_PROJECT_ROOT).as_posix()
    serialized = (json.dumps(result, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    output = cache_dir / "key_decisions.json"
    manifest = cache_dir / "key_decisions_manifest.json"
    if not download:
        saved = json.loads(manifest.read_text(encoding="utf-8"))
        if (saved.get("schema_version") != _SCHEMA or saved.get("json_sha256") != _sha(output.read_bytes())
                or saved.get("json_sha256") != _sha(serialized)):
            raise ValueError("Derived key-decision cache differs from hash-validated source parsing.")
    else:
        output.write_bytes(serialized)
        manifest.write_text(json.dumps({"schema_version": _SCHEMA, "json_sha256": _sha(serialized),
                                       "decision_count": len(result), "archive_url": KEY_DECISION_CALENDAR,
                                       "archive_sha256": provenance["source_sha256"],
                                       "daily_history_sha256": history_sha}, indent=2) + "\n", encoding="utf-8")
    return result
