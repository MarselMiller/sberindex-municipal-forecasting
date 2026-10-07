"""Small offline news readers and explicitly enabled, bounded source retrieval.

A publication timestamp is not evidence that today's headline or text existed
then. Readers retain it but make the strict text availability date equal to
the retrieval date. Even a dated primary CBR release is only an eligible,
unverified archival assumption; no reader silently confirms an old version.
All tests use synthetic HTML/JSON/XML and mocked HTTP responses.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
import xml.etree.ElementTree as ET


SOURCES = {
    "cbr": {"cbr.ru", "www.cbr.ru"},
    "lenta": {"lenta.ru", "www.lenta.ru"},
    "pravo": {"publication.pravo.gov.ru", "pravo.gov.ru"},
    "mchs": {"mchs.gov.ru", "www.mchs.gov.ru"},
}


def reviewed_cbr_pdf_events(raw_dir: str | Path) -> tuple[list[dict], list[dict]]:
    """Reuse only previously reviewed original PDFs with a dated archive link.

    The existing E05a checker verifies PDF/text/archive SHA256, the original
    heading and the archive publication label. Creation/modification labels
    must both equal publication day; metadata alone is never enough. This
    retains E05a's explicit trust in an official dated archival document, not
    an independently collected 2024 SHA snapshot. No PDF executable is run.
    """
    from .macro_cbr import collect_cbr_pdf_sources
    raw_dir = Path(raw_dir)
    manifest = raw_dir / "consumption_pdf_extracts.json"
    if not manifest.exists():
        return [], [{"source": "cbr", "source_url": None, "parse_status": "reviewed_pdf_cache_missing",
                     "error": "No reviewed PDF/text/archive provenance; no historical event inferred"}]
    _, catalog = collect_cbr_pdf_sources(raw_dir)
    by_url = {entry["url"]: entry for entry in catalog}
    records = json.loads(manifest.read_text(encoding="utf-8"))
    docs = []
    for row in records:
        checked = by_url.get(row.get("source_url"), {})
        if checked.get("verification_status") != "parsed_reviewed_pdf_cache" or checked.get("availability_status") != "A":
            continue
        day = str(row["published_at"])
        if row.get("pdf_creation_date") != day.replace("-", "") or row.get("pdf_modification_date") != day.replace("-", ""):
            checked["news_event_status"] = "later_or_unknown_pdf_version_excluded"
            continue
        # The title and snippet are literal original heading lines, not
        # newly written macro conclusions or current HTML descriptions.
        lines = [line.strip() for line in (raw_dir / row["text_file"]).read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        if len(lines) < 2 or "Среднесрочный прогноз Банка России" not in lines[0] or "по ключевой ставке" not in lines[1]:
            checked["news_event_status"] = "unrecognized_original_heading"
            continue
        published, precision = _iso_date(day)
        document = _make_document("cbr", row["source_url"], row["retrieved_at"], lines[0], published, precision,
            snippet=" ".join(lines[1:4]), evidence="Original PDF heading, dated official archive entry, verified PDF/text/archive SHA256 and publication-day PDF creation/modification labels.")
        document.update(historical_version_confirmed=True, available_at=published,
            historical_version_status="reviewed_original_official_pdf",
            availability_evidence="Reviewed original dated official PDF under E05a official-archive trust assumption; no independent historical SHA snapshot. "
                + f"PDF SHA256={row['pdf_sha256']}; text SHA256={row['text_sha256']}; archive SHA256={row['publication_evidence_sha256']}",
            source_file_sha256=row["pdf_sha256"], retrieval_source_url=row["source_url"])
        docs.append(document)
        checked["news_event_status"] = "reviewed_original_pdf_heading_only"
    for checked in catalog:
        checked.update(source="cbr", source_url=checked.get("url"), http_status=None,
            file_size=checked.get("bytes"), parse_status=checked.get("verification_status"),
            period="2023-01-01..2024-12-31", retrieval_mode="existing_reviewed_E05a_cache",
            terms_note="Official dated archival PDF; accepted E05a archive-trust assumption, independent old SHA snapshot absent")
    return docs, catalog
MONTHS = {
    "января": 1, "февраля": 2, "марта": 3, "апреля": 4,
    "мая": 5, "июня": 6, "июля": 7, "августа": 8,
    "сентября": 9, "октября": 10, "ноября": 11, "декабря": 12,
}
TERMS_NOTES = {
    "cbr": "Official public source. Public access does not establish reuse rights; no republication authorized.",
    "lenta": "Publisher content; retain short titles/snippets locally. Reuse terms require a separate check.",
    "pravo": "Official publication portal. Original legal text and portal reuse terms require separate verification.",
    "mchs": "Official current public index probe only; historical archive stability and reuse terms not verified.",
}


def _clean(text: str) -> str:
    return " ".join(str(text).split()).strip()


@dataclass
class _Node:
    tag: str
    attrs: dict[str, str | None] = field(default_factory=dict)
    parent: _Node | None = None
    children: list = field(default_factory=list)

    def text(self) -> str:
        return _clean(" ".join(c.text() if isinstance(c, _Node) else c for c in self.children
                               if not isinstance(c, _Node) or c.tag not in {"script", "style", "noscript"}))

    def walk(self):
        yield self
        for child in self.children:
            if isinstance(child, _Node):
                yield from child.walk()


class _HTML(HTMLParser):
    def __init__(self, text: str):
        super().__init__(convert_charrefs=True)
        self.root = _Node("root")
        self.current = self.root
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        node = _Node(tag, dict(attrs), self.current)
        self.current.children.append(node)
        if tag not in {"meta", "link", "img", "input", "br", "hr", "source", "wbr", "area", "base", "embed", "param", "col", "track"}:
            self.current = node

    def handle_startendtag(self, tag, attrs):
        self.current.children.append(_Node(tag, dict(attrs), self.current))

    def handle_endtag(self, tag):
        node = self.current
        while node.parent is not None:
            if node.tag == tag:
                self.current = node.parent
                return
            node = node.parent

    def handle_data(self, data):
        self.current.children.append(data)


def _iso_date(value: str | None) -> tuple[str | None, str | None]:
    """Keep exact timestamps; date-only labels conservatively use end of day."""
    if not value:
        return None, None
    text = _clean(value)
    iso = re.fullmatch(r"\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?", text)
    if iso:
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
            precision = "timestamp" if "T" in text or " " in text else "day"
            if precision == "day":
                dt = dt.replace(hour=23, minute=59, second=59)
            return dt.isoformat(), precision
        except ValueError:
            return None, None
    found = re.search(r"\b(\d{1,2})[./](\d{1,2})[./](\d{4})(?:\s+(\d{2}):(\d{2})(?::(\d{2}))?)?\b", text)
    hour, minute, second, precision = 23, 59, 59, "day"
    if found:
        day, month, year = map(int, found.groups()[:3])
        if found[4] is not None:
            hour, minute, second = int(found[4]), int(found[5]), int(found[6] or 0)
            precision = "timestamp"
    else:
        found = re.search(r"\b(\d{1,2})\s+(" + "|".join(MONTHS) + r")\s+(\d{4})\b", text.lower())
        if not found:
            try:
                return parsedate_to_datetime(text).isoformat(), "timestamp"
            except (TypeError, ValueError, OverflowError):
                return None, None
        day, month, year = int(found[1]), MONTHS[found[2]], int(found[3])
    try:
        return datetime(year, month, day, hour, minute, second).isoformat(), precision
    except ValueError:
        return None, None


def _metadata(tree: _HTML) -> tuple[dict, list[dict]]:
    meta, structured = {}, []
    for node in tree.root.walk():
        if node.tag == "meta":
            key = node.attrs.get("property") or node.attrs.get("name") or node.attrs.get("itemprop")
            if key:
                meta[str(key).lower()] = node.attrs.get("content") or ""
        if node.tag == "script" and node.attrs.get("type") == "application/ld+json":
            raw = "".join(c for c in node.children if isinstance(c, str))
            try:
                value = json.loads(raw)
            except (ValueError, TypeError):
                continue
            structured.extend(_article_objects(value))
    return meta, structured


def _article_objects(value):
    if isinstance(value, list):
        for item in value:
            yield from _article_objects(item)
    elif isinstance(value, dict):
        types = value.get("@type", [])
        if isinstance(types, str):
            types = [types]
        if any(t in {"Article", "NewsArticle", "ReportageNewsArticle"} for t in types):
            yield value
        if "@graph" in value:
            yield from _article_objects(value["@graph"])


def _make_document(source: str, url: str, retrieved: str, title: str, published=None,
                   precision=None, snippet="", updated=None, *, evidence="", candidate=False,
                   document_kind="article", updated_date_precision=None) -> dict:
    title = _clean(title)
    retrieved_dt = datetime.fromisoformat(str(retrieved).replace("Z", "+00:00"))
    status = "dated_primary_release_unverified" if candidate else "unknown"
    if updated and published:
        # Compare the source's own date labels; a later declared revision is
        # affirmative evidence against treating the current text as original.
        p, u = datetime.fromisoformat(published), datetime.fromisoformat(updated)
        later = u > p if p.tzinfo is not None and u.tzinfo is not None else u.replace(tzinfo=None) > p.replace(tzinfo=None)
        if later:
            status, candidate = "updated_current", False
    return {
        "source": source, "source_url": url, "published_at": published,
        "publication_date_precision": precision, "publication_date_evidence": evidence,
        "updated_date_precision": updated_date_precision,
        "event_date": None, "updated_at": updated, "retrieved_at": retrieved_dt.isoformat(),
        "available_at": retrieved_dt.isoformat(), "title": title, "snippet": _clean(snippet)[:800],
        "historical_version_status": status, "availability_status": "retrieval_only",
        "availability_evidence": "Current downloaded text only; historical version not confirmed. " + evidence,
        "archival_assumption_eligible": candidate, "document_kind": document_kind,
    }


def _article(tree: _HTML, source: str, url: str, retrieved: str, *, cbr=False):
    meta, structured = _metadata(tree)
    article = structured[0] if structured else {}
    h1 = next((n.text() for n in tree.root.walk() if n.tag == "h1"), "")
    title = article.get("headline") or meta.get("og:title") or h1
    if not title and cbr:
        title = next((n.text() for n in tree.root.walk() if n.tag == "title"), "")
    if not title:
        return None
    raw_date = article.get("datePublished") or meta.get("article:published_time") or meta.get("datepublished")
    if not raw_date:
        date_nodes = [n for n in tree.root.walk() if
                      n.attrs.get("itemprop") == "datePublished" or
                      "news-info-line_date" in str(n.attrs.get("class", ""))]
        if date_nodes:
            n = date_nodes[0]
            raw_date = n.attrs.get("content") or n.attrs.get("datetime") or n.text()
    published, precision = _iso_date(raw_date)
    evidence = "Dated publication header or publisher structured metadata; not a version archive."
    if cbr:
        # Archived CBR press filenames expose their own dated publication label.
        # They do not prove that today's HTML is the historical original.
        file_published, file_precision = _cbr_file_date(url)
        if not published and file_published:
            published, precision = file_published, file_precision
            evidence = "Publication timestamp label in official archived press filename; current HTML version unconfirmed."
        if file_published and published and published[:10] == file_published[:10]:
            footer = [n.text() for n in tree.root.walk()
                      if n.tag in {"div", "span", "time", "p"}
                      and re.fullmatch(r"\d{2}\.\d{2}\.\d{4}\s+\d{2}:\d{2}:\d{2}", n.text())]
            dates = [_iso_date(text) for text in footer]
            matching = [date for date in dates if date[0] and date[0][:10] == published[:10]]
            if len(set(matching)) == 1 and (precision == "day" or "filename" in evidence):
                published, precision = matching[0]
                evidence = "Explicit archived press footer timestamp agrees with dated official filename; current HTML version unconfirmed."
    if not published:
        return None
    update = article.get("dateModified") or meta.get("article:modified_time") or meta.get("datemodified")
    if not update and cbr:
        found = re.search(r"Последнее обновление страницы:\s*(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}|\d{2}\.\d{2}\.\d{4}(?:\s+\d{2}:\d{2}:\d{2})?)", tree.root.text())
        update = found[1] if found else None
    updated, updated_precision = _iso_date(update)
    snippet = article.get("description") or meta.get("description") or meta.get("og:description") or ""
    if not snippet:
        paragraphs = [n.text() for n in tree.root.walk() if n.tag == "p" and n.text()]
        snippet = paragraphs[0] if paragraphs else ""
    primary = cbr and "/press/" in urlsplit(url).path
    return _make_document(source, url, retrieved, title, published, precision, snippet, updated,
                          evidence=evidence,
                          candidate=primary, updated_date_precision=updated_precision)


def _cbr_file_date(url: str) -> tuple[str | None, str | None]:
    file = parse_qs(urlsplit(url).query).get("file", [""])[0]
    found = re.fullmatch(r"(\d{2})(\d{2})(\d{4})_(\d{2})(\d{2})(\d{2})(?:key|keyrate)\.htm", file, re.I)
    if not found:
        return None, None
    day, month, year, hour, minute, second = map(int, found.groups())
    try:
        return datetime(year, month, day, hour, minute, second).isoformat(), "timestamp"
    except ValueError:
        return None, None


def _near_date(node: _Node) -> tuple[str | None, str | None]:
    for _ in range(5):
        if node is None:
            break
        dated = []
        for child in node.walk():
            cls = str(child.attrs.get("class", ""))
            if child.tag == "time" or "date" in cls.lower() or child.attrs.get("itemprop") == "datePublished":
                date = _iso_date(child.attrs.get("datetime") or child.attrs.get("content") or child.text())
                if date[0]:
                    dated.append(date)
        unique = list(dict.fromkeys(dated))
        if len(unique) == 1:
            return unique[0]
        if len(unique) > 1:
            return None, None
        node = node.parent
    return None, None


def _validate_url(url: str, source: str) -> None:
    p = urlsplit(url)
    if source not in SOURCES or p.scheme != "https" or p.hostname not in SOURCES[source] or p.username or p.password or p.port not in {None, 443}:
        raise ValueError("Only allowlisted public source HTTPS URLs are accepted.")


def discover_links(html: str, source_url: str, source: str, *, first_date="2023-01-01", last_date="2024-12-31") -> list[dict]:
    """Discover only actual same-source links; never synthesize article URLs."""
    _validate_url(source_url, source)
    if source == "mchs":
        return []
    tree, out = _HTML(html), {}
    for node in tree.root.walk():
        if node.tag != "a" or not node.attrs.get("href"):
            continue
        url = urljoin(source_url, node.attrs["href"])
        try:
            _validate_url(url, source)
        except ValueError:
            continue
        path = urlsplit(url).path
        if source == "lenta":
            match = re.search(r"/news/(\d{4})/(\d{2})/(\d{2})/[^/]+/?$", path)
            if not match:
                continue
            date, precision = _iso_date("-".join(match.groups()))
        elif source == "cbr":
            file_date = _cbr_file_date(url) if path.rstrip("/").endswith("/press/pr") else (None, None)
            if not (file_date[0] or "/press/keypr/" in path or "/press/event/" in path or "/summary_key_rate_" in path):
                continue
            date, precision = file_date if file_date[0] else _near_date(node)
        else:
            if not re.search(r"/(?:document|Document|api/Documents)/", path):
                continue
            date, precision = _near_date(node)
        if date and not first_date <= date[:10] <= last_date:
            continue
        title = node.text()
        if title:
            out.setdefault(url, {"source": source, "url": url, "title": title,
                                 "published_at": date, "publication_date_precision": precision})
    return list(out.values())


def parse_cbr(html: str, source_url: str, retrieved_at: str) -> list[dict]:
    """Read a dated CBR article or dated press-index links, without version claims."""
    _validate_url(source_url, "cbr")
    tree = _HTML(html)
    result = _article(tree, "cbr", source_url, retrieved_at, cbr=True)
    if result:
        return [result]
    return [_make_document("cbr", row["url"], retrieved_at, row["title"], row["published_at"],
                           row["publication_date_precision"], evidence="Current index date label; linked text version not retrieved.",
                           document_kind="index_title")
            for row in discover_links(html, source_url, "cbr")]


def parse_lenta(html: str, source_url: str, retrieved_at: str) -> list[dict]:
    """Read publisher metadata/RSS or archive titles; today's RSS is no old archive."""
    _validate_url(source_url, "lenta")
    if html.lstrip().startswith("<?xml") or html.lstrip().startswith("<rss"):
        root = ET.fromstring(html)
        out = []
        for item in root.findall(".//item"):
            url = item.findtext("link", "")
            try:
                _validate_url(url, "lenta")
            except ValueError:
                continue
            published, precision = _iso_date(item.findtext("pubDate"))
            title = item.findtext("title", "")
            if title:
                snippet = _HTML(item.findtext("description", "")).root.text()
                out.append(_make_document("lenta", url, retrieved_at, title, published, precision, snippet,
                                          evidence="Current RSS pubDate; original RSS edition not confirmed.", document_kind="rss_title"))
        return out
    tree = _HTML(html)
    result = _article(tree, "lenta", source_url, retrieved_at)
    if result:
        return [result]
    return [_make_document("lenta", row["url"], retrieved_at, row["title"], row["published_at"],
                           row["publication_date_precision"], evidence="Date label in article URL, not exact publication timestamp or historical text evidence.",
                           document_kind="index_title")
            for row in discover_links(html, source_url, "lenta")]


def _pravo_objects(value):
    if isinstance(value, list):
        for child in value:
            yield from _pravo_objects(child)
    elif isinstance(value, dict):
        lower = {str(key).lower(): val for key, val in value.items()}
        if any(key in lower for key in {"title", "name", "complexname", "documentname"}):
            yield lower
        else:
            for child in value.values():
                if isinstance(child, (dict, list)):
                    yield from _pravo_objects(child)


def parse_pravo(html: str, source_url: str, retrieved_at: str) -> list[dict]:
    """Read portal metadata, preserving publication and legal event dates separately."""
    _validate_url(source_url, "pravo")
    if html.lstrip().startswith(("{", "[")):
        value = json.loads(html)
        out = []
        for item in _pravo_objects(value):
            title = item.get("title") or item.get("complexname") or item.get("documentname") or item.get("name")
            raw = item.get("publicationdate") or item.get("publishdate") or item.get("publishedat") or item.get("signatorydate")
            # signatoryDate is a legal event date, never a publication fallback.
            if not any(item.get(k) for k in {"publicationdate", "publishdate", "publishedat"}):
                raw = None
            published, precision = _iso_date(raw)
            url = item.get("url") or item.get("link") or source_url
            url = urljoin(source_url, str(url))
            try:
                _validate_url(url, "pravo")
            except ValueError:
                continue
            out.append(_make_document("pravo", url, retrieved_at, title, published, precision,
                                      evidence="Current official portal publication metadata; original text/PDF version not confirmed.",
                                      document_kind="portal_metadata"))
        return out
    tree = _HTML(html)
    result = _article(tree, "pravo", source_url, retrieved_at)
    if result:
        return [result]
    return [_make_document("pravo", row["url"], retrieved_at, row["title"], row["published_at"],
                           row["publication_date_precision"], evidence="Current official index date label; original document not retrieved.",
                           document_kind="index_title")
            for row in discover_links(html, source_url, "pravo")]


def parse_mchs_probe(html: str, source_url: str, retrieved_at: str) -> list[dict]:
    """Probe HTTP access only; no historical archive adapter has been verified."""
    _validate_url(source_url, "mchs")
    return []


PARSERS = {"cbr": parse_cbr, "lenta": parse_lenta, "pravo": parse_pravo, "mchs": parse_mchs_probe}


@dataclass
class _Budget:
    max_requests: int
    max_bytes: int
    requests: int = 0
    bytes: int = 0

    def request(self):
        if self.requests >= self.max_requests:
            raise ValueError("Request budget exhausted.")
        self.requests += 1


class _Redirect(HTTPRedirectHandler):
    def __init__(self, source, budget):
        self.source, self.budget = source, budget

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _validate_url(newurl, self.source)
        self.budget.request()
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class _RetrievalLimitError(ValueError):
    def __init__(self, message: str, provenance: dict):
        super().__init__(message)
        self.provenance = provenance


def _fetch_bytes(url: str, source: str, budget: _Budget, max_file_bytes: int, timeout: float):
    _validate_url(url, source)
    budget.request()
    opener = build_opener(_Redirect(source, budget))
    request = Request(url, headers={"User-Agent": "sberforecast E06b bounded news audit"})
    with opener.open(request, timeout=timeout) as response:
        _validate_url(response.geturl(), source)
        headers = {
            "response_url": response.geturl(), "http_status": response.status,
            "content_type": response.headers.get("Content-Type"),
            "http_last_modified": response.headers.get("Last-Modified"),
            "http_etag": response.headers.get("ETag"),
        }
        limit = min(max_file_bytes, budget.max_bytes - budget.bytes)
        if limit <= 0:
            raise _RetrievalLimitError("Byte budget exhausted.", headers)
        content = response.read(limit + 1)
        budget.bytes += len(content)
        if len(content) > limit:
            raise _RetrievalLimitError("Source exceeds file or total byte budget.", {
                **headers, "bytes_read": len(content), "truncated": True,
                "partial_sha256": hashlib.sha256(content).hexdigest(), "file_retained": False,
            })
        return content, headers


def _decode(content: bytes, content_type: str | None) -> str:
    charset = re.search(r"charset\s*=\s*[\"']?([^;\s\"']+)", content_type or "", re.I)
    return content.decode(charset[1] if charset else "utf-8-sig")


def collect_sources(entries: list[dict], cache_dir: str | Path, *, download: bool = False,
                    max_requests: int = 200, max_bytes: int = 50 * 1024 * 1024,
                    max_file_bytes: int = 2 * 1024 * 1024, timeout: float = 30.0) -> tuple[list[dict], list[dict]]:
    """Read cache or explicitly fetch sources and persist success/failure provenance.

    The caller's CLI must expose downloads as ``--download``. Cache bytes are
    never replaced; cached retrieval metadata stays unchanged. A cache without
    provenance is available only when read now; filesystem mtime is no proof
    of the original download date.
    The total budget includes cached bytes and failed bounded reads. Request
    limits include redirects. No automatic retry or alternative access occurs.
    """
    for name, value in {"max_requests": max_requests, "max_bytes": max_bytes, "max_file_bytes": max_file_bytes}.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    budget, docs, manifest = _Budget(max_requests, max_bytes), [], []
    for spec in entries:
        requests_before, bytes_before = budget.requests, budget.bytes
        source, url = spec.get("source"), spec.get("url") or spec.get("source_url")
        retrieved = datetime.now(timezone.utc).isoformat()
        filename = hashlib.sha256(str(url).encode()).hexdigest() + ".bin"
        path, sidecar = cache_dir / filename, cache_dir / (filename + ".json")
        record = {
            "source": source, "source_url": url, "retrieved_at": retrieved,
            "period": spec.get("period", "2023-01-01..2024-12-31"),
            "file": str(path), "http_status": None, "file_size": None, "sha256": None,
            "parse_status": "not_attempted", "terms_note": spec.get("terms_note") or TERMS_NOTES.get(source, "Terms not verified."),
            "retrieval_mode": None, "error": None,
        }
        manifest.append(record)
        try:
            _validate_url(url, source)
            if path.exists():
                if path.stat().st_size > min(max_file_bytes, max_bytes - budget.bytes):
                    raise ValueError("Cached source exceeds file or total byte budget.")
                content = path.read_bytes()
                budget.bytes += len(content)
                provenance = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.exists() else {}
                record.update({key: provenance[key] for key in ["retrieved_at", "http_status", "response_url", "content_type", "http_last_modified", "http_etag"] if key in provenance})
                if not provenance:
                    record["observed_file_mtime"] = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
                    record["retrieval_time_evidence"] = "Read now; original download provenance missing. File mtime is not historical availability evidence."
                record["retrieval_mode"] = "existing_cache"
            elif download:
                record["retrieval_mode"] = "downloaded"
                if budget.bytes >= budget.max_bytes:
                    raise ValueError("Byte budget exhausted.")
                content, headers = _fetch_bytes(url, source, budget, max_file_bytes, timeout)
                record.update(headers)
                # Record completion time, not the time before the HTTP request.
                record["retrieved_at"] = datetime.now(timezone.utc).isoformat()
                path.write_bytes(content)
                sidecar.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            else:
                record["retrieval_mode"] = "offline_missing"
                raise FileNotFoundError("Cached source missing; download=False.")
            record["file_size"] = len(content)
            record["sha256"] = hashlib.sha256(content).hexdigest()
            if path.exists() and sidecar.exists() and record["retrieval_mode"] == "existing_cache":
                previous = json.loads(sidecar.read_text(encoding="utf-8"))
                if previous.get("sha256") and previous["sha256"] != record["sha256"]:
                    raise ValueError("Cached source hash does not match recorded download provenance.")
            record["parse_status"] = "downloaded_not_parsed"
            text = _decode(content, record.get("content_type"))
            parsed = PARSERS[source](text, url, record["retrieved_at"])
            if source == "mchs":
                record["parser_note"] = "Historical archive adapter unavailable; current page is an access probe only, no events inferred."
            retained = []
            for doc in parsed:
                doc["retrieval_source_url"] = url
                doc["source_file_sha256"] = record["sha256"]
                if doc["published_at"]:
                    day = doc["published_at"][:10]
                    period = spec.get("period", "2023-01-01..2024-12-31").split("..")
                    if len(period) == 2 and not period[0] <= day <= period[1]:
                        continue
                retained.append(doc)
            record["document_count"] = len(parsed)
            record["retained_document_count"] = len(retained)
            record["excluded_outside_period_count"] = len(parsed) - len(retained)
            record["parse_status"] = "parsed" if parsed else "no_supported_documents"
            record["discovered_links"] = discover_links(text, url, source) if text.lstrip().startswith("<") and not text.lstrip().startswith(("<?xml", "<rss")) else []
            docs.extend(retained)
        except (HTTPError, URLError, OSError, ValueError, TypeError, LookupError, ET.ParseError) as exc:
            record["parse_status"] = "parse_error" if record["parse_status"] == "downloaded_not_parsed" else "retrieval_error"
            record["error"] = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, HTTPError):
                record["http_status"] = exc.code
            if isinstance(exc, _RetrievalLimitError):
                record.update(exc.provenance)
        record["requests_consumed"] = budget.requests - requests_before
        record["bytes_accounted"] = budget.bytes - bytes_before
        if record["retrieval_mode"] == "downloaded":
            sidecar.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    history_path = cache_dir / "retrieval_manifest.json"
    history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else []
    history.extend(manifest)
    history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    return docs, manifest
