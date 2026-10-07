"""Synthetic source fixtures and mock transport; no real publisher text or HTTP."""
from __future__ import annotations

import hashlib
import json
import os
from urllib.error import HTTPError

import pytest

from sberforecast import news_sources as ns


RETRIEVED = "2026-10-07T12:00:00+00:00"
CBR_URL = "https://www.cbr.ru/press/keypr/?id=synthetic"
LENTA_URL = "https://lenta.ru/news/2024/01/12/synthetic/"
PRAVO_URL = "https://publication.pravo.gov.ru/document/synthetic"


def test_reviewed_pdf_cache_missing_never_infers_historical_event(tmp_path):
    docs, catalog = ns.reviewed_cbr_pdf_events(tmp_path)
    assert docs == []
    assert catalog[0]["parse_status"] == "reviewed_pdf_cache_missing"


@pytest.mark.parametrize("verified,modified,expected", [(True,"20240216",1),(False,"20240216",0),(True,"20241007",0)])
def test_reviewed_original_pdf_needs_prior_integrity_check_and_no_later_revision(tmp_path,monkeypatch,verified,modified,expected):
    from sberforecast import macro_cbr
    row = dict(source_url="https://cbr.ru/Collection/Collection/File/synthetic/forecast.pdf",
        published_at="2024-02-16",pdf_creation_date="20240216",pdf_modification_date=modified,
        retrieved_at=RETRIEVED,text_file="synthetic.txt",pdf_sha256="synthetic_pdf_hash",
        text_sha256="synthetic_text_hash",publication_evidence_sha256="synthetic_archive_hash")
    (tmp_path/"consumption_pdf_extracts.json").write_text(json.dumps([row]),encoding="utf-8")
    (tmp_path/"synthetic.txt").write_text("Среднесрочный прогноз Банка России\nпо итогам заседания по ключевой ставке 16 февраля 2024 года\nСинтетический документ\n",encoding="utf-8")
    monkeypatch.setattr(macro_cbr,"collect_cbr_pdf_sources",lambda folder:(None,[dict(url=row['source_url'],
        verification_status="parsed_reviewed_pdf_cache" if verified else "integrity_failed",availability_status="A" if verified else "C")]))
    docs,_=ns.reviewed_cbr_pdf_events(tmp_path)
    assert len(docs)==expected
    if docs:
        assert docs[0]['historical_version_confirmed'] is True
        assert docs[0]['available_at']=='2024-02-16T23:59:59'
        assert 'no independent historical SHA' in docs[0]['availability_evidence']


def cbr_html(*, updated="12.01.2024"):
    return ('<h1>Синтетическое решение о ключевой ставке</h1>'
            '<div class="news-info-line_date">12 января 2024 года</div>'
            '<p>Синтетический пример для проверки parser.</p>'
            '<script>secret body must not be a snippet</script>'
            '<div>Последнее обновление страницы: ' + updated + '</div>')


def test_cbr_date_is_not_proof_of_historical_text():
    row, = ns.parse_cbr(cbr_html(), CBR_URL, RETRIEVED)
    assert row["published_at"] == "2024-01-12T23:59:59"
    assert row["publication_date_precision"] == "day"
    assert row["event_date"] is None
    assert row["available_at"] == RETRIEVED
    assert row["availability_status"] == "retrieval_only"
    assert row["historical_version_status"] == "dated_primary_release_unverified"
    assert row["archival_assumption_eligible"] is True
    assert row["snippet"] == "Синтетический пример для проверки parser."


def test_cbr_later_update_is_explicitly_not_original():
    row, = ns.parse_cbr(cbr_html(updated="07.10.2026"), CBR_URL, RETRIEVED)
    assert row["historical_version_status"] == "updated_current"
    assert row["archival_assumption_eligible"] is False
    assert row["available_at"] == RETRIEVED
    assert row["updated_at"] == "2026-10-07T23:59:59"
    assert row["updated_date_precision"] == "day"


@pytest.mark.parametrize("published,modified,publication_precision,update_precision", [
    ("2024-01-12T13:15:00+03:00", "2026-10-07", "timestamp", "day"),
    ("2024-01-12", "2026-10-07T16:00:00+03:00", "day", "timestamp"),
    ("2024-01-12T13:15:00+03:00", "2026-10-08", "timestamp", "day"),
])
def test_article_preserves_update_precision_independently_of_publication(
    published, modified, publication_precision, update_precision,
):
    obj = {"@type": "NewsArticle", "headline": "Синтетическая публикация",
           "datePublished": published, "dateModified": modified}
    html = '<script type="application/ld+json">' + json.dumps(obj) + '</script>'
    row, = ns.parse_lenta(html, LENTA_URL, RETRIEVED)
    assert row["publication_date_precision"] == publication_precision
    assert row["updated_date_precision"] == update_precision
    assert row["available_at"] == RETRIEVED


def test_cbr_archive_titles_are_not_promoted_to_original_release():
    html = ('<h1>Синтетический архив</h1><div class="item">'
            '<div class="date">12.01.2024</div>'
            '<a href="/press/keypr/?id=synthetic">Синтетическое решение</a></div>'
            '<div class="item"><div class="date">12.01.2025</div>'
            '<a href="/press/keypr/?id=future">Будущий заголовок</a></div>')
    row, = ns.parse_cbr(html, "https://www.cbr.ru/press/keypr/", RETRIEVED)
    assert row["document_kind"] == "index_title"
    assert row["historical_version_status"] == "unknown"
    assert row["source_url"] == CBR_URL
    assert row["archival_assumption_eligible"] is False


def test_cbr_calendar_real_link_shape_keeps_only_requested_years():
    html = ('<h1>Синтетический календарь</h1>'
            '<a href="/press/pr/?file=10022023_133000Key.htm">Синтетический пресс-релиз</a>'
            '<a href="/press/pr/?file=16022024_133000Keyrate.htm">Другой синтетический пресс-релиз</a>'
            '<a href="/press/pr/?file=13022026_133000Key.htm">Будущий пресс-релиз</a>'
            '<a href="/press/pr/?file=31022024_133000Key.htm">Неверная дата</a>'
            '<a href="/press/pr/?file=16022024_133000Other.htm">Другая публикация</a>')
    rows = ns.discover_links(html, "https://www.cbr.ru/dkp/cal_mp/", "cbr")
    assert len(rows) == 2
    assert rows[0]["published_at"] == "2023-02-10T13:30:00"
    assert rows[1]["published_at"] == "2024-02-16T13:30:00"
    parsed = ns.parse_cbr(html, "https://www.cbr.ru/dkp/cal_mp/", RETRIEVED)
    assert len(parsed) == 2
    assert all(row["available_at"] == RETRIEVED for row in parsed)
    assert all(row["historical_version_status"] == "unknown" for row in parsed)


def test_cbr_archived_press_footer_preserves_explicit_time_without_version_claim():
    html = ('<title>Синтетический пресс-релиз | Банк России</title>'
            '<div class="col-md-6 col-12 news-info-line_date">10 февраля 2023 года</div>'
            '<p>Только синтетический пример.</p>'
            '<p class="note">10.02.2023 13:30:00</p>')
    url = "https://www.cbr.ru/press/pr/?file=10022023_133000Key.htm"
    row, = ns.parse_cbr(html, url, RETRIEVED)
    assert row["published_at"] == "2023-02-10T13:30:00"
    assert row["publication_date_precision"] == "timestamp"
    assert "footer timestamp" in row["publication_date_evidence"]
    assert row["historical_version_status"] == "dated_primary_release_unverified"
    assert row["available_at"] == RETRIEVED


def test_cbr_archived_press_filename_fallback_is_explicitly_labelled():
    html = '<h1>Синтетический пресс-релиз</h1><p>Синтетический текст.</p>'
    url = "https://www.cbr.ru/press/pr/?file=15082023_103000key.htm"
    row, = ns.parse_cbr(html, url, RETRIEVED)
    assert row["published_at"] == "2023-08-15T10:30:00"
    assert "filename" in row["publication_date_evidence"]
    assert row["available_at"] == RETRIEVED


def test_russian_numeric_timestamp_keeps_explicit_hours():
    assert ns._iso_date("10.02.2023 13:30:00") == ("2023-02-10T13:30:00", "timestamp")


def test_lenta_structured_metadata_keeps_revision_and_timestamp():
    obj = {"@type": "NewsArticle", "headline": "Синтетическая новость",
           "datePublished": "2024-01-12T13:15:00+03:00",
           "dateModified": "2026-10-01T14:00:00+03:00", "description": "Пример."}
    html = '<script type="application/ld+json">' + json.dumps({"@graph": [obj]}) + '</script>'
    row, = ns.parse_lenta(html, LENTA_URL, RETRIEVED)
    assert row["published_at"] == obj["datePublished"]
    assert row["updated_at"] == obj["dateModified"]
    assert row["updated_date_precision"] == "timestamp"
    assert row["historical_version_status"] == "updated_current"
    assert row["available_at"] == RETRIEVED


def test_modified_timestamps_are_compared_with_their_offsets():
    obj = {"@type": "NewsArticle", "headline": "Синтетическая новость",
           "datePublished": "2024-01-12T13:15:00+03:00",
           "dateModified": "2024-01-12T11:00:00+00:00"}
    html = '<script type="application/ld+json">' + json.dumps(obj) + '</script>'
    row, = ns.parse_lenta(html, LENTA_URL, RETRIEVED)
    assert row["historical_version_status"] == "updated_current"


def test_malformed_jsonld_does_not_hide_valid_html_metadata():
    html = ('<script type="application/ld+json">malformed</script>'
            '<meta property="og:title" content="Синтетическая новость">'
            '<meta property="article:published_time" content="2024-01-12T10:00:00Z">')
    row, = ns.parse_lenta(html, LENTA_URL, RETRIEVED)
    assert row["published_at"] == "2024-01-12T10:00:00+00:00"
    assert row["historical_version_status"] == "unknown"


def test_lenta_archive_follows_only_existing_same_source_links():
    html = ('<a href="/news/2024/01/12/synthetic/">Синтетический заголовок</a>'
            '<a href="/news/2024/01/12/synthetic/">Повторный заголовок</a>'
            '<a href="https://other.example/news/2024/01/12/fake/">Внешний</a>'
            '<a href="/news/2025/01/12/future/">Будущее</a>'
            '<a href="/news/2024/02/31/invalid/">Неверная дата</a>')
    rows = ns.parse_lenta(html, "https://lenta.ru/2024/01/12/", RETRIEVED)
    # Invalid URL day cannot establish publication date. It remains visible,
    # with availability at retrieval; downstream audit must expose it.
    assert len(rows) == 2
    assert rows[0]["source_url"] == LENTA_URL
    assert rows[0]["title"] == "Синтетический заголовок"
    assert rows[0]["published_at"] == "2024-01-12T23:59:59"
    assert rows[1]["published_at"] is None
    assert all(row["available_at"] == RETRIEVED for row in rows)


def test_current_rss_does_not_prove_a_historical_edition():
    xml = ('<?xml version="1.0"?><rss><channel><item><title>Синтетическая новость</title>'
           '<link>' + LENTA_URL + '</link><pubDate>Fri, 12 Jan 2024 10:00:00 +0300</pubDate>'
           '<description>&lt;p&gt;Синтетический snippet&lt;/p&gt;</description>'
           '</item></channel></rss>')
    row, = ns.parse_lenta(xml, "https://lenta.ru/rss/news", RETRIEVED)
    assert row["published_at"] == "2024-01-12T10:00:00+03:00"
    assert row["snippet"] == "Синтетический snippet"
    assert row["document_kind"] == "rss_title"
    assert row["available_at"] == RETRIEVED


def test_pravo_signing_date_is_never_publication_date():
    raw = {"items": [{"Name": "Синтетический акт", "SignatoryDate": "2023-12-28",
                      "PublicationDate": "2024-01-12", "Url": PRAVO_URL},
                     {"Name": "Акт без публикации", "SignatoryDate": "2024-01-11"}]}
    rows = ns.parse_pravo(json.dumps(raw), PRAVO_URL, RETRIEVED)
    assert rows[0]["published_at"] == "2024-01-12T23:59:59"
    assert rows[1]["published_at"] is None
    assert all(row["event_date"] is None for row in rows)
    assert all(row["historical_version_status"] == "unknown" for row in rows)


@pytest.mark.parametrize("parser,url", [
    (ns.parse_cbr, "https://other.example/press/keypr/"),
    (ns.parse_lenta, "http://lenta.ru/news/2024/01/12/test/"),
    (ns.parse_pravo, "https://password@publication.pravo.gov.ru/"),
])
def test_readers_accept_only_public_allowlisted_urls(parser, url):
    with pytest.raises(ValueError, match="allowlisted"):
        parser("<h1>Example</h1>", url, RETRIEVED)


def test_offline_missing_cache_persists_failure_without_fetch(tmp_path, monkeypatch):
    monkeypatch.setattr(ns, "_fetch_bytes", lambda *a, **k: pytest.fail("Network not allowed"))
    docs, manifest = ns.collect_sources([{"source": "cbr", "url": CBR_URL}], tmp_path)
    assert docs == []
    assert manifest[0]["parse_status"] == "retrieval_error"
    assert "download=False" in manifest[0]["error"]
    assert json.loads((tmp_path / "retrieval_manifest.json").read_text())[0]["http_status"] is None


def test_download_cache_provenance_and_offline_reuse(tmp_path, monkeypatch):
    content = cbr_html().encode()

    def fake_fetch(url, source, budget, max_file_bytes, timeout):
        budget.request()
        budget.bytes += len(content)
        return content, {"http_status": 200, "response_url": url, "content_type": "text/html; charset=utf-8"}

    monkeypatch.setattr(ns, "_fetch_bytes", fake_fetch)
    docs, manifest = ns.collect_sources([{"source": "cbr", "url": CBR_URL}], tmp_path, download=True)
    first = manifest[0]
    assert first["sha256"] == hashlib.sha256(content).hexdigest()
    assert first["file_size"] == len(content)
    assert first["http_status"] == 200
    assert first["parse_status"] == "parsed"
    assert first["requests_consumed"] == 1
    assert first["bytes_accounted"] == len(content)
    assert docs[0]["source_file_sha256"] == first["sha256"]
    monkeypatch.setattr(ns, "_fetch_bytes", lambda *a, **k: pytest.fail("Cache must not download"))
    second_docs, second_manifest = ns.collect_sources([{"source": "cbr", "url": CBR_URL}], tmp_path, download=True)
    assert second_docs == docs
    assert second_manifest[0]["retrieved_at"] == first["retrieved_at"]
    assert second_manifest[0]["retrieval_mode"] == "existing_cache"
    assert second_manifest[0]["requests_consumed"] == 0
    assert second_manifest[0]["bytes_accounted"] == len(content)
    assert len(json.loads((tmp_path / "retrieval_manifest.json").read_text())) == 2


def test_hash_mismatch_exposes_cache_corruption(tmp_path, monkeypatch):
    content = cbr_html().encode()
    monkeypatch.setattr(ns, "_fetch_bytes", lambda *a: (content, {"http_status": 200}))
    ns.collect_sources([{"source": "cbr", "url": CBR_URL}], tmp_path, download=True)
    path = next(tmp_path.glob("*.bin"))
    path.write_bytes(content + b" corruption")
    docs, manifest = ns.collect_sources([{"source": "cbr", "url": CBR_URL}], tmp_path)
    assert docs == []
    assert "hash does not match" in manifest[0]["error"]


def test_http_error_is_recorded_and_collection_continues(tmp_path, monkeypatch):
    def unavailable(*args):
        raise HTTPError(CBR_URL, 503, "Service unavailable", {}, None)

    monkeypatch.setattr(ns, "_fetch_bytes", unavailable)
    docs, manifest = ns.collect_sources([{"source": "cbr", "url": CBR_URL}], tmp_path, download=True)
    assert docs == []
    assert manifest[0]["http_status"] == 503
    assert manifest[0]["parse_status"] == "retrieval_error"
    assert "HTTPError" in manifest[0]["error"]
    assert json.loads(next(tmp_path.glob("*.bin.json")).read_text())["http_status"] == 503


def test_successful_http_with_no_supported_documents_is_visible(tmp_path, monkeypatch):
    monkeypatch.setattr(ns, "_fetch_bytes", lambda *a: (b"<html><h1>Application shell</h1></html>", {"http_status": 200}))
    docs, manifest = ns.collect_sources([{"source": "pravo", "url": PRAVO_URL}], tmp_path, download=True)
    assert docs == []
    assert manifest[0]["parse_status"] == "no_supported_documents"
    assert manifest[0]["error"] is None


def test_mchs_access_probe_cannot_manufacture_historical_events(tmp_path, monkeypatch):
    content = b'<html><h1>Emergency index</h1><time>2024-01-12</time></html>'
    monkeypatch.setattr(ns, "_fetch_bytes", lambda *a: (content, {"http_status": 200}))
    url = "https://mchs.gov.ru/deyatelnost/press-centr/operativnaya-informaciya"
    docs, manifest = ns.collect_sources([{"source": "mchs", "url": url}], tmp_path, download=True)
    assert docs == []
    assert manifest[0]["http_status"] == 200
    assert manifest[0]["parse_status"] == "no_supported_documents"
    assert "adapter unavailable" in manifest[0]["parser_note"]
    assert manifest[0]["discovered_links"] == []


def test_byte_budget_bounds_cached_reads(tmp_path):
    filename = hashlib.sha256(CBR_URL.encode()).hexdigest() + ".bin"
    (tmp_path / filename).write_bytes(cbr_html().encode())
    docs, manifest = ns.collect_sources([{"source": "cbr", "url": CBR_URL}], tmp_path, max_bytes=10)
    assert docs == []
    assert "budget" in manifest[0]["error"]


def test_old_file_mtime_without_download_provenance_cannot_backdate_availability(tmp_path):
    filename = hashlib.sha256(CBR_URL.encode()).hexdigest() + ".bin"
    path = tmp_path / filename
    path.write_bytes(cbr_html().encode())
    os.utime(path, (1704067200, 1704067200))  # Synthetic old filesystem date.
    docs, manifest = ns.collect_sources([{"source": "cbr", "url": CBR_URL}], tmp_path)
    assert manifest[0]["observed_file_mtime"].startswith("2024-01-01")
    assert docs[0]["available_at"] == manifest[0]["retrieved_at"]
    assert docs[0]["available_at"] != manifest[0]["observed_file_mtime"]


class _Response:
    def __init__(self, content, url=CBR_URL):
        self.content, self.url = content, url
        self.status, self.headers = 200, {"Content-Type": "text/html"}
        self.read_limit = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def geturl(self):
        return self.url

    def read(self, size):
        self.read_limit = size
        return self.content[:size]


def test_http_body_is_read_with_a_bound_and_request_budget(tmp_path, monkeypatch):
    response = _Response(b"x" * 100)

    class Opener:
        def open(self, *args, **kwargs):
            return response

    monkeypatch.setattr(ns, "build_opener", lambda *a: Opener())
    docs, manifest = ns.collect_sources([{"source": "cbr", "url": CBR_URL}], tmp_path, download=True, max_file_bytes=10)
    assert docs == []
    assert response.read_limit == 11
    assert "exceeds" in manifest[0]["error"]
    assert manifest[0]["http_status"] == 200
    assert manifest[0]["bytes_read"] == 11
    assert manifest[0]["truncated"] is True
    assert manifest[0]["sha256"] is None
    assert manifest[0]["requests_consumed"] == 1
    assert manifest[0]["bytes_accounted"] == 11
    assert not list(tmp_path.glob("*.bin"))
    budget = ns._Budget(max_requests=1, max_bytes=100)
    ns._fetch_bytes(CBR_URL, "cbr", budget, 100, 1)
    with pytest.raises(ValueError, match="Request budget"):
        ns._fetch_bytes(CBR_URL, "cbr", budget, 100, 1)


def test_redirect_is_validated_before_following_a_different_host():
    handler = ns._Redirect("cbr", ns._Budget(2, 100))
    with pytest.raises(ValueError, match="allowlisted"):
        handler.redirect_request(None, None, 302, "Found", {}, "https://other.example/secret")


def test_html_encoding_failure_remains_a_parse_error(tmp_path, monkeypatch):
    monkeypatch.setattr(ns, "_fetch_bytes", lambda *a: (b"\xff", {"http_status": 200, "content_type": "text/html; charset=utf-8"}))
    docs, manifest = ns.collect_sources([{"source": "cbr", "url": CBR_URL}], tmp_path, download=True)
    assert docs == []
    assert manifest[0]["parse_status"] == "parse_error"
    assert manifest[0]["file_size"] == 1
    assert "UnicodeDecodeError" in manifest[0]["error"]


@pytest.mark.parametrize("kwargs", [{"max_requests": 0}, {"max_bytes": -1}, {"max_file_bytes": True}, {"timeout": 0}])
def test_invalid_retrieval_budgets_fail_before_actions(tmp_path, kwargs):
    with pytest.raises(ValueError):
        ns.collect_sources([], tmp_path, **kwargs)
