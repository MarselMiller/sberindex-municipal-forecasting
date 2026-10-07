"""Synthetic core/archive fixtures and mocked HTTP; no real source text."""
from __future__ import annotations

import io
import json
from urllib.error import HTTPError, URLError

import pandas as pd
import pytest

from sberforecast import early_warning_news as news


URL = "https://www.cbr.ru/press/pr/?file=16022024_133000key.htm"
RETRIEVED = "2026-10-07T12:00:00+00:00"


def archive(url=URL, date="16 февраля 2024 года"):
    return f'<div class="main-events_day"><span>{date}</span><div><a href="{url}">Пресс-релиз</a></div></div>'


def evidence(url=URL, date="16 февраля 2024 года"):
    return news.archive_release_evidence(archive(url, date), news.CALENDAR_URL, "synthetic_archive_sha", RETRIEVED)[0]


def article(*, action="сохранить", date="16 февраля 2024 года", rate="5,25", footer="16.02.2024 13:30:00",
            extras="", explicit_date=True, effective=""):
    heading_action = {"сохранить": "сохранил", "повысить": "повысил", "снизить": "снизил"}[action]
    amount = f"на уровне {rate}" if action == "сохранить" else f"{effective} на 50 б.п., до {rate}"
    return (f'<h1>Банк России {heading_action} ключевую ставку до {rate}% годовых</h1>'
            f'<div class="news-info-line_date">{date}</div><div class="landing-text"><p> </p>'
            f'<p>Совет директоров Банка России {date if explicit_date else ""} принял решение '
            f'{action} ключевую ставку {amount}% годовых. Синтетический поздний комментарий.</p>'
            '<p>Несвязанный синтетический прогноз: ставка 99%.</p>'
            f'<p class="note">{footer}</p></div>{extras}')


def extract(html=None, url=URL, archive_evidence=None):
    return news.extract_cbr_key_rate_decision(html or article(), url,
                                              archive_evidence or evidence(), RETRIEVED, "synthetic_article_sha")


def test_core_only_exact_publication_unchanged_and_explicit_assumption():
    row = extract()
    assert row["admitted"]
    assert row["key_rate_percent"] == 5.25
    assert row["action"] == "unchanged"
    assert row["change_basis_points"] == 0
    assert row["decision_date"] == "2024-02-16"
    assert row["effective_date"] is None
    assert row["published_at"] == "2024-02-16T13:30:00+03:00"
    assert row["core_evidence"].endswith("годовых.")
    assert "комментарий" not in row["core_evidence"]
    assert "99%" not in row["core_evidence"]
    assert not row["independent_historical_sha_available"]
    raw = news.core_trusted_record(row)
    assert raw["available_at"] == row["published_at"]
    assert raw["event_date"] == row["decision_date"]
    assert "no independent historical SHA" in raw["availability_evidence"]


@pytest.mark.parametrize("action,direction", [("повысить", "increase"), ("снизить", "decrease")])
def test_dots_in_basis_points_do_not_truncate_core(action, direction):
    row = extract(article(action=action))
    assert row["admitted"] and row["action"] == direction
    assert row["change_basis_points"] == 50


def test_extraordinary_meeting_effective_date_explicit_and_decision_archive_rule():
    url = "https://www.cbr.ru/press/pr/?file=15082023_103000Key.htm"
    html = article(action="повысить", date="15 августа 2023 года", footer="15.08.2023 10:30:00",
                   explicit_date=False, effective="с 15 августа 2023 года")
    row = extract(html, url, evidence(url, "15 августа 2023 года"))
    assert row["admitted"]
    assert row["effective_date"] == row["decision_date"] == "2023-08-15"
    assert row["decision_date_rule"] == "dated_original_decision_release_and_archive_meeting_row"


def test_catalog_footer_is_separate_from_article_content_update():
    row = extract(article(extras='<div class="page-info_last-update">07.10.2026</div>'))
    assert row["admitted"]
    assert row["content_updates"] == []
    assert row["catalog_updates"][0]["evidence_scope"] == "page_footer_scope_unknown"
    assert row["catalog_updates"][0]["precision"] == "day"


def test_original_heading_decision_infinitive_agrees_with_core():
    html = article().replace("Банк России сохранил", "Банк России принял решение сохранить")
    assert extract(html)["admitted"]


def test_declared_later_article_revision_fails_closed():
    row = extract(article(extras='<meta property="article:modified_time" content="2025-01-01">'))
    assert not row["admitted"]
    assert row["reason"] == "declared_later_article_revision_core_not_independently_attested"


@pytest.mark.parametrize("html,reason", [
    (article(footer="16.02.2024"), "missing_or_ambiguous_exact_publication_footer"),
    (article(footer="16.02.2024 14:00:00"), "publication_archive_header_filename_disagreement"),
    (article(date="17 февраля 2024 года"), "publication_archive_header_filename_disagreement"),
    (article().replace("5,25%", "6,25%", 1), "heading_core_action_or_rate_disagreement"),
    (article().replace("Совет директоров", "Синтетический редактор", 1), "first_paragraph_not_board_rate_decision"),
    (article().replace("landing-text", "unrelated"), "missing_or_ambiguous_article_core"),
])
def test_inconsistent_or_missing_evidence_not_admitted(html, reason):
    row = extract(html)
    assert not row["admitted"] and row["reason"] == reason


@pytest.mark.parametrize("url", ["https://evil.invalid/press/pr/?file=16022024_133000key.htm",
                                  "https://www.cbr.ru/dkp/mp_dec/summary/", "http://www.cbr.ru/press/pr/?file=16022024_133000key.htm"])
def test_summaries_translations_and_other_domains_not_auto_admitted(url):
    assert extract(url=url)["reason"] == "not_original_official_rate_release"


def test_unmatched_or_undated_archive_link_not_evidence():
    assert news.archive_release_evidence(archive(date="Без даты"), news.CALENDAR_URL, "sha", RETRIEVED) == []
    assert news.archive_release_evidence(archive(date="17 февраля 2024 года"), news.CALENDAR_URL, "sha", RETRIEVED) == []
    assert news.archive_release_evidence(archive(), "https://evil.invalid/calendar/", "sha", RETRIEVED) == []
    bad = evidence(); bad["archive_sha256"] = ""
    assert not extract(archive_evidence=bad)["admitted"]


class Response(io.BytesIO):
    headers = {"Content-Type": "text/html; charset=utf-8"}
    def getcode(self): return 200
    def geturl(self): return URL


class Client:
    def __init__(self, responses): self.responses, self.calls = list(responses), []
    def open(self, request, timeout):
        self.calls.append((request.full_url, timeout))
        item = self.responses.pop(0)
        if isinstance(item, Exception): raise item
        return Response(item)


def test_fetch_success_sha_cache_hit_and_only_isolated_ledger(tmp_path):
    client = Client([b"synthetic HTML"])
    budget = news.RetrievalBudget()
    text, meta = news.fetch_official(URL, tmp_path, budget, opener=client)
    assert text == "synthetic HTML" and budget.requests == 1
    assert meta["sha256"] == news.sha256_bytes(b"synthetic HTML")
    assert news.fetch_official(URL, tmp_path, budget, opener=client)[0] == text
    assert len(client.calls) == 1
    fresh_budget = news.RetrievalBudget()
    assert news.fetch_official(URL, tmp_path, fresh_budget, opener=client)[0] == text
    assert fresh_budget.requests == 1 and fresh_budget.bytes_accounted == len(b"synthetic HTML")
    assert len((tmp_path / "requests.jsonl").read_text().splitlines()) == 1


def test_transient_retry_once_and_total_bound(tmp_path):
    client = Client([URLError("synthetic timeout"), b"ok"])
    budget = news.RetrievalBudget(max_requests=2)
    assert news.fetch_official(URL, tmp_path, budget, opener=client)[0] == "ok"
    assert budget.requests == 2 and len(client.calls) == 2
    next_url = URL.replace("16022024", "22032024")
    assert news.fetch_official(next_url, tmp_path, budget, opener=client)[1]["parse_status"] == "budget_exhausted"
    assert len(client.calls) == 2


def test_failed_url_attempt_limit_persists_across_cli_runs(tmp_path):
    client = Client([URLError("synthetic timeout"), URLError("synthetic timeout")])
    assert news.fetch_official(URL, tmp_path, news.RetrievalBudget(), opener=client)[0] is None
    assert len(client.calls) == 2
    news.fetch_official(URL, tmp_path, news.RetrievalBudget(), opener=client)
    assert len(client.calls) == 2


def test_repeated_access_blocks_stop_no_retry_and_save_failure(tmp_path):
    client = Client([HTTPError(URL, 403, "synthetic forbidden", {}, None),
                     HTTPError(URL, 429, "synthetic limit", {}, None)])
    budget = news.RetrievalBudget()
    assert news.fetch_official(URL, tmp_path, budget, opener=client)[1]["parse_status"] == "access_blocked"
    next_url = URL.replace("16022024", "22032024")
    news.fetch_official(next_url, tmp_path, budget, opener=client)
    assert budget.stopped and budget.requests == 2
    news.fetch_official(URL.replace("16022024", "26042024"), tmp_path, budget, opener=client)
    assert len(client.calls) == 2
    assert len(list(tmp_path.glob("*.bin.json"))) == 2


def test_oversized_response_saved_as_failure_and_budget_accounted(tmp_path):
    client = Client([b"0123456789"])
    budget = news.RetrievalBudget(max_file_bytes=5)
    text, meta = news.fetch_official(URL, tmp_path, budget, opener=client)
    assert text is None and budget.bytes_accounted == 6
    assert "exceeds" in meta["error"]
    assert not list(tmp_path.glob("*.bin"))


def test_cached_corruption_refused(tmp_path):
    file = tmp_path / "synthetic.bin"; file.write_bytes(b"changed")
    with pytest.raises(ValueError, match="provenance mismatch"):
        news.read_verified_cached({"file": str(file), "sha256": "wrong", "file_size": 7})


def test_total_byte_limit_is_never_exceeded_even_for_unknown_oversized_body(tmp_path):
    budget = news.RetrievalBudget(max_bytes=5, max_file_bytes=100)
    text, meta = news.fetch_official(URL, tmp_path, budget, opener=Client([b"0123456789"]))
    assert text is None and budget.bytes_accounted == budget.max_bytes == 5
    assert meta["parse_status"] == "retrieval_failed"


def test_audit_every_snapshot_and_doc_unifies_PDF_release_but_not_lenta():
    decision = extract()
    records = []
    for i, (source, url, status) in enumerate([
        ("cbr", URL, "dated_primary_release_unverified"),
        ("cbr", URL, "unknown"),
        ("cbr", "https://cbr.ru/Collection/synthetic.pdf", "reviewed_original_official_pdf"),
        ("lenta", "https://lenta.ru/news/synthetic/", "unknown")]):
        records.append(dict(document_id=news.document_id(url), snapshot_id=str(i), source=source, source_url=url,
                            title="Синтетический материал", published_at="2024-02-16T10:30:00+00:00",
                            available_at="2024-02-16T10:30:00+00:00" if i==2 else RETRIEVED,
                            retrieved_at=RETRIEVED, historical_version_status=status, canonical_event_id=str(i), updated_at=None))
    result = news.audit_cached_news(pd.DataFrame(records), [], [], [decision], origins=["2023-12", "2024-02"])
    assert len(result["snapshot_audit"]) == 4 and len(result["document_summary"]) == 3
    audit = result["snapshot_audit"]
    assert audit.v3_core_event_admitted.tolist() == [True, True, True, False]
    assert audit.v3_whole_snapshot_admitted.tolist() == [False, False, True, False]
    assert audit.canonical_event_id_v3.iloc[0] == audit.canonical_event_id_v3.iloc[2]
    assert audit.strict_v2_admitted_origin_count.tolist() == [0, 0, 1, 0]
    assert result["rules"]["lenta_policy_relaxed"] is False


def test_modified_equal_publication_does_not_claim_revision_and_origin_uses_midnight():
    rows = []
    for index, update in enumerate(["2024-02-29T13:00:00+03:00", "2024-02-29T13:05:00+03:00"]):
        rows.append(dict(document_id=str(index), snapshot_id=str(index), source="lenta", source_url=str(index),
                         title="Синтетический материал", published_at="2024-02-29T13:00:00+03:00",
                         available_at="2024-02-29T13:00:00+03:00", retrieved_at=RETRIEVED,
                         updated_at=update, canonical_event_id=str(index), historical_version_status="unknown"))
    audit = news.audit_cached_news(pd.DataFrame(rows), [], [], [], origins=["2024-02"])["snapshot_audit"]
    assert audit.strict_v2_reason.tolist() == ["current_article_historical_version_unknown",
                                              "current_article_declares_revision_original_version_not_attested"]
    assert audit.strict_v2_admitted_origin_count.tolist() == [0, 0]
