"""Synthetic source fixtures only: no downloaded financial values or HTTP calls."""
from hashlib import sha256

import pandas as pd
import pytest

from sberforecast import leading_financial_sources as sources


def _release(*, day="7 мая 2021 года", footer="07.05.2021 14:15:23", rate="2,50"):
    return (
        f'<div class="news-info-line_date">{day}</div>'
        '<h1>Синтетический fixture решения по ставке</h1>'
        '<div class="landing-text"><p>'
        f'Совет директоров Банка России {day} принял решение повысить ключевую ставку '
        f'до {rate}% годовых.</p><p class="note">{footer}</p></div>'
    )


def _calendar_row():
    return {
        "decision_date": "2021-05-07",
        "source_url": "https://www.cbr.ru/press/pr/?file=07052021_141523key.htm",
        "archive_url": sources.KEY_DECISION_CALENDAR,
        "archive_row_evidence": "7 мая 2021 года Синтетический fixture",
    }


def _provenance():
    return {"source_sha256": "synthetic-fixture-sha", "retrieved_at": "2026-10-07T00:00:00+00:00",
            "raw_path": "outputs/synthetic-fixture/release.html"}


def test_publication_footer_preserves_actual_seconds_and_precision():
    row = sources.parse_key_decision(_release(), _calendar_row(), _provenance())
    assert row["announced_at"] == "2021-05-07T14:15:23+03:00"
    assert row["time_precision"] == "timestamp"
    assert row["effective_date"] is None
    assert row["rate"] == 2.5


@pytest.mark.parametrize("footer", ["07.05.2021 00:00:00", "7 мая 2021 года", ""])
def test_uncorroborated_intraday_time_remains_date_only(footer):
    row = sources.parse_key_decision(_release(footer=footer), _calendar_row(), _provenance())
    assert row["announced_at"] is None
    assert row["time_precision"] == "date_only"
    assert row["publication_date"] == "2021-05-07"


def test_publication_header_disagreement_is_rejected():
    with pytest.raises(ValueError, match="header"):
        sources.parse_key_decision(_release(day="8 мая 2021 года"), _calendar_row(), _provenance())


def test_effective_transition_uses_daily_evidence_instead_of_monday_rule():
    decision = sources.parse_key_decision(_release(), _calendar_row(), _provenance())
    daily = pd.DataFrame({"date": ["2021-05-07", "2021-05-11"], "rate": [2.0, 2.5]})
    result = sources.resolve_effective_dates([decision], daily)
    assert result[0]["effective_date"] == "2021-05-11"
    assert result[0]["effective_date_basis"] == "observed_transition_in_official_daily_history"
    assert decision["effective_date"] is None


def test_effective_core_and_daily_conflict_is_rejected():
    decision = sources.parse_key_decision(_release(), _calendar_row(), _provenance())
    decision["effective_date"] = "2021-05-10"
    with pytest.raises(ValueError, match="conflicts"):
        sources.resolve_effective_dates([decision], [{"DT": "2021-05-07", "Rate": 2.0},
                                                    {"DT": "2021-05-11", "Rate": 2.5}])


def test_missing_transition_or_conflicting_daily_history_is_rejected():
    decision = sources.parse_key_decision(_release(), _calendar_row(), _provenance())
    with pytest.raises(ValueError, match="different counts"):
        sources.resolve_effective_dates([decision], [{"date": "2021-05-07", "rate": 2.0}])
    with pytest.raises(ValueError, match="Conflicting"):
        sources.resolve_effective_dates([decision], [{"date": "2021-05-07", "rate": 2.0},
                                                    {"date": "2021-05-07", "rate": 2.5}])


@pytest.mark.parametrize("url", ["https://example.org/release", "http://www.cbr.ru/",
                                "https://user@www.cbr.ru/"])
def test_unofficial_or_credentialed_url_is_rejected_without_network(tmp_path, monkeypatch, url):
    monkeypatch.setattr(sources, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(sources, "download_cbr", lambda *args, **kwargs: pytest.fail("HTTP forbidden"))
    with pytest.raises(ValueError, match="official"):
        sources.cached_cbr_document(url, tmp_path / "outputs" / "fixture.html", download=True)


def test_offline_missing_cache_never_downloads(tmp_path, monkeypatch):
    monkeypatch.setattr(sources, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(sources, "download_cbr", lambda *args, **kwargs: pytest.fail("HTTP forbidden"))
    with pytest.raises(FileNotFoundError):
        sources.cached_cbr_document(sources.KEY_DECISION_CALENDAR,
                                    tmp_path / "outputs" / "fixture.html", download=False)


def test_collector_offline_determinism_and_hash_rejection(tmp_path, monkeypatch):
    monkeypatch.setattr(sources, "_PROJECT_ROOT", tmp_path)
    calendar = (
        '<div>7 мая 2021 года <a href="/press/pr/?file=07052021_141523key.htm">'
        'Синтетический fixture</a></div>'
    ).encode("utf-8")
    release = _release().encode("utf-8")
    calls = []

    def synthetic_download(url, timeout):
        calls.append(url)
        content = calendar if url == sources.KEY_DECISION_CALENDAR else release
        return content, {"response_url": url, "content_type": "text/html; charset=utf-8"}

    monkeypatch.setattr(sources, "download_cbr", synthetic_download)
    cache = tmp_path / "outputs" / "synthetic-fixture" / "cache" / "key_decisions"
    cache.parent.mkdir(parents=True)
    history = (
        '<root><KR><DT>2021-05-07T00:00:00+03:00</DT><Rate>2.0</Rate></KR>'
        '<KR><DT>2021-05-11T00:00:00+03:00</DT><Rate>2.5</Rate></KR></root>'
    ).encode("utf-8")
    (cache.parent / "key_rate_2021_2024.xml").write_bytes(history)
    first = sources.collect_key_decisions(cache, download=True)
    assert len(calls) == 2
    monkeypatch.setattr(sources, "download_cbr", lambda *args, **kwargs: pytest.fail("HTTP forbidden"))
    assert sources.collect_key_decisions(cache, download=False) == first
    assert first[0]["source_sha256"] == sha256(release).hexdigest()
    (cache / "2021-05-07.html").write_bytes(b"synthetic cache corruption")
    with pytest.raises(ValueError, match="hash mismatch"):
        sources.collect_key_decisions(cache, download=False)
