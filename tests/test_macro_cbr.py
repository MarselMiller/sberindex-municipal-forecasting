"""Offline checks using synthetic publications, not real downloaded data."""
from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from sberforecast import macro_cbr


def dated_page(body: str, *, date: str = "25 декабря 2023 года", updated: str = "25.12.2023") -> str:
    return (
        '<div class="news-info-line_date">' + date + '</div>' + body
        + '<div class="page-info_last-update">Последнее обновление страницы: ' + updated + '</div>'
    )


def test_archived_comment_keeps_participants_and_conservative_date():
    html = dated_page(
        '<p>По данным декабрьского макроэкономического опроса Банка России, '
        'прогноз аналитиков по инфляции на конец 2023 года составил 7,6%. '
        'Ожидания аналитиков на конец 2024 года не изменились, их медиана составила 5,1%. '
        'В 2025 году и далее аналитики ожидают инфляцию 4,0%.</p>'
        '<p>Согласно прогнозу Банка России, инфляция составит 4,0–4,5% в 2024 году.</p>'
    )
    result = macro_cbr.parse_inflation_comment(html, macro_cbr.INFLATION_START)
    assert result.set_index("target_period").value.to_dict() == {"2023": 7.6, "2024": 5.1, "2025": 4.0}
    assert result.forecast_producer.eq("survey_participants").all()
    assert result.region_id.eq("RU").all()
    assert result.aggregation_basis.eq("december_to_december").all()
    assert result.availability_status.eq("A").all()
    assert result.available_at.eq(pd.Timestamp("2023-12-25")).all()
    assert result.forecast_issue_date.eq(pd.Timestamp("2023-12-25")).all()
    assert result.original_forecast_issue_date.isna().all()
    assert result.survey_issue_period.eq("2023-12").all()


def test_vague_expectation_and_percentage_point_changes_are_not_precise_levels():
    html = dated_page(
        '<p>По данным декабрьского макроэкономического опроса, прогноз аналитиков '
        'на конец 2024 года составил 5,1% (+0,6 п.п.). '
        'В 2025 году они ожидают инфляцию вблизи 4,0%.</p>'
    )
    result = macro_cbr.parse_inflation_comment(html, macro_cbr.INFLATION_START)
    assert result.target_period.tolist() == ["2024"]
    assert result.value.tolist() == [5.1]


def test_separator_before_median_is_not_misread_as_interval():
    html = dated_page('<p>По данным октябрьского макроэкономического опроса, '
                      'прогноз на конец 2024 года – 5,1%.</p>')
    result = macro_cbr.parse_inflation_comment(html, macro_cbr.INFLATION_START)
    assert result.value.tolist() == [5.1]


def test_later_revision_is_not_historical_a():
    html = dated_page('<p>По данным декабрьского макроэкономического опроса, '
                      'прогноз на конец 2024 года составил 5,1%.</p>', updated="02.09.2026")
    result = macro_cbr.parse_inflation_comment(html, macro_cbr.INFLATION_START)
    assert result.availability_status.tolist() == ["B"]
    assert result.published_at.tolist() == [pd.Timestamp("2023-12-25")]
    assert result.available_at.tolist() == [pd.Timestamp("2026-09-02")]


def test_republished_previous_year_survey_keeps_original_month_separate():
    html = dated_page('<p>По данным декабрьского макроэкономического опроса, '
                      'прогноз на конец 2024 года составил 5,1%.</p>',
                      date="31 января 2024 года", updated="31.01.2024")
    result = macro_cbr.parse_inflation_comment(html, macro_cbr.INFLATION_START)
    assert result.survey_issue_period.tolist() == ["2023-12"]
    assert result.available_at.tolist() == [pd.Timestamp("2024-01-31")]


@pytest.mark.parametrize("text,expected", [
    ("−2,0–(−1,0)", (-2.0, -1.0)),
    ("-2,0-(-1,0)", (-2.0, -1.0)),
    ("0,5–1,5", (0.5, 1.5)),
    ("4,0", (4.0, 4.0)),
    ("(-1,0)", (-1.0, -1.0)),
])
def test_forecast_intervals_keep_sign_and_units(text, expected):
    assert macro_cbr.parse_forecast_interval(text) == expected


@pytest.mark.parametrize("text", ["5–4", "вблизи 4%", "", "n/a"])
def test_forecast_intervals_reject_ambiguous_values(text):
    with pytest.raises(ValueError):
        macro_cbr.parse_forecast_interval(text)


def test_consumption_is_annual_real_national_and_midpoint_is_declared():
    html = dated_page(
        '<table><tr><th></th><th>2022 (факт)</th><th>2023</th><th>2024</th></tr>'
        '<tr><td>Расходы на конечное потребление домашних хозяйств</td>'
        '<td>−1,4</td><td>5,5–6,5</td><td>-2,0–(-1,0)</td></tr></table>',
        date="7 ноября 2023 года", updated="07.11.2023",
    )
    result = macro_cbr.parse_consumption_longread(html, macro_cbr.CONSUMPTION_START)
    assert result.target_period.tolist() == ["2023", "2024"]
    assert result.value.tolist() == [6.0, -1.5]
    assert result.forecast_low.tolist() == [5.5, -2.0]
    assert result.forecast_high.tolist() == [6.5, -1.0]
    assert result.unit.eq("percent_growth").all()
    assert result.central_method.eq("interval_midpoint_not_official_central").all()
    assert result.aggregation_basis.eq("annual_real_volume_growth").all()
    assert result.forecast_producer.eq("Bank of Russia").all()
    assert result.geographic_level.eq("national").all()


def test_consumption_requires_verified_header_and_household_definition():
    html = dated_page('<table><tr><td>Расходы на конечное потребление</td><td>3,0–4,0</td></tr></table>')
    with pytest.raises(ValueError, match="household"):
        macro_cbr.parse_consumption_longread(html, macro_cbr.CONSUMPTION_START)


def test_survey_month_does_not_become_publication_day(monkeypatch):
    issue = (pd.Timestamp("2023-12-01") - pd.Timestamp("1899-12-30")).days
    target = (pd.Timestamp("2024-12-31") - pd.Timestamp("1899-12-30")).days
    definitions = {
        "1": "ИПЦ (в % дек. к дек. пред. года)",
        "2": "ИПЦ (% к пред. году, в среднем за год)",
        "7": "Номинальная заработная плата (%, г/г, в среднем за год)",
    }
    cells = {
        name: {
            4: {2: title}, 6: {4: "Прогнозный период", 5: issue},
            8: {3: "Median", 4: target, 5: 5.1},
            17: {3: "Average", 4: target, 5: 6.0},
            26: {3: "Max", 4: target, 5: 8.0},
            71: {3: "Min", 4: target, 5: 3.0},
        } for name, title in definitions.items()
    }
    monkeypatch.setattr(macro_cbr, "read_xlsx_cells", lambda content: cells)
    result = macro_cbr.parse_survey_workbook(b"synthetic workbook handled by tested XLSX reader", "https://cbr.ru/full.xlsx", vintage_id="fixture")
    assert len(result) == 3
    assert result.value.eq(5.1).all()
    assert result.forecast_low.eq(3.0).all()
    assert result.forecast_high.eq(8.0).all()
    assert result.published_at.isna().all()
    assert result.available_at.isna().all()
    assert result.forecast_issue_date.isna().all()
    assert result.availability_status.eq("B").all()
    assert result.survey_issue_period.eq("2023-12").all()
    assert not result.indicator.str.contains("real_wage").any()


def test_source_discovery_follows_existing_links_and_allowed_dates():
    html = (
        '<a href="/analytics/dkp/inflationary_expectations/Infl_exp_23-12/">archive</a>'
        '<a href="/analytics/dkp/inflationary_expectations/Infl_exp_24-12/">future</a>'
        '<a href="https://other.example/Infl_exp_23-12/">unofficial</a>'
    )
    result = macro_cbr.discover_cbr_links(html, macro_cbr.INFLATION_START, "inflation_comment")
    assert len(result) == 1
    assert result[0]["file"] == "inflation_2023_12.html"


def test_offline_collection_records_missing_files_without_network(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Ordinary tests cannot access the network.")
    monkeypatch.setattr(macro_cbr, "download_cbr", forbidden)
    rows, catalog = macro_cbr.collect_cbr_sources(tmp_path)
    assert rows.empty
    assert len(catalog) == 3
    assert all(c["availability_status"] == "C" for c in catalog)
    assert all("FileNotFoundError" in c["error"] for c in catalog)


def test_source_download_rejects_nonofficial_or_credential_urls():
    with pytest.raises(ValueError):
        macro_cbr.download_cbr("https://user:password@cbr.ru/full.xlsx")
    with pytest.raises(ValueError):
        macro_cbr.download_cbr("http://cbr.ru/full.xlsx")
    with pytest.raises(ValueError):
        macro_cbr.download_cbr("https://unofficial.example/full.xlsx")


PDF_TEXT = """Среднесрочный прогноз Банка России
по итогам заседания Совета директоров по ключевой ставке 25 октября 2024 года
Основные параметры прогноза Банка России в рамках базового сценария
(прирост в % к предыдущему году, если не указано иное)
                       2023          2024        2025
                      (факт)
Инфляция                  7,4          8,0-8,5      4,5-5,0
Расходы на конечное потребление  6,6  3,5-4,5  0,0-1,0
 – домашних хозяйств  6,5  4,5-5,5  0,0-1,0
Источник: Банк России.
"""


def write_pdf_cache(tmp_path):
    """Synthetic bytes and publication evidence; no downloaded real data."""
    url = "https://cbr.ru/Content/Document/File/1/forecast_241025.pdf"
    pdf = b"%PDF-1.4\nSynthetic offline fixture, not a real CBR document."
    text = PDF_TEXT.encode("utf-8")
    evidence = ('<div class="document-regular_date">25.10.2024</div>'
                '<a href="/Content/Document/File/1/forecast_241025.pdf">forecast</a>').encode("utf-8")
    for name, content in [("synthetic.pdf", pdf), ("synthetic.txt", text), ("synthetic_index.html", evidence)]:
        (tmp_path / name).write_bytes(content)
    record = {
        "pdf_file": "synthetic.pdf", "text_file": "synthetic.txt", "source_url": url,
        "published_at": "2024-10-25", "pdf_sha256": hashlib.sha256(pdf).hexdigest(),
        "text_sha256": hashlib.sha256(text).hexdigest(),
        "publication_evidence_file": "synthetic_index.html",
        "publication_evidence_url": macro_cbr.KEY_RATE_ARCHIVE,
        "publication_evidence_sha256": hashlib.sha256(evidence).hexdigest(),
        "reviewed": True, "extraction_returncode": 1, "extraction_warning": "Synthetic cleanup failure",
    }
    (tmp_path / "consumption_pdf_extracts.json").write_text(json.dumps([record]), encoding="utf-8")
    return record


def test_pdf_household_subrow_is_distinct_from_total_consumption_and_actual():
    result = macro_cbr.parse_consumption_pdf_text(PDF_TEXT, "https://cbr.ru/forecast.pdf", "2024-10-25")
    assert result.target_period.tolist() == ["2024", "2025"]
    assert result.value.tolist() == [5.0, 0.5]
    assert result.forecast_low.tolist() == [4.5, 0.0]
    assert result.is_forecast.all()
    assert result.forecast_producer.eq("Bank of Russia").all()


def test_pdf_issue_date_must_agree_with_dated_archive():
    with pytest.raises(ValueError, match="date differs"):
        macro_cbr.parse_consumption_pdf_text(PDF_TEXT, "https://cbr.ru/forecast.pdf", "2024-10-24")


def test_pdf_requires_complete_values_and_factual_column():
    with pytest.raises(ValueError, match="values"):
        macro_cbr.parse_consumption_pdf_text(PDF_TEXT.replace("6,5  4,5-5,5  0,0-1,0", "6,5  4,5-5,5"),
                                             "https://cbr.ru/forecast.pdf", "2024-10-25")
    with pytest.raises(ValueError, match="factual"):
        macro_cbr.parse_consumption_pdf_text(PDF_TEXT.replace("(факт)", ""), "https://cbr.ru/forecast.pdf", "2024-10-25")


def test_reviewed_pdf_cache_is_offline_and_preserves_nonzero_extraction_exit(tmp_path, monkeypatch):
    write_pdf_cache(tmp_path)
    monkeypatch.setattr(macro_cbr, "download_cbr", lambda *args: pytest.fail("Network must not be called."))
    rows, catalog = macro_cbr.collect_cbr_pdf_sources(tmp_path)
    assert len(rows) == 2
    assert catalog[0]["availability_status"] == "A"
    assert catalog[0]["extraction_returncode"] == 1
    assert catalog[0]["extraction_warning"] == "Synthetic cleanup failure"
    assert catalog[0]["text_sha256"] == rows.source_text_sha256.iloc[0]


@pytest.mark.parametrize("filename", ["synthetic.pdf", "synthetic.txt", "synthetic_index.html"])
def test_pdf_cache_rejects_changed_source_text_or_evidence(tmp_path, filename):
    write_pdf_cache(tmp_path)
    with (tmp_path / filename).open("ab") as handle:
        handle.write(b"changed")
    rows, catalog = macro_cbr.collect_cbr_pdf_sources(tmp_path)
    assert rows.empty
    assert catalog[0]["availability_status"] == "C"
    assert "SHA256 mismatch" in catalog[0]["error"]


def test_pdf_cache_rejects_unreviewed_or_undated_evidence(tmp_path):
    record = write_pdf_cache(tmp_path)
    record["reviewed"] = False
    (tmp_path / "consumption_pdf_extracts.json").write_text(json.dumps([record]), encoding="utf-8")
    rows, catalog = macro_cbr.collect_cbr_pdf_sources(tmp_path)
    assert rows.empty
    assert "reviewed" in catalog[0]["error"]
    record["reviewed"] = True
    evidence = '<a href="/Content/Document/File/1/forecast_241025.pdf">forecast</a>'.encode()
    (tmp_path / "synthetic_index.html").write_bytes(evidence)
    record["publication_evidence_sha256"] = hashlib.sha256(evidence).hexdigest()
    (tmp_path / "consumption_pdf_extracts.json").write_text(json.dumps([record]), encoding="utf-8")
    rows, catalog = macro_cbr.collect_cbr_pdf_sources(tmp_path)
    assert rows.empty
    assert "dated archive entry" in catalog[0]["error"]
