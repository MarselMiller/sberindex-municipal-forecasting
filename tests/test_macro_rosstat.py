"""Offline public synthetic fixtures; these are not Rosstat observations."""
import io
from zipfile import ZipFile

import pandas as pd
import pytest

from sberforecast.macro_rosstat import (
    ROSSTAT_SOURCES, audit_rosstat_sources, discover_rosstat_links,
    normalize_verified_monthly_records, read_xlsx_cells,
)


def _xlsx() -> bytes:
    stream = io.BytesIO()
    with ZipFile(stream, "w") as archive:
        archive.writestr("xl/workbook.xml", '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="Synthetic" sheetId="1" r:id="rId1"/></sheets></workbook>')
        archive.writestr("xl/_rels/workbook.xml.rels", '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>')
        archive.writestr("xl/sharedStrings.xml", '<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><si><t>Синтетический регион</t></si></sst>')
        archive.writestr("xl/worksheets/sheet1.xml", '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData><row r="1"><c r="A1" t="s"><v>0</v></c><c r="C1"><v>101.5</v></c></row><row r="3"><c r="AA3" t="inlineStr"><is><t>2023-03</t></is></c><c r="AC3"><v>45000</v></c></row></sheetData></worksheet>')
    return stream.getvalue()


def test_sparse_xlsx_positions_are_not_compressed_or_filled():
    decoded = read_xlsx_cells(_xlsx())["Synthetic"]
    assert decoded == {1: {1: "Синтетический регион", 3: 101.5}, 3: {27: "2023-03", 29: 45000.0}}
    assert 2 not in decoded and 2 not in decoded[1] and 28 not in decoded[3]


def test_html_failure_is_not_interpreted_as_workbook():
    with pytest.raises(ValueError, match="real XLSX"):
        read_xlsx_cells(b"<html>access failed</html>")


def test_link_discovery_uses_actual_official_link_and_retains_evidence():
    spec = ROSSTAT_SOURCES[0]
    html = '<p>По полному кругу организаций по субъектам РФ (по месяцам)</p><a href="/storage/mediabank/synthetic.xlsx">XLSX</a><a href="https://unofficial.example/forged.xlsx">копия</a>'
    links = discover_rosstat_links(html, spec["url"], spec["definition_terms"])
    assert len(links) == 1
    assert links[0]["url"] == "https://rosstat.gov.ru/storage/mediabank/synthetic.xlsx"
    assert "По полному" in links[0]["definition_context"]


def test_audit_download_failure_is_explicit_and_has_no_invented_dates(tmp_path):
    def unavailable(url):
        raise OSError("synthetic offline failure")
    records = audit_rosstat_sources(tmp_path, fetcher=unavailable)
    assert len(records) == 2
    assert all(r["availability_status"] == "C" and r["status"] == "download_failed" for r in records)
    assert all(r["downloaded_at"] is None and r["published_at"] is None and r["sha256"] is None for r in records)
    assert all("synthetic offline failure" in r["errors"][0] for r in records)
    assert list(tmp_path.iterdir()) == []


def test_downloaded_decoded_cells_do_not_certify_indicator_definition(tmp_path):
    spec = ROSSTAT_SOURCES[0]
    html = '<p>По полному кругу организаций по субъектам РФ (по месяцам)</p><a href="/storage/mediabank/synthetic.xlsx">XLSX</a>'.encode()
    def fetch(url):
        return (html if url == spec["url"] else _xlsx()), {}
    records = audit_rosstat_sources(tmp_path, fetcher=fetch)
    assert records[0]["status"] == "file_downloaded_definition_unverified"
    assert records[0]["availability_status"] == "C"
    assert records[0]["files"][0]["sheet_names"] == ["Synthetic"]
    assert records[0]["downloaded_at"] and len(records[0]["sha256"]) == 64
    assert records[0]["published_at"] is None


def _normalize(records, indicator="nominal_wage_rub", mapping=None):
    return normalize_verified_monthly_records(
        records, indicator=indicator, source_url=ROSSTAT_SOURCES[0]["url"],
        retrieved_at="2026-10-06T00:00:00+00:00", vintage_id="current_test",
        region_mapping=mapping or {"Синтетический регион": "01"},
    )


def test_verified_current_table_remains_b_and_has_no_assumed_publication_lag():
    frame = _normalize([{"region_name": "Синтетический регион", "reference_period": "2023-01", "value": 45000}])
    assert frame.loc[0, "availability_status"] == "B"
    assert frame.loc[0, "region_id"] == "01"
    assert frame.loc[0, "value"] == 45000
    assert frame.loc[0, "unit"] == "RUB"
    assert frame[["published_at", "available_at", "vintage_published_at"]].isna().all().all()


def test_cpi_index_is_not_silently_changed_into_growth_rate():
    frame = _normalize([{"region_name": "Синтетический регион", "reference_period": "2023-01", "value": 101}], "cpi_mom_index")
    assert frame.loc[0, "value"] == 101
    assert frame.loc[0, "unit"] == "index_percent"
    assert frame.loc[0, "aggregation_basis"] == "month_to_month"


@pytest.mark.parametrize("period", ["2023", "Январь-март 2023", "2023-13"])
def test_annual_ytd_or_invalid_month_are_rejected(period):
    with pytest.raises(ValueError):
        _normalize([{"region_name": "Синтетический регион", "reference_period": period, "value": 45000}])


def test_exact_geography_required_without_fuzzy_matching():
    with pytest.raises(ValueError, match="Unresolved exact"):
        _normalize([{"region_name": "Синтетич. регион", "reference_period": "2023-01", "value": 45000}])


def test_duplicate_geographic_rows_rejected_before_join():
    row = {"region_name": "Синтетический регион", "reference_period": "2023-01", "value": 45000}
    with pytest.raises(ValueError, match="multiply joins"):
        _normalize([row, row.copy()])


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0, -1])
def test_required_anchor_values_are_finite_and_positive(value):
    with pytest.raises(ValueError, match="positive finite"):
        _normalize([{"region_name": "Синтетический регион", "reference_period": "2023-01", "value": value}])


def test_normalization_is_deterministic():
    records = [{"region_name": "Синтетический регион", "reference_period": "2023-01", "value": 45000}]
    pd.testing.assert_frame_equal(_normalize(records), _normalize(records))
