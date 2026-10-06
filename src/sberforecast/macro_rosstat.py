"""Official Rosstat downloads without assumptions about historical vintages.

The page labels describe monthly all-organization wages and total regional CPI.
An up-to-date workbook is status B even when it contains 2022--2024 numbers.
Failed discovery/download/definition verification is C. Neither retrieval time,
HTTP Last-Modified nor a scenario publication lag proves status A.

XLSX cells are decoded with the standard library because openpyxl/xlrd are not
installed. No workbook layout is guessed: a verified sheet/column mapping must
be supplied separately before cells can become indicator observations.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import io
import math
from pathlib import Path
import posixpath
import re
from urllib.parse import urljoin, urlsplit
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET
from zipfile import BadZipFile, ZipFile

import pandas as pd


ROSSTAT_SOURCES = (
    {
        "source": "Rosstat",
        "url": "https://rosstat.gov.ru/labor_market_employment_salaries",
        "name": "Среднемесячная номинальная начисленная заработная плата работников по полному кругу организаций по субъектам Российской Федерации с 2013 года (по месяцам), рублей",
        "indicator": "nominal_wage_rub",
        "unit": "RUB",
        "geographic_level": "RF_subject",
        "data_period_description": "с 2013 года, месячные данные; наличие 2022–2024 требует чтения книги",
        "definition_terms": ("по полному кругу организаций", "по субъектам", "по месяцам"),
        "definition_note": "Начисленная заработная плата работников организаций, не доход всех жителей МО; месячные и накопленные с начала года значения не смешиваются.",
    },
    {
        "source": "Rosstat",
        "url": "https://rosstat.gov.ru/statistics/price",
        "name": "Индексы потребительских цен на товары и услуги по Российской Федерации, федеральным округам и субъектам Российской Федерации (с 2022 г.)",
        "indicator": "cpi_mom_index,cpi_yoy_index",
        "unit": "index_percent",
        "geographic_level": "RF_subject",
        "data_period_description": "с 2022 года; месячные MoM/YoY определения требуют чтения заголовков книги",
        "definition_terms": ("Индексы потребительских цен", "субъектам", "2022"),
        "definition_note": "Общий ИПЦ всех товаров и услуг: MoM и YoY должны различаться по заголовкам; индекс 101% означает рост 1%, но не уровень цен.",
    },
)
XLSX_NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
MAX_DOWNLOAD_BYTES = 20 * 1024 * 1024


def _official_url(url: str) -> None:
    parsed = urlsplit(url)
    hostname = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or parsed.username or parsed.password or not (
        hostname == "rosstat.gov.ru" or hostname.endswith(".rosstat.gov.ru")
    ):
        raise ValueError("Only official public Rosstat HTTPS URLs are allowed.")


def download_rosstat(url: str, timeout: float = 20.0) -> tuple[bytes, dict]:
    """No authentication, disabled TLS checks, or alternate access mechanisms."""
    _official_url(url)
    with urlopen(Request(url, headers={"User-Agent": "sberforecast E05a source audit"}), timeout=timeout) as response:
        _official_url(response.geturl())
        content = response.read(MAX_DOWNLOAD_BYTES + 1)
        if len(content) > MAX_DOWNLOAD_BYTES:
            raise ValueError("Rosstat download exceeds the explicit 20 MiB bound.")
        return content, {
            "response_url": response.geturl(),
            "content_type": response.headers.get("Content-Type"),
            "http_last_modified": response.headers.get("Last-Modified"),
        }


class _PageLinks(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.links: list[dict] = []
        self.active: list[dict] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                link = {"href": href, "part_index": len(self.parts), "text": ""}
                self.links.append(link)
                self.active.append(link)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self.active:
            self.active.pop()

    def handle_data(self, data: str) -> None:
        normalized = " ".join(data.split())
        if normalized:
            self.parts.append(normalized)
            for link in self.active:
                link["text"] += " " + normalized


def discover_rosstat_links(html: str, page_url: str, definition_terms: tuple[str, ...]) -> list[dict]:
    """Return candidates with contextual evidence; never certify their schema.

    Generic XLSX icons can precede the card title. The bounded neighbouring
    text is retained for human verification rather than treated as a definition.
    """
    _official_url(page_url)
    parser = _PageLinks()
    parser.feed(html)
    candidates = []
    seen = set()
    for link in parser.links:
        url = urljoin(page_url, link["href"])
        try:
            _official_url(url)
        except ValueError:
            continue
        if not urlsplit(url).path.lower().endswith((".xlsx", ".xls", ".htm", ".html")):
            continue
        index = link["part_index"]
        context = " ".join(parser.parts[max(0, index - 5):index + 9])[:2400]
        if all(term.casefold() in context.casefold() for term in definition_terms) and url not in seen:
            candidates.append({"url": url, "anchor_text": link["text"].strip(), "definition_context": context})
            seen.add(url)
    return candidates


def audit_rosstat_sources(raw_dir: str | Path, fetcher=None) -> list[dict]:
    """Download two supplied pages, retain at most one candidate workbook each.

    Returns catalogue records, not invented indicator observations. A missing
    verified layout is explicit even if the file's ZIP/XML can be decoded.
    A supplied fetcher makes tests and cached auditing network-independent.
    """
    fetcher = download_rosstat if fetcher is None else fetcher
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    catalogue = []
    for spec in ROSSTAT_SOURCES:
        record = {key: value for key, value in spec.items() if key != "definition_terms"}
        record.update({
            "published_at": None, "vintage_published_at": None, "updated_at": None,
            "downloaded_at": None, "attempted_at": datetime.now(timezone.utc).isoformat(),
            "sha256": None, "availability_status": "C", "status": "download_failed",
            "errors": [], "files": [], "historical_availability_verified": False,
        })
        try:
            content, headers = fetcher(spec["url"])
            timestamp = datetime.now(timezone.utc).isoformat()
            digest = hashlib.sha256(content).hexdigest()
            page = raw_dir / f"{spec['indicator'].split(',')[0]}_{digest[:16]}.html"
            if page.exists() and page.read_bytes() != content:
                raise ValueError("Existing raw file differs; refusing to overwrite.")
            page.write_bytes(content)
            record.update({"downloaded_at": timestamp, "sha256": digest, "raw_path": str(page), "headers": headers, "status": "page_downloaded_file_unverified"})
            candidates = discover_rosstat_links(content.decode("utf-8", errors="replace"), spec["url"], spec["definition_terms"])
            record["file_candidates"] = candidates
            workbooks = [c for c in candidates if urlsplit(c["url"]).path.lower().endswith((".xlsx", ".xls"))]
            if not workbooks:
                record["errors"].append("No actual monthly regional workbook hyperlink verified on the downloaded page.")
                catalogue.append(record)
                continue
            candidate = workbooks[0]
            file_record = dict(candidate, availability_status="C", published_at=None, vintage_published_at=None, downloaded_at=None, sha256=None)
            record["files"].append(file_record)
            binary, file_headers = fetcher(candidate["url"])
            file_digest = hashlib.sha256(binary).hexdigest()
            suffix = Path(urlsplit(candidate["url"]).path).suffix.lower()
            file_path = raw_dir / f"{spec['indicator'].split(',')[0]}_{file_digest[:16]}{suffix}"
            if file_path.exists() and file_path.read_bytes() != binary:
                raise ValueError("Existing raw workbook differs; refusing to overwrite.")
            file_path.write_bytes(binary)
            file_record.update({"downloaded_at": datetime.now(timezone.utc).isoformat(), "sha256": file_digest, "raw_path": str(file_path), "headers": file_headers})
            if suffix == ".xlsx":
                decoded = read_xlsx_cells(binary)
                file_record.update({"sheet_names": list(decoded), "status": "decoded_layout_not_verified"})
            else:
                file_record.update({"status": "format_dependency_unavailable", "error": "xlrd is not installed; legacy XLS is retained but not decoded."})
            record["status"] = "file_downloaded_definition_unverified"
            record["errors"].append("Workbook layout and exact MoM/YoY or monthly wage columns not verified; no numeric observations admitted.")
        except Exception as exc:
            record["errors"].append(f"{type(exc).__name__}: {exc}")
        catalogue.append(record)
    return catalogue


def read_xlsx_cells(content: bytes) -> dict[str, dict[int, dict[int, str | float]]]:
    """Decode sparse cell coordinates; no dependency, formula evaluation or fill."""
    try:
        archive = ZipFile(io.BytesIO(content))
    except BadZipFile as exc:
        raise ValueError("Expected a real XLSX ZIP archive, not an HTML error page.") from exc
    with archive:
        if sum(item.file_size for item in archive.infolist()) > 100 * 1024 * 1024:
            raise ValueError("Expanded XLSX exceeds 100 MiB.")
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        targets = {item.attrib["Id"]: item.attrib["Target"] for item in relationships}
        shared_strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            shared_strings = ["".join(cell.itertext()) for cell in shared_root.findall("s:si", XLSX_NS)]
        result = {}
        relationship_key = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
        for sheet in workbook.findall("s:sheets/s:sheet", XLSX_NS):
            target = targets[sheet.attrib[relationship_key]]
            member = target.lstrip("/") if target.startswith("/") else posixpath.normpath("xl/" + target)
            if not member.startswith("xl/") or ".." in member.split("/"):
                raise ValueError("Worksheet relationship leaves the XLSX xl directory.")
            sheet_root = ET.fromstring(archive.read(member))
            rows: dict[int, dict[int, str | float]] = {}
            for cell in sheet_root.findall("s:sheetData/s:row/s:c", XLSX_NS):
                address = re.fullmatch(r"([A-Z]+)([1-9][0-9]*)", cell.attrib.get("r", ""))
                if address is None:
                    raise ValueError("Explicit cell coordinates are required.")
                column = 0
                for letter in address[1]:
                    column = column * 26 + ord(letter) - ord("A") + 1
                value_node = cell.find("s:v", XLSX_NS)
                raw = value_node.text if value_node is not None else None
                kind = cell.attrib.get("t", "n")
                if kind == "inlineStr":
                    value = "".join(cell.find("s:is", XLSX_NS).itertext())
                elif raw is None:
                    continue
                elif kind == "s":
                    value = shared_strings[int(raw)]
                elif kind in ("str", "e", "b"):
                    value = raw
                else:
                    value = float(raw)
                    if not math.isfinite(value):
                        raise ValueError("Non-finite numeric XLSX cell.")
                row_number = int(address[2])
                if column in rows.setdefault(row_number, {}):
                    raise ValueError("Duplicate XLSX cell coordinates.")
                rows[row_number][column] = value
            result[sheet.attrib["name"]] = rows
        return result


def normalize_verified_monthly_records(records: list[dict], *, indicator: str,
                                       source_url: str, retrieved_at: str,
                                       vintage_id: str, region_mapping: dict[str, str]) -> pd.DataFrame:
    """For an explicitly verified layout only; exact geographic names required.

    These current-table rows remain B with unknown publication/availability.
    Scenario lags are handled elsewhere without relabelling them as A.
    """
    _official_url(source_url)
    allowed = {"nominal_wage_rub": "RUB", "cpi_mom_index": "index_percent", "cpi_yoy_index": "index_percent"}
    if indicator not in allowed:
        raise ValueError("Only explicitly defined monthly Rosstat indicators are supported.")
    output = []
    for record in records:
        name = record["region_name"]
        if name not in region_mapping:
            raise ValueError(f"Unresolved exact Rosstat geographic label: {name}")
        period = str(record["reference_period"])
        if not re.fullmatch(r"\d{4}-\d{2}", period):
            raise ValueError("Expected a monthly YYYY-MM reference period, not an annual/YTD label.")
        period = str(pd.Period(period, freq="M"))
        value = float(record["value"])
        if not math.isfinite(value) or value <= 0:
            raise ValueError("A positive finite wage/CPI observation is required.")
        output.append({"region_id": str(region_mapping[name]), "region_name": name, "indicator": indicator,
                       "reference_period": period, "value": value, "unit": allowed[indicator],
                       "published_at": None, "available_at": None, "vintage_published_at": None,
                       "vintage_id": vintage_id, "availability_status": "B", "source_url": source_url,
                       "retrieved_at": retrieved_at, "geographic_level": "RF_subject", "actual_forecast": "actual",
                       "aggregation_basis": "month_to_month" if indicator == "cpi_mom_index" else "year_on_year" if indicator == "cpi_yoy_index" else "month"})
    frame = pd.DataFrame(output)
    if not frame.empty and frame.duplicated(["region_id", "indicator", "reference_period", "vintage_id"]).any():
        raise ValueError("Duplicate region/indicator/period/vintage would multiply joins.")
    return frame
