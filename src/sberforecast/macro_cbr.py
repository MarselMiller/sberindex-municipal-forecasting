"""Public CBR sources, keeping publication dates and survey editions distinct.

The current aggregate survey workbook is B: its month labels do not establish
the publication day or that every historical number is unrevised. Archived
dated commentaries provide A evidence for their explicitly stated inflation
medians, conservatively from the commentary's own publication date. These are
the participants' forecasts, not the Bank's forecasts. Consumption ranges in
dated monetary-policy publications are the Bank's national real-volume
forecasts. Their midpoint is a declared derived value, not an official central
forecast. No annual forecast is divided by twelve here.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import re
from urllib.parse import urljoin, urlsplit
from urllib.request import Request, urlopen

import pandas as pd

from .macro_rosstat import read_xlsx_cells

SURVEY_PAGE = "https://cbr.ru/statistics/ddkp/mo_br/"
CONSUMPTION_ARCHIVE = "https://cbr.ru/about_br/publ/ddkp/"
CONSUMPTION_START = "https://cbr.ru/about_br/publ/ddkp/longread_4_44/"
INFLATION_START = "https://www.cbr.ru/analytics/dkp/inflationary_expectations/Infl_exp_23-12/"
KEY_RATE_ARCHIVE = "https://www.cbr.ru/dkp/mp_dec/decision_key_rate/"
MONTHS = {
    "января": 1, "февраля": 2, "марта": 3, "апреля": 4, "мая": 5,
    "июня": 6, "июля": 7, "августа": 8, "сентября": 9,
    "октября": 10, "ноября": 11, "декабря": 12,
}
SURVEY_MONTHS = {
    "январского": 1, "февральского": 2, "мартовского": 3,
    "апрельского": 4, "майского": 5, "июньского": 6,
    "июльского": 7, "августовского": 8, "сентябрьского": 9,
    "октябрьского": 10, "ноябрьского": 11, "декабрьского": 12,
}


def _official_url(url: str) -> None:
    p = urlsplit(url)
    if p.scheme != "https" or p.username or p.password or p.hostname not in {"cbr.ru", "www.cbr.ru"}:
        raise ValueError("Only official public CBR HTTPS URLs are allowed.")


def download_cbr(url: str, timeout: float = 30.0) -> tuple[bytes, dict]:
    """Download an actual discovered public URL, without alternate access paths."""
    _official_url(url)
    with urlopen(Request(url, headers={"User-Agent": "sberforecast E05a source audit"}), timeout=timeout) as response:
        _official_url(response.geturl())
        content = response.read(20 * 1024 * 1024 + 1)
        if len(content) > 20 * 1024 * 1024:
            raise ValueError("CBR source exceeds the explicit 20 MiB bound.")
        return content, {
            "response_url": response.geturl(),
            "content_type": response.headers.get("Content-Type"),
            "http_last_modified": response.headers.get("Last-Modified"),
        }


class _Document(HTMLParser):
    """Small HTML reader: keep paragraphs/tables, ignore scripts and charts."""
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.links: list[str] = []
        self.paragraphs: list[str] = []
        self.tables: list[list[list[str]]] = []
        self._skip = 0
        self._paragraph: list[str] | None = None
        self._tables: list[list[list[str]]] = []
        self._rows: list[list[str]] = []
        self._cells: list[list[str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style"}:
            self._skip += 1
        if self._skip:
            return
        if tag == "a" and dict(attrs).get("href"):
            self.links.append(dict(attrs)["href"])
        if tag == "p":
            self._paragraph = []
        if tag == "table":
            self._tables.append([])
        if tag == "tr" and self._tables:
            self._rows.append([])
        if tag in {"td", "th"} and self._rows:
            self._cells.append([])

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"}:
            self._skip = max(0, self._skip - 1)
            return
        if self._skip:
            return
        if tag == "p" and self._paragraph is not None:
            self.paragraphs.append(" ".join(" ".join(self._paragraph).split()))
            self._paragraph = None
        if tag in {"td", "th"} and self._cells:
            self._rows[-1].append(" ".join(" ".join(self._cells.pop()).split()))
        if tag == "tr" and self._rows:
            self._tables[-1].append(self._rows.pop())
        if tag == "table" and self._tables:
            self.tables.append(self._tables.pop())

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        self.parts.append(data)
        if self._paragraph is not None:
            self._paragraph.append(data)
        if self._cells:
            self._cells[-1].append(data)


def _document(html: str) -> _Document:
    doc = _Document()
    doc.feed(html)
    return doc


def publication_dates(html: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Read the actual dated publication header and explicit last-update label."""
    header = re.search(r'<div\b[^>]*class="[^"]*news-info-line_date[^"]*"[^>]*>(.*?)</div>', html, re.S)
    if not header:
        raise ValueError("The archived publication has no dated publication header.")
    text = " ".join(" ".join(_document(header.group(1)).parts).split())
    date = re.fullmatch(r"(\d{1,2})\s+(\w+)\s+(\d{4})\s+года", text)
    if not date or date.group(2) not in MONTHS:
        raise ValueError("Unrecognised Russian publication date.")
    published = pd.Timestamp(int(date.group(3)), MONTHS[date.group(2)], int(date.group(1)))
    updated = re.search(r"Последнее обновление страницы:\s*(\d{2}\.\d{2}\.\d{4})", html)
    return published, pd.to_datetime(updated.group(1), format="%d.%m.%Y") if updated else published


def discover_cbr_links(html: str, page_url: str, kind: str) -> list[dict]:
    """Follow only URLs actually present in a downloaded official index."""
    _official_url(page_url)
    out: dict[str, dict] = {}
    for href in _document(html).links:
        url = urljoin(page_url, href)
        try:
            _official_url(url)
        except ValueError:
            continue
        path = urlsplit(url).path
        if kind == "survey" and path.endswith("/full.xlsx"):
            out[url] = {"url": url, "file": "survey_full.xlsx", "kind": kind}
        if kind == "inflation_comment":
            match = re.search(r"/Infl_exp_(\d{2})-(\d{2})/", path)
            if match and "22-12" <= "-".join(match.groups()) <= "24-11":
                out[url] = {"url": url, "file": f"inflation_20{match[1]}_{match[2]}.html", "kind": kind}
        if kind == "consumption_longread":
            match = re.search(r"/longread_(\d)_(\d+)/", path)
            if match and int(match[2]) in {40, 41, 42, 43, 44}:
                out[url] = {"url": url, "file": f"consumption_longread_{match[1]}_{match[2]}.html", "kind": kind}
    return list(out.values())


def _forecast_row(indicator: str, year: int, value: float, url: str, producer: str,
                  basis: str, status: str, vintage: str, published=pd.NaT,
                  available=pd.NaT, issue=pd.NaT, **extra) -> dict:
    return {
        "region_id": "RU", "geographic_level": "national", "indicator": indicator,
        "reference_period": str(year), "target_period": str(year), "value": value,
        "unit": "percent_growth", "published_at": published, "available_at": available,
        "vintage_id": vintage, "availability_status": status, "source_url": url,
        "forecast_issue_date": issue, "aggregation_basis": basis,
        "forecast_low": math.nan, "forecast_central": value, "forecast_high": math.nan,
        "is_forecast": True, "forecast_producer": producer, **extra,
    }


def parse_survey_workbook(content: bytes, source_url: str, *, vintage_id: str,
                          first_issue: str = "2022-12", last_issue: str = "2024-11") -> pd.DataFrame:
    """Read verified sheet definitions and explicit Median/Min/Max blocks.

    E6 etc. are edition-month labels, not publication dates. B rows therefore
    have no invented daily issue, publication or availability date. The current
    workbook includes no separate historical real-wage forecast sheet.
    """
    _official_url(source_url)
    sheets = read_xlsx_cells(content)
    definitions = {
        "1": ("forecast_inflation_dec_dec_pct", "december_to_december", "дек. к дек."),
        "2": ("forecast_inflation_annual_average_pct", "annual_average_to_annual_average", "в среднем за год"),
        "7": ("forecast_nominal_wage_growth_annual_pct", "annual_average_to_annual_average", "Номинальная заработная плата"),
    }
    rows = []
    for name, (indicator, basis, expected) in definitions.items():
        sheet = sheets.get(name)
        if sheet is None or expected not in str(sheet.get(4, {}).get(2, "")):
            raise ValueError(f"Unverified CBR survey definition for sheet {name}.")
        editions = {}
        for col, raw in sheet.get(6, {}).items():
            if col >= 5 and isinstance(raw, (float, int)):
                month = (pd.Timestamp("1899-12-30") + pd.Timedelta(days=float(raw))).strftime("%Y-%m")
                if first_issue <= month <= last_issue:
                    editions[col] = month
        blocks: dict[str, dict[int, int]] = {}
        block = None
        for row_num, cells in sorted(sheet.items()):
            label = str(cells.get(3, "")).strip()
            if label:
                block = label
            target = cells.get(4)
            if block in {"Median", "Min", "Max"} and isinstance(target, (int, float)):
                year = (pd.Timestamp("1899-12-30") + pd.Timedelta(days=float(target))).year
                blocks.setdefault(block, {})[year] = row_num
        for year, row_num in blocks.get("Median", {}).items():
            for col, month in editions.items():
                value = sheet[row_num].get(col)
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    continue
                bounds = {}
                for label, column in [("Min", "forecast_low"), ("Max", "forecast_high")]:
                    raw = sheet.get(blocks.get(label, {}).get(year, -1), {}).get(col)
                    bounds[column] = float(raw) if isinstance(raw, (int, float)) and math.isfinite(raw) else math.nan
                rows.append(_forecast_row(
                    indicator, year, float(value), source_url, "survey_participants", basis,
                    "B", f"{vintage_id}:survey-{month}", survey_issue_period=month,
                    forecast_interval_kind="respondent_min_max", central_method="respondent_median",
                    original_forecast_issue_date=pd.NaT,
                    availability_note="Current aggregate workbook; original publication day and unrevised historical version unverified.",
                    **bounds,
                ))
    return pd.DataFrame(rows)


def parse_inflation_comment(html: str, source_url: str) -> pd.DataFrame:
    """Extract explicit survey medians from dated paragraphs, never vague values.

    'Near 4%' is not silently turned into exactly 4.0. Bounds, changes in
    percentage points and the Bank's own inflation forecasts are not taken as
    the participants' medians. A republished quotation is available from the
    commentary date, not an assumed original survey publication day.
    """
    _official_url(source_url)
    published, updated = publication_dates(html)
    paragraphs = [p for p in _document(html).paragraphs if "макроэкономического опроса" in p]
    rows = []
    for paragraph in paragraphs:
        month_match = re.search(r"данным\s+(\w+)\s+макроэкономического опроса", paragraph, re.I)
        month = SURVEY_MONTHS.get(month_match[1].lower()) if month_match else None
        issue_period = f"{published.year - (month > published.month):04d}-{month:02d}" if month else None
        values: dict[int, float] = {}
        for match in re.finditer(r"на конец\s+(20\d{2})\s+года([^%]{0,120})%", paragraph):
            section = match[2]
            if "вблизи" in section:
                continue
            value = re.search(r"(?<![\d,])([−-]?\d+(?:,\d+)?)\s*$", section)
            if value and not re.search(r"\d\s*[–-]\s*\(?[−-]?\d", section) and " п.п." not in section:
                values[int(match[1])] = float(value[1].replace("−", "-").replace(",", "."))
        for match in re.finditer(r"В\s+(20\d{2})(?:\s+и\s+(20\d{2}))?\s+год(?:у|ах)([^%]{0,100})%", paragraph):
            section = match[3]
            if "вблизи" in section:
                continue
            value = re.search(r"(?<![\d,])([−-]?\d+(?:,\d+)?)\s*$", section)
            if value and not re.search(r"\d\s*[–-]\s*\(?[−-]?\d", section):
                for year in [match[1], match[2]]:
                    if year:
                        values[int(year)] = float(value[1].replace("−", "-").replace(",", "."))
        for year, value in values.items():
            rows.append(_forecast_row(
                "forecast_inflation_dec_dec_pct", year, value, source_url,
                "survey_participants", "december_to_december",
                "A" if updated == published else "B",
                f"cbr-comment-{published:%Y-%m-%d}-updated-{updated:%Y-%m-%d}",
                published, max(published, updated), published,
                survey_issue_period=issue_period, original_forecast_issue_date=pd.NaT,
                forecast_interval_kind="not_reported", central_method="quoted_rounded_respondent_median",
                availability_note="Explicit dated archived commentary quotation; conservative commentary publication date, original survey day unknown.",
            ))
    return pd.DataFrame(rows)


def parse_forecast_interval(text: str) -> tuple[float, float]:
    """Parse official table ranges, including -2,0–(-1,0), without changing sign."""
    s = text.strip().replace("−", "-").replace("—", "–").replace(" ", "")
    single = re.fullmatch(r"\(?([+-]?\d+(?:,\d+)?)\)?", s)
    if single:
        v = float(single[1].replace(",", "."))
        return v, v
    pair = re.fullmatch(r"\(?([+-]?\d+(?:,\d+)?)\)?[–-]\(?([+-]?\d+(?:,\d+)?)\)?", s)
    if not pair:
        raise ValueError(f"Unrecognised forecast range: {text!r}")
    low, high = [float(v.replace(",", ".")) for v in pair.groups()]
    if low > high:
        raise ValueError("Forecast range has reversed bounds.")
    return low, high


def parse_consumption_longread(html: str, source_url: str) -> pd.DataFrame:
    """Read the household final-consumption annual real-volume forecast table."""
    _official_url(source_url)
    published, updated = publication_dates(html)
    doc = _document(html)
    rows = []
    for table in doc.tables:
        targets = None
        for cells in table:
            if len(cells) > 2 and all(re.search(r"20\d{2}", c) for c in cells[1:]):
                targets = [None if "факт" in c else int(re.search(r"20\d{2}", c)[0]) for c in cells[1:]]
            if not cells or "Расходы на конечное потребление домашних хозяйств" not in cells[0]:
                continue
            if targets is None or len(targets) != len(cells) - 1:
                raise ValueError("Consumption forecast has no verified annual target header.")
            for year, text in zip(targets, cells[1:]):
                if year is None:
                    continue
                low, high = parse_forecast_interval(text)
                midpoint = (low + high) / 2
                rows.append(_forecast_row(
                    "forecast_consumption_growth_annual_pct", year, midpoint,
                    source_url, "Bank of Russia", "annual_real_volume_growth",
                    "A" if updated == published else "B",
                    f"cbr-consumption-{published:%Y-%m-%d}-updated-{updated:%Y-%m-%d}",
                    published, max(published, updated), published,
                    forecast_low=low, forecast_high=high,
                    central_method="interval_midpoint_not_official_central" if low != high else "official_point",
                    forecast_interval_kind="official_forecast_range",
                    availability_note="National annual real household final-consumption volume; interval midpoint derived transparently, not Sber nominal card expenses.",
                ))
    if not rows:
        raise ValueError("No verified household final-consumption forecast table found.")
    return pd.DataFrame(rows)


def parse_consumption_pdf_text(text: str, source_url: str, published_at: str) -> pd.DataFrame:
    """Parse a reviewed -layout text cache of the archived 2024 forecast PDF.

    The collector separately verifies PDF/text SHA256 and the dated archive
    link. The prior-year factual column is excluded; the household row is not
    confused with total final consumption. No PDF program is called here.
    """
    _official_url(source_url)
    published = pd.Timestamp(published_at)
    issue = re.search(r"по итогам заседания Совета директоров по ключевой ставке\s+(\d{1,2})\s+(\w+)\s+(20\d{2})\s+года", text)
    if not issue or issue[2] not in MONTHS:
        raise ValueError("PDF text has no verified dated forecast heading.")
    issue_date = pd.Timestamp(int(issue[3]), MONTHS[issue[2]], int(issue[1]))
    if issue_date != published:
        raise ValueError("Forecast heading date differs from the dated archive publication.")
    first_table = text.split("Основные параметры прогноза Банка России в рамках базового сценария", 1)
    if len(first_table) != 2:
        raise ValueError("PDF text has no verified main forecast table.")
    body = first_table[1].split("Источник: Банк России.", 1)[0]
    header = body.split("Инфляция", 1)[0]
    years = [int(y) for y in re.findall(r"\b20\d{2}\b", header)]
    if not years or years[0] != published.year - 1 or years != sorted(set(years)):
        raise ValueError("PDF text has ambiguous annual table columns.")
    if "факт" not in header:
        raise ValueError("PDF factual prior-year column is not explicitly marked.")
    household = re.search(r"Расходы на конечное потребление[^\n]*\n\s*[–−-]\s*домашних хозяйств\s+([^\n]+)", body)
    if not household:
        raise ValueError("PDF text has no verified household subrow.")
    cells = household[1].split()
    if len(cells) != len(years):
        raise ValueError("PDF household values do not match the annual header.")
    rows = []
    for year, value in zip(years[1:], cells[1:]):
        low, high = parse_forecast_interval(value)
        rows.append(_forecast_row(
            "forecast_consumption_growth_annual_pct", year, (low + high) / 2,
            source_url, "Bank of Russia", "annual_real_volume_growth", "A",
            f"cbr-consumption-pdf-{published:%Y-%m-%d}", published, published, published,
            forecast_low=low, forecast_high=high,
            central_method="interval_midpoint_not_official_central" if low != high else "official_point",
            forecast_interval_kind="official_forecast_range",
            availability_note="Dated archived PDF and publication link verified; national annual real household final-consumption volume, derived midpoint.",
        ))
    return pd.DataFrame(rows)


def collect_cbr_pdf_sources(raw_dir: Path) -> tuple[pd.DataFrame, list[dict]]:
    """Use reviewed SHA-bound PDF/text caches, without network or pdftotext.

    A nonzero extraction return code is never hidden: it requires an explicit
    reviewed cache and remains in the catalog. File dates from PDF metadata
    are corroboration, not a substitute for official publication evidence.
    """
    raw_dir = Path(raw_dir)
    manifest = raw_dir / "consumption_pdf_extracts.json"
    if not manifest.exists():
        return pd.DataFrame(), []
    records = json.loads(manifest.read_text(encoding="utf-8"))
    frames, catalog = [], []
    for record in records:
        entry = {
            "source": "Bank of Russia", "url": record.get("source_url"),
            "name": "Среднесрочный прогноз Банка России: расходы на конечное потребление домашних хозяйств",
            "indicator": "forecast_consumption_growth_annual_pct", "unit": "percent_growth",
            "geographic_level": "national", "published_at": record.get("published_at"),
            "version_published_at": record.get("published_at"),
            "updated_at": record.get("pdf_modification_date"),
            "retrieved_at": record.get("retrieved_at"), "sha256": record.get("pdf_sha256"),
            "availability_status": "C", "verification_status": "not_checked",
            "extraction_method": record.get("extraction_method"),
            "extraction_returncode": record.get("extraction_returncode"),
            "extraction_warning": record.get("extraction_warning"),
            "text_sha256": record.get("text_sha256"),
            "publication_evidence_url": record.get("publication_evidence_url"),
            "publication_evidence_file": str(raw_dir / record.get("publication_evidence_file", "")),
            "publication_evidence_sha256": record.get("publication_evidence_sha256"),
            "pdf_creation_date": record.get("pdf_creation_date"),
            "definition_note": "Household final-consumption annual real-volume forecast; derived interval midpoint. Prior-year factual values are excluded.",
        }
        catalog.append(entry)
        try:
            _official_url(record["source_url"])
            _official_url(record["publication_evidence_url"])
            for field in ["pdf_file", "text_file", "publication_evidence_file"]:
                if Path(record[field]).name != record[field]:
                    raise ValueError("PDF cache references must be local filenames.")
            pdf = (raw_dir / record["pdf_file"]).read_bytes()
            text_bytes = (raw_dir / record["text_file"]).read_bytes()
            evidence = (raw_dir / record["publication_evidence_file"]).read_bytes()
            for content, field in [(pdf, "pdf_sha256"), (text_bytes, "text_sha256"), (evidence, "publication_evidence_sha256")]:
                if hashlib.sha256(content).hexdigest() != record[field]:
                    raise ValueError(f"SHA256 mismatch: {field}")
            if not pdf.startswith(b"%PDF-") or record.get("reviewed") is not True:
                raise ValueError("Only reviewed caches of actual PDF bytes are eligible.")
            archive_html = evidence.decode("utf-8-sig")
            links = _document(archive_html).links
            expected_path = urlsplit(record["source_url"]).path.lower()
            if not any(urlsplit(urljoin(record["publication_evidence_url"], href)).path.lower() == expected_path for href in links):
                raise ValueError("PDF URL is absent from the preserved official publication index.")
            date_label = pd.Timestamp(record["published_at"]).strftime("%d.%m.%Y")
            matched = False
            for anchor in re.finditer(r'<a\b[^>]*href="([^"]+)"[^>]*>.*?</a>', archive_html, re.S):
                if urlsplit(urljoin(record["publication_evidence_url"], anchor[1])).path.lower() != expected_path:
                    continue
                context = archive_html[max(0, anchor.start() - 700):anchor.end()]
                if date_label in " ".join(_document(context).parts):
                    matched = True
            if not matched:
                raise ValueError("No preserved dated archive entry for this PDF URL.")
            frame = parse_consumption_pdf_text(text_bytes.decode("utf-8-sig"), record["source_url"], record["published_at"])
            frame["source_pdf_sha256"] = record["pdf_sha256"]
            frame["source_text_sha256"] = record["text_sha256"]
            entry.update({"availability_status": "A", "verification_status": "parsed_reviewed_pdf_cache", "normalized_rows": len(frame),
                          "file": str(raw_dir / record["pdf_file"]), "text_file": str(raw_dir / record["text_file"]),
                          "bytes": len(pdf), "data_period": f"{frame.reference_period.min()}..{frame.reference_period.max()}"})
            frames.append(frame)
        except (OSError, KeyError, UnicodeError, ValueError, TypeError) as exc:
            entry["verification_status"] = "parse_error"
            entry["error"] = f"{type(exc).__name__}: {exc}"
    archive = raw_dir / "key_rate_archive.html"
    if archive.exists():
        content = archive.read_bytes()
        catalog.append({
            "source": "Bank of Russia", "url": KEY_RATE_ARCHIVE,
            "name": "Материалы по итогам заседаний Совета директоров Банка России по ключевой ставке",
            "file": str(archive), "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(),
            "unit": None, "geographic_level": "national", "data_period": "Dated archive links, including 2024",
            "published_at": None, "version_published_at": None, "updated_at": None,
            "retrieved_at": datetime.fromtimestamp(archive.stat().st_mtime, timezone.utc).isoformat(),
            "availability_status": "B", "verification_status": "verified_dated_pdf_links_current_index",
        })
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(), catalog


def collect_cbr_sources(raw_dir: Path, *, allow_network: bool = False) -> tuple[pd.DataFrame, list[dict]]:
    """Discover, optionally download, and normalize sources; tests need no network.

    Existing cache bytes are read unchanged. Errors remain visible in the
    catalog and do not promote a source to A. No model is fitted.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    catalog: list[dict] = []
    frames = []
    retrieved = datetime.now(timezone.utc).isoformat()

    def read(url: str, filename: str, kind: str) -> tuple[str | bytes | None, dict]:
        path = raw_dir / filename
        entry = {
            "source": "Bank of Russia", "url": url, "file": str(path),
            "name": kind, "geographic_level": "national", "retrieved_at": retrieved,
            "published_at": None, "version_published_at": None, "updated_at": None,
            "sha256": None, "availability_status": "C", "verification_status": "not_read",
            "unit": "percent_growth" if kind not in {"survey_index", "archive_index"} else None,
            "data_period": None, "kind": kind,
        }
        catalog.append(entry)
        try:
            if path.exists():
                content = path.read_bytes()
                entry["retrieval_mode"] = "existing_cache"
                entry["retrieved_at"] = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
            elif allow_network:
                content, headers = download_cbr(url)
                path.write_bytes(content)
                entry.update(headers)
                entry["retrieval_mode"] = "downloaded"
            else:
                raise FileNotFoundError("No cached CBR source; network disabled.")
            entry["sha256"] = hashlib.sha256(content).hexdigest()
            entry["bytes"] = len(content)
            entry["verification_status"] = "downloaded_not_yet_parsed"
            if filename.endswith(".xlsx"):
                entry["name"] = "Агрегированные результаты макроэкономического опроса Банка России (full.xlsx)"
                return content, entry
            decoded = content.decode("utf-8-sig")
            title = re.search(r"<title[^>]*>(.*?)</title>", decoded, re.S | re.I)
            if title:
                entry["name"] = " ".join(" ".join(_document(title[1]).parts).split())
            return decoded, entry
        except (OSError, ValueError, UnicodeError) as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
            entry["verification_status"] = "error"
            return None, entry

    survey, survey_entry = read(SURVEY_PAGE, "survey_current.html", "survey_index")
    archive, archive_entry = read(CONSUMPTION_ARCHIVE, "consumption_archive.html", "archive_index")
    inflation, inflation_entry = read(INFLATION_START, "inflation_2023_12.html", "inflation_comment")
    specs = []
    if isinstance(survey, str):
        specs += discover_cbr_links(survey, SURVEY_PAGE, "survey")
        update = re.search(r"Последнее обновление страницы:\s*(\d{2}\.\d{2}\.\d{4})", survey)
        survey_entry["updated_at"] = pd.to_datetime(update[1], format="%d.%m.%Y").isoformat() if update else None
        survey_entry["availability_status"] = "B"
        survey_entry["verification_status"] = "verified_current_index"
        survey_entry["definition_note"] = "Survey forecasts are participant medians; real-wage median uses respondent-level nominal wage and average CPI, not ratio of aggregate medians."
    if isinstance(archive, str):
        specs += discover_cbr_links(archive, CONSUMPTION_ARCHIVE, "consumption_longread")
        archive_entry["verification_status"] = "verified_archive_links"
        archive_entry["availability_status"] = "B"
    if isinstance(inflation, str):
        specs += discover_cbr_links(inflation, INFLATION_START, "inflation_comment")
    seen = set()
    for spec in specs:
        if spec["url"] in seen:
            continue
        seen.add(spec["url"])
        if spec["file"] == "inflation_2023_12.html":
            content, entry = inflation, inflation_entry
        else:
            content, entry = read(spec["url"], spec["file"], spec["kind"])
        if content is None:
            continue
        try:
            if spec["kind"] == "survey":
                frame = parse_survey_workbook(content, spec["url"], vintage_id=f"cbr-current-{entry['sha256'][:16]}")
                entry["availability_status"] = "B"
                entry["updated_at"] = survey_entry["updated_at"]
                entry["definition_note"] = "Edition months from E6 etc. are not publication dates; Median and respondent Min/Max blocks verified; no real-wage history sheet."
            else:
                published, updated = publication_dates(content)
                entry["published_at"] = published.isoformat()
                entry["updated_at"] = updated.isoformat()
                entry["version_published_at"] = max(published, updated).isoformat()
                entry["availability_status"] = "A" if published == updated else "B"
                frame = parse_inflation_comment(content, spec["url"]) if spec["kind"] == "inflation_comment" else parse_consumption_longread(content, spec["url"])
                entry["definition_note"] = "Dated archived publication; specific values from this edition, conservative date of this publication."
            entry["normalized_rows"] = len(frame)
            entry["verification_status"] = "parsed" if len(frame) else "read_no_explicit_survey_medians"
            if not frame.empty:
                entry["data_period"] = f"{frame.reference_period.min()}..{frame.reference_period.max()}"
                entry["indicator"] = ",".join(sorted(frame.indicator.unique()))
                frames.append(frame)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            entry["availability_status"] = "C"
            entry["verification_status"] = "parse_error"
            entry["error"] = f"{type(exc).__name__}: {exc}"
    pdf_rows, pdf_catalog = collect_cbr_pdf_sources(raw_dir)
    catalog.extend(pdf_catalog)
    if not pdf_rows.empty:
        frames.append(pdf_rows)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(), catalog
