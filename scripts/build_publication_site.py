"""Build the reviewed static publication package; no network, data loading or fits.

The existing docs/ and report Markdown are inputs, never written. Only the exact
allowlist is read/copied. Math SVGs use the already installed Matplotlib mathtext.
No visitor-side dependency, server or external JS/CSS is required.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
from html.parser import HTMLParser
import io
import json
from pathlib import Path, PurePosixPath
import posixpath
import re
import shutil
import tempfile
from urllib.parse import unquote, urlsplit
import zipfile

ROOT = Path(__file__).resolve().parents[1]
GITHUB = "https://github.com/MarselMiller/sberindex-municipal-forecasting"
INVITATION = "Исходный код — доступ по приглашению"
SCHEMA = "sberindex-publication-v1"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def safe_path(name: str) -> str:
    path = PurePosixPath(name)
    if (not name or "\\" in name or ":" in name or path.is_absolute()
            or any(p in {"..", "."} or p.startswith(".") for p in path.parts)
            or str(path) != name):
        raise ValueError(f"Unsafe package/source path: {name!r}")
    return name


def read_source(root: Path, name: str) -> bytes:
    safe_path(name)
    if (not (name.startswith(('docs/', 'reports/final/', 'reports/results/'))
             or name in {'README.md', 'configs/publication_site.json', 'configs/publication_external_links.json', 'scripts/build_publication_site.py'})
            or Path(name).suffix.lower() not in {'.html', '.md', '.json', '.css', '.js', '.png', '.csv', '.pdf', '.py'}):
        raise ValueError(f'Unapproved source class: {name}')
    path = root / name
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Source escapes project root")
    return path.read_bytes()


def audit_text(name: str, data: bytes):
    if Path(name).suffix.lower() not in {".html", ".md", ".json", ".csv", ".js", ".css", ".svg"}:
        return
    text = data.decode("utf-8-sig")
    patterns = {
        "absolute local path": r"(?i)(?:\b[a-z]:[\\/]|file://|/Users/|/home/[^ /]+/|\\\\[^\s]+\\)",
        "credential": r"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9]{30,}|AKIA[A-Z0-9]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----)",
        "credential assignment": r"(?i)[\"']?(?:api_key|access_token|password|secret_key)[\"']?\s*[:=]\s*[\"'][^\"']{8,}[\"']",
    }
    for reason, pattern in patterns.items():
        if re.search(pattern, text):
            raise ValueError(f"Publication denied ({reason}): {name}")
    if name.endswith(".csv"):
        rows = list(csv.reader(io.StringIO(text)))
        forbidden = {"municipality_id", "mo_id", "y_true", "y_pred", "prediction", "series_id", "full_name", "email", "phone"}
        if rows and forbidden.intersection(x.lower() for x in rows[0]):
            raise ValueError(f"Row-level/private CSV denied: {name}")


class Inventory(HTMLParser):
    def __init__(self, text: str):
        super().__init__(convert_charrefs=True)
        self.ids = set()
        self.duplicates = []
        self.links = []
        self.cells = []
        self._cell = None
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            if attrs["id"] in self.ids:
                self.duplicates.append(attrs["id"])
            self.ids.add(attrs["id"])
        for attr in ("href", "src"):
            if attr in attrs:
                self.links.append((tag, attr, attrs[attr], attrs))
        if tag in {"td", "th"}:
            self._cell = []

    def handle_data(self, value):
        if self._cell is not None:
            self._cell.append(value)

    def handle_endtag(self, tag):
        if tag in {"td", "th"} and self._cell is not None:
            self.cells.append("".join(self._cell).strip())
            self._cell = None


def slug(text: str) -> str:
    text = re.sub(r"\[([^]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"[`*_]", "", text).strip().lower()
    return re.sub(r"[^\w\- ]", "", text, flags=re.UNICODE).replace(" ", "-")


def table_cells(line: str) -> list[str]:
    return [x.strip().replace(r"\|", "|") for x in re.split(r"(?<!\\)\|", line.strip().strip("|"))]


class Builder:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.policy = json.loads(read_source(self.root, "configs/publication_site.json"))
        self.external_review = json.loads(read_source(self.root, "configs/publication_external_links.json"))
        if self.policy["schema_version"] != 1:
            raise ValueError("Unsupported publication policy")
        self.payload: dict[str, bytes] = {}
        self.provenance = {}
        self.link_changes = []
        self.mapping = dict(self.policy["aliases"])
        for source in self.policy["runtime"]:
            self.mapping[source] = source.removeprefix("docs/")
        self.mapping.update(self.policy["documents"])
        self.mapping.update(self.policy["copies"])
        for source, dest in self.mapping.items():
            safe_path(source)
            safe_path(dest)
        self.documents = {}
        self.math_count = 0

    def put(self, dest: str, data: bytes, source=None, kind="generated"):
        safe_path(dest)
        audit_text(dest, data)
        if dest in self.payload:
            raise ValueError(f"Duplicate package destination: {dest}")
        self.payload[dest] = data
        record = {"kind": kind}
        if source:
            raw = read_source(self.root, source)
            record.update(source=source, source_sha256=sha(raw))
        self.provenance[dest] = record

    def resolve(self, url: str, source: str, dest: str, image=False):
        url = html.unescape(url)
        parsed = urlsplit(url)
        restricted = False
        target = None
        if parsed.scheme or parsed.netloc:
            if url.rstrip("/") == GITHUB:
                replacement = GITHUB
                restricted = True
            elif url.startswith(GITHUB + "/blob/main/"):
                target = unquote(parsed.path.split("/blob/main/", 1)[1])
            elif parsed.scheme in {"https", "http", "mailto"}:
                return url, False
            else:
                raise ValueError(f"Unsupported URL: {url}")
        else:
            if not parsed.path:
                return url, False
            target = posixpath.normpath(posixpath.join(posixpath.dirname(source), unquote(parsed.path)))
        if target is not None:
            if target in self.mapping:
                mapped = self.mapping[target]
                fragment = parsed.fragment
            else:
                if image:
                    raise ValueError(f"Unreviewed figure: {target}")
                mapped = "references/restricted.html"
                fragment = self.policy["restricted"].get(target, "source-materials")
                restricted = True
            replacement = posixpath.relpath(mapped, posixpath.dirname(dest) or ".")
            if fragment:
                replacement += "#" + fragment
        self.link_changes.append({
            "source_document": source, "package_document": dest,
            "original_url": url, "package_url": replacement,
            "destination_type": "invitation_notice" if restricted and target else ("private_repository" if restricted else "local_file"),
            "anonymous_accessibility": "notice only; original requires invitation" if restricted and target else ("invitation required" if restricted else "yes"),
            "publication_status": "restricted; source excluded" if restricted and target else ("private code" if restricted else "reviewed public copy"),
        })
        return replacement, restricted

    def math(self, latex: str, dest: str, inline=False):
        # SVG IDs and metadata must be stable between builds. Source TeX is kept
        # verbatim in data-latex, alt text and the display-formula disclosure.
        import matplotlib
        from matplotlib import mathtext
        original = latex.strip()
        render = original.replace("\n", " ").replace("\\left", "").replace("\\right", "")
        render = re.sub(r"\\mathcal\s+([A-Za-z])", r"\\mathcal{\1}", render)
        name = "assets/math/" + sha(original.encode())[:20] + ".svg"
        if name not in self.payload:
            output = io.BytesIO()
            with matplotlib.rc_context({"svg.hashsalt": SCHEMA, "svg.fonttype": "path", "font.family": "DejaVu Sans"}):
                mathtext.math_to_image("$" + render + "$", output, format="svg", dpi=120, color="black")
            svg = re.sub(rb"\s*<dc:date>.*?</dc:date>", b"", output.getvalue())
            self.put(name, svg, kind="formula_svg")
        self.math_count += 1
        relative = posixpath.relpath(name, posixpath.dirname(dest) or ".")
        image = f'<img class="math-{"inline" if inline else "display"}" src="{relative}" alt="{html.escape(original, quote=True)}" data-latex="{html.escape(original, quote=True)}">'
        if inline:
            return image
        return '<div class="math-block" tabindex="0" role="region" aria-label="Формула">' + image + '</div><details class="math-source"><summary>Исходная формула LaTeX</summary><pre>' + html.escape(original) + '</pre></details>'

    def inline(self, text: str, source: str, dest: str):
        # Tokenize links/code/math before emphasis, so TeX and URL characters
        # cannot become formatting. Balanced parentheses are allowed in URLs.
        pattern = re.compile(r"(`+)(.*?)(\1)|(!?\[([^\]]*)\]\(([^()\s]*(?:\([^()]*\)[^()\s]*)*)\))|(\$([^$\n]+)\$)|(<https?://[^>]+>)", re.DOTALL)
        pieces = []
        position = 0
        for match in pattern.finditer(text):
            pieces.append(self.emphasis(text[position:match.start()]))
            if match.group(1):
                pieces.append("<code>" + html.escape(match.group(2)) + "</code>")
            elif match.group(4):
                label, url = match.group(5), match.group(6)
                is_image = match.group(4).startswith("!")
                mapped, restricted = self.resolve(url, source, dest, is_image)
                if is_image:
                    pieces.append(f'<img loading="lazy" src="{html.escape(mapped, quote=True)}" alt="{html.escape(label, quote=True)}">')
                else:
                    mark = ' data-access="restricted"' if restricted else ''
                    suffix = ' <span class="restricted-label">— доступ по приглашению</span>' if restricted else ''
                    if url in self.external_review['access_not_confirmed']:
                        suffix += ' <span class="restricted-label">— внешний файл: доступ не подтверждён, HTTP 403 при проверке</span>'
                    pieces.append(f'<a href="{html.escape(mapped, quote=True)}"{mark}>' + self.emphasis(label) + suffix + '</a>')
            elif match.group(7):
                pieces.append(self.math(match.group(8), dest, inline=True))
            else:
                url = match.group(9)[1:-1]
                mapped, restricted = self.resolve(url, source, dest)
                pieces.append(f'<a href="{html.escape(mapped, quote=True)}">{html.escape(url)}</a>')
            position = match.end()
        pieces.append(self.emphasis(text[position:]))
        return "".join(pieces)

    @staticmethod
    def emphasis(text):
        value = html.escape(text)
        value = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", value)
        value = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", value)
        return value

    def markdown(self, text: str, source: str, dest: str):
        lines = text.splitlines()
        output, headings, heading_ids = [], [], set()
        i = 0
        while i < len(lines):
            line = lines[i]
            if not line.strip():
                i += 1
                continue
            if line.strip().startswith("<!--"):
                while "-->" not in lines[i]:
                    i += 1
                i += 1
                continue
            anchor = re.fullmatch(r'\s*<a id="([a-z0-9-]+)"></a>\s*', line)
            if anchor:
                output.append(f'<a id="{anchor[1]}"></a>')
                heading_ids.add(anchor[1])
                i += 1
                continue
            if line.strip().startswith("```"):
                language = line.strip()[3:]
                content = []
                i += 1
                while i < len(lines) and not lines[i].strip().startswith("```"):
                    content.append(lines[i])
                    i += 1
                if i == len(lines):
                    raise ValueError("Unclosed code fence")
                output.append(f'<pre><code data-language="{html.escape(language)}">' + html.escape("\n".join(content)) + '</code></pre>')
                i += 1
                continue
            if line.strip() == "$$":
                content = []
                i += 1
                while i < len(lines) and lines[i].strip() != "$$":
                    content.append(lines[i])
                    i += 1
                if i == len(lines):
                    raise ValueError("Unclosed formula")
                output.append(self.math("\n".join(content), dest))
                i += 1
                continue
            heading = re.match(r"^(#{1,6})\s+(.+)$", line)
            if heading:
                level, label = len(heading[1]), heading[2]
                identifier = slug(label)
                base, n = identifier, 0
                while identifier in heading_ids:
                    n += 1
                    identifier = base + "-" + str(n)
                heading_ids.add(identifier)
                output.append(f'<h{level} id="{html.escape(identifier)}">' + self.inline(label, source, dest) + f'</h{level}>')
                if level == 2:
                    headings.append((identifier, label))
                i += 1
                continue
            if i + 1 < len(lines) and "|" in line and re.fullmatch(r"[ |:\-]+", lines[i+1]) and "---" in lines[i+1]:
                columns = table_cells(line)
                table = ['<div class="table-wrap" tabindex="0" role="region" aria-label="Таблица отчёта"><table><thead><tr>']
                table.extend('<th scope="col">' + self.inline(cell, source, dest) + '</th>' for cell in columns)
                table.append('</tr></thead><tbody>')
                i += 2
                while i < len(lines) and lines[i].lstrip().startswith("|"):
                    cells = table_cells(lines[i])
                    if len(cells) != len(columns):
                        raise ValueError(f"Malformed table: {source}:{i+1}")
                    table.append('<tr>' + ''.join('<td>' + self.inline(cell, source, dest) + '</td>' for cell in cells) + '</tr>')
                    i += 1
                table.append('</tbody></table></div>')
                output.append(''.join(table))
                continue
            if re.fullmatch(r"\s*[-*_]{3,}\s*", line):
                output.append('<hr>')
                i += 1
                continue
            if line.startswith(">"):
                content = []
                while i < len(lines) and lines[i].startswith(">"):
                    content.append(lines[i].lstrip("> "))
                    i += 1
                output.append('<blockquote><p>' + self.inline(' '.join(content), source, dest) + '</p></blockquote>')
                continue
            item = re.match(r"^(\s*)([-*]|\d+[.)])\s+(.+)$", line)
            if item:
                # All nesting levels are retained as separate list items; leading
                # indentation is kept in a data attribute for source inspection.
                ordered = item[2][0].isdigit()
                tag = "ol" if ordered else "ul"
                start = re.match(r"\d+", item[2])
                attr = f' start="{start[0]}"' if start and start[0] != "1" else ""
                listing = [f'<{tag}{attr}>']
                while i < len(lines):
                    item = re.match(r"^(\s*)([-*]|\d+[.)])\s+(.+)$", lines[i])
                    if not item or item[2][0].isdigit() != ordered:
                        break
                    listing.append(f'<li data-indent="{len(item[1])}">' + self.inline(item[3], source, dest) + '</li>')
                    i += 1
                listing.append(f'</{tag}>')
                output.append(''.join(listing))
                continue
            paragraph = [line]
            i += 1
            while i < len(lines) and lines[i].strip() and not re.match(r"^(?:#|```|\$\$|<a id=|>|\s*[-*] |\s*\d+[.)] |\|)", lines[i]):
                paragraph.append(lines[i])
                i += 1
            output.append('<p>' + self.inline(' '.join(paragraph), source, dest) + '</p>')
        return '\n'.join(output), headings

    @staticmethod
    def wrapper(title: str, body: str, headings=(), markdown_name=None):
        toc = ''
        if headings:
            toc = '<nav class="toc" aria-label="Содержание документа"><ul>' + ''.join(f'<li><a href="#{html.escape(identifier)}">{html.escape(label)}</a></li>' for identifier, label in headings) + '</ul></nav>'
        download = f'<p><a href="{markdown_name}">Исходный Markdown этого документа</a> · содержание сохранено; ссылки HTML адаптированы для публикационного пакета.</p>' if markdown_name else ''
        return ('<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
                '<meta name="color-scheme" content="light dark"><title>' + html.escape(title) + '</title><link rel="stylesheet" href="../assets/publication-reference.css"></head><body>'
                '<header><nav aria-label="Навигация приложений"><a href="../index.html">Интерактивный отчёт</a><a href="methodology.html">Методология</a><a href="../presentation/presentation.pdf">Презентация PDF</a><a href="glossary.html">Глоссарий</a><a href="restricted.html">Условия доступа</a></nav>' + download + '</header><main>' + toc + body + '</main><footer>Статическое приложение итогового отчёта. Новые расчёты при сборке не выполняются.</footer></body></html>')

    def main_html(self, text: str):
        def anchor(match):
            attributes, value, contents = match.groups()
            mapped, restricted = self.resolve(value, "docs/index.html", "index.html")
            attributes = attributes.replace(f'href="{value}"', f'href="{html.escape(mapped, quote=True)}"')
            if restricted:
                attributes += ' data-access="restricted"'
                if mapped == GITHUB:
                    contents = INVITATION
                else:
                    contents += ' <span class="restricted-label">— доступ по приглашению</span>'
            return '<a' + attributes + '>' + contents + '</a>'
        return re.sub(r'<a(\b[^>]*href="([^"]+)"[^>]*)>(.*?)</a>', anchor, text, flags=re.DOTALL)

    def build(self):
        self.check_figure_integrity()
        for source in self.policy["runtime"]:
            dest = self.mapping[source]
            raw = read_source(self.root, source)
            data = self.main_html(raw.decode("utf-8-sig")).encode() if dest == "index.html" else raw
            self.put(dest, data, source, "adapted_html" if dest == "index.html" else "verbatim_copy")
        for source, dest in self.policy["copies"].items():
            self.put(dest, read_source(self.root, source), source, "verbatim_copy")
        for source, dest in self.policy["documents"].items():
            raw = read_source(self.root, source)
            audit_text(source, raw)
            text = raw.decode("utf-8-sig")
            body, headings = self.markdown(text, source, dest)
            title = next(line.lstrip('# ') for line in text.splitlines() if line.startswith('# '))
            markdown_name = PurePosixPath(dest).stem + '.md'
            self.put('references/' + markdown_name, raw, source, "verbatim_markdown")
            self.put(dest, self.wrapper(title, body, headings, markdown_name).encode(), source, "rendered_markdown")
            self.documents[source] = {"destination": dest, "headings": len(re.findall(r'^#{1,6} ', text, re.MULTILINE)), "tables": body.count('<table>'), "figures": len(re.findall(r'!\[', text)), "source_sha256": sha(raw)}
        restricted = '''<h1>Доступ к вспомогательным материалам</h1>
<p class="access-note">HTML-отчёт, презентация PDF, методология, глоссарий, итоговая сводка, ограничения и включённые агрегированные приложения доступны в этом пакете без GitHub. Основной репозиторий остаётся приватным.</p>
<h2 id="offline-detection">Подробный отчёт offline detection</h2><p>Полный отчёт содержит локальные пути и подробные муниципальные таблицы и не включён в публикационный пакет. Сводные результаты и разрешённые иллюстрации доступны в методологии. Допустимость публикации подробного приложения требует отдельного решения.</p>
<h2 id="audits">Служебные аудиты</h2><p>Полные журналы воспроизводимости и проверки публикации доступны автору в приватном репозитории. Ссылки на них в пакете обозначены как требующие приглашения.</p>
<h2 id="audit-metadata">Полные metadata и Source of Truth</h2><p>Полные журналы запусков, независимой проверки, manifests и исходный results_summary.json сохранены в проекте без изменений. Они не включены в этот публичный пакет. Агрегированные результаты и JSON интерактивного отчёта доступны локально.</p>
<h2 id="source-materials">Исходный код и дополнительные research artifacts</h2><p>Код, конфигурации, первичные материалы и приложения вне явного allowlist доступны только при предоставленном доступе. Этот пакет не предоставляет доступ к исходным или построчным муниципальным данным.</p>
<h2 id="publication-policy">Политика публикации и рисунок МО 21</h2><p>Право на перераспространение исходных муниципальных parquet остаётся UNCLEAR. Исходные данные, построчные прогнозы и таблицы отдельных муниципалитетов исключены. Сохранены только уже проверенные иллюстрации итогового отчёта, включая существующий пример МО 21; его байты и SHA256 не изменены. Это исключение для иллюстрации не подтверждает право на распространение исходных рядов.</p>'''
        restricted += f'<p><a href="{GITHUB}" data-access="restricted">{INVITATION}</a></p>'
        self.put('references/restricted.html', self.wrapper('Условия доступа', restricted).encode())
        links = self.check_links()
        external_urls = {item['url'] for item in links if item['kind'] == 'external'}
        reviewed_urls = set(self.external_review['retrieved_documents']) | set(self.external_review['spreadsheet_endpoints']) | set(self.external_review['access_not_confirmed']) | set(self.external_review['invitation_required'])
        if external_urls != reviewed_urls:
            raise ValueError('External URL inventory changed: update the explicit read-only link review')
        audit = {"schema": SCHEMA, "policy": self.policy['review'], "original_html_link_changes": [item for item in self.link_changes if item['source_document'] == 'docs/index.html'], "all_document_link_changes": self.link_changes, "checks": links, "document_conversion": self.documents, "external_links": sorted(external_urls), "external_link_review": self.external_review, "external_access_note": "Runtime is entirely local. Private GitHub requires invitation. External primary-source references need a network and are not bundled."}
        self.put('link-audit.json', json_bytes(audit))
        records = [{"path": name, "size": len(data), "sha256": sha(data), **self.provenance[name]} for name, data in sorted(self.payload.items())]
        manifest = {"schema": SCHEMA, "base_commit": self.policy['base_commit'], "allowlist_sha256": sha(read_source(self.root, 'configs/publication_site.json')), "builder_sha256": sha(read_source(self.root, 'scripts/build_publication_site.py')), "external_link_review_sha256": sha(read_source(self.root, 'configs/publication_external_links.json')), "files": records, "file_count_excluding_manifest": len(records), "total_bytes_excluding_manifest": sum(x['size'] for x in records), "manifest_hash_note": "Manifest excludes itself; its SHA256 and ZIP SHA256 are printed by the builder.", "no_model_fits": True, "no_metric_changes": True, "no_raw_data": True, "formula_occurrences": self.math_count}
        self.put('manifest.json', json_bytes(manifest))
        return self.payload

    def check_figure_integrity(self):
        records = list(csv.DictReader(io.StringIO(read_source(self.root, 'reports/final/figure_manifest.csv').decode('utf-8-sig'))))
        registered = {r['figure']: r['sha256'] for r in records}
        for source in self.policy['copies']:
            if source.startswith('reports/final/figures/'):
                key = source.removeprefix('reports/final/')
                if sha(read_source(self.root, source)) != registered.get(key):
                    raise ValueError(f'Figure differs from reviewed report manifest: {source}')
        mo21 = read_source(self.root, 'docs/assets/figures/rolling_forecast.png')
        if sha(mo21) != '0ae922414791a15ad63f1aefc4f7458faa4bdee1e2da53baa2f04da0c143d6c2':
            raise ValueError('MO 21 publication exception requires a new review')
        if mo21 != read_source(self.root, 'reports/final/figures/rolling_forecast.png'):
            raise ValueError('Main and methodology figures differ')

    def check_links(self):
        inventories = {name: Inventory(data.decode('utf-8-sig')) for name, data in self.payload.items() if name.endswith('.html')}
        checks = []
        for name, inventory in inventories.items():
            if inventory.duplicates:
                raise ValueError(f'Duplicate anchors: {name}: {inventory.duplicates}')
            for tag, attr, url, attrs in inventory.links:
                parsed = urlsplit(url)
                if parsed.scheme or parsed.netloc:
                    if parsed.scheme not in {'https', 'http', 'mailto'}:
                        raise ValueError(f'Nonportable URL in {name}: {url}')
                    if tag != 'a':
                        raise ValueError(f'External runtime dependency: {name}: {url}')
                    if parsed.hostname == 'github.com' and (url != GITHUB or attrs.get('data-access') != 'restricted'):
                        raise ValueError('Private GitHub link disguised as a public document')
                    checks.append({'document': name, 'url': url, 'kind': 'external', 'status': 'invitation_required' if parsed.hostname == 'github.com' else 'primary_source_reference'})
                    continue
                if parsed.path.startswith('/') or '\\' in parsed.path:
                    raise ValueError(f'Nonportable path: {name}: {url}')
                target = posixpath.normpath(posixpath.join(posixpath.dirname(name), unquote(parsed.path))) if parsed.path else name
                safe_path(target)
                if target not in self.payload:
                    raise ValueError(f'Broken package link: {name} -> {url} ({target})')
                if parsed.fragment and target in inventories and unquote(parsed.fragment) not in inventories[target].ids:
                    raise ValueError(f'Broken anchor: {name} -> {url}')
                checks.append({'document': name, 'url': url, 'kind': 'local', 'status': 'PASS', 'target': target})
        # CSS URLs also have to be local and resolve to packaged assets.
        for name, data in self.payload.items():
            if name.endswith('.css'):
                for value in re.findall(r'url\([\"\']?([^\)\"\']+)', data.decode('utf-8-sig')):
                    if not value.startswith('data:'):
                        target = posixpath.normpath(posixpath.join(posixpath.dirname(name), value))
                        if target not in self.payload:
                            raise ValueError(f'Broken/external CSS asset: {name}: {value}')
        return checks


def zip_bytes(payload: dict[str, bytes]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(payload.items()):
            info = zipfile.ZipInfo(safe_path(name), date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data, compresslevel=9)
    return output.getvalue()


def build(root=ROOT, check=False):
    root = Path(root).resolve()
    builder = Builder(root)
    payload = builder.build()
    archive = zip_bytes(payload)
    output = root / 'dist/submission-site'
    archive_path = root / 'dist/submission-site.zip'
    if check:
        actual = {p.relative_to(output).as_posix(): p.read_bytes() for p in output.rglob('*') if p.is_file()}
        if actual != payload or archive_path.read_bytes() != archive:
            raise ValueError('Built package/ZIP differs from reproducible inputs')
    else:
        parent = root / 'dist'
        if parent.is_symlink() or not parent.resolve().is_relative_to(root):
            raise ValueError('Output escapes workspace')
        parent.mkdir(exist_ok=True)
        if output.exists():
            if output.is_symlink() or not output.resolve().is_relative_to(parent.resolve()):
                raise ValueError('Refusing to remove output outside dist')
            manifest_path = output / 'manifest.json'
            if not manifest_path.is_file() or json.loads(manifest_path.read_text(encoding='utf-8')).get('schema') != SCHEMA:
                raise ValueError('Refusing to replace an unmanaged directory')
        with tempfile.TemporaryDirectory(prefix='publication-stage-', dir=parent) as temporary:
            stage = Path(temporary) / 'site'
            stage.mkdir()
            for name, data in payload.items():
                path = stage / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            # Removal is limited to the verified, builder-owned absolute target.
            if output.exists():
                shutil.rmtree(output)
            stage.rename(output)
        archive_path.write_bytes(archive)
    return {'status': 'PASS', 'mode': 'check' if check else 'build', 'folder': 'dist/submission-site/', 'zip': 'dist/submission-site.zip', 'files': len(payload), 'bytes': sum(map(len, payload.values())), 'zip_bytes': len(archive), 'zip_sha256': sha(archive), 'manifest_sha256': sha(payload['manifest.json']), 'local_link_checks': sum(c['kind'] == 'local' for c in builder.check_links())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Verify package and ZIP without writing')
    args = parser.parse_args()
    print(json.dumps(build(check=args.check), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
