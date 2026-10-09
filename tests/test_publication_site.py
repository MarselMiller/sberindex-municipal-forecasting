"""Publication-only tests: saved artifacts are inputs, no experiment is run."""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import sys
import zipfile
from urllib.parse import unquote, urljoin, urlsplit

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('publication_builder', ROOT / 'scripts/build_publication_site.py')
publication = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publication)


@pytest.fixture(scope='module')
def package():
    policy = json.loads((ROOT / 'configs/publication_site.json').read_text(encoding='utf-8'))
    inputs = set(policy['runtime']) | set(policy['copies']) | set(policy['documents'])
    inputs |= {'reports/final/results_summary.json', 'configs/forecast_robustness.yaml'}
    before = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in inputs}
    modules_before = set(sys.modules)
    builder = publication.Builder(ROOT)
    payload = builder.build()
    after = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in inputs}
    assert before == after, 'Publication build changed research sources'
    builder.imported_modules = set(sys.modules) - modules_before
    return builder, payload


def test_repeatable_bytes_and_zip(package):
    _, payload = package
    repeated = publication.Builder(ROOT).build()
    assert repeated == payload
    assert publication.zip_bytes(repeated) == publication.zip_bytes(payload)


def test_manifest_complete_hashes_and_sizes(package):
    _, payload = package
    manifest = json.loads(payload['manifest.json'])
    records = {item['path']: item for item in manifest['files']}
    assert set(records) == set(payload) - {'manifest.json'}
    assert manifest['file_count_excluding_manifest'] == len(records)
    assert manifest['total_bytes_excluding_manifest'] == sum(len(payload[name]) for name in records)
    for name, record in records.items():
        assert record['sha256'] == hashlib.sha256(payload[name]).hexdigest()
        assert record['size'] == len(payload[name])


def test_zip_is_root_contents_without_parent_or_unlisted_files(package):
    _, payload = package
    with zipfile.ZipFile(io.BytesIO(publication.zip_bytes(payload))) as archive:
        assert set(archive.namelist()) == set(payload)
        assert 'index.html' in archive.namelist()
        for name in archive.namelist():
            publication.safe_path(name)
            assert archive.read(name) == payload[name]


def test_allowlist_is_exclusive_and_raw_files_absent(package):
    builder, payload = package
    for name in payload:
        assert not re.search(r'(^|/)(?:outputs|data/input|raw|private|\.git|\.venv)(?:/|$)', name)
        assert Path(name).suffix not in {'.parquet', '.gz', '.pkl', '.env', '.key', '.pem', '.ipynb', '.zip'}
        assert Path(name).name not in {'results_summary.json', 'run_manifest.json', 'independent_validation.json', 'figure_manifest.json'}
        publication.audit_text(name, payload[name])
    excluded = set(builder.policy['not_bundled'])
    published_sources = {x.get('source') for x in builder.provenance.values()}
    assert not excluded.intersection(published_sources)
    assert builder.policy['review']['raw_parquet_redistribution'].startswith('CC BY-SA 4.0 confirmed for the reviewed consumption dataset')
    assert 'no raw Parquet or row-level CSV/forecasts added to site' in builder.policy['review']['raw_parquet_redistribution']


@pytest.mark.parametrize('name,data', [
    ('private.csv', b'municipality_id,y_true,y_pred\nsynthetic,1,2\n'),
    ('metadata.json', b'{"path":"C:\\synthetic\\file"}'),
    ('example.txt.json', b'{"api_key":"synthetic_secret_value"}'),
])
def test_unsafe_content_is_rejected(name, data):
    with pytest.raises(ValueError):
        publication.audit_text(name, data)


@pytest.mark.parametrize('name', ['../escape.csv', '/absolute.html', 'C:/local.txt', '.env', 'assets/../../escape', 'assets\\bad.js'])
def test_path_escape_and_hidden_files_are_rejected(name):
    with pytest.raises(ValueError):
        publication.safe_path(name)


def test_private_source_classes_cannot_be_read():
    with pytest.raises(ValueError):
        publication.read_source(ROOT, 'outputs/baseline_local/predictions.csv')


def test_runtime_and_saved_metrics_are_verbatim(package):
    builder, payload = package
    for source in builder.policy['runtime']:
        if source != 'docs/index.html':
            assert payload[builder.mapping[source]] == (ROOT / source).read_bytes()
    for source, dest in builder.policy['copies'].items():
        assert payload[dest] == (ROOT / source).read_bytes()
    canonical = publication.Inventory((ROOT / 'docs/index.html').read_text(encoding='utf-8-sig'))
    bundled = publication.Inventory(payload['index.html'].decode())
    assert bundled.cells == canonical.cells
    assert len(bundled.cells) == 270


def test_local_pdf_methodology_glossary_and_no_root_escapes(package):
    builder, payload = package
    inventory = publication.Inventory(payload['index.html'].decode())
    links = {item[2] for item in inventory.links}
    assert {'presentation/presentation.pdf', 'references/methodology.html', 'references/glossary.html'} <= links
    checks = builder.check_links()
    assert all(item['status'] == 'PASS' for item in checks if item['kind'] == 'local')
    assert payload['presentation/presentation.pdf'].startswith(b'%PDF-')


def test_repository_links_use_tracked_files_without_invitation_claims(package):
    builder, payload = package
    for name, data in payload.items():
        if name.endswith('.html'):
            text = data.decode()
            for tag, attr, url, attrs in publication.Inventory(text).links:
                if 'github.com' in url:
                    assert url == publication.GITHUB or url.startswith(publication.GITHUB + '/blob/main/') or url.startswith(publication.GITHUB + '/tree/main/')
                    assert attrs.get('data-access') != 'restricted'
                    if '/main/' in url:
                        target = unquote(urlsplit(url).path.split('/main/', 1)[1])
                        assert target in builder.tracked or any(p.startswith(target.rstrip('/') + '/') for p in builder.tracked)
            assert 'доступ по приглашению' not in text
            assert 'остаётся приватным' not in text
    audit = json.loads(payload['link-audit.json'])
    main = audit['original_html_link_changes']
    assert len({r['original_url'] for r in main if r['original_url'].startswith(publication.GITHUB)}) == 12
    assert len({r['original_url'] for r in main if r['original_url'].startswith('../reports/')}) == 6


def test_pages_project_prefix_and_excluded_local_artifacts(package):
    builder, payload = package
    for item in builder.check_links():
        if item['kind'] == 'local':
            deployed = urlsplit(urljoin(publication.PAGES + item['document'], item['url']))
            prefix = urlsplit(publication.PAGES).path
            assert deployed.path.startswith(prefix)
            assert unquote(deployed.path[len(prefix):]) in payload
    url, excluded = builder.resolve('../outputs/local-only.json', 'docs/index.html', 'index.html')
    assert excluded and url == 'references/materials.html#local-artifacts'
    url, excluded = builder.resolve(publication.PAGES, 'README.md', 'references/repository-readme.html')
    assert not excluded and url == '../index.html'


def test_publication_rejects_local_only_required_input(monkeypatch):
    original = publication.Builder.__init__
    def without_one_source(self, root):
        original(self, root)
        self.tracked.discard('reports/results/e08b/feature_coverage.csv')
    monkeypatch.setattr(publication.Builder, '__init__', without_one_source)
    with pytest.raises(ValueError, match='not tracked'):
        publication.build(ROOT, check=True, require_tracked_inputs=True)


def test_methodology_content_formulas_tables_figures_preserved(package):
    builder, payload = package
    source = (ROOT / 'reports/final/METHODOLOGY_REPORT.md').read_bytes()
    assert payload['references/methodology.md'] == source
    compiled = payload['references/methodology.html'].decode()
    record = builder.documents['reports/final/METHODOLOGY_REPORT.md']
    assert record['tables'] == 12
    assert record['figures'] == 9
    assert len(re.findall(r'<div class="math-block"', compiled)) == 6
    formulas = re.findall(r'^\$\$\s*\n(.*?)\n\$\$', source.decode('utf-8-sig').replace('\r\n', '\n'), re.MULTILINE | re.DOTALL)
    import html
    for formula in formulas:
        assert html.escape(formula.strip(), quote=True) in compiled
    assert record['headings'] == len(re.findall(r'<h[1-6] id=', compiled))
    # Every table cell must survive conversion, including null/NA and units.
    lines = source.decode('utf-8-sig').splitlines()
    original_cells = []
    for line in lines:
        if line.startswith('|') and not re.fullmatch(r'[ |:\-]+', line):
            original_cells.extend(publication.table_cells(line))
    expected = []
    for cell in original_cells:
        rendered = builder.inline(cell, 'reports/final/METHODOLOGY_REPORT.md', 'references/methodology.html')
        expected.append(re.sub('<[^>]+>', '', html.unescape(rendered)).strip())
    assert publication.Inventory(compiled).cells == expected


def test_glossary_canonical_terms_and_anchors(package):
    _, payload = package
    source = (ROOT / 'reports/final/terminology.md').read_bytes()
    assert payload['references/glossary.md'] == source
    anchors = re.findall(rb'<a id="([^"]+)"', source)
    assert len(anchors) == 75
    inventory = publication.Inventory(payload['references/glossary.html'].decode())
    assert {a.decode() for a in anchors} <= inventory.ids


def test_public_e08_html_omits_workflow_notes_but_preserves_historical_sources(package):
    builder, payload = package
    for source, dest in builder.policy['documents'].items():
        if not source.startswith('reports/results/E08'):
            continue
        raw = (ROOT / source).read_bytes()
        assert payload['references/' + Path(dest).stem + '.md'] == raw
        public_text = publication.public_document_text(source, raw.decode('utf-8-sig'))
        assert re.findall(r'^\|.*$', public_text, re.MULTILINE) == re.findall(r'^\|.*$', raw.decode('utf-8-sig'), re.MULTILINE)
        for term in ['NO COMMIT/PUSH', 'Для review созданы', 'следующий этап']:
            assert term not in payload[dest].decode()


def test_no_ml_modules_loaded_by_publication_build(package):
    builder, _ = package
    assert not {'prophet', 'lightgbm', 'catboost', 'sberforecast'} & builder.imported_modules


def test_pages_workflow_is_manual_with_default_deployment_disabled():
    import yaml
    workflow = yaml.load((ROOT / '.github/workflows/pages.yml').read_text(encoding='utf-8'), Loader=yaml.BaseLoader)
    assert set(workflow['on']) == {'workflow_dispatch'}
    assert workflow['on']['workflow_dispatch']['inputs']['deploy']['default'] == 'false'
    assert workflow['permissions'] == {'contents': 'read'}
    assert 'inputs.deploy' in workflow['jobs']['deploy']['if'] and 'refs/heads/main' in workflow['jobs']['deploy']['if']
    assert workflow['jobs']['deploy']['needs'] == 'build'
    assert workflow['jobs']['deploy']['environment']['name'] == 'github-pages'
    assert workflow['jobs']['deploy']['permissions'] == {'pages': 'write', 'id-token': 'write'}
    steps = workflow['jobs']['build']['steps']
    strict_step = next(i for i, step in enumerate(steps) if '--strict' in step.get('run', ''))
    upload_step = next(i for i, step in enumerate(steps) if step.get('uses', '').startswith('actions/upload-pages-artifact@'))
    assert strict_step < upload_step
    for job in workflow['jobs'].values():
        for step in job['steps']:
            if 'uses' in step:
                assert re.search(r'@[0-9a-f]{40}$', step['uses'])
