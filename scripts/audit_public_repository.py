"""Read-only audit of tracked files and every locally reachable Git blob.

No matched secret values or local path values are printed or saved. Findings
contain file/object identifiers, classifications and counts only. Historical
objects are never changed. This heuristic audit is not a data licence grant.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
SECRETS = {
    'github_token': re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,})'),
    'openai_token': re.compile(rb'sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{40,}'),
    'aws_access_key': re.compile(rb'AKIA[A-Z0-9]{16}'),
    'private_key': re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    'credential_url': re.compile(rb'https?://[^\s/:]+:[^\s/@]{5,}@[^\s/]+'),
}
LOCAL_PATH = re.compile(rb'(?i)(?:\b[a-z]:[\\/]+(?:Users|Documents and Settings)[\\/]+[a-z0-9_.-]+[\\/]+|/(?:Users|home)/[a-z0-9_.-]+/)')
SECRET_NAME = re.compile(r'(^|/)(?:\.env(?:\..+)?|id_rsa|id_ed25519|credentials|[^/]+\.(?:pem|key))$')
RAW_NAME = re.compile(r'(?i)(?:\.(?:parquet|pkl|pickle|pt|pth|safetensors|onnx|zip|rar|7z)$|(?:^|/)(?:data/(?:raw|private|restricted|input)/|\.venv/|node_modules/))')
ROW_HEADERS = {'municipality_id', 'mo_id', 'y_true', 'y_pred', 'prediction', 'target_value'}


def git(*args, root=ROOT):
    return subprocess.check_output(['git', *args], cwd=root)


def reviewed_source_examples(root: Path) -> dict[str, str]:
    """Only the pinned, primary-source-reviewed example; author review remains.

    This exception never licenses other row data or skips secret/path checks.
    The F7 historical decision stays intact; the dated follow-up is separate.
    """
    proof_path = root / 'data/metadata/publication_rights_review.json'
    notice = root / 'reports/final/figure_data/README.md'
    if not proof_path.is_file() or not notice.is_file():
        return {}
    proof = json.loads(proof_path.read_text(encoding='utf-8'))
    primary = proof.get('primary_source', {})
    example = proof.get('example', {})
    expected = '4e5152a0200307fcf376512509f660d57c6555ef56575847333b25ecc22ad0bf'
    if (proof.get('verdict') == 'PUBLICATION PERMITTED'
            and proof.get('licence') == 'CC BY-SA 4.0'
            and primary.get('grant_verified_in_official_template') is True
            and primary.get('http_status') == 200
            and primary.get('api_url') == 'https://sberindex.ru/api/researches/v1/data-sense-opisanie-nabora-dannikh-khakatona-sberindeksa-po-munitsipalnim-dannim'
            and example.get('path') == 'reports/final/figure_data/rolling_forecast.csv'
            and example.get('sha256') == expected
            and example.get('rows') == 12
            and example.get('saved_predictions_match') is True
            and example.get('original_target_observations_match') is True
            and 'CC BY-SA 4.0' in notice.read_text(encoding='utf-8')):
        # The original Windows CSV has CRLF; Git stores the same content as LF.
        # Both representations are reviewed, without rewriting the source file.
        return {example['path']: 'af1e10aa3327738ded377f367f7888f8daaac83cb5689784720bf3dc75a0798f'}
    return {}


def inspect(name: str, data: bytes, reviewed_examples=None) -> list[dict]:
    findings = []
    def record(category, severity, count=1):
        findings.append({'category': category, 'severity': severity, 'count': count})
    if SECRET_NAME.search(name):
        record('secret_file_name', 'BLOCKER')
    if RAW_NAME.search(name) and name != 'data/input/.gitkeep':
        record('raw_or_unreviewed_binary_file', 'BLOCKER')
    for category, pattern in SECRETS.items():
        matches = pattern.findall(data)
        if matches:
            if category == 'credential_url' and name == 'tests/test_macro_cbr.py' and all(
                urlsplit(value.decode()).hostname == 'cbr.ru'
                and urlsplit(value.decode()).username == 'user'
                and urlsplit(value.decode()).password in {'password', 'secret', 'pass'}
                for value in matches
            ):
                record('synthetic_rejected_credential_url', 'INFO', len(matches))
            else:
                record(category, 'BLOCKER', len(matches))
    count = len(LOCAL_PATH.findall(data))
    if count:
        record('absolute_user_path', 'REVIEW', count)
    if name.endswith(('.csv', '.csv.gz')):
        plain = gzip.decompress(data) if name.endswith('.gz') else data
        try:
            rows = list(csv.DictReader(io.StringIO(plain.decode('utf-8-sig'))))
        except (UnicodeError, csv.Error):
            record('unreadable_csv', 'REVIEW')
            return findings
        header = set(rows[0]) if rows else set()
        if ROW_HEADERS.intersection(header):
            # Explicit checked synthetic demo; never classify it as real.
            synthetic = name.startswith('data/synthetic/e07c_demo/')
            licensed = (reviewed_examples or {}).get(name) == hashlib.sha256(data.replace(b'\r\n', b'\n')).hexdigest()
            category = 'synthetic_row_data' if synthetic else 'licensed_source_rows_require_author_review' if licensed else 'municipality_or_prediction_rows'
            record(category, 'INFO' if synthetic else 'REVIEW' if licensed else 'BLOCKER', len(rows))
    if name.endswith('.ipynb'):
        record('notebook_requires_output_review', 'REVIEW')
    return findings


def audit(root=ROOT, include_working_tree=False):
    root = Path(root).resolve()
    reviewed_examples = reviewed_source_examples(root)
    tracked = git('ls-files', '-z', root=root).decode().split('\0')
    tracked = [name for name in tracked if name]
    candidates = [name for name in git('ls-files', '--others', '--exclude-standard', '-z', root=root).decode().split('\0') if name] if include_working_tree else []
    snapshots, current = {}, []
    for name in tracked + candidates:
        path = root / name
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            current.append({'path': name, 'findings': [{'category': 'symlink_or_root_escape', 'severity': 'BLOCKER', 'count': 1}]})
            continue
        data = path.read_bytes()
        snapshots[name] = hashlib.sha256(data).hexdigest()
        found = inspect(name, data, reviewed_examples)
        if found:
            current.append({'path': name, 'findings': found})
    object_lines = git('rev-list', '--objects', '--all', root=root).decode('utf-8', errors='replace').splitlines()
    paths = {}
    for line in object_lines:
        oid, _, name = line.partition(' ')
        paths[oid] = name
    # A blob may have had multiple names (including a private extension/name).
    # Enumerate all reachable trees so a rename cannot conceal its classification.
    historical_names = {}
    commits = git('rev-list', '--all', root=root).decode().splitlines()
    for commit in commits:
        for entry in git('ls-tree', '-r', '-z', commit, root=root).split(b'\0'):
            if not entry:
                continue
            meta, name = entry.split(b'\t', 1)
            mode, kind, oid = meta.decode().split()
            if kind == 'blob':
                historical_names.setdefault(oid, set()).add(name.decode('utf-8', errors='replace'))
    specs = ('\n'.join(paths) + '\n').encode()
    types = subprocess.run(['git', 'cat-file', '--batch-check=%(objectname) %(objecttype) %(objectsize)'], input=specs, capture_output=True, check=True, cwd=root).stdout.decode().splitlines()
    blobs = [(oid, int(size)) for oid, kind, size in (line.split() for line in types) if kind == 'blob']
    metadata_objects = [(oid, kind, int(size)) for oid, kind, size in (line.split() for line in types) if kind in {'commit', 'tag'}]
    process = subprocess.Popen(['git', 'cat-file', '--batch'], cwd=root, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    historical = []
    try:
        for oid, size in blobs:
            process.stdin.write((oid + '\n').encode())
            process.stdin.flush()
            header = process.stdout.readline().decode().split()
            data = process.stdout.read(int(header[2]))
            if process.stdout.read(1) != b'\n':
                raise ValueError('Invalid Git object batch response')
            for name in sorted(historical_names.get(oid, {paths[oid]})):
                found = inspect(name, data, reviewed_examples)
                if found:
                    historical.append({'object': oid, 'path': name, 'bytes': size, 'findings': found})
        for oid, kind, size in metadata_objects:
            process.stdin.write((oid + '\n').encode())
            process.stdin.flush()
            header = process.stdout.readline().decode().split()
            data = process.stdout.read(int(header[2]))
            if process.stdout.read(1) != b'\n':
                raise ValueError('Invalid Git metadata response')
            found = inspect('git-' + kind + '-metadata', data)
            identities = re.findall(rb'^(?:author|committer|tagger) [^\n]*<([^>]+)>', data, re.MULTILINE)
            if any(not value.lower().endswith(b'@users.noreply.github.com') for value in identities):
                found.append({'category': 'git_identity_metadata_requires_review', 'severity': 'REVIEW', 'count': len(identities)})
            if found:
                historical.append({'object': oid, 'path': 'git-' + kind + '-metadata', 'bytes': size, 'findings': found})
    finally:
        process.stdin.close()
        process.stdout.close()
        process.wait(timeout=5)
    def counts(records):
        result = {}
        for item in records:
            for finding in item['findings']:
                result[finding['category']] = result.get(finding['category'], 0) + 1
        return result
    blockers = sum(f['severity'] == 'BLOCKER' for r in current + historical for f in r['findings'])
    return {
        'status': 'FAIL' if blockers else 'PASS_WITH_REVIEW',
        'head': git('rev-parse', 'HEAD', root=root).decode().strip(),
        'branch': git('branch', '--show-current', root=root).decode().strip(),
        'reachable_commits': int(git('rev-list', '--count', '--all', root=root)),
        'tracked_files': len(tracked), 'reachable_blobs_scanned': len(blobs),
        'review_candidates': len(candidates), 'git_metadata_objects_scanned': len(metadata_objects),
        'reachable_blob_bytes': sum(size for _, size in blobs),
        'current_findings': current, 'historical_findings': historical,
        'current_categories': counts(current), 'historical_categories': counts(historical),
        'tracked_sha256': snapshots,
        'reviewed_source_examples': reviewed_examples,
        'limits': ['Locally reachable refs after fetch; unadvertised remote refs, dangling objects and GitHub attachments/releases are not audited.', 'Pattern matching cannot prove absence of all secrets/PII.', 'Synthetic demo remains synthetic; other real municipality rows are blocked.', 'Only the pinned 12-row consumption example has a verified CC BY-SA 4.0 grant. Owner decisions on attribution/adapted contributions and historical identity/path disclosure are recorded in data/metadata/publication_rights_review.json. Scanner REVIEW findings remain visible; they do not authorize new unreviewed data or execute visibility/deployment/commit/push/merge.'],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='outputs/pages_publication_checks/repository_audit.json')
    parser.add_argument('--strict', action='store_true')
    parser.add_argument('--include-working-tree', action='store_true', help='Also inspect untracked review candidates, excluding ignored files')
    args = parser.parse_args()
    output = ROOT / args.output
    if not output.resolve().is_relative_to((ROOT / 'outputs').resolve()):
        parser.error('Audit evidence must stay in ignored outputs/')
    proof = audit(include_working_tree=args.include_working_tree)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(proof, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: proof[key] for key in ['status', 'head', 'branch', 'reachable_commits', 'tracked_files', 'reachable_blobs_scanned', 'reachable_blob_bytes', 'current_categories', 'historical_categories']}, indent=2))
    if args.strict and proof['status'] == 'FAIL':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
