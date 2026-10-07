"""E07c controlled benchmark: independent cohorts, train fit, validation threshold."""
from __future__ import annotations

import argparse
import copy
import ctypes
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

# This experiment uses one CPU job, including BLAS inside SciPy's optimizer.
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def local_path(value: str, root: Path | None = None) -> Path:
    root = ROOT if root is None else root
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('E07c artifacts must remain inside the project')
    return path


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')


def memory_gib() -> float:
    if os.name != 'nt':
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in ('PeakWorkingSetSize', 'WorkingSetSize',
            'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
            'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    if not ctypes.windll.psapi.GetProcessMemoryInfo(wintypes.HANDLE(-1), ctypes.byref(counters), counters.cb):
        raise OSError('Cannot measure process memory')
    return counters.PeakWorkingSetSize / (1024 ** 3)


def load_config(config_path: Path) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    if config['generator']['months'] != 24 or config['labels']['horizons'] != [1, 3]:
        raise ValueError('E07c requires 24 months and horizons 1/3')
    if config['models']['C'] != 1.0 or config['models']['class_weight'] is not None:
        raise ValueError('Fixed ordinary LogisticRegression with C=1 is required')
    if config['runtime']['parallel_jobs'] > 2 or config['runtime']['memory_stop_gib'] > 5:
        raise ValueError('Laptop runtime limits cannot be relaxed')
    seeds = [value['seed'] for value in config['cohorts'].values()]
    if len(set(seeds)) != 3 or set(config['cohorts']) != {'train', 'validation', 'test'}:
        raise ValueError('Three independent cohort seeds are required')
    for key in ('output_dir', 'smoke_output_dir', 'report_path'):
        local_path(config[key])
    if local_path(config['output_dir']) == local_path(config['smoke_output_dir']):
        raise ValueError('Smoke and full outputs must be separate')
    lock = ROOT / 'outputs/e07c_checks/preservation_before.json'
    if lock.exists() and json.loads(lock.read_text(encoding='utf-8'))['config_sha256'] != sha256(config_path):
        raise ValueError('Frozen E07c YAML changed after protocol lock')
    return config


def write_csv(output: Path, name: str, table: pd.DataFrame) -> None:
    compression = dict(method='gzip', mtime=0) if name.endswith('.gz') else None
    table.to_csv(output / name, index=False, compression=compression)


def write_manifest(output: Path, config: dict, config_path: Path, started_at: str,
                   started: float, phases: list[dict], stage: str) -> None:
    code = sorted((ROOT / 'src/sberforecast').glob('early_warning_synthetic*.py')) + [
        Path(__file__), ROOT / 'src/sberforecast/early_warning_full_panel_models.py',
        ROOT / 'src/sberforecast/models.py', ROOT / 'src/sberforecast/online_detection.py']
    versions = {distribution.metadata['Name']: distribution.version for distribution in importlib.metadata.distributions()}
    command = subprocess.list2cmdline([sys.executable] + list(getattr(sys, 'orig_argv', sys.argv))[1:])
    old_path = output / 'manifest.json'
    old = json.loads(old_path.read_text(encoding='utf-8')) if old_path.exists() else {}
    manifest = dict(experiment=config['experiment'], benchmark_type=config['benchmark_type'],
        stage=stage, seed=config['seed'], cohorts=config['cohorts'], command=command,
        python_executable=sys.executable, python_version=sys.version, versions=versions,
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        git_dirty_status=subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True).splitlines(),
        source_config_path=str(config_path.relative_to(ROOT)), source_config_sha256=sha256(config_path),
        started_at=started_at, finished_at=datetime.now(timezone.utc).isoformat(),
        runtime_seconds=time.perf_counter() - started, peak_working_set_gib=memory_gib(),
        parallel_jobs=1, blas_threads=1, network_requests=0, package_installations=0,
        phases=phases, code={str(path.relative_to(ROOT)):sha256(path) for path in code},
        artifacts={str(path.relative_to(output)):dict(sha256=sha256(path), bytes=path.stat().st_size)
            for path in sorted(output.rglob('*')) if path.is_file() and path.name != 'manifest.json'})
    manifest['validation_artifacts'] = {str(path.relative_to(ROOT)):sha256(path) for path in
        (ROOT/'outputs/e07c_checks'/name for name in ('targeted_results.json','independent_audit.json',
         'final_validation.json','preservation_after.json')) if path.exists()}
    if old and stage == 'report':
        manifest['prior_execution'] = {key:old[key] for key in ('stage', 'command', 'runtime_seconds', 'peak_working_set_gib', 'phases')}
        manifest['experiment_runtime_seconds'] = old.get('experiment_runtime_seconds', old['runtime_seconds'])
        manifest['experiment_peak_working_set_gib'] = old.get('experiment_peak_working_set_gib', old['peak_working_set_gib'])
        manifest['experiment_command'] = old.get('experiment_command', old['command'])
        manifest['experiment_code'] = old.get('experiment_code', old['code'])
    else:
        manifest['experiment_runtime_seconds'] = manifest['runtime_seconds']
        manifest['experiment_peak_working_set_gib'] = manifest['peak_working_set_gib']
        manifest['experiment_command'] = manifest['command']
        manifest['experiment_code'] = manifest['code']
    report = local_path(config['report_path'])
    if stage != 'smoke' and report.exists():
        manifest['report'] = dict(path=config['report_path'], sha256=sha256(report))
    save_json(old_path, manifest)


def run_experiment(config: dict, config_path: Path, *, smoke: bool = False) -> Path:
    from sberforecast.early_warning_synthetic import (generate_cohorts, build_features,
        build_cases, feature_groups, feature_dictionary)
    from sberforecast.early_warning_synthetic_models import fit_training_models, predict_models
    from sberforecast.early_warning_synthetic_evaluation import select_threshold, evaluate_predictions
    from sberforecast.early_warning_synthetic_report import render_report

    effective = copy.deepcopy(config)
    if smoke:
        for name in effective['cohorts']:
            effective['cohorts'][name]['size'] = config['smoke']['sizes'][name]
            effective['cohorts'][name]['seed'] += config['smoke']['seed_offset']
        effective['output_dir'] = config['smoke_output_dir']
    output = local_path(effective['output_dir'])
    if output.exists():
        raise FileExistsError('Existing E07c results cannot be overwritten: ' + str(output))
    output.mkdir(parents=True)
    started_at, started = datetime.now(timezone.utc).isoformat(), time.perf_counter()
    phases = []
    previous = started
    def checkpoint(stage: str, **details):
        nonlocal previous
        current, memory = time.perf_counter(), memory_gib()
        if memory > config['runtime']['memory_stop_gib']:
            raise MemoryError('E07c exceeded the 5 GiB process-memory limit')
        phase = dict(stage=stage, runtime_seconds=current - previous, peak_working_set_gib=memory, **details)
        phases.append(phase)
        previous = current
        print(json.dumps(phase, ensure_ascii=False, default=str), flush=True)

    save_json(output / 'generator_config.json', effective['generator'])
    (output / 'config_resolved.yaml').write_text(yaml.safe_dump(effective, allow_unicode=True, sort_keys=False), encoding='utf-8')
    data = generate_cohorts(effective)
    observations, metadata, events, cohort_ids = (data[name] for name in ('observations', 'metadata', 'events', 'cohort_ids'))
    write_csv(output, 'observations.csv.gz', observations)
    write_csv(output, 'series_metadata.csv', metadata)
    write_csv(output, 'true_events.csv', events)
    write_csv(output, 'cohort_ids.csv', cohort_ids)
    for name in effective['cohorts']:
        write_csv(output, name + '_ids.csv', cohort_ids.loc[cohort_ids.cohort.eq(name)])
    checkpoint('generation', series=len(metadata), observations=len(observations), events=len(events))

    features, cases = build_features(observations, effective), build_cases(observations, events, effective)
    groups = feature_groups()
    write_csv(output, 'features.csv.gz', features)
    write_csv(output, 'label_cases.csv.gz', cases)
    write_csv(output, 'feature_dictionary.csv', feature_dictionary())
    save_json(output / 'feature_groups.json', groups)
    admitted = cases.loc[cases.fully_known & cases.at_risk & cases.eligible].copy()
    counts = cases.assign(positive=cases.label.eq(1), negative=cases.label.eq(0), unknown=cases.label.isna(),
        monitored=cases.fully_known & cases.at_risk & cases.eligible).groupby(['cohort', 'k'], as_index=False).agg(
        audit_rows=('label', 'size'), positives=('positive', 'sum'), negatives=('negative', 'sum'),
        unknown=('unknown', 'sum'), active=('active_regime', 'sum'), monitored=('monitored', 'sum'))
    admitted_counts = admitted.groupby(['cohort', 'k'], as_index=False).agg(
        monitored_positives=('label', 'sum'), monitored_rows=('label', 'size'))
    counts = counts.merge(admitted_counts, on=['cohort', 'k'], validate='one_to_one')
    counts['monitored_positive_rate'] = counts.monitored_positives / counts.monitored_rows
    write_csv(output, 'label_counts.csv', counts)
    checkpoint('causal_features_and_labels', feature_rows=len(features), numeric_candidates=sum(map(len, groups.values())),
        estimated_dense_numeric_gib=features.shape[0] * sum(map(len, groups.values())) * 8 / (1024 ** 3))

    train = admitted.loc[admitted.cohort.eq('train')]
    try:
        fitted = fit_training_models(train, features.loc[features.cohort.eq('train')], groups, effective)
    except Exception as error:
        save_json(output / 'execution_failure.json', dict(stage='train_fit', exception_type=type(error).__name__,
            message=str(error), optimizer_diagnostics=getattr(error, 'diagnostics', None),
            experiment_completed=False, test_evaluated=False))
        write_manifest(output, effective, config_path, started_at, started, phases, 'failed_train_fit')
        raise
    write_csv(output, 'model_coefficients.csv', fitted['coefficients'])
    write_csv(output, 'feature_filter_train.csv', fitted['feature_filter'])
    write_csv(output, 'model_status.csv', fitted['model_status'])
    save_json(output / 'preprocessing.json', fitted['preprocessing'])
    save_json(output / 'model_parameters.json', fitted['parameters'])
    checkpoint('train_fit', models=len(fitted['models']))

    validation = admitted.loc[admitted.cohort.eq('validation')]
    val_predictions = predict_models(fitted, validation, features.loc[features.cohort.eq('validation')])
    thresholds = []
    for (model, k), subset in val_predictions.groupby(['model', 'k'], sort=True):
        choice = select_threshold(subset, events.loc[events.cohort.eq('validation')],
            metadata.loc[metadata.cohort.eq('validation')], effective)
        thresholds.append({**choice, 'model': model, 'k': int(k)})
    threshold_table = pd.DataFrame(thresholds)
    write_csv(output, 'selected_thresholds.csv', threshold_table)
    val_predictions = predict_models(fitted, validation, features.loc[features.cohort.eq('validation')], thresholds=threshold_table)
    checkpoint('validation_thresholds', selected_models=len(threshold_table), test_used=False)

    # No test labels, test predictions or scenario scores were read by fitting/selection.
    test = admitted.loc[admitted.cohort.eq('test')]
    test_predictions = predict_models(fitted, test, features.loc[features.cohort.eq('test')], thresholds=threshold_table)
    predictions = pd.concat([val_predictions, test_predictions], ignore_index=True)
    write_csv(output, 'predictions.csv.gz', predictions)
    tables = evaluate_predictions(predictions, events.loc[events.cohort.ne('train')],
        metadata.loc[metadata.cohort.ne('train')], effective)
    for name, table in tables.items():
        if isinstance(table, pd.DataFrame):
            write_csv(output, name + '.csv', table)
    checkpoint('independent_test_evaluation', prediction_rows=len(test_predictions), bootstrap_resamples=effective['evaluation']['bootstrap_resamples'])
    save_json(output / 'execution_status.json', dict(synthetic=True, real_data_used=False,
        generator_changed_after_test_evaluation=False, fitted_model_entries=len(fitted['models']),
        test_used_for_fit=False, test_used_for_threshold=False, experiment_completed=True,
        full_pytest_completed=False, network_requests=0, package_installations=0, smoke=smoke))
    write_manifest(output, effective, config_path, started_at, started, phases, 'smoke' if smoke else 'full')
    render_report(ROOT, effective, smoke=smoke)
    checkpoint('saved_report_and_plots')
    write_manifest(output, effective, config_path, started_at, started, phases, 'smoke' if smoke else 'full')
    print(json.dumps(dict(stage='completed', output_dir=effective['output_dir'], smoke=smoke), ensure_ascii=False), flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/early_warning_synthetic.yaml')
    parser.add_argument('--stage', choices=['smoke', 'full', 'report'], default='smoke')
    args = parser.parse_args()
    config_path = local_path(args.config)
    config = load_config(config_path)
    if args.stage in ('smoke', 'full'):
        run_experiment(config, config_path, smoke=args.stage == 'smoke')
    else:
        from sberforecast.early_warning_synthetic_report import render_report
        output = local_path(config['output_dir'])
        started_at, started = datetime.now(timezone.utc).isoformat(), time.perf_counter()
        render_report(ROOT, config)
        status = json.loads((output / 'execution_status.json').read_text(encoding='utf-8'))
        validation_path = ROOT / 'outputs/e07c_checks/final_validation.json'
        validation = json.loads(validation_path.read_text(encoding='utf-8')) if validation_path.exists() else {}
        status['full_pytest_completed'] = validation.get('exit_code') == 0
        save_json(output / 'execution_status.json', status)
        write_manifest(output, config, config_path, started_at, started, [], 'report')
        print(json.dumps(dict(stage='report_completed', report=config['report_path']), ensure_ascii=False))


if __name__ == '__main__':
    main()
