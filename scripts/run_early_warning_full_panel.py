"""E07b: full-panel feasibility first; simple classifiers only after the gate."""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime,timezone
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from sberforecast.data import load_data,make_panel,sha256_file
from sberforecast.early_warning_full_panel import resolve_config,prepare_panel_residuals,compare_saved_pilot
from sberforecast.early_warning_full_panel_models import (PREDICTION_COLUMNS,METRIC_COLUMNS,EVENT_METRIC_COLUMNS,
    CALIBRATION_COLUMNS,COEFFICIENT_COLUMNS,ALERT_COLUMNS,EVENT_MATCH_COLUMNS,STATUS_COLUMNS)


def project_path(value: str) -> Path:
    target=(ROOT/value).resolve()
    if not target.is_relative_to(ROOT):
        raise ValueError('E07b path outside workspace')
    return target


def save_json(file: Path,value) -> None:
    file.parent.mkdir(parents=True,exist_ok=True)
    file.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf-8')


def read_csv(file: Path) -> pd.DataFrame:
    return pd.read_csv(file,dtype={'municipality_id':str,'series_id':str,'region_id':str},low_memory=False)


def save_csv(output: Path,name: str,table: pd.DataFrame) -> None:
    table.to_csv(output/name,index=False,compression='gzip' if name.endswith('.gz') else None)


def memory_gib() -> float:
    if os.name!='nt':
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2)
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(name,ctypes.c_size_t) for name in
            ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage',
             'QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
    counters=Counters()
    counters.cb=ctypes.sizeof(counters)
    if not ctypes.windll.psapi.GetProcessMemoryInfo(wintypes.HANDLE(-1),ctypes.byref(counters),counters.cb):
        raise OSError('Cannot measure Windows process memory before heavy E07b work')
    return counters.PeakWorkingSetSize/(1024**3)


def progress(value: dict) -> None:
    value=dict(value,peak_working_set_gib=round(memory_gib(),3))
    if value['peak_working_set_gib']>5.5:
        raise MemoryError('E07b process exceeded user 5.5 GiB bound; stopped')
    print(json.dumps(value,ensure_ascii=False,default=str),flush=True)


def preservation(stage: str,config_path: Path) -> None:
    target=ROOT/'outputs/e07b_checks/preservation_before.json'
    if stage=='before':
        if target.exists():
            raise FileExistsError(target)
        tracked=subprocess.check_output(['git','ls-files','src','scripts','configs','tests','README.md','.gitignore','pyproject.toml'],cwd=ROOT,text=True).splitlines()
        files=[ROOT/name for name in tracked if (ROOT/name).is_file()]
        for name in ('outputs/news_events_v1','outputs/news_events_v2','outputs/news_events_v3',
                     'outputs/early_warning_feasibility_v1','outputs/early_warning_news_audit_v1'):
            files.extend(file for file in (ROOT/name).rglob('*') if file.is_file())
        files += [ROOT/'reports/results/E07a_early_warning_feasibility.md',ROOT/'data/input/consumption_all_categories.csv']
        value=dict(recorded_at=datetime.now(timezone.utc).isoformat(),full_panel_config_sha256=sha256_file(config_path),
            git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            scope='Old tracked code/config/tests and reused news/E07a outputs only; no full old-output tree recheck',
            files={str(file.relative_to(ROOT)):sha256_file(file) for file in sorted(set(files))})
        save_json(target,value)
        print(json.dumps(dict(stage='before',files=len(value['files']),config_sha=value['full_panel_config_sha256'])))
    else:
        before=json.loads(target.read_text(encoding='utf-8'))
        changed=[name for name,sha in before['files'].items() if not (ROOT/name).is_file() or sha256_file(ROOT/name)!=sha]
        result=dict(passed=not changed and before['full_panel_config_sha256']==sha256_file(config_path),
            checked_files=len(before['files']),changed=changed,config_unchanged=before['full_panel_config_sha256']==sha256_file(config_path))
        save_json(target.with_name('preservation_after.json'),result)
        print(json.dumps(result,ensure_ascii=False))
        if not result['passed']:
            raise AssertionError('Frozen E01–E07a input or E07b specification changed')


def write_manifest(output: Path,config: dict,stage: str,inputs: list[Path],started: str,clock: float) -> None:
    file=output/'manifest.json'
    old=json.loads(file.read_text(encoding='utf-8')) if file.exists() else dict(stages=[])
    code=sorted((ROOT/'src/sberforecast').glob('early_warning*.py'))+[Path(__file__),ROOT/'scripts/report_early_warning_full_panel.py',ROOT/'src/sberforecast/data.py',ROOT/'src/sberforecast/models.py',ROOT/'src/sberforecast/online_detection.py']
    # Windows venv launchers put the base executable in orig_argv[0].
    command=subprocess.list2cmdline([sys.executable]+sys.orig_argv[1:])
    launcher=subprocess.list2cmdline([sys.orig_argv[0]])
    actual_executable=subprocess.list2cmdline([sys.executable])
    for entry in old['stages']:
        if entry['command'].startswith(launcher+' '):
            entry['command_before_launcher_correction']=entry['command']
            entry['command']=actual_executable+entry['command'][len(launcher):]
    old['stages'].append(dict(stage=stage,command=command,started_at=started,completed_at=datetime.now(timezone.utc).isoformat(),
                             runtime_seconds=time.perf_counter()-clock,peak_working_set_gib=memory_gib()))
    old.update(experiment=config['experiment'],seed=config['seed'],python=sys.version,python_executable=sys.executable,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        git_status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),
        sklearn_installed=importlib.util.find_spec('sklearn') is not None,
        versions={d.metadata['Name']:d.version for d in importlib.metadata.distributions()},
        classifier_training_conditional_on_saved_gate=True,forecast_model_fit=False,network_requests=0,
        code={str(p.relative_to(ROOT)):sha256_file(p) for p in code},
        artifacts={str(p.relative_to(output)):dict(sha256=sha256_file(p),bytes=p.stat().st_size)
                   for p in sorted(output.rglob('*')) if p.is_file() and p!=file})
    old.setdefault('inputs',{}).update({str(p.relative_to(ROOT)):sha256_file(p) for p in inputs})
    save_json(file,old)


EMPTY_SCHEMAS={
 'predictions.csv.gz':PREDICTION_COLUMNS, 'metrics_row.csv':METRIC_COLUMNS,
 'metrics_event.csv':EVENT_METRIC_COLUMNS, 'calibration.csv':CALIBRATION_COLUMNS,
 'model_coefficients.csv':COEFFICIENT_COLUMNS,'alerts.csv':ALERT_COLUMNS,'event_matches.csv':EVENT_MATCH_COLUMNS,
 'examples.csv':['k','model','example_type','municipality_id','forecast_origin','event_id','onset_period','lead_time_months'],
}


def feasibility(config: dict,config_path: Path,smoke: bool=False) -> None:
    from sberforecast.early_warning_full_panel_labels import (build_panel_labels,summarize_feasibility,
         temporal_split,audit_all_calendar_splits,evaluate_gate)
    output=project_path(config['smoke_output_dir'] if smoke else config['output_dir'])
    if output.exists():
        raise FileExistsError(output)
    started,clock=datetime.now(timezone.utc).isoformat(),time.perf_counter()
    progress(dict(stage='preflight',estimated_peak_gib=1.2,memory_stop_gib=5.5,parallel_jobs=1))
    observations=load_data(project_path(config['data']['path']),config['data']['category'])
    panel=make_panel(observations)
    geo=read_csv(project_path(config['features']['geography_dictionary']))
    if smoke:
        # Fixed pilot IDs (including the no-forecast UID), never outcome selected.
        wanted=json.loads((ROOT/'outputs/prophet_comparison_v1/sample_ids.json').read_text(encoding='utf-8'))
        wanted=[str(uid) for uid in wanted[:4]]+['1471']
        panel=panel.loc[:,list(dict.fromkeys(wanted))]
    samples,residuals=prepare_panel_residuals(panel,config,geo,progress=progress)
    pilot=read_csv(ROOT/'outputs/online_detection_v1/real_residuals.csv.gz')
    pilot=pilot.loc[pilot.series_id.isin(panel.columns)]
    equality=compare_saved_pilot(residuals,pilot)
    progress(dict(stage='pilot_residual_equivalence',**equality))
    labels=build_panel_labels(residuals,samples,config,progress=progress)
    split=temporal_split(labels['cases'],config)
    labels['cases']=split
    counts=summarize_feasibility(split,labels['events'],samples=samples,config=config)
    alternatives=audit_all_calendar_splits(split,labels['events'],config)
    gates=evaluate_gate(counts,config)
    output.mkdir(parents=True)
    save_csv(output,'forecast_cases.csv.gz',samples)
    save_csv(output,'residuals.csv.gz',residuals)
    for name,table in labels.items():
        save_csv(output,{'events':'event_registry_weak.csv','cases':'label_cases.csv.gz'}.get(name,'weak_'+name+'.csv.gz'),table)
    save_csv(output,'feasibility.csv',counts)
    save_csv(output,'label_counts.csv',split.groupby(['k','label_status','eligible_at_origin','at_risk','split'],dropna=False).agg(cases=('municipality_id','size')).reset_index())
    save_csv(output,'train_test_dates.csv',split.groupby(['k','split','forecast_origin'],dropna=False).agg(cases=('municipality_id','size'),positives=('label',lambda v:int(v.eq(1).sum())),negatives=('label',lambda v:int(v.eq(0).sum()))).reset_index())
    save_csv(output,'calendar_split_options.csv',alternatives)
    save_csv(output,'feasibility_gate.csv',gates)
    for name,columns in EMPTY_SCHEMAS.items():
        save_csv(output,name,pd.DataFrame(columns=columns))
    save_json(output/'pilot_equivalence.json',equality)
    save_json(output/'execution_status.json',dict(feasibility_executed=True,features_executed=False,classifiers_fitted=0,
        classification_metrics_obtained=False,sklearn_installed=importlib.util.find_spec('sklearn') is not None,
        synthetic_smoke=False,technical_smoke_subset=smoke,gate=gates.to_dict('records'),
        skip_reason='not_trained_before_feasibility_gate',network_requests=0))
    (output/'config_resolved.yaml').write_text(yaml.safe_dump(config,allow_unicode=True,sort_keys=False),encoding='utf-8')
    write_manifest(output,config,'smoke' if smoke else 'feasibility',
        [config_path,project_path(config['protocol']['path']),project_path(config['data']['path']),
         project_path(config['panel']['eligibility_source']),project_path(config['features']['geography_dictionary']),ROOT/'outputs/online_detection_v1/real_residuals.csv.gz'],started,clock)
    progress(dict(stage='feasibility_complete',output=str(output.relative_to(ROOT)),municipalities=len(panel.columns),cases=len(split),weak_events=len(labels['events']),gate=gates.to_dict('records')))


def features(config: dict,config_path: Path,smoke: bool=False) -> None:
    from sberforecast.early_warning_full_panel_features import build_panel_features,feature_groups
    from sberforecast.early_warning_full_panel_models import TrainOnlyPreprocessor
    config=dict(config,output_dir=config['smoke_output_dir']) if smoke else config
    output=project_path(config['output_dir'])
    if not (output/'feasibility_gate.csv').is_file():
        raise ValueError('Run feasibility first, before features/classifiers')
    if (output/'features.csv.gz').exists():
        raise FileExistsError('Feature stage already completed')
    started,clock=datetime.now(timezone.utc).isoformat(),time.perf_counter()
    samples=read_csv(output/'forecast_cases.csv.gz')
    residuals=read_csv(output/'residuals.csv.gz')
    observations=load_data(project_path(config['data']['path']),config['data']['category'])
    selected=json.loads(project_path(config['features']['detector_parameters']).read_text(encoding='utf-8'))
    macro=read_csv(project_path(config['features']['macro_table']))
    source_ids=json.loads(project_path(config['features']['macro_source_ids']).read_text(encoding='utf-8'))
    docs=read_csv(project_path(config['features']['news_documents']))
    cfg=dict(config,feature_assembly={'provenance_dir':'feature_provenance'})
    wide,coverage,dictionary=build_panel_features(observations,residuals,samples,cfg,selected,macro,source_ids,docs,progress=progress)
    save_csv(output,'features.csv.gz',wide)
    save_csv(output,'feature_coverage.csv',coverage)
    save_csv(output,'feature_dictionary.csv',dictionary)
    save_json(output/'feature_groups.json',feature_groups())
    cases=read_csv(output/'label_cases.csv.gz')
    audits=[]
    keys=['municipality_id','forecast_origin']
    for k in [1,3]:
        train_keys=cases.loc[cases.k.eq(k)&cases.split.eq('train'),keys].drop_duplicates()
        training=wide.merge(train_keys,on=keys,validate='one_to_one')
        for model,groups in [('B1',['A']),('B2',['A','B']),('B3',['A','B','C']),('B4',['A','B','C','D'])]:
            columns=[column for group in groups for column in feature_groups()[group]]
            if training.empty:
                audit=pd.DataFrame(dict(feature=columns,selected=False,reason='no_training_rows'))
            else:
                preprocess=TrainOnlyPreprocessor(columns).fit(training)
                audit=preprocess.audit()
            audit['k'],audit['model']=k,model
            audits.append(audit)
    save_csv(output,'feature_filter_train.csv',pd.concat(audits,ignore_index=True))
    old_features=read_csv(ROOT/'outputs/early_warning_feasibility_v1/features.csv.gz')
    old_features=old_features.loc[old_features.municipality_id.isin(wide.municipality_id)]
    pilot=wide.merge(old_features[keys],on=keys,how='inner',validate='one_to_one')
    compare=old_features.merge(pilot,on=keys,suffixes=('_old','_new'),validate='one_to_one')
    numeric=[c for c in old_features if c in wide and c not in keys+['region_id','region'] and pd.api.types.is_numeric_dtype(old_features[c])]
    for col in numeric:
        if not np.allclose(compare[col+'_old'].to_numpy(float),compare[col+'_new'].to_numpy(float),rtol=0,atol=1e-8,equal_nan=True):
            raise AssertionError('Pilot feature changed: '+col)
    save_json(output/'pilot_feature_equivalence.json',dict(passed=True,rows=len(compare),numeric_columns=len(numeric)))
    status=json.loads((output/'execution_status.json').read_text(encoding='utf-8'))
    status.update(features_executed=True,feature_rows=len(wide),feature_dictionary_rows=len(dictionary),
        provenance_files=wide.attrs.get('provenance_files',[]),skip_reason='feasibility_gate_failed_no_classifier' if not read_csv(output/'feasibility_gate.csv').gate_pass.any() else 'awaiting_explicit_baseline_stage')
    save_json(output/'execution_status.json',status)
    write_manifest(output,config,'features',[config_path]+[project_path(config['features'][key]) for key in ('detector_parameters','detector_preparation','macro_table','macro_source_ids','news_documents')],started,clock)
    progress(dict(stage='features_complete',rows=len(wide),dictionary_rows=len(dictionary),classifiers_fitted=0))


def baselines(config: dict,config_path: Path) -> None:
    from sberforecast.early_warning_full_panel_models import fit_baselines
    output=project_path(config['output_dir'])
    started,clock=datetime.now(timezone.utc).isoformat(),time.perf_counter()
    if not (output/'feasibility_gate.csv').is_file():
        raise ValueError('Run feasibility first')
    gate=read_csv(output/'feasibility_gate.csv')
    if not gate.gate_pass.any():
        status=json.loads((output/'execution_status.json').read_text(encoding='utf-8'))
        status.update(classifiers_fitted=0,constant_risks_fitted=0,classification_metrics_obtained=False,skip_reason='feasibility_gate_failed_no_classifier')
        save_json(output/'execution_status.json',status)
        write_manifest(output,config,'baselines-skipped',[config_path],started,clock)
        print('All feasibility gates failed; no baseline fit or metrics produced',flush=True)
        return
    if not (output/'features.csv.gz').is_file():
        raise ValueError('Run features before fitting the allowed baselines')
    if importlib.util.find_spec('sklearn') is None:
        raise ModuleNotFoundError('sklearn absent; no installation authorized and no classifier fitted')
    cases=read_csv(output/'label_cases.csv.gz')
    wide=read_csv(output/'features.csv.gz')
    events=read_csv(output/'event_registry_weak.csv')
    groups=json.loads((output/'feature_groups.json').read_text(encoding='utf-8'))
    results=[]
    for k in [1,3]:
        if not gate.loc[gate.k.eq(k),'gate_pass'].all():
            continue
        results.append(fit_baselines(cases.loc[cases.k.eq(k)&cases.split.eq('train')],cases.loc[cases.k.eq(k)&cases.split.eq('test')],wide,events,groups,k,
            fit_cutoff=config['split']['train_information_cutoff']))
    artifact_keys={'predictions':'predictions.csv.gz','metrics_row':'metrics_row.csv','metrics_event':'metrics_event.csv',
                   'calibration':'calibration.csv','coefficients':'model_coefficients.csv','alerts':'alerts.csv',
                   'event_matches':'event_matches.csv','model_status':'model_status.csv'}
    for key,name in artifact_keys.items():
        save_csv(output,name,pd.concat([result[key] for result in results],ignore_index=True))
    examples=[]
    alerts=read_csv(output/'alerts.csv')
    matches=read_csv(output/'event_matches.csv')
    for example_type,flag in [('successful_warning','successful_warning'),('false_alert','false_alert')]:
        selected=alerts.loc[alerts[flag]].sort_values(['model','k','municipality_id','forecast_origin'],ascending=[False,True,True,True]).head(1)
        for row in selected.to_dict('records'):
            row['example_type']=example_type
            examples.append(row)
    for row in matches.loc[~matches.warning_observed].sort_values(['model','k','municipality_id','onset_period'],ascending=[False,True,True,True]).head(1).to_dict('records'):
        examples.append(dict(row,example_type='missed_event',forecast_origin=None))
    save_csv(output,'examples.csv',pd.DataFrame(examples,columns=EMPTY_SCHEMAS['examples.csv']))
    status=json.loads((output/'execution_status.json').read_text(encoding='utf-8'))
    fitted=pd.concat([result['model_status'] for result in results],ignore_index=True)
    fitted=fitted.loc[fitted.fit_performed]
    status.update(classifiers_fitted=int(fitted.model.ne('B0').sum()),constant_risks_fitted=int(fitted.model.eq('B0').sum()),
        classification_metrics_obtained=any(not result['metrics_row'].empty for result in results),skip_reason='')
    save_json(output/'execution_status.json',status)
    write_manifest(output,config,'baselines',[config_path],started,clock)


def audit_report(config: dict,config_path: Path) -> None:
    from sberforecast.early_warning_full_panel_audit import build_state_counts,build_feature_summary,build_news_summary
    from report_early_warning_full_panel import render_report
    output=project_path(config['output_dir'])
    started,clock=datetime.now(timezone.utc).isoformat(),time.perf_counter()
    cases=read_csv(output/'label_cases.csv.gz')
    wide=read_csv(output/'features.csv.gz')
    coverage=read_csv(output/'feature_coverage.csv')
    groups=json.loads((output/'feature_groups.json').read_text(encoding='utf-8'))
    save_csv(output,'state_counts.csv',build_state_counts(cases))
    save_csv(output,'feature_summary.csv',build_feature_summary(wide,coverage,groups))
    save_csv(output,'news_summary.csv',build_news_summary(cases,wide,groups))
    filters=read_csv(output/'feature_filter_train.csv')
    dictionary=read_csv(output/'feature_dictionary.csv')
    filters=filters.merge(dictionary[['feature','group']],on='feature',how='left',validate='many_to_one')
    filters['all_missing']=filters.reason.eq('all_missing_on_train')
    filters['constant']=filters.reason.eq('constant_after_training_median_imputation')
    filters['exact_duplicate']=filters.reason.eq('exact_numeric_duplicate_on_train')
    filters['no_training_rows']=filters.reason.eq('no_training_rows')
    save_csv(output,'feature_selection_summary.csv',filters.groupby(['k','model','group'],dropna=False).agg(
        candidates=('feature','size'),retained=('selected','sum'),all_missing=('all_missing','sum'),
        constant=('constant','sum'),exact_duplicate=('exact_duplicate','sum'),no_training_rows=('no_training_rows','sum')).reset_index())
    residuals=read_csv(output/'residuals.csv.gz')
    residuals['audit_forecast_origin']=(pd.PeriodIndex(residuals.observation_period,freq='M')-1).to_timestamp(how='end').normalize().strftime('%Y-%m-%d')
    save_csv(output,'forecast_coverage.csv',residuals.groupby(['audit_forecast_origin','forecast_status'],dropna=False).agg(
        cases=('series_id','size'),finite_predictions=('y_pred',lambda values:int(np.isfinite(values).sum())),
        finite_targets=('y_true',lambda values:int(np.isfinite(values).sum())),fallback_cases=('forecast_fallback','sum')).reset_index().rename(columns={'audit_forecast_origin':'forecast_origin'}))
    gates=read_csv(output/'feasibility_gate.csv')
    if not gates.gate_pass.any():
        for name,columns in EMPTY_SCHEMAS.items():
            if (output/name).exists():
                previous=read_csv(output/name)
                if not previous.empty:
                    raise ValueError('Unexpected fitted results despite both failed gates: '+name)
                if previous.columns.tolist()==list(columns):
                    continue
            save_csv(output,name,pd.DataFrame(columns=columns))
        save_csv(output,'model_status.csv',pd.DataFrame([dict(k=row.k,model=model,status='skipped_feasibility_gate',
            reason=row.failure_reasons,fit_performed=False,retained_features=0) for row in gates.itertuples() for model in config['models']['names']],columns=STATUS_COLUMNS))
    audit_inputs=[config_path]+[p for p in (ROOT/'outputs/e07b_checks/final_validation.json',ROOT/'outputs/e07b_checks/independent_label_check.json',
        ROOT/'outputs/e07b_checks/saved_artifact_validation.json',ROOT/'outputs/e07b_checks/preservation_after.json') if p.exists()]
    (output/'config_resolved.yaml').write_text(yaml.safe_dump(dict(config,feature_assembly={'provenance_dir':'feature_provenance'}),allow_unicode=True,sort_keys=False),encoding='utf-8')
    write_manifest(output,config,'audit-report',audit_inputs,started,clock)
    render_report(ROOT,config)
    # Report hashes live in the manifest rather than in the report itself.
    manifest=json.loads((output/'manifest.json').read_text(encoding='utf-8'))
    manifest['report']=dict(path=config['report_path'],sha256=sha256_file(project_path(config['report_path'])))
    save_json(output/'manifest.json',manifest)
    progress(dict(stage='audit_report_complete',report=config['report_path']))


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/early_warning_full_panel.yaml')
    parser.add_argument('--stage',choices=['preserve-before','preserve-after','smoke','features-smoke','feasibility','features','baselines','audit-report'],default='feasibility')
    args=parser.parse_args()
    config_path=project_path(args.config)
    config=resolve_config(ROOT,config_path)
    if args.stage.startswith('preserve-'):
        preservation(args.stage.removeprefix('preserve-'),config_path)
    elif args.stage in ('smoke','feasibility'):
        feasibility(config,config_path,smoke=args.stage=='smoke')
    elif args.stage in ('features','features-smoke'):
        features(config,config_path,smoke=args.stage=='features-smoke')
    elif args.stage=='audit-report':
        audit_report(config,config_path)
    else:
        baselines(config,config_path)


if __name__=='__main__':
    main()
