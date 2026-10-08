"""Synthetic panels only: paired cases, block resampling and reproducibility."""
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from sberforecast import forecast_robustness as module


@pytest.fixture
def forecasts():
    rows=[]
    for uid in ['synthetic_a','synthetic_b','synthetic_c']:
        for index,month in enumerate(pd.period_range('2024-01',periods=5,freq='M')):
            rows.append(dict(municipality_id=uid,forecast_origin=month.end_time.normalize().strftime('%Y-%m-%d'),
                target_period=(month+1).start_time.strftime('%Y-%m-%d'),horizon=1,split='validation',
                history_cutoff=month.end_time.normalize().strftime('%Y-%m-%d'),y_true=100.,y_pred=90.-index,
                status='native'))
    baseline=pd.DataFrame(rows)
    candidate=baseline.copy()
    candidate.y_pred+=np.tile([1,2,3,4,5],3)
    return baseline,candidate


def test_exact_intersection_and_shuffled_alignment(forecasts):
    a,b=forecasts
    result=module.align_pair(a,b.sample(frac=1,random_state=3).iloc[:-1])
    assert len(result)==14
    expected=set(map(tuple,b.sample(frac=1,random_state=3).iloc[:-1][list(module.KEYS)].to_numpy()))
    assert set(map(tuple,result[list(module.KEYS)].to_numpy()))==expected


@pytest.mark.parametrize('damage',['duplicate','truth','split','cutoff','target','failed','missing_forecast'])
def test_pairwise_damage_is_never_hidden(forecasts,damage):
    a,b=forecasts
    if damage=='duplicate': b=pd.concat([b,b.iloc[[0]]],ignore_index=True)
    elif damage=='truth': b.loc[0,'y_true']=101
    elif damage=='split': b.loc[0,'split']='holdout'
    elif damage=='cutoff': b.loc[0,'history_cutoff']='2025-01-31'
    elif damage=='target': b.loc[0,'target_period']='2025-01-01'
    elif damage=='failed': b.loc[0,'status']='failed_model'
    else: b.loc[0,'y_pred']=np.nan
    with pytest.raises(ValueError): module.align_pair(a,b)


def test_missing_truth_preserved_as_exclusion_count_not_imputed(forecasts):
    a,b=forecasts
    a.loc[0,'y_true']=b.loc[0,'y_true']=np.nan
    assert len(module.align_pair(a,b))==14


def test_sign_and_municipality_origin_aggregation(forecasts):
    a,b=forecasts
    rows=module.align_pair(a,b)
    mo=module.aggregate_units(rows,'municipality_id')
    origin=module.aggregate_units(rows,'forecast_origin')
    assert mo.delta_mae.tolist()==[3.,3.,3.]
    assert origin.delta_mae.tolist()==[1.,2.,3.,4.,5.]
    assert mo.mae_baseline.tolist()==[12.,12.,12.]
    assert np.allclose(mo.relative_delta_pct,25.)
    reverse=module.align_pair(b,a)
    assert np.array_equal(rows.delta_mae,-reverse.delta_mae)


def test_origin_block_resampling_preserves_all_municipalities_and_macro_weights(forecasts):
    a,b=forecasts
    rows=module.align_pair(a,b).drop(index=[0,1,5]).reset_index(drop=True)
    origins=sorted(rows.forecast_origin.unique())
    rng=np.random.default_rng(42)
    indices=rng.integers(0,len(origins),size=(31,len(origins)))
    expected=[]
    for draw in indices:
        blocks=[rows.loc[rows.forecast_origin.eq(origins[index])] for index in draw]
        # Every selected block contains precisely the original municipalities for that origin.
        assert all(set(block.municipality_id)==set(rows.loc[rows.forecast_origin.eq(block.forecast_origin.iloc[0])].municipality_id) for block in blocks)
        sample=pd.concat(blocks,ignore_index=True)
        expected.append(sample.groupby('municipality_id').delta_mae.mean().mean())
    actual=module.bootstrap(rows,'origin',31,42)
    assert actual['bootstrap_mean']==pytest.approx(np.mean(expected))
    assert actual['ci_lower']==pytest.approx(np.percentile(expected,2.5))
    assert actual['ci_upper']==pytest.approx(np.percentile(expected,97.5))
    assert actual['share_positive_draws']==np.mean(np.asarray(expected)>0)


@pytest.mark.parametrize('unit',['origin','municipality'])
def test_fixed_seed_determinism_no_new_rows_or_input_mutation(forecasts,unit):
    a,b=forecasts
    rows=module.align_pair(a,b)
    before=rows.copy(deep=True)
    first=module.bootstrap(rows,unit,1000,42)
    second=module.bootstrap(rows,unit,1000,42)
    assert first==second
    pd.testing.assert_frame_equal(before,rows)
    assert first['n_origins']==5 and first['n_municipalities']==3


def test_one_origin_primary_not_estimable_secondary_is_spatial(forecasts):
    rows=module.align_pair(*forecasts).iloc[::5]
    primary=module.bootstrap(rows,'origin',1000,42)
    assert primary['interval_status']=='NOT_ESTIMABLE_ONE_ORIGIN'
    assert primary['iterations']==0 and primary['ci_lower'] is None and primary['share_positive_draws'] is None
    assert module.bootstrap(rows,'municipality',1000,42)['interval_status']=='CONDITIONAL_SPATIAL_SENSITIVITY_ONE_ORIGIN'


def test_two_and_three_origins_no_primary_interval(forecasts):
    rows=module.align_pair(*forecasts)
    for n in [2,3]:
        group=rows.loc[rows.forecast_origin.isin(sorted(rows.forecast_origin.unique())[:n])]
        assert module.bootstrap(group,'origin',1000,42)['ci_upper'] is None


def test_identical_models_zero_delta_and_zero_bootstrap(forecasts):
    a,_=forecasts
    rows=module.align_pair(a,a.copy())
    for unit in ['origin','municipality']:
        result=module.bootstrap(rows,unit,1000,42)
        assert result['observed_delta_mae']==result['ci_lower']==result['ci_upper']==0.
        assert result['share_positive_draws']==0


def test_h12_never_gets_inferential_interval(forecasts):
    rows=module.align_pair(*forecasts)
    rows['horizon']=12
    for unit in ['origin','municipality']:
        result=module.bootstrap(rows,unit,1000,42)
        assert result['interval_status']=='DESCRIPTIVE_ONLY_H12' and result['ci_lower'] is None


def test_concentration_separates_signs_and_zero_mass_is_not_zero_share():
    frame=pd.DataFrame({'delta_mae':[10,5,-20,-2,0]})
    result=module.concentration(frame)
    assert result['positive_mass']==15 and result['negative_mass']==22 and result['net_mass']==-7
    assert result['top20pct_positive_share']==pytest.approx(10/15)
    assert result['top20pct_negative_share']==pytest.approx(20/22)
    empty=module.concentration(pd.DataFrame({'delta_mae':[0.,0.]}))
    assert empty['top5_positive_share'] is None and empty['top5_negative_share'] is None


@pytest.mark.parametrize('args,category',[((1,.6,4,6,1),'A'),((1,.6,4,6,-1),'B'),
    ((1,.6,1,1,None),'B'),((1,.5,6,6,1),'C'),((-1,.8,6,6,1),'D'),((0,.8,6,6,1),'D')])
def test_predeclared_category_boundaries(args,category):
    assert module.classify(*args)==category


def test_across_split_reversal_cannot_be_hidden():
    frame=pd.DataFrame([dict(pair='A',horizon=1,delta_mae=1,interpretation='A',descriptive_only=False),
                        dict(pair='A',horizon=1,delta_mae=-1,interpretation='D',descriptive_only=False)])
    assert module.across_split_categories(frame)[0]['category']=='D'


def test_manifest_reproducibility_and_no_overwrite_on_synthetic_fixture(tmp_path,forecasts,monkeypatch):
    cfg=yaml.safe_load((Path(__file__).resolve().parents[1]/'configs/forecast_robustness.yaml').read_text())
    path=tmp_path/'configs/fixture.yaml'
    path.parent.mkdir()
    path.write_text(yaml.safe_dump(cfg))
    for name in ['src/sberforecast/forecast_robustness.py','scripts/run_forecast_robustness.py','tests/test_forecast_robustness.py']:
        file=tmp_path/name
        file.parent.mkdir(parents=True,exist_ok=True)
        file.write_text('synthetic fixture')
    source=tmp_path/'source.txt'
    source.write_text('synthetic saved predictions')
    rows=module.align_pair(*forecasts)
    rows['split']='holdout'
    monkeypatch.setattr(module,'load_gate',lambda root:({'A':rows},dict(status='PASS',maximum_metric_difference=0.,
        source_sha256={'source.txt':hashlib.sha256(source.read_bytes()).hexdigest()},pairs=[])))
    def git(args,**kwargs):
        return (cfg['branch'] if 'branch' in args else 'fixture_head' if 'rev-parse' in args else '').encode()
    monkeypatch.setattr(module.subprocess,'check_output',git)
    first=module.run(tmp_path,path,smoke=True,command=['python','synthetic_runner','--smoke'])
    out=tmp_path/cfg['smoke_output_dir']
    stored=json.loads((out/'run_manifest.json').read_text(encoding='utf-8'))
    assert stored['n_model_fits']==0 and stored['new_predictions']==0 and stored['seed']==42
    assert stored['code_sha256']==first['code_sha256']
    assert all(hashlib.sha256((out/name).read_bytes()).hexdigest()==digest for name,digest in stored['artifact_sha256'].items())
    with pytest.raises(ValueError,match='new directory'): module.run(tmp_path,path,smoke=True)


def test_no_model_or_network_calls():
    path=Path(module.__file__)
    tree=ast.parse(path.read_text(encoding='utf-8'))
    for node in ast.walk(tree):
        if isinstance(node,(ast.Import,ast.ImportFrom)):
            names=[node.module or ''] if isinstance(node,ast.ImportFrom) else [x.name for x in node.names]
            assert not any(x.split('.')[0] in {'lightgbm','catboost','prophet','requests','torch'} for x in names)
        if isinstance(node,ast.Call):
            name=node.func.attr if isinstance(node.func,ast.Attribute) else getattr(node.func,'id','')
            assert name not in {'fit','train','predict','run_experiment','urlopen'}


def test_report_tables_have_contiguous_rows_and_headlines_cover_fixed_pairs(forecasts):
    root=Path(__file__).resolve().parents[1]
    cfg=yaml.safe_load((root/'configs/forecast_robustness.yaml').read_text())
    rows=module.align_pair(*forecasts)
    tables=module.analyze({p:rows for p in module.PAIRS},cfg)
    manifest={'comparability_gate':{'maximum_metric_difference':0},'full_runtime_seconds':0.,
              'interpretation_across_splits':module.across_split_categories(tables['pairwise_metrics'])}
    # Populate holdout for the headline diagnostic without changing the synthetic forecasts.
    for name,frame in tables.items():
        tables[name]=pd.concat([frame,frame.assign(split='holdout')],ignore_index=True)
    report=module.render_report(tables,manifest)
    assert '| Category |\n|---' in report
    assert '|\n\n|' not in report
    assert 'ProphetAuto' in report and 'ProphetYearly' in report and 'Outlier' in report
