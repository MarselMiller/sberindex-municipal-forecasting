"""Paired diagnostics of saved forecasts; no model imports, fits or predictions."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd
import yaml

from .final_forecasting import KEYS, _normalise, recompute_metrics, verify_report_metrics


MODELS = {
    'SeasonalNaiveYoY': ('outputs/prophet_comparison_v1', 'SeasonalNaiveYoY', 'metrics.csv'),
    'ProphetAuto': ('outputs/prophet_comparison_v1', 'ProphetAuto', 'metrics.csv'),
    'ProphetYearly': ('outputs/prophet_comparison_v1', 'ProphetYearly', 'metrics.csv'),
    'LightGBMDirect': ('outputs/national_local_lightgbm_v1', 'L0', 'metrics_strategy.csv'),
    'National/Local + LightGBM': ('outputs/national_local_lightgbm_v1', 'LN', 'metrics_strategy.csv'),
}
PAIRS = {
    'A': ('SeasonalNaiveYoY', 'National/Local + LightGBM'),
    'B': ('ProphetAuto', 'SeasonalNaiveYoY'),
    'C': ('ProphetYearly', 'SeasonalNaiveYoY'),
    'D': ('ProphetAuto', 'National/Local + LightGBM'),
    'E': ('ProphetYearly', 'National/Local + LightGBM'),
    'F': ('LightGBMDirect', 'National/Local + LightGBM'),
}
SIGN = 'MAE baseline − MAE candidate; positive = candidate better'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe(value):
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [safe(v) for v in value]
    if isinstance(value, np.generic):
        return safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_json(path, value):
    path.write_text(json.dumps(safe(value), ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def align_pair(baseline, candidate):
    """Exact key intersection; never substitute truths, status failures or dates."""
    required = [*KEYS, 'split', 'y_true', 'y_pred', 'status', 'history_cutoff']
    for frame in (baseline, candidate):
        if not set(required).issubset(frame) or frame.duplicated(list(KEYS)).any():
            raise ValueError('Missing columns or duplicate pairwise keys')
        if not pd.to_datetime(frame.history_cutoff).le(pd.to_datetime(frame.forecast_origin)).all():
            raise ValueError('Future history cutoff')
        origin = pd.PeriodIndex(frame.forecast_origin, freq='M')
        target = pd.PeriodIndex(frame.target_period, freq='M')
        if not np.array_equal(target.asi8-origin.asi8, frame.horizon.to_numpy()):
            raise ValueError('Target is not own origin plus horizon')
    joined = baseline[required].merge(candidate[required], on=list(KEYS), suffixes=('_baseline', '_candidate'),
                                     how='inner', validate='one_to_one')
    if not joined.split_baseline.eq(joined.split_candidate).all():
        raise ValueError('Pairwise split mismatch')
    a, b = joined.y_true_baseline.to_numpy(), joined.y_true_candidate.to_numpy()
    if not ((a == b) | (np.isnan(a) & np.isnan(b))).all():
        raise ValueError('Pairwise truth mismatch')
    expected_split = joined.target_period.le('2024-06-01').map({True: 'validation', False: 'holdout'})
    if not joined.split_baseline.eq(expected_split).all():
        raise ValueError('Pairwise fixed calendar split mismatch')
    joined = joined.loc[np.isfinite(a)].copy()
    for side in ('baseline', 'candidate'):
        statuses = joined['status_'+side].astype(str)
        if not (statuses.eq('native') | statuses.str.startswith('fallback')).all():
            raise ValueError('Failed/unavailable predictions cannot be silently excluded')
        if not np.isfinite(joined['y_pred_'+side]).all():
            raise ValueError('Nonfinite predictions')
        joined['ae_'+side] = (joined.y_true_baseline-joined['y_pred_'+side]).abs()
    joined['delta_mae'] = joined.ae_baseline-joined.ae_candidate
    joined['split'] = joined.split_baseline
    return joined.sort_values(list(KEYS)).reset_index(drop=True)


def aggregate_units(rows, unit):
    result = rows.groupby(unit, sort=True).agg(mae_baseline=('ae_baseline', 'mean'),
        mae_candidate=('ae_candidate', 'mean'), n_cases=('delta_mae', 'size')).reset_index()
    result['delta_mae'] = result.mae_baseline-result.mae_candidate
    result['relative_delta_pct'] = np.where(result.mae_baseline > 0,
        100*result.delta_mae/result.mae_baseline, np.nan)
    return result


def bootstrap(rows, unit, iterations, seed, min_origins=4):
    """Origin draws retain all observed municipalities; re-evaluate equal-MO macro."""
    matrix = rows.pivot(index='municipality_id', columns='forecast_origin', values='delta_mae')
    values = matrix.to_numpy(dtype=float)
    observed = float(np.nanmean(np.nanmean(values, axis=1)))
    n_municipalities, n_origins = values.shape
    result = dict(unit=unit, observed_delta_mae=observed, n_origins=n_origins,
        n_municipalities=n_municipalities, iterations=0, seed=seed, bootstrap_mean=None,
        ci_lower=None, ci_upper=None, share_positive_draws=None, draw_sha256=None,
        interval_status='DESCRIPTIVE_EMPIRICAL_SENSITIVITY', delta_convention=SIGN)
    if int(rows.horizon.iloc[0]) == 12:
        result['interval_status'] = 'DESCRIPTIVE_ONLY_H12'
        return result
    if unit == 'origin' and n_origins < min_origins:
        result['interval_status'] = 'NOT_ESTIMABLE_ONE_ORIGIN' if n_origins == 1 else 'NOT_ESTIMATED_FEWER_THAN_FOUR_ORIGINS'
        return result
    if unit not in {'origin', 'municipality'}:
        raise ValueError('Unknown bootstrap unit')
    rng = np.random.default_rng(seed)
    draws = []
    for start in range(0, iterations, 500):
        size = min(500, iterations-start)
        if unit == 'municipality':
            indices = rng.integers(0, n_municipalities, size=(size, n_municipalities))
            estimates = np.nanmean(values, axis=1)[indices].mean(axis=1)
        else:
            indices = rng.integers(0, n_origins, size=(size, n_origins))
            weights = np.eye(n_origins, dtype=float)[indices].sum(axis=1)
            totals = weights @ np.nan_to_num(values).T
            counts = weights @ np.isfinite(values).astype(float).T
            estimates = np.nanmean(np.divide(totals, counts, out=np.full_like(totals, np.nan), where=counts>0), axis=1)
        draws.extend(estimates.tolist())
    draws = np.asarray(draws, dtype=float)
    lower, upper = np.percentile(draws, [2.5, 97.5])
    result.update(iterations=iterations, bootstrap_mean=float(draws.mean()), ci_lower=float(lower), ci_upper=float(upper),
                  share_positive_draws=float((draws>0).mean()), draw_sha256=hashlib.sha256(draws.tobytes()).hexdigest())
    if unit == 'municipality' and n_origins == 1:
        result['interval_status'] = 'CONDITIONAL_SPATIAL_SENSITIVITY_ONE_ORIGIN'
    return result


def classify(delta, mo_wins, origin_wins, n_origins, ci_lower):
    if delta <= 0:
        return 'D'
    if mo_wins <= .5:
        return 'C'
    if origin_wins > n_origins/2 and ci_lower is not None and ci_lower > 0:
        return 'A'
    return 'B'


def concentration(municipalities):
    delta = municipalities.delta_mae.to_numpy()
    result = {'positive_mass': float(np.maximum(delta, 0).sum()),
              'negative_mass': float(np.maximum(-delta, 0).sum()),
              'net_mass': float(delta.sum()), 'n_positive': int((delta>0).sum()),
              'n_negative': int((delta<0).sum()), 'n_municipalities': len(delta)}
    for label, size in [('top5', 5), ('top10', 10), ('top20pct', math.ceil(.2*len(delta)))]:
        result[label+'_count'] = min(size, len(delta))
        for name, values in [('positive', np.maximum(delta, 0)), ('negative', np.maximum(-delta, 0))]:
            mass = values.sum()
            result[label+'_'+name+'_share'] = float(np.sort(values)[-size:].sum()/mass) if mass > 0 else None
    lower, upper = np.quantile(delta, [.05, .95])
    result['winsorized_mean_5pct_sensitivity_only'] = float(np.clip(delta, lower, upper).mean())
    result['median_delta_mae'] = float(np.median(delta))
    return result


def load_gate(root):
    """Rescore saved cases against saved metrics, final CSV/JSON and printed reports."""
    root = Path(root)
    provenance, cache, sources, records = {}, {}, {}, []
    def register(name):
        path = (root/name).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError('Missing/unsafe source: '+name)
        provenance[name] = sha(path)
        return path
    final = pd.read_csv(register('reports/final/forecasting_metrics.csv'), float_precision='round_trip')
    summary = json.loads(register('reports/final/results_summary.json').read_text(encoding='utf-8'))
    maxima = []
    for label, (directory, code, metric_name) in MODELS.items():
        if directory not in cache:
            raw = pd.read_csv(register(directory+'/predictions.csv.gz'), dtype={'municipality_id': str}, float_precision='round_trip')
            frame = _normalise(raw, directory)
            if not pd.to_datetime(frame.history_cutoff).eq(pd.to_datetime(frame.forecast_origin)).all():
                raise ValueError('Saved history cutoff differs from the declared L=0 protocol')
            if not frame.availability_assumption.eq('month_end_plus_0_months').all():
                raise ValueError('Saved availability assumption mismatch')
            cache[directory] = frame
        frame = cache[directory].loc[lambda r: r.model.eq(code)].copy()
        if frame.empty:
            raise ValueError('Missing saved strategy: '+label)
        sources[label] = frame
        saved = pd.read_csv(register(directory+'/'+metric_name), float_precision='round_trip')
        for split in ('validation', 'holdout'):
            for h in (1, 3, 6, 12):
                selected = frame.loc[frame.split.eq(split) & frame.horizon.eq(h) & np.isfinite(frame.y_true)]
                metric = recompute_metrics(selected)
                if not len(selected):
                    continue
                row = saved.loc[saved.model.eq(code) & saved.split.eq(split) & saved.horizon.eq(h)]
                published = final.loc[final.model.eq(label) & final.split.eq(split) & final.horizon.eq(h)]
                js = [r for r in summary['forecasting']['records'] if (r['model'],r['split'],r['horizon']) == (label,split,h)]
                if len(row) != 1 or len(published) != 1 or len(js) != 1:
                    raise ValueError('Missing saved/final metric group')
                for field in ('mae_macro','mae_micro','r2_pooled','n_predictions','n_municipalities','n_origins'):
                    for value in (row.iloc[0][field],published.iloc[0][field],js[0][field]):
                        if not math.isclose(metric[field],value,abs_tol=1e-8,rel_tol=0):
                            raise ValueError('Metric reproduction mismatch: '+label+'/'+field)
                        maxima.append(abs(metric[field]-value))
                records.append(dict(model=label,source_model=code,split=split,horizon=h,**metric))
    for directory, report, experiment in [
        ('outputs/prophet_comparison_v1','reports/results/E01_prophet_comparison.md','E01'),
        ('outputs/national_local_lightgbm_v1','reports/results/E05d_national_local_lightgbm.md','E05d')]:
        scoped = [r for r in records if MODELS[r['model']][0] == directory]
        verify_report_metrics(register(report).read_text(encoding='utf-8'),scoped,experiment)
        register(directory+'/run_manifest.json')
    register('reports/results/national_local_persistence.md')
    register('reports/final/RESULTS_SUMMARY.md')
    register('reports/final/METHODOLOGY_REPORT.md')
    keys = _normalise(pd.read_csv(register('outputs/national_local_lightgbm_v1/strategy_keys.csv'),dtype={'municipality_id':str}), 'strategy_keys',predictions=False)
    pairs, checks = {}, []
    for pair,(base,cand) in PAIRS.items():
        a,b = sources[base],sources[cand]
        aligned = align_pair(a,b)
        expected = set(map(tuple,keys[list(KEYS)].to_numpy()))
        if set(map(tuple,aligned[list(KEYS)].to_numpy())) != expected:
            raise ValueError('Pairwise intersection differs from the fixed evaluable keys: '+pair)
        pairs[pair] = aligned
        checks.append(dict(pair=pair,baseline=base,candidate=cand,primary=pair!='F',raw_baseline_rows=len(a),
            raw_candidate_rows=len(b),comparable_rows=len(aligned),n_municipalities=aligned.municipality_id.nunique(),
            baseline_nonfinite_truth=int((~np.isfinite(a.y_true)).sum()),candidate_nonfinite_truth=int((~np.isfinite(b.y_true)).sum()),
            pairwise_coverage_baseline=1.0,pairwise_coverage_candidate=1.0,gate_status='PASS'))
    return pairs,dict(status='PASS',tolerance_absolute=1e-8,maximum_metric_difference=max(maxima),
        pairs=checks,source_sha256=provenance,verification='saved_prediction_rescoring_no_fits',
        availability='saved cutoff and calendar verified; L=0 and historical vintages remain assumptions')


def analyze(pairs, config, smoke=False):
    tables = {name: [] for name in ['pairwise_metrics','municipality_deltas','municipality_summary','origin_deltas',
                                   'origin_summary','bootstrap_origin','bootstrap_municipality','concentration','applicability_checks']}
    iterations = config['smoke_iterations'] if smoke else config['bootstrap_iterations']
    for pair, rows in pairs.items():
        base,cand = PAIRS[pair]
        if smoke and pair != 'A':
            continue
        for (split,h),group in rows.groupby(['split','horizon'],sort=True):
            if smoke and (split,h) != ('holdout',1):
                continue
            meta = dict(pair=pair,baseline=base,candidate=cand,primary_pair=pair!='F',split=split,horizon=int(h),
                        descriptive_only=int(h)==12,delta_convention=SIGN)
            mo = aggregate_units(group,'municipality_id')
            origin = aggregate_units(group,'forecast_origin')
            refs = {uid: f'M{index+1:03}' for index,uid in enumerate(sorted(group.municipality_id.unique()))}
            mo['municipality_ref'] = mo.municipality_id.map(refs)
            delta = float(mo.delta_mae.mean())
            tol = config['ties_absolute_tolerance']
            def summary_units(units,unit):
                win = int((units.delta_mae>tol).sum())
                tie = int((units.delta_mae.abs()<=tol).sum())
                best,worst = units.loc[units.delta_mae.idxmax()],units.loc[units.delta_mae.idxmin()]
                return dict(n_units=len(units),candidate_wins=win,candidate_losses=len(units)-win-tie,ties=tie,
                    candidate_win_rate=win/len(units),mean_delta_mae=float(units.delta_mae.mean()),
                    median_delta_mae=float(units.delta_mae.median()),q25=float(units.delta_mae.quantile(.25)),
                    q75=float(units.delta_mae.quantile(.75)),min_delta_mae=float(units.delta_mae.min()),
                    max_delta_mae=float(units.delta_mae.max()),best_unit=best[unit],worst_unit=worst[unit])
            ms,os = summary_units(mo,'municipality_ref'),summary_units(origin,'forecast_origin')
            ob = bootstrap(group,'origin',iterations,config['seed'],config['minimum_primary_origins'])
            mb = bootstrap(group,'municipality',iterations,config['seed'],config['minimum_primary_origins'])
            category = 'DESCRIPTIVE_ONLY' if h==12 else classify(delta,ms['candidate_win_rate'],os['candidate_wins'],os['n_units'],ob['ci_lower'])
            tables['pairwise_metrics'].append(dict(**meta,n_comparable_rows=len(group),n_municipalities=len(mo),
                n_origins=len(origin),mae_baseline=float(mo.mae_baseline.mean()),mae_candidate=float(mo.mae_candidate.mean()),
                delta_mae=delta,relative_delta_pct=100*delta/mo.mae_baseline.mean(),interpretation=category,
                baseline_fallback_count=int(group.status_baseline.str.startswith('fallback').sum()),
                candidate_fallback_count=int(group.status_candidate.str.startswith('fallback').sum())))
            for name,units in [('municipality_deltas',mo),('origin_deltas',origin)]:
                tables[name].extend(dict(**meta,**r) for r in units.to_dict('records'))
            for name,data in [('municipality_summary',ms),('origin_summary',os),('bootstrap_origin',ob),
                              ('bootstrap_municipality',mb),('concentration',concentration(mo))]:
                tables[name].append({**meta,**data})
            tables['applicability_checks'].append(dict(**meta,n_target_periods=int(group.target_period.nunique()),
                n_origins=len(origin),horizon_exceeds_origin_spacing=bool(h>1),cluster_dependence='shared_calendar_shocks',
                dm_status='NOT_RELIABLE_NOT_ESTIMATED',dm_reason='Only 1–6 temporal observations; multistep horizons may share information.',
                primary_ci_status=ob['interval_status'],municipality_ci_role='secondary_spatial_sensitivity_only',
                temporal_exchangeability_assumed=True,adjacent_origin_dependence_not_removed=True))
    return {name:pd.DataFrame(records) for name,records in tables.items()}


def across_split_categories(pairwise):
    results=[]
    for (pair,h),group in pairwise.loc[~pairwise.descriptive_only].groupby(['pair','horizon']):
        categories = set(group.interpretation)
        category = 'D' if (group.delta_mae<=0).any() else next(iter(categories)) if len(categories)==1 else 'B'
        results.append(dict(pair=pair,horizon=int(h),category=category,reason='Fixed validation/holdout direction check; no horizon selection'))
    return results


def run(root, config_path, *, gate_only=False, smoke=False, command=None):
    started=time.perf_counter()
    root=Path(root).resolve()
    config_path=Path(config_path).resolve()
    cfg=yaml.safe_load(config_path.read_text(encoding='utf-8'))
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=root).decode().strip()
    if branch!=cfg['branch'] or cfg['seed']!=42 or cfg['bootstrap_iterations']!=10000:
        raise ValueError('Wrong branch or changed predeclared resampling protocol')
    mode='gate' if gate_only else 'smoke' if smoke else 'full'
    out=(root/cfg[mode+'_output_dir' if mode!='full' else 'output_dir']).resolve()
    if not out.is_relative_to(root/'outputs') or out.exists():
        raise ValueError('Output must be a new directory inside outputs')
    pairs,gate=load_gate(root)
    out.mkdir(parents=True)
    save_json(out/'comparability_gate.json',gate)
    if gate_only:
        print('Comparability gate PASS; maximum metric difference',gate['maximum_metric_difference'])
        return gate
    tables=analyze(pairs,cfg,smoke)
    for name,frame in tables.items():
        frame.to_csv(out/(name+'.csv'),index=False)
    public=root/cfg['public_dir']
    if not smoke:
        if public.exists() or (root/cfg['report_path']).exists():
            raise ValueError('Research report/artifacts already exist; no overwrite')
        public.mkdir(parents=True)
        for name,frame in tables.items():
            if name!='municipality_deltas':
                frame.to_csv(public/(name+'.csv'),index=False)
        # Requested per-MO audit is local-only. Never whitelist in Git.
        tables['municipality_deltas'].to_csv(root/cfg['private_municipality_artifact'],index=False)
    tracked=subprocess.check_output(['git','ls-files'],cwd=root).decode().splitlines()
    code=['src/sberforecast/forecast_robustness.py','scripts/run_forecast_robustness.py','tests/test_forecast_robustness.py']
    manifest=dict(experiment=cfg['experiment'],mode=mode,created_at_utc=datetime.now(timezone.utc).isoformat(),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root).decode().strip(),git_branch=branch,
        git_dirty=bool(subprocess.check_output(['git','status','--short'],cwd=root).strip()),command=command,
        config=cfg,config_sha256=sha(config_path),code_sha256={p:sha(root/p) for p in code},
        versions={p:importlib.metadata.version(p) for p in ['numpy','pandas','PyYAML','pytest']},
        comparability_gate=gate,interpretation_across_splits=across_split_categories(tables['pairwise_metrics']),
        n_model_fits=0,new_predictions=0,no_tuning=True,no_feature_search=True,no_primary_subset_selection=True,
        full_runtime_seconds=time.perf_counter()-started,seed=cfg['seed'],bootstrap_iterations=10000 if not smoke else 200,
        delta_convention=SIGN,bootstrap_positive_share_is_p_value=False,
        primary_bootstrap_note='4–6 origins: descriptive empirical sensitivity only; adjacent origin dependence remains.',
        public_data_policy='No target/prediction rows or original municipality IDs. Per-MO deltas are local-only ignored audit.',
        artifact_sha256={p.name:sha(p) for p in out.glob('*.csv')},
        code_available_in_tracked_repository={p:p in tracked for p in code})
    save_json(out/'run_manifest.json',manifest)
    if not smoke:
        public_manifest=dict(manifest)
        public_manifest['artifact_sha256']={p.name:sha(p) for p in public.glob('*.csv') if p.name!='municipality_deltas.csv'}
        save_json(public/'run_manifest.json',public_manifest)
        (root/cfg['report_path']).write_text(render_report(tables,public_manifest),encoding='utf-8')
    if any(sha(root/p)!=digest for p,digest in gate['source_sha256'].items()):
        raise ValueError('Frozen source changed during analysis')
    print(mode,'PASS; fits=0; pair/split/horizon groups=',len(tables['pairwise_metrics']),'; runtime=',manifest['full_runtime_seconds'])
    return manifest


def render_report(tables,manifest):
    def fmt(value):
        return '—' if value is None or pd.isna(value) else f'{float(value):.2f}'
    metric=tables['pairwise_metrics']
    lines=['# Устойчивость прогнозов и неопределённость различий MAE',
        'Исследовательский вопрос: насколько различия MAE устойчивы по муниципальным образованиям и датам выпуска при короткой истории?',
        '## Протокол и сопоставимость',
        f"Gate **PASS**: максимальная разница воспроизведённых метрик {manifest['comparability_gate']['maximum_metric_difference']:.3g}, допуск 1e-8. Для каждой пары 1890 одинаковых случаев с конечным фактом у 63 МО; исходные 1897 запросов и 7 отсутствующих фактов проверены без замены прогнозов.",
        'Знак ΔMAE = MAE baseline − MAE candidate; положительное значение означает меньшую ошибку candidate. MAE macro — среднее MAE каждого МО с равными весами. Validation и holdout разделены по исходному календарному cutoff. F — дополнительная диагностика; A–E — основные сравнения.',
        'Сверены точные ключи МО × origin × target × horizon, факты, split, статусы, history_cutoff и воспроизведённые MAE/R²/counts с сохранёнными CSV, JSON и округлёнными таблицами прежних отчётов. Будущий target используется только для оценки. Проверены сохранённые даты cutoff, но исторические vintages и публикация расходов при L=0 остаются допущением; обучение и его provenance заново не проверялись.',
        '## Неопределённость и заранее заданные правила',
        'Primary: 10 000 origin-cluster draws, seed=42. Каждый выбранный origin переносится со всеми его МО; в каждой выборке заново считается paired equal-MO macro ΔMAE. Secondary: resampling МО со всеми их origins. Доля положительных draws — эмпирическая чувствительность к этому resampling scheme, не p-value и не вероятность преимущества в будущем.',
        'При одной origin primary CI NOT ESTIMABLE; при менее четырёх CI не оценивается. Интервалы при 4–6 origins — только descriptive sensitivity: отдельные origins могут оставаться зависимыми, их независимость или временная exchangeability не доказаны. Municipality bootstrap условен по наблюдаемым датам и не устраняет общие национальные шоки; он не заменяет primary.',
        'Категории фиксированы до расчёта: D при Δ≤0; C при Δ>0 и win-rate МО≤50%; A при Δ>0, строгом большинстве МО и origins и нижней границе 95% origin-интервала>0; остальные положительные случаи с большинством МО — B. Концентрация описывается отдельно, без нового порога. Общая категория pair/horizon — D при неположительном Δ хотя бы в одном split; иначе единая категория split или B при их расхождении. Это описательные категории, не тест значимости.',
        '## Парные результаты']
    lookup={name:{(r.pair,r.split,int(r.horizon)):r for r in tables[name].itertuples()} for name in ['municipality_summary','origin_summary','bootstrap_origin','bootstrap_municipality']}
    for split in ['validation','holdout']:
        lines.extend([f'### {split.capitalize()}','| Pair: candidate vs baseline | h | n | ΔMAE, руб. | МО wins | Origins wins | Origin 95% interval | Municipality 95% sensitivity | Category |','|---|---:|---:|---:|---:|---:|---|---|---|'])
        for r in metric.loc[metric.split.eq(split)].itertuples():
            key=r.pair,r.split,int(r.horizon)
            ms,os,ob,mb=[lookup[name][key] for name in ['municipality_summary','origin_summary','bootstrap_origin','bootstrap_municipality']]
            ci=lambda x: f'[{fmt(x.ci_lower)}; {fmt(x.ci_upper)}]' if pd.notna(x.ci_lower) else x.interval_status
            lines.append(f'| {r.pair}: {r.candidate} vs {r.baseline} | {r.horizon} | {r.n_comparable_rows} | {r.delta_mae:.2f} | {ms.candidate_win_rate:.2%} | {os.candidate_wins}/{os.n_units} | {ci(ob)} | {ci(mb)} | {r.interpretation} |')
    lines.extend(['## Концентрация и чувствительность','Positive improvement mass и negative deterioration mass считаются отдельно по municipality ΔMAE. Top 5/10/ceil(20% всех МО) выбираются по величине массы каждого знака. Доли не делятся на малый или отрицательный net gain. Primary не удаляет МО; дополнительно показаны median Δ и заранее заданное winsorized mean с границами 5/95%.',
        '| Pair | Split | h | Positive mass: top5 / top10 / top20% | Negative mass: top5 / top10 / top20% | Median Δ | Winsorized mean 5% |','|---|---|---:|---|---|---:|---:|'])
    for r in tables['concentration'].loc[lambda f:~f.descriptive_only].itertuples():
        parts=lambda sign: ' / '.join(fmt(100*getattr(r,label+'_'+sign+'_share')) if pd.notna(getattr(r,label+'_'+sign+'_share')) else '—' for label in ['top5','top10','top20pct'])
        lines.append(f'| {r.pair} | {r.split} | {r.horizon} | {parts("positive")}% | {parts("negative")}% | {r.median_delta_mae:.2f} | {r.winsorized_mean_5pct_sensitivity_only:.2f} |')
    lines.extend(['## Общая устойчивость по validation и holdout',
        '| Pair | h | Category |','|---|---:|---|'])
    lines.extend(f"| {r['pair']} | {r['horizon']} | {r['category']} |" for r in manifest['interpretation_across_splits'])
    lines.extend(['## Ответы на исследовательские вопросы',headline_answers(tables),
        '## Применимость формальных тестов и ограничения',
        'Diebold–Mariano: **NOT RELIABLE / NOT ESTIMATED**. Для каждого split/h доступны лишь 1–6 target periods, multi-step horizons разделяют историю и могут иметь зависимые loss differentials. Число МО не увеличивает число независимых временных точек. Two-way bootstrap не выполнялся как необязательное усложнение.',
        'История цели — 24 месяца. Holdout уже просмотрен; анализ post-hoc, не новый blind test. Общие national shocks создают зависимость между МО; origin-bootstrap сохраняет эту зависимость внутри даты, но не устраняет зависимость соседних dates. h12 имеет одну origin, только descriptive, без inference; learned references используют прежний SeasonalNaive fallback. Bootstrap описывает эмпирический sample, а не независимый будущий период; пороги, seed, единица bootstrap, варианты Prophet, horizons и sample не подбирались по результату.',
        f"Новых model fits: **0**; runtime полного анализа: **{manifest['full_runtime_seconds']:.3f} s**; seed=42.",
        '## Сохранённые артефакты',
        'Все публичные CSV содержат только сводные показатели по pair/split/h или origin, без y_true/y_pred и исходных IDs МО. Локальный `forecast_robustness/municipality_deltas.csv` содержит per-MO audit и исключён из Git; его полная копия находится также в ignored output_dir. Best/worst ссылки в summary — локальные порядковые обозначения, не исходные IDs.',
        ' · '.join(f'[{name}.csv](forecast_robustness/{name}.csv)' for name in tables if name!='municipality_deltas'),
        '[Run manifest](forecast_robustness/run_manifest.json) · [Конфигурация](../../configs/forecast_robustness.yaml)'])
    formatted=[]
    for line in lines:
        if formatted and not (line.startswith('|') and formatted[-1].startswith('|')):
            formatted.append('')
        formatted.append(line)
    return '\n'.join(formatted)+'\n'


def headline_answers(tables):
    """Generate factual answers from saved group summaries, without choosing horizons."""
    metrics=tables['pairwise_metrics']
    mo=tables['municipality_summary']
    origins=tables['origin_summary']
    lines=[]
    for pair in ['A','B','C','D','E']:
        group=metrics.loc[metrics.pair.eq(pair)&metrics.horizon.isin([1,3,6])]
        lines.append(f"**{pair}: {PAIRS[pair][1]} vs {PAIRS[pair][0]}.**")
        for split in ['validation','holdout']:
            detail=[]
            for r in group.loc[group.split.eq(split)].itertuples():
                m=mo.loc[mo.pair.eq(pair)&mo.split.eq(split)&mo.horizon.eq(r.horizon)].iloc[0]
                o=origins.loc[origins.pair.eq(pair)&origins.split.eq(split)&origins.horizon.eq(r.horizon)].iloc[0]
                detail.append(f'h{r.horizon}: Δ={r.delta_mae:.2f}, МО win-rate={m.candidate_win_rate:.2%}, origins={int(o.candidate_wins)}/{int(o.n_units)}, {r.interpretation}')
            lines.append(split+': '+'; '.join(detail)+'.')
    c=tables['concentration'].loc[lambda f:f.primary_pair&~f.descriptive_only]
    lines.append(f"Концентрация положительной массы top5 по primary split/h с ненулевой положительной массой находится между {100*c.top5_positive_share.min():.2f}% и {100*c.top5_positive_share.max():.2f}%; отрицательная масса учитывается отдельно.")
    ca=c.loc[c.pair.eq('A')&c.split.eq('holdout')].sort_values('horizon')
    detail=[]
    for r in ca.itertuples():
        ratio=r.top5_positive_share*r.positive_mass/r.net_mass if r.net_mass>0 else np.nan
        detail.append(f'h{r.horizon}: top5={100*r.top5_positive_share:.2f}% положительной массы, '
                      f'positive/negative mass={r.positive_mass:.2f}/{r.negative_mass:.2f}, '
                      f'вклад top5 положительных МО/net gain={ratio:.2f}')
    lines.append('**Outlier-влияние.** NL+LightGBM vs SNY, holdout: '+'; '.join(detail)+'. '
        'Выигрыш присутствует у большинства МО, но small net gain h3/h6 получается из почти компенсирующих '
        'друг друга улучшений и ухудшений. Положительная масса не ограничивается пятью МО; при этом их вклад '
        'может существенно превышать итоговый net gain. Отсутствие влияния отдельных территорий не установлено; '
        'primary результат сохранён без удаления МО.')
    a=metrics.loc[metrics.pair.eq('A')&metrics.horizon.isin([1,3,6])]
    prophet=metrics.loc[metrics.pair.isin(['B','C','D','E'])&metrics.horizon.isin([1,3,6])]
    lines.append(f"Across-split категории A-сравнения: {[(r['horizon'],r['category']) for r in across_split_categories(a)]}. Для четырёх Prophet-сравнений положительное aggregate Δ наблюдается в {int((prophet.delta_mae>0).sum())}/{len(prophet)} фиксированных split/h группах. Это сравнение robustness признаков, не доказательство универсального ranking или будущего превосходства.")
    yearly=prophet.loc[prophet.pair.isin(['C','E'])]
    auto=prophet.loc[prophet.pair.isin(['B','D'])&prophet.split.eq('validation')&prophet.horizon.isin([3,6])]
    lines.append(f'**Какой вывод устойчивее?** Seasonal/panel представление относительно ProphetYearly сохраняет '
        f'положительное aggregate Δ в {int((yearly.delta_mae>0).sum())}/{len(yearly)} группах обоих split. '
        f'Однако temporal интервалы остаются широкими, а сравнение с ProphetAuto на validation h3/h6 '
        f'даёт положительный Δ только в {int((auto.delta_mae>0).sum())}/{len(auto)} группах. '
        'Данные лучше поддерживают ограниченный вывод о сильном seasonal/panel benchmark относительно '
        'проверенного ProphetYearly, чем устойчивый ranking NL+LightGBM против SNY. Универсальное '
        'превосходство над обоими Prophet по всем периодам и горизонтам не подтверждено.')
    return '\n\n'.join(lines)
