"""Select frozen figures; draw only saved observations/predictions/aggregate metrics."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_test_event_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """The source contains VALIDATION as well as TEST: never pool their rows."""
    if 'cohort' not in frame:
        raise ValueError('Event figure requires explicit cohort provenance')
    selected = frame.loc[frame.cohort.eq('test')].copy()
    expected = {(m, k) for m in ('S0', 'S1', 'S2', 'S3') for k in (1, 3)}
    if selected.duplicated(['model', 'k']).any() or set(map(tuple, selected[['model', 'k']].to_numpy())) != expected:
        raise ValueError('Event figure requires exactly eight distinct TEST model/k rows')
    if selected.eligible_events.nunique() != 1:
        raise ValueError('Event figure: synthetic TEST event denominators differ across models/horizons')
    return selected.sort_values(['k', 'model']).reset_index(drop=True)


def draw_synthetic_event_figure(root: Path, out: Path) -> pd.DataFrame:
    """Render one figure from saved TEST metrics only; used for post-build correction."""
    event = select_test_event_metrics(pd.read_csv(root / 'outputs/early_warning_synthetic_v1/metrics_event.csv'))
    event.to_csv(out / 'figure_data/synthetic_event_performance.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    for k, ax in zip((1, 3), axes):
        group = event.loc[event.k.eq(k)].sort_values('model')
        x = np.arange(len(group))
        ax.bar(x-.18, group.event_recall, .36, label='Event recall')
        ax.bar(x+.18, group.alert_precision, .36, label='Alert precision')
        ax.set_xticks(x, group.model)
        ax.set_ylim(0, 1)
        ax.set_title(f'k={k}; TEST: {int(group.eligible_events.iloc[0])} событий')
        ax.grid(axis='y', alpha=.2)
        ax.legend()
    fig.suptitle('Synthetic early warning: S0–S3\nS0 alert precision не определена: предупреждений нет')
    fig.savefig(out / 'figures/synthetic_event_performance.png', dpi=150, bbox_inches='tight',
                metadata={'Software': 'F1 saved-artifact rendering'})
    plt.close(fig)
    return event


def build_figures(root: Path, out: Path, forecasting: dict, detection: dict, evidence: dict) -> list[dict]:
    figures = out / 'figures'
    figures.mkdir(exist_ok=True)
    data_dir = out / 'figure_data'
    data_dir.mkdir(exist_ok=True)
    records = []
    final_relative = out.relative_to(root).as_posix()

    def register(group, filename, title, status, sources, selection, mode='drawn_from_saved_files', exported=None):
        paths = [root / s for s in sources]
        if any(not p.is_file() for p in paths):
            raise ValueError(f'{filename}: missing figure provenance')
        row = dict(group=group, figure=f'figures/{filename}', title=title, data_status=status, mode=mode,
                   sha256=digest(figures / filename), source_paths=json.dumps(sources, ensure_ascii=False),
                   source_sha256=json.dumps({s: digest(root / s) for s in sources}, ensure_ascii=False),
                   selection=selection, plotted_data=exported, visual_review='see verification.json; inspection occurs after build')
        records.append(row)

    def save(fig, filename):
        fig.savefig(figures / filename, dpi=150, bbox_inches='tight', metadata={'Software': 'F1 saved-artifact rendering'})
        plt.close(fig)

    forecast = pd.DataFrame(forecasting['records'])
    hold = forecast.loc[forecast['split'].eq('holdout')].copy()
    hold.to_csv(data_dir / 'forecast_mae.csv', index=False)
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), layout='constrained')
    for h, ax in zip((1, 3, 6, 12), axes.flat):
        group = hold.loc[hold.horizon.eq(h)]
        labels = group.model.str.replace('National/Local', 'N/L', regex=False).to_list()
        bars = ax.barh(labels, group.mae_macro, color='#35698d')
        for bar, fallback in zip(bars, group.fallback_count):
            if fallback:
                bar.set_hatch('///')
                bar.set_facecolor('#aeb8c2')
        ax.invert_yaxis()
        ax.set_title(f'h={h} месяцев' + ('; 1 origin, штриховка — fallback' if h == 12 else '; 6 origin'))
        ax.set_xlabel('MAE macro, руб.')
        ax.grid(axis='x', alpha=.2)
    fig.suptitle('Forecasting: сохранённый holdout, 63 МО\nHoldout уже просмотрен; устойчивый победитель не установлен', fontsize=14)
    save(fig, 'forecast_mae.png')
    register(1, 'forecast_mae.png', 'MAE macro по горизонтам: real pilot holdout', 'real',
             [final_relative + '/forecasting_metrics.csv'], '9 main strategies; holdout; identical E01/E05d keys', exported='figure_data/forecast_mae.csv')

    source = 'outputs/prophet_comparison_v1/predictions.csv.gz'
    predictions = pd.read_csv(root / source, dtype={'municipality_id': str})
    chosen = predictions.loc[predictions.model.eq('SeasonalNaiveYoY') & predictions.horizon.eq(1) & predictions.y_true.notna()].copy()
    uid = min(chosen.municipality_id.unique(), key=int)
    chosen = chosen.loc[chosen.municipality_id.eq(uid)].sort_values('target_period')
    chosen[['municipality_id', 'forecast_origin', 'target_period', 'y_true', 'y_pred', 'split']].to_csv(data_dir / 'rolling_forecast.csv', index=False)
    fig, ax = plt.subplots(figsize=(10, 4.5), layout='constrained')
    dates = pd.to_datetime(chosen.target_period)
    ax.plot(dates, chosen.y_true, marker='o', label='Факт «Все категории»')
    ax.plot(dates, chosen.y_pred, marker='s', linestyle='--', label='Сохранённый SeasonalNaiveYoY, h=1')
    ax.axvline(pd.Timestamp('2024-07-01'), color='#888888', linestyle=':', label='Начало просмотренного holdout')
    ax.set(title=f'МО {uid}: real rolling forecast из сохранённых прогнозов', ylabel='Средние безналичные расходы, руб.', xlabel='Месяц цели')
    ax.grid(alpha=.2)
    ax.legend(fontsize=9)
    save(fig, 'rolling_forecast.png')
    register(2, 'rolling_forecast.png', 'Один реальный rolling forecast: первый числовой ID', 'real', [source],
             f'municipality_id={uid}; SeasonalNaiveYoY; h=1; finite facts; numeric ID order, not accuracy', exported='figure_data/rolling_forecast.csv')

    detections = pd.DataFrame(detection['records'])
    online = detections.loc[detections.family.eq('online') & detections.data_status.eq('synthetic')].copy()
    online.to_csv(data_dir / 'online_comparison.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    x = np.arange(len(online))
    for j, (metric, label) in enumerate([('precision', 'Precision'), ('recall', 'Recall'), ('f1', 'F1')]):
        axes[0].bar(x + (j-1)*.24, online[metric], width=.24, label=label)
    axes[0].set_xticks(x, online.method)
    axes[0].set_ylim(0, 1)
    axes[0].legend()
    axes[0].set_title('Обнаружение: 360 устойчивых сдвигов')
    axes[1].bar(online.method, online.false_positives_per_12_months, color='#35698d')
    axes[1].set_title('FP / 12 наблюдаемых месяцев')
    axes[1].set_ylabel('Частота ложных сигналов')
    fig.suptitle('Online detection — synthetic TEST; качество на реальных МО не оценено')
    save(fig, 'online_comparison.png')
    register(3, 'online_comparison.png', 'Online quality metrics: synthetic test', 'synthetic', [final_relative + '/detection_metrics.csv'],
             'family=online; data_status=synthetic; primary level shifts', exported='figure_data/online_comparison.csv')

    old_manifest = json.loads((root / 'outputs/offline_detection_v1/plots_source.json').read_text(encoding='utf-8'))['figures']
    for group, old, filename, title, status in [
        (4, 'synthetic_level_example.png', 'offline_example.png', 'Offline synthetic example: первый level-series; оба метода пропустили событие', 'synthetic'),
        (5, 'municipality_21.png', 'real_diagnostic.png', 'МО 21: ретроспективные кандидаты, независимых меток нет', 'diagnostic')]:
        base = 'outputs/offline_detection_v1/'
        old_path = base + 'figures/' + old
        meta = old_manifest['figures/' + old]
        if digest(root / old_path) != meta['sha256']:
            raise ValueError(f'Frozen offline figure changed: {old_path}')
        sources = [base + s for s in meta['source_sha256']]
        for s, sha in meta['source_sha256'].items():
            if digest(root / base / s) != sha:
                raise ValueError(f'Offline figure source changed: {s}')
        shutil.copyfile(root / old_path, figures / filename)
        register(group, filename, title, status, sources,
                 json.dumps(meta['selection'], ensure_ascii=False), mode='selected_existing_figure')

    source = 'outputs/news_events_v3/news_features.csv.gz'
    news = pd.read_csv(root / source)
    value_cols = ['news_count_30d', 'news_count_90d', 'national_news_count_30d', 'regional_news_count_30d', 'municipal_news_count_30d']
    if news.groupby('forecast_origin')[value_cols].nunique(dropna=False).gt(1).any().any():
        raise ValueError('News vector varies spatially; national-copy figure definition invalid')
    timeline = news.groupby('forecast_origin', as_index=False)[value_cols].first()
    timeline.to_csv(data_dir / 'news_timeline.csv', index=False)
    fig, ax = plt.subplots(figsize=(11, 4.2), layout='constrained')
    positions = np.arange(len(timeline))
    ax.bar(positions, timeline.news_count_30d, label='Canonical events за 30 дней, national', color='#35698d')
    ax.plot(positions, timeline.news_count_90d, marker='o', color='#ce753a', label='Canonical events за 90 дней')
    ax.set_xticks(positions, timeline.forecast_origin.str[:7], rotation=35, ha='right')
    ax.set(title='News v3: одинаковый national-вектор для 64 МО; 12 origin', xlabel='Forecast origin', ylabel='Наблюдаемые события в ограниченном архиве')
    ax.legend()
    ax.grid(axis='y', alpha=.2)
    save(fig, 'news_timeline.png')
    register(6, 'news_timeline.png', 'Исторически допущенные news: national timeline', 'real', [source],
             'one vector per origin after verifying equality across all 64 MO; incomplete archive', exported='figure_data/news_timeline.csv')

    source = 'outputs/early_warning_full_panel_v1/feasibility.csv'
    feasibility = pd.read_csv(root / source)
    sufficient = feasibility.loc[feasibility.scope.isin(['train', 'test'])].copy()
    sufficient.to_csv(data_dir / 'real_warning_sufficiency.csv', index=False)
    rows = [[int(r.k), r.scope, int(r.positives), int(r.positive_event_onset_dates)] for r in sufficient.itertuples()]
    fig, ax = plt.subplots(figsize=(9, 3.2), layout='constrained')
    ax.axis('off')
    chart = ax.table(cellText=rows, colLabels=['k', 'Часть', 'Положительные метки', 'Onset-дат'], cellLoc='center', loc='center')
    chart.auto_set_font_size(False)
    chart.set_fontsize(11)
    chart.scale(1, 1.5)
    ax.set_title('Real early warning: данных недостаточно, B0–B4 не обучались\nТребования: train ≥30 / test ≥10 positives; ≥3 / ≥2 onset-дат', fontsize=12, pad=12)
    save(fig, 'real_warning_sufficiency.png')
    register(7, 'real_warning_sufficiency.png', 'Реальная панель: недостаточно событий и независимых дат', 'not_evaluated', [source],
             'fixed temporal split; scope=train/test; k1/3; counts not classifier metrics', exported='figure_data/real_warning_sufficiency.csv')

    base = 'outputs/early_warning_synthetic_v1/'
    for old, filename, title in [
        ('first_successful_warning.png', 'warning_success.png', 'S3/k3: успешное synthetic warning'),
        ('first_false_alert.png', 'warning_false.png', 'S3/k3: ложный synthetic alert'),
        ('first_missed_event.png', 'warning_miss.png', 'S3/k3: пропущенное synthetic событие')]:
        shutil.copyfile(root / base / 'plots' / old, figures / filename)
        register(8, filename, title, 'synthetic', [base+s for s in ['observations.csv.gz', 'predictions.csv.gz', 'true_events.csv',
                 'event_matches.csv', 'alerts.csv', 'examples.csv', 'selected_thresholds.csv']],
                 'saved examples.csv: model=S3, k=3, first eligible series lexicographically then earliest origin', mode='selected_existing_figure')

    source = base + 'metrics_event.csv'
    draw_synthetic_event_figure(root, out)
    register(9, 'synthetic_event_performance.png', 'S0–S3: synthetic event metrics', 'synthetic', [source],
             'TEST all events; k1/3; S0 undefined precision omitted', exported='figure_data/synthetic_event_performance.csv')
    return records
