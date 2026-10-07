"""Saved-artifact E06a report and plots; no segmentation or scoring is run here."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .online_report import table

METHODS = ("PELT", "BinSeg")
PRIMARY_METRICS = ("precision", "recall", "f1", "miss_rate", "median_absolute_localisation_error",
                   "median_breakpoint_offset", "false_positives_per_12_months")
OFFLINE_COLUMNS = ["method", "n_events", *PRIMARY_METRICS]
ONLINE_COLUMNS = ["method", "n_events", "precision", "recall", "f1", "missed_fraction",
                  "median_delay", "false_alarms_per_12_months"]
COLORS = {"PELT": "#176b99", "BinSeg": "#d97024"}


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"municipality_id": str, "series_id": str})


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require(frame: pd.DataFrame, fields, label: str) -> None:
    if missing := set(fields) - set(frame):
        raise ValueError(f"Неполная сохранённая схема {label}: {sorted(missing)}.")


def _series_field(frame: pd.DataFrame) -> str:
    if "municipality_id" in frame:
        return "municipality_id"
    if "series_id" in frame:
        return "series_id"
    raise ValueError("Не сохранён идентификатор ряда.")


def _dates(values) -> pd.DatetimeIndex:
    return pd.PeriodIndex(values, freq="M").to_timestamp()


def _primary_tables(output: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics, intervals = _csv(output / "synthetic_test_metrics.csv"), _csv(output / "synthetic_test_ci.csv")
    _require(metrics, ["scope", *OFFLINE_COLUMNS], "offline test metrics")
    _require(intervals, ["method", "scope", "metric", "estimate", "ci_low", "ci_high", "bootstrap_unit",
                         "stratified", "conditional_on_selected_parameters"], "bootstrap CI")
    primary = metrics.loc[metrics.scope.eq("primary_level")].copy()
    ci = intervals.loc[intervals.scope.eq("primary_level")].copy()
    if set(metrics.method) - set(METHODS) or primary.duplicated("method").any():
        raise ValueError("Основная offline-таблица должна содержать фиксированные методы без повторений.")
    if (not ci.bootstrap_unit.eq("series").all() or ci.duplicated(["method", "metric"]).any()
            or not ci.stratified.eq(True).all() or not ci.conditional_on_selected_parameters.eq(True).all()):
        raise ValueError("CI должны использовать целые ряды и уникальные группы method/metric.")
    for row in primary.itertuples(index=False):
        if not np.isclose(row.median_absolute_localisation_error, row.median_breakpoint_offset,
                          equal_nan=True, atol=1e-12, rtol=1e-12):
            raise ValueError("В одностороннем окне T…T+3 медианы absolute error и offset должны совпадать.")
        for metric in PRIMARY_METRICS:
            selected = ci.loc[ci.method.eq(row.method) & ci.metric.eq(metric)]
            if len(selected) != 1:
                raise ValueError("Для каждой основной метрики нужен сохранённый bootstrap-интервал.")
            interval = selected.iloc[0]
            value = getattr(row, metric)
            if not np.isclose(interval.estimate, value, equal_nan=True, atol=1e-12, rtol=1e-12):
                raise ValueError("Point estimate CI отличается от сохранённой основной метрики.")
            if (pd.isna(interval.ci_low) != pd.isna(interval.ci_high)
                    or (pd.notna(interval.ci_low) and interval.ci_low > interval.ci_high)):
                raise ValueError("Некорректные сохранённые границы CI.")
    return primary, ci


def _cached_plots(output: Path, cfg: dict):
    path = output / "plots_source.json"
    if not path.exists():
        return None
    record = _json(path)
    fingerprint = hashlib.sha256(json.dumps(cfg, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if record["configuration_sha256"] != fingerprint:
        raise ValueError("Сохранённые графики относятся к другой конфигурации.")
    for name, item in record["figures"].items():
        image = output / name
        if not image.resolve().is_relative_to(output.resolve()) or not image.exists() or _sha(image) != item["sha256"]:
            raise ValueError("Сохранённый график изменён или находится вне output_dir.")
        for source, digest in item["source_sha256"].items():
            data_path = output / source
            if not data_path.resolve().is_relative_to(output.resolve()) or _sha(data_path) != digest:
                raise ValueError("Источник сохранённого графика изменён.")
    return record


def plot_saved(output: Path, cfg: dict) -> dict:
    """Create or verify static plots of saved rows with a SHA256 source map.

    Truth is used only to annotate one preselected synthetic illustration.
    Breakpoints, parameters, CI and prefix associations are read from disk.
    """
    if cached := _cached_plots(output, cfg):
        return cached
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import dates as mdates, pyplot as plt

    output = Path(output)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    primary, intervals = _primary_tables(output)
    real = _csv(output / "real_residuals.csv.gz")
    real_points = _csv(output / "real_breakpoints.csv")
    segments = _csv(output / "real_segments.csv")
    prefix = _csv(output / "real_prefix_estimates.csv")
    series = _csv(output / "synthetic_test_series.csv.gz")
    observations = _csv(output / "synthetic_test_observations.csv.gz")
    residuals = _csv(output / "synthetic_test_residuals.csv.gz")
    truth = _csv(output / "synthetic_test_events.csv.gz")
    synthetic_points = _csv(output / "synthetic_test_breakpoints.csv.gz")
    _require(real, ["observation_period", "y_true", "y_pred"], "real residuals")
    _require(prefix, ["method", "prefix_end_period", "breakpoint_month"], "prefix estimates")
    _require(series, ["series_id", "scenario"], "synthetic series")
    eligible = series.loc[series.scenario.isin(["level_up", "level_down"]), "series_id"].astype(str)
    if not len(eligible):
        raise ValueError("Не сохранён ни один level-ряд для заранее заданной иллюстрации.")
    synthetic_id = sorted(eligible)[0]
    real_field = _series_field(real)
    all_real_ids = real[real_field].dropna().astype(str).unique().tolist()
    if any(not value.isdigit() for value in all_real_ids):
        raise ValueError("Правило реальных иллюстраций требует числовые ID МО.")
    real_ids = sorted(all_real_ids, key=int)[:3]
    record = {"configuration_sha256": hashlib.sha256(json.dumps(cfg, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
              "detector_fit_performed": False, "metrics_recomputed": False, "network_used": False,
              "illustration_selection": {"rule": "three_smallest_numeric_real_ids_and_smallest_level_series_id",
                                           "municipality_ids": real_ids, "synthetic_series_id": synthetic_id}, "figures": {}}

    def save(fig, name, sources, selection):
        destination = figures / name
        if destination.exists():
            raise FileExistsError("График без проверенного plots_source.json не перезаписывается.")
        fig.tight_layout()
        fig.savefig(destination, dpi=150, metadata={"Software": "sberforecast E06a saved-artifact plotting"})
        plt.close(fig)
        record["figures"][destination.relative_to(output).as_posix()] = {
            "sha256": _sha(destination), "source_sha256": {source: _sha(output / source) for source in sources},
            "selection": selection, "future_access": "retrospective_known_interval_not_early_warning"}

    def month_axis(axis):
        axis.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        axis.tick_params(axis="x", rotation=35)
        axis.grid(alpha=.2)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, metric, label in zip(axes, ("f1", "recall"), ("F1", "Recall")):
        for position, method in enumerate(METHODS):
            selected = primary.loc[primary.method.eq(method)]
            if not len(selected):
                axis.text(position, .05, "Не выбран", ha="center")
                continue
            value = float(selected.iloc[0][metric])
            ci = intervals.loc[intervals.method.eq(method) & intervals.metric.eq(metric)].iloc[0]
            axis.scatter(position, value, color=COLORS[method], s=55, zorder=3)
            if pd.notna(ci.ci_low):
                axis.vlines(position, ci.ci_low, ci.ci_high, color=COLORS[method], linewidth=2)
                axis.hlines([ci.ci_low, ci.ci_high], position-.07, position+.07, color=COLORS[method])
        axis.set(xticks=range(len(METHODS)), xticklabels=METHODS, ylim=(0, 1), title=label)
        axis.grid(axis="y", alpha=.2)
    fig.suptitle("Offline level-shift test: point estimates and whole-series 95% CI")
    save(fig, "offline_metrics_ci.png", ["synthetic_test_metrics.csv", "synthetic_test_ci.csv"], {"scope": "primary_level"})

    for municipality_id in real_ids:
        current = real.loc[real[real_field].astype(str).eq(municipality_id)].sort_values("observation_period")
        dates = _dates(current.observation_period)
        errors = current.error if "error" in current else current.y_true - current.y_pred
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        axes[0].plot(dates, current.y_true, "o-", label="Observed expenses")
        axes[0].plot(dates, current.y_pred, "--", label="Saved SeasonalNaiveYoY h=1")
        axes[0].set_ylabel("Nominal RUB")
        axes[0].legend(fontsize=8)
        axes[1].plot(dates, errors, "o-", color="#666666", label="Saved residual y−prediction")
        for method in METHODS:
            own_points = real_points.loc[real_points[_series_field(real_points)].astype(str).eq(municipality_id) & real_points.method.eq(method)]
            own_segments = segments.loc[segments[_series_field(segments)].astype(str).eq(municipality_id) & segments.method.eq(method)]
            for point in own_points.itertuples(index=False):
                axes[1].axvline(pd.Period(point.breakpoint_month, freq="M").to_timestamp(), color=COLORS[method], linestyle="--", alpha=.75)
            for segment in own_segments.itertuples(index=False):
                axes[1].hlines(segment.mean_error, pd.Period(segment.start_period, freq="M").to_timestamp(),
                               pd.Period(segment.end_period, freq="M").end_time, color=COLORS[method], linewidth=2)
            axes[1].plot([], [], color=COLORS[method], label=f"{method}: retrospective segments")
        axes[1].set_ylabel("Residual, RUB")
        axes[1].legend(fontsize=8)
        month_axis(axes[1])
        fig.suptitle(f"Municipality {municipality_id}: retrospective candidates, no verified event labels")
        save(fig, f"municipality_{municipality_id}.png", ["real_residuals.csv.gz", "real_breakpoints.csv", "real_segments.csv"],
             {"municipality_id": municipality_id, "selection": "numeric_ID_order_not_detector_quality"})

        fig, axis = plt.subplots(figsize=(9, 5))
        for method, marker in zip(METHODS, ("o", "x")):
            own = prefix.loc[prefix[_series_field(prefix)].astype(str).eq(municipality_id) & prefix.method.eq(method)]
            if len(own):
                axis.scatter(_dates(own.prefix_end_period), mdates.date2num(_dates(own.breakpoint_month)),
                             color=COLORS[method], marker=marker, label=f"{method}: independently fitted prefix", alpha=.65)
            full = real_points.loc[real_points[_series_field(real_points)].astype(str).eq(municipality_id) & real_points.method.eq(method)]
            for point in full.itertuples(index=False):
                axis.axhline(mdates.date2num(pd.Period(point.breakpoint_month, freq="M").to_timestamp()),
                             color=COLORS[method], linestyle="--", alpha=.6)
        # Matplotlib expands a constant numeric date around zero relative
        # range, which can show several spurious years. Use the actual saved
        # calendar even when every prefix estimates exactly the same month.
        lower = dates.min() - pd.Timedelta(days=15)
        upper = dates.max() + pd.Timedelta(days=15)
        axis.set_ylim(mdates.date2num(lower), mdates.date2num(upper))
        axis.yaxis.set_major_locator(mdates.MonthLocator(interval=2))
        axis.yaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        axis.set_xlabel("End month of independently fitted prefix")
        axis.set_ylabel("Estimated breakpoint month")
        axis.set_title("Dashed lines: full-sample reference", fontsize=10)
        month_axis(axis)
        if len(axis.collections):
            axis.legend(fontsize=8)
        else:
            axis.text(.5, .5, "No candidate breakpoint in saved prefixes", transform=axis.transAxes, ha="center")
        fig.suptitle(f"Municipality {municipality_id}: prefix stability is a hindsight diagnostic")
        save(fig, f"prefix_stability_{municipality_id}.png", ["real_prefix_estimates.csv", "real_breakpoints.csv"],
             {"municipality_id": municipality_id, "prefix_points": "all_saved_estimates_including_transient_unmatched",
              "calendar_span": [str(dates.min().to_period("M")), str(dates.max().to_period("M"))]})

    current_observations = observations.loc[observations.series_id.eq(synthetic_id)].sort_values("observation_period")
    current_residuals = residuals.loc[residuals.series_id.eq(synthetic_id)].sort_values("observation_period")
    current_truth = truth.loc[truth.series_id.eq(synthetic_id)]
    current_points = synthetic_points.loc[synthetic_points.series_id.eq(synthetic_id)]
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(_dates(current_observations.observation_period), current_observations.value, label="Synthetic expenses")
    axes[0].plot(_dates(current_residuals.observation_period), current_residuals.y_pred, "--", label="Causal saved h=1 forecast")
    axes[0].set_ylabel("Nominal RUB")
    axes[1].plot(_dates(current_residuals.observation_period), current_residuals.error, "o-", label="Saved forecast residual")
    axes[1].set_ylabel("Residual, RUB")
    for event in current_truth.itertuples(index=False):
        for axis in axes:
            axis.axvline(pd.Period(event.event_start_period, freq="M").to_timestamp(), color="#b71c1c", label="Generator truth: plot annotation only")
    for method in METHODS:
        for point in current_points.loc[current_points.method.eq(method)].itertuples(index=False):
            axes[1].axvline(pd.Period(point.breakpoint_month, freq="M").to_timestamp(), color=COLORS[method], linestyle="--", alpha=.75)
        axes[1].plot([], [], color=COLORS[method], linestyle="--", label=f"{method}: offline estimate")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    month_axis(axes[1])
    fig.suptitle(f"{synthetic_id}: first level series by ID; truth never passed to fit")
    save(fig, "synthetic_level_example.png", ["synthetic_test_series.csv.gz", "synthetic_test_observations.csv.gz",
         "synthetic_test_residuals.csv.gz", "synthetic_test_events.csv.gz", "synthetic_test_breakpoints.csv.gz"],
         {"series_id": synthetic_id, "selection": "smallest_ID_among_level_up_down_not_detection_quality"})
    (output / "plots_source.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record


def _selected_table(selected: dict) -> str:
    rows = []
    for method in METHODS:
        item = selected[method]
        rows.append({"method": method, "selected": item["selected"],
                     "parameters": json.dumps(item.get("parameters"), sort_keys=True) if item["selected"] else item.get("reason", "no feasible candidate"),
                     "no_change FAR validation": item.get("validation_control_far", {}).get("no_change", np.nan),
                     "outlier FAR validation": item.get("validation_control_far", {}).get("outlier", np.nan)})
    return table(pd.DataFrame(rows), list(rows[0]))


def _prefix_table(summary: pd.DataFrame) -> str:
    fields = [_series_field(summary), "method", "full_sample_breakpoint", "first_prefix_where_detected",
              "first_breakpoint_estimate", "n_matched_prefixes", "n_date_revisions", "n_presence_losses",
              "minimum_estimated_month", "maximum_estimated_month", "revision_span_months"]
    _require(summary, fields, "real prefix summary")
    return table(summary, fields)


def _real_counts_table(diagnostics: pd.DataFrame, points: pd.DataFrame) -> str:
    fields = ["n_breakpoints", "n_evaluation_breakpoints", "n_warmup_breakpoints"]
    _require(diagnostics, ["method", "status", *fields], "real segmentation diagnostics")
    if not diagnostics.status.eq("complete").all():
        raise ValueError("Итоговый отчёт требует успешных сохранённых полных сегментаций.")
    if not diagnostics.n_breakpoints.eq(diagnostics.n_evaluation_breakpoints + diagnostics.n_warmup_breakpoints).all():
        raise ValueError("Полные кандидаты не равны сумме monitoring и warmup кандидатов.")
    summary = diagnostics.groupby("method", as_index=False)[fields].sum()
    actual = points.groupby("method").size()
    if any(int(row.n_breakpoints) != int(actual.get(row.method, 0)) for row in summary.itertuples(index=False)):
        raise ValueError("Счётчики реальных кандидатов отличаются от сохранённых breakpoint rows.")
    summary = summary.rename(columns={"n_breakpoints": "all_candidates",
                                     "n_evaluation_breakpoints": "monitoring_candidates",
                                     "n_warmup_breakpoints": "warmup_candidates"})
    return table(summary, ["method", "all_candidates", "monitoring_candidates", "warmup_candidates"])


def _offline_conclusion(primary: pd.DataFrame, ci: pd.DataFrame) -> str:
    valid = primary.loc[np.isfinite(primary.f1)]
    if not len(valid):
        return "Нет выбранного offline-метода с конечной основной F1; бюджет не ослаблялся ради результата."
    highest = valid.f1.max()
    best = valid.loc[valid.f1.eq(highest), "method"].tolist()
    result = f"Наибольшая точечная F1 именно в этом offline test: {', '.join(best)}, {highest:.6f}. "
    result += "Этот описательный результат не означает выбор общего победителя offline/online или превосходство на реальных МО. "
    f1_ci = ci.loc[ci.metric.eq("f1") & ci.ci_low.notna() & ci.ci_high.notna()]
    if len(f1_ci) >= 2:
        overlap = f1_ci.ci_low.max() <= f1_ci.ci_high.min()
        result += "Маргинальные 95% интервалы F1 " + ("пересекаются" if overlap else "не пересекаются") + ". "
    return result + "Парная статистическая значимость различий не проверена; неопределённость выбора параметров не включена в CI."


def write_report(output: Path, report_path: Path, cfg: dict) -> None:
    """Create only a completed saved-result report; never overwrite a report."""
    if report_path.exists():
        raise FileExistsError("Существующий отчёт E06a не перезаписывается.")
    status, manifest = _json(output / "run_status.json"), _json(output / "run_manifest.json")
    if not status["complete"] or not manifest["complete"] or manifest["config"] != cfg:
        raise ValueError("Отчёт требует завершённого запуска и совпадающей конфигурации.")
    tests = manifest["tests"]
    if tests["full_pytest"] is not True or tests["exit_code"] != 0:
        raise ValueError("Для отчёта нужен успешный полный pytest до эксперимента.")
    if (manifest.get("selected_parameters_sealed_before_test") is not True
            or manifest.get("parameters_selected_on") != "synthetic_validation_only"
            or manifest.get("labels_in_fit") is not False
            or manifest.get("real_threshold_tuning") is not False
            or manifest.get("forecasting_models_refitted") is not False
            or manifest.get("early_warning") is not False):
        raise ValueError("Manifest должен подтверждать validation seal, отсутствие truth в fit и неизменный прогнозный вход.")
    plots = _cached_plots(output, cfg)
    if plots is None or plots["detector_fit_performed"] or plots["metrics_recomputed"]:
        raise ValueError("Нужны проверяемые графики из сохранённых файлов, без повторного fit/scoring.")
    primary, intervals = _primary_tables(output)
    metrics = _csv(output / "synthetic_test_metrics.csv")
    online = _csv(output / "online_reference_metrics.csv")
    online_ci = _csv(output / "online_reference_ci.csv")
    selected = _json(output / "selected_parameters.json")
    online_selected = _json(output / "online_reference_selected.json")
    selection = _csv(output / "validation_selection.csv")
    real_points = _csv(output / "real_breakpoints.csv")
    real_diagnostics = _csv(output / "real_series_diagnostics.csv")
    prefix = _csv(output / "real_prefix_summary.csv")
    prefix_estimates = _csv(output / "real_prefix_estimates.csv")
    prefix_diagnostics = _csv(output / "real_prefix_diagnostics.csv")
    preflight = _json(output / "preflight.json")
    online_primary = online.loc[online.scope.eq("primary_level")]
    online_primary_ci = online_ci.loc[online_ci.scope.eq("primary_level")]
    _require(online_primary, ONLINE_COLUMNS, "saved E04a online reference")
    if set(online_primary.method) != {"CUSUM", "EWMA", "BOCPD"} or online_primary.duplicated("method").any():
        raise ValueError("Справка online должна содержать все три сохранённые модели E04a.")
    if set(primary.method) != {method for method in METHODS if selected[method]["selected"]}:
        raise ValueError("Основные test-метрики не соответствуют выбранным validation-методам.")
    ci_fields = ["method", "metric", "estimate", "ci_low", "ci_high", "n_bootstrap", "n_finite_bootstrap"]
    _require(intervals, ci_fields, "offline intervals")
    scenario = metrics.loc[metrics.scope.eq("by_scenario")]
    scenario_fields = ["method", "scenario", "n_series", "n_events", *PRIMARY_METRICS]
    _require(scenario, scenario_fields, "offline scenario metrics")
    point_counts = real_points.groupby("method").size().to_dict()
    status_counts = real_diagnostics.groupby(["method", "status"]).size().rename("n_series").reset_index()
    if status["n_failed"] != 0 or prefix_diagnostics.status.eq("failed").any():
        raise ValueError("Завершённый отчёт нельзя создавать при ошибке полного или prefix fit.")
    real_counts = _real_counts_table(real_diagnostics, real_points)
    prefix_status_counts = prefix_diagnostics.groupby(["method", "status"]).size().rename("n_prefixes").reset_index()
    version = manifest["versions"].get("ruptures")
    if not version:
        raise ValueError("Не сохранена фактическая версия ruptures.")
    real_ids = plots["illustration_selection"]["municipality_ids"]
    synthetic_id = plots["illustration_selection"]["synthetic_series_id"]
    generator = cfg["generator"]
    lag = cfg["preparation"]["release_lag_months"]
    lines = ["# E06a — ретроспективная сегментация PELT и Binary Segmentation", "",
        "Код подготовлен, полный pytest и эксперимент выполнены. Получены synthetic test-метрики "
        "и ретроспективные кандидаты на реальные изменения. PELT/BinSeg анализируют весь известный "
        "отрезок, включая данные после оценённой точки. Это offline benchmark; он не является "
        "ранним предупреждением и не заменяет сохранённую online-оценку CUSUM/EWMA/BOCPD.", "",
        "## Данные и фиксированный протокол", "",
        "Основное событие E04a сохранено: положительный/отрицательный сдвиг уровня расходов, "
        "устойчивый минимум три месяца. Наклон и дисперсия — дополнительные сценарии; одиночный "
        "выброс и отсутствие изменения — контроль. Используются тот же generator E04a, его "
        "validation/test seeds и параметры, без нового выгодного для offline генератора.",
        f"Validation: {status['n_validation_series']} рядов; test: {status['n_test_series']}. "
        f"Seeds начинаются с {generator['validation_seed']} и {generator['test_seed']}; "
        f"{generator['replicates_per_cell']} реализаций на scenario×strength×noise. "
        "В полном наборе 1260 рядов, включая 1140 основных и 120 граничных; "
        "основных level-событий 360. Граничные ранние/частичные окна не объявлены основными "
        "устойчивыми событиями. Исходные события, observations, forecasts, residuals и seeds сохранены.",
        "Синтетический test намеренно повторяет E04a: те же seeds и реализации уже использовались "
        "для online-методов и их результаты просмотрены. Поэтому это не новый независимый слепой "
        "benchmark. Для E06a test заново воспроизведён после фиксации validation-параметров; "
        "настройка по нему не выполняется, прежний доступ к результатам явно учитывается.",
        "Вход — error_t = y_true_t − y_pred_t; y_pred_t — причинный SeasonalNaiveYoY h=1 "
        "по неизменному E04a/E01 pipeline. Первые четыре конечные ошибки задают фиксированные "
        "c=median(e₁…e₄), s=max(1.4826 median|e−c|, 0.03 median|ŷ₁…ŷ₄|, 1 рубль), z=(e−c)/s. "
        "Offline cost после доступности этой калибровки использует все 12 ошибок января–декабря, "
        "включая четыре warmup. Первые четыре месяца не входят в оценочное exposure: "
        "основная оценка и контрольный бюджет используют те же восемь месяцев мая–декабря E04a. "
        "Будущие event labels не передаются fit; truth читается только для сопоставления и графика.",
        "Пропуски не заменены нулём. Алгоритм получает конечные наблюдения, а индекс границы b "
        "возвращается в исходный календарь как первый наблюдаемый месяц нового сегмента. "
        "Конечная граница n не считается breakpoint. При разрыве календаря истинная дата "
        "внутри разрыва не идентифицируется; boundary_gap_calendar_months сохраняет это ограничение.", "",
        "## Validation: параметры и бюджет", "",
        "Оба метода имеют cost='l2', min_size=2, jump=1. Сетка penalty=[0.5,1,2,4,8,16] "
        "фиксирована до test; другие cost-функции не перебирались. Для PELT penalty задаёт "
        "штраф сегментации, для BinSeg — заранее фиксированное штрафное правило остановки. "
        "Выбор только по validation: отдельно no_change и outlier FAR ≤1 на 12 наблюдаемых "
        "месяцев мониторинга; затем основной F1, recall и фиксированное правило разрешения равенств. "
        "Если допустимой точки нет, метод не выбран; бюджет не ослабляется. "
        "Test создан после сохранения и фиксации selected_parameters.json.", "", _selected_table(selected), "",
        "L2 cost сегмента — сумма квадратов отклонений z от среднего сегмента. "
        "PELT использует динамическое программирование с отсечением кандидатов для "
        "штрафной сегментации; Binary Segmentation последовательно делит сегменты по "
        "максимальному уменьшению cost и останавливается по фиксированному penalty. "
        "Cost/gain и estimated_shift не являются вероятностями экономического шока. "
        "Online reset/cooldown в сегментацию не добавлены. "
        "[PELT](https://centre-borelli.github.io/ruptures-docs/user-guide/detection/pelt/), "
        "[BinSeg](https://centre-borelli.github.io/ruptures-docs/user-guide/detection/binseg/), "
        "[CostL2](https://centre-borelli.github.io/ruptures-docs/user-guide/costs/costl2/).", "",
        "Все проверенные варианты:", "", table(selection, list(selection.columns)), "",
        "## A. Offline synthetic test", "",
        "Доступ к данным: весь анализируемый известный ряд, включая будущее относительно breakpoint. "
        "Успех — breakpoint в календарном окне T…T+3 включительно. До T — FP, не предупреждение. "
        "Сопоставление one-to-one; лишние/повторные точки — FP. Отдельно отмечены warmup и "
        "неполные окна. Для L>0 offline-окно определяется месяцем наблюдения; availability_date "
        "хранится отдельно и не расширяет историю фактов.", "", table(primary, OFFLINE_COLUMNS), "",
        "median_absolute_localisation_error и median_breakpoint_offset относятся только к detected "
        "events. Это ретроспективная локализация, не online detection delay. Miss rate показан "
        "вместе с локализацией; малое смещение обнаруженной части не означает большой recall.", "",
        "В одностороннем окне успеха T…T+3 offsets принадлежат 0…3: точки до T являются FP. "
        "Поэтому median absolute localisation error совпадает с median breakpoint offset "
        "по определению этого окна, а не вследствие дополнительного качества алгоритма.", "",
        "### По сценариям, включая контроли и граничные случаи", "", table(scenario, scenario_fields), "",
        "У контролей без истинных structural events recall/miss rate/локализация не определены. "
        "Полные результаты по scenario×strength×noise находятся в synthetic_test_metrics.csv; "
        "неудачные и граничные результаты сохранены вместе с основными.", "",
        "### 95% whole-series bootstrap CI", "", table(intervals, ci_fields), "",
        f"Bootstrap: {cfg['evaluation']['bootstrap_replicates']} повторов, "
        f"seed={cfg['evaluation']['bootstrap_seed']}.",
        "Ресэмплируются целые synthetic series внутри scenario×strength×noise, не отдельные месяцы. "
        "Интервалы условны на выбранные validation-параметры и не включают их неопределённость. "
        "Число конечных bootstrap оценок сохранено; отсутствующая локализация не заменяется нулём.", "",
        "## B. Сохранённый online benchmark E04a", "",
        "Доступ к данным: только текущий и прошлый префикс. Таблица скопирована из E04a, "
        "детекторы заново не запускались. median_delay ниже относится только к online alarms "
        "и не сравнивается с offline breakpoint offset как одна метрика задержки.", "", table(online_primary, ONLINE_COLUMNS), "",
        "Сохранённые online CI:", "", table(online_primary_ci, ["method", "metric", "estimate", "ci_low", "ci_high"]), "",
        "Рабочие точки online E04a сохранены без изменения:", "", "```json",
        json.dumps(online_selected, ensure_ascii=False, indent=2, sort_keys=True), "```", "",
        "Таблицы A/B имеют разный доступ к будущему. Общий победитель offline/online не выбирается; "
        "точная F1 offline не является оценкой раннего обнаружения или future warning.", "",
        "## Реальные retrospective candidate breakpoints", "",
        f"Доступны {status['n_real_series']} МО с residual stream E04a. "
        "Независимой реальной разметки нет: real precision/recall не вычисляются, точки "
        "не названы подтверждёнными экономическими шоками. Число кандидатов включает "
        "сохранённые warmup-точки; оценочные monitoring-точки отмечены отдельно в файле.",
        "; ".join(f"{method}: {int(point_counts.get(method, 0))} кандидатов" for method in METHODS) + ".", "",
        real_counts, "",
        table(status_counts, ["method", "status", "n_series"]), "",
        "real_breakpoints.csv сохраняет месяц, сегменты до/после, средние raw residuals, "
        "estimated_shift в рублях, SSE/gain/penalty. L2 ориентирован на смену среднего; "
        "его результаты slope/variance являются дополнительной проверкой, а не доказательством "
        "специализации на этих событиях.", "",
        "## Prefix stability: диагностический риск hindsight", "",
        "Каждый historical prefix сегментируется независимо по данным, доступным к его концу. "
        "Full-sample breakpoint используется только после этих fit для справочного сопоставления. "
        "Фиксированная близость ±1 календарный месяц, one-to-one; transient unmatched prefix "
        "кандидаты сохраняются. First prefix where detected — первый доступный префикс с "
        "сопоставимой оценкой, а не online alarm date, online detection delay или early warning. "
        "Допуск ±1 может связать оценки разных месяцев: это hindsight association, не предсказание будущей точки.", "",
        _prefix_table(prefix), "",
        "Статусы независимо рассчитанных префиксов:", "", table(prefix_status_counts, ["method", "status", "n_prefixes"]), "",
        "Первые три доступных residual-месяца ещё не дают четыре калибровочные ошибки: "
        "not_ready ожидаем и не называется ошибкой fit. Итоговая полная сегментация и "
        "готовые prefix fit имеют сохранённые диагностические статусы; failed в завершённом "
        "запуске отсутствует.", "",
        f"Сохранено {len(prefix_estimates)} prefix-оценок, включая промежуточные точки, "
        "не имеющие пары в полном ряду. Сдвиги даты, исчезновения и диапазон пересмотров "
        "сохранены в real_prefix_summary.csv; статус каждого независимого fit — в "
        "real_prefix_diagnostics.csv, траектории сопоставлений — в real_prefix_matches.csv.", "",
        "## Иллюстрации из сохранённых данных", "",
        f"Правило E04a сохранено для первых трёх МО: минимальные числовые ID {', '.join(real_ids)}; "
        f"синтетическая иллюстрация — первый level-ряд по series_id, {synthetic_id}. "
        "Выбор не зависит от точности/красоты сегментации. Truth на синтетическом графике "
        "служит только подписью; алгоритмы её не получали. Источники и SHA каждой PNG — plots_source.json.", ""]
    for name in plots["figures"]:
        relative = os.path.relpath(output / name, report_path.parent).replace("\\", "/")
        lines += [f"![{Path(name).stem}]({relative})", ""]
    lines += ["## Вывод именно по offline benchmark", "", _offline_conclusion(primary, intervals), "",
        "## Команды, среда и ограничения", "",
        f"ruptures {version}; полный pytest: {tests['summary']}, код {tests['exit_code']}. "
        f"Git HEAD `{manifest.get('git_commit')}`, dirty={manifest.get('has_uncommitted_changes')}; "
        "версии, seed, SHA кода/данных/артефактов и параметры сохранены. "
        "E01–E05d и E04a сохранены; forecasting model search остаётся закрыт.",
        f"Runtime этапов до построения графиков и отчёта: {status.get('runtime_seconds', np.nan):.2f} с. "
        f"Источник E04a проверен: {preflight.get('source_e04_verified')}; "
        f"generator неизменен: {preflight.get('source_generator_protocol_unchanged')}. "
        f"Завершённые сохранённые series diagnostics проверены; failed={status['n_failed']}. "
        "При ошибке сегментации runner прекращает этап, а не формирует successful-only метрики.", "",
        "```powershell", tests['command'], *manifest.get("run_commands", []), "```", "",
        f"L={lag} — прежнее допущение расходов: реальные публикации и vintages неизвестны. "
        "Всего 12 residual-месяцев, четыре ошибки калибровки и восемь monitoring-месяцев "
        "ограничивают надёжность. Изменение прогнозной ошибки может отражать адаптацию "
        "SeasonalNaiveYoY или состав/пересмотры расходов; оно не доказывает экономический шок. "
        "Одиночный выброс может менять последующие прогнозы, однако последующие точки этого "
        "контроля всё равно FP. Gaussian-синтетика не устанавливает качество на реальных МО.",
        "Full-sample сегментация использует будущее относительно найденной точки; prefix "
        "stability не превращает её в online detector. Уже просмотренный реальный holdout "
        "не становится новой независимой проверкой. Будущие изменения здесь не предсказываются.",
        "Следующий согласованный этап E06b — news/events; затем E07 — early warning. "
        "Эти этапы не выполнены; возврат к forecasting tuning не предусмотрен.", ""]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
