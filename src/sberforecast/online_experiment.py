"""E04a: независимые синтетические выборки и неизменные сохранённые ошибки E01."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import importlib.metadata
import itertools
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import yaml

from .data import eligibility, get_prefix, load_data, make_panel, period_end, sha256_file
from .models import baseline_predict
from .online_detection import run_stream
from .online_evaluation import aggregate_metrics, bootstrap_metrics, evaluate_series
from .online_synthetic import generate_benchmark

METHODS = ("CUSUM", "EWMA", "BOCPD")


def save_json(path: Path, value: dict | list) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def safe_path(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Путь должен оставаться внутри проекта.")
    return path


def validate_output_path(root: Path, output: Path) -> None:
    protected_names = ("baseline_v1", "baseline_local", "prophet_comparison_v1", "catboost_direct_v1",
                       "e02_data_audit", "chronos_zero_shot_v1", "chronos_zero_shot_smoke_v1")
    protected = [root / "outputs" / name for name in protected_names]
    if (output == root / "outputs" or not output.is_relative_to(root / "outputs") or
            any(output == path or output.is_relative_to(path) for path in protected)):
        raise ValueError("Нужен отдельный каталог E04a вне прежних результатов.")


def candidates(grid: dict) -> list[dict]:
    keys = sorted(grid)
    if not keys or any(not isinstance(grid[key], list) or not grid[key] for key in keys):
        raise ValueError("Параметры сетки должны быть непустыми списками.")
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[key] for key in keys))]


def select_candidate(rows: pd.DataFrame, budget: float) -> pd.Series | None:
    """Только validation: бюджет для каждого контроля, затем заранее заданный порядок."""
    feasible = rows.loc[rows.max_control_far.le(budget) & rows.primary_f1.notna()].copy()
    if feasible.empty:
        return None
    feasible["delay_order"] = feasible.primary_median_delay.fillna(np.inf)
    return feasible.sort_values(
        ["primary_f1", "primary_recall", "max_control_far", "delay_order", "candidate_id"],
        ascending=[False, False, True, True, True], kind="stable",
    ).iloc[0]


def prepare_real(root: Path, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    source = safe_path(root, cfg["source"]["e01_dir"])
    old = yaml.safe_load((source / "config_resolved.yaml").read_text(encoding="utf-8"))
    manifest = json.loads((source / "run_manifest.json").read_text(encoding="utf-8"))
    data_path = safe_path(root, cfg["source"]["data_path"])
    if (cfg["preparation"]["release_lag_months"] != old["data"]["release_lag_months"] or
            cfg["source"]["category"] != old["data"]["category"] or
            cfg["source"]["forecast_model"] != "SeasonalNaiveYoY" or cfg["source"]["horizon"] != 1):
        raise ValueError("Вход E04a должен соответствовать выбранному протоколу E01/h=1.")
    if sha256_file(data_path) != manifest["data_sha256"]:
        raise ValueError("Исходные данные E01 изменились.")
    for name in ("yearly_growth_window", "yearly_growth_bounds"):
        if cfg["forecast"][name] != old["models"][name]:
            raise ValueError("Синтетический прогноз отличается от алгоритма E01.")
    for name in ("min_history_observations", "max_staleness_months"):
        if cfg["forecast"][name] != old["backtest"][name]:
            raise ValueError("Правила допуска синтетического прогноза отличаются от E01.")
    sample = [str(uid) for uid in json.loads((source / "sample_ids.json").read_text(encoding="utf-8"))]
    if len(sample) != 64 or len(set(sample)) != 64:
        raise ValueError("Ожидались 64 уникальных исходных ID E01.")
    panel = make_panel(load_data(data_path, cfg["source"]["category"]))
    saved = pd.read_csv(source / "predictions.csv.gz", dtype={"municipality_id": str})
    saved = saved.loc[saved.model.eq("SeasonalNaiveYoY") & saved.horizon.eq(1)].copy()
    if not saved.status.eq("native").all() or not np.isfinite(saved.y_pred).all():
        raise ValueError("Ошибки/резервы входного прогноза требуют явного отдельного решения.")
    saved["observation_period"] = pd.PeriodIndex(saved.target_period, freq="M").astype(str)
    key = ["municipality_id", "observation_period"]
    if saved.duplicated(key).any() or set(saved.municipality_id) != set(sample):
        raise ValueError("Дубли либо другой набор ID в сохранённых одномесячных прогнозах.")
    # Сверяем реальные ключи допуска и сам причинный алгоритм; новых моделей не обучаем.
    expected_keys, max_difference = set(), 0.0
    origins = pd.period_range(old["backtest"]["first_origin"], old["backtest"]["last_origin"], freq="M")
    lag = cfg["preparation"]["release_lag_months"]
    for origin in origins:
        prefix = get_prefix(panel, origin, lag)
        cohort = eligibility(prefix, old["backtest"]["min_history_observations"], old["backtest"]["max_staleness_months"])
        ids = sorted(set(cohort.loc[cohort.eligible, "municipality_id"]) & set(sample))
        forecast, _ = baseline_predict(prefix[ids], lag + 1, "SeasonalNaiveYoY", cfg["forecast"])
        target = str(origin + 1)
        part = saved.loc[saved.observation_period.eq(target)].set_index("municipality_id")
        if set(part.index) != set(ids):
            raise ValueError(f"Ключи сохранённых h=1 прогнозов отличаются на {origin}.")
        if not part.forecast_origin.eq(str(period_end(origin).date())).all() or not part.history_cutoff.eq(str(period_end(origin - lag).date())).all():
            raise ValueError("Даты выпуска/cutoff E01 не соответствуют календарю.")
        truth = panel.loc[(origin + 1).to_timestamp(), ids].to_numpy()
        recorded_truth = part.loc[ids].y_true.to_numpy()
        if not ((truth == recorded_truth) | (np.isnan(truth) & np.isnan(recorded_truth))).all():
            raise ValueError("Сохранённый факт E01 отличается от входных данных.")
        difference = float(np.max(np.abs(forecast[-1] - part.loc[ids].y_pred.to_numpy())))
        max_difference = max(max_difference, difference)
        expected_keys.update((uid, target) for uid in ids)
    if set(saved[key].itertuples(index=False, name=None)) != expected_keys or max_difference > 1e-8:
        raise ValueError("Фактические ключи либо SeasonalNaiveYoY не воспроизведены.")
    calendar = pd.period_range(origins[0] + 1, origins[-1] + 1, freq="M").astype(str)
    full = pd.MultiIndex.from_product([sample, calendar], names=key).to_frame(index=False)
    residuals = full.merge(saved, on=key, how="left", validate="one_to_one")
    residuals["source_prediction_present"] = residuals.y_pred.notna()
    residuals = residuals.rename(columns={"municipality_id": "series_id"})
    residuals["source_forecast_max_abs_difference"] = max_difference
    info = pd.DataFrame({"series_id": sample, "scenario": "real_unlabelled", "strength": 0.0,
                         "noise_fraction": np.nan, "split": "real_diagnostic", "release_lag_months": lag,
                         "observation_start_period": calendar[0], "observation_end_period": calendar[-1]})
    observations = panel[sample].rename_axis("ds").reset_index().melt(id_vars="ds", var_name="series_id", value_name="value")
    observations["observation_period"] = observations.ds.dt.to_period("M").astype(str)
    return residuals, info, observations.drop(columns="ds"), sample


def score_and_evaluate(residuals: pd.DataFrame, events: pd.DataFrame, info: pd.DataFrame,
                       method: str, params: dict, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    signals = run_stream(residuals, method, params, cfg["preparation"])
    per_series, matches, alarms = evaluate_series(signals, events, info, cfg["evaluation"]["detection_window_months"])
    return signals, per_series, matches, alarms


def tune(validation: tuple, cfg: dict, output: Path) -> dict:
    residuals, events, info, _ = validation
    selected, records = {}, []
    for method in METHODS:
        method_rows = []
        for index, params in enumerate(candidates(cfg["detectors"][method])):
            _, per_series, _, _ = score_and_evaluate(residuals, events, info, method, params, cfg)
            primary = per_series.loc[per_series.scenario.isin(cfg["evaluation"]["primary_scenarios"])]
            primary_metrics = aggregate_metrics(primary, []).iloc[0]
            control_fars = {}
            for control in cfg["evaluation"]["budget_controls"]:
                control_metrics = aggregate_metrics(per_series.loc[per_series.scenario.eq(control)], []).iloc[0]
                control_fars[control] = float(control_metrics.false_alarms_per_12_months)
            row = {"method": method, "candidate_id": f"{method}_{index:03d}", "parameters": json.dumps(params, sort_keys=True),
                   "primary_f1": float(primary_metrics.f1), "primary_recall": float(primary_metrics.recall),
                   "primary_median_delay": float(primary_metrics.median_delay),
                   "max_control_far": max(control_fars.values()),
                   **{name + "_far": value for name, value in control_fars.items()}}
            method_rows.append(row)
            print(f"validation {method} {index+1}: F1={row['primary_f1']:.3f}, max control FAR={row['max_control_far']:.3f}", flush=True)
        table = pd.DataFrame(method_rows)
        winner = select_candidate(table, cfg["evaluation"]["false_alarm_budget_per_12_months"])
        if winner is None:
            selected[method] = {"selected": False, "reason": "no_candidate_within_fixed_control_budget"}
        else:
            selected[method] = {"selected": True, "candidate_id": winner.candidate_id,
                                "parameters": json.loads(winner.parameters),
                                "validation_primary_f1": float(winner.primary_f1),
                                "validation_control_far": {name: float(winner[name + "_far"]) for name in cfg["evaluation"]["budget_controls"]}}
        table["selected"] = table.candidate_id.eq(None if winner is None else winner.candidate_id)
        records.extend(table.to_dict("records"))
    pd.DataFrame(records).to_csv(output / "validation_candidates.csv", index=False)
    save_json(output / "selected_parameters.json", selected)
    return selected


def save_synthetic(split: str, generated: tuple, selected: dict, cfg: dict, output: Path) -> None:
    residuals, events, info, observations = generated
    for name, frame in (("residuals", residuals), ("events", events), ("series", info), ("observations", observations)):
        frame.to_csv(output / f"synthetic_{split}_{name}.csv.gz", index=False)
    all_signals, all_per_series, all_matches, all_alarms, all_metrics, all_ci = [], [], [], [], [], []
    for method, selection in selected.items():
        if not selection["selected"]:
            continue
        signals, per_series, matches, alarms = score_and_evaluate(residuals, events, info, method, selection["parameters"], cfg)
        for frame, container in ((signals, all_signals), (per_series, all_per_series), (matches, all_matches), (alarms, all_alarms)):
            container.append(frame.assign(method=method))
        metrics = aggregate_metrics(per_series, ["scenario", "strength", "noise_fraction"])
        all_metrics.append(metrics.assign(method=method, scope="by_scenario_strength_noise"))
        by_scenario = aggregate_metrics(per_series, ["scenario"])
        all_metrics.append(by_scenario.assign(method=method, scope="by_scenario"))
        primary = per_series.loc[per_series.scenario.isin(cfg["evaluation"]["primary_scenarios"])]
        all_metrics.append(aggregate_metrics(primary, []).assign(method=method, scope="primary_level", scenario="level_up_and_down"))
        if split == "test":
            bootstrap_settings = cfg["evaluation"]
            for scope, data, grouping in (("primary_level", primary, []), ("by_scenario", per_series, ["scenario"]),
                                          ("by_scenario_strength_noise", per_series, ["scenario", "strength", "noise_fraction"])):
                ci = bootstrap_metrics(data, grouping, bootstrap_settings["bootstrap_replicates"], bootstrap_settings["bootstrap_seed"])
                all_ci.append(ci.assign(method=method, scope=scope))
    for name, frames in (("signals", all_signals), ("per_series", all_per_series), ("event_matches", all_matches),
                          ("alarm_matches", all_alarms), ("metrics", all_metrics), ("uncertainty", all_ci)):
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(output / f"synthetic_{split}_{name}.csv.gz", index=False)


def real_diagnostics(real: tuple, selected: dict, cfg: dict, output: Path) -> dict:
    residuals, info, observations, sample = real
    residuals.to_csv(output / "real_residuals.csv.gz", index=False)
    info.to_csv(output / "real_series.csv", index=False)
    save_json(output / "sample_ids.json", sample)
    coverage_records = []
    for uid, part in residuals.groupby("series_id", sort=False):
        n_finite = int((np.isfinite(part.y_true) & np.isfinite(part.y_pred)).sum())
        n_warmup = min(n_finite, cfg["preparation"]["warmup_observations"])
        coverage_records.append({"series_id": uid, "n_calendar_months": len(part),
                                 "n_monitoring_months": n_finite-n_warmup, "n_warmup_months": n_warmup,
                                 "n_missing_months": len(part)-n_finite})
    coverage = pd.DataFrame(coverage_records)
    coverage.to_csv(output / "real_coverage.csv", index=False)
    frames = [run_stream(residuals, method, selection["parameters"], cfg["preparation"]).assign(method=method)
              for method, selection in selected.items() if selection["selected"]]
    if not frames:
        status = {"selected_methods": [], "reason": "no_feasible_detector_on_validation", "real_precision_recall_computed": False,
                  "n_sample_municipalities": len(sample), "n_municipalities_with_monitoring": int(coverage.n_monitoring_months.gt(0).sum()),
                  "n_observed_monitoring_months_per_method": int(coverage.n_monitoring_months.sum())}
        save_json(output / "real_diagnostics.json", status)
        return status
    signals = pd.concat(frames, ignore_index=True)
    signals.to_csv(output / "real_signals.csv.gz", index=False)
    signals.loc[signals.is_alarm].to_csv(output / "real_alarms.csv", index=False)
    reference = frames[0]
    methods = signals.method.drop_duplicates().tolist()
    sets = {method: set(signals.loc[signals.method.eq(method) & signals.is_alarm, ["series_id", "signal_date"]].itertuples(index=False, name=None)) for method in methods}
    agreement = []
    for first, second in itertools.combinations(methods, 2):
        a, b = sets[first], sets[second]
        agreement.append({"method_a": first, "method_b": second, "same_month_alarm_intersection": len(a & b),
                          "alarm_union": len(a | b), "a_only": len(a - b), "b_only": len(b - a),
                          "jaccard": len(a & b)/len(a | b) if a | b else np.nan})
    agreement_columns = ["method_a", "method_b", "same_month_alarm_intersection", "alarm_union", "a_only", "b_only", "jaccard"]
    pd.DataFrame(agreement, columns=agreement_columns).to_csv(output / "real_agreement.csv", index=False)
    monitoring = reference.loc[reference.phase.eq("monitoring")]
    status = {"n_sample_municipalities": len(sample), "n_municipalities_with_monitoring": int(coverage.n_monitoring_months.gt(0).sum()),
              "n_observed_monitoring_months_per_method": len(monitoring),
              "n_warmup_months_per_method": int(reference.phase.eq("warmup").sum()),
              "n_missing_months_per_method": int(reference.phase.eq("missing").sum()),
              "monitoring_observation_first_month": monitoring.observation_period.min() if len(monitoring) else None,
              "monitoring_observation_last_month": monitoring.observation_period.max() if len(monitoring) else None,
              "monitoring_first_availability": monitoring.availability_date.min() if len(monitoring) else None,
              "monitoring_last_availability": monitoring.availability_date.max() if len(monitoring) else None,
              "alarms_by_method": {method: len(value) for method, value in sets.items()},
              "n_same_month_alarms_all_selected_methods": len(set.intersection(*sets.values())),
              "real_precision_recall_computed": False, "independent_real_labels_available": False,
              "early_warning": False, "forecast_selection_timing": cfg["source"]["selection_timing"]}
    ids = sorted(sample, key=int)[:cfg["illustrations"]["n_smallest"]]
    unavailable = sorted(coverage.loc[coverage.n_monitoring_months.eq(0), "series_id"], key=int)
    if unavailable and unavailable[0] not in ids:
        ids.append(unavailable[0])
    status["illustration_ids"] = ids
    save_json(output / "real_diagnostics.json", status)
    plot_examples(observations, residuals, signals, ids, output)
    return status


def plot_examples(observations: pd.DataFrame, residuals: pd.DataFrame, signals: pd.DataFrame,
                  ids: list[str], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    colors = {"CUSUM": "tab:red", "EWMA": "tab:green", "BOCPD": "tab:purple"}
    for uid in ids:
        obs = observations.loc[observations.series_id.eq(uid)].sort_values("observation_period")
        pred = residuals.loc[residuals.series_id.eq(uid)].sort_values("observation_period")
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        axes[0].plot(pd.to_datetime(obs.observation_period), obs.value, "o-", label="Observed expenses")
        axes[0].plot(pd.to_datetime(pred.observation_period), pred.y_pred, "o--", label="E01 SeasonalNaiveYoY h=1")
        axes[0].set_ylabel("Nominal RUB"); axes[0].legend()
        axes[1].plot(pd.to_datetime(pred.observation_period), pred.y_true-pred.y_pred, "o-", label="Residual: true - forecast")
        axes[1].axhline(0, color="gray", linewidth=.7); axes[1].set_ylabel("Residual RUB")
        for method, part in signals.loc[signals.series_id.eq(uid)].groupby("method"):
            alarms = part.loc[part.is_alarm]
            axes[2].plot(pd.to_datetime(part.availability_date), part.score, ".-", color=colors[method], label=method)
            for signal_date in alarms.signal_date:
                for ax in axes:
                    ax.axvline(pd.Timestamp(signal_date), color=colors[method], alpha=.5, linestyle=":")
        axes[2].set_ylabel("Detector score"); axes[2].legend()
        axes[0].set_title(f"Municipality {uid}: diagnostic signals, no verified shock labels")
        fig.autofmt_xdate(); fig.tight_layout()
        fig.savefig(figures / f"municipality_{uid}.png", dpi=130)
        plt.close(fig)


def run_experiment(root: Path, config_path: Path, *, smoke: bool, command: str) -> dict:
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    protected_record = json.loads((root / "outputs/e04a_checks/initial_preservation.json").read_text(encoding="utf-8"))
    protected = protected_record["hashes"]
    if not all(sha256_file(root / name) == digest for name, digest in protected.items()):
        raise RuntimeError("Защищённые исходные файлы изменились до запуска.")
    files = [root / "src/sberforecast" / f"online_{name}.py" for name in ("detection", "synthetic", "evaluation", "experiment", "report")]
    files += [root / "scripts/run_online_detection.py"]
    signature = {"config_sha256": sha256_file(config_path), "code_sha256": {str(path.relative_to(root)).replace("\\", "/"): sha256_file(path) for path in files}}
    validation_path = root / "outputs/e04a_checks/test_validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if not validation["full_pytest_passed"] or validation["signature"] != signature:
        raise RuntimeError("Нужен успешный полный pytest актуального кода и конфигурации.")
    output = safe_path(root, cfg["smoke_output_dir"] if smoke else cfg["output_dir"])
    validate_output_path(root, output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Сохранённый E04a не перезаписывается: задайте отдельный output_dir.")
    if not smoke:
        smoke_manifest = json.loads((safe_path(root, cfg["smoke_output_dir"]) / "run_manifest.json").read_text(encoding="utf-8"))
        if not smoke_manifest["complete"] or smoke_manifest["signature"] != signature:
            raise RuntimeError("Сначала нужен успешный минимальный запуск того же кода и конфигурации.")
    real = prepare_real(root, cfg)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config_resolved.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "pandas", "matplotlib", "PyYAML", "pytest")}
    manifest = {"experiment": cfg["experiment"], "mode": "smoke" if smoke else "full", "complete": False, "started_at_utc": started_at,
                "signature": signature, "config": cfg, "command": command, "runtime_argv": sys.orig_argv,
                "python": sys.version, "python_executable": sys.executable, "platform": platform.platform(),
                "versions": versions, "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                "git_status": subprocess.check_output(["git", "status", "--short"], cwd=root, text=True).splitlines(),
                "protected_sha256": protected, "test_validation_sha256": sha256_file(validation_path),
                "test_command": validation["command"], "publication_dates_verified": False,
                "real_independent_blind_test": False, "early_warning": False,
                "parameters_selected_on": "synthetic_validation_only", "real_threshold_tuning": False}
    manifest["has_uncommitted_changes"] = bool(manifest["git_status"])
    save_json(output / "run_manifest.json", manifest)
    try:
        generator = copy.deepcopy(cfg["generator"])
        if smoke:
            generator["replicates_per_cell"] = 1
            manifest["smoke_generator_override"] = {"replicates_per_cell": 1}
        val = generate_benchmark(generator, "validation", cfg["forecast"], cfg["preparation"]["release_lag_months"])
        if smoke:
            selected = {method: {"selected": True, "parameters": candidates(cfg["detectors"][method])[0]} for method in METHODS}
            save_json(output / "selected_parameters.json", selected)
            for method in METHODS:
                signals, per_series, _, _ = score_and_evaluate(*val[:3], method, selected[method]["parameters"], cfg)
                if len(signals) != len(val[0]) or not len(per_series) or not np.isfinite(signals.loc[signals.phase.eq("monitoring"), "score"]).all():
                    raise RuntimeError("Минимальная проверка сигналов/метрик не прошла.")
            real_status = real_diagnostics(real, selected, cfg, output)
            status = {"complete": True, "smoke_passed": True, "n_synthetic_series": len(val[2]),
                      "n_real_sample": len(real[3]), "n_real_monitoring_months": real_status["n_observed_monitoring_months_per_method"]}
        else:
            selected = tune(val, cfg, output)
            # Test создаётся и просматривается только после фиксации выбранных параметров.
            save_synthetic("validation", val, selected, cfg, output)
            test = generate_benchmark(generator, "test", cfg["forecast"], cfg["preparation"]["release_lag_months"])
            if set(val[2].series_id) & set(test[2].series_id) or set(val[2].seed) & set(test[2].seed):
                raise RuntimeError("Синтетические validation/test пересекаются.")
            save_synthetic("test", test, selected, cfg, output)
            real_status = real_diagnostics(real, selected, cfg, output)
            status = {"complete": True, "n_validation_series": len(val[2]), "n_test_series": len(test[2]),
                      "selected_methods": [method for method, item in selected.items() if item["selected"]],
                      "n_real_sample": len(real[3]), "real_diagnostics_complete": True}
        if not all(sha256_file(root / name) == digest for name, digest in protected.items()):
            raise RuntimeError("Защищённые файлы изменились во время запуска.")
        status["elapsed_seconds"] = time.perf_counter() - started
        save_json(output / "run_status.json", status)
        manifest.update(complete=True, elapsed_seconds=status["elapsed_seconds"], protected_files_unchanged=True,
                        completed_at_utc=datetime.now(timezone.utc).isoformat())
        save_json(output / "run_manifest.json", manifest)
        if not smoke:
            from .online_report import write_report
            write_report(output, safe_path(root, cfg["report_path"]))
        artifact_hashes = {path.relative_to(root).as_posix(): sha256_file(path) for path in output.rglob("*") if path.is_file() and path.name != "run_manifest.json"}
        if not smoke:
            artifact_hashes[cfg["report_path"]] = sha256_file(safe_path(root, cfg["report_path"]))
        manifest["artifact_sha256"] = artifact_hashes
        save_json(output / "run_manifest.json", manifest)
        print(json.dumps(status, ensure_ascii=False), flush=True)
        return status
    except Exception as exc:
        manifest.update(complete=False, error=repr(exc), elapsed_seconds=time.perf_counter()-started)
        save_json(output / "run_manifest.json", manifest)
        save_json(output / "run_status.json", {"complete": False, "error": repr(exc)})
        raise
