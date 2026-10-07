"""Saved-artifact report/plot tests on synthetic fixtures only; no detector fit."""
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from sberforecast.offline_report import PRIMARY_METRICS, _offline_conclusion, _primary_tables, plot_saved, write_report


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def metrics_fixture():
    records, intervals = [], []
    for method, score in (("PELT", .5), ("BinSeg", .4)):
        tp = int(360 * score)
        base = {"method": method, "scope": "primary_level", "n_series": 360, "n_events": 360,
                "tp": tp, "fp": 360-tp, "fn": 360-tp, "n_breakpoints": 360, "n_monitoring_months": 2880,
                "precision": score, "recall": score, "f1": score, "miss_rate": 1-score,
                "median_absolute_localisation_error": 1., "median_breakpoint_offset": 1.,
                "false_positives_per_12_months": (360-tp)*12/2880}
        records.append(base)
        for metric in PRIMARY_METRICS:
            value = base[metric]
            intervals.append({"method": method, "scope": "primary_level", "metric": metric,
                              "estimate": value, "ci_low": value-.05, "ci_high": value+.05,
                              "n_bootstrap": 500, "n_finite_bootstrap": 500, "bootstrap_unit": "series",
                              "bootstrap_seed": 300000, "stratified": True, "conditional_on_selected_parameters": True})
        for scenario in ("level_up", "level_down", "slope_up", "slope_down", "variance_up", "no_change", "outlier", "level_warmup", "level_end"):
            row = dict(base, scope="by_scenario", scenario=scenario, n_series=180, n_events=180)
            if scenario in ("no_change", "outlier", "level_warmup", "level_end"):
                row.update(n_events=0, recall=np.nan, miss_rate=np.nan,
                           median_absolute_localisation_error=np.nan, median_breakpoint_offset=np.nan)
            records.append(row)
    return pd.DataFrame(records), pd.DataFrame(intervals)


def artifacts(tmp_path):
    metrics, ci = metrics_fixture()
    metrics.to_csv(tmp_path / "synthetic_test_metrics.csv", index=False)
    ci.to_csv(tmp_path / "synthetic_test_ci.csv", index=False)
    online = pd.DataFrame([{"method": method, "scope": "primary_level", "n_events": 360,
                            "precision": .6, "recall": .3, "f1": .4, "missed_fraction": .7,
                            "median_delay": 1., "false_alarms_per_12_months": .4}
                           for method in ("CUSUM", "EWMA", "BOCPD")])
    online.to_csv(tmp_path / "online_reference_metrics.csv", index=False)
    online_ci = pd.DataFrame([{"method": method, "scope": "primary_level", "metric": "f1",
                               "estimate": .4, "ci_low": .3, "ci_high": .5} for method in online.method])
    online_ci.to_csv(tmp_path / "online_reference_ci.csv", index=False)
    save_json(tmp_path / "online_reference_selected.json", {method: {"selected": True, "parameters": {"threshold": 3}} for method in online.method})
    selected = {method: {"selected": True, "parameters": {"model": "l2", "min_size": 2, "jump": 1, "penalty": 4.},
                         "validation_control_far": {"no_change": .5, "outlier": .6}} for method in ("PELT", "BinSeg")}
    save_json(tmp_path / "selected_parameters.json", selected)
    pd.DataFrame([{"method": method, "penalty": 4., "feasible": True, "primary_f1": .4,
                   "no_change_far": .5, "outlier_far": .6} for method in selected]).to_csv(tmp_path / "validation_selection.csv", index=False)
    cfg = {"generator": {"validation_seed": 100000, "test_seed": 200000, "replicates_per_cell": 30},
           "preparation": {"release_lag_months": 0}, "evaluation": {"bootstrap_replicates": 500, "bootstrap_seed": 300000},
           "output_dir": "synthetic fixture only"}
    save_json(tmp_path / "run_status.json", {"complete": True, "n_validation_series": 1260, "n_test_series": 1260,
              "n_real_series": 3, "runtime_seconds": 123.456, "n_failed": 0})
    save_json(tmp_path / "run_manifest.json", {"complete": True, "config": cfg, "versions": {"ruptures": "1.1.10"},
              "git_commit": "synthetic fixture", "has_uncommitted_changes": True,
              "selected_parameters_sealed_before_test": True, "parameters_selected_on": "synthetic_validation_only",
              "labels_in_fit": False, "real_threshold_tuning": False, "forecasting_models_refitted": False, "early_warning": False,
              "tests": {"full_pytest": True, "exit_code": 0, "summary": "synthetic fixture, no actual run",
                        "command": "synthetic pytest command"}, "run_commands": ["synthetic experiment command"]})
    save_json(tmp_path / "preflight.json", {"status": "synthetic fixture", "source_e04_verified": True,
              "source_generator_protocol_unchanged": True})
    real, points, segments, prefix, summaries, diagnostics, prefix_diagnostics = [], [], [], [], [], [], []
    for municipality_id in ("37", "21", "1471", "25"):
        for index, period in enumerate(pd.period_range("2024-01", "2024-12", freq="M")):
            real.append({"municipality_id": municipality_id, "series_id": municipality_id,
                         "observation_period": str(period), "y_true": np.nan if municipality_id == "1471" else 100.+10*(index >= 5),
                         "y_pred": 100.})
        if municipality_id == "1471":
            continue
        for method in selected:
            points.append({"municipality_id": municipality_id, "method": method, "breakpoint_month": "2024-06",
                           "breakpoint_id": f"{municipality_id}-{method}-0", "phase": "monitoring", "estimated_shift": 10.})
            segments.extend([{"municipality_id": municipality_id, "method": method, "start_period": "2024-01",
                              "end_period": "2024-05", "mean_error": 0.},
                             {"municipality_id": municipality_id, "method": method, "start_period": "2024-06",
                              "end_period": "2024-12", "mean_error": 10.}])
            for end in pd.period_range("2024-07", "2024-12", freq="M"):
                prefix.append({"municipality_id": municipality_id, "method": method, "prefix_end_period": str(end),
                               "breakpoint_month": "2024-06", "is_temporary_without_full_match": False})
            summaries.append({"municipality_id": municipality_id, "method": method, "full_sample_breakpoint": "2024-06",
                              "first_prefix_where_detected": "2024-07", "first_breakpoint_estimate": "2024-06",
                              "n_matched_prefixes": 6, "n_date_revisions": 0, "n_presence_losses": 0,
                              "minimum_estimated_month": "2024-06", "maximum_estimated_month": "2024-06", "revision_span_months": 0})
            diagnostics.append({"municipality_id": municipality_id, "method": method, "status": "complete",
                                "n_breakpoints": 1, "n_evaluation_breakpoints": 1, "n_warmup_breakpoints": 0})
            for month in range(1, 13):
                prefix_diagnostics.append({"municipality_id": municipality_id, "method": method,
                                           "status": "not_ready" if month < 4 else "complete"})
    pd.DataFrame(real).to_csv(tmp_path / "real_residuals.csv.gz", index=False)
    for name, rows in (("real_breakpoints", points), ("real_segments", segments), ("real_prefix_estimates", prefix),
                       ("real_prefix_summary", summaries), ("real_series_diagnostics", diagnostics),
                       ("real_prefix_diagnostics", prefix_diagnostics)):
        pd.DataFrame(rows).to_csv(tmp_path / (name + ".csv"), index=False)
    series = pd.DataFrame([{"series_id": "test_002", "scenario": "level_down"},
                           {"series_id": "test_000", "scenario": "no_change"},
                           {"series_id": "test_001", "scenario": "level_up"}])
    series.to_csv(tmp_path / "synthetic_test_series.csv.gz", index=False)
    observations = pd.DataFrame([{"series_id": uid, "observation_period": str(period), "value": 100.+10*(period >= pd.Period("2024-06"))}
                                 for uid in series.series_id for period in pd.period_range("2023-01", "2024-12", freq="M")])
    observations.to_csv(tmp_path / "synthetic_test_observations.csv.gz", index=False)
    residuals = observations.loc[observations.observation_period.ge("2024-01")].rename(columns={"value": "y_true"}).copy()
    residuals["y_pred"] = 100.
    residuals["error"] = residuals.y_true-residuals.y_pred
    residuals.to_csv(tmp_path / "synthetic_test_residuals.csv.gz", index=False)
    pd.DataFrame([{"series_id": "test_001", "event_start_period": "2024-06"}]).to_csv(tmp_path / "synthetic_test_events.csv.gz", index=False)
    pd.DataFrame([{"series_id": "test_001", "method": method, "breakpoint_month": "2024-06"} for method in selected]).to_csv(tmp_path / "synthetic_test_breakpoints.csv.gz", index=False)
    return cfg


def test_plots_use_fixed_ids_saved_rows_and_input_hashes_without_detection(tmp_path):
    cfg = artifacts(tmp_path)
    record = plot_saved(tmp_path, cfg)
    assert record["illustration_selection"]["municipality_ids"] == ["21", "25", "37"]
    assert record["illustration_selection"]["synthetic_series_id"] == "test_001"
    assert record["detector_fit_performed"] is False and record["metrics_recomputed"] is False
    assert len(record["figures"]) == 8
    assert record["figures"]["figures/prefix_stability_21.png"]["selection"]["calendar_span"] == ["2024-01", "2024-12"]
    for name, provenance in record["figures"].items():
        image = tmp_path / name
        assert image.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert provenance["sha256"] == hashlib.sha256(image.read_bytes()).hexdigest()
        for source, digest in provenance["source_sha256"].items():
            assert digest == hashlib.sha256((tmp_path/source).read_bytes()).hexdigest()
    before = {name: (tmp_path/name).stat().st_mtime_ns for name in record["figures"]}
    assert plot_saved(tmp_path, cfg) == record
    assert before == {name: (tmp_path/name).stat().st_mtime_ns for name in record["figures"]}


def test_existing_plot_sources_cannot_change_silently(tmp_path):
    cfg = artifacts(tmp_path)
    plot_saved(tmp_path, cfg)
    path = tmp_path / "synthetic_test_metrics.csv"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Источник"):
        plot_saved(tmp_path, cfg)


@pytest.mark.parametrize("damage", ["point", "repeated", "bootstrap_months", "reversed_interval", "different_medians"])
def test_saved_metric_and_whole_series_ci_consistency_guard(tmp_path, damage):
    artifacts(tmp_path)
    path = tmp_path / "synthetic_test_ci.csv"
    ci = pd.read_csv(path)
    if damage == "point":
        ci.loc[0, "estimate"] += .1
    elif damage == "repeated":
        ci = pd.concat([ci, ci.iloc[[0]]], ignore_index=True)
    elif damage == "bootstrap_months":
        ci["bootstrap_unit"] = "month"
    elif damage == "reversed_interval":
        ci.loc[0, ["ci_low", "ci_high"]] = [.9, .1]
    else:
        metrics_path = tmp_path / "synthetic_test_metrics.csv"
        metrics = pd.read_csv(metrics_path)
        metrics.loc[metrics.scope.eq("primary_level"), "median_breakpoint_offset"] = 2.
        metrics.to_csv(metrics_path, index=False)
    ci.to_csv(path, index=False)
    with pytest.raises(ValueError):
        _primary_tables(tmp_path)


def test_report_is_saved_only_separates_future_access_and_online_delay(tmp_path):
    cfg = artifacts(tmp_path)
    plot_saved(tmp_path, cfg)
    report = tmp_path / "report.md"
    write_report(tmp_path, report, cfg)
    text = report.read_text(encoding="utf-8")
    for phrase in ("A. Offline", "B. Сохранённый online", "весь известный", "median_breakpoint_offset",
                   "не online detection delay", "±1 календарный", "не является ранним предупреждением",
                   "First prefix where detected", "360", "1260", "PELT, 0.500000", "пересекаются",
                   "real precision/recall не вычисляются", "первые четыре", "целые synthetic series",
                   "Test создан после", "L=0", "vintages неизвестны", "E06b", "E07", "123.46 с."):
        assert phrase.lower() in text.lower()
    assert "не новый независимый слепой" in text
    assert "по определению этого окна" in text
    assert "monitoring_candidates" in text and "warmup_candidates" in text
    assert "not_ready ожидаем" in text and "failed=0" in text
    assert "synthetic pytest command" in text and "synthetic experiment command" in text
    assert "figures/prefix_stability_21.png" in text
    assert "Пороговое правило" not in text  # No new scoring is silently introduced.


@pytest.mark.parametrize("damage", ["unfinished", "manifest_unfinished", "pytest_failed", "config_changed", "no_version", "unsealed", "test_tuning", "truth_in_fit"])
def test_report_requires_completed_matching_tested_run(tmp_path, damage):
    cfg = artifacts(tmp_path)
    name = "run_status.json" if damage == "unfinished" else "run_manifest.json"
    path = tmp_path / name
    record = json.loads(path.read_text(encoding="utf-8"))
    if damage in ("unfinished", "manifest_unfinished"):
        record["complete"] = False
    elif damage == "pytest_failed":
        record["tests"]["exit_code"] = 1
    elif damage == "config_changed":
        record["config"]["generator"]["replicates_per_cell"] = 1
    elif damage == "no_version":
        record["versions"] = {}
    elif damage == "unsealed":
        record["selected_parameters_sealed_before_test"] = False
    elif damage == "test_tuning":
        record["parameters_selected_on"] = "synthetic_test"
    else:
        record["labels_in_fit"] = True
    save_json(path, record)
    if damage == "no_version":
        plot_saved(tmp_path, cfg)
    with pytest.raises(ValueError):
        write_report(tmp_path, tmp_path / "report.md", cfg)
    assert not (tmp_path / "report.md").exists()


def test_existing_report_is_preserved_without_loading_results(tmp_path):
    path = tmp_path / "report.md"
    path.write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError, match="не перезаписывается"):
        write_report(tmp_path, path, {})
    assert path.read_text(encoding="utf-8") == "existing"


def test_offline_conclusion_handles_tie_and_no_selected_method():
    metrics, ci = metrics_fixture()
    primary = metrics.loc[metrics.scope.eq("primary_level")].copy()
    primary["f1"] = .5
    assert "PELT, BinSeg" in _offline_conclusion(primary, ci)
    assert "Нет выбранного" in _offline_conclusion(primary.iloc[:0], ci.iloc[:0])
