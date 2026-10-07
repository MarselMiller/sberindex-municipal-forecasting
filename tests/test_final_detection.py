"""Final detection arithmetic uses only synthetic saved artifacts."""
import hashlib
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from sberforecast.final_detection import (
    OFFLINE, ONLINE, _prefix_reconstruction, _reconstruct, collect_detection,
)


def _calendar(method="CUSUM"):
    return pd.DataFrame([
        dict(series_id=uid, method=method, observation_period=str(month),
             availability_date=str(month.end_time.date()), phase="warmup" if month.month < 5 else "monitoring",
             is_alarm=uid == "A" and month.month in (6, 7),
             signal_date=str(month.end_time.date()) if uid == "A" and month.month in (6, 7) else np.nan)
        for uid in ("A", "B") for month in pd.period_range("2024-01", "2024-12", freq="M")
    ])


def _info():
    return pd.DataFrame([dict(series_id="A", scenario="level_up", observation_start_period="2024-01", observation_end_period="2024-12"),
                         dict(series_id="B", scenario="level_down", observation_start_period="2024-01", observation_end_period="2024-12")])


def _events():
    return pd.DataFrame([dict(series_id="A", event_id="AE", event_start_period="2024-05"),
                         dict(series_id="B", event_id="BE", event_start_period="2024-06")])


def test_calendar_reconstruction_counts_repeated_alarm_as_false_positive_and_miss():
    calendar = _calendar()
    quality, matches, points = _reconstruct(calendar, calendar.loc[calendar.is_alarm], _events(), _info(), family="online")
    assert quality["tp"] == quality["fp"] == quality["fn"] == 1
    assert quality["n_events"] == 2 and quality["n_monitoring_months"] == 16
    assert quality["precision"] == quality["recall"] == quality["f1"] == quality["miss_rate"] == .5
    assert quality["median_detection_delay"] == 1
    assert np.isnan(quality["median_absolute_localisation_error"])
    assert quality["false_positives_per_12_months"] == .75
    assert matches.status.tolist() == ["detected", "missed"]
    assert points.status.tolist() == ["true_positive", "false_positive"]


def test_offline_uses_observation_localisation_not_full_analysis_availability_and_excludes_warmup():
    stream = _calendar("PELT")
    points = pd.DataFrame([dict(series_id="A", breakpoint_id="p0", breakpoint_month="2024-03", analyzed_available_date="2024-12-31"),
                           dict(series_id="A", breakpoint_id="p1", breakpoint_month="2024-06", analyzed_available_date="2024-12-31"),
                           dict(series_id="A", breakpoint_id="p2", breakpoint_month="2024-07", analyzed_available_date="2024-12-31")])
    quality, _, classified = _reconstruct(stream, points, _events(), _info(), family="offline")
    assert quality["n_points"] == 3 and quality["n_points_excluded"] == 1
    assert quality["median_absolute_localisation_error"] == 1
    assert np.isnan(quality["median_detection_delay"])
    assert classified.status.tolist() == ["outside_evaluation_warmup", "true_positive", "false_positive"]


def test_online_issue_date_delay_and_missing_months_do_not_compress_calendar():
    calendar = _calendar()
    calendar = calendar.loc[~(calendar.series_id.eq("A") & calendar.observation_period.eq("2024-06"))].copy()
    points = calendar.loc[calendar.is_alarm].copy()
    points["signal_date"] = "2024-08-31"
    quality, _, _ = _reconstruct(calendar, points, _events(), _info(), family="online")
    assert quality["n_monitoring_months"] == 15
    assert quality["median_detection_delay"] == 3  # May to August, not compressed rows.
    assert quality["tp"] == 1 and quality["fp"] == 0


def test_censored_and_before_monitoring_events_retained_but_not_negative_or_missed():
    events = _events()
    events.loc[events.series_id.eq("A"), "event_start_period"] = "2024-03"
    events.loc[events.series_id.eq("B"), "event_start_period"] = "2024-11"
    calendar = _calendar()
    quality, matches, classified = _reconstruct(calendar, calendar.loc[calendar.is_alarm], events, _info(), family="online")
    assert quality["n_events"] == 0 and quality["n_events_excluded"] == 2
    assert np.isnan(quality["recall"]) and np.isnan(quality["miss_rate"])
    assert matches.eligibility.tolist() == ["before_monitoring", "partial_window"]
    assert classified.status.tolist() == ["excluded_censored_event", "false_positive"]


def test_duplicate_calendar_and_invalid_family_are_rejected():
    calendar = _calendar()
    with pytest.raises(ValueError, match="Duplicate detection"):
        _reconstruct(pd.concat([calendar, calendar.iloc[:1]]), calendar.loc[calendar.is_alarm], _events(), _info(), family="online")
    with pytest.raises(ValueError, match="family"):
        _reconstruct(calendar, calendar.loc[calendar.is_alarm], _events(), _info(), family="early_warning")


def _prefix():
    return pd.DataFrame([
        dict(method="PELT", full_breakpoint_id="AP", full_sample_breakpoint="2024-06",
             prefix_end_period=period, matched=matched, comparison_available=ready,
             breakpoint_estimate_by_prefix=estimate, exact_date_match=exact)
        for period, matched, ready, estimate, exact in [
            ("2024-03", False, False, np.nan, False),
            ("2024-06", True, True, "2024-05", False),
            ("2024-07", False, True, np.nan, False),
            ("2024-08", True, True, "2024-06", True),
            ("2024-09", True, True, "2024-05", False),
        ]
    ])


def test_prefix_revision_and_presence_loss_arithmetic_is_hindsight_not_alarm_delay():
    row = _prefix_reconstruction(_prefix()).iloc[0]
    assert row.n_prefixes == 5 and row.n_matched_prefixes == 3 and row.n_exact_prefixes == 1
    assert row.n_date_revisions == 2 and row.n_presence_losses == 1
    assert row.first_prefix_where_detected == "2024-06"
    assert row.first_breakpoint_estimate == "2024-05"
    assert row.first_prefix_offset_months == 0 and row.revision_span_months == 1


def _save(root, path, frame):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)


@pytest.fixture
def saved_project(tmp_path):
    root = tmp_path
    for family, methods in [("online", ONLINE), ("offline", OFFLINE)]:
        directory = f"outputs/{family}_detection_v1"
        config = dict(output_dir=directory, report_path=f"reports/{family}.md",
                      evaluation=dict(detection_window_months=3),
                      offline_analysis=dict(prefix_matching_tolerance_months=1))
        target = root / f"configs/{family}_detection.yaml"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(yaml.safe_dump(config), encoding="utf-8")
        _save(root, directory + "/synthetic_test_series.csv.gz", _info())
        _save(root, directory + "/synthetic_test_events.csv.gz", _events())
        calendars, matches, classified, metrics, raw_points = [], [], [], [], []
        for method in methods:
            calendar = _calendar(method)
            calendars.append(calendar)
            offset_name = "delay_months" if family == "online" else "breakpoint_offset_months"
            id_name = "alarm_id" if family == "online" else "breakpoint_id"
            linked_name = "matched_alarm_id" if family == "online" else "matched_breakpoint_id"
            first_id = "A:0" if family == "online" else f"A_{method}_1"
            matches.extend([dict(series_id="A", method=method, event_id="AE", eligibility="full_window",
                                 status="detected", **{linked_name: first_id, offset_name: 1}),
                            dict(series_id="B", method=method, event_id="BE", eligibility="full_window",
                                 status="missed", **{linked_name: np.nan, offset_name: np.nan})])
            for number, month, status, offset in [(1, "2024-06", "true_positive", 1), (2, "2024-07", "false_positive", np.nan)]:
                identifier = f"A:{number - 1}" if family == "online" else f"A_{method}_{number}"
                classified.append(dict(series_id="A", method=method, status=status, **{id_name: identifier, offset_name: offset}))
            row = dict(method=method, scope="primary_level", n_series=2, tp=1, fp=1, fn=1,
                       n_events=2, n_monitoring_months=16, n_events_total=2, n_events_excluded=0,
                       precision=.5, recall=.5, f1=.5)
            if family == "online":
                row.update(n_alarms=2, n_alarms_excluded=0, missed_fraction=.5, median_delay=1,
                           false_alarms_per_12_months=.75)
            else:
                row.update(n_breakpoints=3, n_breakpoints_excluded=1, miss_rate=.5,
                           median_absolute_localisation_error=1, median_breakpoint_offset=1,
                           false_positives_per_12_months=.75)
                classified.append(dict(series_id="A", method=method, status="outside_evaluation_warmup",
                                       breakpoint_id=f"A_{method}_0", breakpoint_offset_months=np.nan))
                raw_points.extend([dict(series_id="A", method=method, breakpoint_id=f"A_{method}_{index}",
                                        breakpoint_month=month) for index, month in enumerate(["2024-03", "2024-06", "2024-07"])])
            metrics.append(row)
        calendar_name = "signals.csv.gz" if family == "online" else "stream.csv.gz"
        _save(root, directory + "/synthetic_test_" + calendar_name, pd.concat(calendars))
        _save(root, directory + ("/synthetic_test_event_matches.csv.gz" if family == "online" else "/synthetic_test_matches.csv.gz"), pd.DataFrame(matches))
        _save(root, directory + ("/synthetic_test_alarm_matches.csv.gz" if family == "online" else "/synthetic_test_classified_breakpoints.csv.gz"), pd.DataFrame(classified))
        _save(root, directory + ("/synthetic_test_metrics.csv.gz" if family == "online" else "/synthetic_test_metrics.csv"), pd.DataFrame(metrics))
        if family == "offline":
            _save(root, directory + "/synthetic_test_breakpoints.csv.gz", pd.DataFrame(raw_points))
        headings = ["method", "n_events", "precision", "recall", "f1"]
        headings += ["missed_fraction", "median_delay", "false_alarms_per_12_months"] if family == "online" else [
            "miss_rate", "median_absolute_localisation_error", "median_breakpoint_offset", "false_positives_per_12_months"]
        lines = ["| " + " | ".join(headings) + " |", "| " + " | ".join(["---"] * len(headings)) + " |"]
        for row in metrics:
            lines.append("| " + " | ".join(str(row[name]) for name in headings) + " |")
        lines += ["", "Число тревог: CUSUM: 2, EWMA: 2, BOCPD: 2."] if family == "online" else [
            "", "| method | all_candidates | monitoring_candidates | warmup_candidates |",
            "| --- | --- | --- | --- |", "| PELT | 1 | 1 | 0 |", "| BinSeg | 1 | 1 | 0 |",
            "", "| method | status | n_prefixes |", "| --- | --- | --- |",
            "| PELT | complete | 1 |", "| BinSeg | complete | 1 |", "", "Сохранено 2 prefix-оценок."]
        report = root / f"reports/{family}.md"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text("\n".join(lines), encoding="utf-8")
    online = "outputs/online_detection_v1"
    real = pd.concat([_calendar(method) for method in ONLINE], ignore_index=True)
    _save(root, online + "/real_signals.csv.gz", real)
    _save(root, online + "/real_alarms.csv", real.loc[real.is_alarm])
    (root / online / "real_diagnostics.json").write_text(json.dumps(dict(
        alarms_by_method={method: 2 for method in ONLINE}, n_observed_monitoring_months_per_method=16,
        independent_real_labels_available=False, real_precision_recall_computed=False,
        n_same_month_alarms_all_selected_methods=2, n_sample_municipalities=2, n_municipalities_with_monitoring=2)), encoding="utf-8")
    offline = "outputs/offline_detection_v1"
    _save(root, offline + "/real_stream.csv", pd.concat([_calendar(method) for method in OFFLINE]))
    _save(root, offline + "/real_series_diagnostics.csv", pd.DataFrame([dict(series_id=uid, method=method, status="complete") for method in OFFLINE for uid in ["A", "B"]]))
    _save(root, offline + "/real_breakpoints.csv", pd.DataFrame([dict(series_id="A", method=method, phase="monitoring", eligible_evaluation=True) for method in OFFLINE]))
    trajectories = pd.DataFrame([dict(series_id="A", method=method, full_breakpoint_id=f"A_{method}",
                                     full_sample_breakpoint="2024-06", prefix_end_period="2024-12", matched=True,
                                     comparison_available=True, exact_date_match=True, breakpoint_estimate_by_prefix="2024-06",
                                     matching_tolerance_months=1) for method in OFFLINE])
    _save(root, offline + "/real_prefix_matches.csv", trajectories)
    summaries = pd.DataFrame([dict(method=method, full_breakpoint_id=f"A_{method}", n_prefixes=1, n_matched_prefixes=1,
                                  n_exact_prefixes=1, n_date_revisions=0, n_presence_losses=0, first_prefix_where_detected="2024-12",
                                  first_breakpoint_estimate="2024-06", first_prefix_offset_months=6, revision_span_months=0) for method in OFFLINE])
    _save(root, offline + "/real_prefix_summary.csv", summaries)
    _save(root, offline + "/real_prefix_diagnostics.csv", pd.DataFrame([dict(method=method, status="complete") for method in OFFLINE]))
    _save(root, offline + "/real_prefix_estimates.csv", pd.DataFrame([dict(method=method) for method in OFFLINE]))
    manifest = {"figures": {}}
    for name in ["synthetic_level_example.png", "offline_metrics_ci.png", "municipality_21.png", "prefix_stability_21.png"]:
        target = root / offline / "figures" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"synthetic test fixture, no actual image")
        filename = "synthetic_test_metrics.csv" if name.startswith(("synthetic", "offline_metrics")) else "real_breakpoints.csv"
        manifest["figures"]["figures/" + name] = dict(
            sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
            source_sha256={filename: hashlib.sha256((root / offline / filename).read_bytes()).hexdigest()},
            selection={"rule": "fixed synthetic fixture"}, future_access="retrospective_known_interval_not_early_warning")
    (root / offline / "plots_source.json").write_text(json.dumps(manifest), encoding="utf-8")
    native = root / "outputs/e06a_checks/pelt_optimality_full.json"
    native.parent.mkdir(parents=True, exist_ok=True)
    native.write_text(json.dumps(dict(passed=True, native_globally_optimal_not_assumed=True,
                                     no_native_partition_replaced=True, n_native_vs_DP_comparisons=10,
                                     n_objective_gap_cases=2, maximum_objective_gap=1.0,
                                     n_new_stage_objective_gap_cases=0, counting="synthetic verification contexts",
                                     separate_lower_bound="separate unpruned DP")), encoding="utf-8")
    return root


def test_collection_records_explicit_statuses_provenance_hashes_and_no_fit(saved_project, monkeypatch):
    # No detector/forecast module is called, even if its exported fit is poisoned.
    import sberforecast.offline_detection as frozen
    monkeypatch.setattr(frozen, "run_offline", lambda *a, **k: pytest.fail("Collector performed detector fit"))
    before = {path: path.read_bytes() for path in saved_project.rglob("*") if path.is_file()}
    result = collect_detection(saved_project)
    assert len(result["records"]) == 10 and result["discrepancies"] == []
    quality = [row for row in result["records"] if row["data_status"] == "synthetic"]
    diagnostic = [row for row in result["records"] if row["data_status"] == "diagnostic"]
    assert len(quality) == len(diagnostic) == 5
    assert all(row["f1"] == .5 for row in quality)
    assert all(row["precision"] is None and row["recall"] is None and row["n_events"] is None for row in diagnostic)
    offline = [row for row in diagnostic if row["family"] == "offline"]
    assert all(row["n_series"] == 2 and row["n_series_with_candidates"] == 1 for row in offline)
    assert not result["facts"]["detector_fit_performed"] and not result["facts"]["forecast_fit_performed"]
    assert result["facts"]["offline_prefix_stability_is_hindsight_diagnostic"]
    assert result["facts"]["pelt_native_limitation"]["n_native_vs_DP_comparisons"] == 10
    assert result["facts"]["verification_counts"]["numeric_checks"] > 0
    for entry in result["provenance"]:
        assert entry["sha256"] == hashlib.sha256((saved_project / entry["path"]).read_bytes()).hexdigest()
        assert entry["definitions"] and entry["filters"]
    assert all(path.read_bytes() == content for path, content in before.items())
    assert {path for path in saved_project.rglob("*") if path.is_file()} == set(before)
    assert all(not row["early_warning"] for row in result["records"])
    assert all(row["median_absolute_localisation_error"] is None for row in quality if row["family"] == "online")
    assert all(row["median_detection_delay"] is None for row in quality if row["family"] == "offline")


@pytest.mark.parametrize("artifact,column,value,reason", [
    ("synthetic_test_metrics.csv", "precision", .9, "saved_metrics/precision"),
    ("synthetic_test_matches.csv.gz", "matched_breakpoint_id", "wrong_point", "point_link"),
    ("real_prefix_summary.csv", "n_date_revisions", 10, "n_date_revisions"),
])
def test_numeric_or_link_discrepancy_fails_instead_of_silently_copying_saved_summary(saved_project, artifact, column, value, reason):
    path = saved_project / "outputs/offline_detection_v1" / artifact
    frame = pd.read_csv(path)
    frame.loc[0, column] = value
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match=reason):
        collect_detection(saved_project)


def test_rounded_report_number_and_figure_source_corruption_are_detected(saved_project):
    report = saved_project / "reports/online.md"
    report.write_text(report.read_text(encoding="utf-8").replace("| CUSUM | 2 | 0.5", "| CUSUM | 2 | 0.9"), encoding="utf-8")
    with pytest.raises(ValueError, match="CUSUM/report/precision"):
        collect_detection(saved_project)
    report.write_text(report.read_text(encoding="utf-8").replace("| CUSUM | 2 | 0.9", "| CUSUM | 2 | 0.5"), encoding="utf-8")
    picture = saved_project / "outputs/offline_detection_v1/figures/municipality_21.png"
    picture.write_bytes(b"changed")
    with pytest.raises(ValueError, match="archived_image_hash"):
        collect_detection(saved_project)


def test_missing_artifact_and_outside_root_source_are_explicit_failures(saved_project):
    missing = saved_project / "outputs/offline_detection_v1/real_prefix_matches.csv"
    missing.unlink()
    with pytest.raises(ValueError, match="Missing detection artifact"):
        collect_detection(saved_project)
    config = saved_project / "configs/online_detection.yaml"
    data = yaml.safe_load(config.read_text(encoding="utf-8"))
    data["report_path"] = str(saved_project.parent / "outside_report.md")
    config.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError, match="inside project"):
        collect_detection(saved_project)
