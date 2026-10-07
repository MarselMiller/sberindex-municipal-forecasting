"""Read-only final extraction of E04a online and E06a offline detection.

No detector, forecasting model, optimizer or plotting code is imported. Synthetic
TEST quality is rebuilt from saved calendar signals/breakpoints and event truth.
Real rows contain unlabelled diagnostics only. Offline localisation and hindsight
prefix association never become online delays or early-warning measurements.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import yaml


ONLINE = ("CUSUM", "EWMA", "BOCPD")
OFFLINE = ("PELT", "BinSeg")
PRIMARY = ("level_up", "level_down")
METRICS = ("precision", "recall", "f1", "miss_rate", "median_detection_delay",
           "median_absolute_localisation_error", "false_positives_per_12_months")
DEFINITION = ("chronological one-to-one first point in inclusive T..T+3 calendar "
              "window; unmatched repeats are false positives; incomplete event "
              "windows and warmup points are excluded; observed monitoring "
              "months are the exposure, divided by twelve")


def _month(value) -> pd.Period:
    stamp = pd.Period(value, freq="M") if isinstance(value, (str, pd.Period)) else pd.Timestamp(value).to_period("M")
    if pd.isna(stamp):
        raise ValueError("Missing detection calendar month")
    return stamp


def _flags(values: pd.Series) -> pd.Series:
    if values.isna().any() or not values.isin([True, False, 0, 1]).all():
        raise ValueError("Detection flags must be explicit booleans")
    return values.astype(bool)


def _same(a, b, tolerance=1e-10) -> bool:
    if pd.isna(a) or pd.isna(b):
        return bool(pd.isna(a) and pd.isna(b))
    return bool(np.isclose(float(a), float(b), atol=tolerance, rtol=0))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reconstruct(stream, points, events, info, *, family, window=3):
    """Rebuild matching from saved point dates, without trusting saved status.

    This helper accepts one method. Online matching uses issue months, offline
    matching uses estimated observation months. The latter's analysis availability
    is deliberately absent from localisation arithmetic.
    """
    if family not in ("online", "offline"):
        raise ValueError("family must be online/offline")
    if stream.duplicated(["series_id", "observation_period"]).any() or info.series_id.duplicated().any():
        raise ValueError("Duplicate detection calendar/series key")
    if events.duplicated(["series_id", "event_id"]).any():
        raise ValueError("Duplicate event key")
    point_id = "alarm_id" if family == "online" else "breakpoint_id"
    date_field = "signal_date" if family == "online" else "breakpoint_month"
    stream = stream.copy()
    stream["_month"] = stream.observation_period.map(_month)
    by_stream = {uid: group for uid, group in stream.groupby("series_id", sort=False)}
    by_points = {uid: group for uid, group in points.groupby("series_id", sort=False)}
    by_events = {uid: group for uid, group in events.groupby("series_id", sort=False)}
    totals = dict(n_series=len(info), tp=0, fp=0, fn=0, n_events=0, n_points=0,
                  n_monitoring_months=0, n_events_total=0, n_events_excluded=0, n_points_excluded=0)
    matched_events, classified, offsets = [], [], []
    for series in info.itertuples(index=False):
        uid = series.series_id
        calendar = by_stream.get(uid, stream.iloc[:0]).sort_values("_month")
        monitor = calendar.loc[calendar.phase.eq("monitoring")]
        start, end = _month(series.observation_start_period), _month(series.observation_end_period)
        monitor_start = monitor["_month"].min() if len(monitor) else None
        lag = max((_month(available).ordinal - period.ordinal for available, period
                   in zip(calendar.availability_date, calendar["_month"]) if pd.notna(available)), default=0)
        if lag < 0:
            raise ValueError("Observation available before its calendar month")
        evaluation_end = end + lag if family == "online" else end
        current = []
        for event in by_events.get(uid, events.iloc[:0]).to_dict("records"):
            t = _month(event["event_start_period"])
            eligibility = ("outside_observation" if not start <= t <= end else
                           "before_monitoring" if monitor_start is None or t < monitor_start else
                           "partial_window" if t + window > evaluation_end else "full_window")
            current.append(dict(series_id=uid, event_id=str(event["event_id"]), start=t, end=t + window,
                                eligibility=eligibility, status="missed" if eligibility == "full_window" else "excluded",
                                matched_point_id="", offset=np.nan))
        current.sort(key=lambda row: (row["end"], row["start"], row["event_id"]))
        own = by_points.get(uid, points.iloc[:0]).copy()
        if len(own):
            own["_point_month"] = own[date_field].map(_month)
            if family == "online":
                own["_observation"] = own.observation_period.map(_month)
                own = own.sort_values(["_point_month", "_observation"], kind="stable")
            else:
                own = own.sort_values(["_point_month", "breakpoint_id"], kind="stable")
        phases = calendar.set_index("_month").phase.to_dict()
        for index, point in enumerate(own.to_dict("records")):
            t = point["_point_month"]
            identifier = f"{uid}:{index}" if family == "online" else str(point[point_id])
            phase = "monitoring" if family == "online" else phases.get(t)
            if phase is None:
                raise ValueError("Breakpoint absent from saved calendar")
            status, linked, offset = "false_positive", "", np.nan
            if phase != "monitoring":
                status = "outside_evaluation_" + phase
            else:
                eligible = [event for event in current if event["eligibility"] == "full_window"
                            and event["status"] == "missed" and event["start"] <= t <= event["end"]]
                if eligible:
                    chosen = eligible[0]
                    offset = t.ordinal - chosen["start"].ordinal
                    linked, status = chosen["event_id"], "true_positive"
                    chosen.update(status="detected", matched_point_id=identifier, offset=offset)
                    offsets.append(offset)
                elif any(event["eligibility"] != "full_window" and event["start"] <= t <= event["end"] for event in current):
                    status = "excluded_censored_event"
            classified.append(dict(series_id=uid, point_id=identifier, status=status, matched_event_id=linked, offset=offset))
            totals["tp"] += int(status == "true_positive")
            totals["fp"] += int(status == "false_positive")
            totals["n_points_excluded"] += int(status not in ("true_positive", "false_positive"))
        totals["n_monitoring_months"] += len(monitor)
        totals["n_points"] += len(own)
        totals["n_events_total"] += len(current)
        for event in current:
            totals["n_events"] += int(event["eligibility"] == "full_window")
            totals["n_events_excluded"] += int(event["eligibility"] != "full_window")
            matched_events.append({key: event[key] for key in ("series_id", "event_id", "eligibility", "status", "matched_point_id", "offset")})
    totals["fn"] = totals["n_events"] - totals["tp"]
    tp, fp, fn = totals["tp"], totals["fp"], totals["fn"]
    quality = dict(precision=tp / (tp + fp) if tp + fp else np.nan,
                   recall=tp / (tp + fn) if tp + fn else np.nan,
                   f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan,
                   miss_rate=fn / (tp + fn) if tp + fn else np.nan,
                   median_detection_delay=float(np.median(offsets)) if family == "online" and offsets else np.nan,
                   median_absolute_localisation_error=float(np.median(np.abs(offsets))) if family == "offline" and offsets else np.nan,
                   false_positives_per_12_months=12 * fp / totals["n_monitoring_months"] if totals["n_monitoring_months"] else np.nan)
    return {**totals, **quality}, pd.DataFrame(matched_events), pd.DataFrame(classified)


def _prefix_reconstruction(trajectories: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, identifier), group in trajectories.groupby(["method", "full_breakpoint_id"], sort=True):
        group = group.sort_values("prefix_end_period")
        matched = group.loc[_flags(group.matched)]
        comparable = group.loc[_flags(group.comparison_available)]
        present = _flags(comparable.matched).to_numpy()
        months = matched.breakpoint_estimate_by_prefix.map(_month)
        first = matched.iloc[0] if len(matched) else None
        rows.append(dict(method=method, full_breakpoint_id=identifier,
                         n_prefixes=len(group), n_matched_prefixes=len(matched),
                         n_exact_prefixes=int(_flags(matched.exact_date_match).sum()),
                         n_date_revisions=int(np.sum(months.to_numpy()[1:] != months.to_numpy()[:-1])),
                         n_presence_losses=int(np.sum(present[:-1] & ~present[1:])),
                         first_prefix_where_detected=first.prefix_end_period if first is not None else "",
                         first_breakpoint_estimate=first.breakpoint_estimate_by_prefix if first is not None else "",
                         first_prefix_offset_months=_month(first.prefix_end_period).ordinal - _month(first.full_sample_breakpoint).ordinal if first is not None else np.nan,
                         revision_span_months=max(value.ordinal for value in months) - min(value.ordinal for value in months) if len(months) else np.nan))
    return pd.DataFrame(rows)


def _report_table(text: str, required: set[str]) -> list[dict]:
    """Find an exact summary header; scenario and CI tables are not summaries."""
    lines = text.splitlines()
    for index, line in enumerate(lines):
        header = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if not line.startswith("|") or set(header) != required:
            continue
        rows = []
        for row in lines[index + 2:]:
            if not row.startswith("|"):
                break
            cells = [cell.strip() for cell in row.strip().strip("|").split("|")]
            if len(cells) == len(header):
                rows.append(dict(zip(header, cells)))
        return rows
    return []


def collect_detection(root: Path) -> dict:
    """Read existing artifacts and fail on any verified numeric inconsistency.

    Returns five synthetic TEST quality records and five real diagnostic records,
    a facts dictionary, byte-level SHA256 provenance, and an empty discrepancies
    list on success. Missing inputs and mismatching metrics/matches/reports raise
    ValueError. Figures are recommendations only; no output is created.
    """
    root = Path(root).resolve()
    provenance, records, discrepancies, assets = [], [], [], []
    provenance_by_path = {}
    verification_counts = dict(numeric_checks=0, integrity_checks=0)
    def source(path, status, definition, filters="all saved rows"):
        absolute = (root / path).resolve()
        if not absolute.is_relative_to(root):
            raise ValueError("Detection source must stay inside project root")
        if not absolute.is_file():
            raise ValueError(f"Missing detection artifact: {path}")
        relative = absolute.relative_to(root).as_posix()
        if relative not in provenance_by_path:
            entry = dict(path=relative, sha256=_sha(absolute), size_bytes=absolute.stat().st_size,
                         data_status=status, definitions=[], filters=[])
            provenance_by_path[relative] = entry
            provenance.append(entry)
        entry = provenance_by_path[relative]
        if definition not in entry["definitions"]:
            entry["definitions"].append(definition)
        if filters not in entry["filters"]:
            entry["filters"].append(filters)
        return absolute
    def csv(path, status, definition, filters="all saved rows"):
        absolute = source(path, status, definition, filters)
        frame = pd.read_csv(absolute, dtype={"series_id": str}, low_memory=False)
        provenance_by_path[absolute.relative_to(root).as_posix()]["rows"] = len(frame)
        return frame
    def compare(context, actual, expected, tolerance=1e-10):
        verification_counts["numeric_checks"] += 1
        if not _same(actual, expected, tolerance):
            discrepancies.append(dict(context=context, actual=None if pd.isna(actual) else actual,
                                      expected=None if pd.isna(expected) else expected))
    def check(context, condition):
        verification_counts["integrity_checks"] += 1
        if not condition:
            discrepancies.append(dict(context=context, reason="saved evidence mismatch"))
    configurations, reports = {}, {}
    for family, name in (("online", "online_detection"), ("offline", "offline_detection")):
        config = yaml.safe_load(source(f"configs/{name}.yaml", "synthetic", "locked historical protocol").read_text(encoding="utf-8"))
        configurations[family] = config
        directory = config["output_dir"]
        report_path = config["report_path"]
        report = source(report_path, "synthetic", "rounded published summary compared to reconstructed TEST quality").read_text(encoding="utf-8")
        reports[family] = report
        info = csv(f"{directory}/synthetic_test_series.csv.gz", "synthetic", "independent generator TEST series", "scenario in level_up,level_down")
        info = info.loc[info.scenario.isin(PRIMARY)]
        ids = set(info.series_id)
        events = csv(f"{directory}/synthetic_test_events.csv.gz", "synthetic", "generator event truth", "series_id in primary level TEST series")
        events = events.loc[events.series_id.isin(ids)]
        if family == "online":
            calendar_name, match_name, classified_name, metric_name = ("signals.csv.gz", "event_matches.csv.gz", "alarm_matches.csv.gz", "metrics.csv.gz")
            point_name = None
        else:
            calendar_name, match_name, classified_name, metric_name = ("stream.csv.gz", "matches.csv.gz", "classified_breakpoints.csv.gz", "metrics.csv")
            point_name = "breakpoints.csv.gz"
        calendar = csv(f"{directory}/synthetic_test_{calendar_name}", "synthetic", DEFINITION, "primary level series, one method at a time")
        calendar = calendar.loc[calendar.series_id.isin(ids)]
        points = calendar.loc[_flags(calendar.is_alarm)] if family == "online" else csv(
            f"{directory}/synthetic_test_{point_name}", "synthetic", "native saved segment boundaries; no refit", "primary level series")
        points = points.loc[points.series_id.isin(ids)]
        saved_matches = csv(f"{directory}/synthetic_test_{match_name}", "synthetic", "saved one-to-one event matching", "primary level series")
        saved_classified = csv(f"{directory}/synthetic_test_{classified_name}", "synthetic", "saved TP/FP/excluded point classification", "primary level series")
        saved_metrics = csv(f"{directory}/synthetic_test_{metric_name}", "synthetic", "saved primary TEST aggregate", "scope == primary_level")
        columns = {"method", "n_events", "precision", "recall", "f1",
                   "missed_fraction", "median_delay", "false_alarms_per_12_months"} if family == "online" else {
                       "method", "n_events", "precision", "recall", "f1", "miss_rate",
                       "median_absolute_localisation_error", "median_breakpoint_offset", "false_positives_per_12_months"}
        published = {row["method"]: row for row in _report_table(report, columns)}
        methods = ONLINE if family == "online" else OFFLINE
        for method in methods:
            stream, own_points = calendar.loc[calendar.method.eq(method)], points.loc[points.method.eq(method)]
            metrics, reconstructed_matches, reconstructed_points = _reconstruct(
                stream, own_points, events, info, family=family, window=int(config["evaluation"]["detection_window_months"]))
            access = "past_and_current_prefix_at_issue_date" if family == "online" else "full_series_future_relative_to_breakpoint"
            record = dict(family=family, method=method, split="test", scope="primary_level",
                          data_status="synthetic", access_mode=access, early_warning=False,
                          matching_definition=DEFINITION, **metrics)
            records.append({key: None if isinstance(value, (float, np.floating)) and not np.isfinite(value) else value for key, value in record.items()})
            saved = saved_metrics.loc[saved_metrics.method.eq(method) & saved_metrics.scope.eq("primary_level")]
            check(f"{method}/unique_primary_metric_row", len(saved) == 1)
            if len(saved) == 1:
                mapping = dict(miss_rate="missed_fraction", median_detection_delay="median_delay",
                               false_positives_per_12_months="false_alarms_per_12_months") if family == "online" else {}
                for field in ("tp", "fp", "fn", "n_events", "n_monitoring_months", "n_series", *METRICS):
                    if field == "median_absolute_localisation_error" and family == "online" or field == "median_detection_delay" and family == "offline":
                        continue
                    compare(f"{method}/saved_metrics/{field}", saved.iloc[0][mapping.get(field, field)], metrics[field])
                for field in ("n_events_total", "n_events_excluded"):
                    compare(f"{method}/saved_metrics/{field}", saved.iloc[0][field], metrics[field])
                for field, renamed in (("n_points", "n_alarms" if family == "online" else "n_breakpoints"),
                                       ("n_points_excluded", "n_alarms_excluded" if family == "online" else "n_breakpoints_excluded")):
                    compare(f"{method}/saved_metrics/{renamed}", saved.iloc[0][renamed], metrics[field])
            old_events = saved_matches.loc[saved_matches.method.eq(method) & saved_matches.series_id.isin(ids)].set_index(["series_id", "event_id"])
            check(f"{method}/event_keyset", set(old_events.index) == set(map(tuple, reconstructed_matches[["series_id", "event_id"]].to_numpy())))
            offset_field = "delay_months" if family == "online" else "breakpoint_offset_months"
            identifier_field = "matched_alarm_id" if family == "online" else "matched_breakpoint_id"
            for row in reconstructed_matches.itertuples():
                old = old_events.loc[(row.series_id, row.event_id)]
                check(f"{method}/{row.event_id}/eligibility_status", old.eligibility == row.eligibility and old.status == row.status)
                check(f"{method}/{row.event_id}/point_link", (str(old[identifier_field]) if pd.notna(old[identifier_field]) else "") == row.matched_point_id)
                compare(f"{method}/{row.event_id}/offset", old[offset_field], row.offset)
            point_field = "alarm_id" if family == "online" else "breakpoint_id"
            old_points = saved_classified.loc[saved_classified.method.eq(method) & saved_classified.series_id.isin(ids)].set_index(["series_id", point_field])
            check(f"{method}/point_keyset", set(old_points.index) == set(map(tuple, reconstructed_points[["series_id", "point_id"]].to_numpy())))
            for row in reconstructed_points.itertuples():
                old = old_points.loc[(row.series_id, row.point_id)]
                check(f"{method}/{row.point_id}/status", old.status == row.status)
                compare(f"{method}/{row.point_id}/offset", old[offset_field], row.offset)
            check(f"{method}/published_summary_present", method in published)
            for field, rendered in published.get(method, {}).items():
                if field == "method":
                    continue
                normalized = {"missed_fraction": "miss_rate", "median_delay": "median_detection_delay",
                              "median_breakpoint_offset": "median_absolute_localisation_error",
                              "false_alarms_per_12_months": "false_positives_per_12_months"}.get(field, field)
                decimal = len(rendered.split(".")[1]) if "." in rendered else 0
                compare(f"{method}/report/{field}", float(rendered), metrics[normalized], .5 * 10 ** (-decimal) + 1e-12)
    online, offline = configurations["online"]["output_dir"], configurations["offline"]["output_dir"]
    real_signals = csv(f"{online}/real_signals.csv.gz", "diagnostic", "unlabelled causal residual alarm stream")
    real_alarms = csv(f"{online}/real_alarms.csv", "diagnostic", "emitted alarms; no independent shock labels")
    real_json = json.loads(source(f"{online}/real_diagnostics.json", "diagnostic", "saved diagnostic aggregate").read_text(encoding="utf-8"))
    facts = dict(real_independent_labels_available=False, real_precision_recall_computed=False,
                 online_delay_is_after_onset=True, offline_localisation_uses_future=True,
                 offline_prefix_stability_is_hindsight_diagnostic=True,
                 detector_fit_performed=False, forecast_fit_performed=False,
                 publication_dates_and_vintages_verified=False, figures=assets)
    check("real/json_no_independent_labels", real_json["independent_real_labels_available"] is False)
    check("real/json_no_real_precision_recall", real_json["real_precision_recall_computed"] is False)
    alert_keys = []
    for method in ONLINE:
        own = real_signals.loc[real_signals.method.eq(method)]
        emitted = own.loc[_flags(own.is_alarm)]
        compare(f"real/{method}/saved_alarm_count", len(real_alarms.loc[real_alarms.method.eq(method)]), len(emitted))
        compare(f"real/{method}/json_alarm_count", real_json["alarms_by_method"][method], len(emitted))
        compare(f"real/{method}/monitoring_exposure", real_json["n_observed_monitoring_months_per_method"], own.phase.eq("monitoring").sum())
        reported = re.search(rf"\b{method}:\s*(\d+)", reports["online"])
        check(f"real/{method}/report_alarm_count_present", reported is not None)
        if reported:
            compare(f"real/{method}/report_alarm_count", int(reported.group(1)), len(emitted))
        alert_keys.append(set(map(tuple, emitted[["series_id", "observation_period"]].to_numpy())))
        records.append(dict(family="online", method=method, split="real_diagnostic", scope="unlabelled",
                            data_status="diagnostic", access_mode="past_and_current_prefix_at_issue_date",
                            early_warning=False, n_series=int(own.loc[own.phase.eq("monitoring"), "series_id"].nunique()),
                            n_monitoring_months=int(own.phase.eq("monitoring").sum()), n_alarms=len(emitted),
                            **{field: None for field in (*METRICS, "n_events")}))
    compare("real/all_three_alarm_agreement", real_json["n_same_month_alarms_all_selected_methods"], len(set.intersection(*alert_keys)))
    facts["real_online_panel"] = dict(original_series=real_json["n_sample_municipalities"],
                                      monitored_series=real_json["n_municipalities_with_monitoring"],
                                      observed_monitoring_months_per_method=real_json["n_observed_monitoring_months_per_method"],
                                      all_three_same_month_alarms=len(set.intersection(*alert_keys)))
    points = csv(f"{offline}/real_breakpoints.csv", "diagnostic", "retrospective candidate boundaries, includes warmup")
    real_stream = csv(f"{offline}/real_stream.csv", "diagnostic", "real observed monitoring calendar exposure")
    series_diagnostics = csv(f"{offline}/real_series_diagnostics.csv", "diagnostic", "saved full-series segmentation status, not new fit")
    trajectory = csv(f"{offline}/real_prefix_matches.csv", "diagnostic", "independent saved prefixes associated to full boundaries with hindsight")
    summary = csv(f"{offline}/real_prefix_summary.csv", "diagnostic", "saved revision/loss summaries")
    diagnostics = csv(f"{offline}/real_prefix_diagnostics.csv", "diagnostic", "prefix fit statuses, not new fit")
    estimates = csv(f"{offline}/real_prefix_estimates.csv", "diagnostic", "all prefix candidate estimates including transient unmatched points")
    reconstructed = _prefix_reconstruction(trajectory)
    check("real/prefix_matching_tolerance", trajectory.matching_tolerance_months.eq(
        configurations["offline"]["offline_analysis"]["prefix_matching_tolerance_months"]).all())
    old = summary.set_index(["method", "full_breakpoint_id"])
    check("real/prefix_summary_keyset", set(old.index) == set(map(tuple, reconstructed[["method", "full_breakpoint_id"]].to_numpy())))
    for row in reconstructed.to_dict("records"):
        saved = old.loc[(row["method"], row["full_breakpoint_id"])]
        for field, value in row.items():
            if field in ("method", "full_breakpoint_id"):
                continue
            if field in ("first_prefix_where_detected", "first_breakpoint_estimate"):
                check(f"prefix/{row['full_breakpoint_id']}/{field}", (str(saved[field]) if pd.notna(saved[field]) else "") == value)
            else:
                compare(f"prefix/{row['full_breakpoint_id']}/{field}", saved[field], value)
    for method in OFFLINE:
        own = points.loc[points.method.eq(method)]
        prefix = reconstructed.loc[reconstructed.method.eq(method)]
        calendar = real_stream.loc[real_stream.method.eq(method)]
        full_status = series_diagnostics.loc[series_diagnostics.method.eq(method)]
        statuses = diagnostics.loc[diagnostics.method.eq(method), "status"].value_counts().to_dict()
        table = _report_table(reports["offline"], {"method", "all_candidates", "monitoring_candidates", "warmup_candidates"})
        reported = next((row for row in table if row["method"] == method), None)
        check(f"real/{method}/report_candidates_present", reported is not None)
        if reported:
            for name, actual in (("all_candidates", len(own)), ("monitoring_candidates", _flags(own.eligible_evaluation).sum()),
                                 ("warmup_candidates", own.phase.eq("warmup").sum())):
                compare(f"real/{method}/report/{name}", int(reported[name]), actual)
        for row in _report_table(reports["offline"], {"method", "status", "n_prefixes"}):
            if row["method"] == method:
                compare(f"real/{method}/report/prefix_status/{row['status']}", int(row["n_prefixes"]), statuses.get(row["status"], 0))
        records.append(dict(family="offline", method=method, split="real_diagnostic", scope="unlabelled",
                            data_status="diagnostic", access_mode="full_series_future_relative_to_breakpoint", early_warning=False,
                            n_series=int(full_status.series_id.nunique()), n_series_with_candidates=int(own.series_id.nunique()),
                            n_monitoring_months=int(calendar.phase.eq("monitoring").sum()), n_candidates=len(own),
                            n_monitoring_candidates=int(_flags(own.eligible_evaluation).sum()),
                            n_warmup_candidates=int(own.phase.eq("warmup").sum()),
                            n_prefix_estimates=int(estimates.method.eq(method).sum()),
                            n_prefix_fits_complete=int(statuses.get("complete", 0)), n_prefix_fits_not_ready=int(statuses.get("not_ready", 0)),
                            n_prefix_fits_failed=int(statuses.get("failed", 0)),
                            n_date_revisions=int(prefix.n_date_revisions.sum()), n_presence_losses=int(prefix.n_presence_losses.sum()),
                            n_candidates_with_date_revisions=int(prefix.n_date_revisions.gt(0).sum()),
                            n_candidates_with_presence_losses=int(prefix.n_presence_losses.gt(0).sum()),
                            max_revision_span_months=float(prefix.revision_span_months.max()),
                            **{field: None for field in (*METRICS, "n_events")}))
    reported = re.search(r"Сохранено\s+(\d+)\s+prefix", reports["offline"])
    check("real/report_prefix_estimates_count_present", reported is not None)
    if reported:
        compare("real/report_prefix_estimates_count", int(reported.group(1)), len(estimates))
    facts["real_prefix_estimates_total"] = len(estimates)
    native = json.loads(source("outputs/e06a_checks/pelt_optimality_full.json", "diagnostic",
        "native library objective gaps against separate unpruned DP lower bound; counts are verification contexts, not unique series").read_text(encoding="utf-8"))
    check("PELT/native_objective_audit_passed", native["passed"] is True)
    check("PELT/no_claim_of_global_optimality", native["native_globally_optimal_not_assumed"] is True)
    check("PELT/native_partitions_not_replaced", native["no_native_partition_replaced"] is True)
    facts["pelt_native_limitation"] = {name: native[name] for name in (
        "n_native_vs_DP_comparisons", "n_objective_gap_cases", "maximum_objective_gap",
        "n_new_stage_objective_gap_cases", "native_globally_optimal_not_assumed",
        "no_native_partition_replaced", "counting", "separate_lower_bound")}
    plot_manifest = json.loads(source(f"{offline}/plots_source.json", "diagnostic", "archived existing figure source hashes").read_text(encoding="utf-8"))
    for name in ("synthetic_level_example.png", "offline_metrics_ci.png", "municipality_21.png", "prefix_stability_21.png"):
        relative = f"figures/{name}"
        archived = plot_manifest["figures"][relative]
        status = "synthetic" if name.startswith(("synthetic", "offline_metrics")) else "diagnostic"
        picture = source(f"{offline}/{relative}", status, "recommended existing figure; no regeneration")
        check(f"figure/{name}/archived_image_hash", _sha(picture) == archived["sha256"])
        paths = []
        for name_source, digest in archived["source_sha256"].items():
            path = f"{offline}/{name_source}"
            target = source(path, status, "existing figure input", json.dumps(archived["selection"], sort_keys=True))
            check(f"figure/{name}/{name_source}/archived_source_hash", _sha(target) == digest)
            paths.append(path)
        assets.append(dict(path=f"{offline}/{relative}", data_status=status, source_paths=paths,
                           selection=archived["selection"], access_mode=archived["future_access"]))
    if discrepancies:
        raise ValueError("Detection evidence mismatch: " + json.dumps(discrepancies[:12], ensure_ascii=False, default=str))
    facts["verification_counts"] = verification_counts
    return dict(records=records, facts=facts, provenance=provenance, discrepancies=discrepancies)
