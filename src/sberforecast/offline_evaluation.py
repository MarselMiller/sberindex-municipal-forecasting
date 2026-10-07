"""Calendar event matching for retrospective E06a breakpoints.

Localization uses the estimated observation month, not the date on which a
full-sample analysis became available. The inclusive matching window is
[T, T + window_months]. A point before T is a false positive, never a warning.
As in E04a, events before monitoring and incomplete windows remain explicit;
points in their windows are excluded from complete-event metrics. With L=0
this is the same evaluation calendar as E04a. With L>0 publication does not
extend an offline localization window beyond the last observation month.
"""
from __future__ import annotations

import json
from collections.abc import Sequence

import numpy as np
import pandas as pd


METRICS = (
    "precision", "recall", "f1", "miss_rate",
    "median_absolute_localisation_error", "median_breakpoint_offset",
    "false_positives_per_12_months",
)
COUNTS = (
    "tp", "fp", "fn", "n_events", "n_breakpoints", "n_monitoring_months",
    "n_events_total", "n_events_excluded", "n_breakpoints_excluded",
)
MATCH_COLUMNS = (
    "series_id", "event_id", "event_type", "event_start_period", "strength",
    "window_end_period", "eligibility", "status", "matched_breakpoint_id",
    "breakpoint_month", "breakpoint_offset_months",
    "absolute_localisation_error_months",
)
CLASSIFIED_COLUMNS = (
    "series_id", "breakpoint_id", "breakpoint_month", "status",
    "matched_event_id", "breakpoint_offset_months",
    "absolute_localisation_error_months",
)


def _month(value: object) -> pd.Period:
    if pd.isna(value):
        raise ValueError("A calendar month cannot be missing.")
    if isinstance(value, pd.Period):
        return value.asfreq("M")
    return pd.Period(pd.Timestamp(value), freq="M")


def _offsets(value: object) -> list[float]:
    if isinstance(value, str):
        value = json.loads(value)
    if value is None:
        return []
    result = list(value)
    if any(not np.isfinite(offset) or offset < 0 for offset in result):
        raise ValueError("Matched breakpoint offsets must be finite and nonnegative.")
    return result


def evaluate_series(
    calendar_stream: pd.DataFrame,
    breakpoints: pd.DataFrame,
    events: pd.DataFrame,
    series_info: pd.DataFrame,
    *,
    window_months: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return per-series counts, all events, and all classified breakpoints.

    Call once per method. All calendar rows are retained by the detector;
    only its finite ``monitoring`` rows contribute exposure. Warmup points
    are preserved as ``outside_evaluation_warmup``, without becoming FP.
    Missing/unavailable calendar positions never compress a matching window.
    Extra breakpoint columns, including full-analysis dates/costs, are retained
    for provenance and never determine matching or localisation error.
    """
    if isinstance(window_months, bool) or not isinstance(window_months, int) or window_months < 0:
        raise ValueError("window_months must be a nonnegative integer.")
    requirements = (
        (calendar_stream, {"series_id", "observation_period", "availability_date", "phase"}),
        (breakpoints, {"series_id", "breakpoint_id", "breakpoint_month"}),
        (events, {"series_id", "event_id", "event_type", "event_start_period", "strength"}),
        (series_info, {"series_id", "observation_start_period", "observation_end_period"}),
    )
    for frame, required in requirements:
        if missing := required.difference(frame.columns):
            raise ValueError(f"Missing offline evaluation columns: {sorted(missing)}")
        if frame.series_id.isna().any():
            raise ValueError("Empty series identifier.")
    if series_info.series_id.duplicated().any():
        raise ValueError("Repeated series_info identifier.")
    if events.duplicated(["series_id", "event_id"]).any():
        raise ValueError("Repeated event identifier within a series.")
    if breakpoints.duplicated(["series_id", "breakpoint_id"]).any():
        raise ValueError("Repeated breakpoint identifier within a series.")
    if events.event_id.isna().any() or breakpoints.breakpoint_id.isna().any():
        raise ValueError("Event and breakpoint identifiers must be explicit.")
    if "method" in breakpoints and breakpoints.method.nunique(dropna=False) > 1:
        raise ValueError("Different methods must not share a breakpoint pool.")
    allowed_phases = {"warmup", "monitoring", "missing", "unavailable"}
    if not calendar_stream.phase.isin(allowed_phases).all():
        raise ValueError("Unknown offline stream phase.")
    ids = set(series_info.series_id)
    for frame in (calendar_stream, breakpoints, events):
        if not set(frame.series_id).issubset(ids):
            raise ValueError("An evaluation row refers to an unknown series.")
    stream = calendar_stream.copy()
    stream["_period"] = stream.observation_period.map(_month)
    points = breakpoints.copy()
    points["_period"] = points.breakpoint_month.map(_month)
    if stream.duplicated(["series_id", "_period"]).any():
        raise ValueError("Repeated observation month within a series.")
    if points.duplicated(["series_id", "_period"]).any():
        raise ValueError("Repeated breakpoint month within a series.")
    if "eligible_evaluation" in points:
        if points.eligible_evaluation.isna().any() or not points.eligible_evaluation.isin([True, False]).all():
            raise ValueError("eligible_evaluation must contain explicit booleans.")
    per_rows, event_rows, point_rows = [], [], []
    for info in series_info.to_dict("records"):
        series_id = info["series_id"]
        part = stream.loc[stream.series_id.eq(series_id)].sort_values("_period")
        observed_start = _month(info["observation_start_period"])
        observed_end = _month(info["observation_end_period"])
        if observed_start > observed_end:
            raise ValueError("Observation start is later than its end.")
        if len(part) and ((part._period < observed_start) | (part._period > observed_end)).any():
            raise ValueError("Stream observations lie outside the explicit calendar.")
        for available, period in zip(part.availability_date, part._period):
            if pd.notna(available) and _month(available) < period:
                raise ValueError("An observation is available before its month.")
        monitoring = part.loc[part.phase.eq("monitoring")]
        monitor_start = monitoring._period.min() if len(monitoring) else None
        if "release_lag_months" in info and pd.notna(info["release_lag_months"]):
            raw_lag = info["release_lag_months"]
            lag = int(raw_lag)
            if isinstance(raw_lag, bool) or lag < 0 or raw_lag != lag:
                raise ValueError("Publication lag must be a nonnegative integer.")
        else:
            lag = max((_month(available).ordinal - period.ordinal
                       for available, period in zip(part.availability_date, part._period)
                       if pd.notna(available)), default=0)
        current_events = []
        for event in events.loc[events.series_id.eq(series_id)].to_dict("records"):
            start = _month(event["event_start_period"])
            end = start + window_months
            if start < observed_start or start > observed_end:
                eligibility = "outside_observation"
            elif monitor_start is None or start < monitor_start:
                eligibility = "before_monitoring"
            elif end > observed_end:
                eligibility = "partial_window"
            else:
                eligibility = "full_window"
            current_events.append(dict(
                event, event_start_period=str(start), window_end_period=str(end),
                eligibility=eligibility,
                status="missed" if eligibility == "full_window" else "excluded",
                matched_breakpoint_id=None, breakpoint_month=None,
                breakpoint_offset_months=np.nan, absolute_localisation_error_months=np.nan,
                _start=start, _end=end,
            ))
        current_events.sort(key=lambda row: (row["_end"], row["_start"], str(row["event_id"])))
        current_points = []
        phases = part.set_index("_period").phase.to_dict()
        ordered_points = points.loc[points.series_id.eq(series_id)].copy()
        ordered_points["_id_order"] = ordered_points.breakpoint_id.astype(str)
        ordered_points = ordered_points.sort_values(["_period", "_id_order"], kind="stable")
        for point in ordered_points.to_dict("records"):
            month = point.pop("_period")
            point.pop("_id_order")
            if month < observed_start or month > observed_end:
                raise ValueError("A breakpoint lies outside the observation calendar.")
            if month not in phases:
                raise ValueError("A breakpoint month is absent from the retained calendar.")
            phase = phases[month]
            if "phase" in point and point["phase"] != phase:
                raise ValueError("Breakpoint phase differs from its calendar row.")
            if "eligible_evaluation" in point and bool(point["eligible_evaluation"]) != (phase == "monitoring"):
                raise ValueError("Breakpoint eligibility differs from its calendar phase.")
            record = dict(point, breakpoint_month=str(month), phase=phase,
                          eligible_evaluation=phase == "monitoring", status="false_positive",
                          matched_event_id=None, breakpoint_offset_months=np.nan,
                          absolute_localisation_error_months=np.nan)
            if phase != "monitoring":
                record["status"] = "outside_evaluation_" + phase
            else:
                candidates = [event for event in current_events
                              if event["eligibility"] == "full_window" and event["status"] == "missed"
                              and event["_start"] <= month <= event["_end"]]
                if candidates:
                    event = candidates[0]
                    offset = month.ordinal - event["_start"].ordinal
                    event.update(status="detected", matched_breakpoint_id=record["breakpoint_id"],
                                 breakpoint_month=str(month), breakpoint_offset_months=offset,
                                 absolute_localisation_error_months=abs(offset))
                    record.update(status="true_positive", matched_event_id=event["event_id"],
                                  breakpoint_offset_months=offset,
                                  absolute_localisation_error_months=abs(offset))
                elif any(event["eligibility"] != "full_window" and event["_start"] <= month <= event["_end"]
                         for event in current_events):
                    record["status"] = "excluded_censored_event"
            current_points.append(record)
        tp = sum(row["status"] == "true_positive" for row in current_points)
        fp = sum(row["status"] == "false_positive" for row in current_points)
        n_events = sum(row["eligibility"] == "full_window" for row in current_events)
        per_rows.append(dict(
            info, monitoring_start_period=str(monitor_start) if monitor_start is not None else None,
            availability_end_period=str(observed_end + lag),
            localization_end_period=str(observed_end),
            tp=tp, fp=fp, fn=n_events - tp, n_events=n_events,
            n_breakpoints=len(current_points), n_monitoring_months=len(monitoring),
            n_events_total=len(current_events), n_events_excluded=len(current_events) - n_events,
            n_breakpoints_excluded=sum(row["status"] not in ("true_positive", "false_positive")
                                       for row in current_points),
            breakpoint_offsets_months=json.dumps([row["breakpoint_offset_months"]
                                                  for row in current_points if row["status"] == "true_positive"]),
        ))
        event_rows.extend({key: value for key, value in row.items() if not key.startswith("_")}
                          for row in current_events)
        point_rows.extend(current_points)
    match_columns = list(dict.fromkeys([*events.columns, *MATCH_COLUMNS]))
    point_columns = list(dict.fromkeys([*breakpoints.columns, *CLASSIFIED_COLUMNS, "phase", "eligible_evaluation"]))
    return (pd.DataFrame(per_rows), pd.DataFrame(event_rows, columns=match_columns),
            pd.DataFrame(point_rows, columns=point_columns))


def _summarize(frame: pd.DataFrame) -> dict:
    totals = {column: int(frame[column].sum()) for column in COUNTS}
    tp, fp, fn = totals["tp"], totals["fp"], totals["fn"]
    offsets = [offset for values in frame.breakpoint_offsets_months for offset in _offsets(values)]
    return dict(
        n_series=len(frame), **totals,
        precision=tp / (tp + fp) if tp + fp else np.nan,
        recall=tp / (tp + fn) if tp + fn else np.nan,
        f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan,
        miss_rate=fn / (tp + fn) if tp + fn else np.nan,
        median_absolute_localisation_error=float(np.median(np.abs(offsets))) if offsets else np.nan,
        median_breakpoint_offset=float(np.median(offsets)) if offsets else np.nan,
        false_positives_per_12_months=12 * fp / totals["n_monitoring_months"] if totals["n_monitoring_months"] else np.nan,
    )


def _groups(frame: pd.DataFrame, columns: Sequence[str]):
    if not columns:
        yield (), frame
    else:
        for key, group in frame.groupby(list(columns), dropna=False, sort=True):
            yield key if isinstance(key, tuple) else (key,), group


def aggregate_metrics(per_series: pd.DataFrame, group_columns: Sequence[str] = ()) -> pd.DataFrame:
    """Aggregate event counts and observed monitoring months, without warmup."""
    rows = [dict(zip(group_columns, key), **_summarize(group))
            for key, group in _groups(per_series, group_columns)]
    return pd.DataFrame(rows, columns=[*group_columns, "n_series", *COUNTS, *METRICS])


def bootstrap_metrics(
    per_series: pd.DataFrame,
    group_columns: Sequence[str] = (),
    n_bootstrap: int = 500,
    seed: int = 300000,
    strata_columns: Sequence[str] = ("scenario", "strength", "noise_fraction"),
) -> pd.DataFrame:
    """Percentile 95% CIs from stratified resampling of entire series.

    Counts, monitoring exposure and all matched offsets of one series travel
    together. CIs condition on selected validation parameters; uncertainty of
    parameter selection is not included. No calendar month is resampled.
    """
    if isinstance(n_bootstrap, bool) or not isinstance(n_bootstrap, int) or n_bootstrap < 1:
        raise ValueError("n_bootstrap must be a positive integer.")
    rng = np.random.default_rng(seed)
    rows = []
    strata = [column for column in strata_columns if column in per_series and column not in group_columns]
    for key, original in _groups(per_series, group_columns):
        if original.series_id.duplicated().any():
            raise ValueError("Bootstrap requires one complete row per independent series within each group.")
        summary = _summarize(original)
        group = original.reset_index(drop=True)
        pieces = [part.index.to_numpy() for _, part in _groups(group, strata) if len(part)]
        if pieces:
            indices = np.concatenate([part[rng.integers(0, len(part), (n_bootstrap, len(part)))]
                                      for part in pieces], axis=1)
            totals = group[list(COUNTS)].to_numpy(dtype=float)[indices].sum(axis=1)
            tp, fp, fn = totals[:, 0], totals[:, 1], totals[:, 2]

            def divide(numerator, denominator):
                return np.divide(numerator, denominator, out=np.full(n_bootstrap, np.nan), where=denominator != 0)

            samples = dict(
                precision=divide(tp, tp + fp), recall=divide(tp, tp + fn),
                f1=divide(2 * tp, 2 * tp + fp + fn), miss_rate=divide(fn, tp + fn),
                false_positives_per_12_months=divide(12 * fp, totals[:, 5]),
            )
            offsets = [_offsets(value) for value in group.breakpoint_offsets_months]
            width = max(map(len, offsets), default=0)
            medians = np.full(n_bootstrap, np.nan)
            absolutes = np.full(n_bootstrap, np.nan)
            if width:
                padded = np.full((len(group), width), np.nan)
                for index, values in enumerate(offsets):
                    padded[index, :len(values)] = values
                sampled_offsets = padded[indices].reshape(n_bootstrap, -1)
                valid = np.isfinite(sampled_offsets).any(axis=1)
                medians[valid] = np.nanmedian(sampled_offsets[valid], axis=1)
                absolutes[valid] = np.nanmedian(np.abs(sampled_offsets[valid]), axis=1)
            samples["median_breakpoint_offset"] = medians
            samples["median_absolute_localisation_error"] = absolutes
        else:
            samples = {metric: [] for metric in METRICS}
        for metric in METRICS:
            values = np.asarray(samples[metric], dtype=float)
            values = values[np.isfinite(values)]
            low, high = np.percentile(values, [2.5, 97.5]) if len(values) else (np.nan, np.nan)
            rows.append(dict(zip(group_columns, key), metric=metric, estimate=summary[metric],
                             ci_low=float(low), ci_high=float(high), n_bootstrap=n_bootstrap,
                             n_finite_bootstrap=len(values), bootstrap_seed=seed,
                             n_series=len(group), bootstrap_unit="series", stratified=True,
                             conditional_on_selected_parameters=True))
    return pd.DataFrame(rows)


def select_candidate(rows: pd.DataFrame, budget: float) -> pd.Series | None:
    """Select one fixed-grid working point, strictly on synthetic validation.

    Each control separately must meet the unchanged budget. The pre-fixed
    order is maximum primary F1, recall, smaller maximum control FAR, smaller
    detected-event absolute localization error, then stable candidate_id.
    No feasible candidate returns None; budget/grid are never relaxed.
    """
    required = {"split", "candidate_id", "primary_f1", "primary_recall",
                "primary_median_absolute_localisation_error", "no_change_far", "outlier_far"}
    if missing := required.difference(rows.columns):
        raise ValueError(f"Missing candidate selection columns: {sorted(missing)}")
    if not rows.split.eq("validation").all():
        raise ValueError("Candidate parameters may be selected only on validation.")
    if "method" in rows and rows.method.nunique(dropna=False) > 1:
        raise ValueError("Select candidates separately for each method.")
    if rows.candidate_id.isna().any() or rows.candidate_id.duplicated().any():
        raise ValueError("Candidate identifiers must be explicit and unique.")
    if not np.isfinite(budget) or budget < 0:
        raise ValueError("The fixed control budget must be finite and nonnegative.")
    controls = rows[["no_change_far", "outlier_far"]].to_numpy(dtype=float)
    if (controls[np.isfinite(controls)] < 0).any():
        raise ValueError("Control false positive rates cannot be negative.")
    for metric in ("primary_f1", "primary_recall"):
        values = rows[metric].to_numpy(dtype=float)
        if ((values[np.isfinite(values)] < 0) | (values[np.isfinite(values)] > 1)).any():
            raise ValueError("Primary event rates must lie between zero and one.")
    max_control = np.max(controls, axis=1)
    if "max_control_far" in rows and not np.allclose(rows.max_control_far, max_control, rtol=0, atol=1e-12, equal_nan=True):
        raise ValueError("max_control_far disagrees with its explicit two controls.")
    valid_controls = np.isfinite(controls).all(axis=1) & (controls <= budget).all(axis=1)
    feasible = rows.loc[valid_controls & np.isfinite(rows.primary_f1) & np.isfinite(rows.primary_recall)].copy()
    if feasible.empty:
        return None
    feasible["max_control_far"] = max_control[valid_controls & np.isfinite(rows.primary_f1) & np.isfinite(rows.primary_recall)]
    feasible["localisation_order"] = feasible.primary_median_absolute_localisation_error.fillna(np.inf)
    return feasible.sort_values(
        ["primary_f1", "primary_recall", "max_control_far", "localisation_order", "candidate_id"],
        ascending=[False, False, True, True, True], kind="stable",
    ).iloc[0]
