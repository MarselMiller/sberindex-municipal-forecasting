"""Event-level evaluation of online alarms, without retrospective relabelling.

The inclusive window is [event month, event month + window_months]. Alarms
are dated by their actual issue date, so publication delays count as delay.
Events before monitoring or without a complete window are retained but excluded
from complete-event metrics; alarms in their windows are explicitly excluded.
Repeated alarms for an already matched complete event count as false alarms.
"""
from __future__ import annotations

import json
from collections.abc import Sequence

import numpy as np
import pandas as pd


METRICS = (
    "precision", "recall", "f1", "missed_fraction", "median_delay",
    "false_alarms_per_12_months",
)
COUNTS = (
    "tp", "fp", "fn", "n_events", "n_alarms", "n_monitoring_months",
    "n_events_total", "n_events_excluded", "n_alarms_excluded",
)
EVENT_COLUMNS = [
    "series_id", "event_id", "event_type", "event_start_period", "strength",
    "window_end_period", "eligibility", "status", "matched_alarm_id",
    "signal_date", "delay_months",
]
ALARM_COLUMNS = [
    "series_id", "alarm_id", "observation_period", "signal_date", "status",
    "matched_event_id", "delay_months",
]


def _month(value: object) -> pd.Period:
    if isinstance(value, pd.Period):
        return value.asfreq("M")
    return pd.Period(pd.Timestamp(value), freq="M")


def _delays(value: object) -> list[float]:
    if isinstance(value, str):
        value = json.loads(value)
    if value is None:
        return []
    return list(value)


def evaluate_series(
    signals: pd.DataFrame,
    events: pd.DataFrame,
    series_info: pd.DataFrame,
    window_months: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return per-series counts, every event, and every issued alarm.

    ``signals`` has one row per calendar month, including missing and warmup
    rows. ``series_info`` must give the full observation calendar bounds;
    optional ``release_lag_months`` specifies availability of its final month.
    Otherwise the maximum row-wise publication lag determines the final
    availability month. Eligibility starts at the first monitored observation
    month, while matching and delays use signal issue months. A missing month
    contributes no observed monitoring exposure but never compresses a window.

    Call once per detector: different methods must not share an alarm pool.
    """
    if not isinstance(window_months, int) or window_months < 0:
        raise ValueError("window_months must be a nonnegative integer.")
    requirements = (
        (signals, {"series_id", "observation_period", "availability_date", "phase", "is_alarm", "signal_date"}),
        (events, {"series_id", "event_id", "event_type", "event_start_period", "strength"}),
        (series_info, {"series_id", "observation_start_period", "observation_end_period"}),
    )
    for frame, required in requirements:
        if missing := required.difference(frame.columns):
            raise ValueError(f"Missing evaluation columns: {sorted(missing)}")
        if frame.series_id.isna().any():
            raise ValueError("Empty series identifier.")
    if series_info.series_id.duplicated().any():
        raise ValueError("Repeated series_info identifier.")
    if events.duplicated(["series_id", "event_id"]).any():
        raise ValueError("Repeated event identifier within a series.")
    if not signals.phase.isin(["missing", "warmup", "monitoring"]).all():
        raise ValueError("Unknown signal phase.")
    ids = set(series_info.series_id)
    if not set(signals.series_id).issubset(ids) or not set(events.series_id).issubset(ids):
        raise ValueError("Signals/events contain a series absent from series_info.")
    sig = signals.copy()
    sig["_period"] = sig.observation_period.map(_month)
    if sig.duplicated(["series_id", "_period"]).any():
        raise ValueError("Repeated observation month within a series.")
    if sig.is_alarm.isna().any() or not sig.is_alarm.isin([True, False]).all():
        raise ValueError("is_alarm must contain explicit boolean values.")
    if (sig.is_alarm & sig.phase.ne("monitoring")).any():
        raise ValueError("An alarm cannot be issued during missing/warmup phases.")
    per_rows, event_rows, alarm_rows = [], [], []
    for info in series_info.to_dict("records"):
        series_id = info["series_id"]
        part = sig.loc[sig.series_id.eq(series_id)].sort_values("_period")
        observed_start = _month(info["observation_start_period"])
        observed_end = _month(info["observation_end_period"])
        if observed_start > observed_end:
            raise ValueError("Observation start is later than its end.")
        if len(part) and ((part._period < observed_start) | (part._period > observed_end)).any():
            raise ValueError("Signal observations lie outside the explicit calendar.")
        monitoring = part.loc[part.phase.eq("monitoring")]
        monitor_start = monitoring._period.min() if len(monitoring) else None
        if "release_lag_months" in info and pd.notna(info["release_lag_months"]):
            lag = int(info["release_lag_months"])
            if lag < 0:
                raise ValueError("Negative publication lag.")
        else:
            lags = [
                _month(available).ordinal - period.ordinal
                for available, period in zip(part.availability_date, part._period)
                if pd.notna(available)
            ]
            lag = max(lags, default=0)
            if any(value < 0 for value in lags):
                raise ValueError("An observation is available before its month.")
        availability_end = observed_end + lag
        current_events = []
        for event in events.loc[events.series_id.eq(series_id)].to_dict("records"):
            start = _month(event["event_start_period"])
            end = start + window_months
            if start < observed_start or start > observed_end:
                eligibility = "outside_observation"
            elif monitor_start is None or start < monitor_start:
                eligibility = "before_monitoring"
            elif end > availability_end:
                eligibility = "partial_window"
            else:
                eligibility = "full_window"
            current_events.append(dict(
                event, event_start_period=str(start), window_end_period=str(end),
                eligibility=eligibility, status="missed" if eligibility == "full_window" else "excluded",
                matched_alarm_id=None, signal_date=None, delay_months=np.nan,
                _start=start, _end=end,
            ))
        current_events.sort(key=lambda row: (row["_end"], row["_start"], str(row["event_id"])))
        issued = monitoring.loc[monitoring.is_alarm].copy()
        if issued.signal_date.isna().any():
            raise ValueError("An issued alarm has no signal_date.")
        issued["_signal_period"] = issued.signal_date.map(_month)
        issued = issued.sort_values(["_signal_period", "_period"], kind="stable")
        current_alarms = []
        for index, alarm in enumerate(issued.to_dict("records")):
            signal_month = alarm["_signal_period"]
            if pd.isna(alarm["availability_date"]) or pd.Timestamp(alarm["signal_date"]) < pd.Timestamp(alarm["availability_date"]):
                raise ValueError("An alarm precedes observation availability.")
            if signal_month > availability_end:
                raise ValueError("An alarm lies beyond the observation availability calendar.")
            candidates = [event for event in current_events if event["eligibility"] == "full_window"
                          and event["status"] == "missed"
                          and event["_start"] <= signal_month <= event["_end"]]
            record = dict(series_id=series_id, alarm_id=f"{series_id}:{index}",
                          observation_period=str(alarm["_period"]), signal_date=str(pd.Timestamp(alarm["signal_date"])),
                          status="false_positive", matched_event_id=None, delay_months=np.nan)
            if candidates:
                event = candidates[0]  # Earliest closing eligible window first.
                delay = signal_month.ordinal - event["_start"].ordinal
                event.update(status="detected", matched_alarm_id=record["alarm_id"],
                             signal_date=record["signal_date"], delay_months=delay)
                record.update(status="true_positive", matched_event_id=event["event_id"], delay_months=delay)
            elif any(event["eligibility"] != "full_window" and event["_start"] <= signal_month <= event["_end"]
                     for event in current_events):
                record["status"] = "excluded_censored_event"
            current_alarms.append(record)
        tp = sum(row["status"] == "true_positive" for row in current_alarms)
        fp = sum(row["status"] == "false_positive" for row in current_alarms)
        n_events = sum(row["eligibility"] == "full_window" for row in current_events)
        per_rows.append(dict(
            info, monitoring_start_period=str(monitor_start) if monitor_start is not None else None,
            availability_end_period=str(availability_end), tp=tp, fp=fp, fn=n_events - tp,
            n_events=n_events, n_alarms=len(current_alarms), n_monitoring_months=len(monitoring),
            n_events_total=len(current_events), n_events_excluded=len(current_events) - n_events,
            n_alarms_excluded=sum(row["status"] == "excluded_censored_event" for row in current_alarms),
            delays_months=json.dumps([row["delay_months"] for row in current_alarms if row["status"] == "true_positive"]),
        ))
        event_rows.extend({key: row.get(key) for key in EVENT_COLUMNS} for row in current_events)
        alarm_rows.extend(current_alarms)
    return pd.DataFrame(per_rows), pd.DataFrame(event_rows, columns=EVENT_COLUMNS), pd.DataFrame(alarm_rows, columns=ALARM_COLUMNS)


def _summarize(frame: pd.DataFrame) -> dict:
    totals = {column: int(frame[column].sum()) for column in COUNTS}
    tp, fp, fn = totals["tp"], totals["fp"], totals["fn"]
    delays = [delay for values in frame.delays_months for delay in _delays(values)]
    return dict(
        n_series=len(frame), **totals,
        precision=tp / (tp + fp) if tp + fp else np.nan,
        recall=tp / (tp + fn) if tp + fn else np.nan,
        f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan,
        missed_fraction=fn / (tp + fn) if tp + fn else np.nan,
        median_delay=float(np.median(delays)) if delays else np.nan,
        false_alarms_per_12_months=12 * fp / totals["n_monitoring_months"] if totals["n_monitoring_months"] else np.nan,
    )


def _groups(frame: pd.DataFrame, columns: Sequence[str]):
    if not columns:
        yield (), frame
    else:
        for key, group in frame.groupby(list(columns), dropna=False, sort=True):
            yield key if isinstance(key, tuple) else (key,), group


def aggregate_metrics(per_series: pd.DataFrame, group_columns: Sequence[str] = ()) -> pd.DataFrame:
    """Micro event counts and observed-month exposure; no event recall on controls."""
    rows = []
    for key, group in _groups(per_series, group_columns):
        rows.append(dict(zip(group_columns, key), **_summarize(group)))
    return pd.DataFrame(rows)


def bootstrap_metrics(
    per_series: pd.DataFrame,
    group_columns: Sequence[str] = (),
    n_bootstrap: int = 500,
    seed: int = 7319,
    strata_columns: Sequence[str] = ("scenario", "strength", "noise_fraction"),
) -> pd.DataFrame:
    """Percentile 95% intervals by resampling entire independent series.

    The designed mix of available scenario/strength/noise strata is preserved.
    Intervals condition on already selected detector parameters and do not
    include uncertainty of their validation selection. Never resample months.
    """
    if not isinstance(n_bootstrap, int) or n_bootstrap < 1:
        raise ValueError("n_bootstrap must be a positive integer.")
    rng = np.random.default_rng(seed)
    rows = []
    strata = [column for column in strata_columns if column in per_series.columns and column not in group_columns]
    for key, group in _groups(per_series, group_columns):
        summary = _summarize(group)
        group = group.reset_index(drop=True)
        pieces = [part.index.to_numpy() for _, part in _groups(group, strata) if len(part)]
        if pieces:
            # Draw complete rows within strata in one batch. Calendar months
            # and all matched delays belonging to a series travel together.
            indices = np.concatenate([part[rng.integers(0, len(part), (n_bootstrap, len(part)))]
                                      for part in pieces], axis=1)
            totals = group[list(COUNTS)].to_numpy(dtype=float)[indices].sum(axis=1)
            tp, fp, fn = totals[:, 0], totals[:, 1], totals[:, 2]
            def divide(numerator, denominator):
                return np.divide(numerator, denominator, out=np.full(n_bootstrap, np.nan), where=denominator != 0)
            samples = dict(precision=divide(tp, tp + fp), recall=divide(tp, tp + fn),
                           f1=divide(2 * tp, 2 * tp + fp + fn), missed_fraction=divide(fn, tp + fn),
                           false_alarms_per_12_months=divide(12 * fp, totals[:, 5]))
            delays = [_delays(value) for value in group.delays_months]
            width = max(map(len, delays), default=0)
            median = np.full(n_bootstrap, np.nan)
            if width:
                padded = np.full((len(group), width), np.nan)
                for index, values in enumerate(delays):
                    padded[index, :len(values)] = values
                sampled_delays = padded[indices].reshape(n_bootstrap, -1)
                valid = np.isfinite(sampled_delays).any(axis=1)
                median[valid] = np.nanmedian(sampled_delays[valid], axis=1)
            samples["median_delay"] = median
        else:
            samples = {metric: [] for metric in METRICS}
        for metric in METRICS:
            values = np.asarray(samples[metric], dtype=float)
            values = values[np.isfinite(values)]
            low, high = np.percentile(values, [2.5, 97.5]) if len(values) else (np.nan, np.nan)
            rows.append(dict(zip(group_columns, key), metric=metric, estimate=summary[metric],
                             ci_low=float(low), ci_high=float(high), n_bootstrap=n_bootstrap,
                             n_finite_bootstrap=len(values), bootstrap_seed=seed,
                             n_series=len(group), bootstrap_unit="series", stratified=True))
    return pd.DataFrame(rows)
