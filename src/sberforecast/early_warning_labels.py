"""Finite weak labels for sustained *saved forecast-error* shifts, without fit.

These are neither independently annotated shocks nor changes in spending level.
Each onset uses four immediately preceding calendar errors and freezes their
median/scale through three confirmation months. Event confirmation, onset and
warning origin are separate. Online risk states are recomputed on an actually
available prefix; retrospective regime membership is audit-only.
"""
from __future__ import annotations

import json
from collections.abc import Mapping

import numpy as np
import pandas as pd

REQUIRED_RESIDUAL_COLUMNS = ("series_id", "observation_period", "y_true", "y_pred")
EVENT_COLUMNS = (
    "series_id", "municipality_id", "event_id", "onset_period", "confirmation_period",
    "confirmation_available_period", "confirmation_at", "direction", "baseline_center",
    "baseline_scale", "baseline_mad", "past_start_period", "past_end_period",
    "last_candidate_onset_period", "continuation_count",
)
REGIME_COLUMNS = EVENT_COLUMNS + (
    "recovery_start_period", "recovery_confirmation_period", "recovery_known_at",
    "regime_end_period", "regime_end_reason", "state_uncertain", "missing_state_months",
)


def _month(value: object) -> pd.Period:
    if isinstance(value, pd.Period):
        return value.asfreq("M")
    if value is None or pd.isna(value) or str(value).strip() == "":
        raise ValueError("A nonempty monthly period is required")
    return pd.Period(pd.Timestamp(value), freq="M")


def _date(period: pd.Period) -> str:
    # E01 explicitly assumes availability at month-end midnight, not a
    # verified publication timestamp. This output preserves that assumption.
    return str(period.end_time.normalize().date())


def _date_cutoff(value: object) -> pd.Timestamp:
    """Month inputs mean E01 month-end; explicit dates keep their day/time."""
    if isinstance(value, pd.Period):
        return value.asfreq("M").end_time.normalize()
    rendered = str(value).strip()
    if len(rendered) == 7 and rendered[4] == "-" and rendered[:4].isdigit() and rendered[5:].isdigit():
        return _month(value).end_time.normalize()
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError("A nonempty fit/origin date is required")
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("Europe/Moscow").tz_localize(None)
    return timestamp


def _identifier(value: object) -> str:
    if value is None or pd.isna(value) or str(value).strip() == "":
        raise ValueError("Missing municipality/series ID")
    return str(value).strip()


def _settings(config: Mapping) -> dict:
    weak = config.get("weak_label", config)
    required = ("past_window_months", "confirmation_months", "relative_scale_floor",
                "absolute_scale_floor", "strength_threshold", "merge_gap_months",
                "recovery_months", "recovery_strength")
    if any(name not in weak for name in required):
        raise ValueError(f"Missing weak-label specification: {[name for name in required if name not in weak]}")
    spec = {name: weak[name] for name in required}
    for name in ("past_window_months", "confirmation_months", "merge_gap_months", "recovery_months"):
        value = spec[name]
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < (0 if name == "merge_gap_months" else 1):
            raise ValueError(f"Invalid integer specification: {name}")
    if spec["past_window_months"] != 4 or spec["confirmation_months"] != 3:
        raise ValueError("This E07a criterion requires exactly four past and three confirmation months")
    for name in ("relative_scale_floor", "absolute_scale_floor", "strength_threshold", "recovery_strength"):
        if not np.isfinite(spec[name]) or spec[name] <= 0:
            raise ValueError(f"Invalid positive specification: {name}")
    if spec["recovery_strength"] >= spec["strength_threshold"]:
        raise ValueError("Recovery threshold must be below the event threshold")
    return spec


def _prepare(residuals: pd.DataFrame, samples: pd.DataFrame, config: Mapping) -> tuple[dict, str]:
    missing = set(REQUIRED_RESIDUAL_COLUMNS) - set(residuals)
    if missing:
        raise ValueError(f"Missing residual columns: {sorted(missing)}")
    if not {"municipality_id", "forecast_origin"}.issubset(samples):
        raise ValueError("Samples require municipality_id and forecast_origin")
    data = residuals.copy()
    data["_id"] = data.series_id.map(_identifier)
    data["_period"] = data.observation_period.map(_month)
    if data.duplicated(["_id", "_period"]).any():
        raise ValueError("Duplicate series/calendar residual month")
    for name in ("y_true", "y_pred"):
        data[name] = pd.to_numeric(data[name], errors="raise")
    finite_forecast = np.isfinite(data.y_pred)
    if "horizon" in data:
        horizon = pd.to_numeric(data.horizon, errors="raise")
        if (horizon.notna() & horizon.ne(1)).any() or (finite_forecast & horizon.isna()).any():
            raise ValueError("Weak targets require saved h=1 predictions only; missing metadata is allowed only for absent/nonfinite forecast placeholders")
    if "model" in data:
        present_model = data.model.notna() & data.model.astype(str).str.strip().ne("")
        if (present_model & data.model.ne("SeasonalNaiveYoY")).any() or (finite_forecast & ~present_model).any():
            raise ValueError("Weak targets require saved SeasonalNaiveYoY predictions only; missing metadata is allowed only for absent/nonfinite forecast placeholders")
    availability_column = next((name for name in ("available_period", "fact_available_period") if name in data), None)
    if availability_column:
        data["_available"] = data[availability_column].map(lambda value: None if pd.isna(value) or str(value).strip() == "" else _month(value))
        basis = f"explicit_{availability_column}"
    else:
        settings = config.get("data", {})
        if "release_lag_months" not in settings:
            raise ValueError("Residuals need availability periods or an explicit release_lag_months assumption")
        lag = settings["release_lag_months"]
        if isinstance(lag, bool) or not isinstance(lag, (int, np.integer)) or lag < 0:
            raise ValueError("release_lag_months must be a nonnegative integer")
        data["_available"] = data._period.map(lambda period: period + lag)
        basis = f"assumed_observation_period_plus_{lag}_months_no_verified_vintages"
    if any(available is not None and available < period for available, period in zip(data._available, data._period)):
        raise ValueError("Residual availability cannot precede its observation month")
    data["_error"] = data.y_true - data.y_pred
    data["_finite"] = np.isfinite(data.y_true) & np.isfinite(data.y_pred) & np.isfinite(data._error)
    bounds = config.get("data", {})
    sample_periods = samples.forecast_origin.map(_month)
    first = _month(bounds["first_month"]) if "first_month" in bounds else (data._period.min() if len(data) else sample_periods.min())
    last = _month(bounds["last_month"]) if "last_month" in bounds else (data._period.max() if len(data) else sample_periods.max())
    if first is pd.NaT or last is pd.NaT or first > last:
        raise ValueError("Explicit nonempty calendar bounds are required")
    if any(period < first or period > last for period in data._period):
        raise ValueError("Residual calendar lies outside the declared data period")
    series = {}
    for uid in sorted(set(samples.municipality_id.map(_identifier)) | set(data._id)):
        part = data.loc[data._id.eq(uid)].set_index("_period")
        table = part.reindex(pd.period_range(first, last, freq="M"))
        table["_finite"] = table["_finite"].eq(True)
        table["_available"] = table["_available"].map(lambda value: None if pd.isna(value) else value)
        table.attrs["observed_start"] = part.index.min() if len(part) else None
        table.attrs["observed_end"] = part.index.max() if len(part) else None
        series[uid] = table
    return series, basis


def _finite(table: pd.DataFrame, periods: list[pd.Period]) -> bool:
    return all(period in table.index and bool(table.at[period, "_finite"]) and table.at[period, "_available"] is not None for period in periods)


def _json_values(table: pd.DataFrame, periods: list[pd.Period], column: str) -> str:
    values = [float(table.at[period, column]) if period in table.index and pd.notna(table.at[period, column]) and np.isfinite(table.at[period, column]) else None for period in periods]
    return json.dumps(values, allow_nan=False)


def _concatenate(parts: list[pd.DataFrame], columns: tuple = ()) -> pd.DataFrame:
    nonempty = [part for part in parts if not part.empty]
    return pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame(columns=columns)


def _candidates(uid: str, table: pd.DataFrame, spec: dict) -> pd.DataFrame:
    records = []
    finite_periods = table.index[table._finite]
    first_finite = finite_periods.min() if len(finite_periods) else None
    last_observed = table.attrs.get("observed_end")
    for onset in table.index:
        past = list(pd.period_range(onset - 4, onset - 1, freq="M"))
        future = list(pd.period_range(onset, onset + 2, freq="M"))
        past_finite, future_finite = _finite(table, past), _finite(table, future)
        left = first_finite is None or past[0] < first_finite
        right = last_observed is not None and future[-1] > last_observed
        past_available = past_finite and all(table.at[p, "_available"] <= onset - 1 for p in past)
        center = mad = scale = np.nan
        direction = "unknown"
        deviations = [None, None, None]
        confirmed_at = None
        if past_finite:
            values = table.loc[past, "_error"].to_numpy(dtype=float)
            center = float(np.median(values))
            mad = float(np.median(np.abs(values - center)))
            scale = max(1.4826 * mad, spec["relative_scale_floor"] * float(np.median(np.abs(table.loc[past, "y_pred"]))), spec["absolute_scale_floor"])
        if not past_finite:
            status = "insufficient_past_residual_history" if left else "missing_past_residual_history"
        elif not past_available:
            status = "past_baseline_unavailable_before_onset"
        elif not future_finite:
            status = "right_censored_confirmation" if right else "missing_confirmation_residuals"
        else:
            deviations = ((table.loc[future, "_error"].to_numpy(dtype=float) - center) / scale).tolist()
            if all(value >= spec["strength_threshold"] for value in deviations):
                direction = "positive"
            elif all(value <= -spec["strength_threshold"] for value in deviations):
                direction = "negative"
            status = "confirmed_candidate" if direction != "unknown" else "no_sustained_error_shift"
            confirmed_at = max([future[-1]] + [table.at[p, "_available"] for p in past + future])
        records.append(dict(series_id=uid, municipality_id=uid, onset_period=str(onset),
            confirmation_period=str(onset + 2), confirmation_available_period=str(confirmed_at) if confirmed_at else "",
            confirmation_at=_date(confirmed_at) if confirmed_at else "", candidate_direction=direction,
            is_candidate=status == "confirmed_candidate", status=status, baseline_center=center,
            baseline_mad=mad, baseline_scale=scale, past_supported=past_finite and past_available,
            past_finite=past_finite, confirmation_supported=future_finite, left_insufficient=left,
            right_censored=right, evidence_start_period=str(onset - 4), evidence_end_period=str(onset + 2),
            evidence_month_count=7, past_residuals=_json_values(table, past, "_error"),
            past_predictions=_json_values(table, past, "y_pred"),
            confirmation_residuals=_json_values(table, future, "_error"),
            standardized_deviations=json.dumps(deviations, allow_nan=False), disposition="not_candidate", accepted_event_id=""))
    return pd.DataFrame(records)


def _episodes(uid: str, table: pd.DataFrame, candidates: pd.DataFrame, spec: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    result = candidates.copy()
    positions = {row.onset_period: index for index, row in result.iterrows()}
    events = []
    active = None
    recovery = []
    for period in table.index:
        if active is not None:
            if not bool(table.at[period, "_finite"]) or table.at[period, "_available"] is None:
                recovery = []
                active["state_uncertain"] = True
                active["missing_state_months"] += 1
            else:
                sign = 1 if active["direction"] == "positive" else -1
                deviation = sign * (float(table.at[period, "_error"]) - active["baseline_center"]) / active["baseline_scale"]
                if deviation < spec["recovery_strength"]:
                    recovery.append(period)
                else:
                    recovery = []
                if len(recovery) >= spec["recovery_months"]:
                    start = recovery[-spec["recovery_months"]]
                    available = max([period] + [table.at[p, "_available"] for p in recovery[-spec["recovery_months"]:]])
                    active.update(recovery_start_period=str(start), recovery_confirmation_period=str(period),
                        recovery_known_at=_date(available), regime_end_period=str(start), regime_end_reason="two_consecutive_signed_recovery_residuals")
                    active = None
                    recovery = []
        onset = period - 2
        index = positions.get(str(onset))
        if index is None or not bool(result.at[index, "is_candidate"]):
            continue
        row = result.loc[index]
        direction = row.candidate_direction
        previous = events[-1] if events else None
        continuation = active if active is not None and active["direction"] == direction else None
        if continuation is None and previous is not None and previous["direction"] == direction:
            if onset.ordinal - _month(previous["last_candidate_onset_period"]).ordinal <= spec["merge_gap_months"]:
                continuation = previous
        if continuation is not None:
            continuation["last_candidate_onset_period"] = str(onset)
            continuation["continuation_count"] += 1
            result.at[index, "disposition"] = "same_direction_continuation"
            result.at[index, "accepted_event_id"] = continuation["event_id"]
            continue
        if active is not None:
            # Opposite candidates are new onsets. Their confirmation can end
            # an old regime retrospectively at the new onset; prefix states
            # do not perform this update before the new confirmation exists.
            active.update(regime_end_period=str(onset), regime_end_reason="opposite_candidate_confirmed")
        event_id = f"weak_saved_error_{uid}_{onset}_{direction}"
        active = dict(series_id=uid, municipality_id=uid, event_id=event_id,
            onset_period=str(onset), confirmation_period=str(period),
            confirmation_available_period=row.confirmation_available_period,
            confirmation_at=row.confirmation_at, direction=direction,
            baseline_center=float(row.baseline_center), baseline_scale=float(row.baseline_scale),
            baseline_mad=float(row.baseline_mad), past_start_period=str(onset - 4),
            past_end_period=str(onset - 1), last_candidate_onset_period=str(onset), continuation_count=0,
            recovery_start_period="", recovery_confirmation_period="", recovery_known_at="",
            regime_end_period="", regime_end_reason="unrecovered_in_observed_prefix",
            state_uncertain=False, missing_state_months=0)
        events.append(active)
        recovery = []
        result.at[index, "disposition"] = "new_weak_onset"
        result.at[index, "accepted_event_id"] = event_id
    regimes = pd.DataFrame(events, columns=REGIME_COLUMNS)
    accepted = regimes.reindex(columns=EVENT_COLUMNS).copy()
    return result, accepted, regimes


def _prefix_state(uid: str, table: pd.DataFrame, origin: pd.Period, spec: dict, candidates: pd.DataFrame) -> dict:
    prefix = table.loc[table.index <= origin].copy()
    for period in prefix.index:
        available = prefix.at[period, "_available"]
        if available is None or available > origin:
            prefix.at[period, "_finite"] = False
            prefix.at[period, "_error"] = np.nan
            prefix.at[period, "y_pred"] = np.nan
    if prefix.empty:
        return dict(known_active_at_origin=False, active_event_id="", active_direction="unknown",
                    state_uncertain=True, finite_prefix_residual_months=0, recovery_streak_months=0)
    prefix_candidates = candidates.copy()
    known = prefix_candidates.confirmation_available_period.map(lambda value: bool(value) and _month(value) <= origin)
    prefix_candidates["is_candidate"] &= known
    _, _, regimes = _episodes(uid, prefix, prefix_candidates, spec)
    active = regimes.loc[regimes.regime_end_period.eq("")]
    current = active.iloc[-1] if len(active) else None
    streak = 0
    if current is not None:
        sign = 1 if current.direction == "positive" else -1
        for period in reversed(prefix.index):
            if period <= _month(current.confirmation_period) or not bool(prefix.at[period, "_finite"]):
                break
            deviation = sign * (float(prefix.at[period, "_error"]) - current.baseline_center) / current.baseline_scale
            if deviation >= spec["recovery_strength"]:
                break
            streak += 1
    return dict(known_active_at_origin=current is not None,
        active_event_id=current.event_id if current is not None else "",
        active_direction=current.direction if current is not None else "unknown",
        state_uncertain=bool(current.state_uncertain) if current is not None else not bool(prefix._finite.iloc[-1]),
        finite_prefix_residual_months=int(prefix._finite.sum()), recovery_streak_months=streak)


def build_early_warning_labels(residuals: pd.DataFrame, samples: pd.DataFrame, config: Mapping) -> dict[str, pd.DataFrame]:
    """Return candidates/events/regimes/origin_states/cases, preserving queries.

    Fully-known labels require every candidate baseline and the entire future
    window through O+k+2, even when an early positive onset already exists.
    Their common availability is at least that uniform window end. Explicit
    delayed dependencies can only postpone it, never shorten it by class.
    Regime merging also depends on earlier finite history, so the availability
    maximum includes that whole history through the confirmation-window end,
    uniformly for positive and negative cases. No extra missing-month filling
    or all-history completeness requirement is introduced.
    Missing input and censoring remain unknown, not negative. Active regimes
    exclude risk only from causal prefix states; retrospective membership is
    attached separately and must never be used as a historical feature.
    """
    spec = _settings(config)
    horizons = list(config.get("early_warning", {}).get("horizons_months", [1, 3]))
    if horizons != [1, 3]:
        raise ValueError("E07a requires the fixed warning horizons [1, 3]")
    normalized_keys = [( _identifier(row.municipality_id), _month(row.forecast_origin)) for row in samples.itertuples()]
    if len(set(normalized_keys)) != len(normalized_keys):
        raise ValueError("Duplicate municipality/warning origin query")
    series, availability_basis = _prepare(residuals, samples, config)
    candidate_parts, event_parts, regime_parts = [], [], []
    tables = {}
    for uid, table in series.items():
        candidates, events, regimes = _episodes(uid, table, _candidates(uid, table, spec), spec)
        tables[uid] = (candidates, events, regimes)
        candidate_parts.append(candidates)
        event_parts.append(events)
        regime_parts.append(regimes)
    origin_rows, case_rows = [], []
    for sample in samples.to_dict("records"):
        uid, origin = _identifier(sample["municipality_id"]), _month(sample["forecast_origin"])
        table = series[uid]
        candidates, events, regimes = tables[uid]
        state = _prefix_state(uid, table, origin, spec, candidates)
        inside = any(_month(row.onset_period) <= origin and (not row.regime_end_period or origin < _month(row.regime_end_period)) for row in regimes.itertuples())
        state_record = {**sample, **state, "warning_origin_period": str(origin),
            "retrospective_inside_regime_at_origin": inside, "availability_basis": availability_basis}
        origin_rows.append(state_record)
        for k in horizons:
            target_periods = list(pd.period_range(origin + 1, origin + k, freq="M"))
            future_periods = list(pd.period_range(origin + 1, origin + k + 2, freq="M"))
            target_candidates = candidates.loc[candidates.onset_period.isin([str(period) for period in target_periods])]
            past_supported = len(target_candidates) == k and bool(target_candidates.past_supported.all())
            future_supported = _finite(table, future_periods)
            fully_known = past_supported and future_supported
            first_finite = table.index[table._finite].min() if table._finite.any() else None
            left = first_finite is None or origin - 3 < first_finite
            last_observed = table.attrs.get("observed_end")
            right = last_observed is not None and origin + k + 2 > last_observed
            selected = events.loc[events.onset_period.isin([str(period) for period in target_periods])]
            # A much older delayed confirmation can change whether a current
            # same-direction candidate is a new onset or a continuation.
            # The dependency cutoff therefore includes the finite history
            # used by the episode state machine, for both label classes.
            history_dependencies = [p for p in table.index if p <= origin + k + 2
                and bool(table.at[p, "_finite"]) and table.at[p, "_available"] is not None]
            known = max([origin + k + 2] + [table.at[p, "_available"] for p in history_dependencies]) if fully_known else None
            status = "fully_known" if fully_known else "insufficient_past_residual_history" if left else "right_censored_future_window" if right else "missing_past_residual_history" if not past_supported else "missing_future_confirmation_residuals"
            case_rows.append({**state_record, "k": k, "target_start_period": str(origin + 1),
                "target_end_period": str(origin + k), "required_confirmation_end_period": str(origin + k + 2),
                "evidence_start_period": str(origin - 3), "evidence_end_period": str(origin + k + 2),
                "evidence_month_count": k + 6, "earliest_candidate_onset_period": str(origin + 1),
                "regime_history_dependency_start_period": str(history_dependencies[0]) if history_dependencies else "",
                "label_availability_dependency_month_count": len(history_dependencies),
                "past_history_supported": past_supported, "future_window_supported": future_supported,
                "left_insufficient": left, "right_censored": right,
                "missing_past": not past_supported and not left, "missing_future": not future_supported and not right,
                "fully_known": fully_known, "label": float(bool(len(selected))) if fully_known else np.nan,
                "label_known_at": _date(known) if known else "", "label_known_period": str(known) if known else "",
                "label_status": status, "missing_reason": "" if fully_known else status,
                "positive_event_ids": "|".join(selected.event_id.tolist()),
                "positive_onset_count": len(selected) if fully_known else np.nan,
                "at_risk": not state["known_active_at_origin"],
                "risk_exclusion_reason": "confirmed_unrecovered_regime_known_at_origin" if state["known_active_at_origin"] else ""})
    return dict(candidates=_concatenate(candidate_parts),
        events=_concatenate(event_parts, EVENT_COLUMNS),
        regimes=_concatenate(regime_parts, REGIME_COLUMNS),
        origin_states=pd.DataFrame(origin_rows), cases=pd.DataFrame(case_rows))


def admit_training_labels(cases: pd.DataFrame, fit_origin: object) -> pd.DataFrame:
    """Only at-risk, fully-known labels whose complete window is known by fit O."""
    required = {"fully_known", "label", "label_known_at", "at_risk", "forecast_origin"}
    if not required.issubset(cases):
        raise ValueError(f"Missing label-admission fields: {sorted(required - set(cases))}")
    cutoff = _date_cutoff(fit_origin)
    mask = cases.fully_known.astype(bool) & cases.at_risk.astype(bool) & cases.label.notna()
    known = cases.label_known_at.map(lambda value: None if pd.isna(value) or str(value).strip() == "" else _date_cutoff(value))
    mask &= known.map(lambda value: value is not None and pd.notna(value) and value <= cutoff)
    mask &= cases.forecast_origin.map(_date_cutoff).le(cutoff)
    return cases.loc[mask].copy()
