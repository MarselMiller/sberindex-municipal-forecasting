"""Retrospective L2 segmentation and independently fitted prefix diagnostics.

The forecast residuals and first-four median/MAD calibration are those of E04a.
Unlike its online detectors, segmentation can use every finite residual in the
analyzed interval, including warmup, after calibration becomes possible. Only
breakpoints in the original monitoring interval enter the benchmark exposure.
Missing months remain in the calendar; a boundary index maps to the first
observed month of the following segment, never to the preceding observation.

Event labels are never passed to ruptures. Prefixes are fitted independently
using only facts available by their own end date. Association to full-sample
breakpoints occurs afterwards and is a hindsight diagnostic, not an online
alarm or evidence of early warning.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


STREAM_COLUMNS = [
    "series_id", "observation_period", "availability_date", "y_true", "y_pred",
    "error", "phase", "center", "scale", "z", "finite_index",
    "calibration_end_period", "calibration_count", "calibration_ready",
]
SEGMENT_COLUMNS = [
    "series_id", "method", "segment_id", "finite_start_index", "finite_end_index",
    "start_period", "end_period", "calendar_months", "n_observations",
    "mean_error", "std_error", "mean_z", "sse_l2", "analyzed_end_period",
    "analyzed_available_date", "center", "scale",
]
BREAKPOINT_COLUMNS = [
    "series_id", "method", "breakpoint_id", "breakpoint_month", "availability_date",
    "finite_boundary_index", "boundary_gap_calendar_months", "phase", "eligible_evaluation",
    "before_start_period", "before_end_period", "before_n_observations",
    "before_calendar_months", "before_mean_error", "before_std_error", "before_sse_l2",
    "after_start_period", "after_end_period", "after_n_observations",
    "after_calendar_months", "after_mean_error", "after_std_error", "after_sse_l2",
    "estimated_shift", "shift_units", "gain_l2", "penalty", "penalized_gain_l2",
    "total_sse_l2", "penalized_objective", "analyzed_end_period",
    "analyzed_available_date", "center", "scale",
]
DIAGNOSTIC_COLUMNS = [
    "series_id", "method", "status", "error", "n_calendar_months", "n_finite",
    "n_warmup", "n_monitoring", "n_missing", "n_unavailable", "n_breakpoints",
    "n_evaluation_breakpoints", "n_warmup_breakpoints", "calibration_end_period",
    "center", "scale", "analyzed_end_period", "analyzed_available_date",
    "total_sse_l2", "penalized_objective", "penalty", "model", "min_size", "jump",
]
ERROR_COLUMNS = ["series_id", "method", "stage", "error_type", "error"]
TRAJECTORY_COLUMNS = [
    "series_id", "method", "full_breakpoint_id", "full_sample_breakpoint",
    "prefix_end_period", "prefix_end_date", "prefix_status", "matched",
    "breakpoint_estimate_by_prefix", "prefix_breakpoint_id", "offset_from_full_months",
    "exact_date_match", "matching_tolerance_months", "comparison_available",
]
SUMMARY_COLUMNS = [
    "series_id", "method", "full_breakpoint_id", "full_sample_breakpoint",
    "full_breakpoint_phase", "eligible_evaluation", "first_prefix_where_detected",
    "first_prefix_end_date", "first_breakpoint_estimate", "first_prefix_offset_months",
    "first_exact_prefix", "n_prefixes", "n_matched_prefixes", "n_exact_prefixes",
    "n_date_revisions", "n_presence_losses", "minimum_estimated_month",
    "maximum_estimated_month", "revision_span_months", "last_matching_prefix",
    "matching_tolerance_months", "is_hindsight_diagnostic",
]


@dataclass
class OfflineResult:
    stream: pd.DataFrame
    breakpoints: pd.DataFrame
    segments: pd.DataFrame
    series_diagnostics: pd.DataFrame
    errors: pd.DataFrame


@dataclass
class PrefixStabilityResult:
    full_breakpoints: pd.DataFrame
    prefix_diagnostics: pd.DataFrame
    prefix_breakpoints: pd.DataFrame
    trajectories: pd.DataFrame
    summary: pd.DataFrame


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _positive(value: Any, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _end(period: pd.Period) -> str:
    return str(period.end_time.normalize().date())


def _as_of(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    if isinstance(value, pd.Period) or (isinstance(value, str) and len(value) == 7):
        return pd.Period(value, freq="M").end_time.normalize()
    value = pd.Timestamp(value)
    if pd.isna(value):
        raise ValueError("as_of must be a valid date")
    return value.normalize()


def _preparation(preparation: dict) -> tuple[int, int, float, float, pd.Timestamp | None]:
    warmup = _integer(preparation.get("warmup_observations", 4), "warmup_observations", 1)
    lag = _integer(preparation.get("release_lag_months", 0), "release_lag_months")
    relative = float(preparation.get("relative_scale_floor", 0.03))
    if not np.isfinite(relative) or relative < 0:
        raise ValueError("relative_scale_floor must be finite and nonnegative")
    absolute = _positive(preparation.get("absolute_scale_floor", 1.0), "absolute_scale_floor")
    return warmup, lag, relative, absolute, _as_of(preparation.get("as_of"))


def _input(residuals: pd.DataFrame, lag: int) -> pd.DataFrame:
    required = ["series_id", "observation_period", "y_true", "y_pred"]
    if missing := set(required) - set(residuals):
        raise ValueError(f"Missing residual columns: {sorted(missing)}")
    columns = required + (["fact_available_date"] if "fact_available_date" in residuals else [])
    data = residuals[columns].copy()
    if data["series_id"].isna().any():
        raise ValueError("series_id must not be missing")
    data["series_id"] = data["series_id"].astype(str)
    data["period"] = pd.PeriodIndex(data["observation_period"], freq="M")
    if data["period"].isna().any():
        raise ValueError("observation_period must not be missing")
    if data.duplicated(["series_id", "period"]).any():
        raise ValueError("Duplicate series_id / observation_period residual keys")
    expected = pd.DatetimeIndex([(p + lag).end_time.normalize() for p in data["period"]])
    if "fact_available_date" in data:
        dates = pd.to_datetime(data["fact_available_date"], errors="raise").dt.normalize()
        # Empty explicit metadata has the same assumed publication schedule as E04a.
        dates = dates.where(dates.notna(), pd.Series(expected, index=data.index))
        if np.any(dates.to_numpy() < expected.to_numpy()):
            raise ValueError("Fact availability precedes the configured publication lag")
        data["available"] = dates.to_numpy()
    else:
        data["available"] = expected
    return data


def prepare_streams(residuals: pd.DataFrame, preparation: dict) -> pd.DataFrame:
    """Preserve the calendar and use the unchanged E04a calibration formula.

Center and scale are fixed by the first ``warmup_observations`` finite errors
whose facts are available at ``as_of``. The scale is the maximum of their MAD,
the relative floor times their median absolute prediction, and absolute floor.
Warmup errors have z only once that calibration exists; they remain warmup and
are excluded from benchmark exposure. No unavailable value supplies a z.
"""
    warmup, lag, relative, absolute, as_of = _preparation(preparation)
    data = _input(residuals, lag)
    records = []
    for series_id, sub in data.groupby("series_id", sort=True):
        sub = sub.set_index("period").sort_index()
        calendar = pd.period_range(sub.index.min(), sub.index.max(), freq="M")
        sub = sub.reindex(calendar)
        series_rows = []
        errors: list[float] = []
        levels: list[float] = []
        calibration_end = ""
        finite_index = 0
        for period, row in sub.iterrows():
            available = row["available"]
            if pd.isna(available):
                available = (period + lag).end_time.normalize()
            available = pd.Timestamp(available).normalize()
            # iterrows can represent an entirely empty mixed datetime row as
            # NaT, including its numeric cells; all such cells remain NaN.
            y_true = np.nan if pd.isna(row["y_true"]) else float(row["y_true"])
            y_pred = np.nan if pd.isna(row["y_pred"]) else float(row["y_pred"])
            record = dict.fromkeys(STREAM_COLUMNS, np.nan)
            record.update(series_id=series_id, observation_period=str(period),
                          availability_date=str(available.date()), y_true=y_true, y_pred=y_pred,
                          phase="missing", calibration_count=len(errors),
                          calibration_end_period="", calibration_ready=False)
            if as_of is not None and available > as_of:
                record["phase"] = "unavailable"
            elif np.isfinite(y_true) and np.isfinite(y_pred):
                with np.errstate(over="ignore"):
                    error = y_true - y_pred
                if np.isfinite(error):
                    record["error"] = float(error)
                    record["finite_index"] = finite_index
                    finite_index += 1
                    if len(errors) < warmup:
                        record["phase"] = "warmup"
                        errors.append(float(error))
                        levels.append(abs(y_pred))
                        if len(errors) == warmup:
                            calibration_end = str(period)
                    else:
                        record["phase"] = "monitoring"
                    record["calibration_count"] = len(errors)
            series_rows.append(record)
        ready = len(errors) == warmup
        center = scale = np.nan
        if ready:
            center = float(np.median(errors))
            with np.errstate(over="ignore", invalid="ignore"):
                mad = float(1.4826 * np.median(np.abs(np.asarray(errors) - center)))
            scale = max(mad, relative * float(np.median(levels)), absolute)
            if not np.isfinite(center) or not np.isfinite(scale):
                ready = False
        for record in series_rows:
            record.update(center=center, scale=scale, calibration_ready=bool(ready),
                          calibration_end_period=calibration_end)
            if ready and np.isfinite(record["error"]):
                with np.errstate(over="ignore", invalid="ignore"):
                    record["z"] = float((record["error"] - center) / scale)
        records.extend(series_rows)
    result = pd.DataFrame(records, columns=STREAM_COLUMNS)
    result["calibration_ready"] = result["calibration_ready"].astype(bool)
    return result


def _parameters(method: str, params: dict) -> tuple[float, int, int]:
    if method not in {"PELT", "BinSeg"}:
        raise ValueError(f"Unknown offline detector: {method}")
    if unknown := set(params) - {"model", "penalty", "min_size", "jump"}:
        raise ValueError(f"Unsupported offline parameters: {sorted(unknown)}")
    if params.get("model", "l2") != "l2":
        raise ValueError("E06a uses the pre-specified l2 cost only")
    penalty = _positive(params.get("penalty", 3.0), "penalty")
    minimum = _integer(params.get("min_size", 2), "min_size", 2)
    jump = _integer(params.get("jump", 1), "jump", 1)
    if minimum != 2 or jump != 1:
        raise ValueError("E06a fixes min_size=2 and jump=1")
    return penalty, minimum, jump


def _fit_boundaries(values: np.ndarray, method: str, penalty: float,
                    min_size: int, jump: int) -> list[int]:
    """Only the numerical signal reaches ruptures; it has no event metadata."""
    import ruptures as rpt

    constructor = rpt.Pelt if method == "PELT" else rpt.Binseg
    fitted = constructor(model="l2", min_size=min_size, jump=jump).fit(values.reshape(-1, 1))
    return list(fitted.predict(pen=penalty))


def _sse(values: np.ndarray) -> float:
    with np.errstate(over="raise", invalid="raise"):
        return float(np.sum(np.square(values - np.mean(values))))


def run_offline(residuals: pd.DataFrame, method: str, params: dict,
                preparation: dict) -> OfflineResult:
    """Fit full-interval penalized L2 segmentation; never force a breakpoint.

The terminal n returned by ruptures denotes the final segment endpoint and is
not a breakpoint. The saved objective is sum(segment SSE(z)) + penalty * K,
where K is the number of actual interior boundaries. A boundary's gain compares
its adjacent segments to their merge, in standardized L2 units.
"""
    penalty, minimum, jump = _parameters(method, params)
    stream = prepare_streams(residuals, preparation)
    breakpoint_rows, segment_rows, diagnostic_rows, error_rows = [], [], [], []
    for series_id, sub in stream.groupby("series_id", sort=True):
        sub = sub.reset_index(drop=True)
        finite = sub[np.isfinite(sub["error"])].reset_index(drop=True)
        ready = not sub.empty and bool(sub["calibration_ready"].iloc[0])
        analyzed_end = str(sub["observation_period"].iloc[-1])
        as_of = _as_of(preparation.get("as_of"))
        available_end = (str(as_of.date()) if as_of is not None else
                         str(sub["availability_date"].max()))
        diagnostic = dict.fromkeys(DIAGNOSTIC_COLUMNS, np.nan)
        diagnostic.update(series_id=series_id, method=method, status="not_ready", error="",
                          n_calendar_months=len(sub), n_finite=len(finite),
                          n_warmup=int(sub["phase"].eq("warmup").sum()),
                          n_monitoring=int(sub["phase"].eq("monitoring").sum()),
                          n_missing=int(sub["phase"].eq("missing").sum()),
                          n_unavailable=int(sub["phase"].eq("unavailable").sum()),
                          n_breakpoints=0, n_evaluation_breakpoints=0, n_warmup_breakpoints=0,
                          calibration_end_period=sub["calibration_end_period"].iloc[0],
                          center=sub["center"].iloc[0], scale=sub["scale"].iloc[0],
                          analyzed_end_period=analyzed_end, analyzed_available_date=available_end,
                          penalty=penalty, model="l2", min_size=minimum, jump=jump)
        if not ready or len(finite) < minimum:
            diagnostic["error"] = "insufficient_available_calibration_observations"
            diagnostic_rows.append(diagnostic)
            continue
        stage = "fit"
        try:
            values = finite["z"].to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise FloatingPointError("Standardization produced a nonfinite residual")
            endpoints = _fit_boundaries(values, method, penalty, minimum, jump)
            if not endpoints or endpoints[-1] != len(finite):
                raise ValueError("Segmentation must end at the terminal observation count")
            if any(isinstance(b, bool) or int(b) != b for b in endpoints):
                raise ValueError("Segmentation boundaries must be integer indices")
            endpoints = [int(b) for b in endpoints]
            starts = [0] + endpoints[:-1]
            if endpoints != sorted(set(endpoints)) or any(b - a < minimum for a, b in zip(starts, endpoints)):
                raise ValueError("Invalid boundary ordering or segment size")
            stage = "diagnostics"
            own_segments = []
            for segment_number, (start, end) in enumerate(zip(starts, endpoints)):
                part = finite.iloc[start:end]
                raw = part["error"].to_numpy(dtype=float)
                first, last = part["observation_period"].iloc[[0, -1]]
                own_segments.append(dict(
                    series_id=series_id, method=method, segment_id=segment_number,
                    finite_start_index=start, finite_end_index=end, start_period=first,
                    end_period=last, calendar_months=pd.Period(last, "M").ordinal - pd.Period(first, "M").ordinal + 1,
                    n_observations=end - start, mean_error=float(np.mean(raw)),
                    std_error=float(np.std(raw, ddof=0)), mean_z=float(np.mean(values[start:end])),
                    sse_l2=_sse(values[start:end]), analyzed_end_period=analyzed_end,
                    analyzed_available_date=available_end, center=diagnostic["center"], scale=diagnostic["scale"],
                ))
            total_sse = sum(s["sse_l2"] for s in own_segments)
            objective = total_sse + penalty * (len(endpoints) - 1)
            own_breakpoints = []
            for number, boundary in enumerate(endpoints[:-1]):
                before, after = own_segments[number:number + 2]
                first_after = finite.iloc[boundary]
                merged_sse = _sse(values[before["finite_start_index"]:after["finite_end_index"]])
                gain = max(0.0, merged_sse - before["sse_l2"] - after["sse_l2"])
                record = dict(
                    series_id=series_id, method=method, breakpoint_id=f"{series_id}_{method}_{number}",
                    breakpoint_month=first_after["observation_period"],
                    availability_date=first_after["availability_date"], finite_boundary_index=boundary,
                    boundary_gap_calendar_months=(pd.Period(after["start_period"], "M").ordinal -
                                                  pd.Period(before["end_period"], "M").ordinal - 1),
                    phase=first_after["phase"], eligible_evaluation=bool(first_after["phase"] == "monitoring"),
                    estimated_shift=after["mean_error"] - before["mean_error"], shift_units="RUB",
                    gain_l2=gain, penalty=penalty, penalized_gain_l2=gain - penalty,
                    total_sse_l2=total_sse, penalized_objective=objective,
                    analyzed_end_period=analyzed_end, analyzed_available_date=available_end,
                    center=diagnostic["center"], scale=diagnostic["scale"],
                )
                for name, segment in [("before", before), ("after", after)]:
                    for field in ["start_period", "end_period", "n_observations", "calendar_months",
                                  "mean_error", "std_error", "sse_l2"]:
                        record[f"{name}_{field}"] = segment[field]
                own_breakpoints.append(record)
            diagnostic.update(status="complete", n_breakpoints=len(own_breakpoints),
                              n_evaluation_breakpoints=sum(b["eligible_evaluation"] for b in own_breakpoints),
                              n_warmup_breakpoints=sum(not b["eligible_evaluation"] for b in own_breakpoints),
                              total_sse_l2=total_sse, penalized_objective=objective)
            segment_rows.extend(own_segments)
            breakpoint_rows.extend(own_breakpoints)
        except Exception as exc:
            diagnostic.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            error_rows.append(dict(series_id=series_id, method=method, stage=stage,
                                   error_type=type(exc).__name__, error=str(exc)))
        diagnostic_rows.append(diagnostic)
    return OfflineResult(
        stream=stream, breakpoints=pd.DataFrame(breakpoint_rows, columns=BREAKPOINT_COLUMNS),
        segments=pd.DataFrame(segment_rows, columns=SEGMENT_COLUMNS),
        series_diagnostics=pd.DataFrame(diagnostic_rows, columns=DIAGNOSTIC_COLUMNS),
        errors=pd.DataFrame(error_rows, columns=ERROR_COLUMNS),
    )


def _matching(full: pd.DataFrame, prefix: pd.DataFrame, tolerance: int) -> dict[str, dict]:
    """Deterministic nearest-date one-to-one association, only after fitting."""
    edges = []
    for reference in full.to_dict("records"):
        ordinal = pd.Period(reference["breakpoint_month"], "M").ordinal
        for estimate in prefix.to_dict("records"):
            offset = pd.Period(estimate["breakpoint_month"], "M").ordinal - ordinal
            if abs(offset) <= tolerance:
                edges.append((abs(offset), reference["breakpoint_month"], estimate["breakpoint_month"],
                              reference["breakpoint_id"], estimate["breakpoint_id"], offset, estimate))
    result, used = {}, set()
    for _, _, _, reference_id, estimate_id, offset, estimate in sorted(edges, key=lambda e: e[:5]):
        if reference_id not in result and estimate_id not in used:
            result[reference_id] = dict(estimate=estimate, offset=offset)
            used.add(estimate_id)
    return result


def prefix_stability(residuals: pd.DataFrame, method: str, params: dict, preparation: dict,
                     *, matching_tolerance_months: int = 1) -> PrefixStabilityResult:
    """Refit all calendar prefixes using their own available facts and scaler.

Every prefix uses the same frozen penalty and stopping rule as the full fit.
No full-sample boundary or scale constrains a historical fit. Association within
the pre-specified calendar tolerance happens only after all fits are collected.
All prefix-only breakpoints are retained, including temporary ones that have no
full-sample counterpart. An unmatched full breakpoint has an empty first-prefix
date, rather than an invented detection time.
"""
    tolerance = _integer(matching_tolerance_months, "matching_tolerance_months")
    _, lag, _, _, original_as_of = _preparation(preparation)
    _parameters(method, params)
    data = _input(residuals, lag)
    full_preparation = dict(preparation)
    full = run_offline(residuals, method, params, full_preparation)
    prefix_diagnostics, prefix_breakpoints = [], []
    for series_id, sub in data.groupby("series_id", sort=True):
        final_prefix = max(sub["period"].max(), pd.DatetimeIndex(sub["available"]).to_period("M").max())
        calendar = pd.period_range(sub["period"].min(), final_prefix, freq="M")
        for prefix in calendar:
            prefix_date = prefix.end_time.normalize()
            if original_as_of is not None and prefix_date > original_as_of:
                continue
            # Keep unavailable rows as calendar placeholders but conceal their
            # future values; run_offline therefore cannot inspect those values.
            historical = sub[sub["period"] <= prefix].copy()
            hidden = historical["available"] > prefix_date
            historical.loc[hidden, ["y_true", "y_pred"]] = np.nan
            historical["fact_available_date"] = historical["available"]
            current_preparation = dict(preparation, as_of=str(prefix_date.date()))
            result = run_offline(historical, method, params, current_preparation)
            for record in result.series_diagnostics.to_dict("records"):
                record.update(prefix_end_period=str(prefix), prefix_end_date=str(prefix_date.date()))
                prefix_diagnostics.append(record)
            for record in result.breakpoints.to_dict("records"):
                record.update(prefix_end_period=str(prefix), prefix_end_date=str(prefix_date.date()),
                              is_temporary_without_full_match=True)
                prefix_breakpoints.append(record)
    diagnostic_frame = pd.DataFrame(prefix_diagnostics, columns=DIAGNOSTIC_COLUMNS + ["prefix_end_period", "prefix_end_date"])
    breakpoint_frame = pd.DataFrame(prefix_breakpoints, columns=BREAKPOINT_COLUMNS + [
        "prefix_end_period", "prefix_end_date", "is_temporary_without_full_match",
    ])
    trajectories, summaries = [], []
    for series_id, references in full.breakpoints.groupby("series_id", sort=True):
        own_diagnostics = diagnostic_frame[diagnostic_frame["series_id"] == series_id]
        own_trajectories = []
        for diagnostic in own_diagnostics.to_dict("records"):
            prefix = diagnostic["prefix_end_period"]
            estimates = breakpoint_frame[(breakpoint_frame["series_id"] == series_id) &
                                         (breakpoint_frame["prefix_end_period"] == prefix)]
            matches = _matching(references, estimates, tolerance) if diagnostic["status"] == "complete" else {}
            for reference in references.to_dict("records"):
                match = matches.get(reference["breakpoint_id"])
                record = dict(
                    series_id=series_id, method=method, full_breakpoint_id=reference["breakpoint_id"],
                    full_sample_breakpoint=reference["breakpoint_month"], prefix_end_period=prefix,
                    prefix_end_date=diagnostic["prefix_end_date"], prefix_status=diagnostic["status"],
                    matched=bool(match), breakpoint_estimate_by_prefix=(match["estimate"]["breakpoint_month"] if match else ""),
                    prefix_breakpoint_id=(match["estimate"]["breakpoint_id"] if match else ""),
                    offset_from_full_months=(match["offset"] if match else np.nan),
                    exact_date_match=bool(match and match["offset"] == 0),
                    matching_tolerance_months=tolerance, comparison_available=diagnostic["status"] == "complete",
                )
                own_trajectories.append(record)
                if match:
                    matched_rows = ((breakpoint_frame["series_id"] == series_id) &
                                    (breakpoint_frame["prefix_end_period"] == prefix) &
                                    (breakpoint_frame["breakpoint_id"] == match["estimate"]["breakpoint_id"]))
                    breakpoint_frame.loc[matched_rows, "is_temporary_without_full_match"] = False
        for reference in references.to_dict("records"):
            rows = [r for r in own_trajectories if r["full_breakpoint_id"] == reference["breakpoint_id"]]
            matched = [r for r in rows if r["matched"]]
            exact = [r for r in matched if r["exact_date_match"]]
            months = [pd.Period(r["breakpoint_estimate_by_prefix"], "M") for r in matched]
            comparable = [r for r in rows if r["comparison_available"]]
            losses = sum(a["matched"] and not b["matched"] for a, b in zip(comparable, comparable[1:]))
            revisions = sum(a["breakpoint_estimate_by_prefix"] != b["breakpoint_estimate_by_prefix"]
                            for a, b in zip(matched, matched[1:]))
            first = matched[0] if matched else {}
            summaries.append(dict(
                series_id=series_id, method=method, full_breakpoint_id=reference["breakpoint_id"],
                full_sample_breakpoint=reference["breakpoint_month"], full_breakpoint_phase=reference["phase"],
                eligible_evaluation=reference["eligible_evaluation"],
                first_prefix_where_detected=first.get("prefix_end_period", ""),
                first_prefix_end_date=first.get("prefix_end_date", ""),
                first_breakpoint_estimate=first.get("breakpoint_estimate_by_prefix", ""),
                first_prefix_offset_months=(pd.Period(first["prefix_end_period"], "M").ordinal -
                                           pd.Period(reference["breakpoint_month"], "M").ordinal if matched else np.nan),
                first_exact_prefix=exact[0]["prefix_end_period"] if exact else "",
                n_prefixes=len(rows), n_matched_prefixes=len(matched), n_exact_prefixes=len(exact),
                n_date_revisions=revisions, n_presence_losses=losses,
                minimum_estimated_month=str(min(months)) if months else "",
                maximum_estimated_month=str(max(months)) if months else "",
                revision_span_months=max(p.ordinal for p in months) - min(p.ordinal for p in months) if months else np.nan,
                last_matching_prefix=matched[-1]["prefix_end_period"] if matched else "",
                matching_tolerance_months=tolerance, is_hindsight_diagnostic=True,
            ))
        trajectories.extend(own_trajectories)
    return PrefixStabilityResult(
        full_breakpoints=full.breakpoints, prefix_diagnostics=diagnostic_frame,
        prefix_breakpoints=breakpoint_frame,
        trajectories=pd.DataFrame(trajectories, columns=TRAJECTORY_COLUMNS),
        summary=pd.DataFrame(summaries, columns=SUMMARY_COLUMNS),
    )
