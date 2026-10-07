"""E07a prefix features and descriptive column audits; no classifier or fits.

Expense/fact availability defaults to E01's assumed month-end midnight Moscow,
L=0. This is an explicit retrospective simulation, not verified old vintages.
Only saved causal h=1 predictions are read. Existing E04 detectors are replayed
on each origin's own prefix with their already selected fixed parameters.
Offline breakpoints, future calibration and full-history column selection are
excluded. Full-history audits are diagnostics; they never remove a column.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from .data import make_panel
from .macro_forecast_features import (
    MACRO_INDICATORS, macro_feature_dictionary,
    prepare_forecast_features,
)
from .online_detection import run_stream


EXPENSE_COLUMNS = (
    "expense_current_value", "expense_last_value", "expense_last_age_months",
    "expense_previous_month_value", "expense_mom_ratio", "expense_yoy_ratio",
    "expense_mean_3m", "expense_observed_months_3m", "expense_n_history_observations",
)
RESIDUAL_COLUMNS = (
    "residual_current_value", "residual_last_value", "residual_last_age_months",
    "residual_mean_3m", "residual_observed_months_3m",
)
DETECTORS = ("CUSUM", "EWMA", "BOCPD")
DETECTOR_FIELDS = ("score", "last_score", "is_alarm", "calibration_count", "ready", "state_age_months")
DETECTOR_COLUMNS = tuple(f"online_{method.lower()}_{field}" for method in DETECTORS for field in DETECTOR_FIELDS)
FEATURE_COLUMNS = EXPENSE_COLUMNS + RESIDUAL_COLUMNS + DETECTOR_COLUMNS
PHASE_COLUMNS = tuple(f"online_{method.lower()}_phase" for method in DETECTORS)
PROVENANCE_COLUMNS = (
    "sample_position", "municipality_id", "forecast_origin", "as_of_utc", "feature", "value",
    "max_available_at", "dependency_periods", "source_fields", "source", "missing", "missing_reason",
    "availability_assumption", "parameters_assumption",
)
DEFAULT_PREPARATION = dict(warmup_observations=4, relative_scale_floor=0.03,
                           absolute_scale_floor=1.0, cooldown_months=2, release_lag_months=0)


def _id(value: object) -> str:
    if pd.isna(value) or not str(value).strip():
        raise ValueError("Missing municipality identifier")
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") and text[:-2].isdigit() else text


def _instant(value: object) -> pd.Timestamp:
    if isinstance(value, pd.Period):
        value = value.end_time.normalize()
    elif isinstance(value, str) and len(value.strip()) == 7:
        value = pd.Period(value, freq="M").end_time.normalize()
    result = pd.Timestamp(value)
    if pd.isna(result):
        return pd.NaT
    return (result.tz_localize("Europe/Moscow") if result.tzinfo is None else result).tz_convert("UTC")


def _release(period: pd.Period, lag: int = 0) -> pd.Timestamp:
    return _instant((period + lag).end_time.normalize())


def _queries(samples: pd.DataFrame) -> pd.DataFrame:
    required = {"municipality_id", "forecast_origin"}
    if not required.issubset(samples):
        raise ValueError(f"Missing sample columns: {sorted(required - set(samples))}")
    result = samples.copy()
    result["_uid"] = result.municipality_id.map(_id)
    result["_as_of"] = result.forecast_origin.map(_instant)
    if result["_as_of"].isna().any():
        raise ValueError("Missing forecast origin")
    result["_period"] = result["_as_of"].map(lambda stamp: stamp.tz_convert("Europe/Moscow").tz_localize(None).to_period("M"))
    if result.duplicated(["_uid", "_as_of"]).any():
        raise ValueError("Duplicate municipality/forecast origin samples")
    return result


def _reject_offline(table: pd.DataFrame) -> None:
    if {"breakpoint_month", "finite_boundary_index", "full_sample_breakpoint"} & set(table):
        raise ValueError("Offline breakpoints are diagnostics, never prefix features")
    if "method" in table and table.method.astype(str).str.lower().isin(["pelt", "binseg", "offline"]).any():
        raise ValueError("Offline detector rows cannot supply online prefix features")
    if "is_hindsight_diagnostic" in table and table.is_hindsight_diagnostic.astype(str).str.lower().isin(["true", "1"]).any():
        raise ValueError("Hindsight diagnostics cannot supply prefix features")


def _observations(observations: pd.DataFrame, lag: int) -> pd.DataFrame:
    _reject_offline(observations)
    if {"series_id", "value", "observation_period"}.issubset(observations) and "municipality_id" not in observations:
        observations = observations.rename(columns={"series_id": "municipality_id", "value": "y"})
    if {"municipality_id", "y"}.issubset(observations):
        data = observations.copy()
        if "ds" not in data:
            if "observation_period" not in data:
                raise ValueError("Expense rows require ds or observation_period")
            data["ds"] = pd.PeriodIndex(data.observation_period, freq="M").to_timestamp()
    else:
        raise ValueError("Use data.load_data output (municipality_id, ds, y) for expenses")
    data["municipality_id"] = data.municipality_id.map(_id)
    data["ds"] = pd.PeriodIndex(pd.to_datetime(data.ds), freq="M").to_timestamp()
    data["y"] = pd.to_numeric(data.y, errors="raise")
    if np.isinf(data.y).any() or data.y.dropna().lt(0).any():
        raise ValueError("Expenses must be finite nonnegative values or missing")
    if data.duplicated(["municipality_id", "ds"]).any():
        raise ValueError("Duplicate expense month; no unverified vintage selection")
    # Reuse the project's calendar grid, preserving NaN months.
    if data.empty:
        return pd.DataFrame(columns=["_uid", "period", "value", "available_at"])
    panel = make_panel(data)
    panel.index.name = "ds"
    rows = panel.rename_axis(columns="_uid").stack(future_stack=True).rename("value").reset_index()
    rows["period"] = rows.ds.dt.to_period("M")
    rows["available_at"] = rows.period.map(lambda period: _release(period, lag))
    specified = set(zip(data.municipality_id, data.ds.dt.to_period("M")))
    present = np.asarray([key in specified for key in zip(rows._uid, rows.period)])
    # Calendar placeholders are not source facts. Otherwise adding the first
    # future row for a previously absent MO would invent old NaN availability.
    rows.loc[~present, "available_at"] = pd.NaT
    if "available_at" in data:
        supplied = data.assign(_uid=data.municipality_id, period=data.ds.dt.to_period("M"))
        supplied["_explicit"] = supplied.available_at.map(_instant)
        rows = rows.merge(supplied[["_uid", "period", "_explicit"]], on=["_uid", "period"], how="left", validate="one_to_one")
        explicit = rows["_explicit"].notna()
        if rows.loc[explicit, "_explicit"].lt(rows.loc[explicit, "available_at"]).any():
            raise ValueError("Expense availability precedes the completed observation month")
        rows.loc[explicit, "available_at"] = rows.loc[explicit, "_explicit"]
        # An explicit missing availability does not become the E01 assumption.
        rows.loc[present & ~explicit, "available_at"] = pd.NaT
    return rows[["_uid", "period", "value", "available_at"]]


def _residuals(residuals: pd.DataFrame, lag: int) -> pd.DataFrame:
    _reject_offline(residuals)
    required = {"series_id", "observation_period", "y_true", "y_pred"}
    if not required.issubset(residuals):
        raise ValueError(f"Missing saved residual columns: {sorted(required - set(residuals))}")
    data = residuals.copy()
    data["_uid"] = data.series_id.map(_id)
    data["period"] = pd.PeriodIndex(data.observation_period, freq="M")
    if data.period.isna().any() or data.duplicated(["_uid", "period"]).any():
        raise ValueError("Missing or duplicate saved residual month")
    for column in ("y_true", "y_pred"):
        data[column] = pd.to_numeric(data[column], errors="raise")
        if np.isinf(data[column]).any():
            raise ValueError("Residual inputs must be finite or missing")
    present_forecast = data.y_pred.notna()
    # E04 deliberately preserves calendar placeholders after a missing series
    # no longer has an issued forecast. Their absent metadata is not a forecast.
    if "model" in data and ((present_forecast | data.model.notna()) & ~data.model.eq("SeasonalNaiveYoY")).any():
        raise ValueError("Only the saved causal SeasonalNaiveYoY baseline is allowed")
    if "horizon" in data:
        horizon = pd.to_numeric(data.horizon, errors="raise")
        if ((present_forecast | horizon.notna()) & ~horizon.eq(1)).any():
            raise ValueError("Only saved h=1 forecasts can supply residuals")
    if data.empty:
        for column in ("available_at", "forecast_available_at"):
            data[column] = pd.Series(index=data.index, dtype="datetime64[ns, UTC]")
        data["value"] = pd.Series(index=data.index, dtype=float)
        return data
    data["available_at"] = data.period.map(lambda period: _release(period, lag))
    for column in ("fact_available_at", "fact_available_date"):
        if column in data:
            explicit = data[column].map(_instant)
            if explicit.notna().any() and explicit.loc[explicit.notna()].lt(data.loc[explicit.notna(), "available_at"]).any():
                raise ValueError("Residual fact availability precedes the completed month")
            data["available_at"] = explicit
    if "forecast_origin" in data:
        issued = pd.to_datetime(data.forecast_origin.map(_instant), utc=True)
        expected = pd.to_datetime(data.period.map(lambda period: _release(period - 1)), utc=True)
        if (present_forecast & issued.isna()).any() or issued.gt(expected).any():
            raise ValueError("A saved h=1 forecast must be issued before its target observation")
        if "history_cutoff" in data:
            history = pd.to_datetime(data.history_cutoff.map(_instant), utc=True)
            if (present_forecast & history.isna()).any() or history.gt(issued).any():
                raise ValueError("Saved forecast history cutoff is missing or exceeds its issue date")
        data["forecast_available_at"] = issued.where(present_forecast)
    else:
        data["forecast_available_at"] = pd.to_datetime(data.period.map(lambda period: _release(period - 1)), utc=True).where(present_forecast)
    with np.errstate(over="ignore"):
        data["value"] = data.y_true - data.y_pred
    if np.isinf(data.value).any():
        raise ValueError("Saved residual subtraction exceeded finite numeric range")
    data["series_id"] = data["_uid"]
    data["available_at"] = pd.to_datetime(data.available_at, utc=True)
    return data


def _lookup(table: pd.DataFrame, periods: Sequence[pd.Period]) -> pd.DataFrame:
    return table.set_index("period").reindex(periods)


def build_prefix_history_features(
    observations: pd.DataFrame, residuals: pd.DataFrame, samples: pd.DataFrame,
    config: Mapping, selected_parameters: Mapping,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Preserve every input key and return features plus per-feature provenance.

    Expenses accept :func:`data.load_data` output or the equivalent long
    ``series_id,value,observation_period`` schema. Residuals accept saved E04
    ``real_residuals.csv.gz``. The caller may copy E07 configuration and attach
    saved E04 ``preparation`` under ``detector_preparation``. No source is read
    from disk here and no old forecast model is called.
    """
    queries = _queries(samples)
    lag = config.get("data", {}).get("release_lag_months", 0)
    if isinstance(lag, bool) or not isinstance(lag, int) or lag != 0:
        raise ValueError("E07a preserves the explicit E01 L=0 availability assumption")
    preparation = dict(DEFAULT_PREPARATION)
    preparation.update(config.get("detector_preparation", config.get("preparation", {})))
    if preparation.get("release_lag_months", 0) != lag:
        raise ValueError("Detector and expense release lags disagree")
    collisions = set(FEATURE_COLUMNS + PHASE_COLUMNS + tuple(name + "_missing" for name in FEATURE_COLUMNS)) & set(samples)
    if collisions:
        raise ValueError(f"Feature columns already present: {sorted(collisions)}")
    selected = {}
    for method in DETECTORS:
        entry = selected_parameters.get(method)
        if not entry or entry.get("selected", True) is not True:
            raise ValueError(f"Missing fixed E04 selected parameters for {method}")
        selected[method] = dict(entry.get("parameters", entry))
    expenses = _observations(observations, lag)
    errors = _residuals(residuals, lag)
    result_rows, provenance = [], []
    streams = {}
    for position, query in enumerate(queries.to_dict("records")):
        uid, cutoff, period = query["_uid"], query["_as_of"], query["_period"]
        row = dict.fromkeys(FEATURE_COLUMNS, np.nan)
        expense = expenses.loc[expenses._uid.eq(uid) & expenses.available_at.le(cutoff) & expenses.period.le(period)]
        error = errors.loc[errors._uid.eq(uid) & errors.available_at.le(cutoff) & errors.forecast_available_at.le(cutoff) & errors.period.le(period)]

        def record(name, value, dependency: pd.DataFrame, source, fields, reason=""):
            row[name] = float(value)
            available = dependency.available_at.max() if len(dependency) else pd.NaT
            if pd.notna(available) and available > cutoff:
                raise AssertionError("A feature dependency exceeds its own origin")
            provenance.append(dict(sample_position=position, municipality_id=uid, forecast_origin=query["forecast_origin"],
                as_of_utc=cutoff, feature=name, value=float(value), max_available_at=available,
                dependency_periods=json.dumps([str(value) for value in dependency.index if isinstance(value, pd.Period)]
                    if "period" not in dependency else dependency.period.astype(str).tolist()),
                source_fields=fields, source=source, missing=bool(pd.isna(value)), missing_reason=reason if pd.isna(value) else "",
                availability_assumption="E01_L0_month_end_midnight_Moscow_no_verified_expense_vintages",
                parameters_assumption="Fixed previously selected E04 parameters; retrospective prefix replay, not verified historic parameter selection" if source == "E04_prefix_detector" else ""))

        for prefix, source, table in (("expense", "expense_history", expense), ("residual", "saved_E04_h1_residual", error)):
            finite = table.loc[table.value.notna()].sort_values("period")
            current = _lookup(table, [period])
            record(prefix + "_current_value", current.value.iloc[0], current, source, "y" if prefix == "expense" else "y_true,y_pred", "current_month_missing_or_unavailable")
            last = finite.iloc[-1:]
            record(prefix + "_last_value", np.nan if last.empty else last.value.iloc[0], last, source, "last_finite_value", "no_available_finite_history")
            record(prefix + "_last_age_months", np.nan if last.empty else period.ordinal-last.period.iloc[0].ordinal, last, source, "last_finite_observation_period", "no_available_finite_history")
            recent = _lookup(table, [period-2, period-1, period])
            count = int(recent.value.notna().sum())
            record(prefix + "_mean_3m", float(recent.value.mean()) if count == 3 else np.nan, recent, source, "three_immediately_preceding_calendar_values_including_O", "requires_all_three_available_finite_months")
            record(prefix + "_observed_months_3m", count, recent, source, "finite_value_count_in_calendar_3m")
            if prefix == "expense":
                previous = _lookup(table, [period-1])
                yearly = _lookup(table, [period-12])
                record("expense_previous_month_value", previous.value.iloc[0], previous, source, "y_O_minus_1", "previous_calendar_month_missing")
                for name, denominator in (("expense_mom_ratio", previous), ("expense_yoy_ratio", yearly)):
                    top, bottom = current.value.iloc[0], denominator.value.iloc[0]
                    value = top/bottom if pd.notna(top) and pd.notna(bottom) and bottom > 0 else np.nan
                    record(name, value, pd.concat([current, denominator]), source, "y_O/y_O_minus_1" if "mom" in name else "y_O/y_O_minus_12", "missing_input_or_zero_denominator")
                record("expense_n_history_observations", len(finite), finite, source, "count_available_finite_expense_months")

        if cutoff not in streams:
            prefix = errors.loc[errors.available_at.le(cutoff) & errors.forecast_available_at.le(cutoff) & errors.period.le(period)].copy()
            # Skip any later raw rows before invoking E04. Explicitly unavailable
            # or absent observation months remain placeholders with concealed y.
            streams[cutoff] = {}
            for method in DETECTORS:
                prepared = prefix[["series_id", "observation_period", "y_true", "y_pred"]].copy()
                prep = dict(preparation, as_of=cutoff.tz_convert("Europe/Moscow").tz_localize(None))
                stream = run_stream(prepared, method, selected[method], prep)
                streams[cutoff][method] = {identifier: part.copy() for identifier, part in stream.groupby("series_id", sort=False)}
        for method in DETECTORS:
            stem = "online_" + method.lower() + "_"
            stream = streams[cutoff][method].get(uid)
            if stream is None or stream.empty:
                phase, count, current_score, current_alarm, last_score = "not_started", 0, np.nan, np.nan, np.nan
            else:
                latest = stream.iloc[-1]
                is_current = pd.Period(latest.observation_period, freq="M") == period
                phase = str(latest.phase) if is_current else "missing_current_month"
                count = int(latest.calibration_count)
                current_score = float(latest.score) if is_current else np.nan
                current_alarm = float(latest.is_alarm) if is_current and latest.phase == "monitoring" else np.nan
                observed_scores = stream.loc[stream.score.notna()]
                last_score = np.nan if observed_scores.empty else float(observed_scores.iloc[-1].score)
            row[stem + "phase"] = phase
            dependencies = error.loc[error.value.notna()]
            age = np.nan if dependencies.empty else period.ordinal-dependencies.period.max().ordinal
            for field, value in (("score", current_score), ("last_score", last_score), ("is_alarm", current_alarm),
                                 ("calibration_count", count), ("ready", int(count >= preparation["warmup_observations"])),
                                 ("state_age_months", age)):
                record(stem+field, value, dependencies, "E04_prefix_detector", "saved_h1_y_true,y_pred;causal_prefix_warmup_and_stream", "no_available_monitoring_state" if field in ("score", "last_score", "is_alarm") else "no_finite_residual_history")
        result_rows.append(row)
    result = samples.copy()
    added = pd.DataFrame(result_rows, index=samples.index, columns=FEATURE_COLUMNS+PHASE_COLUMNS)
    for name in FEATURE_COLUMNS:
        result[name] = added[name].astype(float)
        result[name + "_missing"] = added[name].isna()
    for name in PHASE_COLUMNS:
        result[name] = added[name]
    lineage = pd.DataFrame(provenance, columns=PROVENANCE_COLUMNS)
    for column in ("as_of_utc", "max_available_at"):
        lineage[column] = pd.to_datetime(lineage[column], utc=True)
    return result, lineage


def build_origin_macro_features(samples: pd.DataFrame, macro_table: pd.DataFrame,
                                source_ids: Mapping[str, str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reuse E05c A-only annual forecasts for O's current calendar year.

    ``target_period`` is explicitly the origin's month. It never uses the
    early-warning horizon k or an expense-forecast H. These published annual
    national forecasts remain annual forecasts, never actual monthly outcomes.
    Source IDs retain overlap with the E05c registry rather than new-source claims.
    """
    queries = _queries(samples)
    adapted = pd.DataFrame({"municipality_id": queries._uid.to_numpy(),
        "as_of_date": [stamp.tz_convert("Europe/Moscow").tz_localize(None) for stamp in queries._as_of],
        "target_period": queries._period.astype(str).to_numpy()}, index=samples.index)
    values, lineage = prepare_forecast_features(adapted, macro_table, source_ids=source_ids)
    result = samples.copy()
    for column in values:
        name = "macro_current_year_" + column
        if name in result:
            raise ValueError("Macro feature column already present")
        result[name] = values[column]
    lineage["forecast_origin"] = np.repeat(samples.forecast_origin.to_numpy(), len(MACRO_INDICATORS))
    lineage["target_rule"] = "annual_forecast_for_current_calendar_year_of_O_not_H_or_k"
    lineage["source_reused_from_E05c"] = lineage.source_id.ne("") & (source_ids is not None)
    lineage["as_of_utc"] = pd.to_datetime(lineage.as_of_date.map(_instant), utc=True)
    lineage["max_available_at"] = pd.to_datetime(lineage.available_at.map(_instant), utc=True)
    if (lineage.max_available_at.notna() & lineage.max_available_at.gt(lineage.as_of_utc)).any():
        raise AssertionError("Macro dependency exceeds its own origin")
    return result, lineage


def feature_dictionary() -> pd.DataFrame:
    """Definitions for expense, saved residual, prefix states and macro fields."""
    rows = []
    for name in FEATURE_COLUMNS:
        source = "expense_history" if name.startswith("expense") else "saved_E04_h1_residual" if name.startswith("residual") else "E04_prefix_detector"
        definition = name.replace("_", " ")
        if name.endswith("mean_3m"):
            definition += "; strict mean of O,O-1,O-2, all three calendar inputs required"
        if name.endswith("mom_ratio"):
            definition += "; y[O]/y[O-1], positive denominator required"
        if name.endswith("yoy_ratio"):
            definition += "; y[O]/y[O-12], positive denominator required"
        if name.endswith("last_score"):
            definition += "; latest observed monitoring score, possibly stale; state_age_months retained"
        if name.endswith("ready"):
            definition += "; saved E04 warmup number (four in the fixed experiment) of available finite residuals has been observed; no future calibration"
        if "current_value" in name:
            definition += "; value at O's calendar month, never substituted by a stale value"
        if "last_value" in name:
            definition += "; most recent available finite value, explicitly paired with last_age_months"
        if name.endswith("is_alarm"):
            definition += "; E04 emitted alarm on the current monitoring month, after cooldown; NaN if not monitoring"
        if name.endswith("score") and not name.endswith("last_score"):
            definition += "; current monitoring score, NaN during warmup, missing months, or an absent current month"
        if name.endswith("state_age_months"):
            definition += "; calendar months since most recent available finite residual, including warmup"
        unit = ("rubles" if "value" in name or "mean_3m" in name else "months" if "age_months" in name
                else "count" if name.endswith(("observed_months_3m", "n_history_observations", "calibration_count"))
                else "binary 0/1" if name.endswith(("ready", "is_alarm")) else "ratio" if name.endswith("ratio") else "detector score")
        row = dict(feature=name, definition=definition, source=source, unit=unit,
            availability_rule="All dependency availability timestamps <= own forecast_origin; month-end midnight Moscow L0 expense-vintage assumption",
            selection_rule="Prepared without feature selection; full-history diagnostics are descriptive, future selection must use eligible training rows only")
        rows.append(row)
        rows.append(dict(row, feature=name+"_missing", definition="True exactly when " + name + " is NaN", unit="boolean"))
    for name in PHASE_COLUMNS:
        rows.append(dict(feature=name, definition="Current prefix state: not_started, warmup, monitoring, missing, or missing_current_month; descriptive readiness indicator, never a hindsight state",
            source="E04_prefix_detector", unit="categorical",
            availability_rule="Computed from exactly the same own-origin residual prefix as detector scores",
            selection_rule="Fixed definition; no full-history feature selection"))
    for row in macro_feature_dictionary(MACRO_INDICATORS):
        rows.append(dict(feature="macro_current_year_"+row["name"], definition=row["source_definition"]+" "+row["formula"],
            source=row["source"], unit=row["unit"], availability_rule=row["availability_rule"]+"; current calendar year of O, no future H/k target substitution",
            selection_rule="Fixed E05c definitions; national copies across MO are not independent macro observations"))
    return pd.DataFrame(rows)


def audit_feature_columns(features: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Describe constant/absent/identical columns; never select or drop them.

    Supply the eligible training subset for a prospective selection audit. The
    default full table is explicitly descriptive. Per-origin vectors are counted
    once per unique O, and national copies across MO remain one temporal vector.
    """
    requested = list(columns)
    if len(requested) != len(set(requested)) or not set(requested).issubset(features):
        raise ValueError("Audit requires unique existing feature names")
    if "forecast_origin" not in features:
        raise ValueError("Audit requires forecast_origin to distinguish temporal vectors")
    rows, previous = [], []
    n_origins = features.forecast_origin.nunique()
    if requested:
        unique_temporal_vectors = features[requested].drop_duplicates().shape[0]
        repeated_national = all(len(frame[requested].drop_duplicates()) <= 1
                                for _, frame in features.groupby("forecast_origin", sort=False))
        if not repeated_national:
            unique_temporal_vectors = np.nan  # local vectors are not a single national vector for O
    else:
        unique_temporal_vectors, repeated_national = 0, True
    for name in requested:
        series = features[name]
        numeric = pd.to_numeric(series, errors="raise")
        finite = np.isfinite(numeric.to_numpy(dtype=float, na_value=np.nan))
        all_missing = bool(series.isna().all())
        distinct_finite = int(pd.Series(numeric.to_numpy()[finite]).nunique())
        finite_constant = bool(finite.any() and distinct_finite == 1)
        exact_constant = bool(series.nunique(dropna=False) == 1)
        duplicate_of = ""
        for other in previous:
            # Numeric flags and floating 0/1 are the same column values for a
            # future numeric model; NaN positions must also be exactly equal.
            if np.array_equal(numeric.to_numpy(dtype=float, na_value=np.nan),
                              pd.to_numeric(features[other]).to_numpy(dtype=float, na_value=np.nan), equal_nan=True):
                duplicate_of = other
                break
        previous.append(name)
        per_origin = features.groupby("forecast_origin", sort=False)[name].nunique(dropna=False)
        temporal = features[["forecast_origin", name]].drop_duplicates("forecast_origin") if per_origin.le(1).all() else None
        rows.append(dict(feature=name, rows=len(features), origin_dates=n_origins,
            nonmissing=int(series.notna().sum()), missing=int(series.isna().sum()), finite=int(finite.sum()),
            distinct_nonmissing=int(series.nunique()), distinct_finite=distinct_finite,
            all_missing=all_missing, finite_constant=finite_constant, exact_constant_including_missing=exact_constant,
            exact_duplicate_of=duplicate_of, duplicate_group=duplicate_of or name,
            identical_across_municipalities_within_origin=bool(per_origin.le(1).all()),
            origin_dates_counted_once=n_origins if per_origin.le(1).all() else np.nan,
            distinct_temporal_values=temporal[name].nunique(dropna=False) if temporal is not None else np.nan,
            unique_temporal_feature_vectors=unique_temporal_vectors,
            all_requested_vectors_identical_within_origin=repeated_national,
            interpretation="Descriptive audit of supplied rows; no column removed; full history is not a training-only selection rule"))
    return pd.DataFrame(rows)
