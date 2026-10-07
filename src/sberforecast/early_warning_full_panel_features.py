"""E07b bounded-memory feature assembly, without labels or feature selection.

Frozen E07a functions remain responsible for causal expense/residual histories,
selected E04 prefix states, and A-only annual macro forecasts. News uses the
frozen first-known document/version rules. The additional changes/volatility
require finite inputs from immediately preceding calendar months, never fills.
All input sample keys survive, including ineligible, stale and absent series.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import early_warning_features as history
from . import news_features as news
from .macro_forecast_features import MACRO_INDICATORS, forecast_feature_names


EXTRA_COLUMNS = (
    "expense_change_1m", "residual_previous_month_value", "residual_change_1m",
    "expense_std_3m", "residual_std_3m", "expense_cv_3m",
)
MACRO_COLUMNS = tuple("macro_current_year_" + name for name in forecast_feature_names(MACRO_INDICATORS))
BASE_HISTORY_COLUMNS = history.EXPENSE_COLUMNS + history.RESIDUAL_COLUMNS


def _with_flags(columns: tuple[str, ...]) -> tuple[str, ...]:
    return columns + tuple(name + "_missing" for name in columns)


FEATURE_GROUPS = {
    "A": _with_flags(BASE_HISTORY_COLUMNS + EXTRA_COLUMNS),
    "B": _with_flags(history.DETECTOR_COLUMNS),
    "C": MACRO_COLUMNS,
    "D": news.FEATURE_COLUMNS + news.MISSING_COLUMNS,
}
NUMERIC_FEATURE_COLUMNS = tuple(name for group in FEATURE_GROUPS.values() for name in group)
GROUP_NAMES = dict(A="History", B="DetectorState", C="Macro", D="News")


def feature_groups() -> dict[str, tuple[str, ...]]:
    """Fixed candidate groups; selection/imputation/scaling belong to train only."""
    return dict(FEATURE_GROUPS)


def feature_dictionary() -> pd.DataFrame:
    """One row per numeric candidate or diagnostic phase, with fixed group IDs."""
    base = history.feature_dictionary().copy()
    definitions = {
        "expense_change_1m": ("y[O]-y[O-1]; both available finite calendar months required", "rubles", "expense_history"),
        "residual_previous_month_value": ("Saved causal h1 y_true-y_pred at calendar O-1; no stale substitution", "rubles", "saved_E04_h1_residual"),
        "residual_change_1m": ("e[O]-e[O-1]; both available finite calendar residuals required", "rubles", "saved_E04_h1_residual"),
        "expense_std_3m": ("Population standard deviation (ddof=0) of y[O-2],y[O-1],y[O]; all three finite and available", "rubles", "expense_history"),
        "residual_std_3m": ("Population standard deviation (ddof=0) of e[O-2],e[O-1],e[O]; all three finite and available", "rubles", "saved_E04_h1_residual"),
        "expense_cv_3m": ("expense_std_3m/mean(y[O-2:O]); all three finite and available, strictly positive mean required", "ratio", "expense_history"),
    }
    rows = []
    for name, (definition, unit, source) in definitions.items():
        row = dict(feature=name, definition=definition, unit=unit, source=source,
            availability_rule="Every dependency available_at <= own O; E01 month-end midnight Moscow L=0 expense-vintage assumption",
            selection_rule="Fixed candidate definition; remove constants/missing/duplicates on training rows only")
        rows.extend([row, dict(row, feature=name+"_missing", definition="True exactly when "+name+" is NaN", unit="boolean")])
    result = pd.concat([base, pd.DataFrame(rows), news.feature_dictionary()], ignore_index=True, sort=False)
    mapping = {name: group for group, names in FEATURE_GROUPS.items() for name in names}
    mapping.update({name: "B" for name in history.PHASE_COLUMNS})
    result["group"] = result.feature.map(mapping)
    result["group_name"] = result.group.map(GROUP_NAMES)
    result["model_candidate"] = result.feature.isin(NUMERIC_FEATURE_COLUMNS)
    result["feature_role"] = np.where(result.model_candidate, "numeric_candidate", "diagnostic_phase")
    result.loc[result.group.eq("D"), "source"] = "Saved E07a admitted news documents; bounded corpus, archive-trust assumption retained"
    result["selection_rule"] = "Fixed candidates; filtering, imputation and scaling fit on eligible training rows only, never full panel/test"
    if result.feature.duplicated().any() or result.group.isna().any():
        raise AssertionError("Feature dictionary and fixed groups disagree")
    return result


def _extra_features(observations: pd.DataFrame, residuals: pd.DataFrame,
                    samples: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    queries = history._queries(samples)
    expenses, errors = history._observations(observations, 0), history._residuals(residuals, 0)
    expense_groups = {uid: part for uid, part in expenses.groupby("_uid", sort=False)}
    error_groups = {uid: part for uid, part in errors.groupby("_uid", sort=False)}
    empty_expense, empty_error = expenses.iloc[:0], errors.iloc[:0]
    rows, records = [], []
    for position, query in enumerate(queries.to_dict("records")):
        uid, origin, period = query["_uid"], query["_as_of"], query["_period"]
        expense = expense_groups.get(uid, empty_expense)
        expense = expense.loc[expense.available_at.le(origin) & expense.period.le(period)]
        error = error_groups.get(uid, empty_error)
        error = error.loc[error.available_at.le(origin) & error.forecast_available_at.le(origin) & error.period.le(period)]
        values = {}

        def record(name, value, dependencies, source, fields, missing_reason):
            value = float(value)
            if np.isinf(value):
                raise ValueError("Feature calculation exceeded finite numeric range: " + name)
            maximum = dependencies.available_at.max() if len(dependencies) else pd.NaT
            if pd.notna(maximum) and maximum > origin:
                raise AssertionError("Future dependency in calendar dynamics")
            values[name] = value
            records.append(dict(sample_position=position, municipality_id=uid,
                forecast_origin=query["forecast_origin"], as_of_utc=origin, feature=name, value=value,
                max_available_at=maximum, dependency_periods=json.dumps([str(month) for month in dependencies.index]),
                source_fields=fields, source=source, missing=bool(pd.isna(value)),
                missing_reason=missing_reason if pd.isna(value) else "",
                availability_assumption="E01_L0_month_end_midnight_Moscow_no_verified_expense_vintages",
                parameters_assumption=""))

        for prefix, source, table in (("expense", "expense_history", expense), ("residual", "saved_E04_h1_residual", error)):
            pair = history._lookup(table, [period, period-1])
            change = pair.value.iloc[0]-pair.value.iloc[1] if pair.value.notna().all() else np.nan
            record(prefix+"_change_1m", change, pair, source, "current_and_previous_calendar_value", "requires_both_available_finite_calendar_months")
            if prefix == "residual":
                record("residual_previous_month_value", pair.value.iloc[1], pair.iloc[1:], source,
                       "saved_h1_y_true,y_pred_at_O_minus_1", "previous_calendar_residual_missing_or_unavailable")
            recent = history._lookup(table, [period-2, period-1, period])
            complete = recent.value.notna().all()
            deviation = float(np.std(recent.value.to_numpy(dtype=float), ddof=0)) if complete else np.nan
            record(prefix+"_std_3m", deviation, recent, source, "three_calendar_values_population_std_ddof_0", "requires_all_three_available_finite_calendar_months")
            if prefix == "expense":
                mean = float(recent.value.mean()) if complete else np.nan
                variation = deviation/mean if complete and mean > 0 else np.nan
                record("expense_cv_3m", variation, recent, source, "population_std_3m_over_mean_3m", "missing_calendar_inputs_or_nonpositive_mean")
        rows.append(values)
    wide = pd.DataFrame(rows, index=samples.index, columns=EXTRA_COLUMNS, dtype=float)
    for name in EXTRA_COLUMNS:
        wide[name+"_missing"] = wide[name].isna()
    lineage = pd.DataFrame(records, columns=history.PROVENANCE_COLUMNS)
    for name in ("as_of_utc", "max_available_at"):
        lineage[name] = pd.to_datetime(lineage[name], utc=True)
    return wide, lineage


def _update_maxima(maxima: dict, provenance: pd.DataFrame) -> None:
    if provenance.empty:
        return
    available = provenance.max_available_at.notna()
    if provenance.loc[available, "max_available_at"].gt(provenance.loc[available, "as_of_utc"]).any():
        raise AssertionError("Prefix lineage contains a future dependency")
    summary = provenance.groupby(["as_of_utc", "feature"], sort=False).max_available_at.max()
    for key, timestamp in summary.items():
        previous = maxima.get(key, pd.NaT)
        if pd.notna(timestamp) and (pd.isna(previous) or timestamp > previous):
            maxima[key] = timestamp


def _macro_maxima(maxima: dict, provenance: pd.DataFrame) -> None:
    for indicator in MACRO_INDICATORS:
        part = provenance.loc[provenance.indicator.eq(indicator)]
        for origin, stamp in part.groupby("as_of_utc", sort=False).max_available_at.max().items():
            for name in MACRO_COLUMNS:
                if name.startswith("macro_current_year_"+indicator):
                    key = (origin, name)
                    previous = maxima.get(key, pd.NaT)
                    if pd.notna(stamp) and (pd.isna(previous) or stamp > previous):
                        maxima[key] = stamp


def _news_maxima(maxima: dict, documents: pd.DataFrame, samples: pd.DataFrame) -> None:
    """Conservative upper bound, not exact per-window/per-feature news lineage."""
    validated = news._validate_events(documents)
    queries = news._validate_samples(samples)
    for origin, cases in queries.groupby("_origin", sort=False):
        _, known = news._known_events(validated, origin)
        national = known.geography_level.eq("national")
        regional = known.geography_level.eq("region") & known._region.isin(cases._region)
        keys = set(zip(cases._municipality, cases._region))
        municipal = known.geography_level.eq("municipality") & pd.Series(
            [(uid, region) in keys for uid, region in zip(known._municipality, known._region)], index=known.index)
        maximum = known.loc[national | regional | municipal, "available_at"].max()
        if pd.notna(maximum) and maximum > origin:
            raise AssertionError("Future news availability in scoped upper bound")
        for name in FEATURE_GROUPS["D"]:
            maxima[(origin, name)] = maximum


def _coverage(wide: pd.DataFrame, dictionary: pd.DataFrame, maxima: dict) -> pd.DataFrame:
    queries = history._queries(wide[["municipality_id", "forecast_origin"]])
    rows = []
    for origin, positions in queries.groupby("_as_of", sort=False).indices.items():
        part = wide.iloc[positions]
        for definition in dictionary.to_dict("records"):
            name = definition["feature"]
            values = part[name]
            numeric = pd.to_numeric(values, errors="raise") if definition["model_candidate"] else None
            finite = int(np.isfinite(numeric.to_numpy(dtype=float, na_value=np.nan)).sum()) if numeric is not None else np.nan
            parent = name[:-8] if name.endswith("_missing") else name
            if name in history.PHASE_COLUMNS:
                parent = name[:-5]+"calibration_count"
            maximum = maxima.get((origin, name), maxima.get((origin, parent), pd.NaT))
            rows.append(dict(forecast_origin=part.forecast_origin.iloc[0], as_of_utc=origin,
                feature=name, group=definition["group"], feature_role=definition["feature_role"],
                dtype=str(values.dtype), n_rows=len(part), nonmissing=int(values.notna().sum()),
                missing=int(values.isna().sum()), missing_share=float(values.isna().mean()), finite=finite,
                n_unique_nonmissing=int(values.nunique()), all_missing=bool(values.isna().all()),
                finite_constant=bool(numeric is not None and finite > 0 and numeric[np.isfinite(numeric)].nunique() == 1),
                max_dependency_available_at=maximum, dependency_violations=int(pd.notna(maximum) and maximum > origin),
                dependency_audit="conservative_scoped_first_known_snapshot_upper_bound" if definition["group"] == "D" else "exact_parent_feature_dependency_maximum_across_MO",
                availability_assumption="E01_expenses_L0_unverified_vintages;fixed_E04_selected_params;official_archive_trust_for_admitted_macro_news"))
    columns = ("forecast_origin", "as_of_utc", "feature", "group", "feature_role", "dtype", "n_rows", "nonmissing", "missing", "missing_share", "finite", "n_unique_nonmissing", "all_missing", "finite_constant", "max_dependency_available_at", "dependency_violations", "dependency_audit", "availability_assumption")
    result = pd.DataFrame(rows, columns=columns)
    for name in ("as_of_utc", "max_dependency_available_at"):
        result[name] = pd.to_datetime(result[name], utc=True)
    return result


def _provenance_directory(config: Mapping) -> Path | None:
    target = config.get("feature_assembly", {}).get("provenance_dir")
    if target is None:
        return None
    if not config.get("output_dir"):
        raise ValueError("Disk provenance requires the new experiment output_dir")
    root = Path(config["output_dir"]).resolve()
    supplied = Path(target)
    directory = supplied.resolve() if supplied.is_absolute() else (root / supplied).resolve()
    if directory == root or not directory.is_relative_to(root):
        raise ValueError("Provenance directory must be a child of the new output_dir")
    return directory


def _write_batch(directory: Path | None, kind: str, batch: int, provenance: pd.DataFrame,
                 positions: np.ndarray, files: list[dict]) -> None:
    if directory is None:
        return
    path = directory / f"{kind}_batch_{batch:04d}.csv.gz"
    # Never overwrite a persisted experiment or an earlier partial attempt.
    if path.exists():
        raise FileExistsError(path)
    directory.mkdir(parents=True, exist_ok=True)
    stored = provenance.copy()
    stored["sample_position"] = positions[stored.sample_position.to_numpy(dtype=int)]
    stored.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0}, mode="x")
    files.append(dict(kind=kind, batch=batch, path=str(path), rows=len(stored)))


def build_panel_features(
    observations: pd.DataFrame, residuals: pd.DataFrame, samples: pd.DataFrame,
    config: Mapping, selected_parameters: Mapping, macro_table: pd.DataFrame,
    source_ids: Mapping[str, str] | None, news_documents: pd.DataFrame,
    progress: Callable[[dict], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return all sample rows, compact coverage and the fixed dictionary.

    ``runtime.feature_batch_size`` defaults to 64 municipalities. Optional
    ``feature_assembly.provenance_dir`` is relative to the *new* ``output_dir``;
    history and macro lineages are then saved separately for every batch.
    Global ``sample_position`` identifies original input order even for repeated
    DataFrame indices. ``wide.attrs['provenance_files']`` lists these artifacts.
    No full-panel long lineage is retained in memory. The coverage records
    exact parent dependency maxima for history/macro and an explicitly labelled
    scoped first-known snapshot upper bound for news; every maximum is <= O.
    No eligibility, label, future-completeness cohort or selection is applied.
    """
    queries = history._queries(samples)
    history._reject_offline(observations)
    history._reject_offline(residuals)
    history._reject_offline(samples)
    if config.get("data", {}).get("release_lag_months", 0) != 0:
        raise ValueError("E07b keeps E01 L=0 expense availability")
    dictionary = feature_dictionary()
    collision = set(dictionary.feature) & set(samples)
    if collision:
        raise ValueError("Samples already contain candidates: " + ",".join(sorted(collision)))
    batch_size = config.get("runtime", {}).get("feature_batch_size", 64)
    if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
        raise ValueError("feature_batch_size must be a positive integer")
    directory = _provenance_directory(config)
    uid_column = "municipality_id" if "municipality_id" in observations else "series_id"
    if uid_column not in observations or "series_id" not in residuals:
        raise ValueError("Expense/residual inputs require municipality/series IDs")
    obs_ids = observations[uid_column].map(history._id)
    error_ids = residuals.series_id.map(history._id)
    ids = queries._uid.drop_duplicates().tolist()
    batches = (len(ids)+batch_size-1)//batch_size
    # Conservative data-size projection, separate from actual process-RSS.
    source_bytes = sum(frame.memory_usage(deep=True).sum() for frame in (observations, residuals, samples, macro_table, news_documents))
    max_cases = min(len(samples), batch_size*max(1, queries._as_of.nunique()))
    estimate_gib = float((3*source_bytes + len(samples)*(len(dictionary)*16+1024) + max_cases*38*2400 + 256*1024**2)/1024**3)
    if estimate_gib > config.get("runtime", {}).get("memory_stop_gib", 5.5):
        raise MemoryError(f"Projected feature assembly {estimate_gib:.3f} GiB exceeds configured memory stop")
    if progress:
        progress(dict(stage="feature_assembly_start", batches=batches, estimated_memory_gib=estimate_gib))
    parts, positions_all, maxima, files = [], [], {}, []
    for batch, start in enumerate(range(0, len(ids), batch_size)):
        group = set(ids[start:start+batch_size])
        positions = np.flatnonzero(queries._uid.isin(group).to_numpy())
        query = samples.iloc[positions]
        obs = observations.loc[obs_ids.isin(group)]
        errors = residuals.loc[error_ids.isin(group)]
        wide, lineage = history.build_prefix_history_features(obs, errors, query, config, selected_parameters)
        extra, extra_lineage = _extra_features(obs, errors, query)
        for name in extra:
            wide[name] = extra[name]
        lineage = pd.concat([lineage, extra_lineage], ignore_index=True)
        _update_maxima(maxima, lineage)
        _write_batch(directory, "history", batch, lineage, positions, files)
        del lineage, extra_lineage
        macro, macro_lineage = history.build_origin_macro_features(query, macro_table, source_ids)
        for name in MACRO_COLUMNS:
            wide[name] = macro[name]
        _macro_maxima(maxima, macro_lineage)
        _write_batch(directory, "macro", batch, macro_lineage, positions, files)
        del macro, macro_lineage
        parts.append(wide)
        positions_all.extend(positions.tolist())
        if progress:
            progress(dict(stage="history_batch_complete", batch=batch+1, batches=batches,
                          municipalities=len(group), rows_completed=len(positions_all)))
    if parts:
        wide = pd.concat(parts, ignore_index=True)
        order = np.argsort(positions_all, kind="stable")
        wide = wide.iloc[order].copy()
        wide.index = samples.index.copy()
    else:
        # Preserve the complete schema for an empty, explicitly supplied subset.
        wide, _ = history.build_prefix_history_features(observations, residuals, samples, config, selected_parameters)
        for name in _with_flags(EXTRA_COLUMNS)+MACRO_COLUMNS:
            wide[name] = pd.Series(index=samples.index, dtype=bool if name.endswith("_missing") else float)
    del parts
    start = str(pd.Period(config.get("data", {}).get("first_month", "2023-01"), freq="M").start_time.date())
    news_wide = news.build_news_features(news_documents, samples, history_start=start)
    # One block append avoids full-panel per-column fragmentation warnings.
    wide = pd.concat([wide, news_wide[list(FEATURE_GROUPS["D"])]], axis=1)
    _news_maxima(maxima, news_documents, samples)
    coverage = _coverage(wide, dictionary, maxima)
    if coverage.dependency_violations.any():
        raise AssertionError("Feature coverage has a dependency beyond its origin")
    if np.isinf(wide[list(NUMERIC_FEATURE_COLUMNS)].to_numpy(dtype=float)).any():
        raise ValueError("Feature matrix contains an infinite value")
    pd.testing.assert_frame_equal(wide[samples.columns], samples)
    metadata = dict(provenance_files=files, feature_groups={key: list(names) for key, names in FEATURE_GROUPS.items()},
        projected_feature_memory_gib=estimate_gib, no_future_cohort_filter=True,
        provenance_policy="Batched on disk when requested; compact per-origin feature maxima; news maxima are explicitly conservative bounds")
    wide.attrs.update(metadata)
    coverage.attrs.update(metadata)
    if progress:
        progress(dict(stage="feature_assembly_complete", rows=len(wide), coverage_rows=len(coverage),
                      provenance_files=len(files), estimated_memory_gib=estimate_gib))
    return wide, coverage, dictionary
