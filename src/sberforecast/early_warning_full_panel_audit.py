"""Reproducible E07b state/feature/news summaries; no model or row selection.

The summaries distinguish causal state from retrospective regime membership,
numeric news values from missing flags, and repeated municipal copies from
distinct vectors. News zeros describe a bounded collected corpus only.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from .early_warning_labels import _date_cutoff
from .news_features import FEATURE_COLUMNS as NEWS_VALUE_COLUMNS


NEWS_INFORMATION_COLUMN = "news_count_90d"
if NEWS_INFORMATION_COLUMN not in NEWS_VALUE_COLUMNS:
    raise AssertionError("The frozen news schema changed")
KEYS = ("municipality_id", "forecast_origin")


def _require(frame: pd.DataFrame, names, label: str) -> None:
    missing = set(names) - set(frame)
    if missing:
        raise ValueError(f"Missing {label} fields: {sorted(missing)}")


def _keyed(frame: pd.DataFrame) -> pd.DataFrame:
    _require(frame, KEYS, "sample key")
    result = frame.copy()
    if result.municipality_id.isna().any() or result.municipality_id.astype(str).str.strip().eq("").any():
        raise ValueError("Sample municipality IDs must be nonempty")
    result["municipality_id"] = result.municipality_id.astype(str).str.strip()
    result["forecast_origin"] = result.forecast_origin.map(_date_cutoff)
    return result


def _boolean(values: pd.Series, name: str) -> pd.Series:
    if values.isna().any() or not values.map(lambda value: isinstance(value, (bool, np.bool_))).all():
        raise ValueError(f"{name} must be an explicit nonmissing boolean")
    return values.astype(bool)


def _group_columns(wide: pd.DataFrame, groups: Mapping) -> dict[str, tuple]:
    assigned = set()
    result = {}
    for group, names in groups.items():
        names = tuple(names)
        if len(set(names)) != len(names) or assigned & set(names):
            raise ValueError("Candidate feature groups must not contain duplicate assignments")
        _require(wide, names, f"group {group} feature")
        assigned.update(names)
        result[group] = names
    return result


def build_state_counts(cases: pd.DataFrame) -> pd.DataFrame:
    """All MO/O/k rows: causal and retrospective states counted separately."""
    fields = ("k", "known_active_at_origin", "state_uncertain", "retrospective_inside_regime_at_origin", "eligible_at_origin")
    _require(cases, fields, "state case")
    data = _keyed(cases)
    if data.duplicated(list(KEYS) + ["k"]).any():
        raise ValueError("Duplicate municipality/origin/horizon state row")
    if not data.k.isin([1, 3]).all():
        raise ValueError("State audits require the fixed k=1/3 horizons")
    for field in fields[1:]:
        data[field] = _boolean(data[field], field)
    records = []
    for (k, origin), rows in data.groupby(["k", "forecast_origin"], sort=True):
        active, inside, uncertain = rows.known_active_at_origin, rows.retrospective_inside_regime_at_origin, rows.state_uncertain
        records.append(dict(k=int(k), forecast_origin=str(origin.date()), cases=len(rows),
            municipalities=rows.municipality_id.nunique(), eligible_cases=int(rows.eligible_at_origin.sum()),
            known_active_cases=int(active.sum()), state_uncertain_cases=int(uncertain.sum()),
            retrospective_inside_cases=int(inside.sum()),
            known_active_and_retrospective_inside_cases=int((active & inside).sum()),
            retrospective_only_cases=int((inside & ~active).sum()), known_active_only_cases=int((active & ~inside).sum()),
            known_active_uncertain_cases=int((active & uncertain).sum()), rows_filtered=False,
            retrospective_use="audit_only_not_feature_or_causal_risk_filter"))
    return pd.DataFrame(records, columns=("k", "forecast_origin", "cases", "municipalities", "eligible_cases",
        "known_active_cases", "state_uncertain_cases", "retrospective_inside_cases", "known_active_and_retrospective_inside_cases",
        "retrospective_only_cases", "known_active_only_cases", "known_active_uncertain_cases", "rows_filtered", "retrospective_use"))


def _numbers(rows: pd.DataFrame, columns: tuple) -> dict:
    finite = missing = nonmissing = all_missing = constant = varying = strict_varying = 0
    numeric = {}
    for name in columns:
        values = pd.to_numeric(rows[name], errors="raise")
        array = values.to_numpy(dtype=float, na_value=np.nan)
        finite_values = array[np.isfinite(array)]
        numeric[name] = pd.Series(array, index=rows.index)
        finite += len(finite_values)
        missing += int(np.isnan(array).sum())
        nonmissing += int(values.notna().sum())
        unique = len(np.unique(finite_values))
        all_missing += int(np.isnan(array).all())
        constant += int(unique == 1)
        varying += int(unique > 1)
        strict_varying += int(values.nunique(dropna=False) > 1)
    duplicates = 0
    retained = []
    if len(rows):
        for name in columns:
            if any(numeric[name].equals(numeric[previous]) for previous in retained):
                duplicates += 1
            else:
                retained.append(name)
    cells = len(rows) * len(columns)
    return dict(feature_columns=len(columns), cells=cells, finite_cells=finite,
                missing_cells=missing, nonfinite_nonmissing_cells=nonmissing - finite,
                finite_cell_coverage=finite / cells if cells else np.nan,
                all_missing_columns=all_missing, observed_constant_columns=constant,
                observed_varying_columns=varying, strict_varying_columns_including_missing=strict_varying,
                exact_duplicate_columns_retained=duplicates)


def build_feature_summary(wide: pd.DataFrame, coverage: pd.DataFrame, groups: Mapping) -> pd.DataFrame:
    """Group×role coverage and independently recomputed dependency violations.

    Value columns and ``*_missing`` flags have separate cell denominators.
    Coverage records retain their exact/upper-bound audit description; a zero
    violation count does not verify historical vintages or archive immutability.
    """
    keyed = _keyed(wide)
    if keyed.duplicated(list(KEYS)).any():
        raise ValueError("Duplicate feature row keys")
    columns = _group_columns(wide, groups)
    _require(coverage, ("forecast_origin", "feature", "as_of_utc", "max_dependency_available_at", "dependency_violations"), "feature coverage")
    audit = coverage.copy()
    audit["forecast_origin"] = audit.forecast_origin.map(_date_cutoff)
    if audit.duplicated(["forecast_origin", "feature"]).any():
        raise ValueError("Duplicate feature coverage record")
    as_of = pd.to_datetime(audit.as_of_utc, utc=True, errors="raise", format="mixed")
    maximum = pd.to_datetime(audit.max_dependency_available_at, utc=True, errors="raise", format="mixed")
    if as_of.isna().any():
        raise ValueError("Coverage origins require an explicit as_of_utc")
    expected_as_of = audit.forecast_origin.map(lambda origin: origin.tz_localize("Europe/Moscow").tz_convert("UTC"))
    reported = pd.to_numeric(audit.dependency_violations, errors="raise")
    if reported.isna().any() or (~np.isfinite(reported)).any() or reported.lt(0).any():
        raise ValueError("Dependency violation counts must be finite and nonnegative")
    audit["_reported"] = reported
    audit["_as_of_mismatch"] = as_of.ne(expected_as_of)
    audit["_computed_future"] = maximum.notna() & maximum.gt(expected_as_of)
    audit["_violation_record"] = audit._reported.gt(0) | audit._computed_future
    records = []
    origins = set(keyed.forecast_origin)
    for group, names in columns.items():
        for role in ("value", "missing_flag"):
            selected_names = tuple(name for name in names if name.endswith("_missing") == (role == "missing_flag"))
            part = audit.loc[audit.feature.isin(selected_names) & audit.forecast_origin.isin(origins)]
            expected = {(origin, name) for origin in origins for name in selected_names}
            actual = set(zip(part.forecast_origin, part.feature))
            missing_records = len(expected - actual)
            records.append(dict(group=group, role=role, rows=len(wide), municipalities=keyed.municipality_id.nunique(),
                forecast_origins=len(origins), **_numbers(wide, selected_names),
                dependency_coverage_records=len(part), missing_dependency_coverage_records=missing_records,
                reported_dependency_violations=int(part._reported.sum()),
                computed_future_dependency_records=int(part._computed_future.sum()),
                coverage_origin_timestamp_mismatches=int(part._as_of_mismatch.sum()),
                dependency_violation_records=int(part._violation_record.sum()),
                temporal_dependency_check_passed=missing_records == 0 and not part._violation_record.any() and not part._as_of_mismatch.any(),
                dependency_audit_kinds="|".join(sorted(part.dependency_audit.dropna().astype(str).unique())) if "dependency_audit" in part else "not_supplied",
                selection_applied=False))
    return pd.DataFrame(records)


def _vectors(rows: pd.DataFrame, columns: tuple) -> tuple[int, int]:
    if rows.empty or not columns:
        return 0, 0
    unique = len(rows[list(columns)].drop_duplicates())
    maximum = max(len(part[list(columns)].drop_duplicates()) for _, part in rows.groupby("forecast_origin", sort=False))
    return unique, maximum


def build_news_summary(cases: pd.DataFrame, wide: pd.DataFrame, groups: Mapping) -> pd.DataFrame:
    """k×all/train/test news diagnostics on the already frozen split.

    Equal feature vectors copied to many MO count once, including when they
    recur on multiple origins. Spatial variation is disclosed separately.
    No diagnostic varying/duplicate result changes the candidate list.
    """
    _require(cases, ("k", "split"), "news case")
    rows = _keyed(cases)
    features = _keyed(wide)
    if rows.duplicated(list(KEYS) + ["k"]).any() or features.duplicated(list(KEYS)).any():
        raise ValueError("News audit keys must be unique within each horizon")
    if not rows.k.isin([1, 3]).all() or not rows.split.isin(["train", "test", "excluded"]).all():
        raise ValueError("News scopes require fixed k=1/3 and a saved temporal split")
    columns = _group_columns(wide, groups)
    if "D" not in columns or NEWS_INFORMATION_COLUMN not in columns["D"]:
        raise ValueError("News group D requires the frozen news_count_90d field")
    values = tuple(name for name in columns["D"] if not name.endswith("_missing"))
    flags = tuple(name for name in columns["D"] if name.endswith("_missing"))
    records = []
    for k in (1, 3):
        for scope in ("all", "train", "test"):
            mask = rows.k.eq(k) & (True if scope == "all" else rows.split.eq(scope))
            query = rows.loc[mask, list(KEYS)]
            selected = query.merge(features[list(KEYS) + list(columns["D"])], on=list(KEYS), how="left", validate="one_to_one", indicator=True)
            if selected._merge.ne("both").any():
                raise ValueError("Some warning-case keys are absent from the feature table")
            value_stats, flag_stats = _numbers(selected, values), _numbers(selected, flags)
            vector_count, spatial_maximum = _vectors(selected, values)
            full_count, full_spatial_maximum = _vectors(selected, columns["D"])
            info = pd.to_numeric(selected[NEWS_INFORMATION_COLUMN], errors="raise")
            informative = np.isfinite(info) & info.gt(0)
            records.append(dict(k=k, scope=scope, cases=len(query), municipalities=query.municipality_id.nunique(),
                forecast_origins=query.forecast_origin.nunique(), value_columns=len(values), missing_flag_columns=len(flags),
                varying_value_columns=value_stats["observed_varying_columns"],
                strict_varying_value_columns_including_missing=value_stats["strict_varying_columns_including_missing"],
                varying_missing_flag_columns=flag_stats["observed_varying_columns"],
                observed_constant_value_columns=value_stats["observed_constant_columns"], all_missing_value_columns=value_stats["all_missing_columns"],
                retained_value_duplicate_columns=value_stats["exact_duplicate_columns_retained"],
                retained_missing_flag_duplicate_columns=flag_stats["exact_duplicate_columns_retained"],
                finite_news_value_cells=value_stats["finite_cells"], news_value_cell_coverage=value_stats["finite_cell_coverage"],
                unique_news_value_vectors=vector_count, unique_news_full_vectors=full_count,
                maximum_spatial_value_vectors_per_origin=spatial_maximum, maximum_spatial_full_vectors_per_origin=full_spatial_maximum,
                one_common_news_vector_per_origin=full_spatial_maximum <= 1 and len(query) > 0,
                unique_temporal_vectors=full_count if full_spatial_maximum <= 1 else np.nan,
                origins_with_news_information=selected.loc[informative, "forecast_origin"].nunique(),
                cases_with_news_information=int(informative.sum()),
                origins_with_unknown_news_count=selected.loc[~np.isfinite(info), "forecast_origin"].nunique(),
                news_information_field=NEWS_INFORMATION_COLUMN, scope_empty=query.empty,
                selection_applied=False, source_archive_complete=False,
                zero_interpretation="no_admitted_event_in_bounded_corpus_not_no_real_event"))
    return pd.DataFrame(records)
