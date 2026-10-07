"""Batch the frozen E07a weak-label engine and audit calendar feasibility.

Nothing here fits a model or selects a split from classifier performance.
Municipal rows, unique municipal events, warning dates and onset dates are
counted separately. Every original case is retained, including unknown labels.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy

import numpy as np
import pandas as pd

from .early_warning_labels import build_early_warning_labels, _date_cutoff, _month


E07A_WEAK_LABEL_SPEC = {
    "target": "sustained_shift_in_SeasonalNaiveYoY_h1_forecast_error_not_spending_level",
    "error": "y_true_minus_saved_causal_y_pred",
    "seasonality": "saved_SeasonalNaiveYoY_h1_same_month_previous_year_prediction",
    "past_window_months": 4, "confirmation_months": 3,
    "center": "median_four_immediately_preceding_calendar_residuals",
    "scale": "max_1.4826_MAD_0.03_median_absolute_past_prediction_1_ruble",
    "relative_scale_floor": 0.03, "absolute_scale_floor": 1.0,
    "strength_threshold": 3.0,
    "criterion": "all_three_future_residuals_same_signed_deviation_at_least_threshold_times_scale",
    "missing": "require_all_four_past_and_three_confirmation_months_finite_no_fill",
    "merge_gap_months": 2,
    "merge": "adjacent_same_direction_candidates_and_same_direction_unrecovered_regime_are_continuations",
    "recovery_months": 2, "recovery_strength": 1.5,
    "recovery": "two_consecutive_residuals_below_signed_recovery_threshold_using_episode_frozen_baseline",
    "active_risk_exclusion": "confirmed_active_regime_known_on_origin_only",
    "retrospective_inside_regime": "diagnostic_only_never_feature_or_causal_risk_filter",
}
E07A_WARNING_SPEC = {
    "horizons_months": [1, 3],
    "target_window": "(O, O+k] onset_months_not_confirmation_or_news_dates",
    "label_known_at": "uniform_complete_future_window_end_O_plus_k_plus_2_months",
    "positive": "at_least_one_new_weak_event_onset_in_target_window",
    "negative": "full_target_and_confirmation_window_observed_no_onset",
    "right_censoring": "unknown_not_zero",
}
GATE_MINIMUMS = {
    "min_train_positives": 30, "min_test_positives": 10,
    "min_train_positive_onset_dates": 3, "min_test_positive_onset_dates": 2,
}
FRAME_KEYS = ("candidates", "events", "regimes", "origin_states", "cases")


def _gate_thresholds(config: Mapping | None) -> dict:
    expected = dict(GATE_MINIMUMS)
    if config is None:
        return expected
    aliases = {"train_positives_min": "min_train_positives", "test_positives_min": "min_test_positives",
               "train_positive_dates_min": "min_train_positive_onset_dates", "test_positive_dates_min": "min_test_positive_onset_dates"}
    supplied = dict(config.get("feasibility_gate", {}))
    supplied.update({aliases[name]: value for name, value in config.get("gate", {}).items() if name in aliases})
    changed = [name for name, value in expected.items() if supplied.get(name, value) != value]
    if changed:
        raise ValueError(f"The prespecified feasibility gate changed: {changed}")
    return expected


def _split_config(config: Mapping) -> Mapping:
    top = config.get("split")
    nested = config.get("early_warning", {}).get("split")
    fields = ("train_information_cutoff", "test_first_origin", "test_last_origin")
    if top is not None and nested is not None and any(top.get(field) != nested.get(field) for field in fields):
        raise ValueError("Top-level and inherited calendar split settings disagree")
    return top if top is not None else nested or {}


def assert_e07a_lock(config: Mapping) -> None:
    """Reject accidental changes to the inherited definition, before counts."""
    for section, expected in (("weak_label", E07A_WEAK_LABEL_SPEC), ("early_warning", E07A_WARNING_SPEC)):
        actual = config.get(section, {})
        changed = [name for name, value in expected.items() if actual.get(name) != value]
        if changed:
            raise ValueError(f"E07a locked {section} changed or missing: {changed}")
    if config.get("data", {}).get("release_lag_months") != 0:
        raise ValueError("E07a release_lag_months=0 assumption must be retained")
    _gate_thresholds(config)


def _bools(series: pd.Series, field: str) -> pd.Series:
    if series.isna().any() or not series.map(lambda value: isinstance(value, (bool, np.bool_))).all():
        raise ValueError(f"{field} must contain explicit nonmissing booleans")
    return series.astype(bool)


def _concat(parts: list[pd.DataFrame]) -> pd.DataFrame:
    nonempty = [part for part in parts if not part.empty]
    return pd.concat(nonempty, ignore_index=True) if nonempty else (parts[0].iloc[:0].copy() if parts else pd.DataFrame())


def build_panel_labels(
    residuals: pd.DataFrame, samples: pd.DataFrame, config: Mapping,
    progress: Callable[[dict], object] | None = None,
) -> dict[str, pd.DataFrame]:
    """Use the unchanged E07a engine in bounded batches; keep all sample keys.

    ``eligible_at_origin`` is an explicit upstream prefix-eligibility decision,
    never inferred from future label completeness. Eligibility does not remove
    rows or events. Intermediate engine state is limited to at most 128 UIDs;
    the final returned tables still scale with the size of the requested panel.
    """
    assert_e07a_lock(config)
    required = {"municipality_id", "forecast_origin", "eligible_at_origin"}
    if not required.issubset(samples):
        raise ValueError(f"Missing sample fields: {sorted(required - set(samples))}")
    if "series_id" not in residuals:
        raise ValueError("Residuals require series_id")
    if "_panel_sample_order" in samples:
        raise ValueError("Reserved sample ordering field")
    _bools(samples.eligible_at_origin, "eligible_at_origin")
    ordered = samples.copy()
    ordered["_panel_sample_order"] = np.arange(len(ordered))
    sample_ids = ordered.municipality_id.map(lambda value: str(value).strip())
    residual_ids = residuals.series_id.map(lambda value: str(value).strip())
    uids = sorted(set(sample_ids) | set(residual_ids))
    batch_size = config.get("runtime", {}).get("label_batch_size", 64)
    if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)) or not 1 <= batch_size <= 128:
        raise ValueError("label_batch_size must be an integer in [1,128]")
    parts = {key: [] for key in FRAME_KEYS}
    engine_config = deepcopy(config)
    periods = residuals.observation_period.map(_month) if len(residuals) else samples.forecast_origin.map(_month)
    if len(periods):
        engine_config["data"].setdefault("first_month", str(periods.min()))
        engine_config["data"].setdefault("last_month", str(periods.max()))
    for number, begin in enumerate(range(0, len(uids), batch_size), start=1):
        batch_ids = set(uids[begin:begin + batch_size])
        batch = build_early_warning_labels(
            residuals.loc[residual_ids.isin(batch_ids)], ordered.loc[sample_ids.isin(batch_ids)], engine_config,
        )
        for key in FRAME_KEYS:
            parts[key].append(batch[key])
        if progress is not None:
            progress(dict(batch=number, batches=(len(uids) + batch_size - 1) // batch_size,
                          processed_municipalities=min(begin + batch_size, len(uids)), total_municipalities=len(uids),
                          batch_cases=len(batch["cases"])))
    frames = {key: _concat(parts[key]) for key in FRAME_KEYS}
    for key in ("origin_states", "cases"):
        if "_panel_sample_order" in frames[key]:
            sort_fields = ["_panel_sample_order"] + (["k"] if key == "cases" else [])
            frames[key] = frames[key].sort_values(sort_fields, kind="stable").drop(columns="_panel_sample_order").reset_index(drop=True)
    if len(frames["origin_states"]) != len(samples) or len(frames["cases"]) != 2 * len(samples):
        raise AssertionError("The full panel must retain every origin and both warning horizons")
    return frames


def _case_fields(cases: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    required = {"municipality_id", "forecast_origin", "k", "eligible_at_origin", "at_risk",
                "fully_known", "label", "label_known_at", "positive_event_ids"}
    if not required.issubset(cases):
        raise ValueError(f"Missing feasibility case fields: {sorted(required - set(cases))}")
    eligible = _bools(cases.eligible_at_origin, "eligible_at_origin")
    at_risk = _bools(cases.at_risk, "at_risk")
    fully_known = _bools(cases.fully_known, "fully_known")
    if cases.duplicated(["municipality_id", "forecast_origin", "k"]).any():
        raise ValueError("Duplicate full-panel warning case")
    if not cases.k.isin([1, 3]).all():
        raise ValueError("Warning horizons must be k=1/3")
    if (fully_known & (~cases.label.isin([0, 1]) | cases.label_known_at.isna() | cases.label_known_at.astype(str).eq(""))).any():
        raise ValueError("Fully-known labels require a binary value and availability date")
    if (~fully_known & cases.label.notna()).any():
        raise ValueError("Unknown/censored labels must remain missing")
    earliest_known = pd.Series([_month(origin) + int(k) + 2 for origin, k in zip(cases.forecast_origin, cases.k)], index=cases.index)
    actual_known = cases.loc[fully_known, "label_known_at"].map(_date_cutoff)
    earliest_known = earliest_known.loc[fully_known].map(lambda period: period.end_time.normalize())
    if actual_known.lt(earliest_known).any():
        raise ValueError("A label cannot be known before its uniform O+k+2 confirmation-window end")
    return eligible, at_risk, fully_known


def temporal_split(cases: pd.DataFrame, config: Mapping) -> pd.DataFrame:
    """Annotate all rows; admit training truth only when known by cutoff.

    Test origins are strictly after the information cutoff, in the fixed
    calendar window. A fully observed test label is retrospective evaluation
    truth and is never admitted to the earlier training pool.
    """
    eligible, at_risk, fully_known = _case_fields(cases)
    split_config = _split_config(config)
    fields = ("train_information_cutoff", "test_first_origin", "test_last_origin")
    if not all(field in split_config for field in fields):
        raise ValueError("An explicit calendar train cutoff and test window are required")
    cutoff = _date_cutoff(split_config["train_information_cutoff"])
    test_first, test_last = _month(split_config["test_first_origin"]), _month(split_config["test_last_origin"])
    if test_first <= _month(cutoff) or test_last < test_first:
        raise ValueError("Test origins must lie strictly after the training calendar cutoff")
    origins = cases.forecast_origin.map(_date_cutoff)
    origin_months = cases.forecast_origin.map(_month)
    known_at = pd.to_datetime(cases.label_known_at.replace("", pd.NaT), errors="raise")
    training_window = origins.le(cutoff)
    testing_window = origins.gt(cutoff) & origin_months.ge(test_first) & origin_months.le(test_last)
    admitted = eligible & at_risk & fully_known
    train = training_window & admitted & known_at.le(cutoff)
    test = testing_window & admitted
    result = cases.copy()
    result["split"] = np.where(train, "train", np.where(test, "test", "excluded"))
    result["split_calendar_window"] = np.where(training_window, "train", np.where(testing_window, "test", "outside"))
    result["split_exclusion_reason"] = ""
    for mask, reason in (
        (~eligible, "ineligible_at_origin"),
        (eligible & ~at_risk, "confirmed_active_regime_known_at_origin"),
        (eligible & at_risk & ~fully_known, "unknown_label_past_missing_or_future_censored"),
        (training_window & admitted & ~known_at.le(cutoff), "label_not_known_by_train_cutoff"),
        (~training_window & ~testing_window, "outside_fixed_calendar_windows"),
    ):
        result.loc[mask & result.split.eq("excluded") & result.split_exclusion_reason.eq(""), "split_exclusion_reason"] = reason
    result["train_information_cutoff"] = str(cutoff.date())
    result["test_first_origin_period"] = str(test_first)
    result["test_last_origin_period"] = str(test_last)
    return result


def _event_ids(rows: pd.DataFrame) -> set[str]:
    positive = rows.loc[rows.fully_known & rows.label.eq(1), "positive_event_ids"]
    return {value for cell in positive.fillna("").astype(str) for value in cell.split("|") if value}


def _registry(events: pd.DataFrame) -> pd.DataFrame:
    required = {"event_id", "onset_period", "municipality_id"}
    if not required.issubset(events):
        raise ValueError(f"Missing event registry fields: {sorted(required - set(events))}")
    if events.event_id.duplicated().any():
        raise ValueError("Weak event IDs must be unique")
    return events.set_index("event_id", drop=False)


def _counts(rows: pd.DataFrame, registry: pd.DataFrame, population: int) -> dict:
    ids = _event_ids(rows)
    absent = ids - set(registry.index)
    if absent:
        raise ValueError(f"Positive cases refer to unknown event IDs: {sorted(absent)}")
    selected = registry.loc[sorted(ids)] if ids else registry.iloc[:0]
    positives = rows.fully_known & rows.label.eq(1)
    negatives = rows.fully_known & rows.label.eq(0)
    known = int(rows.fully_known.sum())
    event_municipalities = selected.municipality_id.nunique()
    return dict(cases=len(rows), municipalities=rows.municipality_id.nunique(),
                forecast_origin_dates=rows.forecast_origin.map(_month).nunique(),
                eligible_cases=int(rows.eligible_at_origin.sum()), fully_known=known,
                positives=int(positives.sum()), negatives=int(negatives.sum()),
                censored_or_missing=int((~rows.fully_known).sum()),
                right_censored=int(rows.right_censored.sum()) if "right_censored" in rows else 0,
                known_active_cases=int((~rows.at_risk).sum()),
                unique_events=len(ids), positive_event_onset_dates=selected.onset_period.map(_month).nunique(),
                positive_warning_origin_dates=rows.loc[positives, "forecast_origin"].map(_month).nunique(),
                municipalities_with_event=event_municipalities,
                share_municipalities_with_event=event_municipalities / population if population else np.nan,
                positive_rate=int(positives.sum()) / known if known else np.nan)


def summarize_feasibility(
    cases: pd.DataFrame, events: pd.DataFrame, samples: pd.DataFrame | None = None,
    config: Mapping | None = None,
) -> pd.DataFrame:
    """Counts by horizon and denominator; event dates come from true onsets.

    ``all`` keeps the requested panel, ``eligible`` applies only historical
    cohort eligibility, and ``at_risk_evaluable`` adds causal active exclusion
    and complete truth. ``train``/``test`` use the prespecified calendar split.
    Registry-wide counts separately include events with no evaluable warning.
    """
    if isinstance(samples, Mapping) and config is None:
        config, samples = samples, None
    if config is None:
        raise ValueError("Feasibility summaries require a calendar-split config")
    eligible, at_risk, fully_known = _case_fields(cases)
    registry = _registry(events)
    annotated = temporal_split(cases, config)
    population = samples.municipality_id.nunique() if samples is not None else cases.municipality_id.nunique()
    records = []
    for k in (1, 3):
        horizon = cases.k.eq(k)
        masks = {"all": horizon, "eligible": horizon & eligible,
                 "at_risk_evaluable": horizon & eligible & at_risk & fully_known,
                 "train": annotated.k.eq(k) & annotated.split.eq("train"),
                 "test": annotated.k.eq(k) & annotated.split.eq("test")}
        for scope, mask in masks.items():
            records.append(dict(k=k, scope=scope, panel_municipalities=population,
                **_counts(cases.loc[mask], registry, population),
                registry_events=len(events), registry_event_onset_dates=events.onset_period.map(_month).nunique(),
                registry_municipalities_with_event=events.municipality_id.nunique(),
                registry_share_municipalities_with_event=events.municipality_id.nunique() / population if population else np.nan))
    return pd.DataFrame(records)


def evaluate_gate(feasibility: pd.DataFrame, config: Mapping | None = None) -> pd.DataFrame:
    """Return one immutable gate decision per k, before any model fit."""
    thresholds = _gate_thresholds(config)
    required = {"k", "scope", "positives", "positive_event_onset_dates", "forecast_origin_dates", "cases"}
    if not required.issubset(feasibility):
        raise ValueError("Gate requires train/test feasibility counts and unique onset dates")
    records = []
    for k in (1, 3):
        by_scope = feasibility.loc[feasibility.k.eq(k) & feasibility.scope.isin(["train", "test"])]
        if len(by_scope) != 2 or by_scope.scope.nunique() != 2:
            raise ValueError("Gate requires exactly one train and test summary for each k")
        train, test = (by_scope.loc[by_scope.scope.eq(name)].iloc[0] for name in ("train", "test"))
        actual = dict(train_positives=int(train.positives), test_positives=int(test.positives),
                      train_positive_onset_dates=int(train.positive_event_onset_dates),
                      test_positive_onset_dates=int(test.positive_event_onset_dates))
        failures = []
        for minimum_name, minimum in thresholds.items():
            name = minimum_name.removeprefix("min_")
            if actual[name] < minimum:
                failures.append(f"{name}={actual[name]}<{minimum}")
        records.append(dict(k=k, **actual, **thresholds, train_cases=int(train.cases), test_cases=int(test.cases),
                            train_origin_dates=int(train.forecast_origin_dates), test_origin_dates=int(test.forecast_origin_dates),
                            gate_pass=not failures, gate_passed=not failures, classifier_allowed=not failures,
                            failure_reasons="|".join(failures), date_basis="unique_weak_event_onset_period_not_municipal_rows"))
    return pd.DataFrame(records)


def audit_all_calendar_splits(cases: pd.DataFrame, events: pd.DataFrame, config: Mapping) -> pd.DataFrame:
    """Show every requested origin-month cutoff; never rank classifier metrics.

    The original cutoff is marked and retained by the caller whenever its test
    is nonempty. An empty original test calls for explicit review of this table,
    not an automatic boundary change. Both warning horizons are audited.
    """
    _case_fields(cases)
    registry = _registry(events)
    original = _month(_split_config(config)["train_information_cutoff"])
    months = sorted(set(cases.forecast_origin.map(_month)) | {original})
    last = _month(_split_config(config)["test_last_origin"])
    records = []
    population = cases.municipality_id.nunique()
    for cutoff in months:
        if cutoff >= last:
            # Still report the final cutoff as an explicitly empty test.
            test_first, test_last = cutoff + 1, cutoff + 1
        else:
            test_first, test_last = cutoff + 1, last
        cfg = deepcopy(config)
        updates = dict(train_information_cutoff=str(cutoff), test_first_origin=str(test_first), test_last_origin=str(test_last))
        if "split" in cfg:
            cfg["split"].update(updates)
        if "split" in cfg["early_warning"]:
            cfg["early_warning"]["split"].update(updates)
        annotated = temporal_split(cases, cfg)
        summaries = []
        for k in (1, 3):
            for scope in ("train", "test"):
                summaries.append(dict(k=k, scope=scope, **_counts(annotated.loc[annotated.k.eq(k) & annotated.split.eq(scope)], registry, population)))
        summary = pd.DataFrame(summaries)
        gate = evaluate_gate(summary, config)
        for row in gate.to_dict("records"):
            records.append(dict(cutoff_period=str(cutoff), is_original_cutoff=cutoff == original,
                test_first_origin_period=str(test_first), test_last_origin_period=str(test_last),
                has_nonempty_train_and_test=row["train_cases"] > 0 and row["test_cases"] > 0,
                diagnostic_only=True, selection_basis="calendar_and_feasibility_only_no_model_metrics", **row))
    return pd.DataFrame(records)
