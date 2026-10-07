"""Conditional E07b baselines and evaluation against the fixed E07a weak truth.

No classifier or preprocessing fit occurs in ``fit_baselines`` before the
feasibility gate. Test labels are used for gate/evaluation only. The module
does not install sklearn, move temporal boundaries, tune C or choose thresholds.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import re

import numpy as np
import pandas as pd


GATE_REQUIREMENTS = dict(train_positives_min=30, test_positives_min=10,
                         train_positive_dates_min=3, test_positive_dates_min=2)
MODEL_GROUPS = {
    "B0": (), "B1": ("history",), "B2": ("history", "detector"),
    "B3": ("history", "detector", "macro"),
    "B4": ("history", "detector", "macro", "news"),
}
THRESHOLD = 0.5
TRUTH = "E07a_weak_sustained_saved_forecast_error_shift_not_independent_shock_truth"
KEYS = ("municipality_id", "forecast_origin")
PREDICTION_COLUMNS = KEYS + ("k", "model", "label", "probability", "alert", "fully_known", "at_risk", "truth_definition")
METRIC_COLUMNS = ("model", "k", "cases", "positives", "negatives", "pr_auc", "pr_auc_definition", "roc_auc",
                  "brier_score", "log_loss", "precision", "recall", "f1", "calibration_ece", "threshold", "truth_definition")
EVENT_METRIC_COLUMNS = ("model", "k", "monitored_cases", "monitored_municipality_years", "eligible_events",
                        "warned_events", "event_recall", "alert_precision", "median_lead_time_months",
                        "raw_alert_count", "deduplicated_alert_count", "repeated_alert_count", "false_alert_count",
                        "false_alerts_per_municipality_year", "truth_definition")
CALIBRATION_COLUMNS = ("model", "k", "bin", "lower", "upper", "count", "mean_probability", "positive_fraction")
COEFFICIENT_COLUMNS = ("model", "k", "feature", "group", "coefficient", "coefficient_scale", "coefficient_original_units",
                       "training_median", "training_mean_after_imputation", "training_scale")
ALERT_COLUMNS = KEYS + ("k", "model", "probability", "label", "event_id", "status", "successful_warning",
                       "repeated_warning", "false_alert", "lead_time_months", "nearest_past_event_id", "truth_definition")
EVENT_MATCH_COLUMNS = ("municipality_id", "k", "model", "event_id", "onset_period", "eligible_origin_count",
                       "warning_observed", "earliest_alert_origin", "lead_time_months", "warning_count_before_event",
                       "successful_warning_count", "truth_definition")
FILTER_COLUMNS = ("model", "k", "feature", "group", "kept", "reason", "duplicate_of", "training_rows",
                  "selected", "drop_reason", "training_missing", "training_distinct_finite", "training_median", "training_mean_after_imputation", "training_scale")
STATUS_COLUMNS = ("model", "k", "status", "reason", "fit_performed", "retained_features")


def _month(value) -> pd.Period:
    if isinstance(value, pd.Period):
        return value.asfreq("M")
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("Europe/Moscow").tz_localize(None)
    if pd.isna(stamp):
        raise ValueError("Missing monthly date")
    return stamp.to_period("M")


def _instant(value) -> pd.Timestamp:
    if isinstance(value, pd.Period) or (isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}", value)):
        value = pd.Period(value, freq="M").end_time.normalize()
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError("Missing availability timestamp")
    return (stamp.tz_localize("Europe/Moscow") if stamp.tzinfo is None else stamp).tz_convert("UTC")


def _truth_flags(values: pd.Series) -> pd.Series:
    if not values.isin([True, False, 0, 1]).all() or values.isna().any():
        raise ValueError("Eligibility flags must be explicit boolean values")
    return values.astype(bool)


def _eligible_cases(cases: pd.DataFrame, k: int) -> pd.DataFrame:
    if k not in (1, 3):
        raise ValueError("The fixed E07a horizons are k=1 and k=3")
    required = set(KEYS) | {"label", "fully_known", "at_risk", "positive_event_ids", "label_known_at"}
    if not required.issubset(cases):
        raise ValueError(f"Missing warning case fields: {sorted(required - set(cases))}")
    result = cases.loc[cases.k.eq(k)].copy() if "k" in cases else cases.copy()
    result["municipality_id"] = result.municipality_id.astype(str)
    if result.duplicated(list(KEYS)).any():
        raise ValueError("Duplicate municipality/origin warning cases")
    known = _truth_flags(result.fully_known)
    risk = _truth_flags(result.at_risk)
    eligible = _truth_flags(result.eligible_at_origin) if "eligible_at_origin" in result else pd.Series(True, index=result.index)
    label = pd.to_numeric(result.label, errors="raise")
    if result.loc[~known, "label"].notna().any():
        raise ValueError("Unknown/censored labels cannot be turned into negatives or positives")
    if not label.loc[known].isin([0, 1]).all():
        raise ValueError("Fully known labels must be binary")
    result = result.loc[known & risk & eligible].copy()
    result["label"] = label.loc[result.index].astype(int)
    return result


def _ids(value) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(x) for x in value if str(x)]
    if value is None or (not isinstance(value, (list, tuple)) and pd.isna(value)) or str(value).strip() == "":
        return []
    rendered = str(value).strip()
    return [str(x) for x in json.loads(rendered)] if rendered.startswith("[") else rendered.split("|")


def _registry(events: pd.DataFrame) -> pd.DataFrame:
    data = events.copy()
    if "municipality_id" not in data and "series_id" in data:
        data["municipality_id"] = data.series_id
    if not {"municipality_id", "event_id", "onset_period"}.issubset(data):
        raise ValueError("Weak event registry needs municipality_id/event_id/onset_period")
    data["municipality_id"] = data.municipality_id.astype(str)
    data["event_id"] = data.event_id.astype(str)
    if data.event_id.duplicated().any() or data.event_id.eq("").any():
        raise ValueError("Weak event IDs must be unique and nonempty")
    data["_onset"] = data.onset_period.map(_month)
    return data


def _positive_event_dates(cases: pd.DataFrame, events: pd.DataFrame, k: int) -> tuple[int, int]:
    by_id = events.set_index("event_id")
    linked = set()
    for row in cases.loc[cases.label.eq(1)].itertuples():
        ids = _ids(row.positive_event_ids)
        if not ids:
            raise ValueError("A positive case requires its weak event IDs")
        origin = _month(row.forecast_origin)
        for identifier in ids:
            if identifier not in by_id.index:
                raise ValueError("Positive case refers to an unknown weak event")
            event = by_id.loc[identifier]
            if event.municipality_id != row.municipality_id or not origin < event._onset <= origin + k:
                raise ValueError("Positive event linkage violates municipality or future-onset window")
            linked.add(identifier)
    return len(linked), by_id.loc[list(linked), "_onset"].nunique() if linked else 0


def feasibility_gate(train_cases: pd.DataFrame, test_cases: pd.DataFrame, event_registry: pd.DataFrame,
                     k: int, *, fit_cutoff=None) -> dict:
    """Fixed minima count unique event *onset dates*, never copies across MO/O.

    Temporal consistency and training-label availability are checked in addition
    to the user minima. Empty or one-class train data cannot trigger model fit.
    Test positive counts only govern admission; they never tune model parameters.
    """
    train, test = _eligible_cases(train_cases, k), _eligible_cases(test_cases, k)
    events = _registry(event_registry)
    train_ids, train_dates = _positive_event_dates(train, events, k)
    test_ids, test_dates = _positive_event_dates(test, events, k)
    counts = dict(train_cases=len(train), test_cases=len(test),
                  train_positives=int(train.label.eq(1).sum()), test_positives=int(test.label.eq(1).sum()),
                  train_negatives=int(train.label.eq(0).sum()), test_negatives=int(test.label.eq(0).sum()),
                  train_positive_event_ids=train_ids, test_positive_event_ids=test_ids,
                  train_positive_event_dates=int(train_dates), test_positive_event_dates=int(test_dates),
                  train_origin_dates=train.forecast_origin.nunique(), test_origin_dates=test.forecast_origin.nunique())
    reasons = []
    for key, minimum in [("train_positives", 30), ("test_positives", 10),
                         ("train_positive_event_dates", 3), ("test_positive_event_dates", 2)]:
        if counts[key] < minimum:
            reasons.append(f"{key}={counts[key]} < {minimum}")
    if not train.empty and not test.empty:
        first_test = min(test.forecast_origin.map(_instant))
        cutoff = _instant(fit_cutoff) if fit_cutoff is not None else _instant((_month(first_test) - 1).end_time.normalize())
        if max(train.forecast_origin.map(_instant)) >= first_test:
            reasons.append("training_origins_not_strictly_before_test")
        if cutoff >= first_test:
            reasons.append("fit_cutoff_not_before_first_test_origin")
        if train.label_known_at.map(_instant).gt(cutoff).any():
            reasons.append("training_labels_not_available_by_fit_cutoff")
    else:
        cutoff = _instant(fit_cutoff) if fit_cutoff is not None else None
    minima_passed = not any(" < " in reason for reason in reasons)
    if len(train.label.unique()) != 2:
        reasons.append("training_labels_require_both_classes_for_logistic_regression")
    return dict(k=k, passed=not reasons, minima_passed=minima_passed, reasons=reasons, **counts,
                requirements=dict(GATE_REQUIREMENTS), fit_cutoff=str(cutoff) if cutoff is not None else None,
                date_definition="unique_onset_dates_linked_to_positive_eligible_at_risk_cases",
                truth_definition=TRUTH)


def _feature_names(columns: Sequence[str]) -> list[str]:
    names = list(columns)
    if len(names) != len(set(names)):
        raise ValueError("Feature names must be unique")
    for name in names:
        if (name in {*KEYS, "k", "y_true", "y_pred", "label", "at_risk", "fully_known", "label_known_at"}
                or any(token in name.lower() for token in ("offline", "pelt", "binseg", "breakpoint", "future", "label"))):
            raise ValueError(f"Noncausal or target metadata cannot enter feature matrix: {name}")
    return names


def _numeric(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    if not set(columns).issubset(frame):
        raise ValueError(f"Missing selected features: {sorted(set(columns) - set(frame))}")
    result = frame[list(columns)].apply(pd.to_numeric, errors="raise").astype(float)
    if np.isinf(result.to_numpy()).any():
        raise ValueError("Infinite feature values are not median-imputable")
    return result


def training_feature_audit(train: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Descriptive train-only filter; no classifier, labels or test input read."""
    names = _feature_names(columns)
    numeric = _numeric(train, names)
    kept, rows = [], []
    for name in names:
        value = numeric[name]
        finite = value.dropna()
        duplicate = ""
        if finite.empty:
            reason = "all_missing_on_train"
        elif finite.nunique() == 1:
            reason = "constant_after_training_median_imputation"
        else:
            duplicate = next((other for other in kept if np.array_equal(
                value.to_numpy(), numeric[other].to_numpy(), equal_nan=True)), "")
            reason = "exact_numeric_duplicate_on_train" if duplicate else "retained_train_varying"
        keep = reason == "retained_train_varying"
        if keep:
            kept.append(name)
        rows.append(dict(feature=name, kept=keep, selected=keep, reason=reason,
                         drop_reason="" if keep else reason, duplicate_of=duplicate,
                         training_rows=len(train), training_missing=int(value.isna().sum()),
                         training_distinct_finite=int(finite.nunique()),
                         training_median=float(finite.median()) if len(finite) else np.nan))
    return pd.DataFrame(rows, columns=("feature", "kept", "selected", "reason", "drop_reason", "duplicate_of", "training_rows",
                                      "training_missing", "training_distinct_finite", "training_median"))


class TrainOnlyPreprocessor:
    """Median imputation and population-variance scaling fitted only on train.

    The numeric scaling formula matches StandardScaler's mean/std (ddof=0).
    Existing supplied missing-indicator columns receive the same train-only
    filtering; no test missingness creates or selects a new column.
    """
    def __init__(self, feature_columns: Sequence[str]):
        self.feature_columns = _feature_names(feature_columns)
        self.fitted = False

    def fit(self, train: pd.DataFrame):
        if train.empty:
            raise ValueError("Cannot fit preprocessing on empty train")
        self.filter_ = training_feature_audit(train, self.feature_columns)
        self.selected_columns_ = self.filter_.loc[self.filter_.kept, "feature"].tolist()
        values = _numeric(train, self.selected_columns_)
        self.medians_ = values.median()
        imputed = values.fillna(self.medians_)
        self.means_ = imputed.mean()
        self.scales_ = imputed.std(ddof=0).replace(0, 1)
        self.training_rows_ = len(train)
        self.fitted = True
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted on train first")
        values = _numeric(frame, self.selected_columns_).fillna(self.medians_)
        return ((values - self.means_) / self.scales_).to_numpy(dtype=np.float64)

    def audit(self) -> pd.DataFrame:
        if not self.fitted:
            raise ValueError("Preprocessor has not been fitted")
        result = self.filter_.copy()
        result["training_mean_after_imputation"] = result.feature.map(self.means_)
        result["training_scale"] = result.feature.map(self.scales_)
        return result


def evaluate_probabilities(y, probability, *, threshold=THRESHOLD) -> tuple[dict, pd.DataFrame]:
    """Average precision (step PR-AUC), ROC-AUC, proper scores and ten fixed bins."""
    if threshold != THRESHOLD:
        raise ValueError("E07b threshold is fixed at 0.5, never selected on test")
    truth = np.asarray(y, dtype=float)
    prob = np.asarray(probability, dtype=float)
    if truth.ndim != 1 or prob.shape != truth.shape or not np.isin(truth, [0, 1]).all():
        raise ValueError("Metrics require equally sized finite binary labels and probabilities")
    if not np.isfinite(prob).all() or (prob < 0).any() or (prob > 1).any():
        raise ValueError("Probabilities must be finite within [0,1]")
    n, positives = len(truth), int(truth.sum())
    negatives = n - positives
    alert = prob >= threshold
    tp, fp, fn = int(((truth == 1) & alert).sum()), int(((truth == 0) & alert).sum()), int(((truth == 1) & ~alert).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0 if n else np.nan
    recall = tp / (tp + fn) if tp + fn else np.nan
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0 if n else np.nan
    average_precision, roc_auc = np.nan, np.nan
    if positives:
        order = np.argsort(-prob, kind="stable")
        sorted_prob, sorted_truth = prob[order], truth[order]
        endpoints = np.flatnonzero(np.r_[sorted_prob[:-1] != sorted_prob[1:], True])
        cumulative = np.cumsum(sorted_truth)[endpoints]
        recalls, precisions = cumulative / positives, cumulative / (endpoints + 1)
        average_precision = float(np.sum(np.diff(np.r_[0, recalls]) * precisions))
    if positives and negatives:
        ranks = pd.Series(prob).rank(method="average").to_numpy()
        roc_auc = float((ranks[truth == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))
    clipped = np.clip(prob, np.finfo(float).eps, 1 - np.finfo(float).eps)
    bins = np.minimum((prob * 10).astype(int), 9)
    calibration = []
    ece = 0.0 if n else np.nan
    for index in range(10):
        chosen = bins == index
        count = int(chosen.sum())
        mean_prob = float(prob[chosen].mean()) if count else np.nan
        frequency = float(truth[chosen].mean()) if count else np.nan
        if count:
            ece += count / n * abs(mean_prob - frequency)
        calibration.append(dict(bin=index + 1, lower=index / 10, upper=(index + 1) / 10,
                                count=count, mean_probability=mean_prob, positive_fraction=frequency))
    metrics = dict(cases=n, positives=positives, negatives=negatives, pr_auc=average_precision,
                   pr_auc_definition="average_precision_step_integral_grouping_probability_ties",
                   roc_auc=roc_auc, brier_score=float(np.mean((prob - truth) ** 2)) if n else np.nan,
                   log_loss=float(-np.mean(truth * np.log(clipped) + (1 - truth) * np.log(1 - clipped))) if n else np.nan,
                   precision=precision, recall=recall, f1=f1, calibration_ece=ece,
                   threshold=threshold, truth_definition=TRUTH)
    return metrics, pd.DataFrame(calibration)


def evaluate_event_alerts(predictions: pd.DataFrame, event_registry: pd.DataFrame, k: int,
                          *, threshold=THRESHOLD) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Greedy chronological alert/event matching, earliest unmatched future event.

    Every event can contribute one success. Later eligible alerts for an already
    warned event are repeats, excluded from deduplicated alert precision. Alerts
    at/after onset are false alerts, never warnings. Events are evaluated only
    when at least one fully known at-risk monitored origin supports their window.
    False-alert exposure is exactly monitored MO-month cases divided by twelve.
    """
    if k not in (1, 3) or threshold != THRESHOLD:
        raise ValueError("Fixed k=1/3 and threshold=0.5 required")
    required = set(KEYS) | {"probability", "label", "fully_known", "at_risk"}
    if not required.issubset(predictions):
        raise ValueError("Missing monitored prediction fields")
    monitored = predictions.copy()
    if "k" in monitored:
        monitored = monitored.loc[monitored.k.eq(k)].copy()
    if "model" in monitored and monitored.model.nunique() > 1:
        raise ValueError("Evaluate one model at a time")
    eligible = _truth_flags(monitored.eligible_at_origin) if "eligible_at_origin" in monitored else pd.Series(True, index=monitored.index)
    monitored = monitored.loc[_truth_flags(monitored.fully_known) & _truth_flags(monitored.at_risk) & eligible].copy()
    monitored["municipality_id"] = monitored.municipality_id.astype(str)
    if monitored.duplicated(list(KEYS)).any():
        raise ValueError("Each monitored MO-month can issue at most one model alert")
    monitored["_origin"] = monitored.forecast_origin.map(_month)
    # Probability validation must also run when no alert crosses 0.5.
    evaluate_probabilities(monitored.label.to_numpy(), monitored.probability.to_numpy())
    model = str(monitored.model.iloc[0]) if len(monitored) and "model" in monitored else ""
    events = _registry(event_registry)
    eligible = []
    for event in events.to_dict("records"):
        origins = monitored.loc[monitored.municipality_id.eq(event["municipality_id"]), "_origin"]
        valid = origins.map(lambda origin: origin < event["_onset"] <= origin + k)
        if valid.any():
            eligible.append({**event, "eligible_origin_count": int(valid.sum()), "warnings": []})
    eligible.sort(key=lambda event: (event["_onset"], event["event_id"]))
    warned, alerts = set(), []
    selected = monitored.loc[monitored.probability.ge(threshold)].sort_values(["_origin", "municipality_id"], kind="stable")
    for prediction in selected.to_dict("records"):
        unit, origin = prediction["municipality_id"], prediction["_origin"]
        possible = [event for event in eligible if event["municipality_id"] == unit and origin < event["_onset"] <= origin + k]
        unmatched = [event for event in possible if event["event_id"] not in warned]
        chosen = unmatched[0] if unmatched else possible[0] if possible else None
        prior = events.loc[events.municipality_id.eq(unit) & events._onset.map(lambda onset: origin - k <= onset <= origin)]
        previous_id = str(prior.sort_values(["_onset", "event_id"]).iloc[-1].event_id) if len(prior) else ""
        success, repeat = chosen is not None and bool(unmatched), chosen is not None and not unmatched
        if chosen:
            chosen["warnings"].append(prediction["forecast_origin"])
            if success:
                warned.add(chosen["event_id"])
        alerts.append(dict(municipality_id=unit, forecast_origin=prediction["forecast_origin"], k=k, model=model,
                           probability=prediction["probability"], label=prediction["label"],
                           event_id=chosen["event_id"] if chosen else "",
                           status="successful_pre_onset_warning" if success else "repeated_pre_onset_warning" if repeat else
                                  "late_or_at_onset_false_alert" if previous_id else "false_alert_no_eligible_future_event",
                           successful_warning=success, repeated_warning=repeat, false_alert=chosen is None,
                           lead_time_months=chosen["_onset"].ordinal - origin.ordinal if chosen else np.nan,
                           nearest_past_event_id=previous_id, truth_definition=TRUTH))
    event_rows = []
    for event in eligible:
        warnings = event["warnings"]
        earliest = min(warnings, key=_month) if warnings else ""
        event_rows.append(dict(municipality_id=event["municipality_id"], k=k, model=model,
                               event_id=event["event_id"], onset_period=str(event["_onset"]),
                               eligible_origin_count=event["eligible_origin_count"], warning_observed=bool(warnings),
                               earliest_alert_origin=earliest,
                               lead_time_months=event["_onset"].ordinal - _month(earliest).ordinal if earliest else np.nan,
                               warning_count_before_event=len(warnings), successful_warning_count=int(bool(warnings)),
                               truth_definition=TRUTH))
    matched = pd.DataFrame(event_rows, columns=EVENT_MATCH_COLUMNS)
    alert_table = pd.DataFrame(alerts, columns=ALERT_COLUMNS)
    successes, repeats = len(warned), int(alert_table.repeated_warning.sum())
    false = int(alert_table.false_alert.sum())
    deduplicated = successes + false
    metrics = dict(model=model, k=k, monitored_cases=len(monitored), monitored_municipality_years=len(monitored) / 12,
                   eligible_events=len(eligible), warned_events=successes,
                   event_recall=successes / len(eligible) if eligible else np.nan,
                   alert_precision=successes / deduplicated if deduplicated else np.nan,
                   median_lead_time_months=float(matched.loc[matched.warning_observed, "lead_time_months"].median()) if successes else np.nan,
                   raw_alert_count=len(alert_table), deduplicated_alert_count=deduplicated,
                   repeated_alert_count=repeats, false_alert_count=false,
                   false_alerts_per_municipality_year=false * 12 / len(monitored) if len(monitored) else np.nan,
                   truth_definition=TRUTH)
    return metrics, matched, alert_table


def empty_conditional_outputs(gate: dict, *, status="gate_failed", reason="") -> dict:
    """Empty saved schemas explicitly mean no predictions/metrics were obtained."""
    return dict(gate=gate, fit_count=0, status=status,
                predictions=pd.DataFrame(columns=PREDICTION_COLUMNS), metrics_row=pd.DataFrame(columns=METRIC_COLUMNS),
                metrics_event=pd.DataFrame(columns=EVENT_METRIC_COLUMNS), calibration=pd.DataFrame(columns=CALIBRATION_COLUMNS),
                coefficients=pd.DataFrame(columns=COEFFICIENT_COLUMNS), alerts=pd.DataFrame(columns=ALERT_COLUMNS),
                event_matches=pd.DataFrame(columns=EVENT_MATCH_COLUMNS), feature_filter=pd.DataFrame(columns=FILTER_COLUMNS),
                model_status=pd.DataFrame([dict(model=model, k=gate["k"], status=status,
                                                reason=reason or " | ".join(gate["reasons"]), fit_performed=False, retained_features=0)
                                           for model in MODEL_GROUPS], columns=STATUS_COLUMNS))


def _logistic_regression_class():
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:
        raise ImportError("sklearn is not installed; E07b does not install or replace LogisticRegression") from exc
    return LogisticRegression


def fit_baselines(train_cases: pd.DataFrame, test_cases: pd.DataFrame, features: pd.DataFrame,
                  event_registry: pd.DataFrame, feature_groups: Mapping[str, Sequence[str]], k: int,
                  *, gate: dict | None = None, fit_cutoff=None) -> dict:
    """Fit fixed B0–B4 only after recomputed admission, on identical test cases.

    Feature groups must be explicit numeric feature lists. Missing indicators,
    if present in the existing feature pipeline, must be included in these lists.
    No validation/test-driven parameter, feature, probability or threshold choice.
    """
    actual = feasibility_gate(train_cases, test_cases, event_registry, k, fit_cutoff=fit_cutoff)
    if not actual["passed"]:
        return empty_conditional_outputs(actual)
    if gate is not None and not gate.get("passed", gate.get("pass", False)):
        return empty_conditional_outputs(actual, status="external_gate_refused", reason="External feasibility decision refuses fit")
    try:
        LogisticRegression = _logistic_regression_class()
    except ImportError as exc:
        return empty_conditional_outputs(actual, status="sklearn_unavailable", reason=str(exc))
    groups = dict(feature_groups)
    group_labels = {name: name for name in ("history", "detector", "macro", "news")}
    if set(groups) == {"A", "B", "C", "D"}:
        group_labels = dict(history="A", detector="B", macro="C", news="D")
        groups = {name: groups[label] for name, label in group_labels.items()}
    if "detectors" in groups and "detector" not in groups:
        groups["detector"] = groups.pop("detectors")
    if set(groups) != {"history", "detector", "macro", "news"}:
        raise ValueError("Explicit history/detector/macro/news feature groups required")
    flattened = [name for group in ("history", "detector", "macro", "news") for name in groups[group]]
    _feature_names(flattened)
    group_by_feature = {name: group_labels[group] for group, names in groups.items() for name in names}
    train, test = _eligible_cases(train_cases, k), _eligible_cases(test_cases, k)
    matrix = features.copy()
    matrix["municipality_id"] = matrix.municipality_id.astype(str)
    if matrix.duplicated(list(KEYS)).any():
        raise ValueError("Feature keys must be unique before warning-horizon joins")
    if set(flattened) & set(train.columns):
        raise ValueError("Feature fields and label-case metadata collide")
    train = train.merge(matrix[list(KEYS) + flattened], on=list(KEYS), how="left", validate="one_to_one", indicator=True)
    test = test.merge(matrix[list(KEYS) + flattened], on=list(KEYS), how="left", validate="one_to_one", indicator=True)
    if not train._merge.eq("both").all() or not test._merge.eq("both").all():
        raise ValueError("All eligible train/test cases require their own-origin feature row")
    result = empty_conditional_outputs(actual, status="prepared")
    predictions, row_metrics, event_metrics, calibration_rows, coefficients, alerts, matches, filters, statuses = [], [], [], [], [], [], [], [], []
    fit_count = 0
    y_train = train.label.to_numpy(dtype=int)
    for model, included in MODEL_GROUPS.items():
        candidates = [name for group in included for name in groups[group]]
        if model == "B0":
            risk = float(y_train.mean())
            probability = np.full(len(test), risk)
            coefficients.append(dict(model=model, k=k, feature="train_positive_rate", group="constant",
                                     coefficient=risk, coefficient_scale="probability"))
            retained = 0
        else:
            processor = TrainOnlyPreprocessor(candidates).fit(train)
            audit = processor.audit()
            audit["model"], audit["k"] = model, k
            audit["group"] = audit.feature.map(group_by_feature)
            filters.extend(audit.to_dict("records"))
            retained = len(processor.selected_columns_)
            if not retained:
                statuses.append(dict(model=model, k=k, status="no_varying_train_features", reason="No retained numerical train features", fit_performed=False, retained_features=0))
                continue
            classifier = LogisticRegression(C=1.0, class_weight=None, random_state=42,
                                            solver="lbfgs", max_iter=2000, n_jobs=1)
            classifier.fit(processor.transform(train), y_train)
            probability = classifier.predict_proba(processor.transform(test))[:, 1]
            for name, coefficient in zip(processor.selected_columns_, classifier.coef_[0]):
                coefficients.append(dict(model=model, k=k, feature=name, group=group_by_feature[name],
                                         coefficient=float(coefficient), coefficient_scale="standardized_train_feature",
                                         coefficient_original_units=float(coefficient / processor.scales_[name]),
                                         training_median=float(processor.medians_[name]),
                                         training_mean_after_imputation=float(processor.means_[name]),
                                         training_scale=float(processor.scales_[name])))
            coefficients.append(dict(model=model, k=k, feature="intercept", group="intercept",
                                     coefficient=float(classifier.intercept_[0]), coefficient_scale="standardized_feature_intercept",
                                     coefficient_original_units=float(classifier.intercept_[0] - np.sum(
                                         classifier.coef_[0] * (processor.means_ / processor.scales_).to_numpy()))))
        fit_count += 1
        current = test[list(KEYS) + ["label", "fully_known", "at_risk"]].copy()
        current["k"], current["model"] = k, model
        current["probability"], current["alert"] = probability, probability >= THRESHOLD
        current["truth_definition"] = TRUTH
        metrics, calibration = evaluate_probabilities(current.label.to_numpy(), probability)
        metrics.update(model=model, k=k)
        calibration["model"], calibration["k"] = model, k
        events_metrics, matched, alert_table = evaluate_event_alerts(current, event_registry, k)
        predictions.extend(current.to_dict("records"))
        row_metrics.append(metrics)
        event_metrics.append(events_metrics)
        calibration_rows.extend(calibration.to_dict("records"))
        matches.extend(matched.to_dict("records"))
        alerts.extend(alert_table.to_dict("records"))
        statuses.append(dict(model=model, k=k, status="fitted", reason="Fixed parameters; no test/validation tuning", fit_performed=True, retained_features=retained))
    result.update(status="fitted_conditional_baselines", fit_count=fit_count,
                  predictions=pd.DataFrame(predictions, columns=PREDICTION_COLUMNS),
                  metrics_row=pd.DataFrame(row_metrics, columns=METRIC_COLUMNS),
                  metrics_event=pd.DataFrame(event_metrics, columns=EVENT_METRIC_COLUMNS),
                  calibration=pd.DataFrame(calibration_rows, columns=CALIBRATION_COLUMNS),
                  coefficients=pd.DataFrame(coefficients, columns=COEFFICIENT_COLUMNS),
                  alerts=pd.DataFrame(alerts, columns=ALERT_COLUMNS),
                  event_matches=pd.DataFrame(matches, columns=EVENT_MATCH_COLUMNS),
                  feature_filter=pd.DataFrame(filters, columns=FILTER_COLUMNS),
                  model_status=pd.DataFrame(statuses, columns=STATUS_COLUMNS))
    return result
