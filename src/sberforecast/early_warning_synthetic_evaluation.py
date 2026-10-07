"""E07c evaluation of independently generated truth, never real shock truth.

Integer month indices are local to each independent series. Threshold selection
accepts validation only. Evaluation admits complete, eligible, pre-onset cases;
the active-state exclusion is a generator oracle used for the target population,
not an available model feature. Whole-series bootstrap repeats all observations
and event contributions of a sampled series with their multiplicity.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from .early_warning_full_panel_models import evaluate_probabilities


TRUTH = "E07c_independent_synthetic_generator_onset_not_real_shock_truth"
GROUP = ["cohort", "model", "k"]
KEY = ["series_id", "cohort"]
PRED_KEY = KEY + ["month_index", "k", "model"]
EVENT_METRICS = (
    "event_recall", "alert_precision", "median_lead_time_months",
    "mean_lead_time_months", "false_alerts_per_12_monitored_months", "miss_rate",
)
MATCH_COLUMNS = KEY + ["model", "k", "event_id", "onset_index", "is_anticipated",
    "precursor_class", "noise_class", "eligible_origin_count", "warning_observed",
    "earliest_alert_index", "lead_time_months", "warning_count_before_event",
    "successful_warning_count", "truth_definition"]
ALERT_COLUMNS = PRED_KEY + ["probability", "threshold", "label", "event_id", "status",
    "successful_warning", "repeated_warning", "false_alert", "lead_time_months",
    "nearest_past_event_id", "truth_definition"]
EXAMPLE_RULE = "test_S3_primary_k3_else_k1_per_type_lexicographic_series_id_then_origin"


def _section(config: Mapping, name: str) -> Mapping:
    value = config.get(name, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"config.{name} must be a mapping")
    return value


def _settings(config: Mapping) -> dict:
    labels, evaluation = _section(config, "labels"), _section(config, "evaluation")
    if labels.get("horizons", [1, 3]) != [1, 3]:
        raise ValueError("E07c uses the locked k=1 and k=3 horizons")
    history = int(labels.get("min_history", 12))
    if history != 12 or int(evaluation.get("calibration_bins", 10)) != 10:
        raise ValueError("E07c min_history=12 and ten calibration bins are fixed")
    bootstrap = int(evaluation.get("bootstrap_resamples", 500))
    ci = float(evaluation.get("ci_level", .95))
    if bootstrap < 1 or not 0 < ci < 1:
        raise ValueError("Positive bootstrap resamples and 0 < ci_level < 1 required")
    return dict(min_origin=history - 1, months=int(_section(config, "generator").get("months", 24)),
                resamples=bootstrap, seed=int(evaluation.get("bootstrap_seed", 420401)), ci=ci)


def _require(frame: pd.DataFrame, fields: list[str], name: str) -> None:
    absent = set(fields) - set(frame)
    if absent:
        raise ValueError(f"Missing {name} fields: {sorted(absent)}")


def _boolean(values: pd.Series, name: str) -> pd.Series:
    if values.isna().any() or not values.isin([True, False, 0, 1]).all():
        raise ValueError(f"{name} must contain explicit boolean values")
    return values.astype(bool)


def _integer(values: pd.Series, name: str) -> pd.Series:
    number = pd.to_numeric(values, errors="raise")
    if not np.isfinite(number.to_numpy(dtype=float)).all() or not number.eq(np.floor(number)).all():
        raise ValueError(f"{name} must be finite integer indices")
    return number.astype(np.int64)


def _identifiers(frame: pd.DataFrame, *, validation_only: bool, allow_train=False) -> None:
    for column in KEY:
        if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Missing {column}")
        frame[column] = frame[column].astype(str)
    cohorts = {"validation"} if validation_only else {"train", "validation", "test"} if allow_train else {"validation", "test"}
    if not frame.cohort.isin(cohorts).all():
        raise ValueError("Threshold input must be validation only" if validation_only else
                         "Evaluation accepts validation/test cohorts only")


def _prepare(predictions: pd.DataFrame, events: pd.DataFrame, metadata: pd.DataFrame,
             config: Mapping, *, validation_only=False, require_alert=True):
    settings = _settings(config)
    data, registry, meta = predictions.copy().reset_index(drop=True), events.copy(), metadata.copy()
    _require(data, PRED_KEY + ["label", "probability"], "prediction")
    _require(registry, KEY + ["event_id", "onset_index"], "event")
    _require(meta, KEY + ["has_event", "is_anticipated", "false_precursor",
                          "precursor_class", "noise_class"], "metadata")
    _identifiers(data, validation_only=validation_only)
    for frame in (registry, meta):
        _identifiers(frame, validation_only=validation_only, allow_train=not validation_only)
    if meta.duplicated(KEY).any() or meta.groupby("series_id").cohort.nunique().gt(1).any():
        raise ValueError("Synthetic series IDs must identify one independent cohort")
    if not validation_only:
        # The runner can pass the complete generator registry. TRAIN truth is
        # discarded here and never enters evaluation or threshold selection.
        registry = registry.loc[~registry.cohort.eq("train")].copy()
        meta = meta.loc[~meta.cohort.eq("train")].copy()
    if data.duplicated(PRED_KEY).any():
        raise ValueError("Duplicate keyed monthly predictions")
    if data.model.isna().any() or data.model.astype(str).str.strip().eq("").any():
        raise ValueError("Missing model name")
    data["model"] = data.model.astype(str)
    for column in ("month_index", "k"):
        data[column] = _integer(data[column], column)
    registry["onset_index"] = _integer(registry.onset_index, "onset_index")
    if not data.k.isin([1, 3]).all():
        raise ValueError("E07c horizons are k=1/3")
    if (data.month_index.lt(0) | data.month_index.ge(settings["months"])).any():
        raise ValueError("Origin index outside synthetic series")
    if (registry.onset_index.lt(0) | registry.onset_index.ge(settings["months"])).any():
        raise ValueError("Onset index outside synthetic series")
    if registry.event_id.isna().any() or registry.event_id.astype(str).str.strip().eq("").any():
        raise ValueError("Event IDs must be nonempty")
    registry["event_id"] = registry.event_id.astype(str)
    if registry.event_id.duplicated().any():
        raise ValueError("Event IDs must be unique")
    for column in ("has_event", "is_anticipated", "false_precursor"):
        meta[column] = _boolean(meta[column], column)
    if not meta.precursor_class.isin(["none", "weak", "strong"]).all():
        raise ValueError("Unknown precursor class")
    if not meta.noise_class.isin(["low", "high"]).all():
        raise ValueError("Unknown noise class")
    lookup = meta.set_index(KEY)
    for frame in (data, registry):
        if not pd.MultiIndex.from_frame(frame[KEY]).isin(lookup.index).all():
            raise ValueError("Prediction/event series absent from metadata")
    event_keys = set(map(tuple, registry[KEY].to_numpy()))
    if any(bool(row.has_event) != ((row.series_id, row.cohort) in event_keys)
           for row in meta.itertuples()):
        raise ValueError("has_event must agree with complete event registry")
    for column in ("is_anticipated", "precursor_class", "noise_class"):
        expected = pd.MultiIndex.from_frame(registry[KEY]).map(lookup[column])
        if column in registry and not np.array_equal(registry[column].to_numpy(), np.asarray(expected)):
            raise ValueError(f"Event/metadata disagreement: {column}")
        registry[column] = np.asarray(expected)
    probability = pd.to_numeric(data.probability, errors="raise").astype(float)
    if not np.isfinite(probability).all() or not probability.between(0, 1).all():
        raise ValueError("Probabilities must be finite within [0,1]")
    data["probability"] = probability
    known = _boolean(data.fully_known, "fully_known") if "fully_known" in data else pd.Series(True, index=data.index)
    risk = _boolean(data.at_risk, "at_risk") if "at_risk" in data else pd.Series(True, index=data.index)
    eligible = _boolean(data.eligible, "eligible") if "eligible" in data else pd.Series(True, index=data.index)
    if "eligible_at_origin" in data:
        additional = _boolean(data.eligible_at_origin, "eligible_at_origin")
        if "eligible" in data and not eligible.equals(additional):
            raise ValueError("Eligibility aliases disagree")
        eligible &= additional
    label = pd.to_numeric(data.label, errors="raise")
    if label.loc[~known].notna().any():
        raise ValueError("Unknown/censored labels cannot be converted to negatives")
    admitted = known & risk & eligible & data.month_index.ge(settings["min_origin"])
    if (admitted & data.month_index.add(data.k).ge(settings["months"])).any():
        raise ValueError("Fully known case requires the complete O+k future window")
    if not label.loc[admitted].isin([0, 1]).all():
        raise ValueError("Admitted labels must be binary")
    # Excluding active generator regimes is a target-population oracle. It is
    # checked separately from model inputs and never enters their feature fit.
    onset_by_key = {key: group.onset_index.to_numpy() for key, group in registry.groupby(KEY, sort=False)}
    for row in data.loc[admitted].itertuples():
        onsets = onset_by_key.get((row.series_id, row.cohort), np.empty(0, dtype=int))
        if np.any(onsets <= row.month_index):
            raise ValueError("Already active synthetic regimes cannot be monitored at risk")
        expected_label = int(np.any((onsets > row.month_index) & (onsets <= row.month_index + row.k)))
        if label.at[row.Index] != expected_label:
            raise ValueError("Label disagrees with independent true-onset future window")
    data = data.loc[admitted].copy()
    data["label"] = label.loc[data.index].astype(int)
    if require_alert:
        _require(predictions, ["threshold", "alert"], "prediction")
        data["threshold"] = pd.to_numeric(data.threshold, errors="raise").astype(float)
        if not np.isfinite(data.threshold).all() or not data.threshold.between(0, 1).all():
            raise ValueError("Threshold must be finite within [0,1]")
        data["alert"] = _boolean(data.alert, "alert")
        if not data.alert.eq(data.probability.ge(data.threshold)).all():
            raise ValueError("Stored alert disagrees with probability >= selected threshold")
        if data.groupby(GROUP).threshold.nunique().gt(1).any():
            raise ValueError("One validation-selected threshold per cohort/model/k required")
    return data.sort_values(PRED_KEY, kind="stable").reset_index(drop=True), registry, meta, settings


def _confusion(truth: np.ndarray, alert: np.ndarray) -> dict:
    tp = int(((truth == 1) & alert).sum())
    fp = int(((truth == 0) & alert).sum())
    fn = int(((truth == 1) & ~alert).sum())
    return dict(true_positives=tp, false_positives=fp, false_negatives=fn,
                true_negatives=int(((truth == 0) & ~alert).sum()),
                precision=tp / (tp + fp) if tp + fp else 0.0 if len(truth) else np.nan,
                recall=tp / (tp + fn) if tp + fn else np.nan,
                f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0 if len(truth) else np.nan)


def _row_metrics(frame: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    metrics, calibration = evaluate_probabilities(frame.label.to_numpy(), frame.probability.to_numpy())
    metrics.update(_confusion(frame.label.to_numpy(), frame.alert.to_numpy(dtype=bool)))
    metrics["threshold"] = float(frame.threshold.iloc[0]) if len(frame) else np.nan
    metrics["truth_definition"] = TRUTH
    return metrics, calibration


def select_threshold(validation_predictions: pd.DataFrame, validation_events: pd.DataFrame,
                     validation_metadata: pd.DataFrame, config: Mapping) -> dict:
    """Maximize validation row F1 under the fixed CONTROL-month alert budget.

    Every control alert is counted monthly. A year is twelve admitted control
    origins, not the number of series and not the number of calendar dates.
    No control exposure makes the constraint unidentifiable and invokes the
    explicit fallback rather than claiming the budget has been satisfied.
    """
    frame, _, metadata, _ = _prepare(validation_predictions, validation_events, validation_metadata,
                                    config, validation_only=True, require_alert=False)
    combinations = validation_predictions[["model", "k"]].drop_duplicates()
    if len(combinations) != 1:
        raise ValueError("Select a threshold for one validation model/k at a time")
    threshold_config = _section(config, "threshold")
    objective = threshold_config.get("objective", "validation_row_f1")
    if objective != "validation_row_f1" or threshold_config.get("tie_break", "higher_threshold") != "higher_threshold":
        raise ValueError("Locked validation F1 objective and higher-threshold tie break required")
    budget = float(threshold_config.get("false_alerts_per_12_control_months_max", 1.0))
    fallback = float(threshold_config.get("fallback", .5))
    if budget != 1.0 or fallback != .5:
        raise ValueError("The E07c alert budget and fallback are fixed")
    controls = set(metadata.loc[~metadata.has_event, "series_id"])
    control = frame.series_id.isin(controls).to_numpy(dtype=int)
    order = np.argsort(frame.probability.to_numpy(), kind="stable")
    probability = frame.probability.to_numpy()[order]
    positive = frame.label.to_numpy()[order]
    control = control[order]
    cumulative_positive = np.r_[0, np.cumsum(positive)]
    cumulative_control = np.r_[0, np.cumsum(control)]
    candidates = np.unique(np.r_[probability, .5, 1.0])
    control_months = int(control.sum())
    positives = int(positive.sum())
    best, candidates_met = None, 0
    def describe(threshold):
        start = int(np.searchsorted(probability, threshold, side="left"))
        tp, count = positives - int(cumulative_positive[start]), len(frame) - start
        fp, fn = count - tp, positives - tp
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0 if len(frame) else np.nan
        false = control_months - int(cumulative_control[start])
        rate = false * 12 / control_months if control_months else np.nan
        return dict(threshold=float(threshold), validation_f1=float(f1), false_control_alerts=false,
                    false_alerts_per_12_control_months=rate,
                    constraint_met=bool(control_months and rate <= budget))
    for threshold in candidates:
        candidate = describe(threshold)
        if candidate["constraint_met"]:
            candidates_met += 1
            if best is None or (candidate["validation_f1"], candidate["threshold"]) > (best["validation_f1"], best["threshold"]):
                best = candidate
    fallback_used = best is None
    if fallback_used:
        best = describe(fallback)
    return dict(**best, cohort="validation", model=str(combinations.iloc[0].model),
                k=int(combinations.iloc[0].k), admitted_cases=len(frame), control_monitored_months=control_months,
                control_monitored_years=control_months / 12, candidates_evaluated=len(candidates),
                candidates_meeting_constraint=candidates_met, fallback_used=fallback_used,
                fallback_reason="no_control_exposure" if not control_months else
                    "no_candidate_meets_control_alert_budget" if fallback_used else "",
                selection_rule="validation_row_f1_max_control_false_alert_budget_ties_higher_threshold",
                threshold_candidates="unique_validation_probabilities_plus_0.5_1.0",
                truth_definition=TRUTH)


def _event_tables(frame: pd.DataFrame, events: pd.DataFrame, model: str, k: int):
    """One chronological alert can warn at most one eligible event."""
    origins = {key: group.month_index.to_numpy() for key, group in frame.groupby(KEY, sort=False)}
    eligible, by_series = [], {}
    for event in events.sort_values(["onset_index", "event_id"], kind="stable").to_dict("records"):
        available = origins.get((event["series_id"], event["cohort"]), np.empty(0, dtype=int))
        valid = (available < event["onset_index"]) & (available >= event["onset_index"] - k)
        if valid.any():
            entry = {**event, "eligible_origin_count": int(valid.sum()), "warnings": []}
            eligible.append(entry)
            by_series.setdefault((entry["series_id"], entry["cohort"]), []).append(entry)
    warned, alerts = set(), []
    past_by_series = {key: group.sort_values(["onset_index", "event_id"], kind="stable")
                      for key, group in events.groupby(KEY, sort=False)}
    for prediction in frame.loc[frame.alert].sort_values(["month_index", "series_id"], kind="stable").to_dict("records"):
        origin, key = prediction["month_index"], (prediction["series_id"], prediction["cohort"])
        possible = [event for event in by_series.get(key, []) if origin < event["onset_index"] <= origin + k]
        unmatched = [event for event in possible if event["event_id"] not in warned]
        chosen = unmatched[0] if unmatched else possible[0] if possible else None
        past = past_by_series.get(key)
        previous = past.loc[past.onset_index.le(origin)] if past is not None else None
        previous_id = str(previous.iloc[-1].event_id) if previous is not None and len(previous) else ""
        success, repeat = chosen is not None and bool(unmatched), chosen is not None and not unmatched
        if chosen is not None:
            chosen["warnings"].append(origin)
            if success:
                warned.add(chosen["event_id"])
        alerts.append({**{column: prediction[column] for column in PRED_KEY + ["probability", "threshold", "label"]},
                       "event_id": chosen["event_id"] if chosen is not None else "",
                       "status": "successful_pre_onset_warning" if success else "repeated_pre_onset_warning" if repeat else
                            "late_or_at_onset_false_alert" if previous_id else "false_alert_no_eligible_future_event",
                       "successful_warning": success, "repeated_warning": repeat, "false_alert": chosen is None,
                       "lead_time_months": chosen["onset_index"] - origin if chosen is not None else np.nan,
                       "nearest_past_event_id": previous_id, "truth_definition": TRUTH})
    matches = []
    for event in eligible:
        earliest = min(event["warnings"]) if event["warnings"] else np.nan
        matches.append({**{column: event[column] for column in KEY + ["event_id", "onset_index", "is_anticipated", "precursor_class", "noise_class"]},
                        "model": model, "k": k, "eligible_origin_count": event["eligible_origin_count"],
                        "warning_observed": bool(event["warnings"]), "earliest_alert_index": earliest,
                        "lead_time_months": event["onset_index"] - earliest if event["warnings"] else np.nan,
                        "warning_count_before_event": len(event["warnings"]),
                        "successful_warning_count": int(bool(event["warnings"])), "truth_definition": TRUTH})
    return pd.DataFrame(matches, columns=MATCH_COLUMNS), pd.DataFrame(alerts, columns=ALERT_COLUMNS)


def _series_statistics(frame, matches, alerts, metadata, k):
    series = sorted(frame.series_id.unique())
    columns = ["monitored", "control_monitored", "events", "warned", "false", "raw", "repeats"] + [f"lead_{lead}" for lead in range(1, k + 1)]
    statistics = pd.DataFrame(0, index=pd.Index(series, name="series_id"), columns=columns, dtype=np.int64)
    if not series:
        return statistics
    statistics["monitored"] = frame.groupby("series_id").size().reindex(series, fill_value=0)
    controls = set(metadata.loc[~metadata.has_event, "series_id"])
    statistics["control_monitored"] = statistics.monitored.where(statistics.index.isin(controls), 0)
    if len(matches):
        statistics["events"] = matches.groupby("series_id").size().reindex(series, fill_value=0)
        statistics["warned"] = matches.groupby("series_id").warning_observed.sum().reindex(series, fill_value=0).astype(int)
        for lead in range(1, k + 1):
            selected = matches.loc[matches.warning_observed & matches.lead_time_months.eq(lead)]
            statistics[f"lead_{lead}"] = selected.groupby("series_id").size().reindex(series, fill_value=0)
    if len(alerts):
        statistics["raw"] = alerts.groupby("series_id").size().reindex(series, fill_value=0)
        statistics["false"] = alerts.groupby("series_id").false_alert.sum().reindex(series, fill_value=0).astype(int)
        statistics["repeats"] = alerts.groupby("series_id").repeated_warning.sum().reindex(series, fill_value=0).astype(int)
    return statistics


def _event_metrics(totals: np.ndarray, k: int) -> dict:
    monitored, control, events, warned, false, raw, repeats = map(int, totals[:7])
    lead_hist = np.asarray(totals[7:], dtype=np.int64)
    mean_lead = float(np.dot(lead_hist, np.arange(1, k + 1)) / warned) if warned else np.nan
    median_lead = np.nan
    if warned:
        cumulative = np.cumsum(lead_hist)
        positions = ((warned - 1) // 2, warned // 2)
        median_lead = float(np.mean([np.searchsorted(cumulative, position, side="right") + 1 for position in positions]))
    recall = warned / events if events else np.nan
    return dict(monitored_cases=monitored, monitored_series_years=monitored / 12,
                control_monitored_months=control, eligible_events=events, warned_events=warned,
                event_recall=recall, alert_precision=warned / (warned + false) if warned + false else np.nan,
                median_lead_time_months=median_lead, mean_lead_time_months=mean_lead,
                raw_alert_count=raw, deduplicated_alert_count=warned + false,
                repeated_alert_count=repeats, false_alert_count=false,
                false_alerts_per_12_monitored_months=false * 12 / monitored if monitored else np.nan,
                miss_rate=1 - recall if events else np.nan, truth_definition=TRUTH,
                exposure_definition="admitted_fully_known_eligible_at_risk_series_months_divided_by_12")


def _bootstrap(statistics: pd.DataFrame, k: int, settings: dict) -> list[dict]:
    point = _event_metrics(statistics.sum(axis=0).to_numpy(), k)
    n = len(statistics)
    draws = {metric: [] for metric in EVENT_METRICS}
    if n:
        rng = np.random.default_rng(settings["seed"])
        # A multinomial vector is exactly the multiplicity representation of
        # drawing n whole independent series with replacement. Never rescan rows.
        weights = rng.multinomial(n, np.full(n, 1 / n), size=settings["resamples"])
        totals = weights @ statistics.to_numpy(dtype=np.int64)
        for total in totals:
            metrics = _event_metrics(total, k)
            for metric in EVENT_METRICS:
                draws[metric].append(metrics[metric])
    alpha = (1 - settings["ci"]) / 2
    rows = []
    for metric in EVENT_METRICS:
        finite = np.asarray(draws[metric], dtype=float)
        finite = finite[np.isfinite(finite)]
        low, high = np.quantile(finite, [alpha, 1 - alpha]) if len(finite) else [np.nan, np.nan]
        rows.append(dict(metric=metric, estimate=point[metric], lower=float(low), upper=float(high),
                         ci_level=settings["ci"], bootstrap_resamples=settings["resamples"],
                         defined_resamples=len(finite), series_count=n, bootstrap_seed=settings["seed"],
                         bootstrap_unit="whole_series_id_with_multiplicity",
                         undefined_resample_policy="omit_undefined_metric_and_report_defined_resamples",
                         truth_definition=TRUTH))
    return rows


def _scenario_metadata(metadata, name):
    if name == "base_all":
        selected, rule = pd.Series(True, index=metadata.index), "all_monitored_series"
    elif name in ("weak_precursor", "strong_precursor"):
        selected = metadata.precursor_class.eq(name.split("_")[0])
        rule = "event_or_control_series_with_requested_precursor_class"
    elif name == "high_noise":
        selected, rule = metadata.noise_class.eq("high"), "high_noise_event_and_control_series"
    elif name == "controls_false_precursor":
        selected = ~metadata.has_event & metadata.false_precursor
        rule = "control_series_with_false_precursor_episode"
    elif name in ("anticipated_events", "unanticipated_events"):
        anticipated = name == "anticipated_events"
        selected = ~metadata.has_event | (metadata.has_event & metadata.is_anticipated.eq(anticipated))
        rule = "requested_event_subset_plus_all_controls_same_paired_control_denominator"
    else:
        raise ValueError(f"Unknown locked evaluation scenario: {name}")
    return metadata.loc[selected], rule


def _ablation(metrics_row, metrics_event):
    rows = []
    metrics = metrics_row.merge(metrics_event, on=GROUP, suffixes=("", "_event"))
    chosen = ["pr_auc", "roc_auc", "brier_score", "log_loss", "f1", "event_recall", "alert_precision",
              "false_alerts_per_12_monitored_months", "median_lead_time_months"]
    for (cohort, k), frame in metrics.groupby(["cohort", "k"], sort=True):
        indexed = frame.set_index("model")
        for earlier, later in [("S1", "S2"), ("S2", "S3")]:
            if earlier not in indexed.index or later not in indexed.index:
                continue
            for metric in chosen:
                a, b = float(indexed.at[earlier, metric]), float(indexed.at[later, metric])
                rows.append(dict(cohort=cohort, k=k, comparison=f"{later}_minus_{earlier}",
                                 metric=metric, earlier_model=earlier, later_model=later,
                                 earlier_value=a, later_value=b, difference=b - a,
                                 preferred_direction="lower" if metric in ("brier_score", "log_loss", "false_alerts_per_12_monitored_months") else "higher",
                                 interpretation="descriptive_fixed_ablation_no_model_selection", truth_definition=TRUTH))
    return pd.DataFrame(rows, columns=["cohort", "k", "comparison", "metric", "earlier_model", "later_model",
                                      "earlier_value", "later_value", "difference", "preferred_direction", "interpretation", "truth_definition"])


def _examples(matches, alerts, config):
    example = _section(_section(config, "evaluation"), "examples")
    if example.get("model", "S3") != "S3" or int(example.get("primary_k", 3)) != 3:
        raise ValueError("The example model S3 and primary k=3 are fixed")
    rows = []
    for kind in ("successful_warning", "false_alert", "missed_event"):
        for k in (3, 1):
            source = matches if kind == "missed_event" else alerts
            selection = source.cohort.eq("test") & source.model.eq("S3") & source.k.eq(k)
            selection &= ~source.warning_observed.astype(bool) if kind == "missed_event" else source[kind].astype(bool)
            eligible = source.loc[selection].copy()
            if eligible.empty:
                continue
            origin_column = "onset_index" if kind == "missed_event" else "month_index"
            row = eligible.sort_values(["series_id", origin_column, "event_id"], kind="stable").iloc[0]
            rows.append(dict(example_type=kind, series_id=row.series_id, cohort="test", model="S3", k=k,
                             month_index=int(row[origin_column]), event_id=str(row.event_id),
                             onset_index=int(row.onset_index) if kind == "missed_event" else
                                 int(row.month_index + row.lead_time_months) if kind == "successful_warning" else np.nan,
                             lead_time_months=float(row.lead_time_months), selection_rule=EXAMPLE_RULE,
                             anchor_definition="true_onset_for_miss_else_alert_origin", truth_definition=TRUTH))
            break
    return pd.DataFrame(rows, columns=["example_type", "series_id", "cohort", "model", "k", "month_index",
                                      "event_id", "onset_index", "lead_time_months", "selection_rule", "anchor_definition", "truth_definition"])


def evaluate_predictions(predictions: pd.DataFrame, events: pd.DataFrame, metadata: pd.DataFrame,
                         config: Mapping) -> dict[str, pd.DataFrame]:
    """Evaluate fixed predictions without fitting, tuning or reading TRAIN.

    Base event estimates and TEST-only bootstrap use all admitted controls plus
    admitted pre-event origins. Scenario rows are descriptive, with their exact
    populations recorded. Events without any admitted pre-onset origin are not
    eligible, rather than manufactured as misses. Examples follow a saved fixed
    selection rule and may be absent when no corresponding observation exists.
    """
    frame, registry, meta, settings = _prepare(predictions, events, metadata, config)
    row_rows, event_rows, calibration_rows = [], [], []
    match_frames, alert_frames, intervals, scenario_rows = [], [], [], []
    scenarios = list(_section(config, "evaluation").get("scenarios", ["base_all", "weak_precursor", "strong_precursor", "high_noise", "controls_false_precursor"]))
    if scenarios != ["base_all", "weak_precursor", "strong_precursor", "high_noise", "controls_false_precursor"]:
        raise ValueError("The five locked E07c scenarios cannot be selected from results")
    scenarios += ["anticipated_events", "unanticipated_events"]
    # Include a requested group even when defensive admission excludes all rows.
    requested_groups = predictions[GROUP].drop_duplicates().sort_values(GROUP, kind="stable")
    requested_groups = requested_groups.astype({"cohort": str, "model": str, "k": int})
    for cohort, model, k in requested_groups.itertuples(index=False, name=None):
        k = int(k)
        keys = dict(cohort=cohort, model=model, k=k)
        group = frame.loc[frame.cohort.eq(cohort) & frame.model.eq(model) & frame.k.eq(k)]
        cohort_meta, cohort_events = meta.loc[meta.cohort.eq(cohort)], registry.loc[registry.cohort.eq(cohort)]
        metrics, calibration = _row_metrics(group)
        row_rows.append({**keys, **metrics})
        calibration_rows.extend({**keys, **row} for row in calibration.to_dict("records"))
        matches, alerts = _event_tables(group, cohort_events, model, k)
        match_frames.append(matches)
        alert_frames.append(alerts)
        statistics = _series_statistics(group, matches, alerts, cohort_meta, k)
        event_metrics = _event_metrics(statistics.sum(axis=0).to_numpy(), k)
        event_rows.append({**keys, **event_metrics})
        if cohort == "test":
            intervals.extend({**keys, **row} for row in _bootstrap(statistics, k, settings))
        for scenario in scenarios:
            selected_meta, rule = _scenario_metadata(cohort_meta, scenario)
            ids = set(selected_meta.series_id)
            selected = group.loc[group.series_id.isin(ids)]
            scenario_matches, scenario_alerts = matches.loc[matches.series_id.isin(ids)], alerts.loc[alerts.series_id.isin(ids)]
            stats = _series_statistics(selected, scenario_matches, scenario_alerts, selected_meta, k)
            row_metric, _ = _row_metrics(selected)
            scenario_rows.append({**keys, "scenario": scenario, "population_rule": rule,
                                  "metadata_series_count": len(selected_meta), "monitored_series_count": len(stats),
                                  **row_metric, **_event_metrics(stats.sum(axis=0).to_numpy(), k)})
    row_metrics = pd.DataFrame(row_rows)
    event_metrics = pd.DataFrame(event_rows)
    nonempty_matches = [table for table in match_frames if not table.empty]
    nonempty_alerts = [table for table in alert_frames if not table.empty]
    matches = pd.concat(nonempty_matches, ignore_index=True) if nonempty_matches else pd.DataFrame(columns=MATCH_COLUMNS)
    alerts = pd.concat(nonempty_alerts, ignore_index=True) if nonempty_alerts else pd.DataFrame(columns=ALERT_COLUMNS)
    return dict(metrics_row=row_metrics, metrics_event=event_metrics, calibration=pd.DataFrame(calibration_rows),
                event_matches=matches, alerts=alerts, bootstrap_intervals=pd.DataFrame(intervals),
                scenario_metrics=pd.DataFrame(scenario_rows),
                ablation=_ablation(row_metrics, event_metrics) if len(row_metrics) else pd.DataFrame(),
                examples=_examples(matches, alerts, config))
