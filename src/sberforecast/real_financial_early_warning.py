"""E08d date-level diagnostics from frozen REAL E07b/E08b artifacts only.

No label rebuilding, learner, threshold, generated observations or events.
Unknown municipal labels are retained when collapsing the monitoring cohort.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = (
    "key_rate_level", "key_rate_delta_last", "key_rate_change_3m",
    "key_rate_change_6m", "months_since_rate_change", "usd_rub_last",
    "usd_rub_change_1m", "usd_rub_change_3m", "usd_rub_vol_1m", "usd_rub_vol_3m",
)
PRIMARY = {"P1": "key_rate_change_3m", "P2": "usd_rub_change_1m",
           "P3": "usd_rub_change_3m", "P4": "usd_rub_vol_3m"}
CHANGES = {"key_rate_delta_last", "key_rate_change_3m", "key_rate_change_6m",
           "usd_rub_change_1m", "usd_rub_change_3m"}
CONCLUSION = "NO REAL PRECURSOR EVIDENCE"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def month(value) -> pd.Period:
    return pd.Period(str(value)[:7], freq="M")


def month_date(value) -> str:
    return str(month(value).to_timestamp("M").date())


def bool_column(values: pd.Series, name: str) -> pd.Series:
    normalized = values.map(lambda x: x if isinstance(x, (bool, np.bool_)) else
                            True if str(x).lower() == "true" else
                            False if str(x).lower() == "false" else None)
    if normalized.isna().any():
        raise ValueError(f"Invalid boolean values: {name}")
    return normalized.astype(bool)


def verify_hash(root: Path, name: str, digest: str) -> None:
    if sha256(root / name) != digest:
        raise ValueError(f"Frozen REAL source changed: {name}")


def load_real_inputs(root: Path, cfg: dict):
    if cfg["primary_diagnostics"] != PRIMARY or cfg["warning_horizons"] != [1, 3]:
        raise ValueError("The declared primary family or warning horizons changed")
    if cfg["multiple_testing"]["planned_family_size"] != 8:
        raise ValueError("E08d requires the predeclared eight-test Holm family")
    if not all(cfg[k] for k in ["real_data_only", "no_synthetic_data", "no_classifier_fits", "no_threshold_tuning", "no_network"]):
        raise ValueError("E08d REAL-only/no-fitting restrictions changed")
    label_manifest = read_json(root / cfg["label_manifest"])
    financial_manifest = read_json(root / cfg["financial_manifest"])
    checked = {}
    for key in ["event_registry", "label_cases"]:
        name = cfg[key]
        digest = label_manifest["artifacts"][Path(name).name]["sha256"]
        verify_hash(root, name, digest)
        checked[name] = digest
    for key in ["label_protocol", "panel_config"]:
        name = cfg[key]
        digest = label_manifest["inputs"][name.replace("/", "\\")]
        verify_hash(root, name, digest)
        checked[name] = digest
    for name in ["src/sberforecast/early_warning_labels.py", "src/sberforecast/early_warning_full_panel_labels.py"]:
        digest = label_manifest["code"][name.replace("/", "\\")]
        verify_hash(root, name, digest)
        checked[name] = digest
    name = cfg["financial_features"]
    digest = financial_manifest["artifact_hashes"][name]
    verify_hash(root, name, digest)
    checked[name] = digest
    for source in financial_manifest["source_catalog"] + [financial_manifest["decision_ledger"]]:
        name = source.get("cache_path", source.get("path"))
        verify_hash(root, name, source["sha256"])
        checked[name] = source["sha256"]
    for key in ["financial_manifest", "label_manifest"]:
        checked[cfg[key]] = sha256(root / cfg[key])
    financial = pd.read_csv(root / cfg["financial_features"], low_memory=False)
    events = pd.read_csv(root / cfg["event_registry"], dtype={"municipality_id": str})
    cases = pd.read_csv(root / cfg["label_cases"], dtype={"municipality_id": str}, low_memory=False)
    for column in ["eligible_at_origin", "at_risk", "fully_known", "left_insufficient", "right_censored"]:
        cases[column] = bool_column(cases[column], column)
    if cases.duplicated(["municipality_id", "forecast_origin", "k"]).any():
        raise ValueError("Duplicate saved municipality/origin/horizon rows")
    if (~cases.fully_known & cases.label.notna()).any():
        raise ValueError("An unknown saved label was converted to a value")
    if not cases.loc[cases.fully_known, "label"].isin([0, 1]).all():
        raise ValueError("A known saved label is not binary")
    if events.event_id.duplicated().any():
        raise ValueError("Duplicate frozen event IDs")
    financial["forecast_origin"] = financial.forecast_origin.map(month_date)
    duplicated = len(financial) - financial.forecast_origin.nunique()
    for _, group in financial.groupby("forecast_origin"):
        if any(group[c].nunique(dropna=False) != 1 for c in FEATURES + ("max_source_date_used", "max_source_available_at_used")):
            raise ValueError("National copies on the same date differ")
    financial = financial.drop_duplicates("forecast_origin").sort_values("forecast_origin").reset_index(drop=True)
    if len(financial) != 12 or set(financial.forecast_origin) != set(cases.forecast_origin):
        raise ValueError("E08b and E07b origin calendars differ")
    financial["_origin"] = pd.to_datetime(financial.forecast_origin).dt.tz_localize("Europe/Moscow")
    for column in ["max_source_date_used", "max_source_available_at_used",
                   "key_rate_last_available_at", "key_rate_last_effective_date",
                   "usd_rub_last_available_at", "usd_rub_last_effective_date"]:
        maxima = pd.to_datetime(financial[column], utc=True)
        if maxima.isna().any() or (maxima > financial._origin).any():
            raise ValueError(f"Future/unavailable financial source: {column}")
    financial = financial.drop(columns="_origin")
    counts = dict(panel_municipalities=cases.municipality_id.nunique(), weak_events=len(events),
                  unique_onset_dates=events.onset_period.nunique(), municipalities_with_events=events.municipality_id.nunique(),
                  n_origins=cases.forecast_origin.nunique())
    if counts != dict(panel_municipalities=2190, weak_events=73, unique_onset_dates=6, municipalities_with_events=71, n_origins=12):
        raise ValueError("Frozen REAL source counts differ from E07b")
    return financial, events, cases, dict(source_counts=counts, source_sha256=checked,
        financial_rows_collapsed=duplicated, financial_source_cutoff_violations=0,
        synthetic_inputs=0, classifier_fits=0, threshold_searches=0, label_rebuilds=0)


def ids_from_cells(values: pd.Series) -> set[str]:
    return {item for cell in values.dropna().astype(str) for item in cell.split("|") if item}


def build_origin_study(financial: pd.DataFrame, events: pd.DataFrame, cases: pd.DataFrame) -> pd.DataFrame:
    """Strict OR of original known labels; registry counts remain observed counts."""
    event_by_id = events.set_index("event_id")
    all_onsets = sorted(events.onset_period.map(month).unique())
    rows = []
    for record in financial.sort_values("forecast_origin").to_dict("records"):
        origin = month(record["forecast_origin"])
        for k in [1, 3]:
            g = cases.loc[cases.forecast_origin.eq(record["forecast_origin"]) & cases.k.eq(k)]
            monitor = g.loc[g.eligible_at_origin & g.at_risk]
            witnesses = monitor.loc[monitor.fully_known & monitor.label.eq(1)]
            negative = len(monitor) > 0 and bool((monitor.fully_known & monitor.label.eq(0)).all())
            positive = not witnesses.empty
            known = positive or negative
            label = 1 if positive else 0 if negative else np.nan
            time_values = witnesses.label_known_at if positive else monitor.label_known_at if negative else pd.Series(dtype=str)
            known_at = (min(time_values) if positive else max(time_values)) if known else ""
            if known and month(known_at) < origin + k + 2:
                raise ValueError("Date-level label availability precedes the E07b uniform window")
            registry = events.loc[events.onset_period.map(month).map(lambda p: origin < p <= origin + k)]
            linked_ids = ids_from_cells(witnesses.positive_event_ids)
            if not linked_ids.issubset(event_by_id.index):
                raise ValueError("Known positive links to an unregistered event")
            if not linked_ids.issubset(set(registry.event_id)):
                raise ValueError("Known positive event lies outside its target window")
            if positive and registry.empty:
                raise ValueError("Date-positive label has no frozen registry onset")
            suffix = f"next_{k}m"
            record[f"known_label_k{k}"] = known
            record[f"event_{suffix}"] = label
            record[f"event_count_{suffix}"] = len(registry) if known else np.nan
            record[f"affected_municipality_count_{suffix}"] = registry.municipality_id.nunique() if known else np.nan
            record[f"affected_share_{suffix}"] = registry.municipality_id.nunique() / 2190 if known else np.nan
            record[f"event_count_complete_k{k}"] = known and bool(monitor.fully_known.all())
            record[f"label_known_at_k{k}"] = known_at
            record[f"monitoring_eligible_at_risk_count_k{k}"] = len(monitor)
            record[f"monitoring_known_municipality_count_k{k}"] = int(monitor.fully_known.sum())
            record[f"monitoring_unknown_municipality_count_k{k}"] = int((~monitor.fully_known).sum())
            record[f"monitoring_positive_municipality_count_k{k}"] = len(witnesses)
            record[f"monitoring_positive_event_count_k{k}"] = len(linked_ids) if known else np.nan
            record[f"label_status_k{k}"] = "KNOWN_POSITIVE_WITNESS_PARTIAL_COUNT_COVERAGE" if positive else \
                "KNOWN_NEGATIVE_COMPLETE_MONITORING_COHORT" if negative else \
                "UNKNOWN_LEFT_INSUFFICIENT" if len(monitor) and monitor.left_insufficient.all() else \
                "UNKNOWN_RIGHT_CENSORED" if len(monitor) and monitor.right_censored.all() else "UNKNOWN_INCOMPLETE_COVERAGE"
        following = [p for p in all_onsets if p > origin]
        record["distance_to_next_onset_months"] = following[0].ordinal - origin.ordinal if following else np.nan
        record["distance_to_next_onset_evaluation_only"] = True
        record["distance_status"] = "RETROSPECTIVE_REGISTERED_ONSET" if following else "NO_LATER_REGISTERED_ONSET_FUTURE_UNOBSERVED"
        rows.append(record)
    study = pd.DataFrame(rows).sort_values("forecast_origin").reset_index(drop=True)
    for k in [1, 3]:
        study[f"event_next_{k}m"] = study[f"event_next_{k}m"].astype("Int64")
        if study.loc[~study[f"known_label_k{k}"], f"event_next_{k}m"].notna().any():
            raise ValueError("Unknown date became negative")
    return study


def eligibility_summary(study: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([dict(k=k, n_origins=len(study), n_eligible=int(study[f"known_label_k{k}"].sum()),
        n_positive=int(study[f"event_next_{k}m"].eq(1).sum()), n_negative=int(study[f"event_next_{k}m"].eq(0).sum()),
        n_unknown=int((~study[f"known_label_k{k}"]).sum())) for k in [1, 3]])


def event_windows(study: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    lookup = study.set_index("forecast_origin")
    rows = []
    for onset, group in events.groupby("onset_period", sort=True):
        period = month(onset)
        previous = {}
        for k in [1, 3]:
            candidates = study.loc[study[f"known_label_k{k}"] & study.forecast_origin.map(month).map(lambda p: p < period <= p + k)]
            previous[k] = candidates.forecast_origin.max() if len(candidates) else ""
        for lead in [1, 2, 3]:
            issued = month_date(period - lead)
            available = issued in lookup.index
            row = dict(onset_period=onset, onset_date=month_date(period), event_count=len(group),
                affected_municipalities=group.municipality_id.nunique(), affected_share_population=group.municipality_id.nunique()/2190,
                lead_months=lead, forecast_origin=issued, financial_available=available,
                previous_eligible_origin_k1=previous[1], previous_eligible_origin_k3=previous[3], onset_date_is_month_marker=True)
            for feature in FEATURES:
                row[feature] = lookup.at[issued, feature] if available else np.nan
            for k in [1, 3]:
                row[f"known_label_k{k}"] = bool(lookup.at[issued, f"known_label_k{k}"]) if available else False
            rows.append(row)
    return pd.DataFrame(rows)


def diagnostic_specs():
    """Primary family is fixed before reading financial outcomes."""
    reverse = {feature: name for name, feature in PRIMARY.items()}
    specs = []
    for feature in FEATURES:
        primary = feature in reverse
        name = reverse.get(feature, f"S{FEATURES.index(feature) + 1}")
        specs.append((name, feature, "PRIMARY" if primary else "SECONDARY",
                      "signed" if feature in CHANGES else "intrinsic"))
        if feature in CHANGES:
            specs.append((name, feature, "PRIMARY" if primary else "SECONDARY", "absolute_descriptive"))
    return specs


def comparison(study, k, diagnostic_id, feature, role, transform):
    selected = study.loc[study[f"known_label_k{k}"]]
    values = pd.to_numeric(selected[feature], errors="raise").to_numpy(dtype=float)
    if transform == "absolute_descriptive":
        values = np.abs(values)
    finite = np.isfinite(values)
    labels = selected[f"event_next_{k}m"].to_numpy(dtype=float)
    pos = values[finite & (labels == 1)]
    neg = values[finite & (labels == 0)]
    estimable = len(pos) > 0 and len(neg) > 0
    status = "ESTIMABLE" if estimable else "NONESTIMABLE_NO_NEGATIVE_DATES" if len(pos) else \
        "NONESTIMABLE_NO_POSITIVE_DATES" if len(neg) else "NONESTIMABLE_NO_KNOWN_DATES"
    rank = float(np.sign(pos[:, None] - neg[None, :]).mean()) if estimable else np.nan
    return dict(diagnostic_id=diagnostic_id, feature=feature, role=role, transform=transform, k=k,
        n_eligible=len(selected), n_positive=len(pos), n_negative=len(neg), n_missing_feature=int((~finite).sum()),
        positive_mean=float(pos.mean()) if len(pos) else np.nan,
        negative_mean=float(neg.mean()) if len(neg) else np.nan,
        positive_median=float(np.median(pos)) if len(pos) else np.nan,
        negative_median=float(np.median(neg)) if len(neg) else np.nan,
        difference_in_means=float(pos.mean()-neg.mean()) if estimable else np.nan,
        difference_in_medians=float(np.median(pos)-np.median(neg)) if estimable else np.nan,
        rank_biserial=rank, comparison_status=status)


def comparisons(study):
    return pd.DataFrame([comparison(study, k, *spec) for k in [1, 3] for spec in diagnostic_specs()])


def permutation_results(study, cfg):
    rows = []
    for k in [1, 3]:
        for name, feature in PRIMARY.items():
            transform = "signed" if feature in CHANGES else "intrinsic"
            record = comparison(study, k, name, feature, "PRIMARY", transform)
            selected = study.loc[study[f"known_label_k{k}"]]
            finite = np.isfinite(selected[feature].to_numpy(dtype=float))
            values = selected.loc[finite, feature].to_numpy(dtype=float)
            npos, nneg = record["n_positive"], record["n_negative"]
            possible = math.comb(len(values), npos)
            estimable = record["comparison_status"] == "ESTIMABLE"
            p = np.nan
            evaluated = 0
            if estimable:
                if possible > cfg["permutation"]["maximum_assignments"]:
                    raise ValueError("Exact date-level assignment limit exceeded; no random approximation")
                observed = abs(record["difference_in_means"])
                extreme = 0
                for indices in itertools.combinations(range(len(values)), npos):
                    mask = np.zeros(len(values), dtype=bool)
                    mask[list(indices)] = True
                    statistic = abs(float(values[mask].mean()-values[~mask].mean()))
                    extreme += statistic >= observed - 1e-12 * max(1.0, observed)
                    evaluated += 1
                p = extreme / evaluated
            rows.append(dict(**record, raw_exact_p=p, holm_adjusted_p=np.nan,
                permutations_possible=possible, permutations_evaluated=evaluated,
                test_status="EXACT_DESCRIPTIVE_REFERENCE" if estimable else record["comparison_status"],
                holm_status="NONESTIMABLE_P_VALUE" if not estimable else "PLANNED_EIGHT_TEST_FAMILY"))
    result = pd.DataFrame(rows)
    # Missing tests remain NA. They never become p=1 or shrink the planned family.
    previous = 0.0
    available = result.raw_exact_p.dropna().sort_values(kind="stable")
    for order, (idx, p) in enumerate(available.items()):
        previous = max(previous, min(1.0, (8-order)*p))
        result.at[idx, "holm_adjusted_p"] = previous
    return result


def spearman(values, outcomes):
    x = np.asarray(values, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if len(x) < 2:
        return len(x), np.nan, "NONESTIMABLE_TOO_FEW_DATES"
    if len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return len(x), np.nan, "NONESTIMABLE_CONSTANT_VALUES"
    rx = pd.Series(x).rank(method="average").to_numpy()
    ry = pd.Series(y).rank(method="average").to_numpy()
    return len(x), float(np.corrcoef(rx, ry)[0, 1]), "DESCRIPTIVE_OBSERVED_REGISTRY_INTENSITY"


def event_intensity(study):
    rows = []
    for k in [1, 3]:
        selected = study.loc[study[f"known_label_k{k}"]]
        for name, feature, role, transform in diagnostic_specs():
            values = selected[feature].to_numpy(dtype=float)
            if transform == "absolute_descriptive":
                values = abs(values)
            for outcome, column in [
                ("event_count", f"event_count_next_{k}m"),
                ("affected_municipality_count", f"affected_municipality_count_next_{k}m"),
                ("affected_share", f"affected_share_next_{k}m"),
            ]:
                n, rho, status = spearman(values, selected[column])
                rows.append(dict(diagnostic_id=name, feature=feature, role=role, transform=transform,
                    k=k, outcome=outcome, n_dates=n, spearman_rho=rho, status=status))
    return pd.DataFrame(rows)


def leave_one_event_out(study, events):
    rows = []
    for onset in sorted(events.onset_period.unique()):
        event_month = month(onset)
        for k in [1, 3]:
            remove = study[f"known_label_k{k}"] & study.forecast_origin.map(month).map(lambda o: o < event_month <= o+k)
            removed = study.loc[remove, "forecast_origin"].tolist()
            remaining = study.loc[~remove]
            for spec in diagnostic_specs():
                if spec[2] != "PRIMARY":
                    continue
                full = comparison(study, k, *spec)
                record = comparison(remaining, k, *spec)
                effect = record["difference_in_means"]
                full_effect = full["difference_in_means"]
                estimable = np.isfinite(effect) and np.isfinite(full_effect)
                rows.append(dict(**record, omitted_onset_date=month_date(event_month),
                    n_origins_removed=len(removed), removed_origins=json.dumps(removed),
                    removal_status="REMOVED" if removed else "NOOP_ONSET_NOT_REPRESENTED",
                    effect_sign=float(np.sign(effect)) if estimable else np.nan,
                    effect_ratio_to_full=float(effect/full_effect) if estimable and full_effect != 0 else np.nan,
                    stability_status="DESCRIPTIVE_COMPARISON_AVAILABLE" if estimable and removed else
                        "NOOP_NOT_INDEPENDENT_SUPPORT" if not removed else "NONESTIMABLE_CONTRAST",
                    one_event_driven=np.nan))
    return pd.DataFrame(rows)


def calculate_tables(financial, events, cases, cfg):
    study = build_origin_study(financial, events, cases)
    return {"origin_event_study.csv": study, "eligibility_summary.csv": eligibility_summary(study),
        "event_window_table.csv": event_windows(study, events),
        "diagnostic_comparisons.csv": comparisons(study),
        "permutation_results.csv": permutation_results(study, cfg),
        "event_intensity.csv": event_intensity(study),
        "leave_one_event_out.csv": leave_one_event_out(study, events)}
