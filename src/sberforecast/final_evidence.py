"""Read-only evidence for F1; real feasibility and synthetic utility stay separate.

Only saved artifacts are read. No model, builder, source adapter, network or
experiment API is imported. Important aggregate counts/metrics are recomputed
and compared with saved summaries and Markdown tables. Missing required files
or discrepant numbers raise ValueError, instead of silently publishing a partial
or invented result. NaN/undefined metrics become JSON null, never zero.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
from numbers import Integral
from pathlib import Path
import re

import pandas as pd


METRICS = ("pr_auc", "row_f1", "event_recall", "alert_precision", "median_lead_time_months",
           "mean_lead_time_months", "false_alerts_per_12_monitored_months")
NEWS = "outputs/news_events_v3/"
REAL_A = "outputs/early_warning_feasibility_v1/"
REAL_B = "outputs/early_warning_full_panel_v1/"
SYNTHETIC = "outputs/early_warning_synthetic_v1/"
REPORTS = {
    "E05a": "reports/results/E05a_macro_data_audit.md",
    "E06b": "reports/results/E06b_news_events.md",
    "E07a": "reports/results/E07a_early_warning_feasibility.md",
    "E07b": "reports/results/E07b_early_warning_full_panel.md",
    "E07c": "reports/results/E07c_synthetic_early_warning.md",
}


def _require(condition, message):
    if not bool(condition):
        raise ValueError("Final evidence mismatch: " + message)


def _clean(value):
    if isinstance(value, Mapping):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if hasattr(value, "item"):
        return _clean(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (pd.Timestamp, pd.Period)):
        return str(value)
    raise TypeError(f"Unsupported evidence value: {type(value).__name__}")


class _Reader:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.provenance = {}

    def path(self, name: str, role: str) -> Path:
        path = (self.root / name).resolve()
        _require(path.is_relative_to(self.root), "artifact outside project root")
        if not path.is_file():
            raise ValueError("Required final evidence artifact missing: " + name)
        if name not in self.provenance:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            self.provenance[name] = dict(path=name, sha256=digest.hexdigest(), role=role)
        return path

    def csv(self, name, role="saved aggregate or audit rows", usecols=None):
        return pd.read_csv(self.path(name, role), low_memory=False, float_precision="round_trip",
                           dtype={"municipality_id": str, "series_id": str}, usecols=usecols)

    def json(self, name, role="saved summary or protocol"):
        return json.loads(self.path(name, role).read_text(encoding="utf-8"))

    def text(self, name, role="saved report or fixed configuration"):
        return self.path(name, role).read_text(encoding="utf-8")


def _flags(values):
    _require(values.notna().all() and values.isin([True, False, 0, 1]).all(), "nonboolean saved eligibility flag")
    return values.astype(bool)


def _assert_number(actual, expected, label, *, rounded=False):
    if pd.isna(actual) or actual is None:
        _require(expected is None or pd.isna(expected), label + ": undefined value must remain null")
    else:
        _require(expected is not None and not pd.isna(expected), label + ": missing saved number")
        if isinstance(actual, Integral):
            _require(float(actual) == float(expected), label + ": integer count differs")
            return
        _require(math.isclose(float(actual), float(expected), rel_tol=6e-6 if rounded else 1e-10,
                              abs_tol=5e-7 if rounded else 1e-12), label)


def _markdown_tables(text):
    tables, rows = [], []
    for line in text.splitlines() + [""]:
        if line.strip().startswith("|"):
            rows.append([cell.strip().replace("\\|", "|") for cell in line.strip().strip("|").split("|")])
        elif rows:
            if len(rows) >= 2 and all(re.fullmatch(r":?-+:?", cell.replace(" ", "")) for cell in rows[1]):
                tables.append((rows[0], [dict(zip(rows[0], row)) for row in rows[2:] if len(row) == len(rows[0])]))
            rows = []
    return tables


def _verify_report_table(text, frame, keys, columns, label):
    tables = [(header, rows) for header, rows in _markdown_tables(text) if set(keys + columns).issubset(header)]
    _require(bool(tables), label + ": required saved Markdown table absent")
    rows = tables[0][1]
    by_key = {tuple(row[key] for key in keys): row for row in rows}
    _require(len(by_key) == len(rows), label + ": duplicate report row keys")
    for item in frame.to_dict("records"):
        key = tuple(str(item[name]) for name in keys)
        _require(key in by_key, label + ": missing report row " + str(key))
        for column in columns:
            token = by_key[key][column]
            expected = None if token in ("", "—", "null", "None", "nan") else float(token)
            _assert_number(item[column], expected, f"{label}/{key}/{column}", rounded=True)


def _news(reader, reports):
    fields = ["document_id", "snapshot_id", "source_url", "canonical_event_id", "published_at", "available_at",
              "availability_status", "geography_level", "event_type"]
    docs = reader.csv(NEWS + "documents.csv.gz", "all retained news snapshots", fields)
    events = reader.csv(NEWS + "events.csv.gz", "first-available canonical news groups", fields)
    features = reader.csv(NEWS + "news_features.csv.gz", "news features on original 64 by 12 origin grid")
    dictionary = reader.csv(NEWS + "feature_dictionary.csv", "fixed news value feature definitions")
    summary = reader.json(NEWS + "coverage_audit.json")
    feature_names = [name for name in dictionary.feature if not name.endswith("_missing")]
    flags = [name + "_missing" for name in feature_names]
    _require(set(dictionary.feature) == set(feature_names + flags) and set(feature_names + flags).issubset(features),
             "news dictionary/missing flags do not match matrix")
    _require(not features.duplicated(["municipality_id", "forecast_origin"]).any(), "duplicate news forecast cases")
    _require(not docs.snapshot_id.duplicated().any() and not events.canonical_event_id.duplicated().any(), "duplicate news snapshot/canonical ID")
    origin = pd.to_datetime(features.forecast_origin, utc=True).max()
    published = pd.to_datetime(docs.published_at, utc=True, format="mixed")
    available = pd.to_datetime(docs.available_at, utc=True, format="mixed")
    admitted = docs.loc[published.le(available) & available.le(origin) & docs.availability_status.eq("confirmed_historical")]
    all_historical = docs.loc[docs.availability_status.eq("confirmed_historical")]
    vectors = features[feature_names + flags].drop_duplicates()
    spatial = features.groupby("forecast_origin")[feature_names + flags].apply(lambda part: len(part.drop_duplicates())).max()
    counts = dict(snapshots=len(docs), documents=docs.document_id.nunique(), canonical_events=len(events),
                  duplicates_removed=docs.document_id.nunique() - len(events),
                  admitted_core_records=int(docs.event_type.eq("policy_decision").sum()),
                  historical_documents_at_last_origin=admitted.document_id.nunique(),
                  historical_snapshots_at_last_origin=len(admitted), feature_rows=len(features),
                  unique_temporal_vectors=len(vectors), nonzero_30d_rows=int(features.news_count_30d.gt(0).sum()),
                  nonzero_30d_origins=features.loc[features.news_count_30d.gt(0), "forecast_origin"].nunique(),
                  regional_nonzero_rows=int(features.regional_news_count_30d.gt(0).sum()))
    for name, value in counts.items():
        _assert_number(value, summary[name], "news summary " + name)
    historical_by_geo = {level: int(all_historical.geography_level.eq(level).sum()) for level in ("national", "regional", "municipality")}
    _require(spatial == 1 and not _flags(features.source_archive_complete).any(), "news spatial copies/archive-completeness mismatch")
    coverage = {"URL-документы": counts["documents"], "Снимки/извлечённые версии": counts["snapshots"],
                "Canonical группы событий": counts["canonical_events"],
                "Исторически доступные URL к последней O": counts["historical_documents_at_last_origin"],
                "Уникальные временные news-векторы": counts["unique_temporal_vectors"],
                "Строки с news_30d>0": counts["nonzero_30d_rows"]}
    report_table = next((rows for header, rows in _markdown_tables(reports["E07a"]) if {"Показатель", "v3"}.issubset(header)), None)
    _require(report_table is not None, "E07a news coverage table missing")
    report_values = {row["Показатель"]: row["v3"] for row in report_table}
    for name, value in coverage.items():
        _assert_number(value, float(report_values[name]), "E07a news report " + name)
    return dict(data_status="real", policy_version="v3", **counts,
                municipalities=features.municipality_id.nunique(), forecast_origins=features.forecast_origin.nunique(),
                value_features=len(feature_names), missing_flags=len(flags), total_numeric_features=len(feature_names + flags),
                historical_admitted_snapshots=len(all_historical), historical_by_geography=historical_by_geo,
                municipal_nonzero_rows=int(features.municipal_news_count_30d.gt(0).sum()),
                maximum_spatial_vectors_per_origin=int(spatial), source_archive_complete=False,
                historical_policy="explicit dated-original official decision-core/PDF archive trust; no independent historical SHA",
                zero_interpretation="no admitted document in bounded corpus, not absence of a real event",
                source_paths=[NEWS + "documents.csv.gz", NEWS + "news_features.csv.gz", REPORTS["E07a"]])


def _external(reader, reports):
    directory = "outputs/macro_data_audit_v1/"
    table = reader.csv(directory + "macro_table.csv.gz", "national annual forecasts, A archive-trust versus B current workbook",
                       ["indicator", "availability_status", "geographic_level", "is_forecast", "value"])
    saved = reader.json(directory + "summary.json")
    rosstat = reader.json("outputs/e05a_checks/rosstat_catalog.json", "preserved unavailable regional CPI/wage source catalog")
    audit = reader.json("outputs/e05c_checks/source_audit.json", "which A forecast indicators were independently admitted for E05c")
    diagnostics = reader.json("outputs/macro_forecast_v1/training_macro_diagnostics.json", "macro indicators actually joined in executed E05c")
    run = reader.json("outputs/macro_forecast_v1/run_status.json", "executed E05c status")
    by_class = table.groupby("availability_status").size().to_dict()
    by_indicator = {f"{status}:{indicator}": len(part) for (status, indicator), part in table.groupby(["availability_status", "indicator"])}
    _require(by_indicator == saved["source_rows"], "macro source counts differ from E05a summary")
    _require(_flags(table.is_forecast).all() and table.geographic_level.eq("national").all(), "macro source forecast/national meaning changed")
    _require(audit["status"] == "PASS" and audit["checks"]["audited_A_rows"] == by_class.get("A", 0), "E05c admitted macro audit mismatch")
    _require(run["complete"] and run["n_new_fits_succeeded"] > 0, "E05c use was not actually executed")
    used = sorted({row["indicator"] for row in diagnostics if row["scope"] == "training" and row["n_available"] > 0})
    _require(set(used).issubset(set(table.loc[table.availability_status.eq("A"), "indicator"])), "actually-used macro outside admitted A sources")
    unavailable = [dict(indicator=row["indicator"], availability_class=row["availability_status"],
                        status=row["status"], parsed_observations=row["parsed_observations"], data_status="not_evaluated") for row in rosstat]
    _require(all(row["parsed_observations"] == 0 and not row["files"] for row in rosstat), "Rosstat unavailable-source status changed")
    _require(re.search(rf"Прочитано\s+\*\*{len(table)}\s+прогнозных строк", reports["E05a"]) is not None, "E05a macro row count report mismatch")
    _require(re.search(rf"Всего A\s*—\s*{by_class.get('A', 0)}, B\s*—\s*{by_class.get('B', 0)}", reports["E05a"]) is not None,
             "E05a A/B report count mismatch")
    return dict(data_status="real", forecast_rows=len(table), rows_by_availability_class=by_class, rows_by_indicator=by_indicator,
                actually_used=dict(data_status="real", experiment="E05c", indicators=used, geography="national",
                                   meaning="annual analyst inflation expectations and CBR real household-consumption growth forecast",
                                   availability="A under explicit official dated archive trust, no independent historical SHA"),
                scenario_only=dict(data_status="diagnostic", availability_class="B", rows=by_class.get("B", 0),
                                   indicators=sorted(table.loc[table.availability_status.eq("B"), "indicator"].unique()),
                                   actually_used_in_E05c=False, deflation_executed=False),
                unavailable=unavailable, regional_monthly_cpi_and_wage_observations=0,
                source_paths=[directory + "macro_table.csv.gz", "outputs/macro_forecast_v1/training_macro_diagnostics.json", REPORTS["E05a"]])


def _event_dates(cases, events):
    linked = set()
    for value in cases.loc[cases.label.eq(1), "positive_event_ids"].dropna():
        rendered = str(value).strip()
        linked.update(json.loads(rendered) if rendered.startswith("[") else rendered.split("|"))
    _require(linked.issubset(set(events.event_id)), "positive weak case links unknown event")
    return events.loc[events.event_id.isin(linked), "onset_period"].nunique()


def _real_counts(cases, events):
    _require(not cases.duplicated(["municipality_id", "forecast_origin", "k"]).any(), "duplicate real warning case")
    known = _flags(cases.fully_known)
    _require(cases.loc[~known, "label"].isna().all() and cases.loc[known, "label"].isin([0, 1]).all(), "unknown/binary real labels changed")
    eligible = _flags(cases.eligible_at_origin) if "eligible_at_origin" in cases else pd.Series(True, index=cases.index)
    admitted = cases.loc[known & _flags(cases.at_risk) & eligible].copy()
    known_at = pd.to_datetime(admitted.label_known_at, utc=True, format="mixed")
    origin = pd.to_datetime(admitted.forecast_origin, utc=True)
    train = admitted.loc[known_at.le(pd.Timestamp("2024-07-31", tz="UTC")) & origin.lt(pd.Timestamp("2024-08-01", tz="UTC"))]
    test = admitted.loc[origin.ge(pd.Timestamp("2024-08-01", tz="UTC")) & origin.le(pd.Timestamp("2024-11-30", tz="UTC"))]
    rows = []
    for k in (1, 3):
        for scope, selected in (("all", cases), ("at_risk_evaluable", admitted), ("train", train), ("test", test)):
            part = selected.loc[selected.k.eq(k)]
            rows.append(dict(k=k, scope=scope, cases=len(part), fully_known=int(_flags(part.fully_known).sum()),
                positives=int(part.label.eq(1).sum()), negatives=int(part.label.eq(0).sum()),
                censored_or_missing=int(part.label.isna().sum()), right_censored=int(_flags(part.right_censored).sum()),
                positive_event_onset_dates=int(_event_dates(part, events)), forecast_origin_dates=part.forecast_origin.nunique()))
    return pd.DataFrame(rows)


def _real(reader, reports):
    records, result = [], {"data_status": "real", "truth_status": "diagnostic",
                           "truth_definition": "weak sustained saved SeasonalNaiveYoY forecast-error shift, not expert economic shock truth"}
    for experiment, directory, case_file, event_file in (
        ("E07a", REAL_A, "weak_cases.csv.gz", "weak_events.csv.gz"),
        ("E07b", REAL_B, "label_cases.csv.gz", "event_registry_weak.csv"),
    ):
        usecols = ["municipality_id", "forecast_origin", "k", "label", "fully_known", "at_risk", "right_censored", "label_known_at", "positive_event_ids"]
        if experiment == "E07b":
            usecols.append("eligible_at_origin")
        cases = reader.csv(directory + case_file, "real weak cases including censored/unknown and active-excluded rows", usecols)
        events = reader.csv(directory + event_file, "diagnostic weak event registry on real saved residuals")
        _require(not events.event_id.duplicated().any(), experiment + " duplicate weak event IDs")
        counts = _real_counts(cases, events)
        if experiment == "E07a":
            saved = reader.json(directory + "feasibility_summary.json")
            _assert_number(len(events), saved["weak_event_count"], "E07a weak events")
            _assert_number(events.onset_period.nunique(), saved["weak_event_unique_onset_dates"], "E07a onset dates")
            _require(not saved["classifier_fit"] and not saved["classification_metrics_obtained"], "E07a classifier unexpectedly evaluated")
            for row in saved["split"]:
                scope = "test" if row["split"] == "test_evaluable" else row["split"]
                if scope not in ("train", "test"):
                    continue
                actual = counts.loc[counts.k.eq(row["k"]) & counts.scope.eq(scope)].iloc[0]
                for expected, field in (("cases", "cases"), ("positive", "positives"), ("negative", "negatives"), ("event_onset_dates", "positive_event_onset_dates")):
                    _assert_number(actual[field], row[expected], f"E07a/{row['k']}/{scope}/{field}")
            models = ["not_trained"]
        else:
            saved = reader.csv(directory + "feasibility.csv")
            for actual in counts.to_dict("records"):
                expected = saved.loc[saved.k.eq(actual["k"]) & saved.scope.eq(actual["scope"])].iloc[0]
                for name in ("cases", "fully_known", "positives", "negatives", "censored_or_missing", "right_censored", "positive_event_onset_dates", "forecast_origin_dates"):
                    _assert_number(actual[name], expected[name], f"E07b/{actual['k']}/{actual['scope']}/{name}")
            _verify_report_table(reports["E07b"], counts, ["k", "scope"], ["cases", "fully_known", "positives", "negatives", "censored_or_missing", "positive_event_onset_dates"], "E07b feasibility report")
            status = reader.json(directory + "execution_status.json")
            _require(status["classifiers_fitted"] == 0 and not status["classification_metrics_obtained"], "E07b classifier unexpectedly evaluated")
            for name in ("metrics_row.csv", "metrics_event.csv", "predictions.csv.gz"):
                _require(reader.csv(directory + name).empty, "unevaluated E07b contains classifier results: " + name)
            model_status = reader.csv(directory + "model_status.csv")
            _require(not _flags(model_status.fit_performed).any(), "E07b model fit status contradiction")
            gate = reader.csv(directory + "feasibility_gate.csv")
            _require(not _flags(gate.gate_pass).any(), "E07b gate no longer failed")
            models = sorted(model_status.model.unique())
        experiment_result = dict(data_status="real", municipalities=cases.municipality_id.nunique(),
            weak_events=len(events), weak_event_onset_dates=events.onset_period.nunique(), municipalities_with_event=events.municipality_id.nunique(),
            case_rows=len(cases), counts=counts.to_dict("records"), classifiers_fitted=0,
            metric_status="not_evaluated", metrics={name: None for name in METRICS},
            source_paths=[directory + case_file, directory + event_file, REPORTS[experiment]])
        result[experiment] = experiment_result
        for k in (1, 3):
            train = counts.loc[counts.k.eq(k) & counts.scope.eq("train")].iloc[0]
            test = counts.loc[counts.k.eq(k) & counts.scope.eq("test")].iloc[0]
            for model in models:
                records.append(dict(experiment=experiment, data_status="not_evaluated", truth_data_status="real",
                    evaluation_status="classifier_not_trained_feasibility_failed", cohort="test", scenario="real_weak_truth",
                    model=model, k=k, cases=int(test.cases), positives=int(test.positives), negatives=int(test.negatives),
                    train_positives=int(train.positives), test_positives=int(test.positives),
                    train_positive_onset_dates=int(train.positive_event_onset_dates), test_positive_onset_dates=int(test.positive_event_onset_dates),
                    **{name: None for name in METRICS}, source_path=directory + case_file, source_report=REPORTS[experiment]))
    return result, records


def _row_metrics(predictions):
    _require(predictions.label.isin([0, 1]).all(), "synthetic ranking labels not binary")
    probability = pd.to_numeric(predictions.probability, errors="raise")
    _require(probability.between(0, 1).all(), "invalid synthetic probability")
    label = predictions.label
    alert = _flags(predictions.alert)
    tp, fp, fn = int((label.eq(1) & alert).sum()), int((label.eq(0) & alert).sum()), int((label.eq(1) & ~alert).sum())
    positives = int(label.sum())
    average_precision = None
    if positives:
        ties = pd.DataFrame(dict(probability=probability, label=label)).groupby("probability").label.agg(["sum", "size"]).sort_index(ascending=False)
        cumulative_pos, cumulative_count = ties["sum"].cumsum(), ties["size"].cumsum()
        average_precision = float(((ties["sum"] / positives) * (cumulative_pos / cumulative_count)).sum())
    return dict(cases=len(predictions), positives=positives, negatives=len(predictions) - positives,
                pr_auc=average_precision, row_f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0 if len(predictions) else None)


def _event_metrics(predictions, matches, alerts):
    warned = matches.loc[_flags(matches.warning_observed)]
    false = int(_flags(alerts.false_alert).sum())
    successes = alerts.loc[_flags(alerts.successful_warning)]
    _require(not matches.event_id.duplicated().any() and len(successes) == len(warned)
             and not successes.event_id.duplicated().any() and set(successes.event_id) == set(warned.event_id),
             "synthetic one-event-one-warning violated")
    horizon = int(predictions.k.iloc[0]) if len(predictions) else 3
    _require(successes.lead_time_months.between(1, horizon).all()
             and warned.lead_time_months.between(1, horizon).all(), "late synthetic alert counted as warning")
    _require(not (_flags(alerts.repeated_warning) & (_flags(alerts.false_alert) | _flags(alerts.successful_warning))).any(),
             "repeated synthetic warning counted as success/false alert")
    n, n_warned = len(matches), len(warned)
    return dict(eligible_events=n, warned_events=n_warned, event_recall=n_warned / n if n else None,
                alert_precision=n_warned / (n_warned + false) if n_warned + false else None,
                median_lead_time_months=float(warned.lead_time_months.median()) if n_warned else None,
                mean_lead_time_months=float(warned.lead_time_months.mean()) if n_warned else None,
                false_alert_count=false, false_alerts_per_12_monitored_months=false * 12 / len(predictions) if len(predictions) else None,
                repeated_alert_count=int(_flags(alerts.repeated_warning).sum()))


def _synthetic(reader, reports):
    predictions = reader.csv(SYNTHETIC + "predictions.csv.gz", "fixed VALIDATION and independent TEST predictions")
    metadata = reader.csv(SYNTHETIC + "series_metadata.csv", "synthetic cohort/scenario metadata, not model features")
    events = reader.csv(SYNTHETIC + "true_events.csv", "synthetic generator onset truth")
    matches, alerts = reader.csv(SYNTHETIC + "event_matches.csv"), reader.csv(SYNTHETIC + "alerts.csv")
    saved_rows, saved_events = reader.csv(SYNTHETIC + "metrics_row.csv"), reader.csv(SYNTHETIC + "metrics_event.csv")
    saved_scenarios = reader.csv(SYNTHETIC + "scenario_metrics.csv")
    thresholds = reader.csv(SYNTHETIC + "selected_thresholds.csv", "thresholds selected on VALIDATION only")
    intervals = reader.csv(SYNTHETIC + "bootstrap_intervals.csv", "whole-series TEST bootstrap intervals")
    execution = reader.json(SYNTHETIC + "execution_status.json")
    _require(execution["synthetic"] and not execution["real_data_used"] and not execution["test_used_for_fit"] and not execution["test_used_for_threshold"], "synthetic/real or fit/threshold scope changed")
    _require(not predictions.duplicated(["series_id", "cohort", "month_index", "model", "k"]).any(), "duplicate synthetic prediction")
    cohorts = []
    for cohort, part in metadata.groupby("cohort", sort=True):
        own = events.loc[events.cohort.eq(cohort)]
        cohorts.append(dict(cohort=cohort, series=len(part), events=len(own),
            anticipated_events=int(_flags(own.is_anticipated).sum()), unanticipated_events=int((~_flags(own.is_anticipated)).sum()),
            controls=int((~_flags(part.has_event)).sum()), false_precursor_controls=int(_flags(part.false_precursor).sum()), data_status="synthetic"))
    _verify_report_table(reports["E07c"], pd.DataFrame(cohorts), ["cohort"], ["series", "events", "false_precursor_controls"], "E07c cohorts")
    records, row_comparison, event_comparison = [], [], []
    for (model, k), frame in predictions.loc[predictions.cohort.eq("test")].groupby(["model", "k"], sort=True):
        selection = thresholds.loc[thresholds.model.eq(model) & thresholds.k.eq(k)]
        _require(len(selection) == 1 and selection.iloc[0].cohort == "validation", "synthetic threshold provenance changed")
        _require(frame.threshold.eq(selection.iloc[0].threshold).all() and _flags(frame.alert).eq(frame.probability.ge(frame.threshold)).all(), "saved TEST alert/threshold contradiction")
        own_matches = matches.loc[matches.cohort.eq("test") & matches.model.eq(model) & matches.k.eq(k)]
        own_alerts = alerts.loc[alerts.cohort.eq("test") & alerts.model.eq(model) & alerts.k.eq(k)]
        own_meta = metadata.loc[metadata.cohort.eq("test")]
        control = ~_flags(own_meta.has_event)
        for scenario in ("base_all", "weak_precursor", "strong_precursor", "high_noise", "controls_false_precursor", "anticipated_events", "unanticipated_events"):
            selected = pd.Series(True, index=own_meta.index)
            if scenario in ("weak_precursor", "strong_precursor"):
                selected = own_meta.precursor_class.eq(scenario.split("_")[0])
            elif scenario == "high_noise":
                selected = own_meta.noise_class.eq("high")
            elif scenario == "controls_false_precursor":
                selected = control & _flags(own_meta.false_precursor)
            elif scenario == "anticipated_events":
                selected = control | _flags(own_meta.is_anticipated)
            elif scenario == "unanticipated_events":
                selected = control | (_flags(own_meta.has_event) & ~_flags(own_meta.is_anticipated))
            ids = set(own_meta.loc[selected, "series_id"])
            p, m, a = frame.loc[frame.series_id.isin(ids)], own_matches.loc[own_matches.series_id.isin(ids)], own_alerts.loc[own_alerts.series_id.isin(ids)]
            measured = dict(**_row_metrics(p), **_event_metrics(p, m, a))
            saved = saved_scenarios.loc[saved_scenarios.cohort.eq("test") & saved_scenarios.model.eq(model) & saved_scenarios.k.eq(k) & saved_scenarios.scenario.eq(scenario)]
            _require(len(saved) == 1, "missing/duplicate saved synthetic scenario")
            expected = saved.iloc[0]
            for field, value in measured.items():
                _assert_number(value, expected["f1" if field == "row_f1" else field], f"E07c/{model}/{k}/{scenario}/{field}")
            records.append(dict(experiment="E07c", data_status="synthetic", evaluation_status="evaluated_on_independent_synthetic_TEST",
                truth_data_status="synthetic", cohort="test", scenario=scenario, model=model, k=int(k), **measured,
                threshold=float(selection.iloc[0].threshold), source_path=SYNTHETIC + "predictions.csv.gz", source_report=REPORTS["E07c"]))
            if scenario == "base_all":
                for saved_table, names in ((saved_rows, ["cases", "positives", "negatives", "pr_auc", "row_f1"]),
                                           (saved_events, list(_event_metrics(p, m, a)))):
                    saved_part = saved_table.loc[saved_table.cohort.eq("test") & saved_table.model.eq(model) & saved_table.k.eq(k)]
                    _require(len(saved_part) == 1, "synthetic base metric entry missing/duplicated")
                    for name in names:
                        _assert_number(measured[name], saved_part.iloc[0]["f1" if name == "row_f1" else name], f"E07c base/{model}/{k}/{name}")
                row_comparison.append(dict(model=model, k=int(k), f1=measured["row_f1"], **{name: measured[name] for name in ("cases", "positives", "negatives", "pr_auc")}))
                event_comparison.append(dict(model=model, k=int(k), **{name: measured[name] for name in ("eligible_events", "warned_events", "event_recall", "alert_precision", "median_lead_time_months", "mean_lead_time_months", "false_alert_count", "false_alerts_per_12_monitored_months")}))
    _require(len(row_comparison) == 8, "S0--S3/horizons 1,3 synthetic results incomplete")
    _verify_report_table(reports["E07c"], pd.DataFrame(row_comparison), ["model", "k"], ["cases", "positives", "negatives", "pr_auc", "f1"], "E07c row metrics")
    _verify_report_table(reports["E07c"], pd.DataFrame(event_comparison), ["model", "k"], list(event_comparison[0].keys())[2:], "E07c event metrics")
    _require(intervals.bootstrap_unit.eq("whole_series_id_with_multiplicity").all(), "synthetic intervals no longer cluster by whole series")
    return dict(data_status="synthetic", cohorts=cohorts, metrics=records, bootstrap_intervals=intervals.to_dict("records"),
                cue_bearing_recall_reference=0.75, reference_is_strict_ceiling=False,
                interpretation="controlled fixed-generator mechanism demonstration; not real economic shocks or real news usefulness",
                source_paths=[SYNTHETIC + "predictions.csv.gz", SYNTHETIC + "metrics_event.csv", REPORTS["E07c"]]), records


def collect_evidence(root: Path) -> dict:
    """Collect verified aggregates, failing closed on missing/discrepant artifacts."""
    reader = _Reader(root)
    reports = {experiment: reader.text(path, "saved report compared against independently recomputed aggregates")
               for experiment, path in REPORTS.items()}
    for name in ("configs/news_events_v3.yaml", "configs/macro_forecast.yaml", "configs/early_warning_feasibility.yaml",
                 "configs/early_warning_full_panel.yaml", "configs/early_warning_synthetic.yaml"):
        reader.path(name, "frozen experiment configuration")
    news = _news(reader, reports)
    external = _external(reader, reports)
    real, real_records = _real(reader, reports)
    synthetic, synthetic_records = _synthetic(reader, reports)
    # Renderer-facing aliases retain the explicitly documented month/exposure units.
    for record in real_records + synthetic_records:
        record["median_lead_time"] = record["median_lead_time_months"]
        record["false_alerts_per_12_months"] = record["false_alerts_per_12_monitored_months"]
    figure_paths = [SYNTHETIC + "plots/first_successful_warning.png", SYNTHETIC + "plots/first_false_alert.png",
                    SYNTHETIC + "plots/first_missed_event.png"]
    for name in figure_paths:
        reader.path(name, "existing deterministic synthetic TEST example figure")
    return _clean(dict(passed=True, news=news, external=external, real_early_warning=real,
        synthetic_early_warning=synthetic, records=real_records + synthetic_records,
        facts=dict(news_documents=news["documents"], news_snapshots=news["snapshots"], news_canonical_groups=news["canonical_events"],
                   real_panel_municipalities=real["E07b"]["municipalities"], real_weak_events=real["E07b"]["weak_events"],
                   real_classifier_fits=0, synthetic_test_events=next(row["events"] for row in synthetic["cohorts"] if row["cohort"] == "test")),
        provenance=list(reader.provenance.values()), discrepancies=[], recommended_figures=figure_paths,
        definitions=dict(pr_auc="average precision, step integral with tied probabilities grouped",
                         real_truth=real["truth_definition"], synthetic_truth="true generated onset in O+1 through O+k",
                         false_alert_exposure="fully known at-risk monitored series-months divided by twelve",
                         undefined_metrics="null, never zero; failed real gate means not evaluated")))
