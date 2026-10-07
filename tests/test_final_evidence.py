"""Synthetic checks of read-only F1 evidence arithmetic and discrepancy guards."""
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from sberforecast.final_evidence import (
    NEWS, _Reader, _assert_number, _clean, _event_metrics, _flags, _news,
    _real_counts, _row_metrics, _verify_report_table, collect_evidence,
)


def test_average_precision_groups_ties_and_f1_uses_saved_threshold():
    frame = pd.DataFrame({"label": [1, 0, 1, 0], "probability": [.8, .8, .4, .1],
                          "alert": [True, True, False, False]})
    result = _row_metrics(frame)
    # At tied .8: recall increment 1/2, precision 1/2. At .4: 1/2 * 2/3.
    assert result["pr_auc"] == pytest.approx(.5 * .5 + .5 * 2 / 3)
    assert result["row_f1"] == .5
    assert (result["cases"], result["positives"], result["negatives"]) == (4, 2, 2)


def test_no_positive_ranking_is_undefined_and_real_no_fit_is_not_zero():
    result = _row_metrics(pd.DataFrame({"label": [0, 0], "probability": [.2, .3],
                                       "alert": [False, False]}))
    assert result["pr_auc"] is None
    assert result["row_f1"] == 0
    assert _clean({"not_evaluated": float("nan")})["not_evaluated"] is None


@pytest.mark.parametrize("labels,probabilities", [([.5], [.2]), ([1], [1.1]), ([0], [float("nan")])])
def test_invalid_saved_rank_inputs_fail(labels, probabilities):
    with pytest.raises(ValueError):
        _row_metrics(pd.DataFrame({"label": labels, "probability": probabilities, "alert": [False]}))


def event_fixture():
    predictions = pd.DataFrame({"k": [3] * 24})
    matches = pd.DataFrame({"event_id": ["e1", "e2"], "warning_observed": [True, False],
                            "lead_time_months": [2., float("nan")]})
    alerts = pd.DataFrame({"event_id": ["e1", "e1", None], "successful_warning": [True, False, False],
                           "false_alert": [False, False, True], "repeated_warning": [False, True, False],
                           "lead_time_months": [2., 1., float("nan")]})
    return predictions, matches, alerts


def test_event_matching_repeats_do_not_add_success_or_false_alert():
    result = _event_metrics(*event_fixture())
    assert result["eligible_events"] == 2 and result["warned_events"] == 1
    assert result["event_recall"] == .5 and result["alert_precision"] == .5
    assert result["median_lead_time_months"] == 2
    assert result["false_alerts_per_12_monitored_months"] == .5
    assert result["false_alert_count"] == 1 and result["repeated_alert_count"] == 1


@pytest.mark.parametrize("mutation", ["late_alert", "late_match", "repeated_success", "repeated_false", "wrong_event", "duplicate_match"])
def test_event_contradictions_fail(mutation):
    predictions, matches, alerts = event_fixture()
    if mutation == "late_alert":
        alerts.loc[0, "lead_time_months"] = 0
    elif mutation == "late_match":
        matches.loc[0, "lead_time_months"] = 0
    elif mutation == "repeated_success":
        alerts.loc[0, "repeated_warning"] = True
    elif mutation == "repeated_false":
        alerts.loc[2, "repeated_warning"] = True
    elif mutation == "wrong_event":
        alerts.loc[0, "event_id"] = "e3"
    else:
        matches.loc[1, "event_id"] = "e1"
    with pytest.raises(ValueError):
        _event_metrics(predictions, matches, alerts)


def test_control_scenario_has_undefined_recall_and_exposure_scaled_false_alerts():
    predictions, matches, alerts = event_fixture()
    matches = matches.iloc[:0]
    alerts = alerts.loc[alerts.false_alert]
    result = _event_metrics(predictions, matches, alerts)
    assert result["eligible_events"] == 0 and result["event_recall"] is None
    assert result["alert_precision"] == 0
    assert result["median_lead_time_months"] is None
    assert result["false_alerts_per_12_monitored_months"] == .5


def real_fixture():
    cases = pd.DataFrame({
        "municipality_id": [str(i) for i in range(8)], "k": [1] * 8,
        "forecast_origin": ["2024-05-31"] * 4 + ["2024-08-31"] * 4,
        "label": [1, 1, 0, 1, 1, float("nan"), 1, 0],
        "fully_known": [True] * 5 + [False, True, True],
        "at_risk": [True] * 7 + [False], "eligible_at_origin": [True] * 6 + [False, True],
        "right_censored": [False] * 5 + [True, False, False],
        "label_known_at": ["2024-07-31"] * 3 + ["2024-08-01"] + ["2024-11-30"] * 4,
        "positive_event_ids": ['["e1"]', '["e2"]', '[]', '["e3"]', '["e3"]', '[]', '["e3"]', '[]'],
    })
    events = pd.DataFrame({"event_id": ["e1", "e2", "e3"],
                           "onset_period": ["2024-06", "2024-06", "2024-09"]})
    return cases, events


def test_real_split_uses_label_availability_excludes_active_unknown_ineligible():
    counts = _real_counts(*real_fixture()).set_index(["k", "scope"])
    assert counts.loc[(1, "all"), "cases"] == 8
    assert counts.loc[(1, "all"), "censored_or_missing"] == 1
    assert counts.loc[(1, "all"), "right_censored"] == 1
    assert counts.loc[(1, "at_risk_evaluable"), "cases"] == 5
    assert counts.loc[(1, "train"), "cases"] == 3
    assert counts.loc[(1, "train"), "positives"] == 2
    assert counts.loc[(1, "train"), "positive_event_onset_dates"] == 1
    assert counts.loc[(1, "test"), "cases"] == 1
    assert counts.loc[(1, "test"), "positives"] == 1
    assert counts.loc[(1, "test"), "positive_event_onset_dates"] == 1
    assert counts.loc[(3, "train"), "cases"] == 0


@pytest.mark.parametrize("mutation", ["unknown_label", "unknown_event", "duplicate_case"])
def test_real_saved_case_contradictions_fail(mutation):
    cases, events = real_fixture()
    if mutation == "unknown_label":
        cases.loc[5, "label"] = 0
    elif mutation == "unknown_event":
        cases.loc[0, "positive_event_ids"] = '["invented"]'
    else:
        cases.loc[1, ["municipality_id", "forecast_origin", "k"]] = cases.loc[0, ["municipality_id", "forecast_origin", "k"]]
    with pytest.raises(ValueError):
        _real_counts(cases, events)


REPORT = """| model | k | cases | pr_auc | median_lead_time_months |
|---|---|---|---|---|
| S0 | 1 | 2722 | 0.0661278 | — |
"""


def report_frame():
    return pd.DataFrame({"model": ["S0"], "k": [1], "cases": [2722],
                         "pr_auc": [180 / 2722], "median_lead_time_months": [None]})


def test_report_reconciles_six_digit_rounding_and_preserves_undefined_metric():
    _verify_report_table(REPORT, report_frame(), ["model", "k"],
                         ["cases", "pr_auc", "median_lead_time_months"], "synthetic report")


@pytest.mark.parametrize("before,after", [("2722", "2723"), ("2722", "2722.01"),
                                          ("0.0661278", "0.0761278"), ("—", "0"), ("S0", "S1")])
def test_report_numeric_or_missing_row_discrepancy_fails(before, after):
    with pytest.raises(ValueError):
        _verify_report_table(REPORT.replace(before, after), report_frame(), ["model", "k"],
                             ["cases", "pr_auc", "median_lead_time_months"], "synthetic report")


def test_duplicate_report_rows_fail_even_when_last_copy_is_correct():
    text = REPORT + "| S0 | 1 | 2722 | 0.0661278 | — |\n"
    with pytest.raises(ValueError, match="duplicate report"):
        _verify_report_table(text, report_frame(), ["model", "k"], ["cases"], "synthetic report")


def test_integer_count_has_no_report_rounding_allowance():
    with pytest.raises(ValueError):
        _assert_number(26280, 26280.01, "count", rounded=True)


def test_reader_requires_files_root_confinement_and_exact_sha(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    payload = b'{"value": 7}\n'
    (root / "audit.json").write_bytes(payload)
    (tmp_path / "outside.json").write_bytes(payload)
    reader = _Reader(root)
    assert reader.json("audit.json", "synthetic source") == {"value": 7}
    reader.path("audit.json", "duplicate reference")
    assert list(reader.provenance.values()) == [{"path": "audit.json", "sha256": hashlib.sha256(payload).hexdigest(), "role": "synthetic source"}]
    with pytest.raises(ValueError, match="outside project root"):
        reader.path("../outside.json", "bad source")
    with pytest.raises(ValueError, match="missing"):
        reader.path("absent.csv", "required")
    with pytest.raises(ValueError, match="missing"):
        collect_evidence(root)


def test_clean_returns_strict_json_primitives_and_nonfinite_null():
    result = _clean({"nan": float("nan"), "infinity": float("inf"), "list": (1, True, None),
                     "time": pd.Timestamp("2024-01-01", tz="UTC")})
    assert result["nan"] is None and result["infinity"] is None
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    with pytest.raises(TypeError):
        _clean(Path("unsupported"))


@pytest.mark.parametrize("value", [None, "False", 2])
def test_saved_flags_do_not_silently_coerce_strings_or_unknown(value):
    with pytest.raises(ValueError):
        _flags(pd.Series([value]))


def save_toy_news(root):
    directory = root / NEWS
    directory.mkdir(parents=True)
    docs = pd.DataFrame({
        "document_id": ["primary", "retrieved"], "snapshot_id": ["s1", "s2"],
        "source_url": ["https://official.invalid/primary", "https://news.invalid/retrieved"],
        "canonical_event_id": ["core", "other"], "published_at": ["2024-01-01", "2024-01-02"],
        "available_at": ["2024-01-01", "2026-01-01"],
        "availability_status": ["confirmed_historical", "retrieval_only"],
        "geography_level": ["national", "unknown"], "event_type": ["policy_decision", "news"],
    })
    docs.to_csv(directory / "documents.csv.gz", index=False)
    docs.to_csv(directory / "events.csv.gz", index=False)
    values = ["news_count_30d", "regional_news_count_30d", "municipal_news_count_30d"]
    features = pd.DataFrame({"municipality_id": ["001", "002", "001", "002"],
                              "forecast_origin": ["2024-01-31"] * 2 + ["2024-02-29"] * 2,
                              "news_count_30d": [0, 0, 1, 1], "regional_news_count_30d": [0] * 4,
                              "municipal_news_count_30d": [0] * 4, "source_archive_complete": [False] * 4})
    for value in values:
        features[value + "_missing"] = False
    features.to_csv(directory / "news_features.csv.gz", index=False)
    # The actual dictionary contains both values and flags, interleaved.
    pd.DataFrame({"feature": [name for value in values for name in (value, value + "_missing")]}).to_csv(directory / "feature_dictionary.csv", index=False)
    counts = dict(snapshots=2, documents=2, canonical_events=2, duplicates_removed=0, admitted_core_records=1,
                  historical_documents_at_last_origin=1, historical_snapshots_at_last_origin=1, feature_rows=4,
                  unique_temporal_vectors=2, nonzero_30d_rows=2, nonzero_30d_origins=1, regional_nonzero_rows=0)
    (directory / "coverage_audit.json").write_text(json.dumps(counts), encoding="utf-8")
    mapping = {"URL-документы": 2, "Снимки/извлечённые версии": 2, "Canonical группы событий": 2,
               "Исторически доступные URL к последней O": 1, "Уникальные временные news-векторы": 2,
               "Строки с news_30d>0": 2}
    report = "| Показатель | v3 |\n|---|---|\n" + "".join(f"| {key} | {value} |\n" for key, value in mapping.items())
    return directory, {"E07a": report}


def test_news_interleaved_dictionary_and_historical_scope(tmp_path):
    _, reports = save_toy_news(tmp_path)
    news = _news(_Reader(tmp_path), reports)
    assert news["value_features"] == 3 and news["missing_flags"] == 3
    assert news["total_numeric_features"] == 6
    assert news["documents"] == 2 and news["historical_documents_at_last_origin"] == 1
    assert news["historical_by_geography"] == {"national": 1, "regional": 0, "municipality": 0}
    assert news["maximum_spatial_vectors_per_origin"] == 1
    assert news["source_archive_complete"] is False
    assert news["data_status"] == "real"


def test_news_stale_summary_and_report_numbers_fail(tmp_path):
    directory, reports = save_toy_news(tmp_path)
    summary_path = directory / "coverage_audit.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["documents"] = 3
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="news summary documents"):
        _news(_Reader(tmp_path), reports)
    summary["documents"] = 2
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    reports["E07a"] = reports["E07a"].replace("| URL-документы | 2 |", "| URL-документы | 3 |")
    with pytest.raises(ValueError, match="E07a news report"):
        _news(_Reader(tmp_path), reports)
