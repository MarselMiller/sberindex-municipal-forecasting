"""Synthetic full-panel audits, independent of real-data event counts."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from sberforecast.early_warning_labels import build_early_warning_labels
from sberforecast.early_warning_full_panel_labels import (
    E07A_WARNING_SPEC, E07A_WEAK_LABEL_SPEC, assert_e07a_lock,
    audit_all_calendar_splits, build_panel_labels, evaluate_gate,
    summarize_feasibility, temporal_split,
)


@pytest.fixture
def config():
    return {"data": {"first_month": "2023-01", "last_month": "2024-12", "release_lag_months": 0},
            "weak_label": deepcopy(E07A_WEAK_LABEL_SPEC),
            "early_warning": {**deepcopy(E07A_WARNING_SPEC), "split": {
                "train_information_cutoff": "2024-07", "test_first_origin": "2024-08", "test_last_origin": "2024-11"}},
            "runtime": {"label_batch_size": 2},
            "gate": {"train_positives_min": 30, "test_positives_min": 10,
                     "train_positive_dates_min": 3, "test_positive_dates_min": 2}}


def residuals(errors, uid="synthetic"):
    periods = pd.period_range("2024-01", periods=len(errors), freq="M").astype(str)
    return pd.DataFrame({"series_id": uid, "observation_period": periods,
                         "y_true": 100 + np.asarray(errors, dtype=float), "y_pred": 100.,
                         "available_period": periods, "model": "SeasonalNaiveYoY", "horizon": 1.0})


def samples(*origins, uid="synthetic", eligible=True):
    return pd.DataFrame({"municipality_id": uid,
                         "forecast_origin": [str(pd.Period(origin, freq="M").end_time.normalize().date()) for origin in origins],
                         "eligible_at_origin": eligible})


def pilot_fixture():
    """64 synthetic municipalities, not the project's private pilot rows."""
    series, queries = [], []
    calendar = pd.period_range("2023-12", "2024-11", freq="M").astype(str)
    for number in range(64):
        uid = f"synthetic_{number:02d}"
        errors = [0] * 12
        if number == 0:
            errors = [0] * 4 + [-12] * 4 + [-6] * 4
        elif number == 1:
            errors = [0] * 5 + [-12] * 3 + [-6] * 4
        elif number == 2:
            errors = [0] * 5 + [-12] * 3 + [12] * 3 + [-6]
        table = residuals(errors, uid)
        if number == 63:
            table.loc[:, ["y_true", "y_pred", "horizon", "model"]] = np.nan
        series.append(table)
        queries.append(samples(*calendar, uid=uid))
    return pd.concat(series, ignore_index=True), pd.concat(queries, ignore_index=True)


def test_batched_engine_matches_frozen_engine_and_preserves_sample_order(config):
    raw = pd.concat([residuals([0] * 4 + [12] * 8, "b"), residuals([0] * 12, "a")], ignore_index=True)
    query = pd.concat([samples("2024-08", uid="b", eligible=False), samples("2024-04", uid="a"), samples("2024-04", uid="b")], ignore_index=True)
    progress = []
    config["runtime"]["label_batch_size"] = 1
    batched = build_panel_labels(raw, query, config, progress.append)
    frozen = build_early_warning_labels(raw, query, config)
    for key in frozen:
        pd.testing.assert_frame_equal(batched[key], frozen[key])
    assert [entry["processed_municipalities"] for entry in progress] == [1, 2]
    assert batched["cases"].eligible_at_origin.tolist() == [False, False, True, True, True, True]
    assert batched["cases"].forecast_origin.tolist() == [date for date in query.forecast_origin for _ in range(2)]


def test_synthetic_64_pilot_shape_five_candidates_four_events_and_schema(config):
    raw, query = pilot_fixture()
    config["runtime"]["label_batch_size"] = 64
    result = build_panel_labels(raw, query, config)
    assert len(result["origin_states"]) == 768 and len(result["cases"]) == 1536
    assert len(result["candidates"]) == 1536
    assert result["candidates"].is_candidate.sum() == 5
    assert len(result["events"]) == 4
    assert result["events"].onset_period.tolist() == ["2024-05", "2024-06", "2024-06", "2024-09"]
    assert result["events"].onset_period.nunique() == 3
    assert {"onset_period", "confirmation_period", "candidate_direction", "status", "accepted_event_id"}.issubset(result["candidates"])
    assert {"label_known_at", "eligible_at_origin", "missing_reason", "known_active_at_origin", "retrospective_inside_regime_at_origin"}.issubset(result["cases"])
    summary = summarize_feasibility(result["cases"], result["events"], query, config)
    gate = evaluate_gate(summary, config)
    assert not gate.gate_passed.any()
    train_k1 = summary.loc[summary.k.eq(1) & summary.scope.eq("train")].iloc[0]
    assert train_k1.cases == 63 and train_k1.positives == 1 and train_k1.positive_event_onset_dates == 1
    assert summary.loc[summary.k.eq(3) & summary.scope.eq("test"), "cases"].iloc[0] == 0


def test_future_confirmation_never_enters_training_and_unknown_stays_missing(config):
    result = build_panel_labels(residuals([0] * 4 + [12] * 8), samples("2024-04", "2024-05", "2024-10"), config)
    split = temporal_split(result["cases"], config)
    assert split.loc[split.split.eq("train"), ["warning_origin_period", "k"]].values.tolist() == [["2024-04", 1]]
    late_known = split.loc[split.warning_origin_period.eq("2024-05") & split.k.eq(1)].iloc[0]
    assert late_known.label_known_at == "2024-08-31" and late_known.split == "excluded"
    assert late_known.split_exclusion_reason == "label_not_known_by_train_cutoff"
    right = split.loc[split.warning_origin_period.eq("2024-10")]
    assert right.label.isna().all() and right.right_censored.all()
    assert len(split) == len(result["cases"])


def test_causal_active_exclusion_is_distinct_from_retrospective_membership(config):
    result = build_panel_labels(residuals([0] * 4 + [12] * 8), samples("2024-05", "2024-06", "2024-07", "2024-08"), config)
    state = result["origin_states"].set_index("warning_origin_period")
    assert state.loc["2024-05", "retrospective_inside_regime_at_origin"]
    assert not state.loc["2024-05", "known_active_at_origin"]
    assert not state.loc["2024-06", "known_active_at_origin"]
    assert state.loc["2024-07", "known_active_at_origin"]
    split = temporal_split(result["cases"], config)
    row = split.loc[split.warning_origin_period.eq("2024-08") & split.k.eq(1)].iloc[0]
    assert row.fully_known and row.split == "excluded"
    assert row.split_exclusion_reason == "confirmed_active_regime_known_at_origin"


def test_missing_rows_and_upstream_eligibility_are_not_future_cohort_filters(config):
    raw = residuals([0] * 12).drop(index=5)
    query = pd.concat([samples("2024-04", eligible=False), samples("2024-06"), samples("2024-08", uid="absent", eligible=False)], ignore_index=True)
    result = build_panel_labels(raw, query, config)
    assert len(result["cases"]) == 6
    assert result["cases"].label.isna().all()
    assert result["cases"].missing_reason.ne("").all()
    summary = summarize_feasibility(result["cases"], result["events"], query, config)
    all_k1 = summary.loc[summary.k.eq(1) & summary.scope.eq("all")].iloc[0]
    eligible_k1 = summary.loc[summary.k.eq(1) & summary.scope.eq("eligible")].iloc[0]
    assert all_k1.cases == 3 and all_k1.eligible_cases == 1
    assert eligible_k1.cases == 1 and eligible_k1.censored_or_missing == 1


def fabricated_cases(rows):
    return pd.DataFrame([dict(municipality_id=uid, forecast_origin=origin, k=k,
        eligible_at_origin=True, at_risk=True, fully_known=True, label=1., label_known_at=known,
        positive_event_ids=event, right_censored=False) for uid, origin, k, known, event in rows])


def test_many_municipal_positives_do_not_replace_independent_onset_dates(config):
    rows, events = [], []
    for number in range(45):
        uid, event = f"municipality_{number}", f"event_{number}"
        origin, known = ("2024-04-30", "2024-07-31") if number < 35 else ("2024-08-31", "2024-11-30")
        rows.append((uid, origin, 1, known, event))
        events.append(dict(municipality_id=uid, event_id=event, onset_period="2024-05" if number < 35 else "2024-09"))
    cases, registry = fabricated_cases(rows), pd.DataFrame(events)
    summary = summarize_feasibility(cases, registry, config)
    gate = evaluate_gate(summary, config)
    assert gate.train_positives.tolist() == [35, 0]
    assert gate.test_positives.tolist() == [10, 0]
    assert gate.train_positive_onset_dates.tolist() == [1, 0]
    assert gate.test_positive_onset_dates.tolist() == [1, 0]
    assert not gate.classifier_allowed.any()


def test_event_id_repeated_across_origins_is_counted_once(config):
    cases = fabricated_cases([("a", "2024-04-30", 3, "2024-09-30", "first|second"),
                              ("a", "2024-05-31", 3, "2024-10-31", "second")])
    registry = pd.DataFrame([dict(municipality_id="a", event_id="first", onset_period="2024-05"),
                             dict(municipality_id="a", event_id="second", onset_period="2024-06")])
    summary = summarize_feasibility(cases, registry, config)
    row = summary.loc[summary.k.eq(3) & summary.scope.eq("all")].iloc[0]
    assert row.positives == 2 and row.unique_events == 2 and row.positive_event_onset_dates == 2
    assert row.positive_warning_origin_dates == 2 and row.municipalities_with_event == 1


def test_premature_fully_known_label_is_rejected_by_split(config):
    cases = fabricated_cases([("a", "2024-04-30", 3, "2024-07-31", "event")])
    with pytest.raises(ValueError, match="uniform O\\+k\\+2"):
        temporal_split(cases, config)


def test_gate_passes_only_when_every_prespecified_count_is_satisfied(config):
    summary = pd.DataFrame([dict(k=k, scope=scope, positives=positive, positive_event_onset_dates=dates,
                                forecast_origin_dates=dates, cases=positive + 20)
                            for k in (1, 3) for scope, positive, dates in (("train", 30, 3), ("test", 10, 2))])
    assert evaluate_gate(summary, config).gate_passed.all()
    summary.loc[summary.k.eq(3) & summary.scope.eq("test"), "positive_event_onset_dates"] = 1
    gate = evaluate_gate(summary, config).set_index("k")
    assert gate.loc[1, "gate_passed"] and not gate.loc[3, "gate_passed"]


def test_all_calendar_options_are_diagnostic_and_original_cutoff_is_present(config):
    raw = residuals([0] * 4 + [12] * 8)
    query = samples(*pd.period_range("2023-12", "2024-11", freq="M").astype(str))
    result = build_panel_labels(raw, query, config)
    options = audit_all_calendar_splits(result["cases"], result["events"], config)
    assert len(options) == 24
    assert options.is_original_cutoff.sum() == 2
    assert options.diagnostic_only.all() and not options.gate_passed.any()
    assert not options.loc[options.k.eq(3), "has_nonempty_train_and_test"].any()
    assert config["early_warning"]["split"]["train_information_cutoff"] == "2024-07"


def test_determinism_batch_size_and_future_values_leave_prefix_state_unchanged(config):
    raw = pd.concat([residuals([0] * 4 + [12] * 8, "b"), residuals([0] * 12, "a")], ignore_index=True)
    query = pd.concat([samples("2024-05", "2024-07", uid="b"), samples("2024-04", uid="a")], ignore_index=True)
    first = build_panel_labels(raw, query, config)
    config["runtime"]["label_batch_size"] = 1
    second = build_panel_labels(raw, query, config)
    for key in first:
        pd.testing.assert_frame_equal(first[key], second[key])
    changed = raw.copy()
    changed.loc[changed.observation_period.gt("2024-07"), "y_true"] = 10000
    altered = build_panel_labels(changed, query, config)
    columns = [column for column in first["origin_states"] if column != "retrospective_inside_regime_at_origin"]
    pd.testing.assert_frame_equal(first["origin_states"][columns], altered["origin_states"][columns])


@pytest.mark.parametrize("mutation", ["threshold", "duration", "lag", "gate", "missing_semantic_field"])
def test_lock_rejects_protocol_or_gate_tuning(config, mutation):
    if mutation == "threshold":
        config["weak_label"]["strength_threshold"] = 2.9
    elif mutation == "duration":
        config["weak_label"]["confirmation_months"] = 2
    elif mutation == "lag":
        config["data"]["release_lag_months"] = 1
    elif mutation == "gate":
        config["gate"]["train_positive_dates_min"] = 2
    else:
        del config["weak_label"]["merge"]
    with pytest.raises(ValueError):
        assert_e07a_lock(config)


def test_eligibility_must_be_explicit_and_duplicate_queries_rejected(config):
    query = samples("2024-04")
    with pytest.raises(ValueError, match="Missing sample"):
        build_panel_labels(residuals([0] * 12), query.drop(columns="eligible_at_origin"), config)
    with pytest.raises(ValueError, match="Duplicate"):
        build_panel_labels(residuals([0] * 12), pd.concat([query, query]), config)


def test_empty_test_does_not_trigger_split_reselection_and_test_must_be_later(config):
    result = build_panel_labels(residuals([0] * 12), samples("2024-04", "2024-05"), config)
    split = temporal_split(result["cases"], config)
    assert not split.split.eq("test").any()
    assert split.train_information_cutoff.eq("2024-07-31").all()
    bad = deepcopy(config)
    bad["early_warning"]["split"]["test_first_origin"] = "2024-07"
    with pytest.raises(ValueError, match="strictly after"):
        temporal_split(result["cases"], bad)


def test_conflicting_split_aliases_and_unregistered_event_ids_are_rejected(config):
    result = build_panel_labels(residuals([0] * 4 + [12] * 8), samples("2024-04"), config)
    bad = deepcopy(config)
    bad["split"] = {**bad["early_warning"]["split"], "train_information_cutoff": "2024-08"}
    with pytest.raises(ValueError, match="disagree"):
        temporal_split(result["cases"], bad)
    result["cases"].loc[result["cases"].label.eq(1), "positive_event_ids"] = "unknown_event"
    with pytest.raises(ValueError, match="unknown event IDs"):
        summarize_feasibility(result["cases"], result["events"], config)
