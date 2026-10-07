"""Offline synthetic E06b tests: point-in-time versions, scopes and joins."""
import numpy as np
import pandas as pd
import pytest

from sberforecast.news_features import (
    EVENT_COLUMNS,
    FEATURE_COLUMNS,
    MISSING_COLUMNS,
    build_news_features,
    feature_dictionary,
    safe_asof_merge,
)


def event(document="d1", available="2024-03-20T00:00:00Z", **extra):
    return dict({
        "document_id": document, "canonical_event_id": "event-" + document,
        "duplicate_count": 1, "published_at": available, "available_at": available,
        "event_date": "2024-01-01", "geography_level": "national",
        "region_id": "RU", "municipality_id": "", "topic": "macro_policy",
        "event_type": "decision", "direction": "neutral",
        "availability_status": "confirmed_historical",
    }, **extra)


def table(*rows):
    return pd.DataFrame(list(rows)) if rows else pd.DataFrame(columns=EVENT_COLUMNS)


def samples(*origins):
    return pd.DataFrame({
        "municipality_id": ["21"] * len(origins), "region_id": ["01"] * len(origins),
        "forecast_origin": list(origins),
    })


def test_future_publication_with_past_event_date_does_not_change_any_feature():
    query = samples("2024-03-31")
    old = table(event())
    future = event("future", "2024-04-01T00:00:00Z", event_date="2023-01-01", direction="negative")
    pd.testing.assert_frame_equal(
        build_news_features(old, query), build_news_features(table(event(), future), query),
    )


def test_future_update_does_not_change_past_snapshot_labels_or_geography():
    original = event(direction="negative")
    updated = event(available="2024-04-02T00:00:00Z", published_at=original["published_at"],
                    availability_status="confirmed_updated_version", direction="positive",
                    geography_level="region", region_id="2", topic="regulation")
    query = samples("2024-03-31")
    pd.testing.assert_frame_equal(
        build_news_features(table(original), query), build_news_features(table(original, updated), query),
    )


def test_future_duplicate_global_audit_metadata_cannot_change_past_features():
    original = event(direction="negative", duplicate_count=1)
    future = event("future-copy", "2024-04-01T00:00:00Z", canonical_event_id=original["canonical_event_id"],
                   published_at=original["published_at"], direction="positive", duplicate_count=999)
    before = build_news_features(table(original), samples("2024-03-31"))
    original["duplicate_count"] = 999
    after = build_news_features(table(original, future), samples("2024-03-31"))
    pd.testing.assert_frame_equal(before, after)


def test_dedup_counts_events_and_observed_document_share_separately():
    original = event(direction="negative")
    reprint = event("copy", "2024-03-21T00:00:00Z", canonical_event_id=original["canonical_event_id"],
                    published_at=original["published_at"], direction="positive", topic="other")
    row = build_news_features(table(original, reprint), samples("2024-03-31")).iloc[0]
    assert row.news_count_30d == row.unique_event_count_30d == 1
    assert row.negative_count_30d == 1
    assert row.positive_count_30d == 0
    assert row.duplicate_share_30d == 0.5


def test_document_updates_do_not_count_as_new_documents_or_rewrite_event():
    original = event(direction="negative")
    update = event(available="2024-03-22T00:00:00Z", published_at=original["published_at"], direction="positive")
    row = build_news_features(table(original, update), samples("2024-03-31")).iloc[0]
    assert row.news_count_30d == row.negative_count_30d == 1
    assert row.positive_count_30d == row.duplicate_share_30d == 0


def test_reprint_of_event_outside_window_does_not_make_a_new_event():
    original = event("old", "2024-01-02T00:00:00Z")
    copy = event("copy", "2024-03-20T00:00:00Z", canonical_event_id=original["canonical_event_id"])
    row = build_news_features(table(original, copy), samples("2024-03-31")).iloc[0]
    assert row.news_count_30d == row.unique_event_count_30d == 0
    assert row.duplicate_share_30d == 1
    assert row.news_count_90d == 1


def test_global_dedup_before_scope_prevents_reprint_from_moving_event():
    original = event(geography_level="region", region_id="1")
    copy = event("copy", "2024-03-21T00:00:00Z", canonical_event_id=original["canonical_event_id"],
                 geography_level="region", region_id="2")
    query = pd.DataFrame({"municipality_id": ["21", "22"], "region_id": ["1", "2"],
                          "forecast_origin": ["2024-03-31"] * 2})
    result = build_news_features(table(original, copy), query)
    assert result.news_count_30d.tolist() == [1, 0]
    assert result.duplicate_share_30d.iloc[0] == 0.5
    assert np.isnan(result.duplicate_share_30d.iloc[1])


def test_scope_separation_and_total_count_without_double_counting():
    query = pd.DataFrame({
        "municipality_id": ["21", "25", "37"], "region_id": ["01", "1", "2"],
        "forecast_origin": ["2024-03-31"] * 3,
    })
    events = table(
        event("national"), event("regional", geography_level="region", region_id="1"),
        event("municipal", geography_level="municipality", municipality_id="21", region_id="1"),
        event("unknown", geography_level="unknown", municipality_id="21", region_id="1"),
    )
    result = build_news_features(events, query)
    assert result.national_news_count_30d.tolist() == [1, 1, 1]
    assert result.regional_news_count_30d.tolist() == [1, 1, 0]
    assert result.municipal_news_count_30d.tolist() == [1, 0, 0]
    assert result.news_count_30d.tolist() == [3, 2, 1]
    pd.testing.assert_series_equal(
        result.news_count_30d, result.national_news_count_30d + result.regional_news_count_30d + result.municipal_news_count_30d,
        check_names=False,
    )


def test_trailing_windows_open_left_closed_right_and_moscow_midnight_origin():
    # E01 2024-03-31 means 2024-03-30 21:00 UTC, not end-of-day.
    cutoff = pd.Timestamp("2024-03-30T21:00:00Z")
    dates = [cutoff - pd.Timedelta(days=n) for n in (90, 30, 7, 0)]
    rows = [event(str(i), date.isoformat()) for i, date in enumerate(dates)]
    rows += [event("after", (cutoff + pd.Timedelta(nanoseconds=1)).isoformat())]
    row = build_news_features(table(*rows), samples("2024-03-31")).iloc[0]
    assert row.news_count_7d == 1
    assert row.news_count_30d == 2
    assert row.news_count_90d == 3
    assert row.news_count_change_30d_vs_prev30d == 1  # left boundary belongs to previous window


def test_publication_time_is_not_used_in_place_of_available_version_time():
    row = event(available="2024-03-31T12:00:00Z", published_at="2024-02-20T12:00:00Z")
    result = build_news_features(table(row), samples("2024-03-31", "2024-04-01"))
    assert result.news_count_30d.tolist() == [0, 1]


def test_unconfirmed_historical_snapshot_is_available_only_on_retrieval():
    row = event(available="2026-10-07T09:00:00Z", published_at="2024-03-20T00:00:00Z",
                availability_status="retrieval_only")
    result = build_news_features(table(row), samples("2024-03-31", "2026-10-08"))
    assert result.news_count_30d.tolist() == [0, 1]


def test_missing_version_times_are_never_imputed_from_publication_or_event_date():
    row = event(available=None, published_at="2024-03-01T00:00:00Z", event_date="2024-01-01")
    result = build_news_features(table(row), samples("2024-03-31"))
    assert result.news_count_30d.iloc[0] == 0


def test_available_before_publication_is_rejected():
    with pytest.raises(ValueError, match="before publication"):
        build_news_features(table(event(published_at="2024-04-01T00:00:00Z")), samples("2024-03-31"))


def test_topic_direction_emergency_recency_and_intensity_definitions():
    origin = pd.Timestamp("2024-03-31T00:00:00Z")
    rows = [
        event("a", (origin - pd.Timedelta(days=10)).isoformat(), direction="negative", topic="regulation"),
        event("b", (origin - pd.Timedelta(days=5)).isoformat(), direction="positive", topic="income_wages", event_type="emergency"),
        event("c", (origin - pd.Timedelta(days=2)).isoformat(), direction="negative", topic="disaster_emergency", event_type="other"),
        event("d", (origin - pd.Timedelta(days=40)).isoformat(), direction="negative", topic="regulation"),
    ]
    row = build_news_features(table(*rows), samples(origin)).iloc[0]
    assert row.negative_count_30d == 2
    assert row.positive_count_30d == 1
    assert row.emergency_count_30d == 2  # OR, never twice for one canonical event
    assert row.regulation_count_30d == row.income_wages_count_30d == 1
    assert row.days_since_last_negative_event == row.days_since_last_emergency == 2
    assert row.days_since_last_regulation_event == 10
    assert row.negative_share_30d == 2 / 3
    assert row.news_intensity_ratio_30d_90d == 9 / 4
    assert row.news_count_change_30d_vs_prev30d == 2
    assert row.negative_count_change_30d_vs_prev30d == 1
    assert row.emergency_count_change_30d_vs_prev30d == 2
    assert row.topic_entropy_30d == pytest.approx(np.log(3))
    assert row.topic_entropy_change == pytest.approx(np.log(3))


def test_fractional_elapsed_days_are_not_calendar_date_rounding():
    row = build_news_features(
        table(event(available="2024-03-30T12:00:00Z", direction="negative")),
        samples("2024-03-31T00:00:00Z"),
    ).iloc[0]
    assert row.days_since_last_negative_event == 0.5


def test_insufficient_history_is_nan_with_flags_without_future_backfill():
    result = build_news_features(table(event("future", "2023-04-01T00:00:00Z")),
                                 samples("2023-01-15", "2023-02-15", "2023-03-15"))
    assert result.news_count_7d.tolist() == [0, 0, 0]
    assert np.isnan(result.news_count_30d.iloc[0])
    assert result.news_count_30d.iloc[1] == 0
    assert result.news_count_90d.isna().all()
    assert result.news_count_change_30d_vs_prev30d.isna().iloc[:2].all()
    assert result.news_count_change_30d_vs_prev30d.iloc[2] == 0
    for feature in FEATURE_COLUMNS:
        pd.testing.assert_series_equal(result[feature].isna(), result[feature + "_missing"], check_names=False)


def test_declared_history_start_is_independent_of_first_observed_news():
    result = build_news_features(table(event()), samples("2024-03-31"), history_start="2024-03-15")
    assert np.isnan(result.news_count_30d.iloc[0])
    assert result.news_count_7d.iloc[0] == 0
    assert result.news_count_change_30d_vs_prev30d_missing.iloc[0]


def test_empty_corpus_keeps_all_64_by_12_forecast_keys():
    query = pd.DataFrame([
        {"municipality_id": str(municipality), "region_id": str(1 + municipality % 4), "forecast_origin": str(period.end_time.normalize().date())}
        for municipality in range(64) for period in pd.period_range("2023-12", "2024-11", freq="M")
    ], index=pd.Index(range(768), name="pilot_row"))
    result = build_news_features(table(), query)
    assert len(result) == 64 * 12
    pd.testing.assert_frame_equal(result[query.columns], query)
    assert result.news_count_30d.eq(0).all()
    assert result.duplicate_share_30d.isna().all()
    assert result.negative_share_30d.isna().all()
    assert result.news_intensity_ratio_30d_90d.isna().all()
    assert result.topic_entropy_30d.isna().all()
    assert result.days_since_last_negative_event.isna().all()
    assert not result.news_count_30d_missing.any()
    assert result.topic_entropy_30d_missing.all()


def test_empty_queries_and_duplicate_query_index_preserve_rows():
    empty = samples()
    assert build_news_features(table(), empty).empty
    query = samples("2024-03-31", "2024-03-31")
    query.index = [7, 7]
    query["horizon"] = [1, 3]
    result = build_news_features(table(event()), query)
    pd.testing.assert_frame_equal(result[query.columns], query)
    assert result.news_count_30d.tolist() == [1, 1]


def test_event_order_does_not_choose_arbitrary_versions():
    original = event(direction="negative")
    duplicate = event("copy", "2024-03-21T00:00:00Z", canonical_event_id=original["canonical_event_id"], direction="positive")
    ordered = table(original, duplicate)
    pd.testing.assert_frame_equal(
        build_news_features(ordered, samples("2024-03-31")),
        build_news_features(ordered.iloc[::-1], samples("2024-03-31")),
    )


def test_forecast_origin_month_and_period_equal_e01_month_end_midnight():
    query = samples("2024-03", pd.Period("2024-03", freq="M"), "2024-03-31")
    result = build_news_features(table(event()), query)
    assert result.news_count_30d.tolist() == [1, 1, 1]


def test_sample_geography_conflicts_are_rejected():
    query = samples("2024-03-31", "2024-04-30")
    query["region_id"] = ["1", "2"]
    with pytest.raises(ValueError, match="Ambiguous"):
        build_news_features(table(), query)


def test_incomplete_geographic_claim_does_not_silently_assign_a_municipality():
    with pytest.raises(ValueError, match="explicit municipality"):
        build_news_features(table(event(geography_level="municipality", region_id="1")), samples("2024-03-31"))


def test_feature_dictionary_covers_values_flags_and_required_provenance():
    dictionary = feature_dictionary()
    assert len(FEATURE_COLUMNS) == 26
    assert len(dictionary) == 52
    assert set(dictionary.feature) == set(FEATURE_COLUMNS + MISSING_COLUMNS)
    assert not dictionary.feature.duplicated().any()
    assert set(("definition", "window", "geographic_scope", "source_fields", "availability_rule", "missing_rule", "unit")) <= set(dictionary)
    assert not dictionary.isna().any().any()
    assert "available_at" in dictionary.availability_rule.iloc[0]


def states(*rows):
    return pd.DataFrame(rows, columns=("municipality_id", "available_at", "score"))


def test_future_detector_state_not_joined_and_forecast_rows_retained():
    query = pd.DataFrame({"municipality_id": ["21", "25", "21"], "forecast_origin": ["2024-03-31", "2024-03-31", "2024-04-30"]})
    data = states(("21", "2024-03-01T00:00:00Z", 2), ("21", "2024-04-01T00:00:00Z", 999), ("25", "2024-04-01T00:00:00Z", 888))
    result = safe_asof_merge(query, data)
    pd.testing.assert_frame_equal(result[query.columns], query)
    assert result.detector_score.iloc[0] == 2
    assert np.isnan(result.detector_score.iloc[1])
    assert result.detector_score.iloc[2] == 999
    assert result.detector_missing.tolist() == [False, True, False]
    known = result.loc[~result.detector_missing]
    cutoff = pd.to_datetime(known.forecast_origin).dt.tz_localize("Europe/Moscow").dt.tz_convert("UTC")
    assert (known.detector_available_at <= cutoff).all()


def test_detector_cutoff_same_moscow_midnight_and_closed_right_boundary():
    query = samples("2024-03-31")
    data = states(("21", "2024-03-30T21:00:00Z", 1), ("21", "2024-03-30T21:00:00.000000001Z", 999))
    assert safe_asof_merge(query, data).detector_score.iloc[0] == 1


def test_future_detector_append_does_not_change_past_join():
    query = samples("2024-03-31")
    old = states(("21", "2024-03-01T00:00:00Z", 1))
    future = states(("21", "2024-04-01T00:00:00Z", 999))
    pd.testing.assert_frame_equal(
        safe_asof_merge(query, old), safe_asof_merge(query, pd.concat([old, future], ignore_index=True)),
    )


def test_detector_duplicate_state_times_require_upstream_pivot():
    with pytest.raises(ValueError, match="pivot"):
        safe_asof_merge(samples("2024-03-31"), states(("21", "2024-03-01T00:00:00Z", 1), ("21", "2024-03-01T00:00:00Z", 2)))


@pytest.mark.parametrize("metadata", [
    {"method": "PELT"}, {"method": "BinSeg"}, {"mode": "offline"}, {"is_hindsight_diagnostic": True},
    {"future_access": "whole_available_series"},
])
def test_offline_backdated_states_never_join_as_online(metadata):
    data = states(("21", "2024-03-01T00:00:00Z", 1)).assign(**metadata)
    with pytest.raises(ValueError, match="Offline|Hindsight"):
        safe_asof_merge(samples("2024-03-31"), data)


@pytest.mark.parametrize("field", ["analyzed_available_date", "analysis_available_at", "max_input_available_at", "state_computed_at", "signal_date"])
def test_backdated_detector_dependency_is_rejected(field):
    data = states(("21", "2024-03-01T00:00:00Z", 1))
    data[field] = "2024-04-01T00:00:00Z"
    with pytest.raises(ValueError, match="precedes its dependency"):
        safe_asof_merge(samples("2024-03-31"), data)


def test_empty_detector_table_preserves_all_keys_and_explicit_missing():
    query = samples("2024-03-31", "2024-04-30")
    query.index = [8, 8]
    result = safe_asof_merge(query, states())
    pd.testing.assert_frame_equal(result[query.columns], query)
    assert result.detector_score.isna().all()
    assert result.detector_missing.all()


def test_detector_requires_explicit_availability_not_observation_date():
    data = states(("21", None, 1))
    data["observation_period"] = "2024-03"
    with pytest.raises(ValueError, match="explicit available_at"):
        safe_asof_merge(samples("2024-03-31"), data)


def test_feature_and_detector_column_collisions_are_rejected():
    query = samples("2024-03-31")
    query["news_count_30d"] = 9
    with pytest.raises(ValueError, match="already contain"):
        build_news_features(table(), query)
    query = samples("2024-03-31")
    query["detector_score"] = 9
    with pytest.raises(ValueError, match="collide"):
        safe_asof_merge(query, states(("21", "2024-03-01T00:00:00Z", 1)))


def test_feature_builder_leaves_event_and_sample_input_tables_unchanged():
    events = table(event())
    query = samples("2024-03-31")
    original_events, original_query = events.copy(deep=True), query.copy(deep=True)
    build_news_features(events, query)
    pd.testing.assert_frame_equal(events, original_events)
    pd.testing.assert_frame_equal(query, original_query)
