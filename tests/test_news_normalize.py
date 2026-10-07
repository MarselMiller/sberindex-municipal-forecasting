import pandas as pd
import pytest

from sberforecast.news_normalize import normalize_documents, canonical_events


def dictionary():
    from sberforecast.news_geo import build_geography_dictionary
    return build_geography_dictionary(pd.DataFrame([dict(territory_id="1", region_code="1",
        region_name="Республика Адыгея", municipal_district_name="город Майкоп",
        municipal_district_name_short="Майкоп")]))


def raw(**changes):
    result = dict(source="cbr", source_url="https://example.invalid/1", title="Банк России повысил ключевую ставку",
        snippet="", published_at="2023-12-15T13:30:00+03:00", retrieved_at="2026-10-07T12:00:00+03:00")
    result.update(changes)
    return result


def test_current_snapshot_not_backdated_by_publication_or_input_available_at():
    frame = normalize_documents([raw(available_at="2023-12-15")], dictionary())
    assert frame.iloc[0].availability_status == "retrieval_only"
    assert frame.iloc[0].available_at == pd.Timestamp("2026-10-07T09:00:00Z")


def test_confirmed_version_requires_evidence_and_update_availability():
    with pytest.raises(ValueError, match="evidence"):
        normalize_documents([raw(historical_version_confirmed=True, available_at="2023-12-16")], dictionary())
    with pytest.raises(ValueError, match="update"):
        normalize_documents([raw(historical_version_confirmed=True, available_at="2023-12-16",
            updated_at="2024-01-10", availability_evidence="Synthetic archived version")], dictionary())


def test_date_only_is_moscow_and_event_date_does_not_set_availability():
    frame = normalize_documents([raw(event_date="2023-01-01")], dictionary())
    assert frame.iloc[0].event_date == pd.Timestamp("2022-12-31T21:00:00Z")
    assert frame.iloc[0].available_at.year == 2026


def test_cross_source_exact_title_dedup_preserves_audit():
    frame = normalize_documents([raw(), raw(source="lenta", source_url="https://example.invalid/2",
        title="БАНК РОССИИ: повысил ключевую ставку!")], dictionary())
    assert len(frame) == 2
    assert len(canonical_events(frame)) == 1
    assert frame.duplicate_count.tolist() == [2, 2]


def test_later_document_version_keeps_event_id_and_snapshot():
    frame = normalize_documents([raw(retrieved_at="2023-12-16"), raw(title="Обновлённый заголовок")], dictionary())
    assert frame.document_id.nunique() == 1
    assert frame.snapshot_id.nunique() == 2
    assert frame.canonical_event_id.nunique() == 1
    assert frame.duplicate_count.tolist() == [1, 1]


def test_retrieval_before_publication_rejected():
    with pytest.raises(ValueError, match="after retrieval"):
        normalize_documents([raw(retrieved_at="2023-12-14")], dictionary())


def test_future_update_snapshot_is_rejected():
    with pytest.raises(ValueError, match="Update timestamp"):
        normalize_documents([raw(updated_at="2027-01-01")], dictionary())


def test_exact_publication_with_day_only_update_today_remains_retrieval_only():
    frame = normalize_documents([raw(publication_date_precision="timestamp",
        updated_at="2026-10-07T23:59:59", updated_date_precision="day")], dictionary())
    row = frame.iloc[0]
    assert row.updated_date_precision == "day"
    assert row.publication_date_precision == "timestamp"
    assert row.availability_status == "retrieval_only"
    assert row.available_at == row.retrieved_at == pd.Timestamp("2026-10-07T09:00:00Z")
    assert row.updated_at == pd.Timestamp("2026-10-07T20:59:59Z")


@pytest.mark.parametrize("precision", ["timestamp", ""])
def test_day_publication_never_allows_exact_future_update_today(precision):
    with pytest.raises(ValueError, match="Update timestamp is after retrieval"):
        normalize_documents([raw(publication_date_precision="day",
            updated_at="2026-10-07T16:00:00+03:00", updated_date_precision=precision)], dictionary())


def test_day_only_update_tomorrow_is_rejected():
    with pytest.raises(ValueError, match="Update timestamp is after retrieval"):
        normalize_documents([raw(publication_date_precision="timestamp",
            updated_at="2026-10-08T23:59:59", updated_date_precision="day")], dictionary())


def test_raw_date_only_update_precision_is_inferred_from_update_itself():
    row = normalize_documents([raw(publication_date_precision="timestamp",
        updated_at="2026-10-07")], dictionary()).iloc[0]
    assert row.updated_date_precision == "day"
    assert row.updated_at == pd.Timestamp("2026-10-07T20:59:59Z")
    assert row.available_at == row.retrieved_at
    assert row.availability_status == "retrieval_only"


def test_raw_date_only_update_tomorrow_is_rejected():
    with pytest.raises(ValueError, match="Update timestamp is after retrieval"):
        normalize_documents([raw(updated_at="2026-10-08")], dictionary())


def test_empty_normalized_schema():
    frame = normalize_documents([], dictionary())
    assert frame.empty and "available_at" in frame
