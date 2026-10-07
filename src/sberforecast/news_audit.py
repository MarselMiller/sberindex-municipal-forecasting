"""Coverage of the observed corpus, not coverage of all economic events."""
from __future__ import annotations

import pandas as pd


def document_audit(documents: pd.DataFrame, start: str, end: str) -> dict:
    n = len(documents)
    unique_documents = documents.document_id.nunique()
    events = documents.canonical_event_id.nunique()
    def counts(column: str) -> dict:
        values = documents[column].fillna("unknown").replace("", "unknown")
        return {str(k): int(v) for k, v in values.value_counts().sort_index().items()}
    dates = pd.to_datetime(documents.published_at, utc=True)
    month_values = dates.dt.tz_convert("Europe/Moscow").dt.strftime("%Y-%m").value_counts()
    months = {str(month): int(month_values.get(str(month), 0)) for month in pd.period_range(start, end, freq="M")}
    share = lambda count: float(count / n) if n else None
    return dict(snapshots=n, documents=int(unique_documents), canonical_events=int(events),
        duplicates_removed=int(unique_documents-events), deduplicated_share=float((unique_documents-events)/unique_documents) if unique_documents else None,
        by_source=counts("source"), by_month=months, by_geography_level=counts("geography_level"),
        by_region=counts("region"), by_topic=counts("topic"), by_event_type=counts("event_type"),
        by_availability_status=counts("availability_status"),
        unknown_geography_share=share(documents.geography_level.eq("unknown").sum()),
        unknown_topic_share=share(documents.topic.isin(["other", "unknown"]).sum()),
        unknown_event_type_share=share(documents.event_type.isin(["other", "unknown"]).sum()),
        missing_publication_date=int(dates.isna().sum()),
        interpretation="Observed bounded source corpus; absent document does not imply absent event")


def feature_coverage(features: pd.DataFrame, feature_names: tuple | list) -> pd.DataFrame:
    rows = []
    for origin, group in features.groupby("forecast_origin", sort=True):
        for name in feature_names:
            values = group[name]
            rows.append(dict(forecast_origin=origin, feature=name, cases=len(group),
                nonmissing=int(values.notna().sum()), missing=int(values.isna().sum()),
                nonzero=int((values.notna() & values.ne(0)).sum()),
                nonmissing_share=float(values.notna().mean())))
    return pd.DataFrame(rows)


def origin_coverage(features: pd.DataFrame) -> pd.DataFrame:
    frame = features[["municipality_id", "forecast_origin", "region_id"]].copy()
    frame["has_regional_30d"] = features.regional_news_count_30d.gt(0)
    frame["only_national_30d"] = (features.national_news_count_30d.gt(0) &
        features.regional_news_count_30d.eq(0) & features.municipal_news_count_30d.eq(0))
    frame["no_observed_news_30d"] = features.news_count_30d.eq(0)
    frame["news_window_missing"] = features.news_count_30d.isna()
    return frame


def assert_feature_keys(features: pd.DataFrame, samples: pd.DataFrame) -> None:
    key = ["municipality_id", "forecast_origin"]
    if features.duplicated(key).any() or samples.duplicated(key).any():
        raise AssertionError("Duplicate municipality / forecast origin key")
    if len(features) != len(samples) or not features[key].reset_index(drop=True).equals(samples[key].reset_index(drop=True)):
        raise AssertionError("News pipeline altered or deleted forecast keys")
