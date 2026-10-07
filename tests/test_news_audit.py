import pandas as pd
import pytest

from sberforecast.news_audit import assert_feature_keys, document_audit, feature_coverage, origin_coverage
from sberforecast.news_features import build_news_features, FEATURE_COLUMNS
from sberforecast.news_normalize import DOCUMENT_COLUMNS


def test_empty_corpus_and_every_month_explicit():
    docs = pd.DataFrame(columns=DOCUMENT_COLUMNS)
    audit = document_audit(docs, "2023-01-01", "2024-12-31")
    assert len(audit["by_month"]) == 24
    assert set(audit["by_month"].values()) == {0}
    assert audit["deduplicated_share"] is None


def test_coverage_keeps_absent_news_cases_and_missing_shares():
    samples = pd.DataFrame([dict(municipality_id="1", region_id="1", forecast_origin="2024-01-31")])
    features = build_news_features(pd.DataFrame(columns=DOCUMENT_COLUMNS), samples)
    assert_feature_keys(features, samples)
    coverage = feature_coverage(features, FEATURE_COLUMNS)
    assert len(coverage) == 26
    assert coverage.set_index("feature").loc["negative_share_30d", "missing"] == 1
    origin = origin_coverage(features)
    assert origin.no_observed_news_30d.iloc[0]


def test_deleted_key_is_failure():
    samples = pd.DataFrame([dict(municipality_id="1", forecast_origin="2024-01-31")])
    with pytest.raises(AssertionError, match="altered or deleted"):
        assert_feature_keys(samples.iloc[:0], samples)
