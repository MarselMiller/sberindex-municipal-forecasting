"""Integration assertions for E07a admission counts and fixed time split."""
import importlib.util
from pathlib import Path

import pandas as pd
from sberforecast.news_features import build_news_features

SPEC = importlib.util.spec_from_file_location("e07a_prepare",Path(__file__).resolve().parents[1]/"scripts/prepare_early_warning.py")
PREPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREPARE)


def cases():
    return pd.DataFrame([
        dict(municipality_id="1",forecast_origin="2024-04-30",k=1,label=1.,fully_known=True,
             label_known_at="2024-07-31",at_risk=True,positive_event_ids='["event_a", "event_b"]'),
        dict(municipality_id="2",forecast_origin="2024-04-30",k=1,label=1.,fully_known=True,
             label_known_at="2024-07-31",at_risk=True,positive_event_ids='["event_a"]'),
        dict(municipality_id="1",forecast_origin="2024-08-31",k=1,label=0.,fully_known=True,
             label_known_at="2024-11-30",at_risk=True,positive_event_ids='[]'),
        dict(municipality_id="1",forecast_origin="2024-11-30",k=1,label=float("nan"),fully_known=False,
             label_known_at="",at_risk=True,positive_event_ids='[]'),
        dict(municipality_id="3",forecast_origin="2024-04-30",k=1,label=0.,fully_known=True,
             label_known_at="2024-07-31",at_risk=False,positive_event_ids='[]'),
    ])


def test_event_count_uses_json_event_identity_not_positive_rows():
    assert PREPARE.event_ids(cases()) == {"event_a","event_b"}
    assert PREPARE.event_ids(cases().iloc[2:]) == set()


def test_fixed_split_waits_for_labels_and_preserves_censoring_and_active_exclusion():
    config = {"early_warning":{"horizons_months":[1,3],"split":{
        "train_information_cutoff":"2024-07","test_first_origin":"2024-08","test_last_origin":"2024-11"}}}
    events = pd.DataFrame({"event_id":["event_a","event_b"],"onset_period":["2024-05","2024-05"]})
    rows,summary = PREPARE.temporal_split(cases(),config,events)
    train = summary.loc[summary.k.eq(1) & summary.split.eq("train")].iloc[0]
    assert (train.cases,train.event_ids,train.event_onset_dates,train.origin_dates) == (2,2,1,1)
    assert summary.loc[summary.k.eq(3),"cases"].sum() == 0
    assert len(rows.loc[rows.split.eq("test_evaluable")]) == 1
    censored = rows.loc[rows.split.eq("test_unavailable_or_active")]
    assert len(censored) == 1 and censored.label.isna().all()
    config["early_warning"]["split"]["train_information_cutoff"] = "2024-06"
    _,earlier = PREPARE.temporal_split(cases(),config,events)
    assert earlier.loc[earlier.split.eq("train"),"cases"].sum() == 0


def test_v3_groups_pdf_and_core_with_mixed_dates_and_locks_early_labels():
    def record(doc,snapshot,available,published,direction="unknown",topic="macro_policy"):
        return dict(document_id=doc,snapshot_id=snapshot,canonical_event_id="v2_"+doc,
            available_at=available,published_at=published,retrieved_at="2026-10-07T00:00:00.123456+00:00",
            updated_at=None,event_date=None,geography_level="national",region="Россия",region_id="",municipality_id="",
            topic=topic,event_type="announcement",direction=direction,classification_rule="old_rule",
            geography_rule="old_rule",availability_status="confirmed_historical")
    pdf = record("pdf","pdf_snapshot","2024-02-16T20:59:59+00:00","2024-02-16T20:59:59+00:00")
    article = record("release","retrieved_article","2026-10-07T00:00:00.123456+00:00","2024-02-16T10:30:00+00:00")
    core = record("release","core_snapshot",pd.Timestamp("2024-02-16T10:30:00+00:00"),pd.Timestamp("2024-02-16T10:30:00+00:00"))
    mapping = {"pdf":"cbr_key_rate_decision_2024-02-16","release":"cbr_key_rate_decision_2024-02-16"}
    combined = PREPARE.combine_decision_snapshots(pd.DataFrame([pdf,article]),pd.DataFrame([core]),mapping)
    assert len(combined)==3 and combined.is_canonical.sum()==1
    assert combined.loc[combined.is_canonical,"snapshot_id"].iloc[0]=="core_snapshot"
    assert combined.loc[combined.snapshot_id.eq("pdf_snapshot"),"classification_rule"].iloc[0]=="old_rule"
    sample = pd.DataFrame([dict(municipality_id="1",region_id="2",forecast_origin="2024-02-29")])
    early = build_news_features(combined,sample)
    assert early.news_count_30d.iloc[0]==1
    assert early.duplicate_share_30d.iloc[0]==0.5
    late = record("release","later_revision","2024-03-01T00:00:00+00:00","2024-02-16T10:30:00+00:00","negative","disaster_emergency")
    with_future = PREPARE.combine_decision_snapshots(pd.DataFrame([pdf,article,late]),pd.DataFrame([core]),mapping)
    pd.testing.assert_frame_equal(early,build_news_features(with_future,sample))
