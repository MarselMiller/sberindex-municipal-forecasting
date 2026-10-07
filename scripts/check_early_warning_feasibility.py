"""Verify new E07a files and their saved provenance; no old full-tree check."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/early_warning_feasibility_v1"
NEWS = ROOT / "outputs/news_events_v3"


def read(filename: Path) -> pd.DataFrame:
    return pd.read_csv(filename,dtype={"municipality_id":str,"series_id":str,"region_id":str})


def main() -> None:
    checks = []
    def verify(condition: bool, name: str):
        if not condition:
            raise AssertionError(name)
        checks.append(name)
    old = read(ROOT / "outputs/news_events_v2/forecast_cases.csv.gz")
    features = read(OUT / "features.csv.gz")
    news = read(NEWS / "news_features.csv.gz")
    keys = ["municipality_id","forecast_origin"]
    verify(features[keys].equals(old[keys]) and news[keys].equals(old[keys]),"all768original_keys_and_order_preserved")
    cases = read(OUT / "weak_cases.csv.gz")
    verify(len(cases)==1536 and not cases.duplicated(keys+["k"]).any(),"all1536warning_cases_preserved")
    verify(cases.loc[~cases.fully_known,"label"].isna().all(),"unknown_future_or_past_never_negative")
    verify(cases.loc[cases.right_censored,"label"].isna().all(),"right_censoring_never_zero")
    verify((cases.at_risk == ~cases.known_active_at_origin).all(),"risk_exclusion_only_known_active_regime")
    known = cases.loc[cases.fully_known].copy()
    origin_months = pd.to_datetime(known.forecast_origin).dt.to_period("M")
    required = [str(month+int(k)+2) for month,k in zip(origin_months,known.k)]
    verify(known.required_confirmation_end_period.tolist()==required,"confirmation_requires_O_plus_k_plus_2")
    verify((pd.to_datetime(known.label_known_at).dt.to_period("M").astype(str)>=known.required_confirmation_end_period).all(),"labels_wait_for_whole_future_window")
    split = read(OUT / "temporal_split_cases.csv.gz")
    train = split.loc[split.split.eq("train")]
    verify(train.fully_known.all() and train.at_risk.all() and pd.to_datetime(train.label_known_at).le(pd.Timestamp("2024-07-31")).all(),"train_labels_known_by_fixed_cutoff")
    for name in ("history_feature_provenance","macro_feature_provenance"):
        table = read(OUT / (name+".csv.gz"))
        available = pd.to_datetime(table.max_available_at,utc=True)
        cutoff = pd.to_datetime(table.as_of_utc,utc=True)
        verify((available.isna()|available.le(cutoff)).all(),name+"_no_future_dependency")
    verify(not any("offline" in col or "breakpoint" in col for col in features),"offline_breakpoints_not_features")
    candidates = read(OUT / "weak_candidates.csv.gz")
    accepted = candidates.loc[candidates.is_candidate]
    for deviations in accepted.standardized_deviations:
        values = np.array(json.loads(deviations))
        verify(bool(np.all(values>=3) or np.all(values<=-3)),"three_month_same_signed_threshold")
    events = read(OUT / "weak_events.csv.gz")
    verify(events.event_id.is_unique,"weak_onsets_have_unique_event_ids")
    for row in known.itertuples():
        period = pd.Period(row.forecast_origin,freq="M")
        eligible = events.loc[events.series_id.eq(row.municipality_id)]
        onsets = pd.PeriodIndex(eligible.onset_period,freq="M")
        expected = bool(((onsets>period)&(onsets<=period+row.k)).any())
        verify(bool(row.label)==expected,"case_label_is_future_onset_not_confirmation")
    documents = read(NEWS / "documents.csv.gz")
    old_documents = read(ROOT / "outputs/news_events_v2/documents.csv.gz")
    verify(set(old_documents.snapshot_id).issubset(set(documents.snapshot_id)),"all180old_snapshots_retained")
    historical = documents.loc[documents.availability_status.eq("confirmed_historical")]
    verify(historical.canonical_event_id.nunique()==17,"release_PDF_versions_unified_to17decisions")
    event_table = read(NEWS / "events.csv.gz")
    core_events = event_table.loc[event_table.canonical_event_id.str.startswith("cbr_key_rate_decision_")]
    verify(len(core_events)==17 and core_events.availability_status.eq("confirmed_historical").all(),"canonical_summary_uses_first_available_core")
    verify(news.groupby("forecast_origin").news_count_30d.nunique().le(1).all(),"national_copies_are_one_vector_per_origin")
    queue = read(OUT / "manual_review_queue.csv.gz")
    verify(queue.annotation_status.eq("pending_human_review").all() and not bool(queue.independent_annotation.any()),"AI_audit_not_independent_human_labels")
    verify(queue.selection.eq("finite_control").any() and queue.selection.eq("missing_or_censored").any(),"manual_queue_includes_controls_and_missing")
    manifests = [OUT / "run_manifest.json",NEWS / "run_manifest.json",ROOT / "outputs/early_warning_news_audit_v1/run_manifest.json"]
    digest = lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    for file in manifests:
        data = json.loads(file.read_text(encoding="utf-8"))
        for relative,sha in data["inputs"].items():
            verify(digest(ROOT / relative)==sha,"saved_input_SHA_matches")
        for relative,record in data["artifacts"].items():
            verify(digest(file.parent / relative)==(record["sha256"] if isinstance(record,dict) else record),"saved_artifact_SHA_matches")
        for relative,sha in data.get("code",{}).items():
            verify(digest(ROOT / relative)==sha,"executed_code_SHA_matches")
    result = dict(passed=True,assertions=len(checks),distinct_checks=sorted(set(checks)),
                  feature_rows=len(features),case_rows=len(cases),weak_events=len(events),independent_human_annotations=0)
    target = ROOT / "outputs/e07a_checks/verification.json"
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(dict(passed=True,assertions=len(checks),weak_events=len(events)),ensure_ascii=False))


if __name__ == "__main__":
    main()
