"""Проверки описательных сравнений на синтетических агрегатах."""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import yaml

from sberforecast.national_local_persistence_experiment import (
    DIRECT, LEARNED, PERSISTENCE, classify, gap_analysis, join_forecasts, municipality_diagnostics, origin_diagnostics, validate_protocol,
)


def metrics(values):
    return pd.DataFrame([dict(split="holdout", horizon=1, model=name,
                              mae_macro=value, metric_status="complete", n_predictions=6)
                         for name, value in zip(["LastValue", "SeasonalNaiveYoY", DIRECT, PERSISTENCE, LEARNED], values)])


@pytest.mark.parametrize("direct,persistence,learned,share", [(100, 80, 90, 2.0), (100, 110, 90, -1.0)])
def test_observed_gap_share_is_not_clipped(direct, persistence, learned, share):
    table = gap_analysis(metrics([140, 120, direct, persistence, learned]))
    row = table.loc[table.from_model.eq(DIRECT)].iloc[0]
    assert row.observed_share_of_mae_gap == share
    assert row.reduction_mae == direct - persistence
    assert row.reduction_pct == pytest.approx(100 * (direct - persistence) / direct)


@pytest.mark.parametrize("learned", [100, 110])
def test_nonpositive_gap_denominator_remains_na(learned):
    table = gap_analysis(metrics([140, 120, 100, 80, learned]))
    row = table.loc[table.from_model.eq(DIRECT)].iloc[0]
    assert np.isnan(row.observed_share_of_mae_gap)
    assert row.share_status == "nonpositive_or_unavailable_denominator"


def test_origin_stability_counts_dates_and_keeps_best_worst():
    first = metrics([140, 120, 100, 80, 70]).assign(forecast_origin="2024-01-31")
    second = metrics([140, 120, 100, 130, 90]).assign(forecast_origin="2024-02-29")
    deltas, stability = origin_diagnostics(pd.concat([first, second], ignore_index=True))
    row = stability.loc[stability.from_model.eq(DIRECT)].iloc[0]
    assert (row.n_origins, row.n_improved, row.n_worsened) == (2, 1, 1)
    assert row.mean_origin_reduction == row.median_origin_reduction == -5
    assert row.best_origin == "2024-01-31" and row.best_reduction == 20
    assert row.worst_origin == "2024-02-29" and row.worst_reduction == -30
    assert len(deltas) == 2 * 4


def test_municipality_diagnostics_aggregate_identifiers_and_do_not_count_ties_as_wins():
    rows = []
    for uid, persistence in [("synthetic_a", 10), ("synthetic_b", 20), ("synthetic_c", 30)]:
        for model in ["SeasonalNaiveYoY", DIRECT, LEARNED, PERSISTENCE]:
            rows.append(dict(split="holdout", horizon=1, municipality_id=uid, model=model,
                             mae=persistence if model == PERSISTENCE else 20))
    table = municipality_diagnostics(pd.DataFrame(rows))
    assert "municipality_id" not in table
    assert not table.astype(str).apply(lambda col: col.str.contains("synthetic_").any()).any()
    assert table.n_better.eq(1).all() and table.n_worse.eq(1).all() and table.n_tied.eq(1).all()
    assert table.share_better.eq(1/3).all() and table.median_municipality_reduction.eq(0).all()


def test_failed_strategy_cannot_get_complete_delta():
    table = metrics([140, 120, 100, 80, 70])
    table.loc[table.model.eq(PERSISTENCE), ["mae_macro", "metric_status"]] = [np.nan, "incomplete_failed"]
    gaps = gap_analysis(table)
    assert gaps.reduction_mae.isna().all()
    assert gaps.metric_status.eq("incomplete").all()


@pytest.mark.parametrize("values,category", [([200,150,120,101,100], "A"),
                                           ([200,150,140,120,100], "B"),
                                           ([150,150,160,155,100], "C")])
def test_predeclared_categories_require_same_direction_in_all_primary_splits(values, category):
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/national_local_persistence.yaml").read_text(encoding="utf-8"))
    rows = [metrics(values).assign(split=split, horizon=horizon)
            for split in ["validation", "holdout"] for horizon in [1,3,6]]
    table = pd.concat(rows, ignore_index=True)
    _, origins = origin_diagnostics(table.assign(forecast_origin="2024-03-31"))
    result = classify(table, origins, gap_analysis(table), cfg["interpretation"])
    assert result["overall"] == category
    assert all(item["category"] == category for item in result["per_horizon"])


def test_mixed_split_direction_has_priority_over_closeness():
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/national_local_persistence.yaml").read_text(encoding="utf-8"))
    rows = [metrics([200,150,120,101,100] if split == "validation" else [200,150,120,151,100]).assign(split=split, horizon=horizon)
            for split in ["validation", "holdout"] for horizon in [1,3,6]]
    table = pd.concat(rows, ignore_index=True)
    _, origins = origin_diagnostics(table.assign(forecast_origin="2024-03-31"))
    assert classify(table, origins, gap_analysis(table), cfg["interpretation"])["overall"] == "D"


def test_partial_large_benefit_cannot_be_classified_as_learned_only():
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/national_local_persistence.yaml").read_text(encoding="utf-8"))
    table = pd.concat([metrics([200,150,160,155,100]).assign(split=split, horizon=horizon)
                       for split in ["validation", "holdout"] for horizon in [1,3,6]], ignore_index=True)
    _, origins = origin_diagnostics(table.assign(forecast_origin="2024-03-31"))
    assert classify(table, origins, gap_analysis(table), cfg["interpretation"])["overall"] == "D"


@pytest.mark.parametrize("field,new_value", [("ratio_rule", "smoothed_ratio"), ("seed", 1),
                                           ("primary_horizons", [1]), ("new_reference_fits", 1)])
def test_protocol_rejects_rule_or_horizon_search(field, new_value):
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/national_local_persistence.yaml").read_text(encoding="utf-8"))
    cfg[field] = new_value
    with pytest.raises(ValueError, match="Pre-specified protocol changed"):
        validate_protocol(cfg)


def test_prediction_join_preserves_all_canonical_keys_and_truth():
    expected = pd.DataFrame(dict(municipality_id=["synthetic_a"], forecast_origin=["2024-03-31"],
                                 target_period=["2024-06-01"], horizon=[3], y_true=[99.0]))
    forecast = expected.drop(columns="y_true").assign(y_pred=90.0)
    forecast["target_period"] = pd.to_datetime(forecast.target_period)
    joined = join_forecasts(expected, forecast)
    assert joined.y_true.iloc[0] == 99
    assert joined.y_pred.iloc[0] == 90
    assert set(expected) <= set(joined)
    assert not any(name.endswith(("_x", "_y")) for name in joined)
    forecast["target_period"] = pd.Timestamp("2024-07-01")
    with pytest.raises(ValueError, match="Несовпадение ключей"):
        join_forecasts(expected, forecast)


def test_failed_reproduction_gate_stops_before_persistence_or_output_creation(tmp_path, monkeypatch):
    import sberforecast.national_local_persistence_experiment as experiment
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/national_local_persistence.yaml").read_text(encoding="utf-8"))
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    monkeypatch.setattr(experiment.subprocess, "check_output", lambda *args, **kwargs: "research/national-local-persistence\n")
    monkeypatch.setattr(experiment, "load_verified_sources", lambda *args, **kwargs: {"gate": {"status": "FAIL"}})
    with pytest.raises(ValueError, match="Reference reproduction gate must PASS"):
        experiment.run_experiment(tmp_path, config_path, command="synthetic_test")
    assert not (tmp_path / "outputs").exists()
