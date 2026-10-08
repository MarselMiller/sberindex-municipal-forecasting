"""Synthetic failure checks for the saved-reference reproduction gate."""
import hashlib

import numpy as np
import pandas as pd
import pytest

from sberforecast.metrics import KEY
from sberforecast.national_local import build_national
from sberforecast.national_local_evaluation import MODELS, evaluate_national_local
from sberforecast.national_local_persistence_gate import (
    METRIC_COLUMNS, _compare, _fresh_national, _verify_hashes, _verify_reference_tables,
)


@pytest.fixture
def sources(tmp_path):
    expected = pd.DataFrame([
        ("synthetic-a", "2024-06-30", "2024-07-01", 1, 10.0, "holdout"),
        ("synthetic-b", "2024-06-30", "2024-07-01", 1, 30.0, "holdout"),
        ("synthetic-a", "2024-05-31", "2024-06-01", 1, 12.0, "validation"),
        ("synthetic-a", "2023-12-31", "2024-12-01", 12, 20.0, "holdout"),
        ("synthetic-b", "2023-12-31", "2024-12-01", 12, np.nan, "holdout"),
    ], columns=KEY + ["y_true", "split"])
    expected["history_cutoff"] = expected.forecast_origin
    expected["availability_assumption"] = "month_end_plus_0_months"
    frames = []
    for i, model in enumerate(MODELS):
        frame = expected.assign(model=model, y_pred=expected.y_true.fillna(20) - i - 1,
                                status="native", effective_model=model, reason="")
        if model in ("C0", "L0", "CN", "LN"):
            frame.loc[frame.horizon.eq(12), ["status", "effective_model", "reason"]] = [
                "fallback_no_training_pairs", "SeasonalNaive", "no_training_pairs"]
        frames.append(frame)
    predictions = pd.concat(frames, ignore_index=True)
    national = expected[["forecast_origin", "target_period", "horizon", "split"]].drop_duplicates()
    national = national.assign(n_hat=18.0, n_actual=20.0, national_forecast_status="native")
    tables = evaluate_national_local(predictions, expected, national)
    for name, frame in tables.items():
        frame.to_csv(tmp_path / f"{name}.csv", index=False)
    return tmp_path, expected, predictions, national, tables


def test_saved_metrics_are_rescored_and_annual_missingness_preserved(sources):
    source, _, _, _, tables = sources
    checks = _verify_reference_tables(tables, source)
    assert checks["metrics_strategy"]["rows"] == 21
    annual = tables["metrics_native"].query("horizon == 12")
    assert annual.metric_status.eq("no_native_forecasts").all()
    assert annual.mae_macro.isna().all()


@pytest.mark.parametrize("column,value", [
    ("mae_macro", 1.0), ("r2_pooled", 0.02), ("n_predictions", 1),
    ("n_origins", 1), ("metric_status", "incomplete_failed"),
])
def test_saved_metric_or_exact_protocol_damage_rejected(sources, column, value):
    source, _, _, _, tables = sources
    name = "metrics_strategy_by_origin"
    damaged = pd.read_csv(source / f"{name}.csv")
    selected = damaged.model.eq("LN") & damaged.horizon.eq(1)
    if column == "metric_status":
        damaged.loc[selected, column] = value
    else:
        damaged.loc[selected, column] += value
    damaged.to_csv(source / f"{name}.csv", index=False)
    with pytest.raises(ValueError, match=f"{name}/{column}"):
        _verify_reference_tables(tables, source)


def test_changed_prediction_cannot_pass_by_reusing_saved_metrics(sources):
    source, expected, predictions, national, _ = sources
    predictions.loc[predictions.model.eq("LN") & predictions.horizon.eq(1), "y_pred"] += 4
    recalculated = evaluate_national_local(predictions, expected, national)
    with pytest.raises(ValueError, match="metrics_strategy/"):
        _verify_reference_tables(recalculated, source)


def test_same_key_count_with_different_municipality_is_rejected(sources):
    source, _, _, _, tables = sources
    damaged = pd.read_csv(source / "strategy_keys.csv", dtype={"municipality_id": str})
    damaged.loc[0, "municipality_id"] = "synthetic-other"
    damaged.to_csv(source / "strategy_keys.csv", index=False)
    with pytest.raises(ValueError, match="strategy_keys/"):
        _verify_reference_tables(tables, source)


def test_true_fact_or_cutoff_change_rejected_before_metric_comparison(sources):
    _, expected, predictions, national, _ = sources
    predictions.loc[0, "history_cutoff"] = "2024-07-31"
    with pytest.raises(ValueError, match="history_cutoff"):
        evaluate_national_local(predictions, expected, national)


def test_roundoff_allowed_but_missing_metric_or_count_change_not_allowed():
    original = pd.DataFrame({"horizon": [1, 12], "mae_macro": [1.0, np.nan], "n_predictions": [3, 0]})
    close = original.copy()
    close.loc[0, "mae_macro"] += 5e-9
    _compare(close, original, ["horizon"], "synthetic_metrics", METRIC_COLUMNS)
    close.loc[1, "mae_macro"] = 0.0
    with pytest.raises(ValueError, match="mae_macro"):
        _compare(close, original, ["horizon"], "synthetic_metrics", METRIC_COLUMNS)


def test_empty_reason_csv_inference_does_not_hide_nonempty_failure_reason():
    fresh = pd.DataFrame({"horizon": [1], "reason": [""]})
    saved = pd.DataFrame({"horizon": [1], "reason": [np.nan]})
    _compare(fresh, saved, ["horizon"], "national_forecasts")
    fresh.loc[0, "reason"] = "synthetic failure"
    with pytest.raises(ValueError, match="reason"):
        _compare(fresh, saved, ["horizon"], "national_forecasts")


def test_fresh_national_prediction_status_and_horizon_are_guarded():
    panel = pd.DataFrame({"synthetic-a": np.arange(36) + 20.0,
                          "synthetic-b": np.arange(36) + 40.0},
                         index=pd.date_range("2022-01-01", periods=36, freq="MS"))
    expected = pd.DataFrame({"forecast_origin": ["2023-12-31"], "horizon": [6], "split": ["holdout"]})
    cfg = {"models": {"yearly_growth_window": 3, "yearly_growth_bounds": [.5, 2]},
           "data": {"release_lag_months": 0}}
    fresh = _fresh_national(build_national(panel), expected, cfg)
    saved = fresh.copy()
    saved.loc[0, "N_hat"] += 1
    with pytest.raises(ValueError, match="N_hat"):
        _compare(fresh, saved, ["forecast_origin", "horizon"], "national_forecasts", {"N_hat"})
    # A future value may change evaluation-only N_actual, but never N_hat.
    changed = panel.copy()
    changed.loc[changed.index > "2023-12-01"] *= 10
    future = _fresh_national(build_national(changed), expected, cfg)
    assert fresh.N_hat.equals(future.N_hat)
    assert fresh.status.equals(future.status)
    assert not fresh.N_actual.equals(future.N_actual)


def test_historical_hash_mismatch_missing_file_and_escape_rejected(tmp_path):
    path = tmp_path / "source.txt"
    path.write_bytes(b"synthetic public fixture")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    verified = {}
    _verify_hashes(tmp_path, {"source.txt": digest}, verified)
    assert verified == {"source.txt": digest}
    path.write_bytes(b"changed synthetic fixture")
    with pytest.raises(ValueError, match="SHA mismatch"):
        _verify_hashes(tmp_path, {"source.txt": digest}, {})
    with pytest.raises(ValueError, match="SHA mismatch"):
        _verify_hashes(tmp_path, {"missing.txt": digest}, {})
    with pytest.raises(ValueError, match="inside the repository"):
        _verify_hashes(tmp_path, {"../outside.txt": digest}, {})
