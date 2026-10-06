"""E05b guards and a one-date synthetic orchestration without model fits."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from sberforecast import trend_calendar_experiment as experiment
from sberforecast.calendar_models import training_signature, transform_month_features
from sberforecast.direct_experiment import expected_cases
from sberforecast.direct_model import DirectResult
from sberforecast.direct_training import build_direct_training

ROOT = Path(__file__).resolve().parents[1]


def configuration():
    return yaml.safe_load((ROOT / "configs/trend_calendar.yaml").read_text(encoding="utf-8"))


@pytest.mark.parametrize("field,value", [("phi", .8), ("min_annual_pairs", 2), ("annual_difference_window_months", 12)])
def test_frozen_trend_specification_rejects_unannounced_changes(field, value):
    cfg = configuration()
    cfg["trends"][field] = value
    with pytest.raises(ValueError, match="fixed E05a"):
        experiment.validate_specification(cfg)


def test_full_test_record_is_required_for_current_source_hashes(tmp_path, monkeypatch):
    cfg = configuration()
    record = tmp_path / "result.json"
    cfg["validation_record"] = "result.json"
    monkeypatch.setattr(experiment, "tested_files", lambda root: {"code": "current"})
    for value in ({"exit_code": 1, "full_pytest": True, "code_sha256": {"code": "current"}},
                  {"exit_code": 0, "full_pytest": False, "code_sha256": {"code": "current"}},
                  {"exit_code": 0, "full_pytest": True, "code_sha256": {"code": "stale"}}):
        record.write_text(json.dumps(value), encoding="utf-8")
        with pytest.raises(ValueError, match="full pytest"):
            experiment.require_test_record(tmp_path, cfg)


@pytest.mark.parametrize("complete_run", [False, True])
def test_one_date_orchestration_reuses_sources_and_resumes_without_fits(tmp_path, monkeypatch, complete_run):
    cfg = configuration()
    panel = pd.DataFrame({"1": np.arange(24.) + 100, "2": np.arange(24.) + 200},
                         index=pd.date_range("2023-01", periods=24, freq="MS"))
    panel.columns.name = "municipality_id"
    all_cases = expected_cases(panel, cfg, {"1", "2"})
    cases = all_cases.loc[all_cases.forecast_origin.eq("2024-03-31")]
    reference = cases.assign(model="SeasonalNaiveYoY", status="native", y_pred=cases.y_true)
    e02 = cases.assign(model="saved_E02", status="native", effective_model="CatBoostDirect",
                       reason="", y_pred=cases.y_true)
    old_training = []
    for h in cases.horizon.unique():
        X, delta, trace = build_direct_training(panel, "2024-03", int(h), release_lag_months=0, mode="legacy")
        old_training.append({"forecast_origin": "2024-03-31", "horizon": int(h), "n_training_rows": len(trace),
            "n_training_municipalities": trace.municipality_id.nunique(), "n_historical_origins": trace.historical_origin.nunique(),
            "n_target_periods": trace.target_period.nunique(), "fit_called": True, "fit_succeeded": True})
    calls = []
    class SyntheticPredictor:
        def __init__(self, parameters, seed, encoding, *args):
            self.encoding = encoding
        def predict(self, data, origin, horizon, ids, baseline_config):
            calls.append((self.encoding, str(origin), horizon))
            X, delta, trace = build_direct_training(data, origin, horizon, release_lag_months=0, mode="legacy")
            transformed = transform_month_features(X, self.encoding)
            training = dict(next(row for row in old_training if row["horizon"] == horizon),
                **training_signature(trace, X, delta), month_encoding=self.encoding,
                feature_names=transformed.columns.tolist(), feature_dtypes={c: str(t) for c, t in transformed.dtypes.items()},
                fit_called=False, fit_succeeded=False, status="native")
            forecast = pd.DataFrame({"municipality_id": ids, "y_pred": np.zeros(len(ids)), "status": "native",
                                    "effective_model": f"CatBoostDirect{self.encoding}", "reason": ""})
            return DirectResult(forecast, training, pd.DataFrame(columns=["feature", "importance"]), [])
    data_path = tmp_path / cfg["data"]["path"]
    data_path.parent.mkdir(parents=True)
    data_path.write_text("synthetic input marker", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    validation_path = tmp_path / cfg["validation_record"]
    validation_path.parent.mkdir(parents=True)
    validation_path.write_text("synthetic successful test marker", encoding="utf-8")
    monkeypatch.setattr(experiment, "load_data", lambda *args: panel)
    monkeypatch.setattr(experiment, "make_panel", lambda data: data)
    monkeypatch.setattr(experiment, "require_test_record", lambda *args: {"exit_code": 0, "full_pytest": True})
    monkeypatch.setattr(experiment, "tested_files", lambda root: {"synthetic_code": "same"})
    monkeypatch.setattr(experiment, "verify_reference", lambda *args: (["1", "2"], reference, all_cases, {}))
    monkeypatch.setattr(experiment, "verify_saved_direct", lambda *args: (e02, pd.DataFrame(old_training), {}))
    monkeypatch.setattr(experiment, "CalendarCatBoostDirect", SyntheticPredictor)
    monkeypatch.setattr(experiment.subprocess, "check_output", lambda args, **kwargs: "synthetic_commit" if args[1] == "rev-parse" else "")
    if complete_run:
        from sberforecast import trend_calendar_report
        monkeypatch.setattr(experiment, "origins_from_config", lambda cfg: [pd.Period("2024-03", freq="M")])
        def require_completed_manifest(output, report_path, cfg):
            assert json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))["complete"] is True
        monkeypatch.setattr(trend_calendar_report, "write_report", require_completed_manifest)
    result = experiment.run_experiment(tmp_path, config_path, ["2024-03"], "synthetic smoke")
    assert result["complete"] is complete_run and result["n_origins_completed"] == 1
    assert result["n_new_fits_called"] == 0 and len(calls) == 6
    assert {encoding for encoding, _, _ in calls} == {"K1", "K2"}
    result = experiment.run_experiment(tmp_path, config_path, ["2024-03"], "synthetic resume")
    assert len(calls) == 6 and result["n_failed"] == 0
    partition = tmp_path / "outputs/trend_calendar_v1/partitions/predictions_2024-03.csv.gz"
    partition.write_bytes(b"changed")
    with pytest.raises(ValueError, match="missing or changed"):
        experiment.run_experiment(tmp_path, config_path, ["2024-03"], "synthetic corrupted resume")
