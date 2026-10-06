"""Offline E05c guards and saved-source orchestration, without model fitting."""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from sberforecast import macro_forecast_experiment as experiment
from sberforecast.calendar_models import training_signature
from sberforecast.direct_experiment import expected_cases
from sberforecast.direct_training import build_direct_training
from sberforecast.macro_forecast_features import CONSUMPTION, INFLATION, prepare_forecast_features
from sberforecast.macro_forecast_model import MacroDirectResult, VARIANT_INDICATORS

ROOT = Path(__file__).resolve().parents[1]


def configuration():
    return yaml.safe_load((ROOT / "configs/macro_forecast.yaml").read_text(encoding="utf-8"))


@pytest.mark.parametrize("field,value", [("one_hot_enabled", True), ("missing_training_rows", "drop"),
                                          ("category_B_enabled", True), ("deflation_enabled", True)])
def test_fixed_macro_specification_rejects_unannounced_changes(field, value):
    cfg = configuration()
    cfg["macro"][field] = value
    with pytest.raises(ValueError, match="fixed feature/source specification"):
        experiment.validate_specification(cfg)


def test_frozen_smoke_origin_is_checked():
    cfg = configuration()
    cfg["smoke_origin"] = "2024-05"
    with pytest.raises(ValueError, match="Smoke origin"):
        experiment.validate_specification(cfg)


@pytest.mark.parametrize("record", [
    {"exit_code": 1, "full_pytest": True, "code_sha256": {"code": "current"}},
    {"exit_code": 0, "full_pytest": False, "code_sha256": {"code": "current"}},
    {"exit_code": 0, "full_pytest": True, "code_sha256": {"code": "stale"}},
])
def test_failed_partial_or_stale_full_test_record_blocks_before_model_constructor(tmp_path, monkeypatch, record):
    cfg = configuration()
    cfg["validation_record"] = "tests.json"
    (tmp_path / "tests.json").write_text(json.dumps(record), encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    monkeypatch.setattr(experiment, "tested_files", lambda root: {"code": "current"})
    def prohibit_constructor(*args, **kwargs):
        pytest.fail("No model constructor may be called before a successful current full pytest.")
    monkeypatch.setattr(experiment, "MacroCatBoostDirect", prohibit_constructor)
    with pytest.raises(ValueError, match="full pytest"):
        experiment.run_experiment(tmp_path, config_path, ["2024-03"], "synthetic blocked run")


@pytest.mark.parametrize("bad_audit", [
    {"status": "FAIL", "errors": ["wrong source date"], "checks": {"audited_A_rows": 73}},
    {"status": "PASS", "errors": ["wrong extracted value"], "checks": {"audited_A_rows": 73}},
    {"status": "PASS", "errors": [], "checks": {"audited_A_rows": 72}},
])
def test_source_audit_failure_blocks_before_macro_join_or_model_constructor(tmp_path, monkeypatch, bad_audit):
    cfg = configuration()
    cfg["validation_record"] = "tests.json"
    cfg["macro"]["source_audit_record"] = "audit.json"
    (tmp_path / "tests.json").write_text(json.dumps({"exit_code": 0, "full_pytest": True,
                                                  "code_sha256": {"code": "current"}}), encoding="utf-8")
    (tmp_path / "audit.json").write_text(json.dumps(bad_audit), encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    monkeypatch.setattr(experiment, "tested_files", lambda root: {"code": "current"})
    def prohibited(*args, **kwargs):
        pytest.fail("Failed independent source audit must stop before using its macro data or constructing models.")
    monkeypatch.setattr(experiment, "MacroCatBoostDirect", prohibited)
    monkeypatch.setattr(experiment, "verified_macro_table", prohibited)
    with pytest.raises(ValueError, match="Source audit failed"):
        experiment.run_experiment(tmp_path, config_path, ["2024-03"], "synthetic audit failure")


@pytest.mark.parametrize("field,new_value", [("depth", 7), ("iterations", 301)])
def test_existing_E01_reference_guard_rejects_changed_CatBoost_parameters(tmp_path, field, new_value):
    cfg = configuration()
    old = copy.deepcopy(cfg)
    source = tmp_path / cfg["comparison"]["source_dir"]
    source.mkdir(parents=True)
    (tmp_path / "configs").mkdir()
    old_yaml = yaml.safe_dump(old)
    (source / "config_resolved.yaml").write_text(old_yaml, encoding="utf-8")
    (tmp_path / "configs/prophet_comparison.yaml").write_text(old_yaml, encoding="utf-8")
    (source / "run_manifest.json").write_text(json.dumps({"config": old}), encoding="utf-8")
    cfg["models"]["catboost"][field] = new_value
    with pytest.raises(ValueError, match="Параметры catboost отличаются от E01"):
        experiment.verify_reference(tmp_path, pd.DataFrame(), cfg)


def macro_table():
    common = {
        "region_id": "RU", "geographic_level": "national", "reference_period": "2024",
        "target_period": "2024", "value": 5.0, "unit": "percent_growth",
        "published_at": "2023-12-15", "available_at": "2023-12-15",
        "vintage_id": "synthetic-2023-12", "availability_status": "A",
        "forecast_issue_date": "2023-12-15", "is_forecast": True,
        "forecast_low": np.nan, "forecast_central": 5.0, "forecast_high": np.nan,
        "central_method": "official_central", "forecast_producer": "synthetic publisher",
    }
    return pd.DataFrame([
        dict(common, indicator=INFLATION, source_url="https://official.example/inflation",
             aggregation_basis="december_to_december"),
        dict(common, indicator=CONSUMPTION, source_url="https://official.example/consumption",
             aggregation_basis="annual_real_volume_growth"),
    ])


@pytest.mark.parametrize("changed_prediction", [True, False])
def test_saved_M0_h1_requires_complete_equivalence_before_new_model_construction(tmp_path, monkeypatch, changed_prediction):
    cfg = configuration()
    panel = pd.DataFrame({"1": np.arange(24.) + 100}, index=pd.date_range("2023-01", periods=24, freq="MS"))
    panel.columns.name = "municipality_id"
    expected = expected_cases(panel, cfg, {"1"})
    cases = expected.loc[expected.forecast_origin.eq("2024-03-31")].copy()
    reference = cases.assign(model="CatBoostRecursive", status="native", y_pred=cases.y_true)
    control = cases.assign(model="saved_E02", status="native", y_pred=cases.y_true)
    h1 = control.horizon.eq(1)
    if changed_prediction:
        control.loc[h1, "y_pred"] += 1.0
    else:
        control.loc[h1, "status"] = "failed"
        control.loc[h1, "y_pred"] = np.nan
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    monkeypatch.setattr(experiment, "require_checks", lambda *args: ({"exit_code": 0}, {"status": "PASS"}))
    monkeypatch.setattr(experiment, "verified_macro_table", lambda *args: (macro_table(), {}, {}))
    monkeypatch.setattr(experiment, "load_data", lambda *args: panel)
    monkeypatch.setattr(experiment, "make_panel", lambda data: data)
    monkeypatch.setattr(experiment, "verify_reference", lambda *args: (["1"], reference, expected, {}))
    monkeypatch.setattr(experiment, "verify_saved_direct", lambda *args: (control, pd.DataFrame(), {}))
    def prohibit_constructor(*args, **kwargs):
        pytest.fail("Saved M0 must pass the real h1 consistency calculation before a macro model is constructed.")
    monkeypatch.setattr(experiment, "MacroCatBoostDirect", prohibit_constructor)
    with pytest.raises(ValueError, match="Saved M0 no longer matches"):
        experiment.run_experiment(tmp_path, config_path, ["2024-03"], "synthetic failed h1 check")
    assert not (tmp_path / cfg["output_dir"]).exists()


@pytest.mark.parametrize("complete_run", [False, True])
def test_synthetic_one_March_orchestration_reuses_M0_preserves_lineage_and_verifies_resume(tmp_path, monkeypatch, complete_run):
    cfg = configuration()
    panel = pd.DataFrame({"1": np.arange(24.) + 100, "2": np.arange(24.) + 200},
                         index=pd.date_range("2023-01", periods=24, freq="MS"))
    panel.columns.name = "municipality_id"
    expected = expected_cases(panel, cfg, {"1", "2"})
    cases = expected.loc[expected.forecast_origin.eq("2024-03-31")].copy()
    reference = pd.concat([cases.assign(model=name, status="native", y_pred=cases.y_true)
                           for name in cfg["comparison"]["models"]], ignore_index=True)
    saved_m0 = cases.assign(model="saved_E02", status="native", effective_model="CatBoostDirect",
                           reason="", y_pred=cases.y_true)
    old_training = []
    for horizon in cases.horizon.unique():
        X, delta, trace = build_direct_training(panel, "2024-03", int(horizon), release_lag_months=0, mode="legacy")
        old_training.append({"forecast_origin": "2024-03-31", "horizon": int(horizon),
            "n_training_rows": len(trace), "n_training_municipalities": int(trace.municipality_id.nunique()),
            "n_historical_origins": int(trace.historical_origin.nunique()), "n_target_periods": int(trace.target_period.nunique()),
            "fit_called": True, "fit_succeeded": True})
    macro = macro_table()
    calls = []

    class SyntheticPredictor:
        def __init__(self, parameters, seed, variant, macrotable, **kwargs):
            assert variant in {"M1", "M2"}  # M0 is copied, never constructed.
            self.variant = variant
        def predict(self, data, origin, horizon, ids, baseline_config):
            calls.append((self.variant, str(origin), horizon))
            X, delta, trace = build_direct_training(data, origin, horizon, release_lag_months=0, mode="legacy")
            query = trace[["municipality_id", "historical_origin", "target_period"]].rename(columns={"historical_origin": "as_of_date"})
            query["sample_id"] = [str(position) for position in range(len(query))]
            added, training_lineage = prepare_forecast_features(query, macro, indicators=VARIANT_INDICATORS[self.variant])
            augmented = pd.concat([X, added], axis=1)
            training = dict(next(row for row in old_training if row["horizon"] == horizon),
                **training_signature(trace, X, delta),
                training_ordered_key_sha256=experiment.ordered_pair_signature(trace),
                model_variant=self.variant, variant=self.variant, month_encoding="K0",
                feature_names=augmented.columns.tolist(), feature_dtypes={c: str(t) for c, t in augmented.dtypes.items()},
                fit_called=False, fit_succeeded=False, status="native")
            future_queries = pd.DataFrame({"municipality_id": ids, "as_of_date": pd.Period(origin, freq="M").end_time.normalize(),
                                           "target_period": (origin + horizon).to_timestamp()})
            added_future, forecast_lineage = prepare_forecast_features(future_queries, macro, indicators=VARIANT_INDICATORS[self.variant])
            forecast = pd.DataFrame({"municipality_id": ids, "y_pred": np.zeros(len(ids)), "status": "native",
                "effective_model": "CatBoostDirect" + self.variant, "reason": "", "macro_features_available": True,
                "inflation_available": True, "consumption_available": self.variant == "M2"})
            for column in added_future:
                forecast[column] = added_future[column].to_numpy()
            forecast_lineage["variant"] = self.variant
            training_lineage["variant"] = self.variant
            return MacroDirectResult(forecast, training, pd.DataFrame(columns=["feature", "importance"]), [],
                                     training_lineage, forecast_lineage, [])

    data_path = tmp_path / cfg["data"]["path"]
    data_path.parent.mkdir(parents=True)
    data_path.write_text("synthetic data marker", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    for key in (cfg["validation_record"], cfg["macro"]["source_audit_record"]):
        path = tmp_path / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic successful marker", encoding="utf-8")
    monkeypatch.setattr(experiment, "load_data", lambda *args: panel)
    monkeypatch.setattr(experiment, "make_panel", lambda data: data)
    monkeypatch.setattr(experiment, "require_checks", lambda *args: ({"exit_code": 0, "full_pytest": True}, {"status": "PASS"}))
    monkeypatch.setattr(experiment, "tested_files", lambda root: {"synthetic_code": "same"})
    monkeypatch.setattr(experiment, "verified_macro_table", lambda *args: (macro, {}, {}))
    monkeypatch.setattr(experiment, "verify_reference", lambda *args: (["1", "2"], reference, expected, {}))
    monkeypatch.setattr(experiment, "verify_saved_direct", lambda *args: (saved_m0, pd.DataFrame(old_training), {}))
    monkeypatch.setattr(experiment, "MacroCatBoostDirect", SyntheticPredictor)
    monkeypatch.setattr(experiment.subprocess, "check_output", lambda args, **kwargs: "synthetic_commit" if args[1] == "rev-parse" else "")
    report_calls = []
    if complete_run:
        monkeypatch.setattr(experiment, "origins_from_config", lambda cfg: [pd.Period("2024-03", freq="M")])
        from sberforecast import macro_forecast_report
        def require_completed_manifest(output, report_path, cfg):
            assert json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))["complete"] is True
            report_calls.append(str(report_path))
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text("synthetic saved-artifact report", encoding="utf-8")
        monkeypatch.setattr(macro_forecast_report, "write_report", require_completed_manifest)
    result = experiment.run_experiment(tmp_path, config_path, ["2024-03"], "synthetic smoke")
    assert result["complete"] is complete_run and result["n_origins_completed"] == 1
    assert result["n_new_fits_called"] == 0 and result["M0_refitted"] is False and len(calls) == 6
    assert {variant for variant, _, _ in calls} == {"M1", "M2"}
    output = tmp_path / "outputs/macro_forecast_v1"
    saved = pd.read_csv(output / "predictions.csv.gz", dtype={"municipality_id": str})
    actual_m0 = saved.loc[saved.model.eq("M0")].sort_values(["municipality_id", "horizon"])
    original_m0 = saved_m0.sort_values(["municipality_id", "horizon"])
    np.testing.assert_array_equal(actual_m0.y_pred.to_numpy(), original_m0.y_pred.to_numpy())
    assert len(list((output / "partitions").glob("training_provenance_*.csv.gz"))) == 6
    lineage = pd.read_csv(output / "forecast_provenance.csv.gz")
    assert len(lineage) == len(cases) * 3  # M1 one indicator, M2 two.
    assert {"source_id", "publication_date", "target_year", "vintage_id"}.issubset(lineage)
    assert len(report_calls) == int(complete_run)
    result = experiment.run_experiment(tmp_path, config_path, ["2024-03"], "synthetic resume")
    assert len(calls) == 6 and result["n_failed"] == 0
    partition = output / "partitions/predictions_2024-03.csv.gz"
    partition.write_bytes(b"changed partition")
    with pytest.raises(ValueError, match="artifact set or SHA changed"):
        experiment.run_experiment(tmp_path, config_path, ["2024-03"], "synthetic corrupt resume")
