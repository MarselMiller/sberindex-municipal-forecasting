"""Offline E05d orchestration: frozen guards, saved controls, resume and artifacts."""
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from sberforecast import national_local_experiment as experiment, national_local_model
from sberforecast.direct_experiment import expected_cases
from sberforecast.direct_training import build_direct_training
from sberforecast.national_local_model import NationalLocalDirect

ROOT = Path(__file__).resolve().parents[1]


def configuration():
    return yaml.safe_load((ROOT / "configs/national_local_lightgbm.yaml").read_text(encoding="utf-8"))


def write_config(root, cfg):
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return path


def write_json(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content), encoding="utf-8")


@pytest.mark.parametrize("section,field,value", [
    ("lightgbm", "num_leaves", 63), ("lightgbm", "n_estimators", 301),
    ("lightgbm", "objective", "regression_l2"), ("lightgbm", "device_type", "gpu"),
    ("national_local", "national_statistic", "pilot_cohort_median"),
    ("national_local", "future_actual_national_for_prediction", True),
    ("national_local", "train_variants", ["C0", "L0", "CN", "LN"]),
    ("comparison", "report_models", ["ProphetYearly"]),
])
def test_frozen_decomposition_and_learner_specification_rejects_tweaks(section, field, value):
    cfg = configuration()
    cfg[section][field] = value
    with pytest.raises(ValueError):
        experiment.validate_specification(cfg)


@pytest.mark.parametrize("field,value", [("macro_features_enabled", True), ("one_hot_enabled", True),
                                        ("trend_features_enabled", True), ("news_features_enabled", True),
                                        ("new_calendar_features_enabled", True), ("lightgbm_version", "4.5.0"),
                                        ("smoke_origin", "2024-05"),
                                        ("holdout_status", "new_independent_test"),
                                        ("forecasting_model_search_final", False)])
def test_fixed_scope_smoke_holdout_and_version_rejected(field, value):
    cfg = configuration()
    cfg[field] = value
    with pytest.raises(ValueError):
        experiment.validate_specification(cfg)


@pytest.mark.parametrize("record", [
    {"exit_code": 1, "full_pytest": True, "code_sha256": {"current": "hash"}},
    {"exit_code": 0, "full_pytest": False, "code_sha256": {"current": "hash"}},
    {"exit_code": 0, "full_pytest": True, "code_sha256": {"stale": "hash"}},
])
def test_failed_partial_stale_pytest_blocks_before_predictor(tmp_path, monkeypatch, record):
    cfg = configuration()
    write_json(tmp_path / cfg["validation_record"], record)
    monkeypatch.setattr(experiment, "tested_files", lambda root: {"current": "hash"})
    monkeypatch.setattr(experiment, "NationalLocalDirect", lambda *args, **kwargs: pytest.fail("Premature predictor"))
    with pytest.raises(ValueError, match="full pytest"):
        experiment.run_experiment(tmp_path, write_config(tmp_path, cfg), ["2024-03"], "synthetic blocked")


@pytest.mark.parametrize("changed", ["old_package", "unexpected_package", "lightgbm_version", "pip_check"])
def test_changed_environment_blocks_before_predictor(tmp_path, monkeypatch, changed):
    cfg = configuration()
    write_json(tmp_path / cfg["validation_record"], {"exit_code": 0, "full_pytest": True,
                                                   "code_sha256": {"current": "hash"}})
    write_json(tmp_path / cfg["preservation_record"], {"packages": {"numpy": "unchanged"}})
    env = {"pip_check_exit_code": 1 if changed == "pip_check" else 0, "version": "4.6.0"}
    write_json(tmp_path / cfg["environment_record"], env)
    packages = {"numpy": "changed" if changed == "old_package" else "unchanged",
                "lightgbm": "4.5.0" if changed == "lightgbm_version" else "4.6.0"}
    if changed == "unexpected_package":
        packages["unapproved"] = "1.0"
    monkeypatch.setattr(experiment, "tested_files", lambda root: {"current": "hash"})
    monkeypatch.setattr(experiment.importlib.metadata, "distributions", lambda: [
        SimpleNamespace(metadata={"Name": name}, version=value) for name, value in packages.items()])
    monkeypatch.setattr(experiment, "NationalLocalDirect", lambda *args, **kwargs: pytest.fail("Premature predictor"))
    with pytest.raises(ValueError, match="authorized LightGBM installation"):
        experiment.run_experiment(tmp_path, write_config(tmp_path, cfg), ["2024-03"], "synthetic env guard")


@pytest.mark.parametrize("field,value", [("depth", 7), ("iterations", 301)])
def test_original_source_guard_blocks_changed_CatBoost_parameters(tmp_path, field, value):
    cfg = configuration()
    old = copy.deepcopy(cfg)
    source = tmp_path / cfg["comparison"]["source_dir"]
    source.mkdir(parents=True)
    (tmp_path / "configs").mkdir()
    old_yaml = yaml.safe_dump(old)
    (source / "config_resolved.yaml").write_text(old_yaml, encoding="utf-8")
    (tmp_path / "configs/prophet_comparison.yaml").write_text(old_yaml, encoding="utf-8")
    write_json(source / "run_manifest.json", {"config": old})
    cfg["models"]["catboost"][field] = value
    with pytest.raises(ValueError, match="Параметры catboost отличаются"):
        experiment.verify_reference(tmp_path, pd.DataFrame(), cfg)


@pytest.mark.parametrize("failure", ["data_sha", "target", "keys"])
def test_real_saved_source_guard_rejects_data_truth_or_key_changes(tmp_path, failure):
    cfg = configuration()
    old = copy.deepcopy(cfg)
    old["models"]["enabled"] = cfg["comparison"]["models"]
    panel = pd.DataFrame({str(i): np.arange(24.) + 100 + i for i in range(64)},
                         index=pd.date_range("2023-01", periods=24, freq="MS"))
    sample = panel.columns.tolist()
    data_path = tmp_path / cfg["data"]["path"]
    data_path.parent.mkdir(parents=True)
    data_path.write_text("synthetic original data marker", encoding="utf-8")
    source = tmp_path / cfg["comparison"]["source_dir"]
    source.mkdir(parents=True)
    (source / "config_resolved.yaml").write_text(yaml.safe_dump(old), encoding="utf-8")
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/prophet_comparison.yaml").write_text(yaml.safe_dump(old), encoding="utf-8")
    code_parts = []
    for name in ("data.py", "features.py", "models.py", "backtest.py"):
        path = tmp_path / "src/sberforecast" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic source " + name, encoding="utf-8")
        code_parts.append(path.read_bytes())
    write_json(source / "run_manifest.json", {"config": old,
        "data_sha256": "different" if failure == "data_sha" else experiment.sha256_file(data_path),
        "forecast_code_sha256": hashlib.sha256(b"".join(code_parts)).hexdigest(),
        "packages": {"catboost": importlib.metadata.version("catboost")}})
    write_json(source / "sample_ids.json", sample)
    expected = expected_cases(panel, cfg, set(sample))
    reference = pd.concat([expected.assign(model=name, status="native", y_pred=expected.y_true)
                           for name in cfg["comparison"]["models"]], ignore_index=True)
    if failure == "target":
        reference.loc[0, "y_true"] += 1
    if failure == "keys":
        reference = reference.iloc[1:]
    reference.to_csv(source / "predictions.csv.gz", index=False)
    match = {"data_sha": "Данные отличаются", "target": "целевых фактов", "keys": "Несовпадение ключей"}[failure]
    with pytest.raises(ValueError, match=match):
        experiment.verify_reference(tmp_path, panel, cfg)


@pytest.mark.parametrize("change", ["prediction", "failed"])
def test_saved_C0_h1_must_match_before_new_predictor(tmp_path, monkeypatch, change):
    cfg = configuration()
    panel = pd.DataFrame({"1": np.arange(24.) + 100}, index=pd.date_range("2023-01", periods=24, freq="MS"))
    expected = expected_cases(panel, cfg, {"1"})
    cases = expected.loc[expected.forecast_origin.eq("2024-03-31")].copy()
    reference = cases.assign(model="CatBoostRecursive", status="native", y_pred=cases.y_true)
    control = cases.assign(model="saved_C0", status="native", y_pred=cases.y_true)
    h1 = control.horizon.eq(1)
    control.loc[h1, "y_pred"] = control.loc[h1, "y_pred"] + 1 if change == "prediction" else np.nan
    if change == "failed":
        control.loc[h1, "status"] = "failed"
    monkeypatch.setattr(experiment, "require_checks", lambda *args: ({}, {}, {"protected_files": {}}))
    monkeypatch.setattr(experiment, "load_data", lambda *args: panel)
    monkeypatch.setattr(experiment, "make_panel", lambda data: data)
    monkeypatch.setattr(experiment, "verify_reference", lambda *args: (["1"], reference, expected, {}))
    monkeypatch.setattr(experiment, "verify_saved_direct", lambda *args: (control, pd.DataFrame(), {}))
    monkeypatch.setattr(experiment, "NationalLocalDirect", lambda *args, **kwargs: pytest.fail("Premature predictor"))
    with pytest.raises(ValueError, match="Saved C0 does not agree"):
        experiment.run_experiment(tmp_path, write_config(tmp_path, cfg), ["2024-03"], "synthetic h1 guard")
    assert not (tmp_path / cfg["output_dir"]).exists()


class RecordingEstimator:
    def __init__(self, **parameters):
        self.parameters = parameters

    def fit(self, X, y):
        self.feature_names_ = X.columns.tolist()
        self.feature_name_ = X.columns.tolist()
        self.feature_importances_ = np.zeros(len(X.columns))

    def predict(self, X):
        return np.zeros(len(X))

    def get_params(self, deep=True):
        return dict(self.parameters)

    def get_all_params(self):
        return dict(self.parameters)

    def get_cat_feature_indices(self):
        return []


@pytest.mark.parametrize("complete_run", [False, True])
def test_synthetic_March_saved_control_nine_calls_artifacts_resume_and_completed_report(
        tmp_path, monkeypatch, complete_run):
    cfg = configuration()
    panel = pd.DataFrame({"1": np.arange(24.) + 100, "2": np.arange(24.) + 200},
                         index=pd.date_range("2023-01", periods=24, freq="MS"))
    panel.columns.name = "municipality_id"
    expected = expected_cases(panel, cfg, {"1", "2"})
    cases = expected.loc[expected.forecast_origin.eq("2024-03-31")].copy()
    reference = pd.concat([cases.assign(model=name, status="native", y_pred=cases.y_true)
                           for name in cfg["comparison"]["models"]], ignore_index=True)
    saved_c0 = cases.assign(model="saved_C0", status="native", effective_model="CatBoostDirect",
                           reason="", y_pred=cases.y_true)
    old_training = []
    for h in cases.horizon.unique():
        _, _, trace = build_direct_training(panel, "2024-03", int(h), release_lag_months=0, mode="legacy")
        old_training.append({"forecast_origin": "2024-03-31", "horizon": int(h),
            "n_training_rows": len(trace), "n_training_municipalities": int(trace.municipality_id.nunique()),
            "n_historical_origins": int(trace.historical_origin.nunique()),
            "n_target_periods": int(trace.target_period.nunique()), "fit_called": True, "fit_succeeded": True})
    calls = []
    class SyntheticPredictor:
        def __init__(self, parameters, seed, variant, **kwargs):
            assert variant in {"L0", "CN", "LN"}  # No real C0 construction/refit.
            self.variant = variant
            self.wrapped = NationalLocalDirect(parameters, seed, variant, **kwargs)
        def predict(self, data, origin, h, ids, baseline_config):
            calls.append((self.variant, str(origin), h))
            return self.wrapped.predict(data, origin, h, ids, baseline_config)
    monkeypatch.setattr(national_local_model, "CatBoostRegressor", RecordingEstimator)
    monkeypatch.setattr(national_local_model, "_lightgbm_regressor", RecordingEstimator)
    data_path = tmp_path / cfg["data"]["path"]
    data_path.parent.mkdir(parents=True)
    data_path.write_text("synthetic expenses marker", encoding="utf-8")
    for key in ("validation_record", "environment_record", "preservation_record"):
        write_json(tmp_path / cfg[key], {"synthetic_successful_marker": True})
    monkeypatch.setattr(experiment, "require_checks", lambda *args: ({"exit_code": 0, "full_pytest": True},
                                                                    {"version": "4.6.0"}, {"protected_files": {}}))
    monkeypatch.setattr(experiment, "tested_files", lambda *args: {"synthetic_code": "unchanged"})
    monkeypatch.setattr(experiment, "load_data", lambda *args: panel)
    monkeypatch.setattr(experiment, "make_panel", lambda data: data)
    monkeypatch.setattr(experiment, "verify_reference", lambda *args: (["1", "2"], reference, expected, {}))
    monkeypatch.setattr(experiment, "verify_saved_direct", lambda *args: (saved_c0, pd.DataFrame(old_training), {}))
    monkeypatch.setattr(experiment, "NationalLocalDirect", SyntheticPredictor)
    monkeypatch.setattr(experiment.subprocess, "check_output", lambda args, **kwargs: "synthetic_commit" if args[1] == "rev-parse" else "")
    report_calls = []
    if complete_run:
        monkeypatch.setattr(experiment, "origins_from_config", lambda cfg: [pd.Period("2024-03", "M")])
        from sberforecast import national_local_report
        def require_completed_manifest(output, path, cfg):
            assert json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))["complete"] is True
            report_calls.append(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("synthetic report from saved artifacts", encoding="utf-8")
        monkeypatch.setattr(national_local_report, "write_report", require_completed_manifest)
    config_path = write_config(tmp_path, cfg)
    result = experiment.run_experiment(tmp_path, config_path, ["2024-03"], "synthetic March smoke")
    assert result["complete"] is complete_run
    assert result["n_origins_completed"] == 1 and result["n_models"] == 7
    assert result["C0_refitted"] is False and result["n_new_fits_called"] == 9
    assert result["n_new_fits_succeeded"] == 9 and result["n_failed"] == 0
    assert len(calls) == 9 and {v for v, _, _ in calls} == {"L0", "CN", "LN"}
    output = tmp_path / cfg["output_dir"]
    saved = pd.read_csv(output / "predictions.csv.gz", dtype={"municipality_id": str})
    for name, original in [("C0", saved_c0), *[(name, reference.loc[reference.model.eq(name)]) for name in experiment.REFERENCES]]:
        own = saved.loc[saved.model.eq(name)].sort_values(["municipality_id", "horizon"])
        source = original.sort_values(["municipality_id", "horizon"])
        np.testing.assert_array_equal(own.y_pred.to_numpy(), source.y_pred.to_numpy())
    national_predictions = pd.read_csv(output / "national_forecasts.csv")
    assert len(national_predictions) == 3
    assert not national_predictions.duplicated(["forecast_origin", "horizon"]).any()
    assert len(list((output / "partitions").glob("training_provenance_*.csv.gz"))) == 9
    local = pd.read_csv(output / "partitions/training_provenance_2024-03_h3_CN.csv.gz")
    assert {"target_national_denominator", "target_national_available_at", "anchor_national_denominator", "delta"}.issubset(local)
    assert (pd.to_datetime(local.target_national_available_at) <= pd.Timestamp("2024-03-31")).all()
    dictionaries = json.loads((output / "feature_dictionary.json").read_text(encoding="utf-8"))
    sets = json.loads((output / "feature_sets.json").read_text(encoding="utf-8"))
    assert set(sets) == {"C0", "L0", "CN", "LN"}
    assert all(len(v["names"]) == 19 and set(v["dtypes"].values()) == {"float64"} for v in sets.values())
    assert all(len(records) == 19 for records in dictionaries.values())
    assert len(json.loads((output / "resolved_parameters.json").read_text(encoding="utf-8"))) == 9
    ratio_diagnostics = json.loads((output / "local_ratio_diagnostics.json").read_text(encoding="utf-8"))
    assert len(ratio_diagnostics) == 15 * 3 * 2  # 15 known months, 3 horizons, CN and LN.
    assert set(row["variant"] for row in ratio_diagnostics) == {"CN", "LN"}
    assert len(report_calls) == int(complete_run)
    result = experiment.run_experiment(tmp_path, config_path, ["2024-03"], "synthetic verified resume")
    assert len(calls) == 9 and result["n_failed"] == 0
    (output / "partitions/predictions_2024-03.csv.gz").write_bytes(b"changed saved partition")
    with pytest.raises(ValueError, match="saved artifact set changed"):
        experiment.run_experiment(tmp_path, config_path, ["2024-03"], "synthetic corrupt resume")
    assert len(calls) == 9
