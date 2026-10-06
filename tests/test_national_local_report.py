"""E05d offline report guards; all fixtures are synthetic."""
import json

import numpy as np
import pandas as pd
import pytest

from sberforecast.metrics import KEY
from sberforecast.national_local_evaluation import MODELS, NATIVE_MODELS, evaluate_national_local
from sberforecast.national_local_report import _feature_checks, _metric_table, _read_metrics, write_report


def features_fixture():
    names = ["lag_1", "lag_2", "lag_3", "lag_6", "lag_12", "last_available",
             "mean_3", "std_3", "count_3", "mean_6", "std_6", "count_6",
             "delta_1", "relative_delta_1", "month", "month_sin", "month_cos", "days_in_month", "time_index"]
    features, training = {}, []
    for variant in NATIVE_MODELS:
        features[variant] = {"names": list(names), "dtypes": {name: "float64" for name in names}}
        training.append({"variant": variant, "forecast_origin": "2024-06-30", "horizon": 1,
                         "n_training_rows": 20, "n_training_municipalities": 2, "n_historical_origins": 10,
                         "n_target_periods": 10, "fit_called": True, "fit_succeeded": True,
                         "fit_called_in_e05d": variant != "C0", "fit_actual_feature_names": list(names),
                         "cat_features": [], "fit_actual_cat_feature_indices": [],
                         "fit_native_resolved_parameters": {"device_type": "cpu", "use_missing": "1", "zero_as_missing": "0"}})
    return features, training


def empty_native():
    return pd.DataFrame([{"split": "holdout", "model": model, "horizon": 12, "n_predictions": 0,
                          "n_municipalities": 0, "n_origins": 0, "scope": "native", "metric_status": "no_native_forecasts",
                          "mae_macro": np.nan, "mae_micro": np.nan, "r2_pooled": np.nan} for model in MODELS])


def test_empty_annual_native_report_never_labels_reserve_metric(tmp_path):
    empty_native().to_csv(tmp_path / "metrics_native.csv", index=False)
    text = _metric_table(_read_metrics(tmp_path, "native"), "holdout", [12])
    assert "CN | 12 | 0 | 0 | 0 | — | — | — | нет нативных прогнозов" in text
    assert "LN | 12 | 0 | 0 | 0 | — | — | —" in text


@pytest.mark.parametrize("field,value", [("n_predictions", 63), ("mae_micro", 4000)])
def test_empty_native_metrics_cannot_hide_reserve_as_fit(tmp_path, field, value):
    frame = empty_native()
    frame.loc[frame.model.eq("LN"), field] = value
    frame.to_csv(tmp_path / "metrics_native.csv", index=False)
    with pytest.raises(ValueError, match="Пустой области"):
        _read_metrics(tmp_path, "native")


def test_feature_and_parameter_schema_guard():
    features, training = features_fixture()
    text = _feature_checks(features, training)
    assert "C0 | расходы, рубли | CatBoost | 19 | 0" in text
    assert "LN | local ratio | LightGBM | 19 | 1" in text


@pytest.mark.parametrize("damage", ["reorder", "dtype", "category", "refitC0", "gpu", "missing_defaults", "no_params"])
def test_hidden_pipeline_environment_changes_rejected(damage):
    features, training = features_fixture()
    if damage == "reorder":
        features["L0"]["names"][0:2] = features["L0"]["names"][1::-1]
    elif damage == "dtype":
        features["CN"]["dtypes"]["month"] = "int64"
    elif damage == "category":
        training[1]["fit_actual_cat_feature_indices"] = [14]
    elif damage == "refitC0":
        training[0]["fit_called_in_e05d"] = True
    elif damage == "gpu":
        training[1]["fit_native_resolved_parameters"]["device_type"] = "gpu"
    elif damage == "missing_defaults":
        training[1]["fit_native_resolved_parameters"]["zero_as_missing"] = "1"
    else:
        training[1]["fit_native_resolved_parameters"] = {}
    with pytest.raises(ValueError):
        _feature_checks(features, training)


def artifacts(tmp_path):
    expected = pd.DataFrame([
        ("1", "2024-06-30", "2024-07-01", 1, 10.0, "holdout"),
        ("2", "2024-06-30", "2024-07-01", 1, 30.0, "holdout"),
        ("1", "2024-07-31", "2024-08-01", 1, 12.0, "holdout"),
        ("1", "2024-05-31", "2024-06-01", 1, 8.0, "validation"),
        ("1", "2023-12-31", "2024-12-01", 12, 20.0, "holdout"),
    ], columns=KEY + ["y_true", "split"])
    expected["history_cutoff"] = expected.forecast_origin
    expected["availability_assumption"] = "month_end_plus_0_months"
    frames = []
    for index, model in enumerate(MODELS):
        frame = expected.copy()
        frame["model"], frame["status"], frame["effective_model"], frame["reason"] = model, "native", model, ""
        frame["y_pred"] = frame.y_true - (index + 1)
        if model in NATIVE_MODELS:
            frame.loc[frame.horizon.eq(12), ["status", "effective_model", "reason"]] = [
                "fallback_no_training_pairs", "SeasonalNaive", "no_training_pairs"]
        if model == "L0":
            selected = frame.horizon.eq(1) & frame.split.eq("holdout")
            frame.loc[selected, "y_pred"] = frame.loc[selected, "y_true"] - [0, 0, 2]
        frames.append(frame)
    national = expected[["forecast_origin", "target_period", "horizon", "split"]].drop_duplicates()
    national["n_hat"], national["n_actual"], national["national_forecast_status"] = 20., 22., "native"
    for name, table in evaluate_national_local(pd.concat(frames, ignore_index=True), expected, national).items():
        table.to_csv(tmp_path / (name + ".csv"), index=False)
    series = pd.DataFrame([{"period": "2024-06", "median": 20., "n_municipalities_used": 2,
                            "n_municipalities_known": 3, "n_missing": 1, "missing_share": 1 / 3,
                            "min": 10., "p10": 12., "q25": 15., "q75": 25., "p90": 28., "max": 30.}])
    series.to_csv(tmp_path / "national_series.csv", index=False)
    features, training = features_fixture()
    cfg = {"data": {"release_lag_months": 0}, "backtest": {"first_origin": "2023-12", "last_origin": "2024-11",
            "horizons": [1, 3, 6, 12], "validation_target_end": "2024-06"}}
    status = {"complete": True, "C0_refitted": False, "source_models_refitted": False, "sources_unchanged": True,
              "n_origins_completed": 12, "n_origins_expected": 12, "n_new_fits_called": 87,
              "n_new_fits_succeeded": 87, "n_failed": 0, "runtime_seconds": 123.456}
    manifest = {"complete": True, "config": cfg, "versions": {"lightgbm": "4.6.0"}, "git_commit": "synthetic",
                "seed": 42, "has_uncommitted_changes": True, "run_commands": ["synthetic smoke", "synthetic full"],
                "tests": {"full_pytest": True, "exit_code": 0, "summary": "synthetic fixture only", "command": "synthetic pytest"}}
    for name, record in {"run_status": status, "run_manifest": manifest, "feature_sets": features,
                         "training_diagnostics": training, "local_ratio_diagnostics": [{"scope": "synthetic"}],
                         "preflight": {"sample_size": 64, "n_raw_keys": 1897, "n_evaluable_keys": 1890}}.items():
        (tmp_path / (name + ".json")).write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    return cfg


def test_complete_report_uses_saved_tables_and_explains_causality(tmp_path):
    cfg = artifacts(tmp_path)
    report = tmp_path / "report.md"
    write_report(tmp_path, report, cfg)
    text = report.read_text(encoding="utf-8")
    for phrase in ("N_t = median_i", "до O−L", "r+h+L ≤ O", "delta_R", "N_hat_(O+h)",
                   "use_missing=true", "zero_as_missing=false", "native API", "LightGBM 4.6.0",
                   "L0−C0", "LN−CN", "CN−C0", "LN−L0", "только в одном месяце", "123.46 с.",
                   "уже просмотрен", "обычный forecasting model search закрыт", "E06a", "E06b", "E07",
                   "SeasonalNaive на исходных расходах", "не переобучались"):
        assert phrase in text
    assert "synthetic pytest" in text and "synthetic smoke" in text and "synthetic full" in text
    assert "Вывод по замене алгоритма" in text and "Вывод по National/Local" in text


def test_existing_report_is_never_overwritten(tmp_path):
    report = tmp_path / "existing.md"
    report.write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError, match="не перезаписывается"):
        write_report(tmp_path, report, {})
    assert report.read_text(encoding="utf-8") == "existing"


@pytest.mark.parametrize("damage", ["unfinished", "refit", "pytest_failed", "sources_changed", "no_version"])
def test_finished_verified_source_and_test_guards(tmp_path, damage):
    cfg = artifacts(tmp_path)
    path = tmp_path / ("run_manifest.json" if damage in ("pytest_failed", "no_version") else "run_status.json")
    record = json.loads(path.read_text(encoding="utf-8"))
    if damage == "unfinished":
        record["complete"] = False
    elif damage == "refit":
        record["C0_refitted"] = True
    elif damage == "sources_changed":
        record["sources_unchanged"] = False
    elif damage == "pytest_failed":
        record["tests"]["exit_code"] = 1
    else:
        record["versions"] = {}
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError):
        write_report(tmp_path, tmp_path / "report.md", cfg)
    assert not (tmp_path / "report.md").exists()
