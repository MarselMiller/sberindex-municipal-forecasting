"""Offline report guards use synthetic artifacts only; no model is trained."""
import json

import numpy as np
import pandas as pd
import pytest

from sberforecast.metrics import KEY
from sberforecast.macro_forecast_evaluation import MODELS, evaluate_macro
from sberforecast.macro_forecast_features import CONSUMPTION, INFLATION, forecast_feature_names, macro_feature_dictionary
from sberforecast.macro_forecast_report import (
    _dictionary_table, _feature_checks, _metric_table, _read_metrics, write_report,
)


def feature_fixture():
    original = ["lag_1", "lag_2", "lag_3", "lag_6", "lag_12", "last_available",
                "mean_3", "std_3", "count_3", "mean_6", "std_6", "count_6",
                "delta_1", "relative_delta_1", "month", "month_sin", "month_cos", "days_in_month", "time_index"]
    features, dictionary = {}, {}
    base = [{"name": name, "formula": "Synthetic fixture definition", "type": "float64",
             "unit": "fixture", "source": "synthetic", "nan_rule": "Keep NaN"} for name in original]
    training = []
    for variant, indicators in (("M0", ()), ("M1", (INFLATION,)), ("M2", (INFLATION, CONSUMPTION))):
        names = original + (forecast_feature_names(indicators) if indicators else [])
        features[variant] = {"names": names, "dtypes": {name: "float64" for name in names}}
        dictionary[variant] = base + (macro_feature_dictionary(indicators) if indicators else [])
        training.append({"variant": variant, "fit_called": True, "fit_succeeded": True,
                         "fit_called_in_e05c": variant != "M0", "cat_features": [],
                         "fit_actual_cat_feature_indices": [], "fit_actual_feature_names": names})
    return features, training, dictionary


def empty_native_metrics(scope="macro_available"):
    return pd.DataFrame([{
        "split": "holdout", "model": model, "horizon": 12, "n_predictions": 0,
        "n_municipalities": 0, "n_origins": 0, "scope": scope,
        "metric_status": "no_macro_available_native_forecasts" if scope == "macro_available" else "no_native_forecasts",
        "mae_macro": np.nan, "mae_micro": np.nan, "r2_pooled": np.nan,
    } for model in MODELS])


def test_empty_macro_annual_scope_cannot_display_reserve_mae(tmp_path):
    empty_native_metrics().to_csv(tmp_path / "metrics_macro_available.csv", index=False)
    metrics = _read_metrics(tmp_path, "macro_available")
    text = _metric_table(metrics, "holdout", [12])
    assert "нет нативных прогнозов с доступными макропризнаками" in text
    assert "M0 | 12 | 0 | 0 | 0 | — | — | —" in text


@pytest.mark.parametrize("field,value", [("n_predictions", 63), ("mae_macro", 4322.14)])
def test_report_rejects_metrics_or_count_on_empty_macro_scope(tmp_path, field, value):
    metrics = empty_native_metrics()
    metrics.loc[metrics.model.eq("M2"), field] = value
    metrics.to_csv(tmp_path / "metrics_macro_available.csv", index=False)
    with pytest.raises(ValueError, match="Пустой области"):
        _read_metrics(tmp_path, "macro_available")


def test_pooled_r2_is_rendered_with_four_decimals_and_mae_two():
    metrics = empty_native_metrics()
    metrics[["n_predictions", "n_municipalities", "n_origins"]] = [2, 2, 1]
    metrics["metric_status"] = "complete"
    metrics[["mae_macro", "mae_micro", "r2_pooled"]] = [12.3456, 15.6789, 0.987654]
    text = _metric_table(metrics, "holdout", [12])
    assert "12.35 | 15.68 | 0.9877" in text


def test_actual_feature_schemas_and_full_dictionary_are_verified():
    features, training, dictionary = feature_fixture()
    text = _feature_checks(features, training)
    assert "M0 | 19 | 0 | 0" in text and "M1 | 24 | 5 | 1" in text and "M2 | 29 | 10 | 1" in text
    definitions = _dictionary_table(dictionary, features)
    for name in features["M2"]["names"]:
        assert name in definitions
    assert "percentage points" in definitions and "calendar days" in definitions


@pytest.mark.parametrize("damage", ["reorder", "dtype", "category", "refitM0"])
def test_report_rejects_hidden_changes_to_K0_or_refit(damage):
    features, training, _ = feature_fixture()
    if damage == "reorder":
        features["M2"]["names"][0:2] = features["M2"]["names"][1::-1]
    elif damage == "dtype":
        features["M1"]["dtypes"]["days_in_month"] = "int64"
    elif damage == "category":
        training[1]["fit_actual_cat_feature_indices"] = [14]
    else:
        training[0]["fit_called_in_e05c"] = True
    with pytest.raises(ValueError):
        _feature_checks(features, training)


def test_incomplete_dictionary_without_base_features_is_rejected():
    features, _, dictionary = feature_fixture()
    dictionary["M0"] = []
    with pytest.raises(ValueError, match="все фактические признаки"):
        _dictionary_table(dictionary, features)


def artifacts(tmp_path):
    expected = pd.DataFrame([
        ("1", "2023-12-31", "2024-12-01", 12, 20.0, "holdout"),
        ("2", "2023-12-31", "2024-12-01", 12, np.nan, "holdout"),
        ("1", "2024-06-30", "2024-07-01", 1, 10.0, "holdout"),
        ("2", "2024-06-30", "2024-07-01", 1, 30.0, "holdout"),
        ("1", "2024-05-31", "2024-06-01", 1, 8.0, "validation"),
    ], columns=KEY + ["y_true", "split"])
    expected["history_cutoff"] = expected.forecast_origin
    expected["availability_assumption"] = "month_end_plus_0_months"
    frames = []
    for model, error in zip(MODELS, range(1, 7)):
        frame = expected.copy()
        frame["model"] = model
        frame["y_pred"] = frame.y_true.fillna(20) - error
        frame["status"] = "native"
        frame["effective_model"] = model
        frame["reason"] = ""
        frame["macro_features_available"] = model in ("M1", "M2")
        if model in ("M0", "M1", "M2"):
            frame.loc[frame.horizon.eq(12), ["status", "reason", "effective_model"]] = [
                "fallback_no_training_pairs", "no_training_pairs", "SeasonalNaive"]
        frames.append(frame)
    tables = evaluate_macro(pd.concat(frames, ignore_index=True), expected)
    for name, table in tables.items():
        table.to_csv(tmp_path / (name + ".csv"), index=False)
    features, training, dictionary = feature_fixture()
    cfg = {"data": {"release_lag_months": 0}, "models": {"catboost": {"iterations": 300}},
           "backtest": {"horizons": [1, 3, 6, 12], "first_origin": "2023-12", "last_origin": "2024-11",
                        "validation_target_end": "2024-06", "min_history_observations": 12, "max_staleness_months": 1}}
    manifest = {"complete": True, "config": cfg, "tests": {"exit_code": 0, "full_pytest": True,
                "summary": "synthetic fixture: no actual pytest result", "command": "synthetic test command"},
                "run_commands": ["synthetic smoke", "synthetic full"], "git_commit": "synthetic",
                "has_uncommitted_changes": True, "seed": 42, "python": "synthetic", "versions": {}}
    status = {"complete": True, "M0_refitted": False, "source_models_refitted": False, "sources_unchanged": True,
              "n_origins_completed": 12, "n_origins_expected": 12, "n_new_fits_called": 58,
              "n_new_fits_succeeded": 58, "n_failed": 0, "n_training_key_signature_matches": 60}
    source = {"status": "PASS", "errors": [], "checks": {"audited_A_rows": 73,
              "audited_inflation_rows": 45, "audited_consumption_rows": 28, "manifest_artifact_hashes": 27,
              "manifest_code_hashes": 6, "manifest_raw_hashes": 43}}
    diagnostics = []
    for variant, indicator in (("M1", INFLATION), ("M2", CONSUMPTION)):
        for scope in ("training", "forecast"):
            diagnostics.append({"forecast_origin": "2024-06-30", "horizon": 1, "variant": variant,
                "indicator": indicator, "scope": scope, "n_rows": 2, "n_historical_origins": 1,
                "n_available": 2, "available_fraction": 1.0, "n_unique_publications": 1,
                "n_unique_finite_values": 1, "n_dates_with_available_macro": 1,
                "all_missing": False, "constant_finite_value": True})
    for name, value in {"feature_sets": features, "feature_dictionary": dictionary,
                        "training_diagnostics": training, "training_macro_diagnostics": diagnostics,
                        "run_manifest": manifest, "run_status": status, "source_audit": source,
                        "preflight": {"sample_size": 64, "n_raw_keys": 1897, "n_evaluable_keys": 1890}}.items():
        (tmp_path / (name + ".json")).write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return cfg


def test_complete_report_uses_saved_tables_and_explains_source_meaning(tmp_path):
    cfg = artifacts(tmp_path)
    report = tmp_path / "report.md"
    write_report(tmp_path, report, cfg)
    text = report.read_text(encoding="utf-8")
    assert "профессиональных участников" in text and "собственный базовый прогноз" in text
    assert "не является официальным консенсусом" in text
    assert "не распространяется назад" in text and "не делится на 12" in text
    assert "h=12" in text and "относится к SeasonalNaive" in text
    assert "уже просмотрен" in text and "не является новой независимой проверкой" in text
    assert "Региональные ИПЦ и зарплата не подключены" in text
    assert "M1-M0" in text and "M2-M1" in text
    assert "Native без макро" in text and "Дат r" in text and "Публикаций" in text
    assert "synthetic smoke" in text and "synthetic test command" in text


def test_existing_report_is_preserved_without_loading_or_overwriting(tmp_path):
    report = tmp_path / "existing.md"
    report.write_text("existing report", encoding="utf-8")
    with pytest.raises(FileExistsError, match="не перезаписывается"):
        write_report(tmp_path, report, {})
    assert report.read_text(encoding="utf-8") == "existing report"


@pytest.mark.parametrize("damage", ["unfinished", "failed_source", "M0_refitted", "pytest_failed"])
def test_report_requires_finished_verified_run(tmp_path, damage):
    cfg = artifacts(tmp_path)
    name = "source_audit" if damage == "failed_source" else "run_manifest" if damage == "pytest_failed" else "run_status"
    path = tmp_path / (name + ".json")
    record = json.loads(path.read_text(encoding="utf-8"))
    if damage == "unfinished":
        record["complete"] = False
    elif damage == "failed_source":
        record["status"] = "FAIL"
        record["errors"] = ["value extraction error"]
    elif damage == "M0_refitted":
        record["M0_refitted"] = True
    else:
        record["tests"]["exit_code"] = 1
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError):
        write_report(tmp_path, tmp_path / "report.md", cfg)
    assert not (tmp_path / "report.md").exists()
