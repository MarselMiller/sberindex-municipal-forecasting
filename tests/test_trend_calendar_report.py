"""Офлайн-проверки, что отчёт не выдаёт резерв за нативную метрику."""
import copy

import numpy as np
import pandas as pd
import pytest

from sberforecast.trend_calendar_report import (
    TREND_MODELS, _feature_checks, _metric_table, _read_metrics,
)


def no_native_row():
    return {"split": "holdout", "model": "SeasonalTrendLinear", "horizon": 12,
            "n_predictions": 0, "n_municipalities": 0, "n_origins": 0,
            "scope": "native", "metric_status": "no_native_forecasts",
            "mae_macro": np.nan, "mae_micro": np.nan, "r2_pooled": np.nan}


def test_report_renders_empty_annual_native_scope_without_fallback_mae(tmp_path):
    pd.DataFrame([no_native_row()]).to_csv(tmp_path / "trends_metrics_native.csv", index=False)
    metrics = _read_metrics(tmp_path, "trends", "native")
    rendered = _metric_table(metrics, TREND_MODELS, "holdout", [12])
    assert "нет нативных прогнозов" in rendered
    assert "SeasonalTrendLinear | 12 | 0 | 0 | 0 | — | — | —" in rendered


@pytest.mark.parametrize("wrong_field,value", [("n_predictions", 63), ("mae_macro", 1234.56)])
def test_report_rejects_metrics_assigned_to_empty_native_scope(tmp_path, wrong_field, value):
    row = no_native_row()
    row[wrong_field] = value
    pd.DataFrame([row]).to_csv(tmp_path / "trends_metrics_native.csv", index=False)
    with pytest.raises(ValueError, match="Пустой нативной"):
        _read_metrics(tmp_path, "trends", "native")


def test_report_rejects_duplicate_metric_groups(tmp_path):
    pd.DataFrame([no_native_row(), no_native_row()]).to_csv(
        tmp_path / "trends_metrics_native.csv", index=False)
    with pytest.raises(ValueError, match="Повторяющиеся"):
        _read_metrics(tmp_path, "trends", "native")


def feature_fixture():
    original = ["lag_1", "lag_2", "lag_3", "lag_6", "lag_12", "last_available",
                "mean_3", "std_3", "count_3", "mean_6", "std_6", "count_6",
                "delta_1", "relative_delta_1", "month", "month_sin", "month_cos",
                "days_in_month", "time_index"]
    features = {"K0": {"names": original, "dtypes": {name: "float64" for name in original}}}
    for variant, removed in (("K1", {"month", "month_sin", "month_cos"}), ("K2", {"month"})):
        names = []
        for name in original:
            if name == "month":
                names.append("month_category")
            elif name not in removed:
                names.append(name)
        features[variant] = {"names": names,
                             "dtypes": {name: "object" if name == "month_category" else "float64"
                                        for name in names}}
    training = [dict(month_encoding="K0", fit_called_in_e05b=False)]
    for variant in ("K1", "K2"):
        training.append(dict(month_encoding=variant, fit_called_in_e05b=True, fit_succeeded=True,
                             cat_features=["month_category"], one_hot_max_size=12,
                             fit_actual_parameters={"one_hot_max_size": 12},
                             fit_actual_cat_feature_indices=[features[variant]["names"].index("month_category")]))
    return features, training


def test_report_verifies_actual_feature_schemas_and_categorical_parameters():
    features, training = feature_fixture()
    rendered = _feature_checks(features, training)
    assert "K0 | 19" in rendered
    assert "K1 | 17" in rendered
    assert "K2 | 19" in rendered


def test_report_rejects_one_hot_not_applied_in_actual_fit():
    features, training = feature_fixture()
    training[1]["fit_actual_parameters"]["one_hot_max_size"] = 2
    with pytest.raises(ValueError, match="one-hot=12"):
        _feature_checks(features, training)


def test_report_rejects_hidden_change_to_other_calendar_features():
    features, training = feature_fixture()
    features = copy.deepcopy(features)
    features["K2"]["dtypes"]["time_index"] = "int64"
    with pytest.raises(ValueError, match="Фактические признаки"):
        _feature_checks(features, training)
