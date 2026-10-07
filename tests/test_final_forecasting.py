"""F1 public synthetic tests: saved-table extraction, with no model fitting."""
import ast
import hashlib
import json
import math
from pathlib import Path
import shutil

import pandas as pd
import pytest

from sberforecast import final_forecasting as module
from sberforecast.final_forecasting import (
    ForecastingAuditError, HORIZONS, KEYS, METRICS, MODEL_SPECS, SOURCES,
    assert_same_keys, collect_forecasting, recompute_metrics, verify_report_metrics,
)


def small_predictions():
    return pd.DataFrame(dict(municipality_id=["a", "a", "b"],
        forecast_origin=["2024-06-30", "2024-07-31", "2024-06-30"],
        target_period=["2024-07-01", "2024-08-01", "2024-07-01"], horizon=[1, 1, 1],
        y_true=[0., 4., 10.], y_pred=[1., 5., 14.], status=["native"]*3))


def test_mae_macro_and_micro_different_weights_and_pooled_r2_independent_formula():
    result = recompute_metrics(small_predictions())
    assert result["mae_macro"] == 2.5  # each MO gets one equal weight: (1+4)/2
    assert result["mae_micro"] == 2.  # (1+1+4)/3
    mean = 14/3
    assert result["r2_pooled"] == pytest.approx(1-18/sum((value-mean)**2 for value in [0, 4, 10]))
    assert result["n_predictions"] == 3 and result["n_municipalities"] == 2 and result["n_origins"] == 2
    assert result["data_status"] == "real" and result["native_count"] == 3


def test_annual_fallback_is_measured_strategy_never_trained_direct_accuracy():
    source = small_predictions().assign(status="fallback_no_training_pairs")
    result = recompute_metrics(source)
    assert result["metric_status"] == "fallback_only" and result["data_status"] == "real"
    assert result["fallback_count"] == 3 and result["native_count"] == 0
    assert result["mae_macro"] == 2.5


@pytest.mark.parametrize("status", ["failed_fit", "unavailable_no_training_pairs"])
def test_failed_or_unavailable_rows_cannot_disappear_from_complete_metric(status):
    source = small_predictions()
    source.loc[0, "status"] = status
    source.loc[0, "y_pred"] = math.nan
    result = recompute_metrics(source)
    assert result["n_predictions"] == 3 and result["failed_count"] == 1
    assert result["mae_macro"] is None and result["mae_micro"] is None and result["r2_pooled"] is None
    assert result["metric_status"] == "incomplete_failed_or_unavailable"
    assert result["data_status"] == "not_evaluated"


def test_missing_native_prediction_cannot_improve_mae_by_finite_row_filter():
    source = small_predictions()
    source.loc[2, "y_pred"] = math.nan
    result = recompute_metrics(source)
    assert result["n_predictions"] == 3 and result["mae_macro"] is None


def test_empty_validation_annual_has_unknown_metrics_not_artificial_zeros():
    result = recompute_metrics(small_predictions().iloc[:0])
    assert result["data_status"] == "not_evaluated" and result["n_predictions"] == 0
    assert all(result[name] is None for name in ("mae_macro", "mae_micro", "r2_pooled"))


@pytest.mark.parametrize("size", [1, 3])
def test_constant_or_one_point_truth_keeps_r2_undefined(size):
    source = small_predictions().iloc[:size].assign(y_true=5.)
    assert recompute_metrics(source)["r2_pooled"] is None


def test_equal_counts_different_case_keys_are_rejected():
    first = small_predictions()
    changed = first.copy()
    changed.loc[0, "municipality_id"] = "other"
    with pytest.raises(ForecastingAuditError, match="exact comparison keys"):
        assert_same_keys(first, changed)
    assert_same_keys(first, first.iloc[::-1])


def test_duplicate_keys_are_rejected_before_any_average():
    first = small_predictions()
    with pytest.raises(ForecastingAuditError, match="duplicate"):
        assert_same_keys(first, pd.concat([first, first.iloc[:1]]))


@pytest.mark.parametrize("problem", ["missing_truth", "unknown_status"])
def test_invalid_evaluation_population_rejected(problem):
    source = small_predictions()
    if problem == "missing_truth":
        source.loc[0, "y_true"] = math.nan
    else:
        source.loc[0, "status"] = "pretend_native"
    with pytest.raises(ForecastingAuditError):
        recompute_metrics(source)


def _manual_metrics(frame):
    # Independent synthetic fixture arithmetic, not the extraction API.
    if frame.empty:
        return dict(zip(METRICS, [math.nan, math.nan, math.nan, 0, 0, 0]))
    errors = (frame.y_true-frame.y_pred).abs()
    macro = sum(errors.loc[frame.municipality_id.eq(uid)].mean() for uid in frame.municipality_id.unique())/frame.municipality_id.nunique()
    mean = sum(frame.y_true)/len(frame)
    denominator = sum((value-mean)**2 for value in frame.y_true)
    return dict(mae_macro=macro, mae_micro=sum(errors)/len(errors),
        r2_pooled=1-sum((a-b)**2 for a, b in zip(frame.y_true, frame.y_pred))/denominator,
        n_predictions=len(frame), n_municipalities=frame.municipality_id.nunique(), n_origins=frame.forecast_origin.nunique())


def _wide(model_names, metrics, split):
    lines = ["| Модель | h=1 | h=3 | h=6 | h=12 |", "| --- | --- | --- | --- | --- |"]
    for name in model_names:
        values = [metrics[(name, split, horizon)]["mae_macro"] for horizon in HORIZONS]
        lines.append("| "+name+" | "+" | ".join("нет случаев" if pd.isna(value) else f"{value:.2f}" for value in values)+" |")
    return "\n".join(lines)


@pytest.fixture(scope="module")
def frozen_synthetic_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("f1_synthetic_saved_forecasts")
    rows = []
    for origin in pd.period_range("2023-12", "2024-11", freq="M"):
        for horizon in HORIZONS:
            target = origin+horizon
            if target > pd.Period("2024-12", freq="M"):
                continue
            for number in range(63):
                rows.append(dict(municipality_id=str(number), forecast_origin=str(origin.end_time.normalize().date()),
                    target_period=str(target.start_time.date()), horizon=horizon,
                    split="validation" if target <= pd.Period("2024-06", freq="M") else "holdout",
                    y_true=1000.+100*number+3*target.month))
    common = pd.DataFrame(rows)
    missing = []
    for origin, horizons in ((pd.Period("2023-12", freq="M"), HORIZONS), (pd.Period("2024-01", freq="M"), (1, 3, 6))):
        for horizon in horizons:
            target = origin+horizon
            missing.append(dict(municipality_id="1471", forecast_origin=str(origin.end_time.normalize().date()),
                target_period=str(target.start_time.date()), horizon=horizon,
                split="validation" if target <= pd.Period("2024-06", freq="M") else "holdout", y_true=math.nan))
    requested = pd.concat([common, pd.DataFrame(missing)], ignore_index=True)
    offsets = {"SeasonalNaive": 10., "SeasonalNaiveYoY": 2., "ProphetAuto": 6., "ProphetYearly": 5.,
        "CatBoostDirect+SeasonalNaive": 4., "C0": 4., "L0": 5., "CN": 1.5, "LN": 1., "Chronos-2": 8.}
    models = {"E01": ["SeasonalNaive", "SeasonalNaiveYoY", "ProphetAuto", "ProphetYearly"],
        "E02": ["CatBoostDirect+SeasonalNaive"],
        "E05d": ["C0", "L0", "CN", "LN", "SeasonalNaiveYoY", "ProphetAuto", "ProphetYearly"], "E03": ["Chronos-2"]}
    all_metrics = {}
    for experiment, model_names in models.items():
        predictions = []
        metrics = []
        for name in model_names:
            frame = requested.assign(model=name, y_pred=lambda data: data.y_true.fillna(150000.)+offsets[name], status="native")
            if name in ("LN", "CN"):
                frame.loc[frame.split.eq("validation"), "y_pred"] = frame.loc[frame.split.eq("validation"), "y_true"].fillna(150000.)+3.
            if name in ("CatBoostDirect+SeasonalNaive", "C0", "L0", "CN", "LN"):
                annual = frame.horizon.eq(12)
                frame.loc[annual, "y_pred"] = frame.loc[annual, "y_true"].fillna(150000.)+10.
                frame.loc[annual, "status"] = "fallback_no_training_pairs"
            predictions.append(frame)
            for split in ("validation", "holdout"):
                for horizon in HORIZONS:
                    subset = frame.loc[frame.y_true.notna() & frame.split.eq(split) & frame.horizon.eq(horizon)]
                    fields = _manual_metrics(subset)
                    all_metrics[(name, split, horizon)] = fields
                    metrics.append(dict(model=name, split=split, horizon=horizon, **fields,
                        n_native=int(subset.status.eq("native").sum()), n_fallback=int(subset.status.str.startswith("fallback").sum()), n_failed=0))
        directory, metrics_name, report_name = SOURCES[experiment]
        destination = root/directory
        destination.mkdir(parents=True)
        pd.concat(predictions).to_csv(destination/"predictions.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
        pd.DataFrame(metrics).to_csv(destination/metrics_name, index=False)
        report_path = root/report_name
        report_path.parent.mkdir(parents=True, exist_ok=True)
        if experiment == "E01":
            report = "## MAE на holdout\n"+_wide(model_names, all_metrics, "holdout")+"\n## Наблюдения\n"
        elif experiment == "E02":
            report = "## A. Стратегия CatBoostDirect + SeasonalNaive\n### Holdout\n"+_wide(model_names, all_metrics, "holdout")
            report += "\n### Validation\n"+_wide(model_names, all_metrics, "validation")+"\nMAE micro\n## B. Только обученный CatBoostDirect\n"
        elif experiment == "E03":
            report = "## MAE macro на полной области\n### Holdout\n"+_wide(model_names, all_metrics, "holdout")
            report += "\n### Validation\n"+_wide(model_names, all_metrics, "validation")+"\n## Smoke и затраты ресурсов\n"
        else:
            report = "## A. Полная стратегия\n"
            for split in ("validation", "holdout"):
                report += "### "+split.title()+"\n| Модель | h | Случаев | МО | Дат | MAE macro | MAE micro | pooled R² | Статус |\n"
                for horizon in HORIZONS:
                    for name in model_names:
                        fields = all_metrics[(name, split, horizon)]
                        printed = [str(fields[field]) for field in ("n_predictions", "n_municipalities", "n_origins")]
                        printed += ["—" if pd.isna(fields[field]) else f"{fields[field]:.2f}" for field in ("mae_macro", "mae_micro")]
                        printed += ["—" if pd.isna(fields["r2_pooled"]) else f"{fields['r2_pooled']:.4f}"]
                        report += "| "+name+" | "+str(horizon)+" | "+" | ".join(printed)+" | полная группа |\n"
            report += "## B. Общая нативная область\n"
        report_path.write_text(report, encoding="utf-8")
    common[list(KEYS)+["split"]].to_csv(root/SOURCES["E05d"][0]/"strategy_keys.csv", index=False)
    common[list(KEYS)+["split"]].to_csv(root/SOURCES["E03"][0]/"evaluation_keys_full.csv", index=False)
    (root/SOURCES["E01"][0]/"sample_ids.json").write_text(json.dumps([str(number) for number in range(63)]+["1471"]), encoding="utf-8")
    (root/"configs").mkdir()
    (root/"configs/prophet_comparison.yaml").write_text("# НЕ ЗАПУСКАЛОСЬ — старый комментарий\n", encoding="utf-8")
    return root


def test_collect_exact_pilot_status_fallback_all_72_records_and_hash_provenance(frozen_synthetic_root):
    before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in frozen_synthetic_root.rglob("*") if path.is_file()}
    result = collect_forecasting(frozen_synthetic_root)
    assert len(result["records"]) == 72
    assert len({(row["model"], row["split"], row["horizon"]) for row in result["records"]}) == 72
    for record in result["records"]:
        if record["split"] == "validation" and record["horizon"] == 12:
            assert record["data_status"] == "not_evaluated" and record["mae_macro"] is None and record["n_predictions"] == 0
        else:
            assert record["data_status"] == "real" and record["n_municipalities"] == 63
        assert "fallback retained" in record["row_filter"]
        assert record["source_sha256"] == hashlib.sha256((frozen_synthetic_root/record["source_path"]).read_bytes()).hexdigest()
    annual_direct = [row for row in result["records"] if row["split"] == "holdout" and row["horizon"] == 12 and row["model"] in
        {"CatBoostDirect", "LightGBMDirect", "National/Local + CatBoost", "National/Local + LightGBM"}]
    assert len(annual_direct) == 4 and all(row["fallback_count"] == 63 and row["native_count"] == 0 and row["metric_status"] == "fallback_only" for row in annual_direct)
    assert result["facts"]["comparison_area"]["evaluable_keys"] == 1890
    assert result["facts"]["verification"]["new_model_fits"] == 0
    assert result["facts"]["verification"]["writes"] == 0
    assert result["discrepancies"][0]["severity"] == "noncritical_legacy_text"
    json.dumps(result, allow_nan=False)
    after = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in frozen_synthetic_root.rglob("*") if path.is_file()}
    assert before == after


@pytest.mark.parametrize("problem", ["same_count_wrong_key", "wrong_truth", "wrong_saved_metric", "wrong_report", "missing_fact_dropped", "wrong_reference_copy"])
def test_collector_fails_when_any_saved_key_truth_metric_report_or_copy_disagrees(frozen_synthetic_root, tmp_path, problem):
    root = tmp_path/"damaged_synthetic_fixture"
    shutil.copytree(frozen_synthetic_root, root)
    if problem == "wrong_saved_metric":
        path = root/SOURCES["E03"][0]/SOURCES["E03"][1]
        frame = pd.read_csv(path)
        frame.loc[0, "mae_macro"] += 1
        frame.to_csv(path, index=False)
    elif problem == "wrong_report":
        path = root/SOURCES["E01"][2]
        path.write_text(path.read_text(encoding="utf-8").replace("10.00", "11.00", 1), encoding="utf-8")
    else:
        experiment = "E05d" if problem == "wrong_reference_copy" else "E03"
        path = root/SOURCES[experiment][0]/"predictions.csv.gz"
        frame = pd.read_csv(path, dtype={"municipality_id": str})
        if problem == "same_count_wrong_key":
            frame.loc[0, "municipality_id"] = "other"
        elif problem == "wrong_truth":
            frame.loc[0, "y_true"] += 1
        elif problem == "missing_fact_dropped":
            frame = frame.loc[frame.y_true.notna()]
        else:
            selected = frame.index[frame.model.eq("ProphetAuto")][0]
            frame.loc[selected, "y_pred"] += 1
        frame.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})
    with pytest.raises(ForecastingAuditError):
        collect_forecasting(root)


@pytest.mark.parametrize("problem", ["wrong_split", "wrong_calendar_horizon", "wrong_origin_date"])
def test_calendar_and_target_split_cannot_change_silently(problem):
    source = small_predictions().assign(model="fake", split="holdout")
    if problem == "wrong_split":
        source.loc[0, "split"] = "validation"
    elif problem == "wrong_calendar_horizon":
        source.loc[0, "horizon"] = 3
    else:
        source.loc[0, "forecast_origin"] = "2024-06-15"
    with pytest.raises(ForecastingAuditError):
        module._normalise(source, "synthetic")


def test_printed_rounding_allowed_but_wrong_report_precision_and_area_rejected():
    record = dict(model="Synthetic", source_model="Synthetic", split="holdout", horizon=1,
        mae_macro=1.2349, mae_micro=1.2349, r2_pooled=.76543,
        n_predictions=3, n_municipalities=2, n_origins=2)
    report = "## A. Полная стратегия\n### Holdout\n| Synthetic | 1 | 3 | 2 | 2 | 1.23 | 1.23 | 0.7654 | complete |\n## B. Общая нативная область\n"
    assert verify_report_metrics(report, [record], "E05d") == 6
    with pytest.raises(ForecastingAuditError, match="differs"):
        verify_report_metrics(report.replace("1.23", "1.25"), [record], "E05d")
    with pytest.raises(ForecastingAuditError, match="cover"):
        verify_report_metrics(report.replace("Synthetic | 1", "Other | 1"), [record], "E05d")


def test_extractor_imports_no_model_packages_and_calls_no_fit_or_prediction_api():
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    imported = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    imported += [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
    assert not any(any(token in (name or "") for token in ("models", "prophet", "catboost", "lightgbm", "chronos", "sklearn", "requests")) for name in imported)
    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in
        {"fit", "fit_predict", "predict", "predict_proba", "train", "forecast"} for node in ast.walk(tree))
