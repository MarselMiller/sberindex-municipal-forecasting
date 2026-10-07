"""F1: audited extraction of frozen pilot predictions, without model imports.

All metrics use the exact E01/E05d strategy keys, including declared fallback.
No prediction is created, fitted, replaced or silently excluded by this module.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pandas as pd


KEYS = ("municipality_id", "forecast_origin", "target_period", "horizon")
HORIZONS = (1, 3, 6, 12)
SPLITS = ("validation", "holdout")
METRICS = ("mae_macro", "mae_micro", "r2_pooled", "n_predictions", "n_municipalities", "n_origins")
SOURCES = {
    "E01": ("outputs/prophet_comparison_v1", "metrics.csv", "reports/results/E01_prophet_comparison.md"),
    "E02": ("outputs/catboost_direct_v1", "metrics_strategy.csv", "reports/results/E02_catboost_direct.md"),
    "E05d": ("outputs/national_local_lightgbm_v1", "metrics_strategy.csv", "reports/results/E05d_national_local_lightgbm.md"),
    "E03": ("outputs/chronos_zero_shot_v1", "metrics_full.csv", "reports/results/E03_chronos_zero_shot.md"),
}
MODEL_SPECS = (
    ("SeasonalNaive", "E01", "SeasonalNaive"),
    ("SeasonalNaiveYoY", "E01", "SeasonalNaiveYoY"),
    ("ProphetAuto", "E01", "ProphetAuto"),
    ("ProphetYearly", "E01", "ProphetYearly"),
    ("CatBoostDirect", "E02", "CatBoostDirect+SeasonalNaive"),
    ("LightGBMDirect", "E05d", "L0"),
    ("National/Local + CatBoost", "E05d", "CN"),
    ("National/Local + LightGBM", "E05d", "LN"),
    ("Chronos-2", "E03", "Chronos-2"),
)


class ForecastingAuditError(ValueError):
    """A frozen key, observed truth, saved metric or report value disagrees."""


def _finite(values: pd.Series) -> pd.Series:
    return values.notna() & values.abs().lt(math.inf)


def _normalise(frame: pd.DataFrame, source: str, *, predictions=True) -> pd.DataFrame:
    required = set(KEYS) | {"split"}
    if predictions:
        required |= {"y_true", "y_pred", "status", "model"}
    if not required.issubset(frame):
        raise ForecastingAuditError(f"{source}: missing columns {sorted(required-set(frame))}")
    data = frame.copy()
    if data.municipality_id.isna().any():
        raise ForecastingAuditError(f"{source}: missing municipality identity")
    data["municipality_id"] = data.municipality_id.astype(str).str.strip()
    if data.municipality_id.eq("").any():
        raise ForecastingAuditError(f"{source}: empty municipality identity")
    for name in ("forecast_origin", "target_period"):
        dates = pd.to_datetime(data[name], errors="raise")
        if dates.isna().any() or not dates.eq(dates.dt.normalize()).all():
            raise ForecastingAuditError(f"{source}: invalid calendar date in {name}")
        if name == "forecast_origin" and not dates.dt.is_month_end.all():
            raise ForecastingAuditError(f"{source}: origins must be calendar month ends")
        if name == "target_period" and not dates.dt.day.eq(1).all():
            raise ForecastingAuditError(f"{source}: target_period must denote month start")
        data[name] = dates.dt.strftime("%Y-%m-%d")
    horizon = pd.to_numeric(data.horizon, errors="raise")
    if horizon.isna().any() or not horizon.isin(HORIZONS).all():
        raise ForecastingAuditError(f"{source}: invalid horizon")
    data["horizon"] = horizon.astype(int)
    origin = pd.PeriodIndex(data.forecast_origin, freq="M")
    target = pd.PeriodIndex(data.target_period, freq="M")
    if not ((target.asi8-origin.asi8) == data.horizon.to_numpy()).all():
        raise ForecastingAuditError(f"{source}: target month is not own O+h")
    expected_split = data.target_period.le("2024-06-01").map({True: "validation", False: "holdout"})
    if not data.split.eq(expected_split).all():
        raise ForecastingAuditError(f"{source}: split differs from fixed target-month June cutoff")
    if predictions:
        for name in ("y_true", "y_pred"):
            data[name] = pd.to_numeric(data[name], errors="raise")
            if data[name].abs().eq(math.inf).any():
                raise ForecastingAuditError(f"{source}: infinite {name}")
        if data.model.isna().any() or data.status.isna().any():
            raise ForecastingAuditError(f"{source}: missing model/status")
        if data.duplicated(["model", *KEYS]).any():
            raise ForecastingAuditError(f"{source}: duplicate model/forecast key")
    elif data.duplicated(list(KEYS)).any():
        raise ForecastingAuditError(f"{source}: duplicate comparison key")
    return data


def assert_same_keys(first: pd.DataFrame, second: pd.DataFrame, source="comparison") -> None:
    """Counts alone cannot establish equal cases; verify the complete key set."""
    if first.duplicated(list(KEYS)).any() or second.duplicated(list(KEYS)).any():
        raise ForecastingAuditError(f"{source}: duplicate comparison key")
    left = pd.MultiIndex.from_frame(first[list(KEYS)]).sort_values()
    right = pd.MultiIndex.from_frame(second[list(KEYS)]).sort_values()
    if not left.equals(right):
        raise ForecastingAuditError(f"{source}: exact comparison keys differ (left={len(left)}, right={len(right)})")


def recompute_metrics(predictions: pd.DataFrame) -> dict:
    """MAE in original rubles, equal-MO macro, pooled R²; no failed-row drop."""
    count = len(predictions)
    result = dict(mae_macro=None, mae_micro=None, r2_pooled=None, n_predictions=count,
        n_municipalities=int(predictions.municipality_id.nunique()),
        n_origins=int(predictions.forecast_origin.nunique()), fallback_count=0, native_count=0,
        failed_count=0, metric_status="no_evaluable_cases", data_status="not_evaluated")
    if not count:
        return result
    truth = pd.to_numeric(predictions.y_true, errors="raise")
    forecast = pd.to_numeric(predictions.y_pred, errors="raise")
    if not _finite(truth).all():
        raise ForecastingAuditError("Metric population includes a missing/nonfinite target fact")
    statuses = predictions.status.astype(str)
    fallback = statuses.str.startswith("fallback")
    native = statuses.eq("native")
    failed = statuses.str.startswith("failed") | statuses.str.startswith("unavailable")
    if not (fallback | native | failed).all():
        raise ForecastingAuditError("Unknown forecast status in metric population")
    result.update(fallback_count=int(fallback.sum()), native_count=int(native.sum()), failed_count=int(failed.sum()))
    if failed.any() or not _finite(forecast).all():
        result["metric_status"] = "incomplete_failed_or_unavailable"
        return result
    absolute = (truth-forecast).abs()
    macro = absolute.groupby(predictions.municipality_id).mean().mean()
    mean = truth.mean()
    denominator = ((truth-mean)**2).sum()
    pooled = 1-float(((truth-forecast)**2).sum()/denominator) if count >= 2 and denominator > 0 else None
    result.update(mae_macro=float(macro), mae_micro=float(absolute.mean()), r2_pooled=pooled,
        metric_status="fallback_only" if fallback.all() else "complete_with_fallback" if fallback.any() else "complete",
        data_status="real")
    return result


def _close(actual, expected, tolerance=1e-8) -> bool:
    if pd.isna(actual) and (expected is None or pd.isna(expected)):
        return True
    if pd.isna(actual) or expected is None or pd.isna(expected):
        return False
    return math.isclose(float(actual), float(expected), abs_tol=tolerance, rel_tol=1e-12)


def _saved_metric_check(records: list[dict], metrics: pd.DataFrame, source: str) -> int:
    checked = 0
    if metrics.duplicated(["split", "model", "horizon"]).any():
        raise ForecastingAuditError(f"{source}: duplicate saved metric group")
    for record in records:
        selected = metrics.loc[metrics.model.eq(record["source_model"]) & metrics.split.eq(record["split"])
            & metrics.horizon.eq(record["horizon"])]
        if selected.empty and record["n_predictions"] == 0:
            continue
        if len(selected) != 1:
            raise ForecastingAuditError(f"{source}: missing saved metrics for {record['model']}/{record['split']}/h{record['horizon']}")
        row = selected.iloc[0]
        for field in METRICS:
            if not _close(row[field], record[field]):
                raise ForecastingAuditError(f"{source}: saved {field} disagrees for {record['model']}/{record['split']}/h{record['horizon']}")
            checked += 1
        for field, column in (("fallback_count", "n_fallback"), ("native_count", "n_native"), ("failed_count", "n_failed")):
            if column in metrics and not _close(row[column], record[field], 0):
                raise ForecastingAuditError(f"{source}: saved {column} disagrees for {record['model']}/{record['split']}/h{record['horizon']}")
    return checked


def _section(text: str, begin: str, end: str | None = None) -> str:
    if begin not in text:
        raise ForecastingAuditError("Required report section absent: "+begin)
    part = text.split(begin, 1)[1]
    return part.split(end, 1)[0] if end is not None else part


def _report_number(text: str):
    cleaned = text.strip().replace(" ", "").replace(",", ".")
    if cleaned in ("—", "-", "NaN") or "нет" in cleaned.lower():
        return None
    try:
        return float(cleaned)
    except ValueError as error:
        raise ForecastingAuditError("Non-numeric report metric: "+text) from error


def verify_report_metrics(text: str, records: list[dict], experiment: str) -> int:
    """Check original report tables, using their printed rounding precision."""
    lookup = {(row["source_model"], row["split"], row["horizon"]): row for row in records}
    if experiment == "E05d":
        # This block is the full strategy; B has a different, explicitly smaller area.
        block = _section(text, "## A. Полная стратегия", "## B. Общая нативная область")
        split, found, checked = None, set(), 0
        for line in block.splitlines():
            if line.startswith("### "):
                split = "validation" if "validation" in line.lower() else "holdout" if "holdout" in line.lower() else None
            if not line.startswith("|") or split is None:
                continue
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if len(cells) != 9 or not cells[1].isdigit():
                continue
            key = (cells[0], split, int(cells[1]))
            if key not in lookup:
                continue
            if key in found:
                raise ForecastingAuditError("Duplicate full-strategy report row")
            found.add(key)
            record = lookup[key]
            fields = ("n_predictions", "n_municipalities", "n_origins", "mae_macro", "mae_micro", "r2_pooled")
            for field, printed in zip(fields, cells[2:8]):
                precision = 0 if field.startswith("n_") else .005000001 if field.startswith("mae") else .000050001
                if not _close(_report_number(printed), record[field], precision):
                    raise ForecastingAuditError(f"E05d report {key} {field} differs from saved predictions")
                checked += 1
        if found != set(lookup):
            raise ForecastingAuditError("E05d report does not cover all required full-strategy metric rows")
        return checked

    if experiment == "E01":
        sections = [("holdout", _section(text, "## MAE на holdout", "## Наблюдения"))]
    elif experiment == "E02":
        full = _section(text, "## A. Стратегия CatBoostDirect + SeasonalNaive", "## B. Только обученный CatBoostDirect")
        sections = [("holdout", _section(full, "### Holdout", "### Validation")),
                    ("validation", _section(full, "### Validation", "MAE micro"))]
    elif experiment == "E03":
        full = _section(text, "## MAE macro на полной области", "## Smoke и затраты ресурсов")
        sections = [("holdout", _section(full, "### Holdout", "### Validation")),
                    ("validation", _section(full, "### Validation"))]
    else:
        raise ForecastingAuditError("Unsupported report experiment: "+experiment)
    found, checked = set(), 0
    for split, section in sections:
        for line in section.splitlines():
            if not line.startswith("|"):
                continue
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if len(cells) != 5:
                continue
            for horizon, printed in zip(HORIZONS, cells[1:]):
                key = (cells[0], split, horizon)
                if key not in lookup:
                    continue
                if key in found:
                    raise ForecastingAuditError(f"{experiment}: duplicate report metric row")
                found.add(key)
                if not _close(_report_number(printed), lookup[key]["mae_macro"], .005000001):
                    raise ForecastingAuditError(f"{experiment} report {key} MAE differs from saved predictions")
                checked += 1
    expected = set(lookup) if experiment != "E01" else {key for key in lookup if key[1] == "holdout"}
    if found != expected:
        raise ForecastingAuditError(f"{experiment}: required report metric cells absent")
    return checked


def _facts(records: list[dict], common: pd.DataFrame, key_sha: str) -> dict:
    lookup = {(row["model"], row["split"], row["horizon"]): row for row in records}
    coverage = {split: {str(horizon): {name: lookup[("SeasonalNaive", split, horizon)][name]
        for name in ("n_predictions", "n_municipalities", "n_origins")} for horizon in HORIZONS} for split in SPLITS}
    yoy, national, chronos = {}, {"holdout_ranks": {}, "validation_deltas_vs_yoy": {}, "holdout_deltas_vs_yoy": {}}, {}
    for horizon in (1, 3, 6):
        baseline = lookup[("SeasonalNaiveYoY", "holdout", horizon)]["mae_macro"]
        auto = lookup[("ProphetAuto", "holdout", horizon)]["mae_macro"]
        yearly = lookup[("ProphetYearly", "holdout", horizon)]["mae_macro"]
        yoy[str(horizon)] = dict(yoy_mae=baseline, prophet_auto_mae=auto, prophet_yearly_mae=yearly,
                                better_than_both=baseline < min(auto, yearly))
        holdout_models = sorted((row for row in records if row["split"] == "holdout" and row["horizon"] == horizon), key=lambda row: row["mae_macro"])
        national["holdout_ranks"][str(horizon)] = [row["model"] for row in holdout_models]
        for split in SPLITS:
            national[split+"_deltas_vs_yoy"][str(horizon)] = (
                lookup[("National/Local + LightGBM", split, horizon)]["mae_macro"]
                - lookup[("SeasonalNaiveYoY", split, horizon)]["mae_macro"])
        chronos[str(horizon)] = dict(chronos_mae=lookup[("Chronos-2", "holdout", horizon)]["mae_macro"],
            yoy_mae=baseline, chronos_improves_yoy=lookup[("Chronos-2", "holdout", horizon)]["mae_macro"] < baseline)
    annual = common.loc[common.horizon.eq(12)]
    chronos["12"] = dict(chronos_mae=lookup[("Chronos-2", "holdout", 12)]["mae_macro"],
        yoy_mae=lookup[("SeasonalNaiveYoY", "holdout", 12)]["mae_macro"],
        chronos_improves_yoy=lookup[("Chronos-2", "holdout", 12)]["mae_macro"] < lookup[("SeasonalNaiveYoY", "holdout", 12)]["mae_macro"])
    fallback_models = [row["model"] for row in records if row["split"] == "holdout" and row["horizon"] == 12 and row["metric_status"] == "fallback_only"]
    yoy_better = all(row["better_than_both"] for row in yoy.values())
    ln_first = all(order[0] == "National/Local + LightGBM" for order in national["holdout_ranks"].values())
    ln_worse_validation = all(delta > 0 for delta in national["validation_deltas_vs_yoy"].values())
    chronos_worse = not any(row["chronos_improves_yoy"] for row in chronos.values())
    narrative = [
        "SeasonalNaiveYoY — сильный baseline: на pilot holdout h=1/3/6 его MAE меньше обоих проверенных Prophet variants."
        if yoy_better else "SeasonalNaiveYoY не превосходит оба Prophet variants на всех трёх коротких горизонтах.",
        ("National/Local + LightGBM показывает наименьшую MAE среди девяти итоговых стратегий на просмотренном holdout h=1/3/6."
         if ln_first else "National/Local + LightGBM не имеет наименьшую MAE на всех коротких горизонтах holdout.")
        + (" На validation уступает SeasonalNaiveYoY; устойчивый окончательный победитель не установлен."
           if ln_worse_validation else " Результаты validation и holdout рассматриваются отдельно; holdout не выбирает окончательную модель."),
        "Chronos-2 zero-shot протестирован на реальных исторических данных, но не улучшил SeasonalNaiveYoY на h=1/3/6."
        if chronos_worse else "Сопоставление Chronos-2 с SeasonalNaiveYoY различается по горизонту; значения приведены отдельно.",
        "Годовые MAE direct стратегий относятся к fallback SeasonalNaive; единственная дата выпуска не позволяет установить устойчивый ranking.",
    ]
    return dict(comparison_area=dict(requested_pilot_municipalities=64, evaluable_municipalities=int(common.municipality_id.nunique()),
        evaluable_keys=len(common), common_key_sha256=key_sha, name="E01/E05d exact full-strategy keys, original nominal RUB"),
        split_coverage=coverage, yoy_vs_prophet_holdout=yoy, national_local_lightgbm=national, chronos_vs_baselines=chronos,
        annual=dict(origins=sorted(annual.forecast_origin.unique().tolist()), n_origins=int(annual.forecast_origin.nunique()),
            validation_cases=0, direct_fallback_models=fallback_models, native_training_pairs=0),
        limitations=[
            "Pilot: 64 МО запрошены; метрики рассчитаны на63 МО с доступными фактами, не на полной панели.",
            "История целевого показателя ограничена24 месяцами; release_lag_months=0 является допущением, vintages расходов не подтверждены.",
            "Holdout был просмотрен в дальнейших исследованиях; последующие опыты не являются новой полностью независимой проверкой.",
            "На h=12 одна forecast origin и нет годового validation; direct стратегии используют63 fallback SeasonalNaive вместо обученной модели.",
            "pooled R² отражает также различия уровней междуМО и не заменяет проверку временной динамики.",
            "Chronos-2 опубликован позже backtest2023–2024; историческая чистота предобученных весов не доказана.",
        ], narrative=narrative)


def collect_forecasting(root: Path) -> dict:
    """Read frozen artifacts and return72 validated records plus source lineage.

    Critical disagreements raise ForecastingAuditError before any final output
    is written. File writes and model calls are outside this extraction API.
    """
    root = Path(root).resolve()
    provenance, loaded, text_reports = {}, {}, {}

    def register(relative: str, role: str) -> Path:
        path = (root/relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ForecastingAuditError("Missing/unsafe forecasting artifact: "+relative)
        entry = provenance.setdefault(relative, dict(path=relative, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), roles=[]))
        if role not in entry["roles"]:
            entry["roles"].append(role)
        return path

    for experiment, (directory, metrics_name, report_name) in SOURCES.items():
        predictions_path = register(directory+"/predictions.csv.gz", "frozen_predictions")
        metrics_path = register(directory+"/"+metrics_name, "saved_metrics_crosscheck")
        report_path = register(report_name, "printed_report_crosscheck")
        prediction_table = _normalise(pd.read_csv(predictions_path, dtype={"municipality_id": str}), str(predictions_path))
        metric_table = pd.read_csv(metrics_path)
        loaded[experiment] = (prediction_table, metric_table)
        text_reports[experiment] = report_path.read_text(encoding="utf-8")
        provenance[directory+"/predictions.csv.gz"]["source_rows"] = len(prediction_table)
    keys_relative = "outputs/national_local_lightgbm_v1/strategy_keys.csv"
    common = _normalise(pd.read_csv(register(keys_relative, "fixed_common_comparison_keys"), dtype={"municipality_id": str}), keys_relative, predictions=False)
    base = loaded["E01"][0].loc[lambda rows: rows.model.eq("SeasonalNaive")]
    sample_path = "outputs/prophet_comparison_v1/sample_ids.json"
    sample_ids = json.loads(register(sample_path, "frozen_requested_pilot_ids").read_text(encoding="utf-8"))
    if not isinstance(sample_ids, list) or len(sample_ids) != 64 or len(set(map(str, sample_ids))) != 64:
        raise ForecastingAuditError("Frozen requested pilot must contain64 distinct E01 IDs")
    if len(base) != 1897 or set(base.municipality_id) != set(map(str, sample_ids)):
        raise ForecastingAuditError("Requested pilot keys must preserve1897 requests across64 original E01 IDs")
    base_evaluable = base.loc[_finite(base.y_true)]
    assert_same_keys(base_evaluable, common, "E01 vs E05d full-strategy area")
    if len(common) != 1890 or common.municipality_id.nunique() != 63:
        raise ForecastingAuditError("Final forecasting requires the frozen1890 E01 keys across63 evaluable municipalities")
    expected_counts = {"validation": {1: 378, 3: 252, 6: 63, 12: 0}, "holdout": {1: 378, 3: 378, 6: 378, 12: 63}}
    for split, counts in expected_counts.items():
        for horizon, expected in counts.items():
            if len(common.loc[common.split.eq(split) & common.horizon.eq(horizon)]) != expected:
                raise ForecastingAuditError(f"Fixed comparison split/horizon count differs: {split}/h{horizon}")
    chronos_keys_relative = "outputs/chronos_zero_shot_v1/evaluation_keys_full.csv"
    chronos_keys = _normalise(pd.read_csv(register(chronos_keys_relative, "independent_saved_common_area_crosscheck"), dtype={"municipality_id": str}), chronos_keys_relative, predictions=False)
    assert_same_keys(common, chronos_keys, "E05d vs E03 full evaluation area")
    canonical_keys = common.sort_values(list(KEYS))[list(KEYS)+["split"]].to_csv(index=False, lineterminator="\n")
    key_sha = hashlib.sha256(canonical_keys.encode("utf-8")).hexdigest()
    base_truth = base_evaluable.set_index(list(KEYS)).y_true.sort_index()
    records = []
    for display, experiment, source_model in MODEL_SPECS:
        directory, metrics_name, report_name = SOURCES[experiment]
        table = loaded[experiment][0].loc[lambda rows: rows.model.eq(source_model)]
        assert_same_keys(base, table, display+" requested keys including unavailable target facts")
        evaluated = table.loc[_finite(table.y_true)]
        assert_same_keys(common, evaluated, display+" exact evaluable comparison area")
        truth = evaluated.set_index(list(KEYS)).y_true.sort_index()
        if not all(_close(value, expected, 1e-8) for value, expected in zip(truth, base_truth)):
            raise ForecastingAuditError(display+": target facts differ from E01")
        for split in SPLITS:
            for horizon in HORIZONS:
                subset = evaluated.loc[evaluated.split.eq(split) & evaluated.horizon.eq(horizon)]
                record = dict(model=display, split=split, horizon=horizon, **recompute_metrics(subset),
                    source_path=directory+"/predictions.csv.gz", source_sha256=provenance[directory+"/predictions.csv.gz"]["sha256"],
                    source_model=source_model, metrics_path=directory+"/"+metrics_name,
                    metrics_sha256=provenance[directory+"/"+metrics_name]["sha256"],
                    report_path=report_name, report_sha256=provenance[report_name]["sha256"],
                    keys_path=keys_relative, keys_sha256=provenance[keys_relative]["sha256"], common_key_sha256=key_sha,
                    row_filter=f"model={source_model}; finite y_true; exact E01/E05d keys; split={split}; horizon={horizon}; fallback retained",
                    experiment_id=experiment, unit="nominal_RUB")
                records.append(record)
    # E05d copies came from frozen E01/E02, without another fit.
    for copied, experiment, original in (("C0", "E02", "CatBoostDirect+SeasonalNaive"),
            ("SeasonalNaiveYoY", "E01", "SeasonalNaiveYoY"), ("ProphetAuto", "E01", "ProphetAuto"),
            ("ProphetYearly", "E01", "ProphetYearly")):
        copy = loaded["E05d"][0].loc[lambda rows: rows.model.eq(copied)]
        source = loaded[experiment][0].loc[lambda rows: rows.model.eq(original)]
        assert_same_keys(copy, source, "E05d "+copied+" vs original saved source")
        copy, source = (frame.set_index(list(KEYS)).sort_index() for frame in (copy, source))
        for column in ("y_true", "y_pred"):
            if not all(_close(value, expected, 1e-8) for value, expected in zip(copy[column], source[column])):
                raise ForecastingAuditError("E05d "+copied+" differs from saved source in "+column)
        if not copy.status.equals(source.status):
            raise ForecastingAuditError("E05d "+copied+" is not an exact saved status copy")
    checked_metrics, checked_reports = 0, 0
    for experiment in SOURCES:
        own_records = [record for record in records if record["experiment_id"] == experiment]
        checked_metrics += _saved_metric_check(own_records, loaded[experiment][1], SOURCES[experiment][0])
        checked_reports += verify_report_metrics(text_reports[experiment], own_records, experiment)
    # Independently check saved E05d reference copies on the same key set too.
    reference_aliases = {"SeasonalNaiveYoY": "SeasonalNaiveYoY", "ProphetAuto": "ProphetAuto", "ProphetYearly": "ProphetYearly", "CatBoostDirect": "C0"}
    copied_records = [dict(record, source_model=reference_aliases[record["model"]]) for record in records if record["model"] in reference_aliases]
    checked_metrics += _saved_metric_check(copied_records, loaded["E05d"][1], "E05d saved reference copies")
    checked_reports += verify_report_metrics(text_reports["E05d"], copied_records, "E05d")
    facts = _facts(records, common, key_sha)
    facts["verification"] = dict(saved_metric_cells_checked=checked_metrics, printed_report_cells_checked=checked_reports,
                                  target_facts_and_exact_keys_checked=True, new_model_fits=0, writes=0)
    for configuration in ("configs/prophet_comparison.yaml", "configs/catboost_direct.yaml",
                          "configs/national_local_lightgbm.yaml", "configs/chronos_zero_shot.yaml"):
        if (root/configuration).is_file():
            register(configuration, "frozen_forecasting_protocol_config")
    discrepancies = []
    legacy_config = root/"configs/prophet_comparison.yaml"
    if legacy_config.is_file() and "НЕ ЗАПУСКАЛОСЬ" in legacy_config.read_text(encoding="utf-8"):
        register("configs/prophet_comparison.yaml", "legacy_text_status_caveat")
        discrepancies.append(dict(severity="noncritical_legacy_text", source_path="configs/prophet_comparison.yaml",
            description="Комментарий НЕ ЗАПУСКАЛОСЬ устарел: сохранённые E01 прогнозы и итоговый отчёт подтверждают выполненный Prophet pilot; старый файл не изменяется."))
    return dict(records=records, facts=facts, provenance=list(provenance.values()), discrepancies=discrepancies)
