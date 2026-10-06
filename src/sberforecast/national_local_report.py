"""Russian E05d report written only from completed saved experiment artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .national_local_evaluation import CONTRASTS, METRICS, MODELS, NATIVE_MODELS, SCOPES
from .trend_calendar_report import _table

STATUS_LABELS = {"complete": "полная группа", "no_native_forecasts": "нет нативных прогнозов",
                 "no_evaluable_cases": "нет случаев с фактом", "incomplete_failed": "неполная группа: failed"}


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _read_metrics(output: Path, scope: str) -> pd.DataFrame:
    frame = pd.read_csv(output / f"metrics_{scope}.csv")
    required = {"split", "model", "horizon", "n_predictions", "n_municipalities", "n_origins",
                "scope", "metric_status", *METRICS}
    if required - set(frame) or set(frame.model.unique()) != set(MODELS):
        raise ValueError("Отчёту нужны все семь моделей и полная схема метрик.")
    if frame.duplicated(["split", "model", "horizon"]).any() or not frame.scope.eq(scope).all():
        raise ValueError("Повторяющиеся группы или неверная область метрик.")
    if set(frame.metric_status) - set(STATUS_LABELS):
        raise ValueError("Неизвестный статус метрики E05d.")
    absent = frame.metric_status.isin(["no_native_forecasts", "no_evaluable_cases"])
    if (frame.loc[absent, "n_predictions"].ne(0).any()
            or frame.loc[absent, list(METRICS)].notna().any().any()):
        raise ValueError("Пустой области нельзя приписывать MAE резервного прогноза.")
    failed = frame.metric_status.eq("incomplete_failed")
    if frame.loc[failed, list(METRICS)].notna().any().any():
        raise ValueError("Группе с failed нельзя приписывать полную метрику.")
    complete = frame.metric_status.eq("complete")
    if (frame.loc[complete, "n_predictions"].le(0).any()
            or not np.isfinite(frame.loc[complete, ["mae_macro", "mae_micro"]]).all().all()):
        raise ValueError("Полным группам нужны конечные MAE и ненулевое число случаев.")
    if frame.groupby(["split", "horizon"]).n_predictions.nunique().gt(1).any():
        raise ValueError("Модели в отчёте должны иметь одинаковые прогнозные области.")
    return frame


def _metric_table(metrics: pd.DataFrame, split: str, horizons: list[int]) -> str:
    records = []
    for horizon in horizons:
        for model in MODELS:
            found = metrics.loc[metrics.split.eq(split) & metrics.horizon.eq(horizon) & metrics.model.eq(model)]
            row = {"Модель": model, "h": horizon, "Случаев": 0, "МО": 0, "Дат": 0,
                   "MAE macro": "—", "MAE micro": "—", "pooled R²": "—", "Статус": "нет случаев с фактом"}
            if len(found):
                current = found.iloc[0]
                row.update({"Случаев": int(current.n_predictions), "МО": int(current.n_municipalities),
                            "Дат": int(current.n_origins), "Статус": STATUS_LABELS[current.metric_status]})
                if current.metric_status == "complete":
                    row.update({"MAE macro": f"{current.mae_macro:.2f}", "MAE micro": f"{current.mae_micro:.2f}",
                                "pooled R²": f"{current.r2_pooled:.4f}" if pd.notna(current.r2_pooled) else "—"})
            records.append(row)
    return _table(pd.DataFrame(records))


def _feature_checks(features: dict, training: list[dict]) -> str:
    original = features["C0"]["names"]
    if len(original) != 19 or not {"month", "month_sin", "month_cos", "days_in_month", "time_index"}.issubset(original):
        raise ValueError("C0 должен сохранять исходные 19 признаков K0.")
    if "territory_id" in original or "municipality_id" in original:
        raise ValueError("Идентификаторы МО не являются признаками.")
    records = []
    for variant in NATIVE_MODELS:
        schema = features[variant]
        if schema["names"] != original or schema["dtypes"] != features["C0"]["dtypes"]:
            raise ValueError("Все четыре варианта должны сохранять структуру и типы исходных 19 признаков.")
        own = [row for row in training if row.get("variant", row.get("model_variant")) == variant]
        if not own:
            raise ValueError("Не сохранены обучающие диагностики одного из вариантов.")
        if variant == "C0" and any(row.get("fit_called_in_e05d") for row in own):
            raise ValueError("C0 нельзя переобучать в E05d.")
        successful = [row for row in own if variant != "C0" and row.get("fit_called") and row.get("fit_succeeded")]
        for row in successful:
            if (row.get("fit_actual_feature_names") != original
                    or row.get("cat_features") != [] or row.get("fit_actual_cat_feature_indices") != []):
                raise ValueError("В fit должны остаться ровно 19 числовых признаков K0.")
            if variant in ("L0", "LN"):
                resolved = row.get("fit_native_resolved_parameters", {})
                if not resolved or str(resolved.get("device_type", "cpu")).lower() != "cpu":
                    raise ValueError("LightGBM требует сохранённых полных resolved parameters и CPU.")
                if str(resolved.get("use_missing", "1")).lower() not in ("1", "true"):
                    raise ValueError("LightGBM должен сохранять стандартную обработку NaN.")
                if str(resolved.get("zero_as_missing", "0")).lower() not in ("0", "false"):
                    raise ValueError("Нули нельзя молча преобразовывать в пропуски LightGBM.")
        records.append({"Вариант": variant, "Цель": "расходы, рубли" if variant in ("C0", "L0") else "local ratio",
                        "Learner": "CatBoost" if variant in ("C0", "CN") else "LightGBM",
                        "Признаков": len(original), "Новых успешных fit": len(successful),
                        "Исходный календарь": "month + sin/cos", "C0": "сохранённый E02b" if variant == "C0" else "—"})
    return _table(pd.DataFrame(records))


def _coverage_table(frame: pd.DataFrame) -> str:
    fields = ["n_requested", "n_native", "n_fallback", "n_failed", "n_evaluable",
              "n_native_evaluable", "n_fallback_evaluable", "n_failed_evaluable", "n_common_native_evaluable"]
    if set(fields + ["model", "horizon"]) - set(frame):
        raise ValueError("Неполная схема native/fallback/failed покрытия.")
    result = frame.groupby(["model", "horizon"], as_index=False)[fields].sum()
    records = []
    for row in result.itertuples(index=False):
        if (row.n_native + row.n_fallback + row.n_failed != row.n_requested
                or row.n_native_evaluable + row.n_fallback_evaluable + row.n_failed_evaluable != row.n_evaluable):
            raise ValueError("Покрытие статусов не равно числу запросов.")
        records.append({"Модель": row.model, "h": row.horizon, "Запрошено": row.n_requested,
                        "Native / fallback / failed": f"{row.n_native} / {row.n_fallback} / {row.n_failed}",
                        "С фактом": row.n_evaluable,
                        "С фактом: native / fallback / failed": f"{row.n_native_evaluable} / {row.n_fallback_evaluable} / {row.n_failed_evaluable}",
                        "Общих нативных с фактом": row.n_common_native_evaluable})
    return _table(pd.DataFrame(records))


def _deltas_table(frame: pd.DataFrame) -> str:
    rows = []
    for current in frame.itertuples(index=False):
        complete = current.metric_status == "complete"
        rows.append({"Область": current.scope, "Раздел": current.split, "h": current.horizon,
                     "Гипотеза": current.hypothesis, "Разность": current.comparison,
                     "Случаев": current.n_predictions,
                     "Δ MAE macro": f"{current.delta_mae_macro:+.2f}" if complete else "—",
                     "Δ MAE micro": f"{current.delta_mae_micro:+.2f}" if complete else "—",
                     "Δ pooled R²": f"{current.delta_r2_pooled:+.4f}" if complete and pd.notna(current.delta_r2_pooled) else "—",
                     "Статус": STATUS_LABELS[current.metric_status]})
    return _table(pd.DataFrame(rows))


def _robustness_table(frame: pd.DataFrame) -> str:
    rows = []
    for row in frame.itertuples(index=False):
        rows.append({"Область": row.scope, "Раздел": row.split, "h": row.horizon, "Разность": row.comparison,
                     "Дат с метрикой / запросов": f"{row.n_origins_evaluable} / {row.n_origins_requested}",
                     "Улучшено / ухудшено / равно": f"{row.n_origins_improved} / {row.n_origins_worsened} / {row.n_origins_tied}",
                     "Медиана Δ MAE micro": f"{row.median_delta_mae_micro:+.2f}" if pd.notna(row.median_delta_mae_micro) else "—",
                     "Главная дата улучшения": str(row.largest_improvement_origin)[:7] if pd.notna(row.largest_improvement_origin) else "—",
                     "Её доля суммы улучшений": f"{row.largest_improvement_share:.2%}" if pd.notna(row.largest_improvement_share) else "—"})
    return _table(pd.DataFrame(rows))


def _national_table(frame: pd.DataFrame) -> str:
    rows = []
    for row in frame.itertuples(index=False):
        rows.append({"Раздел": row.split, "h": row.horizon, "Запросов O/h": row.n_requested_origins,
                     "Дат с фактом": row.n_actual_available, "Дат с ошибкой": row.n_evaluable_origins,
                     "MAE national, руб.": f"{row.mae_national:.2f}" if pd.notna(row.mae_national) else "—",
                     "Статус": row.metric_status})
    return _table(pd.DataFrame(rows))


def _national_series_table(frame: pd.DataFrame) -> str:
    required = {"period", "median", "n_municipalities_used", "n_municipalities_known", "n_missing", "missing_share",
                "min", "p10", "q25", "q75", "p90", "max"}
    if required - set(frame):
        raise ValueError("Не сохранены месячный состав национального ряда, пропуски и распределение.")
    rows = []
    for row in frame.itertuples(index=False):
        valid_missing_share = (0 <= row.missing_share <= 1 if row.n_municipalities_known else pd.isna(row.missing_share))
        if (row.n_municipalities_used + row.n_missing != row.n_municipalities_known
                or row.n_municipalities_used < 0 or not valid_missing_share):
            raise ValueError("Неверная причинная месячная когорта национального ряда.")
        rows.append({"Месяц": row.period, "Конечных МО": row.n_municipalities_used,
                     "МО, известных к месяцу": row.n_municipalities_known,
                     "Пропусков": row.n_missing, "Доля": f"{row.missing_share:.2%}" if pd.notna(row.missing_share) else "—",
                     "Медиана": f"{row.median:.2f}" if pd.notna(row.median) else "—",
                     "min / p10 / q25": f"{row.min:.2f} / {row.p10:.2f} / {row.q25:.2f}",
                     "q75 / p90 / max": f"{row.q75:.2f} / {row.p90:.2f} / {row.max:.2f}"})
    return _table(pd.DataFrame(rows))


def _training_table(training: list[dict]) -> str:
    rows = []
    for row in training:
        variant = row.get("variant", row.get("model_variant"))
        rows.append({"O": str(row["forecast_origin"])[:7], "h": row["horizon"], "Вариант": variant,
                     "Пар": row["n_training_rows"], "МО": row["n_training_municipalities"],
                     "Дат r": row["n_historical_origins"], "Дат целей": row["n_target_periods"],
                     "Исключено из-за ratio": row.get("n_pairs_excluded_by_ratio_validity", 0),
                     "Новый fit": "нет, сохранён E02b" if variant == "C0" else ("да" if row.get("fit_called") else "нет")})
    return _table(pd.DataFrame(rows))


def _hypothesis_description(deltas: pd.DataFrame, hypothesis: str) -> list[str]:
    """Describe fixed native contrasts without choosing a holdout winner."""
    selected = deltas.loc[deltas.scope.eq("native") & deltas.hypothesis.eq(hypothesis)]
    lines = []
    for (comparison, split), group in selected.groupby(["comparison", "split"], sort=True):
        complete = group.loc[group.metric_status.eq("complete")].sort_values("horizon")
        changes = []
        for row in complete.itertuples(index=False):
            direction = "меньше" if row.delta_mae_micro < 0 else "больше" if row.delta_mae_micro > 0 else "равна"
            changes.append(f"h={row.horizon}: MAE micro {direction} (Δ {row.delta_mae_micro:+.2f} руб.)")
        lines.append(f"{comparison}, {split}: " + ("; ".join(changes) if changes else "нет сравнимых native-групп") + ".")
    lines.append("Это описательные результаты фиксированных контрастов. Без новой независимой проверки "
                 "и анализа неопределённости общий эффект не объявляется доказанным.")
    return lines


def write_report(output: Path, report_path: Path, cfg: dict) -> None:
    """Write a completed report without fitting, downloading or overwriting."""
    if report_path.exists():
        raise FileExistsError("Существующий отчёт E05d автоматически не перезаписывается.")
    status, manifest = _json(output / "run_status.json"), _json(output / "run_manifest.json")
    if not status["complete"] or not manifest["complete"] or manifest["config"] != cfg:
        raise ValueError("Отчёт требует завершённого запуска и совпадающей конфигурации.")
    if status.get("C0_refitted", False) or status.get("source_models_refitted", False) or not status["sources_unchanged"]:
        raise ValueError("C0/ориентиры нельзя переобучать или менять старые результаты.")
    tests = manifest["tests"]
    if tests["exit_code"] != 0 or tests.get("full_pytest") is not True:
        raise ValueError("Для отчёта нужен успешный полный pytest до экспериментального fit.")
    features = _json(output / "feature_sets.json")
    training = _json(output / "training_diagnostics.json")
    schema = _feature_checks(features, training)
    ratio_diagnostics = _json(output / "local_ratio_diagnostics.json")
    preflight = _json(output / "preflight.json")
    metrics = {scope: _read_metrics(output, scope) for scope in SCOPES}
    coverage = pd.read_csv(output / "native_coverage_by_model.csv")
    deltas = pd.read_csv(output / "metric_deltas.csv")
    by_origin = pd.read_csv(output / "metric_deltas_by_origin.csv")
    robustness = pd.read_csv(output / "origin_robustness.csv")
    national = pd.read_csv(output / "national_metrics.csv")
    national_series = pd.read_csv(output / "national_series.csv")
    if set(deltas.comparison) != {f"{later}-{earlier}" for earlier, later, _ in CONTRASTS}:
        raise ValueError("Должны быть сохранены все четыре заранее заданные чистые разности.")
    horizons = list(cfg["backtest"]["horizons"])
    version = manifest.get("versions", {}).get("lightgbm", manifest.get("lightgbm_version", "не сохранена"))
    if version == "не сохранена":
        raise ValueError("Не сохранена фактическая версия LightGBM.")
    runtime = status.get("runtime_seconds", status.get("duration_seconds", status.get("total_runtime_seconds")))
    lines = ["# E05d: National/Local decomposition и LightGBM", "",
        "Два заранее согласованных вопроса: меняется ли качество при замене CatBoost на LightGBM; "
        "меняется ли оно при разделении национального уровня и локального отношения. "
        "C0 — сохранённый CatBoostDirect K0 E02b; L0 — LightGBMDirect на исходных расходах; "
        "CN/LN — те же learners на local ratio. C0 и ориентиры не переобучались. "
        "Архитектура и параметры фиксированы до расчёта новых метрик; tuning отсутствует.", "",
        "## Протокол и причинность", "",
        f"Сохранены {preflight['sample_size']} пилотных МО, {preflight['n_raw_keys']} точных ключей E01 "
        f"и {preflight['n_evaluable_keys']} ключей с конечным фактом. Временные границы "
        f"{cfg['backtest']['first_origin']}…{cfg['backtest']['last_origin']}, горизонты {horizons}; "
        f"validation по месяцу цели до {cfg['backtest']['validation_target_end']} включительно. "
        "МО 1471 не исключался заранее. Требования истории выпуска и legacy-примеров различаются; "
        "прямые пары не получают скрытого strict12.",
        "N_t = median_i(y_i,t) по всем конечным значениям всей панели данного календарного месяца; "
        "не только по 64 пилотным МО и не по когорте, отобранной с учётом будущей полноты. "
        "МО, известные к месяцу, — объединение МО с хотя бы одним конечным фактом до этого месяца "
        "включительно; доля пропусков имеет этот причинный знаменатель. Будущие появления/исчезновения "
        "МО не меняют состав старого месяца.",
        "Для O доступен только префикс N до O−L. Национальный прогноз — неизменный SeasonalNaiveYoY: "
        "значение соответствующего месяца прошлого года × медианный наблюдаемый коэффициент YoY "
        "из трёх последних календарных позиций, ограниченный [0.5, 2]. Использован адаптер старого "
        "baseline_predict; правила пропусков и резерва не изменены. Фактический N_(O+h) запрещён "
        "в признаках и восстановлении прогноза; он присутствует только в отдельной диагностике ошибки.",
        "R_i,t = y_i,t / N_t допускается только при конечном N_t > 0. Некорректный знаменатель "
        "не заменяется константой. Для исторического примера r,h признаки берутся до r−L; "
        "цель r+h и её национальный знаменатель допускаются только при r+h+L ≤ O. "
        "Происхождение знаменателей и дата доступности каждой обучающей метки сохранены в "
        "partitions/training_provenance_*.csv.gz. Будущий национальный факт является частью "
        "уже доступной обучающей метки, но не исторического признака.",
        "Все четыре варианта сохраняют исходные 19 имён, порядок, типы и календарь K0. "
        "L0 использует те же расходы X и delta_y = y_(r+h) − last_available_y; восстановление "
        "max(0, anchor_y + predicted_delta_y). CN/LN используют те же формулы на R: "
        "delta_R = R_(r+h) − last_available_R; R_hat = anchor_R + predicted_delta_R; "
        "y_hat = max(0, N_hat_(O+h) × R_hat). Обученные CN/LN используют одинаковые ratio X, "
        "labels, anchor, keys и национальный прогноз. Идентификаторы МО не передаются learner.", "", schema, "",
        "relative_delta_1 сохраняет старую формулу delta_1 / max(abs(lag_2), 1): в CN/LN "
        "единица знаменателя относится к безразмерному ratio, в C0/L0 — к исходной шкале расходов. "
        "Этот порог не перенастраивался. Контраст decomposition включает одновременно "
        "пересчёт лагов/статистик на R, шкалу обучающей метки и отдельный прогноз N; "
        "он не изолирует только качество national forecaster.", "",
        "Макро E05c, one-hot K1/K2, тренд E05b, новости, погода, новые календарные признаки "
        "и дефлирование не добавлены. В календарных пропусках позиции и горизонты сохраняются; "
        "NaN лагов не заполняются будущими наблюдениями. LightGBM использует native API "
        "Dataset/train; sklearn не устанавливался. use_missing=true, zero_as_missing=false; "
        "NaN обрабатываются самим learner. Отличия обработки NaN от CatBoost являются "
        "свойствами алгоритмов, не скрытым изменением выборки.", "",
        "## Национальный ряд: состав, пропуски и распределение", "", _national_series_table(national_series), "",
        "Полный ряд и календарные диагностики сохранены в national_series.csv. "
        f"Сохранено {len(ratio_diagnostics)} записей local-ratio diagnostics; отрицательные, "
        "нулевые, отсутствующие и неконечные знаменатели отражены явно, а не исправлены константой.", "",
        "## Покрытие и резерв", "", _coverage_table(coverage), "",
        "В декабре 2023 для h=12 нет прошлых годовых обучающих пар: искусственный fit не вызывается. "
        "Резерв всех четырёх стратегий — SeasonalNaive на исходных расходах. "
        "Годовая MAE области A относится к резерву и не называется native "
        "CatBoost/LightGBM/NationalLocal. Fit/predict ошибки сохраняются как failed/NaN; "
        "исключением таких строк полная метрика A не улучшается.", "",
        "A содержит все конечные факты E01. B содержит точное пересечение native C0/L0/CN/LN "
        "и конечных прогнозов всех ориентиров; каждый ориентир повторно оценён на тех же ключах. "
        "Причины исключения и индивидуальное native-покрытие сохранены в excluded_native_keys.csv "
        "и native_coverage_by_model.csv. Совпадение проверялось по ключам и фактам, не по числу строк.", ""]
    for scope in SCOPES:
        lines += [f"## {'A. Полная стратегия' if scope == 'strategy' else 'B. Общая нативная область'}", "",
                  "MAE macro — среднее MAE по МО; MAE micro — среднее по случаям. "
                  "Обе MAE в исходных номинальных рублях. pooled R² считается по объединённым случаям, "
                  "не средним R² по МО.", "", "### Validation", "", _metric_table(metrics[scope], "validation", horizons), "",
                  "### Holdout, уже просмотренный", "", _metric_table(metrics[scope], "holdout", horizons), ""]
    lines += ["## Качество национального прогноза отдельно", "", _national_table(national), "",
        "В национальной MAE каждое O/h имеет единичный вес; ошибки не размножаются по МО. "
        "Фактическая медиана будущего месяца используется только для оценки и не передаётся "
        "обученному прогнозу. Это отдельная диагностика, не основная MAE по муниципалитетам.", "",
        "## Чистые разности двух гипотез", "", _deltas_table(deltas), "",
        "Отрицательная ΔMAE означает меньшую ошибку последующего варианта. Algorithm effect: "
        "L0−C0 и LN−CN; decomposition effect: CN−C0 и LN−L0. Каждый контраст рассчитан внутри "
        "одной точной области. ΔR² положительна при увеличении pooled R². "
        "Небольшая разница не названа статистически значимой: статистический тест не выполнялся.", "",
        "## Устойчивость по датам выпуска", "", _robustness_table(robustness), "",
        "Счётчики улучшений/ухудшений относятся к MAE micro отдельных дат и описывают концентрацию. "
        "Доля главной даты — её снижение MAE в сумме снижений по датам; это не доля независимых "
        "экономических событий. Одна дата на h=6 validation и годовой резерв ограничивают выводы.", ""]
    concentrated = robustness.loc[robustness.improvement_only_one_origin.eq(True)]
    if len(concentrated):
        for row in concentrated.itertuples(index=False):
            lines.append(f"Для {row.scope}, {row.split}, h={row.horizon}, {row.comparison} "
                         f"общее улучшение сопровождается улучшением только в одном месяце выпуска: "
                         f"{str(row.largest_improvement_origin)[:7]}; в остальных датах улучшения нет.")
    else:
        lines.append("По сохранённым счётчикам нет контраста с общим улучшением и единственной улучшающей датой; "
                     "это не доказывает статистическую устойчивость или независимость дат.")
    origin_columns = ["scope", "split", "forecast_origin", "horizon", "comparison", "n_predictions",
                      "delta_mae_macro", "delta_mae_micro", "metric_status"]
    lines += ["", "### Разности по каждой дате", "", _table(by_origin[origin_columns]), "",
        "Полные метрики моделей по датам и МО: metrics_strategy_by_origin.csv, "
        "metrics_native_by_origin.csv, metrics_strategy_by_municipality.csv, "
        "metrics_native_by_municipality.csv. Новые варианты и победитель по holdout не выбирались.", "",
        "## Обучающие диагностики", "", _training_table(training), "",
        "Подписи фиксируют keys, порядок, delta labels, X и anchor. Для C0/L0 проверены те же "
        "nominal pairs; для CN/LN — те же ratio pairs и национальные знаменатели. "
        "Количество МО × строк не заменяет число уникальных исторических дат.", "",
        "## Среда, параметры, команды и runtime", "",
        f"LightGBM {version}; CPU only. Фиксированы objective=regression_l1, num_boost_round=300, "
        "learning_rate=0.05, num_leaves=31, seed=42, num_threads=2. deterministic=true и "
        "force_col_wise=true включены для воспроизводимости CPU; verbosity=-1 меняет журнал, "
        "не модель. Остальные параметры — defaults установленной версии; полный native resolved "
        "набор из сохранённого footer находится в training_diagnostics.json, включая missing handling. "
        "CatBoost CN сохраняет исходные параметры E02b без tuning. "
        "[Официальная документация LightGBM 4.6.0](https://lightgbm.readthedocs.io/en/v4.6.0/Parameters.html).",
        f"Полный pytest: {tests.get('summary', 'код 0')}; exit_code={tests['exit_code']}. "
        f"Выпусков {status['n_origins_completed']}/{status['n_origins_expected']}, "
        f"новых fit {status['n_new_fits_succeeded']}/{status['n_new_fits_called']}, "
        f"failed={status['n_failed']}. C0 и ориентиры не переобучались; старые источники совпали по SHA256.",
        f"Runtime выполненных разделов: {float(runtime):.2f} с." if runtime is not None else
        "Runtime разделов сохранён в partitions/timing_*.json; повторные вызовы могут включать проверку и чтение уже готового smoke.",
        f"Git commit: `{manifest.get('git_commit')}`; dirty={manifest.get('has_uncommitted_changes')}; "
        f"seed={manifest.get('seed', 42)}. Версии, команда Python и SHA256 кода/данных/результатов сохранены в manifest.", "",
        "```powershell", tests.get("command", "команда сохранена в manifest"),
        *manifest.get("run_commands", []), "```", "",
        "## Вывод по замене алгоритма", "", *_hypothesis_description(deltas, "algorithm"), "",
        "## Вывод по National/Local decomposition", "", *_hypothesis_description(deltas, "decomposition"), "",
        "## Ограничения и статус исследовательской программы", "",
        f"Расходы имеют L={cfg['data']['release_lag_months']} как прежнее допущение: реальные дни "
        "публикации и история пересмотров СберИндекса не подтверждены. Национальная медиана "
        "не является суммарным оборотом или внешним макроэкономическим индексом; это медиана "
        "наблюдаемых месячных расходов текущего состава МО. Состав может меняться; национальное "
        "масштабирование само по себе не доказывает устранение смещения.",
        "История короткая, ошибки и даты зависимы. Holdout уже просмотрен: E05d не является новой "
        "независимой проверкой. Причинное влияние и статистическая значимость не установлены. "
        "Общий окончательный победитель по этому holdout не объявляется. Native годовая модель "
        "отсутствует; качество её обучения здесь не проверено.",
        "После E05d обычный forecasting model search закрыт. Следующие согласованные этапы: "
        "E06a — PELT + Binary Segmentation; E06b — news/events; E07 — early warning. "
        "Они не помечены выполненными. Обнаружение уже начавшегося изменения и предупреждение "
        "будущего события требуют разных протоколов.", ""]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
