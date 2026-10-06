"""Russian E05c report generated exclusively from saved CSV/JSON artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .macro_forecast_evaluation import MODELS, SCOPES
from .macro_forecast_features import CONSUMPTION, INFLATION, forecast_feature_names
from .trend_calendar_report import _table

METRICS = ("mae_macro", "mae_micro", "r2_pooled")
SCOPE_LABELS = {
    "strategy": "A. Полная стратегия с явным резервом",
    "native": "B. Общая нативная область M0/M1/M2",
    "macro_available": "C. Общая нативная область с доступными инфляцией и потреблением",
}
STATUS_LABELS = {
    "complete": "полная группа", "no_native_forecasts": "нет нативных прогнозов",
    "no_macro_available_native_forecasts": "нет нативных прогнозов с доступными макропризнаками",
    "no_evaluable_cases": "нет случаев с фактом", "incomplete_failed": "неполная группа: failed",
}


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _read_metrics(output: Path, scope: str) -> pd.DataFrame:
    frame = pd.read_csv(output / f"metrics_{scope}.csv")
    required = {"split", "model", "horizon", "n_predictions", "n_municipalities", "n_origins",
                "scope", "metric_status", *METRICS}
    if required - set(frame):
        raise ValueError(f"Неполная схема метрик E05c: {scope}.")
    if frame.duplicated(["split", "model", "horizon"]).any() or not frame.scope.eq(scope).all():
        raise ValueError("Повторяющиеся группы или неверная область метрик E05c.")
    if set(frame.model.unique()) != set(MODELS):
        raise ValueError("Для каждой области нужны все шесть фиксированных моделей.")
    absent = frame.metric_status.isin(["no_native_forecasts", "no_macro_available_native_forecasts", "no_evaluable_cases"])
    if (frame.loc[absent, "n_predictions"].ne(0).any()
            or frame.loc[absent, list(METRICS)].notna().any().any()):
        raise ValueError("Пустой области нельзя приписать метрики резервных прогнозов.")
    failed = frame.metric_status.eq("incomplete_failed")
    if frame.loc[failed, list(METRICS)].notna().any().any():
        raise ValueError("Неполной группе с failed нельзя приписать полную метрику.")
    complete = frame.metric_status.eq("complete")
    if (frame.loc[complete, "n_predictions"].le(0).any()
            or not np.isfinite(frame.loc[complete, ["mae_macro", "mae_micro"]]).all().all()):
        raise ValueError("Полной группе нужны конечные MAE и непустая область.")
    if frame.groupby(["split", "horizon"]).n_predictions.nunique().gt(1).any():
        raise ValueError("Модели в таблице отчёта имеют разные прогнозные области.")
    return frame


def _metric_table(metrics: pd.DataFrame, split: str, horizons: list[int]) -> str:
    records = []
    for horizon in horizons:
        for model in MODELS:
            rows = metrics.loc[metrics.split.eq(split) & metrics.model.eq(model) & metrics.horizon.eq(horizon)]
            record = {"Модель": model, "h": horizon, "Случаев": 0, "МО": 0, "Дат": 0,
                      "MAE macro": "—", "MAE micro": "—", "pooled R²": "—", "Статус": "нет случаев с фактом"}
            if len(rows):
                row = rows.iloc[0]
                record.update({"Случаев": int(row.n_predictions), "МО": int(row.n_municipalities),
                               "Дат": int(row.n_origins), "Статус": STATUS_LABELS.get(row.metric_status, str(row.metric_status))})
                if row.metric_status == "complete":
                    record["MAE macro"] = f"{row.mae_macro:.2f}"
                    record["MAE micro"] = f"{row.mae_micro:.2f}"
                    record["pooled R²"] = f"{row.r2_pooled:.4f}" if pd.notna(row.r2_pooled) else "—"
            records.append(record)
    return _table(pd.DataFrame(records))


def _feature_checks(features: dict, training: list[dict]) -> str:
    original = features["M0"]["names"]
    original_types = features["M0"]["dtypes"]
    if len(original) != 19 or not {"month", "month_sin", "month_cos", "days_in_month", "time_index"}.issubset(original):
        raise ValueError("M0 должен сохранять исходные 19 признаков K0.")
    records = []
    for variant, indicators in (("M0", ()), ("M1", (INFLATION,)), ("M2", (INFLATION, CONSUMPTION))):
        current = features[variant]
        appended = forecast_feature_names(indicators) if indicators else []
        if (current["names"] != original + appended
                or {name: current["dtypes"][name] for name in original} != original_types
                or any(current["dtypes"][name] != "float64" for name in appended)):
            raise ValueError("Нарушены порядок/типы исходных K0 или фиксированные макропризнаки.")
        own = [row for row in training if row.get("variant", row.get("model")) == variant]
        if variant == "M0" and any(row.get("fit_called_in_e05c") for row in own):
            raise ValueError("M0 нельзя переобучать в E05c.")
        successful = [row for row in own if variant != "M0" and row.get("fit_called") and row.get("fit_succeeded")]
        for row in successful:
            if (row.get("cat_features") != [] or row.get("fit_actual_cat_feature_indices") != []
                    or row.get("fit_actual_feature_names") != current["names"]):
                raise ValueError("В реальном fit не подтверждены числовые признаки и календарь K0.")
        records.append({"Вариант": variant, "Признаков": len(current["names"]),
                        "Добавлено": len(appended), "Новых успешных fit": len(successful),
                        "Календарь": "month + month_sin + month_cos",
                        "Источник M0": "сохранённый E02b" if variant == "M0" else "новый E05c"})
    return _table(pd.DataFrame(records))


def _dictionary_table(dictionary, features: dict) -> str:
    if isinstance(dictionary, dict) and all(model in dictionary for model in ("M0", "M1", "M2")):
        definitions = dictionary["M2"]
        for variant in ("M0", "M1", "M2"):
            if [record["name"] for record in dictionary[variant]] != features[variant]["names"]:
                raise ValueError("Словарь не покрывает все фактические признаки в их порядке.")
    elif isinstance(dictionary, list):
        definitions = dictionary
        if [record["name"] for record in definitions] != features["M2"]["names"]:
            raise ValueError("Словарь должен содержать все 29 признаков M2 в их порядке.")
    else:
        raise ValueError("Неизвестный формат полного словаря признаков.")
    required = {"name", "formula", "type", "unit", "source", "nan_rule"}
    records = []
    for definition in definitions:
        if required - set(definition):
            raise ValueError("Словарь требует формулу, тип, единицы, источник и правило NaN.")
        name = definition["name"]
        membership = ", ".join(variant for variant in ("M0", "M1", "M2") if name in features[variant]["names"])
        records.append({"Признак": name, "Варианты": membership, "Формула": definition["formula"],
                        "Тип": definition["type"], "Единицы": definition["unit"],
                        "Источник": definition["source"], "NaN": definition["nan_rule"]})
    return _table(pd.DataFrame(records))


def _coverage_table(coverage: pd.DataFrame) -> str:
    fields = ["n_requested", "n_native", "n_fallback", "n_failed", "n_evaluable",
              "n_native_evaluable", "n_fallback_evaluable", "n_failed_evaluable", "n_common_native_evaluable"]
    if set(fields + ["model", "horizon"]) - set(coverage):
        raise ValueError("Не хватает полей покрытия native/fallback/failed.")
    summary = coverage.groupby(["model", "horizon"], as_index=False)[fields].sum()
    records = []
    for row in summary.itertuples(index=False):
        if (row.n_native + row.n_fallback + row.n_failed != row.n_requested
                or row.n_native_evaluable + row.n_fallback_evaluable + row.n_failed_evaluable != row.n_evaluable):
            raise ValueError("Статусы покрытия не дают число всех запрошенных случаев.")
        records.append({"Модель": row.model, "h": int(row.horizon), "Запрошено": int(row.n_requested),
                        "Native / fallback / failed": f"{int(row.n_native)} / {int(row.n_fallback)} / {int(row.n_failed)}",
                        "С фактом": int(row.n_evaluable),
                        "С фактом: native / fallback / failed": f"{int(row.n_native_evaluable)} / {int(row.n_fallback_evaluable)} / {int(row.n_failed_evaluable)}",
                        "Общих нативных с фактом": int(row.n_common_native_evaluable)})
    return _table(pd.DataFrame(records))


def _macro_coverage_table(coverage: pd.DataFrame) -> str:
    own = coverage.loc[coverage.macro_applicable.eq(True)]
    fields = ["n_requested", "n_evaluable", "n_macro_available", "n_macro_available_evaluable",
              "n_native_with_macro", "n_native_without_macro", "n_fallback_with_macro", "n_failed_with_macro"]
    summary = own.groupby(["model", "horizon"], as_index=False)[fields].sum()
    records = []
    for row in summary.itertuples(index=False):
        records.append({"Модель": row.model, "h": int(row.horizon),
                        "С макро / запросов": f"{int(row.n_macro_available)} / {int(row.n_requested)}",
                        "С макро и фактом / с фактом": f"{int(row.n_macro_available_evaluable)} / {int(row.n_evaluable)}",
                        "Native с макро": int(row.n_native_with_macro), "Native без макро": int(row.n_native_without_macro),
                        "Fallback с макро": int(row.n_fallback_with_macro), "Failed с макро": int(row.n_failed_with_macro)})
    return _table(pd.DataFrame(records))


def _diagnostics_table(diagnostics: list[dict], scope: str) -> str:
    records = []
    # M2 inflation is the same feature as M1 inflation. Keep its full saved
    # records, but render the two distinct indicators without duplicating it.
    selected = [row for row in diagnostics if row["scope"] == scope and (
        (row["variant"] == "M1" and row["indicator"] == INFLATION)
        or (row["variant"] == "M2" and row["indicator"] == CONSUMPTION))]
    for row in sorted(selected, key=lambda item: (item["forecast_origin"], item["horizon"], item["variant"])):
        fraction = row["available_fraction"]
        records.append({"O": str(row["forecast_origin"])[:7], "h": row["horizon"], "Вариант": row["variant"],
                        "Строк": row["n_rows"], "Дат r": row["n_historical_origins"],
                        "Доступно": row["n_available"], "Доля": f"{fraction:.2%}" if fraction is not None else "—",
                        "Публикаций": row["n_unique_publications"], "Значений": row["n_unique_finite_values"],
                        "Дат с макро": row["n_dates_with_available_macro"],
                        "Всё отсутствует": "да" if row["all_missing"] else "нет",
                        "Конечное значение постоянно": "да" if row["constant_finite_value"] else "нет"})
    return _table(pd.DataFrame(records))


def _delta_table(differences: pd.DataFrame) -> str:
    records = []
    for row in differences.itertuples(index=False):
        complete = row.metric_status == "complete"
        records.append({"Область": row.scope, "Раздел": row.split, "h": row.horizon,
                        "Разность": row.comparison, "Случаев": row.n_predictions,
                        "Δ MAE macro": f"{row.delta_mae_macro:+.2f}" if complete else "—",
                        "Δ MAE micro": f"{row.delta_mae_micro:+.2f}" if complete else "—",
                        "Δ pooled R²": f"{row.delta_r2_pooled:+.4f}" if complete and pd.notna(row.delta_r2_pooled) else "—",
                        "Статус": STATUS_LABELS.get(row.metric_status, row.metric_status)})
    return _table(pd.DataFrame(records))


def write_report(output: Path, report_path: Path, cfg: dict) -> None:
    """Write the completed fixed experiment; never overwrite an existing report."""
    if report_path.exists():
        raise FileExistsError("Существующий отчёт E05c автоматически не перезаписывается.")
    status, manifest = _json(output / "run_status.json"), _json(output / "run_manifest.json")
    if not status["complete"] or not manifest["complete"] or manifest["config"] != cfg:
        raise ValueError("Для отчёта нужен завершённый запуск с той же конфигурацией.")
    if status.get("M0_refitted", status.get("m0_refitted", False)) or status.get("source_models_refitted", False) or not status["sources_unchanged"]:
        raise ValueError("M0/ориентиры нельзя переобучать или менять прежние результаты.")
    tests = manifest["tests"]
    if tests["exit_code"] != 0 or tests.get("full_pytest") is not True:
        raise ValueError("Отчёт требует успешного полного pytest до запуска.")
    source_file = output / "source_audit.json"
    if not source_file.exists():
        source_file = output.resolve().parents[1] / "outputs/e05c_checks/source_audit.json"
    source_audit = _json(source_file)
    if source_audit["status"] != "PASS" or source_audit.get("errors"):
        raise ValueError("При ошибке извлечения источников обучение/отчёт E05c должны остановиться.")
    preflight = _json(output / "preflight.json")
    features, training = _json(output / "feature_sets.json"), _json(output / "training_diagnostics.json")
    diagnostics = _json(output / "training_macro_diagnostics.json")
    dictionary = _json(output / "feature_dictionary.json")
    schema = _feature_checks(features, training)
    dictionary_rendered = _dictionary_table(dictionary, features)
    metrics = {scope: _read_metrics(output, scope) for scope in SCOPES}
    coverage = pd.read_csv(output / "native_coverage_by_model.csv")
    macro_coverage = pd.read_csv(output / "macro_coverage_by_model.csv")
    deltas = pd.read_csv(output / "metric_deltas.csv")
    checks = source_audit["checks"]
    horizons = list(cfg["backtest"]["horizons"])
    lines = ["# E05c: добавочная прогнозная полезность архивных инфляции и потребления", "",
        "M0 — сохранённый CatBoostDirect K0 E02b. M1 добавляет национальный прогноз инфляции A; "
        "M2 добавляет к M1 национальный прогноз потребления A. Последовательность, исходные "
        "гиперпараметры и календарь month + month_sin + month_cos фиксированы до просмотра новых метрик. "
        "Результаты E05b не использовались для выбора календаря; тренды и one-hot здесь не добавлены.", "",
        "## Проверенные источники и доступность", "",
        f"Офлайн-аудит: {source_audit['status']}; проверены все {checks['audited_A_rows']} строки A "
        f"({checks['audited_inflation_rows']} инфляции и {checks['audited_consumption_rows']} потребления), "
        f"{checks['manifest_artifact_hashes']} SHA результатов E05a, {checks['manifest_code_hashes']} SHA кода, "
        f"{checks['manifest_raw_hashes']} SHA raw-файлов. Значения и точные даты публикаций сверены с сохранёнными "
        "параграфами/таблицами; PDF, извлечённый текст и архивный индекс связаны хешами. Ошибок не обнаружено. "
        "Старые результаты не исправлялись, повторных загрузок не было.",
        "Инфляция forecast_inflation_dec_dec_pct — национальный рост ИПЦ, декабрь к декабрю предыдущего "
        "года, %. Это опубликованная округлённая медиана прогнозов профессиональных участников "
        "макроэкономического опроса, который публикует Банк России; это не собственный прогноз ЦБ, "
        "не среднегодовая инфляция и не факт итога года. 5.1 означает рост 5.1%, а не индекс 105.1. "
        "Диапазоны этих медиан в A-комментариях не опубликованы: width=NaN, midpoint_used=0.",
        "Потребление forecast_consumption_growth_annual_pct — собственный базовый прогноз Банка России: "
        "национальный годовой рост реального объёма конечного потребления домашних хозяйств к предыдущему "
        "календарному году, %. Это реальный годовой показатель, а цель СберИндекса — месячные номинальные "
        "расходы по МО. В A опубликованы диапазоны без официального центра. Их середина — наша заранее "
        "фиксированная производная величина; она не является официальным консенсусом или математическим ожиданием. "
        "Сохранённый forecast_central E05a для потребления уже содержит midpoint; central_method сохраняет это различие.",
        "Категория A означает доверие датированному официальному архиву по правилам E05a. Загрузка "
        "и SHA в 2026 году не доказывают наличие независимого исторического снимка содержимого 2023–2024. "
        "Точные календарные дни публикаций известны; r/O трактуются как конец месяца. Более ранняя дата "
        "доступности из названия месяца опроса не предполагается.", "",
        "## Протокол, модель и признаки", "",
        f"Сохранены {preflight['sample_size']} МО, {preflight['n_raw_keys']} точных ключей E01 на модель "
        f"и {preflight['n_evaluable_keys']} случаев с конечным фактом. Ключ: municipality_id + forecast_origin "
        "+ target_period + horizon. МО 1471 не исключено заранее; отсутствующие факты остаются в прогнозах.",
        f"История расходов: 2023-01…2024-12; выпуски {cfg['backtest']['first_origin']}…{cfg['backtest']['last_origin']}; "
        f"h={horizons}; validation — цели до {cfg['backtest']['validation_target_end']} включительно. "
        f"release_lag_months={cfg['data']['release_lag_months']}; допуск МО на выпуске минимум "
        f"{cfg['backtest']['min_history_observations']} фактов, давность не более "
        f"{cfg['backtest']['max_staleness_months']} месяца. Требования к legacy-префиксу обучающего примера прежние.",
        "Обучение глобальное по полной доступной панели; фиксированные 64 МО определяют область оценки. "
        "Исходные 19 признаков, NaN, порядок строк и ключи пар сохранены. Модель обучается на delta = "
        "y[r+h] − last_available(r−L); прогноз в номинальных рублях = max(0, anchor(O−L) + predicted_delta). "
        "Обучающая метка допустима только при r+h+L≤O.",
        "Каждый макропрогноз выбирается для календарного года целевого месяца r+h, по последней публикации "
        "не позже собственной r; текущий прогноз использует собственную O. Последняя публикация на O "
        "не распространяется назад на обучающие примеры. Более поздний пересмотр доступен только после "
        "своей даты. Другой год, факт, B и будущая публикация не подставляются; NaN не удаляет строку.",
        "Для каждого показателя добавлены: годовое значение; ширина опубликованного диапазона high−low "
        "в процентных пунктах; флаг нашей середины; флаг отсутствия значения; возраст публикации в днях "
        "относительно собственной r/O. Значение — официальный центр при его наличии, иначе середина "
        "опубликованного диапазона, иначе NaN. Годовой процент не делится на 12; месячная ценовая "
        "траектория и дефлирование не строятся. Рост потребления и инфляция механически не складываются.", "", schema, "",
        "Исходные параметры CatBoost:", "", "```json", json.dumps(cfg['models']['catboost'], ensure_ascii=False, indent=2), "```", "",
        "Ключи, порядок, delta и исходный X подписаны и сверены между M0/M1/M2. E02b не сохранял "
        "построчные подписи пар: контроль реконструирован по неизменному коду/данным и сверенным старым "
        "счётчикам; прогнозы M0 и ориентиров взяты из сохранённых файлов и не переобучались. "
        "Фактически применённые параметры, схемы и подписи находятся в training_diagnostics.json.",
        "Нет пар — явный прежний SeasonalNaive, status=fallback_no_training_pairs. fit/predict-ошибки — "
        "failed с причиной и NaN, без скрытого резерва. Для h=12 декабря 2023 пары отсутствуют: наличие "
        "макропубликации этого не меняет. Годовая MAE стратегии M0/M1/M2 относится к SeasonalNaive, "
        "а не обученной годовой модели CatBoost.", "",
        "## Покрытие и объём исторической информации", "", _coverage_table(coverage), "",
        "Макропокрытие учитывается отдельно от выполнения модели. M1 требует конечного прогноза "
        "инфляции на целевой год; M2 требует одновременно инфляцию и потребление. Обученная модель "
        "с отсутствующим макрозначением остаётся native, но не является прогнозом с доступной макроинформацией. "
        "Для M0 и ориентиров макропокрытие неприменимо.", "", _macro_coverage_table(macro_coverage), "",
        "### Обучающие примеры", "",
        "Ниже M1 — инфляция, M2 — потребление. Инфляция M2 совпадает с инфляцией M1 и полностью "
        "сохранена в диагностике. Число строк суммируется повторно при разных O/h; оно не является "
        "числом независимых макронаблюдений. Публикации и значения считаются отдельно на каждой задаче.", "",
        _diagnostics_table(diagnostics, "training"), "",
        "Национальные значения общие для МО одной исторической даты. Постоянное значение/полное отсутствие "
        "явно отражены; вариация не создаётся искусственно. Возраст и индикаторы могут меняться отдельно "
        "от значения. Подробные состояния всех столбцов записаны в column_states; ширина диапазона "
        "инфляции полностью NaN. Большая панель не заменяет мало различных дат и макропубликаций.", "",
        "### Текущие прогнозы", "", _diagnostics_table(diagnostics, "forecast"), "",
    ]
    for scope in SCOPES:
        lines.extend([f"## {SCOPE_LABELS[scope]}", "",
            "Метрики всех шести моделей пересчитаны на одинаковых фактических ключах данной области. MAE macro "
            "сначала усредняет ошибки внутри МО, затем между МО; MAE micro — по всем случаям. "
            "MAE в исходных номинальных рублях. Pooled R² объединяет МО и не заменяет оценку временной динамики.", ""])
        if scope == "strategy":
            lines.append("В области остаются все случаи с конечным фактом, включая failed; неполная группа имеет NaN метрик.")
        elif scope == "native":
            lines.append("Требуется native M0/M1/M2 и конечный прогноз каждого ориентира на тех же ключах; резерв исключён явно.")
        else:
            lines.append("Дополнительно M1 и M2 должны иметь все нужные конечные макропрогнозы на собственной дате выпуска и целевой год; контроль M0 пересчитан на этих же ключах.")
        lines.append("")
        for split in ("validation", "holdout"):
            lines.extend([f"### {split}", "", _metric_table(metrics[scope], split, horizons), ""])
    for scope, filename in (("native", "excluded_native_keys.csv"), ("macro_available", "excluded_macro_available_keys.csv")):
        excluded = pd.read_csv(output / filename)
        lines.extend([f"Из области {scope} исключено {len(excluded)} конечных ключей. "
                      f"Построчные ключи и причины: {filename}; исключения остаются в полной стратегии.", ""])
    lines.extend(["## Последовательные разности и исследовательский вывод", "",
        "M1−M0 проверяет добавочную прогнозную пользу инфляции; M2−M1 — потребления после инфляции. "
        "Для MAE отрицательная разность означает уменьшение ошибки; для R² положительная — рост. "
        "Порядок компонентов фиксирован: разности не являются независимыми или причинными эффектами.", "",
        _delta_table(deltas), ""])
    for split in ("validation", "holdout"):
        available = deltas.loc[deltas.scope.eq("strategy") & deltas.split.eq(split)
                               & deltas.metric_status.eq("complete") & deltas.horizon.isin([1, 3, 6])]
        for row in available.itertuples(index=False):
            direction = "уменьшилась" if row.delta_mae_macro < 0 else "увеличилась" if row.delta_mae_macro > 0 else "не изменилась"
            lines.append(f"Полная стратегия, {split}, h={row.horizon}, {row.comparison}: MAE macro {direction}; Δ={row.delta_mae_macro:+.2f} руб.")
    lines.extend(["", "Приведены фиксированные результаты без смены протокола ради выигрыша. "
        "Небольшие различия не объявляются статистически значимыми: такая проверка не выполнена. "
        "Это проверка добавочной прогнозной полезности в данном пилоте, а не причинного влияния "
        "инфляции/потребления и не доказательство устойчивости будущего улучшения.", "",
        "## Ограничения", "",
        "Использованы только национальные прогнозы A. Региональные ИПЦ и зарплата не подключены; "
        "сценарные B, новости, тренды, one-hot и дефлирование в E05c не исследованы. Годовые национальные "
        "величины не являются региональными или месячными. История короткая, различных публикаций и "
        "исторических дат мало; на h=12 только одна дата выпуска и нет нативных CatBoost. Сроки "
        "публикации/пересмотры самих расходов неподтверждены, release_lag_months=0 остаётся допущением. "
        "Holdout уже просмотрен и не является новой независимой проверкой.", "",
        "## Полный фиксированный словарь M0/M1/M2", "", dictionary_rendered, "",
        "## Проверки, команды и воспроизводимость", "",
        f"Полный pytest до обучения: {tests.get('summary', tests.get('result', 'см. validation_record'))}; "
        f"код {tests['exit_code']}.", "", "```powershell", tests.get("command", "см. manifest tests"), "```", "",
        "Фактически выполненные smoke и полный запуск:", "", "```powershell",
        *manifest['run_commands'], "```", "",
        f"Завершено {status['n_origins_completed']}/{status['n_origins_expected']} выпусков; "
        f"новых fit вызвано {status['n_new_fits_called']}, успешно {status['n_new_fits_succeeded']}; "
        f"failed-строк {status['n_failed']}; сопоставлений подготовки M1/M2 с M0 "
        f"{status['n_training_key_signature_matches']}. M0/ориентиры не переобучались.",
        f"Git HEAD: {manifest['git_commit']}; dirty={manifest['has_uncommitted_changes']}; seed={manifest['seed']}; "
        f"Python={manifest['python']}; версий пакетов {len(manifest['versions'])}. Конфигурация, фактические "
        "команды, версии, seed, SHA кода/данных/E01/E02/E05a и результатов сохранены в manifest.",
        "predictions.csv.gz сохраняет факты/прогнозы/status/reason/effective_model и макродоступность; "
        "metrics_* — три области и метрики по МО/датам; *_keys — точные области и причины исключения. "
        "training_diagnostics.json сохраняет числа/подписи обучающих пар и реальные параметры; "
        "training_macro_diagnostics.json — число дат, публикаций, значений и отсутствие/константность. "
        "Построчное происхождение содержит source_id, publication_date, target_year, vintage_id "
        "для обучающих примеров и прогнозов. Окружения, зависимости, прежние результаты, .gitignore/.vscode "
        "не менялись; тюнинг, commit/push отсутствуют. Построчные данные автоматически в Git не добавляются.", "",
        "Следующий конкретный шаг — обсудить результат фиксированного сравнения и отдельный протокол "
        "с более длинной историей и подтверждёнными датами расходов; региональные ИПЦ/зарплата и "
        "сценарий дефлирования требуют отдельной подготовки источников и решения.", ""])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
