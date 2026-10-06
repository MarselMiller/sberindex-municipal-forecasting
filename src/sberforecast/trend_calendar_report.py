"""Русский отчёт E05b только по сохранённым результатам; никаких fit/predict."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


TREND_MODELS = ("SeasonalNaiveYoY", "SeasonalTrendLinear", "SeasonalTrendDamped")
CALENDAR_MODELS = ("CatBoostDirectK0", "CatBoostDirectK1", "CatBoostDirectK2")
METRICS = ("mae_macro", "mae_micro", "r2_pooled")
MONTH_FIELDS = {"month", "month_sin", "month_cos", "month_category"}


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "Нет сохранённых строк."
    lines = ["| " + " | ".join(map(str, frame.columns)) + " |",
             "| " + " | ".join("---" for _ in frame.columns) + " |"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value).replace("|", " / ").replace("\n", " ")
                                       for value in row) + " |")
    return "\n".join(lines)


def _read_metrics(output: Path, group: str, scope: str) -> pd.DataFrame:
    frame = pd.read_csv(output / f"{group}_metrics_{scope}.csv")
    required = {"split", "model", "horizon", "n_predictions", "n_municipalities", "n_origins",
                "scope", "metric_status", *METRICS}
    if required - set(frame.columns):
        raise ValueError(f"Неполная схема метрик отчёта: {group}/{scope}.")
    if frame.duplicated(["split", "model", "horizon"]).any() or not frame.scope.eq(scope).all():
        raise ValueError("Повторяющиеся группы или неправильная область метрик отчёта.")
    empty_native = frame.metric_status.eq("no_native_forecasts")
    if scope == "native" and (frame.loc[empty_native, "n_predictions"].ne(0).any()
                               or frame.loc[empty_native, list(METRICS)].notna().any().any()):
        raise ValueError("Пустой нативной области нельзя приписать метрики резервных прогнозов.")
    complete = frame.metric_status.eq("complete")
    if (frame.loc[complete, "n_predictions"].le(0).any()
            or not np.isfinite(frame.loc[complete, ["mae_macro", "mae_micro"]]).all().all()):
        raise ValueError("Полной группе нужны конечные MAE и непустая область.")
    return frame


def _metric_table(metrics: pd.DataFrame, models: tuple[str, ...], split: str,
                  horizons: list[int]) -> str:
    rows = []
    status_labels = {
        "complete": "полная группа", "no_native_forecasts": "нет нативных прогнозов",
        "no_evaluable_cases": "нет случаев с фактом", "incomplete_failed": "неполная группа: failed",
    }
    for horizon in horizons:
        for model in models:
            selected = metrics.loc[metrics.split.eq(split) & metrics.model.eq(model)
                                   & metrics.horizon.eq(horizon)]
            row = {"Модель": model, "h": horizon, "Случаев": 0, "МО": 0, "Дат": 0,
                   "MAE macro": "—", "MAE micro": "—", "pooled R²": "—",
                   "Статус": "нет случаев с фактом"}
            if len(selected):
                result = selected.iloc[0]
                row.update({"Случаев": int(result.n_predictions), "МО": int(result.n_municipalities),
                            "Дат": int(result.n_origins),
                            "Статус": status_labels.get(result.metric_status, str(result.metric_status))})
                if result.metric_status == "complete":
                    for source, destination in zip(METRICS, ("MAE macro", "MAE micro", "pooled R²")):
                        row[destination] = f"{result[source]:.2f}" if pd.notna(result[source]) else "—"
            rows.append(row)
    return _table(pd.DataFrame(rows))


def _coverage_table(frame: pd.DataFrame, models: tuple[str, ...]) -> str:
    count_fields = ["n_requested", "n_native", "n_fallback", "n_failed", "n_evaluable",
                    "n_native_evaluable", "n_fallback_evaluable", "n_failed_evaluable",
                    "n_common_native_evaluable"]
    required = {"model", "horizon", *count_fields}
    if required - set(frame.columns):
        raise ValueError("Неполная схема покрытия отчёта.")
    aggregated = frame.groupby(["model", "horizon"], as_index=False)[count_fields].sum()
    rows = []
    for model in models:
        for record in aggregated.loc[aggregated.model.eq(model)].sort_values("horizon").itertuples(index=False):
            if (record.n_native + record.n_fallback + record.n_failed != record.n_requested
                    or record.n_native_evaluable + record.n_fallback_evaluable + record.n_failed_evaluable
                    != record.n_evaluable):
                raise ValueError("Статусы не покрывают все строки E05b.")
            rows.append({
                "Модель": model, "h": int(record.horizon), "Запрошено": int(record.n_requested),
                "Native": int(record.n_native), "Fallback": int(record.n_fallback), "Failed": int(record.n_failed),
                "С фактом": int(record.n_evaluable),
                "Native / fallback / failed с фактом":
                    f"{int(record.n_native_evaluable)} / {int(record.n_fallback_evaluable)} / {int(record.n_failed_evaluable)}",
                "Общих нативных": int(record.n_common_native_evaluable),
            })
    return _table(pd.DataFrame(rows))


def _feature_checks(features: dict, training: list[dict]) -> str:
    expected_month = {
        "K0": {"month", "month_sin", "month_cos"},
        "K1": {"month_category"}, "K2": {"month_category", "month_sin", "month_cos"},
    }
    rows = []
    base = [name for name in features["K0"]["names"] if name not in MONTH_FIELDS]
    base_types = {name: features["K0"]["dtypes"][name] for name in base}
    for encoding, n_features in (("K0", 19), ("K1", 17), ("K2", 19)):
        current = features[encoding]
        names, dtypes = current["names"], current["dtypes"]
        if (len(names) != n_features or set(names) & MONTH_FIELDS != expected_month[encoding]
                or [name for name in names if name not in MONTH_FIELDS] != base
                or {name: dtypes[name] for name in base} != base_types):
            raise ValueError("Фактические признаки отличаются от фиксированных K0/K1/K2.")
        if not {"days_in_month", "time_index"}.issubset(base):
            raise ValueError("Прочие календарные признаки должны быть сохранены.")
        if encoding != "K0" and dtypes["month_category"] not in ("object", "string"):
            raise ValueError("Месяц должен быть строковой категорией.")
        own = [item for item in training if item["month_encoding"] == encoding]
        successful = [item for item in own if item.get("fit_called_in_e05b") and item.get("fit_succeeded")]
        if encoding == "K0" and any(item.get("fit_called_in_e05b") for item in own):
            raise ValueError("Исходный K0 нельзя переобучать в E05b.")
        for item in successful:
            if (item.get("cat_features") != ["month_category"] or item.get("one_hot_max_size") != 12
                    or item["fit_actual_parameters"].get("one_hot_max_size") != 12
                    or item.get("fit_actual_cat_feature_indices") != [names.index("month_category")]):
                raise ValueError("В фактическом CatBoost не подтверждён категориальный месяц с one-hot=12.")
        rows.append({"Вариант": encoding, "Признаков": n_features,
                     "Представление месяца цели": ", ".join(name for name in names if name in MONTH_FIELDS),
                     "Категориальный dtype": dtypes.get("month_category", "—"),
                     "Новых успешных fit": len(successful),
                     "Источник": "сохранённый E02b" if encoding == "K0" else "E05b"})
    return _table(pd.DataFrame(rows))


def _findings(strategy: pd.DataFrame, native: pd.DataFrame, models: tuple[str, ...]) -> list[str]:
    lines = []
    for label, frame in (("Полная стратегия", strategy), ("Общее нативное пересечение", native)):
        for horizon in (1, 3, 6):
            selected = frame.loc[frame.split.eq("holdout") & frame.horizon.eq(horizon)
                                 & frame.metric_status.eq("complete")]
            reference = selected.loc[selected.model.eq(models[0])]
            if len(reference) != 1:
                continue
            differences = []
            for model in models[1:]:
                variant = selected.loc[selected.model.eq(model)]
                if len(variant) == 1:
                    delta = float(variant.mae_macro.iloc[0] - reference.mae_macro.iloc[0])
                    differences.append(f"{model}: {delta:+.2f} руб.")
            if differences:
                lines.append(f"{label}, holdout h={horizon}: изменение MAE macro относительно "
                             f"{models[0]} — " + "; ".join(differences) + ".")
    return lines or ["Полных сопоставимых метрик недостаточно для сравнения ошибок."]


def write_report(output: Path, report_path: Path, cfg: dict) -> None:
    """Сохранить отчёт завершённого E05b; существующий отчёт не перезаписывается."""
    if report_path.exists():
        raise FileExistsError("Существующий отчёт E05b автоматически не перезаписывается.")
    status, manifest = _json(output / "run_status.json"), _json(output / "run_manifest.json")
    preflight, features = _json(output / "preflight.json"), _json(output / "feature_sets.json")
    training = _json(output / "training_diagnostics.json")
    if not status["complete"] or not manifest["complete"] or manifest["config"] != cfg:
        raise ValueError("Для отчёта нужен завершённый запуск с той же конфигурацией.")
    if status["source_models_refitted"] or manifest["macro_features_enabled"] or not status["sources_unchanged"]:
        raise ValueError("В E05b нельзя переобучать ориентиры или подключать макроданные.")
    tests = manifest["tests"]
    if tests["exit_code"] != 0 or tests.get("full_pytest") is not True:
        raise ValueError("Отчёт требует успешного полного pytest до запуска.")
    if not preflight["e01_e02_data_protocol_keys_targets_verified"] or not preflight["municipality_1471_not_preexcluded"]:
        raise ValueError("Не подтверждены общий протокол и сохранение МО 1471.")
    horizons = list(cfg["backtest"]["horizons"])
    tables = {group: {scope: _read_metrics(output, group, scope) for scope in ("strategy", "native")}
              for group in ("trends", "calendar")}
    features_table = _feature_checks(features, training)
    tests_summary = tests.get("summary", tests.get("result", json.dumps(tests, ensure_ascii=False, sort_keys=True)))
    lines = [
        "# E05b: сезонный тренд и представление месяца в CatBoostDirect", "",
        f"Запуск завершён: {manifest.get('finished_at', cfg.get('experiment_date', 'дата в manifest'))}. "
        "Это исследовательское сравнение заранее фиксированных вариантов; просмотренный holdout "
        "не является новой независимой проверкой.", "",
        "## Общий протокол и границы", "",
        f"Сохранены {preflight['sample_size']} идентификатора МО, "
        f"{preflight['n_raw_keys']} точных запрошенных ключей на модель и "
        f"{preflight['n_evaluable_keys']} случаев с конечным фактом. МО 1471 не исключено заранее. "
        "Ключ: municipality_id + forecast_origin + target_period + horizon. Проверены совпадение "
        "фактов y_true, исходных данных, допуска МО и временного протокола E01/E02b.",
        f"История расходов — 2023-01…2024-12; выпуски "
        f"{cfg['backtest']['first_origin']}…{cfg['backtest']['last_origin']}, горизонты {horizons}. "
        f"Validation: цели до {cfg['backtest']['validation_target_end']} включительно; последующие цели — holdout. "
        f"Минимум истории на выпуске — {cfg['backtest']['min_history_observations']}, "
        f"максимальная давность — {cfg['backtest']['max_staleness_months']} месяц; "
        f"release_lag_months={cfg['data']['release_lag_months']} остаётся прежним допущением. "
        "Фактические даты публикации и пересмотры расходов не подтверждены.",
        "Ориентиры SeasonalNaiveYoY и K0 взяты из сохранённых прогнозов; повторного обучения "
        "соперников нет. Макропризнаки, загрузка источников и поиск гиперпараметров отсутствуют.", "",
        "## A. Сезонный тренд", "",
        "Тренд проверяется отдельно от кодирования месяца: у B/C другой явный алгоритм динамики, "
        "а у K1/K2 меняется только представление календарного признака фиксированного CatBoostDirect.",
        "C=O−L, T=O+h, D=h+L. По шести календарным позициям t=C−5,…,C: "
        "b=median((y_t−y_(t−12))/12), минимум три конечные пары. Это годовые разности "
        "совпадающих календарных месяцев, а не регрессия по сырому сезонному ряду. "
        "Шесть позиций не заменяются шестью последними непустыми парами.",
        "Все 12 месяцев C−11,…,C должны быть конечными. Для месяца года m: "
        "a_m=y[t_m]+b·months_between(C,t_m); level=mean(a_m), seasonal_m=a_m−level. "
        "SeasonalTrendLinear: max(0,level+seasonal_month(T)+bD). "
        "SeasonalTrendDamped: max(0,level+seasonal_month(T)+b·Σ(j=1…D)0.9^j). "
        "Первый будущий прирост равен 0.9b; затухает только будущий участок. "
        "Это заданные варианты, а не готовая реализация Holt-Winters.",
        "Недостаток пар или неполный шаблон дают явный резерв прежнего SeasonalNaiveYoY. "
        "Сохраняются b, n_annual_pairs, полнота шаблона, status, reason, effective_model и "
        "внутренний резерв LastValue. Ошибки исполнения — failed, а не предусмотренный fallback. "
        "На декабрь 2023 нет годовых пар: все h=12 B/C — резерв. Их MAE полной стратегии "
        "не является MAE собственного годового тренда.", "",
    ]
    for group, models, title in (("trends", TREND_MODELS, "A"), ("calendar", CALENDAR_MODELS, "B")):
        if group == "calendar":
            lines.extend(["## B. Кодирование месяца цели", "",
                "K0: числовой month + month_sin + month_cos. K1: строковый month_category "
                "(например 01), без month/sin/cos. K2: month_category + прежние sin/cos, без month. "
                "В K1/K2 явно передаются cat_features=['month_category'] и one_hot_max_size=12. "
                "Используется внутренний one-hot CatBoost; ручных dummy-столбцов нет.",
                "Преобразования применяются к копии X; смешанный DataFrame не приводится обратно "
                "целиком к float. days_in_month, time_index и остальные признаки сохранены. "
                "Месяц относится к цели O+h, включая переход декабрь/январь.", "", features_table, "",
                "Правила legacy, глобальная обучающая панель, seed и все параметры E02b сохраняются. "
                "Цель delta=y[r+h]−last_available(r−L); прогноз в рублях "
                "max(0,last_available(O−L)+prediction). Метка допустима при r+h+L≤O. "
                "На декабрь 2023 h=12 пар нет: прежний резерв SeasonalNaive, не нативный CatBoost.",
                "Общие исходные ключи, delta и X до преобразования месяца подписаны SHA256 и "
                "проверены между K0/K1/K2. В E02b построчные подписи не сохранялись: подпись K0 "
                "реконструирована неизменённым сборщиком при проверенных хешах данных/кода и "
                "сверке старых счётчиков пар/МО/дат. Это явно ограниченное доказательство, "
                "а не восстановленная старая построчная подпись.",
                "Фактически применённые параметры каждой успешной fit, имена/типы признаков и "
                "категориальные индексы сохранены в training_diagnostics.json. "
                "[Документация CatBoost](https://catboost.ai/docs/en/features/categorical-features) "
                "описывает обработку категорий; сама возможность one-hot не доказывает выигрыш MAE.", "",
                "Исходные фиксированные параметры:", "", "```json",
                json.dumps(cfg["models"]["catboost"], ensure_ascii=False, indent=2), "```", ""])
        for scope, scope_title in (("strategy", "Полная стратегия с явным резервом"),
                                   ("native", "Общая нативная область вариантов")):
            lines.extend([f"### {title}: {scope_title}", "",
                "Ориентир оценивается на точно тех же ключах. MAE — исходные номинальные рубли; "
                "pooled R² объединяет МО и не заменяет оценку временной динамики.", ""])
            for split, split_title in (("validation", "Validation"), ("holdout", "Holdout")):
                lines.extend([f"#### {split_title}", "", _metric_table(tables[group][scope], models, split, horizons), ""])
        coverage = pd.read_csv(output / f"{group}_native_coverage_by_model.csv")
        excluded = pd.read_csv(output / f"{group}_excluded_native_keys.csv")
        lines.extend([f"### {title}: нативное покрытие каждой модели", "",
            "Статусы показаны отдельно для всех запросов и строк с конечным фактом; "
            "семь отсутствующих фактов сохраняются в прогнозах. Общая нативная область "
            "требует native обоих новых трендов либо всех K0/K1/K2, соответственно.", "",
            _coverage_table(coverage, models), "",
            f"Из общего нативного пересечения исключено {len(excluded)} оцениваемых ключей. "
            f"Все ключи и причины сохранены в {group}_excluded_native_keys.csv; "
            "они остаются в полной стратегии и не исчезают из отчёта покрытия.", "",
            f"### {title}: фактическое сравнение ошибок", "",
            "Положительная разность означает увеличение ошибки, отрицательная — уменьшение.",
            *_findings(tables[group]["strategy"], tables[group]["native"], models), ""])
    lines.extend([
        "## Ограничения и следующий этап", "",
        "Небольшая разность ошибок без отдельной статистической проверки не объявляется "
        "значимой. Число МО не заменяет число временных дат. История короткая, годовой горизонт "
        "имеет только одну дату выпуска; нативных годовых прогнозов тренда и Direct нет. "
        "Окна, минимум пар, phi и кодирование не выбирались по просмотренному holdout. "
        "Результат одного фиксированного пилота не доказывает универсальное преимущество "
        "one-hot, причинный эффект сезонности или устойчивость будущих прогнозов.",
        "E05c запланирован и НЕ выполнен: основа — фиксированный CatBoostDirect K0, без выбора "
        "лучшего кодирования задним числом; последовательное добавление национальных прогнозов "
        "инфляции и потребления категории A. Их можно проверять без ожидания региональных "
        "данных. B — отдельный сценарный эксперимент. Региональные зарплата/ИПЦ остаются "
        "незавершёнными. Национальные ожидания нельзя выдавать за региональные или механически "
        "складывать с ростом расходов. E05c здесь не запускался.", "",
        "## Тесты, команды и воспроизводимость", "",
        f"Полный pytest до эксперимента: {tests_summary}; код {tests['exit_code']}.", "",
        "```powershell", str(tests.get("command", "Точная команда сохранена в validation_record")), "```", "",
        "Реально сохранённые команды проверочного/полного запуска:", "", "```powershell",
        *manifest["run_commands"], "```", "",
        f"Завершено {status['n_origins_completed']} из {status['n_origins_expected']} выпусков; "
        f"новых fit вызвано {status['n_new_fits_called']}, успешно {status['n_new_fits_succeeded']}; "
        f"failed-строк {status['n_failed']}. Совпадений подписей подготовки "
        f"K1/K2 с K0: {status['n_training_key_signature_matches']}.",
        f"Git HEAD: {manifest['git_commit']}; dirty={manifest['has_uncommitted_changes']}; "
        f"seed={manifest['seed']}; Python={manifest['python']}; "
        f"зафиксировано версий пакетов: {len(manifest['versions'])}. "
        "Manifest хранит конфигурацию, версии, команды, хеши входа/кода/E01/E02 и артефактов. "
        "Окружение, зависимости, прежние результаты и посторонние изменения сохраняются; "
        "commit/push не выполнялись.",
        "predictions.csv.gz содержит прогнозы, факты, статусы, причины и effective_model. "
        "Для каждой группы metrics_strategy/native содержат MAE macro/micro и pooled R²; "
        "metrics_*_by_origin и metrics_*_by_municipality — метрики по датам и МО. "
        "coverage / coverage_evaluable / native_coverage_by_model отражают оба покрытия. "
        "strategy_keys, common_native_keys, own_native_keys, excluded_native_keys задают "
        "точные области. partitions/errors_*.json содержит ошибки; training_diagnostics.json "
        "и feature_sets.json — подготовку, схемы и реальные параметры. Построчные данные "
        "остаются в outputs и автоматически не добавляются в Git.", "",
    ])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
