"""Отчёт E02b: таблицы читаются из сохранённых результатов, без нового fit."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .direct_model import NATIVE_MODEL, STRATEGY


def table(frame: pd.DataFrame) -> str:
    rows = ["| " + " | ".join(map(str, frame.columns)) + " |",
            "| " + " | ".join("---" for _ in frame.columns) + " |"]
    for row in frame.itertuples(index=False, name=None):
        rows.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(rows)


def mae_table(metrics: pd.DataFrame, split: str, models: list[str]) -> str:
    rows = []
    for model in models:
        row = {"Модель": model}
        for h in (1, 3, 6, 12):
            values = metrics.loc[metrics.split.eq(split) & metrics.model.eq(model) & metrics.horizon.eq(h)]
            if values.empty:
                value = "нет случаев"
            elif values.metric_status.iloc[0] == "no_training_pairs":
                value = "нет обучающих пар"
            elif values.metric_status.iloc[0] == "no_native_comparison_keys":
                value = "нет нативных ключей"
            elif values.metric_status.iloc[0] == "incomplete_failed":
                value = "неполное покрытие: failed"
            else:
                value = f"{values.mae_macro.iloc[0]:.2f}"
            row[f"h={h}"] = value
        rows.append(row)
    return table(pd.DataFrame(rows))


def write_report(output: Path, destination: Path, cfg: dict) -> None:
    strategy = pd.read_csv(output / "metrics_strategy.csv")
    native = pd.read_csv(output / "metrics_native.csv")
    coverage = pd.read_csv(output / "coverage_e01.csv")
    training = pd.read_csv(output / "training_diagnostics.csv")
    status = json.loads((output / "run_status.json").read_text(encoding="utf-8"))
    h1 = json.loads((output / "h1_consistency.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    models = cfg["comparison"]["models"]
    if destination.exists():
        raise FileExistsError("Существующий отчёт не перезаписывается автоматически.")
    coverage_display = coverage.copy()
    for name in ("fallback_share", "failed_share"):
        coverage_display[name] = coverage_display[name].map(lambda x: f"{100*x:.2f}%")
    direct_details = strategy.loc[strategy.model.eq(STRATEGY), [
        "split", "horizon", "n_predictions", "mae_macro", "mae_micro", "r2_pooled", "metric_status"]].copy()
    for name in ("mae_macro", "mae_micro", "r2_pooled"):
        direct_details[name] = direct_details[name].map(lambda x: f"{x:.4f}" if pd.notna(x) else "—")
    training_summary = training.groupby("horizon").agg(
        n_fits=("fit_called", "sum"), min_rows=("n_training_rows", "min"), max_rows=("n_training_rows", "max"),
        min_municipalities=("n_training_municipalities", "min"), max_municipalities=("n_training_municipalities", "max"),
        min_dates=("n_historical_origins", "min"), max_dates=("n_historical_origins", "max"),
    ).reset_index()
    lines = ["# E02 — Прямой CatBoost, режим legacy", "",
        f"Дата эксперимента: {cfg['experiment_date']}. Конфигурация: configs/catboost_direct.yaml.",
        f"Результаты: {cfg['output_dir']}/. Git commit на запуске: {manifest['git_commit']}.",
        f"Рабочая папка имела незакоммиченные изменения: {manifest['has_uncommitted_changes']}.", "",
        "## Постановка и сохранение протокола", "",
        "Цель — месячное значение категории «Все категории» по МО в исходных рублях, оценка средних безналичных расходов.",
        "Это исследовательское сравнение на уже просмотренном holdout, не новая независимая проверка.",
        "Режим обучения legacy. Strict12 не запускался как прогнозный эксперимент. Параметры CatBoost, seed=42, признаки истории и delta-преобразование сохранены из E01.",
        "На каждом выпуске и горизонте обучается отдельная глобальная модель по всей доступной панели; 64 МО из сохранённого sample_ids.json ограничивают только прогнозную выборку.",
        "Признаки примера используют cutoff=r−L; цель=r+h допускается только при r+h+L≤O. Восстановление: max(anchor + predicted_delta, 0). Календарные признаки описывают месяц цели; прогнозы других горизонтов не используются.",
        "Даты выпуска: декабрь 2023 — ноябрь 2024; цели validation по июнь 2024, holdout июль–декабрь. Лаг публикации L=0 — неподтверждённое допущение; vintages отсутствуют.",
        "Данные, код E01, параметры, идентификаторы выборки, факты и ключи каждого соперника проверены перед обучением. МО 1471 не исключено заранее; его отсутствующие факты сохранены.",
        "Prophet и другие модели E01 повторно не обучались: использованы сохранённые прогнозы всех шести моделей, без выбора варианта Prophet по выгодному результату.", "",
        "## Команды", "", "```powershell", *manifest["run_commands"], "```", "",
        "## Статус и покрытие", "",
        f"Все выпуски рассчитаны: {status['complete']}; сравнение на всей общей области E01 завершено: {status['full_e01_comparison_complete']}.",
        f"Запрошено {status['n_requested']} прогнозов: native={status['n_native']}, fallback_no_training_pairs={status['n_fallback']}, failed={status['n_failed']}.",
        f"Общая область E01 содержит {status['n_e01_evaluable']} случаев с фактом; семь запрошенных случаев без факта не включены в метрики.",
        "Резерв заранее назначен: SeasonalNaive. Его effective_model сохранена отдельно; fit при отсутствии пар не вызывается. Ошибка fit/predict имеет failed и журнал причины, резервом не заменяется.",
        "При failed метрики стратегии для соответствующей полной группы не вычисляются: NaN/incomplete_failed. Нативная область ограничивается явно и одинаково для всех моделей.", "",
        table(coverage_display), "",
        "## A. Стратегия CatBoostDirect + SeasonalNaive", "",
        "Область: все общие оцениваемые ключи E01. MAE macro — среднее MAE МО с равными весами.", "",
        "### Holdout", "", mae_table(strategy, "holdout", [STRATEGY, *models]), "",
        "### Validation", "", mae_table(strategy, "validation", [STRATEGY, *models]), "",
        "MAE micro и pooled R² стратегии (все соперники также сохранены в metrics_strategy.csv):", "",
        table(direct_details), "",
        "## B. Только обученный CatBoostDirect", "",
        "Из стратегии оставлены только status=native. Каждый соперник оценивается на точно тех же ключах, сохранённых в evaluation_keys_native.csv.", "",
        "### Holdout", "", mae_table(native, "holdout", [NATIVE_MODEL, *models]), "",
        "### Validation", "", mae_table(native, "validation", [NATIVE_MODEL, *models]), "",
        "Для h=12 на декабрь 2023 нет обучающих пар. Годовая MAE стратегии принадлежит резерву SeasonalNaive; MAE обученной прямой модели здесь не существует.", "",
        "## Обучающая история и проверка h=1", "", table(training_summary), "",
        "Число строк по МО не заменяет число независимых дат. Для h=6 на validation всего одна дата оценки; для h=12 — одна годовая дата E01 и ноль обучающих пар.",
        f"h=1: сравнено {h1['n_compared_native_cases']} из {h1['n_expected_cases']} случаев, включая случаи без целевого факта. "
        f"Максимальное абсолютное расхождение с CatBoostRecursive: {h1['max_absolute_difference']}. "
        f"Допуск абсолютный {h1['absolute_tolerance']}, относительный 0; превышений {h1['n_exceeds_tolerance']}.", "",
        "## Вывод и ограничения", "",
        "Результаты описывают этот фиксированный пилот. Рейтинг holdout не используется для подбора параметров, изменения выборки, переноса дат или выбора Prophet.",
        "Прямой способ прогнозирования не гарантирует меньшую ошибку: значения каждой модели представлены без смены протокола. Для h=1 совпадение с рекурсивной моделью служит проверкой неизменности обучения и шкалы.",
        "Нельзя переносить вывод на полную прогнозную выборку примерно из 2000 МО, устойчивость годового прогноза или раннее предупреждение шоков. Независимой разметки шоков и новых данных в E02 нет.", "",
        "## Источники чисел", "",
        "metrics_strategy.csv и metrics_native.csv содержат MAE macro/micro, pooled R², размеры областей и статусы; coverage*.csv — покрытие и failed/резервы; training_diagnostics.csv — обучение по выпуску и горизонту; h1_consistency*.json/csv — сверка h=1.",
        "predictions.csv.gz содержит y_true/y_pred, status, effective_model и причины; partitions/errors_*.json — журналы. Manifest сохраняет команды, конфигурацию, версии, seed, Git/status и хеши исходных данных, кода и E01.",
        "Построчные результаты остаются в игнорируемой папке outputs; этот отчёт содержит только сводки. E01 и исходные данные не изменены.", ""]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")
