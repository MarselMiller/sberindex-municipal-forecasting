"""Отчёт E03 по сохранённым прогнозам и диагностике, без загрузки модели."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


MODELS = ["Chronos-2", "CatBoostDirect", "CatBoostRecursive", "SeasonalNaiveYoY",
          "ProphetAuto", "ProphetYearly"]


def _table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "Нет сохранённых строк."
    rows = ["| " + " | ".join(map(str, frame.columns)) + " |",
            "| " + " | ".join("---" for _ in frame.columns) + " |"]
    for row in frame.itertuples(index=False, name=None):
        rows.append("| " + " | ".join(str(value).replace("|", " / ").replace("\n", " ")
                                       for value in row) + " |")
    return "\n".join(rows)


def _mae_table(metrics: pd.DataFrame, split: str) -> str:
    rows = []
    for model in MODELS:
        row = {"Модель": model}
        for horizon in (1, 3, 6, 12):
            values = metrics.loc[metrics.split.eq(split) & metrics.model.eq(model)
                                 & metrics.horizon.eq(horizon)]
            if len(values) > 1:
                raise ValueError("Повторяющиеся группы метрик в отчёте E03.")
            if values.empty:
                value = "нет случаев"
            else:
                result = values.iloc[0]
                status = result.metric_status
                if status == "no_training_pairs":
                    value = "нет обучающих пар"
                elif status == "incomplete_failed":
                    value = "неполное покрытие: failed"
                elif status == "incomplete_unavailable":
                    value = "неполное покрытие: unavailable"
                elif status == "no_common_success_keys":
                    value = "нет общих успешных ключей"
                elif status == "no_evaluable_cases":
                    value = "нет случаев с фактом"
                elif status == "complete" and pd.notna(result.mae_macro):
                    value = f"{result.mae_macro:.2f}"
                else:
                    value = f"не вычислено ({status})"
            row[f"h={horizon}"] = value
        rows.append(row)
    return _table(pd.DataFrame(rows))


def _display(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for name in result.columns:
        if name.endswith("share") or name.endswith("coverage"):
            result[name] = result[name].map(lambda value: f"{100 * value:.2f}%" if pd.notna(value) else "—")
        elif name in ("mae_macro", "mae_micro", "r2_pooled"):
            result[name] = result[name].map(lambda value: f"{value:.4f}" if pd.notna(value) else "—")
    return result


def _resource_table(manifest: dict, smoke: dict) -> str:
    rows = []
    for phase, source in (("smoke", smoke), ("полный запуск", manifest)):
        for moment in ("before", "after"):
            resources = source.get(f"resources_{moment}")
            if resources is not None:
                rows.append({"Этап": phase, "Момент": moment,
                             "Сохранённые ресурсы": json.dumps(resources, ensure_ascii=False, sort_keys=True)})
    return _table(pd.DataFrame(rows))


def _holdout_findings(metrics: pd.DataFrame) -> list[str]:
    lines = []
    for horizon in (1, 3, 6, 12):
        values = metrics.loc[metrics.split.eq("holdout") & metrics.horizon.eq(horizon)
                             & metrics.metric_status.eq("complete") & metrics.mae_macro.notna()]
        chronos = values.loc[values.model.eq("Chronos-2")]
        rivals = values.loc[values.model.isin(MODELS[1:])]
        if len(chronos) != 1 or rivals.empty:
            continue
        best = rivals.sort_values(["mae_macro", "model"]).iloc[0]
        mae = float(chronos.mae_macro.iloc[0])
        relation = "меньше" if mae < best.mae_macro else "больше" if mae > best.mae_macro else "равна"
        lines.append(f"h={horizon}: MAE Chronos-2 {mae:.2f}; минимальная MAE среди доступных "
                     f"соперников — {best.model}, {best.mae_macro:.2f}. Ошибка Chronos {relation}.")
    return lines or ["Полных метрик недостаточно для сопоставимого вывода о качестве Chronos-2."]


def write_report(output: Path, destination: Path, cfg: dict) -> None:
    """Сформировать сводный Markdown строго из артефактов завершённого запуска."""
    if destination.exists():
        raise FileExistsError("Существующий отчёт E03 не перезаписывается автоматически.")
    metrics = pd.read_csv(output / "metrics_full.csv")
    common = pd.read_csv(output / "metrics_common_success.csv")
    coverage = pd.read_csv(output / "coverage.csv")
    comparison = pd.read_csv(output / "comparison_coverage.csv")
    timings = pd.read_csv(output / "timings.csv")
    status = json.loads((output / "run_status.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    configured_output = Path(cfg["output_dir"])
    project_root = output.resolve()
    for _ in configured_output.parts if not configured_output.is_absolute() else ("outputs", "run"):
        project_root = project_root.parent
    smoke_path = Path(cfg["smoke"]["output_dir"])
    if not smoke_path.is_absolute():
        smoke_path = project_root / smoke_path
    smoke_file = smoke_path / "smoke_status.json"
    smoke = json.loads(smoke_file.read_text(encoding="utf-8")) if smoke_file.exists() else {}
    incomplete = comparison.incomplete.astype(str).str.lower().isin(["true", "1"]).any()
    chronos_details = metrics.loc[metrics.model.eq("Chronos-2")].copy()
    detail_columns = [name for name in ("split", "horizon", "n_predictions", "n_success", "n_failed",
                                        "n_unavailable", "mae_macro", "mae_micro", "r2_pooled", "metric_status")
                      if name in chronos_details]
    commands = manifest.get("invocation_commands", manifest.get("run_commands", []))
    weights = manifest.get("weights_sha256", {})
    weight_lines = [f"{name}: {digest}" for name, digest in sorted(weights.items())] if isinstance(weights, dict) else [str(weights)]
    config = cfg["chronos"]
    lines = ["# E03 — Chronos-2, zero-shot", "",
        f"Дата запуска: {cfg['experiment_date']}. Конфигурация: configs/chronos_zero_shot.yaml.",
        f"Результаты: {cfg['output_dir']}/. Git commit: {manifest.get('git_commit')}.",
        f"Незакоммиченные изменения на запуске: {manifest.get('has_uncommitted_changes')}.", "",
        "## Постановка и сохранение протокола", "",
        "Прогнозируется месячное значение категории «Все категории» по МО в исходных рублях — оценка средних безналичных расходов жителей.",
        "Zero-shot: веса не дообучаются, подбор параметров по validation или holdout не выполняется. "
        "На каждом выпуске модель получает только исторический префикс до O−L, без будущих фактов и прогнозов других выпусков.",
        "Сохранены даты выпуска декабрь 2023 — ноябрь 2024, горизонты 1/3/6/12 календарных месяцев, "
        "validation по июнь 2024 и holdout июль–декабрь 2024. Требования прогнозной истории ≥12 наблюдений "
        "и давности ≤1 месяца сохранены. Лаг публикации L=0 остаётся неподтверждённым допущением.",
        "Выборка — исходные 64 идентификатора E01, без исключения МО 1471 по будущему отсутствию фактов; "
        "63 МО имеют оцениваемые факты. Календарные пропуски остаются NaN и не сдвигают месяцы.",
        "CatBoostDirect, CatBoostRecursive, SeasonalNaiveYoY и оба Prophet взяты из сохранённых прогнозов "
        "E01/E02b. Соперники повторно не обучаются. Годовой резерв E02b исключён из чистого CatBoostDirect: "
        "для h=12 на декабрь 2023 нет обучающих пар и нет годовой MAE обученной прямой модели.", "",
        "## Модель и происхождение весов", "",
        f"Checkpoint: {config['model_id']}; revision: {manifest.get('weight_revision', config['revision'])}. "
        f"Версия пакета: {manifest.get('versions', {}).get('chronos-forecasting', config.get('package_version'))}.",
        f"Device={manifest.get('device', config['device'])}, dtype={manifest.get('dtype', config['dtype'])}, "
        f"batch_size={manifest.get('batch_size', config['batch_size'])}, seed={manifest.get('seed', cfg['seed'])}; "
        f"cross_learning={config['cross_learning']}, quantile={config['quantile']}.",
        "Точечный прогноз — квантиль 0.5. Постобработка отсутствует: отрицательные значения не обрезаются. "
        "Длина совместного прогноза соответствует наибольшему нужному календарному горизонту от cutoff.",
        "Chronos-2 — модель примерно на 120 млн параметров, опубликованная 20 октября 2025 года. "
        "Этот checkpoint не существовал в исторических датах оценки 2023–2024; E03 — ретроспективное "
        "исследовательское сравнение с современной моделью. "
        "[Официальный репозиторий](https://github.com/amazon-science/chronos-forecasting).",
        "Предобучение включает публичные временные ряды из наборов Chronos/GIFT и синтетические данные. "
        "Отсутствие пересечения предобучения с нашими данными и информацией за оцениваемый период не доказано. "
        "Ограничение префикса в адаптере проверяет доступность локального входа, но не историческую чистоту весов. "
        "[Карточка модели: training data](https://huggingface.co/amazon/chronos-2#training-data), "
        "[статья, раздел 4](https://arxiv.org/html/2510.15821v1#S4).", "",
        "## Команды и статус", "", "```powershell", *commands, "```", "",
        f"Все выпуски рассчитаны: {status.get('complete')}; полное сравнение: {status.get('full_comparison_complete')}. "
        f"Выпусков завершено: {status.get('n_origins_completed')} из {status.get('n_origins_expected')}.",
        f"Chronos: запрошено {status.get('n_requested')}, native={status.get('n_native')}, "
        f"failed={status.get('n_failed')}, без целевого факта={status.get('n_missing_truth')}.",
        "Fallback отсутствует. Ошибка инференса сохраняется как failed с NaN-прогнозом и причиной, "
        "а не заменяется другой моделью или исчезает из полной оценки.", "",
        "## Покрытие", "",
        "Полная область определяется наличием факта, независимо от успешности нового прогноза. "
        "Равенство выборок проверяется по municipality_id + forecast_origin + target_period + horizon.", "",
        _table(_display(coverage)), "",
        "Общее успешное пересечение и размер исключений по группам:", "",
        _table(_display(comparison)), "",
        "Для h=12 участники успешного сравнения — Chronos и четыре модели E01; чистый Direct обозначен "
        "как unavailable/no_training_pairs. Это структурное отсутствие обучения, а не сбой Chronos.", "",
        "## MAE macro на полной области", "",
        "MAE macro — среднее MAE муниципалитетов с равными весами. Если модель имеет failed или "
        "unavailable в полной группе, её MAE этой группы не вычисляется. Ошибочные случаи не удаляются молча.", "",
        "### Holdout", "", _mae_table(metrics, "holdout"), "",
        "### Validation", "", _mae_table(metrics, "validation"), "",
        "MAE micro и pooled R² Chronos-2 в исходной шкале:", "",
        _table(_display(chronos_details[detail_columns])), ""]
    if incomplete:
        lines.extend(["## Дополнительная оценка на общем успешном пересечении", "",
            "Полное покрытие нарушено. Эта оценка явно ограничена общими успешными ключами всех "
            "участников группы и не заменяет полную оценку. Исключения сохранены отдельно с причинами.", "",
            "### Holdout", "", _mae_table(common, "holdout"), "",
            "### Validation", "", _mae_table(common, "validation"), ""])
    else:
        lines.extend(["Общее успешное пересечение совпадает с полной областью для всех участвующих моделей; "
                      "его метрики также сохранены отдельно в metrics_common_success.csv.", ""])
    lines.extend(["## Smoke и затраты ресурсов", "",
        f"Smoke: passed={smoke.get('passed', status.get('smoke_passed'))}, "
        f"рядов={smoke.get('n_series')}, prediction_length={smoke.get('prediction_length')}, "
        f"инференс={smoke.get('seconds')} с, загрузка={smoke.get('model_load_seconds')} с.",
        f"Полный запуск: загрузка модели={status.get('model_load_seconds')} с, "
        f"инференс={status.get('inference_seconds')} с, общее время={status.get('total_seconds')} с. "
        "Эти времена относятся к зафиксированной CPU-среде и включают только явно указанные фазы.", "",
        _table(timings), "", _resource_table(manifest, smoke), "",
        "## Вывод и ограничения", "", *_holdout_findings(metrics), "",
        "Эти числа описывают фиксированный пилот. Уже просмотренный holdout не является новой "
        "независимой проверкой, и результат не используется для смены дат, выборки или настройки модели.",
        "Месячная история короткая: число МО и прогнозных строк не заменяет число различных дат. "
        "Годовой прогноз проверяется на одной дате выпуска; validation h=6 также имеет одну дату. "
        "Рейтинг нельзя переносить на полную панель МО или устойчивость прогноза будущих шоков.",
        "Фактические даты публикации и vintages не подтверждены; возможные пересечения предобучения "
        "и недоступность текущего checkpoint в 2023–2024 ограничивают вывод о временной чистоте. "
        "Преимущество фундаментальной модели требует отдельной сопоставимой проверки, а не следует из её класса.", "",
        "## Воспроизводимость и источники чисел", "",
        "predictions.csv.gz сохраняет факты, прогнозы, статусы, effective_model и причины. "
        "metrics_full.csv и metrics_common_success.csv содержат MAE macro/micro, pooled R², размеры "
        "и статусы групп; evaluation_keys_*.csv и excluded_common_success_keys.csv задают области и исключения.",
        "coverage*.csv и comparison_coverage.csv сохраняют покрытие, timings.csv — время по выпуску, "
        "partitions/errors_*.json — ошибки. Smoke сохранён отдельно. Manifest фиксирует команду, seed, "
        "версии, Git/status, хеши входа, кода, исходных результатов и скачанных весов.",
        "Исходные данные, outputs/baseline_v1/, E01 и E02b сохраняются отдельно от E03. "
        "Построчные результаты и веса остаются в outputs и автоматически в Git не добавляются.", "",
        "Хеши весов из сохранённого manifest:", "", "```text", *weight_lines, "```", ""])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")
