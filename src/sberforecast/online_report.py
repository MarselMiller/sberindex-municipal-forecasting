"""Русский отчёт E04a, вычисленный исключительно из сохранённых артефактов."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def table(frame: pd.DataFrame, columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in frame[columns].itertuples(index=False, name=None):
        values = ["—" if pd.isna(value) else f"{value:.3f}" if isinstance(value, (float, np.floating)) else str(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(output: Path, report_path: Path) -> None:
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    cfg = manifest["config"]
    status = json.loads((output / "run_status.json").read_text(encoding="utf-8"))
    selected = json.loads((output / "selected_parameters.json").read_text(encoding="utf-8"))
    real = json.loads((output / "real_diagnostics.json").read_text(encoding="utf-8"))
    metric_path = output / "synthetic_test_metrics.csv.gz"
    metrics = pd.read_csv(metric_path) if metric_path.exists() else pd.DataFrame()
    uncertainty = pd.read_csv(output / "synthetic_test_uncertainty.csv.gz") if len(metrics) else pd.DataFrame()
    primary = metrics.loc[metrics.scope.eq("primary_level")] if len(metrics) else pd.DataFrame()
    params = []
    for method, item in selected.items():
        if item["selected"]:
            params.append({"method": method, "parameters": json.dumps(item["parameters"], sort_keys=True),
                           "validation_no_change_FAR": item["validation_control_far"]["no_change"],
                           "validation_outlier_FAR": item["validation_control_far"]["outlier"]})
        else:
            params.append({"method": method, "parameters": item["reason"], "validation_no_change_FAR": np.nan, "validation_outlier_FAR": np.nan})
    sections = [
        "# E04a — сравнение онлайн-детекторов изменений",
        "Код подготовлен, полный pytest пройден, минимальный и полный запуски выполнены. "
        "Получены синтетические тестовые метрики и диагностические сигналы на сохранённых прогнозах E01. "
        "Это обнаружение уже начавшихся изменений; предсказание будущих шоков не проверялось.",
        f"Источник всех таблиц/графиков: `{cfg['output_dir']}/`. Git HEAD: `{manifest['git_commit']}`; "
        f"незакоммиченные изменения: {manifest['has_uncommitted_changes']}. Команда: `{manifest['command']}`. "
        f"Полный pytest: `{manifest['test_command']}`. Его результат и хеши: `outputs/e04a_checks/test_validation.json`.",
        "## Зафиксированный протокол",
        "Основное событие — добавочный сдвиг уровня расходов, положительный или отрицательный, "
        "сохраняющийся до конца ряда, не менее трёх месяцев. Изменения наклона и дисперсии — "
        "отдельные дополнительные сценарии. Одиночный выброс и обычная неизменная сезонность "
        "не являются структурными событиями. Расходы, ошибки прогноза и дата сигнала различаются: "
        "смена остатка не доказывает экономический шок.",
        "24 месяца (январь 2023 — декабрь 2024), уровень 10000, синусоидальная сезонность 10%, "
        "линейный тренд 0.4% базового уровня в месяц, независимый нормальный шум 1%/3%. "
        "Сила 2/4/6 означает величину сдвига в единицах исходного σ шума; наклон достигает "
        "такой добавки за три месяца, дисперсионный сценарий умножает σ на 2/4/6. "
        "Выброс меняет ровно один месяц, знак чередуется по повтору. Основные события начинаются "
        "в мае–сентябре 2024, после warmup, с полным окном оценки. Истина генератора "
        "не передаётся прогнозу, масштабированию и детекторам.",
        f"Validation: {status['n_validation_series']} рядов, test: {status['n_test_series']}; "
        "30 независимых реализаций на ячейку scenario×strength×noise. В каждом наборе 1140 основных "
        "и 120 граничных рядов. Seed 100000…101259 и 200000…201259 не пересекаются, "
        "series_id также раздельные. Граничный сдвиг в марте до мониторинга и ноябрьский "
        "сдвиг с двумя наблюдаемыми месяцами сохранены отдельно: последний не удовлетворяет "
        "требованию трёх месяцев в наблюдаемой истории и имеет неполное окно. "
        "Выборка test создаётся после сохранения выбранных параметров. Протокол/сетка/правила "
        "фиксированы до запуска в config_resolved.yaml; параметры test не подбирались.",
        "## Общий причинный вход и масштаб",
        "Прогноз SeasonalNaiveYoY h=1: тот же месяц прошлого года × медианный коэффициент "
        "из последних трёх доступных годовых отношений, ограниченный [0.5,2]; при отсутствии "
        "сезонной опоры — последнее известное значение по старому коду. Синтетика использует "
        "неизменный baseline_predict и правила допуска E01; истинные даты событий ему неизвестны. "
        "Реальный вход — сохранённые ошибки y_true−y_pred из E01; повторное обучение отсутствует. "
        "Выбор этого входа сделан после просмотра E01 и не выдаётся за решение до backtest.",
        "Первые четыре конечные ошибки каждого ряда — только warmup. Фиксированные центр c "
        "и масштаб s впервые применяются к следующему наблюдению: c=median(e₁…e₄), "
        "s=max(1.4826 median|e−c|, 0.03 median|ŷ₁…ŷ₄|, 1 рубль); z=(e−c)/s. "
        "Текущий мониторируемый остаток и будущие значения не участвуют в калибровке. "
        "Порог 3% стабилизирует четыре наблюдения, но ограничивает чувствительность к слабым "
        "изменениям. Межмуниципальный pooling не используется. Пропущенные факты сохраняют "
        "календарь, не дают нулевой ошибки и не обновляют состояние.",
        "Наблюдение месяца t используется в конце месяца t+L; availability_date и signal_date "
        "сохранены отдельно от observation_period. L=0 взят из E01 как неподтверждённое допущение. "
        "Точные публикации и vintages не восстановлены. После выданной тревоги каждый детектор "
        "сбрасывается; следующие два полных календарных месяца сигналы подавляются, "
        "но состояние обновляется. Пропуск не сбрасывает историю и не продлевает cooldown.",
        "## Формулы и BOCPD",
        "CUSUM: S⁺=max(0,S⁺+z−k), S⁻=max(0,S⁻−z−k), score=max(S⁺,S⁻); "
        "тревога при score>threshold. EWMA: mₙ=αzₙ+(1−α)mₙ₋₁, m₀=0; "
        "σₙ²=α/(2−α)[1−(1−α)²ⁿ]; score=|mₙ|/σₙ, тревога при score>threshold. "
        "Формулы проверены по [NIST CUSUM](https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc323.htm) "
        "и [NIST EWMA](https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc324.htm).",
        "BOCPD реализует Algorithm 1 [Adams & MacKay](https://arxiv.org/abs/0710.3742), "
        "точную рекурсию run-length без усечения, в логарифмах через logsumexp. "
        "Модель: z|μ,σ²~Normal(μ,σ²), σ²~InvGamma(a,b), μ|σ²~Normal(m,σ²/κ); "
        "prior m=0,κ=1,a=2,b=1, predictive Student-t с ν=2a, location=m, "
        "scale²=b(κ+1)/(aκ). Hazard постоянен и выбирается из 1/24,1/60. "
        "В данной рекурсии P(r=0)=H — константа, это проверено тестом. Полезный score "
        "здесь Σᵣ₌₁²P(r|z₁:ₙ); первые два обновления после reset имеют score=0 "
        "для защиты от стартовой короткой серии. Это дополнительное ограничение чувствительности "
        "на коротких рядах; BOCPD не является готовым универсальным детектором из статьи. "
        "Сумма posterior, P0, короткая вероятность и MAP run-length сохранены в signals.",
        "## Выбор рабочих точек — только validation",
        "Общий бюджет: ≤1 тревоги на 12 наблюдаемых месяцев мониторинга отдельно "
        "на no_change и outlier. Из удовлетворяющих бюджету вариантов выбирается максимальная "
        "F1 основных level_up/down, затем recall, меньший максимальный FAR двух контролей, "
        "меньшая задержка и candidate_id. Если подходящих точек нет, метод не выбирается; "
        "лимит не пересматривается. Все 18 кандидатов и фактические частоты сохранены.",
        table(pd.DataFrame(params), ["method", "parameters", "validation_no_change_FAR", "validation_outlier_FAR"]),
        "## Синтетический test",
        "Сигнал сопоставляется с событием только в календарном окне начала…+3 месяца включительно "
        "по месяцу фактического signal_date. Один сигнал — максимум одно событие; одно событие — "
        "максимум один сигнал. Предшествующие и повторные сигналы считаются FP. "
        "Неполные окна/события до мониторинга исключаются из основной recall, но сохраняются "
        "с явным eligibility; сигналы внутри их окон имеют excluded_censored_event. "
        "Знаменатель FAR — наблюдаемые месяцы мониторинга, включая cooldown, без warmup/пропусков.",
    ]
    metric_columns = ["method", "n_events", "precision", "recall", "f1", "missed_fraction", "median_delay", "false_alarms_per_12_months"]
    if len(primary):
        sections += ["Основные устойчивые сдвиги уровня, оба направления:", table(primary, metric_columns)]
        event_matches = pd.read_csv(output / "synthetic_test_event_matches.csv.gz")
        exclusions = event_matches.groupby(["method", "eligibility", "status"]).size().reset_index(name="n_events")
        sections += ["Фактический учёт окон событий (граничные кандидаты короткого сдвига "
                     "не объявлены основными устойчивыми событиями):",
                     table(exclusions, ["method", "eligibility", "status", "n_events"])]
        scenario = metrics.loc[metrics.scope.eq("by_scenario")]
        sections += ["Сценарии показаны отдельно; у контролей recall/задержка не определены:", table(scenario, ["method", "scenario", "n_series", *metric_columns[1:]])]
        primary_ci = uncertainty.loc[uncertainty.scope.eq("primary_level")]
        sections += ["95% percentile интервалы: 500 bootstrap целых рядов со стратификацией "
                     "scenario×strength×noise, seed=300000. Они условны на уже выбранные параметры, "
                     "не учитывают неопределённость выбора на validation. Строки месяцев не ресэмплируются.",
                     table(primary_ci, ["method", "metric", "estimate", "ci_low", "ci_high"]),
                     "Полные результаты по силе и шуму: synthetic_test_metrics.csv.gz, "
                     "synthetic_test_uncertainty.csv.gz; построчные события/сигналы/сопоставления "
                     "сохранены отдельно. Нельзя переносить эти precision/recall на реальные МО."]
    else:
        sections.append("Ни один метод не выбран в заданной сетке; тестовых метрик выбранных моделей нет.")
    sections += ["## Диагностика реальных ошибок E01"]
    if real.get("alarms_by_method"):
        sections += [f"Исходные ID: {real['n_sample_municipalities']}. После warmup доступны "
                     f"{real['n_municipalities_with_monitoring']} МО и {real['n_observed_monitoring_months_per_method']} "
                     f"наблюдаемых месяцев на метод: {real['monitoring_observation_first_month']}…"
                     f"{real['monitoring_observation_last_month']}. Warmup: {real['n_warmup_months_per_method']} "
                     f"ошибок; отсутствующих календарных строк: {real['n_missing_months_per_method']}. "
                     "МО 1471 сохранено в выборке; ни одной доступной ошибки для его warmup нет.",
                     "Число тревог: " + ", ".join(f"{method}: {count}" for method, count in real["alarms_by_method"].items()) + ". "
                     f"Совпадение всех выбранных методов в одном МО/месяце: {real['n_same_month_alarms_all_selected_methods']}.",
                     table(pd.read_csv(output / "real_agreement.csv"), ["method_a", "method_b", "same_month_alarm_intersection", "a_only", "b_only", "jaccard"]),
                     "Правило иллюстраций фиксировано до запуска: три минимальных числовых ID "
                     "исходной выборки плюс минимальный ID без мониторинга. Получены " + ", ".join(real["illustration_ids"]) + ". "
                     "Показываются история расходов, сохранённый прогноз, ошибка и реально датированные сигналы; "
                     "выбор не зависит от успешности детектора."]
        for uid in real["illustration_ids"]:
            sections.append(f"![МО {uid}](../../{cfg['output_dir']}/figures/municipality_{uid}.png)")
    else:
        sections.append(f"Реальные детекторы не запускались: нет рабочей точки в заданном бюджете. "
                        f"Исходная выборка {real['n_sample_municipalities']} МО; "
                        f"потенциально доступны {real['n_municipalities_with_monitoring']} МО и "
                        f"{real['n_observed_monitoring_months_per_method']} наблюдаемых месяцев после warmup.")
    sections += [
        "## Ограничения и следующий шаг",
        "Нет независимой разметки реальных шоков: реальная precision/recall не рассчитана, "
        "сигналы не названы подтверждёнными экономическими событиями. Короткий мониторинг, "
        "четыре калибровочные ошибки, Gaussian синтетика с фиксированным трендом и сезонностью, "
        "адаптация SeasonalNaiveYoY после изменения и cooldown влияют на результат. "
        "Выводы относятся к этому синтетическому протоколу; общего превосходства на реальных МО "
        "они не устанавливают. Прежний holdout просмотрен, реальная диагностика не стала новой "
        "независимой проверкой. Новости и предупреждение будущих изменений не реализованы.",
        "Следующий согласованный отдельный эксперимент — E05 «Тренд, инфляция и заработная плата». "
        "Он пока запланирован: проверить доступные на дату выпуска публикации/vintages факторов, "
        "компоненты отдельными опытами, избегать будущих фактов и двойного учёта YoY-роста; "
        "MAE в исходных номинальных рублях. Макропризнаки в E04a не подключены. "
        "Протокол независимой реальной разметки изменений остаётся отдельным решением.",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
