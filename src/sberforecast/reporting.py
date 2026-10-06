"""Сводные файлы и отдельные графики. Метрики берутся только из реальных прогнозов."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .metrics import attach_split, common_support, compute_metrics, coverage_table, select_on_validation, compare_with_baseline, selected_policy_predictions


def md_table(frame: pd.DataFrame) -> str:
    # Без необязательной зависимости tabulate.
    lines = ["| " + " | ".join(map(str, frame.columns)) + " |", "| " + " | ".join(["---"]*len(frame.columns)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        vals = [f"{v:.2f}" if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def summarize(output: Path, cfg: dict, complete: bool) -> tuple[pd.DataFrame, dict]:
    paths = sorted((output/"partitions").glob("predictions_*.csv.gz"))
    if not paths:
        raise ValueError("Нет сохранённых прогнозов.")
    all_predictions = pd.concat([pd.read_csv(p, dtype={"municipality_id":str}) for p in paths], ignore_index=True)
    all_predictions = attach_split(all_predictions, cfg["backtest"]["validation_target_end"])
    common = common_support(all_predictions, cfg["models"]["enabled"])
    metrics, per_mo = compute_metrics(common)
    coverage = coverage_table(all_predictions)
    selection = select_on_validation(metrics, cfg)
    all_predictions.to_csv(output/"predictions.csv.gz", index=False)
    metrics.to_csv(output/"metrics.csv", index=False)
    per_mo.to_csv(output/"metrics_by_municipality.csv.gz", index=False)
    coverage.to_csv(output/"coverage.csv", index=False)
    (output/"selection.json").write_text(json.dumps(selection,ensure_ascii=False,indent=2),encoding="utf-8")
    for baseline in ["SeasonalNaive", "ProphetAuto", "ProphetYearly"]:
        if baseline in cfg["models"]["enabled"]:
            compare_with_baseline(per_mo, baseline).to_csv(output/f"comparison_vs_{baseline}.csv",index=False)
    origin_rows = []
    for (model,horizon,origin),g in common.groupby(["model","horizon","forecast_origin"]):
        origin_rows.append({"model":model,"horizon":horizon,"forecast_origin":origin,
                            "n":len(g),"mae":float(np.abs(g.y_true-g.y_pred).mean())})
    pd.DataFrame(origin_rows).to_csv(output/"metrics_by_origin.csv",index=False)
    holdout = metrics[metrics.split.eq("holdout")]
    pivot = holdout.pivot(index="model", columns="horizon", values="mae_macro").reset_index()
    pivot.columns = ["Модель"] + [f"{h} мес., MAE" for h in pivot.columns[1:]]
    policy = selected_policy_predictions(common, selection)
    policy.to_csv(output/"selected_policy_predictions.csv.gz", index=False)
    policy_metrics, _ = compute_metrics(policy)
    policy_metrics.to_csv(output/"selected_policy_metrics.csv", index=False)
    chosen_rows = []
    for h,item in selection.items():
        r = policy_metrics[(policy_metrics.horizon==int(h)) & policy_metrics.model.eq(item["model"])]
        chosen_rows.append({"Горизонт":h,"Выбрано до holdout":item["model"],"Основание":item["selection"], "Выбор известен с":item["selection_available_at"],
                            "Число точек выпуска":int(r.n_origins.iloc[0]) if len(r) else 0,
                            "MAE после выбора":float(r.mae_macro.iloc[0]) if len(r) else np.nan})
    prophet_note = ("5. Prophet включён в текущую конфигурацию; фактическое покрытие и ошибки исполнения см. coverage.csv и errors_*.json. Это не точная базовая модель организаторов и не полноценная настройка гиперпараметров на валидации."
                    if any(m.startswith("Prophet") for m in cfg["models"]["enabled"]) else
                    "5. Prophet: адаптер написан, но в этом эксперименте модель НЕ запускалась. В исходной среде не удалось установить пакет из-за сетевого ограничения. Сравнения с Prophet, результатов фундаментальных моделей и эффекта новостей ещё нет.")
    text = ["# Первый эксперимент: фактические результаты", "",
            f"Статус: {'все заданные точки выпуска рассчитаны' if complete else 'НЕПОЛНЫЙ прогон — не использовать для окончательных выводов'}.",
            "", "## Что сравнивается", "",
            "Цель — месячное значение категории «Все категории», в исходных рублях. Это не сумма за горизонт и не совокупное потребление муниципалитета.",
            "Все модели сравниваются на одинаковых МО, точках выпуска и доступных фактах. Сезонный резервный прогноз помечен в status; ошибки исполнения моделей и отсутствие факта не скрываются.",
            f"Всего записей прогнозов: {len(all_predictions):,}; строк на общей области с известной истиной: {len(common):,}.",
            "", "## MAE с равным весом каждого МО на временном holdout", "", md_table(pivot), "",
            "Периоды целей holdout: июль–декабрь 2024. Модель на каждой дате обучается на уже открывшейся истории; гиперпараметры фиксированы. Это последовательная проверка, не один прогноз всех шести месяцев в июне.",
            "", "## Выбор только на валидации и применение только после даты выбора", "", md_table(pd.DataFrame(chosen_rows)), "",
            "Валидация задаётся по месяцу ЦЕЛИ, не по строкам и не случайным разбиением: доступные цели по июнь 2024 включительно. Для горизонта 12 месяцев таких пар нет, поэтому используется заранее заданный SeasonalNaive, а не победитель по тесту.",
            "",
            "Выбор по валидации до июня становится доступен только в конце июня (с поправкой на условный лаг). Его нельзя задним числом применить к прогнозам, выпущенным в январе–мае. Поэтому selected_policy_metrics.csv оценивает выбранные модели только для точек выпуска не раньше даты выбора. При нулевом лаге остаётся 6 точек для h=1, 4 для h=3 и 1 для h=6. Для h=12 модель заранее назначена без валидации. Полная таблица выше — сравнение отдельных заранее зафиксированных алгоритмов, не результат такой стратегии выбора.",
            "", "## Ограничения и правильное прочтение", "",
            "1. История — 24 месяца. Для h=12 доступна одна точка выпуска: декабрь 2023 → декабрь 2024. Отдельной валидации h=12 нет; для h=6 на валидации только одна точка выпуска. Это недостаточно для устойчивого рейтинга длинных горизонтов.",
            f"2. Даты публикации и исторические версии расходов неизвестны. Применён НЕПРОВЕРЕННЫЙ сценарий release_lag_months={cfg['data']['release_lag_months']}; это ретроспективная симуляция доступности, не доказанный real-time backtest.",
            "3. MAE macro — среднее муниципальных MAE, MAE micro — среднее ошибок всех прогнозных пар. Пуловый R² может выглядеть высоким из-за разницы уровней МО; это не доказательство способности предсказывать динамику каждого МО.",
            "4. Нет независимой разметки структурных изменений. Диагностические CUSUM/EWMA не имеют подтверждённых precision/recall и не предсказывают будущие шоки.",
            prophet_note,
            "6. Новости, Росстат, market_access и географические поля не включены в признаки. Внешняя информация с неподтверждённой исторической доступностью не подставлялась.",
            "7. Модель CatBoost обучается на всех доступных МО, локальные ориентиры — на истории каждого МО. Это осознанное сравнение локального и общего подхода, а не равенство наборов обучающих рядов.",
            "8. Настройки не подбирались по holdout. После просмотра этих результатов будущие итерации уже не смогут называть этот же период совершенно нетронутым тестом. Доверительные интервалы и значимость различий не оценивались.",
            "9. При отсутствии нужного сезонного наблюдения SeasonalNaive использует последнее известное/рекурсивное значение, не удаляя строку. Детали в coverage.csv. Временно недопустимые МО перечислены в cohort_*.csv.",
            "", "## Артефакты", "",
            "metrics.csv — MAE macro/micro, R² и размеры выборок; metrics_by_origin.csv — стабильность по датам; metrics_by_municipality.csv.gz — качество по МО; coverage.csv — покрытие и резервы; predictions.csv.gz — все прогнозные пары; selection.json — выбранная на валидации модель; run_manifest.json — конфигурация, версии и контрольные суммы.", ""]
    (output/"experiment_report.md").write_text("\n".join(text),encoding="utf-8")
    return all_predictions, selection


def make_plots(output: Path, panel: pd.DataFrame, predictions: pd.DataFrame, selection: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    metrics = pd.read_csv(output/"metrics.csv")
    h = metrics[metrics.split.eq("holdout")]
    fig,ax = plt.subplots(figsize=(10,5.5))
    for name,g in h.groupby("model",sort=True):
        g=g.sort_values("horizon")
        ax.plot(g.horizon,g.mae_macro,marker="o",label=name)
    ax.set(title="Первое сравнение: MAE на июле–декабре 2024",xlabel="Горизонт, календарных месяцев",ylabel="MAE, руб. / равный вес МО")
    ax.set_xticks([1,3,6,12]); ax.grid(alpha=0.25); ax.legend()
    fig.text(0.5,0.015,"12 месяцев: только декабрь 2023 → декабрь 2024; одна точка выпуска, без отдельной валидации.",ha="center",fontsize=9)
    fig.tight_layout(rect=[0,0.04,1,1]); fig.savefig(output/"mae_by_horizon.png",dpi=160); plt.close(fig)
    # Пример выбирается по идентификатору среди допустимых на первой дате,
    # а не по качеству будущих прогнозов.
    first = predictions.forecast_origin.min()
    ids = predictions.loc[predictions.forecast_origin.eq(first),"municipality_id"].unique()
    uid = sorted(ids,key=lambda x:int(x) if str(x).isdigit() else str(x))[0]
    model=selection["1"]["model"]
    p=predictions[(predictions.municipality_id==uid)&(predictions.model==model)&(predictions.horizon==1)].sort_values("target_period")
    fig,ax=plt.subplots(figsize=(10,5.5))
    ax.plot(panel.index,panel[str(uid)],marker="o",label="Факт")
    ax.plot(pd.to_datetime(p.target_period),p.y_pred,marker="s",linestyle="--",label=f"{model}, h=1")
    ax.axvline(pd.Timestamp("2024-07-01"),linestyle=":",label="Начало holdout")
    ax.set(title=f"МО {uid}: прогноз на месяц с ежемесячным переобучением",xlabel="Месяц",ylabel="Средние безналичные расходы, руб.")
    ax.grid(alpha=0.25);ax.legend();fig.autofmt_xdate();fig.tight_layout()
    fig.savefig(output/"example_rolling_forecast.png",dpi=160);plt.close(fig)
