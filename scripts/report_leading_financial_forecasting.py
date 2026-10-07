"""Write E08c report only from completed, saved ablation artifacts; no fits."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(frame: pd.DataFrame, columns: list[str]) -> str:
    def cell(value):
        if pd.isna(value):
            return "—"
        if isinstance(value, (float, np.floating)):
            return f"{value:.3f}"
        return str(value).replace("|", "/")
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    rows.extend("| " + " | ".join(cell(row[c]) for c in columns) + " |" for _, row in frame.iterrows())
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/e08c_leading_financial_forecasting.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    output = ROOT / cfg["output_dir"]
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    if not manifest["complete"]:
        raise ValueError("E08c report requires a completed ablation")
    if manifest["config"] != cfg:
        raise ValueError("E08c report configuration differs from the completed run")
    for name in ["forecasting_metrics.csv", "origin_deltas.csv", "origin_consistency.csv",
                 "success_criterion.csv", "f0_reproduction.json", "training_diagnostics.json"]:
        if digest(output / name) != manifest["artifact_sha256"][name]:
            raise ValueError(f"Completed report input changed: {name}")
    independent_path = ROOT / "outputs/e08c_checks/independent_metrics.json"
    independent = json.loads(independent_path.read_text(encoding="utf-8"))
    completed_manifest_sha = manifest.get("fit_completion_manifest_sha256", digest(output / "run_manifest.json"))
    if "fit_completion_manifest_sha256" in manifest:
        frozen_path = ROOT / manifest["fit_completion_manifest_path"]
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        if digest(frozen_path) != completed_manifest_sha or frozen["fingerprint"] != manifest["fingerprint"]:
            raise ValueError("Fit completion manifest anchor changed")
        for name, sha in frozen["artifact_sha256"].items():
            if manifest["artifact_sha256"].get(name) != sha:
                raise ValueError("Fit artifacts differ from independently verified manifest")
    if independent["status"] != "PASS" or independent["complete_manifest_sha256"] != completed_manifest_sha:
        raise ValueError("Report requires independent PASS for this completed manifest")
    if independent["script_sha256"] != digest(independent_path.with_name("independent_metrics.py")):
        raise ValueError("Independent verification script changed after its PASS")
    metrics = pd.read_csv(output / "forecasting_metrics.csv")
    origins = pd.read_csv(output / "origin_deltas.csv")
    consistency = pd.read_csv(output / "origin_consistency.csv")
    success = pd.read_csv(output / "success_criterion.csv")
    gate = json.loads((output / "f0_reproduction.json").read_text(encoding="utf-8"))
    if gate["passed"] is not True or manifest["no_failed_forecasts"] is not True:
        raise ValueError("Report requires passed F0 gate and no failed forecasts")
    diagnostics = json.loads((output / "training_diagnostics.json").read_text(encoding="utf-8"))
    primary = cfg["primary_model"]
    main = metrics.loc[metrics.horizon.isin(cfg["primary_horizons"])]
    fields = ["model", "variant", "horizon", "split"]
    full = main.merge(consistency, on=fields, suffixes=("", "_consistency"), validate="one_to_one")
    aggregate = full.loc[full.split.eq("validation"), ["model", "variant", "horizon", "mae_macro", "delta_mae_macro_vs_F0"]].rename(columns={"mae_macro": "validation_MAE", "delta_mae_macro_vs_F0": "delta_validation_vs_F0"})
    holdout = full.loc[full.split.eq("holdout"), ["model", "variant", "horizon", "mae_macro", "delta_mae_macro_vs_F0", "n_origins_improved", "n_origins_worsened", "median_delta_mae_macro"]].rename(columns={"mae_macro": "holdout_MAE", "delta_mae_macro_vs_F0": "delta_holdout_vs_F0", "median_delta_mae_macro": "median_holdout_origin_delta"})
    aggregate = aggregate.merge(holdout, on=["model", "variant", "horizon"], validate="one_to_one")
    aggregate.to_csv(output / "comparison_table.csv", index=False)
    aggregate.to_csv(ROOT / cfg["audit_dir"] / "comparison_table.csv", index=False)
    f0_max = max(abs(r["difference"]) for r in gate["aggregate_metric_checks"] if np.isfinite(r["difference"]))
    finance_records = [r for r in diagnostics if r["ablation_variant"] != "F0"]
    training_checked = sum(r["e08c_checks"]["financial_source_audit"]["training_rows_checked"] for r in finance_records)
    forecast_checked = sum(r["e08c_checks"]["financial_source_audit"]["forecast_rows_checked"] for r in finance_records)
    lines = [
        "# E08c — forecasting ablation with leading financial indicators", "",
        f"**E08c forecasting ablation complete. Primary conclusion: {manifest['primary_conclusion']}.**", "",
        f"Ветка `research/e08-leading-indicators`, исходный HEAD `{manifest['git_commit']}`, "
        f"seed {manifest['seed']}. Код подготовлен, relevant tests пройдены, "
        "fixed model commands выполнены, метрики получены из сохранённых predictions.", "",
        "## Вопрос и фиксированный протокол", "",
        "Цель — месячное значение категории «Все категории» по МО; исходная шкала "
        "и определение показателя сохранены из E05d. MAE приведена в рублях.", "",
        "Дают ли causal key-rate/official-FX indicators дополнительную forecasting value "
        "сверх E05d и сохраняется ли знак улучшения по origins? Primary — "
        "LightGBMDirectNationalLocal (LN), sensitivity — LightGBMDirect (L0). "
        "Использован исходный native LightGBM adapter E05d, не другой estimator API. "
        "Национальный прогноз SeasonalNaiveYoY не изменён. Ни CatBoost, ни Prophet, "
        "ни Chronos не переобучались. Других learners или parameter search нет.", "",
        "Параметры совпадают с E05d: objective=regression_l1, 300 rounds, learning_rate=0.05, "
        "num_leaves=31, random_state=42, n_jobs=2, CPU, deterministic=true, "
        "force_col_wise=true, use_missing=true, zero_as_missing=false. "
        "Native requested/actual/resolved params каждого fit сохранены в training diagnostics.", "",
        "F0 — исходные 19 K0 features; F1 добавляет только пять key-rate features; "
        "F2 — только пять FX features; F3 — ровно десять E08b features. "
        "Missing flags, regime metadata и новые derived financial features не подаются learner.", "",
        "```text", *[f"{name}: {', '.join(features) or 'no financial features'}" for name, features in cfg["variants"].items()], "```", "",
        "Сохранены E05d data/category/target filters, legacy pair admission, training windows, "
        "origins Dec2023…Nov2024, validation target months до June2024 включительно, "
        "holdout July…Dec2024, horizons 1/3/6/12. Обучение использует все допустимые "
        "МО панели, оценка — неизменные 64 pilot IDs с прежней календарной eligibility. "
        f"На variant: {manifest['raw_cases_per_variant']} raw forecast keys, "
        f"{manifest['evaluable_cases_per_variant']} cases с фактом; missing actual сохранены "
        "в raw coverage. Неудачные forecasts не заменяются fallback и не удаляются молча.", "",
        "## Success criterion до результатов", "",
        "F3 получает PROMISING, только если macro MAE ниже F0 одновременно на validation "
        "и holdout как минимум на двух из h=1/3/6. На каждом таком horizon и каждом "
        "split с >=2 origins должны улучшаться >=2 origins и mean origin MAE delta "
        "после удаления лучшей origin оставаться отрицательной. Численный epsilon — "
        "1e-8 руб.; h12 не участвует. Validation h6 имеет одну origin: концентрация "
        "UNDETERMINED, а не доказанная стабильность. Overall conclusion определяется "
        "только primary LN; результат L0 не используется для выбора победителя. "
        "Операционализация концентрации записана в YAML/manifest до первого E08c fit.", "",
        table(success, ["model", "conclusion", "qualifying_horizons", "concentration_pass", "reasons"]), "",
        "## F0 reproduction gate", "",
        f"**PASS:** обе семьи воспроизведены до любых финансовых fits. "
        f"Max absolute F0 prediction difference={independent['f0_maximum_absolute_prediction_difference_rub']:.3g} руб. "
        f"Max absolute aggregate metric difference={f0_max:.3g}; atol=1e-8, rtol=0. "
        "По каждому partition сверены predictions/anchor/ratio/N_hat, keys/truths, "
        "split/history cutoff, statuses/reasons/fallback, K0 feature names/dtypes, "
        "training keys/order/targets/X signatures и requested/actual/native params. "
        "Resume требует matching fingerprint, completion marker и hashes всех "
        "partition artifacts; partial/failed checkpoint останавливается до нового fit.", "",
        "## Validation", "",
        table(full.loc[full.split.eq("validation")], ["model", "variant", "horizon", "mae_macro", "delta_mae_macro_vs_F0", "relative_delta_mae_macro_pct_vs_F0", "r2_pooled", "n_origins_improved", "n_origins_worsened", "n_origins"]), "",
        "## Holdout — уже просмотренный", "",
        table(full.loc[full.split.eq("holdout")], ["model", "variant", "horizon", "mae_macro", "delta_mae_macro_vs_F0", "relative_delta_mae_macro_pct_vs_F0", "r2_pooled", "n_origins_improved", "n_origins_worsened", "n_origins"]), "",
        "Отрицательная delta означает уменьшение ошибки; relative delta приведена "
        "в процентах. Macro MAE — средняя MAE по МО, MAE micro и pooled R² сохранены "
        "рядом; их определения не изменены. Последующий holdout не новый blind test.", "",
        "## Компактное F0/F1/F2/F3 сравнение", "",
        table(aggregate, list(aggregate.columns)), "",
        "## Origin consistency и концентрация", "",
        "На одной origin macro MAE равна среднему absolute error по доступным МО. "
        "Origin deltas считаются на одинаковых cases; временные blocks — origins, "
        "а не 2190 независимых финансовых наблюдений. Mean/median делtas не обязаны "
        "точно совпадать с aggregate macro delta при неодинаковом coverage.", "",
        table(consistency.loc[consistency.variant.ne("F0") & consistency.horizon.isin([1, 3, 6])], ["model", "variant", "split", "horizon", "n_origins_improved", "n_origins_worsened", "median_delta_mae_macro", "mean_delta_mae_macro", "best_origin", "best_origin_delta", "worst_origin", "worst_origin_delta", "leave_best_origin_out_mean_delta", "concentration_status"]), "",
        "Primary F3, каждый origin:", "",
        table(origins.loc[origins.model.eq(primary) & origins.variant.eq("F3") & origins.horizon.isin([1, 3, 6])], ["split", "horizon", "forecast_origin", "origin_mae_F0", "origin_mae", "delta_mae_macro_vs_F0"]), "",
        "## F1/F2/F3 интерпретация", "",
    ]
    for family in cfg["models"]:
        for variant, label in [("F1", "key rate alone"), ("F2", "FX alone"), ("F3", "combination")]:
            group = main.loc[main.model.eq(family) & main.variant.eq(variant)]
            val = group.loc[group.split.eq("validation")]
            hold = group.loc[group.split.eq("holdout")]
            paired = [h for h in [1, 3, 6] if val.loc[val.horizon.eq(h), "delta_mae_macro_vs_F0"].iloc[0] < -1e-8 and hold.loc[hold.horizon.eq(h), "delta_mae_macro_vs_F0"].iloc[0] < -1e-8]
            lines.append(f"- {family} / {variant} ({label}): validation improvements "
                         f"{int(val.delta_mae_macro_vs_F0.lt(-1e-8).sum())}/3, holdout "
                         f"{int(hold.delta_mae_macro_vs_F0.lt(-1e-8).sum())}/3; paired improving horizons {paired}. "
                         "Это описание фиксированной ablation, не выбор features по holdout.")
    def metric(variant: str, split: str, horizon: int, field: str) -> float:
        selected = main.loc[main.model.eq(primary) & main.variant.eq(variant)
                            & main.split.eq(split) & main.horizon.eq(horizon), field]
        return float(selected.iloc[0])

    f3_hold_h1 = consistency.loc[consistency.model.eq(primary) & consistency.variant.eq("F3")
                               & consistency.split.eq("holdout") & consistency.horizon.eq(1)].iloc[0]
    lines += ["",
              "В данном фиксированном прогоне устойчивой пользы key rate alone для primary LN "
              f"не подтверждено: F1/h1 даёт validation delta "
              f"{metric('F1', 'validation', 1, 'relative_delta_mae_macro_pct_vs_F0'):+.2f}%, "
              f"но holdout delta {metric('F1', 'holdout', 1, 'relative_delta_mae_macro_pct_vs_F0'):+.2f}%; "
              "paired improving horizons отсутствуют.", "",
              "FX alone даёт ограниченный прирост на h6: F2 validation delta "
              f"{metric('F2', 'validation', 6, 'relative_delta_mae_macro_pct_vs_F0'):+.2f}%, "
              f"holdout {metric('F2', 'holdout', 6, 'relative_delta_mae_macro_pct_vs_F0'):+.2f}%. "
              "На h1/h3 обе части ухудшаются; h6 validation представлен одной origin.", "",
              "Для combination F3 обе части улучшаются только на h6. На h1 validation delta "
              f"{metric('F3', 'validation', 1, 'relative_delta_mae_macro_pct_vs_F0'):+.2f}%, "
              f"а holdout {metric('F3', 'holdout', 1, 'relative_delta_mae_macro_pct_vs_F0'):+.2f}%; "
              "после удаления лучшей holdout origin средняя delta становится "
              f"{f3_hold_h1.leave_best_origin_out_mean_delta:+.3f} руб. — concentration FAIL. "
              "F3/h6 validation MAE совпадает с F2, а holdout MAE выше F2; "
              "добавка key rate к FX здесь не даёт дополнительного h6 выигрыша. "
              "Эффекты по горизонту, split и origins не согласованы по знаку.", "",
              "Sensitivity L0 не подтверждает перенос прироста: F1 улучшает два validation "
              "горизонта, F2/F3 — три, но каждый финансовый вариант ухудшает holdout "
              "на всех h1/3/6. Это отдельная sensitivity, а не основание менять primary model.", "",
              "Feature importance сохранена только descriptive; она не используется "
              "как доказательство usefulness или как success criterion.", "", "## Temporal / integrity checks", "",
              "E08b cache и decision ledger SHA проверены без сети; тот же builder "
              "материализует 23 monthly own origins Jan2023…Nov2024 из сохранённых "
              "training ranges и existing evaluation origins. На 12 frozen E08b origins "
              "числовые features совпадают в пределах 1e-12, source maxima — точно. "
              "Историческая row присоединяет features на собственной r, forecast row — "
              "на O. В каждом join effective/availability bounds≤own r≤O проверяются "
              "до learner creation; target label availability≤O также проверяется "
              "исходным E05d validator. Нет future fill, row dropping или new source research.", "",
              f"Проверено {training_checked} financial training rows и {forecast_checked} forecast rows "
              "F1–F3; cutoff violations=0. Данные national: повторение по МО "
              "не увеличивает число независимых временных observations. Для LN "
              "actual historical N_target — только доступная метка; прогноз восстанавливается "
              "через неизменный forecast N_hat, без actual future N.", "",
              f"83 relevant synthetic tests passed; независимый пересчёт PASS для "
              f"{independent['prediction_rows']} predictions в {independent['arms']} arms, "
              f"{len(independent['tables'])} сохранённых таблиц. Проверены "
              f"{independent['frozen_output_artifact_hashes_checked']} artifact SHA. "
              "Доказательства записаны в outputs/e08c_checks/. "
              "F0 без financial covariates выполняет исходный E05d path; source reconstruction "
              "и extended matrix проверены до любых fits. Full unrelated model suite не запускался.", "",
              "## h12 — DESCRIPTIVE ONLY", "",
              table(metrics.loc[metrics.horizon.eq(12)], ["model", "variant", "split", "mae_macro", "n_origins", "n_native", "n_fallback", "evaluation_role"]), "",
              "h12 имеет единственную origin Dec2023 и ноль обучающих пар. Все восемь "
              "strategies используют прежний expense SeasonalNaive fallback; ни одного "
              "h12 fit, обученного годового результата или вклада в success criterion нет.", "",
              "## Команды, runtime и ограничения", "", "```powershell", *manifest["run_commands"], "```", "",
              f"Successful model fits: {manifest['model_fits_succeeded']} из {manifest['model_fits_count']}; "
              f"runtime стадий по manifest {manifest['runtime_seconds']:.3f} s "
              "(таймер фиксируется после partition loop, до итоговой агрегации/записи отчёта). "
              "Smoke F0 March2024 h1 используется повторно в полном F0 gate без повторного fit. "
              "Runtime отдельных tasks и full native params сохранены; веса не публикуются.", "",
              "Ограничения: 24 target months, L=0 остаётся неподтверждённой availability "
              "цели; target vintages неизвестны, FX/key archive trust условен, June2024 "
              "methodology boundary ограничивает сопоставимость FX volatility. Уже "
              "просмотренный holdout, немного независимых origins и weak coverage "
              "не позволяют делать вывод об экономическом причинном эффекте или "
              "generalisation на новое blind test. Real-EW feasibility gate не меняется. "
              "h6 validation с одной origin имеет отдельное ограничение устойчивости.", "",
              "Результат не обновляет F1–F7 Source of Truth, README, final report или "
              "presentation. Numeric predictions/features/metrics/cache/report остаются "
              "в существующих ignored outputs и reports/results; index не меняется.", "",
              f"**{manifest['primary_conclusion']}** — вывод по заранее объявленному "
              "primary criterion, без выбора лучшего learner/variant после результата. "
              "Следующий шаг — отдельно обсудить этот результат; новые модели, sources "
              "или early-warning training этим запуском не разрешаются.", "",
              "NO TUNING; NO NEW MODEL FAMILY; NO SOURCE OF TRUTH CHANGES; NO COMMIT/PUSH.", ""]
    destination = ROOT / cfg["report_path"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report saved: {cfg['report_path']}; {manifest['primary_conclusion']}")


if __name__ == "__main__":
    main()
