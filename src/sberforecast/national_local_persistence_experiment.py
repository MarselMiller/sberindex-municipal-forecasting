"""Одна заранее заданная National/Local ablation на неизменных случаях E05d.

Построчные прогнозы и диагностика остаются в ignored outputs. Публичны только
агрегаты ошибок, календарные сравнения и verification metadata.
"""
from __future__ import annotations

from contextlib import ExitStack
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import time
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

from .data import sha256_file
from .direct_experiment import project_path
from .direct_evaluation import require_same_keys
from .metrics import KEY
from .national_local_persistence_gate import load_verified_sources
from .trend_calendar_evaluation import evaluate_group

PERSISTENCE = "NationalLocalPersistence"
LEARNED = "NationalLocalLightGBM"
DIRECT = "LightGBMDirect"
MODELS = ["LastValue", "SeasonalNaiveYoY", DIRECT, PERSISTENCE, LEARNED]
LABELS = {PERSISTENCE: "National/Local Persistence", LEARNED: "National/Local + LightGBM"}
CONTRASTS = [("LastValue", PERSISTENCE), ("SeasonalNaiveYoY", PERSISTENCE),
             (DIRECT, PERSISTENCE), (PERSISTENCE, LEARNED)]


def validate_protocol(cfg: dict) -> None:
    fixed = {
        "ratio_rule": "last_causal_available_ratio", "ratio_transformations": [], "seed": 42,
        "reference_config": "configs/national_local_lightgbm.yaml",
        "reference_output": "outputs/national_local_lightgbm_v1",
        "primary_horizons": [1,3,6], "descriptive_horizons": [12],
        "reference_tolerance": 1e-8, "local_model_fits": 0, "new_reference_fits": 0,
        "h12_policy": "fixed_persistence_rule_without_training_pair_requirement",
        "failure_policy": "unchanged_NL_batch_failure_no_replacement",
        "national_prediction_guard": "strictly_positive_finite",
        "holdout_status": "previously_inspected_not_independent",
        "output_dir": "outputs/national_local_persistence_v1",
        "smoke_output_dir": "outputs/national_local_persistence_smoke_v1",
        "smoke_origin": "2024-03",
        "public_dir": "reports/results/national_local_persistence",
        "report_path": "reports/results/national_local_persistence.md",
    }
    for name, value in fixed.items():
        if cfg.get(name) != value:
            raise ValueError(f"Pre-specified protocol changed: {name}.")
    rules = cfg["interpretation"]
    if (rules["substantial_improvement_pct"], rules["closeness_to_learned_pct"],
            rules["minimum_share_for_most"], rules["origin_majority"]) != (5.0, 5.0, 0.5, "strictly_more_than_half"):
        raise ValueError("Interpretation thresholds must be fixed before results.")
    if rules["origin_direction_check"] != "improvements_and_deteriorations" or rules["priority"] != ["D_instability", "A", "B", "C", "D_other_mixed"]:
        raise ValueError("Interpretation order and origin stability must remain fixed.")


def save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False,
                              default=lambda x: x.item() if isinstance(x, np.generic) else str(x)),
                    encoding="utf-8")


def join_forecasts(expected: pd.DataFrame, forecasts: pd.DataFrame) -> pd.DataFrame:
    """Preserve canonical evaluation keys instead of merging duplicated fields."""
    current = forecasts.copy()
    for field in ("forecast_origin", "target_period"):
        current[field] = pd.to_datetime(current[field]).dt.strftime("%Y-%m-%d")
    require_same_keys(current, expected)
    return expected.merge(current, on=KEY, validate="one_to_one")


def gap_analysis(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split, horizon), group in metrics.groupby(["split", "horizon"], sort=True):
        indexed = group.set_index("model")
        for earlier, later in CONTRASTS:
            left, right = indexed.loc[earlier], indexed.loc[later]
            complete = left.metric_status == right.metric_status == "complete"
            delta = float(left.mae_macro - right.mae_macro) if complete else np.nan
            share, state = np.nan, "not_applicable"
            if earlier == DIRECT:
                denominator = float(left.mae_macro - indexed.loc[LEARNED, "mae_macro"])
                if complete and indexed.loc[LEARNED, "metric_status"] == "complete" and denominator > 0:
                    share, state = delta / denominator, "estimable_descriptive"
                else:
                    state = "nonpositive_or_unavailable_denominator"
            rows.append(dict(split=split, horizon=int(horizon), from_model=earlier, to_model=later,
                             mae_from=float(left.mae_macro), mae_to=float(right.mae_macro),
                             reduction_mae=delta,
                             reduction_pct=100 * delta / left.mae_macro if complete and left.mae_macro > 0 else np.nan,
                             observed_share_of_mae_gap=share, share_status=state,
                             metric_status="complete" if complete else "incomplete",
                             descriptive_only=int(horizon) == 12))
    return pd.DataFrame(rows)


def origin_diagnostics(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    deltas = []
    for (split, horizon, origin), group in metrics.groupby(["split", "horizon", "forecast_origin"], sort=True):
        indexed = group.set_index("model")
        for earlier, later in CONTRASTS:
            left, right = indexed.loc[earlier], indexed.loc[later]
            complete = left.metric_status == right.metric_status == "complete"
            delta = float(left.mae_macro - right.mae_macro) if complete else np.nan
            deltas.append(dict(split=split, horizon=int(horizon), forecast_origin=origin,
                               from_model=earlier, to_model=later, n_rows=int(right.n_predictions),
                               mae_from=float(left.mae_macro), mae_to=float(right.mae_macro),
                               reduction_mae=delta,
                               reduction_pct=100 * delta / left.mae_macro if complete and left.mae_macro > 0 else np.nan,
                               metric_status="complete" if complete else "incomplete",
                               descriptive_only=int(horizon) == 12))
    frame = pd.DataFrame(deltas)
    summary = []
    for keys, group in frame.groupby(["split", "horizon", "from_model", "to_model"], sort=True):
        good = group.loc[np.isfinite(group.reduction_mae)]
        best = good.loc[good.reduction_mae.idxmax()] if len(good) else None
        worst = good.loc[good.reduction_mae.idxmin()] if len(good) else None
        summary.append(dict(zip(["split", "horizon", "from_model", "to_model"], keys),
            n_origins=len(group), n_evaluable_origins=len(good),
            n_improved=int(good.reduction_mae.gt(0).sum()),
            n_worsened=int(good.reduction_mae.lt(0).sum()), n_tied=int(good.reduction_mae.eq(0).sum()),
            mean_origin_reduction=float(good.reduction_mae.mean()),
            median_origin_reduction=float(good.reduction_mae.median()),
            best_origin=str(best.forecast_origin) if best is not None else "",
            best_reduction=float(best.reduction_mae) if best is not None else np.nan,
            worst_origin=str(worst.forecast_origin) if worst is not None else "",
            worst_reduction=float(worst.reduction_mae) if worst is not None else np.nan,
            descriptive_only=int(keys[1]) == 12))
    return frame, pd.DataFrame(summary)


def municipality_diagnostics(metrics_by_municipality: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split, horizon), group in metrics_by_municipality.groupby(["split", "horizon"], sort=True):
        panel = group.pivot(index="municipality_id", columns="model", values="mae")
        for reference in ["SeasonalNaiveYoY", DIRECT, LEARNED]:
            delta = panel[reference] - panel[PERSISTENCE]
            known = np.isfinite(delta)
            rows.append(dict(split=split, horizon=int(horizon), model=PERSISTENCE, reference=reference,
                             n_municipalities=len(panel), n_evaluable=int(known.sum()),
                             n_better=int(delta.gt(0).sum()), n_worse=int(delta.lt(0).sum()),
                             n_tied=int(delta.eq(0).sum()),
                             share_better=float(delta[known].gt(0).mean()),
                             median_municipality_reduction=float(delta[known].median()),
                             descriptive_only=int(horizon) == 12))
    return pd.DataFrame(rows)


def classify(metrics: pd.DataFrame, origins: pd.DataFrame, gaps: pd.DataFrame, rules: dict) -> dict:
    """Descriptive categories fixed before results; no significance test."""
    per_horizon = []
    for horizon in (1, 3, 6):
        m = metrics.loc[metrics.horizon.eq(horizon)].set_index(["split", "model"])
        g = gaps.loc[gaps.horizon.eq(horizon)].set_index(["split", "from_model", "to_model"])
        o = origins.loc[origins.horizon.eq(horizon)].set_index(["split", "from_model", "to_model"])
        unstable = False
        for earlier, later in CONTRASTS:
            reductions = [float(g.loc[(split, earlier, later), "reduction_mae"]) for split in ("validation", "holdout")]
            if not np.isfinite(reductions).all() or np.sign(reductions[0]) != np.sign(reductions[1]):
                unstable = True
            for split, reduction in zip(("validation", "holdout"), reductions):
                row = o.loc[(split, earlier, later)]
                if reduction > 0 and row.n_improved <= row.n_origins / 2:
                    unstable = True
                if reduction < 0 and row.n_worsened <= row.n_origins / 2:
                    unstable = True
        substantial = all(g.loc[(split, earlier, PERSISTENCE), "reduction_pct"] >= rules["substantial_improvement_pct"]
                          for split in ("validation", "holdout") for earlier in ("LastValue", "SeasonalNaiveYoY"))
        little_benefit = all(g.loc[(split, earlier, PERSISTENCE), "reduction_pct"] < rules["substantial_improvement_pct"]
                             for split in ("validation", "holdout") for earlier in ("LastValue", "SeasonalNaiveYoY"))
        close = all(m.loc[(split, PERSISTENCE), "mae_macro"] <=
                    m.loc[(split, LEARNED), "mae_macro"] * (1 + rules["closeness_to_learned_pct"] / 100)
                    for split in ("validation", "holdout"))
        most = all(np.isfinite(g.loc[(split, DIRECT, PERSISTENCE), "observed_share_of_mae_gap"]) and
                   g.loc[(split, DIRECT, PERSISTENCE), "observed_share_of_mae_gap"] >= rules["minimum_share_for_most"]
                   for split in ("validation", "holdout"))
        learned_value = all(g.loc[(split, PERSISTENCE, LEARNED), "reduction_pct"] >= rules["substantial_improvement_pct"]
                            for split in ("validation", "holdout"))
        learned_raw = all((m.loc[(split, baseline), "mae_macro"] - m.loc[(split, LEARNED), "mae_macro"]) /
                          m.loc[(split, baseline), "mae_macro"] * 100 >= rules["substantial_improvement_pct"]
                          for split in ("validation", "holdout") for baseline in ("LastValue", "SeasonalNaiveYoY", DIRECT))
        category = "D" if unstable else "A" if substantial and close and most else "B" if substantial and learned_value else "C" if little_benefit and learned_value and learned_raw else "D"
        per_horizon.append(dict(horizon=horizon, category=category, instability=bool(unstable),
                                substantial_simple_baseline_benefit=bool(substantial), close_to_learned=bool(close),
                                little_simple_baseline_benefit=bool(little_benefit),
                                closes_most_positive_direct_gap=bool(most), learned_adds_value=bool(learned_value)))
    categories = {row["category"] for row in per_horizon}
    overall = next(iter(categories)) if len(categories) == 1 and "D" not in categories else "D"
    return dict(overall=overall, label=rules["categories"][overall], per_horizon=per_horizon, rules=rules)


def markdown(frame: pd.DataFrame) -> str:
    def fmt(value):
        if pd.isna(value):
            return "NA"
        if isinstance(value, (float, np.floating)):
            return f"{value:.2f}"
        return str(value)
    return "\n".join(["| " + " | ".join(map(str, frame.columns)) + " |",
                       "| " + " | ".join(["---"] * len(frame.columns)) + " |"] +
                      ["| " + " | ".join(fmt(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None)])


def write_report(path: Path, tables: dict, manifest: dict) -> None:
    metrics, gaps, stability = tables["metrics"], tables["gap_analysis"], tables["origin_stability"]
    parts = ["# National/Local Persistence: фиксированная методологическая ablation\n",
        "## Исследовательский вопрос\n\nПроверить, какая часть улучшения National/Local + LightGBM связана с самим разложением ряда на общую национальную и локальную компоненты, а какая — с обучаемой моделью локальной динамики.\n",
        "## Существующее разложение National/Local\n\nN_t — медиана конечных значений полной панели всех доступных МО, а не только пилотной выборки; R_i,t = y_i,t / N_t при конечном положительном N_t. Переиспользованы без изменений build_national, ratio_panel и national_forecast. Национальный прогноз — прежний SeasonalNaiveYoY с теми же параметрами роста и шагом h+L. Протокол, sample, eligibility, факты, split и ключи взяты из [конфигурации E05d](../../configs/national_local_lightgbm.yaml) и сверены с сохранёнными predictions.\n",
        "## Определение Persistence\n\nДля каждого origin используется последний конечный causal-доступный R по t≤O−L; y_hat = N_hat × R_last для всех h. Пропущенный последний ratio заменяется только более ранним известным ratio, без сглаживания или нового ограничения давности. Нет обучения, tuning, clipping ratio и поиска признаков. Невалидный национальный прогноз или отсутствие anchor сохраняют прежнюю failed-политику всей партии, без подстановки другой модели.\n\nУ Persistence нет требования наличия обучающих пар. Поэтому на h=12 он применяет ту же формулу, а LightGBMDirect и National/Local + LightGBM сохраняют SeasonalNaive fallback. h=12 исключён из основного вывода.\n",
        "## Reproduction gate\n\n**PASS.** Использован пересчёт сохранённых predictions, а не новый fit LightGBM. Проверены source SHA, параметры/seed, фиксированные ключи и факты, strategy/native метрики и origin-метрики, национальные forecasts. Численный допуск — 1e−8; подробности находятся в [run_manifest.json](national_local_persistence/run_manifest.json).\n"]
    for split, title in [("validation", "Validation"), ("holdout", "Holdout")]:
        current = metrics.loc[metrics.split.eq(split) & metrics.horizon.isin([1, 3, 6])]
        pivot = current.pivot(index="model", columns="horizon", values="mae_macro").reindex(MODELS)
        pivot.index = [LABELS.get(name, name) for name in pivot.index]
        pivot.columns = ["h1", "h3", "h6"]
        parts += [f"## {title}: MAE macro, исходные рубли\n\n" + markdown(pivot.rename_axis("Strategy").reset_index()) + "\n"]
    parts += ["Все стратегии оценены на прежних конечных фактах. Missing/failed не исключаются для улучшения метрик; coverage и статусы сохранены в [metrics.csv](national_local_persistence/metrics.csv). h12 и pooled R² приведены в этом же CSV.\n",
        "## Устойчивость по forecast origins\n\nПоложительный Δ означает снижение MAE при переходе from → to. Каждая календарная origin учитывается один раз; среднее origin-Δ не подменяет общую macro MAE.\n\n" + markdown(stability.loc[stability.horizon.isin([1,3,6]), ["split", "horizon", "from_model", "to_model", "n_origins", "n_evaluable_origins", "n_improved", "mean_origin_reduction", "median_origin_reduction", "best_origin", "worst_origin"]]) + "\n",
        "Полные best/worst Δ и counts доступны в [origin_stability.csv](national_local_persistence/origin_stability.csv), отдельные даты — в [origin_deltas.csv](national_local_persistence/origin_deltas.csv).\n",
        "## Диагностика по МО\n\nОписательная доля МО, где Persistence имеет меньшую MAE; равенства не считаются выигрышами. Это не significance test. Идентификаторы и построчные расходы не публикуются.\n\n" + markdown(tables["municipality_win_rates"].loc[lambda f: f.horizon.isin([1,3,6]), ["split", "horizon", "reference", "n_municipalities", "n_evaluable", "n_better", "share_better", "median_municipality_reduction"]]) + "\n",
        "## Decomposition gap analysis\n\nΔMAE = MAE_from − MAE_to, relative Δ = 100×ΔMAE/MAE_from. Доля observed MAE gap = (MAE_Direct−MAE_Persistence)/(MAE_Direct−MAE_NL_LGBM), только при положительном знаменателе; отрицательные значения и значения >1 не обрезаются. Это описательная доля наблюдаемого разрыва, не causal contribution.\n\n" + markdown(gaps.loc[gaps.horizon.isin([1,3,6]), ["split", "horizon", "from_model", "to_model", "reduction_mae", "reduction_pct", "observed_share_of_mae_gap"]]) + "\n",
        "Полная точность: [gap_analysis.csv](national_local_persistence/gap_analysis.csv).\n",
        f"## Категория интерпретации\n\n**{manifest['interpretation']['overall']}: {manifest['interpretation']['label']}**.\n\n",
        "Правило зафиксировано в [новой конфигурации](../../configs/national_local_persistence.yaml) до расчёта baseline: существенное улучшение ≥5% к LastValue и SeasonalNaiveYoY; близость к learned MAE — не хуже более чем на 5%; для «most» требуется закрыть ≥50% положительного Direct→NL gap. Строгое большинство origins — >1/2. Сначала D при смене направления между split либо отсутствии строгого большинства origins, согласующихся с направлением агрегированной разницы. Далее A требует существенного улучшения простых baselines, близости к learned MAE и большинства положительного gap; B — такого же улучшения и дополнительного снижения MAE с learner минимум на 5%; C — улучшения менее 5% к каждому простому baseline в обоих split и снижения MAE с learner минимум на 5% относительно Persistence и raw baselines. Остальные смешанные случаи относятся к D. Общая A/B/C допустима только при одинаковой категории на всех h=1/3/6, иначе D. Это описательная классификация, не статистическая значимость.\n\n" + markdown(pd.DataFrame(manifest["interpretation"]["per_horizon"])) + "\n",
        "## Ограничения\n\nHoldout уже просмотрен; этот эксперимент не является новым blind test. История целевого показателя — 24 месяца (2023–2024), сохранена пилотная выборка и категория «Все категории». Доступность при release_lag_months=0 остаётся допущением. Для validation h=6 есть одна origin, для holdout h=1/3/6 — шесть; h12 имеет одну origin и только описательное значение. Число МО не создаёт независимые временные наблюдения. Разложение включает national forecast и сохраняемое относительное положение; доля MAE gap не изолирует causal contribution. Пересчёт saved predictions подтверждает метрики и артефакты, но не является свежим переобучением reference learners.\n",
        f"Новых model fits: **0**. Runtime полного запуска: **{manifest['runtime_seconds']:.3f} s**. Seed=42; версии и source hashes сохранены в manifest. Построчные predictions, ratio-provenance, resolved config и диагностические таблицы остаются в ignored output_dir.\n",
        "[Агрегаты по МО](national_local_persistence/municipality_win_rates.csv) · [Метаданные запуска](national_local_persistence/run_manifest.json)\n"]
    path.write_text("\n".join(parts), encoding="utf-8")


def run_experiment(root: Path, config_path: Path, *, gate_only: bool = False,
                   smoke: bool = False, command: str) -> dict:
    started = time.perf_counter()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=root, text=True).strip()
    if branch != "research/national-local-persistence" or cfg["branch"] != branch:
        raise ValueError("Experiment requires research/national-local-persistence.")
    validate_protocol(cfg)
    fit_attempts = []
    def forbid_fit(*args, **kwargs):
        fit_attempts.append(True)
        raise RuntimeError("This experiment does not permit model fits.")
    with ExitStack() as stack:
        stack.enter_context(patch("catboost.CatBoostRegressor.fit", forbid_fit))
        stack.enter_context(patch("sberforecast.national_local_model.NativeLightGBMRegressor.fit", forbid_fit))
        sources = load_verified_sources(root, project_path(root, cfg["reference_config"]))
        gate = sources["gate"]
        if gate.get("status") != "PASS":
            raise ValueError("Reference reproduction gate must PASS before Persistence.")
        checks = root / "outputs/national_local_persistence_checks"
        checks.mkdir(parents=True, exist_ok=True)
        save_json(checks / "reproduction_gate.json", gate)
        if gate_only:
            print(json.dumps({key: gate[key] for key in ("status", "verification_mode", "model_fits", "raw_requested_keys", "finite_target_keys", "historical_artifacts_verified", "source_hashes_verified")}, ensure_ascii=False), flush=True)
            return gate
        output = project_path(root, cfg["smoke_output_dir"] if smoke else cfg["output_dir"])
        if not output.is_relative_to(root / "outputs") or output.exists():
            raise FileExistsError("A new, unused ignored output directory is required.")
        from .national_local_persistence_model import NationalLocalPersistence
        predictor = NationalLocalPersistence(sources["national"], sources["cfg"]["data"]["release_lag_months"])
        expected = sources["expected"]
        if smoke:
            expected = expected.loc[expected.forecast_origin.str.startswith(cfg["smoke_origin"])].copy()
        records, national_records = [], []
        deterministic_checks = 0
        for (origin, horizon), current in expected.groupby(["forecast_origin", "horizon"], sort=True):
            ids = current.municipality_id.tolist()
            result = predictor.predict(sources["panel"], pd.Period(origin, freq="M"), int(horizon), ids, sources["cfg"]["models"])
            repeat = predictor.predict(sources["panel"], pd.Period(origin, freq="M"), int(horizon), ids, sources["cfg"]["models"])
            pd.testing.assert_frame_equal(result.forecasts, repeat.forecasts, check_exact=True)
            deterministic_checks += 1
            records.append(join_forecasts(current, result.forecasts).assign(model=PERSISTENCE))
            national_records.append(result.national_forecast)
        persistence = pd.concat(records, ignore_index=True)
        references = sources["predictions"].merge(expected[KEY], on=KEY, how="inner", validate="many_to_one").copy()
        references["model"] = references.model.replace({"L0": DIRECT, "LN": LEARNED})
        predictions = pd.concat([references, persistence], ignore_index=True)
        evaluated = evaluate_group(predictions, expected, MODELS, [DIRECT, PERSISTENCE, LEARNED])
    assert not fit_attempts
    output.mkdir(parents=True)
    predictions.to_csv(output / "predictions.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    pd.DataFrame(national_records).to_csv(output / "national_forecasts.csv", index=False)
    for name, table in evaluated.items():
        table.to_csv(output / f"{name}.csv", index=False)
    metrics = evaluated["metrics_strategy"].merge(evaluated["coverage"].rename(columns={"n_requested": "n_requested_all"})[
        ["split", "model", "horizon", "n_requested_all", "forecast_coverage"]], on=["split", "model", "horizon"], validate="one_to_one")
    metrics["n_rows"] = metrics.n_predictions
    metrics["descriptive_only"] = metrics.horizon.eq(12)
    gaps = gap_analysis(metrics)
    origin_deltas, origin_stability = origin_diagnostics(evaluated["metrics_strategy_by_origin"])
    municipality = municipality_diagnostics(evaluated["metrics_strategy_by_municipality"])
    tables = {"metrics": metrics, "gap_analysis": gaps, "origin_deltas": origin_deltas,
              "origin_stability": origin_stability, "municipality_win_rates": municipality}
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False)
    interpretation = classify(metrics, origin_stability, gaps, cfg["interpretation"]) if not smoke else {"overall": "not_classified_smoke"}
    (output / "config_resolved.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    code_paths = [config_path, root / "scripts/run_national_local_persistence.py"] + sorted((root / "src/sberforecast").glob("national_local_persistence*.py")) + sorted((root / "tests").glob("test_national_local_persistence*.py"))
    manifest = dict(experiment=cfg["experiment"], mode="smoke" if smoke else "full",
        created_at_utc=datetime.now(timezone.utc).isoformat(), seed=cfg["seed"], command=command,
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        git_branch=branch, git_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()),
        python=platform.python_version(), versions={name: importlib.metadata.version(name) for name in ("numpy", "pandas", "PyYAML", "catboost", "lightgbm")},
        config=cfg, config_sha256=sha256_file(config_path),
        code_sha256={p.relative_to(root).as_posix(): sha256_file(p) for p in code_paths},
        source_sha256=sources["source_hashes"], reproduction_gate=gate,
        n_model_fits=0, n_model_fit_attempts=len(fit_attempts), n_deterministic_checks=deterministic_checks,
        n_raw_cases_per_strategy=len(expected), n_evaluable_cases_per_strategy=int(np.isfinite(expected.y_true).sum()),
        runtime_seconds=time.perf_counter() - started, interpretation=interpretation,
        primary_horizons=[1,3,6], h12_descriptive_only=True,
        public_data_policy="aggregate metrics only; no identifiers, target rows, predictions, ratios or training rows",
        no_tuning=True, no_feature_search=True, no_new_model_family=True,
        no_source_of_truth_changes=True, no_commit_push=True,
        output_sha256={p.name: sha256_file(p) for p in output.iterdir() if p.is_file()})
    save_json(output / "run_manifest.json", manifest)
    if not smoke:
        public = project_path(root, cfg["public_dir"])
        report = project_path(root, cfg["report_path"])
        if public.exists() or report.exists():
            raise FileExistsError("New public report paths must not already exist.")
        public.mkdir(parents=True)
        for name, table in tables.items():
            table.to_csv(public / f"{name}.csv", index=False)
        public_manifest = dict(manifest)
        public_manifest.pop("output_sha256")
        public_manifest["artifact_sha256"] = {p.name: sha256_file(p) for p in public.iterdir() if p.is_file()}
        save_json(public / "run_manifest.json", public_manifest)
        write_report(report, tables, manifest)
    summary = {key: manifest[key] for key in ("mode", "n_model_fits", "n_model_fit_attempts", "n_deterministic_checks", "n_raw_cases_per_strategy", "n_evaluable_cases_per_strategy", "runtime_seconds", "interpretation")}
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary
