"""Изолированный запуск E02b с сохранённым E01; модели E01 не переобучаются."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml

from .backtest import fixed_sample, origins_from_config
from .data import eligibility, get_prefix, load_data, make_panel, period_end, sha256_file
from .direct_evaluation import evaluate_direct, h1_consistency, require_same_keys, validate_direct_against_reference
from .direct_model import CatBoostDirect, STRATEGY, TemporalIntegrityError
from .metrics import KEY, attach_split


def save_json(path: Path, value: dict | list) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def project_path(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Путь должен находиться внутри проекта.")
    return path


def expected_cases(panel: pd.DataFrame, cfg: dict, sample: set[str]) -> pd.DataFrame:
    """Допуск по прошлому; будущие факты используются только для сверки/оценки."""
    frames = []
    lag = cfg["data"]["release_lag_months"]
    bt = cfg["backtest"]
    for origin in origins_from_config(cfg):
        prefix = get_prefix(panel, origin, lag)
        cohort = eligibility(prefix, bt["min_history_observations"], bt["max_staleness_months"])
        ids = cohort.loc[cohort.eligible & cohort.municipality_id.isin(sample), "municipality_id"].tolist()
        for h in bt["horizons"]:
            target = (origin + h).to_timestamp()
            if target > panel.index.max():
                continue
            frames.append(pd.DataFrame({
                "municipality_id": ids, "forecast_origin": str(period_end(origin).date()),
                "target_period": str(target.date()), "horizon": h,
                "history_cutoff": str(period_end(origin-lag).date()),
                "availability_assumption": f"month_end_plus_{lag}_months",
                "y_true": panel.loc[target, ids].to_numpy(),
            }))
    return attach_split(pd.concat(frames, ignore_index=True), bt["validation_target_end"])


def verify_reference(root: Path, panel: pd.DataFrame, cfg: dict) -> tuple:
    source = project_path(root, cfg["comparison"]["source_dir"])
    old = yaml.safe_load((source / "config_resolved.yaml").read_text(encoding="utf-8"))
    manifest = json.loads((source / "run_manifest.json").read_text(encoding="utf-8"))
    if old != manifest["config"] or old != yaml.safe_load((root / "configs/prophet_comparison.yaml").read_text(encoding="utf-8")):
        raise ValueError("Конфигурации E01 не совпадают.")
    for field in ("data", "backtest", "seed"):
        if cfg[field] != old[field]:
            raise ValueError(f"Протокол {field} отличается от E01.")
    for field in ("catboost", "yearly_growth_window", "yearly_growth_bounds"):
        if cfg["models"][field] != old["models"][field]:
            raise ValueError(f"Параметры {field} отличаются от E01.")
    if cfg["direct_training"] != {"mode": "legacy", "max_staleness_months": None,
                                   "no_training_pairs_fallback": "SeasonalNaive"}:
        raise ValueError("Для E02b зафиксирован legacy без ограничения давности и с SeasonalNaive.")
    if sha256_file(project_path(root, cfg["data"]["path"])) != manifest["data_sha256"]:
        raise ValueError("Данные отличаются от E01.")
    legacy_files = [root / "src/sberforecast" / n for n in ("data.py", "features.py", "models.py", "backtest.py")]
    if hashlib.sha256(b"".join(p.read_bytes() for p in legacy_files)).hexdigest() != manifest["forecast_code_sha256"]:
        raise ValueError("Исходный прогнозный код E01 изменён.")
    if importlib.metadata.version("catboost") != manifest["packages"]["catboost"]:
        raise ValueError("Версия CatBoost отличается от E01; сравнение h=1 требует объяснения.")
    ids = json.loads(project_path(root, cfg["comparison"]["sample_ids_path"]).read_text(encoding="utf-8"))
    source_ids = json.loads((source / "sample_ids.json").read_text(encoding="utf-8"))
    if len(ids) != 64 or len(set(ids)) != 64 or set(ids) != set(source_ids) or set(ids) != fixed_sample(panel, old):
        raise ValueError("Фактические идентификаторы пилотной выборки отличаются от E01.")
    reference = pd.read_csv(source / "predictions.csv.gz", dtype={"municipality_id": str})
    if set(reference.model.unique()) != set(old["models"]["enabled"]):
        raise ValueError("Набор моделей в прогнозах E01 отличается от конфигурации.")
    if set(cfg["comparison"]["models"]) != set(old["models"]["enabled"]):
        raise ValueError("Соперники зафиксированы: все шесть моделей E01.")
    expected = expected_cases(panel, cfg, set(ids))
    for name, ref in reference.groupby("model"):
        require_same_keys(expected, ref)
        joined = expected.merge(ref, on=KEY, suffixes=("", "_e01"), validate="one_to_one")
        for field in ("split", "history_cutoff", "availability_assumption"):
            if not joined[field].equals(joined[field + "_e01"]):
                raise ValueError(f"Несовпадение {field} у {name}.")
        a, b = joined.y_true.to_numpy(), joined.y_true_e01.to_numpy()
        if not ((a == b) | (np.isnan(a) & np.isnan(b))).all():
            raise ValueError(f"Несовпадение целевых фактов у {name}.")
    protected = [p for p in source.rglob("*") if p.is_file()] + [root / "configs/prophet_comparison.yaml"]
    source_hashes = {str(p.relative_to(root)): sha256_file(p) for p in protected}
    return ids, reference, expected, source_hashes


def run_experiment(root: Path, config_path: Path, requested_origins: list[str] | None,
                   command: str) -> dict:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_path = project_path(root, cfg["data"]["path"])
    panel = make_panel(load_data(data_path, cfg["data"]["category"]))
    sample, reference, expected, source_hashes = verify_reference(root, panel, cfg)
    output = project_path(root, cfg["output_dir"])
    source = project_path(root, cfg["comparison"]["source_dir"])
    if not output.is_relative_to(root / "outputs") or output == source or output.is_relative_to(source) or output == root / "outputs/baseline_v1":
        raise ValueError("Нужна отдельная папка результатов E02.")
    all_origins = origins_from_config(cfg)
    selected = [pd.Period(x, freq="M") for x in requested_origins] if requested_origins else all_origins
    if len(set(selected)) != len(selected) or any(o not in all_origins for o in selected):
        raise ValueError("Повторная дата или дата вне протокола.")
    code_paths = [root / "src/sberforecast" / n for n in (
        "data.py", "features.py", "models.py", "backtest.py", "metrics.py", "direct_training.py",
        "direct_model.py", "direct_evaluation.py", "direct_experiment.py", "direct_report.py")]
    code_paths.append(root / "scripts/run_catboost_direct.py")
    hashes = {str(p.relative_to(root)): sha256_file(p) for p in code_paths}
    versions = {n: importlib.metadata.version(n) for n in ("numpy", "pandas", "catboost", "PyYAML", "pytest")}
    signature = {"config": cfg, "config_sha256": sha256_file(config_path), "code_sha256": hashes,
                 "data_sha256": sha256_file(data_path), "source_e01_sha256": source_hashes, "versions": versions}
    fingerprint = hashlib.sha256(json.dumps(signature, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    manifest_path = output / "run_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if fingerprint != manifest["fingerprint"]:
            raise ValueError("Код, данные, конфигурация или E01 изменились: продолжение запрещено.")
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("Нельзя писать в непустую папку без manifest этого эксперимента.")
        output.mkdir(parents=True, exist_ok=True)
        manifest = dict(signature, experiment=cfg["experiment"], fingerprint=fingerprint,
                        experiment_date=cfg["experiment_date"], seed=cfg["seed"],
                        python=sys.version, platform=platform.platform(), run_commands=[],
                        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                        git_status=subprocess.check_output(["git", "status", "--short"], cwd=root, text=True).splitlines(),
                        training_scope="all available municipalities", publication_dates_verified=False,
                        independent_blind_test=False, strict12_forecast_experiment=False)
        (output / "config_resolved.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
        save_json(output / "sample_ids.json", sample)
        save_json(output / "preflight.json", {"data_and_protocol_match": True, "sample_identifiers_match": True,
                                              "raw_keys_per_e01_model": len(expected),
                                              "source_models": sorted(reference.model.unique()),
                                              "municipality_1471_not_preexcluded": "1471" in sample})
    manifest["run_commands"].append(command)
    manifest["has_uncommitted_changes"] = bool(manifest["git_status"])
    save_json(manifest_path, manifest)
    partitions = output / "partitions"
    partitions.mkdir(exist_ok=True)
    model = CatBoostDirect(cfg["models"]["catboost"], cfg["seed"], cfg["data"]["release_lag_months"],
                          cfg["direct_training"]["mode"], cfg["direct_training"]["max_staleness_months"])
    for origin in selected:
        destination = partitions / f"predictions_{origin}.csv.gz"
        if destination.exists():
            print(f"{origin}: сохранённый раздел, повторного обучения нет", flush=True)
            continue
        issue = str(period_end(origin).date())
        origin_cases = expected.loc[expected.forecast_origin.eq(issue)]
        ids = origin_cases.municipality_id.drop_duplicates().tolist()
        lag = cfg["data"]["release_lag_months"]
        cohort = eligibility(get_prefix(panel, origin, lag), cfg["backtest"]["min_history_observations"],
                             cfg["backtest"]["max_staleness_months"])
        cohort["in_sample"] = cohort.municipality_id.isin(sample)
        cohort["selected"] = cohort.eligible & cohort.in_sample
        rows, training, importance, errors = [], [], [], []
        for h in origin_cases.horizon.unique():
            try:
                result = model.predict(panel, origin, int(h), ids, cfg["models"])
            except TemporalIntegrityError as exc:
                save_json(output / "run_status.json", {"complete": False, "stopped_temporal_integrity": True,
                                                        "forecast_origin": issue, "horizon": int(h), "error": str(exc)})
                raise
            case = origin_cases.loc[origin_cases.horizon.eq(h)]
            forecast = case.merge(result.forecasts, on="municipality_id", validate="one_to_one")
            forecast["model"] = STRATEGY
            rows.append(forecast)
            training.append(result.training)
            importance.append(result.importance.assign(forecast_origin=issue, horizon=int(h)))
            errors.extend(result.errors)
            print(f"{origin}, h={h}: {result.training['n_training_rows']} строк, "
                  f"{result.training['n_historical_origins']} дат; {result.training['status']}", flush=True)
        forecasts = pd.concat(rows, ignore_index=True)
        ref_origin = reference.loc[reference.forecast_origin.eq(issue)]
        validate_direct_against_reference(forecasts, ref_origin)
        check, _ = h1_consistency(forecasts, ref_origin, cfg["comparison"]["h1_absolute_tolerance"])
        forecasts.to_csv(destination, index=False)
        pd.DataFrame(training).to_csv(partitions / f"training_{origin}.csv", index=False)
        pd.concat(importance, ignore_index=True).to_csv(partitions / f"importance_{origin}.csv", index=False)
        cohort.to_csv(partitions / f"cohort_{origin}.csv", index=False)
        save_json(partitions / f"errors_{origin}.json", errors)
        save_json(partitions / f"h1_{origin}.json", check)
        if check["n_exceeds_tolerance"]:
            save_json(output / "run_status.json", {"complete": False, "stopped_h1_difference": True, **check})
            raise RuntimeError("h=1 существенно отличается от E01: требуется выяснить причину.")
    available_origins = [o for o in all_origins if (partitions / f"predictions_{o}.csv.gz").exists()]
    direct = pd.concat([pd.read_csv(partitions / f"predictions_{o}.csv.gz", dtype={"municipality_id": str})
                        for o in available_origins], ignore_index=True)
    ref_subset = reference.loc[reference.forecast_origin.isin(direct.forecast_origin.unique())]
    evaluations = evaluate_direct(direct, ref_subset, cfg["comparison"]["models"])
    direct.to_csv(output / "predictions.csv.gz", index=False)
    for name, frame in evaluations.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    diagnostics = pd.concat([pd.read_csv(partitions / f"training_{o}.csv") for o in available_origins], ignore_index=True)
    diagnostics.to_csv(output / "training_diagnostics.csv", index=False)
    check, by_origin = h1_consistency(direct, ref_subset, cfg["comparison"]["h1_absolute_tolerance"])
    save_json(output / "h1_consistency.json", check)
    by_origin.to_csv(output / "h1_consistency_by_origin.csv", index=False)
    unchanged = all(sha256_file(root / p) == sha for p, sha in source_hashes.items())
    if not unchanged:
        raise RuntimeError("E01 изменился во время запуска; результаты нельзя считать сопоставимыми.")
    coverage = evaluations["coverage_e01"]
    status = {
        "complete": len(available_origins) == len(all_origins),
        "n_origins_completed": len(available_origins), "n_origins_expected": len(all_origins),
        "n_requested": len(direct), "n_native": int(direct.status.eq("native").sum()),
        "n_fallback": int(direct.status.eq("fallback_no_training_pairs").sum()),
        "n_failed": int(direct.status.eq("failed").sum()),
        "n_e01_evaluable": int(coverage.n_e01_cases.sum()),
        "full_e01_comparison_complete": bool(len(available_origins) == len(all_origins) and coverage.n_failed.sum() == 0),
        "e01_unchanged": unchanged, "h1_check_complete": check["complete"],
        "h1_n_exceeds_tolerance": check["n_exceeds_tolerance"],
        "native_yearly_status": "no_training_pairs",
    }
    save_json(output / "run_status.json", status)
    manifest["complete"] = status["complete"]
    manifest["full_e01_comparison_complete"] = status["full_e01_comparison_complete"]
    save_json(manifest_path, manifest)
    if status["complete"]:
        from .direct_report import write_report
        write_report(output, project_path(root, cfg["report_path"]), cfg)
    print(json.dumps(status, ensure_ascii=False), flush=True)
    return status
