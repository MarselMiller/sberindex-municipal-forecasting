"""E03: изолированный zero-shot запуск с сохранёнными соперниками E01/E02."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import yaml

from .backtest import fixed_sample, origins_from_config
from .chronos_evaluation import E01_MODELS, DIRECT, evaluate_chronos, require_same_keys, validate_predictions
from .chronos_model import MODEL, ChronosModel, load_pipeline, prepare_history
from .data import eligibility, get_prefix, load_data, make_panel, period_end, sha256_file
from .metrics import KEY, attach_split


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def project_path(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Путь должен находиться внутри проекта.")
    return path


def resources(root: Path) -> dict:
    import psutil
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage(str(root))
    process_memory = psutil.Process().memory_info()
    return {"ram_total_bytes": memory.total, "ram_available_bytes": memory.available,
            "disk_free_bytes": disk.free, "process_rss_bytes": process_memory.rss,
            "process_peak_rss_bytes": getattr(process_memory, "peak_wset", process_memory.rss),
            "cpu_logical_count": os.cpu_count()}


def expected_cases(panel: pd.DataFrame, cfg: dict, sample: list[str]) -> pd.DataFrame:
    """Переиспользует допуск/календарь E01, без импорта модели CatBoost из E02."""
    frames = []
    bt, lag = cfg["backtest"], cfg["data"]["release_lag_months"]
    for origin in origins_from_config(cfg):
        prefix = get_prefix(panel, origin, lag)
        cohort = eligibility(prefix, bt["min_history_observations"], bt["max_staleness_months"])
        ids = cohort.loc[cohort.eligible & cohort.municipality_id.isin(sample), "municipality_id"].tolist()
        for horizon in bt["horizons"]:
            target = (origin + horizon).to_timestamp()
            if target > panel.index.max():
                continue
            frames.append(pd.DataFrame({
                "municipality_id": ids, "forecast_origin": str(period_end(origin).date()),
                "target_period": str(target.date()), "horizon": horizon,
                "history_cutoff": str(period_end(origin-lag).date()),
                "availability_assumption": f"month_end_plus_{lag}_months",
                "y_true": panel.loc[target, ids].to_numpy(),
            }))
    return attach_split(pd.concat(frames, ignore_index=True), bt["validation_target_end"])


def verify_references(root: Path, panel: pd.DataFrame, cfg: dict) -> tuple:
    dirs = [project_path(root, cfg["comparison"][key]) for key in ("e01_dir", "e02_dir")]
    manifests = [json.loads((path / "run_manifest.json").read_text(encoding="utf-8")) for path in dirs]
    old_configs = [yaml.safe_load((path / "config_resolved.yaml").read_text(encoding="utf-8")) for path in dirs]
    data_hash = sha256_file(project_path(root, cfg["data"]["path"]))
    for label, old, manifest, source_config in zip(("E01", "E02"), old_configs, manifests,
                                                 ("configs/prophet_comparison.yaml", "configs/catboost_direct.yaml")):
        if old != manifest["config"] or old != yaml.safe_load((root / source_config).read_text(encoding="utf-8")):
            raise ValueError(f"Конфигурации {label} не совпадают.")
        if any(cfg[field] != old[field] for field in ("data", "backtest", "seed")):
            raise ValueError(f"Протокол E03 отличается от {label}.")
        if data_hash != manifest["data_sha256"]:
            raise ValueError(f"Данные отличаются от {label}.")
    original_code = [root / "src/sberforecast" / name for name in ("data.py", "features.py", "models.py", "backtest.py")]
    if hashlib.sha256(b"".join(path.read_bytes() for path in original_code)).hexdigest() != manifests[0]["forecast_code_sha256"]:
        raise ValueError("Прогнозный код E01 изменён.")
    for name, digest in manifests[1]["code_sha256"].items():
        if sha256_file(root / name) != digest:
            raise ValueError(f"Код сохранённого E02 изменён: {name}.")
    for name, digest in manifests[1]["source_e01_sha256"].items():
        if sha256_file(root / name) != digest:
            raise ValueError(f"Источник E01 отличается от использованного в E02: {name}.")
    if not json.loads((dirs[1] / "run_status.json").read_text(encoding="utf-8"))["complete"]:
        raise ValueError("E02 не завершён.")
    sample = json.loads(project_path(root, cfg["comparison"]["sample_ids_path"]).read_text(encoding="utf-8"))
    if len(sample) != 64 or len(set(sample)) != 64 or set(sample) != fixed_sample(panel, old_configs[0]):
        raise ValueError("Фактический набор 64 МО отличается от E01.")
    for source in dirs:
        if set(sample) != set(json.loads((source / "sample_ids.json").read_text(encoding="utf-8"))):
            raise ValueError("Идентификаторы выборок E01/E02 отличаются.")
    if set(cfg["comparison"]["models"]) != {DIRECT, *E01_MODELS}:
        raise ValueError("Состав сравнения E03 изменён.")
    c = cfg["chronos"]
    if (c["model_id"] != "amazon/chronos-2" or len(c["revision"]) != 40 or c["cross_learning"] or
            c["quantile"] != 0.5 or c["missing_values"] != "preserve_nan_calendar_grid" or c["postprocessing"] != "none"):
        raise ValueError("Нарушен фиксированный протокол Chronos-2.")
    smoke = cfg["smoke"]
    if smoke["municipality_ids"] != sorted(sample, key=int)[:2] or smoke["origin"] != "2023-12" or smoke["prediction_length"] != 12:
        raise ValueError("Smoke test должен использовать заранее выбранные два наименьших ID и декабрь 2023.")
    e01, e02 = [pd.read_csv(path / "predictions.csv.gz", dtype={"municipality_id": str}) for path in dirs]
    expected = expected_cases(panel, cfg, sample)
    if set(e01.model.unique()) != set(old_configs[0]["models"]["enabled"]):
        raise ValueError("Набор сохранённых моделей E01 отличается.")
    for label, frame in [(name, part) for name, part in e01.groupby("model")] + [("E02", e02)]:
        require_same_keys(frame, expected)
        paired = frame.merge(expected, on=KEY, suffixes=("", "_expected"), validate="one_to_one")
        a, b = paired.y_true.to_numpy(), paired.y_true_expected.to_numpy()
        if not ((a == b) | (np.isnan(a) & np.isnan(b))).all():
            raise ValueError(f"Целевые факты {label} отличаются от данных.")
        for field in ("history_cutoff", "split", "availability_assumption"):
            if not paired[field].equals(paired[field + "_expected"]):
                raise ValueError(f"Поле {field} отличается у {label}.")
    training = pd.read_csv(dirs[1] / "training_diagnostics.csv")
    annual = training.loc[training.horizon.eq(12)]
    if len(annual) != 1 or not annual.n_training_rows.eq(0).all() or annual.fit_called.any():
        raise ValueError("Нельзя подтвердить отсутствие годового fit CatBoostDirect.")
    # Проверить семантику статусов E02 до загрузки весов, используя копию эталона как заглушку.
    probe = expected.assign(model=MODEL, y_pred=0.0, status="native", effective_model=MODEL, reason="")
    evaluate_chronos(probe, e01, e02)
    protected = [path for directory in dirs for path in directory.rglob("*") if path.is_file()]
    protected += [root / "configs/prophet_comparison.yaml", root / "configs/catboost_direct.yaml"]
    source_hashes = {str(path.relative_to(root)): sha256_file(path) for path in protected}
    return sample, e01, e02, expected, source_hashes


def run_experiment(root: Path, config_path: Path, *, smoke: bool = False, preflight_only: bool = False,
                   command: str) -> dict:
    started = time.perf_counter()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    panel = make_panel(load_data(project_path(root, cfg["data"]["path"]), cfg["data"]["category"]))
    sample, e01, e02, expected, source_hashes = verify_references(root, panel, cfg)
    if preflight_only:
        result = {"preflight_passed": True, "n_sample": len(sample), "n_raw_keys": len(expected),
                  "n_evaluable_keys": int(np.isfinite(expected.y_true).sum())}
        print(json.dumps(result), flush=True)
        return result
    output = project_path(root, cfg["smoke"]["output_dir"] if smoke else cfg["output_dir"])
    protected_dirs = [project_path(root, cfg["comparison"][key]) for key in ("e01_dir", "e02_dir")]
    protected_dirs += [root / "outputs/baseline_v1", root / "outputs/e02_data_audit"]
    if not output.is_relative_to(root / "outputs") or any(output == path or output.is_relative_to(path) for path in protected_dirs):
        raise ValueError("Нужна отдельная папка результатов E03.")
    modules = [root / "src/sberforecast" / name for name in (
        "data.py", "backtest.py", "models.py", "metrics.py", "chronos_model.py", "chronos_evaluation.py",
        "chronos_experiment.py", "chronos_report.py")]
    modules += [root / "scripts/run_chronos.py"]
    versions = {name: importlib.metadata.version(name) for name in
                ("chronos-forecasting", "torch", "transformers", "accelerate", "huggingface-hub", "safetensors", "numpy", "pandas", "PyYAML", "psutil", "pytest")}
    signature = {"config": cfg, "config_sha256": sha256_file(config_path),
                 "code_sha256": {str(path.relative_to(root)): sha256_file(path) for path in modules},
                 "data_sha256": sha256_file(project_path(root, cfg["data"]["path"])),
                 "source_sha256": source_hashes, "versions": versions,
                 "dependency_lock_sha256": sha256_file(root / "requirements-chronos-lock.txt")}
    fingerprint = hashlib.sha256(json.dumps(signature, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if not smoke:
        smoke_output = project_path(root, cfg["smoke"]["output_dir"])
        smoke_status = json.loads((smoke_output / "smoke_status.json").read_text(encoding="utf-8"))
        smoke_manifest = json.loads((smoke_output / "run_manifest.json").read_text(encoding="utf-8"))
        if not smoke_status["passed"] or smoke_manifest["fingerprint"] != fingerprint:
            raise RuntimeError("Нужен успешный smoke test того же кода, данных и конфигурации.")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Существующие результаты E03 не перезаписываются. Для повторного опыта нужен новый каталог.")
    output.mkdir(parents=True, exist_ok=True)
    (output / "config_resolved.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    save_json(output / "sample_ids.json", sample)
    save_json(output / "preflight.json", {"passed": True, "n_raw_keys": len(expected),
                                         "same_sample_identifiers": True, "municipality_1471_not_preexcluded": "1471" in sample})
    manifest = dict(signature, fingerprint=fingerprint, experiment=cfg["experiment"], mode="smoke" if smoke else "full",
                    run_commands=[command], runtime_argv=sys.orig_argv, python=sys.version, python_executable=sys.executable,
                    platform=platform.platform(), seed=cfg["seed"], weight_revision=cfg["chronos"]["revision"],
                    device=cfg["chronos"]["device"], dtype=cfg["chronos"]["dtype"], batch_size=cfg["chronos"]["batch_size"],
                    zero_shot=True, fit_performed=False, cross_learning=False, point_quantile=0.5,
                    model_public_release="2025-10-20", independent_blind_test=False,
                    pretraining_overlap_ruled_out=False, publication_dates_verified=False,
                    git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                    git_status=subprocess.check_output(["git", "status", "--short"], cwd=root, text=True).splitlines(),
                    resources_before=resources(root))
    manifest["has_uncommitted_changes"] = bool(manifest["git_status"])
    save_json(output / "run_manifest.json", manifest)
    try:
        loading_started = time.perf_counter()
        pipeline = load_pipeline(root, cfg)
        loading_seconds = time.perf_counter() - loading_started
        import torch
        parameters = next(pipeline.model.parameters())
        if str(parameters.device) != "cpu" or parameters.dtype != torch.float32:
            raise RuntimeError("Фактическое устройство/dtype отличается от протокола.")
        manifest.update(model_load_seconds=loading_seconds, cuda_available=torch.cuda.is_available(),
                        actual_device=str(parameters.device), actual_dtype=str(parameters.dtype),
                        parameter_count=sum(parameter.numel() for parameter in pipeline.model.parameters()),
                        resources_after_model_load=resources(root))
        snapshot = project_path(root, cfg["chronos"]["cache_dir"]) / "hub/models--amazon--chronos-2/snapshots" / cfg["chronos"]["revision"]
        weight_files = list(snapshot.glob("*.safetensors"))
        if not weight_files:
            raise RuntimeError("Не найдены веса в snapshot зафиксированного revision.")
        manifest["weights_sha256"] = {path.name: sha256_file(path) for path in weight_files}
        save_json(output / "run_manifest.json", manifest)
        model = ChronosModel(pipeline, cfg["chronos"]["batch_size"], cfg["chronos"]["context_length"])
        if smoke:
            origin = pd.Period(cfg["smoke"]["origin"], freq="M")
            ids = cfg["smoke"]["municipality_ids"]
            cases = expected.loc[expected.forecast_origin.eq(str(period_end(origin).date())) & expected.municipality_id.isin(ids)]
            history = prepare_history(panel, origin, cfg["data"]["release_lag_months"], ids)
            result = model.predict(history, origin, cfg["backtest"]["horizons"], cfg["data"]["release_lag_months"])
            forecasts = cases.merge(result.forecasts, on=["municipality_id", "target_period", "horizon"], validate="one_to_one").assign(model=MODEL)
            validate_predictions(forecasts, cases)
            forecasts.to_csv(output / "predictions.csv.gz", index=False)
            result.paths.to_csv(output / "forecast_paths.csv.gz", index=False)
            save_json(output / "errors.json", result.errors)
            passed = bool(len(result.paths) == 24 and result.forecasts.status.eq("native").all() and
                          np.isfinite(result.paths.y_pred).all() and result.diagnostics["prediction_length"] == 12)
            status = dict(result.diagnostics, passed=passed, n_series=2, prediction_length=12,
                          model_load_seconds=loading_seconds, total_seconds=time.perf_counter()-started,
                          path_shape=[12, 2], resources_before=manifest["resources_before"], resources_after=resources(root))
            save_json(output / "smoke_status.json", status)
            if not passed:
                raise RuntimeError("Smoke test неуспешен; полный эксперимент запрещён.")
        else:
            partitions = output / "partitions"
            partitions.mkdir()
            predictions, timings, errors = [], [], []
            for origin in origins_from_config(cfg):
                issue = str(period_end(origin).date())
                cases = expected.loc[expected.forecast_origin.eq(issue)]
                ids = cases.municipality_id.drop_duplicates().tolist()
                horizons = sorted(cases.horizon.unique().tolist())
                history = prepare_history(panel, origin, cfg["data"]["release_lag_months"], ids)
                result = model.predict(history, origin, horizons, cfg["data"]["release_lag_months"])
                # Факты присоединяются после завершения predict; модели они не передавались.
                forecasts = cases.merge(result.forecasts, on=["municipality_id", "target_period", "horizon"], validate="one_to_one").assign(model=MODEL)
                validate_predictions(forecasts, cases)
                forecasts.to_csv(partitions / f"predictions_{origin}.csv.gz", index=False)
                result.paths.to_csv(partitions / f"paths_{origin}.csv.gz", index=False)
                prefix = get_prefix(panel, origin, cfg["data"]["release_lag_months"])
                cohort = eligibility(prefix, cfg["backtest"]["min_history_observations"], cfg["backtest"]["max_staleness_months"])
                cohort["in_sample"] = cohort.municipality_id.isin(sample)
                cohort["selected"] = cohort.eligible & cohort.in_sample
                cohort.to_csv(partitions / f"cohort_{origin}.csv", index=False)
                origin_errors = [dict(error, forecast_origin=issue) for error in result.errors]
                save_json(partitions / f"errors_{origin}.json", origin_errors)
                predictions.append(forecasts)
                timings.append(dict(result.diagnostics, forecast_origin=issue, **resources(root)))
                errors.extend(origin_errors)
                print(f"{origin}: {len(ids)} МО, {result.diagnostics['seconds']:.2f} s, failed={forecasts.status.eq('failed').sum()}", flush=True)
            current = pd.concat(predictions, ignore_index=True)
            current.to_csv(output / "predictions.csv.gz", index=False)
            pd.DataFrame(timings).to_csv(output / "timings.csv", index=False)
            save_json(output / "errors.json", errors)
            tables = evaluate_chronos(current, e01, e02)
            for name, table in tables.items():
                table.to_csv(output / f"{name}.csv", index=False)
            status = {"complete": True, "full_comparison_complete": not tables["comparison_coverage"].incomplete.any(),
                      "n_origins_completed": len(timings), "n_origins_expected": len(origins_from_config(cfg)),
                      "n_requested": len(current), "n_native": int(current.status.eq("native").sum()),
                      "n_failed": int(current.status.eq("failed").sum()), "n_missing_truth": int(current.y_true.isna().sum()),
                      "inference_seconds": sum(row["seconds"] for row in timings), "total_seconds": time.perf_counter()-started,
                      "model_load_seconds": loading_seconds, "smoke_passed": True}
            save_json(output / "run_status.json", status)
        if not all(sha256_file(root / name) == digest for name, digest in source_hashes.items()):
            raise RuntimeError("Защищённые результаты E01/E02 изменились во время запуска.")
        manifest.update(complete=True, resources_after=resources(root), total_seconds=time.perf_counter()-started)
        save_json(output / "run_manifest.json", manifest)
        if not smoke:
            from .chronos_report import write_report
            write_report(output, project_path(root, cfg["report_path"]), cfg)
        print(json.dumps(status, ensure_ascii=False), flush=True)
        return status
    except Exception as exc:
        manifest.update(complete=False, resources_after=resources(root), total_seconds=time.perf_counter()-started,
                        technical_error=repr(exc))
        save_json(output / "run_manifest.json", manifest)
        save_json(output / "run_status.json", {"complete": False, "stage": "smoke" if smoke else "full", "error": repr(exc)})
        raise
