"""E02a: воспроизводимый аудит обучающих пар; обучение моделей не вызывается."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sberforecast.data import get_prefix, load_data, make_panel, sha256_file
from sberforecast.direct_training import PAIR_KEY, build_direct_training
from sberforecast.features import supervised_training
from sberforecast.metrics import KEY, common_support


def project_path(relative: str) -> Path:
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Путь должен находиться внутри проекта.")
    return path


def legacy_keys(prefix: pd.DataFrame) -> list[tuple]:
    """Ключи фактических finite-фильтров старого одношагового сборщика."""
    keys = []
    for idx in range(1, len(prefix)):
        anchor = prefix.iloc[:idx].ffill().iloc[-1].to_numpy(dtype=float)
        valid = np.isfinite(prefix.iloc[idx]) & np.isfinite(anchor)
        target = prefix.index[idx]
        r = (target.to_period("M") - 1).end_time.normalize()
        keys.extend((str(uid), r, target, 1) for uid in prefix.columns[valid])
    return keys


def independent_keys(prefix: pd.DataFrame, origin: pd.Period, h: int,
                     lag: int, mode: str, max_staleness: int | None) -> set[tuple]:
    """Независимая проверка календарных индексов и допуска, без сборщика признаков."""
    finite = np.isfinite(prefix.to_numpy())
    cumulative = np.cumsum(finite, axis=0)
    positions = np.arange(len(prefix))[:, None]
    last = np.maximum.accumulate(np.where(finite, positions, -1), axis=0)
    result = set()
    for cidx, cutoff in enumerate(prefix.index.to_period("M")):
        r = cutoff + lag
        target = r + h
        if target + lag > origin:
            continue
        tidx = cidx + lag + h
        valid = (cumulative[cidx] >= (12 if mode == "strict12" else 1)) & finite[tidx]
        if max_staleness is not None:
            valid &= cidx - last[cidx] <= max_staleness
        result.update((str(uid), r.end_time.normalize(), target.to_timestamp(), h)
                      for uid in prefix.columns[valid])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/e02_data_audit.yaml")
    args = parser.parse_args()
    config_path = project_path(args.config)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output = project_path(cfg["output_dir"])
    if output != ROOT / "outputs/e02_data_audit":
        raise ValueError("Этот аудит пишет только в outputs/e02_data_audit.")
    if (output / "training_availability.csv").exists():
        raise FileExistsError("Аудит уже сохранён: существующие результаты не перезаписываются.")
    source = project_path(cfg["forecast_cohort"]["source_results"])
    protected_paths = [p for p in source.rglob("*") if p.is_file()]
    protected_paths.append(ROOT / "configs/prophet_comparison.yaml")
    before = {p: sha256_file(p) for p in protected_paths}
    source_cfg = yaml.safe_load((source / "config_resolved.yaml").read_text(encoding="utf-8"))
    for name in ("data", "seed"):
        assert cfg[name] == source_cfg[name]
    for name, value in cfg["backtest"].items():
        assert value == source_cfg["backtest"][name]
    for name in ("min_history_observations", "max_staleness_months"):
        assert cfg["forecast_cohort"][name] == source_cfg["backtest"][name]
    data_path = project_path(cfg["data"]["path"])
    panel = make_panel(load_data(data_path, cfg["data"]["category"]))
    predictions = pd.read_csv(source / "predictions.csv.gz", dtype={"municipality_id": str})
    common = common_support(predictions, source_cfg["models"]["enabled"])
    cases = predictions.loc[predictions.model.eq(source_cfg["models"]["enabled"][0]),
                            KEY + ["split", "y_true"]].copy()
    common_keys = set(map(tuple, common[KEY].drop_duplicates().to_numpy()))
    cases["in_common_e01"] = [tuple(row) in common_keys for row in cases[KEY].to_numpy()]
    cases["actual_available"] = np.isfinite(cases.y_true)
    cases = cases.drop(columns="y_true")
    assert not cases.duplicated(KEY).any()
    # Подтверждённый аудит E01: 1897 запрошенных и 1890 оцениваемых ключей.
    assert len(cases) == 1897 and cases.in_common_e01.sum() == 1890
    modes = cfg["direct_training"]["modes"]
    max_staleness = cfg["direct_training"]["max_staleness_months"]
    lag = cfg["data"]["release_lag_months"]
    origins = pd.period_range(cfg["backtest"]["first_origin"], cfg["backtest"]["last_origin"], freq="M")
    rows, comparisons = [], []
    for O in origins:
        prefix = get_prefix(panel, O, lag)
        for h in cfg["backtest"]["horizons"]:
            for mode in modes:
                X, y, meta = build_direct_training(panel, O, h, release_lag_months=lag,
                                                   mode=mode, max_staleness_months=max_staleness)
                assert len(X) == len(y) == len(meta)
                actual_keys = set(map(tuple, meta[PAIR_KEY].to_numpy()))
                expected_keys = independent_keys(prefix, O, h, lag, mode, max_staleness)
                assert actual_keys == expected_keys and len(actual_keys) == len(meta)
                if len(meta):
                    assert (meta.target_available_at <= O.end_time.normalize()).all()
                    assert (meta.feature_cutoff <= meta.historical_origin).all()
                    np.testing.assert_array_equal(y, meta.target_value - meta.anchor)
                rows.append({
                    "forecast_origin": str(O.end_time.date()), "horizon": h, "mode": mode,
                    "n_training_rows": len(meta),
                    "n_training_municipalities": meta.municipality_id.nunique(),
                    "n_historical_origins": meta.historical_origin.nunique(),
                    "n_target_periods": meta.target_period.nunique(),
                    "min_historical_origin": str(meta.historical_origin.min()) if len(meta) else None,
                    "max_historical_origin": str(meta.historical_origin.max()) if len(meta) else None,
                    "max_target_available_at": str(meta.target_available_at.max()) if len(meta) else None,
                    "training_available": bool(len(meta)),
                    "reason": "pairs_available" if len(meta) else "no_training_pairs",
                    "evaluation_target_in_data": (O + h).to_timestamp() <= panel.index.max(),
                })
                if mode == "legacy" and h == 1 and lag == 0 and max_staleness is None:
                    old_X, old_y, _ = supervised_training(prefix)
                    old_keys = legacy_keys(prefix)
                    new_keys = list(map(tuple, meta[PAIR_KEY].to_numpy()))
                    assert new_keys == old_keys
                    pd.testing.assert_frame_equal(X, old_X)
                    np.testing.assert_array_equal(y, old_y)
                    comparisons.append({"forecast_origin": str(O.end_time.date()),
                                        "n_legacy_rows": len(old_keys), "n_direct_rows": len(meta),
                                        "missing_keys": len(set(old_keys) - actual_keys),
                                        "extra_keys": len(actual_keys - set(old_keys)),
                                        "key_order_equal": True, "features_equal": True,
                                        "labels_equal": True})
        print(f"{O}: проверены оба режима и четыре горизонта", flush=True)
    availability = pd.DataFrame(rows)
    case_availability = cases.merge(availability, on=["forecast_origin", "horizon"],
                                    how="left", validate="many_to_many")
    assert len(case_availability) == len(cases) * len(modes)
    assert not case_availability.duplicated(KEY + ["mode"]).any()
    evaluated = case_availability.loc[case_availability.in_common_e01]
    summary = evaluated.groupby(["mode", "split", "horizon"]).agg(
        n_evaluation_cases=("municipality_id", "size"),
        n_training_available=("training_available", "sum"),
        n_evaluation_origins=("forecast_origin", "nunique"),
        min_training_dates=("n_historical_origins", "min"),
        max_training_dates=("n_historical_origins", "max"),
    ).reset_index()
    summary["n_training_unavailable"] = summary.n_evaluation_cases - summary.n_training_available
    assert all(sha256_file(p) == checksum for p, checksum in before.items())
    output.mkdir(parents=True, exist_ok=True)
    availability.to_csv(output / "training_availability.csv", index=False)
    case_availability.to_csv(output / "e01_case_availability.csv.gz", index=False)
    summary.to_csv(output / "e01_evaluation_summary.csv", index=False)
    pd.DataFrame(comparisons).to_csv(output / "legacy_h1_comparison.csv", index=False)
    (output / "config_resolved.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    manifest = {
        "experiment": cfg["experiment"], "command": subprocess.list2cmdline(sys.orig_argv),
        "python": sys.version, "seed": cfg["seed"], "config": cfg,
        "versions": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "PyYAML")},
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).splitlines(),
        "data_sha256": sha256_file(data_path),
        "audit_code_sha256": hashlib.sha256(Path(__file__).read_bytes() +
                                            (ROOT / "src/sberforecast/direct_training.py").read_bytes()).hexdigest(),
        "source_e01_sha256": {str(p.relative_to(ROOT)): checksum for p, checksum in before.items()},
        "e01_unchanged": True, "independent_pair_set_checks_passed": len(rows),
        "legacy_h1_exact_comparisons_passed": len(comparisons),
        "model_training_performed": False, "forecast_metrics_computed": False,
        "availability_dates_assumed_not_verified": True,
        "training_scope": "all municipalities; not restricted to E01 forecast cohort",
        "has_pairs_is_not_evidence_of_model_reliability": True,
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Сохранено: {output}; моделей не обучали.")


if __name__ == "__main__":
    main()
