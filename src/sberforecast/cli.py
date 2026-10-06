"""Командная строка; расчёты можно выполнять последовательно по группам дат."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time
import pandas as pd
import yaml
from .data import load_data, make_panel, audit, sha256_file
from .models import check_dependencies
from .backtest import run_origin, origins_from_config, fixed_sample
from .reporting import summarize, make_plots
from .detection import run_detection


def save_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    ap=argparse.ArgumentParser(description="СберИндекс: воспроизводимый первый эксперимент")
    ap.add_argument("command",choices=["audit","backtest","report","detect","all"])
    ap.add_argument("--config",default="configs/baseline.yaml")
    ap.add_argument("--origins",nargs="*",help="Подмножество точек выпуска, например 2023-12 2024-01")
    ap.add_argument("--rerun",action="store_true",help="Пересчитать существующие разделы при неизменной конфигурации")
    args=ap.parse_args(argv)
    root=Path(__file__).resolve().parents[2]
    config_path=Path(args.config)
    if not config_path.is_absolute():config_path=root/config_path
    cfg=yaml.safe_load(config_path.read_text(encoding="utf-8"))
    for key in ("models","data","backtest","seed","output_dir"):
        if key not in cfg:raise ValueError(f"В конфигурации отсутствует {key}")
    horizons=cfg["backtest"]["horizons"]
    if not horizons or any(not isinstance(h,int) or h<1 for h in horizons) or len(horizons)!=len(set(horizons)):
        raise ValueError("horizons должны быть уникальными положительными целыми числами.")
    lag=cfg["data"]["release_lag_months"]
    if not isinstance(lag,int) or lag<0:raise ValueError("Лаг должен быть неотрицательным целым числом.")
    data_path=root/cfg["data"]["path"]
    output=root/cfg["output_dir"];output.mkdir(parents=True,exist_ok=True)
    df=load_data(data_path,cfg["data"].get("category","Все категории"));panel=make_panel(df)
    summary=audit(df,panel);save_json(output/"data_audit.json",summary)
    if args.command=="audit":
        print(json.dumps(summary,ensure_ascii=False,indent=2));return
    # Контроль целостности исходных данных и прогнозирующего кода для resume.
    code_files=[Path(__file__).parent/name for name in ["data.py","features.py","models.py","backtest.py"]]
    code_hash=hashlib.sha256(b"".join(p.read_bytes() for p in code_files)).hexdigest()
    fingerprint=hashlib.sha256((json.dumps(cfg,sort_keys=True,ensure_ascii=False)+sha256_file(data_path)+code_hash).encode()).hexdigest()
    manifest_path=output/"run_manifest.json"
    if manifest_path.exists():
        old=json.loads(manifest_path.read_text(encoding="utf-8"))
        if old["fingerprint"]!=fingerprint:
            raise RuntimeError("Конфигурация, прогнозный код или данные изменены. Задайте НОВЫЙ output_dir, чтобы не смешать эксперименты.")
    else:
        versions={}
        for name in ["numpy","pandas","scipy","catboost","matplotlib","PyYAML","prophet","pytest"]:
            try:versions[name]=importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:versions[name]="NOT_INSTALLED"
        save_json(manifest_path,{"fingerprint":fingerprint,"data_sha256":sha256_file(data_path),
                               "forecast_code_sha256":code_hash,"config":cfg,"python":sys.version,
                               "platform":platform.platform(),"packages":versions,
                               "as_of_dates_assumed_not_verified":True})
        (output/"config_resolved.yaml").write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False),encoding="utf-8")
    expected=origins_from_config(cfg)
    selected=[pd.Period(x,freq="M") for x in args.origins] if args.origins else expected
    if any(p not in expected for p in selected):raise ValueError("Дата за пределами first_origin/last_origin.")
    partition_dir=output/"partitions";partition_dir.mkdir(exist_ok=True)
    if args.command in ("backtest","all"):
        check_dependencies(cfg["models"]["enabled"])
        sampled=fixed_sample(panel,cfg)
        if sampled is not None:save_json(output/"sample_ids.json",sorted(sampled))
        for origin in selected:
            dest=partition_dir/f"predictions_{origin}.csv.gz"
            if dest.exists() and not args.rerun:
                print(f"{origin}: сохранённый раздел, пропуск",flush=True);continue
            start=time.perf_counter()
            predictions,cohort,timings,importance,errors=run_origin(panel,cfg,origin,sampled)
            if predictions.empty:
                raise ValueError(f"{origin}: нет допустимых прогнозных пар. Проверьте границы и требования истории.")
            predictions.to_csv(dest,index=False)
            cohort.to_csv(partition_dir/f"cohort_{origin}.csv",index=False)
            pd.DataFrame(timings).to_csv(partition_dir/f"timing_{origin}.csv",index=False)
            if not importance.empty:importance.to_csv(partition_dir/f"feature_importance_{origin}.csv",index=False)
            save_json(partition_dir/f"errors_{origin}.json",errors)
            print(f"{origin}: {int(cohort.selected.sum())} МО, {len(predictions)} прогнозов; {time.perf_counter()-start:.2f} сек.",flush=True)
    complete=all((partition_dir/f"predictions_{o}.csv.gz").exists() for o in expected)
    if args.command in ("backtest","report","detect","all"):
        predictions,selection=summarize(output,cfg,complete)
        if complete:
            make_plots(output,panel,predictions,selection)
            if args.command in ("all","detect"):
                run_detection(predictions,selection,cfg,output)
        elif args.command in ("detect","all"):
            print("Прогон неполный: итоговый выбор и детекторы не интерпретировать; расчёт детекторов пропущен.")
        print(f"Результаты: {output}; complete={complete}",flush=True)
