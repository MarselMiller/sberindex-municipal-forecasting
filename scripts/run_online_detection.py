"""Отдельный E04a: минимальная проверка, затем синтетическое сравнение и диагностика."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sberforecast.online_experiment import run_experiment

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/online_detection.yaml")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    command = subprocess.list2cmdline([sys.executable, *sys.orig_argv[1:]])
    run_experiment(ROOT, ROOT / args.config, smoke=args.smoke, command=command)
