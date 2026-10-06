"""Отдельный E03: preflight → smoke test → полный zero-shot запуск."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sberforecast.chronos_experiment import project_path, run_experiment

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/chronos_zero_shot.yaml")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    command = subprocess.list2cmdline([sys.executable, *sys.orig_argv[1:]])
    run_experiment(ROOT, project_path(ROOT, args.config), smoke=args.smoke,
                   preflight_only=args.preflight_only, command=command)
