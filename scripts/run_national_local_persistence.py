"""Фиксированная ablation National/Local без обучения локальной модели."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sberforecast.national_local_persistence_experiment import run_experiment

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/national_local_persistence.yaml")
mode = parser.add_mutually_exclusive_group()
mode.add_argument("--gate-only", action="store_true")
mode.add_argument("--smoke", action="store_true")
args = parser.parse_args()
executable = Path(sys.executable).relative_to(ROOT).as_posix()
run_experiment(ROOT, ROOT / args.config, gate_only=args.gate_only, smoke=args.smoke,
               command=subprocess.list2cmdline([executable, *sys.orig_argv[1:]]))
