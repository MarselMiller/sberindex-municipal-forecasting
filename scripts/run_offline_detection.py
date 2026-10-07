"""Run the smoke, synthetic or independently gated real E06a stage."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sberforecast.offline_experiment import run_experiment

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/offline_detection.yaml")
parser.add_argument("--stage", choices=["smoke", "synthetic", "real"], required=True)
args = parser.parse_args()
command = subprocess.list2cmdline([sys.executable, *sys.orig_argv[1:]])
run_experiment(ROOT, ROOT / args.config, stage=args.stage, command=command)
