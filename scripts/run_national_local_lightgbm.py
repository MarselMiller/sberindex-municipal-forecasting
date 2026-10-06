"""Run only the fixed E05d experiment in the main environment."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sberforecast.national_local_experiment import run_experiment

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/national_local_lightgbm.yaml")
parser.add_argument("--origins", nargs="+")
args = parser.parse_args()
run_experiment(ROOT, ROOT / args.config, args.origins,
               subprocess.list2cmdline([sys.executable, *sys.orig_argv[1:]]))
