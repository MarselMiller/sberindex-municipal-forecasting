"""E05c: fixed national archived forecasts, no downloads or tuning."""
from pathlib import Path
import argparse
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sberforecast.direct_experiment import project_path
from sberforecast.macro_forecast_experiment import run_experiment

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/macro_forecast.yaml")
parser.add_argument("--origins", nargs="+")
args = parser.parse_args()
run_experiment(ROOT, project_path(ROOT, args.config), args.origins,
               subprocess.list2cmdline([sys.executable, *sys.orig_argv[1:]]))
