"""E05b: fixed seasonal trends and month encodings, no macro data or tuning."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sberforecast.direct_experiment import project_path
from sberforecast.trend_calendar_experiment import run_experiment

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/trend_calendar.yaml")
    parser.add_argument("--origins", nargs="+", help="One-date smoke, then resume with no flag.")
    args = parser.parse_args()
    run_experiment(ROOT, project_path(ROOT, args.config), args.origins, subprocess.list2cmdline(sys.orig_argv))
