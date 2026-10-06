"""E02b: отдельная команда; запуск E01 и его модели не изменяются."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sberforecast.direct_experiment import project_path, run_experiment

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/catboost_direct.yaml")
    parser.add_argument("--origins", nargs="+", help="Минимальный прогон; затем запуск без флага продолжает все даты.")
    args = parser.parse_args()
    run_experiment(ROOT, project_path(ROOT, args.config), args.origins, subprocess.list2cmdline(sys.orig_argv))
