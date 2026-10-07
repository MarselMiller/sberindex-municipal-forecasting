"""F1: assemble frozen saved results. No models, downloads or experiments."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from sberforecast.final_summary import build_final_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default='reports/final')
    args = parser.parse_args()
    result = build_final_summary(ROOT, args.output_dir, command=sys.orig_argv)
    print(f"F1: {result['status']}; {result['forecasting_records']} forecasting / "
          f"{result['detection_records']} detection / {result['early_warning_records']} early-warning rows; "
          f"{result['figures']} figures. NO NEW MODEL FITS / NO NEW EXPERIMENTS.")


if __name__ == '__main__':
    main()
