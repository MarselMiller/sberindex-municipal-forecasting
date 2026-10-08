"""F1: assemble frozen saved results. No models, downloads or experiments."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from sberforecast.final_summary import build_final_summary
from sberforecast.final_ablation import refresh_forecasting_ablation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default='reports/final')
    parser.add_argument('--forecasting-ablation-only', action='store_true',
                        help='Refresh only the completed ablation JSON object; preserve other final artifacts')
    parser.add_argument('--check', action='store_true', help='Check the additive ablation without writing')
    args = parser.parse_args()
    if args.forecasting_ablation_only:
        result = refresh_forecasting_ablation(ROOT, args.output_dir, check=args.check)
        print(f"Forecasting ablation: {result['status']}; {result['forecasting_ablation_records']} saved rows. "
              'NO NEW MODEL FITS / NO NEW EXPERIMENTS.')
        return
    if args.check:
        parser.error('--check requires --forecasting-ablation-only')
    result = build_final_summary(ROOT, args.output_dir, command=sys.orig_argv)
    print(f"F1: {result['status']}; {result['forecasting_records']} forecasting / "
          f"{result['detection_records']} detection / {result['early_warning_records']} early-warning rows; "
          f"{result['figures']} figures. NO NEW MODEL FITS / NO NEW EXPERIMENTS.")


if __name__ == '__main__':
    main()
