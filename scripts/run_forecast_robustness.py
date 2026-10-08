"""Saved-prediction robustness analysis. No new model fits."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from sberforecast.forecast_robustness import run

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--config',default='configs/forecast_robustness.yaml')
mode=parser.add_mutually_exclusive_group()
mode.add_argument('--gate-only',action='store_true')
mode.add_argument('--smoke',action='store_true')
args=parser.parse_args()
run(ROOT,ROOT/args.config,gate_only=args.gate_only,smoke=args.smoke,
    command=[Path(sys.executable).relative_to(ROOT).as_posix(),*sys.argv])
