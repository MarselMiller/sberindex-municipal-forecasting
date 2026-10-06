from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import yaml

@pytest.fixture
def panel():
    dates=pd.date_range("2023-01-01",periods=24,freq="MS")
    return pd.DataFrame({"1":100+np.arange(24,dtype=float),"2":200+2*np.arange(24,dtype=float)},index=dates)

@pytest.fixture
def cfg():
    root=Path(__file__).resolve().parents[1]
    c=yaml.safe_load((root/"configs/baseline.yaml").read_text(encoding="utf-8"))
    c["models"]["catboost"]["iterations"]=5
    return c
