import numpy as np
import pandas as pd
import pytest
from sberforecast.models import baseline_predict
from sberforecast.features import next_features,supervised_training
from sberforecast.backtest import run_origin


def test_last_value(panel,cfg):
    p,_=baseline_predict(panel.iloc[:12],12,"LastValue",cfg["models"])
    assert p.shape==(12,2)
    np.testing.assert_allclose(p,np.tile([111,222],(12,1)))


def test_seasonal_naive_matches_calendar(panel,cfg):
    p,fb=baseline_predict(panel.iloc[:12],12,"SeasonalNaive",cfg["models"])
    np.testing.assert_allclose(p,panel.iloc[:12].to_numpy())
    assert not fb.any()


def test_seasonal_fallback_does_not_drop_row(panel,cfg):
    x=panel.iloc[:12].copy();x.iloc[0,0]=np.nan
    p,fb=baseline_predict(x,1,"SeasonalNaive",cfg["models"])
    assert p[0,0]==111 and fb[0,0]


def test_yoy_without_year_pairs_is_seasonal(panel,cfg):
    a,_=baseline_predict(panel.iloc[:12],12,"SeasonalNaive",cfg["models"])
    b,_=baseline_predict(panel.iloc[:12],12,"SeasonalNaiveYoY",cfg["models"])
    np.testing.assert_allclose(a,b)


def test_features_use_previous_calendar_month(panel):
    X,anchor=next_features(panel.iloc[:12],pd.Timestamp("2024-01-01"))
    assert X.loc["1","lag_1"]==111 and X.loc["1","lag_12"]==100
    assert anchor[0]==111


def test_next_month_contract(panel):
    with pytest.raises(ValueError):next_features(panel.iloc[:12],pd.Timestamp("2024-02-01"))


def test_labels_end_at_cutoff(panel):
    X,y,meta=supervised_training(panel.iloc[:12])
    assert meta["max_training_target"]=="2023-12-01"
    assert len(X)==22 and len(y)==22


def test_backtest_does_not_use_future_values(panel,cfg):
    # Проверяем и глобальную обучаемую модель, не только простые ориентиры.
    one=run_origin(panel,cfg,pd.Period("2023-12",freq="M"))[0]
    changed=panel.copy();changed.loc["2024-01-01":]*=10000
    two=run_origin(changed,cfg,pd.Period("2023-12",freq="M"))[0]
    np.testing.assert_allclose(one.y_pred,two.y_pred,rtol=0,atol=0)
    assert not np.array_equal(one.y_true,two.y_true)


def test_lag_horizon_is_from_issue_month(panel,cfg):
    cfg["data"]["release_lag_months"]=1
    cfg["models"]["enabled"]=["SeasonalNaive"]
    pred=run_origin(panel,cfg,pd.Period("2024-01",freq="M"))[0]
    r=pred[(pred.horizon==1)&(pred.municipality_id=="1")].iloc[0]
    assert r.history_cutoff=="2023-12-31" and r.target_period=="2024-02-01"
    assert r.y_pred==101
