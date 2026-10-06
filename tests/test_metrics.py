import numpy as np
import pandas as pd
from sberforecast.metrics import safe_r2,attach_split,common_support,compute_metrics,select_on_validation,selected_policy_predictions


def test_r2_constant_is_undefined():
    assert np.isnan(safe_r2(np.array([1,1]),np.array([1,1])))


def test_r2_single_observation_is_undefined():
    assert np.isnan(safe_r2(np.array([1]),np.array([2])))


def test_split_by_target_not_origin():
    x=pd.DataFrame({"target_period":["2024-06-01","2024-07-01"]})
    assert attach_split(x,"2024-06").split.tolist()==["validation","holdout"]


def test_common_support_excludes_missing_predictions():
    rows=[]
    for uid in ["1","2"]:
        for m in ["A","B"]:
            rows.append(dict(municipality_id=uid,forecast_origin="2024-06-30",target_period="2024-07-01",horizon=1,model=m,y_true=10,y_pred=np.nan if uid=="2" and m=="B" else 9))
    c=common_support(pd.DataFrame(rows),["A","B"])
    assert len(c)==2 and set(c.municipality_id)=={"1"}


def test_macro_and_micro_differ_with_unequal_counts():
    d=pd.DataFrame(dict(split=["holdout"]*3,model=["A"]*3,horizon=[1]*3,municipality_id=["1","1","2"],forecast_origin=["2024-01-31","2024-02-29","2024-01-31"],y_true=[10,10,100],y_pred=[9,9,90]))
    m,_=compute_metrics(d)
    assert m.mae_macro.iloc[0]==5.5 and m.mae_micro.iloc[0]==4


def test_selection_does_not_inspect_holdout(cfg):
    m=pd.DataFrame({"split":["validation","validation","holdout","holdout"],"horizon":[1]*4,"model":["A","B","A","B"],"mae_macro":[1.,2.,1000.,0.],"n_origins":[6]*4})
    s=select_on_validation(m,cfg)
    assert s["1"]["model"]=="A"
    assert s["12"]["model"]=="SeasonalNaive"


def test_june_selection_cannot_apply_to_january_forecast():
    d=pd.DataFrame(dict(split=["holdout"]*2,horizon=[6]*2,model=["A"]*2,forecast_origin=["2024-01-31","2024-06-30"]))
    selection={"6":{"model":"A","selection_available_at":"2024-06-30"}}
    p=selected_policy_predictions(d,selection)
    assert len(p)==1 and p.forecast_origin.iloc[0]=="2024-06-30"
