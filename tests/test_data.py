import numpy as np
import pandas as pd
import pytest
from sberforecast.data import get_prefix,eligibility,make_panel,load_data,period_end


def test_prefix_excludes_future(panel):
    p=get_prefix(panel,pd.Period("2023-12",freq="M"),0)
    assert len(p)==12 and p.index.max()==pd.Timestamp("2023-12-01")


def test_release_lag(panel):
    p=get_prefix(panel,pd.Period("2024-01",freq="M"),1)
    assert p.index.max()==pd.Timestamp("2023-12-01")


def test_negative_release_lag(panel):
    with pytest.raises(ValueError):get_prefix(panel,pd.Period("2023-12",freq="M"),-1)


def test_eligibility_ignores_future_gaps(panel):
    altered=panel.copy();altered.loc["2024-01-01":,"1"]=np.nan
    e1=eligibility(get_prefix(panel,pd.Period("2023-12",freq="M"),0),12,1)
    e2=eligibility(get_prefix(altered,pd.Period("2023-12",freq="M"),0),12,1)
    pd.testing.assert_frame_equal(e1,e2)


def test_missing_month_is_not_zero():
    df=pd.DataFrame({"municipality_id":["1","1"],"ds":pd.to_datetime(["2023-01-01","2023-03-01"]),"y":[10.,20.]})
    p=make_panel(df)
    assert len(p)==3 and np.isnan(p.loc["2023-02-01","1"])


def test_duplicate_join_rejected(tmp_path):
    p=tmp_path/"x.csv"
    pd.DataFrame({"territory_id":[1,1],"date":["2023-01","2023-01"],"value":[10,10]}).to_csv(p,index=False)
    with pytest.raises(ValueError,match="Дубликаты"):load_data(p)


def test_leap_month_end():
    assert period_end(pd.Period("2024-02",freq="M"))==pd.Timestamp("2024-02-29")
