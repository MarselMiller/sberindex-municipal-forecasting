"""Full-panel eligibility/residual adapter uses only each origin's prefix."""
import copy

import numpy as np
import pandas as pd

from sberforecast.early_warning_full_panel import prepare_panel_residuals,compare_saved_pilot


def config():
    return dict(data=dict(release_lag_months=0,availability_assumption="synthetic_L0"),
        panel=dict(first_origin="2023-12",last_origin="2024-02",min_history_observations=12,
            max_staleness_months=1,residual_model="SeasonalNaiveYoY",residual_horizon=1,
            yearly_growth_window=3,yearly_growth_bounds=[.5,2.]))


def panel():
    values=pd.DataFrame({"1":np.arange(15,dtype=float)+100,"2":np.arange(15,dtype=float)+200},
                        index=pd.date_range("2023-01-01",periods=15,freq="MS"))
    values.loc[:"2023-12-01","2"]=np.nan
    return values


def test_future_outcome_and_incomplete_series_never_determine_cohort_or_prediction():
    original=panel()
    queries,residuals=prepare_panel_residuals(original,config())
    assert len(queries)==6 and len(residuals)==6
    assert not queries.loc[queries.municipality_id.eq("2"),"eligible_at_origin"].any()
    altered=original.copy()
    altered.loc["2024-01-01":,"1"]=999999.
    altered.loc["2024-01-01":,"2"]=np.nan
    second,changed=prepare_panel_residuals(altered,config())
    early=queries.forecast_origin.eq("2023-12-31")
    pd.testing.assert_frame_equal(queries.loc[early],second.loc[early])
    before=residuals.loc[residuals.observation_period.eq("2024-01"),"y_pred"]
    after=changed.loc[changed.observation_period.eq("2024-01"),"y_pred"]
    pd.testing.assert_series_equal(before,after)


def test_eligibility_changes_with_past_history_no_future_completeness_filter():
    values=panel()
    values.loc["2023-01-01":"2023-12-01","2"]=200.
    values.loc["2024-01-01":,"2"]=np.nan
    q,r=prepare_panel_residuals(values,config())
    eligible=q.loc[q.municipality_id.eq("2"),"eligible_at_origin"].tolist()
    assert eligible==[True,True,False]
    absent=r.loc[r.series_id.eq("2")&r.observation_period.eq("2024-03")]
    assert absent.y_pred.isna().all() and absent.model.isna().all()


def test_added_future_first_uid_does_not_rewrite_existing_prediction_and_replay_deterministic():
    original=panel()
    firstq,first=prepare_panel_residuals(original,config())
    extended=original.assign(new=np.nan)
    extended.loc["2024-03-01","new"]=500.
    secondq,second=prepare_panel_residuals(extended,config())
    assert len(secondq)==9
    assert not secondq.loc[secondq.municipality_id.eq("new"),"eligible_at_origin"].any()
    audit=compare_saved_pilot(second,first)
    assert audit["passed"] and audit["rows"]==6 and audit["max_abs_difference"]["y_pred"]==0
    repeatq,repeat=prepare_panel_residuals(extended,copy.deepcopy(config()))
    pd.testing.assert_frame_equal(secondq,repeatq)
    pd.testing.assert_frame_equal(second,repeat)
