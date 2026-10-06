"""Проверки временной доступности на синтетических данных, без fit моделей."""
import numpy as np
import pandas as pd
import pytest

from sberforecast.data import eligibility, get_prefix
from sberforecast.direct_training import PAIR_KEY, build_direct_training, direct_features
from sberforecast.features import next_features, supervised_training


def build(panel, origin, h=1, mode="legacy", lag=0, staleness=None):
    return build_direct_training(panel, origin, h, release_lag_months=lag,
                                 mode=mode, max_staleness_months=staleness)


@pytest.mark.parametrize("mode", ["legacy", "strict12"])
@pytest.mark.parametrize("lag", [0, 1])
@pytest.mark.parametrize("h", [1, 3])
def test_values_after_model_cutoff_change_nothing(panel, mode, lag, h):
    origin = pd.Period("2024-07", freq="M")
    original = panel.copy(deep=True)
    changed = panel.copy()
    changed.loc[changed.index > (origin - lag).to_timestamp()] = np.nan
    X, y, meta = build(panel, origin, h, mode, lag)
    X2, y2, meta2 = build(changed, origin, h, mode, lag)
    assert len(meta) > 0
    pd.testing.assert_frame_equal(X, X2)
    np.testing.assert_array_equal(y, y2)
    pd.testing.assert_frame_equal(meta, meta2)
    pd.testing.assert_frame_equal(panel, original)


@pytest.mark.parametrize("mode", ["legacy", "strict12"])
@pytest.mark.parametrize("lag", [0, 1])
def test_example_features_use_its_own_cutoff_but_label_can_change(panel, mode, lag):
    r = pd.Period("2024-01", freq="M")
    changed = panel.copy()
    changed.loc[changed.index > (r - lag).to_timestamp()] *= 100
    X, y, meta = build(panel, "2024-06", 3, mode, lag)
    X2, y2, meta2 = build(changed, "2024-06", 3, mode, lag)
    take = meta.historical_origin.eq(r.end_time.normalize())
    take2 = meta2.historical_origin.eq(r.end_time.normalize())
    assert take.sum() == take2.sum() == 2
    pd.testing.assert_frame_equal(meta.loc[take, PAIR_KEY].reset_index(drop=True),
                                  meta2.loc[take2, PAIR_KEY].reset_index(drop=True))
    pd.testing.assert_frame_equal(X.loc[take].reset_index(drop=True),
                                  X2.loc[take2].reset_index(drop=True))
    assert not np.array_equal(y[take], y2[take2])


def test_lag_limits_both_feature_cutoff_and_label_release_at_boundary(panel):
    X, y, meta = build(panel, "2023-04", lag=1)
    assert len(meta) == 2
    assert meta.historical_origin.eq(pd.Timestamp("2023-02-28")).all()
    assert meta.feature_cutoff.eq(pd.Timestamp("2023-01-31")).all()
    assert meta.target_period.eq(pd.Timestamp("2023-03-01")).all()
    assert meta.target_available_at.eq(pd.Timestamp("2023-04-30")).all()
    assert meta.model_origin.eq(pd.Timestamp("2023-04-30")).all()
    assert meta.n_history_observations.eq(1).all()
    np.testing.assert_array_equal(X.lag_1, [100, 200])
    assert X.lag_2.isna().all()
    np.testing.assert_array_equal(y, [2, 4])
    assert build(panel, "2023-03", lag=1)[2].empty


def test_strict_lag_first_label_is_available_only_in_march(panel):
    assert build(panel, "2024-02", mode="strict12", lag=1)[2].empty
    meta = build(panel, "2024-03", mode="strict12", lag=1)[2]
    assert len(meta) == 2
    assert meta.historical_origin.eq(pd.Timestamp("2024-01-31")).all()
    assert meta.feature_cutoff.eq(pd.Timestamp("2023-12-31")).all()
    assert meta.target_period.eq(pd.Timestamp("2024-02-01")).all()
    assert meta.target_available_at.eq(pd.Timestamp("2024-03-31")).all()


def test_missing_calendar_rows_do_not_shift_lags_or_target(panel):
    gapped = panel.drop(pd.Timestamp("2023-02-01"))
    X, y, meta = build(gapped, "2023-06", 3)
    take = meta.historical_origin.eq(pd.Timestamp("2023-03-31"))
    assert take.sum() == 2
    np.testing.assert_array_equal(X.loc[take, "lag_1"], [102, 204])
    assert X.loc[take, "lag_2"].isna().all()
    np.testing.assert_array_equal(X.loc[take, "lag_3"], [100, 200])
    assert meta.loc[take, "target_period"].eq(pd.Timestamp("2023-06-01")).all()
    assert meta.loc[take, "n_history_observations"].eq(2).all()
    np.testing.assert_array_equal(y[take], [3, 6])
    assert X.loc[take, "month"].eq(6).all()


def test_missing_target_month_is_not_replaced_by_next_observation(panel):
    gapped = panel.drop(pd.Timestamp("2023-04-01"))
    meta = build(gapped, "2023-06", 3)[2]
    assert not meta.historical_origin.eq(pd.Timestamp("2023-01-31")).any()


@pytest.mark.parametrize("h", [1, 3, 6, 12])
def test_strict12_december_has_no_pairs(panel, h):
    X, y, meta = build(panel, "2023-12", h, "strict12")
    assert X.empty and y.size == 0 and meta.empty
    assert "lag_12" in X and set(PAIR_KEY) <= set(meta.columns)


@pytest.mark.parametrize("origin", pd.period_range("2023-12", "2024-11", freq="M"))
def test_strict12_yearly_has_no_pairs_through_november(panel, origin):
    assert build(panel, origin, 12, "strict12")[2].empty


@pytest.mark.parametrize("mode", ["legacy", "strict12"])
def test_both_modes_have_no_past_yearly_pairs_in_december_2023(panel, mode):
    assert build(panel, "2023-12", 12, mode)[2].empty


def test_legacy_h1_matches_actual_legacy_keys_features_and_labels(panel):
    changed = panel.copy()
    changed.loc["2023-03-01":"2023-06-01", "1"] = np.nan
    changed.loc["2023-01-01":"2023-04-01", "2"] = np.nan
    prefix = get_prefix(changed, pd.Period("2023-12", freq="M"), 0)
    old_X, old_y, old_summary = supervised_training(prefix)
    # Восстановление ключей по фактическому порядку старого сборщика:
    # следующая цель, исходный порядок МО, фильтр finite(truth) & finite(anchor).
    old_keys = []
    for idx in range(1, len(prefix)):
        date = prefix.index[idx]
        _, anchor = next_features(prefix.iloc[:idx], date)
        valid = np.isfinite(prefix.iloc[idx]) & np.isfinite(anchor)
        for uid in prefix.columns[valid]:
            old_keys.append((str(uid), date.to_period("M") - 1, date, 1))
    X, y, meta = build(changed, "2023-12")
    new_keys = [(row.municipality_id, row.historical_origin.to_period("M"),
                 row.target_period, row.horizon) for row in meta.itertuples()]
    assert new_keys == old_keys  # Проверяем реальные ключи, не только длину.
    pd.testing.assert_frame_equal(X, old_X)
    np.testing.assert_array_equal(y, old_y)
    assert len(meta) == old_summary["n_training_rows"]
    assert meta.n_history_observations.min() == 1
    assert meta.staleness_months.max() > 1


def test_legacy_with_lag_intentionally_differs_from_one_step_training(panel):
    prefix = get_prefix(panel, pd.Period("2023-04", freq="M"), 1)
    old_X, old_y, _ = supervised_training(prefix)
    X, y, meta = build(panel, "2023-04", lag=1)
    assert len(old_X) == 4 and len(X) == 2
    # Старый сборщик учит январь -> февраль и февраль -> март;
    # прямой прогноз h=1, L=1 учит январские признаки -> мартовскую цель.
    np.testing.assert_array_equal(old_y, [1, 2, 1, 2])
    np.testing.assert_array_equal(y, [2, 4])
    assert meta.target_period.eq(pd.Timestamp("2023-03-01")).all()


def test_training_history_is_separate_from_current_forecast_cohort(panel):
    changed = panel.copy()
    changed.loc["2024-01-01":, "1"] = np.nan
    cohort = eligibility(get_prefix(changed, pd.Period("2024-06", freq="M"), 0), 12, 1)
    assert not cohort.set_index("municipality_id").loc["1", "eligible"]
    _, _, meta = build(changed, "2024-06")
    assert "1" in set(meta.municipality_id)


def test_staleness_is_historical_and_filter_is_explicit(panel):
    changed = panel.copy()
    changed.loc["2023-02-01":"2023-04-01", "1"] = np.nan
    _, _, meta = build(changed, "2023-06")
    take = meta.municipality_id.eq("1") & meta.historical_origin.eq(pd.Timestamp("2023-04-30"))
    assert take.sum() == 1
    assert meta.loc[take, "staleness_months"].iloc[0] == 3
    assert meta.loc[take, "anchor"].iloc[0] == 100
    filtered = build(changed, "2023-06", staleness=1)[2]
    assert not (filtered.municipality_id.eq("1") &
                filtered.historical_origin.eq(pd.Timestamp("2023-04-30"))).any()


def test_strict12_counts_facts_instead_of_calendar_rows(panel):
    changed = panel.copy()
    changed.loc["2023-02-01", "1"] = np.nan
    meta = build(changed, "2024-01", mode="strict12")[2]
    assert meta.municipality_id.tolist() == ["2"]
    assert meta.n_history_observations.tolist() == [12]


def test_nonfinite_anchor_and_target_are_excluded(panel):
    changed = panel.copy()
    changed.loc["2023-01-01", "1"] = np.inf
    changed.loc["2023-02-01", "2"] = -np.inf
    assert build(changed, "2023-02")[2].empty


def test_direct_h1_features_are_exactly_legacy_features(panel):
    history = panel.iloc[:12]
    X, anchor = direct_features(history, pd.Timestamp("2024-01-01"))
    old_X, old_anchor = next_features(history, pd.Timestamp("2024-01-01"))
    pd.testing.assert_frame_equal(X, old_X)
    np.testing.assert_array_equal(anchor, old_anchor)


def test_no_available_history_returns_empty_schema(panel):
    X, y, meta = build(panel, "2022-12")
    assert X.empty and len(y) == len(meta) == 0


@pytest.mark.parametrize("kwargs", [
    {"horizon": 0}, {"horizon": True}, {"release_lag_months": -1},
    {"release_lag_months": 0.5}, {"mode": "automatic"}, {"max_staleness_months": -1},
])
def test_invalid_rules_are_rejected(panel, kwargs):
    args = dict(horizon=1, release_lag_months=0, mode="legacy")
    args.update(kwargs)
    with pytest.raises(ValueError):
        build_direct_training(panel, "2023-12", **args)
