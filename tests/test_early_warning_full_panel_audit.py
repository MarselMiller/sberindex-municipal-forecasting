"""Synthetic audits distinguish label states, finite values and municipal copies."""
import numpy as np
import pandas as pd
import pytest

from sberforecast.early_warning_full_panel_audit import (
    NEWS_INFORMATION_COLUMN, build_feature_summary, build_news_summary, build_state_counts,
)


@pytest.fixture
def wide():
    return pd.DataFrame({"municipality_id": ["a", "b", "a", "b"],
        "forecast_origin": ["2024-06-30"] * 2 + ["2024-07-31"] * 2,
        "expense": [10., np.nan, 12., 15.], "expense_missing": [False, True, False, False],
        "detector_score": [0.] * 4,
        "news_count_90d": [1., 1., 3., 3.], "news_alias": [1., 1., 3., 3.],
        "news_count_90d_missing": [False] * 4, "news_alias_missing": [False] * 4})


@pytest.fixture
def groups():
    return dict(A=("expense", "expense_missing"), B=("detector_score",), C=(),
                D=("news_count_90d", "news_alias", "news_count_90d_missing", "news_alias_missing"))


@pytest.fixture
def cases(wide):
    records = []
    for k in (1, 3):
        for row in wide[["municipality_id", "forecast_origin"]].to_dict("records"):
            june = row["forecast_origin"] == "2024-06-30"
            records.append(dict(**row, k=k, split="train" if june else "test",
                known_active_at_origin=row["municipality_id"] == "a" and not june,
                retrospective_inside_regime_at_origin=row["municipality_id"] == "a",
                state_uncertain=row["municipality_id"] == "b" and june,
                eligible_at_origin=not (row["municipality_id"] == "b" and june)))
    return pd.DataFrame(records)


@pytest.fixture
def coverage(wide, groups):
    records = []
    for origin in wide.forecast_origin.unique():
        as_of = pd.Timestamp(origin).tz_localize("Europe/Moscow").tz_convert("UTC")
        for group, names in groups.items():
            for feature in names:
                records.append(dict(forecast_origin=origin, feature=feature, as_of_utc=as_of,
                    max_dependency_available_at=as_of - pd.Timedelta(days=1), dependency_violations=0,
                    dependency_audit="conservative_news_bound" if group == "D" else "exact_parent_maximum"))
    return pd.DataFrame(records)


def test_state_counts_keep_every_case_and_separate_causal_from_retrospective(cases):
    result = build_state_counts(cases)
    assert len(result) == 4 and result.cases.sum() == len(cases)
    assert result.cases.tolist() == [2, 2, 2, 2]
    june = result.loc[result.forecast_origin.str.startswith("2024-06-30")]
    assert june.known_active_cases.eq(0).all()
    assert june.retrospective_inside_cases.eq(1).all()
    assert june.retrospective_only_cases.eq(1).all()
    assert june.state_uncertain_cases.eq(1).all()
    assert june.eligible_cases.eq(1).all()
    assert not result.rows_filtered.any()


def test_recovery_can_leave_known_active_without_retrospective_membership(cases):
    row = cases.index[cases.known_active_at_origin][0]
    cases.loc[row, "retrospective_inside_regime_at_origin"] = False
    result = build_state_counts(cases)
    assert result.known_active_only_cases.sum() == 1
    assert result.retrospective_only_cases.sum() == 2


def test_feature_coverage_uses_separate_denominators_for_values_and_flags(wide, coverage, groups):
    result = build_feature_summary(wide, coverage, groups).set_index(["group", "role"])
    assert result.loc[("A", "value"), "finite_cells"] == 3
    assert result.loc[("A", "value"), "finite_cell_coverage"] == .75
    assert result.loc[("A", "missing_flag"), "finite_cell_coverage"] == 1.
    assert result.loc[("D", "value"), "exact_duplicate_columns_retained"] == 1
    assert result.loc[("D", "value"), "dependency_audit_kinds"] == "conservative_news_bound"
    assert result.temporal_dependency_check_passed.all()
    assert not result.selection_applied.any()


def test_future_dependency_is_recomputed_even_when_reported_counter_is_zero(wide, coverage, groups):
    position = coverage.index[coverage.feature.eq("expense")][0]
    coverage.loc[position, "max_dependency_available_at"] = coverage.loc[position, "as_of_utc"] + pd.Timedelta(hours=1)
    result = build_feature_summary(wide, coverage, groups)
    row = result.loc[result.group.eq("A") & result.role.eq("value")].iloc[0]
    assert row.reported_dependency_violations == 0
    assert row.computed_future_dependency_records == 1
    assert row.dependency_violation_records == 1
    assert not row.temporal_dependency_check_passed


def test_forged_later_as_of_cannot_hide_dependency_after_own_origin(wide, coverage, groups):
    position = coverage.index[coverage.feature.eq("expense")][0]
    real_origin = coverage.loc[position, "as_of_utc"]
    coverage.loc[position, "as_of_utc"] = real_origin + pd.Timedelta(days=30)
    coverage.loc[position, "max_dependency_available_at"] = real_origin + pd.Timedelta(days=1)
    result = build_feature_summary(wide, coverage, groups)
    row = result.loc[result.group.eq("A") & result.role.eq("value")].iloc[0]
    assert row.coverage_origin_timestamp_mismatches == 1 and row.computed_future_dependency_records == 1
    assert not row.temporal_dependency_check_passed


def test_missing_lineage_records_are_not_silently_reported_as_passed(wide, coverage, groups):
    result = build_feature_summary(wide, coverage.iloc[1:], groups)
    row = result.loc[result.group.eq("A") & result.role.eq("value")].iloc[0]
    assert row.missing_dependency_coverage_records == 1
    assert not row.temporal_dependency_check_passed


def test_infinity_is_nonfinite_and_not_mislabeled_as_missing(wide, coverage, groups):
    wide["detector_score"] = np.inf
    result = build_feature_summary(wide, coverage, groups)
    row = result.loc[result.group.eq("B") & result.role.eq("value")].iloc[0]
    assert row.nonfinite_nonmissing_cells == 4 and row.finite_cells == 0
    assert row.all_missing_columns == 0


def test_national_municipal_copies_are_not_independent_news_vectors(cases, wide, groups):
    result = build_news_summary(cases, wide, groups)
    assert len(result) == 6 and NEWS_INFORMATION_COLUMN == "news_count_90d"
    row = result.loc[result.k.eq(1) & result.scope.eq("all")].iloc[0]
    assert row.cases == 4 and row.forecast_origins == 2
    assert row.unique_temporal_vectors == 2 and row.unique_news_full_vectors == 2
    assert row.maximum_spatial_full_vectors_per_origin == 1
    assert row.varying_value_columns == 2 and row.varying_missing_flag_columns == 0
    assert row.retained_value_duplicate_columns == 1
    assert row.origins_with_news_information == 2
    assert row.cases_with_news_information == 4
    assert not row.source_archive_complete and not row.selection_applied


def test_missing_mask_variation_does_not_become_observed_news_value_variation(cases, wide, groups):
    july = wide.forecast_origin.eq("2024-07-31")
    wide.loc[july, ["news_count_90d", "news_alias"]] = np.nan
    wide.loc[july, ["news_count_90d_missing", "news_alias_missing"]] = True
    row = build_news_summary(cases, wide, groups).loc[lambda table: table.k.eq(1) & table.scope.eq("all")].iloc[0]
    assert row.varying_value_columns == 0
    assert row.strict_varying_value_columns_including_missing == 2
    assert row.varying_missing_flag_columns == 2
    assert row.origins_with_news_information == 1 and row.origins_with_unknown_news_count == 1


def test_spatial_news_variation_is_disclosed_not_called_one_national_temporal_vector(cases, wide, groups):
    wide.loc[3, ["news_count_90d", "news_alias"]] = 4.
    row = build_news_summary(cases, wide, groups).loc[lambda table: table.k.eq(1) & table.scope.eq("all")].iloc[0]
    assert row.maximum_spatial_value_vectors_per_origin == 2
    assert row.unique_news_value_vectors == 3
    assert np.isnan(row.unique_temporal_vectors)
    assert not row.one_common_news_vector_per_origin


def test_future_test_values_cannot_change_train_news_audit(cases, wide, groups):
    first = build_news_summary(cases, wide, groups)
    changed = wide.copy()
    changed.loc[changed.forecast_origin.eq("2024-07-31"), ["news_count_90d", "news_alias"]] = 50000.
    second = build_news_summary(cases, changed, groups)
    pd.testing.assert_frame_equal(first.loc[first.scope.eq("train")], second.loc[second.scope.eq("train")])


def test_empty_train_scope_stays_explicit_with_undefined_cell_coverage(cases, wide, groups):
    cases.loc[cases.k.eq(3) & cases.split.eq("train"), "split"] = "excluded"
    row = build_news_summary(cases, wide, groups).loc[lambda table: table.k.eq(3) & table.scope.eq("train")].iloc[0]
    assert row.scope_empty and row.cases == 0 and row.varying_value_columns == 0
    assert np.isnan(row.news_value_cell_coverage) and row.unique_temporal_vectors == 0


def test_missing_case_features_and_duplicate_keys_are_rejected(cases, wide, groups):
    with pytest.raises(ValueError, match="absent"):
        build_news_summary(cases, wide.iloc[1:], groups)
    with pytest.raises(ValueError, match="unique"):
        build_news_summary(cases, pd.concat([wide, wide.iloc[:1]]), groups)
    with pytest.raises(ValueError, match="Duplicate"):
        build_state_counts(pd.concat([cases, cases.iloc[:1]]))


def test_state_uncertainty_never_gets_filled_and_inputs_remain_unchanged(cases, wide, coverage, groups):
    snapshots = [frame.copy(deep=True) for frame in (cases, wide, coverage)]
    build_state_counts(cases)
    build_feature_summary(wide, coverage, groups)
    build_news_summary(cases, wide, groups)
    for frame, before in zip((cases, wide, coverage), snapshots):
        pd.testing.assert_frame_equal(frame, before)
    cases["state_uncertain"] = cases.state_uncertain.astype(object)
    cases.loc[0, "state_uncertain"] = None
    with pytest.raises(ValueError, match="explicit nonmissing"):
        build_state_counts(cases)
