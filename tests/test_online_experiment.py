"""Synthetic tests for validation-only selection and protected experiment paths."""
from pathlib import Path

import pandas as pd
import pytest

from sberforecast.online_experiment import candidates, real_diagnostics, safe_path, select_candidate, validate_output_path


def test_grid_is_fixed_cartesian_and_reproducible():
    grid = {"threshold": [3, 5], "allowance": [.5, 1.]}
    assert candidates(grid) == [
        {"allowance": .5, "threshold": 3}, {"allowance": .5, "threshold": 5},
        {"allowance": 1., "threshold": 3}, {"allowance": 1., "threshold": 5},
    ]
    assert candidates(grid) == candidates(grid)


def test_validation_budget_does_not_relax_for_better_f1():
    rows = pd.DataFrame([
        {"candidate_id": "A", "primary_f1": .9, "primary_recall": .9, "max_control_far": 1.01, "primary_median_delay": 0},
        {"candidate_id": "B", "primary_f1": .2, "primary_recall": .3, "max_control_far": 1.0, "primary_median_delay": 2},
    ])
    assert select_candidate(rows, 1.0).candidate_id == "B"
    assert select_candidate(rows.iloc[:1], 1.0) is None


def test_selection_ties_follow_predeclared_order():
    rows = pd.DataFrame([
        {"candidate_id": "B", "primary_f1": .5, "primary_recall": .6, "max_control_far": .1, "primary_median_delay": 1},
        {"candidate_id": "A", "primary_f1": .5, "primary_recall": .6, "max_control_far": .1, "primary_median_delay": 1},
        {"candidate_id": "C", "primary_f1": .5, "primary_recall": .5, "max_control_far": .0, "primary_median_delay": 0},
    ])
    assert select_candidate(rows, 1.0).candidate_id == "A"


def test_no_path_outside_project(tmp_path):
    with pytest.raises(ValueError):
        safe_path(tmp_path, "../outside")
    assert safe_path(tmp_path, "outputs/new") == (tmp_path / "outputs/new").resolve()


@pytest.mark.parametrize("name", ["outputs", "outputs/prophet_comparison_v1/new", "outputs/e02_data_audit/new", "data/new"])
def test_previous_outputs_cannot_be_used_even_as_ancestor(tmp_path, name):
    with pytest.raises(ValueError):
        validate_output_path(tmp_path, (tmp_path / name).resolve())


def test_real_agreement_keeps_header_when_only_one_method_selected(tmp_path, monkeypatch):
    import sberforecast.online_experiment as experiment
    monkeypatch.setattr(experiment, "plot_examples", lambda *args: None)
    residuals = pd.DataFrame({"series_id": "1", "observation_period": pd.period_range("2024-01", periods=6, freq="M").astype(str),
                              "y_true": [100, 101, 99, 100, 100, 100], "y_pred": 100.})
    cfg = {"preparation": {"warmup_observations": 4}, "illustrations": {"n_smallest": 3},
           "source": {"selection_timing": "chosen_after_viewing_E01"}}
    selected = {"CUSUM": {"selected": True, "parameters": {"threshold": 5, "allowance": .5}},
                "EWMA": {"selected": False}, "BOCPD": {"selected": False}}
    real_diagnostics((residuals, pd.DataFrame(), pd.DataFrame(), ["1"]), selected, cfg, tmp_path)
    agreement = pd.read_csv(tmp_path / "real_agreement.csv")
    assert agreement.empty
    assert {"method_a", "method_b", "jaccard"}.issubset(agreement.columns)


def test_no_selected_detector_still_reports_data_coverage(tmp_path):
    residuals = pd.DataFrame({"series_id": "1", "observation_period": pd.period_range("2024-01", periods=6, freq="M").astype(str),
                              "y_true": [100, 101, 99, 100, 100, 100], "y_pred": 100.})
    selected = {method: {"selected": False} for method in ("CUSUM", "EWMA", "BOCPD")}
    status = real_diagnostics((residuals, pd.DataFrame(), pd.DataFrame(), ["1"]), selected,
                              {"preparation": {"warmup_observations": 4}}, tmp_path)
    assert status["n_observed_monitoring_months_per_method"] == 2
    assert pd.read_csv(tmp_path / "real_coverage.csv").n_warmup_months.tolist() == [4]
