"""E06a protocol, provenance and stage gates on small synthetic fixtures only."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from sberforecast import offline_experiment as experiment
from sberforecast.data import sha256_file
from sberforecast.offline_detection import BREAKPOINT_COLUMNS
from sberforecast.offline_evaluation import COUNTS


PROJECT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


@pytest.fixture
def specification(tmp_path):
    cfg = yaml.safe_load((PROJECT / "configs/offline_detection.yaml").read_text(encoding="utf-8"))
    old = yaml.safe_load((PROJECT / "configs/online_detection.yaml").read_text(encoding="utf-8"))
    old_path = tmp_path / cfg["reference"]["e04_config"]
    old_path.parent.mkdir(parents=True)
    old_path.write_text(yaml.safe_dump(old, allow_unicode=True), encoding="utf-8")
    config_path = tmp_path / "configs/offline_detection.yaml"
    config_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return tmp_path, cfg, old, config_path


@pytest.fixture
def checked_project(specification, monkeypatch):
    root, cfg, old, path = specification
    for name in ["src/sberforecast/offline_stub.py", "tests/test_offline_stub.py",
                 "scripts/run_offline_detection.py", "requirements-ruptures.txt"]:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# synthetic fixture only\n", encoding="utf-8")
    protected = root / "legacy.txt"
    protected.write_text("unchanged synthetic legacy fixture\n", encoding="utf-8")
    write_json(root / cfg["preservation_record"], dict(packages={"numpy": "2.3.5"},
               protected_files={"legacy.txt": sha256_file(protected)}))
    write_json(root / cfg["environment_record"], dict(version="1.1.10", pip_check_exit_code=0))
    write_json(root / cfg["validation_record"], dict(full_pytest=True, exit_code=0,
               code_sha256=experiment.tested_files(root)))
    monkeypatch.setattr(experiment.importlib.metadata, "distributions", lambda: [
        SimpleNamespace(metadata={"Name": "numpy"}, version="2.3.5"),
        SimpleNamespace(metadata={"Name": "ruptures"}, version="1.1.10"),
    ])
    return root, cfg, old, path


@pytest.fixture
def runnable(checked_project, monkeypatch):
    root, cfg, old, path = checked_project
    monkeypatch.setattr(experiment, "verify_source", lambda *args: {"legacy.txt": sha256_file(root / "legacy.txt")})
    monkeypatch.setattr(experiment.subprocess, "check_output", lambda argv, **kwargs: "fake_head\n" if "rev-parse" in argv else "?? fake_new_file\n")
    source = root / cfg["reference"]["e04_dir"]
    source.mkdir(parents=True)
    for name in ["synthetic_test_metrics.csv.gz", "synthetic_test_uncertainty.csv.gz"]:
        pd.DataFrame({"synthetic_fixture": [1]}).to_csv(source / name, index=False)
    write_json(source / "selected_parameters.json", {"CUSUM": {"selected": True}})
    write_json(root / cfg["smoke_record"], dict(passed=True, code_sha256=experiment.tested_files(root),
               config_sha256=sha256_file(path)))
    return root, cfg, old, path


def generated(split="validation", *, seed=None):
    sid = f"{split}_synthetic"
    periods = ["2024-01", "2024-02"]
    residuals = pd.DataFrame(dict(series_id=[sid] * 2, observation_period=periods,
                                  y_true=[100.0, 100.0], y_pred=[100.0, 100.0]))
    events = pd.DataFrame(columns=["series_id", "event_id", "event_start_period"])
    info = pd.DataFrame([dict(series_id=sid, split=split, seed=100000 if split == "validation" else 200000,
                             replicate=0, scenario="no_change", strength=0.0, noise_fraction=0.01)])
    if seed is not None:
        info["seed"] = seed
    observations = pd.DataFrame(dict(series_id=[sid] * 2, observation_period=periods, value=[100.0, 100.0]))
    return residuals, events, info, observations


def fake_selection(output):
    selected = {method: dict(selected=True, parameters=dict(model="l2", min_size=2, jump=1, penalty=2.0))
                for method in experiment.METHODS}
    experiment.save_json(output / "selected_parameters.json", selected)
    pd.DataFrame({"candidate_id": ["synthetic"]}).to_csv(output / "validation_selection.csv", index=False)
    experiment.save_json(output / "selection_seal.json", dict(
        selected_on="synthetic_validation_only", test_generated=False,
        selection_sha256=sha256_file(output / "selected_parameters.json"),
        candidate_table_sha256=sha256_file(output / "validation_selection.csv")))
    return selected


def fake_synthetic_dependencies(monkeypatch, root, cfg, *, test_seed=None, mutate_after_test=False):
    calls = []
    def verified(_root, _cfg, split):
        calls.append(("generate", split))
        if split == "test":
            output = root / cfg["output_dir"]
            seal = json.loads((output / "selection_seal.json").read_text(encoding="utf-8"))
            assert seal["test_generated"] is False
            assert seal["selected_on"] == "synthetic_validation_only"
            assert seal["selection_sha256"] == sha256_file(output / "selected_parameters.json")
        return generated(split, seed=test_seed if split == "test" else None)
    def tune(validation, _cfg, output):
        assert validation[2]["split"].tolist() == ["validation"]
        calls.append(("tune", "validation"))
        return fake_selection(output)
    def save(split, data, selected, _cfg, output):
        calls.append(("evaluate", split))
        (output / f"fixture_{split}.txt").write_text("synthetic fixture only\n", encoding="utf-8")
        if split == "test" and mutate_after_test:
            experiment.save_json(output / "selected_parameters.json", {"tampered": True})
    monkeypatch.setattr(experiment, "verified_synthetic", verified)
    monkeypatch.setattr(experiment, "tune", tune)
    monkeypatch.setattr(experiment, "save_synthetic", save)
    return calls


def synthetic_completed(runnable, monkeypatch):
    root, cfg, _, path = runnable
    calls = fake_synthetic_dependencies(monkeypatch, root, cfg)
    status = experiment.run_experiment(root, path, stage="synthetic", command="synthetic fixture command")
    output = root / cfg["output_dir"]
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    record = dict(passed=True, artifact_sha256=manifest["artifact_sha256"],
                  selection_sha256=sha256_file(output / "selected_parameters.json"),
                  code_sha256=experiment.tested_files(root), config_sha256=sha256_file(path))
    write_json(root / cfg["synthetic_verification_record"], record)
    return status, calls, manifest, record


def test_exact_shared_protocol_and_disjoint_seeds_are_accepted(specification):
    root, cfg, old, _ = specification
    assert experiment.validate_specification(root, cfg) == old
    assert cfg["generator"]["validation_seed"] == 100000
    assert cfg["generator"]["test_seed"] == 200000


@pytest.mark.parametrize("section,field,value", [
    ("source", "horizon", 3), ("preparation", "warmup_observations", 3),
    ("forecast", "yearly_growth_window", 4), ("generator", "validation_seed", 1),
    ("generator", "test_seed", 100000), ("generator", "months", 36),
    ("illustrations", "n_smallest", 4),
    ("evaluation", "detection_window_months", 2),
    ("evaluation", "false_alarm_budget_per_12_months", 2.0),
    ("evaluation", "budget_controls", ["no_change"]),
    ("evaluation", "bootstrap_replicates", 10),
    ("evaluation", "matching", "nearest_before_or_after"),
    ("evaluation", "selection_rule", "best_test_F1"),
    ("offline_analysis", "labels_in_fit", True),
    ("offline_analysis", "future_access", "online"),
    ("offline_analysis", "prefix_matching_tolerance_months", 2),
    ("offline_analysis", "breakpoint_month", "last_observed_month_of_left_segment"),
])
def test_shared_config_and_calendar_rules_cannot_be_silently_changed(specification, section, field, value):
    root, cfg, _, _ = specification
    cfg[section][field] = value
    with pytest.raises(ValueError):
        experiment.validate_specification(root, cfg)


@pytest.mark.parametrize("field,value", [
    ("seed", 7), ("ruptures_version", "1.1.9"), ("early_warning", True),
    ("forecasting_model_search_closed", False), ("output_dir", "outputs/online_detection_v1"),
    ("smoke_output_dir", "outputs/online_detection_v1"),
])
def test_top_level_protocol_is_fixed(specification, field, value):
    root, cfg, _, _ = specification
    cfg[field] = value
    with pytest.raises(ValueError):
        experiment.validate_specification(root, cfg)


@pytest.mark.parametrize("field,value", [("model", "rbf"), ("min_size", 3), ("jump", 2),
                                         ("penalty", [0.01, 0.02])])
def test_penalty_grid_and_cost_are_fixed_before_validation(specification, field, value):
    root, cfg, _, _ = specification
    cfg["detectors"]["PELT"][field] = value
    with pytest.raises(ValueError):
        experiment.validate_specification(root, cfg)


def test_require_checks_accepts_only_full_current_tests_and_single_install(checked_project):
    root, cfg, _, _ = checked_project
    tests, environment, initial = experiment.require_checks(root, cfg)
    assert tests["full_pytest"] and tests["code_sha256"] == experiment.tested_files(root)
    assert environment["version"] == "1.1.10"
    assert list(initial["protected_files"]) == ["legacy.txt"]


@pytest.mark.parametrize("mutation", ["partial_tests", "failed_tests", "code", "old_file", "pip_check", "version"])
def test_failed_or_stale_checks_stop_before_any_detector_fit(checked_project, mutation):
    root, cfg, _, _ = checked_project
    tests_path = root / cfg["validation_record"]
    env_path = root / cfg["environment_record"]
    if mutation in {"partial_tests", "failed_tests"}:
        tests = json.loads(tests_path.read_text(encoding="utf-8"))
        tests["full_pytest" if mutation == "partial_tests" else "exit_code"] = False if mutation == "partial_tests" else 1
        write_json(tests_path, tests)
    elif mutation == "code":
        (root / "src/sberforecast/offline_stub.py").write_text("changed\n", encoding="utf-8")
    elif mutation == "old_file":
        (root / "legacy.txt").write_text("changed\n", encoding="utf-8")
    else:
        env = json.loads(env_path.read_text(encoding="utf-8"))
        env["pip_check_exit_code" if mutation == "pip_check" else "version"] = 1 if mutation == "pip_check" else "1.1.9"
        write_json(env_path, env)
    with pytest.raises(ValueError):
        experiment.require_checks(root, cfg)


def test_unrelated_package_changes_are_rejected(checked_project, monkeypatch):
    root, cfg, _, _ = checked_project
    monkeypatch.setattr(experiment.importlib.metadata, "distributions", lambda: [
        SimpleNamespace(metadata={"Name": "numpy"}, version="2.4.0"),
        SimpleNamespace(metadata={"Name": "ruptures"}, version="1.1.10"),
    ])
    with pytest.raises(ValueError, match="authorized"):
        experiment.require_checks(root, cfg)


def test_hash_inventory_keeps_nested_manifests_and_excludes_only_own(tmp_path):
    (tmp_path / "run_manifest.json").write_text("root manifest", encoding="utf-8")
    nested = tmp_path / "nested/run_manifest.json"
    nested.parent.mkdir()
    nested.write_text("nested artifact", encoding="utf-8")
    assert experiment.artifact_hashes(tmp_path) == {"nested/run_manifest.json": sha256_file(nested)}


def test_source_equality_checks_actual_keys_not_only_counts():
    actual = pd.DataFrame({"series_id": ["a", "b"], "observation_period": ["2024-01", "2024-01"], "value": [1.0, 2.0]})
    saved = actual.copy()
    saved.loc[1, "series_id"] = "c"
    with pytest.raises(AssertionError):
        experiment.require_same_frames(actual, saved, ["series_id", "observation_period"])
    with pytest.raises(ValueError):
        experiment.require_same_frames(pd.concat([actual, actual.iloc[[0]]]), actual, ["series_id", "observation_period"])
    experiment.require_same_frames(actual.iloc[::-1], actual, ["series_id", "observation_period"])


def test_technical_smoke_has_no_parameter_selection_or_test_generation(runnable, monkeypatch):
    root, cfg, _, path = runnable
    calls = []
    monkeypatch.setattr(experiment, "verified_synthetic", lambda _root, _cfg, split: calls.append(split) or generated(split))
    monkeypatch.setattr(experiment, "tune", lambda *args: pytest.fail("Smoke must not tune"))
    monkeypatch.setattr(experiment, "save_synthetic", lambda *args: None)
    status = experiment.run_experiment(root, path, stage="smoke", command="synthetic smoke fixture")
    assert calls == ["validation"]
    assert status["complete"] and status["parameters_selected"] is False
    assert status["n_test_series"] == 0 and status["real_complete"] is False
    assert (root / cfg["smoke_output_dir"] / "run_manifest.json").exists()
    assert not (root / cfg["output_dir"]).exists()


def test_selection_is_sealed_before_test_is_generated(runnable, monkeypatch):
    root, cfg, _, path = runnable
    calls = fake_synthetic_dependencies(monkeypatch, root, cfg)
    status = experiment.run_experiment(root, path, stage="synthetic", command="synthetic fixture")
    assert calls == [("generate", "validation"), ("tune", "validation"),
                     ("evaluate", "validation"), ("generate", "test"), ("evaluate", "test")]
    assert status["synthetic_complete"] and not status["real_complete"] and not status["complete"]
    manifest = json.loads((root / cfg["output_dir"] / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["selected_parameters_sealed_before_test"] is True
    assert manifest["selection_sha256"] == sha256_file(root / cfg["output_dir"] / "selected_parameters.json")


def test_validation_test_seed_overlap_stops_before_test_scoring(runnable, monkeypatch):
    root, cfg, _, path = runnable
    calls = fake_synthetic_dependencies(monkeypatch, root, cfg, test_seed=100000)
    with pytest.raises(ValueError, match="seeds/identifiers overlap"):
        experiment.run_experiment(root, path, stage="synthetic", command="synthetic fixture")
    assert ("evaluate", "test") not in calls


def test_selected_parameters_cannot_change_after_test(runnable, monkeypatch):
    root, cfg, _, path = runnable
    fake_synthetic_dependencies(monkeypatch, root, cfg, mutate_after_test=True)
    with pytest.raises(ValueError, match="after seeing test"):
        experiment.run_experiment(root, path, stage="synthetic", command="synthetic fixture")


@pytest.mark.parametrize("mutation", ["not_passed", "stale_code", "stale_config"])
def test_synthetic_requires_successful_current_smoke(runnable, monkeypatch, mutation):
    root, cfg, _, path = runnable
    record = dict(passed=mutation != "not_passed", code_sha256={} if mutation == "stale_code" else experiment.tested_files(root),
                  config_sha256="bad_sha" if mutation == "stale_config" else sha256_file(path))
    write_json(root / cfg["smoke_record"], record)
    monkeypatch.setattr(experiment, "verified_synthetic", lambda *args: pytest.fail("No generation before smoke gate"))
    with pytest.raises(ValueError, match="Successful smoke"):
        experiment.run_experiment(root, path, stage="synthetic", command="synthetic fixture")


def test_new_nonempty_directory_is_not_overwritten(runnable, monkeypatch):
    root, cfg, _, path = runnable
    output = root / cfg["output_dir"]
    output.mkdir(parents=True)
    foreign = output / "foreign.txt"
    foreign.write_text("do not overwrite", encoding="utf-8")
    monkeypatch.setattr(experiment, "verified_synthetic", lambda *args: pytest.fail("No generation"))
    with pytest.raises(FileExistsError):
        experiment.run_experiment(root, path, stage="synthetic", command="synthetic fixture")
    assert foreign.read_text(encoding="utf-8") == "do not overwrite"


def test_completed_synthetic_resume_does_not_fit_again(runnable, monkeypatch):
    root, cfg, _, path = runnable
    previous, _, _, _ = synthetic_completed(runnable, monkeypatch)
    monkeypatch.setattr(experiment, "verified_synthetic", lambda *args: pytest.fail("No repeated generation"))
    monkeypatch.setattr(experiment, "run_offline", lambda *args: pytest.fail("No repeated fitting"))
    assert experiment.run_experiment(root, path, stage="synthetic", command="synthetic resume fixture") == previous


@pytest.mark.parametrize("mutation", ["edit", "add", "delete"])
def test_changed_artifact_content_or_set_blocks_resume(runnable, monkeypatch, mutation):
    root, cfg, _, path = runnable
    synthetic_completed(runnable, monkeypatch)
    target = root / cfg["output_dir"] / "fixture_test.txt"
    if mutation == "edit":
        target.write_text("changed fixture", encoding="utf-8")
    elif mutation == "add":
        (target.parent / "extra.txt").write_text("new fixture", encoding="utf-8")
    else:
        target.unlink()
    monkeypatch.setattr(experiment, "prepare_real_source", lambda *args: pytest.fail("No real access before SHA checks"))
    with pytest.raises(ValueError, match="continuation prohibited"):
        experiment.run_experiment(root, path, stage="real", command="real fixture")


@pytest.mark.parametrize("mutation", ["passed", "artifact_sha256", "selection_sha256", "code_sha256", "config_sha256"])
def test_real_stage_requires_exact_passed_independent_synthetic_verification(runnable, monkeypatch, mutation):
    root, cfg, _, path = runnable
    _, _, _, record = synthetic_completed(runnable, monkeypatch)
    record[mutation] = False if mutation == "passed" else ({} if mutation in {"artifact_sha256", "code_sha256"} else "bad_sha")
    write_json(root / cfg["synthetic_verification_record"], record)
    monkeypatch.setattr(experiment, "prepare_real_source", lambda *args: pytest.fail("No real access before verification"))
    with pytest.raises(ValueError, match="Independent verification"):
        experiment.run_experiment(root, path, stage="real", command="real fixture")


def test_real_stage_cannot_start_without_saved_synthetic_results(runnable, monkeypatch):
    root, _, _, path = runnable
    monkeypatch.setattr(experiment, "prepare_real_source", lambda *args: pytest.fail("No real access"))
    with pytest.raises(ValueError, match="Synthetic results"):
        experiment.run_experiment(root, path, stage="real", command="real fixture")


def test_real_stage_opens_real_source_only_after_all_gates(runnable, monkeypatch):
    root, cfg, _, path = runnable
    synthetic_completed(runnable, monkeypatch)
    calls = []
    sentinel = object()
    def prepare(*args):
        calls.append("prepare")
        return sentinel
    def diagnose(real, selected, _cfg, output):
        assert real is sentinel and all(c["selected"] for c in selected.values())
        calls.append("diagnose")
        return dict(n_real_series=63, n_real_breakpoints=0, n_real_precision_recall_computed=0)
    monkeypatch.setattr(experiment, "prepare_real_source", prepare)
    monkeypatch.setattr(experiment, "real_diagnostics", diagnose)
    from sberforecast import offline_report
    monkeypatch.setattr(offline_report, "plot_saved", lambda *args: calls.append("plot"))
    report = root / cfg["report_path"]
    report.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(offline_report, "write_report", lambda *args: report.write_text("synthetic fixture report", encoding="utf-8"))
    status = experiment.run_experiment(root, path, stage="real", command="real fixture")
    assert calls == ["prepare", "diagnose", "plot"]
    assert status["complete"] and status["real_complete"]
    assert status["n_real_precision_recall_computed"] == 0


def test_completed_real_resume_verifies_external_report_hash(runnable, monkeypatch):
    root, cfg, _, path = runnable
    synthetic_completed(runnable, monkeypatch)
    monkeypatch.setattr(experiment, "prepare_real_source", lambda *args: object())
    monkeypatch.setattr(experiment, "real_diagnostics", lambda *args: dict(n_real_series=63, n_real_breakpoints=0))
    from sberforecast import offline_report
    monkeypatch.setattr(offline_report, "plot_saved", lambda *args: None)
    report = root / cfg["report_path"]
    report.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(offline_report, "write_report", lambda *args: report.write_text("synthetic fixture report", encoding="utf-8"))
    previous = experiment.run_experiment(root, path, stage="real", command="real fixture")
    assert experiment.run_experiment(root, path, stage="real", command="real resume fixture") == previous
    report.write_text("changed fixture report", encoding="utf-8")
    with pytest.raises(ValueError, match="report"):
        experiment.run_experiment(root, path, stage="real", command="real resume fixture")


@pytest.mark.parametrize("mutation", [None, "code", "artifact", "data", "configuration", "incomplete"])
def test_e04_source_manifest_config_data_and_artifacts_are_verified(specification, mutation):
    root, cfg, old, _ = specification
    directory = root / cfg["reference"]["e04_dir"]
    directory.mkdir(parents=True)
    old_code = root / "src/sberforecast/online_fixture.py"
    old_code.parent.mkdir(parents=True)
    old_code.write_text("synthetic old code fixture", encoding="utf-8")
    artifact = directory / "fixture.csv"
    artifact.write_text("synthetic old artifact fixture", encoding="utf-8")
    data = root / cfg["source"]["data_path"]
    data.parent.mkdir(parents=True)
    data.write_text("synthetic old data fixture", encoding="utf-8")
    report = root / cfg["reference"]["e04_report"]
    report.parent.mkdir(parents=True)
    report.write_text("synthetic old report fixture", encoding="utf-8")
    old_config = root / cfg["reference"]["e04_config"]
    manifest = dict(complete=True, config=old, data_sha256=sha256_file(data),
                    signature=dict(config_sha256=sha256_file(old_config),
                                   code_sha256={old_code.relative_to(root).as_posix(): sha256_file(old_code)}),
                    artifact_sha256={artifact.relative_to(root).as_posix(): sha256_file(artifact)})
    if mutation == "incomplete":
        manifest["complete"] = False
    write_json(directory / "run_manifest.json", manifest)
    targets = dict(code=old_code, artifact=artifact, data=data, configuration=old_config)
    if mutation in targets:
        targets[mutation].write_text("changed synthetic old fixture", encoding="utf-8")
    if mutation is None:
        hashes = experiment.verify_source(root, cfg, old)
        assert hashes[artifact.relative_to(root).as_posix()] == sha256_file(artifact)
        assert hashes[cfg["reference"]["e04_report"]] == sha256_file(report)
    else:
        with pytest.raises(ValueError):
            experiment.verify_source(root, cfg, old)


def test_tune_rejects_test_before_any_fit(tmp_path, monkeypatch):
    cfg = yaml.safe_load((PROJECT / "configs/offline_detection.yaml").read_text(encoding="utf-8"))
    monkeypatch.setattr(experiment, "scored", lambda *args: pytest.fail("Test must never enter tuning"))
    with pytest.raises(ValueError, match="validation only"):
        experiment.tune(generated("test"), cfg, tmp_path)


def test_tune_uses_both_fixed_control_budgets_and_saves_pretest_seal(tmp_path, monkeypatch):
    cfg = yaml.safe_load((PROJECT / "configs/offline_detection.yaml").read_text(encoding="utf-8"))
    calls = []
    def scored(data, method, params, _cfg):
        assert data[2]["split"].tolist() == ["validation"]
        calls.append((method, params["penalty"]))
        rows = []
        for scenario in ["level_up", "level_down", "no_change", "outlier"]:
            primary = scenario.startswith("level_")
            fp = 1 if primary else (3 if params["penalty"] < 2 else 0)
            row = {name: 0 for name in COUNTS}
            row.update(series_id=scenario, scenario=scenario, method=method,
                       tp=4 if primary else 0, fp=fp, fn=1 if primary else 0,
                       n_events=5 if primary else 0, n_monitoring_months=12,
                       n_breakpoints=(4 if primary else 0) + fp,
                       breakpoint_offsets_months=json.dumps([0] * (4 if primary else 0)))
            rows.append(row)
        return SimpleNamespace(breakpoints=pd.DataFrame(columns=BREAKPOINT_COLUMNS)), pd.DataFrame(rows), pd.DataFrame(columns=["series_id"]), None
    monkeypatch.setattr(experiment, "scored", scored)
    choices = experiment.tune(generated(), cfg, tmp_path)
    assert calls == [(method, pen) for method in experiment.METHODS for pen in experiment.PENALTIES]
    assert all(choice["parameters"]["penalty"] == 2.0 for choice in choices.values())
    table = pd.read_csv(tmp_path / "validation_selection.csv")
    assert table.loc[table["selected"], "max_control_far"].le(1.0).all()
    seal = json.loads((tmp_path / "selection_seal.json").read_text(encoding="utf-8"))
    assert seal["selection_sha256"] == sha256_file(tmp_path / "selected_parameters.json")
    assert seal["candidate_table_sha256"] == sha256_file(tmp_path / "validation_selection.csv")
    assert seal["selected_on"] == "synthetic_validation_only" and seal["test_generated"] is False


def test_no_feasible_method_does_not_attempt_a_test_fit(tmp_path, monkeypatch):
    cfg = yaml.safe_load((PROJECT / "configs/offline_detection.yaml").read_text(encoding="utf-8"))
    monkeypatch.setattr(experiment, "scored", lambda *args: pytest.fail("Unselected method must not fit"))
    selected = {m: dict(selected=False, reason="no_candidate_within_fixed_control_budget") for m in experiment.METHODS}
    with pytest.raises(RuntimeError, match="No offline method"):
        experiment.save_synthetic("test", generated("test"), selected, cfg, tmp_path)


def test_json_manifest_serialization_is_finite_and_handles_timestamps(tmp_path):
    target = tmp_path / "fixture.json"
    experiment.save_json(target, dict(array=np.array([1, 2]), scalar=np.int64(3),
                                     date=pd.Timestamp("2024-01-31"), missing=np.nan, infinite=np.inf))
    result = json.loads(target.read_text(encoding="utf-8"))
    assert result == dict(array=[1, 2], scalar=3, date="2024-01-31T00:00:00", missing=None, infinite=None)
