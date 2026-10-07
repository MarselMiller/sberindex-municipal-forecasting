"""Offline checks of E07c controlled truth and genuinely prefix-only features."""
import copy

import numpy as np
import pandas as pd
import pytest

from sberforecast.early_warning_synthetic import (
    COHORTS, DETECTOR_VALUES, EVENT_COLUMNS, METADATA_COLUMNS, OBSERVATION_COLUMNS,
    VALUES, build_cases, build_features, feature_dictionary, feature_groups, generate_cohorts,
)
from sberforecast.models import baseline_predict
from sberforecast.online_detection import BOCPD, CUSUM, EWMA


CONFIG = {
    "cohorts": {name: {"size": size, "seed": seed} for name, size, seed in
        (("train", 600, 420101), ("validation", 200, 420201), ("test", 300, 420301))},
    "generator": {
        "months": 24, "event_fraction": .6, "unanticipated_fraction": .25,
        "controls_false_precursor_fraction": .35, "onset_index_range": [16, 20],
        "precursor_lead_range": [1, 3], "level_range": [800., 1600.],
        "trend_monthly_fraction_range": [-.002, .003],
        "seasonality_amplitude_fraction_range": [.08, .2],
        "shift_magnitude_fraction_range": [.25, .5], "high_noise_fraction": .25,
        "low_noise_fraction_range": [.02, .05], "high_noise_fraction_range": [.08, .12],
        "strong_precursor_fraction": .5, "weak_precursor_strength_range": [.35, .8],
        "strong_precursor_strength_range": [.9, 1.7], "endogenous_channel_probability": .7,
        "external_channel_probability": .8, "endogenous_slope_fraction": .025,
        "precursor_volatility_multiplier": .7, "macro_ar_phi": .55,
        "macro_noise_sigma": .6, "macro_pulse_multiplier": 2.,
        "news_baseline_rate": 1.2, "news_pulse_multiplier": 4.,
    },
    "labels": {"min_history": 12, "horizons": [1, 3]},
    "features": {
        "baseline_model": "SeasonalNaiveYoY", "yearly_growth_window": 3,
        "yearly_growth_bounds": [.5, 2.], "rolling_window": 3, "detector_warmup": 3,
        "detector_relative_scale_floor": .03, "detector_absolute_scale_floor": 1.,
        "cusum": {"allowance": .5, "threshold": 5.},
        "ewma": {"alpha": .3, "threshold": 3.},
        "bocpd": {"hazard": 1/24, "threshold": .5, "recent_run_max": 2},
    },
}
SIZES = {"train": 20, "validation": 12, "test": 16}


def observations(uid="train_a", length=24):
    month = np.arange(length)
    return pd.DataFrame(dict(series_id=uid, cohort="train", month_index=month,
        y=100.+3*month+10*np.sin(2*np.pi*month/12),
        macro_pressure=.25*month+np.sin(month), news_intensity=(month % 5).astype(float)))


def events(uid="train_a", onset=16):
    return pd.DataFrame([dict(series_id=uid, cohort="train", event_id=uid+"_event", onset_index=onset)])


def empty_events():
    return pd.DataFrame(columns=EVENT_COLUMNS)


def row(table, origin, k=None):
    selected = table.month_index.eq(origin)
    if k is not None:
        selected &= table.k.eq(k)
    return table.loc[selected].iloc[0]


def test_generator_deterministic_disjoint_exact_fraction_counts_and_plain_observations():
    first = generate_cohorts(CONFIG, SIZES)
    second = generate_cohorts(CONFIG, SIZES)
    for name in first:
        pd.testing.assert_frame_equal(first[name], second[name])
    assert tuple(first["observations"]) == OBSERVATION_COLUMNS
    assert len(first["observations"]) == sum(SIZES.values())*24
    assert not first["observations"].duplicated(["series_id", "month_index"]).any()
    assert not first["cohort_ids"].series_seed.duplicated().any()
    assert not first["cohort_ids"].series_id.duplicated().any()
    for cohort, size in SIZES.items():
        meta = first["metadata"].loc[lambda data: data.cohort.eq(cohort)]
        event_count = int(np.floor(size*.6+.5))
        unexpected_count = int(np.floor(event_count*.25+.5))
        false_count = int(np.floor((size-event_count)*.35+.5))
        assert len(meta) == size and meta.has_event.sum() == event_count
        assert (meta.has_event & ~meta.is_anticipated).sum() == unexpected_count
        assert meta.false_precursor.sum() == false_count
        assert not (meta.has_event & meta.false_precursor).any()
        assert meta.series_seed.between(CONFIG["cohorts"][cohort]["seed"]*100000,
            CONFIG["cohorts"][cohort]["seed"]*100000+size-1).all()
    event_table = first["events"]
    assert event_table.onset_index.between(16, 20).all()
    assert event_table.shift_magnitude_fraction.between(.25, .5).all()
    assert set(event_table.direction).issubset({"positive", "negative"})


def test_unanticipated_events_have_identical_pre_onset_baseline_and_external_background():
    generated = generate_cohorts(CONFIG, SIZES)
    baseline_config = copy.deepcopy(CONFIG)
    baseline_config["generator"]["event_fraction"] = 0.
    baseline_config["generator"]["controls_false_precursor_fraction"] = 0.
    baseline = generate_cohorts(baseline_config, SIZES)["observations"]
    unexpected = generated["events"].loc[lambda data: ~data.is_anticipated]
    assert len(unexpected) > 0
    for event in unexpected.itertuples():
        selected = generated["observations"].series_id.eq(event.series_id)
        original = generated["observations"].loc[selected & generated["observations"].month_index.lt(event.onset_index)]
        reference = baseline.loc[baseline.series_id.eq(event.series_id) & baseline.month_index.lt(event.onset_index)]
        pd.testing.assert_frame_equal(original.reset_index(drop=True), reference.reset_index(drop=True))
        meta = generated["metadata"].loc[lambda data: data.series_id.eq(event.series_id)].iloc[0]
        assert meta.precursor_strength == 0 and meta.precursor_class == "none"
        assert not meta.endogenous_channel and not meta.external_channel


def test_event_shift_persists_to_end_without_post_onset_cue_extension():
    generated = generate_cohorts(CONFIG, SIZES)
    no_event_config = copy.deepcopy(CONFIG)
    no_event_config["generator"].update(event_fraction=0., controls_false_precursor_fraction=0.)
    reference = generate_cohorts(no_event_config, SIZES)["observations"]
    for event in generated["events"].itertuples():
        meta = generated["metadata"].loc[lambda data: data.series_id.eq(event.series_id)].iloc[0]
        actual = generated["observations"].loc[lambda data: data.series_id.eq(event.series_id) & data.month_index.ge(event.onset_index)]
        original = reference.loc[lambda data: data.series_id.eq(event.series_id) & data.month_index.ge(event.onset_index)]
        shift = meta.shift_direction*meta.base_level*meta.shift_magnitude_fraction
        np.testing.assert_allclose(actual.y, np.maximum(original.y.to_numpy()+shift, .01))
        np.testing.assert_array_equal(actual.macro_pressure, original.macro_pressure)
        np.testing.assert_array_equal(actual.news_intensity, original.news_intensity)


def test_false_control_episodes_use_nonempty_same_configured_channels_without_an_event():
    generated = generate_cohorts(CONFIG, SIZES)
    meta = generated["metadata"]
    cues = meta.loc[meta.is_anticipated | meta.false_precursor]
    assert len(meta.loc[meta.false_precursor]) > 0
    assert (cues.endogenous_channel | cues.external_channel).all()
    assert cues.precursor_lead_months.between(1, 3).all()
    assert (cues.precursor_end_index-cues.precursor_start_index+1).eq(cues.precursor_lead_months).all()
    for data in cues.itertuples():
        lo, hi = CONFIG["generator"][data.precursor_class+"_precursor_strength_range"]
        assert lo <= data.precursor_strength <= hi
    control_ids = set(meta.loc[meta.false_precursor, "series_id"])
    assert not control_ids.intersection(generated["events"].series_id)
    labels = build_cases(generated["observations"], generated["events"], CONFIG)
    controls = labels.loc[labels.series_id.isin(control_ids) & labels.fully_known]
    assert controls.label.eq(0).all() and controls.at_risk.all()


def test_future_test_seed_change_leaves_train_and_validation_generation_identical():
    changed = copy.deepcopy(CONFIG)
    changed["cohorts"]["test"]["seed"] += 1
    first, second = generate_cohorts(CONFIG, SIZES), generate_cohorts(changed, SIZES)
    for name in first:
        pd.testing.assert_frame_equal(first[name].loc[lambda data: data.cohort.ne("test")].reset_index(drop=True),
            second[name].loc[lambda data: data.cohort.ne("test")].reset_index(drop=True))


@pytest.mark.parametrize("cutoff", [11, 12, 14, 15, 18, 21])
def test_prefix_recalculation_identical_to_full_stream_without_future_calibration(cutoff):
    obs = observations()
    full = build_features(obs, CONFIG)
    prefix = build_features(obs.loc[obs.month_index.le(cutoff)], CONFIG)
    pd.testing.assert_frame_equal(prefix, full.loc[full.month_index.le(cutoff)].reset_index(drop=True))
    assert prefix.max_dependency_month_index.le(prefix.month_index).all()


@pytest.mark.parametrize("column", ["y", "macro_pressure", "news_intensity"])
def test_future_observation_replacement_cannot_change_any_past_feature(column):
    source = observations()
    changed = source.copy()
    changed.loc[changed.month_index.gt(15), column] += 100000.
    first, second = build_features(source, CONFIG), build_features(changed, CONFIG)
    pd.testing.assert_frame_equal(first.loc[first.month_index.le(15)].reset_index(drop=True),
        second.loc[second.month_index.le(15)].reset_index(drop=True))
    feature = {"y": "history_y", "macro_pressure": "external_macro_pressure", "news_intensity": "external_news_intensity"}[column]
    assert not first.loc[first.month_index.gt(15), feature].equals(second.loc[second.month_index.gt(15), feature])


def test_other_series_future_and_extreme_values_do_not_affect_local_feature_stream():
    source = observations()
    unrelated = observations("validation_b").assign(cohort="validation", y=lambda data: data.y*1000)
    alone, together = build_features(source, CONFIG), build_features(pd.concat([source, unrelated]), CONFIG)
    pd.testing.assert_frame_equal(alone, together.loc[together.series_id.eq("train_a")].reset_index(drop=True))


def test_h1_residual_exactly_reuses_frozen_seasonal_baseline_on_past_only():
    obs = observations()
    features = build_features(obs, CONFIG)
    assert pd.isna(row(features, 11).history_residual)
    assert row(features, 11).history_residual_missing
    for target in range(12, 24):
        past = pd.DataFrame({"train_a": obs.y.iloc[:target].to_numpy()},
            index=pd.date_range("2023-01-01", periods=target, freq="MS"))
        prediction, _ = baseline_predict(past, 1, "SeasonalNaiveYoY", CONFIG["features"])
        expected = obs.y.iloc[target]-prediction[0, 0]
        assert row(features, target).history_residual == pytest.approx(expected)


def test_warmup_uses_first_three_residuals_then_existing_online_classes_without_resets():
    obs = observations()
    features = build_features(obs, CONFIG)
    assert features.loc[features.month_index.lt(14), "detector_ready"].eq(0).all()
    assert features.loc[features.month_index.ge(14), "detector_ready"].eq(1).all()
    scores = [name for name in DETECTOR_VALUES if name.endswith("_score")]
    assert features.loc[features.month_index.le(14), scores].isna().all().all()
    assert features.loc[features.month_index.ge(15), scores].notna().all().all()
    errors, forecasts = [], []
    for target in range(12, 24):
        past = pd.DataFrame({"train_a": obs.y.iloc[:target].to_numpy()},
            index=pd.date_range("2023-01-01", periods=target, freq="MS"))
        prediction, _ = baseline_predict(past, 1, "SeasonalNaiveYoY", CONFIG["features"])
        forecasts.append(prediction[0, 0])
        errors.append(obs.y.iloc[target]-prediction[0, 0])
    center = np.median(errors[:3])
    scale = max(1.4826*np.median(np.abs(np.asarray(errors[:3])-center)), .03*np.median(np.abs(forecasts[:3])), 1.)
    detectors = {"cusum": CUSUM(**CONFIG["features"]["cusum"]), "ewma": EWMA(**CONFIG["features"]["ewma"]),
        "bocpd": BOCPD(**CONFIG["features"]["bocpd"])}
    for target, error in zip(range(15, 24), errors[3:]):
        for name, detector in detectors.items():
            result = detector.update((error-center)/scale)
            assert row(features, target)["detector_"+name+"_score"] == pytest.approx(result.score)
            assert row(features, target)["detector_"+name+"_crossed"] == float(result.crossed_threshold)
    assert row(features, 15).detector_bocpd_score == 0  # recent-run score not yet armed


def test_missing_past_defers_warmup_and_strict_windows_without_filling():
    obs = observations()
    obs.loc[obs.month_index.eq(12), "y"] = np.nan
    obs.loc[obs.month_index.eq(16), "macro_pressure"] = np.nan
    features = build_features(obs, CONFIG)
    assert row(features, 14).detector_ready == 0 and row(features, 15).detector_ready == 1
    assert pd.isna(row(features, 15).detector_cusum_score)
    assert np.isfinite(row(features, 16).detector_cusum_score)
    assert row(features, 12).history_y_missing and row(features, 12).history_mean_3m_missing
    assert row(features, 14).history_mean_3m_missing
    assert row(features, 16).external_macro_pressure_missing
    assert row(features, 18).external_macro_mean_3m_missing
    assert not row(features, 19).external_macro_mean_3m_missing


def test_strict_calendar_population_volatility_changes_ratios_and_flags():
    obs = observations()
    obs.loc[obs.month_index.eq(12), "y"] = 0.
    features = build_features(obs, CONFIG)
    current = row(features, 13)
    recent = obs.y.iloc[11:14].to_numpy()
    assert current.history_ratio_1m_missing and pd.isna(current.history_ratio_1m)
    assert current.history_change_1m == obs.y.iloc[13]-obs.y.iloc[12]
    assert current.history_mean_3m == pytest.approx(np.mean(recent))
    assert current.history_std_3m == pytest.approx(np.std(recent, ddof=0))
    assert current.history_cv_3m == pytest.approx(np.std(recent, ddof=0)/np.mean(recent))
    assert current.history_slope_3m == pytest.approx((recent[-1]-recent[0])/2)
    assert current.history_yoy_ratio == pytest.approx(obs.y.iloc[13]/obs.y.iloc[1])
    assert all(bool(current[name+"_missing"]) == pd.isna(current[name]) for name in VALUES)


def test_labels_true_future_window_availability_active_exclusion_and_right_censoring():
    cases = build_cases(observations(), events(), CONFIG)
    assert len(cases) == 13*2 and not cases.duplicated(["series_id", "month_index", "k"]).any()
    assert row(cases, 12, 3).label == 0
    assert row(cases, 13, 3).label == 1 and row(cases, 13, 3).label_known_month_index == 16
    assert row(cases, 15, 1).label == 1 and row(cases, 15, 1).label_known_month_index == 16
    assert row(cases, 15, 1).positive_event_id == "train_a_event"
    assert row(cases, 16, 1).active_regime and not row(cases, 16, 1).at_risk
    assert row(cases, 16, 1).fully_known and row(cases, 16, 1).label == 0
    assert row(cases, 20, 3).fully_known and row(cases, 20, 3).label_known_month_index == 23
    for origin, k in ((21, 3), (23, 1), (23, 3)):
        censored = row(cases, origin, k)
        assert censored.right_censored and not censored.fully_known and pd.isna(censored.label)
        assert pd.isna(censored.label_known_month_index)


def test_future_missingness_cannot_turn_unknown_label_into_negative_or_change_origin_eligibility():
    source = observations()
    first = build_cases(source, events(), CONFIG)
    source.loc[source.month_index.eq(16), "y"] = np.nan
    changed = build_cases(source, events(), CONFIG)
    before, after = row(first, 15, 1), row(changed, 15, 1)
    assert before.label == 1 and pd.isna(after.label) and not after.fully_known
    assert before.eligible == after.eligible and before.at_risk == after.at_risk
    assert not after.right_censored


def test_short_or_stale_history_retains_all_calendar_cases_with_ineligible_flag():
    source = observations()
    source.loc[source.month_index.le(2), "y"] = np.nan
    source.loc[source.month_index.between(20, 22), "y"] = np.nan
    cases = build_cases(source, empty_events(), CONFIG)
    assert len(cases) == 26
    assert not row(cases, 11, 1).eligible and not row(cases, 11, 1).at_risk
    assert row(cases, 14, 1).eligible
    assert not row(cases, 22, 1).eligible
    assert row(cases, 23, 1).eligible and not row(cases, 23, 1).fully_known


def test_fixed_dictionary_numeric_groups_exclude_calendar_identity_and_generator_truth():
    dictionary = feature_dictionary()
    groups = feature_groups()
    assert {name: len(columns) for name, columns in groups.items()} == {"history": 32, "detector": 22, "external": 16}
    columns = tuple(column for group in groups.values() for column in group)
    assert len(columns) == len(set(columns)) == len(dictionary) == 70
    assert set(columns) == set(dictionary.feature)
    assert dictionary.model_candidate.all() and dictionary.source.eq("synthetic_observations_only").all()
    assert not any(token in name for name in columns for token in ("month_index", "onset", "event", "precursor", "seed", "cohort", "future"))
    built = build_features(observations(), CONFIG)
    assert len(built) == 13 and set(columns).issubset(built)
    assert all(pd.api.types.is_numeric_dtype(built[column]) or pd.api.types.is_bool_dtype(built[column]) for column in columns)
    assert not np.isinf(built[list(VALUES)].to_numpy(dtype=float)).any()


@pytest.mark.parametrize("oracle", ["onset_index", "event_id", "series_seed", "precursor_class", "noise_class", "shift_magnitude_fraction", "active_regime", "label", "offline_breakpoint", "pelt_segment", "future_regime"])
def test_oracle_metadata_and_offline_segmentation_fields_rejected(oracle):
    source = observations().assign(**{oracle: 0})
    with pytest.raises(ValueError, match="Oracle/offline"):
        build_features(source, CONFIG)


@pytest.mark.parametrize("problem", ["unknown_id", "wrong_cohort", "duplicate_event", "duplicate_series", "bad_onset", "missing_id"])
def test_event_registry_identity_and_calendar_are_validated(problem):
    registry = events()
    if problem == "unknown_id":
        registry.loc[0, "series_id"] = "absent"
    elif problem == "wrong_cohort":
        registry.loc[0, "cohort"] = "test"
    elif problem == "duplicate_event":
        registry = pd.concat([registry, registry.assign(series_id="train_b")])
    elif problem == "duplicate_series":
        registry = pd.concat([registry, registry.assign(event_id="other_event")])
    elif problem == "bad_onset":
        registry.loc[0, "onset_index"] = 24
    else:
        registry.loc[0, "event_id"] = None
    with pytest.raises(ValueError):
        build_cases(observations(), registry, CONFIG)


@pytest.mark.parametrize("problem", ["duplicate_month", "cross_cohort", "negative_y", "infinite_macro", "bad_calendar"])
def test_observation_identity_calendar_and_numeric_values_validated(problem):
    source = observations()
    if problem == "duplicate_month":
        source = pd.concat([source, source.iloc[:1]])
    elif problem == "cross_cohort":
        source.loc[0, "cohort"] = "validation"
    elif problem == "negative_y":
        source.loc[0, "y"] = -1
    elif problem == "infinite_macro":
        source.loc[0, "macro_pressure"] = np.inf
    else:
        source.loc[0, "month_index"] = 24
    with pytest.raises(ValueError):
        build_features(source, CONFIG)


@pytest.mark.parametrize("problem", ["shared_seed", "bad_sizes", "negative_probability", "future_warmup", "negative_scale"])
def test_invalid_fixed_design_configuration_rejected(problem):
    config = copy.deepcopy(CONFIG)
    sizes = dict(SIZES)
    if problem == "shared_seed":
        config["cohorts"]["test"]["seed"] = config["cohorts"]["train"]["seed"]
    elif problem == "bad_sizes":
        sizes["test"] = -1
    elif problem == "negative_probability":
        config["generator"]["unanticipated_fraction"] = -.1
    elif problem == "future_warmup":
        config["features"]["detector_warmup"] = 4
    else:
        config["features"]["detector_absolute_scale_floor"] = -1
    with pytest.raises(ValueError):
        generate_cohorts(config, sizes)


def test_empty_cohorts_keep_all_schemas_without_calibration_or_labels():
    generated = generate_cohorts(CONFIG, {name: 0 for name in COHORTS})
    assert tuple(generated["metadata"]) == METADATA_COLUMNS
    assert tuple(generated["observations"]) == OBSERVATION_COLUMNS
    assert tuple(generated["events"]) == EVENT_COLUMNS
    assert all(data.empty for data in generated.values())
    assert build_features(generated["observations"], CONFIG).empty
    assert build_cases(generated["observations"], generated["events"], CONFIG).empty
