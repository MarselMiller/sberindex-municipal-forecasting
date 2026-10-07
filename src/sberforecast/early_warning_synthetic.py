"""E07c controlled synthetic truth, causal observations and prefix features.

Synthetic cues are noisy observations, also present in some event-free controls.
True onsets, sampled parameters and precursor classes live outside observations
and never enter feature construction. This benchmark is not real shock truth.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from .models import baseline_predict
from .online_detection import BOCPD, CUSUM, EWMA


COHORTS = ("train", "validation", "test")
OBSERVATION_COLUMNS = ("series_id", "month_index", "cohort", "y", "macro_pressure", "news_intensity")
KEYS = ("series_id", "month_index", "cohort")
EVENT_COLUMNS = ("series_id", "cohort", "event_id", "onset_index", "is_anticipated", "precursor_class", "noise_class", "direction", "shift_magnitude_fraction")
METADATA_COLUMNS = (
    "series_id", "cohort", "has_event", "is_anticipated", "false_precursor", "precursor_class", "noise_class",
    "series_seed", "cohort_seed", "base_level", "trend_monthly_fraction", "seasonality_amplitude_fraction",
    "seasonal_phase", "noise_fraction", "shift_magnitude_fraction", "shift_direction", "precursor_strength",
    "precursor_lead_months", "precursor_start_index", "precursor_end_index", "endogenous_channel", "external_channel", "origin",
)
HISTORY_VALUES = (
    "history_y", "history_previous_y", "history_change_1m", "history_ratio_1m", "history_yoy_ratio",
    "history_mean_3m", "history_std_3m", "history_cv_3m", "history_slope_3m",
    "history_residual", "history_previous_residual", "history_residual_change_1m",
    "history_residual_mean_3m", "history_residual_std_3m", "history_residual_slope_3m", "history_relative_residual",
)
DETECTOR_VALUES = (
    "detector_cusum_score", "detector_cusum_positive", "detector_cusum_negative", "detector_cusum_crossed",
    "detector_ewma_score", "detector_ewma_value", "detector_ewma_crossed",
    "detector_bocpd_score", "detector_bocpd_recent_probability", "detector_bocpd_crossed", "detector_ready",
)
EXTERNAL_VALUES = (
    "external_macro_pressure", "external_macro_change_1m", "external_macro_mean_3m", "external_macro_std_3m",
    "external_news_intensity", "external_news_change_1m", "external_news_mean_3m", "external_news_std_3m",
)
VALUES = HISTORY_VALUES + DETECTOR_VALUES + EXTERNAL_VALUES


def feature_groups() -> dict[str, tuple[str, ...]]:
    return {name: columns + tuple(column+"_missing" for column in columns)
            for name, columns in (("history", HISTORY_VALUES), ("detector", DETECTOR_VALUES), ("external", EXTERNAL_VALUES))}


def _integer(value, name, minimum=0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _settings(config: Mapping) -> tuple[dict, dict, dict]:
    generator, features, labels = (dict(config[name]) for name in ("generator", "features", "labels"))
    months = _integer(generator["months"], "months", 1)
    minimum = _integer(labels["min_history"], "min_history", 1)
    warmup = _integer(features["detector_warmup"], "detector_warmup", 1)
    if months != 24 or minimum != 12 or warmup != 3 or labels["horizons"] != [1, 3] or features["rolling_window"] != 3:
        raise ValueError("E07c fixed design requires 24 months, history12, warmup3, rolling3 and k1/3")
    if features["baseline_model"] != "SeasonalNaiveYoY":
        raise ValueError("Only the causal SeasonalNaiveYoY h1 baseline is supported")
    for name in ("event_fraction", "unanticipated_fraction", "controls_false_precursor_fraction", "high_noise_fraction", "strong_precursor_fraction", "endogenous_channel_probability", "external_channel_probability"):
        if not np.isfinite(generator[name]) or not 0 <= generator[name] <= 1:
            raise ValueError("Invalid probability: " + name)
    onset_low, onset_high = (_integer(value, "onset bound") for value in generator["onset_index_range"])
    lead_low, lead_high = (_integer(value, "precursor lead bound", 1) for value in generator["precursor_lead_range"])
    if not minimum <= onset_low <= onset_high < months or not 1 <= lead_low <= lead_high <= 3 or onset_low-lead_high < minimum-1:
        raise ValueError("Onset/precursor ranges leave insufficient causal warning history")
    if not np.isfinite(generator["macro_ar_phi"]) or not abs(generator["macro_ar_phi"]) < 1:
        raise ValueError("Invalid stochastic external process parameters")
    for name in ("macro_noise_sigma", "news_baseline_rate", "endogenous_slope_fraction", "precursor_volatility_multiplier", "macro_pulse_multiplier", "news_pulse_multiplier"):
        if not np.isfinite(generator[name]) or generator[name] < 0 or (name == "macro_noise_sigma" and generator[name] == 0):
            raise ValueError("Invalid stochastic process parameter: " + name)
    for name in ("level_range", "trend_monthly_fraction_range", "seasonality_amplitude_fraction_range", "shift_magnitude_fraction_range", "low_noise_fraction_range", "high_noise_fraction_range", "weak_precursor_strength_range", "strong_precursor_strength_range"):
        bounds = np.asarray(generator[name], dtype=float)
        if bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] > bounds[1]:
            raise ValueError("Invalid generator range: " + name)
        if name != "trend_monthly_fraction_range" and bounds[0] < 0:
            raise ValueError("Generator range must be nonnegative: " + name)
        if name in ("level_range", "weak_precursor_strength_range", "strong_precursor_strength_range") and bounds[0] == 0:
            raise ValueError("Generator range must be positive: " + name)
    for name in ("detector_relative_scale_floor", "detector_absolute_scale_floor"):
        if not np.isfinite(features[name]) or features[name] < 0 or (name == "detector_absolute_scale_floor" and features[name] == 0):
            raise ValueError("Invalid detector scale floor: " + name)
    return generator, features, labels


def _round_count(total: int, fraction: float) -> int:
    """Nearest count with halves rounded upwards; independent of outcomes."""
    return int(np.floor(total*fraction+0.5))


def generate_cohorts(config: Mapping, sizes: Mapping[str, int] | None = None) -> dict[str, pd.DataFrame]:
    """Generate disjoint cohorts with fixed fractions, before any model metrics.

    Independent RNG substreams separate baseline parameters/noise, external
    background, interventions and onset. An unanticipated shift injects no
    pre-onset cue. False control episodes use the same cue distribution.
    """
    g, _, _ = _settings(config)
    if sizes is not None and set(sizes) != set(COHORTS):
        raise ValueError("sizes must contain train, validation and test")
    seeds = [_integer(config["cohorts"][name]["seed"], name+" seed") for name in COHORTS]
    if len(set(seeds)) != 3:
        raise ValueError("Cohort seeds must be disjoint")
    observation_rows, metadata_rows, event_rows, id_rows = [], [], [], []
    n_months = g["months"]
    month = np.arange(n_months)
    for cohort, cohort_seed in zip(COHORTS, seeds):
        size = _integer((sizes or {}).get(cohort, config["cohorts"][cohort]["size"]), cohort+" size")
        if size >= 100000:
            raise ValueError("Cohort size exceeds disjoint series-seed allocation")
        allocation = np.random.default_rng(cohort_seed).permutation(size)
        count_events = _round_count(size, g["event_fraction"])
        event_set = set(allocation[:count_events])
        count_unanticipated = _round_count(count_events, g["unanticipated_fraction"])
        unanticipated_set = set(allocation[:count_unanticipated])
        control_ids = allocation[count_events:]
        false_set = set(control_ids[:_round_count(len(control_ids), g["controls_false_precursor_fraction"])])
        for number in range(size):
            uid = f"{cohort}_{number:06d}"
            seed = cohort_seed*100000+number
            rngs = [np.random.default_rng(np.random.SeedSequence([seed, stream])) for stream in range(6)]
            params, noise_rng, macro_rng, news_rng, cue_rng, onset_rng = rngs
            level = float(params.uniform(*g["level_range"]))
            trend = float(params.uniform(*g["trend_monthly_fraction_range"]))
            amplitude = float(params.uniform(*g["seasonality_amplitude_fraction_range"]))
            phase = float(params.uniform(0, 2*np.pi))
            high_noise = bool(params.random() < g["high_noise_fraction"])
            noise_class = "high" if high_noise else "low"
            noise_fraction = float(params.uniform(*g["high_noise_fraction_range" if high_noise else "low_noise_fraction_range"]))
            background_noise = noise_rng.normal(0, level*noise_fraction, n_months)
            baseline = level*(1+trend*month+amplitude*np.sin(2*np.pi*month/12+phase))
            macro = np.empty(n_months)
            innovations = macro_rng.normal(0, g["macro_noise_sigma"], n_months)
            for index in range(n_months):
                macro[index] = g["macro_ar_phi"]*(macro[index-1] if index else 0)+innovations[index]
            news_count = news_rng.poisson(g["news_baseline_rate"], n_months)
            anchor = int(onset_rng.integers(g["onset_index_range"][0], g["onset_index_range"][1]+1))
            direction = int(onset_rng.choice([-1, 1]))
            magnitude = float(onset_rng.uniform(*g["shift_magnitude_fraction_range"]))
            has_event = number in event_set
            anticipated = has_event and number not in unanticipated_set
            false_precursor = number in false_set
            cue_present = anticipated or false_precursor
            lead = int(cue_rng.integers(g["precursor_lead_range"][0], g["precursor_lead_range"][1]+1))
            strong = bool(cue_rng.random() < g["strong_precursor_fraction"])
            strength = float(cue_rng.uniform(*g["strong_precursor_strength_range" if strong else "weak_precursor_strength_range"]))
            endogenous = bool(cue_rng.random() < g["endogenous_channel_probability"])
            external = bool(cue_rng.random() < g["external_channel_probability"])
            if not endogenous and not external:
                endogenous = bool(cue_rng.integers(0, 2))
                external = not endogenous
            y = baseline+background_noise
            if cue_present:
                window = np.arange(anchor-lead, anchor)
                if endogenous:
                    # Stochastic strength, direction and background variance;
                    # the same transient pattern also occurs without a shock.
                    y[window] += direction*level*g["endogenous_slope_fraction"]*strength*np.arange(1, lead+1)
                    y[window] += background_noise[window]*g["precursor_volatility_multiplier"]*strength
                if external:
                    pulse = cue_rng.lognormal(-0.125, 0.5, lead)
                    macro[window] += g["macro_pulse_multiplier"]*strength*pulse
                    news_count[window] += cue_rng.poisson(g["news_pulse_multiplier"]*strength*pulse)
            if has_event:
                y[anchor:] += direction*level*magnitude
            y = np.maximum(y, 0.01)
            precursor_class = ("strong" if strong else "weak") if cue_present else "none"
            for index in month:
                observation_rows.append(dict(series_id=uid, month_index=int(index), cohort=cohort,
                    y=float(y[index]), macro_pressure=float(macro[index]), news_intensity=int(news_count[index])))
            metadata_rows.append(dict(series_id=uid, cohort=cohort, has_event=has_event, is_anticipated=anticipated,
                false_precursor=false_precursor, precursor_class=precursor_class, noise_class=noise_class,
                series_seed=seed, cohort_seed=cohort_seed, base_level=level, trend_monthly_fraction=trend,
                seasonality_amplitude_fraction=amplitude, seasonal_phase=phase, noise_fraction=noise_fraction,
                shift_magnitude_fraction=magnitude if has_event else 0., shift_direction=direction if has_event else 0,
                precursor_strength=strength if cue_present else 0., precursor_lead_months=lead if cue_present else 0,
                precursor_start_index=anchor-lead if cue_present else np.nan,
                precursor_end_index=anchor-1 if cue_present else np.nan,
                endogenous_channel=endogenous if cue_present else False, external_channel=external if cue_present else False,
                origin="controlled_synthetic_generator_not_real_economic_data"))
            id_rows.append(dict(series_id=uid, cohort=cohort, series_seed=seed, cohort_seed=cohort_seed))
            if has_event:
                event_rows.append(dict(series_id=uid, cohort=cohort, event_id=uid+"_level_shift", onset_index=anchor,
                    is_anticipated=anticipated, precursor_class=precursor_class, noise_class=noise_class,
                    direction="positive" if direction > 0 else "negative", shift_magnitude_fraction=magnitude))
    metadata = pd.DataFrame(metadata_rows, columns=METADATA_COLUMNS)
    ids = pd.DataFrame(id_rows, columns=("series_id", "cohort", "series_seed", "cohort_seed"))
    if ids.series_seed.duplicated().any():
        raise ValueError("Series seeds overlap between cohorts")
    return dict(observations=pd.DataFrame(observation_rows, columns=OBSERVATION_COLUMNS), metadata=metadata,
                events=pd.DataFrame(event_rows, columns=EVENT_COLUMNS), cohort_ids=ids)


def _observations(observations: pd.DataFrame, config: Mapping) -> pd.DataFrame:
    missing = set(OBSERVATION_COLUMNS)-set(observations)
    if missing:
        raise ValueError("Missing observation columns: " + ",".join(sorted(missing)))
    oracle_fields = set(METADATA_COLUMNS) | set(EVENT_COLUMNS) | {
        "label", "fully_known", "active_regime", "eligible", "at_risk", "label_known_month_index", "right_censored", "synthetic_truth",
    }
    forbidden = [name for name in observations if name not in OBSERVATION_COLUMNS and (name in oracle_fields or
        any(token in name.lower() for token in ("onset", "event", "precursor", "seed", "future", "breakpoint", "offline", "pelt", "binseg")))]
    if forbidden:
        raise ValueError("Oracle/offline metadata cannot enter feature observations: " + ",".join(forbidden))
    data = observations[list(OBSERVATION_COLUMNS)].copy()
    if data.series_id.isna().any() or data.series_id.astype(str).str.strip().eq("").any() or not data.cohort.isin(COHORTS).all():
        raise ValueError("Invalid series/cohort identity")
    data["series_id"] = data.series_id.astype(str)
    month = pd.to_numeric(data.month_index, errors="raise")
    if month.isna().any() or not np.isfinite(month).all() or not month.eq(np.floor(month)).all() or not month.between(0, config["generator"]["months"]-1).all():
        raise ValueError("month_index must be an integer within the declared calendar")
    data["month_index"] = month.astype(int)
    if data.duplicated(["series_id", "month_index"]).any() or data.groupby("series_id").cohort.nunique().gt(1).any():
        raise ValueError("Duplicate series/month or overlapping cohort identity")
    for name in ("y", "macro_pressure", "news_intensity"):
        data[name] = pd.to_numeric(data[name], errors="raise")
        if np.isinf(data[name]).any() or (name != "macro_pressure" and data[name].dropna().lt(0).any()):
            raise ValueError("Observation must be finite or missing and y/news nonnegative")
    return data.sort_values(["cohort", "series_id", "month_index"], kind="stable").reset_index(drop=True)


def _forecasts(data: pd.DataFrame, features: dict, minimum: int) -> dict[str, np.ndarray]:
    predictions = {}
    for _, cohort in data.groupby("cohort", sort=False):
        panel = cohort.pivot(index="month_index", columns="series_id", values="y")
        panel = panel.reindex(range(int(panel.index.max())+1))
        panel.index = pd.date_range("2023-01-01", periods=len(panel), freq="MS")
        forecast = np.full(panel.shape, np.nan)
        for target in range(minimum, len(panel)):
            prefix = panel.iloc[:target]
            finite = prefix.notna().to_numpy()
            count = finite.sum(axis=0)
            last = np.where(finite, np.arange(target)[:, None], -1).max(axis=0)
            valid = (count >= minimum) & (target-1-last <= 1)
            if valid.any():
                values, _ = baseline_predict(prefix.loc[:, valid], 1, "SeasonalNaiveYoY", features)
                forecast[target, valid] = values[0]
        predictions.update({str(uid): forecast[:, position] for position, uid in enumerate(panel.columns)})
    return predictions


def _strict(window: np.ndarray, operation) -> float:
    return float(operation(window)) if len(window) == 3 and np.isfinite(window).all() else np.nan


def build_features(observations: pd.DataFrame, config: Mapping) -> pd.DataFrame:
    """Single forward stream per series; no event truth or full-sample transform.

    The old h1 seasonal baseline is issued from y[:T], then y[T] supplies the
    observed residual. First three finite errors calibrate fixed median/MAD;
    monitoring begins on the next finite error. Threshold crossings are causal
    diagnostic state, without alarm-driven resets or future calibration.
    """
    _, f, labels = _settings(config)
    data = _observations(observations, config)
    forecasts = _forecasts(data, f, labels["min_history"]) if len(data) else {}
    records = []
    for uid, source in data.groupby("series_id", sort=True):
        maximum = int(source.month_index.max())
        part = source.set_index("month_index").reindex(range(maximum+1))
        y, macro, news_count = (part[name].to_numpy(dtype=float) for name in ("y", "macro_pressure", "news_intensity"))
        forecast = forecasts[uid][:maximum+1]
        residual = y-forecast
        detectors = dict(cusum=CUSUM(**f["cusum"]), ewma=EWMA(**f["ewma"]), bocpd=BOCPD(**f["bocpd"]))
        calibration, calibration_predictions = [], []
        center = scale = np.nan
        for origin in range(maximum+1):
            values = dict.fromkeys(VALUES, np.nan)
            ready_before = len(calibration) >= f["detector_warmup"]
            if np.isfinite(residual[origin]):
                if not ready_before:
                    calibration.append(float(residual[origin]))
                    calibration_predictions.append(abs(float(forecast[origin])))
                    if len(calibration) == f["detector_warmup"]:
                        center = float(np.median(calibration))
                        mad = float(1.4826*np.median(np.abs(np.asarray(calibration)-center)))
                        scale = max(mad, f["detector_relative_scale_floor"]*float(np.median(calibration_predictions)), f["detector_absolute_scale_floor"])
                else:
                    z = (residual[origin]-center)/scale
                    for name, detector in detectors.items():
                        update = detector.update(z)
                        values["detector_"+name+"_score"] = update.score
                        values["detector_"+name+"_crossed"] = float(update.crossed_threshold)
                        if name == "cusum":
                            values["detector_cusum_positive"] = detector.positive
                            values["detector_cusum_negative"] = detector.negative
                        elif name == "ewma":
                            values["detector_ewma_value"] = detector.value
                        else:
                            values["detector_bocpd_recent_probability"] = update.diagnostics["posterior_recent_probability"]
            values["detector_ready"] = float(len(calibration) >= f["detector_warmup"])
            if origin < labels["min_history"]-1:
                continue
            recent, errors = y[max(0, origin-2):origin+1], residual[max(0, origin-2):origin+1]
            current, previous = y[origin], y[origin-1] if origin else np.nan
            error, previous_error = residual[origin], residual[origin-1] if origin else np.nan
            mean, deviation = _strict(recent, np.mean), _strict(recent, lambda x: np.std(x, ddof=0))
            values.update(history_y=current, history_previous_y=previous, history_change_1m=current-previous,
                history_ratio_1m=current/previous if np.isfinite(current) and np.isfinite(previous) and previous > 0 else np.nan,
                history_yoy_ratio=current/y[origin-12] if origin >= 12 and np.isfinite(current) and np.isfinite(y[origin-12]) and y[origin-12] > 0 else np.nan,
                history_mean_3m=mean, history_std_3m=deviation, history_cv_3m=deviation/mean if np.isfinite(mean) and mean > 0 else np.nan,
                history_slope_3m=_strict(recent, lambda x: (x[-1]-x[0])/2),
                history_residual=error, history_previous_residual=previous_error, history_residual_change_1m=error-previous_error,
                history_residual_mean_3m=_strict(errors, np.mean), history_residual_std_3m=_strict(errors, lambda x: np.std(x, ddof=0)),
                history_residual_slope_3m=_strict(errors, lambda x: (x[-1]-x[0])/2),
                history_relative_residual=error/forecast[origin] if np.isfinite(error) and forecast[origin] > 0 else np.nan)
            for name, observed in (("macro", macro), ("news", news_count)):
                stem = "external_"+name+"_"
                values[stem+("pressure" if name == "macro" else "intensity")] = observed[origin]
                values[stem+"change_1m"] = observed[origin]-observed[origin-1]
                values[stem+"mean_3m"] = _strict(observed[origin-2:origin+1], np.mean)
                values[stem+"std_3m"] = _strict(observed[origin-2:origin+1], lambda x: np.std(x, ddof=0))
            records.append(dict(series_id=uid, month_index=origin, cohort=source.cohort.iloc[0],
                max_dependency_month_index=origin, **values, **{name+"_missing": bool(pd.isna(value)) for name, value in values.items()}))
    columns = KEYS+("max_dependency_month_index",)+VALUES+tuple(name+"_missing" for name in VALUES)
    result = pd.DataFrame(records, columns=columns)
    if len(result) and np.isinf(result[list(VALUES)].to_numpy(dtype=float)).any():
        raise ValueError("Feature calculations exceeded finite range")
    return result


def build_cases(observations: pd.DataFrame, events: pd.DataFrame, config: Mapping) -> pd.DataFrame:
    """True future-onset labels; active rows retained solely for excluded audit.

    Label knowledge waits for the complete finite future y-window O+1..O+k,
    not E07a's weak-event confirmation. Eligibility uses finite past only.
    """
    _, _, labels = _settings(config)
    data = _observations(observations, config)
    required = {"series_id", "cohort", "event_id", "onset_index"}
    if not required.issubset(events):
        raise ValueError("Missing true-event registry fields")
    registry = events.copy()
    if registry.series_id.isna().any() or registry.event_id.isna().any() or registry.event_id.astype(str).str.strip().eq("").any():
        raise ValueError("True-event identity must be present")
    registry["series_id"] = registry.series_id.astype(str)
    registry["event_id"] = registry.event_id.astype(str)
    if registry.event_id.duplicated().any() or registry.series_id.duplicated().any():
        raise ValueError("A series has at most one uniquely identified persistent event")
    identities = data.groupby("series_id").cohort.first().to_dict()
    for event in registry.itertuples(index=False):
        onset = _integer(event.onset_index, "onset_index")
        if event.series_id not in identities or event.cohort != identities[event.series_id] or onset >= config["generator"]["months"]:
            raise ValueError("True event does not belong to its series/cohort/calendar")
    by_id = {str(row.series_id): row for row in registry.itertuples(index=False)}
    rows = []
    for uid, source in data.groupby("series_id", sort=True):
        maximum = int(source.month_index.max())
        y = source.set_index("month_index").y.reindex(range(maximum+1)).to_numpy(dtype=float)
        event = by_id.get(uid)
        if event is not None:
            onset = _integer(event.onset_index, "onset_index")
        else:
            onset = None
        for origin in range(labels["min_history"]-1, maximum+1):
            finite = np.flatnonzero(np.isfinite(y[:origin+1]))
            eligible = len(finite) >= labels["min_history"] and len(finite) > 0 and origin-finite[-1] <= 1
            active = onset is not None and onset <= origin
            for k in labels["horizons"]:
                end = origin+k
                known = end <= maximum and np.isfinite(y[origin+1:end+1]).all()
                label = float(onset is not None and origin < onset <= end) if known else np.nan
                rows.append(dict(series_id=uid, month_index=origin, cohort=source.cohort.iloc[0], k=k,
                    label=label, fully_known=bool(known), active_regime=bool(active), eligible=bool(eligible),
                    at_risk=bool(eligible and not active), label_known_month_index=end if known else np.nan,
                    positive_event_id=str(event.event_id) if known and label == 1 else "",
                    right_censored=end > maximum, synthetic_truth=True))
    columns = KEYS+("k", "label", "fully_known", "active_regime", "eligible", "at_risk", "label_known_month_index", "positive_event_id", "right_censored", "synthetic_truth")
    return pd.DataFrame(rows, columns=columns)


def feature_dictionary() -> pd.DataFrame:
    """Fixed observed feature definitions; no sampled generator truth included."""
    rows = []
    for group, names in feature_groups().items():
        for name in names:
            parent = name[:-8] if name.endswith("_missing") else name
            definition = parent.replace("_", " ")
            if "3m" in parent:
                definition += "; strict finite O-2,O-1,O inputs; std ddof=0 and OLS slope=(last-first)/2"
            if "residual" in parent:
                definition += "; y[T]-causal SeasonalNaiveYoY h1 prediction issued from y[:T]"
            if parent == "history_ratio_1m":
                definition += "; y[O]/y[O-1], missing unless denominator is finite and positive"
            if parent == "history_yoy_ratio":
                definition += "; y[O]/y[O-12], missing unless denominator is finite and positive"
            if parent == "history_cv_3m":
                definition += "; population std/mean of the strict three-month window; positive mean required"
            if parent == "history_relative_residual":
                definition += "; residual divided by positive h1 prediction"
            if group == "detector":
                definition += "; existing E04 class, first three past finite residuals fixed median/MAD scale; no alarm resets"
            if group == "external":
                definition += "; observed stochastic synthetic channel, not real macro/news or latent event metadata"
            rows.append(dict(feature=name, group=group, definition=("True exactly when "+parent+" is NaN") if name.endswith("_missing") else definition,
                availability_rule="All observations and calibration dependencies month_index <= own O",
                model_candidate=True, source="synthetic_observations_only", missing_rule="No filling before train-only model preprocessing"))
    return pd.DataFrame(rows)
