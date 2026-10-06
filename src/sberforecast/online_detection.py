"""Independent E04a detectors on a causally calibrated monthly residual stream.

These alarms describe a change after its observation. They do not predict shocks.
The legacy :mod:`sberforecast.detection` implementation is deliberately untouched.

For each series, the first ``warmup_observations`` finite RUB errors calibrate a
fixed median and MAD scale. Their forecasts also set a relative scale floor.
Only later errors are monitored. Missing months never become zero errors, never
advance detector observation counts, and never reset a state. Publication and
cooldown dates follow calendar months, not the number of available rows.

CUSUM and EWMA follow the NIST handbook, sections 6.3.2.3 and 6.3.2.4. BOCPD
uses Algorithm 1 of Adams and MacKay (2007), arXiv:0710.3742, with a normal /
normal-inverse-gamma observation model and its Student-t predictive density.
For constant hazard H that algorithm gives P(r=0|data)=H, so this value is a
diagnostic, NOT our score. Our score is P(1 <= r <= recent_run_max|data), after
more than recent_run_max observations since initialization or an alarm reset.
Run length is measured in observed residuals; it does not advance over gaps.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import gammaln, logsumexp


@dataclass
class DetectorUpdate:
    score: float
    crossed_threshold: bool
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _positive(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _integer(value: int, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _finite(value: float) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError("Detector input must be finite")
    return value


class CUSUM:
    """Two-sided tabular CUSUM in standardized residual units.

    C+_t=max(0,C+_(t-1)+z_t-k); C-_t=max(0,C-_(t-1)-z_t-k).
    An alarm candidate crosses max(C+,C-)>h. The runner owns alarm resets.
    """

    def __init__(self, allowance: float = 0.5, threshold: float = 5.0):
        self.allowance = float(allowance)
        if not np.isfinite(self.allowance) or self.allowance < 0:
            raise ValueError("allowance must be finite and nonnegative")
        self.threshold = _positive(threshold, "threshold")
        self.reset()

    def reset(self) -> None:
        self.positive = self.negative = 0.0
        self.n = 0

    def update(self, z: float) -> DetectorUpdate:
        z = _finite(z)
        self.n += 1
        self.positive = max(0.0, self.positive + z - self.allowance)
        self.negative = max(0.0, self.negative - z - self.allowance)
        score = max(self.positive, self.negative)
        return DetectorUpdate(score, score > self.threshold, {
            "cusum_positive": self.positive,
            "cusum_negative": self.negative,
            "n_updates_since_reset": self.n,
        })


class EWMA:
    """Two-sided EWMA with finite-start variance and a zero initial mean.

    w_t=alpha*z_t+(1-alpha)*w_(t-1). The score is |w_t|/sd_t, where
    sd_t^2=alpha/(2-alpha)*(1-(1-alpha)^(2*n)), n=observed updates.
    """

    def __init__(self, alpha: float = 0.3, threshold: float = 3.0):
        self.alpha = _positive(alpha, "alpha")
        if self.alpha > 1:
            raise ValueError("alpha must be <= 1")
        self.threshold = _positive(threshold, "threshold")
        self.reset()

    def reset(self) -> None:
        self.value = 0.0
        self.n = 0

    def update(self, z: float) -> DetectorUpdate:
        z = _finite(z)
        self.n += 1
        self.value = self.alpha * z + (1 - self.alpha) * self.value
        if self.alpha == 1:
            variance = 1.0
        else:
            variance = self.alpha / (2 - self.alpha) * (
                -np.expm1(2 * self.n * np.log1p(-self.alpha))
            )
        score = float(abs(self.value) / np.sqrt(variance))
        return DetectorUpdate(score, score > self.threshold, {
            "ewma_value": self.value,
            "ewma_sd": float(np.sqrt(variance)),
            "n_updates_since_reset": self.n,
        })


class BOCPD:
    """Exact log-space BOCPD for an unknown Gaussian mean and variance.

    sigma^2 ~ InverseGamma(alpha,beta), mu|sigma^2 ~ N(m,sigma^2/kappa).
    Predictive: StudentT(df=2*alpha,loc=m,
                         scale=sqrt(beta*(kappa+1)/(alpha*kappa))).
    Hazard is constant per observed residual, with no run-length truncation.
    """

    def __init__(self, hazard: float = 1 / 24, threshold: float = 0.5,
                 recent_run_max: int = 2, prior_mean: float = 0.0,
                 prior_kappa: float = 1.0, prior_alpha: float = 2.0,
                 prior_beta: float = 1.0):
        self.hazard = _positive(hazard, "hazard")
        if self.hazard >= 1:
            raise ValueError("hazard must be < 1")
        self.threshold = _positive(threshold, "threshold")
        if self.threshold > 1:
            raise ValueError("BOCPD threshold must be <= 1")
        self.recent_run_max = _integer(recent_run_max, "recent_run_max", 1)
        self.prior_mean = _finite(prior_mean)
        self.prior_kappa = _positive(prior_kappa, "prior_kappa")
        self.prior_alpha = _positive(prior_alpha, "prior_alpha")
        self.prior_beta = _positive(prior_beta, "prior_beta")
        self.reset()

    def reset(self) -> None:
        self.log_posterior = np.array([0.0])
        self.mean = np.array([self.prior_mean])
        self.kappa = np.array([self.prior_kappa])
        self.alpha = np.array([self.prior_alpha])
        self.beta = np.array([self.prior_beta])
        self.n = 0

    @property
    def posterior(self) -> np.ndarray:
        return np.exp(self.log_posterior)

    def update(self, z: float) -> DetectorUpdate:
        z = _finite(z)
        df = 2 * self.alpha
        log_scale_squared = np.log(self.beta) + np.log1p(1 / self.kappa) - np.log(self.alpha)
        difference = z - self.mean
        with np.errstate(divide="ignore"):
            log_squared_distance = 2 * np.log(np.abs(difference))
        log_predictive = (
            gammaln((df + 1) / 2) - gammaln(df / 2)
            - 0.5 * (np.log(df) + np.log(np.pi) + log_scale_squared)
            - (df + 1) / 2 * np.logaddexp(
                0.0, log_squared_distance - np.log(df) - log_scale_squared
            )
        )
        weighted = self.log_posterior + log_predictive
        reset_mass = logsumexp(weighted) + np.log(self.hazard)
        growth_mass = weighted + np.log1p(-self.hazard)
        new_log_posterior = np.concatenate(([reset_mass], growth_mass))
        new_log_posterior -= logsumexp(new_log_posterior)
        new_kappa = self.kappa + 1
        new_mean = self.mean + difference / new_kappa
        new_beta = self.beta + (self.kappa / new_kappa) * (difference * difference / 2)
        if not np.isfinite(new_beta).all() or not np.isfinite(new_mean).all():
            raise FloatingPointError("BOCPD sufficient statistics exceeded numeric range")
        self.log_posterior = new_log_posterior
        self.mean = np.concatenate(([self.prior_mean], new_mean))
        self.kappa = np.concatenate(([self.prior_kappa], new_kappa))
        self.alpha = np.concatenate(([self.prior_alpha], self.alpha + 0.5))
        self.beta = np.concatenate(([self.prior_beta], new_beta))
        self.n += 1
        posterior = self.posterior
        recent_probability = float(posterior[1:self.recent_run_max + 1].sum())
        armed = self.n > self.recent_run_max
        score = recent_probability if armed else 0.0
        return DetectorUpdate(score, armed and score > self.threshold, {
            "posterior_sum": float(posterior.sum()),
            "posterior_cp_probability": float(posterior[0]),
            "posterior_recent_probability": recent_probability,
            "posterior_map_run_length": int(np.argmax(posterior)),
            "bocpd_armed": armed,
            "n_updates_since_reset": self.n,
        })


def _detector(method: str, params: dict) -> CUSUM | EWMA | BOCPD:
    constructors = {"CUSUM": CUSUM, "EWMA": EWMA, "BOCPD": BOCPD}
    if method not in constructors:
        raise ValueError(f"Unknown online detector: {method}")
    return constructors[method](**params)


def _date(period: pd.Period) -> str:
    return str(period.end_time.normalize().date())


def _as_of_timestamp(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    if isinstance(value, pd.Period) or (isinstance(value, str) and len(value) == 7):
        return pd.Period(value, freq="M").end_time.normalize()
    return pd.Timestamp(value).normalize()


OUTPUT_COLUMNS = [
    "series_id", "observation_period", "availability_date", "signal_date",
    "method", "y_true", "y_pred", "error", "phase", "center", "scale", "z",
    "score", "threshold", "is_alarm", "crossed_threshold", "cooldown_suppressed",
    "calibration_end_period", "calibration_count", "n_updates_since_reset",
    "state_reset", "cusum_positive", "cusum_negative", "ewma_value", "ewma_sd",
    "posterior_sum", "posterior_cp_probability", "posterior_recent_probability",
    "posterior_map_run_length", "bocpd_armed",
]


def run_stream(residuals: pd.DataFrame, method: str, params: dict,
               preparation: dict) -> pd.DataFrame:
    """Run one detector independently per series on a complete calendar grid.

    Input columns are ``series_id, observation_period, y_true, y_pred``.
    Output preserves explicit or inserted missing calendar months. An optional
    ``as_of`` prevents unavailable facts from calibrating or updating the state.
    Fixed calibration is first used on the observation AFTER warmup completes.
    Alarm comparison is strictly ``score > threshold`` for all three methods.
    After an emitted alarm, reset the detector and suppress emissions in the
    next ``cooldown_months`` calendar months; still update on observed errors.
    """
    required = {"series_id", "observation_period", "y_true", "y_pred"}
    missing = required - set(residuals.columns)
    if missing:
        raise ValueError(f"Missing residual columns: {sorted(missing)}")
    warmup = _integer(preparation.get("warmup_observations", 4), "warmup_observations", 1)
    cooldown = _integer(preparation.get("cooldown_months", 2), "cooldown_months")
    lag = _integer(preparation.get("release_lag_months", 0), "release_lag_months")
    relative_floor = float(preparation.get("relative_scale_floor", 0.03))
    if not np.isfinite(relative_floor) or relative_floor < 0:
        raise ValueError("relative_scale_floor must be finite and nonnegative")
    absolute_floor = _positive(preparation.get("absolute_scale_floor", 1.0), "absolute_scale_floor")
    as_of = _as_of_timestamp(preparation.get("as_of"))
    template = _detector(method, params)  # Validate even when the input is empty.
    if residuals.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    data = residuals[list(required)].copy()
    if data["series_id"].isna().any():
        raise ValueError("series_id must not be missing")
    data["series_id"] = data["series_id"].astype(str)
    data["period"] = pd.PeriodIndex(data["observation_period"], freq="M")
    if data["period"].isna().any():
        raise ValueError("observation_period must not be missing")
    if data.duplicated(["series_id", "period"]).any():
        raise ValueError("Duplicate series_id / observation_period residual keys")
    records = []
    for series_id, sub in data.groupby("series_id", sort=True):
        sub = sub.set_index("period").sort_index()
        calendar = pd.period_range(sub.index.min(), sub.index.max(), freq="M")
        sub = sub.reindex(calendar)
        detector = _detector(method, params)
        errors: list[float] = []
        levels: list[float] = []
        center = scale = np.nan
        calibration_end = ""
        last_alarm_ordinal: int | None = None
        for period, row in sub.iterrows():
            availability_period = period + lag
            availability_date = _date(availability_period)
            y_true = float(row["y_true"])
            y_pred = float(row["y_pred"])
            record = {column: np.nan for column in OUTPUT_COLUMNS}
            record.update({
                "series_id": series_id, "observation_period": str(period),
                "availability_date": availability_date, "signal_date": "",
                "method": method, "y_true": y_true, "y_pred": y_pred,
                "phase": "missing", "center": center, "scale": scale,
                "threshold": template.threshold, "is_alarm": False,
                "crossed_threshold": False, "cooldown_suppressed": False,
                "calibration_end_period": calibration_end,
                "calibration_count": len(errors), "state_reset": False,
            })
            if as_of is not None and availability_period.end_time.normalize() > as_of:
                record["phase"] = "unavailable"
            elif np.isfinite(y_true) and np.isfinite(y_pred):
                with np.errstate(over="ignore"):
                    error = y_true - y_pred
                if np.isfinite(error):
                    record["error"] = float(error)
                    if len(errors) < warmup:
                        record["phase"] = "warmup"
                        errors.append(float(error))
                        levels.append(abs(y_pred))
                        record["calibration_count"] = len(errors)
                        if len(errors) == warmup:
                            center = float(np.median(errors))
                            mad = float(1.4826 * np.median(np.abs(np.asarray(errors) - center)))
                            scale = max(mad, relative_floor * float(np.median(levels)), absolute_floor)
                            calibration_end = str(period)
                    else:
                        record["phase"] = "monitoring"
                        record["z"] = float((error - center) / scale)
                        update = detector.update(record["z"])
                        suppressed = (last_alarm_ordinal is not None and
                                      availability_period.ordinal - last_alarm_ordinal <= cooldown)
                        alarm = update.crossed_threshold and not suppressed
                        record.update(update.diagnostics)
                        record.update({
                            "score": update.score, "crossed_threshold": update.crossed_threshold,
                            "cooldown_suppressed": bool(suppressed), "is_alarm": bool(alarm),
                        })
                        if alarm:
                            record["signal_date"] = availability_date
                            record["state_reset"] = True
                            last_alarm_ordinal = availability_period.ordinal
                            detector.reset()
            records.append(record)
    result = pd.DataFrame(records, columns=OUTPUT_COLUMNS)
    for column in ["is_alarm", "crossed_threshold", "cooldown_suppressed", "state_reset"]:
        result[column] = result[column].astype(bool)
    return result
