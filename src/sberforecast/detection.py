"""Диагностические онлайн-тревоги ПОСЛЕ наблюдения ошибки; не предсказание шоков."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .data import period_end


@dataclass
class CUSUM:
    allowance: float = 0.5
    threshold: float = 5.0
    cooldown: int = 2
    positive: float = 0.0
    negative: float = 0.0
    remaining: int = 0

    def update(self, z: float) -> tuple[float, bool]:
        if not np.isfinite(z):
            raise ValueError("CUSUM ожидает конечное наблюдение.")
        if self.remaining:
            self.remaining -= 1
            return 0.0, False
        self.positive = max(0.0, self.positive + z - self.allowance)
        self.negative = min(0.0, self.negative + z + self.allowance)
        score = max(self.positive, -self.negative) / self.threshold
        alarm = score > 1.0
        if alarm:
            self.positive = self.negative = 0.0
            self.remaining = self.cooldown
        return score, alarm


@dataclass
class EWMA:
    alpha: float = 0.3
    threshold: float = 3.0
    cooldown: int = 2
    value: float = 0.0
    n: int = 0
    remaining: int = 0

    def update(self, z: float) -> tuple[float, bool]:
        if not np.isfinite(z):
            raise ValueError("EWMA ожидает конечное наблюдение.")
        if self.remaining:
            self.remaining -= 1
            return 0.0, False
        self.n += 1
        self.value = self.alpha*z + (1-self.alpha)*self.value
        sd = np.sqrt(self.alpha/(2-self.alpha) * (1-(1-self.alpha)**(2*self.n)))
        score = abs(self.value) / (self.threshold * sd)
        alarm = score > 1.0
        if alarm:
            self.value = 0.0; self.n = 0; self.remaining = self.cooldown
        return score, alarm


def robust_center_scale(values: np.ndarray, floor: float) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError("Нет ошибок для калибровки детектора.")
    center = float(np.median(values))
    scale = max(float(1.4826 * np.median(np.abs(values-center))), floor)
    return center, scale


def run_detection(predictions: pd.DataFrame, selection: dict, cfg: dict,
                  output: Path) -> pd.DataFrame:
    name = selection["1"]["model"]
    data = predictions[(predictions.horizon == 1) & predictions.model.eq(name)].copy()
    data = data[np.isfinite(data.y_true) & np.isfinite(data.y_pred)]
    data["relative_error"] = (data.y_true-data.y_pred) / np.maximum(np.abs(data.y_pred), 1.0)
    validation = data[data.split.eq("validation")]
    testing = data[data.split.eq("holdout")]
    params = cfg["detection"]
    global_center, global_scale = robust_center_scale(validation.relative_error.to_numpy(), params["relative_scale_floor"])
    calibrations = {}
    for uid, sub in validation.groupby("municipality_id"):
        if len(sub) >= params["calibration_min_observations"]:
            calibrations[str(uid)] = robust_center_scale(sub.relative_error.to_numpy(), params["relative_scale_floor"])
    records = []
    for uid, sub in testing.groupby("municipality_id", sort=True):
        center, scale = calibrations.get(str(uid), (global_center, global_scale))
        detectors = {}
        previous_period = None
        for row in sub.sort_values("target_period").itertuples(index=False):
            period = pd.Period(row.target_period, freq="M")
            # После разрыва месячной последовательности начинаем состояния заново.
            reset = previous_period is None or period.ordinal-previous_period.ordinal != 1
            if reset:
                detectors = {
                    "CUSUM": CUSUM(params["cusum_allowance"], params["cusum_threshold"], params["cooldown_months"]),
                    "EWMA": EWMA(params["ewma_alpha"], params["ewma_threshold"], params["cooldown_months"]),
                }
            z = (row.relative_error-center)/scale
            for method, detector in detectors.items():
                score, alarm = detector.update(float(z))
                records.append({"municipality_id": str(uid), "target_period": row.target_period,
                                "alert_time_assumed": str(period_end(period+cfg["data"]["release_lag_months"]).date()),
                                "forecast_model": name, "method": method,
                                "relative_error": row.relative_error, "z": z, "score": score,
                                "is_alarm": bool(alarm), "calibration_center": center,
                                "calibration_scale": scale, "calibration": "local" if str(uid) in calibrations else "pooled",
                                "state_reset": reset,
                                "meaning": "diagnostic_after_observation_not_early_warning"})
            previous_period = period
    columns = ["municipality_id", "target_period", "alert_time_assumed", "forecast_model", "method", "relative_error", "z", "score", "is_alarm", "calibration_center", "calibration_scale", "calibration", "state_reset", "meaning"]
    frame = pd.DataFrame(records, columns=columns)
    frame["is_alarm"] = frame["is_alarm"].astype(bool)
    frame.to_csv(output/"detector_scores.csv.gz", index=False)
    frame[frame.is_alarm].to_csv(output/"diagnostic_alarms.csv", index=False)
    info = {"model": name, "n_scored_observations_per_method": int(len(frame)//2),
            "alarms_by_method": {k:int(v) for k,v in frame.groupby("method").is_alarm.sum().items()},
            "verified_shock_labels": False, "precision_recall_delay_measured": False,
            "thresholds_tuned": False, "early_warning_implemented": False,
            "warning": "Это неподтверждённые диагностические тревоги, а не найденные или предсказанные экономические шоки."}
    (output/"detection_status.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return frame
