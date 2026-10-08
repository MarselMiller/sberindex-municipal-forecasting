"""Parameter-free persistence of the last causally available local ratio.

The supplied national table and its SeasonalNaiveYoY forecast use the existing
National/Local implementation. No local learner, training-pair admission,
smoothing, or ratio clipping is involved. Annual predictions follow the same
persistence formula; the learned direct strategies' empty-training fallback is
not applicable to this baseline.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import get_prefix, period_end
from .direct_training import _nonnegative_integer, _validate_panel
from .national_local import (
    _lag, _validate_national, national_forecast, ratio_panel,
    restore_local_prediction,
)

MODEL_NAME = "NationalLocalPersistence"


@dataclass
class PersistenceResult:
    forecasts: pd.DataFrame
    national_forecast: dict


class NationalLocalPersistence:
    """Keep the last finite y/N ratio on the common causal history.

    Callers provide the national table built from the complete panel and the
    same forecast IDs as the reference strategies. Eligibility and evaluation
    splits belong to the existing backtest, not to this prediction rule.
    """

    def __init__(self, national: pd.DataFrame, release_lag_months: int = 0):
        _validate_national(national)
        self.lag = _lag(release_lag_months)
        self.national = national.copy(deep=True)

    def predict(
        self, panel: pd.DataFrame, origin: str | pd.Period, horizon: int,
        forecast_ids: list[str], baseline_config: dict,
    ) -> PersistenceResult:
        """Return all requested keys, including explicitly failed batches.

        The ratio is selected from the last month where both expense and its
        finite positive national denominator are available by O. Missing latest
        months can use an earlier finite ratio, without a new staleness limit.
        As in the learned National/Local path, any missing requested anchor or
        invalid national forecast fails the complete origin/horizon batch.
        """
        _validate_panel(panel)
        _nonnegative_integer(horizon, "horizon", minimum=1)
        horizon = int(horizon)
        origin = pd.Period(origin, freq="M")
        ids = list(forecast_ids)
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("Forecast IDs must be nonempty and unique.")
        if any(municipality not in panel.columns for municipality in ids):
            raise ValueError("Forecast ID is absent from the supplied expense panel.")

        prefix = get_prefix(panel, origin, self.lag)
        values = prefix.to_numpy(dtype=float)
        if (np.isfinite(values) & (values < 0)).any():
            raise ValueError("Finite observed expenses must be nonnegative.")
        ratios, _ = ratio_panel(prefix, self.national)
        selected = self.national.reindex(prefix.index)
        denominator_dates = pd.to_datetime(selected["available_at"])
        expense_dates = pd.DatetimeIndex([
            period_end(month + self.lag) for month in prefix.index.to_period("M")
        ])
        origin_date = period_end(origin)
        joint_available = (expense_dates <= origin_date) & (
            denominator_dates <= origin_date
        ).to_numpy()
        national = national_forecast(
            self.national, origin, horizon, baseline_config,
            release_lag_months=self.lag,
        )
        n_hat = national.get("N_hat", np.nan)

        forecasts = pd.DataFrame({
            "municipality_id": ids,
            "forecast_origin": str(origin_date.date()),
            "target_period": (origin + horizon).to_timestamp(),
            "horizon": horizon,
            "feature_cutoff": period_end(origin - self.lag),
            "release_lag_months": self.lag,
            "y_pred": np.nan, "status": "failed", "effective_model": "none",
            "reason": "", "failure_stage": "", "fallback_baseline_status": "",
            "anchor": np.nan, "ratio_hat": np.nan, "national_hat": n_hat,
            "national_forecast_status": national.get("status", "failed"),
            "last_ratio_period": pd.NaT, "ratio_available_at": pd.NaT,
            "ratio_national_available_at": pd.NaT,
            "ratio_expense_value": np.nan, "ratio_national_denominator": np.nan,
            "n_ratio_observations": 0, "ratio_staleness_months": np.nan,
            "ratio_status": "missing", "ratio_reason": "no_causal_ratio",
            "fit_called": False,
        })
        for position, municipality in enumerate(ids):
            history = ratios[municipality].to_numpy(dtype=float)
            usable = np.isfinite(history) & joint_available
            observations = np.flatnonzero(usable)
            forecasts.loc[position, "n_ratio_observations"] = len(observations)
            if not len(observations):
                continue
            last = int(observations[-1])
            timestamp = prefix.index[last]
            forecasts.loc[position, "anchor"] = history[last]
            forecasts.loc[position, "last_ratio_period"] = timestamp
            forecasts.loc[position, "ratio_available_at"] = expense_dates[last]
            forecasts.loc[position, "ratio_national_available_at"] = denominator_dates.iloc[last]
            forecasts.loc[position, "ratio_expense_value"] = prefix.iloc[last][municipality]
            forecasts.loc[position, "ratio_national_denominator"] = selected.N.iloc[last]
            forecasts.loc[position, "ratio_staleness_months"] = (
                origin - self.lag - timestamp.to_period("M")
            ).n
            forecasts.loc[position, "ratio_status"] = "available"
            forecasts.loc[position, "ratio_reason"] = ""

        if (national.get("status") not in {"native", "fallback_last_value"}
                or not np.isfinite(n_hat) or n_hat <= 0):
            forecasts["failure_stage"] = "national_forecast"
            forecasts["reason"] = national.get("reason") or "nonpositive_or_nonfinite_national_forecast"
            return PersistenceResult(forecasts, national)
        if not np.isfinite(forecasts.anchor.to_numpy(dtype=float)).all():
            forecasts["failure_stage"] = "local_ratio"
            forecasts["reason"] = np.where(
                forecasts.ratio_status.eq("missing"), "no_causal_ratio",
                "forecast_batch_contains_missing_causal_ratio",
            )
            return PersistenceResult(forecasts, national)

        anchors = forecasts.anchor.to_numpy(dtype=float)
        try:
            # Valid expense observations make these ratios nonnegative, so the
            # existing nominal restoration applies the exact N_hat * R_last.
            restored = restore_local_prediction(n_hat, anchors)
        except ValueError as exc:
            forecasts["failure_stage"] = "restore"
            forecasts["reason"] = repr(exc)
            return PersistenceResult(forecasts, national)
        forecasts["ratio_hat"] = anchors
        forecasts["y_pred"] = restored
        forecasts["status"] = "native"
        forecasts["effective_model"] = MODEL_NAME
        return PersistenceResult(forecasts, national)
