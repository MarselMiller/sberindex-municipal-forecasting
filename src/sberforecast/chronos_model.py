"""Опциональный Chronos-2: только история, независимые МО, медиана 0.5."""
from __future__ import annotations

from dataclasses import dataclass
import importlib.metadata
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .data import get_prefix

MODEL = "Chronos-2"


def prepare_history(panel: pd.DataFrame, origin: pd.Period, release_lag_months: int,
                    forecast_ids: list[str]) -> pd.DataFrame:
    """Сетка заканчивается cutoff, включая месяцы с неизвестным последним фактом."""
    return get_prefix(panel, pd.Period(origin, freq="M"), release_lag_months)[forecast_ids]


def load_pipeline(root: Path, cfg: dict):
    """Единственная загрузка весов на запуск; обычный импорт не требует Chronos."""
    params = cfg["chronos"]
    cache = (root / params["cache_dir"]).resolve()
    if not cache.is_relative_to(root / "outputs"):
        raise ValueError("Кэш весов должен находиться в игнорируемой папке outputs.")
    os.environ["HF_HOME"] = str(cache)
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    from chronos import Chronos2Pipeline
    import torch
    if importlib.metadata.version("chronos-forecasting") != params["package_version"]:
        raise RuntimeError("Версия Chronos отличается от зафиксированного протокола.")
    if params["device"] != "cpu" or params["dtype"] != "float32":
        raise ValueError("Для первого E03 зафиксированы CPU/float32.")
    torch.set_num_threads(params["cpu_threads"])
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    return Chronos2Pipeline.from_pretrained(
        params["model_id"], revision=params["revision"], device_map="cpu",
        torch_dtype=torch.float32, cache_dir=str(cache / "hub"), token=False,
        use_safetensors=True, low_cpu_mem_usage=True,
    )


@dataclass
class ChronosResult:
    forecasts: pd.DataFrame
    paths: pd.DataFrame
    errors: list[dict]
    diagnostics: dict


class ChronosModel:
    def __init__(self, pipeline, batch_size: int = 1, context_length: int = 8192):
        if batch_size < 1 or context_length < 1:
            raise ValueError("Размер пакета и контекста должны быть положительными.")
        self.pipeline = pipeline
        self.batch_size = batch_size
        self.context_length = context_length

    def predict(self, history: pd.DataFrame, origin: pd.Period, horizons: list[int],
                release_lag_months: int) -> ChronosResult:
        started = time.perf_counter()
        origin = pd.Period(origin, freq="M")
        cutoff = (origin - release_lag_months).to_timestamp()
        if release_lag_months < 0 or not horizons or min(horizons) < 1:
            raise ValueError("Недопустимый лаг или горизонт.")
        if history.empty or history.index[-1] != cutoff or (history.index > cutoff).any():
            raise ValueError("История должна заканчиваться на cutoff, без будущих данных.")
        grid = pd.date_range(history.index[0], cutoff, freq="MS")
        if not history.index.equals(grid):
            raise ValueError("История должна сохранять календарную месячную сетку.")
        steps = release_lag_months + max(horizons)
        dates = pd.date_range(cutoff + pd.offsets.MonthBegin(1), periods=steps, freq="MS")
        ids = list(history.columns)
        values = np.full((steps, len(ids)), np.nan)
        reasons = {}
        errors = []
        for start in range(0, len(ids), self.batch_size):
            positions = list(range(start, min(start + self.batch_size, len(ids))))
            good, inputs = [], []
            for j in positions:
                vector = history.iloc[:, j].to_numpy(dtype=np.float32, copy=True)
                if np.isinf(vector).any() or not np.isfinite(vector).any():
                    reason = "nonfinite_or_empty_history"
                    reasons[j] = reason
                    errors.append({"municipality_id": str(ids[j]), "stage": "history", "error": reason})
                else:
                    good.append(j)
                    inputs.append(vector)
            if not good:
                continue
            try:
                quantiles, _ = self.pipeline.predict_quantiles(
                    inputs=inputs, prediction_length=steps, quantile_levels=[0.5],
                    batch_size=self.batch_size, context_length=self.context_length,
                    cross_learning=False,
                )
                if len(quantiles) != len(good):
                    raise ValueError("Неверное число прогнозных рядов Chronos.")
                for j, output in zip(good, quantiles):
                    if hasattr(output, "detach"):
                        output = output.detach().cpu().numpy()
                    array = np.asarray(output)
                    if array.shape != (1, steps, 1):
                        raise ValueError(f"Неверная форма квантилей: {array.shape}.")
                    values[:, j] = array[0, :, 0]
            except Exception as exc:
                for j in good:
                    values[:, j] = np.nan
                    reasons[j] = repr(exc)
                    errors.append({"municipality_id": str(ids[j]), "stage": "predict", "error": repr(exc)})
        frames = []
        for h in horizons:
            index = release_lag_months + h - 1
            forecast = values[index]
            finite = np.isfinite(forecast)
            reason = ["" if finite[j] else reasons.get(j, "nonfinite_prediction") for j in range(len(ids))]
            for j in np.flatnonzero(~finite):
                if j not in reasons:
                    errors.append({"municipality_id": str(ids[j]), "horizon": int(h),
                                   "stage": "predict", "error": "nonfinite_prediction"})
            frames.append(pd.DataFrame({
                "municipality_id": ids, "target_period": str((origin + h).to_timestamp().date()),
                "horizon": h, "y_pred": np.where(finite, forecast, np.nan),
                "status": np.where(finite, "native", "failed"),
                "effective_model": MODEL, "reason": reason,
            }))
        paths = pd.concat([pd.DataFrame({"municipality_id": uid,
                                        "target_period": dates.strftime("%Y-%m-%d"),
                                        "y_pred": values[:, j]}) for j, uid in enumerate(ids)], ignore_index=True)
        return ChronosResult(pd.concat(frames, ignore_index=True), paths, errors, {
            "seconds": time.perf_counter() - started, "n_series": len(ids),
            "n_calendar_history_months": len(history), "n_missing_history_values": int(history.isna().sum().sum()),
            "n_history_observations": int(np.isfinite(history.to_numpy()).sum()),
            "prediction_length": steps, "history_cutoff": str(cutoff.date()),
            "path_first_month": str(dates[0].date()), "path_last_month": str(dates[-1].date()),
            "quantile": 0.5, "cross_learning": False,
        })
